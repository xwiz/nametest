"""
srm/pipeline.py — end-to-end SRM query pipeline.

Pipeline:
    1. expand_query    bridge vocabulary gap with domain synonyms
    2. encode          project query into Hamming space via SimHash
    3. traverse        fire NUM_CASTS noisy probes; vote on attractors
    4. hybrid re-rank  0.55 × vote_share + 0.45 × TF-IDF cosine
    5. filter          require MIN_COS or VOTE_FLOOR to admit attractor
    6. synthesise      RWEA: score sentences, MMR select, add connectives

Returns a rich result dict suitable for display, testing, or JSON export.
"""

from __future__ import annotations

import random
import re
import numpy as np

from .config import (
    TOP_K, NUM_CASTS, NOISE,
    MIN_COS, VOTE_FLOOR,
    W_VOTE, W_COS,
    SIM_THRESH, MAX_WORDS,
    STOPWORDS,
)
from .nlp import tokenise, tfidf_vec, cosine, expand_query
from .encoding import encode, hamming_batch
from .traversal import traverse
from .synthesis import synthesise
from .store import MemoryStore
from .meaning import _EXPANSION_SENTINEL


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _encode_text_with_sentinel(
    query_text: str,
    expanded: str,
    meaning_db,
) -> str:
    """Build the text string passed to encode(), inserting the expansion
    sentinel so that SVO detection resets between original query and
    expansion terms.  Only inserts the sentinel when meaning_db is active
    (otherwise the sentinel has no effect on standard SimHash).
    """
    if meaning_db is not None and expanded != query_text:
        extras = expanded[len(query_text):].strip()
        if extras:
            return f"{query_text} {_EXPANSION_SENTINEL} {extras}"
    return expanded


def _traverse_narrowed(
    q_code: np.ndarray,
    codes: np.ndarray,
    *,
    num_casts: int,
    noise: float,
    rng,
    candidate_limit: int,
) -> tuple[dict[int, int], list[dict], np.ndarray, int]:
    """Run traversal against only the nearest Hamming candidates.

    This preserves the existing traversal semantics while avoiding full-store
    nearest-neighbour search for every cast.
    """
    initial_dists = hamming_batch(q_code, codes)
    if len(initial_dists) <= candidate_limit:
        votes, cast_log = traverse(
            q_code,
            codes,
            num_casts=num_casts,
            noise=noise,
            rng=rng,
        )
        return votes, cast_log, initial_dists, len(initial_dists)

    candidate_idx = np.argsort(initial_dists)[:candidate_limit]
    candidate_codes = codes[candidate_idx]
    sub_votes, sub_cast_log = traverse(
        q_code,
        candidate_codes,
        num_casts=num_casts,
        noise=noise,
        rng=rng,
    )

    votes: dict[int, int] = {}
    cast_log: list[dict] = []
    for sub_i, count in sub_votes.items():
        votes[int(candidate_idx[sub_i])] = int(count)
    for item in sub_cast_log:
        orig_i = int(candidate_idx[int(item["landed_on"])])
        cast_log.append({
            "cast_id": item["cast_id"],
            "landed_on": orig_i,
            "hamming_dist": item["hamming_dist"],
        })
    return votes, cast_log, initial_dists, len(candidate_idx)


def _response_quality(
    query_text: str,
    response_text: str,
    *,
    idf: dict[str, float],
    source_texts: list[str],
    q_vec: dict[str, float] | None = None,
) -> float:
    resp = (response_text or "").strip()
    if not resp:
        return -1e9

    q_toks = [t for t in tokenise(query_text) if t not in STOPWORDS and len(t) > 2]
    r_toks = tokenise(resp)
    if not r_toks:
        return -1e9

    # 1) Query token coverage (does the response mention the query's key terms?)
    q_set = set(q_toks)
    r_set = set(t for t in r_toks if t not in STOPWORDS)
    if q_set:
        coverage = len(q_set & r_set) / len(q_set)
    else:
        coverage = 0.0

    # 2) Overall relevance (cosine between query and response in TF-IDF space)
    if q_vec is None:
        q_vec = tfidf_vec(tokenise(expand_query(query_text)), idf)
    r_vec = tfidf_vec(r_toks, idf)
    rel = cosine(q_vec, r_vec)

    signal = max(coverage, rel)

    # 3) How many distinct source memories appear to contribute to the response?
    # Count a source as "used" if it overlaps ≥2 content tokens with the response.
    used = 0
    for s in source_texts:
        st = [t for t in tokenise(s) if t not in STOPWORDS and len(t) > 2]
        if len(set(st) & r_set) >= 2:
            used += 1

    # 3b) Prefer a small number of high-signal sources (conversational answers)
    # over stitching many loosely-related snippets.
    used_over = max(0, used - 3)

    # 4) Penalise very short fragments (many 1-3 word "sentences" indicates
    # a fragment KB was joined too literally, or synthesis failed.
    short = 0
    segments = [seg.strip() for seg in _SENTENCE_SPLIT_RE.split(resp) if seg.strip()]
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg.split()) < 4:
            short += 1

    # 4b) Reward discourse connectives (synthesis adds these) for flow.
    connective_words = {
        "furthermore", "additionally", "however", "notably",
        "therefore", "altogether", "summary",
    }
    conn = 0
    for seg in segments[1:]:
        first = tokenise(seg[:24])
        if first and first[0] in connective_words:
            conn += 1

    # 4c) Penalise redundancy across sentences.
    redundant = 0
    if len(segments) >= 2:
        seg_vecs = [tfidf_vec(tokenise(s), idf) for s in segments]
        for i in range(len(seg_vecs)):
            for j in range(i + 1, len(seg_vecs)):
                if cosine(seg_vecs[i], seg_vecs[j]) > 0.78:
                    redundant += 1

    # 5) Soft length penalty (discourage dumping lots of text)
    length = len(r_toks)

    # If the response doesn't match the query at all, heavily penalise “busy”
    # answers (many sources / lots of text). This prevents auto-mode from
    # preferring stitched noise over a single coherent sentence.
    low_signal_penalty = 0.0
    if signal < 0.05:
        low_signal_penalty = 0.8 + 0.25 * used + 0.0006 * length

    return (
        2.2 * coverage
        + 1.6 * rel
        + (0.12 * min(used, 4) * (0.25 + 0.75 * signal))
        - (0.20 * used_over * (0.4 + 0.6 * (1.0 - signal)))
        - 0.12 * short
        + (0.16 * min(conn, 2) * (0.2 + 0.8 * signal))
        - (0.18 * min(redundant, 4))
        - 0.0008 * max(0, length - 120)
        - low_signal_penalty
    )


def srm_query(
    query_text: str,
    store: MemoryStore,
    top_k:     int   = TOP_K,
    num_casts: int   = NUM_CASTS,
    noise:     float = NOISE,
    min_cos:   float = MIN_COS,
    vote_floor: int  = VOTE_FLOOR,
    w_vote:    float = W_VOTE,
    w_cos:     float = W_COS,
    sim_thresh: float = SIM_THRESH,
    max_words: int    = MAX_WORDS,
    meaning_db = None,     # srm.meaning.MeaningDB | None
    seed: int | None = None,
) -> dict:
    """
    Run a full SRM query against *store*.

    Args:
        query_text:  natural-language question or topic
        store:       populated MemoryStore
        top_k:       maximum candidate attractors to retrieve
        num_casts:   stochastic probes per query
        noise:       per-bit flip probability for each probe
        meaning_db:  optional MeaningDB for verb-polarity-aware encoding

    When meaning_db is provided, verb polarity masks and adjective weight
    multipliers are applied during both memory encoding and query encoding,
    so semantically opposite predicates (kill vs protect) map to different
    Hamming regions and the stochastic traversal can distinguish them.

    Returns a dict with keys:
        query              original query string
        expanded_query     query after vocabulary expansion (or None)
        response           synthesised paragraph answer
        top_attractors     list of (mem_idx, votes, text)
        attractor_details  list of per-attractor score dicts
        cast_log           per-cast debug log
        vote_distribution  {mem_idx: votes} for all voted memories
        num_memories       size of the memory store
        num_casts          number of casts used
        noise              noise level used
        meaning_enabled    True if meaning_db was supplied
    """
    ids, texts = store.load_all()
    if not texts:
        return {"error": "Memory store is empty. Use --seed or /add to populate."}

    idf   = store.get_idf()
    codes = store.get_codes(meaning_db=meaning_db)

    # ── 1. Expand ──────────────────────────────────────────────────────
    expanded = expand_query(query_text)

    # ── 2. Encode ──────────────────────────────────────────────────────
    encode_text = _encode_text_with_sentinel(query_text, expanded, meaning_db)
    q_code = encode(encode_text, idf, meaning_db=meaning_db)

    # ── 3. Traverse ────────────────────────────────────────────────────
    rng = np.random.default_rng(seed) if seed is not None else None
    votes, cast_log, dists, traversal_candidates = _traverse_narrowed(
        q_code,
        codes,
        num_casts=num_casts,
        noise=noise,
        rng=rng,
        candidate_limit=max(top_k * 24, 128),
    )

    # ── 4. Candidate-only hybrid re-rank (#1 improvement) ─────────────
    # Instead of computing TF-IDF cosine for *every* memory, collect
    # candidates from two cheap signals: top by vote count and top by
    # Hamming proximity.  Then compute expensive cosine only for those.
    n = len(texts)
    candidate_width = max(top_k * 4, 20)

    top_by_votes = sorted(votes.keys(), key=lambda i: -votes[i])[:candidate_width]
    top_by_hamming = list(np.argsort(dists)[:candidate_width])
    candidate_set = sorted(set(top_by_votes) | set(int(i) for i in top_by_hamming))

    q_vec = tfidf_vec(tokenise(expanded), idf)
    cand_vecs: dict[int, dict[str, float]] = {}
    for i in candidate_set:
        cand_vecs[i] = tfidf_vec(tokenise(texts[i]), idf)

    scores: dict[int, float] = {}
    for i in candidate_set:
        scores[i] = (
            w_vote * (votes.get(i, 0) / num_casts)
            + w_cos * cosine(q_vec, cand_vecs[i])
        )
    cand_cos = {i: cosine(q_vec, cand_vecs[i]) for i in candidate_set}

    ranked = sorted(candidate_set, key=lambda i: -scores[i])[:top_k]

    # ── 5. Combined filter (#3 improvement) ───────────────────────────
    # Require minimal relevance (cos > tiny floor) AND either votes or
    # cosine above their respective thresholds.  This prevents pure-noise
    # entries from sneaking in on votes alone.
    _COS_FLOOR = 0.01
    ranked = [
        i for i in ranked
        if cand_cos[i] > _COS_FLOOR
        and (cand_cos[i] > min_cos or votes.get(i, 0) > vote_floor)
    ]

    top_attractors = [(i, votes.get(i, 0), texts[i]) for i in ranked]

    attractor_details = [
        {
            "mem_id":       ids[i],
            "mem_idx":      i,
            "votes":        votes.get(i, 0),
            "text":         texts[i],
            "hamming_dist": int(dists[i]),
            "similarity":   round(1.0 - dists[i] / 128, 3),
            "cosine":       round(cand_cos.get(i, 0.0), 3),
            "hybrid_score": round(scores.get(i, 0.0), 3),
        }
        for i in ranked
    ]

    return {
        "query":             query_text,
        "expanded_query":    expanded if expanded != query_text else None,
        "response":          (
            synthesise(
                query_text,
                top_attractors,
                num_casts=num_casts,
                max_words=max_words,
                sim_thresh=sim_thresh,
                rng=(random.Random(seed) if seed is not None else None),
            )
            if top_attractors
            else "No relevant memories found."
        ),
        "top_attractors":    top_attractors,
        "attractor_details": attractor_details,
        "cast_log":          cast_log,
        "vote_distribution": {i: int(v) for i, v in votes.items()},
        "num_memories":      n,
        "num_traversal_candidates": traversal_candidates,
        "num_rerank_candidates": len(candidate_set),
        "num_casts":         num_casts,
        "noise":             noise,
        "meaning_enabled":   meaning_db is not None,
    }


def srm_query_auto(
    query_text: str,
    store: MemoryStore,
    top_k: int = TOP_K,
    num_casts: int = NUM_CASTS,
    noise: float = NOISE,
    min_cos: float = MIN_COS,
    vote_floor: int = VOTE_FLOOR,
    w_vote: float = W_VOTE,
    w_cos: float = W_COS,
    sim_thresh: float = SIM_THRESH,
    max_words: int = MAX_WORDS,
    meaning_db=None,
    seed: int | None = None,
) -> dict:
    """Try both response strategies and pick the one that scores better.

    This is the testable, non-assumptive way to handle both:
    - "sentence KB" mode (RWEA synthesis)
    - "fragment KB" mode (cast-level reconstruction)
    """
    # Compute IDF + q_vec once for fair scoring (#4 improvement).
    idf = store.get_idf()
    expanded = expand_query(query_text)
    q_vec = tfidf_vec(tokenise(expanded), idf)

    synth = srm_query(
        query_text,
        store,
        top_k=top_k,
        num_casts=num_casts,
        noise=noise,
        min_cos=min_cos,
        vote_floor=vote_floor,
        w_vote=w_vote,
        w_cos=w_cos,
        sim_thresh=sim_thresh,
        max_words=max_words,
        meaning_db=meaning_db,
        seed=seed,
    )

    recon = srm_query_cast_reconstruct(
        query_text,
        store,
        num_casts=num_casts,
        noise=noise,
        sim_thresh=sim_thresh,
        max_words=max_words,
        meaning_db=meaning_db,
        seed=seed,
    )

    s_sources = [t for _, _, t in (synth.get("top_attractors") or [])]
    r_sources = list(recon.get("selected_outputs") or [])

    s_score = _response_quality(
        query_text,
        synth.get("response", ""),
        idf=idf,
        source_texts=s_sources,
        q_vec=q_vec,
    )
    r_score = _response_quality(
        query_text,
        recon.get("response", ""),
        idf=idf,
        source_texts=r_sources,
        q_vec=q_vec,
    )

    if r_score > s_score:
        chosen = recon
        chosen_mode = "reconstruct"
    else:
        chosen = synth
        chosen_mode = "synth"

    out = dict(chosen)
    out["auto_selected_mode"] = chosen_mode
    out["auto_scores"] = {
        "synth": round(float(s_score), 6),
        "reconstruct": round(float(r_score), 6),
    }
    return out


def srm_query_cast_reconstruct(
    query_text: str,
    store: MemoryStore,
    num_casts: int = 4,
    noise: float   = NOISE,
    sim_thresh: float = SIM_THRESH,
    max_words: int    = MAX_WORDS,
    meaning_db=None,          # srm.meaning.MeaningDB | None
    seed: int | None = None,
) -> dict:
    """Run a small number of casts and reconstruct from unique cast landings.

    This mode is designed for *fragment KBs* where each memory is a small fact.
    You can think of each cast as producing a small "snippet"; we then merge
    the unique snippets into a final response.
    """
    ids, texts = store.load_all()
    if not texts:
        return {"error": "Memory store is empty. Use --seed or /add to populate."}

    idf   = store.get_idf()
    codes = store.get_codes(meaning_db=meaning_db)

    expanded = expand_query(query_text)
    encode_text = _encode_text_with_sentinel(query_text, expanded, meaning_db)

    q_code = encode(encode_text, idf, meaning_db=meaning_db)

    rng = np.random.default_rng(seed) if seed is not None else None
    votes, cast_log, _, traversal_candidates = _traverse_narrowed(
        q_code,
        codes,
        num_casts=num_casts,
        noise=noise,
        rng=rng,
        candidate_limit=max(num_casts * 16, 64),
    )

    cast_outputs: list[str] = []
    cast_indices: list[int] = []
    for c in cast_log:
        idx = int(c["landed_on"])
        cast_indices.append(idx)
        cast_outputs.append(texts[idx])

    seen: set[int] = set()
    unique_indices: list[int] = []
    unique_outputs: list[str] = []
    for idx in cast_indices:
        if idx in seen:
            continue
        seen.add(idx)
        unique_indices.append(idx)
        unique_outputs.append(texts[idx])

    selected_indices = list(unique_indices)
    selected_outputs = list(unique_outputs)

    # If all casts land on the same memory, we still want to be able to
    # reconstruct from multiple KB fragments. Fill with additional highly
    # similar memories (deterministic).
    target = min(len(texts), max(2, min(4, num_casts)))
    if len(selected_indices) < target and len(texts) > len(selected_indices):
        q_vec = tfidf_vec(tokenise(expanded), idf)
        cand = []
        for i, t in enumerate(texts):
            if i in seen:
                continue
            c = cosine(q_vec, tfidf_vec(tokenise(t), idf))
            cand.append((c, i))
        cand.sort(key=lambda x: (-x[0], x[1]))
        for c, i in cand:
            if len(selected_indices) >= target:
                break
            if c <= 0.0:
                break
            selected_indices.append(i)
            selected_outputs.append(texts[i])

    top_attractors = [(i, int(votes.get(i, 0)), texts[i]) for i in selected_indices]

    def _as_sentence(t: str) -> str:
        t = (t or "").strip()
        if not t:
            return ""
        return t if t[-1] in ".!?" else (t + ".")

    parts: list[str] = []
    used = 0
    for frag in selected_outputs:
        sent = _as_sentence(frag)
        if not sent:
            continue
        wc = len(sent.split())
        if parts and used + wc > max_words:
            break
        parts.append(sent)
        used += wc

    response = " ".join(parts) if parts else "No relevant memories found."

    return {
        "query":               query_text,
        "expanded_query":      expanded if expanded != query_text else None,
        "cast_outputs":        cast_outputs,
        "unique_cast_outputs": unique_outputs,
        "unique_cast_indices": unique_indices,
        "selected_outputs":    selected_outputs,
        "selected_indices":    selected_indices,
        "response":            response,
        "vote_distribution":   {i: int(v) for i, v in votes.items()},
        "num_memories":        len(texts),
        "num_traversal_candidates": traversal_candidates,
        "num_casts":           num_casts,
        "noise":               noise,
        "meaning_enabled":     meaning_db is not None,
    }
