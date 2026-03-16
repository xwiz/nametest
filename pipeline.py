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

from .config import TOP_K, NUM_CASTS, NOISE, MIN_COS, VOTE_FLOOR, W_VOTE, W_COS
from .nlp import tokenise, tfidf_vec, cosine, expand_query
from .encoding import encode, hamming_batch
from .traversal import traverse
from .synthesis import synthesise
from .store import MemoryStore


def srm_query(
    query_text: str,
    store: MemoryStore,
    top_k:     int   = TOP_K,
    num_casts: int   = NUM_CASTS,
    noise:     float = NOISE,
    meaning_db = None,     # srm.meaning.MeaningDB | None
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
    q_code = encode(expanded, idf, meaning_db=meaning_db)

    # ── 3. Traverse ────────────────────────────────────────────────────
    votes, cast_log = traverse(q_code, codes, num_casts=num_casts, noise=noise)

    # Hamming distances from query to every memory (for reporting)
    dists = hamming_batch(q_code, codes)

    # ── 4. Hybrid re-rank ──────────────────────────────────────────────
    q_vec    = tfidf_vec(tokenise(expanded), idf)
    mem_vecs = [tfidf_vec(tokenise(t), idf) for t in texts]

    n = len(texts)
    scores = [
        W_VOTE * (votes.get(i, 0) / num_casts) + W_COS * cosine(q_vec, mem_vecs[i])
        for i in range(n)
    ]

    ranked = sorted(range(n), key=lambda i: -scores[i])[:top_k]

    # ── 5. Filter ──────────────────────────────────────────────────────
    ranked = [
        i for i in ranked
        if cosine(q_vec, mem_vecs[i]) > MIN_COS or votes.get(i, 0) > VOTE_FLOOR
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
            "cosine":       round(cosine(q_vec, mem_vecs[i]), 3),
            "hybrid_score": round(scores[i], 3),
        }
        for i in ranked
    ]

    return {
        "query":             query_text,
        "expanded_query":    expanded if expanded != query_text else None,
        "response":          (
            synthesise(query_text, top_attractors, num_casts=num_casts)
            if top_attractors
            else "No relevant memories found."
        ),
        "top_attractors":    top_attractors,
        "attractor_details": attractor_details,
        "cast_log":          cast_log,
        "vote_distribution": {i: int(v) for i, v in votes.items()},
        "num_memories":      n,
        "num_casts":         num_casts,
        "noise":             noise,
        "meaning_enabled":   meaning_db is not None,
    }
