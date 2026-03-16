"""
srm/synthesis.py — Resonance-Weighted Extractive Assembly (RWEA).

Builds a coherent response from the top-ranked attractor memories
using sentence-level scoring and greedy MMR de-duplication.

Scoring:
    score(s) = resonance × cosine(query, s) × pos_weight × content_ratio

    resonance     = votes / max_votes   (normalised stochastic signal)
    pos_weight    = 1.0 for first sentence, 0.82 for subsequent
    content_ratio = fraction of non-stopword tokens in sentence

Selection (greedy MMR):
    Pick the highest-scoring sentence not too similar (cosine > SIM_THRESH)
    to any already-selected sentence.  Prevents redundant facts.

Assembly:
    Lightweight discourse connectives (support / contrast / conclude)
    connect selected sentences into a flowing paragraph.
"""

from __future__ import annotations

import re
import random

from .config import (
    SIM_THRESH, MAX_WORDS,
    CONN_SUPPORT, CONN_CONTRAST, CONN_CONCLUDE,
    STOPWORDS, NUM_CASTS,
)
from .nlp import tokenise, idf_table, tfidf_vec, cosine, expand_query


# Negation markers that trigger a contrast connective
_NEG_WORDS = frozenset({
    "not", "no", "never", "without", "unlike", "but", "yet",
    "although", "however", "despite",
})


def _split_sentences(text: str) -> list[str]:
    """
    Split text on sentence boundaries.
    Keeps only sentences with ≥ 4 words to filter noise.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.split()) >= 4]


def synthesise(
    query: str,
    top_attractors: list[tuple[int, int, str]],
    num_casts: int    = NUM_CASTS,
    max_words: int    = MAX_WORDS,
    sim_thresh: float = SIM_THRESH,
) -> str:
    """
    Build a response string from the top-ranked attractor memories.

    Args:
        query:          original (un-expanded) query string
        top_attractors: list of (mem_idx, votes, text) tuples, ranked
        num_casts:      used to normalise vote share into resonance
        max_words:      word budget for the assembled response
        sim_thresh:     MMR cosine threshold for redundancy filtering

    Returns:
        A synthesised paragraph, or falls back to the top attractor's
        raw text if no sentences pass the scoring filters.
    """
    if not top_attractors:
        return "(no resonant attractors found)"

    all_texts = [t for _, _, t in top_attractors]
    idf = idf_table(all_texts + [query])

    # Use the expanded query for richer sentence-to-query similarity
    expanded = expand_query(query)
    q_vec    = tfidf_vec(tokenise(expanded), idf)
    max_v    = max(v for _, v, _ in top_attractors) or 1

    # ── Score every sentence from every attractor ──────────────────────
    candidates: list[dict] = []
    for mem_idx, votes, text in top_attractors:
        resonance = votes / max_v
        for pos, sent in enumerate(_split_sentences(text)):
            toks     = tokenise(sent)
            svec     = tfidf_vec(toks, idf)
            q_sim    = cosine(q_vec, svec)
            pos_w    = 1.0 if pos == 0 else 0.82
            content_r = (
                sum(1 for t in toks if t not in STOPWORDS) / max(len(toks), 1)
            )
            score = resonance * q_sim * pos_w * (0.4 + 0.6 * content_r)
            candidates.append({
                "sent":    sent,
                "score":   score,
                "vec":     svec,
                "mem_idx": mem_idx,
            })

    # Discard near-zero scores (no meaningful match)
    candidates.sort(key=lambda x: -x["score"])
    candidates = [c for c in candidates if c["score"] > 1e-4]

    # ── Greedy MMR selection ───────────────────────────────────────────
    selected: list[dict] = []
    budget = max_words
    for cand in candidates:
        wc = len(cand["sent"].split())
        if wc > budget:
            continue
        # Skip if too similar to any already-chosen sentence
        if any(cosine(cand["vec"], s["vec"]) > sim_thresh for s in selected):
            continue
        selected.append(cand)
        budget -= wc
        if budget <= 0:
            break

    # Fall back to the highest-ranked attractor's raw text
    if not selected:
        return top_attractors[0][2]

    # ── Assemble with discourse connectives ───────────────────────────
    parts: list[str] = []
    n = len(selected)
    for i, sel in enumerate(selected):
        sent = sel["sent"].rstrip().rstrip(".") + "."
        if i == 0:
            prefix = ""
        elif i == n - 1 and n > 2:
            prefix = random.choice(CONN_CONCLUDE)
        elif any(w in tokenise(sent) for w in _NEG_WORDS):
            prefix = random.choice(CONN_CONTRAST)
        else:
            prefix = random.choice(CONN_SUPPORT)
        parts.append(prefix + sent)

    return " ".join(parts)
