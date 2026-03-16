"""
srm/nlp.py — text processing utilities.

Intentionally free of external NLP libraries; uses only stdlib + numpy
so the whole system stays dependency-light.
"""

from __future__ import annotations

import re
import math
from collections import Counter

from .config import STOPWORDS, EXPANSIONS


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    """Word tokens, min length 2, lowercased, ASCII letters only."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]


# ── IDF ───────────────────────────────────────────────────────────────────────

def idf_table(corpus: list[str]) -> dict[str, float]:
    """
    Smooth IDF over a corpus of raw strings.

    idf(t) = log((N+1) / (df(t)+1)) + 0.5

    The +1 Laplace smoothing prevents division-by-zero for unseen terms
    and prevents log(0) for terms present in every document.
    """
    N = len(corpus)
    if N == 0:
        return {}
    df: Counter = Counter()
    for doc in corpus:
        for tok in set(tokenise(doc)):
            df[tok] += 1
    return {t: math.log((N + 1) / (f + 1)) + 0.5 for t, f in df.items()}


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def tfidf_vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """
    TF-IDF vector over *content* tokens (stopwords and very short tokens
    are removed first).

    Stopword removal prevents common function words from generating
    false cosine similarity between unrelated documents.
    Falls back to the full token list if filtering leaves nothing.
    """
    content = [t for t in tokens if t not in STOPWORDS and len(t) > 2] or tokens
    tf = Counter(content)
    n  = max(len(content), 1)
    return {t: (c / n) * idf.get(t, 0.0) for t, c in tf.items()}


# ── Cosine similarity ─────────────────────────────────────────────────────────

def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na  = math.sqrt(sum(v * v for v in a.values()))
    nb  = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ── Query expansion ───────────────────────────────────────────────────────────

def expand_query(text: str) -> str:
    """
    Append domain synonyms for key terms found in the query.

    Bridges the vocabulary gap between natural-language queries and
    the specific terminology stored in memories — without embeddings.

    Example:
        "How does the brain learn?" →
        "How does the brain learn? neuron synapse neurotransmitter plasticity
         neural train weights backprop gradient"
    """
    toks = tokenise(text)
    extras: list[str] = []
    seen: set[str] = set(toks)
    for t in toks:
        for exp in EXPANSIONS.get(t, []):
            if exp not in seen:
                extras.append(exp)
                seen.add(exp)
    return (text + " " + " ".join(extras)) if extras else text
