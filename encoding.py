"""
srm/encoding.py — SimHash projection into Hamming space.

Architecture:
  Each memory (or query) is projected to a CODE_BITS-dimensional binary
  code via weighted SimHash over three token types:

    Unigrams  (weight 2.5 × IDF)  — primary semantic signal
    Bigrams   (weight 1.5 × mean IDF) — phrase proximity
    Char 3-grams (weight 0.8)     — morphological fallback

  The MD5 hash of each token gives a 128-bit projection direction.
  Bit b of the code is 1 iff the weighted sum of projections along b > 0.

  Vectorised via np.unpackbits(..., bitorder='little') — roughly 8×
  faster than a Python-level loop over 128 bits.
"""

from __future__ import annotations

import re
import hashlib

import numpy as np

from .config import CODE_BITS, PACK_BYTES

# Pre-computed population-count LUT for fast Hamming distance
_POPCOUNT: np.ndarray = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.int32
)


# ── Token+weight generator ────────────────────────────────────────────────────

def _sim_tokens(
    text: str,
    idf: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """
    Generate (token, weight) pairs for SimHash projection.

    Token types and base weights:
      - Unigrams:   2.5 × IDF   dominant signal
      - Bigrams:    1.5 × mean IDF  captures phrase proximity
      - Char 3-grams: 0.8       morphological fallback for OOV terms

    IDF scaling ensures rare discriminative terms (e.g. "mitochondria",
    "backpropagation") dominate over common terms ("the", "cell").
    """
    t     = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    words = t.split()
    base  = idf or {}
    out: list[tuple[str, float]] = []

    for w in words:
        out.append((w, 2.5 * base.get(w, 1.0)))

    for i in range(len(words) - 1):
        bigram = words[i] + "_" + words[i + 1]
        w_idf  = (base.get(words[i], 1.0) + base.get(words[i + 1], 1.0)) / 2
        out.append((bigram, 1.5 * w_idf))

    for i in range(len(t) - 2):
        out.append((t[i : i + 3], 0.8))

    return out


# ── Encode ────────────────────────────────────────────────────────────────────

def encode(
    text: str,
    idf: dict[str, float] | None = None,
    meaning_db=None,              # srm.meaning.MeaningDB | None
) -> np.ndarray:
    """
    Project *text* into CODE_BITS-dimensional Hamming space via SimHash.

    Standard mode (meaning_db=None):
        For each token t with weight w:
            h = MD5(t)
            weights[b] += w if bit b of h is 1, else -w

    Meaning-aware mode (meaning_db provided):
        Object tokens following a known verb additionally have the verb's
        128-bit polarity_mask XOR-ed into h before weight accumulation:
            h = MD5(t) XOR polarity_mask
        This pushes semantically opposite predicates (kill/protect) to
        different Hamming regions while keeping unrelated tokens unchanged.

        Adjective weight multipliers are also applied (strong→×1.7,
        slight→×0.4, negation→weight flip).

    Both modes share the same bigram and char-3gram supplementary tokens
    (these are always unmasked — the mask applies only to object unigrams).

    Code bit b = 1  iff  weights[b] > 0.
    Returns a packed uint8 array of shape (PACK_BYTES,).
    """
    weights = np.zeros(CODE_BITS, dtype=np.float64)

    if meaning_db is not None:
        # ── Meaning-aware path ────────────────────────────────────────
        from .nlp import tokenise
        from .meaning import apply_meaning

        tokens = tokenise(text)
        triples = apply_meaning(tokens, idf or {}, meaning_db)

        for tok, w, mask in triples:
            raw   = hashlib.md5(tok.encode()).digest()
            rbytes = np.frombuffer(raw, dtype=np.uint8)

            # XOR polarity mask into the hash bytes for object tokens
            if mask is not None:
                rbytes = rbytes ^ mask          # element-wise XOR

            bits = np.unpackbits(rbytes, bitorder="little")
            weights += np.where(bits, w, -w)

        # Bigrams and char-3grams are always unmasked (structural signal)
        t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        words = t.split()
        base  = idf or {}
        for i in range(len(words) - 1):
            bigram = words[i] + "_" + words[i + 1]
            w_idf  = (base.get(words[i], 1.0) + base.get(words[i + 1], 1.0)) / 2
            raw    = hashlib.md5(bigram.encode()).digest()
            bits   = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
            weights += np.where(bits, 1.5 * w_idf, -1.5 * w_idf)
        for i in range(len(t) - 2):
            raw  = hashlib.md5(t[i:i+3].encode()).digest()
            bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
            weights += np.where(bits, 0.8, -0.8)

    else:
        # ── Standard path (no meaning DB) ────────────────────────────
        for tok, w in _sim_tokens(text, idf):
            raw  = hashlib.md5(tok.encode()).digest()
            bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
            weights += np.where(bits, w, -w)

    return np.packbits((weights > 0).astype(np.uint8))


def encode_batch(
    texts: list[str],
    idf: dict[str, float],
    meaning_db=None,
) -> np.ndarray:
    """Encode a list of texts, returning shape (len(texts), PACK_BYTES).

    Passes meaning_db through to encode() when provided.
    """
    return np.array(
        [encode(t, idf, meaning_db=meaning_db) for t in texts],
        dtype=np.uint8,
    )


# ── Hamming distance ──────────────────────────────────────────────────────────

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two packed binary codes."""
    return int(_POPCOUNT[a ^ b].sum())


def hamming_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Hamming distances from *query* to every row in *matrix*.

    Uses XOR + LUT pop-count; avoids any Python loop over rows.

    Args:
        query:  shape (PACK_BYTES,)
        matrix: shape (N, PACK_BYTES)

    Returns:
        distances: shape (N,) int32
    """
    return _POPCOUNT[matrix ^ query[np.newaxis, :]].sum(axis=1)
