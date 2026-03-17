"""
srm/traversal.py — stochastic resonance traversal in Hamming space.

The key idea: fire many noisy probes from the query attractor.
True attractors (memories semantically close to the query) will
consistently attract probes despite the random bit-flips, and
accumulate votes across NUM_CASTS independent trials.

False positives that happen to be close in one cast will rarely win
consistently, so vote count is a robust retrieval signal.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.random import Generator

from .config import NUM_CASTS, NOISE
from .encoding import hamming_batch


# ── Single probe ──────────────────────────────────────────────────────────────

def _cast(code: np.ndarray, noise: float, rng: Generator) -> np.ndarray:
    """
    Perturb *code* by flipping each bit independently with probability *noise*.

    Unpacks to bit-level, applies Bernoulli flips, repacks.
    Returns a new array; does not mutate *code*.
    """
    bits  = np.unpackbits(code)
    mask  = (rng.random(len(bits)) < noise).astype(np.uint8)
    return np.packbits(bits ^ mask)


# ── Traversal ─────────────────────────────────────────────────────────────────

def traverse(
    query_code: np.ndarray,
    mem_codes: np.ndarray,
    num_casts: int = NUM_CASTS,
    noise: float   = NOISE,
    rng: Generator | None = None,
) -> tuple[Counter, list[dict]]:
    """
    Cast *num_casts* noisy probes from the query attractor and collect votes.

    Each probe is the query code with random bit-flips at rate *noise*.
    The probe "lands on" its nearest memory (argmin Hamming distance).
    Memories that are true attractors for the query accumulate votes
    across many independent casts despite the noise.

    Args:
        query_code: encoded query, shape (PACK_BYTES,)
        mem_codes:  encoded memories, shape (N, PACK_BYTES)
        num_casts:  number of independent probes
        noise:      per-bit flip probability [0, 1]

    Returns:
        votes:    Counter  memory_index → vote count
        cast_log: list of {cast_id, landed_on, hamming_dist} for debugging
                  and visualisation
    """
    votes: Counter       = Counter()
    cast_log: list[dict] = []

    if rng is None:
        rng = np.random.default_rng()

    # Precompute unpacked bits once; only the flip mask changes per cast.
    q_bits = np.unpackbits(query_code)

    for i in range(num_casts):
        mask  = (rng.random(len(q_bits)) < noise).astype(np.uint8)
        probe = np.packbits(q_bits ^ mask)
        dists = hamming_batch(probe, mem_codes)
        best  = int(np.argmin(dists))
        votes[best] += 1
        cast_log.append({
            "cast_id":     i,
            "landed_on":   best,
            "hamming_dist": int(dists[best]),
        })

    return votes, cast_log
