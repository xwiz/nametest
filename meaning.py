"""
srm/meaning.py — meaning-aware encoding layer for SRM.

What this module adds
─────────────────────
Standard SRM SimHash treats all tokens as a flat weighted bag.
"Antibiotics kill bacteria" and "Antibiotics protect bacteria"
land in nearly the same Hamming region because noun IDF weights
dominate and the verb barely shifts the code.

This module adds two semantic transforms applied *inside* encode():

  1. Verb polarity masking
     ─────────────────────
     When a known verb is detected in a token sequence, its pre-computed
     128-bit polarity bitmask is XOR-ed into the MD5 hash bits of the
     following object nouns before weight accumulation.  Synonym verbs
     (kill / inhibit / disrupt) share one mask; antonym verbs (kill /
     protect) get maximally distant masks (hamming ≈ 64).

     Concretely:
       encode_bits(token)           = MD5(token)
       encode_bits(token, mask)     = MD5(token) XOR mask

     This pushes "bacteria-as-object-of-kill" to a different Hamming
     region than "bacteria-as-object-of-protect".

  2. Adjective weight scaling
     ─────────────────────────
     Intensifying adjectives (strong, critical, primary) multiply the
     IDF weight of the following noun.  Attenuating adjectives (slight,
     minor) reduce it.  Negation words (no, without, absent) flip the
     noun's weight to negative, so it contributes the inverse direction.

  3. SVO detection (heuristic, no parser required)
     ───────────────────────────────────────────────
     English SVO order: scan left-to-right, mark the first known verb,
     treat subsequent content tokens (until the next verb or sentence
     boundary) as objects.  Wrong ~15–20% of the time on complex
     sentences; adequate for factual KB entries.

Public API
──────────
  db = MeaningDB("meaning.db")
  db.get_verb(verb)          → VerbMeaning | None
  db.get_polarity_mask(verb) → np.ndarray | None   # 16 bytes
  db.known_verbs()           → set[str]

  # Used by encoding.py
  apply_meaning(tokens, idf, db) → list[(token, weight, mask|None)]
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


# ── Adjective weight multipliers (hardcoded; no JSON source needed) ──────────
#
# These amplify or attenuate the IDF weight of the noun they precede.
# Negation words produce weight = -(original), effectively flipping
# the noun's contribution direction in Hamming space.

ADJ_MULTIPLIERS: dict[str, float] = {
    # amplifiers
    "strong":       1.7,
    "strongly":     1.7,
    "critical":     1.8,
    "primary":      1.6,
    "major":        1.6,
    "significant":  1.6,
    "severe":       1.8,
    "extreme":      1.9,
    "complete":     1.5,
    "direct":       1.4,
    "essential":    1.5,
    "key":          1.4,
    "central":      1.4,
    "dominant":     1.6,
    "profound":     1.6,
    "intense":      1.6,
    "acute":        1.7,
    "chronic":      1.5,
    # attenuators
    "slight":       0.4,
    "slightly":     0.4,
    "minor":        0.4,
    "weak":         0.4,
    "weakly":       0.4,
    "mild":         0.5,
    "mildly":       0.5,
    "partial":      0.6,
    "limited":      0.5,
    "small":        0.5,
    "marginal":     0.4,
    "minimal":      0.3,
    "moderate":     0.7,
    "indirect":     0.6,
    # negation → weight flip
    "no":           -1.0,
    "not":          -1.0,
    "never":        -1.0,
    "without":      -1.0,
    "absent":       -1.0,
    "lack":         -1.0,
    "lacking":      -1.0,
    "none":         -1.0,
    "non":          -1.0,
    "un":           -1.0,
}

# Simple heuristic content-word POS signals: if a token ends with one of
# these suffixes it is likely a noun/adjective and a candidate object.
_NOUN_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ism", "ist",
    "sis", "ase", "ose", "ine", "ide", "ate", "oma",
)


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class VerbMeaning:
    verb:           str
    polarity_class: str
    polarity_mask:  np.ndarray      # uint8, shape (16,)
    goals:          list[str]       = field(default_factory=list)
    mechanisms:     list[str]       = field(default_factory=list)
    final_object:   list[str]       = field(default_factory=list)  # physical states


# ── MeaningDB ─────────────────────────────────────────────────────────────────

class MeaningDB:
    """
    Read-only interface to the meaning SQLite database.

    The connection is opened once; verb lookups are LRU-cached so repeated
    calls within a query cycle (common for short KB sentences) are free.

    Usage:
        db = MeaningDB("meaning.db")
        vm = db.get_verb("kill")          # VerbMeaning or None
        mask = db.get_polarity_mask("kill")  # np.ndarray or None
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._verify()
        self._verb_cache: dict[str, VerbMeaning | None] = {}
        self._known: set[str] | None = None

    def _verify(self) -> None:
        tables = {r[0] for r in
                  self._conn.execute(
                      "SELECT name FROM sqlite_master WHERE type='table'"
                  ).fetchall()}
        required = {"verbs", "polarity_classes", "verb_polarity"}
        if not required.issubset(tables):
            raise RuntimeError(
                f"meaning.db is missing tables {required - tables}. "
                "Run scripts/build_meaning_db.py first."
            )

    # ── Lookups ───────────────────────────────────────────────────────

    def get_verb(self, verb: str) -> VerbMeaning | None:
        """Return VerbMeaning for *verb*, or None if unknown."""
        v = verb.lower().strip()
        if v in self._verb_cache:
            return self._verb_cache[v]

        row = self._conn.execute(
            "SELECT verb, polarity_class, polarity_mask "
            "FROM verb_meaning WHERE verb=?", (v,)
        ).fetchone()

        if row is None:
            self._verb_cache[v] = None
            return None

        goals = [r[0] for r in self._conn.execute(
            "SELECT goal FROM verb_goals WHERE verb_id="
            "(SELECT id FROM verbs WHERE verb=?)", (v,)
        ).fetchall()]

        mechs = [r[0] for r in self._conn.execute(
            "SELECT mechanism FROM verb_mechanisms WHERE verb_id="
            "(SELECT id FROM verbs WHERE verb=?)", (v,)
        ).fetchall()]

        final_obj = [r[0] for r in self._conn.execute(
            """SELECT state_value FROM state_transforms
               WHERE verb_id=(SELECT id FROM verbs WHERE verb=?)
               AND role='object' AND phase='final' AND dimension='physical'""",
            (v,)
        ).fetchall()]

        mask = np.frombuffer(row["polarity_mask"], dtype=np.uint8).copy()

        vm = VerbMeaning(
            verb           = row["verb"],
            polarity_class = row["polarity_class"],
            polarity_mask  = mask,
            goals          = goals,
            mechanisms     = mechs,
            final_object   = final_obj,
        )
        self._verb_cache[v] = vm
        return vm

    def get_polarity_mask(self, verb: str) -> np.ndarray | None:
        """Return the 128-bit polarity bitmask for *verb*, or None."""
        vm = self.get_verb(verb)
        return vm.polarity_mask if vm else None

    def known_verbs(self) -> set[str]:
        """Return the full set of verbs in the DB (cached after first call)."""
        if self._known is None:
            rows = self._conn.execute("SELECT verb FROM verbs").fetchall()
            self._known = {r[0] for r in rows}
        return self._known

    def class_mask(self, class_name: str) -> np.ndarray | None:
        """Return the bitmask for a polarity class by name."""
        row = self._conn.execute(
            "SELECT bitmask FROM polarity_classes WHERE name=?", (class_name,)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.uint8).copy()

    def close(self) -> None:
        self._conn.close()

    def __repr__(self) -> str:
        n = self._conn.execute("SELECT COUNT(*) FROM verbs").fetchone()[0]
        return f"MeaningDB(path={self.db_path!r}, verbs={n})"


# ── SVO detection ─────────────────────────────────────────────────────────────

def _detect_svo(
    tokens: list[str],
    known_verbs: set[str],
) -> list[tuple[str | None, list[str]]]:
    """
    Heuristic SVO chunking without a parser.

    Scan left-to-right.  When a known verb is found:
      - everything to its left (up to the previous verb) = subject tokens
      - everything to its right (until the next verb) = object tokens

    Returns list of (verb | None, [object_tokens]).
    "None" means tokens that precede the first verb (subject region).

    Example:
        ["antibiotics", "kill", "bacteria", "by", "disrupting", "cell", "walls"]
        → [(None, ["antibiotics"]),
           ("kill", ["bacteria", "by"]),
           ("disrupting", ["cell", "walls"])]
    """
    result: list[tuple[str | None, list[str]]] = []
    current_verb: str | None = None
    current_obj:  list[str]  = []

    for tok in tokens:
        if tok in known_verbs:
            result.append((current_verb, current_obj))
            current_verb = tok
            current_obj  = []
        else:
            current_obj.append(tok)

    result.append((current_verb, current_obj))
    return result


# ── Main integration function ─────────────────────────────────────────────────

def apply_meaning(
    tokens: list[str],
    idf: dict[str, float],
    db: MeaningDB,
) -> list[tuple[str, float, np.ndarray | None]]:
    """
    Produce meaning-aware (token, weight, mask | None) triples.

    Extends the standard _sim_tokens() unigram list:
      - Object nouns following a known verb get the verb's polarity_mask.
      - Nouns preceded by a scaling adjective get a modified IDF weight.
      - Negation words apply a −1.0 multiplier to the next content token.

    The mask (or None) is passed to encode() which XORs it into the
    MD5 hash bits for that token's projection, displacing it to the
    appropriate polarity region of Hamming space.

    Args:
        tokens:   lowercased word tokens from tokenise()
        idf:      corpus IDF table
        db:       open MeaningDB instance

    Returns:
        list of (token, weight, mask_or_None)
    """
    known = db.known_verbs()
    chunks = _detect_svo(tokens, known)

    result: list[tuple[str, float, np.ndarray | None]] = []

    for verb_tok, obj_tokens in chunks:
        # Look up the mask once per verb (None for unknown verbs)
        mask: np.ndarray | None = None
        if verb_tok is not None:
            mask = db.get_polarity_mask(verb_tok)
            # Add the verb itself as a high-weight unmasked token
            w_verb = 2.5 * idf.get(verb_tok, 1.0)
            result.append((verb_tok, w_verb, None))

        # Walk object tokens, tracking pending adj multipliers
        pending_mult: float = 1.0
        for tok in obj_tokens:
            if tok in ADJ_MULTIPLIERS:
                pending_mult *= ADJ_MULTIPLIERS[tok]
                # Add the adjective itself at normal weight, no mask
                w_adj = 2.5 * idf.get(tok, 1.0)
                result.append((tok, w_adj, None))
                continue

            base_w = 2.5 * idf.get(tok, 1.0)
            final_w = base_w * pending_mult

            # Negative weight = flip sign; still apply mask so the
            # token lands in the correct polarity region, just inverted.
            result.append((tok, final_w, mask))
            pending_mult = 1.0  # reset after each content token

    return result


# ── Convenience: describe a verb's meaning for display / debug ───────────────

def describe_verb(verb: str, db: MeaningDB) -> str:
    """Return a one-line human-readable summary of a verb's semantics."""
    vm = db.get_verb(verb)
    if vm is None:
        return f"{verb!r} — not in meaning DB"
    goals  = ", ".join(vm.goals[:3]) or "—"
    states = ", ".join(vm.final_object[:3]) or "—"
    return (
        f"{verb!r}  [{vm.polarity_class}]"
        f"  goals: {goals}"
        f"  → object ends: {states}"
    )
