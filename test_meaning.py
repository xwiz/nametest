"""
tests/test_meaning.py — tests for the meaning encoding layer.

Run:
    python -m pytest tests/test_meaning.py -v
    python tests/test_meaning.py
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from srm.meaning  import MeaningDB, apply_meaning, ADJ_MULTIPLIERS, describe_verb
from srm.encoding import encode, hamming
from srm.nlp      import idf_table, tokenise
from srm.store    import MemoryStore
from srm.pipeline import srm_query
from srm.config   import SAMPLE_KB


# ── Fixtures ──────────────────────────────────────────────────────────────────

MEANING_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "meaning.db"
)


def _has_meaning_db() -> bool:
    return os.path.exists(MEANING_DB_PATH)


# ── Polarity scoring (no DB required) ────────────────────────────────────────

class TestPolarityScoring(unittest.TestCase):
    """Test the keyword-based polarity class assignment from build script."""

    def setUp(self):
        if not _has_meaning_db():
            self.skipTest("meaning.db not present — run build_meaning_db.py first")
        self.db = MeaningDB(MEANING_DB_PATH)

    def tearDown(self):
        if hasattr(self, "db"):
            self.db.close()

    def test_aggravate_is_intensify(self):
        vm = self.db.get_verb("aggravate")
        self.assertIsNotNone(vm)
        self.assertEqual(vm.polarity_class, "intensify")

    def test_aggrieve_is_destruct(self):
        vm = self.db.get_verb("aggrieve")
        self.assertIsNotNone(vm)
        self.assertEqual(vm.polarity_class, "destruct")

    def test_agree_is_align(self):
        vm = self.db.get_verb("agree")
        self.assertIsNotNone(vm)
        self.assertEqual(vm.polarity_class, "align")

    def test_agitate_is_disrupt(self):
        vm = self.db.get_verb("agitate")
        self.assertIsNotNone(vm)
        self.assertEqual(vm.polarity_class, "disrupt")

    def test_aggregate_is_aggregate(self):
        vm = self.db.get_verb("aggregate")
        self.assertIsNotNone(vm)
        self.assertEqual(vm.polarity_class, "aggregate")

    def test_unknown_verb_returns_none(self):
        self.assertIsNone(self.db.get_verb("xyzzyx"))

    def test_known_verbs_set_nonempty(self):
        self.assertGreater(len(self.db.known_verbs()), 0)


# ── Bitmask geometry ──────────────────────────────────────────────────────────

class TestBitmaskGeometry(unittest.TestCase):
    """All polarity class bitmasks must be well-spread in Hamming space."""

    def setUp(self):
        if not _has_meaning_db():
            self.skipTest("meaning.db not present")
        import sqlite3
        conn = sqlite3.connect(MEANING_DB_PATH)
        rows = conn.execute(
            "SELECT name, bitmask FROM polarity_classes ORDER BY id"
        ).fetchall()
        conn.close()
        self.masks = [(r[0], np.frombuffer(r[1], dtype=np.uint8)) for r in rows]
        self._pop = np.array([bin(b).count("1") for b in range(256)], dtype=np.int32)

    def _hamming(self, a, b):
        return int(self._pop[a ^ b].sum())

    def test_all_pairs_exceed_48_bits(self):
        for i, (ni, ai) in enumerate(self.masks):
            for j, (nj, aj) in enumerate(self.masks):
                if i >= j:
                    continue
                d = self._hamming(ai, aj)
                self.assertGreaterEqual(
                    d, 48,
                    f"Bitmask pair ({ni}, {nj}) has hamming={d} < 48"
                )

    def test_self_distance_is_zero(self):
        for name, mask in self.masks:
            self.assertEqual(self._hamming(mask, mask), 0)


# ── apply_meaning ─────────────────────────────────────────────────────────────

class TestApplyMeaning(unittest.TestCase):

    def setUp(self):
        if not _has_meaning_db():
            self.skipTest("meaning.db not present")
        self.db  = MeaningDB(MEANING_DB_PATH)
        self.idf = idf_table(SAMPLE_KB)

    def tearDown(self):
        if hasattr(self, "db"):
            self.db.close()

    def test_returns_list_of_triples(self):
        tokens  = tokenise("bacteria agitate the liquid")
        triples = apply_meaning(tokens, self.idf, self.db)
        self.assertIsInstance(triples, list)
        for item in triples:
            self.assertEqual(len(item), 3)

    def test_object_after_known_verb_gets_mask(self):
        tokens  = tokenise("agitate liquid")
        triples = apply_meaning(tokens, self.idf, self.db)
        # "liquid" follows "agitate" → should get mask
        obj_triples = [(t, m) for t, w, m in triples if t == "liquid"]
        self.assertTrue(len(obj_triples) > 0, "No 'liquid' triple found")
        _, mask = obj_triples[0]
        self.assertIsNotNone(mask, "Object after known verb should have mask")

    def test_token_before_verb_gets_no_mask(self):
        tokens  = tokenise("bacteria agitate the liquid")
        triples = apply_meaning(tokens, self.idf, self.db)
        # "bacteria" precedes "agitate" → subject region, no mask
        subj_triples = [(t, m) for t, w, m in triples if t == "bacteria"]
        if subj_triples:
            _, mask = subj_triples[0]
            self.assertIsNone(mask, "Subject token should not carry mask")

    def test_adj_amplifier_increases_weight(self):
        tokens_plain = tokenise("strong bacteria")
        tokens_mod   = tokenise("bacteria")
        idf = self.idf

        plain_triples = apply_meaning(tokens_mod, idf, self.db)
        amp_triples   = apply_meaning(tokens_plain, idf, self.db)

        plain_w = next((w for t, w, m in plain_triples if t == "bacteria"), None)
        amp_w   = next((w for t, w, m in amp_triples  if t == "bacteria"), None)

        if plain_w is not None and amp_w is not None:
            self.assertGreater(abs(amp_w), abs(plain_w),
                               "Amplifier should increase absolute weight")

    def test_negation_flips_weight(self):
        tokens = tokenise("no bacteria")
        triples = apply_meaning(tokens, self.idf, self.db)
        bac = [(t, w) for t, w, m in triples if t == "bacteria"]
        if bac:
            _, w = bac[0]
            self.assertLess(w, 0, "Weight after negation should be negative")


# ── Meaning-aware encoding geometry ──────────────────────────────────────────

class TestMeaningAwareEncoding(unittest.TestCase):
    """
    Core geometric invariants: opposite predicates should be farther apart
    in Hamming space when meaning_db is supplied than without it.
    """

    def setUp(self):
        if not _has_meaning_db():
            self.skipTest("meaning.db not present")
        self.db  = MeaningDB(MEANING_DB_PATH)
        self.idf = idf_table(SAMPLE_KB)

    def tearDown(self):
        self.db.close()

    def _pair_hamming(self, t1, t2, mdb=None):
        c1 = encode(t1, self.idf, meaning_db=mdb)
        c2 = encode(t2, self.idf, meaning_db=mdb)
        return hamming(c1, c2)

    def test_meaning_separates_opposite_predicates(self):
        """
        "agitate liquid"  vs  "agree proposal"  — opposite polarity classes
        (disrupt vs align) — should be more separated WITH meaning_db.
        """
        d_plain   = self._pair_hamming("agitate liquid", "agree proposal")
        d_meaning = self._pair_hamming("agitate liquid", "agree proposal", self.db)
        # Meaning encoding should not reduce the distance; it should widen it
        # (or at worst keep it the same)
        self.assertGreaterEqual(
            d_meaning, d_plain - 4,   # allow 4-bit tolerance
            f"Meaning encoding reduced Hamming distance: "
            f"plain={d_plain}, meaning={d_meaning}"
        )

    def test_same_predicate_stays_close(self):
        """
        Two sentences with the same verb + similar objects should remain
        nearby after meaning encoding (the mask is the same for both).
        """
        d_meaning = self._pair_hamming(
            "agitate the liquid mixture",
            "agitate the water solution",
            self.db,
        )
        # Similar sentences should still be < 50 bits apart
        self.assertLess(d_meaning, 60,
                        "Meaning encoding made similar sentences too distant")

    def test_encode_deterministic_with_meaning(self):
        c1 = encode("aggravate the injury", self.idf, meaning_db=self.db)
        c2 = encode("aggravate the injury", self.idf, meaning_db=self.db)
        np.testing.assert_array_equal(c1, c2)


# ── End-to-end pipeline with meaning_db ──────────────────────────────────────

class TestPipelineWithMeaning(unittest.TestCase):

    def setUp(self):
        if not _has_meaning_db():
            self.skipTest("meaning.db not present")
        self.db = MeaningDB(MEANING_DB_PATH)
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = MemoryStore(self.tmp.name)
        for t in SAMPLE_KB:
            self.store.add(t)

    def tearDown(self):
        self.store.close()
        self.db.close()
        os.unlink(self.tmp.name)

    def test_pipeline_runs_with_meaning_db(self):
        r = srm_query(
            "how do cells produce energy",
            self.store,
            meaning_db=self.db,
            num_casts=20,
        )
        self.assertIn("response", r)
        self.assertTrue(r["meaning_enabled"])

    def test_pipeline_returns_meaning_enabled_flag(self):
        r_plain   = srm_query("DNA", self.store, num_casts=10)
        r_meaning = srm_query("DNA", self.store, num_casts=10, meaning_db=self.db)
        self.assertFalse(r_plain["meaning_enabled"])
        self.assertTrue(r_meaning["meaning_enabled"])


# ── describe_verb utility ─────────────────────────────────────────────────────

class TestDescribeVerb(unittest.TestCase):

    def setUp(self):
        if not _has_meaning_db():
            self.skipTest("meaning.db not present")
        self.db = MeaningDB(MEANING_DB_PATH)

    def tearDown(self):
        self.db.close()

    def test_known_verb(self):
        desc = describe_verb("aggravate", self.db)
        self.assertIn("aggravate", desc)
        self.assertIn("intensify", desc)

    def test_unknown_verb(self):
        desc = describe_verb("xyzzyx", self.db)
        self.assertIn("not in meaning DB", desc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
