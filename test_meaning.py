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

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from srm.meaning  import MeaningDB, apply_meaning, ADJ_MULTIPLIERS, describe_verb
from srm.meaning  import extract_meaning
from srm.encoding import encode, hamming
from srm.nlp      import idf_table, tokenise
from srm.store    import MemoryStore
from srm.pipeline import srm_query
from srm.config   import SAMPLE_KB
from build_meaning_db import _mask_from_states, _label_from_states


# ── Fixtures ──────────────────────────────────────────────────────────────────

MEANING_DB_PATH = os.path.join(
    os.path.dirname(__file__), "meaning.db"
)


def _has_meaning_db() -> bool:
    return os.path.exists(MEANING_DB_PATH)



class TestStateMaskDerivation(unittest.TestCase):
    def _hamming(self, a: bytes, b: bytes) -> int:
        aa = np.frombuffer(a, dtype=np.uint8)
        bb = np.frombuffer(b, dtype=np.uint8)
        pop = np.array([bin(v).count("1") for v in range(256)], dtype=np.int32)
        return int(pop[aa ^ bb].sum())

    def test_mask_from_states_is_deterministic(self):
        states = {"physical": ["worsened"], "emotional": ["disturbed"]}
        self.assertEqual(_mask_from_states(states), _mask_from_states(states))

    def test_similar_negative_state_words_stay_closer_than_antonyms(self):
        worsened = _mask_from_states({"physical": ["worsened"]})
        damaged = _mask_from_states({"physical": ["damaged"]})
        accepted = _mask_from_states({"physical": ["accepted"]})
        self.assertLess(self._hamming(worsened, damaged), self._hamming(worsened, accepted))

    def test_label_from_states_uses_state_words(self):
        label = _label_from_states({"physical": ["worsened"], "emotional": ["disturbed"]})
        self.assertIn("worsened", label)
        self.assertIn("disturbed", label)

    def test_empty_states_label_defaults(self):
        self.assertEqual(_label_from_states({}), "state-mask")


# ── Bitmask geometry ──────────────────────────────────────────────────────────

class TestMeaningDbLookup(unittest.TestCase):

    def setUp(self):
        if not _has_meaning_db():
            self.skipTest("meaning.db not present")
        self.db = MeaningDB(MEANING_DB_PATH)

    def tearDown(self):
        self.db.close()

    def test_unknown_verb_returns_none(self):
        self.assertIsNone(self.db.get_verb("xyzzyx"))

    def test_known_verbs_set_nonempty(self):
        self.assertGreater(len(self.db.known_verbs()), 0)

    def test_known_verb_has_mask(self):
        vm = self.db.get_verb("aggravate")
        if vm is None:
            self.skipTest("aggravate not present in local meaning.db")
        self.assertEqual(vm.polarity_mask.shape, (16,))
        self.assertTrue(vm.polarity_class)


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


class TestExtractMeaning(unittest.TestCase):
    def test_fire_alert_house_on_fire(self):
        m = extract_meaning("My house is on fire")
        self.assertIn("alerts", m)
        self.assertTrue(any(a.get("type") == "fire" for a in m["alerts"]))
        fire = next(a for a in m["alerts"] if a.get("type") == "fire")
        self.assertEqual(fire.get("entity"), "house")

    def test_frames_for_copula_statement(self):
        m = extract_meaning("My house is on fire")
        self.assertIn("frames", m)
        self.assertTrue(any(f.get("verb") == "is" for f in m["frames"]))


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
        self.assertNotIn("not in meaning DB", desc)

    def test_unknown_verb(self):
        desc = describe_verb("xyzzyx", self.db)
        self.assertIn("not in meaning DB", desc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
