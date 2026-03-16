"""
tests/test_srm.py — integration + unit tests for SRM.

Run:
    python -m pytest tests/ -v
    # or without pytest:
    python tests/test_srm.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from srm.config   import CODE_BITS, PACK_BYTES, SAMPLE_KB
from srm.nlp      import tokenise, idf_table, tfidf_vec, cosine, expand_query
from srm.encoding import encode, encode_batch, hamming, hamming_batch
from srm.traversal import traverse
from srm.store    import MemoryStore
from srm.pipeline import srm_query


# ── NLP ───────────────────────────────────────────────────────────────────────

class TestTokenise(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(tokenise("Hello World"), ["hello", "world"])

    def test_strips_short(self):
        # single-char tokens excluded
        self.assertNotIn("a", tokenise("a cat sat"))

    def test_numbers_excluded(self):
        # digits-only fragments: [a-z]+ won't match pure digits
        result = tokenise("test 123 abc")
        self.assertIn("test", result)
        self.assertIn("abc", result)
        self.assertNotIn("123", result)


class TestIDF(unittest.TestCase):
    def test_rare_term_higher_idf(self):
        corpus = ["apple banana", "apple cherry", "apple date"]
        idf = idf_table(corpus)
        # "apple" appears in all 3 docs; "banana" in only 1 → banana has higher IDF
        self.assertGreater(idf.get("banana", 0), idf.get("apple", 0))

    def test_empty_corpus(self):
        self.assertEqual(idf_table([]), {})


class TestCosine(unittest.TestCase):
    def test_identical(self):
        v = {"a": 1.0, "b": 0.5}
        self.assertAlmostEqual(cosine(v, v), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(cosine({"a": 1.0}, {"b": 1.0}), 0.0)

    def test_empty(self):
        self.assertAlmostEqual(cosine({}, {"a": 1.0}), 0.0)


class TestExpandQuery(unittest.TestCase):
    def test_expansion_fires(self):
        expanded = expand_query("brain")
        self.assertIn("neuron", expanded)

    def test_no_duplicates(self):
        expanded = expand_query("brain learn")
        tokens = expanded.split()
        self.assertEqual(len(tokens), len(set(tokens)),
                         "Expanded query contains duplicate tokens")

    def test_unknown_term_unchanged(self):
        self.assertEqual(expand_query("xyzzyx"), "xyzzyx")


# ── Encoding ──────────────────────────────────────────────────────────────────

class TestEncode(unittest.TestCase):
    def test_shape(self):
        code = encode("test sentence")
        self.assertEqual(code.shape, (PACK_BYTES,))
        self.assertEqual(code.dtype, np.uint8)

    def test_deterministic(self):
        c1 = encode("hello world")
        c2 = encode("hello world")
        np.testing.assert_array_equal(c1, c2)

    def test_different_texts_differ(self):
        c1 = encode("mitochondria powerhouse cell")
        c2 = encode("neural network gradient descent")
        self.assertGreater(hamming(c1, c2), 0)

    def test_similar_texts_closer(self):
        base     = encode("The mitochondria produces ATP")
        similar  = encode("The mitochondria is the powerhouse of the cell")
        dissimilar = encode("Black holes warp spacetime near the event horizon")
        d_similar    = hamming(base, similar)
        d_dissimilar = hamming(base, dissimilar)
        self.assertLess(d_similar, d_dissimilar)


class TestHamming(unittest.TestCase):
    def test_self_distance_is_zero(self):
        c = encode("anything")
        self.assertEqual(hamming(c, c), 0)

    def test_batch_shape(self):
        texts = ["one", "two", "three"]
        idf   = idf_table(texts)
        codes = encode_batch(texts, idf)
        q     = encode("one", idf)
        dists = hamming_batch(q, codes)
        self.assertEqual(dists.shape, (3,))

    def test_batch_self_is_minimum(self):
        texts = ["one", "two three four five"]
        idf   = idf_table(texts)
        codes = encode_batch(texts, idf)
        q     = encode("one", idf)
        dists = hamming_batch(q, codes)
        self.assertEqual(int(np.argmin(dists)), 0)


# ── Traversal ─────────────────────────────────────────────────────────────────

class TestTraverse(unittest.TestCase):
    def setUp(self):
        texts = SAMPLE_KB[:5]
        from srm.nlp import idf_table
        self.idf   = idf_table(texts)
        self.codes = encode_batch(texts, self.idf)

    def test_returns_votes_and_log(self):
        q_code        = encode("mitochondria ATP", self.idf)
        votes, log    = traverse(q_code, self.codes, num_casts=20, noise=0.1)
        self.assertIsInstance(votes.total(), int)
        self.assertEqual(len(log), 20)

    def test_total_votes_equals_casts(self):
        q_code     = encode("mitochondria", self.idf)
        votes, _   = traverse(q_code, self.codes, num_casts=30, noise=0.1)
        self.assertEqual(votes.total(), 30)

    def test_relevant_memory_wins_more(self):
        # Memory 0: "The mitochondria is the powerhouse..."
        q_code   = encode("mitochondria ATP powerhouse", self.idf)
        votes, _ = traverse(q_code, self.codes, num_casts=100, noise=0.08)
        top      = votes.most_common(1)[0][0]
        self.assertEqual(top, 0)


# ── MemoryStore ───────────────────────────────────────────────────────────────

class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = MemoryStore(self.tmp.name)

    def tearDown(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def test_add_and_count(self):
        self.store.add("First memory.")
        self.store.add("Second memory.")
        self.assertEqual(self.store.count(), 2)

    def test_duplicate_rejected(self):
        self.store.add("Duplicate.")
        ok = self.store.add("Duplicate.")
        self.assertFalse(ok)
        self.assertEqual(self.store.count(), 1)

    def test_delete(self):
        self.store.add("To be deleted.")
        ids, _ = self.store.load_all()
        deleted = self.store.delete(ids[0])
        self.assertTrue(deleted)
        self.assertEqual(self.store.count(), 0)

    def test_delete_nonexistent(self):
        self.assertFalse(self.store.delete(99999))

    def test_clear(self):
        for t in SAMPLE_KB[:5]:
            self.store.add(t)
        self.store.clear()
        self.assertEqual(self.store.count(), 0)

    def test_cache_invalidated_after_add(self):
        self.store.add("First.")
        _ = self.store.get_idf()          # populate cache
        self.store.add("Second.")
        # cache should be flushed → idf recomputed
        self.assertIsNone(self.store._idf)

    def test_codes_shape(self):
        for t in SAMPLE_KB[:4]:
            self.store.add(t)
        codes = self.store.get_codes()
        self.assertEqual(codes.shape, (4, PACK_BYTES))


# ── End-to-end pipeline ───────────────────────────────────────────────────────

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = MemoryStore(self.tmp.name)
        for t in SAMPLE_KB:
            self.store.add(t)

    def tearDown(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def _query(self, q):
        return srm_query(q, self.store, num_casts=40, noise=0.12)

    def test_returns_dict_with_required_keys(self):
        r = self._query("mitochondria")
        for key in ("query", "response", "attractor_details",
                    "vote_distribution", "num_memories"):
            self.assertIn(key, r)

    def test_response_is_nonempty_string(self):
        r = self._query("how does DNA replication work")
        self.assertIsInstance(r["response"], str)
        self.assertGreater(len(r["response"]), 10)

    def test_correct_attractor_for_cell_energy(self):
        r = self._query("cellular energy production ATP")
        top_text = r["attractor_details"][0]["text"].lower()
        self.assertTrue(
            "mitochondria" in top_text or "atp" in top_text,
            f"Unexpected top attractor: {top_text!r}",
        )

    def test_correct_attractor_for_gravity(self):
        r = self._query("what is gravity and curved spacetime")
        texts = [a["text"].lower() for a in r["attractor_details"]]
        self.assertTrue(
            any("relativity" in t or "spacetime" in t for t in texts),
            f"No spacetime/relativity attractor found: {texts}",
        )

    def test_correct_attractor_for_neural_networks(self):
        r = self._query("how do machine learning models train")
        texts = [a["text"].lower() for a in r["attractor_details"]]
        self.assertTrue(
            any("neural" in t or "backprop" in t or "gradient" in t
                for t in texts),
            f"No neural/backprop attractor found: {texts}",
        )

    def test_empty_store_returns_error(self):
        tmp2 = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp2.close()
        empty = MemoryStore(tmp2.name)
        r = srm_query("anything", empty)
        self.assertIn("error", r)
        empty.close()
        os.unlink(tmp2.name)

    def test_num_memories_correct(self):
        r = self._query("test")
        self.assertEqual(r["num_memories"], len(SAMPLE_KB))

    def test_vote_distribution_sums_to_num_casts(self):
        r = self._query("DNA")
        total = sum(r["vote_distribution"].values())
        self.assertEqual(total, r["num_casts"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
