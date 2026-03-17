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
import random

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from srm.config   import CODE_BITS, PACK_BYTES, SAMPLE_KB, CHAT_KB, JS_KB, STOPWORDS
from srm.nlp      import tokenise, idf_table, tfidf_vec, cosine, expand_query
from srm.encoding import encode, encode_batch, hamming, hamming_batch
from srm.traversal import traverse
from srm.store    import MemoryStore
from srm.pipeline import srm_query, srm_query_cast_reconstruct, srm_query_auto


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

    def test_total_votes_equals_casts(self):
        texts = ["mitochondria produces ATP", "black hole gravity"]
        idf   = idf_table(texts)
        codes = encode_batch(texts, idf)
        q     = encode("mitochondria ATP", idf)
        votes, _ = traverse(q, codes, num_casts=30, noise=0.12,
                            rng=np.random.default_rng(123))
        self.assertEqual(sum(votes.values()), 30)

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
        votes, _   = traverse(q_code, self.codes, num_casts=30, noise=0.1,
                              rng=np.random.default_rng(123))
        self.assertEqual(votes.total(), 30)

    def test_relevant_memory_wins_more(self):
        # Memory 0: "The mitochondria is the powerhouse..."
        q_code   = encode("mitochondria ATP powerhouse", self.idf)
        votes, _ = traverse(q_code, self.codes, num_casts=40, noise=0.12,
                            rng=np.random.default_rng(123))
        # memory 0 should attract more probes
        self.assertGreater(votes[0], votes[1])


# ── Cast-level reconstruction (fragment KB) ───────────────────────────────────

class TestCastReconstruct(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = MemoryStore(self.tmp.name)

        # Tiny KB fragments (not full sentences on purpose)
        self.frags = [
            "mitochondria produce atp",
            "cellular respiration generates energy",
            "photosynthesis makes glucose",
            "entropy measures disorder",
        ]
        for f in self.frags:
            self.store.add(f)

    def tearDown(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def test_reconstruct_includes_multiple_unique_fragments(self):
        r = srm_query_cast_reconstruct(
            "how do cells produce energy",
            self.store,
            num_casts=4,
            noise=0.12,
            seed=7,
            max_words=80,
        )
        self.assertIn("response", r)
        self.assertIn("cast_outputs", r)
        self.assertEqual(len(r["cast_outputs"]), 4)
        self.assertGreaterEqual(len(r["selected_outputs"]), 2)

        # Response should contain at least 2 distinct fragment strings.
        hits = sum(1 for f in self.frags if f.split()[0] in r["response"])
        self.assertGreaterEqual(hits, 2)

    def test_reconstruct_is_deterministic_with_seed(self):
        r1 = srm_query_cast_reconstruct(
            "how do cells produce energy",
            self.store,
            num_casts=4,
            noise=0.12,
            seed=123,
        )
        r2 = srm_query_cast_reconstruct(
            "how do cells produce energy",
            self.store,
            num_casts=4,
            noise=0.12,
            seed=123,
        )
        self.assertEqual(r1["unique_cast_indices"], r2["unique_cast_indices"])


class TestAutoModeSelection(unittest.TestCase):
    def _sources_for(self, r: dict) -> list[str]:
        mode = r.get("auto_selected_mode")
        if mode == "reconstruct":
            return list(r.get("selected_outputs") or [])
        return [t for _, _, t in (r.get("top_attractors") or [])]

    def _assert_sources_overlap_expanded_query(self, q: str, r: dict) -> None:
        expanded = expand_query(q)
        q_terms = [t for t in tokenise(expanded) if t not in ("the", "and") and len(t) > 2]
        self.assertGreater(len(q_terms), 0, f"No content terms extracted from expanded query: {expanded!r}")

        srcs = "\n".join(self._sources_for(r)).lower()
        self.assertGreater(len(srcs.strip()), 0)
        self.assertTrue(
            any(t in srcs for t in q_terms),
            f"Expected at least one expanded query term in source memories. q={q!r} expanded={expanded!r} terms={q_terms!r} srcs={srcs!r}",
        )

    def test_auto_prefers_reconstruct_for_fragment_kb(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = MemoryStore(tmp.name)
        try:
            frags = [
                "bread food carbohydrate",
                "bread is baked from flour",
                "flour comes from wheat",
                "wheat is a grain",
            ]
            for f in frags:
                store.add(f)

            r = srm_query_auto(
                "bread wheat",
                store,
                num_casts=4,
                noise=0.14,
                seed=7,
                max_words=80,
            )
            self.assertIn("auto_selected_mode", r)
            self.assertIn(r["auto_selected_mode"], ("synth", "reconstruct"))
            self.assertEqual(r["auto_selected_mode"], "reconstruct")
            self.assertIn("auto_scores", r)
            self.assertGreaterEqual(r["auto_scores"]["reconstruct"], r["auto_scores"]["synth"])
        finally:
            store.close()
            os.unlink(tmp.name)

    def test_chat_kb_conversation_prompts(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = MemoryStore(tmp.name)
        try:
            for t in CHAT_KB:
                store.add(t)

            cases = [
                "I am bored",
                "What is the weather today",
                "How do you feel",
                "Who are you",
                "Who am I",
            ]

            for qi, q in enumerate(cases):
                r = srm_query_auto(q, store, num_casts=16, noise=0.12, seed=101 + qi)
                self.assertIn("response", r)
                self.assertIn("auto_selected_mode", r)
                resp = r["response"].strip().lower()
                self.assertGreater(len(resp), 0)
                self._assert_sources_overlap_expanded_query(q, r)
        finally:
            store.close()
            os.unlink(tmp.name)

    def test_chat_kb_health_prompts(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = MemoryStore(tmp.name)
        try:
            for t in CHAT_KB:
                store.add(t)

            cases = [
                "I have headache",
                "My stomach aches",
            ]

            for qi, q in enumerate(cases):
                r = srm_query_auto(q, store, num_casts=18, noise=0.12, seed=202 + qi)
                self.assertIn("response", r)
                self.assertIn("auto_selected_mode", r)
                resp = r["response"].strip().lower()
                self.assertGreater(len(resp), 0)
                self._assert_sources_overlap_expanded_query(q, r)
        finally:
            store.close()
            os.unlink(tmp.name)

    def test_auto_chat_kb_prefers_synth(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = MemoryStore(tmp.name)
        try:
            chat_kb = [
                "Hello. How can I help you today?",
                "If you say a food name, you might mean you want to eat it.",
                "If you are requesting something, you can say: Could you give me bread?",
                "If there is an emergency like a fire, call local emergency services and get to safety.",
            ]
            for t in chat_kb:
                store.add(t)

            r = srm_query_auto("hello", store, num_casts=12, noise=0.12, seed=5)
            self.assertEqual(r["auto_selected_mode"], "synth")
            self.assertIn("hello", r["response"].lower())

            r2 = srm_query_auto("bread", store, num_casts=12, noise=0.12, seed=6)
            self.assertEqual(r2["auto_selected_mode"], "synth")
            self.assertIn("bread", r2["response"].lower())
        finally:
            store.close()
            os.unlink(tmp.name)

    def test_auto_prefers_synth_for_sentence_kb(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = MemoryStore(tmp.name)
        try:
            for t in SAMPLE_KB:
                store.add(t)

            r = srm_query_auto(
                "How does DNA replication work?",
                store,
                num_casts=30,
                noise=0.12,
                seed=11,
            )
            self.assertEqual(r["auto_selected_mode"], "synth")
        finally:
            store.close()
            os.unlink(tmp.name)

    def test_javascript_fragment_kb_returns_code_terms(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = MemoryStore(tmp.name)
        try:
            for t in JS_KB:
                store.add(t)

            r = srm_query_auto(
                "write javascript code to fetch json from an api",
                store,
                num_casts=18,
                noise=0.12,
                seed=29,
            )
            self.assertIn("response", r)
            self.assertIn(r["auto_selected_mode"], ("synth", "reconstruct"))
            resp = r["response"].lower()
            self.assertTrue(any(term in resp for term in ["fetch", "json", "response", "await", "error"]))
        finally:
            store.close()
            os.unlink(tmp.name)

    def test_auto_is_deterministic_with_seed(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = MemoryStore(tmp.name)
        try:
            for t in SAMPLE_KB:
                store.add(t)

            r1 = srm_query_auto("mitochondria ATP", store, num_casts=20, noise=0.12, seed=123)
            r2 = srm_query_auto("mitochondria ATP", store, num_casts=20, noise=0.12, seed=123)
            self.assertEqual(r1["auto_selected_mode"], r2["auto_selected_mode"])
            self.assertEqual(r1["response"], r2["response"])
        finally:
            store.close()
            os.unlink(tmp.name)


# ── Randomized but stable smoke tests ────────────────────────────────────────

class TestRandomSmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = MemoryStore(self.tmp.name)
        for t in SAMPLE_KB:
            self.store.add(t)

    def tearDown(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def test_many_queries_return_nonempty_responses(self):
        rng = random.Random(123)
        # Build a small query pool from content tokens.
        toks: list[str] = []
        for t in SAMPLE_KB:
            toks.extend([w for w in tokenise(t) if w not in ("the", "and") and len(w) > 3])
        self.assertGreater(len(toks), 10)

        for i in range(25):
            q = " ".join(rng.sample(toks, 3))
            r = srm_query(q, self.store, num_casts=20, noise=0.12, seed=i)
            self.assertIn("response", r)
            self.assertIsInstance(r["response"], str)
            self.assertGreater(len(r["response"].strip()), 0)


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
