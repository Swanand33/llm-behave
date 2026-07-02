"""Tests for the semantic similarity engine.

These tests require sentence-transformers installed.
They validate that the embedding model produces meaningful similarity scores.
"""

from __future__ import annotations

import pytest

from llm_behave.engines.semantic import SemanticEngine, get_semantic_engine


@pytest.fixture(scope="module")
def engine():
    """Load engine once for all tests in this module (model loading is slow)."""
    return get_semantic_engine()


class TestSemanticEngine:
    """Test that the semantic engine produces correct similarity scores."""

    def test_identical_texts_high_similarity(self, engine):
        score = engine.similarity("I love pizza", "I love pizza")
        assert score > 0.95, f"Identical texts should score >0.95, got {score}"

    def test_similar_texts_moderate_similarity(self, engine):
        score = engine.similarity(
            "I want to return this product for a refund",
            "refund policy",
        )
        assert score > 0.3, f"Related texts should score >0.3, got {score}"

    def test_unrelated_texts_low_similarity(self, engine):
        score = engine.similarity(
            "The weather is beautiful today",
            "quantum physics equations",
        )
        assert score < 0.3, f"Unrelated texts should score <0.3, got {score}"

    def test_encoding_returns_numpy_array(self, engine):
        import numpy as np
        emb = engine.encode("test text")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)  # all-MiniLM-L6-v2 output dim

    def test_empty_string_does_not_crash(self, engine):
        score = engine.similarity("", "hello")
        assert isinstance(score, float)

    def test_long_text_works(self, engine):
        long_text = "This is a test. " * 200
        score = engine.similarity(long_text, "test")
        assert isinstance(score, float)

    def test_singleton_returns_same_instance(self):
        e1 = get_semantic_engine()
        e2 = get_semantic_engine()
        assert e1 is e2

    def test_model_name_exposed(self, engine):
        assert engine.model_name == "all-MiniLM-L6-v2"


class TestMultiModelCache:
    """Test that get_semantic_engine() caches multiple models independently."""

    def test_same_model_returns_same_instance(self):
        e1 = get_semantic_engine("all-MiniLM-L6-v2")
        e2 = get_semantic_engine("all-MiniLM-L6-v2")
        assert e1 is e2

    def test_different_model_names_return_different_instances(self):
        from llm_behave.engines.semantic import _engines

        # Two distinct names should produce two distinct entries in the cache
        e_default = get_semantic_engine("all-MiniLM-L6-v2")
        fake_name = "all-MiniLM-L6-v2-fake-test-model"
        # Manually insert a fake entry so we don't trigger a real download
        _engines[fake_name] = SemanticEngine(fake_name)

        assert _engines["all-MiniLM-L6-v2"] is e_default
        assert _engines[fake_name] is not e_default

        # Clean up the fake entry so other tests aren't affected
        del _engines[fake_name]

    def test_first_model_survives_second_model_load(self):
        from llm_behave.engines.semantic import _engines

        e1 = get_semantic_engine("all-MiniLM-L6-v2")
        fake = "all-MiniLM-L6-v2-second-fake"
        _engines[fake] = SemanticEngine(fake)

        # Original engine should still be in cache, not replaced
        assert get_semantic_engine("all-MiniLM-L6-v2") is e1
        del _engines[fake]


class TestTopSentenceSimilarities:
    """Test top-k sentence scoring used for failure diagnostics."""

    def test_returns_sorted_descending(self, engine):
        text = "I love pizza. The weather is nice. Refunds take 30 days."
        top = engine.top_sentence_similarities(text, "refund policy", k=3)
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True)

    def test_respects_k(self, engine):
        text = "One. Two. Three. Four."
        assert len(engine.top_sentence_similarities(text, "number", k=2)) == 2

    def test_best_sentence_is_most_relevant(self, engine):
        text = "The sky is blue. You can request a refund within 30 days."
        best_sentence, _ = engine.top_sentence_similarities(text, "refund", k=3)[0]
        assert "refund" in best_sentence

    def test_max_sentence_similarity_matches_top1(self, engine):
        text = "Hello there. Refunds take 30 days."
        top = engine.top_sentence_similarities(text, "refund", k=1)
        assert engine.max_sentence_similarity(text, "refund") == pytest.approx(top[0][1])

    def test_empty_text_returns_empty(self, engine):
        assert engine.top_sentence_similarities("", "refund") == []
