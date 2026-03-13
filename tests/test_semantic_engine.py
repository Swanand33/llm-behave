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
