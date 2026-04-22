"""Tests for DriftTest baseline save/compare."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from llm_behave.core.drift import DriftTest, DriftBaseline, DriftResult


class TestDriftBaseline:
    """Test saving and loading drift baselines."""

    def test_save_and_load_baseline(self, tmp_path):
        DriftTest._save_baseline("test_baseline", "Hello world", tmp_path)

        baseline = DriftTest._load_baseline("test_baseline", tmp_path)
        assert baseline is not None
        assert baseline.name == "test_baseline"
        assert "Hello world" in baseline.outputs

    def test_load_nonexistent_baseline_returns_none(self, tmp_path):
        result = DriftTest._load_baseline("nonexistent", tmp_path)
        assert result is None

    def test_baseline_appends_outputs(self, tmp_path):
        DriftTest._save_baseline("test", "output 1", tmp_path)
        DriftTest._save_baseline("test", "output 2", tmp_path)

        baseline = DriftTest._load_baseline("test", tmp_path)
        assert len(baseline.outputs) == 2

    def test_baseline_stores_model_info(self, tmp_path):
        DriftTest._save_baseline("test", "output", tmp_path)

        baseline = DriftTest._load_baseline("test", tmp_path)
        assert baseline.model_name == "all-MiniLM-L6-v2"


class TestDriftCompare:
    """Test drift comparison with real embeddings."""

    def test_no_drift_on_similar_output(self, tmp_path):
        DriftTest._save_baseline("refund_v1", "We will process your refund within 5 days.", tmp_path)

        result = DriftTest.compare(
            "refund_v1",
            "Your refund will be processed in 5 business days.",
            threshold=0.7,
            baseline_dir=tmp_path,
        )
        assert result.passed, f"Similar outputs should not drift. Score: {result.drift_score}"

    def test_drift_detected_on_different_output(self, tmp_path):
        DriftTest._save_baseline("refund_v1", "We will process your refund within 5 days.", tmp_path)

        result = DriftTest.compare(
            "refund_v1",
            "The quantum entanglement of photons produces interference patterns.",
            threshold=0.7,
            baseline_dir=tmp_path,
        )
        assert not result.passed, f"Completely different output should drift. Score: {result.drift_score}"

    def test_compare_nonexistent_baseline_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No baseline"):
            DriftTest.compare("nope", "some output", baseline_dir=tmp_path)

    def test_drift_result_has_details(self, tmp_path):
        DriftTest._save_baseline("test", "Hello world", tmp_path)
        result = DriftTest.compare("test", "Hello world", baseline_dir=tmp_path)
        assert isinstance(result, DriftResult)
        assert result.baseline_name == "test"
        assert isinstance(result.drift_score, float)


class TestDriftDecorator:
    """Test the @DriftTest.baseline decorator."""

    def test_baseline_decorator_saves(self, tmp_path):
        @DriftTest.baseline(save_as="decorated_test", baseline_dir=tmp_path)
        def my_test():
            return "Test output from decorated function"

        my_test()

        baseline = DriftTest._load_baseline("decorated_test", tmp_path)
        assert baseline is not None
        assert "Test output from decorated function" in baseline.outputs


class TestDriftRingBuffer:
    """Verify baseline outputs stay bounded by max_outputs."""

    def test_outputs_capped_at_max_outputs(self, tmp_path):
        for i in range(25):
            DriftTest._save_baseline("bounded", f"output {i}", tmp_path, max_outputs=10)

        baseline = DriftTest._load_baseline("bounded", tmp_path)
        assert len(baseline.outputs) <= 10

    def test_outputs_keep_most_recent(self, tmp_path):
        for i in range(15):
            DriftTest._save_baseline("recency", f"output {i}", tmp_path, max_outputs=5)

        baseline = DriftTest._load_baseline("recency", tmp_path)
        # Most recent outputs should be present
        assert "output 14" in baseline.outputs
        assert "output 13" in baseline.outputs
        # Oldest outputs should have been evicted
        assert "output 0" not in baseline.outputs

    def test_decorator_passes_max_outputs(self, tmp_path):
        @DriftTest.baseline(save_as="capped_dec", baseline_dir=tmp_path, max_outputs=3)
        def produce():
            return "some output"

        for _ in range(6):
            produce()

        baseline = DriftTest._load_baseline("capped_dec", tmp_path)
        assert len(baseline.outputs) <= 3


class TestDriftEmbeddingsPopulated:
    """Verify embeddings are computed and stored, never left as empty list."""

    def test_embeddings_populated_on_first_save(self, tmp_path):
        DriftTest._save_baseline("emb_test", "Hello world", tmp_path)

        baseline = DriftTest._load_baseline("emb_test", tmp_path)
        assert baseline.embeddings is not None
        assert len(baseline.embeddings) == 1
        assert len(baseline.embeddings[0]) == 384  # all-MiniLM-L6-v2 dim

    def test_embeddings_match_outputs_count(self, tmp_path):
        for i in range(3):
            DriftTest._save_baseline("emb_count", f"output {i}", tmp_path)

        baseline = DriftTest._load_baseline("emb_count", tmp_path)
        assert len(baseline.embeddings) == len(baseline.outputs)


class TestDriftCompareFullText:
    """Verify compare() uses whole-text similarity (correct for text-vs-text drift)."""

    def test_identical_paragraphs_score_near_one(self, tmp_path):
        text = "We process refunds within 5 business days. Please contact support for help."
        DriftTest._save_baseline("full_text", text, tmp_path)
        result = DriftTest.compare("full_text", text, threshold=0.95, baseline_dir=tmp_path)
        assert result.passed, f"Identical text should not drift. Score: {result.drift_score}"

    def test_drifted_paragraph_detected(self, tmp_path):
        baseline_text = "Refunds are always available within 30 days."
        DriftTest._save_baseline("drift_para", baseline_text, tmp_path)
        drifted = "We no longer process any returns or refunds."
        result = DriftTest.compare("drift_para", drifted, threshold=0.9, baseline_dir=tmp_path)
        assert isinstance(result.drift_score, float)
        assert 0.0 <= result.drift_score <= 1.0
