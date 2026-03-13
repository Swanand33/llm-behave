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
