"""Drift edge cases that don't need the real embedding model (fake engine)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from llm_behave.core.drift import DriftTest


class FakeEngine:
    """Deterministic stand-in for SemanticEngine: embeds text as char-count features."""

    model_name = "fake-model"
    model_version = "fake-model"

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(8)
        for i, ch in enumerate(text):
            vec[i % 8] += ord(ch)
        return vec

    def similarity(self, a: str, b: str) -> float:
        ea, eb = self.encode(a), self.encode(b)
        return float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb)))


@pytest.fixture
def fake_engine(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(
        "llm_behave.engines.semantic.get_semantic_engine", lambda *a, **k: engine
    )
    return engine


class TestBaselineNames:
    @pytest.mark.parametrize(
        "name", ["../evil", "a/b", "a\\b", "..", ".hidden", "", "x/../y", "name\x00"]
    )
    def test_unsafe_names_rejected(self, name, tmp_path, fake_engine):
        with pytest.raises(ValueError, match="Invalid baseline name"):
            DriftTest._save_baseline(name, "out", tmp_path)
        with pytest.raises(ValueError, match="Invalid baseline name"):
            DriftTest._load_baseline(name, tmp_path)
        with pytest.raises(ValueError, match="Invalid baseline name"):
            DriftTest.compare(name, "out", baseline_dir=tmp_path)

    def test_decorator_rejects_unsafe_name_up_front(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid baseline name"):
            DriftTest.baseline(save_as="../escape", baseline_dir=tmp_path)

    @pytest.mark.parametrize("name", ["refund_v1", "support-tone", "v1.2", "A9"])
    def test_normal_names_accepted(self, name, tmp_path, fake_engine):
        DriftTest._save_baseline(name, "out", tmp_path)
        assert (tmp_path / f"{name}.json").exists()

    def test_nothing_written_outside_directory(self, tmp_path, fake_engine):
        target = tmp_path / "store"
        with pytest.raises(ValueError):
            DriftTest._save_baseline("../outside", "out", target)
        assert not (tmp_path / "outside.json").exists()


class TestCorruptBaseline:
    def test_invalid_json_gives_clear_error(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not json")
        with pytest.raises(ValueError, match="corrupt"):
            DriftTest._load_baseline("bad", tmp_path)

    def test_wrong_shape_gives_clear_error(self, tmp_path):
        (tmp_path / "bad.json").write_text(json.dumps({"name": "bad"}))
        with pytest.raises(ValueError, match="corrupt"):
            DriftTest._load_baseline("bad", tmp_path)


class TestMaxOutputs:
    def test_new_max_outputs_applies_to_existing_baseline(self, tmp_path, fake_engine):
        for i in range(5):
            DriftTest._save_baseline("ring", f"out {i}", tmp_path, max_outputs=10)
        DriftTest._save_baseline("ring", "out 5", tmp_path, max_outputs=3)

        baseline = DriftTest._load_baseline("ring", tmp_path)
        assert baseline.outputs == ["out 3", "out 4", "out 5"]
        assert len(baseline.embeddings) == 3
        assert baseline.max_outputs == 3


class TestAtomicWrite:
    def test_no_temp_files_left_behind(self, tmp_path, fake_engine):
        for i in range(3):
            DriftTest._save_baseline("clean", f"out {i}", tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["clean.json"]

    def test_failed_write_keeps_old_baseline(self, tmp_path, fake_engine, monkeypatch):
        DriftTest._save_baseline("keep", "original", tmp_path)

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("llm_behave.core.drift.os.replace", boom)
        with pytest.raises(OSError):
            DriftTest._save_baseline("keep", "second", tmp_path)

        monkeypatch.undo()
        assert DriftTest._load_baseline("keep", tmp_path).outputs == ["original"]
        assert [p.name for p in tmp_path.iterdir()] == ["keep.json"]


class TestDecorator:
    def test_preserves_function_metadata(self, tmp_path):
        @DriftTest.baseline(save_as="meta", baseline_dir=tmp_path)
        def my_documented_test():
            """Docstring survives."""
            return "x"

        assert my_documented_test.__name__ == "my_documented_test"
        assert my_documented_test.__doc__ == "Docstring survives."

    def test_appends_on_every_call(self, tmp_path, fake_engine):
        @DriftTest.baseline(save_as="every", baseline_dir=tmp_path)
        def run():
            return "same output"

        run()
        run()
        assert len(DriftTest._load_baseline("every", tmp_path).outputs) == 2
