"""Drift detection for LLM behavior over time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel


class DriftBaseline(BaseModel):
    """Stored baseline for drift comparison."""

    name: str
    model_name: str
    model_version: str
    outputs: list[str]
    embeddings: list[list[float]] = []
    max_outputs: int = 20
    metadata: dict[str, Any] = {}


class DriftResult(BaseModel):
    """Result of a drift comparison."""

    baseline_name: str
    drift_score: float
    threshold: float
    passed: bool
    details: str = ""


class DriftTest:
    """Drift detection for LLM outputs.

    Saves baseline behavior and compares future runs against it.
    Designed for CI pipelines to catch silent model regressions.
    """

    DEFAULT_DIR = Path(".llm_behave_baselines")

    @staticmethod
    def baseline(
        save_as: str,
        baseline_dir: str | Path | None = None,
        max_outputs: int = 20,
    ) -> Callable:
        """Decorator that saves test output as a baseline.

        Args:
            save_as: Name for this baseline.
            baseline_dir: Directory to store baselines. Defaults to .llm_behave_baselines/
            max_outputs: Maximum outputs to retain (ring buffer). Defaults to 20.
        """
        storage = Path(baseline_dir) if baseline_dir else DriftTest.DEFAULT_DIR

        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = func(*args, **kwargs)
                DriftTest._save_baseline(save_as, result, storage, max_outputs)
                return result

            wrapper._drift_baseline_name = save_as  # type: ignore[attr-defined]
            return wrapper

        return decorator

    @staticmethod
    def compare(
        baseline_name: str,
        current_output: str,
        threshold: float = 0.8,
        baseline_dir: str | Path | None = None,
    ) -> DriftResult:
        """Compare current output against a saved baseline.

        Args:
            baseline_name: Name of the saved baseline.
            current_output: Current LLM output to compare.
            threshold: Minimum similarity score (0-1). Below = drift detected.
            baseline_dir: Directory where baselines are stored.

        Returns:
            DriftResult with pass/fail and drift score.
        """
        storage = Path(baseline_dir) if baseline_dir else DriftTest.DEFAULT_DIR
        baseline = DriftTest._load_baseline(baseline_name, storage)

        if baseline is None:
            raise FileNotFoundError(
                f"No baseline '{baseline_name}' found in {storage}. "
                "Run the baseline test first to create it."
            )

        from llm_behave.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()

        scores = [engine.similarity(current_output, b) for b in baseline.outputs]
        max_score = max(scores) if scores else 0.0

        passed = max_score >= threshold
        return DriftResult(
            baseline_name=baseline_name,
            drift_score=max_score,
            threshold=threshold,
            passed=passed,
            details=f"Max similarity to baseline: {max_score:.3f}"
            if passed
            else f"Drift detected! Max similarity {max_score:.3f} < threshold {threshold}",
        )

    @staticmethod
    def _save_baseline(name: str, output: Any, directory: Path, max_outputs: int = 20) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / f"{name}.json"

        from llm_behave.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        text = str(output)
        existing = DriftTest._load_baseline(name, directory)

        if existing:
            existing.outputs.append(text)
            existing.outputs = existing.outputs[-existing.max_outputs:]
            existing.embeddings = [engine.encode(o).tolist() for o in existing.outputs]
            data = existing.model_dump()
        else:
            data = DriftBaseline(
                name=name,
                model_name=engine.model_name,
                model_version=engine.model_version,
                outputs=[text],
                embeddings=[engine.encode(text).tolist()],
                max_outputs=max_outputs,
            ).model_dump()

        filepath.write_text(json.dumps(data, indent=2))

    @staticmethod
    def _load_baseline(name: str, directory: Path) -> DriftBaseline | None:
        filepath = directory / f"{name}.json"
        if not filepath.exists():
            return None
        data = json.loads(filepath.read_text())
        return DriftBaseline(**data)
