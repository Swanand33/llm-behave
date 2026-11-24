"""Drift detection for LLM behavior over time."""

from __future__ import annotations

import functools
import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

# Baseline names become file names, so they must not be able to escape the directory.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not _SAFE_NAME.match(name) or ".." in name:
        raise ValueError(
            f"Invalid baseline name {name!r}. Use letters, digits, '.', '_' or '-' "
            "(starting with a letter or digit); path separators are not allowed."
        )
    return name


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

    Every call of a ``@DriftTest.baseline`` function *appends* its output to the
    named baseline (a ring buffer capped at ``max_outputs``); it does not only
    record on the first run. ``compare`` scores a new output against the most
    similar stored output.
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
                If a baseline already exists, this value replaces its stored limit.
        """
        _validate_name(save_as)
        storage = Path(baseline_dir) if baseline_dir else DriftTest.DEFAULT_DIR

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
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
        _validate_name(baseline_name)
        storage = Path(baseline_dir) if baseline_dir else DriftTest.DEFAULT_DIR
        baseline = DriftTest._load_baseline(baseline_name, storage)

        if baseline is None:
            raise FileNotFoundError(
                f"No baseline '{baseline_name}' found in {storage}. "
                "Run the baseline test first to create it."
            )

        from llm_behave.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()

        if baseline.model_name != engine.model_name:
            raise ValueError(
                f"Baseline '{baseline_name}' was recorded with model "
                f"'{baseline.model_name}', but the current model is "
                f"'{engine.model_name}'. Scores are not comparable across models -- "
                "delete the baseline file and re-run the baseline test."
            )

        import numpy as np

        if len(baseline.embeddings) == len(baseline.outputs) and baseline.embeddings:
            # Reuse stored embeddings: only the current output needs encoding
            current = engine.encode(current_output)
            stored = np.asarray(baseline.embeddings)
            norms = np.linalg.norm(stored, axis=1) * np.linalg.norm(current)
            dots = stored @ current
            scores = np.divide(dots, norms, out=np.zeros_like(dots), where=norms != 0)
            max_score = float(scores.max())
        else:
            # Legacy baseline without embeddings
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
        _validate_name(name)
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / f"{name}.json"

        from llm_behave.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        text = str(output)
        existing = DriftTest._load_baseline(name, directory)

        if existing:
            if len(existing.embeddings) != len(existing.outputs):
                # Legacy baseline without aligned embeddings: rebuild them once
                existing.embeddings = [engine.encode(o).tolist() for o in existing.outputs]
            existing.max_outputs = max_outputs
            existing.outputs.append(text)
            existing.embeddings.append(engine.encode(text).tolist())
            existing.outputs = existing.outputs[-max_outputs:]
            existing.embeddings = existing.embeddings[-max_outputs:]
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

        # Write to a temp file then rename, so a crash can't leave a half-written baseline
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            os.replace(tmp_name, filepath)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @staticmethod
    def _load_baseline(name: str, directory: Path) -> DriftBaseline | None:
        _validate_name(name)
        filepath = directory / f"{name}.json"
        if not filepath.exists():
            return None
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            return DriftBaseline(**data)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise ValueError(
                f"Baseline file {filepath} is corrupt or not a valid baseline "
                f"({type(exc).__name__}). Delete it and re-run the baseline test."
            ) from exc
