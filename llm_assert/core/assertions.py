"""Core assertion primitives for behavioral testing of LLM outputs."""

from __future__ import annotations

import functools
from typing import Any, Callable

from pydantic import BaseModel


class AssertionResult(BaseModel):
    """Result of a single behavioral assertion."""

    passed: bool
    assertion_type: str
    expected: str
    actual_summary: str
    similarity_score: float | None = None
    message: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        msg = f"[{status}] {self.assertion_type}: expected={self.expected!r}"
        if self.similarity_score is not None:
            msg += f" (score={self.similarity_score:.3f})"
        if not self.passed and self.message:
            msg += f" -- {self.message}"
        return msg


class AssertBehavior:
    """Fluent assertion interface for LLM output behavior.

    Usage:
        assert_behavior(output).mentions("refund policy")
        assert_behavior(output).tone("empathetic")
        assert_behavior(output).not_mentions("competitor name")
    """

    def __init__(self, text: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        if not isinstance(text, str):
            raise TypeError(
                f"assert_behavior() expects a string, got {type(text).__name__}. "
                "Pass the text content of your LLM response."
            )
        self._text = text
        self._tool_calls = tool_calls or []
        self._results: list[AssertionResult] = []

    @property
    def text(self) -> str:
        return self._text

    @property
    def results(self) -> list[AssertionResult]:
        return list(self._results)

    def mentions(self, concept: str, threshold: float = 0.45) -> AssertBehavior:
        """Assert that the output semantically mentions a concept.

        Uses embedding similarity, not exact string matching.
        """
        from llm_assert.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        score = engine.max_sentence_similarity(self._text, concept)
        passed = score >= threshold

        result = AssertionResult(
            passed=passed,
            assertion_type="mentions",
            expected=concept,
            actual_summary=self._text[:100],
            similarity_score=score,
            message=f"Semantic similarity {score:.3f} below threshold {threshold}"
            if not passed
            else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM output does not mention '{concept}'. "
            f"Semantic similarity: {score:.3f} (threshold: {threshold})\n"
            f"Output: {self._text[:200]}"
        )
        return self

    def not_mentions(self, concept: str, threshold: float = 0.45) -> AssertBehavior:
        """Assert that the output does NOT semantically mention a concept."""
        from llm_assert.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        score = engine.max_sentence_similarity(self._text, concept)
        passed = score < threshold

        result = AssertionResult(
            passed=passed,
            assertion_type="not_mentions",
            expected=f"NOT {concept}",
            actual_summary=self._text[:100],
            similarity_score=score,
            message=f"Semantic similarity {score:.3f} above threshold {threshold}"
            if not passed
            else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM output unexpectedly mentions '{concept}'. "
            f"Semantic similarity: {score:.3f} (threshold: {threshold})\n"
            f"Output: {self._text[:200]}"
        )
        return self

    def tone(self, expected_tone: str, threshold: float = 0.5) -> AssertBehavior:
        """Assert that the output has a specific tone.

        Uses embedding similarity between the output and tone descriptions.
        Supported tones: empathetic, professional, friendly, formal, casual, rude, etc.
        """
        from llm_assert.engines.tone import get_tone_engine

        engine = get_tone_engine()
        score = engine.check_tone(self._text, expected_tone)
        passed = score >= threshold

        result = AssertionResult(
            passed=passed,
            assertion_type="tone",
            expected=expected_tone,
            actual_summary=self._text[:100],
            similarity_score=score,
            message=f"Tone similarity {score:.3f} below threshold {threshold}"
            if not passed
            else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM output tone is not '{expected_tone}'. "
            f"Tone similarity: {score:.3f} (threshold: {threshold})\n"
            f"Output: {self._text[:200]}"
        )
        return self

    def intent(self, expected_intent: str, threshold: float = 0.3) -> AssertBehavior:
        """Assert that the output's intent matches the expected description."""
        from llm_assert.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        score = engine.max_sentence_similarity(self._text, expected_intent)
        passed = score >= threshold

        result = AssertionResult(
            passed=passed,
            assertion_type="intent",
            expected=expected_intent,
            actual_summary=self._text[:100],
            similarity_score=score,
            message=f"Intent similarity {score:.3f} below threshold {threshold}"
            if not passed
            else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM output intent does not match '{expected_intent}'. "
            f"Similarity: {score:.3f} (threshold: {threshold})\n"
            f"Output: {self._text[:200]}"
        )
        return self

    def calls_tool(self, tool_name: str) -> AssertBehavior:
        """Assert that the LLM called a specific tool."""
        called_names = [tc.get("name", tc.get("function", {}).get("name", "")) for tc in self._tool_calls]
        passed = tool_name in called_names

        result = AssertionResult(
            passed=passed,
            assertion_type="calls_tool",
            expected=tool_name,
            actual_summary=f"Called tools: {called_names}",
            message=f"Tool '{tool_name}' not found in calls: {called_names}"
            if not passed
            else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM did not call tool '{tool_name}'. "
            f"Tools called: {called_names}"
        )
        return self


def assert_behavior(text: str, tool_calls: list[dict[str, Any]] | None = None) -> AssertBehavior:
    """Create a behavioral assertion on LLM output.

    Args:
        text: The text output from an LLM.
        tool_calls: Optional list of tool call dicts from the LLM response.

    Returns:
        AssertBehavior instance for fluent chaining.
    """
    return AssertBehavior(text, tool_calls)


def behavioral_test(func: Callable) -> Callable:
    """Decorator that marks a function as a behavioral test.

    Provides enhanced error reporting for behavioral assertions.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except AssertionError as e:
            raise AssertionError(f"[llm-assert] Behavioral test failed: {e}") from e

    wrapper._is_behavioral_test = True  # type: ignore[attr-defined]
    return wrapper
