"""Core assertion primitives for behavioral testing of LLM outputs."""

from __future__ import annotations

import functools
import json
import re
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

# Default thresholds are intentionally different per assertion type:
#   mentions / not_mentions — keyword-level signal, mid-range sensitivity
#   intent               — semantic purpose is broader, needs a looser gate
#   tone                 — style signal is subtle, needs a tighter gate
#   contradicts          — NLI raw score; 0.5 is mid-point of [0, 1]
_DEFAULT_MENTIONS = 0.45
_DEFAULT_NOT_MENTIONS = 0.45
_DEFAULT_INTENT = 0.30
_DEFAULT_TONE = 0.50
_DEFAULT_CONTRADICTS = 0.50
#   grounded_in / no_hallucination — NLI entailment probability; a claim is
#   considered supported when some context sentence entails it above this gate.
_DEFAULT_GROUNDED = 0.50


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty, trimmed sentences on ., !, ? boundaries."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences and text.strip():
        return [text.strip()]
    return sentences


def _format_top_matches(matches: list[tuple[str, float]]) -> str:
    """Render top-k (sentence, score) pairs for failure diagnostics.

    Answers the "was it 0.41 or 0.05?" question: shows which sentences were
    closest to the query and their exact scores, instead of a lone max.
    """
    if not matches:
        return ""
    lines = [f"    {score:.3f}  {sentence[:80]}" for sentence, score in matches]
    return "\n  Closest sentences:\n" + "\n".join(lines)


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

    def mentions(self, concept: str, threshold: float = _DEFAULT_MENTIONS) -> AssertBehavior:
        """Assert that the output semantically mentions a concept.

        Uses embedding similarity, not exact string matching.
        """
        from llm_behave.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        top = engine.top_sentence_similarities(self._text, concept, k=3)
        score = top[0][1] if top else engine.max_sentence_similarity(self._text, concept)
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
            f"Semantic similarity: {score:.3f} (threshold: {threshold})"
            f"{_format_top_matches(top)}\n"
            f"Output: {self._text[:200]}"
        )
        return self

    def not_mentions(self, concept: str, threshold: float = _DEFAULT_NOT_MENTIONS) -> AssertBehavior:
        """Assert that the output does NOT semantically mention a concept."""
        from llm_behave.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        top = engine.top_sentence_similarities(self._text, concept, k=3)
        score = top[0][1] if top else engine.max_sentence_similarity(self._text, concept)
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
            f"Semantic similarity: {score:.3f} (threshold: {threshold})"
            f"{_format_top_matches(top)}\n"
            f"Output: {self._text[:200]}"
        )
        return self

    def tone(self, expected_tone: str, threshold: float = _DEFAULT_TONE) -> AssertBehavior:
        """Assert that the output has a specific tone.

        Uses embedding similarity between the output and tone descriptions.
        Supported tones: empathetic, professional, friendly, formal, casual, rude, etc.
        """
        from llm_behave.engines.tone import get_tone_engine

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

    def intent(self, expected_intent: str, threshold: float = _DEFAULT_INTENT) -> AssertBehavior:
        """Assert that the output's intent matches the expected description."""
        from llm_behave.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        top = engine.top_sentence_similarities(self._text, expected_intent, k=3)
        score = top[0][1] if top else engine.max_sentence_similarity(self._text, expected_intent)
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
            f"Similarity: {score:.3f} (threshold: {threshold})"
            f"{_format_top_matches(top)}\n"
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

    def contradicts(self, other_text: str, threshold: float = _DEFAULT_CONTRADICTS) -> AssertBehavior:
        """Assert that this output contradicts the given text.

        Uses an offline NLI model. Useful for multi-turn consistency checks
        (e.g. the LLM reversed a policy it stated earlier).
        """
        from llm_behave.engines.contradiction import get_contradiction_engine

        engine = get_contradiction_engine()
        score = engine.check_contradiction(other_text, self._text)
        passed = score >= threshold

        result = AssertionResult(
            passed=passed,
            assertion_type="contradicts",
            expected=f"contradicts: {other_text[:60]}",
            actual_summary=self._text[:100],
            similarity_score=score,
            message=f"Contradiction score {score:.3f} below threshold {threshold}"
            if not passed
            else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM output does not contradict the reference text. "
            f"Contradiction score: {score:.3f} (threshold: {threshold})\n"
            f"Reference: {other_text[:200]}\n"
            f"Output:    {self._text[:200]}"
        )
        return self

    # ------------------------------------------------------------------
    # Structured-output assertions (pure Python, no ML model needed)
    # ------------------------------------------------------------------

    def _parse_json(self) -> Any:
        """Parse the output as JSON, tolerating ```json fenced code blocks.

        Returns the parsed object. Raises json.JSONDecodeError on failure.
        """
        raw = self._text.strip()
        if raw.startswith("```"):
            # Strip a leading ```json / ``` fence and the trailing ```
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        return json.loads(raw)

    def returns_valid_json(self) -> AssertBehavior:
        """Assert that the output is parseable as JSON.

        Tolerates ```json fenced blocks, which most LLMs emit.
        """
        try:
            self._parse_json()
            passed = True
            error = ""
        except (json.JSONDecodeError, ValueError) as exc:
            passed = False
            error = str(exc)

        result = AssertionResult(
            passed=passed,
            assertion_type="returns_valid_json",
            expected="valid JSON",
            actual_summary=self._text[:100],
            message=f"JSON parse error: {error}" if not passed else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM output is not valid JSON. Parse error: {error}\n"
            f"Output: {self._text[:200]}"
        )
        return self

    def matches_schema(self, model: type[BaseModel]) -> AssertBehavior:
        """Assert that the output JSON validates against a Pydantic model."""
        try:
            data = self._parse_json()
            model.model_validate(data)
            passed = True
            error = ""
        except (json.JSONDecodeError, ValueError) as exc:
            passed = False
            error = str(exc)
        except ValidationError as exc:
            passed = False
            error = str(exc)

        result = AssertionResult(
            passed=passed,
            assertion_type="matches_schema",
            expected=model.__name__,
            actual_summary=self._text[:100],
            message=f"Schema validation failed: {error}" if not passed else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM output does not match schema '{model.__name__}'.\n"
            f"{error}\n"
            f"Output: {self._text[:200]}"
        )
        return self

    def has_keys(self, keys: list[str]) -> AssertBehavior:
        """Assert that the output JSON is an object containing all given keys."""
        missing: list[str] = []
        error = ""
        try:
            data = self._parse_json()
            if not isinstance(data, dict):
                error = f"expected a JSON object, got {type(data).__name__}"
                passed = False
            else:
                missing = [k for k in keys if k not in data]
                passed = not missing
        except (json.JSONDecodeError, ValueError) as exc:
            passed = False
            error = f"invalid JSON: {exc}"

        detail = error or f"missing keys: {missing}"
        result = AssertionResult(
            passed=passed,
            assertion_type="has_keys",
            expected=f"keys: {keys}",
            actual_summary=self._text[:100],
            message=detail if not passed else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM output JSON is missing required keys. {detail}\n"
            f"Output: {self._text[:200]}"
        )
        return self

    # ------------------------------------------------------------------
    # RAG / grounding assertions (offline NLI entailment)
    # ------------------------------------------------------------------

    def _ground_claims(self, context: str) -> list[tuple[str, float, str]]:
        """Score each output sentence for entailment against the context.

        Returns (claim, best_entailment_score, supporting_context_sentence) for
        every sentence in the output. Shared by grounded_in / no_hallucination.
        """
        from llm_behave.engines.contradiction import get_contradiction_engine

        engine = get_contradiction_engine()
        context_sentences = _split_sentences(context)
        claims = _split_sentences(self._text)

        grounded: list[tuple[str, float, str]] = []
        for claim in claims:
            score, support = engine.max_entailment(claim, context_sentences)
            grounded.append((claim, score, support))
        return grounded

    def grounded_in(self, context: str, threshold: float = _DEFAULT_GROUNDED) -> AssertBehavior:
        """Assert every claim in the output is entailed by the given context.

        Uses offline NLI entailment (no judge LLM, no API). The output passes
        only if *all* of its sentences are supported by some context sentence
        above `threshold`. The weakest (least-grounded) claim drives the score.
        """
        grounded = self._ground_claims(context)
        weakest_score = min((score for _, score, _ in grounded), default=1.0)
        ungrounded = [(c, s) for c, s, _ in grounded if s < threshold]
        passed = not ungrounded

        detail = ""
        if not passed:
            lines = [f"    {s:.3f}  {c[:80]}" for c, s in ungrounded]
            detail = "\n  Ungrounded claims:\n" + "\n".join(lines)

        result = AssertionResult(
            passed=passed,
            assertion_type="grounded_in",
            expected=f"grounded in context ({len(context)} chars)",
            actual_summary=self._text[:100],
            similarity_score=weakest_score,
            message=f"{len(ungrounded)} claim(s) not grounded (min score {weakest_score:.3f})"
            if not passed
            else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM output contains claims not grounded in the context "
            f"(threshold: {threshold}).{detail}\n"
            f"Output: {self._text[:200]}"
        )
        return self

    def no_hallucination(self, context: str, threshold: float = _DEFAULT_GROUNDED) -> AssertBehavior:
        """Assert the output introduces no claims unsupported by the context.

        Inverse framing of grounded_in(): fails if any sentence is a
        hallucination — i.e. not entailed by any context sentence. Reports the
        specific hallucinated claims so a green check is trustworthy.
        """
        grounded = self._ground_claims(context)
        hallucinated = [(c, s) for c, s, _ in grounded if s < threshold]
        passed = not hallucinated
        weakest_score = min((score for _, score, _ in grounded), default=1.0)

        detail = ""
        if not passed:
            lines = [f"    {s:.3f}  {c[:80]}" for c, s in hallucinated]
            detail = "\n  Hallucinated claims (unsupported by context):\n" + "\n".join(lines)

        result = AssertionResult(
            passed=passed,
            assertion_type="no_hallucination",
            expected=f"no hallucinations vs context ({len(context)} chars)",
            actual_summary=self._text[:100],
            similarity_score=weakest_score,
            message=f"{len(hallucinated)} hallucinated claim(s) (min score {weakest_score:.3f})"
            if not passed
            else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM output hallucinates claims not supported by the context "
            f"(threshold: {threshold}).{detail}\n"
            f"Output: {self._text[:200]}"
        )
        return self

    def cites(self, source_id: str) -> AssertBehavior:
        """Assert that the output references a specific source identifier.

        A literal substring check (case-insensitive) — useful for asserting a
        RAG answer cites the expected doc id, URL, or `[3]`-style marker.
        """
        passed = source_id.lower() in self._text.lower()

        result = AssertionResult(
            passed=passed,
            assertion_type="cites",
            expected=source_id,
            actual_summary=self._text[:100],
            message=f"Source '{source_id}' not cited in output" if not passed else "",
        )
        self._results.append(result)

        assert passed, (
            f"LLM output does not cite source '{source_id}'.\n"
            f"Output: {self._text[:200]}"
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
            raise AssertionError(f"[llm-behave] Behavioral test failed: {e}") from e

    wrapper._is_behavioral_test = True  # type: ignore[attr-defined]
    return wrapper
