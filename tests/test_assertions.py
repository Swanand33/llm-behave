"""Tests for core assertion primitives.

These tests use MockProvider -- no real LLM calls, no torch dependency.
They validate the assertion API, error messages, and fluent chaining.
"""

from __future__ import annotations

import pytest

from llm_behave.core.assertions import AssertBehavior, assert_behavior, behavioral_test


class TestAssertBehaviorConstruction:
    """Test that assert_behavior() creates objects correctly."""

    def test_creates_assert_behavior_from_string(self):
        ab = assert_behavior("Hello, how can I help you?")
        assert isinstance(ab, AssertBehavior)
        assert ab.text == "Hello, how can I help you?"

    def test_rejects_non_string_input(self):
        with pytest.raises(TypeError, match="expects a string"):
            assert_behavior(123)

    def test_rejects_none_input(self):
        with pytest.raises(TypeError, match="expects a string"):
            assert_behavior(None)

    def test_accepts_empty_string(self):
        ab = assert_behavior("")
        assert ab.text == ""

    def test_stores_tool_calls(self):
        tools = [{"name": "lookup_order", "arguments": {"id": "1234"}}]
        ab = assert_behavior("Looking up your order", tool_calls=tools)
        assert ab._tool_calls == tools


class TestCallsTool:
    """Test the calls_tool() assertion -- no ML models needed."""

    def test_passes_when_tool_was_called(self):
        tools = [{"name": "lookup_order", "arguments": {"id": "1234"}}]
        ab = assert_behavior("Looking up your order", tool_calls=tools)
        result = ab.calls_tool("lookup_order")
        assert result is ab  # fluent chaining

    def test_fails_when_tool_not_called(self):
        tools = [{"name": "search_products", "arguments": {}}]
        ab = assert_behavior("Here are the products", tool_calls=tools)
        with pytest.raises(AssertionError, match="did not call tool 'lookup_order'"):
            ab.calls_tool("lookup_order")

    def test_fails_with_no_tool_calls(self):
        ab = assert_behavior("Just a text response")
        with pytest.raises(AssertionError, match="did not call tool"):
            ab.calls_tool("any_tool")

    def test_works_with_openai_format(self):
        """OpenAI nests tool name under function.name."""
        tools = [{"function": {"name": "get_weather"}, "id": "call_123"}]
        ab = assert_behavior("Weather is sunny", tool_calls=tools)
        ab.calls_tool("get_weather")  # should not raise


class TestBehavioralTestDecorator:
    """Test the @behavioral_test decorator."""

    def test_decorator_preserves_function_name(self):
        @behavioral_test
        def my_test():
            pass

        assert my_test.__name__ == "my_test"

    def test_decorator_marks_function(self):
        @behavioral_test
        def my_test():
            pass

        assert my_test._is_behavioral_test is True

    def test_decorator_enhances_assertion_errors(self):
        @behavioral_test
        def failing_test():
            assert False, "something broke"

        with pytest.raises(AssertionError, match="llm-behave"):
            failing_test()

    def test_decorator_passes_through_on_success(self):
        @behavioral_test
        def passing_test():
            return 42

        assert passing_test() == 42


class TestAssertionResults:
    """Test that assertion results are tracked."""

    def test_calls_tool_tracks_result(self):
        tools = [{"name": "search", "arguments": {}}]
        ab = assert_behavior("results", tool_calls=tools)
        ab.calls_tool("search")
        assert len(ab.results) == 1
        assert ab.results[0].passed is True
        assert ab.results[0].assertion_type == "calls_tool"


class TestContradicts:
    """Test the contradicts() assertion using the NLI engine."""

    def test_contradiction_detected(self):
        reference = "Refunds are always available within 30 days."
        contradicting = "We never offer refunds under any circumstances."
        ab = assert_behavior(contradicting)
        # Should pass — the output clearly contradicts the reference
        ab.contradicts(reference, threshold=0.3)
        assert ab.results[-1].passed is True
        assert ab.results[-1].assertion_type == "contradicts"

    def test_non_contradiction_fails_assertion(self):
        reference = "Refunds are always available within 30 days."
        agreeing = "Yes, we offer refunds within 30 days of purchase."
        ab = assert_behavior(agreeing)
        with pytest.raises(AssertionError, match="does not contradict"):
            ab.contradicts(reference, threshold=0.9)

    def test_contradiction_score_is_float(self):
        ab = assert_behavior("We never give refunds.")
        ab._results.clear()
        try:
            ab.contradicts("Refunds are always available.", threshold=0.0)
        except AssertionError:
            pass
        if ab.results:
            assert isinstance(ab.results[-1].similarity_score, float)

    def test_contradicts_result_tracked(self):
        reference = "The sky is blue."
        contradiction = "The sky is not blue, it is red."
        ab = assert_behavior(contradiction)
        try:
            ab.contradicts(reference, threshold=0.0)
        except AssertionError:
            pass
        result = next((r for r in ab.results if r.assertion_type == "contradicts"), None)
        assert result is not None
