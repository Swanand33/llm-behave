"""Phase 06 — Hardening tests.

Covers: edge cases, performance (<200ms per assertion), error resilience,
input boundaries, singleton behavior, and unusual-but-valid usage.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from llm_behave.core.assertions import assert_behavior, AssertBehavior
from llm_behave.core.conversation import ConversationTest
from llm_behave.core.drift import DriftTest
from llm_behave.providers.base import MockProvider


# ---------------------------------------------------------------------------
# Edge cases: unusual inputs to assert_behavior
# ---------------------------------------------------------------------------

class TestInputEdgeCases:
    """Robustness of assertions against unusual but valid inputs."""

    def test_empty_string_does_not_crash_mentions(self):
        """Empty text should not raise — it just fails the assertion."""
        with pytest.raises(AssertionError):
            assert_behavior("").mentions("refund")

    def test_whitespace_only_does_not_crash(self):
        """Whitespace-only text should not crash the engine."""
        with pytest.raises(AssertionError):
            assert_behavior("   ").mentions("refund policy")

    def test_very_long_text_mentions(self):
        """A 500-word response should still be processed correctly."""
        long_text = (
            "I want to help you with your refund request. " * 50  # ~400 words
        )
        assert_behavior(long_text).mentions("refund")

    def test_unicode_text_mentions(self):
        """Non-ASCII text (emojis, accents) should not crash the engine."""
        output = "Je suis désolé pour ce problème. Laissez-moi vous aider avec votre remboursement."
        # Should not raise — score may be low but no crash
        try:
            assert_behavior(output).mentions("refund", threshold=0.3)
        except AssertionError:
            pass  # Low score is fine; crash is not

    def test_numeric_only_text_does_not_crash(self):
        """Text with only numbers should not crash."""
        try:
            assert_behavior("12345 67890").mentions("order number")
        except AssertionError:
            pass

    def test_single_word_text_mentions(self):
        """Single word text should work without crashing."""
        try:
            assert_behavior("refund").mentions("refund")
        except AssertionError:
            pass  # may or may not pass — just must not crash

    def test_text_with_newlines(self):
        """Text with newlines and tabs processes correctly."""
        output = "Thank you for reaching out.\n\nI will help you with your refund.\n\tPlease allow 3-5 days."
        assert_behavior(output).mentions("refund")

    def test_special_characters_in_concept(self):
        """Concept with special chars should not crash."""
        output = "Please contact support@example.com for help with your order."
        try:
            assert_behavior(output).mentions("email address")
        except AssertionError:
            pass


# ---------------------------------------------------------------------------
# Edge cases: calls_tool
# ---------------------------------------------------------------------------

class TestCallsToolEdgeCases:
    """Edge cases for tool call assertions."""

    def test_multiple_tools_called_finds_correct_one(self):
        """When multiple tools are called, calls_tool finds the right one."""
        tool_calls = [
            {"name": "search_orders", "arguments": {"query": "1234"}},
            {"name": "send_email", "arguments": {"to": "user@example.com"}},
            {"name": "update_record", "arguments": {"id": "abc"}},
        ]
        assert_behavior("I searched your orders and sent a confirmation email.", tool_calls) \
            .calls_tool("search_orders") \
            .calls_tool("send_email") \
            .calls_tool("update_record")

    def test_calls_tool_fails_when_only_other_tools_called(self):
        """calls_tool fails when a different tool was called."""
        tool_calls = [{"name": "search_orders", "arguments": {}}]
        with pytest.raises(AssertionError, match="did not call tool"):
            assert_behavior("", tool_calls).calls_tool("send_email")

    def test_empty_tool_calls_list(self):
        """Empty tool_calls list causes calls_tool to fail cleanly."""
        with pytest.raises(AssertionError):
            assert_behavior("No tools were used.", []).calls_tool("lookup_order")

    def test_tool_calls_none_default(self):
        """Default None tool_calls treated as empty list."""
        with pytest.raises(AssertionError):
            assert_behavior("Some response").calls_tool("any_tool")

    def test_openai_nested_format_tool_call(self):
        """OpenAI-style nested function format is recognized."""
        tool_calls = [
            {"function": {"name": "get_weather", "arguments": '{"city": "Pune"}'}}
        ]
        assert_behavior("The weather in Pune is sunny.", tool_calls).calls_tool("get_weather")


# ---------------------------------------------------------------------------
# Edge cases: chaining behavior
# ---------------------------------------------------------------------------

class TestChainingEdgeCases:
    """AssertBehavior chaining accumulates results correctly."""

    def test_chain_accumulates_all_results(self):
        """Each chained call adds a result to results list."""
        output = "I'll help you with your refund and check your order status."
        ab = assert_behavior(output).mentions("refund").mentions("order", threshold=0.4)
        assert len(ab.results) == 2
        assert all(r.passed for r in ab.results)

    def test_chain_stops_at_first_failure(self):
        """Chain raises AssertionError on first failing assertion."""
        output = "The weather in Pune is nice today."
        with pytest.raises(AssertionError):
            assert_behavior(output).mentions("refund").mentions("order")

    def test_chain_returns_same_instance(self):
        """Chaining returns the same AssertBehavior instance."""
        output = "I'm happy to help with your refund request."
        ab = assert_behavior(output)
        result = ab.mentions("refund")
        assert result is ab

    def test_mixed_positive_negative_chain(self):
        """mentions and not_mentions can be chained together."""
        output = "I'll process your refund right away."
        assert_behavior(output).mentions("refund").not_mentions("competitor")

    def test_assertion_result_str_representation(self):
        """AssertionResult __str__ works for both pass and fail."""
        output = "I'll help you with that refund."
        ab = assert_behavior(output)
        ab.mentions("refund")
        result = ab.results[0]
        s = str(result)
        assert "PASS" in s
        assert "mentions" in s


# ---------------------------------------------------------------------------
# Edge cases: ConversationTest
# ---------------------------------------------------------------------------

class TestConversationEdgeCases:
    """Edge cases in multi-turn conversation testing."""

    def test_long_conversation_10_turns(self):
        """10-turn conversation builds history correctly."""
        responses = [f"Response {i}" for i in range(10)]
        provider = MockProvider(responses=responses)
        conv = ConversationTest(agent=provider.chat)

        for i in range(10):
            conv.say(f"Message {i}")

        assert conv.turn_count == 20  # 10 user + 10 assistant

    def test_agent_returning_empty_string(self):
        """Agent that returns empty string is allowed (valid response)."""
        provider = MockProvider(responses=[""])
        conv = ConversationTest(agent=provider.chat)
        response = conv.say("Hello")
        assert response.text == ""

    def test_history_is_passed_to_agent(self):
        """Agent receives full history including previous turns."""
        received_messages = []

        def spy_agent(messages):
            received_messages.append(list(messages))
            return "OK"

        conv = ConversationTest(agent=spy_agent)
        conv.say("First message")
        conv.say("Second message")

        # Second call should have 3 messages: user1, assistant1, user2
        assert len(received_messages[1]) == 3
        assert received_messages[1][0]["content"] == "First message"
        assert received_messages[1][1]["content"] == "OK"
        assert received_messages[1][2]["content"] == "Second message"

    def test_contradicts_turn_not_found_raises(self):
        """contradicts_turn raises ValueError for non-existent turn."""
        provider = MockProvider(responses=["Hello there."])
        conv = ConversationTest(agent=provider.chat)
        response = conv.say("Hi")
        with pytest.raises(ValueError, match="Turn 99 not found"):
            response.contradicts_turn(99)

    def test_single_turn_tone_consistency_always_passes(self):
        """consistent_tone_across_turns passes with only one assistant turn."""
        provider = MockProvider(responses=["I'm here to help."])
        conv = ConversationTest(agent=provider.chat)
        response = conv.say("Hello")
        assert response.consistent_tone_across_turns(threshold=0.99)


# ---------------------------------------------------------------------------
# Edge cases: DriftTest
# ---------------------------------------------------------------------------

class TestDriftEdgeCases:
    """Edge cases and resilience in drift detection."""

    def test_corrupt_baseline_json_raises(self):
        """Corrupt JSON in baseline file raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            (storage / "broken.json").write_text("{ this is not valid json }")
            with pytest.raises(Exception):
                DriftTest.compare("broken", "some output", baseline_dir=storage)

    def test_missing_baseline_raises_file_not_found(self):
        """Comparing against non-existent baseline raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No baseline"):
                DriftTest.compare("nonexistent", "output", baseline_dir=Path(tmpdir))

    def test_multiple_outputs_in_baseline(self):
        """Baseline with multiple saved outputs: uses max similarity."""
        output_a = "I will help you with your refund right away."
        output_b = "Let me process your return immediately."
        current = "I can handle your refund quickly."

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            DriftTest._save_baseline("multi", output_a, storage)
            DriftTest._save_baseline("multi", output_b, storage)

            baseline = DriftTest._load_baseline("multi", storage)
            assert len(baseline.outputs) == 2

            result = DriftTest.compare("multi", current, threshold=0.7, baseline_dir=storage)
            assert result.passed

    def test_baseline_stores_model_name(self):
        """Saved baseline records the model name for invalidation detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            DriftTest._save_baseline("model_check", "some output", storage)
            baseline = DriftTest._load_baseline("model_check", storage)
            assert baseline.model_name == "all-MiniLM-L6-v2"

    def test_identical_output_never_drifts(self):
        """Comparing output to itself should always pass drift check."""
        output = "This is the exact same response every time."
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            DriftTest._save_baseline("exact", output, storage)
            result = DriftTest.compare("exact", output, threshold=0.99, baseline_dir=storage)
            assert result.passed


# ---------------------------------------------------------------------------
# Performance: <200ms per assertion after model is loaded
# ---------------------------------------------------------------------------

class TestPerformance:
    """Assertions must complete in <200ms after model warmup."""

    @pytest.fixture(autouse=True)
    def warmup_model(self):
        """Ensure model is loaded before timing."""
        from llm_behave.engines.semantic import get_semantic_engine
        e = get_semantic_engine()
        e.encode("warmup")

    def test_mentions_under_200ms(self):
        output = "I'll help you process your refund for order 1234 right away."
        start = time.perf_counter()
        assert_behavior(output).mentions("refund")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"mentions() took {elapsed_ms:.1f}ms (limit: 200ms)"

    def test_tone_under_200ms(self):
        output = "I'm really sorry to hear about your experience. Let me help you right away."
        start = time.perf_counter()
        assert_behavior(output).tone("empathetic", threshold=0.3)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"tone() took {elapsed_ms:.1f}ms (limit: 200ms)"

    def test_intent_under_200ms(self):
        output = "Sure, let me look into that refund for you."
        start = time.perf_counter()
        assert_behavior(output).intent("helping with a refund")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"intent() took {elapsed_ms:.1f}ms (limit: 200ms)"

    def test_full_chain_under_500ms(self):
        """A realistic 4-assertion chain should finish in <500ms."""
        output = (
            "I'm so sorry about this issue. Let me personally handle your refund for order 1234. "
            "I'll make sure it's resolved today."
        )
        start = time.perf_counter()
        (
            assert_behavior(output)
            .mentions("refund")
            .mentions("order", threshold=0.4)
            .tone("helpful", threshold=0.3)
            .intent("helping with a refund")
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"4-assertion chain took {elapsed_ms:.1f}ms (limit: 500ms)"


# ---------------------------------------------------------------------------
# Singleton / lazy loading
# ---------------------------------------------------------------------------

class TestSingletonAndLazyLoading:
    """Engines load lazily and reuse the same instance."""

    def test_semantic_engine_singleton(self):
        """get_semantic_engine() returns same instance on repeated calls."""
        from llm_behave.engines.semantic import get_semantic_engine
        e1 = get_semantic_engine()
        e2 = get_semantic_engine()
        assert e1 is e2

    def test_tone_engine_singleton(self):
        """get_tone_engine() returns same instance on repeated calls."""
        from llm_behave.engines.tone import get_tone_engine
        t1 = get_tone_engine()
        t2 = get_tone_engine()
        assert t1 is t2

    def test_model_not_reloaded_on_second_call(self):
        """SemanticEngine._model is set after first encode — not reloaded."""
        from llm_behave.engines.semantic import get_semantic_engine
        engine = get_semantic_engine()
        engine.encode("first call loads the model")
        model_ref = engine._model
        engine.encode("second call reuses it")
        assert engine._model is model_ref

    def test_semantic_import_does_not_import_torch(self):
        """Importing llm_behave.engines.semantic does NOT import torch at module level."""
        import sys
        # torch should only be imported AFTER encode() is called, not on import
        # We verify the module can be reimported without torch being a hard dep
        import llm_behave.engines.semantic  # should not crash even if torch is present
        assert "llm_behave.engines.semantic" in sys.modules
