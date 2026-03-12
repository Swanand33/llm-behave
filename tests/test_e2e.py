"""Phase 05 — Integration / E2E tests.

Tests that exercise multiple features together in realistic scenarios.
All tests use MockProvider — no real LLM calls, no API cost.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from llm_assert.core.assertions import assert_behavior, behavioral_test
from llm_assert.core.conversation import ConversationTest
from llm_assert.core.drift import DriftTest
from llm_assert.providers.base import MockProvider


# ---------------------------------------------------------------------------
# Scenario 1: Customer support bot — full assertion chain
# ---------------------------------------------------------------------------

class TestCustomerSupportE2E:
    """Realistic customer support scenario: user asks about refund."""

    def test_support_response_full_chain(self):
        """E2E: response passes mentions + tone + intent together."""
        provider = MockProvider(responses=[
            "I'm really sorry to hear about that. I completely understand your frustration. "
            "Let me look into your refund for order #1234 right away. "
            "I'll make sure this is resolved for you today."
        ])
        output = provider.chat([{"role": "user", "content": "I want a refund for order 1234"}])

        (
            assert_behavior(output)
            .mentions("refund")
            .mentions("order", threshold=0.4)
            .tone("empathetic", threshold=0.3)
            .tone("helpful", threshold=0.3)
            .intent("helping with a refund")
        )

    def test_tool_call_e2e(self):
        """E2E: bot calls a tool and text response is also assertable."""
        provider = MockProvider(
            responses=["Let me look up your order details."],
            tool_responses=[
                ("Let me look up your order details.", [{"name": "lookup_order", "arguments": {"order_id": "1234"}}])
            ],
        )
        text, tool_calls = provider.chat_with_tools(
            [{"role": "user", "content": "What's the status of order 1234?"}],
            tools=[{"name": "lookup_order", "description": "Look up an order"}],
        )

        assert_behavior(text, tool_calls).calls_tool("lookup_order").mentions("order")

    def test_refusal_response(self):
        """E2E: bot politely declines — correct tone + intent."""
        provider = MockProvider(responses=[
            "I'm sorry, but I'm unable to process that request due to our company policy. "
            "If you have any other questions, I'm happy to help."
        ])
        output = provider.chat([{"role": "user", "content": "Can you share user passwords?"}])

        (
            assert_behavior(output)
            .intent("declining a request politely")
            .not_mentions("password")
        )


# ---------------------------------------------------------------------------
# Scenario 2: Multi-turn conversation — memory + tone consistency
# ---------------------------------------------------------------------------

class TestMultiTurnE2E:
    """Multi-turn conversation with context recall and tone consistency."""

    def test_context_recall_across_turns(self):
        """E2E: bot remembers user name and order from earlier turns."""
        provider = MockProvider(responses=[
            "Hello Swanand! Nice to meet you. How can I help you today?",
            "Got it, I can see your order #5678. Let me check the status for you.",
            "Swanand, I've confirmed that order #5678 will arrive by Friday.",
        ])
        conv = ConversationTest(agent=provider.chat)

        conv.say("Hi, my name is Swanand")
        conv.say("I placed order #5678 last week")
        response = conv.say("When will my order arrive?")

        assert response.recalls("order", threshold=0.4)
        assert response.recalls("Friday", threshold=0.4)

    def test_tone_consistency_across_turns(self):
        """E2E: professional tone is maintained across all turns."""
        provider = MockProvider(responses=[
            "Thank you for contacting us. How may I assist you today?",
            "I understand. Let me review your account details.",
            "Your account has been updated. Is there anything else I can help with?",
        ])
        conv = ConversationTest(agent=provider.chat)

        conv.say("I need help with my account")
        conv.say("My email changed")
        response = conv.say("Is that all done?")

        assert response.consistent_tone_across_turns(threshold=0.5)

    def test_multi_turn_with_assertions_each_turn(self):
        """E2E: assert behavior on each individual turn response."""
        responses = [
            "Of course! I'd be happy to help you with your refund.",
            "I can see your order #9999. The refund will be processed within 5 business days.",
            "You're welcome! The refund for order #9999 has been submitted successfully.",
        ]
        provider = MockProvider(responses=responses)
        conv = ConversationTest(agent=provider.chat)

        r1 = conv.say("I need a refund")
        assert_behavior(r1.text).tone("helpful", threshold=0.3).intent("offering to help")

        r2 = conv.say("It's order 9999")
        assert_behavior(r2.text).mentions("refund").mentions("order")

        r3 = conv.say("Thank you")
        assert_behavior(r3.text).mentions("refund")


# ---------------------------------------------------------------------------
# Scenario 3: Drift detection + semantic assertions
# ---------------------------------------------------------------------------

class TestDriftWithSemanticsE2E:
    """Drift detection combined with semantic assertions."""

    def test_drift_baseline_then_assert(self):
        """E2E: save baseline output, then assert it semantically, then compare."""
        output = (
            "I'm happy to help you with your refund. "
            "Please allow 5-7 business days for processing."
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            DriftTest._save_baseline("support_refund", output, storage)

            # Semantic assertions on the same output
            assert_behavior(output).mentions("refund").tone("helpful", threshold=0.3)

            # Compare — same output should show no drift
            result = DriftTest.compare("support_refund", output, baseline_dir=storage)
            assert result.passed, f"Unexpected drift: {result}"

    def test_drift_detected_when_tone_changes(self):
        """E2E: drift is detected when response tone changes significantly."""
        original = "I'm so sorry for the trouble. I'll personally make sure this is fixed for you."
        drifted = "Your request has been logged. Processing time is 10 business days."

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            DriftTest._save_baseline("support_tone", original, storage)

            result = DriftTest.compare("support_tone", drifted, baseline_dir=storage)
            assert not result.passed, "Should detect drift when tone changes significantly"

    def test_no_drift_with_paraphrased_output(self):
        """E2E: similar paraphrased output does not trigger drift."""
        original = "Let me help you with your refund request right away."
        paraphrased = "I'll take care of your refund immediately."

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            DriftTest._save_baseline("support_paraphrase", original, storage)

            result = DriftTest.compare("support_paraphrase", paraphrased, threshold=0.7, baseline_dir=storage)
            assert result.passed, f"Should not drift on paraphrase, got: {result}"


# ---------------------------------------------------------------------------
# Scenario 4: pytest plugin fixtures
# ---------------------------------------------------------------------------

class TestPluginFixtures:
    """Tests that verify pytest plugin fixtures work correctly."""

    def test_assert_llm_fixture(self, assert_llm):
        """Plugin fixture: assert_llm() works like assert_behavior()."""
        output = "I'll help you process that refund right away."
        assert_llm(output).mentions("refund").intent("offering to help")

    def test_mock_provider_fixture(self, mock_provider):
        """Plugin fixture: mock_provider() creates a working MockProvider."""
        provider = mock_provider(responses=["Hello! How can I help you today?"])
        output = provider.chat([{"role": "user", "content": "Hi"}])
        assert output == "Hello! How can I help you today?"
        assert_behavior(output).tone("friendly", threshold=0.3)

    def test_conversation_fixture(self, conversation):
        """Plugin fixture: conversation() creates a working ConversationTest."""
        conv = conversation(responses=[
            "Hi there! I'm here to help.",
            "Sure, I can help with your refund.",
        ])
        r1 = conv.say("Hello")
        assert "help" in r1.text.lower()

        r2 = conv.say("I need a refund")
        assert_behavior(r2.text).mentions("refund")

    def test_behavioral_test_decorator_with_fixture(self, assert_llm):
        """Plugin + decorator: @behavioral_test wraps assertion errors correctly."""
        @behavioral_test
        def check_output():
            output = "We're unable to process refunds at this time."
            assert_llm(output).mentions("refund")

        check_output()  # Should pass without error


# ---------------------------------------------------------------------------
# Scenario 5: Cross-feature — tool calls + multi-turn + assertions
# ---------------------------------------------------------------------------

class TestCrossFeatureE2E:
    """Cross-feature tests that combine tools, conversation, and assertions."""

    def test_tool_call_then_conversation_assertion(self):
        """E2E: tool is called, then follow-up conversation is also asserted."""
        # Tool call phase
        tool_provider = MockProvider(
            tool_responses=[
                ("I found your order. Let me pull up the details.", [
                    {"name": "get_order", "arguments": {"order_id": "111"}}
                ])
            ],
        )
        text, tool_calls = tool_provider.chat_with_tools(
            [{"role": "user", "content": "What's my order status?"}],
            tools=[{"name": "get_order"}],
        )
        assert_behavior(text, tool_calls).calls_tool("get_order").mentions("order", threshold=0.4)

        # Conversation phase — separate provider with follow-up responses
        conv_provider = MockProvider(responses=[
            "Your order #111 has shipped and will arrive by Thursday.",
        ])
        conv = ConversationTest(agent=conv_provider.chat)
        response = conv.say("When will order 111 arrive?")
        assert_behavior(response.text).mentions("order", threshold=0.4).mentions("shipped", threshold=0.4)

    def test_not_mentions_competitor_full_flow(self):
        """E2E: full support flow never mentions a competitor."""
        responses = [
            "I'd be happy to help you find the right product for your needs.",
            "Based on your requirements, I recommend our Premium Plan.",
            "Great choice! Let me set that up for you right away.",
        ]
        provider = MockProvider(responses=responses)
        conv = ConversationTest(agent=provider.chat)

        for question in ["What plans do you offer?", "Which is best for me?", "I'll take it"]:
            response = conv.say(question)
            assert_behavior(response.text).not_mentions("competitor")
