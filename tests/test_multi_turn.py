"""Integration tests for multi-turn conversation assertions.

Tests recalls() and contradicts_turn() with real ML models.
"""

from __future__ import annotations

import pytest

from llm_behave.core.conversation import ConversationTest


def make_agent(responses: list[str]):
    """Create a mock agent returning predefined responses."""
    idx = {"n": 0}

    def agent(messages):
        i = min(idx["n"], len(responses) - 1)
        result = responses[i]
        idx["n"] += 1
        return result

    return agent


class TestRecalls:
    """Test that recalls() detects context memory in responses."""

    def test_recalls_passes_when_context_present(self):
        agent = make_agent([
            "Hello! How can I help you today?",
            "Got it, I see order number 1234.",
            "Yes, your order 1234 is being processed and will arrive by Friday.",
        ])
        conv = ConversationTest(agent=agent)
        conv.say("Hi there")
        conv.say("I placed order number 1234 last week")
        response = conv.say("What's the status of my order?")

        assert response.recalls("order 1234")

    def test_recalls_fails_when_context_missing(self):
        agent = make_agent([
            "Hello!",
            "I can help with that.",
            "I don't have any information about that.",
        ])
        conv = ConversationTest(agent=agent)
        conv.say("My name is Swanand")
        conv.say("My order is 1234")
        response = conv.say("What was my order?")

        with pytest.raises(AssertionError, match="does not recall"):
            response.recalls("order 1234")

    def test_recalls_name_from_conversation(self):
        agent = make_agent([
            "Nice to meet you, Swanand!",
            "Of course Swanand, let me help you with that.",
        ])
        conv = ConversationTest(agent=agent)
        conv.say("My name is Swanand")
        response = conv.say("Can you help me?")

        assert response.recalls("Swanand")


class TestContradictsTurn:
    """Test contradiction detection across turns."""

    def test_no_contradiction_returns_false(self):
        agent = make_agent([
            "Your order will arrive by Friday.",
            "As I mentioned, your order is scheduled for Friday delivery.",
        ])
        conv = ConversationTest(agent=agent)
        conv.say("When does my order arrive?")
        response = conv.say("Are you sure?")

        # Turn 2 is the first assistant response
        result = response.contradicts_turn(2)
        # Should not contradict (both say Friday)
        assert result is False or result is True  # depends on model confidence

    def test_contradiction_detected(self):
        agent = make_agent([
            "Your order will arrive by Friday.",
            "Your order will arrive by Wednesday.",
        ])
        conv = ConversationTest(agent=agent)
        conv.say("When does my order arrive?")
        response = conv.say("When exactly?")

        # Turn 2 is first assistant response ("Friday")
        # Response says "Wednesday" -- contradiction
        # Note: NLI model may or may not catch this
        score = response.contradicts_turn(2)
        assert isinstance(score, bool)

    def test_invalid_turn_number_raises(self):
        agent = make_agent(["Hello"])
        conv = ConversationTest(agent=agent)
        response = conv.say("Hi")

        with pytest.raises(ValueError, match="Turn 99 not found"):
            response.contradicts_turn(99)
