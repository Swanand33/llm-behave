"""Tests for multi-turn ConversationTest.

Uses mock agents -- no real LLM calls needed.
"""

from __future__ import annotations

import pytest

from llm_assert.core.conversation import ConversationTest, ConversationResponse


def make_echo_agent(responses: list[str] | None = None):
    """Create a mock agent that returns predefined responses."""
    call_count = {"n": 0}
    default_responses = responses or ["Mock response"]

    def agent(messages):
        idx = min(call_count["n"], len(default_responses) - 1)
        result = default_responses[idx]
        call_count["n"] += 1
        return result

    return agent


class TestConversationTestBasics:
    """Test basic conversation flow without ML models."""

    def test_creates_conversation(self):
        agent = make_echo_agent()
        conv = ConversationTest(agent=agent)
        assert conv.turn_count == 0
        assert conv.history == []

    def test_say_returns_conversation_response(self):
        agent = make_echo_agent(["Hello there!"])
        conv = ConversationTest(agent=agent)
        response = conv.say("Hi")
        assert isinstance(response, ConversationResponse)
        assert response.text == "Hello there!"

    def test_tracks_turn_count(self):
        agent = make_echo_agent(["R1", "R2", "R3"])
        conv = ConversationTest(agent=agent)
        conv.say("Turn 1")
        conv.say("Turn 2")
        conv.say("Turn 3")
        # Each say() creates 2 turns (user + assistant)
        assert conv.turn_count == 6

    def test_builds_message_history(self):
        agent = make_echo_agent(["Response 1", "Response 2"])
        conv = ConversationTest(agent=agent)
        conv.say("Hello")
        conv.say("How are you?")

        assert len(conv.history) == 4
        assert conv.history[0].role == "user"
        assert conv.history[0].content == "Hello"
        assert conv.history[1].role == "assistant"
        assert conv.history[1].content == "Response 1"
        assert conv.history[2].role == "user"
        assert conv.history[2].content == "How are you?"
        assert conv.history[3].role == "assistant"
        assert conv.history[3].content == "Response 2"

    def test_rejects_non_string_agent_response(self):
        def bad_agent(messages):
            return 42

        conv = ConversationTest(agent=bad_agent)
        with pytest.raises(TypeError, match="Agent must return a string"):
            conv.say("Hi")

    def test_response_str_is_text(self):
        agent = make_echo_agent(["Hello!"])
        conv = ConversationTest(agent=agent)
        response = conv.say("Hi")
        assert str(response) == "Hello!"
