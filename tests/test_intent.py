"""Integration tests for intent() assertion."""

from __future__ import annotations

import pytest

from llm_assert.core.assertions import assert_behavior


class TestIntent:
    """Test assert_behavior().intent() with real embeddings."""

    def test_intent_matches_help_request(self):
        output = "Sure, let me pull up your account and help you with that right away."
        assert_behavior(output).intent("offering to help the customer")

    def test_intent_matches_refusal(self):
        output = "I'm sorry, but I'm unable to process that request due to our policy."
        assert_behavior(output).intent("declining a request politely")

    def test_intent_fails_on_mismatch(self):
        output = "The weather today is sunny with a high of 25 degrees."
        with pytest.raises(AssertionError, match="intent does not match"):
            assert_behavior(output).intent("processing a refund request")

    def test_intent_chains(self):
        output = (
            "I completely understand. Let me look into the refund for order #1234. "
            "I'll have this sorted out for you shortly."
        )
        (
            assert_behavior(output)
            .intent("helping with a refund")
            .mentions("order")
            .tone("helpful", threshold=0.3)
        )
