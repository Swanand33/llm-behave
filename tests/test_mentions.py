"""Integration tests for mentions() and not_mentions() assertions.

These tests use real sentence-transformer embeddings to validate
that semantic assertions catch real behavioral issues.
"""

from __future__ import annotations

import pytest

from llm_assert.core.assertions import assert_behavior


class TestMentions:
    """Test assert_behavior().mentions() with real embeddings."""

    def test_mentions_passes_on_related_content(self):
        output = (
            "I understand your frustration. Let me look into our refund policy "
            "for your order. We typically process refunds within 5-7 business days."
        )
        assert_behavior(output).mentions("refund policy")

    def test_mentions_passes_on_semantic_match(self):
        output = "We will send you your money back within a week."
        # "refund" is semantically close to "send money back"
        assert_behavior(output).mentions("refund")

    def test_mentions_fails_on_unrelated_content(self):
        output = "The weather in Pune is really nice today, isn't it?"
        with pytest.raises(AssertionError, match="does not mention"):
            assert_behavior(output).mentions("refund policy")

    def test_mentions_custom_threshold(self):
        output = "We apologize for the inconvenience with your purchase."
        # With a very high threshold, even somewhat related content should fail
        with pytest.raises(AssertionError):
            assert_behavior(output).mentions("refund policy", threshold=0.9)

    def test_mentions_chaining(self):
        output = (
            "Thank you for contacting support. I see your order #1234. "
            "Our refund policy allows returns within 30 days."
        )
        # Fluent chaining should work
        result = (
            assert_behavior(output)
            .mentions("refund policy")
            .mentions("order")
        )
        assert len(result.results) == 2
        assert all(r.passed for r in result.results)


class TestNotMentions:
    """Test assert_behavior().not_mentions() with real embeddings."""

    def test_not_mentions_passes_on_unrelated_content(self):
        output = "Here is your order status. It will arrive by Friday."
        assert_behavior(output).not_mentions("competitor pricing")

    def test_not_mentions_fails_on_related_content(self):
        output = "Our competitor offers a much better price for this product."
        with pytest.raises(AssertionError, match="unexpectedly mentions"):
            assert_behavior(output).not_mentions("competitor")

    def test_not_mentions_chaining_with_mentions(self):
        output = (
            "I'd be happy to help with your refund. "
            "Let me process that right away."
        )
        assert_behavior(output).mentions("refund").not_mentions("competitor")
