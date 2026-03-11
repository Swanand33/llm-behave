"""Integration tests for tone() assertion.

Tests that the embedding-similarity approach to tone detection
produces meaningful results for common tones.
"""

from __future__ import annotations

import pytest

from llm_assert.core.assertions import assert_behavior
from llm_assert.engines.tone import get_tone_engine, TONE_DESCRIPTIONS


class TestToneEngine:
    """Test the tone engine directly."""

    def test_empathetic_text_scores_high(self):
        engine = get_tone_engine()
        text = (
            "I completely understand how frustrating this must be for you. "
            "I'm so sorry you're going through this. Let me help make it right."
        )
        score = engine.check_tone(text, "empathetic")
        assert score > 0.3, f"Empathetic text should score >0.3, got {score}"

    def test_professional_text_scores_high(self):
        engine = get_tone_engine()
        text = (
            "Per our records, your account has been updated accordingly. "
            "Please find the attached invoice for your reference."
        )
        score = engine.check_tone(text, "professional")
        assert score > 0.3, f"Professional text should score >0.3, got {score}"

    def test_rude_text_scores_high_for_rude(self):
        engine = get_tone_engine()
        text = "That's not my problem. Figure it out yourself. Stop wasting my time."
        score = engine.check_tone(text, "rude")
        assert score > 0.3, f"Rude text should score >0.3 for rude, got {score}"

    def test_custom_tone_uses_fallback_description(self):
        engine = get_tone_engine()
        text = "Haha that's hilarious! What a great joke!"
        score = engine.check_tone(text, "humorous")
        # "humorous" isn't in TONE_DESCRIPTIONS, falls back to generic description
        assert isinstance(score, float)

    def test_tone_descriptions_cover_common_tones(self):
        expected = {"empathetic", "professional", "friendly", "formal", "casual", "rude", "helpful"}
        assert expected.issubset(set(TONE_DESCRIPTIONS.keys()))


class TestToneAssertion:
    """Test assert_behavior().tone() end-to-end."""

    def test_empathetic_tone_passes(self):
        output = (
            "I'm really sorry to hear about your experience. "
            "That must have been very frustrating. Let me see what I can do to help."
        )
        assert_behavior(output).tone("empathetic", threshold=0.3)

    def test_wrong_tone_fails(self):
        output = "Read the manual. It's not that hard."
        with pytest.raises(AssertionError, match="tone is not 'empathetic'"):
            assert_behavior(output).tone("empathetic", threshold=0.6)

    def test_tone_chains_with_mentions(self):
        output = (
            "I understand your concern about the refund. "
            "Let me personally ensure this is handled properly for you."
        )
        assert_behavior(output).tone("empathetic", threshold=0.3).mentions("refund")
