"""Tests for the shared sentence splitter. Pure Python -- no ML model needed."""

from __future__ import annotations

from llm_behave.engines.text import split_sentences


class TestSplitSentences:
    def test_basic_split(self):
        assert split_sentences("Hello there. How are you? Great!") == [
            "Hello there", "How are you", "Great",
        ]

    def test_prices_and_decimals_stay_intact(self):
        assert split_sentences("The plan costs $4.99 per month. Version 2.1 is out.") == [
            "The plan costs $4.99 per month", "Version 2.1 is out",
        ]

    def test_urls_stay_intact(self):
        assert split_sentences("Visit example.com for details.") == [
            "Visit example.com for details",
        ]

    def test_abbreviations_do_not_split(self):
        assert split_sentences("See e.g. section 2. Ask Dr. Rao.") == [
            "See e.g. section 2", "Ask Dr. Rao",
        ]

    def test_bullet_list_splits_per_item(self):
        text = "Our policy:\n- Refunds within 30 days\n- Free returns\n1. Contact support"
        assert split_sentences(text) == [
            "Our policy:", "- Refunds within 30 days", "- Free returns", "1. Contact support",
        ]

    def test_blank_line_splits(self):
        assert split_sentences("First paragraph\n\nSecond paragraph") == [
            "First paragraph", "Second paragraph",
        ]

    def test_wrapped_line_does_not_split(self):
        assert split_sentences("A sentence that is\nhard-wrapped.") == [
            "A sentence that is\nhard-wrapped",
        ]

    def test_empty_and_blank(self):
        assert split_sentences("") == []
        assert split_sentences("   \n ") == []

    def test_punctuation_only_falls_back_to_text(self):
        assert split_sentences("...") == ["..."]

    def test_no_terminal_punctuation(self):
        assert split_sentences("just one fragment") == ["just one fragment"]
