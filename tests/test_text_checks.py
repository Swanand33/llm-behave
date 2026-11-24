"""Model-free checks: length, word_count, matches_regex, not_matches_regex."""

from __future__ import annotations

import re

import pytest

from llm_behave import assert_behavior


class TestLength:
    def test_within_bounds_passes(self):
        assert_behavior("hello").length(min=5, max=5)

    def test_only_max(self):
        assert_behavior("short").length(max=10)

    def test_only_min(self):
        assert_behavior("long enough").length(min=5)

    def test_too_long_fails_with_counts(self):
        with pytest.raises(AssertionError, match="has 11 characters"):
            assert_behavior("hello world").length(max=5)

    def test_too_short_fails(self):
        with pytest.raises(AssertionError, match="has 2 characters"):
            assert_behavior("hi").length(min=5)

    def test_empty_string_with_min_zero(self):
        assert_behavior("").length(min=0, max=0)

    def test_requires_a_bound(self):
        with pytest.raises(ValueError):
            assert_behavior("x").length()


class TestWordCount:
    def test_within_bounds(self):
        assert_behavior("one two  three\nfour").word_count(min=4, max=4)

    def test_too_many(self):
        with pytest.raises(AssertionError, match=r"has 4 word\(s\)"):
            assert_behavior("a b c d").word_count(max=3)

    def test_too_few(self):
        with pytest.raises(AssertionError, match=r"has 1 word\(s\)"):
            assert_behavior("a").word_count(min=2)

    def test_empty_is_zero_words(self):
        assert_behavior("").word_count(max=0)

    def test_requires_a_bound(self):
        with pytest.raises(ValueError):
            assert_behavior("x").word_count()


class TestRegex:
    def test_match_passes(self):
        assert_behavior("Order #1234 shipped").matches_regex(r"#\d{4}")

    def test_no_match_fails(self):
        with pytest.raises(AssertionError, match="does not match regex"):
            assert_behavior("no number").matches_regex(r"\d+")

    def test_flags_supported(self):
        assert_behavior("REFUND issued").matches_regex("refund", re.IGNORECASE)

    def test_not_matches_passes(self):
        assert_behavior("all clear").not_matches_regex(r"\b\d{16}\b")

    def test_not_matches_fails_and_shows_match(self):
        with pytest.raises(AssertionError, match="4111111111111111"):
            assert_behavior("card 4111111111111111").not_matches_regex(r"\b\d{16}\b")

    def test_invalid_regex_raises_re_error(self):
        with pytest.raises(re.error):
            assert_behavior("x").matches_regex("(")


class TestChainingAndResults:
    def test_chains_and_records_results(self):
        check = (
            assert_behavior("Your refund of $5 is on the way.")
            .length(max=100)
            .word_count(min=3)
            .matches_regex(r"\$\d+")
            .not_matches_regex(r"password")
        )
        assert [r.assertion_type for r in check.results] == [
            "length", "word_count", "matches_regex", "not_matches_regex",
        ]
        assert all(r.passed for r in check.results)

    def test_failure_is_recorded(self):
        check = assert_behavior("too long text")
        with pytest.raises(AssertionError):
            check.length(max=3)
        assert check.results[-1].passed is False
