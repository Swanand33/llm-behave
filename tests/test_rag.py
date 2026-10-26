"""Tests for RAG / grounding assertions.

These require sentence-transformers + the NLI model (the [semantic] extra).
They validate grounded_in(), no_hallucination(), and cites().
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="requires the [semantic] extra (sentence-transformers + torch)",
)

from llm_behave.core.assertions import assert_behavior
from llm_behave.engines.contradiction import get_contradiction_engine

CONTEXT = (
    "The refund window is 30 days from the delivery date. "
    "Refunds are issued to the original payment method. "
    "Shipping fees are non-refundable."
)


@pytest.fixture(scope="module", autouse=True)
def _warm_engine():
    """Load the NLI model once for the module (loading is slow)."""
    get_contradiction_engine()._load_model()


class TestGroundedIn:
    def test_grounded_answer_passes(self):
        answer = "You can get a refund within 30 days of delivery."
        assert_behavior(answer).grounded_in(CONTEXT)

    def test_ungrounded_claim_fails(self):
        answer = "Refunds are processed within 24 hours and include free express shipping."
        with pytest.raises(AssertionError, match="not grounded"):
            assert_behavior(answer).grounded_in(CONTEXT)


class TestNoHallucination:
    def test_supported_answer_passes(self):
        answer = "Shipping fees are non-refundable."
        assert_behavior(answer).no_hallucination(CONTEXT)

    def test_hallucinated_claim_fails(self):
        answer = "We also offer a lifetime warranty on all products."
        with pytest.raises(AssertionError, match="hallucinat"):
            assert_behavior(answer).no_hallucination(CONTEXT)

    def test_reports_the_hallucinated_sentence(self):
        answer = "Refunds go to your original payment method. You also earn 500 loyalty points."
        with pytest.raises(AssertionError, match="loyalty points"):
            assert_behavior(answer).no_hallucination(CONTEXT)


class TestCites:
    def test_cites_source_id_passes(self):
        answer = "According to the policy [doc-42], refunds take 30 days."
        assert_behavior(answer).cites("doc-42")

    def test_case_insensitive(self):
        answer = "See KB-Article-7 for details."
        assert_behavior(answer).cites("kb-article-7")

    def test_missing_citation_fails(self):
        answer = "Refunds take 30 days."
        with pytest.raises(AssertionError, match="does not cite"):
            assert_behavior(answer).cites("doc-42")


class TestEntailmentEngine:
    def test_max_entailment_prefers_supporting_sentence(self):
        engine = get_contradiction_engine()
        premises = [
            "The sky is blue.",
            "The refund window is 30 days from delivery.",
        ]
        score, support = engine.max_entailment(
            "Customers have 30 days to request a refund.", premises
        )
        assert score > 0.5
        assert "refund window" in support

    def test_max_entailment_empty_premises(self):
        engine = get_contradiction_engine()
        score, support = engine.max_entailment("anything", [])
        assert score == 0.0
        assert support == ""


class TestContradictionProbabilities:
    """check_contradiction() must return a probability, not a raw logit (v0.2.1)."""

    def test_contradiction_score_in_unit_interval(self):
        engine = get_contradiction_engine()
        contra = engine.check_contradiction(
            "Refunds are allowed within 30 days.", "Refunds are never allowed."
        )
        agree = engine.check_contradiction(
            "Refunds are allowed within 30 days.", "You can get a refund within 30 days."
        )
        assert 0.0 <= agree <= 1.0
        assert 0.0 <= contra <= 1.0
        assert contra > 0.9
        assert agree < 0.1

    def test_engines_cached_per_model(self):
        from llm_behave.engines.contradiction import ContradictionEngine, _engines

        default = get_contradiction_engine()
        fake = "fake/nli-model"
        _engines[fake] = ContradictionEngine(fake)
        try:
            assert get_contradiction_engine() is default
        finally:
            del _engines[fake]


class TestGroundingWithNumbers:
    """Prices, decimals and URLs must not be split into fake claims (v0.2.1)."""

    PRICING = (
        "The Pro plan costs $4.99 per month. "
        "Full details are at example.com/pricing."
    )

    def test_price_claim_is_grounded(self):
        assert_behavior("The Pro plan costs $4.99 per month.").grounded_in(self.PRICING)

    def test_wrong_price_is_hallucination(self):
        with pytest.raises(AssertionError, match="Hallucinated claims"):
            assert_behavior("The Pro plan costs $9.99 per month.").no_hallucination(
                self.PRICING
            )
