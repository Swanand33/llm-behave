"""Refusal detection with the offline NLI model (no judge LLM, no API cost).

The response is the premise and a short statement about refusing is the
hypothesis; the NLI model's entailment probability is the refusal score. This
beat embedding similarity to refusal phrasings, which could not separate a
refusal ("No, I cannot provide medical diagnoses.") from an offer to help
("I can help with that. Please share your order number.").
"""

from __future__ import annotations

REFUSAL_HYPOTHESES: list[str] = [
    "The speaker declines to do what was asked.",
    "The assistant will not help with this.",
]


def refusal_score(text: str) -> tuple[float, str]:
    """Return (refusal probability 0-1, the hypothesis that scored highest).

    Calibrated on real model output: refusals score from about 0.14 to 0.99 and
    ordinary helpful replies stay below 0.02. Some soft refusals phrased as "I'm
    not able to share X" can score low; use a smaller threshold if you need them.
    """
    from llm_behave.engines.contradiction import get_contradiction_engine

    engine = get_contradiction_engine()
    scored = [(engine.check_entailment(text, h), h) for h in REFUSAL_HYPOTHESES]
    return max(scored)
