"""Contradiction detection engine using NLI (Natural Language Inference).

Uses a cross-encoder NLI model to detect if two statements contradict
each other. Runs offline -- no LLM judge needed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Multi-model engine cache: model_name -> ContradictionEngine
_engines: dict[str, ContradictionEngine] = {}

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# Label order of the NLI cross-encoder's output: [contradiction, entailment, neutral]
_CONTRADICTION = 0
_ENTAILMENT = 1


class ContradictionEngine:
    """NLI-based contradiction detection."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL) -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        """Load the NLI model lazily."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading NLI model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("NLI model loaded successfully.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for contradiction detection. "
                "Install it with: pip install llm-behave[semantic]"
            )

    def _predict_probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score (premise, hypothesis) pairs in one batched forward pass.

        The cross-encoder returns raw logits (e.g. 6.5 / -4.3), so they are
        softmaxed into probabilities over [contradiction, entailment, neutral].
        A single-score model has no separable classes and is returned as-is
        with shape (n,).
        """
        self._load_model()
        raw = np.asarray(self._model.predict(pairs))
        if raw.ndim == 1:
            return raw.astype(float)
        shifted = raw - raw.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        return probs / probs.sum(axis=1, keepdims=True)

    def check_contradiction(self, text_a: str, text_b: str) -> float:
        """Check if text_b contradicts text_a.

        Returns:
            Contradiction probability between 0 and 1.
            Higher = more likely a contradiction.
        """
        probs = self._predict_probs([(text_a, text_b)])
        if probs.ndim == 1:
            return float(probs[0])
        return float(probs[0, _CONTRADICTION])

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """Return the entailment probability that `premise` supports `hypothesis`.

        This powers grounding/hallucination checks: a claim is "grounded" when
        some piece of the retrieved context entails it. Returns a probability
        in [0, 1] (softmax over [contradiction, entailment, neutral]).
        """
        return self.max_entailment(hypothesis, [premise])[0]

    def max_entailment(
        self, hypothesis: str, premises: list[str]
    ) -> tuple[float, str]:
        """Return the best entailment score for `hypothesis` over `premises`.

        Scores every (premise, hypothesis) pair in a single batched forward
        pass and returns (best_score, best_premise) — the context sentence that
        most strongly supports the claim.
        """
        premises = [p for p in premises if p and p.strip()]
        if not premises:
            return 0.0, ""

        probs = self._predict_probs([(premise, hypothesis) for premise in premises])
        entailment = probs if probs.ndim == 1 else probs[:, _ENTAILMENT]

        best_idx = int(np.argmax(entailment))
        return float(entailment[best_idx]), premises[best_idx]


def get_contradiction_engine(model_name: str = DEFAULT_NLI_MODEL) -> ContradictionEngine:
    """Get or create a cached ContradictionEngine for the given model."""
    if model_name not in _engines:
        _engines[model_name] = ContradictionEngine(model_name)
    return _engines[model_name]
