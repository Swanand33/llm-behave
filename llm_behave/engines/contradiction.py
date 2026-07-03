"""Contradiction detection engine using NLI (Natural Language Inference).

Uses a cross-encoder NLI model to detect if two statements contradict
each other. Runs offline -- no LLM judge needed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_engine: ContradictionEngine | None = None

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


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

    def check_contradiction(self, text_a: str, text_b: str) -> float:
        """Check if text_b contradicts text_a.

        Returns:
            Contradiction score between 0 and 1.
            Higher = more likely a contradiction.
        """
        self._load_model()

        # NLI models return scores for [contradiction, entailment, neutral]
        scores = self._model.predict([(text_a, text_b)])

        if hasattr(scores[0], "__len__"):
            # Multi-class output: [contradiction, entailment, neutral]
            contradiction_score = float(scores[0][0])
        else:
            # Single score output
            contradiction_score = float(scores[0])

        return contradiction_score

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
        self._load_model()

        premises = [p for p in premises if p and p.strip()]
        if not premises:
            return 0.0, ""

        pairs = [(premise, hypothesis) for premise in premises]
        raw = np.asarray(self._model.predict(pairs))

        if raw.ndim == 1:
            # Single-score model: no separable entailment class available.
            entailment = raw.astype(float)
        else:
            # Softmax over [contradiction, entailment, neutral]; take entailment.
            shifted = raw - raw.max(axis=1, keepdims=True)
            probs = np.exp(shifted)
            probs /= probs.sum(axis=1, keepdims=True)
            entailment = probs[:, 1]

        best_idx = int(np.argmax(entailment))
        return float(entailment[best_idx]), premises[best_idx]


def get_contradiction_engine(model_name: str = DEFAULT_NLI_MODEL) -> ContradictionEngine:
    """Get the singleton contradiction engine instance."""
    global _engine
    if _engine is None or _engine._model_name != model_name:
        _engine = ContradictionEngine(model_name)
    return _engine
