"""Tone detection engine using embedding similarity.

Instead of a binary sentiment classifier (SST-2), this uses the semantic
engine to compare output embeddings against tone description embeddings.
This supports arbitrary tone labels: empathetic, professional, friendly, etc.
"""

from __future__ import annotations

_engine: ToneEngine | None = None

# Tone descriptions -- the engine compares output embeddings against these
TONE_DESCRIPTIONS: dict[str, str] = {
    "empathetic": "This text shows deep understanding and compassion for someone's feelings and situation.",
    "professional": "This text is formal, business-appropriate, clear, and maintains a professional demeanor.",
    "friendly": "This text is warm, approachable, conversational, and makes the reader feel welcome.",
    "formal": "This text uses formal language, proper grammar, and maintains a serious, official tone.",
    "casual": "This text is informal, relaxed, uses everyday language and a laid-back style.",
    "rude": "This text is impolite, dismissive, hostile, or shows disrespect toward the reader.",
    "helpful": "This text actively tries to assist, provides useful information, and guides the reader.",
    "urgent": "This text conveys urgency, importance, and a need for immediate attention or action.",
    "apologetic": "This text expresses sincere regret, takes responsibility, and offers to make things right.",
    "confident": "This text is assertive, self-assured, and communicates with authority and certainty.",
}


class ToneEngine:
    """Tone detection via embedding similarity against tone descriptions."""

    def check_tone(self, text: str, expected_tone: str) -> float:
        """Check how well text matches an expected tone.

        Args:
            text: The text to analyze.
            expected_tone: Tone label (e.g., "empathetic", "professional").

        Returns:
            Similarity score between 0 and 1.
        """
        from llm_assert.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()

        tone_desc = TONE_DESCRIPTIONS.get(
            expected_tone.lower(),
            f"This text has a {expected_tone} tone.",
        )

        return engine.similarity(text, tone_desc)

    def tone_similarity(self, text_a: str, text_b: str) -> float:
        """Check how similar the tone is between two texts.

        Used for consistent_tone_across_turns().
        """
        from llm_assert.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        return engine.similarity(text_a, text_b)


def get_tone_engine() -> ToneEngine:
    """Get the singleton tone engine instance."""
    global _engine
    if _engine is None:
        _engine = ToneEngine()
    return _engine
