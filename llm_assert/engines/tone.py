"""Tone detection engine using embedding similarity.

Instead of a binary sentiment classifier (SST-2), this uses the semantic
engine to compare output embeddings against tone description embeddings.
This supports arbitrary tone labels: empathetic, professional, friendly, etc.
"""

from __future__ import annotations

_engine: ToneEngine | None = None

# Tone descriptions -- fallback for unknown tones
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

# Concrete example sentences for each tone -- used for max-similarity scoring
TONE_EXAMPLES: dict[str, list[str]] = {
    "empathetic": [
        "I completely understand how frustrating this must be for you.",
        "I'm so sorry you're going through this difficult situation.",
        "That must have been really hard. I'm here to help you.",
        "I hear you, and I truly understand your concerns.",
        "I can imagine how upsetting this is. Let me help make it right.",
    ],
    "professional": [
        "Please find the attached documentation for your reference.",
        "As per our records, your account has been updated accordingly.",
        "We will process your request within the specified timeframe.",
        "Thank you for bringing this matter to our attention.",
        "Per our policy, the following steps will be taken.",
    ],
    "friendly": [
        "Hey! Great to hear from you, happy to help!",
        "That's awesome! Let's get this sorted out for you.",
        "Hi there! No worries at all, we've got you covered.",
        "Sure thing! I'd love to help you with that.",
    ],
    "formal": [
        "We hereby acknowledge receipt of your correspondence.",
        "Please be advised that the aforementioned policy applies.",
        "In accordance with our terms and conditions, we notify you.",
        "We regret to inform you of the following decision.",
    ],
    "casual": [
        "Sure thing! Let me take a quick look at that for you.",
        "No worries! We'll get that fixed up in no time.",
        "Yeah, I can totally help with that.",
        "Hey, that's easy to fix!",
    ],
    "rude": [
        "That's not my problem, figure it out yourself.",
        "Stop bothering me with this, read the manual.",
        "I don't care. It's not our fault.",
        "This is a waste of my time.",
    ],
    "helpful": [
        "Let me walk you through the steps to resolve this.",
        "Here's exactly what you need to do to fix this issue.",
        "I'll make sure this gets resolved for you right away.",
        "Let me look into this and get back to you with a solution.",
    ],
    "urgent": [
        "This requires immediate action. Please respond right away.",
        "URGENT: Your account will be affected if not addressed now.",
        "Act now — this needs to be resolved immediately.",
        "Please take action on this as soon as possible.",
    ],
    "apologetic": [
        "I sincerely apologize for the inconvenience we've caused.",
        "I'm truly sorry about this error. We take full responsibility.",
        "Please accept our deepest apologies for this mistake.",
        "We are sorry for any trouble this has caused you.",
    ],
    "confident": [
        "We guarantee this solution will resolve your issue.",
        "I am certain that we can fix this for you.",
        "This is the best approach and it will work.",
        "Trust me, this will solve your problem completely.",
    ],
}


class ToneEngine:
    """Tone detection via embedding similarity against tone descriptions."""

    def check_tone(self, text: str, expected_tone: str) -> float:
        """Check how well text matches an expected tone.

        Args:
            text: The text to analyze.
            expected_tone: Tone label (e.g., "empathetic", "professional").

        Returns:
            Max similarity score against tone example sentences (0 to 1).
        """
        from llm_assert.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()

        examples = TONE_EXAMPLES.get(expected_tone.lower())
        if examples is None:
            fallback = TONE_DESCRIPTIONS.get(
                expected_tone.lower(),
                f"This text has a {expected_tone} tone.",
            )
            return engine.similarity(text, fallback)

        # Batch encode all examples + text in one forward pass for speed
        import numpy as np
        engine._load_model()
        all_texts = [text] + examples
        embeddings = engine._model.encode(all_texts, convert_to_numpy=True)
        text_emb = embeddings[0]
        example_embs = embeddings[1:]

        scores = []
        for emb in example_embs:
            dot = np.dot(text_emb, emb)
            norm = np.linalg.norm(text_emb) * np.linalg.norm(emb)
            scores.append(float(dot / norm) if norm > 0 else 0.0)
        return max(scores)

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
