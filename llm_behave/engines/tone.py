"""Tone detection engine using embedding similarity.

Instead of a binary sentiment classifier (SST-2), this uses the semantic
engine to compare output embeddings against tone description embeddings.
This supports arbitrary tone labels: empathetic, professional, friendly, etc.
"""

from __future__ import annotations

import threading

import numpy as np

_engine: ToneEngine | None = None
_LOCK = threading.Lock()

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
    """Tone detection via embedding similarity against tone examples."""

    def __init__(self) -> None:
        # Example embeddings are cached per semantic model name
        self._example_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _labels() -> list[str]:
        return list(TONE_EXAMPLES)

    def _example_embeddings(self, engine) -> np.ndarray:
        """Embeddings of every example sentence, in TONE_EXAMPLES order (cached)."""
        cached = self._example_cache.get(engine.model_name)
        if cached is None:
            flat = [ex for label in self._labels() for ex in TONE_EXAMPLES[label]]
            cached = engine.encode_batch(flat)
            self._example_cache[engine.model_name] = cached
        return cached

    @staticmethod
    def _cosine(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)
        dots = matrix @ vec
        scores: np.ndarray = np.divide(dots, norms, out=np.zeros_like(dots), where=norms != 0)
        return scores

    def check_tone(self, text: str, expected_tone: str) -> float:
        """Check how well text matches an expected tone.

        Args:
            text: The text to analyze.
            expected_tone: Tone label (e.g., "empathetic", "professional").

        Returns:
            Max similarity score against tone example sentences (0 to 1).
        """
        from llm_behave.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()

        examples = TONE_EXAMPLES.get(expected_tone.lower())
        if examples is None:
            fallback = TONE_DESCRIPTIONS.get(
                expected_tone.lower(),
                f"This text has a {expected_tone} tone.",
            )
            return engine.similarity(text, fallback)

        # Batch encode all examples + text in one forward pass for speed
        embeddings = engine.encode_batch([text] + examples)
        return float(self._cosine(embeddings[0], embeddings[1:]).max())

    def tone_profile(self, text: str) -> dict[str, float]:
        """Score text against every known tone (best-matching example per tone)."""
        from llm_behave.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        example_embs = self._example_embeddings(engine)
        scores = self._cosine(engine.encode_batch([text])[0], example_embs)

        profile: dict[str, float] = {}
        offset = 0
        for label in self._labels():
            count = len(TONE_EXAMPLES[label])
            profile[label] = float(scores[offset:offset + count].max())
            offset += count
        return profile

    def tone_similarity(self, text_a: str, text_b: str) -> float:
        """How similar the *tone* of two texts is, from 0 to 1.

        Compares tone profiles rather than raw embeddings: each text is scored
        against every tone, the profile is centred (so a text's general
        similarity to "tone-like" sentences cancels out), and the cosine of the
        two centred profiles is returned. Two replies on different topics but in
        the same register score high; a polite reply followed by a rude one
        scores low. Used by consistent_tone_across_turns().
        """
        labels = self._labels()
        profile_a, profile_b = self.tone_profile(text_a), self.tone_profile(text_b)
        a = np.array([profile_a[label] for label in labels])
        b = np.array([profile_b[label] for label in labels])
        a, b = a - a.mean(), b - b.mean()
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            # A flat profile carries no tone signal, so there is nothing to compare
            return 0.0
        return float(np.clip(a @ b / (norm_a * norm_b), 0.0, 1.0))


def get_tone_engine() -> ToneEngine:
    """Get the singleton tone engine instance."""
    global _engine
    with _LOCK:
        if _engine is None:
            _engine = ToneEngine()
        return _engine
