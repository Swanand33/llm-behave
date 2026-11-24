"""Tone-profile logic tested with a fake encoder (no model download needed)."""

from __future__ import annotations

import numpy as np
import pytest

from llm_behave import ConversationTest, MockProvider
from llm_behave.engines import tone as tone_module
from llm_behave.engines.tone import TONE_EXAMPLES, ToneEngine

LABELS = list(TONE_EXAMPLES)
DIM = len(LABELS) + 1  # one axis per tone + one shared "topic" axis

# Text keywords -> which tone axis they light up
KEYWORDS = {
    "sorry": "empathetic",
    "regards": "professional",
    "awesome": "friendly",
    "whatever": "rude",
}


class FakeEngine:
    """Embeds text on tone axes: examples sit on their own label's axis."""

    model_name = "fake-tone-model"

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(DIM)
        vec[-1] = 0.5  # shared component, like "topic" similarity
        lowered = text.lower()
        for word, label in KEYWORDS.items():
            if word in lowered:
                vec[LABELS.index(label)] = 1.0
        for label in LABELS:
            if text in TONE_EXAMPLES[label]:
                vec[LABELS.index(label)] = 1.0
        return vec

    def encode_batch(self, texts):
        return np.stack([self._embed(t) for t in texts])

    def similarity(self, a, b):
        ea, eb = self._embed(a), self._embed(b)
        return float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb)))


@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(
        "llm_behave.engines.semantic.get_semantic_engine", lambda *a, **k: fake
    )
    monkeypatch.setattr(tone_module, "_engine", None)
    return ToneEngine()


class TestToneProfile:
    def test_profile_covers_every_tone(self, engine):
        profile = engine.tone_profile("sorry about that")
        assert set(profile) == set(LABELS)

    def test_profile_peaks_at_matching_tone(self, engine):
        profile = engine.tone_profile("so sorry to hear that")
        assert max(profile, key=profile.get) == "empathetic"

    def test_example_embeddings_cached(self, engine):
        engine.tone_profile("sorry")
        cache = dict(engine._example_cache)
        engine.tone_profile("regards")
        assert engine._example_cache["fake-tone-model"] is cache["fake-tone-model"]


class TestToneSimilarity:
    def test_same_tone_different_topic_is_high(self, engine):
        score = engine.tone_similarity(
            "so sorry about your refund", "sorry, the shipping was late"
        )
        assert score > 0.9

    def test_different_tone_is_low(self, engine):
        score = engine.tone_similarity("so sorry about that", "whatever, not my problem")
        assert score < 0.2

    def test_score_stays_in_unit_range(self, engine):
        for a, b in [("sorry", "regards"), ("awesome", "awesome"), ("x", "y")]:
            assert 0.0 <= engine.tone_similarity(a, b) <= 1.0

    def test_flat_profile_returns_zero(self, engine):
        # neither text lights any tone axis -> nothing to compare
        assert engine.tone_similarity("plain words", "other words") == 0.0


class TestConsistentToneAcrossTurns:
    def _run(self, replies, monkeypatch, engine):
        monkeypatch.setattr(tone_module, "_engine", engine)
        conv = ConversationTest(agent=MockProvider(responses=replies).chat)
        response = None
        for i in range(len(replies)):
            response = conv.say(f"message {i}")
        return response

    def test_consistent_register_passes(self, engine, monkeypatch):
        response = self._run(
            ["sorry about the delay", "sorry, I will fix the invoice"], monkeypatch, engine
        )
        assert response.consistent_tone_across_turns(threshold=0.9)

    def test_tone_flip_fails(self, engine, monkeypatch):
        response = self._run(
            ["sorry about the delay", "whatever, not my problem"], monkeypatch, engine
        )
        with pytest.raises(AssertionError, match="Tone inconsistency"):
            response.consistent_tone_across_turns(threshold=0.5)

    def test_current_turn_not_double_counted(self, engine, monkeypatch):
        # Before the fix the last reply was compared with itself (score 1.0),
        # masking a flip when it was averaged with the other pair.
        response = self._run(
            ["sorry A", "whatever B", "whatever C"], monkeypatch, engine
        )
        scores = []
        turns = [t.content for t in response._history if t.role == "assistant"]
        assert turns == ["sorry A", "whatever B", "whatever C"]
        for a, b in zip(turns, turns[1:]):
            scores.append(engine.tone_similarity(a, b))
        assert scores[0] < 0.2 and scores[1] > 0.9
        with pytest.raises(AssertionError):
            response.consistent_tone_across_turns(threshold=0.7)

    def test_single_assistant_turn_passes(self, engine, monkeypatch):
        response = self._run(["only reply"], monkeypatch, engine)
        assert response.consistent_tone_across_turns(threshold=0.99)
