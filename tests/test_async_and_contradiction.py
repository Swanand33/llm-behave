"""Async support and contradiction assertions, with fake engines (no model needed)."""

from __future__ import annotations

import asyncio

import pytest

from llm_behave import ConversationTest, MockProvider, assert_behavior


class TestAsyncProvider:
    def test_achat_returns_same_as_chat(self):
        provider = MockProvider(responses=["one", "two"])
        assert asyncio.run(provider.achat([])) == "one"
        assert provider.chat([]) == "two"

    def test_achat_with_tools(self):
        provider = MockProvider(tool_responses=[("hi", [{"name": "t", "arguments": {}}])])
        text, calls = asyncio.run(provider.achat_with_tools([], []))
        assert text == "hi" and calls[0]["name"] == "t"


class TestAsyncConversation:
    def test_asay_with_provider_achat(self):
        async def run():
            conv = ConversationTest(agent=MockProvider(responses=["A", "B"]).achat)
            first = await conv.asay("hello")
            second = await conv.asay("again")
            return conv, first, second

        conv, first, second = asyncio.run(run())
        assert (first.text, second.text) == ("A", "B")
        assert conv.turn_count == 4
        assert [t.role for t in conv.history] == ["user", "assistant", "user", "assistant"]

    def test_asay_accepts_sync_agent(self):
        conv = ConversationTest(agent=lambda messages: "sync reply")
        assert asyncio.run(conv.asay("hi")).text == "sync reply"

    def test_asay_with_async_def_agent_sees_history(self):
        seen = []

        async def agent(messages):
            seen.append(len(messages))
            return f"reply {len(messages)}"

        conv = ConversationTest(agent=agent)

        async def run():
            await conv.asay("one")
            return await conv.asay("two")

        assert asyncio.run(run()).text == "reply 3"
        assert seen == [1, 3]

    def test_say_with_async_agent_gives_helpful_error(self):
        async def agent(messages):
            return "x"

        conv = ConversationTest(agent=agent)
        with pytest.raises(TypeError, match="asay"):
            conv.say("hi")

    def test_asay_non_string_result_rejected(self):
        async def agent(messages):
            return 42

        with pytest.raises(TypeError, match="string"):
            asyncio.run(ConversationTest(agent=agent).asay("hi"))


class FakeNLI:
    """Flags a contradiction when the earlier turn says 'approved' and the new one 'cannot'."""

    def check_contradiction(self, a: str, b: str) -> float:
        return 1.0 if ("approved" in a and "cannot" in b) else 0.0


@pytest.fixture
def fake_nli(monkeypatch):
    engine = FakeNLI()
    monkeypatch.setattr(
        "llm_behave.engines.contradiction.get_contradiction_engine", lambda *a, **k: engine
    )


def _conversation(replies):
    conv = ConversationTest(agent=MockProvider(responses=replies).chat)
    response = None
    for i in range(len(replies)):
        response = conv.say(f"message {i}")
    return response


class TestNotContradicts:
    def test_history_passes_when_consistent(self, fake_nli):
        response = _conversation(["Your refund is approved.", "It arrives Friday."])
        assert response.not_contradicts_history() is True

    def test_history_raises_and_names_the_turn(self, fake_nli):
        response = _conversation(["Your refund is approved.", "Actually we cannot refund it."])
        with pytest.raises(AssertionError, match=r"turn 2 \(score 1.000\)"):
            response.not_contradicts_history()

    def test_history_ignores_user_turns_and_current_turn(self, fake_nli):
        conv = ConversationTest(agent=MockProvider(responses=["we cannot do that"]).chat)
        # A user turn containing "approved" must not be compared; the reply itself must not
        # be compared with itself.
        response = conv.say("It was approved yesterday")
        assert response.not_contradicts_history() is True

    def test_history_threshold_respected(self, monkeypatch):
        class Mid:
            def check_contradiction(self, a, b):
                return 0.6

        monkeypatch.setattr(
            "llm_behave.engines.contradiction.get_contradiction_engine", lambda *a, **k: Mid()
        )
        response = _conversation(["first", "second"])
        assert response.not_contradicts_history(threshold=0.7) is True
        with pytest.raises(AssertionError):
            response.not_contradicts_history(threshold=0.5)

    def test_not_contradicts_turn(self, fake_nli):
        response = _conversation(["Your refund is approved.", "Actually we cannot refund it."])
        with pytest.raises(AssertionError, match="contradicts turn 2"):
            response.not_contradicts_turn(2)
        assert response.contradicts_turn(2) is True  # predicate form still works

    def test_not_contradicts_turn_passes(self, fake_nli):
        response = _conversation(["Your refund is approved.", "It arrives Friday."])
        assert response.not_contradicts_turn(2) is True

    def test_not_contradicts_turn_unknown_turn(self, fake_nli):
        response = _conversation(["a", "b"])
        with pytest.raises(ValueError):
            response.not_contradicts_turn(99)


class TestRefusesWiring:
    @pytest.fixture
    def score(self, monkeypatch):
        state = {"value": 0.9}
        monkeypatch.setattr(
            "llm_behave.engines.refusal.refusal_score",
            lambda text: (state["value"], "The speaker declines to do what was asked."),
        )
        return state

    def test_refuses_passes_above_threshold(self, score):
        check = assert_behavior("no").refuses()
        assert check.results[-1].assertion_type == "refuses"
        assert check.results[-1].similarity_score == 0.9

    def test_refuses_fails_below_threshold(self, score):
        score["value"] = 0.01
        with pytest.raises(AssertionError, match="does not look like a refusal"):
            assert_behavior("sure").refuses()

    def test_threshold_is_configurable(self, score):
        score["value"] = 0.05
        assert_behavior("x").refuses(threshold=0.02)
        with pytest.raises(AssertionError):
            assert_behavior("x").refuses(threshold=0.5)

    def test_not_refuses_passes_when_low(self, score):
        score["value"] = 0.0
        assert_behavior("sure").not_refuses()

    def test_not_refuses_fails_when_high(self, score):
        with pytest.raises(AssertionError, match="looks like a refusal"):
            assert_behavior("no").not_refuses()
