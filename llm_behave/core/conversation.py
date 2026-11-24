"""Multi-turn conversation testing for LLM applications."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel


class TurnRecord(BaseModel):
    """Record of a single conversation turn."""

    turn_number: int
    role: str
    content: str


class ConversationResponse:
    """Response from a conversation turn, with assertion methods."""

    def __init__(self, content: str, turn_number: int, history: list[TurnRecord]) -> None:
        self._content = content
        self._turn_number = turn_number
        self._history = history

    @property
    def text(self) -> str:
        return self._content

    def __str__(self) -> str:
        return self._content

    def recalls(self, concept: str, threshold: float = 0.5) -> bool:
        """Assert the response recalls/references a concept from conversation history.

        Scores the concept against each sentence and takes the best match, so a
        long response isn't diluted by its unrelated sentences.
        Raises AssertionError if the concept is not recalled.
        """
        from llm_behave.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        score = engine.max_sentence_similarity(self._content, concept)
        if score < threshold:
            raise AssertionError(
                f"Response does not recall '{concept}'. "
                f"Similarity: {score:.3f} (threshold: {threshold})\n"
                f"Response: {self._content[:200]}"
            )
        return True

    def contradicts_turn(self, turn_number: int, threshold: float = 0.7) -> bool:
        """Return True if the response contradicts a specific earlier turn.

        Unlike recalls(), this is a predicate and does not raise on its own --
        wrap it: `assert response.contradicts_turn(2)` or
        `assert not response.contradicts_turn(2)`.
        `threshold` is the NLI contradiction probability (0 to 1).
        """
        target_turn = None
        for turn in self._history:
            if turn.turn_number == turn_number:
                target_turn = turn
                break

        if target_turn is None:
            raise ValueError(f"Turn {turn_number} not found in conversation history.")

        from llm_behave.engines.contradiction import get_contradiction_engine

        engine = get_contradiction_engine()
        contradiction_score = engine.check_contradiction(
            target_turn.content, self._content
        )
        return contradiction_score >= threshold

    def not_contradicts_turn(self, turn_number: int, threshold: float = 0.7) -> bool:
        """Assert the response does not contradict a specific earlier turn.

        The raising counterpart of contradicts_turn(): raises AssertionError on a
        contradiction, returns True otherwise.
        """
        if self.contradicts_turn(turn_number, threshold):
            earlier = next(t for t in self._history if t.turn_number == turn_number)
            raise AssertionError(
                f"Response contradicts turn {turn_number} (threshold: {threshold}).\n"
                f"Turn {turn_number}: {earlier.content[:200]}\n"
                f"Response: {self._content[:200]}"
            )
        return True

    def not_contradicts_history(self, threshold: float = 0.7) -> bool:
        """Assert the response does not contradict anything the assistant said earlier.

        Checks the response against every previous assistant turn. Raises
        AssertionError naming the turns it contradicts.
        """
        from llm_behave.engines.contradiction import get_contradiction_engine

        engine = get_contradiction_engine()
        offenders = []
        for turn in self._history:
            if turn.role != "assistant" or turn.turn_number >= self._turn_number:
                continue
            score = engine.check_contradiction(turn.content, self._content)
            if score >= threshold:
                offenders.append((turn, score))

        if offenders:
            lines = [
                f"  turn {t.turn_number} (score {score:.3f}): {t.content[:80]!r}"
                for t, score in offenders
            ]
            raise AssertionError(
                f"Response contradicts earlier assistant turns (threshold: {threshold}):\n"
                + "\n".join(lines)
                + f"\nResponse: {self._content[:200]}"
            )
        return True

    def consistent_tone_across_turns(self, threshold: float = 0.7) -> bool:
        """Assert tone is consistent across all assistant turns.

        Compares tone profiles (how strongly each reply reads as empathetic,
        professional, rude, ...) between consecutive assistant turns, so replies
        on different topics but in the same register still count as consistent.
        Raises AssertionError if the average falls below threshold.
        """
        from llm_behave.engines.tone import get_tone_engine

        engine = get_tone_engine()
        # say() records the current reply in history before building this response
        assistant_turns = [t.content for t in self._history if t.role == "assistant"]

        if len(assistant_turns) < 2:
            return True

        scores = []
        for i in range(len(assistant_turns) - 1):
            score = engine.tone_similarity(assistant_turns[i], assistant_turns[i + 1])
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        if avg_score < threshold:
            raise AssertionError(
                f"Tone inconsistency detected across turns. "
                f"Average consistency: {avg_score:.3f} (threshold: {threshold})"
            )
        return True


class ConversationTest:
    """Multi-turn conversation testing harness.

    Usage:
        conv = ConversationTest(agent=my_agent_function)
        conv.say("Hello, my name is Swanand")
        conv.say("I placed order 1234")
        response = conv.say("What was my order?")
        assert response.recalls("order 1234")
        assert response.recalls("Swanand")
    """

    def __init__(self, agent: Callable[..., str | Awaitable[str]]) -> None:
        self._agent = agent
        self._history: list[TurnRecord] = []
        self._turn_counter = 0

    @property
    def history(self) -> list[TurnRecord]:
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    def _record_user(self, message: str) -> list[dict[str, str]]:
        self._turn_counter += 1
        self._history.append(
            TurnRecord(turn_number=self._turn_counter, role="user", content=message)
        )
        return [{"role": t.role, "content": t.content} for t in self._history]

    def _record_assistant(self, response_text: object) -> ConversationResponse:
        if inspect.isawaitable(response_text):
            raise TypeError(
                "Agent returned an awaitable. Use `await conv.asay(...)` for async agents."
            )
        if not isinstance(response_text, str):
            raise TypeError(
                f"Agent must return a string, got {type(response_text).__name__}. "
                "Wrap your LLM call to return the text content."
            )
        self._turn_counter += 1
        self._history.append(
            TurnRecord(
                turn_number=self._turn_counter, role="assistant", content=response_text
            )
        )
        return ConversationResponse(
            content=response_text,
            turn_number=self._turn_counter,
            history=list(self._history),
        )

    def say(self, message: str) -> ConversationResponse:
        """Send a message and get the agent's response.

        Args:
            message: The user message to send.

        Returns:
            ConversationResponse with assertion methods.
        """
        messages = self._record_user(message)
        return self._record_assistant(self._agent(messages))

    async def asay(self, message: str) -> ConversationResponse:
        """Async version of say(). The agent may be a sync or an ``async def`` callable.

        Usage:
            conv = ConversationTest(agent=provider.achat)
            response = await conv.asay("Hello")
        """
        messages = self._record_user(message)
        result: Any = self._agent(messages)
        if inspect.isawaitable(result):
            result = await result
        return self._record_assistant(result)
