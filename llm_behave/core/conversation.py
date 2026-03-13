"""Multi-turn conversation testing for LLM applications."""

from __future__ import annotations

from typing import Any, Callable

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
        """Check if the response recalls/references a concept from conversation history."""
        from llm_behave.engines.semantic import get_semantic_engine

        engine = get_semantic_engine()
        score = engine.similarity(self._content, concept)
        if score < threshold:
            raise AssertionError(
                f"Response does not recall '{concept}'. "
                f"Similarity: {score:.3f} (threshold: {threshold})\n"
                f"Response: {self._content[:200]}"
            )
        return True

    def contradicts_turn(self, turn_number: int, threshold: float = 0.7) -> bool:
        """Check if the response contradicts a specific earlier turn."""
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

    def consistent_tone_across_turns(self, threshold: float = 0.7) -> bool:
        """Check if tone is consistent across all assistant turns."""
        from llm_behave.engines.tone import get_tone_engine

        engine = get_tone_engine()
        assistant_turns = [t.content for t in self._history if t.role == "assistant"]
        assistant_turns.append(self._content)

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

    def __init__(self, agent: Callable[..., str]) -> None:
        self._agent = agent
        self._history: list[TurnRecord] = []
        self._turn_counter = 0

    @property
    def history(self) -> list[TurnRecord]:
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    def say(self, message: str) -> ConversationResponse:
        """Send a message and get the agent's response.

        Args:
            message: The user message to send.

        Returns:
            ConversationResponse with assertion methods.
        """
        self._turn_counter += 1
        user_turn = TurnRecord(
            turn_number=self._turn_counter,
            role="user",
            content=message,
        )
        self._history.append(user_turn)

        # Build message list for the agent
        messages = [{"role": t.role, "content": t.content} for t in self._history]

        # Call the agent
        response_text = self._agent(messages)
        if not isinstance(response_text, str):
            raise TypeError(
                f"Agent must return a string, got {type(response_text).__name__}. "
                "Wrap your LLM call to return the text content."
            )

        self._turn_counter += 1
        assistant_turn = TurnRecord(
            turn_number=self._turn_counter,
            role="assistant",
            content=response_text,
        )
        self._history.append(assistant_turn)

        return ConversationResponse(
            content=response_text,
            turn_number=self._turn_counter,
            history=list(self._history),
        )
