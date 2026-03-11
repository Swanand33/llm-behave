"""Abstract base for LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Base class for LLM provider adapters.

    Implement this to add support for any LLM provider.
    The adapter normalizes provider-specific responses into
    a standard format that llm-assert can test.
    """

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send messages and return the text response.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.

        Returns:
            The assistant's text response.
        """
        ...

    @abstractmethod
    def chat_with_tools(
        self, messages: list[dict[str, str]], tools: list[dict], **kwargs: Any
    ) -> tuple[str, list[dict[str, Any]]]:
        """Send messages with tool definitions and return response + tool calls.

        Args:
            messages: Conversation messages.
            tools: Tool definitions in provider format.

        Returns:
            Tuple of (text_response, tool_calls_list).
            tool_calls_list items: {"name": "...", "arguments": {...}}
        """
        ...


class MockProvider(LLMProvider):
    """Mock provider for testing. Returns predefined responses.

    Usage:
        mock = MockProvider(responses=["Hello!", "How can I help?"])
        output = mock.chat([{"role": "user", "content": "Hi"}])
        # Returns "Hello!", then "How can I help?" on next call
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        tool_responses: list[tuple[str, list[dict]]] | None = None,
    ) -> None:
        self._responses = list(responses or ["Mock response"])
        self._tool_responses = list(tool_responses or [])
        self._call_count = 0
        self._tool_call_count = 0

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self._call_count += 1
        return response

    def chat_with_tools(
        self, messages: list[dict[str, str]], tools: list[dict], **kwargs: Any
    ) -> tuple[str, list[dict[str, Any]]]:
        if self._tool_responses:
            idx = min(self._tool_call_count, len(self._tool_responses) - 1)
            response = self._tool_responses[idx]
            self._tool_call_count += 1
            return response
        return self.chat(messages, **kwargs), []
