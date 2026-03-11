"""Anthropic (Claude) provider adapter."""

from __future__ import annotations

from typing import Any

from llm_assert.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Adapter for Anthropic's messages API.

    Usage:
        provider = AnthropicProvider(model="claude-sonnet-4-6")
        response = provider.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, model: str = "claude-sonnet-4-6", **client_kwargs: Any) -> None:
        try:
            import anthropic

            self._client = anthropic.Anthropic(**client_kwargs)
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install llm-assert[anthropic]"
            )
        self._model = model

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        response = self._client.messages.create(
            model=self._model,
            messages=messages,
            max_tokens=1024,
            **kwargs,
        )
        # Extract text from content blocks
        text_parts = [
            block.text for block in response.content if hasattr(block, "text")
        ]
        return "".join(text_parts)

    def chat_with_tools(
        self, messages: list[dict[str, str]], tools: list[dict], **kwargs: Any
    ) -> tuple[str, list[dict[str, Any]]]:
        response = self._client.messages.create(
            model=self._model,
            messages=messages,
            tools=tools,
            max_tokens=1024,
            **kwargs,
        )

        text_parts = []
        tool_calls = []

        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "arguments": block.input,
                })

        return "".join(text_parts), tool_calls
