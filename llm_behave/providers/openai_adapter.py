"""OpenAI provider adapter."""

from __future__ import annotations

from typing import Any

from llm_behave.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """Adapter for OpenAI's chat completions API.

    Usage:
        provider = OpenAIProvider(model="gpt-4o-mini")
        response = provider.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, model: str = "gpt-4o-mini", **client_kwargs: Any) -> None:
        try:
            import openai

            self._client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install llm-behave[openai]"
            )
        self._model = model

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def chat_with_tools(
        self, messages: list[dict[str, str]], tools: list[dict], **kwargs: Any
    ) -> tuple[str, list[dict[str, Any]]]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            **kwargs,
        )
        message = response.choices[0].message
        text = message.content or ""

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })

        return text, tool_calls
