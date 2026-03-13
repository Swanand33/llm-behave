"""Ollama provider adapter for local/offline LLM testing."""

from __future__ import annotations

from typing import Any

from llm_behave.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    """Adapter for Ollama local models.

    Usage:
        provider = OllamaProvider(model="llama3")
        response = provider.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, model: str = "llama3", host: str | None = None) -> None:
        try:
            import ollama

            self._client = ollama.Client(host=host) if host else ollama.Client()
        except ImportError:
            raise ImportError(
                "ollama package required. Install with: pip install llm-behave[ollama]"
            )
        self._model = model

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        response = self._client.chat(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        return response["message"]["content"]

    def chat_with_tools(
        self, messages: list[dict[str, str]], tools: list[dict], **kwargs: Any
    ) -> tuple[str, list[dict[str, Any]]]:
        # Ollama tool calling support varies by model
        response = self._client.chat(
            model=self._model,
            messages=messages,
            tools=tools,
            **kwargs,
        )
        text = response["message"]["content"]
        tool_calls = []

        if "tool_calls" in response.get("message", {}):
            for tc in response["message"]["tool_calls"]:
                tool_calls.append({
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                })

        return text, tool_calls
