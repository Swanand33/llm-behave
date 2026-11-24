"""Ollama provider adapter for local/offline LLM testing."""

from __future__ import annotations

from typing import Any

from llm_behave.providers.base import LLMProvider, normalize_tool_arguments


class OllamaProvider(LLMProvider):
    """Adapter for Ollama local models.

    Usage:
        provider = OllamaProvider(model="llama3")
        response = provider.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, model: str = "llama3", host: str | None = None) -> None:
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama package required. Install with: pip install llm-behave[ollama]"
            ) from None
        self._client = ollama.Client(host=host) if host else ollama.Client()
        self._model = model

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        response = self._client.chat(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        return str(response["message"]["content"])

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
        message = response["message"]
        text = message["content"] or ""
        tool_calls = []

        # `message` is a dict or an ollama response object; both support .get()
        for tc in message.get("tool_calls") or []:
            function = tc["function"]
            tool_calls.append({
                "name": function["name"],
                "arguments": normalize_tool_arguments(function["arguments"]),
            })

        return text, tool_calls
