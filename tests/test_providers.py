"""Tests for MockProvider and provider base class."""

from __future__ import annotations

import pytest

from llm_behave.providers.base import MockProvider


class TestMockProvider:
    """Test that MockProvider works correctly for testing."""

    def test_returns_default_response(self):
        provider = MockProvider()
        result = provider.chat([{"role": "user", "content": "Hi"}])
        assert result == "Mock response"

    def test_returns_custom_responses_in_order(self):
        provider = MockProvider(responses=["First", "Second", "Third"])
        assert provider.chat([]) == "First"
        assert provider.chat([]) == "Second"
        assert provider.chat([]) == "Third"

    def test_repeats_last_response_when_exhausted(self):
        provider = MockProvider(responses=["Only one"])
        assert provider.chat([]) == "Only one"
        assert provider.chat([]) == "Only one"

    def test_chat_with_tools_returns_empty_tool_calls(self):
        provider = MockProvider(responses=["Hello"])
        text, tools = provider.chat_with_tools([], [])
        assert text == "Hello"
        assert tools == []

    def test_chat_with_tools_returns_custom_tool_calls(self):
        tool_resp = [("Looking up", [{"name": "search", "arguments": {}}])]
        provider = MockProvider(tool_responses=tool_resp)
        text, tools = provider.chat_with_tools([], [])
        assert text == "Looking up"
        assert len(tools) == 1
        assert tools[0]["name"] == "search"


# ---------------------------------------------------------------------------
# Real provider adapters, tested against fake SDK modules (no network, no keys)
# ---------------------------------------------------------------------------

import builtins  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
from types import SimpleNamespace as NS  # noqa: E402

from llm_behave.providers.base import normalize_tool_arguments  # noqa: E402


def _install_fake(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _block_import(monkeypatch, name):
    """Make `import <name>` raise ImportError, as if the extra isn't installed."""
    monkeypatch.delitem(sys.modules, name, raising=False)
    real_import = builtins.__import__

    def fake_import(mod, *args, **kwargs):
        if mod == name:
            raise ImportError(f"No module named {name!r}")
        return real_import(mod, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


class TestNormalizeToolArguments:
    def test_json_string_becomes_dict(self):
        assert normalize_tool_arguments('{"city": "Paris"}') == {"city": "Paris"}

    def test_dict_passes_through(self):
        assert normalize_tool_arguments({"a": 1}) == {"a": 1}

    def test_empty_values(self):
        assert normalize_tool_arguments(None) == {}
        assert normalize_tool_arguments("") == {}

    def test_invalid_json_preserved_as_raw(self):
        assert normalize_tool_arguments("{oops") == {"_raw": "{oops"}

    def test_non_object_json_preserved_as_raw(self):
        assert normalize_tool_arguments("[1, 2]") == {"_raw": "[1, 2]"}

    def test_mapping_like_object(self):
        from collections import OrderedDict

        assert normalize_tool_arguments(OrderedDict(x=1)) == {"x": 1}


class TestOpenAIProvider:
    def _provider(self, monkeypatch, create):
        client = NS(chat=NS(completions=NS(create=create)))
        _install_fake(monkeypatch, "openai", OpenAI=lambda **kw: client)
        from llm_behave.providers.openai_adapter import OpenAIProvider

        return OpenAIProvider(model="test-model")

    def test_chat_returns_text(self, monkeypatch):
        seen = {}

        def create(**kwargs):
            seen.update(kwargs)
            return NS(choices=[NS(message=NS(content="Hi there", tool_calls=None))])

        provider = self._provider(monkeypatch, create)
        assert provider.chat([{"role": "user", "content": "Hi"}]) == "Hi there"
        assert seen["model"] == "test-model"

    def test_none_content_becomes_empty_string(self, monkeypatch):
        provider = self._provider(
            monkeypatch,
            lambda **kw: NS(choices=[NS(message=NS(content=None, tool_calls=None))]),
        )
        assert provider.chat([]) == ""

    def test_tool_call_arguments_are_dicts(self, monkeypatch):
        call = NS(function=NS(name="get_weather", arguments='{"city": "Paris"}'))
        provider = self._provider(
            monkeypatch,
            lambda **kw: NS(choices=[NS(message=NS(content="", tool_calls=[call]))]),
        )
        _, calls = provider.chat_with_tools([], [{"type": "function"}])
        assert calls == [{"name": "get_weather", "arguments": {"city": "Paris"}}]

    def test_missing_sdk_raises_helpful_error(self, monkeypatch):
        _block_import(monkeypatch, "openai")
        from llm_behave.providers.openai_adapter import OpenAIProvider

        with pytest.raises(ImportError, match=r"llm-behave\[openai\]"):
            OpenAIProvider()


class TestAnthropicProvider:
    def _provider(self, monkeypatch, create, **kwargs):
        client = NS(messages=NS(create=create))
        _install_fake(monkeypatch, "anthropic", Anthropic=lambda **kw: client)
        from llm_behave.providers.anthropic_adapter import AnthropicProvider

        return AnthropicProvider(model="test-model", **kwargs)

    def test_chat_joins_text_blocks(self, monkeypatch):
        content = [NS(type="text", text="Hello "), NS(type="text", text="world")]
        provider = self._provider(monkeypatch, lambda **kw: NS(content=content))
        assert provider.chat([{"role": "user", "content": "Hi"}]) == "Hello world"

    def test_default_max_tokens(self, monkeypatch):
        seen = {}

        def create(**kwargs):
            seen.update(kwargs)
            return NS(content=[])

        self._provider(monkeypatch, create).chat([])
        assert seen["max_tokens"] == 1024

    def test_max_tokens_configurable_on_init_and_per_call(self, monkeypatch):
        seen = {}

        def create(**kwargs):
            seen.update(kwargs)
            return NS(content=[])

        provider = self._provider(monkeypatch, create, max_tokens=256)
        provider.chat([])
        assert seen["max_tokens"] == 256
        provider.chat([], max_tokens=64)
        assert seen["max_tokens"] == 64

    def test_tool_use_blocks_become_tool_calls(self, monkeypatch):
        content = [
            NS(type="text", text="Checking."),
            NS(type="tool_use", name="get_weather", input={"city": "Paris"}),
        ]
        provider = self._provider(monkeypatch, lambda **kw: NS(content=content))
        text, calls = provider.chat_with_tools([], [{"name": "get_weather"}])
        assert text == "Checking."
        assert calls == [{"name": "get_weather", "arguments": {"city": "Paris"}}]

    def test_missing_sdk_raises_helpful_error(self, monkeypatch):
        _block_import(monkeypatch, "anthropic")
        from llm_behave.providers.anthropic_adapter import AnthropicProvider

        with pytest.raises(ImportError, match=r"llm-behave\[anthropic\]"):
            AnthropicProvider()


class TestOllamaProvider:
    def _provider(self, monkeypatch, chat, **kwargs):
        client = NS(chat=chat)
        _install_fake(monkeypatch, "ollama", Client=lambda **kw: client)
        from llm_behave.providers.ollama_adapter import OllamaProvider

        return OllamaProvider(model="test-model", **kwargs)

    def test_chat_returns_text(self, monkeypatch):
        provider = self._provider(
            monkeypatch, lambda **kw: {"message": {"content": "Hi from local"}}
        )
        assert provider.chat([]) == "Hi from local"

    def test_tool_calls_normalised(self, monkeypatch):
        response = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"q": "pytest"}}}
                ],
            }
        }
        provider = self._provider(monkeypatch, lambda **kw: response)
        _, calls = provider.chat_with_tools([], [{"type": "function"}])
        assert calls == [{"name": "search", "arguments": {"q": "pytest"}}]

    def test_no_tool_calls_returns_empty_list(self, monkeypatch):
        provider = self._provider(
            monkeypatch, lambda **kw: {"message": {"content": "No tools needed"}}
        )
        text, calls = provider.chat_with_tools([], [])
        assert text == "No tools needed"
        assert calls == []

    def test_none_tool_calls_returns_empty_list(self, monkeypatch):
        provider = self._provider(
            monkeypatch,
            lambda **kw: {"message": {"content": "x", "tool_calls": None}},
        )
        assert provider.chat_with_tools([], [])[1] == []

    def test_missing_sdk_raises_helpful_error(self, monkeypatch):
        _block_import(monkeypatch, "ollama")
        from llm_behave.providers.ollama_adapter import OllamaProvider

        with pytest.raises(ImportError, match=r"llm-behave\[ollama\]"):
            OllamaProvider()
