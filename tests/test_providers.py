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
