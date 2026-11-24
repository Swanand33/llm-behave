"""Public API surface, typing marker and plugin registration."""

from __future__ import annotations

import importlib.resources

import llm_behave


def test_all_names_resolve():
    for name in llm_behave.__all__:
        assert getattr(llm_behave, name) is not None, name


def test_result_and_provider_types_exported():
    for name in ("AssertionResult", "DriftResult", "ConversationResponse", "LLMProvider"):
        assert name in llm_behave.__all__


def test_real_providers_lazily_importable():
    for name in ("OpenAIProvider", "AnthropicProvider", "OllamaProvider"):
        assert getattr(llm_behave, name).__name__ == name


def test_unknown_attribute_raises():
    import pytest

    with pytest.raises(AttributeError):
        llm_behave.DoesNotExist


def test_py_typed_marker_present():
    assert importlib.resources.files("llm_behave").joinpath("py.typed").is_file()


def test_plugin_fixtures_available(conversation, mock_provider, assert_llm):
    conv = conversation(["Hello there"])
    assert conv.say("hi").text == "Hello there"
    assert mock_provider(["x"]).chat([]) == "x"
    assert assert_llm("some text").text == "some text"
