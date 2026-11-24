"""llm-behave: Behavioral testing for LLM applications."""

from __future__ import annotations

from llm_behave.core.assertions import (
    AssertBehavior,
    AssertionResult,
    assert_behavior,
    behavioral_test,
)
from llm_behave.core.conversation import ConversationResponse, ConversationTest
from llm_behave.core.drift import DriftResult, DriftTest
from llm_behave.engines.contradiction import get_contradiction_engine
from llm_behave.providers.base import LLMProvider, MockProvider

__version__ = "0.2.1"

__all__ = [
    "assert_behavior",
    "AssertBehavior",
    "AssertionResult",
    "behavioral_test",
    "ConversationResponse",
    "ConversationTest",
    "DriftResult",
    "DriftTest",
    "get_contradiction_engine",
    "LLMProvider",
    "MockProvider",
]


def __getattr__(name: str):
    """Lazily expose the real provider adapters.

    They are imported on demand so that ``import llm_behave`` never touches the
    optional openai / anthropic / ollama SDKs.
    """
    lazy = {
        "OpenAIProvider": "llm_behave.providers.openai_adapter",
        "AnthropicProvider": "llm_behave.providers.anthropic_adapter",
        "OllamaProvider": "llm_behave.providers.ollama_adapter",
    }
    if name in lazy:
        import importlib

        return getattr(importlib.import_module(lazy[name]), name)
    raise AttributeError(f"module 'llm_behave' has no attribute {name!r}")
