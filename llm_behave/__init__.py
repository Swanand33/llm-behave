"""llm-behave: Behavioral testing for LLM applications."""

from llm_behave.core.assertions import AssertBehavior, assert_behavior, behavioral_test
from llm_behave.core.conversation import ConversationTest
from llm_behave.core.drift import DriftTest
from llm_behave.engines.contradiction import get_contradiction_engine
from llm_behave.providers.base import MockProvider

__version__ = "0.1.2"

__all__ = [
    "assert_behavior",
    "AssertBehavior",
    "behavioral_test",
    "ConversationTest",
    "DriftTest",
    "get_contradiction_engine",
    "MockProvider",
]
