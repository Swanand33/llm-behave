"""pytest plugin for llm-assert.

Registered via pyproject.toml [project.entry-points.pytest11].
Provides fixtures and hooks for behavioral testing.
"""

from __future__ import annotations

import pytest

from llm_assert.core.assertions import AssertBehavior, assert_behavior
from llm_assert.core.conversation import ConversationTest
from llm_assert.providers.base import MockProvider


@pytest.fixture
def assert_llm():
    """Fixture that provides the assert_behavior function."""
    return assert_behavior


@pytest.fixture
def mock_provider():
    """Fixture that creates a MockProvider for testing without real LLM calls."""
    def _factory(responses: list[str] | None = None, **kwargs):
        return MockProvider(responses=responses, **kwargs)
    return _factory


@pytest.fixture
def conversation(mock_provider):
    """Fixture that creates a ConversationTest with a mock provider."""
    def _factory(responses: list[str] | None = None):
        provider = MockProvider(responses=responses or ["Mock response"])
        return ConversationTest(agent=provider.chat)
    return _factory


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "behavioral: marks a test as a behavioral LLM test",
    )
    config.addinivalue_line(
        "markers",
        "drift: marks a test as a drift detection test",
    )
