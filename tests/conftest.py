"""Shared test configuration.

Tests that need the optional ``[semantic]`` extra (sentence-transformers) are
skipped, not failed, when it isn't installed -- so a core-only install can run
the whole suite and only exercise what it actually supports.
"""

from __future__ import annotations

import pytest

_SEMANTIC_HINT = "llm-behave[semantic]"


def _skip_if_semantic_missing(exc: ImportError) -> None:
    if _SEMANTIC_HINT in str(exc):
        pytest.skip(f"needs the semantic extra: pip install {_SEMANTIC_HINT}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    # Covers fixtures that load the model (e.g. warm-up fixtures)
    try:
        return (yield)
    except ImportError as exc:
        _skip_if_semantic_missing(exc)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except ImportError as exc:
        _skip_if_semantic_missing(exc)
        raise
