"""Engines load their model once even when many threads hit them at the same time."""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np

from llm_behave.engines import contradiction as contradiction_module
from llm_behave.engines import semantic as semantic_module


def _run_threads(target, count=8):
    errors = []

    def wrapper():
        try:
            target()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=wrapper) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_semantic_model_loaded_once_under_threads(monkeypatch):
    constructed = []

    class SlowModel:
        def __init__(self, name):
            constructed.append(name)
            time.sleep(0.05)  # widen the race window

        def encode(self, text, convert_to_numpy=True):
            return np.ones(3)

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = SlowModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    engine = semantic_module.SemanticEngine("thread-test-model")
    _run_threads(lambda: engine.encode("hello"))
    assert constructed == ["thread-test-model"]


def test_contradiction_model_loaded_once_under_threads(monkeypatch):
    constructed = []

    class SlowCross:
        def __init__(self, name):
            constructed.append(name)
            time.sleep(0.05)

        def predict(self, pairs):
            return np.zeros((len(pairs), 3))

    fake = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = SlowCross
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    engine = contradiction_module.ContradictionEngine("thread-test-nli")
    _run_threads(lambda: engine.check_entailment("a", "b"))
    assert constructed == ["thread-test-nli"]


def test_getters_return_one_shared_engine(monkeypatch):
    monkeypatch.setattr(semantic_module, "_engines", {})
    monkeypatch.setattr(contradiction_module, "_engines", {})
    seen_semantic, seen_nli = set(), set()

    def grab():
        seen_semantic.add(id(semantic_module.get_semantic_engine("shared")))
        seen_nli.add(id(contradiction_module.get_contradiction_engine("shared")))

    _run_threads(grab, count=16)
    assert len(seen_semantic) == 1 and len(seen_nli) == 1
