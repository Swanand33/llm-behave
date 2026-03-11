# llm-assert — Project Context

## What This Is
`llm-assert` is a Python pytest plugin for behavioral testing of LLM applications.
It uses OFFLINE transformer models for semantic analysis — NO LLM-as-judge, NO API cost.
Key differentiators: multi-turn conversation testing, drift detection, any LLM provider.
Full project plan is in `llm-assert-project-report.pdf` (17 pages).

## Who Is Building This
Swanand from Pune, India. Solo developer. Self-described beginner but ambitious.
IMPORTANT: If Swanand starts going off-track or over-engineering, tell him directly:
"SWANAND, DONT — that's not needed right now, let's stay focused on [current task]."

## Build Phases (8-Phase Workflow)
- Phase 01 DISCOVERY (PRD) — DONE (the PDF)
- Phase 02 ARCHITECTURE — DONE (decisions below)
- Phase 03 SCAFFOLD — DONE (project structure, 25 tests passing)
- Phase 04 IMPLEMENT — IN PROGRESS (semantic engine + ML-powered assertions)
- Phase 05 INTEGRATE — pending (E2E, cross-feature tests)
- Phase 06 HARDEN — pending (security, performance, edge cases)
- Phase 07 SHIP — pending (PyPI, docs, README, launch)
- Phase 08 REVIEW — pending (postmortem, v2 planning)

## Architecture (4 Layers)
```
llm_assert/
  core/            — Public API
    assertions.py  — assert_behavior().mentions/tone/intent/calls_tool/not_mentions
    conversation.py — ConversationTest multi-turn class with recalls/contradicts_turn
    drift.py       — DriftTest baseline save/compare for CI
  engines/         — Offline ML engines (lazy loaded)
    semantic.py    — sentence-transformers (all-MiniLM-L6-v2, 384-dim embeddings)
    tone.py        — embedding similarity against tone descriptions (NOT SST-2)
    contradiction.py — NLI model (cross-encoder/nli-deberta-v3-small)
  providers/       — LLM adapters
    base.py        — LLMProvider ABC + MockProvider (for testing without real LLM)
    openai_adapter.py
    anthropic_adapter.py
    ollama_adapter.py
  plugin/          — pytest integration
    pytest_plugin.py — registered via pyproject.toml [pytest11] entry point
tests/             — Library tests itself (dogfooding)
```

## Key Design Decisions Made
1. Tone detection: Uses embedding similarity (Option A), NOT SST-2 binary classifier
2. torch + sentence-transformers are OPTIONAL: `pip install llm-assert[semantic]`
3. Models load LAZILY on first use — import time stays fast
4. ConversationTest stores state in-memory, not files
5. DriftTest baselines store model_name + model_version to detect baseline invalidation
6. Provider adapters normalize tool_calls to `{"name": ..., "arguments": ...}` format
7. MockProvider is built-in for testing without real LLM API calls

## Current State (Phase 04 — IMPLEMENT)
### What's DONE and working:
- Full project structure scaffolded
- pyproject.toml with all deps, pytest11 entry point
- assert_behavior(text).calls_tool("name") — works end-to-end
- ConversationTest builds multi-turn history correctly
- MockProvider enables testing without LLM calls
- pytest plugin auto-registers, fixtures work
- sentence-transformers installed, model downloads working
- 25 unit tests passing (no ML deps needed)

### What's IN PROGRESS:
- Integration tests written for: semantic engine, mentions, not_mentions, tone, intent,
  multi-turn recalls/contradicts_turn, drift save/compare
- Need to run full test suite (`python -m pytest tests/ -v`) and fix any failures
- Tests are in: test_semantic_engine.py, test_mentions.py, test_tone.py, test_intent.py,
  test_multi_turn.py, test_drift.py

### What's NEXT after tests pass:
- Phase 05 INTEGRATE: E2E tests, cross-feature interactions
- Phase 06 HARDEN: edge cases, performance (<200ms per assertion), error resilience
- Phase 07 SHIP: PyPI publish, full README, MkDocs, launch on HN/Reddit

## Code Conventions
- Type hints on ALL function signatures
- Python 3.10+ (use `X | Y` union syntax, not `Union[X, Y]`)
- Pydantic v2 for all data models
- `from __future__ import annotations` in every file
- Never import torch/sentence_transformers at module level — always inside functions

## Testing
- Run all: `python -m pytest tests/ -v`
- Run single: `python -m pytest tests/test_assertions.py::TestCallsTool -v`
- Run without ML (fast): `python -m pytest tests/test_assertions.py tests/test_conversation.py tests/test_providers.py -v`
- Coverage: `python -m pytest --cov=llm_assert --cov-report=term-missing`

## Environment
- Python 3.12.1, Windows (MSYS), pip, git
- Project dir: c:\Users\Swanand Potnis\Desktop\LLM-ASS
- Installed: llm-assert (editable), sentence-transformers, torch, pydantic, pytest
- Permission issue: use `pip install --user` if `pip install` fails with access denied

## Competitive Edge (From PDF)
- Only library with native multi-turn conversation testing
- Only library using offline models (zero API cost for test suite)
- Only library where behavioral tests look like normal pytest tests
- Drift detection built-in from day one
- Framework agnostic (not tied to OpenAI or LangChain)
