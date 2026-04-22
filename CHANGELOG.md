# Changelog

All notable changes to llm-behave will be documented here.

## [0.1.2] — 2026-04-22

### Added
- `.contradicts(other_text)` assertion on `AssertBehavior` — uses the NLI engine to assert that the output contradicts a reference statement. Useful for detecting policy reversals across turns.
- `get_contradiction_engine` exported from top-level `llm_behave` package.
- `max_outputs` parameter on `@DriftTest.baseline()` (default 20) — baseline file now acts as a ring buffer, evicting oldest outputs once the cap is reached.

### Fixed
- `DriftBaseline.embeddings` was always `None`; `_save_baseline()` now computes and stores embeddings for every output on each save.
- `get_semantic_engine()` replaced a previous model when called with a different `model_name` argument. It now maintains a per-model cache so multiple models can coexist.
- Assertion default thresholds extracted to named constants (`_DEFAULT_MENTIONS`, `_DEFAULT_NOT_MENTIONS`, `_DEFAULT_INTENT`, `_DEFAULT_TONE`, `_DEFAULT_CONTRADICTS`) with documentation explaining why they differ.

## [0.1.1] — 2026-03-14

### Fixed
- v0.1.0 wheel shipped with package folder named `llm_assert` (old internal name) instead of `llm_behave`, causing `ModuleNotFoundError` on all imports. Rebuilt wheel with correct `llm_behave` package structure.

## [0.1.0] — 2026-03-12

First public release.

### Added
- `assert_behavior(text)` — fluent assertion interface for LLM outputs
- `.mentions(concept)` — semantic similarity using sentence-level max comparison
- `.not_mentions(concept)` — assert a topic is absent
- `.tone(label)` — detect empathetic / professional / rude / helpful / formal / casual etc.
- `.intent(description)` — match output intent against a natural language description
- `.calls_tool(name)` — assert a specific tool was called (supports OpenAI + Anthropic formats)
- `ConversationTest` — multi-turn conversation harness with full history tracking
- `.recalls(concept)` — assert response references a concept from earlier in the conversation
- `.contradicts_turn(n)` — NLI-based contradiction detection across turns
- `.consistent_tone_across_turns()` — tone consistency check across all assistant turns
- `DriftTest` — baseline save and compare for CI regression detection
- `@DriftTest.baseline()` — decorator to save test output as baseline
- `MockProvider` — built-in mock for testing without real LLM API calls
- `OpenAIProvider` — adapter for OpenAI chat completions
- `AnthropicProvider` — adapter for Anthropic messages API
- `OllamaProvider` — adapter for Ollama local models
- pytest fixtures: `assert_llm`, `mock_provider`, `conversation`
- pytest markers: `behavioral`, `drift`
- Offline ML engine: `all-MiniLM-L6-v2` (80MB, CPU, no internet needed)
- Batch encoding for tone detection (~4x faster than sequential)
- Lazy model loading — fast import, model loads on first assertion
