# Changelog

All notable changes to llm-behave will be documented here.

## [Unreleased]

## [0.3.0] — 2026-11-24

### Fixed
- **OpenAI tool-call `arguments` are now dicts**, like Anthropic and Ollama (they were a raw JSON string). Unparseable arguments are kept under `"_raw"`.
- **`consistent_tone_across_turns()` now compares tone, not topic.** `ToneEngine.tone_similarity()` scores each reply against every tone and compares the centred profiles. It also no longer counts the latest reply twice. Thresholds tuned against the old behaviour may need re-tuning; validate against your own conversations.
- **Drift baselines:** a new `max_outputs` now applies to an existing baseline; `@DriftTest.baseline` keeps the wrapped function's name and docstring; baseline names that could escape the directory (`../x`, `a/b`) are rejected; writes are atomic; a corrupt baseline file raises a clear `ValueError`.

### Added
- `AnthropicProvider(max_tokens=...)`, also overridable per call.
- Mocked-SDK tests for the OpenAI, Anthropic and Ollama adapters.
- `py.typed` marker and `Typing :: Typed` / Python 3.13 classifiers.
- `AssertionResult`, `DriftResult`, `ConversationResponse`, `LLMProvider` exported from `llm_behave`; `OpenAIProvider`, `AnthropicProvider`, `OllamaProvider` importable from it lazily.
- Tests that need the `[semantic]` extra are skipped, not failed, on a core-only install.

### Changed
- CI now runs ruff, mypy, a core-only install job, Python 3.10–3.13 with coverage, and a build + `twine check` job. Timing assertions scale with `LLM_BEHAVE_PERF_FACTOR` (CI sets 5).
- Releases publish through PyPI trusted publishing (no stored token), after checking the tag matches `pyproject.toml` and `__version__`.
- README: replaced the hand-typed test-count badge with a live CI badge, added a downloads badge, and clarified how `DriftTest.baseline` accumulates outputs.

### Removed
- Empty `llm_behave/plugin/fixtures.py`.

## [0.2.1] — 2026-10-26

### Fixed
- **`contradicts()` / `contradicts_turn()` now score a real probability.** `ContradictionEngine.check_contradiction()` returned the NLI model's raw logit (e.g. `6.55` / `-4.34`), not a 0–1 score, so thresholds like `0.5` and `0.7` only worked by accident. It is now softmaxed over [contradiction, entailment, neutral], matching `max_entailment()`. If you passed a custom `threshold` tuned against the old logits, re-tune it.
- **Sentence splitting no longer breaks on every period.** `"$4.99"`, `"v2.1"`, `"example.com"` and `"e.g."` used to be split into fragments that `grounded_in()` / `no_hallucination()` then scored as separate claims. Sentences now end only at `.`/`!`/`?` followed by whitespace, plus blank lines and list items. Shared by the assertions and `SemanticEngine`.
- `ConversationResponse.recalls()` scores the concept per sentence (max), so long responses are no longer diluted by unrelated sentences.
- `get_contradiction_engine()` caches one engine per model name instead of replacing a single instance (same fix `get_semantic_engine()` got in 0.1.2).
- `DriftTest.compare()` raises `ValueError` when the baseline was recorded with a different embedding model, instead of silently comparing incompatible scores.

### Changed
- `DriftTest.compare()` reuses the embeddings stored in the baseline instead of re-encoding every output; saving a baseline encodes only the new output.
- `SemanticEngine.top_sentence_similarities()` encodes the query and all sentences in one batched forward pass.
- CI runs the test suite on Python 3.10, 3.11 and 3.12.

### Known limitations
- `consistent_tone_across_turns()` compares whole-response embeddings, so it measures overall semantic similarity (topic as well as tone). A dedicated tone comparison is planned.

## [0.2.0] — 2026-07-02

### Added
- **RAG / grounding assertions** (offline NLI entailment — no judge LLM, no API):
  - `.grounded_in(context)` — assert every claim in the output is entailed by the retrieved context.
  - `.no_hallucination(context)` — assert the output introduces no claims unsupported by the context; failure output lists the specific hallucinated sentences.
  - `.cites(source_id)` — assert the output references a given source id / URL / marker (case-insensitive).
- **Structured-output assertions** (pure Python, no model needed):
  - `.returns_valid_json()` — parse output as JSON, tolerating ```json fenced blocks.
  - `.matches_schema(pydantic_model)` — validate output JSON against a Pydantic model.
  - `.has_keys([...])` — assert the output JSON object contains the given keys.
- `ContradictionEngine.max_entailment(hypothesis, premises)` and `.check_entailment(premise, hypothesis)` — batched NLI entailment scoring (softmax over [contradiction, entailment, neutral]).
- `SemanticEngine.top_sentence_similarities(text, query, k)` — returns the top-k (sentence, score) pairs.

### Changed
- `.mentions()`, `.not_mentions()`, and `.intent()` failure messages now show the **top-3 closest sentences with their scores**, answering "was it 0.41 or 0.05?" instead of reporting a lone max.

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

[Unreleased]: https://github.com/Swanand33/llm-behave/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Swanand33/llm-behave/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Swanand33/llm-behave/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Swanand33/llm-behave/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/Swanand33/llm-behave/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Swanand33/llm-behave/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Swanand33/llm-behave/releases/tag/v0.1.0
