# Changelog

All notable changes to llm-behave will be documented here.

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
