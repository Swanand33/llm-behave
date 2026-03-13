# Changelog

All notable changes to llm-behave will be documented here.

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
