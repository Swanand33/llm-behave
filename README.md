# llm-behave

**Behavioral testing for LLM applications. A pytest plugin.**

No LLM judge. No API cost. Offline models. Works with any provider.

```python
def test_support_bot():
    output = my_support_bot("I want a refund for order 1234")

    assert_behavior(output) \
        .mentions("refund policy") \
        .tone("empathetic") \
        .not_mentions("competitor")
```

---

## Why llm-behave?

Most LLM testing tools either:
- Use another LLM to judge the output (expensive, slow, circular)
- Only do exact string matching (misses semantic meaning)
- Don't support multi-turn conversations at all

**llm-behave** uses small offline transformer models (80MB, runs on CPU) to understand meaning — no API calls, no cost, no internet required during tests.

---

## Install

```bash
# Core (tool call assertions, structure tests — no ML deps)
pip install llm-behave

# Full (semantic assertions: mentions, tone, intent, drift)
pip install llm-behave[semantic]
```

---

## Features at a glance

| Feature | What it does |
|---|---|
| `mentions()` | Semantic similarity — "money back" matches "refund" |
| `not_mentions()` | Assert a topic is NOT brought up |
| `tone()` | Detect empathetic / professional / rude / helpful etc. |
| `intent()` | Does the response intend to help? refuse? apologize? |
| `calls_tool()` | Assert which tool the LLM called |
| `contradicts()` | Assert the output contradicts a reference statement (NLI) |
| `ConversationTest` | Multi-turn testing with memory, contradiction detection |
| `DriftTest` | Save baseline behavior, detect regressions in CI |

---

## Usage

### Basic assertions

```python
from llm_behave import assert_behavior

output = my_llm("I want a refund")

# Semantic match — not exact string
assert_behavior(output).mentions("refund policy")

# Tone detection
assert_behavior(output).tone("empathetic")
assert_behavior(output).tone("professional", threshold=0.6)

# Intent
assert_behavior(output).intent("offering to help the customer")

# Negative assertions
assert_behavior(output).not_mentions("competitor")

# Fluent chaining
assert_behavior(output) \
    .mentions("refund") \
    .tone("empathetic") \
    .not_mentions("competitor")
```

### Contradiction assertions

Assert that an output contradicts a previous statement — useful for detecting policy reversals across conversation turns.

```python
# Turn 1 said refunds are available. Does turn 3 contradict that?
assert_behavior(turn_3_response).contradicts("Refunds are always available within 30 days.")
```

### Tool call assertions

```python
text, tool_calls = my_llm.chat_with_tools(messages, tools=my_tools)

assert_behavior(text, tool_calls) \
    .calls_tool("lookup_order") \
    .mentions("order")
```

### Multi-turn conversation testing

```python
from llm_behave import ConversationTest, MockProvider

conv = ConversationTest(agent=my_agent)

conv.say("Hi, my name is Alex")
conv.say("I placed order #5678 last week")
response = conv.say("When will it arrive?")

# Does it remember context from earlier turns?
assert response.recalls("order")
assert response.recalls("Alex")

# Is tone consistent across the whole conversation?
assert response.consistent_tone_across_turns(threshold=0.6)
```

### Drift detection (for CI)

Catch silent regressions when you update your model or prompts.

```python
from llm_behave import DriftTest

# First run: save baseline
@DriftTest.baseline(save_as="support_refund_flow")
def get_baseline_output():
    return my_llm("I need a refund")

# Every CI run: compare against baseline
result = DriftTest.compare("support_refund_flow", current_output)
assert result.passed, f"Behavior drift detected: {result.details}"
```

### pytest fixtures (auto-registered)

```python
# These fixtures are available in any test file automatically

def test_with_mock(mock_provider, assert_llm):
    provider = mock_provider(responses=["I'll help with your refund right away."])
    output = provider.chat([{"role": "user", "content": "refund please"}])
    assert_llm(output).mentions("refund").tone("helpful")

def test_conversation(conversation):
    conv = conversation(responses=["Hello!", "Sure, I can help with that."])
    conv.say("Hi")
    response = conv.say("I need help")
    assert "help" in response.text.lower()
```

---

## Providers

Built-in adapters for all major LLM providers:

```python
from llm_behave.providers.openai_adapter import OpenAIProvider
from llm_behave.providers.anthropic_adapter import AnthropicProvider
from llm_behave.providers.ollama_adapter import OllamaProvider
from llm_behave import MockProvider  # for tests, no API calls

# All providers have the same interface
provider = OpenAIProvider(model="gpt-4o-mini")
provider = AnthropicProvider(model="claude-haiku-4-5")
provider = OllamaProvider(model="llama3")

output = provider.chat([{"role": "user", "content": "Hello"}])
text, tool_calls = provider.chat_with_tools(messages, tools=my_tools)
```

Bring your own provider by subclassing `LLMProvider`:

```python
from llm_behave.providers.base import LLMProvider

class MyProvider(LLMProvider):
    def chat(self, messages, **kwargs):
        ...
    def chat_with_tools(self, messages, tools, **kwargs):
        ...
```

---

## How it works

llm-behave uses [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — an 80MB sentence-transformer model that runs fully offline on CPU.

- **`mentions()` / `not_mentions()`** — splits text into sentences, computes max cosine similarity between any sentence and your concept
- **`tone()`** — batch-encodes input text against example sentences for each tone, returns max similarity
- **`intent()`** — semantic similarity between output and your intent description
- **`contradicts()`** — NLI (Natural Language Inference) model detects if the output contradicts a reference statement
- **`contradicts_turn()`** — same NLI model, applied across conversation turns

Models load **lazily** on first use and are cached for the rest of the test session. Import time stays fast.

---

## Performance

Measured after model warmup (model loads once per test session):

| Assertion | Time |
|---|---|
| `mentions()` | ~32ms |
| `tone()` | ~40ms |
| `intent()` | ~32ms |
| 4-assertion chain | ~350ms |

---

## pytest markers

```python
import pytest

@pytest.mark.behavioral
def test_refund_flow():
    ...

@pytest.mark.drift
def test_no_regression():
    ...
```

Run only behavioral tests:
```bash
pytest -m behavioral
pytest -m drift
```

---

## Full install options

```bash
pip install llm-behave                          # core only
pip install llm-behave[semantic]                # + sentence-transformers + torch
pip install llm-behave[openai]                  # + openai SDK
pip install llm-behave[anthropic]               # + anthropic SDK
pip install llm-behave[ollama]                  # + ollama SDK
pip install llm-behave[all]                     # everything
```

---

## Requirements

- Python 3.10+
- pytest 7.0+
- For semantic assertions: `pip install llm-behave[semantic]`

---

## License

MIT — free to use in personal and commercial projects.

---

## Author

Built by [Swanand Potnis](https://github.com/Swanand33) — Pune, India.
