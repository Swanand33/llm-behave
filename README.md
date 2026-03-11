# llm-assert

Behavioral testing for LLM applications. A pytest plugin.

**No LLM judge. Offline models. Millisecond assertions.**

```python
from llm_assert import assert_behavior

output = my_ai("I want a refund for order 1234")
assert_behavior(output).mentions("refund policy")
assert_behavior(output).tone("empathetic")
assert_behavior(output).not_mentions("competitor")
```

## Install

```bash
pip install llm-assert
```

## Status

Under active development. V1 coming soon.
