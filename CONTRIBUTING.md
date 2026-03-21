# Contributing to llm-behave

Thanks for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/Swanand33/llm-behave.git
cd llm-behave
pip install -e ".[semantic,dev]"
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Fast (no ML models)
python -m pytest tests/test_assertions.py tests/test_conversation.py tests/test_providers.py -v

# With coverage
python -m pytest --cov=llm_behave --cov-report=term-missing
```

## Making Changes

1. Fork the repo and create a branch from `main`
2. Make your changes with type hints on all function signatures
3. Add tests for new behavior
4. Ensure all 119+ tests pass before submitting
5. Open a pull request with a clear description of what you changed and why

## Code Conventions

- Python 3.10+, use `X | Y` union syntax (not `Union[X, Y]`)
- `from __future__ import annotations` in every file
- Pydantic v2 for data models
- Never import `torch` or `sentence_transformers` at module level — always inside functions (lazy loading)

## Reporting Bugs

Use the [bug report template](https://github.com/Swanand33/llm-behave/issues/new?template=bug_report.md).

## Suggesting Features

Use the [feature request template](https://github.com/Swanand33/llm-behave/issues/new?template=feature_request.md).
