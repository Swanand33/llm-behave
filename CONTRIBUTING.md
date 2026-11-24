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

# Core-only install (no ML extras): tests that need sentence-transformers are skipped
pip install -e . && python -m pytest tests/ -q -rs

# With coverage
python -m pytest --cov=llm_behave --cov-report=term-missing
```

## Lint and types

CI runs both on every pull request:

```bash
ruff check .
mypy llm_behave
```

## Making Changes

1. Fork the repo and create a branch from `main`
2. Make your changes with type hints on all function signatures
3. Add tests for new behavior
4. Ensure the tests, `ruff check .` and `mypy llm_behave` pass, and add a line to `CHANGELOG.md` under `[Unreleased]`
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

## Releasing (maintainers)

1. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new version heading and date.
2. Bump the version in **both** `pyproject.toml` and `llm_behave/__init__.py`.
3. Merge to `main` once CI is green.
4. Publish a GitHub release whose tag is `v<version>` (e.g. `v0.3.0`).
   `publish.yml` checks that the tag matches both version strings, runs the tests, builds,
   and uploads to PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/)
   (no API token is stored). One-time setup: on PyPI add a trusted publisher for this repo,
   workflow `publish.yml`, environment `pypi`.
