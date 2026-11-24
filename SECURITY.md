# Security Policy

## Supported versions

Only the latest released version of `llm-behave` receives security fixes.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it privately through GitHub:
[Report a vulnerability](https://github.com/Swanand33/llm-behave/security/advisories/new).

Include the version, a description of the problem, and steps to reproduce. You can expect
an initial reply within a few days. Fixes ship in a new release and are noted in
[CHANGELOG.md](CHANGELOG.md).

## Scope notes

- `llm-behave` runs models locally and makes no network calls of its own, apart from the
  one-time Hugging Face model download and any provider adapter you choose to use.
- Drift baselines are plain JSON files in `.llm_behave_baselines/`. Baseline names are
  validated so they cannot write outside that directory. Treat baseline files from untrusted
  sources like any other test fixture.
