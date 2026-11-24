# Review later

Open items from the 0.3.0 work. Delete this file once they are checked off.

## Needs a machine that can download Hugging Face models
The cloud sandbox could not download `all-MiniLM-L6-v2` or
`cross-encoder/nli-deberta-v3-small` (the CDN host `us.aws.cdn.hf.co` was blocked), so
**106 model-dependent tests were skipped or failing there, not verified.**

- [ ] `pip install -e ".[semantic,dev]"` then `pytest` locally; fix anything real.
- [ ] Tone: `consistent_tone_across_turns()` now compares tone profiles
      (`ToneEngine.tone_similarity`). Only tested with a fake encoder. Check the default
      threshold 0.7 and the existing e2e test (threshold 0.5) against the real model.
- [ ] Drift: `DriftTest` changes were tested with a fake engine only.
- [ ] CI on GitHub (first run after merge): confirm ruff, mypy, core-only, 3.10-3.13 and
      build jobs are green; raise `LLM_BEHAVE_PERF_FACTOR` in `test.yml` if timing tests flake.

## Release (not done yet)
- [ ] Add a PyPI trusted publisher: repo `Swanand33/llm-behave`, workflow `publish.yml`,
      environment `pypi`. Create the GitHub environment `pypi`.
- [ ] After the checks above pass: publish a GitHub release tagged `v0.3.0`.
- [ ] Check the downloads badge renders once 0.3.0 is on PyPI.

## Decisions parked
- [ ] `ConversationResponse.contradicts_turn()` returns a bool while `recalls()` and
      `consistent_tone_across_turns()` raise `AssertionError`. Pick one convention.
- [ ] Which new assertions to add beyond the model-free length/format checks.
