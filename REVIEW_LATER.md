# Review later

Open items from the 0.3.0 work. Delete this file once they are checked off.

## Verified on GitHub Actions (commit 559c3d9, real models)
The sandbox could not download the Hugging Face models, so the full suite was verified by
GitHub CI instead: **260 passed, 0 failed, 0 skipped on Python 3.10, 3.11, 3.12 and 3.13**;
core-only job 154 passed / 106 skipped (as designed); ruff, mypy and build jobs green.
That covers the tone-profile change, drift changes and the e2e tone test against the real models.

- [x] Full suite against real models
- [x] CI jobs green on first run
- [ ] Optional: eyeball tone thresholds on your own conversations. The default 0.7 for
      `consistent_tone_across_turns()` passes the repo's tests but is untuned for real traffic.

## Release (not done yet)
- [ ] Add a PyPI trusted publisher: repo `Swanand33/llm-behave`, workflow `publish.yml`,
      environment `pypi`. Create the GitHub environment `pypi`.
- [ ] After the checks above pass: publish a GitHub release tagged `v0.3.0`.
- [ ] Check the downloads badge renders once 0.3.0 is on PyPI.

## Decisions parked
- [ ] `ConversationResponse.contradicts_turn()` returns a bool while `recalls()` and
      `consistent_tone_across_turns()` raise `AssertionError`. Pick one convention.
- [ ] Which further assertions to add (model-free length/word-count/regex checks are done). Candidates: `refuses()`, `not_contradicts_history`, async provider support.
