# Review later

Open items from the 0.3.0 work. Delete this file once they are checked off.

## Verified on GitHub Actions (commit 559c3d9, real models)
The sandbox could not download the Hugging Face models, so the full suite was verified by
GitHub CI instead: **260 passed, 0 failed, 0 skipped on Python 3.10, 3.11, 3.12 and 3.13**;
core-only job 154 passed / 106 skipped (as designed); ruff, mypy and build jobs green.
That covers the tone-profile change, drift changes and the e2e tone test against the real models.

- [x] Full suite against real models
- [x] CI jobs green on first run
- [x] Tone default recalibrated on real model output: 0.7 was too strict (same-register pairs
      scored 0.66-0.94), so the default is now 0.4 (flips scored 0.0-0.15).
- [x] `refuses()` calibrated on real output (default 0.10). One soft refusal
      ("Unfortunately I'm not able to share another customer's account details.") scored only
      0.013 and is not detected; pass a lower threshold if that matters to you.
- [x] Latest full-model run: 309 passed on Python 3.10-3.13.
- [ ] Optional: sanity-check `refuses()` and the tone default on your own real conversations.

## Release (not done yet)
- [ ] Add a PyPI trusted publisher: repo `Swanand33/llm-behave`, workflow `publish.yml`,
      environment `pypi`. Create the GitHub environment `pypi`.
- [ ] After the checks above pass: publish a GitHub release tagged `v0.3.0`.
- [ ] Check the downloads badge renders once 0.3.0 is on PyPI.

## Decisions parked
- [x] Contradiction convention: kept `contradicts_turn()` as a bool predicate (no breaking change)
      and added the raising `not_contradicts_turn()` / `not_contradicts_history()`.
- [ ] Natively async provider clients (`achat` currently runs the sync call in a thread).
