"""Measure assertion latency after model warmup. Run: python tools/benchmark.py

Prints a Markdown table (median of N runs). Numbers depend on the machine, so the README
states where they were measured.
"""

from __future__ import annotations

import platform
import statistics
import time

from llm_behave import assert_behavior

RUNS = 30
OUTPUT = (
    "I'm so sorry about this issue. Let me personally handle your refund for order 1234. "
    "I'll make sure it's resolved today."
)

CASES = {
    "mentions()": lambda: assert_behavior(OUTPUT).mentions("refund"),
    "tone()": lambda: assert_behavior(OUTPUT).tone("empathetic", threshold=0.3),
    "intent()": lambda: assert_behavior(OUTPUT).intent("helping with a refund", threshold=0.1),
    "refuses() (NLI)": lambda: assert_behavior(OUTPUT).not_refuses(),
    "contradicts() (NLI)": lambda: assert_behavior("We do not offer refunds.").contradicts(
        "Refunds are available within 30 days."
    ),
    "4-assertion chain": lambda: (
        assert_behavior(OUTPUT)
        .mentions("refund")
        .tone("empathetic", threshold=0.3)
        .intent("helping with a refund", threshold=0.1)
        .not_mentions("weather", threshold=0.9)
    ),
}


def median_ms(fn) -> float:
    fn()  # warmup: loads the models
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


if __name__ == "__main__":
    print(f"Machine: {platform.platform()}, Python {platform.python_version()}, runs={RUNS}")
    print("| Assertion | Median time |")
    print("|---|---|")
    for name, fn in CASES.items():
        print(f"| `{name}` | {median_ms(fn):.0f}ms |")
