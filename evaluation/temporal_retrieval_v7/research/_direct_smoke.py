"""Smoke test: run the V7 direct planner on a handful of queries to
sanity-check that the output refs look right (correct intervals,
correct one-vs-many-ref structure) end-to-end.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._direct_smoke
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from temporal_retrieval_v7 import (
    NEG_INF,
    POS_INF,
    DirectQueryPlanner,
    TimeRange,
)

from temporal_retrieval.research._common import setup_env

setup_env()


def _fmt_us(t: int) -> str:
    if t <= NEG_INF + 1:
        return "-∞"
    if t >= POS_INF - 1:
        return "+∞"
    return datetime.fromtimestamp(t / 1_000_000, tz=timezone.utc).strftime("%Y-%m-%d")


def _fmt_range(r: TimeRange) -> str:
    if not r.intervals:
        return "∅"
    return " ∪ ".join(
        f"[{_fmt_us(iv.earliest_us)}, {_fmt_us(iv.latest_us)})"
        for iv in r.intervals
    )


SMOKE_CASES = [
    "in March 2024",
    "after 2020",
    "before 1999",
    "not in 2023",
    "in 2024 not in summer",
    "not in 2020 or 2022",
    "in 2020 and 2024",
    "in Q1 or Q4 of 2023",
    "between 2020 and 2024",
    "since the v3 launch",
    "what happened recently",
    "How do I plan my morning?",
    "What did NOT happen on May 3, 2024?",
    "What did I do in 2024 not in Q1?",
    "latest budget review in Q2 2024",
]


async def main():
    planner = DirectQueryPlanner()
    ref_time = "2025-01-15T12:00:00Z"

    for q in SMOKE_CASES:
        plan = await planner.plan(q, ref_time)
        print(f"\nQ: {q}")
        if plan.parse_error:
            print(f"   ERROR: {plan.parse_error}")
            continue
        if not plan.refs:
            print("   (empty plan)")
        for i, r in enumerate(plan.refs):
            print(f"   ref[{i}] = {_fmt_range(r)}")
        if plan.extremum:
            print(f"   extremum={plan.extremum}")
    print("\nstats:", planner.stats())


if __name__ == "__main__":
    asyncio.run(main())
