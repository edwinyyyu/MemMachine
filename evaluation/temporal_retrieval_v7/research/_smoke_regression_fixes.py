"""Quick smoke: re-run the 4 problem queries through the v2 direct
planner to validate the prompt fix.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from temporal_retrieval_v7 import DirectQueryPlanner, NEG_INF, POS_INF

from temporal_retrieval.research._common import setup_env

setup_env()


def _fmt_us(t: int) -> str:
    if t <= NEG_INF + 1:
        return "-∞"
    if t >= POS_INF - 1:
        return "+∞"
    return datetime.fromtimestamp(t / 1_000_000, tz=timezone.utc).strftime("%Y-%m-%d")


def _fmt_range(r) -> str:
    if not r.intervals:
        return "∅"
    return " ∪ ".join(
        f"[{_fmt_us(iv.earliest_us)},{_fmt_us(iv.latest_us)})"
        for iv in r.intervals
    )


CASES = [
    # (query, ref_time, expected_clauses_nonempty, expected_extremum)
    ("What issue did we hit during the migration last quarter?",
     "2025-01-15T00:00:00Z", True, None),
    ("We just had an incident - what severity is this and should I page?",
     "2024-06-24T10:00:00Z", False, None),
    ("When did we take our first Utah road trip?",
     "2026-04-23T12:00:00Z", False, None),
    ("When did I write my first novel?",
     "2026-04-23T12:00:00Z", False, None),
    # Also smoke a few prior working ones to ensure no regression
    ("in March 2024", "2025-01-15T12:00:00Z", True, None),
    ("Most recent meeting in March 2024", "2025-01-15T12:00:00Z", True, "latest"),
    ("not in 2023", "2025-01-15T12:00:00Z", True, None),
]


async def main():
    planner = DirectQueryPlanner()
    pass_n = fail_n = 0
    for q, rt, want_clauses, want_extremum in CASES:
        plan = await planner.plan(q, rt)
        has_clauses = len(plan.clauses) > 0
        clauses_ok = has_clauses == want_clauses
        extremum_ok = plan.extremum == want_extremum
        verdict = "PASS" if clauses_ok and extremum_ok else "FAIL"
        if verdict == "PASS":
            pass_n += 1
        else:
            fail_n += 1
        print(f"\n[{verdict}] Q: {q}")
        print(f"  expect: clauses_nonempty={want_clauses}, extremum={want_extremum}")
        print(f"  got:    clauses={len(plan.clauses)}, extremum={plan.extremum}")
        for ci, c in enumerate(plan.clauses):
            print(f"    [{ci}] bind={c.bind}, refs={[_fmt_range(r) for r in c.refs]}")
    print(f"\n{pass_n} PASS / {fail_n} FAIL of {pass_n + fail_n} cases")


if __name__ == "__main__":
    asyncio.run(main())
