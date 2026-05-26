"""Viability probe: does fixing the absolute+anaphoric planner bug work?

The direct_v3 prompt emits `refs=[]` for queries like "in Q3 2023 after
the launch" — it drops the KNOWN absolute period because an anaphor is
present. The prompt covers deictic+anaphoric mixes but not
absolute+anaphoric.

v4 prompt: extend the anaphora rule — drop only the anaphoric part,
still emit every absolute/deictic scope the query also contains.

This probe runs both prompts on the 3 buggy D queries + sanity queries
and prints the refs. Success criterion: v4 emits the absolute period
for D_015/016/017 while leaving pure-anaphoric queries empty.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._planner_v4_probe
"""
from __future__ import annotations

import asyncio

from temporal_retrieval_v7 import NEG_INF, POS_INF
from temporal_retrieval_v7.planner_direct import PROMPT, DirectQueryPlanner

from temporal_retrieval.research._common import setup_env

setup_env()

_OLD_ANAPHORA = """ANAPHORIC EVENT REFERENCES
==========================
If the query references an event by name and you don't know its date
("since the v3 launch", "after the redesign", "before the merger"), emit
{{"refs": []}}. The downstream pipeline resolves anaphoric references
separately via semantic search."""

_NEW_ANAPHORA = """ANAPHORIC EVENT REFERENCES
==========================
An anaphoric reference names an event whose date you don't know
("since the v3 launch", "after the redesign", "before the merger").
DROP the anaphoric part — but STILL emit every absolute or deictic
temporal scope the query ALSO contains. Only emit {{"refs": []}} when
the query's ONLY temporal scope is anaphoric.

  "since the v3 launch" -> {{"refs": []}} — anaphoric is the only scope.
  "in Q3 2023 after the launch" -> emit the Q3 2023 ref; drop "after the
    launch". The absolute period Q3 2023 is known and MUST be emitted.
  "what happened in 2024 after the migration" -> emit the [2024] ref;
    drop "after the migration"."""

PROMPT_V4 = PROMPT.replace(_OLD_ANAPHORA, _NEW_ANAPHORA)


def _fmt_us(t: int) -> str:
    if t <= NEG_INF + 1:
        return "-inf"
    if t >= POS_INF - 1:
        return "+inf"
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t / 1_000_000, tz=timezone.utc).strftime("%Y-%m-%d")


def _fmt(plan) -> str:
    if plan.parse_error:
        return f"ERR:{plan.parse_error}"
    if not plan.refs:
        return f"[] extremum={plan.extremum}"
    parts = []
    for r in plan.refs:
        ivs = ",".join(f"[{_fmt_us(iv.earliest_us)},{_fmt_us(iv.latest_us)})"
                       for iv in r.intervals)
        parts.append("{" + ivs + "}")
    return f"{' '.join(parts)} extremum={plan.extremum}"


CASES = [
    # (query, ref_time, expectation)
    ("What did I do in Q3 2023 after the launch?", "2025-06-15T00:00:00Z",
     "should emit Q3 2023"),
    ("What happened in 2024 after the migration?", "2025-06-15T00:00:00Z",
     "should emit [2024]"),
    ("What I worked on in March 2024 after the freeze", "2025-06-15T00:00:00Z",
     "should emit March 2024"),
    ("Things I did in Q4 2024 before year-end review", "2025-06-15T00:00:00Z",
     "should emit Q4 2024 (already works)"),
    # Pure-anaphoric — MUST stay empty
    ("My most recent update after the migration", "2025-06-15T00:00:00Z",
     "MUST stay [] extremum=latest"),
    ("The latest thing I did since the launch", "2025-06-15T00:00:00Z",
     "MUST stay [] extremum=latest"),
    ("since the v3 launch", "2025-06-15T00:00:00Z", "MUST stay []"),
    # Sanity — absolute-only, no anaphor
    ("in March 2024", "2025-06-15T00:00:00Z", "absolute, unchanged"),
    ("what did I do recently", "2025-06-15T00:00:00Z", "deictic-empty, unchanged"),
]


async def main() -> None:
    p_v3 = DirectQueryPlanner(cache_subdir="probe_v3")
    p_v4 = DirectQueryPlanner(prompt_template=PROMPT_V4, cache_subdir="probe_v4")
    for q, rt, expect in CASES:
        plan_v3 = await p_v3.plan(q, rt)
        plan_v4 = await p_v4.plan(q, rt)
        print(f"\nQ: {q}")
        print(f"   expect: {expect}")
        print(f"   v3: {_fmt(plan_v3)}")
        print(f"   v4: {_fmt(plan_v4)}")


if __name__ == "__main__":
    asyncio.run(main())
