"""Qualitative probe: feed the planner DOCUMENT text instead of query text.

Reversed-scoring hypothesis: build a temporal DNF from each doc, then
score a query's extracted intervals against the doc's DNF.

This probe is the viability gate — if the planner produces structurally
trivial output on docs (just one intersect-leaf per extracted date,
which is what the extractor already gives us), reversed scoring adds
nothing beyond the extractor.

If it produces interesting relational structure (e.g. ``after(X)`` for
a doc that says "valid from 2024 onwards"), the approach might add
signal.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._probe_doc_side_planning
"""
from __future__ import annotations

import asyncio
import json

from temporal_retrieval.planner import QueryPlanner

from ._common import setup_env

setup_env()


# Hand-picked mix of doc styles to stress the planner:
DOCS = [
    ("plain_dated_event",
     "Robin Brown was awarded employee of the month on Nov 2, 2023.",
     "2025-01-15T00:00:00Z"),
    ("dated_event_2",
     "Kim Johnson attended the company offsite in Tokyo on Mar 13, 2023.",
     "2025-01-15T00:00:00Z"),
    ("two_dates",
     "We launched v1 in Q4 2023 and replaced it with v2 in Q1 2024.",
     "2025-01-15T00:00:00Z"),
    ("validity_range",
     "Backup retention policy: applies to all data written from "
     "January 1, 2024 onwards.",
     "2025-01-15T00:00:00Z"),
    ("scoped_until",
     "Discount promotion valid until December 31, 2024.",
     "2025-01-15T00:00:00Z"),
    ("range_explicit",
     "The Q4 outage post-mortem covers events from October 15, 2023 "
     "to November 30, 2023.",
     "2025-01-15T00:00:00Z"),
    ("policy_no_anchor",
     "Backups must run within the last hour before any database "
     "schema migration is applied.",
     "2025-01-15T00:00:00Z"),
    ("era_reference",
     "Back in college I studied algorithms. I graduated in 2008.",
     "2025-01-15T00:00:00Z"),
    ("conjunctive",
     "Maya was the offsite host in March 2024, and again in March 2025.",
     "2025-01-15T00:00:00Z"),
    ("topical_no_temporal",
     "Lessons from the v3 launch postmortem.",
     "2025-01-15T00:00:00Z"),
]


def render_plan(p) -> str:
    if p.parse_error:
        return f"PARSE_ERROR: {p.parse_error}"
    if not p.expr:
        return f"(no temporal scope)  extremum={p.extremum!r}"
    clauses = []
    for clause in p.expr:
        leaves = [f"{l.phrase!r}:{l.relation}" for l in clause]
        clauses.append("AND[" + ", ".join(leaves) + "]")
    return " OR ".join(clauses) + f"  extremum={p.extremum!r}"


def categorize(p) -> str:
    """Bucket each plan to summarize structural diversity."""
    if p.parse_error or not p.expr:
        return "empty"
    n_clauses = len(p.expr)
    n_leaves_total = sum(len(c) for c in p.expr)
    relations = {l.relation for c in p.expr for l in c}
    if n_clauses == 1 and n_leaves_total == 1:
        if "intersect" in relations:
            return "single_intersect"
        return f"single_{next(iter(relations))}"
    if relations == {"intersect"} and n_clauses == 1:
        return f"AND_of_{n_leaves_total}_intersect"
    if relations == {"intersect"} and n_clauses > 1:
        return f"OR_of_{n_clauses}_intersect"
    return f"mixed({sorted(relations)})_c{n_clauses}_l{n_leaves_total}"


async def main():
    p = QueryPlanner()
    print(f"Planning {len(DOCS)} representative DOCUMENTS via the query planner.\n", flush=True)
    plans = await asyncio.gather(*[p.plan(text, ref) for _, text, ref in DOCS])
    categories: dict[str, int] = {}
    for (qid, text, _ref), plan in zip(DOCS, plans, strict=False):
        print(f"=== {qid} ===")
        print(f"  text: {text}")
        print(f"  plan: {render_plan(plan)}")
        cat = categorize(plan)
        print(f"  cat:  {cat}\n")
        categories[cat] = categories.get(cat, 0) + 1

    print("--- structural diversity summary ---")
    for cat, n in sorted(categories.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>2d}  {cat}")
    print(f"\nplanner stats: {json.dumps(p.stats(), indent=2)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
