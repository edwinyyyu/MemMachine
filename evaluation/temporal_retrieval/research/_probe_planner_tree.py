"""Quick qualitative probe: TreePlanner output on 10 critical queries.

Compares DNF vs Tree plans side-by-side to verify the tree planner
produces sensible structure on representative queries before relying
on the full 35-bench A/B.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._probe_planner_tree
"""
from __future__ import annotations

import asyncio
import json

from temporal_retrieval.planner import QueryPlanner
from temporal_retrieval.planner_tree import TreePlanner

from ._common import setup_env

setup_env()


# Critical queries chosen to exercise: single leaf, AND, OR, NOT,
# composition (relative phrase inside larger window), event anchor,
# extremum, no-temporal-scope, anaphoric ref.
PROBES = [
    ("simple_leaf",       "What did I work on in Q4 2023?",                              "2024-01-15T12:00:00Z"),
    ("relative_deictic",  "What happened last quarter?",                                   "2024-04-15T12:00:00Z"),
    ("after_leaf",        "Updates after the migration shipped",                          "2024-06-01T12:00:00Z"),
    ("composition_and",   "in 2024 not in summer",                                         "2025-01-01T12:00:00Z"),
    ("explicit_or",       "in Q1 or Q4 of 2023",                                           "2024-03-01T12:00:00Z"),
    ("event_anchor",      "Four days after Election Day 2020, what state did AP call?",   "2024-01-01T12:00:00Z"),
    ("extremum_latest",   "What was my latest budget review",                              "2024-09-01T12:00:00Z"),
    ("no_scope",          "Notes from the team retreat",                                   "2024-07-01T12:00:00Z"),
    ("policy_skip",       "What happens during the day in a beehive?",                    "2024-07-01T12:00:00Z"),
    ("excl_with_period",  "Recent changes in Q1 2024 excluding February",                  "2024-05-01T12:00:00Z"),
]


async def probe_one(qid: str, query: str, ref_time: str,
                    dnf: QueryPlanner, tree: TreePlanner) -> None:
    dnf_plan, tree_plan = await asyncio.gather(
        dnf.plan(query, ref_time),
        tree.plan(query, ref_time),
    )
    print(f"\n=== {qid}: {query!r}  (ref={ref_time}) ===", flush=True)

    if dnf_plan.parse_error:
        print(f"  DNF  parse_error: {dnf_plan.parse_error}", flush=True)
    else:
        # Compact DNF rendering
        if not dnf_plan.expr:
            dnf_repr = "(no temporal scope)"
        else:
            clauses = []
            for clause in dnf_plan.expr:
                leaves = [f"{l.phrase!r}:{l.relation}" for l in clause]
                clauses.append("AND[" + ", ".join(leaves) + "]")
            dnf_repr = " OR ".join(clauses)
        print(f"  DNF  : {dnf_repr}", flush=True)
        print(f"         extremum={dnf_plan.extremum!r}", flush=True)

    if tree_plan.parse_error:
        print(f"  TREE parse_error: {tree_plan.parse_error}", flush=True)
    else:
        tree_repr = _render_tree(tree_plan.expr)
        print(f"  TREE : {tree_repr}", flush=True)
        print(f"         extremum={tree_plan.extremum!r}", flush=True)


def _render_tree(node) -> str:
    if node is None:
        return "(no temporal scope)"
    from temporal_retrieval.planner_tree import Leaf, And, Or, Not
    if isinstance(node, Leaf):
        return f"leaf({node.phrase!r}:{node.relation})"
    if isinstance(node, And):
        return "AND[" + ", ".join(_render_tree(c) for c in node.children) + "]"
    if isinstance(node, Or):
        return "OR[" + ", ".join(_render_tree(c) for c in node.children) + "]"
    if isinstance(node, Not):
        return "NOT(" + _render_tree(node.child) + ")"
    return repr(node)


async def main():
    print(f"Probing {len(PROBES)} queries — DNF vs TREE planners side-by-side", flush=True)
    dnf = QueryPlanner()
    tree = TreePlanner()
    for qid, query, ref_time in PROBES:
        await probe_one(qid, query, ref_time, dnf, tree)
    print("\n--- planner stats ---", flush=True)
    print(f"DNF : {json.dumps(dnf.stats(), indent=2)}", flush=True)
    print(f"TREE: {json.dumps(tree.stats(), indent=2)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
