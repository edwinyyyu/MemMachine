"""Adversarial probe: queries where TREE *should* differ from DNF.

The point: if every query folds to a flat single-leaf or single AND/OR,
the tree planner has no expressive advantage. Tree's edge would come
from nested-negation and mixed boolean structures that DNF must
distribute (potentially poorly).

Each query is designed to either:
  (a) Force a nested Not(Or(...)) where DNF must distribute into multiple
      AND-of-disjoints clauses, OR
  (b) Express (A or B) and (C or D) where DNF must distribute into 4
      AND clauses, OR
  (c) Have implicit negation around a complex shape where the LLM has
      to choose between deep disjoint-leaves vs a single Not node.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._probe_planner_tree_adversarial
"""
from __future__ import annotations

import asyncio
import json

from temporal_retrieval.planner import QueryPlanner
from temporal_retrieval.planner_tree import And, Leaf, Not, Or, TreePlanner

from ._common import setup_env

setup_env()


PROBES = [
    # (a) Nested negation across multiple targets.
    ("nested_not_or",
     "What did I work on in 2024 but not in Q1 or Q2?",
     "2025-01-15T12:00:00Z"),

    ("nested_not_or_v2",
     "Show me changes from 2023, excluding January and February",
     "2024-04-01T12:00:00Z"),

    # (b) Mixed (A or B) and (C or D).
    ("mixed_or_and",
     "What did I do in Q1 or Q4 of 2023, in either January or December",
     "2024-03-01T12:00:00Z"),

    ("mixed_year_or_month_or",
     "Events from 2022 or 2023, in March or April",
     "2024-07-01T12:00:00Z"),

    # (c) Negation of a temporal range with anchor cue.
    ("not_after_event",
     "Updates not after the launch, but in 2023",
     "2024-06-01T12:00:00Z"),

    # (a) Three-way disjunction in scope, negated.
    ("triple_negation",
     "What happened outside of summer, fall, and winter 2023",
     "2024-04-01T12:00:00Z"),

    # (b) Complex bracketing with explicit nesting cues.
    ("explicit_grouping",
     "Either in 2022 during the migration, or in 2023 during the redesign",
     "2024-01-15T12:00:00Z"),

    # (c) Negation of a multi-leaf positive group.
    ("not_complex",
     "Notes from 2024, but not from the summer offsite or the fall retreat",
     "2025-01-15T12:00:00Z"),
]


def _render_tree(node) -> str:
    if node is None:
        return "(no temporal scope)"
    if isinstance(node, Leaf):
        return f"leaf({node.phrase!r}:{node.relation})"
    if isinstance(node, And):
        return "AND[" + ", ".join(_render_tree(c) for c in node.children) + "]"
    if isinstance(node, Or):
        return "OR[" + ", ".join(_render_tree(c) for c in node.children) + "]"
    if isinstance(node, Not):
        return "NOT(" + _render_tree(node.child) + ")"
    return repr(node)


def _count_nodes(node) -> tuple[int, int, int, int]:
    """Returns (n_leaf, n_and, n_or, n_not) recursively."""
    if node is None:
        return (0, 0, 0, 0)
    if isinstance(node, Leaf):
        return (1, 0, 0, 0)
    if isinstance(node, (And, Or)):
        n = [0, 0, 0, 0]
        for c in node.children:
            cc = _count_nodes(c)
            for i in range(4):
                n[i] += cc[i]
        if isinstance(node, And):
            n[1] += 1
        else:
            n[2] += 1
        return tuple(n)  # type: ignore
    if isinstance(node, Not):
        cc = _count_nodes(node.child)
        return (cc[0], cc[1], cc[2], cc[3] + 1)
    return (0, 0, 0, 0)


async def probe_one(qid: str, query: str, ref_time: str,
                    dnf: QueryPlanner, tree: TreePlanner) -> None:
    dnf_plan, tree_plan = await asyncio.gather(
        dnf.plan(query, ref_time),
        tree.plan(query, ref_time),
    )
    print(f"\n=== {qid}: {query!r}  (ref={ref_time}) ===", flush=True)

    if dnf_plan.parse_error:
        print(f"  DNF  parse_error: {dnf_plan.parse_error}", flush=True)
    elif not dnf_plan.expr:
        print(f"  DNF  : (no temporal scope) extremum={dnf_plan.extremum!r}", flush=True)
    else:
        clauses = []
        for clause in dnf_plan.expr:
            leaves = [f"{l.phrase!r}:{l.relation}" for l in clause]
            clauses.append("AND[" + ", ".join(leaves) + "]")
        dnf_repr = " OR ".join(clauses)
        dnf_leaf_count = sum(len(c) for c in dnf_plan.expr)
        dnf_clause_count = len(dnf_plan.expr)
        print(f"  DNF  : {dnf_repr}", flush=True)
        print(f"         extremum={dnf_plan.extremum!r}  "
              f"clauses={dnf_clause_count} total_leaves={dnf_leaf_count}",
              flush=True)

    if tree_plan.parse_error:
        print(f"  TREE parse_error: {tree_plan.parse_error}", flush=True)
    elif tree_plan.expr is None:
        print(f"  TREE : (no temporal scope) extremum={tree_plan.extremum!r}", flush=True)
    else:
        tree_repr = _render_tree(tree_plan.expr)
        n_leaf, n_and, n_or, n_not = _count_nodes(tree_plan.expr)
        print(f"  TREE : {tree_repr}", flush=True)
        print(f"         extremum={tree_plan.extremum!r}  "
              f"leaves={n_leaf} and={n_and} or={n_or} not={n_not}", flush=True)


async def main():
    print(f"Adversarial probe: {len(PROBES)} queries designed to exercise "
          "tree's expressive advantage over DNF", flush=True)
    dnf = QueryPlanner()
    tree = TreePlanner()
    for qid, query, ref_time in PROBES:
        await probe_one(qid, query, ref_time, dnf, tree)

    print("\n--- planner stats ---", flush=True)
    print(f"DNF : {json.dumps(dnf.stats(), indent=2)}", flush=True)
    print(f"TREE: {json.dumps(tree.stats(), indent=2)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
