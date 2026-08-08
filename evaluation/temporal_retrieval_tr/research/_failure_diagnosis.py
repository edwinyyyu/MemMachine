"""Diagnose specific Duckling-both failures on per-bench queries.

For each picked bench, find queries where both-Duckling misses but
LLM-planner gets right. Show the LLM plan and the Duckling plan, plus
the gold doc.
"""
from __future__ import annotations

import asyncio
import json

from temporal_retrieval_tr.planner import QueryPlanner
from temporal_retrieval_tr.research._duckling_extractor import DucklingHTTPExtractor
from temporal_retrieval_tr.research._rule_planners import DucklingPlanner
from temporal_retrieval.research._common import setup_env
from temporal_retrieval_min.schema import from_us
from temporal_retrieval_tr.research.bench import load_bench

setup_env()


def fmt_us(us):
    try:
        return from_us(us).isoformat()[:19]
    except Exception:
        return str(us)


def fmt_targets(targets):
    out = []
    for i, t in enumerate(targets):
        for iv in t.intervals:
            out.append(f"target[{i}]: [{fmt_us(iv.earliest_us)}, {fmt_us(iv.latest_us)})")
    return out or ["(empty)"]


async def diagnose_bench(bench, llm_planner, rule_planner, n=3):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return
    docs_jsonl, queries, gold = loaded
    docs_by_id = {d["doc_id"]: d for d in docs_jsonl}

    print(f"\n=== {bench} ===\n")
    shown = 0
    for q in queries:
        qid = q["query_id"]
        ref = q["ref_time"]
        gold_ids = gold.get(qid, set())
        if not gold_ids:
            continue

        llm_plan = await llm_planner.plan(q["text"], ref)
        rule_plan = await rule_planner.plan(q["text"], ref)

        # Heuristic: queries where LLM emits structurally richer plan than rule
        llm_has_more = (
            (len(llm_plan.targets) != len(rule_plan.targets))
            or llm_plan.extremum is not None
            or any(len(t.intervals) > 1 for t in llm_plan.targets)
        )
        if not llm_has_more:
            continue

        # Show the gold doc, query, and both plans.
        gold_id = next(iter(gold_ids))
        gold_doc = docs_by_id.get(gold_id, {})
        print(f"  Q[{qid}]: {q['text']!r}")
        print(f"    ref_time = {ref}")
        print(f"    gold doc[{gold_id}] @ {gold_doc.get('ref_time','?')}: {gold_doc.get('text','?')[:120]}...")
        print(f"    LLM plan:  extremum={llm_plan.extremum}")
        for line in fmt_targets(llm_plan.targets):
            print(f"      {line}")
        print(f"    Rule plan: extremum={rule_plan.extremum}")
        for line in fmt_targets(rule_plan.targets):
            print(f"      {line}")
        print()

        shown += 1
        if shown >= n:
            break


async def main():
    llm_planner = QueryPlanner()
    rule_planner = DucklingPlanner()

    for bench in [
        "negation_temporal",
        "composition",
        "edge_conjunctive_temporal",
        "engagement_disjoint",
        "open_ended_date",
        "v7_compound_hard",
    ]:
        await diagnose_bench(bench, llm_planner, rule_planner, n=3)


if __name__ == "__main__":
    asyncio.run(main())
