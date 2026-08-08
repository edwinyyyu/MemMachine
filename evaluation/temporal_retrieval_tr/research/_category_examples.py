"""Show representative queries from natural-conversational and stress
bench categories, with LLM-plan and rule-plan output side-by-side.
"""
from __future__ import annotations

import asyncio

from temporal_retrieval_tr.planner import QueryPlanner
from temporal_retrieval_tr.research._duckling_extractor import DucklingHTTPExtractor
from temporal_retrieval_tr.research._rule_planners import DucklingPlanner
from temporal_retrieval.research._common import setup_env
from temporal_retrieval_min.schema import from_us
from temporal_retrieval_tr.research.bench import load_bench

setup_env()


def fmt_us(us):
    try:
        return from_us(us).isoformat()[:16]
    except Exception:
        return str(us)


def fmt_plan(plan):
    if not plan.targets and plan.extremum is None:
        return "  (empty plan)"
    lines = []
    if plan.extremum:
        lines.append(f"  extremum={plan.extremum}")
    for i, t in enumerate(plan.targets):
        ivs = []
        for iv in t.intervals:
            ivs.append(f"[{fmt_us(iv.earliest_us)}, {fmt_us(iv.latest_us)})")
        lines.append(f"  target[{i}]: {', '.join(ivs)}")
    return "\n".join(lines)


CATEGORIES = {
    "NATURAL CONVERSATIONAL  (LoCoMo-style)": [
        ("realq", 2),
        ("realq_v2", 2),
        ("mixed_cue", 2),
        ("cotemporal", 1),
        ("dense_cluster", 1),
    ],
    "DATE PATTERN STRESS  (synthetic benches)": [
        ("negation_temporal", 2),
        ("engagement_disjoint", 2),
        ("composition", 2),
        ("edge_conjunctive_temporal", 1),
        ("v7_compound_hard", 1),
    ],
}


async def main():
    llm_planner = QueryPlanner()
    rule_planner = DucklingPlanner()

    for cat_label, bench_list in CATEGORIES.items():
        print(f"\n{'='*72}\n{cat_label}\n{'='*72}")
        for bench, n in bench_list:
            loaded = load_bench(bench)
            if loaded[0] is None:
                continue
            docs_jsonl, queries, gold = loaded
            docs_by_id = {d["doc_id"]: d for d in docs_jsonl}
            print(f"\n--- bench: {bench} ---")
            shown = 0
            for q in queries:
                if shown >= n:
                    break
                gold_ids = gold.get(q["query_id"], set())
                if not gold_ids:
                    continue
                gold_doc = docs_by_id.get(next(iter(gold_ids)), {})
                llm_plan = await llm_planner.plan(q["text"], q["ref_time"])
                rule_plan = await rule_planner.plan(q["text"], q["ref_time"])
                print(f"\nQ: {q['text']!r}")
                print(f"   ref_time={q['ref_time']}")
                print(f"   gold: {gold_doc.get('text', '?')[:90]}...")
                print(f"   gold ref_time={gold_doc.get('ref_time','?')}")
                print(f"  LLM plan:")
                print(fmt_plan(llm_plan))
                print(f"  Rule plan:")
                print(fmt_plan(rule_plan))
                shown += 1


if __name__ == "__main__":
    asyncio.run(main())
