"""Diagnose variant A regressions on v7_doc_directional, axis,
same_topic_recency, same_topic_recency_hard.

For each regressing bench, identify queries where:
  - baseline (no-anaphora) ranks gold @1
  - variant A (recurring → empty) ranks gold > 1
And dump both plans so we can see what the recurring rule did.
"""
from __future__ import annotations

import asyncio
import json

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    load_bench, make_cached_embed_fn, make_cosine_rerank_fn,
)
from temporal_retrieval_tr.research._no_anaphora_planner import NoAnaphoraPlanner
from temporal_retrieval_tr.research._recurring_empty_planner import (
    RecurringEmptyPlanner,
)
from temporal_retrieval_tr.time_range import is_inf

setup_env()


REGRESSING = [
    "v7_doc_directional",
    "axis",
    "same_topic_recency",
    "same_topic_recency_hard",
]


def _us_iso(us):
    if is_inf(us):
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc).strftime("%Y-%m-%d")


def plan_to_summary(plan) -> dict:
    return {
        "targets": [
            [(_us_iso(iv.earliest_us), _us_iso(iv.latest_us))
             for iv in t.intervals]
            for t in plan.targets
        ],
        "extremum": plan.extremum,
    }


def rank_of(gold, ranked) -> int | None:
    for i, did in enumerate(ranked):
        if did in gold:
            return i + 1
    return None


async def diagnose(bench, embed_fn, rerank_fn):
    docs_jsonl, queries, gold = load_bench(bench)
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    base_planner = NoAnaphoraPlanner()
    var_planner = RecurringEmptyPlanner()

    vd_b = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn,
                             planner=base_planner)
    await vd_b.index(docs)
    vd_v = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn,
                             planner=var_planner)
    await vd_v.index(docs)

    print(f"\n===== {bench} =====", flush=True)
    n_diverge = 0
    for q in queries:
        qid = q["query_id"]
        q_gold = set(gold.get(qid, []))
        if not q_gold:
            continue
        plan_b = await base_planner.plan(q["text"], q["ref_time"])
        plan_v = await var_planner.plan(q["text"], q["ref_time"])
        res_b = await vd_b.query(q["text"], q["ref_time"], k=10)
        res_v = await vd_v.query(q["text"], q["ref_time"], k=10)
        rb = [r.doc_id for r in res_b]
        rv = [r.doc_id for r in res_v]
        rank_b = rank_of(q_gold, rb)
        rank_v = rank_of(q_gold, rv)
        if rank_b == rank_v:
            continue
        n_diverge += 1
        ref = q["ref_time"]
        print(f"  Q: {q['text']!r}  ref={ref}", flush=True)
        print(f"     base rank={rank_b}   var rank={rank_v}", flush=True)
        print(f"     base plan: {json.dumps(plan_to_summary(plan_b))}",
              flush=True)
        print(f"     var  plan: {json.dumps(plan_to_summary(plan_v))}",
              flush=True)
        # show the gold doc anchor + the displacing doc anchor (top 3 of var)
        for top_id in rv[:3]:
            text = next((d["text"] for d in docs_jsonl
                         if d["doc_id"] == top_id), "")[:120]
            marker = "GOLD" if top_id in q_gold else "    "
            print(f"     [{marker}] {top_id}: {text!r}", flush=True)
    print(f"  ({n_diverge} divergences)", flush=True)


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    for bench in REGRESSING:
        await diagnose(bench, embed_fn, rerank_fn)


if __name__ == "__main__":
    asyncio.run(main())
