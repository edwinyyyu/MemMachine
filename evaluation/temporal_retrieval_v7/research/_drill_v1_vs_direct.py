"""Drill into per-query disagreements between V1 baseline and V7-Direct
on a given bench. Surfaces plan + ranking deltas.

Run:
    uv run python -m temporal_retrieval_v7.research._drill_v1_vs_direct <bench>
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone

from temporal_retrieval_min import Doc as DocV1
from temporal_retrieval_min import TemporalRetriever
from temporal_retrieval_v7 import (
    DirectQueryPlanner,
    Doc as DocV7,
    NEG_INF,
    POS_INF,
    TemporalRetrieverV7Direct,
)

from temporal_retrieval.research._common import DATA_DIR, make_embed_fn, setup_env
from temporal_retrieval_v7.research._full_ab import make_cosine_rerank_fn

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


async def main(bench: str):
    with open(DATA_DIR / f"{bench}_docs.jsonl") as f:
        docs_jsonl = [json.loads(l) for l in f]
    with open(DATA_DIR / f"{bench}_queries.jsonl") as f:
        queries = [json.loads(l) for l in f]
    with open(DATA_DIR / f"{bench}_gold.jsonl") as f:
        gold = {g["query_id"]: set(g["relevant_doc_ids"])
                for g in (json.loads(l) for l in f)}

    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    v1 = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    vd = TemporalRetrieverV7Direct(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await v1.index([DocV1(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
                    for d in docs_jsonl])
    await vd.index([DocV7(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
                    for d in docs_jsonl])

    direct_planner = DirectQueryPlanner()
    docs_by_id = {d["doc_id"]: d for d in docs_jsonl}

    print(f"\n=== {bench}: V1 vs V7-Direct disagreements ===")
    for q in queries:
        qid = q["query_id"]
        gset = gold.get(qid, set())
        if not gset:
            continue
        r1 = await v1.query(q["text"], q["ref_time"], k=10)
        rD = await vd.query(q["text"], q["ref_time"], k=10)
        v1_ids = [r.doc_id for r in r1]
        vD_ids = [r.doc_id for r in rD]
        v1_first = next((i+1 for i, d in enumerate(v1_ids) if d in gset), None)
        vD_first = next((i+1 for i, d in enumerate(vD_ids) if d in gset), None)
        if v1_first == vD_first:
            continue
        marker = "V1 BETTER" if (vD_first or 99) > (v1_first or 99) else "V7-DIRECT BETTER"
        print(f"\n[{marker}] {qid}: V1={v1_first}  D={vD_first}")
        print(f"  query: {q['text']}")
        print(f"  ref_time: {q['ref_time']}")
        dp = await direct_planner.plan(q["text"], q["ref_time"])
        print(f"  Direct plan ({len(dp.clauses)} clauses, extremum={dp.extremum}):")
        for ci, c in enumerate(dp.clauses):
            print(f"    [{ci}] bind={c.bind}, refs={[_fmt_range(r) for r in c.refs]}")
        print(f"  gold ({len(gset)}): {gset}")
        print(f"  V1 top5: {v1_ids[:5]}")
        print(f"  D  top5: {vD_ids[:5]}")
        # Show gold doc text for the gold that was found by V1 (or first gold)
        for g in gset:
            txt = docs_by_id.get(g, {}).get("text", "")
            print(f"    {g}: {txt[:120]}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "goldilocks"))
