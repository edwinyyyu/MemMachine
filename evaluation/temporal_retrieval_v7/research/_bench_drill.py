"""Drill into per-query diffs on a specific bench. Surfaces queries
where V1 and V7 disagree on rank-1 / rank-5, with the planner output
and V7 query_refs printed for diagnosis.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._bench_drill <bench_name>
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

from temporal_retrieval_min import Doc as DocV1
from temporal_retrieval_min import TemporalRetriever
from temporal_retrieval_v7 import Doc as DocV7
from temporal_retrieval_v7 import TemporalRetrieverV7

from temporal_retrieval.research._common import (
    DATA_DIR,
    make_embed_fn,
    setup_env,
)
from temporal_retrieval_v7.research._full_ab import make_cosine_rerank_fn

setup_env()


async def main(bench: str):
    with open(DATA_DIR / f"{bench}_docs.jsonl") as f:
        docs_jsonl = [json.loads(line) for line in f]
    with open(DATA_DIR / f"{bench}_queries.jsonl") as f:
        queries = [json.loads(line) for line in f]
    with open(DATA_DIR / f"{bench}_gold.jsonl") as f:
        gold = {g["query_id"]: set(g["relevant_doc_ids"])
                for g in (json.loads(line) for line in f)}

    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    v1_docs = [DocV1(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
               for d in docs_jsonl]
    v7_docs = [DocV7(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
               for d in docs_jsonl]

    v1 = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    v7 = TemporalRetrieverV7(embed_fn=embed_fn, rerank_fn=rerank_fn)

    await v1.index(v1_docs)
    await v7.index(v7_docs)

    diffs = []
    for q in queries:
        qid = q["query_id"]
        r1 = await v1.query(q["text"], q["ref_time"], k=10)
        r7 = await v7.query(q["text"], q["ref_time"], k=10)
        v1_ids = [r.doc_id for r in r1]
        v7_ids = [r.doc_id for r in r7]
        gs = gold.get(qid, set())
        if not gs:
            continue
        v1_first = next((i+1 for i, d in enumerate(v1_ids) if d in gs), None)
        v7_first = next((i+1 for i, d in enumerate(v7_ids) if d in gs), None)
        if v1_first != v7_first:
            diffs.append({
                "qid": qid,
                "text": q["text"],
                "ref_time": q["ref_time"],
                "v1_first": v1_first,
                "v7_first": v7_first,
                "v1_top5": v1_ids[:5],
                "v7_top5": v7_ids[:5],
                "gold": list(gs),
            })

    print(f"\n=== {bench}: queries where V1 ≠ V7 first-gold rank ===")
    for d in diffs:
        sign = (
            "V1 BETTER"
            if (d["v7_first"] or 99) > (d["v1_first"] or 99)
            else "V7 BETTER"
        )
        print(f"\n[{sign}] {d['qid']}: V1={d['v1_first']}  V7={d['v7_first']}")
        print(f"  query: {d['text']}")
        print(f"  gold: {d['gold']}")
        print(f"  V1 top5: {d['v1_top5']}")
        print(f"  V7 top5: {d['v7_top5']}")
    print(f"\n{len(diffs)} disagreement(s) out of {len(queries)} queries")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "negation_temporal"))
