"""Diff queries between shipped (v3.3) and v3.6 surgical extractor.

For each bench with non-zero ΔR@1, identify which specific queries
flipped (gained or lost gold@1) and dump the doc-side anchors from
both extractors so we can see what actually changed."""
from __future__ import annotations

import asyncio

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    load_bench, make_cached_embed_fn, make_cosine_rerank_fn,
)
from temporal_retrieval_tr.research._extractor_v3_6_surgical import (
    TemporalExtractorV3_6,
)

setup_env()


# Benches with non-zero deltas
BENCHES = ["adversarial", "disc", "era", "goldilocks", "negation_temporal"]


def rank_of(gold, ranked):
    for i, did in enumerate(ranked):
        if did in gold:
            return i + 1
    return None


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    for bench in BENCHES:
        loaded = load_bench(bench)
        if loaded[0] is None:
            continue
        docs_jsonl, queries, gold = loaded
        docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
                for d in docs_jsonl]
        vd_a = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
        await vd_a.index(docs)
        vd_b = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn,
                                 extractor=TemporalExtractorV3_6())
        await vd_b.index(docs)

        print(f"\n===== {bench} =====")
        for q in queries:
            qid = q["query_id"]
            g = set(gold.get(qid, []))
            if not g:
                continue
            res_a = await vd_a.query(q["text"], q["ref_time"], k=10)
            res_b = await vd_b.query(q["text"], q["ref_time"], k=10)
            ra = [r.doc_id for r in res_a]
            rb = [r.doc_id for r in res_b]
            rank_a = rank_of(g, ra)
            rank_b = rank_of(g, rb)
            # Only show queries where rank@1 status changed (flipped pass/fail)
            a_pass = rank_a == 1
            b_pass = rank_b == 1
            if a_pass == b_pass:
                continue
            label = "GAIN" if b_pass else "LOSS"
            print(f"  [{label}] Q: {q['text']!r}")
            print(f"           rank_v33={rank_a}  rank_v36={rank_b}")
            # Show top-3 with their doc text
            for top_id in rb[:3]:
                text = next((d["text"] for d in docs_jsonl if d["doc_id"] == top_id), "")[:100]
                marker = "GOLD" if top_id in g else "    "
                print(f"           [{marker}] {top_id}: {text!r}")


if __name__ == "__main__":
    asyncio.run(main())
