"""Asymmetric 2x2: LLM vs rule on each side (extractor / planner) + pure-semantic.

Fills the missing cells (LLM-extr + rule-plan) so the full matrix is bench'd.

Macro on non-recency benches answers: which side does the LLM matter most on?
"""
from __future__ import annotations

import asyncio, gc

import numpy as np

from temporal_retrieval_tr import Doc, TemporalRetriever, TemporalExtractor
from temporal_retrieval_tr.planner import QueryPlanner
from temporal_retrieval_tr.research._dateparser_extractor import DateparserExtractor
from temporal_retrieval_tr.research._duckling_extractor import DucklingHTTPExtractor
from temporal_retrieval_tr.research._rule_planners import (
    DateparserPlanner, DucklingPlanner,
)
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()

# Configs we don't yet have benched in this session.
# (LLM-extr + LLM-plan, *-extr + LLM-plan, and *-both already exist.)
CONFIGS = [
    ("LLM-extr   + duckling-plan",   TemporalExtractor,    DucklingPlanner),
    ("LLM-extr   + dateparser-plan", TemporalExtractor,    DateparserPlanner),
]


async def run_bench_temporal(extractor, planner, bench, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        extractor=extractor, planner=planner,
        ranking_method="copeland_pairwise",
    )
    await vd.index(docs)
    rk = {q["query_id"]: [x.doc_id for x in
          await vd.query(q["text"], q["ref_time"], k=10)]
          for q in queries}
    m = metrics(rk, gold)
    del vd, docs
    gc.collect()
    return m


async def run_pure_semantic(bench, embed_fn):
    """Pure cosine top-K against the whole corpus. No temporal anything."""
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    doc_ids = [d["doc_id"] for d in docs_jsonl]
    doc_embs = await embed_fn([d["text"] for d in docs_jsonl])
    doc_mat = np.stack(
        [np.asarray(e, dtype=np.float32) for e in doc_embs], axis=0
    )
    doc_norms = np.linalg.norm(doc_mat, axis=1) + 1e-9
    rankings: dict[str, list[str]] = {}
    for q in queries:
        q_emb = (await embed_fn([q["text"]]))[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_emb)) + 1e-9
        sims = (doc_mat @ q_emb) / (doc_norms * q_norm)
        order = np.argsort(-sims)[:10]
        rankings[q["query_id"]] = [doc_ids[i] for i in order]
    return metrics(rankings, gold)


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    print("\n=== Pure semantic (no temporal anywhere) ===\n", flush=True)
    rows = []
    for bench in BENCH_NAMES:
        try:
            m = await run_pure_semantic(bench, embed_fn)
        except Exception as e:
            print(f"  ERROR {bench}: {e}", flush=True)
            continue
        if m is None:
            continue
        rows.append((bench, m))
        print(
            f"  {bench:30s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  n={m['n']}",
            flush=True,
        )
    n = len(rows)
    r1 = sum(m["R@1"] for _, m in rows) / n
    r5 = sum(m["R@5"] for _, m in rows) / n
    print(f"\nMACRO pure-semantic:  R@1={r1:.4f}  R@5={r5:.4f}  n={n}", flush=True)

    for label, ExtractorCls, PlannerCls in CONFIGS:
        print(f"\n=== {label} ===\n", flush=True)
        extractor = ExtractorCls()
        planner = PlannerCls()
        rows = []
        for bench in BENCH_NAMES:
            try:
                m = await run_bench_temporal(extractor, planner, bench,
                                             embed_fn, rerank_fn)
            except Exception as e:
                print(f"  ERROR {bench}: {e}", flush=True)
                continue
            if m is None:
                continue
            rows.append((bench, m))
            print(
                f"  {bench:30s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  n={m['n']}",
                flush=True,
            )
            if hasattr(extractor, "save_caches"):
                extractor.save_caches()
            if hasattr(planner, "save_caches"):
                planner.save_caches()
        n = len(rows)
        r1 = sum(m["R@1"] for _, m in rows) / n
        r5 = sum(m["R@5"] for _, m in rows) / n
        print(f"\nMACRO {label}:  R@1={r1:.4f}  R@5={r5:.4f}  n={n}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
