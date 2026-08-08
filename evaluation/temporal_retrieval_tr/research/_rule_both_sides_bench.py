"""A/B: rule-based extractor + rule-based planner (both sides).

Both Duckling and dateparser variants. Compares against:
  LLM-extr + LLM-plan (ship):       R@1=0.8044  R@5=0.9618
  dateparser-extr + LLM-plan:       R@1=0.7946  R@5=0.9492
  duckling-extr  + LLM-plan:        R@1=0.8013  R@5=0.9533

Without the LLM planner, queries with extremum intent ("most recently",
"latest", "earliest") lose the recency Copeland tournament -- expect
recency-discriminating benches to regress.
"""
from __future__ import annotations

import asyncio, gc

from temporal_retrieval_tr import Doc, TemporalRetriever
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


CONFIGS = [
    ("dateparser-both", DateparserExtractor, DateparserPlanner),
    ("duckling-both",   DucklingHTTPExtractor, DucklingPlanner),
]


async def run_bench(label, extractor, planner, bench, embed_fn, rerank_fn):
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


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    for label, ExtractorCls, PlannerCls in CONFIGS:
        print(f"\n=== {label} ===\n", flush=True)
        extractor = ExtractorCls()
        planner = PlannerCls()
        rows = []
        for bench in BENCH_NAMES:
            try:
                m = await run_bench(label, extractor, planner, bench,
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

    print("\n=== Reference baselines ===")
    print("  LLM-extr        + LLM-plan        R@1=0.8044  R@5=0.9618")
    print("  duckling-extr   + LLM-plan        R@1=0.8013  R@5=0.9533")
    print("  dateparser-extr + LLM-plan        R@1=0.7946  R@5=0.9492")


if __name__ == "__main__":
    asyncio.run(main())
