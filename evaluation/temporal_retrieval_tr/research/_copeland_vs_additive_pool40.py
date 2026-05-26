"""Copeland vs additive head-to-head at overfetch=4 (pool_size=40).

Previous Copeland data was at pool_size=10. Architecture changed. Need
to re-test whether Copeland's bounded pairwise mechanic outperforms
additive when wider pools are at stake.

Comparing on the same 39 benches:
- Best additive: W=0.5 (from prior sweep)
- Best additive (high W reference): W=1.5
- Copeland sweep: bonus ∈ {0.10, 0.20, 0.30, 0.50, 0.80}

If Copeland strictly dominates additive on macro across (R@1, R@5, R@10),
ship it. If additive(W=0.5) dominates, ship that (simpler mechanism).

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._copeland_vs_additive_pool40
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_tr import Doc, TemporalRetriever

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES,
    load_bench,
    make_cached_embed_fn,
    make_cosine_rerank_fn,
    metrics,
)

setup_env()

# (label, mode, value)
ARMS: list[tuple[str, str, float]] = [
    ("add_W0.5",  "additive", 0.5),
    ("add_W1.0",  "additive", 1.0),
    ("add_W1.5",  "additive", 1.5),
    ("cope_0.10", "copeland", 0.10),
    ("cope_0.20", "copeland", 0.20),
    ("cope_0.30", "copeland", 0.30),
    ("cope_0.50", "copeland", 0.50),
    ("cope_0.80", "copeland", 0.80),
]


async def run_bench(bench: str, embed_fn, rerank_fn) -> dict | None:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)
    out = {}
    for label, mode, value in ARMS:
        if mode == "additive":
            vd.recency_weight = value
            vd.copeland_bonus = None
        else:
            vd.copeland_bonus = value
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[label] = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    labels = [a[0] for a in ARMS]
    print(f"=== Copeland vs additive @ pool=40, {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    print(f"Arms: {labels}\n", flush=True)
    header = "  ".join(f"{L:>9s}" for L in labels)
    hdr = f"{'bench':28s}  {header}    n"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    key = {"composition", "cotemporal", "same_topic_recency"}
    for bench in BENCH_NAMES:
        try:
            res = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:28s}  ERROR: {e}", flush=True)
            continue
        if res is None:
            print(f"{bench:28s}  SKIPPED", flush=True)
            continue
        rows[bench] = res
        mark = ">" if bench in key else " "
        cells = "  ".join(f"{res[L]['R@1']:>9.3f}" for L in labels)
        print(f"{mark} {bench:26s}  {cells}  {res[labels[0]]['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        print("-" * len(hdr), flush=True)
        for k_metric in ("R@1", "R@5", "R@10"):
            macro = {L: sum(r[L].get(k_metric, 0) for r in rows.values()) / n
                     for L in labels}
            cells = "  ".join(f"{macro[L]:>9.4f}" for L in labels)
            print(f"  {'MACRO ' + k_metric:26s}  {cells}  n={n}", flush=True)
        print(flush=True)
        print("=== Key benches ===")
        for kb in ("composition", "cotemporal", "same_topic_recency"):
            if kb not in rows:
                continue
            for k_metric in ("R@1", "R@5", "R@10"):
                cells = "  ".join(f"{rows[kb][L].get(k_metric, 0):>9.3f}"
                                  for L in labels)
                print(f"  {kb + ' ' + k_metric:26s}  {cells}"
                      f"  n={rows[kb][labels[0]]['n']}", flush=True)
        print(flush=True)
        for k_metric in ("R@1", "R@5", "R@10"):
            macro = {L: sum(r[L].get(k_metric, 0) for r in rows.values()) / n
                     for L in labels}
            best_label = max(labels, key=lambda L: macro[L])
            print(f"Best macro {k_metric}: {best_label} = {macro[best_label]:.4f}",
                  flush=True)


if __name__ == "__main__":
    asyncio.run(main())
