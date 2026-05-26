"""Consolidated recency study: additive w=1.0, w=1.5 vs Copeland bonus sweep.

Combines the prior 3-arm + bonus sweep into a single run so all arms see
identical planner/extractor outputs (no inter-run stochasticity). The
partial data from the killed jobs showed:
- additive w=1.0:  composition 0.440, cotemporal 0.950
- additive w=1.5:  composition 0.520, cotemporal 0.900
- Copeland(0.15): composition 0.360, cotemporal 0.950

Copeland(0.15) is MORE CONSERVATIVE than additive w=1.0 because the
fixed pairwise bonus 0.15 is less aggressive than additive's
accumulated rank-gap boost W·1 = 1.0. This sweep maps Copeland's
bonus curve to see if any value (a) lifts composition above 0.440
while (b) keeping cotemporal at 0.950.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._recency_consolidated
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_tr import Doc, TemporalRetriever

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES,
    load_bench,
    make_cosine_rerank_fn,
    metrics,
)

setup_env()

# (label, mode, value) where mode = "additive" | "copeland"
ARMS: list[tuple[str, str, float]] = [
    ("add_w1.0", "additive", 1.0),
    ("add_w1.5", "additive", 1.5),
    ("cope_0.05", "copeland", 0.05),
    ("cope_0.10", "copeland", 0.10),
    ("cope_0.15", "copeland", 0.15),
    ("cope_0.20", "copeland", 0.20),
    ("cope_0.25", "copeland", 0.25),
    ("cope_0.30", "copeland", 0.30),
    ("cope_0.40", "copeland", 0.40),
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
        else:  # copeland
            vd.copeland_bonus = value
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[label] = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


def fmt_row(name: str, vals: dict, labels: list[str], n: int | None = None) -> str:
    cells = "  ".join(f"{vals[L]:>5.3f}" for L in labels)
    suffix = f"  n={n}" if n is not None else ""
    return f"{name:26s}  {cells}{suffix}"


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    labels = [a[0] for a in ARMS]
    print(f"=== Consolidated recency study: {len(ARMS)} arms × {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    print(f"Arms: {labels}\n", flush=True)
    header_cells = "  ".join(f"{L:>9s}" for L in labels)
    hdr = f"{'bench':28s}  {header_cells}    n"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    key_benches = {"composition", "cotemporal", "latest_recent"}
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
        mark = ">" if bench in key_benches else " "
        cells = "  ".join(f"{res[L]['R@1']:>9.3f}" for L in labels)
        print(f"{mark} {bench:26s}  {cells}  {res[labels[0]]['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        print("-" * len(hdr), flush=True)
        macro_r1 = {L: sum(r[L]["R@1"] for r in rows.values()) / n for L in labels}
        macro_r5 = {L: sum(r[L]["R@5"] for r in rows.values()) / n for L in labels}
        cells = "  ".join(f"{macro_r1[L]:>9.3f}" for L in labels)
        print(f"  {'MACRO R@1':26s}  {cells}  n={n}", flush=True)
        cells = "  ".join(f"{macro_r5[L]:>9.3f}" for L in labels)
        print(f"  {'MACRO R@5':26s}  {cells}  n={n}", flush=True)
        print(flush=True)
        # Key-bench detail
        print("=== Key benches ===", flush=True)
        for kb in ("composition", "cotemporal", "latest_recent"):
            if kb in rows:
                cells = "  ".join(f"{rows[kb][L]['R@1']:>9.3f}" for L in labels)
                print(f"  {kb:26s}  {cells}  n={rows[kb][labels[0]]['n']}",
                      flush=True)
        # Find best Copeland arm by macro R@1
        cope_labels = [L for L in labels if L.startswith("cope")]
        if cope_labels:
            best_cope = max(cope_labels, key=lambda L: macro_r1[L])
            print(f"\nBest Copeland by macro R@1: {best_cope} = {macro_r1[best_cope]:.4f}",
                  flush=True)
            print(f"vs add_w1.0={macro_r1['add_w1.0']:.4f}, "
                  f"add_w1.5={macro_r1['add_w1.5']:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
