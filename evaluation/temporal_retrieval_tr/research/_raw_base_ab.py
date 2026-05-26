"""A/B: normalized base (current) vs raw cosine base (no normalize_dict).

Removing normalization gives bonus a stable interpretation across
queries (no per-pool amplification). Risk: if pool cosine scores are
in a narrow range, raw base + match becomes nearly-tied for all docs,
and small bonus may dominate.

For each arm, also test multiple bonus values to find the right one
in the raw-base regime.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._raw_base_ab
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

# (label, use_raw_base, copeland_bonus)
# Test at multiple bonus values for each base mode to find the right setting.
ARMS: list[tuple[str, bool, float]] = [
    ("norm_b0.30", False, 0.30),
    ("norm_b0.40", False, 0.40),
    ("norm_b0.50", False, 0.50),
    ("raw_b0.10",  True,  0.10),
    ("raw_b0.15",  True,  0.15),
    ("raw_b0.20",  True,  0.20),
    ("raw_b0.30",  True,  0.30),
    ("raw_b0.40",  True,  0.40),
    ("raw_b0.50",  True,  0.50),
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
    for label, use_raw, bonus in ARMS:
        vd.use_raw_base = use_raw
        vd.copeland_bonus = bonus
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
    print(f"=== Raw base vs normalized base, multiple bonus values ===\n",
          flush=True)
    print(f"Arms: {labels}\n", flush=True)
    header = "  ".join(f"{L:>11s}" for L in labels)
    hdr = f"{'bench':28s}  {header}    n"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    key = {"composition", "cotemporal", "same_topic_recency",
           "same_topic_recency_hard", "recency_stress_deep", "recency_vs_rerank"}
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
        cells = "  ".join(f"{res[L]['R@1']:>11.3f}" for L in labels)
        print(f"{mark} {bench:26s}  {cells}  {res[labels[0]]['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        print("-" * len(hdr), flush=True)
        for k_metric in ("R@1", "R@5", "R@10"):
            macro = {L: sum(r[L].get(k_metric, 0) for r in rows.values()) / n
                     for L in labels}
            cells = "  ".join(f"{macro[L]:>11.4f}" for L in labels)
            print(f"  {'MACRO ' + k_metric:26s}  {cells}  n={n}", flush=True)
        print(flush=True)
        # Best arm per metric
        for k_metric in ("R@1", "R@5", "R@10"):
            macro = {L: sum(r[L].get(k_metric, 0) for r in rows.values()) / n
                     for L in labels}
            best = max(labels, key=lambda L: macro[L])
            print(f"Best macro {k_metric}: {best} = {macro[best]:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
