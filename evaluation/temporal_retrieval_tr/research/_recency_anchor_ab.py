"""Per-doc recency anchor A/B: extreme vs ref_time vs median.

Tests whether bypassing extracted-interval-extremum and using doc
ref_time directly (or median midpoint) for recency gives better
macro performance than the current extreme-midpoint approach.

Hypothesis to validate or refute:
- Current "extreme" mode lets unrelated extreme dates in doc text
  hijack recency rankings (e.g., cot_2_a — "Boulder retreat in 2012"
  stole "first keynote" because its 2012 anchor was the global earliest).
- "ref_time" mode would eliminate that hijack.
- BUT: might lose information when doc.ref_time differs from event time.

Tested for both additive(W=0.5) and Copeland(0.20) so we see whether
the anchor choice interacts with the scoring mechanism.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._recency_anchor_ab
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

# (label, mode, value, anchor)
ARMS: list[tuple[str, str, float, str]] = [
    ("add_extreme",   "additive", 0.5, "extreme"),
    ("add_ref_time",  "additive", 0.5, "ref_time"),
    ("add_median",    "additive", 0.5, "median"),
    ("cope_extreme",  "copeland", 0.20, "extreme"),
    ("cope_ref_time", "copeland", 0.20, "ref_time"),
    ("cope_median",   "copeland", 0.20, "median"),
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
    for label, mode, value, anchor in ARMS:
        vd.recency_anchor = anchor
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
    print(f"=== Recency anchor A/B over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    header = "  ".join(f"{L:>13s}" for L in labels)
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
        cells = "  ".join(f"{res[L]['R@1']:>13.3f}" for L in labels)
        print(f"{mark} {bench:26s}  {cells}  {res[labels[0]]['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        print("-" * len(hdr), flush=True)
        for k_metric in ("R@1", "R@5", "R@10"):
            macro = {L: sum(r[L].get(k_metric, 0) for r in rows.values()) / n
                     for L in labels}
            cells = "  ".join(f"{macro[L]:>13.4f}" for L in labels)
            print(f"  {'MACRO ' + k_metric:26s}  {cells}  n={n}", flush=True)
        print(flush=True)
        # Identify benches where anchor choice matters
        print("=== Benches where anchor choice changes R@1 ===")
        for bench, r in rows.items():
            add_e = r["add_extreme"]["R@1"]
            add_r = r["add_ref_time"]["R@1"]
            add_m = r["add_median"]["R@1"]
            if max(add_e, add_r, add_m) - min(add_e, add_r, add_m) > 0.001:
                print(f"  {bench:28s}  add_extreme={add_e:.3f}  "
                      f"ref_time={add_r:.3f}  median={add_m:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
