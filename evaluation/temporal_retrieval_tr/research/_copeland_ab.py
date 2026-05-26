"""3-arm A/B: additive W=1.0 vs additive W=1.5 vs Copeland bonus=0.15.

Tests whether bounded per-pair recency bonus (Copeland tournament) keeps
the composition gain (+0.080 at W=1.5) while avoiding the cotemporal
regression (-0.050 at W=1.5). Predicted by the additive-recency
dilemma: rank-gap accumulation in additive linear-rank cannot be both
big enough to flip composition and small enough to preserve cotemporal.

Hypothesis: Copeland with bonus=0.15 (in normalized rerank-score
space) bounds recency advantage at +0.15 in any pair regardless of
rank-gap. Should:
- keep composition >= 0.520 (when sim gap < 0.15, flip stands)
- keep cotemporal >= 0.950 (when sim gap > 0.15, big-similar wins)

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._copeland_ab
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

# (label, recency_weight, copeland_bonus)
ARMS: list[tuple[str, float, float | None]] = [
    ("add_w1.0", 1.0, None),
    ("add_w1.5", 1.5, None),
    ("copeland_0.15", 0.0, 0.15),  # recency_weight irrelevant when copeland_bonus set
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
    for label, w, copeland in ARMS:
        vd.recency_weight = w
        vd.copeland_bonus = copeland
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[label] = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== 3-arm: {[a[0] for a in ARMS]} over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    hdr = (f"{'bench':28s}  {'add_w1.0':>9s} {'add_w1.5':>9s} {'cope_.15':>9s} "
           f"{'Δcope-w1.0':>10s} {'Δcope-w1.5':>10s}  {'R5_w1.0':>8s} "
           f"{'R5_w1.5':>8s} {'R5_cope':>8s}  {'n':>4s}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
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
        a, b, c = res["add_w1.0"], res["add_w1.5"], res["copeland_0.15"]
        d_ca = c["R@1"] - a["R@1"]
        d_cb = c["R@1"] - b["R@1"]
        # Mark significant deltas at the R@1 level vs add_w1.5 (the
        # current production arm we're comparing against).
        if d_cb > 0.02:
            mark = "+"
        elif d_cb < -0.02:
            mark = "*"
        else:
            mark = " "
        print(f"{mark} {bench:26s}  {a['R@1']:>9.3f} {b['R@1']:>9.3f} "
              f"{c['R@1']:>9.3f} {d_ca:>+10.3f} {d_cb:>+10.3f}  "
              f"{a['R@5']:>8.3f} {b['R@5']:>8.3f} {c['R@5']:>8.3f}  "
              f"{a['n']:>4d}", flush=True)
    if rows:
        n = len(rows)
        macro_r1 = {k: sum(r[k]["R@1"] for r in rows.values()) / n
                    for k in ["add_w1.0", "add_w1.5", "copeland_0.15"]}
        macro_r5 = {k: sum(r[k]["R@5"] for r in rows.values()) / n
                    for k in ["add_w1.0", "add_w1.5", "copeland_0.15"]}
        dca = macro_r1["copeland_0.15"] - macro_r1["add_w1.0"]
        dcb = macro_r1["copeland_0.15"] - macro_r1["add_w1.5"]
        print("-" * len(hdr), flush=True)
        print(f"  {'MACRO':26s}  {macro_r1['add_w1.0']:>9.3f} "
              f"{macro_r1['add_w1.5']:>9.3f} {macro_r1['copeland_0.15']:>9.3f} "
              f"{dca:>+10.3f} {dcb:>+10.3f}  "
              f"{macro_r5['add_w1.0']:>8.3f} {macro_r5['add_w1.5']:>8.3f} "
              f"{macro_r5['copeland_0.15']:>8.3f}  n={n}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
