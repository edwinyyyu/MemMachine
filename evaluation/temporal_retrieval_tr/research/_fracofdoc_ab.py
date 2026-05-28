"""Frac-of-doc denominator A/B.

Replace `denom = min(|target|, |doc|)` with `denom = |doc|` in pair_overlap.

Expected: regression on benches where docs commonly have anchors WIDER
than query targets (Q1/year-anchored docs vs month/day queries). Confirms
the user's intuition that temporal-breadth shouldn't be penalized as
specificity — that's cosine's job.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._fracofdoc_ab
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_tr import Doc, TemporalRetriever, IntervalSet
from temporal_retrieval_tr.scoring import final_score, pair_overlap
import temporal_retrieval_tr.scoring as scoring_mod
from temporal_retrieval_tr.time_range import intersect, is_empty, is_inf, measure

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn,
    metrics,
)

setup_env()


def pair_overlap_doc_denom(A: IntervalSet, B: IntervalSet) -> float:
    """Frac-of-DOC overlap. Same hard gate + both-inf shortcut, but the
    denominator is always |B| (the doc anchor), not min(|A|, |B|).

    `best_per_target` always calls `pair_overlap(target, anchor)`, so B is
    the doc anchor by call convention.
    """
    inter = intersect(A, B)
    if is_empty(inter):
        return 0.0

    a_w = measure(A)
    b_w = measure(B)
    inter_w = measure(inter)

    if is_inf(a_w) and is_inf(b_w):
        return 1.0

    if is_inf(b_w):
        # Doc anchor is unbounded — intersection with a finite target gives
        # a finite numerator, but b_w is infinite → frac ≈ 0. This is the
        # extreme of the "penalize breadth" idea.
        return 0.0

    if b_w <= 0:
        return 0.0

    inter_w_val = 0 if is_inf(inter_w) else inter_w
    frac = inter_w_val / b_w
    return min(1.0, frac)


async def run_bench(bench: str, embed_fn, rerank_fn, use_frac_of_doc: bool) -> dict | None:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    # Monkey-patch pair_overlap on the scoring module so the retriever's
    # call into final_score uses the variant.
    original = scoring_mod.pair_overlap
    if use_frac_of_doc:
        scoring_mod.pair_overlap = pair_overlap_doc_denom

    try:
        vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
        await vd.index(docs)
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        m = metrics(rk, gold)
        del vd
        gc.collect()
    finally:
        scoring_mod.pair_overlap = original
    return m


async def main() -> None:
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== Frac-of-doc denominator A/B over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    hdr = f"{'bench':30s}  {'fmin R@1':>9s} {'fdoc R@1':>9s} {'Δ R@1':>8s}  {'fmin R@5':>9s} {'fdoc R@5':>9s} {'n':>4s}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    for bench in BENCH_NAMES:
        try:
            fmin = await run_bench(bench, embed_fn, rerank_fn, False)
            fdoc = await run_bench(bench, embed_fn, rerank_fn, True)
        except Exception as e:
            print(f"{bench:30s}  ERROR: {e}", flush=True)
            continue
        if fmin is None:
            continue
        rows[bench] = (fmin, fdoc)
        d_r1 = fdoc["R@1"] - fmin["R@1"]
        mark = ">" if abs(d_r1) >= 0.02 else " "
        print(f"{mark} {bench:28s}  {fmin['R@1']:>9.3f} {fdoc['R@1']:>9.3f} "
              f"{d_r1:>+8.3f}  {fmin['R@5']:>9.3f} {fdoc['R@5']:>9.3f} "
              f"{fmin['n']:>4d}", flush=True)
    if rows:
        n = len(rows)
        macro_fmin_r1 = sum(r[0]["R@1"] for r in rows.values()) / n
        macro_fdoc_r1 = sum(r[1]["R@1"] for r in rows.values()) / n
        macro_fmin_r5 = sum(r[0]["R@5"] for r in rows.values()) / n
        macro_fdoc_r5 = sum(r[1]["R@5"] for r in rows.values()) / n
        print("-" * len(hdr), flush=True)
        print(f"  {'MACRO':28s}  {macro_fmin_r1:>9.4f} {macro_fdoc_r1:>9.4f} "
              f"{macro_fdoc_r1 - macro_fmin_r1:>+8.4f}  "
              f"{macro_fmin_r5:>9.4f} {macro_fdoc_r5:>9.4f}  n={n}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
