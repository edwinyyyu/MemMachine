"""Copeland bonus sweep: find the bonus value with best composition/cotemporal trade-off.

Conceptual map:
- bonus=0.00: recency never flips a pair → equivalent to no-recency baseline
- bonus≈0.10: only very-close-sim pairs flipped (conservative)
- bonus≈0.15: my initial pick (a clean fraction of normalized rerank range)
- bonus≈0.20-0.25: middle of suspected sweet spot
- bonus=0.30: ~15% of [base+match] range (aggressive)
- bonus=0.40+: recency dominates most pairs (≈ rank-by-recency)

Watching for:
- composition: max R@1 (the recency-needed gain)
- cotemporal: stays at 0.950 (no regression)
- macro: aggregate of both effects
- bonus that maximizes composition WHILE preserving cotemporal

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._copeland_bonus_sweep
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

BONUSES: list[float] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


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
    for b in BONUSES:
        vd.copeland_bonus = b
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[b] = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== Copeland bonus sweep {BONUSES} over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    bcols = "  ".join(f"{b:>5.2f}" for b in BONUSES)
    hdr = f"{'bench':28s}  {bcols}  {'n':>4s}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    key_benches = {"composition", "cotemporal"}  # benches we care most about
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
        vals = "  ".join(f"{res[b]['R@1']:>5.3f}" for b in BONUSES)
        mark = ">" if bench in key_benches else " "
        print(f"{mark} {bench:26s}  {vals}  {res[BONUSES[0]]['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        print("-" * len(hdr), flush=True)
        macro_r1 = [sum(r[b]["R@1"] for r in rows.values()) / n for b in BONUSES]
        macro_r5 = [sum(r[b]["R@5"] for r in rows.values()) / n for b in BONUSES]
        vals = "  ".join(f"{v:>5.3f}" for v in macro_r1)
        print(f"  {'MACRO R@1':26s}  {vals}  n={n}", flush=True)
        vals = "  ".join(f"{v:>5.3f}" for v in macro_r5)
        print(f"  {'MACRO R@5':26s}  {vals}  n={n}", flush=True)
        # Highlight best per axis
        best_macro = max(range(len(BONUSES)), key=lambda i: macro_r1[i])
        print(f"\nMacro R@1 peak: bonus={BONUSES[best_macro]} R@1={macro_r1[best_macro]:.4f}",
              flush=True)
        if "composition" in rows:
            comp = [rows["composition"][b]["R@1"] for b in BONUSES]
            best_c = max(range(len(BONUSES)), key=lambda i: comp[i])
            print(f"Composition peak: bonus={BONUSES[best_c]} R@1={comp[best_c]:.3f}",
                  flush=True)
        if "cotemporal" in rows:
            cot = [rows["cotemporal"][b]["R@1"] for b in BONUSES]
            print(f"Cotemporal across sweep: {[f'{v:.3f}' for v in cot]}",
                  flush=True)


if __name__ == "__main__":
    asyncio.run(main())
