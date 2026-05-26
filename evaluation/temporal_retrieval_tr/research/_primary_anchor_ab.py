"""Five-way recency anchor A/B: extreme / ref_time / median / primary-idx / primary-sep.

Tests whether the LLM-tagged "primary event time" beats the programmatic
heuristics. Two primary-tagging schemas:
- "primary_idx" (Option 1): {refs: [...], primary_index: int|null}
- "primary_sep" (Option 2): {primary: {...}|null, others: [...]}

Both use the same extraction prompt with a primary-identification
instruction appended, only the JSON output schema differs.

Comparison axis: do either of the LLM-tagged variants beat the
existing extreme/ref_time/median heuristics on macro R@1/R@5/R@10?

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._primary_anchor_ab
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_min.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval_min.extractor_v3_4_primary import TemporalExtractorV3_4Primary

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

# Each arm = (label, anchor, extractor_factory)
# All arms use additive W=0.5 (the empirically-best scoring tune)
ARMS: list[tuple[str, str, callable]] = [
    ("extreme",      "extreme",  TemporalExtractorV3_3),
    ("ref_time",     "ref_time", TemporalExtractorV3_3),
    ("median",       "median",   TemporalExtractorV3_3),
    ("primary_idx",  "primary",  lambda: TemporalExtractorV3_4Primary(mode="idx")),
    ("primary_sep",  "primary",  lambda: TemporalExtractorV3_4Primary(mode="sep")),
]


async def run_bench_for_extractor(
    bench: str, extractor_factory, embed_fn, rerank_fn
) -> tuple[TemporalRetriever, list, dict] | None:
    """Index once with the given extractor; return retriever + queries + gold."""
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    extractor = extractor_factory()
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        extractor=extractor,
    )
    await vd.index(docs)
    return vd, queries, gold


async def run_bench(bench: str, embed_fn, rerank_fn) -> dict | None:
    """Run all 5 arms on a bench. Each arm needs its own extractor + index."""
    out = {}
    for label, anchor, extractor_factory in ARMS:
        result = await run_bench_for_extractor(bench, extractor_factory, embed_fn, rerank_fn)
        if result is None:
            return None
        vd, queries, gold = result
        vd.recency_weight = 0.5
        vd.copeland_bonus = None
        vd.recency_anchor = anchor
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[label] = metrics(rk, gold)
        # Track how many docs have a primary tagged (for primary arms)
        if anchor == "primary":
            n_tagged = sum(1 for v in vd._doc_primary_idx.values() if v is not None)
            n_total = len(vd._doc_primary_idx)
            out[label]["_primary_tagged"] = f"{n_tagged}/{n_total}"
        del vd
        gc.collect()
    return out


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    labels = [a[0] for a in ARMS]
    print(f"=== 5-way recency anchor A/B over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    print(f"Arms: {labels}\n", flush=True)
    header = "  ".join(f"{L:>13s}" for L in labels)
    hdr = f"{'bench':28s}  {header}    n"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    key = {"composition", "cotemporal", "same_topic_recency",
           "same_topic_recency_hard", "recency_stress_deep", "recency_vs_rerank",
           "v7_doc_directional"}
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
        # Primary-tagging stats
        print("=== Primary-tagging coverage (per bench, for primary_idx) ===")
        for bench, r in rows.items():
            tag = r.get("primary_idx", {}).get("_primary_tagged", "n/a")
            print(f"  {bench:30s}  {tag}")
        print(flush=True)
        # Benches where anchor choice matters
        print("=== Benches where anchor choice changes R@1 ===")
        for bench, r in rows.items():
            vals = [r[L]["R@1"] for L in labels]
            if max(vals) - min(vals) > 0.001:
                cells = "  ".join(f"{L}={r[L]['R@1']:.3f}" for L in labels)
                print(f"  {bench:30s}  {cells}")


if __name__ == "__main__":
    asyncio.run(main())
