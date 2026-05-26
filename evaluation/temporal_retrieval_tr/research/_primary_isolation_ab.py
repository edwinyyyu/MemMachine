"""Isolate the primary_idx contribution from the extractor-prompt difference.

Previous A/B showed primary_idx (Option 1) beats extreme — but the win
might be from (a) the slightly-different extraction prompt or (b) using
primary as the recency anchor. This script holds extraction constant
and varies only whether we USE primary_idx:

- v33_extreme:  V3.3 extractor + extreme anchor (current production baseline)
- v34_extreme:  V3.4 extractor (gets primary_idx) + extreme anchor (IGNORE primary)
- v34_primary:  V3.4 extractor (gets primary_idx) + primary anchor (USE primary)

A vs A' (v33_extreme vs v34_extreme): does the v3.4 prompt itself
change extraction quality? (Same recency mechanism.)

A' vs B (v34_extreme vs v34_primary): does USING primary_idx help,
holding extraction constant? This is the clean isolation.

Cache is warm from prior runs — fast.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._primary_isolation_ab
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

# (label, extractor_factory, anchor)
ARMS: list[tuple[str, callable, str]] = [
    ("v33_extreme", TemporalExtractorV3_3, "extreme"),
    ("v34_extreme", lambda: TemporalExtractorV3_4Primary(mode="idx"), "extreme"),
    ("v34_primary", lambda: TemporalExtractorV3_4Primary(mode="idx"), "primary"),
]


async def run_bench(bench: str, embed_fn, rerank_fn) -> dict | None:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    out = {}
    for label, extractor_factory, anchor in ARMS:
        extractor = extractor_factory()
        vd = TemporalRetriever(
            embed_fn=embed_fn, rerank_fn=rerank_fn,
            extractor=extractor,
        )
        await vd.index(docs)
        vd.recency_weight = 0.5
        vd.copeland_bonus = None
        vd.recency_anchor = anchor
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[label] = metrics(rk, gold)
        del vd
        gc.collect()
    return out


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    labels = [a[0] for a in ARMS]
    print(f"=== Primary isolation A/B over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    print(f"Arms: {labels}", flush=True)
    print(f"v33_extreme vs v34_extreme: extraction-prompt effect (same recency)",
          flush=True)
    print(f"v34_extreme vs v34_primary: primary-anchor effect (same extraction)\n",
          flush=True)
    header = "  ".join(f"{L:>14s}" for L in labels)
    hdr = f"{'bench':28s}  {header}    n"
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
        cells = "  ".join(f"{res[L]['R@1']:>14.3f}" for L in labels)
        print(f"  {bench:26s}  {cells}  {res[labels[0]]['n']:>4d}", flush=True)
    if rows:
        n = len(rows)
        print("-" * len(hdr), flush=True)
        for k_metric in ("R@1", "R@5", "R@10"):
            macro = {L: sum(r[L].get(k_metric, 0) for r in rows.values()) / n
                     for L in labels}
            cells = "  ".join(f"{macro[L]:>14.4f}" for L in labels)
            print(f"  {'MACRO ' + k_metric:26s}  {cells}  n={n}", flush=True)
        print(flush=True)
        # Decomposition
        print("=== Decomposition (where deltas occur) ===")
        for bench, r in rows.items():
            v33e = r["v33_extreme"]["R@1"]
            v34e = r["v34_extreme"]["R@1"]
            v34p = r["v34_primary"]["R@1"]
            d_prompt = v34e - v33e  # prompt effect alone
            d_primary = v34p - v34e  # primary anchor effect alone
            d_total = v34p - v33e
            if abs(d_prompt) > 0.005 or abs(d_primary) > 0.005:
                print(f"  {bench:30s}  v33e={v33e:.3f}  v34e={v34e:.3f}  v34p={v34p:.3f}"
                      f"  Δprompt={d_prompt:+.3f}  Δprimary={d_primary:+.3f}"
                      f"  Δtotal={d_total:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
