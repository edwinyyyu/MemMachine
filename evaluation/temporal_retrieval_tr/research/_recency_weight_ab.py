"""A/B: recency_weight sweep on extremum-relevant benches.

Failure mode (from composition drill): for an extremum="latest" query
the recency boost is one equal-weighted term in `score = base_s + m +
r_v`, so a doc that's semantically ~0.11+ stronger than the actually-
latest doc wins even when the user explicitly asked for the latest.
Fix: scale recency by `recency_weight` for extremum queries.

Sweeps recency_weight ∈ {1.0, 1.5, 2.0, 2.5, 3.0} on benches with
extremum queries. Includes one non-extremum control (allen) — should
be flat in recency_weight by construction.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._recency_weight_ab
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_tr import Doc, TemporalRetriever

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    load_bench,
    make_cosine_rerank_fn,
    metrics,
)

setup_env()

WEIGHTS = [1.0, 1.5, 2.0, 2.5, 3.0]

# Extremum-relevant benches + one non-extremum control.
BENCHES = [
    "composition", "latest_recent", "realq_deictic", "realq", "realq_v2",
    "ambiguous_year_adv", "edge_relative_time", "hard_bench",
    "allen",  # control — should be flat
]


async def run_bench(bench: str, embed_fn, rerank_fn) -> dict[float, dict] | None:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)
    out: dict[float, dict] = {}
    for w in WEIGHTS:
        vd.recency_weight = w
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[w] = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print("=== recency_weight sweep (extremum-only contribution) ===\n",
          flush=True)
    hdr = f"{'bench':24s} " + " ".join(f"w={w:<5g}" for w in WEIGHTS) + "    n"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    for bench in BENCHES:
        try:
            res = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:24s} ERROR: {e}", flush=True)
            continue
        if res is None:
            print(f"{bench:24s} SKIPPED", flush=True)
            continue
        rows[bench] = res
        cells = " ".join(f"{res[w]['R@1']:<7.3f}" for w in WEIGHTS)
        n = res[WEIGHTS[0]]["n"]
        print(f"{bench:24s} {cells}  {n:>4d}", flush=True)
    if rows:
        n = len(rows)
        print("-" * len(hdr), flush=True)
        macro1 = " ".join(
            f"{sum(r[w]['R@1'] for r in rows.values()) / n:<7.3f}"
            for w in WEIGHTS
        )
        macro5 = " ".join(
            f"{sum(r[w]['R@5'] for r in rows.values()) / n:<7.3f}"
            for w in WEIGHTS
        )
        print(f"{'MACRO R@1':24s} {macro1}", flush=True)
        print(f"{'MACRO R@5':24s} {macro5}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
