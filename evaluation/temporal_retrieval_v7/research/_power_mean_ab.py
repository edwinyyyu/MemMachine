"""A/B: power-mean exponent sweep for the cross-ref score combiner.

STALE — the `score_p` retriever knob this script mutates was reverted
from production after the experiment concluded (p=1 is the joint
optimum; see project_v7_combiner_not_a_lever). Re-running needs the
`score_p` threading re-applied to scoring.py + retriever.py. Kept as
the evidence snapshot.


`final_score` folds per-ref bests with a power mean M_p. p=1.0 is the
arithmetic mean (current). p>1 softens the penalty for unmatched refs
(more max-like). This sweeps p and reports R@1 / R@5 per bench.

Power mean only affects MULTI-ref queries — single-ref benches should
show identical numbers across p (built-in sanity check). Each bench is
indexed once; p is swept by mutating `retriever.score_p`.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._power_mean_ab
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_v7 import Doc as DocV7
from temporal_retrieval_v7 import TemporalRetrieverV7Direct

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_v7.research._full_ab import (
    _load_bench,
    _metrics,
    make_cosine_rerank_fn,
)

setup_env()

P_VALUES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

# Multi-ref-relevant benches + single-ref controls (should be flat in p).
BENCHES = [
    "engagement_disjoint", "edge_conjunctive_temporal", "notin_multi_interval",
    "composition", "adversarial", "hard_bench", "axis", "cotemporal",
    "lattice", "allen", "polarity",
]


async def run_bench(bench: str, embed_fn, rerank_fn) -> dict[float, dict] | None:
    loaded = _load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [DocV7(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetrieverV7Direct(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)
    out: dict[float, dict] = {}
    for p in P_VALUES:
        vd.score_p = p
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[p] = _metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print("=== power-mean exponent sweep (cross-ref combiner) ===\n", flush=True)
    hdr = f"{'bench':26s} " + " ".join(f"p={p:<5g}" for p in P_VALUES) + "    n"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    for bench in BENCHES:
        try:
            res = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:26s} ERROR: {e}", flush=True)
            continue
        if res is None:
            print(f"{bench:26s} SKIPPED", flush=True)
            continue
        rows[bench] = res
        cells = " ".join(f"{res[p]['R@1']:<7.3f}" for p in P_VALUES)
        n = res[P_VALUES[0]]["n"]
        print(f"{bench:26s} {cells}  {n:>4d}", flush=True)
    if rows:
        n = len(rows)
        print("-" * len(hdr), flush=True)
        macro = " ".join(
            f"{sum(r[p]['R@1'] for r in rows.values()) / n:<7.3f}"
            for p in P_VALUES
        )
        print(f"{'MACRO R@1':26s} {macro}", flush=True)
        macro5 = " ".join(
            f"{sum(r[p]['R@5'] for r in rows.values()) / n:<7.3f}"
            for p in P_VALUES
        )
        print(f"{'MACRO R@5':26s} {macro5}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
