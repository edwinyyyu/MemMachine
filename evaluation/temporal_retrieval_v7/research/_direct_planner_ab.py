"""V1 baseline vs V7-legacy-planner vs V7-direct-planner.

Runs all three retrievers on a representative set of benches that
discriminate (avoids saturated ones). Validates the direct-planner
approach quantitatively against the architectural alternative.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._direct_planner_ab
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from temporal_retrieval_min import Doc as DocV1
from temporal_retrieval_min import TemporalRetriever
from temporal_retrieval_v7 import (
    Doc as DocV7,
)
from temporal_retrieval_v7 import (
    TemporalRetrieverV7,
    TemporalRetrieverV7Direct,
)

from temporal_retrieval.research._common import (
    DATA_DIR,
    make_embed_fn,
    setup_env,
)
from temporal_retrieval_v7.research._full_ab import (
    _load_bench,
    _metrics,
    make_cosine_rerank_fn,
)

setup_env()

# Discriminating benches (avoid the 22 saturated ones for signal density).
# Mix of high-frequency wins for V7 and the V7-targeted benches.
FOCUS_BENCHES = [
    "adversarial",
    "composition",
    "edge_conjunctive_temporal",
    "engagement_disjoint",
    "hard_bench",
    "mixed_cue",
    "negation_temporal",
    "polarity",
    "realq",
    "realq_v2",
    "sensitivity_curated",
    "v7_compound_hard",
    "v7_doc_directional",
]


async def run_one_bench(bench: str, embed_fn, rerank_fn) -> dict | None:
    docs_jsonl, queries, gold = _load_bench(bench)
    if docs_jsonl is None:
        return None

    docs_v1 = [DocV1(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
               for d in docs_jsonl]
    docs_v7 = [DocV7(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
               for d in docs_jsonl]

    v1 = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    v7_legacy = TemporalRetrieverV7(embed_fn=embed_fn, rerank_fn=rerank_fn)
    v7_direct = TemporalRetrieverV7Direct(embed_fn=embed_fn, rerank_fn=rerank_fn)

    await v1.index(docs_v1)
    await v7_legacy.index(docs_v7)
    await v7_direct.index(docs_v7)

    v1_rk, vL_rk, vD_rk = {}, {}, {}
    for q in queries:
        qid = q["query_id"]
        r1 = await v1.query(q["text"], q["ref_time"], k=10)
        rL = await v7_legacy.query(q["text"], q["ref_time"], k=10)
        rD = await v7_direct.query(q["text"], q["ref_time"], k=10)
        v1_rk[qid] = [r.doc_id for r in r1]
        vL_rk[qid] = [r.doc_id for r in rL]
        vD_rk[qid] = [r.doc_id for r in rD]

    return {
        "v1": _metrics(v1_rk, gold),
        "v7_legacy": _metrics(vL_rk, gold),
        "v7_direct": _metrics(vD_rk, gold),
    }


async def main():
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    print(f"=== V1 vs V7-legacy vs V7-direct over {len(FOCUS_BENCHES)} benches ===\n",
          flush=True)
    header = (
        f"{'bench':28s}"
        f"  {'V1 R@1':>7s} {'L R@1':>7s} {'D R@1':>7s}"
        f"  {'V1 R@5':>7s} {'L R@5':>7s} {'D R@5':>7s}"
        f"  {'ΔL R@1':>7s} {'ΔD R@1':>7s}  {'n':>4s}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    rows: dict[str, dict] = {}
    for bench in FOCUS_BENCHES:
        try:
            res = await run_one_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:28s}  ERROR: {e}", flush=True)
            continue
        if res is None:
            print(f"{bench:28s}  SKIPPED", flush=True)
            continue
        rows[bench] = res
        v1m, vLm, vDm = res["v1"], res["v7_legacy"], res["v7_direct"]
        dL = vLm["R@1"] - v1m["R@1"]
        dD = vDm["R@1"] - v1m["R@1"]
        print(
            f"  {bench:26s}"
            f"  {v1m['R@1']:>7.3f} {vLm['R@1']:>7.3f} {vDm['R@1']:>7.3f}"
            f"  {v1m['R@5']:>7.3f} {vLm['R@5']:>7.3f} {vDm['R@5']:>7.3f}"
            f"  {dL:>+7.3f} {dD:>+7.3f}  {v1m['n']:>4d}",
            flush=True,
        )

    # Macro
    if rows:
        n = len(rows)
        v1_r1 = sum(r["v1"]["R@1"] for r in rows.values()) / n
        vL_r1 = sum(r["v7_legacy"]["R@1"] for r in rows.values()) / n
        vD_r1 = sum(r["v7_direct"]["R@1"] for r in rows.values()) / n
        v1_r5 = sum(r["v1"]["R@5"] for r in rows.values()) / n
        vL_r5 = sum(r["v7_legacy"]["R@5"] for r in rows.values()) / n
        vD_r5 = sum(r["v7_direct"]["R@5"] for r in rows.values()) / n
        print("-" * len(header), flush=True)
        print(
            f"  {'MACRO':26s}"
            f"  {v1_r1:>7.3f} {vL_r1:>7.3f} {vD_r1:>7.3f}"
            f"  {v1_r5:>7.3f} {vL_r5:>7.3f} {vD_r5:>7.3f}"
            f"  {vL_r1-v1_r1:>+7.3f} {vD_r1-v1_r1:>+7.3f}  n_benches={n}",
            flush=True,
        )

    out = Path(__file__).resolve().parent.parent / "ab_direct_planner.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
