"""V3.4 extractor A/B on v7_doc_directional + sanity benches.

Tests whether the new doc-side directional rules (since/after/until
bounded at ref_time) close the -0.083 regression on v7_doc_directional.
Includes a few control benches (engagement_disjoint, edge_conjunctive_temporal,
hard_bench) to ensure no regression elsewhere.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._v34_extractor_ab
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from temporal_retrieval_min import Doc as DocV1
from temporal_retrieval_min import TemporalRetriever
from temporal_retrieval_v7 import (
    Doc as DocV7,
    TemporalRetrieverV7Direct,
)
from temporal_retrieval_v7.extractor_v3_4 import TemporalExtractorV3_4

from temporal_retrieval.research._common import (
    make_embed_fn,
    setup_env,
)
from temporal_retrieval_v7.research._full_ab import (
    _load_bench,
    _metrics,
    make_cosine_rerank_fn,
)

setup_env()

# Focus on doc-directional + a few controls
BENCHES = [
    "v7_doc_directional",   # the gap we're closing
    "engagement_disjoint",  # control: should not regress
    "edge_conjunctive_temporal",  # control
    "hard_bench",           # control: largest bench
    "v7_compound_hard",     # control
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
    # V7-Direct with V3.3 (current default)
    vd_v33 = TemporalRetrieverV7Direct(embed_fn=embed_fn, rerank_fn=rerank_fn)
    # V7-Direct with V3.4
    vd_v34 = TemporalRetrieverV7Direct(
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        extractor=TemporalExtractorV3_4(),
    )

    await v1.index(docs_v1)
    await vd_v33.index(docs_v7)
    await vd_v34.index(docs_v7)

    v1_rk, v33_rk, v34_rk = {}, {}, {}
    for q in queries:
        r1 = await v1.query(q["text"], q["ref_time"], k=10)
        r33 = await vd_v33.query(q["text"], q["ref_time"], k=10)
        r34 = await vd_v34.query(q["text"], q["ref_time"], k=10)
        v1_rk[q["query_id"]] = [r.doc_id for r in r1]
        v33_rk[q["query_id"]] = [r.doc_id for r in r33]
        v34_rk[q["query_id"]] = [r.doc_id for r in r34]

    return {
        "v1": _metrics(v1_rk, gold),
        "v33": _metrics(v33_rk, gold),
        "v34": _metrics(v34_rk, gold),
    }


async def main():
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== V1 vs V7-Direct/v3.3 vs V7-Direct/v3.4 ===\n", flush=True)
    header = (
        f"{'bench':28s}"
        f"  {'V1':>6s} {'v33':>6s} {'v34':>6s}"
        f"  {'Δ33':>6s} {'Δ34':>6s}  n"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    rows = {}
    for bench in BENCHES:
        try:
            res = await run_one_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:28s}  ERROR: {e}", flush=True)
            continue
        if res is None:
            print(f"{bench:28s}  SKIPPED", flush=True)
            continue
        rows[bench] = res
        v1m, v33m, v34m = res["v1"], res["v33"], res["v34"]
        print(
            f"  {bench:26s}  {v1m['R@1']:>6.3f} {v33m['R@1']:>6.3f} {v34m['R@1']:>6.3f}"
            f"  {v33m['R@1']-v1m['R@1']:>+6.3f} {v34m['R@1']-v1m['R@1']:>+6.3f}  {v1m['n']:>2d}",
            flush=True,
        )

    out = Path(__file__).resolve().parent.parent / "ab_v34_extractor.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
