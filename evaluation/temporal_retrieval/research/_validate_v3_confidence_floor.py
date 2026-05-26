"""v3 single-pass + confidence_floor sweep.

v3 leaks unanchored-duration extractions at confidence ~0.6 (e.g.
"6 weeks", "12 minutes", "3 hours" in postmortem docs). These pass
the default 0.5 floor and crowd the precedents-bench pool, costing
the +0.083 R@5 win that v2.1's two-pass extractor captures via its
PASS2 "refuse to fabricate" filter.

Test whether bumping confidence_floor recovers the precedents win
without hurting other benches. The other benches' high-confidence
extractions (≥0.8) should be unaffected.

Variants (all v3 single-pass extractor, only floor differs):
  v3_f0.5  - default
  v3_f0.7  - drop everything below 0.7
  v3_f0.8  - drop everything below 0.8

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_v3_confidence_floor
"""

from __future__ import annotations

import asyncio
import json

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval.extractor_v3 import TemporalExtractorV3

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()

BENCHES = {
    "precedents": (
        "precedents_docs.jsonl",
        "precedents_queries.jsonl",
        "precedents_gold.jsonl",
    ),
    "timeless_policies": (
        "timeless_policies_docs.jsonl",
        "timeless_policies_queries.jsonl",
        "timeless_policies_gold.jsonl",
    ),
    "composition": (
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
    ),
    "adversarial": (
        "adversarial_docs.jsonl",
        "adversarial_queries.jsonl",
        "adversarial_gold.jsonl",
    ),
    "realq_v2": (
        "realq_v2_docs.jsonl",
        "realq_v2_queries.jsonl",
        "realq_v2_gold.jsonl",
    ),
    "hard_bench": (
        "hard_bench_docs.jsonl",
        "hard_bench_queries.jsonl",
        "hard_bench_gold.jsonl",
    ),
    "ambiguous_year": (
        "ambiguous_year_docs.jsonl",
        "ambiguous_year_queries.jsonl",
        "ambiguous_year_gold.jsonl",
    ),
}


VARIANTS = {
    "v3_f0.5": 0.5,
    "v3_f0.7": 0.7,
    "v3_f0.8": 0.8,
}


async def evaluate(
    bench: str,
    variant_name: str,
    floor: float,
    embed_fn,
    rerank_fn,
) -> dict:
    docs_file, queries_file, gold_file = BENCHES[bench]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(
        docs_file, queries_file, gold_file
    )
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
        for d in docs_jsonl
    ]

    retriever = TemporalRetriever(
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        extractor=TemporalExtractorV3(),
        confidence_floor=floor,
    )
    await retriever.index(docs)

    total_ivs = sum(len(v) for v in retriever.doc_intervals().values())

    n_eval = 0
    n_r5 = 0
    n_r1 = 0
    all_recall_at_5: list[float] = []
    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_eval += 1
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
        first_gold = next(
            (i + 1 for i, d in enumerate(ranking) if d in gold_set), None
        )
        if first_gold is not None:
            if first_gold <= 1:
                n_r1 += 1
            if first_gold <= 5:
                n_r5 += 1
        top5 = set(ranking[:5])
        all_recall_at_5.append(len(top5 & gold_set) / len(gold_set))

    return {
        "bench": bench,
        "variant": variant_name,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "all_recall@5": sum(all_recall_at_5) / max(1, len(all_recall_at_5)),
        "n_eval": n_eval,
        "total_ivs": total_ivs,
    }


async def main() -> None:
    print("Loading embed_fn + rerank_fn...", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()

    rows: list[dict] = []
    for bench in BENCHES:
        print(f"\n=== bench: {bench} ===", flush=True)
        for vname, floor in VARIANTS.items():
            print(f"  {vname} (floor={floor})...", flush=True)
            r = await evaluate(bench, vname, floor, embed_fn, rerank_fn)
            print(
                f"    R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} "
                f"all_R@5={r['all_recall@5']:.3f} ivs={r['total_ivs']}",
                flush=True,
            )
            rows.append(r)

    print("\n" + "=" * 100, flush=True)
    print(
        f"{'bench':22s} {'variant':10s} {'R@1':>6s} {'R@5':>6s} "
        f"{'all_R@5':>8s} {'ivs':>5s} {'dR@5':>8s}",
        flush=True,
    )
    print("-" * 100, flush=True)
    for bench in BENCHES:
        baseline = next(
            r for r in rows if r["bench"] == bench and r["variant"] == "v3_f0.5"
        )
        for vname in VARIANTS:
            r = next(
                row
                for row in rows
                if row["bench"] == bench and row["variant"] == vname
            )
            d5 = r["R@5"] - baseline["R@5"]
            tag5 = "" if vname == "v3_f0.5" else f"{d5:+.3f}"
            print(
                f"{bench:22s} {vname:10s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} "
                f"{r['all_recall@5']:>8.3f} {r['total_ivs']:>5d} {tag5:>8s}",
                flush=True,
            )
        print("-" * 100, flush=True)

    print("\nMacro R@5 by variant:", flush=True)
    for vname in VARIANTS:
        rs = [r for r in rows if r["variant"] == vname]
        macro_r5 = sum(r["R@5"] for r in rs) / len(rs)
        print(f"  {vname:10s}  R@5={macro_r5:.3f}", flush=True)

    out_path = ROOT / "v3_confidence_floor_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
