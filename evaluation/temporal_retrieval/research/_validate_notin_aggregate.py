"""A/B test the strict (max-pairs) vs aggregate `not_in` containment
behavior.

Strict (current production): a doc with any one interval fully inside
the excluded window gets containment = 1, leaf factor = 0, and is
dropped. Implemented as max-over-pairs in `excluded_containment`.

Aggregate (alternative): total fraction of doc-time inside excluded
window. Multi-interval docs whose intervals are mostly outside the
excluded window keep most of their score. Implemented in
`excluded_containment_aggregate`.

Tests on:
- notin_multi_interval (new): focused reproducer with multi-interval
  docs and `not in X` queries; gold includes mixed docs that are partly
  in / partly out of the excluded window.
- composition: existing bench; many `in X not in Y` queries but mostly
  single-interval docs.
- adversarial, realq_v2, hard_bench, ambiguous_year: regression check.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_notin_aggregate
"""

from __future__ import annotations

import asyncio
import json

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval.planner import QueryPlanner

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()

BENCHES = {
    "notin_multi_interval": (
        "notin_multi_interval_docs.jsonl",
        "notin_multi_interval_queries.jsonl",
        "notin_multi_interval_gold.jsonl",
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
    "strict": {"notin_aggregate": False},
    "aggregate": {"notin_aggregate": True},
}


async def evaluate(
    bench: str,
    variant_name: str,
    variant_kw: dict,
    embed_fn,
    rerank_fn,
) -> dict:
    docs_file, queries_file, gold_file = BENCHES[bench]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(
        docs_file, queries_file, gold_file
    )
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl
    ]

    planner = QueryPlanner()
    retriever = TemporalRetriever(
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        notin_aggregate=variant_kw["notin_aggregate"],
        planner=planner,
    )
    await retriever.index(docs)

    n_eval = 0
    n_r5 = 0
    n_r1 = 0
    all_recall_at_5: list[float] = []
    n_with_notin = 0
    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_eval += 1
        plan = await planner.plan(q["text"], q["ref_time"])
        if any(leaf.direction == "not_in" for cl in plan.expr for leaf in cl):
            n_with_notin += 1
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
        first_gold = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
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
        "n_with_notin": n_with_notin,
    }


async def main() -> None:
    print("Loading embed_fn + rerank_fn...", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()

    rows: list[dict] = []
    for bench in BENCHES:
        print(f"\n=== bench: {bench} ===", flush=True)
        for vname, vkw in VARIANTS.items():
            print(f"  {vname}...", flush=True)
            r = await evaluate(bench, vname, vkw, embed_fn, rerank_fn)
            print(
                f"    R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} "
                f"all_R@5={r['all_recall@5']:.3f} not_in={r['n_with_notin']}/{r['n_eval']}",
                flush=True,
            )
            rows.append(r)

    print("\n" + "=" * 110, flush=True)
    print(
        f"{'bench':22s} {'variant':10s} {'R@1':>6s} {'R@5':>6s} "
        f"{'all_R@5':>8s} {'not_in':>8s} {'delta_R@5':>10s} {'delta_all':>10s}",
        flush=True,
    )
    print("-" * 110, flush=True)
    for bench in BENCHES:
        baseline = next(
            r for r in rows if r["bench"] == bench and r["variant"] == "strict"
        )
        for vname in VARIANTS:
            r = next(
                row for row in rows if row["bench"] == bench and row["variant"] == vname
            )
            d5 = r["R@5"] - baseline["R@5"]
            da = r["all_recall@5"] - baseline["all_recall@5"]
            tag5 = "" if vname == "strict" else f"{d5:+.3f}"
            taga = "" if vname == "strict" else f"{da:+.3f}"
            print(
                f"{bench:22s} {vname:10s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} "
                f"{r['all_recall@5']:>8.3f} {r['n_with_notin']:>4d}/{r['n_eval']:<3d} "
                f"{tag5:>10s} {taga:>10s}",
                flush=True,
            )
        print("-" * 110, flush=True)

    out_path = ROOT / "notin_aggregate_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
