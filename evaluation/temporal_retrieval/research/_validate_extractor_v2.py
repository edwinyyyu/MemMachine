"""Validate v2 unified-envelope extractor against v1 + ablation config.

The v1 extractor emits 4 kinds (instant/interval/duration/recurrence)
with nested FuzzyInstant/FuzzyInterval shapes and a `best` field.
Validated ablations showed:
  - recurrence expansion: dead weight (reframe to dtstart matches full)
  - best_us: decorative (midpoint matches production 6/7 benches)
  - interval END: load-bearing (keep envelope width)

v2 extractor emits ONE flat envelope shape (earliest/latest/granularity)
and skips kind decisions, recurrence rules, and the best field. The
question this script answers: does the LLM, when freed from the kind
decision and given a simpler schema, produce envelopes that match the
v1+ablation-config baseline's recall?

Comparison:
  v1_ablated   - v1 extractor + reframe_kinds={"recurrence"} +
                 best_strategy="midpoint" (config-knob proxy for v2)
  v2_unified   - v2 extractor; downstream pipeline unchanged

Benches: precedents (34 docs) + timeless_policies (35 docs). These are
the two failure-mode reproducers where mask softening and the ablation
config knobs deliver their biggest wins, so they're the highest-risk
case for the v2 extractor to lose ground on.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_extractor_v2
"""

from __future__ import annotations

import asyncio
import json

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval.extractor_v2 import TemporalExtractorV2
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
    "ambiguous_year": (
        "ambiguous_year_docs.jsonl",
        "ambiguous_year_queries.jsonl",
        "ambiguous_year_gold.jsonl",
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
}


VARIANTS: dict[str, dict] = {
    # Note: v1 (legacy two-pass with discriminated-union schema) was deleted
    # when v3.1 shipped. This script compares v2.1 two-pass against
    # v3.1 single-pass on the unified TimeEnvelope schema.
    "v2_unified": {
        "extractor": "v2",  # sentinel; instantiated in evaluate()
    },
    "v3_single_pass": {
        "extractor": "v3",  # sentinel; instantiated in evaluate()
    },
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
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
        for d in docs_jsonl
    ]

    kw = dict(variant_kw)
    if kw.get("extractor") == "v2":
        kw["extractor"] = TemporalExtractorV2()
    elif kw.get("extractor") == "v3":
        kw["extractor"] = TemporalExtractorV3()
    retriever = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn, **kw)
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
        for vname, vkw in VARIANTS.items():
            print(f"  {vname}...", flush=True)
            r = await evaluate(bench, vname, vkw, embed_fn, rerank_fn)
            print(
                f"    R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} "
                f"all_R@5={r['all_recall@5']:.3f} ivs={r['total_ivs']}",
                flush=True,
            )
            rows.append(r)

    print("\n" + "=" * 100, flush=True)
    print(
        f"{'bench':22s} {'variant':14s} {'R@1':>6s} {'R@5':>6s} "
        f"{'all_R@5':>8s} {'ivs':>7s} {'dR@5':>8s} {'dR@1':>8s}",
        flush=True,
    )
    print("-" * 100, flush=True)
    for bench in BENCHES:
        baseline = next(
            r for r in rows if r["bench"] == bench and r["variant"] == "v2_unified"
        )
        for vname in VARIANTS:
            r = next(
                row
                for row in rows
                if row["bench"] == bench and row["variant"] == vname
            )
            d5 = r["R@5"] - baseline["R@5"]
            d1 = r["R@1"] - baseline["R@1"]
            tag5 = "" if vname == "v1_ablated" else f"{d5:+.3f}"
            tag1 = "" if vname == "v1_ablated" else f"{d1:+.3f}"
            print(
                f"{bench:22s} {vname:14s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} "
                f"{r['all_recall@5']:>8.3f} {r['total_ivs']:>7d} "
                f"{tag5:>8s} {tag1:>8s}",
                flush=True,
            )
        print("-" * 100, flush=True)

    out_path = ROOT / "extractor_v2_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
