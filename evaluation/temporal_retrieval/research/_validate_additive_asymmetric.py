"""Asymmetric mask/recency weight sweep for additive scoring.

The prior `_validate_additive_scoring.py` swept symmetric weights
(mask_weight == recency_weight ∈ {0.5, 1.0, 2.0}). w=0.5 hit macro
R@5 +0.051 over multiplicative but regressed adversarial −0.057.

Mask is the actual temporal-constraint signal; recency is a
chronological tiebreaker. Hypothesis: skewing toward mask should
recover the adversarial regression (mask=1 docs pulled farther above
mask=0 docs, approaching gate-like behavior without zeroing) while
preserving the macro lift.

Variants:
  multiplicative   - production baseline
  add_m05_r05      - prior best, equal weights (control)
  add_m08_r02      - user's suggested 0.8/0.2 skew
  add_m1_r02       - mask=1.0, recency=0.2
  add_m1_r05       - mask=1.0, recency=0.5
  add_m2_r02       - mask=2.0, recency=0.2 (aggressive mask)

Tests across the 7 standing benches. Continues the additive-scoring
investigation — keeps reranker for apples-to-apples with prior run.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_additive_asymmetric
"""

from __future__ import annotations

import asyncio
import json

from temporal_retrieval import Doc, TemporalRetriever

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()

BENCHES = {
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
    "timeless_policies": (
        "timeless_policies_docs.jsonl",
        "timeless_policies_queries.jsonl",
        "timeless_policies_gold.jsonl",
    ),
    "precedents": (
        "precedents_docs.jsonl",
        "precedents_queries.jsonl",
        "precedents_gold.jsonl",
    ),
}


VARIANTS = {
    "multiplicative": {"scoring": "multiplicative"},
    "add_m05_r05": {"scoring": "additive", "mask_weight": 0.5, "recency_weight": 0.5},
    "add_m08_r02": {"scoring": "additive", "mask_weight": 0.8, "recency_weight": 0.2},
    "add_m1_r02": {"scoring": "additive", "mask_weight": 1.0, "recency_weight": 0.2},
    "add_m1_r05": {"scoring": "additive", "mask_weight": 1.0, "recency_weight": 0.5},
    "add_m2_r02": {"scoring": "additive", "mask_weight": 2.0, "recency_weight": 0.2},
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

    retriever = TemporalRetriever(
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        **variant_kw,
    )
    await retriever.index(docs)

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
                f"all_R@5={r['all_recall@5']:.3f}",
                flush=True,
            )
            rows.append(r)

    print("\n" + "=" * 110, flush=True)
    print(
        f"{'bench':22s} {'variant':14s} {'R@1':>6s} {'R@5':>6s} "
        f"{'all_R@5':>8s} {'dR@5':>8s} {'dR@1':>8s}",
        flush=True,
    )
    print("-" * 110, flush=True)
    for bench in BENCHES:
        baseline = next(
            r
            for r in rows
            if r["bench"] == bench and r["variant"] == "multiplicative"
        )
        for vname in VARIANTS:
            r = next(
                row
                for row in rows
                if row["bench"] == bench and row["variant"] == vname
            )
            d5 = r["R@5"] - baseline["R@5"]
            d1 = r["R@1"] - baseline["R@1"]
            tag5 = "" if vname == "multiplicative" else f"{d5:+.3f}"
            tag1 = "" if vname == "multiplicative" else f"{d1:+.3f}"
            print(
                f"{bench:22s} {vname:14s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} "
                f"{r['all_recall@5']:>8.3f} {tag5:>8s} {tag1:>8s}",
                flush=True,
            )
        print("-" * 110, flush=True)

    # Macro averages per variant
    print("\nMacro R@5 by variant:", flush=True)
    for vname in VARIANTS:
        rs = [r for r in rows if r["variant"] == vname]
        macro_r5 = sum(r["R@5"] for r in rs) / len(rs)
        macro_r1 = sum(r["R@1"] for r in rs) / len(rs)
        print(f"  {vname:14s}  R@1={macro_r1:.3f}  R@5={macro_r5:.3f}", flush=True)

    out_path = ROOT / "additive_asymmetric_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
