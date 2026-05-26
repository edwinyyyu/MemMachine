"""Hybrid scoring + multiplicative α sweep.

Settles two open questions:

1. Is α=3 (recency_alpha) for pure multiplicative actually optimal? It
   was inherited as production default but never deeply ablated. α=3
   means recency can scale a doc's score by up to 4×, which is much
   louder than additive's λ_r·recency contribution. If a smaller α
   makes multiplicative win MORE, the prior additive-beats-mult result
   may have been partly an artifact of α=3 being too hot.

2. Does hybrid (multiplicative mask, additive recency) beat both pure
   modes? Mask gate behavior helps composition R@1 / adversarial;
   additive recency softens the recency multiplier without losing the
   tiebreaker.

Variants:
  mult_a0          - multiplicative, α=0 (recency disabled)
  mult_a0.5        - multiplicative, α=0.5
  mult_a1          - multiplicative, α=1
  mult_a2          - multiplicative, α=2
  mult_a3          - production default (α=3)
  mult_a5          - multiplicative, α=5
  hybrid_r0.1      - base·mask + 0.1·recency
  hybrid_r0.2      - base·mask + 0.2·recency
  hybrid_r0.5      - base·mask + 0.5·recency
  add_m08_r02      - current best additive (control)

Tests across the 7 standing benches. Continues additive-scoring
investigation — keeps reranker for apples-to-apples.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_hybrid_and_alpha
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


VARIANTS: dict[str, dict] = {}
for a in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
    VARIANTS[f"mult_a{a}"] = {"scoring": "multiplicative", "recency_alpha": a}
for r in [0.1, 0.2, 0.5]:
    VARIANTS[f"hybrid_r{r}"] = {
        "scoring": "hybrid_mm_ar",
        "recency_weight": r,
    }
VARIANTS["add_m08_r02"] = {
    "scoring": "additive",
    "mask_weight": 0.8,
    "recency_weight": 0.2,
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
        f"{'bench':22s} {'variant':14s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s}",
        flush=True,
    )
    print("-" * 110, flush=True)
    for bench in BENCHES:
        for vname in VARIANTS:
            r = next(
                row
                for row in rows
                if row["bench"] == bench and row["variant"] == vname
            )
            print(
                f"{bench:22s} {vname:14s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} "
                f"{r['all_recall@5']:>8.3f}",
                flush=True,
            )
        print("-" * 110, flush=True)

    print("\nMacro R@1 / R@5 by variant:", flush=True)
    for vname in VARIANTS:
        rs = [r for r in rows if r["variant"] == vname]
        macro_r5 = sum(r["R@5"] for r in rs) / len(rs)
        macro_r1 = sum(r["R@1"] for r in rs) / len(rs)
        print(f"  {vname:14s}  R@1={macro_r1:.3f}  R@5={macro_r5:.3f}", flush=True)

    out_path = ROOT / "hybrid_and_alpha_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
