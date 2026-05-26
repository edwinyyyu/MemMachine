"""A/B the three scoring composition modes WITHOUT the cross-encoder
reranker. Substitutes cosine into the rerank slot so the pipeline still
runs through normalize_rerank_full / mask / recency, but the base score
is text-embedding-3-small cosine instead of cross-encoder rerank.

Question: with cosine as the only similarity signal, which scoring
composition (multiplicative vs additive vs hybrid mask-mul + recency-add)
gives the best macro R@1 / R@5 / all_R@5?

This is a probe at whether we can drop the cross-encoder dependency
without losing recall. If a scoring mode + cosine matches or beats
the rerank baseline on enough benches, the rerank is dispensable.

Variants:
  mul    : base * mask * (1 + recency_alpha * rec)         — production
  add    : base + mask_weight * mask + recency_weight * rec
  hyb    : base * mask + recency_weight * rec

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_no_rerank_scoring
"""
from __future__ import annotations

import asyncio
import json

import numpy as np

from temporal_retrieval import Doc, TemporalRetriever

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    setup_env,
)

setup_env()


BENCHES = {
    "realq_v2": (
        "realq_v2_docs.jsonl",
        "realq_v2_queries.jsonl",
        "realq_v2_gold.jsonl",
    ),
    "ambiguous_year": (
        "ambiguous_year_docs.jsonl",
        "ambiguous_year_queries.jsonl",
        "ambiguous_year_gold.jsonl",
    ),
    "composition": (
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
    ),
    "hard_bench": (
        "hard_bench_docs.jsonl",
        "hard_bench_queries.jsonl",
        "hard_bench_gold.jsonl",
    ),
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
    "temporal_essential": (
        "temporal_essential_docs.jsonl",
        "temporal_essential_queries.jsonl",
        "temporal_essential_gold.jsonl",
    ),
}


# Scoring variants
VARIANTS: dict[str, dict] = {
    "mul":       {"scoring": "multiplicative", "recency_alpha": 3.0},
    "add_11":    {"scoring": "additive",       "mask_weight": 1.0, "recency_weight": 1.0},
    "add_80_20": {"scoring": "additive",       "mask_weight": 0.8, "recency_weight": 0.2},
    "hyb_30":    {"scoring": "hybrid_mm_ar",   "recency_weight": 0.3},
}


def make_cosine_rerank_fn(embed_fn):
    """Rerank slot replacement: returns cosine of query embedding vs
    each pool doc's embedding. Re-embeds for simplicity — embeddings
    are cheap.
    """
    async def cosine_rerank(query: str, doc_texts: list[str]) -> list[float]:
        if not doc_texts:
            return []
        qe = (await embed_fn([query]))[0]
        des = await embed_fn(doc_texts)
        qn = float(np.linalg.norm(qe)) or 1e-9
        out = []
        for de in des:
            den = float(np.linalg.norm(de)) or 1e-9
            out.append(float(np.dot(qe, de) / (qn * den)))
        return out

    return cosine_rerank


async def evaluate(
    bench: str, vname: str, vkw: dict, embed_fn, rerank_fn
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

    retriever = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn, **vkw)
    await retriever.index(docs)

    n_eval = n_r1 = n_r5 = 0
    all_r5: list[float] = []
    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_eval += 1
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
        first = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
        if first is not None:
            if first <= 1: n_r1 += 1
            if first <= 5: n_r5 += 1
        top5 = set(ranking[:5])
        all_r5.append(len(top5 & gold_set) / len(gold_set))

    return {
        "bench": bench,
        "variant": vname,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "all_R@5": sum(all_r5) / max(1, len(all_r5)),
        "n_eval": n_eval,
    }


async def main() -> None:
    print("Loading embed_fn (no cross-encoder)...", flush=True)
    embed_fn  = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    rows: list[dict] = []
    for bench in BENCHES:
        print(f"\n=== {bench} ===", flush=True)
        for vname, vkw in VARIANTS.items():
            print(f"  {vname}...", flush=True)
            r = await evaluate(bench, vname, vkw, embed_fn, rerank_fn)
            print(f"    R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} all_R@5={r['all_R@5']:.3f}", flush=True)
            rows.append(r)

    print("\n" + "=" * 100, flush=True)
    print(f"{'bench':22s} {'variant':12s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s}", flush=True)
    print("-" * 100, flush=True)
    macro = {v: {"R@1": [], "R@5": [], "all_R@5": []} for v in VARIANTS}
    for bench in BENCHES:
        for vname in VARIANTS:
            r = next(row for row in rows if row["bench"] == bench and row["variant"] == vname)
            print(f"{bench:22s} {vname:12s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} {r['all_R@5']:>8.3f}", flush=True)
            macro[vname]["R@1"].append(r["R@1"])
            macro[vname]["R@5"].append(r["R@5"])
            macro[vname]["all_R@5"].append(r["all_R@5"])
        print("-" * 100, flush=True)

    print("\nMACRO (no rerank, cosine base):", flush=True)
    print(f"{'variant':12s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s}", flush=True)
    for vname, vals in macro.items():
        m1 = sum(vals["R@1"]) / len(vals["R@1"])
        m5 = sum(vals["R@5"]) / len(vals["R@5"])
        ma = sum(vals["all_R@5"]) / len(vals["all_R@5"])
        print(f"{vname:12s} {m1:>6.3f} {m5:>6.3f} {ma:>8.3f}", flush=True)

    out_path = ROOT / "no_rerank_scoring_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
