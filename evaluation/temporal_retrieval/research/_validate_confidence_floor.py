"""Sweep `confidence_floor` to confirm or refute the 0.5 default.

The v3 extractor's refuse-to-fabricate principle assigns confidence=0 to
truly-unanchorable phrases and high confidence (0.7-1.0) to clean
extractions. Mid-range confidences (0.4-0.7) appear for under-specified
or borderline cases (e.g., fuzzy years, eras with implicit anchoring).

The floor controls which envelopes survive into the temporal-match
scoring. Too low → noisy extractions pollute the match score. Too
high → legitimate-but-uncertain extractions are discarded, losing
their constraint contribution.

Variants:
  c_00 : confidence_floor = 0.0   (keep everything)
  c_03 : 0.3
  c_05 : 0.5  (current default)
  c_07 : 0.7
  c_09 : 0.9                       (only very-confident)

Run with cosine base (no cross-encoder) and additive scoring (the
shipped configuration).

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_confidence_floor
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


VARIANTS = {
    "c_00": {"confidence_floor": 0.0},
    "c_03": {"confidence_floor": 0.3},
    "c_05": {"confidence_floor": 0.5},
    "c_07": {"confidence_floor": 0.7},
    "c_09": {"confidence_floor": 0.9},
}


def make_cosine_rerank_fn(embed_fn):
    async def cosine_rerank(query, doc_texts):
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


async def evaluate(bench, vname, vkw, embed_fn, rerank_fn):
    docs_file, queries_file, gold_file = BENCHES[bench]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(docs_file, queries_file, gold_file)
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl]
    retriever = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn, **vkw)
    await retriever.index(docs)

    n_eval = n_r1 = n_r5 = 0
    all_r5 = []
    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set: continue
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
        "bench": bench, "variant": vname,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "all_R@5": sum(all_r5) / max(1, len(all_r5)),
        "n_eval": n_eval,
    }


async def main():
    print("Loading embed_fn...", flush=True)
    embed_fn  = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    rows = []
    for bench in BENCHES:
        print(f"\n=== {bench} ===", flush=True)
        for vname, vkw in VARIANTS.items():
            print(f"  {vname}...", flush=True)
            r = await evaluate(bench, vname, vkw, embed_fn, rerank_fn)
            print(f"    R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} all_R@5={r['all_R@5']:.3f}", flush=True)
            rows.append(r)

    print("\n" + "=" * 90, flush=True)
    print(f"{'bench':22s} {'variant':6s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s}", flush=True)
    print("-" * 90, flush=True)
    macro = {v: {"R@1": [], "R@5": [], "all_R@5": []} for v in VARIANTS}
    for bench in BENCHES:
        for vname in VARIANTS:
            r = next(x for x in rows if x["bench"] == bench and x["variant"] == vname)
            print(f"{bench:22s} {vname:6s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} {r['all_R@5']:>8.3f}", flush=True)
            macro[vname]["R@1"].append(r["R@1"])
            macro[vname]["R@5"].append(r["R@5"])
            macro[vname]["all_R@5"].append(r["all_R@5"])
        print("-" * 90, flush=True)

    print("\nMACRO:", flush=True)
    print(f"{'variant':6s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s}", flush=True)
    for vname, vals in macro.items():
        m1 = sum(vals["R@1"]) / len(vals["R@1"])
        m5 = sum(vals["R@5"]) / len(vals["R@5"])
        ma = sum(vals["all_R@5"]) / len(vals["all_R@5"])
        print(f"{vname:6s} {m1:>6.3f} {m5:>6.3f} {ma:>8.3f}", flush=True)

    out_path = ROOT / "confidence_floor_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
