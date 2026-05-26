"""A/B v3.1 (emit-then-filter at conf_floor=0.5) vs v3.2 (skip-don't-emit).

V3.2 removes the confidence field from the LLM output and instructs the
LLM to skip phrases it can't anchor instead of emitting them with
conf=0. If v3.2 matches v3.1 on benches, we can drop the confidence
field + confidence_floor entirely from the schema and parameter set.

Variants:
  v3_1   : current production. Extractor v3.1 + retriever
           confidence_floor=0.5.
  v3_2   : new variant. Extractor v3.2 (skip-don't-emit) + retriever
           confidence_floor=0.0 (no-op; all v3.2 envelopes are conf=1.0).

Benches: same 7 used in the no-rerank scoring + confidence-floor
ablations.

Uses cosine in the rerank slot (no cross-encoder) and additive scoring
(the shipped configuration).

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_extractor_v3_2
"""
from __future__ import annotations

import asyncio
import json

import numpy as np

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval.extractor_v3 import TemporalExtractorV3
from temporal_retrieval.extractor_v3_2 import TemporalExtractorV3_2

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


async def evaluate(bench, vname, embed_fn, rerank_fn):
    docs_file, queries_file, gold_file = BENCHES[bench]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(docs_file, queries_file, gold_file)
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl]

    if vname == "v3_1":
        extractor = TemporalExtractorV3()
        floor = 0.5
    elif vname == "v3_2":
        extractor = TemporalExtractorV3_2()
        floor = 0.0  # no-op; all v3.2 envelopes are conf=1.0
    else:
        raise ValueError(vname)

    retriever = TemporalRetriever(
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        extractor=extractor,
        confidence_floor=floor,
    )
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

    # also report how many envelopes the extractor produced (after floor)
    total_envelopes = sum(len(v) for v in retriever.doc_envelopes().values())

    return {
        "bench": bench, "variant": vname,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "all_R@5": sum(all_r5) / max(1, len(all_r5)),
        "n_eval": n_eval,
        "total_envelopes": total_envelopes,
    }


async def main():
    print("Loading embed_fn...", flush=True)
    embed_fn  = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    rows = []
    for bench in BENCHES:
        print(f"\n=== {bench} ===", flush=True)
        for vname in ("v3_1", "v3_2"):
            print(f"  {vname}...", flush=True)
            r = await evaluate(bench, vname, embed_fn, rerank_fn)
            print(f"    R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} all_R@5={r['all_R@5']:.3f} envs={r['total_envelopes']}", flush=True)
            rows.append(r)

    print("\n" + "=" * 100, flush=True)
    print(f"{'bench':22s} {'variant':6s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s} {'envs':>6s} {'ΔR@1':>7s}", flush=True)
    print("-" * 100, flush=True)
    macro = {v: {"R@1": [], "R@5": [], "all_R@5": [], "envs": []} for v in ("v3_1", "v3_2")}
    for bench in BENCHES:
        base = next(r for r in rows if r["bench"] == bench and r["variant"] == "v3_1")
        for vname in ("v3_1", "v3_2"):
            r = next(x for x in rows if x["bench"] == bench and x["variant"] == vname)
            d1 = r["R@1"] - base["R@1"]
            tag = "" if vname == "v3_1" else f"{d1:+.3f}"
            print(f"{bench:22s} {vname:6s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} {r['all_R@5']:>8.3f} {r['total_envelopes']:>6d} {tag:>7s}", flush=True)
            macro[vname]["R@1"].append(r["R@1"])
            macro[vname]["R@5"].append(r["R@5"])
            macro[vname]["all_R@5"].append(r["all_R@5"])
            macro[vname]["envs"].append(r["total_envelopes"])
        print("-" * 100, flush=True)

    print("\nMACRO:", flush=True)
    print(f"{'variant':6s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s}", flush=True)
    for vname in ("v3_1", "v3_2"):
        vals = macro[vname]
        m1 = sum(vals["R@1"]) / len(vals["R@1"])
        m5 = sum(vals["R@5"]) / len(vals["R@5"])
        ma = sum(vals["all_R@5"]) / len(vals["all_R@5"])
        print(f"{vname:6s} {m1:>6.3f} {m5:>6.3f} {ma:>8.3f}", flush=True)

    out_path = ROOT / "extractor_v3_2_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
