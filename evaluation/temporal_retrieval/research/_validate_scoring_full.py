"""Factor scoring effect from prompt/simplification effect on the full
35-bench suite.

Four variants on cosine base (no cross-encoder):
  v31_mul  : v3.1 extractor + multiplicative scoring     (closest to historical default)
  v32_mul  : v3.2 extractor + multiplicative scoring
  v32_add  : v3.2 extractor + additive scoring           (current shipped)
  v32_hyb  : v3.2 extractor + hybrid_mm_ar scoring

  v31_mul -> v32_mul : isolates extractor PROMPT change
  v32_mul vs v32_add vs v32_hyb : isolates SCORING change

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_scoring_full
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


BENCH_NAMES = [
    "adversarial", "allen", "ambiguous_year", "ambiguous_year_adv",
    "axis", "causal_relative", "composition", "cotemporal",
    "dense_cluster", "disc", "edge_conjunctive_temporal", "edge_era_refs",
    "edge_multi_te_doc", "edge_relative_time", "era", "goldilocks",
    "goldilocks_v2", "hard_bench", "hard_dense_cluster", "latest_recent",
    "lattice", "mixed_cue", "negation_temporal", "notin_multi_interval",
    "open_ended_date", "polarity", "precedents", "realq", "realq_deictic",
    "realq_v2", "sensitivity_curated", "speculative_anchors",
    "temporal_essential", "timeless_policies", "utterance",
]

BENCHES = {
    name: (f"{name}_docs.jsonl", f"{name}_queries.jsonl", f"{name}_gold.jsonl")
    for name in BENCH_NAMES
}


VARIANTS = {
    "v31_mul": ("v3.1", "multiplicative"),
    "v32_mul": ("v3.2", "multiplicative"),
    "v32_add": ("v3.2", "additive"),
    "v32_hyb": ("v3.2", "hybrid_mm_ar"),
}


def make_cosine_rerank_fn(embed_fn):
    async def cosine_rerank(query, doc_texts):
        if not doc_texts: return []
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
    try:
        docs_jsonl, queries, gold_rows = load_bench_jsonl(docs_file, queries_file, gold_file)
    except FileNotFoundError as e:
        return {"bench": bench, "variant": vname, "error": str(e)}
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl]

    extractor_label, scoring = VARIANTS[vname]
    extractor = TemporalExtractorV3() if extractor_label == "v3.1" else TemporalExtractorV3_2()

    retriever = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        extractor=extractor, scoring=scoring,
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

    return {
        "bench": bench, "variant": vname,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "all_R@5": sum(all_r5) / max(1, len(all_r5)),
        "n_eval": n_eval,
    }


async def main():
    print(f"Loading embed_fn... ({len(BENCHES)} benches x {len(VARIANTS)} variants)", flush=True)
    embed_fn  = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    rows = []
    for bench in BENCHES:
        print(f"\n=== {bench} ===", flush=True)
        for vname in VARIANTS:
            print(f"  {vname}...", flush=True)
            r = await evaluate(bench, vname, embed_fn, rerank_fn)
            if "error" in r:
                print(f"    ERROR: {r['error']}", flush=True)
            else:
                print(f"    R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} all_R@5={r['all_R@5']:.3f} n={r['n_eval']}", flush=True)
            rows.append(r)

    # Summary
    print("\n" + "=" * 110, flush=True)
    print(f"{'bench':28s} " + " ".join(f"{v:>8s}" for v in VARIANTS) + "   (R@1 / R@5 / all_R@5)", flush=True)
    print("-" * 110, flush=True)

    macro = {v: {"R@1": [], "R@5": [], "all_R@5": []} for v in VARIANTS}
    for bench in BENCHES:
        # one row per metric
        for metric in ("R@1", "R@5", "all_R@5"):
            vals = []
            for vname in VARIANTS:
                try:
                    r = next(x for x in rows if x["bench"] == bench and x["variant"] == vname and "error" not in x)
                    vals.append(r[metric])
                    if metric == "R@1":
                        macro[vname]["R@1"].append(r["R@1"])
                        macro[vname]["R@5"].append(r["R@5"])
                        macro[vname]["all_R@5"].append(r["all_R@5"])
                except StopIteration:
                    vals.append(None)
            label = f"{bench} [{metric}]" if metric == "R@1" else f"        [{metric}]"
            cells = " ".join((f"{v:>8.3f}" if v is not None else "  N/A   ") for v in vals)
            print(f"{label:28s} {cells}", flush=True)
        print("-" * 110, flush=True)

    print("\nMACRO across benches that ran:", flush=True)
    print(f"{'variant':8s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s}", flush=True)
    for vname in VARIANTS:
        m1 = sum(macro[vname]["R@1"]) / max(1, len(macro[vname]["R@1"]))
        m5 = sum(macro[vname]["R@5"]) / max(1, len(macro[vname]["R@5"]))
        ma = sum(macro[vname]["all_R@5"]) / max(1, len(macro[vname]["all_R@5"]))
        print(f"{vname:8s} {m1:>6.3f} {m5:>6.3f} {ma:>8.3f}", flush=True)

    out_path = ROOT / "scoring_full_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
