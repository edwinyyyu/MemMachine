"""Full-bench A/B: DNF planner vs boolean-tree planner.

Cosine base (no cross-encoder rerank), additive scoring, v3.2 extractor.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_planner_tree
"""
from __future__ import annotations

import asyncio
import json

import numpy as np

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval.extractor_v3_2 import TemporalExtractorV3_2
from temporal_retrieval.planner import QueryPlanner
from temporal_retrieval.planner_tree import TreePlanner

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
    try:
        docs_jsonl, queries, gold_rows = load_bench_jsonl(docs_file, queries_file, gold_file)
    except FileNotFoundError as e:
        return {"bench": bench, "variant": vname, "error": str(e)}
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl]

    extractor = TemporalExtractorV3_2()
    if vname == "dnf":
        planner = QueryPlanner()
    elif vname == "tree":
        planner = TreePlanner()
    else:
        raise ValueError(vname)

    retriever = TemporalRetriever(
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        extractor=extractor,
        planner=planner,
    )
    await retriever.index(docs)

    n_eval = n_r1 = n_r5 = 0
    all_r5 = []
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
            if first <= 1:
                n_r1 += 1
            if first <= 5:
                n_r5 += 1
        top5 = set(ranking[:5])
        all_r5.append(len(top5 & gold_set) / len(gold_set))

    return {
        "bench": bench, "variant": vname,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "all_R@5": sum(all_r5) / max(1, len(all_r5)),
        "n_eval": n_eval,
        "planner_stats": planner.stats(),
    }


async def main():
    print(f"Loading embed_fn... ({len(BENCHES)} benches)", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    rows = []
    for bench in BENCHES:
        print(f"\n=== {bench} ===", flush=True)
        for vname in ("dnf", "tree"):
            print(f"  {vname}...", flush=True)
            r = await evaluate(bench, vname, embed_fn, rerank_fn)
            if "error" in r:
                print(f"    ERROR: {r['error']}", flush=True)
            else:
                pf = r["planner_stats"].get("parse_failures", 0)
                print(
                    f"    R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} "
                    f"all_R@5={r['all_R@5']:.3f} n={r['n_eval']} "
                    f"parse_failures={pf}",
                    flush=True,
                )
            rows.append(r)

    # Summary
    print("\n" + "=" * 110, flush=True)
    print(f"{'bench':28s} {'dnf R@1':>9s} {'tree R@1':>9s} {'ΔR@1':>7s}   "
          f"{'dnf R@5':>9s} {'tree R@5':>9s} {'ΔR@5':>7s}", flush=True)
    print("-" * 110, flush=True)

    macro = {v: {"R@1": [], "R@5": [], "all_R@5": []} for v in ("dnf", "tree")}
    regressions: list[str] = []
    improvements: list[str] = []
    for bench in BENCHES:
        try:
            r1 = next(x for x in rows if x["bench"] == bench and x["variant"] == "dnf" and "error" not in x)
            r2 = next(x for x in rows if x["bench"] == bench and x["variant"] == "tree" and "error" not in x)
        except StopIteration:
            print(f"{bench:28s} (error in one or both variants)", flush=True)
            continue
        d1 = r2["R@1"] - r1["R@1"]
        d5 = r2["R@5"] - r1["R@5"]
        print(f"{bench:28s} {r1['R@1']:>9.3f} {r2['R@1']:>9.3f} {d1:>+7.3f}   "
              f"{r1['R@5']:>9.3f} {r2['R@5']:>9.3f} {d5:>+7.3f}", flush=True)
        for k in ("R@1", "R@5", "all_R@5"):
            macro["dnf"][k].append(r1[k])
            macro["tree"][k].append(r2[k])
        if d1 < -0.001 or d5 < -0.001:
            regressions.append(f"{bench}: ΔR@1={d1:+.3f} ΔR@5={d5:+.3f}")
        if d1 > 0.001 or d5 > 0.001:
            improvements.append(f"{bench}: ΔR@1={d1:+.3f} ΔR@5={d5:+.3f}")

    print("\nMACRO across benches that ran:", flush=True)
    print(f"{'variant':6s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s}", flush=True)
    for vname in ("dnf", "tree"):
        m1 = sum(macro[vname]["R@1"]) / max(1, len(macro[vname]["R@1"]))
        m5 = sum(macro[vname]["R@5"]) / max(1, len(macro[vname]["R@5"]))
        ma = sum(macro[vname]["all_R@5"]) / max(1, len(macro[vname]["all_R@5"]))
        print(f"{vname:6s} {m1:>6.3f} {m5:>6.3f} {ma:>8.3f}", flush=True)

    print(f"\n{len(improvements)} improvements, {len(regressions)} regressions:", flush=True)
    if improvements:
        print("  Improvements:", flush=True)
        for r in improvements:
            print(f"    {r}", flush=True)
    if regressions:
        print("  Regressions:", flush=True)
        for r in regressions:
            print(f"    {r}", flush=True)

    out_path = ROOT / "planner_tree_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
