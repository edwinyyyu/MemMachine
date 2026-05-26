"""Full 35-bench A/B: V1 strict (production) vs V7 TimeRange.

Uses the cosine reranker (fast) — matches _validate_disjoint_variants
methodology. The expensive LLM calls (extractor, planner) are shared
via the on-disk cache.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._full_ab
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from temporal_retrieval_min import Doc as DocV1
from temporal_retrieval_min import TemporalRetriever
from temporal_retrieval_v7 import Doc as DocV7
from temporal_retrieval_v7 import TemporalRetrieverV7

from temporal_retrieval.research._common import (
    DATA_DIR,
    ROOT as TR_ROOT,
    make_embed_fn,
    setup_env,
)

setup_env()

# The same 35-bench list used by _validate_disjoint_variants.
# engagement_disjoint added as the 36th (new test, V7 target).
BENCH_NAMES = [
    "adversarial", "allen", "ambiguous_year", "ambiguous_year_adv",
    "axis", "causal_relative", "composition", "cotemporal",
    "dense_cluster", "disc", "edge_conjunctive_temporal", "edge_era_refs",
    "edge_multi_te_doc", "edge_relative_time", "engagement_disjoint",
    "era", "goldilocks", "goldilocks_v2", "hard_bench", "hard_dense_cluster",
    "latest_recent", "lattice", "mixed_cue", "negation_temporal",
    "notin_multi_interval", "open_ended_date", "polarity", "precedents",
    "realq", "realq_deictic", "realq_v2", "sensitivity_curated",
    "speculative_anchors", "temporal_essential", "timeless_policies",
    "utterance",
    # V7-targeted benches (added for direct planner validation)
    "v7_compound_hard", "v7_doc_directional",
]


def _load_bench(bench: str):
    try:
        with open(DATA_DIR / f"{bench}_docs.jsonl") as f:
            docs = [json.loads(line) for line in f]
        with open(DATA_DIR / f"{bench}_queries.jsonl") as f:
            queries = [json.loads(line) for line in f]
        with open(DATA_DIR / f"{bench}_gold.jsonl") as f:
            gold_rows = [json.loads(line) for line in f]
    except FileNotFoundError as e:
        return None, None, None
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    return docs, queries, gold


def make_cosine_rerank_fn(embed_fn):
    """Cosine-similarity reranker — fast, no cross-encoder."""
    async def rerank(query: str, doc_texts: list[str]) -> list[float]:
        if not doc_texts:
            return []
        qe = (await embed_fn([query]))[0]
        des = await embed_fn(doc_texts)
        qn = float(np.linalg.norm(qe)) or 1e-9
        out = []
        for de in des:
            dn = float(np.linalg.norm(de)) or 1e-9
            out.append(float(np.dot(qe, de) / (qn * dn)))
        return out
    return rerank


def _metrics(rankings: dict, gold: dict, k_r5: int = 5) -> dict:
    n_r1 = n_r5 = n_eval = 0
    all_r5 = []
    for qid, ranking in rankings.items():
        gs = gold.get(qid, set())
        if not gs:
            continue
        n_eval += 1
        first = next((i + 1 for i, d in enumerate(ranking) if d in gs), None)
        if first is not None:
            if first <= 1:
                n_r1 += 1
            if first <= k_r5:
                n_r5 += 1
        top = set(ranking[:k_r5])
        all_r5.append(len(top & gs) / len(gs))
    return {
        "n": n_eval,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "all_R@5": sum(all_r5) / max(1, len(all_r5)),
    }


async def run_one_bench(
    bench: str, embed_fn, rerank_fn,
) -> tuple[dict, dict] | None:
    docs_jsonl, queries, gold = _load_bench(bench)
    if docs_jsonl is None:
        return None

    v1_docs = [DocV1(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
               for d in docs_jsonl]
    v7_docs = [DocV7(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
               for d in docs_jsonl]

    v1 = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    v7 = TemporalRetrieverV7(embed_fn=embed_fn, rerank_fn=rerank_fn)

    await v1.index(v1_docs)
    await v7.index(v7_docs)

    v1_rankings = {}
    v7_rankings = {}
    for q in queries:
        r1 = await v1.query(q["text"], q["ref_time"], k=10)
        r7 = await v7.query(q["text"], q["ref_time"], k=10)
        v1_rankings[q["query_id"]] = [r.doc_id for r in r1]
        v7_rankings[q["query_id"]] = [r.doc_id for r in r7]

    return _metrics(v1_rankings, gold), _metrics(v7_rankings, gold)


async def main():
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    print(f"=== V1 vs V7 over {len(BENCH_NAMES)} benches (cosine rerank) ===\n")
    print(f"{'bench':28s}  {'V1 R@1':>8s} {'V7 R@1':>8s} {'ΔR@1':>8s}  "
          f"{'V1 R@5':>8s} {'V7 R@5':>8s} {'ΔR@5':>8s}  {'n':>4s}",
          flush=True)
    print("-" * 105)

    all_results = {}
    for bench in BENCH_NAMES:
        try:
            res = await run_one_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:28s}  ERROR: {e}", flush=True)
            continue
        if res is None:
            print(f"{bench:28s}  SKIPPED (no data)", flush=True)
            continue
        v1m, v7m = res
        all_results[bench] = {"v1": v1m, "v7": v7m}
        d1 = v7m["R@1"] - v1m["R@1"]
        d5 = v7m["R@5"] - v1m["R@5"]
        marker = "*" if d1 < -0.05 else "+" if d1 > 0.05 else " "
        print(
            f"{marker} {bench:26s}  {v1m['R@1']:>8.3f} {v7m['R@1']:>8.3f} "
            f"{d1:>+8.3f}  {v1m['R@5']:>8.3f} {v7m['R@5']:>8.3f} "
            f"{d5:>+8.3f}  {v1m['n']:>4d}",
            flush=True,
        )

    # Macro
    n = len(all_results)
    if n:
        m1 = sum(r["v1"]["R@1"] for r in all_results.values()) / n
        m7 = sum(r["v7"]["R@1"] for r in all_results.values()) / n
        m1_5 = sum(r["v1"]["R@5"] for r in all_results.values()) / n
        m7_5 = sum(r["v7"]["R@5"] for r in all_results.values()) / n
        m1_a = sum(r["v1"]["all_R@5"] for r in all_results.values()) / n
        m7_a = sum(r["v7"]["all_R@5"] for r in all_results.values()) / n
        print("-" * 105)
        print(f"  {'MACRO':26s}  {m1:>8.3f} {m7:>8.3f} {m7-m1:>+8.3f}  "
              f"{m1_5:>8.3f} {m7_5:>8.3f} {m7_5-m1_5:>+8.3f}  n_benches={n}")
        print(f"  {'(all_R@5)':26s}                    "
              f"                          {m1_a:>8.3f} {m7_a:>8.3f} "
              f"{m7_a-m1_a:>+8.3f}")

    out_path = Path(__file__).resolve().parent.parent / "ab_v1_vs_v7.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
