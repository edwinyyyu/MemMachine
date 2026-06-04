"""Proximity-anchor pipeline — full 47-bench evaluation.

Single-arm run of the refactored pipeline:
- Planner now emits `proximity_anchor` (latest / earliest / ISO date / null)
  instead of `extremum` (latest / earliest / null).
- Retriever now dispatches on `plan.proximity_anchor_us` and runs
  `_copeland_proximity_rerank` which combines heterogeneous-match
  (timed/timeless asymmetric base+match treatment) with a proximity
  bonus to whichever doc's anchor is closer to the query anchor.

Compared to the baseline below (saved from the prior extremum-Copeland
ship arm), this run shows:
- The refactor's effect (heterogeneous match preserved, proximity
  generalized to finite anchors).
- Whether the planner uses finite anchors for "around X" queries on
  benches like metadata_only, cotemporal, goldilocks_v2.

The macro Δ is computed inline against the hardcoded baseline below.
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_tr import Doc, TemporalExtractor, TemporalRetriever
from temporal_retrieval_tr.planner import QueryPlanner
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()


# Baseline: saved arm A "extremum Copeland (SHIP)" from prior bench run,
# /tmp/claude/anchor_copeland_full.log lines 3-49. Same 47 benches.
BASELINE_R1: dict[str, float] = {
    "adversarial": 0.800, "allen": 0.350, "ambiguous_year": 0.917,
    "ambiguous_year_adv": 0.833, "axis": 0.950, "causal_relative": 0.200,
    "composition": 0.400, "cotemporal": 0.950, "dense_cluster": 0.967,
    "disc": 0.667, "edge_conjunctive_temporal": 0.833, "edge_era_refs": 0.167,
    "edge_multi_te_doc": 1.000, "edge_relative_time": 0.917,
    "engagement_disjoint": 0.800, "era": 0.950, "goldilocks": 0.867,
    "goldilocks_v2": 0.600, "hard_bench": 0.960, "hard_dense_cluster": 1.000,
    "latest_recent": 1.000, "lattice": 1.000, "mixed_cue": 0.975,
    "negation_temporal": 0.800, "notin_multi_interval": 0.250,
    "open_ended_date": 0.800, "polarity": 0.933, "precedents": 1.000,
    "realq": 0.692, "realq_deictic": 1.000, "realq_v2": 0.853,
    "sensitivity_curated": 0.636, "speculative_anchors": 0.500,
    "temporal_essential": 1.000, "timeless_policies": 1.000,
    "utterance": 0.800, "v7_compound_hard": 0.778, "v7_doc_directional": 0.750,
    "same_topic_recency": 0.967, "same_topic_recency_hard": 1.000,
    "recency_stress_deep": 1.000, "recency_vs_rerank": 0.400,
    "state_vs_event": 0.900, "state_vs_event_v2": 0.960,
    "comparative_recency": 0.917, "metadata_only": 0.000,
    "deictic_in_content": 0.714,
}
BASELINE_R5: dict[str, float] = {
    "adversarial": 0.914, "allen": 1.000, "ambiguous_year": 1.000,
    "ambiguous_year_adv": 0.917, "axis": 1.000, "causal_relative": 1.000,
    "composition": 0.720, "cotemporal": 1.000, "dense_cluster": 1.000,
    "disc": 0.767, "edge_conjunctive_temporal": 1.000, "edge_era_refs": 1.000,
    "edge_multi_te_doc": 1.000, "edge_relative_time": 1.000,
    "engagement_disjoint": 0.900, "era": 1.000, "goldilocks": 0.933,
    "goldilocks_v2": 1.000, "hard_bench": 1.000, "hard_dense_cluster": 1.000,
    "latest_recent": 1.000, "lattice": 1.000, "mixed_cue": 1.000,
    "negation_temporal": 0.933, "notin_multi_interval": 1.000,
    "open_ended_date": 0.867, "polarity": 1.000, "precedents": 1.000,
    "realq": 1.000, "realq_deictic": 1.000, "realq_v2": 1.000,
    "sensitivity_curated": 0.818, "speculative_anchors": 1.000,
    "temporal_essential": 1.000, "timeless_policies": 1.000,
    "utterance": 0.900, "v7_compound_hard": 1.000, "v7_doc_directional": 1.000,
    "same_topic_recency": 1.000, "same_topic_recency_hard": 1.000,
    "recency_stress_deep": 1.000, "recency_vs_rerank": 0.500,
    "state_vs_event": 1.000, "state_vs_event_v2": 1.000,
    "comparative_recency": 1.000, "metadata_only": 0.214,
    "deictic_in_content": 0.929,
}


async def run_bench(bench, extractor, planner, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        extractor=extractor, planner=planner,
        ranking_method="copeland_pairwise",
    )
    await vd.index(docs)
    rk = {
        q["query_id"]: [
            x.doc_id for x in await vd.query(q["text"], q["ref_time"], k=10)
        ]
        for q in queries
    }
    m = metrics(rk, gold)
    del vd, docs
    gc.collect()
    return m


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    extractor = TemporalExtractor()
    planner = QueryPlanner()

    print("=== Proximity-anchor pipeline (NEW), per-bench vs baseline ===",
          flush=True)
    print(f"{'bench':30s}  {'R@1':>6s} {'Δ':>8s}  {'R@5':>6s} {'Δ':>8s}",
          flush=True)
    print("-" * 70, flush=True)

    rows: list[tuple[str, dict]] = []
    d1_total = 0.0
    d5_total = 0.0
    n_used = 0
    for bench in BENCH_NAMES:
        try:
            m = await run_bench(bench, extractor, planner, embed_fn, rerank_fn)
        except Exception as e:
            print(f"  ERROR {bench}: {e}", flush=True)
            continue
        if m is None:
            continue
        rows.append((bench, m))
        b1 = BASELINE_R1.get(bench)
        b5 = BASELINE_R5.get(bench)
        if b1 is None or b5 is None:
            print(f"  {bench:30s}  {m['R@1']:.3f}        {m['R@5']:.3f}",
                  flush=True)
            continue
        d1 = m['R@1'] - b1
        d5 = m['R@5'] - b5
        print(
            f"  {bench:30s}  {m['R@1']:.3f} ({d1:+.3f})  "
            f"{m['R@5']:.3f} ({d5:+.3f})",
            flush=True,
        )
        d1_total += d1
        d5_total += d5
        n_used += 1

    if n_used > 0:
        print()
        print(f"MACRO Δ over {n_used} benches:  "
              f"ΔR@1={d1_total/n_used:+.4f}  ΔR@5={d5_total/n_used:+.4f}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
