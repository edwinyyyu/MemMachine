"""Noisy-OR vs mean A/B on focus benches.

Sweeps alpha ∈ {0.5, 0.7, 0.85, 1.0} against the current mean baseline.
Uses the existing V7-Direct pipeline; only the cross-clause aggregator
differs (final_score_noisy_or with the given alpha vs final_score).

Within-clause AND-incompat aggregator stays as mean (the existing
clause_score behavior) so the experiment isolates the OUTER aggregator.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._noisy_or_ab
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from temporal_retrieval_min import Doc as DocV1
from temporal_retrieval_min import TemporalRetriever
from temporal_retrieval_min.core import (
    build_pool,
    normalize_dict,
    normalize_rerank_full,
    recency_scores,
)
from temporal_retrieval_min.schema import parse_iso, to_us
from temporal_retrieval_v7 import (
    Doc as DocV7,
    TemporalRetrieverV7Direct,
)
from temporal_retrieval_v7.scoring import (
    Clause,
    final_score_noisy_or,
)
from temporal_retrieval_v7.time_range import is_infinite_measure

from temporal_retrieval.research._common import (
    make_embed_fn,
    setup_env,
)
from temporal_retrieval_v7.research._full_ab import (
    _load_bench,
    _metrics,
    make_cosine_rerank_fn,
)

setup_env()


class V7DirectNoisyOR(TemporalRetrieverV7Direct):
    """V7-Direct with noisy-OR outer aggregator."""

    def __init__(self, *args, alpha: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = alpha

    async def query(self, query: str, ref_time: str, k: int = 10):
        from temporal_retrieval_v7.retriever import Result
        from temporal_retrieval_v7.planner_direct import DirectPlan

        direct_plan: DirectPlan = await self._planner.plan(query, ref_time)
        clauses: list[Clause] = direct_plan.clauses
        bounded_clause_present = any(
            any(not is_infinite_measure(r) for r in c.refs)
            for c in clauses
        )

        q_emb = (await self.embed_fn([query]))[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sem_scores = self._cosine_all(q_emb)
        all_dids = list(self._doc_ref_us.keys())

        match_all = {}
        eligible = []
        for did in all_dids:
            d_refs = self._doc_refs.get(did, [])
            if not d_refs:
                match_all[did] = 1.0
                if not clauses or not bounded_clause_present:
                    eligible.append(did)
            else:
                s = final_score_noisy_or(clauses, d_refs, alpha=self._alpha)
                match_all[did] = s
                if s > 0.0:
                    eligible.append(did)

        pool = build_pool(sem_scores, all_dids, eligible, self.pool_size)
        if not pool:
            return []
        pool_texts = [self._docs[did].text for did in pool]
        rerank_scores = await self.rerank_fn(query, pool_texts)
        rerank_pool = dict(zip(pool, rerank_scores, strict=False))
        r_full = normalize_rerank_full(rerank_pool, all_dids, 0.0)
        rec = self._compute_recency_direct(direct_plan, pool, match_all)
        r_pool = {did: r_full.get(did, 0.0) for did in pool}
        base = normalize_dict(r_pool)
        results = []
        for did in pool:
            b = base.get(did, 0.0) + match_all[did]
            if direct_plan.latest_intent or direct_plan.earliest_intent:
                b += rec.get(did, 0.0)
            if b > 0:
                results.append(Result(doc_id=did, score=b,
                                      rerank=base.get(did, 0.0),
                                      match=match_all[did],
                                      recency=rec.get(did, 0.0)))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]


# Focused set including benches with diverse partial-vs-full match dynamics
BENCHES = [
    "engagement_disjoint",       # gold incl partial; favors noisy-OR
    "edge_conjunctive_temporal", # gold = both-period; favors mean
    "polarity",
    "sensitivity_curated",
    "hard_bench",
    "composition",
    "axis",
    "v7_compound_hard",
    "v7_doc_directional",
]
ALPHAS = [0.5, 0.7, 0.85, 1.0]


async def run_bench(bench, embed_fn, rerank_fn) -> dict | None:
    docs_jsonl, queries, gold = _load_bench(bench)
    if docs_jsonl is None:
        return None
    docs_v1 = [DocV1(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
               for d in docs_jsonl]
    docs_v7 = [DocV7(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
               for d in docs_jsonl]

    v1 = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    base = TemporalRetrieverV7Direct(embed_fn=embed_fn, rerank_fn=rerank_fn)
    variants = {
        a: V7DirectNoisyOR(embed_fn=embed_fn, rerank_fn=rerank_fn, alpha=a)
        for a in ALPHAS
    }
    await v1.index(docs_v1)
    await base.index(docs_v7)
    for r in variants.values():
        await r.index(docs_v7)

    rankings_v1 = {}
    rankings_base = {}
    rankings = {a: {} for a in ALPHAS}
    for q in queries:
        r1 = await v1.query(q["text"], q["ref_time"], k=10)
        rb = await base.query(q["text"], q["ref_time"], k=10)
        rankings_v1[q["query_id"]] = [r.doc_id for r in r1]
        rankings_base[q["query_id"]] = [r.doc_id for r in rb]
        for a, r in variants.items():
            res = await r.query(q["text"], q["ref_time"], k=10)
            rankings[a][q["query_id"]] = [r.doc_id for r in res]

    return {
        "v1": _metrics(rankings_v1, gold),
        "mean": _metrics(rankings_base, gold),
        **{f"nor_a{a}": _metrics(rankings[a], gold) for a in ALPHAS},
    }


async def main():
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== Noisy-OR (alpha sweep) vs Mean — {len(BENCHES)} benches ===\n",
          flush=True)
    print(f"{'bench':28s}  {'V1':>6s} {'mean':>6s}  "
          f"{'α=0.5':>6s} {'α=0.7':>6s} {'α=0.85':>7s} {'α=1.0':>6s}",
          flush=True)
    rows = {}
    for bench in BENCHES:
        try:
            res = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:28s}  ERROR: {e}", flush=True)
            continue
        if res is None:
            continue
        rows[bench] = res
        print(
            f"  {bench:26s}  "
            f"{res['v1']['R@1']:>6.3f} {res['mean']['R@1']:>6.3f}  "
            f"{res['nor_a0.5']['R@1']:>6.3f} {res['nor_a0.7']['R@1']:>6.3f} "
            f"{res['nor_a0.85']['R@1']:>7.3f} {res['nor_a1.0']['R@1']:>6.3f}",
            flush=True,
        )
    # Macro
    n = len(rows)
    if n:
        macros = {k: sum(r[k]["R@1"] for r in rows.values()) / n
                  for k in ("v1", "mean", *(f"nor_a{a}" for a in ALPHAS))}
        print("-" * 80)
        print(
            f"  {'MACRO':26s}  {macros['v1']:>6.3f} {macros['mean']:>6.3f}  "
            f"{macros['nor_a0.5']:>6.3f} {macros['nor_a0.7']:>6.3f} "
            f"{macros['nor_a0.85']:>7.3f} {macros['nor_a1.0']:>6.3f}",
            flush=True,
        )
    out = Path(__file__).resolve().parent.parent / "ab_noisy_or.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
