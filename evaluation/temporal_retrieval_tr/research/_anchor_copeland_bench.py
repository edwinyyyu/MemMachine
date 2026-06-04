"""Anchor-closeness Copeland — generalization of directional extremum Copeland.

Current: per-doc anchor = extreme midpoint over intervals; pair winner = doc
with later (or earlier) anchor on the timeline.

Proposed: per-query anchor = single time point T (ref_time for "latest",
NEG_INF sentinel for "earliest", midpoint of bounded targets for
"around X"). Per-doc anchor = midpoint of intervals CLOSEST to T. Pair
winner = doc whose anchor is closer to T (smaller |D - T|).

This subsumes extremum: for latest_intent with T = max(corpus ref_us),
all-past-doc closeness ordering = recency ordering, same as current.
For "around X" queries it adds gradient credit for docs whose anchor
is NEAR but not inside the target interval — current pair_overlap
gives 0 to a near-miss.

Bench:
- A: current extremum Copeland (SHIP)
- B: anchor-closeness Copeland (NEW)
On the scoring-discriminating benches plus a few control benches.
"""
from __future__ import annotations

import asyncio
import gc
from dataclasses import dataclass

from temporal_retrieval_min.core import build_pool
from temporal_retrieval_min.schema import parse_iso, to_us
from temporal_retrieval_tr import Doc, TemporalExtractor, TemporalRetriever
from temporal_retrieval_tr.planner import Plan, QueryPlanner
from temporal_retrieval_tr.retriever import Result
from temporal_retrieval_tr.scoring import final_score
from temporal_retrieval_tr.time_range import is_infinite_measure
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)
import numpy as np

setup_env()


# Sentinel for "very old" / "very new" (used when earliest_intent / latest_intent
# but corpus is empty — shouldn't happen, but be safe).
NEG_SENTINEL_US = -(10**18)
POS_SENTINEL_US = 10**18


class AnchorCopelandRetriever(TemporalRetriever):
    """TemporalRetriever variant: replaces extremum Copeland with
    anchor-closeness Copeland.

    Query anchor derivation:
    - latest_intent  → max(corpus ref_us)
    - earliest_intent → min(corpus ref_us)
    - else if any bounded target → midpoint of average bounded interval midpoint
    - else → no anchor (fall through to existing non-extremum logic)
    """

    def _derive_query_anchor(self, plan: Plan) -> int | None:
        if plan.latest_intent:
            if self._doc_ref_us:
                return max(self._doc_ref_us.values())
            return POS_SENTINEL_US
        if plan.earliest_intent:
            if self._doc_ref_us:
                return min(self._doc_ref_us.values())
            return NEG_SENTINEL_US
        # Bounded targets: average midpoint of bounded intervals.
        bounded_mids: list[int] = []
        for t in plan.targets:
            for iv in t.intervals:
                if not iv.left_unbounded and not iv.right_unbounded:
                    bounded_mids.append((iv.earliest_us + iv.latest_us) // 2)
        if bounded_mids:
            return sum(bounded_mids) // len(bounded_mids)
        return None

    async def query(self, query: str, ref_time: str, k: int = 10) -> list[Result]:
        plan: Plan = await self._planner.plan(query, ref_time)
        query_targets = plan.targets

        bounded_target_present = any(
            not is_infinite_measure(t) for t in query_targets
        )

        q_emb = (await self.embed_fn([query]))[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sem_scores = self._cosine_all(q_emb)
        all_dids = list(self._doc_ref_us.keys())

        match_all: dict[str, float] = {}
        eligible: list[str] = []
        timeless_in_scope: set[str] = set()
        for did in all_dids:
            d_anchors = self._doc_anchors.get(did, [])
            if not d_anchors:
                if not query_targets or not bounded_target_present:
                    match_all[did] = 1.0
                    eligible.append(did)
                else:
                    if isinstance(self.timeless_match_in_scope, str):
                        timeless_in_scope.add(did)
                    else:
                        match_all[did] = self.timeless_match_in_scope
            else:
                s = final_score(query_targets, d_anchors)
                match_all[did] = s
                if s > 0.0:
                    eligible.append(did)

        pool = build_pool(sem_scores, all_dids, eligible, self.pool_size)
        if not pool:
            return []

        pool_texts = [self._docs[did].text for did in pool]
        rerank_scores = await self.rerank_fn(query, pool_texts)
        base = dict(zip(pool, rerank_scores, strict=False))

        if isinstance(self.timeless_match_in_scope, str) and \
                self.timeless_match_in_scope == "base":
            for did in pool:
                if did in timeless_in_scope:
                    match_all[did] = base.get(did, 0.0)

        if base:
            base_vals = list(base.values())
            pool_spread = max(base_vals) - min(base_vals)
        else:
            pool_spread = 1.0
        match_eff = {did: match_all.get(did, 0.0) * pool_spread for did in pool}

        # ANCHOR-COPELAND DISPATCH
        query_anchor_us = self._derive_query_anchor(plan)
        if query_anchor_us is not None and self.extremum_copeland:
            return self._copeland_anchor_rerank(
                pool, base, match_eff, query_anchor_us, k,
            )

        # Fall-through: copeland_pairwise (timed/timeless heterogeneity)
        anchored = {did for did in pool if self._doc_anchors.get(did)}
        return self._copeland_pairwise_rerank(
            pool, base, match_eff, anchored, k
        )

    def _copeland_anchor_rerank(
        self,
        pool: list[str],
        base: dict[str, float],
        match_eff: dict[str, float],
        query_anchor_us: int,
        k: int,
    ) -> list[Result]:
        """Pairwise: closer-to-query-anchor wins +bonus.

        Per-doc anchor = midpoint of doc intervals CLOSEST to query_anchor_us;
        ref_us when no intervals.
        """
        bonus = float(self.copeland_bonus or 0.0)
        if base:
            vals = list(base.values())
            bonus = bonus * (max(vals) - min(vals))

        # Per-doc anchor: pick midpoint closest to query_anchor_us
        doc_anchor: dict[str, int] = {}
        for did in pool:
            ivs = self._doc_ivs.get(did, [])
            if ivs:
                mids = [(iv.earliest_us + iv.latest_us) // 2 for iv in ivs]
                doc_anchor[did] = min(mids, key=lambda m: abs(m - query_anchor_us))
            else:
                doc_anchor[did] = self._doc_ref_us[did]

        sim = {did: base.get(did, 0.0) + match_eff.get(did, 0.0) for did in pool}
        wins: dict[str, int] = dict.fromkeys(pool, 0)
        margins: dict[str, float] = dict.fromkeys(pool, 0.0)

        def closeness(d: str) -> int:
            # negative distance: larger (less negative) = closer
            return -abs(doc_anchor[d] - query_anchor_us)

        for a in pool:
            ca = closeness(a)
            for b in pool:
                if a == b:
                    continue
                cb = closeness(b)
                sa = sim[a] + (bonus if ca > cb else 0.0)
                sb = sim[b] + (bonus if cb > ca else 0.0)
                if sa > sb:
                    wins[a] += 1
                    margins[a] += sa - sb

        tertiary = sim if self.copeland_tiebreak == "sim" else base
        ranked = sorted(
            pool,
            key=lambda d: (-wins[d], -margins[d], -tertiary.get(d, 0.0)),
        )
        results: list[Result] = []
        for did in ranked:
            score = float(wins[did]) + margins[did] * 1e-4
            results.append(
                Result(
                    doc_id=did,
                    score=score,
                    rerank=base.get(did, 0.0),
                    match=match_eff.get(did, 0.0),
                    recency=0.0,
                )
            )
        return results[:k]


# Discriminating benches (per project_recency_bench_modality.md memory):
# the only ones that move with scoring changes.
DISCRIMINATING = [
    "composition", "cotemporal", "same_topic_recency", "same_topic_recency_hard",
    "latest_recent", "goldilocks", "goldilocks_v2", "v7_doc_directional",
    "recency_stress_deep", "recency_vs_rerank",
    # Add a few date-explicit benches to test the "around X" generalization:
    "edge_relative_time", "edge_era_refs", "polarity",
]


async def run_bench(bench, RetrCls, extractor, planner, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = RetrCls(
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


async def arm(label, RetrCls, extractor, planner, embed_fn, rerank_fn, benches):
    print(f"\n=== {label} ===", flush=True)
    rows = []
    for bench in benches:
        try:
            m = await run_bench(bench, RetrCls, extractor, planner,
                                embed_fn, rerank_fn)
        except Exception as e:
            print(f"  ERROR {bench}: {e}", flush=True)
            continue
        if m is None:
            continue
        rows.append((bench, m))
        print(f"  {bench:30s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  n={m['n']}",
              flush=True)
    return rows


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    extractor = TemporalExtractor()
    planner = QueryPlanner()

    # Run on the FULL bench list to detect regressions on non-discriminating
    # benches (where the anchor formulation newly fires for any bounded target).
    benches = BENCH_NAMES

    rows_A = await arm("A: extremum Copeland (SHIP)",
                       TemporalRetriever, extractor, planner,
                       embed_fn, rerank_fn, benches)
    rows_B = await arm("B: anchor-closeness Copeland (NEW)",
                       AnchorCopelandRetriever, extractor, planner,
                       embed_fn, rerank_fn, benches)

    # Aligned summary
    print("\n=== Per-bench Δ (B - A) ===", flush=True)
    a_map = dict(rows_A)
    b_map = dict(rows_B)
    benches_seen = [b for b, _ in rows_A]
    d1_total = 0.0
    d5_total = 0.0
    n_used = 0
    for b in benches_seen:
        if b not in b_map:
            continue
        a, bn = a_map[b], b_map[b]
        d1 = bn["R@1"] - a["R@1"]
        d5 = bn["R@5"] - a["R@5"]
        print(f"  {b:30s}  ΔR@1={d1:+.3f}  ΔR@5={d5:+.3f}  "
              f"A=({a['R@1']:.3f}/{a['R@5']:.3f})  "
              f"B=({bn['R@1']:.3f}/{bn['R@5']:.3f})", flush=True)
        d1_total += d1
        d5_total += d5
        n_used += 1
    print(f"\nMACRO Δ over {n_used} benches:  "
          f"ΔR@1={d1_total/n_used:+.4f}  ΔR@5={d5_total/n_used:+.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
