"""Temporal retriever: IntervalSet-based temporal layer.

Symmetric query and doc sides:
- Query side: `QueryPlanner` emits a list of IntervalSet targets.
- Doc side: `TemporalExtractor` emits a list of IntervalSet anchors.

Each side's IntervalSets can be multi-interval when the underlying
claim has internal structure (gaps, complements, disjunctions).

Scoring is `final_score(query_targets, doc_anchors)` — mean over
query targets of the best per-anchor `pair_overlap` (frac-min on
interval-set intersection).
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from temporal_retrieval_min.core import (
    Interval as V1Interval,
)
from temporal_retrieval_min.core import (
    build_pool,
)
from temporal_retrieval_min.schema import parse_iso, to_us

from .extractor import TemporalExtractor
from .planner import Plan, QueryPlanner
from .scoring import final_score
from .time_range import Interval, IntervalSet, is_infinite_measure


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class Doc:
    id: str
    text: str
    ref_time: str  # ISO 8601 UTC


@dataclass
class Result:
    doc_id: str
    score: float
    rerank: float
    match: float
    recency: float


EmbedFn = Callable[[list[str]], Awaitable[list[np.ndarray]]]
RerankFn = Callable[[str, list[str]], Awaitable[list[float]]]


# ---------------------------------------------------------------------------
# Doc-side adapter
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class TemporalRetriever:
    """Temporal retriever — IntervalSet semantics on both query and doc sides.

    Each side issues one LLM call per item:
    - Query side: planner emits a list of IntervalSet targets, where
      each target can be multi-interval (e.g. complement, disjoint
      union).
    - Doc side: extractor emits a list of IntervalSet anchors, where
      each anchor can also be multi-interval when the doc describes
      a single logical claim with internal structure (gaps,
      complements, disjunctions).

    The extractor contract is `extract_anchors(text, ref_time) ->
    list[IntervalSet]`. The retriever does not adapt other interfaces.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        rerank_fn: RerankFn,
        cache_dir: str | Path = "cache/temporal_retrieval_tr",
        pool_size: int = 40,
        planner: QueryPlanner | None = None,
        extractor: TemporalExtractor | None = None,
        copeland_bonus: float = 0.40,
        copeland_tiebreak: str = "sim",
        timeless_match_in_scope: float | str = 0.8,
        ranking_method: str = "copeland_pairwise",
        proximity_copeland: bool = True,
    ) -> None:
        # pool_size=40 matches production overfetch=4× (final K=10).
        # Scoring (recency etc.) then has real effect on R@K because it
        # selects which of the 40 candidates survive into the final top-K.
        self.embed_fn = embed_fn
        self.rerank_fn = rerank_fn
        self.pool_size = pool_size
        # Copeland per-pair bonus for the more-recent doc in extremum queries.
        # Default 0.40 in raw cosine units; scaled per-query by pool spread.
        self.copeland_bonus = copeland_bonus
        # Copeland tertiary tiebreak ("sim" = base+match; "base" = rerank-only).
        self.copeland_tiebreak = copeland_tiebreak
        # Per-doc proximity anchor for the closeness tournament is derived
        # automatically from `plan.proximity_anchor_us`:
        # - For finite query anchor T: pick the doc-interval midpoint
        #   CLOSEST to T (min by |mid − T|), fallback to ref_us.
        # - For POS_INF (latest): pick max midpoint, fallback ref_us.
        # - For NEG_INF (earliest): pick min midpoint, fallback ref_us.
        # See `_copeland_proximity_rerank`.

        # Match score for timeless docs (no extracted anchors) when the
        # query has bounded scope. Accepts:
        #   float: fixed credit (0.0 = strict, 1.0 = vacuous match,
        #          0.8 = empirical macro optimum on 44-bench sweep).
        #   "base": adaptive — use the doc's own base (rerank) score
        #          as match. High-semantic timeless docs get more
        #          temporal credit; low-semantic noise gets less.
        # When the query has NO bounded scope, timeless docs always
        # get 1.0 (uniform with anchored docs that satisfy the
        # unbounded targets vacuously).
        self.timeless_match_in_scope = timeless_match_in_scope
        # Ranking method for the pool:
        # - "additive": score = base + match_eff; sort by score.
        # - "copeland_pairwise": Copeland tournament where timed-vs-
        #   timed pairs use base + match_eff; pairs involving a
        #   timeless doc use base only. Avoids assigning timeless
        #   docs an artificial temporal match value.
        self.ranking_method = ranking_method
        # When False, the proximity Copeland tournament is bypassed even
        # when plan.proximity_anchor_us is set — queries fall through to
        # the configured ranking_method (additive or copeland_pairwise)
        # instead. Used for ablations isolating the non-proximity scoring
        # policy.
        self.proximity_copeland = proximity_copeland
        # Scoring substrate (shipped — see _SCORING_ARCHITECTURE.md for rationale):
        # - base = raw cosine (kept in native units, query-independent)
        # - match_eff = match * pool_cosine_spread (scales system-specific score)
        # - bonus_eff = copeland_bonus * pool_cosine_spread (same)
        # This preserves cosine's universal interpretability while keeping the
        # base-vs-match-vs-recency balance stable across pool compositions.
        self._cache_dir = Path(cache_dir)

        self._planner = planner or QueryPlanner()
        self._extractor = extractor or TemporalExtractor()

        # Indexed state
        self._docs: dict[str, Doc] = {}
        self._doc_ivs: dict[str, list[V1Interval]] = {}
        self._doc_anchors: dict[str, list[IntervalSet]] = {}
        self._doc_emb: dict[str, np.ndarray] = {}
        self._doc_ref_us: dict[str, int] = {}

    # ----------------------------------------------------------------- Index
    async def index(self, docs: list[Doc]) -> None:
        self._docs = {d.id: d for d in docs}
        self._doc_ref_us = {d.id: to_us(parse_iso(d.ref_time)) for d in docs}

        async def _extract_one(
            d: Doc,
        ) -> tuple[str, list[V1Interval], list[IntervalSet]]:
            try:
                anchors = await self._extractor.extract_anchors(
                    d.text, parse_iso(d.ref_time)
                )
                # Flatten for recency-anchor computation in Copeland rerank.
                # Each interval in each anchor is treated as a candidate
                # midpoint for the extreme/median/ref_time selection.
                # Skip unbounded intervals (e.g. 'since 2020', 'before X')
                # — their midpoint is ±∞ and not meaningful for recency
                # selection. They still contribute to overlap scoring via
                # the anchor IntervalSet.
                ivs = [
                    V1Interval(iv.earliest_us, iv.latest_us)
                    for a in anchors
                    for iv in a.intervals
                    if not iv.left_unbounded and not iv.right_unbounded
                ]
            except Exception:
                ivs, anchors = [], []
            return d.id, ivs, anchors

        results = await asyncio.gather(*(_extract_one(d) for d in docs))
        for did, ivs, anchors in results:
            self._doc_ivs[did] = ivs
            self._doc_anchors[did] = anchors
        self._extractor.save_caches()

        embs = await self.embed_fn([d.text for d in docs])
        for d, e in zip(docs, embs, strict=False):
            self._doc_emb[d.id] = np.asarray(e, dtype=np.float32)

    # ----------------------------------------------------------------- Query
    async def query(self, query: str, ref_time: str, k: int = 10) -> list[Result]:
        plan: Plan = await self._planner.plan(query, ref_time)
        query_targets: list[IntervalSet] = plan.targets

        # Bounded vs unbounded targets determine timeless-doc filter
        # admission. A target with FINITE measure (points to a specific
        # bounded calendar window) signals "this query is asking about
        # a specific time" — timeless docs don't carry the temporal
        # evidence to satisfy it and should not filter-pass (they only
        # ride into the pool via the semantic top-up). All-unbounded
        # targets (pure disjoint complements, or open-ended directional
        # ranges) allow timeless to pass — the query is more about
        # excluding/orienting than naming a specific time.
        bounded_target_present = any(
            not is_infinite_measure(t) for t in query_targets
        )

        # Semantic
        q_emb = (await self.embed_fn([query]))[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sem_scores = self._cosine_all(q_emb)

        all_dids = list(self._doc_ref_us.keys())

        # Two-track filter / score:
        # - filter (pool admission): doc must have anchors AND score>0,
        #   OR be timeless when no target is bounded.
        # - rank score: timeless docs still pass-through to 1.0 so
        #   semantic + rerank decide ordering for them within the pool.
        match_all: dict[str, float] = {}
        eligible: list[str] = []
        timeless_in_scope: set[str] = set()  # for deferred match if "base" mode
        for did in all_dids:
            d_anchors = self._doc_anchors.get(did, [])
            if not d_anchors:
                if not query_targets or not bounded_target_present:
                    # Vacuous match: query has no bounded scope, timeless
                    # doc trivially satisfies. Eligible.
                    match_all[did] = 1.0
                    eligible.append(did)
                else:
                    # Bounded scope present and doc has no temporal
                    # evidence. Configurable rank credit; not filter-
                    # admitted (may still enter via semantic top-up).
                    if isinstance(self.timeless_match_in_scope, str):
                        timeless_in_scope.add(did)  # defer until rerank
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

        # Cross-encoder rerank
        pool_texts = [self._docs[did].text for did in pool]
        rerank_scores = await self.rerank_fn(query, pool_texts)
        rerank_pool = dict(zip(pool, rerank_scores, strict=False))

        # Base = raw cosine (kept in native units, cross-query stable).
        base = dict(rerank_pool)

        # Adaptive timeless match: if "base" mode, set match_all for
        # timeless-in-scope docs to their own base score (higher
        # semantic = more temporal credit).
        if isinstance(self.timeless_match_in_scope, str) and \
                self.timeless_match_in_scope == "base":
            for did in pool:
                if did in timeless_in_scope:
                    match_all[did] = base.get(did, 0.0)

        # Pool cosine spread is used to scale the system-specific scores
        # (match, copeland_bonus) so they track the pool's natural scale.
        # This preserves the base-vs-match balance across pools without
        # distorting cosine itself.
        if base:
            base_vals = list(base.values())
            pool_spread = max(base_vals) - min(base_vals)
        else:
            pool_spread = 1.0
        match_eff = {did: match_all.get(did, 0.0) * pool_spread for did in pool}

        # Proximity-anchor queries → Copeland tournament re-rank with per-pair
        # bonus to whichever doc's anchor is closer to plan.proximity_anchor_us.
        # No proximity anchor → just base + match (no closeness layer).
        # base_only_strict skips this too — true no-temporal-scoring floor.
        # proximity_copeland=False disables dispatch for ablation.
        if plan.proximity_anchor_us is not None \
                and self.ranking_method != "base_only_strict" \
                and self.proximity_copeland:
            return self._copeland_proximity_rerank(
                pool, base, match_eff, plan.proximity_anchor_us, k,
            )

        if self.ranking_method == "copeland_pairwise":
            anchored = {did for did in pool if self._doc_anchors.get(did)}
            return self._copeland_pairwise_rerank(
                pool, base, match_eff, anchored, k
            )

        if self.ranking_method in ("base_only", "base_only_strict"):
            # Rank by base (rerank/semantic) alone — no temporal layer.
            # base_only:        skips additive/copeland_pairwise scoring
            #                   but extremum Copeland still fires above.
            # base_only_strict: skips extremum Copeland too — TRUE floor.
            results_b: list[Result] = []
            for did in pool:
                base_s = base.get(did, 0.0)
                if base_s > 0:
                    results_b.append(Result(
                        doc_id=did, score=base_s,
                        rerank=base_s, match=0.0, recency=0.0,
                    ))
            results_b.sort(key=lambda r: r.score, reverse=True)
            return results_b[:k]

        results: list[Result] = []
        for did in pool:
            base_s = base.get(did, 0.0)
            m = match_eff[did]
            b = base_s + m
            if b > 0:
                results.append(
                    Result(
                        doc_id=did,
                        score=b,
                        rerank=base_s,
                        match=m,
                        recency=0.0,
                    )
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def _copeland_pairwise_rerank(
        self,
        pool: list[str],
        base: dict[str, float],
        match_eff: dict[str, float],
        anchored: set[str],
        k: int,
    ) -> list["Result"]:
        """Copeland pairwise rerank with heterogeneous comparison rule.

        - timed vs timed: compare by (base + match_eff)
        - any pair involving a timeless doc: compare by base only

        Avoids fabricating a temporal match value for timeless docs.
        """
        wins: dict[str, int] = dict.fromkeys(pool, 0)
        margins: dict[str, float] = dict.fromkeys(pool, 0.0)
        for a in pool:
            for b in pool:
                if a == b:
                    continue
                if a in anchored and b in anchored:
                    sa = base.get(a, 0.0) + match_eff.get(a, 0.0)
                    sb = base.get(b, 0.0) + match_eff.get(b, 0.0)
                else:
                    sa = base.get(a, 0.0)
                    sb = base.get(b, 0.0)
                if sa > sb:
                    wins[a] += 1
                    margins[a] += sa - sb
        ranked = sorted(
            pool, key=lambda d: (-wins[d], -margins[d], -base.get(d, 0.0))
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

    # ----------------------------------------------------------------- Util
    def _cosine_all(self, q_emb: np.ndarray) -> dict[str, float]:
        if not self._doc_emb:
            return {}
        qn = float(np.linalg.norm(q_emb)) or 1e-9
        out = {}
        for did, v in self._doc_emb.items():
            vn = float(np.linalg.norm(v)) or 1e-9
            out[did] = float(np.dot(q_emb, v) / (qn * vn))
        return out

    def _copeland_proximity_rerank(
        self,
        pool: list[str],
        base: dict[str, float],
        match_eff: dict[str, float],
        query_anchor_us,  # Endpoint: int µs, POS_INF, or NEG_INF
        k: int,
    ) -> list[Result]:
        """Copeland proximity tournament.

        Two ORTHOGONAL concerns combined per-pair:

        1. Match dimension (heterogeneous, from `_copeland_pairwise_rerank`):
           - timed-vs-timed pairs compare on `base + match_eff`
           - any pair involving a timeless doc compares on `base` only
           Avoids fabricating an overlap match score for timeless docs.

        2. Proximity dimension:
           - Each doc has a proximity anchor: midpoint of its intervals
             that is CLOSEST to `query_anchor_us` (for finite anchors),
             or the MAX midpoint when query anchor is POS_INF, or MIN
             when NEG_INF. Timeless docs use `ref_us` as their anchor.
           - In each pair, whichever doc's anchor is closer to
             `query_anchor_us` gets `+bonus` in the head-to-head.

        The two dimensions add into the pairwise score independently.
        """
        from .time_range import is_inf

        bonus = float(self.copeland_bonus or 0.0)
        if base:
            vals = list(base.values())
            bonus = bonus * (max(vals) - min(vals))

        anchored = {did for did in pool if self._doc_anchors.get(did)}

        # Per-doc proximity anchor.
        # - Finite query_anchor: pick midpoint closest to it.
        # - POS_INF: max midpoint (later wins).
        # - NEG_INF: min midpoint (earlier wins).
        # - No intervals (timeless): use doc's ref_us — the message/event
        #   metadata time. Timeless docs DO have a time, just not an
        #   extracted content-derived interval.
        doc_anchor: dict[str, int] = {}
        for did in pool:
            ivs = self._doc_ivs.get(did, [])
            if ivs:
                mids = [(iv.earliest_us + iv.latest_us) // 2 for iv in ivs]
                if is_inf(query_anchor_us):
                    # POS_INF → max, NEG_INF → min
                    doc_anchor[did] = (
                        max(mids) if query_anchor_us > 0 else min(mids)
                    )
                else:
                    doc_anchor[did] = min(
                        mids, key=lambda m: abs(m - query_anchor_us)
                    )
            else:
                doc_anchor[did] = self._doc_ref_us[did]

        def closeness(d: str):
            """Closeness score; larger = closer. Handles ±∞ specially."""
            da = doc_anchor[d]
            if is_inf(query_anchor_us):
                return da if query_anchor_us > 0 else -da
            return -abs(da - query_anchor_us)

        wins: dict[str, int] = dict.fromkeys(pool, 0)
        margins: dict[str, float] = dict.fromkeys(pool, 0.0)

        for a in pool:
            ca = closeness(a)
            for b in pool:
                if a == b:
                    continue
                cb = closeness(b)
                # Match dimension: heterogeneous (timeless = base only)
                if a in anchored and b in anchored:
                    sa = base.get(a, 0.0) + match_eff.get(a, 0.0)
                    sb = base.get(b, 0.0) + match_eff.get(b, 0.0)
                else:
                    sa = base.get(a, 0.0)
                    sb = base.get(b, 0.0)
                # Proximity dimension: closer doc gets +bonus
                if ca > cb:
                    sa += bonus
                elif cb > ca:
                    sb += bonus
                if sa > sb:
                    wins[a] += 1
                    margins[a] += sa - sb

        if self.copeland_tiebreak == "base":
            tertiary = base
        else:
            # "sim" mode tertiary uses heterogeneous match too
            tertiary = {
                d: base.get(d, 0.0)
                + (match_eff.get(d, 0.0) if d in anchored else 0.0)
                for d in pool
            }
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

    def stats(self) -> dict[str, Any]:
        return {
            "n_docs": len(self._docs),
            "planner": self._planner.stats(),
        }

    def doc_intervals(self) -> dict[str, list[V1Interval]]:
        return dict(self._doc_ivs)

    def doc_anchors(self) -> dict[str, list[IntervalSet]]:
        return dict(self._doc_anchors)
