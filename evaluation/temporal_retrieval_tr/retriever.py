"""Temporal retriever: IntervalSet-based temporal layer.

The retriever composes the `QueryPlanner` (IntervalSet targets from a
single LLM call) with a doc-side extractor producing one envelope per
temporal reference. Scoring is `final_score(query_targets, doc_anchors)`
plus a semantic / rerank base, with optional recency boost when the
planner flags an extremum.
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
    normalize_dict,
    normalize_rerank_full,
    recency_scores,
)
from temporal_retrieval_min.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval_min.schema import parse_iso, to_us

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


def extractor_to_anchors(ivs: list[V1Interval]) -> list[IntervalSet]:
    """Convert extractor envelopes → one IntervalSet per envelope.

    Each envelope is already a half-open `[earliest_us, latest_us)`.
    Wrap each as a single-interval IntervalSet (preserves the
    per-mention granularity expected by per-anchor scoring).
    """
    out: list[IntervalSet] = []
    for iv in ivs:
        if iv.latest_us > iv.earliest_us:
            out.append(IntervalSet(intervals=(Interval(iv.earliest_us, iv.latest_us),)))
    return out


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class TemporalRetriever:
    """Temporal retriever — IntervalSet semantics via the QueryPlanner.

    The query side issues one LLM call per query (the planner emits
    IntervalSet JSON directly). The doc side uses the production
    single-pass extractor for envelope extraction.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        rerank_fn: RerankFn,
        cache_dir: str | Path = "cache/temporal_retrieval_tr",
        pool_size: int = 40,
        planner: QueryPlanner | None = None,
        extractor: TemporalExtractorV3_3 | None = None,
        copeland_bonus: float = 0.40,
        copeland_tiebreak: str = "sim",
        recency_anchor: str = "extreme",
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
        # Per-doc recency anchor selection (Copeland uses these):
        # - "extreme": extreme-midpoint of extracted intervals
        #   (max for latest / min for earliest), fallback to ref_time
        # - "ref_time": ignore extracted intervals; always use ref_time
        # - "median": median midpoint of extracted intervals, fallback to ref_time
        # - "primary": use the LLM-tagged primary interval (requires extractor
        #              that emits primary_index), fallback to ref_time
        self.recency_anchor = recency_anchor
        # Scoring substrate (shipped — see _SCORING_ARCHITECTURE.md for rationale):
        # - base = raw cosine (kept in native units, query-independent)
        # - match_eff = match * pool_cosine_spread (scales system-specific score)
        # - bonus_eff = copeland_bonus * pool_cosine_spread (same)
        # This preserves cosine's universal interpretability while keeping the
        # base-vs-match-vs-recency balance stable across pool compositions.
        self._cache_dir = Path(cache_dir)

        self._planner = planner or QueryPlanner()
        self._extractor = extractor or TemporalExtractorV3_3()

        # Indexed state
        self._docs: dict[str, Doc] = {}
        self._doc_ivs: dict[str, list[V1Interval]] = {}
        self._doc_anchors: dict[str, list[IntervalSet]] = {}
        self._doc_emb: dict[str, np.ndarray] = {}
        self._doc_ref_us: dict[str, int] = {}
        # Optional per-doc primary index (None when extractor doesn't emit it
        # or when the LLM declined to mark one)
        self._doc_primary_idx: dict[str, int | None] = {}

    # ----------------------------------------------------------------- Index
    async def index(self, docs: list[Doc]) -> None:
        self._docs = {d.id: d for d in docs}
        self._doc_ref_us = {d.id: to_us(parse_iso(d.ref_time)) for d in docs}

        async def _extract_one(d: Doc) -> tuple[str, list[V1Interval], int | None]:
            try:
                out = await self._extractor.extract(d.text, parse_iso(d.ref_time))
                # Backward-compatible: V3.3 returns list[Interval],
                # V3.4 returns (list[Interval], primary_index | None)
                if isinstance(out, tuple):
                    ivs, primary_idx = out
                else:
                    ivs, primary_idx = out, None
            except Exception:
                ivs, primary_idx = [], None
            return d.id, ivs, primary_idx

        results = await asyncio.gather(*(_extract_one(d) for d in docs))
        for did, ivs, primary_idx in results:
            self._doc_ivs[did] = ivs
            self._doc_anchors[did] = extractor_to_anchors(ivs)
            self._doc_primary_idx[did] = primary_idx
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
        for did in all_dids:
            d_anchors = self._doc_anchors.get(did, [])
            if not d_anchors:
                match_all[did] = 1.0
                if not query_targets or not bounded_target_present:
                    eligible.append(did)
                # Bounded-target + timeless: pass-through for rank,
                # but NOT filter-admitted (may still enter via top-up).
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

        # Extremum queries → Copeland tournament re-rank with per-pair
        # bonus to the more-recent doc. Non-extremum queries → just
        # base + match (no recency layer applies).
        if plan.latest_intent or plan.earliest_intent:
            return self._copeland_rerank(pool, base, match_eff, plan, k)

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

    def _compute_recency(
        self,
        plan: Plan,
        pool: list[str],
        match: dict[str, float],
    ) -> dict[str, float]:
        if not (plan.latest_intent or plan.earliest_intent):
            return {}
        direction = "latest" if plan.latest_intent else "earliest"
        match_passers = [d for d in pool if match.get(d, 0.0) > 0.0]
        target = match_passers if len(match_passers) >= 2 else pool
        if len(target) < 2:
            return {}
        bundles = {}
        for did in target:
            if self.recency_anchor == "ref_time":
                # Bypass extracted intervals; recency_scores will fall back to ref_us
                bundles[did] = []
            elif self.recency_anchor == "median":
                ivs = self._doc_ivs.get(did, [])
                if ivs:
                    mids = sorted((iv.earliest_us + iv.latest_us) // 2 for iv in ivs)
                    med = mids[len(mids) // 2]
                    # Wrap median as a single zero-width interval so recency_scores
                    # picks it (the extreme-of-one is itself).
                    bundles[did] = [{"intervals": [V1Interval(earliest_us=med, latest_us=med + 1)]}]
                else:
                    bundles[did] = []
            elif self.recency_anchor == "primary":
                ivs = self._doc_ivs.get(did, [])
                pidx = self._doc_primary_idx.get(did)
                if pidx is not None and 0 <= pidx < len(ivs):
                    # Wrap primary as a single interval — recency_scores will pick its midpoint
                    bundles[did] = [{"intervals": [ivs[pidx]]}]
                else:
                    # No primary marked → fall back to ref_time
                    bundles[did] = []
            else:  # "extreme" (default, current behavior)
                ivs = self._doc_ivs.get(did, [])
                bundles[did] = [{"intervals": ivs}] if ivs else []
        return recency_scores(
            bundles,
            {d: self._doc_ref_us[d] for d in target},
            direction=direction,
        )

    def _copeland_rerank(
        self,
        pool: list[str],
        base: dict[str, float],
        match_all: dict[str, float],
        plan: Plan,
        k: int,
    ) -> list[Result]:
        """Copeland tournament re-rank for extremum queries.

        For each pair (a, b) in the pool, the more-temporally-anchored
        doc gets `+copeland_bonus` in the head-to-head comparison
        against `sim = base + match`. Doc-level temporal anchor:
        max-midpoint over intervals for "latest" (min for "earliest"),
        falling back to the doc's ref_us when no intervals exist.

        Final ranking is (wins, margin_sum) lexicographically. Synthetic
        Result.score = wins + ε·margins for sortable monotonicity.

        Fixed-bonus property: A's pairwise advantage over any
        less-recent doc B is exactly `bonus`, regardless of rank-gap
        — what additive linear-rank cannot express.
        """
        # Scale bonus by pool cosine spread (system-specific score tracks
        # the pool's natural scale, while base stays in raw cosine).
        bonus = float(self.copeland_bonus or 0.0)
        if base:
            vals = list(base.values())
            bonus = bonus * (max(vals) - min(vals))
        direction_latest = plan.latest_intent
        pool_set = set(pool)

        # Per-doc temporal anchor — honors `self.recency_anchor` setting
        # so this matches the additive path's anchor selection.
        anchors: dict[str, int] = {}
        for did in pool:
            ivs = self._doc_ivs.get(did, [])
            anchor: int | None = None
            if ivs and self.recency_anchor == "extreme":
                for iv in ivs:
                    mid = (iv.earliest_us + iv.latest_us) // 2
                    if anchor is None:
                        anchor = mid
                    elif direction_latest and mid > anchor:
                        anchor = mid
                    elif (not direction_latest) and mid < anchor:
                        anchor = mid
            elif ivs and self.recency_anchor == "median":
                mids = sorted((iv.earliest_us + iv.latest_us) // 2 for iv in ivs)
                anchor = mids[len(mids) // 2]
            elif ivs and self.recency_anchor == "primary":
                pidx = self._doc_primary_idx.get(did)
                if pidx is not None and 0 <= pidx < len(ivs):
                    iv = ivs[pidx]
                    anchor = (iv.earliest_us + iv.latest_us) // 2
            # "ref_time" mode (or no intervals / no primary): anchor stays None → fallback
            anchors[did] = anchor if anchor is not None else self._doc_ref_us[did]

        sim: dict[str, float] = {}
        for did in pool:
            sim[did] = base.get(did, 0.0) + match_all[did]

        wins: dict[str, int] = dict.fromkeys(pool, 0)
        margins: dict[str, float] = dict.fromkeys(pool, 0.0)

        def is_more_recent(a: str, b: str) -> bool:
            aa, ab = anchors[a], anchors[b]
            if aa == ab:
                return False
            return (aa > ab) if direction_latest else (aa < ab)

        for a in pool:
            for b in pool:
                if a == b:
                    continue
                sa = sim[a] + (bonus if is_more_recent(a, b) else 0.0)
                sb = sim[b] + (bonus if is_more_recent(b, a) else 0.0)
                if sa > sb:
                    wins[a] += 1
                    margins[a] += sa - sb

        if self.copeland_tiebreak == "base":
            tertiary = base
        else:  # "sim" (default)
            tertiary = sim
        ranked = sorted(
            pool,
            key=lambda d: (-wins[d], -margins[d], -tertiary.get(d, 0.0)),
        )
        results: list[Result] = []
        for did in ranked:
            # Synthetic score: wins dominate, margins as fine tiebreak.
            score = float(wins[did]) + margins[did] * 1e-4
            results.append(
                Result(
                    doc_id=did,
                    score=score,
                    rerank=base.get(did, 0.0),
                    match=match_all[did],
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
