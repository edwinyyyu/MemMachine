"""V7 retriever: TimeRange-based temporal layer composed with V1's
production planner (DNF) and extractor (single-pass envelope).

Behavioral differences from V1:
- The temporal-match score `match` is computed via
  `final_score(query_refs, doc_refs)` instead of `evaluate_dnf_match`.
- The filter / pool admission is fully derived from `match > 0`
  (no separate `doc_passes_filter`).
- All clause-and-leaf composition happens via TimeRange set ops.

Everything else (semantic embed, hybrid pool, cross-encoder rerank,
recency boost, additive scoring) mirrors V1 — to make the A/B
comparison clean.
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
from temporal_retrieval_min.planner import Constraint, QueryPlan, QueryPlanner
from temporal_retrieval_min.schema import parse_iso, to_us

from .adapters import (
    extractor_to_doc_refs,
    plan_to_query_refs,
)
from .planner_direct import DirectPlan, DirectQueryPlanner
from .scoring import final_score
from .time_range import TimeRange, is_infinite_measure


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
# Retriever
# ---------------------------------------------------------------------------


class TemporalRetrieverV7:
    """V7 temporal retriever — TimeRange semantics composed with V1 LLMs.

    Parameters mirror V1's `TemporalRetriever` so A/B comparison is direct.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        rerank_fn: RerankFn,
        cache_dir: str | Path = "cache/temporal_retrieval_v7",
        pool_size: int = 10,
        planner: QueryPlanner | None = None,
        extractor: TemporalExtractorV3_3 | None = None,
    ) -> None:
        self.embed_fn = embed_fn
        self.rerank_fn = rerank_fn
        self.pool_size = pool_size
        self._cache_dir = Path(cache_dir)

        self._planner = planner or QueryPlanner()
        self._extractor = extractor or TemporalExtractorV3_3()

        # Indexed state
        self._docs: dict[str, Doc] = {}
        self._doc_ivs: dict[str, list[V1Interval]] = {}
        self._doc_refs: dict[str, list[TimeRange]] = {}
        self._doc_emb: dict[str, np.ndarray] = {}
        self._doc_ref_us: dict[str, int] = {}

    # ----------------------------------------------------------------- Index
    async def index(self, docs: list[Doc]) -> None:
        self._docs = {d.id: d for d in docs}
        self._doc_ref_us = {d.id: to_us(parse_iso(d.ref_time)) for d in docs}

        async def _extract_one(d: Doc) -> tuple[str, list[V1Interval]]:
            try:
                ivs = await self._extractor.extract(d.text, parse_iso(d.ref_time))
            except Exception:
                ivs = []
            return d.id, ivs

        results = await asyncio.gather(*(_extract_one(d) for d in docs))
        for did, ivs in results:
            self._doc_ivs[did] = ivs
            self._doc_refs[did] = extractor_to_doc_refs(ivs)
        self._extractor.save_caches()

        embs = await self.embed_fn([d.text for d in docs])
        for d, e in zip(docs, embs, strict=False):
            self._doc_emb[d.id] = np.asarray(e, dtype=np.float32)

    # ----------------------------------------------------------------- Query
    async def query(self, query: str, ref_time: str, k: int = 10) -> list[Result]:
        plan: QueryPlan = await self._planner.plan(query, ref_time)

        # Resolve every distinct leaf phrase once via the extractor.
        all_leaves: list[Constraint] = [
            leaf for clause in plan.expr for leaf in clause
        ]
        anchors_by_phrase = await self._resolve_leaf_phrases(
            ref_time, all_leaves
        )

        def resolver(leaf: Constraint) -> list[V1Interval]:
            return anchors_by_phrase.get(leaf.phrase, [])

        query_refs: list[TimeRange] = plan_to_query_refs(plan, resolver)

        # Bounded vs unbounded refs determine timeless-doc filter
        # admission. A ref with FINITE measure (points to a specific
        # bounded calendar window) signals "this query is asking about
        # a specific time" — timeless docs don't carry the temporal
        # evidence to satisfy it and should not filter-pass (they only
        # ride into the pool via the semantic top-up). All-unbounded
        # refs (pure disjoint complements, or open-ended directional
        # ranges) allow timeless to pass — the query is more about
        # excluding/orienting than naming a specific time. Pure
        # TimeRange-shape rule, not relation-aware.
        bounded_ref_present = any(
            not is_infinite_measure(r) for r in query_refs
        )

        # Semantic
        q_emb = (await self.embed_fn([query]))[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sem_scores = self._cosine_all(q_emb)

        all_dids = list(self._doc_ref_us.keys())

        # Two-track filter / score:
        # - filter (pool admission): doc must have refs AND score>0,
        #   OR be timeless when no clause is bounded.
        # - rank score: timeless docs still pass-through to 1.0 so
        #   semantic + rerank decide ordering for them within the pool.
        match_all: dict[str, float] = {}
        eligible: list[str] = []
        for did in all_dids:
            d_refs = self._doc_refs.get(did, [])
            if not d_refs:
                match_all[did] = 1.0
                if not query_refs or not bounded_ref_present:
                    eligible.append(did)
                # Bounded-ref + timeless: pass-through for rank,
                # but NOT filter-admitted (may still enter via top-up).
            else:
                s = final_score(query_refs, d_refs)
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
        r_full = normalize_rerank_full(rerank_pool, all_dids, 0.0)

        # Recency
        rec = self._compute_recency(plan, pool, match_all)

        # Additive scoring
        r_pool = {did: r_full.get(did, 0.0) for did in pool}
        base = normalize_dict(r_pool)
        results: list[Result] = []
        for did in pool:
            base_s = base.get(did, 0.0)
            r_v = rec.get(did, 0.0)
            m = match_all[did]
            b = base_s + m
            if plan.latest_intent or plan.earliest_intent:
                b += r_v
            if b > 0:
                results.append(
                    Result(
                        doc_id=did,
                        score=b,
                        rerank=base_s,
                        match=m,
                        recency=r_v,
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

    async def _resolve_leaf_phrases(
        self,
        ref_time: str,
        leaves: list[Constraint],
    ) -> dict[str, list[V1Interval]]:
        """Run the extractor on every distinct leaf phrase.

        Returns a map `phrase -> list[Interval]`. The extractor's output
        for a phrase depends only on the phrase text and `ref_time`; the
        planner's per-leaf `relation` tag has no role here — V7 absorbs
        relations into TimeRange shapes downstream via `leaf_to_range`.
        """
        if not leaves:
            return {}
        rt = parse_iso(ref_time)
        unique_phrases = list({leaf.phrase for leaf in leaves})

        async def _extract(phrase: str):
            try:
                return phrase, await self._extractor.extract(phrase, rt)
            except Exception:
                return phrase, []

        results = await asyncio.gather(*(_extract(p) for p in unique_phrases))
        return dict(results)

    def _compute_recency(
        self,
        plan: QueryPlan,
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
            ivs = self._doc_ivs.get(did, [])
            bundles[did] = [{"intervals": ivs}] if ivs else []
        return recency_scores(
            bundles,
            {d: self._doc_ref_us[d] for d in target},
            direction=direction,
        )

    def _compute_recency_direct(
        self,
        direct_plan: DirectPlan,
        pool: list[str],
        match: dict[str, float],
    ) -> dict[str, float]:
        """Recency for DirectQueryPlanner outputs (no QueryPlan object)."""
        if not (direct_plan.latest_intent or direct_plan.earliest_intent):
            return {}
        direction = "latest" if direct_plan.latest_intent else "earliest"
        match_passers = [d for d in pool if match.get(d, 0.0) > 0.0]
        target = match_passers if len(match_passers) >= 2 else pool
        if len(target) < 2:
            return {}
        bundles = {}
        for did in target:
            ivs = self._doc_ivs.get(did, [])
            bundles[did] = [{"intervals": ivs}] if ivs else []
        return recency_scores(
            bundles,
            {d: self._doc_ref_us[d] for d in target},
            direction=direction,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "n_docs": len(self._docs),
            "planner": self._planner.stats(),
        }

    def doc_intervals(self) -> dict[str, list[V1Interval]]:
        return dict(self._doc_ivs)

    def doc_refs(self) -> dict[str, list[TimeRange]]:
        return dict(self._doc_refs)


# ---------------------------------------------------------------------------
# Direct-planner subclass
# ---------------------------------------------------------------------------


class TemporalRetrieverV7Direct(TemporalRetrieverV7):
    """V7 retriever using `DirectQueryPlanner` — drops the relation enum.

    Differences from `TemporalRetrieverV7`:
    - One LLM call per query (planner emits TimeRange JSON directly), no
      separate leaf-extractor calls for query phrases.
    - `_planner` is a `DirectQueryPlanner` instead of the legacy
      `QueryPlanner`. The doc-side extractor is unchanged.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        rerank_fn: RerankFn,
        cache_dir: str | Path = "cache/temporal_retrieval_v7_direct",
        pool_size: int = 10,
        planner: DirectQueryPlanner | None = None,
        extractor: TemporalExtractorV3_3 | None = None,
    ) -> None:
        super().__init__(
            embed_fn=embed_fn,
            rerank_fn=rerank_fn,
            cache_dir=cache_dir,
            pool_size=pool_size,
            planner=None,  # override below
            extractor=extractor,
        )
        # Replace the legacy planner with the direct one.
        self._planner = planner or DirectQueryPlanner()  # type: ignore[assignment]

    async def query(self, query: str, ref_time: str, k: int = 10) -> list[Result]:
        # Direct planner returns refs + extremum in one LLM call —
        # no per-leaf extractor needed for the query side.
        direct_plan: DirectPlan = await self._planner.plan(query, ref_time)  # type: ignore[assignment]
        query_refs: list[TimeRange] = direct_plan.refs

        # Filter strictness rule mirrors the legacy retriever's:
        # timeless docs only filter-pass when no ref has finite measure.
        bounded_ref_present = any(
            not is_infinite_measure(r) for r in query_refs
        )

        q_emb = (await self.embed_fn([query]))[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sem_scores = self._cosine_all(q_emb)

        all_dids = list(self._doc_ref_us.keys())
        match_all: dict[str, float] = {}
        eligible: list[str] = []
        for did in all_dids:
            d_refs = self._doc_refs.get(did, [])
            if not d_refs:
                match_all[did] = 1.0
                if not query_refs or not bounded_ref_present:
                    eligible.append(did)
            else:
                s = final_score(query_refs, d_refs)
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
        results: list[Result] = []
        for did in pool:
            base_s = base.get(did, 0.0)
            r_v = rec.get(did, 0.0)
            m = match_all[did]
            b = base_s + m
            if direct_plan.latest_intent or direct_plan.earliest_intent:
                b += r_v
            if b > 0:
                results.append(
                    Result(
                        doc_id=did,
                        score=b,
                        rerank=base_s,
                        match=m,
                        recency=r_v,
                    )
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
