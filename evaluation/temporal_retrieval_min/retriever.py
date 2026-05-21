"""Temporal-aware retriever — clean reference implementation.

Orchestrates the LLM stages (DNF planner, single-pass unified-envelope
extractor) and the deterministic post-LLM scoring pipeline (filter +
hybrid pool + cross-encoder rerank + DNF mask + recency boost) into a
single class with `index()` / `query()` methods.

Embeddings and reranking are injected — bring your own:

    embed_fn(texts: list[str]) -> Awaitable[list[np.ndarray]]
    rerank_fn(query: str, doc_texts: list[str]) -> Awaitable[list[float]]

Usage:
    retriever = TemporalRetriever(embed_fn, rerank_fn)
    await retriever.index(docs)
    results = await retriever.query(text, ref_time, k=10)

Algorithm summary:
  1. Doc-time (one-shot per corpus): single-call LLM extractor produces
     temporal envelopes (earliest, latest) for each doc. Embed text.
  2. Query-time:
     a. LLM plan -> DNF expression of (phrase, relation) leaves.
     b. For each leaf, run the extractor against ref_time. The
        extractor's skip-don't-emit rule means an empty emission IS
        the "not a calendar phrase" signal; whatever it returns
        becomes the leaf's anchor.
     c. Build hybrid pool: top-K/2 semantic U top-K/2 filter-passing.
     d. Score = pool_norm(base) + match_weight*match + recency_weight*recency.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    Interval,
    build_pool,
    doc_passes_filter,
    normalize_dict,
    normalize_rerank_full,
    recency_scores,
)
from .extractor_v3_3 import TemporalExtractorV3_3
from .planner import QueryPlan, QueryPlanner, evaluate_dnf_match
from .schema import parse_iso, to_us

# ===========================================================================
# Public data types
# ===========================================================================


@dataclass
class Doc:
    id: str
    text: str
    ref_time: str  # ISO 8601 UTC, e.g. "2024-03-15T12:00:00Z"


@dataclass
class Result:
    doc_id: str
    score: float
    rerank: float
    match: float
    recency: float


# ===========================================================================
# Retriever
# ===========================================================================

EmbedFn = Callable[[list[str]], Awaitable[list[np.ndarray]]]
RerankFn = Callable[[str, list[str]], Awaitable[list[float]]]


class TemporalRetriever:
    """End-to-end temporal-aware retriever."""

    def __init__(
        self,
        embed_fn: EmbedFn,
        rerank_fn: RerankFn,
        cache_dir: str | Path = "cache/temporal_retrieval",
        pool_size: int = 10,
        planner: QueryPlanner | None = None,
        extractor: TemporalExtractorV3_3 | None = None,
    ) -> None:
        """Construct the retriever.

        `planner` and `extractor` can be supplied to override the
        defaults — useful when a caller wants to vary the prompt or
        cache subdir per-instance without accessing private state.

        Scoring is ADDITIVE with unit weights:
            score = pool_norm(base) + match + recency  (recency only when
                                                        plan has extremum)

        `match` is the doc's temporal-constraint match score in [0, 1]
        — how well the doc's extracted envelopes satisfy the planner's
        DNF constraint.

        Parameters that were previously configurable but had no
        proven-better alternative were removed from this clean
        surface; the full research variant in `temporal_retrieval/`
        retains them as A/B switches. The hardcoded behaviors here:
        - match_floor=0.0 (hard-gate DNF match)
        - empty_doc_match=1.0 (timeless docs trust cosine fully)
        - no ref_time fallback
        - notin_aggregate=False (strict max-pair containment)
        - match_weight=recency_weight=1.0 (equal weights)
        - no confidence_floor (the extractor's skip-don't-emit
          semantics replace it: empty output means "no calendar
          reference", not "low confidence")
        """
        self.embed_fn = embed_fn
        self.rerank_fn = rerank_fn
        self.pool_size = pool_size
        self._cache_dir = Path(cache_dir)

        self._planner = planner or QueryPlanner()
        self._extractor = extractor or TemporalExtractorV3_3()

        # Indexed state
        self._docs: dict[str, Doc] = {}
        self._doc_ivs: dict[str, list[Interval]] = {}
        self._doc_emb: dict[str, np.ndarray] = {}
        self._doc_ref_us: dict[str, int] = {}

    # -----------------------------------------------------------------
    # Indexing
    # -----------------------------------------------------------------
    async def index(self, docs: list[Doc]) -> None:
        """One-time pass: extract intervals (LLM, cached) and embed text."""
        self._docs = {d.id: d for d in docs}
        self._doc_ref_us = {d.id: to_us(parse_iso(d.ref_time)) for d in docs}

        import asyncio

        async def _extract_one(d: Doc) -> tuple[str, list[Interval]]:
            try:
                ivs = await self._extractor.extract(d.text, parse_iso(d.ref_time))
            except Exception:
                ivs = []
            return d.id, ivs

        results = await asyncio.gather(*(_extract_one(d) for d in docs))
        for did, ivs in results:
            self._doc_ivs[did] = ivs

        # Persist extractor caches
        self._extractor.save_caches()

        # Embed
        embs = await self.embed_fn([d.text for d in docs])
        for d, e in zip(docs, embs, strict=False):
            self._doc_emb[d.id] = np.asarray(e, dtype=np.float32)

    # -----------------------------------------------------------------
    # Querying
    # -----------------------------------------------------------------
    async def query(self, query: str, ref_time: str, k: int = 10) -> list[Result]:
        # 1. Plan
        plan: QueryPlan = await self._planner.plan(query, ref_time)
        leaves_flat = [
            (ci, li, leaf)
            for ci, clause in enumerate(plan.expr)
            for li, leaf in enumerate(clause)
        ]

        # 2. Resolve each leaf's anchor by running the extractor on the
        # leaf phrase. Skip-don't-emit means an empty emission IS the
        # "not a calendar phrase" signal.
        anchors = await self._resolve_anchors(ref_time, leaves_flat)

        # 4. Build filter sets
        valid_includes: list[tuple[str, list[Interval]]] = []
        valid_excludes: list[list[Interval]] = []
        for ci, li, leaf in leaves_flat:
            ivs = anchors.get((ci, li), [])
            if not ivs:
                continue
            if leaf.relation == "disjoint":
                valid_excludes.append(ivs)
            else:
                valid_includes.append((leaf.relation, ivs))

        all_dids = list(self._doc_ref_us.keys())
        eligible = [
            did
            for did in all_dids
            if doc_passes_filter(
                self._doc_ivs.get(did, []), valid_includes, valid_excludes
            )
        ]

        # 5. Hybrid pool
        q_emb = (await self.embed_fn([query]))[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sem_scores = self._cosine_all(q_emb)
        pool = build_pool(sem_scores, all_dids, eligible, self.pool_size)

        if not pool:
            return []

        # 6. Cross-encoder rerank
        pool_texts = [self._docs[did].text for did in pool]
        rerank_scores = await self.rerank_fn(query, pool_texts)
        rerank_pool = dict(zip(pool, rerank_scores, strict=False))
        r_full = normalize_rerank_full(rerank_pool, all_dids, 0.0)

        # 7. Match score per pool member (DNF constraint evaluation)
        def _resolver(ci: int, li: int, leaf) -> list[Interval]:
            return anchors.get((ci, li), [])

        match: dict[str, float] = {}
        for did in pool:
            extracted_ivs = self._doc_ivs.get(did, [])
            if not extracted_ivs and plan.expr:
                # Timeless doc + temporal query: full pass-through so
                # cosine + recency decide.
                match[did] = 1.0
            else:
                match[did] = evaluate_dnf_match(plan, extracted_ivs, _resolver)

        # 8. Recency boost (only when extremum is requested)
        rec = self._compute_recency(plan, pool, match)

        # 9. Final scoring — additive with unit weights:
        #    score = base + match + recency
        # `recency` already encodes direction (latest vs earliest) —
        # higher means "better matches the requested extremum direction."
        r_pool = {did: r_full.get(did, 0.0) for did in pool}
        base = normalize_dict(r_pool)
        results: list[Result] = []
        for did in pool:
            base_s = base.get(did, 0.0)
            r_v = rec.get(did, 0.0)
            b = base_s + match[did]
            if plan.latest_intent or plan.earliest_intent:
                b += r_v
            if b > 0:
                results.append(
                    Result(
                        doc_id=did,
                        score=b,
                        rerank=base_s,
                        match=match[did],
                        recency=r_v,
                    )
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _cosine_all(self, q_emb: np.ndarray) -> dict[str, float]:
        if not self._doc_emb:
            return {}
        qn = float(np.linalg.norm(q_emb)) or 1e-9
        out = {}
        for did, v in self._doc_emb.items():
            vn = float(np.linalg.norm(v)) or 1e-9
            out[did] = float(np.dot(q_emb, v) / (qn * vn))
        return out

    async def _resolve_anchors(
        self,
        ref_time: str,
        leaves_flat: list[tuple],
    ) -> dict[tuple[int, int], list[Interval]]:
        """Resolve each planner leaf to a list of anchor intervals.

        Principle: trust the extractor. Every leaf phrase goes through
        the extractor against `ref_time`. The extractor's skip-don't-emit
        rule for unanchorable phrases like "March", "the launch",
        "grad school" means an empty emission IS the "this isn't a
        calendar phrase" signal. Whatever the extractor returns is used
        directly as the leaf's anchor; an empty list means the leaf is
        a no-op.
        """
        if not leaves_flat:
            return {}

        import asyncio

        rt = parse_iso(ref_time)

        async def _extract_phrase(text: str):
            try:
                return await self._extractor.extract(text, rt)
            except Exception:
                return []

        envs_list = await asyncio.gather(
            *(_extract_phrase(leaf.phrase) for (_, _, leaf) in leaves_flat)
        )

        out: dict[tuple[int, int], list[Interval]] = {}
        for (ci, li, _), envs in zip(leaves_flat, envs_list, strict=False):
            ivs = flatten_intervals(envs)
            if ivs:
                out[(ci, li)] = ivs
        return out

    def _compute_recency(
        self,
        plan: QueryPlan,
        pool: list[str],
        match: dict[str, float],
    ) -> dict[str, float]:
        """Compute per-doc recency scores for the pool, restricted to
        docs whose temporal-match score is positive (i.e., they have
        some evidence of satisfying the planner's DNF constraint).

        Restricting to constraint-satisfying docs preserves the
        relative recency ordering of relevant docs: a 2090-anchored
        off-topic doc shouldn't compress the recency ranking of docs
        that actually satisfy the constraint. Combined with the
        rank-based normalization in `recency_scores`, this gives
        outlier-robust recency among the docs that matter.

        If fewer than 2 docs have positive match, fall back to
        normalizing over the whole pool — recency comparison needs at
        least 2 anchors.
        """
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

    # -----------------------------------------------------------------
    # Diagnostics & introspection
    # -----------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "n_docs": len(self._docs),
            "planner": self._planner.stats(),
        }

    def doc_intervals(self) -> dict[str, list[Interval]]:
        """Return the extracted intervals per indexed doc."""
        return dict(self._doc_ivs)
