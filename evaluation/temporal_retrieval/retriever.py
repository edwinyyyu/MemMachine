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
    flatten_intervals,
    normalize_dict,
    normalize_rerank_full,
    recency_scores,
)
from .extractor_v3_2 import TemporalExtractorV3_2
from .extractor_v3_3 import TemporalExtractorV3_3
from .planner import QueryPlan, QueryPlanner, evaluate_dnf_match
from .planner_tree import (
    Leaf as TreeLeaf,
)
from .planner_tree import (
    TreePlanner,
    TreeQueryPlan,
    evaluate_tree_match,
)
from .schema import TimeEnvelope, parse_iso, to_us

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
        match_floor: float = 0.0,
        empty_doc_match: float | None = 1.0,
        ref_time_fallback: str | None = None,
        notin_aggregate: bool = False,
        scoring: str = "additive",
        match_weight: float = 1.0,
        recency_weight: float = 1.0,
        recency_alpha: float = 3.0,
        planner: QueryPlanner | TreePlanner | None = None,
        extractor: TemporalExtractorV3_2 | TemporalExtractorV3_3 | None = None,
    ) -> None:
        """Construct the retriever.

        `planner` and `extractor` can be supplied to override the
        defaults — useful when a caller wants to vary the prompt or
        cache subdir per-instance without accessing private state.

        Scoring is ADDITIVE:
            score = pool_norm(base) + match_weight * match
                                    + recency_weight * recency  (extremum only)

        `match` is the doc's temporal-constraint match score in [0, 1]
        — how well the doc's extracted envelopes satisfy the planner's
        DNF constraint.

        `match_weight` / `recency_weight` control how strongly the
        temporal match and recency boost weigh against the base
        cosine signal. The 1.0/1.0 default treats all three
        contributions equally.

        `match_floor` softens the DNF match from a hard binary [0, 1]
        gate into a linear interpolation. Effective match becomes
        `match_floor + (1 - match_floor) * match`.

        `empty_doc_match` is the match value assigned to docs with no
        extracted temporal envelopes (timeless content). Default 1.0
        lets cosine decide for these docs without temporal penalty.

        `ref_time_fallback` controls whether to synthesize a
        point-envelope at the doc's `ref_time` for filter/match
        evaluation. None (default) | "when_empty" | "always".

        Note: there is no `confidence_floor`. The v3.3 extractor uses
        skip-don't-emit semantics for unanchorable phrases, so every
        emitted envelope is fully trusted. The floor was needed only
        for v3.1's emit-then-filter contract.

        Default extractor is v3.3 (drops `surface` and `granularity`
        from the output schema vs v3.2 — both fields were unused by
        retrieval). 35-bench A/B vs v3.2: macro +0.4pp R@1/+0.4pp R@5,
        −21% extractor latency, −16% output tokens.
        """
        self.embed_fn = embed_fn
        self.rerank_fn = rerank_fn
        self.pool_size = pool_size
        self.match_floor = match_floor
        self.empty_doc_match = empty_doc_match
        if ref_time_fallback not in (None, "when_empty", "always"):
            raise ValueError(
                f"ref_time_fallback must be None, 'when_empty', or 'always'; got {ref_time_fallback!r}"
            )
        self.ref_time_fallback = ref_time_fallback
        self.notin_aggregate = notin_aggregate
        if scoring not in ("additive", "multiplicative", "hybrid_mm_ar"):
            raise ValueError(
                f"scoring must be 'additive' | 'multiplicative' | 'hybrid_mm_ar'; got {scoring!r}"
            )
        self.scoring = scoring
        self.match_weight = match_weight
        self.recency_weight = recency_weight
        self.recency_alpha = recency_alpha
        self._cache_dir = Path(cache_dir)

        self._planner = planner or QueryPlanner()
        self._extractor = extractor or TemporalExtractorV3_3()

        # Indexed state
        self._docs: dict[str, Doc] = {}
        self._doc_envelopes: dict[str, list[TimeEnvelope]] = {}
        self._doc_ivs: dict[str, list[Interval]] = {}
        self._doc_emb: dict[str, np.ndarray] = {}
        self._doc_ref_us: dict[str, int] = {}

    # -----------------------------------------------------------------
    # Indexing
    # -----------------------------------------------------------------
    async def index(self, docs: list[Doc]) -> None:
        """One-time pass: extract envelopes (LLM, cached) and embed text."""
        self._docs = {d.id: d for d in docs}
        self._doc_ref_us = {d.id: to_us(parse_iso(d.ref_time)) for d in docs}

        import asyncio

        async def _extract_one(d: Doc) -> tuple[str, list[TimeEnvelope]]:
            try:
                envs = await self._extractor.extract(d.text, parse_iso(d.ref_time))
            except Exception:
                envs = []
            return d.id, envs

        results = await asyncio.gather(*(_extract_one(d) for d in docs))
        for did, envs in results:
            self._doc_envelopes[did] = envs
            self._doc_ivs[did] = flatten_intervals(envs)

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
        # 1. Plan (DNF or tree depending on injected planner)
        is_tree = isinstance(self._planner, TreePlanner)
        plan = await self._planner.plan(query, ref_time)

        # Build a flat list of (id, leaf) pairs to resolve. For DNF, id is
        # (ci, li); for tree, id is the object id of the leaf so the same
        # leaf identity maps to one anchor.
        leaves_flat: list[tuple]
        if is_tree:
            leaves_flat = []
            seen_ids: set[int] = set()
            for leaf in plan.iter_leaves():
                lid = id(leaf)
                if lid in seen_ids:
                    continue
                seen_ids.add(lid)
                leaves_flat.append((lid, leaf))
        else:
            leaves_flat = [
                (ci, li, leaf)
                for ci, clause in enumerate(plan.expr)
                for li, leaf in enumerate(clause)
            ]

        # 2. Resolve each leaf's anchor by running the extractor on the
        # leaf phrase.
        anchors = await self._resolve_anchors(ref_time, leaves_flat, is_tree=is_tree)

        # 4. Build filter sets
        valid_includes: list[tuple[str, list[Interval]]] = []
        valid_excludes: list[list[Interval]] = []
        if is_tree:
            # Tree filter: only push leaves down through `and`/leaf
            # spines. Inside `or`/`not` the doc-level filter would
            # over-prune, so we evaluate those purely at scoring time.
            includes, excludes = _tree_filter_leaves(plan.expr)
            for leaf in includes:
                ivs = anchors.get(id(leaf), [])
                if ivs:
                    valid_includes.append((leaf.relation, ivs))
            for leaf in excludes:
                ivs = anchors.get(id(leaf), [])
                if ivs:
                    valid_excludes.append(ivs)
        else:
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
                self._effective_ivs(did), valid_includes, valid_excludes
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

        # 7. Match score per pool member (constraint evaluation)
        if is_tree:
            def _tree_resolver(leaf: TreeLeaf) -> list[Interval]:
                return anchors.get(id(leaf), [])
        else:
            def _dnf_resolver(ci: int, li: int, leaf) -> list[Interval]:
                return anchors.get((ci, li), [])

        # `plan_has_expr`: True when there is any temporal constraint.
        plan_has_expr = (plan.expr is not None) if is_tree else bool(plan.expr)

        match: dict[str, float] = {}
        for did in pool:
            extracted_ivs = self._doc_ivs.get(did, [])
            effective_ivs = self._effective_ivs(did)
            if (
                not extracted_ivs
                and not effective_ivs
                and self.empty_doc_match is not None
                and plan_has_expr
            ):
                match[did] = self.empty_doc_match
            else:
                if is_tree:
                    match[did] = evaluate_tree_match(
                        plan,
                        effective_ivs,
                        _tree_resolver,
                        notin_aggregate=self.notin_aggregate,
                    )
                else:
                    match[did] = evaluate_dnf_match(
                        plan,
                        effective_ivs,
                        _dnf_resolver,
                        notin_aggregate=self.notin_aggregate,
                    )

        # 8. Recency boost (only when extremum is requested)
        rec = self._compute_recency(plan, pool, match)

        # 9. Final scoring. Three modes (full research surface):
        #    additive       : base + mw*match + rw*recency
        #    multiplicative : base * match * (1 + alpha*recency)
        #    hybrid_mm_ar   : base * match + rw*recency
        # `recency` already encodes direction (latest vs earliest) —
        # higher means "better matches the requested extremum direction."
        r_pool = {did: r_full.get(did, 0.0) for did in pool}
        base = normalize_dict(r_pool)
        results: list[Result] = []
        for did in pool:
            effective_match = self.match_floor + (1.0 - self.match_floor) * match[did]
            base_s = base.get(did, 0.0)
            r_v = rec.get(did, 0.0)
            has_extremum = plan.latest_intent or plan.earliest_intent
            if self.scoring == "additive":
                b = base_s + self.match_weight * effective_match
                if has_extremum:
                    b += self.recency_weight * r_v
            elif self.scoring == "multiplicative":
                b = base_s * effective_match
                if has_extremum:
                    b *= 1.0 + self.recency_alpha * r_v
            else:  # hybrid_mm_ar
                b = base_s * effective_match
                if has_extremum:
                    b += self.recency_weight * r_v
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
    def _effective_ivs(self, did: str) -> list[Interval]:
        """Return the intervals to use for filter/mask evaluation.

        Per `ref_time_fallback`:
        - None: extracted envelopes only (the default).
        - "when_empty": fall back to a point-envelope at ref_time only
          when the doc has no extracted envelopes.
        - "always": OR-merge a point-envelope at ref_time with extracted
          envelopes.
        """
        extracted = self._doc_ivs.get(did, [])
        if self.ref_time_fallback is None:
            return extracted
        if self.ref_time_fallback == "when_empty" and extracted:
            return extracted
        ref_us = self._doc_ref_us.get(did)
        if ref_us is None:
            return extracted
        synth = Interval(earliest_us=ref_us, latest_us=ref_us)
        return [*extracted, synth]

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
        is_tree: bool = False,
    ) -> dict:
        """Resolve each planner leaf to a list of anchor intervals.

        For DNF: leaves_flat entries are (ci, li, leaf); key is (ci, li).
        For tree: leaves_flat entries are (lid, leaf); key is lid.
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

        if is_tree:
            envs_list = await asyncio.gather(
                *(_extract_phrase(leaf.phrase) for (_, leaf) in leaves_flat)
            )
            out: dict = {}
            for (lid, _), envs in zip(leaves_flat, envs_list, strict=False):
                ivs = flatten_intervals(envs)
                if ivs:
                    out[lid] = ivs
            return out

        envs_list = await asyncio.gather(
            *(_extract_phrase(leaf.phrase) for (_, _, leaf) in leaves_flat)
        )
        out2: dict[tuple[int, int], list[Interval]] = {}
        for (ci, li, _), envs in zip(leaves_flat, envs_list, strict=False):
            ivs = flatten_intervals(envs)
            if ivs:
                out2[(ci, li)] = ivs
        return out2

    def _compute_recency(
        self,
        plan,
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

    def doc_envelopes(self) -> dict[str, list[TimeEnvelope]]:
        """Return the extracted envelopes per indexed doc."""
        return dict(self._doc_envelopes)


# ---------------------------------------------------------------------------
# Tree filter pushdown — collect leaves that can be filter-pushed
# ---------------------------------------------------------------------------
def _tree_filter_leaves(node):
    """Return (includes, excludes) lists of Leaf nodes that can safely be
    pushed to the doc-level filter without over-pruning the OR side of
    the tree.

    Rule: only push leaves that are reachable via a chain of `and` nodes
    from the root (or are the root themselves). Leaves below `or` or
    `not` are evaluated only at scoring time.

    `disjoint` leaves become excludes; everything else becomes includes.
    """
    from .planner_tree import And, Leaf, Not, Or  # noqa: F401

    includes = []
    excludes = []
    if node is None:
        return includes, excludes
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Leaf):
            if n.relation == "disjoint":
                excludes.append(n)
            else:
                includes.append(n)
        elif isinstance(n, And):
            stack.extend(n.children)
        # Stop at Or / Not — don't push those leaves to filter.
    return includes, excludes
