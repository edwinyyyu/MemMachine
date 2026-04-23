"""Ensemble retrieval over per-specialist cached outputs.

Given cached LLM + embedding caches for multiple retrieval specialists, run
each specialist on a question (cache-only — no new LLM calls), collect each
specialist's ordered retrieved segments, attach a cosine score against the
raw query embedding for each retrieved segment, and merge into a single
ranked list using one of several merging strategies. Then fair-backfill
(with cosine top-K on the raw query) to reach budget K.

This module defines:
  - `SpecialistOutput`: (segments, scores, rank) per specialist per question.
  - `run_specialists_cached`: collect outputs for all 5 specialists.
  - `merge_*`: 4 merging strategies.
  - `ensemble_at_k`: run one ensemble composition × merging strategy for a
    question and return the final top-K set of turn_ids after fair-backfill.

No LLM calls are made outside of the cache-only retrieval step. If a cache
miss occurs, the underlying specialist's LLM proxy emits a DONE response
(which results in no new cues), matching `specialist_complementarity.py`.

This module does not touch framework files or other specialists' sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from associative_recall import Segment, SegmentStore
from best_shot import MetaV2f
from domain_agnostic import (
    DomainAgnosticVariant,
    V2F_STYLE_EXPLICIT_PROMPT,
    NEUTRAL_HEADER,
)
from goal_chain import GoalChainRetriever
from type_enumerated import TypeEnumeratedVariant, V2fPlusTypesVariant


SPECIALISTS = (
    "v2f",
    "v2f_plus_types",
    "type_enumerated",
    "chain_with_scratchpad",
    "v2f_style_explicit",
)


ENSEMBLE_COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "ens_2_v2f_v2fplus": ("v2f", "v2f_plus_types"),
    "ens_2_v2f_typeenum": ("v2f", "type_enumerated"),
    "ens_3": ("v2f", "v2f_plus_types", "type_enumerated"),
    "ens_5": (
        "v2f",
        "v2f_plus_types",
        "type_enumerated",
        "chain_with_scratchpad",
        "v2f_style_explicit",
    ),
}


MERGING_STRATEGIES = ("max_cosine", "sum_cosine", "rrf", "round_robin")


RRF_K = 60  # standard reciprocal-rank-fusion constant


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class SpecialistOutput:
    name: str
    segments: list[Segment]           # ordered, deduped by index
    cosine_scores: list[float]        # per-segment cosine vs raw query emb
    llm_calls: int


# ---------------------------------------------------------------------------
# Cache-only specialist builder
# ---------------------------------------------------------------------------
def build_specialist(name: str, store: SegmentStore):
    if name == "v2f":
        arch = MetaV2f(store)
    elif name == "v2f_plus_types":
        arch = V2fPlusTypesVariant(store)
    elif name == "type_enumerated":
        arch = TypeEnumeratedVariant(store)
    elif name == "chain_with_scratchpad":
        arch = GoalChainRetriever(store, use_scratchpad=True)
    elif name == "v2f_style_explicit":
        arch = DomainAgnosticVariant(
            store,
            prompt_template=V2F_STYLE_EXPLICIT_PROMPT,
            context_header=NEUTRAL_HEADER,
        )
    else:
        raise KeyError(name)

    def cached_only(prompt: str, model: str = "gpt-5-mini") -> str:
        cached = arch.llm_cache.get(model, prompt)
        if cached is not None:
            arch.llm_calls += 1
            return cached
        arch.llm_calls += 1
        return "ACTION: DONE\nREASONING: cache-miss; skipping\n"

    arch.llm_call = cached_only
    return arch


def _dedupe_preserve_order(segments: Iterable[Segment]) -> list[Segment]:
    seen: set[int] = set()
    out: list[Segment] = []
    for s in segments:
        if s.index not in seen:
            seen.add(s.index)
            out.append(s)
    return out


def _attach_cosine_scores(
    store: SegmentStore,
    segments: list[Segment],
    query_emb: np.ndarray,
) -> list[float]:
    if not segments:
        return []
    qn = query_emb / max(float(np.linalg.norm(query_emb)), 1e-10)
    idxs = np.array([s.index for s in segments], dtype=np.int64)
    seg_embs = store.normalized_embeddings[idxs]
    sims = seg_embs @ qn
    return [float(x) for x in sims.tolist()]


def run_specialists_cached(
    specialists: dict,
    store: SegmentStore,
    question: str,
    conversation_id: str,
    query_emb: np.ndarray,
) -> dict[str, SpecialistOutput]:
    """Run each specialist on the question and collect outputs.

    Each specialist's segments are deduped preserving order, and cosine scores
    are computed vs the raw query embedding.
    """
    out: dict[str, SpecialistOutput] = {}
    for name, arch in specialists.items():
        arch.reset_counters()
        res = arch.retrieve(question, conversation_id)
        segs = _dedupe_preserve_order(res.segments)
        scores = _attach_cosine_scores(store, segs, query_emb)
        out[name] = SpecialistOutput(
            name=name,
            segments=segs,
            cosine_scores=scores,
            llm_calls=arch.llm_calls,
        )
    return out


# ---------------------------------------------------------------------------
# Merging strategies
# ---------------------------------------------------------------------------
def _gather_pools(
    ensemble_outputs: dict[str, SpecialistOutput],
) -> dict[int, dict]:
    """Return {seg_index: {segment, per_specialist: {name: (rank, cosine)}}}"""
    pool: dict[int, dict] = {}
    for name, so in ensemble_outputs.items():
        for rank, (seg, cos) in enumerate(zip(so.segments, so.cosine_scores)):
            entry = pool.setdefault(seg.index, {
                "segment": seg,
                "per_specialist": {},
            })
            # Keep the first (best) rank for this specialist if duplicated,
            # though _dedupe_preserve_order already handled that per-spec.
            entry["per_specialist"][name] = (rank, cos)
    return pool


def merge_max_cosine(
    ensemble_outputs: dict[str, SpecialistOutput],
) -> list[tuple[Segment, float]]:
    pool = _gather_pools(ensemble_outputs)
    ranked = []
    for idx, entry in pool.items():
        cos_max = max(c for _, c in entry["per_specialist"].values())
        ranked.append((entry["segment"], cos_max))
    ranked.sort(key=lambda rc: -rc[1])
    return ranked


def merge_sum_cosine(
    ensemble_outputs: dict[str, SpecialistOutput],
) -> list[tuple[Segment, float]]:
    pool = _gather_pools(ensemble_outputs)
    ranked = []
    for idx, entry in pool.items():
        cos_sum = sum(c for _, c in entry["per_specialist"].values())
        ranked.append((entry["segment"], cos_sum))
    ranked.sort(key=lambda rc: -rc[1])
    return ranked


def merge_rrf(
    ensemble_outputs: dict[str, SpecialistOutput],
    rrf_k: int = RRF_K,
) -> list[tuple[Segment, float]]:
    pool = _gather_pools(ensemble_outputs)
    ranked = []
    for idx, entry in pool.items():
        rrf_score = sum(
            1.0 / (r + rrf_k + 1)  # r is 0-indexed; use (rank+1) in denom
            for r, _ in entry["per_specialist"].values()
        )
        ranked.append((entry["segment"], rrf_score))
    ranked.sort(key=lambda rc: -rc[1])
    return ranked


def merge_round_robin(
    ensemble_outputs: dict[str, SpecialistOutput],
    specialist_order: tuple[str, ...] | None = None,
) -> list[tuple[Segment, float]]:
    """Interleave specialists' top picks one at a time, skipping duplicates.

    Score assigned is the inverse position in the fused list (higher is
    better) — used only for tie-breaking / reporting, not for ordering.
    """
    names = list(specialist_order) if specialist_order else list(
        ensemble_outputs.keys()
    )
    # Per-specialist pointer
    pointers = {n: 0 for n in names}
    chosen_indices: set[int] = set()
    fused: list[Segment] = []
    total = sum(len(ensemble_outputs[n].segments) for n in names)
    while len(fused) < total:
        made_progress = False
        for n in names:
            segs = ensemble_outputs[n].segments
            while pointers[n] < len(segs) and segs[pointers[n]].index in chosen_indices:
                pointers[n] += 1
            if pointers[n] < len(segs):
                seg = segs[pointers[n]]
                fused.append(seg)
                chosen_indices.add(seg.index)
                pointers[n] += 1
                made_progress = True
        if not made_progress:
            break
    # Score: higher for earlier position
    ranked = [(s, float(len(fused) - i)) for i, s in enumerate(fused)]
    return ranked


MERGE_FUNCS = {
    "max_cosine": merge_max_cosine,
    "sum_cosine": merge_sum_cosine,
    "rrf": merge_rrf,
    "round_robin": merge_round_robin,
}


# ---------------------------------------------------------------------------
# Fair-backfill utility (applied AFTER merging)
# ---------------------------------------------------------------------------
def fair_backfill(
    ensemble_ranked: list[tuple[Segment, float]],
    cosine_segments: list[Segment],
    budget: int,
) -> list[Segment]:
    """Truncate ranked ensemble to K; if fewer than K unique segments are
    available, fill the rest from cosine top-K of the raw query (skipping
    duplicates)."""
    seen: set[int] = set()
    out: list[Segment] = []
    for seg, _ in ensemble_ranked:
        if seg.index in seen:
            continue
        out.append(seg)
        seen.add(seg.index)
        if len(out) >= budget:
            break
    if len(out) < budget:
        for seg in cosine_segments:
            if seg.index in seen:
                continue
            out.append(seg)
            seen.add(seg.index)
            if len(out) >= budget:
                break
    return out[:budget]


def ensemble_at_k(
    ensemble_outputs: dict[str, SpecialistOutput],
    specialist_names: tuple[str, ...],
    merging_strategy: str,
    cosine_segments: list[Segment],
    budget: int,
) -> list[Segment]:
    """Select specialists, merge with the strategy, fair-backfill, truncate."""
    sub = {n: ensemble_outputs[n] for n in specialist_names if n in ensemble_outputs}
    merge_fn = MERGE_FUNCS[merging_strategy]
    if merging_strategy == "round_robin":
        ranked = merge_fn(sub, specialist_order=specialist_names)
    else:
        ranked = merge_fn(sub)
    return fair_backfill(ranked, cosine_segments, budget)
