"""Novelty-gated adaptive ensemble over per-specialist cached outputs.

Given cached specialist outputs (run_specialists_cached), call specialists
sequentially in a fixed order. After each specialist beyond the first,
measure "novelty" — the fraction of the specialist's top-K segments that are
NOT already present in the accumulated retrieved set. If novelty is at or
below a threshold tau, stop early; else merge the specialist's segments into
the accumulator via sum_cosine and continue to the next specialist.

At the end, return top-K by accumulated sum_cosine scores, with fair-backfill
from the raw-query cosine top-K if under-budget.

Metrics recorded per (question, variant):
  - n_specialists_called   (integer in [1, len(order)])
  - specialists_called     (list of names)
  - final_recall @ K
  - novelty per step       (list of floats, one per specialist after v2f)

This module depends only on `ensemble_retrieval` for cached specialist
outputs; it does not mutate any framework files or underlying specialist
sources.
"""

from __future__ import annotations

from dataclasses import dataclass

from associative_recall import Segment
from ensemble_retrieval import (
    SpecialistOutput,
    fair_backfill,
)

# Ordered specialist list for adaptive gating (v2f must be first).
ADAPTIVE_ORDER: tuple[str, ...] = (
    "v2f",
    "type_enumerated",
    "chain_with_scratchpad",
    "v2f_plus_types",
    "v2f_style_explicit",
)

# Cost multipliers mirror ensemble_eval.SPECIALIST_COST.
SPECIALIST_COST: dict[str, float] = {
    "v2f": 1.0,
    "v2f_plus_types": 2.0,
    "type_enumerated": 1.0,
    "chain_with_scratchpad": 5.0,
    "v2f_style_explicit": 1.0,
}


@dataclass
class AdaptiveResult:
    segments: list[Segment]
    specialists_called: list[str]
    novelty_per_step: list[float]  # length = len(specialists_called) - 1
    llm_cost: float


def _top_k_turn_ids(segments: list[Segment], budget: int) -> set[int]:
    """Unique turn_ids in the top-K (by position) of a specialist's output."""
    seen: set[int] = set()
    tids: set[int] = set()
    for s in segments:
        if s.index in seen:
            continue
        seen.add(s.index)
        tids.add(s.turn_id)
        if len(seen) >= budget:
            break
    return tids


def adaptive_ensemble(
    ensemble_outputs: dict[str, SpecialistOutput],
    cosine_segments: list[Segment],
    budget: int,
    tau: float,
    order: tuple[str, ...] = ADAPTIVE_ORDER,
) -> AdaptiveResult:
    """Run novelty-gated adaptive ensemble.

    Accumulates retrieved segments per-specialist. Novelty is measured as the
    fraction of the incoming specialist's top-K turn_ids that are new (not
    already in the accumulated turn_id set).

    Merging: sum_cosine over all segments from specialists that have been
    called (merged) so far.

    Params
    ------
    ensemble_outputs: dict of name -> SpecialistOutput for all specialists
        in `order`.
    cosine_segments: raw-query cosine top-K segments for fair-backfill.
    budget: K.
    tau: novelty threshold in [0.0, 1.0]. If novelty <= tau, stop.
    order: specialist call order. `order[0]` is always run.

    Returns
    -------
    AdaptiveResult with final top-K segments after fair-backfill.
    """
    assert order, "specialist order is empty"
    called: list[str] = []
    novelty_per_step: list[float] = []
    # Accumulated pool: seg_idx -> (segment, cos_sum across called specialists)
    pool: dict[int, tuple[Segment, float]] = {}
    accumulated_turn_ids: set[int] = set()
    llm_cost: float = 0.0

    def merge_specialist(name: str) -> None:
        """Merge all of specialist `name`'s segments into pool via sum_cosine."""
        so = ensemble_outputs[name]
        for seg, cos in zip(so.segments, so.cosine_scores):
            if seg.index in pool:
                prev_seg, prev_score = pool[seg.index]
                pool[seg.index] = (prev_seg, prev_score + cos)
            else:
                pool[seg.index] = (seg, cos)
            accumulated_turn_ids.add(seg.turn_id)

    # Always call the first specialist.
    first = order[0]
    merge_specialist(first)
    called.append(first)
    llm_cost += SPECIALIST_COST[first]

    # Sequentially consider subsequent specialists.
    for name in order[1:]:
        # Compute novelty of this specialist's top-K vs accumulated turn_ids.
        incoming_tids = _top_k_turn_ids(ensemble_outputs[name].segments, budget)
        if not incoming_tids:
            novelty = 0.0
        else:
            new_tids = incoming_tids - accumulated_turn_ids
            # Spec: |R_s \ R_acc| / K  (use K, not len(R_s), so specialists
            # that return <K items are penalized — consistent with spec).
            novelty = len(new_tids) / float(budget)

        novelty_per_step.append(novelty)

        if novelty > tau:
            # Merge this specialist in and continue.
            merge_specialist(name)
            called.append(name)
            llm_cost += SPECIALIST_COST[name]
        else:
            # Stop. Do not call further specialists.
            break

    # Rank accumulated pool by sum_cosine score.
    ranked = sorted(pool.values(), key=lambda sc: -sc[1])
    segments = fair_backfill(ranked, cosine_segments, budget)

    return AdaptiveResult(
        segments=segments,
        specialists_called=called,
        novelty_per_step=novelty_per_step,
        llm_cost=llm_cost,
    )
