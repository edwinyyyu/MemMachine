"""Temporal Retrieval V7 — TimeRange-based principled rewrite.

See SPEC.md for the architectural specification.
"""
from __future__ import annotations

from .time_range import (
    NEG_INF,
    POS_INF,
    SENTINEL_THRESHOLD,
    Interval,
    TimeRange,
    complement,
    difference,
    intersect,
    intersect_all,
    is_empty,
    is_infinite_measure,
    measure,
    symmetric_difference,
    union,
    union_all,
)
from .scoring import (
    best_per_ref,
    final_score,
    pair_overlap,
    temporal_pass,
)
from .planner_direct import DirectPlan, DirectQueryPlanner
from .retriever import Doc, Result, TemporalRetrieverV7, TemporalRetrieverV7Direct

__all__ = [
    "DirectPlan",
    "DirectQueryPlanner",
    "TemporalRetrieverV7Direct",
    "Doc",
    "Interval",
    "NEG_INF",
    "POS_INF",
    "Result",
    "SENTINEL_THRESHOLD",
    "TemporalRetrieverV7",
    "TimeRange",
    "best_per_ref",
    "complement",
    "difference",
    "final_score",
    "intersect",
    "intersect_all",
    "is_empty",
    "is_infinite_measure",
    "measure",
    "pair_overlap",
    "symmetric_difference",
    "temporal_pass",
    "union",
    "union_all",
]
