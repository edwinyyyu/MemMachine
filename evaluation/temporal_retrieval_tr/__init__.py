"""Temporal Retrieval — IntervalSet-based principled implementation."""
from __future__ import annotations

from .time_range import (
    NEG_INF,
    POS_INF,
    Interval,
    IntervalSet,
    Endpoint,
    complement,
    difference,
    intersect,
    intersect_all,
    is_empty,
    is_inf,
    is_infinite_measure,
    measure,
    symmetric_difference,
    union,
    union_all,
)
from .scoring import (
    best_per_target,
    final_score,
    pair_overlap,
    temporal_pass,
)
from .planner import AnaphoricTarget, Plan, QueryPlanner
from .retriever import Doc, Result, TemporalRetriever

__all__ = [
    "AnaphoricTarget",
    "Doc",
    "Interval",
    "IntervalSet",
    "NEG_INF",
    "POS_INF",
    "Plan",
    "QueryPlanner",
    "Result",
    "TemporalRetriever",
    "Endpoint",
    "best_per_target",
    "complement",
    "difference",
    "final_score",
    "intersect",
    "intersect_all",
    "is_empty",
    "is_inf",
    "is_infinite_measure",
    "measure",
    "pair_overlap",
    "symmetric_difference",
    "temporal_pass",
    "union",
    "union_all",
]
