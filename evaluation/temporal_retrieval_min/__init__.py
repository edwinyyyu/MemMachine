"""Temporal-aware retrieval as a layer on top of any RAG system.

The pipeline:
- Single-pass LLM extractor produces an `Interval` per temporal
  reference in a doc.
- Algorithm primitives (mask, filter, hybrid pool, recency, norm) live
  in `core.py` and operate on microsecond-precision `Interval` objects.
- `TemporalRetriever` orchestrates the LLM stages (DNF planner,
  extractor) and the post-LLM scoring pipeline.

Embed and rerank functions are injected — bring your own.
"""

from .core import (
    Interval,
    build_pool,
    constraint_factor_for_doc,
    doc_passes_filter,
    excluded_containment,
    normalize_dict,
    normalize_rerank_full,
    rank_from_scores,
    recency_scores,
)
from .planner import Constraint, QueryPlan, evaluate_dnf_match
from .retriever import Doc, Result, TemporalRetriever

__all__ = [
    "Constraint",
    "Doc",
    "Interval",
    "QueryPlan",
    "Result",
    "TemporalRetriever",
    "build_pool",
    "constraint_factor_for_doc",
    "doc_passes_filter",
    "evaluate_dnf_match",
    "excluded_containment",
    "normalize_dict",
    "normalize_rerank_full",
    "rank_from_scores",
    "recency_scores",
]
