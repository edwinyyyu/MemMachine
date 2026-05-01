"""Bidirectional lattice lookup + scoring.

Given a query ``TimeExpression``:

1. Tag at native precision + cyclical (same rules as ingest).
2. For each absolute tag, expand
      UP to ancestor cells (containment parents)
      DOWN one level to direct-children cells
3. Keep cyclical tags verbatim (lateral match).
4. SQL lookup each expanded cell in the LatticeStore.
5. Score each candidate doc by its best matching tag:

       cell_score(tag) = 1 / log2(2 + tag_span / query_span)
       direction_bonus = 1.0 if doc_tag.precision is finer-or-equal to
                         query_tag.precision else 0.5
       cand_score = 0.4 * cell_score + 0.2 * direction_bonus

   We take the max over shared tags per candidate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from lattice_cells import (
    ABSOLUTE_AXIS_SPAN_DAYS,
    CYCLICAL_AXES,
    LatticeTagSet,
    ancestors_of_absolute,
    children_of_absolute,
    precision_of_tag,
    span_days_of_tag,
    tags_for_expression,
)
from lattice_store import LatticeStore
from schema import TimeExpression

# Ordering for the absolute lattice, finest -> coarsest
_ABS_ORDER_INDEX = {
    "minute": 0,
    "hour": 1,
    "day": 2,
    "week": 3,
    "month": 4,
    "quarter": 5,
    "year": 6,
    "decade": 7,
    "century": 8,
}


def _is_finer_or_equal(doc_prec: str, query_prec: str) -> bool:
    if doc_prec in CYCLICAL_AXES or query_prec in CYCLICAL_AXES:
        return True
    return _ABS_ORDER_INDEX.get(doc_prec, 9) <= _ABS_ORDER_INDEX.get(query_prec, 9)


def expand_query_tags(
    tagset: LatticeTagSet,
    down_levels: int = 1,
) -> dict[str, dict]:
    """Expand the query tagset into a candidate set of cell tags.

    Returns a dict {expanded_tag -> {"origin_tag": native_tag, "direction":
    "up"|"down"|"self"|"cyclical", "origin_prec": prec}} used later for
    scoring.
    """
    expanded: dict[str, dict] = {}

    for prec, tag in tagset.absolute:
        # self
        expanded[tag] = {
            "origin_tag": tag,
            "origin_prec": prec,
            "direction": "self",
        }
        # up
        for anc in ancestors_of_absolute(tag):
            # Prefer first-seen origin
            expanded.setdefault(
                anc,
                {
                    "origin_tag": tag,
                    "origin_prec": prec,
                    "direction": "up",
                },
            )
        # down — BFS across `down_levels` steps
        frontier = [tag]
        for _ in range(down_levels):
            next_frontier: list[str] = []
            for t in frontier:
                for child in children_of_absolute(t):
                    expanded.setdefault(
                        child,
                        {
                            "origin_tag": tag,
                            "origin_prec": prec,
                            "direction": "down",
                        },
                    )
                    next_frontier.append(child)
            frontier = next_frontier

    for tag in tagset.cyclical:
        prec = precision_of_tag(tag)
        expanded[tag] = {
            "origin_tag": tag,
            "origin_prec": prec,
            "direction": "cyclical",
        }

    return expanded


def score_candidates(
    query_tags: LatticeTagSet,
    matched_per_doc: dict[str, set[str]],
    expanded: dict[str, dict],
) -> dict[str, float]:
    """Score each doc by its best matching tag.

    cell_score(tag) = 1 / log2(2 + tag_span / query_span)
    direction_bonus = 1.0 if doc_tag.precision finer-or-equal to
                      query_tag.precision else 0.5

    cand_score = 0.4 * cell_score + 0.2 * direction_bonus

    max over shared tags per doc.
    """
    # Query native span (use finest absolute precision; fallback to 1 day for
    # pure-cyclical queries).
    q_abs_prec_idx = min(
        (_ABS_ORDER_INDEX.get(prec, 9) for prec, _t in query_tags.absolute),
        default=2,
    )
    q_span_days = min(
        (ABSOLUTE_AXIS_SPAN_DAYS.get(prec, 365.0) for prec, _t in query_tags.absolute),
        default=1.0,
    )

    # Cyclical-channel weight factor. Cyclical tags repeat indefinitely
    # in absolute time (e.g. weekday:Thursday matches ~52 Thursdays/year),
    # so they are weaker evidence than an absolute cell match. We down-
    # weight cyclical contributions so an absolute match (even a coarse
    # one) out-ranks a pure cyclical match.
    CYCLICAL_WEIGHT = 0.6
    ABSOLUTE_WEIGHT = 1.0

    scores: dict[str, float] = {}
    for doc_id, matched_tags in matched_per_doc.items():
        best = 0.0
        for mt in matched_tags:
            doc_prec = precision_of_tag(mt)
            origin = expanded.get(mt)
            if origin is None:
                continue
            if origin["direction"] == "cyclical":
                # Cyclical: narrow on cyclical axis but wide in absolute
                # time. Use a fixed cell_score and down-weight.
                cell_score = 0.5
                dir_bonus = 1.0
                channel_w = CYCLICAL_WEIGHT
            else:
                tag_span = span_days_of_tag(mt)
                ratio = tag_span / max(q_span_days, 1e-6)
                cell_score = 1.0 / math.log2(2 + ratio)
                if _is_finer_or_equal(doc_prec, origin["origin_prec"]):
                    dir_bonus = 1.0
                else:
                    dir_bonus = 0.5
                channel_w = ABSOLUTE_WEIGHT
            s = channel_w * (0.4 * cell_score + 0.2 * dir_bonus)
            if s > best:
                best = s
        scores[doc_id] = best
    return scores


def retrieve(
    store: LatticeStore,
    query_te: TimeExpression,
    *,
    down_levels: int = 1,
) -> tuple[dict[str, float], dict]:
    """Retrieve candidates for a query TimeExpression.

    Returns (scores, debug_info) where scores is doc_id -> float and
    debug_info contains the expanded cell set and match counts.
    """
    qtags = tags_for_expression(query_te)
    expanded = expand_query_tags(qtags, down_levels=down_levels)
    matched = store.query_by_tags(expanded.keys())
    scores = score_candidates(qtags, matched, expanded)
    debug = {
        "query_native_abs_tags": [t for _p, t in qtags.absolute],
        "query_cyclical_tags": sorted(qtags.cyclical),
        "expanded_count": len(expanded),
        "n_docs_matched": len(matched),
    }
    return scores, debug


def retrieve_multi(
    store: LatticeStore,
    query_tes: Iterable[TimeExpression],
    *,
    down_levels: int = 1,
) -> tuple[dict[str, float], dict]:
    """Retrieve candidates over multiple query TEs by taking max-per-doc
    across the TEs (union with max-merge)."""
    combined: dict[str, float] = {}
    all_native_abs: list[str] = []
    all_cyclical: set[str] = set()
    total_expanded = 0
    total_matched = 0
    for te in query_tes:
        qtags = tags_for_expression(te)
        expanded = expand_query_tags(qtags, down_levels=down_levels)
        matched = store.query_by_tags(expanded.keys())
        scores = score_candidates(qtags, matched, expanded)
        for d, s in scores.items():
            if s > combined.get(d, 0.0):
                combined[d] = s
        all_native_abs.extend([t for _p, t in qtags.absolute])
        all_cyclical |= qtags.cyclical
        total_expanded += len(expanded)
        total_matched += len(matched)
    debug = {
        "query_native_abs_tags": all_native_abs,
        "query_cyclical_tags": sorted(all_cyclical),
        "expanded_count": total_expanded,
        "n_docs_matched_total": total_matched,
    }
    return combined, debug
