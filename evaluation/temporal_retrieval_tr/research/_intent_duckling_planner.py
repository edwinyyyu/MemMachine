"""Duckling planner + minimal regex intent layer.

Adds three lexical-intent rules on top of Duckling's date extraction:
  1. Extremum keywords  -> set extremum="latest"/"earliest"
  2. Negation keywords  -> emit complement of extracted intervals
  3. Default            -> Duckling targets as-is

Determines whether a small rule-intent layer can recover the bulk of
the LLM-planner's advantage.
"""
from __future__ import annotations

import re
from datetime import datetime

from temporal_retrieval_min.schema import parse_iso
from temporal_retrieval_tr.planner import Plan
from temporal_retrieval_tr.time_range import (
    NEG_INF, POS_INF, Interval, IntervalSet,
)

from ._duckling_extractor import DucklingHTTPExtractor


_LATEST_RE = re.compile(
    r"\b(most\s+recent(?:ly)?|recently|latest|just\s+(?:now|had|did)|"
    r"last\s+time|previously|the\s+last)\b",
    re.IGNORECASE,
)
_EARLIEST_RE = re.compile(
    r"\b(first|earliest|originally|initially|the\s+first\s+time)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(not\s+in|outside(?:\s+of)?|except(?:ing|\s+for)?|excluding|"
    r"other\s+than|aside\s+from|but\s+not)\b",
    re.IGNORECASE,
)


def _complement_of(intervals: list[Interval]) -> list[Interval]:
    """Complement of the UNION of intervals over the time line.

    [-inf, +inf) minus the union of given intervals. Returns a sorted
    list of disjoint intervals covering everything else.
    """
    if not intervals:
        return [Interval(NEG_INF, POS_INF)]
    # Sort by start; we assume well-formed (no inversion).
    sorted_ivs = sorted(intervals, key=lambda iv: iv.earliest_us)
    # Merge overlapping/adjacent.
    merged: list[Interval] = [sorted_ivs[0]]
    for iv in sorted_ivs[1:]:
        prev = merged[-1]
        if iv.earliest_us <= prev.latest_us:
            new_end = (
                iv.latest_us
                if iv.latest_us > prev.latest_us
                else prev.latest_us
            )
            merged[-1] = Interval(prev.earliest_us, new_end)
        else:
            merged.append(iv)
    # Walk and emit gaps.
    result: list[Interval] = []
    prev_end = NEG_INF
    for iv in merged:
        if prev_end < iv.earliest_us:
            result.append(Interval(prev_end, iv.earliest_us))
        prev_end = iv.latest_us
    if prev_end < POS_INF:
        result.append(Interval(prev_end, POS_INF))
    return result


def _detect_extremum(query: str) -> str | None:
    if _LATEST_RE.search(query):
        return "latest"
    if _EARLIEST_RE.search(query):
        return "earliest"
    return None


class IntentDucklingPlanner:
    """Duckling planner enhanced with regex extremum/negation detection."""

    def __init__(self) -> None:
        self._extractor = DucklingHTTPExtractor()
        self._cache: dict[str, Plan] = {}

    def save_caches(self) -> None:
        self._extractor.save_caches()

    async def plan(self, query: str, ref_time: str) -> Plan:
        key = f"{query}|{ref_time}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            rt = parse_iso(ref_time)
        except Exception as e:
            return Plan(parse_error=str(e))

        entities = await self._extractor.extract_anchors(query, rt)
        extremum = _detect_extremum(query)
        has_negation = bool(_NEGATION_RE.search(query))

        # Default: each entity becomes its own target.
        targets = list(entities)

        # Negation: complement of the union of extracted intervals,
        # emitted as ONE multi-interval target (set membership over the
        # complement region).
        if has_negation and entities:
            all_ivs = [iv for ent in entities for iv in ent.intervals]
            comp = _complement_of(all_ivs)
            if comp:
                targets = [IntervalSet.from_intervals(comp)]

        plan = Plan(targets=targets, extremum=extremum)
        self._cache[key] = plan
        return plan
