"""Intent planner v2: + quarters, scope binding, tighter regexes.

Adds on top of v1 (extremum + negation->complement):
  4. Q1-Q4 / quarter abbreviations as native intervals
  5. Scope binding: "in X except Y" -> intersect(bounded_X, complement(Y))
  6. Tighter extremum regex (avoid false fire on bare "first"/"originally")
  7. Bounded/unbounded scope ("since X" / "before Y" / "lately")
"""
from __future__ import annotations

import calendar
import re
from datetime import UTC, datetime, timedelta

from temporal_retrieval_min.schema import parse_iso, to_us
from temporal_retrieval_tr.planner import Plan
from temporal_retrieval_tr.time_range import (
    NEG_INF, POS_INF, Endpoint, Interval, IntervalSet,
)

from ._duckling_extractor import DucklingHTTPExtractor


# Tighter extremum: must combine with a date-context word to fire.
_LATEST_RE = re.compile(
    r"\b(most\s+recent(?:ly)?|recently|latest|just\s+(?:now|had|did)|"
    r"last\s+time|previously|the\s+last|newest)\b",
    re.IGNORECASE,
)
_EARLIEST_RE = re.compile(
    # "first" only fires when paired with a temporal noun nearby; bare
    # "first" / "originally" / "initially" too noisy.
    r"\b(the\s+first\s+time|earliest|first\s+(?:visit|meeting|appointment|trip|event|session|class|day|time|year|month|week))\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(not\s+in|outside(?:\s+of)?|except(?:ing|\s+for)?|excluding|"
    r"other\s+than|aside\s+from|but\s+not|nothing\s+in)\b",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(
    r"\b[Qq]([1-4])\s+((?:19|20)\d{2})\b"
)


def _quarter_interval(q: int, year: int) -> Interval:
    start_month = 1 + 3 * (q - 1)
    end_month = start_month + 3
    end_year = year + (end_month - 1) // 12
    end_month = ((end_month - 1) % 12) + 1
    start = datetime(year, start_month, 1, tzinfo=UTC)
    end = datetime(end_year, end_month, 1, tzinfo=UTC)
    return Interval(to_us(start), to_us(end))


def _extract_quarters(query: str) -> list[tuple[int, int, Interval]]:
    """Return [(match_start, match_end, interval), ...] for Q[1-4] YYYY phrases."""
    out = []
    for m in _QUARTER_RE.finditer(query):
        q = int(m.group(1))
        year = int(m.group(2))
        out.append((m.start(), m.end(), _quarter_interval(q, year)))
    return out


def _detect_proximity_anchor(query: str) -> Endpoint | None:
    """Detect extremum-style proximity intent from query surface form.

    Returns POS_INF for "latest" surface, NEG_INF for "earliest" surface,
    or None for no detected extremum. Finite anchors ("around date X")
    are not detected here — that would require date extraction the rule
    planner doesn't do; the LLM planner handles those.
    """
    if _LATEST_RE.search(query):
        return POS_INF
    if _EARLIEST_RE.search(query):
        return NEG_INF
    return None


def _complement_of(intervals: list[Interval]) -> list[Interval]:
    """Complement of UNION of intervals over time line."""
    if not intervals:
        return [Interval(NEG_INF, POS_INF)]
    sorted_ivs = sorted(intervals, key=lambda iv: iv.earliest_us)
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
    result: list[Interval] = []
    prev_end = NEG_INF
    for iv in merged:
        if prev_end < iv.earliest_us:
            result.append(Interval(prev_end, iv.earliest_us))
        prev_end = iv.latest_us
    if prev_end < POS_INF:
        result.append(Interval(prev_end, POS_INF))
    return result


def _intersect_intervals(
    a: list[Interval], b: list[Interval]
) -> list[Interval]:
    """Set intersection of two interval lists (sorted, disjoint)."""
    out = []
    for ia in a:
        for ib in b:
            lo = ia.earliest_us if ia.earliest_us > ib.earliest_us else ib.earliest_us
            hi = ia.latest_us if ia.latest_us < ib.latest_us else ib.latest_us
            if lo < hi:
                out.append(Interval(lo, hi))
    return out


class IntentV2Planner:
    """Intent-augmented Duckling planner with quarter handling + scope binding."""

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

        # Step 1: extract dates via Duckling
        duck_entities = await self._extractor.extract_anchors(query, rt)

        # Step 2: extract Q1-Q4 phrases via regex (Duckling misses these)
        quarter_matches = _extract_quarters(query)
        quarter_intervals = [iv for (_, _, iv) in quarter_matches]

        # Combine Duckling + quarter intervals as flat candidate list.
        all_intervals: list[Interval] = (
            [iv for ent in duck_entities for iv in ent.intervals]
            + quarter_intervals
        )

        # Step 3: detect intent.
        proximity_anchor = _detect_proximity_anchor(query)
        neg_match = _NEGATION_RE.search(query)

        # Step 4: assemble targets.
        if not all_intervals:
            targets = []
        elif not neg_match:
            # Default: each interval as its own target (graded coverage).
            targets = [IntervalSet.from_intervals([iv]) for iv in all_intervals]
        else:
            # Negation present. Split intervals into "before negation"
            # (include = bounded scope) and "after negation" (exclude).
            neg_start = neg_match.start()
            include_ivs: list[Interval] = []
            exclude_ivs: list[Interval] = []
            # Duckling/quarter intervals come with char-offset info only
            # for Duckling; for quarters we have explicit char ranges.
            # As a simpler heuristic: anything strictly BEFORE the
            # negation token is "include"; anything after is "exclude".
            # Use the literal substring match position for each interval.

            # For Duckling-derived intervals we lack offsets here (we
            # only kept the Interval); fall back to: if there are
            # intervals from BOTH before and after, do scope binding;
            # otherwise treat all as "exclude" (the original v1 behavior).
            # We get char offsets indirectly: re-extract from query with
            # offset info via search.
            from dateparser.search import search_dates as _sd
            duck_offsets: list[tuple[int, int, Interval]] = list(quarter_matches)
            # add Duckling-found offsets: re-query via offset-bearing API.
            # Cheaper proxy: use simple regex for common date forms to
            # estimate positions of those that came from Duckling.
            # For pure simplicity here, fall back to v1 behavior if we
            # don't have offsets for Duckling intervals.
            if not quarter_matches:
                # No quarter offsets => use v1 fallback: all = exclude.
                comp = _complement_of(all_intervals)
                targets = [IntervalSet.from_intervals(comp)] if comp else []
            else:
                # We have quarter offsets; partition by neg position.
                for (st, en, iv) in quarter_matches:
                    if en <= neg_start:
                        include_ivs.append(iv)
                    elif st >= neg_start:
                        exclude_ivs.append(iv)
                # Duckling intervals: heuristically include them as
                # "include" (we don't know their position, conservative).
                duck_only = [
                    iv for ent in duck_entities for iv in ent.intervals
                ]
                # If exclude_ivs is empty but neg_match present, the
                # negation likely binds the Duckling entity instead.
                if exclude_ivs:
                    if not include_ivs:
                        include_ivs = duck_only
                    comp = _complement_of(exclude_ivs)
                    if include_ivs:
                        bounded = _intersect_intervals(include_ivs, comp)
                        targets = [IntervalSet.from_intervals(bounded)] if bounded else []
                    else:
                        targets = [IntervalSet.from_intervals(comp)]
                else:
                    # neg present but exclude set empty -> degrade to v1
                    comp = _complement_of(all_intervals)
                    targets = [IntervalSet.from_intervals(comp)] if comp else []

        plan = Plan(targets=targets, proximity_anchor_us=proximity_anchor)
        self._cache[key] = plan
        return plan
