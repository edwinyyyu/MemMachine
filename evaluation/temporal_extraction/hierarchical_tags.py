"""Hierarchical granularity-tag generation from TimeExpression (F2).

Each extracted ``TimeExpression`` emits a set of discrete string tags spanning
every granularity it "contains":

    day:2024-03-15
    week:2024-W11
    month:2024-03
    quarter:2024-Q1
    year:2024
    decade:2020s
    century:21st

Intervals emit tag sets covering the span at each granularity (months,
quarters, years; days only if span < 100d). Decades/centuries expand to
component years/decades. Recurrences collapse to the DTSTART tag set plus
pattern markers (``pattern:weekly``, ``weekday:mon``).

A budget cap of 50 tags per expression keeps index size bounded (coarser
granularities are retained over finer when truncated).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone

from schema import (
    FuzzyInstant,
    FuzzyInterval,
    Recurrence,
    TimeExpression,
)

MAX_TAGS_PER_EXPR = 50

# Coarser-first priority so truncation keeps the broad tags.
_GRAN_PRIORITY = {
    "century": 0,
    "decade": 1,
    "year": 2,
    "quarter": 3,
    "month": 4,
    "week": 5,
    "day": 6,
    "hour": 7,
    "minute": 8,
    "second": 9,
    "pattern": 0,
    "weekday": 1,
}


def _prio_of(tag: str) -> int:
    head = tag.split(":", 1)[0]
    return _GRAN_PRIORITY.get(head, 9)


# ---------------------------------------------------------------------------
# Single-date tag helpers
# ---------------------------------------------------------------------------
def _century_tag(year: int) -> str:
    # 1900 -> 20th, 2024 -> 21st
    cent = year // 100 + 1
    suffix = "th"
    if cent % 100 in (11, 12, 13):
        suffix = "th"
    elif cent % 10 == 1:
        suffix = "st"
    elif cent % 10 == 2:
        suffix = "nd"
    elif cent % 10 == 3:
        suffix = "rd"
    return f"century:{cent}{suffix}"


def _decade_tag(year: int) -> str:
    return f"decade:{(year // 10) * 10}s"


def _year_tag(year: int) -> str:
    return f"year:{year:04d}"


def _quarter_tag(year: int, month: int) -> str:
    q = (month - 1) // 3 + 1
    return f"quarter:{year:04d}-Q{q}"


def _month_tag(year: int, month: int) -> str:
    return f"month:{year:04d}-{month:02d}"


def _week_tag(d: date) -> str:
    iso = d.isocalendar()
    return f"week:{iso[0]:04d}-W{iso[1]:02d}"


def _day_tag(d: date) -> str:
    return f"day:{d.isoformat()}"


def _date_to_all_tags(d: date) -> list[str]:
    return [
        _day_tag(d),
        _week_tag(d),
        _month_tag(d.year, d.month),
        _quarter_tag(d.year, d.month),
        _year_tag(d.year),
        _decade_tag(d.year),
        _century_tag(d.year),
    ]


# ---------------------------------------------------------------------------
# Date iteration helpers
# ---------------------------------------------------------------------------
def _iter_months(start: date, end: date) -> Iterable[tuple[int, int]]:
    y, m = start.year, start.month
    end_y, end_m = end.year, end.month
    while (y, m) <= (end_y, end_m):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def _iter_quarters(start: date, end: date) -> Iterable[tuple[int, int]]:
    sq = (start.month - 1) // 3 + 1
    eq = (end.month - 1) // 3 + 1
    y, q = start.year, sq
    end_y = end.year
    while (y, q) <= (end_y, eq):
        yield y, q
        if q == 4:
            y += 1
            q = 1
        else:
            q += 1


# ---------------------------------------------------------------------------
# Instant / interval tagging
# ---------------------------------------------------------------------------
def _dt_to_date(dt: datetime) -> date:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).date()


def tags_for_instant(fi: FuzzyInstant) -> set[str]:
    """Emit hierarchical tags for a FuzzyInstant.

    - day/hour/minute/second granularity: use the 'best' date (or earliest)
      and emit all containing granularity tags.
    - month granularity: emit the bracketing month + its quarter/year/decade/
      century (no day/week).
    - quarter granularity: quarter + year + decade + century (skip month).
    - year: year + decade + century.
    - decade: decade + each component year + century.
    - century: century + each component decade.
    - week: compute week from best; emit week + month + quarter + year + ...
    """
    gran = fi.granularity
    best = fi.best or fi.earliest
    earliest = fi.earliest
    latest = fi.latest

    if best is None:
        return set()

    tags: set[str] = set()

    if gran in ("second", "minute", "hour", "day"):
        d = _dt_to_date(best)
        tags.update(_date_to_all_tags(d))
        return tags

    if gran == "week":
        d = _dt_to_date(best)
        tags.add(_week_tag(d))
        tags.add(_month_tag(d.year, d.month))
        tags.add(_quarter_tag(d.year, d.month))
        tags.add(_year_tag(d.year))
        tags.add(_decade_tag(d.year))
        tags.add(_century_tag(d.year))
        return tags

    if gran == "month":
        d = _dt_to_date(best)
        tags.add(_month_tag(d.year, d.month))
        tags.add(_quarter_tag(d.year, d.month))
        tags.add(_year_tag(d.year))
        tags.add(_decade_tag(d.year))
        tags.add(_century_tag(d.year))
        return tags

    if gran == "quarter":
        d = _dt_to_date(best)
        tags.add(_quarter_tag(d.year, d.month))
        tags.add(_year_tag(d.year))
        tags.add(_decade_tag(d.year))
        tags.add(_century_tag(d.year))
        return tags

    if gran == "year":
        d = _dt_to_date(best)
        tags.add(_year_tag(d.year))
        tags.add(_decade_tag(d.year))
        tags.add(_century_tag(d.year))
        return tags

    if gran == "decade":
        # Emit each component year + the decade + century.
        y0 = (_dt_to_date(best).year // 10) * 10
        for y in range(y0, y0 + 10):
            tags.add(_year_tag(y))
        tags.add(_decade_tag(y0))
        tags.add(_century_tag(y0))
        return tags

    if gran == "century":
        # Emit each component decade + century.
        y = _dt_to_date(best).year
        cent_start = ((y - 1) // 100) * 100 + 1  # 21st century = 2001..2100
        # Simpler: treat decade buckets within the [earliest, latest] window.
        es = _dt_to_date(earliest)
        ls = _dt_to_date(latest)
        d0 = (es.year // 10) * 10
        d1 = (ls.year // 10) * 10
        for dy in range(d0, d1 + 1, 10):
            tags.add(_decade_tag(dy))
        tags.add(_century_tag(cent_start))
        return tags

    # Unknown granularity: fall back to date tags.
    d = _dt_to_date(best)
    tags.update(_date_to_all_tags(d))
    return tags


def tags_for_interval(fi: FuzzyInterval, max_day_expand: int = 100) -> set[str]:
    """Emit hierarchical tags for a FuzzyInterval.

    Strategy: expand at each granularity (day-only if span < max_day_expand).
    """
    start_date = _dt_to_date(fi.start.earliest)
    end_date = _dt_to_date(fi.end.latest)
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    tags: set[str] = set()
    span_days = (end_date - start_date).days

    # Day tags only if span < max_day_expand.
    if span_days < max_day_expand:
        d = start_date
        while d <= end_date:
            tags.add(_day_tag(d))
            tags.add(_week_tag(d))
            d += timedelta(days=1)

    # Month tags.
    for y, m in _iter_months(start_date, end_date):
        tags.add(_month_tag(y, m))
        tags.add(_quarter_tag(y, m))

    # Quarter (already added via month loop, but ensure coverage if months
    # loop didn't capture any full quarter).
    for y, q in _iter_quarters(start_date, end_date):
        tags.add(f"quarter:{y:04d}-Q{q}")

    # Years.
    for y in range(start_date.year, end_date.year + 1):
        tags.add(_year_tag(y))

    # Decades.
    d0 = (start_date.year // 10) * 10
    d1 = (end_date.year // 10) * 10
    for dy in range(d0, d1 + 1, 10):
        tags.add(_decade_tag(dy))

    # Centuries.
    c_start = ((start_date.year - 1) // 100) * 100 + 1
    c_end = ((end_date.year - 1) // 100) * 100 + 1
    for cy in range(c_start, c_end + 1, 100):
        tags.add(_century_tag(cy))

    return tags


# ---------------------------------------------------------------------------
# Recurrence tagging
# ---------------------------------------------------------------------------
_WEEKDAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

_FREQ_TO_PATTERN = {
    "SECONDLY": "secondly",
    "MINUTELY": "minutely",
    "HOURLY": "hourly",
    "DAILY": "daily",
    "WEEKLY": "weekly",
    "MONTHLY": "monthly",
    "YEARLY": "yearly",
}

_BYDAY_TO_NAME = {
    "MO": "mon",
    "TU": "tue",
    "WE": "wed",
    "TH": "thu",
    "FR": "fri",
    "SA": "sat",
    "SU": "sun",
}


def tags_for_recurrence(r: Recurrence) -> set[str]:
    """Recurrence: dtstart tag set + pattern markers (no instance fan-out)."""
    tags: set[str] = set()
    tags.update(tags_for_instant(r.dtstart))

    # Parse RRULE for FREQ + BYDAY.
    rrule = r.rrule.upper()
    for part in rrule.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k == "FREQ":
            pat = _FREQ_TO_PATTERN.get(v)
            if pat:
                tags.add(f"pattern:{pat}")
        elif k == "BYDAY":
            for code in v.split(","):
                # Strip leading prefix like "1" / "-1" if any.
                code_s = code.strip().lstrip("+-0123456789")
                name = _BYDAY_TO_NAME.get(code_s)
                if name:
                    tags.add(f"weekday:{name}")
    return tags


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def tags_for_expression(
    te: TimeExpression, max_tags: int = MAX_TAGS_PER_EXPR
) -> set[str]:
    """Emit hierarchical tags for a TimeExpression (any kind).

    Applies a budget cap prioritizing coarser granularities when truncating.
    Duration-only expressions emit no tags (no anchor).
    """
    tags: set[str] = set()
    if te.kind == "instant" and te.instant is not None:
        tags = tags_for_instant(te.instant)
    elif te.kind == "interval" and te.interval is not None:
        tags = tags_for_interval(te.interval)
    elif te.kind == "recurrence" and te.recurrence is not None:
        tags = tags_for_recurrence(te.recurrence)
    else:
        return set()

    if len(tags) <= max_tags:
        return tags

    # Truncate: keep coarser-first.
    ordered = sorted(tags, key=lambda t: (_prio_of(t), t))
    return set(ordered[:max_tags])


# ---------------------------------------------------------------------------
# Rarity weighting
# ---------------------------------------------------------------------------
# Shorter (coarser) granularities are more common -> lower weight.
# Very specific tags (day, week) get higher weight.
_DEFAULT_WEIGHTS = {
    "day": 5.0,
    "week": 3.0,
    "month": 2.0,
    "quarter": 1.5,
    "year": 1.0,
    "decade": 0.3,
    "century": 0.1,
    "hour": 6.0,
    "minute": 7.0,
    "second": 8.0,
    "pattern": 0.5,
    "weekday": 0.5,
}


def tag_weight(tag: str, weights: dict[str, float] | None = None) -> float:
    w = weights if weights is not None else _DEFAULT_WEIGHTS
    head = tag.split(":", 1)[0]
    return w.get(head, 1.0)
