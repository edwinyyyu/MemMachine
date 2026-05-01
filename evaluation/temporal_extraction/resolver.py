"""Deterministic post-processing for extracted TimeExpression objects.

Responsibilities:
- ISO validation (parse every datetime field).
- RRULE validation via ``dateutil.rrule.rrulestr``.
- Granularity bracket sanity check (warn only).
- Auto-correct obvious "N <unit>s ago" / "in N <unit>s" arithmetic where
  the LLM is far off.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Literal

from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrulestr
from schema import (
    GRANULARITY_ORDER,
    FuzzyInstant,
    Recurrence,
    TimeExpression,
    iso,
)

BracketMode = Literal["narrow", "quarter", "half", "full_unit"]

# granularity → (lo_seconds, hi_seconds) tolerance for span sanity check
# Very permissive — we only flag mismatches of several orders of magnitude.
_GRAN_SPAN_SECONDS: dict[str, tuple[float, float]] = {
    "second": (0.5, 120),
    "minute": (30, 2 * 3600),
    "hour": (30 * 60, 2 * 24 * 3600),
    "day": (12 * 3600, 4 * 24 * 3600),
    "week": (3 * 24 * 3600, 14 * 24 * 3600),
    "month": (20 * 24 * 3600, 45 * 24 * 3600),
    "quarter": (80 * 24 * 3600, 110 * 24 * 3600),
    "year": (350 * 24 * 3600, 400 * 24 * 3600),
    "decade": (8 * 365 * 24 * 3600, 12 * 365 * 24 * 3600),
    "century": (80 * 365 * 24 * 3600, 120 * 365 * 24 * 3600),
}


_REL_AGO_RE = re.compile(
    r"\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)
_REL_IN_RE = re.compile(
    r"\bin\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
    re.IGNORECASE,
)
_REL_FROMNOW_RE = re.compile(
    r"\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+from\s+now\b",
    re.IGNORECASE,
)

# Counted relative expressions — these get widened per bracket mode.
# Matches: "2 weeks ago", "in 3 months", "3 years from now", "5 years later",
# "5 years earlier", "5 weeks before", "5 weeks after".
_COUNTED_REL_RES = [
    re.compile(
        r"\b(\d+)\s+(second|minute|hour|day|week|month|year|decade)s?\s+(ago|from\s+now|later|earlier|before|after)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bin\s+(\d+)\s+(second|minute|hour|day|week|month|year|decade)s?\b",
        re.IGNORECASE,
    ),
]

# Named-relative (calendar-unit) expressions — these get widened to full
# surrounding unit under full_unit mode; narrow/quarter/half leave them alone
# since they're already unit-sized.
_NAMED_UNIT_RES = {
    "day": re.compile(r"\b(yesterday|today|tomorrow)\b", re.IGNORECASE),
    "week": re.compile(r"\b(last|this|next)\s+week\b", re.IGNORECASE),
    "month": re.compile(r"\b(last|this|next)\s+month\b", re.IGNORECASE),
    "year": re.compile(r"\b(last|this|next)\s+year\b", re.IGNORECASE),
}


class ResolverError(Exception):
    pass


def _expected_best(ref_time: datetime, n: int, unit: str, sign: int) -> datetime:
    u = unit.lower()
    if u == "second":
        return ref_time + sign * timedelta(seconds=n)
    if u == "minute":
        return ref_time + sign * timedelta(minutes=n)
    if u == "hour":
        return ref_time + sign * timedelta(hours=n)
    if u == "day":
        return ref_time + sign * timedelta(days=n)
    if u == "week":
        return ref_time + sign * timedelta(weeks=n)
    if u == "month":
        return ref_time + sign * relativedelta(months=n)
    if u == "year":
        return ref_time + sign * relativedelta(years=n)
    return ref_time


def _unit_seconds(unit: str) -> float:
    u = unit.lower()
    return {
        "second": 1,
        "minute": 60,
        "hour": 3600,
        "day": 86400,
        "week": 86400 * 7,
        "month": 86400 * 30,
        "year": 86400 * 365,
    }[u]


def validate_instant(fi: FuzzyInstant, warnings: list[str], tag: str) -> None:
    if fi.earliest is None or fi.latest is None:
        raise ResolverError(f"{tag}: missing earliest/latest")
    if fi.latest <= fi.earliest:
        warnings.append(f"{tag}: latest<=earliest; swapping/expanding")
        # Expand to 1 sec minimum
        fi.latest = fi.earliest + timedelta(seconds=1)
    if fi.granularity not in GRANULARITY_ORDER:
        warnings.append(f"{tag}: unknown granularity {fi.granularity!r}; using 'day'")
        fi.granularity = "day"
    span = (fi.latest - fi.earliest).total_seconds()
    lo, hi = _GRAN_SPAN_SECONDS.get(fi.granularity, (0, float("inf")))
    if not (lo / 2 <= span <= hi * 2):
        warnings.append(
            f"{tag}: span {span:.0f}s inconsistent with granularity "
            f"{fi.granularity} (expected ~{lo}..{hi})"
        )


def validate_rrule(r: Recurrence, warnings: list[str], tag: str) -> None:
    # Build full RRULE block and try parsing.
    dtstart = r.dtstart.earliest or r.dtstart.best
    if dtstart is None:
        raise ResolverError(f"{tag}: recurrence dtstart missing")
    block = f"DTSTART:{dtstart.strftime('%Y%m%dT%H%M%SZ')}\nRRULE:{r.rrule}"
    try:
        rrulestr(block)
    except Exception as e:
        raise ResolverError(f"{tag}: invalid RRULE {r.rrule!r}: {e}") from e


def arithmetic_check(
    te: TimeExpression, warnings: list[str], auto_correct: bool = True
) -> TimeExpression:
    """If surface is '<N> <unit>s ago' or 'in <N> <unit>s', verify best matches.

    Auto-correct the instant's earliest/latest/best to the deterministic value
    if LLM disagrees by more than ``max(unit, 1 day)``.
    """
    surf = te.surface.lower()
    m = _REL_AGO_RE.search(surf)
    sign = -1
    if m is None:
        m = _REL_IN_RE.search(surf) or _REL_FROMNOW_RE.search(surf)
        sign = 1
    if m is None:
        return te
    if te.kind != "instant" or te.instant is None:
        return te
    n = int(m.group(1))
    unit = m.group(2)
    expected_best = _expected_best(te.reference_time, n, unit, sign)
    have_best = te.instant.best or te.instant.earliest
    delta = abs((expected_best - have_best).total_seconds())
    tol = max(_unit_seconds(unit), 86400.0)
    if delta > tol:
        warnings.append(
            f"arithmetic off by {delta:.0f}s for {te.surface!r} "
            f"(expected {iso(expected_best)}, got {iso(have_best)})"
        )
        if auto_correct:
            half = timedelta(seconds=_unit_seconds(unit) / 2)
            te.instant.earliest = expected_best - half
            te.instant.latest = expected_best + half
            te.instant.best = expected_best
    return te


def post_process(
    te: TimeExpression, auto_correct: bool = True
) -> tuple[TimeExpression, list[str]]:
    """Validate & clean up a TimeExpression. Returns (cleaned, warnings)."""
    warnings: list[str] = []
    if te.kind == "instant":
        if te.instant is None:
            raise ResolverError("instant kind missing instant payload")
        validate_instant(te.instant, warnings, "instant")
    elif te.kind == "interval":
        if te.interval is None:
            raise ResolverError("interval kind missing interval payload")
        validate_instant(te.interval.start, warnings, "interval.start")
        validate_instant(te.interval.end, warnings, "interval.end")
        if te.interval.end.earliest < te.interval.start.earliest:
            warnings.append("interval end < start; swapping")
            te.interval.start, te.interval.end = (
                te.interval.end,
                te.interval.start,
            )
    elif te.kind == "duration":
        if te.duration is None:
            raise ResolverError("duration kind missing duration payload")
    elif te.kind == "recurrence":
        if te.recurrence is None:
            raise ResolverError("recurrence kind missing recurrence payload")
        validate_instant(te.recurrence.dtstart, warnings, "rec.dtstart")
        if te.recurrence.until is not None:
            validate_instant(te.recurrence.until, warnings, "rec.until")
        for i, ex in enumerate(te.recurrence.exdates):
            validate_instant(ex, warnings, f"rec.exdate[{i}]")
        validate_rrule(te.recurrence, warnings, "rec")
    else:
        raise ResolverError(f"unknown kind {te.kind!r}")

    te = arithmetic_check(te, warnings, auto_correct=auto_correct)
    return te, warnings


# ---------------------------------------------------------------------------
# Bracket-mode widening (ablation H1)
# ---------------------------------------------------------------------------
def _start_of_unit(dt: datetime, unit: str) -> datetime:
    u = unit.lower()
    if u == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if u == "week":
        d0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return d0 - timedelta(days=d0.weekday())
    if u == "month":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if u == "year":
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    if u == "decade":
        y = (dt.year // 10) * 10
        return dt.replace(
            year=y, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )
    if u == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    if u == "minute":
        return dt.replace(second=0, microsecond=0)
    if u == "second":
        return dt.replace(microsecond=0)
    return dt


def _end_of_unit(dt: datetime, unit: str) -> datetime:
    start = _start_of_unit(dt, unit)
    u = unit.lower()
    if u == "day":
        return start + timedelta(days=1)
    if u == "week":
        return start + timedelta(days=7)
    if u == "month":
        # jump to first of next month
        y = start.year + (1 if start.month == 12 else 0)
        m = 1 if start.month == 12 else start.month + 1
        return start.replace(year=y, month=m)
    if u == "year":
        return start.replace(year=start.year + 1)
    if u == "decade":
        return start.replace(year=start.year + 10)
    if u == "hour":
        return start + timedelta(hours=1)
    if u == "minute":
        return start + timedelta(minutes=1)
    if u == "second":
        return start + timedelta(seconds=1)
    return dt


def _counted_rel_match(surface: str) -> tuple[int, str, int] | None:
    """Return (N, unit, sign) if surface contains a counted relative
    expression; else None. sign = -1 for "ago/earlier/before", +1 for
    "in X", "X from now", "X later/after"."""
    s = surface.lower()
    for rex in _COUNTED_REL_RES:
        m = rex.search(s)
        if m is None:
            continue
        groups = m.groups()
        if len(groups) == 3:
            n_s, unit, tail = groups
            sign = -1 if tail in ("ago", "earlier", "before") else 1
            return int(n_s), unit, sign
        if len(groups) == 2:
            n_s, unit = groups
            return int(n_s), unit, 1
    return None


def _named_unit_match(surface: str) -> str | None:
    s = surface.lower()
    for unit, rex in _NAMED_UNIT_RES.items():
        if rex.search(s):
            return unit
    return None


def _widen_instant_for_mode(
    fi: FuzzyInstant,
    surface: str,
    mode: BracketMode,
) -> FuzzyInstant:
    """Produce a new FuzzyInstant whose earliest/latest follow ``mode``.

    If ``surface`` is a counted-relative expression ("N <unit>s ago"), the
    width rule applies:
    - narrow: ±1 unit at granularity
    - quarter: ±25% of N units, min 0.5 unit
    - half: ±50% of N units, min 0.5 unit
    - full_unit: whole surrounding calendar unit containing best

    If ``surface`` is a named calendar-unit ("yesterday", "last week", ...)
    we keep it tight (narrow/quarter/half) or widen to the full unit for
    full_unit mode — which for these is effectively a no-op since they're
    already unit-sized.

    If neither pattern matches, we leave the instant as extracted.
    """
    best = fi.best or (fi.earliest + (fi.latest - fi.earliest) / 2)
    counted = _counted_rel_match(surface)
    if counted is not None:
        n, unit, sign = counted
        u_s = (
            _unit_seconds(unit)
            if unit in ("second", "minute", "hour", "day", "week", "month", "year")
            else 86400 * 365 * 10
        )  # decade
        if mode == "narrow":
            # ±1 unit width total (half width = 0.5 unit)
            half = timedelta(seconds=u_s * 0.5)
            return FuzzyInstant(
                earliest=best - half,
                latest=best + half,
                best=best,
                granularity=fi.granularity,
            )
        if mode == "quarter":
            half_seconds = max(0.25 * n * u_s, 0.5 * u_s)
            half = timedelta(seconds=half_seconds)
            return FuzzyInstant(
                earliest=best - half,
                latest=best + half,
                best=best,
                granularity=fi.granularity,
            )
        if mode == "half":
            half_seconds = max(0.5 * n * u_s, 0.5 * u_s)
            half = timedelta(seconds=half_seconds)
            return FuzzyInstant(
                earliest=best - half,
                latest=best + half,
                best=best,
                granularity=fi.granularity,
            )
        if mode == "full_unit":
            # Whole surrounding calendar unit (e.g., for "2 weeks ago" the
            # week containing ref−14d).
            start = _start_of_unit(best, unit)
            end = _end_of_unit(best, unit)
            return FuzzyInstant(
                earliest=start,
                latest=end,
                best=best,
                granularity=fi.granularity,
            )

    named_unit = _named_unit_match(surface)
    if named_unit is not None:
        # These are already calendar-unit sized. For full_unit mode, snap to
        # calendar boundaries. For narrow mode, shrink to a small window
        # around best. For quarter/half, leave as-is (fuzzy widening would
        # extend a named unit into its neighbors — rarely helpful).
        if mode == "narrow":
            u_s = (
                _unit_seconds(named_unit)
                if named_unit
                in ("second", "minute", "hour", "day", "week", "month", "year")
                else 86400
            )
            half = timedelta(seconds=u_s * 0.5)
            return FuzzyInstant(
                earliest=best - half,
                latest=best + half,
                best=best,
                granularity=fi.granularity,
            )
        if mode == "full_unit":
            start = _start_of_unit(best, named_unit)
            end = _end_of_unit(best, named_unit)
            return FuzzyInstant(
                earliest=start,
                latest=end,
                best=best,
                granularity=fi.granularity,
            )
        # quarter / half: leave as-is
        return fi

    return fi


def apply_bracket_mode(te: TimeExpression, mode: BracketMode) -> TimeExpression:
    """Post-process an already-extracted TimeExpression by widening or
    tightening its bracket per ``mode``. No-op for narrow mode when the
    surface does not match a counted/named relative pattern."""
    if te.kind == "instant" and te.instant is not None:
        te.instant = _widen_instant_for_mode(te.instant, te.surface, mode)
    elif te.kind == "interval" and te.interval is not None:
        te.interval.start = _widen_instant_for_mode(te.interval.start, te.surface, mode)
        te.interval.end = _widen_instant_for_mode(te.interval.end, te.surface, mode)
    # recurrences: instance brackets generated at expansion time are fixed
    # per RRULE granularity; mode does not change them.
    return te
