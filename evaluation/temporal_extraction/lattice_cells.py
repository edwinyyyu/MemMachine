"""Lattice cell tagger.

Tags a ``TimeExpression`` at its NATIVE absolute precision ONLY, plus
applicable cyclical-axis tags. This is the inverse of F2's hierarchical
over-tagging: we deliberately do NOT emit coarser absolute tags (e.g. a
``year:1999`` doc does NOT get a ``decade:1990s`` tag). Coarser matches
are discovered at lookup time by walking the lattice.

Axes:

    Absolute (totally ordered, containment):
        century > decade > year > quarter > month > week > day > hour > minute

    Cyclical (orthogonal, lateral match only):
        weekday, month-of-year, day-of-month, hour-of-day,
        season, part-of-day, weekend

Rules:
- Absolute: emit EXACTLY ONE tag at native precision (or multiple at the
  SAME precision when the bracket straddles ~ a few cells, e.g. a fuzzy
  "a couple of years ago" -> year:Y1, year:Y2, year:Y3).
- Cyclical: emit ONLY when the absolute bracket is narrow enough that the
  cyclical axis is concentrated (i.e. does NOT span the full range). We
  gate with a simple span-based heuristic:
    * weekday:          emit iff span_days <= ~3
    * month-of-year:    emit iff span_days <= ~40
    * day-of-month:     emit iff span_days <= ~3
    * hour-of-day:      emit iff the TE has meaningful sub-day granularity
                        AND span_hours <= ~4
    * season:           emit iff span_days <= ~90 AND fits inside a season
    * part-of-day:      emit iff hour-meaningful and span_hours <= ~12
    * weekend:          emit iff span_days <= ~2

Recurrence:
    weekday/hour/month-of-year derived from RRULE BYDAY/BYHOUR/BYMONTH.
    Absolute tag only if dtstart+until form a narrow bracket (same rules).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from schema import (
    FuzzyInstant,
    FuzzyInterval,
    Recurrence,
    TimeExpression,
)

# ---------------------------------------------------------------------------
# Axis ordering & span (in approximate days). Used for scoring.
# ---------------------------------------------------------------------------
ABSOLUTE_AXIS_ORDER: list[str] = [
    "century",
    "decade",
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "hour",
    "minute",
]

ABSOLUTE_AXIS_SPAN_DAYS: dict[str, float] = {
    "minute": 1.0 / (24 * 60),
    "hour": 1.0 / 24,
    "day": 1.0,
    "week": 7.0,
    "month": 30.4,
    "quarter": 91.3,
    "year": 365.25,
    "decade": 3652.5,
    "century": 36525.0,
}

CYCLICAL_AXES: list[str] = [
    "weekday",
    "month_of_year",
    "day_of_month",
    "hour_of_day",
    "season",
    "part_of_day",
    "weekend",
]

_MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
_WEEKDAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
_BYDAY_TO_WK = {
    "MO": 0,
    "TU": 1,
    "WE": 2,
    "TH": 3,
    "FR": 4,
    "SA": 5,
    "SU": 6,
}


# ---------------------------------------------------------------------------
# Cyclical helpers
# ---------------------------------------------------------------------------
def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _part_of_day(hour: int) -> str:
    if 0 <= hour < 6:
        return "night"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_date(dt: datetime) -> date:
    return _to_utc(dt).date()


def _century_str(year: int) -> str:
    cent = year // 100 + 1
    if cent % 100 in (11, 12, 13):
        suf = "th"
    elif cent % 10 == 1:
        suf = "st"
    elif cent % 10 == 2:
        suf = "nd"
    elif cent % 10 == 3:
        suf = "rd"
    else:
        suf = "th"
    return f"{cent}{suf}"


# ---------------------------------------------------------------------------
# Absolute tag formatting
# ---------------------------------------------------------------------------
def abs_tag_day(d: date) -> str:
    return f"day:{d.isoformat()}"


def abs_tag_week(d: date) -> str:
    iso = d.isocalendar()
    return f"week:{iso[0]:04d}-W{iso[1]:02d}"


def abs_tag_month(y: int, m: int) -> str:
    return f"month:{y:04d}-{m:02d}"


def abs_tag_quarter(y: int, m: int) -> str:
    q = (m - 1) // 3 + 1
    return f"quarter:{y:04d}-Q{q}"


def abs_tag_year(y: int) -> str:
    return f"year:{y:04d}"


def abs_tag_decade(y: int) -> str:
    return f"decade:{(y // 10) * 10}s"


def abs_tag_century(y: int) -> str:
    return f"century:{_century_str(y)}"


def abs_tag_hour(dt: datetime) -> str:
    dt = _to_utc(dt)
    return f"hour:{dt.strftime('%Y-%m-%dT%H')}"


def abs_tag_minute(dt: datetime) -> str:
    dt = _to_utc(dt)
    return f"minute:{dt.strftime('%Y-%m-%dT%H:%M')}"


def abs_tag_for_precision(precision: str, dt: datetime) -> str:
    d = _to_date(dt)
    if precision == "minute":
        return abs_tag_minute(dt)
    if precision == "hour":
        return abs_tag_hour(dt)
    if precision == "day":
        return abs_tag_day(d)
    if precision == "week":
        return abs_tag_week(d)
    if precision == "month":
        return abs_tag_month(d.year, d.month)
    if precision == "quarter":
        return abs_tag_quarter(d.year, d.month)
    if precision == "year":
        return abs_tag_year(d.year)
    if precision == "decade":
        return abs_tag_decade(d.year)
    if precision == "century":
        return abs_tag_century(d.year)
    # second -> collapse to minute for index sanity
    return abs_tag_minute(dt)


# ---------------------------------------------------------------------------
# Absolute-tag enumeration across a span
# ---------------------------------------------------------------------------
def _iter_days(s: date, e: date) -> Iterable[date]:
    cur = s
    while cur <= e:
        yield cur
        cur += timedelta(days=1)


def _iter_months(s: date, e: date) -> Iterable[tuple[int, int]]:
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def _iter_quarters(s: date, e: date) -> Iterable[tuple[int, int]]:
    sq = (s.month - 1) // 3 + 1
    eq = (e.month - 1) // 3 + 1
    y, q = s.year, sq
    while (y, q) <= (e.year, eq):
        yield y, q
        if q == 4:
            y += 1
            q = 1
        else:
            q += 1


def native_absolute_tags(
    precision: str, earliest: datetime, latest: datetime, best: datetime | None
) -> list[str]:
    """Emit absolute tags at the NATIVE precision covering the bracket.

    Treats ``[earliest, latest)`` as HALF-OPEN (standard convention: a
    day-granularity bracket is ``[Y-M-D, Y-M-D+1)``). This avoids emitting
    year:2000 for a "year:1999" bracket that ends at midnight Jan 1 2000.

    - Narrow bracket (e.g. 1 day at precision=day) -> exactly one tag.
    - Wide fuzzy bracket (e.g. "a couple years ago" spanning 3 years at
      precision=year) -> one tag per cell.
    - Cap at 50 tags to avoid explosion.
    """
    if precision == "second":
        precision = "minute"
    earliest = _to_utc(earliest)
    latest = _to_utc(latest)
    if latest < earliest:
        earliest, latest = latest, earliest
    # Shrink latest by 1 microsecond so boundary midnights don't leak a
    # spurious trailing cell.
    latest_shrunk = latest - timedelta(microseconds=1)
    if latest_shrunk < earliest:
        latest_shrunk = earliest

    tags: list[str] = []
    MAX = 50

    if precision in ("minute", "hour"):
        step = timedelta(minutes=1 if precision == "minute" else 60)
        cur = earliest.replace(second=0, microsecond=0)
        if precision == "hour":
            cur = cur.replace(minute=0)
        while cur < latest and len(tags) < MAX:
            tags.append(abs_tag_for_precision(precision, cur))
            cur += step
        if not tags:
            tags.append(abs_tag_for_precision(precision, earliest))
        return tags

    s_date = earliest.date()
    e_date = latest_shrunk.date()

    if precision == "day":
        for d in _iter_days(s_date, e_date):
            if len(tags) >= MAX:
                break
            tags.append(abs_tag_day(d))
        return tags

    if precision == "week":
        seen_weeks: set[str] = set()
        for d in _iter_days(s_date, e_date):
            if len(tags) >= MAX:
                break
            t = abs_tag_week(d)
            if t not in seen_weeks:
                seen_weeks.add(t)
                tags.append(t)
        return tags

    if precision == "month":
        for y, m in _iter_months(s_date, e_date):
            if len(tags) >= MAX:
                break
            tags.append(abs_tag_month(y, m))
        return tags

    if precision == "quarter":
        for y, q in _iter_quarters(s_date, e_date):
            if len(tags) >= MAX:
                break
            tags.append(f"quarter:{y:04d}-Q{q}")
        return tags

    if precision == "year":
        y0, y1 = s_date.year, e_date.year
        for y in range(y0, y1 + 1):
            if len(tags) >= MAX:
                break
            tags.append(abs_tag_year(y))
        return tags

    if precision == "decade":
        d0 = (s_date.year // 10) * 10
        d1 = (e_date.year // 10) * 10
        y = d0
        while y <= d1 and len(tags) < MAX:
            tags.append(abs_tag_decade(y))
            y += 10
        return tags

    if precision == "century":
        c0 = s_date.year // 100
        c1 = e_date.year // 100
        c = c0
        while c <= c1 and len(tags) < MAX:
            tags.append(abs_tag_century(c * 100 + 50))
            c += 1
        return tags

    # fallback
    return [abs_tag_year(s_date.year)]


# ---------------------------------------------------------------------------
# Cyclical tags — only if bracket is narrow enough for the axis to be
# concentrated.
# ---------------------------------------------------------------------------
def _span_days(earliest: datetime, latest: datetime) -> float:
    return max(0.0, (latest - earliest).total_seconds() / 86400.0)


def _span_hours(earliest: datetime, latest: datetime) -> float:
    return max(0.0, (latest - earliest).total_seconds() / 3600.0)


def cyclical_tags_for_instant(fi: FuzzyInstant) -> set[str]:
    """Emit cyclical tags whose axis concentrates over the bracket.

    Uses the ``best`` point (fallback earliest) to determine the axis
    value.
    """
    if fi.best is None and fi.earliest is None:
        return set()
    best = _to_utc(fi.best if fi.best is not None else fi.earliest)
    e = _to_utc(fi.earliest)
    l = _to_utc(fi.latest)
    span_d = _span_days(e, l)
    span_h = _span_hours(e, l)
    gran = fi.granularity

    tags: set[str] = set()

    # weekday / day-of-month / weekend — concentrate when narrow (~day)
    if span_d <= 3.0:
        wk = best.weekday()
        tags.add(f"weekday:{_WEEKDAY_NAMES[wk]}")
        tags.add(f"weekend:{'yes' if wk >= 5 else 'no'}")
        tags.add(f"day_of_month:{best.day}")

    # month-of-year — concentrate when bracket fits in a month-ish
    if span_d <= 40.0:
        # Emit month_of_year from best (or earliest/latest if best unclear)
        if e.month == l.month and e.year == l.year:
            tags.add(f"month_of_year:{_MONTH_NAMES[best.month]}")
        else:
            # Fallback: if best is within bracket and single-month
            tags.add(f"month_of_year:{_MONTH_NAMES[best.month]}")

    # season — concentrate when bracket fits in a season (~90d)
    if span_d <= 90.0:
        sboth = _season(e.month) == _season(l.month)
        if sboth:
            tags.add(f"season:{_season(best.month)}")
        else:
            tags.add(f"season:{_season(best.month)}")

    # hour / part-of-day — only if TE has sub-day granularity
    if gran in ("hour", "minute", "second") and span_h <= 12.0:
        tags.add(f"hour_of_day:{best.hour:02d}")
    if gran in ("hour", "minute", "second") and span_h <= 12.0:
        tags.add(f"part_of_day:{_part_of_day(best.hour)}")

    return tags


# ---------------------------------------------------------------------------
# Recurrence cyclical tags (directly from RRULE)
# ---------------------------------------------------------------------------
def cyclical_tags_for_recurrence(r: Recurrence) -> set[str]:
    tags: set[str] = set()
    pairs: dict[str, str] = {}
    for part in r.rrule.upper().split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            pairs[k] = v

    if "BYDAY" in pairs:
        for code in pairs["BYDAY"].split(","):
            code_s = code.strip().lstrip("+-0123456789")
            wk = _BYDAY_TO_WK.get(code_s)
            if wk is not None:
                tags.add(f"weekday:{_WEEKDAY_NAMES[wk]}")
                tags.add(f"weekend:{'yes' if wk >= 5 else 'no'}")

    if "BYHOUR" in pairs:
        for h in pairs["BYHOUR"].split(","):
            try:
                hv = int(h)
                tags.add(f"hour_of_day:{hv:02d}")
                tags.add(f"part_of_day:{_part_of_day(hv)}")
            except ValueError:
                continue

    if "BYMONTH" in pairs:
        for m in pairs["BYMONTH"].split(","):
            try:
                mv = int(m)
                tags.add(f"month_of_year:{_MONTH_NAMES[mv]}")
                tags.add(f"season:{_season(mv)}")
            except (ValueError, KeyError):
                continue

    if "BYMONTHDAY" in pairs:
        for d in pairs["BYMONTHDAY"].split(","):
            try:
                dv = int(d)
                if 1 <= dv <= 31:
                    tags.add(f"day_of_month:{dv}")
            except ValueError:
                continue

    freq = pairs.get("FREQ")
    if freq == "WEEKLY" and "BYDAY" not in pairs:
        # No weekday info — don't guess.
        pass

    return tags


# ---------------------------------------------------------------------------
# Main entry: tags_for_expression
# ---------------------------------------------------------------------------
@dataclass
class LatticeTagSet:
    absolute: list[tuple[str, str]]  # (precision, full_tag)
    cyclical: set[str]  # full_tags like "weekday:Thursday"

    @property
    def all_tags(self) -> set[str]:
        out: set[str] = set()
        for _prec, t in self.absolute:
            out.add(t)
        out |= self.cyclical
        return out


def _effective_precision(fi: FuzzyInstant) -> str:
    """Pick the native precision from the FuzzyInstant. Honor the declared
    granularity but override if the bracket is much wider than that unit."""
    gran = fi.granularity or "day"
    e = _to_utc(fi.earliest)
    l = _to_utc(fi.latest)
    span = _span_days(e, l)
    span_h = span * 24
    # If the bracket is significantly wider than declared (e.g. declared
    # "year" with a 12-year span, or declared "minute" but spanning a month),
    # coarsen up the lattice until it fits.
    if gran == "minute" and span_h > 2:
        gran = "hour"
    if gran == "hour" and span_h > 30:
        gran = "day"
    if gran == "day" and span > 8:
        gran = "week"
    if gran == "week" and span > 60:
        gran = "month"
    if gran == "month" and span > 14 * 30:
        gran = "year"
    if gran == "year" and span > 4 * 365:
        gran = "decade"
    if gran == "year" and span > 15 * 365:
        gran = "century"
    if gran == "decade" and span > 15 * 365:
        gran = "century"
    # Special handling: day coarsening to month when very wide
    if gran == "day" and span > 60:
        gran = "month"
    return gran


def tags_for_instant(fi: FuzzyInstant) -> LatticeTagSet:
    prec = _effective_precision(fi)
    abs_list: list[tuple[str, str]] = []
    if fi.best is not None or fi.earliest is not None:
        abs_tags = native_absolute_tags(prec, fi.earliest, fi.latest, fi.best)
        abs_list = [(prec, t) for t in abs_tags]
    cyc = cyclical_tags_for_instant(fi)
    # If the effective precision is coarser than week (year/decade/century),
    # drop fine cyclical tags (weekday, day-of-month) — they can't be
    # concentrated.
    if prec in ("year", "decade", "century", "quarter", "month"):
        cyc = {t for t in cyc if not t.startswith(("weekday:", "day_of_month:"))}
    if prec in ("year", "decade", "century"):
        cyc = {t for t in cyc if not t.startswith(("month_of_year:", "season:"))}
    if prec in ("day", "week", "month", "quarter", "year", "decade", "century"):
        cyc = {t for t in cyc if not t.startswith(("hour_of_day:", "part_of_day:"))}
    return LatticeTagSet(absolute=abs_list, cyclical=cyc)


def tags_for_interval(fi: FuzzyInterval) -> LatticeTagSet:
    # Collapse to a synthetic instant spanning the whole interval.
    synthetic = FuzzyInstant(
        earliest=fi.start.earliest,
        latest=fi.end.latest,
        best=fi.start.best or fi.start.earliest,
        granularity=fi.start.granularity,
    )
    return tags_for_instant(synthetic)


def tags_for_recurrence(r: Recurrence) -> LatticeTagSet:
    cyc = cyclical_tags_for_recurrence(r)
    abs_list: list[tuple[str, str]] = []
    # Only attach absolute tag if dtstart+until form a bounded narrow range.
    if r.until is not None:
        span = _span_days(_to_utc(r.dtstart.earliest), _to_utc(r.until.latest))
        if span <= 366:
            ts = tags_for_instant(
                FuzzyInstant(
                    earliest=r.dtstart.earliest,
                    latest=r.until.latest,
                    best=r.dtstart.best,
                    granularity=r.dtstart.granularity,
                )
            )
            abs_list = ts.absolute
    return LatticeTagSet(absolute=abs_list, cyclical=cyc)


def tags_for_expression(te: TimeExpression) -> LatticeTagSet:
    if te.kind == "instant" and te.instant is not None:
        return tags_for_instant(te.instant)
    if te.kind == "interval" and te.interval is not None:
        return tags_for_interval(te.interval)
    if te.kind == "recurrence" and te.recurrence is not None:
        return tags_for_recurrence(te.recurrence)
    return LatticeTagSet(absolute=[], cyclical=set())


# ---------------------------------------------------------------------------
# Lattice walk helpers (for retrieval expansion)
# ---------------------------------------------------------------------------
def ancestors_of_absolute(tag: str) -> list[str]:
    """Walk UP the absolute lattice from a tag. Returns ancestor tags at
    coarser granularities. Does not include the tag itself.

    Example: ``day:2024-03-15`` ->
        [``week:2024-W11``, ``month:2024-03``, ``quarter:2024-Q1``,
         ``year:2024``, ``decade:2020s``, ``century:21st``]
    """
    if ":" not in tag:
        return []
    prec, val = tag.split(":", 1)
    try:
        if prec == "day":
            d = date.fromisoformat(val)
            return [
                abs_tag_week(d),
                abs_tag_month(d.year, d.month),
                abs_tag_quarter(d.year, d.month),
                abs_tag_year(d.year),
                abs_tag_decade(d.year),
                abs_tag_century(d.year),
            ]
        if prec == "hour":
            dt = datetime.strptime(val, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            d = dt.date()
            return [
                abs_tag_day(d),
                abs_tag_week(d),
                abs_tag_month(d.year, d.month),
                abs_tag_quarter(d.year, d.month),
                abs_tag_year(d.year),
                abs_tag_decade(d.year),
                abs_tag_century(d.year),
            ]
        if prec == "minute":
            dt = datetime.strptime(val, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
            d = dt.date()
            return [
                abs_tag_hour(dt),
                abs_tag_day(d),
                abs_tag_week(d),
                abs_tag_month(d.year, d.month),
                abs_tag_quarter(d.year, d.month),
                abs_tag_year(d.year),
                abs_tag_decade(d.year),
                abs_tag_century(d.year),
            ]
        if prec == "week":
            # Parse ISO year-week.
            yr, wk = val.split("-W")
            iso_year = int(yr)
            iso_wk = int(wk)
            # Pick the Thursday of the week (stable year)
            d = date.fromisocalendar(iso_year, iso_wk, 4)
            return [
                abs_tag_month(d.year, d.month),
                abs_tag_quarter(d.year, d.month),
                abs_tag_year(d.year),
                abs_tag_decade(d.year),
                abs_tag_century(d.year),
            ]
        if prec == "month":
            y, m = val.split("-")
            yr = int(y)
            mo = int(m)
            return [
                abs_tag_quarter(yr, mo),
                abs_tag_year(yr),
                abs_tag_decade(yr),
                abs_tag_century(yr),
            ]
        if prec == "quarter":
            y, q = val.split("-Q")
            yr = int(y)
            return [
                abs_tag_year(yr),
                abs_tag_decade(yr),
                abs_tag_century(yr),
            ]
        if prec == "year":
            yr = int(val)
            return [
                abs_tag_decade(yr),
                abs_tag_century(yr),
            ]
        if prec == "decade":
            # val like "1990s"
            yr = int(val.rstrip("s"))
            return [abs_tag_century(yr)]
        if prec == "century":
            return []
    except Exception:
        return []
    return []


def children_of_absolute(tag: str) -> list[str]:
    """Walk DOWN the absolute lattice ONE level. Returns direct-children
    tags (the cells contained within ``tag``).

    Capped at 1 level to bound expansion.
    Example: ``year:2024`` -> ``month:2024-01`` .. ``month:2024-12``.
    """
    if ":" not in tag:
        return []
    prec, val = tag.split(":", 1)
    try:
        if prec == "century":
            # val like "21st"; find decade range for the century
            # 21st century = 2001..2100; decades 2000..2090 (inclusive)
            num_part = ""
            for ch in val:
                if ch.isdigit():
                    num_part += ch
                else:
                    break
            if not num_part:
                return []
            cent = int(num_part)
            start_year = (cent - 1) * 100
            return [abs_tag_decade(y) for y in range(start_year, start_year + 100, 10)]
        if prec == "decade":
            yr0 = int(val.rstrip("s"))
            return [abs_tag_year(y) for y in range(yr0, yr0 + 10)]
        if prec == "year":
            yr = int(val)
            return [abs_tag_quarter(yr, (q - 1) * 3 + 1) for q in range(1, 5)]
        if prec == "quarter":
            y, q = val.split("-Q")
            yr = int(y)
            qn = int(q)
            m0 = (qn - 1) * 3 + 1
            return [abs_tag_month(yr, m) for m in range(m0, m0 + 3)]
        if prec == "month":
            y, m = val.split("-")
            yr = int(y)
            mo = int(m)
            # Days in month
            from calendar import monthrange

            _, ndays = monthrange(yr, mo)
            return [abs_tag_day(date(yr, mo, d)) for d in range(1, ndays + 1)]
        if prec == "week":
            yr_s, wk_s = val.split("-W")
            yr = int(yr_s)
            wk = int(wk_s)
            d0 = date.fromisocalendar(yr, wk, 1)
            return [abs_tag_day(d0 + timedelta(days=i)) for i in range(7)]
        if prec == "day":
            d = date.fromisoformat(val)
            return [
                abs_tag_hour(datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc))
                for h in range(24)
            ]
        if prec == "hour":
            dt = datetime.strptime(val, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            return [abs_tag_minute(dt.replace(minute=m)) for m in range(60)]
    except Exception:
        return []
    return []


# ---------------------------------------------------------------------------
# Span lookup for scoring
# ---------------------------------------------------------------------------
def precision_of_tag(tag: str) -> str:
    if ":" not in tag:
        return "year"
    head = tag.split(":", 1)[0]
    return head


def span_days_of_tag(tag: str) -> float:
    prec = precision_of_tag(tag)
    if prec in CYCLICAL_AXES:
        # Treat cyclical tags as "day-width repeating" for scoring purposes —
        # they're very narrow on the cyclical axis but repeat.
        return 1.0
    return ABSOLUTE_AXIS_SPAN_DAYS.get(prec, 365.0)
