"""Per-axis categorical distributions derived from a FuzzyInstant.

Given a FuzzyInstant(earliest, latest, best, granularity) we discretize the
interval at day granularity (hour if time-of-day is meaningful) and compute
a weight per discretization point (Gaussian around ``best`` with sigma =
(latest - earliest)/4, or uniform if ``best`` is None). For each axis we
aggregate point weights by axis-value, then normalize to sum to 1.

An axis is marked "informative" when its entropy is below 95% of the
maximum entropy (log(support)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from schema import (
    FuzzyInstant,
    FuzzyInterval,
    Recurrence,
    TimeExpression,
)

# Axes we track. Each axis value is an int or a short string.
AXES = [
    "year",
    "month",
    "day_of_month",
    "weekday",
    "hour",
    "quarter",
    "decade",
    "season",
    "part_of_day",
    "weekend",
]

# Nominal support sizes (used for max-entropy comparison + uniform default).
_AXIS_SUPPORT = {
    "year": 1,  # dynamic; computed at runtime
    "month": 12,
    "day_of_month": 31,
    "weekday": 7,
    "hour": 24,
    "quarter": 4,
    "decade": 1,
    "season": 4,
    "part_of_day": 4,
    "weekend": 2,
}

UNIFORM_ENTROPY_RATIO = 0.95
TAG_PROB_THRESHOLD = 0.1

# Discretization caps to avoid exploding very wide intervals.
MAX_DAY_POINTS = 366 * 20  # 20 years of days
MAX_HOUR_POINTS = 24 * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _season(month: int) -> str:
    # Meteorological, northern-hemisphere convention.
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


# ---------------------------------------------------------------------------
# Axis distribution computation
# ---------------------------------------------------------------------------
@dataclass
class AxisDistribution:
    """Categorical distribution over one axis.

    values :: {axis_value -> probability}; all prob >= 0 and sum to 1 (unless
    the axis has no support, then empty).
    informative :: True if distribution entropy < 0.95 * max_entropy.
    """

    axis: str
    values: dict
    informative: bool


def _time_of_day_is_meaningful(fi: FuzzyInstant) -> bool:
    gran = fi.granularity
    if gran in ("second", "minute", "hour"):
        return True
    # If the best time has a non-midnight wall clock and granularity is day,
    # the bracket spans a full day so treat hour as uninformative anyway.
    return False


def _discretize(
    fi: FuzzyInstant,
) -> tuple[list[datetime], list[float], bool]:
    """Return (points, weights, hour_meaningful)."""
    earliest = _to_utc(fi.earliest)
    latest = _to_utc(fi.latest)
    best = _to_utc(fi.best) if fi.best is not None else None

    if latest <= earliest:
        latest = earliest + timedelta(days=1)

    hour_mean = _time_of_day_is_meaningful(fi)
    if hour_mean:
        step = timedelta(hours=1)
        max_points = MAX_HOUR_POINTS
    else:
        step = timedelta(days=1)
        max_points = MAX_DAY_POINTS

    # Snap earliest down to step boundary.
    if hour_mean:
        cur = earliest.replace(minute=0, second=0, microsecond=0)
    else:
        cur = earliest.replace(hour=0, minute=0, second=0, microsecond=0)

    points: list[datetime] = []
    while cur < latest and len(points) < max_points:
        points.append(cur)
        cur += step
    if not points:
        points = [earliest]

    # Weights: Gaussian around best with sigma = span/4 (or uniform).
    span_s = max(1.0, (latest - earliest).total_seconds())
    sigma_s = span_s / 4.0
    if best is None:
        weights = [1.0] * len(points)
    else:
        weights = []
        denom = 2.0 * sigma_s * sigma_s
        for p in points:
            d = (p - best).total_seconds()
            w = math.exp(-(d * d) / denom)
            weights.append(w)
    # Normalize.
    total = sum(weights)
    if total <= 0:
        weights = [1.0 / len(points)] * len(points)
    else:
        weights = [w / total for w in weights]

    return points, weights, hour_mean


def _entropy(dist: dict) -> float:
    s = 0.0
    for p in dist.values():
        if p > 0:
            s -= p * math.log(p)
    return s


def _max_entropy(axis: str, support_seen: int) -> float:
    n = max(1, support_seen)
    if axis in ("year", "decade"):
        return math.log(n) if n > 1 else 0.0
    nominal = _AXIS_SUPPORT[axis]
    return math.log(max(n, nominal)) if max(n, nominal) > 1 else 0.0


def _normalize(d: dict) -> dict:
    total = sum(d.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in d.items()}


# ---------------------------------------------------------------------------
# Per-FuzzyInstant axes
# ---------------------------------------------------------------------------
def axes_for_instant(fi: FuzzyInstant) -> dict[str, AxisDistribution]:
    """Compute categorical distributions per axis for a FuzzyInstant.

    Returns a dict keyed by axis name. If the FuzzyInstant has no meaningful
    time-of-day info, the hour/part_of_day axes will be empty (informative=
    False).
    """
    points, weights, hour_meaningful = _discretize(fi)

    raw: dict[str, dict] = {a: {} for a in AXES}
    for p, w in zip(points, weights):
        y = p.year
        m = p.month
        d = p.day
        wk = p.weekday()  # Monday=0
        hr = p.hour
        q = (m - 1) // 3 + 1
        dec = (y // 10) * 10
        seas = _season(m)
        raw["year"][y] = raw["year"].get(y, 0.0) + w
        raw["month"][m] = raw["month"].get(m, 0.0) + w
        raw["day_of_month"][d] = raw["day_of_month"].get(d, 0.0) + w
        raw["weekday"][wk] = raw["weekday"].get(wk, 0.0) + w
        raw["quarter"][q] = raw["quarter"].get(q, 0.0) + w
        raw["decade"][dec] = raw["decade"].get(dec, 0.0) + w
        raw["season"][seas] = raw["season"].get(seas, 0.0) + w
        raw["weekend"]["yes" if wk >= 5 else "no"] = (
            raw["weekend"].get("yes" if wk >= 5 else "no", 0.0) + w
        )
        if hour_meaningful:
            raw["hour"][hr] = raw["hour"].get(hr, 0.0) + w
            pod = _part_of_day(hr)
            raw["part_of_day"][pod] = raw["part_of_day"].get(pod, 0.0) + w

    result: dict[str, AxisDistribution] = {}
    for axis in AXES:
        dist = _normalize(raw[axis])
        if not dist:
            result[axis] = AxisDistribution(axis=axis, values={}, informative=False)
            continue
        e = _entropy(dist)
        me = _max_entropy(axis, len(dist))
        if me <= 0:
            # Single-support axis (n=1): trivially informative.
            inf = True
        else:
            inf = (e / me) < UNIFORM_ENTROPY_RATIO
        result[axis] = AxisDistribution(axis=axis, values=dist, informative=inf)
    return result


# ---------------------------------------------------------------------------
# Per-recurrence axes (concentrated on recurrence field + uniform on others)
# ---------------------------------------------------------------------------
_BYDAY_TO_WK = {
    "MO": 0,
    "TU": 1,
    "WE": 2,
    "TH": 3,
    "FR": 4,
    "SA": 5,
    "SU": 6,
}


def axes_for_recurrence(r: Recurrence) -> dict[str, AxisDistribution]:
    """Axes for a Recurrence: concentrate on the recurrence pattern fields.

    Weekly/BYDAY -> weekday point mass on the specified day(s), other axes
    uniform over a long window (so they score as uninformative).
    BYHOUR -> hour point mass; part-of-day follows.
    MONTHLY/YEARLY without BYMONTHDAY -> day_of_month/month uniform.
    """
    rrule = r.rrule.upper()
    pairs: dict[str, str] = {}
    for part in rrule.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            pairs[k] = v

    # Start from a uniform-over-year distribution using the dtstart bracket
    # (so year/decade remain informative if the dtstart bracket is narrow).
    base = axes_for_instant(r.dtstart)

    # Apply RRULE-specific point masses that override.
    # BYDAY -> weekday.
    if "BYDAY" in pairs:
        wkmap: dict = {}
        for code in pairs["BYDAY"].split(","):
            code_s = code.strip().lstrip("+-0123456789")
            w = _BYDAY_TO_WK.get(code_s)
            if w is not None:
                wkmap[w] = 1.0
        if wkmap:
            n = len(wkmap)
            for k in list(wkmap.keys()):
                wkmap[k] = 1.0 / n
            base["weekday"] = AxisDistribution(
                axis="weekday", values=wkmap, informative=True
            )
            # weekend derived from weekday
            we_y = sum(v for k, v in wkmap.items() if k >= 5)
            we_n = sum(v for k, v in wkmap.items() if k < 5)
            we_dist = {}
            if we_y > 0:
                we_dist["yes"] = we_y
            if we_n > 0:
                we_dist["no"] = we_n
            we_dist = _normalize(we_dist)
            base["weekend"] = AxisDistribution(
                axis="weekend",
                values=we_dist,
                informative=bool(we_dist)
                and (len(we_dist) == 1 or max(we_dist.values()) > 0.75),
            )

    # BYHOUR -> hour point mass and derived part_of_day.
    if "BYHOUR" in pairs:
        hours: dict = {}
        for h in pairs["BYHOUR"].split(","):
            try:
                hv = int(h)
                hours[hv] = 1.0
            except ValueError:
                continue
        if hours:
            n = len(hours)
            for k in list(hours.keys()):
                hours[k] = 1.0 / n
            base["hour"] = AxisDistribution(axis="hour", values=hours, informative=True)
            pod_map: dict = {}
            for hv, w in hours.items():
                pod_map[_part_of_day(hv)] = pod_map.get(_part_of_day(hv), 0.0) + w
            base["part_of_day"] = AxisDistribution(
                axis="part_of_day",
                values=_normalize(pod_map),
                informative=True,
            )

    # BYMONTH -> month point mass.
    if "BYMONTH" in pairs:
        months: dict = {}
        for m in pairs["BYMONTH"].split(","):
            try:
                mv = int(m)
                months[mv] = 1.0
            except ValueError:
                continue
        if months:
            n = len(months)
            for k in list(months.keys()):
                months[k] = 1.0 / n
            base["month"] = AxisDistribution(
                axis="month", values=months, informative=True
            )
            # derived season distribution
            season_map: dict = {}
            for mv, w in months.items():
                s = _season(mv)
                season_map[s] = season_map.get(s, 0.0) + w
            base["season"] = AxisDistribution(
                axis="season",
                values=_normalize(season_map),
                informative=True,
            )
            # derived quarter
            q_map: dict = {}
            for mv, w in months.items():
                q = (mv - 1) // 3 + 1
                q_map[q] = q_map.get(q, 0.0) + w
            base["quarter"] = AxisDistribution(
                axis="quarter",
                values=_normalize(q_map),
                informative=True,
            )

    # FREQ=WEEKLY, MONTHLY, YEARLY without explicit BY-fields -> make year/
    # month uniform (less informative), but keep weekday if BYDAY set.
    freq = pairs.get("FREQ")
    if freq == "WEEKLY":
        # year/month/day_of_month/quarter uninformative (repeats over time).
        for ax in ("year", "month", "day_of_month", "quarter", "season", "decade"):
            base[ax] = AxisDistribution(axis=ax, values={}, informative=False)
    if freq == "DAILY":
        for ax in (
            "year",
            "month",
            "day_of_month",
            "weekday",
            "quarter",
            "season",
            "decade",
            "weekend",
        ):
            base[ax] = AxisDistribution(axis=ax, values={}, informative=False)
    if freq == "MONTHLY":
        for ax in ("year", "quarter", "season", "decade", "weekday", "weekend"):
            base[ax] = AxisDistribution(axis=ax, values={}, informative=False)
    if freq == "YEARLY":
        for ax in ("year", "decade", "weekday", "weekend"):
            base[ax] = AxisDistribution(axis=ax, values={}, informative=False)

    return base


# ---------------------------------------------------------------------------
# Per-interval axes
# ---------------------------------------------------------------------------
def axes_for_interval(fi: FuzzyInterval) -> dict[str, AxisDistribution]:
    # Treat the entire [start.earliest, end.latest] as one big fuzzy instant.
    synthetic = FuzzyInstant(
        earliest=fi.start.earliest,
        latest=fi.end.latest,
        best=fi.start.best or fi.start.earliest,
        granularity=fi.start.granularity,
    )
    return axes_for_instant(synthetic)


# ---------------------------------------------------------------------------
# TimeExpression dispatcher
# ---------------------------------------------------------------------------
def axes_for_expression(te: TimeExpression) -> dict[str, AxisDistribution]:
    if te.kind == "instant" and te.instant is not None:
        return axes_for_instant(te.instant)
    if te.kind == "interval" and te.interval is not None:
        return axes_for_interval(te.interval)
    if te.kind == "recurrence" and te.recurrence is not None:
        return axes_for_recurrence(te.recurrence)
    # duration-only: no axes
    return {a: AxisDistribution(axis=a, values={}, informative=False) for a in AXES}


def merge_axis_dists(
    per_expr: list[dict[str, AxisDistribution]],
) -> dict[str, AxisDistribution]:
    """Merge multiple per-expression axis dists into one (arithmetic mean,
    union of informativeness)."""
    if not per_expr:
        return {a: AxisDistribution(axis=a, values={}, informative=False) for a in AXES}
    out: dict[str, AxisDistribution] = {}
    for axis in AXES:
        merged: dict = {}
        inf = False
        n_contrib = 0
        for d in per_expr:
            ad = d.get(axis)
            if ad is None or not ad.values:
                continue
            n_contrib += 1
            inf = inf or ad.informative
            for k, v in ad.values.items():
                merged[k] = merged.get(k, 0.0) + v
        merged = _normalize(merged)
        out[axis] = AxisDistribution(axis=axis, values=merged, informative=inf)
    return out
