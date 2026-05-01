"""Dataclasses + JSON (de)serializers for temporal-extraction schema.

Mirrors DESIGN.md. All datetimes are timezone-aware (UTC). Durations are
stored as ``timedelta``. JSON round-trip:

    >>> d = TimeExpression(kind="instant", ...)
    >>> s = time_expression_to_dict(d)
    >>> time_expression_from_dict(s) == d
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

Granularity = Literal[
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "quarter",
    "year",
    "decade",
    "century",
]

GRANULARITY_ORDER: dict[str, int] = {
    "second": 0,
    "minute": 1,
    "hour": 2,
    "day": 3,
    "week": 4,
    "month": 5,
    "quarter": 6,
    "year": 7,
    "decade": 8,
    "century": 9,
}


# ---------------------------------------------------------------------------
# ISO helpers
# ---------------------------------------------------------------------------
def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str | None) -> datetime | None:
    if s is None:
        return None
    # Normalize trailing Z
    s2 = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s2)
    except ValueError:
        # Try truncated
        dt = datetime.fromisoformat(s2.split(".")[0])
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_us(dt: datetime | None) -> int | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def from_us(us: int | None) -> datetime | None:
    if us is None:
        return None
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
@dataclass
class FuzzyInstant:
    earliest: datetime
    latest: datetime
    best: datetime | None
    granularity: str  # Granularity literal


@dataclass
class FuzzyInterval:
    start: FuzzyInstant
    end: FuzzyInstant


@dataclass
class Recurrence:
    rrule: str
    dtstart: FuzzyInstant
    until: FuzzyInstant | None = None
    exdates: list[FuzzyInstant] = field(default_factory=list)


@dataclass
class TimeExpression:
    kind: str  # instant|interval|duration|recurrence
    surface: str
    reference_time: datetime
    confidence: float = 1.0
    instant: FuzzyInstant | None = None
    interval: FuzzyInterval | None = None
    duration: timedelta | None = None
    recurrence: Recurrence | None = None
    # Char offsets of surface in source (optional, helps span-overlap match)
    span_start: int | None = None
    span_end: int | None = None


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------
def fuzzy_instant_to_dict(fi: FuzzyInstant) -> dict[str, Any]:
    return {
        "earliest": iso(fi.earliest),
        "latest": iso(fi.latest),
        "best": iso(fi.best),
        "granularity": fi.granularity,
    }


def fuzzy_instant_from_dict(d: dict[str, Any]) -> FuzzyInstant:
    return FuzzyInstant(
        earliest=parse_iso(d["earliest"]),
        latest=parse_iso(d["latest"]),
        best=parse_iso(d.get("best")),
        granularity=d["granularity"],
    )


def fuzzy_interval_to_dict(fi: FuzzyInterval) -> dict[str, Any]:
    return {
        "start": fuzzy_instant_to_dict(fi.start),
        "end": fuzzy_instant_to_dict(fi.end),
    }


def fuzzy_interval_from_dict(d: dict[str, Any]) -> FuzzyInterval:
    return FuzzyInterval(
        start=fuzzy_instant_from_dict(d["start"]),
        end=fuzzy_instant_from_dict(d["end"]),
    )


def recurrence_to_dict(r: Recurrence) -> dict[str, Any]:
    return {
        "rrule": r.rrule,
        "dtstart": fuzzy_instant_to_dict(r.dtstart),
        "until": fuzzy_instant_to_dict(r.until) if r.until else None,
        "exdates": [fuzzy_instant_to_dict(e) for e in r.exdates],
    }


def recurrence_from_dict(d: dict[str, Any]) -> Recurrence:
    return Recurrence(
        rrule=d["rrule"],
        dtstart=fuzzy_instant_from_dict(d["dtstart"]),
        until=fuzzy_instant_from_dict(d["until"]) if d.get("until") else None,
        exdates=[fuzzy_instant_from_dict(e) for e in d.get("exdates", [])],
    )


def time_expression_to_dict(te: TimeExpression) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": te.kind,
        "surface": te.surface,
        "reference_time": iso(te.reference_time),
        "confidence": te.confidence,
        "instant": fuzzy_instant_to_dict(te.instant) if te.instant else None,
        "interval": (fuzzy_interval_to_dict(te.interval) if te.interval else None),
        "duration": (
            {"seconds": int(te.duration.total_seconds())}
            if te.duration is not None
            else None
        ),
        "recurrence": (recurrence_to_dict(te.recurrence) if te.recurrence else None),
    }
    if te.span_start is not None:
        out["span_start"] = te.span_start
    if te.span_end is not None:
        out["span_end"] = te.span_end
    return out


def time_expression_from_dict(d: dict[str, Any]) -> TimeExpression:
    dur = None
    if d.get("duration"):
        dur = timedelta(seconds=int(d["duration"]["seconds"]))
    return TimeExpression(
        kind=d["kind"],
        surface=d["surface"],
        reference_time=parse_iso(d["reference_time"]),
        confidence=float(d.get("confidence", 1.0)),
        instant=(fuzzy_instant_from_dict(d["instant"]) if d.get("instant") else None),
        interval=(
            fuzzy_interval_from_dict(d["interval"]) if d.get("interval") else None
        ),
        duration=dur,
        recurrence=(
            recurrence_from_dict(d["recurrence"]) if d.get("recurrence") else None
        ),
        span_start=d.get("span_start"),
        span_end=d.get("span_end"),
    )
