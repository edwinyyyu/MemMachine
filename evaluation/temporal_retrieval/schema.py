"""Dataclass + JSON (de)serializer for the temporal-envelope schema.

Single shape across the LLM boundary AND in-Python representation:

    TimeEnvelope(
        surface=str,        # verbatim source span (debug breadcrumb)
        earliest=datetime,  # inclusive left edge of the temporal envelope
        latest=datetime,    # exclusive right edge
        granularity=str,    # one of GRANULARITY_ORDER
    )

A pinpoint reference is just a narrow envelope ("March 15, 2024" =
[2024-03-15T00:00:00Z, 2024-03-16T00:00:00Z)). A span is a wide
envelope ("Q1 2024" = [Jan 1, Apr 1)). A fuzzy reference widens by
one granularity unit. The shape is the same in every case — there is
no kind discriminator.

All datetimes are timezone-aware (UTC). JSON round-trip:

    >>> e = TimeEnvelope(surface="Q1 2024", earliest=..., latest=..., ...)
    >>> s = time_envelope_to_dict(e)
    >>> time_envelope_from_dict(s) == e
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, overload

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


@overload
def parse_iso(s: str) -> datetime: ...
@overload
def parse_iso(s: None) -> None: ...
def parse_iso(s: str | None) -> datetime | None:
    if s is None:
        return None
    s2 = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s2)
    except ValueError:
        dt = datetime.fromisoformat(s2.split(".")[0])
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@overload
def to_us(dt: datetime) -> int: ...
@overload
def to_us(dt: None) -> None: ...
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
# Primitive
# ---------------------------------------------------------------------------
@dataclass
class TimeEnvelope:
    """A half-open temporal interval [earliest, latest) on the calendar.

    `granularity` describes the precision the envelope was derived from
    (the LLM's stated unit, e.g. "day", "quarter") for display and
    optional downstream consumers. The retrieval pipeline scores on
    earliest/latest only.
    """

    surface: str
    earliest: datetime
    latest: datetime
    granularity: str  # Granularity literal


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------
def time_envelope_to_dict(e: TimeEnvelope) -> dict[str, Any]:
    return {
        "surface": e.surface,
        "earliest": iso(e.earliest),
        "latest": iso(e.latest),
        "granularity": e.granularity,
    }


def time_envelope_from_dict(d: dict[str, Any]) -> TimeEnvelope:
    return TimeEnvelope(
        surface=d["surface"],
        earliest=parse_iso(d["earliest"]),
        latest=parse_iso(d["latest"]),
        granularity=d["granularity"],
    )
