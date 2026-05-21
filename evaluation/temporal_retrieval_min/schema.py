"""ISO 8601 / microsecond helpers shared by the LLM boundary.

The pipeline's single temporal datatype is `core.Interval` (a half-open
microsecond range). These helpers parse the LLM's ISO-string output
into datetimes and convert to/from microsecond timestamps, so the
extractor can hand intervals directly to the scoring layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import overload


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
