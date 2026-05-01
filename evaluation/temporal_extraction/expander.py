"""Expand a Recurrence to a list of FuzzyInstants within a window.

Applies EXDATE filtering. Granularity of each instance is inferred from the
RRULE's finest explicit field (BYHOUR/BYMINUTE/BYSECOND -> minute/second,
else day/week/month per FREQ).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dateutil.rrule import rrulestr
from schema import FuzzyInstant, Recurrence


def _granularity_from_rrule(rrule: str) -> str:
    u = rrule.upper()
    if "BYSECOND" in u:
        return "second"
    if "BYMINUTE" in u:
        return "minute"
    if "BYHOUR" in u:
        return "hour"
    if "FREQ=DAILY" in u:
        return "day"
    if "FREQ=WEEKLY" in u:
        return "day"  # a weekly meeting is still a day-scale event
    if "FREQ=MONTHLY" in u:
        return "day"
    if "FREQ=YEARLY" in u:
        return "day"
    if "FREQ=HOURLY" in u:
        return "hour"
    if "FREQ=MINUTELY" in u:
        return "minute"
    if "FREQ=SECONDLY" in u:
        return "second"
    return "day"


def _bracket_for_granularity(t: datetime, granularity: str) -> FuzzyInstant:
    if granularity == "second":
        e = t.replace(microsecond=0)
        return FuzzyInstant(
            earliest=e, latest=e + timedelta(seconds=1), best=e, granularity=granularity
        )
    if granularity == "minute":
        e = t.replace(second=0, microsecond=0)
        return FuzzyInstant(
            earliest=e, latest=e + timedelta(minutes=1), best=t, granularity=granularity
        )
    if granularity == "hour":
        e = t.replace(minute=0, second=0, microsecond=0)
        return FuzzyInstant(
            earliest=e, latest=e + timedelta(hours=1), best=t, granularity=granularity
        )
    # day-ish
    e = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return FuzzyInstant(
        earliest=e,
        latest=e + timedelta(days=1),
        best=t if t != e else e + timedelta(hours=12),
        granularity=granularity,
    )


def expand(
    r: Recurrence,
    window_start: datetime,
    window_end: datetime,
    cap: int = 2000,
) -> list[FuzzyInstant]:
    """Expand recurrence ``r`` to instances in [window_start, window_end]."""
    dtstart = r.dtstart.best or r.dtstart.earliest
    if dtstart is None:
        return []
    block = (
        f"DTSTART:{dtstart.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}\n"
        f"RRULE:{r.rrule}"
    )
    try:
        rule = rrulestr(block)
    except Exception:
        return []

    # Build exclusion set (keyed by UTC second-level datetime).
    exset: set[datetime] = set()
    for ex in r.exdates:
        t = ex.best or ex.earliest
        if t is None:
            continue
        exset.add(t.astimezone(timezone.utc).replace(microsecond=0))

    until = (r.until.latest or r.until.earliest) if r.until is not None else None
    w_start = max(window_start, dtstart)
    w_end = window_end if until is None else min(window_end, until)

    if w_end <= w_start:
        return []

    granularity = _granularity_from_rrule(r.rrule)
    out: list[FuzzyInstant] = []
    for t in rule.between(w_start, w_end, inc=True):
        if t.astimezone(timezone.utc).replace(microsecond=0) in exset:
            continue
        out.append(_bracket_for_granularity(t.astimezone(timezone.utc), granularity))
        if len(out) >= cap:
            break
    return out
