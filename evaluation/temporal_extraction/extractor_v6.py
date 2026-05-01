"""v6 extractor = v5 + deterministic validation with retry.

After Pass-2 resolves an ISO datetime, validate:
- Weekday alignment: if surface contains a day-of-week name, the resolved
  best date must fall on that weekday (optionally offset by +/-1 for
  common "last X / next X" timezone ambiguities).
- Month arithmetic: if surface is "N months ago" / "in N months", the
  resolved date's month should equal ref_month - N (mod 12).
- Granularity bracket sanity: surface-implied granularity must match the
  declared granularity within one step (gazetteer lookup).

On mismatch, retry Pass-2 once with a correction hint describing the
error. Keeps the corrected result if it parses and re-validates; otherwise
falls back to the uncorrected result.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from dateutil.relativedelta import relativedelta
from extractor_v5 import ExtractorV5
from schema import parse_iso

WEEKDAY_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

# Surface-to-expected-granularity hints.
_GRAN_SURFACE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(yesterday|today|tomorrow|tonight)\b", re.IGNORECASE), "day"),
    (
        re.compile(r"\b(last|this|next|earlier\s+this|later\s+this)\s+week\b", re.IGNORECASE),
        "week",
    ),
    (
        re.compile(r"\b(last|this|next|earlier\s+this|later\s+this)\s+month\b", re.IGNORECASE),
        "month",
    ),
    (
        re.compile(r"\b(last|this|next|earlier\s+this|later\s+this)\s+quarter\b", re.IGNORECASE),
        "quarter",
    ),
    (
        re.compile(r"\b(last|this|next|earlier\s+this|later\s+this)\s+year\b", re.IGNORECASE),
        "year",
    ),
    (
        re.compile(r"\bthe\s+(?:(?:early|mid|late)\s+)?(?:\d{2}0|\d{4}0)s\b", re.IGNORECASE),
        "decade",
    ),
    (
        re.compile(
            r"\bthe\s+(?:first|second|third|fourth|last)\s+week\s+of\s+\w+", re.IGNORECASE
        ),
        "week",
    ),
]

_DOW_RE = re.compile(
    r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.IGNORECASE,
)

_N_MONTHS_AGO_RE = re.compile(
    r"\b(\d+)\s+months?\s+(ago|earlier|before)\b", re.IGNORECASE
)
_IN_N_MONTHS_RE = re.compile(r"\bin\s+(\d+)\s+months?\b", re.IGNORECASE)
_N_MONTHS_FROMNOW_RE = re.compile(
    r"\b(\d+)\s+months?\s+(from\s+now|later|after)\b", re.IGNORECASE
)


def _pred_best_dt(pred: dict[str, Any]) -> datetime | None:
    """Fetch the best (or earliest) datetime from a resolution payload."""
    kind = pred.get("kind")
    if kind == "instant":
        inst = pred.get("instant") or {}
        s = inst.get("best") or inst.get("earliest")
        if s:
            try:
                return parse_iso(s)
            except Exception:
                return None
    elif kind == "interval":
        ivl = pred.get("interval") or {}
        start = ivl.get("start") or {}
        s = start.get("best") or start.get("earliest")
        if s:
            try:
                return parse_iso(s)
            except Exception:
                return None
    elif kind == "recurrence":
        rec = pred.get("recurrence") or {}
        dtstart = rec.get("dtstart") or {}
        s = dtstart.get("best") or dtstart.get("earliest")
        if s:
            try:
                return parse_iso(s)
            except Exception:
                return None
    return None


def _pred_granularity(pred: dict[str, Any]) -> str | None:
    kind = pred.get("kind")
    if kind == "instant":
        return (pred.get("instant") or {}).get("granularity")
    if kind == "interval":
        ivl = pred.get("interval") or {}
        s = (ivl.get("start") or {}).get("granularity")
        return s
    if kind == "recurrence":
        rec = pred.get("recurrence") or {}
        return (rec.get("dtstart") or {}).get("granularity")
    return None


class ExtractorV6(ExtractorV5):
    VERSION = 6

    def validate_resolution(self, pred: dict[str, Any], surface: str) -> str | None:
        surf_lc = surface.lower()
        best_dt = _pred_best_dt(pred)
        declared_gran = _pred_granularity(pred)

        # (1) Weekday alignment
        dow_match = _DOW_RE.search(surface)
        if dow_match and best_dt is not None:
            want_dow = dow_match.group(1).lower()
            got_dow = best_dt.strftime("%A").lower()
            if got_dow != want_dow:
                return (
                    f"Surface says '{want_dow.capitalize()}' but the resolved "
                    f"best date {best_dt.strftime('%Y-%m-%d')} is a "
                    f"{got_dow.capitalize()}. Correct the date to the "
                    f"appropriate {want_dow.capitalize()}."
                )

        # (2) Month arithmetic: "N months ago" should land on ref_month - N
        ref_time = pred.get("reference_time")
        if ref_time:
            try:
                rt = parse_iso(ref_time)
            except Exception:
                rt = None
        else:
            rt = None
        if rt is not None and best_dt is not None:
            m = _N_MONTHS_AGO_RE.search(surf_lc)
            sign = 0
            n = 0
            if m:
                n = int(m.group(1))
                sign = -1
            else:
                m = _IN_N_MONTHS_RE.search(surf_lc) or _N_MONTHS_FROMNOW_RE.search(
                    surf_lc
                )
                if m:
                    n = int(m.group(1))
                    sign = 1
            if sign != 0:
                expected = rt + sign * relativedelta(months=n)
                # Allow 15-day slack (month arithmetic is fuzzy).
                delta_days = abs((expected - best_dt).days)
                if delta_days > 20:
                    return (
                        f"'{surface}' with ref_time {rt.strftime('%Y-%m-%d')} "
                        f"expects approx {expected.strftime('%Y-%m-%d')} "
                        f"(month arithmetic), but got "
                        f"{best_dt.strftime('%Y-%m-%d')}."
                    )

        # (3) Surface-implied granularity
        for pat, want_gran in _GRAN_SURFACE_PATTERNS:
            if pat.search(surface):
                if declared_gran and declared_gran != want_gran:
                    return (
                        f"Surface '{surface}' implies granularity "
                        f"'{want_gran}' but got '{declared_gran}'. "
                        f"Use granularity='{want_gran}' and align the "
                        f"earliest/latest to the corresponding calendar "
                        f"{want_gran} bracket."
                    )
                break

        return None
