"""Holiday / season / cultural-calendar gazetteer and resolvers.

Rule-based (NO LLM). Fed into the v2'' extractor as a post-processor after
Pass-2 resolution so that surfaces like "Ramadan 2025", "Easter 2015",
"Christmas 2020", "Thanksgiving 2023", "Chinese New Year 2022" resolve to
the concrete wall-clock interval even if the LLM emitted a generic
year-only resolution.

Easter / Ramadan / Chinese New Year dates for 2020-2030 are hardcoded
from public tables (Gregorian calendar).

Christmas / Halloween / Valentine's Day use fixed annual dates.
Thanksgiving (US) = 4th Thursday of November (computed).

Seasons (spring/summer/autumn/fall/winter) + academic terms
(spring term, fall semester, ...) resolve to meteorological
Northern-Hemisphere quarters when combined with a year.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Movable-feast tables (Gregorian dates)
# ---------------------------------------------------------------------------
EASTER = {
    2010: (4, 4),
    2011: (4, 24),
    2012: (4, 8),
    2013: (3, 31),
    2014: (4, 20),
    2015: (4, 5),
    2016: (3, 27),
    2017: (4, 16),
    2018: (4, 1),
    2019: (4, 21),
    2020: (4, 12),
    2021: (4, 4),
    2022: (4, 17),
    2023: (4, 9),
    2024: (3, 31),
    2025: (4, 20),
    2026: (4, 5),
    2027: (3, 28),
    2028: (4, 16),
    2029: (4, 1),
    2030: (4, 21),
}

# Ramadan START dates (Gregorian). Ramadan lasts ~29-30 days.
RAMADAN_START = {
    2010: (8, 11),
    2011: (8, 1),
    2012: (7, 20),
    2013: (7, 9),
    2014: (6, 28),
    2015: (6, 18),
    2016: (6, 6),
    2017: (5, 27),
    2018: (5, 16),
    2019: (5, 6),
    2020: (4, 23),  # to May 23
    2021: (4, 12),  # to May 12
    2022: (4, 1),  # to May 1
    2023: (3, 22),  # to Apr 21
    2024: (3, 10),  # to Apr 9
    2025: (2, 28),  # to Mar 29
    2026: (2, 17),  # to Mar 19
    2027: (2, 7),  # to Mar 8
    2028: (1, 27),  # to Feb 25
    2029: (1, 15),  # to Feb 13
    2030: (1, 5),  # to Feb 3
}

# Chinese New Year (Lunar New Year) — start date; celebration ~15 days
CNY = {
    2010: (2, 14),
    2011: (2, 3),
    2012: (1, 23),
    2013: (2, 10),
    2014: (1, 31),
    2015: (2, 19),
    2016: (2, 8),
    2017: (1, 28),
    2018: (2, 16),
    2019: (2, 5),
    2020: (1, 25),
    2021: (2, 12),
    2022: (2, 1),
    2023: (1, 22),
    2024: (2, 10),
    2025: (1, 29),
    2026: (2, 17),
    2027: (2, 6),
    2028: (1, 26),
    2029: (2, 13),
    2030: (2, 3),
}


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _thanksgiving_us(year: int) -> datetime:
    """4th Thursday of November, US."""
    first = datetime(year, 11, 1, tzinfo=timezone.utc)
    # weekday: Mon=0..Sun=6; Thu=3
    offset = (3 - first.weekday()) % 7
    first_thursday = first + timedelta(days=offset)
    fourth_thursday = first_thursday + timedelta(days=21)
    return fourth_thursday


SEASON_MONTHS_NH = {
    "spring": (3, 5),  # Mar-May
    "summer": (6, 8),  # Jun-Aug
    "autumn": (9, 11),  # Sep-Nov
    "fall": (9, 11),
    "winter": (12, 2),  # Dec-Feb (wraps)
}


# ---------------------------------------------------------------------------
# Surface detection regexes — case-insensitive, permissive.
# ---------------------------------------------------------------------------
_YEAR = r"(\d{4})"

HOLIDAY_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "easter",
        re.compile(
            r"\beaster(?:\s+sunday)?\b(?:\s+(?:of\s+)?" + _YEAR + r")?", re.IGNORECASE
        ),
    ),
    (
        "ramadan",
        re.compile(r"\bramadan\b(?:\s+(?:of\s+)?" + _YEAR + r")?", re.IGNORECASE),
    ),
    (
        "chinese_new_year",
        re.compile(
            r"\b(?:chinese|lunar)\s+new\s+year\b(?:\s+(?:of\s+)?" + _YEAR + r")?",
            re.IGNORECASE,
        ),
    ),
    (
        "thanksgiving",
        re.compile(r"\bthanksgiving\b(?:\s+(?:of\s+)?" + _YEAR + r")?", re.IGNORECASE),
    ),
    (
        "christmas",
        re.compile(
            r"\bchristmas(?:\s+day|\s+eve)?\b(?:\s+(?:of\s+)?" + _YEAR + r")?",
            re.IGNORECASE,
        ),
    ),
    (
        "halloween",
        re.compile(r"\bhalloween\b(?:\s+(?:of\s+)?" + _YEAR + r")?", re.IGNORECASE),
    ),
    (
        "valentines",
        re.compile(
            r"\bvalentine'?s(?:\s+day)?\b(?:\s+(?:of\s+)?" + _YEAR + r")?",
            re.IGNORECASE,
        ),
    ),
    (
        "new_year",
        re.compile(
            r"\bnew\s+year'?s?(?:\s+day|\s+eve)?\b(?:\s+(?:of\s+)?" + _YEAR + r")?",
            re.IGNORECASE,
        ),
    ),
    (
        "independence",
        re.compile(
            r"\b(?:independence\s+day|fourth\s+of\s+july|4th\s+of\s+july|july\s+4)\b(?:\s*,?\s*"
            + _YEAR
            + r")?",
            re.IGNORECASE,
        ),
    ),
]

SEASON_PATTERN = re.compile(
    r"\b(?:(?:early|mid|late)\s+)?(spring|summer|autumn|fall|winter)"
    r"(?:\s+(?:of\s+)?(\d{4}))?\b",
    re.IGNORECASE,
)

ACADEMIC_TERM_PATTERN = re.compile(
    r"\b(spring|summer|fall|winter|autumn)\s+(?:term|semester|quarter)\b"
    r"(?:\s+(?:of\s+)?(\d{4}))?",
    re.IGNORECASE,
)


def resolve_holiday(surface: str, ref_time: datetime) -> dict[str, Any] | None:
    """Given a surface string (already known to be a temporal reference),
    check if it references a holiday / season / academic term. Return a
    dict {earliest, latest, best, granularity} in UTC datetimes, or None if
    no match.

    If a year is explicitly named, use it. Otherwise, resolve relative to
    ref_time: for named holidays, use MOST-RECENT past if ref_time > that
    year's holiday, else current-year.
    """
    s = surface.strip()
    s_low = s.lower()

    # Holiday patterns
    for name, pat in HOLIDAY_PATTERNS:
        m = pat.search(s_low)
        if not m:
            continue
        # Determine year
        year_str = None
        for g in m.groups():
            if g and re.fullmatch(r"\d{4}", g):
                year_str = g
                break
        year = int(year_str) if year_str else ref_time.year

        if name == "easter":
            if year in EASTER:
                mo, d = EASTER[year]
                start = _utc(year, mo, d)
                return {
                    "earliest": start,
                    "latest": start + timedelta(days=1),
                    "best": start,
                    "granularity": "day",
                }
        elif name == "ramadan":
            if year in RAMADAN_START:
                mo, d = RAMADAN_START[year]
                start = _utc(year, mo, d)
                end = start + timedelta(days=30)
                return {
                    "earliest": start,
                    "latest": end,
                    "best": start + timedelta(days=15),
                    "granularity": "month",
                }
        elif name == "chinese_new_year":
            if year in CNY:
                mo, d = CNY[year]
                start = _utc(year, mo, d)
                end = start + timedelta(days=15)
                return {
                    "earliest": start,
                    "latest": end,
                    "best": start,
                    "granularity": "week",
                }
        elif name == "thanksgiving":
            tg = _thanksgiving_us(year)
            return {
                "earliest": tg,
                "latest": tg + timedelta(days=1),
                "best": tg,
                "granularity": "day",
            }
        elif name == "christmas":
            start = _utc(year, 12, 25)
            if "eve" in s_low:
                start = _utc(year, 12, 24)
            return {
                "earliest": start,
                "latest": start + timedelta(days=1),
                "best": start,
                "granularity": "day",
            }
        elif name == "halloween":
            start = _utc(year, 10, 31)
            return {
                "earliest": start,
                "latest": start + timedelta(days=1),
                "best": start,
                "granularity": "day",
            }
        elif name == "valentines":
            start = _utc(year, 2, 14)
            return {
                "earliest": start,
                "latest": start + timedelta(days=1),
                "best": start,
                "granularity": "day",
            }
        elif name == "new_year":
            if "eve" in s_low:
                start = _utc(year, 12, 31)
            else:
                start = _utc(year, 1, 1)
            return {
                "earliest": start,
                "latest": start + timedelta(days=1),
                "best": start,
                "granularity": "day",
            }
        elif name == "independence":
            start = _utc(year, 7, 4)
            return {
                "earliest": start,
                "latest": start + timedelta(days=1),
                "best": start,
                "granularity": "day",
            }

    # Academic term
    am = ACADEMIC_TERM_PATTERN.search(s_low)
    if am:
        season = am.group(1).lower()
        year = int(am.group(2)) if am.group(2) else ref_time.year
        months = SEASON_MONTHS_NH.get(season)
        if months:
            return _season_interval(season, year, months)

    # Season with year
    sm = SEASON_PATTERN.search(s_low)
    if sm and sm.group(2):
        season = sm.group(1).lower()
        year = int(sm.group(2))
        months = SEASON_MONTHS_NH.get(season)
        if months:
            return _season_interval(season, year, months)

    return None


def _season_interval(season: str, year: int, months: tuple[int, int]) -> dict[str, Any]:
    m_start, m_end = months
    if season == "winter":
        # Dec Y .. end of Feb Y+1
        start = _utc(year, 12, 1)
        end = _utc(year + 1, 3, 1)
    else:
        start = _utc(year, m_start, 1)
        # Last day of m_end: use first of m_end+1
        if m_end == 12:
            end = _utc(year + 1, 1, 1)
        else:
            end = _utc(year, m_end + 1, 1)
    mid = start + (end - start) / 2
    return {
        "earliest": start,
        "latest": end,
        "best": mid,
        "granularity": "month",
    }


def surface_matches_holiday(surface: str) -> bool:
    """Cheap check: does the surface contain any holiday/season/term name?"""
    s = surface.lower()
    for _, pat in HOLIDAY_PATTERNS:
        if pat.search(s):
            return True
    if ACADEMIC_TERM_PATTERN.search(s):
        return True
    sm = SEASON_PATTERN.search(s)
    if sm and sm.group(2):
        return True
    return False
