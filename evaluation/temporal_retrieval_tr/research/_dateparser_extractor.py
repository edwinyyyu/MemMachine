"""Pure-dateparser date extractor — no custom regex, no LLM.

Uses dateparser's native features only:
- `search_dates(languages=['en'])` finds candidate date substrings;
  language restriction filters non-English false positives.
- `DateDataParser.get_date_data()` returns both the parsed datetime
  AND a native `period` field ('day' / 'month' / 'year') that we
  use directly as granularity.
- Discard matches where `date_obj` is None.

Patterns dateparser doesn't natively handle (decades-as-range,
quarters, seasons, recurring patterns) are not extracted.

Interface mirrors TemporalExtractor: extract_anchors(text, ref_time).
"""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from dateparser.date import DateDataParser
from dateparser.search import search_dates

if TYPE_CHECKING:
    from temporal_retrieval_tr.time_range import IntervalSet

from temporal_retrieval_min.schema import to_us
from temporal_retrieval_tr.time_range import Interval, IntervalSet


def _instant_to_interval(instant: datetime, period: str) -> Interval | None:
    """Expand a parsed instant into the interval at the given period
    granularity. Only 'day' / 'month' / 'year' are supported (the
    natural outputs of dateparser's `period` field)."""
    if period == "day":
        start = instant.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
    elif period == "week":
        # date_obj is somewhere in the week; expand to Mon-Sun (ISO).
        midnight = instant.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        monday = midnight - timedelta(days=instant.weekday())
        start = monday
        end = (monday + timedelta(days=6)).replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
    elif period == "month":
        start = datetime(instant.year, instant.month, 1)
        last_day = calendar.monthrange(instant.year, instant.month)[1]
        end = datetime(
            instant.year, instant.month, last_day,
            23, 59, 59, 999999,
        )
    elif period == "year":
        start = datetime(instant.year, 1, 1)
        end = datetime(instant.year, 12, 31, 23, 59, 59, 999999)
    else:
        return None
    return Interval(to_us(start), to_us(end))


class DateparserExtractor:
    """Pure-dateparser extractor — no custom regex."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        # DateDataParser is constructed once; settings are passed
        # per-call to keep RELATIVE_BASE up to date.
        pass

    def save_caches(self) -> None:
        pass

    async def extract_anchors(
        self, text: str, ref_time: datetime
    ) -> list[IntervalSet]:
        settings = {
            "RELATIVE_BASE": ref_time,
            "PREFER_DATES_FROM": "past",
        }
        # search_dates finds candidate date substrings; languages='en'
        # filters non-English false positives like 'Most' (Hungarian).
        candidates = search_dates(text, languages=["en"], settings=settings)
        if not candidates:
            return []
        # DateDataParser exposes both parsed datetime and period (grain).
        ddp = DateDataParser(languages=["en"], settings=settings)
        intervals: list[Interval] = []
        for matched_str, _ in candidates:
            data = ddp.get_date_data(matched_str)
            if data.date_obj is None:
                continue
            iv = _instant_to_interval(data.date_obj, data.period)
            if iv is not None:
                intervals.append(iv)
        if not intervals:
            return []
        seen: set[tuple] = set()
        uniq: list[Interval] = []
        for iv in intervals:
            key = (iv.earliest_us, iv.latest_us)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(iv)
        return [IntervalSet.from_intervals([iv]) for iv in uniq]
