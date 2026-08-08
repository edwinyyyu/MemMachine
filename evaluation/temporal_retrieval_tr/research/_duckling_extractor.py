"""Duckling-backed date extractor — calls a local Duckling HTTP server.

Duckling (https://github.com/facebook/duckling) is a rule-based NLP
entity recognizer. Compared to dateparser it natively handles:
- Intervals (from/to) for seasons, ranges, quarters
- Multi-value recurring patterns ("Friday night" → next 3 Fridays)
- Quarter / week / hour granularities
- "last Monday at 3pm" type compound phrases
- Better false-positive rejection

Server: run `docker run -d -p 8000:8000 rasa/duckling`.

Interface mirrors TemporalExtractor: extract_anchors(text, ref_time).
"""
from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass

from temporal_retrieval_min.schema import parse_iso, to_us
from temporal_retrieval_tr.time_range import (
    NEG_INF, POS_INF, Endpoint, Interval, IntervalSet,
)


DUCKLING_URL = "http://localhost:8000/parse"
CACHE_DIR = (
    Path(__file__).resolve().parents[1] / "cache" / "duckling"
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _grain_to_end(start: datetime, grain: str) -> datetime:
    """End of the period containing `start` at the given grain."""
    if grain == "second":
        return start.replace(microsecond=999999)
    if grain == "minute":
        return start.replace(second=59, microsecond=999999)
    if grain == "hour":
        return start.replace(minute=59, second=59, microsecond=999999)
    if grain == "day":
        return start.replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
    if grain == "week":
        # Duckling returns the week's start instant (Sunday for en_US,
        # Monday for en_GB, etc.) -- end-of-week is the same time of
        # day 6 days later, regardless of locale.
        last_day = (start + _td(days=6)).replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
        return last_day
    if grain == "month":
        last_day = calendar.monthrange(start.year, start.month)[1]
        return datetime(
            start.year, start.month, last_day,
            23, 59, 59, 999999,
            tzinfo=start.tzinfo,
        )
    if grain == "quarter":
        # quarter starts: 1, 4, 7, 10 (Jan, Apr, Jul, Oct)
        q_start_month = ((start.month - 1) // 3) * 3 + 1
        q_end_month = q_start_month + 3
        end_year = start.year + (q_end_month - 1) // 12
        end_month_norm = ((q_end_month - 1) % 12) + 1
        end = datetime(
            end_year, end_month_norm, 1,
            tzinfo=start.tzinfo,
        )
        # end is exclusive next-quarter start; subtract 1us
        return end.replace(
            microsecond=end.microsecond - 1
        ) if end.microsecond > 0 else end + _td(microseconds=-1)
    if grain == "year":
        return datetime(
            start.year, 12, 31, 23, 59, 59, 999999,
            tzinfo=start.tzinfo,
        )
    # Unknown grain: treat as instant → 1us span
    return start.replace(microsecond=min(start.microsecond + 1, 999999))


def _td(**kw):
    from datetime import timedelta
    return timedelta(**kw)


def _entity_to_intervals(entity: dict) -> list[Interval]:
    """Convert one Duckling time-dim entity into Intervals.

    Use ONLY the primary value (`value.value` or `value.from`/`value.to`)
    per entity, ignoring the `values[]` list of alternative candidates.
    Duckling lists alternatives for ambiguous parses like 'the first'
    → next 3 month-firsts, or 'October 15' → next 3 Octobers. The
    primary is Duckling's top-ranked disambiguation; the alternatives
    are usually noise or marginally-less-likely candidates.

    Each entity may be:
      - point: {type: value, value: ISO, grain: G}  → [start, end_of_grain)
      - range: {type: interval, from: {value,grain}, to: {value,grain}}
    """
    v = entity.get("value", {})
    t = v.get("type")
    if t == "value":
        try:
            start = parse_iso(v["value"])
        except Exception:
            return []
        grain = v.get("grain", "day")
        end = _grain_to_end(start, grain)
        if start < end:
            return [Interval(to_us(start), to_us(end))]
        return []
    elif t == "interval":
        # Duckling intervals may be unbounded on one side:
        # 'before June 5'  → only `to`   → [-∞, June 5)
        # 'after June 5'   → only `from` → [June 5, +∞)
        # 'since 2020'     → only `from`
        # 'until 2024'     → only `to`
        f = v.get("from")
        tt = v.get("to")
        start: Endpoint
        end: Endpoint
        if f is not None:
            try:
                start = to_us(parse_iso(f["value"]))
            except Exception:
                return []
        else:
            start = NEG_INF
        if tt is not None:
            try:
                end = to_us(parse_iso(tt["value"]))
            except Exception:
                return []
        else:
            end = POS_INF
        if start < end:
            return [Interval(start, end)]
        return []
    return []


class DucklingHTTPExtractor:
    """Calls a local Duckling HTTP server at DUCKLING_URL.

    Disk-cached by (text, ref_time) hash so re-runs hit the cache.
    """

    def __init__(
        self,
        url: str = DUCKLING_URL,
        cache_dir: Path | None = None,
        concurrency: int = 8,
    ) -> None:
        self.url = url
        self.cache_file = (cache_dir or CACHE_DIR) / "duckling_cache.pkl"
        self._cache: dict[str, list[Interval]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    self._cache = pickle.load(f)
            except Exception:
                self._cache = {}
        self._sem = asyncio.Semaphore(concurrency)
        self._client: httpx.AsyncClient | None = None
        self._dirty = False

    def save_caches(self) -> None:
        if not self._dirty:
            return
        tmp = self.cache_file.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self._cache, f)
        tmp.replace(self.cache_file)
        self._dirty = False

    def _key(self, text: str, ref_time: datetime) -> str:
        ref_us = int(ref_time.timestamp() * 1000)
        return hashlib.sha256(
            f"{ref_us}|{text}".encode("utf-8")
        ).hexdigest()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _call(self, text: str, ref_time: datetime) -> list[Interval]:
        ref_ms = int(ref_time.timestamp() * 1000)
        client = await self._ensure_client()
        async with self._sem:
            try:
                r = await client.post(
                    self.url,
                    data={
                        "locale": "en_US",
                        "tz": "UTC",
                        "reftime": ref_ms,
                        "dims": json.dumps(["time"]),
                        "text": text,
                    },
                )
                if r.status_code != 200:
                    return []
                entities = r.json()
            except Exception:
                return []
        intervals: list[Interval] = []
        for e in entities:
            if e.get("dim") != "time":
                continue
            intervals.extend(_entity_to_intervals(e))
        return intervals

    async def extract_anchors(
        self, text: str, ref_time: datetime
    ) -> list[IntervalSet]:
        key = self._key(text, ref_time)
        if key in self._cache:
            intervals = self._cache[key]
        else:
            # Ensure ref_time has tz so timestamp() is well-defined.
            rt = ref_time if ref_time.tzinfo else ref_time.replace(
                tzinfo=timezone.utc
            )
            intervals = await self._call(text, rt)
            self._cache[key] = intervals
            self._dirty = True
        if not intervals:
            return []
        # Dedupe
        seen: set[tuple] = set()
        uniq: list[Interval] = []
        for iv in intervals:
            k = (iv.earliest_us, iv.latest_us)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(iv)
        # Multi-value entities (e.g., recurring Fridays) → ONE
        # IntervalSet with multiple intervals, matching the LLM
        # extractor's multi-interval contract for logically-grouped
        # claims. But across DIFFERENT entities in the same doc, each
        # is its own ref (different IntervalSet) — same as dateparser
        # extractor. We can't tell from the cache which intervals came
        # from which entity, so emit each as its own single-interval
        # IntervalSet. (Acceptable — recurring queries are typically
        # short and produce one entity per recurrence pattern, all of
        # which become separate refs.)
        return [IntervalSet.from_intervals([iv]) for iv in uniq]
