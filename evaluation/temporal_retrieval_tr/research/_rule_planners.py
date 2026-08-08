"""Rule-based planners for the bench: same logic as the rule-based
extractors, applied to the QUERY text. extremum=None always.

Used to measure dateparser/Duckling with BOTH sides rule-based
(no LLM anywhere in the temporal pipeline).
"""
from __future__ import annotations

from datetime import datetime

from temporal_retrieval_tr.planner import Plan
from temporal_retrieval_min.schema import parse_iso

from ._dateparser_extractor import DateparserExtractor
from ._duckling_extractor import DucklingHTTPExtractor


class DateparserPlanner:
    def __init__(self) -> None:
        self._extractor = DateparserExtractor()
        self._cache: dict[str, Plan] = {}

    async def plan(self, query: str, ref_time: str) -> Plan:
        key = f"{query}|{ref_time}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            rt = parse_iso(ref_time)
        except Exception as e:
            return Plan(parse_error=str(e))
        targets = await self._extractor.extract_anchors(query, rt)
        plan = Plan(targets=targets, extremum=None)
        self._cache[key] = plan
        return plan


class DucklingPlanner:
    def __init__(self) -> None:
        self._extractor = DucklingHTTPExtractor()
        self._cache: dict[str, Plan] = {}

    def save_caches(self) -> None:
        self._extractor.save_caches()

    async def plan(self, query: str, ref_time: str) -> Plan:
        key = f"{query}|{ref_time}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            rt = parse_iso(ref_time)
        except Exception as e:
            return Plan(parse_error=str(e))
        targets = await self._extractor.extract_anchors(query, rt)
        plan = Plan(targets=targets, extremum=None)
        self._cache[key] = plan
        return plan
