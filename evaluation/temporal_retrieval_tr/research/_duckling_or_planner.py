"""Variant: emit all Duckling entities as ONE multi-interval target.

The default DucklingPlanner emits each entity as a separate target (AND
semantics in the retriever). This variant emits all entities as a single
target with multiple intervals (OR semantics — set membership).

If the bench is more permissive (gold usually matches ONE of several
periods), OR should help. If queries are typically conjunctive, AND
helps.
"""
from __future__ import annotations

from temporal_retrieval_tr.planner import Plan
from temporal_retrieval_tr.time_range import IntervalSet
from temporal_retrieval_min.schema import parse_iso

from ._duckling_extractor import DucklingHTTPExtractor


class DucklingORPlanner:
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
        entities = await self._extractor.extract_anchors(query, rt)
        # Flatten all intervals into ONE target -> OR / set-membership.
        all_intervals = [iv for ent in entities for iv in ent.intervals]
        if all_intervals:
            targets = [IntervalSet.from_intervals(all_intervals)]
        else:
            targets = []
        plan = Plan(targets=targets, extremum=None)
        self._cache[key] = plan
        return plan
