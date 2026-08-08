"""Trivial extractor: ignore doc content, emit one anchor at ref_time.

Tests whether doc-side CONTENT extraction adds anything beyond just
trusting the doc's metadata timestamp. If `reftime-only + LLM-plan`
matches `duckling-extr + LLM-plan`, doc content extraction is dead
weight.
"""
from datetime import datetime, timedelta

from temporal_retrieval_min.schema import to_us
from temporal_retrieval_tr.time_range import Interval, IntervalSet


class RefTimeOnlyExtractor:
    """Each doc gets exactly one anchor: [ref_time, ref_time + 1 day)."""

    def __init__(self) -> None:
        pass

    def save_caches(self) -> None:
        pass

    async def extract_anchors(
        self, text: str, ref_time: datetime
    ) -> list[IntervalSet]:
        start = ref_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        iv = Interval(to_us(start), to_us(end))
        return [IntervalSet.from_intervals([iv])]
