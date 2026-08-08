"""Research-side adapter for the legacy v3.3 single-interval extractor.

The reference impl in temporal_retrieval_tr expects every extractor to
expose `extract_anchors -> list[IntervalSet]`. The production extractor
in temporal_retrieval_min (v3.3) only has `extract -> list[Interval]`.

To compare the new multi-interval extractor against the v3.3 baseline
without polluting the reference impl with a fallback path, this
research-side adapter wraps v3.3 to conform to the new interface.

It is NOT exported by the reference impl. Tests / benches that want
the v3.3 baseline import this adapter explicitly.
"""
from __future__ import annotations

from datetime import datetime

from temporal_retrieval_min.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval_tr.time_range import Interval, IntervalSet


class V33LegacyExtractorAdapter:
    """Wraps v3.3 (flat list[Interval]) to expose extract_anchors.

    Each emitted envelope becomes a singleton IntervalSet — preserving
    the historical v3.3 behavior under the new interface. This is the
    "wrong" implementation in the sense the user objected to (always
    singletons, can't express multi-interval doc claims), but it's
    useful as the historical baseline for A/B comparisons.
    """

    def __init__(self, base: TemporalExtractorV3_3 | None = None) -> None:
        self._base = base or TemporalExtractorV3_3()

    async def extract_anchors(
        self, text: str, ref_time: datetime
    ) -> list[IntervalSet]:
        out = await self._base.extract(text, ref_time)
        ivs = out[0] if isinstance(out, tuple) else out
        anchors: list[IntervalSet] = []
        for iv in ivs:
            if iv.latest_us > iv.earliest_us:
                anchors.append(
                    IntervalSet(intervals=(Interval(iv.earliest_us, iv.latest_us),))
                )
        return anchors

    def save_caches(self) -> None:
        self._base.save_caches()
