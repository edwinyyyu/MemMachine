"""Tests for final_score graded coverage."""
from __future__ import annotations

from datetime import datetime, timezone

from temporal_retrieval_tr import IntervalSet, final_score


def us(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1_000_000)


def test_final_score_graded_coverage() -> None:
    """Flat-list scoring rewards graded coverage. Doc matching all targets
    → 1.0; doc matching half → 0.5; none → 0. Emergent from cross-target
    mean — no separate 'and_incompat' machinery needed."""
    y2020 = IntervalSet.closed(us(2020, 1, 1), us(2021, 1, 1))
    y2024 = IntervalSet.closed(us(2024, 1, 1), us(2025, 1, 1))
    targets = [y2020, y2024]

    mar2020 = IntervalSet.closed(us(2020, 3, 1), us(2020, 4, 1))
    jun2024 = IntervalSet.closed(us(2024, 6, 1), us(2024, 7, 1))

    assert final_score(targets, [mar2020, jun2024]) == 1.0
    assert final_score(targets, [mar2020]) == 0.5
    assert final_score(targets, [jun2024]) == 0.5
    assert final_score(targets, [IntervalSet.closed(us(2022, 1, 1), us(2023, 1, 1))]) == 0.0
