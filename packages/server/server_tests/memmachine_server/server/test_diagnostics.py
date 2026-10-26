"""Tests for the event-loop diagnostics."""

import asyncio
import contextlib
import time
from unittest.mock import MagicMock

import pytest

from memmachine_server.common.metrics_factory import MetricsFactory
from memmachine_server.server import diagnostics
from memmachine_server.server.diagnostics import report_event_loop_lag


def _keep_the_loop_busy(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pass


def _start_lag_reporting(
    *, worker_count: int
) -> tuple[asyncio.Task[None], MagicMock, MagicMock]:
    """Run report_event_loop_lag against mocked metrics: (task, lag, lag ratio)."""
    lag = MagicMock()
    lag_ratio = MagicMock()
    metrics_factory = MagicMock(spec=MetricsFactory)
    metrics_factory.get_histogram.return_value = lag
    metrics_factory.get_gauge.return_value = lag_ratio
    task = asyncio.create_task(
        report_event_loop_lag(metrics_factory, worker_count=worker_count)
    )
    return task, lag, lag_ratio


async def _stop(task: asyncio.Task[None]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _hold_the_loop_past_a_wakeup(seconds: float) -> None:
    # Let the reporter schedule its next wakeup, hold the loop past it, then let
    # the reporter observe how late it ran.
    await asyncio.sleep(0)
    _keep_the_loop_busy(seconds)
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_a_busy_loop_is_reported_as_lag():
    task, lag, _ = _start_lag_reporting(worker_count=1)
    try:
        await _hold_the_loop_past_a_wakeup(0.5)
    finally:
        await _stop(task)

    observed = [observation.args[0] for observation in lag.observe.call_args_list]
    assert max(observed) >= 0.35
    assert min(observed) >= 0


@pytest.mark.asyncio
async def test_the_lag_ratio_covers_only_the_trailing_window(monkeypatch):
    monkeypatch.setattr(diagnostics, "_EVENT_LOOP_LAG_RATIO_WINDOW_SECONDS", 1)
    task, _, lag_ratio = _start_lag_reporting(worker_count=1)
    try:
        await _hold_the_loop_past_a_wakeup(0.5)
        assert 0.35 <= lag_ratio.set.call_args.args[0] <= 0.5

        await asyncio.sleep(1.2)
        assert lag_ratio.set.call_args.args[0] < 0.05
    finally:
        await _stop(task)


@pytest.mark.asyncio
async def test_a_stall_longer_than_the_window_reports_the_whole_window(monkeypatch):
    monkeypatch.setattr(diagnostics, "_EVENT_LOOP_LAG_RATIO_WINDOW_SECONDS", 1)
    task, _, lag_ratio = _start_lag_reporting(worker_count=1)
    try:
        await _hold_the_loop_past_a_wakeup(1.5)
    finally:
        await _stop(task)

    assert lag_ratio.set.call_args.args[0] == pytest.approx(1)


@pytest.mark.asyncio
async def test_each_worker_reports_its_share_of_the_mean_lag_ratio(monkeypatch):
    monkeypatch.setattr(diagnostics, "_EVENT_LOOP_LAG_RATIO_WINDOW_SECONDS", 1)
    task, _, lag_ratio = _start_lag_reporting(worker_count=4)
    try:
        await _hold_the_loop_past_a_wakeup(1.5)
    finally:
        await _stop(task)

    assert lag_ratio.set.call_args.args[0] == pytest.approx(0.25)
