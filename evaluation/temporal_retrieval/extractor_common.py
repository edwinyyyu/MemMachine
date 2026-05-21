"""Shared helpers for temporal extractors (cache, reference-time context)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

MODEL = "gpt-5-mini"
CACHE_ROOT = Path(__file__).resolve().parent / "cache"
CACHE_ROOT.mkdir(exist_ok=True)


class _LLMCache:
    """On-disk JSON cache keyed by (model, prompt_key) SHA-256."""

    def __init__(self, cache_file: Path) -> None:
        self.path = cache_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        if cache_file.exists():
            with cache_file.open() as f:
                self._cache = json.load(f)
        self._new: dict[str, str] = {}

    @staticmethod
    def _key(model: str, prompt_key: str) -> str:
        return hashlib.sha256(f"{model}|{prompt_key}".encode()).hexdigest()

    def get(self, model: str, prompt_key: str) -> str | None:
        return self._cache.get(self._key(model, prompt_key))

    def put(self, model: str, prompt_key: str, response: str) -> None:
        k = self._key(model, prompt_key)
        self._cache[k] = response
        self._new[k] = response

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, str] = {}
        if self.path.exists():
            with self.path.open() as f:
                existing = json.load(f)
        existing.update(self._new)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(existing, f)
        tmp.replace(self.path)
        self._new.clear()


def full_ref_context(ref_time: datetime) -> str:
    """Verbose deictic context: yesterday/today/tomorrow, this/last/next
    week-month-year-quarter, all relative to `ref_time`. Pass-1 prompt
    embeds this so the LLM doesn't need to recompute deictic anchors
    from a bare ISO timestamp."""
    wk = ref_time.strftime("%A")
    iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    yest = ref_time - timedelta(days=1)
    tom = ref_time + timedelta(days=1)
    iso_weekday = ref_time.isoweekday()
    this_week_start = (ref_time - timedelta(days=iso_weekday - 1)).date()
    this_week_end = this_week_start + timedelta(days=6)
    last_week_start = this_week_start - timedelta(days=7)
    last_week_end = last_week_start + timedelta(days=6)
    next_week_start = this_week_start + timedelta(days=7)
    next_week_end = next_week_start + timedelta(days=6)

    def month_label(dt: datetime) -> str:
        return dt.strftime("%B %Y")

    this_month = month_label(ref_time)
    if ref_time.month == 1:
        last_month_dt = ref_time.replace(year=ref_time.year - 1, month=12, day=1)
    else:
        last_month_dt = ref_time.replace(month=ref_time.month - 1, day=1)
    if ref_time.month == 12:
        next_month_dt = ref_time.replace(year=ref_time.year + 1, month=1, day=1)
    else:
        next_month_dt = ref_time.replace(month=ref_time.month + 1, day=1)
    this_quarter = (ref_time.month - 1) // 3 + 1
    this_year = ref_time.year
    return (
        f"Reference time: {iso_ref} ({wk}).\n"
        f"Today = {ref_time.strftime('%A, %B %-d, %Y')}. "
        f"Yesterday = {yest.strftime('%A, %b %-d, %Y')}. "
        f"Tomorrow = {tom.strftime('%A, %b %-d, %Y')}.\n"
        f"This week = {this_week_start.strftime('%b %-d')}"
        f"-{this_week_end.strftime('%b %-d, %Y')} (Mon-Sun).\n"
        f"Last week = {last_week_start.strftime('%b %-d')}"
        f"-{last_week_end.strftime('%b %-d, %Y')}. "
        f"Next week = {next_week_start.strftime('%b %-d')}"
        f"-{next_week_end.strftime('%b %-d, %Y')}.\n"
        f"This month = {this_month}. "
        f"Last month = {month_label(last_month_dt)}. "
        f"Next month = {month_label(next_month_dt)}.\n"
        f"This quarter = Q{this_quarter} {this_year}. "
        f"This year = {this_year}. Last year = {this_year - 1}. "
        f"Next year = {this_year + 1}."
    )
