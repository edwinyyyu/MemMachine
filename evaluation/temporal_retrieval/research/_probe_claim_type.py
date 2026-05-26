"""Qualitative probe: does v3.4 extractor reliably tag claim_type?

20 hand-curated passages spanning the three categories with
expected-type labels. We print each text + extracted refs + the
LLM's claim_type tag, and report agreement vs expected.

If agreement is high (≥18/20), the LLM can do this reliably and we
can use claim_type in scoring. If lower, the prompt needs revision.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._probe_claim_type
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from temporal_retrieval.extractor_v3_4 import TemporalExtractorV3_4
from temporal_retrieval.schema import from_us

from ._common import setup_env

setup_env()


REF = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)


# Each case: (id, expected_type_for_each_emitted_ref_in_order, text)
# When multiple refs are expected, list types in extraction order; if
# extraction order varies, we'll just match presence-of-type per
# multi-set.
CASES = [
    # === EVENTS ===
    ("event_join",          ["event"], "Joined TechCorp in June 2012 as a junior developer."),
    ("event_trip",          ["event"], "Italy trip in July 2018 — explored Rome and Florence."),
    ("event_marry",         ["event"], "Got married on July 14, 2016 at the courthouse."),
    ("event_fireworks",     ["event"], "Saw the Boston Pops fireworks July 4, 2025."),
    ("event_deploy",        ["event"], "Deployed the migration yesterday — biggest release of the year."),
    ("event_buy_house",     ["event"], "Bought our first house in March 2019."),
    ("event_release",       ["event"], "Released v5 on March 15, 2024 — the team celebrated."),
    # === STATES ===
    ("state_employed",      ["state"], "Worked at Initech from April 2022 through 2025 as principal engineer."),
    ("state_studying",      ["state"], "Throughout 2024 I was studying for the bar exam — kept me busy all year."),
    ("state_lived",         ["state"], "Lived in California from 2018 onward, after relocating from New York."),
    ("state_lockdown",      ["state"], "Lockdown was March to June 2020 for us in California — at home the whole time."),
    ("state_owned",         ["state"], "Owned the Honda from 2015 until 2021 — kept us on the road."),
    ("state_rotation",      ["state"], "I was on the strategy rotation all of Q1 2024."),
    # === HABITS ===
    ("habit_oatmeal",       ["state"], "I ate oatmeal every weekday morning in 2024."),
    ("habit_yoga",          ["state"], "Started yoga every Tuesday in the fall of 2019."),
    ("habit_gather",        ["state"], "Each December our family gathers for the holidays."),
    ("habit_mentor",        ["state"], "Met with my mentor monthly throughout 2023."),
    # === MIXED / TRICKY ===
    ("mixed_event_state",   ["event", "state"], "Joined BigCo on January 1, 2018; worked there until March 2022."),
    ("edge_short_state",    ["state"], "Was on vacation the week of July 4, 2024."),
    ("edge_event_phrasing", ["event"], "The team retreat happened in October 2022 at the lake house."),
]


def _fmt_iv(us: int) -> str:
    return from_us(us).strftime("%Y-%m-%dT%H:%M:%SZ")


async def main():
    ext = TemporalExtractorV3_4()
    print(f"Probing v3.4 claim_type on {len(CASES)} cases\n", flush=True)
    n_ok_total = 0
    n_total = 0
    per_class = {"event": [0, 0], "state": [0, 0]}  # [correct, total]
    coros = [ext.extract(txt, REF) for _, _, txt in CASES]
    results = await asyncio.gather(*coros)

    for (case_id, expected_types, text), refs in zip(CASES, results, strict=False):
        emitted_types = [r["claim_type"] for r in refs]
        # Order-independent type-multiset match:
        exp_sorted = sorted(expected_types)
        emit_sorted = sorted(emitted_types)
        ok = exp_sorted == emit_sorted
        marker = "OK  " if ok else "MISS"
        print(f"[{marker}] {case_id}", flush=True)
        print(f"  text:      {text}")
        print(f"  expected:  {expected_types}")
        print(f"  emitted:   {emitted_types}")
        for r in refs:
            print(f"    [{r['claim_type']:5s}] {_fmt_iv(r['earliest_us'])}  ..  {_fmt_iv(r['latest_us'])}")
        # Per-class accounting (just count first expected type as the "kind" of case)
        kind = expected_types[0] if expected_types else "event"
        per_class[kind][1] += 1
        if ok:
            per_class[kind][0] += 1
            n_ok_total += 1
        n_total += 1
        print(flush=True)

    print(f"\n=== SUMMARY: {n_ok_total}/{n_total} cases match expected types ===", flush=True)
    for kind, (ok, tot) in per_class.items():
        if tot:
            print(f"  {kind:5s}: {ok}/{tot}", flush=True)
    ext.save_caches()


if __name__ == "__main__":
    asyncio.run(main())
