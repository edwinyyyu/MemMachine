"""Qualitative probe: v3.2 vs v3.3 extracted intervals on representative texts.

v3.3 drops `surface` and `granularity` from the LLM output schema and
the prompt's "Output" section. Both fields are unused by retrieval —
only earliest/latest are read.

Hypothesis: removing them won't change which intervals get emitted or
their boundaries (the WIDENING logic remains in the prompt body).
Validate by side-by-side extraction on representative texts spanning:
  - Pinpoint dates
  - Spans (months, quarters)
  - Fuzzy / "around" widening
  - Eras and recurring events
  - Skip cases (policies, generic vocab)

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._probe_extractor_v3_3
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from temporal_retrieval.extractor_v3_2 import TemporalExtractorV3_2
from temporal_retrieval.extractor_v3_3 import TemporalExtractorV3_3

from ._common import setup_env

setup_env()


REF = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

CASES: list[tuple[str, str]] = [
    ("pinpoint",       "Shipped the v5 release on March 15, 2024."),
    ("month",          "Worked on the migration in October 2023."),
    ("quarter",        "Q4 2023 was rough for the team."),
    ("year",           "In 2007 we launched the original product."),
    ("fuzzy_around",   "Around 2008 the team grew to ten people."),
    ("fuzzy_a_few",    "A few weeks ago we shipped the redesign."),
    ("deictic",        "Yesterday I reviewed the new dashboard."),
    ("era",            "Back in college I studied algorithms."),
    ("recurring",      "Every Thursday at 3pm I have therapy."),
    ("anchor_offset",  "Two weeks after the iPhone launched in 2007 we noticed huge interest."),
    ("policy_skip",    "Policy: backups must run within the last hour."),
    ("generic_vocab",  "What happens during the day in a beehive?"),
    ("impact_dur",     "Postmortem from May 12, 2024: the outage lasted 45 minutes."),
    ("multi_te",       "Shipped in Q3 2023; replaced by the v2 in Q4 2024."),
    ("nested_era",     "During the pandemic, in 2020, we adopted remote work."),
]


def _fmt(envs) -> str:
    if not envs:
        return "(no envelopes)"
    parts = []
    for e in envs:
        ea = e.earliest.isoformat().replace("+00:00", "Z")
        la = e.latest.isoformat().replace("+00:00", "Z")
        parts.append(f"[{ea} .. {la})")
    return " ; ".join(parts)


async def probe(name: str, text: str, v2, v3) -> None:
    envs_v2, envs_v3 = await asyncio.gather(
        v2.extract(text, REF),
        v3.extract(text, REF),
    )
    same = len(envs_v2) == len(envs_v3) and all(
        a.earliest == b.earliest and a.latest == b.latest
        for a, b in zip(
            sorted(envs_v2, key=lambda e: (e.earliest, e.latest)),
            sorted(envs_v3, key=lambda e: (e.earliest, e.latest)),
            strict=False,
        )
    )
    marker = "==" if same else "DIFF"
    print(f"\n[{marker}] {name}", flush=True)
    print(f"  text: {text}", flush=True)
    print(f"  v3.2: {_fmt(envs_v2)}", flush=True)
    print(f"  v3.3: {_fmt(envs_v3)}", flush=True)


async def main():
    v2 = TemporalExtractorV3_2()
    v3 = TemporalExtractorV3_3()
    print(f"Probe ref_time: {REF.isoformat()}", flush=True)
    n_same = 0
    for name, text in CASES:
        await probe(name, text, v2, v3)
    # Recount after iteration (simpler than tracking inside probe).
    # We'll just print the diff count from the markers in the output.


if __name__ == "__main__":
    asyncio.run(main())
