"""Qualitative critical-case tester for v3 single-pass extractor.

Same 16 cases as v2.1's qual test. Tests whether single-pass extraction
preserves the v2.1 prompt's fundamental viability — all 10 positives
emit sensible envelopes, all 6 negatives skip correctly.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._qual_v3_critical
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from temporal_retrieval.extractor_v3_2 import TemporalExtractorV3_2
from temporal_retrieval.schema import parse_iso

REF = "2024-06-15T10:00:00Z"

CASES = [
    ("P1_pinpoint", "We shipped v2.0 on March 15, 2024.",
     "narrow [Mar 15 2024, Mar 16) day envelope"),
    ("P2_span_quarter", "Q1 2024 was rough for the team.",
     "[Jan 1, Apr 1) quarter"),
    ("P3_fuzzy_year", "I bought my first iPhone around 2008.",
     "year envelope ~2008 ± 2"),
    ("P4_deictic_yesterday", "Yesterday we deployed the new search backend.",
     "day envelope of ref-1d"),
    ("P5_decade", "Back in the 90s, dial-up was the norm.",
     "[1990, 2000) decade"),
    ("P6_compound_week", "We took the first week of April 2024 off.",
     "[Apr 1 2024, Apr 8 2024) week"),
    ("P7_relative_offset", "Three weeks ago we kicked off the migration.",
     "day envelope at ref-21d"),
    ("P8_anaphoric_month", "Earlier this month we restructured the team.",
     "earlier portion of current month"),
    ("P9_recurrence_first", "I have therapy every Thursday at 3pm.",
     "first/nearest Thursday envelope"),
    ("P10_era", "During the pandemic everything moved remote.",
     "broad 2020-2022-ish envelope"),
    ("N1_policy_duration",
     "Production policy: every release requires a 30-minute monitoring window post-deploy.",
     "SKIP all — generic policy"),
    ("N2_policy_deictic",
     "Database migration policy: never run schema changes without a backup confirmed within the last hour.",
     "SKIP all — generic policy constraint"),
    ("N3_template",
     "Subject line format for outage notifications: '[OUTAGE] [SystemName] [Date] - [Status]'.",
     "SKIP [Date] — template literal"),
    ("N4_frequency",
     "Convention: we sometimes run fire drills in the office.",
     "SKIP 'sometimes' — bare frequency"),
    ("N5_descriptor",
     "The modern era of web frameworks started with React.",
     "SKIP 'modern era' — vague descriptor"),
    ("N6_unanchored_recurrence",
     "Christmas is always magical with the family.",
     "SKIP 'Christmas', 'always' — no year anchor, frequency adverb"),
    # v3.1 impact-duration cases
    ("N8_impact_dur_postmortem",
     "Postmortem from May 12: APM was double-counting latency. Impact: dashboards over-reported by 2x for 6 weeks.",
     "EMIT 'May 12' as the anchor; SKIP 'for 6 weeks' — impact duration"),
    ("N9_impact_dur_freeze",
     "Deploy froze production for 12 minutes due to config typo.",
     "SKIP 'for 12 minutes' — no calendar anchor, just incident duration"),
    ("P16_anchored_duration_with_start",
     "I'll be on vacation for 3 weeks starting June 1, 2024.",
     "EMIT 'June 1, 2024' + envelope spanning 3 weeks (anchored)"),
    # General-doc / specific-query interplay (added after probe found the
    # classifier was silencing year-less calendar phrases on queries like
    # "Where was Bob on July 23?"). These verify the extractor itself
    # produces the right envelopes for both sides; classifier-free routing
    # then turns those into mask anchors.
    ("P17_doc_all_summer",
     "I'll be working at a youth summer camp all summer.",
     "EMIT a summer envelope anchored to ref_time's year (e.g. [Jun, Sep) 2024)"),
    ("P18_doc_specific_day_anchor",
     "I'm going to a fireworks show in my hometown on July 4th.",
     "EMIT [Jul 4 2024, Jul 5 2024) day envelope"),
    ("Q1_query_year_implicit_day",
     "July 23",
     "Treated as deictic against ref_time -> EMIT [Jul 23 2024, Jul 24 2024)"),
    ("Q2_query_year_implicit_month",
     "in July",
     "Treated as deictic against ref_time -> EMIT [Jul 1 2024, Aug 1 2024)"),
    ("Q3_query_bare_month",
     "July",
     "Bare month, no preposition or specific day -> REFUSE (skip)"),
    ("Q4_query_anaphoric_event",
     "the launch",
     "Event reference with no calendar info -> REFUSE (skip)"),
]


async def main() -> None:
    extractor = TemporalExtractorV3_2(
        cache_dir=Path(__file__).resolve().parent.parent / "cache" / "qual_v3_2",
    )
    rt = parse_iso(REF)

    print(f"REF: {REF}\n")
    results = []
    for cid, passage, expected in CASES:
        envs = await extractor.extract(passage, rt)
        emits = [
            {
                "surface": e.surface,
                "earliest": e.earliest.isoformat(),
                "latest": e.latest.isoformat(),
                "granularity": e.granularity,
            }
            for e in envs
        ]
        print(f"=== {cid} ===")
        print(f"  passage : {passage}")
        print(f"  expected: {expected}")
        print(f"  v3 emits: {len(emits)} ref(s)")
        for e in emits:
            print(f"    - surface='{e['surface']}' env=[{e['earliest']}, {e['latest']}) "
                  f"gran={e['granularity']}")
        print()
        results.append({"id": cid, "passage": passage, "expected": expected, "emits": emits})

    out = Path(__file__).resolve().parent.parent / "qual_v3_critical.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
