"""Qualitative critical-case tester for v2 extractor prompt.

Goal: before any benchmark, verify the v2 unified-envelope prompt handles
the fundamental categories correctly. Failures in this sweep are
must-fix; the prompt isn't viable for benchmarking otherwise.

Categories:
  POSITIVE (should emit a sensible envelope):
    P1 pinpoint date         "What happened on March 15, 2024?"   -> [Mar 15, Mar 16) day
    P2 wide span             "Q1 2024 retrospective"              -> [Jan 1, Apr 1) quarter
    P3 fuzzy near anchor     "around 2008"                         -> [2006, 2011) year
    P4 deictic-relative      "yesterday"                           -> [ref-1d, ref) day
    P5 decade era            "the 90s"                             -> [1990, 2000) decade
    P6 compound week         "the first week of April 2024"        -> [Apr 1, Apr 8) week
    P7 relative offset       "3 weeks ago"                         -> day env at ref-21d
    P8 anaphoric within-passage "earlier this month"               -> [month-start, ref) month
    P9 recurrence-as-anchor  "every Thursday at 3pm"               -> first/next Thursday
    P10 era                  "during the pandemic"                 -> some 2020-2022 envelope

  NEGATIVE (should NOT emit — policy/template/generic):
    N1 policy duration       "30-minute monitoring window post-deploy"  -> SKIP
    N2 policy deictic        "backup confirmed within the last hour"    -> SKIP
    N3 template literal      "Subject: [Date]"                          -> SKIP
    N4 bare frequency        "we sometimes run drills"                  -> SKIP
    N5 vague descriptor      "the modern era"                           -> SKIP
    N6 unanchored recurring  "Christmas is always fun"                  -> SKIP

Each case is one short passage with a clear correct answer. Uses a
separate cache so it doesn't pollute the validation cache.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._qual_v2_critical
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from temporal_retrieval.extractor_v2 import TemporalExtractorV2
from temporal_retrieval.schema import parse_iso

REF = "2024-06-15T10:00:00Z"

CASES = [
    # POSITIVE
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
    # NEGATIVE
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
    # v2.2 additions — reported speech / bounded recurrence
    ("P11_reported_speech_this_month",
     "In 2020, my sister said 'this month has been rough'.",
     "EMIT 'In 2020' as year; 'this month' anchors to April 2020 (not ref month)"),
    ("P12_bounded_recurrence_year",
     "The third Saturday of every odd-numbered month in 2024 we have the open house.",
     "EMIT full 2024 bound [Jan 1 2024, Jan 1 2025), not just first Saturday"),
    ("P13_bounded_recurrence_until",
     "Weekly retros every Friday until June 2024.",
     "EMIT range up to June 2024 (around ref time start, ending June 2024)"),
    ("P14_unbounded_recurrence",
     "I have therapy every Thursday at 3pm.",
     "EMIT nearest Thursday (narrow envelope) - unchanged from v2.1"),
    ("P15_past_narrative_deictic",
     "Back in 2015 I wrote that next year would be the breakthrough.",
     "EMIT 'Back in 2015' as year; 'next year' anchors to 2016 (not ref+1)"),
    ("N7_quoted_unanchored_deictic",
     "She always says 'tomorrow will be better'.",
     "SKIP — 'tomorrow' in habitual reported speech has no concrete anchor"),
]


async def main() -> None:
    extractor = TemporalExtractorV2(
        cache_dir=Path(__file__).resolve().parent.parent / "cache" / "qual_v2",
    )
    rt = parse_iso(REF)

    print(f"REF: {REF}\n")
    results = []
    for cid, passage, expected in CASES:
        tes = await extractor.extract(passage, rt)
        emits = [
            {
                "surface": te.surface,
                "earliest": te.instant.earliest.isoformat() if te.instant else None,
                "latest": te.instant.latest.isoformat() if te.instant else None,
                "granularity": te.instant.granularity if te.instant else None,
                "confidence": te.confidence,
            }
            for te in tes
        ]
        print(f"=== {cid} ===")
        print(f"  passage : {passage}")
        print(f"  expected: {expected}")
        print(f"  v2 emits: {len(emits)} ref(s)")
        for e in emits:
            print(f"    - surface='{e['surface']}' env=[{e['earliest']}, {e['latest']}) "
                  f"gran={e['granularity']} conf={e['confidence']:.2f}")
        print()
        results.append({"id": cid, "passage": passage, "expected": expected, "emits": emits})

    out = Path(__file__).resolve().parent.parent / "qual_v2_critical.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
