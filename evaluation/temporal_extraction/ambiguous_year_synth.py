"""Ambiguous-year benchmark: queries with year-unspecified temporal phrases.

Tests whether the retrieval system FUSES across plausible year completions
(e.g. "What did I work on in March?" should surface March 2023 + March 2024
+ March 2025 docs, not just one) when the LLM upstream has decided NOT to
ask the user for clarification.

Design:
  ref_time = 2025-06-15
  Each cluster has 3 GOLD docs: same topic, same target month/season,
  spread across 2023 / 2024 / 2025 (all in the past, all equally plausible
  given the deictic phrasing).

  Distractors:
    - same_topic_other_period: same topic, different month/season, in
      gold-set years (forces the system to honor the temporal cue;
      pure-semantic would surface these too)
    - other_topic_target_period: different topic, target month/season
      (forces the system to honor the topical cue; pure-temporal would
      surface these too)
    - noise: different topic, different period

A hard-pruning system that resolves "March" deictically to a single year
caps at all_recall = 1/3. A fusion system that ORs across years can hit
all_recall = 1.0.

Output: data/ambiguous_year_{docs,queries,gold}.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2025-06-15T00:00:00Z"
GOLD_YEARS = (2023, 2024, 2025)  # all in the past relative to ref_time


def fmt_date(month: int, day: int, year: int) -> str:
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    return f"{months[month - 1]} {day}, {year}"


def iso(month: int, day: int, year: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}T12:00:00Z"


# Each cluster:
#   query : ambiguous-year temporal query
#   topic_template : f-string, single occurrence of {date}; SAME wording
#                    across all 3 gold docs so semantics collapses
#   target_period : (month_low, month_high) inclusive — defines the
#                    "March", "summer", etc. window
#   gold_dates : (m, d) per gold year (2023, 2024, 2025)
#   same_topic_other_period_dates : [(m, d, year), ...] — same topic,
#                    OUTSIDE target_period
#   other_topic_target_period : list of (template, m, d, year) — different
#                    topic, INSIDE target_period
#   noise : list of (template, m, d, year) — different topic, different period
CLUSTERS = [
    {
        "query": "What did I work on in March?",
        "topic_template": "I worked on the inventory automation project, on {date}.",
        "target_period": (3, 3),
        "gold_dates": [(3, 12), (3, 18), (3, 14)],
        "same_topic_other_period_dates": [
            ("I worked on the inventory automation project, on {date}.", 7, 5, 2023),
            ("I worked on the inventory automation project, on {date}.", 11, 9, 2024),
        ],
        "other_topic_target_period": [
            ("I attended a kids' soccer practice, on {date}.", 3, 22, 2023),
            ("I scheduled a dental cleaning, on {date}.", 3, 8, 2024),
        ],
        "noise": [
            ("Picked up a package from the post office, on {date}.", 8, 11, 2024),
            ("Renewed my driver's license, on {date}.", 1, 6, 2024),
        ],
    },
    {
        "query": "What conferences did I attend in October?",
        "topic_template": "I attended the annual industry conference in Berlin, on {date}.",
        "target_period": (10, 10),
        "gold_dates": [(10, 17), (10, 22), (10, 9)],
        "same_topic_other_period_dates": [
            (
                "I attended the annual industry conference in Berlin, on {date}.",
                4,
                4,
                2023,
            ),
            (
                "I attended the annual industry conference in Berlin, on {date}.",
                6,
                18,
                2024,
            ),
        ],
        "other_topic_target_period": [
            ("I went pumpkin picking with the kids, on {date}.", 10, 14, 2023),
            ("I had a routine eye exam, on {date}.", 10, 5, 2024),
        ],
        "noise": [
            ("Replaced the kitchen faucet, on {date}.", 2, 11, 2024),
            ("Booked a babysitter for date night, on {date}.", 5, 22, 2023),
        ],
    },
    {
        "query": "What trips did I take in summer?",
        "topic_template": "I took a road trip up the Pacific Coast Highway, on {date}.",
        "target_period": (6, 8),
        "gold_dates": [(7, 14), (8, 22), (6, 30)],
        "same_topic_other_period_dates": [
            (
                "I took a road trip up the Pacific Coast Highway, on {date}.",
                11,
                5,
                2023,
            ),
            (
                "I took a road trip up the Pacific Coast Highway, on {date}.",
                2,
                14,
                2025,
            ),
        ],
        "other_topic_target_period": [
            ("I refinished the backyard deck, on {date}.", 7, 8, 2024),
            ("I attended an outdoor concert downtown, on {date}.", 8, 12, 2023),
        ],
        "noise": [
            ("Filed my quarterly tax estimate, on {date}.", 1, 18, 2024),
            ("Went to the parent-teacher conference, on {date}.", 11, 22, 2023),
        ],
    },
    {
        "query": "What was my Q3 status update on the platform migration?",
        "topic_template": "I posted a Q3 status update on the platform migration, on {date}.",
        "target_period": (7, 9),
        "gold_dates": [(8, 11), (9, 18), (7, 30)],
        "same_topic_other_period_dates": [
            (
                "I posted a Q3 status update on the platform migration, on {date}.",
                4,
                4,
                2023,
            ),
            (
                "I posted a Q3 status update on the platform migration, on {date}.",
                1,
                17,
                2025,
            ),
        ],
        "other_topic_target_period": [
            (
                "I posted a Q3 status update on the marketing redesign, on {date}.",
                8,
                7,
                2023,
            ),
            ("I posted a Q3 status update on the hiring plan, on {date}.", 9, 12, 2024),
        ],
        "noise": [
            ("Bought groceries for the week, on {date}.", 5, 5, 2024),
            ("Watched the playoff game with friends, on {date}.", 2, 8, 2024),
        ],
    },
    {
        "query": "What did I do during the winter holidays?",
        "topic_template": "I hosted my parents for the winter holidays, on {date}.",
        "target_period": (12, 12),
        "gold_dates": [(12, 24), (12, 27), (12, 22)],
        "same_topic_other_period_dates": [
            ("I hosted my parents for the winter holidays, on {date}.", 3, 14, 2024),
            ("I hosted my parents for the winter holidays, on {date}.", 6, 18, 2023),
        ],
        "other_topic_target_period": [
            ("I went snowboarding at the local resort, on {date}.", 12, 18, 2023),
            (
                "I attended a New Year's Eve party with neighbors, on {date}.",
                12,
                31,
                2024,
            ),
        ],
        "noise": [
            ("Mowed the lawn before the rain, on {date}.", 5, 11, 2024),
            ("Set up the new home printer, on {date}.", 8, 22, 2023),
        ],
    },
    {
        "query": "Who did I meet with in November?",
        "topic_template": "I met with my financial advisor for the annual review, on {date}.",
        "target_period": (11, 11),
        "gold_dates": [(11, 14), (11, 22), (11, 8)],
        "same_topic_other_period_dates": [
            (
                "I met with my financial advisor for the annual review, on {date}.",
                4,
                30,
                2024,
            ),
            (
                "I met with my financial advisor for the annual review, on {date}.",
                6,
                25,
                2023,
            ),
        ],
        "other_topic_target_period": [
            (
                "I had coffee with a college roommate visiting town, on {date}.",
                11,
                6,
                2024,
            ),
            (
                "I attended a friend's birthday dinner downtown, on {date}.",
                11,
                19,
                2023,
            ),
        ],
        "noise": [
            ("Replaced a flat tire on my bike, on {date}.", 7, 11, 2024),
            ("Reorganized the home office desk, on {date}.", 2, 4, 2023),
        ],
    },
    {
        "query": "What did I publish in May?",
        "topic_template": "I published a long-form essay on systems design, on {date}.",
        "target_period": (5, 5),
        "gold_dates": [(5, 14), (5, 21), (5, 7)],
        "same_topic_other_period_dates": [
            (
                "I published a long-form essay on systems design, on {date}.",
                8,
                12,
                2024,
            ),
            (
                "I published a long-form essay on systems design, on {date}.",
                1,
                22,
                2023,
            ),
        ],
        "other_topic_target_period": [
            (
                "I attended a Mother's Day brunch with the family, on {date}.",
                5,
                12,
                2024,
            ),
            ("I planted tomatoes in the backyard garden, on {date}.", 5, 18, 2023),
        ],
        "noise": [
            ("Got my hair cut at the new place downtown, on {date}.", 9, 22, 2024),
            ("Filled the car up with gas, on {date}.", 3, 6, 2023),
        ],
    },
    {
        "query": "What workouts did I do in winter?",
        "topic_template": "I completed an interval training workout at the gym, on {date}.",
        "target_period": (12, 2),  # season-spanning, handled below
        "season_months": (12, 1, 2),
        "gold_dates": [
            (1, 18, 2023),
            (2, 14, 2024),
            (12, 22, 2024),
        ],  # explicit (m,d,y)
        "season_explicit": True,
        "same_topic_other_period_dates": [
            (
                "I completed an interval training workout at the gym, on {date}.",
                7,
                5,
                2023,
            ),
            (
                "I completed an interval training workout at the gym, on {date}.",
                4,
                11,
                2024,
            ),
        ],
        "other_topic_target_period": [
            ("I went sledding with the kids in the park, on {date}.", 1, 28, 2024),
            ("I attended a winter book club meeting, on {date}.", 2, 6, 2023),
        ],
        "noise": [
            ("Cleaned out the garage, on {date}.", 6, 18, 2023),
            ("Bought new running shoes online, on {date}.", 9, 11, 2024),
        ],
    },
    {
        "query": "What apartment tours did I attend in August?",
        "topic_template": "I toured a one-bedroom apartment in the east district, on {date}.",
        "target_period": (8, 8),
        "gold_dates": [(8, 14), (8, 7), (8, 22)],
        "same_topic_other_period_dates": [
            (
                "I toured a one-bedroom apartment in the east district, on {date}.",
                11,
                22,
                2024,
            ),
            (
                "I toured a one-bedroom apartment in the east district, on {date}.",
                2,
                6,
                2025,
            ),
        ],
        "other_topic_target_period": [
            ("I attended a backyard BBQ with neighbors, on {date}.", 8, 18, 2024),
            ("I went hiking in the regional park, on {date}.", 8, 4, 2023),
        ],
        "noise": [
            ("Picked up dry cleaning, on {date}.", 4, 22, 2024),
            ("Watched a movie at the new theater downtown, on {date}.", 12, 5, 2023),
        ],
    },
    {
        "query": "What courses did I take in February?",
        "topic_template": "I completed an online course on distributed systems, on {date}.",
        "target_period": (2, 2),
        "gold_dates": [(2, 14), (2, 8), (2, 22)],
        "same_topic_other_period_dates": [
            (
                "I completed an online course on distributed systems, on {date}.",
                9,
                22,
                2023,
            ),
            (
                "I completed an online course on distributed systems, on {date}.",
                6,
                11,
                2024,
            ),
        ],
        "other_topic_target_period": [
            ("I attended Valentine's dinner with my partner, on {date}.", 2, 14, 2025),
            ("I went cross-country skiing for the weekend, on {date}.", 2, 18, 2024),
        ],
        "noise": [
            ("Sorted through old tax receipts, on {date}.", 5, 5, 2024),
            ("Paid the rent on time, on {date}.", 10, 1, 2023),
        ],
    },
    {
        "query": "What were my Q1 retros?",
        "topic_template": "I led a Q1 team retro on velocity and blockers, on {date}.",
        "target_period": (1, 3),
        "gold_dates": [(3, 28), (4, 2), (3, 31)],
        "season_explicit": False,
        "same_topic_other_period_dates": [
            ("I led a Q1 team retro on velocity and blockers, on {date}.", 7, 18, 2024),
            (
                "I led a Q1 team retro on velocity and blockers, on {date}.",
                10,
                22,
                2023,
            ),
        ],
        "other_topic_target_period": [
            ("I led a hiring sync on Q1 onboarding plans, on {date}.", 2, 14, 2024),
            ("I led a roadmap planning session for the year, on {date}.", 1, 18, 2023),
        ],
        "noise": [
            ("Replaced the kitchen light bulb, on {date}.", 8, 5, 2024),
            ("Took the cat to the vet for shots, on {date}.", 5, 22, 2023),
        ],
    },
    {
        "query": "What restaurants did I try in spring?",
        "topic_template": "I tried a new neighborhood ramen restaurant, on {date}.",
        "target_period": (3, 5),
        "gold_dates": [(4, 7), (5, 19), (3, 30)],
        "same_topic_other_period_dates": [
            ("I tried a new neighborhood ramen restaurant, on {date}.", 11, 14, 2023),
            ("I tried a new neighborhood ramen restaurant, on {date}.", 8, 11, 2024),
        ],
        "other_topic_target_period": [
            ("I attended a community garden cleanup, on {date}.", 4, 14, 2024),
            ("I went to a kid's soccer tournament, on {date}.", 5, 4, 2023),
        ],
        "noise": [
            ("Replaced the smoke detector batteries, on {date}.", 1, 6, 2024),
            ("Went to a holiday party with coworkers, on {date}.", 12, 14, 2023),
        ],
    },
]


def _is_in_period(month: int, period: tuple[int, int]) -> bool:
    lo, hi = period
    if lo <= hi:
        return lo <= month <= hi
    # wraparound (e.g., winter Dec-Feb)
    return month >= lo or month <= hi


def main() -> None:
    rng = random.Random(20260504)
    docs = []
    queries = []
    gold_rows = []

    for i, c in enumerate(CLUSTERS):
        gold_ids = []
        # Gold docs
        for j, gd in enumerate(c["gold_dates"]):
            if c.get("season_explicit"):
                m, d, y = gd
            else:
                m, d = gd
                y = GOLD_YEARS[j]
            gid = f"ay_{i:03d}_g{j}"
            gold_ids.append(gid)
            docs.append(
                {
                    "doc_id": gid,
                    "text": c["topic_template"].format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )

        # Same-topic, other-period distractors
        for j, (template, m, d, y) in enumerate(c["same_topic_other_period_dates"]):
            assert not _is_in_period(m, c["target_period"]) or c.get(
                "season_explicit"
            ), (
                f"cluster {i}: same-topic distractor m={m} is INSIDE target period "
                f"{c['target_period']}; would contaminate gold"
            )
            did = f"ay_{i:03d}_st{j}"
            docs.append(
                {
                    "doc_id": did,
                    "text": template.format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )

        # Other-topic, target-period distractors
        for j, (template, m, d, y) in enumerate(c["other_topic_target_period"]):
            did = f"ay_{i:03d}_ot{j}"
            docs.append(
                {
                    "doc_id": did,
                    "text": template.format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )

        # Noise
        for j, (template, m, d, y) in enumerate(c["noise"]):
            did = f"ay_{i:03d}_n{j}"
            docs.append(
                {
                    "doc_id": did,
                    "text": template.format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )

        qid = f"ay_q_{i:03d}"
        queries.append({"query_id": qid, "text": c["query"], "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": gold_ids})

    rng.shuffle(docs)
    with open(DATA_DIR / "ambiguous_year_docs.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with open(DATA_DIR / "ambiguous_year_queries.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(DATA_DIR / "ambiguous_year_gold.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)
    print(
        f"ambiguous_year: {len(docs)} docs, {len(queries)} queries, "
        f"{sum(len(g['relevant_doc_ids']) for g in gold_rows)} gold judgments"
    )


if __name__ == "__main__":
    main()
