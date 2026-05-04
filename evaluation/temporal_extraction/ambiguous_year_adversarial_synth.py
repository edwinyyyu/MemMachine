"""Adversarial extension of the ambiguous_year benchmark.

Tests fusion behavior on year-unspecified queries that use NON-CANONICAL
forms of the temporal phrase — the kind regex-based phrase-class gating
falls over on, but a smart classifier (LLM) should handle.

Categories:
  - misspellings (English): "Marh", "sumer"
  - multilingual periods: "en marzo" (es), "en été" (fr), "in oktober" (de)
  - non-Western recurring observances: Lunar New Year, Ramadan, Diwali
  - era references: "during the pandemic"

Each cluster keeps the same shape as the base bench: 3 gold (one per
year/instance), 2 same-topic-other-period, 2 other-topic-target-period,
2 noise.

Output: data/ambiguous_year_adv_{docs,queries,gold}.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2025-06-15T00:00:00Z"


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
#   query, topic_template, gold_dates [(m,d,y)], same_topic_other_period_dates,
#   other_topic_target_period [(template,m,d,y)], noise [(template,m,d,y)]
ADVERSARIAL_CLUSTERS = [
    # 1. Misspelled month — regex won't match "marh"
    {
        "query": "What did I work on in Marh?",
        "topic_template": "I worked on the inventory automation project, on {date}.",
        "gold_dates": [(3, 12, 2023), (3, 18, 2024), (3, 14, 2025)],
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
    # 2. Misspelled season — "sumer"
    {
        "query": "What trips did I take in sumer?",
        "topic_template": "I took a road trip up the Pacific Coast Highway, on {date}.",
        "gold_dates": [(7, 14, 2023), (8, 22, 2024), (6, 30, 2025)],
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
    # 3. Multilingual: Spanish month — "en marzo"
    {
        "query": "What conferences did I attend en marzo?",
        "topic_template": "I attended the annual industry conference in Berlin, on {date}.",
        "gold_dates": [(3, 17, 2023), (3, 22, 2024), (3, 9, 2025)],
        "same_topic_other_period_dates": [
            (
                "I attended the annual industry conference in Berlin, on {date}.",
                9,
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
            ("I went pumpkin picking with the kids, on {date}.", 3, 14, 2024),
            ("I had a routine eye exam, on {date}.", 3, 5, 2023),
        ],
        "noise": [
            ("Replaced the kitchen faucet, on {date}.", 2, 11, 2024),
            ("Booked a babysitter for date night, on {date}.", 5, 22, 2023),
        ],
    },
    # 4. Multilingual: French season — "en été"
    {
        "query": "What did I publish en été?",
        "topic_template": "I published a long-form essay on systems design, on {date}.",
        "gold_dates": [(7, 14, 2023), (8, 21, 2024), (6, 7, 2025)],
        "same_topic_other_period_dates": [
            (
                "I published a long-form essay on systems design, on {date}.",
                11,
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
            ("I attended a backyard BBQ with neighbors, on {date}.", 7, 12, 2024),
            ("I planted tomatoes in the backyard garden, on {date}.", 6, 18, 2023),
        ],
        "noise": [
            ("Got my hair cut at the new place downtown, on {date}.", 9, 22, 2024),
            ("Filled the car up with gas, on {date}.", 3, 6, 2023),
        ],
    },
    # 5. Multilingual: German month — "im Oktober"
    {
        "query": "Who did I meet im Oktober?",
        "topic_template": "I met with my financial advisor for the annual review, on {date}.",
        "gold_dates": [(10, 14, 2023), (10, 22, 2024), (10, 8, 2025)],
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
                10,
                6,
                2024,
            ),
            (
                "I attended a friend's birthday dinner downtown, on {date}.",
                10,
                19,
                2023,
            ),
        ],
        "noise": [
            ("Replaced a flat tire on my bike, on {date}.", 7, 11, 2024),
            ("Reorganized the home office desk, on {date}.", 2, 4, 2023),
        ],
    },
    # 6. Non-Western recurring: Lunar New Year (~Jan/Feb)
    {
        "query": "What did I cook for Lunar New Year?",
        "topic_template": "I cooked a hand-pulled noodle dish for the family gathering, on {date}.",
        # Real Lunar New Year dates: 2023-01-22, 2024-02-10, 2025-01-29
        "gold_dates": [(1, 22, 2023), (2, 10, 2024), (1, 29, 2025)],
        "same_topic_other_period_dates": [
            (
                "I cooked a hand-pulled noodle dish for the family gathering, on {date}.",
                7,
                4,
                2023,
            ),
            (
                "I cooked a hand-pulled noodle dish for the family gathering, on {date}.",
                11,
                28,
                2024,
            ),
        ],
        "other_topic_target_period": [
            (
                "I attended the city's lantern festival downtown, on {date}.",
                2,
                11,
                2024,
            ),
            ("I went on a winter hike with friends, on {date}.", 1, 28, 2023),
        ],
        "noise": [
            ("Booked a haircut appointment, on {date}.", 6, 18, 2024),
            ("Picked up a new houseplant, on {date}.", 9, 5, 2023),
        ],
    },
    # 7. Non-Western recurring: Ramadan (~Mar-Apr)
    {
        "query": "What did I cook during Ramadan?",
        "topic_template": "I cooked an iftar dinner with friends from the mosque, on {date}.",
        # Real Ramadan dates: 2023 starts Mar 22, 2024 starts Mar 11, 2025 starts Feb 28
        "gold_dates": [(4, 5, 2023), (3, 25, 2024), (3, 15, 2025)],
        "same_topic_other_period_dates": [
            (
                "I cooked an iftar dinner with friends from the mosque, on {date}.",
                8,
                18,
                2023,
            ),
            (
                "I cooked an iftar dinner with friends from the mosque, on {date}.",
                12,
                22,
                2024,
            ),
        ],
        "other_topic_target_period": [
            ("I attended the spring book fair downtown, on {date}.", 4, 12, 2023),
            ("I went on a community service trip, on {date}.", 3, 18, 2024),
        ],
        "noise": [
            ("Cleaned out the garage, on {date}.", 6, 14, 2023),
            ("Bought new running shoes online, on {date}.", 9, 11, 2024),
        ],
    },
    # 8. Non-Western recurring: Diwali (Oct/Nov)
    {
        "query": "What did I do for Diwali?",
        "topic_template": "I hosted a Diwali dinner for the neighbors, on {date}.",
        # Real Diwali dates: 2023-11-12, 2024-11-01, 2025-10-21
        "gold_dates": [(11, 12, 2023), (11, 1, 2024), (10, 21, 2025)],
        "same_topic_other_period_dates": [
            ("I hosted a Diwali dinner for the neighbors, on {date}.", 4, 14, 2024),
            ("I hosted a Diwali dinner for the neighbors, on {date}.", 7, 22, 2023),
        ],
        "other_topic_target_period": [
            (
                "I attended a Halloween costume party with friends, on {date}.",
                10,
                28,
                2024,
            ),
            ("I went pumpkin picking with the family, on {date}.", 11, 5, 2023),
        ],
        "noise": [
            ("Bought groceries for the week, on {date}.", 5, 8, 2024),
            ("Got an oil change for the car, on {date}.", 3, 12, 2023),
        ],
    },
    # 9. Era reference — "during the pandemic" (2020-early 2022)
    {
        "query": "What courses did I take during the pandemic?",
        "topic_template": "I completed an online course on distributed systems, on {date}.",
        "gold_dates": [(5, 22, 2020), (11, 14, 2020), (3, 8, 2021)],
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
            ("I attended a small backyard wedding, on {date}.", 8, 4, 2020),
            ("I went cross-country skiing for the weekend, on {date}.", 2, 18, 2021),
        ],
        "noise": [
            ("Sorted through old tax receipts, on {date}.", 5, 5, 2024),
            ("Paid the rent on time, on {date}.", 10, 1, 2023),
        ],
    },
    # 10. Compound: "two summers ago" — actually less ambiguous (specific
    # past summer), included to test classifier's ability to distinguish
    # specific deictic from genuinely ambiguous. Should resolve to summer 2023.
    # Single gold (this is a control case, not a fusion case).
    {
        "query": "Where did I camp two summers ago?",
        "topic_template": "I camped at the redwood state park with old friends, on {date}.",
        "gold_dates": [(7, 18, 2023)],  # only one — specific past summer
        "same_topic_other_period_dates": [
            (
                "I camped at the redwood state park with old friends, on {date}.",
                7,
                22,
                2022,
            ),
            (
                "I camped at the redwood state park with old friends, on {date}.",
                8,
                4,
                2024,
            ),
            (
                "I camped at the redwood state park with old friends, on {date}.",
                6,
                30,
                2025,
            ),
        ],
        "other_topic_target_period": [
            ("I attended an outdoor music festival, on {date}.", 8, 12, 2023),
        ],
        "noise": [
            ("Refinanced the mortgage, on {date}.", 4, 14, 2024),
            ("Subscribed to a new streaming service, on {date}.", 1, 6, 2024),
        ],
    },
    # 11. Phrasal year — "two thousand twenty-four" (year IS specific)
    {
        "query": "What did I work on in two thousand twenty-four?",
        "topic_template": "I led a Q1 team retro on velocity and blockers, on {date}.",
        "gold_dates": [(2, 14, 2024), (5, 22, 2024), (10, 8, 2024)],
        "same_topic_other_period_dates": [
            ("I led a Q1 team retro on velocity and blockers, on {date}.", 7, 18, 2023),
            ("I led a Q1 team retro on velocity and blockers, on {date}.", 4, 22, 2025),
        ],
        "other_topic_target_period": [
            ("I led a hiring sync on Q1 onboarding plans, on {date}.", 3, 18, 2024),
            ("I attended an industry summit, on {date}.", 8, 14, 2024),
        ],
        "noise": [
            ("Replaced the kitchen light bulb, on {date}.", 8, 5, 2023),
            ("Took the cat to the vet for shots, on {date}.", 5, 22, 2025),
        ],
    },
    # 12. Mixed-script multilingual: "in 春" (Chinese spring)
    {
        "query": "What restaurants did I try in 春?",
        "topic_template": "I tried a new neighborhood ramen restaurant, on {date}.",
        "gold_dates": [(4, 7, 2023), (5, 19, 2024), (3, 30, 2025)],
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


def main() -> None:
    rng = random.Random(20260504)
    docs = []
    queries = []
    gold_rows = []

    for i, c in enumerate(ADVERSARIAL_CLUSTERS):
        gold_ids = []
        for j, (m, d, y) in enumerate(c["gold_dates"]):
            gid = f"ayadv_{i:03d}_g{j}"
            gold_ids.append(gid)
            docs.append(
                {
                    "doc_id": gid,
                    "text": c["topic_template"].format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )
        for j, (template, m, d, y) in enumerate(c["same_topic_other_period_dates"]):
            did = f"ayadv_{i:03d}_st{j}"
            docs.append(
                {
                    "doc_id": did,
                    "text": template.format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )
        for j, (template, m, d, y) in enumerate(c["other_topic_target_period"]):
            did = f"ayadv_{i:03d}_ot{j}"
            docs.append(
                {
                    "doc_id": did,
                    "text": template.format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )
        for j, (template, m, d, y) in enumerate(c["noise"]):
            did = f"ayadv_{i:03d}_n{j}"
            docs.append(
                {
                    "doc_id": did,
                    "text": template.format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )

        qid = f"ayadv_q_{i:03d}"
        queries.append({"query_id": qid, "text": c["query"], "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": gold_ids})

    rng.shuffle(docs)
    with open(DATA_DIR / "ambiguous_year_adv_docs.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with open(DATA_DIR / "ambiguous_year_adv_queries.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(DATA_DIR / "ambiguous_year_adv_gold.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)
    print(
        f"ambiguous_year_adv: {len(docs)} docs, {len(queries)} queries, "
        f"{sum(len(g['relevant_doc_ids']) for g in gold_rows)} gold judgments"
    )


if __name__ == "__main__":
    main()
