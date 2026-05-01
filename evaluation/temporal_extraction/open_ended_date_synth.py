"""Open-ended date benchmark: queries with one-sided temporal bounds.

Each cluster has 5 docs spread across dates. GOLD = doc in the open window;
distractors outside.

Queries CAN contain dates (the open-ended bound IS the date).

Output: data/open_ended_date_{docs,queries,gold}.jsonl
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


# Each cluster: query, gold_dates (in window), distractor_dates (outside),
# topic_template (filled with a date)
# Each doc has the SAME template (so semantic doesn't disambiguate; date does)
CLUSTERS = [
    {
        "query": "What did I work on after 2022?",
        "template": "I worked on the inventory automation project, on {date}.",
        "gold_dates": [(5, 14, 2023)],
        "distractor_dates": [(2, 4, 2021), (8, 22, 2021), (3, 18, 2022), (11, 9, 2022)],
    },
    {
        "query": "What did I work on before the pandemic (January 2020)?",
        "template": "I worked on the customer-segmentation analysis, on {date}.",
        "gold_dates": [(7, 12, 2019)],
        "distractor_dates": [(3, 4, 2020), (9, 18, 2020), (2, 11, 2021), (6, 25, 2022)],
    },
    {
        "query": "What's my activity since I moved in June 2023?",
        "template": "I attended the local neighborhood improvement meetup, on {date}.",
        "gold_dates": [(8, 11, 2023)],
        "distractor_dates": [(2, 4, 2022), (10, 7, 2022), (1, 15, 2023), (4, 23, 2023)],
    },
    {
        "query": "What did I do before I joined Acme in March 2022?",
        "template": "I led a freelance design engagement, on {date}.",
        "gold_dates": [(11, 14, 2021)],
        "distractor_dates": [(5, 18, 2022), (9, 9, 2022), (1, 25, 2023), (4, 4, 2024)],
    },
    {
        "query": "What courses did I complete after graduating in May 2020?",
        "template": "I completed an online course on distributed systems, on {date}.",
        "gold_dates": [(3, 8, 2021)],
        "distractor_dates": [
            (2, 14, 2018),
            (9, 22, 2018),
            (4, 30, 2019),
            (1, 17, 2020),
        ],
    },
    {
        "query": "What investments did I make before retirement (December 2023)?",
        "template": "I added to my index-fund position, on {date}.",
        "gold_dates": [(5, 21, 2022)],
        "distractor_dates": [(2, 4, 2024), (6, 18, 2024), (10, 5, 2024), (3, 30, 2025)],
    },
    {
        "query": "What did I publish since I started the blog in March 2024?",
        "template": "I published a long-form essay on systems design, on {date}.",
        "gold_dates": [(8, 4, 2024)],
        "distractor_dates": [
            (1, 22, 2023),
            (5, 30, 2023),
            (10, 11, 2023),
            (2, 8, 2024),
        ],
    },
    {
        "query": "What trips did I take after my child was born in August 2023?",
        "template": "I took a family trip to the coast, on {date}.",
        "gold_dates": [(6, 18, 2024)],
        "distractor_dates": [
            (3, 14, 2021),
            (8, 7, 2021),
            (5, 22, 2022),
            (11, 30, 2022),
        ],
    },
    {
        "query": "What did I do before I started grad school in September 2021?",
        "template": "I worked as a research assistant in the systems lab, on {date}.",
        "gold_dates": [(4, 19, 2021)],
        "distractor_dates": [
            (10, 11, 2021),
            (3, 17, 2022),
            (7, 5, 2023),
            (12, 1, 2023),
        ],
    },
    {
        "query": "What conferences did I attend since 2024?",
        "template": "I attended a developer-tooling conference, on {date}.",
        "gold_dates": [(9, 12, 2024)],
        "distractor_dates": [(4, 4, 2021), (10, 17, 2022), (3, 9, 2023), (8, 22, 2023)],
    },
    {
        "query": "What treatments did I get before the surgery in November 2024?",
        "template": "I had a follow-up imaging session at the clinic, on {date}.",
        "gold_dates": [(7, 22, 2024)],
        "distractor_dates": [(12, 14, 2024), (1, 22, 2025), (3, 6, 2025), (5, 1, 2025)],
    },
    {
        "query": "What restaurants did I try after the city move (April 2024)?",
        "template": "I tried a new neighborhood ramen restaurant, on {date}.",
        "gold_dates": [(7, 30, 2024)],
        "distractor_dates": [
            (8, 11, 2022),
            (1, 6, 2023),
            (5, 19, 2023),
            (11, 14, 2023),
        ],
    },
    {
        "query": "What books did I read before the book club ended in May 2023?",
        "template": "I finished a non-fiction book about urban policy, on {date}.",
        "gold_dates": [(2, 14, 2023)],
        "distractor_dates": [
            (7, 5, 2023),
            (10, 22, 2023),
            (1, 18, 2024),
            (8, 11, 2024),
        ],
    },
    {
        "query": "What workouts did I do since I joined the gym in January 2025?",
        "template": "I completed an interval training workout, on {date}.",
        "gold_dates": [(4, 7, 2025)],
        "distractor_dates": [(2, 5, 2023), (8, 18, 2023), (4, 22, 2024), (10, 9, 2024)],
    },
    {
        "query": "What apartments did I tour before I signed the lease in October 2023?",
        "template": "I toured a one-bedroom apartment in the east district, on {date}.",
        "gold_dates": [(8, 14, 2023)],
        "distractor_dates": [
            (11, 22, 2023),
            (2, 6, 2024),
            (5, 18, 2024),
            (9, 30, 2024),
        ],
    },
]


def main() -> None:
    rng = random.Random(20260429)
    docs = []
    queries = []
    gold_rows = []

    for i, c in enumerate(CLUSTERS):
        gold_ids = []
        for j, (m, d, y) in enumerate(c["gold_dates"]):
            gid = f"oe_{i:03d}_g{j}"
            gold_ids.append(gid)
            docs.append(
                {
                    "doc_id": gid,
                    "text": c["template"].format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )
        for j, (m, d, y) in enumerate(c["distractor_dates"]):
            did = f"oe_{i:03d}_d{j}"
            docs.append(
                {
                    "doc_id": did,
                    "text": c["template"].format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )

        qid = f"oe_q_{i:03d}"
        queries.append({"query_id": qid, "text": c["query"], "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": gold_ids})

    rng.shuffle(docs)
    with open(DATA_DIR / "open_ended_date_docs.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with open(DATA_DIR / "open_ended_date_queries.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(DATA_DIR / "open_ended_date_gold.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)
    print(
        f"open_ended_date: {len(docs)} docs, {len(queries)} queries, {len(gold_rows)} gold rows"
    )


if __name__ == "__main__":
    main()
