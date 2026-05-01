"""Negation-temporal benchmark: queries with explicit time exclusion.

Each cluster: 5 docs spread across dates. GOLD = doc(s) OUTSIDE the excluded
window. Distractors INSIDE the excluded window (with correct topic).

All docs share the same topic template; only date differs. The excluded
window must contain MULTIPLE distractor docs (i.e. the query's "in 2023"
date is the popular one — gold is the one OUTSIDE).

Output: data/negation_temporal_{docs,queries,gold}.jsonl
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


# query, topic_template, gold_dates (OUTSIDE excluded window),
# distractor_dates (INSIDE excluded window)
CLUSTERS = [
    {
        "query": "What did I do not in 2023?",
        "template": "I attended the company offsite, on {date}.",
        "gold_dates": [(4, 18, 2024)],
        "distractor_dates": [
            (2, 12, 2023),
            (5, 22, 2023),
            (8, 9, 2023),
            (11, 14, 2023),
        ],
    },
    {
        "query": "What expenses do I have outside of the holiday season (November–December)?",
        "template": "I logged an expense for travel and meals, on {date}.",
        "gold_dates": [(3, 14, 2024)],
        "distractor_dates": [
            (11, 22, 2023),
            (12, 11, 2023),
            (11, 8, 2024),
            (12, 21, 2024),
        ],
    },
    {
        "query": "Meetings excluding Q4 2023",
        "template": "I had a leadership-sync meeting with the directors, on {date}.",
        "gold_dates": [(2, 14, 2024)],
        "distractor_dates": [
            (10, 4, 2023),
            (10, 28, 2023),
            (11, 18, 2023),
            (12, 14, 2023),
        ],
    },
    {
        "query": "What workouts did I do not in January 2025?",
        "template": "I completed a strength training workout, on {date}.",
        "gold_dates": [(4, 22, 2025)],
        "distractor_dates": [(1, 6, 2025), (1, 14, 2025), (1, 22, 2025), (1, 30, 2025)],
    },
    {
        "query": "What appointments did I have outside of the summer (June–August 2024)?",
        "template": "I had a routine medical appointment, on {date}.",
        "gold_dates": [(11, 12, 2024)],
        "distractor_dates": [(6, 5, 2024), (7, 11, 2024), (8, 3, 2024), (8, 24, 2024)],
    },
    {
        "query": "What trips did I take excluding 2022?",
        "template": "I took a road trip to a national park, on {date}.",
        "gold_dates": [(5, 18, 2024)],
        "distractor_dates": [(3, 14, 2022), (6, 22, 2022), (9, 8, 2022), (12, 4, 2022)],
    },
    {
        "query": "What classes did I take outside of the spring 2024 semester?",
        "template": "I attended a continuing-education class, on {date}.",
        "gold_dates": [(10, 8, 2024)],
        "distractor_dates": [(2, 4, 2024), (3, 18, 2024), (4, 9, 2024), (5, 1, 2024)],
    },
    {
        "query": "What grocery runs did I do not in March 2025?",
        "template": "I did the weekly grocery run at the co-op, on {date}.",
        "gold_dates": [(5, 18, 2025)],
        "distractor_dates": [(3, 4, 2025), (3, 11, 2025), (3, 19, 2025), (3, 27, 2025)],
    },
    {
        "query": "What design reviews did I attend excluding the second half of 2023?",
        "template": "I attended a design-review session for the new dashboard, on {date}.",
        "gold_dates": [(2, 22, 2024)],
        "distractor_dates": [
            (7, 14, 2023),
            (9, 8, 2023),
            (10, 22, 2023),
            (12, 4, 2023),
        ],
    },
    {
        "query": "What books did I read not in 2024?",
        "template": "I finished reading a novel from my reading list, on {date}.",
        "gold_dates": [(3, 4, 2025)],
        "distractor_dates": [
            (1, 14, 2024),
            (4, 22, 2024),
            (8, 9, 2024),
            (11, 17, 2024),
        ],
    },
    {
        "query": "What therapy sessions did I have outside of February 2025?",
        "template": "I had a therapy session with my counselor, on {date}.",
        "gold_dates": [(5, 6, 2025)],
        "distractor_dates": [(2, 3, 2025), (2, 10, 2025), (2, 17, 2025), (2, 24, 2025)],
    },
    {
        "query": "What expenses do I have not in Q1 2024?",
        "template": "I logged a business expense for software licensing, on {date}.",
        "gold_dates": [(7, 18, 2024)],
        "distractor_dates": [(1, 11, 2024), (2, 14, 2024), (3, 5, 2024), (3, 28, 2024)],
    },
    {
        "query": "What client meetings did I have excluding the last week of May 2024?",
        "template": "I had a check-in meeting with the Acme client team, on {date}.",
        "gold_dates": [(8, 14, 2024)],
        "distractor_dates": [
            (5, 27, 2024),
            (5, 28, 2024),
            (5, 30, 2024),
            (5, 31, 2024),
        ],
    },
    {
        "query": "What family events did I attend not in December 2024?",
        "template": "I attended a family birthday gathering, on {date}.",
        "gold_dates": [(4, 12, 2025)],
        "distractor_dates": [
            (12, 5, 2024),
            (12, 14, 2024),
            (12, 22, 2024),
            (12, 28, 2024),
        ],
    },
    {
        "query": "What presentations did I give outside of the fall 2024 semester (September–December)?",
        "template": "I gave a technical presentation to the engineering team, on {date}.",
        "gold_dates": [(2, 19, 2025)],
        "distractor_dates": [
            (9, 11, 2024),
            (10, 14, 2024),
            (11, 22, 2024),
            (12, 9, 2024),
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
            gid = f"nt_{i:03d}_g{j}"
            gold_ids.append(gid)
            docs.append(
                {
                    "doc_id": gid,
                    "text": c["template"].format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )
        for j, (m, d, y) in enumerate(c["distractor_dates"]):
            did = f"nt_{i:03d}_d{j}"
            docs.append(
                {
                    "doc_id": did,
                    "text": c["template"].format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )

        qid = f"nt_q_{i:03d}"
        queries.append({"query_id": qid, "text": c["query"], "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": gold_ids})

    rng.shuffle(docs)
    with open(DATA_DIR / "negation_temporal_docs.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with open(DATA_DIR / "negation_temporal_queries.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(DATA_DIR / "negation_temporal_gold.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)
    print(
        f"negation_temporal: {len(docs)} docs, {len(queries)} queries, {len(gold_rows)} gold rows"
    )


if __name__ == "__main__":
    main()
