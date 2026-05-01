"""Latest-recent benchmark: queries asking for the MOST RECENT instance.

Each cluster: 5 docs of the SAME topic at different dates spread over months.
Gold = the most recent (closest to query's ref_time).
Distractors = the 4 older instances.

Queries do NOT contain explicit dates (would trivialize the test).

Output: data/latest_recent_{docs,queries,gold}.jsonl
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


# (topic_label, doc_template, query, dates_oldest_to_newest_5)
# query asks for "latest"/"last"/"most recent" — no explicit dates
CLUSTERS = [
    (
        "status_update_alpha",
        "Project Alpha status update: progress is on track and milestones updated, on {date}.",
        "What's my latest project Alpha status update?",
        [(1, 5, 2025), (2, 12, 2025), (3, 18, 2025), (4, 22, 2025), (5, 28, 2025)],
    ),
    (
        "dr_patel",
        "Appointment with Dr. Patel for the routine checkup, on {date}.",
        "When was my last appointment with Dr. Patel?",
        [(7, 10, 2024), (10, 4, 2024), (12, 12, 2024), (3, 6, 2025), (5, 19, 2025)],
    ),
    (
        "design_feedback",
        "Feedback on the new dashboard design from the review committee, on {date}.",
        "What's the most recent feedback on the dashboard design?",
        [(11, 14, 2024), (1, 9, 2025), (2, 27, 2025), (4, 11, 2025), (5, 30, 2025)],
    ),
    (
        "car_service",
        "Brought the car in for service at the dealership, on {date}.",
        "When was my last car service?",
        [(8, 21, 2023), (2, 14, 2024), (7, 9, 2024), (12, 3, 2024), (4, 25, 2025)],
    ),
    (
        "performance_review",
        "Performance review with my manager covering the recent cycle, on {date}.",
        "What's the most recent performance review I had?",
        [(6, 4, 2023), (12, 18, 2023), (6, 12, 2024), (12, 9, 2024), (5, 22, 2025)],
    ),
    (
        "contract_renewal",
        "Renewed the office lease contract with the landlord, on {date}.",
        "When did I last renew the office lease?",
        [(3, 1, 2021), (3, 1, 2022), (3, 1, 2023), (3, 1, 2024), (3, 1, 2025)],
    ),
    (
        "therapy_session",
        "Therapy session with my counselor focused on weekly check-in, on {date}.",
        "When was my latest therapy session?",
        [(2, 3, 2025), (3, 10, 2025), (4, 7, 2025), (5, 5, 2025), (6, 2, 2025)],
    ),
    (
        "dentist_visit",
        "Visited the dentist for a routine cleaning and exam, on {date}.",
        "When was my last dentist visit?",
        [(9, 12, 2023), (3, 8, 2024), (8, 26, 2024), (2, 17, 2025), (5, 14, 2025)],
    ),
    (
        "haircut",
        "Got a haircut at the barber on Main Street, on {date}.",
        "When did I last get a haircut?",
        [(11, 22, 2024), (1, 15, 2025), (2, 26, 2025), (4, 9, 2025), (5, 27, 2025)],
    ),
    (
        "groceries_run",
        "Did the weekly grocery run at the co-op, on {date}.",
        "When was my most recent grocery run?",
        [(4, 30, 2025), (5, 7, 2025), (5, 14, 2025), (5, 21, 2025), (6, 4, 2025)],
    ),
    (
        "client_checkin",
        "Quarterly check-in with the Acme client account team, on {date}.",
        "What's my latest Acme client check-in?",
        [(6, 8, 2024), (9, 11, 2024), (12, 4, 2024), (3, 13, 2025), (5, 31, 2025)],
    ),
    (
        "blood_test",
        "Got blood work done at the lab as part of routine monitoring, on {date}.",
        "When was my most recent blood test?",
        [(5, 16, 2023), (11, 22, 2023), (5, 9, 2024), (11, 14, 2024), (5, 6, 2025)],
    ),
    (
        "pr_review",
        "Reviewed a pull request from the platform team, on {date}.",
        "When was the last pull request I reviewed from the platform team?",
        [(4, 4, 2025), (4, 18, 2025), (5, 2, 2025), (5, 16, 2025), (6, 6, 2025)],
    ),
    (
        "portfolio_rebalance",
        "Rebalanced the retirement portfolio to the target allocation, on {date}.",
        "When did I most recently rebalance my portfolio?",
        [(7, 1, 2023), (1, 1, 2024), (7, 1, 2024), (1, 1, 2025), (4, 1, 2025)],
    ),
    (
        "yoga_class",
        "Attended yoga class at the studio downtown, on {date}.",
        "When was my latest yoga class?",
        [(3, 25, 2025), (4, 8, 2025), (4, 22, 2025), (5, 6, 2025), (5, 27, 2025)],
    ),
]


def main() -> None:
    rng = random.Random(20260429)
    docs = []
    queries = []
    gold_rows = []

    for i, (topic, template, query, dates) in enumerate(CLUSTERS):
        # Last date is the gold (most recent)
        gold_idx = len(dates) - 1
        gold_id = f"lr_{i:03d}_g"
        for j, (m, d, y) in enumerate(dates):
            doc_id = gold_id if j == gold_idx else f"lr_{i:03d}_d{j}"
            text = template.format(date=fmt_date(m, d, y))
            docs.append({"doc_id": doc_id, "text": text, "ref_time": iso(m, d, y)})

        qid = f"lr_q_{i:03d}"
        queries.append({"query_id": qid, "text": query, "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": [gold_id]})

    rng.shuffle(docs)
    with open(DATA_DIR / "latest_recent_docs.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with open(DATA_DIR / "latest_recent_queries.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(DATA_DIR / "latest_recent_gold.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)
    print(
        f"latest_recent: {len(docs)} docs, {len(queries)} queries, {len(gold_rows)} gold rows"
    )


if __name__ == "__main__":
    main()
