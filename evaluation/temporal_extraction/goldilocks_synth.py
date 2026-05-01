"""Goldilocks benchmark: queries where middle w_T (~0.2) beats BOTH extremes.

Each query has:
  - GOLD: matches BOTH a fuzzy temporal anchor AND a specific topic
  - S-LOUD distractor: matches topic perfectly, wrong date (rerank picks this at w_T=0)
  - T-LOUD distractor: matches date perfectly, wrong topic (T pushes this at w_T=0.4+)
  - 2 noise docs

If gate (rerank_only vs fuse_T_R(0.4)) is forced to pick:
  - rerank_only → S-loud wins → gold missed
  - fuse_T_R(w=0.4) → T-loud wins → gold missed
The optimum requires w_T~0.15-0.25 to balance both signals.

Generates: data/goldilocks_{docs,queries,gold}.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2025-01-15T00:00:00Z"


def fmt(month: int, day: int, year: int) -> str:
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


# Each cluster: query has fuzzy temporal anchor + specific topic.
# Format: (query_template, gold_topic_phrase, gold_date, s_loud_date_offset, t_loud_topic, noise_topics)
# We construct 5 docs per cluster: gold, S-loud, T-loud, noise1, noise2
SPECS = [
    # (query, gold_full_text, s_loud_text, t_loud_text, noise1, noise2, gold_iso)
    (
        "What did I order from the thai place last summer?",
        (
            "Got pad see ew from Bangkok Garden on July 14, 2024.",
            iso(7, 14, 2024),
        ),  # GOLD: thai (Bangkok Garden) + summer
        (
            "Got pad see ew from Bangkok Garden on January 22, 2024.",
            iso(1, 22, 2024),
        ),  # S-LOUD: same restaurant, wrong season
        (
            "Booked the dentist appointment downtown on July 12, 2024.",
            iso(7, 12, 2024),
        ),  # T-LOUD: same week, wrong topic
        ("Picked up groceries from the corner shop on May 3, 2023.", iso(5, 3, 2023)),
        ("Watched the new documentary on March 11, 2024.", iso(3, 11, 2024)),
    ),
    (
        "What was Jamie's feedback on the design proposal a few weeks ago?",
        (
            "Jamie thought the design proposal needed clearer hierarchy on December 28, 2024.",
            iso(12, 28, 2024),
        ),  # GOLD
        (
            "Jamie thought the design proposal needed clearer hierarchy on March 5, 2023.",
            iso(3, 5, 2023),
        ),  # S-LOUD: same feedback, old
        (
            "Submitted the quarterly budget review on December 30, 2024.",
            iso(12, 30, 2024),
        ),  # T-LOUD: recent, wrong topic
        ("Reviewed the marketing slides with Pat on May 17, 2024.", iso(5, 17, 2024)),
        ("Caught up on email backlog on November 4, 2024.", iso(11, 4, 2024)),
    ),
    (
        "What issue did we hit during the migration last quarter?",
        (
            "Hit a database deadlock during the migration on November 15, 2024.",
            iso(11, 15, 2024),
        ),  # GOLD: Q4 + migration deadlock
        (
            "Hit a database deadlock during the migration on April 8, 2023.",
            iso(4, 8, 2023),
        ),  # S-LOUD: same issue, wrong quarter
        (
            "Wrapped up the customer onboarding on November 18, 2024.",
            iso(11, 18, 2024),
        ),  # T-LOUD: same week, wrong topic
        ("Reviewed Q1 hiring plan with HR on February 6, 2024.", iso(2, 6, 2024)),
        ("Patched the staging server config on August 22, 2023.", iso(8, 22, 2023)),
    ),
    (
        "When was my last appointment with the cardiologist?",
        (
            "Annual cardiologist checkup on November 20, 2024.",
            iso(11, 20, 2024),
        ),  # GOLD: most recent + cardiologist
        (
            "Annual cardiologist checkup on March 4, 2022.",
            iso(3, 4, 2022),
        ),  # S-LOUD: same activity, way older
        (
            "Met with the new dermatologist on December 12, 2024.",
            iso(12, 12, 2024),
        ),  # T-LOUD: recent, wrong specialty
        ("Picked up prescription refills on June 9, 2024.", iso(6, 9, 2024)),
        ("Booked the dentist for next month on October 14, 2023.", iso(10, 14, 2023)),
    ),
    (
        "What did Priya pitch at the offsite a couple months ago?",
        (
            "Priya pitched the new analytics dashboard at the offsite on November 8, 2024.",
            iso(11, 8, 2024),
        ),  # GOLD
        (
            "Priya pitched the new analytics dashboard at the offsite on May 17, 2023.",
            iso(5, 17, 2023),
        ),  # S-LOUD: same pitch, old
        (
            "Closed the Q4 hiring plan with Carla on November 11, 2024.",
            iso(11, 11, 2024),
        ),  # T-LOUD: same week, wrong topic
        ("Submitted travel reimbursement on April 22, 2024.", iso(4, 22, 2024)),
        ("Reviewed retention metrics on July 30, 2024.", iso(7, 30, 2024)),
    ),
    (
        "What did Marcus mention about the budget around late spring?",
        (
            "Marcus said the budget would need a 10% cut on May 28, 2024.",
            iso(5, 28, 2024),
        ),  # GOLD: late spring + budget
        (
            "Marcus said the budget would need a 10% cut on November 17, 2023.",
            iso(11, 17, 2023),
        ),  # S-LOUD
        (
            "Discussed the new vendor contract on May 30, 2024.",
            iso(5, 30, 2024),
        ),  # T-LOUD
        ("Wrapped up the team retro on October 9, 2024.", iso(10, 9, 2024)),
        ("Updated the client presentation on January 22, 2024.", iso(1, 22, 2024)),
    ),
    (
        "What did Sarah recommend at the bookstore back in early 2024?",
        (
            "Sarah recommended the new Murakami novel at the bookstore on February 18, 2024.",
            iso(2, 18, 2024),
        ),  # GOLD
        (
            "Sarah recommended the new Murakami novel at the bookstore on October 9, 2022.",
            iso(10, 9, 2022),
        ),  # S-LOUD
        (
            "Picked up coffee beans from the roastery on February 22, 2024.",
            iso(2, 22, 2024),
        ),  # T-LOUD
        ("Met Alex at the gallery opening on August 14, 2024.", iso(8, 14, 2024)),
        ("Submitted the conference abstract on July 5, 2023.", iso(7, 5, 2023)),
    ),
    (
        "What feature did the team ship around mid-October?",
        (
            "Team shipped the dark-mode toggle on October 16, 2024.",
            iso(10, 16, 2024),
        ),  # GOLD
        (
            "Team shipped the dark-mode toggle on June 4, 2023.",
            iso(6, 4, 2023),
        ),  # S-LOUD
        ("Closed the security audit on October 14, 2024.", iso(10, 14, 2024)),  # T-LOUD
        ("Sent quarterly investor update on April 27, 2024.", iso(4, 27, 2024)),
        (
            "Met with the legal team about contracts on December 1, 2024.",
            iso(12, 1, 2024),
        ),
    ),
    (
        "What did Olivia say about the redesign last fall?",
        (
            "Olivia said the redesign needs better mobile responsiveness on October 22, 2024.",
            iso(10, 22, 2024),
        ),  # GOLD
        (
            "Olivia said the redesign needs better mobile responsiveness on March 11, 2023.",
            iso(3, 11, 2023),
        ),  # S-LOUD
        (
            "Reviewed the customer support tickets on October 25, 2024.",
            iso(10, 25, 2024),
        ),  # T-LOUD
        (
            "Booked the venue for the holiday party on August 14, 2024.",
            iso(8, 14, 2024),
        ),
        ("Updated the engineering wiki on January 9, 2024.", iso(1, 9, 2024)),
    ),
    (
        "When did Tom borrow the camping gear last summer?",
        ("Tom borrowed the camping gear on July 21, 2024.", iso(7, 21, 2024)),  # GOLD
        (
            "Tom borrowed the camping gear on December 14, 2023.",
            iso(12, 14, 2023),
        ),  # S-LOUD
        ("Returned the library books on July 24, 2024.", iso(7, 24, 2024)),  # T-LOUD
        ("Picked up the dry cleaning on May 30, 2024.", iso(5, 30, 2024)),
        ("Filed the warranty claim on October 17, 2023.", iso(10, 17, 2023)),
    ),
    (
        "Around when did we discuss the pricing model?",
        (
            "Discussed the pricing model with the leadership team on September 5, 2024.",
            iso(9, 5, 2024),
        ),  # GOLD
        (
            "Discussed the pricing model with the leadership team on February 19, 2023.",
            iso(2, 19, 2023),
        ),  # S-LOUD
        (
            "Wrapped up the partner agreement on September 8, 2024.",
            iso(9, 8, 2024),
        ),  # T-LOUD
        ("Reviewed the campaign metrics on December 22, 2023.", iso(12, 22, 2023)),
        ("Sent the renewal proposal on April 11, 2024.", iso(4, 11, 2024)),
    ),
    (
        "What did the team retro a few months back surface as the top issue?",
        (
            "Team retro flagged unclear PR review process as the top issue on October 3, 2024.",
            iso(10, 3, 2024),
        ),  # GOLD
        (
            "Team retro flagged unclear PR review process as the top issue on January 22, 2023.",
            iso(1, 22, 2023),
        ),  # S-LOUD
        (
            "Reviewed the security incident postmortem on October 7, 2024.",
            iso(10, 7, 2024),
        ),  # T-LOUD
        ("Onboarded the new contractor on May 13, 2024.", iso(5, 13, 2024)),
        ("Renewed the office lease on December 30, 2023.", iso(12, 30, 2023)),
    ),
    (
        "What did Pat suggest about the launch around late summer?",
        (
            "Pat suggested delaying the launch by two weeks on August 24, 2024.",
            iso(8, 24, 2024),
        ),  # GOLD
        (
            "Pat suggested delaying the launch by two weeks on March 16, 2023.",
            iso(3, 16, 2023),
        ),  # S-LOUD
        (
            "Approved the marketing campaign budget on August 27, 2024.",
            iso(8, 27, 2024),
        ),  # T-LOUD
        (
            "Met with the recruiter for senior candidates on December 4, 2024.",
            iso(12, 4, 2024),
        ),
        ("Wrapped up the contract negotiations on April 9, 2024.", iso(4, 9, 2024)),
    ),
    (
        "What was the bug we fixed during the holiday week?",
        (
            "Fixed the timezone parsing bug on December 27, 2024.",
            iso(12, 27, 2024),
        ),  # GOLD
        (
            "Fixed the timezone parsing bug on June 14, 2023.",
            iso(6, 14, 2023),
        ),  # S-LOUD
        (
            "Wrapped up the year-end performance reviews on December 30, 2024.",
            iso(12, 30, 2024),
        ),  # T-LOUD
        ("Reviewed the Q3 hiring plan on September 18, 2024.", iso(9, 18, 2024)),
        ("Renewed the cloud provider contract on May 7, 2024.", iso(5, 7, 2024)),
    ),
    (
        "When did Henry finish the manuscript draft last winter?",
        (
            "Henry finished the manuscript draft on January 28, 2024.",
            iso(1, 28, 2024),
        ),  # GOLD: winter early-2024
        (
            "Henry finished the manuscript draft on August 9, 2023.",
            iso(8, 9, 2023),
        ),  # S-LOUD
        (
            "Booked the conference travel on January 31, 2024.",
            iso(1, 31, 2024),
        ),  # T-LOUD
        ("Picked up the bicycle from the shop on July 12, 2024.", iso(7, 12, 2024)),
        (
            "Sent feedback on the marketing brief on October 22, 2023.",
            iso(10, 22, 2023),
        ),
    ),
]


def main() -> None:
    rng = random.Random(20260430)
    docs = []
    queries = []
    gold_rows = []

    for i, spec in enumerate(SPECS):
        (
            q_text,
            (gold_text, gold_iso),
            (sl_text, sl_iso),
            (tl_text, tl_iso),
            (n1_text, n1_iso),
            (n2_text, n2_iso),
        ) = spec
        gold_id = f"gl_{i:03d}_g"
        sl_id = f"gl_{i:03d}_sl"
        tl_id = f"gl_{i:03d}_tl"
        n1_id = f"gl_{i:03d}_n1"
        n2_id = f"gl_{i:03d}_n2"

        for did, text, iso_t in [
            (gold_id, gold_text, gold_iso),
            (sl_id, sl_text, sl_iso),
            (tl_id, tl_text, tl_iso),
            (n1_id, n1_text, n1_iso),
            (n2_id, n2_text, n2_iso),
        ]:
            docs.append({"doc_id": did, "text": text, "ref_time": iso_t})

        qid = f"gl_q_{i:03d}"
        queries.append({"query_id": qid, "text": q_text, "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": [gold_id]})

    rng.shuffle(docs)
    docs_path = DATA_DIR / "goldilocks_docs.jsonl"
    queries_path = DATA_DIR / "goldilocks_queries.jsonl"
    gold_path = DATA_DIR / "goldilocks_gold.jsonl"
    with open(docs_path, "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    with open(queries_path, "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(gold_path, "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)

    print(f"Wrote {len(docs)} docs, {len(queries)} queries, {len(gold_rows)} gold rows")
    print("Each cluster: gold + S-loud + T-loud + 2 noise = 5 docs")
    print("  S-loud distractor: same topic as gold, wrong date")
    print("  T-loud distractor: same week as gold, wrong topic")
    print(f"\nSample query: {queries[0]['text']}")


if __name__ == "__main__":
    main()
