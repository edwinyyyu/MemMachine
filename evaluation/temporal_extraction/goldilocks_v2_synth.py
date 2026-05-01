"""Goldilocks v2: doc text has NO explicit dates. Only ref_time metadata
encodes when. Cross-encoder reranker can't use temporal word-match;
disambiguation requires T's interval/lattice scoring.

For each cluster:
  - GOLD: matches topic + the query's fuzzy temporal anchor (via ref_time)
  - S-LOUD: identical doc text as gold, ref_time wrong (different season/year)
  - T-LOUD: different topic, ref_time within gold's anchor range
  - 2 noise: different topic, different time

Without T, gold and S-loud are byte-identical text → cross-encoder ties them
at top (50/50 R@1). T must contribute enough weight to break the tie via
ref_time, but not so much that T-loud overtakes (since T-loud is also in
the right time window).

Optimum should sit at moderate w_T. Tests if gate (0 vs 0.4) misses this.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2025-01-15T00:00:00Z"


def iso(month: int, day: int, year: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}T12:00:00Z"


# (query, gold_doc_text, gold_(m,d,y), s_loud_(m,d,y), t_loud_text, t_loud_(m,d,y),
#  noise1_text, noise1_(m,d,y), noise2_text, noise2_(m,d,y))
SPECS = [
    (
        "What did I order from the thai place last summer?",
        "Got pad see ew from Bangkok Garden.",
        (7, 14, 2024),  # gold = summer 2024
        (1, 22, 2024),  # S-loud = winter 2024 (same text)
        "Booked the dentist appointment downtown.",
        (7, 12, 2024),  # T-loud = same week, wrong topic
        "Picked up groceries from the corner shop.",
        (5, 3, 2023),
        "Watched the new documentary.",
        (3, 11, 2024),
    ),
    (
        "What was Jamie's feedback on the design proposal a few weeks ago?",
        "Jamie thought the design proposal needed clearer hierarchy.",
        (12, 28, 2024),  # gold = recent (a few weeks before ref 2025-01-15)
        (3, 5, 2023),  # S-loud = old
        "Submitted the quarterly budget review.",
        (12, 30, 2024),
        "Reviewed the marketing slides with Pat.",
        (5, 17, 2024),
        "Caught up on email backlog.",
        (11, 4, 2024),
    ),
    (
        "What issue did we hit during the migration last quarter?",
        "Hit a database deadlock during the migration.",
        (11, 15, 2024),  # gold = Q4 2024 (last quarter from ref 2025-01-15)
        (4, 8, 2023),
        "Wrapped up the customer onboarding.",
        (11, 18, 2024),
        "Reviewed Q1 hiring plan with HR.",
        (2, 6, 2024),
        "Patched the staging server config.",
        (8, 22, 2023),
    ),
    (
        "When was my last appointment with the cardiologist?",
        "Annual cardiologist checkup.",
        (11, 20, 2024),  # gold = most recent
        (3, 4, 2022),  # S-loud = way older
        "Met with the new dermatologist.",
        (12, 12, 2024),
        "Picked up prescription refills.",
        (6, 9, 2024),
        "Booked the dentist for next month.",
        (10, 14, 2023),
    ),
    (
        "What did Priya pitch at the offsite a couple months ago?",
        "Priya pitched the new analytics dashboard at the offsite.",
        (11, 8, 2024),
        (5, 17, 2023),
        "Closed the Q4 hiring plan with Carla.",
        (11, 11, 2024),
        "Submitted travel reimbursement.",
        (4, 22, 2024),
        "Reviewed retention metrics.",
        (7, 30, 2024),
    ),
    (
        "What did Marcus mention about the budget around late spring?",
        "Marcus said the budget would need a 10% cut.",
        (5, 28, 2024),
        (11, 17, 2023),
        "Discussed the new vendor contract.",
        (5, 30, 2024),
        "Wrapped up the team retro.",
        (10, 9, 2024),
        "Updated the client presentation.",
        (1, 22, 2024),
    ),
    (
        "What did Sarah recommend at the bookstore back in early 2024?",
        "Sarah recommended the new Murakami novel at the bookstore.",
        (2, 18, 2024),
        (10, 9, 2022),
        "Picked up coffee beans from the roastery.",
        (2, 22, 2024),
        "Met Alex at the gallery opening.",
        (8, 14, 2024),
        "Submitted the conference abstract.",
        (7, 5, 2023),
    ),
    (
        "What feature did the team ship around mid-October?",
        "Team shipped the dark-mode toggle.",
        (10, 16, 2024),
        (6, 4, 2023),
        "Closed the security audit.",
        (10, 14, 2024),
        "Sent quarterly investor update.",
        (4, 27, 2024),
        "Met with the legal team about contracts.",
        (12, 1, 2024),
    ),
    (
        "What did Olivia say about the redesign last fall?",
        "Olivia said the redesign needs better mobile responsiveness.",
        (10, 22, 2024),
        (3, 11, 2023),
        "Reviewed the customer support tickets.",
        (10, 25, 2024),
        "Booked the venue for the holiday party.",
        (8, 14, 2024),
        "Updated the engineering wiki.",
        (1, 9, 2024),
    ),
    (
        "When did Tom borrow the camping gear last summer?",
        "Tom borrowed the camping gear.",
        (7, 21, 2024),
        (12, 14, 2023),
        "Returned the library books.",
        (7, 24, 2024),
        "Picked up the dry cleaning.",
        (5, 30, 2024),
        "Filed the warranty claim.",
        (10, 17, 2023),
    ),
    (
        "Around when did we discuss the pricing model?",
        "Discussed the pricing model with the leadership team.",
        (9, 5, 2024),
        (2, 19, 2023),
        "Wrapped up the partner agreement.",
        (9, 8, 2024),
        "Reviewed the campaign metrics.",
        (12, 22, 2023),
        "Sent the renewal proposal.",
        (4, 11, 2024),
    ),
    (
        "What did the team retro a few months back surface as the top issue?",
        "Team retro flagged unclear PR review process as the top issue.",
        (10, 3, 2024),
        (1, 22, 2023),
        "Reviewed the security incident postmortem.",
        (10, 7, 2024),
        "Onboarded the new contractor.",
        (5, 13, 2024),
        "Renewed the office lease.",
        (12, 30, 2023),
    ),
    (
        "What did Pat suggest about the launch around late summer?",
        "Pat suggested delaying the launch by two weeks.",
        (8, 24, 2024),
        (3, 16, 2023),
        "Approved the marketing campaign budget.",
        (8, 27, 2024),
        "Met with the recruiter for senior candidates.",
        (12, 4, 2024),
        "Wrapped up the contract negotiations.",
        (4, 9, 2024),
    ),
    (
        "What was the bug we fixed during the holiday week?",
        "Fixed the timezone parsing bug.",
        (12, 27, 2024),
        (6, 14, 2023),
        "Wrapped up the year-end performance reviews.",
        (12, 30, 2024),
        "Reviewed the Q3 hiring plan.",
        (9, 18, 2024),
        "Renewed the cloud provider contract.",
        (5, 7, 2024),
    ),
    (
        "When did Henry finish the manuscript draft last winter?",
        "Henry finished the manuscript draft.",
        (1, 28, 2024),
        (8, 9, 2023),
        "Booked the conference travel.",
        (1, 31, 2024),
        "Picked up the bicycle from the shop.",
        (7, 12, 2024),
        "Sent feedback on the marketing brief.",
        (10, 22, 2023),
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
            gold_text,
            gold_dt,
            sl_dt,
            tl_text,
            tl_dt,
            n1_text,
            n1_dt,
            n2_text,
            n2_dt,
        ) = spec
        gold_id = f"gl2_{i:03d}_g"
        sl_id = f"gl2_{i:03d}_sl"
        tl_id = f"gl2_{i:03d}_tl"
        n1_id = f"gl2_{i:03d}_n1"
        n2_id = f"gl2_{i:03d}_n2"

        for did, text, dt in [
            (gold_id, gold_text, gold_dt),
            (sl_id, gold_text, sl_dt),  # IDENTICAL text to gold, different ref_time
            (tl_id, tl_text, tl_dt),
            (n1_id, n1_text, n1_dt),
            (n2_id, n2_text, n2_dt),
        ]:
            docs.append({"doc_id": did, "text": text, "ref_time": iso(*dt)})

        qid = f"gl2_q_{i:03d}"
        queries.append({"query_id": qid, "text": q_text, "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": [gold_id]})

    rng.shuffle(docs)
    docs_path = DATA_DIR / "goldilocks_v2_docs.jsonl"
    queries_path = DATA_DIR / "goldilocks_v2_queries.jsonl"
    gold_path = DATA_DIR / "goldilocks_v2_gold.jsonl"
    with open(docs_path, "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    with open(queries_path, "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(gold_path, "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)

    print(f"Wrote {len(docs)} docs, {len(queries)} queries, {len(gold_rows)} gold rows")
    print("Each cluster: gold + S-loud (IDENTICAL text) + T-loud + 2 noise = 5 docs")
    print("  Doc text contains NO dates; only ref_time encodes when.")
    print("  Without T: gold and S-loud tie on rerank → R@1 = 0.5 floor")
    print("  With T moderate: gold wins via ref_time alignment with query anchor")
    print("  With T strong: T-loud (right time, wrong topic) may overtake gold")
    print(f"\nSample query: {queries[0]['text']}")
    print(f"Sample gold doc: {gold_text}")


if __name__ == "__main__":
    main()
