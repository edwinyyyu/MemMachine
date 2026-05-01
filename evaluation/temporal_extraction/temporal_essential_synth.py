"""Temporal-essential benchmark: ALL docs in each cluster share the
EXACT same person+noun phrase, varying ONLY the date. Only the temporal
channel can disambiguate.

Each cluster has 1 gold and 5 sibling distractors with identical text
modulo date. Query specifies a temporal anchor (quarter/month/season).
The cross-encoder reranker has zero discriminative signal — every doc
matches the query identically on words. Only T can pick gold.

Generates:
- ~150 docs (25 clusters × 6 docs)
- 25 queries with explicit temporal anchors
- 1 gold per query

Output: data/temporal_essential_{docs,queries,gold}.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2025-01-01T00:00:00Z"


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


# Each cluster: (person, noun_phrase, gold_(month, day, year), anchor_text, sibling_dates)
# anchor_text is the temporal hint added to the query
# sibling_dates: 5 (month, day, year) tuples for distractors at OTHER times
CLUSTERS = [
    (
        "Sarah Park",
        "had her dental cleaning",
        (4, 3, 2024),
        "in early April 2024",
        [(1, 15, 2024), (7, 11, 2024), (10, 22, 2024), (3, 5, 2023), (5, 28, 2024)],
    ),
    (
        "Marcus Davis",
        "delivered the quarterly review",
        (12, 10, 2022),
        "in Q4 2022",
        [(3, 15, 2023), (6, 8, 2024), (9, 22, 2023), (2, 4, 2023), (11, 17, 2024)],
    ),
    (
        "Priya Johnson",
        "led the team retrospective",
        (5, 5, 2024),
        "in early May 2024",
        [(2, 12, 2024), (8, 25, 2023), (11, 30, 2024), (1, 9, 2025), (6, 18, 2024)],
    ),
    (
        "Kim Patel",
        "completed the kitchen remodel",
        (3, 14, 2024),
        "in March 2024",
        [(7, 21, 2023), (10, 5, 2024), (4, 28, 2025), (12, 11, 2023), (8, 16, 2024)],
    ),
    (
        "Aiden Park",
        "gave the investor pitch",
        (10, 10, 2024),
        "in October 2024",
        [(2, 17, 2024), (5, 24, 2023), (7, 9, 2024), (12, 30, 2023), (3, 12, 2025)],
    ),
    (
        "Olivia Roberts",
        "signed up for yoga class",
        (1, 6, 2025),
        "in early January 2025",
        [(4, 13, 2024), (9, 20, 2023), (11, 4, 2024), (6, 29, 2024), (2, 15, 2024)],
    ),
    (
        "Henry Ford",
        "had his performance review",
        (9, 9, 2023),
        "in Q3 2023",
        [(3, 16, 2024), (5, 23, 2024), (12, 30, 2022), (10, 9, 2024), (1, 17, 2024)],
    ),
    (
        "Felix Wood",
        "went on the client onsite",
        (5, 11, 2024),
        "in May 2024",
        [(8, 18, 2023), (1, 4, 2024), (11, 25, 2023), (3, 7, 2024), (6, 20, 2024)],
    ),
    (
        "Quinn Reeves",
        "did the lease signing",
        (4, 7, 2024),
        "in early April 2024",
        [(8, 14, 2023), (10, 21, 2024), (2, 28, 2024), (12, 5, 2023), (6, 11, 2024)],
    ),
    (
        "Sara Lee",
        "completed the puppy adoption",
        (5, 12, 2024),
        "in May 2024",
        [(2, 5, 2024), (10, 19, 2023), (8, 26, 2024), (3, 8, 2025), (11, 14, 2024)],
    ),
    (
        "Tom Reed",
        "finished the marathon",
        (4, 15, 2024),
        "in mid-April 2024",
        [(11, 8, 2023), (3, 22, 2024), (7, 29, 2024), (1, 16, 2025), (5, 4, 2025)],
    ),
    (
        "Layla Smith",
        "completed the tax filing",
        (4, 4, 2024),
        "in early April 2024",
        [(11, 11, 2023), (3, 18, 2024), (7, 25, 2024), (10, 2, 2024), (1, 13, 2024)],
    ),
    (
        "Maya Singh",
        "made the code freeze announcement",
        (12, 13, 2024),
        "in December 2024",
        [(8, 6, 2023), (3, 20, 2024), (5, 27, 2024), (10, 30, 2024), (7, 9, 2024)],
    ),
    (
        "Alice Liu",
        "hosted the garage sale",
        (4, 17, 2024),
        "in mid-April 2024",
        [(8, 10, 2024), (1, 24, 2024), (10, 3, 2023), (6, 15, 2024), (11, 28, 2023)],
    ),
    (
        "Marcus Chen",
        "went to the wedding rehearsal",
        (5, 21, 2024),
        "in May 2024",
        [(2, 14, 2024), (8, 28, 2023), (11, 7, 2024), (1, 10, 2025), (6, 30, 2024)],
    ),
    (
        "Mira Khan",
        "ran the data migration cutover",
        (5, 18, 2024),
        "in May 2024",
        [(8, 11, 2023), (1, 25, 2024), (10, 4, 2024), (3, 22, 2025), (12, 8, 2023)],
    ),
    (
        "Diego Lopez",
        "had his annual physical",
        (3, 23, 2024),
        "in late March 2024",
        [(7, 16, 2024), (12, 30, 2023), (5, 9, 2025), (10, 11, 2023), (1, 18, 2024)],
    ),
    (
        "Paul Clark",
        "delivered the all-hands presentation",
        (5, 26, 2024),
        "in late May 2024",
        [(8, 19, 2023), (1, 5, 2024), (11, 12, 2024), (3, 27, 2025), (6, 22, 2023)],
    ),
    (
        "Vera Lin",
        "did the moving day",
        (1, 28, 2024),
        "in late January 2024",
        [(7, 21, 2023), (10, 14, 2024), (4, 7, 2025), (12, 5, 2024), (5, 11, 2024)],
    ),
    (
        "Yuki Tanaka",
        "completed the scuba dive",
        (5, 30, 2024),
        "at the end of May 2024",
        [(2, 23, 2024), (8, 16, 2023), (11, 9, 2024), (3, 12, 2025), (6, 26, 2024)],
    ),
    (
        "Hannah Hall",
        "attended the design summit",
        (6, 15, 2023),
        "in June 2023",
        [(2, 8, 2024), (9, 22, 2023), (12, 30, 2022), (10, 5, 2024), (5, 18, 2024)],
    ),
    (
        "Eric Hall",
        "gave the keynote speech",
        (2, 28, 2023),
        "in February 2023",
        [(8, 14, 2024), (11, 21, 2023), (5, 7, 2024), (3, 30, 2024), (1, 16, 2025)],
    ),
    (
        "Ethan Thomas",
        "completed onboarding",
        (3, 3, 2023),
        "in early March 2023",
        [(7, 16, 2024), (10, 24, 2023), (12, 11, 2024), (6, 29, 2023), (2, 19, 2025)],
    ),
    (
        "Carla Smith",
        "received the employee of the month award",
        (11, 9, 2023),
        "in Q4 2023",
        [(5, 4, 2024), (3, 22, 2025), (8, 18, 2023), (1, 25, 2024), (7, 30, 2024)],
    ),
    (
        "Kim Chen",
        "attended the company offsite",
        (1, 28, 2024),
        "in late January 2024",
        [(7, 11, 2023), (10, 4, 2024), (4, 21, 2025), (12, 18, 2023), (5, 8, 2024)],
    ),
]


def main() -> None:
    rng = random.Random(20260430)
    docs = []
    queries = []
    gold_rows = []

    for i, (person, noun, gold_date, anchor, sib_dates) in enumerate(CLUSTERS):
        # gold doc
        gm, gd, gy = gold_date
        gold_id = f"te_{i:03d}_g"
        gold_text = f"{person} {noun} on {fmt_date(gm, gd, gy)}."
        docs.append({"doc_id": gold_id, "text": gold_text, "ref_time": iso(gm, gd, gy)})

        # 5 sibling distractors with identical text, different dates
        for j, (sm, sd, sy) in enumerate(sib_dates):
            sib_id = f"te_{i:03d}_s{j}"
            sib_text = f"{person} {noun} on {fmt_date(sm, sd, sy)}."
            docs.append(
                {"doc_id": sib_id, "text": sib_text, "ref_time": iso(sm, sd, sy)}
            )

        # query: "When did {person} {noun} {anchor}?"
        # transform "had her" → "have her" etc. for grammar
        q_verb = noun
        if noun.startswith("had "):
            q_verb = "have " + noun[4:]
        elif noun.startswith("did "):
            q_verb = "do " + noun[4:]
        elif noun.startswith("led "):
            q_verb = "lead " + noun[4:]
        elif noun.startswith("delivered "):
            q_verb = "deliver " + noun[len("delivered ") :]
        elif noun.startswith("completed "):
            q_verb = "complete " + noun[len("completed ") :]
        elif noun.startswith("gave "):
            q_verb = "give " + noun[5:]
        elif noun.startswith("signed "):
            q_verb = "sign " + noun[len("signed ") :]
        elif noun.startswith("went "):
            q_verb = "go " + noun[5:]
        elif noun.startswith("finished "):
            q_verb = "finish " + noun[len("finished ") :]
        elif noun.startswith("made "):
            q_verb = "make " + noun[5:]
        elif noun.startswith("hosted "):
            q_verb = "host " + noun[len("hosted ") :]
        elif noun.startswith("ran "):
            q_verb = "run " + noun[4:]
        elif noun.startswith("attended "):
            q_verb = "attend " + noun[len("attended ") :]
        elif noun.startswith("received "):
            q_verb = "receive " + noun[len("received ") :]
        qid = f"te_q_{i:03d}"
        q_text = f"When did {person} {q_verb} {anchor}?"
        queries.append({"query_id": qid, "text": q_text, "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": [gold_id]})

    rng.shuffle(docs)
    docs_path = DATA_DIR / "temporal_essential_docs.jsonl"
    queries_path = DATA_DIR / "temporal_essential_queries.jsonl"
    gold_path = DATA_DIR / "temporal_essential_gold.jsonl"
    with open(docs_path, "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    with open(queries_path, "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(gold_path, "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)

    print(f"Wrote {len(docs)} docs, {len(queries)} queries, {len(gold_rows)} gold rows")
    print("  6 docs per cluster (1 gold + 5 sibling distractors)")
    print("  All docs in cluster have IDENTICAL person+noun, only date differs")
    print("  Reranker has zero discriminative signal; only T can pick gold")
    print(f"\nSample query: {queries[0]['text']}")
    print(
        f"Sample gold:  {[d for d in docs if d['doc_id'] == gold_rows[0]['relevant_doc_ids'][0]][0]['text']}"
    )
    print(
        f"Sample sibling: {[d for d in docs if d['doc_id'].startswith(gold_rows[0]['relevant_doc_ids'][0][:7]) and d['doc_id'].endswith('s0')][0]['text']}"
    )


if __name__ == "__main__":
    main()
