"""Harder dense-cluster benchmark.

Breaks saturation by adding 3-5 SAME-NOUN sibling distractors per gold doc:
each gold has identical key noun phrase but differs in person/date/detail.
Forces T (temporal/entity) channels to disambiguate; reranker alone fails.

Generates:
- ~150 docs across April-May 2024 (gold + siblings)
- ~30 queries asking for specific PERSON+EVENT+DATE combinations
- 1 gold doc per query

Output: data/hard_dense_cluster_{docs,queries,gold}.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2024-06-01T00:00:00Z"

# Gold + sibling specs: each tuple is the GOLD; siblings include SAME PERSON
# at different dates (forces T to disambiguate by date, not name) AND
# different persons sharing the activity (forces R to use names).
# Query MUST include a temporal anchor so T can score docs.
#
# Format: (person, key_noun, gold_month, gold_day, [(sib_person, sib_month, sib_day) for siblings],
#          query_template_with_temporal_anchor)
SPECS = [
    # The trick: gold person ALSO appears in a sibling at a different time.
    # Reranker can't disambiguate by name; date is essential.
    (
        "Sarah Park",
        "dental cleaning",
        4,
        3,
        [("Sarah Park", 5, 11), ("Mira Khan", 4, 19), ("Olivia Choi", 4, 27)],
        "When did Sarah Park have her dental cleaning in early April?",
    ),
    (
        "Marcus Davis",
        "annual checkup",
        4,
        8,
        [("Marcus Davis", 5, 16), ("Henry Ford", 4, 22), ("Layla Smith", 4, 30)],
        "When did Marcus Davis go in for the annual checkup back in April?",
    ),
    (
        "Priya Johnson",
        "team retrospective",
        4,
        5,
        [("Priya Johnson", 5, 12), ("Felix Wood", 4, 18), ("Maya Singh", 4, 25)],
        "When did Priya Johnson lead the early-April team retrospective?",
    ),
    (
        "Kim Patel",
        "kitchen remodel update",
        4,
        14,
        [("Kim Patel", 5, 7), ("Alice Liu", 4, 21), ("Quinn Reeves", 4, 28)],
        "When did Kim Patel post the kitchen remodel update in mid-April?",
    ),
    (
        "Aiden Park",
        "investor pitch",
        4,
        10,
        [("Aiden Park", 5, 17), ("Sara Lee", 4, 24), ("Marcus Chen", 4, 30)],
        "When did Aiden Park give the investor pitch in April?",
    ),
    (
        "Olivia Roberts",
        "yoga class signup",
        4,
        6,
        [("Olivia Roberts", 5, 13), ("Tom Reed", 4, 20), ("Sarah Park", 4, 29)],
        "When did Olivia Roberts sign up for yoga class in April?",
    ),
    (
        "Henry Ford",
        "performance review",
        5,
        9,
        [("Henry Ford", 4, 16), ("Layla Smith", 5, 23), ("Diego Lopez", 5, 30)],
        "When did Henry Ford have the May performance review?",
    ),
    (
        "Felix Wood",
        "client onsite visit",
        5,
        11,
        [("Felix Wood", 4, 18), ("Maya Singh", 5, 4), ("Paul Clark", 5, 25)],
        "When did Felix Wood go on the May client onsite visit?",
    ),
    (
        "Quinn Reeves",
        "lease signing",
        4,
        7,
        [("Quinn Reeves", 5, 14), ("Alice Liu", 4, 21), ("Vera Lin", 4, 28)],
        "When did Quinn Reeves do the lease signing in early-April?",
    ),
    (
        "Sara Lee",
        "puppy adoption",
        5,
        12,
        [("Sara Lee", 4, 5), ("Aiden Park", 5, 19), ("Marcus Chen", 5, 26)],
        "When did Sara Lee complete the puppy adoption in May?",
    ),
    (
        "Tom Reed",
        "marathon finish",
        4,
        15,
        [("Tom Reed", 5, 8), ("Olivia Roberts", 4, 22), ("Mira Khan", 4, 29)],
        "When did Tom Reed finish the marathon in April?",
    ),
    (
        "Layla Smith",
        "tax filing",
        4,
        4,
        [("Layla Smith", 5, 11), ("Henry Ford", 4, 18), ("Diego Lopez", 4, 25)],
        "When did Layla Smith complete the tax filing in early April?",
    ),
    (
        "Maya Singh",
        "code freeze announcement",
        5,
        13,
        [("Maya Singh", 4, 6), ("Felix Wood", 5, 20), ("Paul Clark", 5, 27)],
        "When did Maya Singh make the May code freeze announcement?",
    ),
    (
        "Alice Liu",
        "garage sale",
        4,
        17,
        [("Alice Liu", 5, 10), ("Quinn Reeves", 4, 24), ("Vera Lin", 4, 3)],
        "When did Alice Liu host the garage sale in April?",
    ),
    (
        "Marcus Chen",
        "wedding rehearsal",
        5,
        21,
        [("Marcus Chen", 4, 14), ("Aiden Park", 5, 28), ("Sara Lee", 5, 7)],
        "When did Marcus Chen go to the May wedding rehearsal?",
    ),
    (
        "Mira Khan",
        "data migration cutover",
        5,
        18,
        [("Mira Khan", 4, 11), ("Tom Reed", 5, 25), ("Olivia Roberts", 5, 4)],
        "When did Mira Khan run the data migration cutover in May?",
    ),
    (
        "Diego Lopez",
        "annual physical",
        4,
        23,
        [("Diego Lopez", 5, 16), ("Layla Smith", 4, 30), ("Henry Ford", 4, 9)],
        "When did Diego Lopez have his annual physical in late April?",
    ),
    (
        "Paul Clark",
        "all-hands presentation",
        5,
        26,
        [("Paul Clark", 4, 19), ("Felix Wood", 5, 5), ("Maya Singh", 5, 12)],
        "When did Paul Clark deliver the all-hands presentation in May?",
    ),
    (
        "Vera Lin",
        "moving day",
        4,
        28,
        [("Vera Lin", 5, 21), ("Quinn Reeves", 4, 14), ("Alice Liu", 4, 7)],
        "When did Vera Lin do the moving day in April?",
    ),
    (
        "Yuki Tanaka",
        "scuba certification dive",
        5,
        30,
        [("Yuki Tanaka", 4, 23), ("Aiden Park", 5, 16), ("Sara Lee", 5, 9)],
        "When did Yuki Tanaka complete the scuba certification dive at the end of May?",
    ),
]


def fmt(month: int, day: int) -> str:
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
    return f"{months[month - 1]} {day}, 2024"


def iso(month: int, day: int) -> str:
    return f"2024-{month:02d}-{day:02d}T12:00:00Z"


def main() -> None:
    rng = random.Random(20260429)
    docs = []
    queries = []
    gold_rows = []

    # Each gold spec produces: 1 gold doc, 3 sibling distractor docs.
    # Crucially: one sibling has the SAME PERSON as gold but at a different
    # date (forces T to disambiguate by date, not name).
    for i, (gold_person, key_noun, g_month, g_day, siblings, q_template) in enumerate(
        SPECS
    ):
        gold_id = f"hdc_{i:03d}_g"
        gold_text = f"{gold_person} had a {key_noun} on {fmt(g_month, g_day)}."
        docs.append(
            {
                "doc_id": gold_id,
                "text": gold_text,
                "ref_time": iso(g_month, g_day),
            }
        )

        for j, (sib_person, sib_month, sib_day) in enumerate(siblings):
            sib_id = f"hdc_{i:03d}_s{j}"
            sib_text = f"{sib_person} had a {key_noun} on {fmt(sib_month, sib_day)}."
            docs.append(
                {
                    "doc_id": sib_id,
                    "text": sib_text,
                    "ref_time": iso(sib_month, sib_day),
                }
            )

        # Add 2-3 unrelated noise docs per spec
        noise_actions = [
            "paid the electric bill",
            "watered the plants",
            "took out the recycling",
            "called the dentist",
            "checked the mailbox",
            "walked the dog",
            "made coffee",
            "vacuumed the living room",
        ]
        for n in range(2):
            noise_day = rng.randint(1, 30)
            noise_month = rng.choice([4, 5])
            noise_action = rng.choice(noise_actions)
            noise_id = f"hdc_{i:03d}_n{n}"
            noise_text = f"Someone {noise_action} on {fmt(noise_month, noise_day)}."
            docs.append(
                {
                    "doc_id": noise_id,
                    "text": noise_text,
                    "ref_time": iso(noise_month, noise_day),
                }
            )

        # Query: asks WHEN, references gold person + key noun
        qid = f"hdc_q_{i:03d}"
        queries.append(
            {
                "query_id": qid,
                "text": q_template,
                "ref_time": REF_TIME,
            }
        )
        gold_rows.append({"query_id": qid, "relevant_doc_ids": [gold_id]})

    # Shuffle docs to remove positional artifacts
    rng.shuffle(docs)

    docs_path = DATA_DIR / "hard_dense_cluster_docs.jsonl"
    queries_path = DATA_DIR / "hard_dense_cluster_queries.jsonl"
    gold_path = DATA_DIR / "hard_dense_cluster_gold.jsonl"

    with open(docs_path, "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    with open(queries_path, "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(gold_path, "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)

    print(f"Wrote {len(docs)} docs to {docs_path}")
    print(f"Wrote {len(queries)} queries to {queries_path}")
    print(f"Wrote {len(gold_rows)} gold rows to {gold_path}")
    print("\nDoc breakdown:")
    print(f"  gold: {len(SPECS)}")
    print(f"  same-noun siblings: {sum(len(s[4]) for s in SPECS)}")
    print(f"  noise: {len(SPECS) * 2}")


if __name__ == "__main__":
    main()
