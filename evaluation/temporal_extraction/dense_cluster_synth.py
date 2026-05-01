"""Dense time-cluster synthetic benchmark.

Generates:
- 100 docs all in April 2024 (dates spread Apr 1-30), each a single specific
  event/fact statement. Diverse activities (meetings, errands, dinners,
  appointments, etc.). Some entity reuse (Dr. Patel in 2-3 docs; Tom in ~5)
  so semantic has signal.
- 30 queries asking by CONTENT (not date), each with exactly 1 gold doc.

Writes to data/dense_cluster_{docs,queries,gold}.jsonl.

Architectural test: T channel should score uniformly high across all April
docs (no disambiguation), S should do the actual ranking. Fusion should
collapse to ~semantic. But untested in pipeline.

Usage: uv run python dense_cluster_synth.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2024-05-01T00:00:00Z"  # ref time AFTER cluster (memo from May 1)


def fmt(day: int, *, with_year: bool = True) -> str:
    """Format April day → 'April D, 2024' or 'April D'."""
    return f"April {day}, 2024" if with_year else f"April {day}"


def doc(text: str, day: int) -> dict:
    return {"text": text, "_day": day}


def build_docs() -> list[dict]:
    """Return 100 (doc_id, text, ref_time) entries."""
    rng = random.Random(20260424)
    docs: list[dict] = []

    # ---- Fixed docs that we will write specific gold-target queries against ----
    # Each gold target appears here exactly once with unique discriminative content.
    gold_specs: list[tuple[str, int]] = [
        # (text, day)
        (f"Dentist appointment with Dr. Patel at 2pm on {fmt(3)}.", 3),
        (
            f"Picked up the package from the post office on {fmt(3)}.",
            3,
        ),  # same day as dentist
        (f"Ran a 5k around the lake on {fmt(7)}.", 7),
        (f"Picked up the annual report draft from Jennifer on {fmt(8)}.", 8),
        (f"Dinner with Tom and Sarah at Trattoria Bianca on {fmt(12)}.", 12),
        (f"Annual checkup with Dr. Morales on {fmt(15)} at 10am.", 15),
        (f"Quarterly review meeting with the product team on {fmt(22)}.", 22),
        (f"Submitted the tax return online on {fmt(14)}.", 14),
        (f"Phone call with Marcus about the contract renewal on {fmt(9)}.", 9),
        (f"Booked flights to Lisbon for the summer trip on {fmt(11)}.", 11),
        (f"Dropped off the cat at the vet for grooming on {fmt(16)}.", 16),
        (f"Picked up the wedding gift for Hannah at the boutique on {fmt(18)}.", 18),
        (f"Yoga class at the studio on Maple Street on {fmt(20)}.", 20),
        (
            f"Shipped the watercolor paintings to the gallery in Brooklyn on {fmt(25)}.",
            25,
        ),
        (f"Lunch with Tom at the noodle place near work on {fmt(4)}.", 4),
        (f"Paid the quarterly estimated taxes on {fmt(17)}.", 17),
        (f"Watched the eclipse with Tom and the kids on {fmt(8)}.", 8),
        (f"Replaced the kitchen faucet with help from Dad on {fmt(13)}.", 13),
        (f"Visited the new modern art exhibit downtown on {fmt(21)}.", 21),
        (f"Got the car inspected and renewed the registration on {fmt(26)}.", 26),
        (f"Coffee with Priya to talk about the startup pitch on {fmt(5)}.", 5),
        (f"Repotted the fiddle leaf fig and the monstera on {fmt(27)}.", 27),
        (f"Volunteer shift at the food bank on {fmt(6)}.", 6),
        (f"Returned the wrong-size jacket to the department store on {fmt(19)}.", 19),
        (f"Helped Tom move his couch to the new apartment on {fmt(23)}.", 23),
        (f"Performance review with my manager Carla on {fmt(24)}.", 24),
        (f"Delivered the final pitch deck to the investors on {fmt(29)}.", 29),
        (
            f"Eye exam with Dr. Patel's colleague Dr. Liu on {fmt(28)}.",
            28,
        ),  # Patel reuse
        (f"Sent the birthday card to Grandma on {fmt(2)}.", 2),
        (f"Cleaned out the garage and donated old tools on {fmt(30)}.", 30),
    ]

    # ---- Distractor pool: events at the SAME day-of-month or with overlapping
    # entities/topics, to ensure semantic actually has to discriminate. ----
    distractor_specs: list[tuple[str, int]] = [
        # Tom-related distractors (ensures semantic-by-name isn't trivial)
        (f"Texted Tom about hiking plans on {fmt(2)}.", 2),
        (f"Tom borrowed the camping tent on {fmt(10)}.", 10),
        (f"Tom returned the borrowed book on {fmt(26)}.", 26),
        # Doctor distractors (Patel and Morales reuse)
        (
            f"Confirmed the cleaning slot at Dr. Patel's office for next month on {fmt(10)}.",
            10,
        ),
        (f"Picked up the prescription Dr. Morales sent in on {fmt(17)}.", 17),
        # Generic same-day distractors (April 3 cluster)
        (f"Replaced the smoke detector battery on {fmt(3)}.", 3),
        (f"Renewed the gym membership at the front desk on {fmt(3)}.", 3),
        # April 12 cluster (gold dinner is on Apr 12)
        (f"Watered the plants and topped off the bird feeder on {fmt(12)}.", 12),
        (f"Mailed the property tax check on {fmt(12)}.", 12),
        (f"Paid the babysitter for last weekend on {fmt(12)}.", 12),
        # April 15 cluster (annual checkup)
        (f"Took out the recycling on {fmt(15)}.", 15),
        (f"Updated the resume and sent it to the recruiter on {fmt(15)}.", 15),
        # Other generic stand-alones, all April 2024
        (f"Picked up grocery delivery from the corner shop on {fmt(1)}.", 1),
        (f"Went for an evening walk along the harbor on {fmt(1)}.", 1),
        (f"Read the new neighborhood newsletter on {fmt(2)}.", 2),
        (f"Took the laundry to the dry cleaner on {fmt(4)}.", 4),
        (f"Reorganized the spice cabinet on {fmt(4)}.", 4),
        (f"Watched a documentary about deep sea exploration on {fmt(5)}.", 5),
        (f"Backed up the laptop to the external drive on {fmt(5)}.", 5),
        (f"Cleaned the coffee machine and descaled it on {fmt(6)}.", 6),
        (f"Wrote a long email to Aunt Beverly on {fmt(6)}.", 6),
        (f"Tried the new ramen place on Eastside on {fmt(7)}.", 7),
        (f"Edited the family photo album on {fmt(7)}.", 7),
        (f"Bought running shoes at the outlet store on {fmt(8)}.", 8),
        (f"Submitted the conference travel reimbursement form on {fmt(9)}.", 9),
        (f"Cooked a big batch of chili for the freezer on {fmt(9)}.", 9),
        (f"Reviewed the apartment lease before signing on {fmt(10)}.", 10),
        (f"Donated old paperback books to the library on {fmt(10)}.", 10),
        (f"Hosted board game night with the neighbors on {fmt(11)}.", 11),
        (f"Caught up on three episodes of the cooking show on {fmt(11)}.", 11),
        (f"Made sourdough starter from scratch on {fmt(13)}.", 13),
        (f"Replaced the bike chain at the bike co-op on {fmt(13)}.", 13),
        (f"Got a haircut at the barbershop on Pine Street on {fmt(14)}.", 14),
        (f"Tried out the new oat milk latte on {fmt(14)}.", 14),
        (f"Reviewed the quarterly budget spreadsheet on {fmt(16)}.", 16),
        (f"Met Priya for tea at the cafe on {fmt(16)}.", 16),
        (f"Returned the borrowed power drill to Eric on {fmt(17)}.", 17),
        (f"Set up the patio chairs for spring on {fmt(18)}.", 18),
        (f"Trimmed the hedge in the front yard on {fmt(18)}.", 18),
        (f"Took the kids to the children's museum on {fmt(19)}.", 19),
        (f"Cleaned out the email inbox down to zero on {fmt(19)}.", 19),
        (f"Fixed the squeaky bedroom door hinge on {fmt(20)}.", 20),
        (f"Met with the real-estate agent about the listing on {fmt(20)}.", 20),
        (f"Ordered new running socks online on {fmt(21)}.", 21),
        (f"Helped Mom set up her new tablet on {fmt(21)}.", 21),
        (f"Updated the household emergency contact list on {fmt(22)}.", 22),
        (f"Returned the library books before they were overdue on {fmt(22)}.", 22),
        (f"Watched the youth orchestra concert at the high school on {fmt(23)}.", 23),
        (f"Cleaned the bathroom grout with the new brush on {fmt(23)}.", 23),
        (f"Bought farmers' market vegetables for the week on {fmt(24)}.", 24),
        (f"Sent a thank-you card to the dinner hosts on {fmt(24)}.", 24),
        (f"Started the small herb garden on the balcony on {fmt(25)}.", 25),
        (f"Filed the medical receipts in the binder on {fmt(25)}.", 25),
        (f"Played pickup basketball at the rec center on {fmt(26)}.", 26),
        (f"Practiced piano for an hour after dinner on {fmt(27)}.", 27),
        (f"Reorganized the bookshelf by color on {fmt(27)}.", 27),
        (f"Repaired the loose handle on the suitcase on {fmt(28)}.", 28),
        (f"Cleared out the backyard storage shed on {fmt(28)}.", 28),
        (f"Took a long bath and finished a paperback on {fmt(29)}.", 29),
        (f"Tested the smoke alarms throughout the house on {fmt(29)}.", 29),
        (f"Mowed the front and back lawn on {fmt(30)}.", 30),
        (f"Bought potting soil and new gardening gloves on {fmt(30)}.", 30),
        (f"Ran errands at the hardware store on {fmt(15)}.", 15),
        (f"Listened to the new podcast episode while walking on {fmt(11)}.", 11),
        (f"Cleaned the windows on the front porch on {fmt(6)}.", 6),
        (f"Re-strung the acoustic guitar on {fmt(2)}.", 2),
        (f"Joined the photography meetup at the park on {fmt(7)}.", 7),
        (f"Wrote postcards from the local stationery store on {fmt(5)}.", 5),
        (f"Brewed a fresh batch of cold brew coffee on {fmt(8)}.", 8),
        (f"Sorted through old mail and shredded the junk on {fmt(12)}.", 12),
    ]

    # 30 gold + 70 distractors = 100 docs
    all_specs = list(gold_specs) + list(distractor_specs)
    assert len(gold_specs) == 30, len(gold_specs)
    # Trim distractors to exactly 70
    distractor_specs = distractor_specs[:70]
    all_specs = list(gold_specs) + list(distractor_specs)
    assert len(all_specs) == 100, len(all_specs)

    # Shuffle the order so gold docs aren't all up front
    rng.shuffle(all_specs)

    out: list[dict] = []
    gold_pairs: list[tuple[str, str]] = []  # (gold_text, doc_id)
    gold_text_set = {t for t, _ in gold_specs}
    for i, (text, day) in enumerate(all_specs):
        did = f"d_{i:03d}"
        out.append({"doc_id": did, "text": text, "ref_time": REF_TIME})
        if text in gold_text_set:
            gold_pairs.append((text, did))
    return out, gold_pairs


def build_queries(gold_pairs: list[tuple[str, str]]) -> tuple[list[dict], list[dict]]:
    """30 queries by CONTENT, each with exactly 1 gold doc."""
    # Map gold_text -> probe query
    probes: dict[str, str] = {
        "Dentist appointment with Dr. Patel": "When was my dentist appointment with Dr. Patel?",
        "Picked up the package from the post office": "What did I pick up from the post office?",
        "Ran a 5k around the lake": "When did I run the 5k around the lake?",
        "Picked up the annual report draft": "When did I pick up the annual report draft from Jennifer?",
        "Dinner with Tom and Sarah at Trattoria Bianca": "When was the dinner with Tom and Sarah at Trattoria Bianca?",
        "Annual checkup with Dr. Morales": "When was my annual checkup with Dr. Morales?",
        "Quarterly review meeting with the product team": "When was the quarterly review meeting with the product team?",
        "Submitted the tax return online": "When did I submit my tax return?",
        "Phone call with Marcus about the contract renewal": "When did I have the call with Marcus about the contract renewal?",
        "Booked flights to Lisbon": "When did I book the Lisbon flights?",
        "Dropped off the cat at the vet for grooming": "When did I drop off the cat for grooming?",
        "Picked up the wedding gift for Hannah": "When did I pick up the wedding gift for Hannah?",
        "Yoga class at the studio on Maple Street": "When was my yoga class on Maple Street?",
        "Shipped the watercolor paintings": "When did I ship the watercolor paintings to the Brooklyn gallery?",
        "Lunch with Tom at the noodle place": "When was lunch with Tom at the noodle place?",
        "Paid the quarterly estimated taxes": "When did I pay the quarterly estimated taxes?",
        "Watched the eclipse with Tom and the kids": "When did we watch the eclipse?",
        "Replaced the kitchen faucet with help from Dad": "When did Dad help me replace the kitchen faucet?",
        "Visited the new modern art exhibit downtown": "When did I visit the modern art exhibit?",
        "Got the car inspected and renewed the registration": "When did I get the car inspected?",
        "Coffee with Priya to talk about the startup pitch": "When did I have coffee with Priya about the startup pitch?",
        "Repotted the fiddle leaf fig and the monstera": "When did I repot the fiddle leaf fig?",
        "Volunteer shift at the food bank": "When was my food bank volunteer shift?",
        "Returned the wrong-size jacket": "When did I return the wrong-size jacket?",
        "Helped Tom move his couch": "When did I help Tom move his couch?",
        "Performance review with my manager Carla": "When was my performance review with Carla?",
        "Delivered the final pitch deck to the investors": "When did I deliver the pitch deck to the investors?",
        "Eye exam with Dr. Patel's colleague Dr. Liu": "When was my eye exam with Dr. Liu?",
        "Sent the birthday card to Grandma": "When did I send Grandma's birthday card?",
        "Cleaned out the garage and donated old tools": "When did I clean out the garage?",
    }

    queries: list[dict] = []
    gold: list[dict] = []
    for i, (gold_text, did) in enumerate(gold_pairs):
        # Find the matching probe by checking which probe key is a substring of gold_text
        match_key = next((k for k in probes if k in gold_text), None)
        assert match_key is not None, f"No probe for gold text: {gold_text}"
        qtext = probes[match_key]
        qid = f"q_{i:03d}"
        # Query ref_time same as docs (the journal owner asks "when did X happen")
        queries.append(
            {
                "query_id": qid,
                "text": qtext,
                "ref_time": REF_TIME,
                "subset": "dense_cluster",
            }
        )
        gold.append({"query_id": qid, "relevant_doc_ids": [did]})
    return queries, gold


def main() -> None:
    docs, gold_pairs = build_docs()
    queries, gold = build_queries(gold_pairs)

    docs_path = DATA_DIR / "dense_cluster_docs.jsonl"
    q_path = DATA_DIR / "dense_cluster_queries.jsonl"
    g_path = DATA_DIR / "dense_cluster_gold.jsonl"
    docs_path.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    q_path.write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    g_path.write_text("\n".join(json.dumps(g) for g in gold) + "\n")

    # Stats
    print(f"Docs: {len(docs)} written to {docs_path}")
    print(f"Queries: {len(queries)} written to {q_path}")
    print(f"Gold: {len(gold)} written to {g_path}")

    # Entity reuse sanity check
    text_blob = " ".join(d["text"] for d in docs)
    import re as _re

    tom_n = len(_re.findall(r"\bTom\b", text_blob))
    print(f"  Tom mentions: {tom_n}")
    print(f"  Dr. Patel mentions: {text_blob.count('Dr. Patel')}")
    print(f"  Dr. Morales mentions: {text_blob.count('Dr. Morales')}")
    print(f"  Priya mentions: {text_blob.count('Priya')}")

    # Day distribution
    from collections import Counter

    days = []
    for d in docs:
        # parse "April N, 2024"
        import re

        m = re.search(r"April (\d+), 2024", d["text"])
        if m:
            days.append(int(m.group(1)))
    cnt = Counter(days)
    print(f"  Day range: {min(days)}..{max(days)}, distinct days: {len(cnt)}")
    print(f"  Per-day count: {dict(sorted(cnt.items()))}")


if __name__ == "__main__":
    main()
