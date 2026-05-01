"""Synthetic lattice-test corpus.

Three test families:

1. Cross-precision retrieval (15-20 docs, 15-20 queries):
   - narrow-query / broad-doc:  "What happened on March 15, 2015?"
     should find doc "things were tough in 2015" (year-precise).
   - broad-query / narrow-doc:  "anything from the 90s?"
     should find doc "Jan 1, 1999 — celebration".
   - same-precision matches across day/month/year/decade.

2. Cyclical-only queries (5-10 queries):
   - "Thursday events?"   -> docs tagged weekday:Thursday
   - "afternoon meetings?" -> docs tagged hour_of_day ~13-17 or part_of_day:afternoon
   - "March of any year"   -> docs tagged month_of_year:March

3. Doc-doc sharing (S8-style) — queries need to find cross-doc evidence
   via shared-year / shared-decade lattice cells:
   - q: "when did I meet my wife?"
     docs: A ("met my wife at the 2012 retreat" year:2012),
           C ("2012 retreat was unforgettable" year:2012)
     both relevant — shared year:2012 cell.

Writes:
   data/lattice_docs.jsonl
   data/lattice_queries.jsonl
   data/lattice_gold.jsonl

These are DOMAIN-NEUTRAL. Extraction is done LATER by the evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2026-04-24T12:00:00Z"


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
# Each doc: id, text, ref_time, precision_hint (for reporting), family
DOCS = [
    # ---- day-precise ----
    {
        "id": "lat_doc_day_jan1_1999",
        "text": "On January 1, 1999 we hosted a millennium-eve countdown party with friends.",
        "precision": "day",
        "family": "day_precise",
    },
    {
        "id": "lat_doc_day_mar15_2015",
        "text": "March 15, 2015 was the day I submitted the dissertation.",
        "precision": "day",
        "family": "day_precise",
    },
    {
        "id": "lat_doc_day_jul4_2020",
        "text": "On July 4, 2020 the neighborhood block party was cancelled due to rain.",
        "precision": "day",
        "family": "day_precise",
    },
    {
        "id": "lat_doc_day_dec25_1995",
        "text": "December 25, 1995 was the Christmas we all drove to the lake house.",
        "precision": "day",
        "family": "day_precise",
    },
    {
        "id": "lat_doc_day_sep11_2001",
        "text": "On September 11, 2001 I was stuck in a train tunnel for an hour.",
        "precision": "day",
        "family": "day_precise",
    },
    # ---- month-precise ----
    {
        "id": "lat_doc_mar2020",
        "text": "March 2020 was when everyone started working from home for the first time.",
        "precision": "month",
        "family": "month_precise",
    },
    {
        "id": "lat_doc_jun2018",
        "text": "June 2018 was the month I moved into the new apartment.",
        "precision": "month",
        "family": "month_precise",
    },
    {
        "id": "lat_doc_nov1999",
        "text": "November 1999 was rainy and cold; I remember it well.",
        "precision": "month",
        "family": "month_precise",
    },
    # ---- year-precise ----
    {
        "id": "lat_doc_y2015_tough",
        "text": "2015 was a tough year — lots of transitions and moves.",
        "precision": "year",
        "family": "year_precise",
    },
    {
        "id": "lat_doc_y2012_wife_met",
        "text": "I met my wife at the 2012 retreat and we talked the whole weekend.",
        "precision": "year",
        "family": "year_precise_s8",
    },
    {
        "id": "lat_doc_y2012_retreat",
        "text": "The 2012 retreat was unforgettable — great workshops and hikes.",
        "precision": "year",
        "family": "year_precise_s8",
    },
    {
        "id": "lat_doc_y1999_milestone",
        "text": "1999 was a milestone year for our team — we shipped two major products.",
        "precision": "year",
        "family": "year_precise",
    },
    # ---- decade-precise ----
    {
        "id": "lat_doc_1990s_dialup",
        "text": "Back in the 1990s, dial-up modems were the norm and websites loaded slowly.",
        "precision": "decade",
        "family": "decade_precise",
    },
    {
        "id": "lat_doc_1980s_mixtapes",
        "text": "The 1980s were the era of mixtapes and giant shoulder pads.",
        "precision": "decade",
        "family": "decade_precise",
    },
    {
        "id": "lat_doc_2000s_broadband",
        "text": "The 2000s brought broadband and the rise of social media.",
        "precision": "decade",
        "family": "decade_precise",
    },
    # ---- cyclical-only / recurrence ----
    {
        "id": "lat_doc_thu_standup",
        "text": "Every Thursday at 10am we have our team standup meeting.",
        "precision": "recurrence",
        "family": "cyclical_weekday",
    },
    {
        "id": "lat_doc_thu_run",
        "text": "I go running every Thursday evening along the river path.",
        "precision": "recurrence",
        "family": "cyclical_weekday",
    },
    {
        "id": "lat_doc_afternoon_coffee",
        "text": "Every afternoon around 3pm I take a short coffee break.",
        "precision": "recurrence",
        "family": "cyclical_partofday",
    },
    {
        "id": "lat_doc_afternoon_walk",
        "text": "On most afternoons I walk the dog before dinner.",
        "precision": "recurrence",
        "family": "cyclical_partofday",
    },
    {
        "id": "lat_doc_march_busy",
        "text": "March is always the busiest month for our tax team.",
        "precision": "recurrence",
        "family": "cyclical_monthofyear",
    },
    {
        "id": "lat_doc_march_garden",
        "text": "Every March I start seedlings indoors for the garden.",
        "precision": "recurrence",
        "family": "cyclical_monthofyear",
    },
    # ---- distractors (unrelated temporal content) ----
    {
        "id": "lat_doc_distractor_2024",
        "text": "In 2024 I finally learned how to bake sourdough bread at home.",
        "precision": "year",
        "family": "distractor",
    },
    {
        "id": "lat_doc_distractor_jul2023",
        "text": "July 2023 was when the heat wave finally broke.",
        "precision": "month",
        "family": "distractor",
    },
    {
        "id": "lat_doc_distractor_1970s",
        "text": "The 1970s had some incredible music and fashion trends.",
        "precision": "decade",
        "family": "distractor",
    },
    {
        "id": "lat_doc_distractor_tuesday",
        "text": "Every Tuesday we have bowling night at the lanes.",
        "precision": "recurrence",
        "family": "distractor",
    },
]


# ---------------------------------------------------------------------------
# Queries + gold
# ---------------------------------------------------------------------------
QUERIES = [
    # ----- narrow-query / broad-doc -----
    # "March 15, 2015" (day-precise query) should find year-precise doc via UP walk
    {
        "id": "lat_q_mar15_2015",
        "text": "What happened on March 15, 2015?",
        "gold": ["lat_doc_day_mar15_2015", "lat_doc_y2015_tough"],
        "subset": "narrow_query_broad_doc",
    },
    # "January 1, 1999" day query should find decade-precise 1990s docs via UP
    {
        "id": "lat_q_jan1_1999",
        "text": "What happened on January 1, 1999?",
        "gold": [
            "lat_doc_day_jan1_1999",
            "lat_doc_y1999_milestone",
            "lat_doc_1990s_dialup",
            "lat_doc_nov1999",
        ],
        "subset": "narrow_query_broad_doc",
    },
    # "July 4, 2020" day-precise should find distractor nothing besides itself
    {
        "id": "lat_q_jul4_2020",
        "text": "What did I do on July 4, 2020?",
        "gold": ["lat_doc_day_jul4_2020"],
        "subset": "narrow_query_broad_doc",
    },
    # "December 25, 1995" day -> should find itself and 1990s decade doc
    {
        "id": "lat_q_dec25_1995",
        "text": "What happened on December 25, 1995?",
        "gold": ["lat_doc_day_dec25_1995", "lat_doc_1990s_dialup"],
        "subset": "narrow_query_broad_doc",
    },
    # "September 11, 2001" day -> itself + 2000s decade doc
    {
        "id": "lat_q_sep11_2001",
        "text": "What happened on September 11, 2001?",
        "gold": ["lat_doc_day_sep11_2001", "lat_doc_2000s_broadband"],
        "subset": "narrow_query_broad_doc",
    },
    # ----- broad-query / narrow-doc -----
    # "anything from the 90s?" should find 1990s doc + year 1999 doc + day 1999 + Nov 1999 + Dec 1995
    {
        "id": "lat_q_the_90s",
        "text": "Anything from the 90s?",
        "gold": [
            "lat_doc_1990s_dialup",
            "lat_doc_y1999_milestone",
            "lat_doc_day_jan1_1999",
            "lat_doc_nov1999",
            "lat_doc_day_dec25_1995",
        ],
        "subset": "broad_query_narrow_doc",
    },
    # "anything in 2015?" should find year:2015 + day:2015-03-15
    {
        "id": "lat_q_in_2015",
        "text": "Anything that happened in 2015?",
        "gold": ["lat_doc_y2015_tough", "lat_doc_day_mar15_2015"],
        "subset": "broad_query_narrow_doc",
    },
    # "in the 1980s?" -> 1980s decade doc only (no narrower)
    {
        "id": "lat_q_the_80s",
        "text": "What about the 1980s?",
        "gold": ["lat_doc_1980s_mixtapes"],
        "subset": "broad_query_narrow_doc",
    },
    # "anything in 2000?" -> 2000s decade doc (via DOWN walk) + Sep11 2001 (no - 2001 not in 2000)
    # actually 2000 != 2000s decade. Let's skip that.
    # "anything in 2020?" -> March 2020 doc via UP walk
    {
        "id": "lat_q_in_2020",
        "text": "Anything from 2020?",
        "gold": ["lat_doc_mar2020", "lat_doc_day_jul4_2020"],
        "subset": "broad_query_narrow_doc",
    },
    # "in the 2000s?" -> 2000s decade + Sep11 2001
    {
        "id": "lat_q_the_2000s",
        "text": "What about the 2000s?",
        "gold": ["lat_doc_2000s_broadband", "lat_doc_day_sep11_2001"],
        "subset": "broad_query_narrow_doc",
    },
    # ----- same-precision -----
    # "March 2020" month-precise query -> matches March 2020 month doc
    {
        "id": "lat_q_mar2020",
        "text": "What happened in March 2020?",
        "gold": ["lat_doc_mar2020"],
        "subset": "same_precision",
    },
    # "June 2018"
    {
        "id": "lat_q_jun2018",
        "text": "What about June 2018?",
        "gold": ["lat_doc_jun2018"],
        "subset": "same_precision",
    },
    # "2012"
    {
        "id": "lat_q_2012",
        "text": "What happened in 2012?",
        "gold": ["lat_doc_y2012_wife_met", "lat_doc_y2012_retreat"],
        "subset": "same_precision_s8",
    },
    # ----- cyclical -----
    {
        "id": "lat_q_thursdays",
        "text": "What do I do on Thursdays?",
        "gold": ["lat_doc_thu_standup", "lat_doc_thu_run"],
        "subset": "cyclical",
    },
    {
        "id": "lat_q_afternoons",
        "text": "What are my afternoon activities?",
        "gold": ["lat_doc_afternoon_coffee", "lat_doc_afternoon_walk"],
        "subset": "cyclical",
    },
    {
        "id": "lat_q_march_any",
        "text": "What happens in March each year?",
        "gold": ["lat_doc_march_busy", "lat_doc_march_garden"],
        "subset": "cyclical",
    },
    {
        "id": "lat_q_tuesday",
        "text": "Anything on Tuesdays?",
        "gold": ["lat_doc_distractor_tuesday"],
        "subset": "cyclical",
    },
    # mixed-cyclical + date
    {
        "id": "lat_q_morning_1pm",
        "text": "Do I have any 3pm habits?",
        "gold": ["lat_doc_afternoon_coffee"],
        "subset": "cyclical",
    },
    # ----- S8 cross-doc linking -----
    # "when did I meet my wife?" -> both 2012 docs should link (shared year:2012)
    {
        "id": "lat_q_meet_wife",
        "text": "When did I meet my wife?",
        "gold": ["lat_doc_y2012_wife_met", "lat_doc_y2012_retreat"],
        "subset": "s8_crossdoc",
    },
    # "when did the 2012 retreat happen?" linked
    {
        "id": "lat_q_2012_retreat",
        "text": "When was the 2012 retreat?",
        "gold": ["lat_doc_y2012_retreat", "lat_doc_y2012_wife_met"],
        "subset": "s8_crossdoc",
    },
]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
def main() -> None:
    docs_out = []
    for d in DOCS:
        docs_out.append(
            {
                "doc_id": d["id"],
                "text": d["text"],
                "ref_time": REF_TIME,
                "precision": d["precision"],
                "family": d["family"],
            }
        )

    queries_out = []
    gold_out = []
    for q in QUERIES:
        queries_out.append(
            {
                "query_id": q["id"],
                "text": q["text"],
                "ref_time": REF_TIME,
                "subset": q["subset"],
            }
        )
        gold_out.append(
            {
                "query_id": q["id"],
                "relevant_doc_ids": q["gold"],
                "subset": q["subset"],
            }
        )

    with (DATA_DIR / "lattice_docs.jsonl").open("w") as f:
        for d in docs_out:
            f.write(json.dumps(d) + "\n")
    with (DATA_DIR / "lattice_queries.jsonl").open("w") as f:
        for q in queries_out:
            f.write(json.dumps(q) + "\n")
    with (DATA_DIR / "lattice_gold.jsonl").open("w") as f:
        for g in gold_out:
            f.write(json.dumps(g) + "\n")

    print(
        f"Wrote {len(docs_out)} docs, {len(queries_out)} queries, {len(gold_out)} gold."
    )
    # Subset summary
    from collections import Counter

    subs = Counter(q["subset"] for q in queries_out)
    print("Subsets:", dict(subs))
    fams = Counter(d["family"] for d in docs_out)
    print("Families:", dict(fams))


if __name__ == "__main__":
    main()
