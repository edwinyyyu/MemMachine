"""Generate 15 new docs + 20 axis-specific queries for multi-axis eval.

Axis-specific docs describe patterns that concentrate on ONE axis dimension
(weekday, month, part-of-day, quarter, season, weekend) across many years
— so they cannot be distinguished by interval overlap alone.

Queries ask about those single or two-axis patterns without anchoring to
an absolute year. Gold pairs are defined manually per the axis each doc
expresses.

Writes:
- data/axis_docs.jsonl
- data/axis_queries.jsonl
- data/axis_gold.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2026-04-23T12:00:00Z"


# ---------------------------------------------------------------------------
# Axis docs: each doc encodes a recurrence or a specific period that
# concentrates on ONE or TWO axes.
# ---------------------------------------------------------------------------
DOCS: list[dict[str, Any]] = [
    {
        "doc_id": "axis_doc_thu_run",
        "text": "I go running every Thursday at 6am in the park.",
        "axes_expressed": ["weekday:thu", "hour:06", "part_of_day:morning"],
    },
    {
        "doc_id": "axis_doc_mar_busy",
        "text": "March is my busiest month at work — tax deadlines pile up.",
        "axes_expressed": ["month:mar", "quarter:Q1", "season:spring"],
    },
    {
        "doc_id": "axis_doc_jun_vac",
        "text": "Most of my family vacations happen in June when school lets out.",
        "axes_expressed": ["month:jun", "quarter:Q2", "season:summer"],
    },
    {
        "doc_id": "axis_doc_tue_book",
        "text": "Tuesday evenings are book club at the community center.",
        "axes_expressed": ["weekday:tue", "part_of_day:evening"],
    },
    {
        "doc_id": "axis_doc_fri_off",
        "text": "I almost always take Friday afternoons off to decompress.",
        "axes_expressed": ["weekday:fri", "part_of_day:afternoon"],
    },
    {
        "doc_id": "axis_doc_q2_plan",
        "text": "We hold our quarterly planning review every Q2.",
        "axes_expressed": ["quarter:Q2"],
    },
    {
        "doc_id": "axis_doc_sat_hike",
        "text": "Saturday mornings we hike the coastal trail.",
        "axes_expressed": ["weekday:sat", "part_of_day:morning", "weekend:yes"],
    },
    {
        "doc_id": "axis_doc_sun_brunch",
        "text": "Sundays we do brunch with friends downtown.",
        "axes_expressed": ["weekday:sun", "weekend:yes"],
    },
    {
        "doc_id": "axis_doc_winter_sick",
        "text": "I tend to catch a cold every winter without fail.",
        "axes_expressed": ["season:winter"],
    },
    {
        "doc_id": "axis_doc_oct_harvest",
        "text": "October is when we do the apple harvest up north.",
        "axes_expressed": ["month:oct", "quarter:Q4", "season:autumn"],
    },
    {
        "doc_id": "axis_doc_wed_lunch",
        "text": "Every Wednesday I meet my mentor for lunch at noon.",
        "axes_expressed": ["weekday:wed", "hour:12", "part_of_day:afternoon"],
    },
    {
        "doc_id": "axis_doc_evening_reading",
        "text": "Evenings after 8pm are reserved for reading.",
        "axes_expressed": ["part_of_day:evening", "hour:20"],
    },
    {
        "doc_id": "axis_doc_summer_camp",
        "text": "Every summer the kids go to camp in the mountains.",
        "axes_expressed": ["season:summer"],
    },
    {
        "doc_id": "axis_doc_thu_morning_standup",
        "text": "Thursday mornings at 9am we run the team standup.",
        "axes_expressed": ["weekday:thu", "part_of_day:morning", "hour:09"],
    },
    {
        "doc_id": "axis_doc_dec_holiday",
        "text": "December is always wall-to-wall holiday parties and travel.",
        "axes_expressed": ["month:dec", "quarter:Q4", "season:winter"],
    },
]


# ---------------------------------------------------------------------------
# Axis queries: each targets a subset of the above docs via axis match.
# ---------------------------------------------------------------------------
QUERIES: list[dict[str, Any]] = [
    {
        "query_id": "axis_q_thu",
        "text": "What do I do on Thursdays?",
        "relevant": ["axis_doc_thu_run", "axis_doc_thu_morning_standup"],
    },
    {
        "query_id": "axis_q_mar",
        "text": "What happens in March?",
        "relevant": ["axis_doc_mar_busy"],
    },
    {
        "query_id": "axis_q_afternoon",
        "text": "My afternoon activities?",
        "relevant": [
            "axis_doc_fri_off",
            "axis_doc_wed_lunch",
        ],
    },
    {
        "query_id": "axis_q_weekend",
        "text": "What weekend events do I have?",
        "relevant": ["axis_doc_sat_hike", "axis_doc_sun_brunch"],
    },
    {
        "query_id": "axis_q_q2",
        "text": "Anything in Q2?",
        "relevant": ["axis_doc_q2_plan", "axis_doc_jun_vac"],
    },
    {
        "query_id": "axis_q_tue",
        "text": "Tuesday specials?",
        "relevant": ["axis_doc_tue_book"],
    },
    {
        "query_id": "axis_q_thu_morning",
        "text": "What do I do on Thursday mornings?",
        "relevant": ["axis_doc_thu_morning_standup", "axis_doc_thu_run"],
    },
    {
        "query_id": "axis_q_june_weekends",
        "text": "June weekends?",
        "relevant": ["axis_doc_jun_vac"],
    },
    {
        "query_id": "axis_q_evening",
        "text": "What are my evening activities?",
        "relevant": ["axis_doc_tue_book", "axis_doc_evening_reading"],
    },
    {
        "query_id": "axis_q_morning",
        "text": "What are my morning activities?",
        "relevant": [
            "axis_doc_thu_run",
            "axis_doc_sat_hike",
            "axis_doc_thu_morning_standup",
        ],
    },
    {
        "query_id": "axis_q_summer",
        "text": "Summer events?",
        "relevant": ["axis_doc_jun_vac", "axis_doc_summer_camp"],
    },
    {
        "query_id": "axis_q_winter",
        "text": "What happens in winter?",
        "relevant": ["axis_doc_winter_sick", "axis_doc_dec_holiday"],
    },
    {
        "query_id": "axis_q_autumn",
        "text": "Autumn activities?",
        "relevant": ["axis_doc_oct_harvest"],
    },
    {
        "query_id": "axis_q_q4",
        "text": "What do I do in Q4?",
        "relevant": ["axis_doc_oct_harvest", "axis_doc_dec_holiday"],
    },
    {
        "query_id": "axis_q_fri",
        "text": "What do I do on Fridays?",
        "relevant": ["axis_doc_fri_off"],
    },
    {
        "query_id": "axis_q_saturday",
        "text": "Saturday activities?",
        "relevant": ["axis_doc_sat_hike"],
    },
    {
        "query_id": "axis_q_sunday",
        "text": "Sunday plans?",
        "relevant": ["axis_doc_sun_brunch"],
    },
    {
        "query_id": "axis_q_wed",
        "text": "What do I have on Wednesdays?",
        "relevant": ["axis_doc_wed_lunch"],
    },
    {
        "query_id": "axis_q_weekday_morning",
        "text": "Weekday morning events?",
        "relevant": [
            "axis_doc_thu_run",
            "axis_doc_thu_morning_standup",
        ],
    },
    {
        "query_id": "axis_q_october",
        "text": "Anything in October?",
        "relevant": ["axis_doc_oct_harvest"],
    },
]


def main() -> None:
    # docs.jsonl
    with (DATA_DIR / "axis_docs.jsonl").open("w") as f:
        for d in DOCS:
            out = {
                "doc_id": d["doc_id"],
                "text": d["text"],
                "ref_time": REF_TIME,
                "axes_expressed": d["axes_expressed"],
            }
            f.write(json.dumps(out) + "\n")

    with (DATA_DIR / "axis_queries.jsonl").open("w") as f:
        for q in QUERIES:
            out = {
                "query_id": q["query_id"],
                "text": q["text"],
                "ref_time": REF_TIME,
            }
            f.write(json.dumps(out) + "\n")

    with (DATA_DIR / "axis_gold.jsonl").open("w") as f:
        for q in QUERIES:
            out = {
                "query_id": q["query_id"],
                "relevant_doc_ids": q["relevant"],
            }
            f.write(json.dumps(out) + "\n")

    print(f"Wrote {len(DOCS)} docs, {len(QUERIES)} queries")


if __name__ == "__main__":
    main()
