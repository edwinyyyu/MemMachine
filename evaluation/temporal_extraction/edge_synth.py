"""Generate 4 small adversarial benchmarks targeting T_lblend failure modes:
1. era_refs        — query: era reference; doc: explicit dates
2. relative_time   — query: "3 weeks ago" relative to ref_time; doc: dates
3. conjunctive_temporal — query: TWO temporal anchors; gold doc covers both
4. multi_te_doc    — doc: 5+ dates each (long meeting notes); query: single date
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


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


# =============================================================================
# Benchmark 1: era_refs — query references named era, doc has explicit date
# =============================================================================
def gen_era_refs():
    """Each cluster: a person + an "era" (interval bound by life events).
    Within the era → gold doc(s).
    Outside the era → distractor docs about same person/topic at different times.
    Query references the era name, NOT a date.
    """
    REF_TIME = "2025-04-01T00:00:00Z"

    # (person, topic, era_name, era_start (m,y), era_end (m,y), gold_date (m,d,y), distractor_dates)
    CLUSTERS = [
        (
            "Sarah Park",
            "took her European backpacking trip",
            "during grad school",
            (9, 2018),
            (5, 2020),
            (3, 12, 2019),
            [(2, 8, 2017), (10, 22, 2021), (6, 14, 2022), (4, 5, 2016)],
        ),
        (
            "Marcus Davis",
            "hosted his first dinner party",
            "back when I worked at Acme",
            (1, 2015),
            (12, 2017),
            (11, 8, 2016),
            [(3, 10, 2014), (5, 21, 2018), (8, 30, 2020), (1, 4, 2019)],
        ),
        (
            "Priya Johnson",
            "completed her marathon",
            "while living in Sweden",
            (6, 2019),
            (8, 2022),
            (5, 17, 2021),
            [(4, 2, 2018), (10, 9, 2023), (3, 15, 2024), (12, 1, 2017)],
        ),
        (
            "Kim Patel",
            "got the promotion",
            "during my time at the startup",
            (3, 2020),
            (9, 2022),
            (8, 14, 2021),
            [(11, 20, 2019), (4, 5, 2023), (7, 18, 2024), (2, 11, 2018)],
        ),
        (
            "Aiden Park",
            "had his first child",
            "back in college",
            (9, 2010),
            (5, 2014),
            (10, 4, 2012),
            [(2, 17, 2009), (5, 24, 2015), (7, 9, 2017), (12, 30, 2008)],
        ),
        (
            "Olivia Roberts",
            "bought her first car",
            "when I lived in Boston",
            (6, 2016),
            (4, 2019),
            (1, 6, 2018),
            [(4, 13, 2015), (9, 20, 2020), (11, 4, 2021), (6, 29, 2014)],
        ),
        (
            "Henry Ford",
            "started learning piano",
            "during my parental leave",
            (3, 2022),
            (9, 2022),
            (5, 9, 2022),
            [(3, 16, 2020), (5, 23, 2023), (12, 30, 2021), (10, 9, 2024)],
        ),
        (
            "Felix Wood",
            "ran the half-marathon",
            "while training for the Olympics",
            (1, 2019),
            (7, 2020),
            (4, 11, 2020),
            [(8, 18, 2018), (1, 4, 2022), (11, 25, 2017), (3, 7, 2024)],
        ),
        (
            "Quinn Reeves",
            "adopted his dog",
            "during the pandemic year",
            (3, 2020),
            (3, 2021),
            (7, 7, 2020),
            [(8, 14, 2019), (10, 21, 2022), (2, 28, 2024), (12, 5, 2018)],
        ),
        (
            "Sara Lee",
            "moved into her first apartment",
            "right after I graduated college",
            (5, 2019),
            (12, 2019),
            (8, 12, 2019),
            [(2, 5, 2017), (10, 19, 2021), (8, 26, 2023), (3, 8, 2025)],
        ),
        (
            "Tom Reed",
            "switched to a vegetarian diet",
            "back when I worked at Globex",
            (4, 2017),
            (10, 2020),
            (2, 15, 2019),
            [(11, 8, 2016), (3, 22, 2021), (7, 29, 2023), (1, 16, 2025)],
        ),
        (
            "Layla Smith",
            "ran her first 10K",
            "during my fitness phase",
            (1, 2021),
            (12, 2021),
            (9, 4, 2021),
            [(11, 11, 2019), (3, 18, 2022), (7, 25, 2023), (10, 2, 2020)],
        ),
    ]

    docs = []
    queries = []
    gold_rows = []
    for i, (person, topic, era, e_start, e_end, gold_date, distractors) in enumerate(
        CLUSTERS
    ):
        gm, gd, gy = gold_date
        gid = f"era_{i:03d}_g"
        docs.append(
            {
                "doc_id": gid,
                "text": f"{person} {topic} on {fmt_date(gm, gd, gy)}.",
                "ref_time": iso(gm, gd, gy),
            }
        )
        for j, (sm, sd, sy) in enumerate(distractors):
            sid = f"era_{i:03d}_d{j}"
            docs.append(
                {
                    "doc_id": sid,
                    "text": f"{person} {topic} on {fmt_date(sm, sd, sy)}.",
                    "ref_time": iso(sm, sd, sy),
                }
            )
        qid = f"era_q_{i:03d}"
        # phrase the query without restating the topic verbatim — we want T to do the heavy lifting
        verb = topic
        for prefix, replacement in [
            ("took her", "take her"),
            ("hosted his", "host his"),
            ("completed her", "complete her"),
            ("got the", "get the"),
            ("had his", "have his"),
            ("bought her", "buy her"),
            ("started", "start"),
            ("ran the", "run the"),
            ("adopted his", "adopt his"),
            ("moved into her", "move into her"),
            ("switched to", "switch to"),
            ("ran her", "run her"),
        ]:
            if topic.startswith(prefix):
                verb = replacement + topic[len(prefix) :]
                break
        q_text = f"When did {person} {verb} {era}?"
        queries.append({"query_id": qid, "text": q_text, "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": [gid]})

    return docs, queries, gold_rows, "era_refs"


# =============================================================================
# Benchmark 2: relative_time — query uses relative phrases anchored to ref_time
# =============================================================================
def gen_relative_time():
    """Query ref_time = 2025-04-15. Relative anchors map to specific windows.
    Gold doc dates fall in the window; distractors fall outside.
    """
    REF_TIME = "2025-04-15T00:00:00Z"

    # (person, topic, relative_phrase, gold_date, distractor_dates)
    CLUSTERS = [
        # "yesterday" → 2025-04-14
        (
            "Sarah Park",
            "renewed her gym membership",
            "yesterday",
            (4, 14, 2025),
            [(4, 1, 2025), (3, 28, 2025), (5, 5, 2025), (12, 14, 2024)],
        ),
        # "last week" → 2025-04-07 to 2025-04-13
        (
            "Marcus Davis",
            "had his quarterly check-in",
            "last week",
            (4, 9, 2025),
            [(3, 20, 2025), (4, 22, 2025), (5, 1, 2025), (2, 15, 2025)],
        ),
        # "3 weeks ago" → ~2025-03-25
        (
            "Priya Johnson",
            "led the design review",
            "three weeks ago",
            (3, 25, 2025),
            [(4, 12, 2025), (2, 10, 2025), (1, 30, 2025), (5, 20, 2025)],
        ),
        # "last month" → 2025-03
        (
            "Kim Patel",
            "completed the audit prep",
            "last month",
            (3, 18, 2025),
            [(1, 5, 2025), (4, 8, 2025), (5, 22, 2025), (10, 30, 2024)],
        ),
        # "two months ago" → ~2025-02
        (
            "Aiden Park",
            "gave his TED talk",
            "two months ago",
            (2, 14, 2025),
            [(4, 1, 2025), (12, 5, 2024), (5, 20, 2025), (1, 3, 2025)],
        ),
        # "earlier this year" → 2025-01 to 2025-04-15
        (
            "Olivia Roberts",
            "started her new role",
            "earlier this year",
            (1, 22, 2025),
            [(11, 4, 2024), (5, 30, 2025), (8, 18, 2024), (10, 9, 2023)],
        ),
        # "last year" → 2024
        (
            "Henry Ford",
            "had his annual review",
            "last year",
            (8, 14, 2024),
            [(3, 16, 2023), (5, 23, 2025), (12, 30, 2022), (1, 10, 2025)],
        ),
        # "a few days ago" → ~2025-04-10 to 2025-04-14
        (
            "Felix Wood",
            "submitted the manuscript",
            "a few days ago",
            (4, 12, 2025),
            [(3, 20, 2025), (4, 25, 2025), (5, 5, 2025), (1, 8, 2025)],
        ),
        # "last quarter" → Q1 2025 (Jan-Mar)
        (
            "Quinn Reeves",
            "closed the funding round",
            "last quarter",
            (2, 28, 2025),
            [(4, 8, 2025), (10, 15, 2024), (5, 1, 2025), (12, 5, 2023)],
        ),
        # "this past summer" → 2024 Jun-Aug
        (
            "Sara Lee",
            "took her sabbatical",
            "this past summer",
            (7, 8, 2024),
            [(3, 5, 2025), (10, 19, 2023), (4, 26, 2024), (11, 14, 2024)],
        ),
        # "yesterday afternoon" → 2025-04-14
        (
            "Tom Reed",
            "wrapped the all-hands",
            "yesterday afternoon",
            (4, 14, 2025),
            [(4, 7, 2025), (3, 22, 2025), (5, 4, 2025), (1, 16, 2025)],
        ),
        # "earlier this month" → 2025-04-01 to 2025-04-14
        (
            "Layla Smith",
            "filed the patent",
            "earlier this month",
            (4, 4, 2025),
            [(3, 18, 2025), (5, 2, 2025), (10, 22, 2024), (1, 13, 2025)],
        ),
    ]

    docs = []
    queries = []
    gold_rows = []
    for i, (person, topic, rel, gold_date, distractors) in enumerate(CLUSTERS):
        gm, gd, gy = gold_date
        gid = f"rel_{i:03d}_g"
        docs.append(
            {
                "doc_id": gid,
                "text": f"{person} {topic} on {fmt_date(gm, gd, gy)}.",
                "ref_time": iso(gm, gd, gy),
            }
        )
        for j, (sm, sd, sy) in enumerate(distractors):
            sid = f"rel_{i:03d}_d{j}"
            docs.append(
                {
                    "doc_id": sid,
                    "text": f"{person} {topic} on {fmt_date(sm, sd, sy)}.",
                    "ref_time": iso(sm, sd, sy),
                }
            )
        qid = f"rel_q_{i:03d}"
        # extract verb in present
        verb_map = {
            "renewed her": "renew her",
            "had his": "have his",
            "led the": "lead the",
            "completed the": "complete the",
            "gave his": "give his",
            "started her": "start her",
            "submitted the": "submit the",
            "closed the": "close the",
            "took her": "take her",
            "wrapped the": "wrap the",
            "filed the": "file the",
        }
        verb = topic
        for prefix, repl in verb_map.items():
            if topic.startswith(prefix):
                verb = repl + topic[len(prefix) :]
                break
        q_text = f"When did {person} {verb}, {rel}?"
        queries.append({"query_id": qid, "text": q_text, "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": [gid]})

    return docs, queries, gold_rows, "relative_time"


# =============================================================================
# Benchmark 3: conjunctive_temporal — query has TWO temporal anchors; gold covers BOTH
# =============================================================================
def gen_conjunctive_temporal():
    """Each cluster has:
      - GOLD: doc covering BOTH temporal anchors (mentions both dates)
      - 2 SINGLE distractors: each covers ONE of the anchors (correctly)
      - 2 OTHER distractors: cover unrelated dates
    Query asks about the conjunction.
    """
    REF_TIME = "2025-04-01T00:00:00Z"

    # (person, topic, anchor1_phrase, date1, anchor2_phrase, date2, anchor_query_phrase)
    CLUSTERS = [
        (
            "Sarah Park",
            "her dental appointments",
            "Q3 2023",
            (8, 12, 2023),
            "Q1 2024",
            (2, 18, 2024),
            "Q3 2023 and Q1 2024",
        ),
        (
            "Marcus Davis",
            "his quarterly reviews",
            "March 2024",
            (3, 15, 2024),
            "August 2024",
            (8, 20, 2024),
            "between March and August 2024",
        ),
        (
            "Priya Johnson",
            "her client visits",
            "January 2023",
            (1, 22, 2023),
            "October 2023",
            (10, 8, 2023),
            "January and October 2023",
        ),
        (
            "Kim Patel",
            "kitchen renovations",
            "summer 2022",
            (7, 14, 2022),
            "winter 2023",
            (12, 8, 2023),
            "summer 2022 and winter 2023",
        ),
        (
            "Aiden Park",
            "his investor meetings",
            "April 2024",
            (4, 10, 2024),
            "November 2024",
            (11, 15, 2024),
            "April and November of 2024",
        ),
        (
            "Olivia Roberts",
            "yoga classes",
            "spring 2024",
            (4, 6, 2024),
            "fall 2024",
            (10, 20, 2024),
            "both spring and fall 2024",
        ),
        (
            "Henry Ford",
            "performance reviews",
            "Q2 2023",
            (5, 9, 2023),
            "Q4 2024",
            (11, 12, 2024),
            "Q2 2023 and Q4 2024",
        ),
        (
            "Felix Wood",
            "client onsites",
            "May 2023",
            (5, 11, 2023),
            "September 2024",
            (9, 4, 2024),
            "May 2023 and September 2024",
        ),
        (
            "Quinn Reeves",
            "lease signings",
            "early 2023",
            (2, 7, 2023),
            "late 2024",
            (11, 28, 2024),
            "early 2023 and late 2024",
        ),
        (
            "Sara Lee",
            "puppy training sessions",
            "March 2024",
            (3, 12, 2024),
            "June 2024",
            (6, 19, 2024),
            "March and June 2024",
        ),
        (
            "Tom Reed",
            "marathons",
            "April 2023",
            (4, 15, 2023),
            "October 2024",
            (10, 27, 2024),
            "April 2023 and October 2024",
        ),
        (
            "Layla Smith",
            "tax filings",
            "April 2023",
            (4, 4, 2023),
            "April 2024",
            (4, 4, 2024),
            "both April 2023 and April 2024",
        ),
    ]

    docs = []
    queries = []
    gold_rows = []
    for i, (person, topic, ph1, d1, ph2, d2, q_anchor) in enumerate(CLUSTERS):
        m1, day1, y1 = d1
        m2, day2, y2 = d2
        # GOLD: covers BOTH dates explicitly
        gid = f"conj_{i:03d}_g"
        docs.append(
            {
                "doc_id": gid,
                "text": (
                    f"{person} had {topic} on {fmt_date(m1, day1, y1)} "
                    f"and on {fmt_date(m2, day2, y2)}."
                ),
                "ref_time": iso(m1, day1, y1),  # ref_time is first date
            }
        )
        # SINGLE-anchor distractors (cover only one of the two)
        s1id = f"conj_{i:03d}_s1"
        docs.append(
            {
                "doc_id": s1id,
                "text": f"{person} had {topic} on {fmt_date(m1, day1, y1)}.",
                "ref_time": iso(m1, day1, y1),
            }
        )
        s2id = f"conj_{i:03d}_s2"
        docs.append(
            {
                "doc_id": s2id,
                "text": f"{person} had {topic} on {fmt_date(m2, day2, y2)}.",
                "ref_time": iso(m2, day2, y2),
            }
        )
        # 2 OTHER distractors at unrelated dates
        for j, off_date in enumerate([(7, 7, 2021), (12, 22, 2025)]):
            om, od, oy = off_date
            oid = f"conj_{i:03d}_o{j}"
            docs.append(
                {
                    "doc_id": oid,
                    "text": f"{person} had {topic} on {fmt_date(om, od, oy)}.",
                    "ref_time": iso(om, od, oy),
                }
            )
        qid = f"conj_q_{i:03d}"
        q_text = f"What were {person}'s {topic} in {q_anchor}?"
        queries.append({"query_id": qid, "text": q_text, "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": [gid]})

    return docs, queries, gold_rows, "conjunctive_temporal"


# =============================================================================
# Benchmark 4: multi_te_doc — long docs with 5+ dates each; queries are simple single-date
# =============================================================================
def gen_multi_te_doc():
    """Each cluster: one GOLD doc (a long meeting note covering 5+ dates including
    the gold date), plus 4 distractor docs (also long, each covering 5+ dates that
    DON'T include the gold date but might overlap on neighbor dates).
    Query: "What did <person> do on <gold_date>?"
    Tests tag-union dilution and axis averaging.
    """
    REF_TIME = "2025-04-01T00:00:00Z"

    # (person, gold_date, surrounding_other_dates_for_gold_doc, distractor_dates_lists, gold_event_text)
    # gold doc = meeting note that mentions gold_date PLUS 4 other unrelated dates
    # distractors = meeting notes mentioning 5 dates NOT including gold_date
    CLUSTERS = [
        (
            "Sarah Park",
            (3, 12, 2024),
            [(1, 5, 2024), (5, 20, 2024), (8, 14, 2024), (11, 30, 2024)],
            [
                [
                    (2, 14, 2024),
                    (6, 1, 2024),
                    (9, 22, 2024),
                    (12, 8, 2024),
                    (4, 5, 2025),
                ],
                [
                    (7, 9, 2023),
                    (10, 30, 2023),
                    (1, 18, 2024),
                    (4, 22, 2024),
                    (7, 11, 2024),
                ],
                [
                    (11, 4, 2023),
                    (2, 28, 2024),
                    (5, 17, 2024),
                    (8, 25, 2024),
                    (11, 1, 2024),
                ],
                [
                    (8, 6, 2023),
                    (12, 22, 2023),
                    (3, 30, 2024),
                    (7, 2, 2024),
                    (10, 14, 2024),
                ],
            ],
            "the dental cleaning",
        ),
        (
            "Marcus Davis",
            (10, 8, 2023),
            [(1, 15, 2023), (4, 22, 2023), (7, 11, 2023), (12, 30, 2023)],
            [
                [
                    (2, 4, 2023),
                    (5, 28, 2023),
                    (8, 17, 2023),
                    (11, 19, 2023),
                    (3, 5, 2024),
                ],
                [
                    (6, 9, 2022),
                    (9, 22, 2022),
                    (1, 8, 2023),
                    (3, 16, 2023),
                    (6, 24, 2023),
                ],
                [
                    (11, 1, 2022),
                    (2, 13, 2023),
                    (5, 7, 2023),
                    (8, 28, 2023),
                    (11, 30, 2023),
                ],
                [
                    (7, 12, 2022),
                    (10, 25, 2022),
                    (2, 8, 2023),
                    (5, 30, 2023),
                    (9, 2, 2023),
                ],
            ],
            "the quarterly review",
        ),
        (
            "Priya Johnson",
            (5, 5, 2024),
            [(2, 12, 2024), (8, 20, 2024), (10, 30, 2024), (1, 9, 2025)],
            [
                [
                    (3, 14, 2024),
                    (6, 18, 2024),
                    (9, 1, 2024),
                    (11, 22, 2024),
                    (3, 30, 2025),
                ],
                [
                    (8, 25, 2023),
                    (11, 14, 2023),
                    (2, 1, 2024),
                    (6, 8, 2024),
                    (9, 30, 2024),
                ],
                [
                    (11, 30, 2023),
                    (4, 17, 2024),
                    (7, 6, 2024),
                    (10, 11, 2024),
                    (1, 25, 2025),
                ],
                [
                    (12, 3, 2023),
                    (3, 24, 2024),
                    (6, 1, 2024),
                    (8, 12, 2024),
                    (11, 4, 2024),
                ],
            ],
            "the team retrospective",
        ),
        (
            "Kim Patel",
            (3, 14, 2024),
            [(7, 21, 2023), (10, 5, 2023), (12, 11, 2023), (4, 28, 2024)],
            [
                [
                    (2, 19, 2024),
                    (6, 4, 2024),
                    (9, 9, 2024),
                    (11, 25, 2024),
                    (3, 8, 2025),
                ],
                [
                    (8, 16, 2023),
                    (11, 4, 2023),
                    (2, 7, 2024),
                    (5, 22, 2024),
                    (8, 30, 2024),
                ],
                [
                    (10, 18, 2023),
                    (3, 1, 2024),
                    (6, 27, 2024),
                    (9, 15, 2024),
                    (12, 14, 2024),
                ],
                [
                    (7, 7, 2023),
                    (10, 11, 2023),
                    (1, 5, 2024),
                    (4, 19, 2024),
                    (7, 30, 2024),
                ],
            ],
            "the kitchen remodel",
        ),
        (
            "Aiden Park",
            (10, 10, 2024),
            [(2, 17, 2024), (5, 24, 2024), (7, 9, 2024), (12, 30, 2024)],
            [
                [
                    (3, 12, 2024),
                    (6, 8, 2024),
                    (8, 22, 2024),
                    (11, 14, 2024),
                    (3, 5, 2025),
                ],
                [
                    (9, 20, 2023),
                    (12, 14, 2023),
                    (3, 22, 2024),
                    (7, 1, 2024),
                    (10, 24, 2024),
                ],
                [
                    (11, 4, 2023),
                    (4, 11, 2024),
                    (7, 22, 2024),
                    (9, 18, 2024),
                    (12, 2, 2024),
                ],
                [
                    (8, 14, 2023),
                    (11, 9, 2023),
                    (4, 25, 2024),
                    (8, 5, 2024),
                    (11, 18, 2024),
                ],
            ],
            "the investor pitch",
        ),
        (
            "Olivia Roberts",
            (1, 6, 2025),
            [(4, 13, 2024), (9, 20, 2024), (11, 4, 2024), (3, 18, 2025)],
            [
                [
                    (2, 22, 2025),
                    (5, 8, 2025),
                    (8, 14, 2025),
                    (11, 1, 2025),
                    (1, 30, 2026),
                ],
                [
                    (6, 29, 2024),
                    (9, 5, 2024),
                    (12, 20, 2024),
                    (4, 4, 2025),
                    (7, 18, 2025),
                ],
                [
                    (8, 15, 2024),
                    (11, 22, 2024),
                    (2, 14, 2025),
                    (5, 30, 2025),
                    (8, 25, 2025),
                ],
                [
                    (10, 7, 2024),
                    (1, 22, 2025),
                    (5, 2, 2025),
                    (8, 11, 2025),
                    (10, 28, 2025),
                ],
            ],
            "the yoga signup",
        ),
        (
            "Henry Ford",
            (9, 9, 2023),
            [(3, 16, 2023), (5, 23, 2023), (12, 30, 2023), (1, 17, 2024)],
            [
                [
                    (2, 8, 2023),
                    (5, 11, 2023),
                    (8, 20, 2023),
                    (11, 28, 2023),
                    (3, 7, 2024),
                ],
                [
                    (7, 14, 2022),
                    (10, 22, 2022),
                    (1, 30, 2023),
                    (4, 13, 2023),
                    (7, 25, 2023),
                ],
                [
                    (11, 4, 2022),
                    (2, 18, 2023),
                    (5, 5, 2023),
                    (8, 16, 2023),
                    (11, 25, 2023),
                ],
                [
                    (6, 8, 2022),
                    (9, 30, 2022),
                    (1, 18, 2023),
                    (4, 27, 2023),
                    (8, 1, 2023),
                ],
            ],
            "the performance review",
        ),
        (
            "Felix Wood",
            (5, 11, 2024),
            [(8, 18, 2023), (1, 4, 2024), (11, 25, 2023), (3, 7, 2024)],
            [
                [
                    (2, 9, 2024),
                    (6, 1, 2024),
                    (9, 22, 2024),
                    (11, 14, 2024),
                    (3, 30, 2025),
                ],
                [
                    (7, 14, 2023),
                    (10, 28, 2023),
                    (2, 18, 2024),
                    (5, 30, 2024),
                    (9, 5, 2024),
                ],
                [
                    (11, 25, 2023),
                    (3, 22, 2024),
                    (6, 11, 2024),
                    (9, 1, 2024),
                    (12, 8, 2024),
                ],
                [
                    (8, 5, 2023),
                    (11, 18, 2023),
                    (3, 1, 2024),
                    (6, 22, 2024),
                    (9, 30, 2024),
                ],
            ],
            "the client onsite",
        ),
        (
            "Quinn Reeves",
            (4, 7, 2024),
            [(8, 14, 2023), (10, 21, 2023), (12, 5, 2023), (6, 11, 2024)],
            [
                [
                    (3, 1, 2024),
                    (5, 22, 2024),
                    (8, 30, 2024),
                    (11, 14, 2024),
                    (2, 18, 2025),
                ],
                [
                    (7, 8, 2023),
                    (10, 14, 2023),
                    (1, 25, 2024),
                    (5, 5, 2024),
                    (8, 22, 2024),
                ],
                [
                    (11, 11, 2023),
                    (2, 28, 2024),
                    (5, 18, 2024),
                    (8, 25, 2024),
                    (11, 30, 2024),
                ],
                [
                    (9, 4, 2023),
                    (12, 14, 2023),
                    (3, 22, 2024),
                    (6, 28, 2024),
                    (9, 8, 2024),
                ],
            ],
            "the lease signing",
        ),
        (
            "Sara Lee",
            (5, 12, 2024),
            [(2, 5, 2024), (8, 26, 2024), (10, 19, 2023), (3, 8, 2025)],
            [
                [
                    (3, 22, 2024),
                    (6, 1, 2024),
                    (9, 5, 2024),
                    (11, 28, 2024),
                    (2, 14, 2025),
                ],
                [
                    (8, 30, 2023),
                    (11, 25, 2023),
                    (2, 19, 2024),
                    (5, 30, 2024),
                    (9, 12, 2024),
                ],
                [
                    (11, 14, 2023),
                    (3, 28, 2024),
                    (6, 22, 2024),
                    (9, 18, 2024),
                    (12, 5, 2024),
                ],
                [
                    (8, 11, 2023),
                    (12, 1, 2023),
                    (3, 30, 2024),
                    (7, 5, 2024),
                    (10, 9, 2024),
                ],
            ],
            "the puppy adoption",
        ),
        (
            "Tom Reed",
            (4, 15, 2024),
            [(11, 8, 2023), (3, 22, 2024), (7, 29, 2024), (1, 16, 2025)],
            [
                [
                    (2, 22, 2024),
                    (5, 1, 2024),
                    (8, 12, 2024),
                    (10, 30, 2024),
                    (3, 8, 2025),
                ],
                [
                    (8, 4, 2023),
                    (11, 18, 2023),
                    (2, 8, 2024),
                    (5, 25, 2024),
                    (9, 1, 2024),
                ],
                [
                    (10, 25, 2023),
                    (3, 1, 2024),
                    (6, 14, 2024),
                    (9, 28, 2024),
                    (12, 22, 2024),
                ],
                [
                    (9, 9, 2023),
                    (12, 5, 2023),
                    (4, 1, 2024),
                    (7, 14, 2024),
                    (10, 7, 2024),
                ],
            ],
            "the marathon",
        ),
        (
            "Layla Smith",
            (4, 4, 2024),
            [(11, 11, 2023), (3, 18, 2024), (7, 25, 2024), (10, 2, 2024)],
            [
                [
                    (2, 14, 2024),
                    (5, 22, 2024),
                    (8, 30, 2024),
                    (11, 18, 2024),
                    (2, 28, 2025),
                ],
                [
                    (7, 5, 2023),
                    (10, 22, 2023),
                    (1, 30, 2024),
                    (5, 14, 2024),
                    (8, 22, 2024),
                ],
                [
                    (11, 4, 2023),
                    (2, 22, 2024),
                    (5, 5, 2024),
                    (8, 18, 2024),
                    (11, 25, 2024),
                ],
                [
                    (9, 1, 2023),
                    (12, 22, 2023),
                    (3, 28, 2024),
                    (6, 30, 2024),
                    (9, 14, 2024),
                ],
            ],
            "the tax filing",
        ),
    ]

    docs = []
    queries = []
    gold_rows = []
    for i, (person, gold_date, other_dates, distractor_lists, event) in enumerate(
        CLUSTERS
    ):
        gm, gd, gy = gold_date
        # GOLD doc: long meeting-style note that mentions gold_date plus 4 unrelated dates
        gid = f"mte_{i:03d}_g"
        gold_text = (
            f"Meeting notes for {person}. "
            f"On {fmt_date(gm, gd, gy)}, {person} completed {event}. "
            f"Other items: project review on {fmt_date(*other_dates[0])}, "
            f"team sync on {fmt_date(*other_dates[1])}, "
            f"vendor call on {fmt_date(*other_dates[2])}, "
            f"and a planning session on {fmt_date(*other_dates[3])}."
        )
        docs.append(
            {
                "doc_id": gid,
                "text": gold_text,
                "ref_time": iso(gm, gd, gy),
            }
        )
        # 4 distractor docs, each with 5 dates (none of which is the gold date)
        for j, dlist in enumerate(distractor_lists):
            did = f"mte_{i:03d}_d{j}"
            ds = dlist
            text = (
                f"Meeting notes for {person}. "
                f"Items reviewed: status update on {fmt_date(*ds[0])}, "
                f"design review on {fmt_date(*ds[1])}, "
                f"vendor sync on {fmt_date(*ds[2])}, "
                f"performance check on {fmt_date(*ds[3])}, "
                f"and roadmap planning on {fmt_date(*ds[4])}."
            )
            docs.append(
                {
                    "doc_id": did,
                    "text": text,
                    # ref_time = first date in list (arbitrary anchor)
                    "ref_time": iso(*ds[0]),
                }
            )
        qid = f"mte_q_{i:03d}"
        q_text = f"What did {person} do on {fmt_date(gm, gd, gy)}?"
        queries.append({"query_id": qid, "text": q_text, "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": [gid]})

    return docs, queries, gold_rows, "multi_te_doc"


def write_bench(docs, queries, gold_rows, name):
    rng = random.Random(20260429)
    rng.shuffle(docs)
    docs_path = DATA_DIR / f"edge_{name}_docs.jsonl"
    queries_path = DATA_DIR / f"edge_{name}_queries.jsonl"
    gold_path = DATA_DIR / f"edge_{name}_gold.jsonl"
    with open(docs_path, "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    with open(queries_path, "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(gold_path, "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)
    print(f"  [{name}] {len(docs)} docs, {len(queries)} queries, {len(gold_rows)} gold")


def main():
    print("Generating 4 adversarial edge benchmarks...")
    for fn in (
        gen_era_refs,
        gen_relative_time,
        gen_conjunctive_temporal,
        gen_multi_te_doc,
    ):
        docs, queries, gold_rows, name = fn()
        write_bench(docs, queries, gold_rows, name)
    print("Done.")


if __name__ == "__main__":
    main()
