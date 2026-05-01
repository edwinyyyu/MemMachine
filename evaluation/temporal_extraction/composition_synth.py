"""Composition benchmark: queries that combine MULTIPLE temporal cues.

Tests whether the cue-gated modules (recency, open_ended, negation,
causal) compose correctly when more than one fires.

Five composition types (5 queries each):
  A. recency × absolute: "latest from Q4 2023"
  B. negation × absolute: "in 2024 not in summer"
  C. causal × recency:   "most recent after the migration"
  D. causal × absolute:  "in Q3 2023 after the launch"
  E. open_ended × negation: "after 2020 but not in 2023"

Each cluster has:
  - 1 GOLD that satisfies ALL constraints
  - 2-3 SINGLE-CUE distractors (each satisfies exactly one of the
    constraints but not all)
  - 1-2 noise (different topic, irrelevant date)

Output: data/composition_{docs,queries,gold}.jsonl
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


# Each cluster has shape:
#   query, gold (template, m, d, y), distractors [(template, m, d, y), ...],
#   noise [(template, m, d, y), ...]
# Distractors satisfy one constraint but not another.

CLUSTERS_A = [  # recency × absolute
    {
        "query": "What's my latest project Alpha update from Q4 2023?",
        "gold": (
            "Posted a project Alpha status update covering pipeline finalization, on {date}.",
            12,
            14,
            2023,
        ),
        "distractors": [
            # Recent but outside Q4 2023 (recency-only)
            (
                "Posted a project Alpha status update on the new dashboard, on {date}.",
                5,
                8,
                2025,
            ),
            (
                "Posted a project Alpha status update on Q1 milestones, on {date}.",
                2,
                11,
                2025,
            ),
            # Old but inside Q4 2023 (absolute-only — older than gold)
            (
                "Posted a project Alpha status update on the kickoff plan, on {date}.",
                10,
                4,
                2023,
            ),
        ],
        "noise": [
            ("Booked a flight to Boston for the team summit, on {date}.", 6, 14, 2024),
        ],
    },
    {
        "query": "Most recent meeting in March 2024",
        "gold": (
            "Held a stakeholder meeting on roadmap commitments, on {date}.",
            3,
            28,
            2024,
        ),
        "distractors": [
            ("Held a stakeholder meeting on Q3 retro topics, on {date}.", 5, 18, 2025),
            ("Held a stakeholder meeting on the budget freeze, on {date}.", 4, 2, 2025),
            (
                "Held a stakeholder meeting on the kickoff agenda, on {date}.",
                3,
                4,
                2024,
            ),
        ],
        "noise": [
            ("Picked up a package from the post office, on {date}.", 7, 22, 2023),
        ],
    },
    {
        "query": "My latest budget review in Q2 2024",
        "gold": ("Conducted a budget review with finance, on {date}.", 6, 26, 2024),
        "distractors": [
            ("Conducted a budget review with finance, on {date}.", 3, 14, 2025),
            ("Conducted a budget review with finance, on {date}.", 1, 9, 2025),
            ("Conducted a budget review with finance, on {date}.", 4, 8, 2024),
        ],
        "noise": [
            ("Visited the Botanical Gardens with family, on {date}.", 11, 5, 2023),
        ],
    },
    {
        "query": "The most recent design review in 2023",
        "gold": ("Ran a design review for the new dashboard, on {date}.", 12, 11, 2023),
        "distractors": [
            ("Ran a design review for the new dashboard, on {date}.", 4, 22, 2025),
            ("Ran a design review for the new dashboard, on {date}.", 8, 1, 2024),
            ("Ran a design review for the new dashboard, on {date}.", 3, 6, 2023),
        ],
        "noise": [
            ("Caught up on email and inbox triage, on {date}.", 9, 9, 2022),
        ],
    },
    {
        "query": "Latest workout I did in January 2025",
        "gold": ("Did a strength training workout at the gym, on {date}.", 1, 28, 2025),
        "distractors": [
            ("Did a strength training workout at the gym, on {date}.", 5, 10, 2025),
            ("Did a strength training workout at the gym, on {date}.", 4, 14, 2025),
            ("Did a strength training workout at the gym, on {date}.", 1, 4, 2025),
        ],
        "noise": [
            ("Renewed the office lease paperwork, on {date}.", 11, 18, 2023),
        ],
    },
]


CLUSTERS_B = [  # negation × absolute
    {
        "query": "What client meetings did I have in 2024 not in summer?",
        "gold": ("Held a client meeting with the Acme team, on {date}.", 3, 12, 2024),
        "distractors": [
            # In 2024 but in summer (matches absolute window but excluded)
            ("Held a client meeting with the Acme team, on {date}.", 7, 17, 2024),
            ("Held a client meeting with the Acme team, on {date}.", 8, 8, 2024),
            # Outside 2024 (matches the negation but not absolute)
            ("Held a client meeting with the Acme team, on {date}.", 3, 21, 2023),
        ],
        "noise": [
            (
                "Caught a film at the local theater with friends, on {date}.",
                1,
                14,
                2022,
            ),
        ],
    },
    {
        "query": "Meetings outside Q3 2023",
        "gold": ("Met with the architecture review board, on {date}.", 5, 20, 2024),
        "distractors": [
            ("Met with the architecture review board, on {date}.", 7, 14, 2023),
            ("Met with the architecture review board, on {date}.", 8, 28, 2023),
            ("Met with the architecture review board, on {date}.", 9, 4, 2023),
        ],
        "noise": [
            ("Picked up groceries for the week, on {date}.", 11, 2, 2022),
        ],
    },
    {
        "query": "What classes did I take in 2024 excluding the spring semester?",
        "gold": (
            "Attended a continuing-education class on data ethics, on {date}.",
            10,
            8,
            2024,
        ),
        "distractors": [
            (
                "Attended a continuing-education class on data ethics, on {date}.",
                2,
                14,
                2024,
            ),
            (
                "Attended a continuing-education class on data ethics, on {date}.",
                4,
                9,
                2024,
            ),
            (
                "Attended a continuing-education class on data ethics, on {date}.",
                11,
                3,
                2022,
            ),
        ],
        "noise": [
            ("Hiked the river trail with the dogs, on {date}.", 6, 18, 2025),
        ],
    },
    {
        "query": "What did I do in 2025 not in January?",
        "gold": ("Filed the annual tax extension paperwork, on {date}.", 4, 14, 2025),
        "distractors": [
            ("Filed the annual tax extension paperwork, on {date}.", 1, 6, 2025),
            ("Filed the annual tax extension paperwork, on {date}.", 1, 22, 2025),
            ("Filed the annual tax extension paperwork, on {date}.", 4, 11, 2024),
        ],
        "noise": [
            ("Repotted the houseplants in the living room, on {date}.", 3, 1, 2023),
        ],
    },
    {
        "query": "Trips I took in 2023 outside of December",
        "gold": (
            "Took a road trip to a national park with friends, on {date}.",
            7,
            18,
            2023,
        ),
        "distractors": [
            (
                "Took a road trip to a national park with friends, on {date}.",
                12,
                8,
                2023,
            ),
            (
                "Took a road trip to a national park with friends, on {date}.",
                12,
                22,
                2023,
            ),
            (
                "Took a road trip to a national park with friends, on {date}.",
                7,
                4,
                2024,
            ),
        ],
        "noise": [
            ("Bought a new keyboard for the home setup, on {date}.", 10, 15, 2025),
        ],
    },
]


CLUSTERS_C = [  # causal × recency
    {
        "query": "My most recent update after the migration",
        "gold": (
            "Posted an update saying latency stabilized and SLOs were back to green, on {date}.",
            5,
            30,
            2025,
        ),
        "distractors": [
            # After the migration but old
            (
                "Posted an update saying the new pipeline was holding through the spike test, on {date}.",
                3,
                5,
                2024,
            ),
            (
                "Posted an update saying error budgets had partially recovered, on {date}.",
                6,
                18,
                2024,
            ),
            # Recent but BEFORE the migration (wrong direction)
            (
                "Posted an update saying the cutover plan still needed two more checkpoints, on {date}.",
                1,
                20,
                2024,
            ),
        ],
        "anchor": ("The data migration was completed on {date}.", 2, 15, 2024),
        "noise": [
            ("Watched a documentary on coastal ecosystems, on {date}.", 8, 11, 2023),
        ],
    },
    {
        "query": "The latest thing I did since the launch",
        "gold": (
            "Audited support tickets for trends from the last sprint, on {date}.",
            4,
            12,
            2025,
        ),
        "distractors": [
            (
                "Audited support tickets for trends from the rollout, on {date}.",
                5,
                22,
                2024,
            ),
            (
                "Audited support tickets for trends from the post-launch period, on {date}.",
                9,
                8,
                2024,
            ),
            (
                "Audited support tickets for trends from the early beta, on {date}.",
                2,
                6,
                2024,
            ),
        ],
        "anchor": ("Product launch occurred on {date}.", 5, 1, 2024),
        "noise": [
            ("Reorganized the kitchen cabinets, on {date}.", 11, 22, 2022),
        ],
    },
    {
        "query": "Most recent thing Maya reported after the last review",
        "gold": (
            "Maya reported the new tracing dashboard now covers all production services, on {date}.",
            5,
            28,
            2025,
        ),
        "distractors": [
            (
                "Maya reported two regressions caught by the dashboard, on {date}.",
                4,
                12,
                2024,
            ),
            (
                "Maya reported the alerting noise had dropped meaningfully, on {date}.",
                9,
                15,
                2024,
            ),
            (
                "Maya reported the dashboard prototype was still missing service breakdowns, on {date}.",
                1,
                25,
                2024,
            ),
        ],
        "anchor": (
            "The most recent quarterly review wrapped up on {date}.",
            2,
            28,
            2024,
        ),
        "noise": [
            ("Joined a friend's birthday dinner downtown, on {date}.", 10, 4, 2023),
        ],
    },
    {
        "query": "Most recent change since the redesign shipped",
        "gold": (
            "Shipped a tooltip refinement across the analytics views, on {date}.",
            6,
            5,
            2025,
        ),
        "distractors": [
            (
                "Shipped a tooltip refinement on the dashboard cards, on {date}.",
                11,
                20,
                2024,
            ),
            (
                "Shipped a tooltip refinement on the legacy reports, on {date}.",
                8,
                12,
                2024,
            ),
            (
                "Shipped a tooltip refinement during the QA pass, on {date}.",
                5,
                30,
                2024,
            ),
        ],
        "anchor": ("The product redesign shipped to all users on {date}.", 6, 18, 2024),
        "noise": [
            ("Tried a new pasta recipe for dinner, on {date}.", 1, 14, 2023),
        ],
    },
    {
        "query": "Latest meeting I had after the offsite",
        "gold": (
            "Met with the leadership pod on Q3 priorities, on {date}.",
            5,
            22,
            2025,
        ),
        "distractors": [
            ("Met with the leadership pod on Q4 retro, on {date}.", 12, 3, 2024),
            (
                "Met with the leadership pod on the post-offsite plan, on {date}.",
                10,
                18,
                2024,
            ),
            ("Met with the leadership pod on offsite prep, on {date}.", 9, 14, 2024),
        ],
        "anchor": ("The company offsite was held on {date}.", 9, 28, 2024),
        "noise": [
            ("Volunteered at the local pet shelter, on {date}.", 8, 6, 2023),
        ],
    },
]


CLUSTERS_D = [  # causal × absolute
    {
        "query": "What did I do in Q3 2023 after the launch?",
        "gold": ("Drafted the post-launch performance report, on {date}.", 8, 22, 2023),
        "distractors": [
            # In Q3 2023 but before launch (wrong direction)
            ("Drafted the pre-launch readiness checklist, on {date}.", 7, 5, 2023),
            # After launch but outside Q3 2023
            (
                "Drafted the year-end summary that included rollout details, on {date}.",
                12,
                18,
                2023,
            ),
            ("Drafted a follow-up retro on the launch, on {date}.", 2, 14, 2024),
        ],
        "anchor": ("The product launch occurred on {date}.", 7, 21, 2023),
        "noise": [
            ("Bought a new bookshelf for the study, on {date}.", 4, 8, 2024),
        ],
    },
    {
        "query": "What happened in 2024 after the migration?",
        "gold": (
            "Closed out the legacy database decommissioning paperwork, on {date}.",
            6,
            14,
            2024,
        ),
        "distractors": [
            (
                "Closed out the legacy database decommissioning paperwork, on {date}.",
                1,
                20,
                2024,
            ),
            (
                "Closed out the legacy database decommissioning paperwork, on {date}.",
                11,
                30,
                2025,
            ),
            (
                "Closed out the legacy database decommissioning paperwork, on {date}.",
                8,
                9,
                2023,
            ),
        ],
        "anchor": ("The data migration was completed on {date}.", 2, 15, 2024),
        "noise": [
            ("Replaced the kitchen faucet, on {date}.", 9, 5, 2022),
        ],
    },
    {
        "query": "What I worked on in March 2024 after the freeze",
        "gold": (
            "Worked on the experimentation framework new tooling, on {date}.",
            3,
            21,
            2024,
        ),
        "distractors": [
            # In March 2024 but BEFORE freeze
            (
                "Worked on the experimentation framework spec doc, on {date}.",
                3,
                4,
                2024,
            ),
            # After freeze but outside March 2024
            (
                "Worked on the experimentation framework rollout pilot, on {date}.",
                5,
                17,
                2024,
            ),
            (
                "Worked on the experimentation framework intro guide, on {date}.",
                7,
                8,
                2024,
            ),
        ],
        "anchor": ("The code freeze started on {date}.", 3, 12, 2024),
        "noise": [
            ("Donated old clothes to the community center, on {date}.", 12, 2, 2022),
        ],
    },
    {
        "query": "Things I did in Q4 2024 before year-end review",
        "gold": (
            "Wrote up the Q4 retrospective draft for the team, on {date}.",
            11,
            15,
            2024,
        ),
        "distractors": [
            # In Q4 2024 but AFTER year-end review (wrong direction)
            (
                "Wrote up the Q4 retrospective draft for the team, on {date}.",
                12,
                28,
                2024,
            ),
            # Before year-end review but outside Q4 2024
            (
                "Wrote up the H1 retrospective draft for the team, on {date}.",
                7,
                9,
                2024,
            ),
            (
                "Wrote up the planning retrospective draft for the team, on {date}.",
                3,
                14,
                2024,
            ),
        ],
        "anchor": ("The year-end review was held on {date}.", 12, 18, 2024),
        "noise": [
            ("Repainted the front porch railing, on {date}.", 5, 4, 2023),
        ],
    },
    {
        "query": "What I did in May 2024 since the kickoff",
        "gold": (
            "Synced with the engineering pod on milestone alignment, on {date}.",
            5,
            24,
            2024,
        ),
        "distractors": [
            # In May 2024 but BEFORE kickoff
            (
                "Synced with the engineering pod on pre-kickoff prep, on {date}.",
                5,
                2,
                2024,
            ),
            # After kickoff but outside May 2024
            (
                "Synced with the engineering pod on early sprint progress, on {date}.",
                6,
                18,
                2024,
            ),
            (
                "Synced with the engineering pod on Q3 deliverables, on {date}.",
                8,
                8,
                2024,
            ),
        ],
        "anchor": ("The project kickoff happened on {date}.", 5, 13, 2024),
        "noise": [
            ("Tried a new pottery class downtown, on {date}.", 11, 18, 2022),
        ],
    },
]


CLUSTERS_E = [  # open_ended × negation
    {
        "query": "What did I do after 2020 but not in 2023?",
        "gold": (
            "Renovated the back patio with new tiles and seating, on {date}.",
            6,
            18,
            2024,
        ),
        "distractors": [
            # After 2020 but in 2023 (excluded)
            (
                "Renovated the back patio with new tiles and seating, on {date}.",
                4,
                12,
                2023,
            ),
            (
                "Renovated the back patio with new tiles and seating, on {date}.",
                9,
                8,
                2023,
            ),
            # In 2023 — same time
            (
                "Renovated the back patio with new tiles and seating, on {date}.",
                11,
                22,
                2023,
            ),
            # Before 2020 (matches negation but not open-ended bound)
            (
                "Renovated the back patio with new tiles and seating, on {date}.",
                5,
                4,
                2019,
            ),
        ],
        "noise": [
            ("Watched the regional soccer final on TV, on {date}.", 7, 14, 2017),
        ],
    },
    {
        "query": "Books I read since 2021 excluding 2024",
        "gold": ("Finished a novel from my reading list, on {date}.", 3, 4, 2025),
        "distractors": [
            ("Finished a novel from my reading list, on {date}.", 1, 14, 2024),
            ("Finished a novel from my reading list, on {date}.", 8, 9, 2024),
            ("Finished a novel from my reading list, on {date}.", 11, 17, 2024),
            ("Finished a novel from my reading list, on {date}.", 5, 22, 2020),
        ],
        "noise": [
            ("Cooked a four-course dinner for the in-laws, on {date}.", 4, 1, 2018),
        ],
    },
    {
        "query": "Things I did before 2024 not in summer 2022",
        "gold": ("Hosted a backyard barbecue with neighbors, on {date}.", 5, 14, 2023),
        "distractors": [
            # Before 2024 but in summer 2022 (excluded)
            ("Hosted a backyard barbecue with neighbors, on {date}.", 6, 22, 2022),
            ("Hosted a backyard barbecue with neighbors, on {date}.", 7, 30, 2022),
            ("Hosted a backyard barbecue with neighbors, on {date}.", 8, 14, 2022),
            # After 2024
            ("Hosted a backyard barbecue with neighbors, on {date}.", 6, 4, 2024),
        ],
        "noise": [
            ("Tested a new bread recipe over the weekend, on {date}.", 1, 8, 2026),
        ],
    },
    {
        "query": "What I did since 2022 outside of Q1 2023",
        "gold": (
            "Volunteered at the river cleanup with friends, on {date}.",
            9,
            28,
            2024,
        ),
        "distractors": [
            ("Volunteered at the river cleanup with friends, on {date}.", 1, 14, 2023),
            ("Volunteered at the river cleanup with friends, on {date}.", 2, 22, 2023),
            ("Volunteered at the river cleanup with friends, on {date}.", 3, 11, 2023),
            ("Volunteered at the river cleanup with friends, on {date}.", 6, 8, 2021),
        ],
        "noise": [
            (
                "Watched a documentary on Antarctic expeditions, on {date}.",
                11,
                18,
                2017,
            ),
        ],
    },
    {
        "query": "Therapy sessions I had after 2022 not in February 2025",
        "gold": ("Had a therapy session with my counselor, on {date}.", 5, 6, 2025),
        "distractors": [
            ("Had a therapy session with my counselor, on {date}.", 2, 3, 2025),
            ("Had a therapy session with my counselor, on {date}.", 2, 17, 2025),
            ("Had a therapy session with my counselor, on {date}.", 2, 24, 2025),
            ("Had a therapy session with my counselor, on {date}.", 8, 12, 2021),
        ],
        "noise": [
            ("Repaired a fence post in the back yard, on {date}.", 4, 14, 2018),
        ],
    },
]


def main() -> None:
    rng = random.Random(20260429)
    all_clusters = (
        [("A", c) for c in CLUSTERS_A]
        + [("B", c) for c in CLUSTERS_B]
        + [("C", c) for c in CLUSTERS_C]
        + [("D", c) for c in CLUSTERS_D]
        + [("E", c) for c in CLUSTERS_E]
    )

    docs = []
    queries = []
    gold_rows = []

    for i, (typ, c) in enumerate(all_clusters):
        # Gold doc
        gtmpl, gm, gd, gy = c["gold"]
        gid = f"comp_{typ}_{i:03d}_g0"
        docs.append(
            {
                "doc_id": gid,
                "text": gtmpl.format(date=fmt_date(gm, gd, gy)),
                "ref_time": iso(gm, gd, gy),
            }
        )
        # Distractors
        for j, (dtmpl, m, d, y) in enumerate(c["distractors"]):
            did = f"comp_{typ}_{i:03d}_d{j}"
            docs.append(
                {
                    "doc_id": did,
                    "text": dtmpl.format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )
        # Anchor (for causal types only)
        if "anchor" in c:
            atmpl, am, ad, ay = c["anchor"]
            aid = f"comp_{typ}_{i:03d}_a"
            docs.append(
                {
                    "doc_id": aid,
                    "text": atmpl.format(date=fmt_date(am, ad, ay)),
                    "ref_time": iso(am, ad, ay),
                }
            )
        # Noise
        for j, (ntmpl, m, d, y) in enumerate(c["noise"]):
            nid = f"comp_{typ}_{i:03d}_n{j}"
            docs.append(
                {
                    "doc_id": nid,
                    "text": ntmpl.format(date=fmt_date(m, d, y)),
                    "ref_time": iso(m, d, y),
                }
            )

        qid = f"comp_q_{typ}_{i:03d}"
        queries.append(
            {
                "query_id": qid,
                "text": c["query"],
                "ref_time": REF_TIME,
                "comp_type": typ,
            }
        )
        gold_rows.append({"query_id": qid, "relevant_doc_ids": [gid]})

    rng.shuffle(docs)
    with open(DATA_DIR / "composition_docs.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with open(DATA_DIR / "composition_queries.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(DATA_DIR / "composition_gold.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)

    by_type: dict[str, int] = {}
    for q in queries:
        by_type[q["comp_type"]] = by_type.get(q["comp_type"], 0) + 1
    print(
        f"composition: {len(docs)} docs, {len(queries)} queries, {len(gold_rows)} gold"
    )
    print(f"  by_type: {by_type}")


if __name__ == "__main__":
    main()
