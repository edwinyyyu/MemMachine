"""Synthetic co-temporal linked data.

Generates cross-linked docs where the answer to a query requires traversing
a co-temporal edge between docs.

Structure (per cluster):
- cot_N_event:     has a date + event marker (e.g., "2012 Boulder retreat")
- cot_N_connected: references event marker WITHOUT date (e.g., "my wife and I met there")
- cot_N_context:   has date + complementary context (co-mention bridge)
- cot_N_distractor: mentions same time bucket but unrelated topic (should NOT link)

Queries:
- cot_N_qa: requires event + connected to answer (year-of-meeting style)
- cot_N_qb: requires event + context (what happened during the retreat)

Plus 5-10 topic-drift distractor docs: generic "in 2020..." unrelated things.

Author: hand-crafted for determinism and reproducibility (no LLM for synth).
Domain: generic life-event (retreats, conferences, moves), domain-neutral.
Writes: data/cotemporal_{docs,queries,gold}.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2026-04-23T12:00:00Z"


# Each cluster: (cluster_id, year, event_marker, event_text, connected_text,
#                context_text, distractor_text, qa_text, qb_text, qa_gold, qb_gold)
# Tuples of (id, text) for the docs; queries reference doc ids via gold
CLUSTERS = [
    # 1. Retreat + met partner + hiking
    {
        "cid": "cot_1",
        "year": 2012,
        "event": ("cot_1_event", "I went to the Boulder mountain retreat in 2012."),
        "connected": (
            "cot_1_connected",
            "I first met my wife at the Boulder mountain retreat.",
        ),
        "context": (
            "cot_1_context",
            "The Boulder mountain retreat that summer of 2012 changed my perspective on hiking.",
        ),
        "distractor": (
            "cot_1_distractor",
            "Boulder is a city in Colorado known for its climbing culture.",
        ),
        "queries": [
            {
                "qid": "q_cot_1_a",
                "text": "What year did I meet my wife?",
                "gold": ["cot_1_connected", "cot_1_event"],
                "note": "connected has no date; event has 2012 date + matching event marker.",
            },
            {
                "qid": "q_cot_1_b",
                "text": "What happened at the Boulder retreat?",
                "gold": ["cot_1_event", "cot_1_connected", "cot_1_context"],
                "note": "all three cotemporal docs discuss the retreat.",
            },
        ],
    },
    # 2. Conference keynote
    {
        "cid": "cot_2",
        "year": 2019,
        "event": ("cot_2_event", "The Pacific AI Summit took place in March 2019."),
        "connected": (
            "cot_2_connected",
            "I gave my first keynote talk at the Pacific AI Summit.",
        ),
        "context": (
            "cot_2_context",
            "In March 2019 I traveled to Seattle to speak about retrieval systems.",
        ),
        "distractor": (
            "cot_2_distractor",
            "Pacific is an ocean bordered by Asia and the Americas.",
        ),
        "queries": [
            {
                "qid": "q_cot_2_a",
                "text": "When did I give my first keynote?",
                "gold": ["cot_2_connected", "cot_2_event"],
                "note": "connected has no date; must link through 'Pacific AI Summit' to 2019.",
            },
            {
                "qid": "q_cot_2_b",
                "text": "What did I do in Seattle in 2019?",
                "gold": ["cot_2_context", "cot_2_connected", "cot_2_event"],
                "note": "direct via 2019+Seattle; expansion picks up keynote + summit.",
            },
        ],
    },
    # 3. Move to new house
    {
        "cid": "cot_3",
        "year": 2021,
        "event": ("cot_3_event", "We bought the Riverside house in 2021."),
        "connected": (
            "cot_3_connected",
            "The Riverside house is where I finally set up my woodworking shop.",
        ),
        "context": (
            "cot_3_context",
            "Summer of 2021 we spent every weekend fixing up the place by the river.",
        ),
        "distractor": (
            "cot_3_distractor",
            "Riverside is a broad term used for many neighborhoods.",
        ),
        "queries": [
            {
                "qid": "q_cot_3_a",
                "text": "When did I finally set up my woodworking shop?",
                "gold": ["cot_3_connected", "cot_3_event"],
                "note": "connected has no date; must link via 'Riverside house' to 2021.",
            },
            {
                "qid": "q_cot_3_b",
                "text": "What happened in 2021?",
                "gold": ["cot_3_event", "cot_3_context"],
                "note": "direct temporal via 2021.",
            },
        ],
    },
    # 4. Marathon
    {
        "cid": "cot_4",
        "year": 2015,
        "event": ("cot_4_event", "I ran my first marathon in October 2015."),
        "connected": (
            "cot_4_connected",
            "Training for that first marathon is where I met my running coach.",
        ),
        "context": (
            "cot_4_context",
            "Fall 2015 I was in the best physical shape of my life.",
        ),
        "distractor": (
            "cot_4_distractor",
            "Marathons are 26.2 miles long, a tradition from ancient Greek history.",
        ),
        "queries": [
            {
                "qid": "q_cot_4_a",
                "text": "When did I meet my running coach?",
                "gold": ["cot_4_connected", "cot_4_event"],
                "note": "connected -> 'first marathon' -> 2015 via event.",
            },
            {
                "qid": "q_cot_4_b",
                "text": "What was happening in fall 2015?",
                "gold": ["cot_4_context", "cot_4_event"],
                "note": "direct temporal.",
            },
        ],
    },
    # 5. Sabbatical
    {
        "cid": "cot_5",
        "year": 2020,
        "event": ("cot_5_event", "In June 2020 I took a sabbatical from the lab."),
        "connected": (
            "cot_5_connected",
            "During my sabbatical I learned to play the cello.",
        ),
        "context": (
            "cot_5_context",
            "That summer of 2020 was the quietest period of my career.",
        ),
        "distractor": (
            "cot_5_distractor",
            "June is the sixth month of the Gregorian calendar.",
        ),
        "queries": [
            {
                "qid": "q_cot_5_a",
                "text": "When did I learn cello?",
                "gold": ["cot_5_connected", "cot_5_event"],
                "note": "connected says 'during sabbatical'; event anchors the year.",
            },
            {
                "qid": "q_cot_5_b",
                "text": "What happened in mid-2020?",
                "gold": ["cot_5_event", "cot_5_context"],
                "note": "direct temporal via June/summer 2020.",
            },
        ],
    },
    # 6. Daughter born
    {
        "cid": "cot_6",
        "year": 2017,
        "event": ("cot_6_event", "Our daughter was born in March 2017."),
        "connected": (
            "cot_6_connected",
            "The year our daughter came along I quit my consulting job.",
        ),
        "context": (
            "cot_6_context",
            "Spring 2017 was a whirlwind of sleepless nights and pediatrician visits.",
        ),
        "distractor": ("cot_6_distractor", "Daughter is a kinship term in English."),
        "queries": [
            {
                "qid": "q_cot_6_a",
                "text": "When did I quit consulting?",
                "gold": ["cot_6_connected", "cot_6_event"],
                "note": "connected references event anchor ('daughter came along').",
            },
            {
                "qid": "q_cot_6_b",
                "text": "What was spring 2017 like?",
                "gold": ["cot_6_context", "cot_6_event"],
                "note": "direct via spring 2017.",
            },
        ],
    },
    # 7. Car purchase
    {
        "cid": "cot_7",
        "year": 2023,
        "event": ("cot_7_event", "I bought the red Subaru in 2023."),
        "connected": (
            "cot_7_connected",
            "The red Subaru took us on our first road trip through Utah.",
        ),
        "context": (
            "cot_7_context",
            "Summer 2023 we explored national parks every other weekend.",
        ),
        "distractor": (
            "cot_7_distractor",
            "Subarus are Japanese cars with a distinctive boxer engine.",
        ),
        "queries": [
            {
                "qid": "q_cot_7_a",
                "text": "When did we take our first Utah road trip?",
                "gold": ["cot_7_connected", "cot_7_event"],
                "note": "connected anchors through 'red Subaru' to 2023.",
            },
            {
                "qid": "q_cot_7_b",
                "text": "What was summer 2023 like?",
                "gold": ["cot_7_context", "cot_7_event"],
                "note": "direct temporal.",
            },
        ],
    },
    # 8. Graduate school
    {
        "cid": "cot_8",
        "year": 2008,
        "event": ("cot_8_event", "I enrolled at Madison State in September 2008."),
        "connected": (
            "cot_8_connected",
            "Grad school at Madison State is where I met my thesis advisor.",
        ),
        "context": (
            "cot_8_context",
            "Fall 2008 the financial crisis made scholarships scarce.",
        ),
        "distractor": ("cot_8_distractor", "Madison is the capital of Wisconsin."),
        "queries": [
            {
                "qid": "q_cot_8_a",
                "text": "When did I meet my thesis advisor?",
                "gold": ["cot_8_connected", "cot_8_event"],
                "note": "connected anchors through 'Madison State' to Sep 2008.",
            },
            {
                "qid": "q_cot_8_b",
                "text": "What was fall 2008 like?",
                "gold": ["cot_8_context", "cot_8_event"],
                "note": "direct temporal.",
            },
        ],
    },
    # 9. Book club
    {
        "cid": "cot_9",
        "year": 2014,
        "event": ("cot_9_event", "We started the Tuesday book club in 2014."),
        "connected": (
            "cot_9_connected",
            "At the Tuesday book club I became friends with Priya.",
        ),
        "context": (
            "cot_9_context",
            "The winter of 2014 we read classics every two weeks.",
        ),
        "distractor": ("cot_9_distractor", "Tuesday is named after the Norse god Tyr."),
        "queries": [
            {
                "qid": "q_cot_9_a",
                "text": "When did I become friends with Priya?",
                "gold": ["cot_9_connected", "cot_9_event"],
                "note": "connected anchors through book club to 2014.",
            },
            {
                "qid": "q_cot_9_b",
                "text": "What was the winter of 2014 like?",
                "gold": ["cot_9_context", "cot_9_event"],
                "note": "direct temporal.",
            },
        ],
    },
    # 10. Apartment in Brooklyn
    {
        "cid": "cot_10",
        "year": 2016,
        "event": (
            "cot_10_event",
            "We signed the Brooklyn apartment lease in May 2016.",
        ),
        "connected": (
            "cot_10_connected",
            "The Brooklyn apartment is where I wrote my first novel.",
        ),
        "context": (
            "cot_10_context",
            "May 2016 was hectic with movers, boxes, and paperwork.",
        ),
        "distractor": (
            "cot_10_distractor",
            "Brooklyn is the most populous borough in New York City.",
        ),
        "queries": [
            {
                "qid": "q_cot_10_a",
                "text": "When did I write my first novel?",
                "gold": ["cot_10_connected", "cot_10_event"],
                "note": "connected anchors through Brooklyn apartment to May 2016.",
            },
            {
                "qid": "q_cot_10_b",
                "text": "What happened in May 2016?",
                "gold": ["cot_10_context", "cot_10_event"],
                "note": "direct temporal.",
            },
        ],
    },
]


# Topic-drift distractors: share temporal ground (common years) but unrelated
TOPIC_DRIFT_DOCS = [
    ("drift_2020_a", "In 2020 the global supply chain was disrupted."),
    ("drift_2020_b", "During 2020 remote work became the default for many industries."),
    ("drift_2019_a", "2019 was the warmest year on record at the time."),
    ("drift_2017_a", "In 2017 cryptocurrency prices spiked dramatically."),
    ("drift_2015_a", "The Paris climate accord was signed in 2015."),
    ("drift_2012_a", "In 2012 the Mayan calendar ended, sparking apocalyptic memes."),
    ("drift_2023_a", "2023 saw major advancements in language models."),
    ("drift_2008_a", "The 2008 financial crisis had global ripples for years."),
]


def main() -> None:
    docs = []
    queries = []
    gold = []

    for cluster in CLUSTERS:
        for role in ("event", "connected", "context", "distractor"):
            did, text = cluster[role]
            docs.append(
                {
                    "doc_id": did,
                    "category": f"COT_{role.upper()}",
                    "cluster": cluster["cid"],
                    "text": text,
                    "ref_time": REF_TIME,
                }
            )
        for q in cluster["queries"]:
            queries.append(
                {
                    "query_id": q["qid"],
                    "category": "COT",
                    "cluster": cluster["cid"],
                    "text": q["text"],
                    "ref_time": REF_TIME,
                }
            )
            gold.append(
                {
                    "query_id": q["qid"],
                    "category": "COT",
                    "cluster": cluster["cid"],
                    "relevant_doc_ids": q["gold"],
                    "expected_behavior": q["note"],
                }
            )

    for did, text in TOPIC_DRIFT_DOCS:
        docs.append(
            {
                "doc_id": did,
                "category": "COT_DRIFT",
                "cluster": None,
                "text": text,
                "ref_time": REF_TIME,
            }
        )

    docs_p = DATA_DIR / "cotemporal_docs.jsonl"
    queries_p = DATA_DIR / "cotemporal_queries.jsonl"
    gold_p = DATA_DIR / "cotemporal_gold.jsonl"
    docs_p.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    queries_p.write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    gold_p.write_text("\n".join(json.dumps(g) for g in gold) + "\n")

    n_clusters = len(CLUSTERS)
    print(
        f"Wrote {len(docs)} docs ({n_clusters * 4} cluster + {len(TOPIC_DRIFT_DOCS)} drift)"
    )
    print(f"Wrote {len(queries)} queries ({n_clusters * 2})")
    print(f"Wrote {len(gold)} gold entries")


if __name__ == "__main__":
    main()
