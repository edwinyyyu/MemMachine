"""Causal-relative benchmark: queries reference time relative to ANOTHER EVENT.

Each cluster has 6 docs:
  - 1 GOLD: doc IN the right relative window (after/before/since the anchor)
  - 1 ANCHOR: doc describing the referenced anchor event itself
  - 1 WRONG-DIRECTION: same topic, wrong side of anchor
  - 1 SAME-TIME-WRONG-TOPIC distractor (close in time to gold, off-topic)
  - 2 noise docs (different topic, different times)

Output: data/causal_relative_{docs,queries,gold}.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2025-06-01T00:00:00Z"


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


# Each cluster:
#   anchor_event: name of the referenced event (used in query)
#   topic_subj: person + topic (shared between gold and wrong-direction)
#   anchor_date, gold_date, wrong_dir_date (same topic, wrong side)
#   gold_text: what gold says (the relative-window event)
#   wrong_dir_text: same topic, wrong side
#   anchor_text: describes the anchor event
#   wrong_topic_text: same time as gold, different topic
#   wrong_topic_date
#   noise_text_1, noise_date_1
#   noise_text_2, noise_date_2
#   relation: "after" / "before" / "since"
#   query_text
CLUSTERS = [
    {
        "person": "Sarah",
        "anchor_event": "the migration",
        "anchor_date": (2, 15, 2024),
        "anchor_text": "The data migration was completed on February 15, 2024.",
        "gold_date": (3, 5, 2024),
        "gold_text": "Sarah said the latency dropped by half once the new pipeline stabilized, on March 5, 2024.",
        "wrong_dir_date": (1, 20, 2024),
        "wrong_dir_text": "Sarah said the new pipeline still needed two more checkpoints before cutover, on January 20, 2024.",
        "wrong_topic_date": (3, 6, 2024),
        "wrong_topic_text": "Marcus filed the Q1 expense report on March 6, 2024.",
        "noise": [
            ((9, 8, 2023), "Hannah attended the design summit on September 8, 2023."),
            (
                (11, 19, 2024),
                "Eric gave a keynote at the architecture conference on November 19, 2024.",
            ),
        ],
        "relation": "after",
        "query": "What did Sarah say after the migration was complete?",
    },
    {
        "person": "team",
        "anchor_event": "the launch",
        "anchor_date": (5, 1, 2024),
        "anchor_text": "Product launch occurred on May 1, 2024.",
        "gold_date": (4, 18, 2024),
        "gold_text": "The team froze the marketing copy and ran the final dry-run on April 18, 2024.",
        "wrong_dir_date": (5, 22, 2024),
        "wrong_dir_text": "The team triaged customer-reported issues from the rollout on May 22, 2024.",
        "wrong_topic_date": (4, 17, 2024),
        "wrong_topic_text": "HR sent the benefits-enrollment reminder on April 17, 2024.",
        "noise": [
            (
                (1, 4, 2023),
                "Aiden reviewed Q4 metrics with the board on January 4, 2023.",
            ),
            (
                (10, 30, 2024),
                "Vera completed her annual self-assessment on October 30, 2024.",
            ),
        ],
        "relation": "before",
        "query": "What happened before the launch?",
    },
    {
        "person": "Maya",
        "anchor_event": "the last review",
        "anchor_date": (2, 28, 2024),
        "anchor_text": "The most recent quarterly review wrapped up on February 28, 2024.",
        "gold_date": (4, 12, 2024),
        "gold_text": "Maya reported the new tracing dashboard caught two regressions in production on April 12, 2024.",
        "wrong_dir_date": (1, 25, 2024),
        "wrong_dir_text": "Maya reported the dashboard prototype was still missing service breakdowns on January 25, 2024.",
        "wrong_topic_date": (4, 11, 2024),
        "wrong_topic_text": "Diego booked a flight to Boston on April 11, 2024.",
        "noise": [
            (
                (7, 3, 2023),
                "Yuki finished her open-water certification on July 3, 2023.",
            ),
            (
                (11, 14, 2023),
                "Carla received the employee-of-the-month award on November 14, 2023.",
            ),
        ],
        "relation": "since",
        "query": "What did Maya report since the last review?",
    },
    {
        "person": "Priya",
        "anchor_event": "the offsite",
        "anchor_date": (6, 10, 2024),
        "anchor_text": "The team offsite ran from June 10 through June 12, 2024.",
        "gold_date": (6, 21, 2024),
        "gold_text": "Priya circulated a written retro pulling out the three biggest organizational gaps on June 21, 2024.",
        "wrong_dir_date": (5, 28, 2024),
        "wrong_dir_text": "Priya drafted the offsite agenda and a list of pre-reads on May 28, 2024.",
        "wrong_topic_date": (6, 22, 2024),
        "wrong_topic_text": "Tom completed his Boston Marathon training plan on June 22, 2024.",
        "noise": [
            ((2, 4, 2023), "Layla filed her tax return early on February 4, 2023."),
            (
                (12, 18, 2024),
                "Felix went on the Tokyo client onsite on December 18, 2024.",
            ),
        ],
        "relation": "after",
        "query": "What did Priya circulate after the offsite?",
    },
    {
        "person": "Marcus",
        "anchor_event": "the merger",
        "anchor_date": (9, 15, 2023),
        "anchor_text": "The two companies legally merged on September 15, 2023.",
        "gold_date": (8, 22, 2023),
        "gold_text": "Marcus mapped out the redundant teams and a 90-day integration plan on August 22, 2023.",
        "wrong_dir_date": (10, 4, 2023),
        "wrong_dir_text": "Marcus walked the combined leadership team through the integration scoreboard on October 4, 2023.",
        "wrong_topic_date": (8, 21, 2023),
        "wrong_topic_text": "Olivia signed up for the fall yoga series on August 21, 2023.",
        "noise": [
            (
                (3, 16, 2024),
                "Henry completed his performance review on March 16, 2024.",
            ),
            ((1, 9, 2025), "Mira hosted the data-privacy training on January 9, 2025."),
        ],
        "relation": "before",
        "query": "What planning did Marcus do before the merger?",
    },
    {
        "person": "Aiden",
        "anchor_event": "the funding round",
        "anchor_date": (3, 1, 2024),
        "anchor_text": "The Series B funding round closed on March 1, 2024.",
        "gold_date": (3, 18, 2024),
        "gold_text": "Aiden hired the new VP of Engineering and revised the 18-month roadmap on March 18, 2024.",
        "wrong_dir_date": (2, 12, 2024),
        "wrong_dir_text": "Aiden walked investors through the projection model on February 12, 2024.",
        "wrong_topic_date": (3, 19, 2024),
        "wrong_topic_text": "Quinn renewed the office lease on March 19, 2024.",
        "noise": [
            ((6, 8, 2023), "Kim completed her kitchen remodel on June 8, 2023."),
            ((11, 4, 2024), "Sara adopted a second puppy on November 4, 2024."),
        ],
        "relation": "after",
        "query": "What did Aiden do after the funding round closed?",
    },
    {
        "person": "Eric",
        "anchor_event": "the keynote",
        "anchor_date": (11, 9, 2023),
        "anchor_text": "Eric delivered the industry keynote on November 9, 2023.",
        "gold_date": (10, 25, 2023),
        "gold_text": "Eric rehearsed the keynote with the comms team and tightened the demo script on October 25, 2023.",
        "wrong_dir_date": (11, 28, 2023),
        "wrong_dir_text": "Eric followed up with twelve press contacts who attended the keynote on November 28, 2023.",
        "wrong_topic_date": (10, 26, 2023),
        "wrong_topic_text": "Hannah booked her flight to the Lisbon design summit on October 26, 2023.",
        "noise": [
            ((4, 13, 2024), "Tom finished the marathon on April 13, 2024."),
            ((7, 2, 2024), "Vera moved to her new apartment on July 2, 2024."),
        ],
        "relation": "before",
        "query": "What did Eric do before the keynote?",
    },
    {
        "person": "Layla",
        "anchor_event": "the audit",
        "anchor_date": (4, 15, 2024),
        "anchor_text": "The annual financial audit concluded on April 15, 2024.",
        "gold_date": (5, 3, 2024),
        "gold_text": "Layla closed three of the audit findings and assigned owners to the rest on May 3, 2024.",
        "wrong_dir_date": (3, 22, 2024),
        "wrong_dir_text": "Layla pulled the journal entries the auditors had requested on March 22, 2024.",
        "wrong_topic_date": (5, 2, 2024),
        "wrong_topic_text": "Diego had his annual physical on May 2, 2024.",
        "noise": [
            (
                (1, 25, 2024),
                "Felix attended the engineering all-hands on January 25, 2024.",
            ),
            (
                (8, 14, 2023),
                "Sara finished a beach-cleanup volunteer day on August 14, 2023.",
            ),
        ],
        "relation": "after",
        "query": "What did Layla do after the audit?",
    },
    {
        "person": "Yuki",
        "anchor_event": "the move",
        "anchor_date": (7, 1, 2023),
        "anchor_text": "Yuki moved from Seattle to Singapore on July 1, 2023.",
        "gold_date": (8, 10, 2023),
        "gold_text": "Yuki joined a local diving club and signed up for advanced certification on August 10, 2023.",
        "wrong_dir_date": (5, 19, 2023),
        "wrong_dir_text": "Yuki sold her car and downsized two thirds of her apartment on May 19, 2023.",
        "wrong_topic_date": (8, 11, 2023),
        "wrong_topic_text": "Henry filed his Q2 expense report on August 11, 2023.",
        "noise": [
            (
                (3, 30, 2024),
                "Marcus delivered the all-hands presentation on March 30, 2024.",
            ),
            ((10, 2, 2024), "Olivia hosted a housewarming brunch on October 2, 2024."),
        ],
        "relation": "since",
        "query": "What has Yuki taken up since the move?",
    },
    {
        "person": "Mira",
        "anchor_event": "the cutover",
        "anchor_date": (5, 18, 2024),
        "anchor_text": "The data-warehouse cutover ran on May 18, 2024.",
        "gold_date": (5, 25, 2024),
        "gold_text": "Mira ran a one-week stability review and signed off on the new warehouse on May 25, 2024.",
        "wrong_dir_date": (4, 30, 2024),
        "wrong_dir_text": "Mira finalized the rollback plan and the cutover runbook on April 30, 2024.",
        "wrong_topic_date": (5, 26, 2024),
        "wrong_topic_text": "Paul gave the all-hands presentation on May 26, 2024.",
        "noise": [
            (
                (11, 17, 2023),
                "Carla attended the company offsite on November 17, 2023.",
            ),
            ((1, 13, 2024), "Layla completed her tax filing on January 13, 2024."),
        ],
        "relation": "after",
        "query": "What did Mira sign off on after the cutover?",
    },
    {
        "person": "Hannah",
        "anchor_event": "the design summit",
        "anchor_date": (6, 15, 2023),
        "anchor_text": "Hannah attended the Lisbon design summit on June 15, 2023.",
        "gold_date": (7, 3, 2023),
        "gold_text": "Hannah brought the new sketching method back to the team and ran an internal workshop on July 3, 2023.",
        "wrong_dir_date": (5, 28, 2023),
        "wrong_dir_text": "Hannah finalized her summit travel and registered for the workshop track on May 28, 2023.",
        "wrong_topic_date": (7, 4, 2023),
        "wrong_topic_text": "Eric finished his half-marathon on July 4, 2023.",
        "noise": [
            (
                (2, 14, 2024),
                "Marcus went to the wedding rehearsal on February 14, 2024.",
            ),
            ((9, 22, 2023), "Aiden gave the investor pitch on September 22, 2023."),
        ],
        "relation": "after",
        "query": "What did Hannah bring back after the design summit?",
    },
    {
        "person": "Tom",
        "anchor_event": "the marathon",
        "anchor_date": (4, 15, 2024),
        "anchor_text": "Tom ran the Boston Marathon on April 15, 2024.",
        "gold_date": (3, 22, 2024),
        "gold_text": "Tom completed his final 22-mile training run and tapered his mileage on March 22, 2024.",
        "wrong_dir_date": (5, 4, 2024),
        "wrong_dir_text": "Tom posted his recovery diary and a knee-PT plan on May 4, 2024.",
        "wrong_topic_date": (3, 21, 2024),
        "wrong_topic_text": "Quinn signed her new apartment lease on March 21, 2024.",
        "noise": [
            ((10, 5, 2023), "Diego had his annual physical on October 5, 2023."),
            (
                (12, 10, 2024),
                "Maya made the December code-freeze announcement on December 10, 2024.",
            ),
        ],
        "relation": "before",
        "query": "What did Tom do before the marathon?",
    },
    {
        "person": "Olivia",
        "anchor_event": "the relocation",
        "anchor_date": (10, 1, 2024),
        "anchor_text": "Olivia relocated from New York to Austin on October 1, 2024.",
        "gold_date": (10, 18, 2024),
        "gold_text": "Olivia joined a co-working space and started attending Austin's Tuesday tech meetup on October 18, 2024.",
        "wrong_dir_date": (8, 14, 2024),
        "wrong_dir_text": "Olivia toured three apartments and shipped her first set of boxes on August 14, 2024.",
        "wrong_topic_date": (10, 19, 2024),
        "wrong_topic_text": "Henry submitted his self-evaluation on October 19, 2024.",
        "noise": [
            ((6, 15, 2023), "Hannah attended the design summit on June 15, 2023."),
            (
                (2, 8, 2025),
                "Sara took her puppy to its second vet visit on February 8, 2025.",
            ),
        ],
        "relation": "since",
        "query": "What has Olivia been doing since the relocation?",
    },
    {
        "person": "Felix",
        "anchor_event": "the client onsite",
        "anchor_date": (5, 11, 2024),
        "anchor_text": "Felix went on a one-week client onsite in Tokyo on May 11, 2024.",
        "gold_date": (5, 28, 2024),
        "gold_text": "Felix wrote a 12-page trip report and pushed three contract amendments on May 28, 2024.",
        "wrong_dir_date": (4, 22, 2024),
        "wrong_dir_text": "Felix prepared the onsite agenda and the demo environments on April 22, 2024.",
        "wrong_topic_date": (5, 29, 2024),
        "wrong_topic_text": "Vera signed her new apartment lease on May 29, 2024.",
        "noise": [
            (
                (11, 30, 2024),
                "Priya led the December team retrospective on November 30, 2024.",
            ),
            ((9, 9, 2023), "Henry had his Q3 performance review on September 9, 2023."),
        ],
        "relation": "after",
        "query": "What did Felix do after the client onsite?",
    },
    {
        "person": "Carla",
        "anchor_event": "the promotion",
        "anchor_date": (1, 8, 2024),
        "anchor_text": "Carla was promoted to senior director on January 8, 2024.",
        "gold_date": (1, 23, 2024),
        "gold_text": "Carla onboarded two new managers and rebalanced the team into three pods on January 23, 2024.",
        "wrong_dir_date": (12, 2, 2023),
        "wrong_dir_text": "Carla circulated her promotion case and lined up the formal review on December 2, 2023.",
        "wrong_topic_date": (1, 24, 2024),
        "wrong_topic_text": "Kim attended a kitchen-design class on January 24, 2024.",
        "noise": [
            (
                (7, 21, 2023),
                "Quinn signed a one-year apartment lease on July 21, 2023.",
            ),
            ((3, 10, 2025), "Yuki completed an open-water dive on March 10, 2025."),
        ],
        "relation": "after",
        "query": "What did Carla do after the promotion?",
    },
]


def main() -> None:
    rng = random.Random(20260429)
    docs = []
    queries = []
    gold_rows = []

    for i, c in enumerate(CLUSTERS):
        gm, gd, gy = c["gold_date"]
        am, ad, ay = c["anchor_date"]
        wm, wd, wy = c["wrong_dir_date"]
        tm, td, ty = c["wrong_topic_date"]

        gold_id = f"cr_{i:03d}_g"
        anchor_id = f"cr_{i:03d}_a"
        wrong_dir_id = f"cr_{i:03d}_wd"
        wrong_topic_id = f"cr_{i:03d}_wt"

        docs.append(
            {"doc_id": gold_id, "text": c["gold_text"], "ref_time": iso(gm, gd, gy)}
        )
        docs.append(
            {"doc_id": anchor_id, "text": c["anchor_text"], "ref_time": iso(am, ad, ay)}
        )
        docs.append(
            {
                "doc_id": wrong_dir_id,
                "text": c["wrong_dir_text"],
                "ref_time": iso(wm, wd, wy),
            }
        )
        docs.append(
            {
                "doc_id": wrong_topic_id,
                "text": c["wrong_topic_text"],
                "ref_time": iso(tm, td, ty),
            }
        )
        for j, ((nm, nd, ny), ntext) in enumerate(c["noise"]):
            docs.append(
                {
                    "doc_id": f"cr_{i:03d}_n{j}",
                    "text": ntext,
                    "ref_time": iso(nm, nd, ny),
                }
            )

        qid = f"cr_q_{i:03d}"
        queries.append({"query_id": qid, "text": c["query"], "ref_time": REF_TIME})
        gold_rows.append({"query_id": qid, "relevant_doc_ids": [gold_id]})

    rng.shuffle(docs)
    with open(DATA_DIR / "causal_relative_docs.jsonl", "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    with open(DATA_DIR / "causal_relative_queries.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(DATA_DIR / "causal_relative_gold.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gold_rows)
    print(
        f"causal_relative: {len(docs)} docs, {len(queries)} queries, {len(gold_rows)} gold rows"
    )


if __name__ == "__main__":
    main()
