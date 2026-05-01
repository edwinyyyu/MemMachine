"""F5 — Synthetic data for Allen-relation retrieval.

Builds three jsonl files under data/:
- allen_docs.jsonl    — docs (anchor-event + content docs)
- allen_queries.jsonl — queries testing before/after/during/overlaps/contains
- allen_gold.jsonl    — per-query relevant_doc_ids + relation label

All intervals are UTC. ref_time is fixed so the synthetic data is
deterministic and auditable.

Corpus design:
- 10 anchor-event docs define named events with absolute dates
  (e.g., "I had my wedding on June 12, 2020"). Each event gets 2-4
  content docs that reference it with a qualitative relation.
- 20 content docs reference anchor events via before/after/during/
  overlaps/contains relations. Each has an absolute time ALSO mentioned
  so a base retriever *could* nominally match on the anchor's span —
  but gold is tied to the relation, not the co-mention.
- Some distractor docs at unrelated dates to keep retrieval non-trivial.

20 queries, 4 per relation:
- before    × 4
- after     × 4
- during    × 4
- overlaps  × 4
- contains  × 4
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
ISO_REF = REF_TIME.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Anchor events (name, date-range iso strings, doc text)
# ---------------------------------------------------------------------------
@dataclass
class Anchor:
    event_id: str  # short slug
    surface: str  # canonical surface (e.g., "my wedding")
    anchor_text: str  # doc text that establishes the event
    # The canonical interval the event occupies.
    earliest: datetime
    latest: datetime

    @property
    def span(self) -> timedelta:
        return self.latest - self.earliest


ANCHORS: list[Anchor] = [
    Anchor(
        event_id="wedding",
        surface="my wedding",
        anchor_text="I had my wedding on June 12, 2020.",
        earliest=datetime(2020, 6, 12, tzinfo=timezone.utc),
        latest=datetime(2020, 6, 13, tzinfo=timezone.utc),
    ),
    Anchor(
        event_id="graduation",
        surface="my graduation",
        anchor_text="I graduated college on May 18, 2015.",
        earliest=datetime(2015, 5, 18, tzinfo=timezone.utc),
        latest=datetime(2015, 5, 19, tzinfo=timezone.utc),
    ),
    Anchor(
        event_id="promotion",
        surface="my promotion",
        anchor_text="I got my promotion to senior engineer on October 3, 2022.",
        earliest=datetime(2022, 10, 3, tzinfo=timezone.utc),
        latest=datetime(2022, 10, 4, tzinfo=timezone.utc),
    ),
    Anchor(
        event_id="move_denver",
        surface="my move to Denver",
        anchor_text="I moved to Denver on August 1, 2018.",
        earliest=datetime(2018, 8, 1, tzinfo=timezone.utc),
        latest=datetime(2018, 8, 2, tzinfo=timezone.utc),
    ),
    Anchor(
        event_id="europe_trip",
        surface="my Europe trip",
        anchor_text=(
            "I was in Europe from July 5 through July 26, 2019 on my Europe trip."
        ),
        earliest=datetime(2019, 7, 5, tzinfo=timezone.utc),
        latest=datetime(2019, 7, 27, tzinfo=timezone.utc),  # latest exclusive
    ),
    Anchor(
        event_id="marathon",
        surface="the marathon",
        anchor_text="I ran the marathon on April 14, 2024.",
        earliest=datetime(2024, 4, 14, tzinfo=timezone.utc),
        latest=datetime(2024, 4, 15, tzinfo=timezone.utc),
    ),
    Anchor(
        event_id="house_purchase",
        surface="the house purchase",
        anchor_text="I closed on the house on February 9, 2021.",
        earliest=datetime(2021, 2, 9, tzinfo=timezone.utc),
        latest=datetime(2021, 2, 10, tzinfo=timezone.utc),
    ),
    Anchor(
        event_id="conference",
        surface="the conference",
        anchor_text=("I attended the conference from March 10 through March 13, 2023."),
        earliest=datetime(2023, 3, 10, tzinfo=timezone.utc),
        latest=datetime(2023, 3, 14, tzinfo=timezone.utc),
    ),
    Anchor(
        event_id="new_job",
        surface="my new job",
        anchor_text="I started my new job on September 6, 2016.",
        earliest=datetime(2016, 9, 6, tzinfo=timezone.utc),
        latest=datetime(2016, 9, 7, tzinfo=timezone.utc),
    ),
    Anchor(
        event_id="honeymoon",
        surface="my honeymoon",
        anchor_text=(
            "My honeymoon was from June 20 through July 4, 2020, right "
            "after the wedding."
        ),
        earliest=datetime(2020, 6, 20, tzinfo=timezone.utc),
        latest=datetime(2020, 7, 5, tzinfo=timezone.utc),
    ),
]

ANCHOR_BY_ID: dict[str, Anchor] = {a.event_id: a for a in ANCHORS}


# ---------------------------------------------------------------------------
# Content docs: each references an anchor with a known relation
# ---------------------------------------------------------------------------
@dataclass
class ContentDoc:
    doc_id: str
    text: str
    anchor_id: str
    relation: str  # before/after/during/overlaps/contains
    # Absolute bracket for the event described in the doc.
    earliest: datetime
    latest: datetime


CONTENT: list[ContentDoc] = [
    # --- wedding (single-day anchor) ---
    ContentDoc(
        "c_wedding_before_1",
        "Two weeks before my wedding we tried the cake sampling.",
        "wedding",
        "before",
        datetime(2020, 5, 29, tzinfo=timezone.utc),
        datetime(2020, 5, 30, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_wedding_after_1",
        "The month after my wedding we went to Vermont.",
        "wedding",
        "after",
        datetime(2020, 7, 12, tzinfo=timezone.utc),
        datetime(2020, 8, 12, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_wedding_during_1",
        "During my wedding ceremony my cousin got engaged to her partner.",
        "wedding",
        "during",
        datetime(2020, 6, 12, tzinfo=timezone.utc),
        datetime(2020, 6, 13, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_wedding_contains_1",
        (
            "Between June 1 and June 30, 2020 — which of course includes "
            "our wedding — the hydrangeas were in bloom."
        ),
        "wedding",
        "contains",
        datetime(2020, 6, 1, tzinfo=timezone.utc),
        datetime(2020, 7, 1, tzinfo=timezone.utc),
    ),
    # --- graduation (single day) ---
    ContentDoc(
        "c_grad_before_1",
        "The week before my graduation I turned in my final project.",
        "graduation",
        "before",
        datetime(2015, 5, 11, tzinfo=timezone.utc),
        datetime(2015, 5, 18, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_grad_after_1",
        "Right after my graduation I backpacked through Asia.",
        "graduation",
        "after",
        datetime(2015, 5, 20, tzinfo=timezone.utc),
        datetime(2015, 8, 1, tzinfo=timezone.utc),
    ),
    # --- promotion (single day) ---
    ContentDoc(
        "c_promo_before_1",
        "Two months before my promotion I gave a big internal talk.",
        "promotion",
        "before",
        datetime(2022, 8, 3, tzinfo=timezone.utc),
        datetime(2022, 8, 4, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_promo_after_1",
        "Shortly after my promotion I bought a new laptop.",
        "promotion",
        "after",
        datetime(2022, 10, 10, tzinfo=timezone.utc),
        datetime(2022, 10, 11, tzinfo=timezone.utc),
    ),
    # --- move_denver ---
    ContentDoc(
        "c_move_before_1",
        "The month before my move to Denver I sold a lot of furniture.",
        "move_denver",
        "before",
        datetime(2018, 7, 1, tzinfo=timezone.utc),
        datetime(2018, 8, 1, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_move_after_1",
        "The week after my move to Denver I joined a climbing gym.",
        "move_denver",
        "after",
        datetime(2018, 8, 2, tzinfo=timezone.utc),
        datetime(2018, 8, 9, tzinfo=timezone.utc),
    ),
    # --- europe_trip (3-week interval) ---
    ContentDoc(
        "c_europe_during_1",
        "During my Europe trip I spent three days in Prague.",
        "europe_trip",
        "during",
        datetime(2019, 7, 14, tzinfo=timezone.utc),
        datetime(2019, 7, 17, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_europe_during_2",
        "While I was on my Europe trip I turned 30 in a cafe in Paris.",
        "europe_trip",
        "during",
        datetime(2019, 7, 12, tzinfo=timezone.utc),
        datetime(2019, 7, 13, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_europe_before_1",
        "The day before my Europe trip I misplaced my passport.",
        "europe_trip",
        "before",
        datetime(2019, 7, 4, tzinfo=timezone.utc),
        datetime(2019, 7, 5, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_europe_after_1",
        "The week after my Europe trip I caught a nasty cold.",
        "europe_trip",
        "after",
        datetime(2019, 7, 27, tzinfo=timezone.utc),
        datetime(2019, 8, 3, tzinfo=timezone.utc),
    ),
    # --- marathon (single day) ---
    ContentDoc(
        "c_marathon_before_1",
        "Three weeks before the marathon I tapered my training.",
        "marathon",
        "before",
        datetime(2024, 3, 24, tzinfo=timezone.utc),
        datetime(2024, 3, 25, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_marathon_after_1",
        "The day after the marathon I could barely walk.",
        "marathon",
        "after",
        datetime(2024, 4, 15, tzinfo=timezone.utc),
        datetime(2024, 4, 16, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_marathon_contains_1",
        (
            "From April 10 through April 20, 2024 I was in Boston for "
            "the running series, which included the marathon mid-week."
        ),
        "marathon",
        "contains",
        datetime(2024, 4, 10, tzinfo=timezone.utc),
        datetime(2024, 4, 21, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_grad_contains_1",
        (
            "From May 12 through May 25, 2015 my parents were visiting "
            "for a run of events, which included my graduation."
        ),
        "graduation",
        "contains",
        datetime(2015, 5, 12, tzinfo=timezone.utc),
        datetime(2015, 5, 26, tzinfo=timezone.utc),
    ),
    # --- honeymoon (interval) ---
    ContentDoc(
        "c_honeymoon_during_1",
        "During my honeymoon we hiked a coastal trail and saw dolphins.",
        "honeymoon",
        "during",
        datetime(2020, 6, 25, tzinfo=timezone.utc),
        datetime(2020, 6, 26, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_honeymoon_overlaps_1",
        ("From late June into early July 2020 I journaled every morning."),
        "honeymoon",
        "overlaps",
        datetime(2020, 6, 25, tzinfo=timezone.utc),
        datetime(2020, 7, 10, tzinfo=timezone.utc),
    ),
    # --- conference (interval) ---
    ContentDoc(
        "c_conference_during_1",
        "During the conference I met an old mentor and we had dinner.",
        "conference",
        "during",
        datetime(2023, 3, 11, tzinfo=timezone.utc),
        datetime(2023, 3, 12, tzinfo=timezone.utc),
    ),
    ContentDoc(
        "c_conference_contains_1",
        (
            "From March 5 through March 20, 2023 I was traveling for "
            "work, which included the conference in the middle."
        ),
        "conference",
        "contains",
        datetime(2023, 3, 5, tzinfo=timezone.utc),
        datetime(2023, 3, 21, tzinfo=timezone.utc),
    ),
]


# ---------------------------------------------------------------------------
# Distractor docs (absolute times with no relational language).
# Written deliberately close-in-time to some anchor events so they'd
# falsely match on co-mention-based retrievers.
# ---------------------------------------------------------------------------
@dataclass
class DistractorDoc:
    doc_id: str
    text: str


DISTRACTORS: list[DistractorDoc] = [
    DistractorDoc(
        "d_abs_1",
        "On June 12, 2020 the stock market had a volatile session.",
    ),
    DistractorDoc(
        "d_abs_2",
        "On May 18, 2015 I read a book about lighthouses.",
    ),
    DistractorDoc(
        "d_abs_3",
        "On April 14, 2024 there was a partial solar eclipse reported in the news.",
    ),
    DistractorDoc(
        "d_abs_4",
        "On February 9, 2021 the local library reopened.",
    ),
]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
@dataclass
class Query:
    query_id: str
    text: str
    relation: str
    anchor_id: str
    # Optional gold overrides; when empty, gold is computed as the set
    # of content docs with matching (anchor_id, relation).
    gold_override: list[str] = field(default_factory=list)


QUERIES: list[Query] = [
    # before ×4
    Query("q_before_wedding", "What happened before my wedding?", "before", "wedding"),
    Query(
        "q_before_grad", "What happened before my graduation?", "before", "graduation"
    ),
    Query(
        "q_before_marathon",
        "What happened in the weeks before the marathon?",
        "before",
        "marathon",
    ),
    Query(
        "q_before_europe",
        "What happened right before my Europe trip?",
        "before",
        "europe_trip",
    ),
    # after ×4
    Query("q_after_wedding", "What happened after my wedding?", "after", "wedding"),
    Query("q_after_promo", "What happened after my promotion?", "after", "promotion"),
    Query(
        "q_after_move",
        "What happened shortly after my move to Denver?",
        "after",
        "move_denver",
    ),
    Query(
        "q_after_marathon",
        "What happened the day after the marathon?",
        "after",
        "marathon",
    ),
    # during ×4
    Query(
        "q_during_europe",
        "What happened during my Europe trip?",
        "during",
        "europe_trip",
    ),
    Query(
        "q_during_honeymoon",
        "What happened during my honeymoon?",
        "during",
        "honeymoon",
    ),
    Query(
        "q_during_wedding",
        "What happened during my wedding weekend?",
        "during",
        "wedding",
    ),
    Query(
        "q_during_conference",
        "What happened during the conference?",
        "during",
        "conference",
    ),
    # overlaps ×4
    Query(
        "q_overlaps_wedding",
        "What events overlapped with my wedding month?",
        "overlaps",
        "wedding",
    ),
    Query(
        "q_overlaps_honeymoon",
        "What overlapped with my honeymoon?",
        "overlaps",
        "honeymoon",
    ),
    Query(
        "q_overlaps_europe",
        "What coincided with my Europe trip?",
        "overlaps",
        "europe_trip",
    ),
    Query(
        "q_overlaps_conference",
        "What overlapped with the conference?",
        "overlaps",
        "conference",
    ),
    # contains ×4 (doc interval fully contains the anchor event)
    Query(
        "q_contains_wedding",
        "What larger event contained my wedding day?",
        "contains",
        "wedding",
    ),
    Query(
        "q_contains_conference",
        "What larger trip contained the conference?",
        "contains",
        "conference",
    ),
    Query(
        "q_contains_marathon",
        "What period contained the marathon?",
        "contains",
        "marathon",
    ),
    Query(
        "q_contains_grad",
        "What period contained my graduation?",
        "contains",
        "graduation",
    ),
]


# ---------------------------------------------------------------------------
# Gold computation
# ---------------------------------------------------------------------------
def _content_interval(c: ContentDoc) -> tuple[datetime, datetime]:
    return c.earliest, c.latest


def _iv_overlap(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    return a0 < b1 and b0 < a1


def _compute_gold(q: Query) -> list[str]:
    """Gold: content docs whose (anchor_id, relation) MATCHES the query
    relation semantics against the anchor interval.

    We directly check the Allen relation on the pre-declared intervals
    so gold is principled — the relation tags on ContentDoc are
    author-provided but we re-check against intervals for safety.
    """
    if q.gold_override:
        return q.gold_override
    anchor = ANCHOR_BY_ID[q.anchor_id]
    a0, a1 = anchor.earliest, anchor.latest
    gold: list[str] = []
    for c in CONTENT:
        c0, c1 = _content_interval(c)
        # Only consider content docs that reference THIS anchor.
        if c.anchor_id != q.anchor_id:
            continue
        if q.relation == "before":
            if c1 <= a0:
                gold.append(c.doc_id)
        elif q.relation == "after":
            if c0 >= a1:
                gold.append(c.doc_id)
        elif q.relation == "during":
            # Doc interval fully inside anchor
            if c0 >= a0 and c1 <= a1:
                gold.append(c.doc_id)
        elif q.relation == "contains":
            # Doc interval contains anchor
            if c0 <= a0 and c1 >= a1:
                gold.append(c.doc_id)
        elif q.relation == "overlaps":
            # Any temporal overlap counts (excluding before/after);
            # during and contains also count as overlap candidates.
            if _iv_overlap(a0, a1, c0, c1):
                gold.append(c.doc_id)
    return gold


# ---------------------------------------------------------------------------
# Emit jsonl
# ---------------------------------------------------------------------------
def main() -> None:
    docs_path = DATA_DIR / "allen_docs.jsonl"
    queries_path = DATA_DIR / "allen_queries.jsonl"
    gold_path = DATA_DIR / "allen_gold.jsonl"

    with docs_path.open("w") as f:
        # anchor docs
        for a in ANCHORS:
            obj = {
                "doc_id": f"a_{a.event_id}",
                "text": a.anchor_text,
                "ref_time": ISO_REF,
                "anchor_id": a.event_id,
                "is_anchor": True,
            }
            f.write(json.dumps(obj) + "\n")
        # content docs
        for c in CONTENT:
            obj = {
                "doc_id": c.doc_id,
                "text": c.text,
                "ref_time": ISO_REF,
                "anchor_id": c.anchor_id,
                "declared_relation": c.relation,
                "is_anchor": False,
            }
            f.write(json.dumps(obj) + "\n")
        # distractors
        for d in DISTRACTORS:
            obj = {
                "doc_id": d.doc_id,
                "text": d.text,
                "ref_time": ISO_REF,
                "anchor_id": None,
                "is_anchor": False,
            }
            f.write(json.dumps(obj) + "\n")

    with queries_path.open("w") as f:
        for q in QUERIES:
            obj = {
                "query_id": q.query_id,
                "text": q.text,
                "ref_time": ISO_REF,
                "relation": q.relation,
                "anchor_id": q.anchor_id,
                "anchor_span": ANCHOR_BY_ID[q.anchor_id].surface,
            }
            f.write(json.dumps(obj) + "\n")

    with gold_path.open("w") as f:
        for q in QUERIES:
            relevant = _compute_gold(q)
            f.write(
                json.dumps(
                    {
                        "query_id": q.query_id,
                        "relevant_doc_ids": relevant,
                        "relation": q.relation,
                        "anchor_id": q.anchor_id,
                    }
                )
                + "\n"
            )

    print(
        f"Wrote {len(ANCHORS) + len(CONTENT) + len(DISTRACTORS)} docs, "
        f"{len(QUERIES)} queries, gold under {gold_path}"
    )


if __name__ == "__main__":
    main()
