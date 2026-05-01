"""Synthetic polarity test set.

Generates:
- 20 docs with mixed polarities (1-3 time expressions each). At least 5
  affirmed/negated pairs: same event span + same referenced time window
  but opposite polarity.
- 15 queries covering three polarity intents:
    - affirmed-matching ("when did X happen?"): only affirmed docs relevant
    - negation-preserving ("what did not happen X?"): only negated relevant
    - polarity-agnostic ("what was discussed about X?"): both relevant
- Gold relevance mapping for each query.

Writes to:
    data/polarity_docs.jsonl
    data/polarity_queries.jsonl
    data/polarity_gold.jsonl

Reference time is fixed at 2026-04-23T12:00:00Z to match the base corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2026-04-23T12:00:00Z"


# ---------------------------------------------------------------------------
# Docs: (doc_id, text, polarity_by_span, "topic key")
#
# The topic key groups docs that talk about the same {event, time} — used
# to construct affirmed/negated pairs and gold sets. Polarity here is a
# doc-level annotation used by the gold builder; the extractor will
# (hopefully) recover it from the text.
# ---------------------------------------------------------------------------
# 5 affirmed/negated pairs (10 docs), 5 hypothetical, 3 uncertain, 2 extra
# affirmed w/ different topics. Total = 20.
DOCS: list[dict] = [
    # --- Pair 1: conference, last March ---
    {
        "doc_id": "pol_d01",
        "text": "She attended the annual conference last March.",
        "topic": "conference_last_march",
        "polarity": "affirmed",
    },
    {
        "doc_id": "pol_d02",
        "text": "She didn't attend the annual conference last March.",
        "topic": "conference_last_march",
        "polarity": "negated",
    },
    # --- Pair 2: meeting Alice, last week ---
    {
        "doc_id": "pol_d03",
        "text": "I met Alice last week at the cafe.",
        "topic": "meet_alice_last_week",
        "polarity": "affirmed",
    },
    {
        "doc_id": "pol_d04",
        "text": "I didn't meet Alice last week; she was out of town.",
        "topic": "meet_alice_last_week",
        "polarity": "negated",
    },
    # --- Pair 3: launch in 2024 ---
    {
        "doc_id": "pol_d05",
        "text": "The team launched the product in 2024.",
        "topic": "product_launch_2024",
        "polarity": "affirmed",
    },
    {
        "doc_id": "pol_d06",
        "text": "The team never launched the product in 2024.",
        "topic": "product_launch_2024",
        "polarity": "negated",
    },
    # --- Pair 4: submit the report on April 10 ---
    {
        "doc_id": "pol_d07",
        "text": "He submitted the report on April 10, 2026.",
        "topic": "report_april10",
        "polarity": "affirmed",
    },
    {
        "doc_id": "pol_d08",
        "text": "He failed to submit the report on April 10, 2026.",
        "topic": "report_april10",
        "polarity": "negated",
    },
    # --- Pair 5: travel to Berlin in December ---
    {
        "doc_id": "pol_d09",
        "text": "They traveled to Berlin in December 2025.",
        "topic": "berlin_dec_2025",
        "polarity": "affirmed",
    },
    {
        "doc_id": "pol_d10",
        "text": "They didn't travel to Berlin in December 2025.",
        "topic": "berlin_dec_2025",
        "polarity": "negated",
    },
    # --- Hypothetical docs ---
    {
        "doc_id": "pol_d11",
        "text": "If she had attended, it would have been last March.",
        "topic": "conference_last_march",
        "polarity": "hypothetical",
    },
    {
        "doc_id": "pol_d12",
        "text": ("If the team had launched in 2024, sales would have doubled."),
        "topic": "product_launch_2024",
        "polarity": "hypothetical",
    },
    {
        "doc_id": "pol_d13",
        "text": "They are planning to travel to Berlin in December 2025.",
        "topic": "berlin_dec_2025",
        "polarity": "hypothetical",
    },
    {
        "doc_id": "pol_d14",
        "text": (
            "He would have submitted the report on April 10, 2026 "
            "if the data had arrived."
        ),
        "topic": "report_april10",
        "polarity": "hypothetical",
    },
    {
        "doc_id": "pol_d15",
        "text": ("If I had met Alice last week, I would have asked about the project."),
        "topic": "meet_alice_last_week",
        "polarity": "hypothetical",
    },
    # --- Uncertain docs ---
    {
        "doc_id": "pol_d16",
        "text": "She probably attended the conference last March.",
        "topic": "conference_last_march",
        "polarity": "uncertain",
    },
    {
        "doc_id": "pol_d17",
        "text": "They maybe traveled to Berlin in December 2025.",
        "topic": "berlin_dec_2025",
        "polarity": "uncertain",
    },
    {
        "doc_id": "pol_d18",
        "text": ("I think the team launched the product in 2024, but I'm not certain."),
        "topic": "product_launch_2024",
        "polarity": "uncertain",
    },
    # --- Extra affirmed docs in the same topics (for richer gold) ---
    {
        "doc_id": "pol_d19",
        "text": ("During the conference last March, she delivered a keynote."),
        "topic": "conference_last_march",
        "polarity": "affirmed",
    },
    {
        "doc_id": "pol_d20",
        "text": ("I had coffee with Alice last week and we discussed the roadmap."),
        "topic": "meet_alice_last_week",
        "polarity": "affirmed",
    },
]


# ---------------------------------------------------------------------------
# Queries: (query_id, text, intent, topic)
# intents: "affirmed" | "negation" | "agnostic"
# ---------------------------------------------------------------------------
QUERIES: list[dict] = [
    # --- Affirmed-matching queries ---
    {
        "query_id": "pol_q01",
        "text": "When did she attend the conference?",
        "intent": "affirmed",
        "topic": "conference_last_march",
    },
    {
        "query_id": "pol_q02",
        "text": "When did I meet Alice?",
        "intent": "affirmed",
        "topic": "meet_alice_last_week",
    },
    {
        "query_id": "pol_q03",
        "text": "When was the product launched?",
        "intent": "affirmed",
        "topic": "product_launch_2024",
    },
    {
        "query_id": "pol_q04",
        "text": "When did he submit the report?",
        "intent": "affirmed",
        "topic": "report_april10",
    },
    {
        "query_id": "pol_q05",
        "text": "When did they travel to Berlin?",
        "intent": "affirmed",
        "topic": "berlin_dec_2025",
    },
    # --- Negation-preserving queries ---
    {
        "query_id": "pol_q06",
        "text": "What didn't happen last March?",
        "intent": "negation",
        "topic": "conference_last_march",
    },
    {
        "query_id": "pol_q07",
        "text": "Who did I fail to meet last week?",
        "intent": "negation",
        "topic": "meet_alice_last_week",
    },
    {
        "query_id": "pol_q08",
        "text": "What didn't the team launch in 2024?",
        "intent": "negation",
        "topic": "product_launch_2024",
    },
    {
        "query_id": "pol_q09",
        "text": "What report wasn't submitted on April 10, 2026?",
        "intent": "negation",
        "topic": "report_april10",
    },
    {
        "query_id": "pol_q10",
        "text": "What trip didn't happen in December 2025?",
        "intent": "negation",
        "topic": "berlin_dec_2025",
    },
    # --- Polarity-agnostic queries ---
    {
        "query_id": "pol_q11",
        "text": "What was discussed about last March?",
        "intent": "agnostic",
        "topic": "conference_last_march",
    },
    {
        "query_id": "pol_q12",
        "text": "Anything related to Alice last week?",
        "intent": "agnostic",
        "topic": "meet_alice_last_week",
    },
    {
        "query_id": "pol_q13",
        "text": "Any information about the 2024 product launch?",
        "intent": "agnostic",
        "topic": "product_launch_2024",
    },
    {
        "query_id": "pol_q14",
        "text": "Anything about the April 10, 2026 report?",
        "intent": "agnostic",
        "topic": "report_april10",
    },
    {
        "query_id": "pol_q15",
        "text": "What was mentioned about Berlin in December 2025?",
        "intent": "agnostic",
        "topic": "berlin_dec_2025",
    },
]


# ---------------------------------------------------------------------------
# Gold: topic + intent -> relevant doc_ids
# ---------------------------------------------------------------------------
def _docs_for_topic(topic: str) -> list[dict]:
    return [d for d in DOCS if d["topic"] == topic]


def _gold_for(query: dict) -> list[str]:
    topic = query["topic"]
    intent = query["intent"]
    docs = _docs_for_topic(topic)
    if intent == "affirmed":
        return sorted(d["doc_id"] for d in docs if d["polarity"] == "affirmed")
    if intent == "negation":
        return sorted(d["doc_id"] for d in docs if d["polarity"] == "negated")
    # agnostic: every doc that mentions the topic, regardless of polarity
    return sorted(d["doc_id"] for d in docs)


def main() -> None:
    # Docs: match the base corpus format so existing helpers can parse.
    with (DATA_DIR / "polarity_docs.jsonl").open("w") as f:
        for d in DOCS:
            rec = {
                "doc_id": d["doc_id"],
                "text": d["text"],
                "ref_time": REF_TIME,
                "polarity_gold": d["polarity"],
                "topic": d["topic"],
            }
            f.write(json.dumps(rec) + "\n")

    with (DATA_DIR / "polarity_queries.jsonl").open("w") as f:
        for q in QUERIES:
            rec = {
                "query_id": q["query_id"],
                "text": q["text"],
                "ref_time": REF_TIME,
                "intent": q["intent"],
                "topic": q["topic"],
            }
            f.write(json.dumps(rec) + "\n")

    with (DATA_DIR / "polarity_gold.jsonl").open("w") as f:
        for q in QUERIES:
            rec = {
                "query_id": q["query_id"],
                "intent": q["intent"],
                "topic": q["topic"],
                "relevant_doc_ids": _gold_for(q),
            }
            f.write(json.dumps(rec) + "\n")

    print(
        f"Wrote {len(DOCS)} docs, {len(QUERIES)} queries, {len(QUERIES)} gold records."
    )


if __name__ == "__main__":
    main()
