"""Build a real-distribution temporal retrieval benchmark from TempReason.

Source: tonytan48/TempReason test_l2.json + test_l3.json (downloaded raw).
Each record has:
- question:   "Who was the head coach of <team> in <date>?"  (L2)
              "Who was the head of <X> after <Y>?"           (L3)
- date:       reference date for L2 questions (e.g. "February 03, 2018")
- text_answers: {"text": [<gold answer entity>]}
- fact_context: multi-line "<E> <rel> <T> from <month_year> to <month_year>"

We adapt to retrieval format:
- Each fact line becomes one document (deduped across questions, sharing answer entity).
- For each query, gold_doc = the fact line containing the answer entity AND
  satisfying the query's temporal constraint.
- Query ref_time = the question's `date` (L2) or 2026-04-24 default (L3, since
  these are entity-relative).
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

L2_PATH = Path("/tmp/claude/test_l2.json")
L3_PATH = Path("/tmp/claude/test_l3.json")
OUT_DIR = Path(
    "/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/temporal_extraction/data"
)

N_L2 = 40
N_L3 = 30
SEED = 42
MAX_FACTS_PER_QUERY = 3  # cap to keep corpus small (gold + 2 distractors)

# Default ref_time for L3 queries (no explicit date)
DEFAULT_REF_TIME = datetime(2026, 4, 24, tzinfo=timezone.utc)

MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def parse_facts(fact_context: str) -> list[str]:
    """Split a fact_context blob into individual fact statements."""
    out: list[str] = []
    for line in fact_context.split("\n"):
        line = line.strip()
        if not line or line == ".":
            continue
        # Some lines end with period
        if not line.endswith("."):
            line = line + "."
        out.append(line)
    return out


_RE_DATE_RANGE = re.compile(
    r"from\s+([A-Z][a-z]{2}),\s*(\d{4})\s+to\s+([A-Z][a-z]{2}),\s*(\d{4})"
)


def fact_covers_date(fact: str, target: datetime) -> bool:
    """Check if the date range in `fact` covers `target`."""
    m = _RE_DATE_RANGE.search(fact)
    if not m:
        return False
    sm, sy, em, ey = m.groups()
    sm = MONTH_MAP[sm]
    em = MONTH_MAP[em]
    sy = int(sy)
    ey = int(ey)
    start = datetime(sy, sm, 1, tzinfo=timezone.utc)
    # end month is exclusive in TempReason (from..to where to is the start of next)
    if em == 12:
        end = datetime(ey + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(ey, em + 1, 1, tzinfo=timezone.utc)
    return start <= target < end


_RE_QUESTION_DATE = re.compile(r"in\s+([A-Z][a-z]{2}),?\s+(\d{4})", re.IGNORECASE)


def parse_l2_date(date_str: str) -> datetime:
    """Parse 'February 03, 2018' to datetime."""
    return datetime.strptime(date_str, "%B %d, %Y").replace(tzinfo=timezone.utc)


def fact_contains_answer(fact: str, answer: str) -> bool:
    return answer in fact


def make_corpus_and_queries() -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(SEED)

    # Load L2
    l2_records = []
    with L2_PATH.open() as f:
        for line in f:
            if line.strip():
                l2_records.append(json.loads(line))
    rng.shuffle(l2_records)

    # Load L3
    l3_records = []
    with L3_PATH.open() as f:
        for line in f:
            if line.strip():
                l3_records.append(json.loads(line))
    rng.shuffle(l3_records)

    docs: list[dict] = []
    fact_to_id: dict[str, str] = {}
    queries: list[dict] = []
    gold: list[dict] = []

    def get_or_add_doc(fact: str, ref_time: datetime) -> str:
        if fact in fact_to_id:
            return fact_to_id[fact]
        did = f"d_{len(docs):05d}"
        fact_to_id[fact] = did
        docs.append(
            {
                "doc_id": did,
                "text": fact,
                "ref_time": ref_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        return did

    # Process L2
    seen_l2 = 0
    for rec in l2_records:
        if seen_l2 >= N_L2:
            break
        try:
            target = parse_l2_date(rec["date"])
        except Exception:
            continue
        answers = rec.get("text_answers", {}).get("text", [])
        if not answers:
            continue
        ans = answers[0]
        facts = parse_facts(rec.get("fact_context", ""))
        if len(facts) < 2:
            continue
        # Cap facts per query: keep gold + a few distractors
        gold_idxs = [
            i
            for i, f in enumerate(facts)
            if fact_contains_answer(f, ans) and fact_covers_date(f, target)
        ]
        if not gold_idxs:
            continue
        keep_idxs = set(gold_idxs)
        # Add distractors prioritizing same-entity facts (other dates)
        same_entity_idxs = [i for i in range(len(facts)) if i not in keep_idxs]
        for i in same_entity_idxs[: MAX_FACTS_PER_QUERY - len(keep_idxs)]:
            keep_idxs.add(i)
        facts = [f for i, f in enumerate(facts) if i in keep_idxs]
        # Assign every fact to the corpus, all sharing the L2 query date as ref_time
        # (real distribution: docs and queries will likely share ref_time for L2).
        gold_dids: list[str] = []
        for fact in facts:
            did = get_or_add_doc(fact, target)
            if fact_contains_answer(fact, ans) and fact_covers_date(fact, target):
                gold_dids.append(did)
        if not gold_dids:
            continue  # discard if we can't find a unambiguous gold
        qid = f"q_l2_{seen_l2:04d}"
        queries.append(
            {
                "query_id": qid,
                "text": rec["question"],
                "ref_time": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "subset": "L2",
            }
        )
        gold.append({"query_id": qid, "relevant_doc_ids": list(set(gold_dids))})
        seen_l2 += 1

    # Process L3 — uses "before/after <entity>" semantics
    seen_l3 = 0
    for rec in l3_records:
        if seen_l3 >= N_L3:
            break
        answers = rec.get("text_answers", {}).get("text", [])
        if not answers:
            continue
        ans = answers[0]
        facts = parse_facts(rec.get("fact_context", ""))
        if len(facts) < 2:
            continue
        # Cap facts per query
        gold_idxs = [i for i, f in enumerate(facts) if fact_contains_answer(f, ans)]
        if not gold_idxs:
            continue
        keep_idxs = set(gold_idxs)
        same_entity_idxs = [i for i in range(len(facts)) if i not in keep_idxs]
        for i in same_entity_idxs[: MAX_FACTS_PER_QUERY - len(keep_idxs)]:
            keep_idxs.add(i)
        facts = [f for i, f in enumerate(facts) if i in keep_idxs]
        # For L3, no concrete date — we use DEFAULT_REF_TIME.
        ref_time = DEFAULT_REF_TIME
        gold_dids: list[str] = []
        for fact in facts:
            did = get_or_add_doc(fact, ref_time)
            if fact_contains_answer(fact, ans):
                gold_dids.append(did)
        if not gold_dids:
            continue
        qid = f"q_l3_{seen_l3:04d}"
        queries.append(
            {
                "query_id": qid,
                "text": rec["question"],
                "ref_time": ref_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "subset": "L3",
            }
        )
        gold.append({"query_id": qid, "relevant_doc_ids": list(set(gold_dids))})
        seen_l3 += 1

    return docs, queries, gold


def main() -> None:
    docs, queries, gold = make_corpus_and_queries()
    print(f"Built: {len(docs)} docs, {len(queries)} queries, {len(gold)} gold mappings")

    # Subset breakdown
    n_l2 = sum(1 for q in queries if q["subset"] == "L2")
    n_l3 = sum(1 for q in queries if q["subset"] == "L3")
    print(f"  L2: {n_l2}, L3: {n_l3}")

    # Gold cardinality stats
    counts = [len(g["relevant_doc_ids"]) for g in gold]
    print(
        f"  Gold per query: min={min(counts)}, max={max(counts)}, mean={sum(counts) / len(counts):.2f}"
    )

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    with (OUT_DIR / "real_benchmark_docs.jsonl").open("w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with (OUT_DIR / "real_benchmark_queries.jsonl").open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    with (OUT_DIR / "real_benchmark_gold.jsonl").open("w") as f:
        for g in gold:
            f.write(json.dumps(g) + "\n")
    print(f"Wrote to {OUT_DIR}")


if __name__ == "__main__":
    main()
