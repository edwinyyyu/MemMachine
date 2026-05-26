"""Diagnose specific engagement_disjoint failures.

Prints, for a small set of queries: the V1 plan, the V7 query_refs
(after range composition + incompat split), and the doc refs +
final_score for each gold + distractor doc.
"""
from __future__ import annotations

import asyncio
import json

from temporal_retrieval_min.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval_min.planner import QueryPlanner
from temporal_retrieval_min.schema import parse_iso
from temporal_retrieval_v7.adapters import (
    extractor_to_doc_refs,
    plan_to_query_refs,
)
from temporal_retrieval_v7.scoring import final_score, pair_overlap
from temporal_retrieval_v7.time_range import NEG_INF, POS_INF, measure

from temporal_retrieval.research._common import DATA_DIR, setup_env

setup_env()

PROBE_QIDS = [
    "ed_q_2disj",
    "ed_q_colloq_2020_2024",
    "ed_q_engage_summer",
    "ed_q_engage_2023",
]


def _fmt_us(t: int) -> str:
    if t <= NEG_INF + 1:
        return "-∞"
    if t >= POS_INF - 1:
        return "+∞"
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t / 1_000_000, tz=timezone.utc).strftime("%Y-%m-%d")


def _fmt_range(r) -> str:
    if not r.intervals:
        return "∅"
    return " ∪ ".join(f"[{_fmt_us(iv.earliest_us)}, {_fmt_us(iv.latest_us)})"
                      for iv in r.intervals)


async def main():
    with open(DATA_DIR / "engagement_disjoint_docs.jsonl") as f:
        docs_jsonl = [json.loads(line) for line in f]
    with open(DATA_DIR / "engagement_disjoint_queries.jsonl") as f:
        queries = {q["query_id"]: q for q in (json.loads(line) for line in f)}
    with open(DATA_DIR / "engagement_disjoint_gold.jsonl") as f:
        gold = {g["query_id"]: set(g["relevant_doc_ids"])
                for g in (json.loads(line) for line in f)}

    extractor = TemporalExtractorV3_3()
    planner = QueryPlanner()

    # Index doc refs once
    doc_refs = {}
    doc_texts = {}
    print("Extracting doc envelopes...", flush=True)
    for d in docs_jsonl:
        ivs = await extractor.extract(d["text"], parse_iso(d["ref_time"]))
        doc_refs[d["doc_id"]] = extractor_to_doc_refs(ivs)
        doc_texts[d["doc_id"]] = d["text"]
    extractor.save_caches()

    for qid in PROBE_QIDS:
        q = queries[qid]
        print(f"\n=== {qid} ===")
        print(f"query: {q['text']}")
        print(f"ref_time: {q['ref_time']}")

        plan = await planner.plan(q["text"], q["ref_time"])
        print(f"\nplan: expr={plan.expr}, extremum={plan.extremum}")

        # Resolve leaves
        leaves = [leaf for clause in plan.expr for leaf in clause]
        anchors_by_phrase = {}
        if leaves:
            rt = parse_iso(q["ref_time"])
            for leaf in leaves:
                ivs = await extractor.extract(leaf.phrase, rt)
                anchors_by_phrase[(leaf.phrase, leaf.relation)] = ivs

        def resolver(leaf):
            return anchors_by_phrase.get((leaf.phrase, leaf.relation), [])

        query_refs = plan_to_query_refs(plan, resolver)
        print(f"\n{len(query_refs)} query_refs:")
        for i, qr in enumerate(query_refs):
            print(f"  [{i}]: {_fmt_range(qr)}")

        gset = gold.get(qid, set())
        print(f"\ngold: {gset}")
        print("\ndoc scores (gold marked *):")
        scored = []
        for did, drefs in doc_refs.items():
            s = final_score(query_refs, drefs)
            scored.append((did, s, drefs))
        scored.sort(key=lambda x: -x[1])
        for did, s, drefs in scored:
            mark = "*" if did in gset else " "
            print(f"  {mark} {s:.3f}  {did:30s}  refs={_fmt_range(drefs[0]) if drefs else '∅'}{'...' if len(drefs)>1 else ''}")
            for dr in drefs[1:]:
                print(f"                              and  {_fmt_range(dr)}")
            # Also print text
            print(f"      text: {doc_texts[did][:90]}")
        extractor.save_caches()


if __name__ == "__main__":
    asyncio.run(main())
