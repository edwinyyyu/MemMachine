"""Failure-mode diagnostic for 4 underperforming benches.

For each failed query (gold not at rank 1), dump:
  - query text, ref_time
  - planner output (TimeRange list, extremum flags)
  - gold doc text + ref_time + extractor TimeRanges
  - top-3 retrieved docs (text snippet, rank, score)
  - position of gold in pool / in final rank

Goal: identify whether failures are
  (a) extraction-side (planner/extractor miss the temporal cue)
  (b) retrieval-side (pool doesn't include gold)
  (c) scoring-side (gold in pool but ranked below distractors)
  (d) bench artifact (gold ambiguous / unidentifiable)

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._failure_mode_diag
"""
from __future__ import annotations

import asyncio

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()

BENCHES = ["causal_relative", "allen", "speculative_anchors", "edge_era_refs"]


def _snippet(text: str, n: int = 180) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > n:
        return text[:n] + "…"
    return text


async def diag_bench(bench: str, embed_fn, rerank_fn) -> None:
    docs_jsonl, queries, gold = load_bench(bench)
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    doc_lookup = {d["doc_id"]: d for d in docs_jsonl}

    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)

    print(f"\n{'=' * 78}", flush=True)
    print(f"=== {bench}  ({len(queries)} queries)", flush=True)
    print(f"{'=' * 78}", flush=True)

    fail_count = 0
    for q in queries:
        qid = q["query_id"]
        qtext = q["text"]
        gset = gold.get(qid, set())
        # Get top-10 with score breakdown
        results = await vd.query(qtext, q["ref_time"], k=10)
        ranked_ids = [r.doc_id for r in results]
        # Pass if any gold in top-1
        hit_at_1 = len(gset.intersection(ranked_ids[:1])) > 0
        if hit_at_1:
            continue
        fail_count += 1

        # Get planner output
        plan = await vd._planner.plan(qtext, q["ref_time"])

        # Find gold rank in returned list
        gold_ranks = []
        for gid in gset:
            if gid in ranked_ids:
                gold_ranks.append((gid, ranked_ids.index(gid) + 1))
            else:
                gold_ranks.append((gid, None))

        print(f"\n--- FAIL: {qid} ---", flush=True)
        print(f"  Q: {qtext}  (ref_time={q['ref_time']})", flush=True)
        print(f"  PLAN: latest={plan.latest_intent} earliest={plan.earliest_intent}",
              flush=True)
        for i, tr in enumerate(plan.targets):
            print(f"    target[{i}] = {tr}", flush=True)

        print(f"  GOLD ({len(gset)}):", flush=True)
        for gid, rank in gold_ranks:
            gd = doc_lookup.get(gid)
            if gd:
                gd_anchors = vd._doc_anchors.get(gid, [])
                print(f"    {gid} (rank={rank}, ref_time={gd['ref_time']})", flush=True)
                print(f"      anchors[{len(gd_anchors)}]: "
                      f"{[str(a)[:80] for a in gd_anchors[:3]]}", flush=True)
                print(f"      text: {_snippet(gd['text'])}", flush=True)

        print(f"  TOP-3 RETRIEVED:", flush=True)
        for rank, r in enumerate(results[:3], 1):
            d = doc_lookup.get(r.doc_id)
            tag = "✓" if r.doc_id in gset else " "
            print(f"    {tag} #{rank} {r.doc_id} score={r.score:.3f} "
                  f"(rerank={r.rerank:.3f} match={r.match:.3f}) ref={d['ref_time'] if d else '?'}",
                  flush=True)
            if d:
                print(f"        {_snippet(d['text'], 140)}", flush=True)

    m = metrics({q["query_id"]: [r.doc_id for r in
                                  await vd.query(q["text"], q["ref_time"], k=10)]
                 for q in queries}, gold)
    print(f"\n  {bench}: R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  "
          f"fails={fail_count}/{len(queries)}", flush=True)


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    for bench in BENCHES:
        await diag_bench(bench, embed_fn, rerank_fn)


if __name__ == "__main__":
    asyncio.run(main())
