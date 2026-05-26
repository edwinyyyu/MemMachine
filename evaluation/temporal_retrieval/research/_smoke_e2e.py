"""End-to-end smoke test: run the reference TemporalRetriever on the
ambiguous_year benchmark and verify all_recall@5 matches the v5.1
baseline of 1.000.

Run from `evaluation/`:
    uv run python -m temporal_retrieval.research._smoke_e2e
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json

from temporal_retrieval import Doc, TemporalRetriever

from ._common import DATA_DIR, make_embed_fn, make_rerank_fn, setup_env

setup_env()


def _doc_year(ref_time: str) -> int:
    s = ref_time.replace("Z", "+00:00")
    return _dt.datetime.fromisoformat(s).year


async def main():
    with open(DATA_DIR / "ambiguous_year_docs.jsonl") as f:
        docs_jsonl = [json.loads(line) for line in f]
    with open(DATA_DIR / "ambiguous_year_queries.jsonl") as f:
        queries = [json.loads(line) for line in f]
    with open(DATA_DIR / "ambiguous_year_gold.jsonl") as f:
        gold_rows = [json.loads(line) for line in f]
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl
    ]

    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()
    retriever = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)

    print(f"Indexing {len(docs)} docs...", flush=True)
    await retriever.index(docs)

    print(f"Running {len(queries)} queries...", flush=True)
    rows = []
    K = 5
    for q in queries:
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
        gold_set = gold.get(q["query_id"], set())
        topk = set(ranking[:K])
        n_in_topk = len(topk & gold_set)
        all_recall = n_in_topk / max(1, len(gold_set))
        # Compute year coverage too
        years_in_topk = {
            _doc_year(d["ref_time"])
            for d in docs_jsonl
            if d["doc_id"] in (topk & gold_set)
        }
        gold_years = {
            _doc_year(d["ref_time"]) for d in docs_jsonl if d["doc_id"] in gold_set
        }
        year_cov = len(years_in_topk) / max(1, len(gold_years))

        # Per-doc gold ranks for diagnostic
        gold_ranks = {}
        for g in gold_set:
            try:
                gold_ranks[g] = ranking.index(g) + 1
            except ValueError:
                gold_ranks[g] = None

        rows.append(
            {
                "qid": q["query_id"],
                "query": q["text"],
                "all_recall": all_recall,
                "year_coverage": year_cov,
                "gold_ranks": gold_ranks,
            }
        )
        rank_str = " ".join(
            f"{_doc_year(next(d['ref_time'] for d in docs_jsonl if d['doc_id'] == g))}:{rk if rk is not None else 'X'}"
            for g, rk in sorted(
                gold_ranks.items(),
                key=lambda kv: _doc_year(
                    next(d["ref_time"] for d in docs_jsonl if d["doc_id"] == kv[0])
                ),
            )
        )
        print(f"  {q['query_id']}: all_recall={all_recall:.3f}  ranks={rank_str}")

    n = len(rows)
    macro_all_recall = sum(r["all_recall"] for r in rows) / n
    macro_year_cov = sum(r["year_coverage"] for r in rows) / n
    print()
    print(f"=== ambiguous_year (n={n}) ===")
    print(f"all_recall@{K}    = {macro_all_recall:.3f}  (v5.1 baseline: 1.000)")
    print(f"year_coverage@{K} = {macro_year_cov:.3f}  (v5.1 baseline: 1.000)")

    if abs(macro_all_recall - 1.0) < 0.01:
        print("\n✓ MATCHES v5.1 baseline.")
    else:
        print(f"\n✗ DEVIATES from v5.1 (Δ = {macro_all_recall - 1.0:+.3f}).")
        print("Per-query breakdown:")
        for r in rows:
            if r["all_recall"] < 1.0:
                print(
                    f"  {r['qid']:14s} all_recall={r['all_recall']:.3f}  "
                    f"ranks={r['gold_ranks']}"
                )

    print()
    print("retriever.stats():")
    print(json.dumps(retriever.stats(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
