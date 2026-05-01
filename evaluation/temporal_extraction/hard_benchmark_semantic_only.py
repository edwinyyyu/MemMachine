"""Semantic-only sanity check on hard benchmark.

Runs text-embedding-3-small cosine retrieval on hard_bench_*. Targets:
- R@5 < 0.95 → benchmark is hard enough for the full pipeline run.
- R@5 < 0.80 → benchmark has genuine difficulty.

Writes results/hard_bench_semantic.json and prints a per-tier summary.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baselines import embed_all

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked, relevant):
    if not relevant:
        return float("nan")
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    if not relevant:
        return float("nan")
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def nanmean(xs):
    vs = [v for v in xs if not math.isnan(v)]
    return sum(vs) / len(vs) if vs else float("nan")


async def main():
    docs = load_jsonl(DATA_DIR / "hard_bench_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "hard_bench_queries.jsonl")
    gold_raw = load_jsonl(DATA_DIR / "hard_bench_gold.jsonl")
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_raw}
    subset_of_q = {q["query_id"]: q["subset"] for q in queries}

    print(f"Hard bench: {len(docs)} docs, {len(queries)} queries", flush=True)

    print("Embedding docs and queries...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    # Compute cosine similarities (rank only, R@k)
    by_subset_r1 = defaultdict(list)
    by_subset_r3 = defaultdict(list)
    by_subset_r5 = defaultdict(list)
    by_subset_r10 = defaultdict(list)
    by_subset_mrr = defaultdict(list)
    by_subset_ndcg = defaultdict(list)

    for q in queries:
        qid = q["query_id"]
        rel = gold.get(qid, set())
        if not rel:
            continue
        qv = q_embs[qid]
        qn = np.linalg.norm(qv) or 1e-9
        scored = []
        for did, v in doc_embs.items():
            vn = np.linalg.norm(v) or 1e-9
            scored.append((did, float(np.dot(qv, v) / (qn * vn))))
        ranked = [d for d, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
        sub = subset_of_q[qid]
        by_subset_r1[sub].append(recall_at_k(ranked, rel, 1))
        by_subset_r1["all"].append(recall_at_k(ranked, rel, 1))
        by_subset_r3[sub].append(recall_at_k(ranked, rel, 3))
        by_subset_r3["all"].append(recall_at_k(ranked, rel, 3))
        by_subset_r5[sub].append(recall_at_k(ranked, rel, 5))
        by_subset_r5["all"].append(recall_at_k(ranked, rel, 5))
        by_subset_r10[sub].append(recall_at_k(ranked, rel, 10))
        by_subset_r10["all"].append(recall_at_k(ranked, rel, 10))
        by_subset_mrr[sub].append(mrr(ranked, rel))
        by_subset_mrr["all"].append(mrr(ranked, rel))
        by_subset_ndcg[sub].append(ndcg_at_k(ranked, rel, 10))
        by_subset_ndcg["all"].append(ndcg_at_k(ranked, rel, 10))

    out = {}
    print(
        f"\n{'subset':<10} {'n':>4} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'NDCG':>6}"
    )
    for sub in ["all", "easy", "medium", "hard"]:
        if not by_subset_r5[sub]:
            continue
        m = {
            "n": len(by_subset_r5[sub]),
            "recall@1": nanmean(by_subset_r1[sub]),
            "recall@3": nanmean(by_subset_r3[sub]),
            "recall@5": nanmean(by_subset_r5[sub]),
            "recall@10": nanmean(by_subset_r10[sub]),
            "mrr": nanmean(by_subset_mrr[sub]),
            "ndcg@10": nanmean(by_subset_ndcg[sub]),
        }
        out[sub] = m
        print(
            f"{sub:<10} {m['n']:>4d} {m['recall@1']:>6.3f} {m['recall@3']:>6.3f} "
            f"{m['recall@5']:>6.3f} {m['recall@10']:>6.3f} {m['mrr']:>6.3f} {m['ndcg@10']:>6.3f}"
        )

    out_path = RESULTS_DIR / "hard_bench_semantic.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    overall_r5 = out["all"]["recall@5"]
    if overall_r5 >= 0.95:
        print(f"\nWARN: R@5 = {overall_r5:.3f} >= 0.95 — benchmark too easy")
    elif overall_r5 < 0.80:
        print(f"\nGOOD: R@5 = {overall_r5:.3f} < 0.80 — genuine difficulty")
    else:
        print(f"\nOK: R@5 = {overall_r5:.3f} ∈ [0.80, 0.95) — usable")


if __name__ == "__main__":
    asyncio.run(main())
