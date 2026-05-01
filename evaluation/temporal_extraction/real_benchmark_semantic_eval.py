"""Semantic-only baseline on the real-distribution TempReason benchmark.

Avoids LLM extraction (out of budget); uses text-embedding-3-small
cosine similarity only.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/temporal_extraction")
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
    docs = load_jsonl(DATA_DIR / "real_benchmark_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "real_benchmark_queries.jsonl")
    gold = {
        g["query_id"]: set(g["relevant_doc_ids"])
        for g in load_jsonl(DATA_DIR / "real_benchmark_gold.jsonl")
    }
    subset_of_q = {q["query_id"]: q["subset"] for q in queries}

    print(f"Real benchmark: {len(docs)} docs, {len(queries)} queries", flush=True)

    print("Embedding...", flush=True)
    t0 = time.time()
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    print(f"Embedded in {time.time() - t0:.1f}s", flush=True)

    doc_embs = np.array(doc_embs_arr)  # (N_d, dim)
    q_embs = np.array(q_embs_arr)  # (N_q, dim)

    # Normalize
    doc_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-9)
    q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)

    # Cosine sim matrix
    sims = q_norm @ doc_norm.T  # (N_q, N_d)
    rankings = np.argsort(-sims, axis=1)  # (N_q, N_d) descending

    doc_ids = [d["doc_id"] for d in docs]
    ranked_per_q: dict[str, list[str]] = {}
    for i, q in enumerate(queries):
        ranked_per_q[q["query_id"]] = [doc_ids[j] for j in rankings[i]]

    # Eval per subset
    L2_qids = {qid for qid, sub in subset_of_q.items() if sub == "L2"}
    L3_qids = {qid for qid, sub in subset_of_q.items() if sub == "L3"}
    all_qids = set(subset_of_q.keys())

    results: dict[str, dict[str, float]] = {}
    for sub_name, qids in [("all", all_qids), ("L2", L2_qids), ("L3", L3_qids)]:
        r5, r10, mr, nd = [], [], [], []
        for qid in qids:
            rel = gold.get(qid, set())
            if not rel:
                continue
            ranked = ranked_per_q.get(qid, [])
            r5.append(recall_at_k(ranked, rel, 5))
            r10.append(recall_at_k(ranked, rel, 10))
            mr.append(mrr(ranked, rel))
            nd.append(ndcg_at_k(ranked, rel, 10))
        results[sub_name] = {
            "n": len(qids),
            "recall@5": nanmean(r5),
            "recall@10": nanmean(r10),
            "mrr": nanmean(mr),
            "ndcg@10": nanmean(nd),
        }

    # Failure analysis (semantic-only)
    failures = []
    for q in queries:
        qid = q["query_id"]
        rel = gold.get(qid, set())
        ranked = ranked_per_q.get(qid, [])
        if not (set(ranked[:5]) & rel):
            gold_text = next((d["text"] for d in docs if d["doc_id"] in rel), "<?>")
            top5_texts = [
                next((d["text"] for d in docs if d["doc_id"] == did), "<?>")
                for did in ranked[:5]
            ]
            failures.append(
                {
                    "qid": qid,
                    "subset": q["subset"],
                    "query": q["text"],
                    "ref_time": q["ref_time"],
                    "gold_doc_id": list(rel)[0] if rel else None,
                    "gold_doc_text": gold_text,
                    "top5_doc_ids": ranked[:5],
                    "top5_doc_texts": top5_texts,
                }
            )

    out = {
        "benchmark": {
            "name": "TempReason-derived (test L2 + L3)",
            "n_docs": len(docs),
            "n_queries": len(queries),
            "n_L2": len(L2_qids),
            "n_L3": len(L3_qids),
            "source": "tonytan48/TempReason test_l2.json + test_l3.json",
        },
        "system": "SEMANTIC-ONLY (text-embedding-3-small cosine)",
        "metrics": results,
        "failures": {
            "n": len(failures),
            "by_subset": {
                "L2": sum(1 for f in failures if f["subset"] == "L2"),
                "L3": sum(1 for f in failures if f["subset"] == "L3"),
            },
            "samples": failures[:10],
        },
        "wall_seconds": time.time() - t0,
    }
    (RESULTS_DIR / "real_benchmark_semantic.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    print(
        f"\nFailures: {out['failures']['n']} (L2={out['failures']['by_subset']['L2']}, L3={out['failures']['by_subset']['L3']})",
        flush=True,
    )
    print(f"Wrote {RESULTS_DIR / 'real_benchmark_semantic.json'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
