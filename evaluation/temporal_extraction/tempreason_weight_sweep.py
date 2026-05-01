"""Sweep fusion weights on TempReason small.

Loads cached extractions, computes T and S rankings once, then sweeps over
(w_T, w_S, w_L) outer weights + (α, β, γ) inner multi-axis weights.
Reports R@1, R@3, MRR per configuration.

Everything is pure math on cached artifacts — no LLM calls.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Apply LLM-patch (harmless; ensures no re-extraction hangs if any cache miss)
import extractor_common

_orig_call = extractor_common.BaseImprovedExtractor._call


async def _patched_call(self, *args, **kwargs):
    original_create = self.client.chat.completions.create

    async def patched_create(**call_kwargs):
        model = call_kwargs.get("model", "")
        if isinstance(model, str) and model.startswith("gpt-5"):
            call_kwargs["reasoning_effort"] = "minimal"
        return await original_create(**call_kwargs)

    self.client.chat.completions.create = patched_create
    try:
        return await _orig_call(self, *args, **kwargs)
    finally:
        self.client.chat.completions.create = original_create


extractor_common.BaseImprovedExtractor._call = _patched_call

# Re-use the pipeline eval's helpers
from datetime import datetime

import tempreason_pipeline_eval as tre
from openai import AsyncOpenAI

DATA_DIR = ROOT / "data"


async def main():
    # 1. Load data
    docs = tre.load_jsonl(DATA_DIR / "real_benchmark_small_docs.jsonl")
    queries = tre.load_jsonl(DATA_DIR / "real_benchmark_small_queries.jsonl")
    gold_rows = tre.load_jsonl(DATA_DIR / "real_benchmark_small_gold.jsonl")
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    doc_items = [
        (
            d["doc_id"],
            d["text"],
            datetime.fromisoformat(d["ref_time"].replace("Z", "+00:00")),
        )
        for d in docs
    ]
    query_items = [
        (
            q["query_id"],
            q["text"],
            datetime.fromisoformat(q["ref_time"].replace("Z", "+00:00")),
        )
        for q in queries
    ]

    # 2. Run extraction (cached) via pipeline's helpers
    print("Extracting (cached)...")
    doc_extracted, _doc_usage, _doc_to, _doc_err = await tre.run_v2_extract(
        doc_items, "docs", "tempreason_v2"
    )
    q_extracted, _q_usage, _q_to, _q_err = await tre.run_v2_extract(
        query_items, "queries", "tempreason_v2"
    )

    # 3. Build memory + embeddings
    doc_mem = tre.build_memory(doc_extracted)

    # Embeddings (cached)
    client = AsyncOpenAI()
    from baselines import embed_all

    doc_texts = [d["text"] for d in docs]
    query_texts = [q["text"] for q in queries]
    doc_embs = await embed_all(doc_texts)
    q_embs = await embed_all(query_texts)
    doc_emb_by_id = {d["doc_id"]: e for d, e in zip(docs, doc_embs)}
    q_emb_by_id = {q["query_id"]: e for q, e in zip(queries, q_embs)}

    # 4. Compute per-query T and S rankings once
    q_mem = tre.build_memory(q_extracted)
    query_ids = [q["query_id"] for q in queries]
    doc_ids = [d["doc_id"] for d in docs]

    subsets = defaultdict(list)
    for q in queries:
        subsets["all"].append(q["query_id"])
        subsets[q.get("subset", "all")].append(q["query_id"])

    # Compute T scores with default multi-axis weights (α=0.5, β=0.35, γ=0.15)
    # The multi-axis function uses these internally; we'll also sweep these
    # via the rank_multi_axis_t helper.
    async def t_scores(alpha=0.5, beta=0.35, gamma=0.15):
        out = {}
        for qid in query_ids:
            scores = tre.rank_multi_axis_t(
                q_mem.get(qid, {}), doc_mem, alpha=alpha, beta=beta, gamma=gamma
            )
            out[qid] = scores
        return out

    # S scores (semantic cosine)
    s_scores = {}
    for qid in query_ids:
        qe = q_emb_by_id[qid]
        scores = {}
        for did, de in doc_emb_by_id.items():
            import numpy as np

            cos = float(
                np.dot(qe, de) / (np.linalg.norm(qe) * np.linalg.norm(de) + 1e-9)
            )
            scores[did] = cos
        s_scores[qid] = scores

    # Lattice scores
    from lattice_store import LatticeStore

    lat_db = ROOT / "cache" / "tempreason_v2" / "lattice_sweep.sqlite"
    if lat_db.exists():
        lat_db.unlink()
    lat_store = LatticeStore(str(lat_db))
    tre.ingest_lattice(lat_store, doc_extracted)
    l_scores_raw = tre.retrieve_lattice_scores(lat_store, q_extracted, query_ids)

    # 5. Rank by any weighted blend
    def rank_weighted(qid, t, s, l, w_t, w_s, w_l):
        # Min-max normalize each within top-K-ish of docs
        def normalize(sc):
            if not sc:
                return {}
            vs = list(sc.values())
            lo, hi = min(vs), max(vs)
            if hi - lo < 1e-12:
                return dict.fromkeys(sc, 0.0)
            return {k: (v - lo) / (hi - lo) for k, v in sc.items()}

        tn = normalize(t.get(qid, {}))
        sn = normalize(s.get(qid, {}))
        ln = normalize(l.get(qid, {}))
        combined = {}
        for did in doc_ids:
            combined[did] = (
                w_t * tn.get(did, 0.0) + w_s * sn.get(did, 0.0) + w_l * ln.get(did, 0.0)
            )
        return sorted(combined, key=combined.get, reverse=True)

    def eval_variant(rankings, qids):
        r1 = r3 = r5 = 0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        n = 0
        for qid in qids:
            g = set(gold.get(qid, []))
            if not g:
                continue
            r = rankings.get(qid, [])
            hit = None
            for i, d in enumerate(r):
                if d in g:
                    hit = i + 1
                    break
            if hit:
                if hit <= 1:
                    r1 += 1
                if hit <= 3:
                    r3 += 1
                if hit <= 5:
                    r5 += 1
                mrr_sum += 1.0 / hit
                dcg = sum(
                    1.0 / math.log2(i + 2) for i, d in enumerate(r[:10]) if d in g
                )
                ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(g), 10)))
                ndcg_sum += dcg / ideal if ideal else 0.0
            n += 1
        return {
            "n": n,
            "r@1": r1 / n,
            "r@3": r3 / n,
            "r@5": r5 / n,
            "mrr": mrr_sum / n,
            "ndcg@10": ndcg_sum / n,
        }

    # 6. Sweep outer weights (T, S, L) — keep A, E off for TempReason
    print("Sweeping outer fusion weights...")
    t_default = await t_scores()
    # First: also sweep inner multi-axis weights (α for interval, β for axis, γ for tag)
    print("Inner multi-axis weights (α/β/γ) at outer T=0.5, S=0.5, L=0:")
    inner_results = []
    for alpha, beta, gamma in [
        (1.0, 0.0, 0.0),
        (0.8, 0.15, 0.05),
        (0.6, 0.3, 0.1),
        (0.5, 0.35, 0.15),  # current default
        (0.4, 0.4, 0.2),
        (0.3, 0.5, 0.2),
        (0.5, 0.5, 0.0),
        (0.7, 0.2, 0.1),
    ]:
        t_sc = await t_scores(alpha, beta, gamma)
        rankings = {
            qid: rank_weighted(qid, t_sc, s_scores, l_scores_raw, 0.5, 0.5, 0)
            for qid in query_ids
        }
        res = eval_variant(rankings, subsets["L2"])
        inner_results.append({"alpha": alpha, "beta": beta, "gamma": gamma, **res})
        print(
            f"  α={alpha:.2f} β={beta:.2f} γ={gamma:.2f} L2 R@1={res['r@1']:.3f} MRR={res['mrr']:.3f}"
        )

    # Use best inner weights for outer sweep
    best_inner = max(inner_results, key=lambda r: (r["r@1"], r["mrr"]))
    print(
        f"\nBest inner: α={best_inner['alpha']} β={best_inner['beta']} γ={best_inner['gamma']}"
    )
    t_best = await t_scores(
        best_inner["alpha"], best_inner["beta"], best_inner["gamma"]
    )

    print("\nOuter fusion weights (T, S, L) at best-inner:")
    outer_results = []
    for w_t, w_s, w_l in [
        (0.0, 1.0, 0.0),  # semantic only
        (1.0, 0.0, 0.0),  # T only
        (0.3, 0.7, 0.0),
        (0.4, 0.6, 0.0),
        (0.5, 0.5, 0.0),
        (0.6, 0.4, 0.0),
        (0.7, 0.3, 0.0),
        (0.4, 0.4, 0.2),  # V7L
        (0.45, 0.45, 0.1),
        (0.3, 0.6, 0.1),
        (0.5, 0.4, 0.1),
        (0.3, 0.5, 0.2),
        (0.6, 0.3, 0.1),
    ]:
        rankings_l2 = {
            qid: rank_weighted(qid, t_best, s_scores, l_scores_raw, w_t, w_s, w_l)
            for qid in query_ids
        }
        res_l2 = eval_variant(rankings_l2, subsets["L2"])
        res_all = eval_variant(rankings_l2, subsets["all"])
        outer_results.append(
            {
                "w_t": w_t,
                "w_s": w_s,
                "w_l": w_l,
                "L2_r@1": res_l2["r@1"],
                "L2_mrr": res_l2["mrr"],
                "all_r@1": res_all["r@1"],
                "all_mrr": res_all["mrr"],
            }
        )
        print(
            f"  T={w_t:.2f} S={w_s:.2f} L={w_l:.2f} "
            f"L2 R@1={res_l2['r@1']:.3f} MRR={res_l2['mrr']:.3f} | "
            f"all R@1={res_all['r@1']:.3f} MRR={res_all['mrr']:.3f}"
        )

    best_outer = max(
        outer_results, key=lambda r: (r["L2_r@1"], r["L2_mrr"], r["all_r@1"])
    )
    print(
        f"\nBest outer: T={best_outer['w_t']} S={best_outer['w_s']} L={best_outer['w_l']}"
        f"  L2 R@1={best_outer['L2_r@1']:.3f}"
    )

    out = {
        "inner_results": inner_results,
        "outer_results": outer_results,
        "best_inner": best_inner,
        "best_outer": best_outer,
    }
    with open(ROOT / "results" / "tempreason_weight_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results/tempreason_weight_sweep.json")


if __name__ == "__main__":
    asyncio.run(main())
