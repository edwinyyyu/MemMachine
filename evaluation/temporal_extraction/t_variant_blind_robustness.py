"""Cross-T-variant robustness: 5 T scoring variants × 5 shuffle seeds × blind_pair on hard_bench.

Tests:
  T_iv      = interval Jaccard alone (α=1, β=0, γ=0)
  T_axis    = multi-axis Bhattacharyya alone (β=1)
  T_tag     = tag Jaccard alone (γ=1)
  T_iv_axis = interval + axis (0.5/0.5)
  T_full    = current 0.5/0.35/0.15 blend

Reports mean ± std R@1 for each (T_variant, blind_pair) pair across seeds.
This is the proper apples-to-apples comparison the previous test was missing.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import v7l_ts_blind_eval as base
from rag_fusion import score_blend
from salience_eval import (
    AXES,
    DATA_DIR,
    AxisDistribution,
    axis_score_fn,
    build_memory,
    embed_all,
    interval_pair_best,
    parse_iso,
    rank_semantic,
    run_v2_extract,
    tag_score,
)
from v7l_ts_blind_eval import BlindJudge, metrics, run_pair

N_SEEDS = 5


T_VARIANTS = {
    "T_iv": (1.0, 0.0, 0.0),
    "T_axis": (0.0, 1.0, 0.0),
    "T_tag": (0.0, 0.0, 1.0),
    "T_iv_axis": (0.5, 0.5, 0.0),
    "T_full": (0.5, 0.35, 0.15),
}


def t_scores(q_mem, doc_mem, alpha, beta, gamma):
    qa = q_mem.get("axes_merged") or {}
    q_tags = q_mem.get("multi_tags") or set()
    q_ivs = q_mem.get("intervals") or []
    raw_iv = {
        did: interval_pair_best(q_ivs, b["intervals"]) for did, b in doc_mem.items()
    }
    max_iv = max(raw_iv.values()) if raw_iv else 0.0
    out = {}
    for did, b in doc_mem.items():
        iv_norm = raw_iv[did] / max_iv if max_iv > 0 else 0.0
        a_sc = axis_score_fn(qa, b["axes_merged"]) if beta > 0 else 0.0
        t_sc = tag_score(q_tags, b["multi_tags"]) if gamma > 0 else 0.0
        out[did] = alpha * iv_norm + beta * a_sc + gamma * t_sc
    return out


def rank_blend_ts(t, s, w_T):
    w_S = max(0.0, 1.0 - w_T)
    chans = {"T": t, "S": s}
    weights = {"T": w_T, "S": w_S}
    fused = score_blend(chans, weights, top_k_per=40, dispersion_cv_ref=0.20)
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


base.rank_blend_ts = rank_blend_ts


class CacheVariant:
    def __init__(self, t, s, doc_text):
        self.t, self.s, self.doc_text = t, s, doc_text
        self._cache = {}

    def get(self, w_T):
        key = round(max(0.0, min(1.0, w_T)), 3)
        if key not in self._cache:
            ranked = rank_blend_ts(self.t, self.s, key)
            top5_text = [
                self.doc_text.get(d, "")[: base.MAX_TEXT_LEN] for d in ranked[:5]
            ]
            self._cache[key] = (ranked, top5_text)
        return self._cache[key]


async def run_pair_with_seed_offset(
    qid, query_text, cache, judge, seed_offset, **kwargs
):
    orig = judge.pick_best

    async def patched(query, candidate_sets, rng_seed, ref_S=None, ref_T=None):
        return await orig(
            query,
            candidate_sets,
            rng_seed=(rng_seed ^ seed_offset) & 0xFFFFFFFF,
            ref_S=ref_S,
            ref_T=ref_T,
        )

    judge.pick_best = patched
    try:
        return await run_pair(qid, query_text, cache, judge, **kwargs)
    finally:
        judge.pick_best = orig


async def main():
    name = "hard_bench"
    docs = [json.loads(l) for l in open(DATA_DIR / "hard_bench_docs.jsonl")]
    queries = [json.loads(l) for l in open(DATA_DIR / "hard_bench_queries.jsonl")]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / "hard_bench_gold.jsonl")]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(
        f"=== {name}: {len(docs)} docs, {len(queries)} queries, {N_SEEDS} seeds × {len(T_VARIANTS)} T variants ==="
    )

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", "v7l-hard_bench")
    q_ext = await run_v2_extract(q_items, f"{name}-queries", "v7l-hard_bench")

    doc_mem = build_memory(doc_ext)
    q_mem = build_memory(q_ext)
    for d in docs:
        doc_mem.setdefault(
            d["doc_id"],
            {
                "intervals": [],
                "axes_merged": {
                    a: AxisDistribution(axis=a, values={}, informative=False)
                    for a in AXES
                },
                "multi_tags": set(),
            },
        )

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # Compute T scores per variant
    per_q_t_by_variant = {}
    for vname, (a, b, g) in T_VARIANTS.items():
        per_q_t_by_variant[vname] = {
            qid: t_scores(
                q_mem.get(
                    qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}
                ),
                doc_mem,
                a,
                b,
                g,
            )
            for qid in qids
        }

    # Fixed-best per variant
    print(f"\n{'T variant':12} {'fixed (w=0.2) R@1':>20}")
    for vname, per_q_t in per_q_t_by_variant.items():
        ranks = {qid: rank_blend_ts(per_q_t[qid], per_q_s[qid], 0.2) for qid in qids}
        m = metrics(ranks, gold, qids)
        print(f"{vname:12} {m['r@1']:>20.3f}")

    judge = BlindJudge()

    # blind_pair on each variant × seed
    all_results = {v: [] for v in T_VARIANTS}
    print("\n=== blind_pair across seeds ===")
    print(
        f"{'T variant':12} "
        + " ".join(f"seed{i}".rjust(7) for i in range(N_SEEDS))
        + f"   {'mean±std':>12} {'R@5 mean':>10}"
    )
    for vname in T_VARIANTS:
        seed_ms = []
        for seed_idx in range(N_SEEDS):
            seed_offset = (seed_idx + 1) * 0x9E3779B1
            opt_results = {}

            async def run_one(qid):
                cache = CacheVariant(
                    per_q_t_by_variant[vname][qid], per_q_s[qid], doc_text
                )
                r, _ = await run_pair_with_seed_offset(
                    qid,
                    q_text[qid],
                    cache,
                    judge,
                    seed_offset,
                    with_references=False,
                )
                opt_results[qid] = r

            await asyncio.gather(*(run_one(qid) for qid in qids))
            m = metrics(opt_results, gold, qids)
            seed_ms.append(m)
        all_results[vname] = seed_ms
        r1s = [m["r@1"] for m in seed_ms]
        r5s = [m["r@5"] for m in seed_ms]
        seed_str = " ".join(f"{r1:>7.3f}" for r1 in r1s)
        print(
            f"{vname:12} {seed_str}   {np.mean(r1s):.3f}±{np.std(r1s):.3f}  {np.mean(r5s):>10.3f}"
        )

    judge.save()

    print(f"\nLLM calls (incremental): {judge.calls}, failed: {judge.failed}")

    # Save
    out = {
        "fixed": {
            v: metrics(
                {
                    qid: rank_blend_ts(per_q_t_by_variant[v][qid], per_q_s[qid], 0.2)
                    for qid in qids
                },
                gold,
                qids,
            )
            for v in T_VARIANTS
        },
        "blind_pair_seeds": {
            v: [
                {"r@1": m["r@1"], "r@5": m["r@5"], "mrr": m["mrr"]}
                for m in all_results[v]
            ]
            for v in T_VARIANTS
        },
        "blind_pair_summary": {
            v: {
                "r1_mean": float(np.mean([m["r@1"] for m in all_results[v]])),
                "r1_std": float(np.std([m["r@1"] for m in all_results[v]])),
                "r5_mean": float(np.mean([m["r@5"] for m in all_results[v]])),
            }
            for v in T_VARIANTS
        },
    }
    out_path = ROOT / "results" / "t_variant_blind.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
