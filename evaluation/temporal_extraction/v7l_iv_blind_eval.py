"""Combine T-channel ablation finding with blind_pair weight tuning.

Tests on hard_bench (only benchmark with headroom):
  - pure_S baseline
  - T_iv fixed-w_T sweep (find best fixed)
  - T_iv + blind_pair LLM tuning (does weight tuning still help on simpler T?)
  - For comparison: T_full + blind_pair (the previous winner at 0.827)
  - And T_axis fixed-w_T (best single component on hard_bench)

If T_iv + blind_pair > T_full + blind_pair, stacked win.
If T_iv + blind_pair ≤ T_iv fixed best, weight tuning doesn't help when T is clean.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

# Reuse blind eval infrastructure (BlindJudge, run_pair) but inject custom T scores
import v7l_ts_blind_eval as base
from rag_fusion import score_blend
from salience_eval import (  # type: ignore
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


def t_scores_iv(q_mem, doc_mem):
    """T_iv: interval Jaccard alone. α=1, β=0, γ=0."""
    q_ivs = q_mem.get("intervals") or []
    raw_iv = {
        did: interval_pair_best(q_ivs, b["intervals"]) for did, b in doc_mem.items()
    }
    max_iv = max(raw_iv.values()) if raw_iv else 0.0
    return {did: (v / max_iv if max_iv > 0 else 0.0) for did, v in raw_iv.items()}


def t_scores_axis(q_mem, doc_mem):
    """T_axis: multi-axis Bhattacharyya alone. α=0, β=1, γ=0."""
    qa = q_mem.get("axes_merged") or {}
    return {did: axis_score_fn(qa, b["axes_merged"]) for did, b in doc_mem.items()}


def t_scores_full(q_mem, doc_mem):
    """T_full: original 0.5/0.35/0.15 blend."""
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
        a_sc = axis_score_fn(qa, b["axes_merged"])
        t_sc = tag_score(q_tags, b["multi_tags"])
        out[did] = 0.5 * iv_norm + 0.35 * a_sc + 0.15 * t_sc
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


# Patch base module's rank_blend_ts so RetrievalCache uses our local version
# (it's the same; just being explicit)
base.rank_blend_ts = rank_blend_ts


WEIGHT_GRID = [round(i * 0.1, 1) for i in range(11)]


async def main():
    name = "hard_bench"
    docs = [json.loads(l) for l in open(DATA_DIR / "hard_bench_docs.jsonl")]
    queries = [json.loads(l) for l in open(DATA_DIR / "hard_bench_queries.jsonl")]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / "hard_bench_gold.jsonl")]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"=== {name}: {len(docs)} docs, {len(queries)} queries ===")

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

    # Compute T scores under each variant
    per_q_t_iv = {}
    per_q_t_axis = {}
    per_q_t_full = {}
    for q in queries:
        qid = q["query_id"]
        m = q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()})
        per_q_t_iv[qid] = t_scores_iv(m, doc_mem)
        per_q_t_axis[qid] = t_scores_axis(m, doc_mem)
        per_q_t_full[qid] = t_scores_full(m, doc_mem)

    # Sweep best-fixed-w_T per T variant
    print(f"\n{'T variant':10} {'best w_T':>9} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    fixed_results = {}
    for vname, per_q_t in (
        ("T_iv", per_q_t_iv),
        ("T_axis", per_q_t_axis),
        ("T_full", per_q_t_full),
    ):
        best_m = None
        best_w = None
        for w_T in WEIGHT_GRID:
            ranks = {
                qid: rank_blend_ts(per_q_t[qid], per_q_s[qid], w_T) for qid in qids
            }
            m = metrics(ranks, gold, qids)
            if best_m is None or (m["r@5"], m["r@1"], m["mrr"]) > (
                best_m["r@5"],
                best_m["r@1"],
                best_m["mrr"],
            ):
                best_m = m
                best_w = w_T
        fixed_results[vname] = (best_w, best_m)
        print(
            f"{vname:10} {best_w:>9.1f} {best_m['r@1']:>6.3f} {best_m['r@5']:>6.3f} {best_m['mrr']:>6.3f}"
        )

    pure_S_ranks = {qid: rank_blend_ts({}, per_q_s[qid], 0.0) for qid in qids}
    pure_S_m = metrics(pure_S_ranks, gold, qids)
    print(
        f"{'pure_S':10} {'-':>9} {pure_S_m['r@1']:>6.3f} {pure_S_m['r@5']:>6.3f} {pure_S_m['mrr']:>6.3f}"
    )

    # Now run blind_pair on each T variant
    judge = BlindJudge()
    print("\n=== blind_pair on each T variant ===")
    print(f"{'T variant':10} {'R@1':>6} {'R@5':>6} {'MRR':>6} {'avg_final_w_T':>15}")

    # Custom RetrievalCache that uses a specific T variant
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

    blind_results = {}
    for vname, per_q_t in (
        ("T_iv", per_q_t_iv),
        ("T_axis", per_q_t_axis),
        ("T_full", per_q_t_full),
    ):
        opt_results = {}
        opt_diag = {}

        async def run_one(qid, vn=vname, ptt=per_q_t):
            cache = CacheVariant(ptt[qid], per_q_s[qid], doc_text)
            r, diag = await run_pair(
                qid, q_text[qid], cache, judge, with_references=False
            )
            opt_results[qid] = r
            opt_diag[qid] = diag

        await asyncio.gather(*(run_one(qid) for qid in qids))
        m = metrics(opt_results, gold, qids)
        finals = [d["final_w_T"] for d in opt_diag.values()]
        blind_results[vname] = m
        print(
            f"{vname:10} {m['r@1']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f} {np.mean(finals):>15.3f}"
        )

    judge.save()

    print("\n=== SUMMARY ===")
    print(f"{'Variant':30} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    print(
        f"{'pure_S':30} {pure_S_m['r@1']:>6.3f} {pure_S_m['r@5']:>6.3f} {pure_S_m['mrr']:>6.3f}"
    )
    for vname, (w, m) in fixed_results.items():
        print(
            f"{vname + f' fixed (w={w})':30} {m['r@1']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}"
        )
    for vname, m in blind_results.items():
        print(
            f"{vname + ' + blind_pair':30} {m['r@1']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}"
        )

    out = {
        "pure_S": pure_S_m,
        "fixed": {v: {"best_w_T": w, **m} for v, (w, m) in fixed_results.items()},
        "blind_pair": blind_results,
    }
    out_path = ROOT / "results" / "iv_blind.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
