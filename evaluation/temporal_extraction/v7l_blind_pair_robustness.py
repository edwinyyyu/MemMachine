"""Robustness check for blind_pair on hard_bench.

Re-runs the winning blind_pair design with 5 different shuffle seeds. If the
+0.053 R@1 win over fixed-0.2 is real signal it should hold up across seeds;
if it's lucky shuffle, R@1 will swing widely.

Reports: mean ± std of R@1, R@5, MRR across seeds. Flags whether the gain
is reliably above the fixed-0.2 baseline.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

# Reuse the winning eval's core implementation
from salience_eval import (
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from salience_eval import (
    rank_t as rank_multi_axis_t,
)
from v7l_ts_blind_eval import (
    BlindJudge,
    RetrievalCache,
    metrics,
    rank_blend_ts,
    run_pair,
)

N_SEEDS = 5


async def run_pair_with_seed_offset(
    qid, query_text, cache, judge, seed_offset, **kwargs
):
    """Wrapper that perturbs the seed by mixing in seed_offset."""
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
    # Load hard_bench
    name = "hard_bench"
    docs_path = "hard_bench_docs.jsonl"
    queries_path = "hard_bench_queries.jsonl"
    gold_path = "hard_bench_gold.jsonl"
    cache_doc = "v7l-hard_bench"
    cache_q = "v7l-hard_bench"

    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(
        f"=== {name} robustness: {len(docs)} docs, {len(queries)} queries, {N_SEEDS} seeds ==="
    )

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_doc)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_q)

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

    per_q_t, per_q_s = {}, {}
    for q in queries:
        qid = q["query_id"]
        per_q_t[qid] = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        per_q_s[qid] = rank_semantic(qid, q_embs, doc_embs)

    qids = [q["query_id"] for q in queries]

    # Fixed-0.2 baseline
    fixed_02 = {qid: rank_blend_ts(per_q_t[qid], per_q_s[qid], w_T=0.2) for qid in qids}
    base_m = metrics(fixed_02, gold, qids)
    print(
        f"\nFixed (w=0.2)            R@1={base_m['r@1']:.3f}  R@5={base_m['r@5']:.3f}  MRR={base_m['mrr']:.3f}"
    )

    judge = BlindJudge()

    seed_results = []
    for seed_offset in range(N_SEEDS):
        seed_offset_val = (
            seed_offset + 1
        ) * 0x9E3779B1  # Avoid offset=0 to actually perturb
        # Re-run blind_pair with this offset
        opt_results: dict[str, list[str]] = {}
        opt_diag: dict[str, dict] = {}

        async def run_one(qid):
            cache = RetrievalCache(per_q_t[qid], per_q_s[qid], doc_text)
            r, diag = await run_pair_with_seed_offset(
                qid,
                q_text[qid],
                cache,
                judge,
                seed_offset_val,
                with_references=False,
            )
            opt_results[qid] = r
            opt_diag[qid] = diag

        await asyncio.gather(*(run_one(qid) for qid in qids))
        m = metrics(opt_results, gold, qids)
        seed_results.append(m)
        finals = [d["final_w_T"] for d in opt_diag.values()]
        print(
            f"Seed {seed_offset}                   R@1={m['r@1']:.3f}  R@5={m['r@5']:.3f}  "
            f"MRR={m['mrr']:.3f}  avg_final_w_T={np.mean(finals):.3f}"
        )

    judge.save()

    print(f"\nLLM calls (incremental): {judge.calls}, failed: {judge.failed}")

    r1s = [m["r@1"] for m in seed_results]
    r5s = [m["r@5"] for m in seed_results]
    mrrs = [m["mrr"] for m in seed_results]

    print(f"\n=== ROBUSTNESS SUMMARY ({N_SEEDS} seeds) ===")
    print(
        f"Fixed (w=0.2)         R@1={base_m['r@1']:.3f}            R@5={base_m['r@5']:.3f}            MRR={base_m['mrr']:.3f}"
    )
    print(
        f"blind_pair  mean±std  R@1={np.mean(r1s):.3f}±{np.std(r1s):.3f}  "
        f"R@5={np.mean(r5s):.3f}±{np.std(r5s):.3f}  MRR={np.mean(mrrs):.3f}±{np.std(mrrs):.3f}"
    )
    print(
        f"            min       R@1={np.min(r1s):.3f}            R@5={np.min(r5s):.3f}            MRR={np.min(mrrs):.3f}"
    )
    print(
        f"            max       R@1={np.max(r1s):.3f}            R@5={np.max(r5s):.3f}            MRR={np.max(mrrs):.3f}"
    )
    n_above = sum(1 for x in r1s if x > base_m["r@1"])
    n_at = sum(1 for x in r1s if abs(x - base_m["r@1"]) < 1e-9)
    n_below = sum(1 for x in r1s if x < base_m["r@1"])
    print(
        f"R@1 vs fixed: above={n_above}/{N_SEEDS}, equal={n_at}/{N_SEEDS}, below={n_below}/{N_SEEDS}"
    )

    out_path = ROOT / "results" / "blind_pair_robustness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "fixed_02": base_m,
                "seeds": [
                    {"seed_idx": i, "metrics": m} for i, m in enumerate(seed_results)
                ],
                "summary": {
                    "r1_mean": float(np.mean(r1s)),
                    "r1_std": float(np.std(r1s)),
                    "r5_mean": float(np.mean(r5s)),
                    "r5_std": float(np.std(r5s)),
                    "mrr_mean": float(np.mean(mrrs)),
                    "mrr_std": float(np.std(mrrs)),
                    "n_above": n_above,
                    "n_below": n_below,
                },
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
