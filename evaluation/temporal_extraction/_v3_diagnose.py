"""Failure diagnostic: classify each non-rank-1 query under R-S + B-RMX.

For each failed query, determine:
  semantic_miss:    gold doc not in the rerank pool (semantic top-K missed it)
  rerank_miss:      gold in pool, rerank ranked gold below #1, no temporal
                    signal to fix it (mask permits, extremum doesn't apply
                    or doesn't lift gold)
  temporal_miss:    gold in pool, mask/extremum could have lifted it, but
                    the signals didn't align (e.g., wrong direction extracted)

Outputs per-bench failure-class counts.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

for _k in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "FTP_PROXY",
    "ftp_proxy",
):
    os.environ.pop(_k, None)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")

from composition_eval_v3 import (
    EXTREMUM_MULT_ALPHA,
    constraint_factor_for_doc,
    hit_rank,
    linear_recency_scores,
    normalize_dict,
    normalize_rerank_full,
    rank_from_scores,
)
from force_pick_optimizers_eval import rerank_topk
from negation import excluded_containment
from query_planner_v2 import QueryPlan, QueryPlanner
from salience_eval import (
    DATA_DIR,
    embed_all,
    flatten_intervals,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from schema import to_us

POOL_CAP = 10


async def diagnose_bench(
    name,
    docs_path,
    queries_path,
    gold_path,
    cache_label,
    reranker,
    planner: QueryPlanner,
):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===", flush=True)

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_label)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_label)

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    all_dids = list(doc_ref_us.keys())
    n_total = len(all_dids)

    plan_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    plans = await planner.plan_many(plan_items)

    win_items = []
    for q in queries:
        qid = q["query_id"]
        ref = parse_iso(q["ref_time"])
        plan = plans.get(qid)
        if not plan:
            continue
        for i, c in enumerate(plan.constraints):
            tag = f"{qid}__c{i}"
            win_items.append((tag, c.phrase, ref))
    win_ext = (
        await run_v2_extract(
            win_items,
            f"{name}-constraints",
            f"{cache_label}-constraints",
        )
        if win_items
        else {}
    )

    doc_ivs_flat = {}
    for did, tes in doc_ext.items():
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        doc_ivs_flat[did] = ivs
    for d in docs:
        doc_ivs_flat.setdefault(d["doc_id"], [])

    doc_bundles_for_rec = {}
    for did, tes in doc_ext.items():
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        doc_bundles_for_rec[did] = [{"intervals": ivs}] if ivs else []
    for d in docs:
        doc_bundles_for_rec.setdefault(d["doc_id"], [])

    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # Pool: semantic top-K (R-S retrieval)
    per_q_pool = {}
    per_q_r_full = {}
    for q in queries:
        qid = q["query_id"]
        s_scores = per_q_s[qid]
        items = [(did, s_scores.get(did, 0.0)) for did in all_dids]
        items.sort(key=lambda x: x[1], reverse=True)
        pool = [did for did, _ in items[:POOL_CAP]]
        per_q_pool[qid] = pool
        rs = await rerank_topk(reranker, q_text[qid], pool, doc_text, len(pool))
        per_q_r_full[qid] = normalize_rerank_full(
            rs,
            [d["doc_id"] for d in docs],
            0.0,
        )

    failure_classes = {
        "semantic_miss": [],  # gold not in pool
        "rerank_miss": [],  # gold in pool, rerank ranked low, no temporal helps
        "temporal_miss": [],  # gold in pool, mask zeroed gold or extremum didn't lift
        "borderline": [],  # gold in pool, rerank gave reasonable rank, scoring failed
    }

    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        plan = plans.get(qid) or QueryPlan()
        plan_latest = plan.latest_intent
        plan_earliest = plan.earliest_intent

        valid_includes_post = []
        valid_excludes_post = []
        for i, c in enumerate(plan.constraints):
            tes = win_ext.get(f"{qid}__c{i}", [])
            anchor_ivs = []
            for te in tes:
                anchor_ivs.extend(flatten_intervals(te))
            if not anchor_ivs:
                continue
            if c.direction == "not_in":
                valid_excludes_post.append((c, anchor_ivs))
            else:
                valid_includes_post.append((c, anchor_ivs))

        pool = per_q_pool[qid]
        pool_set = set(pool)
        r_full = per_q_r_full[qid]

        mask = {}
        for did in pool:
            inc_max = 1.0
            if valid_includes_post:
                inc_max = 0.0
                for c, anchor_ivs in valid_includes_post:
                    f = constraint_factor_for_doc(
                        doc_ivs_flat.get(did, []),
                        anchor_ivs,
                        c.direction,
                    )
                    if f > inc_max:
                        inc_max = f
            exc_factor = 1.0
            for c, anchor_ivs in valid_excludes_post:
                cont = excluded_containment(
                    doc_ivs_flat.get(did, []),
                    anchor_ivs,
                )
                exc_factor *= max(0.0, 1.0 - cont)
            mask[did] = inc_max * exc_factor

        mask_passers = [did for did in pool if mask[did] >= 0.5]
        if (plan_latest or plan_earliest) and len(mask_passers) >= 2:
            rec_lin_mode = linear_recency_scores(
                {did: doc_bundles_for_rec.get(did, []) for did in mask_passers},
                {did: doc_ref_us[did] for did in mask_passers},
            )
        elif (plan_latest or plan_earliest) and len(pool) >= 2:
            rec_lin_mode = linear_recency_scores(
                {did: doc_bundles_for_rec.get(did, []) for did in pool},
                {did: doc_ref_us[did] for did in pool},
            )
        else:
            rec_lin_mode = {}

        r_pool = {did: r_full.get(did, 0.0) for did in pool}
        base = normalize_dict(r_pool)
        rs = {}
        for did in pool:
            b = base.get(did, 0.0) * mask[did]
            if plan_latest or plan_earliest:
                r = rec_lin_mode.get(did, 0.0)
                if plan_earliest:
                    r = 1.0 - r
                b *= 1.0 + EXTREMUM_MULT_ALPHA * r
            rs[did] = b

        rank = [d for d in rank_from_scores(rs) if d in pool_set and rs[d] > 0.0]
        h = hit_rank(rank, gold_set)
        if h == 1:
            continue  # Successful query

        # Classify failure
        gold_in_pool = bool(gold_set & pool_set)
        if not gold_in_pool:
            failure_classes["semantic_miss"].append(
                {
                    "qid": qid,
                    "qtext": q["text"],
                    "gold_semantic_rank": min(
                        (
                            i + 1
                            for i, did in enumerate(
                                sorted(
                                    per_q_s[qid].keys(),
                                    key=lambda d: per_q_s[qid].get(d, 0.0),
                                    reverse=True,
                                )
                            )
                            if did in gold_set
                        ),
                        default=None,
                    ),
                    "rank": h,
                }
            )
            continue

        # Gold in pool. Why did it not rank #1?
        gold_in_pool_id = next(d for d in pool if d in gold_set)
        gold_mask = mask[gold_in_pool_id]
        gold_score = rs[gold_in_pool_id]
        gold_rerank_norm = base.get(gold_in_pool_id, 0.0)
        top_score = max(rs.values()) if rs else 0.0
        top_did = max(rs, key=rs.get) if rs else None
        top_mask = mask.get(top_did, 0.0)
        top_rerank_norm = base.get(top_did, 0.0)

        if gold_mask < 0.5:
            # Gold zeroed by mask -- mask wrong on gold
            failure_classes["temporal_miss"].append(
                {
                    "qid": qid,
                    "qtext": q["text"],
                    "gold_mask": gold_mask,
                    "rank": h,
                    "reason": "mask_zeroed_gold",
                    "plan": plan.to_dict(),
                }
            )
        elif top_mask < 0.5:
            # Top doc fails mask but gold passes -- shouldn't happen if rs filters mask=0
            failure_classes["temporal_miss"].append(
                {
                    "qid": qid,
                    "qtext": q["text"],
                    "gold_mask": gold_mask,
                    "top_mask": top_mask,
                    "reason": "rs_didnt_zero_top_correctly",
                    "rank": h,
                }
            )
        else:
            # Both gold and top pass mask. Compare rerank.
            if gold_rerank_norm < 0.3 * top_rerank_norm:
                failure_classes["rerank_miss"].append(
                    {
                        "qid": qid,
                        "qtext": q["text"],
                        "gold_rerank_norm": gold_rerank_norm,
                        "top_rerank_norm": top_rerank_norm,
                        "rank": h,
                    }
                )
            else:
                failure_classes["borderline"].append(
                    {
                        "qid": qid,
                        "qtext": q["text"],
                        "gold_rerank_norm": gold_rerank_norm,
                        "top_rerank_norm": top_rerank_norm,
                        "gold_score": gold_score,
                        "top_score": top_score,
                        "rank": h,
                    }
                )

    return failure_classes


async def main():
    print("Loading cross-encoder...", flush=True)
    from memmachine_server.common.reranker.cross_encoder_reranker import (
        CrossEncoderReranker,
        CrossEncoderRerankerParams,
    )
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    reranker = CrossEncoderReranker(
        CrossEncoderRerankerParams(
            cross_encoder=ce,
            max_input_length=512,
        )
    )
    planner = QueryPlanner()

    benches_def = [
        (
            "composition",
            "composition_docs.jsonl",
            "composition_queries.jsonl",
            "composition_gold.jsonl",
            "edge-composition",
        ),
        (
            "hard_bench",
            "hard_bench_docs.jsonl",
            "hard_bench_queries.jsonl",
            "hard_bench_gold.jsonl",
            "v7l-hard_bench",
        ),
        (
            "temporal_essential",
            "temporal_essential_docs.jsonl",
            "temporal_essential_queries.jsonl",
            "temporal_essential_gold.jsonl",
            "v7l-temporal_essential",
        ),
        (
            "tempreason_small",
            "real_benchmark_small_docs.jsonl",
            "real_benchmark_small_queries.jsonl",
            "real_benchmark_small_gold.jsonl",
            "v7l-tempreason_small",
        ),
        (
            "conjunctive_temporal",
            "edge_conjunctive_temporal_docs.jsonl",
            "edge_conjunctive_temporal_queries.jsonl",
            "edge_conjunctive_temporal_gold.jsonl",
            "edge-conjunctive_temporal",
        ),
        (
            "multi_te_doc",
            "edge_multi_te_doc_docs.jsonl",
            "edge_multi_te_doc_queries.jsonl",
            "edge_multi_te_doc_gold.jsonl",
            "edge-multi_te_doc",
        ),
        (
            "relative_time",
            "edge_relative_time_docs.jsonl",
            "edge_relative_time_queries.jsonl",
            "edge_relative_time_gold.jsonl",
            "edge-relative_time",
        ),
        (
            "era_refs",
            "edge_era_refs_docs.jsonl",
            "edge_era_refs_queries.jsonl",
            "edge_era_refs_gold.jsonl",
            "edge-era_refs",
        ),
        (
            "open_ended_date",
            "open_ended_date_docs.jsonl",
            "open_ended_date_queries.jsonl",
            "open_ended_date_gold.jsonl",
            "edge-open_ended_date",
        ),
        (
            "causal_relative",
            "causal_relative_docs.jsonl",
            "causal_relative_queries.jsonl",
            "causal_relative_gold.jsonl",
            "edge-causal_relative",
        ),
        (
            "latest_recent",
            "latest_recent_docs.jsonl",
            "latest_recent_queries.jsonl",
            "latest_recent_gold.jsonl",
            "edge-latest_recent",
        ),
        (
            "negation_temporal",
            "negation_temporal_docs.jsonl",
            "negation_temporal_queries.jsonl",
            "negation_temporal_gold.jsonl",
            "edge-negation_temporal",
        ),
    ]

    summary = {}
    for nm, dp, qp, gp, cl in benches_def:
        try:
            fc = await diagnose_bench(
                nm,
                dp,
                qp,
                gp,
                cl,
                reranker,
                planner,
            )
            summary[nm] = {k: len(v) for k, v in fc.items()}
            summary[nm]["details"] = fc
        except Exception as e:
            import traceback

            traceback.print_exc()
            summary[nm] = {"error": str(e)}

    print("\n\n========== FAILURE CLASS COUNTS ==========")
    print(
        f"{'bench':24s}  {'sem_miss':>8s}  {'rerank':>8s}  {'temporal':>9s}  {'border':>8s}"
    )
    total = {"semantic_miss": 0, "rerank_miss": 0, "temporal_miss": 0, "borderline": 0}
    for nm, s in summary.items():
        if "error" in s:
            print(f"{nm:24s}  ERROR")
            continue
        print(
            f"{nm:24s}  {s['semantic_miss']:>8d}  "
            f"{s['rerank_miss']:>8d}  {s['temporal_miss']:>9d}  "
            f"{s['borderline']:>8d}"
        )
        for k in total:
            total[k] += s[k]
    print(
        f"{'TOTAL':24s}  {total['semantic_miss']:>8d}  "
        f"{total['rerank_miss']:>8d}  {total['temporal_miss']:>9d}  "
        f"{total['borderline']:>8d}"
    )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_diagnose.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
