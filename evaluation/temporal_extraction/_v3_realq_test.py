"""Sanity-check the production architecture (Q10 hybrid + B-RMX) on a small
handcrafted set of real-world-shaped queries spanning four ChronoQA/ArchivalQA
categories: ArchivalQA-Time, ChronoQA-AbsoluteExplicit,
ChronoQA-RelativeImplicit (deictic), ChronoQA-RelativeExplicit (deictic+offset).

Per-query reporting so we can see WHICH categories work and which don't.
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

from _v3_q1_retrieval_ablation import (
    build_filter_constraints,
    doc_passes_filter,
)
from _v3_q10_hybrid import build_pool
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


async def run_realq(reranker, planner: QueryPlanner):
    docs = [json.loads(l) for l in open(DATA_DIR / "realq_docs.jsonl")]
    queries = [json.loads(l) for l in open(DATA_DIR / "realq_queries.jsonl")]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / "realq_gold.jsonl")]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== realq: {len(docs)} docs, {len(queries)} queries ===", flush=True)

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, "realq-docs", "realq")
    q_ext = await run_v2_extract(q_items, "realq-queries", "realq")

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    q_cat = {q["query_id"]: q.get("category", "?") for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    all_dids = list(doc_ref_us.keys())

    print("  planning...", flush=True)
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
            "realq-constraints",
            "realq-constraints",
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

    print("  embedding...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    print("  retrieving (hybrid R-S/2 + R-SF/2 + topup) and reranking...", flush=True)

    rows = []
    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid) or QueryPlan()
        valid_includes, valid_excludes = build_filter_constraints(
            plan,
            win_ext,
            qid,
        )
        eligible_filt = [
            did
            for did in doc_ref_us
            if doc_passes_filter(
                doc_ivs_flat.get(did, []), valid_includes, valid_excludes
            )
        ]
        pool = build_pool("R-S_half_SF_half", per_q_s[qid], all_dids, eligible_filt)
        rs_partial = await rerank_topk(
            reranker,
            q_text[qid],
            pool,
            doc_text,
            len(pool),
        )
        r_full = normalize_rerank_full(
            rs_partial,
            [d["doc_id"] for d in docs],
            0.0,
        )

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

        pool_set = set(pool)
        rank = [d for d in rank_from_scores(rs) if d in pool_set and rs[d] > 0.0]

        gold_set = set(gold.get(qid, []))
        h = hit_rank(rank, gold_set, k=10)
        gold_in_pool = bool(gold_set & pool_set)
        rows.append(
            {
                "qid": qid,
                "category": q_cat[qid],
                "query": q_text[qid],
                "gold": list(gold_set),
                "rank": h,
                "top3": rank[:3],
                "pool_size": len(pool),
                "gold_in_pool": gold_in_pool,
                "n_eligible_filt": len(eligible_filt),
                "plan": plan.to_dict(),
            }
        )

    return rows


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
    rows = await run_realq(reranker, planner)

    # Per-query report
    print("\n\n" + "=" * 100)
    print(f"{'qid':40s}  {'cat':28s}  rank  pool  gold_in")
    print("-" * 100)
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
        rank_str = str(r["rank"]) if r["rank"] else "-"
        gp = "Y" if r["gold_in_pool"] else "N"
        print(
            f"{r['qid']:40s}  {r['category'][:28]:28s}  "
            f"{rank_str:>4s}  {r['pool_size']:>4d}    {gp}"
        )

    # Per-category summary
    print("\n" + "=" * 60)
    print("Per-category R@1 / R@5:")
    for cat, rs in by_cat.items():
        n = len(rs)
        r1 = sum(1 for r in rs if r["rank"] is not None and r["rank"] <= 1)
        r5 = sum(1 for r in rs if r["rank"] is not None and r["rank"] <= 5)
        print(f"  {cat[:30]:30s}: R@1={r1}/{n}={r1 / n:.2f}, R@5={r5}/{n}={r5 / n:.2f}")

    n = len(rows)
    r1 = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 1)
    r5 = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 5)
    print(f"\n  OVERALL: R@1={r1}/{n}={r1 / n:.2f}, R@5={r5}/{n}={r5 / n:.2f}")

    # Detailed plans for failed queries
    print("\n" + "=" * 60)
    print("Failed queries (rank > 1):")
    for r in rows:
        if r["rank"] == 1:
            continue
        print(f"\n  {r['qid']}  ({r['category']}, rank={r['rank']})")
        print(f"    Q: {r['query']}")
        print(f"    plan: {r['plan']}")
        print(
            f"    n_eligible_filt: {r['n_eligible_filt']}, pool_size: {r['pool_size']}"
        )
        print(f"    gold: {r['gold']}, gold_in_pool: {r['gold_in_pool']}")
        print(f"    top3: {r['top3']}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_realq_test.json"
    with open(json_path, "w") as f:
        json.dump(
            {"rows": rows, "planner_stats": planner.stats()}, f, indent=2, default=str
        )
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
