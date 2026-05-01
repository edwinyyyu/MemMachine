"""Compare planner v2 (current) vs planner v3 (event-anchor + offset
resolution) on the expanded realq_v2 set with 7 added EventRef queries.
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
from query_planner_v2 import QueryPlan
from query_planner_v2 import QueryPlanner as QueryPlannerV2
from query_planner_v3 import QueryPlannerV3
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


async def run_with_planner(planner_name, planner, reranker):
    docs = [json.loads(l) for l in open(DATA_DIR / "realq_v2_docs.jsonl")]
    queries = [json.loads(l) for l in open(DATA_DIR / "realq_v2_queries.jsonl")]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / "realq_v2_gold.jsonl")]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    cache_label = f"realq_v2_{planner_name}"

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{cache_label}-docs", cache_label)
    q_ext = await run_v2_extract(q_items, f"{cache_label}-queries", cache_label)

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    q_cat = {q["query_id"]: q.get("category", "?") for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    all_dids = list(doc_ref_us.keys())

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
            f"{cache_label}-constraints",
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

    print("\nRunning planner v2...", flush=True)
    rows_v2 = await run_with_planner("v2", QueryPlannerV2(), reranker)

    print("\nRunning planner v3...", flush=True)
    rows_v3 = await run_with_planner("v3", QueryPlannerV3(), reranker)

    # Per-query side-by-side
    print("\n\n" + "=" * 130)
    print(f"{'qid':40s}  {'cat':30s}  {'rank v2':>8s}  {'rank v3':>8s}   plan v3")
    print("-" * 130)

    by_cat_v2 = {}
    by_cat_v3 = {}
    for r2, r3 in zip(rows_v2, rows_v3):
        assert r2["qid"] == r3["qid"]
        by_cat_v2.setdefault(r2["category"], []).append(r2)
        by_cat_v3.setdefault(r3["category"], []).append(r3)
        v2 = str(r2["rank"]) if r2["rank"] else "-"
        v3 = str(r3["rank"]) if r3["rank"] else "-"
        # Compact plan v3
        p3 = r3["plan"]
        plan_str = (
            "["
            + ", ".join(
                f"{c['phrase'][:20]}/{c['direction']}"
                for c in p3.get("constraints", [])
            )
            + f"] ext={p3.get('extremum')}"
        )
        flag = ""
        if r2["rank"] != 1 and r3["rank"] == 1:
            flag = " (FIX)"
        elif r2["rank"] == 1 and r3["rank"] != 1:
            flag = " (REGRESS)"
        print(
            f"{r2['qid']:40s}  {r2['category'][:30]:30s}  "
            f"{v2:>8s}  {v3:>8s}{flag}  {plan_str[:60]}"
        )

    # Per-category comparison
    print("\n" + "=" * 80)
    print(f"{'cat':35s}  {'v2 R@1':>10s}  {'v3 R@1':>10s}  Δ")
    cats = sorted(set(by_cat_v2.keys()))
    for cat in cats:
        n = len(by_cat_v2[cat])
        v2_r1 = sum(1 for r in by_cat_v2[cat] if r["rank"] == 1)
        v3_r1 = sum(1 for r in by_cat_v3[cat] if r["rank"] == 1)
        d = v3_r1 - v2_r1
        print(
            f"{cat[:35]:35s}  {v2_r1}/{n}={v2_r1 / n:.2f}  "
            f"{v3_r1}/{n}={v3_r1 / n:.2f}  {d:+d}"
        )

    n = len(rows_v2)
    v2_r1 = sum(1 for r in rows_v2 if r["rank"] == 1)
    v3_r1 = sum(1 for r in rows_v3 if r["rank"] == 1)
    print(
        f"\n  OVERALL R@1: v2={v2_r1}/{n}={v2_r1 / n:.2f}, "
        f"v3={v3_r1}/{n}={v3_r1 / n:.2f}, Δ={v3_r1 - v2_r1:+d}"
    )

    # Detailed diffs
    print("\n" + "=" * 80)
    print("Queries where v3 differs from v2:")
    for r2, r3 in zip(rows_v2, rows_v3):
        if r2["rank"] == r3["rank"]:
            continue
        print(f"\n  {r2['qid']}  ({r2['category']})")
        print(f"    Q: {r2['query']}")
        print(f"    v2 plan: {r2['plan']}")
        print(f"    v3 plan: {r3['plan']}")
        print(f"    v2 rank: {r2['rank']}, v3 rank: {r3['rank']}")
        print(f"    gold_in_pool: v2={r2['gold_in_pool']} v3={r3['gold_in_pool']}")
        print(f"    v3 top3: {r3['top3']}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_realq_v2_v3_compare.json"
    with open(json_path, "w") as f:
        json.dump({"v2": rows_v2, "v3": rows_v3}, f, indent=2, default=str)
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
