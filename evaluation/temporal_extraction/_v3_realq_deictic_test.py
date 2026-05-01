"""Deictic cross-anchor test: doc and query reference the same absolute date
via different relative phrases anchored to different ref_times. Should match
when the LLM extractor resolves both sides to absolute timestamps.
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
    constraint_factor_for_doc,
    hit_rank,
    normalize_dict,
    normalize_rerank_full,
    rank_from_scores,
)
from force_pick_optimizers_eval import rerank_topk
from negation import excluded_containment
from query_planner_v2 import QueryPlan
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
    planner = QueryPlannerV3()

    docs = [json.loads(l) for l in open(DATA_DIR / "realq_deictic_docs.jsonl")]
    queries = [json.loads(l) for l in open(DATA_DIR / "realq_deictic_queries.jsonl")]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / "realq_deictic_gold.jsonl")]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== deictic: {len(docs)} docs, {len(queries)} queries ===", flush=True)
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, "deictic-docs", "deictic")
    q_ext = await run_v2_extract(q_items, "deictic-queries", "deictic")

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    all_dids = list(doc_ref_us.keys())

    # Print doc-side resolved intervals to verify "3 days ago" → absolute date
    print("\nDoc-side resolved intervals:")
    for did, tes in doc_ext.items():
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        if not ivs:
            print(f"  {did}: <no intervals>")
        for iv in ivs:
            from datetime import datetime

            t1 = datetime.fromtimestamp(iv.earliest_us / 1e6).strftime("%Y-%m-%d")
            t2 = datetime.fromtimestamp(iv.latest_us / 1e6).strftime("%Y-%m-%d")
            print(
                f"  {did}: [{t1} .. {t2}]  (ref={[d for d in docs if d['doc_id'] == did][0]['ref_time'][:10]})"
            )

    plans = await planner.plan_many(
        [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    )

    # Print query plans
    print("\nQuery plans:")
    for qid, plan in plans.items():
        print(f"  {qid}: {plan.to_dict()}")

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
            "deictic-constraints",
            "deictic-constraints",
        )
        if win_items
        else {}
    )

    # Print resolved query anchor intervals
    print("\nResolved query anchor intervals (constraint phrase → date):")
    for qid in [q["query_id"] for q in queries]:
        plan = plans.get(qid) or QueryPlan()
        for i, c in enumerate(plan.constraints):
            tes = win_ext.get(f"{qid}__c{i}", [])
            ivs = []
            for te in tes:
                ivs.extend(flatten_intervals(te))
            for iv in ivs:
                from datetime import datetime

                t1 = datetime.fromtimestamp(iv.earliest_us / 1e6).strftime("%Y-%m-%d")
                t2 = datetime.fromtimestamp(iv.latest_us / 1e6).strftime("%Y-%m-%d")
                print(f"  {qid} c{i} ({c.phrase!r}): [{t1} .. {t2}]")

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

    print("\n\nPer-query results (R@1, R@5):")
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

        r_pool = {did: r_full.get(did, 0.0) for did in pool}
        base = normalize_dict(r_pool)
        rs = {}
        for did in pool:
            b = base.get(did, 0.0) * mask[did]
            rs[did] = b

        pool_set = set(pool)
        rank = [d for d in rank_from_scores(rs) if d in pool_set and rs[d] > 0.0]

        gold_set = set(gold.get(qid, []))
        h = hit_rank(rank, gold_set, k=10)
        # Also recall@K — fraction of gold in top-K
        r_at_5 = sum(1 for d in rank[:5] if d in gold_set) / max(1, len(gold_set))
        n_gold_found = sum(1 for d in rank[:10] if d in gold_set)

        print(f"\n  {qid}: '{q_text[qid]}' (ref={q['ref_time'][:10]})")
        print(f"    plan: {plan.to_dict()}")
        print(f"    n_eligible_filt: {len(eligible_filt)}, pool_size: {len(pool)}")
        print(f"    gold ({len(gold_set)} docs): {sorted(gold_set)}")
        print(f"    rank (R@1={'Y' if h == 1 else 'N'}): {rank}")
        print(
            f"    R@5: {n_gold_found}/{len(gold_set)} found in top-10; "
            f"R@5_rank-cov={r_at_5:.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
