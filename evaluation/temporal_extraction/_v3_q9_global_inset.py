"""Q9 — close the remaining 0.013 gap by computing rec_lin over GLOBAL
mask-passers (DB-compatible via events side-table aggregation), instead
of just pool mask-passers.

Same architecture as Q8 best (wT=0.4 + IDF), but:
  in_set = ALL corpus docs that pass the mask, not just pool mask-passers.

This requires evaluating the mask over the corpus (per-query), which is
DB-compatible since the mask is per-doc EXISTS-overlap on the events
side-table — same operation as filter-pushdown at retrieval time.
The result is a set of doc_ids; rec_lin normalizes over their max-
event-timestamps. Per-pool-doc lookup of the resulting rec_lin.
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

from _v3_q7_idf_cv_gate import (
    compute_idf_map,
    expand_for_query_tes,
    per_query_cv_gate,
)
from _v3_q8_additive_blend_dbcompat import (
    fused_additive,
    lattice_idf_factor_per_doc,
)
from composition_eval_v3 import (
    EXTREMUM_MULT_ALPHA,
    EXTREMUM_TOPK,
    constraint_factor_for_doc,
    hit_rank,
    linear_recency_scores,
    normalize_dict,
    normalize_rerank_full,
    rank_from_scores,
)
from force_pick_optimizers_eval import rerank_topk
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import (
    retrieve_multi as lattice_retrieve_multi,
)
from lattice_store import LatticeStore
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

POOL_CAP = int(os.environ.get("POOL_CAP", "10"))
W_T = 0.4
W_R = 0.6
USE_IDF = True


async def run_bench_q9(
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

    print(f"  planning ({len(queries)} queries)...", flush=True)
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

    lat_db = ROOT / "cache" / "composition_v3_q9" / f"lat_{name}_K{POOL_CAP}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)

    idf_map = compute_idf_map(lat, n_total)

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

    per_q_lat = {
        qid: lattice_retrieve_multi(
            lat,
            q_ext.get(qid, []),
            down_levels=1,
        )[0]
        for qid in qids
    }
    per_q_t_max = {
        qid: max(scores.values(), default=0.0) for qid, scores in per_q_lat.items()
    }
    per_q_cv = {qid: per_query_cv_gate(per_q_lat[qid], n_total) for qid in qids}
    per_q_expanded_tags = {
        qid: expand_for_query_tes(q_ext.get(qid, [])) for qid in qids
    }

    print(f"  pool (R-S, K={POOL_CAP}) and reranking...", flush=True)
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

    results = []
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
        rerank_pool = {did: r_full.get(did, 0.0) for did in pool}

        # Per-pool-doc mask.
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

        # Q9: GLOBAL mask-passers (over the corpus) for rec_lin candidate set.
        # In production: a single events-side-table EXISTS query that returns
        # all doc_ids satisfying the include/exclude predicate. Here we
        # simulate it by iterating doc_ivs_flat (DB-equivalent: one indexed
        # query per query-time predicate).
        if plan_latest or plan_earliest:
            global_mask_passers = []
            for did in all_dids:
                inc_ok = not valid_includes_post
                if not inc_ok:
                    for c, anchor_ivs in valid_includes_post:
                        if (
                            constraint_factor_for_doc(
                                doc_ivs_flat.get(did, []),
                                anchor_ivs,
                                c.direction,
                            )
                            >= 1.0
                        ):
                            inc_ok = True
                            break
                if not inc_ok:
                    continue
                exc_ok = True
                for c, anchor_ivs in valid_excludes_post:
                    if (
                        constraint_factor_for_doc(
                            doc_ivs_flat.get(did, []),
                            anchor_ivs,
                            "in",
                        )
                        >= 1.0
                    ):
                        exc_ok = False
                        break
                if not exc_ok:
                    continue
                global_mask_passers.append(did)
            if not global_mask_passers and not valid_includes_post:
                # No constraints emitted -- fall back to top-K rerank-fused docs
                # in the corpus (matches Q1 behavior).
                global_mask_passers = sorted(
                    all_dids,
                    key=lambda d: r_full.get(d, 0.0),
                    reverse=True,
                )[:EXTREMUM_TOPK]
            if len(global_mask_passers) >= 2:
                rec_lin_mode = linear_recency_scores(
                    {
                        did: doc_bundles_for_rec.get(did, [])
                        for did in global_mask_passers
                    },
                    {did: doc_ref_us[did] for did in global_mask_passers},
                )
            else:
                rec_lin_mode = {}
        else:
            rec_lin_mode = {}

        idf_factor = (
            lattice_idf_factor_per_doc(
                pool,
                lat,
                per_q_expanded_tags[qid],
                idf_map,
            )
            if USE_IDF
            else None
        )

        fused = fused_additive(
            pool,
            per_q_lat[qid],
            per_q_t_max[qid],
            per_q_cv[qid],
            rerank_pool,
            W_T,
            W_R,
            idf_factor=idf_factor,
        )
        base = normalize_dict(fused)
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

        results.append(
            {
                "qid": qid,
                "gold": list(gold_set),
                "global_mask_passers_n": len(global_mask_passers)
                if (plan_latest or plan_earliest)
                else 0,
                "hit_q9": hit_rank(rank, gold_set),
            }
        )

    return results


def aggregate_overall(results):
    n = len(results)
    if not n:
        return {
            "n": 0,
            "R@1": 0.0,
            "R@5": 0.0,
            "MRR": 0.0,
            "r1_count": 0,
            "r5_count": 0,
        }
    ranks = [r["hit_q9"] for r in results]
    r1 = sum(1 for x in ranks if x is not None and x <= 1)
    r5 = sum(1 for x in ranks if x is not None and x <= 5)
    mrr = sum(1.0 / x for x in ranks if x is not None) / n
    return {
        "n": n,
        "R@1": r1 / n,
        "R@5": r5 / n,
        "MRR": mrr,
        "r1_count": r1,
        "r5_count": r5,
    }


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

    out = {"benches": {}, "POOL_CAP": POOL_CAP}
    for nm, dp, qp, gp, cl in benches_def:
        try:
            results = await run_bench_q9(nm, dp, qp, gp, cl, reranker, planner)
            overall = aggregate_overall(results)
            out["benches"][nm] = {
                "n": overall["n"],
                "overall": overall,
                "per_q": results,
            }
            print(
                f"  Q9 (R-S + addblend wT=0.4 + IDF + mask + global-rec): "
                f"R@1={overall['R@1']:.3f} ({overall['r1_count']}/{overall['n']})",
                flush=True,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][nm] = {"error": str(e), "n": 0}

    out["planner_stats"] = planner.stats()
    valid = [
        k for k, v in out["benches"].items() if "error" not in v and v.get("n", 0) > 0
    ]
    macro = sum(out["benches"][b]["overall"]["R@1"] for b in valid) / max(1, len(valid))
    print(f"\nMacro R@1 across {len(valid)} benches: {macro:.3f}")
    out["macro"] = macro

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"T_q9_global_inset_K{POOL_CAP}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
