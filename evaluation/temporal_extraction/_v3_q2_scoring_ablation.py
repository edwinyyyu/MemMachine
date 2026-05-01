"""Q2 post-retrieval scoring ablation.

Holds retrieval fixed at the Q1 winner (K=10, EXISTS-overlap filter, strict
candidate set = pool members). Ablates which components contribute to the
final score:
    R = rerank score (always present)
    M = mask (multi-interval EXISTS, binary 0/1, multiplicative)
    T = T-router fusion (T_lblend or T_v5 blended into rerank via score_blend)
    X = extremum boost (within-set linear recency, multiplicative * (1+α*r))

8 component variants (R always present):

    B-R      : sort by rerank only
    B-RM     : rerank * mask
    B-RX     : rerank * (1 + α*rec)
    B-RT     : T-router fusion (R blended with T)
    B-RMT    : (T-fusion) * mask
    B-RMX    : rerank * mask * (1 + α*rec)         <- drops T-fusion
    B-RTX    : (T-fusion) * (1 + α*rec)
    B-RMTX   : (T-fusion) * mask * (1 + α*rec)     <- current v3

Goal: identify the smallest component set that ties or beats current v3.
Hypothesis: B-RMX (drop T-fusion) ties B-RMTX once retrieval is well-filtered.
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
    semantic_topk_in_set,
)
from composition_eval_v3 import (
    EXTREMUM_MULT_ALPHA,
    constraint_factor_for_doc,
    fuse_T_R_blend_scores,
    hit_rank,
    linear_recency_scores,
    normalize_dict,
    normalize_rerank_full,
    rank_from_scores,
)
from force_pick_optimizers_eval import (
    make_t_scores,
    rerank_topk,
)
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from negation import excluded_containment
from query_planner_v2 import QueryPlan, QueryPlanner
from salience_eval import (
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    flatten_intervals,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from schema import to_us
from t_v5_eval import per_te_bundles_v5, t_v5_doc_scores

# Q1 winner: K=10 strict R-SF.
POOL_CAP = int(os.environ.get("POOL_CAP", "10"))
W_T_FUSE_TR = 0.4
# T-fusion CV gate / normalization domain. "pool" = DB-compatible (only see
# pool docs); "corpus" = matches Q1 (T values + CV computed over full corpus).
T_NORM_DOMAIN = os.environ.get("T_NORM_DOMAIN", "pool")
# Retrieval mode: "R-SF" = filter at retrieval (Q1 winner);
# "R-S" = no retrieval-side filter (mask carries the temporal logic at
# scoring time, if use_mask is enabled).
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "R-SF")

# 8 component variants. Each is a tuple of flags (use_mask, use_T, use_X).
VARIANTS = [
    ("B-R", (False, False, False)),
    ("B-RM", (True, False, False)),
    ("B-RX", (False, False, True)),
    ("B-RT", (False, True, False)),
    ("B-RMT", (True, True, False)),
    ("B-RMX", (True, False, True)),
    ("B-RTX", (False, True, True)),
    ("B-RMTX", (True, True, True)),
]


def score_one_variant(
    *,
    use_mask,
    use_T,
    use_X,
    pool_set,
    doc_ids,
    rerank_full,  # dict[did]: rerank score (normalized over corpus)
    t_router_p,  # dict[did]: T_lblend or T_v5 score (over corpus)
    include_factors_per_doc,  # dict[did]: 0.0 or 1.0
    exclude_factors_per_doc,  # dict[did]: 0.0 or 1.0
    rec_lin_mode,  # dict[did]: rec score in [0,1] within candidate set
    plan_latest,
    plan_earliest,
    t_norm_domain="pool",  # "pool" | "corpus" — where to compute T-fusion CV/normalize
):
    """Compute final scores for one component variant.

    Strict pool: docs not in `pool_set` get 0. Within pool, the score is:
        base = T-fusion(t_router_p, rerank_full) if use_T else rerank_full
        base = normalize over t_norm_domain
        score = base * (mask if use_mask else 1) * ((1 + α*r) if use_X else 1)

    The `t_norm_domain` controls where T-fusion's CV gate and normalize_dict
    operate. "corpus" matches Q1 (full-corpus normalization, T's informativeness
    measured globally — non-DB-compatible since lattice retrieval is a
    side channel). "pool" is the strict DB-compatible interpretation
    (T values restricted to pool members; CV measured locally).
    """
    pool_dids = [did for did in doc_ids if did in pool_set]

    # Base: T-fusion or just rerank.
    if use_T:
        if t_norm_domain == "corpus":
            fused = fuse_T_R_blend_scores(
                t_router_p,
                rerank_full,
                w_T=W_T_FUSE_TR,
            )
            base_full = normalize_dict(fused)
            base = {did: base_full.get(did, 0.0) for did in pool_dids}
        else:
            t_pool = {did: t_router_p.get(did, 0.0) for did in pool_dids}
            r_pool = {did: rerank_full.get(did, 0.0) for did in pool_dids}
            fused = fuse_T_R_blend_scores(t_pool, r_pool, w_T=W_T_FUSE_TR)
            base = normalize_dict(fused)
    else:
        r_pool = {did: rerank_full.get(did, 0.0) for did in pool_dids}
        base = normalize_dict(r_pool)

    rs = {}
    for did in doc_ids:
        if did not in pool_set:
            rs[did] = 0.0
            continue
        b = base.get(did, 0.0)
        if use_mask:
            m = include_factors_per_doc.get(did, 1.0) * exclude_factors_per_doc.get(
                did, 1.0
            )
            b *= m
        if use_X and (plan_latest or plan_earliest):
            r = rec_lin_mode.get(did, 0.0)
            if plan_earliest:
                r = 1.0 - r
            b *= 1.0 + EXTREMUM_MULT_ALPHA * r
        rs[did] = b
    return rs


async def run_bench_q2(
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
    q_type = {q["query_id"]: q.get("comp_type", "?") for q in queries}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===", flush=True)

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_label)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_label)

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    q_ref_us = {q["query_id"]: to_us(parse_iso(q["ref_time"])) for q in queries}

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

    doc_mem = build_memory(doc_ext)
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

    lat_db = (
        ROOT
        / "cache"
        / "composition_v3_q2ablation"
        / f"lat_{name}_K{POOL_CAP}_{T_NORM_DOMAIN}_{RETRIEVAL_MODE}.sqlite"
    )
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)

    doc_ivs_flat = {}
    for did, tes in doc_ext.items():
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        doc_ivs_flat[did] = ivs
    for d in docs:
        doc_ivs_flat.setdefault(d["doc_id"], [])

    doc_bundles_for_rec = {}
    for did, mem in doc_mem.items():
        ivs = mem.get("intervals") or []
        doc_bundles_for_rec[did] = [{"intervals": ivs}] if ivs else []

    print("  embedding...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    per_q_l = {
        qid: lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)[0]
        for qid in qids
    }
    q_mem = build_memory(q_ext)
    per_q_t = {
        qid: make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }
    for qid in qids:
        for d in docs:
            per_q_t[qid].setdefault(d["doc_id"], 0.0)

    doc_bundles_v5 = per_te_bundles_v5(doc_ext)
    for d in docs:
        doc_bundles_v5.setdefault(d["doc_id"], [])
    q_bundles_v5 = per_te_bundles_v5(q_ext)
    per_q_tv5 = {
        qid: t_v5_doc_scores(q_bundles_v5.get(qid, []), doc_bundles_v5) for qid in qids
    }
    for qid in qids:
        for d in docs:
            per_q_tv5[qid].setdefault(d["doc_id"], 0.0)

    # Retrieval: configurable. "R-SF" filters at retrieval (Q1 winner);
    # "R-S" pulls semantic top-K without filter (mask handled at scoring).
    print(
        f"  retrieving (K={POOL_CAP}, mode={RETRIEVAL_MODE}) and reranking...",
        flush=True,
    )
    all_dids = list(doc_ref_us.keys())
    per_q_pool = {}
    per_q_r_partial = {}
    per_q_r_full = {}

    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid) or QueryPlan()
        if RETRIEVAL_MODE == "R-SF":
            valid_includes, valid_excludes = build_filter_constraints(
                plan,
                win_ext,
                qid,
            )
            eligible = [
                did
                for did in doc_ref_us
                if doc_passes_filter(
                    doc_ivs_flat.get(did, []), valid_includes, valid_excludes
                )
            ]
        elif RETRIEVAL_MODE == "R-S":
            eligible = all_dids
        else:
            raise ValueError(f"Unknown RETRIEVAL_MODE: {RETRIEVAL_MODE}")
        pool = semantic_topk_in_set(per_q_s[qid], eligible, POOL_CAP)
        per_q_pool[qid] = pool
        rs = await rerank_topk(reranker, q_text[qid], pool, doc_text, len(pool))
        per_q_r_partial[qid] = rs
        per_q_r_full[qid] = normalize_rerank_full(
            rs,
            [d["doc_id"] for d in docs],
            0.0,
        )

    results = []
    rec_lin_global = linear_recency_scores(doc_bundles_for_rec, doc_ref_us)

    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        text = q["text"]
        ctype = q_type.get(qid, "?")

        plan = plans.get(qid) or QueryPlan()
        plan_has_open = plan.has_open_constraint
        plan_latest = plan.latest_intent
        plan_earliest = plan.earliest_intent

        # Per-doc mask (multi-interval EXISTS).
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

        if valid_includes_post:
            include_factors_per_doc = dict.fromkeys(doc_ref_us, 0.0)
            for did in doc_ref_us:
                for c, anchor_ivs in valid_includes_post:
                    fac = constraint_factor_for_doc(
                        doc_ivs_flat.get(did, []),
                        anchor_ivs,
                        c.direction,
                    )
                    if fac > include_factors_per_doc[did]:
                        include_factors_per_doc[did] = fac
        else:
            include_factors_per_doc = dict.fromkeys(doc_ref_us, 1.0)

        exclude_factors_per_doc = dict.fromkeys(doc_ref_us, 1.0)
        for c, anchor_ivs in valid_excludes_post:
            for did in doc_ref_us:
                cont = excluded_containment(
                    doc_ivs_flat.get(did, []),
                    anchor_ivs,
                )
                exclude_factors_per_doc[did] *= max(0.0, 1.0 - cont)

        # Within-pool linear recency for extremum normalization.
        pool = per_q_pool[qid]
        pool_set = set(pool)
        if (plan_latest or plan_earliest) and len(pool) >= 2:
            mask_passers = [
                did
                for did in pool
                if include_factors_per_doc.get(did, 1.0) >= 0.5
                and exclude_factors_per_doc.get(did, 1.0) > 0.0
            ]
            cand = mask_passers if len(mask_passers) >= 2 else pool
            rec_lin_mode = linear_recency_scores(
                {did: doc_bundles_for_rec.get(did, []) for did in cand},
                {did: doc_ref_us[did] for did in cand},
            )
        else:
            rec_lin_mode = {}

        t_router_p = per_q_tv5[qid] if plan_has_open else per_q_t[qid]
        rerank_full = per_q_r_full[qid]

        h_per_var = {}
        top5_per_var = {}
        for var_name, (use_mask, use_T, use_X) in VARIANTS:
            rs = score_one_variant(
                use_mask=use_mask,
                use_T=use_T,
                use_X=use_X,
                pool_set=pool_set,
                doc_ids=list(doc_ref_us.keys()),
                rerank_full=rerank_full,
                t_router_p=t_router_p,
                include_factors_per_doc=include_factors_per_doc,
                exclude_factors_per_doc=exclude_factors_per_doc,
                rec_lin_mode=rec_lin_mode,
                plan_latest=plan_latest,
                plan_earliest=plan_earliest,
                t_norm_domain=T_NORM_DOMAIN,
            )
            # Strict pool ranking: only docs in the pool with non-zero score
            # (passed the mask if M is on) appear. No semantic tail-fill —
            # otherwise gold-not-in-pool can sneak into top-10 via semantic
            # rank, violating the "K candidates only" architecture.
            rank = [d for d in rank_from_scores(rs) if d in pool_set and rs[d] > 0.0]
            h_per_var[var_name] = hit_rank(rank, gold_set)
            top5_per_var[var_name] = rank[:5]

        results.append(
            {
                "qid": qid,
                "type": ctype,
                "qtext": text,
                "gold": list(gold_set),
                "plan": plan.to_dict(),
                "plan_extremum": plan.extremum,
                "n_includes_post": len(valid_includes_post),
                "n_excludes_post": len(valid_excludes_post),
                "pool_size": len(pool),
                **{f"hit_{v}": h_per_var[v] for v, _ in VARIANTS},
                **{f"top5_{v}": top5_per_var[v] for v, _ in VARIANTS},
            }
        )
    return results


def aggregate_overall(results, variants):
    n = len(results)
    out = {"n": n}
    for v in variants:
        ranks = [r[f"hit_{v}"] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
        out[v] = {
            "R@1": r1 / n if n else 0.0,
            "R@5": r5 / n if n else 0.0,
            "MRR": mrr,
            "r1_count": r1,
            "r5_count": r5,
        }
    return out


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

    var_names = [v for v, _ in VARIANTS]
    out = {"benches": {}, "variants": var_names, "POOL_CAP": POOL_CAP}
    for nm, dp, qp, gp, cl in benches_def:
        try:
            results = await run_bench_q2(nm, dp, qp, gp, cl, reranker, planner)
            overall = aggregate_overall(results, var_names)
            out["benches"][nm] = {
                "n": overall["n"],
                "overall": overall,
                "per_q": results,
            }
            print(
                f"  {'variant':10s}  {'R@1':>10s}  {'R@5':>10s}  {'MRR':>10s}",
                flush=True,
            )
            for v in var_names:
                d = overall[v]
                print(
                    f"  {v:10s}  {d['R@1']:.3f}      {d['R@5']:.3f}      "
                    f"{d['MRR']:.3f}",
                    flush=True,
                )
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][nm] = {"error": str(e), "n": 0}

    out["planner_stats"] = planner.stats()
    print(f"\nplanner stats: {out['planner_stats']}", flush=True)

    valid = [
        k for k, v in out["benches"].items() if "error" not in v and v.get("n", 0) > 0
    ]
    macro = {
        v: sum(out["benches"][k]["overall"][v]["R@1"] for k in valid)
        / max(1, len(valid))
        for v in var_names
    }
    print(f"\nMacro R@1 across {len(valid)} benches:")
    for v in var_names:
        print(f"  {v:10s}: {macro[v]:.3f}")
    out["macro"] = macro

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"K{POOL_CAP}_retr-{RETRIEVAL_MODE}_Tnorm-{T_NORM_DOMAIN}_noTailfill"
    json_path = out_dir / f"T_q2_scoring_ablation_{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
