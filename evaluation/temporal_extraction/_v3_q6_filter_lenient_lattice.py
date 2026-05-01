"""Q6 — two architectural moves driven by user follow-up:

1. **Lenient filter at retrieval**: avoid binary EXISTS-overlap on excludes.
   Three modes tested:
     R-S      : no retrieval filter
     R-SF     : strict (filter on both includes and excludes, binary EXISTS)
     R-SFi    : filter on includes only (excludes deferred to scoring mask)
     R-SFle   : filter on includes + filter excludes only when fully contained
                (excluded_containment == 1.0)

2. **Lattice as point-wise scoring (no aggregation)**: for each pool doc,
   look up its lattice tags (DB inverted index), compute overlap with
   expanded query tags, get per-doc score from `lattice_retrieve_multi`'s
   per-tag formula `0.4*cell_score + 0.2*direction_bonus`. No corpus-wide
   normalization, no CV gate. Multiplicative boost as `(1 + β · lattice)`.

Scoring variants:
    B-RMX   : rerank * mask * (1 + α·rec_lin)              (Q5 winner)
    B-RMXL  : rerank * mask * (1 + α·rec_lin) * (1 + β·lattice_pointwise)

α=3 (extremum boost). β tested at {1, 2, 3, 5}.

Pool-norm, no tail-fill, K=10. All operations are per-pool-doc lookups,
DB-compatible (no corpus-wide aggregation beyond the rerank's own normalize).
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
    hit_rank,
    linear_recency_scores,
    normalize_dict,
    normalize_rerank_full,
    rank_from_scores,
)
from force_pick_optimizers_eval import rerank_topk
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
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
LATTICE_BETAS = [0.0, 1.0, 2.0, 3.0, 5.0]

RETRIEVAL_MODES = ["R-S", "R-SF", "R-SFi", "R-SFle"]


def doc_passes_filter_inc_only(doc_intervals, valid_includes, valid_excludes_unused):
    """Filter on includes only; excludes are deferred to the scoring mask."""
    if not valid_includes:
        return True
    for direction, anchor_ivs in valid_includes:
        if (
            constraint_factor_for_doc(
                doc_intervals,
                anchor_ivs,
                direction,
            )
            >= 1.0
        ):
            return True
    return False


def doc_passes_filter_lenient(doc_intervals, valid_includes, valid_excludes):
    """Lenient filter: drop on includes EXISTS as before; drop on excludes
    only if doc is FULLY contained in the excluded period (cont==1.0).
    Partial-overlap docs survive retrieval and get fractional penalty
    via the scoring mask."""
    if valid_includes:
        passed = False
        for direction, anchor_ivs in valid_includes:
            if (
                constraint_factor_for_doc(
                    doc_intervals,
                    anchor_ivs,
                    direction,
                )
                >= 1.0
            ):
                passed = True
                break
        if not passed:
            return False
    for anchor_ivs in valid_excludes:
        cont = excluded_containment(doc_intervals, anchor_ivs)
        if cont >= 1.0 - 1e-9:
            return False
    return True


def build_eligible(
    retrieval_mode, all_dids, doc_ivs_flat, valid_includes, valid_excludes
):
    if retrieval_mode == "R-S":
        return all_dids
    if retrieval_mode == "R-SF":
        return [
            did
            for did in all_dids
            if doc_passes_filter(
                doc_ivs_flat.get(did, []), valid_includes, valid_excludes
            )
        ]
    if retrieval_mode == "R-SFi":
        return [
            did
            for did in all_dids
            if doc_passes_filter_inc_only(
                doc_ivs_flat.get(did, []), valid_includes, valid_excludes
            )
        ]
    if retrieval_mode == "R-SFle":
        return [
            did
            for did in all_dids
            if doc_passes_filter_lenient(
                doc_ivs_flat.get(did, []), valid_includes, valid_excludes
            )
        ]
    raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")


async def run_bench_q6(
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
    all_dids = list(doc_ref_us.keys())

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

    lat_db = ROOT / "cache" / "composition_v3_q6" / f"lat_{name}_K{POOL_CAP}.sqlite"
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

    # Per-query lattice scores: dict[doc_id] -> raw score from
    # `lattice_retrieve_multi` (point-wise per-doc score; uses query-tag/doc-tag
    # overlap and span-bias formula, no corpus stats).
    per_q_lat = {
        qid: lattice_retrieve_multi(
            lat,
            q_ext.get(qid, []),
            down_levels=1,
        )[0]
        for qid in qids
    }

    print(
        f"  building pools and reranking ({len(RETRIEVAL_MODES)} retrieval modes)...",
        flush=True,
    )
    per_q_pool_by_mode = {qid: {} for qid in qids}
    per_q_r_full_by_mode = {qid: {} for qid in qids}

    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid) or QueryPlan()
        valid_includes, valid_excludes = build_filter_constraints(
            plan,
            win_ext,
            qid,
        )
        for retrieval_mode in RETRIEVAL_MODES:
            eligible = build_eligible(
                retrieval_mode,
                all_dids,
                doc_ivs_flat,
                valid_includes,
                valid_excludes,
            )
            pool = semantic_topk_in_set(per_q_s[qid], eligible, POOL_CAP)
            per_q_pool_by_mode[qid][retrieval_mode] = pool

    rerank_cache = {}
    for q in queries:
        qid = q["query_id"]
        for retrieval_mode in RETRIEVAL_MODES:
            pool = per_q_pool_by_mode[qid][retrieval_mode]
            key = (qid, tuple(pool))
            if key not in rerank_cache:
                rerank_cache[key] = await rerank_topk(
                    reranker,
                    q_text[qid],
                    pool,
                    doc_text,
                    len(pool),
                )
            rs = rerank_cache[key]
            per_q_r_full_by_mode[qid][retrieval_mode] = normalize_rerank_full(
                rs,
                [d["doc_id"] for d in docs],
                0.0,
            )

    # Per-query, per-(retrieval, β) scoring with B-RMXL.
    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        plan = plans.get(qid) or QueryPlan()
        plan_latest = plan.latest_intent
        plan_earliest = plan.earliest_intent

        # Per-doc mask anchors (for scoring-time mask).
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

        per_config = {}
        for retrieval_mode in RETRIEVAL_MODES:
            pool = per_q_pool_by_mode[qid][retrieval_mode]
            pool_set = set(pool)
            r_full = per_q_r_full_by_mode[qid][retrieval_mode]

            # Per-pool-doc mask scoring.
            mask = {}
            for did in pool:
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
                else:
                    inc_max = 1.0
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
            lat_pool = {did: per_q_lat[qid].get(did, 0.0) for did in pool}

            for beta in LATTICE_BETAS:
                rs = {}
                for did in pool:
                    b = base.get(did, 0.0) * mask[did]
                    if plan_latest or plan_earliest:
                        r = rec_lin_mode.get(did, 0.0)
                        if plan_earliest:
                            r = 1.0 - r
                        b *= 1.0 + EXTREMUM_MULT_ALPHA * r
                    if beta > 0.0:
                        b *= 1.0 + beta * lat_pool[did]
                    rs[did] = b
                rank = [
                    d for d in rank_from_scores(rs) if d in pool_set and rs[d] > 0.0
                ]
                key = f"{retrieval_mode}_b{beta:.0f}"
                per_config[key] = hit_rank(rank, gold_set)

        results.append(
            {
                "qid": qid,
                "gold": list(gold_set),
                **{f"hit_{k}": v for k, v in per_config.items()},
            }
        )

    return results


def aggregate_overall(results, configs):
    n = len(results)
    out = {"n": n}
    for c in configs:
        ranks = [r[f"hit_{c}"] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
        out[c] = {
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

    configs = [f"{m}_b{b:.0f}" for m in RETRIEVAL_MODES for b in LATTICE_BETAS]
    out = {
        "benches": {},
        "configs": configs,
        "POOL_CAP": POOL_CAP,
        "retrieval_modes": RETRIEVAL_MODES,
        "lattice_betas": LATTICE_BETAS,
    }
    for nm, dp, qp, gp, cl in benches_def:
        try:
            results = await run_bench_q6(nm, dp, qp, gp, cl, reranker, planner)
            overall = aggregate_overall(results, configs)
            out["benches"][nm] = {
                "n": overall["n"],
                "overall": overall,
                "per_q": results,
            }
            print(f"  {'config':16s}  R@1", flush=True)
            for c in configs:
                d = overall[c]
                print(f"  {c:16s}  {d['R@1']:.3f}", flush=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][nm] = {"error": str(e), "n": 0}

    out["planner_stats"] = planner.stats()
    valid = [
        k for k, v in out["benches"].items() if "error" not in v and v.get("n", 0) > 0
    ]
    print(f"\nMacro R@1 across {len(valid)} benches:")
    print(f"  {'config':16s}  R@1")
    macro = {}
    for c in configs:
        m = sum(out["benches"][b]["overall"][c]["R@1"] for b in valid) / max(
            1, len(valid)
        )
        macro[c] = m
        print(f"  {c:16s}  {m:.3f}")
    out["macro"] = macro

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"T_q6_filter_lenient_lattice_K{POOL_CAP}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
