"""Q4 — does adding a DB-compatible tag-retrieval channel recover the
~0.045 macro R@1 lift that corpus-norm T-fusion was extracting?

Architecture: same B-RX scoring (rerank × extremum, pool-norm, no tail-fill)
across all modes; only the retrieval-pool composition varies.

Modes:
    R-SF        : semantic top-K filtered (Q1 winner)
    R-SF_UTagF  : (semantic top-K/2 ∪ lattice-tag top-K/2), both filtered

Tag retrieval is DB-compatible: each doc's lattice tags are an indexed
payload array (Qdrant payload-array MatchAny / SQL JOIN tag_table on
B-tree(tag)). Lookup is O(log N + matches) per tag. Query-side tags come
from the planner constraints' phrases + `expand_query_tags(...)` for
hierarchical lattice (year ↔ quarter ↔ month etc.). lattice_retrieve_multi
returns scored doc_ids; we take the top-K/2.
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
from query_planner_v2 import QueryPlan, QueryPlanner
from salience_eval import (
    DATA_DIR,
    build_memory,
    embed_all,
    flatten_intervals,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from schema import to_us

POOL_CAP = int(os.environ.get("POOL_CAP", "10"))
PER_CHANNEL = max(1, POOL_CAP // 2)
MODES = ["R-SF", "R-SF_UTagF"]


def tag_topk_in_set(tag_scores, eligible_set, k):
    """Top-k by lattice tag score, restricted to `eligible_set`.

    `tag_scores` comes from lattice_retrieve_multi — only docs with at
    least one matching tag. Restricting to eligible_set is the EXISTS-
    overlap filter applied to the tag-channel results."""
    if not tag_scores:
        return []
    items = [(did, s) for did, s in tag_scores.items() if did in eligible_set]
    items.sort(key=lambda x: x[1], reverse=True)
    return [did for did, _ in items[:k]]


def build_pool(mode, s_scores, tag_scores, eligible_filt_set):
    """Construct the rerank pool. Both modes use the same EXISTS-overlap
    filter; mode differs only in whether the tag-channel contributes."""
    if mode == "R-SF":
        return semantic_topk_in_set(
            s_scores,
            list(eligible_filt_set),
            POOL_CAP,
        )
    if mode == "R-SF_UTagF":
        sem_top = semantic_topk_in_set(
            s_scores,
            list(eligible_filt_set),
            PER_CHANNEL,
        )
        tag_top = tag_topk_in_set(tag_scores, eligible_filt_set, PER_CHANNEL)
        return list(dict.fromkeys(sem_top + tag_top))[:POOL_CAP]
    raise ValueError(f"Unknown mode: {mode}")


async def run_bench_q4(
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

    # Build the lattice (== inverted-tag index): one tag-set per doc.
    # The query is `WHERE tag IN (expanded_query_tags)` which is exactly
    # what a B-tree-indexed tag column or Qdrant MatchAny payload supports.
    lat_db = ROOT / "cache" / "composition_v3_q4tag" / f"lat_{name}_K{POOL_CAP}.sqlite"
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
    doc_mem = build_memory(doc_ext)
    for d in docs:
        doc_mem.setdefault(
            d["doc_id"],
            {
                "intervals": [],
                "axes_merged": {},
                "multi_tags": set(),
            },
        )
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

    # Tag retrieval: lattice_retrieve_multi already does query_by_tags
    # (SQL WHERE tag IN (expanded_keys)). Returns scored doc_ids.
    per_q_tag = {
        qid: lattice_retrieve_multi(
            lat,
            q_ext.get(qid, []),
            down_levels=1,
        )[0]
        for qid in qids
    }

    print(f"  building pools and reranking ({len(MODES)} modes)...", flush=True)
    per_q_pool_by_mode = {qid: {} for qid in qids}
    per_q_r_partial_by_mode = {qid: {} for qid in qids}
    per_q_r_full_by_mode = {qid: {} for qid in qids}
    per_q_eligible_filt = {}

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
        per_q_eligible_filt[qid] = eligible_filt
        eligible_filt_set = set(eligible_filt)

        for mode in MODES:
            pool = build_pool(
                mode,
                per_q_s[qid],
                per_q_tag[qid],
                eligible_filt_set,
            )
            per_q_pool_by_mode[qid][mode] = pool

    rerank_cache = {}
    for q in queries:
        qid = q["query_id"]
        for mode in MODES:
            pool = per_q_pool_by_mode[qid][mode]
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
            per_q_r_partial_by_mode[qid][mode] = rs
            per_q_r_full_by_mode[qid][mode] = normalize_rerank_full(
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
        text = q["text"]
        ctype = q_type.get(qid, "?")
        plan = plans.get(qid) or QueryPlan()
        plan_latest = plan.latest_intent
        plan_earliest = plan.earliest_intent

        # Mask passers (used only for rec_lin candidate set; pool-norm scoring).
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

        h_per_mode = {}
        top5_per_mode = {}
        pool_size_per_mode = {}

        for mode in MODES:
            pool = per_q_pool_by_mode[qid][mode]
            pool_set = set(pool)
            pool_size_per_mode[mode] = len(pool)
            r_full = per_q_r_full_by_mode[qid][mode]

            # Compute mask passers within the pool to drive rec_lin candidate set.
            if valid_includes_post:
                mask_passers = []
                for did in pool:
                    inc_ok = False
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
                    if exc_ok:
                        mask_passers.append(did)
            else:
                mask_passers = list(pool)

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

            # B-RX scoring, pool-norm, no tail-fill.
            r_pool = {did: r_full.get(did, 0.0) for did in pool}
            base = normalize_dict(r_pool)
            rs = {}
            for did in pool:
                b = base.get(did, 0.0)
                if plan_latest or plan_earliest:
                    r = rec_lin_mode.get(did, 0.0)
                    if plan_earliest:
                        r = 1.0 - r
                    b *= 1.0 + EXTREMUM_MULT_ALPHA * r
                rs[did] = b
            rank = [d for d in rank_from_scores(rs) if d in pool_set and rs[d] > 0.0]
            h_per_mode[mode] = hit_rank(rank, gold_set)
            top5_per_mode[mode] = rank[:5]

        results.append(
            {
                "qid": qid,
                "type": ctype,
                "qtext": text,
                "gold": list(gold_set),
                "plan": plan.to_dict(),
                "plan_extremum": plan.extremum,
                "n_eligible_filt": len(per_q_eligible_filt[qid]),
                **{f"hit_{m}": h_per_mode[m] for m in MODES},
                **{f"top5_{m}": top5_per_mode[m] for m in MODES},
                **{f"pool_size_{m}": pool_size_per_mode[m] for m in MODES},
            }
        )

    return results


def aggregate_overall(results, modes):
    n = len(results)
    out = {"n": n}
    for m in modes:
        ranks = [r[f"hit_{m}"] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
        out[m] = {
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

    out = {"benches": {}, "modes": MODES, "POOL_CAP": POOL_CAP}
    for nm, dp, qp, gp, cl in benches_def:
        try:
            results = await run_bench_q4(nm, dp, qp, gp, cl, reranker, planner)
            overall = aggregate_overall(results, MODES)
            out["benches"][nm] = {
                "n": overall["n"],
                "overall": overall,
                "per_q": results,
            }
            for m in MODES:
                d = overall[m]
                print(
                    f"  {m:14s}  R@1={d['R@1']:.3f} "
                    f"({d['r1_count']}/{overall['n']})  "
                    f"R@5={d['R@5']:.3f}",
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
    macro = {
        m: sum(out["benches"][k]["overall"][m]["R@1"] for k in valid)
        / max(1, len(valid))
        for m in MODES
    }
    print(f"\nMacro R@1 across {len(valid)} benches:")
    for m in MODES:
        print(f"  {m:14s}: {macro[m]:.3f}")
    out["macro"] = macro

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"T_q4_tag_channel_K{POOL_CAP}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
