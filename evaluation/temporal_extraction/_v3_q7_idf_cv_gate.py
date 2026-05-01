"""Q7 — try Options 1 (tag-IDF) and 2 (per-query CV-gate via inverted index)
in parallel to recover the corpus-T-fusion +0.018 gap on top of the Q5
DB-compatible best (R-S retrieval + B-RMX, macro R@1 = 0.753).

Both options are DB-compatible: no full-corpus scan per query.

Variants:
    B-RMX            : baseline (rerank × mask × (1 + α·rec))
    B-RMX-L_raw      : + (1 + β·raw_lattice)            [Q6 baseline; flat]
    B-RMX-L_idf      : + (1 + β·idf_weighted_lattice)   [Option 1]
    B-RMX-L_cv       : + (1 + β·cv_gated_lattice)       [Option 2]
    B-RMX-L_idfcv    : + (1 + β·idf · cv · raw_lattice) [Both]

Implementation notes:

  Option 1 (IDF):
    Precomputed at ingestion: idf(t) = log(N_total / n_docs_with_tag(t)).
    Per-pool-doc score: raw_lattice * max(idf(t) for t in matched tags).

  Option 2 (CV gate):
    At query time, lattice_retrieve_multi returns scores for matched docs.
    Treat unmatched docs as score 0. Compute corpus-wide mean and stddev:
        mean = sum(matched_scores) / n_total
        var  = sum(s²) / n_total - mean²
        cv   = sqrt(var) / mean   (if mean > 0)
    No corpus scan: only matched docs (typically <<n_total) contribute.
    gate = min(1, cv / 0.20).

  Combined: idfcv = idf · cv · raw.

β tested at {1.0, 2.0, 5.0}.
"""

from __future__ import annotations

import asyncio
import json
import math
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
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import (
    expand_query_tags,
)
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
LATTICE_BETAS = [0.0, 1.0, 2.0, 5.0]
LATTICE_VARIANTS = ["none", "raw", "idf", "cv", "idfcv"]
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "R-S")
CV_REF = 0.20


def compute_idf_map(lat: LatticeStore, n_total: int) -> dict[str, float]:
    """Per-tag IDF: log(N / n_docs_with_tag). Computed once after ingestion.

    Implementation: scan the lattice index for tag-doc pairs, count docs
    per tag, compute IDF. In production this is a precomputed side-table.
    """
    cur = lat.con.execute(
        "SELECT tag, COUNT(DISTINCT doc_id) FROM lattice_tags GROUP BY tag",
    )
    idf = {}
    for tag, cnt in cur.fetchall():
        idf[tag] = math.log(max(1, n_total) / max(1, cnt))
    return idf


def per_query_cv_gate(scores_dict: dict[str, float], n_total: int) -> float:
    """Compute corpus-CV from the matched-doc subset + n_total.

    For docs NOT in scores_dict, score=0. This lets us compute mean and
    variance over the whole corpus from just the matched subset:
        mean = sum_matched / n_total
        sum_sq = sum(s² for s in matched_scores)
        var = sum_sq / n_total - mean²
        cv = sqrt(var) / mean  if mean > 0 else 0
    """
    if n_total <= 0 or not scores_dict:
        return 0.0
    sum_s = sum(scores_dict.values())
    sum_sq = sum(s * s for s in scores_dict.values())
    mean = sum_s / n_total
    if mean <= 1e-12:
        return 0.0
    var = max(0.0, sum_sq / n_total - mean * mean)
    cv = math.sqrt(var) / mean
    return min(1.0, cv / CV_REF)


def lattice_score_idf_per_doc(
    pool_dids,
    lat: LatticeStore,
    expanded_query_tags,
    idf_map,
    base_scores,
):
    """Per-doc IDF-weighted lattice score: raw * max-IDF-of-matched-query-tags.

    For each pool doc, look up matched query-tags from the inverted index.
    Multiply the raw lattice score by max IDF of any matched query tag.
    """
    matched = lat.query_by_tags(expanded_query_tags.keys())
    out = {}
    for did in pool_dids:
        raw = base_scores.get(did, 0.0)
        if raw <= 0.0 or did not in matched:
            out[did] = 0.0
            continue
        matched_q_tags = matched[did]  # set of tags
        if not matched_q_tags:
            out[did] = 0.0
            continue
        max_idf = max(idf_map.get(t, 0.0) for t in matched_q_tags)
        out[did] = raw * max_idf
    return out


def expand_for_query_tes(query_tes):
    """Union of expanded query tags across all query TEs (matches what
    lattice_retrieve_multi does internally for retrieval)."""
    expanded = {}
    for te in query_tes:
        qtags = lattice_tags_for_expression(te)
        ex = expand_query_tags(qtags, down_levels=1)
        expanded.update(ex)
    return expanded


async def run_bench_q7(
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

    lat_db = (
        ROOT
        / "cache"
        / "composition_v3_q7"
        / f"lat_{name}_K{POOL_CAP}_{RETRIEVAL_MODE}.sqlite"
    )
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)

    # Option 1: precompute IDF map per tag
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

    # Lattice raw scores per query (over all matched docs).
    per_q_lat = {
        qid: lattice_retrieve_multi(
            lat,
            q_ext.get(qid, []),
            down_levels=1,
        )[0]
        for qid in qids
    }

    # Per-query CV gate from corpus distribution (Option 2).
    per_q_cv_gate = {qid: per_query_cv_gate(per_q_lat[qid], n_total) for qid in qids}

    # Per-query expanded query tags (for IDF lookup of matched tags).
    per_q_expanded_tags = {
        qid: expand_for_query_tes(q_ext.get(qid, [])) for qid in qids
    }

    # Build pool: R-S (semantic top-K).
    print(f"  building pools (mode={RETRIEVAL_MODE}) and reranking...", flush=True)
    per_q_pool = {}
    per_q_r_full = {}
    for q in queries:
        qid = q["query_id"]
        if RETRIEVAL_MODE == "R-S":
            eligible = all_dids
        else:
            raise ValueError(f"This Q7 script supports only R-S; got {RETRIEVAL_MODE}")
        s_scores = per_q_s[qid]
        items = [(did, s_scores.get(did, 0.0)) for did in eligible]
        items.sort(key=lambda x: x[1], reverse=True)
        pool = [did for did, _ in items[:POOL_CAP]]
        per_q_pool[qid] = pool
        rs = await rerank_topk(reranker, q_text[qid], pool, doc_text, len(pool))
        per_q_r_full[qid] = normalize_rerank_full(
            rs,
            [d["doc_id"] for d in docs],
            0.0,
        )

    # Per-query, per-(variant, β) scoring.
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
        lat_raw = {did: per_q_lat[qid].get(did, 0.0) for did in pool}
        lat_idf = lattice_score_idf_per_doc(
            pool,
            lat,
            per_q_expanded_tags[qid],
            idf_map,
            lat_raw,
        )
        cv_gate = per_q_cv_gate[qid]
        lat_cv = {did: cv_gate * lat_raw[did] for did in pool}
        lat_idfcv = {did: cv_gate * lat_idf[did] for did in pool}

        per_config = {}
        for variant in LATTICE_VARIANTS:
            if variant == "none":
                betas = [0.0]
                lat_scores = dict.fromkeys(pool, 0.0)
            elif variant == "raw":
                betas = LATTICE_BETAS
                lat_scores = lat_raw
            elif variant == "idf":
                betas = LATTICE_BETAS
                lat_scores = lat_idf
            elif variant == "cv":
                betas = LATTICE_BETAS
                lat_scores = lat_cv
            elif variant == "idfcv":
                betas = LATTICE_BETAS
                lat_scores = lat_idfcv
            else:
                continue
            for beta in betas:
                rs = {}
                for did in pool:
                    b = base.get(did, 0.0) * mask[did]
                    if plan_latest or plan_earliest:
                        r = rec_lin_mode.get(did, 0.0)
                        if plan_earliest:
                            r = 1.0 - r
                        b *= 1.0 + EXTREMUM_MULT_ALPHA * r
                    if beta > 0.0:
                        b *= 1.0 + beta * lat_scores[did]
                    rs[did] = b
                rank = [
                    d for d in rank_from_scores(rs) if d in pool_set and rs[d] > 0.0
                ]
                key = f"{variant}_b{beta:.0f}"
                per_config[key] = hit_rank(rank, gold_set)

        results.append(
            {
                "qid": qid,
                "gold": list(gold_set),
                "cv_gate": cv_gate,
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

    # Build configs flat list.
    configs = ["none_b0"]
    for v in LATTICE_VARIANTS:
        if v == "none":
            continue
        for b in LATTICE_BETAS:
            if b == 0.0:
                continue
            configs.append(f"{v}_b{b:.0f}")

    out = {
        "benches": {},
        "configs": configs,
        "POOL_CAP": POOL_CAP,
        "retrieval_mode": RETRIEVAL_MODE,
        "lattice_variants": LATTICE_VARIANTS,
        "lattice_betas": LATTICE_BETAS,
    }
    for nm, dp, qp, gp, cl in benches_def:
        try:
            results = await run_bench_q7(nm, dp, qp, gp, cl, reranker, planner)
            overall = aggregate_overall(results, configs)
            out["benches"][nm] = {
                "n": overall["n"],
                "overall": overall,
                "per_q": results,
            }
            print(f"  {'config':14s}  R@1", flush=True)
            for c in configs:
                d = overall[c]
                print(f"  {c:14s}  {d['R@1']:.3f}", flush=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][nm] = {"error": str(e), "n": 0}

    out["planner_stats"] = planner.stats()
    valid = [
        k for k, v in out["benches"].items() if "error" not in v and v.get("n", 0) > 0
    ]
    macro = {
        c: sum(out["benches"][b]["overall"][c]["R@1"] for b in valid)
        / max(1, len(valid))
        for c in configs
    }
    print(f"\nMacro R@1 across {len(valid)} benches:")
    for c in configs:
        print(f"  {c:14s}: {macro[c]:.3f}")
    out["macro"] = macro

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"T_q7_idf_cv_gate_K{POOL_CAP}_{RETRIEVAL_MODE}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
