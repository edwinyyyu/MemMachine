"""Q1 retrieval-mode ablation for the v3 planner stack.

Tests four DB-compatible retrieval modes that combine semantic top-K with an
optional event-time filter and an optional temporal-ordering channel:

    R-S    : semantic top-K
    R-SF   : semantic top-K within {EXISTS-overlap filter}
    R-SUT  : semantic top-K/2 union temporal-ordered top-K/2
    R-SUTF : (semantic top-K/2 union temporal-ordered top-K/2) within filter

Schema assumption (DB-compatible under extension):
  - messages(doc_id, ...) carries the message-level scalars
    (incl. message timestamp; not used for temporal filtering since the
    message timestamp is the authoring time, not the event time).
  - events(doc_id, event_earliest_us, event_latest_us) is a side table
    populated at ingestion from the message's extracted temporal
    intervals (one row per extracted interval).

Filter semantics (EXISTS-overlap on the events side table):
  - direction=in     -> EXISTS e: e.earliest_us <= a.latest AND e.latest_us >= a.earliest
  - direction=after  -> EXISTS e: e.latest_us > a.latest
  - direction=before -> EXISTS e: e.earliest_us < a.earliest
  - direction=not_in -> NOT EXISTS e: e.earliest_us <= a.latest AND e.latest_us >= a.earliest
  multi-include = OR over EXISTS, multi-exclude = AND of NOT-EXISTS.

Docs without any extracted intervals bypass the filter (treated as
not subject to temporal constraints; no rows in events to fail EXISTS).

This is portable to both backends:
  - SQL: WHERE EXISTS (SELECT 1 FROM events e WHERE e.doc_id = m.doc_id AND ...)
  - Qdrant: denormalize events as vector_store records sharing a doc_id
    payload key, query with property_filter on event_earliest_us /
    event_latest_us, dedup by doc_id post-fetch.

Temporal channel (per-doc canonical event timestamp via aggregation):
  - extremum=latest   -> ORDER BY MAX(e.latest_us) DESC LIMIT K/2 (within filter)
  - extremum=earliest -> ORDER BY MIN(e.earliest_us) ASC  LIMIT K/2
  - extremum=null     -> empty
  Aggregation across event rows; doc admitted in order of its extreme
  event timestamp.

Post-retrieval scoring is held fixed across all modes at the v3 best:
T-router fusion (T_lblend|T_v5 blended into rerank), constraint mask
(multi-interval, post-retrieval), and extremum boost (multiplicative alpha=3).
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
    EXTREMUM_TOPK,
    constraint_factor_for_doc,
    fuse_T_R_blend_scores,
    hit_rank,
    linear_recency_scores,
    normalize_dict,
    normalize_rerank_full,
    rank_from_scores,
)
from force_pick_optimizers_eval import (
    RERANK_TOP_K,
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

# Fixed pool size into rerank for all modes. Override via env var POOL_CAP for
# corpus-size sweeps; the four 60-doc benchmarks tautologically tie at pool=75
# (whole corpus reranked). Default is 1.5x RERANK_TOP_K to match the original v3.
POOL_CAP = int(os.environ.get("POOL_CAP", RERANK_TOP_K * 1.5))
PER_CHANNEL = max(1, POOL_CAP // 2)

# STRICT_POOL=1 enforces that the candidate set for post-retrieval scoring is
# exactly the rerank pool. Docs outside the pool get final score 0. This is
# the DB-compatible architecture: the DB returns K candidates and post-
# retrieval scoring only ranks those K. Without it, the T-router fusion
# (T_lblend over the full corpus) acts as a parallel non-DB-compatible
# retrieval channel, contaminating the per-mode comparison.
STRICT_POOL = bool(int(os.environ.get("STRICT_POOL", "0")))


# ----------------------------------------------------------------------
# DB-compatible filter and temporal-channel construction
# ----------------------------------------------------------------------
def doc_passes_filter(doc_intervals, valid_includes, valid_excludes):
    """EXISTS-overlap filter on the events side table.

    `doc_intervals` is the per-doc list of extracted FuzzyInterval rows
    (the events side table for that doc_id). `valid_includes` and
    `valid_excludes` carry per-constraint anchor intervals.

    A doc passes iff:
      - any include constraint has an EXISTS-match (OR over includes), AND
      - no exclude constraint has an EXISTS-match (AND of NOT-EXISTS).

    Docs with no extracted intervals trivially satisfy NOT-EXISTS for any
    exclude clause, but cannot satisfy any include EXISTS clause; they
    pass when there are no includes (filter is exclude-only or empty).
    """
    # Includes: at least one include must have an EXISTS-match.
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

    # Excludes: no exclude clause may have an EXISTS-match.
    for anchor_ivs in valid_excludes:
        # not_in's EXISTS-match uses "in" semantics on the same anchor.
        if (
            constraint_factor_for_doc(
                doc_intervals,
                anchor_ivs,
                "in",
            )
            >= 1.0
        ):
            return False

    return True


def build_filter_constraints(plan, win_ext, qid):
    """Project planner constraints to anchor-interval lists, partitioned
    by include-vs-exclude, for the EXISTS-overlap filter.

    Returns (valid_includes, valid_excludes):
      valid_includes: list of (direction, anchor_intervals)
      valid_excludes: list of anchor_intervals
    Constraints whose phrase produced no extracted intervals are skipped
    (the LLM emitted a non-date phrase that the extractor returned empty).
    """
    valid_includes = []
    valid_excludes = []
    for i, c in enumerate(plan.constraints):
        tes = win_ext.get(f"{qid}__c{i}", [])
        anchor_ivs = []
        for te in tes:
            anchor_ivs.extend(flatten_intervals(te))
        if not anchor_ivs:
            continue
        if c.direction == "not_in":
            valid_excludes.append(anchor_ivs)
        else:
            valid_includes.append((c.direction, anchor_ivs))
    return valid_includes, valid_excludes


def doc_event_extreme_us(doc_intervals, kind):
    """Per-doc canonical event timestamp via aggregation over the side
    table. `kind` selects the SQL aggregate:
      kind=='latest'   -> MAX(event_latest_us)   (None if no events)
      kind=='earliest' -> MIN(event_earliest_us) (None if no events)
    """
    if not doc_intervals:
        return None
    if kind == "latest":
        return max(iv.latest_us for iv in doc_intervals)
    if kind == "earliest":
        return min(iv.earliest_us for iv in doc_intervals)
    return None


def temporal_ordered_topk(doc_ivs_flat, eligible_dids, extremum, k):
    """DB-compatible ORDER BY aggregate event timestamp, LIMIT k.

    - extremum=='latest'   : ORDER BY MAX(e.latest_us) DESC
    - extremum=='earliest' : ORDER BY MIN(e.earliest_us) ASC
    - else                 : empty list

    Docs with no extracted intervals are skipped (NULL aggregate).
    """
    if not extremum or k <= 0:
        return []
    items = []
    for did in eligible_dids:
        agg = doc_event_extreme_us(doc_ivs_flat.get(did, []), extremum)
        if agg is None:
            continue
        items.append((did, agg))
    if extremum == "latest":
        items.sort(key=lambda x: x[1], reverse=True)
    elif extremum == "earliest":
        items.sort(key=lambda x: x[1])
    else:
        return []
    return [did for did, _ in items[:k]]


def semantic_topk_in_set(s_scores, eligible, k):
    """Semantic top-k restricted to `eligible` doc-id set (DB property_filter)."""
    if not eligible:
        return []
    items = [(did, s_scores.get(did, 0.0)) for did in eligible]
    items.sort(key=lambda x: x[1], reverse=True)
    return [did for did, _ in items[:k]]


def build_pool(mode, s_scores, doc_ivs_flat, eligible_all, eligible_filt, extremum):
    """Construct the rerank pool for one of the four ablation modes.

    eligible_all  : full corpus doc-id list
    eligible_filt : doc-id list satisfying the EXISTS-overlap filter
    """
    if mode == "R-S":
        return semantic_topk_in_set(s_scores, eligible_all, POOL_CAP)

    if mode == "R-SF":
        return semantic_topk_in_set(s_scores, eligible_filt, POOL_CAP)

    if mode == "R-SUT":
        sem_top = semantic_topk_in_set(s_scores, eligible_all, PER_CHANNEL)
        tmp_top = temporal_ordered_topk(
            doc_ivs_flat,
            eligible_all,
            extremum,
            PER_CHANNEL,
        )
        return list(dict.fromkeys(sem_top + tmp_top))[:POOL_CAP]

    if mode == "R-SUTF":
        sem_top = semantic_topk_in_set(s_scores, eligible_filt, PER_CHANNEL)
        tmp_top = temporal_ordered_topk(
            doc_ivs_flat,
            eligible_filt,
            extremum,
            PER_CHANNEL,
        )
        return list(dict.fromkeys(sem_top + tmp_top))[:POOL_CAP]

    raise ValueError(f"Unknown mode: {mode}")


# ----------------------------------------------------------------------
# Per-bench evaluation
# ----------------------------------------------------------------------
MODES = ["R-S", "R-SF", "R-SUT", "R-SUTF"]

# Tunables for post-retrieval scoring (held fixed across modes).
W_T_FUSE_TR = 0.4
CV_REF = 0.20


async def run_bench_ablation(
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

    # Constraint phrase extractions
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

    # Doc memory + lattice (used in T-fusion, post-retrieval)
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

    # K-suffix the lattice cache so parallel POOL_CAP runs don't collide.
    lat_db = (
        ROOT / "cache" / "composition_v3_q1ablation" / f"lat_{name}_K{POOL_CAP}.sqlite"
    )
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)

    # Doc intervals flat (used by post-retrieval mask)
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

    # T_lblend and T_v5 (used post-retrieval in T-router fusion)
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

    # ----------------------------------------------------------------
    # Per-mode rerank pools and rerank scores
    # ----------------------------------------------------------------
    print("  building pools and reranking (4 modes)...", flush=True)
    eligible_all = list(doc_ref_us.keys())

    # Cache eligible-filt and pools per query
    per_q_eligible_filt = {}
    per_q_extremum = {}
    per_q_valid_includes = {}
    per_q_valid_excludes = {}
    per_q_pool_by_mode = {qid: {} for qid in qids}
    per_q_r_partial_by_mode = {qid: {} for qid in qids}
    per_q_r_full_by_mode = {qid: {} for qid in qids}

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
        per_q_extremum[qid] = plan.extremum
        per_q_valid_includes[qid] = valid_includes
        per_q_valid_excludes[qid] = valid_excludes

        for mode in MODES:
            pool = build_pool(
                mode,
                per_q_s[qid],
                doc_ivs_flat,
                eligible_all,
                eligible_filt,
                plan.extremum,
            )
            per_q_pool_by_mode[qid][mode] = pool

    # Dedup rerank calls: each (qid, doc_id_set) pair is unique.
    # Many queries collapse R-S=R-SUT and R-SF=R-SUTF when extremum is null.
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

    # ----------------------------------------------------------------
    # Per-query, per-mode scoring (post-retrieval pipeline held fixed)
    # ----------------------------------------------------------------
    results = []
    rec_lin_scores = linear_recency_scores(doc_bundles_for_rec, doc_ref_us)

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

        # ---- post-retrieval mask (multi-interval, same as v3) -------
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

        # ---- candidate set for within-set recency normalization -----
        if plan_latest or plan_earliest:
            if valid_includes_post:
                in_set = [
                    did
                    for did in doc_ref_us
                    if include_factors_per_doc.get(did, 1.0) >= 0.5
                    and exclude_factors_per_doc.get(did, 1.0) > 0.0
                ]
            else:
                # base for in_set requires the per-mode rerank-fused base; we
                # compute below per-mode and re-derive in_set if needed for
                # simplicity, here we use the global rec_lin_scores fallback.
                in_set = None
            if in_set and len(in_set) >= 2:
                rec_lin = linear_recency_scores(
                    {did: doc_bundles_for_rec.get(did, []) for did in in_set},
                    {did: doc_ref_us[did] for did in in_set},
                )
            else:
                rec_lin = None  # set per-mode below
        else:
            rec_lin = {}

        h_per_mode = {}
        top5_per_mode = {}

        for mode in MODES:
            r_full = per_q_r_full_by_mode[qid][mode]

            t_router_p = per_q_tv5[qid] if plan_has_open else per_q_t[qid]
            fused_TR_p = fuse_T_R_blend_scores(
                t_router_p,
                r_full,
                w_T=W_T_FUSE_TR,
            )
            base_p = normalize_dict(fused_TR_p)

            # If we couldn't compute in_set above (no windows) and the query
            # has extremum, use top-K rerank/T docs as candidate set.
            if (plan_latest or plan_earliest) and rec_lin is None:
                topk_rerank = sorted(base_p, key=base_p.get, reverse=True)[
                    :EXTREMUM_TOPK
                ]
                if len(topk_rerank) >= 2:
                    rec_lin_mode = linear_recency_scores(
                        {did: doc_bundles_for_rec.get(did, []) for did in topk_rerank},
                        {did: doc_ref_us[did] for did in topk_rerank},
                    )
                else:
                    rec_lin_mode = rec_lin_scores
            else:
                rec_lin_mode = rec_lin if rec_lin is not None else {}

            # Under STRICT_POOL, the candidate set is exactly the rerank pool;
            # docs outside the pool get final score 0. Otherwise the T-router
            # fusion contributes a non-zero base for all corpus docs and acts
            # as a parallel retrieval channel that masks per-mode differences.
            pool_set = set(per_q_pool_by_mode[qid][mode]) if STRICT_POOL else None

            rs_p = {}
            for did in doc_ref_us:
                if pool_set is not None and did not in pool_set:
                    rs_p[did] = 0.0
                    continue
                mask = include_factors_per_doc.get(
                    did, 1.0
                ) * exclude_factors_per_doc.get(did, 1.0)
                base_m = base_p.get(did, 0.0) * mask
                if plan_latest or plan_earliest:
                    r = rec_lin_mode.get(did, 0.0)
                    if plan_earliest:
                        r = 1.0 - r
                    v = base_m * (1.0 + EXTREMUM_MULT_ALPHA * r)
                else:
                    v = base_m
                rs_p[did] = v

            rank_p = rank_from_scores(rs_p)
            # Tail-fill from semantic ranking so unranked docs still get
            # a stable position; under STRICT_POOL the tail is irrelevant
            # for hit_rank when gold is in pool, and irrelevant for misses.
            rank_p = rank_p + [
                d for d in rank_from_scores(per_q_s[qid]) if d not in set(rank_p)
            ]
            h_per_mode[mode] = hit_rank(rank_p, gold_set)
            top5_per_mode[mode] = rank_p[:5]

        results.append(
            {
                "qid": qid,
                "type": ctype,
                "qtext": text,
                "gold": list(gold_set),
                "plan": plan.to_dict(),
                "plan_extremum": plan.extremum,
                "n_includes_db": len(per_q_valid_includes[qid]),
                "n_excludes_db": len(per_q_valid_excludes[qid]),
                "n_eligible_filt": len(per_q_eligible_filt[qid]),
                "n_corpus": len(doc_ref_us),
                **{f"hit_{m}": h_per_mode[m] for m in MODES},
                **{f"top5_{m}": top5_per_mode[m] for m in MODES},
            }
        )

    return results


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def aggregate_overall(results, modes):
    n = len(results)
    out = {"n": n}
    for m in modes:
        ranks = [r[f"hit_{m}"] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr_v = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
        out[m] = {
            "R@1": r1 / n if n else 0.0,
            "R@5": r5 / n if n else 0.0,
            "MRR": mrr_v,
            "r1_count": r1,
            "r5_count": r5,
        }
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
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

    out = {"benches": {}, "modes": MODES}
    for nm, dp, qp, gp, cl in benches_def:
        try:
            results = await run_bench_ablation(
                nm,
                dp,
                qp,
                gp,
                cl,
                reranker,
                planner,
            )
            overall = aggregate_overall(results, MODES)
            out["benches"][nm] = {
                "n": overall["n"],
                "overall": overall,
                "per_q": results,
            }
            print(f"  {'mode':12s}  {'R@1':>16s}", flush=True)
            for m in MODES:
                d = overall[m]
                print(
                    f"  {m:12s}  R@1={d['R@1']:.3f} ({d['r1_count']}/{overall['n']})",
                    flush=True,
                )
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][nm] = {"error": str(e), "n": 0}

    out["planner_stats"] = planner.stats()
    print(f"\nplanner stats: {out['planner_stats']}", flush=True)

    # Macro per mode
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
        print(f"  {m:8s}: {macro[m]:.3f}")
    out["macro"] = macro

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"K{POOL_CAP}" + ("_strict" if STRICT_POOL else "")
    json_path = out_dir / f"T_q1_retrieval_ablation_{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
