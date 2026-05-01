"""composition_eval_v3: simplified TimeWindow planner stack.

Drops from v2:
  - separate `absolute_anchor` / `open_ended` / `negation` schema fields
    (now: a single TimeWindow list with op/open_lower/open_upper)
  - causal stack entirely (multi-hop event resolution; not temporal
    retrieval per se)
  - earliest_intent (folded into `extremum`)
  - normalize_plan post-processor (no more absolute_anchor leak)

Stack comparison:
  - rerank_only       : pure cross-encoder rerank baseline
  - regex_stack       : same as composition_eval_v2 (kept for regression)
  - planner_v2_stack  : NEW — windows applied multiplicatively + recency
                         multiplier when extremum=="latest"

Scoring (planner_v2_stack):
  T_router = T_v5 if any window has open_lower or open_upper else T_lblend
  base     = score_blend({T_router, R}, {0.4, 0.6}, CV gate); normalized
  for each include window:
    if window has valid anchor intervals:
      base *= window_factor(doc, window)  # hard 1/0 mask
  for each exclude window:
    if window has valid anchor intervals:
      base *= (1 - excluded_containment(doc, window))
  if extremum == "latest":
    base *= (1 + α * recency_score(doc))
  if extremum == "earliest":
    base *= (1 + α * (1 - recency_score(doc)))
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

from force_pick_optimizers_eval import (
    RERANK_TOP_K,
    make_t_scores,
    merge_with_tail,
    rerank_topk,
    topk_from_scores,
)
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from negation import (
    apply_signed,
    excluded_containment,
    has_negation_cue,
    parse_negation_query,
)
from query_planner_v2 import QueryPlan, QueryPlanner
from rag_fusion import score_blend
from recency import (
    has_recency_cue,
    lambda_for_half_life,
    recency_scores_for_docs,
)
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
from T_causal_eval import (
    CAUSAL_SIGNED_LAMBDA,
    causal_signed_scores,
    cue_direction,
    detect_causal,
    resolve_anchor,
)
from T_open_ended_router_eval import has_open_ended_cue
from t_v5_eval import per_te_bundles_v5, t_v5_doc_scores

HALF_LIFE_DAYS = 21.0
CV_REF = 0.20
W_T_FUSE_TR = 0.4
ADDITIVE_ALPHA = 0.5
LAM_NEG_SIGNED = 1.0

# planner_v2_stack scoring constants
EXTREMUM_MULT_ALPHA = 3.0  # multiplicative recency boost: score *= (1 + α·rec_lin).
# α=3 lets a recency-leading mid-base gold
# outrank a high-base low-recency distractor.
EXTREMUM_TOPK = 15  # top-K rerank/T docs as the candidate set
# for within-set rec_lin normalization when
# the query has no include window. Avoids
# global-recency contamination.
WINDOW_OUTSIDE_FACTOR = 0.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fuse_T_R_blend_scores(t_scores, r_scores, w_T=W_T_FUSE_TR):
    fused = score_blend(
        {"T": t_scores, "R": r_scores},
        {"T": w_T, "R": 1.0 - w_T},
        top_k_per=40,
        dispersion_cv_ref=CV_REF,
    )
    return dict(fused)


def additive_with_recency(base, rec, cue, alpha=ADDITIVE_ALPHA):
    if not cue:
        return dict(base)
    out = {}
    docs = set(base) | set(rec)
    for d in docs:
        out[d] = (1.0 - alpha) * base.get(d, 0.0) + alpha * rec.get(d, 0.0)
    return out


def rank_from_scores(scores):
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


def normalize_rerank_full(rerank_partial, all_doc_ids, tail_score=0.0):
    if not rerank_partial:
        return dict.fromkeys(all_doc_ids, tail_score)
    vals = list(rerank_partial.values())
    rmin, rmax = min(vals), max(vals)
    span = (rmax - rmin) or 1.0
    out = {}
    for did in all_doc_ids:
        if did in rerank_partial:
            out[did] = (rerank_partial[did] - rmin) / span
        else:
            out[did] = tail_score
    return out


def normalize_dict(d):
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in d.items()}


def linear_recency_scores(doc_bundles_for_rec, doc_ref_us):
    """Linear recency: each doc gets (best_anchor_us - min) / (max - min).

    The exp(-λ·Δt) recency in `recency.py` saturates to 0 once docs are
    older than ~3 half-lives, which kills within-window differentiation
    for queries like "latest in Q2 2024" when ref_time is much later.
    Linear normalization across docs preserves order regardless of how
    far back the window sits.

    Doc anchor = MAX best_us across the doc's TEs (latest TE point),
    falling back to doc.ref_time when the doc has no TE.
    """
    anchors = {}
    for did in doc_ref_us:
        bundles = doc_bundles_for_rec.get(did, [])
        best = None
        for b in bundles:
            for iv in b.get("intervals", []) or []:
                cand = (
                    iv.best_us
                    if iv.best_us is not None
                    else ((iv.earliest_us + iv.latest_us) // 2)
                )
                if best is None or cand > best:
                    best = cand
        anchors[did] = best if best is not None else doc_ref_us[did]
    if not anchors:
        return {}
    vals = list(anchors.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {did: (a - lo) / span for did, a in anchors.items()}


# -----------------------------------------------------------------------------
# Constraint scoring
# -----------------------------------------------------------------------------
def constraint_factor_for_doc(doc_intervals, anchor_intervals, direction: str) -> float:
    """Return 1.0 if doc has any TE satisfying the constraint, 0.0 otherwise.

    - "in":     doc TE overlaps anchor
    - "after":  doc TE has any time strictly past anchor.latest
    - "before": doc TE has any time strictly before anchor.earliest
    """
    if not anchor_intervals:
        return 1.0
    a_e = min(ai.earliest_us for ai in anchor_intervals)
    a_l = max(ai.latest_us for ai in anchor_intervals)
    for di in doc_intervals or []:
        if direction == "after":
            if di.latest_us > a_l:
                return 1.0
        elif direction == "before":
            if di.earliest_us < a_e:
                return 1.0
        else:  # "in" (closed overlap)
            for ai in anchor_intervals:
                if di.earliest_us <= ai.latest_us and ai.earliest_us <= di.latest_us:
                    return 1.0
    return 0.0


# -----------------------------------------------------------------------------
# Per-bench evaluation
# -----------------------------------------------------------------------------
async def run_bench(
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

    # ---- Plan all queries -------------------------------------------
    print(f"  planning ({len(queries)} queries)...", flush=True)
    plan_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    plans = await planner.plan_many(plan_items)

    # ---- Window-phrase extractions (one extraction per unique phrase) -----
    # Gather all phrases keyed by (qid, idx, op, open_lower, open_upper)
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

    # ---- Regex parses (negation) for the regex_stack -----
    parsed_meta = {}
    pos_items_regex = []
    excl_items_regex = []
    for q in queries:
        qid = q["query_id"]
        ref = parse_iso(q["ref_time"])
        text = q["text"]
        cue = has_negation_cue(text)
        if cue:
            pos_q, excl_q = parse_negation_query(text)
        else:
            pos_q, excl_q = text, None
        parsed_meta[qid] = (cue, pos_q, excl_q)
        pos_items_regex.append((f"{qid}__pos", pos_q, ref))
        if cue and excl_q:
            excl_items_regex.append((f"{qid}__excl", excl_q, ref))
    pos_ext_regex = await run_v2_extract(
        pos_items_regex,
        f"{name}-pos",
        f"{cache_label}-pos",
    )
    excl_ext_regex = (
        await run_v2_extract(
            excl_items_regex,
            f"{name}-excl",
            f"{cache_label}-excl",
        )
        if excl_items_regex
        else {}
    )

    # ---- Doc memory + lattice (T_lblend) ------------------------------
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

    lat_db = ROOT / "cache" / "composition_v3" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)

    # ---- Doc intervals flat ---
    doc_ivs_flat = {}
    for did, tes in doc_ext.items():
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        doc_ivs_flat[did] = ivs
    for d in docs:
        doc_ivs_flat.setdefault(d["doc_id"], [])

    # Recency anchor bundles
    doc_bundles_for_rec = {}
    for did, mem in doc_mem.items():
        ivs = mem.get("intervals") or []
        doc_bundles_for_rec[did] = [{"intervals": ivs}] if ivs else []

    # Linear (rank-style) recency: normalized [0,1] across all docs by
    # latest-TE timestamp (or doc.ref_time fallback). Used by the planner_v2
    # extremum boost so within-window ordering survives even when the
    # window sits a year or more before query ref_time.
    rec_lin_scores = linear_recency_scores(doc_bundles_for_rec, doc_ref_us)

    # ---- Embeddings ---------------------------------------------------
    print("  embedding...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    pos_q_texts_regex = [parsed_meta[q["query_id"]][1] for q in queries]
    pos_q_embs_arr = await embed_all(pos_q_texts_regex)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    pos_q_embs = {q["query_id"]: pos_q_embs_arr[i] for i, q in enumerate(queries)}

    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}
    per_q_s_pos = {qid: rank_semantic(qid, pos_q_embs, doc_embs) for qid in qids}

    # T_lblend per query (full query)
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

    # T_lblend on positive (regex) query
    per_q_l_pos = {
        qid: (
            lattice_retrieve_multi(
                lat, pos_ext_regex.get(f"{qid}__pos", []), down_levels=1
            )[0]
            if pos_ext_regex.get(f"{qid}__pos")
            else {}
        )
        for qid in qids
    }
    pos_mem_regex = build_memory(
        {qid: pos_ext_regex.get(f"{qid}__pos", []) for qid in qids}
    )
    per_q_t_pos = {
        qid: make_t_scores(
            pos_mem_regex.get(
                qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}
            ),
            doc_mem,
            per_q_l_pos.get(qid, {}),
        )
        for qid in qids
    }
    for qid in qids:
        for d in docs:
            per_q_t_pos[qid].setdefault(d["doc_id"], 0.0)

    # T_v5 per query (open-ended router)
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

    # ---- Rerank: union(top-50 sem, top-50 T_lblend, top-50 T_v5) -----
    print("  reranking...", flush=True)
    per_q_r_full = {}
    per_q_r_partial = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        v5_top = topk_from_scores(per_q_tv5[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top + v5_top))[: int(RERANK_TOP_K * 1.5)]
        rs = await rerank_topk(reranker, q_text[qid], union, doc_text, len(union))
        per_q_r_partial[qid] = rs
        per_q_r_full[qid] = normalize_rerank_full(rs, [d["doc_id"] for d in docs], 0.0)

    # ---- Causal anchor resolution (regex) -----------------------------
    causal_info_regex = {}
    anchor_phrases_regex = []
    for q in queries:
        info = detect_causal(q["text"])
        if info is None:
            continue
        cue, phrase = info
        causal_info_regex[q["query_id"]] = {
            "cue": cue,
            "phrase": phrase,
            "direction": cue_direction(cue),
        }
        anchor_phrases_regex.append(phrase)
    unique_phrases_regex = list(dict.fromkeys(anchor_phrases_regex))
    if unique_phrases_regex:
        ph_embs = await embed_all(unique_phrases_regex)
        phrase_emb_regex = {p: ph_embs[i] for i, p in enumerate(unique_phrases_regex)}
    else:
        phrase_emb_regex = {}
    anchor_resolution_regex = {}
    for qid, info in causal_info_regex.items():
        emb = phrase_emb_regex.get(info["phrase"])
        if emb is None:
            continue
        res = resolve_anchor(info["phrase"], emb, doc_embs)
        if res is None:
            continue
        did, sim = res
        anchor_resolution_regex[qid] = {
            "anchor_did": did,
            "anchor_us": doc_ref_us[did],
            "direction": info["direction"],
            "phrase": info["phrase"],
            "sim": sim,
        }

    lam_rec = lambda_for_half_life(HALF_LIFE_DAYS)

    # =================================================================
    # Per-query evaluation
    # =================================================================
    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        text = q["text"]
        ctype = q_type.get(qid, "?")

        plan = plans.get(qid) or QueryPlan()

        # ---- Regex cue detection (for regex_stack) --------------------
        rec_active_re = has_recency_cue(text)
        neg_active_re, _, _ = parsed_meta[qid]
        oe_active_re = has_open_ended_cue(text)
        causal_re = qid in anchor_resolution_regex
        ar_re = anchor_resolution_regex.get(qid)

        # ---- Plan v2 cue detection -----------------------------------
        plan_has_open = plan.has_open_constraint
        plan_includes = plan.includes
        plan_excludes = plan.excludes
        plan_latest = plan.latest_intent
        plan_earliest = plan.earliest_intent

        # ---- Common precomputed scores --------------------------------
        rec_scores = recency_scores_for_docs(
            doc_bundles_for_rec,
            doc_ref_us,
            q_ref_us[qid],
            lam_rec,
        )
        r_full = per_q_r_full[qid]
        rerank_partial = per_q_r_partial[qid]
        s_scores = per_q_s[qid]
        s_pos_scores = per_q_s_pos[qid]

        # Excluded containment (regex)
        excl_te_re = excl_ext_regex.get(f"{qid}__excl", []) if neg_active_re else []
        excl_ivs_re = []
        for te in excl_te_re:
            excl_ivs_re.extend(flatten_intervals(te))
        excl_cont_re = {
            did: excluded_containment(doc_ivs_flat.get(did, []), excl_ivs_re)
            for did in doc_ref_us
        }

        # ============================================================
        # rerank_only baseline
        # ============================================================
        rerank_only_rank = merge_with_tail(
            [
                d
                for d, _ in sorted(
                    rerank_partial.items(), key=lambda x: x[1], reverse=True
                )
            ],
            s_scores,
        )

        # ============================================================
        # regex_stack (S1 sequential)
        # ============================================================
        t_router_re = per_q_tv5[qid] if oe_active_re else per_q_t[qid]
        fused_TR_re = fuse_T_R_blend_scores(t_router_re, r_full, w_T=W_T_FUSE_TR)
        rs_re = additive_with_recency(
            fused_TR_re, rec_scores, rec_active_re, ADDITIVE_ALPHA
        )
        if causal_re and ar_re:
            rs_re = causal_signed_scores(
                rs_re,
                doc_ref_us,
                ar_re["anchor_us"],
                ar_re["direction"],
                ar_re["anchor_did"],
                lam=CAUSAL_SIGNED_LAMBDA,
            )
        if neg_active_re and excl_ivs_re:
            positive_composite = {
                did: 0.7 * s_pos_scores.get(did, 0.0)
                + 0.3 * per_q_t_pos[qid].get(did, 0.0)
                for did in doc_ref_us
            }
            rs_re = apply_signed(positive_composite, excl_cont_re, lam=LAM_NEG_SIGNED)
        rank_regex = rank_from_scores(rs_re)
        rank_regex = rank_regex + [
            d for d in rank_from_scores(s_scores) if d not in set(rank_regex)
        ]

        # ============================================================
        # planner_v2_stack (windows + extremum)
        # ============================================================
        # T_router: T_v5 if any open window, else T_lblend.
        t_router_p = per_q_tv5[qid] if plan_has_open else per_q_t[qid]
        fused_TR_p = fuse_T_R_blend_scores(t_router_p, r_full, w_T=W_T_FUSE_TR)
        base_p = normalize_dict(fused_TR_p)

        # Per-constraint anchor extraction; split by direction.
        # Includes (in/after/before) compose as OR (max): a doc passes
        # if it satisfies ANY include constraint. "in March and April"
        # means either is fine.
        # Excludes (not_in) compose as AND-not (product of 1-cont): a
        # doc must avoid ALL exclude constraints.
        valid_includes = []  # (constraint, anchor_ivs)
        valid_excludes = []
        for i, c in enumerate(plan.constraints):
            tes = win_ext.get(f"{qid}__c{i}", [])
            anchor_ivs = []
            for te in tes:
                anchor_ivs.extend(flatten_intervals(te))
            if not anchor_ivs:
                # Non-date phrase that slipped past the LLM filter — skip.
                continue
            if c.direction == "not_in":
                valid_excludes.append((c, anchor_ivs))
            else:
                valid_includes.append((c, anchor_ivs))

        if valid_includes:
            include_factors_per_doc = dict.fromkeys(doc_ref_us, 0.0)
            for did in doc_ref_us:
                for c, anchor_ivs in valid_includes:
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
        for c, anchor_ivs in valid_excludes:
            for did in doc_ref_us:
                cont = excluded_containment(
                    doc_ivs_flat.get(did, []),
                    anchor_ivs,
                )
                exclude_factors_per_doc[did] *= max(0.0, 1.0 - cont)

        # Within-set linear recency: normalize ONLY across docs that pass
        # the include mask. A globally-normalized recency over a tight
        # window (e.g. Q4 2023 within a 5-year corpus) collapses to a
        # 0.05-wide range and stops differentiating; per-query
        # normalization keeps the [0,1] span.
        if plan_latest or plan_earliest:
            # Candidate set:
            #   - With windows: the include-mask survivors.
            #   - Without windows: top-K topical docs by rerank/T (small
            #     enough that within-set recency picks the topical recent,
            #     not the global recent).
            if valid_includes:
                in_set = [
                    did
                    for did in doc_ref_us
                    if include_factors_per_doc.get(did, 1.0) >= 0.5
                    and exclude_factors_per_doc.get(did, 1.0) > 0.0
                ]
            else:
                in_set = sorted(base_p, key=base_p.get, reverse=True)[:EXTREMUM_TOPK]
            if in_set and len(in_set) >= 2:
                rec_lin = linear_recency_scores(
                    {did: doc_bundles_for_rec.get(did, []) for did in in_set},
                    {did: doc_ref_us[did] for did in in_set},
                )
            else:
                rec_lin = rec_lin_scores
        else:
            rec_lin = {}

        # Single multiplicative regime: base × mask × (1 + α·rec).
        # The candidate-set choice (include-mask survivors with windows;
        # top-K rerank docs without) keeps rec_lin's normalization tied
        # to the topical neighborhood, so the multiplicative boost works
        # the same way in both cases.
        rs_p = {}
        for did in doc_ref_us:
            mask = include_factors_per_doc.get(did, 1.0) * exclude_factors_per_doc.get(
                did, 1.0
            )
            base_m = base_p.get(did, 0.0) * mask
            if plan_latest or plan_earliest:
                r = rec_lin.get(did, 0.0)
                if plan_earliest:
                    r = 1.0 - r
                v = base_m * (1.0 + EXTREMUM_MULT_ALPHA * r)
            else:
                v = base_m
            rs_p[did] = v
        rank_p = rank_from_scores(rs_p)
        rank_p = rank_p + [
            d for d in rank_from_scores(s_scores) if d not in set(rank_p)
        ]

        h_ro = hit_rank(rerank_only_rank, gold_set)
        h_re = hit_rank(rank_regex, gold_set)
        h_p = hit_rank(rank_p, gold_set)

        results.append(
            {
                "qid": qid,
                "type": ctype,
                "qtext": text,
                "gold": list(gold_set),
                "plan": plan.to_dict(),
                "regex_rec_active": rec_active_re,
                "regex_neg_active": neg_active_re,
                "regex_oe_active": oe_active_re,
                "regex_causal": causal_re,
                "regex_anchor_did": ar_re["anchor_did"] if ar_re else None,
                "plan_has_open": plan_has_open,
                "plan_n_includes": len(plan_includes),
                "plan_n_excludes": len(plan_excludes),
                "plan_extremum": plan.extremum,
                "rerank_only": h_ro,
                "regex_stack": h_re,
                "planner_v2_stack": h_p,
                "top5_regex": rank_regex[:5],
                "top5_planner_v2": rank_p[:5],
            }
        )

    return results


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def aggregate_overall(results, variants):
    n = len(results)
    out = {"n": n}
    for v in variants:
        ranks = [r[v] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr_v = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
        out[v] = {
            "R@1": r1 / n if n else 0.0,
            "R@5": r5 / n if n else 0.0,
            "MRR": mrr_v,
            "r1_count": r1,
            "r5_count": r5,
        }
    return out


def aggregate_per_type(results, variants):
    by_type = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    out = {}
    for typ, rs in by_type.items():
        out[typ] = aggregate_overall(rs, variants)
    out["ALL"] = aggregate_overall(results, variants)
    return out


# -----------------------------------------------------------------------------
# Markdown writer
# -----------------------------------------------------------------------------
def write_md(report, path):
    benches = report["benches"]
    comp = benches.get("composition")
    planner_stats = report["planner_stats"]
    variants = ["rerank_only", "regex_stack", "planner_v2_stack"]

    valid = [k for k, v in benches.items() if "error" not in v and v.get("n", 0) > 0]
    macro = {
        v: sum(benches[k]["overall"][v]["R@1"] for k in valid) / max(1, len(valid))
        for v in variants
    }

    lines = []
    lines.append("# T_planner_v2 — simplified TimeWindow planner\n")
    lines.append(
        "Schema collapse: absolute_anchor + open_ended + negation -> single "
        "TimeWindow list with op/open_lower/open_upper. Drops causal "
        "(multi-hop), drops earliest_intent (folded into extremum), drops "
        "normalize_plan post-processor (no more absolute_anchor leak).\n"
    )

    # ---- Composition lead ----
    lines.append("## Composition R@1 — lead\n")
    if comp and "error" not in comp:
        per_type = comp["per_type"]
        lines.append(
            "| Type | n | rerank_only | regex_stack | **planner_v2_stack** | Δ(planner_v2 − regex) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        type_label = {
            "A": "A: recency × absolute",
            "B": "B: negation × absolute",
            "C": "C: causal × recency",
            "D": "D: causal × absolute",
            "E": "E: open_ended × negation",
            "ALL": "ALL",
        }
        for t in ["A", "B", "C", "D", "E", "ALL"]:
            if t not in per_type:
                continue
            a = per_type[t]
            n = a["n"]
            ro = a["rerank_only"]["R@1"]
            re_ = a["regex_stack"]["R@1"]
            pl = a["planner_v2_stack"]["R@1"]
            d = pl - re_
            lines.append(
                f"| {type_label.get(t, t)} | {n} | "
                f"{ro:.3f} ({a['rerank_only']['r1_count']}/{n}) | "
                f"{re_:.3f} ({a['regex_stack']['r1_count']}/{n}) | "
                f"**{pl:.3f}** ({a['planner_v2_stack']['r1_count']}/{n}) | "
                f"**{d:+.3f}** |"
            )
        lines.append("")

    # ---- Single-cue regression check ----
    lines.append("## Single-cue regression check (R@1)\n")
    lines.append(
        "| Benchmark | n | rerank_only | regex_stack | planner_v2_stack | Δ(p2 − regex) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k in benches:
        b = benches[k]
        if "error" in b:
            lines.append(f"| {k} | err | - | - | - | - |")
            continue
        if k == "composition":
            continue
        n = b["overall"]["n"]
        ro = b["overall"]["rerank_only"]["R@1"]
        re_ = b["overall"]["regex_stack"]["R@1"]
        pl = b["overall"]["planner_v2_stack"]["R@1"]
        d = pl - re_
        flag = ""
        if d > 0.005:
            flag = " (+)"
        elif d < -0.005:
            flag = " (regress!)"
        lines.append(
            f"| {k} | {n} | "
            f"{ro:.3f} ({b['overall']['rerank_only']['r1_count']}/{n}) | "
            f"{re_:.3f} ({b['overall']['regex_stack']['r1_count']}/{n}) | "
            f"{pl:.3f} ({b['overall']['planner_v2_stack']['r1_count']}/{n}) | "
            f"**{d:+.3f}**{flag} |"
        )
    lines.append("")
    lines.append(f"### Macro-average R@1 across {len(valid)} benches\n")
    lines.append(f"- rerank_only: {macro['rerank_only']:.3f}")
    lines.append(f"- regex_stack: {macro['regex_stack']:.3f}")
    lines.append(f"- **planner_v2_stack**: **{macro['planner_v2_stack']:.3f}**")
    lines.append(
        f"- Δ(p2 − regex): **{macro['planner_v2_stack'] - macro['regex_stack']:+.3f}**\n"
    )

    # ---- Cost ----
    lines.append("## Planner cost\n")
    lines.append(f"- Model: `{planner_stats['model']}`")
    lines.append(f"- Total queries: {planner_stats['total_queries']}")
    lines.append(f"- Live calls: {planner_stats['calls']}")
    lines.append(f"- Cache hits: {planner_stats['cache_hits']}")
    lines.append(f"- Hit rate: {planner_stats['cache_hit_rate'] * 100:.1f}%")
    lines.append(f"- Parse failures: {planner_stats['parse_failures']}\n")

    # ---- Failure analysis ----
    lines.append("## Failure analysis\n")
    if comp and "error" not in comp:
        per_q = comp["per_q"]
        lines.append("### composition\n")
        for r in per_q:
            if r["planner_v2_stack"] == 1:
                continue
            rank = r["planner_v2_stack"]
            t = r["type"]
            lines.append(f"#### {r['qid']} — Type {t}, p2 rank={rank}")
            lines.append(f"- Query: `{r['qtext']}`")
            lines.append(f"- Gold: `{r['gold']}`")
            lines.append(f"- Plan: `{json.dumps(r['plan'])}`")
            lines.append(f"- p2 top5: `{r['top5_planner_v2']}`")
            lines.append(
                f"- regex top5 (regex_rank={r['regex_stack']}): `{r['top5_regex']}`"
            )
            lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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

    variants = ["rerank_only", "regex_stack", "planner_v2_stack"]
    out = {"benches": {}}
    for nm, dp, qp, gp, cl in benches_def:
        try:
            results = await run_bench(nm, dp, qp, gp, cl, reranker, planner)
            if nm == "composition":
                per_type = aggregate_per_type(results, variants)
                overall = per_type["ALL"]
                out["benches"][nm] = {
                    "n": overall["n"],
                    "per_type": per_type,
                    "overall": overall,
                    "per_q": results,
                }
            else:
                overall = aggregate_overall(results, variants)
                out["benches"][nm] = {
                    "n": overall["n"],
                    "overall": overall,
                    "per_q": results,
                }
            for v in variants:
                d = out["benches"][nm]["overall"][v]
                print(
                    f"  {v:24s} R@1={d['R@1']:.3f} ({d['r1_count']}/{out['benches'][nm]['n']})",
                    flush=True,
                )
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][nm] = {"error": str(e), "n": 0}

    out["planner_stats"] = planner.stats()
    print(f"\nplanner stats: {out['planner_stats']}", flush=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_planner_v2.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {json_path}", flush=True)

    md_path = out_dir / "T_planner_v2.md"
    md_path.write_text(write_md(out, md_path))
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
