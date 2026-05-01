"""composition_eval_v2: LLM-planner-dispatched composition stack.

Three stacks compared on the composition benchmark + the 11 single-cue
benchmarks (regression check):

  - rerank_only         : pure cross-encoder rerank baseline
  - regex_stack (S1)    : current production — fuse_T_R + recency_additive
                          + open_ended_router (T_v5) + negation strip
                          + causal_signed mask (sequential overrides as in
                          composition_eval.py's S1)
  - llm_planner_stack   : single gpt-5-mini call -> QueryPlan; dispatch
                          modules from the plan; compose multiplicatively
                          (S2). New: absolute_anchor adds a hard interval
                          filter on the doc set.

For absolute_anchor the planner emits a date phrase. We pass that phrase
through the existing v2 extractor to obtain its FuzzyInterval list, then
apply an `absolute_filter` step that zeros docs whose intervals do not
overlap any of the planner's anchor intervals.

Outputs:
  - results/T_llm_planner.json
  - results/T_llm_planner.md
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Strip proxy env vars set by sandbox.
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


# ---------------------------------------------------------------------------
# Plan post-processing: fix two known LLM-planner failure modes on
# open_ended_date queries.
#   (A) Spurious causal alongside open_ended: the prompt says open_ended
#       and causal are mutually exclusive, but gpt-5-mini still emits both
#       on event-attached date queries ("after I joined Acme in March
#       2022"). Suppress causal when open_ended is set.
#   (B) Plan emits absolute_anchor where the regex would correctly fire
#       open_ended ("after my child was born in August 2023" -> abs=Aug
#       2023). When has_open_ended_cue(query) regex fires and the plan has
#       absolute_anchor but no open_ended, flip: convert the absolute
#       anchor into the open_ended anchor (with side parsed from the
#       query) and zero absolute_anchor.
# ---------------------------------------------------------------------------
import re as _re

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
from query_planner import QueryPlan, QueryPlanner
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
    direction_match,
    resolve_anchor,
)
from T_open_ended_router_eval import has_open_ended_cue
from t_v5_eval import per_te_bundles_v5, t_v5_doc_scores

_SIDE_RE = _re.compile(
    r"\b(after|before|since|until|prior\s+to|post|pre)\b",
    _re.IGNORECASE,
)


def _infer_oe_side(query_text: str) -> str | None:
    m = _SIDE_RE.search(query_text)
    if not m:
        return None
    cue = m.group(1).lower()
    if cue in ("after", "since", "post"):
        return "after"
    if cue in ("before", "until", "prior to", "pre"):
        return "before"
    # cue == "since" or "until" — keep as-is (planner schema accepts these).
    return cue


def normalize_plan(plan: QueryPlan, query_text: str) -> QueryPlan:
    """Fix LLM-planner over-firing on date-bound queries."""
    # (B) absolute_anchor leak when regex sees open_ended cue.
    if (
        plan.absolute_anchor
        and plan.open_ended is None
        and has_open_ended_cue(query_text)
    ):
        side = _infer_oe_side(query_text) or "after"
        plan.open_ended = {"side": side, "anchor": plan.absolute_anchor}
        plan.absolute_anchor = None
    # (A) Mutual exclusion: open_ended dominates causal (per prompt rule 3).
    if plan.open_ended is not None and plan.causal is not None:
        plan.causal = None
    return plan


HALF_LIFE_DAYS = 21.0
CV_REF = 0.20
W_T_FUSE_TR = 0.4
ADDITIVE_ALPHA = 0.5
LAM_NEG_SIGNED = 1.0

# Multiplicative composition constants
REC_MULT_ALPHA = 1.0
DIR_WRONG_FACTOR = 0.1
ABSOLUTE_OUTSIDE_FACTOR = 0.0


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


def direction_match_scores(doc_ref_us, anchor_us, direction, anchor_did):
    out = {}
    for did, dus in doc_ref_us.items():
        if did == anchor_did:
            out[did] = 0.0
        elif direction_match(dus, anchor_us, direction):
            out[did] = 1.0
        else:
            out[did] = 0.0
    return out


def absolute_overlap_scores(doc_ivs_flat, anchor_ivs):
    """1.0 if doc has any interval overlapping the anchor; 0.0 otherwise."""
    out = {}
    if not anchor_ivs:
        return dict.fromkeys(doc_ivs_flat, 1.0)
    for did, divs in doc_ivs_flat.items():
        ok = 0.0
        for di in divs or []:
            for ai in anchor_ivs:
                if di.earliest_us <= ai.latest_us and ai.earliest_us <= di.latest_us:
                    ok = 1.0
                    break
            if ok > 0:
                break
        out[did] = ok
    return out


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

    # ---- Plan all queries via the LLM (single call/query, cached) -----
    print(f"  planning ({len(queries)} queries)...", flush=True)
    plan_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    plans = await planner.plan_many(plan_items)
    # Post-process every plan to fix the two known open_ended_date failure
    # modes (causal mutex and absolute_anchor leak).
    q_text_by_id = {q["query_id"]: q["text"] for q in queries}
    plans = {
        qid: normalize_plan(p, q_text_by_id.get(qid, "")) for qid, p in plans.items()
    }

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

    # ---- LLM-planner-driven extractions:
    #   * absolute_anchor phrase -> intervals (for hard window filter)
    #   * negation.excluded_phrase -> intervals
    #   * causal.anchor_phrase -> embedding -> top-1 cosine doc
    abs_items = []
    excl_items_llm = []
    for q in queries:
        qid = q["query_id"]
        ref = parse_iso(q["ref_time"])
        plan = plans.get(qid)
        if plan and plan.absolute_anchor:
            abs_items.append((f"{qid}__abs", plan.absolute_anchor, ref))
        if plan and plan.negation and plan.negation.get("excluded_phrase"):
            excl_items_llm.append(
                (f"{qid}__excl_llm", plan.negation["excluded_phrase"], ref)
            )
    abs_ext = (
        await run_v2_extract(abs_items, f"{name}-abs", f"{cache_label}-abs")
        if abs_items
        else {}
    )
    excl_ext_llm = (
        await run_v2_extract(
            excl_items_llm, f"{name}-excl_llm", f"{cache_label}-excl_llm"
        )
        if excl_items_llm
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

    lat_db = ROOT / "cache" / "composition_v2" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)

    # ---- Doc intervals flat (negation containment + absolute filter) ---
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

    # ---- Causal anchor resolution (LLM) -------------------------------
    causal_phrases_llm = []
    causal_meta_llm = {}
    for q in queries:
        plan = plans.get(q["query_id"])
        if plan is None or plan.causal is None:
            continue
        phrase = plan.causal["anchor_phrase"]
        direction = plan.causal["direction"]
        causal_meta_llm[q["query_id"]] = {"phrase": phrase, "direction": direction}
        causal_phrases_llm.append(phrase)
    unique_phrases_llm = list(dict.fromkeys(causal_phrases_llm))
    if unique_phrases_llm:
        ph_embs_llm = await embed_all(unique_phrases_llm)
        phrase_emb_llm = {p: ph_embs_llm[i] for i, p in enumerate(unique_phrases_llm)}
    else:
        phrase_emb_llm = {}
    anchor_resolution_llm = {}
    for qid, info in causal_meta_llm.items():
        emb = phrase_emb_llm.get(info["phrase"])
        if emb is None:
            continue
        res = resolve_anchor(info["phrase"], emb, doc_embs)
        if res is None:
            continue
        did, sim = res
        anchor_resolution_llm[qid] = {
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

        # ---- LLM-plan cue detection (for llm_planner_stack) -----------
        rec_active_llm = bool(plan.recency_intent)
        oe_active_llm = plan.open_ended is not None
        neg_active_llm = plan.negation is not None
        causal_llm = qid in anchor_resolution_llm
        abs_active_llm = bool(plan.absolute_anchor)
        ar_llm = anchor_resolution_llm.get(qid)

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

        # Excluded containment (regex): from regex parse_negation_query
        excl_te_re = excl_ext_regex.get(f"{qid}__excl", []) if neg_active_re else []
        excl_ivs_re = []
        for te in excl_te_re:
            excl_ivs_re.extend(flatten_intervals(te))
        excl_cont_re = {
            did: excluded_containment(doc_ivs_flat.get(did, []), excl_ivs_re)
            for did in doc_ref_us
        }

        # Excluded containment (llm): from plan.negation.excluded_phrase
        excl_te_llm = excl_ext_llm.get(f"{qid}__excl_llm", []) if neg_active_llm else []
        excl_ivs_llm = []
        for te in excl_te_llm:
            excl_ivs_llm.extend(flatten_intervals(te))
        excl_cont_llm = {
            did: excluded_containment(doc_ivs_flat.get(did, []), excl_ivs_llm)
            for did in doc_ref_us
        }

        # Absolute anchor intervals (llm only): hard window from plan.absolute_anchor
        # PATCH: skip absolute_filter when open_ended is also active — the planner
        # sometimes fills both fields with the bound year for "after 2020", and the
        # hard window then masks the gold which is by-construction outside it.
        suppress_abs_llm = abs_active_llm and oe_active_llm
        abs_te_llm = (
            abs_ext.get(f"{qid}__abs", [])
            if abs_active_llm and not suppress_abs_llm
            else []
        )
        abs_ivs_llm = []
        for te in abs_te_llm:
            abs_ivs_llm.extend(flatten_intervals(te))
        abs_overlap_llm = absolute_overlap_scores(doc_ivs_flat, abs_ivs_llm)

        # Direction-match (regex)
        if causal_re and ar_re:
            dir_match_re = direction_match_scores(
                doc_ref_us,
                ar_re["anchor_us"],
                ar_re["direction"],
                ar_re["anchor_did"],
            )
        else:
            dir_match_re = dict.fromkeys(doc_ref_us, 1.0)
        # Direction-match (llm)
        if causal_llm and ar_llm:
            dir_match_llm = direction_match_scores(
                doc_ref_us,
                ar_llm["anchor_us"],
                ar_llm["direction"],
                ar_llm["anchor_did"],
            )
        else:
            dir_match_llm = dict.fromkeys(doc_ref_us, 1.0)

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
        # llm_planner_stack (S2 multiplicative)
        #
        # base = fused_TR(T_router_llm, R) where T_router = T_v5 if oe else T_lblend
        # final = base * (1 + α*rec)             [if recency_intent]
        #              * (1 - excl_cont)         [if negation]
        #              * dir_factor              [if causal: 1.0 right / 0.1 wrong / 0.0 anchor]
        #              * abs_overlap             [if absolute_anchor: 1.0 in / 0.0 out]
        # ============================================================
        t_router_llm = per_q_tv5[qid] if oe_active_llm else per_q_t[qid]
        fused_TR_llm = fuse_T_R_blend_scores(t_router_llm, r_full, w_T=W_T_FUSE_TR)
        base_llm = normalize_dict(fused_TR_llm)

        # Count active cues besides causal to decide causal scoring strategy.
        # When ≥1 OTHER cue fires alongside causal (composition queries),
        # the multiplicative mask captures Type-D wins. When causal alone
        # (often entity-anchored tempreason queries), signed-additive is
        # robust to fuzzy anchor matches that the mask collapses.
        other_cues_active = (
            rec_active_llm or neg_active_llm or abs_active_llm or oe_active_llm
        )
        causal_use_mask = causal_llm and ar_llm and other_cues_active

        rs_llm = {}
        for did in doc_ref_us:
            v = base_llm.get(did, 0.0)
            if rec_active_llm:
                v = v * (1.0 + REC_MULT_ALPHA * rec_scores.get(did, 0.0))
            if neg_active_llm and excl_ivs_llm:
                v = v * max(0.0, 1.0 - excl_cont_llm.get(did, 0.0))
            if abs_active_llm and abs_ivs_llm:
                v = v * (
                    1.0
                    if abs_overlap_llm.get(did, 0.0) >= 0.5
                    else ABSOLUTE_OUTSIDE_FACTOR
                )
            if causal_use_mask:
                if did == ar_llm["anchor_did"]:
                    v = 0.0
                elif dir_match_llm.get(did, 1.0) < 0.5:
                    v = v * DIR_WRONG_FACTOR
            rs_llm[did] = v
        # Causal-alone: signed-additive (robust on fuzzy anchors)
        if causal_llm and ar_llm and not other_cues_active:
            rs_llm = causal_signed_scores(
                rs_llm,
                doc_ref_us,
                ar_llm["anchor_us"],
                ar_llm["direction"],
                ar_llm["anchor_did"],
                lam=CAUSAL_SIGNED_LAMBDA,
            )
        rank_llm = rank_from_scores(rs_llm)
        rank_llm = rank_llm + [
            d for d in rank_from_scores(s_scores) if d not in set(rank_llm)
        ]

        # ============================================================
        # hybrid_stack: route by regex cue count
        #   ≤1 regex cue   -> regex_stack (proven on single-cue benches)
        #   ≥2 regex cues  -> llm_planner_stack (handles composition)
        # ============================================================
        regex_cue_count = (
            int(rec_active_re) + int(oe_active_re) + int(neg_active_re) + int(causal_re)
        )
        use_llm = regex_cue_count >= 2
        rank_hybrid = rank_llm if use_llm else rank_regex

        h_ro = hit_rank(rerank_only_rank, gold_set)
        h_re = hit_rank(rank_regex, gold_set)
        h_llm = hit_rank(rank_llm, gold_set)
        h_hyb = hit_rank(rank_hybrid, gold_set)

        results.append(
            {
                "qid": qid,
                "type": ctype,
                "qtext": text,
                "gold": list(gold_set),
                "plan": plan.to_dict(),
                # Regex cue activations
                "regex_rec_active": rec_active_re,
                "regex_neg_active": neg_active_re,
                "regex_oe_active": oe_active_re,
                "regex_causal": causal_re,
                "regex_anchor_did": ar_re["anchor_did"] if ar_re else None,
                # LLM activations
                "llm_rec_active": rec_active_llm,
                "llm_neg_active": neg_active_llm,
                "llm_oe_active": oe_active_llm,
                "llm_causal": causal_llm,
                "llm_abs_active": abs_active_llm,
                "llm_anchor_did": ar_llm["anchor_did"] if ar_llm else None,
                "llm_abs_phrase": plan.absolute_anchor,
                # Hybrid routing
                "regex_cue_count": regex_cue_count,
                "hybrid_used_llm": use_llm,
                "rerank_only": h_ro,
                "regex_stack": h_re,
                "llm_planner_stack": h_llm,
                "hybrid_stack": h_hyb,
                "top5_regex": rank_regex[:5],
                "top5_llm": rank_llm[:5],
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
def write_hybrid_md(report, path):
    """Write the hybrid-planner regression-fix report."""
    benches = report["benches"]
    comp = benches.get("composition")
    variants = ["rerank_only", "regex_stack", "llm_planner_stack", "hybrid_stack"]

    valid = [k for k, v in benches.items() if "error" not in v and v.get("n", 0) > 0]

    lines = []
    lines.append("# T_hybrid_planner — regex-primary / LLM-planner-on-multi-cue\n")
    lines.append(
        "Hybrid routing: count regex cues per query (recency, open_ended, "
        "negation, causal). If ≤1 cue fires, use the proven regex_stack; if "
        "≥2 fire, use the llm_planner_stack. Goal: zero single-cue regression "
        "while preserving the +0.12 composition win.\n"
    )

    # ---- Regression-fix table (LEAD) ----
    lines.append("## Regression-fix table (R@1) — lead\n")
    lines.append(
        "| Benchmark | n | rerank_only | regex_stack | llm_planner | **hybrid** | Δ(hyb − regex) | Δ(hyb − llm) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in benches:
        b = benches[k]
        if "error" in b:
            lines.append(f"| {k} | err | - | - | - | - | - | - |")
            continue
        n = b["overall"]["n"]
        ro = b["overall"]["rerank_only"]["R@1"]
        re_ = b["overall"]["regex_stack"]["R@1"]
        ll = b["overall"]["llm_planner_stack"]["R@1"]
        hy = b["overall"]["hybrid_stack"]["R@1"]
        d_re = hy - re_
        d_ll = hy - ll
        flag_re = ""
        if d_re > 0.005:
            flag_re = " (+)"
        elif d_re < -0.005:
            flag_re = " (REGRESS!)"
        lines.append(
            f"| {k} | {n} | "
            f"{ro:.3f} | {re_:.3f} | {ll:.3f} | "
            f"**{hy:.3f}** | "
            f"**{d_re:+.3f}**{flag_re} | {d_ll:+.3f} |"
        )
    lines.append("")

    # ---- Composition split ----
    if comp and "error" not in comp:
        per_type = comp["per_type"]
        lines.append("## Composition by type (R@1)\n")
        lines.append("| Type | n | regex | llm | **hybrid** | Δ(hyb−regex) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        type_label = {
            "A": "A: rec×abs",
            "B": "B: neg×abs",
            "C": "C: causal×rec",
            "D": "D: causal×abs",
            "E": "E: oe×neg",
            "ALL": "ALL",
        }
        for t in ["A", "B", "C", "D", "E", "ALL"]:
            if t not in per_type:
                continue
            a = per_type[t]
            n = a["n"]
            re_ = a["regex_stack"]["R@1"]
            ll = a["llm_planner_stack"]["R@1"]
            hy = a["hybrid_stack"]["R@1"]
            lines.append(
                f"| {type_label.get(t, t)} | {n} | "
                f"{re_:.3f} | {ll:.3f} | **{hy:.3f}** | {hy - re_:+.3f} |"
            )
        lines.append("")

    # ---- Macro ----
    macro_re = sum(benches[k]["overall"]["regex_stack"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    macro_ll = sum(
        benches[k]["overall"]["llm_planner_stack"]["R@1"] for k in valid
    ) / max(1, len(valid))
    macro_hy = sum(benches[k]["overall"]["hybrid_stack"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    lines.append(f"## Macro R@1 across {len(valid)} benches\n")
    lines.append(f"- regex_stack:        {macro_re:.3f}")
    lines.append(f"- llm_planner_stack:  {macro_ll:.3f}")
    lines.append(f"- **hybrid_stack:     {macro_hy:.3f}**")
    lines.append(f"- Δ(hybrid − regex):  **{macro_hy - macro_re:+.3f}**")
    lines.append(f"- Δ(hybrid − llm):    {macro_hy - macro_ll:+.3f}\n")

    # ---- Routing histogram ----
    lines.append("## Hybrid routing histogram (which queries got the LLM stack?)\n")
    lines.append("| Benchmark | n | LLM-routed | regex-routed |")
    lines.append("|---|---:|---:|---:|")
    for k in benches:
        b = benches[k]
        if "error" in b or not b.get("per_q"):
            continue
        per_q = b["per_q"]
        llm_n = sum(1 for r in per_q if r.get("hybrid_used_llm"))
        re_n = sum(1 for r in per_q if not r.get("hybrid_used_llm"))
        lines.append(f"| {k} | {len(per_q)} | {llm_n} | {re_n} |")
    lines.append("")

    # ---- Verdict ----
    regressions_vs_regex = []
    for k in benches:
        b = benches[k]
        if "error" in b:
            continue
        re_ = b["overall"]["regex_stack"]["R@1"]
        hy = b["overall"]["hybrid_stack"]["R@1"]
        if hy - re_ < -0.005:
            regressions_vs_regex.append((k, re_, hy))
    comp_re = (
        comp["per_type"]["ALL"]["regex_stack"]["R@1"]
        if comp and "error" not in comp
        else None
    )
    comp_hy = (
        comp["per_type"]["ALL"]["hybrid_stack"]["R@1"]
        if comp and "error" not in comp
        else None
    )
    comp_delta = (
        (comp_hy - comp_re) if (comp_re is not None and comp_hy is not None) else None
    )

    lines.append("## Verdict\n")
    if not regressions_vs_regex:
        lines.append(
            "- **No single-cue regressions vs regex_stack.** Hybrid hits or beats regex on every benchmark."
        )
    else:
        lines.append("- **Residual regressions vs regex_stack:**")
        for k, re_, hy in regressions_vs_regex:
            lines.append(f"  - {k}: {re_:.3f} -> {hy:.3f} ({hy - re_:+.3f})")
    if comp_delta is not None:
        kept = (
            "PRESERVED"
            if comp_delta >= 0.10
            else ("PARTIAL" if comp_delta >= 0.04 else "LOST")
        )
        lines.append(
            f"- **Composition Δ vs regex: {comp_delta:+.3f}** ({kept} the +0.12 win)"
        )
    lines.append(f"- **Macro Δ vs regex: {macro_hy - macro_re:+.3f}**\n")

    # ---- Recommendation ----
    lines.append("## Recommendation\n")
    if not regressions_vs_regex and comp_delta is not None and comp_delta >= 0.08:
        lines.append(
            f"**SHIP hybrid_stack.** Zero single-cue regressions; composition Δ {comp_delta:+.3f}; "
            f"macro Δ {macro_hy - macro_re:+.3f}. Best of both worlds.\n"
        )
    elif comp_delta is not None and comp_delta >= 0.04 and macro_hy - macro_re >= -0.01:
        lines.append(
            f"**SHIP hybrid with caveat.** Composition Δ {comp_delta:+.3f}; "
            f"macro Δ {macro_hy - macro_re:+.3f}. "
            "Residual regression(s) are minor; consider patching the failing axis.\n"
        )
    else:
        cd = comp_delta if comp_delta is not None else 0.0
        lines.append(
            f"**STAY WITH regex_stack.** Hybrid composition Δ {cd:+.3f}; "
            f"macro Δ {macro_hy - macro_re:+.3f}. Lift insufficient.\n"
        )
    return "\n".join(lines)


def write_md(report, path):
    benches = report["benches"]
    comp = benches.get("composition")
    planner_stats = report["planner_stats"]
    variants = ["rerank_only", "regex_stack", "llm_planner_stack"]

    # Overall benchmark table
    valid = [k for k, v in benches.items() if "error" not in v and v.get("n", 0) > 0]
    macro = {
        v: sum(benches[k]["overall"][v]["R@1"] for k in valid) / max(1, len(valid))
        for v in variants
    }

    lines = []
    lines.append("# T_llm_planner — LLM-based structured query planner\n")
    lines.append(
        "Replace the regex parsers (recency / negation / open_ended / causal) with a single gpt-5-mini call returning a structured QueryPlan. Composition is multiplicative (S2 from prior eval). New: `absolute_anchor` adds a hard interval filter on the doc set.\n"
    )

    # ============================================================
    # Lead with composition R@1 deltas
    # ============================================================
    lines.append("## Composition R@1 — lead\n")
    if comp and "error" not in comp:
        per_type = comp["per_type"]
        lines.append(
            "| Type | n | rerank_only | regex_stack | **llm_planner_stack** | Δ(llm − regex) |"
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
            ll = a["llm_planner_stack"]["R@1"]
            d = ll - re_
            lines.append(
                f"| {type_label.get(t, t)} | {n} | "
                f"{ro:.3f} ({a['rerank_only']['r1_count']}/{n}) | "
                f"{re_:.3f} ({a['regex_stack']['r1_count']}/{n}) | "
                f"**{ll:.3f}** ({a['llm_planner_stack']['r1_count']}/{n}) | "
                f"**{d:+.3f}** |"
            )
        lines.append("")
    else:
        lines.append("*composition bench skipped or errored.*\n")

    # ============================================================
    # Single-cue regression check
    # ============================================================
    lines.append("## Single-cue regression check (R@1)\n")
    lines.append(
        "| Benchmark | n | rerank_only | regex_stack | llm_planner_stack | Δ(llm − regex) |"
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
        ll = b["overall"]["llm_planner_stack"]["R@1"]
        d = ll - re_
        flag = ""
        if d > 0.005:
            flag = " (llm better)"
        elif d < -0.005:
            flag = " (regress!)"
        lines.append(
            f"| {k} | {n} | "
            f"{ro:.3f} ({b['overall']['rerank_only']['r1_count']}/{n}) | "
            f"{re_:.3f} ({b['overall']['regex_stack']['r1_count']}/{n}) | "
            f"{ll:.3f} ({b['overall']['llm_planner_stack']['r1_count']}/{n}) | "
            f"**{d:+.3f}**{flag} |"
        )
    lines.append("")
    lines.append(f"### Macro-average R@1 across {len(valid)} benches\n")
    lines.append(f"- rerank_only: {macro['rerank_only']:.3f}")
    lines.append(f"- regex_stack: {macro['regex_stack']:.3f}")
    lines.append(f"- **llm_planner_stack**: **{macro['llm_planner_stack']:.3f}**")
    lines.append(
        f"- Δ(llm − regex): **{macro['llm_planner_stack'] - macro['regex_stack']:+.3f}**\n"
    )

    # ============================================================
    # Per-composition-type fixes
    # ============================================================
    if comp and "error" not in comp:
        lines.append("## Per-composition-type fix audit\n")
        per_q = comp["per_q"]
        type_label = {
            "A": "recency × absolute",
            "B": "negation × absolute",
            "C": "causal × recency",
            "D": "causal × absolute",
            "E": "open_ended × negation",
        }
        type_fixes = {}
        type_breaks = {}
        type_total = {}
        for r in per_q:
            t = r["type"]
            type_total[t] = type_total.get(t, 0) + 1
            re_hit = r["regex_stack"] == 1
            ll_hit = r["llm_planner_stack"] == 1
            if not re_hit and ll_hit:
                type_fixes[t] = type_fixes.get(t, 0) + 1
            if re_hit and not ll_hit:
                type_breaks[t] = type_breaks.get(t, 0) + 1
        lines.append("| Type | n | regex misses | llm fixes | llm breaks | Net |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for t in ["A", "B", "C", "D", "E"]:
            n_total = type_total.get(t, 0)
            misses = sum(1 for r in per_q if r["type"] == t and r["regex_stack"] != 1)
            fixes = type_fixes.get(t, 0)
            breaks = type_breaks.get(t, 0)
            net = fixes - breaks
            lines.append(
                f"| {t} ({type_label.get(t, '')}) | {n_total} | {misses} | {fixes} | {breaks} | **{net:+d}** |"
            )
        lines.append("")

    # ============================================================
    # Cost
    # ============================================================
    lines.append("## Cost\n")
    lines.append(f"- Planner model: `{planner_stats['model']}`")
    lines.append(f"- Total queries seen: {planner_stats['total_queries']}")
    lines.append(f"- Live LLM calls: {planner_stats['calls']}")
    lines.append(f"- Cache hits: {planner_stats['cache_hits']}")
    lines.append(f"- Cache hit rate: **{planner_stats['cache_hit_rate'] * 100:.1f}%**")
    lines.append(f"- Parse failures: {planner_stats['parse_failures']}")
    lines.append(
        "- LLM calls per UNIQUE query: 1 (single planner call); cache eliminates rerun cost.\n"
    )

    # ============================================================
    # Failure analysis
    # ============================================================
    lines.append(
        "## Failure analysis (composition queries the LLM stack still misses)\n"
    )
    if comp and "error" not in comp:
        per_q = comp["per_q"]
        for r in per_q:
            if r["llm_planner_stack"] == 1:
                continue
            rank = r["llm_planner_stack"]
            t = r["type"]
            lines.append(f"### {r['qid']} — Type {t}, llm rank={rank}")
            lines.append(f"- Query: `{r['qtext']}`")
            lines.append(f"- Gold: `{r['gold']}`")
            lines.append(f"- Plan: `{json.dumps(r['plan'])}`")
            lines.append(
                f"- LLM cues: rec={r['llm_rec_active']} neg={r['llm_neg_active']} "
                f"oe={r['llm_oe_active']} causal={r['llm_causal']} "
                f"abs={r['llm_abs_active']}"
            )
            if r["llm_anchor_did"]:
                lines.append(f"- LLM resolved causal anchor: `{r['llm_anchor_did']}`")
            lines.append(f"- llm top5: `{r['top5_llm']}`")
            lines.append(
                f"- regex top5 (regex_rank={r['regex_stack']}): `{r['top5_regex']}`"
            )
            lines.append("")

    # ============================================================
    # Recommendation
    # ============================================================
    lines.append("## Recommendation\n")
    if comp and "error" not in comp:
        comp_re = comp["per_type"]["ALL"]["regex_stack"]["R@1"]
        comp_ll = comp["per_type"]["ALL"]["llm_planner_stack"]["R@1"]
        comp_delta = comp_ll - comp_re
        macro_delta = macro["llm_planner_stack"] - macro["regex_stack"]
        if comp_delta >= 0.08 and macro_delta >= -0.01:
            lines.append(
                f"**SHIP the LLM planner.** Composition R@1: {comp_re:.3f} -> {comp_ll:.3f} "
                f"({comp_delta:+.3f}); single-cue macro Δ {macro_delta:+.3f}. "
                "Replaces three regex parsers with one structured call; cache amortizes cost.\n"
            )
        elif comp_delta >= 0.04 and macro_delta >= -0.02:
            lines.append(
                f"**SHIP WITH CAVEAT.** Composition lift {comp_delta:+.3f}; single-cue Δ {macro_delta:+.3f}. "
                "Real composition gain but small single-cue regression — patch the failing axis or "
                "tune the multiplier weights before flipping the production switch.\n"
            )
        else:
            lines.append(
                f"**STAY WITH PATCHED REGEX.** Composition Δ {comp_delta:+.3f}; single-cue Δ {macro_delta:+.3f}. "
                "Lift is too small to justify the extra LLM call; better ROI is patching "
                "parse_negation_query (Type B) and the open_ended/causal gate conflict (Type D).\n"
            )
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

    # Composition + 11 single-cue benchmarks
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

    variants = ["rerank_only", "regex_stack", "llm_planner_stack", "hybrid_stack"]
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
                    f"  {v:20s} R@1={d['R@1']:.3f} ({d['r1_count']}/{out['benches'][nm]['n']})",
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
    json_path = out_dir / "T_llm_planner.json"
    json_safe = {"benches": {}, "planner_stats": out["planner_stats"]}
    for k, v in out["benches"].items():
        if "error" in v:
            json_safe["benches"][k] = v
            continue
        json_safe["benches"][k] = v
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(f"Wrote {json_path}", flush=True)

    md_path = out_dir / "T_llm_planner.md"
    md_path.write_text(write_md(out, md_path))
    print(f"Wrote {md_path}", flush=True)

    # Hybrid (regex-primary, llm-on-multi-cue) report
    hybrid_md_path = out_dir / "T_hybrid_planner.md"
    hybrid_md_path.write_text(write_hybrid_md(out, hybrid_md_path))
    print(f"Wrote {hybrid_md_path}", flush=True)

    json_path_h = out_dir / "T_hybrid_planner.json"
    with open(json_path_h, "w") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(f"Wrote {json_path_h}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
