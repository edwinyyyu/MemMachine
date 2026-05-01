"""Composition eval: tests how the cue-gated modules combine on
multi-cue queries.

Strategies:
  S0  rerank_only baseline
  S1  current_sequential — sequential overrides as in current production:
      base = fuse_T_R + recency_additive (recency cue on)
      then if causal cue: causal_signed mask
      then if negation cue: replace base with negation_signed (positive +
        excluded penalty)
  S2  multiplicative — final = base * recency_mult * (1-excl_cont) * dir_match
      (each module's contribution is a score multiplier; defaults to 1)
  S3  additive — final = base + α_rec*recency_add - λ_excl*excl_cont
        - λ_dir*wrong_direction
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

HALF_LIFE_DAYS = 21.0
CV_REF = 0.20
W_T_FUSE_TR = 0.4
ADDITIVE_ALPHA = 0.5

LAM_NEG_SIGNED = 1.0


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


def normalize_dict(d: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a score dict to [0, 1]."""
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in d.items()}


# -----------------------------------------------------------------------------
# Direction match scoring
# -----------------------------------------------------------------------------
def direction_match_scores(
    doc_ref_us: dict[str, int],
    anchor_us: int,
    direction: str,
    anchor_did: str,
) -> dict[str, float]:
    """Return 1.0 if doc on the right side of anchor, 0.0 otherwise (excluding anchor itself)."""
    out = {}
    for did, dus in doc_ref_us.items():
        if did == anchor_did:
            out[did] = 0.0
        elif direction_match(dus, anchor_us, direction):
            out[did] = 1.0
        else:
            out[did] = 0.0
    return out


# -----------------------------------------------------------------------------
# Bench
# -----------------------------------------------------------------------------
async def run_bench(name, docs_path, queries_path, gold_path, cache_label, reranker):
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

    # Negation parses (positive query + excluded phrase) for each query.
    parsed_meta = {}
    pos_items = []
    excl_items = []
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
        pos_items.append((f"{qid}__pos", pos_q, ref))
        if cue and excl_q:
            excl_items.append((f"{qid}__excl", excl_q, ref))

    pos_ext = await run_v2_extract(pos_items, f"{name}-pos", f"{cache_label}-pos")
    excl_ext_raw = (
        await run_v2_extract(excl_items, f"{name}-excl", f"{cache_label}-excl")
        if excl_items
        else {}
    )

    # Doc memory + lattice (T_lblend)
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

    lat_db = ROOT / "cache" / "composition" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)

    # Doc intervals flat for negation excluded-containment
    doc_ivs_flat = {}
    for did, tes in doc_ext.items():
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        doc_ivs_flat[did] = ivs
    for d in docs:
        doc_ivs_flat.setdefault(d["doc_id"], [])

    # Recency anchor bundles
    doc_bundles_for_rec: dict[str, list[dict]] = {}
    for did, mem in doc_mem.items():
        ivs = mem.get("intervals") or []
        doc_bundles_for_rec[did] = [{"intervals": ivs}] if ivs else []

    # Embeddings
    print("  embedding...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    pos_q_texts = [parsed_meta[q["query_id"]][1] for q in queries]
    pos_q_embs_arr = await embed_all(pos_q_texts)
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

    # T_lblend per query (POSITIVE query — needed for negation)
    per_q_l_pos = {
        qid: lattice_retrieve_multi(lat, pos_ext.get(f"{qid}__pos", []), down_levels=1)[
            0
        ]
        if pos_ext.get(f"{qid}__pos")
        else {}
        for qid in qids
    }
    pos_mem = build_memory({qid: pos_ext.get(f"{qid}__pos", []) for qid in qids})
    per_q_t_pos = {
        qid: make_t_scores(
            pos_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l_pos.get(qid, {}),
        )
        for qid in qids
    }
    for qid in qids:
        for d in docs:
            per_q_t_pos[qid].setdefault(d["doc_id"], 0.0)

    # T_v5 per query (open_ended router)
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

    # Rerank: union of top-50 sem, top-50 T_lblend, top-50 T_v5
    print("  reranking...", flush=True)
    per_q_r_full: dict[str, dict[str, float]] = {}
    per_q_r_partial: dict[str, dict[str, float]] = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        v5_top = topk_from_scores(per_q_tv5[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top + v5_top))[: int(RERANK_TOP_K * 1.5)]
        rs = await rerank_topk(reranker, q_text[qid], union, doc_text, len(union))
        per_q_r_partial[qid] = rs
        per_q_r_full[qid] = normalize_rerank_full(rs, [d["doc_id"] for d in docs], 0.0)

    # ----- Causal anchor resolution (precompute, batch embed phrases) ----
    causal_info: dict[str, dict] = {}
    anchor_phrases: list[str] = []
    for q in queries:
        info = detect_causal(q["text"])
        if info is None:
            continue
        cue, phrase = info
        causal_info[q["query_id"]] = {
            "cue": cue,
            "phrase": phrase,
            "direction": cue_direction(cue),
        }
        anchor_phrases.append(phrase)

    unique_phrases = list(dict.fromkeys(anchor_phrases))
    if unique_phrases:
        ph_embs = await embed_all(unique_phrases)
        phrase_emb = {p: ph_embs[i] for i, p in enumerate(unique_phrases)}
    else:
        phrase_emb = {}

    anchor_resolution: dict[str, dict] = {}
    for qid, info in causal_info.items():
        emb = phrase_emb.get(info["phrase"])
        if emb is None:
            continue
        res = resolve_anchor(info["phrase"], emb, doc_embs)
        if res is None:
            continue
        did, sim = res
        anchor_resolution[qid] = {
            "anchor_did": did,
            "anchor_us": doc_ref_us[did],
            "direction": info["direction"],
            "sim": sim,
            "phrase": info["phrase"],
        }

    lam_rec = lambda_for_half_life(HALF_LIFE_DAYS)

    # =================================================================
    # Per-query evaluation: 4 strategies
    # =================================================================
    results = []
    fail_examples = []

    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        text = q["text"]
        ctype = q_type.get(qid, "?")

        # Cue detection
        rec_active = has_recency_cue(text)
        neg_active, pos_q, excl_q = parsed_meta[qid]
        oe_active = has_open_ended_cue(text)
        causal = qid in anchor_resolution
        ar = anchor_resolution.get(qid)

        # Recency scores
        rec_scores = recency_scores_for_docs(
            doc_bundles_for_rec,
            doc_ref_us,
            q_ref_us[qid],
            lam_rec,
        )

        # T_router: T_v5 if open_ended else T_lblend (full query)
        t_router = per_q_tv5[qid] if oe_active else per_q_t[qid]
        t_router_pos = per_q_tv5[qid] if oe_active else per_q_t_pos[qid]

        r_full = per_q_r_full[qid]
        rerank_partial = per_q_r_partial[qid]
        s_scores = per_q_s[qid]
        s_pos_scores = per_q_s_pos[qid]

        # ---- Excluded containment for negation ---------------------------
        excl_te = excl_ext_raw.get(f"{qid}__excl", []) if neg_active else []
        excl_ivs = []
        for te in excl_te:
            excl_ivs.extend(flatten_intervals(te))
        excl_cont = {
            did: excluded_containment(doc_ivs_flat.get(did, []), excl_ivs)
            for did in doc_ref_us
        }

        # ---- Direction-match (causal) ------------------------------------
        if causal and ar:
            dir_match_full = direction_match_scores(
                doc_ref_us,
                ar["anchor_us"],
                ar["direction"],
                ar["anchor_did"],
            )
        else:
            dir_match_full = dict.fromkeys(doc_ref_us, 1.0)

        # =============================================================
        # S0: rerank_only
        # =============================================================
        rerank_only_rank = merge_with_tail(
            [
                d
                for d, _ in sorted(
                    rerank_partial.items(), key=lambda x: x[1], reverse=True
                )
            ],
            s_scores,
        )

        # =============================================================
        # S1: CURRENT SEQUENTIAL
        # base = fuse_T_R + recency_additive(rec)
        # if causal: causal_signed mask
        # if negation: REPLACE with positive_composite + signed mask
        # =============================================================
        # Step 1: base
        fused_TR = fuse_T_R_blend_scores(t_router, r_full, w_T=W_T_FUSE_TR)
        s1 = additive_with_recency(
            fused_TR, rec_scores, cue=rec_active, alpha=ADDITIVE_ALPHA
        )
        # Step 2: causal mask (signed)
        if causal and ar:
            s1 = causal_signed_scores(
                s1,
                doc_ref_us,
                ar["anchor_us"],
                ar["direction"],
                ar["anchor_did"],
                lam=CAUSAL_SIGNED_LAMBDA,
            )
        # Step 3: negation override
        if neg_active and excl_ivs:
            # Replicate negation_eval's positive_composite construction
            positive_composite = {
                did: 0.7 * s_pos_scores.get(did, 0.0)
                + 0.3 * per_q_t_pos[qid].get(did, 0.0)
                for did in doc_ref_us
            }
            s1 = apply_signed(positive_composite, excl_cont, lam=LAM_NEG_SIGNED)

        rank_s1 = rank_from_scores(s1)
        rank_s1 = rank_s1 + [
            d for d in rank_from_scores(s_scores) if d not in set(rank_s1)
        ]

        # =============================================================
        # S2: MULTIPLICATIVE
        # base = fused_TR (NOT additive recency — recency comes in as
        #   a multiplier so we don't double-count)
        # final = base * recency_mult * (1 - excl_cont) * dir_match
        # where:
        #   recency_mult = 1 + α * rec_score  (no cue → α=0)
        #   dir_match: 1 if right side or no causal, 0.1 if wrong side
        # =============================================================
        REC_MULT_ALPHA = 1.0  # how strongly recency lifts
        DIR_WRONG_FACTOR = 0.1  # multiplier for wrong-direction docs (vs 1.0)

        # Use the positive-query base when negation, full-query base otherwise.
        if neg_active and excl_ivs:
            # Positive-only base — semantic + T_lblend(positive)
            base = {
                did: 0.7 * s_pos_scores.get(did, 0.0)
                + 0.3 * per_q_t_pos[qid].get(did, 0.0)
                for did in doc_ref_us
            }
            base = normalize_dict(base)
        else:
            base = normalize_dict(fused_TR)

        s2 = {}
        for did in doc_ref_us:
            b = base.get(did, 0.0)
            r_mult = (
                1.0 + REC_MULT_ALPHA * rec_scores.get(did, 0.0) if rec_active else 1.0
            )
            e_factor = (
                1.0 - excl_cont.get(did, 0.0) if (neg_active and excl_ivs) else 1.0
            )
            d_factor = 1.0
            if causal and ar:
                d_factor = (
                    1.0 if dir_match_full.get(did, 1.0) >= 0.5 else DIR_WRONG_FACTOR
                )
                if did == ar["anchor_did"]:
                    d_factor = 0.0  # suppress anchor itself
            s2[did] = b * r_mult * e_factor * d_factor
        rank_s2 = rank_from_scores(s2)
        rank_s2 = rank_s2 + [
            d for d in rank_from_scores(s_scores) if d not in set(rank_s2)
        ]

        # =============================================================
        # S3: ADDITIVE
        # final = base + α_rec*rec - λ_excl*excl - λ_dir*wrong_dir
        # where wrong_dir = 1 - dir_match (1.0 means wrong side)
        # =============================================================
        ALPHA_REC = 0.5
        LAM_EXCL = 1.0
        LAM_DIR = 0.5

        if neg_active and excl_ivs:
            base3 = {
                did: 0.7 * s_pos_scores.get(did, 0.0)
                + 0.3 * per_q_t_pos[qid].get(did, 0.0)
                for did in doc_ref_us
            }
        else:
            base3 = dict(fused_TR)
            for d in doc_ref_us:
                base3.setdefault(d, 0.0)

        s3 = {}
        for did in doc_ref_us:
            b = base3.get(did, 0.0)
            v = b
            if rec_active:
                v += ALPHA_REC * rec_scores.get(did, 0.0)
            if neg_active and excl_ivs:
                v -= LAM_EXCL * excl_cont.get(did, 0.0)
            if causal and ar:
                if did == ar["anchor_did"]:
                    v -= LAM_DIR  # suppress anchor
                else:
                    wrong = 1.0 - dir_match_full.get(did, 1.0)
                    v -= LAM_DIR * wrong
            s3[did] = v
        rank_s3 = rank_from_scores(s3)
        rank_s3 = rank_s3 + [
            d for d in rank_from_scores(s_scores) if d not in set(rank_s3)
        ]

        h0 = hit_rank(rerank_only_rank, gold_set)
        h1 = hit_rank(rank_s1, gold_set)
        h2 = hit_rank(rank_s2, gold_set)
        h3 = hit_rank(rank_s3, gold_set)

        results.append(
            {
                "qid": qid,
                "type": ctype,
                "qtext": text,
                "gold": list(gold_set),
                "rec_active": rec_active,
                "neg_active": neg_active,
                "oe_active": oe_active,
                "causal": causal,
                "anchor_did": ar["anchor_did"] if ar else None,
                "anchor_dir": ar["direction"] if ar else None,
                "rerank_only": h0,
                "S1_sequential": h1,
                "S2_multiplicative": h2,
                "S3_additive": h3,
                "top5_S1": rank_s1[:5],
                "top5_S2": rank_s2[:5],
                "top5_S3": rank_s3[:5],
            }
        )

        # Capture failures for the report
        if h1 != 1 and len(fail_examples) < 5:
            fail_examples.append(
                {
                    "qid": qid,
                    "type": ctype,
                    "query": text,
                    "gold": list(gold_set)[0],
                    "S1_top5": rank_s1[:5],
                    "S1_rank": h1,
                    "S2_top5": rank_s2[:5],
                    "S2_rank": h2,
                    "S3_top5": rank_s3[:5],
                    "S3_rank": h3,
                    "rec_active": rec_active,
                    "neg_active": neg_active,
                    "oe_active": oe_active,
                    "causal": causal,
                }
            )

    return results, fail_examples


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def aggregate_per_type(results: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    out = {}
    variants = ["rerank_only", "S1_sequential", "S2_multiplicative", "S3_additive"]
    for typ, rs in by_type.items():
        n = len(rs)
        out[typ] = {"n": n}
        for v in variants:
            ranks = [r[v] for r in rs]
            r1 = sum(1 for x in ranks if x is not None and x <= 1)
            r5 = sum(1 for x in ranks if x is not None and x <= 5)
            mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
            out[typ][v] = {
                "R@1": r1 / n if n else 0.0,
                "R@5": r5 / n if n else 0.0,
                "MRR": mrr,
                "r1_count": r1,
                "r5_count": r5,
            }
    # Overall
    n_all = len(results)
    out["ALL"] = {"n": n_all}
    for v in variants:
        ranks = [r[v] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr = sum(1.0 / x for x in ranks if x is not None) / n_all if n_all else 0.0
        out["ALL"][v] = {
            "R@1": r1 / n_all if n_all else 0.0,
            "R@5": r5 / n_all if n_all else 0.0,
            "MRR": mrr,
            "r1_count": r1,
            "r5_count": r5,
        }
    return out


def write_md(report: dict, path: Path):
    agg = report["agg"]
    fails = report["fails"]
    results = report["results"]

    types = ["A", "B", "C", "D", "E", "ALL"]
    type_label = {
        "A": "A: recency × absolute",
        "B": "B: negation × absolute",
        "C": "C: causal × recency",
        "D": "D: causal × absolute",
        "E": "E: open_ended × negation",
        "ALL": "ALL",
    }

    lines = []
    lines.append("# T_composition — multi-cue composition test\n")
    lines.append(
        "Tests how the cue-gated modules (recency, open_ended, negation, causal) compose when more than one fires per query.\n"
    )

    lines.append("## R@1 by composition type, per strategy\n")
    lines.append(
        "| Type | n | rerank_only | S1 sequential | S2 multiplicative | S3 additive |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for t in types:
        a = agg.get(t)
        if a is None:
            continue
        n = a["n"]
        ro = a["rerank_only"]["R@1"]
        s1 = a["S1_sequential"]["R@1"]
        s2 = a["S2_multiplicative"]["R@1"]
        s3 = a["S3_additive"]["R@1"]
        lines.append(
            f"| {type_label.get(t, t)} | {n} | "
            f"{ro:.3f} ({a['rerank_only']['r1_count']}/{n}) | "
            f"{s1:.3f} ({a['S1_sequential']['r1_count']}/{n}) | "
            f"{s2:.3f} ({a['S2_multiplicative']['r1_count']}/{n}) | "
            f"{s3:.3f} ({a['S3_additive']['r1_count']}/{n}) |"
        )
    lines.append("")

    lines.append("## R@5 by composition type, per strategy\n")
    lines.append(
        "| Type | n | rerank_only | S1 sequential | S2 multiplicative | S3 additive |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for t in types:
        a = agg.get(t)
        if a is None:
            continue
        n = a["n"]
        lines.append(
            f"| {type_label.get(t, t)} | {n} | "
            f"{a['rerank_only']['R@5']:.3f} | {a['S1_sequential']['R@5']:.3f} | "
            f"{a['S2_multiplicative']['R@5']:.3f} | {a['S3_additive']['R@5']:.3f} |"
        )
    lines.append("")

    # Where does sequential fail?
    lines.append("## Where sequential (S1) fails\n")
    s1_fail_by_type: dict[str, int] = {}
    for r in results:
        if r["S1_sequential"] != 1:
            s1_fail_by_type[r["type"]] = s1_fail_by_type.get(r["type"], 0) + 1
    if s1_fail_by_type:
        for t in sorted(s1_fail_by_type):
            n = agg[t]["n"]
            lines.append(
                f"- **Type {t}** ({type_label.get(t, t)}): {s1_fail_by_type[t]}/{n} S1 R@1 misses"
            )
    else:
        lines.append("- S1 hit R@1 on every composition query.")
    lines.append("")

    lines.append("## Does multiplicative (S2) fix it?\n")
    s2_helps = 0
    s2_hurts = 0
    for r in results:
        if r["S1_sequential"] != 1 and r["S2_multiplicative"] == 1:
            s2_helps += 1
        if r["S1_sequential"] == 1 and r["S2_multiplicative"] != 1:
            s2_hurts += 1
    lines.append(f"- S2 fixes **{s2_helps}** queries that S1 missed at R@1")
    lines.append(f"- S2 breaks **{s2_hurts}** queries that S1 had at R@1")
    lines.append("")

    s3_helps = 0
    s3_hurts = 0
    for r in results:
        if r["S1_sequential"] != 1 and r["S3_additive"] == 1:
            s3_helps += 1
        if r["S1_sequential"] == 1 and r["S3_additive"] != 1:
            s3_hurts += 1
    lines.append(f"- S3 (additive) fixes **{s3_helps}**, breaks **{s3_hurts}** vs S1\n")

    # Recommendation
    s1_r1 = agg["ALL"]["S1_sequential"]["R@1"]
    s2_r1 = agg["ALL"]["S2_multiplicative"]["R@1"]
    s3_r1 = agg["ALL"]["S3_additive"]["R@1"]
    best = max([("S1", s1_r1), ("S2", s2_r1), ("S3", s3_r1)], key=lambda x: x[1])
    lines.append("## Recommended composition logic\n")
    lines.append(f"- Overall ALL R@1: S1={s1_r1:.3f}, S2={s2_r1:.3f}, S3={s3_r1:.3f}")
    lines.append(f"- Winner: **{best[0]}** ({best[1]:.3f})")
    lines.append("")
    if best[0] == "S1":
        lines.append(
            "Sequential composition wins — current production behavior is robust on these composition cases.\n"
        )
    elif best[0] == "S2":
        lines.append(
            "Multiplicative composition wins — recommend transitioning each module to expose a [0,1] multiplier so they compose by product. Concretely: recency_mult=1+α*rec_score, negation_mult=(1-excl_cont), causal_mult=1.0 right side / 0.1 wrong side, anchor=0. Negation handles base substitution (positive_composite) before the multipliers fire.\n"
        )
    else:
        lines.append(
            "Additive composition wins — recommend exposing each module's contribution as a signed delta (recency: +α*rec; negation: -λ*excl; causal: -λ*wrong_side). Composition becomes simple summation; ties survive monotonically.\n"
        )

    lines.append("## Failure cases\n")
    for ex in fails[:5]:
        lines.append(f"### {ex['qid']} — Type {ex['type']}")
        lines.append(f"- Query: `{ex['query']}`")
        lines.append(f"- Gold: `{ex['gold']}`")
        lines.append(
            f"- Cues: rec={ex['rec_active']} neg={ex['neg_active']} oe={ex['oe_active']} causal={ex['causal']}"
        )
        lines.append(f"- S1 rank={ex['S1_rank']}, top5: {ex['S1_top5']}")
        lines.append(f"- S2 rank={ex['S2_rank']}, top5: {ex['S2_top5']}")
        lines.append(f"- S3 rank={ex['S3_rank']}, top5: {ex['S3_top5']}")
        lines.append("")

    path.write_text("\n".join(lines))


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

    name = "composition"
    results, fails = await run_bench(
        name,
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
        "edge-composition",
        reranker,
    )
    agg = aggregate_per_type(results)

    print("\n" + "=" * 60)
    print("Aggregate R@1 by type:")
    print("=" * 60)
    for t in ["A", "B", "C", "D", "E", "ALL"]:
        if t not in agg:
            continue
        a = agg[t]
        n = a["n"]
        print(
            f"  Type {t} (n={n}): "
            f"rerank_only={a['rerank_only']['R@1']:.3f}  "
            f"S1={a['S1_sequential']['R@1']:.3f}  "
            f"S2={a['S2_multiplicative']['R@1']:.3f}  "
            f"S3={a['S3_additive']['R@1']:.3f}",
            flush=True,
        )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "T_composition.json"
    with open(json_path, "w") as f:
        json.dump(
            {"agg": agg, "results": results, "fails": fails}, f, indent=2, default=str
        )
    print(f"\nWrote {json_path}", flush=True)

    md_path = out_dir / "T_composition.md"
    write_md({"agg": agg, "results": results, "fails": fails}, md_path)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
