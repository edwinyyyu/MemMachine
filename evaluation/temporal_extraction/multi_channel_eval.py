"""Multi-channel temporal scoring eval.

Tests a multi-channel scoring system combining T_lblend + recency + R via
score_blend (rag_fusion.score_blend with CV gate), gated by simple regex
switches that determine which channels enter the blend per-query.

Five variants:
  1. rerank_only — baseline (cross-encoder over union(top-50 sem, top-50 T_lblend))
  2. fuse_T_lblend_R — current shipping (w_T=0.4 score_blend over {T_lblend, R})
  3. fuse_T_R + recency_additive (α=0.5 add when cue detected) — Agent I's recipe
  4. multi_channel_switches — switches gate which channels enter score_blend
  5. multi_channel_no_switches — same as 4 but ALL channels always active
     (CV gate must do the suppression alone)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from schema import to_us

HALF_LIFE_DAYS = 21.0  # default; matches Agent I shipping
CV_REF = 0.20
W_T = 0.30
W_REC = 0.30
W_R = 0.40

# For variant 2 (current shipping): w_T = 0.4 over {T, R} (the already-tuned
# default in fuse_at_w in force_pick_optimizers_eval.py).
W_T_FUSE_TR = 0.4
W_R_FUSE_TR = 0.6

# For variant 3 (recency additive on top of fuse_T_R): α=0.5
ADDITIVE_ALPHA = 0.5


# -----------------------------------------------------------------------------
# Switch regexes
# -----------------------------------------------------------------------------
# Temporal-anchor switch: years 19xx/20xx, quarters Q1-Q4, month names,
# season words, "early/mid/late" + decade/century, ISO-style date strings.
_MONTHS = (
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)
_SEASONS = r"spring|summer|autumn|fall|winter"
_TEMPORAL_ANCHOR_RE = re.compile(
    r"\b("
    r"(?:19|20)\d{2}"  # year 19xx/20xx
    r"|q[1-4]\b"  # Q1-Q4
    r"|(?:" + _MONTHS + r")\b"  # month names
    r"|(?:" + _SEASONS + r")\b"  # seasons
    r"|(?:early|mid|late)\s+(?:19|20)?\d{2}s?"  # "early 1990s", "late 80s"
    r"|(?:early|mid|late)\s+(?:" + _MONTHS + r"|" + _SEASONS + r")"
    r"|\d{4}-\d{2}-\d{2}"  # ISO date
    r"|\d{1,2}/\d{1,2}/\d{2,4}"  # slash dates
    r"|\d{1,2}\s+(?:" + _MONTHS + r")"  # "5 March"
    r"|(?:" + _MONTHS + r")\s+\d{1,2}(?:st|nd|rd|th)?"  # "March 5th"
    r")\b",
    re.IGNORECASE,
)


def has_temporal_anchor(query_text: str) -> bool:
    """Return True if the query contains an explicit temporal anchor.

    Anchors covered: year, quarter, month, season, era ("early/mid/late
    1990s"), ISO/slash date, "5 March"/"March 5".
    """
    if not query_text:
        return False
    return _TEMPORAL_ANCHOR_RE.search(query_text) is not None


# -----------------------------------------------------------------------------
# Ranking helpers
# -----------------------------------------------------------------------------
def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


def rank_from_scores(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def normalize_rerank_full(
    rerank_partial: dict[str, float], all_doc_ids, tail_score: float = 0.0
) -> dict[str, float]:
    """Min-max normalize rerank scores to [0,1]; docs outside the rerank
    candidate set get `tail_score` (default 0) so they cannot win in
    score_blend after CV scaling."""
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


def fuse_T_R_blend(t_scores, r_scores, w_T=W_T_FUSE_TR):
    """Current shipping: score_blend over {T, R} with CV gate."""
    fused = score_blend(
        {"T": t_scores, "R": r_scores},
        {"T": w_T, "R": 1.0 - w_T},
        top_k_per=40,
        dispersion_cv_ref=CV_REF,
    )
    return [d for d, _ in fused]


def additive_with_recency(
    base_scores: dict[str, float],
    rec_scores: dict[str, float],
    cue: bool,
    alpha: float = ADDITIVE_ALPHA,
):
    """Agent I's recipe: simple α=0.5 add of recency to base when cue detected."""
    if not cue:
        return dict(base_scores)
    docs = set(base_scores) | set(rec_scores)
    out = {}
    for d in docs:
        out[d] = (1 - alpha) * base_scores.get(d, 0.0) + alpha * rec_scores.get(d, 0.0)
    return out


def multi_channel_blend(
    t_scores,
    rec_scores,
    r_scores,
    switches: dict[str, bool],
    weights: dict[str, float] | None = None,
    dispersion_cv_ref: float = CV_REF,
):
    """Build score_blend over channels gated by switches.

    Switches map: {"T_active": bool, "Recency_active": bool}.
    R is always active. weights = full {"T":, "recency":, "R":} dict;
    inactive channels are dropped.
    """
    if weights is None:
        weights = {"T": W_T, "recency": W_REC, "R": W_R}

    channels: dict[str, dict[str, float]] = {}
    if switches.get("T_active", True):
        channels["T"] = t_scores
    if switches.get("Recency_active", True):
        channels["recency"] = rec_scores
    channels["R"] = r_scores

    use_w = {k: weights[k] for k in channels}
    fused = score_blend(
        channels, use_w, top_k_per=40, dispersion_cv_ref=dispersion_cv_ref
    )
    return [d for d, _ in fused]


# -----------------------------------------------------------------------------
# Bench loop
# -----------------------------------------------------------------------------
async def run_bench(name, docs_path, queries_path, gold_path, cache_label, reranker):
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
    q_ref_us = {q["query_id"]: to_us(parse_iso(q["ref_time"])) for q in queries}

    doc_mem = build_memory(doc_ext)
    q_mem = build_memory(q_ext)
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

    # Lattice for T_lblend
    lat_db = ROOT / "cache" / "multi_channel" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)
    qids = [q["query_id"] for q in queries]
    per_q_l = {
        qid: lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)[0]
        for qid in qids
    }

    # Recency anchor: use TE bundles directly from doc_mem (intervals).
    # `recency_scores_for_docs` accepts {doc_id: [{intervals: [...]}]} but
    # build_memory yields {doc_id: {intervals: [...]}} (no list wrapper).
    # Wrap per-doc into a 1-element bundle list for compatibility.
    doc_bundles_for_rec: dict[str, list[dict]] = {}
    for did, mem in doc_mem.items():
        ivs = mem.get("intervals") or []
        if ivs:
            doc_bundles_for_rec[did] = [{"intervals": ivs}]
        else:
            doc_bundles_for_rec[did] = []

    # Semantic + cross-encoder rerank
    print("  embedding + reranking...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # T_lblend scores per query
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

    # Rerank: union(top-50 sem, top-50 T_lblend) -> cross-encoder
    per_q_r_full: dict[str, dict[str, float]] = {}
    per_q_r_partial: dict[str, dict[str, float]] = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        rs = await rerank_topk(reranker, q_text[qid], union, doc_text, len(union))
        per_q_r_partial[qid] = rs
        # Min-max normalize rerank for use in score_blend; tail = 0.
        per_q_r_full[qid] = normalize_rerank_full(
            rs, [d["doc_id"] for d in docs], tail_score=0.0
        )

    # Recency lambda for the chosen half-life
    lam = lambda_for_half_life(HALF_LIFE_DAYS)

    # Per-query switch firing counters (for diagnostic logging)
    n_T_active = 0
    n_Rec_active = 0
    n_both = 0
    n_neither = 0

    # Track per-query examples of switch firings for false-positive auditing
    switch_examples: list[dict] = []

    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue

        # Detect switches
        T_active = has_temporal_anchor(q["text"])
        Recency_active = has_recency_cue(q["text"])

        if T_active:
            n_T_active += 1
        if Recency_active:
            n_Rec_active += 1
        if T_active and Recency_active:
            n_both += 1
        if not T_active and not Recency_active:
            n_neither += 1

        if len(switch_examples) < 12:
            switch_examples.append(
                {
                    "qid": qid,
                    "query": q["text"][:140],
                    "T_active": T_active,
                    "Recency_active": Recency_active,
                }
            )

        t_scores = per_q_t[qid]
        r_full = per_q_r_full[qid]
        rerank_partial = per_q_r_partial[qid]
        s_scores = per_q_s[qid]

        # Recency scores per doc
        rec_scores = recency_scores_for_docs(
            doc_bundles_for_rec,
            doc_ref_us,
            q_ref_us[qid],
            lam,
        )

        # ---- Variant 1: rerank_only -----------------------------------------
        rerank_only_rank = merge_with_tail(
            [
                d
                for d, _ in sorted(
                    rerank_partial.items(), key=lambda x: x[1], reverse=True
                )
            ],
            s_scores,
        )

        # ---- Variant 2: fuse_T_lblend_R (current shipping) -------------------
        # Use existing score_blend over {T, R} with w_T=0.4 (default in
        # force_pick_optimizers_eval.fuse_at_w). Tail by semantic.
        primary_2 = fuse_T_R_blend(t_scores, r_full, w_T=W_T_FUSE_TR)
        rank_fuse_TR = primary_2 + [
            d
            for d in [
                dd
                for dd, _ in sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            if d not in set(primary_2)
        ]

        # ---- Variant 3: fuse_T_R + recency_additive (Agent I's recipe) -------
        # Same as variant 2 but if recency cue detected, add α=0.5 * recency
        # to the fused score before re-ranking.
        # Implementation: compute the final fused scores (not just ranks) by
        # running score_blend, then blend in recency post-hoc when cue.
        fused_TR_scores = dict(
            score_blend(
                {"T": t_scores, "R": r_full},
                {"T": W_T_FUSE_TR, "R": W_R_FUSE_TR},
                top_k_per=40,
                dispersion_cv_ref=CV_REF,
            )
        )
        fused_TR_with_rec = additive_with_recency(
            fused_TR_scores,
            rec_scores,
            cue=Recency_active,
            alpha=ADDITIVE_ALPHA,
        )
        # Tail with semantic for docs missing from score_blend output.
        primary_3 = rank_from_scores(fused_TR_with_rec)
        rank_fuse_TR_recAdd = primary_3 + [
            d for d in rank_from_scores(s_scores) if d not in set(primary_3)
        ]

        # ---- Variant 4: multi_channel_switches -------------------------------
        switches = {"T_active": T_active, "Recency_active": Recency_active}
        primary_4 = multi_channel_blend(
            t_scores,
            rec_scores,
            r_full,
            switches=switches,
            weights={"T": W_T, "recency": W_REC, "R": W_R},
            dispersion_cv_ref=CV_REF,
        )
        rank_mc_switches = primary_4 + [
            d for d in rank_from_scores(s_scores) if d not in set(primary_4)
        ]

        # ---- Variant 5: multi_channel_no_switches ----------------------------
        all_on = {"T_active": True, "Recency_active": True}
        primary_5 = multi_channel_blend(
            t_scores,
            rec_scores,
            r_full,
            switches=all_on,
            weights={"T": W_T, "recency": W_REC, "R": W_R},
            dispersion_cv_ref=CV_REF,
        )
        rank_mc_no_switches = primary_5 + [
            d for d in rank_from_scores(s_scores) if d not in set(primary_5)
        ]

        results.append(
            {
                "qid": qid,
                "qtext": q.get("text", "")[:200],
                "gold": list(gold_set),
                "T_active": T_active,
                "Recency_active": Recency_active,
                "rerank_only": hit_rank(rerank_only_rank, gold_set),
                "fuse_T_R": hit_rank(rank_fuse_TR, gold_set),
                "fuse_T_R_recAdd": hit_rank(rank_fuse_TR_recAdd, gold_set),
                "mc_switches": hit_rank(rank_mc_switches, gold_set),
                "mc_no_switches": hit_rank(rank_mc_no_switches, gold_set),
            }
        )

    return aggregate(
        results, name, n_T_active, n_Rec_active, n_both, n_neither, switch_examples
    )


def aggregate(
    results, label, n_T_active, n_Rec_active, n_both, n_neither, switch_examples
):
    n = len(results)
    out = {
        "label": label,
        "n": n,
        "n_T_active": n_T_active,
        "n_Rec_active": n_Rec_active,
        "n_both": n_both,
        "n_neither": n_neither,
        "switch_examples": switch_examples,
        "per_q": results,
    }

    variants = [
        "rerank_only",
        "fuse_T_R",
        "fuse_T_R_recAdd",
        "mc_switches",
        "mc_no_switches",
    ]
    for var in variants:
        ranks = [r[var] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr_v = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
        out[var] = {
            "R@1": r1 / n if n else 0.0,
            "R@5": r5 / n if n else 0.0,
            "MRR": mrr_v,
            "r1_count": r1,
            "r5_count": r5,
        }

    print(
        f"  n={n}  T_active={n_T_active}  Recency_active={n_Rec_active}  both={n_both}  neither={n_neither}",
        flush=True,
    )
    for var in variants:
        d = out[var]
        print(
            f"  {var:22s}  R@1={d['R@1']:.3f} ({d['r1_count']}/{n})  "
            f"R@5={d['R@5']:.3f} ({d['r5_count']}/{n})  MRR={d['MRR']:.3f}",
            flush=True,
        )
    return out


# -----------------------------------------------------------------------------
# MD report writer
# -----------------------------------------------------------------------------
def write_md(report: dict, path: Path):
    benches = report["benches"]
    lines = []
    lines.append("# T_multi_channel — multi-channel temporal scoring eval\n")
    lines.append(
        f"Channels: T_lblend, recency (half-life={HALF_LIFE_DAYS}d), R (cross-encoder).\n"
    )
    lines.append(f"Weights (all-active): T={W_T}, recency={W_REC}, R={W_R}.\n")
    lines.append(
        f"score_blend with CV gate (cv_ref={CV_REF}); CV-scaled effective weights renormalized.\n"
    )
    lines.append(
        "Switches:\n  - `T_active`: regex matches year / quarter / month / season / era / ISO/slash date / 'March 5th'\n"
        "  - `Recency_active`: `has_recency_cue()` (latest, most recent, recently, ...) with verb-form `present` suppressor\n"
    )

    # ---- R@1 main table -------------------------------------------------------
    lines.append("\n## R@1 by benchmark\n")
    lines.append(
        "| Benchmark | n | T_act | Rec_act | rerank_only | fuse_T_R | fuse_T_R + rec_add | mc_switches | mc_no_switches |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            lines.append(f"| {name} | err | - | - | - | - | - | - | - |")
            continue
        n = b["n"]
        tac = b["n_T_active"]
        rac = b["n_Rec_active"]
        ro = b["rerank_only"]["R@1"]
        ft = b["fuse_T_R"]["R@1"]
        fta = b["fuse_T_R_recAdd"]["R@1"]
        mcs = b["mc_switches"]["R@1"]
        mcn = b["mc_no_switches"]["R@1"]
        lines.append(
            f"| {name} | {n} | {tac} | {rac} | "
            f"{ro:.3f} | {ft:.3f} | {fta:.3f} | {mcs:.3f} | {mcn:.3f} |"
        )
    lines.append("")

    # ---- R@5 table for context ------------------------------------------------
    lines.append("## R@5 by benchmark\n")
    lines.append(
        "| Benchmark | n | rerank_only | fuse_T_R | fuse_T_R + rec_add | mc_switches | mc_no_switches |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            continue
        n = b["n"]
        ro = b["rerank_only"]["R@5"]
        ft = b["fuse_T_R"]["R@5"]
        fta = b["fuse_T_R_recAdd"]["R@5"]
        mcs = b["mc_switches"]["R@5"]
        mcn = b["mc_no_switches"]["R@5"]
        lines.append(
            f"| {name} | {n} | {ro:.3f} | {ft:.3f} | {fta:.3f} | {mcs:.3f} | {mcn:.3f} |"
        )
    lines.append("")

    # ---- Switch firing diagnostics --------------------------------------------
    lines.append("## Switch firing pattern (per benchmark)\n")
    lines.append("| Benchmark | n | T_active | Recency_active | both | neither |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            continue
        n = b["n"]
        lines.append(
            f"| {name} | {n} | {b['n_T_active']} | {b['n_Rec_active']} | "
            f"{b['n_both']} | {b['n_neither']} |"
        )
    lines.append("")

    # ---- Switch examples (qualitative) ---------------------------------------
    lines.append("## Sample switch firings (first 12 queries per benchmark)\n")
    for name, b in benches.items():
        if "error" in b:
            continue
        lines.append(f"### {name}")
        for ex in b.get("switch_examples", []):
            lines.append(
                f"- T={int(ex['T_active'])} R={int(ex['Recency_active'])}: "
                f"`{ex['query']}`"
            )
        lines.append("")

    # ---- Headline answers -----------------------------------------------------
    lines.append("## Headline answers\n")
    if "hard_bench" in benches and "error" not in benches["hard_bench"]:
        hb = benches["hard_bench"]
        lines.append(
            f"- **hard_bench R@1**: rerank_only={hb['rerank_only']['R@1']:.3f}, "
            f"fuse_T_R={hb['fuse_T_R']['R@1']:.3f}, "
            f"fuse_T_R+recAdd={hb['fuse_T_R_recAdd']['R@1']:.3f}, "
            f"mc_switches={hb['mc_switches']['R@1']:.3f}, "
            f"mc_no_switches={hb['mc_no_switches']['R@1']:.3f}"
        )
    if "era_refs" in benches and "error" not in benches["era_refs"]:
        er = benches["era_refs"]
        lines.append(
            f"- **era_refs R@1**: rerank_only={er['rerank_only']['R@1']:.3f}, "
            f"fuse_T_R={er['fuse_T_R']['R@1']:.3f}, "
            f"fuse_T_R+recAdd={er['fuse_T_R_recAdd']['R@1']:.3f}, "
            f"mc_switches={er['mc_switches']['R@1']:.3f}, "
            f"mc_no_switches={er['mc_no_switches']['R@1']:.3f}"
        )
    if "latest_recent" in benches and "error" not in benches["latest_recent"]:
        lr = benches["latest_recent"]
        lines.append(
            f"- **latest_recent R@1**: rerank_only={lr['rerank_only']['R@1']:.3f}, "
            f"fuse_T_R={lr['fuse_T_R']['R@1']:.3f}, "
            f"fuse_T_R+recAdd={lr['fuse_T_R_recAdd']['R@1']:.3f}, "
            f"mc_switches={lr['mc_switches']['R@1']:.3f}, "
            f"mc_no_switches={lr['mc_no_switches']['R@1']:.3f}"
        )
    lines.append("")

    path.write_text("\n".join(lines))


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

    benches_main = [
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
            "latest_recent",
            "latest_recent_docs.jsonl",
            "latest_recent_queries.jsonl",
            "latest_recent_gold.jsonl",
            "edge-latest_recent",
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
            "negation_temporal",
            "negation_temporal_docs.jsonl",
            "negation_temporal_queries.jsonl",
            "negation_temporal_gold.jsonl",
            "edge-negation_temporal",
        ),
    ]

    out = {"benches": {}}
    for name, dp, qp, gp, cache_label in benches_main:
        if not (DATA_DIR / dp).exists():
            alt = f"edge_{dp}"
            if (DATA_DIR / alt).exists():
                dp = alt
        if not (DATA_DIR / qp).exists():
            alt = f"edge_{qp}"
            if (DATA_DIR / alt).exists():
                qp = alt
        if not (DATA_DIR / gp).exists():
            alt = f"edge_{gp}"
            if (DATA_DIR / alt).exists():
                gp = alt
        if not (DATA_DIR / dp).exists():
            print(f"  [{name}] missing {dp} - skipping", flush=True)
            continue
        try:
            agg = await run_bench(name, dp, qp, gp, cache_label, reranker)
            out["benches"][name] = agg
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][name] = {"error": str(e), "n": 0}

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_multi_channel.json"
    json_safe = {"benches": {}}
    for k, v in out["benches"].items():
        if "error" in v:
            json_safe["benches"][k] = v
            continue
        v2 = {kk: vv for kk, vv in v.items() if kk != "per_q"}
        v2["per_q"] = v.get("per_q", [])
        json_safe["benches"][k] = v2
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(f"\nWrote {json_path}", flush=True)

    md_path = out_dir / "T_multi_channel.md"
    write_md(out, md_path)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
