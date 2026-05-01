"""T_open_ended_router evaluation.

Adds a per-query open-ended-date switch that swaps T_lblend → T_v5 in
the score_blend fusion. Goal: capture T_v5's +0.133 R@1 win on
open_ended_date without disturbing T_lblend's wins on the closed-range
benchmarks.

Compare:
  Baseline: fuse_T_lblend_R + recency_additive   (current best)
  New:      fuse_T_router_R   + recency_additive
  Where T_router = T_v5 if has_open_ended_cue(query) else T_lblend.

Reuses the multi_channel_eval pattern (score_blend over {T, R} with
recency_additive on top when recency cue), but switches the T channel
based on the open_ended cue.

Tested on the 11 standard temporal benchmarks.

Writes `results/T_open_ended_router.md` and `results/T_open_ended_router.json`.
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
from t_v5_eval import per_te_bundles_v5, t_v5_doc_scores

# -----------------------------------------------------------------------------
# Open-ended date cue
# -----------------------------------------------------------------------------
# Matches one-sided temporal bounds: phrases like
#   "after 2022", "before 2020", "since 2024", "until 2023",
#   "from 2022 onwards", "<2022", ">2022",
#   "before/after/since/until <event> in/(<MONTH> YYYY)"  (event with explicit date)
#   "before the pandemic" (only if a (Month YYYY) parenthetical follows OR a
#   year is in the same clause).
#
# We DON'T fire on:
#   - causal_relative ("after the migration", "before the launch") — no date
#   - era_refs ("during grad school", "back when ...") — no date
# So the gate requires explicit YYYY OR Month-YYYY in the same query.

_MONTHS = (
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)

# Phrase 1: side-keyword anywhere in the query AND a year/Month-YYYY anywhere.
# Side keywords: after, before, since, until, prior to, post, pre.
_SIDE_KW_RE = re.compile(
    r"\b(?:after|before|since|until|prior\s+to|post|pre)\b",
    re.IGNORECASE,
)
_FROM_ONWARDS_RE = re.compile(
    r"\bfrom\s+(?:19|20)\d{2}\s+onwards?\b",
    re.IGNORECASE,
)
# YYYY or "Month YYYY" or "YYYY-MM-DD"
_DATE_ANCHOR_RE = re.compile(
    r"\b(?:(?:" + _MONTHS + r")\s+(?:19|20)\d{2}"
    r"|(?:19|20)\d{2}-\d{2}-\d{2}"
    r"|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
# Symbolic forms: "<2022", ">2022"
_SYMBOL_RE = re.compile(r"[<>]\s*(?:19|20)\d{2}")


def has_open_ended_cue(query_text: str) -> bool:
    """Return True if the query is a one-sided / open-ended date query.

    Triggers on:
      - "after/before/since/until/prior to/post/pre" + a YYYY or Month YYYY
        appearing anywhere in the query.
      - "from YYYY onwards"
      - symbolic forms "<YYYY" / ">YYYY"

    Does NOT trigger on:
      - causal_relative ("after the migration") — no date anchor
      - era_refs ("during grad school") — no date anchor and no side keyword
    """
    if not query_text:
        return False
    if _FROM_ONWARDS_RE.search(query_text):
        return True
    if _SYMBOL_RE.search(query_text):
        return True
    if _SIDE_KW_RE.search(query_text) and _DATE_ANCHOR_RE.search(query_text):
        return True
    return False


# -----------------------------------------------------------------------------
# Fusion helpers
# -----------------------------------------------------------------------------
HALF_LIFE_DAYS = 21.0
CV_REF = 0.20
W_T_FUSE_TR = 0.4
W_R_FUSE_TR = 0.6
ADDITIVE_ALPHA = 0.5


def fuse_T_R_blend_scores(t_scores, r_scores, w_T=W_T_FUSE_TR):
    """score_blend over {T, R} returning the dict-of-fused-scores."""
    fused = score_blend(
        {"T": t_scores, "R": r_scores},
        {"T": w_T, "R": 1.0 - w_T},
        top_k_per=40,
        dispersion_cv_ref=CV_REF,
    )
    return dict(fused)


def additive_with_recency(base_scores, rec_scores, cue, alpha=ADDITIVE_ALPHA):
    if not cue:
        return dict(base_scores)
    docs = set(base_scores) | set(rec_scores)
    out = {}
    for d in docs:
        out[d] = (1.0 - alpha) * base_scores.get(d, 0.0) + alpha * rec_scores.get(
            d, 0.0
        )
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


# -----------------------------------------------------------------------------
# Bench
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
    lat_db = ROOT / "cache" / "open_ended_router" / f"lat_{name}.sqlite"
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

    # Recency anchor bundles (intervals per doc).
    doc_bundles_for_rec: dict[str, list[dict]] = {}
    for did, mem in doc_mem.items():
        ivs = mem.get("intervals") or []
        doc_bundles_for_rec[did] = [{"intervals": ivs}] if ivs else []

    # Embeddings + semantic
    print("  embedding + reranking...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # T_lblend per query
    per_q_tlb = {
        qid: make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }

    # T_v5 per query
    doc_bundles_v5 = per_te_bundles_v5(doc_ext)
    for d in docs:
        doc_bundles_v5.setdefault(d["doc_id"], [])
    q_bundles_v5 = per_te_bundles_v5(q_ext)
    per_q_tv5 = {
        qid: t_v5_doc_scores(q_bundles_v5.get(qid, []), doc_bundles_v5) for qid in qids
    }

    for qid in qids:
        for d in docs:
            per_q_tlb[qid].setdefault(d["doc_id"], 0.0)
            per_q_tv5[qid].setdefault(d["doc_id"], 0.0)

    # Rerank: union(top-50 sem, top-50 T_lblend, top-50 T_v5)
    per_q_r_full: dict[str, dict[str, float]] = {}
    per_q_r_partial: dict[str, dict[str, float]] = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        tlb_top = topk_from_scores(per_q_tlb[qid], RERANK_TOP_K)
        tv5_top = topk_from_scores(per_q_tv5[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + tlb_top + tv5_top))[
            : int(RERANK_TOP_K * 1.5)
        ]
        rs = await rerank_topk(reranker, q_text[qid], union, doc_text, len(union))
        per_q_r_partial[qid] = rs
        per_q_r_full[qid] = normalize_rerank_full(rs, [d["doc_id"] for d in docs], 0.0)

    lam = lambda_for_half_life(HALF_LIFE_DAYS)

    # Switch firing diagnostics
    n_oe = 0
    n_rec = 0
    oe_examples: list[dict] = []
    oe_fp_audit: list[
        dict
    ] = []  # examples where switch fires but query doesn't look open-ended
    oe_fn_audit: list[
        dict
    ] = []  # cases where it should have fired but didn't (only on open_ended_date bench)

    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue

        oe_active = has_open_ended_cue(q["text"])
        rec_active = has_recency_cue(q["text"])
        if oe_active:
            n_oe += 1
        if rec_active:
            n_rec += 1
        if oe_active and len(oe_examples) < 8:
            oe_examples.append({"qid": qid, "query": q["text"][:140], "bench": name})

        # On open_ended_date, log misses (false negatives)
        if name == "open_ended_date" and not oe_active:
            oe_fn_audit.append({"qid": qid, "query": q["text"][:140]})
        # On benches where we don't expect oe to fire (causal_relative, era_refs,
        # latest_recent), log fires (false positives)
        if (
            name
            in (
                "causal_relative",
                "era_refs",
                "latest_recent",
                "negation_temporal",
                "relative_time",
            )
            and oe_active
        ):
            oe_fp_audit.append({"qid": qid, "query": q["text"][:140], "bench": name})

        t_lb = per_q_tlb[qid]
        t_v5 = per_q_tv5[qid]
        r_full = per_q_r_full[qid]
        rerank_partial = per_q_r_partial[qid]
        s_scores = per_q_s[qid]

        # T_router: pick T_v5 when open_ended cue, else T_lblend
        t_router = t_v5 if oe_active else t_lb

        # Recency
        rec_scores = recency_scores_for_docs(
            doc_bundles_for_rec,
            doc_ref_us,
            q_ref_us[qid],
            lam,
        )

        # ---- rerank_only baseline -----------------------------------------
        rerank_only_rank = merge_with_tail(
            [
                d
                for d, _ in sorted(
                    rerank_partial.items(), key=lambda x: x[1], reverse=True
                )
            ],
            s_scores,
        )

        # ---- BASELINE: fuse_T_lblend_R + recency_additive -------------------
        fused_TR_lb = fuse_T_R_blend_scores(t_lb, r_full, w_T=W_T_FUSE_TR)
        fused_TR_lb_rec = additive_with_recency(
            fused_TR_lb, rec_scores, rec_active, ADDITIVE_ALPHA
        )
        primary_baseline = rank_from_scores(fused_TR_lb_rec)
        rank_baseline = primary_baseline + [
            d for d in rank_from_scores(s_scores) if d not in set(primary_baseline)
        ]

        # ---- NEW: fuse_T_router_R + recency_additive ------------------------
        fused_TR_router = fuse_T_R_blend_scores(t_router, r_full, w_T=W_T_FUSE_TR)
        fused_TR_router_rec = additive_with_recency(
            fused_TR_router, rec_scores, rec_active, ADDITIVE_ALPHA
        )
        primary_router = rank_from_scores(fused_TR_router_rec)
        rank_router = primary_router + [
            d for d in rank_from_scores(s_scores) if d not in set(primary_router)
        ]

        # ---- For diagnostics: pure fuse_T_v5_R (no router) -------------------
        fused_TR_v5 = fuse_T_R_blend_scores(t_v5, r_full, w_T=W_T_FUSE_TR)
        fused_TR_v5_rec = additive_with_recency(
            fused_TR_v5, rec_scores, rec_active, ADDITIVE_ALPHA
        )
        primary_v5_only = rank_from_scores(fused_TR_v5_rec)
        rank_v5_only = primary_v5_only + [
            d for d in rank_from_scores(s_scores) if d not in set(primary_v5_only)
        ]

        results.append(
            {
                "qid": qid,
                "qtext": q.get("text", "")[:200],
                "gold": list(gold_set),
                "oe_active": oe_active,
                "rec_active": rec_active,
                "rerank_only": hit_rank(rerank_only_rank, gold_set),
                "baseline": hit_rank(rank_baseline, gold_set),
                "router": hit_rank(rank_router, gold_set),
                "v5_only": hit_rank(rank_v5_only, gold_set),
            }
        )

    return aggregate(results, name, n_oe, n_rec, oe_examples, oe_fp_audit, oe_fn_audit)


def aggregate(results, label, n_oe, n_rec, oe_examples, oe_fp, oe_fn):
    n = len(results)
    out = {
        "label": label,
        "n": n,
        "n_oe_active": n_oe,
        "n_rec_active": n_rec,
        "oe_examples": oe_examples,
        "oe_fp": oe_fp,
        "oe_fn": oe_fn,
        "per_q": results,
    }
    variants = ["rerank_only", "baseline", "router", "v5_only"]
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
    print(f"  n={n} oe_active={n_oe} rec_active={n_rec}", flush=True)
    for var in variants:
        d = out[var]
        print(
            f"  {var:14s} R@1={d['R@1']:.3f} ({d['r1_count']}/{n}) "
            f"R@5={d['R@5']:.3f} ({d['r5_count']}/{n})",
            flush=True,
        )
    return out


# -----------------------------------------------------------------------------
# MD writer
# -----------------------------------------------------------------------------
def write_md(report: dict, path: Path):
    benches = report["benches"]
    valid = [k for k, v in benches.items() if "error" not in v and v["n"] > 0]
    macro_baseline = sum(benches[k]["baseline"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    macro_router = sum(benches[k]["router"]["R@1"] for k in valid) / max(1, len(valid))
    macro_ro = sum(benches[k]["rerank_only"]["R@1"] for k in valid) / max(1, len(valid))
    macro_v5 = sum(benches[k]["v5_only"]["R@1"] for k in valid) / max(1, len(valid))

    lines = []
    lines.append(
        "# T_open_ended_router — switched T-channel for open-ended date queries\n"
    )
    lines.append(
        "Per-query gate: `has_open_ended_cue(query)` regex on side-keyword (after/before/since/until/prior-to/post/pre) + YYYY or Month-YYYY anchor; OR `from YYYY onwards`; OR symbolic `<YYYY`/`>YYYY`. When fired, T-channel = T_v5; else T_lblend.\n"
    )
    lines.append(
        "Top-line stack: `fuse_T_router_R + recency_additive(α=0.5 when recency_cue)`. Baseline is the same recipe with T_lblend pinned.\n\n"
    )

    # ============================================================
    # R@1 leads
    # ============================================================
    lines.append("## R@1 — Baseline vs Router\n")
    lines.append(
        "| Benchmark | n | oe_act | rec_act | rerank_only | baseline (T_lblend) | **router** | v5_only (no gate) | Δ(router − baseline) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in benches:
        b = benches[name]
        if "error" in b:
            lines.append(f"| {name} | err | - | - | - | - | - | - | - |")
            continue
        n = b["n"]
        oe = b["n_oe_active"]
        rc = b["n_rec_active"]
        ro = b["rerank_only"]["R@1"]
        bl = b["baseline"]["R@1"]
        rt = b["router"]["R@1"]
        v5 = b["v5_only"]["R@1"]
        d = rt - bl
        flag = ""
        if d > 0.005:
            flag = " (router better)"
        elif d < -0.005:
            flag = " (regress!)"
        lines.append(
            f"| {name} | {n} | {oe} | {rc} | "
            f"{ro:.3f} ({b['rerank_only']['r1_count']}/{n}) | "
            f"{bl:.3f} ({b['baseline']['r1_count']}/{n}) | "
            f"**{rt:.3f} ({b['router']['r1_count']}/{n})** | "
            f"{v5:.3f} ({b['v5_only']['r1_count']}/{n}) | "
            f"**{d:+.3f}**{flag} |"
        )
    lines.append("")

    lines.append(f"## Macro-average R@1 across {len(valid)} benches\n")
    lines.append(f"- rerank_only:     {macro_ro:.3f}")
    lines.append(f"- baseline (T_lblend): **{macro_baseline:.3f}**")
    lines.append(f"- **router**:      **{macro_router:.3f}**")
    lines.append(f"- v5_only (no gate): {macro_v5:.3f}")
    lines.append(f"- Δ(router − baseline): **{macro_router - macro_baseline:+.3f}**")
    lines.append(f"- Δ(v5_only − baseline): {macro_v5 - macro_baseline:+.3f}")
    lines.append("")

    # ============================================================
    # Switch firing
    # ============================================================
    lines.append("## Switch firing pattern\n")
    lines.append("| Benchmark | n | oe_active | %  | expected? |")
    lines.append("|---|---:|---:|---:|---|")
    expectations = {
        "open_ended_date": "yes (15/15 ideal)",
        "hard_bench": "no (closed-range 'in YYYY')",
        "temporal_essential": "no",
        "tempreason_small": "varies",
        "conjunctive_temporal": "no",
        "multi_te_doc": "no",
        "relative_time": "no",
        "era_refs": "no (event refs, no date)",
        "causal_relative": "no (event refs, no date)",
        "latest_recent": "no (recency, no anchor)",
        "negation_temporal": "no (mostly closed-range)",
    }
    for name in benches:
        b = benches[name]
        if "error" in b:
            continue
        n = b["n"]
        oe = b["n_oe_active"]
        pct = oe / n if n else 0.0
        lines.append(
            f"| {name} | {n} | {oe} | {pct * 100:.0f}% | {expectations.get(name, '-')} |"
        )
    lines.append("")

    # ============================================================
    # False positives
    # ============================================================
    fp_total = sum(len(benches[k].get("oe_fp", [])) for k in valid)
    fn_total = sum(len(benches[k].get("oe_fn", [])) for k in valid)
    lines.append("## Switch correctness audit\n")
    lines.append(f"- False positives across non-open-ended benches: **{fp_total}**")
    lines.append(f"- False negatives on `open_ended_date`: **{fn_total}** / 15")
    lines.append("")
    if fp_total > 0:
        lines.append("### False-positive examples\n")
        for k in valid:
            fps = benches[k].get("oe_fp", [])
            if fps:
                lines.append(f"**{k}**:")
                for fp in fps[:5]:
                    lines.append(f"- `{fp['qid']}`: {fp['query']}")
                lines.append("")
    if fn_total > 0:
        lines.append("### False-negative examples (on open_ended_date)\n")
        for fn in benches.get("open_ended_date", {}).get("oe_fn", []):
            lines.append(f"- `{fn['qid']}`: {fn['query']}")
        lines.append("")

    # ============================================================
    # open_ended_date deep dive
    # ============================================================
    lines.append("## Did the router capture T_v5's win on open_ended_date?\n")
    if "open_ended_date" in benches and "error" not in benches["open_ended_date"]:
        oed = benches["open_ended_date"]
        n = oed["n"]
        bl = oed["baseline"]["R@1"]
        rt = oed["router"]["R@1"]
        v5 = oed["v5_only"]["R@1"]
        ro = oed["rerank_only"]["R@1"]
        lines.append(f"- rerank_only:  {ro:.3f} ({oed['rerank_only']['r1_count']}/{n})")
        lines.append(
            f"- baseline (T_lblend): {bl:.3f} ({oed['baseline']['r1_count']}/{n})"
        )
        lines.append(f"- **router**:    **{rt:.3f}** ({oed['router']['r1_count']}/{n})")
        lines.append(f"- v5_only:      {v5:.3f} ({oed['v5_only']['r1_count']}/{n})")
        lines.append(f"- Δ(router − baseline) = {rt - bl:+.3f}  (target ≥ +0.133)")
        lines.append("")
        if rt - bl >= 0.10:
            lines.append("**YES** — router captured T_v5's open-ended-date advantage.")
        elif rt - bl >= 0.02:
            lines.append(
                "**PARTIALLY** — router improves but does not fully match T_v5's standalone fusion delta."
            )
        else:
            lines.append("**NO** — router did not capture T_v5's edge.")
        lines.append("")

    # ============================================================
    # Regression check
    # ============================================================
    lines.append("## Regression check (other benches)\n")
    regressions = []
    for name in valid:
        if name == "open_ended_date":
            continue
        b = benches[name]
        d = b["router"]["R@1"] - b["baseline"]["R@1"]
        if d <= -0.005:
            regressions.append((name, d))
    if not regressions:
        lines.append("No regressions on non-open-ended benches.")
    else:
        lines.append("| Benchmark | Δ(router − baseline) |")
        lines.append("|---|---:|")
        for name, d in sorted(regressions, key=lambda x: x[1]):
            lines.append(f"| {name} | {d:+.3f} |")
    lines.append("")

    # ============================================================
    # Recommendation
    # ============================================================
    lines.append("## Recommendation\n")
    captured = (
        "open_ended_date" in benches
        and "error" not in benches["open_ended_date"]
        and (
            benches["open_ended_date"]["router"]["R@1"]
            - benches["open_ended_date"]["baseline"]["R@1"]
        )
        >= 0.10
    )
    no_regress = len(regressions) == 0 and macro_router >= macro_baseline - 0.005
    if captured and no_regress:
        lines.append(
            "**SHIP** the open-ended router. Captures T_v5's open_ended_date win without regressing other benches."
        )
    elif captured and not no_regress:
        lines.append(
            "**SHIP WITH CAVEAT** — router captures open-ended-date win but introduces small regressions elsewhere. Tune the cue regex (tighten on regressing benches) before shipping."
        )
    elif no_regress and not captured:
        lines.append(
            "**HOLD** — router does no harm but doesn't capture the open-ended-date advantage. Investigate the cue gate or the T_v5 score weight."
        )
    else:
        lines.append(
            "**DO NOT SHIP** — router fails to capture the open-ended-date win and introduces regressions."
        )
    lines.append("")

    # Sample switch firings
    lines.append("## Sample firings (router fired on these)\n")
    for name in valid:
        b = benches[name]
        ex = b.get("oe_examples", [])
        if not ex:
            continue
        lines.append(f"### {name}")
        for e in ex:
            lines.append(f"- `{e['qid']}`: {e['query']}")
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

    benches = [
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

    out = {"benches": {}}
    for name, dp, qp, gp, cl in benches:
        try:
            agg = await run_bench(name, dp, qp, gp, cl, reranker)
            out["benches"][name] = agg
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][name] = {"error": str(e), "n": 0}

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "T_open_ended_router.json"
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

    md_path = out_dir / "T_open_ended_router.md"
    write_md(out, md_path)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
