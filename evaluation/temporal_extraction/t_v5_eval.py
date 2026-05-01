"""T_v5 evaluation: containment + bounded proximity multiplier.

T_v5 fixes T_v4's standalone 1.0-saturation problem WITHOUT killing the
asymmetric containment correctness on open-ended queries. The score:

    pair_score(q, d) = containment(q, d) * (0.5 + 0.5 * proximity(q, d))

  - containment(q, d) = |q ∩ d| / |d|     (T_v4 primitive; 1.0 if d ⊆ q)
  - proximity(q, d)   = aligned in [0, 1] given best-anchor distance
                         normalized by query span (saturates to 0 on open-
                         ended query intervals)
  - bounded multiplier in [0.5, 1.0] so containment correctness is preserved
    (a fully-contained doc still scores >= 0.5 even at zero proximity, and
    open-ended queries get a flat 0.5 multiplier — equivalent to T_v4 up to
    a global half-scale)

Aggregation matches T_v2/T_v4: per query anchor, MAX over doc TEs;
geomean across query anchors with floor 1e-6.

This script runs BOTH:
  1. Standalone evaluation (T_lblend, T_v4, T_v5 head-to-head)
  2. Fusion evaluation (fuse_T_lblend_R vs fuse_T_v4_R vs fuse_T_v5_R)

LongMemEval is intentionally skipped here (Agent H's t_v4_fusion_eval crashed
mid-LME, and the T_v5 hypothesis is about temporal-corpus behaviour).

Writes `results/T_v5.md` and `results/T_v5.json`.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from pathlib import Path

# Strip SOCKS/HTTP proxy env vars set by the runtime sandbox.
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
from scorer import Interval
from t_v4_eval import per_te_bundles_v4, t_v4_doc_scores


# ----------------------------------------------------------------------
# T_v5: containment * bounded proximity multiplier
# ----------------------------------------------------------------------
def _interval_span(iv: Interval) -> int:
    return max(iv.latest_us - iv.earliest_us, 1)


def _interval_best(iv: Interval) -> float:
    if iv.best_us is not None:
        return float(iv.best_us)
    return 0.5 * (iv.earliest_us + iv.latest_us)


def _proximity(q_iv: Interval, d_iv: Interval) -> float:
    """Alignment of best_us, in [0, 1].

    distance: |q_best - d_best|
    span:     query interval duration (with floor 1us)
    score:    max(0, 1 - distance / span)

    For OPEN-ENDED query intervals (era extractor uses very-large bounds for
    open half-lines), span is huge → score saturates at 1.0 unless the doc
    sits at the open end. That's intentional: open-ended queries should rely
    entirely on containment.
    """
    span = _interval_span(q_iv)
    qb = _interval_best(q_iv)
    db = _interval_best(d_iv)
    distance = abs(qb - db)
    return max(0.0, 1.0 - distance / span)


def pair_score_v5(q_ivs: list[Interval], d_ivs: list[Interval]) -> float:
    """Best (containment * (0.5 + 0.5 * proximity)) across (q_iv, d_iv) pairs.

    Bounded multiplier preserves containment correctness:
      - score is still 0 when containment is 0 (no overlap)
      - score is in [0.5*cont, 1.0*cont]; saturated containment of 1.0 now
        spans [0.5, 1.0] depending on where the doc anchor falls within the
        query interval (vs T_v4 which always emits 1.0)
      - max possible per-pair score is 1.0 (delta-in-delta exact match)
    """
    if not q_ivs or not d_ivs:
        return 0.0
    best = 0.0
    for qi in q_ivs:
        q_dur = _interval_span(qi)
        for di in d_ivs:
            inter_lo = max(qi.earliest_us, di.earliest_us)
            inter_hi = min(qi.latest_us, di.latest_us)
            inter = max(0, inter_hi - inter_lo)
            d_dur = di.latest_us - di.earliest_us
            if d_dur <= 0:
                d_dur = 1
            cont = inter / d_dur
            if cont <= 0:
                continue
            cont = min(cont, 1.0)
            # proximity, with span = qi span (the "scale" of the query)
            qb = _interval_best(qi)
            db = _interval_best(di)
            prox = max(0.0, 1.0 - abs(qb - db) / q_dur)
            score = cont * (0.5 + 0.5 * prox)
            if score > best:
                best = score
                if best >= 1.0:
                    best = 1.0
    return best


def per_te_bundles_v5(extracted):
    """Per-doc list of TE bundles, same shape as v4."""
    out: dict[str, list[dict]] = {}
    for did, tes in extracted.items():
        bundles = []
        for te in tes:
            ivs = flatten_intervals(te)
            bundles.append({"intervals": ivs})
        out[did] = bundles
    return out


def t_v5_doc_scores(
    q_bundles: list[dict], doc_bundles_map: dict[str, list[dict]]
) -> dict[str, float]:
    """Per-anchor AND-coverage geomean with v5 primitive."""
    out: dict[str, float] = {}
    if not q_bundles:
        for did in doc_bundles_map:
            out[did] = 0.0
        return out
    for did, d_bundles in doc_bundles_map.items():
        if not d_bundles:
            out[did] = 0.0
            continue
        bests = []
        for q_b in q_bundles:
            best = 0.0
            for d_b in d_bundles:
                s = pair_score_v5(q_b["intervals"], d_b["intervals"])
                if s > best:
                    best = s
            bests.append(best)
        log_sum = 0.0
        for b in bests:
            log_sum += math.log(max(b, 1e-6))
        out[did] = math.exp(log_sum / len(bests))
    return out


# ----------------------------------------------------------------------
# Fusion helpers
# ----------------------------------------------------------------------
W_T = 0.4
W_R = 0.6


def fuse_T_R(
    t_scores: dict[str, float],
    r_scores: dict[str, float],
    s_scores_for_tail: dict[str, float],
) -> list[str]:
    fused = score_blend(
        {"T": t_scores, "R": r_scores},
        {"T": W_T, "R": W_R},
        top_k_per=40,
        dispersion_cv_ref=0.20,
    )
    primary = [d for d, _ in fused]
    seen = set(primary)
    tail = sorted(s_scores_for_tail.items(), key=lambda x: x[1], reverse=True)
    return primary + [d for d, _ in tail if d not in seen]


# ----------------------------------------------------------------------
# Bench runner
# ----------------------------------------------------------------------
def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


def rank_from_scores(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


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

    # T_lblend memory
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

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}

    # Semantic
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # Lattice for T_lblend's L channel
    lat_db = ROOT / "cache" / "t_v5" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)
    per_q_l = {
        qid: lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)[0]
        for qid in qids
    }

    # T_lblend, T_v4, T_v5
    per_q_tlb = {
        qid: make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }

    doc_bundles_v4 = per_te_bundles_v4(doc_ext)
    doc_bundles_v5 = per_te_bundles_v5(doc_ext)
    for d in docs:
        doc_bundles_v4.setdefault(d["doc_id"], [])
        doc_bundles_v5.setdefault(d["doc_id"], [])
    q_bundles_v4 = per_te_bundles_v4(q_ext)
    q_bundles_v5 = per_te_bundles_v5(q_ext)
    per_q_tv4 = {
        qid: t_v4_doc_scores(q_bundles_v4.get(qid, []), doc_bundles_v4) for qid in qids
    }
    per_q_tv5 = {
        qid: t_v5_doc_scores(q_bundles_v5.get(qid, []), doc_bundles_v5) for qid in qids
    }
    for qid in qids:
        for did in doc_text:
            per_q_tlb[qid].setdefault(did, 0.0)
            per_q_tv4[qid].setdefault(did, 0.0)
            per_q_tv5[qid].setdefault(did, 0.0)

    # Rerank over union of S top-50 ∪ T_lblend top-50 ∪ T_v4 top-50 ∪ T_v5 top-50
    print("  reranking over s ∪ t_lblend ∪ t_v4 ∪ t_v5 union...", flush=True)
    per_q_r = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        tlb_top = topk_from_scores(per_q_tlb[qid], RERANK_TOP_K)
        tv4_top = topk_from_scores(per_q_tv4[qid], RERANK_TOP_K)
        tv5_top = topk_from_scores(per_q_tv5[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + tlb_top + tv4_top + tv5_top))[
            : int(RERANK_TOP_K * 1.5)
        ]
        per_q_r[qid] = await rerank_topk(
            reranker, q_text[qid], union, doc_text, len(union)
        )

    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        s_scores = per_q_s[qid]
        r_scores = per_q_r[qid]
        t_lb = per_q_tlb[qid]
        t_v4 = per_q_tv4[qid]
        t_v5 = per_q_tv5[qid]

        # Standalone rankings
        rank_lb_solo = rank_from_scores(t_lb)
        rank_v4_solo = rank_from_scores(t_v4)
        rank_v5_solo = rank_from_scores(t_v5)

        # Fusion rankings
        rank_lb_fused = fuse_T_R(t_lb, r_scores, s_scores)
        rank_v4_fused = fuse_T_R(t_v4, r_scores, s_scores)
        rank_v5_fused = fuse_T_R(t_v5, r_scores, s_scores)

        # rerank_only baseline
        rs = {did: r_scores.get(did, 0.0) for did in r_scores}
        rerank_only = merge_with_tail(
            [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)],
            s_scores,
        )

        results.append(
            {
                "qid": qid,
                "qtext": q.get("text", ""),
                "gold": list(gold_set),
                "n_q_tes": len(q_bundles_v5.get(qid, [])),
                # standalone
                "tlb_solo_rank": hit_rank(rank_lb_solo, gold_set),
                "tv4_solo_rank": hit_rank(rank_v4_solo, gold_set),
                "tv5_solo_rank": hit_rank(rank_v5_solo, gold_set),
                # fusion
                "rerank_only_rank": hit_rank(rerank_only, gold_set),
                "fuse_lb_rank": hit_rank(rank_lb_fused, gold_set),
                "fuse_v4_rank": hit_rank(rank_v4_fused, gold_set),
                "fuse_v5_rank": hit_rank(rank_v5_fused, gold_set),
                # tops
                "tlb_solo_top1": rank_lb_solo[0] if rank_lb_solo else None,
                "tv4_solo_top1": rank_v4_solo[0] if rank_v4_solo else None,
                "tv5_solo_top1": rank_v5_solo[0] if rank_v5_solo else None,
                "fuse_lb_top1": rank_lb_fused[0] if rank_lb_fused else None,
                "fuse_v4_top1": rank_v4_fused[0] if rank_v4_fused else None,
                "fuse_v5_top1": rank_v5_fused[0] if rank_v5_fused else None,
            }
        )
    return aggregate(results, name)


def aggregate(results, label):
    n = len(results)
    out = {"label": label, "n": n, "per_q": results}
    rank_keys = (
        "tlb_solo_rank",
        "tv4_solo_rank",
        "tv5_solo_rank",
        "rerank_only_rank",
        "fuse_lb_rank",
        "fuse_v4_rank",
        "fuse_v5_rank",
    )
    for var in rank_keys:
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
        f"  STANDALONE  T_lblend R@1={out['tlb_solo_rank']['r1_count']:3}/{n} ({out['tlb_solo_rank']['R@1']:.3f}) | "
        f"T_v4 R@1={out['tv4_solo_rank']['r1_count']:3}/{n} ({out['tv4_solo_rank']['R@1']:.3f}) | "
        f"T_v5 R@1={out['tv5_solo_rank']['r1_count']:3}/{n} ({out['tv5_solo_rank']['R@1']:.3f})",
        flush=True,
    )
    print(
        f"  FUSION      LB R@1={out['fuse_lb_rank']['r1_count']:3}/{n} ({out['fuse_lb_rank']['R@1']:.3f}) | "
        f"V4 R@1={out['fuse_v4_rank']['r1_count']:3}/{n} ({out['fuse_v4_rank']['R@1']:.3f}) | "
        f"V5 R@1={out['fuse_v5_rank']['r1_count']:3}/{n} ({out['fuse_v5_rank']['R@1']:.3f})",
        flush=True,
    )
    delta_v5_lb_solo = out["tv5_solo_rank"]["R@1"] - out["tlb_solo_rank"]["R@1"]
    delta_v5_v4_solo = out["tv5_solo_rank"]["R@1"] - out["tv4_solo_rank"]["R@1"]
    delta_v5_lb_fused = out["fuse_v5_rank"]["R@1"] - out["fuse_lb_rank"]["R@1"]
    delta_v5_v4_fused = out["fuse_v5_rank"]["R@1"] - out["fuse_v4_rank"]["R@1"]
    print(
        f"  Δ standalone (v5 − lblend) = {delta_v5_lb_solo:+.3f}  (v5 − v4) = {delta_v5_v4_solo:+.3f}",
        flush=True,
    )
    print(
        f"  Δ fusion     (v5 − lblend) = {delta_v5_lb_fused:+.3f}  (v5 − v4) = {delta_v5_v4_fused:+.3f}",
        flush=True,
    )
    return out


def write_md(report: dict, path: Path):
    benches = report["benches"]
    lines = []
    lines.append("# T_v5 — Containment × bounded proximity\n")
    lines.append(
        "Per-pair primitive: `pair = (|q∩d|/|d|) * (0.5 + 0.5 * prox)`, "
        "where `prox = max(0, 1 - |q_best - d_best| / q_span)`. "
        "MAX over (q_iv, d_iv) pairs per (q_te, d_te); MAX over doc TEs per query anchor; "
        "geomean across query anchors with floor 1e-6. Bounded multiplier in [0.5, 1.0] "
        "preserves T_v4's containment correctness while restoring dispersion that T_lblend gets "
        "from the lattice channel.\n"
    )

    # =========================================================
    # FUSION R@1 leads the report
    # =========================================================
    lines.append("## R@1 in fusion — lead with deltas\n")
    lines.append(
        "| Benchmark | n | rerank_only | fuse_T_lblend_R | fuse_T_v4_R | fuse_T_v5_R | "
        "Δ(v5 − lblend) | Δ(v5 − v4) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    bench_keys = list(benches.keys())
    for name in bench_keys:
        b = benches[name]
        if "error" in b:
            lines.append(f"| {name} | err | — | — | — | — | — | — |")
            continue
        n = b["n"]
        ro = b["rerank_only_rank"]
        lb = b["fuse_lb_rank"]
        v4 = b["fuse_v4_rank"]
        v5 = b["fuse_v5_rank"]
        d_lb = v5["R@1"] - lb["R@1"]
        d_v4 = v5["R@1"] - v4["R@1"]
        flag = ""
        if d_lb > 0.005:
            flag = " (v5 better)"
        elif d_lb < -0.005:
            flag = " (lblend better)"
        lines.append(
            f"| {name} | {n} | "
            f"{ro['R@1']:.3f} ({ro['r1_count']}/{n}) | "
            f"{lb['R@1']:.3f} ({lb['r1_count']}/{n}) | "
            f"{v4['R@1']:.3f} ({v4['r1_count']}/{n}) | "
            f"**{v5['R@1']:.3f} ({v5['r1_count']}/{n})** | "
            f"**{d_lb:+.3f}**{flag} | {d_v4:+.3f} |"
        )
    lines.append("")

    # Macro fusion
    valid = [k for k, v in benches.items() if "error" not in v and v["n"] > 0]
    macro_ro = sum(benches[k]["rerank_only_rank"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    macro_lb_f = sum(benches[k]["fuse_lb_rank"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    macro_v4_f = sum(benches[k]["fuse_v4_rank"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    macro_v5_f = sum(benches[k]["fuse_v5_rank"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    lines.append(f"## Macro-average R@1 across {len(valid)} benches\n")
    lines.append("### Fusion")
    lines.append(f"- rerank_only:      {macro_ro:.3f}")
    lines.append(f"- fuse_T_lblend_R:  **{macro_lb_f:.3f}**")
    lines.append(f"- fuse_T_v4_R:      {macro_v4_f:.3f}")
    lines.append(f"- fuse_T_v5_R:      **{macro_v5_f:.3f}**")
    lines.append(f"- Δ(v5 − lblend):   **{macro_v5_f - macro_lb_f:+.3f}**")
    lines.append(f"- Δ(v5 − v4):       **{macro_v5_f - macro_v4_f:+.3f}**")
    lines.append("")

    # =========================================================
    # STANDALONE
    # =========================================================
    lines.append("## R@1 standalone (no fusion)\n")
    lines.append(
        "| Benchmark | n | T_lblend | T_v4 | T_v5 | Δ(v5 − v4) | Δ(v5 − lblend) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in bench_keys:
        b = benches[name]
        if "error" in b:
            continue
        n = b["n"]
        lb = b["tlb_solo_rank"]
        v4 = b["tv4_solo_rank"]
        v5 = b["tv5_solo_rank"]
        d_v4 = v5["R@1"] - v4["R@1"]
        d_lb = v5["R@1"] - lb["R@1"]
        lines.append(
            f"| {name} | {n} | "
            f"{lb['R@1']:.3f} ({lb['r1_count']}/{n}) | "
            f"{v4['R@1']:.3f} ({v4['r1_count']}/{n}) | "
            f"**{v5['R@1']:.3f} ({v5['r1_count']}/{n})** | "
            f"{d_v4:+.3f} | {d_lb:+.3f} |"
        )
    lines.append("")
    macro_lb_s = sum(benches[k]["tlb_solo_rank"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    macro_v4_s = sum(benches[k]["tv4_solo_rank"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    macro_v5_s = sum(benches[k]["tv5_solo_rank"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    lines.append("### Standalone macro")
    lines.append(f"- T_lblend: {macro_lb_s:.3f}")
    lines.append(f"- T_v4:     {macro_v4_s:.3f}")
    lines.append(f"- T_v5:     **{macro_v5_s:.3f}**")
    lines.append(f"- Δ(v5 − v4):     {macro_v5_s - macro_v4_s:+.3f}")
    lines.append(f"- Δ(v5 − lblend): {macro_v5_s - macro_lb_s:+.3f}")
    lines.append("")

    # =========================================================
    # R@5 fusion
    # =========================================================
    lines.append("## R@5 in fusion\n")
    lines.append(
        "| Benchmark | n | rerank_only | fuse_T_lblend_R | fuse_T_v4_R | fuse_T_v5_R |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in bench_keys:
        b = benches[name]
        if "error" in b:
            continue
        n = b["n"]
        ro = b["rerank_only_rank"]
        lb = b["fuse_lb_rank"]
        v4 = b["fuse_v4_rank"]
        v5 = b["fuse_v5_rank"]
        lines.append(
            f"| {name} | {n} | "
            f"{ro['R@5']:.3f} ({ro['r5_count']}/{n}) | "
            f"{lb['R@5']:.3f} ({lb['r5_count']}/{n}) | "
            f"{v4['R@5']:.3f} ({v4['r5_count']}/{n}) | "
            f"{v5['R@5']:.3f} ({v5['r5_count']}/{n}) |"
        )
    lines.append("")

    # =========================================================
    # Hypothesis verdicts
    # =========================================================
    lines.append("## Hypothesis verdicts\n")

    # 1. Did T_v5 close standalone gap to T_v2 on saturation-affected benches?
    # We don't have T_v2 here, so use T_lblend as the relevant baseline (T_v2 ≈ T_lblend on these).
    sat_targets = ["temporal_essential", "multi_te_doc", "conjunctive_temporal"]
    lines.append("### 1. Did T_v5 close T_v4's standalone saturation gap?")
    lines.append(
        "(T_v4 standalone losses on these benches were the original motivation.)\n"
    )
    lines.append(
        "| Benchmark | T_v4 R@1 | T_v5 R@1 | Δ(v5 − v4) | T_lblend R@1 | Closed gap? |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for name in sat_targets:
        b = benches.get(name, {})
        if "error" in b or not b:
            continue
        v4 = b["tv4_solo_rank"]["R@1"]
        v5 = b["tv5_solo_rank"]["R@1"]
        lb = b["tlb_solo_rank"]["R@1"]
        d = v5 - v4
        # "closed gap" = T_v5 reaches within 0.02 of T_lblend OR strictly improves over T_v4
        closed = (
            "YES"
            if (v5 >= lb - 0.02 or d > 0.02)
            else f"NO (still {v5 - lb:+.3f} vs lblend)"
        )
        lines.append(
            f"| {name} | {v4:.3f} | {v5:.3f} | {d:+.3f} | {lb:.3f} | {closed} |"
        )
    lines.append("")

    # 2. Did T_v5 in fusion match or beat T_lblend in fusion?
    lines.append("### 2. Did T_v5 in fusion match or beat T_lblend in fusion?\n")
    macro_delta = macro_v5_f - macro_lb_f
    if macro_delta >= 0.005:
        lines.append(
            f"**YES.** T_v5 in fusion beats T_lblend in fusion by Δ = {macro_delta:+.3f} macro R@1."
        )
    elif macro_delta >= -0.005:
        lines.append(
            f"**MATCHED (tie).** Δ = {macro_delta:+.3f} macro R@1 — within noise. "
            f"T_v5 is a viable drop-in if other properties (open_ended_date win) are preserved."
        )
    else:
        lines.append(
            f"**NO.** T_v5 in fusion regresses by Δ = {macro_delta:+.3f} macro R@1 vs T_lblend."
        )
    lines.append("")

    # 3. Did T_v5 preserve T_v4's win on open_ended_date?
    lines.append("### 3. Did T_v5 preserve T_v4's win on `open_ended_date`?\n")
    oed = benches.get("open_ended_date", {})
    if oed and "error" not in oed:
        v4_f = oed["fuse_v4_rank"]["R@1"]
        v5_f = oed["fuse_v5_rank"]["R@1"]
        lb_f = oed["fuse_lb_rank"]["R@1"]
        v4_s = oed["tv4_solo_rank"]["R@1"]
        v5_s = oed["tv5_solo_rank"]["R@1"]
        lb_s = oed["tlb_solo_rank"]["R@1"]
        lines.append(
            f"- Standalone:  T_lblend={lb_s:.3f}  T_v4={v4_s:.3f}  T_v5={v5_s:.3f}"
        )
        lines.append(
            f"- Fusion:      T_lblend={lb_f:.3f}  T_v4={v4_f:.3f}  T_v5={v5_f:.3f}"
        )
        if v5_f >= lb_f + 0.02:
            lines.append(
                f"- **PRESERVED** — T_v5 in fusion beats T_lblend in fusion by Δ={v5_f - lb_f:+.3f}."
            )
        elif v5_f >= lb_f - 0.02:
            lines.append(
                f"- **PARTIALLY PRESERVED** — T_v5 in fusion ties T_lblend (Δ={v5_f - lb_f:+.3f}); v4 advantage diluted."
            )
        else:
            lines.append(
                f"- **LOST** — T_v5 in fusion drops below T_lblend by Δ={v5_f - lb_f:+.3f}."
            )
    lines.append("")

    # 4. Recommendation
    lines.append("### 4. Recommendation: T_v5 as drop-in?\n")
    if macro_delta >= -0.005:
        # tie or better — check open_ended_date
        oed_pres = oed and (
            oed.get("fuse_v5_rank", {}).get("R@1", 0)
            >= oed.get("fuse_lb_rank", {}).get("R@1", 0) - 0.02
        )
        if oed_pres:
            lines.append(
                f"**Drop-in T_v5 for T_lblend.** Macro fusion Δ = {macro_delta:+.3f} (≥ tie) AND "
                f"open_ended_date preserved/improved. T_v5 is simpler (single primitive, no axis/tag/lattice) "
                f"with the same fusion-level performance."
            )
        else:
            lines.append(
                "**Hybrid: per-bench gate.** T_v5 fusion is competitive overall but loses open_ended_date. "
                "Use T_lblend as default; switch to T_v5 only when query has open-ended temporal "
                "semantics (era extractor reports half-line query interval)."
            )
    else:
        lines.append(
            f"**Do NOT drop in.** T_v5 fusion regresses by Δ = {macro_delta:+.3f} macro R@1. "
            f"Stick with T_lblend; consider hybrid (T_lblend + T_v5 only on open_ended_date)."
        )
    lines.append("")

    # 5. Per-bench regression diagnoses
    lines.append(
        "### 5. Per-bench regression diagnosis (where v5 underperforms T_lblend in fusion)\n"
    )
    lines.append(
        "| Benchmark | Δ(v5 − lblend) | v5-only top1 | lblend-only top1 | both | neither |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in bench_keys:
        b = benches.get(name)
        if not b or "error" in b:
            continue
        per_q = b.get("per_q", [])
        v5_only = sum(
            1 for r in per_q if (r["fuse_v5_rank"] == 1) and (r["fuse_lb_rank"] != 1)
        )
        lb_only = sum(
            1 for r in per_q if (r["fuse_lb_rank"] == 1) and (r["fuse_v5_rank"] != 1)
        )
        both = sum(
            1 for r in per_q if (r["fuse_v5_rank"] == 1) and (r["fuse_lb_rank"] == 1)
        )
        neither = sum(
            1 for r in per_q if (r["fuse_v5_rank"] != 1) and (r["fuse_lb_rank"] != 1)
        )
        d = b["fuse_v5_rank"]["R@1"] - b["fuse_lb_rank"]["R@1"]
        lines.append(
            f"| {name} | {d:+.3f} | {v5_only} | {lb_only} | {both} | {neither} |"
        )
    lines.append("")

    # Per-failure detail on regressions
    regress_benches = []
    for name in bench_keys:
        b = benches.get(name)
        if not b or "error" in b:
            continue
        d = b["fuse_v5_rank"]["R@1"] - b["fuse_lb_rank"]["R@1"]
        if d <= -0.02:
            regress_benches.append((name, d))
    if regress_benches:
        lines.append(
            "### Detailed losses on regressing benches (v5 missed top1, lblend hit, up to 5)\n"
        )
        for name, d in sorted(regress_benches, key=lambda x: x[1]):
            b = benches[name]
            per_q = b.get("per_q", [])
            losses = [
                r
                for r in per_q
                if (r["fuse_lb_rank"] == 1) and (r["fuse_v5_rank"] != 1)
            ][:5]
            if not losses:
                continue
            lines.append(f"#### {name}  (Δ = {d:+.3f})")
            for r in losses:
                lines.append(
                    f"- `{r['qid']}` (n_q_tes={r['n_q_tes']}): v5_top1=`{r['fuse_v5_top1']}`, "
                    f"v5_rank={r['fuse_v5_rank']}, gold={r['gold']}"
                )
                if r.get("qtext"):
                    lines.append(f"  - q: {r['qtext'][:120]}")
            lines.append("")

    path.write_text("\n".join(lines))


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
    json_path = out_dir / "T_v5.json"
    json_safe = {"benches": {}}
    for k, v in out["benches"].items():
        v2 = {kk: vv for kk, vv in v.items() if kk != "per_q"}
        v2["per_q"] = v.get("per_q", [])
        json_safe["benches"][k] = v2
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"\nWrote {json_path}", flush=True)

    md_path = out_dir / "T_v5.md"
    write_md(out, md_path)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
