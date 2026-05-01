"""T_lblend channel ablation — does T earn its place as scoring & retrieval?

Variants (all use the LLM planner for filter/abs_overlap):
  A. filter_only          : union(sem top-50, T top-50) -> rerank -> filter * rerank.
                            (T retrieved but not scored.)
  B. filter_T (current)   : union(sem top-50, T top-50) -> rerank ->
                            filter * score_blend({T, R}, CV gate).
  C. filter_no_T_retrieval: sem top-50 only -> rerank -> filter * rerank.
                            (T neither retrieved nor scored.)
  D. T_retrieval_wider    : union(sem top-50, T top-100) -> rerank ->
                            filter * score_blend({T, R}, CV gate).

Filter = abs_overlap window from QueryPlan.absolute_anchor (0/1 multiplier).
Mirrors composition_eval_v2.py minus negation/causal/recency stack
(decay/recency/causal removed per directive). For variant parity we only
apply absolute-window filter (the principal "filter from LLM plan").
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
    rerank_topk,
    topk_from_scores,
)
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from query_planner import QueryPlan, QueryPlanner
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
from schema import to_us

CV_REF = 0.20
W_T_FUSE_TR = 0.4
ABSOLUTE_OUTSIDE_FACTOR = 0.0


def fuse_TR(t_scores, r_scores, w_T=W_T_FUSE_TR):
    fused = score_blend(
        {"T": t_scores, "R": r_scores},
        {"T": w_T, "R": 1.0 - w_T},
        top_k_per=40,
        dispersion_cv_ref=CV_REF,
    )
    return dict(fused)


def normalize_dict(d):
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in d.items()}


def normalize_rerank_full(rerank_partial, all_doc_ids, tail_score=0.0):
    if not rerank_partial:
        return dict.fromkeys(all_doc_ids, tail_score)
    vals = list(rerank_partial.values())
    rmin, rmax = min(vals), max(vals)
    span = (rmax - rmin) or 1.0
    return {
        did: ((rerank_partial[did] - rmin) / span)
        if did in rerank_partial
        else tail_score
        for did in all_doc_ids
    }


def absolute_overlap_scores(doc_ivs_flat, anchor_ivs):
    if not anchor_ivs:
        return dict.fromkeys(doc_ivs_flat, 1.0)
    out = {}
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


def rank_from_scores(scores):
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


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

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===", flush=True)

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_label)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_label)

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}

    # Plan all queries (cached)
    print(f"  planning ({len(queries)} queries)...", flush=True)
    plan_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    plans = await planner.plan_many(plan_items)

    # Absolute-anchor extraction for filter
    abs_items = []
    for q in queries:
        qid = q["query_id"]
        ref = parse_iso(q["ref_time"])
        plan = plans.get(qid)
        if plan and plan.absolute_anchor:
            abs_items.append((f"{qid}__abs", plan.absolute_anchor, ref))
    abs_ext = (
        await run_v2_extract(abs_items, f"{name}-abs", f"{cache_label}-abs")
        if abs_items
        else {}
    )

    # Doc memory & lattice for T_lblend
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

    lat_db = ROOT / "cache" / "t_ablation" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    import time as _time

    lat = None
    last_exc = None
    for _attempt in range(5):
        try:
            lat = LatticeStore(str(lat_db))
            break
        except Exception as _e:
            last_exc = _e
            print(
                f"  LatticeStore init attempt {_attempt + 1} failed: {_e}", flush=True
            )
            try:
                if lat_db.exists():
                    lat_db.unlink()
            except Exception:
                pass
            _time.sleep(0.5)
    if lat is None:
        raise last_exc
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)

    # Doc intervals flat
    doc_ivs_flat = {}
    for did, tes in doc_ext.items():
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        doc_ivs_flat[did] = ivs
    for d in docs:
        doc_ivs_flat.setdefault(d["doc_id"], [])

    # Embeddings
    print("  embedding...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # T_lblend
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

    # Rerank candidate sets (per variant)
    print("  reranking...", flush=True)
    # For A/B: union(sem top-50, T top-50)
    # For C:   sem top-50
    # For D:   union(sem top-50, T top-100)
    per_q_r_AB = {}
    per_q_r_C = {}
    per_q_r_D = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top50 = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        t_top100 = topk_from_scores(per_q_t[qid], 100)
        union_AB = list(dict.fromkeys(s_top + t_top50))[: int(RERANK_TOP_K * 1.5)]
        union_C = s_top
        union_D = list(dict.fromkeys(s_top + t_top100))[: int(RERANK_TOP_K * 2.0)]
        rs_AB = await rerank_topk(
            reranker, q_text[qid], union_AB, doc_text, len(union_AB)
        )
        rs_C = await rerank_topk(reranker, q_text[qid], union_C, doc_text, len(union_C))
        rs_D = await rerank_topk(reranker, q_text[qid], union_D, doc_text, len(union_D))
        all_dids = [d["doc_id"] for d in docs]
        per_q_r_AB[qid] = (rs_AB, normalize_rerank_full(rs_AB, all_dids, 0.0))
        per_q_r_C[qid] = (rs_C, normalize_rerank_full(rs_C, all_dids, 0.0))
        per_q_r_D[qid] = (rs_D, normalize_rerank_full(rs_D, all_dids, 0.0))

    # Per-query eval
    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        plan = plans.get(qid) or QueryPlan()
        abs_active = bool(plan.absolute_anchor)
        oe_active = plan.open_ended is not None
        suppress_abs = abs_active and oe_active
        abs_te = (
            abs_ext.get(f"{qid}__abs", []) if abs_active and not suppress_abs else []
        )
        abs_ivs = []
        for te in abs_te:
            abs_ivs.extend(flatten_intervals(te))
        abs_overlap = absolute_overlap_scores(doc_ivs_flat, abs_ivs)
        filter_active = abs_active and (not suppress_abs) and bool(abs_ivs)

        s_scores = per_q_s[qid]
        rerank_AB_partial, rerank_AB_full = per_q_r_AB[qid]
        rerank_C_partial, rerank_C_full = per_q_r_C[qid]
        rerank_D_partial, rerank_D_full = per_q_r_D[qid]
        t_scores = per_q_t[qid]

        # ---- A: filter_only -- rerank as final score, filter as 0/1 mult
        scores_A = {}
        for did in doc_ref_us:
            v = rerank_AB_full.get(did, 0.0)
            if filter_active:
                v = v * (
                    1.0 if abs_overlap.get(did, 0.0) >= 0.5 else ABSOLUTE_OUTSIDE_FACTOR
                )
            scores_A[did] = v
        rank_A = rank_from_scores(scores_A)
        rank_A = rank_A + [
            d for d in rank_from_scores(s_scores) if d not in set(rank_A)
        ]

        # ---- B: filter_T (current) -- score_blend({T, R}, CV gate) * filter
        fused_TR_B = fuse_TR(t_scores, rerank_AB_full, W_T_FUSE_TR)
        base_B = normalize_dict(fused_TR_B)
        scores_B = {}
        for did in doc_ref_us:
            v = base_B.get(did, 0.0)
            if filter_active:
                v = v * (
                    1.0 if abs_overlap.get(did, 0.0) >= 0.5 else ABSOLUTE_OUTSIDE_FACTOR
                )
            scores_B[did] = v
        rank_B = rank_from_scores(scores_B)
        rank_B = rank_B + [
            d for d in rank_from_scores(s_scores) if d not in set(rank_B)
        ]

        # ---- C: filter_no_T_retrieval -- sem-only union, rerank as final, filter mult
        scores_C = {}
        for did in doc_ref_us:
            v = rerank_C_full.get(did, 0.0)
            if filter_active:
                v = v * (
                    1.0 if abs_overlap.get(did, 0.0) >= 0.5 else ABSOLUTE_OUTSIDE_FACTOR
                )
            scores_C[did] = v
        rank_C = rank_from_scores(scores_C)
        rank_C = rank_C + [
            d for d in rank_from_scores(s_scores) if d not in set(rank_C)
        ]

        # ---- D: T_retrieval_wider -- T top-100, score_blend, filter
        fused_TR_D = fuse_TR(t_scores, rerank_D_full, W_T_FUSE_TR)
        base_D = normalize_dict(fused_TR_D)
        scores_D = {}
        for did in doc_ref_us:
            v = base_D.get(did, 0.0)
            if filter_active:
                v = v * (
                    1.0 if abs_overlap.get(did, 0.0) >= 0.5 else ABSOLUTE_OUTSIDE_FACTOR
                )
            scores_D[did] = v
        rank_D = rank_from_scores(scores_D)
        rank_D = rank_D + [
            d for d in rank_from_scores(s_scores) if d not in set(rank_D)
        ]

        results.append(
            {
                "qid": qid,
                "gold": list(gold_set),
                "filter_active": filter_active,
                "A_filter_only": hit_rank(rank_A, gold_set),
                "B_filter_T": hit_rank(rank_B, gold_set),
                "C_no_T_retrieval": hit_rank(rank_C, gold_set),
                "D_T_wider": hit_rank(rank_D, gold_set),
            }
        )

    return results


def aggregate(results, variants):
    n = len(results)
    out = {"n": n}
    for v in variants:
        ranks = [r[v] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        out[v] = {
            "R@1": r1 / n if n else 0.0,
            "R@5": r5 / n if n else 0.0,
            "r1_count": r1,
            "r5_count": r5,
        }
    return out


def write_md(report, path):
    benches = report["benches"]
    variants = ["A_filter_only", "B_filter_T", "C_no_T_retrieval", "D_T_wider"]
    valid = [k for k, v in benches.items() if "error" not in v and v.get("n", 0) > 0]
    macro = {
        v: sum(benches[k]["overall"][v]["R@1"] for k in valid) / max(1, len(valid))
        for v in variants
    }

    lines = []
    lines.append("# T_lblend ablation — does T earn its place?\n")
    lines.append(
        "Variants:\n"
        "- **A. filter_only**: union(sem-top50, T-top50) -> rerank; final = rerank * filter. T retrieved, not scored.\n"
        "- **B. filter_T (current)**: same retrieval; final = score_blend({T,R}, CV) * filter.\n"
        "- **C. filter_no_T_retrieval**: sem-top50 only -> rerank; final = rerank * filter. T not retrieved.\n"
        "- **D. T_retrieval_wider**: union(sem-top50, T-top100) -> rerank; final = score_blend({T,R}, CV) * filter.\n"
        "\nFilter = absolute_anchor 0/1 window from LLM plan (decay/recency/causal removed).\n"
    )

    # ---- LEAD: per-question answers ----
    lines.append("## Per-question answers (lead)\n")
    dAB = macro["B_filter_T"] - macro["A_filter_only"]
    dCA = macro["A_filter_only"] - macro["C_no_T_retrieval"]
    dBD = macro["D_T_wider"] - macro["B_filter_T"]

    if abs(dAB) <= 0.005:
        ans1 = (
            f"**No.** A ({macro['A_filter_only']:.3f}) ≈ B ({macro['B_filter_T']:.3f}), "
            f"Δ={dAB:+.3f}. T_lblend's scoring contribution is ~zero — drop the T channel from score_blend."
        )
    elif dAB > 0.005:
        ans1 = (
            f"**Yes.** B ({macro['B_filter_T']:.3f}) beats A ({macro['A_filter_only']:.3f}) by "
            f"Δ={dAB:+.3f}. T scoring earns its place."
        )
    else:
        ans1 = (
            f"**No — actively hurts.** B ({macro['B_filter_T']:.3f}) is worse than A "
            f"({macro['A_filter_only']:.3f}) by Δ={dAB:+.3f}. Drop T from score_blend."
        )

    if abs(dCA) <= 0.005:
        ans2 = (
            f"**No.** A ({macro['A_filter_only']:.3f}) ≈ C ({macro['C_no_T_retrieval']:.3f}), "
            f"Δ={dCA:+.3f}. T as a retrieval channel adds nothing — sem top-50 covers it."
        )
    elif dCA > 0.005:
        ans2 = (
            f"**Yes.** A ({macro['A_filter_only']:.3f}) beats C ({macro['C_no_T_retrieval']:.3f}) "
            f"by Δ={dCA:+.3f}. T retrieves docs sem misses; keep the T index for candidate sourcing."
        )
    else:
        ans2 = f"**No — hurts.** A is below C by Δ={dCA:+.3f}. T-retrieved noise displaces good rerank candidates."

    if abs(dBD) <= 0.005:
        ans3 = (
            f"**No.** D ({macro['D_T_wider']:.3f}) ≈ B ({macro['B_filter_T']:.3f}), "
            f"Δ={dBD:+.3f}. Existing T top-50 is sufficient; widening to 100 doesn't help."
        )
    elif dBD > 0.005:
        ans3 = (
            f"**Yes.** D ({macro['D_T_wider']:.3f}) beats B ({macro['B_filter_T']:.3f}) by "
            f"Δ={dBD:+.3f}. Widen T retrieval to top-100."
        )
    else:
        ans3 = f"**No — wider hurts.** D below B by Δ={dBD:+.3f}; extra T candidates add rerank noise."

    lines.append(f"1. **Does T_lblend earn its place as scoring? (A vs B)** — {ans1}")
    lines.append(f"2. **Does T_lblend earn its place as retrieval? (C vs A)** — {ans2}")
    lines.append(f"3. **Does wider T retrieval help? (B vs D)** — {ans3}\n")

    # ---- R@1 table ----
    lines.append("## R@1 table (per benchmark)\n")
    lines.append(
        "| Benchmark | n | A: filter_only | B: filter_T (current) | C: no_T_retrieval | D: T_wider |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k in benches:
        b = benches[k]
        if "error" in b:
            lines.append(f"| {k} | err | - | - | - | - |")
            continue
        n = b["overall"]["n"]
        a = b["overall"]["A_filter_only"]["R@1"]
        bb = b["overall"]["B_filter_T"]["R@1"]
        c = b["overall"]["C_no_T_retrieval"]["R@1"]
        d = b["overall"]["D_T_wider"]["R@1"]
        lines.append(f"| {k} | {n} | {a:.3f} | {bb:.3f} | {c:.3f} | {d:.3f} |")
    lines.append("")

    lines.append(f"## Macro R@1 across {len(valid)} benches\n")
    lines.append(f"- A (filter_only):       {macro['A_filter_only']:.3f}")
    lines.append(f"- **B (filter_T, current): {macro['B_filter_T']:.3f}**")
    lines.append(f"- C (no_T_retrieval):    {macro['C_no_T_retrieval']:.3f}")
    lines.append(f"- D (T_wider):           {macro['D_T_wider']:.3f}\n")
    lines.append("Δ vs current best (B):")
    lines.append(f"- A − B: {macro['A_filter_only'] - macro['B_filter_T']:+.3f}")
    lines.append(f"- C − B: {macro['C_no_T_retrieval'] - macro['B_filter_T']:+.3f}")
    lines.append(f"- D − B: {macro['D_T_wider'] - macro['B_filter_T']:+.3f}\n")

    # ---- Recommendation ----
    lines.append("## Recommendation\n")
    best_v, best_v_score = max(macro.items(), key=lambda x: x[1])
    if best_v == "B_filter_T":
        lines.append(
            f"**SHIP B (current filter_T).** Macro R@1 {best_v_score:.3f}. T earns its place as both retrieval and scoring channel."
        )
    elif best_v == "D_T_wider":
        lines.append(
            f"**SHIP D (T_wider).** Macro R@1 {best_v_score:.3f}, +{best_v_score - macro['B_filter_T']:.3f} over current. Widen T retrieval to top-100."
        )
    elif best_v == "A_filter_only":
        lines.append(
            f"**SHIP A (filter_only).** Macro R@1 {best_v_score:.3f}, +{best_v_score - macro['B_filter_T']:.3f} over B. "
            "Drop T from score_blend; keep T as retrieval channel only."
        )
    else:
        lines.append(
            f"**SHIP C (no_T_retrieval).** Macro R@1 {best_v_score:.3f}, "
            f"+{best_v_score - macro['B_filter_T']:.3f} over B. "
            "T is dead weight — drop the T index entirely; semantic + filter is enough."
        )
    lines.append("")

    # ---- Followup ----
    lines.append("## Followup: do we still need a temporal index?\n")
    if abs(dAB) <= 0.005 and abs(dCA) <= 0.005:
        lines.append(
            "A ≈ B AND C ≈ A: T_lblend contributes nothing in either role. "
            "**Drop the temporal index entirely.** Filter-aware semantic retrieval (sem top-50 + LLM-plan absolute filter + cross-encoder rerank) is sufficient. "
            "Saves the lattice ingestion + lattice retrieval + T-scoring code path."
        )
    elif abs(dAB) <= 0.005 and dCA > 0.005:
        lines.append(
            "A ≈ B but A > C: T matters as retrieval, not as scoring. "
            "Keep the temporal index for candidate sourcing; drop T from score_blend (use rerank as final score with the filter multiplier)."
        )
    elif dAB > 0.005 and abs(dCA) <= 0.005:
        lines.append(
            "B > A but C ≈ A: T matters as scoring, not retrieval. "
            "Keep T_lblend in score_blend; drop the T retrieval channel (sem top-50 covers candidate sourcing)."
        )
    else:
        lines.append(
            "T contributes in both roles (B > A > C). Keep both the temporal index and the T scoring channel."
        )
    lines.append("")
    return "\n".join(lines)


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

    variants = ["A_filter_only", "B_filter_T", "C_no_T_retrieval", "D_T_wider"]
    out = {"benches": {}}
    for nm, dp, qp, gp, cl in benches_def:
        try:
            results = await run_bench(nm, dp, qp, gp, cl, reranker, planner)
            overall = aggregate(results, variants)
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

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_ablation.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {json_path}", flush=True)

    md_path = out_dir / "T_ablation.md"
    md_path.write_text(write_md(out, md_path))
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
