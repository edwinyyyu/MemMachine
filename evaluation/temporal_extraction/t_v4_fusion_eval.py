"""T_v4 vs T_lblend in FUSION with cross-encoder reranker.

Hypothesis: T_v4's standalone 1.0-saturation problem (when distractors are
also inside the query window) is broken by an orthogonal channel — the
cross-encoder rerank score is uncorrelated with temporal containment.
score_blend({T, R}, {0.4, 0.6}) with CV-gated channel weights should let
the rerank channel break ties at 1.0, recovering ranking precision.

Stack:
  1. Compute T_lblend scores (alpha*iv + gamma*tag + delta*lattice)
  2. Compute T_v4 scores (asymmetric containment ratio, geomean over q anchors)
  3. Compute semantic scores (text-embedding-3-small cosine)
  4. Build candidate union from S top-50 ∪ T_lblend top-50 ∪ T_v4 top-50
  5. Cross-encoder rerank over union
  6. Two fusions (per query):
        fuse_T_lblend_R = score_blend({T_lblend, R}, {0.4, 0.6})
        fuse_T_v4_R    = score_blend({T_v4,    R}, {0.4, 0.6})
     Both with top_k_per=40, dispersion_cv_ref=0.20.
  7. Tail with semantic for any docs that fall through the fusion.
  8. R@1 / R@5 / MRR per benchmark.

Also reports the rerank_only baseline for sanity (sorted on R alone, tail S).

Writes results/T_v4_fusion.md.
"""

from __future__ import annotations

import asyncio
import json
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
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from t_v4_eval import per_te_bundles_v4, t_v4_doc_scores

W_T = 0.4
W_R = 0.6


def fuse_T_R(
    t_scores: dict[str, float],
    r_scores: dict[str, float],
    s_scores_for_tail: dict[str, float],
) -> list[str]:
    """score_blend({T, R}, {0.4, 0.6}) with CV gate; tail with S."""
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


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


# ------------------------------------------------------------------
# Benchmark drivers
# ------------------------------------------------------------------
async def run_temporal_bench(
    name, docs_path, queries_path, gold_path, cache_label, reranker
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
    lat_db = ROOT / "cache" / "t_v4_fusion" / f"lat_{name}.sqlite"
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

    # T_lblend
    per_q_tlb = {
        qid: make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }

    # T_v4 bundles
    doc_bundles_v4 = per_te_bundles_v4(doc_ext)
    for d in docs:
        doc_bundles_v4.setdefault(d["doc_id"], [])
    q_bundles_v4 = per_te_bundles_v4(q_ext)
    per_q_tv4 = {
        qid: t_v4_doc_scores(q_bundles_v4.get(qid, []), doc_bundles_v4) for qid in qids
    }
    # Ensure all docs present
    for qid in qids:
        for did in doc_text:
            per_q_tv4[qid].setdefault(did, 0.0)
            per_q_tlb[qid].setdefault(did, 0.0)

    # Rerank over union of S top-50 ∪ T_lblend top-50 ∪ T_v4 top-50
    print("  reranking over s ∪ t_lblend ∪ t_v4 union...", flush=True)
    per_q_r = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        tlb_top = topk_from_scores(per_q_tlb[qid], RERANK_TOP_K)
        tv4_top = topk_from_scores(per_q_tv4[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + tlb_top + tv4_top))[
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

        # rerank_only: sorted by R, tail by S
        rs = {did: r_scores.get(did, 0.0) for did in r_scores}
        rerank_only = merge_with_tail(
            [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)],
            s_scores,
        )
        rank_lb = fuse_T_R(t_lb, r_scores, s_scores)
        rank_v4 = fuse_T_R(t_v4, r_scores, s_scores)

        results.append(
            {
                "qid": qid,
                "qtext": q.get("text", ""),
                "gold": list(gold_set),
                "rerank_only_rank": hit_rank(rerank_only, gold_set),
                "fuse_lb_rank": hit_rank(rank_lb, gold_set),
                "fuse_v4_rank": hit_rank(rank_v4, gold_set),
                "rerank_only_top1": rerank_only[0] if rerank_only else None,
                "fuse_lb_top1": rank_lb[0] if rank_lb else None,
                "fuse_v4_top1": rank_v4[0] if rank_v4 else None,
            }
        )
    return aggregate(results, name)


async def run_lme_bench(label, types, reranker):
    data = json.load(
        open(ROOT.parent / "associative_recall" / "data" / "longmemeval_s_50q.json")
    )
    queries = [q for q in data if q["question_type"] in types]
    print(f"\n=== longmemeval ({label}): {len(queries)} queries ===", flush=True)
    results = []
    skipped = 0
    for q in queries:
        qid = q["question_id"]
        q_text = q["question"]
        q_date = q["question_date"].split(" ")[0].replace("/", "-")
        gold_ids = q["answer_session_ids"]
        gold_set = set(gold_ids)
        sessions_dict = {
            sid: " ".join(t.get("content", "") for t in sess)[:24000]
            for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])
        }
        session_dates = {
            sid: d.split(" ")[0].replace("/", "-")
            for sid, d in zip(q["haystack_session_ids"], q["haystack_dates"])
        }
        doc_ids = list(sessions_dict.keys())
        doc_text = sessions_dict

        doc_items = [
            (did, doc_text[did], parse_iso(session_dates.get(did, q_date)))
            for did in doc_ids
        ]
        q_items = [(qid, q_text, parse_iso(q_date))]
        cache_label = f"lme-q-{qid}"
        try:
            doc_ext = await run_v2_extract(
                doc_items, cache_label + "-docs", cache_label
            )
            q_ext = await run_v2_extract(q_items, cache_label + "-queries", cache_label)
            doc_mem = build_memory(doc_ext)
            q_mem = build_memory(q_ext)
        except (OverflowError, ValueError) as e:
            skipped += 1
            print(f"  skipping {qid}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        for did in doc_ids:
            doc_mem.setdefault(
                did,
                {
                    "intervals": [],
                    "axes_merged": {
                        a: AxisDistribution(axis=a, values={}, informative=False)
                        for a in AXES
                    },
                    "multi_tags": set(),
                },
            )

        doc_embs_arr = await embed_all([doc_text[did] for did in doc_ids])
        q_embs_arr = await embed_all([q_text])
        doc_embs = {did: doc_embs_arr[i] for i, did in enumerate(doc_ids)}
        q_embs = {qid: q_embs_arr[0]}
        s_scores = rank_semantic(qid, q_embs, doc_embs)

        lat_db = ROOT / "cache" / "t_v4_fusion_lme" / f"lat_{qid}.sqlite"
        lat_db.parent.mkdir(parents=True, exist_ok=True)
        if lat_db.exists():
            lat_db.unlink()
        lat = LatticeStore(str(lat_db))
        for did, tes in doc_ext.items():
            for te in tes:
                ts = lattice_tags_for_expression(te)
                lat.insert(did, ts.absolute, ts.cyclical)
        l_scores, _ = lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)

        t_lb = make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            l_scores,
        )
        doc_bundles_v4 = per_te_bundles_v4(doc_ext)
        for did in doc_ids:
            doc_bundles_v4.setdefault(did, [])
        q_bundles_v4 = per_te_bundles_v4(q_ext)
        t_v4 = t_v4_doc_scores(q_bundles_v4.get(qid, []), doc_bundles_v4)
        for did in doc_ids:
            t_lb.setdefault(did, 0.0)
            t_v4.setdefault(did, 0.0)

        s_top = topk_from_scores(s_scores, RERANK_TOP_K)
        tlb_top = topk_from_scores(t_lb, RERANK_TOP_K)
        tv4_top = topk_from_scores(t_v4, RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + tlb_top + tv4_top))[
            : int(RERANK_TOP_K * 1.5)
        ]
        r_scores = await rerank_topk(reranker, q_text, union, doc_text, len(union))

        rs = {did: r_scores.get(did, 0.0) for did in r_scores}
        rerank_only = merge_with_tail(
            [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)],
            s_scores,
        )
        rank_lb = fuse_T_R(t_lb, r_scores, s_scores)
        rank_v4 = fuse_T_R(t_v4, r_scores, s_scores)

        results.append(
            {
                "qid": qid,
                "qtext": q_text,
                "gold": list(gold_set),
                "rerank_only_rank": hit_rank(rerank_only, gold_set),
                "fuse_lb_rank": hit_rank(rank_lb, gold_set),
                "fuse_v4_rank": hit_rank(rank_v4, gold_set),
                "rerank_only_top1": rerank_only[0] if rerank_only else None,
                "fuse_lb_top1": rank_lb[0] if rank_lb else None,
                "fuse_v4_top1": rank_v4[0] if rank_v4 else None,
            }
        )
    if skipped:
        print(f"  ({skipped} queries skipped due to extraction errors)", flush=True)
    return aggregate(results, f"longmemeval ({label})")


def aggregate(results, label):
    n = len(results)
    out = {"label": label, "n": n, "per_q": results}
    for var in ("rerank_only_rank", "fuse_lb_rank", "fuse_v4_rank"):
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
        f"  rerank_only      R@1={out['rerank_only_rank']['r1_count']:3}/{n} "
        f"({out['rerank_only_rank']['R@1']:.3f})  R@5={out['rerank_only_rank']['r5_count']:3}/{n} "
        f"({out['rerank_only_rank']['R@5']:.3f})  MRR={out['rerank_only_rank']['MRR']:.3f}",
        flush=True,
    )
    print(
        f"  fuse_T_lblend_R  R@1={out['fuse_lb_rank']['r1_count']:3}/{n} "
        f"({out['fuse_lb_rank']['R@1']:.3f})  R@5={out['fuse_lb_rank']['r5_count']:3}/{n} "
        f"({out['fuse_lb_rank']['R@5']:.3f})  MRR={out['fuse_lb_rank']['MRR']:.3f}",
        flush=True,
    )
    print(
        f"  fuse_T_v4_R      R@1={out['fuse_v4_rank']['r1_count']:3}/{n} "
        f"({out['fuse_v4_rank']['R@1']:.3f})  R@5={out['fuse_v4_rank']['r5_count']:3}/{n} "
        f"({out['fuse_v4_rank']['R@5']:.3f})  MRR={out['fuse_v4_rank']['MRR']:.3f}",
        flush=True,
    )
    delta = out["fuse_v4_rank"]["R@1"] - out["fuse_lb_rank"]["R@1"]
    print(f"  Δ R@1 (v4 − lblend) = {delta:+.3f}", flush=True)
    return out


def write_md(report: dict, path: Path):
    benches = report["benches"]
    lines = []
    lines.append("# T_v4 vs T_lblend in fusion with cross-encoder rerank\n")
    lines.append(
        "Fusion: `score_blend({T, R}, {0.4, 0.6}, top_k_per=40, dispersion_cv_ref=0.20)`. "
        "Tail uses semantic. Rerank candidate pool: union(S top-50, T_lblend top-50, T_v4 top-50), "
        "capped at 75. `T_v4` = asymmetric containment ratio `|q∩d|/|d|` per (q_iv, d_iv) pair, "
        "MAX over doc TEs per query anchor, geomean across query anchors. Both fusions share the "
        "same R-channel (so any delta is purely from the T-channel).\n"
    )
    lines.append("## R@1 — Δ(v4 − lblend) leads\n")
    lines.append(
        "| Benchmark | n | rerank_only | fuse_T_lblend_R | fuse_T_v4_R | Δ(v4 − lblend) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    bench_keys = list(benches.keys())
    for name in bench_keys:
        b = benches[name]
        if "error" in b:
            lines.append(f"| {name} | err | — | — | — | — |")
            continue
        n = b["n"]
        ro = b["rerank_only_rank"]
        lb = b["fuse_lb_rank"]
        v4 = b["fuse_v4_rank"]
        d = v4["R@1"] - lb["R@1"]
        flag = ""
        if d > 0.005:
            flag = " (v4 better)"
        elif d < -0.005:
            flag = " (lblend better)"
        lines.append(
            f"| {name} | {n} | "
            f"{ro['R@1']:.3f} ({ro['r1_count']}/{n}) | "
            f"{lb['R@1']:.3f} ({lb['r1_count']}/{n}) | "
            f"{v4['R@1']:.3f} ({v4['r1_count']}/{n}) | "
            f"**{d:+.3f}**{flag} |"
        )
    lines.append("")

    lines.append("## R@5\n")
    lines.append(
        "| Benchmark | n | rerank_only R@5 | fuse_T_lblend_R R@5 | fuse_T_v4_R R@5 |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for name in bench_keys:
        b = benches[name]
        if "error" in b:
            continue
        n = b["n"]
        ro = b["rerank_only_rank"]
        lb = b["fuse_lb_rank"]
        v4 = b["fuse_v4_rank"]
        lines.append(
            f"| {name} | {n} | "
            f"{ro['R@5']:.3f} ({ro['r5_count']}/{n}) | "
            f"{lb['R@5']:.3f} ({lb['r5_count']}/{n}) | "
            f"{v4['R@5']:.3f} ({v4['r5_count']}/{n}) |"
        )
    lines.append("")

    # Macro averages
    valid = [k for k, v in benches.items() if "error" not in v and v["n"] > 0]
    macro_ro = sum(benches[k]["rerank_only_rank"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    macro_lb = sum(benches[k]["fuse_lb_rank"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    macro_v4 = sum(benches[k]["fuse_v4_rank"]["R@1"] for k in valid) / max(
        1, len(valid)
    )
    lines.append(f"## Macro-average R@1 across {len(valid)} benches\n")
    lines.append(f"- rerank_only:      {macro_ro:.3f}")
    lines.append(f"- fuse_T_lblend_R:  {macro_lb:.3f}")
    lines.append(f"- fuse_T_v4_R:      {macro_v4:.3f}")
    lines.append(f"- Δ(v4 − lblend):   **{macro_v4 - macro_lb:+.3f}**")
    lines.append("")

    # Verdict
    lines.append("## Verdict\n")
    delta_macro = macro_v4 - macro_lb
    if delta_macro >= 0.005:
        lines.append(
            f"**T_v4 ≥ T_lblend in fusion (Δ = {delta_macro:+.3f}).** "
            f"Recommend T_v4 as the production temporal primitive: "
            f"the rerank channel breaks the standalone 1.0-saturation problem on inside-window "
            f"distractors."
        )
    elif delta_macro <= -0.005:
        lines.append(
            f"**T_v4 < T_lblend in fusion (Δ = {delta_macro:+.3f}).** "
            f"Fusion did not rescue the saturation problem. Stick with T_lblend."
        )
    else:
        lines.append(
            f"**T_v4 ≈ T_lblend in fusion (Δ = {delta_macro:+.3f}).** "
            f"Roughly neutral; T_v4's simplicity (single primitive, no axis/tag/lattice) is a tie-break "
            f"argument, but the macro-R@1 difference is below noise."
        )
    lines.append("")

    # Per-bench saturation analysis: did fusion fix the standalone losses?
    standalone_losses = {
        "conjunctive_temporal": -0.250,
        "multi_te_doc": -0.083,
        "temporal_essential": -0.040,
    }
    standalone_gains = {
        "open_ended_date": 0.067,
        "tempreason_small": 0.017,
        "era_refs": 0.083,
        "hard_bench": 0.027,
    }
    lines.append("## Saturation rescue: did fusion fix T_v4's standalone losses?\n")
    lines.append(
        "| Benchmark | T_v4 standalone vs T_lblend | T_v4 fusion vs T_lblend | Rescued? |"
    )
    lines.append("|---|---:|---:|---|")
    for name, prior in {**standalone_losses, **standalone_gains}.items():
        b = benches.get(name, {})
        if "error" in b or "fuse_v4_rank" not in b:
            continue
        d_fused = b["fuse_v4_rank"]["R@1"] - b["fuse_lb_rank"]["R@1"]
        if prior < 0:
            verdict = "RESCUED" if d_fused >= -0.005 else f"still −{abs(d_fused):.3f}"
        else:
            verdict = (
                "PRESERVED" if d_fused >= -0.005 else f"GAIN LOST (−{abs(d_fused):.3f})"
            )
        lines.append(f"| {name} | {prior:+.3f} | {d_fused:+.3f} | {verdict} |")
    lines.append("")

    # Per-bench conclusion
    lines.append("## Per-benchmark conclusion\n")
    lines.append("| Benchmark | Better T-channel | Margin |")
    lines.append("|---|---|---:|")
    for name in bench_keys:
        b = benches[name]
        if "error" in b:
            continue
        d = b["fuse_v4_rank"]["R@1"] - b["fuse_lb_rank"]["R@1"]
        if d > 0.005:
            choice = "T_v4"
        elif d < -0.005:
            choice = "T_lblend"
        else:
            choice = "tie"
        lines.append(f"| {name} | {choice} | {d:+.3f} |")
    lines.append("")

    # Per-failure swap analysis on losses
    lines.append("## Per-bench v4 vs lblend swap counts (in fusion)\n")
    lines.append(
        "| Benchmark | v4-only top1 (gain) | lblend-only top1 (loss) | both | neither |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for name in bench_keys:
        b = benches[name]
        if "error" in b:
            continue
        per_q = b.get("per_q", [])
        v4_only = sum(
            1 for r in per_q if (r["fuse_v4_rank"] == 1) and (r["fuse_lb_rank"] != 1)
        )
        lb_only = sum(
            1 for r in per_q if (r["fuse_lb_rank"] == 1) and (r["fuse_v4_rank"] != 1)
        )
        both = sum(
            1 for r in per_q if (r["fuse_v4_rank"] == 1) and (r["fuse_lb_rank"] == 1)
        )
        neither = sum(
            1 for r in per_q if (r["fuse_v4_rank"] != 1) and (r["fuse_lb_rank"] != 1)
        )
        lines.append(f"| {name} | {v4_only} | {lb_only} | {both} | {neither} |")
    lines.append("")

    path.write_text("\n".join(lines))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
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
            agg = await run_temporal_bench(name, dp, qp, gp, cl, reranker)
            out["benches"][name] = agg
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][name] = {"error": str(e), "n": 0}

    # Optional LME runs
    try:
        NON_TEMP_TYPES = {
            "single-session-preference",
            "single-session-user",
            "single-session-assistant",
            "knowledge-update",
            "multi-session",
        }
        TEMP_TYPES = {"temporal-reasoning"}
        out["benches"]["lme_nontemp"] = await run_lme_bench(
            "non-temp", NON_TEMP_TYPES, reranker
        )
    except Exception as e:
        print(f"[lme_nontemp] failed: {e}", flush=True)
        out["benches"]["lme_nontemp"] = {"error": str(e), "n": 0}
    try:
        out["benches"]["lme_temp"] = await run_lme_bench("temp", TEMP_TYPES, reranker)
    except Exception as e:
        print(f"[lme_temp] failed: {e}", flush=True)
        out["benches"]["lme_temp"] = {"error": str(e), "n": 0}

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_v4_fusion.json"
    json_safe = {"benches": {}}
    for k, v in out["benches"].items():
        v2 = {kk: vv for kk, vv in v.items() if kk != "per_q"}
        v2["per_q"] = v.get("per_q", [])
        json_safe["benches"][k] = v2
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"\nWrote {json_path}", flush=True)

    md_path = out_dir / "T_v4_fusion.md"
    write_md(out, md_path)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
