"""T_v4 evaluation: unified asymmetric containment ratio temporal scoring.

T_v4 uses a SINGLE primitive that mathematically subsumes most temporal
matching cases:

    pair_score(q_te, d_te) = max over (q_iv, d_iv) of:
        |q_iv ∩ d_iv| / |d_iv|        (asymmetric containment ratio,
                                        normalized by doc duration)

By construction this handles:
  - Exact match (delta-in-delta): 1.0 if same instant.
  - Specific doc inside fuzzy/era query: 1.0 (delta entirely inside q).
  - Open-ended query (e.g. ``after 2020`` ⇒ q latest = +∞-ish): 1.0
    if d_te lies inside the open half-line.
  - Range-vs-range: fractional overlap normalized by doc duration —
    a tight doc inside a wider query gets full credit; a broad doc
    that only partially covers query gets dilute credit.
  - Granularity-mixed: handled naturally by interval bounds.

Aggregation matches T_v2/T_v3: per-anchor MAX over doc TEs, geomean
across query anchors with floor 1e-6.

Reports T_lblend / T_v2 / T_v3 / T_v4 R@1 across the standard temporal
benches. Writes results/T_v4.md.
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

from force_pick_optimizers_eval import make_t_scores
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from salience_eval import (
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    flatten_intervals,
    parse_iso,
    run_v2_extract,
)
from scorer import Interval
from t_v2_eval import (
    per_te_bundles as v2_per_te_bundles,
)
from t_v2_eval import (
    t_v2_doc_scores,
)
from t_v3_eval import (
    per_te_bundles_v3,
    t_v3_doc_scores,
)

# ----------------------------------------------------------------------
# T_v4: asymmetric containment primitive
# ----------------------------------------------------------------------
# Microsecond bounds are bounded (datetime in microseconds is typically
# within ±2.5e17 us for years 1..9999); but we still treat zero-duration
# intervals as 1us to avoid div-by-zero. Numerator (intersection) is
# clamped at zero.


def pair_score_v4(q_ivs: list[Interval], d_ivs: list[Interval]) -> float:
    """Best asymmetric containment ratio across all (q_iv, d_iv) pairs.

    Score = |q ∩ d| / |d|. So:
      - d ⊆ q  → 1.0   (doc instant or doc range fully inside query range)
      - d ⊃ q  → |q|/|d|   (doc broader than query, fractional credit)
      - partial overlap → small fraction
    """
    if not q_ivs or not d_ivs:
        return 0.0
    best = 0.0
    for qi in q_ivs:
        for di in d_ivs:
            inter_lo = max(qi.earliest_us, di.earliest_us)
            inter_hi = min(qi.latest_us, di.latest_us)
            inter = max(0, inter_hi - inter_lo)
            d_dur = di.latest_us - di.earliest_us
            if d_dur <= 0:
                d_dur = 1  # treat instant as 1us
            score = inter / d_dur
            if score > best:
                best = score
                if best >= 1.0:
                    # Cap at 1.0 (instant doc fully inside instant query
                    # could go slightly above due to integer arithmetic);
                    best = 1.0
    return best


def per_te_bundles_v4(extracted):
    """Per-doc list of TE bundles. Each bundle just needs `intervals`."""
    out: dict[str, list[dict]] = {}
    for did, tes in extracted.items():
        bundles = []
        for te in tes:
            ivs = flatten_intervals(te)
            bundles.append({"intervals": ivs})
        out[did] = bundles
    return out


def t_v4_doc_scores(
    q_bundles: list[dict], doc_bundles_map: dict[str, list[dict]]
) -> dict[str, float]:
    """Per-anchor AND-coverage geomean with asymmetric containment primitive."""
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
                s = pair_score_v4(q_b["intervals"], d_b["intervals"])
                if s > best:
                    best = s
            bests.append(best)
        log_sum = 0.0
        for b in bests:
            log_sum += math.log(max(b, 1e-6))
        out[did] = math.exp(log_sum / len(bests))
    return out


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


async def run_bench(name, docs_path, queries_path, gold_path, cache_label):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===", flush=True)
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_label)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_label)

    # T_lblend bag-merged memory
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

    # Lattice retrieval (lattice_score channel for T_lblend)
    lat_db = ROOT / "cache" / "t_v4" / f"lat_{name}.sqlite"
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

    # Bundles for v2, v3, v4
    doc_bundles_v2 = v2_per_te_bundles(doc_ext)
    doc_bundles_v3 = per_te_bundles_v3(doc_ext)
    doc_bundles_v4 = per_te_bundles_v4(doc_ext)
    for d in docs:
        doc_bundles_v2.setdefault(d["doc_id"], [])
        doc_bundles_v3.setdefault(d["doc_id"], [])
        doc_bundles_v4.setdefault(d["doc_id"], [])
    q_bundles_v2 = v2_per_te_bundles(q_ext)
    q_bundles_v3 = per_te_bundles_v3(q_ext)
    q_bundles_v4 = per_te_bundles_v4(q_ext)

    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue

        # T_lblend
        t_lblend = make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for d in docs:
            t_lblend.setdefault(d["doc_id"], 0.0)
        rank_lb = rank_from_scores(t_lblend)

        # T_v2
        qb2 = q_bundles_v2.get(qid, [])
        t_v2 = t_v2_doc_scores(qb2, doc_bundles_v2)
        rank_v2 = rank_from_scores(t_v2)

        # T_v3
        qb3 = q_bundles_v3.get(qid, [])
        t_v3 = t_v3_doc_scores(qb3, doc_bundles_v3)
        rank_v3 = rank_from_scores(t_v3)

        # T_v4
        qb4 = q_bundles_v4.get(qid, [])
        t_v4 = t_v4_doc_scores(qb4, doc_bundles_v4)
        rank_v4 = rank_from_scores(t_v4)

        results.append(
            {
                "qid": qid,
                "qtext": q.get("text", ""),
                "gold": list(gold_set),
                "t_lblend_rank": hit_rank(rank_lb, gold_set),
                "t_v2_rank": hit_rank(rank_v2, gold_set),
                "t_v3_rank": hit_rank(rank_v3, gold_set),
                "t_v4_rank": hit_rank(rank_v4, gold_set),
                "n_q_tes": len(qb4),
                "t_lblend_top1": rank_lb[0] if rank_lb else None,
                "t_v2_top1": rank_v2[0] if rank_v2 else None,
                "t_v3_top1": rank_v3[0] if rank_v3 else None,
                "t_v4_top1": rank_v4[0] if rank_v4 else None,
            }
        )

    return aggregate(results, name)


def aggregate(results, label):
    n = len(results)
    out = {"label": label, "n": n, "per_q": results}
    for var in ("t_lblend_rank", "t_v2_rank", "t_v3_rank", "t_v4_rank"):
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
        f"  T_lblend  R@1={out['t_lblend_rank']['r1_count']:3}/{n} ({out['t_lblend_rank']['R@1']:.3f})  "
        f"R@5={out['t_lblend_rank']['r5_count']:3}/{n} ({out['t_lblend_rank']['R@5']:.3f})  "
        f"MRR={out['t_lblend_rank']['MRR']:.3f}",
        flush=True,
    )
    print(
        f"  T_v2      R@1={out['t_v2_rank']['r1_count']:3}/{n} ({out['t_v2_rank']['R@1']:.3f})  "
        f"R@5={out['t_v2_rank']['r5_count']:3}/{n} ({out['t_v2_rank']['R@5']:.3f})  "
        f"MRR={out['t_v2_rank']['MRR']:.3f}",
        flush=True,
    )
    print(
        f"  T_v3      R@1={out['t_v3_rank']['r1_count']:3}/{n} ({out['t_v3_rank']['R@1']:.3f})  "
        f"R@5={out['t_v3_rank']['r5_count']:3}/{n} ({out['t_v3_rank']['R@5']:.3f})  "
        f"MRR={out['t_v3_rank']['MRR']:.3f}",
        flush=True,
    )
    print(
        f"  T_v4      R@1={out['t_v4_rank']['r1_count']:3}/{n} ({out['t_v4_rank']['R@1']:.3f})  "
        f"R@5={out['t_v4_rank']['r5_count']:3}/{n} ({out['t_v4_rank']['R@5']:.3f})  "
        f"MRR={out['t_v4_rank']['MRR']:.3f}",
        flush=True,
    )
    delta_v4_v2 = out["t_v4_rank"]["R@1"] - out["t_v2_rank"]["R@1"]
    delta_v4_lb = out["t_v4_rank"]["R@1"] - out["t_lblend_rank"]["R@1"]
    delta_v4_v3 = out["t_v4_rank"]["R@1"] - out["t_v3_rank"]["R@1"]
    print(
        f"  Δ R@1 (v4 − v2) = {delta_v4_v2:+.3f}  |  Δ R@1 (v4 − v3) = {delta_v4_v3:+.3f}  |  "
        f"Δ R@1 (v4 − lblend) = {delta_v4_lb:+.3f}",
        flush=True,
    )
    return out


def write_md(report: dict, path: Path):
    lines = []
    lines.append("# T_v4 — Asymmetric containment ratio temporal scoring\n")
    lines.append(
        "Single primitive: `|q_iv ∩ d_iv| / |d_iv|` (asymmetric, normalized by doc duration). "
        "MAX across (q_iv, d_iv) pairs per (q_te, d_te); MAX across d_tes per q_te; geomean across "
        "q anchors with floor 1e-6. No tag_jaccard, no axis, no lattice.\n"
    )
    lines.append("## R@1 table\n")
    lines.append(
        "| Benchmark | n | T_lblend R@1 | T_v2 R@1 | T_v3 R@1 | T_v4 R@1 | Δ(v4−v2) | Δ(v4−lblend) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    benches = report["benches"]
    for name, b in benches.items():
        if "error" in b:
            lines.append(f"| {name} | err | — | — | — | — | — | — |")
            continue
        n = b["n"]
        lb = b["t_lblend_rank"]
        v2 = b["t_v2_rank"]
        v3 = b["t_v3_rank"]
        v4 = b["t_v4_rank"]
        d42 = v4["R@1"] - v2["R@1"]
        d4l = v4["R@1"] - lb["R@1"]
        lines.append(
            f"| {name} | {n} | "
            f"{lb['R@1']:.3f} ({lb['r1_count']}/{n}) | "
            f"{v2['R@1']:.3f} ({v2['r1_count']}/{n}) | "
            f"{v3['R@1']:.3f} ({v3['r1_count']}/{n}) | "
            f"{v4['R@1']:.3f} ({v4['r1_count']}/{n}) | "
            f"{d42:+.3f} | {d4l:+.3f} |"
        )
    lines.append("")
    lines.append("## R@5 table\n")
    lines.append("| Benchmark | n | T_lblend R@5 | T_v2 R@5 | T_v3 R@5 | T_v4 R@5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            continue
        n = b["n"]
        lb = b["t_lblend_rank"]
        v2 = b["t_v2_rank"]
        v3 = b["t_v3_rank"]
        v4 = b["t_v4_rank"]
        lines.append(
            f"| {name} | {n} | "
            f"{lb['R@5']:.3f} ({lb['r5_count']}/{n}) | "
            f"{v2['R@5']:.3f} ({v2['r5_count']}/{n}) | "
            f"{v3['R@5']:.3f} ({v3['r5_count']}/{n}) | "
            f"{v4['R@5']:.3f} ({v4['r5_count']}/{n}) |"
        )
    lines.append("")

    # Headline
    lines.append("## Headline\n")
    tr = benches.get("tempreason_small", {})
    te_b = benches.get("temporal_essential", {})
    multi = benches.get("multi_te_doc", {})

    target_msgs = []
    if tr and "error" not in tr:
        d_v4_lb = tr["t_v4_rank"]["R@1"] - tr["t_lblend_rank"]["R@1"]
        verdict = (
            "FIXED"
            if d_v4_lb >= -0.005
            else f"NOT FIXED (still Δ={d_v4_lb:+.3f} vs T_lblend)"
        )
        target_msgs.append(
            f"- **tempreason regression**: {verdict} — T_v4 R@1 = {tr['t_v4_rank']['R@1']:.3f}, "
            f"T_lblend = {tr['t_lblend_rank']['R@1']:.3f}, T_v2 = {tr['t_v2_rank']['R@1']:.3f}."
        )
    if te_b and "error" not in te_b:
        d_v4_v2 = te_b["t_v4_rank"]["R@1"] - te_b["t_v2_rank"]["R@1"]
        verdict = (
            "PRESERVED" if d_v4_v2 >= -0.02 else f"LOST (Δ={d_v4_v2:+.3f} vs T_v2)"
        )
        target_msgs.append(
            f"- **temporal_essential win**: {verdict} — T_v4 R@1 = {te_b['t_v4_rank']['R@1']:.3f}, "
            f"T_v2 = {te_b['t_v2_rank']['R@1']:.3f}."
        )
    if multi and "error" not in multi:
        d_v4_v2 = multi["t_v4_rank"]["R@1"] - multi["t_v2_rank"]["R@1"]
        verdict = (
            "PRESERVED" if d_v4_v2 >= -0.02 else f"LOST (Δ={d_v4_v2:+.3f} vs T_v2)"
        )
        target_msgs.append(
            f"- **multi_te_doc win**: {verdict} — T_v4 R@1 = {multi['t_v4_rank']['R@1']:.3f}, "
            f"T_v2 = {multi['t_v2_rank']['R@1']:.3f}."
        )
    for m in target_msgs:
        lines.append(m)
    lines.append("")

    # Per-bench v4 vs v2 swap analysis
    lines.append("## Per-bench v4 vs v2 swap counts\n")
    lines.append(
        "| Benchmark | v4-only top1 (gain) | v2-only top1 (loss) | both | neither |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            continue
        per_q = b.get("per_q", [])
        v4_only = sum(
            1 for r in per_q if (r["t_v4_rank"] == 1) and (r["t_v2_rank"] != 1)
        )
        v2_only = sum(
            1 for r in per_q if (r["t_v2_rank"] == 1) and (r["t_v4_rank"] != 1)
        )
        both = sum(1 for r in per_q if (r["t_v2_rank"] == 1) and (r["t_v4_rank"] == 1))
        neither = sum(
            1 for r in per_q if (r["t_v2_rank"] != 1) and (r["t_v4_rank"] != 1)
        )
        lines.append(f"| {name} | {v4_only} | {v2_only} | {both} | {neither} |")
    lines.append("")

    # Per-failure detail (top losses on key benches)
    lines.append("## Per-failure diagnosis\n")
    for name in (
        "tempreason_small",
        "temporal_essential",
        "multi_te_doc",
        "conjunctive_temporal",
        "relative_time",
        "era_refs",
        "open_ended_date",
        "latest_recent",
        "negation_temporal",
        "causal_relative",
        "hard_bench",
    ):
        b = benches.get(name)
        if not b or "error" in b:
            continue
        per_q = b.get("per_q", [])
        # Show v4 losses (v4 missed but v2 hit at top-1) and v4 gains (v4 hit but v2 missed)
        losses = [r for r in per_q if (r["t_v2_rank"] == 1) and (r["t_v4_rank"] != 1)][
            :5
        ]
        gains = [r for r in per_q if (r["t_v4_rank"] == 1) and (r["t_v2_rank"] != 1)][
            :5
        ]
        if not losses and not gains:
            continue
        lines.append(f"### {name}\n")
        if losses:
            lines.append("**Losses (v4 missed, v2 hit, up to 5):**")
            for r in losses:
                lines.append(
                    f"- `{r['qid']}` (n_q_tes={r['n_q_tes']}): v4_top1=`{r['t_v4_top1']}`, "
                    f"v4_rank={r['t_v4_rank']}, gold={r['gold']}"
                )
                if r.get("qtext"):
                    lines.append(f"  - q: {r['qtext'][:120]}")
        if gains:
            lines.append("**Gains (v4 hit, v2 missed, up to 5):**")
            for r in gains:
                lines.append(
                    f"- `{r['qid']}` (n_q_tes={r['n_q_tes']}): v4_top1=`{r['t_v4_top1']}`, "
                    f"v2_top1=`{r['t_v2_top1']}`, gold={r['gold']}"
                )
                if r.get("qtext"):
                    lines.append(f"  - q: {r['qtext'][:120]}")
        lines.append("")

    # Verdict / next steps
    lines.append("## Verdict & next steps\n")
    # Macro-average across benches
    bench_keys = [k for k, v in benches.items() if "error" not in v and v["n"] > 0]
    macro_v4 = sum(benches[k]["t_v4_rank"]["R@1"] for k in bench_keys) / max(
        1, len(bench_keys)
    )
    macro_v2 = sum(benches[k]["t_v2_rank"]["R@1"] for k in bench_keys) / max(
        1, len(bench_keys)
    )
    macro_v3 = sum(benches[k]["t_v3_rank"]["R@1"] for k in bench_keys) / max(
        1, len(bench_keys)
    )
    macro_lb = sum(benches[k]["t_lblend_rank"]["R@1"] for k in bench_keys) / max(
        1, len(bench_keys)
    )
    lines.append(
        f"Macro-average R@1 across {len(bench_keys)} benches: "
        f"T_lblend={macro_lb:.3f}, T_v2={macro_v2:.3f}, T_v3={macro_v3:.3f}, **T_v4={macro_v4:.3f}**."
    )
    lines.append("")

    lines.append("### If T_v4 wins overall\n")
    lines.append(
        "- **Open-ended polarity**: the asymmetric containment is symmetric in coverage but blind "
        "to query polarity (`before X` vs `after X` produce the same q_iv). If extractor exposes "
        "polarity, restrict containment to the matching half-line."
    )
    lines.append(
        "- **Multi-anchor combination**: geomean assumes anchors are independent ANDs. Consider "
        "soft-max blending when one anchor clearly dominates (e.g. cued single-anchor essentials)."
    )
    lines.append(
        "- **Granularity penalty**: a year-precision doc inside a year-precision query gets 1.0; "
        "a day-precision doc inside the same year-precision query also gets 1.0. Consider tiebreak by "
        "specificity (smaller |d_iv| wins for ties at 1.0)."
    )
    lines.append(
        "- **Cyclical anchors**: pure interval containment doesn't see weekday/month-of-year cycles. "
        "Add a cyclical Jaccard side-channel only when q_te declares a recurrence kind."
    )

    lines.append("\n### If T_v4 loses overall\n")
    lines.append(
        "- **Asymmetric tiebreak**: many docs may achieve 1.0 (delta inside era query). Add tiebreaker "
        "by `|d_iv|` (smaller = more specific) or by exact-tag overlap."
    )
    lines.append(
        "- **Containment polarity inversion**: when `q ⊂ d` (broad doc, narrow query), v4 returns "
        "|q|/|d| which may be tiny even though the doc is semantically relevant. Consider symmetric "
        "blending: `α * (|q∩d|/|d|) + (1-α) * (|q∩d|/|q|)` with α=0.5 = Sørensen, α=0 = recall, α=1 = our v4."
    )
    lines.append(
        "- **Hybrid v4 + tag sidecar**: if v4 ties many docs at 1.0, restore `tag_jaccard` as a "
        "secondary tiebreaker only (not a primary weighted component)."
    )
    lines.append("")

    path.write_text("\n".join(lines))


async def main():
    benches = [
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
    ]

    # Optional new benches; check both `edge_<name>_*.jsonl` and `<name>_*.jsonl` patterns.
    optional = [
        ("causal_relative", "causal_relative", "edge-causal_relative"),
        ("latest_recent", "latest_recent", "edge-latest_recent"),
        ("open_ended_date", "open_ended_date", "edge-open_ended_date"),
        ("negation_temporal", "negation_temporal", "edge-negation_temporal"),
    ]
    for name, file_stem, cl in optional:
        # Try both naming patterns
        for prefix in (f"edge_{file_stem}", file_stem):
            dp = f"{prefix}_docs.jsonl"
            qp = f"{prefix}_queries.jsonl"
            gp = f"{prefix}_gold.jsonl"
            if (
                (DATA_DIR / dp).exists()
                and (DATA_DIR / qp).exists()
                and (DATA_DIR / gp).exists()
            ):
                benches.append((name, dp, qp, gp, cl))
                print(f"  (optional bench {name}: present at {prefix}_*)", flush=True)
                break

    out = {"benches": {}}
    for name, dp, qp, gp, cache_label in benches:
        try:
            agg = await run_bench(name, dp, qp, gp, cache_label)
            out["benches"][name] = agg
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][name] = {
                "error": str(e),
                "n": 0,
                "t_lblend_rank": {
                    "R@1": 0,
                    "R@5": 0,
                    "MRR": 0,
                    "r1_count": 0,
                    "r5_count": 0,
                },
                "t_v2_rank": {
                    "R@1": 0,
                    "R@5": 0,
                    "MRR": 0,
                    "r1_count": 0,
                    "r5_count": 0,
                },
                "t_v3_rank": {
                    "R@1": 0,
                    "R@5": 0,
                    "MRR": 0,
                    "r1_count": 0,
                    "r5_count": 0,
                },
                "t_v4_rank": {
                    "R@1": 0,
                    "R@5": 0,
                    "MRR": 0,
                    "r1_count": 0,
                    "r5_count": 0,
                },
            }

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_v4.json"
    json_safe = {"benches": {}}
    for k, v in out["benches"].items():
        v2 = {kk: vv for kk, vv in v.items() if kk != "per_q"}
        v2["per_q"] = v.get("per_q", [])
        json_safe["benches"][k] = v2
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"\nWrote {json_path}", flush=True)

    md_path = out_dir / "T_v4.md"
    write_md(out, md_path)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
