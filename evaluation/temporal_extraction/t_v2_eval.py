"""T_v2 evaluation: per-anchor AND-coverage temporal scoring.

T_v2 = geomean over q_TEs of max over d_TEs of per-pair scalar:
    per_pair(q_te, d_te) = 0.40 * interval_overlap
                         + 0.40 * lattice_match
                         + 0.20 * axis_match

Comparison vs T_lblend (current): per-doc-bag scores with
    0.20*interval_jaccard_global_max + 0.20*tag_jaccard + 0.60*lattice_score.

Reports T_lblend R@1 vs T_v2 R@1 across:
  - edge_conjunctive_temporal  (target benchmark)
  - edge_multi_te_doc          (target benchmark)
  - hard_bench                 (regression check)
  - temporal_essential         (regression check)
  - tempreason_small           (regression check; real_benchmark_small_*)

Writes results/T_v2.md with per-benchmark deltas and diagnosis.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from pathlib import Path

# Strip SOCKS/HTTP proxy env vars set by the runtime sandbox so AsyncOpenAI
# can construct httpx clients without the optional socksio dependency.
# Cache hits dominate this eval; we shouldn't need network at all.
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

from axis_distributions import axes_for_expression
from force_pick_optimizers_eval import make_t_scores
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from multi_axis_scorer import axis_score
from salience_eval import (
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    flatten_intervals,
    parse_iso,
    run_v2_extract,
)
from scorer import score_jaccard_composite

# ----------------------------------------------------------------------
# T_v2: per-TE bundles + per-anchor AND-coverage
# ----------------------------------------------------------------------


def per_te_bundles(extracted):
    """Return per-doc list of TE bundles. Each bundle is a dict:
    {"intervals": [Interval, ...], "tags": set[str], "axes": dict[axis, AxisDistribution]}
    """
    out: dict[str, list[dict]] = {}
    for did, tes in extracted.items():
        bundles = []
        for te in tes:
            ivs = flatten_intervals(te)
            tagset = lattice_tags_for_expression(te)
            tags = tagset.all_tags
            ax = axes_for_expression(te)
            bundles.append({"intervals": ivs, "tags": tags, "axes": ax})
        out[did] = bundles
    return out


def interval_overlap_pair(q_ivs, d_ivs):
    """Best Jaccard-composite overlap across all (q_iv, d_iv) pairs from
    one query TE's flattened intervals against one doc TE's intervals."""
    if not q_ivs or not d_ivs:
        return 0.0
    best = 0.0
    for qi in q_ivs:
        for di in d_ivs:
            s = score_jaccard_composite(qi, di)
            if s > best:
                best = s
    return best


def lattice_match_pair(q_tags: set[str], d_tags: set[str]) -> float:
    """Per-TE pair lattice tag Jaccard. With per-TE tags (not unioned across
    a multi-TE doc), Jaccard is well-behaved."""
    if not q_tags or not d_tags:
        return 0.0
    inter = len(q_tags & d_tags)
    union = len(q_tags | d_tags)
    return inter / union if union else 0.0


def per_pair_scalar(
    q_b: dict, d_b: dict, w_iv: float = 0.40, w_lat: float = 0.40, w_ax: float = 0.20
) -> float:
    iv = interval_overlap_pair(q_b["intervals"], d_b["intervals"])
    lat = lattice_match_pair(q_b["tags"], d_b["tags"])
    ax = axis_score(q_b["axes"], d_b["axes"])
    # axis_score returns 1.0 when no axes are informative -> neutralize that
    # so it doesn't flatten the doc score; treat that as 0 here (no signal).
    if not q_b["axes"] or not d_b["axes"]:
        ax = 0.0
    return w_iv * iv + w_lat * lat + w_ax * ax


def t_v2_doc_scores(
    q_bundles: list[dict], doc_bundles_map: dict[str, list[dict]]
) -> dict[str, float]:
    """Per-anchor AND-coverage geomean.

    For each q_te:    best_per_anchor = max over d_te of per_pair_scalar
    Doc score:        geomean over anchors of best_per_anchor (floor 1e-6).

    Edge cases:
      - 0 q_tes: return 0 for all docs.
      - 0 d_tes: every anchor's max over [] -> 0.
      - 1 q_te: geomean of one element = that element (= max).
    """
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
                s = per_pair_scalar(q_b, d_b)
                if s > best:
                    best = s
            bests.append(best)
        # geomean with floor
        log_sum = 0.0
        for b in bests:
            log_sum += math.log(max(b, 1e-6))
        out[did] = math.exp(log_sum / len(bests))
    return out


# ----------------------------------------------------------------------
# T_lblend (current shipping; reused via make_t_scores from force_pick_*)
# ----------------------------------------------------------------------
# make_t_scores expects q_mem, doc_mem (built by build_memory), l_per_doc.
# l_per_doc comes from lattice_retrieve_multi.


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


def rank_from_scores(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ----------------------------------------------------------------------
# Bench runner
# ----------------------------------------------------------------------
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

    # T_lblend: aggregated mem
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

    # Lattice retrieval (lattice score component used by T_lblend)
    lat_db = ROOT / "cache" / "t_v2" / f"lat_{name}.sqlite"
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

    # Per-TE bundles for T_v2
    doc_bundles = per_te_bundles(doc_ext)
    for d in docs:
        doc_bundles.setdefault(d["doc_id"], [])
    q_bundles = per_te_bundles(q_ext)

    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue

        # T_lblend scores
        t_lblend = make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        # also fill any docs missing from t_lblend (no signal)
        for d in docs:
            t_lblend.setdefault(d["doc_id"], 0.0)
        rank_lb = rank_from_scores(t_lblend)

        # T_v2 scores
        qb = q_bundles.get(qid, [])
        t_v2 = t_v2_doc_scores(qb, doc_bundles)
        rank_v2 = rank_from_scores(t_v2)

        results.append(
            {
                "qid": qid,
                "gold": list(gold_set),
                "t_lblend_rank": hit_rank(rank_lb, gold_set),
                "t_v2_rank": hit_rank(rank_v2, gold_set),
                "n_q_tes": len(qb),
                "t_lblend_top1": rank_lb[0] if rank_lb else None,
                "t_v2_top1": rank_v2[0] if rank_v2 else None,
            }
        )

    return aggregate(results, name)


def aggregate(results, label):
    n = len(results)
    out = {"label": label, "n": n, "per_q": results}
    for var in ("t_lblend_rank", "t_v2_rank"):
        ranks = [r[var] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
        out[var] = {
            "R@1": r1 / n if n else 0.0,
            "R@5": r5 / n if n else 0.0,
            "MRR": mrr,
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
    delta = out["t_v2_rank"]["R@1"] - out["t_lblend_rank"]["R@1"]
    print(f"  Δ R@1 (v2 − lblend) = {delta:+.3f}", flush=True)
    return out


def write_md(report: dict, path: Path):
    lines = []
    lines.append("# T_v2 — Per-anchor AND-coverage temporal scoring\n")
    lines.append(
        "Per-pair scalar: `0.40*interval_overlap + 0.40*lattice_match + 0.20*axis_match`. "
        "Aggregate per-anchor max over doc TEs, then geomean across query anchors (floor 1e-6).\n"
    )
    lines.append("## R@1 table\n")
    lines.append(
        "| Benchmark | n | T_lblend R@1 | T_v2 R@1 | Δ R@1 | T_lblend R@5 | T_v2 R@5 | Δ R@5 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    benches = report["benches"]
    for name, b in benches.items():
        n = b["n"]
        lb = b["t_lblend_rank"]
        v2 = b["t_v2_rank"]
        d1 = v2["R@1"] - lb["R@1"]
        d5 = v2["R@5"] - lb["R@5"]
        lines.append(
            f"| {name} | {n} | "
            f"{lb['R@1']:.3f} ({lb['r1_count']}/{n}) | "
            f"{v2['R@1']:.3f} ({v2['r1_count']}/{n}) | "
            f"{d1:+.3f} | "
            f"{lb['R@5']:.3f} ({lb['r5_count']}/{n}) | "
            f"{v2['R@5']:.3f} ({v2['r5_count']}/{n}) | "
            f"{d5:+.3f} |"
        )
    lines.append("")

    lines.append("## Diagnosis\n")
    conj = benches.get("conjunctive_temporal", {})
    multi = benches.get("multi_te_doc", {})
    hb = benches.get("hard_bench", {})
    te = benches.get("temporal_essential", {})
    tr = benches.get("tempreason_small", {})

    def _delta(b, key="t_v2_rank", base="t_lblend_rank"):
        if not b:
            return 0.0
        return b[key]["R@1"] - b[base]["R@1"]

    lines.append("### Conjunctive temporal (target failure 1)")
    if conj:
        d = _delta(conj)
        if d > 0:
            lines.append(
                f"- T_v2 R@1 improves by {d:+.3f} on conjunctive_temporal. "
                "Per-anchor AND-coverage rewards docs that hit BOTH query anchors "
                "instead of penalizing them via tag-union dilution."
            )
        elif d == 0:
            lines.append(
                f"- T_v2 R@1 unchanged on conjunctive_temporal (Δ={d:+.3f}). "
                "Likely both T variants tied at the floor or already the docs lined up under both."
            )
        else:
            lines.append(
                f"- T_v2 R@1 regresses by {d:+.3f} on conjunctive_temporal. "
                "Geomean floor on a single failed anchor may be over-penalizing partial-coverage gold."
            )

    lines.append("\n### Multi-TE doc (target failure 2)")
    if multi:
        d = _delta(multi)
        if d > 0:
            lines.append(
                f"- T_v2 R@1 improves by {d:+.3f} on multi_te_doc. "
                "Per-TE pair scoring eliminates tag-union dilution and global-max interval normalization, "
                "so peripheral doc TEs no longer drag focal-TE matching."
            )
        elif d == 0:
            lines.append(f"- T_v2 R@1 unchanged on multi_te_doc (Δ={d:+.3f}).")
        else:
            lines.append(f"- T_v2 R@1 regresses by {d:+.3f} on multi_te_doc.")

    lines.append("\n### Regression check")
    for label, b in [
        ("hard_bench", hb),
        ("temporal_essential", te),
        ("tempreason_small", tr),
    ]:
        if b:
            d = _delta(b)
            verdict = "HOLDS" if d >= -0.02 else "REGRESSES"
            lines.append(
                f"- {label}: T_lblend R@1 = {b['t_lblend_rank']['R@1']:.3f}, "
                f"T_v2 R@1 = {b['t_v2_rank']['R@1']:.3f}, Δ={d:+.3f} ({verdict})."
            )

    lines.append("\n## Suggested next experiments\n")
    lines.append(
        "- **Tunable hybrid**: compose `T_hybrid = β * T_v2 + (1-β) * T_lblend` and "
        "sweep β ∈ {0.25, 0.5, 0.75, 1.0}. The two T variants disagree on different "
        "shapes (per-anchor AND-coverage vs bag-of-anchors lattice prior), and a fixed "
        "β=0.5 may capture both regimes."
    )
    lines.append(
        "- **Per-pair weight tuning**: the 0.40/0.40/0.20 weights are a guess. Run a "
        "small grid (interval, lattice, axis) ∈ {0.2, 0.4, 0.6} simplex over "
        "edge_conjunctive_temporal + multi_te_doc + hard_bench jointly."
    )
    lines.append(
        "- **Geomean vs softmin**: geomean with floor 1e-6 zeros heavily on any one "
        "missed anchor. Try softmin (β·logsumexp(-β·x)/β) or just the arithmetic mean "
        "of best_per_anchor as a softer aggregator."
    )
    lines.append(
        "- **Per-query gate** (per the existing per_query_gate_eval.py pattern): "
        "let an LLM pick T_lblend vs T_v2 per query when their top-5 sets diverge; "
        "this captured the fuse_T_R distribution split previously and may capture this one too."
    )
    lines.append(
        "- **Anchor-count features**: wire `n_q_tes` as a feature into a simple gate "
        "(use T_lblend when n_q_tes==1; use T_v2 when n_q_tes≥2). Manual ablation suggests "
        "T_v2 is principally a multi-anchor improvement."
    )

    lines.append("")
    path.write_text("\n".join(lines))


async def main():
    # Reuse existing extraction caches: edge benches use "edge-{name}",
    # the rest use "v7l-{name}" (matches bisect_rrf_tune_eval / edge_eval).
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
            }

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_v2.json"
    # Strip per_q before JSON (large) but keep aggregate
    json_safe = {"benches": {}}
    for k, v in out["benches"].items():
        v2 = {kk: vv for kk, vv in v.items() if kk != "per_q"}
        v2["per_q"] = v.get("per_q", [])
        json_safe["benches"][k] = v2
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"\nWrote {json_path}", flush=True)

    md_path = out_dir / "T_v2.md"
    write_md(out, md_path)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
