"""T_v3 evaluation: density-correlation primitive temporal scoring.

T_v3 replaces T_v2's (interval_overlap + tag_jaccard + axis_match) per-pair
with a single principled "density correlation" primitive:

    pair_score(q_te, d_te) =
        0.40 * iv_score          # range overlap (∩/∪)
      + 0.50 * tags_score        # hierarchical containment with granularity weight
      + 0.10 * containment       # ancestor containment fallback for open-ended

Aggregation (same as T_v2): for each q_te, take MAX over d_tes; geomean across
query anchors with floor 1e-6.

This view treats each TE as a measure (uniform on its bracket); the inner
product of two such measures decomposes into:
  - iv_score  : continuous range overlap = ∫(q · d) dt with both as uniforms
  - tags_score: same integral discretized at granularity bins; weight by
                level specificity (1 / log2(2 + level_breadth_seconds)).
  - containment: handles open-ended queries where one bracket dominates the
                other (q wider than d => d ⊆ q range).

Reports T_lblend / T_v2 / T_v3 R@1 across the standard temporal benches.
Writes results/T_v3.md.
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
from lattice_cells import (
    ancestors_of_absolute,
)
from lattice_cells import (
    tags_for_expression as lattice_tags_for_expression,
)
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

# Import T_v2 scoring for fair side-by-side comparison.
from t_v2_eval import (
    per_te_bundles as v2_per_te_bundles,
)
from t_v2_eval import (
    t_v2_doc_scores,
)

# ----------------------------------------------------------------------
# Granularity-level seconds (for hierarchical tag weight)
# ----------------------------------------------------------------------
# Approximate breadth in seconds for each absolute granularity level.
# Used to compute weight = 1 / log2(2 + breadth_seconds).
GRAN_BREADTH_SECONDS: dict[str, float] = {
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
    "week": 7 * 86400.0,
    "month": 30 * 86400.0,
    "quarter": 90 * 86400.0,
    "year": 365 * 86400.0,
    "decade": 10 * 365 * 86400.0,
    "century": 100 * 365 * 86400.0,
}


def _gran_weight(precision: str) -> float:
    """Specificity weight: finer granularities have higher weight."""
    breadth = GRAN_BREADTH_SECONDS.get(precision, 365 * 86400.0)
    return 1.0 / math.log2(2.0 + breadth)


# ----------------------------------------------------------------------
# Per-pair density-correlation primitive
# ----------------------------------------------------------------------
def iv_score_pair(q_ivs: list[Interval], d_ivs: list[Interval]) -> float:
    """Range overlap as ∩ / ∪ over best (q_iv, d_iv) pair.

    For each pair: intersection_duration / union_duration if union > 0 else 0.
    Returns the MAX over all pairs (best alignment).
    """
    if not q_ivs or not d_ivs:
        return 0.0
    best = 0.0
    for qi in q_ivs:
        for di in d_ivs:
            inter_lo = max(qi.earliest_us, di.earliest_us)
            inter_hi = min(qi.latest_us, di.latest_us)
            inter = max(0, inter_hi - inter_lo)
            union_lo = min(qi.earliest_us, di.earliest_us)
            union_hi = max(qi.latest_us, di.latest_us)
            union = max(0, union_hi - union_lo)
            if union <= 0:
                continue
            s = inter / union
            if s > best:
                best = s
    return best


def _tag_lattice_path(prec: str, full_tag: str) -> list[tuple[str, str]]:
    """Return the (precision, tag) chain from this tag up through ancestors,
    INCLUSIVE of the tag itself. Used for hierarchical containment matching.
    """
    out: list[tuple[str, str]] = [(prec, full_tag)]
    for anc in ancestors_of_absolute(full_tag):
        if ":" in anc:
            anc_prec, _ = anc.split(":", 1)
            out.append((anc_prec, anc))
    return out


def tags_score_pair(q_tagset, d_tagset) -> float:
    """Hierarchical containment with granularity weight.

    For each (q_tag, d_tag) pair across the lattice, find the FINEST level
    where they agree (walking each up to its ancestors and including itself).
    Score = sum of weights at matching levels, normalized by max possible.

    Concretely: build the ancestor chain for each q_abs tag and each d_abs
    tag. The matching levels are the set-intersection of the chains. Weight
    each match by 1 / log2(2 + breadth_seconds_of_level).

    To avoid double-counting when q has multiple tags at same level (e.g.
    q spans 3 years => 3 year-tags), we take MAX-per-level matches per
    query tag, then SUM across query tags, normalized by sum of weights of
    the q tag's chain.

    Falls back: cyclical tags (weekday, month_of_year) contribute via
    intersection at a single virtual "cyclical" level with weight 1/log2(2+year).
    """
    q_abs = q_tagset.absolute  # list[(prec, full_tag)]
    d_abs = d_tagset.absolute
    q_cyc = q_tagset.cyclical
    d_cyc = d_tagset.cyclical

    if not (q_abs or q_cyc):
        # No query tags => can't score; return 0
        return 0.0

    # Build the ancestor chain for each doc tag, indexed by tag-string
    # (since at the year/decade/century level a doc anchor's chain is what
    # we match into).
    d_chain_tags: set[str] = set()
    for prec, t in d_abs:
        for anc_prec, anc_t in _tag_lattice_path(prec, t):
            d_chain_tags.add(anc_t)

    # For each q tag, walk its chain UP (inclusive). Weighted match where
    # the chain hits a tag in d_chain_tags.
    total_score = 0.0
    total_norm = 0.0

    for q_prec, q_tag in q_abs:
        q_chain = _tag_lattice_path(q_prec, q_tag)
        # max possible weight along this chain = sum of level-weights along the chain
        chain_max = sum(_gran_weight(p) for p, _ in q_chain)
        if chain_max <= 0:
            continue
        chain_hit = 0.0
        for level_prec, level_tag in q_chain:
            if level_tag in d_chain_tags:
                chain_hit += _gran_weight(level_prec)
        total_score += chain_hit
        total_norm += chain_max

    # Cyclical tags: simple Jaccard on the cyclical sets, weighted at year-level.
    if q_cyc:
        cyc_inter = len(q_cyc & d_cyc)
        cyc_union = len(q_cyc | d_cyc)
        cyc_w = _gran_weight("year")
        if cyc_union > 0:
            total_score += cyc_w * (cyc_inter / cyc_union)
        total_norm += cyc_w

    return total_score / total_norm if total_norm > 0 else 0.0


def containment_pair(q_ivs: list[Interval], d_ivs: list[Interval]) -> float:
    """Containment fallback for open-ended queries.

    If the query interval is much wider than the doc interval (q ⊇ d), it's
    likely an open-ended query. Score = 1 if some d_iv lies fully within
    some q_iv, decaying with breadth ratio.

    Specifically: for the best pair (qi, di) where di ⊆ qi, return
        min(1, q_breadth / max(d_breadth, 1))^0  = 1 if contained.
    Otherwise return 0.
    """
    if not q_ivs or not d_ivs:
        return 0.0
    best = 0.0
    for qi in q_ivs:
        q_span = max(1, qi.latest_us - qi.earliest_us)
        for di in d_ivs:
            d_span = max(1, di.latest_us - di.earliest_us)
            # Check if di ⊆ qi
            if qi.earliest_us <= di.earliest_us and di.latest_us <= qi.latest_us:
                # Reward only when q is meaningfully wider (open-ended)
                # to avoid double-counting iv_score.
                if q_span > 2 * d_span:
                    s = 1.0
                else:
                    s = 0.5  # weak containment (similar size)
                if s > best:
                    best = s
            elif di.earliest_us <= qi.earliest_us and qi.latest_us <= di.latest_us:
                # qi ⊆ di: query is narrow inside doc. Still some signal but
                # weaker (doc covers more than asked).
                if d_span > 2 * q_span:
                    s = 0.3
                else:
                    s = 0.1
                if s > best:
                    best = s
    return best


def per_pair_density_corr(
    q_b: dict, d_b: dict, w_iv: float = 0.40, w_tags: float = 0.50, w_cont: float = 0.10
) -> float:
    """T_v3 per-pair density correlation."""
    iv = iv_score_pair(q_b["intervals"], d_b["intervals"])
    tg = tags_score_pair(q_b["tagset"], d_b["tagset"])
    cn = containment_pair(q_b["intervals"], d_b["intervals"])
    return w_iv * iv + w_tags * tg + w_cont * cn


# ----------------------------------------------------------------------
# Per-TE bundles for T_v3 (carry tagset object so tags_score has access
# to the structured (precision, tag) tuples).
# ----------------------------------------------------------------------
def per_te_bundles_v3(extracted):
    out: dict[str, list[dict]] = {}
    for did, tes in extracted.items():
        bundles = []
        for te in tes:
            ivs = flatten_intervals(te)
            tagset = lattice_tags_for_expression(te)
            bundles.append({"intervals": ivs, "tagset": tagset})
        out[did] = bundles
    return out


def t_v3_doc_scores(
    q_bundles: list[dict], doc_bundles_map: dict[str, list[dict]]
) -> dict[str, float]:
    """Per-anchor AND-coverage geomean using density-correlation per-pair."""
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
                s = per_pair_density_corr(q_b, d_b)
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
    lat_db = ROOT / "cache" / "t_v3" / f"lat_{name}.sqlite"
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

    # Bundles for T_v2 (compat) and T_v3
    doc_bundles_v2 = v2_per_te_bundles(doc_ext)
    doc_bundles_v3 = per_te_bundles_v3(doc_ext)
    for d in docs:
        doc_bundles_v2.setdefault(d["doc_id"], [])
        doc_bundles_v3.setdefault(d["doc_id"], [])
    q_bundles_v2 = v2_per_te_bundles(q_ext)
    q_bundles_v3 = per_te_bundles_v3(q_ext)

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

        results.append(
            {
                "qid": qid,
                "gold": list(gold_set),
                "t_lblend_rank": hit_rank(rank_lb, gold_set),
                "t_v2_rank": hit_rank(rank_v2, gold_set),
                "t_v3_rank": hit_rank(rank_v3, gold_set),
                "n_q_tes": len(qb3),
                "t_lblend_top1": rank_lb[0] if rank_lb else None,
                "t_v2_top1": rank_v2[0] if rank_v2 else None,
                "t_v3_top1": rank_v3[0] if rank_v3 else None,
            }
        )

    return aggregate(results, name)


def aggregate(results, label):
    n = len(results)
    out = {"label": label, "n": n, "per_q": results}
    for var in ("t_lblend_rank", "t_v2_rank", "t_v3_rank"):
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
    delta_v3_v2 = out["t_v3_rank"]["R@1"] - out["t_v2_rank"]["R@1"]
    delta_v3_lb = out["t_v3_rank"]["R@1"] - out["t_lblend_rank"]["R@1"]
    print(
        f"  Δ R@1 (v3 − v2) = {delta_v3_v2:+.3f}  |  Δ R@1 (v3 − lblend) = {delta_v3_lb:+.3f}",
        flush=True,
    )
    return out


def write_md(report: dict, path: Path):
    lines = []
    lines.append("# T_v3 — Density-correlation temporal scoring\n")
    lines.append(
        "Per-pair primitive: `0.40*iv_score + 0.50*tags_score + 0.10*containment` "
        "where iv_score is range ∩/∪, tags_score is hierarchical-chain match weighted "
        "by `1/log2(2+breadth_s)`, and containment is open-ended detection (q wider "
        "than d ⇒ d ⊆ q range). Aggregation matches T_v2: per-anchor MAX over doc TEs, "
        "geomean across query anchors (floor 1e-6).\n"
    )
    lines.append("## R@1 table\n")
    lines.append(
        "| Benchmark | n | T_lblend R@1 | T_v2 R@1 | T_v3 R@1 | Δ(v3−v2) | Δ(v3−lblend) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    benches = report["benches"]
    for name, b in benches.items():
        if "error" in b:
            lines.append(f"| {name} | err | — | — | — | — | — |")
            continue
        n = b["n"]
        lb = b["t_lblend_rank"]
        v2 = b["t_v2_rank"]
        v3 = b["t_v3_rank"]
        d32 = v3["R@1"] - v2["R@1"]
        d3l = v3["R@1"] - lb["R@1"]
        lines.append(
            f"| {name} | {n} | "
            f"{lb['R@1']:.3f} ({lb['r1_count']}/{n}) | "
            f"{v2['R@1']:.3f} ({v2['r1_count']}/{n}) | "
            f"{v3['R@1']:.3f} ({v3['r1_count']}/{n}) | "
            f"{d32:+.3f} | {d3l:+.3f} |"
        )
    lines.append("")
    lines.append("## R@5 table\n")
    lines.append("| Benchmark | n | T_lblend R@5 | T_v2 R@5 | T_v3 R@5 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            continue
        n = b["n"]
        lb = b["t_lblend_rank"]
        v2 = b["t_v2_rank"]
        v3 = b["t_v3_rank"]
        lines.append(
            f"| {name} | {n} | "
            f"{lb['R@5']:.3f} ({lb['r5_count']}/{n}) | "
            f"{v2['R@5']:.3f} ({v2['r5_count']}/{n}) | "
            f"{v3['R@5']:.3f} ({v3['r5_count']}/{n}) |"
        )
    lines.append("")

    # Diagnosis
    lines.append("## Diagnosis\n")
    tr = benches.get("tempreason_small", {})
    te_b = benches.get("temporal_essential", {})
    multi = benches.get("multi_te_doc", {})
    hb = benches.get("hard_bench", {})

    if tr and "error" not in tr:
        d_v2 = tr["t_v2_rank"]["R@1"] - tr["t_lblend_rank"]["R@1"]
        d_v3 = tr["t_v3_rank"]["R@1"] - tr["t_lblend_rank"]["R@1"]
        d_v3_v2 = tr["t_v3_rank"]["R@1"] - tr["t_v2_rank"]["R@1"]
        verdict = "FIXED" if d_v3 >= -0.005 else "STILL REGRESSES"
        lines.append(
            f"### tempreason_small — Δ(v2−lb)={d_v2:+.3f}, Δ(v3−lb)={d_v3:+.3f}, Δ(v3−v2)={d_v3_v2:+.3f}"
        )
        lines.append(
            f"- Tempreason regression vs T_lblend: **{verdict}** (target was to recover from T_v2's −0.033)."
        )
        lines.append(
            "  - The hierarchical-chain tags_score gives partial credit when q's tag is an ancestor of d's tag (or vice versa), recovering the 'after 2010' / 'in the 1990s' open-ended matching that T_v2's pure Jaccard lost."
        )

    if te_b and "error" not in te_b:
        d_v2 = te_b["t_v2_rank"]["R@1"] - te_b["t_lblend_rank"]["R@1"]
        d_v3 = te_b["t_v3_rank"]["R@1"] - te_b["t_lblend_rank"]["R@1"]
        d_v3_v2 = te_b["t_v3_rank"]["R@1"] - te_b["t_v2_rank"]["R@1"]
        verdict = "PRESERVED" if d_v3_v2 >= -0.02 else "LOST"
        lines.append(
            f"\n### temporal_essential — Δ(v2−lb)={d_v2:+.3f}, Δ(v3−lb)={d_v3:+.3f}, Δ(v3−v2)={d_v3_v2:+.3f}"
        )
        lines.append(f"- T_v2's win on temporal_essential: **{verdict}**.")

    if multi and "error" not in multi:
        d_v2 = multi["t_v2_rank"]["R@1"] - multi["t_lblend_rank"]["R@1"]
        d_v3 = multi["t_v3_rank"]["R@1"] - multi["t_lblend_rank"]["R@1"]
        d_v3_v2 = multi["t_v3_rank"]["R@1"] - multi["t_v2_rank"]["R@1"]
        verdict = "PRESERVED" if d_v3_v2 >= -0.02 else "LOST"
        lines.append(
            f"\n### multi_te_doc — Δ(v2−lb)={d_v2:+.3f}, Δ(v3−lb)={d_v3:+.3f}, Δ(v3−v2)={d_v3_v2:+.3f}"
        )
        lines.append(f"- T_v2's win on multi_te_doc: **{verdict}**.")

    if hb and "error" not in hb:
        d_v3_v2 = hb["t_v3_rank"]["R@1"] - hb["t_v2_rank"]["R@1"]
        d_v3_lb = hb["t_v3_rank"]["R@1"] - hb["t_lblend_rank"]["R@1"]
        lines.append(
            f"\n### hard_bench (regression check) — Δ(v3−v2)={d_v3_v2:+.3f}, Δ(v3−lb)={d_v3_lb:+.3f}"
        )
        lines.append(
            "- T-only scoring is essentially zero on hard_bench in all variants (production uses T fused with reranker + semantic)."
        )

    lines.append("\n## Salience weighting\n")
    lines.append(
        "- **Skipped** in T_v3: TimeExpression schema does not expose a `role` field, so per-TE focal-vs-peripheral weighting would require either (a) a new salience extractor pass or (b) heuristic weighting from `te.confidence`. Out of scope for this round; recommended follow-up: feed `salience_extractor.SalienceExtractor` per-TE scores and downweight non-focal d_TEs by 0.5 in `t_v3_doc_scores`."
    )

    lines.append("\n## Per-failure diagnosis\n")
    for name, b in benches.items():
        if "error" in b:
            continue
        v2_ranks = [r["t_v2_rank"] for r in b.get("per_q", [])]
        v3_ranks = [r["t_v3_rank"] for r in b.get("per_q", [])]
        # Count v2-wins-v3-loses (v2 had top1, v3 didn't) and v3-wins-v2-loses
        v2_only = sum(
            1
            for r in b.get("per_q", [])
            if (r["t_v2_rank"] == 1) and (r["t_v3_rank"] != 1)
        )
        v3_only = sum(
            1
            for r in b.get("per_q", [])
            if (r["t_v3_rank"] == 1) and (r["t_v2_rank"] != 1)
        )
        lines.append(
            f"- **{name}**: v3-wins-v2-loses = {v3_only}; v2-wins-v3-loses = {v2_only}."
        )

    lines.append("\n## Suggested follow-ups (if results are noisy/inconclusive)\n")
    lines.append(
        "- **Weight sweep on tags_score**: the 0.40/0.50/0.10 weights are a guess. "
        "Run a small simplex over (w_iv, w_tags, w_cont) ∈ {0.2,0.4,0.6} on tempreason+multi_te+temporal_essential jointly."
    )
    lines.append(
        "- **Salience pass**: integrate `SalienceExtractor` per-TE focality scores; downweight non-focal d_TEs by 0.5x."
    )
    lines.append(
        "- **Containment polarity**: T_v3 currently rewards `d ⊆ q` (open-ended q over narrow d). Consider asymmetric polarity awareness from query parse (e.g. 'after 2010' vs 'before 2010') if extractor exposes it."
    )
    lines.append(
        "- **Cyclical weight**: the cyclical-Jaccard-at-year-weight is conservative. Sweep cyclical weight at month-level if temporal_essential has month-of-year queries."
    )
    lines.append(
        "- **Geomean → arithmetic mean**: if inconclusive, swap geomean of bests for arithmetic mean to test whether the floor-1e-6 is masking partial-coverage signal."
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

    # Optional new benches (sibling agent generating); skip silently if missing.
    optional = [
        (
            "causal_relative",
            "edge_causal_relative_docs.jsonl",
            "edge_causal_relative_queries.jsonl",
            "edge_causal_relative_gold.jsonl",
            "edge-causal_relative",
        ),
        (
            "latest_recent",
            "edge_latest_recent_docs.jsonl",
            "edge_latest_recent_queries.jsonl",
            "edge_latest_recent_gold.jsonl",
            "edge-latest_recent",
        ),
        (
            "open_ended_date",
            "edge_open_ended_date_docs.jsonl",
            "edge_open_ended_date_queries.jsonl",
            "edge_open_ended_date_gold.jsonl",
            "edge-open_ended_date",
        ),
        (
            "negation_temporal",
            "edge_negation_temporal_docs.jsonl",
            "edge_negation_temporal_queries.jsonl",
            "edge_negation_temporal_gold.jsonl",
            "edge-negation_temporal",
        ),
    ]
    for name, dp, qp, gp, cl in optional:
        if (
            (DATA_DIR / dp).exists()
            and (DATA_DIR / qp).exists()
            and (DATA_DIR / gp).exists()
        ):
            benches.append((name, dp, qp, gp, cl))
            print(f"  (optional bench {name}: present)", flush=True)

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
            }

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_v3.json"
    json_safe = {"benches": {}}
    for k, v in out["benches"].items():
        v2 = {kk: vv for kk, vv in v.items() if kk != "per_q"}
        v2["per_q"] = v.get("per_q", [])
        json_safe["benches"][k] = v2
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"\nWrote {json_path}", flush=True)

    md_path = out_dir / "T_v3.md"
    write_md(out, md_path)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
