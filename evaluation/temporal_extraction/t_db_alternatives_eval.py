"""Database-friendly T-channel alternatives.

Compares 3 candidate T-scoring designs that fit cleanly into Postgres /
Elasticsearch (inverted-index + range-overlap), against the baseline
T_lblend (= 0.2*interval_jaccard + 0.2*tag_jaccard + 0.6*lattice_score).

Constraints assumed:
  - T-channel and S-channel live in SEPARATE indexes.
  - Each component query returns top-K per-channel.
  - Final ranking is built from union-of-top-K (no full-table scan).

Candidates (T-only score; S/R fusion not modeled here — we evaluate the
T-channel signal on its own + as input to a simple T+S RRF):

  A. HIER_TAGS  — single inverted-index query on a query-expanded tag set
                   (lattice tags + cyclical), score = sum of granularity-
                   weighted matches. Pure inverted-index.

  B. RANGE_OVERLAP_PLUS_YEAR  — range-overlap on (doc_range, query_range)
                   primary signal + year-tag inverted-index for queries
                   with no usable range (e.g. recurrences).

  C. PER_TE_MAX  — each doc-TE indexed as its OWN row (multi-TE friendly).
                   Per-doc score = MAX over its TE rows of [tag-Jaccard +
                   range-overlap]. Avoids per-doc tag dilution.

  D. RRF_TAG_RANGE  — A + B as TWO separate index queries; merge by RRF.
                   (No score addition across indexes; fully DB-native.)

Baseline for reference: T_lblend (current production).

We test on hard_bench, temporal_essential, real_benchmark_small (tempreason).
We use the cached extractions under cache/v7l-<bench>/. No LLM calls needed.

R@1 is the primary metric. We also report R@5.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")


from lattice_cells import (
    tags_for_expression as lattice_tags_for_expression,
)
from lattice_retrieval import expand_query_tags
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from rag_fusion import rrf, score_blend
from salience_eval import (  # type: ignore
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    flatten_intervals,
    interval_pair_best,
    parse_iso,
    rank_semantic,
    run_v2_extract,
    tag_score,
)
from schema import TimeExpression

# -------- Granularity weight (more specific tag => more weight) -----
TAG_WEIGHT_BY_PREC = {
    "minute": 9.0,
    "hour": 8.0,
    "day": 7.0,
    "week": 6.0,
    "month": 5.0,
    "quarter": 4.0,
    "year": 3.0,
    "decade": 2.0,
    "century": 1.0,
    # cyclical
    "weekday": 0.5,
    "month_of_year": 1.5,
    "day_of_month": 0.5,
    "hour_of_day": 0.5,
    "season": 0.7,
    "part_of_day": 0.3,
    "weekend": 0.2,
}


def _prec_of(tag: str) -> str:
    return tag.split(":", 1)[0] if ":" in tag else "year"


# ---------------------------------------------------------------------------
# Candidate A: HIER_TAGS — pure inverted-index, granularity-weighted.
# ---------------------------------------------------------------------------
def candidate_A_hier_tags(
    q_tes: list[TimeExpression],
    doc_tag_index: dict[str, set[str]],  # doc_id -> set[tag]
) -> dict[str, float]:
    """Database equivalent (Postgres):
        SELECT doc_id, sum(weight)
        FROM   doc_tags
        WHERE  tag = ANY(:expanded_tags)
        GROUP BY doc_id
        ORDER BY sum(weight) DESC
        LIMIT  :K;

    Elasticsearch: terms query against doc.tags.keyword with function_score
    weighting per-tag granularity.

    Indexable as: B-tree / GIN on tag column.
    """
    # Build query-expansion tagset (UP-walk + cyclical, no DOWN — we don't
    # over-broaden into siblings).
    q_expanded: dict[str, dict] = {}
    for te in q_tes:
        ts = lattice_tags_for_expression(te)
        # Use down_levels=0: avoid expanding into doc cells we don't want.
        # Lattice's expand_query_tags already does ancestors + cyclical at
        # down_levels=0 (only self + ancestors + cyclical).
        exp = expand_query_tags(ts, down_levels=0)
        q_expanded.update(exp)
    if not q_expanded:
        return {}
    # Score each doc by sum of weighted tag matches.
    out: dict[str, float] = {}
    for did, doc_tags in doc_tag_index.items():
        s = 0.0
        for tag in doc_tags & q_expanded.keys():
            s += TAG_WEIGHT_BY_PREC.get(_prec_of(tag), 1.0)
        if s > 0:
            out[did] = s
    return out


# ---------------------------------------------------------------------------
# Candidate B: RANGE_OVERLAP + year tag boost.
# ---------------------------------------------------------------------------
def _te_to_range(te: TimeExpression) -> tuple[int, int] | None:
    """Returns (earliest_us, latest_us). Falls back to None if unbounded."""
    ivs = flatten_intervals(te)
    if not ivs:
        return None
    # union of all intervals
    e = min(iv.earliest_us for iv in ivs if iv.earliest_us is not None)
    l = max(iv.latest_us for iv in ivs if iv.latest_us is not None)
    return (e, l) if e <= l else None


def _te_to_year_tags(te: TimeExpression) -> set[str]:
    ts = lattice_tags_for_expression(te)
    out: set[str] = set()
    for prec, t in ts.absolute:
        if prec in ("year", "quarter", "month"):
            out.add(t)
    return out


def candidate_B_range_overlap(
    q_tes: list[TimeExpression],
    doc_ranges: dict[str, list[tuple[int, int]]],
    doc_year_tags: dict[str, set[str]],
) -> dict[str, float]:
    """Database equivalent (Postgres):
        -- Primary:
        SELECT doc_id, EXTRACT(EPOCH FROM (LEAST(d.latest, :q_l) -
                                            GREATEST(d.earliest, :q_e)))
        FROM   doc_ranges d
        WHERE  daterange(d.earliest, d.latest) && daterange(:q_e, :q_l);

        -- Boost (when query has only year tag):
        SELECT doc_id FROM doc_year_tags WHERE tag = ANY(:year_tags);

    Indexable as: GIST on tsrange / daterange (range-overlap).

    Score = sum over q-TEs of: max-over-d-TEs of (overlap_seconds /
    union_seconds), plus year-tag-boost when no overlap.
    """
    q_ranges: list[tuple[int, int]] = []
    q_year_tags: set[str] = set()
    for te in q_tes:
        r = _te_to_range(te)
        if r is not None:
            q_ranges.append(r)
        q_year_tags |= _te_to_year_tags(te)

    out: dict[str, float] = {}
    if q_ranges:
        for did, drs in doc_ranges.items():
            if not drs:
                continue
            best = 0.0
            for qe, ql in q_ranges:
                qspan = max(1, ql - qe)
                for de, dl in drs:
                    o_lo = max(qe, de)
                    o_hi = min(ql, dl)
                    if o_hi > o_lo:
                        u_lo = min(qe, de)
                        u_hi = max(ql, dl)
                        uspan = max(1, u_hi - u_lo)
                        # Jaccard-ish.
                        sc = (o_hi - o_lo) / uspan
                        if sc > best:
                            best = sc
            if best > 0:
                out[did] = best
    # Year-tag boost regardless (helps recurrences and pure-year queries)
    if q_year_tags:
        for did, dts in doc_year_tags.items():
            if dts & q_year_tags:
                out[did] = max(out.get(did, 0.0), 0.3)
    return out


# ---------------------------------------------------------------------------
# Candidate C: PER_TE_MAX — each doc-TE is a separate row.
# ---------------------------------------------------------------------------
def candidate_C_per_te_max(
    q_tes: list[TimeExpression],
    doc_te_rows: list[tuple[str, set[str], tuple[int, int] | None]],
    # row: (doc_id, tag_set_for_this_TE, range_for_this_TE_or_None)
) -> dict[str, float]:
    """Database equivalent (Postgres / Elasticsearch nested):

        -- Each doc has multiple temporal-rows; index per-row.
        SELECT doc_id, MAX(per_row_score) FROM (
          SELECT doc_id,
            (count(matched tags) * w_tag +
             range_overlap_jaccard) AS per_row_score
          FROM doc_te_rows
          WHERE tag = ANY(:expanded) OR range && :q_range
          GROUP BY te_row_id
        ) GROUP BY doc_id;

    Avoids tag-Jaccard dilution where a doc with 3 TEs has its
    intersection-with-query / union-with-query crashed by 2 unrelated TEs.
    Picks MAX per doc.
    """
    # Build query expanded set + ranges
    q_tagset: set[str] = set()
    for te in q_tes:
        ts = lattice_tags_for_expression(te)
        exp = expand_query_tags(ts, down_levels=0)
        q_tagset |= set(exp.keys())
    q_ranges: list[tuple[int, int]] = []
    for te in q_tes:
        r = _te_to_range(te)
        if r is not None:
            q_ranges.append(r)

    if not q_tagset and not q_ranges:
        return {}

    out: dict[str, float] = {}
    for did, row_tags, row_range in doc_te_rows:
        # Tag component: weighted granularity overlap, normalized by
        # (1 + |row_tags|) so a tag-rich row isn't unfairly boosted.
        tag_sc = 0.0
        if q_tagset and row_tags:
            for t in row_tags & q_tagset:
                tag_sc += TAG_WEIGHT_BY_PREC.get(_prec_of(t), 1.0)
            tag_sc /= max(1.0, math.log2(2 + len(row_tags)))
        # Range component: best Jaccard-ish overlap.
        rng_sc = 0.0
        if row_range is not None and q_ranges:
            de, dl = row_range
            for qe, ql in q_ranges:
                o_lo = max(qe, de)
                o_hi = min(ql, dl)
                if o_hi > o_lo:
                    u_lo = min(qe, de)
                    u_hi = max(ql, dl)
                    uspan = max(1, u_hi - u_lo)
                    s = (o_hi - o_lo) / uspan
                    if s > rng_sc:
                        rng_sc = s
        # Per-row score: tags + range_bonus.
        row_score = tag_sc + 2.0 * rng_sc
        if row_score > out.get(did, 0.0):
            out[did] = row_score
    return out


# ---------------------------------------------------------------------------
# Candidate D: RRF over A + B.
# ---------------------------------------------------------------------------
def candidate_D_rrf_tag_range(
    scores_A: dict, scores_B: dict, k: int = 60
) -> dict[str, float]:
    """Run A and B as separate DB queries, merge by RRF on top-K each."""
    K = 100
    rA = [d for d, _ in sorted(scores_A.items(), key=lambda x: x[1], reverse=True)[:K]]
    rB = [d for d, _ in sorted(scores_B.items(), key=lambda x: x[1], reverse=True)[:K]]
    fused = rrf([rA, rB], k=k)
    return dict(fused)


# ---------------------------------------------------------------------------
# Build per-bench data structures.
# ---------------------------------------------------------------------------
def build_doc_tag_index(
    doc_ext: dict[str, list[TimeExpression]],
) -> dict[str, set[str]]:
    """Per-doc UNION of all TE tags (lattice-style)."""
    out: dict[str, set[str]] = {}
    for did, tes in doc_ext.items():
        s: set[str] = set()
        for te in tes:
            ts = lattice_tags_for_expression(te)
            for _p, t in ts.absolute:
                s.add(t)
            s |= ts.cyclical
        out[did] = s
    return out


def build_doc_ranges(
    doc_ext: dict[str, list[TimeExpression]],
) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for did, tes in doc_ext.items():
        rows: list[tuple[int, int]] = []
        for te in tes:
            r = _te_to_range(te)
            if r is not None:
                rows.append(r)
        out[did] = rows
    return out


def build_doc_year_tags(
    doc_ext: dict[str, list[TimeExpression]],
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for did, tes in doc_ext.items():
        s: set[str] = set()
        for te in tes:
            s |= _te_to_year_tags(te)
        out[did] = s
    return out


def build_doc_te_rows(doc_ext: dict[str, list[TimeExpression]]):
    """Per-TE rows for candidate C."""
    rows: list[tuple[str, set[str], tuple[int, int] | None]] = []
    for did, tes in doc_ext.items():
        if not tes:
            rows.append((did, set(), None))  # empty placeholder
            continue
        for te in tes:
            ts = lattice_tags_for_expression(te)
            tag_set: set[str] = set()
            for _p, t in ts.absolute:
                tag_set.add(t)
            tag_set |= ts.cyclical
            rng = _te_to_range(te)
            rows.append((did, tag_set, rng))
    return rows


# ---------------------------------------------------------------------------
# T_lblend baseline (current production), exactly mirroring force_pick_optimizers_eval.
# ---------------------------------------------------------------------------
def baseline_t_lblend(q_mem, doc_mem, l_per_doc) -> dict[str, float]:
    T_ALPHA, T_GAMMA, T_DELTA = 0.20, 0.20, 0.60
    q_tags = q_mem.get("multi_tags") or set()
    q_ivs = q_mem.get("intervals") or []
    raw_iv = {
        did: interval_pair_best(q_ivs, b["intervals"]) for did, b in doc_mem.items()
    }
    max_iv = max(raw_iv.values()) if raw_iv else 0.0
    out: dict[str, float] = {}
    for did, b in doc_mem.items():
        iv_norm = raw_iv[did] / max_iv if max_iv > 0 else 0.0
        l_sc = l_per_doc.get(did, 0.0)
        out[did] = (
            T_ALPHA * iv_norm
            + T_GAMMA * tag_score(q_tags, b["multi_tags"])
            + T_DELTA * l_sc
        )
    return out


# ---------------------------------------------------------------------------
# Eval helpers.
# ---------------------------------------------------------------------------
def topk(scores: dict[str, float], k: int) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def hit_rank(ranking: list[str], gold: set[str], k: int = 10) -> int | None:
    for i, d in enumerate(ranking[:k]):
        if d in gold:
            return i + 1
    return None


def fuse_with_s(t_scores: dict, s_scores: dict, w_T: float = 0.4) -> list[str]:
    """T+S linear-blend (matches the production fusion top-K-merged style)."""
    if not t_scores:
        return [
            d for d, _ in sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
        ]
    fused = score_blend(
        {"T": t_scores, "S": s_scores},
        {"T": w_T, "S": 1.0 - w_T},
        top_k_per=40,
        dispersion_cv_ref=0.20,
    )
    primary = [d for d, _ in fused]
    seen = set(primary)
    tail = sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
    return primary + [d for d, _ in tail if d not in seen]


def fuse_rrf_with_s(t_scores: dict, s_scores: dict, K: int = 100) -> list[str]:
    """RRF-fuse T and S top-K each. Pure DB-friendly merge."""
    rT = topk(t_scores, K) if t_scores else []
    rS = topk(s_scores, K)
    fused = rrf([rT, rS], k=60)
    primary = [d for d, _ in fused]
    seen = set(primary)
    tail = [
        d
        for d, _ in sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return primary + tail


# ---------------------------------------------------------------------------
# Run one bench.
# ---------------------------------------------------------------------------
async def run_bench(
    name: str, docs_path: str, queries_path: str, gold_path: str, cache_label: str
):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    print(
        f"\n=== {name}: {len(docs)} docs, {len(queries)} queries (cache={cache_label}) ===",
        flush=True,
    )
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_label)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_label)
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
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # Baseline lattice (for T_lblend).
    lat_db = ROOT / "cache" / "t_db_alts" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for did, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(did, ts.absolute, ts.cyclical)
    per_q_l = {
        qid: lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)[0]
        for qid in qids
    }

    # Pre-build candidate indexes
    doc_tag_idx = build_doc_tag_index(doc_ext)
    doc_ranges = build_doc_ranges(doc_ext)
    doc_year_tags = build_doc_year_tags(doc_ext)
    doc_te_rows = build_doc_te_rows(doc_ext)

    variants = [
        "T_lblend",
        "A_hier_tags",
        "B_range_overlap",
        "C_per_te_max",
        "D_rrf_AB",
    ]
    # report buckets
    R_T_only = {v: [] for v in variants}
    R_TS_blend = {v: [] for v in variants}
    R_TS_rrf = {v: [] for v in variants}

    n_used = 0
    for q in queries:
        qid = q["query_id"]
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_used += 1
        q_tes = q_ext.get(qid, [])
        q_m = q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()})

        # Compute T scores per candidate
        t_baseline = baseline_t_lblend(q_m, doc_mem, per_q_l.get(qid, {}))
        t_A = candidate_A_hier_tags(q_tes, doc_tag_idx)
        t_B = candidate_B_range_overlap(q_tes, doc_ranges, doc_year_tags)
        t_C = candidate_C_per_te_max(q_tes, doc_te_rows)
        t_D = candidate_D_rrf_tag_range(t_A, t_B)

        cands = {
            "T_lblend": t_baseline,
            "A_hier_tags": t_A,
            "B_range_overlap": t_B,
            "C_per_te_max": t_C,
            "D_rrf_AB": t_D,
        }

        s_scores = per_q_s[qid]
        for var, t_sc in cands.items():
            t_only = topk(t_sc, 10) if t_sc else []
            R_T_only[var].append(hit_rank(t_only, gold_set))
            R_TS_blend[var].append(
                hit_rank(fuse_with_s(t_sc, s_scores, w_T=0.4), gold_set)
            )
            R_TS_rrf[var].append(hit_rank(fuse_rrf_with_s(t_sc, s_scores), gold_set))

    def metric_table(buckets):
        out = {}
        for var, ranks in buckets.items():
            n = len(ranks)
            r1 = sum(1 for x in ranks if x is not None and x <= 1) / n if n else 0.0
            r5 = sum(1 for x in ranks if x is not None and x <= 5) / n if n else 0.0
            mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
            out[var] = {"R@1": r1, "R@5": r5, "MRR": mrr, "n": n}
        return out

    summary = {
        "n_queries": n_used,
        "T_only": metric_table(R_T_only),
        "T+S_blend(w=0.4)": metric_table(R_TS_blend),
        "T+S_rrf": metric_table(R_TS_rrf),
    }
    # Print
    print("  -- T-only --")
    for var in variants:
        m = summary["T_only"][var]
        print(
            f"    {var:18}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  MRR={m['MRR']:.3f}"
        )
    print("  -- T+S blend (w_T=0.4) --")
    for var in variants:
        m = summary["T+S_blend(w=0.4)"][var]
        print(
            f"    {var:18}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  MRR={m['MRR']:.3f}"
        )
    print("  -- T+S RRF --")
    for var in variants:
        m = summary["T+S_rrf"][var]
        print(
            f"    {var:18}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  MRR={m['MRR']:.3f}"
        )
    return summary


async def main():
    out: dict[str, Any] = {}
    out["hard_bench"] = await run_bench(
        "hard_bench",
        "hard_bench_docs.jsonl",
        "hard_bench_queries.jsonl",
        "hard_bench_gold.jsonl",
        "v7l-hard_bench",
    )
    out["temporal_essential"] = await run_bench(
        "temporal_essential",
        "temporal_essential_docs.jsonl",
        "temporal_essential_queries.jsonl",
        "temporal_essential_gold.jsonl",
        "v7l-temporal_essential",
    )
    out["tempreason_small"] = await run_bench(
        "tempreason_small",
        "real_benchmark_small_docs.jsonl",
        "real_benchmark_small_queries.jsonl",
        "real_benchmark_small_gold.jsonl",
        "v7l-tempreason_small",
    )

    out_path = ROOT / "results" / "T_db_alternatives.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
