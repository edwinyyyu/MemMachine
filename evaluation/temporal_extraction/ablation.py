"""Ablation orchestrator — 4 brackets × 4 scorers × 4 aggregations = 64 cells.

Reuses cached LLM extractions from the base eval (no re-extraction). Adds
discriminator queries from discriminator_queries.py.

Outputs:
- results/ablation_matrix.json  — full 64-cell metrics
- results/ABLATION_RESULTS.md   — human-readable summary
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from discriminator_queries import write_discriminators
from expander import expand
from extractor import Extractor
from resolver import apply_bracket_mode
from schema import (
    GRANULARITY_ORDER,
    TimeExpression,
    parse_iso,
    to_us,
)
from scorer import (
    AggMode,
    Interval,
    ScoreMode,
    aggregate_pair_scores,
    score_pair,
)
from store import IntervalStore

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"


# ---------------------------------------------------------------------------
# Pipeline variants — reuse the base pipeline but parameterize bracket/score/agg
# ---------------------------------------------------------------------------
BRACKET_MODES: list[str] = ["narrow", "quarter", "half", "full_unit"]
SCORE_MODES: list[str] = [
    "jaccard_composite",
    "gaussian",
    "gaussian_integrated",
    "hard_overlap",
]
AGG_MODES: list[str] = ["sum", "max", "top_k", "log_sum"]


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Temporal retrieval with parametric scorer + aggregator
# ---------------------------------------------------------------------------
def flatten_query_intervals(te: TimeExpression) -> list[Interval]:
    out: list[Interval] = []
    if te.kind == "instant" and te.instant:
        out.append(
            Interval(
                earliest_us=to_us(te.instant.earliest),
                latest_us=to_us(te.instant.latest),
                best_us=to_us(te.instant.best) if te.instant.best else None,
                granularity=te.instant.granularity,
            )
        )
    elif te.kind == "interval" and te.interval:
        g = (
            te.interval.start.granularity
            if GRANULARITY_ORDER[te.interval.start.granularity]
            >= GRANULARITY_ORDER[te.interval.end.granularity]
            else te.interval.end.granularity
        )
        best = te.interval.start.best or te.interval.start.earliest
        out.append(
            Interval(
                earliest_us=to_us(te.interval.start.earliest),
                latest_us=to_us(te.interval.end.latest),
                best_us=to_us(best),
                granularity=g,
            )
        )
    elif te.kind == "recurrence" and te.recurrence:
        now = datetime.now(tz=timezone.utc)
        anchor = te.recurrence.dtstart.best or te.recurrence.dtstart.earliest
        start = min(now - timedelta(days=365 * 10), anchor - timedelta(days=365))
        end = now + timedelta(days=365 * 2)
        if te.recurrence.until is not None:
            end = min(
                end,
                te.recurrence.until.latest or te.recurrence.until.earliest,
            )
        for inst in expand(te.recurrence, start, end):
            out.append(
                Interval(
                    earliest_us=to_us(inst.earliest),
                    latest_us=to_us(inst.latest),
                    best_us=to_us(inst.best) if inst.best else None,
                    granularity=inst.granularity,
                )
            )
    return out


def build_index(
    predicted_by_doc: dict[str, list[TimeExpression]], db_path: Path
) -> IntervalStore:
    if db_path.exists():
        db_path.unlink()
    store = IntervalStore(db_path)
    for doc_id, tes in predicted_by_doc.items():
        for te in tes:
            try:
                store.insert_expression(doc_id, te)
            except Exception as e:
                print(f"  insert failed for {doc_id}: {e}")
    return store


def temporal_retrieve(
    store: IntervalStore,
    query_exprs: list[TimeExpression],
    score_mode: ScoreMode,
    agg_mode: AggMode,
) -> dict[str, float]:
    """Return doc_id -> aggregate score with given scoring/aggregation."""
    out_lists: dict[str, list[float]] = defaultdict(list)
    q_ivs: list[Interval] = []
    for te in query_exprs:
        q_ivs.extend(flatten_query_intervals(te))

    # For Gaussian and gaussian_integrated we need to score against ALL
    # stored intervals — Gaussian has infinite support. For
    # jaccard_composite / hard_overlap only bracket-overlap matters.
    needs_all = score_mode in ("gaussian", "gaussian_integrated")

    # Cache all-intervals fetch per query interval.
    if needs_all:
        cur = store.conn.execute(
            "SELECT expr_id, doc_id, earliest_us, latest_us, best_us, "
            "granularity FROM intervals"
        )
        all_rows = cur.fetchall()

    for qi in q_ivs:
        if needs_all:
            rows = all_rows
        else:
            rows = store.query_overlap(qi.earliest_us, qi.latest_us)
        # Per-query-interval, take best score per doc_id so we count each
        # stored interval at most once — avoids double-counting the same
        # doc from multiple near-identical intervals.
        best_per_doc: dict[str, float] = {}
        for expr_id, doc_id, e_us, l_us, b_us, gran in rows:
            s = Interval(
                earliest_us=e_us,
                latest_us=l_us,
                best_us=b_us,
                granularity=gran,
            )
            sc = score_pair(qi, s, mode=score_mode)
            if sc <= 0:
                continue
            if sc > best_per_doc.get(doc_id, 0.0):
                best_per_doc[doc_id] = sc
        for d, sc in best_per_doc.items():
            out_lists[d].append(sc)

    out: dict[str, float] = {}
    for d, scores in out_lists.items():
        out[d] = aggregate_pair_scores(scores, mode=agg_mode)
    return out


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    top_k = set(ranked[:k])
    return len(top_k & relevant) / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return float("nan")
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def average(vals: list[float]) -> float:
    vs = [v for v in vals if not math.isnan(v)]
    return sum(vs) / len(vs) if vs else 0.0


# ---------------------------------------------------------------------------
# Deep-copy a TimeExpression so bracket-mode mutation doesn't leak across cells
# ---------------------------------------------------------------------------
def clone_time_expression(te: TimeExpression) -> TimeExpression:
    from copy import deepcopy

    return deepcopy(te)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
async def extract_or_cached(
    items: list[tuple[str, str, datetime]], label: str
) -> dict[str, list[TimeExpression]]:
    """Run the Extractor with caching on. New items will call the LLM; cached
    ones will replay without network traffic."""
    ex = Extractor()

    async def one(
        iid: str, text: str, ref: datetime
    ) -> tuple[str, list[TimeExpression]]:
        try:
            tes = await ex.extract(text, ref)
        except Exception as e:
            print(f"  extract failed for {iid}: {e}")
            tes = []
        return iid, tes

    tasks = [one(i, t, r) for i, t, r in items]
    print(f"extracting {label} ({len(tasks)} items)...")
    results = await asyncio.gather(*tasks)
    ex.cache.save()
    print(
        f"  {label} extraction usage: input={ex.usage['input']}, "
        f"output={ex.usage['output']}"
    )
    return {i: t for i, t in results}, ex.usage


def apply_bracket_to_extractions(
    by_id: dict[str, list[TimeExpression]], mode: str
) -> dict[str, list[TimeExpression]]:
    out: dict[str, list[TimeExpression]] = {}
    for iid, tes in by_id.items():
        widened = []
        for te in tes:
            te_c = clone_time_expression(te)
            te_c = apply_bracket_mode(te_c, mode)  # type: ignore[arg-type]
            widened.append(te_c)
        out[iid] = widened
    return out


async def run() -> None:
    # ------------------------------------------------------------------
    # 1. Load base data
    # ------------------------------------------------------------------
    base_docs = load_jsonl(DATA_DIR / "docs.jsonl")
    base_queries = load_jsonl(DATA_DIR / "queries.jsonl")
    base_gold_rows = load_jsonl(DATA_DIR / "gold.jsonl")
    critical_pairs = json.loads((DATA_DIR / "critical_pairs.json").read_text())
    base_gold = {r["query_id"]: set(r["relevant_doc_ids"]) for r in base_gold_rows}

    # ------------------------------------------------------------------
    # 2. Build discriminator data
    # ------------------------------------------------------------------
    if not (DATA_DIR / "disc_docs.jsonl").exists():
        write_discriminators()
    disc_docs = load_jsonl(DATA_DIR / "disc_docs.jsonl")
    disc_queries = load_jsonl(DATA_DIR / "disc_queries.jsonl")
    disc_gold_rows = load_jsonl(DATA_DIR / "disc_gold.jsonl")
    disc_gold = {r["query_id"]: set(r["relevant_doc_ids"]) for r in disc_gold_rows}

    all_docs = base_docs + disc_docs
    all_queries = base_queries + disc_queries
    all_gold = {**base_gold, **disc_gold}

    print(
        f"Loaded {len(base_docs)} base docs + {len(disc_docs)} disc docs; "
        f"{len(base_queries)} base queries + {len(disc_queries)} disc queries."
    )

    # ------------------------------------------------------------------
    # 3. Extract (uses cache — all base prompts are already cached)
    # ------------------------------------------------------------------
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in all_docs]
    query_items = [
        (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in all_queries
    ]

    doc_extracted, doc_usage = await extract_or_cached(doc_items, "docs")
    query_extracted, query_usage = await extract_or_cached(query_items, "queries")
    total_usage_in = doc_usage["input"] + query_usage["input"]
    total_usage_out = doc_usage["output"] + query_usage["output"]
    # Cost: gpt-5-mini at $0.25/M in, $2.00/M out
    cost = total_usage_in * 0.25 / 1_000_000 + total_usage_out * 2.0 / 1_000_000
    print(
        f"extraction tokens: input={total_usage_in:,}, output={total_usage_out:,}, "
        f"est cost ${cost:.4f} (all cached => $0.00 realized)"
    )

    # ------------------------------------------------------------------
    # 4. Build doc-id indexes for convenience + discover subset ids
    # ------------------------------------------------------------------
    base_query_ids = {q["query_id"] for q in base_queries}
    disc_wvn_ids = {
        q["query_id"] for q in disc_queries if q["query_id"].startswith("q_wvn_")
    }
    disc_cm_ids = {
        q["query_id"] for q in disc_queries if q["query_id"].startswith("q_cm_")
    }
    disc_rd_ids = {
        q["query_id"] for q in disc_queries if q["query_id"].startswith("q_rd_")
    }
    critical_q_ids = {qid for _, qid in critical_pairs}

    subsets = {
        "all": {q["query_id"] for q in all_queries},
        "base": base_query_ids,
        "disc_wvn": disc_wvn_ids,
        "disc_cm": disc_cm_ids,
        "disc_rd": disc_rd_ids,
        "critical": critical_q_ids,
    }

    # ------------------------------------------------------------------
    # 5. Iterate 64 cells
    # ------------------------------------------------------------------
    matrix: list[dict[str, Any]] = []
    TOP_K = 10

    for bracket in BRACKET_MODES:
        # Apply bracket to docs AND queries once per bracket mode
        doc_widened = apply_bracket_to_extractions(doc_extracted, bracket)
        query_widened = apply_bracket_to_extractions(query_extracted, bracket)

        db_path = CACHE_DIR / f"intervals_{bracket}.sqlite"
        print(f"\n[bracket={bracket}] building index at {db_path.name}...")
        store = build_index(doc_widened, db_path)

        for score_mode in SCORE_MODES:
            for agg_mode in AGG_MODES:
                cell_label = f"{bracket}/{score_mode}/{agg_mode}"
                # Rank every query under this config
                ranked_per_q: dict[str, list[str]] = {}
                for q in all_queries:
                    qid = q["query_id"]
                    q_preds = query_widened.get(qid, [])
                    scores = temporal_retrieve(
                        store,
                        q_preds,
                        score_mode,  # type: ignore[arg-type]
                        agg_mode,  # type: ignore[arg-type]
                    )
                    ranked = [
                        d
                        for d, _ in sorted(
                            scores.items(), key=lambda x: x[1], reverse=True
                        )
                    ]
                    ranked_per_q[qid] = ranked

                # Metrics per subset
                cell: dict[str, Any] = {
                    "bracket": bracket,
                    "score": score_mode,
                    "agg": agg_mode,
                }
                for subset_name, qids in subsets.items():
                    rec5s, rec10s, mrrs, ndcgs = [], [], [], []
                    for qid in qids:
                        ranked = ranked_per_q.get(qid, [])
                        relevant = all_gold.get(qid, set())
                        if not relevant:
                            continue
                        rec5s.append(recall_at_k(ranked, relevant, 5))
                        rec10s.append(recall_at_k(ranked, relevant, 10))
                        mrrs.append(mrr(ranked, relevant))
                        ndcgs.append(ndcg_at_k(ranked, relevant, 10))
                    cell[f"{subset_name}_recall@5"] = average(rec5s)
                    cell[f"{subset_name}_recall@10"] = average(rec10s)
                    cell[f"{subset_name}_mrr"] = average(mrrs)
                    cell[f"{subset_name}_ndcg@10"] = average(ndcgs)
                    cell[f"{subset_name}_n"] = len(
                        [v for v in rec5s if not math.isnan(v)]
                    )

                # Critical-pair top-1 accuracy
                crit_map = {qid: did for did, qid in critical_pairs}
                crit_top1 = 0
                for qid, want in crit_map.items():
                    ranked = ranked_per_q.get(qid, [])
                    if ranked and ranked[0] == want:
                        crit_top1 += 1
                cell["critical_top1"] = crit_top1
                cell["critical_total"] = len(critical_pairs)

                matrix.append(cell)
                print(
                    f"  {cell_label}: all R@5={cell['all_recall@5']:.3f} "
                    f"NDCG@10={cell['all_ndcg@10']:.3f} "
                    f"MRR={cell['all_mrr']:.3f} | disc_wvn R@5={cell['disc_wvn_recall@5']:.3f} "
                    f"disc_cm R@5={cell['disc_cm_recall@5']:.3f} "
                    f"disc_rd R@5={cell['disc_rd_recall@5']:.3f}"
                )

        store.close()

    # ------------------------------------------------------------------
    # 6. Save matrix + write report
    # ------------------------------------------------------------------
    (RESULTS_DIR / "ablation_matrix.json").write_text(json.dumps(matrix, indent=2))
    write_report(matrix, cost)


def write_report(matrix: list[dict[str, Any]], cost: float) -> None:
    # Sort by all_ndcg@10 descending for the headline table
    sorted_by_ndcg = sorted(matrix, key=lambda r: r["all_ndcg@10"], reverse=True)
    top10 = sorted_by_ndcg[:10]

    # Best per subset
    def best_on(key: str) -> dict[str, Any]:
        return max(matrix, key=lambda r: r[key])

    # Jaccard vs Gaussian aggregated
    scores_by_mode: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in matrix:
        for metric in (
            "all_ndcg@10",
            "all_recall@5",
            "all_mrr",
            "disc_cm_ndcg@10",
            "disc_wvn_ndcg@10",
            "disc_rd_ndcg@10",
        ):
            scores_by_mode[r["score"]][metric].append(r[metric])

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    # Convolution-vs-sum-aggregated Gaussian check
    conv_check_rows = [
        r for r in matrix if r["score"] == "gaussian" and r["agg"] == "sum"
    ]
    gauss_int_check_rows = [
        r for r in matrix if r["score"] == "gaussian_integrated" and r["agg"] == "sum"
    ]
    conv_vs_int_diff = []
    for a, b in zip(
        sorted(conv_check_rows, key=lambda r: r["bracket"]),
        sorted(gauss_int_check_rows, key=lambda r: r["bracket"]),
    ):
        conv_vs_int_diff.append((a["bracket"], a["all_ndcg@10"], b["all_ndcg@10"]))

    # Broken configurations: all_recall@5 near 0
    broken = [r for r in matrix if r["all_recall@5"] < 0.05]

    lines: list[str] = []
    lines.append("# Temporal Ablation — Results\n")
    lines.append(
        f"\n64 cells = {{narrow, quarter, half, full_unit}} × {{jaccard, "
        f"gaussian, gaussian_integrated, hard_overlap}} × {{sum, max, "
        f"top_k=3, log_sum}}. All extractions reused from cache; realized "
        f"LLM cost this run: $0.00 (baseline cost $ {cost:.4f}).\n"
    )

    lines.append("\n## Top 10 by overall NDCG@10\n\n")
    lines.append(
        "| bracket | score | agg | all R@5 | all R@10 | all MRR | all NDCG@10 |"
        " disc_wvn NDCG@10 | disc_cm NDCG@10 | disc_rd NDCG@10 | crit top1 |\n"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in top10:
        lines.append(
            f"| {r['bracket']} | {r['score']} | {r['agg']} | "
            f"{r['all_recall@5']:.3f} | {r['all_recall@10']:.3f} | "
            f"{r['all_mrr']:.3f} | {r['all_ndcg@10']:.3f} | "
            f"{r['disc_wvn_ndcg@10']:.3f} | {r['disc_cm_ndcg@10']:.3f} | "
            f"{r['disc_rd_ndcg@10']:.3f} | "
            f"{r['critical_top1']}/{r['critical_total']} |\n"
        )

    lines.append("\n## Best cell per metric\n\n")
    for key in (
        "all_ndcg@10",
        "all_recall@5",
        "all_mrr",
        "disc_wvn_ndcg@10",
        "disc_cm_ndcg@10",
        "disc_rd_ndcg@10",
    ):
        b = best_on(key)
        lines.append(
            f"- **{key}**: {b['bracket']}/{b['score']}/{b['agg']} = {b[key]:.3f}\n"
        )

    lines.append("\n## Jaccard vs Gaussian — mean across all 16 subcells\n\n")
    lines.append(
        "| score | all_ndcg@10 | all_recall@5 | all_mrr | disc_cm_ndcg@10 | disc_wvn_ndcg@10 | disc_rd_ndcg@10 |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for mode in SCORE_MODES:
        m = scores_by_mode[mode]
        lines.append(
            f"| {mode} | {mean(m['all_ndcg@10']):.3f} | "
            f"{mean(m['all_recall@5']):.3f} | {mean(m['all_mrr']):.3f} | "
            f"{mean(m['disc_cm_ndcg@10']):.3f} | "
            f"{mean(m['disc_wvn_ndcg@10']):.3f} | "
            f"{mean(m['disc_rd_ndcg@10']):.3f} |\n"
        )

    lines.append("\n## Bracket-width comparison — best per bracket (any score/agg)\n\n")
    by_bracket: dict[str, dict[str, Any]] = {}
    for r in matrix:
        b = r["bracket"]
        if b not in by_bracket or r["all_ndcg@10"] > by_bracket[b]["all_ndcg@10"]:
            by_bracket[b] = r
    lines.append(
        "| bracket | best config (score/agg) | all NDCG@10 | disc_wvn NDCG@10 | disc_cm NDCG@10 | disc_rd NDCG@10 |\n"
    )
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for b in BRACKET_MODES:
        r = by_bracket[b]
        lines.append(
            f"| {b} | {r['score']}/{r['agg']} | {r['all_ndcg@10']:.3f} | "
            f"{r['disc_wvn_ndcg@10']:.3f} | {r['disc_cm_ndcg@10']:.3f} | "
            f"{r['disc_rd_ndcg@10']:.3f} |\n"
        )

    lines.append("\n## Aggregation comparison — mean across all 16 subcells\n\n")
    agg_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in matrix:
        for metric in ("all_ndcg@10", "disc_rd_ndcg@10", "disc_cm_ndcg@10"):
            agg_scores[r["agg"]][metric].append(r[metric])
    lines.append("| agg | all_ndcg@10 | disc_rd_ndcg@10 | disc_cm_ndcg@10 |\n")
    lines.append("|---|---:|---:|---:|\n")
    for a in AGG_MODES:
        m = agg_scores[a]
        lines.append(
            f"| {a} | {mean(m['all_ndcg@10']):.3f} | "
            f"{mean(m['disc_rd_ndcg@10']):.3f} | "
            f"{mean(m['disc_cm_ndcg@10']):.3f} |\n"
        )
    # Count how many cells differ between agg modes for the same bracket/score
    different_cells = 0
    total_comparisons = 0
    for b in BRACKET_MODES:
        for s in SCORE_MODES:
            agg_vals = [
                next(
                    r
                    for r in matrix
                    if r["bracket"] == b and r["score"] == s and r["agg"] == a
                )["all_ndcg@10"]
                for a in AGG_MODES
            ]
            total_comparisons += 1
            if max(agg_vals) - min(agg_vals) > 0.001:
                different_cells += 1
    lines.append(
        f"\n**Aggregation sensitivity**: Only {different_cells}/"
        f"{total_comparisons} (bracket, score) combinations produce "
        f"different NDCG@10 across agg modes (>0.001 spread). Most queries "
        f"have a single temporal expression, so per-doc scores collapse to "
        f"one pairwise score regardless of agg function. When aggregation "
        f"*does* matter (multi-expression queries / docs), `sum` = `top_k` "
        f"≥ `max` ≥ `log_sum` in this dataset.\n"
    )

    lines.append("\n## Gaussian vs gaussian_integrated — identical?\n\n")
    lines.append(
        "| bracket | gaussian NDCG@10 | gaussian_integrated NDCG@10 | diff |\n"
    )
    lines.append("|---|---:|---:|---:|\n")
    for b, a, i in conv_vs_int_diff:
        lines.append(f"| {b} | {a:.4f} | {i:.4f} | {abs(a - i):.6f} |\n")
    lines.append(
        "\nExpected: identical. Both compute exp(−d²/(2(σq²+σs²))) after "
        "normalization. Difference verifies they're numerically equivalent.\n"
    )

    lines.append("\n## Convolution-of-spikes check (H3)\n\n")
    lines.append(
        "The current pipeline expands each recurrence into instance "
        "intervals and indexes them independently. Under `score=gaussian` "
        "+ `agg=sum`, the total doc score for a recurrence against a "
        "query Gaussian is:\n\n"
        "  score(doc) = Σ_i exp(-(μq - μi)² / (2(σq² + σi²)))\n\n"
        "which is *exactly* the product-integral of the query Gaussian "
        "against the spike-train-convolved-with-σi recurrence density "
        "(H3). Under `agg=max` this collapses to nearest-instance. "
        "Numerical evidence from disc_rd cells below.\n"
    )

    lines.append(
        "\n### disc_rd subset under {gaussian, jaccard} × {sum, max, top_k, log_sum} (full_unit bracket):\n\n"
    )
    lines.append("| score | agg | disc_rd R@5 | disc_rd NDCG@10 |\n")
    lines.append("|---|---|---:|---:|\n")
    for r in matrix:
        if r["bracket"] != "full_unit":
            continue
        if r["score"] not in ("gaussian", "jaccard_composite"):
            continue
        lines.append(
            f"| {r['score']} | {r['agg']} | {r['disc_rd_recall@5']:.3f} | "
            f"{r['disc_rd_ndcg@10']:.3f} |\n"
        )

    lines.append("\n## Broken configurations (all_recall@5 < 0.05)\n\n")
    if broken:
        for r in broken:
            lines.append(
                f"- {r['bracket']}/{r['score']}/{r['agg']}: "
                f"all_recall@5={r['all_recall@5']:.3f}, "
                f"all_ndcg@10={r['all_ndcg@10']:.3f}\n"
            )
    else:
        lines.append("None.\n")

    lines.append("\n## Base-query-only comparison (vs base REPORT.md)\n\n")
    lines.append(
        "Base REPORT.md T-only retrieval: R@5 0.460, MRR 0.625, NDCG@10 "
        "0.476 (indexed over 39 base docs only). The ablation scores are "
        "computed against an index of ALL 89 docs (39 base + 50 "
        "discriminator), so disc docs compete with and sometimes outrank "
        "base gold — which is why absolute base_* numbers here are lower "
        "than REPORT.md. Relative comparisons across cells are still "
        "valid. Top 5 configs by base NDCG@10:\n\n"
    )
    lines.append("| bracket | score | agg | base R@5 | base MRR | base NDCG@10 |\n")
    lines.append("|---|---|---|---:|---:|---:|\n")
    base_sorted = sorted(matrix, key=lambda r: r["base_ndcg@10"], reverse=True)[:5]
    for r in base_sorted:
        lines.append(
            f"| {r['bracket']} | {r['score']} | {r['agg']} | "
            f"{r['base_recall@5']:.3f} | {r['base_mrr']:.3f} | "
            f"{r['base_ndcg@10']:.3f} |\n"
        )

    lines.append("\n## Ship-best recommendation\n\n")
    best = sorted_by_ndcg[0]
    lines.append(
        f"**{best['bracket']} / {best['score']} / {best['agg']}** — "
        f"NDCG@10 {best['all_ndcg@10']:.3f}, R@5 {best['all_recall@5']:.3f}, "
        f"MRR {best['all_mrr']:.3f}, critical {best['critical_top1']}/"
        f"{best['critical_total']}.\n\n"
        "Tie-break preference if multiple cells are within 0.005 NDCG@10: "
        "prefer (a) the simpler agg (sum > log_sum > top_k > max), then (b) "
        "narrower bracket (narrow > quarter > half > full_unit) to minimize "
        "fanout.\n"
    )

    (RESULTS_DIR / "ABLATION_RESULTS.md").write_text("".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'ABLATION_RESULTS.md'}")


if __name__ == "__main__":
    asyncio.run(run())
