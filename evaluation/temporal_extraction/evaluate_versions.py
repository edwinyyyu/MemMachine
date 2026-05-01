"""Evaluate extractor versions v1-v6 and the gpt-5 upper bound.

Metrics per version:
- Extraction F1 (precision / recall) on docs + queries + discriminator.
- Resolution MAE (mean / median / P95) in seconds on matched pairs.
- Failure-case recall — how many of the v1-missed surfaces this version
  recovers (matches span-overlap ≥ 50%).
- Downstream retrieval R@5, R@10, MRR, NDCG@10 using the ship-best
  quarter / jaccard_composite / sum config from the existing ablation.
- LLM token usage + cost estimate.

Outputs:
- results/extractor_improvements.json — raw per-version metrics.
- results/extractor_improvements.md   — ranked summary table.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from eval import (
    extraction_metrics,
    load_jsonl,
    match_expressions,
)
from expander import expand
from extractor import Extractor as ExtractorV1
from extractor_common import (
    BaseImprovedExtractor,
)
from extractor_common import (
    extract_many as improved_extract_many,
)
from extractor_v2 import ExtractorV2
from extractor_v3 import ExtractorV3
from extractor_v4 import ExtractorV4
from extractor_v5 import ExtractorV5
from extractor_v6 import ExtractorV6
from resolver import apply_bracket_mode
from schema import (
    GRANULARITY_ORDER,
    TimeExpression,
    parse_iso,
    time_expression_from_dict,
    to_us,
)
from scorer import Interval, aggregate_pair_scores, score_pair
from store import IntervalStore

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache"

# gpt-5-mini price (est): $0.25/M in, $2.00/M out (per project eval.py)
# gpt-5 price (est):       $1.25/M in, $10.00/M out
COST_PER_MTOK = {
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5": (1.25, 10.0),
}


def _cost(usage: dict[str, int], model: str) -> float:
    pin, pout = COST_PER_MTOK.get(model, (0.25, 2.0))
    return (usage.get("input", 0) * pin + usage.get("output", 0) * pout) / 1_000_000


# ---------------------------------------------------------------------------
# v1 extraction wrapper (uses existing Extractor)
# ---------------------------------------------------------------------------
async def extract_v1(
    items: list[tuple[str, str, datetime]],
) -> tuple[dict[str, list[TimeExpression]], dict[str, int]]:
    ex = ExtractorV1()

    async def one(iid: str, text: str, ref: datetime):
        try:
            tes = await ex.extract(text, ref)
        except Exception as e:
            print(f"  v1 extract failed for {iid}: {e}")
            tes = []
        return iid, tes

    results = await asyncio.gather(*(one(i, t, r) for i, t, r in items))
    ex.cache.save()
    return dict(results), ex.usage


# ---------------------------------------------------------------------------
# vX extraction wrapper
# ---------------------------------------------------------------------------
async def extract_improved(
    cls,
    items: list[tuple[str, str, datetime]],
    model: str = "gpt-5-mini",
    cache_subdir: str | None = None,
) -> tuple[dict[str, list[TimeExpression]], dict[str, int]]:
    ex: BaseImprovedExtractor = cls(model=model, cache_subdir=cache_subdir)
    result = await improved_extract_many(ex, items)
    return result, ex.usage


# ---------------------------------------------------------------------------
# Extraction-quality evaluation
# ---------------------------------------------------------------------------
def _combined_metrics(
    docs: list[dict],
    queries: list[dict],
    pred_by_doc: dict[str, list[TimeExpression]],
    pred_by_q: dict[str, list[TimeExpression]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    doc_mr = []
    for d in docs:
        pred = pred_by_doc.get(d["doc_id"], [])
        gold = [time_expression_from_dict(g) for g in d["gold_expressions"]]
        doc_mr.append(match_expressions(pred, gold, d["text"]))
    q_mr = []
    for q in queries:
        pred = pred_by_q.get(q["query_id"], [])
        gold = [time_expression_from_dict(g) for g in q["gold_expressions"]]
        q_mr.append(match_expressions(pred, gold, q["text"]))
    return (
        extraction_metrics(doc_mr + q_mr),
        extraction_metrics(doc_mr),
        extraction_metrics(q_mr),
    )


# ---------------------------------------------------------------------------
# Failure-case recall
# ---------------------------------------------------------------------------
def _failure_recall(
    failure_cases: list[dict],
    docs: list[dict],
    queries: list[dict],
    pred_by_doc: dict[str, list[TimeExpression]],
    pred_by_q: dict[str, list[TimeExpression]],
) -> dict[str, Any]:
    docs_by_id = {d["doc_id"]: d for d in docs}
    queries_by_id = {q["query_id"]: q for q in queries}
    total = len(failure_cases)
    recovered = 0
    per_surface: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "recovered": 0}
    )
    recovered_ids: list[tuple[str, str]] = []
    missing_ids: list[tuple[str, str]] = []
    for fc in failure_cases:
        item_id = fc["item_id"]
        surface = fc["missed_surface"]
        stype = fc["type"]
        if stype == "doc":
            item = docs_by_id.get(item_id)
            pred = pred_by_doc.get(item_id, [])
        else:
            item = queries_by_id.get(item_id)
            pred = pred_by_q.get(item_id, [])
        if item is None:
            continue
        text = item["text"]
        # Gold span for this missed surface
        gold_expr = time_expression_from_dict(fc["gold_expr"])
        gs = gold_expr.span_start
        ge = gold_expr.span_end
        if gs is None or ge is None:
            idx = text.find(surface)
            if idx < 0:
                continue
            gs, ge = idx, idx + len(surface)
        g_len = max(1, ge - gs)
        # Check any predicted span overlaps ≥ 50% of min length
        got = False
        for te in pred:
            ps = te.span_start
            pe = te.span_end
            if ps is None or pe is None:
                idx = text.find(te.surface)
                if idx < 0:
                    continue
                ps, pe = idx, idx + len(te.surface)
            p_len = max(1, pe - ps)
            overlap = max(0, min(pe, ge) - max(ps, gs))
            frac = overlap / min(p_len, g_len)
            if frac >= 0.5:
                got = True
                break
        key = surface.lower()
        per_surface[key]["total"] += 1
        if got:
            recovered += 1
            per_surface[key]["recovered"] += 1
            recovered_ids.append((item_id, surface))
        else:
            missing_ids.append((item_id, surface))
    recall = recovered / total if total else 0.0
    return {
        "total": total,
        "recovered": recovered,
        "recall": recall,
        "by_surface": dict(per_surface),
        "missing_examples": missing_ids[:10],
    }


# ---------------------------------------------------------------------------
# Temporal retrieval at ship-best (quarter / jaccard_composite / sum)
# ---------------------------------------------------------------------------
def _build_index(
    predicted_by_doc: dict[str, list[TimeExpression]], db_path: Path
) -> IntervalStore:
    if db_path.exists():
        db_path.unlink()
    store = IntervalStore(db_path)
    for doc_id, tes in predicted_by_doc.items():
        for te in tes:
            try:
                store.insert_expression(doc_id, te)
            except Exception:
                pass
    return store


def _flatten_query_intervals(te: TimeExpression) -> list[Interval]:
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


def _temporal_retrieve(
    store: IntervalStore,
    query_exprs: list[TimeExpression],
) -> dict[str, float]:
    """quarter bracket applied upstream; scorer=jaccard_composite, agg=sum."""
    out_lists: dict[str, list[float]] = defaultdict(list)
    q_ivs: list[Interval] = []
    for te in query_exprs:
        q_ivs.extend(_flatten_query_intervals(te))
    for qi in q_ivs:
        rows = store.query_overlap(qi.earliest_us, qi.latest_us)
        best_per_doc: dict[str, float] = {}
        for expr_id, doc_id, e_us, l_us, b_us, gran in rows:
            s = Interval(
                earliest_us=e_us,
                latest_us=l_us,
                best_us=b_us,
                granularity=gran,
            )
            sc = score_pair(qi, s, mode="jaccard_composite")
            if sc > best_per_doc.get(doc_id, 0.0):
                best_per_doc[doc_id] = sc
        for d, sc in best_per_doc.items():
            out_lists[d].append(sc)
    out: dict[str, float] = {}
    for d, scores in out_lists.items():
        out[d] = aggregate_pair_scores(scores, mode="sum")
    return out


def _apply_bracket_quarter(
    by_id: dict[str, list[TimeExpression]],
) -> dict[str, list[TimeExpression]]:
    out: dict[str, list[TimeExpression]] = {}
    for iid, tes in by_id.items():
        widened = []
        for te in tes:
            te_c = copy.deepcopy(te)
            widened.append(apply_bracket_mode(te_c, "quarter"))
        out[iid] = widened
    return out


def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def _mrr(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return float("nan")
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def _avg(vals: list[float]) -> float:
    vs = [v for v in vals if not math.isnan(v)]
    return sum(vs) / len(vs) if vs else 0.0


def _retrieval_metrics(
    predicted_by_doc: dict[str, list[TimeExpression]],
    predicted_by_q: dict[str, list[TimeExpression]],
    queries: list[dict],
    gold: dict[str, set[str]],
    db_path: Path,
) -> dict[str, float]:
    docs_w = _apply_bracket_quarter(predicted_by_doc)
    q_w = _apply_bracket_quarter(predicted_by_q)
    store = _build_index(docs_w, db_path)
    rec5s, rec10s, mrrs, ndcgs = [], [], [], []
    for q in queries:
        qid = q["query_id"]
        relevant = gold.get(qid, set())
        if not relevant:
            continue
        q_preds = q_w.get(qid, [])
        scores = _temporal_retrieve(store, q_preds)
        ranked = [
            d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
        rec5s.append(_recall_at_k(ranked, relevant, 5))
        rec10s.append(_recall_at_k(ranked, relevant, 10))
        mrrs.append(_mrr(ranked, relevant))
        ndcgs.append(_ndcg_at_k(ranked, relevant, 10))
    return {
        "recall@5": _avg(rec5s),
        "recall@10": _avg(rec10s),
        "mrr": _avg(mrrs),
        "ndcg@10": _avg(ndcgs),
        "n_queries_scored": len(rec5s),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def run_version(
    name: str,
    extract_fn,
    doc_items,
    q_items,
    docs,
    queries,
    gold: dict[str, set[str]],
    failure_cases: list[dict],
    model_for_cost: str,
) -> dict[str, Any]:
    print(f"\n=== Running {name} ===")
    doc_pred, doc_usage = await extract_fn(doc_items)
    q_pred, q_usage = await extract_fn(q_items)
    total_usage = {
        "input": doc_usage.get("input", 0) + q_usage.get("input", 0),
        "output": doc_usage.get("output", 0) + q_usage.get("output", 0),
    }
    cost = _cost(total_usage, model_for_cost)

    overall, docs_m, queries_m = _combined_metrics(docs, queries, doc_pred, q_pred)
    fc_stats = _failure_recall(failure_cases, docs, queries, doc_pred, q_pred)
    retrieval = _retrieval_metrics(
        doc_pred,
        q_pred,
        queries,
        gold,
        CACHE_DIR / f"intervals_{name}.sqlite",
    )

    return {
        "name": name,
        "model": model_for_cost,
        "usage": total_usage,
        "cost_usd": cost,
        "extraction": {
            "overall": overall,
            "docs": docs_m,
            "queries": queries_m,
        },
        "failure_cases": fc_stats,
        "retrieval": retrieval,
    }


async def run_gpt5_upper_bound(
    failure_cases: list[dict],
    docs: list[dict],
    queries: list[dict],
) -> dict[str, Any]:
    """Upper-bound: run v5's prompt with gpt-5 on the queries/docs in
    failure_cases (capped at 20 unique items)."""
    # Pick 20 unique (type, item_id) pairs from failure_cases.
    seen = set()
    sel: list[tuple[str, str]] = []
    for fc in failure_cases:
        k = (fc["type"], fc["item_id"])
        if k in seen:
            continue
        seen.add(k)
        sel.append(k)
        if len(sel) >= 20:
            break
    docs_by_id = {d["doc_id"]: d for d in docs}
    queries_by_id = {q["query_id"]: q for q in queries}
    items: list[tuple[str, str, datetime, str]] = []
    for t, iid in sel:
        if t == "doc":
            d = docs_by_id[iid]
            items.append((d["doc_id"], d["text"], parse_iso(d["ref_time"]), "doc"))
        else:
            q = queries_by_id[iid]
            items.append((q["query_id"], q["text"], parse_iso(q["ref_time"]), "query"))

    print(f"\n=== Running gpt-5 upper-bound on {len(items)} failure items ===")
    ex = ExtractorV5(model="gpt-5", cache_subdir="extractor_v5_gpt5")

    async def one(iid, text, ref, _t):
        try:
            return iid, await ex.extract(text, ref)
        except Exception as e:
            print(f"  upper-bound failed for {iid}: {e}")
            return iid, []

    pairs = await asyncio.gather(*(one(*t) for t in items))
    ex.cache.save()
    pred: dict[str, list[TimeExpression]] = dict(pairs)

    # Evaluate: for each failure_case whose item_id is in sel, check if
    # we recovered it.
    sel_ids = {iid for _, iid in sel}
    fc_subset = [fc for fc in failure_cases if fc["item_id"] in sel_ids]
    docs_by_id_full = {d["doc_id"]: d for d in docs}
    queries_by_id_full = {q["query_id"]: q for q in queries}

    # Build pred_by_doc / pred_by_q partials for fc_subset.
    pred_by_doc: dict[str, list[TimeExpression]] = {}
    pred_by_q: dict[str, list[TimeExpression]] = {}
    for iid, tes in pred.items():
        if iid in docs_by_id_full:
            pred_by_doc[iid] = tes
        else:
            pred_by_q[iid] = tes

    # For extraction F1 compute only on the 20 items (not corpus-wide).
    doc_subset = [d for d in docs if d["doc_id"] in pred_by_doc]
    q_subset = [q for q in queries if q["query_id"] in pred_by_q]
    overall, _docs_m, _q_m = _combined_metrics(
        doc_subset, q_subset, pred_by_doc, pred_by_q
    )

    fc_stats = _failure_recall(fc_subset, doc_subset, q_subset, pred_by_doc, pred_by_q)

    usage = ex.usage
    cost = _cost(usage, "gpt-5")
    return {
        "name": "gpt5_upper_bound",
        "model": "gpt-5",
        "n_items": len(items),
        "usage": usage,
        "cost_usd": cost,
        "subset_extraction": overall,
        "failure_cases": fc_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    base_docs = load_jsonl(DATA_DIR / "docs.jsonl")
    base_queries = load_jsonl(DATA_DIR / "queries.jsonl")
    base_gold = {
        r["query_id"]: set(r["relevant_doc_ids"])
        for r in load_jsonl(DATA_DIR / "gold.jsonl")
    }
    disc_docs = load_jsonl(DATA_DIR / "disc_docs.jsonl")
    disc_queries = load_jsonl(DATA_DIR / "disc_queries.jsonl")
    disc_gold = {
        r["query_id"]: set(r["relevant_doc_ids"])
        for r in load_jsonl(DATA_DIR / "disc_gold.jsonl")
    }
    docs = base_docs + disc_docs
    queries = base_queries + disc_queries
    gold = {**base_gold, **disc_gold}

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    print(f"Loaded {len(docs)} docs, {len(queries)} queries.")

    # Failure cases from v1 (already computed).
    fc_path = RESULTS_DIR / "failure_cases.json"
    if fc_path.exists():
        failure_cases = json.loads(fc_path.read_text())
    else:
        print("WARN: failure_cases.json missing; run v1 first.")
        failure_cases = []

    results: list[dict[str, Any]] = []

    # v1 (baseline)
    async def v1_fn(items):
        return await extract_v1(items)

    results.append(
        await run_version(
            "v1",
            v1_fn,
            doc_items,
            q_items,
            docs,
            queries,
            gold,
            failure_cases,
            "gpt-5-mini",
        )
    )

    # v2..v6
    versions = [
        ("v2", ExtractorV2),
        ("v3", ExtractorV3),
        ("v4", ExtractorV4),
        ("v5", ExtractorV5),
        ("v6", ExtractorV6),
    ]
    for name, cls in versions:

        async def _fn(items, cls=cls, name=name):
            return await extract_improved(cls, items, cache_subdir=f"extractor_{name}")

        results.append(
            await run_version(
                name,
                _fn,
                doc_items,
                q_items,
                docs,
                queries,
                gold,
                failure_cases,
                "gpt-5-mini",
            )
        )

    # gpt-5 upper bound
    upper = await run_gpt5_upper_bound(failure_cases, docs, queries)
    results.append(upper)

    # Dump JSON
    out_path = RESULTS_DIR / "extractor_improvements.json"
    with out_path.open("w") as f:
        json.dump(
            {"versions": results, "failure_cases_total": len(failure_cases)},
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote {out_path}")

    # Markdown
    md = ["# Extractor Improvements — Results\n\n"]
    md.append(f"Total v1-missed surfaces on corpus: **{len(failure_cases)}**.\n\n")
    md.append("## Per-version table\n\n")
    md.append(
        "| Version | Model | F1 | Precision | Recall | Failure recall | "
        "R@5 | R@10 | MRR | NDCG@10 | Input tok | Output tok | Cost |\n"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in results:
        if r["name"] == "gpt5_upper_bound":
            continue
        ext = r["extraction"]["overall"]
        ret = r["retrieval"]
        fc = r["failure_cases"]
        md.append(
            f"| {r['name']} | {r['model']} | {ext['f1']:.3f} | "
            f"{ext['precision']:.3f} | {ext['recall']:.3f} | "
            f"{fc['recovered']}/{fc['total']} ({fc['recall']:.2f}) | "
            f"{ret['recall@5']:.3f} | {ret['recall@10']:.3f} | "
            f"{ret['mrr']:.3f} | {ret['ndcg@10']:.3f} | "
            f"{r['usage']['input']:,} | {r['usage']['output']:,} | "
            f"${r['cost_usd']:.4f} |\n"
        )
    # gpt5 row
    if upper:
        fc = upper["failure_cases"]
        ext = upper["subset_extraction"]
        md.append(
            f"| gpt5_upper ({upper['n_items']} items) | gpt-5 | "
            f"{ext['f1']:.3f} | {ext['precision']:.3f} | {ext['recall']:.3f} | "
            f"{fc['recovered']}/{fc['total']} ({fc['recall']:.2f}) | — | — | — | — | "
            f"{upper['usage']['input']:,} | {upper['usage']['output']:,} | "
            f"${upper['cost_usd']:.4f} |\n"
        )

    md.append("\n## Resolution MAE\n\n")
    md.append("| Version | Median (s) | Mean (s) | P95 (s) | Pairs |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for r in results:
        if r["name"] == "gpt5_upper_bound":
            continue
        ext = r["extraction"]["overall"]
        md.append(
            f"| {r['name']} | {ext['resolution_abs_err_median_s']:.0f} | "
            f"{ext['resolution_abs_err_mean_s']:.0f} | "
            f"{ext['resolution_abs_err_p95_s']:.0f} | "
            f"{ext['matched_pairs']} |\n"
        )

    md.append("\n## Failure-case recovery breakdown (by surface)\n\n")
    md.append("| Surface | v1 | v2 | v3 | v4 | v5 | v6 |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    surfaces: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for r in results:
        if r["name"] == "gpt5_upper_bound":
            continue
        for s, stats in r["failure_cases"].get("by_surface", {}).items():
            surfaces[s][r["name"]] = (stats["recovered"], stats["total"])
    for s, per_v in sorted(surfaces.items()):
        row = f"| {s!r} |"
        for vname in ("v1", "v2", "v3", "v4", "v5", "v6"):
            rec, tot = per_v.get(vname, (0, 0))
            row += f" {rec}/{tot} |"
        md.append(row + "\n")

    md.append("\n## Total cost\n\n")
    total_cost = sum(r.get("cost_usd", 0) for r in results) + (
        upper.get("cost_usd", 0) if upper else 0
    )
    md.append(
        f"- Sum across all versions (incl gpt-5 upper-bound): **${total_cost:.4f}**\n"
    )

    with (RESULTS_DIR / "extractor_improvements.md").open("w") as f:
        f.writelines(md)
    print(f"Wrote {RESULTS_DIR / 'extractor_improvements.md'}")


if __name__ == "__main__":
    asyncio.run(main())
