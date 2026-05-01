"""Orchestration: synth data -> extract -> index -> retrieve -> metrics.

Outputs:
- results/extraction_quality.json
- results/retrieval_results.json
- results/REPORT.md
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from baselines import embed_all, semantic_rank
from expander import expand
from extractor import Extractor
from schema import (
    TimeExpression,
    parse_iso,
    time_expression_from_dict,
    to_us,
)
from scorer import Interval, score_pair
from store import IntervalStore

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DB_PATH = ROOT / "cache" / "intervals.sqlite"

TOP_K = 10


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Extraction F1 & Resolution MAE
# ---------------------------------------------------------------------------
def _span_overlap_frac(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    a_len = max(1, a_end - a_start)
    b_len = max(1, b_end - b_start)
    return inter / min(a_len, b_len)


def match_expressions(
    predicted: list[TimeExpression],
    gold: list[TimeExpression],
    source_text: str,
    threshold: float = 0.5,
) -> tuple[
    list[tuple[TimeExpression, TimeExpression]],
    list[TimeExpression],
    list[TimeExpression],
]:
    """Match predicted to gold by span overlap >= threshold chars.

    Returns (matches, false_positives, false_negatives).
    """

    def locate(te: TimeExpression) -> tuple[int, int] | None:
        if te.span_start is not None and te.span_end is not None:
            return te.span_start, te.span_end
        idx = source_text.find(te.surface)
        if idx >= 0:
            return idx, idx + len(te.surface)
        return None

    g_positions = [locate(g) for g in gold]
    p_positions = [locate(p) for p in predicted]
    matched_g: set[int] = set()
    matches: list[tuple[TimeExpression, TimeExpression]] = []
    fps: list[TimeExpression] = []
    for i, p in enumerate(predicted):
        pp = p_positions[i]
        if pp is None:
            fps.append(p)
            continue
        best_j, best_ov = -1, 0.0
        for j, g in enumerate(gold):
            if j in matched_g:
                continue
            gp = g_positions[j]
            if gp is None:
                continue
            ov = _span_overlap_frac(pp[0], pp[1], gp[0], gp[1])
            if ov > best_ov:
                best_ov = ov
                best_j = j
        if best_j >= 0 and best_ov >= threshold:
            matches.append((p, gold[best_j]))
            matched_g.add(best_j)
        else:
            fps.append(p)
    fns = [gold[j] for j in range(len(gold)) if j not in matched_g]
    return matches, fps, fns


def _best_of(te: TimeExpression) -> datetime | None:
    if te.kind == "instant" and te.instant:
        return te.instant.best or te.instant.earliest
    if te.kind == "interval" and te.interval:
        return te.interval.start.best or te.interval.start.earliest
    if te.kind == "recurrence" and te.recurrence:
        return te.recurrence.dtstart.best or te.recurrence.dtstart.earliest
    return None


def extraction_metrics(
    per_item_matches: list[
        tuple[
            list[tuple[TimeExpression, TimeExpression]],
            list[TimeExpression],
            list[TimeExpression],
        ]
    ],
) -> dict[str, Any]:
    tp = sum(len(m) for m, _, _ in per_item_matches)
    fp = sum(len(f) for _, f, _ in per_item_matches)
    fn = sum(len(f) for _, _, f in per_item_matches)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Resolution MAE on matched expressions
    abs_errors_s: list[float] = []
    for matches, _, _ in per_item_matches:
        for pred, gold in matches:
            pb = _best_of(pred)
            gb = _best_of(gold)
            if pb is not None and gb is not None:
                abs_errors_s.append(abs((pb - gb).total_seconds()))
    if abs_errors_s:
        median = sorted(abs_errors_s)[len(abs_errors_s) // 2]
        p95 = sorted(abs_errors_s)[max(0, int(0.95 * len(abs_errors_s)) - 1)]
        mean = sum(abs_errors_s) / len(abs_errors_s)
    else:
        median = p95 = mean = 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "resolution_abs_err_mean_s": mean,
        "resolution_abs_err_median_s": median,
        "resolution_abs_err_p95_s": p95,
        "matched_pairs": len(abs_errors_s),
    }


# ---------------------------------------------------------------------------
# Build temporal index
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Temporal retrieval
# ---------------------------------------------------------------------------
def flatten_query_intervals(te: TimeExpression) -> list[Interval]:
    """Flatten predicted query expressions to Interval objects."""
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
        from schema import GRANULARITY_ORDER

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


def temporal_retrieve(
    store: IntervalStore,
    query_exprs: list[TimeExpression],
) -> dict[str, float]:
    """Return doc_id -> aggregate score."""
    out: dict[str, float] = defaultdict(float)
    q_ivs: list[Interval] = []
    for te in query_exprs:
        q_ivs.extend(flatten_query_intervals(te))
    for qi in q_ivs:
        rows = store.query_overlap(qi.earliest_us, qi.latest_us)
        # Dedupe per query-interval by (doc_id, iv_row): each stored row only
        # contributes once per query-interval.
        best_per_doc: dict[str, float] = {}
        for expr_id, doc_id, e_us, l_us, b_us, gran in rows:
            s = Interval(
                earliest_us=e_us,
                latest_us=l_us,
                best_us=b_us,
                granularity=gran,
            )
            sc = score_pair(qi, s)
            if sc > best_per_doc.get(doc_id, 0.0):
                best_per_doc[doc_id] = sc
        for d, sc in best_per_doc.items():
            out[d] += sc
    return dict(out)


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
# Orchestration
# ---------------------------------------------------------------------------
async def run() -> None:
    # 1. Ensure synth data exists
    if not (DATA_DIR / "docs.jsonl").exists():
        import synth_data

        synth_data.main()

    docs = load_jsonl(DATA_DIR / "docs.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    gold = {
        r["query_id"]: set(r["relevant_doc_ids"])
        for r in load_jsonl(DATA_DIR / "gold.jsonl")
    }
    critical_pairs = json.loads((DATA_DIR / "critical_pairs.json").read_text())

    print(f"Loaded {len(docs)} docs, {len(queries)} queries.")

    # 2. Extraction — docs
    ex = Extractor()

    async def extract_for(
        item_id: str, text: str, ref: datetime
    ) -> tuple[str, list[TimeExpression]]:
        try:
            tes = await ex.extract(text, ref)
        except Exception as e:
            print(f"  extract failed for {item_id}: {e}")
            tes = []
        return item_id, tes

    doc_tasks = [
        extract_for(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs
    ]
    query_tasks = [
        extract_for(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries
    ]

    print("Extracting temporal expressions from docs...")
    doc_results = await asyncio.gather(*doc_tasks)
    print("Extracting temporal expressions from queries...")
    query_results = await asyncio.gather(*query_tasks)
    ex.cache.save()

    predicted_by_doc: dict[str, list[TimeExpression]] = {i: t for i, t in doc_results}
    predicted_by_query: dict[str, list[TimeExpression]] = {
        i: t for i, t in query_results
    }

    print(f"Extractor usage: {ex.usage}")
    # gpt-5-mini pricing (Apr 2026): $0.25/M in, $2.00/M out (approx)
    cost_in = ex.usage["input"] * 0.25 / 1_000_000
    cost_out = ex.usage["output"] * 2.0 / 1_000_000
    total_cost = cost_in + cost_out

    # 3. Extraction metrics — vs gold
    doc_match_results = []
    for d in docs:
        pred = predicted_by_doc.get(d["doc_id"], [])
        gold_tes = [time_expression_from_dict(g) for g in d["gold_expressions"]]
        doc_match_results.append(match_expressions(pred, gold_tes, d["text"]))
    query_match_results = []
    for q in queries:
        pred = predicted_by_query.get(q["query_id"], [])
        gold_tes = [time_expression_from_dict(g) for g in q["gold_expressions"]]
        query_match_results.append(match_expressions(pred, gold_tes, q["text"]))

    all_match_results = doc_match_results + query_match_results
    ext_metrics = extraction_metrics(all_match_results)
    ext_metrics_docs = extraction_metrics(doc_match_results)
    ext_metrics_queries = extraction_metrics(query_match_results)

    with (RESULTS_DIR / "extraction_quality.json").open("w") as f:
        json.dump(
            {
                "overall": ext_metrics,
                "docs": ext_metrics_docs,
                "queries": ext_metrics_queries,
                "cost_usd": total_cost,
                "usage": ex.usage,
            },
            f,
            indent=2,
        )
    print(
        f"Extraction F1={ext_metrics['f1']:.3f} ({ext_metrics['tp']}tp/{ext_metrics['fp']}fp/{ext_metrics['fn']}fn)"
    )

    # 4. Build temporal index on predicted-docs
    print("Building temporal index from predicted expressions...")
    store = build_index(predicted_by_doc, DB_PATH)

    # 5. Run retrieval
    # Semantic embeddings
    print("Embedding docs and queries...")
    doc_texts = [d["text"] for d in docs]
    query_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    query_embs_arr = await embed_all(query_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    query_embs = {q["query_id"]: query_embs_arr[i] for i, q in enumerate(queries)}

    all_doc_ids = [d["doc_id"] for d in docs]

    def rank_semantic(qid: str) -> list[tuple[str, float]]:
        return semantic_rank(query_embs[qid], doc_embs)

    def rank_temporal(qid: str) -> list[tuple[str, float]]:
        q_preds = predicted_by_query.get(qid, [])
        scores = temporal_retrieve(store, q_preds)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def rank_hybrid(qid: str) -> list[tuple[str, float]]:
        # Temporal candidates top-K, then semantic rerank within.
        t_ranked = rank_temporal(qid)
        if not t_ranked:
            # No temporal signal -> fall back to semantic
            return rank_semantic(qid)
        # Take top-20 temporal cands (all if fewer).
        cand = [d for d, _ in t_ranked[:20]]
        sem_scores = {
            d: float(
                np.dot(query_embs[qid], doc_embs[d])
                / (
                    (np.linalg.norm(query_embs[qid]) * np.linalg.norm(doc_embs[d]))
                    or 1e-9
                )
            )
            for d in cand
        }
        return sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)

    # 6. Compute metrics per condition
    conditions = {"T": rank_temporal, "S": rank_semantic, "T_and_S": rank_hybrid}
    per_condition: dict[str, dict[str, Any]] = {}

    critical_doc_ids = {dp for dp, _ in critical_pairs}
    critical_query_ids = {qp for _, qp in critical_pairs}
    crit_map = dict(critical_pairs)  # doc_id -> q_id? actually list of (doc,q)
    crit_map = {q_id: doc_id for (doc_id, q_id) in critical_pairs}

    for name, fn in conditions.items():
        rec5s = []
        rec10s = []
        mrrs = []
        ndcgs = []
        per_query_scores: dict[str, Any] = {}
        crit_top1 = 0
        for q in queries:
            qid = q["query_id"]
            relevant = gold.get(qid, set())
            ranked_pairs = fn(qid)
            ranked = [d for d, _ in ranked_pairs]
            if qid in crit_map:
                want = crit_map[qid]
                if ranked and ranked[0] == want:
                    crit_top1 += 1
            if not relevant:
                continue  # skip no-gold queries for recall metrics
            rec5s.append(recall_at_k(ranked, relevant, 5))
            rec10s.append(recall_at_k(ranked, relevant, 10))
            mrrs.append(mrr(ranked, relevant))
            ndcgs.append(ndcg_at_k(ranked, relevant, 10))
            per_query_scores[qid] = {
                "top10": ranked[:10],
                "relevant": sorted(relevant),
            }
        per_condition[name] = {
            "recall@5": average(rec5s),
            "recall@10": average(rec10s),
            "mrr": average(mrrs),
            "ndcg@10": average(ndcgs),
            "critical_top1": crit_top1,
            "critical_total": len(critical_pairs),
        }

    with (RESULTS_DIR / "retrieval_results.json").open("w") as f:
        json.dump(per_condition, f, indent=2)

    # 7. REPORT.md
    lines = [
        "# Temporal Extraction + Retrieval — Results\n",
        "\n## Extraction quality\n",
        f"- Overall F1: **{ext_metrics['f1']:.3f}** "
        f"(precision {ext_metrics['precision']:.3f}, recall {ext_metrics['recall']:.3f})\n",
        f"- Docs F1: {ext_metrics_docs['f1']:.3f} "
        f"(tp={ext_metrics_docs['tp']}, fp={ext_metrics_docs['fp']}, fn={ext_metrics_docs['fn']})\n",
        f"- Queries F1: {ext_metrics_queries['f1']:.3f} "
        f"(tp={ext_metrics_queries['tp']}, fp={ext_metrics_queries['fp']}, fn={ext_metrics_queries['fn']})\n",
        f"- Resolution MAE on matched pairs ({ext_metrics['matched_pairs']} pairs): "
        f"mean={ext_metrics['resolution_abs_err_mean_s']:.0f}s, "
        f"median={ext_metrics['resolution_abs_err_median_s']:.0f}s, "
        f"p95={ext_metrics['resolution_abs_err_p95_s']:.0f}s\n",
        "\n## Retrieval\n",
        "| Condition | Recall@5 | Recall@10 | MRR | NDCG@10 | Critical top-1 |\n",
        "|-----------|---------:|----------:|----:|--------:|---------------:|\n",
    ]
    for name in ["T", "S", "T_and_S"]:
        m = per_condition[name]
        lines.append(
            f"| {name} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg@10']:.3f} | "
            f"{m['critical_top1']}/{m['critical_total']} |\n"
        )

    lines += [
        "\n## Cost\n",
        f"- Total LLM tokens: input={ex.usage['input']:,}, output={ex.usage['output']:,}\n",
        f"- Estimated LLM cost (gpt-5-mini @ $0.25/M in, $2.00/M out): ${total_cost:.4f}\n",
        "\n## Verdict\n",
    ]
    best_sem = per_condition["S"]
    verdict_parts = []
    for name in ["T", "T_and_S"]:
        m = per_condition[name]
        for metric_name in ["recall@5", "recall@10", "mrr", "ndcg@10"]:
            if m[metric_name] > best_sem[metric_name] + 0.001:
                verdict_parts.append(
                    f"{name} beats S on {metric_name} ({m[metric_name]:.3f} vs {best_sem[metric_name]:.3f})"
                )
    if verdict_parts:
        lines.append("Temporal structure helps retrieval. Wins:\n\n")
        for p in verdict_parts:
            lines.append(f"- {p}\n")
    else:
        lines.append("Temporal structure did not beat pure semantic on any metric.\n")

    with (RESULTS_DIR / "REPORT.md").open("w") as f:
        f.writelines(lines)

    print("Wrote results/REPORT.md")
    print(json.dumps(per_condition, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
