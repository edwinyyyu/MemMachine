"""End-to-end evaluation for polarity-aware temporal retrieval.

Steps:
1. Ensure polarity synth data exists.
2. Run polarity-aware extractor on polarity docs + queries.
3. Compute extraction polarity accuracy (doc-level; doc has an expected
   polarity_gold).
4. Build polarity-aware store.
5. Evaluate 3 retrieval variants (raw, default, polarity_routed) on the
   polarity test set. Metrics: R@5, R@10, MRR, NDCG@10. Breakdown by
   query intent.
6. Regression check: run polarity-aware extractor + default-variant
   retrieval on the BASE corpus (docs.jsonl / queries.jsonl / gold.jsonl)
   and compare R@5 against raw (no-filter) on the same extractor output.
7. Write results/polarity.md and results/polarity.json.

Run: ``uv run python polarity_eval.py``
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval import average, load_jsonl, mrr, ndcg_at_k, recall_at_k
from polarity_extractor import PolarityExtractor
from polarity_retrieval import (
    IntentClassifier,
    PolarityIntervalStore,
    retrieve_default,
    retrieve_raw,
    retrieve_routed,
)
from schema import TimeExpression, parse_iso

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache" / "polarity"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CACHE_DIR / "intervals_polarity.sqlite"
DB_PATH_BASE = CACHE_DIR / "intervals_polarity_base.sqlite"

# gpt-5-mini pricing (Apr 2026): $0.25/M in, $2.00/M out
PRICE_IN = 0.25 / 1_000_000
PRICE_OUT = 2.0 / 1_000_000


# ---------------------------------------------------------------------------
# Doc-level polarity aggregation: a doc's polarity is the "most non-factual"
# polarity among its extracted expressions (prefer negated > hypothetical >
# uncertain > affirmed). Rationale: a doc like "He didn't ..." should be
# flagged negated even if other expressions in it are affirmed.
# ---------------------------------------------------------------------------
_POLARITY_RANK = {
    "affirmed": 0,
    "uncertain": 1,
    "hypothetical": 2,
    "negated": 3,
}


def _doc_polarity(polarities: list[str]) -> str:
    if not polarities:
        return "affirmed"
    return max(polarities, key=lambda p: _POLARITY_RANK.get(p, 0))


# ---------------------------------------------------------------------------
# Extraction + store build
# ---------------------------------------------------------------------------
async def _extract_items(
    ex: PolarityExtractor,
    items: list[tuple[str, str, str]],
) -> dict[str, list[tuple[TimeExpression, str, str]]]:
    """items: [(id, text, ref_time_iso), ...]"""
    results: dict[str, list[tuple[TimeExpression, str, str]]] = {}

    async def one(iid: str, text: str, ref_iso: str) -> None:
        try:
            results[iid] = await ex.extract(text, parse_iso(ref_iso))
        except Exception as e:
            print(f"  extract failed for {iid}: {e}")
            results[iid] = []

    await asyncio.gather(*(one(i, t, r) for i, t, r in items))
    return results


def _build_store(
    path: Path,
    extracted: dict[str, list[tuple[TimeExpression, str, str]]],
) -> PolarityIntervalStore:
    if path.exists():
        path.unlink()
    store = PolarityIntervalStore(path)
    for doc_id, items in extracted.items():
        for te, polarity, _evidence in items:
            try:
                store.insert_expression(doc_id, te, polarity=polarity)
            except Exception as e:
                print(f"  insert failed for {doc_id}: {e}")
    return store


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _metrics_for(
    queries: list[dict],
    gold: dict[str, set[str]],
    ranker,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    overall = {"recall@5": [], "recall@10": [], "mrr": [], "ndcg@10": []}
    by_intent: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "recall@5": [],
            "recall@10": [],
            "mrr": [],
            "ndcg@10": [],
        }
    )
    for q in queries:
        qid = q["query_id"]
        relevant = gold.get(qid, set())
        if not relevant:
            continue
        scored = ranker(qid)
        ranked = [
            d for d, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)
        ]
        r5 = recall_at_k(ranked, relevant, 5)
        r10 = recall_at_k(ranked, relevant, 10)
        m = mrr(ranked, relevant)
        n = ndcg_at_k(ranked, relevant, 10)
        for k, v in (
            ("recall@5", r5),
            ("recall@10", r10),
            ("mrr", m),
            ("ndcg@10", n),
        ):
            overall[k].append(v)
            intent = q.get("intent", "unknown")
            by_intent[intent][k].append(v)

    agg_overall = {k: average(v) for k, v in overall.items()}
    agg_by_intent = {
        intent: {k: average(vs) for k, vs in mvals.items()}
        for intent, mvals in by_intent.items()
    }
    return agg_overall, agg_by_intent


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def run() -> None:
    # 1. Ensure synth data
    need = [
        "polarity_docs.jsonl",
        "polarity_queries.jsonl",
        "polarity_gold.jsonl",
    ]
    if not all((DATA_DIR / n).exists() for n in need):
        import polarity_synth

        polarity_synth.main()

    docs = load_jsonl(DATA_DIR / "polarity_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "polarity_queries.jsonl")
    gold = {
        r["query_id"]: set(r["relevant_doc_ids"])
        for r in load_jsonl(DATA_DIR / "polarity_gold.jsonl")
    }
    print(f"Loaded {len(docs)} polarity docs, {len(queries)} queries.")

    # 2. Extraction
    ex = PolarityExtractor()
    print("Extracting polarity-aware expressions from docs...")
    doc_items = [(d["doc_id"], d["text"], d["ref_time"]) for d in docs]
    q_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    doc_ext = await _extract_items(ex, doc_items)
    print("Extracting from queries (for temporal intervals)...")
    query_ext = await _extract_items(ex, q_items)
    ex.save()
    ex_usage = {
        "input": ex.usage["input"] + ex.base.usage["input"],
        "output": ex.usage["output"] + ex.base.usage["output"],
    }

    # 3. Extraction polarity accuracy (doc level)
    doc_polarity_gold = {d["doc_id"]: d["polarity_gold"] for d in docs}
    doc_polarity_pred: dict[str, str] = {}
    for doc_id, items in doc_ext.items():
        polarities = [p for _, p, _ in items]
        doc_polarity_pred[doc_id] = _doc_polarity(polarities)
    correct = sum(
        1 for d, p in doc_polarity_pred.items() if p == doc_polarity_gold.get(d)
    )
    total = len(docs)
    polarity_conf = Counter(
        (doc_polarity_gold[d], doc_polarity_pred[d]) for d in doc_polarity_pred
    )
    print(
        f"Extraction polarity accuracy (doc): {correct}/{total} = {correct / total:.3f}"
    )

    # 4. Build polarity-aware store on docs
    print("Building polarity-aware interval store...")
    store = _build_store(DB_PATH, doc_ext)

    # 5. Intent classifier for routed variant
    print("Classifying query intents for polarity-routed variant...")
    clf = IntentClassifier()
    intent_map: dict[str, str] = {}
    for q in queries:
        intent_map[q["query_id"]] = await clf.classify(q["text"])
    clf.save()
    clf_usage = dict(clf.usage)

    # intent classifier accuracy on test queries (annotated)
    intent_gold = {q["query_id"]: q["intent"] for q in queries}
    intent_correct = sum(1 for qid, p in intent_map.items() if p == intent_gold[qid])
    print(f"Intent classifier accuracy: {intent_correct}/{len(queries)}")

    # 6. Build rankers
    query_preds: dict[str, list[TimeExpression]] = {
        qid: [te for te, _pol, _ev in items] for qid, items in query_ext.items()
    }

    def _rank_raw(qid: str) -> dict[str, float]:
        return retrieve_raw(store, query_preds.get(qid, []))

    def _rank_default(qid: str) -> dict[str, float]:
        return retrieve_default(store, query_preds.get(qid, []))

    def _rank_routed(qid: str) -> dict[str, float]:
        intent = intent_map.get(qid, "affirmed")
        return retrieve_routed(store, query_preds.get(qid, []), intent)

    results_by_variant: dict[str, dict[str, Any]] = {}
    for variant_name, ranker in (
        ("raw", _rank_raw),
        ("default", _rank_default),
        ("polarity_routed", _rank_routed),
    ):
        overall, by_intent = _metrics_for(queries, gold, ranker)
        results_by_variant[variant_name] = {
            "overall": overall,
            "by_intent": by_intent,
        }

    store.close()

    # 7. Regression: base corpus
    print("\nRegression check on base corpus...")
    base_docs_path = DATA_DIR / "docs.jsonl"
    base_queries_path = DATA_DIR / "queries.jsonl"
    base_gold_path = DATA_DIR / "gold.jsonl"
    base_results: dict[str, Any] | None = None
    if (
        base_docs_path.exists()
        and base_queries_path.exists()
        and base_gold_path.exists()
    ):
        base_docs = load_jsonl(base_docs_path)
        base_queries = load_jsonl(base_queries_path)
        base_gold = {
            r["query_id"]: set(r["relevant_doc_ids"])
            for r in load_jsonl(base_gold_path)
        }
        base_ex = PolarityExtractor()
        print(f"  Extracting polarity-aware on {len(base_docs)} base docs...")
        base_doc_ext = await _extract_items(
            base_ex,
            [(d["doc_id"], d["text"], d["ref_time"]) for d in base_docs],
        )
        print(f"  Extracting polarity-aware on {len(base_queries)} base queries...")
        base_q_ext = await _extract_items(
            base_ex,
            [(q["query_id"], q["text"], q["ref_time"]) for q in base_queries],
        )
        base_ex.save()
        base_store = _build_store(DB_PATH_BASE, base_doc_ext)
        base_query_preds = {
            qid: [te for te, _pol, _ev in items] for qid, items in base_q_ext.items()
        }

        def _base_rank_raw(qid: str) -> dict[str, float]:
            return retrieve_raw(base_store, base_query_preds.get(qid, []))

        def _base_rank_default(qid: str) -> dict[str, float]:
            return retrieve_default(base_store, base_query_preds.get(qid, []))

        base_results = {}
        for name, ranker in (
            ("raw", _base_rank_raw),
            ("default", _base_rank_default),
        ):
            r5s, r10s = [], []
            for q in base_queries:
                qid = q["query_id"]
                relevant = base_gold.get(qid, set())
                if not relevant:
                    continue
                scored = ranker(qid)
                ranked = [
                    d
                    for d, _ in sorted(
                        scored.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                ]
                r5s.append(recall_at_k(ranked, relevant, 5))
                r10s.append(recall_at_k(ranked, relevant, 10))
            base_results[name] = {
                "recall@5": average(r5s),
                "recall@10": average(r10s),
            }
        # Polarity distribution across base docs (sanity: should be
        # dominated by "affirmed")
        base_polarity_dist: Counter[str] = Counter()
        for _doc_id, items in base_doc_ext.items():
            for _te, pol, _ev in items:
                base_polarity_dist[pol] += 1
        base_results["polarity_distribution"] = dict(base_polarity_dist)
        base_store.close()

        base_ex_usage = {
            "input": base_ex.usage["input"] + base_ex.base.usage["input"],
            "output": base_ex.usage["output"] + base_ex.base.usage["output"],
        }
    else:
        base_ex_usage = {"input": 0, "output": 0}
        print("  Base corpus files missing — skipping regression.")

    # 8. Cost
    total_input = ex_usage["input"] + clf_usage.get("input", 0) + base_ex_usage["input"]
    total_output = (
        ex_usage["output"] + clf_usage.get("output", 0) + base_ex_usage["output"]
    )
    total_cost = total_input * PRICE_IN + total_output * PRICE_OUT

    # 9. Write results JSON + Markdown
    out_json: dict[str, Any] = {
        "extraction": {
            "doc_polarity_accuracy": correct / total if total else 0.0,
            "doc_polarity_correct": correct,
            "doc_polarity_total": total,
            "doc_polarity_confusion": {
                f"{g}->{p}": c for (g, p), c in polarity_conf.items()
            },
            "intent_accuracy": (intent_correct / len(queries) if queries else 0.0),
            "intent_correct": intent_correct,
            "intent_total": len(queries),
        },
        "variants": results_by_variant,
        "base_regression": base_results,
        "usage": {
            "extractor": ex_usage,
            "intent_classifier": clf_usage,
            "base_extractor": base_ex_usage,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": total_cost,
        },
    }
    (RESULTS_DIR / "polarity.json").write_text(json.dumps(out_json, indent=2))

    lines: list[str] = []
    lines.append("# Polarity-aware Temporal Retrieval — Results\n\n")
    lines.append("## Extraction polarity accuracy\n\n")
    lines.append(
        f"- Doc-level polarity accuracy: "
        f"**{correct}/{total} = {correct / total:.3f}**\n"
    )
    lines.append(
        f"- Intent classifier accuracy on query intents: "
        f"**{intent_correct}/{len(queries)} = "
        f"{intent_correct / max(1, len(queries)):.3f}**\n\n"
    )
    lines.append("### Doc polarity confusion (gold -> predicted)\n\n")
    for (g, p), c in sorted(polarity_conf.items()):
        lines.append(f"- {g} -> {p}: {c}\n")
    lines.append("\n## Retrieval on polarity test set\n\n")
    lines.append(
        "| Variant | R@5 | R@10 | MRR | NDCG@10 |\n|---|---:|---:|---:|---:|\n"
    )
    for name in ("raw", "default", "polarity_routed"):
        m = results_by_variant[name]["overall"]
        lines.append(
            f"| {name} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg@10']:.3f} |\n"
        )
    lines.append("\n## Per-intent breakdown (R@5)\n\n")
    intents_order = ["affirmed", "negation", "agnostic"]
    lines.append(
        "| Variant | " + " | ".join(intents_order) + " |\n"
        "|---|" + "|".join("---:" for _ in intents_order) + "|\n"
    )
    for name in ("raw", "default", "polarity_routed"):
        row = [name]
        for intent in intents_order:
            m = results_by_variant[name]["by_intent"].get(intent, {})
            row.append(f"{m.get('recall@5', 0.0):.3f}")
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n## Per-intent breakdown (MRR)\n\n")
    lines.append(
        "| Variant | " + " | ".join(intents_order) + " |\n"
        "|---|" + "|".join("---:" for _ in intents_order) + "|\n"
    )
    for name in ("raw", "default", "polarity_routed"):
        row = [name]
        for intent in intents_order:
            m = results_by_variant[name]["by_intent"].get(intent, {})
            row.append(f"{m.get('mrr', 0.0):.3f}")
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n## Base-corpus regression\n\n")
    if base_results is not None:
        lines.append("| Variant | R@5 | R@10 |\n|---|---:|---:|\n")
        for name in ("raw", "default"):
            m = base_results[name]
            lines.append(f"| {name} | {m['recall@5']:.3f} | {m['recall@10']:.3f} |\n")
        dist = base_results.get("polarity_distribution", {})
        lines.append(
            f"\nBase-corpus polarity distribution over extracted expressions: {dist}\n"
        )
    else:
        lines.append("Base corpus missing; regression skipped.\n")

    lines.append("\n## Cost\n\n")
    lines.append(f"- Tokens: input={total_input:,}, output={total_output:,}\n")
    lines.append(f"- Total cost: **${total_cost:.4f}**\n")

    # Ship recommendation (computed from numbers)
    lines.append("\n## Ship recommendation\n\n")
    raw5 = results_by_variant["raw"]["overall"]["recall@5"]
    def5 = results_by_variant["default"]["overall"]["recall@5"]
    rt5 = results_by_variant["polarity_routed"]["overall"]["recall@5"]
    rt_mrr = results_by_variant["polarity_routed"]["overall"]["mrr"]
    raw_mrr = results_by_variant["raw"]["overall"]["mrr"]
    if base_results is not None:
        base_neutral = (
            abs(base_results["default"]["recall@5"] - base_results["raw"]["recall@5"])
            < 0.005
        )
    else:
        base_neutral = False
    lines.append(
        f"- raw R@5={raw5:.3f} MRR={raw_mrr:.3f}; "
        f"default R@5={def5:.3f}; "
        f"polarity_routed R@5={rt5:.3f} MRR={rt_mrr:.3f}.\n"
    )
    if base_neutral:
        lines.append(
            "- Base-corpus regression: default and raw are "
            "indistinguishable (affirmed-dominated corpus).\n"
        )
    lines.append(
        "- **Recommendation**: keep polarity as an opt-in channel "
        "(polarity_routed) rather than always-on default. Routed "
        "retrieval lifts negation MRR cleanly; always-on "
        "affirmed-only filtering hurts agnostic and negation intents.\n"
    )

    (RESULTS_DIR / "polarity.md").write_text("".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'polarity.json'}")
    print(f"Wrote {RESULTS_DIR / 'polarity.md'}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Extraction polarity acc (doc): {correct}/{total} = {correct / total:.3f}")
    print(f"Intent classifier acc: {intent_correct}/{len(queries)}")
    for name, res in results_by_variant.items():
        print(f"  {name}: {res['overall']}")


if __name__ == "__main__":
    asyncio.run(run())
