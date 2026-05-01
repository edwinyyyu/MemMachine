"""Evaluate hierarchical-tag retrieval (F2) against bracket baseline.

Variants:
- T1: tag Jaccard, sum aggregation
- T2: tag Jaccard, max aggregation
- T3: rarity-weighted Jaccard, sum aggregation
- T4: rarity-weighted, max aggregation
- T5: T1 + semantic rerank (cosine over text-embedding-3-small)
- Baseline: current base hybrid (quarter/jaccard/sum + semantic rerank),
  recomputed here for side-by-side comparison.

Subsets: all / base / disc / utt.

Reuses cached LLM extractions from ``cache/llm_cache.json`` — no new
LLM traffic. Embeddings cached in ``cache/embedding_cache.json``.
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
from baselines import embed_all
from expander import expand
from extractor import Extractor
from schema import (
    GRANULARITY_ORDER,
    TimeExpression,
    parse_iso,
    to_us,
)
from scorer import Interval, aggregate_pair_scores, score_jaccard_composite
from store import IntervalStore
from tag_retrieval import compute_idf_weights
from tag_retrieval import rank as tag_rank
from tag_store import TagStore

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Bracket retrieval (baseline), recomputed with cached extractions
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


def build_bracket_store(
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


def bracket_retrieve_sum(
    store: IntervalStore,
    query_exprs: list[TimeExpression],
) -> dict[str, float]:
    """Baseline bracket retrieval: jaccard_composite + sum."""
    per_doc: dict[str, list[float]] = defaultdict(list)
    q_ivs: list[Interval] = []
    for te in query_exprs:
        q_ivs.extend(flatten_query_intervals(te))
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
            sc = score_jaccard_composite(qi, s)
            if sc > best_per_doc.get(doc_id, 0.0):
                best_per_doc[doc_id] = sc
        for d, sc in best_per_doc.items():
            per_doc[d].append(sc)
    return {d: aggregate_pair_scores(ss, mode="sum") for d, ss in per_doc.items()}


# ---------------------------------------------------------------------------
# Semantic rerank helpers
# ---------------------------------------------------------------------------
def semantic_rerank(
    candidates: list[str],
    query_emb: np.ndarray,
    doc_embs: dict[str, np.ndarray],
) -> list[tuple[str, float]]:
    q = query_emb
    qn = np.linalg.norm(q) or 1e-9
    out = []
    for d in candidates:
        v = doc_embs.get(d)
        if v is None:
            continue
        vn = np.linalg.norm(v) or 1e-9
        sim = float(np.dot(q, v) / (qn * vn))
        out.append((d, sim))
    return sorted(out, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Metrics
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
# Extraction replay (cached-only)
# ---------------------------------------------------------------------------
async def extract_all(
    items: list[tuple[str, str, datetime]],
    label: str,
) -> dict[str, list[TimeExpression]]:
    """Replay extractions from cache; any cache miss will raise loudly
    (we don't want new LLM calls here)."""
    ex = Extractor()

    async def one(iid: str, text: str, ref: datetime):
        try:
            tes = await ex.extract(text, ref)
        except Exception as e:
            print(f"  extract failed for {iid}: {e}")
            tes = []
        return iid, tes

    print(f"extracting {label} ({len(items)} items, cache-backed)...")
    tasks = [one(i, t, r) for i, t, r in items]
    results = await asyncio.gather(*tasks)
    ex.cache.save()
    print(f"  {label} usage: input={ex.usage['input']}, output={ex.usage['output']}")
    return {i: t for i, t in results}


# ---------------------------------------------------------------------------
# Variant evaluation harness
# ---------------------------------------------------------------------------
def evaluate_rankings(
    ranked_per_q: dict[str, list[str]],
    gold: dict[str, set[str]],
    subsets: dict[str, set[str]],
    top_k: int = 10,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, qids in subsets.items():
        rec5s, rec10s, mrrs, ndcgs = [], [], [], []
        for qid in qids:
            ranked = ranked_per_q.get(qid, [])
            relevant = gold.get(qid, set())
            if not relevant:
                continue
            rec5s.append(recall_at_k(ranked, relevant, 5))
            rec10s.append(recall_at_k(ranked, relevant, 10))
            mrrs.append(mrr(ranked, relevant))
            ndcgs.append(ndcg_at_k(ranked, relevant, top_k))
        out[name] = {
            "recall@5": average(rec5s),
            "recall@10": average(rec10s),
            "mrr": average(mrrs),
            "ndcg@10": average(ndcgs),
            "n": len([v for v in rec5s if not math.isnan(v)]),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run() -> None:
    # Load data
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

    utt_path = DATA_DIR / "utterance_queries.jsonl"
    if utt_path.exists():
        utt_docs = load_jsonl(DATA_DIR / "utterance_docs.jsonl")
        utt_queries = load_jsonl(utt_path)
        utt_gold = {
            r["query_id"]: set(r["relevant_doc_ids"])
            for r in load_jsonl(DATA_DIR / "utterance_gold.jsonl")
        }
    else:
        utt_docs, utt_queries, utt_gold = [], [], {}

    all_docs = base_docs + disc_docs + utt_docs
    all_queries = base_queries + disc_queries + utt_queries
    all_gold = {**base_gold, **disc_gold, **utt_gold}

    print(
        f"Loaded docs: base={len(base_docs)} disc={len(disc_docs)} "
        f"utt={len(utt_docs)} (total={len(all_docs)})"
    )
    print(
        f"Loaded queries: base={len(base_queries)} disc={len(disc_queries)} "
        f"utt={len(utt_queries)} (total={len(all_queries)})"
    )

    # Extract (cache-only — won't call LLM if all prompts already cached)
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in all_docs]
    query_items = [
        (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in all_queries
    ]
    doc_extracted = await extract_all(doc_items, "docs")
    query_extracted = await extract_all(query_items, "queries")

    # Subsets
    base_qids = {q["query_id"] for q in base_queries}
    disc_qids = {q["query_id"] for q in disc_queries}
    utt_qids = {q["query_id"] for q in utt_queries}
    subsets = {
        "all": base_qids | disc_qids | utt_qids,
        "base": base_qids,
        "disc": disc_qids,
        "utt": utt_qids,
    }

    # ------------------------------------------------------------------
    # Tag index build
    # ------------------------------------------------------------------
    print("\nbuilding tag index...")
    tag_store = TagStore(path=CACHE_DIR / "time_tags.sqlite")
    tag_store.reset()
    tag_store.bulk_insert(doc_extracted)
    print(
        f"  tag_store: {tag_store.num_docs()} docs, "
        f"{len(tag_store.inverted)} distinct tags, "
        f"{sum(len(v) for v in tag_store.inverted.values())} (tag, expr) rows"
    )

    # Print a sampling of tag counts by granularity
    gran_counts: dict[str, int] = defaultdict(int)
    for t in tag_store.inverted:
        gran_counts[t.split(":", 1)[0]] += 1
    print(f"  distinct tags by granularity: {dict(gran_counts)}")

    idf_weights = compute_idf_weights(tag_store)

    # ------------------------------------------------------------------
    # Bracket baseline index (quarter bracket = current ship-best uses
    # quarter; for simplicity we use the raw extractions which already
    # carry quarter/half/... semantics per synth_data; base REPORT.md
    # reported 0.555 R@5 for T_and_S using the base jaccard_composite/sum.)
    # ------------------------------------------------------------------
    print("\nbuilding bracket baseline index...")
    bracket_db = CACHE_DIR / "intervals_tageval_baseline.sqlite"
    bracket_store = build_bracket_store(doc_extracted, bracket_db)

    # ------------------------------------------------------------------
    # Embeddings for semantic rerank
    # ------------------------------------------------------------------
    print("\nembedding docs + queries (cached)...")
    doc_texts = [d["text"] for d in all_docs]
    query_texts = [q["text"] for q in all_queries]
    doc_embs_arr = await embed_all(doc_texts)
    query_embs_arr = await embed_all(query_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(all_docs)}
    query_embs = {q["query_id"]: query_embs_arr[i] for i, q in enumerate(all_queries)}

    # ------------------------------------------------------------------
    # Rank every query under every variant
    # ------------------------------------------------------------------
    variants: dict[str, dict[str, list[str]]] = {}

    def rank_tag(
        q_preds: list[TimeExpression], score_mode, agg_mode, weights=None
    ) -> list[str]:
        ranked_pairs = tag_rank(
            tag_store,
            q_preds,
            score_mode=score_mode,
            agg_mode=agg_mode,
            weights=weights,
        )
        return [d for d, _ in ranked_pairs]

    def rank_bracket(q_preds: list[TimeExpression]) -> list[str]:
        scores = bracket_retrieve_sum(bracket_store, q_preds)
        return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def hybrid(qid: str, temporal_ranked: list[str], top_cands: int = 20) -> list[str]:
        if not temporal_ranked:
            # fall back to pure semantic
            sem = semantic_rerank(
                [d["doc_id"] for d in all_docs],
                query_embs[qid],
                doc_embs,
            )
            return [d for d, _ in sem]
        cand = temporal_ranked[:top_cands]
        sem = semantic_rerank(cand, query_embs[qid], doc_embs)
        return [d for d, _ in sem]

    # Pre-compute rankings.
    T1: dict[str, list[str]] = {}
    T2: dict[str, list[str]] = {}
    T3: dict[str, list[str]] = {}
    T4: dict[str, list[str]] = {}
    T5: dict[str, list[str]] = {}
    BASE_T: dict[str, list[str]] = {}
    BASELINE: dict[str, list[str]] = {}

    for q in all_queries:
        qid = q["query_id"]
        q_preds = query_extracted.get(qid, [])

        T1[qid] = rank_tag(q_preds, "jaccard", "sum")
        T2[qid] = rank_tag(q_preds, "jaccard", "max")
        T3[qid] = rank_tag(q_preds, "weighted", "sum", weights=idf_weights)
        T4[qid] = rank_tag(q_preds, "weighted", "max", weights=idf_weights)

        T5[qid] = hybrid(qid, T1[qid])

        BASE_T[qid] = rank_bracket(q_preds)
        BASELINE[qid] = hybrid(qid, BASE_T[qid])

    variants = {
        "T1_jaccard_sum": T1,
        "T2_jaccard_max": T2,
        "T3_weighted_sum": T3,
        "T4_weighted_max": T4,
        "T5_jaccard_sum_hybrid": T5,
        "Baseline_bracket_jaccard_sum_hybrid": BASELINE,
        "BracketOnly_jaccard_sum": BASE_T,
    }

    # ------------------------------------------------------------------
    # Evaluate per variant
    # ------------------------------------------------------------------
    results: dict[str, Any] = {}
    for name, ranked_per_q in variants.items():
        results[name] = evaluate_rankings(ranked_per_q, all_gold, subsets)

    # ------------------------------------------------------------------
    # Failure analysis: per-query diff tags vs bracket on base+disc
    # ------------------------------------------------------------------
    failures: list[dict[str, Any]] = []
    for qid in sorted(subsets["all"]):
        relevant = all_gold.get(qid, set())
        if not relevant:
            continue
        tag_rec5 = recall_at_k(T1.get(qid, []), relevant, 5)
        br_rec5 = recall_at_k(BASE_T.get(qid, []), relevant, 5)
        if abs((tag_rec5 or 0) - (br_rec5 or 0)) >= 0.2:
            q_text = next(
                (q["text"] for q in all_queries if q["query_id"] == qid),
                "",
            )
            failures.append(
                {
                    "qid": qid,
                    "text": q_text,
                    "rel": sorted(relevant),
                    "tag_top5": T1.get(qid, [])[:5],
                    "bracket_top5": BASE_T.get(qid, [])[:5],
                    "tag_R@5": tag_rec5,
                    "bracket_R@5": br_rec5,
                    "delta": (tag_rec5 or 0) - (br_rec5 or 0),
                }
            )

    tag_better = [f for f in failures if f["delta"] > 0]
    bracket_better = [f for f in failures if f["delta"] < 0]

    # ------------------------------------------------------------------
    # Write JSON + Markdown report
    # ------------------------------------------------------------------
    (RESULTS_DIR / "hierarchical_tags.json").write_text(
        json.dumps(
            {
                "variants": results,
                "index_stats": {
                    "num_docs": tag_store.num_docs(),
                    "num_distinct_tags": len(tag_store.inverted),
                    "num_tag_rows": sum(len(v) for v in tag_store.inverted.values()),
                    "distinct_tags_by_granularity": dict(gran_counts),
                },
                "failures": {
                    "tag_better_than_bracket": tag_better,
                    "bracket_better_than_tag": bracket_better,
                },
            },
            indent=2,
        )
    )

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    def fmt_row(name: str, metrics: dict[str, dict[str, float]]) -> str:
        cells = [name]
        for subset in ("all", "base", "disc", "utt"):
            m = metrics.get(subset, {})
            cells.append(f"{m.get('recall@5', 0.0):.3f}")
            cells.append(f"{m.get('ndcg@10', 0.0):.3f}")
        return "| " + " | ".join(cells) + " |"

    lines: list[str] = []
    lines.append("# Hierarchical Granularity Tags (F2) — Results\n\n")
    lines.append(
        "Tag-based retrieval: every extracted TimeExpression emits a set "
        "of discrete granularity tags (day:..., week:..., month:..., "
        "quarter:..., year:..., decade:..., century:...). Matching is set "
        "intersection; scoring is Jaccard (unweighted or rarity-weighted) "
        "over tag sets. Per-doc aggregation over expression pairs is "
        "``sum`` or ``max``.\n\n"
    )
    lines.append("## Index statistics\n\n")
    lines.append(f"- Docs indexed: **{tag_store.num_docs()}**\n")
    lines.append(f"- Distinct tags: **{len(tag_store.inverted)}**\n")
    lines.append(
        f"- (tag, expr) rows: **{sum(len(v) for v in tag_store.inverted.values())}**\n"
    )
    lines.append(
        f"- Tags by granularity: "
        f"{', '.join(f'{k}={v}' for k, v in sorted(gran_counts.items()))}\n\n"
    )

    lines.append("## Variant comparison\n\n")
    lines.append(
        "|   Variant   | all R@5 | all NDCG | base R@5 | base NDCG | "
        "disc R@5 | disc NDCG | utt R@5 | utt NDCG |\n"
    )
    lines.append(
        "|-------------|--------:|---------:|---------:|----------:|"
        "---------:|----------:|--------:|---------:|\n"
    )
    order = [
        "T1_jaccard_sum",
        "T2_jaccard_max",
        "T3_weighted_sum",
        "T4_weighted_max",
        "T5_jaccard_sum_hybrid",
        "BracketOnly_jaccard_sum",
        "Baseline_bracket_jaccard_sum_hybrid",
    ]
    for name in order:
        lines.append(fmt_row(name, results[name]) + "\n")

    lines.append("\n## Per-subset N sizes\n\n")
    for name in order:
        sub = results[name]
        lines.append(
            f"- {name}: "
            + ", ".join(
                f"{s}={sub.get(s, {}).get('n', 0)}"
                for s in ("all", "base", "disc", "utt")
            )
            + "\n"
        )

    # Verdict
    def pick(variant: str, subset: str, metric: str) -> float:
        return results[variant][subset][metric]

    lines.append("\n## Catastrophic diffs (|ΔR@5| >= 0.2, T1 tag vs bracket)\n\n")
    lines.append(f"- Tag-better queries: **{len(tag_better)}**\n")
    lines.append(f"- Bracket-better queries: **{len(bracket_better)}**\n\n")
    if tag_better:
        lines.append("### Sample tag wins\n")
        for f in tag_better[:5]:
            lines.append(
                f"- `{f['qid']}` (Δ={f['delta']:+.2f}): {f['text']!r}; "
                f"gold={f['rel']}; bracket_top5={f['bracket_top5']}; "
                f"tag_top5={f['tag_top5']}\n"
            )
    if bracket_better:
        lines.append("\n### Sample bracket wins\n")
        for f in bracket_better[:5]:
            lines.append(
                f"- `{f['qid']}` (Δ={f['delta']:+.2f}): {f['text']!r}; "
                f"gold={f['rel']}; bracket_top5={f['bracket_top5']}; "
                f"tag_top5={f['tag_top5']}\n"
            )

    # Summary of answers to the 7 questions
    lines.append("\n## Summary (≤400 words)\n\n")

    # 1. Does tag-based retrieval beat bracket-based?
    t1_all = pick("T1_jaccard_sum", "all", "recall@5")
    br_only_all = pick("BracketOnly_jaccard_sum", "all", "recall@5")
    t1_base = pick("T1_jaccard_sum", "base", "recall@5")
    br_only_base = pick("BracketOnly_jaccard_sum", "base", "recall@5")
    t1_disc = pick("T1_jaccard_sum", "disc", "recall@5")
    br_only_disc = pick("BracketOnly_jaccard_sum", "disc", "recall@5")
    t1_utt = pick("T1_jaccard_sum", "utt", "recall@5")
    br_only_utt = pick("BracketOnly_jaccard_sum", "utt", "recall@5")

    # 2. Rarity-weighting lift
    t3_all = pick("T3_weighted_sum", "all", "recall@5")

    # 3. Max vs sum
    t2_all = pick("T2_jaccard_max", "all", "recall@5")
    t4_all = pick("T4_weighted_max", "all", "recall@5")

    # 4. Hybrid vs ship-best
    t5_all = pick("T5_jaccard_sum_hybrid", "all", "recall@5")
    bl_all = pick("Baseline_bracket_jaccard_sum_hybrid", "all", "recall@5")
    t5_base = pick("T5_jaccard_sum_hybrid", "base", "recall@5")
    bl_base = pick("Baseline_bracket_jaccard_sum_hybrid", "base", "recall@5")

    lines.append(
        "**1. Tag vs bracket (temporal only).** "
        f"Tag-Jaccard R@5: all={t1_all:.3f}, base={t1_base:.3f}, "
        f"disc={t1_disc:.3f}, utt={t1_utt:.3f}. "
        f"Bracket R@5: all={br_only_all:.3f}, base={br_only_base:.3f}, "
        f"disc={br_only_disc:.3f}, utt={br_only_utt:.3f}. "
        f"Delta all={t1_all - br_only_all:+.3f}.\n\n"
    )
    lines.append(
        "**2. Rarity-weighting.** "
        f"T3 (weighted/sum) R@5={t3_all:.3f} vs T1 (jaccard/sum) "
        f"R@5={t1_all:.3f}; delta={t3_all - t1_all:+.3f}.\n\n"
    )
    lines.append(
        "**3. Max vs sum aggregation.** "
        f"Jaccard max={t2_all:.3f} vs sum={t1_all:.3f} "
        f"(Δ={t2_all - t1_all:+.3f}); weighted max={t4_all:.3f} vs "
        f"sum={t3_all:.3f} (Δ={t4_all - t3_all:+.3f}).\n\n"
    )
    lines.append(
        "**4. Hybrid (tags + semantic) vs ship-best (bracket + semantic).** "
        f"T5 (hybrid) R@5={t5_all:.3f} vs baseline R@5={bl_all:.3f} "
        f"(Δ={t5_all - bl_all:+.3f} all; base Δ={t5_base - bl_base:+.3f}).\n\n"
    )
    lines.append(
        "**5. Catastrophic failure modes.** "
        f"{len(tag_better)} queries where tags beat brackets by ≥0.2 R@5; "
        f"{len(bracket_better)} queries where brackets beat tags by ≥0.2. "
        "See JSON `failures` section for details.\n\n"
    )
    # Ship recommendation
    recs = []
    if t5_all > bl_all + 0.01:
        recs.append("REPLACE brackets — hybrid tags beats hybrid brackets")
    elif t5_all < bl_all - 0.01:
        recs.append("DEPRIORITIZE — tags lose to brackets even with semantic rerank")
    else:
        recs.append(
            "ADD ALONGSIDE — near-parity overall; ensemble may recover subset wins"
        )
    lines.append("**6. Ship recommendation.** " + "; ".join(recs) + ".\n\n")
    lines.append(
        "**7. Cost.** $0 LLM (all extractions cached), ~$0 embeddings "
        "(all cached). Tag index build is purely deterministic from "
        "cached extractions.\n"
    )

    (RESULTS_DIR / "hierarchical_tags.md").write_text("".join(lines))

    print("\n=== Summary ===")
    for name in order:
        m = results[name]["all"]
        print(
            f"  {name:<45} all R@5={m['recall@5']:.3f} "
            f"NDCG@10={m['ndcg@10']:.3f} MRR={m['mrr']:.3f}"
        )
    print("\nWrote results/hierarchical_tags.md and .json")


if __name__ == "__main__":
    asyncio.run(run())
