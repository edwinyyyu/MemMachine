"""R1 + R2 representation experiments — utterance anchors + dual-score
aggregation. See REPRESENTATION_EXPERIMENTS.md.

Runs:
- R1a anchor-only
- R1b referent-only (sanity-check vs current temporal-only baseline)
- R1c union (any overlap), score = max-of-(anchor, best-referent)
- R2a sum: anchor + Σ referents
- R2b max: max(anchor, best-referent)
- R2c weighted α=0.3 β=0.7 (prefer referents)
- R2d weighted α=0.7 β=0.3 (prefer anchors)
- R2e query-intent routed via gpt-5-mini

Subsets:
- all  (base 55 + disc 30 + utt 10)
- base (55)
- disc (30)
- utt  (10)
- utt_utterance (q_utt_0..4, intent=utterance)
- utt_referent  (q_utt_5..9, intent=referent)

Metrics: R@5, R@10, MRR, NDCG@10.

Also runs a hybrid (semantic rerank of temporal top-K) variant over the
best R1/R2 source+agg config to compare vs the documented base hybrid
(R@5 0.555).

Outputs:
- results/representation_r1r2.json
- results/representation_r1r2.md
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from anchor_retrieval import retrieve as anchor_retrieve
from anchor_store import UtteranceAnchorStore
from baselines import embed_all
from dotenv import load_dotenv
from extractor import Extractor
from openai import AsyncOpenAI
from schema import (
    GRANULARITY_ORDER,
    TimeExpression,
    parse_iso,
    to_us,
)
from scorer import Interval
from store import IntervalStore

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache"
ANCHOR_CACHE_DIR = CACHE_DIR / "anchor"
ANCHOR_CACHE_DIR.mkdir(exist_ok=True)
DB_PATH = CACHE_DIR / "temporal_anchor.db"
INTENT_CACHE_FILE = ANCHOR_CACHE_DIR / "intent_cache.json"

TOP_K = 10
INTENT_MODEL = "gpt-5-mini"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Flatten query TEs -> intervals (same algorithm used in eval.py)
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
    # Recurrence: skipped at query side for R1/R2 — not in utt queries.
    return out


def flatten_all(tes: list[TimeExpression]) -> list[Interval]:
    out: list[Interval] = []
    for te in tes:
        out.extend(flatten_query_intervals(te))
    return out


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
# Intent classifier (R2e)
# ---------------------------------------------------------------------------
class IntentClassifier:
    SYSTEM = (
        "Classify a user temporal-retrieval query into ONE of three intents:\n"
        '- "utterance": the query asks about when MESSAGES WERE WRITTEN / sent / '
        'recorded, even if mentioning a time range (e.g. "what did I write '
        '2 years ago?", "messages from 2024", "things I said last spring").\n'
        '- "referent": the query asks about when REFERENCED EVENTS happened, '
        'regardless of when they were recorded (e.g. "what happened in 1995?", '
        '"moon landing year", "when did the war end?").\n'
        '- "ambiguous": neither clearly dominates.\n'
        'Reply with strict JSON: {"intent": "utterance"|"referent"|"ambiguous"}.'
    )

    def __init__(self) -> None:
        self.client = AsyncOpenAI()
        self.cache: dict[str, str] = {}
        if INTENT_CACHE_FILE.exists():
            with INTENT_CACHE_FILE.open() as f:
                self.cache = json.load(f)
        self._new: dict[str, str] = {}
        self.usage = {"input": 0, "output": 0}
        self.sem = asyncio.Semaphore(8)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(f"{INTENT_MODEL}|{text}".encode()).hexdigest()

    async def classify(self, query_text: str) -> str:
        k = self._key(query_text)
        if k in self.cache:
            return self.cache[k]
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": f"Query: {query_text}\nReply with JSON."},
        ]
        async with self.sem:
            resp = await self.client.chat.completions.create(
                model=INTENT_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=800,
            )
        usage = resp.usage
        if usage:
            self.usage["input"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["output"] += getattr(usage, "completion_tokens", 0) or 0
        raw = resp.choices[0].message.content or "{}"
        try:
            intent = json.loads(raw).get("intent", "ambiguous")
        except json.JSONDecodeError:
            intent = "ambiguous"
        if intent not in ("utterance", "referent", "ambiguous"):
            intent = "ambiguous"
        self.cache[k] = intent
        self._new[k] = intent
        return intent

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, str] = {}
        if INTENT_CACHE_FILE.exists():
            with INTENT_CACHE_FILE.open() as f:
                existing = json.load(f)
        existing.update(self._new)
        tmp = INTENT_CACHE_FILE.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(existing, f, indent=2)
        tmp.replace(INTENT_CACHE_FILE)
        self._new.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run() -> None:
    base_docs = load_jsonl(DATA_DIR / "docs.jsonl")
    base_queries = load_jsonl(DATA_DIR / "queries.jsonl")
    base_gold_rows = load_jsonl(DATA_DIR / "gold.jsonl")
    base_gold = {r["query_id"]: set(r["relevant_doc_ids"]) for r in base_gold_rows}

    disc_docs = load_jsonl(DATA_DIR / "disc_docs.jsonl")
    disc_queries = load_jsonl(DATA_DIR / "disc_queries.jsonl")
    disc_gold_rows = load_jsonl(DATA_DIR / "disc_gold.jsonl")
    disc_gold = {r["query_id"]: set(r["relevant_doc_ids"]) for r in disc_gold_rows}

    utt_docs = load_jsonl(DATA_DIR / "utterance_docs.jsonl")
    utt_queries = load_jsonl(DATA_DIR / "utterance_queries.jsonl")
    utt_gold_rows = load_jsonl(DATA_DIR / "utterance_gold.jsonl")
    utt_gold = {r["query_id"]: set(r["relevant_doc_ids"]) for r in utt_gold_rows}

    all_docs = base_docs + disc_docs + utt_docs
    all_queries = base_queries + disc_queries + utt_queries
    all_gold = {**base_gold, **disc_gold, **utt_gold}

    print(
        f"Loaded docs: {len(base_docs)} base + {len(disc_docs)} disc + "
        f"{len(utt_docs)} utt = {len(all_docs)}"
    )
    print(
        f"Loaded queries: {len(base_queries)} base + {len(disc_queries)} disc + "
        f"{len(utt_queries)} utt = {len(all_queries)}"
    )

    # -------------------------------------------------------------------
    # Extraction (cached for base + disc already; utt may incur small cost)
    # -------------------------------------------------------------------
    ex = Extractor()

    async def extract_for(
        iid: str, text: str, ref: datetime
    ) -> tuple[str, list[TimeExpression]]:
        try:
            tes = await ex.extract(text, ref)
        except Exception as e:
            print(f"  extract failed for {iid}: {e}")
            tes = []
        return iid, tes

    doc_tasks = [
        extract_for(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in all_docs
    ]
    query_tasks = [
        extract_for(q["query_id"], q["text"], parse_iso(q["ref_time"]))
        for q in all_queries
    ]

    print("Extracting docs + queries (cached where possible)...")
    doc_results = await asyncio.gather(*doc_tasks)
    query_results = await asyncio.gather(*query_tasks)
    ex.cache.save()

    predicted_by_doc = {i: t for i, t in doc_results}
    predicted_by_query = {i: t for i, t in query_results}

    print(f"Extractor usage: {ex.usage}")
    # gpt-5-mini cost
    extract_cost = (
        ex.usage["input"] * 0.25 / 1_000_000 + ex.usage["output"] * 2.0 / 1_000_000
    )

    # -------------------------------------------------------------------
    # Build indexes
    # -------------------------------------------------------------------
    if DB_PATH.exists():
        DB_PATH.unlink()
    store = IntervalStore(DB_PATH)
    for doc_id, tes in predicted_by_doc.items():
        for te in tes:
            try:
                store.insert_expression(doc_id, te)
            except Exception as e:
                print(f"  insert failed for {doc_id}: {e}")

    astore = UtteranceAnchorStore(DB_PATH)
    # Use per-doc granularity if present in metadata, else "day".
    astore.reset()
    bulk_rows = []
    for d in all_docs:
        gran = d.get("granularity", "day")
        bulk_rows.append((d["doc_id"], parse_iso(d["ref_time"]), gran))
    astore.bulk_insert(bulk_rows)

    print(
        f"Indexes built. intervals: {len(store.all_doc_ids())} docs. "
        f"anchors: {len(astore.all_doc_ids())} docs."
    )

    # -------------------------------------------------------------------
    # Embeddings (for hybrid variant check)
    # -------------------------------------------------------------------
    print("Embedding docs and queries (cached)...")
    doc_texts = [d["text"] for d in all_docs]
    query_texts = [q["text"] for q in all_queries]
    doc_embs_arr = await embed_all(doc_texts)
    query_embs_arr = await embed_all(query_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(all_docs)}
    query_embs = {q["query_id"]: query_embs_arr[i] for i, q in enumerate(all_queries)}

    # -------------------------------------------------------------------
    # Intent classification for all queries (R2e)
    # -------------------------------------------------------------------
    ic = IntentClassifier()
    print("Classifying query intents (cached)...")
    intents: dict[str, str] = {}
    sem = asyncio.Semaphore(8)

    async def classify_one(qid: str, text: str) -> None:
        intents[qid] = await ic.classify(text)

    await asyncio.gather(*(classify_one(q["query_id"], q["text"]) for q in all_queries))
    ic.save()
    intent_cost = (
        ic.usage["input"] * 0.25 / 1_000_000 + ic.usage["output"] * 2.0 / 1_000_000
    )
    print(f"Intent classifier usage: {ic.usage}, est cost ${intent_cost:.4f}")

    # -------------------------------------------------------------------
    # Flatten all query intervals ONCE
    # -------------------------------------------------------------------
    query_intervals: dict[str, list[Interval]] = {}
    for q in all_queries:
        qid = q["query_id"]
        query_intervals[qid] = flatten_all(predicted_by_query.get(qid, []))

    # -------------------------------------------------------------------
    # Subsets
    # -------------------------------------------------------------------
    base_ids = {q["query_id"] for q in base_queries}
    disc_ids = {q["query_id"] for q in disc_queries}
    utt_ids = {q["query_id"] for q in utt_queries}
    utt_utterance_ids = {
        q["query_id"] for q in utt_queries if q.get("intent") == "utterance"
    }
    utt_referent_ids = {
        q["query_id"] for q in utt_queries if q.get("intent") == "referent"
    }

    subsets = {
        "all": {q["query_id"] for q in all_queries},
        "base": base_ids,
        "disc": disc_ids,
        "utt": utt_ids,
        "utt_utterance": utt_utterance_ids,
        "utt_referent": utt_referent_ids,
    }

    # -------------------------------------------------------------------
    # Variants
    # -------------------------------------------------------------------
    variants: list[dict[str, Any]] = [
        {"name": "R1a_anchor_only", "source": "anchor_only", "agg": "sum"},
        {"name": "R1b_referent_only", "source": "referent_only", "agg": "sum"},
        {"name": "R1c_union_max", "source": "union", "agg": "max"},
        {"name": "R2a_union_sum", "source": "union", "agg": "sum"},
        {"name": "R2b_union_max", "source": "union", "agg": "max"},
        {
            "name": "R2c_union_w_a03_b07",
            "source": "union",
            "agg": "weighted",
            "alpha": 0.3,
            "beta": 0.7,
        },
        {
            "name": "R2d_union_w_a07_b03",
            "source": "union",
            "agg": "weighted",
            "alpha": 0.7,
            "beta": 0.3,
        },
        {
            "name": "R2e_union_routed",
            "source": "union",
            "agg": "routed",
        },
        # Conservative: mostly-referent with small anchor bonus. Tests whether
        # smaller β preserves base parity while still lifting utt.
        {
            "name": "R2f_union_w_a09_b01",
            "source": "union",
            "agg": "weighted",
            "alpha": 0.9,
            "beta": 0.1,
        },
        # sum_weighted: use referent-sum (floods long docs but rewards
        # multi-hit referents) + tiny anchor bump.
        {
            "name": "R2g_union_sumw_a1_b03",
            "source": "union",
            "agg": "sum_weighted",
            "alpha": 1.0,
            "beta": 0.3,
        },
    ]

    # -------------------------------------------------------------------
    # Run each variant; compute metrics per subset
    # -------------------------------------------------------------------
    all_variant_results: dict[str, dict[str, Any]] = {}

    def rank_doc_ids_from_scores(scores: dict[str, float]) -> list[str]:
        return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    for v in variants:
        name = v["name"]
        ranked_per_q: dict[str, list[str]] = {}
        score_per_q: dict[str, dict[str, float]] = {}

        for q in all_queries:
            qid = q["query_id"]
            q_ivs = query_intervals[qid]
            kwargs: dict[str, Any] = {
                "source": v["source"],
                "agg": v["agg"],
            }
            if "alpha" in v:
                kwargs["alpha"] = v["alpha"]
            if "beta" in v:
                kwargs["beta"] = v["beta"]
            if v["agg"] == "routed":
                kwargs["intent"] = intents.get(qid, "ambiguous")
            scores = anchor_retrieve(store, astore, q_ivs, **kwargs)
            score_per_q[qid] = scores
            ranked_per_q[qid] = rank_doc_ids_from_scores(scores)

        subset_metrics: dict[str, dict[str, float]] = {}
        for sname, qids in subsets.items():
            rec5s, rec10s, mrrs, ndcgs = [], [], [], []
            for qid in qids:
                ranked = ranked_per_q.get(qid, [])
                rel = all_gold.get(qid, set())
                if not rel:
                    continue
                rec5s.append(recall_at_k(ranked, rel, 5))
                rec10s.append(recall_at_k(ranked, rel, 10))
                mrrs.append(mrr(ranked, rel))
                ndcgs.append(ndcg_at_k(ranked, rel, 10))
            subset_metrics[sname] = {
                "recall@5": average(rec5s),
                "recall@10": average(rec10s),
                "mrr": average(mrrs),
                "ndcg@10": average(ndcgs),
                "n": len(rec5s),
            }

        all_variant_results[name] = {
            "config": v,
            "subsets": subset_metrics,
        }
        r5 = subset_metrics["all"]["recall@5"]
        r5_utt = subset_metrics["utt"]["recall@5"]
        r5_base = subset_metrics["base"]["recall@5"]
        print(
            f"{name}: all R@5={r5:.3f} | base R@5={r5_base:.3f} | utt R@5={r5_utt:.3f}"
        )

    # -------------------------------------------------------------------
    # Targeted: hard case check — doc_utt_0 for q_utt_0
    # -------------------------------------------------------------------
    hard_case: dict[str, Any] = {}
    for name, res in all_variant_results.items():
        # Recompute the single-query ranking for q_utt_0 under this variant.
        v = res["config"]
        kwargs: dict[str, Any] = {"source": v["source"], "agg": v["agg"]}
        if "alpha" in v:
            kwargs["alpha"] = v["alpha"]
        if "beta" in v:
            kwargs["beta"] = v["beta"]
        if v["agg"] == "routed":
            kwargs["intent"] = intents.get("q_utt_0", "ambiguous")
        scores = anchor_retrieve(store, astore, query_intervals["q_utt_0"], **kwargs)
        ranked = [
            d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
        want = "doc_utt_0"
        rank = ranked.index(want) + 1 if want in ranked else None
        hard_case[name] = {
            "rank_of_doc_utt_0": rank,
            "top5": ranked[:5],
            "score_of_doc_utt_0": scores.get(want, 0.0),
        }

    # -------------------------------------------------------------------
    # Hybrid comparison for best R2 variant (semantic rerank of top-20)
    # -------------------------------------------------------------------
    # Pick the best variant by all R@5 (tie-break by all mrr)
    best = max(
        all_variant_results.items(),
        key=lambda kv: (
            kv[1]["subsets"]["all"]["recall@5"],
            kv[1]["subsets"]["all"]["mrr"],
        ),
    )
    best_name = best[0]

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    def hybrid_rank(qid: str, scores: dict[str, float]) -> list[tuple[str, float]]:
        t_ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not t_ranked:
            return sorted(
                ((d, cosine(query_embs[qid], doc_embs[d])) for d in doc_embs),
                key=lambda x: x[1],
                reverse=True,
            )
        cand = [d for d, _ in t_ranked[:20]]
        sem_scores = {d: cosine(query_embs[qid], doc_embs[d]) for d in cand}
        return sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)

    # Hybrid over best + hybrid over R1b (referent-only, standing in for
    # current system)
    hybrid_results: dict[str, dict[str, Any]] = {}
    for variant_name in [best_name, "R1b_referent_only"]:
        v = all_variant_results[variant_name]["config"]
        subset_m: dict[str, dict[str, float]] = {}
        # Compute per-subset metrics
        for sname, qids in subsets.items():
            rec5s, rec10s, mrrs, ndcgs = [], [], [], []
            for qid in qids:
                kwargs: dict[str, Any] = {"source": v["source"], "agg": v["agg"]}
                if "alpha" in v:
                    kwargs["alpha"] = v["alpha"]
                if "beta" in v:
                    kwargs["beta"] = v["beta"]
                if v["agg"] == "routed":
                    kwargs["intent"] = intents.get(qid, "ambiguous")
                scores = anchor_retrieve(store, astore, query_intervals[qid], **kwargs)
                if not scores:
                    ranked_pairs = sorted(
                        ((d, cosine(query_embs[qid], doc_embs[d])) for d in doc_embs),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                else:
                    ranked_pairs = hybrid_rank(qid, scores)
                ranked = [d for d, _ in ranked_pairs]
                rel = all_gold.get(qid, set())
                if not rel:
                    continue
                rec5s.append(recall_at_k(ranked, rel, 5))
                rec10s.append(recall_at_k(ranked, rel, 10))
                mrrs.append(mrr(ranked, rel))
                ndcgs.append(ndcg_at_k(ranked, rel, 10))
            subset_m[sname] = {
                "recall@5": average(rec5s),
                "recall@10": average(rec10s),
                "mrr": average(mrrs),
                "ndcg@10": average(ndcgs),
                "n": len(rec5s),
            }
        hybrid_results[f"HYBRID_{variant_name}"] = {
            "config": {**v, "hybrid": True},
            "subsets": subset_m,
        }
        print(
            f"HYBRID_{variant_name}: all R@5={subset_m['all']['recall@5']:.3f} "
            f"| base R@5={subset_m['base']['recall@5']:.3f} "
            f"| utt R@5={subset_m['utt']['recall@5']:.3f}"
        )

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    total_cost = extract_cost + intent_cost
    payload = {
        "subsets_sizes": {k: len(v) for k, v in subsets.items()},
        "variants": all_variant_results,
        "hybrid": hybrid_results,
        "best_variant": best_name,
        "hard_case_q_utt_0_doc_utt_0": hard_case,
        "intents": intents,
        "cost": {
            "extraction_usd": extract_cost,
            "intent_classifier_usd": intent_cost,
            "total_usd": total_cost,
            "extractor_tokens": ex.usage,
            "intent_tokens": ic.usage,
        },
        "base_hybrid_reference": {
            "recall@5": 0.555,
            "mrr": 0.918,
            "ndcg@10": 0.652,
        },
    }

    with (RESULTS_DIR / "representation_r1r2.json").open("w") as f:
        json.dump(payload, f, indent=2, default=str)

    # -------------------------------------------------------------------
    # Markdown report
    # -------------------------------------------------------------------
    lines: list[str] = [
        "# R1 + R2 Representation Experiments\n\n",
        "Utterance anchor (R1) + dual-score aggregation (R2). See "
        "`REPRESENTATION_EXPERIMENTS.md` for spec.\n\n",
        "## Reference baselines\n",
        "- Current base hybrid (T+S): R@5 0.555, MRR 0.918, NDCG@10 0.652 "
        "(on base 55 queries).\n\n",
        "## Variants — all subsets, R@5\n\n",
        "| Variant | all | base | disc | utt | utt_utterance | utt_referent |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    rows = list(all_variant_results.items()) + list(hybrid_results.items())
    for name, res in rows:
        s = res["subsets"]
        lines.append(
            f"| {name} | "
            f"{s['all']['recall@5']:.3f} | "
            f"{s['base']['recall@5']:.3f} | "
            f"{s['disc']['recall@5']:.3f} | "
            f"{s['utt']['recall@5']:.3f} | "
            f"{s['utt_utterance']['recall@5']:.3f} | "
            f"{s['utt_referent']['recall@5']:.3f} |\n"
        )

    lines.append("\n## Variants — all subsets, MRR\n\n")
    lines.append(
        "| Variant | all | base | disc | utt | utt_utterance | utt_referent |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for name, res in rows:
        s = res["subsets"]
        lines.append(
            f"| {name} | "
            f"{s['all']['mrr']:.3f} | "
            f"{s['base']['mrr']:.3f} | "
            f"{s['disc']['mrr']:.3f} | "
            f"{s['utt']['mrr']:.3f} | "
            f"{s['utt_utterance']['mrr']:.3f} | "
            f"{s['utt_referent']['mrr']:.3f} |\n"
        )

    lines.append("\n## Variants — all subsets, NDCG@10\n\n")
    lines.append(
        "| Variant | all | base | disc | utt | utt_utterance | utt_referent |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for name, res in rows:
        s = res["subsets"]
        lines.append(
            f"| {name} | "
            f"{s['all']['ndcg@10']:.3f} | "
            f"{s['base']['ndcg@10']:.3f} | "
            f"{s['disc']['ndcg@10']:.3f} | "
            f"{s['utt']['ndcg@10']:.3f} | "
            f"{s['utt_utterance']['ndcg@10']:.3f} | "
            f"{s['utt_referent']['ndcg@10']:.3f} |\n"
        )

    lines.append(
        '\n## Hard case: q_utt_0 ("What did I write 2 years ago?") → doc_utt_0\n\n'
    )
    lines.append(
        "Doc_utt_0 was written 2024-04-23 (= 2y before today) saying "
        '"Back in the 90s my dad taught me to fish." Should be retrieved '
        "via the utterance anchor, NOT via the 90s referent.\n\n"
    )
    lines.append("| Variant | rank_of_doc_utt_0 | top5 |\n")
    lines.append("|---|---:|---|\n")
    for name, info in hard_case.items():
        rank = info["rank_of_doc_utt_0"]
        rank_str = str(rank) if rank else "not-ranked"
        lines.append(f"| {name} | {rank_str} | {', '.join(info['top5'])} |\n")

    lines.append("\n## Intents assigned (utterance queries)\n\n")
    lines.append("| QID | text | assigned_intent | expected_intent |\n")
    lines.append("|---|---|---|---|\n")
    for q in utt_queries:
        lines.append(
            f"| {q['query_id']} | {q['text']} | {intents.get(q['query_id'], '?')} | "
            f"{q.get('intent', '?')} |\n"
        )

    lines.append("\n## Cost\n")
    lines.append(
        f"- Extraction: ${extract_cost:.4f} ({ex.usage['input']} in / "
        f"{ex.usage['output']} out tokens)\n"
    )
    lines.append(
        f"- Intent classifier: ${intent_cost:.4f} ({ic.usage['input']} in / "
        f"{ic.usage['output']} out tokens)\n"
    )
    lines.append(f"- **Total**: ${total_cost:.4f}\n")

    lines.append("\n## Best variant (by all R@5 → tiebreak all MRR)\n")
    lines.append(f"- **{best_name}**\n")

    with (RESULTS_DIR / "representation_r1r2.md").open("w") as f:
        f.writelines(lines)

    print(f"\nTotal cost this run: ${total_cost:.4f}")
    print("Wrote results/representation_r1r2.json and .md")


if __name__ == "__main__":
    asyncio.run(run())
