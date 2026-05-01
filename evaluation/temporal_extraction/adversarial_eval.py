"""Adversarial stress-test evaluation.

End-to-end pipeline for the adversarial corpus:
  1. Load data/adversarial_{docs,queries,gold}.jsonl
  2. Run v2 extractor + era extractor on all items (cached under
     cache/adversarial/). Also run Allen extractor on queries+docs so
     the relational channel is available.
  3. Build multi-axis memory + utterance-anchor store + Allen exprs index.
  4. For each query:
       - Detect whether query is a relational Allen query (has
         relation+anchor). If yes, route to Allen retrieval.
       - Otherwise, run multi-axis scorer (α=0.5, β=0.35, γ=0.15) over
         referent intervals + utterance-anchor dual-score, then semantic
         rerank of the top-20.
  5. Compute per-category R@5, R@10, MRR, NDCG@10 and extraction-level
     signals (did we emit something; did correct-skip cases skip).
  6. Produce results/adversarial.{md,json}.

Per-call timeout: 30s. Hard cap on overall wall-clock via asyncio.wait_for.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from advanced_common import LLMCaller
from allen_extractor import AllenExtractor
from allen_retrieval import allen_retrieve, te_interval
from allen_schema import AllenExpression
from anchor_retrieval import retrieve as anchor_retrieve
from anchor_store import UtteranceAnchorStore
from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from baselines import embed_all, semantic_rank
from era_extractor import EraExtractor
from extractor_v2 import ExtractorV2
from multi_axis_scorer import axis_score as axis_score_fn
from multi_axis_scorer import tag_score
from multi_axis_tags import tags_for_axes
from openai import AsyncOpenAI
from schema import TimeExpression, parse_iso, to_us
from scorer import Interval, score_jaccard_composite
from store import IntervalStore

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache" / "adversarial"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

INTERVAL_DB = CACHE_DIR / "intervals.sqlite"
ANCHOR_DB = CACHE_DIR / "anchors.sqlite"

TOP_K = 10
# Per-LLM-call timeout — the spec's "30s" applies to a single API call,
# not the aggregate per-item extraction (which fans out pass1 + many
# pass2 calls). We enforce 30s at the OpenAI-client level.
LLM_CALL_TIMEOUT_S = 30.0
# Outer per-item wall-clock cap as a safety net. With pass1 + ~5 pass2
# calls per item + concurrency 10, a batch of 60 items spends ~6 LLM
# waves. We give each item 240s.
CALL_TIMEOUT_S = 240.0


def _patched_openai_client() -> AsyncOpenAI:
    """AsyncOpenAI client with a 30s per-call timeout."""
    return AsyncOpenAI(timeout=LLM_CALL_TIMEOUT_S, max_retries=1)


REF_ANCHOR_ALPHA = 1.0  # referent side weight (anchor dual-score default)
REF_ANCHOR_BETA = 0.3  # anchor side weight

ALPHA_IV = 0.5
BETA_AXIS = 0.35
GAMMA_TAG = 0.15


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Flatten TimeExpression -> list[Interval]
# ---------------------------------------------------------------------------
from datetime import timedelta

from expander import expand
from schema import GRANULARITY_ORDER


def flatten_intervals(te: TimeExpression) -> list[Interval]:
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
            if GRANULARITY_ORDER.get(te.interval.start.granularity, 0)
            >= GRANULARITY_ORDER.get(te.interval.end.granularity, 0)
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
            end = min(end, te.recurrence.until.latest or te.recurrence.until.earliest)
        try:
            for inst in expand(te.recurrence, start, end):
                out.append(
                    Interval(
                        earliest_us=to_us(inst.earliest),
                        latest_us=to_us(inst.latest),
                        best_us=to_us(inst.best) if inst.best else None,
                        granularity=inst.granularity,
                    )
                )
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


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


def nanmean(xs: list[float]) -> float:
    vs = [v for v in xs if not math.isnan(v)]
    return sum(vs) / len(vs) if vs else float("nan")


# ---------------------------------------------------------------------------
# Extraction with per-item timeout
# ---------------------------------------------------------------------------
async def run_v2_extract(
    items: list[tuple[str, str, datetime]],
    label: str,
) -> tuple[dict[str, list[TimeExpression]], dict[str, int]]:
    ex = ExtractorV2(cache_subdir="adversarial/extractor_v2", concurrency=6)
    # Replace client with 30s-per-call-timeout variant.
    ex.client = _patched_openai_client()
    results: dict[str, list[TimeExpression]] = {}

    async def one(
        iid: str, text: str, ref: datetime
    ) -> tuple[str, list[TimeExpression]]:
        try:
            tes = await asyncio.wait_for(ex.extract(text, ref), timeout=CALL_TIMEOUT_S)
            return iid, tes
        except asyncio.TimeoutError:
            print(f"  [{label}] v2-extract TIMEOUT for {iid}")
            return iid, []
        except Exception as e:
            print(f"  [{label}] v2-extract failed for {iid}: {e}")
            return iid, []

    print(f"v2-extracting {label} ({len(items)} items)...")
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    ex.cache.save()
    ex.shared_pass2_cache.save()
    print(f"  [{label}] usage: {ex.usage}")
    return results, ex.usage


async def run_era_extract(
    items: list[tuple[str, str, datetime]],
    label: str,
) -> tuple[dict[str, list[TimeExpression]], dict[str, int]]:
    llm = LLMCaller(concurrency=6)
    llm.client = _patched_openai_client()
    ex = EraExtractor(llm)
    results: dict[str, list[TimeExpression]] = {}

    async def one(
        iid: str, text: str, ref: datetime
    ) -> tuple[str, list[TimeExpression]]:
        try:
            tes = await asyncio.wait_for(ex.extract(text, ref), timeout=CALL_TIMEOUT_S)
            return iid, tes
        except asyncio.TimeoutError:
            print(f"  [{label}] era-extract TIMEOUT for {iid}")
            return iid, []
        except Exception as e:
            print(f"  [{label}] era-extract failed for {iid}: {e}")
            return iid, []

    print(f"era-extracting {label} ({len(items)} items)...")
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    llm.save()
    print(f"  [{label}] era usage: {llm.usage}, cost=${llm.cost_usd():.4f}")
    return results, llm.usage


async def run_allen_extract(
    items: list[tuple[str, str, datetime]],
    label: str,
) -> tuple[dict[str, list[AllenExpression]], dict[str, int]]:
    ex = AllenExtractor(concurrency=6)
    ex.client = _patched_openai_client()
    results: dict[str, list[AllenExpression]] = {}

    async def one(iid: str, text: str, ref: datetime):
        try:
            tes = await asyncio.wait_for(ex.extract(text, ref), timeout=CALL_TIMEOUT_S)
            return iid, tes
        except asyncio.TimeoutError:
            print(f"  [{label}] allen-extract TIMEOUT for {iid}")
            return iid, []
        except Exception as e:
            print(f"  [{label}] allen-extract failed for {iid}: {e}")
            return iid, []

    print(f"allen-extracting {label} ({len(items)} items)...")
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    ex.save()
    print(f"  [{label}] allen usage: {ex.usage}, cost=${ex.cost_usd():.4f}")
    return results, ex.usage


# ---------------------------------------------------------------------------
# Doc-memory structures for multi-axis scorer
# ---------------------------------------------------------------------------
def build_memory(
    extracted: dict[str, list[TimeExpression]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for did, tes in extracted.items():
        intervals: list[Interval] = []
        axes_per: list[dict[str, AxisDistribution]] = []
        multi_tags: set[str] = set()
        for te in tes:
            intervals.extend(flatten_intervals(te))
            ax = axes_for_expression(te)
            axes_per.append(ax)
            multi_tags |= tags_for_axes(ax)
        axes_merged = merge_axis_dists(axes_per)
        out[did] = {
            "intervals": intervals,
            "axes_merged": axes_merged,
            "multi_tags": multi_tags,
        }
    return out


def interval_pair_best(q_ivs: list[Interval], d_ivs: list[Interval]) -> float:
    if not q_ivs or not d_ivs:
        return 0.0
    total = 0.0
    for qi in q_ivs:
        best = 0.0
        for si in d_ivs:
            s = score_jaccard_composite(qi, si)
            if s > best:
                best = s
        total += best
    return total


def rank_multi_axis(
    q_mem: dict[str, Any],
    doc_mem: dict[str, dict[str, Any]],
    alpha: float,
    beta: float,
    gamma: float,
) -> list[tuple[str, float]]:
    qa = q_mem["axes_merged"]
    q_tags = q_mem["multi_tags"]
    q_ivs = q_mem["intervals"]
    raw_iv: dict[str, float] = {}
    for did, bundle in doc_mem.items():
        raw_iv[did] = interval_pair_best(q_ivs, bundle["intervals"])
    max_iv = max(raw_iv.values()) if raw_iv else 0.0
    scores: dict[str, float] = {}
    for did, bundle in doc_mem.items():
        iv_norm = raw_iv[did] / max_iv if max_iv > 0 else 0.0
        a_sc = axis_score_fn(qa, bundle["axes_merged"])
        t_sc = tag_score(q_tags, bundle["multi_tags"])
        scores[did] = alpha * iv_norm + beta * a_sc + gamma * t_sc
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    t0 = time.time()
    # 1. Ensure synth
    if not (DATA_DIR / "adversarial_docs.jsonl").exists():
        import adversarial_synth

        adversarial_synth.main()

    docs = load_jsonl(DATA_DIR / "adversarial_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "adversarial_queries.jsonl")
    gold_entries = load_jsonl(DATA_DIR / "adversarial_gold.jsonl")
    gold_map: dict[str, set[str]] = {
        g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_entries
    }
    query_cat: dict[str, str] = {q["query_id"]: q["category"] for q in queries}
    doc_cat: dict[str, str] = {d["doc_id"]: d["category"] for d in docs}
    query_expected_beh: dict[str, str] = {
        g["query_id"]: g.get("expected_behavior", "") for g in gold_entries
    }

    print(
        f"Loaded {len(docs)} docs, {len(queries)} queries, {len(gold_entries)} gold entries."
    )

    # 2. Extract
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    v2_docs, u1 = await run_v2_extract(doc_items, "docs-v2")
    v2_qs, u2 = await run_v2_extract(q_items, "queries-v2")
    era_docs, u3 = await run_era_extract(doc_items, "docs-era")
    era_qs, u4 = await run_era_extract(q_items, "queries-era")
    allen_docs, u5 = await run_allen_extract(doc_items, "docs-allen")
    allen_qs, u6 = await run_allen_extract(q_items, "queries-allen")

    # Merge v2 + era extractions into a single "referent" extraction.
    def merge_tes(
        a: list[TimeExpression], b: list[TimeExpression]
    ) -> list[TimeExpression]:
        seen: set[tuple[str, str]] = set()
        merged: list[TimeExpression] = []
        for te in list(a) + list(b):
            key = (te.kind, (te.surface or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(te)
        return merged

    doc_ext: dict[str, list[TimeExpression]] = {
        d["doc_id"]: merge_tes(
            v2_docs.get(d["doc_id"], []), era_docs.get(d["doc_id"], [])
        )
        for d in docs
    }
    q_ext: dict[str, list[TimeExpression]] = {
        q["query_id"]: merge_tes(
            v2_qs.get(q["query_id"], []), era_qs.get(q["query_id"], [])
        )
        for q in queries
    }

    # 3. Build stores
    if INTERVAL_DB.exists():
        INTERVAL_DB.unlink()
    if ANCHOR_DB.exists():
        ANCHOR_DB.unlink()
    store = IntervalStore(INTERVAL_DB)
    astore = UtteranceAnchorStore(ANCHOR_DB)
    for d in docs:
        for te in doc_ext.get(d["doc_id"], []):
            try:
                store.insert_expression(d["doc_id"], te)
            except Exception as e:
                print(f"  interval insert failed {d['doc_id']}: {e}")
        astore.upsert_anchor(d["doc_id"], parse_iso(d["ref_time"]), "day")

    # 4. Multi-axis memory
    doc_mem = build_memory(doc_ext)
    q_mem = build_memory(q_ext)

    # Ensure every doc covered in memory (some may have no extractions)
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

    # 5. Embeddings for semantic rerank
    print("Embedding docs + queries (cached)...")
    doc_texts_list = [d["text"] for d in docs]
    q_texts_list = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts_list)
    q_embs_arr = await embed_all(q_texts_list)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    def semantic_rerank(cand: list[str], qid: str) -> list[tuple[str, float]]:
        qv = q_embs[qid]
        qn = np.linalg.norm(qv) or 1e-9
        out = []
        for d in cand:
            v = doc_embs.get(d)
            if v is None:
                continue
            vn = np.linalg.norm(v) or 1e-9
            sim = float(np.dot(qv, v) / (qn * vn))
            out.append((d, sim))
        return sorted(out, key=lambda x: x[1], reverse=True)

    # 6. Build flatten cache of query intervals for anchor+referent retrieval
    def query_intervals(qid: str) -> list[Interval]:
        out: list[Interval] = []
        for te in q_ext.get(qid, []):
            out.extend(flatten_intervals(te))
        return out

    # 7. Allen query routing: a query is Allen-relational if Allen extractor
    # found a (relation, anchor) on it.
    def allen_query_info(qid: str) -> tuple[str | None, str | None]:
        for ae in allen_qs.get(qid, []):
            if ae.relation is not None and ae.anchor is not None:
                return ae.relation, ae.anchor.span
        return None, None

    # 8. Rank each query
    rankings: dict[str, list[str]] = {}
    routing_info: dict[str, dict[str, Any]] = {}

    # Prebuild an anchor-cache for Allen resolution — we'll use a dumb
    # resolver: look across all doc extractions for a TimeExpression whose
    # surface fuzzy-matches the anchor span.
    def resolve_anchor_from_docs(span: str) -> Any:
        """Return an _Iv-like object from the best-matching absolute doc
        expression, if any. Returns None otherwise."""
        if not span:
            return None
        span_lc = span.lower().strip().strip("'.,\"")
        for did, tes in doc_ext.items():
            for te in tes:
                iv = te_interval(te)
                if iv is None:
                    continue
                # heuristic: any token overlap with surface
                surf = (te.surface or "").lower()
                if span_lc in surf or surf in span_lc:
                    return iv
        return None

    # Accumulate doc Allen expressions
    doc_allen_by_doc: dict[str, list[AllenExpression]] = {
        d["doc_id"]: allen_docs.get(d["doc_id"], []) for d in docs
    }

    for q in queries:
        qid = q["query_id"]
        relation, anchor_span = allen_query_info(qid)

        # 8a. If relational, run Allen retrieval first
        allen_ranked_ids: list[str] = []
        used_allen = False
        if relation and anchor_span:
            # Try to construct an anchor TimeExpression. We try the
            # query's own resolved anchor first (unlikely; anchor is an
            # event). Otherwise resolve via doc-sweep.
            anchor_te = None
            for ae in allen_qs.get(qid, []):
                if ae.anchor and ae.anchor.resolved is not None:
                    anchor_te = ae.anchor.resolved
                    break
            if anchor_te is None:
                iv = resolve_anchor_from_docs(anchor_span)
                if iv is not None:
                    # synthesize a fake TE with an instant matching the iv
                    from schema import FuzzyInstant

                    anchor_te = TimeExpression(
                        kind="instant",
                        surface=anchor_span,
                        reference_time=parse_iso(q["ref_time"]),
                        instant=FuzzyInstant(
                            earliest=datetime.fromtimestamp(
                                iv.earliest / 1e6, tz=timezone.utc
                            ),
                            latest=datetime.fromtimestamp(
                                iv.latest / 1e6, tz=timezone.utc
                            ),
                            best=None,
                            granularity="day",
                        ),
                    )
            if anchor_te is not None:
                try:
                    allen_scores = allen_retrieve(
                        relation,  # type: ignore
                        anchor_te,
                        doc_allen_by_doc,
                        resolve_anchor=lambda s: resolve_anchor_from_docs(s),
                    )
                    allen_ranked_ids = [
                        d
                        for d, _ in sorted(
                            allen_scores.items(), key=lambda x: x[1], reverse=True
                        )
                    ]
                    used_allen = len(allen_ranked_ids) > 0
                except Exception as e:
                    print(f"  allen retrieval failed for {qid}: {e}")

        # 8b. Multi-axis + anchor referent retrieval
        q_ivs = query_intervals(qid)

        # Use anchor_retrieval.retrieve in 'union' mode with sum_weighted
        # to replicate the ship-best "α=1 anchor + β=0.3 Σreferent" dual
        # score. (Our param convention is inverted from the anchor_store
        # docs: ship-best uses "α=1 anchor + β=0.3 ref".)
        anchor_ref_scores = anchor_retrieve(
            store,
            astore,
            q_ivs,
            source="union",
            agg="sum_weighted",
            alpha=REF_ANCHOR_BETA,  # referent Σ weight (0.3)
            beta=REF_ANCHOR_ALPHA,  # anchor weight (1.0)
        )

        # Multi-axis separate ranking
        ma_ranked = rank_multi_axis(
            q_mem.get(
                qid,
                {
                    "intervals": [],
                    "axes_merged": {
                        a: AxisDistribution(axis=a, values={}, informative=False)
                        for a in AXES
                    },
                    "multi_tags": set(),
                },
            ),
            doc_mem,
            ALPHA_IV,
            BETA_AXIS,
            GAMMA_TAG,
        )

        # Combine: weighted sum of normalized anchor_ref and MA score
        # then sem-rerank top-20.
        # Normalize each signal to [0,1] by max.
        ar_max = max(anchor_ref_scores.values()) if anchor_ref_scores else 0.0
        ma_max = max(s for _, s in ma_ranked) if ma_ranked else 0.0
        combined: dict[str, float] = {}
        for d in {di["doc_id"] for di in docs}:
            ar = anchor_ref_scores.get(d, 0.0)
            ma = dict(ma_ranked).get(d, 0.0)
            ar_n = ar / ar_max if ar_max > 0 else 0.0
            ma_n = ma / ma_max if ma_max > 0 else 0.0
            combined[d] = 0.5 * ar_n + 0.5 * ma_n
        cand = [
            d for d, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)
        ][:20]
        ma_top20 = cand[:]
        sem = semantic_rerank(cand, qid) if cand else []
        ma_ranked_ids = [d for d, _ in sem]

        # 8c. Final routing
        if used_allen and allen_ranked_ids:
            final = allen_ranked_ids[:TOP_K]
            # Backfill with multi-axis
            for d in ma_ranked_ids:
                if d not in final:
                    final.append(d)
        else:
            final = ma_ranked_ids
            # If MA returned nothing (no extracted ivs + no axis signal),
            # fall back to pure semantic
            if not final:
                sem_all = semantic_rank(q_embs[qid], doc_embs)
                final = [d for d, _ in sem_all]

        rankings[qid] = final
        routing_info[qid] = {
            "used_allen": used_allen,
            "relation": relation,
            "anchor_span": anchor_span,
            "allen_top5": allen_ranked_ids[:5],
            "ma_top5": ma_ranked_ids[:5],
            "ma_top20": ma_top20[:20],
        }

    # 9. Evaluate per category
    cats_queries: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        cats_queries[q["category"]].append(q["query_id"])

    per_cat: dict[str, dict[str, float]] = {}
    failure_examples: list[dict[str, Any]] = []
    for cat, qids in sorted(cats_queries.items()):
        r5, r10, mr, nd = [], [], [], []
        for qid in qids:
            rel = gold_map.get(qid, set())
            ranked = rankings.get(qid, [])
            # For cases where expected_behavior says "return empty",
            # we count correct-skip as R@5=1.0 when rel is empty and no
            # doc_id that shares the same category sneaks into top-5.
            if not rel:
                # Expected empty retrieval. "Good" = no thematic doc
                # from the SAME category in top-5 (for A7 / R7 / S3 / S7).
                bad_in_top5 = any(doc_cat.get(d) == cat for d in ranked[:5])
                r5.append(0.0 if bad_in_top5 else 1.0)
                r10.append(0.0 if bad_in_top5 else 1.0)
                mr.append(float("nan"))
                nd.append(float("nan"))
            else:
                r5.append(recall_at_k(ranked, rel, 5))
                r10.append(recall_at_k(ranked, rel, 10))
                mr.append(mrr(ranked, rel))
                nd.append(ndcg_at_k(ranked, rel, TOP_K))
            # Log a failure example if R@5 is strictly below 1.0
            if rel and (recall_at_k(ranked, rel, 5) < 1.0):
                failure_examples.append(
                    {
                        "qid": qid,
                        "category": cat,
                        "query_text": next(
                            q["text"] for q in queries if q["query_id"] == qid
                        ),
                        "gold": sorted(rel),
                        "top5": ranked[:5],
                        "routing": routing_info.get(qid, {}),
                        "expected_behavior": query_expected_beh.get(qid, ""),
                    }
                )
            if not rel:
                # Negative query false-positive failure
                if ranked[:5] and any(doc_cat.get(d) == cat for d in ranked[:5]):
                    failure_examples.append(
                        {
                            "qid": qid,
                            "category": cat,
                            "query_text": next(
                                q["text"] for q in queries if q["query_id"] == qid
                            ),
                            "gold": "[expected empty]",
                            "top5": ranked[:5],
                            "routing": routing_info.get(qid, {}),
                            "expected_behavior": query_expected_beh.get(qid, ""),
                        }
                    )
        per_cat[cat] = {
            "n": len(qids),
            "recall@5": nanmean(r5),
            "recall@10": nanmean(r10),
            "mrr": nanmean(mr),
            "ndcg@10": nanmean(nd),
        }

    # 10. Extraction-level metrics per category
    # For each doc in a category, is the extractor non-empty?
    # For A7 (fictional) and some R3 (zero-width), an empty extraction
    # may itself be desirable.
    doc_by_cat: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        doc_by_cat[d["category"]].append(d["doc_id"])

    extraction_signals: dict[str, dict[str, float]] = {}
    for cat, dids in sorted(doc_by_cat.items()):
        n = len(dids)
        n_emit = sum(1 for did in dids if doc_ext.get(did))
        # Fine-grained: number of TEs
        total_tes = sum(len(doc_ext.get(did, [])) for did in dids)
        # correct-skip rate for A7: correct when NO emission
        correct_skip = None
        if cat == "A7":
            correct_skip = sum(1 for did in dids if not doc_ext.get(did)) / n
        extraction_signals[cat] = {
            "n_docs": n,
            "emit_rate": n_emit / n if n else 0.0,
            "avg_tes_per_doc": total_tes / n if n else 0.0,
            "correct_skip_rate": correct_skip
            if correct_skip is not None
            else float("nan"),
        }

    # 11. Overall metric
    all_r5, all_r10, all_mr, all_nd = [], [], [], []
    for cat, m in per_cat.items():
        all_r5.append(m["recall@5"])
        all_r10.append(m["recall@10"])
        all_mr.append(m["mrr"])
        all_nd.append(m["ndcg@10"])
    overall = {
        "recall@5": nanmean(all_r5),
        "recall@10": nanmean(all_r10),
        "mrr": nanmean(all_mr),
        "ndcg@10": nanmean(all_nd),
    }

    # 12. Cost estimate (token-based, gpt-5-mini pricing)
    # gpt-5-mini: $0.25/M in, $2.00/M out
    usages = [u1, u2, u3, u4, u5, u6]
    total_in = sum(u.get("input", 0) for u in usages)
    total_out = sum(u.get("output", 0) for u in usages)
    cost_usd = total_in * 0.25 / 1_000_000 + total_out * 2.0 / 1_000_000

    wall_s = time.time() - t0

    # 13. Write JSON
    # doc-level extraction inspection for report
    doc_ext_summary: list[dict[str, Any]] = []
    for d in docs:
        did = d["doc_id"]
        tes = doc_ext.get(did, [])
        doc_ext_summary.append(
            {
                "doc_id": did,
                "category": d["category"],
                "text": d["text"],
                "n_tes": len(tes),
                "surfaces": [te.surface for te in tes],
                "kinds": [te.kind for te in tes],
            }
        )

    out_json = {
        "corpus": {
            "n_docs": len(docs),
            "n_queries": len(queries),
            "doc_categories": {
                c: n for c, n in Counter(d["category"] for d in docs).items()
            },
            "query_categories": {
                c: n for c, n in Counter(q["category"] for q in queries).items()
            },
        },
        "overall": overall,
        "per_category": per_cat,
        "extraction_signals": extraction_signals,
        "failure_examples": failure_examples,
        "doc_extraction_summary": doc_ext_summary,
        "query_routing": routing_info,
        "cost": {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "usd": cost_usd,
        },
        "wall_seconds": wall_s,
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float) and math.isnan(o):
            return None
        return o

    (RESULTS_DIR / "adversarial.json").write_text(
        json.dumps(_clean(out_json), indent=2, default=str)
    )

    # 14. Markdown report
    lines: list[str] = []
    lines.append("# Adversarial Stress-Test — Results\n\n")
    lines.append(
        f"Corpus: {len(docs)} docs, {len(queries)} queries. Wall clock: {wall_s:.1f}s. LLM cost: ${cost_usd:.4f}.\n\n"
    )

    lines.append("## Per-category retrieval metrics\n\n")
    lines.append(
        "| Category | N | R@5 | R@10 | MRR | NDCG@10 | Emit rate | Avg TEs/doc | Correct-skip |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for cat in sorted(per_cat):
        m = per_cat[cat]
        e = extraction_signals.get(cat, {})
        cs = e.get("correct_skip_rate", float("nan"))
        cs_s = f"{cs:.2f}" if not math.isnan(cs) else "-"
        emit = e.get("emit_rate", float("nan"))
        emit_s = f"{emit:.2f}" if not math.isnan(emit) else "-"
        tes = e.get("avg_tes_per_doc", float("nan"))
        tes_s = f"{tes:.2f}" if not math.isnan(tes) else "-"
        r5 = m["recall@5"]
        r10 = m["recall@10"]
        mr = m["mrr"]
        nd = m["ndcg@10"]

        def _fmt(x):
            return "-" if math.isnan(x) else f"{x:.3f}"

        lines.append(
            f"| {cat} | {m['n']} | {_fmt(r5)} | {_fmt(r10)} | {_fmt(mr)} | {_fmt(nd)} | {emit_s} | {tes_s} | {cs_s} |\n"
        )

    lines.append(
        f"\n**Overall**: R@5={_fmt(overall['recall@5'])}, R@10={_fmt(overall['recall@10'])}, MRR={_fmt(overall['mrr'])}, NDCG@10={_fmt(overall['ndcg@10'])}\n\n"
    )

    lines.append("## Top failure examples\n\n")
    # Sort failures by most egregious first (R@5 == 0 with gold non-empty)
    sorted_failures = sorted(
        failure_examples,
        key=lambda f: (str(f.get("gold")), f.get("qid", "")),
    )[:15]
    for i, f in enumerate(sorted_failures, start=1):
        lines.append(f"### {i}. `{f['qid']}` ({f['category']}) — {f['query_text']!r}\n")
        lines.append(f"- Gold: {f['gold']}\n")
        lines.append(f"- Top-5: {f['top5']}\n")
        lines.append(
            f"- Routing: used_allen={f['routing'].get('used_allen')}, relation={f['routing'].get('relation')}, anchor={f['routing'].get('anchor_span')}\n"
        )
        lines.append(f"- Expected behavior: {f['expected_behavior']}\n\n")

    lines.append("## Extraction summary (sample)\n\n")
    for e in doc_ext_summary[:25]:
        lines.append(
            f"- **{e['doc_id']}** ({e['category']}) `{e['text']}` -> {e['n_tes']} TEs"
            f" (surfaces={e['surfaces']}, kinds={e['kinds']})\n"
        )

    lines.append("\n## Cost & timing\n\n")
    lines.append(f"- Total LLM tokens: input={total_in}, output={total_out}\n")
    lines.append(f"- Estimated cost: ${cost_usd:.4f}\n")
    lines.append(f"- Wall clock: {wall_s:.1f}s\n")

    (RESULTS_DIR / "adversarial.md").write_text("".join(lines))
    print("\nWrote results/adversarial.{md,json}")
    print(
        f"Overall: R@5={overall['recall@5']:.3f}, MRR={overall['mrr']:.3f}, NDCG@10={overall['ndcg@10']:.3f}"
    )
    print(f"Cost: ${cost_usd:.4f}, Wall: {wall_s:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
