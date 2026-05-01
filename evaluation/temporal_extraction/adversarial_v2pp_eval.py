"""Adversarial re-eval with v2'' extractor (modality + fuzzy + holidays).

Based on adversarial_v2p_eval.py (identical pipeline) except:
- Uses ExtractorV2PP.
- Cache dir ``cache/adversarial_v2pp/``.
- Applies modality filter at retrieval (docs with ALL non-actual TEs are
  dropped from rankings).
- Writes results to ``results/adversarial_v2pp.{json,md}``.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
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
from expander import expand
from extractor_v2pp import ExtractorV2PP
from modality_filter import filter_ranking, partition_by_modality
from modality_schema import get_modality
from multi_axis_scorer import axis_score as axis_score_fn
from multi_axis_scorer import tag_score
from multi_axis_tags import tags_for_axes
from openai import AsyncOpenAI
from schema import GRANULARITY_ORDER, TimeExpression, parse_iso, to_us
from scorer import Interval, score_jaccard_composite
from store import IntervalStore

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache" / "adversarial_v2pp"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

INTERVAL_DB = CACHE_DIR / "intervals.sqlite"
ANCHOR_DB = CACHE_DIR / "anchors.sqlite"

TOP_K = 10
LLM_CALL_TIMEOUT_S = 30.0
CALL_TIMEOUT_S = 240.0

# Retrieval filter toggle
FILTER_MODALITY = True


def _patched_client() -> AsyncOpenAI:
    return AsyncOpenAI(timeout=LLM_CALL_TIMEOUT_S, max_retries=1)


REF_ANCHOR_ALPHA = 1.0
REF_ANCHOR_BETA = 0.3

ALPHA_IV = 0.5
BETA_AXIS = 0.35
GAMMA_TAG = 0.15


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


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


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked, relevant):
    if not relevant:
        return float("nan")
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    if not relevant:
        return float("nan")
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def nanmean(xs):
    vs = [v for v in xs if not math.isnan(v)]
    return sum(vs) / len(vs) if vs else float("nan")


async def run_v2pp_extract(items, label):
    ex = ExtractorV2PP(concurrency=8)
    from extractor_common import LLMCache

    ex.cache = LLMCache(CACHE_DIR / "extractor_v2pp_pass1" / "llm_cache.json")
    ex.client = _patched_client()

    # Per-doc modality map — we need this to survive after extract() returns.
    modality_map: dict[str, dict[str, str]] = {}

    results: dict[str, list[TimeExpression]] = {}

    async def one(iid, text, ref):
        try:
            # Each extract resets modality_by_surface; we snapshot after.
            tes = await asyncio.wait_for(ex.extract(text, ref), timeout=CALL_TIMEOUT_S)
            # Record modality snapshot per iid.
            modality_map[iid] = {
                (te.surface or "").lower(): get_modality(te) for te in tes
            }
            return iid, tes
        except asyncio.TimeoutError:
            print(f"  [{label}] v2pp TIMEOUT for {iid}")
            return iid, []
        except Exception as e:
            print(f"  [{label}] v2pp failed for {iid}: {e}")
            return iid, []

    print(f"v2pp-extracting {label} ({len(items)} items)...")
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    ex.cache.save()
    ex.shared_pass2_cache.save()
    print(f"  [{label}] v2pp usage: {ex.usage}")
    return results, ex.usage, modality_map


async def run_era_extract(items, label):
    llm = LLMCaller(concurrency=8)
    llm.client = _patched_client()
    ex = EraExtractor(llm)
    results: dict[str, list[TimeExpression]] = {}

    async def one(iid, text, ref):
        try:
            tes = await asyncio.wait_for(ex.extract(text, ref), timeout=CALL_TIMEOUT_S)
            return iid, tes
        except asyncio.TimeoutError:
            return iid, []
        except Exception as e:
            print(f"  [{label}] era failed for {iid}: {e}")
            return iid, []

    print(f"era-extracting {label} ({len(items)} items)...")
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    llm.save()
    print(f"  [{label}] era usage: {llm.usage}")
    return results, llm.usage


async def run_allen_extract(items, label):
    ex = AllenExtractor(concurrency=8)
    ex.client = _patched_client()
    results: dict[str, list[AllenExpression]] = {}

    async def one(iid, text, ref):
        try:
            tes = await asyncio.wait_for(ex.extract(text, ref), timeout=CALL_TIMEOUT_S)
            return iid, tes
        except asyncio.TimeoutError:
            return iid, []
        except Exception as e:
            print(f"  [{label}] allen failed for {iid}: {e}")
            return iid, []

    print(f"allen-extracting {label} ({len(items)} items)...")
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    ex.save()
    print(f"  [{label}] allen usage: {ex.usage}")
    return results, ex.usage


def build_memory(extracted):
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


def interval_pair_best(q_ivs, d_ivs):
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


def rank_multi_axis(q_mem, doc_mem, alpha, beta, gamma):
    qa = q_mem["axes_merged"]
    q_tags = q_mem["multi_tags"]
    q_ivs = q_mem["intervals"]
    raw_iv = {
        did: interval_pair_best(q_ivs, bundle["intervals"])
        for did, bundle in doc_mem.items()
    }
    max_iv = max(raw_iv.values()) if raw_iv else 0.0
    scores: dict[str, float] = {}
    for did, bundle in doc_mem.items():
        iv_norm = raw_iv[did] / max_iv if max_iv > 0 else 0.0
        a_sc = axis_score_fn(qa, bundle["axes_merged"])
        t_sc = tag_score(q_tags, bundle["multi_tags"])
        scores[did] = alpha * iv_norm + beta * a_sc + gamma * t_sc
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def main() -> None:
    t0 = time.time()
    docs = load_jsonl(DATA_DIR / "adversarial_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "adversarial_queries.jsonl")
    gold_entries = load_jsonl(DATA_DIR / "adversarial_gold.jsonl")
    gold_map = {
        g["query_id"]: set(g.get("relevant_doc_ids") or []) for g in gold_entries
    }
    query_cat = {q["query_id"]: q["category"] for q in queries}
    doc_cat = {d["doc_id"]: d["category"] for d in docs}
    query_expected_beh = {
        g["query_id"]: g.get("expected_behavior", "") for g in gold_entries
    }

    print(
        f"Loaded {len(docs)} docs, {len(queries)} queries, {len(gold_entries)} gold entries."
    )

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    v2pp_docs, u1, doc_modality = await run_v2pp_extract(doc_items, "docs-v2pp")
    v2pp_qs, u2, q_modality = await run_v2pp_extract(q_items, "queries-v2pp")
    era_docs, u3 = await run_era_extract(doc_items, "docs-era")
    era_qs, u4 = await run_era_extract(q_items, "queries-era")
    allen_docs_ex, u5 = await run_allen_extract(doc_items, "docs-allen")
    allen_qs, u6 = await run_allen_extract(q_items, "queries-allen")

    def merge_tes(a, b):
        seen = set()
        merged = []
        for te in list(a) + list(b):
            key = (te.kind, (te.surface or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(te)
        return merged

    doc_ext = {
        d["doc_id"]: merge_tes(
            v2pp_docs.get(d["doc_id"], []), era_docs.get(d["doc_id"], [])
        )
        for d in docs
    }
    q_ext = {
        q["query_id"]: merge_tes(
            v2pp_qs.get(q["query_id"], []), era_qs.get(q["query_id"], [])
        )
        for q in queries
    }

    # -------------------------------------------------------------
    # Modality-aware doc filtering: docs whose ALL v2pp TEs are
    # non-actual are "skipped" — their non-actual expressions should
    # NOT contribute to interval/axis scoring either. We use the
    # v2pp-only extraction for this decision (era TEs are always
    # treated as actual).
    # -------------------------------------------------------------
    v2pp_doc_ext_for_mod = {d["doc_id"]: v2pp_docs.get(d["doc_id"], []) for d in docs}
    keep_ids, skip_ids = partition_by_modality(v2pp_doc_ext_for_mod)
    print(
        f"Modality filter: {len(keep_ids)} keep, {len(skip_ids)} skipped "
        f"(all-non-actual). Skipped: {sorted(skip_ids)}"
    )

    if INTERVAL_DB.exists():
        INTERVAL_DB.unlink()
    if ANCHOR_DB.exists():
        ANCHOR_DB.unlink()
    store = IntervalStore(INTERVAL_DB)
    astore = UtteranceAnchorStore(ANCHOR_DB)
    for d in docs:
        for te in doc_ext.get(d["doc_id"], []):
            # Skip non-actual expressions at ingest time.
            if get_modality(te) != "actual":
                continue
            try:
                store.insert_expression(d["doc_id"], te)
            except Exception as e:
                print(f"  interval insert failed {d['doc_id']}: {e}")
        astore.upsert_anchor(d["doc_id"], parse_iso(d["ref_time"]), "day")

    # Build memory, but zero-out memory for skipped docs
    def filtered_ext(ext_map):
        return {
            k: [te for te in v if get_modality(te) == "actual"]
            for k, v in ext_map.items()
        }

    doc_mem = build_memory(filtered_ext(doc_ext))
    q_mem = build_memory(filtered_ext(q_ext))
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

    print("Embedding docs + queries...")
    doc_texts_list = [d["text"] for d in docs]
    q_texts_list = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts_list)
    q_embs_arr = await embed_all(q_texts_list)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    def semantic_rerank(cand, qid):
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

    def query_intervals(qid):
        out = []
        for te in q_ext.get(qid, []):
            if get_modality(te) != "actual":
                continue
            out.extend(flatten_intervals(te))
        return out

    def allen_query_info(qid):
        for ae in allen_qs.get(qid, []):
            if ae.relation is not None and ae.anchor is not None:
                return ae.relation, ae.anchor.span
        return None, None

    def resolve_anchor_from_docs(span):
        if not span:
            return None
        span_lc = span.lower().strip().strip("'.,\"")
        for did, tes in doc_ext.items():
            for te in tes:
                if get_modality(te) != "actual":
                    continue
                iv = te_interval(te)
                if iv is None:
                    continue
                surf = (te.surface or "").lower()
                if span_lc in surf or surf in span_lc:
                    return iv
        return None

    doc_allen_by_doc = {d["doc_id"]: allen_docs_ex.get(d["doc_id"], []) for d in docs}

    rankings: dict[str, list[str]] = {}
    routing_info: dict[str, dict[str, Any]] = {}

    for q in queries:
        qid = q["query_id"]
        relation, anchor_span = allen_query_info(qid)
        allen_ranked_ids: list[str] = []
        used_allen = False
        if relation and anchor_span:
            anchor_te = None
            for ae in allen_qs.get(qid, []):
                if ae.anchor and ae.anchor.resolved is not None:
                    anchor_te = ae.anchor.resolved
                    break
            if anchor_te is None:
                iv = resolve_anchor_from_docs(anchor_span)
                if iv is not None:
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
                        relation,
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

        q_ivs = query_intervals(qid)
        anchor_ref_scores = anchor_retrieve(
            store,
            astore,
            q_ivs,
            source="union",
            agg="sum_weighted",
            alpha=REF_ANCHOR_BETA,
            beta=REF_ANCHOR_ALPHA,
        )

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

        if used_allen and allen_ranked_ids:
            final = allen_ranked_ids[:TOP_K]
            for d in ma_ranked_ids:
                if d not in final:
                    final.append(d)
        else:
            final = ma_ranked_ids
            if not final:
                sem_all = semantic_rank(q_embs[qid], doc_embs)
                final = [d for d, _ in sem_all]

        # ---------- v2'' addition: modality filter on final ranking ----------
        if FILTER_MODALITY:
            final = filter_ranking(final, v2pp_doc_ext_for_mod, filter_modality=True)

        rankings[qid] = final
        routing_info[qid] = {
            "used_allen": used_allen,
            "relation": relation,
            "anchor_span": anchor_span,
            "allen_top5": allen_ranked_ids[:5],
            "ma_top5": ma_ranked_ids[:5],
            "ma_top20": ma_top20[:20],
        }

    cats_queries = defaultdict(list)
    for q in queries:
        cats_queries[q["category"]].append(q["query_id"])

    per_cat: dict[str, dict[str, float]] = {}
    failure_examples: list[dict[str, Any]] = []
    for cat, qids in sorted(cats_queries.items()):
        r5, r10, mr, nd = [], [], [], []
        for qid in qids:
            rel = gold_map.get(qid, set())
            ranked = rankings.get(qid, [])
            if not rel:
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
        per_cat[cat] = {
            "n": len(qids),
            "recall@5": nanmean(r5),
            "recall@10": nanmean(r10),
            "mrr": nanmean(mr),
            "ndcg@10": nanmean(nd),
        }

    doc_by_cat = defaultdict(list)
    for d in docs:
        doc_by_cat[d["category"]].append(d["doc_id"])

    extraction_signals: dict[str, dict[str, float]] = {}
    for cat, dids in sorted(doc_by_cat.items()):
        n = len(dids)
        v2pp_emit = sum(1 for did in dids if v2pp_docs.get(did))
        total_tes = sum(len(v2pp_docs.get(did, [])) for did in dids)
        # Modality-aware correct-skip: for A7 we expect all emissions to be
        # NON-actual so the doc is skipped at retrieval.
        correct_skip = None
        if cat == "A7":
            correct_skip = sum(1 for did in dids if did in skip_ids) / n
        # For other categories, emit-rate comes from all extraction (any
        # modality) to be comparable with v2'/v2 baselines.
        extraction_signals[cat] = {
            "n_docs": n,
            "emit_rate": v2pp_emit / n if n else 0.0,
            "avg_tes_per_doc": total_tes / n if n else 0.0,
            "correct_skip_rate": correct_skip
            if correct_skip is not None
            else float("nan"),
        }

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

    usages = [u1, u2, u3, u4, u5, u6]
    total_in = sum(u.get("input", 0) for u in usages)
    total_out = sum(u.get("output", 0) for u in usages)
    cost_usd = total_in * 0.25 / 1_000_000 + total_out * 2.0 / 1_000_000

    wall_s = time.time() - t0

    doc_ext_summary = []
    for d in docs:
        did = d["doc_id"]
        tes = v2pp_docs.get(did, [])
        doc_ext_summary.append(
            {
                "doc_id": did,
                "category": d["category"],
                "text": d["text"],
                "n_tes": len(tes),
                "surfaces": [te.surface for te in tes],
                "kinds": [te.kind for te in tes],
                "modalities": [get_modality(te) for te in tes],
                "skipped": did in skip_ids,
            }
        )

    out_json = {
        "extractor": "v2pp",
        "filter_modality": FILTER_MODALITY,
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
        "modality_partition": {"skipped": sorted(skip_ids), "kept": len(keep_ids)},
        "cost": {"input_tokens": total_in, "output_tokens": total_out, "usd": cost_usd},
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

    (RESULTS_DIR / "adversarial_v2pp.json").write_text(
        json.dumps(_clean(out_json), indent=2, default=str)
    )

    # Load v2 and v2' baselines for comparison
    baseline_v2, baseline_v2p = {}, {}
    try:
        baseline_v2 = json.loads((RESULTS_DIR / "adversarial.json").read_text())
    except Exception:
        pass
    try:
        baseline_v2p = json.loads((RESULTS_DIR / "adversarial_v2p.json").read_text())
    except Exception:
        pass

    lines: list[str] = []
    lines.append(
        "# Adversarial Re-Eval with v2'' Extractor (modality + fuzzy + holidays)\n\n"
    )
    lines.append(
        f"Corpus: {len(docs)} docs, {len(queries)} queries. Wall: {wall_s:.1f}s. LLM cost: ${cost_usd:.4f}.\n"
    )
    lines.append(f"filter_modality = {FILTER_MODALITY}\n\n")

    lines.append("## Per-category — v2 vs v2' vs v2''\n\n")
    lines.append(
        "| Cat | N | v2 R@5 | v2' R@5 | **v2'' R@5** | ΔvsV2' | v2'' Emit | v2'' Avg TEs | CorrSkip(A7) |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for cat in sorted(per_cat):
        m = per_cat[cat]
        e = extraction_signals.get(cat, {})
        bm_v2 = baseline_v2.get("per_category", {}).get(cat, {})
        bm_v2p = baseline_v2p.get("per_category", {}).get(cat, {})

        def _fmt(x):
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "-"
            return f"{x:.3f}"

        def _pct(x):
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "-"
            return f"{x:.2f}"

        cs = e.get("correct_skip_rate", float("nan"))
        cs_s = f"{cs:.2f}" if not (isinstance(cs, float) and math.isnan(cs)) else "-"
        dr5 = (m["recall@5"] - bm_v2p.get("recall@5", 0.0)) if bm_v2p else float("nan")
        lines.append(
            f"| {cat} | {m['n']} | {_fmt(bm_v2.get('recall@5'))} | "
            f"{_fmt(bm_v2p.get('recall@5'))} | **{_fmt(m['recall@5'])}** | "
            f"{_fmt(dr5)} | {_pct(e.get('emit_rate'))} | {_pct(e.get('avg_tes_per_doc'))} | {cs_s} |\n"
        )

    v2_overall = baseline_v2.get("overall", {}).get("recall@5", 0.0) or 0.0
    v2p_overall = baseline_v2p.get("overall", {}).get("recall@5", 0.0) or 0.0
    lines.append(
        f"\n**Overall v2'' R@5**: {overall['recall@5']:.3f} "
        f"(v2 {v2_overall:.3f}, v2' {v2p_overall:.3f}, "
        f"Δ vs v2' {overall['recall@5'] - v2p_overall:+.3f})\n"
    )
    lines.append(
        f"**Overall v2'' R@10**: {overall['recall@10']:.3f}, "
        f"MRR: {overall['mrr']:.3f}, NDCG@10: {overall['ndcg@10']:.3f}\n\n"
    )

    lines.append("## Modality partition\n\n")
    lines.append(f"Docs skipped (ALL extracted TEs non-actual): {len(skip_ids)}\n")
    if skip_ids:
        for did in sorted(skip_ids):
            summary = next((s for s in doc_ext_summary if s["doc_id"] == did), None)
            if summary:
                lines.append(
                    f"- `{did}` ({summary['category']}): `{summary['text'][:80]}` "
                    f"-> modalities={summary['modalities']}\n"
                )
    lines.append("\n")

    lines.append("## Top failures (after v2'')\n\n")
    for i, f in enumerate(
        sorted(failure_examples, key=lambda f: f.get("qid", ""))[:20], 1
    ):
        lines.append(f"### {i}. `{f['qid']}` ({f['category']}) — {f['query_text']!r}\n")
        lines.append(
            f"- Gold: {f['gold']}\n- Top-5: {f['top5']}\n- Expected: {f['expected_behavior']}\n\n"
        )

    lines.append("## Extraction sample (first 30 docs)\n\n")
    for e in doc_ext_summary[:30]:
        mark = " [SKIP]" if e.get("skipped") else ""
        lines.append(
            f"- **{e['doc_id']}** ({e['category']}){mark} `{e['text'][:80]}` -> "
            f"{e['n_tes']} TEs (surfaces={e['surfaces']}, modalities={e['modalities']})\n"
        )

    lines.append("\n## Cost & timing\n\n")
    lines.append(f"- Total LLM tokens: input={total_in}, output={total_out}\n")
    lines.append(f"- Estimated cost: ${cost_usd:.4f}\n")
    lines.append(f"- Wall clock: {wall_s:.1f}s\n")

    (RESULTS_DIR / "adversarial_v2pp.md").write_text("".join(lines))
    print("\nWrote results/adversarial_v2pp.{md,json}")
    print(
        f"v2'' overall: R@5={overall['recall@5']:.3f} "
        f"(v2 was {v2_overall:.3f}, v2' was {v2p_overall:.3f})"
    )
    print(f"Cost: ${cost_usd:.4f}, Wall: {wall_s:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
