"""TempReason pipeline eval — v2 extractor (NOT v2pp), reduced subset.

Variants:
- SEMANTIC-ONLY (text-embedding-3-small cosine)
- T-only        (multi-axis intervals + axes + tags)
- V7            (T + S, weights 0.4 / 0.4)
- V7L           (T + S + L lattice, weights 0.3 / 0.3 / 0.2)

Optional channels (Allen, era) are tested on queries first to gauge fire
rate. If they fire on more than 0 queries, a Allen-augmented variant
("V7+A") is added; otherwise they're skipped.

Hard per-call timeout: 20s. Fail-fast if >25% docs time out.

Usage: uv run python tempreason_pipeline_eval.py
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from baselines import embed_all
from expander import expand
from extractor_v2 import ExtractorV2
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from multi_axis_scorer import axis_score as axis_score_fn
from multi_axis_scorer import tag_score
from multi_axis_tags import tags_for_axes
from openai import AsyncOpenAI
from rag_fusion import score_blend
from schema import GRANULARITY_ORDER, TimeExpression, parse_iso, to_us
from scorer import Interval, score_jaccard_composite

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache" / "tempreason_v2"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
LATTICE_DB = CACHE_DIR / "lattice.sqlite"

TOP_K = 10
PER_CALL_TIMEOUT_S = 60.0
CONCURRENCY = 8
HARD_TIMEOUT_FRAC = 0.25

# Pricing for gpt-5-mini.
PRICE_IN_PER_M = 0.25
PRICE_OUT_PER_M = 2.00


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Interval flatten
# ---------------------------------------------------------------------------
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
        start = min(now - timedelta(days=365 * 5), anchor - timedelta(days=365))
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
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


def eval_rankings(ranked_per_q, gold, qids):
    r5, r10, mr, nd = [], [], [], []
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            continue
        ranked = ranked_per_q.get(qid, [])
        r5.append(recall_at_k(ranked, rel, 5))
        r10.append(recall_at_k(ranked, rel, 10))
        mr.append(mrr(ranked, rel))
        nd.append(ndcg_at_k(ranked, rel, TOP_K))
    return {
        "n": len(r5),
        "recall@5": nanmean(r5),
        "recall@10": nanmean(r10),
        "mrr": nanmean(mr),
        "ndcg@10": nanmean(nd),
    }


def query_rank_of_gold(ranked, relevant) -> int | None:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return i
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
async def run_v2_extract(items, label: str, cache_subdir: str):
    """Run ExtractorV2.extract on each (id, text, ref) item.

    Wraps each call in a per-item asyncio timeout. Returns mapping +
    timeout count + usage.
    """
    ex = ExtractorV2(concurrency=CONCURRENCY, cache_subdir=cache_subdir)
    # Per-call timeout via the OpenAI client; also asyncio.wait_for for
    # belt-and-suspenders.
    ex.client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S, max_retries=1)

    results: dict[str, list[TimeExpression]] = {}
    n_timeout = 0
    n_error = 0
    completed = [0]
    total = len(items)

    async def one(iid, text, ref):
        nonlocal n_timeout, n_error
        try:
            tes = await asyncio.wait_for(
                ex.extract(text, ref), timeout=PER_CALL_TIMEOUT_S * 3
            )
        except asyncio.TimeoutError:
            n_timeout += 1
            tes = []
        except Exception as e:
            n_error += 1
            print(f"  [{label}] FAIL {iid}: {e}", flush=True)
            tes = []
        completed[0] += 1
        if completed[0] % 20 == 0:
            print(
                f"  [{label}] {completed[0]}/{total} (timeout={n_timeout})", flush=True
            )
        return iid, tes

    print(f"v2 {label}: {total} items", flush=True)
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    ex.cache.save()
    try:
        ex.shared_pass2_cache.save()
    except Exception:
        pass

    cost = (
        ex.usage["input"] * PRICE_IN_PER_M / 1_000_000
        + ex.usage["output"] * PRICE_OUT_PER_M / 1_000_000
    )
    print(
        f"  [{label}] usage in={ex.usage['input']}, out={ex.usage['output']}, cost=${cost:.4f}",
        flush=True,
    )
    print(f"  [{label}] timeouts={n_timeout}, errors={n_error}", flush=True)
    return results, ex.usage, n_timeout, n_error


# ---------------------------------------------------------------------------
# Memory build
# ---------------------------------------------------------------------------
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


def rank_multi_axis_t(q_mem, doc_mem, alpha=0.5, beta=0.35, gamma=0.15):
    qa = q_mem.get("axes_merged") or {}
    q_tags = q_mem.get("multi_tags") or set()
    q_ivs = q_mem.get("intervals") or []
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
    return scores


def rank_semantic_s(qid, q_embs, doc_embs):
    qv = q_embs[qid]
    qn = np.linalg.norm(qv) or 1e-9
    out: dict[str, float] = {}
    for d, v in doc_embs.items():
        vn = np.linalg.norm(v) or 1e-9
        out[d] = float(np.dot(qv, v) / (qn * vn))
    return out


def ingest_lattice(store: LatticeStore, extracted):
    store.clear()
    for did, tes in extracted.items():
        all_abs: list[tuple[str, str]] = []
        all_cyc: set[str] = set()
        for te in tes:
            ts = lattice_tags_for_expression(te)
            all_abs.extend(ts.absolute)
            all_cyc |= ts.cyclical
        seen = set()
        dedup = []
        for prec, t in all_abs:
            if t in seen:
                continue
            seen.add(t)
            dedup.append((prec, t))
        store.insert(did, dedup, all_cyc)


def retrieve_lattice_scores(store, query_extracted, all_qids):
    per_q_scores: dict[str, dict[str, float]] = {}
    for qid in all_qids:
        tes = query_extracted.get(qid, [])
        if not tes:
            per_q_scores[qid] = {}
            continue
        scores, _ = lattice_retrieve_multi(store, tes, down_levels=1)
        per_q_scores[qid] = scores
    return per_q_scores


def rank_v7l(t, s, l, weights=None):
    if weights is None:
        weights = {"T": 0.3, "S": 0.3, "L": 0.2}
    fused = score_blend({"T": t, "S": s, "L": l}, weights, top_k_per=40)
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


def rank_v7(t, s, weights=None):
    if weights is None:
        weights = {"T": 0.4, "S": 0.4}
    fused = score_blend({"T": t, "S": s}, weights, top_k_per=40)
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


def rank_t_only(t, all_doc_ids, s_scores):
    """Rank purely by T scores; tie-break by S."""
    return sorted(
        all_doc_ids, key=lambda d: (t.get(d, 0.0), s_scores.get(d, 0.0)), reverse=True
    )


# ---------------------------------------------------------------------------
# Allen channel — minimal direct implementation focused on L3 queries
# ---------------------------------------------------------------------------
async def run_allen_query_extract(queries, ref_times):
    """Run Allen extraction on queries to gauge fire rate."""
    from allen_extractor import AllenExtractor

    ex = AllenExtractor(concurrency=CONCURRENCY)
    ex.client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S, max_retries=1)

    results: dict[str, list[Any]] = {}
    n_timeout = 0
    n_error = 0
    total = len(queries)
    completed = [0]

    async def one(qid, text, ref):
        nonlocal n_timeout, n_error
        try:
            aes = await asyncio.wait_for(
                ex.extract(text, ref), timeout=PER_CALL_TIMEOUT_S * 3
            )
        except asyncio.TimeoutError:
            n_timeout += 1
            aes = []
        except Exception as e:
            n_error += 1
            print(f"  [allen-q] FAIL {qid}: {e}", flush=True)
            aes = []
        completed[0] += 1
        if completed[0] % 10 == 0:
            print(f"  [allen-q] {completed[0]}/{total}", flush=True)
        return qid, aes

    pairs = await asyncio.gather(
        *(one(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries)
    )
    for qid, aes in pairs:
        results[qid] = aes
    ex.save()

    cost = (
        ex.usage["input"] * PRICE_IN_PER_M / 1_000_000
        + ex.usage["output"] * PRICE_OUT_PER_M / 1_000_000
    )
    print(
        f"  [allen-q] cost=${cost:.4f}, timeouts={n_timeout}, errors={n_error}",
        flush=True,
    )
    return results, ex.usage


def count_allen_relations(allen_extracted) -> tuple[int, int, dict[str, int]]:
    """Return (n_queries_with_any_relation, total_relations_across_queries,
    per_relation_counts)."""
    n_qs = 0
    total = 0
    per_rel: dict[str, int] = defaultdict(int)
    for qid, aes in allen_extracted.items():
        had = False
        for ae in aes:
            if ae.relation is not None:
                total += 1
                per_rel[ae.relation] += 1
                had = True
        if had:
            n_qs += 1
    return n_qs, total, dict(per_rel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    t0 = time.time()
    wall_start = time.time()
    WALL_CAP_S = 25 * 60  # 25 min

    docs = load_jsonl(DATA_DIR / "real_benchmark_small_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "real_benchmark_small_queries.jsonl")
    gold_raw = load_jsonl(DATA_DIR / "real_benchmark_small_gold.jsonl")
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_raw}
    subset_of_q = {q["query_id"]: q["subset"] for q in queries}

    print(f"TempReason small: {len(docs)} docs, {len(queries)} queries", flush=True)

    # --- Extract (docs) ---
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    print("=== Extracting docs (v2) ===", flush=True)
    doc_ext, doc_usage, doc_to, doc_err = await run_v2_extract(
        doc_items, "docs", "tempreason_v2_docs"
    )
    if doc_to / max(1, len(doc_items)) > HARD_TIMEOUT_FRAC:
        print(
            f"FAIL-FAST: {doc_to}/{len(doc_items)} doc timeouts > {HARD_TIMEOUT_FRAC}",
            flush=True,
        )
        return

    print("=== Extracting queries (v2) ===", flush=True)
    q_ext, q_usage, q_to, q_err = await run_v2_extract(
        q_items, "queries", "tempreason_v2_queries"
    )

    if time.time() - wall_start > WALL_CAP_S:
        print("WALL CAP HIT after extraction", flush=True)

    total_in = doc_usage["input"] + q_usage["input"]
    total_out = doc_usage["output"] + q_usage["output"]
    cost_extract = total_in * PRICE_IN_PER_M / 1e6 + total_out * PRICE_OUT_PER_M / 1e6
    print(f"v2 extraction cost: ${cost_extract:.4f}", flush=True)

    # --- Build memories ---
    print("Building T-channel memory...", flush=True)
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

    # --- Build lattice ---
    print("Building lattice index...", flush=True)
    if LATTICE_DB.exists():
        LATTICE_DB.unlink()
    store = LatticeStore(LATTICE_DB)
    ingest_lattice(store, doc_ext)
    lat_stats = store.stats()
    print(f"Lattice: {lat_stats}", flush=True)

    # --- Embed ---
    print("Embedding (text-embedding-3-small)...", flush=True)
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    # --- Allen test on L3 queries (and a few L2 to confirm) ---
    print("=== Testing Allen channel on queries ===", flush=True)
    allen_q_ext: dict[str, list[Any]] = {}
    allen_q_usage = {"input": 0, "output": 0}
    try:
        allen_q_ext, allen_q_usage = await run_allen_query_extract(queries, None)
    except Exception as e:
        print(f"Allen query extraction failed: {e}", flush=True)
        traceback.print_exc()

    n_q_with_rel, total_rels, per_rel_count = count_allen_relations(allen_q_ext)
    n_l3_with_rel = sum(
        1
        for qid in allen_q_ext
        if subset_of_q.get(qid) == "L3"
        and any(ae.relation is not None for ae in allen_q_ext[qid])
    )
    n_l3 = sum(1 for q in queries if q["subset"] == "L3")
    print(
        f"Allen fire on queries: {n_q_with_rel}/{len(queries)} (L3: {n_l3_with_rel}/{n_l3})",
        flush=True,
    )
    print(f"Allen relations: total={total_rels}, breakdown={per_rel_count}", flush=True)

    cost_allen_q = (
        allen_q_usage["input"] * PRICE_IN_PER_M / 1e6
        + allen_q_usage["output"] * PRICE_OUT_PER_M / 1e6
    )

    # If Allen fires meaningfully on L3, run on docs and use event resolver
    allen_d_ext: dict[str, list[Any]] = {}
    allen_d_usage = {"input": 0, "output": 0}
    a_scores_per_q: dict[str, dict[str, float]] = {}
    n_l3_resolved = 0
    if n_l3_with_rel >= 5 and time.time() - wall_start < WALL_CAP_S - 6 * 60:
        print("=== Allen fired on L3 — running on docs ===", flush=True)
        try:
            from allen_extractor import AllenExtractor

            ex_d = AllenExtractor(concurrency=CONCURRENCY)
            ex_d.client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S, max_retries=1)

            n_to = 0
            n_err = 0
            completed = [0]
            total = len(docs)

            async def one_d(d):
                nonlocal n_to, n_err
                try:
                    aes = await asyncio.wait_for(
                        ex_d.extract(d["text"], parse_iso(d["ref_time"])),
                        timeout=PER_CALL_TIMEOUT_S * 3,
                    )
                except asyncio.TimeoutError:
                    n_to += 1
                    aes = []
                except Exception:
                    n_err += 1
                    aes = []
                completed[0] += 1
                if completed[0] % 30 == 0:
                    print(f"  [allen-d] {completed[0]}/{total}", flush=True)
                return d["doc_id"], aes

            pairs = await asyncio.gather(*(one_d(d) for d in docs))
            for did, aes in pairs:
                allen_d_ext[did] = aes
            ex_d.save()
            allen_d_usage = ex_d.usage
            print(f"  [allen-d] timeouts={n_to}, errors={n_err}", flush=True)

            # Build Allen retrieval scores: for each L3 query with a relation,
            # try to resolve the anchor span via the event resolver against
            # the doc corpus, then score docs.
            from allen_retrieval import _Iv, te_interval
            from event_resolver import EventResolver

            resolver = EventResolver()
            try:
                await resolver.index_docs(docs)
            except Exception as e:
                print(f"event_resolver.index_docs failed: {e}", flush=True)
                traceback.print_exc()

            # Wrap async resolver in sync for allen_retrieval.
            anchor_resolve_cache: dict[str, _Iv | None] = {}

            def sync_resolve(span: str):
                if span in anchor_resolve_cache:
                    return anchor_resolve_cache[span]
                # Fire and wait — we're inside main's event loop so use nest
                # via run_until_complete is not available; use the resolver's
                # synchronous-like flow: it caches embeddings.
                try:
                    loop = asyncio.get_event_loop()
                    fut = asyncio.ensure_future(resolver.resolve(span))
                    # NOTE: this won't work in the same loop; we'll resolve
                    # all anchors up-front instead.
                except Exception:
                    pass
                return None

            # Pre-resolve all unique anchor spans referenced by query Allen exprs.
            unique_spans: set[str] = set()
            for qid, aes in allen_q_ext.items():
                for ae in aes:
                    if ae.relation is not None and ae.anchor and ae.anchor.span:
                        unique_spans.add(ae.anchor.span)

            print(
                f"  [allen] {len(unique_spans)} unique anchor spans to resolve",
                flush=True,
            )
            resolve_results: dict[str, _Iv | None] = {}
            for span in unique_spans:
                try:
                    entry = await resolver.resolve(span)
                    if entry is not None:
                        iv = te_interval(entry.time)
                        resolve_results[span] = iv
                    else:
                        resolve_results[span] = None
                except Exception:
                    resolve_results[span] = None

            n_resolved = sum(1 for v in resolve_results.values() if v is not None)
            print(
                f"  [allen] resolved {n_resolved}/{len(unique_spans)} anchor spans",
                flush=True,
            )

            def sync_resolve_fast(span: str):
                return resolve_results.get(span)

            # Score per query.
            doc_aes_by_id = allen_d_ext
            for qid, aes in allen_q_ext.items():
                # First Allen relation in the query (if any).
                rel_ae = next((ae for ae in aes if ae.relation is not None), None)
                if rel_ae is None:
                    a_scores_per_q[qid] = {}
                    continue
                anchor_iv = None
                if rel_ae.anchor and rel_ae.anchor.span:
                    anchor_iv = resolve_results.get(rel_ae.anchor.span)
                if anchor_iv is None:
                    a_scores_per_q[qid] = {}
                    continue
                # Build a synthetic anchor TE for allen_retrieve API: we can
                # just call the inner _score_doc_ae loop directly.
                from allen_retrieval import _score_doc_ae

                scores: dict[str, float] = {}
                for did, daes in doc_aes_by_id.items():
                    best = 0.0
                    for dae in daes:
                        s = _score_doc_ae(
                            rel_ae.relation, anchor_iv, dae, sync_resolve_fast
                        )
                        if s > best:
                            best = s
                    scores[did] = best
                a_scores_per_q[qid] = scores
                if subset_of_q.get(qid) == "L3":
                    n_l3_resolved += 1
        except Exception as e:
            print(f"Allen doc pass failed: {e}", flush=True)
            traceback.print_exc()
            a_scores_per_q = {}

    cost_allen_d = (
        allen_d_usage["input"] * PRICE_IN_PER_M / 1e6
        + allen_d_usage["output"] * PRICE_OUT_PER_M / 1e6
    )

    # --- Score per-query ---
    print("Scoring per-query...", flush=True)
    all_qids = {q["query_id"] for q in queries}
    all_doc_ids = [d["doc_id"] for d in docs]
    l_scores_per_q = retrieve_lattice_scores(store, q_ext, all_qids)

    variants: dict[str, dict[str, list[str]]] = {
        "SEMANTIC-ONLY": {},
        "T-only": {},
        "V7 (T+S)": {},
        "V7L (T+S+L)": {},
    }
    has_allen = bool(a_scores_per_q) and any(
        scores for scores in a_scores_per_q.values()
    )
    if has_allen:
        variants["V7+A (T+S+A)"] = {}
        variants["V7L+A (T+S+L+A)"] = {}

    per_q_t_scores: dict[str, dict[str, float]] = {}
    per_q_s_scores: dict[str, dict[str, float]] = {}

    for q in queries:
        qid = q["query_id"]
        t_scores = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        s_scores = rank_semantic_s(qid, q_embs, doc_embs)
        per_q_t_scores[qid] = t_scores
        per_q_s_scores[qid] = s_scores
        l_scores = l_scores_per_q.get(qid, {})

        sem_ranked = sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
        variants["SEMANTIC-ONLY"][qid] = [d for d, _ in sem_ranked]
        variants["T-only"][qid] = rank_t_only(t_scores, all_doc_ids, s_scores)
        variants["V7 (T+S)"][qid] = rank_v7(t_scores, s_scores)
        variants["V7L (T+S+L)"][qid] = rank_v7l(t_scores, s_scores, l_scores)

        if has_allen:
            a_scores = a_scores_per_q.get(qid, {})
            fused_a = score_blend(
                {"T": t_scores, "S": s_scores, "A": a_scores},
                {"T": 0.4, "S": 0.4, "A": 0.2},
                top_k_per=40,
            )
            ranked_a = [d for d, _ in fused_a]
            seen = set(ranked_a)
            tail = [d for d, _ in sem_ranked if d not in seen]
            variants["V7+A (T+S+A)"][qid] = ranked_a + tail

            fused_la = score_blend(
                {"T": t_scores, "S": s_scores, "L": l_scores, "A": a_scores},
                {"T": 0.3, "S": 0.3, "L": 0.2, "A": 0.2},
                top_k_per=40,
            )
            ranked_la = [d for d, _ in fused_la]
            seen2 = set(ranked_la)
            tail2 = [d for d, _ in sem_ranked if d not in seen2]
            variants["V7L+A (T+S+L+A)"][qid] = ranked_la + tail2

    # --- Eval ---
    L2_qids = {qid for qid, sub in subset_of_q.items() if sub == "L2"}
    L3_qids = {qid for qid, sub in subset_of_q.items() if sub == "L3"}
    subsets = {"all": all_qids, "L2": L2_qids, "L3": L3_qids}

    per_variant: dict[str, dict[str, dict[str, float]]] = {}
    for var, ranked_per_q in variants.items():
        per_variant[var] = {}
        for sub_name, qids in subsets.items():
            per_variant[var][sub_name] = eval_rankings(ranked_per_q, gold, qids)

    # --- Failure analysis: where SEMANTIC put gold at rank 2-3, did T push to 1? ---
    rank_compare = []
    for q in queries:
        qid = q["query_id"]
        rel = gold.get(qid, set())
        if not rel:
            continue
        sem_rank = query_rank_of_gold(variants["SEMANTIC-ONLY"][qid], rel)
        t_rank = query_rank_of_gold(variants["T-only"].get(qid, []), rel)
        v7_rank = query_rank_of_gold(variants["V7 (T+S)"][qid], rel)
        v7l_rank = query_rank_of_gold(variants["V7L (T+S+L)"][qid], rel)
        gold_doc_id = next(iter(rel))
        gold_text = next((d["text"] for d in docs if d["doc_id"] == gold_doc_id), "<?>")
        sem_top3 = variants["SEMANTIC-ONLY"][qid][:3]
        sem_top3_texts = [
            next((d["text"] for d in docs if d["doc_id"] == did), "<?>")
            for did in sem_top3
        ]
        rank_compare.append(
            {
                "qid": qid,
                "subset": q["subset"],
                "query": q["text"],
                "ref_time": q["ref_time"],
                "gold_doc_id": gold_doc_id,
                "gold_text": gold_text,
                "rank_sem": sem_rank,
                "rank_t": t_rank,
                "rank_v7": v7_rank,
                "rank_v7l": v7l_rank,
                "sem_top3": sem_top3,
                "sem_top3_texts": sem_top3_texts,
                "n_q_extractions": len(q_ext.get(qid, [])),
                "n_gold_extractions": len(doc_ext.get(gold_doc_id, [])),
            }
        )

    # Top examples where semantic was rank 2/3 but V7/V7L pushed to 1
    promoted_rank2_3 = [
        r
        for r in rank_compare
        if (r["rank_sem"] in (2, 3)) and ((r["rank_v7"] == 1) or (r["rank_v7l"] == 1))
    ]
    demoted_rank1 = [
        r
        for r in rank_compare
        if (r["rank_sem"] == 1)
        and (
            (r["rank_v7"] not in (1, None) and r["rank_v7"] > 1)
            or (r["rank_v7l"] not in (1, None) and r["rank_v7l"] > 1)
        )
    ]
    persistent_low = [
        r
        for r in rank_compare
        if (r["rank_sem"] in (2, 3))
        and not ((r["rank_v7"] == 1) or (r["rank_v7l"] == 1))
    ]

    cost_total = cost_extract + cost_allen_q + cost_allen_d
    wall_s = time.time() - t0

    # --- Write JSON ---
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, set):
            return sorted(o)
        return o

    out_json = {
        "benchmark": {
            "name": "TempReason-derived (subsampled)",
            "n_docs": len(docs),
            "n_queries": len(queries),
            "n_L2": len(L2_qids),
            "n_L3": len(L3_qids),
        },
        "extraction": {
            "doc_timeouts": doc_to,
            "doc_errors": doc_err,
            "query_timeouts": q_to,
            "query_errors": q_err,
            "doc_extractions_per_doc_mean": (
                sum(len(v) for v in doc_ext.values()) / max(1, len(doc_ext))
            ),
            "q_extractions_per_q_mean": (
                sum(len(v) for v in q_ext.values()) / max(1, len(q_ext))
            ),
        },
        "allen": {
            "n_q_with_relation": n_q_with_rel,
            "n_L3_with_relation": n_l3_with_rel,
            "n_L3_total": n_l3,
            "L3_fire_rate": n_l3_with_rel / max(1, n_l3),
            "total_relations": total_rels,
            "per_relation": per_rel_count,
            "n_L3_resolved": n_l3_resolved,
            "doc_pass_run": bool(allen_d_ext),
            "n_doc_with_relation": sum(
                1
                for v in allen_d_ext.values()
                if any(ae.relation is not None for ae in v)
            ),
        },
        "lattice_stats": _clean(lat_stats),
        "per_variant": _clean(per_variant),
        "failure_analysis": {
            "n_total": len(rank_compare),
            "promoted_rank2_3_to_1_count": len(promoted_rank2_3),
            "demoted_rank1_count": len(demoted_rank1),
            "persistent_low_rank_count": len(persistent_low),
            "promoted_examples": _clean(promoted_rank2_3[:5]),
            "demoted_examples": _clean(demoted_rank1[:5]),
            "persistent_low_examples": _clean(persistent_low[:5]),
        },
        "cost": {
            "extraction_v2_usd": cost_extract,
            "allen_q_usd": cost_allen_q,
            "allen_d_usd": cost_allen_d,
            "total_usd": cost_total,
        },
        "wall_seconds": wall_s,
    }

    out_path = RESULTS_DIR / "tempreason_pipeline.json"
    out_path.write_text(json.dumps(out_json, indent=2, default=str))
    print(f"Wrote {out_path}", flush=True)

    # Print summary
    print("\n=== Summary ===")
    print(
        f"{'Variant':<26} {'subset':<6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'NDCG':>6}"
    )
    for var in variants:
        for sub in ["all", "L2", "L3"]:
            m = per_variant[var][sub]
            print(
                f"{var:<26} {sub:<6} {m['recall@5']:>6.3f} {m['recall@10']:>6.3f} "
                f"{m['mrr']:>6.3f} {m['ndcg@10']:>6.3f}"
            )
    print(f"\nCost: ${cost_total:.4f}, wall: {wall_s:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
