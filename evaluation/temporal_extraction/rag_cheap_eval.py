"""Cheap RAG fusion re-eval: V1, V2, V3, V4, V5, V7 only (skip V6/V8/V9).

Uses the ship-best v2' extractor (not the basic Extractor that
rag_eval.py uses) + multi-axis scorer + Allen + Era channels over existing
subsets: base, disc, utt, era, axis, allen.

SKIPS V8 (LLM-RERANK — expensive, often timeouts).
SKIPS V6 (adds nothing above V4 RRF).
SKIPS V9 (just a mix of V1/V3/V4).
SKIPS adversarial subset here — Part A covers that.

Router (V5) makes 1 LLM call per query, budget-capped.

Outputs results/rag_cheap.{json,md}.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from advanced_common import LLMCaller
from allen_extractor import AllenExtractor
from allen_retrieval import allen_retrieve, te_interval
from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from baselines import embed_all
from era_extractor import EraExtractor
from event_resolver import EventResolver
from expander import expand
from extractor_v2p import ExtractorV2P
from multi_axis_scorer import axis_score, tag_score
from multi_axis_tags import tags_for_axes
from openai import AsyncOpenAI
from rag_pipeline import (
    v1_cascade,
    v2_temporal_only,
    v3_semantic_only,
    v4_rrf_all,
    v5_routed_single,
    v7_score_blend,
)
from rag_router import RagRouter
from schema import GRANULARITY_ORDER, TimeExpression, parse_iso, to_us
from scorer import Interval, score_jaccard_composite

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
RAG_CACHE_DIR = ROOT / "cache" / "rag_cheap"
RAG_CACHE_DIR.mkdir(exist_ok=True, parents=True)

TOP_K = 10
LLM_CALL_TIMEOUT_S = 30.0


def _patched_client() -> AsyncOpenAI:
    return AsyncOpenAI(timeout=LLM_CALL_TIMEOUT_S, max_retries=1)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


SUBSET_FILES = [
    ("base", "queries.jsonl", "docs.jsonl", "gold.jsonl"),
    ("discriminator", "disc_queries.jsonl", "disc_docs.jsonl", "disc_gold.jsonl"),
    (
        "utterance",
        "utterance_queries.jsonl",
        "utterance_docs.jsonl",
        "utterance_gold.jsonl",
    ),
    ("era", "era_queries.jsonl", "era_docs.jsonl", "era_gold.jsonl"),
    ("axis", "axis_queries.jsonl", "axis_docs.jsonl", "axis_gold.jsonl"),
    ("allen", "allen_queries.jsonl", "allen_docs.jsonl", "allen_gold.jsonl"),
]


def load_all():
    subset_to_queries: dict[str, list[dict]] = {}
    all_docs: list[dict] = []
    seen: set[str] = set()
    all_gold: dict[str, set[str]] = {}
    query_meta: dict[str, dict] = {}

    for subset, qf, df, gf in SUBSET_FILES:
        queries = load_jsonl(DATA_DIR / qf)
        docs = load_jsonl(DATA_DIR / df)
        gold = load_jsonl(DATA_DIR / gf)
        subset_to_queries[subset] = queries
        for d in docs:
            if d["doc_id"] in seen:
                continue
            seen.add(d["doc_id"])
            all_docs.append(d)
        for g in gold:
            qid = g["query_id"]
            all_gold[qid] = set(g.get("relevant_doc_ids") or [])
        for q in queries:
            meta = {"subset": subset}
            if "relation" in q:
                meta["allen_relation"] = q["relation"]
                meta["allen_anchor_span"] = q.get("anchor_span")
                meta["allen_anchor_id"] = q.get("anchor_id")
            query_meta[q["query_id"]] = meta
    return subset_to_queries, all_docs, all_gold, query_meta


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


def build_time_memory(tes: list[TimeExpression]) -> dict[str, Any]:
    intervals: list[Interval] = []
    axes_per: list[dict[str, AxisDistribution]] = []
    multi_tags: set[str] = set()
    for te in tes:
        intervals.extend(flatten_intervals(te))
        ax = axes_for_expression(te)
        axes_per.append(ax)
        multi_tags |= tags_for_axes(ax)
    axes_merged = merge_axis_dists(axes_per)
    return {
        "intervals": intervals,
        "axes_merged": axes_merged,
        "multi_tags": multi_tags,
    }


def _empty_memory():
    return {
        "intervals": [],
        "axes_merged": {
            a: AxisDistribution(axis=a, values={}, informative=False) for a in AXES
        },
        "multi_tags": set(),
    }


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


def multi_axis_scores(q_mem, doc_mems, alpha=0.5, beta=0.35, gamma=0.15):
    qa = q_mem["axes_merged"]
    q_tags = q_mem["multi_tags"]
    q_ivs = q_mem["intervals"]
    raw_iv = {d: interval_pair_best(q_ivs, b["intervals"]) for d, b in doc_mems.items()}
    max_iv = max(raw_iv.values(), default=0.0) if raw_iv else 0.0
    scores: dict[str, float] = {}
    for d, b in doc_mems.items():
        iv_norm = raw_iv[d] / max_iv if max_iv > 0 else 0.0
        a_sc = axis_score(qa, b["axes_merged"])
        t_sc = tag_score(q_tags, b["multi_tags"])
        scores[d] = alpha * iv_norm + beta * a_sc + gamma * t_sc
    return scores


def semantic_scores(q_emb, doc_embs):
    qn = float(np.linalg.norm(q_emb)) or 1e-9
    out = {}
    for d, v in doc_embs.items():
        vn = float(np.linalg.norm(v)) or 1e-9
        out[d] = float(np.dot(q_emb, v) / (qn * vn))
    return out


def te_window_us(te: TimeExpression):
    if te.kind == "instant" and te.instant:
        return to_us(te.instant.earliest), to_us(te.instant.latest)
    if te.kind == "interval" and te.interval:
        return to_us(te.interval.start.earliest), to_us(te.interval.end.latest)
    return None


def era_scores(q_era_tes, doc_era_tes):
    q_windows = [te_window_us(t) for t in q_era_tes]
    q_windows = [w for w in q_windows if w is not None and w[1] > w[0]]
    if not q_windows:
        return {}
    out = {}
    for doc_id, tes in doc_era_tes.items():
        d_windows = [te_window_us(t) for t in tes]
        d_windows = [w for w in d_windows if w is not None and w[1] > w[0]]
        best_total = 0.0
        for qw in q_windows:
            qe, ql = qw
            best = 0.0
            for dw in d_windows:
                de, dl = dw
                inter = min(ql, dl) - max(qe, de)
                if inter <= 0:
                    continue
                union = max(ql, dl) - min(qe, de)
                if union <= 0:
                    continue
                j = inter / union
                if j > best:
                    best = j
            best_total += best
        if best_total > 0:
            out[doc_id] = best_total
    return out


def allen_scores_for_query(
    q_meta, exprs_by_doc, resolver_anchor_cache, resolver_anchor_doc
):
    relation = q_meta.get("allen_relation")
    span = q_meta.get("allen_anchor_span")
    if not relation or not span:
        return {}
    anchor_iv = resolver_anchor_cache.get(span)
    if anchor_iv is None:
        return {}
    anchor_doc_id = resolver_anchor_doc.get(span)
    from schema import FuzzyInstant

    ai = anchor_iv
    earliest = datetime.fromtimestamp(ai.earliest / 1_000_000, tz=timezone.utc)
    latest = datetime.fromtimestamp(ai.latest / 1_000_000, tz=timezone.utc)
    fake_te = TimeExpression(
        kind="instant",
        surface=span,
        reference_time=datetime.now(timezone.utc),
        confidence=1.0,
        instant=FuzzyInstant(
            earliest=earliest, latest=latest, best=None, granularity="day"
        ),
    )
    scores = allen_retrieve(
        relation,
        fake_te,
        exprs_by_doc,
        resolve_anchor=lambda s: resolver_anchor_cache.get(s),
        anchor_doc_id=anchor_doc_id,
    )
    return dict(scores)


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


def avg(vs):
    xs = [v for v in vs if not math.isnan(v)]
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(ranked_map, gold, qids):
    r5, r10, mr, nd = [], [], [], []
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            continue
        ranked = ranked_map.get(qid, [])
        r5.append(recall_at_k(ranked, rel, 5))
        r10.append(recall_at_k(ranked, rel, 10))
        mr.append(mrr(ranked, rel))
        nd.append(ndcg_at_k(ranked, rel, 10))
    return {
        "recall@5": avg(r5),
        "recall@10": avg(r10),
        "mrr": avg(mr),
        "ndcg@10": avg(nd),
        "n": len(r5),
    }


async def extract_v2p_for_corpus(items, label):
    ex = ExtractorV2P(concurrency=8)
    ex.client = _patched_client()

    async def one(iid, text, ref):
        try:
            return iid, await asyncio.wait_for(ex.extract(text, ref), timeout=240.0)
        except asyncio.TimeoutError:
            print(f"  [{label}] v2p TIMEOUT for {iid}", flush=True)
            return iid, []
        except Exception as e:
            print(f"  [{label}] v2p failed for {iid}: {e}", flush=True)
            return iid, []

    print(f"  v2p-extracting ({label}, n={len(items)})...", flush=True)
    results = await asyncio.gather(*(one(*it) for it in items))
    ex.cache.save()
    ex.shared_pass2_cache.save()
    print(f"  [{label}] v2p usage: {ex.usage}", flush=True)
    return dict(results), ex.usage


async def extract_era_for_corpus(items, label):
    llm = LLMCaller(concurrency=8)
    llm.client = _patched_client()
    ex = EraExtractor(llm)

    async def one(iid, text, ref):
        try:
            return iid, await asyncio.wait_for(ex.extract(text, ref), timeout=240.0)
        except asyncio.TimeoutError:
            return iid, []
        except Exception:
            return iid, []

    print(f"  era-extracting ({label}, n={len(items)})...", flush=True)
    results = await asyncio.gather(*(one(*it) for it in items))
    llm.save()
    print(f"  [{label}] era usage: {llm.usage}", flush=True)
    return dict(results), llm.usage


async def extract_allen_for_corpus(items, label):
    ex = AllenExtractor(concurrency=8)
    ex.client = _patched_client()

    async def one(iid, text, ref):
        try:
            return iid, await asyncio.wait_for(ex.extract(text, ref), timeout=240.0)
        except asyncio.TimeoutError:
            return iid, []
        except Exception:
            return iid, []

    print(f"  allen-extracting ({label}, n={len(items)})...", flush=True)
    results = await asyncio.gather(*(one(*it) for it in items))
    ex.save()
    print(f"  [{label}] allen usage: {ex.usage}", flush=True)
    return dict(results), ex.usage


async def main() -> None:
    t0 = time.time()
    print("=" * 60, flush=True)
    print("Cheap RAG Fusion Eval (V1-V5, V7; skip V6/V8/V9)", flush=True)
    print("=" * 60, flush=True)

    subset_to_queries, all_docs, all_gold, query_meta = load_all()
    all_doc_ids = [d["doc_id"] for d in all_docs]
    total_queries = sum(len(q) for q in subset_to_queries.values())
    print(f"Loaded {len(all_docs)} unique docs, {total_queries} queries.", flush=True)
    for s, qs in subset_to_queries.items():
        print(f"  {s}: {len(qs)} queries", flush=True)

    doc_by_id = {d["doc_id"]: d for d in all_docs}

    # --- Extract ---
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in all_docs]
    all_queries_flat: list[dict] = []
    for qs in subset_to_queries.values():
        all_queries_flat.extend(qs)
    q_items = [
        (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in all_queries_flat
    ]

    print("\n[1/5] v2p TE extraction (T channel)...", flush=True)
    doc_tes, u1 = await extract_v2p_for_corpus(doc_items, "docs")
    q_tes, u2 = await extract_v2p_for_corpus(q_items, "queries")

    print("\n[2/5] Era extraction (E channel)...", flush=True)
    era_doc_tes, u3 = await extract_era_for_corpus(doc_items, "docs")
    era_q_tes, u4 = await extract_era_for_corpus(q_items, "queries")

    print("\n[3/5] Embed docs + queries (S channel)...", flush=True)
    doc_texts = [d["text"] for d in all_docs]
    q_texts = [q["text"] for q in all_queries_flat]
    doc_embs_arr = await embed_all(doc_texts, concurrency=10)
    q_embs_arr = await embed_all(q_texts, concurrency=10)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(all_docs)}
    query_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(all_queries_flat)}
    print(f"  embedded {len(doc_embs)} docs, {len(query_embs)} queries.", flush=True)

    # --- T memories ---
    doc_mems = {did: build_time_memory(tes) for did, tes in doc_tes.items()}
    for did in all_doc_ids:
        doc_mems.setdefault(did, _empty_memory())
    q_mems = {qid: build_time_memory(tes) for qid, tes in q_tes.items()}
    for q in all_queries_flat:
        q_mems.setdefault(q["query_id"], _empty_memory())

    # --- Allen (allen subset only) ---
    print("\n[4/5] Allen extraction (A channel, allen subset)...", flush=True)
    allen_queries = subset_to_queries.get("allen", [])
    allen_docs_raw = load_jsonl(DATA_DIR / "allen_docs.jsonl")
    allen_doc_id_set = {d["doc_id"] for d in allen_docs_raw}
    allen_docs = [d for d in all_docs if d["doc_id"] in allen_doc_id_set]
    allen_doc_exprs = {}
    resolver_anchor_cache = {}
    resolver_anchor_doc = {}
    allen_cost = 0.0
    resolver_cost = 0.0
    allen_usage_in = allen_usage_out = 0
    if allen_docs and allen_queries:
        a_doc_items = [
            (d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in allen_docs
        ]
        a_q_items = [
            (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in allen_queries
        ]
        allen_doc_exprs, u5 = await extract_allen_for_corpus(a_doc_items, "allen-docs")
        allen_q_exprs, u6 = await extract_allen_for_corpus(a_q_items, "allen-queries")
        allen_usage_in = u5.get("input", 0) + u6.get("input", 0)
        allen_usage_out = u5.get("output", 0) + u6.get("output", 0)
        resolver = EventResolver()
        await resolver.index_docs(allen_docs)
        anchor_spans = {q["anchor_span"] for q in allen_queries if "anchor_span" in q}
        for aes in allen_doc_exprs.values():
            for ae in aes:
                if ae.anchor and ae.anchor.kind == "event":
                    anchor_spans.add(ae.anchor.span)

        async def _prewarm(span: str):
            e = await resolver.resolve(span)
            if e is not None:
                resolver_anchor_cache[span] = te_interval(e.time)
                resolver_anchor_doc[span] = e.doc_id
            else:
                resolver_anchor_cache[span] = None

        await asyncio.gather(*[_prewarm(s) for s in anchor_spans])
        resolver_cost = resolver.cost_usd()

    # --- Router (V5) ---
    print("\n[5/5] Routing queries (LLM, 1 call/query)...", flush=True)
    router_llm = LLMCaller(concurrency=8)
    router_llm.client = _patched_client()
    router = RagRouter(router_llm)
    q_pairs = [(q["query_id"], q["text"]) for q in all_queries_flat]
    router_intents = await router.classify_all(q_pairs)
    router_llm.save()
    router_cost = router_llm.cost_usd()
    print(f"  router usage: {router_llm.usage}, cost ${router_cost:.4f}", flush=True)

    # --- Per-query score dicts ---
    print("\nComputing per-query T/S/A/E scores...", flush=True)
    per_q: dict[str, dict[str, dict[str, float]]] = {}
    for q in all_queries_flat:
        qid = q["query_id"]
        qm = q_mems.get(qid, _empty_memory())
        t_s = multi_axis_scores(qm, doc_mems)
        s_s = semantic_scores(query_embs[qid], doc_embs)
        meta = query_meta[qid]
        if meta.get("allen_relation"):
            a_s = allen_scores_for_query(
                meta, allen_doc_exprs, resolver_anchor_cache, resolver_anchor_doc
            )
        else:
            a_s = {}
        q_era = era_q_tes.get(qid, [])
        e_s = era_scores(q_era, era_doc_tes) if q_era else {}
        per_q[qid] = {"T": t_s, "S": s_s, "A": a_s, "E": e_s}

    # --- Run variants ---
    print("\nRunning 6 variants...", flush=True)
    variants = [
        "V1_CASCADE",
        "V2_TEMPORAL-ONLY",
        "V3_SEMANTIC-ONLY",
        "V4_RRF-ALL",
        "V5_ROUTED-SINGLE",
        "V7_SCORE-BLEND",
    ]
    ranked: dict[str, dict[str, list[str]]] = {v: {} for v in variants}
    for q in all_queries_flat:
        qid = q["query_id"]
        s = per_q[qid]
        intents = router_intents.get(qid, ["semantic"])
        ranked["V1_CASCADE"][qid] = v1_cascade(s["T"], s["S"], all_doc_ids)
        ranked["V2_TEMPORAL-ONLY"][qid] = v2_temporal_only(s["T"], s["S"])
        ranked["V3_SEMANTIC-ONLY"][qid] = v3_semantic_only(s["S"])
        ranked["V4_RRF-ALL"][qid] = v4_rrf_all(s["T"], s["S"], s["A"], s["E"])
        ranked["V5_ROUTED-SINGLE"][qid] = v5_routed_single(
            intents, s["T"], s["S"], s["A"], s["E"]
        )
        ranked["V7_SCORE-BLEND"][qid] = v7_score_blend(s["T"], s["S"], s["A"], s["E"])

    # --- Evaluate ---
    subset_order = [
        "base",
        "discriminator",
        "utterance",
        "era",
        "axis",
        "allen",
        "combined",
    ]
    metrics = {v: {} for v in variants}
    all_qids = [q["query_id"] for q in all_queries_flat]
    for v in variants:
        for s, qs in subset_to_queries.items():
            qids = [qq["query_id"] for qq in qs]
            metrics[v][s] = evaluate(ranked[v], all_gold, qids)
        metrics[v]["combined"] = evaluate(ranked[v], all_gold, all_qids)

    # --- Cost ---
    def _usd(u_in, u_out):
        return u_in * 0.25 / 1_000_000 + u_out * 2.0 / 1_000_000

    v2p_cost = _usd(
        u1.get("input", 0) + u2.get("input", 0),
        u1.get("output", 0) + u2.get("output", 0),
    )
    era_cost = _usd(
        u3.get("input", 0) + u4.get("input", 0),
        u3.get("output", 0) + u4.get("output", 0),
    )
    allen_cost = _usd(allen_usage_in, allen_usage_out)
    total_cost = v2p_cost + era_cost + allen_cost + router_cost + resolver_cost

    # --- Router accuracy ---
    def expected_intents(subset, q):
        if subset == "allen":
            return ["relational"]
        if subset == "era":
            return ["era"]
        if subset in {"utterance", "axis", "discriminator", "base"}:
            return ["temporal"]
        return ["semantic"]

    confusion = defaultdict(lambda: defaultdict(int))
    router_correct_top1 = router_total = 0
    for subset, qs in subset_to_queries.items():
        for q in qs:
            qid = q["query_id"]
            exp = expected_intents(subset, q)
            got = router_intents.get(qid, [])
            router_total += 1
            if got and got[0] in exp:
                router_correct_top1 += 1
            exp0 = exp[0] if exp else "semantic"
            top = got[0] if got else "?"
            confusion[exp0][top] += 1
    router_acc = router_correct_top1 / router_total if router_total else 0.0

    llm_calls_per_q = {
        "V1_CASCADE": 0,
        "V2_TEMPORAL-ONLY": 0,
        "V3_SEMANTIC-ONLY": 0,
        "V4_RRF-ALL": 0,
        "V5_ROUTED-SINGLE": 1,
        "V7_SCORE-BLEND": 0,
    }

    wall_s = time.time() - t0

    # --- Output ---
    out = {
        "variant_metrics": metrics,
        "cost": {
            "v2p_extractor": v2p_cost,
            "era_extractor": era_cost,
            "allen_extractor": allen_cost,
            "event_resolver": resolver_cost,
            "router": router_cost,
            "total": total_cost,
        },
        "llm_calls_per_query": llm_calls_per_q,
        "router": {
            "top1_accuracy": router_acc,
            "confusion": {k: dict(v) for k, v in confusion.items()},
        },
        "subset_sizes": {k: len(v) for k, v in subset_to_queries.items()},
        "total_queries": total_queries,
        "total_docs": len(all_docs),
        "wall_time_s": wall_s,
    }
    (RESULTS_DIR / "rag_cheap.json").write_text(json.dumps(out, indent=2, default=str))

    # --- Markdown ---
    md: list[str] = []
    md.append("# Cheap RAG Fusion Re-Eval (v2' extractor; V1-V5, V7)\n\n")
    md.append(
        f"Docs: {len(all_docs)}. Queries: {total_queries} "
        f"({', '.join(f'{k}={len(v)}' for k, v in subset_to_queries.items())}).\n\n"
    )
    md.append(f"Wall: {wall_s:.1f}s. Total LLM cost: ${total_cost:.4f}.\n\n")
    md.append("V8 (LLM-RERANK), V6 (ROUTED-MULTI), V9 (HYBRID) SKIPPED.\n\n")

    md.append("## R@5 by variant × subset\n\n")
    md.append("| Variant | " + " | ".join(subset_order) + " |\n")
    md.append("|---|" + "|".join(["---:"] * len(subset_order)) + "|\n")
    for v in variants:
        row = [v] + [
            f"{metrics[v].get(s, {}).get('recall@5', 0.0):.3f}" for s in subset_order
        ]
        md.append("| " + " | ".join(row) + " |\n")

    md.append("\n## NDCG@10 by variant × subset\n\n")
    md.append("| Variant | " + " | ".join(subset_order) + " |\n")
    md.append("|---|" + "|".join(["---:"] * len(subset_order)) + "|\n")
    for v in variants:
        row = [v] + [
            f"{metrics[v].get(s, {}).get('ndcg@10', 0.0):.3f}" for s in subset_order
        ]
        md.append("| " + " | ".join(row) + " |\n")

    md.append("\n## MRR by variant × subset\n\n")
    md.append("| Variant | " + " | ".join(subset_order) + " |\n")
    md.append("|---|" + "|".join(["---:"] * len(subset_order)) + "|\n")
    for v in variants:
        row = [v] + [
            f"{metrics[v].get(s, {}).get('mrr', 0.0):.3f}" for s in subset_order
        ]
        md.append("| " + " | ".join(row) + " |\n")

    md.append("\n## Per-subset winner (R@5; ties by NDCG@10)\n\n")
    md.append("| Subset | Best | R@5 | NDCG@10 | MRR |\n")
    md.append("|---|---|---:|---:|---:|\n")
    for s in subset_order:
        best_v = max(
            variants,
            key=lambda v: (
                metrics[v].get(s, {}).get("recall@5", 0.0),
                metrics[v].get(s, {}).get("ndcg@10", 0.0),
            ),
        )
        m = metrics[best_v].get(s, {})
        md.append(
            f"| {s} | {best_v} | {m.get('recall@5', 0):.3f} | "
            f"{m.get('ndcg@10', 0):.3f} | {m.get('mrr', 0):.3f} |\n"
        )

    md.append("\n## Cost-adjusted ranking (combined R@5 / (1 + LLM_calls/q))\n\n")
    md.append("| Variant | Combined R@5 | LLM calls/q | R@5 / (1+calls) |\n")
    md.append("|---|---:|---:|---:|\n")
    ca = []
    for v in variants:
        r5 = metrics[v]["combined"].get("recall@5", 0.0)
        c = llm_calls_per_q[v]
        ca.append((v, r5, c, r5 / (1.0 + c)))
    ca.sort(key=lambda r: r[3], reverse=True)
    for v, r5, c, eff in ca:
        md.append(f"| {v} | {r5:.3f} | {c} | {eff:.3f} |\n")

    md.append("\n## Router accuracy\n\n")
    md.append(
        f"- Top-1 accuracy: {router_acc:.1%} ({router_correct_top1}/{router_total})\n\n"
    )

    md.append("## Cost\n\n")
    md.append(f"- v2p extractor: ${v2p_cost:.4f}\n")
    md.append(f"- Era extractor: ${era_cost:.4f}\n")
    md.append(f"- Allen extractor: ${allen_cost:.4f}\n")
    md.append(f"- Event resolver: ${resolver_cost:.4f}\n")
    md.append(f"- Router: ${router_cost:.4f}\n")
    md.append(f"- **Total**: ${total_cost:.4f}\n")

    # Ship pick
    md.append("\n## Ship pick\n\n")
    combined = [(v, metrics[v]["combined"].get("recall@5", 0.0)) for v in variants]
    combined.sort(key=lambda r: r[1], reverse=True)
    top_overall = combined[0][0]
    zero_call = [v for v, _ in combined if llm_calls_per_q[v] == 0]
    cheap_best = zero_call[0] if zero_call else top_overall
    if llm_calls_per_q[top_overall] == 0:
        md.append(
            f"**Ship {top_overall}.** Combined R@5 {combined[0][1]:.3f}, 0 LLM calls per query.\n\n"
        )
    else:
        md.append(
            f"**Default:** {cheap_best} "
            f"(R@5 {metrics[cheap_best]['combined']['recall@5']:.3f}, 0 LLM calls/q).\n\n"
            f"**Higher-quality fallback:** {top_overall} "
            f"(R@5 {combined[0][1]:.3f}, {llm_calls_per_q[top_overall]} LLM calls/q).\n\n"
        )

    (RESULTS_DIR / "rag_cheap.md").write_text("".join(md))

    print("\n=== Combined R@5 ===", flush=True)
    for v in variants:
        print(
            f"  {v:<22} R@5={metrics[v]['combined']['recall@5']:.3f} "
            f"NDCG@10={metrics[v]['combined']['ndcg@10']:.3f}",
            flush=True,
        )
    print(f"Total cost ${total_cost:.4f}, wall {wall_s:.1f}s", flush=True)
    print("Wrote results/rag_cheap.{md,json}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
