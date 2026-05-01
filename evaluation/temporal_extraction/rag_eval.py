"""Orchestrate all 9 RAG variants across query subsets and report.

Steps:
1. Load all docs + queries across subsets into a unified corpus.
2. Extract TimeExpressions for every doc + query using the base Extractor
   (cached; gpt-5-mini). Also run AllenExtractor on allen queries/docs for
   the A channel.
3. Embed every doc + query (text-embedding-3-small, cached).
4. For each query, precompute per-retriever score dicts:
   - T (multi-axis, alpha=0.5/beta=0.35/gamma=0.15) over full corpus.
   - S (cosine) over full corpus.
   - A (Allen) only if query is in allen subset (has relation+anchor_span).
   - E (era) using era-extractor windows.
5. Route each query with rag_router (gpt-5-mini).
6. Run variants V1-V9 and compute R@5/10, MRR, NDCG@10 per subset.
7. V8 LLM-rerank samples up to 30 queries (budget-aware).
8. Write results/rag_integration.{json,md}.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from advanced_common import Embedder, LLMCaller
from allen_extractor import AllenExtractor
from allen_retrieval import allen_retrieve, te_interval
from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from event_resolver import EventResolver
from extractor import Extractor as BaseExtractor
from multi_axis_scorer import axis_score, tag_score
from multi_axis_tags import tags_for_axes
from rag_fusion import scores_to_ranked
from rag_pipeline import (
    v1_cascade,
    v2_temporal_only,
    v3_semantic_only,
    v4_rrf_all,
    v5_routed_single,
    v6_routed_multi,
    v7_score_blend,
    v8_llm_rerank,
    v9_hybrid_cascade_rrf,
)
from rag_router import RagRouter
from schema import GRANULARITY_ORDER, TimeExpression, parse_iso, to_us
from scorer import Interval, score_jaccard_composite

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
RAG_CACHE_DIR = ROOT / "cache" / "rag"
RAG_CACHE_DIR.mkdir(exist_ok=True, parents=True)

TOP_K = 10
RERANK_TOP_UNION = 20
RERANK_SUBSET_CAP = 30  # V8 LLM-rerank query cap per budget


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
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
    (
        "adversarial",
        "adversarial_queries.jsonl",
        "adversarial_docs.jsonl",
        "adversarial_gold.jsonl",
    ),
]


def load_all() -> tuple[
    dict[str, list[dict]], list[dict], dict[str, set[str]], dict[str, dict]
]:
    """Returns:
    - subset_to_queries: subset_name -> list of query dicts
    - all_docs: combined unique doc list
    - all_gold: query_id -> relevant_doc_ids set
    - query_meta: query_id -> dict with subset, allen_relation, allen_anchor_span
    """
    subset_to_queries: dict[str, list[dict]] = {}
    all_docs: list[dict] = []
    seen_doc_ids: set[str] = set()
    all_gold: dict[str, set[str]] = {}
    query_meta: dict[str, dict] = {}

    for subset, qf, df, gf in SUBSET_FILES:
        queries = load_jsonl(DATA_DIR / qf)
        docs = load_jsonl(DATA_DIR / df)
        gold = load_jsonl(DATA_DIR / gf)
        subset_to_queries[subset] = queries
        for d in docs:
            if d["doc_id"] in seen_doc_ids:
                continue
            seen_doc_ids.add(d["doc_id"])
            all_docs.append(d)
        for g in gold:
            qid = g["query_id"]
            rel = set(g.get("relevant_doc_ids") or [])
            all_gold[qid] = rel
        for q in queries:
            qid = q["query_id"]
            meta = {"subset": subset}
            if "relation" in q:
                meta["allen_relation"] = q["relation"]
                meta["allen_anchor_span"] = q.get("anchor_span")
                meta["allen_anchor_id"] = q.get("anchor_id")
            query_meta[qid] = meta

    return subset_to_queries, all_docs, all_gold, query_meta


# ---------------------------------------------------------------------------
# TimeExpression helpers for retrieval (copied patterns from multi_axis_eval)
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
    return out


def build_time_memory(tes: list[TimeExpression]) -> dict[str, Any]:
    intervals: list[Interval] = []
    axes_per: list[dict[str, AxisDistribution]] = []
    multi_tags: set[str] = set()
    for te in tes:
        intervals.extend(flatten_query_intervals(te))
        ax = axes_for_expression(te)
        axes_per.append(ax)
        multi_tags |= tags_for_axes(ax)
    axes_merged = merge_axis_dists(axes_per)
    return {
        "intervals": intervals,
        "axes_merged": axes_merged,
        "multi_tags": multi_tags,
    }


def _empty_memory() -> dict[str, Any]:
    return {
        "intervals": [],
        "axes_merged": {
            a: AxisDistribution(axis=a, values={}, informative=False) for a in AXES
        },
        "multi_tags": set(),
    }


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


def multi_axis_scores(
    q_mem: dict[str, Any],
    doc_mems: dict[str, dict[str, Any]],
    alpha: float = 0.5,
    beta: float = 0.35,
    gamma: float = 0.15,
) -> dict[str, float]:
    qa = q_mem["axes_merged"]
    q_multi_tags = q_mem["multi_tags"]
    q_ivs = q_mem["intervals"]
    raw_iv: dict[str, float] = {}
    for doc_id, bundle in doc_mems.items():
        raw_iv[doc_id] = interval_pair_best(q_ivs, bundle["intervals"])
    max_iv = max(raw_iv.values(), default=0.0) if raw_iv else 0.0

    scores: dict[str, float] = {}
    for doc_id, bundle in doc_mems.items():
        iv_norm = raw_iv[doc_id] / max_iv if max_iv > 0 else 0.0
        a_sc = axis_score(qa, bundle["axes_merged"])
        t_sc = tag_score(q_multi_tags, bundle["multi_tags"])
        scores[doc_id] = alpha * iv_norm + beta * a_sc + gamma * t_sc
    return scores


def semantic_scores(
    q_emb: np.ndarray, doc_embs: dict[str, np.ndarray]
) -> dict[str, float]:
    qn = float(np.linalg.norm(q_emb)) or 1e-9
    out: dict[str, float] = {}
    for d, v in doc_embs.items():
        vn = float(np.linalg.norm(v)) or 1e-9
        out[d] = float(np.dot(q_emb, v) / (qn * vn))
    return out


# ---------------------------------------------------------------------------
# Era scoring: temporal-Jaccard on era-extracted windows
# ---------------------------------------------------------------------------
def te_window_us(te: TimeExpression) -> tuple[int, int] | None:
    if te.kind == "instant" and te.instant:
        return to_us(te.instant.earliest), to_us(te.instant.latest)
    if te.kind == "interval" and te.interval:
        return to_us(te.interval.start.earliest), to_us(te.interval.end.latest)
    return None


def era_scores(
    q_era_tes: list[TimeExpression],
    doc_era_tes: dict[str, list[TimeExpression]],
) -> dict[str, float]:
    """Era channel: temporal Jaccard between query era windows and doc era windows.

    Only fires if the query has at least one era-type TimeExpression.
    """
    q_windows = [te_window_us(t) for t in q_era_tes]
    q_windows = [w for w in q_windows if w is not None and w[1] > w[0]]
    if not q_windows:
        return {}
    out: dict[str, float] = {}
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


# ---------------------------------------------------------------------------
# Allen scoring (only fires on queries with relation+anchor)
# ---------------------------------------------------------------------------
def allen_scores_for_query(
    q_meta: dict,
    exprs_by_doc: dict,
    resolver_anchor_cache: dict[str, Any],
    resolver_anchor_doc: dict[str, str],
) -> dict[str, float]:
    relation = q_meta.get("allen_relation")
    span = q_meta.get("allen_anchor_span")
    if not relation or not span:
        return {}
    anchor_iv = resolver_anchor_cache.get(span)
    if anchor_iv is None:
        return {}
    anchor_doc_id = resolver_anchor_doc.get(span)

    # Build a synthetic TimeExpression for anchor (only needs te_interval
    # to work, which inspects instant/interval). We can construct a fake
    # instant from the _Iv directly by building a dict.
    from datetime import datetime

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
            earliest=earliest,
            latest=latest,
            best=None,
            granularity="day",
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


# ---------------------------------------------------------------------------
# LLM reranker for V8
# ---------------------------------------------------------------------------
RERANK_SYSTEM = """You rerank documents for relevance to a query about
time-structured memories.

Given a query and a list of candidate docs (each {id, text}), output the
doc IDs in order from MOST relevant to LEAST. Consider temporal match
(explicit dates, named eras, recurrences, relational cues), not just topic.

Output JSON: {"ranked_ids": ["d1", "d2", ...]}. Include every given id
exactly once.
"""


async def llm_rerank(
    llm: LLMCaller,
    query: str,
    candidates: list[tuple[str, str]],
) -> list[str]:
    """candidates: [(doc_id, text)]. Returns ranked doc_id list."""
    if not candidates:
        return []
    cand_str = "\n".join(
        f'{{"id": "{cid}", "text": {json.dumps(txt)}}}' for cid, txt in candidates
    )
    user = (
        f'Query: "{query}"\n\n'
        f"Candidates (JSON lines):\n{cand_str}\n\n"
        'Return {"ranked_ids": [...]} with EVERY candidate id exactly once, '
        "most relevant first."
    )
    try:
        raw = await asyncio.wait_for(
            llm.chat(
                RERANK_SYSTEM,
                user,
                json_object=True,
                max_completion_tokens=600,
                cache_tag="rag_rerank_v1",
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        return [c[0] for c in candidates]
    if not raw:
        return [c[0] for c in candidates]
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return [c[0] for c in candidates]
    ids = d.get("ranked_ids") or []
    cand_id_set = {c[0] for c in candidates}
    seen = set()
    ordered = []
    for i in ids:
        if i in cand_id_set and i not in seen:
            ordered.append(i)
            seen.add(i)
    # Append any missing in original order
    for c in candidates:
        if c[0] not in seen:
            ordered.append(c[0])
    return ordered


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


def avg(vs):
    xs = [v for v in vs if not math.isnan(v)]
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(
    per_query_ranking: dict[str, list[str]],
    gold: dict[str, set[str]],
    qids: list[str],
) -> dict[str, float]:
    r5, r10, mr, nd = [], [], [], []
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            continue
        ranked = per_query_ranking.get(qid, [])
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def extract_base_for_corpus(
    items: list[tuple[str, str, datetime]],
    extractor: BaseExtractor,
    label: str,
) -> dict[str, list[TimeExpression]]:
    async def one(iid, text, ref):
        try:
            return iid, await extractor.extract(text, ref)
        except Exception as e:
            print(f"  base extract failed for {iid}: {e}")
            return iid, []

    print(f"  extracting ({label}, n={len(items)})...")
    results = await asyncio.gather(*(one(*it) for it in items))
    return dict(results)


async def extract_era_for_corpus(
    items: list[tuple[str, str, datetime]],
    era_ex,
    label: str,
) -> dict[str, list[TimeExpression]]:
    async def one(iid, text, ref):
        try:
            return iid, await era_ex.extract(text, ref)
        except Exception as e:
            print(f"  era extract failed for {iid}: {e}")
            return iid, []

    print(f"  era-extracting ({label}, n={len(items)})...")
    results = await asyncio.gather(*(one(*it) for it in items))
    return dict(results)


async def extract_allen_for_corpus(
    items: list[tuple[str, str, datetime]],
    allen_ex: AllenExtractor,
    label: str,
) -> dict:
    async def one(iid, text, ref):
        try:
            return iid, await allen_ex.extract(text, ref)
        except Exception as e:
            print(f"  allen extract failed for {iid}: {e}")
            return iid, []

    print(f"  allen-extracting ({label}, n={len(items)})...")
    results = await asyncio.gather(*(one(*it) for it in items))
    return dict(results)


async def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("RAG Integration — 9 Fusion Variants")
    print("=" * 60)

    # ----- Load -----
    subset_to_queries, all_docs, all_gold, query_meta = load_all()
    all_doc_ids = [d["doc_id"] for d in all_docs]
    total_queries = sum(len(qs) for qs in subset_to_queries.values())
    print(
        f"Loaded {len(all_docs)} unique docs across {len(subset_to_queries)} subsets; "
        f"{total_queries} total queries."
    )
    for s, qs in subset_to_queries.items():
        print(f"  {s}: {len(qs)} queries")

    doc_by_id = {d["doc_id"]: d for d in all_docs}

    # ----- Base extraction -----
    print("\n[1/6] Base TimeExpression extraction (multi-axis T channel)...")
    base_ex = BaseExtractor()
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in all_docs]
    all_queries_flat: list[dict] = []
    for qs in subset_to_queries.values():
        all_queries_flat.extend(qs)
    query_items = [
        (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in all_queries_flat
    ]
    base_doc_tes = await extract_base_for_corpus(doc_items, base_ex, "docs")
    base_q_tes = await extract_base_for_corpus(query_items, base_ex, "queries")
    base_ex.cache.save()
    base_cost = (
        base_ex.usage["input"] * 0.25 / 1_000_000
        + base_ex.usage["output"] * 2.0 / 1_000_000
    )
    print(f"  base usage: {base_ex.usage}, cost ${base_cost:.4f}")

    # ----- Era extraction -----
    print("\n[2/6] Era TimeExpression extraction (E channel)...")
    era_llm = LLMCaller(concurrency=10)
    from era_extractor import EraExtractor

    era_ex = EraExtractor(era_llm)
    era_doc_tes = await extract_era_for_corpus(doc_items, era_ex, "docs")
    era_q_tes = await extract_era_for_corpus(query_items, era_ex, "queries")
    era_llm.save()
    era_cost = era_llm.cost_usd()
    print(f"  era usage: {era_llm.usage}, cost ${era_cost:.4f}")

    # ----- Embeddings -----
    print("\n[3/6] Embedding docs + queries (S channel)...")
    embedder = Embedder(concurrency=10)
    doc_texts = [d["text"] for d in all_docs]
    q_texts = [q["text"] for q in all_queries_flat]
    await embedder.embed_batch(doc_texts + q_texts)
    embedder.save()

    doc_embs: dict[str, np.ndarray] = {}
    for d in all_docs:
        doc_embs[d["doc_id"]] = await embedder.embed(d["text"])
    query_embs: dict[str, np.ndarray] = {}
    for q in all_queries_flat:
        query_embs[q["query_id"]] = await embedder.embed(q["text"])
    print(f"  embedded {len(doc_embs)} docs, {len(query_embs)} queries (cached).")

    # ----- Build T memories -----
    print("\nBuilding T memories (multi-axis)...")
    doc_mems: dict[str, dict[str, Any]] = {
        did: build_time_memory(tes) for did, tes in base_doc_tes.items()
    }
    # Ensure every doc id has a memory entry (empty if extractor gave nothing)
    for did in all_doc_ids:
        if did not in doc_mems:
            doc_mems[did] = _empty_memory()
    q_mems: dict[str, dict[str, Any]] = {
        qid: build_time_memory(tes) for qid, tes in base_q_tes.items()
    }
    for q in all_queries_flat:
        if q["query_id"] not in q_mems:
            q_mems[q["query_id"]] = _empty_memory()

    # ----- Allen extraction: only on allen subset docs/queries (limited scope) -----
    print("\n[4/6] Allen extraction (A channel, allen subset only)...")
    allen_queries = subset_to_queries.get("allen", [])
    allen_docs = [
        d
        for d in all_docs
        if d["doc_id"]
        in {d2["doc_id"] for d2 in load_jsonl(DATA_DIR / "allen_docs.jsonl")}
    ]
    allen_ex = AllenExtractor()

    allen_doc_exprs: dict = {}
    if allen_docs and allen_queries:
        doc_items_allen = [
            (d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in allen_docs
        ]
        q_items_allen = [
            (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in allen_queries
        ]
        allen_doc_exprs = await extract_allen_for_corpus(
            doc_items_allen, allen_ex, "allen-docs"
        )
        allen_q_exprs = await extract_allen_for_corpus(
            q_items_allen, allen_ex, "allen-queries"
        )
        allen_ex.save()
        print(f"  allen usage: {allen_ex.usage}, cost ${allen_ex.cost_usd():.4f}")

        # EventResolver over allen docs, then prewarm anchor cache.
        resolver = EventResolver()
        await resolver.index_docs(allen_docs)
        anchor_spans = {q["anchor_span"] for q in allen_queries}
        for aes in allen_doc_exprs.values():
            for ae in aes:
                if ae.anchor and ae.anchor.kind == "event":
                    anchor_spans.add(ae.anchor.span)
        resolver_anchor_cache: dict[str, Any] = {}
        resolver_anchor_doc: dict[str, str] = {}

        async def _prewarm(span: str) -> None:
            e = await resolver.resolve(span)
            if e is not None:
                resolver_anchor_cache[span] = te_interval(e.time)
                resolver_anchor_doc[span] = e.doc_id
            else:
                resolver_anchor_cache[span] = None

        await asyncio.gather(*[_prewarm(s) for s in anchor_spans])
        resolver_cost = resolver.cost_usd()
    else:
        allen_doc_exprs = {}
        resolver_anchor_cache = {}
        resolver_anchor_doc = {}
        resolver_cost = 0.0

    # ----- Router -----
    print("\n[5/6] Routing queries with LLM intent classifier...")
    router_llm = LLMCaller(concurrency=10)
    router = RagRouter(router_llm)
    q_pairs = [(q["query_id"], q["text"]) for q in all_queries_flat]
    router_intents = await router.classify_all(q_pairs)
    router_llm.save()
    router_cost = router_llm.cost_usd()
    print(f"  router usage: {router_llm.usage}, cost ${router_cost:.4f}")

    # ----- Per-query score dicts for all 4 retrievers -----
    print("\nComputing per-query score dicts (T, S, A, E)...")
    all_doc_id_set = set(all_doc_ids)

    per_q_scores: dict[str, dict[str, dict[str, float]]] = {}
    for q in all_queries_flat:
        qid = q["query_id"]
        qm = q_mems.get(qid, _empty_memory())

        # T
        t_s = multi_axis_scores(qm, doc_mems)

        # S
        s_s = semantic_scores(query_embs[qid], doc_embs)

        # A: only for allen queries
        meta = query_meta[qid]
        if meta.get("allen_relation"):
            a_s = allen_scores_for_query(
                meta, allen_doc_exprs, resolver_anchor_cache, resolver_anchor_doc
            )
        else:
            a_s = {}

        # E: only if query has any era TE
        q_era = era_q_tes.get(qid, [])
        if q_era:
            e_s = era_scores(q_era, era_doc_tes)
        else:
            e_s = {}

        per_q_scores[qid] = {"T": t_s, "S": s_s, "A": a_s, "E": e_s}

    # ----- Run variants -----
    print("\n[6/6] Running 9 variants across all queries...")
    variants_ranked: dict[str, dict[str, list[str]]] = {
        "V1_CASCADE": {},
        "V2_TEMPORAL-ONLY": {},
        "V3_SEMANTIC-ONLY": {},
        "V4_RRF-ALL": {},
        "V5_ROUTED-SINGLE": {},
        "V6_ROUTED-MULTI": {},
        "V7_SCORE-BLEND": {},
        "V8_LLM-RERANK": {},
        "V9_HYBRID": {},
    }

    for q in all_queries_flat:
        qid = q["query_id"]
        s = per_q_scores[qid]
        intents = router_intents.get(qid, ["semantic"])
        variants_ranked["V1_CASCADE"][qid] = v1_cascade(s["T"], s["S"], all_doc_ids)
        variants_ranked["V2_TEMPORAL-ONLY"][qid] = v2_temporal_only(s["T"], s["S"])
        variants_ranked["V3_SEMANTIC-ONLY"][qid] = v3_semantic_only(s["S"])
        variants_ranked["V4_RRF-ALL"][qid] = v4_rrf_all(s["T"], s["S"], s["A"], s["E"])
        variants_ranked["V5_ROUTED-SINGLE"][qid] = v5_routed_single(
            intents, s["T"], s["S"], s["A"], s["E"]
        )
        variants_ranked["V6_ROUTED-MULTI"][qid] = v6_routed_multi(
            intents, s["T"], s["S"], s["A"], s["E"]
        )
        variants_ranked["V7_SCORE-BLEND"][qid] = v7_score_blend(
            s["T"], s["S"], s["A"], s["E"]
        )
        variants_ranked["V9_HYBRID"][qid] = v9_hybrid_cascade_rrf(
            intents, s["T"], s["S"], s["A"], s["E"]
        )

    # ----- V8 LLM-RERANK (sample up to RERANK_SUBSET_CAP per subset) -----
    print(f"\nRunning V8 LLM-RERANK (sampled, cap {RERANK_SUBSET_CAP} per subset)...")
    rerank_llm = LLMCaller(concurrency=8)
    v8_sampled_qids: set[str] = set()

    import random

    rng = random.Random(1337)
    for subset, qs in subset_to_queries.items():
        if not qs:
            continue
        # Deterministic sample
        sample = (
            qs if len(qs) <= RERANK_SUBSET_CAP else rng.sample(qs, RERANK_SUBSET_CAP)
        )
        for q in sample:
            v8_sampled_qids.add(q["query_id"])

    async def do_rerank(q):
        qid = q["query_id"]
        s = per_q_scores[qid]
        union_top: list[str] = []
        seen: set[str] = set()
        for scored in (s["T"], s["S"], s["A"], s["E"]):
            ranked = scores_to_ranked(scored)
            for d in ranked[:RERANK_TOP_UNION]:
                if d not in seen:
                    union_top.append(d)
                    seen.add(d)
                if len(union_top) >= 40:  # cap for prompt size
                    break
        candidates = [(d, doc_by_id[d]["text"]) for d in union_top if d in doc_by_id]
        reranked = await llm_rerank(rerank_llm, q["text"], candidates)
        return qid, v8_llm_rerank(reranked, s["S"])

    rerank_results = await asyncio.gather(
        *(do_rerank(q) for q in all_queries_flat if q["query_id"] in v8_sampled_qids)
    )
    for qid, rl in rerank_results:
        variants_ranked["V8_LLM-RERANK"][qid] = rl
    # For non-sampled queries, fall back to SEMANTIC-ONLY (so metrics are
    # computed only where we reranked; we will explicitly scope V8 eval).
    rerank_llm.save()
    rerank_cost = rerank_llm.cost_usd()
    print(f"  rerank usage: {rerank_llm.usage}, cost ${rerank_cost:.4f}")
    print(f"  V8 sampled {len(v8_sampled_qids)} queries out of {len(all_queries_flat)}")

    # ----- Evaluate -----
    print("\nEvaluating variants per subset...")
    variant_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for variant_name, ranked_map in variants_ranked.items():
        variant_metrics[variant_name] = {}
        for subset, qs in subset_to_queries.items():
            qids = [q["query_id"] for q in qs]
            if variant_name == "V8_LLM-RERANK":
                # scope to sampled qids within this subset
                qids = [q for q in qids if q in v8_sampled_qids]
            if not qids:
                variant_metrics[variant_name][subset] = {
                    "recall@5": 0.0,
                    "recall@10": 0.0,
                    "mrr": 0.0,
                    "ndcg@10": 0.0,
                    "n": 0,
                }
                continue
            variant_metrics[variant_name][subset] = evaluate(ranked_map, all_gold, qids)
        # Combined
        all_qids = [q["query_id"] for q in all_queries_flat]
        if variant_name == "V8_LLM-RERANK":
            all_qids = [q for q in all_qids if q in v8_sampled_qids]
        variant_metrics[variant_name]["combined"] = evaluate(
            ranked_map, all_gold, all_qids
        )

    # ----- Cost -----
    total_cost = (
        base_cost
        + era_cost
        + resolver_cost
        + router_cost
        + rerank_cost
        + (allen_ex.cost_usd() if allen_queries else 0.0)
    )
    cost_report = {
        "base_extractor": base_cost,
        "era_extractor": era_cost,
        "allen_extractor": allen_ex.cost_usd() if allen_queries else 0.0,
        "event_resolver": resolver_cost,
        "router": router_cost,
        "rerank_v8": rerank_cost,
        "total": total_cost,
    }

    # ----- LLM call counts per variant (for cost-adjusted ranking) -----
    llm_call_count_per_query = {
        "V1_CASCADE": 0,
        "V2_TEMPORAL-ONLY": 0,
        "V3_SEMANTIC-ONLY": 0,
        "V4_RRF-ALL": 0,
        "V5_ROUTED-SINGLE": 1,  # router
        "V6_ROUTED-MULTI": 1,  # router
        "V7_SCORE-BLEND": 0,
        "V8_LLM-RERANK": 1,  # rerank call
        "V9_HYBRID": 1,  # uses router
    }

    # ----- Router accuracy: compare router intent vs subset-derived intent -----
    def expected_intents(subset: str, q: dict) -> list[str]:
        if subset == "allen":
            return ["relational"]
        if subset == "era":
            return ["era"]
        if subset == "adversarial":
            # adversarial is mixed — treat as semantic expected (no gold intent)
            return ["semantic"]
        if subset == "utterance":
            return ["temporal"]
        if subset == "axis":
            return ["temporal"]
        if subset == "discriminator":
            return ["temporal"]
        if subset == "base":
            return ["temporal"]
        return ["semantic"]

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    router_correct_top1 = 0
    router_correct_in = 0
    router_total = 0
    per_query_intent: dict[str, dict] = {}
    for subset, qs in subset_to_queries.items():
        for q in qs:
            qid = q["query_id"]
            expected = expected_intents(subset, q)
            got = router_intents.get(qid, [])
            per_query_intent[qid] = {
                "subset": subset,
                "expected": expected,
                "got": got,
            }
            router_total += 1
            if got and got[0] in expected:
                router_correct_top1 += 1
            if any(g in expected for g in got):
                router_correct_in += 1
            exp = expected[0] if expected else "semantic"
            top = got[0] if got else "?"
            confusion[exp][top] += 1

    router_acc_top1 = router_correct_top1 / router_total if router_total else 0.0
    router_acc_any = router_correct_in / router_total if router_total else 0.0

    # ----- Output JSON -----
    results = {
        "variant_metrics": variant_metrics,
        "v8_sampled_count": len(v8_sampled_qids),
        "cost": cost_report,
        "llm_calls_per_query": llm_call_count_per_query,
        "router": {
            "top1_accuracy": router_acc_top1,
            "any_match_accuracy": router_acc_any,
            "confusion": {k: dict(v) for k, v in confusion.items()},
            "per_query": per_query_intent,
        },
        "subset_sizes": {k: len(v) for k, v in subset_to_queries.items()},
        "total_queries": total_queries,
        "total_docs": len(all_docs),
        "wall_time_s": time.time() - t0,
    }
    (RESULTS_DIR / "rag_integration.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    # ----- Build markdown report -----
    variant_order = [
        "V1_CASCADE",
        "V2_TEMPORAL-ONLY",
        "V3_SEMANTIC-ONLY",
        "V4_RRF-ALL",
        "V5_ROUTED-SINGLE",
        "V6_ROUTED-MULTI",
        "V7_SCORE-BLEND",
        "V8_LLM-RERANK",
        "V9_HYBRID",
    ]
    subset_order = [
        "base",
        "discriminator",
        "utterance",
        "era",
        "axis",
        "allen",
        "adversarial",
        "combined",
    ]

    md: list[str] = []
    md.append("# RAG Integration — 9 Fusion Variants\n\n")
    md.append(
        f"Docs: {len(all_docs)}. Queries: {total_queries} "
        f"({', '.join(f'{k}={len(v)}' for k, v in subset_to_queries.items())}).\n\n"
    )
    md.append(
        f"Wall time: {results['wall_time_s']:.1f}s. Total LLM cost: ${cost_report['total']:.4f}.\n\n"
    )
    md.append(
        f"V8 LLM-RERANK sampled {len(v8_sampled_qids)} queries (cap "
        f"{RERANK_SUBSET_CAP}/subset).\n\n"
    )

    # Per-variant per-subset R@5 table
    md.append("## Recall@5 by variant × subset\n\n")
    md.append("| Variant | " + " | ".join(subset_order) + " |\n")
    md.append("|---|" + "|".join(["---:"] * len(subset_order)) + "|\n")
    for v in variant_order:
        row = [v]
        for s in subset_order:
            m = variant_metrics[v].get(s, {})
            r5 = m.get("recall@5", 0.0)
            row.append(f"{r5:.3f}")
        md.append("| " + " | ".join(row) + " |\n")

    md.append("\n## NDCG@10 by variant × subset\n\n")
    md.append("| Variant | " + " | ".join(subset_order) + " |\n")
    md.append("|---|" + "|".join(["---:"] * len(subset_order)) + "|\n")
    for v in variant_order:
        row = [v]
        for s in subset_order:
            m = variant_metrics[v].get(s, {})
            row.append(f"{m.get('ndcg@10', 0.0):.3f}")
        md.append("| " + " | ".join(row) + " |\n")

    md.append("\n## MRR by variant × subset\n\n")
    md.append("| Variant | " + " | ".join(subset_order) + " |\n")
    md.append("|---|" + "|".join(["---:"] * len(subset_order)) + "|\n")
    for v in variant_order:
        row = [v]
        for s in subset_order:
            m = variant_metrics[v].get(s, {})
            row.append(f"{m.get('mrr', 0.0):.3f}")
        md.append("| " + " | ".join(row) + " |\n")

    # Per-subset winners
    md.append("\n## Per-subset winner (by R@5, ties broken by NDCG@10)\n\n")
    md.append("| Subset | Best variant | R@5 | NDCG@10 | MRR |\n")
    md.append("|---|---|---:|---:|---:|\n")
    for s in subset_order:
        best_v = max(
            variant_order,
            key=lambda v: (
                variant_metrics[v].get(s, {}).get("recall@5", 0.0),
                variant_metrics[v].get(s, {}).get("ndcg@10", 0.0),
            ),
        )
        m = variant_metrics[best_v].get(s, {})
        md.append(
            f"| {s} | {best_v} | {m.get('recall@5', 0):.3f} | "
            f"{m.get('ndcg@10', 0):.3f} | {m.get('mrr', 0):.3f} |\n"
        )

    # Cost-adjusted ranking: combined R@5 / (1 + LLM calls per query)
    md.append("\n## Cost-adjusted ranking (R@5 / (1 + LLM_calls))\n\n")
    md.append("| Variant | Combined R@5 | LLM calls/query | R@5 / (1+calls) |\n")
    md.append("|---|---:|---:|---:|\n")
    cost_adj = []
    for v in variant_order:
        r5 = variant_metrics[v]["combined"].get("recall@5", 0.0)
        calls = llm_call_count_per_query[v]
        cost_adj.append((v, r5, calls, r5 / (1.0 + calls)))
    cost_adj.sort(key=lambda r: r[3], reverse=True)
    for v, r5, c, eff in cost_adj:
        md.append(f"| {v} | {r5:.3f} | {c} | {eff:.3f} |\n")

    # Router confusion
    md.append("\n## Router accuracy\n\n")
    md.append(
        f"- Top-1 accuracy: {router_acc_top1:.1%} ({router_correct_top1}/{router_total})\n"
    )
    md.append(
        f"- Any-match accuracy: {router_acc_any:.1%} ({router_correct_in}/{router_total})\n\n"
    )
    md.append("### Confusion matrix (rows=expected, cols=predicted top-1)\n\n")
    all_intents = sorted(
        {"temporal", "semantic", "relational", "era", "?"}
        | set(k for c in confusion.values() for k in c.keys())
    )
    md.append("| expected\\pred | " + " | ".join(all_intents) + " |\n")
    md.append("|---|" + "|".join(["---:"] * len(all_intents)) + "|\n")
    for e in sorted(confusion.keys()):
        row = [e]
        for p in all_intents:
            row.append(str(confusion[e].get(p, 0)))
        md.append("| " + " | ".join(row) + " |\n")

    # Biggest improvement over CASCADE
    md.append("\n## Improvement vs V1 CASCADE (baseline)\n\n")
    md.append("| Variant | Combined ΔR@5 | Combined ΔNDCG@10 |\n")
    md.append("|---|---:|---:|\n")
    base_r5 = variant_metrics["V1_CASCADE"]["combined"]["recall@5"]
    base_nd = variant_metrics["V1_CASCADE"]["combined"]["ndcg@10"]
    for v in variant_order:
        dr = variant_metrics[v]["combined"]["recall@5"] - base_r5
        dn = variant_metrics[v]["combined"]["ndcg@10"] - base_nd
        md.append(f"| {v} | {dr:+.3f} | {dn:+.3f} |\n")

    # Cheap vs LLM-RERANK gap (on V8 sampled subset)
    md.append("\n## Cheap variants vs V8 LLM-RERANK (same sampled queries)\n\n")
    # Evaluate ALL variants on the V8 sampled subset for fair comparison
    md.append("| Variant | Sampled R@5 | Sampled NDCG@10 |\n")
    md.append("|---|---:|---:|\n")
    sampled_qids = [
        q["query_id"] for q in all_queries_flat if q["query_id"] in v8_sampled_qids
    ]
    for v in variant_order:
        m = evaluate(variants_ranked[v], all_gold, sampled_qids)
        md.append(f"| {v} | {m['recall@5']:.3f} | {m['ndcg@10']:.3f} |\n")

    # Ship recommendation
    md.append("\n## Ship recommendation\n\n")
    combined_r5 = [
        (v, variant_metrics[v]["combined"]["recall@5"]) for v in variant_order
    ]
    combined_r5.sort(key=lambda r: r[1], reverse=True)
    cheapest_top2 = [v for v, _ in combined_r5[:3] if llm_call_count_per_query[v] == 0]
    cheap_winner = cheapest_top2[0] if cheapest_top2 else combined_r5[0][0]
    top_overall = combined_r5[0][0]
    top_overall_calls = llm_call_count_per_query[top_overall]

    if top_overall_calls == 0:
        ship = top_overall
        md.append(
            f"**Ship {top_overall}.** Combined R@5 {combined_r5[0][1]:.3f} "
            f"with **0 LLM calls per query** at retrieval time (extraction cost amortized).\n\n"
        )
    else:
        md.append(
            f"**Default:** {cheap_winner} (R@5 "
            f"{variant_metrics[cheap_winner]['combined']['recall@5']:.3f}, 0 LLM calls/query).\n\n"
            f"**Fallback:** {top_overall} when router emits low-confidence or the default's top-3 "
            f"show low temporal signal.\n\n"
        )

    # Cost
    md.append("## Cost\n\n")
    md.append(f"- Base extractor: ${cost_report['base_extractor']:.4f}\n")
    md.append(f"- Era extractor: ${cost_report['era_extractor']:.4f}\n")
    md.append(f"- Allen extractor: ${cost_report['allen_extractor']:.4f}\n")
    md.append(f"- Event resolver: ${cost_report['event_resolver']:.4f}\n")
    md.append(f"- Router: ${cost_report['router']:.4f}\n")
    md.append(f"- V8 rerank: ${cost_report['rerank_v8']:.4f}\n")
    md.append(f"- **Total**: ${cost_report['total']:.4f}\n")

    (RESULTS_DIR / "rag_integration.md").write_text("".join(md))

    # Console summary
    print("\n=== Per-subset R@5 ===")
    print(f"{'Variant':<20}" + "".join(f"{s:>14}" for s in subset_order))
    for v in variant_order:
        row = f"{v:<20}"
        for s in subset_order:
            r5 = variant_metrics[v].get(s, {}).get("recall@5", 0.0)
            row += f"{r5:>14.3f}"
        print(row)

    print(f"\nRouter top-1 acc: {router_acc_top1:.1%}")
    print(f"Total LLM cost: ${cost_report['total']:.4f}")
    print(f"Wall time: {results['wall_time_s']:.1f}s")
    print("\nWrote results/rag_integration.{md,json}")


if __name__ == "__main__":
    asyncio.run(main())
