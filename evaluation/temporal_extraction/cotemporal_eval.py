"""Co-temporal graph-expansion retrieval evaluation.

Compares direct (V7 SCORE-BLEND) vs cotemporal-expansion retrieval on:
- base 55 queries
- adversarial full set
- adversarial S8 subset (cross-doc linking)
- new cotemporal queries (20, designed for co-mention traversal)

Reuses extraction caches (cache/adversarial_v2pp/, cache/extractor_v2p/,
etc). Extracts the new synthetic docs with v2pp + era (reusing pass-2
cache where possible).

Outputs: results/cotemporal.{json,md}
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from advanced_common import LLMCaller
from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from baselines import embed_all
from cotemporal_graph import CotemporalGraph, DocTemporalBundle
from cotemporal_retrieval import cotemporal_rerank
from era_extractor import EraExtractor
from expander import expand
from extractor_common import LLMCache
from extractor_v2pp import ExtractorV2PP
from modality_schema import get_modality
from multi_axis_scorer import axis_score as axis_score_fn
from multi_axis_scorer import tag_score
from multi_axis_tags import tags_for_axes
from openai import AsyncOpenAI
from rag_pipeline import v7_score_blend
from schema import GRANULARITY_ORDER, TimeExpression, parse_iso, to_us
from scorer import Interval, score_jaccard_composite

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR = ROOT / "cache" / "cotemporal"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

LLM_CALL_TIMEOUT_S = 30.0
CALL_TIMEOUT_S = 180.0

TOP_K = 10

ALPHA_IV = 0.5
BETA_AXIS = 0.35
GAMMA_TAG = 0.15


def _patched_client() -> AsyncOpenAI:
    return AsyncOpenAI(timeout=LLM_CALL_TIMEOUT_S, max_retries=1)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
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


def _empty_mem() -> dict:
    return {
        "intervals": [],
        "axes_merged": {
            a: AxisDistribution(axis=a, values={}, informative=False) for a in AXES
        },
        "multi_tags": set(),
    }


def build_mem(tes: list[TimeExpression]) -> dict:
    ivs = []
    axes_per = []
    tags: set[str] = set()
    for te in tes:
        if get_modality(te) != "actual":
            continue
        ivs.extend(flatten_intervals(te))
        ax = axes_for_expression(te)
        axes_per.append(ax)
        tags |= tags_for_axes(ax)
    axes_merged = (
        merge_axis_dists(axes_per) if axes_per else _empty_mem()["axes_merged"]
    )
    return {"intervals": ivs, "axes_merged": axes_merged, "multi_tags": tags}


def interval_pair_best(q, d):
    if not q or not d:
        return 0.0
    t = 0.0
    for qi in q:
        best = 0.0
        for si in d:
            s = score_jaccard_composite(qi, si)
            if s > best:
                best = s
        t += best
    return t


def multi_axis_scores(
    qm: dict,
    dms: dict[str, dict],
    alpha: float = ALPHA_IV,
    beta: float = BETA_AXIS,
    gamma: float = GAMMA_TAG,
) -> dict[str, float]:
    qa = qm["axes_merged"]
    qtg = qm["multi_tags"]
    qivs = qm["intervals"]
    raw = {d: interval_pair_best(qivs, b["intervals"]) for d, b in dms.items()}
    mx = max(raw.values(), default=0.0)
    out = {}
    for d, b in dms.items():
        iv = raw[d] / mx if mx > 0 else 0.0
        ax = axis_score_fn(qa, b["axes_merged"])
        tg = tag_score(qtg, b["multi_tags"])
        out[d] = alpha * iv + beta * ax + gamma * tg
    return out


def semantic_scores(qe, des):
    qn = float(np.linalg.norm(qe)) or 1e-9
    out = {}
    for d, v in des.items():
        vn = float(np.linalg.norm(v)) or 1e-9
        out[d] = float(np.dot(qe, v) / (qn * vn))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def r_at_k(r, rel, k):
    if not rel:
        return float("nan")
    return len(set(r[:k]) & rel) / len(rel)


def mrr(r, rel):
    if not rel:
        return float("nan")
    for i, d in enumerate(r, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg(r, rel, k):
    if not rel:
        return float("nan")
    dcg = 0.0
    for i, d in enumerate(r[:k], 1):
        if d in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def avg(vs):
    xs = [v for v in vs if not math.isnan(v)]
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(
    ranked_per_q: dict[str, list[str]], gold: dict[str, set], qids: list[str]
) -> dict:
    r5, r10, mr, nd = [], [], [], []
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            continue
        r = ranked_per_q.get(qid, [])
        r5.append(r_at_k(r, rel, 5))
        r10.append(r_at_k(r, rel, 10))
        mr.append(mrr(r, rel))
        nd.append(ndcg(r, rel, 10))
    return {
        "n": len(r5),
        "recall@5": avg(r5),
        "recall@10": avg(r10),
        "mrr": avg(mr),
        "ndcg@10": avg(nd),
    }


# ---------------------------------------------------------------------------
# Extraction (v2pp + era)
# ---------------------------------------------------------------------------
async def extract_v2pp(items, label, cache_subdir):
    ex = ExtractorV2PP(concurrency=8)
    ex.cache = LLMCache(cache_subdir / "llm_cache.json")
    ex.client = _patched_client()

    async def one(iid, text, ref):
        try:
            tes = await asyncio.wait_for(ex.extract(text, ref), timeout=CALL_TIMEOUT_S)
            return iid, tes
        except asyncio.TimeoutError:
            print(f"  [{label}] TIMEOUT {iid}", flush=True)
            return iid, []
        except Exception as e:
            print(f"  [{label}] failed {iid}: {e}", flush=True)
            return iid, []

    print(f"  v2pp-extracting {label} ({len(items)})...", flush=True)
    pairs = await asyncio.gather(*(one(*it) for it in items))
    ex.cache.save()
    ex.shared_pass2_cache.save()
    return dict(pairs), ex.usage


async def extract_era(items, label):
    llm = LLMCaller(concurrency=8)
    llm.client = _patched_client()
    ex = EraExtractor(llm)

    async def one(iid, text, ref):
        try:
            tes = await asyncio.wait_for(ex.extract(text, ref), timeout=CALL_TIMEOUT_S)
            return iid, tes
        except asyncio.TimeoutError:
            return iid, []
        except Exception:
            return iid, []

    print(f"  era-extracting {label} ({len(items)})...", flush=True)
    pairs = await asyncio.gather(*(one(*it) for it in items))
    llm.save()
    return dict(pairs), llm.usage


def merge_tes(a: list[TimeExpression], b: list[TimeExpression]) -> list[TimeExpression]:
    seen = set()
    merged = []
    for te in list(a) + list(b):
        key = (te.kind, (te.surface or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(te)
    return merged


# ---------------------------------------------------------------------------
# Direct retrieval: replicate the V7 SCORE-BLEND used in adversarial_v2pp_eval.
# Here we do: T = multi-axis-only (no allen/era-specific separate channel
# beyond what v2pp+era extraction provides), S = semantic cosine.
# ---------------------------------------------------------------------------
def v7_direct_rank(
    qid: str,
    q_mem: dict,
    doc_mems: dict[str, dict],
    q_emb,
    doc_embs: dict[str, np.ndarray],
    all_doc_ids: list[str],
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Return (t_scores, s_scores, ranked_list)."""
    t = multi_axis_scores(q_mem, doc_mems)
    s = semantic_scores(q_emb, doc_embs)
    # Use V7 with empty a/e channels.
    ranked = v7_score_blend(
        t, s, {}, {}, weights={"T": 0.4, "S": 0.4, "A": 0.1, "E": 0.1}
    )
    return t, s, ranked


def build_scoring_dict_from_ranked(ranked: list[str]) -> dict[str, float]:
    """Monotonic positional score 1..0 for ranked list."""
    n = len(ranked)
    if n == 0:
        return {}
    return {d: (n - i) / n for i, d in enumerate(ranked)}


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
async def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("Cotemporal retrieval evaluation")
    print("=" * 70)

    # Base, adversarial, new cotemporal. We build a UNIFIED corpus that
    # includes all docs and run all queries over it.
    base_docs = load_jsonl(DATA_DIR / "docs.jsonl")
    base_qs = load_jsonl(DATA_DIR / "queries.jsonl")
    base_gold = load_jsonl(DATA_DIR / "gold.jsonl")

    adv_docs = load_jsonl(DATA_DIR / "adversarial_docs.jsonl")
    adv_qs = load_jsonl(DATA_DIR / "adversarial_queries.jsonl")
    adv_gold = load_jsonl(DATA_DIR / "adversarial_gold.jsonl")

    cot_docs = load_jsonl(DATA_DIR / "cotemporal_docs.jsonl")
    cot_qs = load_jsonl(DATA_DIR / "cotemporal_queries.jsonl")
    cot_gold = load_jsonl(DATA_DIR / "cotemporal_gold.jsonl")

    # Unify: union of doc_ids across all three
    all_docs_map: dict[str, dict] = {}
    for d in base_docs + adv_docs + cot_docs:
        all_docs_map.setdefault(d["doc_id"], d)
    all_docs = list(all_docs_map.values())

    # Queries (keep all separate via subset labels)
    def mark(qs, subset):
        return [{**q, "_subset": subset} for q in qs]

    all_qs = (
        mark(base_qs, "base") + mark(adv_qs, "adversarial") + mark(cot_qs, "cotemporal")
    )

    # Gold map
    gold_map: dict[str, set] = {}
    for g in base_gold + adv_gold + cot_gold:
        gold_map[g["query_id"]] = set(g.get("relevant_doc_ids") or [])

    print(f"Corpus: {len(all_docs)} docs, {len(all_qs)} queries")

    # --------- Extraction ---------
    print("\n[1/5] Extraction (v2pp + era), reusing caches where possible...")
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in all_docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in all_qs]

    v2pp_cache = CACHE_DIR / "extractor_v2pp_pass1"
    v2pp_cache.mkdir(exist_ok=True, parents=True)

    doc_v2pp, u1 = await extract_v2pp(doc_items, "docs-v2pp", v2pp_cache)
    q_v2pp, u2 = await extract_v2pp(q_items, "queries-v2pp", v2pp_cache)
    doc_era, u3 = await extract_era(doc_items, "docs-era")
    q_era, u4 = await extract_era(q_items, "queries-era")

    doc_tes = {
        did: merge_tes(doc_v2pp.get(did, []), doc_era.get(did, []))
        for did in [d["doc_id"] for d in all_docs]
    }
    q_tes = {
        qid: merge_tes(q_v2pp.get(qid, []), q_era.get(qid, []))
        for qid in [q["query_id"] for q in all_qs]
    }

    # --------- Memories ---------
    print("[2/5] Building memories...")
    doc_mems = {did: build_mem(tes) for did, tes in doc_tes.items()}
    for d in all_docs:
        doc_mems.setdefault(d["doc_id"], _empty_mem())
    q_mems = {qid: build_mem(tes) for qid, tes in q_tes.items()}
    for q in all_qs:
        q_mems.setdefault(q["query_id"], _empty_mem())

    # --------- Embeddings ---------
    print("[3/5] Embeddings...")
    doc_texts = [d["text"] for d in all_docs]
    q_texts = [q["text"] for q in all_qs]
    dea = await embed_all(doc_texts)
    qea = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: dea[i] for i, d in enumerate(all_docs)}
    q_embs = {q["query_id"]: qea[i] for i, q in enumerate(all_qs)}

    # --------- Co-temporal graph ---------
    print("[4/5] Building cotemporal graph...")
    graph_db = CACHE_DIR / "graph.sqlite"
    if graph_db.exists():
        graph_db.unlink()
    graph = CotemporalGraph(graph_db)
    # Use the same filtered-by-modality doc TEs.
    doc_bundles = {}
    for did in [d["doc_id"] for d in all_docs]:
        # Filter non-actual for graph construction too
        actual_tes = [te for te in doc_tes.get(did, []) if get_modality(te) == "actual"]
        doc_bundles[did] = DocTemporalBundle(
            doc_id=did,
            intervals=[iv for te in actual_tes for iv in flatten_intervals(te)],
            axes_merged=doc_mems[did]["axes_merged"],
            multi_tags=doc_mems[did]["multi_tags"],
        )
    graph_stats = graph.build(
        doc_bundles,
        threshold=0.3,
        cap_per_node=20,
        doc_embs=doc_embs,
        sem_fallback_top_k=5,
        sem_fallback_min_cos=0.45,
    )
    print(
        f"  Graph: {graph_stats['nodes']} nodes, "
        f"{graph_stats['edges_undirected']} undirected edges "
        f"(directed {graph_stats['edges_directed']}), "
        f"avg_deg={graph_stats['avg_degree']:.2f}, "
        f"max_deg={graph_stats['max_degree']}, "
        f"sem_bridge_kept={graph_stats['sem_bridge_edges_kept']}"
    )

    # --------- Retrieval ---------
    print("[5/5] Retrieval: direct V7 vs V7 + cotemporal expansion...")
    all_doc_ids = [d["doc_id"] for d in all_docs]
    direct_ranked: dict[str, list[str]] = {}
    cot_ranked: dict[str, list[str]] = {}
    diag_by_q: dict[str, dict] = {}

    from rag_fusion import score_blend as _score_blend

    for q in all_qs:
        qid = q["query_id"]
        qm = q_mems.get(qid, _empty_mem())
        qemb = q_embs[qid]
        t_scores = multi_axis_scores(qm, doc_mems)
        s_scores = semantic_scores(qemb, doc_embs)

        # V7 direct: score_blend with T, S (empty A, E)
        per_ret = {"T": t_scores, "S": s_scores, "A": {}, "E": {}}
        v7_scored_list = _score_blend(
            per_ret, weights={"T": 0.4, "S": 0.4, "A": 0.1, "E": 0.1}, top_k_per=40
        )
        v7_scored_dict = dict(v7_scored_list)
        # Append missing from s_scores (semantic tail) with zero so graph
        # expansion can still reach them.
        v7_ranked = [d for d, _ in v7_scored_list]
        seen = set(v7_ranked)
        for d in sorted(s_scores.items(), key=lambda x: x[1], reverse=True):
            if d[0] not in seen:
                v7_ranked.append(d[0])
        direct_ranked[qid] = v7_ranked

        # Cotemporal rerank: seed the graph expansion with V7's own scored
        # output (preserving V7 shape), blend β*expansion + γ*semantic as
        # additive perturbation.
        ranked_with, diag = cotemporal_rerank(
            direct_scores=v7_scored_dict,
            semantic_scores=s_scores,
            graph=graph,
            all_doc_ids=all_doc_ids,
            use_cotemporal=True,
            K_seed=20,
            M_neighbors=10,
            alpha=0.6,
            beta=0.25,
            gamma=0.15,
            decay=0.5,
        )
        cot_ranked_list = [d for d, _ in ranked_with]
        # Append any remaining docs (by semantic) so full ordering exists
        seen2 = set(cot_ranked_list)
        for d, _ in sorted(s_scores.items(), key=lambda x: x[1], reverse=True):
            if d not in seen2:
                cot_ranked_list.append(d)
        cot_ranked[qid] = cot_ranked_list
        diag_by_q[qid] = diag

    # --------- Subset evaluations ---------
    subsets = {
        "base": [q["query_id"] for q in base_qs],
        "adversarial_full": [q["query_id"] for q in adv_qs],
        "adversarial_S8": [q["query_id"] for q in adv_qs if q.get("category") == "S8"],
        "cotemporal": [q["query_id"] for q in cot_qs],
    }

    results = {}
    for subset, qids in subsets.items():
        results[subset] = {
            "n": len(qids),
            "direct_v7": evaluate(direct_ranked, gold_map, qids),
            "cotemporal_v7_plus_graph": evaluate(cot_ranked, gold_map, qids),
        }

    # --------- Expansion source breakdown (cotemporal subset) ---------
    breakdown = {"direct_or_mixed_at5": 0, "expansion_only_at5": 0, "total": 0}
    for qid in subsets["cotemporal"]:
        diag = diag_by_q.get(qid, {})
        top5 = cot_ranked[qid][:5]
        for d in top5:
            breakdown["total"] += 1
            if diag.get(d, {}).get("via_expansion_only"):
                breakdown["expansion_only_at5"] += 1
            else:
                breakdown["direct_or_mixed_at5"] += 1

    # Per-query cotemporal subset detail
    cot_per_query = []
    for q in cot_qs:
        qid = q["query_id"]
        rel = gold_map.get(qid, set())
        d_top5 = direct_ranked[qid][:5]
        c_top5 = cot_ranked[qid][:5]
        d_r5 = r_at_k(direct_ranked[qid], rel, 5)
        c_r5 = r_at_k(cot_ranked[qid], rel, 5)
        cot_per_query.append(
            {
                "qid": qid,
                "text": q["text"],
                "gold": sorted(rel),
                "direct_top5": d_top5,
                "cot_top5": c_top5,
                "direct_r5": d_r5,
                "cot_r5": c_r5,
                "delta_r5": c_r5 - d_r5,
            }
        )

    # --------- S8 recovery breakdown ---------
    s8_detail = []
    for qid in subsets["adversarial_S8"]:
        rel = gold_map.get(qid, set())
        direct_hit = rel & set(direct_ranked[qid][:5])
        cot_hit = rel & set(cot_ranked[qid][:5])
        s8_detail.append(
            {
                "qid": qid,
                "gold": sorted(rel),
                "direct_top5": direct_ranked[qid][:5],
                "cot_top5": cot_ranked[qid][:5],
                "direct_hit": sorted(direct_hit),
                "cot_hit": sorted(cot_hit),
            }
        )

    # --------- Base regression: per-query diff in R@5 ---------
    regressions = []
    for qid in subsets["base"]:
        rel = gold_map.get(qid, set())
        if not rel:
            continue
        d_r5 = r_at_k(direct_ranked[qid], rel, 5)
        c_r5 = r_at_k(cot_ranked[qid], rel, 5)
        if c_r5 < d_r5:
            regressions.append(
                {"qid": qid, "direct": d_r5, "cot": c_r5, "diff": c_r5 - d_r5}
            )

    # --------- Topic-drift diagnosis ---------
    drift_ids = {d["doc_id"] for d in cot_docs if d.get("category") == "COT_DRIFT"}
    # How many expansion candidates (across all queries) land on drift docs
    drift_hit_count = 0
    exp_cand_total = 0
    drift_in_top10 = 0
    drift_in_top10_total = 0
    for qid in subsets["cotemporal"]:
        diag = diag_by_q.get(qid, {})
        # candidate = docs with expansion > 0
        for d, info in diag.items():
            if info["expansion"] > 0:
                exp_cand_total += 1
                if d in drift_ids:
                    drift_hit_count += 1
        for d in cot_ranked[qid][:10]:
            drift_in_top10_total += 1
            if d in drift_ids:
                drift_in_top10 += 1

    # --------- Cost ---------
    usages = [u1, u2, u3, u4]
    total_in = sum(u.get("input", 0) for u in usages)
    total_out = sum(u.get("output", 0) for u in usages)
    cost_usd = total_in * 0.25 / 1_000_000 + total_out * 2.0 / 1_000_000

    wall_s = time.time() - t0

    # --------- Output ---------
    out = {
        "corpus": {
            "n_docs": len(all_docs),
            "n_queries": len(all_qs),
        },
        "graph_stats": graph_stats,
        "subsets": results,
        "cot_expansion_source_top5": breakdown,
        "cot_per_query_detail": cot_per_query,
        "adversarial_S8_detail": s8_detail,
        "base_regressions_count": len(regressions),
        "base_regressions": regressions[:20],
        "topic_drift": {
            "drift_doc_count": len(drift_ids),
            "expansion_candidates_total": exp_cand_total,
            "expansion_candidates_on_drift": drift_hit_count,
            "drift_fraction_of_candidates": (drift_hit_count / exp_cand_total)
            if exp_cand_total
            else 0.0,
            "drift_in_top10_total_slots": drift_in_top10_total,
            "drift_in_top10_slots_with_drift": drift_in_top10,
        },
        "cost": {"input_tokens": total_in, "output_tokens": total_out, "usd": cost_usd},
        "wall_seconds": wall_s,
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, tuple):
            return [_clean(v) for v in o]
        if isinstance(o, set):
            return sorted(_clean(v) for v in o)
        if isinstance(o, float) and math.isnan(o):
            return None
        return o

    (RESULTS_DIR / "cotemporal.json").write_text(
        json.dumps(_clean(out), indent=2, default=str)
    )

    # --------- Markdown report ---------
    lines = []
    lines.append("# Co-temporal Graph Expansion — Retrieval Evaluation\n\n")
    lines.append(
        f"Corpus: {len(all_docs)} docs, {len(all_qs)} queries (base+adversarial+cotemporal merged).\n"
    )
    lines.append(f"Wall: {wall_s:.1f}s. LLM cost: ${cost_usd:.4f}.\n\n")

    lines.append("## Graph stats\n\n")
    lines.append(f"- Nodes: {graph_stats['nodes']}\n")
    lines.append(f"- Edges (undirected): {graph_stats['edges_undirected']}\n")
    lines.append(f"- Avg degree: {graph_stats['avg_degree']:.2f}\n")
    lines.append(f"- Max degree: {graph_stats['max_degree']}\n")
    lines.append(
        f"- Threshold: {graph_stats['threshold']}, cap/node: {graph_stats['cap_per_node']}\n"
    )
    lines.append(
        f"- Raw pairs above threshold: {graph_stats['raw_pairs_above_threshold']} / {graph_stats['raw_pairs_considered']}\n"
    )
    lines.append(
        f"- Sem-bridge edges kept (no-temporal-signal docs): {graph_stats.get('sem_bridge_edges_kept', 0)}\n"
    )
    lines.append(f"- Hub docs (top-10 by degree): {graph_stats['hub_docs_top10']}\n\n")

    lines.append("## Retrieval metrics per subset\n\n")
    lines.append(
        "| Subset | N | Direct R@5 | Cot R@5 | Δ R@5 | Direct R@10 | Cot R@10 | Direct MRR | Cot MRR | Direct NDCG | Cot NDCG |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for subset, m in results.items():
        d = m["direct_v7"]
        c = m["cotemporal_v7_plus_graph"]
        lines.append(
            f"| {subset} | {m['n']} | {d['recall@5']:.3f} | {c['recall@5']:.3f} | "
            f"{c['recall@5'] - d['recall@5']:+.3f} | "
            f"{d['recall@10']:.3f} | {c['recall@10']:.3f} | "
            f"{d['mrr']:.3f} | {c['mrr']:.3f} | "
            f"{d['ndcg@10']:.3f} | {c['ndcg@10']:.3f} |\n"
        )

    lines.append("\n## Expansion source breakdown (cotemporal subset, top-5)\n\n")
    tot = breakdown["total"] or 1
    lines.append(f"- Total slots: {breakdown['total']}\n")
    lines.append(
        f"- Direct-or-mixed: {breakdown['direct_or_mixed_at5']} ({breakdown['direct_or_mixed_at5'] / tot * 100:.1f}%)\n"
    )
    lines.append(
        f"- Expansion-only: {breakdown['expansion_only_at5']} ({breakdown['expansion_only_at5'] / tot * 100:.1f}%)\n\n"
    )

    lines.append("## Cotemporal subset — per-query R@5\n\n")
    lines.append("| qid | direct R@5 | cot R@5 | Δ |\n|---|---:|---:|---:|\n")
    for q in cot_per_query:
        lines.append(
            f"| `{q['qid']}` | {q['direct_r5']:.2f} | {q['cot_r5']:.2f} | {q['delta_r5']:+.2f} |\n"
        )
    lines.append("\n")

    lines.append("## S8 adversarial recovery\n\n")
    for s in s8_detail:
        lines.append(f"### {s['qid']}\n")
        lines.append(f"- gold: {s['gold']}\n")
        lines.append(f"- direct top-5: {s['direct_top5']}\n")
        lines.append(f"- cot top-5: {s['cot_top5']}\n")
        lines.append(f"- direct hit: {s['direct_hit']} | cot hit: {s['cot_hit']}\n\n")

    lines.append("## Base-set regressions (queries where cot-R@5 < direct-R@5)\n\n")
    lines.append(f"Count: {len(regressions)}\n\n")
    for r in regressions[:10]:
        lines.append(
            f"- `{r['qid']}`: direct={r['direct']:.2f}, cot={r['cot']:.2f}, diff={r['diff']:+.2f}\n"
        )

    lines.append("\n## Topic drift diagnosis\n\n")
    td = out["topic_drift"]
    lines.append(f"- Drift docs in corpus: {td['drift_doc_count']}\n")
    lines.append(
        f"- Expansion candidates (all cot queries): {td['expansion_candidates_total']}\n"
    )
    lines.append(
        f"- …of which on drift docs: {td['expansion_candidates_on_drift']} "
        f"({td['drift_fraction_of_candidates'] * 100:.1f}%)\n"
    )
    lines.append(
        f"- Drift docs in top-10 of cot queries: {td['drift_in_top10_slots_with_drift']} / {td['drift_in_top10_total_slots']} slots\n\n"
    )

    lines.append("## Cost\n\n")
    lines.append(f"- Input tokens: {total_in}, output tokens: {total_out}\n")
    lines.append(f"- Estimated cost: ${cost_usd:.4f}\n")
    lines.append(f"- Wall clock: {wall_s:.1f}s\n\n")

    lines.append("## Ship recommendation\n\n")

    # Heuristic rec
    s8_dr5 = (
        results["adversarial_S8"]["cotemporal_v7_plus_graph"]["recall@5"]
        - results["adversarial_S8"]["direct_v7"]["recall@5"]
    )
    cot_dr5 = (
        results["cotemporal"]["cotemporal_v7_plus_graph"]["recall@5"]
        - results["cotemporal"]["direct_v7"]["recall@5"]
    )
    base_dr5 = (
        results["base"]["cotemporal_v7_plus_graph"]["recall@5"]
        - results["base"]["direct_v7"]["recall@5"]
    )
    adv_dr5 = (
        results["adversarial_full"]["cotemporal_v7_plus_graph"]["recall@5"]
        - results["adversarial_full"]["direct_v7"]["recall@5"]
    )

    rec = []
    if cot_dr5 > 0.05 and s8_dr5 >= 0 and base_dr5 >= -0.01 and adv_dr5 >= -0.01:
        rec.append(
            "INTEGRATE ALWAYS: graph expansion lifts cot + S8 without base regression."
        )
    elif cot_dr5 > 0.05 and base_dr5 < -0.01:
        rec.append(
            "GATE: graph expansion helps co-mention cases but regresses base — "
            "trigger only when direct retrieval has weak top-k (low confidence) "
            "or query lacks explicit time anchor."
        )
    elif cot_dr5 <= 0:
        rec.append("DEPRIORITIZE: graph expansion does not lift co-mention queries.")
    else:
        rec.append(
            "CONDITIONAL INTEGRATE: modest lift, re-evaluate with richer corpus."
        )

    lines.append(
        f"Δ R@5 by subset: base={base_dr5:+.3f}, adversarial={adv_dr5:+.3f}, "
        f"S8={s8_dr5:+.3f}, cotemporal={cot_dr5:+.3f}.\n\n"
    )
    for r in rec:
        lines.append(f"- {r}\n")

    (RESULTS_DIR / "cotemporal.md").write_text("".join(lines))

    print("\nWrote results/cotemporal.{json,md}")
    print("Subsets summary:")
    for subset, m in results.items():
        d = m["direct_v7"]
        c = m["cotemporal_v7_plus_graph"]
        print(
            f"  {subset:20s} (n={m['n']:3d}) direct R@5={d['recall@5']:.3f} | cot R@5={c['recall@5']:.3f} | Δ={c['recall@5'] - d['recall@5']:+.3f}"
        )
    print(f"Cost: ${cost_usd:.4f}, Wall: {wall_s:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
