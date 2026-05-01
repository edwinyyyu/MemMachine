"""Multi-axis retrieval evaluation.

Extracts time expressions on the axis-synth docs/queries (+ replays base
corpus from cache). Builds per-doc memory of (Interval, axis_dists, tags)
triples. Scores all docs under each variant:

- INTERVAL-ONLY: jaccard_composite + sum (same as baseline used in tag_eval).
- TAGS-ONLY: hierarchical tags (F2 T1 variant: Jaccard/sum).
- AXIS-DIST: beta=1 (axis only).
- MULTI-AXIS: alpha/beta/gamma sweep.
- HYBRID: best MULTI-AXIS + semantic rerank of top-20.

Writes results/multi_axis.{md,json}. Uses fresh LLM cache at
cache/multi_axis/llm_cache.json for new axis corpus items; replays base
extractions from the existing shared cache.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
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
from hierarchical_tags import tags_for_expression as hier_tags_for_expression
from multi_axis_scorer import (
    axis_score,
    tag_score,
)
from multi_axis_tags import tags_for_axes
from schema import (
    GRANULARITY_ORDER,
    TimeExpression,
    parse_iso,
    to_us,
)
from scorer import Interval, score_jaccard_composite

# ---------------------------------------------------------------------------
# Paths + cache
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MA_CACHE_DIR = ROOT / "cache" / "multi_axis"
MA_CACHE_DIR.mkdir(exist_ok=True, parents=True)

TOP_K = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Query-interval flatten (adapted from eval.py)
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
            end = min(end, te.recurrence.until.latest or te.recurrence.until.earliest)
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


# ---------------------------------------------------------------------------
# Per-doc memory
# ---------------------------------------------------------------------------
def build_doc_memory(
    doc_extracted: dict[str, list[TimeExpression]],
) -> dict[str, dict[str, Any]]:
    """For each doc_id, return a bundle:

    {
      "intervals": list[Interval],          # flattened, for interval score
      "axes_merged": dict[axis -> AxisDistribution],  # merged across exprs
      "axes_per_expr": list[dict[axis -> AxisDist]],
      "multi_tags": set[str],               # multi-axis tags
      "hier_tags": set[str],                # hierarchical tags (F2)
    }
    """
    out: dict[str, dict[str, Any]] = {}
    for doc_id, tes in doc_extracted.items():
        intervals: list[Interval] = []
        axes_per: list[dict[str, AxisDistribution]] = []
        multi_tags: set[str] = set()
        hier_tags: set[str] = set()
        for te in tes:
            intervals.extend(flatten_query_intervals(te))
            ax = axes_for_expression(te)
            axes_per.append(ax)
            multi_tags |= tags_for_axes(ax)
            hier_tags |= hier_tags_for_expression(te)
        axes_merged = merge_axis_dists(axes_per)
        out[doc_id] = {
            "intervals": intervals,
            "axes_merged": axes_merged,
            "axes_per_expr": axes_per,
            "multi_tags": multi_tags,
            "hier_tags": hier_tags,
        }
    return out


# ---------------------------------------------------------------------------
# Query memory
# ---------------------------------------------------------------------------
def build_query_memory(
    query_extracted: dict[str, list[TimeExpression]],
) -> dict[str, dict[str, Any]]:
    return build_doc_memory(query_extracted)


# ---------------------------------------------------------------------------
# Scoring per variant
# ---------------------------------------------------------------------------
def interval_pair_best(q_ivs: list[Interval], d_ivs: list[Interval]) -> float:
    """Max over q-interval of (sum over docs' best score given that q-iv).

    Returns sum-aggregation of per-q-interval best matches (same pattern as
    tag_eval baseline).
    """
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


def rank_interval(
    q_mem: dict[str, Any],
    doc_mem: dict[str, dict[str, Any]],
) -> list[tuple[str, float]]:
    q_ivs = q_mem["intervals"]
    scores: dict[str, float] = {}
    for doc_id, bundle in doc_mem.items():
        scores[doc_id] = interval_pair_best(q_ivs, bundle["intervals"])
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rank_hier_tags(
    q_mem: dict[str, Any],
    doc_mem: dict[str, dict[str, Any]],
) -> list[tuple[str, float]]:
    """Hierarchical tag Jaccard (no weighting)."""
    qt = q_mem["hier_tags"]
    scores: dict[str, float] = {}
    for doc_id, bundle in doc_mem.items():
        dt = bundle["hier_tags"]
        if (not qt and not dt) or not qt or not dt:
            scores[doc_id] = 0.0
        else:
            inter = len(qt & dt)
            union = len(qt | dt)
            scores[doc_id] = inter / union if union else 0.0
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rank_axis_only(
    q_mem: dict[str, Any],
    doc_mem: dict[str, dict[str, Any]],
    skip_axis: str | None = None,
) -> list[tuple[str, float]]:
    qa = q_mem["axes_merged"]
    scores: dict[str, float] = {}
    for doc_id, bundle in doc_mem.items():
        scores[doc_id] = axis_score(qa, bundle["axes_merged"], skip_axis=skip_axis)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rank_multi_axis(
    q_mem: dict[str, Any],
    doc_mem: dict[str, dict[str, Any]],
    alpha: float,
    beta: float,
    gamma: float,
    skip_axis: str | None = None,
) -> list[tuple[str, float]]:
    qa = q_mem["axes_merged"]
    q_multi_tags = q_mem["multi_tags"]
    q_ivs = q_mem["intervals"]
    # Normalize interval score to [0, 1] across docs by dividing by max;
    # otherwise the unnormalized sum can dominate. We'll min-max scale
    # across the candidate set.
    raw_iv: dict[str, float] = {}
    for doc_id, bundle in doc_mem.items():
        raw_iv[doc_id] = interval_pair_best(q_ivs, bundle["intervals"])
    max_iv = max(raw_iv.values()) if raw_iv else 0.0

    scores: dict[str, float] = {}
    for doc_id, bundle in doc_mem.items():
        iv_norm = raw_iv[doc_id] / max_iv if max_iv > 0 else 0.0
        a_sc = axis_score(qa, bundle["axes_merged"], skip_axis=skip_axis)
        t_sc = tag_score(q_multi_tags, bundle["multi_tags"])
        total = alpha * iv_norm + beta * a_sc + gamma * t_sc
        scores[doc_id] = total
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def semantic_rerank(
    candidates: list[str],
    query_emb: np.ndarray,
    doc_embs: dict[str, np.ndarray],
) -> list[tuple[str, float]]:
    qn = np.linalg.norm(query_emb) or 1e-9
    out = []
    for d in candidates:
        v = doc_embs.get(d)
        if v is None:
            continue
        vn = np.linalg.norm(v) or 1e-9
        sim = float(np.dot(query_emb, v) / (qn * vn))
        out.append((d, sim))
    return sorted(out, key=lambda x: x[1], reverse=True)


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


def average(vals: list[float]) -> float:
    vs = [v for v in vals if not math.isnan(v)]
    return sum(vs) / len(vs) if vs else 0.0


def eval_rankings(
    ranked_per_q: dict[str, list[str]],
    gold: dict[str, set[str]],
    qids: set[str],
) -> dict[str, float]:
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
        "recall@5": average(r5),
        "recall@10": average(r10),
        "mrr": average(mr),
        "ndcg@10": average(nd),
        "n": len([v for v in r5 if not math.isnan(v)]),
    }


# ---------------------------------------------------------------------------
# Extractor setup — uses a multi_axis-specific cache for axis docs/queries,
# but leverages the shared base cache for base corpus.
# ---------------------------------------------------------------------------
def _build_extractor(cache_file: Path):
    from extractor import Extractor, LLMCache

    ex = Extractor()
    # Replace cache with a namespaced one.
    ex.cache = LLMCache(path=cache_file)
    return ex


async def extract_items(
    items: list[tuple[str, str, datetime]],
    cache_file: Path,
    label: str,
) -> tuple[dict[str, list[TimeExpression]], dict[str, int]]:
    ex = _build_extractor(cache_file)

    async def one(iid: str, text: str, ref: datetime):
        try:
            tes = await ex.extract(text, ref)
        except Exception as e:
            print(f"  extract failed for {iid}: {e}")
            tes = []
        return iid, tes

    print(f"extracting {label} ({len(items)} items, cache={cache_file.name})...")
    results = await asyncio.gather(*(one(*it) for it in items))
    ex.cache.save()
    print(f"  {label} usage: input={ex.usage['input']}, output={ex.usage['output']}")
    return {i: t for i, t in results}, ex.usage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run() -> None:
    # Ensure axis synth data exists
    if not (DATA_DIR / "axis_docs.jsonl").exists():
        import axis_synth

        axis_synth.main()

    # ----- Load base + axis data -----
    base_docs = load_jsonl(DATA_DIR / "docs.jsonl")
    base_queries = load_jsonl(DATA_DIR / "queries.jsonl")
    base_gold = {
        r["query_id"]: set(r["relevant_doc_ids"])
        for r in load_jsonl(DATA_DIR / "gold.jsonl")
    }

    axis_docs = load_jsonl(DATA_DIR / "axis_docs.jsonl")
    axis_queries = load_jsonl(DATA_DIR / "axis_queries.jsonl")
    axis_gold = {
        r["query_id"]: set(r["relevant_doc_ids"])
        for r in load_jsonl(DATA_DIR / "axis_gold.jsonl")
    }

    all_docs = base_docs + axis_docs
    all_queries = base_queries + axis_queries
    all_gold = {**base_gold, **axis_gold}

    base_qids = {q["query_id"] for q in base_queries}
    axis_qids = {q["query_id"] for q in axis_queries}
    all_qids = base_qids | axis_qids

    print(f"Loaded docs: base={len(base_docs)}, axis={len(axis_docs)}")
    print(f"Loaded queries: base={len(base_queries)}, axis={len(axis_queries)}")

    # ----- Extractions -----
    # Base corpus -> shared LLM cache (read-only in practice since prior runs
    # populated it). Axis corpus -> namespaced cache.
    base_items_docs = [
        (d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in base_docs
    ]
    base_items_queries = [
        (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in base_queries
    ]
    axis_items_docs = [
        (d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in axis_docs
    ]
    axis_items_queries = [
        (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in axis_queries
    ]

    base_doc_extracted, u1 = await extract_items(
        base_items_docs, ROOT / "cache" / "llm_cache.json", "base-docs"
    )
    base_query_extracted, u2 = await extract_items(
        base_items_queries, ROOT / "cache" / "llm_cache.json", "base-queries"
    )
    axis_doc_extracted, u3 = await extract_items(
        axis_items_docs, MA_CACHE_DIR / "llm_cache.json", "axis-docs"
    )
    axis_query_extracted, u4 = await extract_items(
        axis_items_queries, MA_CACHE_DIR / "llm_cache.json", "axis-queries"
    )

    total_usage = {
        "input": sum(u["input"] for u in (u1, u2, u3, u4)),
        "output": sum(u["output"] for u in (u1, u2, u3, u4)),
    }
    # Per-call from axis corpus only (new LLM cost)
    axis_new_usage = {
        "input": u3["input"] + u4["input"],
        "output": u3["output"] + u4["output"],
    }
    # gpt-5-mini: $0.25/M in, $2.00/M out
    axis_cost = (
        axis_new_usage["input"] * 0.25 / 1_000_000
        + axis_new_usage["output"] * 2.0 / 1_000_000
    )
    print(
        f"Axis-corpus new LLM cost: ${axis_cost:.4f} (in={axis_new_usage['input']}, out={axis_new_usage['output']})"
    )

    # Merge extractions
    doc_extracted: dict[str, list[TimeExpression]] = {
        **base_doc_extracted,
        **axis_doc_extracted,
    }
    query_extracted: dict[str, list[TimeExpression]] = {
        **base_query_extracted,
        **axis_query_extracted,
    }

    # Extraction inspection on axis queries (what does the extractor think
    # "Thursdays" / "March" / "afternoons" are?)
    axis_ext_report: list[dict[str, Any]] = []
    for q in axis_queries:
        qid = q["query_id"]
        tes = query_extracted.get(qid, [])
        entries: list[dict[str, Any]] = []
        for te in tes:
            entry = {
                "kind": te.kind,
                "surface": te.surface,
            }
            if te.instant:
                entry["instant"] = {
                    "earliest": str(te.instant.earliest),
                    "latest": str(te.instant.latest),
                    "best": str(te.instant.best) if te.instant.best else None,
                    "granularity": te.instant.granularity,
                }
            if te.recurrence:
                entry["recurrence"] = {
                    "rrule": te.recurrence.rrule,
                    "dtstart_best": str(te.recurrence.dtstart.best)
                    if te.recurrence.dtstart.best
                    else None,
                }
            entries.append(entry)
        axis_ext_report.append({"qid": qid, "text": q["text"], "extracted": entries})

    # ----- Build memory -----
    print("Building doc memory (axis distributions + tags)...")
    doc_mem = build_doc_memory(doc_extracted)
    print("Building query memory...")
    query_mem = build_query_memory(query_extracted)

    # ----- Embeddings for hybrid -----
    print("Embedding (cached) docs + queries...")
    doc_texts = [d["text"] for d in all_docs]
    query_texts = [q["text"] for q in all_queries]
    doc_embs_arr = await embed_all(doc_texts)
    query_embs_arr = await embed_all(query_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(all_docs)}
    query_embs = {q["query_id"]: query_embs_arr[i] for i, q in enumerate(all_queries)}

    # ----- Run rankings -----
    all_doc_ids = [d["doc_id"] for d in all_docs]

    sweeps: list[tuple[float, float, float]] = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.35, 0.15),
        (0.4, 0.4, 0.2),
        (0.3, 0.5, 0.2),
    ]

    # Precompute rankings for each variant.
    # We restrict the doc pool to all_doc_ids (which doc_mem covers).
    for doc_id in all_doc_ids:
        if doc_id not in doc_mem:
            doc_mem[doc_id] = {
                "intervals": [],
                "axes_merged": {
                    a: AxisDistribution(axis=a, values={}, informative=False)
                    for a in AXES
                },
                "axes_per_expr": [],
                "multi_tags": set(),
                "hier_tags": set(),
            }

    variants: dict[str, dict[str, list[str]]] = {}

    def rank_all(fn) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for qid in all_qids:
            qm = query_mem.get(qid)
            if qm is None:
                qm = {
                    "intervals": [],
                    "axes_merged": {
                        a: AxisDistribution(axis=a, values={}, informative=False)
                        for a in AXES
                    },
                    "axes_per_expr": [],
                    "multi_tags": set(),
                    "hier_tags": set(),
                }
            ranked = fn(qm)
            out[qid] = [d for d, _ in ranked]
        return out

    print("Running INTERVAL-ONLY ranking...")
    variants["INTERVAL-ONLY"] = rank_all(lambda qm: rank_interval(qm, doc_mem))

    print("Running TAGS-ONLY (hierarchical F2) ranking...")
    variants["TAGS-ONLY (hierarchical)"] = rank_all(
        lambda qm: rank_hier_tags(qm, doc_mem)
    )

    print("Running AXIS-DIST ranking...")
    variants["AXIS-DIST"] = rank_all(lambda qm: rank_axis_only(qm, doc_mem))

    print("Running MULTI-AXIS sweeps...")
    for a, b, g in sweeps:
        key = f"MULTI-AXIS α={a} β={b} γ={g}"
        variants[key] = rank_all(
            lambda qm, a=a, b=b, g=g: rank_multi_axis(qm, doc_mem, a, b, g)
        )

    # ----- Evaluate per variant -----
    per_variant: dict[str, dict[str, dict[str, float]]] = {}
    for name, ranked_per_q in variants.items():
        per_variant[name] = {
            "axis": eval_rankings(ranked_per_q, all_gold, axis_qids),
            "base": eval_rankings(ranked_per_q, all_gold, base_qids),
            "all": eval_rankings(ranked_per_q, all_gold, all_qids),
        }

    # ----- Pick best MULTI-AXIS by axis-subset R@5 -----
    sweep_names = [k for k in variants if k.startswith("MULTI-AXIS")]
    best_name = max(
        sweep_names,
        key=lambda n: per_variant[n]["axis"]["recall@5"],
    )
    print(f"Best MULTI-AXIS: {best_name}")
    best_weights = best_name.replace("MULTI-AXIS ", "")

    # ----- HYBRID: best MULTI-AXIS top-20 + semantic rerank -----
    print("Running HYBRID (best MULTI-AXIS + semantic rerank)...")
    best_alpha, best_beta, best_gamma = [
        float(s.split("=", 1)[1]) for s in best_weights.split(" ")
    ]

    def hybrid_fn(qm: dict[str, Any], qid: str) -> list[tuple[str, float]]:
        ma = rank_multi_axis(qm, doc_mem, best_alpha, best_beta, best_gamma)
        if not ma:
            return []
        cand = [d for d, _ in ma[:20]]
        sem = semantic_rerank(cand, query_embs[qid], doc_embs)
        return sem

    hybrid_ranked: dict[str, list[str]] = {}
    for qid in all_qids:
        qm = query_mem.get(qid) or {
            "intervals": [],
            "axes_merged": {
                a: AxisDistribution(axis=a, values={}, informative=False) for a in AXES
            },
            "axes_per_expr": [],
            "multi_tags": set(),
            "hier_tags": set(),
        }
        hybrid_ranked[qid] = [d for d, _ in hybrid_fn(qm, qid)]
    variants["HYBRID (MULTI-AXIS + semantic)"] = hybrid_ranked
    per_variant["HYBRID (MULTI-AXIS + semantic)"] = {
        "axis": eval_rankings(hybrid_ranked, all_gold, axis_qids),
        "base": eval_rankings(hybrid_ranked, all_gold, base_qids),
        "all": eval_rankings(hybrid_ranked, all_gold, all_qids),
    }

    # ----- Per-axis ablation (remove one axis at a time from MULTI-AXIS best) -----
    print("Running per-axis ablation...")
    ablation: dict[str, dict[str, float]] = {}
    for axis in AXES:
        ranked_per_q: dict[str, list[str]] = {}
        for qid in all_qids:
            qm = query_mem.get(qid) or {
                "intervals": [],
                "axes_merged": {
                    a: AxisDistribution(axis=a, values={}, informative=False)
                    for a in AXES
                },
                "axes_per_expr": [],
                "multi_tags": set(),
                "hier_tags": set(),
            }
            ranked = rank_multi_axis(
                qm,
                doc_mem,
                best_alpha,
                best_beta,
                best_gamma,
                skip_axis=axis,
            )
            ranked_per_q[qid] = [d for d, _ in ranked]
        ablation[axis] = eval_rankings(ranked_per_q, all_gold, axis_qids)

    # ----- Failure modes on axis queries -----
    failures: list[dict[str, Any]] = []
    best_ranked = variants[best_name]
    hybrid_ranked_map = variants["HYBRID (MULTI-AXIS + semantic)"]
    iv_ranked = variants["INTERVAL-ONLY"]
    for q in axis_queries:
        qid = q["query_id"]
        relevant = axis_gold.get(qid, set())
        best_r5 = recall_at_k(best_ranked.get(qid, []), relevant, 5)
        hyb_r5 = recall_at_k(hybrid_ranked_map.get(qid, []), relevant, 5)
        iv_r5 = recall_at_k(iv_ranked.get(qid, []), relevant, 5)
        failures.append(
            {
                "qid": qid,
                "text": q["text"],
                "relevant": sorted(relevant),
                "multi_axis_top5": best_ranked.get(qid, [])[:5],
                "hybrid_top5": hybrid_ranked_map.get(qid, [])[:5],
                "interval_top5": iv_ranked.get(qid, [])[:5],
                "R@5_multi_axis": best_r5,
                "R@5_hybrid": hyb_r5,
                "R@5_interval": iv_r5,
            }
        )

    # ----- Write JSON -----
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    results_json = {
        "variants": {k: _clean(v) for k, v in per_variant.items()},
        "best_multi_axis": {
            "name": best_name,
            "alpha": best_alpha,
            "beta": best_beta,
            "gamma": best_gamma,
        },
        "ablation_axis_subset": _clean(ablation),
        "axis_extraction_report": axis_ext_report,
        "axis_query_failures": _clean(failures),
        "cost": {
            "axis_new_usage": axis_new_usage,
            "axis_new_cost_usd": axis_cost,
        },
    }
    (RESULTS_DIR / "multi_axis.json").write_text(json.dumps(results_json, indent=2))

    # ----- Markdown -----
    lines: list[str] = []
    lines.append("# Multi-Axis + Distributional Time Representation\n\n")
    lines.append(
        "Per-axis categorical distributions + cross-axis tags on top of interval brackets.\n\n"
    )
    lines.append("## Corpus\n\n")
    lines.append(f"- Base: {len(base_docs)} docs, {len(base_queries)} queries.\n")
    lines.append(
        f"- Axis (new): {len(axis_docs)} docs, {len(axis_queries)} queries.\n\n"
    )
    lines.append("## Per-variant metrics\n\n")
    lines.append(
        "| Variant | axis R@5 | axis R@10 | axis MRR | axis NDCG | "
        "base R@5 | base NDCG | all R@5 | all NDCG |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    order = (
        [
            "INTERVAL-ONLY",
            "TAGS-ONLY (hierarchical)",
            "AXIS-DIST",
        ]
        + [n for n in variants if n.startswith("MULTI-AXIS")]
        + [
            "HYBRID (MULTI-AXIS + semantic)",
        ]
    )
    for name in order:
        m = per_variant[name]
        lines.append(
            f"| {name} | "
            f"{m['axis']['recall@5']:.3f} | {m['axis']['recall@10']:.3f} | "
            f"{m['axis']['mrr']:.3f} | {m['axis']['ndcg@10']:.3f} | "
            f"{m['base']['recall@5']:.3f} | {m['base']['ndcg@10']:.3f} | "
            f"{m['all']['recall@5']:.3f} | {m['all']['ndcg@10']:.3f} |\n"
        )

    lines.append("\n## Best MULTI-AXIS blend\n\n")
    lines.append(
        f"- {best_name} (by axis-subset R@5)\n"
        f"- α={best_alpha}, β={best_beta}, γ={best_gamma}\n\n"
    )

    lines.append("## Per-axis ablation (skip one axis)\n\n")
    lines.append(
        "Removing each axis from the MULTI-AXIS best blend; axis-subset R@5:\n\n"
    )
    baseline_r5 = per_variant[best_name]["axis"]["recall@5"]
    lines.append(f"- Baseline (all axes): R@5={baseline_r5:.3f}\n")
    rows = [
        (a, abl["recall@5"], baseline_r5 - abl["recall@5"])
        for a, abl in ablation.items()
    ]
    rows.sort(key=lambda r: r[2], reverse=True)
    for axis, r5, delta in rows:
        lines.append(f"- skip `{axis}`: R@5={r5:.3f} (Δ={-delta:+.3f})\n")

    lines.append("\n## Extraction quality on axis queries (sample)\n\n")
    for entry in axis_ext_report[:8]:
        lines.append(f"- **{entry['qid']}** `{entry['text']}` ->\n")
        for e in entry["extracted"]:
            lines.append(f"  - kind={e['kind']}, surface=`{e.get('surface')}`")
            if "instant" in e:
                lines.append(
                    f"; gran={e['instant']['granularity']}, "
                    f"[{e['instant']['earliest']}..{e['instant']['latest']}]"
                )
            if "recurrence" in e:
                lines.append(f"; rrule={e['recurrence']['rrule']}")
            lines.append("\n")

    lines.append("\n## Failure modes (axis queries)\n\n")
    # Show the queries where multi-axis notably lost or won over interval-only.
    for f in failures:
        d = f["R@5_multi_axis"] - (f["R@5_interval"] or 0.0)
        if d == 0:
            continue
        sign = "+" if d > 0 else ""
        lines.append(
            f"- `{f['qid']}` ({f['text']!r}) Δ{sign}{d:+.2f}: "
            f"gold={f['relevant']}; interval_top5={f['interval_top5']}; "
            f"multi_axis_top5={f['multi_axis_top5']}; hybrid_top5={f['hybrid_top5']}\n"
        )

    lines.append("\n## Cost\n\n")
    lines.append(
        f"- New LLM tokens (axis corpus): input={axis_new_usage['input']}, output={axis_new_usage['output']}\n"
    )
    lines.append(f"- Estimated cost: ${axis_cost:.4f}\n")

    (RESULTS_DIR / "multi_axis.md").write_text("".join(lines))

    # ----- Console summary -----
    print("\n=== Summary (axis subset) ===")
    for name in order:
        m = per_variant[name]["axis"]
        print(
            f"  {name:<50} axis R@5={m['recall@5']:.3f} NDCG={m['ndcg@10']:.3f} MRR={m['mrr']:.3f}"
        )
    print("\n=== Summary (base subset, regression check) ===")
    for name in order:
        m = per_variant[name]["base"]
        print(f"  {name:<50} base R@5={m['recall@5']:.3f} NDCG={m['ndcg@10']:.3f}")

    print(f"\nBest MULTI-AXIS: {best_name}")
    print(f"Axis-corpus LLM cost: ${axis_cost:.4f}")
    print("Wrote results/multi_axis.{md,json}")


if __name__ == "__main__":
    asyncio.run(run())
