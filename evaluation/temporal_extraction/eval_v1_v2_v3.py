"""Cache-only evaluation of extractor v1/v2/v3.

Loads each version's existing llm_cache.json, runs the full extraction
pipeline by monkey-patching the OpenAI client so that cache misses return
a no-op response INSTEAD of making a live LLM call. This guarantees 0
token spend and no possibility of hanging on CoT-reasoning tokens.

Computes:
- Extraction F1 on docs + queries that have gold_expressions.
- Failure-case recall on the 38 v1-missed surfaces.
- Downstream R@5/MRR/NDCG@10 on the base 55 queries using the ship-best
  multi-axis scorer (alpha=0.5, beta=0.35, gamma=0.15).
- Axis-specific R@5 on the 20 axis queries (same scorer).
- Per-surface success rate for bare months/seasons/quarters that the
  multi-axis experiment flagged (march, q2, june weekends, evening,
  summer, winter, autumn, q4, october).

Writes:
- results/extractor_v1_v2_v3.json
- results/extractor_v1_v2_v3.md
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from eval import extraction_metrics, load_jsonl, match_expressions
from extractor import Extractor as ExtractorV1
from extractor import LLMCache as V1LLMCache
from extractor_common import BaseImprovedExtractor
from extractor_common import LLMCache as VxLLMCache
from extractor_v2 import ExtractorV2
from extractor_v3 import ExtractorV3
from multi_axis_scorer import axis_score, tag_score
from multi_axis_tags import tags_for_axes
from schema import (
    GRANULARITY_ORDER,
    TimeExpression,
    parse_iso,
    time_expression_from_dict,
    to_us,
)
from scorer import Interval, score_jaccard_composite

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache"

MODEL = "gpt-5-mini"
ALPHA, BETA, GAMMA = 0.5, 0.35, 0.15


# ---------------------------------------------------------------------------
# Cache-only OpenAI stub. Any cache miss is logged and returns empty.
# ---------------------------------------------------------------------------
class CacheMissCounter:
    """Counts how many cache-miss LLM calls were requested."""

    def __init__(self) -> None:
        self.misses = 0
        self.miss_samples: list[str] = []

    def record(self, prompt_user: str) -> None:
        self.misses += 1
        if len(self.miss_samples) < 5:
            self.miss_samples.append(prompt_user[:200])


class StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class StubChoice:
    def __init__(self, content: str) -> None:
        self.message = StubMessage(content)


class StubResponse:
    def __init__(self) -> None:
        self.choices = [StubChoice("")]
        self.usage = None


class StubChatCompletions:
    def __init__(self, counter: CacheMissCounter) -> None:
        self.counter = counter

    async def create(self, *args: Any, **kwargs: Any) -> StubResponse:
        messages = kwargs.get("messages", [])
        user_prompt = ""
        for m in messages:
            if m.get("role") == "user":
                user_prompt = m.get("content", "")
                break
        self.counter.record(user_prompt)
        return StubResponse()


class StubChat:
    def __init__(self, counter: CacheMissCounter) -> None:
        self.completions = StubChatCompletions(counter)


class StubAsyncOpenAI:
    def __init__(self, counter: CacheMissCounter) -> None:
        self.chat = StubChat(counter)


# ---------------------------------------------------------------------------
# Extractor construction with stubbed client + cache wired to the correct
# version directory. Each version has its own llm_cache.json.
# ---------------------------------------------------------------------------
def make_v1_extractor(counter: CacheMissCounter) -> ExtractorV1:
    ex = ExtractorV1()
    ex.client = StubAsyncOpenAI(counter)
    # v1 uses the shared cache/llm_cache.json
    ex.cache = V1LLMCache(CACHE_DIR / "llm_cache.json")
    return ex


def make_vx_extractor(
    cls: type[BaseImprovedExtractor],
    counter: CacheMissCounter,
) -> BaseImprovedExtractor:
    ex = cls(model=MODEL, cache_subdir=f"extractor_v{cls.VERSION}")
    ex.client = StubAsyncOpenAI(counter)
    # reload both caches to be safe
    ex.cache = VxLLMCache(CACHE_DIR / f"extractor_v{cls.VERSION}" / "llm_cache.json")
    ex.shared_pass2_cache = VxLLMCache(
        CACHE_DIR / "extractor_shared_pass2" / "llm_cache.json"
    )
    return ex


# ---------------------------------------------------------------------------
# Per-call timeout wrapper — even stub calls get a hard ceiling.
# ---------------------------------------------------------------------------
async def extract_with_timeout(
    extractor: Any,
    iid: str,
    text: str,
    ref_time: datetime,
    timeout_s: float = 45.0,
) -> tuple[str, list[TimeExpression]]:
    try:
        tes = await asyncio.wait_for(
            extractor.extract(text, ref_time), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        print(f"  timeout on {iid}")
        tes = []
    except Exception as e:
        print(f"  extract failed on {iid}: {e}")
        tes = []
    return iid, tes


async def run_extraction(
    extractor: Any,
    items: list[tuple[str, str, datetime]],
    label: str,
) -> dict[str, list[TimeExpression]]:
    t0 = time.time()
    results = await asyncio.gather(
        *(extract_with_timeout(extractor, i, t, r) for i, t, r in items)
    )
    dt = time.time() - t0
    print(f"  {label}: {len(items)} items in {dt:.1f}s")
    return dict(results)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def compute_extraction_f1(
    items_with_gold: list[dict],
    pred_by_id: dict[str, list[TimeExpression]],
    id_field: str,
) -> dict[str, Any]:
    """Extraction F1 on items that have gold_expressions."""
    match_results = []
    for item in items_with_gold:
        iid = item[id_field]
        pred = pred_by_id.get(iid, [])
        gold_raw = item.get("gold_expressions") or []
        if not gold_raw:
            continue
        gold = [time_expression_from_dict(g) for g in gold_raw]
        match_results.append(match_expressions(pred, gold, item["text"]))
    return extraction_metrics(match_results)


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
        # Expand recurrence to concrete instances
        try:
            from expander import expand

            now = datetime.now(tz=timezone.utc)
            anchor = te.recurrence.dtstart.best or te.recurrence.dtstart.earliest
            start = min(now - timedelta(days=365 * 5), anchor - timedelta(days=365))
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
        except Exception:
            pass
    return out


def build_memory(
    pred_by_id: dict[str, list[TimeExpression]],
) -> dict[str, dict[str, Any]]:
    """For each id return {intervals, axes_merged, multi_tags}."""
    out: dict[str, dict[str, Any]] = {}
    for iid, tes in pred_by_id.items():
        intervals: list[Interval] = []
        axes_per: list[dict[str, AxisDistribution]] = []
        multi_tags: set[str] = set()
        for te in tes:
            intervals.extend(flatten_intervals(te))
            ax = axes_for_expression(te)
            axes_per.append(ax)
            multi_tags |= tags_for_axes(ax)
        out[iid] = {
            "intervals": intervals,
            "axes_merged": merge_axis_dists(axes_per),
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
    q_multi_tags = q_mem["multi_tags"]
    q_ivs = q_mem["intervals"]
    raw_iv: dict[str, float] = {}
    for doc_id, bundle in doc_mem.items():
        raw_iv[doc_id] = interval_pair_best(q_ivs, bundle["intervals"])
    max_iv = max(raw_iv.values()) if raw_iv else 0.0
    scores: dict[str, float] = {}
    for doc_id, bundle in doc_mem.items():
        iv_norm = raw_iv[doc_id] / max_iv if max_iv > 0 else 0.0
        a_sc = axis_score(qa, bundle["axes_merged"])
        t_sc = tag_score(q_multi_tags, bundle["multi_tags"])
        scores[doc_id] = alpha * iv_norm + beta * a_sc + gamma * t_sc
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


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
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, d in enumerate(ranked[:k], start=1)
        if d in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def _avg(vs: list[float]) -> float:
    cs = [v for v in vs if not math.isnan(v)]
    return sum(cs) / len(cs) if cs else 0.0


def retrieval_metrics(
    doc_mem: dict[str, dict[str, Any]],
    query_mem: dict[str, dict[str, Any]],
    queries: list[dict],
    gold: dict[str, set[str]],
) -> dict[str, float]:
    r5, r10, mr, nd = [], [], [], []
    neutral_qm = {
        "intervals": [],
        "axes_merged": {
            a: AxisDistribution(axis=a, values={}, informative=False) for a in AXES
        },
        "multi_tags": set(),
    }
    for q in queries:
        qid = q["query_id"]
        rel = gold.get(qid, set())
        if not rel:
            continue
        qm = query_mem.get(qid, neutral_qm)
        ranked = [d for d, _ in rank_multi_axis(qm, doc_mem, ALPHA, BETA, GAMMA)]
        r5.append(recall_at_k(ranked, rel, 5))
        r10.append(recall_at_k(ranked, rel, 10))
        mr.append(mrr(ranked, rel))
        nd.append(ndcg_at_k(ranked, rel, 10))
    return {
        "recall@5": _avg(r5),
        "recall@10": _avg(r10),
        "mrr": _avg(mr),
        "ndcg@10": _avg(nd),
        "n": sum(1 for v in r5 if not math.isnan(v)),
    }


# ---------------------------------------------------------------------------
# Failure recall
# ---------------------------------------------------------------------------
def failure_recall(
    failure_cases: list[dict],
    docs: list[dict],
    queries: list[dict],
    pred_by_doc: dict[str, list[TimeExpression]],
    pred_by_q: dict[str, list[TimeExpression]],
) -> dict[str, Any]:
    docs_by_id = {d["doc_id"]: d for d in docs}
    queries_by_id = {q["query_id"]: q for q in queries}
    total = len(failure_cases)
    recovered = 0
    per_surface: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "recovered": 0}
    )
    for fc in failure_cases:
        iid = fc["item_id"]
        surface = fc["missed_surface"]
        stype = fc["type"]
        if stype == "doc":
            item = docs_by_id.get(iid)
            pred = pred_by_doc.get(iid, [])
        else:
            item = queries_by_id.get(iid)
            pred = pred_by_q.get(iid, [])
        key = surface.lower()
        per_surface[key]["total"] += 1
        if item is None:
            continue
        text = item["text"]
        gold_expr = time_expression_from_dict(fc["gold_expr"])
        gs = gold_expr.span_start
        ge = gold_expr.span_end
        if gs is None or ge is None:
            idx = text.find(surface)
            if idx < 0:
                continue
            gs, ge = idx, idx + len(surface)
        g_len = max(1, ge - gs)
        got = False
        for te in pred:
            ps, pe = te.span_start, te.span_end
            if ps is None or pe is None:
                idx = text.find(te.surface)
                if idx < 0:
                    continue
                ps, pe = idx, idx + len(te.surface)
            p_len = max(1, pe - ps)
            overlap = max(0, min(pe, ge) - max(ps, gs))
            frac = overlap / min(p_len, g_len)
            if frac >= 0.5:
                got = True
                break
        if got:
            recovered += 1
            per_surface[key]["recovered"] += 1
    return {
        "total": total,
        "recovered": recovered,
        "recall": recovered / total if total else 0.0,
        "by_surface": dict(per_surface),
    }


# ---------------------------------------------------------------------------
# Axis-gap-specific surface check — simple: did the extractor capture ANY
# span covering the surface text in the source?
# ---------------------------------------------------------------------------
AXIS_GAP_SURFACES = [
    "March",
    "Q2",
    "June weekends",
    "evening",
    "Summer",
    "winter",
    "Autumn",
    "Q4",
    "October",
]


def axis_gap_rate(
    items: list[dict],
    pred_by_id: dict[str, list[TimeExpression]],
    id_field: str,
) -> dict[str, Any]:
    """For each axis-gap surface, count how many items contain it (case-
    insensitive) and how many had at least one predicted extraction
    overlapping the surface position >=50%."""
    stats: dict[str, dict[str, int]] = {
        s.lower(): {"occurs": 0, "captured": 0} for s in AXIS_GAP_SURFACES
    }
    for item in items:
        text = item["text"]
        lower = text.lower()
        iid = item[id_field]
        pred = pred_by_id.get(iid, [])
        for surf in AXIS_GAP_SURFACES:
            key = surf.lower()
            idx = lower.find(key)
            if idx < 0:
                continue
            stats[key]["occurs"] += 1
            ge = idx + len(key)
            captured = False
            for te in pred:
                ps, pe = te.span_start, te.span_end
                if ps is None or pe is None:
                    pidx = text.find(te.surface)
                    if pidx < 0:
                        continue
                    ps, pe = pidx, pidx + len(te.surface)
                overlap = max(0, min(pe, ge) - max(ps, idx))
                min_len = min(max(1, pe - ps), max(1, ge - idx))
                if overlap / min_len >= 0.5:
                    captured = True
                    break
            if captured:
                stats[key]["captured"] += 1
    total_occ = sum(s["occurs"] for s in stats.values())
    total_cap = sum(s["captured"] for s in stats.values())
    return {
        "by_surface": stats,
        "total_occurrences": total_occ,
        "total_captured": total_cap,
        "rate": total_cap / total_occ if total_occ else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_version(
    name: str,
    make_extractor,  # callable -> extractor
    all_docs: list[dict],
    all_queries: list[dict],
) -> tuple[
    dict[str, list[TimeExpression]], dict[str, list[TimeExpression]], int, list[str]
]:
    """Extract everything; return per-doc and per-query dicts + miss count."""
    counter = CacheMissCounter()
    extractor = make_extractor(counter)
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in all_docs]
    q_items = [
        (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in all_queries
    ]
    print(f"\n=== {name}: {len(doc_items)} docs + {len(q_items)} queries ===")
    pred_docs = await run_extraction(extractor, doc_items, f"{name}-docs")
    pred_queries = await run_extraction(extractor, q_items, f"{name}-queries")
    return pred_docs, pred_queries, counter.misses, counter.miss_samples


async def main() -> None:
    # ---------- load data ----------
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
    disc_docs = load_jsonl(DATA_DIR / "disc_docs.jsonl")
    disc_queries = load_jsonl(DATA_DIR / "disc_queries.jsonl")
    era_docs = load_jsonl(DATA_DIR / "era_docs.jsonl")
    era_queries = load_jsonl(DATA_DIR / "era_queries.jsonl")
    utt_docs_path = DATA_DIR / "utterance_docs.jsonl"
    utt_queries_path = DATA_DIR / "utterance_queries.jsonl"
    utt_docs = load_jsonl(utt_docs_path) if utt_docs_path.exists() else []
    utt_queries = load_jsonl(utt_queries_path) if utt_queries_path.exists() else []

    all_docs = base_docs + disc_docs + axis_docs + era_docs + utt_docs
    all_queries = base_queries + disc_queries + axis_queries + era_queries + utt_queries
    print(f"Loaded docs={len(all_docs)}, queries={len(all_queries)}")

    # failure_cases
    fc_path = RESULTS_DIR / "failure_cases.json"
    failure_cases = json.loads(fc_path.read_text()) if fc_path.exists() else []
    print(f"Failure cases: {len(failure_cases)}")

    # Docs/queries with gold
    docs_with_gold = [d for d in all_docs if d.get("gold_expressions")]
    queries_with_gold = [q for q in all_queries if q.get("gold_expressions")]
    print(
        f"Docs w/ gold_expressions: {len(docs_with_gold)}; queries w/ gold: {len(queries_with_gold)}"
    )

    # ---------- run versions ----------
    versions = [
        ("v1", lambda c: make_v1_extractor(c)),
        ("v2", lambda c: make_vx_extractor(ExtractorV2, c)),
        ("v3", lambda c: make_vx_extractor(ExtractorV3, c)),
    ]
    results: list[dict[str, Any]] = []
    for name, fn in versions:
        pred_docs, pred_queries, misses, miss_samples = await run_version(
            name, fn, all_docs, all_queries
        )

        # Extraction F1
        ext_docs = compute_extraction_f1(docs_with_gold, pred_docs, "doc_id")
        ext_queries = compute_extraction_f1(queries_with_gold, pred_queries, "query_id")
        # Combined: manually merge match lists
        combined_match_lists = []
        for d in docs_with_gold:
            pred = pred_docs.get(d["doc_id"], [])
            gold = [time_expression_from_dict(g) for g in d["gold_expressions"]]
            combined_match_lists.append(match_expressions(pred, gold, d["text"]))
        for q in queries_with_gold:
            pred = pred_queries.get(q["query_id"], [])
            gold = [time_expression_from_dict(g) for g in q["gold_expressions"]]
            combined_match_lists.append(match_expressions(pred, gold, q["text"]))
        combined = extraction_metrics(combined_match_lists)

        # Failure recall
        fc_stats = failure_recall(
            failure_cases, all_docs, all_queries, pred_docs, pred_queries
        )

        # Axis-gap rate
        axis_gap = (
            axis_gap_rate(
                all_docs + all_queries, {**pred_docs, **pred_queries}, "doc_id"
            )
            if False
            else None
        )
        # simpler: treat docs and queries separately
        doc_gap = axis_gap_rate(all_docs, pred_docs, "doc_id")
        q_gap = axis_gap_rate(all_queries, pred_queries, "query_id")

        # Downstream retrieval on base corpus — use only base docs as pool
        base_doc_mem = build_memory(
            {d["doc_id"]: pred_docs.get(d["doc_id"], []) for d in base_docs}
        )
        base_q_mem = build_memory(
            {q["query_id"]: pred_queries.get(q["query_id"], []) for q in base_queries}
        )
        base_retrieval = retrieval_metrics(
            base_doc_mem, base_q_mem, base_queries, base_gold
        )

        # Axis retrieval: axis queries against base + axis docs
        axis_doc_mem = build_memory(
            {
                d["doc_id"]: pred_docs.get(d["doc_id"], [])
                for d in (base_docs + axis_docs)
            }
        )
        axis_q_mem = build_memory(
            {q["query_id"]: pred_queries.get(q["query_id"], []) for q in axis_queries}
        )
        axis_retrieval = retrieval_metrics(
            axis_doc_mem, axis_q_mem, axis_queries, axis_gold
        )

        results.append(
            {
                "name": name,
                "cache_misses": misses,
                "miss_samples": miss_samples,
                "extraction": {
                    "docs": ext_docs,
                    "queries": ext_queries,
                    "combined": combined,
                    "n_docs_scored": len(docs_with_gold),
                    "n_queries_scored": len(queries_with_gold),
                },
                "failure_cases": fc_stats,
                "axis_gap_docs": doc_gap,
                "axis_gap_queries": q_gap,
                "retrieval_base": base_retrieval,
                "retrieval_axis": axis_retrieval,
            }
        )

        # Dump a small console summary
        print(
            f"  {name} combined F1={combined['f1']:.3f} P={combined['precision']:.3f} R={combined['recall']:.3f}"
        )
        print(
            f"  {name} failure recall: {fc_stats['recovered']}/{fc_stats['total']} ({fc_stats['recall']:.2f})"
        )
        print(
            f"  {name} axis-gap docs: {doc_gap['total_captured']}/{doc_gap['total_occurrences']}; queries: {q_gap['total_captured']}/{q_gap['total_occurrences']}"
        )
        print(
            f"  {name} base R@5={base_retrieval['recall@5']:.3f}; axis R@5={axis_retrieval['recall@5']:.3f}"
        )
        print(f"  {name} cache misses: {misses}")

    # ---------- write outputs ----------
    json_out = {"versions": results, "alpha": ALPHA, "beta": BETA, "gamma": GAMMA}
    (RESULTS_DIR / "extractor_v1_v2_v3.json").write_text(
        json.dumps(json_out, indent=2, default=str)
    )
    print(f"Wrote {RESULTS_DIR / 'extractor_v1_v2_v3.json'}")

    # Markdown
    lines: list[str] = []
    lines.append("# Extractor v1 / v2 / v3 — Cache-Only Evaluation\n\n")
    lines.append(
        "All three versions evaluated against their prior cached extractions "
        "(no new LLM calls). Scorer: multi-axis "
        f"α={ALPHA} β={BETA} γ={GAMMA}.\n\n"
    )
    lines.append("## Extraction F1\n\n")
    lines.append(
        "| Version | Combined F1 | Combined P | Combined R | Docs F1 | Queries F1 | Cache misses |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in results:
        ext = r["extraction"]
        c = ext["combined"]
        lines.append(
            f"| {r['name']} | {c['f1']:.3f} | {c['precision']:.3f} | {c['recall']:.3f} | "
            f"{ext['docs']['f1']:.3f} | {ext['queries']['f1']:.3f} | {r['cache_misses']} |\n"
        )

    lines.append("\n## Failure-case recovery (v1-missed surfaces)\n\n")
    lines.append("| Version | Recovered | Total | Rate |\n|---|---:|---:|---:|\n")
    for r in results:
        fc = r["failure_cases"]
        lines.append(
            f"| {r['name']} | {fc['recovered']} | {fc['total']} | {fc['recall']:.2f} |\n"
        )

    lines.append("\n### Per-surface recovery (docs + queries)\n\n")
    surfs = sorted(
        {s for r in results for s in r["failure_cases"].get("by_surface", {}).keys()}
    )
    lines.append("| Surface | " + " | ".join(r["name"] for r in results) + " |\n")
    lines.append("|---|" + "|".join(["---:"] * len(results)) + "|\n")
    for s in surfs:
        row = f"| `{s}` |"
        for r in results:
            st = r["failure_cases"]["by_surface"].get(s, {"recovered": 0, "total": 0})
            row += f" {st['recovered']}/{st['total']} |"
        lines.append(row + "\n")

    lines.append("\n## Axis-gap surfaces (bare months / quarters / parts-of-day)\n\n")
    lines.append(
        "Capture rate for the multi-axis experiment's flagged surfaces "
        "(" + ", ".join(AXIS_GAP_SURFACES) + ") across docs + queries.\n\n"
    )
    lines.append(
        "| Version | Docs captured | Queries captured | Total rate |\n|---|---:|---:|---:|\n"
    )
    for r in results:
        dg = r["axis_gap_docs"]
        qg = r["axis_gap_queries"]
        total_occ = dg["total_occurrences"] + qg["total_occurrences"]
        total_cap = dg["total_captured"] + qg["total_captured"]
        rate = total_cap / total_occ if total_occ else 0.0
        lines.append(
            f"| {r['name']} | {dg['total_captured']}/{dg['total_occurrences']} | "
            f"{qg['total_captured']}/{qg['total_occurrences']} | {rate:.2f} |\n"
        )

    lines.append("\n### Per-surface axis-gap rate (docs + queries combined)\n\n")
    lines.append("| Surface | " + " | ".join(r["name"] for r in results) + " |\n")
    lines.append("|---|" + "|".join(["---:"] * len(results)) + "|\n")
    for s in AXIS_GAP_SURFACES:
        key = s.lower()
        row = f"| `{s}` |"
        for r in results:
            d = r["axis_gap_docs"]["by_surface"].get(key, {"occurs": 0, "captured": 0})
            q = r["axis_gap_queries"]["by_surface"].get(
                key, {"occurs": 0, "captured": 0}
            )
            occ = d["occurs"] + q["occurs"]
            cap = d["captured"] + q["captured"]
            row += f" {cap}/{occ} |"
        lines.append(row + "\n")

    lines.append("\n## Downstream retrieval (base 55 queries)\n\n")
    lines.append(
        "| Version | R@5 | R@10 | MRR | NDCG@10 |\n|---|---:|---:|---:|---:|\n"
    )
    for r in results:
        m = r["retrieval_base"]
        lines.append(
            f"| {r['name']} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg@10']:.3f} |\n"
        )

    lines.append("\n## Axis-subset retrieval (20 axis queries)\n\n")
    lines.append(
        "| Version | R@5 | R@10 | MRR | NDCG@10 |\n|---|---:|---:|---:|---:|\n"
    )
    for r in results:
        m = r["retrieval_axis"]
        lines.append(
            f"| {r['name']} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg@10']:.3f} |\n"
        )

    lines.append("\n## Cost\n\n")
    total_misses = sum(r["cache_misses"] for r in results)
    lines.append(
        f"- All three versions ran cache-only: total uncached LLM requests = **{total_misses}**.\n"
        "- LLM token spend: **$0.00** (stubbed client; cache misses returned empty).\n"
    )

    (RESULTS_DIR / "extractor_v1_v2_v3.md").write_text("".join(lines))
    print(f"Wrote {RESULTS_DIR / 'extractor_v1_v2_v3.md'}")


if __name__ == "__main__":
    asyncio.run(main())
