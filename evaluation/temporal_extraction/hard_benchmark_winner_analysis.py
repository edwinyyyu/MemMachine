"""Re-do failure analysis against V7 cv=0.10 (the winning config).

Reads the existing extraction caches and recomputes V7 cv=0.10 rankings,
then prints win/loss examples vs SEMANTIC-ONLY.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# patch matching tempreason_fast_run
import extractor_common  # noqa: E402

_orig_call = extractor_common.BaseImprovedExtractor._call


async def _patched_call(self, *args, **kwargs):
    original_create = self.client.chat.completions.create

    async def patched_create(**call_kwargs):
        model = call_kwargs.get("model", "")
        if isinstance(model, str) and model.startswith("gpt-5"):
            call_kwargs["reasoning_effort"] = "minimal"
        return await original_create(**call_kwargs)

    self.client.chat.completions.create = patched_create
    try:
        return await _orig_call(self, *args, **kwargs)
    finally:
        self.client.chat.completions.create = original_create


extractor_common.BaseImprovedExtractor._call = _patched_call

from datetime import datetime, timedelta, timezone

from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from baselines import embed_all  # noqa: E402
from expander import expand  # noqa: E402
from extractor_v2 import ExtractorV2  # noqa: E402
from multi_axis_scorer import axis_score as axis_score_fn  # noqa: E402
from multi_axis_scorer import tag_score
from multi_axis_tags import tags_for_axes  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from rag_fusion import score_blend  # noqa: E402
from schema import GRANULARITY_ORDER, TimeExpression, parse_iso, to_us  # noqa: E402
from scorer import Interval, score_jaccard_composite  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache" / "hard_bench_v2"

PER_CALL_TIMEOUT_S = 60.0
CONCURRENCY = 6


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


def rank_v7(t, s, weights=None, cv_ref=0.10):
    if weights is None:
        weights = {"T": 0.5, "S": 0.5}
    fused = score_blend(
        {"T": t, "S": s}, weights, top_k_per=40, dispersion_cv_ref=cv_ref
    )
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


async def run_extract_cached_only(items, cache_subdir):
    """Use cached extractions only — don't make new LLM calls."""
    ex = ExtractorV2(concurrency=CONCURRENCY, cache_subdir=cache_subdir)
    ex.client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S, max_retries=1)

    results = {}
    for iid, text, ref in items:
        try:
            tes = await ex.extract(text, ref)
            results[iid] = tes
        except Exception:
            results[iid] = []
    return results


async def main():
    docs = load_jsonl(DATA_DIR / "hard_bench_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "hard_bench_queries.jsonl")
    gold_raw = load_jsonl(DATA_DIR / "hard_bench_gold.jsonl")
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_raw}

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    print("Loading extractions from cache...", flush=True)
    doc_ext = await run_extract_cached_only(doc_items, "hard_bench_v2_docs")
    q_ext = await run_extract_cached_only(q_items, "hard_bench_v2_queries")

    print("Building memory...", flush=True)
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

    print("Embedding...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    # Compute rankings
    sem_ranks: dict[str, list[str]] = {}
    v7_ranks: dict[str, list[str]] = {}
    t_ranks: dict[str, list[str]] = {}
    all_doc_ids = [d["doc_id"] for d in docs]
    for q in queries:
        qid = q["query_id"]
        qv = q_embs[qid]
        qn = np.linalg.norm(qv) or 1e-9
        s_scores = {
            d: float(np.dot(qv, v) / (qn * (np.linalg.norm(v) or 1e-9)))
            for d, v in doc_embs.items()
        }
        t_scores = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        sem_ranks[qid] = [
            d for d, _ in sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        v7_ranks[qid] = rank_v7(t_scores, s_scores, cv_ref=0.10)
        t_ranks[qid] = sorted(
            all_doc_ids,
            key=lambda d: (t_scores.get(d, 0.0), s_scores.get(d, 0.0)),
            reverse=True,
        )

    def rank_of(rl, rel):
        for i, d in enumerate(rl, start=1):
            if d in rel:
                return i
        return None

    rank_compare = []
    for q in queries:
        qid = q["query_id"]
        rel = gold[qid]
        if not rel:
            continue
        sem_r = rank_of(sem_ranks[qid], rel)
        v7_r = rank_of(v7_ranks[qid], rel)
        t_r = rank_of(t_ranks[qid], rel)
        gold_id = next(iter(rel))
        gold_text = next(d["text"] for d in docs if d["doc_id"] == gold_id)
        rank_compare.append(
            {
                "qid": qid,
                "subset": q["subset"],
                "query": q["text"],
                "gold_text": gold_text,
                "rank_sem": sem_r,
                "rank_v7": v7_r,
                "rank_t": t_r,
                "sem_top3": [
                    next(d["text"] for d in docs if d["doc_id"] == did)
                    for did in sem_ranks[qid][:3]
                ],
                "v7_top3": [
                    next(d["text"] for d in docs if d["doc_id"] == did)
                    for did in v7_ranks[qid][:3]
                ],
                "n_q_extractions": len(q_ext.get(qid, [])),
            }
        )

    wins = [
        r
        for r in rank_compare
        if (r["rank_sem"] is None or r["rank_sem"] > 1) and r["rank_v7"] == 1
    ]
    losses = [
        r
        for r in rank_compare
        if r["rank_sem"] == 1 and (r["rank_v7"] is None or r["rank_v7"] > 1)
    ]
    persistent = [
        r
        for r in rank_compare
        if (r["rank_sem"] is None or r["rank_sem"] > 5)
        and (r["rank_v7"] is None or r["rank_v7"] > 5)
    ]

    print("\n=== V7 cv=0.10 vs SEMANTIC-ONLY ===")
    print(f"Wins (V7 rank=1, sem rank>1): {len(wins)}")
    print(f"Losses (V7 rank>1, sem rank=1): {len(losses)}")
    print(f"Persistent misses (both >5): {len(persistent)}")

    print("\n=== TOP V7 WINS ===")
    for r in wins[:10]:
        print(f'  [{r["subset"]}] {r["qid"]}: "{r["query"]}"')
        print(f"    gold: {r['gold_text']}")
        print(
            f"    sem rank={r['rank_sem']} -> V7 rank=1, T-only rank={r['rank_t']}, q_extractions={r['n_q_extractions']}"
        )
        print("    sem_top3:")
        for i, t in enumerate(r["sem_top3"], 1):
            print(f"      {i}. {t}")
        print()

    print("\n=== TOP V7 LOSSES ===")
    for r in losses[:10]:
        print(f'  [{r["subset"]}] {r["qid"]}: "{r["query"]}"')
        print(f"    gold: {r['gold_text']}")
        print(
            f"    sem rank=1 -> V7 rank={r['rank_v7']}, T-only rank={r['rank_t']}, q_extractions={r['n_q_extractions']}"
        )
        print("    V7_top3:")
        for i, t in enumerate(r["v7_top3"], 1):
            print(f"      {i}. {t}")
        print()

    print("\n=== PERSISTENT MISSES (both lost) ===")
    for r in persistent[:5]:
        print(f'  [{r["subset"]}] {r["qid"]}: "{r["query"]}"')
        print(f"    gold: {r['gold_text']}")
        print(
            f"    sem rank={r['rank_sem']}, V7 rank={r['rank_v7']}, T-only rank={r['rank_t']}, q_extractions={r['n_q_extractions']}"
        )
        print("    sem_top3:")
        for i, t in enumerate(r["sem_top3"], 1):
            print(f"      {i}. {t}")
        print()

    # Save
    out = {
        "n_total": len(rank_compare),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_persistent": len(persistent),
        "wins": wins[:10],
        "losses": losses[:10],
        "persistent": persistent[:10],
    }
    out_path = RESULTS_DIR / "hard_bench_winner_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
