"""A/B test: filter state-doc results for event-locator queries.

Mechanism: after the retriever's `query()` returns top-K, filter results
by removing state-classified docs — but only when the query's target is
narrow (a specific date/window) rather than an era. The target-width
gate distinguishes event-locator from state-locator queries.

Why this works as a probe: a wide planner target (years/decades)
signals the query is asking about an era/period — state docs are the
natural answer there. A narrow target (one day, one month) signals an
event-locator query — state docs would be distractors.

Threshold: 31 days = one calendar month. Targets ≤ 31 days are
considered event-locator; wider targets are state-locator and no
filtering applies.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._state_event_ab
"""
from __future__ import annotations

import asyncio
import gc
import json

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval_tr.time_range import is_inf
from temporal_retrieval.research._common import DATA_DIR, make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)
from temporal_retrieval_tr.research._state_event_llm_classifier import get_classifier

setup_env()

# Target-width threshold for event-locator vs state-locator classification.
# 31 days ≈ 1 calendar month: narrow targets (specific dates) are
# event-locator queries; wider targets are about eras/periods.
EVENT_LOCATOR_MAX_TARGET_US = 31 * 86_400_000_000  # 31 days


def is_event_locator_query(plan_targets) -> bool:
    """True iff any target is narrow enough to be event-locator-shaped.

    Wide targets (years, decades) signal era/period queries where state
    docs are valid answers. Narrow targets (one day, one month) signal
    specific event-locator queries where state docs are distractors.
    """
    if not plan_targets:
        return False
    for target in plan_targets:
        for iv in target.intervals:
            # Unbounded intervals are eras / open-ended — not event-locator
            if is_inf(iv.earliest_us) or is_inf(iv.latest_us):
                continue
            width = iv.latest_us - iv.earliest_us
            if width <= EVENT_LOCATOR_MAX_TARGET_US:
                return True
    return False


async def run_bench(bench: str, embed_fn, rerank_fn, k_top: int = 10) -> tuple[dict, dict, int]:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None, None, 0
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    # Classify all docs in parallel via LLM (cached)
    classifier = get_classifier()
    is_state_flags = await classifier.classify_many([d["text"] for d in docs_jsonl])
    classifier.save()
    state_classified = {d["doc_id"]
                        for d, flag in zip(docs_jsonl, is_state_flags, strict=True)
                        if flag}

    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)

    # Only filter state-classified docs that ALSO have extracted temporal
    # anchors. Timeless state-shape docs (policies / rules with no extractable
    # dates) surface only via semantic top-up — filtering them costs gold
    # without preventing any temporal-credit abuse.
    state_docs = {did for did in state_classified if vd._doc_anchors.get(did)}

    rk_baseline: dict[str, list[str]] = {}
    rk_state_filter: dict[str, list[str]] = {}
    for q in queries:
        results = await vd.query(q["text"], q["ref_time"], k=k_top * 3)
        baseline = [r.doc_id for r in results][:k_top]
        # Only filter for event-locator queries (narrow target width)
        plan = await vd._planner.plan(q["text"], q["ref_time"])
        if is_event_locator_query(plan.targets):
            filtered = [r.doc_id for r in results
                        if r.doc_id not in state_docs][:k_top]
        else:
            filtered = baseline  # no filtering for state-locator queries
        rk_baseline[q["query_id"]] = baseline
        rk_state_filter[q["query_id"]] = filtered

    m_baseline = metrics(rk_baseline, gold)
    m_filter = metrics(rk_state_filter, gold)
    del vd
    gc.collect()
    return m_baseline, m_filter, len(state_docs)


async def main() -> None:
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"\n=== State-event filter A/B over {len(BENCH_NAMES)} benches ===\n", flush=True)
    hdr = (f"{'bench':30s}  {'base R@1':>8s} {'sef R@1':>8s} {'ΔR@1':>7s}  "
           f"{'base R@5':>8s} {'sef R@5':>8s} {'#state':>7s}  {'n':>4s}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    for bench in BENCH_NAMES:
        try:
            mb, mf, n_state = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:30s}  ERROR: {e}", flush=True)
            continue
        if mb is None:
            continue
        rows[bench] = (mb, mf, n_state)
        d_r1 = mf["R@1"] - mb["R@1"]
        mark = ">" if abs(d_r1) >= 0.02 else " "
        print(f"{mark} {bench:28s}  {mb['R@1']:>8.3f} {mf['R@1']:>8.3f} {d_r1:>+7.3f}  "
              f"{mb['R@5']:>8.3f} {mf['R@5']:>8.3f} {n_state:>7d}  {mb['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        macro_b_r1 = sum(r[0]["R@1"] for r in rows.values()) / n
        macro_f_r1 = sum(r[1]["R@1"] for r in rows.values()) / n
        macro_b_r5 = sum(r[0]["R@5"] for r in rows.values()) / n
        macro_f_r5 = sum(r[1]["R@5"] for r in rows.values()) / n
        print("-" * len(hdr), flush=True)
        print(f"  {'MACRO':28s}  {macro_b_r1:>8.4f} {macro_f_r1:>8.4f} "
              f"{macro_f_r1 - macro_b_r1:>+7.4f}  "
              f"{macro_b_r5:>8.4f} {macro_f_r5:>8.4f}      n={n}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
