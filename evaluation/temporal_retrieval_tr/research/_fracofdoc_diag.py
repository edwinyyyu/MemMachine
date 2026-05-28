"""Diagnose adversarial/disc/axis flips under frac-of-doc vs frac-min.

For each query where the two arms disagree on the gold's rank, dump:
  - query text + ref_time
  - planner targets (width)
  - gold doc text + extractor anchor widths
  - top-3 retrieved under each arm
  - rerank cosine for gold vs top-1 displacer

Goal: identify whether frac-of-doc's adversarial loss is
  (a) gold has a broad anchor that frac-of-doc tanks (the predicted failure mode)
  (b) something else — distractor with narrow anchor displacing
  (c) bench-design (gold itself shouldn't beat the displacer semantically)

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._fracofdoc_diag
"""
from __future__ import annotations

import asyncio

import temporal_retrieval_tr.scoring as scoring_mod
from temporal_retrieval_tr import Doc, IntervalSet, TemporalRetriever
from temporal_retrieval_tr.time_range import (
    Endpoint, intersect, is_empty, is_inf, measure,
)

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    load_bench, make_cached_embed_fn, make_cosine_rerank_fn,
)

setup_env()

BENCHES = ["adversarial", "disc", "axis"]
ONE_DAY_US = 86_400_000_000


def pair_overlap_doc_denom(A: IntervalSet, B: IntervalSet) -> float:
    inter = intersect(A, B)
    if is_empty(inter):
        return 0.0
    a_w = measure(A)
    b_w = measure(B)
    inter_w = measure(inter)
    if is_inf(a_w) and is_inf(b_w):
        return 1.0
    if is_inf(b_w):
        return 0.0
    if b_w <= 0:
        return 0.0
    inter_w_val = 0 if is_inf(inter_w) else inter_w
    return min(1.0, inter_w_val / b_w)


def fmt_width(w: Endpoint) -> str:
    """Human-readable width for a measure."""
    if is_inf(w):
        return "∞"
    days = w / ONE_DAY_US
    if days < 1:
        return f"{w/3_600_000_000:.0f}h"
    if days < 31:
        return f"{days:.0f}d"
    if days < 366:
        return f"{days/30.4:.1f}mo"
    return f"{days/365.25:.1f}yr"


def anchor_widths(anchors: list[IntervalSet]) -> str:
    """Compact display of each anchor's width."""
    out = []
    for a in anchors:
        for iv in a.intervals:
            out.append(fmt_width(iv.width))
    return f"[{', '.join(out)}]"


def target_widths(targets: list[IntervalSet]) -> str:
    out = []
    for t in targets:
        for iv in t.intervals:
            out.append(fmt_width(iv.width))
    return f"[{', '.join(out)}]"


def snippet(text: str, n: int = 90) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + ("…" if len(text) > n else "")


async def diag_bench(bench: str, embed_fn, rerank_fn) -> None:
    docs_jsonl, queries, gold = load_bench(bench)
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    doc_lookup = {d["doc_id"]: d for d in docs_jsonl}

    print(f"\n{'=' * 88}", flush=True)
    print(f"=== {bench} ({len(queries)} queries)", flush=True)
    print(f"{'=' * 88}", flush=True)

    # Two runs: frac-min, frac-of-doc
    original = scoring_mod.pair_overlap

    # frac-min
    vd_fmin = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd_fmin.index(docs)
    res_fmin = {}
    for q in queries:
        r = await vd_fmin.query(q["text"], q["ref_time"], k=10)
        res_fmin[q["query_id"]] = r

    # frac-of-doc
    scoring_mod.pair_overlap = pair_overlap_doc_denom
    try:
        vd_fdoc = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
        await vd_fdoc.index(docs)
        res_fdoc = {}
        for q in queries:
            r = await vd_fdoc.query(q["text"], q["ref_time"], k=10)
            res_fdoc[q["query_id"]] = r
    finally:
        scoring_mod.pair_overlap = original

    # Find disagreements where gold ranks differ
    for q in queries:
        qid = q["query_id"]
        gset = gold.get(qid, set())
        if not gset:
            continue
        rfmin = [r.doc_id for r in res_fmin[qid]]
        rfdoc = [r.doc_id for r in res_fdoc[qid]]

        # find gold's rank under each arm
        def first_rank(ranking, gset):
            for i, did in enumerate(ranking, 1):
                if did in gset:
                    return i
            return None

        gr_fmin = first_rank(rfmin, gset)
        gr_fdoc = first_rank(rfdoc, gset)
        if gr_fmin == gr_fdoc:
            continue
        # Only show cases where ranking diverges
        plan = await vd_fmin._planner.plan(q["text"], q["ref_time"])
        gold_id = next(iter(gset))
        gold_anchors = vd_fmin._doc_anchors.get(gold_id, [])

        print(f"\n--- {qid}  rank fmin={gr_fmin} → fdoc={gr_fdoc} ---", flush=True)
        print(f"  Q: {q['text']}  (ref={q['ref_time']})", flush=True)
        print(f"  PLAN: targets={target_widths(plan.targets)} "
              f"latest={plan.latest_intent} earliest={plan.earliest_intent}",
              flush=True)
        print(f"  GOLD {gold_id}: anchors={anchor_widths(gold_anchors)}", flush=True)
        print(f"    text: {snippet(doc_lookup[gold_id]['text'])}", flush=True)
        print(f"  fmin top-3:", flush=True)
        for i, r in enumerate(res_fmin[qid][:3], 1):
            ax = anchor_widths(vd_fmin._doc_anchors.get(r.doc_id, []))
            tag = "✓" if r.doc_id in gset else " "
            print(f"    {tag} #{i} {r.doc_id} score={r.score:.3f} "
                  f"(rerank={r.rerank:.3f} match={r.match:.3f}) anchors={ax}",
                  flush=True)
            print(f"      {snippet(doc_lookup[r.doc_id]['text'])}", flush=True)
        print(f"  fdoc top-3:", flush=True)
        for i, r in enumerate(res_fdoc[qid][:3], 1):
            ax = anchor_widths(vd_fdoc._doc_anchors.get(r.doc_id, []))
            tag = "✓" if r.doc_id in gset else " "
            print(f"    {tag} #{i} {r.doc_id} score={r.score:.3f} "
                  f"(rerank={r.rerank:.3f} match={r.match:.3f}) anchors={ax}",
                  flush=True)
            print(f"      {snippet(doc_lookup[r.doc_id]['text'])}", flush=True)


async def main() -> None:
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    for bench in BENCHES:
        await diag_bench(bench, embed_fn, rerank_fn)


if __name__ == "__main__":
    asyncio.run(main())
