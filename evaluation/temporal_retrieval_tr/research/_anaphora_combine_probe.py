"""Combine-without-LLM anchor probe.

For each causal_relative failing query, run:
  - Baseline: current production (no anchor lookup)
  - Combine-top-K: resolve anaphor via top-K semantic search;
    for each candidate anchor's date + query relation, compute a
    target IntervalSet; score docs against the UNION/MAX of these
    targets.

Comparison metric: R@1 on the 15-query bench.

Hypothesis (user's): combining without LLM disambiguation gives a
broader target window and will probably HURT R@1 — the wrong-anchor
intervals admit too many distractors.

Surprise case: if combine somehow helps anyway, we know there's
extra signal in the candidate set we should keep.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._anaphora_combine_probe
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime

import numpy as np

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval_tr.time_range import (
    NEG_INF, POS_INF, Interval, IntervalSet,
)
from temporal_retrieval_tr.scoring import final_score
from temporal_retrieval_min.schema import parse_iso, to_us
from temporal_retrieval.research._common import DATA_DIR, make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()


# (anaphor_phrase, relation) for each causal_relative query.
CR_ANNOT = {
    "cr_q_000": ("the migration",          "after"),
    "cr_q_001": ("the launch",             "before"),
    "cr_q_002": ("the last review",        "since"),
    "cr_q_003": ("the offsite",            "after"),
    "cr_q_004": ("the merger",             "before"),
    "cr_q_005": ("the funding round",      "after"),
    "cr_q_006": ("the keynote",            "before"),
    "cr_q_007": ("the audit",              "after"),
    "cr_q_008": ("the move",               "since"),
    "cr_q_009": ("the cutover",            "after"),
    "cr_q_010": ("the design summit",      "after"),
    "cr_q_011": ("the marathon",           "before"),
    "cr_q_012": ("the relocation",         "since"),
    "cr_q_013": ("the client onsite",      "after"),
    "cr_q_014": ("the promotion",          "after"),
}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) or 1e-9
    nb = float(np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / (na * nb))


def make_relation_target(anchor_date_us: int, relation: str) -> IntervalSet:
    """Build the query target IntervalSet from anchor date and relation.

    before  → [-inf, anchor)
    after   → (anchor, +inf]
    since   → [anchor, +inf]
    during  → [anchor, anchor + small) — punt for now, treat as after
    """
    if relation == "before":
        return IntervalSet((Interval(NEG_INF + 1, anchor_date_us),))
    if relation in ("after", "since"):
        return IntervalSet((Interval(anchor_date_us, POS_INF - 1),))
    if relation == "during":
        # treat anchor as a single moment; doc matches if anchor is in doc interval
        return IntervalSet((Interval(anchor_date_us, anchor_date_us + 86_400_000_000),))
    return IntervalSet((Interval(NEG_INF + 1, POS_INF - 1),))


async def run_combine_probe(top_k: int, embed_fn, rerank_fn,
                            baseline_results: dict) -> dict:
    bench = "causal_relative"
    docs_jsonl, queries, gold = load_bench(bench)
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    doc_lookup = {d["doc_id"]: d for d in docs_jsonl}

    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)

    # Pre-embed all docs (already cached) + all phrases
    doc_ids = [d.id for d in docs]
    doc_texts = [d.text for d in docs]
    doc_embs = await embed_fn(doc_texts)

    phrases = sorted({p for p, _ in CR_ANNOT.values()})
    phrase_embs_list = await embed_fn(phrases)
    phrase_embs = dict(zip(phrases, phrase_embs_list, strict=True))

    rankings = {}
    for q in queries:
        qid = q["query_id"]
        phrase, relation = CR_ANNOT[qid]
        # Top-K anchor candidates by semantic similarity
        pe = phrase_embs[phrase]
        scored = sorted(
            [(doc_ids[i], cosine(pe, doc_embs[i])) for i in range(len(doc_ids))],
            key=lambda x: -x[1],
        )
        top_candidates = scored[:top_k]

        # Build a multi-target list from each candidate's ref_time
        all_targets: list[IntervalSet] = []
        for cand_id, _ in top_candidates:
            cand_doc = doc_lookup[cand_id]
            anchor_us = to_us(parse_iso(cand_doc["ref_time"]))
            all_targets.append(make_relation_target(anchor_us, relation))

        # Score each doc with each target, take max
        # Combine with base semantic ranking like the production retriever
        # does for non-extremum queries. We use the same pool/scoring as
        # production but inject the anaphora-derived match.
        q_emb = (await embed_fn([q["text"]]))[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sem_scores = vd._cosine_all(q_emb)

        match_all: dict[str, float] = {}
        for did in doc_ids:
            d_anchors = vd._doc_anchors.get(did, [])
            if not d_anchors:
                match_all[did] = 0.0
                continue
            scores = [final_score([t], d_anchors) for t in all_targets]
            match_all[did] = max(scores) if scores else 0.0

        # Now run the production retriever's pool selection + scoring,
        # but with the injected match scores
        from temporal_retrieval_min.core import build_pool
        eligible = [did for did in doc_ids if match_all.get(did, 0.0) > 0.0]
        pool = build_pool(sem_scores, doc_ids, eligible, vd.pool_size)
        pool_texts = [doc_lookup[did]["text"] for did in pool]
        rerank_scores = await rerank_fn(q["text"], pool_texts)
        rerank_pool = dict(zip(pool, rerank_scores, strict=False))
        base_vals = list(rerank_pool.values())
        pool_spread = max(base_vals) - min(base_vals) if base_vals else 0.0
        final = {
            did: rerank_pool[did] + match_all.get(did, 0.0) * pool_spread
            for did in pool
        }
        ranked = sorted(pool, key=lambda d: -final[d])
        rankings[qid] = ranked[:10]

    m = metrics(rankings, gold)
    return m


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    bench = "causal_relative"
    docs_jsonl, queries, gold = load_bench(bench)
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)
    rk = {}
    for q in queries:
        r = await vd.query(q["text"], q["ref_time"], k=10)
        rk[q["query_id"]] = [x.doc_id for x in r]
    baseline = metrics(rk, gold)
    print(f"\n=== Combine probe on causal_relative (n=15) ===\n", flush=True)
    print(f"  baseline (production, no anchor lookup):  "
          f"R@1={baseline['R@1']:.3f}  R@5={baseline['R@5']:.3f}", flush=True)

    for k in [1, 2, 3, 5]:
        m = await run_combine_probe(k, embed_fn, rerank_fn, baseline)
        print(f"  combine top-{k}:                           "
              f"R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
