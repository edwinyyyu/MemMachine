"""Pure-semantic baseline: rank docs by cosine alone.

No extractor, no planner, no temporal pool filtering — just
cosine(query, doc_text) over the full corpus, take top-K.

This is the true "zero-temporal" floor. Compares against:
- base_only ranking (temporal-filtered pool + base rank): 0.7989
- additive+0.8 (same):                                    0.7989
- copeland_pairwise (same):                               0.8044

If pure-semantic ≈ base_only, the temporal layer is contributing
nothing — pool filtering is just shaping which docs get reranked
but the rerank picks the same top-K regardless.

If pure-semantic << base_only, the temporal pool filtering IS
doing meaningful work, even though the temporal RANKING layer
is mostly inert.
"""
from __future__ import annotations

import asyncio, gc

import numpy as np

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, metrics,
)

setup_env()


async def run_bench(bench, embed_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded

    # Embed all docs once
    doc_ids = [d["doc_id"] for d in docs_jsonl]
    doc_texts = [d["text"] for d in docs_jsonl]
    doc_embs = await embed_fn(doc_texts)
    doc_mat = np.stack(
        [np.asarray(e, dtype=np.float32) for e in doc_embs], axis=0
    )
    doc_norms = np.linalg.norm(doc_mat, axis=1) + 1e-9

    rankings: dict[str, list[str]] = {}
    for q in queries:
        q_emb = (await embed_fn([q["text"]]))[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_emb)) + 1e-9
        sims = (doc_mat @ q_emb) / (doc_norms * q_norm)
        order = np.argsort(-sims)[:10]
        rankings[q["query_id"]] = [doc_ids[i] for i in order]

    m = metrics(rankings, gold)
    gc.collect()
    return m


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    print("\n=== Pure-semantic baseline (cosine top-K, no temporal) ===\n",
          flush=True)
    rows = []
    for bench in BENCH_NAMES:
        try:
            m = await run_bench(bench, embed_fn)
        except Exception as e:
            print(f"  ERROR {bench}: {e}", flush=True)
            continue
        if m is None:
            continue
        rows.append((bench, m))
        print(
            f"  {bench:30s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  n={m['n']}",
            flush=True,
        )
    n = len(rows)
    r1 = sum(m["R@1"] for _, m in rows) / n
    r5 = sum(m["R@5"] for _, m in rows) / n
    print(
        f"\nMACRO pure-semantic:        R@1={r1:.4f}  R@5={r5:.4f}  n={n}",
        flush=True,
    )
    print(
        "MACRO base_only/additive:   R@1=0.7989  R@5=0.9611",
        flush=True,
    )
    print(
        "MACRO copeland_pairwise:    R@1=0.8044  R@5=0.9618",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
