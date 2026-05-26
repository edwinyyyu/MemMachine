"""Does additive fail with noise? Controlled noise injection.

Inject uniform(-Δ/2, +Δ/2) into each doc's normalized base score per query,
then measure R@1 for additive(W=0.5) vs Copeland(0.20) at varying
pool sizes.

Theoretical prediction:
- Additive per-pair recency gap = W/(N-1). For W=0.5, N=200: 0.0025.
  Fails when noise > recency gap.
- Copeland per-pair bonus = 0.20. Independent of N.
  Fails when noise > bonus.

So additive should fail at lower noise and at larger pool sizes,
while Copeland holds up to noise=0.20.

Test bench: recency_stress_deep — 17 scenarios × ~8 same-topic docs.
With identical text, recency is the ONLY signal that can resolve
gold from decoys. Adding noise to base means we directly stress
how well the recency mechanism survives rerank-noise pressure.

Cache for queries/docs is warm — only retriever-side logic varies.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._noise_discrimination
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_tr import Doc, TemporalRetriever

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    load_bench,
    make_cached_embed_fn,
    make_cosine_rerank_fn,
    metrics,
)

setup_env()

NOISE_SCALES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
POOL_SIZES = [40, 80, 200]
ARMS: list[tuple[str, str, float]] = [
    ("add_W0.5",  "additive", 0.5),
    ("cope_0.20", "copeland", 0.20),
]
TARGET_BENCHES = ["recency_stress_deep", "same_topic_recency_hard"]


async def run_cell(
    bench: str, pool_size: int, noise_scale: float, embed_fn, rerank_fn
) -> dict:
    loaded = load_bench(bench)
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        pool_size=pool_size,
        noise_scale=noise_scale,
    )
    await vd.index(docs)
    out = {}
    for label, mode, value in ARMS:
        if mode == "additive":
            vd.recency_weight = value
            vd.copeland_bonus = None
        else:
            vd.copeland_bonus = value
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[label] = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== Noise discrimination ===", flush=True)
    print(f"Noise scales: {NOISE_SCALES}", flush=True)
    print(f"Pool sizes:   {POOL_SIZES}\n", flush=True)

    for bench in TARGET_BENCHES:
        print(f"\n========== {bench} ==========\n", flush=True)
        for ps in POOL_SIZES:
            print(f"--- pool_size = {ps} ---", flush=True)
            hdr = (f"  {'noise':>6s} | {'add R@1':>9s} {'cope R@1':>10s} {'Δ(c−a)':>8s}  |  "
                   f"{'add R@5':>9s} {'cope R@5':>10s}  |  "
                   f"{'add R@10':>10s} {'cope R@10':>11s}")
            print(hdr, flush=True)
            print("  " + "-" * (len(hdr) - 2), flush=True)
            for noise in NOISE_SCALES:
                res = await run_cell(bench, ps, noise, embed_fn, rerank_fn)
                a = res["add_W0.5"]
                c = res["cope_0.20"]
                d = c["R@1"] - a["R@1"]
                print(f"  {noise:>6.3f} | {a['R@1']:>9.3f} {c['R@1']:>10.3f} "
                      f"{d:>+8.3f}  |  "
                      f"{a['R@5']:>9.3f} {c['R@5']:>10.3f}  |  "
                      f"{a['R@10']:>10.3f} {c['R@10']:>11.3f}",
                      flush=True)
            print(flush=True)


if __name__ == "__main__":
    asyncio.run(main())
