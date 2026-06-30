"""Does MMR rescue the buried Copeland-recency verdict for a recency cue?

Empirical test of the claim "MMR helps surface the correct-but-crowded memory."
For each cue: pull the store's top-N cosine pool, re-embed with embeddinggemma,
then re-rank by (a) plain cosine and (b) MMR over an alpha sweep, and report the
rank of the FIRST gold memory (a genuine pre-this-session Copeland-recency turn).

Standard MMR: greedily pick argmax_d [ alpha*cos(q,d) - (1-alpha)*max_{s in S} cos(d,s) ].
alpha=1.0 == plain cosine (the baseline). Same cosine metric for both terms.
Relevance uses the asymmetric query embedding; diversity uses doc embeddings.
"""

import asyncio
from datetime import UTC, datetime

import numpy as np

from claude_memory.engine import (
    DISPLAY_FORMAT,
    EventMemory,
    MemoryConfig,
    MemoryCore,
    memory_id_for_segment_uuid,
)

POOL_N = 200
ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
# Self-generated Copeland turns must NOT count as gold; this whole conversation
# happened on/after Jun 11. The experiment work was Apr 30 - Jun 5.
GOLD_CUTOFF = datetime(2026, 6, 11, tzinfo=UTC)

CUES = [
    "What did we conclude was the best way to do recency scoring -- rank-based "
    "recency with a recency_weight, vs. an anchor-date 'closest to this date' "
    "bonus, and did we keep or remove the two-extrema recency/extremum system?",
    "best way to do recency scoring for extremum latest/earliest queries",
    "how should recency ranking combine with the base relevance score",
]


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _is_gold(text: str, timestamp: datetime) -> bool:
    return "copeland" in text.lower() and timestamp < GOLD_CUTOFF


def _mmr_order(rel: np.ndarray, doc_sim: np.ndarray, alpha: float) -> list[int]:
    n = len(rel)
    selected: list[int] = []
    remaining = set(range(n))
    while remaining:
        if not selected:
            pick = max(remaining, key=lambda i: rel[i])
        else:
            sel = np.array(selected)
            pick = max(
                remaining,
                key=lambda i: alpha * rel[i] - (1 - alpha) * doc_sim[i, sel].max(),
            )
        selected.append(pick)
        remaining.discard(pick)
    return selected


def _first_gold(order: list[int], gold: list[bool]) -> int | None:
    for rank, idx in enumerate(order, start=1):
        if gold[idx]:
            return rank
    return None


async def run_cue(core: MemoryCore, cue: str) -> None:
    result = await core.memory.query(
        cue,
        vector_search_limit=POOL_N,
        expand_context=0,
        format_options=DISPLAY_FORMAT,
    )
    contexts = result.scored_segment_contexts
    texts, store_scores, golds, ids = [], [], [], []
    for ctx in contexts:
        seed = next(s for s in ctx.segments if s.uuid == ctx.seed_segment_uuid)
        text = EventMemory.string_from_segment_context(
            ctx.segments, format_options=DISPLAY_FORMAT
        )
        texts.append(text)
        store_scores.append(ctx.score)
        golds.append(_is_gold(text, seed.timestamp))
        ids.append(memory_id_for_segment_uuid(ctx.seed_segment_uuid))

    emb = core.stores.embedder
    qv = _l2(np.array(await emb.search_embed([cue]), dtype=float))[0]
    dv = _l2(np.array(await emb.ingest_embed(texts), dtype=float))
    rel = dv @ qv
    doc_sim = dv @ dv.T

    n_gold = sum(golds)
    print(f"\n=== cue: {cue[:70]}...")
    print(f"pool={len(texts)}  gold-in-pool={n_gold}")
    if n_gold == 0:
        print("  no gold in pool -> MMR cannot help (out-of-pool); skipping")
        return

    store_order = sorted(range(len(texts)), key=lambda i: -store_scores[i])
    cos_order = list(np.argsort(-rel))

    # Label-free pool-redundancy signal: mean off-diagonal doc-doc cosine of the
    # cosine top-K. High => the pool is dominated by near-duplicates (diversity
    # should help). Also redundancy among items ABOVE the first gold (oracle, for
    # diagnosis only -- needs the label).
    def redundancy(idx: list[int]) -> float:
        if len(idx) < 2:
            return float("nan")
        sub = doc_sim[np.ix_(idx, idx)]
        return (sub.sum() - np.trace(sub)) / (len(idx) * (len(idx) - 1))

    fg_cos = _first_gold(cos_order, golds)
    above = cos_order[: (fg_cos - 1)] if fg_cos else cos_order
    print(
        f"  redundancy(meanpair-cos): top20={redundancy(cos_order[:20]):.3f} "
        f"top50={redundancy(cos_order[:50]):.3f} "
        f"above-first-gold={redundancy(above):.3f}"
    )
    print(
        f"  first-gold rank | store-cosine={_first_gold(store_order, golds)} "
        f"| reembed-cosine={fg_cos}"
    )
    # show the first gold's identity for eyeballing
    fg = next(i for i in cos_order if golds[i])
    print(
        f"  first gold (reembed-cos): rank{cos_order.index(fg) + 1} {ids[fg][:12]} "
        f":: {texts[fg][:90].replace(chr(10), ' ')}"
    )
    print("  alpha :  first-gold-rank  golds@top10  golds@top20")
    for alpha in ALPHAS:
        order = _mmr_order(rel, doc_sim, alpha)
        fgr = _first_gold(order, golds)
        g10 = sum(golds[i] for i in order[:10])
        g20 = sum(golds[i] for i in order[:20])
        tag = "  (== cosine)" if alpha == 1.0 else ""
        print(f"  {alpha:<5} :  {fgr!s:<15} {g10:<12} {g20}{tag}")


async def main() -> None:
    cfg = MemoryConfig.load()
    print(f"store home: {cfg.home}  backend={cfg.vector_backend}")
    core = await MemoryCore.open(cfg)
    try:
        for cue in CUES:
            await run_cue(core, cue)
    finally:
        await core.aclose()


if __name__ == "__main__":
    asyncio.run(main())
