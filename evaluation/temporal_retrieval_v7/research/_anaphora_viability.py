"""Viability probe for anaphora resolution.

When a query says "after the migration", V7-Direct emits refs=[] and
drops the anaphor. A 2-stage resolver would: (1) semantic-search the
anaphoric phrase over the corpus, (2) find the event-defining anchor
doc, (3) extract its date, (4) apply the relation -> a TimeRange.

THE MUST-PASS GATE: can step (1) reliably find the anchor doc? This
probe embeds each C/D query's anaphoric phrase and cosine-ranks the
composition corpus, reporting the anchor doc's rank. If anchor docs
are reliably top-1 (or top-3), the resolver is viable.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._anaphora_viability
"""
from __future__ import annotations

import asyncio
import json

import numpy as np

from temporal_retrieval.research._common import DATA_DIR, make_embed_fn, setup_env

setup_env()

# (query_id, anaphoric phrase, anchor doc id)
CASES = [
    ("comp_q_C_010", "the migration", "comp_C_010_a"),
    ("comp_q_C_011", "the launch", "comp_C_011_a"),
    ("comp_q_C_012", "the last review", "comp_C_012_a"),
    ("comp_q_C_013", "the redesign", "comp_C_013_a"),
    ("comp_q_C_014", "the offsite", "comp_C_014_a"),
    ("comp_q_D_015", "the launch", "comp_D_015_a"),
    ("comp_q_D_016", "the migration", "comp_D_016_a"),
    ("comp_q_D_017", "the freeze", "comp_D_017_a"),
    ("comp_q_D_018", "year-end review", "comp_D_018_a"),
    ("comp_q_D_019", "the kickoff", "comp_D_019_a"),
]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) or 1e-9) * (np.linalg.norm(b) or 1e-9)))


async def main() -> None:
    embed_fn = await make_embed_fn()
    docs = [json.loads(l) for l in
            open(DATA_DIR / "composition_docs.jsonl")]
    doc_ids = [d["doc_id"] for d in docs]
    doc_text = {d["doc_id"]: d["text"] for d in docs}
    doc_embs = await embed_fn([d["text"] for d in docs])
    doc_embs = [np.asarray(e, dtype=np.float32) for e in doc_embs]

    # Two probe variants: bare phrase, and "When did X happen?"
    print("=== Anaphora resolution viability: anchor-doc retrieval ===\n",
          flush=True)
    print(f"{'query':16s} {'phrase':22s} {'bare rank':>10s} {'when-q rank':>12s}",
          flush=True)
    print("-" * 64, flush=True)

    bare_ranks, whenq_ranks = [], []
    for qid, phrase, anchor_id in CASES:
        bare_emb = np.asarray((await embed_fn([phrase]))[0], dtype=np.float32)
        whenq = f"When did {phrase} happen?"
        whenq_emb = np.asarray((await embed_fn([whenq]))[0], dtype=np.float32)

        def rank_of(qemb):
            scored = sorted(
                ((_cos(qemb, doc_embs[i]), doc_ids[i]) for i in range(len(docs))),
                reverse=True,
            )
            for r, (_, did) in enumerate(scored, 1):
                if did == anchor_id:
                    return r, scored[:3]
            return -1, scored[:3]

        b_rank, b_top = rank_of(bare_emb)
        w_rank, w_top = rank_of(whenq_emb)
        bare_ranks.append(b_rank)
        whenq_ranks.append(w_rank)
        print(f"{qid:16s} {phrase:22s} {b_rank:>10d} {w_rank:>12d}", flush=True)
        if b_rank > 3:
            print(f"   bare top3: {[t[1] for t in b_top]}", flush=True)
        if w_rank > 3:
            print(f"   when-q top3: {[t[1] for t in w_top]}", flush=True)

    print("-" * 64, flush=True)
    for label, ranks in [("bare phrase", bare_ranks), ("when-q", whenq_ranks)]:
        top1 = sum(1 for r in ranks if r == 1)
        top3 = sum(1 for r in ranks if 1 <= r <= 3)
        print(f"{label:14s}: top-1 {top1}/10   top-3 {top3}/10", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
