"""Anaphora probe — Stage A (query-side): can we find the anchor doc?

For each failing query in causal_relative (and a subset of allen / edge_era_refs),
extract the anaphor noun phrase, semantic-search for it, and check whether
the correct anchor doc lands in top-1/top-3.

This is the prerequisite probe: if anchor docs aren't recoverable via
semantic search, the resolver mechanism has a ceiling regardless of how
it's wired in. A success here doesn't prove the full resolver works —
just that the upstream signal exists.

Hand-annotated anaphor extraction for transparency: we want to know if
the *retrieval primitive* works, not whether GPT can parse the query.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._anaphora_probe_stage_a
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from temporal_retrieval.research._common import DATA_DIR, make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import make_cached_embed_fn

setup_env()


# Hand-annotated anaphor extractions for causal_relative.
# Format: query_id -> (anaphor_phrase, expected_anchor_doc_id)
CR_ANAPHORS = {
    "cr_q_000": ("the migration",          "cr_000_a"),
    "cr_q_001": ("the launch",             "cr_001_a"),
    "cr_q_002": ("the last review",        "cr_002_a"),
    "cr_q_003": ("the offsite",            "cr_003_a"),
    "cr_q_004": ("the merger",             "cr_004_a"),
    "cr_q_005": ("the funding round",      "cr_005_a"),
    "cr_q_006": ("the keynote",            "cr_006_a"),
    "cr_q_007": ("the audit",              "cr_007_a"),
    "cr_q_008": ("the move",               "cr_008_a"),
    "cr_q_009": ("the cutover",            "cr_009_a"),
    "cr_q_010": ("the design summit",      "cr_010_a"),
    "cr_q_011": ("the marathon",           "cr_011_a"),
    "cr_q_012": ("the relocation",         "cr_012_a"),
    "cr_q_013": ("the client onsite",      "cr_013_a"),
    "cr_q_014": ("the promotion",          "cr_014_a"),
}

# Allen — anaphor is "my <event>"; the anchor doc is `a_<event>`.
# These docs DO carry dates, so even Stage A is enough to verify retrieval.
ALLEN_ANAPHORS = {
    "q_before_wedding":   ("my wedding",        "a_wedding"),
    "q_before_grad":      ("my graduation",     "a_graduation"),
    "q_before_marathon":  ("the marathon",      "a_marathon"),
    "q_before_europe":    ("my Europe trip",    "a_europe_trip"),
    "q_after_wedding":    ("my wedding",        "a_wedding"),
    "q_after_promo":      ("my promotion",      "a_promotion"),
    "q_after_move":       ("my move to Denver", "a_move_denver"),
    "q_during_europe":    ("my Europe trip",    "a_europe_trip"),
    "q_during_honeymoon": ("my honeymoon",      "a_honeymoon"),
    "q_during_wedding":   ("my wedding",        "a_wedding"),
    "q_during_conference":("the conference",    "a_conference"),
}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) or 1e-9
    nb = float(np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / (na * nb))


async def probe_bench(bench: str, anaphors: dict[str, tuple[str, str]],
                      embed_fn) -> None:
    docs_path = DATA_DIR / f"{bench}_docs.jsonl"
    with open(docs_path) as f:
        docs = [json.loads(line) for line in f]
    doc_ids = [d["doc_id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    doc_embs = await embed_fn(doc_texts)
    by_id = {d["doc_id"]: d for d in docs}

    phrases = sorted({p for p, _ in anaphors.values()})
    phrase_embs_list = await embed_fn(phrases)
    phrase_embs = dict(zip(phrases, phrase_embs_list, strict=True))

    print(f"\n{'=' * 78}", flush=True)
    print(f"=== STAGE A probe: {bench}", flush=True)
    print(f"{'=' * 78}", flush=True)

    hits_top1 = 0
    hits_top3 = 0
    n = len(anaphors)
    for qid, (phrase, expected) in anaphors.items():
        qe = phrase_embs[phrase]
        scored = [(doc_ids[i], cosine(qe, doc_embs[i]))
                  for i in range(len(doc_ids))]
        scored.sort(key=lambda x: -x[1])
        top3 = [d for d, _ in scored[:3]]
        rank = (top3.index(expected) + 1) if expected in top3 else None
        if rank == 1:
            hits_top1 += 1
        if rank is not None:
            hits_top3 += 1
        mark = "✓" if rank == 1 else ("·" if rank else "✗")
        print(f"  {mark} {qid:24s}  '{phrase}' → expected {expected}", flush=True)
        for r, (d, s) in enumerate(scored[:3], 1):
            tag = "★" if d == expected else " "
            print(f"      {tag} #{r} {d:18s} {s:+.4f}  {by_id[d]['text'][:90]}",
                  flush=True)
    print(f"\n  {bench}: top-1 = {hits_top1}/{n} = {hits_top1/n:.2%}   "
          f"top-3 = {hits_top3}/{n} = {hits_top3/n:.2%}", flush=True)


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    await probe_bench("causal_relative", CR_ANAPHORS, embed_fn)
    await probe_bench("allen", ALLEN_ANAPHORS, embed_fn)


if __name__ == "__main__":
    asyncio.run(main())
