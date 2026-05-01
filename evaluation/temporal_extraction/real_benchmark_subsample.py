"""Subsample the TempReason-derived benchmark into a smaller (~60-query)
real_benchmark_small_*.jsonl set covering 35 L2 + 25 L3 queries plus
exactly the docs each query needs.

Usage: uv run python real_benchmark_subsample.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

N_L2 = 35
N_L3 = 25
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, items) -> None:
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def main() -> None:
    docs = load_jsonl(DATA_DIR / "real_benchmark_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "real_benchmark_queries.jsonl")
    gold = {
        g["query_id"]: g["relevant_doc_ids"]
        for g in load_jsonl(DATA_DIR / "real_benchmark_gold.jsonl")
    }
    docs_by_id = {d["doc_id"]: d for d in docs}

    # Build entity -> doc_ids set: docs sharing an entity tend to come from
    # the same query. We approximate "needed by query" by gold docs only —
    # but to give the L2 query distractors that share an entity, we re-add
    # all sibling docs for each gold doc.
    rng = random.Random(SEED)

    l2_qs = [q for q in queries if q["subset"] == "L2"]
    l3_qs = [q for q in queries if q["subset"] == "L3"]
    rng.shuffle(l2_qs)
    rng.shuffle(l3_qs)

    sampled = l2_qs[:N_L2] + l3_qs[:N_L3]
    sampled_qids = {q["query_id"] for q in sampled}

    # Collect gold doc_ids for each sampled query, then add same-entity
    # siblings from the original corpus. We treat docs as "siblings" if
    # they share at least 3 leading whitespace-separated tokens (entity
    # name) — the TempReason format is "<Entity ...> <rel> ... from X to Y."
    def entity_key(text: str) -> str:
        # Take everything before " works for "/" plays for "/" head of "/etc.
        # Heuristic: first 3 tokens.
        toks = text.split()
        return " ".join(toks[:3]).lower()

    needed_doc_ids: set[str] = set()
    for q in sampled:
        gids = gold.get(q["query_id"], [])
        needed_doc_ids.update(gids)

    # Add sibling docs (same leading tokens) per gold doc.
    sibling_map: dict[str, list[str]] = {}
    for d in docs:
        sibling_map.setdefault(entity_key(d["text"]), []).append(d["doc_id"])

    for did in list(needed_doc_ids):
        text = docs_by_id[did]["text"]
        for sib in sibling_map.get(entity_key(text), []):
            needed_doc_ids.add(sib)

    sampled_docs = [docs_by_id[d] for d in sorted(needed_doc_ids) if d in docs_by_id]
    sampled_gold = [
        {"query_id": q["query_id"], "relevant_doc_ids": gold[q["query_id"]]}
        for q in sampled
        if q["query_id"] in gold
    ]

    print(
        f"Sampled: {len(sampled)} queries (L2={N_L2}, L3={N_L3}), {len(sampled_docs)} docs"
    )

    write_jsonl(DATA_DIR / "real_benchmark_small_docs.jsonl", sampled_docs)
    write_jsonl(DATA_DIR / "real_benchmark_small_queries.jsonl", sampled)
    write_jsonl(DATA_DIR / "real_benchmark_small_gold.jsonl", sampled_gold)

    print(f"Wrote real_benchmark_small_*.jsonl to {DATA_DIR}")


if __name__ == "__main__":
    main()
