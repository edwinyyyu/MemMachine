"""Show concrete same-topic doc pairs at specific raw cosine gap magnitudes.

For each gap bucket (~0.05, 0.10, 0.15, 0.20, 0.30), print examples of
(query, gold_text, decoy_text) along with their cosine similarities.
Grounds the abstract noise discussion in actual text.
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict

import numpy as np

from temporal_retrieval.research._common import DATA_DIR, make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import make_cached_embed_fn

setup_env()

BENCHES = [
    "composition", "same_topic_recency", "same_topic_recency_hard",
    "recency_stress_deep", "recency_vs_rerank",
]

# Gap buckets: (target_gap, tolerance, label)
BUCKETS = [
    (0.05, 0.015, "0.05"),
    (0.10, 0.015, "0.10"),
    (0.15, 0.020, "0.15"),
    (0.20, 0.025, "0.20"),
    (0.25, 0.025, "0.25"),
    (0.30, 0.030, "0.30"),
]


def scenario_key(doc_id: str) -> str:
    parts = doc_id.rsplit("_", 1)
    if len(parts) == 2:
        suffix = parts[1]
        if suffix in ("g0", "a") or (suffix.startswith("d") and suffix[1:].isdigit()):
            return parts[0]
    return doc_id


def query_scenario(query_id: str) -> str:
    return query_id.replace("_q_", "_", 1)


def cosine(a, b) -> float:
    na = float(np.linalg.norm(a)) or 1e-9
    nb = float(np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / (na * nb))


async def gather_pairs(embed_fn) -> list[dict]:
    """For each query, compute cosines for all docs in same-scenario group.
    Return pairs (top-cos, bottom-cos within group) with gap."""
    rows = []
    for bench in BENCHES:
        try:
            with open(DATA_DIR / f"{bench}_docs.jsonl") as f:
                docs = [json.loads(line) for line in f]
            with open(DATA_DIR / f"{bench}_queries.jsonl") as f:
                queries = [json.loads(line) for line in f]
            with open(DATA_DIR / f"{bench}_gold.jsonl") as f:
                gold_rows = [json.loads(line) for line in f]
        except FileNotFoundError:
            continue
        gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
        doc_text = {d["doc_id"]: d["text"] for d in docs}
        docs_by_scen: dict[str, list] = defaultdict(list)
        for d in docs:
            docs_by_scen[scenario_key(d["doc_id"])].append(d["doc_id"])

        # Embed all docs + queries
        doc_id_list = list(doc_text.keys())
        doc_embs = await embed_fn([doc_text[d] for d in doc_id_list])
        doc_emb = dict(zip(doc_id_list, doc_embs, strict=False))
        query_id_list = [q["query_id"] for q in queries]
        query_embs = await embed_fn([q["text"] for q in queries])
        query_emb = dict(zip(query_id_list, query_embs, strict=False))

        for q in queries:
            qid = q["query_id"]
            scen = query_scenario(qid)
            group = docs_by_scen.get(scen, [])
            if len(group) < 2:
                continue
            qe = query_emb[qid]
            sims = [(did, cosine(qe, doc_emb[did])) for did in group]
            sims.sort(key=lambda x: -x[1])
            top_did, top_sim = sims[0]
            bot_did, bot_sim = sims[-1]
            gap = top_sim - bot_sim
            gold_set = gold.get(qid, set())
            gold_did = next((d for d in group if d in gold_set), None)
            rows.append({
                "bench": bench,
                "query_id": qid,
                "query_text": q["text"],
                "top_did": top_did,
                "top_sim": top_sim,
                "top_text": doc_text[top_did],
                "bot_did": bot_did,
                "bot_sim": bot_sim,
                "bot_text": doc_text[bot_did],
                "gold_did": gold_did,
                "gold_text": doc_text.get(gold_did, "(gold not in pool)") if gold_did else "(no gold)",
                "gold_sim": (cosine(qe, doc_emb[gold_did]) if gold_did else None),
                "gap": gap,
            })
    return rows


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rows = await gather_pairs(embed_fn)
    rows.sort(key=lambda r: r["gap"])

    print("=== Concrete same-topic pairs at target gap magnitudes ===\n", flush=True)
    print("(Each scenario: query + the doc with highest cosine + the doc with lowest cosine)", flush=True)
    print("The 'gap' = top_sim − bot_sim. Both docs are in the same scenario family.\n", flush=True)

    for target, tol, label in BUCKETS:
        bucket = [r for r in rows if abs(r["gap"] - target) <= tol]
        if not bucket:
            print(f"\n========== gap ≈ {label} (no examples in ±{tol}) ==========\n")
            continue
        # Take up to 3 examples per bucket
        print(f"\n========== gap ≈ {label} ({len(bucket)} cases; showing top 3) ==========")
        for r in bucket[:3]:
            print(f"\n  bench={r['bench']:30s}  qid={r['query_id']}  gap={r['gap']:.3f}")
            print(f"    Q: {r['query_text']}")
            top_is_gold = r['top_did'] == r['gold_did']
            bot_is_gold = r['bot_did'] == r['gold_did']
            print(f"    TOP  ({r['top_did']:35s}  cos={r['top_sim']:.3f})"
                  f"{'  [GOLD]' if top_is_gold else ''}: {r['top_text'][:130]}")
            print(f"    BOT  ({r['bot_did']:35s}  cos={r['bot_sim']:.3f})"
                  f"{'  [GOLD]' if bot_is_gold else ''}: {r['bot_text'][:130]}")
            if r['gold_did'] and not (top_is_gold or bot_is_gold):
                print(f"    GOLD ({r['gold_did']:35s}  cos={r['gold_sim']:.3f}): "
                      f"{r['gold_text'][:130]}")

    # And the very-high-gap ones (>0.30) for completeness
    high = sorted([r for r in rows if r["gap"] > 0.30], key=lambda r: -r["gap"])[:3]
    if high:
        print(f"\n========== gap > 0.30 ({len(high)} shown — beyond bonus=0.40 reach) ==========")
        for r in high:
            print(f"\n  bench={r['bench']:30s}  qid={r['query_id']}  gap={r['gap']:.3f}")
            print(f"    Q: {r['query_text']}")
            top_is_gold = r['top_did'] == r['gold_did']
            bot_is_gold = r['bot_did'] == r['gold_did']
            print(f"    TOP  ({r['top_did']:35s}  cos={r['top_sim']:.3f})"
                  f"{'  [GOLD]' if top_is_gold else ''}: {r['top_text'][:130]}")
            print(f"    BOT  ({r['bot_did']:35s}  cos={r['bot_sim']:.3f})"
                  f"{'  [GOLD]' if bot_is_gold else ''}: {r['bot_text'][:130]}")
            if r['gold_did'] and not (top_is_gold or bot_is_gold):
                print(f"    GOLD ({r['gold_did']:35s}  cos={r['gold_sim']:.3f}): "
                      f"{r['gold_text'][:130]}")


if __name__ == "__main__":
    asyncio.run(main())
