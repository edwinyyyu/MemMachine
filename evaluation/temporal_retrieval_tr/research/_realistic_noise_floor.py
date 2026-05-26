"""Measure realistic rerank-noise floor: among docs that are clearly
"about the same thing" as the query, what's the actual cosine gap?

Methodology:
- For each query in same-topic benches, identify the topic-equivalent
  docs (typically scenario family: all `<scenario>_d*` + `<scenario>_g0`)
- Compute query→doc cosine similarity for each
- Report max-min gap within the topic group: the actual rerank noise
  that recency must overcome

The noise from our discrimination test (0.05, 0.10, 0.20) needs to
correspond to something realistic. If real same-topic gaps are typically
<0.02, our noise injection was unrealistic and Copeland's advantage is
moot in practice. If real gaps are 0.05-0.20, the discrimination is
production-relevant.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._realistic_noise_floor
"""
from __future__ import annotations

import asyncio
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from temporal_retrieval.research._common import DATA_DIR, make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import make_cached_embed_fn

setup_env()


# Map bench → function that returns same-topic groups per query.
# A "same-topic group" is the set of docs we'd consider equally relevant
# to the query (only recency should disambiguate).
SAME_TOPIC_BENCHES = [
    "composition",         # comp_A_NNN_g0/d0/d1/d2 are same-topic
    "same_topic_recency",  # str_TOPIC_NN_d*/g0 are same-topic
    "same_topic_recency_hard",  # recency_hard_TOPIC_NN_d*/g0
    "recency_stress_deep",      # stress_TOPIC_NN_d*/g0
    "recency_vs_rerank",        # rvr_TOPIC_NN_d*/g0
]


def scenario_key(doc_id: str) -> str:
    """Extract scenario identifier from doc_id by stripping trailing _d*/_g* suffix."""
    parts = doc_id.rsplit("_", 1)
    if len(parts) == 2 and (parts[1].startswith("d") or parts[1].startswith("g")):
        if parts[1][1:].isdigit() or parts[1] in ("g0", "a"):
            return parts[0]
    return doc_id


def load_bench_raw(bench: str):
    with open(DATA_DIR / f"{bench}_docs.jsonl") as f:
        docs = [json.loads(line) for line in f]
    with open(DATA_DIR / f"{bench}_queries.jsonl") as f:
        queries = [json.loads(line) for line in f]
    with open(DATA_DIR / f"{bench}_gold.jsonl") as f:
        gold = [json.loads(line) for line in f]
    return docs, queries, gold


def query_scenario(query_id: str) -> str:
    """Map a query_id to its scenario key for same-topic grouping.

    Conventions across benches:
      comp_q_A_000 → comp_A_000
      str_q_morningrun_001 → str_morningrun_001
      recency_hard_q_stretching_001 → recency_hard_stretching_001
      stress_q_morningrun_001 → stress_morningrun_001
      rvr_q_morningrun_001 → rvr_morningrun_001
    """
    return query_id.replace("_q_", "_", 1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) or 1e-9
    nb = float(np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / (na * nb))


async def analyze_bench(bench: str, embed_fn) -> list[dict] | None:
    try:
        docs, queries, _ = load_bench_raw(bench)
    except FileNotFoundError:
        return None

    # Group docs by scenario key
    docs_by_scenario: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        docs_by_scenario[scenario_key(d["doc_id"])].append(d)

    # Embed all docs and queries
    doc_id_list = [d["doc_id"] for d in docs]
    doc_text_list = [d["text"] for d in docs]
    doc_embs_arr = await embed_fn(doc_text_list)
    doc_emb = dict(zip(doc_id_list, doc_embs_arr, strict=False))

    query_id_list = [q["query_id"] for q in queries]
    query_text_list = [q["text"] for q in queries]
    query_embs_arr = await embed_fn(query_text_list)
    query_emb = dict(zip(query_id_list, query_embs_arr, strict=False))

    rows = []
    for q in queries:
        qid = q["query_id"]
        scen = query_scenario(qid)
        group_docs = docs_by_scenario.get(scen, [])
        if len(group_docs) < 2:
            continue
        qe = query_emb[qid]
        sims = []
        for d in group_docs:
            sim = cosine(qe, doc_emb[d["doc_id"]])
            sims.append((d["doc_id"], sim))
        sims.sort(key=lambda x: -x[1])
        raw_min = min(s for _, s in sims)
        raw_max = max(s for _, s in sims)
        rows.append({
            "bench": bench,
            "query_id": qid,
            "query_text": q["text"],
            "scenario": scen,
            "n_docs_in_group": len(sims),
            "raw_cosine_max": raw_max,
            "raw_cosine_min": raw_min,
            "raw_cosine_gap": raw_max - raw_min,
            "top_doc": sims[0][0],
            "bottom_doc": sims[-1][0],
            "all_sims": sims,
        })
    return rows


def report(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"=== {label}: no data ===\n")
        return
    raw_gaps = [r["raw_cosine_gap"] for r in rows]
    print(f"\n=== {label} (n_queries={len(rows)}) ===")
    print(f"  Raw cosine gap (max-min within same-topic group):")
    print(f"    min  = {min(raw_gaps):.4f}")
    print(f"    p25  = {statistics.quantiles(raw_gaps, n=4)[0]:.4f}" if len(raw_gaps) >= 4 else "")
    print(f"    p50  = {statistics.median(raw_gaps):.4f}")
    print(f"    p75  = {statistics.quantiles(raw_gaps, n=4)[2]:.4f}" if len(raw_gaps) >= 4 else "")
    print(f"    p90  = {statistics.quantiles(raw_gaps, n=10)[8]:.4f}" if len(raw_gaps) >= 10 else "")
    print(f"    max  = {max(raw_gaps):.4f}")
    print(f"    mean = {statistics.mean(raw_gaps):.4f}")

    # Distribution bins
    bins = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    print(f"  Distribution of raw cosine gaps:")
    for lo, hi in zip(bins[:-1], bins[1:]):
        c = sum(1 for g in raw_gaps if lo <= g < hi)
        if c:
            print(f"    [{lo:.2f}, {hi:.2f}): {c:>3d}")


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    print("=== Realistic noise floor: cosine gap within same-topic doc groups ===\n",
          flush=True)
    print("Methodology: per query, find docs in the same scenario family", flush=True)
    print("(same-topic decoys + gold), measure max-min cosine similarity gap\n",
          flush=True)

    all_rows = []
    for bench in SAME_TOPIC_BENCHES:
        rows = await analyze_bench(bench, embed_fn)
        if rows is None:
            print(f"  {bench}: SKIPPED (missing files)")
            continue
        report(rows, bench)
        all_rows.extend(rows)

    report(all_rows, "ALL BENCHES POOLED")

    # Pick a few queries to spot-check
    print("\n=== Sample of high-gap cases (rerank noise that recency must overcome) ===")
    all_rows.sort(key=lambda r: -r["raw_cosine_gap"])
    for r in all_rows[:8]:
        print(f"  bench={r['bench']:30s}  gap={r['raw_cosine_gap']:.3f}")
        print(f"    Q: {r['query_text']}")
        for did, sim in r["all_sims"][:5]:
            print(f"      cos={sim:.3f}  {did}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
