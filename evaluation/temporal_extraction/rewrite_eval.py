"""F11 — Temporal query rewriting + fusion evaluation.

Compares three retrieval modes on five subsets (base 55, axis 20,
utterance 10, era 20, allen 20):

1. "baseline"      — no rewriting; extract original query and retrieve.
2. "rrf"           — RRF fusion over [original, *K variants].
3. "max"           — max-of-scores fusion over [original, *K variants].

All subsets share a single IntervalStore populated by running the base
``Extractor`` over the union of docs across subsets. Variant extractions
run in an isolated cache so the base cache is not polluted.

Metrics: R@5, R@10, MRR, NDCG@10 per subset, plus cost.

Outputs ``results/query_rewrite.{md,json}``.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from extractor import Extractor
from query_rewriter import QueryRewriter
from rewrite_retrieval import (
    build_variant_extractor,
    rank_docs,
    retrieve_with_rewrites,
)
from schema import parse_iso
from store import IntervalStore

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
REWRITE_DB_PATH = ROOT / "cache" / "rewrite" / "intervals.sqlite"
REWRITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# gpt-5-mini prices
PRICE_IN_PER_M = 0.25
PRICE_OUT_PER_M = 2.00

TOP_K = 10
RRF_K = 60


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


SUBSETS = [
    ("base", "docs.jsonl", "queries.jsonl", "gold.jsonl"),
    ("axis", "axis_docs.jsonl", "axis_queries.jsonl", "axis_gold.jsonl"),
    (
        "utterance",
        "utterance_docs.jsonl",
        "utterance_queries.jsonl",
        "utterance_gold.jsonl",
    ),
    ("era", "era_docs.jsonl", "era_queries.jsonl", "era_gold.jsonl"),
    ("allen", "allen_docs.jsonl", "allen_queries.jsonl", "allen_gold.jsonl"),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return float("nan")
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def mean(xs: list[float]) -> float:
    vs = [v for v in xs if not math.isnan(v)]
    return sum(vs) / len(vs) if vs else 0.0


def eval_block(
    ranked_per_q: dict[str, list[str]],
    gold: dict[str, set[str]],
    qids: Iterable[str],
) -> dict[str, float]:
    r5, r10, mr, nd = [], [], [], []
    n = 0
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            continue
        ranked = ranked_per_q.get(qid, [])
        r5.append(recall_at_k(ranked, rel, 5))
        r10.append(recall_at_k(ranked, rel, 10))
        mr.append(mrr(ranked, rel))
        nd.append(ndcg_at_k(ranked, rel, TOP_K))
        n += 1
    return {
        "recall@5": mean(r5),
        "recall@10": mean(r10),
        "mrr": mean(mr),
        "ndcg@10": mean(nd),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
async def build_shared_store(
    all_docs: list[dict], db_path: Path
) -> tuple[IntervalStore, Extractor, dict[str, int]]:
    """Extract every doc once using the base Extractor (cached) and populate
    a fresh IntervalStore.
    """
    if db_path.exists():
        db_path.unlink()
    store = IntervalStore(db_path)
    ex = Extractor()

    async def one(d):
        did = d["doc_id"]
        try:
            tes = await ex.extract(d["text"], parse_iso(d["ref_time"]))
        except Exception as e:
            print(f"  doc extract failed for {did}: {e}")
            tes = []
        return did, tes

    print(f"Extracting {len(all_docs)} docs (base cache)...")
    results = await asyncio.gather(*(one(d) for d in all_docs))
    ex.cache.save()
    for did, tes in results:
        for te in tes:
            store.insert_expression(did, te)
    return store, ex, ex.usage


async def main() -> None:
    overall_start = time.time()

    # 1. Load every subset
    docs_map: dict[str, list[dict]] = {}
    queries_map: dict[str, list[dict]] = {}
    gold_map: dict[str, dict[str, set[str]]] = {}
    for name, docs_f, queries_f, gold_f in SUBSETS:
        docs_map[name] = load_jsonl(DATA_DIR / docs_f)
        queries_map[name] = load_jsonl(DATA_DIR / queries_f)
        gold_list = load_jsonl(DATA_DIR / gold_f)
        gold_map[name] = {
            r["query_id"]: set(r.get("relevant_doc_ids", [])) for r in gold_list
        }
        print(
            f"  loaded {name}: {len(docs_map[name])} docs, "
            f"{len(queries_map[name])} queries, {len(gold_list)} gold rows"
        )

    # Union docs by doc_id (dedup in case of overlap)
    all_docs_dedup: dict[str, dict] = {}
    for name, ds in docs_map.items():
        for d in ds:
            all_docs_dedup[d["doc_id"]] = d
    all_docs = list(all_docs_dedup.values())
    all_doc_ids = list(all_docs_dedup.keys())
    print(f"Total unique docs across subsets: {len(all_doc_ids)}")

    # 2. Build shared IntervalStore
    store, doc_ex, doc_usage = await build_shared_store(all_docs, REWRITE_DB_PATH)
    doc_cost = (
        doc_usage["input"] * PRICE_IN_PER_M / 1_000_000
        + doc_usage["output"] * PRICE_OUT_PER_M / 1_000_000
    )
    print(
        f"Doc extraction usage: input={doc_usage['input']}, out={doc_usage['output']} (${doc_cost:.4f})"
    )

    # 3. Generate rewrites for every query
    all_queries: list[dict] = []
    subset_qids: dict[str, list[str]] = {}
    for name, qs in queries_map.items():
        subset_qids[name] = [q["query_id"] for q in qs]
        all_queries.extend(qs)

    # Dedup queries by query_id (in case of accidental overlap).
    seen_qids: set[str] = set()
    unique_queries: list[dict] = []
    for q in all_queries:
        if q["query_id"] in seen_qids:
            continue
        seen_qids.add(q["query_id"])
        unique_queries.append(q)
    print(f"Total queries: {len(unique_queries)}")

    rewriter = QueryRewriter()
    rewrite_items = [(q["query_id"], q["text"], q["ref_time"]) for q in unique_queries]
    print(f"Rewriting {len(rewrite_items)} queries via gpt-5-mini...")
    rw_start = time.time()
    variants_by_qid = await rewriter.rewrite_many(rewrite_items)
    rw_cost = rewriter.cost_usd()
    print(
        f"  rewrite usage: in={rewriter.usage['input']}, out={rewriter.usage['output']} "
        f"(${rw_cost:.4f}), wall={time.time() - rw_start:.1f}s"
    )

    # Print a small sample for report
    sample_rewrites: list[dict] = []
    sample_qids_for_report = [
        q["query_id"]
        for q in unique_queries
        if q["text"].strip()
        and any(
            key in q["query_id"]
            for key in [
                "rel_day",
                "fuzzy",
                "rec",
                "notime",
                "interval",
                "utt",
                "era",
                "axis",
                "allen",
            ]
        )
    ][:20]
    for qid in sample_qids_for_report:
        q = next(q for q in unique_queries if q["query_id"] == qid)
        sample_rewrites.append(
            {
                "qid": qid,
                "text": q["text"],
                "variants": variants_by_qid.get(qid, []),
            }
        )

    # 4. Run retrieval for each query under three modes.
    variant_ex = build_variant_extractor()
    print(
        f"Using variant extractor cache at {variant_ex.cache.path} "
        f"(warm-started from base cache if available)"
    )

    baseline_ranked: dict[str, list[str]] = {}
    rrf_ranked: dict[str, list[str]] = {}
    max_ranked: dict[str, list[str]] = {}
    per_query_debug: dict[str, dict[str, Any]] = {}

    print(f"Running retrieval for {len(unique_queries)} queries...")
    ret_start = time.time()
    # We concurrently process queries; each query itself triggers K+1 extractions.
    # Outer semaphore is large — the real bottleneck is the extractor's
    # internal Semaphore (set to 20 via build_variant_extractor).
    semaphore = asyncio.Semaphore(20)
    done_count = {"n": 0}

    async def one_query(q: dict) -> None:
        qid = q["query_id"]
        ref_time = parse_iso(q["ref_time"])
        variants = [
            v for v in variants_by_qid.get(qid, []) if v.strip() != q["text"].strip()
        ]
        async with semaphore:
            # Run retrieval under RRF to collect per-variant scores; then reuse
            # those scores for baseline + max.
            fused_rrf, per_scores, per_tes = await retrieve_with_rewrites(
                variant_ex,
                store,
                q["text"],
                variants,
                ref_time,
                all_doc_ids,
                fuse_mode="rrf",
            )
        done_count["n"] += 1
        if done_count["n"] % 10 == 0 or done_count["n"] == len(unique_queries):
            print(
                f"    ...{done_count['n']}/{len(unique_queries)} queries done, "
                f"wall={time.time() - ret_start:.0f}s",
                flush=True,
            )
            # Periodic cache save so we don't lose progress on interruption
            variant_ex.cache.save()
        baseline_scores = per_scores.get(q["text"], {})
        baseline_ranked[qid] = rank_docs(baseline_scores, all_doc_ids)
        rrf_ranked[qid] = rank_docs(fused_rrf, all_doc_ids)
        # max fusion from per-variant scores
        max_scores: dict[str, float] = defaultdict(float)
        for _t, sc in per_scores.items():
            for d, s in sc.items():
                if s > max_scores[d]:
                    max_scores[d] = s
        max_ranked[qid] = rank_docs(dict(max_scores), all_doc_ids)

        per_query_debug[qid] = {
            "text": q["text"],
            "variants": list(per_scores.keys())[1:],
            "variant_hit_counts": {
                t: sum(1 for v in sc.values() if v > 0) for t, sc in per_scores.items()
            },
            "n_tes_per_variant": {t: len(tes) for t, tes in per_tes.items()},
        }

    await asyncio.gather(*(one_query(q) for q in unique_queries))
    # Persist variant-extractor cache for reruns
    variant_ex.cache.save()
    print(f"  retrieval wall={time.time() - ret_start:.1f}s")

    variant_extract_usage = dict(variant_ex.usage)
    variant_extract_cost = (
        variant_extract_usage["input"] * PRICE_IN_PER_M / 1_000_000
        + variant_extract_usage["output"] * PRICE_OUT_PER_M / 1_000_000
    )

    # 5. Evaluate per subset.
    results: dict[str, dict[str, dict[str, float]]] = {}
    modes = {
        "baseline": baseline_ranked,
        "rrf": rrf_ranked,
        "max": max_ranked,
    }
    global_gold: dict[str, set[str]] = {}
    for g in gold_map.values():
        global_gold.update(g)

    for name in ["base", "axis", "utterance", "era", "allen"]:
        qids = subset_qids[name]
        results[name] = {}
        for mode_name, ranked in modes.items():
            results[name][mode_name] = eval_block(ranked, global_gold, qids)

    # Overall across all subsets (flat average).
    all_qids = [qid for qids in subset_qids.values() for qid in qids]
    results["all"] = {}
    for mode_name, ranked in modes.items():
        results["all"][mode_name] = eval_block(ranked, global_gold, all_qids)

    # 6. Failure / help analysis.
    help_rows: list[dict[str, Any]] = []
    hurt_rows: list[dict[str, Any]] = []
    for q in unique_queries:
        qid = q["query_id"]
        # Which subset?
        subset = None
        for s, qids in subset_qids.items():
            if qid in qids:
                subset = s
                break
        rel = global_gold.get(qid, set())
        if not rel:
            continue
        b_r5 = recall_at_k(baseline_ranked.get(qid, []), rel, 5)
        f_r5 = recall_at_k(rrf_ranked.get(qid, []), rel, 5)
        m_r5 = recall_at_k(max_ranked.get(qid, []), rel, 5)
        delta_rrf = f_r5 - b_r5
        delta_max = m_r5 - b_r5
        row = {
            "qid": qid,
            "subset": subset,
            "text": q["text"],
            "variants": variants_by_qid.get(qid, []),
            "baseline_R@5": b_r5,
            "rrf_R@5": f_r5,
            "max_R@5": m_r5,
            "delta_rrf": delta_rrf,
            "delta_max": delta_max,
            "gold": sorted(rel),
            "baseline_top5": baseline_ranked.get(qid, [])[:5],
            "rrf_top5": rrf_ranked.get(qid, [])[:5],
        }
        if max(delta_rrf, delta_max) > 0.01:
            help_rows.append(row)
        if min(delta_rrf, delta_max) < -0.01:
            hurt_rows.append(row)

    # Sort
    help_rows.sort(key=lambda r: max(r["delta_rrf"], r["delta_max"]), reverse=True)
    hurt_rows.sort(key=lambda r: min(r["delta_rrf"], r["delta_max"]))

    # 7. Costs summary
    total_new_cost = rw_cost + variant_extract_cost  # doc_cost is amortized
    total_tokens = {
        "rewrite": rewriter.usage,
        "variant_extract": variant_extract_usage,
        "doc_extract": doc_usage,
    }
    cost_per_query_overhead = (rw_cost + variant_extract_cost) / max(
        len(unique_queries), 1
    )

    # 8. Emit JSON + Markdown
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float) and math.isnan(o):
            return None
        return o

    out_json = {
        "subsets": _clean(results),
        "sample_rewrites": sample_rewrites,
        "help_rows": _clean(help_rows[:20]),
        "hurt_rows": _clean(hurt_rows[:20]),
        "usage": total_tokens,
        "cost_usd": {
            "rewrite": rw_cost,
            "variant_extract": variant_extract_cost,
            "doc_extract": doc_cost,
            "total_new": total_new_cost,
            "per_query_overhead": cost_per_query_overhead,
        },
        "n_queries_total": len(unique_queries),
    }
    (RESULTS_DIR / "query_rewrite.json").write_text(json.dumps(out_json, indent=2))

    lines: list[str] = []
    lines.append("# F11 — Temporal query rewriting + fusion\n\n")
    lines.append(
        "Compares three retrieval modes on five subsets. For each query we "
        "ask gpt-5-mini for up to 5 temporal paraphrases (temperature=0), "
        "extract each variant with the same base extractor, and fuse the "
        "resulting ranked lists.\n\n"
    )

    lines.append("## Per-subset metrics\n\n")
    lines.append(
        "| subset | mode | n | R@5 | R@10 | MRR | NDCG@10 |\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
    )
    for name in ["base", "axis", "utterance", "era", "allen", "all"]:
        for mode in ["baseline", "rrf", "max"]:
            m = results[name][mode]
            lines.append(
                f"| {name} | {mode} | {m['n']} | {m['recall@5']:.3f} | "
                f"{m['recall@10']:.3f} | {m['mrr']:.3f} | {m['ndcg@10']:.3f} |\n"
            )
        lines.append("|  |  |  |  |  |  |  |\n")

    lines.append("\n## Lift over baseline (R@5)\n\n")
    lines.append("| subset | Δ RRF | Δ max |\n|---|---:|---:|\n")
    for name in ["base", "axis", "utterance", "era", "allen", "all"]:
        b = results[name]["baseline"]["recall@5"]
        r = results[name]["rrf"]["recall@5"]
        m = results[name]["max"]["recall@5"]
        lines.append(f"| {name} | {r - b:+.3f} | {m - b:+.3f} |\n")

    lines.append("\n## Cost\n\n")
    lines.append(
        f"- Rewriter (gpt-5-mini): in={rewriter.usage['input']}, out={rewriter.usage['output']} -> ${rw_cost:.4f}\n"
    )
    lines.append(
        f"- Variant extraction (gpt-5-mini, cached vs base): in={variant_extract_usage['input']}, out={variant_extract_usage['output']} -> ${variant_extract_cost:.4f}\n"
    )
    lines.append(
        f"- Doc extraction (amortised): in={doc_usage['input']}, out={doc_usage['output']} -> ${doc_cost:.4f}\n"
    )
    lines.append(
        f"- **New cost attributable to rewriting**: ${total_new_cost:.4f} across {len(unique_queries)} queries = ${cost_per_query_overhead * 1000:.2f} / 1000 queries\n\n"
    )

    lines.append("## Sample rewrites\n\n")
    for e in sample_rewrites[:12]:
        lines.append(f"- **{e['qid']}** `{e['text']}`\n")
        for v in e["variants"]:
            lines.append(f"  - `{v}`\n")

    lines.append("\n## Where rewriting helped most (top 10 by ΔR@5)\n\n")
    for r in help_rows[:10]:
        lines.append(
            f"- `{r['qid']}` [{r['subset']}] `{r['text']}` — Δrrf={r['delta_rrf']:+.2f}, Δmax={r['delta_max']:+.2f}\n"
        )
        for v in r["variants"][:5]:
            lines.append(f"  - variant: `{v}`\n")

    lines.append("\n## Where rewriting hurt (top 10 by ΔR@5)\n\n")
    for r in hurt_rows[:10]:
        lines.append(
            f"- `{r['qid']}` [{r['subset']}] `{r['text']}` — Δrrf={r['delta_rrf']:+.2f}, Δmax={r['delta_max']:+.2f}\n"
        )
        for v in r["variants"][:5]:
            lines.append(f"  - variant: `{v}`\n")

    lines.append("\n## Analysis\n\n")
    # Ship recommendation heuristics
    best_delta_rrf = {
        name: results[name]["rrf"]["recall@5"] - results[name]["baseline"]["recall@5"]
        for name in ["base", "axis", "utterance", "era", "allen"]
    }
    best_delta_max = {
        name: results[name]["max"]["recall@5"] - results[name]["baseline"]["recall@5"]
        for name in ["base", "axis", "utterance", "era", "allen"]
    }
    pos_rrf = [n for n, d in best_delta_rrf.items() if d > 0.005]
    neg_rrf = [n for n, d in best_delta_rrf.items() if d < -0.005]
    pos_max = [n for n, d in best_delta_max.items() if d > 0.005]
    neg_max = [n for n, d in best_delta_max.items() if d < -0.005]
    lines.append(
        f"- RRF positive subsets: {pos_rrf or 'none'}; negative: {neg_rrf or 'none'}.\n"
    )
    lines.append(
        f"- max-of positive subsets: {pos_max or 'none'}; negative: {neg_max or 'none'}.\n"
    )
    lines.append(f"- Wall time total: {time.time() - overall_start:.1f}s.\n")

    (RESULTS_DIR / "query_rewrite.md").write_text("".join(lines))

    # Console recap
    print("\n=== Summary ===")
    for name in ["base", "axis", "utterance", "era", "allen", "all"]:
        for mode in ["baseline", "rrf", "max"]:
            m = results[name][mode]
            print(
                f"  {name:<10} {mode:<9} n={m['n']:<3} "
                f"R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f} "
                f"MRR={m['mrr']:.3f} NDCG={m['ndcg@10']:.3f}"
            )
    print(
        f"\nNew cost: ${total_new_cost:.4f} across {len(unique_queries)} queries "
        f"(${cost_per_query_overhead * 1000:.2f} / 1000 queries)"
    )
    print(f"Wall: {time.time() - overall_start:.1f}s")
    print("Wrote results/query_rewrite.{md,json}")


if __name__ == "__main__":
    asyncio.run(main())
