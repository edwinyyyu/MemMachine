"""Fair-backfill eval of inverse-query variants vs v2f baseline.

Runs 3 variants (inverse_query, inverse_query_top3, inverse_query_v2f) plus
a dedicated meta_v2f baseline (for orthogonality comparison) on LoCoMo-30
and synthetic-19 at K=20 and K=50 using the fair-backfill methodology.

Also computes the orthogonality measure: fraction of gold turns found by
inverse-query-family variants that were NOT found by v2f at K=50.

Usage:
    uv run python inverse_query_eval.py
    uv run python inverse_query_eval.py --archs inverse_query,inverse_query_v2f
    uv run python inverse_query_eval.py --datasets locomo_30q
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from associative_recall import Segment, SegmentStore
from fair_backfill_eval import (
    BUDGETS,
    DATA_DIR,
    DATASETS,
    RESULTS_DIR,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)
from inverse_query import (
    ARCH_CLASSES as INV_ARCH_CLASSES,
    InverseQuery,
    InverseQueryTop3,
    InverseQueryV2f,
)
from antipara_cue_gen import MetaV2fDedicated

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# Restrict to LoCoMo-30 and synthetic-19 per study plan
EVAL_DATASETS = ("locomo_30q", "synthetic_19q")

ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    "inverse_query": InverseQuery,
    "inverse_query_top3": InverseQueryTop3,
    "inverse_query_v2f": InverseQueryV2f,
}


def evaluate_question(arch, question: dict) -> dict:
    """Run arch on a single question, produce fair-backfill metrics + metadata."""
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])
    category = question.get("category", "unknown")

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            arch_segments.append(seg)
            seen.add(seg.index)

    query_emb = arch.embed_text(q_text)
    max_K = max(BUDGETS)
    cosine_result = arch.store.search(
        query_emb, top_k=max_K, conversation_id=conv_id
    )
    cosine_segments = list(cosine_result.segments)

    row = {
        "conversation_id": conv_id,
        "category": category,
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "total_arch_retrieved": len(arch_segments),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "inverse_queries": result.metadata.get("inverse_queries", []),
        "v2f_cues": result.metadata.get("v2f_cues", []),
        "probe_outcomes": result.metadata.get("probe_outcomes", []),
        "v2f_outcomes": result.metadata.get("v2f_outcomes", []),
        "num_probes": result.metadata.get("num_probes"),
        "hop0_empty": result.metadata.get("hop0_empty", False),
    }

    # Compute per-K arch-at-K turn_ids (for orthogonality analysis)
    arch_ids_at_K: dict[int, set[int]] = {}
    for K in BUDGETS:
        b_rec, a_rec, a_segs_at_K = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)
        arch_ids_at_K[K] = {s.turn_id for s in a_segs_at_K}

    # Attach found-gold sets for orthogonality analysis
    row["gold_found_at_K"] = {
        str(K): sorted(arch_ids_at_K[K] & source_ids) for K in BUDGETS
    }

    return row


def run_one(
    arch_name: str,
    arch,
    dataset: str,
    questions: list[dict],
) -> tuple[list[dict], dict, dict]:
    print(f"\n{'=' * 70}")
    print(f"{arch_name} | {dataset} | {len(questions)} questions")
    print(f"{'=' * 70}")

    results = []
    for i, q in enumerate(questions):
        q_short = q["question"][:55]
        print(
            f"  [{i+1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(arch, q)
            results.append(row)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = summarize(results, arch_name, dataset)
    by_cat = summarize_by_category(results)

    print(f"\n--- {arch_name} on {dataset} ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: baseline={summary[f'baseline_r@{K}']:.3f} "
            f"arch={summary[f'arch_r@{K}']:.3f} "
            f"delta={summary[f'delta_r@{K}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    print(
        f"  avg total_retrieved={summary['avg_total_retrieved']:.0f} "
        f"llm={summary['avg_llm_calls']:.1f} "
        f"embed={summary['avg_embed_calls']:.1f}"
    )
    for cat, c in by_cat.items():
        print(
            f"    {cat:28s} (n={c['n']}): "
            f"r@20 d={c['delta_r@20']:+.3f} r@50 d={c['delta_r@50']:+.3f} "
            f"W/T/L@50={c['W/T/L_r@50']}"
        )

    return results, summary, by_cat


def compute_orthogonality(
    inv_rows: list[dict],
    v2f_rows: list[dict],
    K: int = 50,
) -> dict:
    """For each question: what fraction of gold turns found by `inv` variant
    at K were NOT found by v2f at K?

    Uses `gold_found_at_K` populated during evaluate_question.
    """
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    total_inv_gold = 0
    novel_inv_gold = 0
    per_q: list[dict] = []
    for r in inv_rows:
        key = (r["conversation_id"], r["question_index"])
        inv_gold = set(r["gold_found_at_K"].get(str(K), []))
        v2f_gold = v2f_by_key.get(key, set())
        novel = inv_gold - v2f_gold
        total_inv_gold += len(inv_gold)
        novel_inv_gold += len(novel)
        per_q.append(
            {
                "conversation_id": r["conversation_id"],
                "question_index": r["question_index"],
                "inv_gold_count": len(inv_gold),
                "novel_vs_v2f": len(novel),
                "novel_turn_ids": sorted(novel),
            }
        )

    frac_novel = novel_inv_gold / total_inv_gold if total_inv_gold else 0.0
    return {
        "total_inv_gold": total_inv_gold,
        "novel_vs_v2f": novel_inv_gold,
        "fraction_novel": round(frac_novel, 4),
        "per_question": per_q,
    }


def qualitative_examples(
    inv_rows: list[dict],
    v2f_rows: list[dict],
    K: int = 50,
    max_examples: int = 3,
) -> list[dict]:
    """Find examples where inv_query found gold that v2f missed. Attributes
    the novel gold back to a generated inverse question via probe_outcomes.
    """
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    examples: list[dict] = []
    for r in inv_rows:
        if len(examples) >= max_examples:
            break
        key = (r["conversation_id"], r["question_index"])
        inv_gold = set(r["gold_found_at_K"].get(str(K), []))
        v2f_gold = v2f_by_key.get(key, set())
        novel = inv_gold - v2f_gold
        if not novel:
            continue

        for probe in r.get("probe_outcomes", []):
            hit = novel & set(probe.get("retrieved_turn_ids", []))
            if hit:
                examples.append(
                    {
                        "question": r["question"],
                        "category": r["category"],
                        "source_turn_id": probe["source_turn_id"],
                        "generated_inverse_question": probe["question"],
                        "novel_gold_turn_id": sorted(hit)[0],
                    }
                )
                break
    return examples


def top_categories_delta(by_cat: dict, K: int = 50) -> tuple[list, list]:
    """Return top 2 gaining and top 2 losing categories by delta_r@K."""
    rows = []
    for cat, c in by_cat.items():
        rows.append((cat, c[f"delta_r@{K}"], c[f"W/T/L_r@{K}"], c["n"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    gaining = [
        {"category": cat, "delta": d, "W/T/L": wtl, "n": n}
        for cat, d, wtl, n in rows[:2]
        if d > 0
    ]
    losing = [
        {"category": cat, "delta": d, "W/T/L": wtl, "n": n}
        for cat, d, wtl, n in rows[-2:]
        if d < 0
    ]
    return gaining, losing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archs",
        default=",".join(ARCH_CLASSES.keys()),
        help="Comma-separated arch names",
    )
    p.add_argument(
        "--datasets",
        default=",".join(EVAL_DATASETS),
        help="Comma-separated dataset names",
    )
    args = p.parse_args()

    arch_names = [a.strip() for a in args.archs.split(",") if a.strip()]
    ds_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for a in arch_names:
        if a not in ARCH_CLASSES:
            raise SystemExit(f"Unknown arch: {a}")
    for d in ds_names:
        if d not in DATASETS:
            raise SystemExit(f"Unknown dataset: {d}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # results[arch][ds] = {summary, by_cat, results}
    all_results: dict[str, dict] = defaultdict(dict)

    for ds_name in ds_names:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        for arch_name in arch_names:
            cls = ARCH_CLASSES[arch_name]
            arch = cls(store)
            results, summary, by_cat = run_one(
                arch_name, arch, ds_name, questions
            )
            all_results[arch_name][ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
            }

    # Orthogonality analysis vs meta_v2f
    orthogonality: dict[str, dict] = {}
    if "meta_v2f" in all_results:
        for arch_name in arch_names:
            if arch_name == "meta_v2f":
                continue
            orthogonality[arch_name] = {}
            for ds_name in ds_names:
                if ds_name not in all_results[arch_name]:
                    continue
                if ds_name not in all_results["meta_v2f"]:
                    continue
                inv_rows = all_results[arch_name][ds_name]["results"]
                v2f_rows = all_results["meta_v2f"][ds_name]["results"]
                orth = compute_orthogonality(inv_rows, v2f_rows, K=50)
                orthogonality[arch_name][ds_name] = {
                    "total_inv_gold": orth["total_inv_gold"],
                    "novel_vs_v2f": orth["novel_vs_v2f"],
                    "fraction_novel": orth["fraction_novel"],
                }

    # Qualitative examples from inverse_query on locomo_30q
    examples: list[dict] = []
    if (
        "inverse_query" in all_results
        and "meta_v2f" in all_results
        and "locomo_30q" in all_results["inverse_query"]
        and "locomo_30q" in all_results["meta_v2f"]
    ):
        examples = qualitative_examples(
            all_results["inverse_query"]["locomo_30q"]["results"],
            all_results["meta_v2f"]["locomo_30q"]["results"],
            K=50,
            max_examples=3,
        )

    # Top categories gaining/losing for inverse_query on locomo_30q
    top_gaining: list = []
    top_losing: list = []
    if (
        "inverse_query" in all_results
        and "locomo_30q" in all_results["inverse_query"]
    ):
        top_gaining, top_losing = top_categories_delta(
            all_results["inverse_query"]["locomo_30q"]["category_breakdown"], K=50
        )

    # Raw JSON output (excluding `results` for top-level aggregated view,
    # full results stored per-arch per-dataset)
    raw: dict = {
        "archs": arch_names,
        "datasets": ds_names,
        "summaries": {
            a: {
                d: {
                    "summary": all_results[a][d]["summary"],
                    "category_breakdown": all_results[a][d]["category_breakdown"],
                }
                for d in all_results[a]
            }
            for a in all_results
        },
        "orthogonality_vs_v2f_at_K50": orthogonality,
        "qualitative_examples": examples,
        "top_gaining_categories": top_gaining,
        "top_losing_categories": top_losing,
    }

    raw_path = RESULTS_DIR / "inverse_query_study.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset full results saved separately
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"invq_{a}_{d}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "arch": a,
                        "dataset": d,
                        "summary": all_results[a][d]["summary"],
                        "category_breakdown": all_results[a][d][
                            "category_breakdown"
                        ],
                        "results": all_results[a][d]["results"],
                    },
                    f,
                    indent=2,
                    default=str,
                )

    # Markdown report
    md_lines: list[str] = []
    md_lines.append("# Inverse Query Generation Study\n")
    md_lines.append(
        "Motivation: LoCoMo gold sits off-center from queries "
        "(+0.14 cosine gap, 36% kNN-adjacency). Inverse query generation "
        "starts from retrieved content and works backward — the generated "
        "question is anchored in actual corpus text, not user phrasing. "
        "Orthogonal to v2f which imagines chat content forward from the query.\n"
    )

    # Recall table
    md_lines.append("## Fair-backfill recall\n")
    md_lines.append(
        "| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |"
    )
    md_lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            md_lines.append(
                f"| {a} | {d} | "
                f"{s['baseline_r@20']:.3f} | {s['arch_r@20']:.3f} | "
                f"{s['delta_r@20']:+.3f} | "
                f"{s['baseline_r@50']:.3f} | {s['arch_r@50']:.3f} | "
                f"{s['delta_r@50']:+.3f} | "
                f"{s['avg_llm_calls']:.1f} |"
            )

    # Orthogonality
    if orthogonality:
        md_lines.append("\n## Orthogonality vs v2f (K=50)\n")
        md_lines.append(
            "Fraction of gold turns found by the variant that v2f did NOT find.\n"
        )
        md_lines.append("| Arch | Dataset | inv_gold | novel_vs_v2f | frac_novel |")
        md_lines.append("|---|---|---:|---:|---:|")
        for a in orthogonality:
            for d in orthogonality[a]:
                o = orthogonality[a][d]
                md_lines.append(
                    f"| {a} | {d} | {o['total_inv_gold']} | "
                    f"{o['novel_vs_v2f']} | {o['fraction_novel']:.3f} |"
                )

    # Qualitative examples
    if examples:
        md_lines.append("\n## Qualitative examples (LoCoMo, inverse_query)\n")
        md_lines.append(
            "Each row: question → generated inverse question → novel gold "
            "turn_id (not retrieved by v2f)."
        )
        for ex in examples:
            md_lines.append(
                f"\n- **Q:** {ex['question']}\n"
                f"  - Source turn (hop0): turn_id={ex['source_turn_id']}\n"
                f"  - Generated inverse question: _{ex['generated_inverse_question']}_\n"
                f"  - Novel gold turn_id found: {ex['novel_gold_turn_id']}"
            )

    # Top gaining / losing categories
    if top_gaining or top_losing:
        md_lines.append(
            "\n## Top categories by Δr@50 (inverse_query on LoCoMo-30)\n"
        )
        md_lines.append("Gaining:")
        for g in top_gaining:
            md_lines.append(
                f"  - {g['category']} (n={g['n']}): Δ={g['delta']:+.3f} "
                f"W/T/L={g['W/T/L']}"
            )
        md_lines.append("Losing:")
        for l in top_losing:
            md_lines.append(
                f"  - {l['category']} (n={l['n']}): Δ={l['delta']:+.3f} "
                f"W/T/L={l['W/T/L']}"
            )

    # Verdict section (left for manual fill but start with a data-driven draft)
    md_lines.append("\n## Verdict\n")
    # Heuristic: compare deltas vs meta_v2f on locomo_30q@50
    verdict = "(see numbers above)"
    if (
        "meta_v2f" in all_results
        and "locomo_30q" in all_results["meta_v2f"]
        and "inverse_query" in all_results
        and "locomo_30q" in all_results["inverse_query"]
    ):
        v2f50 = all_results["meta_v2f"]["locomo_30q"]["summary"]["arch_r@50"]
        inv50 = all_results["inverse_query"]["locomo_30q"]["summary"]["arch_r@50"]
        inv_v2f50 = None
        if (
            "inverse_query_v2f" in all_results
            and "locomo_30q" in all_results["inverse_query_v2f"]
        ):
            inv_v2f50 = all_results["inverse_query_v2f"]["locomo_30q"][
                "summary"
            ]["arch_r@50"]

        if inv50 > v2f50 + 0.005:
            verdict = (
                f"**SHIP**: inverse_query beats v2f on LoCoMo-30 @K=50 "
                f"({inv50:.3f} vs {v2f50:.3f})."
            )
        elif inv_v2f50 is not None and inv_v2f50 > v2f50 + 0.005:
            verdict = (
                f"**SUPPLEMENT-ONLY**: inverse_query alone ties/loses, but "
                f"inverse_query_v2f beats v2f @K=50 "
                f"({inv_v2f50:.3f} vs {v2f50:.3f})."
            )
        else:
            verdict = (
                f"**ABANDON**: neither variant beats v2f on LoCoMo-30 @K=50 "
                f"(v2f={v2f50:.3f}, inv={inv50:.3f}"
                + (
                    f", inv+v2f={inv_v2f50:.3f}"
                    if inv_v2f50 is not None
                    else ""
                )
                + ")."
            )
    md_lines.append(verdict + "\n")

    md_path = RESULTS_DIR / "inverse_query_study.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved: {md_path}")

    # Final table
    print("\n" + "=" * 100)
    print("INVERSE QUERY SUMMARY")
    print("=" * 100)
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:22s} {d:14s} "
                f"b@20={s['baseline_r@20']:.3f} a@20={s['arch_r@20']:.3f} "
                f"d@20={s['delta_r@20']:+.3f}  "
                f"b@50={s['baseline_r@50']:.3f} a@50={s['arch_r@50']:.3f} "
                f"d@50={s['delta_r@50']:+.3f}  "
                f"llm={s['avg_llm_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
