"""Fair-backfill eval for clause decomposition variants.

Runs:
  - meta_v2f (baseline)
  - clause_cosine_n2, clause_cosine_n3
  - clause_v2f_n2
  - clause_plus_v2f
on LoCoMo-30 and synthetic-19 at K=20 and K=50.

Reports:
  - clause distribution (n=1 / n=2 / n=3+)
  - recall table vs baselines
  - per-category deltas
  - multi-clause slice (queries with >=2 clauses)
  - verdict + cost

Usage:
    uv run python clause_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from associative_recall import Segment
from fair_backfill_eval import (
    BUDGETS,
    RESULTS_DIR,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)
from antipara_cue_gen import MetaV2fDedicated
from clause_decomposition import (
    ARCH_CLASSES as CLAUSE_ARCH_CLASSES,
    split_query_into_clauses,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")

# Composite registry: baseline + clause variants
ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    **CLAUSE_ARCH_CLASSES,
}


def evaluate_question(arch, question: dict) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

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

    md = result.metadata or {}
    row = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "total_arch_retrieved": len(arch_segments),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        # clause-specific
        "clauses": md.get("clauses", []),
        "n_clauses": md.get("n_clauses", 1),
        "split": md.get("split", False),
    }

    arch_ids_at_K: dict[int, set[int]] = {}
    for K in BUDGETS:
        b_rec, a_rec, a_segs_at_K = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)
        arch_ids_at_K[K] = {s.turn_id for s in a_segs_at_K}

    row["gold_found_at_K"] = {
        str(K): sorted(arch_ids_at_K[K] & source_ids) for K in BUDGETS
    }
    return row


def run_one(arch_name, arch, dataset, questions):
    print(f"\n{'=' * 70}")
    print(f"{arch_name} | {dataset} | {len(questions)} questions")
    print(f"{'=' * 70}")
    results: list[dict] = []
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
        f"  avg llm/q={summary['avg_llm_calls']:.2f} "
        f"embed/q={summary['avg_embed_calls']:.2f}"
    )
    return results, summary, by_cat


def clause_distribution(questions: list[dict], max_clauses: int = 3) -> dict:
    """Report clause-split distribution across a question set."""
    counts = Counter()
    per_query: list[dict] = []
    for q in questions:
        c2 = split_query_into_clauses(q["question"], max_clauses=2)
        c3 = split_query_into_clauses(q["question"], max_clauses=3)
        n2 = len(c2)
        n3 = len(c3)
        counts[n3] += 1
        per_query.append(
            {
                "question": q["question"],
                "category": q.get("category"),
                "n_clauses_n2": n2,
                "n_clauses_n3": n3,
                "clauses_n3": c3,
            }
        )
    dist = {
        "n_questions": len(questions),
        "n=1": counts.get(1, 0),
        "n=2": counts.get(2, 0),
        "n>=3": sum(v for k, v in counts.items() if k >= 3),
        "per_query": per_query,
    }
    return dist


def multiclause_slice_metrics(results: list[dict]) -> dict:
    """Return avg deltas and W/T/L only for queries with n_clauses >= 2."""
    subset = [r for r in results if (r.get("n_clauses") or 1) >= 2]
    n = len(subset)
    out = {"n_multiclause": n}
    if n == 0:
        return out
    for K in BUDGETS:
        b = [r["fair_backfill"][f"baseline_r@{K}"] for r in subset]
        a = [r["fair_backfill"][f"arch_r@{K}"] for r in subset]
        wins = sum(1 for bb, aa in zip(b, a) if aa > bb + 0.001)
        losses = sum(1 for bb, aa in zip(b, a) if bb > aa + 0.001)
        ties = n - wins - losses
        out[f"baseline_r@{K}"] = round(sum(b) / n, 4)
        out[f"arch_r@{K}"] = round(sum(a) / n, 4)
        out[f"delta_r@{K}"] = round((sum(a) - sum(b)) / n, 4)
        out[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = defaultdict(dict)
    clause_dist_by_ds: dict[str, dict] = {}

    for ds_name in EVAL_DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        # Clause distribution (no model calls)
        clause_dist_by_ds[ds_name] = clause_distribution(questions)
        cd = clause_dist_by_ds[ds_name]
        print(
            f"  Clause distribution (n3 cap): n=1 {cd['n=1']}  "
            f"n=2 {cd['n=2']}  n>=3 {cd['n>=3']}"
        )

        for arch_name, cls in ARCH_CLASSES.items():
            arch = cls(store)
            results, summary, by_cat = run_one(
                arch_name, arch, ds_name, questions
            )
            all_results[arch_name][ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
                "multiclause_slice": multiclause_slice_metrics(results),
            }

    # --- Save raw JSON ---
    raw = {
        "archs": list(ARCH_CLASSES.keys()),
        "datasets": list(EVAL_DATASETS),
        "summaries": {
            a: {
                d: {
                    "summary": all_results[a][d]["summary"],
                    "category_breakdown": all_results[a][d][
                        "category_breakdown"
                    ],
                    "multiclause_slice": all_results[a][d][
                        "multiclause_slice"
                    ],
                }
                for d in all_results[a]
            }
            for a in all_results
        },
        "clause_distribution": {
            d: {
                "n_questions": clause_dist_by_ds[d]["n_questions"],
                "n=1": clause_dist_by_ds[d]["n=1"],
                "n=2": clause_dist_by_ds[d]["n=2"],
                "n>=3": clause_dist_by_ds[d]["n>=3"],
            }
            for d in EVAL_DATASETS
        },
    }
    raw_path = RESULTS_DIR / "clause_decomposition.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch-per-dataset details
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"clause_decomposition_{a}_{d}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "arch": a,
                        "dataset": d,
                        "summary": all_results[a][d]["summary"],
                        "category_breakdown": all_results[a][d][
                            "category_breakdown"
                        ],
                        "multiclause_slice": all_results[a][d][
                            "multiclause_slice"
                        ],
                        "results": all_results[a][d]["results"],
                    },
                    f,
                    indent=2,
                    default=str,
                )

    # --- Markdown report ---
    md: list[str] = []
    md.append("# Query Clause Decomposition\n")
    md.append(
        "Mechanical split of multi-part queries on sentence boundaries, "
        "semicolons, and safe conjunctions. Per-clause retrieval, unioned. "
        "No LLM decomposition (context_tree_v2 failed because an LLM couldn't "
        "decompose without losing intent — this is cheap and preserves "
        "literal clause tokens).\n"
    )

    md.append("## Clause distribution (n3 cap)\n")
    md.append("| Dataset | n queries | n=1 | n=2 | n>=3 |")
    md.append("|---|---:|---:|---:|---:|")
    for d, cd in clause_dist_by_ds.items():
        md.append(
            f"| {d} | {cd['n_questions']} | {cd['n=1']} | "
            f"{cd['n=2']} | {cd['n>=3']} |"
        )
    md.append("")

    md.append("## Fair-backfill recall\n")
    md.append(
        "| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | "
        "Δ@50 | llm/q |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in ARCH_CLASSES:
        for d in EVAL_DATASETS:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            md.append(
                f"| {a} | {d} | {s['baseline_r@20']:.3f} | "
                f"{s['arch_r@20']:.3f} | {s['delta_r@20']:+.3f} | "
                f"{s['baseline_r@50']:.3f} | {s['arch_r@50']:.3f} | "
                f"{s['delta_r@50']:+.3f} | {s['avg_llm_calls']:.2f} |"
            )
    md.append("")

    md.append("## Multi-clause slice (queries with >=2 clauses)\n")
    md.append(
        "| Arch | Dataset | n | base@20 | arch@20 | Δ@20 | "
        "base@50 | arch@50 | Δ@50 | W/T/L@50 |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in ARCH_CLASSES:
        for d in EVAL_DATASETS:
            if d not in all_results.get(a, {}):
                continue
            m = all_results[a][d]["multiclause_slice"]
            n = m.get("n_multiclause", 0)
            if n == 0:
                md.append(
                    f"| {a} | {d} | 0 | - | - | - | - | - | - | - |"
                )
                continue
            md.append(
                f"| {a} | {d} | {n} | "
                f"{m.get('baseline_r@20', 0):.3f} | "
                f"{m.get('arch_r@20', 0):.3f} | "
                f"{m.get('delta_r@20', 0):+.3f} | "
                f"{m.get('baseline_r@50', 0):.3f} | "
                f"{m.get('arch_r@50', 0):.3f} | "
                f"{m.get('delta_r@50', 0):+.3f} | "
                f"{m.get('W/T/L_r@50', '-')} |"
            )
    md.append("")

    md.append("## Per-category (clause_plus_v2f)\n")
    pick = "clause_plus_v2f"
    if pick in all_results:
        for d in EVAL_DATASETS:
            if d not in all_results[pick]:
                continue
            by_cat = all_results[pick][d]["category_breakdown"]
            md.append(f"### {d}\n")
            md.append("| category | n | Δ@20 | Δ@50 | W/T/L@50 |")
            md.append("|---|---:|---:|---:|---:|")
            for cat, c in by_cat.items():
                md.append(
                    f"| {cat} | {c['n']} | {c['delta_r@20']:+.3f} | "
                    f"{c['delta_r@50']:+.3f} | {c['W/T/L_r@50']} |"
                )
            md.append("")

    md.append("## Sample splits\n")
    for d in EVAL_DATASETS:
        cd = clause_dist_by_ds[d]
        md.append(f"### {d}\n")
        sample = [
            e
            for e in cd["per_query"]
            if e["n_clauses_n3"] >= 2
        ][:6]
        for e in sample:
            md.append(
                f"- **{e.get('category')}** `{e['question'][:120]}`  \n"
                f"  -> {e['clauses_n3']}"
            )
        md.append("")

    # Per-category for clause_v2f_n2 (the more LLM-expensive but better variant)
    md.append("## Per-category (clause_v2f_n2)\n")
    pick2 = "clause_v2f_n2"
    if pick2 in all_results:
        for d in EVAL_DATASETS:
            if d not in all_results[pick2]:
                continue
            by_cat = all_results[pick2][d]["category_breakdown"]
            md.append(f"### {d}\n")
            md.append("| category | n | Δ@20 | Δ@50 | W/T/L@50 |")
            md.append("|---|---:|---:|---:|---:|")
            for cat, c in by_cat.items():
                md.append(
                    f"| {cat} | {c['n']} | {c['delta_r@20']:+.3f} | "
                    f"{c['delta_r@50']:+.3f} | {c['W/T/L_r@50']} |"
                )
            md.append("")

    # Verdict
    md.append("## Verdict\n")
    verdict_lines: list[str] = []
    for variant in ("clause_plus_v2f", "clause_v2f_n2", "clause_cosine_n2"):
        if "meta_v2f" not in all_results or variant not in all_results:
            continue
        verdict_lines.append(f"### {variant}")
        for d in EVAL_DATASETS:
            if d not in all_results["meta_v2f"] or d not in all_results[variant]:
                continue
            v2f = all_results["meta_v2f"][d]["summary"]
            cpv = all_results[variant][d]["summary"]
            verdict_lines.append(
                f"- {d} K=50: meta_v2f={v2f['arch_r@50']:.3f}, "
                f"{variant}={cpv['arch_r@50']:.3f}, "
                f"Δ={cpv['arch_r@50'] - v2f['arch_r@50']:+.3f} "
                f"(llm/q {v2f['avg_llm_calls']:.2f} -> "
                f"{cpv['avg_llm_calls']:.2f})"
            )
            ms = all_results[variant][d]["multiclause_slice"]
            if ms.get("n_multiclause", 0) > 0:
                verdict_lines.append(
                    f"  - multi-clause subset (n={ms['n_multiclause']}) K=50: "
                    f"Δ={ms.get('delta_r@50', 0):+.3f} "
                    f"W/T/L={ms.get('W/T/L_r@50', '-')}"
                )
    md.extend(verdict_lines)
    md.append("")

    md_path = RESULTS_DIR / "clause_decomposition.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final console table
    print("\n" + "=" * 100)
    print("CLAUSE DECOMPOSITION SUMMARY")
    print("=" * 100)
    for a in ARCH_CLASSES:
        for d in EVAL_DATASETS:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:22s} {d:14s} "
                f"a@20={s['arch_r@20']:.3f} d@20={s['delta_r@20']:+.3f}  "
                f"a@50={s['arch_r@50']:.3f} d@50={s['delta_r@50']:+.3f}  "
                f"llm/q={s['avg_llm_calls']:.2f}"
            )


if __name__ == "__main__":
    main()
