"""Fair-backfill eval of alias-tracker variants vs v2f baseline.

Runs alias_trk_context (primary) and alias_trk_drift vs meta_v2f baseline
(and optionally compares against alias_expand_v2f_full results already on
disk) on LoCoMo-30 at K=20 and K=50.

Usage:
    uv run python aliastrk_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from associative_recall import Segment
from fair_backfill_eval import (
    BUDGETS,
    DATASETS,
    RESULTS_DIR,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)
from alias_tracker import (
    AliasTrkContext,
    AliasTrkDrift,
)
from antipara_cue_gen import MetaV2fDedicated

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q",)

ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    "alias_trk_context": AliasTrkContext,
    "alias_trk_drift": AliasTrkDrift,
}


def evaluate_question(arch, question: dict) -> dict:
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

    md = result.metadata
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
        "alias_note": md.get("alias_note", ""),
        "matches": md.get("matches", []),
        "num_matches": md.get("num_matches", 0),
        "early_bias": md.get("early_bias", False),
        "cues": md.get("cues", []),
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
        f"  avg retrieved={summary['avg_total_retrieved']:.0f} "
        f"llm={summary['avg_llm_calls']:.1f} "
        f"embed={summary['avg_embed_calls']:.1f}"
    )

    return results, summary, by_cat


def load_reference_expand_v2f(dataset: str) -> dict | None:
    """Load prior alias_expand_v2f full results for side-by-side comparison."""
    p = RESULTS_DIR / f"alias_alias_expand_v2f_{dataset}.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


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

    # Load reference alias_expand_v2f numbers from prior run.
    reference_rows: dict[str, dict] = {}
    for d in ds_names:
        ref = load_reference_expand_v2f(d)
        if ref is not None:
            reference_rows[d] = ref

    # Tracker stats (alias group / alias count / coverage of turn indices)
    from alias_tracker import AliasTracker  # lazy import

    trk = AliasTracker()
    # Collect stats across all conversations referenced in LoCoMo-30 questions.
    locomo_convs: set[str] = set()
    if "locomo_30q" in all_results.get(arch_names[0], {}):
        for r in all_results[arch_names[0]]["locomo_30q"]["results"]:
            locomo_convs.add(r["conversation_id"])
    stats_per_conv: dict[str, dict] = {}
    for cid in sorted(locomo_convs):
        groups = trk.groups(cid)
        if not groups:
            continue
        alias_counts = [len(g["aliases"]) for g in groups]
        with_turns = sum(1 for g in groups if g["first_seen_turn"] >= 0)
        stats_per_conv[cid] = {
            "num_groups": len(groups),
            "total_aliases": sum(alias_counts),
            "mean_aliases_per_group": round(
                sum(alias_counts) / len(alias_counts), 2
            ),
            "groups_with_turn_coverage": with_turns,
            "turn_range": list(trk.conv_turn_range(cid)),
            "sample_group": groups[0] if groups else None,
        }

    # Cost comparison
    cost_comparison: dict[str, dict] = {}
    for d in ds_names:
        row = {}
        for a in arch_names:
            if d in all_results.get(a, {}):
                row[a] = all_results[a][d]["summary"]["avg_llm_calls"]
        if d in reference_rows:
            row["alias_expand_v2f (reference)"] = (
                reference_rows[d]["summary"]["avg_llm_calls"]
            )
        cost_comparison[d] = row

    # Sample alias injections (2 examples from LoCoMo)
    samples: list[dict] = []
    primary_arch = "alias_trk_context"
    if primary_arch in all_results and "locomo_30q" in all_results[primary_arch]:
        rows = all_results[primary_arch]["locomo_30q"]["results"]
        # Prefer rows with matches AND at least one gold found at K=50.
        for r in rows:
            if len(samples) >= 2:
                break
            if r.get("num_matches", 0) == 0:
                continue
            gold50 = r.get("gold_found_at_K", {}).get("50", [])
            if not gold50:
                continue
            samples.append(
                {
                    "conversation_id": r["conversation_id"],
                    "question": r["question"],
                    "alias_note": r.get("alias_note", ""),
                    "matches": r.get("matches", []),
                    "cues": r.get("cues", []),
                    "gold_found_at_50": gold50,
                }
            )
        # Fallback: any row with matches.
        if len(samples) < 2:
            for r in rows:
                if len(samples) >= 2:
                    break
                if r.get("num_matches", 0) == 0:
                    continue
                if any(
                    s["conversation_id"] == r["conversation_id"]
                    and s["question"] == r["question"]
                    for s in samples
                ):
                    continue
                samples.append(
                    {
                        "conversation_id": r["conversation_id"],
                        "question": r["question"],
                        "alias_note": r.get("alias_note", ""),
                        "matches": r.get("matches", []),
                        "cues": r.get("cues", []),
                        "gold_found_at_50": r.get(
                            "gold_found_at_K", {}
                        ).get("50", []),
                    }
                )

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
        "reference_alias_expand_v2f": {
            d: {
                "summary": reference_rows[d]["summary"],
            }
            for d in reference_rows
        },
        "cost_comparison_llm_per_query": cost_comparison,
        "tracker_stats_per_conv": stats_per_conv,
        "samples": samples,
    }

    raw_path = RESULTS_DIR / "alias_tracker.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset full results
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"aliastrk_{a}_{d}.json"
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

    # Markdown
    md: list[str] = []
    md.append("# Query-time alias expansion via dynamic alias tracker\n")
    md.append(
        "Anaphora-resolution at INGEST fails (the first alias introduction is "
        "itself a cue). Instead, inject alias-sibling context at QUERY time "
        "into a SINGLE v2f call so the LLM picks whichever form matches the "
        "imagined conversation register. Tracker records first/last-seen turn "
        "indices for drift-aware variants.\n"
    )

    # Tracker stats
    if stats_per_conv:
        md.append("## Alias tracker stats (LoCoMo-30 conversations)\n")
        total_groups = sum(v["num_groups"] for v in stats_per_conv.values())
        total_aliases = sum(v["total_aliases"] for v in stats_per_conv.values())
        total_with_cov = sum(
            v["groups_with_turn_coverage"] for v in stats_per_conv.values()
        )
        md.append(
            f"- {len(stats_per_conv)} conversations, {total_groups} alias "
            f"groups total, {total_aliases} aliases total"
        )
        md.append(
            f"- {total_with_cov}/{total_groups} groups have turn-index "
            f"coverage (first/last seen turn populated)"
        )
        for cid in sorted(stats_per_conv):
            v = stats_per_conv[cid]
            md.append(
                f"  - {cid}: {v['num_groups']} groups, {v['total_aliases']} "
                f"aliases, turns {v['turn_range'][0]}-{v['turn_range'][1]}"
            )

    # Recall table
    md.append("\n## Fair-backfill recall (LoCoMo-30)\n")
    md.append(
        "| Arch | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            md.append(
                f"| {a} | "
                f"{s['baseline_r@20']:.3f} | {s['arch_r@20']:.3f} | "
                f"{s['delta_r@20']:+.3f} | "
                f"{s['baseline_r@50']:.3f} | {s['arch_r@50']:.3f} | "
                f"{s['delta_r@50']:+.3f} | "
                f"{s['avg_llm_calls']:.1f} |"
            )
    # Reference
    for d in reference_rows:
        s = reference_rows[d]["summary"]
        md.append(
            f"| alias_expand_v2f (ref, 3x cost) | "
            f"{s['baseline_r@20']:.3f} | {s['arch_r@20']:.3f} | "
            f"{s['delta_r@20']:+.3f} | "
            f"{s['baseline_r@50']:.3f} | {s['arch_r@50']:.3f} | "
            f"{s['delta_r@50']:+.3f} | "
            f"{s['avg_llm_calls']:.1f} |"
        )

    # Cost comparison
    md.append("\n## Cost comparison (avg LLM calls/query)\n")
    for d, row in cost_comparison.items():
        md.append(f"**{d}**")
        for k, v in row.items():
            md.append(f"  - {k}: {v:.2f}")

    # Sample injections
    if samples:
        md.append("\n## Sample alias injections\n")
        for i, ex in enumerate(samples, 1):
            md.append(f"\n**Sample {i}** ({ex['conversation_id']})")
            md.append(f"- **Q:** {ex['question']}")
            if ex.get("alias_note"):
                md.append(f"- **Note:** {ex['alias_note']}")
            if ex.get("matches"):
                for m in ex["matches"]:
                    md.append(
                        f"- Match `{m['matched_form']}` (canonical: "
                        f"`{m['canonical']}`); siblings used: "
                        f"{m['filtered_siblings']}; first/last seen at turns "
                        f"{m['first_seen_turn']}/{m['last_seen_turn']}"
                    )
            if ex.get("cues"):
                md.append("- Generated cues:")
                for c in ex["cues"]:
                    md.append(f"  - `{c}`")
            if ex.get("gold_found_at_50"):
                md.append(f"- Gold turns found @50: {ex['gold_found_at_50']}")

    # Verdict
    md.append("\n## Verdict\n")
    verdict = "(see numbers above)"
    if (
        "meta_v2f" in all_results
        and "locomo_30q" in all_results["meta_v2f"]
    ):
        v2f50 = all_results["meta_v2f"]["locomo_30q"]["summary"]["arch_r@50"]
        ref50 = None
        if "locomo_30q" in reference_rows:
            ref50 = reference_rows["locomo_30q"]["summary"]["arch_r@50"]

        variant_scores: list[tuple[str, float, float]] = []
        for a in arch_names:
            if a == "meta_v2f":
                continue
            if "locomo_30q" not in all_results.get(a, {}):
                continue
            s = all_results[a]["locomo_30q"]["summary"]
            variant_scores.append(
                (a, s["arch_r@50"], s["avg_llm_calls"])
            )
        best = max(variant_scores, key=lambda t: t[1]) if variant_scores else None

        if best is None:
            verdict = "No tracker variants evaluated."
        else:
            detail = ", ".join(
                f"{n}={r:.3f}@{c:.1f}llm" for n, r, c in variant_scores
            )
            if ref50 is not None:
                if best[1] >= ref50 - 0.005:
                    verdict = (
                        f"**SHIP** (replace expensive): {best[0]} matches/beats "
                        f"alias_expand_v2f ({best[1]:.3f} vs {ref50:.3f}) at "
                        f"{best[2]:.1f} LLM/q (vs ~2.9 for expand_v2f).\n"
                        f"Details: {detail}."
                    )
                elif best[1] > v2f50 + 0.005:
                    verdict = (
                        f"**SHIP** (narrow): {best[0]} beats vanilla v2f "
                        f"({best[1]:.3f} vs {v2f50:.3f}) but is below "
                        f"alias_expand_v2f ({ref50:.3f}). Cheaper middle "
                        f"ground.\n"
                        f"Details: {detail}."
                    )
                else:
                    verdict = (
                        f"**ABANDON**: no tracker variant beats vanilla v2f "
                        f"(v2f={v2f50:.3f}). v2f-per-variant mechanism (the "
                        f"expensive version) was the actual value, not just "
                        f"seeing the aliases.\n"
                        f"Details: {detail}; alias_expand_v2f ref={ref50:.3f}."
                    )
            else:
                if best[1] > v2f50 + 0.005:
                    verdict = (
                        f"**SHIP**: {best[0]} beats v2f on LoCoMo-30 @K=50 "
                        f"({best[1]:.3f} vs {v2f50:.3f}).\n"
                        f"Details: {detail}."
                    )
                else:
                    verdict = (
                        f"**ABANDON**: no tracker variant beats v2f on "
                        f"LoCoMo-30 @K=50 (v2f={v2f50:.3f}).\n"
                        f"Details: {detail}."
                    )
    md.append(verdict + "\n")

    md_path = RESULTS_DIR / "alias_tracker.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final table
    print("\n" + "=" * 100)
    print("ALIAS TRACKER SUMMARY")
    print("=" * 100)
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:25s} {d:14s} "
                f"b@20={s['baseline_r@20']:.3f} a@20={s['arch_r@20']:.3f} "
                f"d@20={s['delta_r@20']:+.3f}  "
                f"b@50={s['baseline_r@50']:.3f} a@50={s['arch_r@50']:.3f} "
                f"d@50={s['delta_r@50']:+.3f}  "
                f"llm={s['avg_llm_calls']:.1f}"
            )
    for d in reference_rows:
        s = reference_rows[d]["summary"]
        print(
            f"{'alias_expand_v2f (ref)':25s} {d:14s} "
            f"b@20={s['baseline_r@20']:.3f} a@20={s['arch_r@20']:.3f} "
            f"d@20={s['delta_r@20']:+.3f}  "
            f"b@50={s['baseline_r@50']:.3f} a@50={s['arch_r@50']:.3f} "
            f"d@50={s['delta_r@50']:+.3f}  "
            f"llm={s['avg_llm_calls']:.1f}"
        )


if __name__ == "__main__":
    main()
