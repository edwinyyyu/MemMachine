"""Fair-backfill eval of alias-expansion variants vs v2f baseline.

Runs alias_expand_cosine, alias_expand_v2f_cheap, alias_expand_v2f plus a
dedicated v2f baseline on LoCoMo-30 and synthetic-19 at K=20 and K=50
using the fair-backfill methodology.

Also reports, per dataset:
  - Per-category deltas vs v2f (to check evolving_terminology-like gains).
  - Orthogonality: gold turns found by alias variant that v2f missed at K=50.
  - Summary of alias groups per conversation (count + 2 samples).
  - Sample (question, expanded variants, gold-retrieval) trios.

Usage:
    uv run python alias_expand_eval.py
    uv run python alias_expand_eval.py --archs alias_expand_cosine
    uv run python alias_expand_eval.py --datasets locomo_30q
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
from alias_expansion import (
    ARCH_CLASSES as ALIAS_ARCH_CLASSES,
    AliasExpandCosine,
    AliasExpandV2fCheap,
    AliasExpandV2fFull,
)
from antipara_cue_gen import MetaV2fDedicated

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")

ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    "alias_expand_cosine": AliasExpandCosine,
    "alias_expand_v2f_cheap": AliasExpandV2fCheap,
    "alias_expand_v2f": AliasExpandV2fFull,
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
        "num_variants": md.get("num_variants"),
        "query_variants": md.get("query_variants", []),
        "match_records": md.get("match_records", []),
        "variant_outcomes": md.get("variant_outcomes", []),
        "sibling_probe_outcomes": md.get("sibling_probe_outcomes", []),
        "v2f_cues": md.get("v2f_cues", []),
        "per_variant_v2f": md.get("per_variant_v2f", []),
        "num_probes": md.get("num_probes"),
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
    for cat, c in by_cat.items():
        print(
            f"    {cat:30s} (n={c['n']}): "
            f"r@20 d={c['delta_r@20']:+.3f} r@50 d={c['delta_r@50']:+.3f} "
            f"W/T/L@50={c['W/T/L_r@50']}"
        )

    return results, summary, by_cat


def compute_orthogonality(
    alias_rows: list[dict],
    v2f_rows: list[dict],
    K: int = 50,
) -> dict:
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    total_gold = 0
    novel_gold = 0
    per_q: list[dict] = []
    for r in alias_rows:
        key = (r["conversation_id"], r["question_index"])
        gold = set(r["gold_found_at_K"].get(str(K), []))
        v2f_gold = v2f_by_key.get(key, set())
        novel = gold - v2f_gold
        total_gold += len(gold)
        novel_gold += len(novel)
        per_q.append(
            {
                "conversation_id": r["conversation_id"],
                "question_index": r["question_index"],
                "gold_count": len(gold),
                "novel_vs_v2f": len(novel),
                "novel_turn_ids": sorted(novel),
            }
        )

    frac_novel = novel_gold / total_gold if total_gold else 0.0
    return {
        "total_gold": total_gold,
        "novel_vs_v2f": novel_gold,
        "fraction_novel": round(frac_novel, 4),
        "per_question": per_q,
    }


def qualitative_trios(
    alias_rows: list[dict],
    v2f_rows: list[dict],
    K: int = 50,
    max_examples: int = 5,
) -> list[dict]:
    """Find examples where alias-expansion variants retrieved gold turns."""
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    examples: list[dict] = []
    # First pass: prefer questions with matches AND novel hits
    for pass_mode in ("novel", "any_with_match", "any"):
        if len(examples) >= max_examples:
            break
        for r in alias_rows:
            if len(examples) >= max_examples:
                break
            matches = r.get("match_records", [])
            variants = r.get("query_variants", [])
            if len(variants) < 2 and pass_mode != "any":
                continue
            key = (r["conversation_id"], r["question_index"])
            gold = set(r["gold_found_at_K"].get(str(K), []))
            v2f_gold = v2f_by_key.get(key, set())
            novel = gold - v2f_gold

            if pass_mode == "novel" and not novel:
                continue
            if pass_mode == "any_with_match" and not gold:
                continue
            if pass_mode == "any" and not gold:
                continue
            # Avoid duplicates
            if any(
                ex["conversation_id"] == r["conversation_id"]
                and ex["question_index"] == r["question_index"]
                for ex in examples
            ):
                continue

            # Attribute gold to specific variant outcome
            attributions: list[dict] = []
            target = novel if pass_mode == "novel" else gold
            for vo in r.get("variant_outcomes", []):
                hit_ids = target & set(vo.get("retrieved_turn_ids", []))
                if hit_ids:
                    attributions.append(
                        {
                            "variant": vo["variant"],
                            "gold_turns_retrieved": sorted(hit_ids),
                        }
                    )
            for sp in r.get("sibling_probe_outcomes", []):
                hit_ids = target & set(sp.get("retrieved_turn_ids", []))
                if hit_ids:
                    attributions.append(
                        {
                            "sibling_probe": sp["sibling"],
                            "gold_turns_retrieved": sorted(hit_ids),
                        }
                    )

            if not attributions:
                continue

            examples.append(
                {
                    "conversation_id": r["conversation_id"],
                    "question_index": r["question_index"],
                    "question": r["question"],
                    "category": r["category"],
                    "match_records": matches,
                    "query_variants": variants,
                    "gold_found": sorted(gold),
                    "novel_vs_v2f": sorted(novel),
                    "attributions": attributions,
                    "pass": pass_mode,
                }
            )

    return examples


def top_categories_delta(by_cat: dict, K: int = 50) -> tuple[list, list]:
    rows = []
    for cat, c in by_cat.items():
        rows.append((cat, c[f"delta_r@{K}"], c[f"W/T/L_r@{K}"], c["n"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    gaining = [
        {"category": cat, "delta": d, "W/T/L": wtl, "n": n}
        for cat, d, wtl, n in rows
        if d > 0.001
    ][:3]
    losing = [
        {"category": cat, "delta": d, "W/T/L": wtl, "n": n}
        for cat, d, wtl, n in rows[::-1]
        if d < -0.001
    ][:3]
    return gaining, losing


def summarize_alias_groups(alias_rows: list[dict]) -> dict:
    """Aggregate: per-conversation alias group count + sample groups."""
    by_conv: dict[str, list[list[str]]] = {}
    qs_with_matches = 0
    variant_counts: list[int] = []
    for r in alias_rows:
        cid = r["conversation_id"]
        # Try each row for groups; they're part of metadata but we stored
        # match_records + query_variants only. Fetch groups via extractor file.
        variant_counts.append(r.get("num_variants") or 0)
        if r.get("match_records"):
            qs_with_matches += 1
    # Read persistent alias groups file for ground truth
    path = RESULTS_DIR / "conversation_alias_groups.json"
    groups_per_conv: dict[str, list[list[str]]] = {}
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            groups_per_conv = data.get("groups", {}) or {}
        except (json.JSONDecodeError, OSError):
            groups_per_conv = {}

    summary: dict = {}
    for cid, gs in groups_per_conv.items():
        summary[cid] = {
            "num_groups": len(gs),
            "sample_groups": gs[:3],
            "total_aliases": sum(len(g) for g in gs),
        }
    summary["_queries_with_alias_match"] = qs_with_matches
    summary["_total_queries"] = len(alias_rows)
    if variant_counts:
        summary["_mean_variants_per_query"] = round(
            sum(variant_counts) / len(variant_counts), 2
        )
        summary["_max_variants"] = max(variant_counts)
    return summary


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

    # Orthogonality vs meta_v2f
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
                rows = all_results[arch_name][ds_name]["results"]
                v2f_rows = all_results["meta_v2f"][ds_name]["results"]
                orth = compute_orthogonality(rows, v2f_rows, K=50)
                orthogonality[arch_name][ds_name] = {
                    "total_gold": orth["total_gold"],
                    "novel_vs_v2f": orth["novel_vs_v2f"],
                    "fraction_novel": orth["fraction_novel"],
                }

    # Qualitative trios from cheap variant on LoCoMo (primary showcase)
    trios: list[dict] = []
    primary_arch = "alias_expand_v2f_cheap"
    if primary_arch not in arch_names and arch_names:
        primary_arch = next(
            (a for a in arch_names if a.startswith("alias_")), arch_names[0]
        )
    if (
        primary_arch in all_results
        and "meta_v2f" in all_results
        and "locomo_30q" in all_results[primary_arch]
        and "locomo_30q" in all_results["meta_v2f"]
    ):
        trios = qualitative_trios(
            all_results[primary_arch]["locomo_30q"]["results"],
            all_results["meta_v2f"]["locomo_30q"]["results"],
            K=50,
            max_examples=5,
        )

    # Top categories for primary variant
    top_gaining: list = []
    top_losing: list = []
    if primary_arch in all_results and "locomo_30q" in all_results[primary_arch]:
        top_gaining, top_losing = top_categories_delta(
            all_results[primary_arch]["locomo_30q"]["category_breakdown"], K=50
        )

    # Alias group summary (LoCoMo)
    alias_group_summary: dict = {}
    if primary_arch in all_results and "locomo_30q" in all_results[primary_arch]:
        alias_group_summary = summarize_alias_groups(
            all_results[primary_arch]["locomo_30q"]["results"]
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
        "orthogonality_vs_v2f_at_K50": orthogonality,
        "qualitative_trios": trios,
        "top_gaining_categories": top_gaining,
        "top_losing_categories": top_losing,
        "alias_group_summary": alias_group_summary,
        "primary_arch": primary_arch,
    }

    raw_path = RESULTS_DIR / "alias_expansion.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset full results
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"alias_{a}_{d}.json"
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
    md: list[str] = []
    md.append("# Query-time alias expansion via ingest-extracted alias groups\n")
    md.append(
        "Motivation: evolving-terminology / multi-referent entities resist v2f "
        "cue gen because aliases are corpus-specific. Ingest extracts alias "
        "groups once per conversation; at query time, if the query mentions "
        "an alias, expanded queries replacing that alias with each sibling "
        "drive extra cosine retrievals (and optionally extra v2f runs).\n"
    )

    if alias_group_summary:
        md.append("## Alias groups extracted (LoCoMo)\n")
        convs = [k for k in alias_group_summary if not k.startswith("_")]
        for cid in sorted(convs):
            v = alias_group_summary[cid]
            md.append(
                f"- {cid}: {v['num_groups']} groups, "
                f"{v['total_aliases']} aliases total"
            )
            for g in v.get("sample_groups", []):
                md.append(f"  - {g}")
        qm = alias_group_summary.get("_queries_with_alias_match")
        qt = alias_group_summary.get("_total_queries")
        mv = alias_group_summary.get("_mean_variants_per_query")
        mxv = alias_group_summary.get("_max_variants")
        if qt:
            md.append(
                f"\n{qm}/{qt} LoCoMo queries matched an extracted alias "
                f"(mean variants/query = {mv}, max = {mxv})."
            )

    md.append("\n## Fair-backfill recall\n")
    md.append(
        "| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            md.append(
                f"| {a} | {d} | "
                f"{s['baseline_r@20']:.3f} | {s['arch_r@20']:.3f} | "
                f"{s['delta_r@20']:+.3f} | "
                f"{s['baseline_r@50']:.3f} | {s['arch_r@50']:.3f} | "
                f"{s['delta_r@50']:+.3f} | "
                f"{s['avg_llm_calls']:.1f} |"
            )

    if orthogonality:
        md.append("\n## Orthogonality vs v2f (K=50)\n")
        md.append(
            "Fraction of gold turns found by the variant that v2f did NOT find.\n"
        )
        md.append("| Arch | Dataset | gold_found | novel_vs_v2f | frac_novel |")
        md.append("|---|---|---:|---:|---:|")
        for a in orthogonality:
            for d in orthogonality[a]:
                o = orthogonality[a][d]
                md.append(
                    f"| {a} | {d} | {o['total_gold']} | "
                    f"{o['novel_vs_v2f']} | {o['fraction_novel']:.3f} |"
                )

    if trios:
        md.append(f"\n## Qualitative trios ({primary_arch}, LoCoMo, K=50)\n")
        md.append(
            "Each row: matched alias → expanded variants → which variant "
            "surfaced gold turns."
        )
        for ex in trios:
            novel_tag = " **(novel vs v2f)**" if ex.get("novel_vs_v2f") else ""
            md.append(f"\n- **Q:** {ex['question']}{novel_tag}")
            if ex.get("match_records"):
                for m in ex["match_records"]:
                    md.append(
                        f"  - Matched alias `{m['matched_in_query']}` → "
                        f"siblings: {m['siblings']}"
                    )
            if ex.get("query_variants"):
                md.append(f"  - Variants ({len(ex['query_variants'])}):")
                for v in ex["query_variants"]:
                    md.append(f"    - `{v}`")
            for attr in ex["attributions"][:4]:
                if "variant" in attr:
                    md.append(
                        f"  - Variant `{attr['variant']}` retrieved gold "
                        f"turn(s) {attr['gold_turns_retrieved']}"
                    )
                else:
                    md.append(
                        f"  - Sibling probe `{attr['sibling_probe']}` "
                        f"retrieved gold turn(s) {attr['gold_turns_retrieved']}"
                    )

    if top_gaining or top_losing:
        md.append(
            f"\n## Top categories by Δr@50 ({primary_arch}, LoCoMo-30)\n"
        )
        md.append("Gaining:")
        if not top_gaining:
            md.append("  - (none with Δ > 0.001)")
        for g in top_gaining:
            md.append(
                f"  - {g['category']} (n={g['n']}): Δ={g['delta']:+.3f} "
                f"W/T/L={g['W/T/L']}"
            )
        md.append("Losing:")
        if not top_losing:
            md.append("  - (none with Δ < -0.001)")
        for l in top_losing:
            md.append(
                f"  - {l['category']} (n={l['n']}): Δ={l['delta']:+.3f} "
                f"W/T/L={l['W/T/L']}"
            )

    # Verdict
    md.append("\n## Verdict\n")
    verdict = "(see numbers above)"
    if (
        "meta_v2f" in all_results
        and "locomo_30q" in all_results["meta_v2f"]
        and primary_arch in all_results
        and "locomo_30q" in all_results[primary_arch]
    ):
        v2f50 = all_results["meta_v2f"]["locomo_30q"]["summary"]["arch_r@50"]
        variant_scores: list[tuple[str, float]] = []
        for a in arch_names:
            if a == "meta_v2f":
                continue
            if "locomo_30q" not in all_results.get(a, {}):
                continue
            variant_scores.append(
                (
                    a,
                    all_results[a]["locomo_30q"]["summary"]["arch_r@50"],
                )
            )
        best = max(variant_scores, key=lambda t: t[1]) if variant_scores else None
        if best and best[1] > v2f50 + 0.005:
            verdict = (
                f"**SHIP**: {best[0]} beats v2f on LoCoMo-30 @K=50 "
                f"({best[1]:.3f} vs {v2f50:.3f})."
            )
        else:
            details = ", ".join(f"{n}={s:.3f}" for n, s in variant_scores)
            verdict = (
                f"**ABANDON**: no alias variant beats v2f on LoCoMo-30 @K=50 "
                f"(v2f={v2f50:.3f}; {details})."
            )
    md.append(verdict + "\n")

    md_path = RESULTS_DIR / "alias_expansion.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final table
    print("\n" + "=" * 100)
    print("ALIAS EXPANSION SUMMARY")
    print("=" * 100)
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:26s} {d:14s} "
                f"b@20={s['baseline_r@20']:.3f} a@20={s['arch_r@20']:.3f} "
                f"d@20={s['delta_r@20']:+.3f}  "
                f"b@50={s['baseline_r@50']:.3f} a@50={s['arch_r@50']:.3f} "
                f"d@50={s['delta_r@50']:+.3f}  "
                f"llm={s['avg_llm_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
