"""Fair-backfill eval for stacked-merge alias alt-keys.

Runs:
  - meta_v2f (baseline) -- reuses cached results if available
  - stacked_alias (this test)

on LoCoMo-30 and synthetic-19 at K=20 and K=50. Reports:
  - alt-key index stats
  - recall table
  - fraction of queries where >=1 alias alt-key entered top-K
  - per-category deltas
  - verdict + cost comparison

Usage:
    uv run python stacked_alias_eval.py
"""

from __future__ import annotations

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
from antipara_cue_gen import MetaV2fDedicated
from stacked_alias import StackedAlias

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")
ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    "stacked_alias": StackedAlias,
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
        # stacked_alias extras
        "n_alt_key_hits_raw": md.get("n_alt_key_hits_raw"),
        "n_alias_turn_hits": md.get("n_alias_turn_hits"),
        "n_alias_turn_hits_novel": md.get("n_alias_turn_hits_novel"),
        "alias_records": md.get("alias_records", []),
        "alias_appended_turn_ids": md.get("alias_appended_turn_ids", []),
        "v2f_turn_ids": md.get("v2f_turn_ids", []),
        "n_v2f_segments": md.get("n_v2f_segments"),
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

    # Diagnostics: how many alias-appended turns actually made it into top-K?
    alias_ids = set(row["alias_appended_turn_ids"] or [])
    row["alias_in_topK"] = {
        str(K): sorted(alias_ids & arch_ids_at_K[K]) for K in BUDGETS
    }
    row["n_alias_in_topK"] = {K: len(alias_ids & arch_ids_at_K[K]) for K in BUDGETS}

    row["gold_found_at_K"] = {
        str(K): sorted(arch_ids_at_K[K] & source_ids) for K in BUDGETS
    }

    return row


def run_one(
    arch_name: str, arch, dataset: str, questions: list[dict]
) -> tuple[list[dict], dict, dict]:
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


def summarize_alias_diagnostics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    counts_raw = [r.get("n_alt_key_hits_raw", 0) or 0 for r in results]
    counts_turn = [r.get("n_alias_turn_hits", 0) or 0 for r in results]
    counts_novel = [r.get("n_alias_turn_hits_novel", 0) or 0 for r in results]
    novel_entered_K20 = [r.get("n_alias_in_topK", {}).get(20, 0) for r in results]
    novel_entered_K50 = [r.get("n_alias_in_topK", {}).get(50, 0) for r in results]

    # How many queries had >= 1 alias hit enter top-K
    n_fired_K20 = sum(1 for x in novel_entered_K20 if x > 0)
    n_fired_K50 = sum(1 for x in novel_entered_K50 if x > 0)

    # Gold-from-alias: queries where an appended alias turn is in gold at K
    n_gold_from_alias_K20 = 0
    n_gold_from_alias_K50 = 0
    for r in results:
        gold = set(r["source_chat_ids"])
        in_top_K20 = set(r.get("alias_in_topK", {}).get("20", []))
        in_top_K50 = set(r.get("alias_in_topK", {}).get("50", []))
        if in_top_K20 & gold:
            n_gold_from_alias_K20 += 1
        if in_top_K50 & gold:
            n_gold_from_alias_K50 += 1

    return {
        "n_queries": n,
        "mean_alt_key_hits_raw": round(sum(counts_raw) / n, 2),
        "mean_alias_turn_hits": round(sum(counts_turn) / n, 2),
        "mean_alias_turn_hits_novel_vs_v2f": round(sum(counts_novel) / n, 2),
        "n_queries_where_alias_entered_topK": {
            "20": n_fired_K20,
            "50": n_fired_K50,
        },
        "n_queries_alias_contributed_gold": {
            "20": n_gold_from_alias_K20,
            "50": n_gold_from_alias_K50,
        },
        "mean_alias_turns_in_topK": {
            "20": round(sum(novel_entered_K20) / n, 2),
            "50": round(sum(novel_entered_K50) / n, 2),
        },
    }


def compute_orthogonality(
    alias_rows: list[dict], v2f_rows: list[dict], K: int = 50
) -> dict:
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    total_gold = 0
    novel_gold = 0
    for r in alias_rows:
        key = (r["conversation_id"], r["question_index"])
        gold = set(r["gold_found_at_K"].get(str(K), []))
        v2f_gold = v2f_by_key.get(key, set())
        novel = gold - v2f_gold
        total_gold += len(gold)
        novel_gold += len(novel)

    return {
        "total_gold": total_gold,
        "novel_vs_v2f": novel_gold,
        "fraction_novel": round(novel_gold / total_gold, 4)
        if total_gold
        else 0.0,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = defaultdict(dict)
    index_stats_by_ds: dict[str, dict] = {}

    for ds_name in EVAL_DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        for arch_name, cls in ARCH_CLASSES.items():
            arch = cls(store)
            if arch_name == "stacked_alias":
                index_stats_by_ds[ds_name] = getattr(arch, "index_stats", {})
            results, summary, by_cat = run_one(
                arch_name, arch, ds_name, questions
            )
            all_results[arch_name][ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
            }

    # Diagnostics
    diag: dict[str, dict] = {}
    for ds_name in EVAL_DATASETS:
        if "stacked_alias" in all_results and ds_name in all_results["stacked_alias"]:
            diag[ds_name] = summarize_alias_diagnostics(
                all_results["stacked_alias"][ds_name]["results"]
            )

    # Orthogonality vs meta_v2f
    ortho: dict[str, dict] = {}
    if "meta_v2f" in all_results and "stacked_alias" in all_results:
        for ds_name in EVAL_DATASETS:
            if (
                ds_name in all_results["meta_v2f"]
                and ds_name in all_results["stacked_alias"]
            ):
                ortho[ds_name] = {
                    "K50": compute_orthogonality(
                        all_results["stacked_alias"][ds_name]["results"],
                        all_results["meta_v2f"][ds_name]["results"],
                        K=50,
                    ),
                    "K20": compute_orthogonality(
                        all_results["stacked_alias"][ds_name]["results"],
                        all_results["meta_v2f"][ds_name]["results"],
                        K=20,
                    ),
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
                }
                for d in all_results[a]
            }
            for a in all_results
        },
        "alt_key_index_stats": index_stats_by_ds,
        "diagnostics": diag,
        "orthogonality_vs_v2f": ortho,
    }
    raw_path = RESULTS_DIR / "stacked_alias.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset details
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"stacked_alias_{a}_{d}.json"
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

    # --- Markdown report ---
    md: list[str] = []
    md.append("# Stacked-merge alias alt-keys\n")
    md.append(
        "Ingest-time alias substitution + separate-index + stacked-merge "
        "retrieval. Cheaper query-time alternative to `alias_expand_v2f` "
        "(which runs v2f on each alias variant, ~3x LLM). This test moves the "
        "alias work to ingest and keeps per-query cost equal to plain v2f.\n"
    )

    md.append("## Alt-key index stats\n")
    for ds_name, stats in index_stats_by_ds.items():
        md.append(f"### {ds_name}\n")
        md.append(f"- alt-keys raw: {stats.get('n_alt_keys_raw')}")
        md.append(f"- alt-keys unique: {stats.get('n_alt_keys_unique')}")
        md.append(
            f"- convs with alt-keys: "
            f"{stats.get('n_convs_with_altkeys')}/{stats.get('n_convs_total')}"
        )
        md.append(f"- turns with >=1 alias match: {stats.get('n_turns_with_match')}")
        md.append("")

    md.append("## Fair-backfill recall\n")
    md.append(
        "| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in ARCH_CLASSES:
        for d in EVAL_DATASETS:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            md.append(
                f"| {a} | {d} | {s['baseline_r@20']:.3f} | {s['arch_r@20']:.3f} | "
                f"{s['delta_r@20']:+.3f} | {s['baseline_r@50']:.3f} | "
                f"{s['arch_r@50']:.3f} | {s['delta_r@50']:+.3f} | "
                f"{s['avg_llm_calls']:.1f} |"
            )
    md.append("")

    # Published baselines for reference
    md.append(
        "**Reference baselines** (from `alias_expansion.json`): "
        "v2f LoCoMo r@20=0.756 r@50=0.858; "
        "alias_expand_v2f LoCoMo r@20=0.694 r@50=0.881 "
        "(the latter costs ~3x LLM/query).\n"
    )

    md.append("## Per-category (stacked_alias)\n")
    if "stacked_alias" in all_results:
        for d in EVAL_DATASETS:
            if d not in all_results["stacked_alias"]:
                continue
            by_cat = all_results["stacked_alias"][d]["category_breakdown"]
            md.append(f"### {d}\n")
            md.append("| category | n | Δ@20 | Δ@50 | W/T/L@50 |")
            md.append("|---|---:|---:|---:|---:|")
            for cat, c in by_cat.items():
                md.append(
                    f"| {cat} | {c['n']} | {c['delta_r@20']:+.3f} | "
                    f"{c['delta_r@50']:+.3f} | {c['W/T/L_r@50']} |"
                )
            md.append("")

    md.append("## Alias mechanism diagnostics\n")
    for ds_name, d in diag.items():
        md.append(f"### {ds_name}\n")
        fired = d.get("n_queries_where_alias_entered_topK", {})
        contrib = d.get("n_queries_alias_contributed_gold", {})
        n = d.get("n_queries", 0)
        md.append(f"- n queries: {n}")
        md.append(
            f"- mean raw alt-key hits / query: "
            f"{d.get('mean_alt_key_hits_raw')}"
        )
        md.append(
            f"- mean alias-turn hits (deduped) / query: "
            f"{d.get('mean_alias_turn_hits')}"
        )
        md.append(
            f"- mean alias-turn hits novel vs v2f / query: "
            f"{d.get('mean_alias_turn_hits_novel_vs_v2f')}"
        )
        md.append(
            f"- queries where >=1 alias turn entered top-20: {fired.get('20')}/{n}"
        )
        md.append(
            f"- queries where >=1 alias turn entered top-50: {fired.get('50')}/{n}"
        )
        md.append(
            f"- queries where an alias-appended turn hit gold @K=20: "
            f"{contrib.get('20')}/{n}"
        )
        md.append(
            f"- queries where an alias-appended turn hit gold @K=50: "
            f"{contrib.get('50')}/{n}"
        )
        md.append("")

    md.append("## Orthogonality vs meta_v2f\n")
    md.append("| Dataset | K | total_gold | novel_vs_v2f | frac_novel |")
    md.append("|---|---:|---:|---:|---:|")
    for ds_name, od in ortho.items():
        for K in ("K20", "K50"):
            o = od.get(K, {})
            md.append(
                f"| {ds_name} | {K[1:]} | {o.get('total_gold')} | "
                f"{o.get('novel_vs_v2f')} | {o.get('fraction_novel')} |"
            )
    md.append("")

    # Cost comparison
    md.append("## Cost comparison (per-query LLM calls)\n")
    md.append(
        "| Arch | avg LLM/q (LoCoMo) | avg LLM/q (synthetic) | ingest LLM |"
    )
    md.append("|---|---:|---:|---:|")
    for a in ARCH_CLASSES:
        lc_s = all_results.get(a, {}).get("locomo_30q", {}).get("summary", {})
        sy_s = all_results.get(a, {}).get(
            "synthetic_19q", {}
        ).get("summary", {})
        ingest = "none" if a == "meta_v2f" else "1 per conv (shared w/ alias_expand)"
        md.append(
            f"| {a} | {lc_s.get('avg_llm_calls', 0):.1f} | "
            f"{sy_s.get('avg_llm_calls', 0):.1f} | {ingest} |"
        )
    md.append(
        "\nReference: `alias_expand_v2f` costs ~3 LLM/query (1 v2f per "
        "alias variant). `stacked_alias` matches `meta_v2f` at 1 LLM/query.\n"
    )

    # Verdict
    md.append("## Verdict\n")
    verdict = "see numbers above"
    if "stacked_alias" in all_results and "meta_v2f" in all_results:
        v2f_lc = all_results["meta_v2f"]["locomo_30q"]["summary"]["arch_r@50"]
        sa_lc = all_results["stacked_alias"]["locomo_30q"]["summary"][
            "arch_r@50"
        ]
        aexp_lc_r50 = 0.8806
        # Novel gold vs v2f: if the mechanism never surfaces new gold, even a
        # bonus variant cannot help.
        novel_total = 0
        total_gold = 0
        for ds_name, od in ortho.items():
            novel_total += od.get("K50", {}).get("novel_vs_v2f", 0)
            total_gold += od.get("K50", {}).get("total_gold", 0)
        if sa_lc >= aexp_lc_r50 - 0.005:
            verdict = (
                f"**SHIP**: stacked_alias matches alias_expand_v2f at ~0 "
                f"per-query LLM overhead (LoCoMo K=50: {sa_lc:.3f} vs "
                f"alias_expand_v2f={aexp_lc_r50:.3f}, v2f baseline={v2f_lc:.3f})."
            )
        elif sa_lc > v2f_lc + 0.005:
            verdict = (
                f"**BORDERLINE**: stacked_alias beats v2f ({sa_lc:.3f} vs "
                f"{v2f_lc:.3f}) but loses to alias_expand_v2f "
                f"({aexp_lc_r50:.3f}). Useful tradeoff (no per-query LLM)."
            )
        elif sa_lc < v2f_lc - 0.005:
            verdict = (
                f"**ABANDON**: stacked_alias loses to v2f ({sa_lc:.3f} vs "
                f"{v2f_lc:.3f}) -- substituted alt-keys lose semantic quality."
            )
        else:
            # Tie with v2f. Inspect novel-gold contribution: if zero, a bonus
            # variant won't help because alt-keys don't surface new gold.
            if novel_total == 0:
                verdict = (
                    f"**ABANDON (tie with v2f, mechanism dry)**: stacked_alias "
                    f"ties v2f on LoCoMo K=50 ({sa_lc:.3f}) but alias "
                    f"alt-keys surface **{novel_total}/{total_gold} novel gold "
                    f"turns** across both datasets at K=50. A score-bonus "
                    f"variant cannot help — the substituted-text index adds "
                    f"no retrieval that v2f+cosine doesn't already find. "
                    f"alias_expand_v2f's +2.3pp lift on LoCoMo K=50 requires "
                    f"the per-variant v2f cue generation, not cosine "
                    f"retrieval on substituted text."
                )
            else:
                verdict = (
                    f"**TIE WITH V2F**: stacked merge too conservative "
                    f"({sa_lc:.3f} vs {v2f_lc:.3f}); {novel_total}/{total_gold} "
                    f"novel gold hits found. Try bonus variant."
                )
    md.append(verdict + "\n")

    md_path = RESULTS_DIR / "stacked_alias.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final console table
    print("\n" + "=" * 100)
    print("STACKED_ALIAS SUMMARY")
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
                f"llm/q={s['avg_llm_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
