"""Fair-backfill eval for context-enriched embeddings (stacked merge).

Runs:
  - meta_v2f (baseline) -- via dedicated caches to avoid concurrent-agent
    cross-contamination
  - contextemb_w1_stacked
  - contextemb_w2_stacked
  - contextemb_prev_stacked
  - contextemb_w1_bonus

on LoCoMo-30 and synthetic-19 at K=20 and K=50. Reports:
  - index stats per variant
  - recall table
  - per-category deltas
  - mechanism firing: fraction of queries where >=1 context hit entered
    the final top-K
  - orthogonality vs meta_v2f gold
  - verdict + cost

Usage:
    uv run python contextemb_eval.py
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
    RESULTS_DIR,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)
from antipara_cue_gen import MetaV2fDedicated
from context_embedding import (
    ContextEmbW1Stacked,
    ContextEmbW2Stacked,
    ContextEmbPrevStacked,
    ContextEmbW1Bonus,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")
ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    "contextemb_w1_stacked": ContextEmbW1Stacked,
    "contextemb_w2_stacked": ContextEmbW2Stacked,
    "contextemb_prev_stacked": ContextEmbPrevStacked,
    "contextemb_w1_bonus": ContextEmbW1Bonus,
}
CONTEXTEMB_VARIANTS = (
    "contextemb_w1_stacked",
    "contextemb_w2_stacked",
    "contextemb_prev_stacked",
    "contextemb_w1_bonus",
)


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
        # contextemb extras
        "variant": md.get("variant"),
        "score_bonus": md.get("score_bonus"),
        "n_ctx_hits_raw": md.get("n_ctx_hits_raw"),
        "n_ctx_turn_hits": md.get("n_ctx_turn_hits"),
        "n_ctx_turn_hits_novel": md.get("n_ctx_turn_hits_novel"),
        "ctx_records": md.get("ctx_records", []),
        "ctx_appended_turn_ids": md.get("ctx_appended_turn_ids", []),
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

    # Diagnostics: how many ctx-appended turns made it into final top-K?
    ctx_ids = set(row["ctx_appended_turn_ids"] or [])
    row["ctx_in_topK"] = {
        str(K): sorted(ctx_ids & arch_ids_at_K[K]) for K in BUDGETS
    }
    row["n_ctx_in_topK"] = {K: len(ctx_ids & arch_ids_at_K[K]) for K in BUDGETS}

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


def summarize_ctx_diagnostics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    counts_raw = [r.get("n_ctx_hits_raw", 0) or 0 for r in results]
    counts_turn = [r.get("n_ctx_turn_hits", 0) or 0 for r in results]
    counts_novel = [r.get("n_ctx_turn_hits_novel", 0) or 0 for r in results]
    novel_entered_K20 = [r.get("n_ctx_in_topK", {}).get(20, 0) for r in results]
    novel_entered_K50 = [r.get("n_ctx_in_topK", {}).get(50, 0) for r in results]

    n_fired_K20 = sum(1 for x in novel_entered_K20 if x > 0)
    n_fired_K50 = sum(1 for x in novel_entered_K50 if x > 0)

    n_gold_from_ctx_K20 = 0
    n_gold_from_ctx_K50 = 0
    for r in results:
        gold = set(r["source_chat_ids"])
        in_top_K20 = set(r.get("ctx_in_topK", {}).get("20", []))
        in_top_K50 = set(r.get("ctx_in_topK", {}).get("50", []))
        if in_top_K20 & gold:
            n_gold_from_ctx_K20 += 1
        if in_top_K50 & gold:
            n_gold_from_ctx_K50 += 1

    return {
        "n_queries": n,
        "mean_ctx_hits_raw": round(sum(counts_raw) / n, 2),
        "mean_ctx_turn_hits": round(sum(counts_turn) / n, 2),
        "mean_ctx_turn_hits_novel_vs_v2f": round(sum(counts_novel) / n, 2),
        "n_queries_where_ctx_entered_topK": {
            "20": n_fired_K20,
            "50": n_fired_K50,
        },
        "n_queries_ctx_contributed_gold": {
            "20": n_gold_from_ctx_K20,
            "50": n_gold_from_ctx_K50,
        },
        "mean_ctx_turns_in_topK": {
            "20": round(sum(novel_entered_K20) / n, 2),
            "50": round(sum(novel_entered_K50) / n, 2),
        },
    }


def compute_orthogonality(
    arch_rows: list[dict], v2f_rows: list[dict], K: int = 50
) -> dict:
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    total_gold = 0
    novel_gold = 0
    for r in arch_rows:
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
    index_stats_by_variant_ds: dict[str, dict[str, dict]] = defaultdict(dict)

    for ds_name in EVAL_DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        for arch_name, cls in ARCH_CLASSES.items():
            arch = cls(store)
            if arch_name in CONTEXTEMB_VARIANTS:
                variant = getattr(arch, "variant", arch_name)
                index_stats_by_variant_ds[variant][ds_name] = getattr(
                    arch, "index_stats", {}
                )
            results, summary, by_cat = run_one(
                arch_name, arch, ds_name, questions
            )
            all_results[arch_name][ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
            }

    # Diagnostics per contextemb variant per dataset
    diag: dict[str, dict[str, dict]] = defaultdict(dict)
    for arch_name in CONTEXTEMB_VARIANTS:
        for ds_name in EVAL_DATASETS:
            if ds_name in all_results.get(arch_name, {}):
                diag[arch_name][ds_name] = summarize_ctx_diagnostics(
                    all_results[arch_name][ds_name]["results"]
                )

    # Orthogonality per contextemb variant vs meta_v2f
    ortho: dict[str, dict[str, dict]] = defaultdict(dict)
    if "meta_v2f" in all_results:
        for arch_name in CONTEXTEMB_VARIANTS:
            for ds_name in EVAL_DATASETS:
                if (
                    ds_name in all_results.get(arch_name, {})
                    and ds_name in all_results["meta_v2f"]
                ):
                    ortho[arch_name][ds_name] = {
                        "K50": compute_orthogonality(
                            all_results[arch_name][ds_name]["results"],
                            all_results["meta_v2f"][ds_name]["results"],
                            K=50,
                        ),
                        "K20": compute_orthogonality(
                            all_results[arch_name][ds_name]["results"],
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
        "context_index_stats": dict(index_stats_by_variant_ds),
        "diagnostics": dict(diag),
        "orthogonality_vs_v2f": dict(ortho),
    }
    raw_path = RESULTS_DIR / "context_embedding.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset detail files
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"context_embedding_{a}_{d}.json"
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
    md.append("# Context-enriched embeddings (stacked merge)\n")
    md.append(
        "Ingest-time embedding of `{prev} [SEP] {curr} [SEP] {next}` "
        "style context-enriched texts. Query-time: run v2f, then "
        "stacked-append context-index hits in score order (no displacement "
        "of v2f's top-K picks). Zero per-query LLM overhead beyond v2f.\n"
    )

    md.append("## Index stats\n")
    for variant, ds_map in index_stats_by_variant_ds.items():
        md.append(f"### {variant}\n")
        for ds_name, stats in ds_map.items():
            md.append(
                f"- {ds_name}: entries_raw={stats.get('n_entries_raw')} "
                f"unique={stats.get('n_entries_unique')} "
                f"convs={stats.get('n_convs')}"
            )
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

    md.append("## Per-category breakdown\n")
    for a in CONTEXTEMB_VARIANTS:
        if a not in all_results:
            continue
        for d in EVAL_DATASETS:
            if d not in all_results[a]:
                continue
            by_cat = all_results[a][d]["category_breakdown"]
            md.append(f"### {a} / {d}\n")
            md.append("| category | n | Δ@20 | Δ@50 | W/T/L@50 |")
            md.append("|---|---:|---:|---:|---:|")
            for cat, c in by_cat.items():
                md.append(
                    f"| {cat} | {c['n']} | {c['delta_r@20']:+.3f} | "
                    f"{c['delta_r@50']:+.3f} | {c['W/T/L_r@50']} |"
                )
            md.append("")

    md.append("## Mechanism diagnostics\n")
    for arch_name in CONTEXTEMB_VARIANTS:
        if arch_name not in diag:
            continue
        for ds_name, d in diag[arch_name].items():
            md.append(f"### {arch_name} / {ds_name}\n")
            fired = d.get("n_queries_where_ctx_entered_topK", {})
            contrib = d.get("n_queries_ctx_contributed_gold", {})
            n = d.get("n_queries", 0)
            md.append(f"- n queries: {n}")
            md.append(
                f"- mean raw ctx hits / query: "
                f"{d.get('mean_ctx_hits_raw')}"
            )
            md.append(
                f"- mean ctx-turn hits (deduped) / query: "
                f"{d.get('mean_ctx_turn_hits')}"
            )
            md.append(
                f"- mean ctx-turn hits novel vs v2f / query: "
                f"{d.get('mean_ctx_turn_hits_novel_vs_v2f')}"
            )
            md.append(
                f"- queries where >=1 ctx turn entered top-20: "
                f"{fired.get('20')}/{n}"
            )
            md.append(
                f"- queries where >=1 ctx turn entered top-50: "
                f"{fired.get('50')}/{n}"
            )
            md.append(
                f"- queries where a ctx-appended turn was gold @K=20: "
                f"{contrib.get('20')}/{n}"
            )
            md.append(
                f"- queries where a ctx-appended turn was gold @K=50: "
                f"{contrib.get('50')}/{n}"
            )
            md.append("")

    md.append("## Orthogonality vs meta_v2f\n")
    md.append("| Variant | Dataset | K | total_gold | novel_vs_v2f | frac_novel |")
    md.append("|---|---|---:|---:|---:|---:|")
    for arch_name in CONTEXTEMB_VARIANTS:
        if arch_name not in ortho:
            continue
        for ds_name, od in ortho[arch_name].items():
            for K in ("K20", "K50"):
                o = od.get(K, {})
                md.append(
                    f"| {arch_name} | {ds_name} | {K[1:]} | "
                    f"{o.get('total_gold')} | {o.get('novel_vs_v2f')} | "
                    f"{o.get('fraction_novel')} |"
                )
    md.append("")

    md.append("## Sample retrievals (first 3 per variant/dataset)\n")
    for a in CONTEXTEMB_VARIANTS:
        if a not in all_results:
            continue
        for d in EVAL_DATASETS:
            if d not in all_results[a]:
                continue
            rs = all_results[a][d]["results"][:3]
            md.append(f"### {a} / {d}\n")
            for r in rs:
                md.append(
                    f"- **Q**: {r['question'][:100]}  "
                    f"gold={r['source_chat_ids']}"
                )
                md.append(
                    f"  ctx appended turn_ids: "
                    f"{r.get('ctx_appended_turn_ids', [])[:10]}  "
                    f"ctx in top-50: {r.get('ctx_in_topK', {}).get('50', [])}  "
                    f"Δ@50={r['fair_backfill'].get('delta_r@50'):+.3f}"
                )
            md.append("")

    # Verdict
    md.append("## Verdict\n")
    verdict_lines: list[str] = []
    if "meta_v2f" in all_results:
        for ds_name in EVAL_DATASETS:
            if ds_name not in all_results["meta_v2f"]:
                continue
            v2f_r50 = all_results["meta_v2f"][ds_name]["summary"].get(
                "arch_r@50", 0.0
            )
            v2f_r20 = all_results["meta_v2f"][ds_name]["summary"].get(
                "arch_r@20", 0.0
            )
            verdict_lines.append(
                f"**{ds_name}** — v2f baseline: r@20={v2f_r20:.3f} "
                f"r@50={v2f_r50:.3f}"
            )
            for arch_name in CONTEXTEMB_VARIANTS:
                if ds_name not in all_results.get(arch_name, {}):
                    continue
                s = all_results[arch_name][ds_name]["summary"]
                d20 = s["arch_r@20"] - v2f_r20
                d50 = s["arch_r@50"] - v2f_r50
                tag = "tie"
                if d50 > 0.005 or d20 > 0.005:
                    tag = "WIN"
                elif d50 < -0.005 or d20 < -0.005:
                    tag = "LOSS"
                verdict_lines.append(
                    f"  - {arch_name}: r@20={s['arch_r@20']:.3f} "
                    f"(Δ {d20:+.3f}) r@50={s['arch_r@50']:.3f} "
                    f"(Δ {d50:+.3f}) -> {tag}"
                )
    md.extend(verdict_lines)
    md.append("")

    md_path = RESULTS_DIR / "context_embedding.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final console table
    print("\n" + "=" * 110)
    print("CONTEXT EMBEDDING SUMMARY")
    print("=" * 110)
    for a in ARCH_CLASSES:
        for d in EVAL_DATASETS:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:28s} {d:14s} "
                f"a@20={s['arch_r@20']:.3f} d@20={s['delta_r@20']:+.3f}  "
                f"a@50={s['arch_r@50']:.3f} d@50={s['delta_r@50']:+.3f}  "
                f"llm/q={s['avg_llm_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
