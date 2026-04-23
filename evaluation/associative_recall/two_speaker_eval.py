"""Fair-backfill eval for TWO-speaker attributed retrieval.

Compares:
  - meta_v2f              (v2f reference, shared caches)
  - speaker_user_filter   (single-speaker baseline — user-only filter)
  - two_speaker_filter    (hard role filter on matched side)
  - two_speaker_boost_0.05 (score-bonus variant)

on LoCoMo-30 and synthetic-19 at K=20 and K=50.

Usage:
    uv run python two_speaker_eval.py
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
from speaker_attributed import SpeakerUserFilter
from two_speaker_filter import (
    ARCH_CLASSES as TS_ARCH_CLASSES,
    _CONV_TWO_SPEAKERS_FILE,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")
ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    "speaker_user_filter": SpeakerUserFilter,
    **TS_ARCH_CLASSES,
}
# Speaker-based arches (those with speaker-transform metadata keys).
SPEAKER_ARCHES = {
    "speaker_user_filter",
    "two_speaker_filter",
    "two_speaker_boost_0.05",
}
# Two-speaker arches only (those with matched_side metadata).
TWO_SPEAKER_ARCHES = set(TS_ARCH_CLASSES.keys())


def evaluate_question(arch, question: dict, arch_name: str) -> dict:
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
    }

    if arch_name in SPEAKER_ARCHES:
        # Unified metadata projection for reporting.
        row["conv_user_name"] = md.get("conv_user_name")
        row["query_name_tokens"] = md.get("query_name_tokens", [])
        row["applied_speaker_transform"] = md.get(
            "applied_speaker_transform", False
        )
        row["n_user_in_v2f"] = md.get("n_user_in_v2f")
        if arch_name == "speaker_user_filter":
            row["matched_side"] = (
                "user" if md.get("query_mentions_conv_user") else "none"
            )
        else:  # two_speaker_*
            row["conv_assistant_name"] = md.get("conv_assistant_name")
            row["matched_side"] = md.get("matched_side", "none")
            row["n_assistant_in_v2f"] = md.get("n_assistant_in_v2f")

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

    results: list[dict] = []
    for i, q in enumerate(questions):
        q_short = q["question"][:55]
        print(
            f"  [{i+1}/{len(questions)}] "
            f"{q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(arch, q, arch_name)
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
    return results, summary, by_cat


def coverage_by_side(speaker_results: list[dict]) -> dict:
    """Count how queries split across user/assistant/both/none sides."""
    n = len(speaker_results)
    if n == 0:
        return {}
    sides: dict[str, int] = defaultdict(int)
    for r in speaker_results:
        sides[r.get("matched_side", "none")] += 1
    out = {
        "n_queries": n,
        "sides": dict(sides),
        "frac_user_only": round(sides.get("user", 0) / n, 4),
        "frac_assistant_only": round(sides.get("assistant", 0) / n, 4),
        "frac_both": round(sides.get("both", 0) / n, 4),
        "frac_none": round(sides.get("none", 0) / n, 4),
        "frac_any_single_side": round(
            (sides.get("user", 0) + sides.get("assistant", 0)) / n, 4
        ),
    }
    # Per-category breakdown.
    cat_sides: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for r in speaker_results:
        cat_sides[r.get("category", "?")][r.get("matched_side", "none")] += 1
    out["by_category"] = {
        c: dict(s) for c, s in cat_sides.items()
    }
    return out


def subset_delta_table(
    arch_results: list[dict],
    ref_results: list[dict],
    subset_filter,
    subset_name: str,
) -> dict:
    """Per-K recall on a SUBSET (defined by subset_filter on arch_results)."""
    ref_by_key = {
        (r["conversation_id"], r["question_index"]): r for r in ref_results
    }
    subset = [r for r in arch_results if subset_filter(r)]
    n = len(subset)
    out: dict[str, object] = {"subset": subset_name, "n": n}
    if n == 0:
        return out
    for K in BUDGETS:
        ref_vals, arch_vals = [], []
        for r in subset:
            k = (r["conversation_id"], r["question_index"])
            ref_r = ref_by_key.get(k)
            if ref_r is None:
                continue
            ref_vals.append(ref_r["fair_backfill"][f"arch_r@{K}"])
            arch_vals.append(r["fair_backfill"][f"arch_r@{K}"])
        if not ref_vals:
            continue
        r_m = sum(ref_vals) / len(ref_vals)
        a_m = sum(arch_vals) / len(arch_vals)
        wins = sum(1 for v, a in zip(ref_vals, arch_vals) if a > v + 0.001)
        losses = sum(1 for v, a in zip(ref_vals, arch_vals) if v > a + 0.001)
        ties = len(ref_vals) - wins - losses
        out[f"ref_r@{K}"] = round(r_m, 4)
        out[f"arch_r@{K}"] = round(a_m, 4)
        out[f"delta_r@{K}"] = round(a_m - r_m, 4)
        out[f"W/T/L@{K}"] = f"{wins}/{ties}/{losses}"
    return out


def two_speaker_id_summary() -> dict:
    if not _CONV_TWO_SPEAKERS_FILE.exists():
        return {"n_conversations": 0, "pairs": {}}
    try:
        with open(_CONV_TWO_SPEAKERS_FILE) as f:
            data = json.load(f)
        pairs = data.get("speakers", {}) or {}
    except (json.JSONDecodeError, OSError):
        pairs = {}
    n = len(pairs)
    both_known = sum(
        1 for p in pairs.values()
        if p.get("user", "UNKNOWN") != "UNKNOWN"
        and p.get("assistant", "UNKNOWN") != "UNKNOWN"
    )
    one_known = sum(
        1 for p in pairs.values()
        if (p.get("user", "UNKNOWN") != "UNKNOWN")
        ^ (p.get("assistant", "UNKNOWN") != "UNKNOWN")
    )
    none_known = n - both_known - one_known
    return {
        "n_conversations": n,
        "n_both_identified": both_known,
        "n_one_identified": one_known,
        "n_none_identified": none_known,
        "both_hit_rate": round(both_known / n, 4) if n else 0.0,
        "any_hit_rate": (
            round((both_known + one_known) / n, 4) if n else 0.0
        ),
        "pairs": pairs,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = defaultdict(dict)

    for ds_name in EVAL_DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
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
            }

    # --- Two-speaker ID summary ---
    sid_summary = two_speaker_id_summary()

    # --- Coverage diagnostics (from two_speaker_filter on each dataset) ---
    primary_arch = "two_speaker_filter"
    coverage: dict[str, dict] = {}
    for ds_name in EVAL_DATASETS:
        rs = all_results.get(primary_arch, {}).get(ds_name, {}).get(
            "results", []
        )
        coverage[ds_name] = coverage_by_side(rs)

    # --- Subset delta tables (two-speaker arches) ---
    subset_tables: dict[str, dict] = defaultdict(dict)
    for ds_name in EVAL_DATASETS:
        v2f_rs = all_results.get("meta_v2f", {}).get(ds_name, {}).get(
            "results", []
        )
        sp_rs = all_results.get("speaker_user_filter", {}).get(ds_name, {}).get(
            "results", []
        )
        for arch_name in TWO_SPEAKER_ARCHES:
            rs = all_results.get(arch_name, {}).get(ds_name, {}).get(
                "results", []
            )
            if not rs:
                continue
            # 1. user-only subset vs v2f.
            subset_tables[arch_name][f"{ds_name}__user_only_vs_v2f"] = (
                subset_delta_table(
                    rs,
                    v2f_rs,
                    lambda r: r.get("matched_side") == "user",
                    "mentions_user_only",
                )
            )
            # 2. assistant-only subset vs v2f (the NEW coverage).
            subset_tables[arch_name][f"{ds_name}__assistant_only_vs_v2f"] = (
                subset_delta_table(
                    rs,
                    v2f_rs,
                    lambda r: r.get("matched_side") == "assistant",
                    "mentions_assistant_only",
                )
            )
            # 3. assistant-only subset vs speaker_user_filter (does the NEW
            #    coverage add wins beyond the single-speaker baseline?).
            subset_tables[arch_name][
                f"{ds_name}__assistant_only_vs_spf"
            ] = subset_delta_table(
                rs,
                sp_rs,
                lambda r: r.get("matched_side") == "assistant",
                "mentions_assistant_only",
            )
            # 4. Full transform-fired subset (user|assistant) vs v2f.
            subset_tables[arch_name][f"{ds_name}__transform_fired_vs_v2f"] = (
                subset_delta_table(
                    rs,
                    v2f_rs,
                    lambda r: r.get("matched_side")
                    in ("user", "assistant"),
                    "transform_fired",
                )
            )

    # --- Save raw JSON ---
    raw = {
        "archs": list(ARCH_CLASSES.keys()),
        "datasets": list(EVAL_DATASETS),
        "two_speaker_id_summary": sid_summary,
        "coverage_by_side": coverage,
        "subset_tables": subset_tables,
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
    }
    raw_path = RESULTS_DIR / "two_speaker_filter.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset details.
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"two_speaker_{a}_{d}.json"
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
    md.append("# Two-speaker attributed retrieval\n")
    md.append(
        "Extends `speaker_user_filter` (user-side only) to cover BOTH\n"
        "participants. At ingest: one LLM call per conv identifies both\n"
        "the user-role and assistant-role speakers (LoCoMo's 'assistant'\n"
        "is often a second human). At query time: if the query mentions\n"
        "exactly one side, apply the speaker-aware transform to that\n"
        "role; if both or neither, no transform.\n"
    )

    # Two-speaker IDs
    md.append("## Two-speaker identification\n")
    md.append(
        f"- Conversations scanned: {sid_summary['n_conversations']}"
    )
    md.append(
        f"- Both identified: {sid_summary['n_both_identified']} "
        f"({sid_summary['both_hit_rate']:.1%})"
    )
    md.append(
        f"- One side identified: {sid_summary['n_one_identified']}"
    )
    md.append(
        f"- Neither identified: {sid_summary['n_none_identified']}"
    )
    md.append(
        f"- Any-side hit rate: {sid_summary['any_hit_rate']:.1%}\n"
    )
    md.append("| Conversation | user | assistant |")
    md.append("|---|---|---|")
    for cid, pair in sorted(sid_summary["pairs"].items()):
        md.append(
            f"| {cid} | {pair.get('user', '?')} | "
            f"{pair.get('assistant', '?')} |"
        )
    md.append("")

    # Query coverage by side
    md.append("## Query coverage by side\n")
    md.append("| Dataset | n | user-only | assistant-only | both | none |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for ds_name, cov in coverage.items():
        s = cov.get("sides", {})
        md.append(
            f"| {ds_name} | {cov.get('n_queries')} | "
            f"{s.get('user', 0)} ({cov.get('frac_user_only', 0):.1%}) | "
            f"{s.get('assistant', 0)} "
            f"({cov.get('frac_assistant_only', 0):.1%}) | "
            f"{s.get('both', 0)} ({cov.get('frac_both', 0):.1%}) | "
            f"{s.get('none', 0)} ({cov.get('frac_none', 0):.1%}) |"
        )
    md.append("")
    md.append("### Per-category side counts\n")
    for ds_name, cov in coverage.items():
        md.append(f"**{ds_name}**: {cov.get('by_category', {})}\n")

    # Full-set recall
    md.append("## Fair-backfill recall (full question sets)\n")
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
                f"{s['delta_r@50']:+.3f} | {s['avg_llm_calls']:.1f} |"
            )
    md.append("")

    # Subset tables
    md.append("## Subset recall tables\n")
    md.append(
        "Each row restricts to a subset of queries (by side-mention) and\n"
        "compares the two-speaker arch vs a reference (meta_v2f or\n"
        "speaker_user_filter).\n"
    )
    md.append(
        "| Arch | Subset key | n | ref@20 | arch@20 | Δ@20 | ref@50 | "
        "arch@50 | Δ@50 | W/T/L@50 |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arch_name in TWO_SPEAKER_ARCHES:
        by_ds = subset_tables.get(arch_name, {})
        for key, row in by_ds.items():
            if not row or row.get("n", 0) == 0:
                continue
            md.append(
                f"| {arch_name} | {key} | {row['n']} | "
                f"{row.get('ref_r@20', 0):.3f} | "
                f"{row.get('arch_r@20', 0):.3f} | "
                f"{row.get('delta_r@20', 0):+.3f} | "
                f"{row.get('ref_r@50', 0):.3f} | "
                f"{row.get('arch_r@50', 0):.3f} | "
                f"{row.get('delta_r@50', 0):+.3f} | "
                f"{row.get('W/T/L@50', 'NA')} |"
            )
    md.append("")

    # Per-category breakdown for two_speaker_filter
    md.append("## Per-category (two_speaker_filter vs base cosine)\n")
    if "two_speaker_filter" in all_results:
        for d in EVAL_DATASETS:
            if d not in all_results["two_speaker_filter"]:
                continue
            by_cat = all_results["two_speaker_filter"][d][
                "category_breakdown"
            ]
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
    v2f_lc = all_results.get("meta_v2f", {}).get("locomo_30q", {}).get(
        "summary", {}
    )
    spf_lc = all_results.get("speaker_user_filter", {}).get(
        "locomo_30q", {}
    ).get("summary", {})

    verdict_lines.append(
        f"- Both-side ID hit rate: {sid_summary['both_hit_rate']:.1%}"
    )
    for ds_name in EVAL_DATASETS:
        cov = coverage.get(ds_name, {})
        verdict_lines.append(
            f"- {ds_name} side coverage: "
            f"user-only={cov.get('frac_user_only', 0):.1%}, "
            f"assistant-only={cov.get('frac_assistant_only', 0):.1%}, "
            f"both={cov.get('frac_both', 0):.1%}, "
            f"none={cov.get('frac_none', 0):.1%}"
        )

    # Compare two_speaker_filter vs speaker_user_filter on LoCoMo.
    tsf_lc = all_results.get("two_speaker_filter", {}).get(
        "locomo_30q", {}
    ).get("summary", {})
    if tsf_lc and spf_lc:
        d20 = tsf_lc.get("arch_r@20", 0) - spf_lc.get("arch_r@20", 0)
        d50 = tsf_lc.get("arch_r@50", 0) - spf_lc.get("arch_r@50", 0)
        d20_v = tsf_lc.get("arch_r@20", 0) - v2f_lc.get("arch_r@20", 0)
        d50_v = tsf_lc.get("arch_r@50", 0) - v2f_lc.get("arch_r@50", 0)
        verdict_lines.append(
            f"- LoCoMo two_speaker_filter vs speaker_user_filter: "
            f"Δ@20={d20:+.3f}, Δ@50={d50:+.3f}"
        )
        verdict_lines.append(
            f"- LoCoMo two_speaker_filter vs meta_v2f: "
            f"Δ@20={d20_v:+.3f}, Δ@50={d50_v:+.3f}"
        )

        # Decision.
        if d20 > 0.005 or d50 > 0.005:
            verdict = (
                f"**SHIP two_speaker_filter** — extends speaker_user_filter"
                f" to the assistant side and improves on it (Δ@20={d20:+.3f}"
                f", Δ@50={d50:+.3f}) at ~0 extra per-query LLM cost."
            )
        elif d20 < -0.005 or d50 < -0.005:
            verdict = (
                f"**KEEP speaker_user_filter** — two-speaker extension "
                f"regresses on LoCoMo (Δ@20={d20:+.3f}, Δ@50={d50:+.3f})."
            )
        else:
            # Is the assistant-only subset bringing any real wins?
            asst_key = (
                "locomo_30q__assistant_only_vs_v2f"
            )
            asst_row = subset_tables.get(
                "two_speaker_filter", {}
            ).get(asst_key, {})
            n_asst = asst_row.get("n", 0)
            if n_asst == 0:
                verdict = (
                    "**ABANDON** — no LoCoMo queries mention the assistant"
                    " name; extension cannot fire."
                )
            else:
                verdict = (
                    f"**FLAT / ALREADY SATURATED** — two-speaker "
                    f"extension fires on {n_asst} additional queries, "
                    f"but net Δ vs speaker_user_filter is ~0 "
                    f"(Δ@20={d20:+.3f}, Δ@50={d50:+.3f}). "
                    f"The user-only filter + fair-backfill was already "
                    f"surfacing enough gold for assistant-mention queries."
                )
    else:
        verdict = "incomplete run"
    md.append("\n".join(verdict_lines))
    md.append("")
    md.append(verdict + "\n")

    md_path = RESULTS_DIR / "two_speaker_filter.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final console table
    print("\n" + "=" * 100)
    print("TWO-SPEAKER ATTRIBUTED SUMMARY")
    print("=" * 100)
    for a in ARCH_CLASSES:
        for d in EVAL_DATASETS:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:24s} {d:14s} "
                f"a@20={s['arch_r@20']:.3f} d@20={s['delta_r@20']:+.3f}  "
                f"a@50={s['arch_r@50']:.3f} d@50={s['delta_r@50']:+.3f}  "
                f"llm/q={s['avg_llm_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
