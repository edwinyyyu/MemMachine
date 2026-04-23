"""Fair-backfill eval for SPEAKER-CONDITIONAL cue generation.

Compares on LoCoMo-30 (primary) at K=20 and K=50:
  - meta_v2f                     reference baseline (no filter, generic cues)
  - two_speaker_filter           role-filter baseline (ceiling to beat: 0.892)
  - speaker_cond_cue_only        first-person conditioning, no filter
  - speaker_cond_plus_filter     conditioning + role-filter (expected best)
  - v2f_mention_tag              ablation: hint only, no conditioning

Usage:
    uv run python speakerCue_eval.py
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
from two_speaker_filter import TwoSpeakerFilter
from speaker_conditional_cue import (
    ARCH_CLASSES as SC_ARCH_CLASSES,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q",)
ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    "two_speaker_filter": TwoSpeakerFilter,
    **SC_ARCH_CLASSES,
}
SPEAKER_COND_ARCHES = set(SC_ARCH_CLASSES.keys())


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

    # Unified metadata projection for reporting (speaker-aware arches only).
    if arch_name in SPEAKER_COND_ARCHES or arch_name == "two_speaker_filter":
        row["conv_user_name"] = md.get("conv_user_name")
        row["conv_assistant_name"] = md.get("conv_assistant_name")
        row["query_name_tokens"] = md.get("query_name_tokens", [])
        row["matched_side"] = md.get("matched_side", "none")
        row["n_user_in_v2f"] = md.get("n_user_in_v2f")
        row["n_assistant_in_v2f"] = md.get("n_assistant_in_v2f")
    if arch_name in SPEAKER_COND_ARCHES:
        row["conditioned_on"] = md.get("conditioned_on")
        row["applied_cue_conditioning"] = md.get(
            "applied_cue_conditioning", False
        )
        row["applied_filter"] = md.get("applied_filter", False)
        row["v2f_cues"] = md.get("v2f_cues", [])

    # Also keep v2f cues for meta_v2f for qualitative comparison.
    if arch_name == "meta_v2f":
        row["v2f_cues"] = md.get("cues", [])

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


def coverage_by_side(rs: list[dict]) -> dict:
    n = len(rs)
    if n == 0:
        return {}
    sides: dict[str, int] = defaultdict(int)
    for r in rs:
        sides[r.get("matched_side", "none")] += 1
    out = {
        "n_queries": n,
        "sides": dict(sides),
        "frac_user_only": round(sides.get("user", 0) / n, 4),
        "frac_assistant_only": round(sides.get("assistant", 0) / n, 4),
        "frac_both": round(sides.get("both", 0) / n, 4),
        "frac_none": round(sides.get("none", 0) / n, 4),
        "frac_single_side": round(
            (sides.get("user", 0) + sides.get("assistant", 0)) / n, 4
        ),
    }
    return out


def collect_sample_cues(
    v2f_results: list[dict],
    cond_results: list[dict],
    limit: int = 3,
) -> list[dict]:
    """For each query where cond conditioning fired, pair v2f cues with
    cond cues for side-by-side qualitative inspection."""
    v2f_by_key = {
        (r["conversation_id"], r["question_index"]): r for r in v2f_results
    }
    out: list[dict] = []
    for r in cond_results:
        if not r.get("applied_cue_conditioning"):
            continue
        key = (r["conversation_id"], r["question_index"])
        v2f_row = v2f_by_key.get(key)
        if v2f_row is None:
            continue
        out.append(
            {
                "conversation_id": r["conversation_id"],
                "question_index": r["question_index"],
                "question": r["question"],
                "conditioned_on": r.get("conditioned_on"),
                "v2f_cues": v2f_row.get("v2f_cues", []),
                "cond_cues": r.get("v2f_cues", []),
                "cond_arch_r@20": r["fair_backfill"].get("arch_r@20"),
                "v2f_arch_r@20": v2f_row["fair_backfill"].get("arch_r@20"),
                "cond_arch_r@50": r["fair_backfill"].get("arch_r@50"),
                "v2f_arch_r@50": v2f_row["fair_backfill"].get("arch_r@50"),
            }
        )
        if len(out) >= limit:
            break
    return out


def subset_delta_table(
    arch_results: list[dict],
    ref_results: list[dict],
    subset_filter,
    subset_name: str,
) -> dict:
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
        wins = sum(
            1 for v, a in zip(ref_vals, arch_vals) if a > v + 0.001
        )
        losses = sum(
            1 for v, a in zip(ref_vals, arch_vals) if v > a + 0.001
        )
        ties = len(ref_vals) - wins - losses
        out[f"ref_r@{K}"] = round(r_m, 4)
        out[f"arch_r@{K}"] = round(a_m, 4)
        out[f"delta_r@{K}"] = round(a_m - r_m, 4)
        out[f"W/T/L@{K}"] = f"{wins}/{ties}/{losses}"
    return out


def _fmt_delta_row(
    summary_a: dict, summary_ref: dict, label: str
) -> str:
    d20 = summary_a.get("arch_r@20", 0) - summary_ref.get("arch_r@20", 0)
    d50 = summary_a.get("arch_r@50", 0) - summary_ref.get("arch_r@50", 0)
    return (
        f"- {label}: Δ@20={d20:+.3f}, Δ@50={d50:+.3f}"
    )


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

    # --- Coverage diagnostics (from speaker_cond_cue_only) ---
    primary = "speaker_cond_cue_only"
    coverage: dict[str, dict] = {}
    for ds_name in EVAL_DATASETS:
        rs = all_results.get(primary, {}).get(ds_name, {}).get("results", [])
        coverage[ds_name] = coverage_by_side(rs)

    # --- Subset delta tables (where conditioning fires) ---
    subset_tables: dict[str, dict] = defaultdict(dict)
    for ds_name in EVAL_DATASETS:
        v2f_rs = (
            all_results.get("meta_v2f", {}).get(ds_name, {}).get("results", [])
        )
        tsf_rs = (
            all_results.get("two_speaker_filter", {})
            .get(ds_name, {}).get("results", [])
        )
        for arch_name in SPEAKER_COND_ARCHES:
            rs = all_results.get(arch_name, {}).get(ds_name, {}).get(
                "results", []
            )
            if not rs:
                continue
            subset_tables[arch_name][
                f"{ds_name}__conditioning_fired_vs_v2f"
            ] = subset_delta_table(
                rs,
                v2f_rs,
                lambda r: r.get("applied_cue_conditioning"),
                "conditioning_fired",
            )
            subset_tables[arch_name][
                f"{ds_name}__conditioning_fired_vs_tsf"
            ] = subset_delta_table(
                rs,
                tsf_rs,
                lambda r: r.get("applied_cue_conditioning"),
                "conditioning_fired",
            )

    # --- Sample cues for qualitative comparison ---
    sample_cues_by_arch: dict[str, list[dict]] = {}
    v2f_rs = all_results.get("meta_v2f", {}).get("locomo_30q", {}).get(
        "results", []
    )
    for arch_name in SPEAKER_COND_ARCHES:
        rs = all_results.get(arch_name, {}).get("locomo_30q", {}).get(
            "results", []
        )
        sample_cues_by_arch[arch_name] = collect_sample_cues(
            v2f_rs, rs, limit=3
        )

    # --- Save raw JSON ---
    raw = {
        "archs": list(ARCH_CLASSES.keys()),
        "datasets": list(EVAL_DATASETS),
        "coverage_by_side": coverage,
        "subset_tables": subset_tables,
        "sample_cues": sample_cues_by_arch,
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
    raw_path = RESULTS_DIR / "speaker_conditional_cue.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset details.
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"speakerCue_{a}_{d}.json"
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
    md.append("# Speaker-conditional cue generation\n")
    md.append(
        "When a query mentions one of the two conversation participants, "
        "condition v2f's cue generation to produce cues AS IF that "
        "participant were speaking — first-person, casual chat register. "
        "Goal: cues embedded in the same register as gold (first-person) "
        "turns should cosine-match more tightly.\n"
    )

    # Coverage
    md.append("## Query coverage by side (speaker_cond_cue_only view)\n")
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

    # Full-set recall
    md.append("## Fair-backfill recall (LoCoMo-30)\n")
    md.append(
        "| Arch | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | "
        "llm/q |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in ARCH_CLASSES:
        for d in EVAL_DATASETS:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            md.append(
                f"| {a} | {s['baseline_r@20']:.3f} | "
                f"{s['arch_r@20']:.3f} | {s['delta_r@20']:+.3f} | "
                f"{s['baseline_r@50']:.3f} | {s['arch_r@50']:.3f} | "
                f"{s['delta_r@50']:+.3f} | {s['avg_llm_calls']:.1f} |"
            )
    md.append("")

    # Subset tables
    md.append("## Subset recall on fired queries\n")
    md.append(
        "Each row restricts to queries where cue-conditioning fired "
        "(i.e. query mentions exactly one known speaker) and compares vs "
        "a reference arch.\n"
    )
    md.append(
        "| Arch | Subset key | n | ref@20 | arch@20 | Δ@20 | ref@50 | "
        "arch@50 | Δ@50 | W/T/L@50 |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arch_name in SPEAKER_COND_ARCHES:
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

    # Per-category breakdown for speaker_cond_plus_filter
    for arch_to_show in (
        "speaker_cond_plus_filter",
        "speaker_cond_cue_only",
    ):
        md.append(f"## Per-category ({arch_to_show})\n")
        if arch_to_show in all_results:
            for d in EVAL_DATASETS:
                if d not in all_results[arch_to_show]:
                    continue
                by_cat = all_results[arch_to_show][d][
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

    # Sample cues
    md.append("## Sample cues (v2f vs conditioned)\n")
    for arch_name in SPEAKER_COND_ARCHES:
        samples = sample_cues_by_arch.get(arch_name, [])
        if not samples:
            continue
        md.append(f"### {arch_name}\n")
        for sa in samples:
            md.append(
                f"**Q** (conv={sa['conversation_id']}, "
                f"conditioned_on={sa['conditioned_on']}): "
                f"{sa['question']}\n"
            )
            md.append("- v2f cues:")
            for c in sa.get("v2f_cues", []) or []:
                md.append(f"  - {c}")
            md.append(f"- {arch_name} cues:")
            for c in sa.get("cond_cues", []) or []:
                md.append(f"  - {c}")
            md.append(
                f"- r@20: v2f={sa.get('v2f_arch_r@20')}, "
                f"{arch_name}={sa.get('cond_arch_r@20')}; "
                f"r@50: v2f={sa.get('v2f_arch_r@50')}, "
                f"{arch_name}={sa.get('cond_arch_r@50')}\n"
            )

    # Verdict
    md.append("## Verdict\n")
    v2f_lc = all_results.get("meta_v2f", {}).get("locomo_30q", {}).get(
        "summary", {}
    )
    tsf_lc = all_results.get("two_speaker_filter", {}).get(
        "locomo_30q", {}
    ).get("summary", {})
    scc_lc = all_results.get("speaker_cond_cue_only", {}).get(
        "locomo_30q", {}
    ).get("summary", {})
    scpf_lc = all_results.get("speaker_cond_plus_filter", {}).get(
        "locomo_30q", {}
    ).get("summary", {})
    mtag_lc = all_results.get("v2f_mention_tag", {}).get(
        "locomo_30q", {}
    ).get("summary", {})

    verdict_lines: list[str] = []
    if v2f_lc:
        if scc_lc:
            verdict_lines.append(
                _fmt_delta_row(scc_lc, v2f_lc, "speaker_cond_cue_only vs v2f")
            )
        if scpf_lc:
            verdict_lines.append(
                _fmt_delta_row(
                    scpf_lc, v2f_lc, "speaker_cond_plus_filter vs v2f"
                )
            )
        if mtag_lc:
            verdict_lines.append(
                _fmt_delta_row(mtag_lc, v2f_lc, "v2f_mention_tag vs v2f")
            )
    if tsf_lc and scpf_lc:
        verdict_lines.append(
            _fmt_delta_row(
                scpf_lc, tsf_lc, "speaker_cond_plus_filter vs two_speaker_filter"
            )
        )

    # Decision per spec.
    verdict = "undetermined"
    if scpf_lc and tsf_lc:
        d20 = scpf_lc.get("arch_r@20", 0) - tsf_lc.get("arch_r@20", 0)
        d50 = scpf_lc.get("arch_r@50", 0) - tsf_lc.get("arch_r@50", 0)
        if d20 > 0.005 or d50 > 0.005:
            verdict = (
                f"**SHIP speaker_cond_plus_filter** — beats "
                f"two_speaker_filter ceiling (Δ@20={d20:+.3f}, "
                f"Δ@50={d50:+.3f}). New LoCoMo ceiling."
            )
        elif d20 < -0.005 and d50 < -0.005:
            verdict = (
                f"**ABANDON conditioning+filter** — regresses vs "
                f"two_speaker_filter (Δ@20={d20:+.3f}, "
                f"Δ@50={d50:+.3f}). Conditioning hurts once filter "
                f"already narrows candidates."
            )
        else:
            # Sub-decisions based on the other variants.
            if scc_lc and v2f_lc:
                d_cue20 = scc_lc.get("arch_r@20", 0) - v2f_lc.get(
                    "arch_r@20", 0
                )
                d_cue50 = scc_lc.get("arch_r@50", 0) - v2f_lc.get(
                    "arch_r@50", 0
                )
                if d_cue20 > 0.005 or d_cue50 > 0.005:
                    verdict = (
                        f"**SUPPLEMENT / FLAT AT CEILING** — "
                        f"speaker_cond_plus_filter ties "
                        f"two_speaker_filter at its saturated ceiling "
                        f"(Δ@20={d20:+.3f}, Δ@50={d50:+.3f}), but "
                        f"conditioning alone helps cues without filter "
                        f"(vs v2f: Δ@20={d_cue20:+.3f}, "
                        f"Δ@50={d_cue50:+.3f}). Conditioning has "
                        f"independent value in other pipelines."
                    )
                else:
                    verdict = (
                        f"**ABANDON** — conditioning adds nothing on top "
                        f"of v2f (cue-only vs v2f Δ@20={d_cue20:+.3f}, "
                        f"Δ@50={d_cue50:+.3f}) and is saturated by "
                        f"filter."
                    )
        # Hint-alone ablation.
        if mtag_lc and v2f_lc and scc_lc:
            d_tag20 = mtag_lc.get("arch_r@20", 0) - v2f_lc.get(
                "arch_r@20", 0
            )
            d_tag50 = mtag_lc.get("arch_r@50", 0) - v2f_lc.get(
                "arch_r@50", 0
            )
            d_cue20 = scc_lc.get("arch_r@20", 0) - v2f_lc.get(
                "arch_r@20", 0
            )
            d_cue50 = scc_lc.get("arch_r@50", 0) - v2f_lc.get(
                "arch_r@50", 0
            )
            if d_tag50 >= d_cue50 - 0.005 and d_tag20 >= d_cue20 - 0.005:
                verdict_lines.append(
                    "- **Hint-alone matches conditioning** — first-person "
                    "conditioning adds nothing beyond the NAME HINT. "
                    "Use the cheaper `v2f_mention_tag` if anything."
                )

    md.append("\n".join(verdict_lines))
    md.append("")
    md.append(verdict + "\n")

    md_path = RESULTS_DIR / "speaker_conditional_cue.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Console summary
    print("\n" + "=" * 100)
    print("SPEAKER-CONDITIONAL CUE SUMMARY")
    print("=" * 100)
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
