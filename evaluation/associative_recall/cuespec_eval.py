"""Evaluate the model-agnostic cue-generation spec.

Variants:
  meta_v2f                (reference — pristine, loaded from existing results)
  cuespec_mini            (gpt-5-mini + spec + 1 repair)
  cuespec_nano            (gpt-5-nano + spec + 2 repair rounds)
  cuespec_nano_no_repair  (gpt-5-nano + spec, ablation)
  v2f_nano                (gpt-5-nano + vanilla v2f, replicates failure mode)

Datasets: locomo_30q, synthetic_19q. K=20, K=50.

Outputs:
  results/cuespec_<variant>_<dataset>.json        — per-variant per-dataset raw
  results/cue_spec.json                           — combined raw
  results/cue_spec.md                             — final report

Usage:
  uv run python cuespec_eval.py
  uv run python cuespec_eval.py --force
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from associative_recall import Segment
from cue_spec import (
    CueSpecBase,
    build_variants,
)
from dotenv import load_dotenv
from type_enumerated import (
    BUDGETS,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

TARGET_DATASETS = ["locomo_30q", "synthetic_19q"]
VARIANT_NAMES = [
    "cuespec_mini",
    "cuespec_nano",
    "cuespec_nano_no_repair",
    "v2f_nano",
]


def evaluate_one(arch: CueSpecBase, question: dict) -> dict:
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
    cosine_result = arch.store.search(query_emb, top_k=max_K, conversation_id=conv_id)
    cosine_segments = list(cosine_result.segments)

    fair: dict[str, float] = {}
    for K in BUDGETS:
        b, a = fair_backfill_evaluate(arch_segments, cosine_segments, source_ids, K)
        fair[f"baseline_r@{K}"] = round(b, 4)
        fair[f"arch_r@{K}"] = round(a, 4)
        fair[f"delta_r@{K}"] = round(a - b, 4)

    return {
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
        "fair_backfill": fair,
        "metadata": result.metadata,
    }


def run_variant_on_dataset(
    variant_name: str,
    ds_name: str,
    force: bool,
) -> dict:
    out_path = RESULTS_DIR / f"cuespec_{variant_name}_{ds_name}.json"
    if out_path.exists() and not force:
        print(f"  [SKIP] {out_path.name} exists (use --force to rerun)")
        with open(out_path) as f:
            return json.load(f)

    store, questions = load_dataset(ds_name)
    variants = build_variants(store)
    arch = variants[variant_name]

    print(f"\n--- {variant_name} on {ds_name} (n={len(questions)}) ---")

    rows: list[dict] = []
    for i, q in enumerate(questions):
        q_short = q["question"][:60]
        print(
            f"  [{i + 1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_one(arch, q)
            rows.append(row)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()

        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()

    summary = summarize(rows, variant_name, ds_name)
    by_cat = summarize_by_category(rows)

    # Spec-specific repair statistics
    repair_stats = _compute_repair_stats(rows)

    saved = {
        "variant": variant_name,
        "model": arch.model,
        "dataset": ds_name,
        "summary": summary,
        "category_breakdown": by_cat,
        "repair_stats": repair_stats,
        "results": rows,
    }
    with open(out_path, "w") as f:
        json.dump(saved, f, indent=2, default=str)

    _print_variant_summary(saved)
    return saved


def _compute_repair_stats(rows: list[dict]) -> dict:
    """Per-variant stats on the verify-repair loop."""
    n = len(rows)
    if n == 0:
        return {"n": 0}

    n_needed_repair = 0
    n_final_ok = 0
    repair_rounds: list[int] = []
    llm_calls: list[int] = []

    # Count frequency of each failure reason (first-attempt)
    reason_counts: dict[str, int] = {}

    for r in rows:
        meta = r.get("metadata", {})
        attempts = meta.get("attempts", [])
        n_attempts = len(attempts)
        repair_rounds.append(max(n_attempts - 1, 0))
        llm_calls.append(r.get("llm_calls", 0))

        if attempts:
            first = attempts[0]
            vd = first.get("validation") or {}
            if not vd.get("ok", True):
                n_needed_repair += 1
                for f in vd.get("failures", []):
                    for reason in f.get("reasons", []):
                        # normalize numeric details out of the reason tag
                        base = reason.split("(")[0]
                        reason_counts[base] = reason_counts.get(base, 0) + 1
                for reason in vd.get("set_level_failures", []):
                    base = reason.split("(")[0]
                    reason_counts[base] = reason_counts.get(base, 0) + 1

        if meta.get("final_validation_ok"):
            n_final_ok += 1

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "n": n,
        "n_needed_repair": n_needed_repair,
        "repair_rate": round(n_needed_repair / n, 4),
        "n_final_ok": n_final_ok,
        "final_ok_rate": round(n_final_ok / n, 4),
        "avg_repair_rounds": round(_mean(repair_rounds), 3),
        "max_repair_rounds": max(repair_rounds) if repair_rounds else 0,
        "avg_llm_calls": round(_mean(llm_calls), 3),
        "first_attempt_failure_reasons": dict(
            sorted(reason_counts.items(), key=lambda kv: -kv[1])
        ),
    }


def _print_variant_summary(saved: dict) -> None:
    s = saved["summary"]
    rs = saved["repair_stats"]
    name = saved["variant"]
    ds = saved["dataset"]
    print(
        f"  [{name}/{ds}] r@20: base={s.get('baseline_r@20', 0):.3f} "
        f"arch={s.get('arch_r@20', 0):.3f} "
        f"delta={s.get('delta_r@20', 0):+.3f} "
        f"W/T/L={s.get('W/T/L_r@20', '?')}  |  "
        f"r@50: arch={s.get('arch_r@50', 0):.3f}"
    )
    print(
        f"    repair_rate={rs.get('repair_rate', 0):.2f}  "
        f"final_ok_rate={rs.get('final_ok_rate', 0):.2f}  "
        f"avg_repair_rounds={rs.get('avg_repair_rounds', 0):.2f}  "
        f"avg_llm_calls={rs.get('avg_llm_calls', 0):.2f}"
    )


# ---------------------------------------------------------------------------
# Reference baseline — load pristine mini `meta_v2f` numbers from existing
# fairbackfill result files (do not re-run).
# ---------------------------------------------------------------------------


def _load_meta_v2f_reference(ds_name: str) -> dict | None:
    path = RESULTS_DIR / f"fairbackfill_meta_v2f_{ds_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _collect_all(runs: dict) -> dict:
    """Produce a nested combined summary for cue_spec.json."""
    combined: dict = {
        "variants": {},
        "reference_meta_v2f": {},
    }
    for variant_name, per_ds in runs.items():
        combined["variants"][variant_name] = {}
        for ds_name, saved in per_ds.items():
            combined["variants"][variant_name][ds_name] = {
                "summary": saved["summary"],
                "category_breakdown": saved["category_breakdown"],
                "repair_stats": saved["repair_stats"],
                "model": saved["model"],
            }
    for ds_name in TARGET_DATASETS:
        ref = _load_meta_v2f_reference(ds_name)
        if ref is not None:
            combined["reference_meta_v2f"][ds_name] = {
                "summary": ref.get("summary", {}),
                "category_breakdown": ref.get("category_breakdown", {}),
            }
    return combined


def _sample_cues(runs: dict, n_samples: int = 4) -> list[dict]:
    """Collect side-by-side cue samples across variants for the report.

    For each of a few representative questions, show cues from each variant.
    Uses locomo_30q by default.
    """
    ds = "locomo_30q"
    # Pick the first n_samples questions that exist across all variants.
    if ds not in runs.get("cuespec_nano", {}):
        return []
    nano_rows = runs["cuespec_nano"][ds].get("results", [])
    mini_rows = runs.get("cuespec_mini", {}).get(ds, {}).get("results", [])
    v2f_nano_rows = runs.get("v2f_nano", {}).get(ds, {}).get("results", [])
    nano_noreprows = (
        runs.get("cuespec_nano_no_repair", {}).get(ds, {}).get("results", [])
    )

    def _index_rows(rs: list[dict]) -> dict[tuple[str, int], dict]:
        return {(r["conversation_id"], r["question_index"]): r for r in rs}

    mi = _index_rows(mini_rows)
    nr = _index_rows(nano_noreprows)
    vn = _index_rows(v2f_nano_rows)

    samples: list[dict] = []
    for row in nano_rows[:n_samples]:
        key = (row["conversation_id"], row["question_index"])
        entry: dict = {
            "dataset": ds,
            "question": row["question"],
            "category": row["category"],
            "cuespec_nano_cues": row["metadata"].get("cues", []),
            "cuespec_nano_num_attempts": row["metadata"].get("num_attempts", 0),
            "cuespec_nano_attempt_0_cues": (
                row["metadata"].get("attempts", [{}])[0].get("parsed_cues", [])
                if row["metadata"].get("attempts")
                else []
            ),
        }
        if key in mi:
            entry["cuespec_mini_cues"] = mi[key]["metadata"].get("cues", [])
        if key in nr:
            entry["cuespec_nano_no_repair_cues"] = nr[key]["metadata"].get("cues", [])
        if key in vn:
            entry["v2f_nano_cues"] = vn[key]["metadata"].get("cues", [])
        samples.append(entry)
    return samples


def render_report(combined: dict, samples: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Model-agnostic cue-generation spec — results")
    lines.append("")
    lines.append(
        "Mechanism: structural constraints + verify-repair loop. "
        "Same prompt template for any competent model."
    )
    lines.append("")

    lines.append("## Spec constraints (per cue)")
    lines.append("")
    lines.append("1. Length 8-35 words.")
    lines.append(
        "2. Entity overlap: >= 1 non-stopword content token taken from the "
        "query OR from the retrieved hop-0 excerpts."
    )
    lines.append(
        "3. Anti-paraphrase: cue does not start with "
        "{what, when, how, why, who, which, where}."
    )
    lines.append("4. Anti-duplication: Jaccard similarity with the query < 0.40.")
    lines.append("5. Casual-chat register: at most 2 sentences.")
    lines.append("")
    lines.append("## Spec constraints (set level)")
    lines.append("")
    lines.append("6. Pairwise cue cosine < 0.85 (anti-redundant).")
    lines.append("7. Each cue's cosine with the query > 0.30 (anti-random).")
    lines.append("")
    lines.append(
        "Prompt guidance (single template, used by every model): generate "
        "1-2-sentence first-person chat-style text that would answer the "
        "question, pulling specific vocabulary from the hop-0 excerpts when "
        "relevant."
    )
    lines.append("")

    lines.append("## Recall table")
    lines.append("")
    lines.append(
        "| Variant | Dataset | r@20 (arch) | r@50 (arch) | "
        "baseline r@20 | delta_r@20 | W/T/L@20 |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    # meta_v2f reference rows first
    for ds in TARGET_DATASETS:
        ref = combined["reference_meta_v2f"].get(ds)
        if ref:
            s = ref.get("summary", {})
            lines.append(
                f"| meta_v2f (reference mini) | {ds} | "
                f"{s.get('arch_r@20', 0):.3f} | {s.get('arch_r@50', 0):.3f} | "
                f"{s.get('baseline_r@20', 0):.3f} | "
                f"{s.get('delta_r@20', 0):+.3f} | "
                f"{s.get('W/T/L_r@20', '?')} |"
            )

    for variant_name in VARIANT_NAMES:
        for ds in TARGET_DATASETS:
            v = combined["variants"].get(variant_name, {}).get(ds)
            if v is None:
                continue
            s = v["summary"]
            lines.append(
                f"| {variant_name} | {ds} | "
                f"{s.get('arch_r@20', 0):.3f} | {s.get('arch_r@50', 0):.3f} | "
                f"{s.get('baseline_r@20', 0):.3f} | "
                f"{s.get('delta_r@20', 0):+.3f} | "
                f"{s.get('W/T/L_r@20', '?')} |"
            )
    lines.append("")

    lines.append("## Repair-loop statistics (per dataset)")
    lines.append("")
    lines.append(
        "| Variant | Dataset | repair_rate | final_ok_rate | "
        "avg_repair_rounds | avg_llm_calls |"
    )
    lines.append("|---|---|---|---|---|---|")
    for variant_name in VARIANT_NAMES:
        for ds in TARGET_DATASETS:
            v = combined["variants"].get(variant_name, {}).get(ds)
            if v is None:
                continue
            rs = v["repair_stats"]
            lines.append(
                f"| {variant_name} | {ds} | "
                f"{rs.get('repair_rate', 0):.2f} | "
                f"{rs.get('final_ok_rate', 0):.2f} | "
                f"{rs.get('avg_repair_rounds', 0):.2f} | "
                f"{rs.get('avg_llm_calls', 0):.2f} |"
            )
    lines.append("")

    lines.append("## First-attempt failure reasons (cuespec_nano, locomo_30q)")
    lines.append("")
    nano_locomo = combined["variants"].get("cuespec_nano", {}).get("locomo_30q", {})
    reasons = nano_locomo.get("repair_stats", {}).get(
        "first_attempt_failure_reasons", {}
    )
    if reasons:
        for reason, count in reasons.items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("(no first-attempt failures recorded)")
    lines.append("")

    lines.append("## Sample cues (locomo_30q)")
    lines.append("")
    for s in samples:
        lines.append(f"**Q:** {s['question']}")
        if s.get("cuespec_mini_cues"):
            lines.append(f"- mini + spec: {s['cuespec_mini_cues']}")
        if s.get("v2f_nano_cues"):
            lines.append(f"- nano + v2f : {s['v2f_nano_cues']}")
        if s.get("cuespec_nano_no_repair_cues"):
            lines.append(
                f"- nano + spec (no repair): {s['cuespec_nano_no_repair_cues']}"
            )
        if s.get("cuespec_nano_attempt_0_cues"):
            lines.append(
                f"- nano + spec attempt 0  : {s['cuespec_nano_attempt_0_cues']}"
            )
        lines.append(
            f"- nano + spec final     : {s['cuespec_nano_cues']} "
            f"(attempts={s['cuespec_nano_num_attempts']})"
        )
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    verdict_lines = _compute_verdict(combined)
    for ln in verdict_lines:
        lines.append(ln)

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `results/cue_spec.md` (this report)")
    lines.append("- `results/cue_spec.json` (combined raw)")
    for variant_name in VARIANT_NAMES:
        for ds in TARGET_DATASETS:
            lines.append(f"- `results/cuespec_{variant_name}_{ds}.json`")
    return "\n".join(lines) + "\n"


def _compute_verdict(combined: dict) -> list[str]:
    lines: list[str] = []

    # Reference (mini meta_v2f) averaged across K=20 on both datasets
    def _avg_metric(source: dict, metric: str) -> float:
        vals = []
        for ds in TARGET_DATASETS:
            s = source.get(ds, {}).get("summary", {}) if source else {}
            v = s.get(metric)
            if v is not None:
                vals.append(v)
        return sum(vals) / len(vals) if vals else 0.0

    ref_20 = _avg_metric(combined.get("reference_meta_v2f", {}), "arch_r@20")
    nano_rep_20 = _avg_metric(combined["variants"].get("cuespec_nano", {}), "arch_r@20")
    nano_noreprep_20 = _avg_metric(
        combined["variants"].get("cuespec_nano_no_repair", {}), "arch_r@20"
    )
    v2f_nano_20 = _avg_metric(combined["variants"].get("v2f_nano", {}), "arch_r@20")
    mini_spec_20 = _avg_metric(
        combined["variants"].get("cuespec_mini", {}), "arch_r@20"
    )

    lines.append(f"- Reference mini+v2f (meta_v2f) r@20 avg: **{ref_20:.3f}**")
    lines.append(
        f"- nano + vanilla v2f          r@20 avg: **{v2f_nano_20:.3f}**  "
        f"(delta vs ref = {v2f_nano_20 - ref_20:+.3f})"
    )
    lines.append(
        f"- nano + spec (no repair)     r@20 avg: **{nano_noreprep_20:.3f}**  "
        f"(delta vs ref = {nano_noreprep_20 - ref_20:+.3f})"
    )
    lines.append(
        f"- nano + spec + repair        r@20 avg: **{nano_rep_20:.3f}**  "
        f"(delta vs ref = {nano_rep_20 - ref_20:+.3f})"
    )
    lines.append(
        f"- mini + spec + repair        r@20 avg: **{mini_spec_20:.3f}**  "
        f"(delta vs ref = {mini_spec_20 - ref_20:+.3f})"
    )
    lines.append("")

    if ref_20 > 0:
        nano_pct = nano_rep_20 / ref_20 * 100 if ref_20 else 0
        lines.append(
            f"nano+spec+repair reaches **{nano_pct:.1f}%** of the mini+v2f "
            f"reference recall."
        )
        if nano_pct >= 90.0:
            lines.append(
                "**SHIP.** The spec is model-agnostic in practice — "
                "nano closes >=90% of the mini gap."
            )
        elif nano_rep_20 - v2f_nano_20 >= 0.05:
            lines.append(
                "**PARTIAL.** The spec meaningfully rescues nano versus "
                "vanilla v2f but does not fully close the gap to mini."
            )
        else:
            lines.append(
                "**FAIL.** Spec + repair did not rescue nano. Nano's "
                "language-model capacity is likely insufficient for this task."
            )

    # Spec regression check on mini
    if mini_spec_20 and ref_20:
        mini_delta = mini_spec_20 - ref_20
        if mini_delta < -0.02:
            lines.append(
                f"Spec regresses mini by {mini_delta:+.3f} vs pristine v2f "
                "— the spec may be over-constrained."
            )
        else:
            lines.append(
                f"Spec does not regress mini (delta vs v2f = {mini_delta:+.3f})."
            )

    # Repair contribution
    if nano_rep_20 and nano_noreprep_20:
        lift = nano_rep_20 - nano_noreprep_20
        lines.append(
            f"Repair loop lift for nano: {lift:+.3f} r@20 "
            f"(spec prompt alone vs spec+repair)."
        )
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the cue-generation spec.")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite cached result files."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Run a single variant only (for iterative debugging).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run a single dataset only.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variants = VARIANT_NAMES if args.variant is None else [args.variant]
    datasets = TARGET_DATASETS if args.dataset is None else [args.dataset]

    runs: dict[str, dict[str, dict]] = {v: {} for v in VARIANT_NAMES}
    for variant_name in variants:
        for ds in datasets:
            saved = run_variant_on_dataset(variant_name, ds, force=args.force)
            runs[variant_name][ds] = saved

    # If we did partial runs, also load any existing files for the report.
    for variant_name in VARIANT_NAMES:
        for ds in TARGET_DATASETS:
            if ds in runs.get(variant_name, {}):
                continue
            path = RESULTS_DIR / f"cuespec_{variant_name}_{ds}.json"
            if path.exists():
                with open(path) as f:
                    runs.setdefault(variant_name, {})[ds] = json.load(f)

    combined = _collect_all(runs)
    samples = _sample_cues(runs, n_samples=4)

    json_path = RESULTS_DIR / "cue_spec.json"
    with open(json_path, "w") as f:
        json.dump(
            {"combined": combined, "samples": samples},
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved combined raw: {json_path}")

    md_path = RESULTS_DIR / "cue_spec.md"
    md = render_report(combined, samples)
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved report: {md_path}")


if __name__ == "__main__":
    main()
