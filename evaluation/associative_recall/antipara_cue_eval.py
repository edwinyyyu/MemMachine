"""Fair-backfill eval of anti-paraphrase / verbatim-quote v2f variants.

Architectures:
  meta_v2f                      — v2f baseline (reference)
  v2f_anti_paraphrase           — v2f + explicit 'Do NOT restate/guess'
  v2f_verbatim_quote            — v2f + verbatim 2-5 word phrase requirement
  v2f_anti_paraphrase_verbatim  — both combined

Runs on 4 datasets (locomo_30q, synthetic_19q, puzzle_16q, advanced_23q)
at K=20 and K=50 using the fair-backfill methodology.

Usage:
    uv run python antipara_cue_eval.py
    uv run python antipara_cue_eval.py --archs meta_v2f,v2f_anti_paraphrase
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
from antipara_cue_gen import (
    MetaV2fDedicated,
    V2fAntiParaphrase,
    V2fAntiParaphraseVerbatim,
    V2fVerbatimQuote,
    verbatim_check,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


ARCH_CLASSES = {
    "meta_v2f": MetaV2fDedicated,
    "v2f_anti_paraphrase": V2fAntiParaphrase,
    "v2f_verbatim_quote": V2fVerbatimQuote,
    "v2f_anti_paraphrase_verbatim": V2fAntiParaphraseVerbatim,
}

TARGET_CATEGORIES = ("locomo_temporal", "locomo_single_hop")


def evaluate_question(arch, question: dict) -> dict:
    """Run arch on a single question, produce fair-backfill metrics + cues."""
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
        "cues": result.metadata.get("cues", []),
        "cues_raw": result.metadata.get("cues_raw", []),
        "cue_outcomes": result.metadata.get("cue_outcomes", []),
        "hop0_empty": result.metadata.get("hop0_empty", False),
    }

    for K in BUDGETS:
        b_rec, a_rec, _ = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)

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

    return results, summary, by_cat


def compile_cue_stats(all_results: dict) -> dict:
    """Aggregate cue-level stats per arch (paraphrase + verbatim compliance)."""
    stats: dict = {}
    for arch_name, by_ds in all_results.items():
        n_cues = 0
        n_kept = 0
        n_dropped_no_verbatim = 0
        n_empty_outcomes = 0
        n_quoting = 0  # cues that actually quote hop0 context
        for _ds, rows in by_ds.items():
            for r in rows:
                outcomes = r.get("cue_outcomes", [])
                cues = r.get("cues", [])
                for o in outcomes:
                    n_cues += 1
                    if o.get("kept"):
                        n_kept += 1
                    elif o.get("reason") == "no_verbatim":
                        n_dropped_no_verbatim += 1
                    elif o.get("reason") == "empty":
                        n_empty_outcomes += 1
                # For archs without outcomes (meta_v2f) count cues directly
                if not outcomes and cues:
                    n_cues += len(cues)
                    n_kept += len(cues)
        stats[arch_name] = {
            "total_cues_emitted": n_cues,
            "kept": n_kept,
            "dropped_no_verbatim": n_dropped_no_verbatim,
            "empty": n_empty_outcomes,
        }
    return stats


def paraphrase_rate(rows: list[dict]) -> float:
    """Fraction of cues whose text reads like a paraphrase of the question.

    Uses a cheap proxy: cue starts with a question word OR cue ends with '?'
    OR cue shares >=60% of non-stopword tokens with the question.
    """
    STOP = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "with", "at", "by", "is", "are", "was", "were", "be", "been",
        "what", "who", "where", "when", "why", "how", "did", "do", "does",
        "i", "you", "they", "we", "he", "she", "it", "that", "this",
    }
    q_words = ("what", "who", "where", "when", "why", "how", "which", "did", "do", "does", "is", "are", "was", "were")

    def toks(s: str) -> set[str]:
        return {
            t.lower()
            for t in __import__("re").findall(r"[A-Za-z][A-Za-z'\-]+", s)
            if t.lower() not in STOP
        }

    n = 0
    p = 0
    for r in rows:
        q_toks = toks(r["question"])
        for cue in r.get("cues", []):
            n += 1
            cue_low = cue.strip().lower()
            if cue_low.endswith("?"):
                p += 1
                continue
            if cue_low.split(" ", 1)[0] in q_words:
                p += 1
                continue
            c_toks = toks(cue)
            if q_toks and len(c_toks & q_toks) / max(len(c_toks), 1) >= 0.6:
                p += 1
    return p / max(n, 1)


def quoting_rate(rows: list[dict]) -> float:
    """Fraction of cues that contain a 2-5 word phrase from the question text.

    Diagnostic only. Not the same as quoting hop0 excerpts (we don't have
    excerpts at this stage), but gives a rough sense of verbatim adherence
    when combined with cue_outcomes.
    """
    # We rely on cue_outcomes for archs that enforce verbatim. Otherwise skip.
    n = 0
    q = 0
    for r in rows:
        for o in r.get("cue_outcomes", []):
            if o.get("reason") in ("ok", "no_verbatim"):
                n += 1
                if o.get("reason") == "ok" and o.get("kept"):
                    q += 1
    return q / max(n, 1) if n else float("nan")


def render_markdown(
    all_summaries: dict, all_results: dict, arch_names: list[str]
) -> str:
    """Produce the study's markdown report."""
    ds_order = list(DATASETS.keys())
    lines: list[str] = []
    lines.append("# Anti-Paraphrase Cue Generation Study\n")
    lines.append(
        "Empirical test of whether adding explicit anti-paraphrase "
        "instructions and/or a verbatim-quote rule to the v2f cue-generation "
        "prompt improves retrieval recall.\n"
    )

    # Constraints
    lines.append("## Constraints")
    lines.append("- `text-embedding-3-small`, `gpt-5-mini` (fixed).")
    lines.append("- No framework edits. Reuses hop0 top-10 + 2 cues top-10 structure.")
    lines.append("- Fair-backfill eval to K=20 and K=50.\n")

    # Variants
    lines.append("## Variants")
    lines.append(
        "- `v2f_anti_paraphrase` — v2f + two negative instructions: "
        "'Do NOT restate or paraphrase the question.' and 'Do NOT guess "
        "the answer.'"
    )
    lines.append(
        "- `v2f_verbatim_quote` — v2f + each cue must include a 2-5 word "
        "verbatim phrase from hop0 excerpts; 15-40 word length cap; "
        "post-filter drops cues that fail `verbatim_check`."
    )
    lines.append(
        "- `v2f_anti_paraphrase_verbatim` — both together.\n"
    )

    # Overall recall
    lines.append("## Overall recall (avg over 4 datasets)")
    lines.append("| Arch | avg r@20 | avg r@50 |")
    lines.append("|------|---------:|---------:|")
    for arch in arch_names:
        if arch not in all_summaries:
            continue
        r20 = []
        r50 = []
        for ds in ds_order:
            s = all_summaries[arch].get(ds, {}).get("summary")
            if not s:
                continue
            r20.append(s["arch_r@20"])
            r50.append(s["arch_r@50"])
        avg20 = sum(r20) / len(r20) if r20 else 0.0
        avg50 = sum(r50) / len(r50) if r50 else 0.0
        lines.append(f"| {arch} | {avg20:.4f} | {avg50:.4f} |")
    lines.append("")

    # Per-dataset recall
    lines.append("## Per-dataset arch recall")
    lines.append("| Dataset | K | " + " | ".join(arch_names) + " |")
    lines.append("|---------|---|" + "|".join(["------:"] * len(arch_names)) + "|")
    for ds in ds_order:
        for K in BUDGETS:
            cells = []
            for arch in arch_names:
                s = all_summaries.get(arch, {}).get(ds, {}).get("summary")
                cells.append(f"{s[f'arch_r@{K}']:.3f}" if s else "n/a")
            lines.append(f"| {ds} | {K} | " + " | ".join(cells) + " |")
    lines.append("")

    # Deltas vs v2f
    base_arch = "meta_v2f"
    if base_arch in all_summaries:
        lines.append("## Deltas vs `meta_v2f` (arch_r@K)")
        lines.append("| Dataset | K | " + " | ".join(
            a for a in arch_names if a != base_arch
        ) + " |")
        lines.append(
            "|---------|---|"
            + "|".join(["------:"] * (len(arch_names) - 1))
            + "|"
        )
        for ds in ds_order:
            for K in BUDGETS:
                base_s = all_summaries[base_arch].get(ds, {}).get("summary")
                if not base_s:
                    continue
                base_v = base_s[f"arch_r@{K}"]
                cells = []
                for arch in arch_names:
                    if arch == base_arch:
                        continue
                    s = all_summaries.get(arch, {}).get(ds, {}).get("summary")
                    if not s:
                        cells.append("n/a")
                    else:
                        d = s[f"arch_r@{K}"] - base_v
                        cells.append(f"{d:+.3f}")
                lines.append(f"| {ds} | {K} | " + " | ".join(cells) + " |")
        lines.append("")

    # Target categories: locomo_temporal and locomo_single_hop
    lines.append("## Target categories (paraphrase concentration)")
    lines.append(
        "These are the categories `per_cue_attribution` identified as "
        "paraphrase-loser hotspots."
    )
    for tc in TARGET_CATEGORIES:
        lines.append(f"\n### `{tc}` (locomo_30q)")
        lines.append("| Arch | n | r@20 | Δ@20 | r@50 | Δ@50 | W/T/L@50 |")
        lines.append("|------|---|-----:|-----:|-----:|-----:|---------:|")
        base_s = (
            all_summaries.get(base_arch, {})
            .get("locomo_30q", {})
            .get("category_breakdown", {})
            .get(tc)
        )
        for arch in arch_names:
            cat = (
                all_summaries.get(arch, {})
                .get("locomo_30q", {})
                .get("category_breakdown", {})
                .get(tc)
            )
            if not cat:
                continue
            d20 = cat["arch_r@20"] - (base_s["arch_r@20"] if base_s else 0)
            d50 = cat["arch_r@50"] - (base_s["arch_r@50"] if base_s else 0)
            lines.append(
                f"| {arch} | {cat['n']} | {cat['arch_r@20']:.3f} | "
                f"{d20:+.3f} | {cat['arch_r@50']:.3f} | {d50:+.3f} | "
                f"{cat['W/T/L_r@50']} |"
            )
    lines.append("")

    # Negative check: other locomo categories (multi_hop, open_domain, adversarial)
    lines.append("## Negative-check categories (don't regress here)")
    other_cats: set[str] = set()
    for arch in arch_names:
        cb = (
            all_summaries.get(arch, {})
            .get("locomo_30q", {})
            .get("category_breakdown", {})
        )
        other_cats.update(c for c in cb if c not in TARGET_CATEGORIES)
    for tc in sorted(other_cats):
        lines.append(f"\n### `{tc}` (locomo_30q)")
        lines.append("| Arch | n | r@50 | Δ vs v2f |")
        lines.append("|------|---|-----:|---------:|")
        base_s = (
            all_summaries.get(base_arch, {})
            .get("locomo_30q", {})
            .get("category_breakdown", {})
            .get(tc)
        )
        for arch in arch_names:
            cat = (
                all_summaries.get(arch, {})
                .get("locomo_30q", {})
                .get("category_breakdown", {})
                .get(tc)
            )
            if not cat:
                continue
            d50 = cat["arch_r@50"] - (base_s["arch_r@50"] if base_s else 0)
            lines.append(
                f"| {arch} | {cat['n']} | {cat['arch_r@50']:.3f} | {d50:+.3f} |"
            )
    lines.append("")

    # Qualitative: cue stats
    lines.append("## Qualitative: cue compliance")
    cue_stats = compile_cue_stats(all_results)
    lines.append("| Arch | total cues | kept | dropped (no verbatim) | empty |")
    lines.append("|------|-----------:|-----:|----------------------:|------:|")
    for arch in arch_names:
        s = cue_stats.get(arch, {})
        lines.append(
            f"| {arch} | {s.get('total_cues_emitted', 0)} | "
            f"{s.get('kept', 0)} | {s.get('dropped_no_verbatim', 0)} | "
            f"{s.get('empty', 0)} |"
        )

    # Paraphrase rate (heuristic)
    lines.append("\n### Heuristic paraphrase rate on locomo_30q cues")
    lines.append(
        "Proxy: cue starts with a question word OR ends with '?' OR shares "
        "≥60% of non-stopword tokens with the question. Lower is better."
    )
    lines.append("| Arch | paraphrase rate |")
    lines.append("|------|----------------:|")
    for arch in arch_names:
        rows = all_results.get(arch, {}).get("locomo_30q", [])
        pr = paraphrase_rate(rows)
        lines.append(f"| {arch} | {pr:.3f} |")
    lines.append("")

    # Sample cues (locomo_temporal)
    lines.append("## Sample cues on `locomo_temporal` (first 3 questions)")
    for arch in arch_names:
        rows = all_results.get(arch, {}).get("locomo_30q", [])
        tcs = [r for r in rows if r["category"] == "locomo_temporal"][:3]
        if not tcs:
            continue
        lines.append(f"\n**{arch}**")
        for r in tcs:
            lines.append(f"- Q: _{r['question']}_")
            for cue in r.get("cues", []):
                lines.append(f"    - CUE: {cue}")
            dropped = [
                o for o in r.get("cue_outcomes", [])
                if not o.get("kept")
            ]
            for o in dropped:
                lines.append(
                    f"    - DROPPED ({o.get('reason')}): {o.get('cue')}"
                )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append(
        "See overall + target-category tables. Decision rules from the "
        "brief: (a) narrow win on target categories without hurting "
        "controls → ship; (b) loss to v2f on locomo → abandon; (c) LLM "
        "ignores Do NOT → try stronger phrasing then stop.\n"
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archs",
        type=str,
        default=None,
        help=(
            "Comma-separated archs. Default: meta_v2f,v2f_anti_paraphrase,"
            "v2f_verbatim_quote,v2f_anti_paraphrase_verbatim"
        ),
    )
    args = parser.parse_args()

    if args.archs:
        arch_names = [a.strip() for a in args.archs.split(",")]
    else:
        arch_names = [
            "meta_v2f",
            "v2f_anti_paraphrase",
            "v2f_verbatim_quote",
            "v2f_anti_paraphrase_verbatim",
        ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries: dict = {}
    all_results: dict = {}

    for ds_name in DATASETS:
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

            out_path = RESULTS_DIR / f"antipara_{arch_name}_{ds_name}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "arch": arch_name,
                        "dataset": ds_name,
                        "summary": summary,
                        "category_breakdown": by_cat,
                        "results": results,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            print(f"  Saved: {out_path}")

            all_summaries.setdefault(arch_name, {})[ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
            }
            all_results.setdefault(arch_name, {})[ds_name] = results

    # Consolidated JSON + MD
    summary_path = RESULTS_DIR / "antipara_cue_study.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved summary JSON: {summary_path}")

    md = render_markdown(all_summaries, all_results, arch_names)
    md_path = RESULTS_DIR / "antipara_cue_study.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved summary MD: {md_path}")

    # Console table
    print("\n" + "=" * 100)
    print("ANTI-PARAPHRASE CUE STUDY SUMMARY")
    print("=" * 100)
    header = (
        f"{'Arch':<32s} {'Dataset':<14s} "
        f"{'base@20':>8s} {'arch@20':>8s} {'d@20':>7s} {'W/T/L@20':>10s} "
        f"{'base@50':>8s} {'arch@50':>8s} {'d@50':>7s} {'W/T/L@50':>10s}"
    )
    print(header)
    print("-" * len(header))
    for arch_name in arch_names:
        for ds_name in DATASETS:
            s = all_summaries.get(arch_name, {}).get(ds_name, {}).get("summary")
            if not s:
                continue
            print(
                f"{arch_name:<32s} {ds_name:<14s} "
                f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
                f"{s['delta_r@20']:>+7.3f} {s['W/T/L_r@20']:>10s} "
                f"{s['baseline_r@50']:>8.3f} {s['arch_r@50']:>8.3f} "
                f"{s['delta_r@50']:>+7.3f} {s['W/T/L_r@50']:>10s}"
            )


if __name__ == "__main__":
    main()
