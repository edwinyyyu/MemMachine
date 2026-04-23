"""Evaluate adaptive cue count variants with fair-backfill on LoCoMo-30
and synthetic-19 at K=20 and K=50.

Usage:
    uv run python acc_eval.py
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from associative_recall import SegmentStore, Segment
from adaptive_cue_count import ARCHITECTURES

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50]

DATASETS = {
    "locomo_30q": {
        "npz": "segments_extended.npz",
        "questions": "questions_extended.json",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "max_questions": 30,
    },
    "synthetic_19q": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "filter": None,
        "max_questions": None,
    },
}


def load_dataset(ds_name: str) -> tuple[SegmentStore, list[dict]]:
    cfg = DATASETS[ds_name]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        questions = json.load(f)
    if cfg["filter"]:
        questions = [q for q in questions if cfg["filter"](q)]
    if cfg["max_questions"]:
        questions = questions[: cfg["max_questions"]]
    return store, questions


def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill_evaluate(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    source_ids: set[int],
    budget: int,
) -> tuple[float, float]:
    seen: set[int] = set()
    arch_unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            arch_unique.append(s)
            seen.add(s.index)

    arch_at_K = arch_unique[:budget]
    arch_indices = {s.index for s in arch_at_K}
    if len(arch_at_K) < budget:
        backfill = [s for s in cosine_segments if s.index not in arch_indices]
        needed = budget - len(arch_at_K)
        arch_at_K = arch_at_K + backfill[:needed]
    arch_at_K = arch_at_K[:budget]

    baseline_at_K = cosine_segments[:budget]

    arch_ids = {s.turn_id for s in arch_at_K}
    baseline_ids = {s.turn_id for s in baseline_at_K}
    return (
        compute_recall(baseline_ids, source_ids),
        compute_recall(arch_ids, source_ids),
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
        "completion_tokens": arch.total_completion_tokens,
        "prompt_tokens": arch.total_prompt_tokens,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "difficulty": result.metadata.get("difficulty", "?"),
        "top1_cosine": round(result.metadata.get("top1_cosine", 0.0), 4),
        "num_cues_requested": result.metadata.get("num_cues_requested", 0),
        "num_cues_parsed": result.metadata.get("num_cues_parsed", 0),
    }

    for K in BUDGETS:
        b_rec, a_rec = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)

    return row


def summarize(results: list[dict], arch_name: str, dataset: str) -> dict:
    n = len(results)
    if n == 0:
        return {"arch": arch_name, "dataset": dataset, "n": 0}

    summary: dict = {"arch": arch_name, "dataset": dataset, "n": n}
    for K in BUDGETS:
        b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in results]
        a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in results]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses
        summary[f"baseline_r@{K}"] = round(b_mean, 4)
        summary[f"arch_r@{K}"] = round(a_mean, 4)
        summary[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
        summary[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"

    summary["avg_total_retrieved"] = round(
        sum(r["total_arch_retrieved"] for r in results) / n, 1
    )
    summary["avg_llm_calls"] = round(
        sum(r["llm_calls"] for r in results) / n, 1
    )
    summary["avg_embed_calls"] = round(
        sum(r["embed_calls"] for r in results) / n, 1
    )
    summary["avg_completion_tokens"] = round(
        sum(r["completion_tokens"] for r in results) / n, 1
    )
    summary["avg_prompt_tokens"] = round(
        sum(r["prompt_tokens"] for r in results) / n, 1
    )
    return summary


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry = {"n": n}
        for K in BUDGETS:
            b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in rs]
            a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in rs]
            b_mean = sum(b_vals) / n
            a_mean = sum(a_vals) / n
            wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
            losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
            ties = n - wins - losses
            entry[f"baseline_r@{K}"] = round(b_mean, 4)
            entry[f"arch_r@{K}"] = round(a_mean, 4)
            entry[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
            entry[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
        out[cat] = entry
    return out


def summarize_by_difficulty(results: list[dict]) -> dict[str, dict]:
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_diff[r.get("difficulty", "?")].append(r)
    out = {}
    for diff, rs in sorted(by_diff.items()):
        n = len(rs)
        entry = {"n": n}
        for K in BUDGETS:
            b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in rs]
            a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in rs]
            b_mean = sum(b_vals) / n
            a_mean = sum(a_vals) / n
            wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
            losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
            ties = n - wins - losses
            entry[f"baseline_r@{K}"] = round(b_mean, 4)
            entry[f"arch_r@{K}"] = round(a_mean, 4)
            entry[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
            entry[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
        out[diff] = entry
    return out


def run_one(
    arch_name: str,
    dataset: str,
    store: SegmentStore,
    questions: list[dict],
) -> tuple[list[dict], dict, dict, dict]:
    print(f"\n{'=' * 70}")
    print(f"{arch_name} | {dataset} | {len(questions)} questions")
    print(f"{'=' * 70}")

    cls = ARCHITECTURES[arch_name]
    arch = cls(store)

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
    by_diff = summarize_by_difficulty(results)

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
        f"embed={summary['avg_embed_calls']:.1f} "
        f"completion_tok={summary['avg_completion_tokens']:.0f}"
    )
    print(f"\n  By difficulty:")
    for d, c in by_diff.items():
        print(
            f"    {d:8s} (n={c['n']}): "
            f"r@20 d={c['delta_r@20']:+.3f} "
            f"r@50 d={c['delta_r@50']:+.3f} "
            f"W/T/L@50={c['W/T/L_r@50']}"
        )
    print(f"\n  By category:")
    for cat, c in by_cat.items():
        print(
            f"    {cat:28s} (n={c['n']}): "
            f"r@20 d={c['delta_r@20']:+.3f} "
            f"r@50 d={c['delta_r@50']:+.3f} "
            f"W/T/L@50={c['W/T/L_r@50']}"
        )

    return results, summary, by_cat, by_diff


def write_md_report(all_data: dict, path: Path) -> None:
    lines = [
        "# Adaptive Cue Count per Query Difficulty",
        "",
        "Modifies the v2f cue generation prompt to request more cues for "
        "queries whose top-1 raw-question cosine is weak (topic_drift "
        "candidates). Fair-backfill evaluation at K=20 and K=50 on "
        "LoCoMo-30 and synthetic-19.",
        "",
        "Difficulty signal: c1 = cosine(question, top-1 segment)",
        "",
        "Variants:",
        "- `meta_v2f_ref`: always 2 cues (v2f reference)",
        "- `adaptive_cue_3_tier`: EASY(c1>=0.5)=2, MEDIUM(0.3<=c1<0.5)=4, "
        "HARD(c1<0.3)=7",
        "- `adaptive_cue_binary`: c1>=0.4 -> 2 cues; else 6 cues",
        "- `always_6cues`: flat 6 cues (control for adaptation vs flat-increase)",
        "",
    ]

    # Difficulty distribution from reference variant
    lines.append("## Difficulty distribution (from meta_v2f_ref)")
    lines.append("")
    for ds in DATASETS:
        ref_blob = all_data.get("meta_v2f_ref", {}).get(ds)
        if not ref_blob:
            continue
        results = ref_blob["results"]
        total = len(results)
        counts: dict[str, int] = defaultdict(int)
        for r in results:
            counts[r["difficulty"]] += 1
        dist = ", ".join(
            f"{k}={v} ({100*v/total:.0f}%)"
            for k, v in sorted(counts.items())
        )
        lines.append(f"- **{ds}** (n={total}): {dist}")
    lines.append("")

    # Main recall table
    lines.append("## Recall (fair-backfill)")
    lines.append("")
    lines.append(
        "| variant | dataset | base@20 | arch@20 | Δ@20 | W/T/L@20 | "
        "base@50 | arch@50 | Δ@50 | W/T/L@50 | avg LLM tok | avg LLM calls |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for arch_name in ARCHITECTURES:
        for ds in DATASETS:
            blob = all_data.get(arch_name, {}).get(ds)
            if not blob:
                continue
            s = blob["summary"]
            lines.append(
                f"| {arch_name} | {ds} | "
                f"{s['baseline_r@20']:.3f} | {s['arch_r@20']:.3f} | "
                f"{s['delta_r@20']:+.3f} | {s['W/T/L_r@20']} | "
                f"{s['baseline_r@50']:.3f} | {s['arch_r@50']:.3f} | "
                f"{s['delta_r@50']:+.3f} | {s['W/T/L_r@50']} | "
                f"{s['avg_completion_tokens']:.0f} | "
                f"{s['avg_llm_calls']:.1f} |"
            )
    lines.append("")

    # Per-difficulty breakdown
    lines.append("## Per-difficulty recall delta")
    lines.append("")
    lines.append("For each variant x dataset, Δr@20 within each difficulty bucket.")
    lines.append("")
    for ds in DATASETS:
        lines.append(f"### {ds}")
        lines.append("")
        lines.append(
            "| variant | EASY Δ@20 | MEDIUM Δ@20 | HARD Δ@20 | "
            "FLAT Δ@20 | V2F Δ@20 |"
        )
        lines.append("|---|---|---|---|---|---|")
        for arch_name in ARCHITECTURES:
            blob = all_data.get(arch_name, {}).get(ds)
            if not blob:
                continue
            by_diff = blob["by_difficulty"]
            cells = []
            for d in ["EASY", "MEDIUM", "HARD", "FLAT", "V2F"]:
                if d in by_diff:
                    cells.append(
                        f"{by_diff[d]['delta_r@20']:+.3f} (n={by_diff[d]['n']})"
                    )
                else:
                    cells.append("—")
            lines.append(f"| {arch_name} | " + " | ".join(cells) + " |")
        lines.append("")

    # Per-category LoCoMo
    lines.append("## Per-category (LoCoMo-30) Δr@20")
    lines.append("")
    cats_seen: set[str] = set()
    for arch_name in ARCHITECTURES:
        blob = all_data.get(arch_name, {}).get("locomo_30q")
        if not blob:
            continue
        cats_seen.update(blob["by_category"].keys())
    cats_sorted = sorted(cats_seen)
    header = "| variant | " + " | ".join(cats_sorted) + " |"
    sep = "|---|" + "---|" * len(cats_sorted)
    lines.append(header)
    lines.append(sep)
    for arch_name in ARCHITECTURES:
        blob = all_data.get(arch_name, {}).get("locomo_30q")
        if not blob:
            continue
        by_cat = blob["by_category"]
        cells = []
        for cat in cats_sorted:
            if cat in by_cat:
                cells.append(
                    f"{by_cat[cat]['delta_r@20']:+.3f} (n={by_cat[cat]['n']})"
                )
            else:
                cells.append("—")
        lines.append(f"| {arch_name} | " + " | ".join(cells) + " |")
    lines.append("")

    # Cost
    lines.append("## Cost per query (mean)")
    lines.append("")
    lines.append(
        "| variant | dataset | LLM calls | completion tok | prompt tok | "
        "embed calls |"
    )
    lines.append("|---|---|---|---|---|---|")
    for arch_name in ARCHITECTURES:
        for ds in DATASETS:
            blob = all_data.get(arch_name, {}).get(ds)
            if not blob:
                continue
            s = blob["summary"]
            lines.append(
                f"| {arch_name} | {ds} | {s['avg_llm_calls']:.1f} | "
                f"{s['avg_completion_tokens']:.0f} | "
                f"{s['avg_prompt_tokens']:.0f} | "
                f"{s['avg_embed_calls']:.1f} |"
            )
    lines.append("")

    # Verdict — computed mechanically
    lines.append("## Verdict")
    lines.append("")
    avg_delta = {}
    for arch_name in ARCHITECTURES:
        deltas = []
        for ds in DATASETS:
            blob = all_data.get(arch_name, {}).get(ds)
            if blob:
                deltas.append(blob["summary"]["delta_r@20"])
                deltas.append(blob["summary"]["delta_r@50"])
        if deltas:
            avg_delta[arch_name] = sum(deltas) / len(deltas)

    ref = avg_delta.get("meta_v2f_ref", 0.0)
    three_tier = avg_delta.get("adaptive_cue_3_tier", 0.0)
    binary = avg_delta.get("adaptive_cue_binary", 0.0)
    flat6 = avg_delta.get("always_6cues", 0.0)
    lines.append(
        f"- Mean Δrecall across K=20/50 × datasets: "
        f"v2f={ref:+.3f}, 3_tier={three_tier:+.3f}, "
        f"binary={binary:+.3f}, always_6={flat6:+.3f}"
    )
    best = max(avg_delta, key=lambda k: avg_delta[k]) if avg_delta else "?"
    adaptive_best = max(three_tier, binary)
    if (
        adaptive_best > ref + 0.005
        and adaptive_best > flat6 + 0.005
    ):
        verdict = (
            "SHIP (narrow): adaptive variant beats both v2f and always_6. "
            "Adaptation captures benefit flat cue-bumps miss."
        )
    elif flat6 > ref + 0.005 and flat6 >= adaptive_best - 0.005:
        verdict = (
            "NARROW: gains come from 'more cues always', not adaptation. "
            "Replace v2f default num_cues with 6 rather than shipping "
            "adaptive logic."
        )
    elif adaptive_best <= ref + 0.005 and flat6 <= ref + 0.005:
        verdict = (
            "ABANDON: extra cues add redundancy without recall gain. "
            "v2f's 2-cue default is at the Pareto frontier."
        )
    else:
        verdict = (
            f"MIXED: best variant={best}. Inspect per-difficulty and "
            f"per-category breakdown; may be ship-worthy for specific categories."
        )
    lines.append(f"- **{verdict}**")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `results/adaptive_cue_count.json` — full raw")
    lines.append("- `results/adaptive_cue_count.md` — this report")
    lines.append("- Caches: `cache/adaptive_cue_embedding_cache.json`, "
                 "`cache/adaptive_cue_llm_cache.json`")

    path.write_text("\n".join(lines))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_data: dict = {}

    for ds_name in DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        for arch_name in ARCHITECTURES:
            results, summary, by_cat, by_diff = run_one(
                arch_name, ds_name, store, questions
            )
            all_data.setdefault(arch_name, {})[ds_name] = {
                "summary": summary,
                "by_category": by_cat,
                "by_difficulty": by_diff,
                "results": results,
            }

    # Save JSON
    raw_path = RESULTS_DIR / "adaptive_cue_count.json"
    with open(raw_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Write MD report
    md_path = RESULTS_DIR / "adaptive_cue_count.md"
    write_md_report(all_data, md_path)
    print(f"Saved: {md_path}")

    # Final table
    print("\n" + "=" * 100)
    print("ADAPTIVE CUE COUNT SUMMARY")
    print("=" * 100)
    header = (
        f"{'Arch':<22s} {'Dataset':<14s} "
        f"{'base@20':>8s} {'arch@20':>8s} {'d@20':>7s} "
        f"{'base@50':>8s} {'arch@50':>8s} {'d@50':>7s} "
        f"{'LLMtok':>8s}"
    )
    print(header)
    print("-" * len(header))
    for arch_name in ARCHITECTURES:
        for ds_name in DATASETS:
            blob = all_data.get(arch_name, {}).get(ds_name)
            if not blob:
                continue
            s = blob["summary"]
            print(
                f"{arch_name:<22s} {ds_name:<14s} "
                f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
                f"{s['delta_r@20']:>+7.3f} "
                f"{s['baseline_r@50']:>8.3f} {s['arch_r@50']:>8.3f} "
                f"{s['delta_r@50']:>+7.3f} "
                f"{s['avg_completion_tokens']:>8.0f}"
            )


if __name__ == "__main__":
    main()
