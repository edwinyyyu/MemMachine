"""Evaluate the LLM-weighted multi-channel retrieval architecture.

Runs 3 variants + baselines on LoCoMo-30 and synthetic-19 at K=20, K=50,
using fair-backfill (arch's segments + cosine top-K backfill, truncated to K).

Variants:
  - multich_llm_weighted (per-query LLM routing)
  - multich_uniform (uniform weights over all channels — control)
  - multich_binary (LLM binary routing)
  - meta_v2f (reference — for consistency with fair-backfill baseline)

Outputs:
  results/multichannel_weighted.json (raw)
  results/multichannel_weighted.md (human-readable report)

Usage:
    uv run python multich_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from associative_recall import SegmentStore, Segment
from best_shot import MetaV2f
from multichannel_weighted import (
    CHANNEL_DESCRIPTIONS,
    CHANNEL_NAMES,
    VARIANTS,
    build_variant,
)

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
        qs = json.load(f)
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    if cfg["max_questions"]:
        qs = qs[: cfg["max_questions"]]
    return store, qs


def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    budget: int,
) -> list[Segment]:
    seen: set[int] = set()
    unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            unique.append(s)
            seen.add(s.index)
    at_K = unique[:budget]
    have = {s.index for s in at_K}
    if len(at_K) < budget:
        for s in cosine_segments:
            if s.index in have:
                continue
            at_K.append(s)
            have.add(s.index)
            if len(at_K) >= budget:
                break
    return at_K[:budget]


def evaluate_question(arch, question: dict) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedupe arch segments preserving order
    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            arch_segments.append(seg)
            seen.add(seg.index)

    # Cosine top-K baseline (single call)
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
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "metadata": result.metadata,
    }

    for K in BUDGETS:
        arch_at_K = fair_backfill(arch_segments, cosine_segments, K)
        baseline_at_K = cosine_segments[:K]
        arch_ids = {s.turn_id for s in arch_at_K}
        base_ids = {s.turn_id for s in baseline_at_K}
        b_rec = compute_recall(base_ids, source_ids)
        a_rec = compute_recall(arch_ids, source_ids)
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
        sum(r["llm_calls"] for r in results) / n, 2
    )
    summary["avg_embed_calls"] = round(
        sum(r["embed_calls"] for r in results) / n, 2
    )
    summary["avg_time_s"] = round(
        sum(r["time_s"] for r in results) / n, 2
    )
    return summary


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out: dict[str, dict] = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry: dict = {"n": n}
        for K in BUDGETS:
            b = sum(r["fair_backfill"][f"baseline_r@{K}"] for r in rs) / n
            a = sum(r["fair_backfill"][f"arch_r@{K}"] for r in rs) / n
            entry[f"baseline_r@{K}"] = round(b, 4)
            entry[f"arch_r@{K}"] = round(a, 4)
            entry[f"delta_r@{K}"] = round(a - b, 4)
        out[cat] = entry
    return out


def run_arch_on_dataset(
    arch_name: str, store: SegmentStore, questions: list[dict]
) -> tuple[list[dict], dict, dict]:
    print(f"\n{'=' * 70}\n{arch_name} | {len(questions)} questions\n{'=' * 70}")
    if arch_name == "meta_v2f":
        arch = MetaV2f(store)
    elif arch_name in VARIANTS:
        arch = build_variant(arch_name, store)
    else:
        raise KeyError(arch_name)

    rows: list[dict] = []
    for i, q in enumerate(questions):
        q_short = q["question"][:55]
        print(
            f"  [{i+1}/{len(questions)}] {q.get('category', '?')}: {q_short}",
            flush=True,
        )
        try:
            row = evaluate_question(arch, q)
            rows.append(row)
        except Exception as e:
            print(f"  ERROR on question {i}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            arch.save_caches()
    arch.save_caches()

    summary = summarize(rows, arch_name, "?")
    by_cat = summarize_by_category(rows)
    print(f"\n--- {arch_name} summary ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: base={summary[f'baseline_r@{K}']:.4f} "
            f"arch={summary[f'arch_r@{K}']:.4f} "
            f"delta={summary[f'delta_r@{K}']:+.4f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    print(
        f"  avg llm calls={summary['avg_llm_calls']:.2f} "
        f"embed={summary['avg_embed_calls']:.2f} "
        f"time={summary['avg_time_s']:.2f}s"
    )
    return rows, summary, by_cat


# ---------------------------------------------------------------------------
# Weight pattern analysis
# ---------------------------------------------------------------------------
def analyze_weight_patterns(
    rows: list[dict],
) -> dict:
    """Aggregate LLM-chosen weights across queries."""
    weights_per_channel: dict[str, list[float]] = {
        ch: [] for ch in CHANNEL_NAMES
    }
    category_weights: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {ch: [] for ch in CHANNEL_NAMES}
    )
    n_zero: dict[str, int] = {ch: 0 for ch in CHANNEL_NAMES}
    n_total = 0
    sample_queries = []

    for r in rows:
        meta = r.get("metadata", {})
        w = meta.get("weights", {})
        if not w:
            continue
        n_total += 1
        cat = r.get("category", "unknown")
        for ch in CHANNEL_NAMES:
            v = float(w.get(ch, 0.0))
            weights_per_channel[ch].append(v)
            category_weights[cat][ch].append(v)
            if v <= 0.001:
                n_zero[ch] += 1
        if len(sample_queries) < 5:
            sample_queries.append(
                {
                    "question": r.get("question", "")[:120],
                    "category": cat,
                    "weights": {ch: round(float(w.get(ch, 0.0)), 2)
                                 for ch in CHANNEL_NAMES},
                    "reasoning": meta.get("reasoning", "")[:200],
                    "channels_executed": meta.get("channels_executed", []),
                }
            )

    avg = {
        ch: round(sum(v) / max(len(v), 1), 3)
        for ch, v in weights_per_channel.items()
    }
    zero_rate = {
        ch: round(n_zero[ch] / max(n_total, 1), 3)
        for ch in CHANNEL_NAMES
    }
    # Per-category averages
    cat_avg: dict[str, dict[str, float]] = {}
    for cat, w_map in category_weights.items():
        cat_avg[cat] = {
            ch: round(sum(v) / max(len(v), 1), 3) for ch, v in w_map.items()
        }

    return {
        "n": n_total,
        "avg_weight_per_channel": avg,
        "zero_rate_per_channel": zero_rate,
        "per_category_avg_weights": cat_avg,
        "sample_queries": sample_queries,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_report(all_data: dict) -> str:
    lines: list[str] = []
    lines.append("# LLM-Weighted Multi-Channel Retrieval\n")
    lines.append(
        "This study tests whether letting the LLM choose per-query channel "
        "weights (an **LLM conductor**) outperforms both the baseline (cosine "
        "only or v2f) and a uniform-weight multi-channel control.\n"
    )
    lines.append("\n## Channels\n")
    for ch in CHANNEL_NAMES:
        lines.append(f"- **{ch}** — {CHANNEL_DESCRIPTIONS[ch]}")
    lines.append("")

    lines.append("\n## Recall Matrix (fair-backfill)\n")
    header = (
        "| Architecture | Dataset | base r@20 | arch r@20 | Δ@20 | base r@50 "
        "| arch r@50 | Δ@50 | avg LLM | avg embed |"
    )
    sep = "|" + ("---|" * 10)
    lines.append(header)
    lines.append(sep)
    for arch_name in ["meta_v2f"] + list(VARIANTS):
        for ds in DATASETS:
            s = all_data.get(arch_name, {}).get(ds, {}).get("summary")
            if not s:
                continue
            lines.append(
                f"| {arch_name} | {ds} | "
                f"{s['baseline_r@20']:.4f} | {s['arch_r@20']:.4f} | "
                f"{s['delta_r@20']:+.4f} | "
                f"{s['baseline_r@50']:.4f} | {s['arch_r@50']:.4f} | "
                f"{s['delta_r@50']:+.4f} | "
                f"{s['avg_llm_calls']:.1f} | {s['avg_embed_calls']:.1f} |"
            )
    lines.append("")

    # Weight patterns for multich_llm_weighted
    lines.append("\n## Weight patterns (multich_llm_weighted)\n")
    for ds in DATASETS:
        pat = all_data.get("multich_llm_weighted", {}).get(ds, {}).get(
            "weight_patterns"
        )
        if not pat:
            continue
        lines.append(f"### {ds}\n")
        lines.append(
            f"n={pat['n']}. Average LLM-chosen weight per channel (0.0 = "
            f"never engaged, 1.0 = always fully engaged):\n"
        )
        lines.append("| channel | avg weight | zero-rate |")
        lines.append("|---|---|---|")
        for ch in CHANNEL_NAMES:
            lines.append(
                f"| {ch} | {pat['avg_weight_per_channel'][ch]:.3f} "
                f"| {pat['zero_rate_per_channel'][ch]:.2f} |"
            )
        lines.append("")
        lines.append("Sample queries with LLM-chosen weights:\n")
        for q in pat.get("sample_queries", [])[:3]:
            lines.append(f"- **[{q['category']}]** {q['question']}")
            non_zero = {k: v for k, v in q["weights"].items() if v > 0.01}
            lines.append(
                f"  - Weights (non-zero): {non_zero}"
            )
            if q.get("reasoning"):
                lines.append(f"  - Reasoning: {q['reasoning']}")
        lines.append("")

    # Cost comparison
    lines.append("\n## Cost Comparison (avg LLM calls per query)\n")
    lines.append(
        "| Architecture | LoCoMo-30 | Synthetic-19 |"
    )
    lines.append("|---|---|---|")
    for arch_name in ["meta_v2f"] + list(VARIANTS):
        row = [arch_name]
        for ds in DATASETS:
            s = all_data.get(arch_name, {}).get(ds, {}).get("summary")
            if s:
                row.append(f"{s['avg_llm_calls']:.2f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Verdict
    lines.append("\n## Verdict\n")
    for ds in DATASETS:
        lines.append(f"### {ds}\n")
        for K in BUDGETS:
            ref = all_data.get("meta_v2f", {}).get(ds, {}).get(
                "summary", {}
            ).get(f"arch_r@{K}", 0.0)
            llm_s = all_data.get("multich_llm_weighted", {}).get(ds, {}).get(
                "summary", {}
            ).get(f"arch_r@{K}", 0.0)
            uni = all_data.get("multich_uniform", {}).get(ds, {}).get(
                "summary", {}
            ).get(f"arch_r@{K}", 0.0)
            binr = all_data.get("multich_binary", {}).get(ds, {}).get(
                "summary", {}
            ).get(f"arch_r@{K}", 0.0)
            lines.append(
                f"- **K={K}**: meta_v2f={ref:.4f}, "
                f"llm_weighted={llm_s:.4f} ({llm_s - ref:+.4f}), "
                f"uniform={uni:.4f} ({uni - ref:+.4f}), "
                f"binary={binr:.4f} ({binr - ref:+.4f})"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_data: dict = {}

    for ds_name in DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\n\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        for arch_name in ["meta_v2f"] + list(VARIANTS):
            rows, summary, by_cat = run_arch_on_dataset(
                arch_name, store, questions
            )
            entry = {
                "arch": arch_name,
                "dataset": ds_name,
                "summary": summary,
                "category_breakdown": by_cat,
                "results": rows,
            }
            if arch_name == "multich_llm_weighted":
                entry["weight_patterns"] = analyze_weight_patterns(rows)
            all_data.setdefault(arch_name, {})[ds_name] = entry

    # Save raw JSON (strip large 'results' per arch to keep file manageable,
    # but keep per-question metadata because the user wants to inspect
    # weight patterns).
    raw_out = RESULTS_DIR / "multichannel_weighted.json"
    with open(raw_out, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nRaw saved: {raw_out}")

    report = render_report(all_data)
    md_out = RESULTS_DIR / "multichannel_weighted.md"
    with open(md_out, "w") as f:
        f.write(report)
    print(f"Report saved: {md_out}")


if __name__ == "__main__":
    main()
