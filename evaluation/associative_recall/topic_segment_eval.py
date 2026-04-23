"""Evaluate topic-segment hierarchical retrieval vs v2f baseline.

Loads pre-built topic segments from topic_segment.py, constructs a
TopicSegRetriever, and runs fair-backfill evaluation on LoCoMo-30 and
synthetic_19q at K=20 and K=50. Compares against v2f (MetaV2f) and
cosine-only baseline.

Usage:
    uv run python topic_segment_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from associative_recall import Segment, SegmentStore
from best_shot import MetaV2f
from topic_segment import (
    TopicSegBuilder,
    TopicSegRetriever,
    TopicSegStore,
    load_segmentation,
    save_segmentation,
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
        "conv_filter_prefix": "locomo",
    },
    "synthetic_19q": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "filter": None,
        "max_questions": None,
        "conv_filter_prefix": None,
    },
}


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------
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


def _conv_ids(store: SegmentStore, prefix: str | None) -> list[str]:
    cids = sorted({s.conversation_id for s in store.segments})
    if prefix:
        cids = [c for c in cids if c.startswith(prefix)]
    return cids


def ensure_segmentation(
    ds_name: str, store: SegmentStore, variant: str,
    chunk_size: int = 10, window_size: int = 40,
):
    """Build segmentation (cached) and return the SegmentationResult."""
    cfg = DATASETS[ds_name]
    prefix = cfg["conv_filter_prefix"]
    if variant == "fixed":
        suffix = f"fixed_n{chunk_size}"
    else:
        suffix = f"llm_w{window_size}"
    seg_path = RESULTS_DIR / f"topic_segments_{ds_name}_{suffix}.json"
    if seg_path.exists():
        return load_segmentation(seg_path)

    print(f"  Building {variant} segmentation for {ds_name}...")
    conv_ids = _conv_ids(store, prefix)
    builder = TopicSegBuilder(store)
    if variant == "fixed":
        result = builder.build_fixed(conv_ids, chunk_size=chunk_size)
    else:
        result = builder.build_llm(conv_ids, window_size=window_size)
    builder.save_caches()
    save_segmentation(result, seg_path)
    return result


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
        query_emb, top_k=max_K, conversation_id=conv_id,
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
    }
    for K in BUDGETS:
        b_rec, a_rec = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K,
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
        sum(r["total_arch_retrieved"] for r in results) / n, 1,
    )
    summary["avg_llm_calls"] = round(
        sum(r["llm_calls"] for r in results) / n, 1,
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


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------
def run_arch(
    name: str, arch, dataset: str, questions: list[dict],
) -> tuple[list[dict], dict, dict]:
    print(f"\n{'=' * 70}")
    print(f"{name} | {dataset} | {len(questions)} questions")
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

    summary = summarize(results, name, dataset)
    by_cat = summarize_by_category(results)

    print(f"\n--- {name} on {dataset} ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: baseline={summary[f'baseline_r@{K}']:.3f} "
            f"arch={summary[f'arch_r@{K}']:.3f} "
            f"delta={summary[f'delta_r@{K}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    print(
        f"  avg retrieved={summary['avg_total_retrieved']:.0f} "
        f"llm={summary['avg_llm_calls']:.1f}"
    )
    return results, summary, by_cat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}
    segmentation_stats: dict = {}

    # Variants to test
    fixed_chunk_sizes = [10, 15]
    llm_window = 40

    for ds_name in DATASETS:
        store, questions = load_dataset(ds_name)
        print(f"\n\n========= Dataset {ds_name}: {len(questions)} questions "
              f"across {len(_conv_ids(store, DATASETS[ds_name]['conv_filter_prefix']))} "
              f"conversations =========")

        # ----- Baselines -----
        # v2f baseline: use MetaV2f
        v2f = MetaV2f(store)
        res_v2f, summ_v2f, cat_v2f = run_arch("v2f", v2f, ds_name, questions)
        all_results.setdefault("v2f", {})[ds_name] = {
            "summary": summ_v2f,
            "category_breakdown": cat_v2f,
            "results": res_v2f,
        }

        # ----- Topic segment variants -----
        # Fixed-size
        for n in fixed_chunk_sizes:
            seg_result = ensure_segmentation(
                ds_name, store, "fixed", chunk_size=n,
            )
            total_segs = sum(len(v) for v in seg_result.conversations.values())
            avg_turns = (
                sum(len(s.turn_ids) for v in seg_result.conversations.values()
                    for s in v) / max(total_segs, 1)
            )
            segmentation_stats[f"{ds_name}_fixed_n{n}"] = {
                "variant": f"fixed_n{n}",
                "total_segments": total_segs,
                "avg_turns_per_segment": round(avg_turns, 2),
                "num_conversations": len(seg_result.conversations),
            }
            topic_store = TopicSegStore(seg_result, store)
            arch = TopicSegRetriever(
                store, topic_store,
                top_m_summaries=3, top_kt_turns=50, alpha=1.0,
            )
            name = f"topic_seg_fixed_n{n}"
            res, summ, cat = run_arch(name, arch, ds_name, questions)
            all_results.setdefault(name, {})[ds_name] = {
                "summary": summ,
                "category_breakdown": cat,
                "results": res,
            }

        # LLM-driven
        seg_result = ensure_segmentation(
            ds_name, store, "llm", window_size=llm_window,
        )
        total_segs = sum(len(v) for v in seg_result.conversations.values())
        avg_turns = (
            sum(len(s.turn_ids) for v in seg_result.conversations.values()
                for s in v) / max(total_segs, 1)
        )
        segmentation_stats[f"{ds_name}_llm_w{llm_window}"] = {
            "variant": f"llm_w{llm_window}",
            "total_segments": total_segs,
            "avg_turns_per_segment": round(avg_turns, 2),
            "num_conversations": len(seg_result.conversations),
        }
        topic_store = TopicSegStore(seg_result, store)
        for top_m in [3, 5]:
            arch = TopicSegRetriever(
                store, topic_store,
                top_m_summaries=top_m, top_kt_turns=50, alpha=1.0,
            )
            name = f"topic_seg_llm_m{top_m}"
            res, summ, cat = run_arch(name, arch, ds_name, questions)
            all_results.setdefault(name, {})[ds_name] = {
                "summary": summ,
                "category_breakdown": cat,
                "results": res,
            }

    # Save results
    out_json = RESULTS_DIR / "topic_segmentation.json"
    out_data = {
        "segmentation_stats": segmentation_stats,
        "results": all_results,
    }
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")

    # Build markdown report
    build_report(out_data, RESULTS_DIR / "topic_segmentation.md")


def build_report(data: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Topic-Segment Hierarchical Retrieval — Eval Report\n")

    # Segmentation stats
    lines.append("## Segmentation statistics\n")
    lines.append(
        "| Dataset/Variant | #Convs | Total segments | Avg turns/seg |"
    )
    lines.append("|---|---|---|---|")
    for k, v in data["segmentation_stats"].items():
        lines.append(
            f"| {k} | {v['num_conversations']} | "
            f"{v['total_segments']} | {v['avg_turns_per_segment']} |"
        )
    lines.append("")

    # Recall table
    lines.append("## Recall table (fair-backfill)\n")
    lines.append(
        "| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | W/T/L@50 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    arch_order = sorted(data["results"].keys())
    for arch in arch_order:
        for ds in data["results"][arch]:
            s = data["results"][arch][ds]["summary"]
            lines.append(
                f"| {arch} | {ds} | "
                f"{s['baseline_r@20']:.4f} | {s['arch_r@20']:.4f} | "
                f"{s['delta_r@20']:+.4f} | "
                f"{s['baseline_r@50']:.4f} | {s['arch_r@50']:.4f} | "
                f"{s['delta_r@50']:+.4f} | "
                f"{s['W/T/L_r@50']} |"
            )
    lines.append("")

    # Category breakdown: for best topic_seg variant on locomo
    lines.append("## Category deltas (locomo_30q, best variant vs baseline)\n")
    best_variant = _pick_best_variant(data, "locomo_30q")
    lines.append(f"Best variant: **{best_variant}**\n")
    if best_variant in data["results"] and "locomo_30q" in data["results"][best_variant]:
        cb = data["results"][best_variant]["locomo_30q"]["category_breakdown"]
        rows = sorted(
            cb.items(),
            key=lambda kv: kv[1]["delta_r@50"],
            reverse=True,
        )
        lines.append(
            "| Category | n | Δ@20 | Δ@50 | W/T/L@50 |"
        )
        lines.append("|---|---|---|---|---|")
        for cat, entry in rows:
            lines.append(
                f"| {cat} | {entry['n']} | "
                f"{entry['delta_r@20']:+.4f} | {entry['delta_r@50']:+.4f} | "
                f"{entry['W/T/L_r@50']} |"
            )
        lines.append("")

    # Verdict
    lines.append("## Verdict\n")
    verdict = _assess_verdict(data)
    lines.append(verdict)
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved report: {out_path}")


def _pick_best_variant(data: dict, ds: str) -> str:
    """Return the topic_seg* variant with highest r@50 on given dataset."""
    best_name = None
    best_r50 = -1.0
    for arch, per_ds in data["results"].items():
        if not arch.startswith("topic_seg"):
            continue
        if ds not in per_ds:
            continue
        r50 = per_ds[ds]["summary"].get("arch_r@50", 0)
        if r50 > best_r50:
            best_r50 = r50
            best_name = arch
    return best_name or "(none)"


def _assess_verdict(data: dict) -> str:
    # v2f baseline
    v2f_locomo_r50 = data["results"].get("v2f", {}).get(
        "locomo_30q", {}).get("summary", {}).get("arch_r@50", 0.0)
    # Best topic_seg variant
    best = _pick_best_variant(data, "locomo_30q")
    best_locomo_r50 = data["results"].get(best, {}).get(
        "locomo_30q", {}).get("summary", {}).get("arch_r@50", 0.0)
    delta = best_locomo_r50 - v2f_locomo_r50
    line = (
        f"- v2f locomo_30q r@50 = {v2f_locomo_r50:.4f}\n"
        f"- best topic_seg variant ({best}) locomo_30q r@50 = "
        f"{best_locomo_r50:.4f}\n"
        f"- delta vs v2f: {delta:+.4f}\n"
    )
    if delta > 0.01:
        line += "\n**SHIP** — topic segmentation beats v2f on locomo r@50."
    elif delta > -0.005:
        line += "\n**MIXED** — topic segmentation ~matches v2f. Evaluate W/T/L and category deltas."
    else:
        line += "\n**ABANDON** — topic segmentation does not beat v2f on locomo r@50."
    return line


if __name__ == "__main__":
    main()
