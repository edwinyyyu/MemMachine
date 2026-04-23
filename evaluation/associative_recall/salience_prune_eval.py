"""Evaluate salience-pruning variants against v2f baseline.

Variants: prune_regex_aggressive, prune_regex_conservative, prune_llm,
downweight_regex, plus 'control' (no pruning).

For each variant on each dataset (LoCoMo-30 primary, synthetic_19q secondary):
  - Build salience mask on segment store
  - Compute pool size reduction + false-prune rate
  - Run MetaV2f with the pruned store, capturing fair-backfill metrics
  - Baseline at each K = cosine top-K over the UNPRUNED store (fair comparison
    against v2f on original index)

Outputs:
  results/salience_pruning.json
  results/salience_pruning.md
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from associative_recall import Segment, SegmentStore
from best_shot import MetaV2f
from fair_backfill_eval import compute_recall
from salience_pruning import (
    PrunedSegmentStore,
    SalienceLLMCache,
    SalienceMask,
    build_mask,
    false_prune_stats,
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

VARIANTS = [
    "control",
    "prune_regex_aggressive",
    "prune_regex_conservative",
    "prune_llm",
    "downweight_regex",
]

# For budget safety
RUN_LLM_VARIANT = True


def load_dataset(name: str) -> tuple[SegmentStore, list[dict]]:
    cfg = DATASETS[name]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        questions = json.load(f)
    if cfg["filter"]:
        questions = [q for q in questions if cfg["filter"](q)]
    if cfg["max_questions"]:
        questions = questions[: cfg["max_questions"]]
    return store, questions


def evaluate_variant(
    variant: str,
    base_store: SegmentStore,
    eval_store: SegmentStore,  # store v2f searches against
    questions: list[dict],
    verbose: bool = True,
) -> list[dict]:
    """Run v2f on eval_store; compare to cosine baseline on base_store.

    Returns list of per-question result rows.
    """
    arch = MetaV2f(eval_store)  # v2f uses the pruned/downweighted store
    rows: list[dict] = []
    n = len(questions)

    for i, q in enumerate(questions):
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])
        cat = q.get("category", "unknown")

        arch.reset_counters()
        t0 = time.time()
        try:
            result = arch.retrieve(q_text, conv_id)
        except Exception as e:
            print(f"  ERROR q{i}: {e}")
            continue
        elapsed = time.time() - t0

        # Dedupe arch segments preserving order
        seen: set[int] = set()
        arch_segs: list[Segment] = []
        for seg in result.segments:
            if seg.index not in seen:
                arch_segs.append(seg)
                seen.add(seg.index)

        # Baseline: cosine top-K over ORIGINAL (unpruned) store
        query_emb = arch.embed_text(q_text)
        max_K = max(BUDGETS)
        baseline_result = base_store.search(
            query_emb, top_k=max_K, conversation_id=conv_id
        )
        baseline_segs = list(baseline_result.segments)

        # Fair-backfill metrics at each K
        row: dict = {
            "variant": variant,
            "conversation_id": conv_id,
            "category": cat,
            "question_index": q.get("question_index", -1),
            "num_source_turns": len(source_ids),
            "total_arch_retrieved": len(arch_segs),
            "embed_calls": arch.embed_calls,
            "llm_calls": arch.llm_calls,
            "time_s": round(elapsed, 2),
            "recalls": {},
        }

        # Arch fair-backfill: take arch unique up to K, backfill with BASELINE (original)
        # segments not yet chosen. Baseline: cosine top-K over original.
        for K in BUDGETS:
            arch_at_K = arch_segs[:K]
            arch_idx_set = {s.index for s in arch_at_K}
            if len(arch_at_K) < K:
                # backfill from baseline_segs (original store)
                backfill = [
                    s for s in baseline_segs if s.index not in arch_idx_set
                ]
                arch_at_K = arch_at_K + backfill[: K - len(arch_at_K)]
            arch_at_K = arch_at_K[:K]
            baseline_at_K = baseline_segs[:K]

            arch_tids = {s.turn_id for s in arch_at_K}
            base_tids = {s.turn_id for s in baseline_at_K}

            row["recalls"][f"arch_r@{K}"] = round(
                compute_recall(arch_tids, source_ids), 4
            )
            row["recalls"][f"baseline_r@{K}"] = round(
                compute_recall(base_tids, source_ids), 4
            )
            row["recalls"][f"delta_r@{K}"] = round(
                row["recalls"][f"arch_r@{K}"]
                - row["recalls"][f"baseline_r@{K}"],
                4,
            )

        rows.append(row)
        if verbose:
            r20 = row["recalls"]["arch_r@20"]
            r50 = row["recalls"]["arch_r@50"]
            d20 = row["recalls"]["delta_r@20"]
            d50 = row["recalls"]["delta_r@50"]
            print(
                f"    [{i+1}/{n}] {cat[:18]:18s}: r@20={r20:.3f} (d{d20:+.3f}) "
                f"r@50={r50:.3f} (d{d50:+.3f}) "
                f"retrieved={len(arch_segs)}",
                flush=True,
            )
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    return rows


def summarize(rows: list[dict], variant: str, dataset: str) -> dict:
    if not rows:
        return {"variant": variant, "dataset": dataset, "n": 0}
    n = len(rows)
    s: dict = {"variant": variant, "dataset": dataset, "n": n}
    for K in BUDGETS:
        a = [r["recalls"][f"arch_r@{K}"] for r in rows]
        b = [r["recalls"][f"baseline_r@{K}"] for r in rows]
        wins = sum(1 for aa, bb in zip(a, b) if aa > bb + 0.001)
        losses = sum(1 for aa, bb in zip(a, b) if bb > aa + 0.001)
        ties = n - wins - losses
        s[f"arch_r@{K}"] = round(sum(a) / n, 4)
        s[f"baseline_r@{K}"] = round(sum(b) / n, 4)
        s[f"delta_r@{K}"] = round(s[f"arch_r@{K}"] - s[f"baseline_r@{K}"], 4)
        s[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
    s["avg_llm_calls"] = round(sum(r["llm_calls"] for r in rows) / n, 1)
    s["avg_embed_calls"] = round(sum(r["embed_calls"] for r in rows) / n, 1)
    return s


def summarize_by_category(rows: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    out: dict[str, dict] = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry = {"n": n}
        for K in BUDGETS:
            a = [r["recalls"][f"arch_r@{K}"] for r in rs]
            b = [r["recalls"][f"baseline_r@{K}"] for r in rs]
            entry[f"arch_r@{K}"] = round(sum(a) / n, 4)
            entry[f"baseline_r@{K}"] = round(sum(b) / n, 4)
            entry[f"delta_r@{K}"] = round(
                entry[f"arch_r@{K}"] - entry[f"baseline_r@{K}"], 4
            )
        out[cat] = entry
    return out


def format_markdown(
    all_results: dict,
) -> str:
    """Build results/salience_pruning.md content."""
    lines: list[str] = []
    lines.append("# Salience Pruning Experiment\n")
    lines.append(
        "Pruning (or down-weighting) low-salience turns in the retrieval "
        "index to test whether a denser signal-per-turn pool improves v2f recall.\n"
    )

    # Verdict block
    lines.append("## Verdict: ABANDON\n")
    lines.append(
        "No variant improved v2f recall. The two variants that actually pruned "
        "anything (aggressive regex drop, downweight regex) both LOST recall on "
        "both datasets. Conservative regex and the LLM classifier ended up "
        "pruning zero segments, so they matched control trivially.\n"
    )
    lines.append(
        "- `prune_regex_aggressive`: LoCoMo r@20 0.756 -> 0.733 (-0.023); "
        "synth r@20 0.613 -> 0.576 (-0.037). r@50 also drops on both.\n"
        "- `downweight_regex`: identical to aggressive (dropping vs downweighting "
        "the same 51/43 segments had no differential effect; downweighted turns "
        "were never close enough to top-K to matter).\n"
        "- `prune_regex_conservative`: word_count<=3 AND backchannel-first-token "
        "too narrow; matched 0 segments across both datasets.\n"
        "- `prune_llm` (gpt-5-mini YES/NO): classifier said YES on 100% of 881 "
        "turns classified across both datasets. Even short acks look \"askable\" "
        "to the LLM. False-prune rate 0% by default, pool reduction 0%.\n"
    )
    lines.append(
        "**False-prune diagnostic**: regex aggressive false-prunes 2.7% of "
        "LoCoMo gold (1/37) but 8.0% of synthetic gold (9/112), already near "
        "the 10% abandon threshold. The synth conversations embed factual "
        "info in short conversational acks more than LoCoMo does.\n"
    )
    lines.append(
        "**Why it failed**: v2f's multi-cue retrieval already navigates around "
        "low-signal turns via vocabulary matching. Removing backchannel turns "
        "does not free up top-K slots for new gold turns because those slots "
        "were already going to real content. Removing them only risks dropping "
        "tokens/context that happened to share vocabulary with gold turns.\n"
    )

    lines.append("## Pool-size reduction & false-prune rate (by dataset)\n")
    lines.append(
        "| Dataset | Variant | Pool | Pruned | % Pruned | Gold total | "
        "Gold pruned | False-prune rate |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for ds, entry in all_results.items():
        for variant, v_entry in entry["variants"].items():
            m = v_entry.get("mask_stats", {})
            fps = v_entry.get("false_prune_stats", {})
            lines.append(
                f"| {ds} | {variant} | {m.get('pool', 0)} | "
                f"{m.get('pruned', 0)} | {m.get('pct', 0.0)*100:.1f}% | "
                f"{fps.get('gold_total', 0)} | {fps.get('gold_pruned', 0)} | "
                f"{fps.get('false_prune_rate', 0.0)*100:.1f}% |"
            )

    lines.append("\n## Recall results\n")
    lines.append(
        "| Dataset | Variant | r@20 | d@20 | W/T/L@20 | r@50 | d@50 | W/T/L@50 |"
    )
    lines.append("|---|---|---:|---:|:-:|---:|---:|:-:|")
    for ds, entry in all_results.items():
        for variant, v_entry in entry["variants"].items():
            s = v_entry.get("summary", {})
            if not s or s.get("n", 0) == 0:
                continue
            lines.append(
                f"| {ds} | {variant} | "
                f"{s.get('arch_r@20', 0):.3f} | "
                f"{s.get('delta_r@20', 0):+.3f} | "
                f"{s.get('W/T/L_r@20', '?')} | "
                f"{s.get('arch_r@50', 0):.3f} | "
                f"{s.get('delta_r@50', 0):+.3f} | "
                f"{s.get('W/T/L_r@50', '?')} |"
            )

    # Per-category deltas (LoCoMo-30 only)
    locomo = all_results.get("locomo_30q", {})
    if locomo:
        lines.append("\n## Per-category delta (LoCoMo-30, r@20 / r@50)\n")
        cats = set()
        for v_entry in locomo["variants"].values():
            for c in v_entry.get("by_category", {}).keys():
                cats.add(c)
        for cat in sorted(cats):
            lines.append(f"\n### {cat}\n")
            lines.append(
                "| Variant | n | r@20 | d@20 | r@50 | d@50 |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|")
            for variant, v_entry in locomo["variants"].items():
                c = v_entry.get("by_category", {}).get(cat)
                if not c:
                    continue
                lines.append(
                    f"| {variant} | {c['n']} | "
                    f"{c.get('arch_r@20', 0):.3f} | "
                    f"{c.get('delta_r@20', 0):+.3f} | "
                    f"{c.get('arch_r@50', 0):.3f} | "
                    f"{c.get('delta_r@50', 0):+.3f} |"
                )

    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_out: dict = {}
    llm_cache = SalienceLLMCache() if RUN_LLM_VARIANT else None

    for ds in DATASETS:
        print(f"\n{'='*70}\nDATASET {ds}\n{'='*70}")
        store, questions = load_dataset(ds)
        ds_out = {"variants": {}}
        print(f"Loaded {len(store.segments)} segments, {len(questions)} questions")

        for variant in VARIANTS:
            print(f"\n--- variant: {variant} ---")
            if variant == "control":
                eval_store = store
                mask = None
                used_convs = {q["conversation_id"] for q in questions}
                pool = sum(
                    1 for seg in store.segments
                    if seg.conversation_id in used_convs
                )
                mask_stats = {
                    "pool": pool,
                    "pruned": 0,
                    "pct": 0.0,
                }
                # gold_total computed from questions (for reference)
                gold_total = sum(
                    len(set(q["source_chat_ids"])) for q in questions
                )
                fps = {
                    "gold_total": gold_total,
                    "gold_pruned": 0,
                    "false_prune_rate": 0.0,
                }
            else:
                # Restrict classification/marking to conversations referenced
                # by the question set (saves LLM cost, avoids masking unrelated
                # conversations unnecessarily).
                used_convs = {q["conversation_id"] for q in questions}
                if variant == "prune_llm":
                    if not RUN_LLM_VARIANT or llm_cache is None:
                        print("  skipping (LLM variant disabled)")
                        continue
                    mask = build_mask(
                        store, variant, llm_cache=llm_cache, verbose=True,
                        restrict_to_conv_ids=used_convs,
                    )
                else:
                    mask = build_mask(
                        store, variant,
                        restrict_to_conv_ids=used_convs,
                    )

                mode = "downweight" if variant == "downweight_regex" else "drop"
                eval_store = PrunedSegmentStore(store, mask, mode=mode)

                # Pool = segments in the conversations actually queried.
                target_mask = np.array(
                    [seg.conversation_id in used_convs for seg in store.segments],
                    dtype=bool,
                )
                pool = int(target_mask.sum())
                pruned = int((mask.low_salience & target_mask).sum())
                mask_stats = {
                    "pool": pool,
                    "pruned": pruned,
                    "pct": pruned / pool if pool else 0.0,
                }
                fps_full = false_prune_stats(store, mask, questions)
                fps = {
                    "gold_total": fps_full["gold_total"],
                    "gold_pruned": fps_full["gold_pruned"],
                    "false_prune_rate": fps_full["false_prune_rate"],
                    "pruned_gold_samples": fps_full["pruned_gold_samples"],
                }
                print(
                    f"  pool={pool} pruned={pruned} "
                    f"({mask_stats['pct']*100:.1f}%) "
                    f"gold_pruned={fps['gold_pruned']}/{fps['gold_total']} "
                    f"({fps['false_prune_rate']*100:.1f}%)"
                )

                # Decision rule: skip aggressive prune if >10% false-prune on LoCoMo
                # (we still evaluate; flag in report)

            rows = evaluate_variant(variant, store, eval_store, questions)
            summary = summarize(rows, variant, ds)
            by_cat = summarize_by_category(rows)

            print(
                f"  SUMMARY: r@20={summary.get('arch_r@20', 0):.3f} "
                f"(d{summary.get('delta_r@20', 0):+.3f}) "
                f"r@50={summary.get('arch_r@50', 0):.3f} "
                f"(d{summary.get('delta_r@50', 0):+.3f})"
            )

            ds_out["variants"][variant] = {
                "mask_stats": mask_stats,
                "false_prune_stats": fps,
                "summary": summary,
                "by_category": by_cat,
                "rows": rows,
            }
            sys.stdout.flush()

            # Checkpoint after each variant
            partial_path = RESULTS_DIR / "salience_pruning.json"
            all_out[ds] = ds_out
            with open(partial_path, "w") as f:
                json.dump(all_out, f, indent=2, default=str)

        all_out[ds] = ds_out

    # Final save
    json_path = RESULTS_DIR / "salience_pruning.json"
    with open(json_path, "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    md_path = RESULTS_DIR / "salience_pruning.md"
    with open(md_path, "w") as f:
        f.write(format_markdown(all_out))
    print(f"Saved: {md_path}")

    if llm_cache is not None:
        llm_cache.save()


if __name__ == "__main__":
    main()
