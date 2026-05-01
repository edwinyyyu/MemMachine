"""Fair-backfill evaluation of the contrastive retrieval probe.

Architectures evaluated:
  v2f                            — MetaV2f baseline (contrast caches)
  cosine_baseline                — pure cosine top-K baseline
  contrast_only_a{0.2,0.5,1.0}   — raw query + distractor, re-rank whole conv
  contrast_v2f_a{0.2,0.5,1.0}    — v2f retrieval + contrastive re-scoring

Datasets: locomo_30q, synthetic_19q.
Budgets:  K=20 and K=50, fair-backfill.

Usage:
    uv run python contrastive_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from associative_recall import Segment, SegmentStore
from contrastive_retrieval import (
    ContrastiveOnly,
    ContrastiveV2F,
    CosineBaseline,
    V2FReference,
)
from dotenv import load_dotenv
from fair_backfill_eval import (
    BUDGETS,
    DATA_DIR,
    RESULTS_DIR,
    fair_backfill_evaluate,
    summarize,
    summarize_by_category,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


ALPHAS = (0.2, 0.5, 1.0)

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


def build_arch(arch_name: str, store: SegmentStore):
    if arch_name == "v2f":
        return V2FReference(store)
    if arch_name == "cosine_baseline":
        return CosineBaseline(store)
    if arch_name.startswith("contrast_only_a"):
        alpha = float(arch_name.split("_a", 1)[1])
        return ContrastiveOnly(store, alpha=alpha)
    if arch_name.startswith("contrast_v2f_a"):
        alpha = float(arch_name.split("_a", 1)[1])
        return ContrastiveV2F(store, alpha=alpha)
    raise ValueError(f"unknown arch: {arch_name}")


def alpha_str(a: float) -> str:
    # 0.2 -> "0.2", 1.0 -> "1" (matches {alpha:g})
    return f"{a:g}"


def all_arch_names() -> list[str]:
    names = ["v2f", "cosine_baseline"]
    for a in ALPHAS:
        names.append(f"contrast_only_a{alpha_str(a)}")
    for a in ALPHAS:
        names.append(f"contrast_v2f_a{alpha_str(a)}")
    return names


def evaluate_question(arch, question: dict) -> dict:
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
    cosine_result = arch.store.search(query_emb, top_k=max_K, conversation_id=conv_id)
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
        "answer_probe": result.metadata.get("answer_probe"),
        "distractor_probe": result.metadata.get("distractor_probe"),
        "sample_scores": result.metadata.get("sample_scores", []),
    }

    for K in BUDGETS:
        b_rec, a_rec, _ = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)

    return row


def run_one(arch_name: str, arch, dataset: str, questions: list[dict]):
    print(f"\n{'=' * 70}")
    print(f"{arch_name} | {dataset} | {len(questions)} questions")
    print(f"{'=' * 70}")

    results = []
    for i, q in enumerate(questions):
        q_short = q["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
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


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _category_deltas(
    all_summaries: dict, arch_name: str, base_arch: str, dataset: str, K: int
) -> list[tuple[str, int, float]]:
    """Return list of (category, n, delta_arch_vs_base) at r@K on dataset."""
    out = []
    base_cb = (
        all_summaries.get(base_arch, {}).get(dataset, {}).get("category_breakdown", {})
    )
    cb = all_summaries.get(arch_name, {}).get(dataset, {}).get("category_breakdown", {})
    for cat, row in cb.items():
        base_row = base_cb.get(cat)
        if not base_row:
            continue
        delta = row[f"arch_r@{K}"] - base_row[f"arch_r@{K}"]
        out.append((cat, row["n"], delta))
    return out


def render_markdown(
    all_summaries: dict, all_results: dict, arch_names: list[str]
) -> str:
    lines: list[str] = []
    lines.append("# Contrastive Retrieval Probe\n")
    lines.append(
        "Score = cos(answer_probe, turn) - α · cos(distractor_probe, turn). "
        "Turns matching BOTH probes (paraphrase-style matches) get penalized; "
        "turns matching only the answer-probe get surfaced.\n"
    )

    # Constraints
    lines.append("## Setup")
    lines.append("- `text-embedding-3-small`, `gpt-5-mini` (fixed).")
    lines.append("- Fair-backfill eval at K=20 and K=50.")
    lines.append("- Datasets: locomo_30q, synthetic_19q.")
    lines.append(
        "- Variants: `contrast_only_a{0.2,0.5,1.0}` (raw query answer-probe, "
        "re-rank whole conversation) and `contrast_v2f_a{0.2,0.5,1.0}` "
        "(v2f retrieval + contrastive re-rank of candidate pool).\n"
    )

    # Recall table
    lines.append("## Recall table (arch_r@K, fair-backfill)")
    lines.append("| Dataset | K | " + " | ".join(arch_names) + " |")
    lines.append("|---------|---|" + "|".join(["------:"] * len(arch_names)) + "|")
    for ds in DATASETS:
        for K in BUDGETS:
            cells = []
            for arch in arch_names:
                s = all_summaries.get(arch, {}).get(ds, {}).get("summary")
                cells.append(f"{s[f'arch_r@{K}']:.3f}" if s else "n/a")
            lines.append(f"| {ds} | {K} | " + " | ".join(cells) + " |")
    lines.append("")

    # Deltas vs v2f
    base = "v2f"
    lines.append(f"## Deltas vs `{base}` (arch_r@K)")
    others = [a for a in arch_names if a != base]
    lines.append("| Dataset | K | " + " | ".join(others) + " |")
    lines.append("|---------|---|" + "|".join(["------:"] * len(others)) + "|")
    for ds in DATASETS:
        for K in BUDGETS:
            base_s = all_summaries.get(base, {}).get(ds, {}).get("summary")
            if not base_s:
                continue
            base_v = base_s[f"arch_r@{K}"]
            cells = []
            for arch in others:
                s = all_summaries.get(arch, {}).get(ds, {}).get("summary")
                if not s:
                    cells.append("n/a")
                else:
                    d = s[f"arch_r@{K}"] - base_v
                    cells.append(f"{d:+.3f}")
            lines.append(f"| {ds} | {K} | " + " | ".join(cells) + " |")
    lines.append("")

    # W/T/L vs cosine baseline (fair-backfill 'delta_r@K' is vs cosine)
    lines.append("## W/T/L vs cosine baseline (delta_r@K)")
    lines.append("| Dataset | K | " + " | ".join(arch_names) + " |")
    lines.append("|---------|---|" + "|".join(["------:"] * len(arch_names)) + "|")
    for ds in DATASETS:
        for K in BUDGETS:
            cells = []
            for arch in arch_names:
                s = all_summaries.get(arch, {}).get(ds, {}).get("summary")
                cells.append(s[f"W/T/L_r@{K}"] if s else "n/a")
            lines.append(f"| {ds} | {K} | " + " | ".join(cells) + " |")
    lines.append("")

    # Optimal alpha
    lines.append("## Best α per family (on locomo_30q, K=50)")
    for family in ("contrast_only", "contrast_v2f"):
        arch_and_score: list[tuple[str, float]] = []
        for a in ALPHAS:
            name = f"{family}_a{alpha_str(a)}"
            s = all_summaries.get(name, {}).get("locomo_30q", {}).get("summary")
            if s:
                arch_and_score.append((name, s["arch_r@50"]))
        arch_and_score.sort(key=lambda x: x[1], reverse=True)
        if arch_and_score:
            best, score = arch_and_score[0]
            lines.append(f"- `{family}` best: `{best}` at r@50={score:.3f}")
    lines.append("")

    # Category breakdown (locomo_30q, K=50) — gainers vs losers
    lines.append("## Top 2 gaining / losing categories (locomo_30q, r@50)")
    lines.append("For each contrastive variant, top 2 categories by delta vs `v2f`.")
    for arch in others:
        deltas = _category_deltas(all_summaries, arch, base, "locomo_30q", 50)
        if not deltas:
            continue
        deltas.sort(key=lambda x: x[2], reverse=True)
        gainers = deltas[:2]
        losers = deltas[-2:][::-1]
        g_str = ", ".join(f"{c} ({n}, {d:+.3f})" for c, n, d in gainers)
        l_str = ", ".join(f"{c} ({n}, {d:+.3f})" for c, n, d in losers)
        lines.append(f"- `{arch}`:")
        lines.append(f"    - Gainers: {g_str}")
        lines.append(f"    - Losers: {l_str}")
    lines.append("")

    # Paraphrase penalty inspection (samples)
    lines.append("## Paraphrase-penalty inspection (samples)")
    lines.append(
        "For the first 3 locomo_30q questions: show answer_probe, "
        "distractor_probe, and the top-5 ranked turns for v2f vs "
        "`contrast_v2f_a0.5`. `cos_a` = cosine with answer-probe (query); "
        "`cos_d` = cosine with distractor-probe; `score` = cos_a − α·cos_d."
    )
    v2f_rows = all_results.get("v2f", {}).get("locomo_30q", [])
    cv_rows = all_results.get(f"contrast_v2f_a{alpha_str(0.5)}", {}).get(
        "locomo_30q", []
    )
    for i in range(min(3, len(cv_rows))):
        cv = cv_rows[i]
        v2f_r = v2f_rows[i] if i < len(v2f_rows) else None
        lines.append(f"\n### Question: _{cv['question']}_")
        lines.append(f"- category: `{cv['category']}`")
        lines.append(f"- distractor_probe: _{cv.get('distractor_probe', '')}_")
        gold = set(cv.get("source_chat_ids", []))
        lines.append(f"- gold turn_ids: {sorted(gold)}")

        # v2f top-5 (retrieval order)
        lines.append("\n**v2f top 5 (retrieval order):**")
        if v2f_r:
            # v2f has no sample_scores; use cues and 'hit' marker via
            # fair_backfill only. We print cues.
            for k, cue in enumerate(v2f_r.get("cues", [])[:4]):
                lines.append(f"- CUE {k}: {cue}")
            lines.append(
                f"- fair-backfill r@20={v2f_r['fair_backfill']['arch_r@20']:.3f} "
                f"r@50={v2f_r['fair_backfill']['arch_r@50']:.3f}"
            )
        else:
            lines.append("- (not available)")

        lines.append("\n**contrast_v2f_a0.5 re-rank top 5:**")
        for s in cv.get("sample_scores", [])[:5]:
            hit = "HIT" if s["turn_id"] in gold else "   "
            lines.append(
                f"- [{hit}] turn {s['turn_id']:>3d} "
                f"cos_a={s['cos_answer']:+.3f} "
                f"cos_d={s['cos_distractor']:+.3f} "
                f"score={s['score']:+.3f}  "
                f"{s['role']}: {s['text_preview']}"
            )
        lines.append(
            f"- fair-backfill r@20={cv['fair_backfill']['arch_r@20']:.3f} "
            f"r@50={cv['fair_backfill']['arch_r@50']:.3f}"
        )
    lines.append("")

    # Verdict scaffolding
    lines.append("## Verdict")
    lines.append("Apply the decision rules from the brief:")
    lines.append(
        "1. If any `contrast_v2f_a*` beats `v2f` at any (ds, K) — clean win, "
        "note best α."
    )
    lines.append(
        "2. If `contrast_only` beats `v2f` but `contrast_v2f` doesn't — "
        "distractor-penalty works alone but conflicts with cue merging."
    )
    lines.append(
        "3. If all α lose — the distractor-probe is bad or paraphrase matches "
        "are already filtered by v2f's retrieval structure."
    )
    lines.append(
        "4. If even α=0.2 loses by >1pp to v2f — inspect distractor samples; "
        "probe generation likely broken."
    )

    # Compute a crude machine verdict for convenience.
    def _cv_vs_v2f(a: float, ds: str, K: int) -> float:
        v = (
            all_summaries.get("v2f", {})
            .get(ds, {})
            .get("summary", {})
            .get(f"arch_r@{K}")
        )
        c = (
            all_summaries.get(f"contrast_v2f_a{alpha_str(a)}", {})
            .get(ds, {})
            .get("summary", {})
            .get(f"arch_r@{K}")
        )
        if v is None or c is None:
            return 0.0
        return c - v

    any_cv_win = False
    best_cv = None
    best_delta = -1e9
    for a in ALPHAS:
        for ds in DATASETS:
            for K in BUDGETS:
                d = _cv_vs_v2f(a, ds, K)
                if d > best_delta:
                    best_delta = d
                    best_cv = (a, ds, K, d)
                if d > 0.001:
                    any_cv_win = True

    lines.append("")
    if any_cv_win and best_cv:
        a, ds, K, d = best_cv
        lines.append(
            f"**Machine verdict (tentative):** `contrast_v2f_a{alpha_str(a)}` "
            f"beats v2f on {ds} K={K} by {d:+.3f}. Consider supplement/ship."
        )
    elif best_cv:
        a, ds, K, d = best_cv
        lines.append(
            f"**Machine verdict (tentative):** no contrast_v2f variant beats "
            f"v2f. Best single cell: `contrast_v2f_a{alpha_str(a)}` on {ds} "
            f"K={K} (Δ={d:+.3f}). Likely ABANDON or try supplement-only usage."
        )

    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    arch_names = all_arch_names()
    all_summaries: dict = {}
    all_results: dict = {}

    for ds_name in DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        for arch_name in arch_names:
            arch = build_arch(arch_name, store)
            results, summary, by_cat = run_one(arch_name, arch, ds_name, questions)

            out_path = RESULTS_DIR / f"contrast_{arch_name}_{ds_name}.json"
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
    summary_path = RESULTS_DIR / "contrastive_retrieval.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved: {summary_path}")

    md = render_markdown(all_summaries, all_results, arch_names)
    md_path = RESULTS_DIR / "contrastive_retrieval.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved: {md_path}")

    # Console summary
    print("\n" + "=" * 110)
    print("CONTRASTIVE RETRIEVAL SUMMARY")
    print("=" * 110)
    header = (
        f"{'Arch':<28s} {'Dataset':<14s} "
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
                f"{arch_name:<28s} {ds_name:<14s} "
                f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
                f"{s['delta_r@20']:>+7.3f} {s['W/T/L_r@20']:>10s} "
                f"{s['baseline_r@50']:>8.3f} {s['arch_r@50']:>8.3f} "
                f"{s['delta_r@50']:>+7.3f} {s['W/T/L_r@50']:>10s}"
            )


if __name__ == "__main__":
    main()
