"""Cross-architecture failure overlap analysis.

Goal: identify questions that fail on a large fraction of tested retrieval
architectures, then characterize their surface features. This disentangles
"cue generation is the bottleneck" from "structural ceiling".

Outputs:
  results/failure_overlap_analysis.md  -- human-readable report
  results/failure_overlap_analysis.json -- raw failure matrix
"""

from __future__ import annotations

import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

# Dataset patterns mapping result-file substring -> dataset key.
DATASET_FILE_MAP = {
    "locomo_30q": "locomo_30q",
    "locomo_30": "locomo_30q",   # some files use "locomo" (fulleval)
    "synthetic_19q": "synthetic_19q",
    "puzzle_16q": "puzzle_16q",
    "advanced_23q": "advanced_23q",
}

DATASETS = ["locomo_30q", "synthetic_19q", "puzzle_16q", "advanced_23q"]

FAILURE_THRESHOLD = 0.5   # recall < 0.5 at K=50 counts as a failure
SUCCESS_THRESHOLD = 0.5   # recall >= 0.5 at K=50 counts as a success
STUBBORN_FAIL_FRACTION = 0.80   # >=80% of architectures must fail
EASY_SUCCESS_FRACTION = 0.80    # >=80% of architectures must succeed


def detect_dataset(filename: str) -> str | None:
    """Return dataset key given a result filename, or None if ambiguous."""
    f = filename.lower()
    # Specific full-name matches first (handles "fulleval_puzzle_v15_control")
    for key in ("locomo_30q", "synthetic_19q", "puzzle_16q", "advanced_23q"):
        if key in f:
            return key
    # fulleval files say "fulleval_<dataset>_<arch>.json"
    if f.startswith("fulleval_"):
        rest = f[len("fulleval_"):]
        for short, full in [
            ("locomo", "locomo_30q"),
            ("synthetic", "synthetic_19q"),
            ("puzzle", "puzzle_16q"),
            ("advanced", "advanced_23q"),
        ]:
            if rest.startswith(short + "_"):
                return full
    return None


def derive_arch_name(filename: str, dataset: str | None) -> str:
    """Derive a short architecture identifier from a filename."""
    stem = filename[:-5] if filename.endswith(".json") else filename
    # Strip dataset suffix if present
    if dataset:
        stem = stem.replace("_" + dataset, "")
        # also strip short forms
        short_map = {
            "locomo_30q": "locomo",
            "synthetic_19q": "synthetic",
            "puzzle_16q": "puzzle",
            "advanced_23q": "advanced",
        }
        short = short_map.get(dataset)
        if short and ("_" + short) in stem:
            stem = stem.replace("_" + short, "")
    return stem


def extract_k50_recall(result: dict, arch_key: str | None = None) -> float | None:
    """Extract K=50 recall from a per-question result entry. Returns None if unavailable."""
    # Pattern 1: fair_backfill.arch_r@50
    if "fair_backfill" in result and isinstance(result["fair_backfill"], dict):
        fb = result["fair_backfill"]
        if "arch_r@50" in fb:
            return float(fb["arch_r@50"])
    # Pattern 2: arch_recalls or <name>_recalls dicts
    for key in result:
        if key.endswith("_recalls") or key == "arch_recalls":
            val = result[key]
            # skip baseline_recalls; we handle that separately for "baseline" arch.
            if key == "baseline_recalls":
                continue
            if isinstance(val, dict) and "r@50" in val:
                return float(val["r@50"])
    # Pattern 3: plain recall field (budget files use this; budget specifies K)
    if "recall" in result and "budget" in result:
        # Only use if this is the K=50 file -- caller should guarantee that.
        if float(result["budget"]) >= 50:
            return float(result["recall"])
    return None


def extract_baseline_k50(result: dict) -> float | None:
    if "fair_backfill" in result and isinstance(result["fair_backfill"], dict):
        fb = result["fair_backfill"]
        if "baseline_r@50" in fb:
            return float(fb["baseline_r@50"])
    if "baseline_recalls" in result and isinstance(result["baseline_recalls"], dict):
        return float(result["baseline_recalls"].get("r@50"))
    return None


def load_architecture_file(path: Path) -> tuple[dict | None, str | None]:
    """Load file and return (arch_label, dataset, per_question_recalls).

    per_question_recalls is dict[str -> dict] mapping a normalized question key
    to {recall, question, category, source_chat_ids, num_source_turns, conversation_id}.
    Returns ({}, None) if no K=50 per-question data is present.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None, None

    dataset = detect_dataset(path.name)
    if dataset is None:
        return None, None

    # Extract result list
    results = None
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict) and "results" in data:
        results = data["results"]
    if not isinstance(results, list) or not results:
        return None, dataset

    # Build per-question map
    per_q: dict[str, dict] = {}
    saw_recall = False
    for r in results:
        if not isinstance(r, dict):
            continue
        q = r.get("question")
        if not q:
            continue
        recall = extract_k50_recall(r)
        if recall is None:
            continue
        saw_recall = True
        key = make_key(q, r.get("conversation_id"), r.get("question_index"))
        per_q[key] = {
            "recall": float(recall),
            "question": q,
            "category": r.get("category"),
            "source_chat_ids": r.get("source_chat_ids"),
            "num_source_turns": r.get("num_source_turns"),
            "conversation_id": r.get("conversation_id"),
            "question_index": r.get("question_index"),
        }
    if not saw_recall:
        return None, dataset
    return per_q, dataset


def make_key(question: str, conv_id: Any, qidx: Any) -> str:
    """Stable question key. Combines conv_id + question_index + normalized text."""
    norm = re.sub(r"\s+", " ", (question or "").strip().lower())
    return f"{conv_id}|{qidx}|{norm}"


# Architecture families to include. We prefer unique "full-coverage" architectures
# that have per-question K=50 data across multiple datasets.
INCLUDE_PATTERNS = [
    # fair_backfill family
    "fairbackfill_v15_control_",
    "fairbackfill_meta_v2f_",
    "fairbackfill_hybrid_v2f_gencheck_",
    # type_enumerated family
    "type_enum_v2f_plus_types_",
    "type_enum_type_enumerated_",
    "type_enum_type_enumerated_selective_",
    # chain / goal_chain family
    "goal_chain_chain_with_scratchpad_",
    "goal_chain_chain_goal_tracking_",
    # self_dispatch family
    "self_cot_",
    "self_v2_",
    "self_v3_",
    # domain-agnostic family
    "domain_agnostic_v2f_style_explicit_",
    "domain_agnostic_v2f_register_inferred_",
    "domain_agnostic_v2f_minimal_",
    # two_call
    "two_call_",
    # cot_chain_of_thought
    "cot_chain_of_thought_",
    # budget baselines (cosine baselines) at K=50
    "budget_baseline_50_",
    "budget_v15_tight_50_",
    "budget_gencheck_50_",
    # fulleval (covers synthetic, puzzle, advanced via fulleval_<ds>_<arch>)
    "fulleval_",
    # v15_hybrid variants
    "v15_hybrid_hybrid_v15_cot_",
    "v15_hybrid_hybrid_v15_dual_",
    "v15_hybrid_hybrid_v15_memidx_",
]

EXCLUDE_PATTERNS = [
    "summary",
    "all_summaries",
    "category_breakdown",
]


def should_include(filename: str) -> bool:
    lower = filename.lower()
    if not lower.endswith(".json"):
        return False
    if any(bad in lower for bad in EXCLUDE_PATTERNS):
        return False
    return any(lower.startswith(pref) for pref in INCLUDE_PATTERNS)


def main() -> None:
    # dataset -> arch_label -> question_key -> record
    arch_data: dict[str, dict[str, dict[str, dict]]] = {ds: {} for ds in DATASETS}

    files = sorted(os.listdir(RESULTS_DIR))
    catalogued = []
    skipped = []
    for fname in files:
        if not should_include(fname):
            continue
        path = RESULTS_DIR / fname
        per_q, dataset = load_architecture_file(path)
        if per_q is None or dataset is None:
            skipped.append(fname)
            continue
        arch_label = derive_arch_name(fname, dataset)
        # If duplicate arch for this dataset, keep the one with more coverage
        existing = arch_data[dataset].get(arch_label)
        if existing is not None and len(existing) >= len(per_q):
            continue
        arch_data[dataset][arch_label] = per_q
        catalogued.append((dataset, arch_label, len(per_q), fname))

    # Build the failure matrix per dataset.
    # For each dataset, we need the full question universe (union across archs).
    # Use the richest architecture to provide question metadata.
    per_dataset_summary = {}
    question_meta: dict[str, dict[str, dict]] = {ds: {} for ds in DATASETS}
    stubborn_failures: dict[str, list[dict]] = {ds: [] for ds in DATASETS}
    easy_questions: dict[str, list[dict]] = {ds: [] for ds in DATASETS}
    category_stubborn_rates: dict[str, dict[str, dict[str, int]]] = {ds: {} for ds in DATASETS}
    failure_matrix_raw: dict[str, dict[str, dict]] = {ds: {} for ds in DATASETS}

    for ds in DATASETS:
        archs = arch_data[ds]
        if not archs:
            per_dataset_summary[ds] = {"n_arch": 0, "stubborn": 0, "easy": 0}
            continue

        # Gather all question keys across archs.
        all_keys: set[str] = set()
        for arch_label, per_q in archs.items():
            all_keys.update(per_q.keys())
        # Populate metadata by preferring first arch that has it.
        for key in all_keys:
            for arch_label, per_q in archs.items():
                if key in per_q:
                    rec = per_q[key]
                    question_meta[ds][key] = {
                        "question": rec["question"],
                        "category": rec["category"],
                        "num_source_turns": rec["num_source_turns"],
                        "source_chat_ids": rec["source_chat_ids"],
                        "conversation_id": rec["conversation_id"],
                        "question_index": rec["question_index"],
                    }
                    break

        # For stubbornness we only count an arch that measured this question.
        # For each question: list of (arch_label, recall) tuples where measured.
        per_q_measurements: dict[str, list[tuple[str, float]]] = {}
        for key in all_keys:
            measured = []
            for arch_label, per_q in archs.items():
                if key in per_q:
                    measured.append((arch_label, per_q[key]["recall"]))
            per_q_measurements[key] = measured

        # Only consider questions measured by >=50% of archs (to avoid bias toward small samples)
        n_archs = len(archs)
        min_measurements = max(3, int(0.5 * n_archs))

        n_stubborn = 0
        n_easy = 0
        stubborn_list = []
        easy_list = []
        per_category_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"n_questions": 0, "n_stubborn": 0, "n_easy": 0})

        for key, measurements in per_q_measurements.items():
            if len(measurements) < min_measurements:
                continue
            meta = question_meta[ds].get(key, {})
            cat = meta.get("category", "unknown")
            per_category_counts[cat]["n_questions"] += 1

            recalls = [r for _, r in measurements]
            n_fail = sum(1 for r in recalls if r < FAILURE_THRESHOLD)
            n_pass = sum(1 for r in recalls if r >= SUCCESS_THRESHOLD)
            fail_frac = n_fail / len(measurements)
            pass_frac = n_pass / len(measurements)

            best_recall = max(recalls)
            best_arch = max(measurements, key=lambda t: t[1])[0]

            failure_matrix_raw[ds][key] = {
                "question": meta.get("question"),
                "category": cat,
                "num_source_turns": meta.get("num_source_turns"),
                "conversation_id": meta.get("conversation_id"),
                "question_index": meta.get("question_index"),
                "n_measurements": len(measurements),
                "n_fail": n_fail,
                "n_pass": n_pass,
                "fail_frac": fail_frac,
                "pass_frac": pass_frac,
                "best_recall": best_recall,
                "best_arch": best_arch,
                "recalls": dict(measurements),
            }

            if fail_frac >= STUBBORN_FAIL_FRACTION:
                n_stubborn += 1
                per_category_counts[cat]["n_stubborn"] += 1
                stubborn_list.append({
                    "key": key,
                    "question": meta.get("question"),
                    "category": cat,
                    "num_source_turns": meta.get("num_source_turns"),
                    "conversation_id": meta.get("conversation_id"),
                    "n_measurements": len(measurements),
                    "n_fail": n_fail,
                    "fail_frac": fail_frac,
                    "best_recall": best_recall,
                    "best_arch": best_arch,
                })
            if pass_frac >= EASY_SUCCESS_FRACTION:
                n_easy += 1
                per_category_counts[cat]["n_easy"] += 1
                easy_list.append({
                    "key": key,
                    "question": meta.get("question"),
                    "category": cat,
                    "num_source_turns": meta.get("num_source_turns"),
                    "conversation_id": meta.get("conversation_id"),
                    "n_measurements": len(measurements),
                    "pass_frac": pass_frac,
                })

        per_dataset_summary[ds] = {
            "n_arch": n_archs,
            "n_questions_assessed": sum(1 for k, m in per_q_measurements.items() if len(m) >= min_measurements),
            "n_total_questions": len(all_keys),
            "min_measurements": min_measurements,
            "stubborn": n_stubborn,
            "easy": n_easy,
        }
        stubborn_failures[ds] = stubborn_list
        easy_questions[ds] = easy_list
        category_stubborn_rates[ds] = dict(per_category_counts)

    # Compute cross-dataset stats and best-arch-on-stubborn comparisons.
    arch_stubborn_hits: dict[str, dict[str, int]] = {ds: defaultdict(int) for ds in DATASETS}
    for ds in DATASETS:
        for stub in stubborn_failures[ds]:
            best_arch = stub["best_arch"]
            if stub["best_recall"] >= FAILURE_THRESHOLD:
                arch_stubborn_hits[ds][best_arch] += 1

    # Prepare JSON output.
    json_out = {
        "threshold": FAILURE_THRESHOLD,
        "stubborn_fraction": STUBBORN_FAIL_FRACTION,
        "easy_fraction": EASY_SUCCESS_FRACTION,
        "catalogued": [
            {"dataset": ds, "arch": a, "n_questions": n, "file": fn}
            for (ds, a, n, fn) in catalogued
        ],
        "skipped_files_count": len(skipped),
        "per_dataset_summary": per_dataset_summary,
        "stubborn_failures": stubborn_failures,
        "easy_questions_count": {ds: len(easy_questions[ds]) for ds in DATASETS},
        "category_stubborn_rates": category_stubborn_rates,
        "arch_stubborn_hits": {ds: dict(v) for ds, v in arch_stubborn_hits.items()},
        "failure_matrix": failure_matrix_raw,
    }
    (RESULTS_DIR / "failure_overlap_analysis.json").write_text(json.dumps(json_out, indent=2))

    # Generate Markdown report.
    lines: list[str] = []
    lines.append("# Cross-Architecture Failure Overlap")
    lines.append("")
    lines.append(f"Failure threshold: recall @K=50 < {FAILURE_THRESHOLD}. "
                 f"Stubborn threshold: >= {int(STUBBORN_FAIL_FRACTION*100)}% of archs fail. "
                 f"Easy threshold: >= {int(EASY_SUCCESS_FRACTION*100)}% of archs succeed.")
    lines.append("")

    lines.append("## Architectures catalogued")
    lines.append("")
    lines.append("| Dataset | Architecture | Questions |")
    lines.append("| --- | --- | --- |")
    for ds in DATASETS:
        for a, per_q in sorted(arch_data[ds].items()):
            lines.append(f"| {ds} | {a} | {len(per_q)} |")
    lines.append("")

    lines.append("## Per-dataset summary")
    lines.append("")
    lines.append("| Dataset | # Arch | # Questions assessed | Stubborn failures (>=80% fail) | Easy (>=80% pass) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for ds in DATASETS:
        s = per_dataset_summary[ds]
        lines.append(f"| {ds} | {s['n_arch']} | {s.get('n_questions_assessed', 0)} | {s.get('stubborn',0)} | {s.get('easy',0)} |")
    lines.append("")

    # Detailed stubborn-failure listings per dataset.
    for ds in DATASETS:
        stubs = sorted(stubborn_failures[ds], key=lambda x: (-x["fail_frac"], x.get("question_index") or 0))
        if not stubs:
            continue
        lines.append(f"## Stubborn failures -- {ds}")
        lines.append("")
        lines.append("| Conv | Category | # gold | Fail frac | Best recall | Best arch | Question |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- | --- |")
        for s in stubs:
            qtext = (s["question"] or "").replace("|", "\\|")
            lines.append(
                f"| {s.get('conversation_id','')} | {s.get('category','')} | "
                f"{s.get('num_source_turns','')} | "
                f"{s['n_fail']}/{s['n_measurements']} ({s['fail_frac']:.2f}) | "
                f"{s['best_recall']:.2f} | {s['best_arch']} | {qtext[:180]} |"
            )
        lines.append("")

    # Top-5 hardest questions per dataset regardless of stubborn threshold.
    lines.append("## Top-5 hardest questions per dataset (any fail frac)")
    lines.append("")
    for ds in DATASETS:
        fm = failure_matrix_raw[ds]
        if not fm:
            continue
        hardest = sorted(fm.values(), key=lambda v: -v["fail_frac"])[:5]
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Fail frac | Best recall | Best arch | Cat | Gold | Question |")
        lines.append("| ---: | ---: | --- | --- | ---: | --- |")
        for h in hardest:
            q = (h.get("question") or "").replace("|", "\\|")
            lines.append(
                f"| {h['fail_frac']:.2f} | {h['best_recall']:.2f} | {h['best_arch']} | "
                f"{h['category']} | {h['num_source_turns']} | {q[:180]} |"
            )
        lines.append("")

    # Category-level stubbornness rates.
    lines.append("## Category-level stubbornness rates")
    lines.append("")
    lines.append("Percent of questions in a category that are stubborn failures (assessed questions only).")
    lines.append("")
    for ds in DATASETS:
        cats = category_stubborn_rates[ds]
        if not cats:
            continue
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Category | N | # Stubborn | Stubborn rate | # Easy | Easy rate |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for cat, counts in sorted(cats.items(), key=lambda x: -(x[1]["n_stubborn"]/max(x[1]["n_questions"],1))):
            n = counts["n_questions"]
            sb = counts["n_stubborn"]
            ez = counts["n_easy"]
            sb_rate = sb / n if n else 0
            ez_rate = ez / n if n else 0
            lines.append(f"| {cat} | {n} | {sb} | {sb_rate:.2f} | {ez} | {ez_rate:.2f} |")
        lines.append("")

    # Surface feature contrast: stubborn vs. easy.
    lines.append("## Surface features: stubborn vs. easy")
    lines.append("")
    lines.append("| Dataset | Group | N | mean Q length (chars) | mean # gold turns | multi-part rate |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")

    def safe_mean(xs: Iterable[float]) -> float:
        xs = [x for x in xs if x is not None]
        return statistics.fmean(xs) if xs else 0.0

    def multi_part_rate(qs: Iterable[str]) -> float:
        qs = [q for q in qs if q]
        if not qs:
            return 0.0
        n_multi = sum(1 for q in qs if (" and " in q.lower()) or "?" in q[:-1] or ", " in q)
        return n_multi / len(qs)

    for ds in DATASETS:
        stubs = stubborn_failures[ds]
        easies = easy_questions[ds]
        if not stubs and not easies:
            continue
        for label, group in [("stubborn", stubs), ("easy", easies)]:
            qs = [x.get("question") for x in group]
            ng = [x.get("num_source_turns") for x in group if x.get("num_source_turns") is not None]
            lines.append(
                f"| {ds} | {label} | {len(group)} | {safe_mean([len(q) for q in qs if q]):.1f} | "
                f"{safe_mean(ng):.2f} | {multi_part_rate(qs):.2f} |"
            )
    lines.append("")

    # Best-arch on stubborn: which architectures recover stubborn questions?
    lines.append("## Which architecture is best on stubborn failures?")
    lines.append("")
    lines.append("Count of stubborn questions where this arch was the best (and achieved recall >= 0.5).")
    lines.append("")
    for ds in DATASETS:
        hits = arch_stubborn_hits[ds]
        if not hits:
            continue
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Arch | # stubborn Qs recovered |")
        lines.append("| --- | ---: |")
        for arch, n in sorted(hits.items(), key=lambda x: -x[1]):
            lines.append(f"| {arch} | {n} |")
        lines.append("")

    # Verdict.
    lines.append("## Verdict")
    lines.append("")

    total_stubborn = sum(len(stubborn_failures[ds]) for ds in DATASETS)
    total_structural = 0
    total_recoverable = 0
    for ds in DATASETS:
        for s in stubborn_failures[ds]:
            if s["best_recall"] < FAILURE_THRESHOLD:
                total_structural += 1
            else:
                total_recoverable += 1
    lines.append(f"- Total stubborn failures across all datasets: **{total_stubborn}**")
    lines.append(f"- Of those, **{total_structural}** are *structural ceilings* (no architecture reached recall >= 0.5).")
    lines.append(f"- **{total_recoverable}** are *architecture-specific solvable but unstable* (at least one architecture achieved recall >= 0.5).")
    lines.append("")

    if total_stubborn == 0:
        lines.append("Cue generation clearly helps: no question is structurally unreachable.")
    else:
        ratio_structural = total_structural / max(total_stubborn, 1)
        if ratio_structural >= 0.6:
            lines.append("**Verdict: mostly structural ceiling.** Over 60% of stubborn questions have no architecture that solves them, pointing to a retrieval-index limitation or gold-labeling noise rather than a cue-generation deficiency.")
        elif ratio_structural >= 0.3:
            lines.append("**Verdict: mixed.** A meaningful fraction of stubborn questions are structurally unreachable, but others have at least one arch that solves them — suggesting both cue-generation instability *and* some genuine ceilings.")
        else:
            lines.append("**Verdict: fixable.** Most stubborn questions have at least one architecture that solves them, implying the bottleneck is cue-generation stability / routing rather than structural retrieval limits.")
    lines.append("")
    lines.append("## Pointers")
    lines.append("")
    lines.append("- Raw failure matrix: `results/failure_overlap_analysis.json`")
    lines.append("- Source script: `failure_overlap.py`")
    lines.append("")

    (RESULTS_DIR / "failure_overlap_analysis.md").write_text("\n".join(lines))

    # Short console summary
    print("# catalogued:", len(catalogued), "skipped:", len(skipped))
    for ds in DATASETS:
        s = per_dataset_summary[ds]
        print(f"{ds}: archs={s['n_arch']} assessed={s.get('n_questions_assessed')} "
              f"stubborn={s.get('stubborn')} easy={s.get('easy')}")


if __name__ == "__main__":
    main()
