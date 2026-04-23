"""Evaluate the confidence-gated conditional channel overlay v2.

Baselines (shown but not re-trained):
  meta_v2f
  gated_threshold_0.7  (gated v1 primary, from gated_overlay.py)
  intent_parser_full   (stacked signal addition, from intent_parser.py)

v2 variants:
  gated_v2_all
  gated_v2_intent_only
  gated_v2_minus_critical

Datasets: LoCoMo-30 and Synthetic-19 at K=20 and K=50.

Fair-backfill from raw cosine at each K.

Outputs:
  results/gated_v2.json
  results/gated_v2.md

Usage:
    uv run python gatedv2_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from associative_recall import Segment, SegmentStore
from best_shot import MetaV2f
from gated_overlay import build_variant as build_gated_v1
from intent_parser import build_variant as build_intent_variant
from gated_v2 import (
    SUPPLEMENT_DESCRIPTIONS_V2,
    SUPPLEMENT_NAMES_V2,
    VARIANTS_V2,
    K_EXPANDER_NAME,
    build_variant_v2,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50]

# Baselines (not v2 variants, but reported alongside).
BASELINES_GATED_V1 = ("gated_threshold_0.7",)
BASELINES_INTENT = ("intent_parser_full",)

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


# ---------------------------------------------------------------------------
# Per-question evaluators
# ---------------------------------------------------------------------------
def evaluate_per_k_question(arch, question: dict, k_dependent: bool) -> dict:
    """Generic evaluator. If k_dependent=True, run retrieve(K) for each K;
    otherwise call retrieve(question, conv_id) once and trim."""
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()

    results_per_k: dict[int, list[Segment]] = {}
    meta = None
    if k_dependent:
        meta_lastK = None
        for K in BUDGETS:
            res = arch.retrieve(q_text, conv_id, K=K)
            results_per_k[K] = list(res.segments)
            meta_lastK = res.metadata
        meta = meta_lastK
    else:
        res = arch.retrieve(q_text, conv_id)
        all_segs = list(res.segments)
        for K in BUDGETS:
            results_per_k[K] = all_segs  # (will be trimmed by fair_backfill)
        meta = res.metadata

    elapsed = time.time() - t0

    # Cosine baseline for fair-backfill
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
        "total_arch_retrieved": len(results_per_k[max_K]),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "metadata": meta or {},
    }

    for K in BUDGETS:
        arch_segments = results_per_k[K]
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


def evaluate_metav2f_question(arch, question: dict) -> dict:
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
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "metadata": {"name": "meta_v2f"},
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


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
def summarize(
    results: list[dict], arch_name: str, dataset: str
) -> dict:
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


def per_question_wtl(
    v2_rows: list[dict], ref_rows: list[dict], K: int
) -> dict:
    """W/T/L of v2 architecture vs a reference arch (e.g. gated v1 or
    intent_parser_full) at budget K keyed by question_index."""
    ref_map = {r["question_index"]: r for r in ref_rows}
    wins = losses = ties = 0
    win_examples: list[dict] = []
    loss_examples: list[dict] = []
    for gr in v2_rows:
        qi = gr["question_index"]
        vr = ref_map.get(qi)
        if not vr:
            continue
        g_rec = gr["fair_backfill"][f"arch_r@{K}"]
        v_rec = vr["fair_backfill"][f"arch_r@{K}"]
        if g_rec > v_rec + 0.001:
            wins += 1
            if len(win_examples) < 3:
                win_examples.append({
                    "question": gr["question"][:140],
                    "category": gr["category"],
                    "ref_recall": v_rec,
                    "v2_recall": g_rec,
                    "firing": gr["metadata"].get("firing_channels", []),
                    "confidences": gr["metadata"].get("confidences", {}),
                    "overlay": gr["metadata"].get("overlay", {}),
                    "list_expander_on":
                        gr["metadata"].get("list_expander_on", False),
                })
        elif v_rec > g_rec + 0.001:
            losses += 1
            if len(loss_examples) < 3:
                loss_examples.append({
                    "question": gr["question"][:140],
                    "category": gr["category"],
                    "ref_recall": v_rec,
                    "v2_recall": g_rec,
                    "firing": gr["metadata"].get("firing_channels", []),
                    "confidences": gr["metadata"].get("confidences", {}),
                    "overlay": gr["metadata"].get("overlay", {}),
                    "list_expander_on":
                        gr["metadata"].get("list_expander_on", False),
                })
        else:
            ties += 1
    return {
        "wins": wins, "ties": ties, "losses": losses,
        "win_examples": win_examples,
        "loss_examples": loss_examples,
    }


def summarize_firing_v2(rows: list[dict]) -> dict:
    """Aggregate channel-fire statistics; v2-aware (tracks list_expander)."""
    fire_counts: dict[str, int] = {ch: 0 for ch in SUPPLEMENT_NAMES_V2}
    contrib_counts: dict[str, int] = {ch: 0 for ch in SUPPLEMENT_NAMES_V2}
    conf_sums: dict[str, float] = {
        ch: 0.0 for ch in list(SUPPLEMENT_NAMES_V2) + [K_EXPANDER_NAME]
    }
    list_expander_fire = 0
    per_q_firing_counts: list[int] = []
    n = 0
    for r in rows:
        meta = r.get("metadata", {}) or {}
        firing = meta.get("firing_channels", []) or []
        confs = meta.get("confidences", {}) or {}
        overlay = meta.get("overlay", {}) or {}
        contribs = overlay.get("channels_contributing", []) or []
        per_q_firing_counts.append(len(firing))
        for ch in firing:
            if ch in fire_counts:
                fire_counts[ch] += 1
        for ch in contribs:
            if ch in contrib_counts:
                contrib_counts[ch] += 1
        for ch in conf_sums:
            conf_sums[ch] += float(confs.get(ch, 0.0))
        if meta.get("list_expander_on"):
            list_expander_fire += 1
        n += 1

    dist: dict[int, int] = defaultdict(int)
    for c in per_q_firing_counts:
        dist[c] += 1
    firing_dist = {str(k): dist[k] for k in sorted(dist)}

    return {
        "n": n,
        "avg_firing_per_query": round(
            sum(per_q_firing_counts) / max(n, 1), 3
        ),
        "firing_distribution": firing_dist,
        "fire_rate_per_channel": {
            ch: round(fire_counts[ch] / max(n, 1), 3)
            for ch in SUPPLEMENT_NAMES_V2
        },
        "contribution_rate_per_channel": {
            ch: round(contrib_counts[ch] / max(n, 1), 3)
            for ch in SUPPLEMENT_NAMES_V2
        },
        "avg_confidence_per_channel": {
            ch: round(conf_sums[ch] / max(n, 1), 3)
            for ch in list(SUPPLEMENT_NAMES_V2) + [K_EXPANDER_NAME]
        },
        "list_expander_fire_rate": round(list_expander_fire / max(n, 1), 3),
    }


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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_arch_on_dataset(
    arch_name: str, store: SegmentStore, questions: list[dict]
) -> tuple[list[dict], dict, dict]:
    print(
        f"\n{'=' * 70}\n{arch_name} | {len(questions)} questions\n"
        f"{'=' * 70}"
    )
    if arch_name == "meta_v2f":
        arch = MetaV2f(store)
        evaluator = evaluate_metav2f_question
        k_dependent = False
    elif arch_name in VARIANTS_V2:
        arch = build_variant_v2(arch_name, store)
        evaluator = None  # handled below
        k_dependent = True
    elif arch_name in BASELINES_GATED_V1:
        arch = build_gated_v1(arch_name, store)
        evaluator = None
        k_dependent = True
    elif arch_name in BASELINES_INTENT:
        arch = build_intent_variant(arch_name, store)
        evaluator = None
        k_dependent = False
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
            if arch_name == "meta_v2f":
                row = evaluate_metav2f_question(arch, q)
            else:
                row = evaluate_per_k_question(arch, q, k_dependent=k_dependent)
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
# Reporting
# ---------------------------------------------------------------------------
def render_report(all_data: dict) -> str:
    lines: list[str] = []
    lines.append("# Gated Overlay v2\n")
    lines.append(
        "v2f-primary retrieval with confidence-gated supplement "
        "displacement (same mechanism as gated_overlay v1). v2 extends v1 "
        "with three new supplement channels (preference_markers, "
        "negation_markers, date_answer_boost) and a K-expansion signal "
        "(list_aggregation) derived from intent_parser's unique signal "
        "coverage.\n"
    )

    lines.append("\n## Channel catalog\n")
    for ch in SUPPLEMENT_NAMES_V2:
        lines.append(f"- **{ch}** - {SUPPLEMENT_DESCRIPTIONS_V2[ch]}")
    lines.append(
        f"- **{K_EXPANDER_NAME}** (K-expander) - "
        f"{SUPPLEMENT_DESCRIPTIONS_V2[K_EXPANDER_NAME]}"
    )
    lines.append("")

    arch_order = (
        ["meta_v2f"]
        + list(BASELINES_GATED_V1)
        + list(BASELINES_INTENT)
        + list(VARIANTS_V2)
    )

    lines.append("\n## Recall matrix (fair-backfill)\n")
    lines.append(
        "| Architecture | Dataset | base r@20 | arch r@20 | Δ@20 | "
        "base r@50 | arch r@50 | Δ@50 | avg LLM | avg embed |"
    )
    lines.append("|" + ("---|" * 10))
    for arch_name in arch_order:
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

    # v2 vs each baseline
    for ref in ["meta_v2f"] + list(BASELINES_GATED_V1) + list(BASELINES_INTENT):
        lines.append(f"\n## W/T/L of v2 variants vs {ref} (per-question)\n")
        lines.append("| v2 variant | Dataset | W/T/L @20 | W/T/L @50 |")
        lines.append("|---|---|---|---|")
        for v in VARIANTS_V2:
            for ds in DATASETS:
                entry = all_data.get(v, {}).get(ds, {})
                wtl_by_ref = entry.get("wtl", {}).get(ref, {})
                w20 = wtl_by_ref.get("K20", {})
                w50 = wtl_by_ref.get("K50", {})
                if not w20 and not w50:
                    continue
                lines.append(
                    f"| {v} | {ds} | "
                    f"{w20.get('wins', '-')}/{w20.get('ties', '-')}/"
                    f"{w20.get('losses', '-')} | "
                    f"{w50.get('wins', '-')}/{w50.get('ties', '-')}/"
                    f"{w50.get('losses', '-')} |"
                )
        lines.append("")

    # Firing stats for v2 variants
    for v in VARIANTS_V2:
        for ds in DATASETS:
            fp = all_data.get(v, {}).get(ds, {}).get("firing_stats")
            if not fp:
                continue
            lines.append(f"\n### Firing stats: {v} / {ds}\n")
            lines.append(
                f"n={fp['n']}; avg firing channels/query: "
                f"{fp['avg_firing_per_query']:.2f}; "
                f"list_expander fire rate: "
                f"{fp['list_expander_fire_rate']:.2f}\n"
            )
            lines.append("Firing-count distribution (channels per query):")
            for k, c in sorted(
                fp["firing_distribution"].items(), key=lambda x: int(x[0])
            ):
                lines.append(f"- {k} channels: {c}")
            lines.append("")
            lines.append(
                "| channel | avg confidence | fire rate | contribution "
                "rate |"
            )
            lines.append("|---|---|---|---|")
            for ch in SUPPLEMENT_NAMES_V2:
                lines.append(
                    f"| {ch} | "
                    f"{fp['avg_confidence_per_channel'][ch]:.3f} | "
                    f"{fp['fire_rate_per_channel'][ch]:.2f} | "
                    f"{fp['contribution_rate_per_channel'][ch]:.2f} |"
                )
            lines.append(
                f"| {K_EXPANDER_NAME} | "
                f"{fp['avg_confidence_per_channel'][K_EXPANDER_NAME]:.3f} | "
                f"{fp['list_expander_fire_rate']:.2f} | "
                f"(K-expander, no displacement) |"
            )
            lines.append("")

    # Sample wins/losses for primary v2 vs gated v1
    primary = "gated_v2_all"
    ref_primary = "gated_threshold_0.7"
    for ds in DATASETS:
        entry = all_data.get(primary, {}).get(ds, {})
        wtl50 = entry.get("wtl", {}).get(ref_primary, {}).get("K50", {})
        if not wtl50:
            continue
        lines.append(
            f"\n### Sample W/L ({primary} vs {ref_primary} @ K=50, {ds})\n"
        )
        lines.append("Wins:")
        for ex in wtl50.get("win_examples", [])[:3]:
            lines.append(f"- **[{ex['category']}]** {ex['question']}")
            lines.append(
                f"  - ref={ex['ref_recall']:.3f} -> v2="
                f"{ex['v2_recall']:.3f}"
            )
            lines.append(f"  - fired: {ex['firing']}")
            lines.append(
                f"  - contributed: "
                f"{ex.get('overlay', {}).get('channels_contributing', [])}"
            )
            lines.append(
                f"  - list_expander: {ex.get('list_expander_on', False)}"
            )
        lines.append("\nLosses:")
        for ex in wtl50.get("loss_examples", [])[:3]:
            lines.append(f"- **[{ex['category']}]** {ex['question']}")
            lines.append(
                f"  - ref={ex['ref_recall']:.3f} -> v2="
                f"{ex['v2_recall']:.3f}"
            )
            lines.append(f"  - fired: {ex['firing']}")
            lines.append(
                f"  - contributed: "
                f"{ex.get('overlay', {}).get('channels_contributing', [])}"
            )
            lines.append(
                f"  - list_expander: {ex.get('list_expander_on', False)}"
            )
        lines.append("")

    # Verdict
    lines.append("\n## Verdict\n")
    for ds in DATASETS:
        ref_v2f = all_data.get("meta_v2f", {}).get(ds, {}).get("summary", {})
        ref_g1 = (
            all_data.get("gated_threshold_0.7", {}).get(ds, {}).get("summary",
                                                                    {})
        )
        ref_ip = (
            all_data.get("intent_parser_full", {}).get(ds, {}).get("summary",
                                                                   {})
        )
        if not ref_v2f:
            continue
        lines.append(f"### {ds}\n")
        for K in BUDGETS:
            v2f_score = ref_v2f.get(f"arch_r@{K}", 0.0)
            g1_score = ref_g1.get(f"arch_r@{K}", 0.0)
            ip_score = ref_ip.get(f"arch_r@{K}", 0.0)
            line = (
                f"- **K={K}**: meta_v2f={v2f_score:.4f}, "
                f"gated_threshold_0.7={g1_score:.4f}, "
                f"intent_parser_full={ip_score:.4f}"
            )
            for v in VARIANTS_V2:
                s = all_data.get(v, {}).get(ds, {}).get("summary", {})
                if not s:
                    continue
                g = s.get(f"arch_r@{K}", 0.0)
                line += (
                    f"; {v}={g:.4f} "
                    f"(vs v1 {g - g1_score:+.4f}, "
                    f"vs ip {g - ip_score:+.4f})"
                )
            lines.append(line)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_data: dict = {}

    arch_order = (
        ["meta_v2f"]
        + list(BASELINES_GATED_V1)
        + list(BASELINES_INTENT)
        + list(VARIANTS_V2)
    )

    for ds_name in DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\n\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        # meta_v2f is the reference for wtl comparisons.
        v2f_rows = None
        g1_rows = None
        ip_rows = None

        for arch_name in arch_order:
            rows, summary, by_cat = run_arch_on_dataset(
                arch_name, store, questions
            )
            entry: dict = {
                "arch": arch_name,
                "dataset": ds_name,
                "summary": summary,
                "category_breakdown": by_cat,
                "results": rows,
            }
            if arch_name == "meta_v2f":
                v2f_rows = rows
            if arch_name in BASELINES_GATED_V1:
                g1_rows = rows
            if arch_name in BASELINES_INTENT:
                ip_rows = rows
            if arch_name in VARIANTS_V2:
                entry["firing_stats"] = summarize_firing_v2(rows)
                wtl_dict: dict[str, dict] = {}
                if v2f_rows is not None:
                    wtl_dict["meta_v2f"] = {
                        "K20": per_question_wtl(rows, v2f_rows, 20),
                        "K50": per_question_wtl(rows, v2f_rows, 50),
                    }
                if g1_rows is not None:
                    wtl_dict["gated_threshold_0.7"] = {
                        "K20": per_question_wtl(rows, g1_rows, 20),
                        "K50": per_question_wtl(rows, g1_rows, 50),
                    }
                if ip_rows is not None:
                    wtl_dict["intent_parser_full"] = {
                        "K20": per_question_wtl(rows, ip_rows, 20),
                        "K50": per_question_wtl(rows, ip_rows, 50),
                    }
                entry["wtl"] = wtl_dict
            all_data.setdefault(arch_name, {})[ds_name] = entry

    raw_out = RESULTS_DIR / "gated_v2.json"
    with open(raw_out, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nRaw saved: {raw_out}")

    report = render_report(all_data)
    md_out = RESULTS_DIR / "gated_v2.md"
    with open(md_out, "w") as f:
        f.write(report)
    print(f"Report saved: {md_out}")


if __name__ == "__main__":
    main()
