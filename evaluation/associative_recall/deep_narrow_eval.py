"""Evaluate DeepNarrow variants on LoCoMo (30q) and synthetic (19q).

Compares against cached meta_v2f (from fair_backfill_eval.py outputs) and
chain_with_scratchpad (from goal_chain) as baselines.

Writes:
  results/deep_narrow_study.json - raw numbers
  results/deep_narrow_study.md   - tables, per-category, verdict

Usage:
    uv run python deep_narrow_eval.py [--variants v1,wide_probe,no_stop]
        [--max-questions N] [--datasets locomo,synthetic]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from associative_recall import Segment, SegmentStore
from deep_narrow import (
    DeepNarrowBase,
    DeepNarrowNoStop,
    DeepNarrowV1,
    DeepNarrowWideProbe,
)
from dotenv import load_dotenv

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

VARIANTS = {
    "deep_narrow_v1": DeepNarrowV1,
    "deep_narrow_wide_probe": DeepNarrowWideProbe,
    "deep_narrow_no_stop": DeepNarrowNoStop,
}


# ---------------------------------------------------------------------------
# Fair-backfill eval helpers (mirror fair_backfill_eval.py)
# ---------------------------------------------------------------------------
def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill_eval(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    source_ids: set[int],
    budget: int,
) -> tuple[float, float]:
    """Return (baseline_recall, arch_recall) at budget K with fair backfill."""
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


def arch_at_k_turn_ids(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    budget: int,
) -> set[int]:
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
    return {s.turn_id for s in arch_at_K}


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


def evaluate_question(arch: DeepNarrowBase, question: dict) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedup arch segments in order.
    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for s in result.segments:
        if s.index not in seen:
            arch_segments.append(s)
            seen.add(s.index)

    # Cosine top-max(BUDGETS) baseline.
    q_emb = arch.embed_text(q_text)
    max_K = max(BUDGETS)
    cosine_result = arch.store.search(q_emb, top_k=max_K, conversation_id=conv_id)
    cosine_segments = list(cosine_result.segments)

    fair_backfill = {}
    for K in BUDGETS:
        b_r, a_r = fair_backfill_eval(arch_segments, cosine_segments, source_ids, K)
        fair_backfill[f"baseline_r@{K}"] = round(b_r, 4)
        fair_backfill[f"arch_r@{K}"] = round(a_r, 4)
        fair_backfill[f"delta_r@{K}"] = round(a_r - b_r, 4)

    # Per-hop cumulative recall curve — compute recall at each hop-end.
    hop_records = result.metadata.get("hop_records", [])
    # Reconstruct cumulative arch pool after each hop
    # by iterating and rebuilding: we don't have per-hop snapshots, but we
    # can use hop_records.total_after as the count, and segment order to
    # approximate (segments are appended in hop order).
    per_hop_recall = []
    cum_count = len(cosine_result.segments)  # unused, just a placeholder
    # Walk arch_segments in append order; each hop adds N segments.
    # Build cumulative pool at each hop.
    running_pool: list[Segment] = []
    seg_iter = iter(arch_segments)
    # Initial: initial_k segments (hop 0, though not in hop_records).
    initial_n = arch.initial_k
    for _ in range(min(initial_n, len(arch_segments))):
        try:
            running_pool.append(next(seg_iter))
        except StopIteration:
            break
    # Record after initial retrieval (hop 0)
    K20_ids = arch_at_k_turn_ids(running_pool, cosine_segments, 20)
    K50_ids = arch_at_k_turn_ids(running_pool, cosine_segments, 50)
    per_hop_recall.append(
        {
            "hop": 0,
            "pool_size": len(running_pool),
            "r@20": round(compute_recall(K20_ids, source_ids), 4),
            "r@50": round(compute_recall(K50_ids, source_ids), 4),
        }
    )

    for rec in hop_records:
        nf = rec.get("new_found", 0) if isinstance(rec, dict) else rec.new_found
        for _ in range(nf):
            try:
                running_pool.append(next(seg_iter))
            except StopIteration:
                break
        K20_ids = arch_at_k_turn_ids(running_pool, cosine_segments, 20)
        K50_ids = arch_at_k_turn_ids(running_pool, cosine_segments, 50)
        per_hop_recall.append(
            {
                "hop": rec["hop"] if isinstance(rec, dict) else rec.hop,
                "pool_size": len(running_pool),
                "r@20": round(compute_recall(K20_ids, source_ids), 4),
                "r@50": round(compute_recall(K50_ids, source_ids), 4),
            }
        )

    # Saturation & hop stats.
    hops_with_new = sum(
        1
        for h in hop_records
        if (h.get("new_found", 0) if isinstance(h, dict) else h.new_found) > 0
        and not (h.get("stopped", False) if isinstance(h, dict) else h.stopped)
    )
    hops_with_cue = sum(
        1 for h in hop_records if (h.get("cue", "") if isinstance(h, dict) else h.cue)
    )
    saturation_rate = (
        (hops_with_cue - hops_with_new) / hops_with_cue if hops_with_cue > 0 else 0.0
    )

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
        "fair_backfill": fair_backfill,
        "hops_used": result.metadata.get("hops_used", 0),
        "hit_max_hops": result.metadata.get("hit_max_hops", False),
        "stop_reason": result.metadata.get("stop_reason", ""),
        "saturation_rate": round(saturation_rate, 4),
        "per_hop_recall": per_hop_recall,
    }


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
    summary["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
    summary["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 1)
    summary["avg_hops_used"] = round(sum(r["hops_used"] for r in results) / n, 2)
    summary["pct_hit_max_hops"] = round(
        100.0 * sum(1 for r in results if r["hit_max_hops"]) / n, 1
    )
    summary["avg_saturation_rate"] = round(
        sum(r["saturation_rate"] for r in results) / n, 3
    )
    stop_counts: dict[str, int] = defaultdict(int)
    for r in results:
        reason = r.get("stop_reason", "")
        # Normalize: group "self-stop:*" and "saturation:*" and "max_hops"
        if reason.startswith("self-stop"):
            stop_counts["self-stop"] += 1
        elif reason.startswith("saturation"):
            stop_counts["saturation"] += 1
        elif reason.startswith("max_hops"):
            stop_counts["max_hops"] += 1
        elif reason.startswith("duplicate"):
            stop_counts["duplicate_cue"] += 1
        elif reason.startswith("segment cap"):
            stop_counts["segment_cap"] += 1
        elif reason.startswith("no cue"):
            stop_counts["no_cue"] += 1
        else:
            stop_counts["other"] += 1
    summary["stop_reason_counts"] = dict(stop_counts)
    return summary


def summarize_by_category(results: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry: dict = {"n": n}
        for K in BUDGETS:
            b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in rs]
            a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in rs]
            entry[f"baseline_r@{K}"] = round(sum(b_vals) / n, 4)
            entry[f"arch_r@{K}"] = round(sum(a_vals) / n, 4)
            entry[f"delta_r@{K}"] = round(
                entry[f"arch_r@{K}"] - entry[f"baseline_r@{K}"], 4
            )
        entry["avg_hops_used"] = round(sum(r["hops_used"] for r in rs) / n, 2)
        out[cat] = entry
    return out


# ---------------------------------------------------------------------------
# Aggregation from cached baseline files
# ---------------------------------------------------------------------------
def load_cached_baseline(arch_name: str, dataset: str) -> dict | None:
    """Load a cached summary + category_breakdown from prior fair-backfill runs."""
    candidates = [
        RESULTS_DIR / f"fairbackfill_{arch_name}_{dataset}.json",
        RESULTS_DIR / f"goal_chain_{arch_name}_{dataset}.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            results = d.get("results", [])
            # Normalize: get per-question (baseline_r@K, arch_r@K, category, llm_calls)
            qs = []
            for r in results:
                fb = r.get("fair_backfill", {})
                qs.append(
                    {
                        "category": r.get("category", "unknown"),
                        "conversation_id": r.get("conversation_id", ""),
                        "question_index": r.get("question_index", -1),
                        "num_source_turns": r.get("num_source_turns", 0),
                        "baseline_r@20": fb.get("baseline_r@20", 0.0),
                        "arch_r@20": fb.get("arch_r@20", 0.0),
                        "baseline_r@50": fb.get("baseline_r@50", 0.0),
                        "arch_r@50": fb.get("arch_r@50", 0.0),
                        "llm_calls": r.get("llm_calls", 0),
                        "embed_calls": r.get("embed_calls", 0),
                    }
                )
            return {
                "summary": d.get("summary", {}),
                "category_breakdown": d.get("category_breakdown", {}),
                "per_q": qs,
            }
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_variant(
    variant_name: str,
    arch_cls,
    dataset: str,
    store: SegmentStore,
    questions: list[dict],
    max_questions: int | None = None,
    verbose: bool = True,
) -> tuple[list[dict], dict, dict]:
    print(f"\n{'=' * 70}")
    print(f"{variant_name} | {dataset} | {len(questions)} questions")
    print(f"{'=' * 70}", flush=True)

    arch = arch_cls(store)
    results: list[dict] = []
    qs = questions if max_questions is None else questions[:max_questions]
    for i, q in enumerate(qs):
        q_short = q["question"][:55]
        print(
            f"  [{i + 1}/{len(qs)}] {q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(arch, q)
            results.append(row)
            if verbose:
                fb = row["fair_backfill"]
                print(
                    f"    hops={row['hops_used']} "
                    f"pool={row['total_arch_retrieved']} "
                    f"r@20 b={fb['baseline_r@20']:.3f} a={fb['arch_r@20']:.3f} "
                    f"r@50 b={fb['baseline_r@50']:.3f} a={fb['arch_r@50']:.3f} "
                    f"llm={row['llm_calls']} stop={row['stop_reason'][:30]}",
                    flush=True,
                )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 3 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = summarize(results, variant_name, dataset)
    by_cat = summarize_by_category(results)

    print(f"\n--- {variant_name} on {dataset} ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: baseline={summary[f'baseline_r@{K}']:.3f} "
            f"arch={summary[f'arch_r@{K}']:.3f} "
            f"delta={summary[f'delta_r@{K}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    print(
        f"  avg hops={summary['avg_hops_used']:.2f} "
        f"pct_hit_max={summary['pct_hit_max_hops']:.1f}% "
        f"sat_rate={summary['avg_saturation_rate']:.3f} "
        f"llm={summary['avg_llm_calls']:.1f} "
        f"embed={summary['avg_embed_calls']:.1f}"
    )
    print(f"  stop reasons: {summary['stop_reason_counts']}")
    return results, summary, by_cat


def make_report(
    per_variant_dataset: dict,
    cached_baselines: dict,
) -> str:
    """per_variant_dataset: {variant: {dataset: {summary, by_cat, results}}}
    cached_baselines: {arch_name: {dataset: {summary, category_breakdown, per_q}}}
    """
    lines: list[str] = []
    lines.append("# Deep-Narrow Iterative Retrieval Study")
    lines.append("")
    lines.append(
        "Tests a 15-20 hop x 1 cue per hop retrieval architecture "
        "(untested shape) against v2f (shallow, 1x3 cues) and "
        "chain_with_scratchpad (3-5 hops x 1 cue)."
    )
    lines.append("")
    lines.append("## Headline: fair-backfill recall")
    lines.append("")
    lines.append(
        "| Variant | Dataset | n | base@20 | arch@20 | d@20 | base@50 | arch@50 | d@50 | avg LLM |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    all_rows: list[tuple[str, str, dict]] = []
    # Cached baselines first
    for bname, per_ds in cached_baselines.items():
        for ds_name, payload in per_ds.items():
            s = payload.get("summary", {}) or {}
            if not s:
                continue
            all_rows.append((bname, ds_name, s))
    # Deep-narrow variants
    for v_name, per_ds in per_variant_dataset.items():
        for ds_name, payload in per_ds.items():
            s = payload["summary"]
            all_rows.append((v_name, ds_name, s))

    for name, ds, s in all_rows:

        def g(k, default=0.0):
            return s.get(k, default)

        lines.append(
            f"| {name} | {ds} | {s.get('n', 0)} | "
            f"{g('baseline_r@20', 0):.3f} | {g('arch_r@20', 0):.3f} | "
            f"{g('delta_r@20', 0):+.3f} | "
            f"{g('baseline_r@50', 0):.3f} | {g('arch_r@50', 0):.3f} | "
            f"{g('delta_r@50', 0):+.3f} | "
            f"{g('avg_llm_calls', 0)} |"
        )

    lines.append("")
    lines.append("## Hop behavior (deep-narrow only)")
    lines.append("")
    lines.append(
        "| Variant | Dataset | avg hops | % hit max_hops | sat rate | stop reasons |"
    )
    lines.append("|---|---|---|---|---|---|")
    for v_name, per_ds in per_variant_dataset.items():
        for ds_name, payload in per_ds.items():
            s = payload["summary"]
            sr = s.get("stop_reason_counts", {})
            sr_str = ", ".join(f"{k}={v}" for k, v in sr.items())
            lines.append(
                f"| {v_name} | {ds_name} | "
                f"{s['avg_hops_used']:.2f} | "
                f"{s['pct_hit_max_hops']:.1f}% | "
                f"{s['avg_saturation_rate']:.3f} | "
                f"{sr_str} |"
            )

    # Per-category comparison: deep_narrow_v1 vs meta_v2f
    lines.append("")
    lines.append("## Per-category recall (deep_narrow_v1 vs meta_v2f)")
    lines.append("")
    for ds_name in ["locomo_30q", "synthetic_19q"]:
        if "deep_narrow_v1" not in per_variant_dataset:
            continue
        if ds_name not in per_variant_dataset.get("deep_narrow_v1", {}):
            continue
        by_cat_dn = per_variant_dataset["deep_narrow_v1"][ds_name]["by_cat"]
        by_cat_v2f = (
            cached_baselines.get("meta_v2f", {})
            .get(ds_name, {})
            .get("category_breakdown", {})
        )
        lines.append(f"### {ds_name}")
        lines.append("")
        lines.append(
            "| Category | n | v2f arch@20 | DN arch@20 | DN-v2f | v2f arch@50 | DN arch@50 | DN-v2f@50 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        all_cats = set(by_cat_dn) | set(by_cat_v2f)
        for cat in sorted(all_cats):
            dn = by_cat_dn.get(cat, {})
            v2f = by_cat_v2f.get(cat, {})
            n = dn.get("n") or v2f.get("n", 0)
            v_a20 = v2f.get("arch_r@20", 0.0)
            d_a20 = dn.get("arch_r@20", 0.0)
            v_a50 = v2f.get("arch_r@50", 0.0)
            d_a50 = dn.get("arch_r@50", 0.0)
            lines.append(
                f"| {cat} | {n} | "
                f"{v_a20:.3f} | {d_a20:.3f} | {d_a20 - v_a20:+.3f} | "
                f"{v_a50:.3f} | {d_a50:.3f} | {d_a50 - v_a50:+.3f} |"
            )
        lines.append("")

    # Recall-vs-hop curve (aggregated across dataset)
    lines.append("## Recall vs hop curve (deep_narrow_v1)")
    lines.append("")
    for ds_name, payload in per_variant_dataset.get("deep_narrow_v1", {}).items():
        results = payload["results"]
        max_hop = max(
            (max((p["hop"] for p in r["per_hop_recall"]), default=0) for r in results),
            default=0,
        )
        hop_table: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for r in results:
            seen_hops: set[int] = set()
            for p in r["per_hop_recall"]:
                hop_table[p["hop"]].append((p["r@20"], p["r@50"]))
                seen_hops.add(p["hop"])
            # Forward-fill: for hops past the last recorded one, keep last values
            last = r["per_hop_recall"][-1] if r["per_hop_recall"] else None
            if last:
                for h in range(last["hop"] + 1, max_hop + 1):
                    hop_table[h].append((last["r@20"], last["r@50"]))
        lines.append(f"### {ds_name}")
        lines.append("")
        lines.append("| hop | avg r@20 | avg r@50 | n_questions |")
        lines.append("|---|---|---|---|")
        for h in sorted(hop_table):
            pairs = hop_table[h]
            if not pairs:
                continue
            r20 = sum(p[0] for p in pairs) / len(pairs)
            r50 = sum(p[1] for p in pairs) / len(pairs)
            lines.append(f"| {h} | {r20:.4f} | {r50:.4f} | {len(pairs)} |")
        lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    # Compute the dominant comparison: deep_narrow_v1 vs meta_v2f on locomo
    v2f_loc = (
        cached_baselines.get("meta_v2f", {}).get("locomo_30q", {}).get("summary", {})
    )
    dn_loc = (
        per_variant_dataset.get("deep_narrow_v1", {})
        .get("locomo_30q", {})
        .get("summary", {})
    )
    v2f_syn = (
        cached_baselines.get("meta_v2f", {}).get("synthetic_19q", {}).get("summary", {})
    )
    dn_syn = (
        per_variant_dataset.get("deep_narrow_v1", {})
        .get("synthetic_19q", {})
        .get("summary", {})
    )

    def _delta(dn_s: dict, v2f_s: dict, k: str) -> float:
        return dn_s.get(k, 0.0) - v2f_s.get(k, 0.0)

    if dn_loc and v2f_loc:
        lines.append(
            f"- LoCoMo d(DN-v2f) r@20: {_delta(dn_loc, v2f_loc, 'arch_r@20'):+.4f}, "
            f"r@50: {_delta(dn_loc, v2f_loc, 'arch_r@50'):+.4f}. "
            f"LLM calls: DN={dn_loc.get('avg_llm_calls', 0):.1f} "
            f"vs v2f={v2f_loc.get('avg_llm_calls', 0):.1f}."
        )
    if dn_syn and v2f_syn:
        lines.append(
            f"- Synthetic d(DN-v2f) r@20: {_delta(dn_syn, v2f_syn, 'arch_r@20'):+.4f}, "
            f"r@50: {_delta(dn_syn, v2f_syn, 'arch_r@50'):+.4f}. "
            f"LLM calls: DN={dn_syn.get('avg_llm_calls', 0):.1f} "
            f"vs v2f={v2f_syn.get('avg_llm_calls', 0):.1f}."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        type=str,
        default="deep_narrow_v1,deep_narrow_wide_probe",
        help="Comma-separated variant names.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="locomo_30q,synthetic_19q",
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--early-stop-after",
        type=int,
        default=None,
        help="After this many q on LoCoMo, check if DN<<v2f "
        "and stop if so (decision rule).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # Load cached baselines.
    cached_baselines: dict[str, dict] = {}
    for b in ("meta_v2f", "chain_with_scratchpad", "v15_control"):
        cached_baselines[b] = {}
        for ds in dataset_names:
            cached = load_cached_baseline(b, ds)
            if cached:
                cached_baselines[b][ds] = cached

    per_variant_dataset: dict = {}

    for v_name in variant_names:
        if v_name not in VARIANTS:
            print(f"Unknown variant: {v_name}")
            continue
        per_variant_dataset[v_name] = {}
        for ds_name in dataset_names:
            out_file = RESULTS_DIR / f"deep_narrow_{v_name}_{ds_name}.json"
            if out_file.exists() and not args.force:
                with open(out_file) as f:
                    d = json.load(f)
                per_variant_dataset[v_name][ds_name] = {
                    "summary": d["summary"],
                    "by_cat": d.get("category_breakdown", {}),
                    "results": d.get("results", []),
                }
                print(f"[cached] {v_name} on {ds_name}")
                continue

            store, questions = load_dataset(ds_name)
            results, summary, by_cat = run_variant(
                v_name,
                VARIANTS[v_name],
                ds_name,
                store,
                questions,
                max_questions=args.max_questions,
                verbose=True,
            )
            per_variant_dataset[v_name][ds_name] = {
                "summary": summary,
                "by_cat": by_cat,
                "results": results,
            }
            with open(out_file, "w") as f:
                json.dump(
                    {
                        "arch": v_name,
                        "dataset": ds_name,
                        "summary": summary,
                        "category_breakdown": by_cat,
                        "results": results,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            print(f"  Saved: {out_file}")

            # Early-stop decision rule: if DN clearly worse than v2f on LoCoMo,
            # stop further variants/datasets.
            if (
                args.early_stop_after
                and ds_name == "locomo_30q"
                and v_name == "deep_narrow_v1"
                and len(results) >= args.early_stop_after
            ):
                v2f = (
                    cached_baselines.get("meta_v2f", {})
                    .get("locomo_30q", {})
                    .get("summary", {})
                )
                dn_a20 = summary.get("arch_r@20", 0)
                v2f_a20 = v2f.get("arch_r@20", 0)
                if dn_a20 + 0.05 < v2f_a20:
                    print(
                        f"\n*** EARLY STOP: deep_narrow_v1 r@20={dn_a20:.3f} "
                        f"<< meta_v2f r@20={v2f_a20:.3f}. "
                        "Skipping remaining work per decision rule."
                    )
                    # Write report + summary and exit.
                    report = make_report(per_variant_dataset, cached_baselines)
                    (RESULTS_DIR / "deep_narrow_study.md").write_text(report)
                    (RESULTS_DIR / "deep_narrow_study.json").write_text(
                        json.dumps(
                            {
                                "variants": per_variant_dataset,
                                "baselines": cached_baselines,
                            },
                            indent=2,
                            default=str,
                        )
                    )
                    return

    # Write aggregate outputs.
    report = make_report(per_variant_dataset, cached_baselines)
    (RESULTS_DIR / "deep_narrow_study.md").write_text(report)
    print("\n--- deep_narrow_study.md ---")
    print(report)

    (RESULTS_DIR / "deep_narrow_study.json").write_text(
        json.dumps(
            {
                "variants": per_variant_dataset,
                "baselines": cached_baselines,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nSaved: {RESULTS_DIR / 'deep_narrow_study.json'}")
    print(f"Saved: {RESULTS_DIR / 'deep_narrow_study.md'}")


if __name__ == "__main__":
    main()
