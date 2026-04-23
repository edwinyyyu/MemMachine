"""Build the Pareto frontier across all retrieval architectures.

Extracts (arch, dataset, r@20, r@50, llm_calls, embed_calls, total_retrieved)
from every result file in results/, deduplicates, and computes per-dataset
Pareto-dominance relationships (higher recall + lower cost = dominates).

Outputs:
 - results/pareto_frontier.json
 - results/pareto_summary.md
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

RESULTS_DIR = Path(__file__).parent / "results"

# Canonical dataset names (normalize e.g. "advanced" -> "advanced_23q").
DATASET_ALIASES = {
    "advanced": "advanced_23q",
    "advanced_23q": "advanced_23q",
    "locomo": "locomo_30q",
    "locomo_30q": "locomo_30q",
    "locomo_ext_30q": "locomo_30q",
    "locomo_ext_60q": "locomo_60q",
    "puzzle": "puzzle_16q",
    "puzzle_16q": "puzzle_16q",
    "synthetic": "synthetic_19q",
    "synthetic_19q": "synthetic_19q",
    "beam_30q": "beam_30q",
    "beam_ext_30q": "beam_30q",
    "beam_ext_60q": "beam_60q",
    "beam": "beam_30q",
}

DATASETS_OF_INTEREST = {
    "locomo_30q",
    "synthetic_19q",
    "puzzle_16q",
    "advanced_23q",
    "beam_30q",
}


def canonical_dataset(name: str | None) -> str | None:
    if not name:
        return None
    return DATASET_ALIASES.get(name, name)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def safe_num(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


# -----------------------------------------------------------------------------
# Record collection
# -----------------------------------------------------------------------------

Record = dict[str, Any]


class Collector:
    """Collect records keyed by (arch, dataset). Keeps best-populated entry.

    When duplicates appear we prefer the record with the most signal: lower
    llm/embed costs when recall is equivalent, else the one with more fields
    populated. For simplicity, we keep the first entry that has full data and
    skip later duplicates unless the new one has strictly higher recall.
    """

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], Record] = {}

    def add(self, arch: str, dataset: str | None, rec: Record) -> None:
        ds = canonical_dataset(dataset)
        if ds is None or ds not in DATASETS_OF_INTEREST:
            return
        key = (arch, ds)
        if key not in self.records:
            self.records[key] = rec
            return
        existing = self.records[key]
        # Prefer the record that has both r@20 and r@50 filled.
        have_new = rec.get("r@20") is not None and rec.get("r@50") is not None
        have_old = existing.get("r@20") is not None and existing.get("r@50") is not None
        if have_new and not have_old:
            self.records[key] = rec
            return
        if have_old and not have_new:
            return
        # Prefer the one with cost info.
        new_has_cost = rec.get("llm_calls") is not None and rec.get("embed_calls") is not None
        old_has_cost = existing.get("llm_calls") is not None and existing.get("embed_calls") is not None
        if new_has_cost and not old_has_cost:
            self.records[key] = rec
            return
        if old_has_cost and not new_has_cost:
            return
        # Both comparable — keep the one with higher r@20.
        nr = rec.get("r@20") or 0.0
        er = existing.get("r@20") or 0.0
        if nr > er:
            self.records[key] = rec


# -----------------------------------------------------------------------------
# Parsers for each result-file schema
# -----------------------------------------------------------------------------


def parse_summary_block(block: dict, arch_hint: str | None = None, dataset_hint: str | None = None) -> Record | None:
    """Parse a dict that looks like {... 'arch_r@20':..., 'avg_llm_calls':..., ...}."""
    arch = block.get("arch") or block.get("variant") or block.get("arch_name") or arch_hint
    dataset = block.get("dataset") or block.get("benchmark") or dataset_hint
    if not arch:
        return None
    r20 = safe_num(block.get("arch_r@20"))
    r50 = safe_num(block.get("arch_r@50"))
    llm = safe_num(block.get("avg_llm_calls"))
    emb = safe_num(block.get("avg_embed_calls"))
    tot = safe_num(block.get("avg_total_retrieved"))
    if r20 is None and r50 is None:
        return None
    return {
        "arch": arch,
        "dataset": canonical_dataset(dataset),
        "r@20": r20,
        "r@50": r50,
        "llm_calls": llm,
        "embed_calls": emb,
        "total_retrieved": tot,
        "n": block.get("n"),
    }


def parse_fairbackfill(data: Any, collector: Collector, file_stem: str) -> None:
    """fairbackfill_summary.json has shape {arch: {dataset: {summary: {...}}}}."""
    if file_stem == "fairbackfill_summary" and isinstance(data, dict):
        for arch, by_ds in data.items():
            if not isinstance(by_ds, dict):
                continue
            for dataset, payload in by_ds.items():
                if not isinstance(payload, dict):
                    continue
                summary = payload.get("summary") or payload
                rec = parse_summary_block(summary, arch_hint=arch, dataset_hint=dataset)
                if rec:
                    collector.add(f"fairbackfill:{arch}", dataset, rec)
        return

    if file_stem.startswith("fairbackfill_") and isinstance(data, dict):
        summary = data.get("summary") or data
        rec = parse_summary_block(summary)
        if rec:
            collector.add(f"fairbackfill:{rec['arch']}", rec["dataset"], rec)


def parse_bestshot(data: Any, collector: Collector, file_stem: str) -> None:
    if file_stem == "bestshot_all_summaries" and isinstance(data, dict):
        for variant, by_ds in data.items():
            if not isinstance(by_ds, dict):
                continue
            for dataset, block in by_ds.items():
                if isinstance(block, dict):
                    rec = parse_summary_block(block, arch_hint=variant, dataset_hint=dataset)
                    if rec:
                        collector.add(f"bestshot:{variant}", dataset, rec)
        return

    if file_stem.startswith("bestshot_") and isinstance(data, dict):
        summary = data.get("summary") or data
        rec = parse_summary_block(summary)
        if rec:
            collector.add(f"bestshot:{rec['arch']}", rec["dataset"], rec)


def parse_list_of_summary_dicts(data: Any, collector: Collector, prefix: str) -> None:
    """Many files store a list of dicts, one per (arch, benchmark)."""
    if not isinstance(data, list):
        return
    for entry in data:
        if not isinstance(entry, dict):
            continue
        rec = parse_summary_block(entry)
        if rec is None:
            continue
        collector.add(f"{prefix}:{rec['arch']}", rec["dataset"], rec)


def parse_nested_summary_dict(data: Any, collector: Collector, prefix: str) -> None:
    """Files like memindex_summary.json: {variant: {dataset: {...summary keys...}}}."""
    if not isinstance(data, dict):
        return
    for variant, by_ds in data.items():
        if not isinstance(by_ds, dict):
            continue
        for dataset, block in by_ds.items():
            if not isinstance(block, dict):
                continue
            summary = block.get("summary") or block
            rec = parse_summary_block(summary, arch_hint=variant, dataset_hint=dataset)
            if rec:
                collector.add(f"{prefix}:{variant}", dataset, rec)


def parse_budget(data: Any, collector: Collector) -> None:
    """budget_all_summaries.json: {'arch@dataset': {...}}."""
    if not isinstance(data, dict):
        return
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        arch = entry.get("arch", "")
        dataset = entry.get("benchmark") or entry.get("dataset")
        # mean_recall is typically a single K; skip if it's not attached to r@20/r@50.
        # budget files measure "mean_recall at budget K" — treat as r@K where K=budget.
        budget = entry.get("budget")
        mean_recall = safe_num(entry.get("mean_recall"))
        if mean_recall is None:
            continue
        rec = {
            "arch": f"{arch}_b{budget}",
            "dataset": canonical_dataset(dataset),
            "r@20": mean_recall if budget == 20 else None,
            "r@50": mean_recall if budget == 50 else None,
            "r@100": mean_recall if budget == 100 else None,
            "llm_calls": safe_num(entry.get("avg_llm_calls")),
            "embed_calls": safe_num(entry.get("avg_embed_calls")),
            "total_retrieved": safe_num(entry.get("mean_actual_count")),
            "n": entry.get("n"),
        }
        collector.add(f"budget:{arch}_b{budget}", dataset, rec)


# -----------------------------------------------------------------------------
# Main loader
# -----------------------------------------------------------------------------


def collect_all() -> Collector:
    collector = Collector()
    files = sorted(RESULTS_DIR.glob("*.json"))
    for path in files:
        stem = path.stem
        try:
            data = load_json(path)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {path.name}: {exc}")
            continue

        # ----- Summary/aggregate files -----
        if stem == "fairbackfill_summary":
            parse_fairbackfill(data, collector, stem)
            continue
        if stem.startswith("fairbackfill_") and stem != "fairbackfill_summary":
            parse_fairbackfill(data, collector, stem)
            continue
        if stem == "bestshot_all_summaries" or stem.startswith("bestshot_"):
            parse_bestshot(data, collector, stem)
            continue
        if stem in {
            "agent_all_summaries",
            "arch_all_summaries",
            "chain_all_summaries",
            "frontier_all_summaries",
            "meta_all_summaries",
            "optim_all_summaries",
        }:
            prefix = stem.split("_")[0]
            parse_list_of_summary_dicts(data, collector, prefix)
            continue
        if stem == "human_signals_summary":
            parse_nested_summary_dict(data, collector, "human_signals")
            continue
        if stem == "memindex_summary":
            parse_nested_summary_dict(data, collector, "memindex")
            continue
        if stem == "v15_hybrid_summary":
            # This file has per-category breakdowns, no llm_calls. Per-file fallback handles it.
            continue
        if stem == "retlog_summary":
            # Category-level only, no cost info. Handled via per-dataset files.
            continue
        if stem == "budget_all_summaries":
            parse_budget(data, collector)
            continue
        if stem == "query_rewrite_all_summaries":
            if isinstance(data, dict) and "summaries" in data:
                for variant, by_ds in data["summaries"].items():
                    for dataset, block in by_ds.items():
                        if not isinstance(block, dict):
                            continue
                        rec = {
                            "arch": variant,
                            "dataset": canonical_dataset(dataset),
                            "r@20": safe_num(block.get("r@20")),
                            "r@50": safe_num(block.get("r@50")),
                            "llm_calls": safe_num(block.get("avg_llm@20") or block.get("avg_llm@50")),
                            "embed_calls": safe_num(block.get("avg_embed@20") or block.get("avg_embed@50")),
                            "total_retrieved": None,
                            "n": block.get("n"),
                        }
                        if rec["r@20"] is not None or rec["r@50"] is not None:
                            collector.add(f"query_rewrite:{variant}", dataset, rec)
            continue
        if stem == "cot_universal_summary":
            # No cost data in this file; per-dataset cot_ files cover it.
            continue
        if stem == "self_cot_summary" or stem == "self_v2_summary" or stem == "self_v3_summary":
            # Category-level recall-only; per-dataset files have cost info.
            continue
        if stem == "summaries":
            continue
        if stem == "arch_report" or stem == "overnight_summary" or stem == "research_summary":
            continue
        if stem == "constraint_completeness" or stem == "constraint_investigation" or stem == "constraint_procedural":
            continue
        if stem == "adaptive_comparison":
            continue
        if stem == "proactive_summary" or stem == "proactive_detailed":
            continue
        if stem == "error_analysis_details":
            continue
        if stem == "v2f_benchmark_comparison":
            continue

        # ----- Per-dataset files: aggregate from the list of per-question records -----
        if isinstance(data, dict) and "summary" in data and isinstance(data["summary"], dict):
            summary = data["summary"]
            rec = parse_summary_block(summary)
            if rec:
                prefix = stem.split("_")[0]
                collector.add(f"{prefix}:{rec['arch']}", rec["dataset"], rec)
            continue

        # Per-dataset list-of-dicts (one dict per question). Compute means.
        if isinstance(data, list) and data and isinstance(data[0], dict):
            rec = summarize_per_question_list(data, stem)
            if rec:
                prefix = stem.split("_")[0]
                collector.add(f"{prefix}:{rec['arch']}", rec["dataset"], rec)
            continue

    return collector


def _extract_dataset_from_stem(stem: str) -> str | None:
    for ds in ("locomo_30q", "synthetic_19q", "puzzle_16q", "advanced_23q", "beam_30q", "beam_ext_30q"):
        if stem.endswith(ds):
            return canonical_dataset(ds)
    # stem-level heuristics
    for key, canon in DATASET_ALIASES.items():
        if stem.endswith(key):
            return canon
    return None


def _recall_mean(entries: list[dict], recall_key: str) -> float | None:
    """Compute mean r@K across per-question entries.

    Each entry may store recall under different keys depending on the
    architecture.
    """
    values = []
    for e in entries:
        # Preferred nested schemas first
        for parent_key in (
            "arch_recalls",
            "union_recalls",
            "self_v2_recalls",
            "self_v3_recalls",
            "self_cot_recalls",
            "cot_recalls",
            "v2f_recalls",
            "v15_recalls",
            "hybrid_recalls",
            "entity_recalls",
            "memindex_recalls",
            "chain_recalls",
            "tree_recalls",
            "frontier_recalls",
            "agent_recalls",
            "meta_recalls",
            "bestshot_recalls",
            "optim_recalls",
            "adaptive_recalls",
            "normalized_recalls",
            "precision_recalls",
            "supervisor_recalls",
            "full_recalls",
            "lite_recalls",
            "arch",
        ):
            if parent_key in e and isinstance(e[parent_key], dict) and recall_key in e[parent_key]:
                v = safe_num(e[parent_key][recall_key])
                if v is not None:
                    values.append(v)
                    break
        else:
            # Flat keys e.g. "r@20"
            if recall_key in e:
                v = safe_num(e[recall_key])
                if v is not None:
                    values.append(v)
    if not values:
        return None
    return sum(values) / len(values)


def summarize_per_question_list(entries: list[dict], stem: str) -> Record | None:
    """Aggregate a per-question result list into a single summary record."""
    ds = _extract_dataset_from_stem(stem)
    if ds is None:
        return None

    r20 = _recall_mean(entries, "r@20")
    r50 = _recall_mean(entries, "r@50")
    if r20 is None and r50 is None:
        return None

    # Cost
    llm_vals = [safe_num(e.get("llm_calls")) for e in entries]
    llm_vals = [v for v in llm_vals if v is not None]
    emb_vals = [safe_num(e.get("embed_calls")) for e in entries]
    emb_vals = [v for v in emb_vals if v is not None]
    # total retrieved from pool_size or num_cue_segments or similar
    tot_vals = [safe_num(e.get("pool_size") or e.get("num_cue_segments")) for e in entries]
    tot_vals = [v for v in tot_vals if v is not None]

    # Derive architecture name from file stem
    arch = stem
    # Heuristic cleanups: strip common dataset suffixes
    for suffix in (
        "_locomo_30q",
        "_synthetic_19q",
        "_puzzle_16q",
        "_advanced_23q",
        "_beam_30q",
        "_beam_ext_30q",
        "_locomo_ext_30q",
        "_synthetic",
        "_advanced",
        "_locomo",
        "_puzzle",
    ):
        if arch.endswith(suffix):
            arch = arch[: -len(suffix)]
            break

    return {
        "arch": arch,
        "dataset": ds,
        "r@20": r20,
        "r@50": r50,
        "llm_calls": (sum(llm_vals) / len(llm_vals)) if llm_vals else None,
        "embed_calls": (sum(emb_vals) / len(emb_vals)) if emb_vals else None,
        "total_retrieved": (sum(tot_vals) / len(tot_vals)) if tot_vals else None,
        "n": len(entries),
    }


# -----------------------------------------------------------------------------
# Pareto analysis
# -----------------------------------------------------------------------------


def cost_scalar(rec: Record) -> float:
    """Composite cost ≈ embed_calls + 5 * llm_calls (LLM calls dominate latency/$)."""
    llm = rec.get("llm_calls") or 0.0
    emb = rec.get("embed_calls") or 0.0
    return emb + 5.0 * llm


def dominates(a: Record, b: Record) -> bool:
    """Return True if a strictly dominates b (>= on recall, <= on cost, strict somewhere)."""
    ar20 = a.get("r@20") or 0.0
    br20 = b.get("r@20") or 0.0
    ar50 = a.get("r@50") or 0.0
    br50 = b.get("r@50") or 0.0
    a_cost = cost_scalar(a)
    b_cost = cost_scalar(b)

    # a must be >= b on both recalls and <= on cost
    if ar20 < br20 or ar50 < br50:
        return False
    if a_cost > b_cost:
        return False
    # strict somewhere
    return (ar20 > br20) or (ar50 > br50) or (a_cost < b_cost)


def compute_pareto(records: list[Record]) -> list[Record]:
    """Annotate each record with 'dominated_by' list and return the list."""
    # Only compare records that have both r@20 and r@50 and cost info
    usable = [r for r in records if r.get("r@20") is not None and r.get("r@50") is not None and r.get("llm_calls") is not None and r.get("embed_calls") is not None]
    for r in usable:
        r["dominated_by"] = []
    for i, a in enumerate(usable):
        for j, b in enumerate(usable):
            if i == j:
                continue
            if dominates(b, a):
                a["dominated_by"].append(b["arch"])
    return usable


# -----------------------------------------------------------------------------
# Output generation
# -----------------------------------------------------------------------------


def build_outputs() -> None:
    collector = collect_all()
    print(f"Collected {len(collector.records)} (arch, dataset) entries")

    # Group by dataset
    by_ds: dict[str, list[Record]] = defaultdict(list)
    for (arch, ds), rec in collector.records.items():
        # Ensure arch in record is the display name we use as key
        rec = dict(rec)
        rec["arch"] = arch
        rec["dataset"] = ds
        by_ds[ds].append(rec)

    pareto_json: dict[str, list[Record]] = {}
    pareto_optimal_per_ds: dict[str, set[str]] = {}
    all_usable_per_ds: dict[str, list[Record]] = {}

    for ds in sorted(by_ds):
        usable = compute_pareto(by_ds[ds])
        usable_sorted = sorted(
            usable,
            key=lambda r: (-(r.get("r@20") or 0.0), -(r.get("r@50") or 0.0), cost_scalar(r)),
        )
        pareto_json[ds] = [
            {
                "arch": r["arch"],
                "r@20": round(r["r@20"], 4) if r["r@20"] is not None else None,
                "r@50": round(r["r@50"], 4) if r["r@50"] is not None else None,
                "llm_calls": round(r["llm_calls"], 2) if r["llm_calls"] is not None else None,
                "embed_calls": round(r["embed_calls"], 2) if r["embed_calls"] is not None else None,
                "total_retrieved": round(r["total_retrieved"], 1) if r.get("total_retrieved") is not None else None,
                "n": r.get("n"),
                "dominated_by": sorted(set(r.get("dominated_by") or [])),
            }
            for r in usable_sorted
        ]
        pareto_optimal_per_ds[ds] = {
            r["arch"] for r in usable_sorted if not r.get("dominated_by")
        }
        all_usable_per_ds[ds] = usable_sorted

    # Write JSON
    out_json_path = RESULTS_DIR / "pareto_frontier.json"
    with out_json_path.open("w") as f:
        json.dump(pareto_json, f, indent=2)
    print(f"Wrote {out_json_path}")

    # Build Markdown summary
    md_lines: list[str] = []
    md_lines.append("# Pareto Frontier Summary (Recall vs Cost)")
    md_lines.append("")
    md_lines.append("Cost metric used for Pareto analysis: `embed_calls + 5 * llm_calls`.")
    md_lines.append("Dominance: architecture A dominates B iff A has >= r@20, >= r@50, and <= cost, with strict inequality somewhere.")
    md_lines.append("Only architectures with complete recall + cost data are included (n=%d total (arch,dataset) pairs)." % sum(len(v) for v in all_usable_per_ds.values()))
    md_lines.append("")

    for ds in sorted(pareto_json):
        entries = pareto_json[ds]
        md_lines.append(f"## Dataset: `{ds}` ({len(entries)} architectures)")
        md_lines.append("")
        md_lines.append("| Rank | Arch | r@20 | r@50 | LLM | Embed | Cost | Status |")
        md_lines.append("|------|------|------|------|-----|-------|------|--------|")
        for i, r in enumerate(entries, 1):
            cost = (r["embed_calls"] or 0) + 5 * (r["llm_calls"] or 0)
            dom = r["dominated_by"]
            if not dom:
                status = "**PARETO**"
            else:
                # Show up to 2 dominators
                status = f"dominated by {dom[0]}" + (f", {dom[1]}" if len(dom) > 1 else "") + (f", +{len(dom) - 2} more" if len(dom) > 2 else "")
            md_lines.append(
                f"| {i} | {r['arch']} | {r['r@20']:.3f} | {r['r@50']:.3f} | {r['llm_calls']:.1f} | {r['embed_calls']:.1f} | {cost:.1f} | {status} |"
            )
        md_lines.append("")

    # Cross-dataset Pareto table
    md_lines.append("## Cross-Dataset Pareto Membership")
    md_lines.append("")
    all_arches = set()
    for s in pareto_optimal_per_ds.values():
        all_arches.update(s)
    # Count in how many datasets each arch is Pareto-optimal
    arch_counts: dict[str, list[str]] = {a: [] for a in all_arches}
    for ds, optimal_set in pareto_optimal_per_ds.items():
        for a in optimal_set:
            arch_counts[a].append(ds)
    # Also: architectures that appeared on every dataset they were tested on
    # Need the set of datasets each arch appears in.
    arch_datasets: dict[str, list[str]] = defaultdict(list)
    for ds, recs in all_usable_per_ds.items():
        for r in recs:
            arch_datasets[r["arch"]].append(ds)

    md_lines.append("| Arch | #Datasets Pareto-Optimal | Tested on | Pareto on |")
    md_lines.append("|------|--------------------------|-----------|-----------|")
    for a in sorted(arch_counts, key=lambda x: (-len(arch_counts[x]), x)):
        tested = sorted(set(arch_datasets.get(a, [])))
        pareto_on = sorted(arch_counts[a])
        md_lines.append(
            f"| {a} | {len(pareto_on)} | {','.join(tested)} | {','.join(pareto_on)} |"
        )
    md_lines.append("")

    # Always-Pareto section
    n_datasets = len(pareto_optimal_per_ds)
    md_lines.append("### Architectures on the Pareto Frontier for ALL Datasets They Were Tested On")
    md_lines.append("")
    always_pareto = [
        a for a, pareto in arch_counts.items()
        if pareto and len(set(arch_datasets.get(a, []))) == len(pareto)
        and len(pareto) >= 2  # require at least two datasets
    ]
    for a in sorted(always_pareto, key=lambda x: (-len(arch_counts[x]), x)):
        md_lines.append(f"- **{a}** — Pareto on {len(arch_counts[a])} datasets: {', '.join(sorted(arch_counts[a]))}")
    md_lines.append("")

    # Strictly-dominated-everywhere architectures
    md_lines.append("### Architectures Strictly Dominated On Every Dataset They Were Tested On")
    md_lines.append("")
    dominated_everywhere: list[str] = []
    for a, tested in arch_datasets.items():
        tested_set = set(tested)
        if not tested_set:
            continue
        pareto_on = set(arch_counts.get(a, []))
        if pareto_on:
            continue
        # arch was tested but never Pareto-optimal
        dominated_everywhere.append(a)
    for a in sorted(dominated_everywhere):
        md_lines.append(f"- {a} (tested on: {', '.join(sorted(set(arch_datasets[a])))})")
    md_lines.append("")

    # Cost tiers analysis (locomo_30q as canonical)
    md_lines.append("## Cost-Tier Recall Plateau Analysis")
    md_lines.append("")
    md_lines.append("For each dataset, best r@20 and best r@50 achievable at each cost tier.")
    md_lines.append("")
    for ds in sorted(all_usable_per_ds):
        md_lines.append(f"### {ds}")
        md_lines.append("")
        md_lines.append("| Cost tier | Best r@20 | Best r@50 | Example arch |")
        md_lines.append("|-----------|-----------|-----------|--------------|")
        tiers = [
            ("<=1 LLM, <=4 embed (v2f/v15 class)", lambda r: (r["llm_calls"] or 0) <= 1 and (r["embed_calls"] or 0) <= 4),
            ("<=2 LLM, <=8 embed", lambda r: (r["llm_calls"] or 0) <= 2 and (r["embed_calls"] or 0) <= 8),
            ("<=3 LLM, <=12 embed", lambda r: (r["llm_calls"] or 0) <= 3 and (r["embed_calls"] or 0) <= 12),
            ("<=5 LLM, any embed", lambda r: (r["llm_calls"] or 0) <= 5),
            ("any cost", lambda r: True),
        ]
        for name, fn in tiers:
            subset = [r for r in all_usable_per_ds[ds] if fn(r)]
            if not subset:
                md_lines.append(f"| {name} | n/a | n/a | n/a |")
                continue
            best20 = max(subset, key=lambda r: r["r@20"] or 0)
            best50 = max(subset, key=lambda r: r["r@50"] or 0)
            md_lines.append(
                f"| {name} | {best20['r@20']:.3f} ({best20['arch']}) | {best50['r@50']:.3f} ({best50['arch']}) | |"
            )
        md_lines.append("")

    out_md_path = RESULTS_DIR / "pareto_summary.md"
    with out_md_path.open("w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {out_md_path}")


if __name__ == "__main__":
    build_outputs()
