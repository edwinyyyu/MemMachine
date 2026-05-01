"""Final oracle ceiling across all architectures tested this session.

Answers:
  "If we could magically pick the best architecture per question, what is the
  recall ceiling? Have we saturated it with our shipped recipes?"

For each question (across locomo_30q + synthetic_19q + puzzle_16q + advanced_23q)
we collect the per-question recall at K=20 and K=50 from every architecture we
have per-question data for. The oracle ceiling is the best-per-question recall.
We compare that to the shipped recipes:
  - K=20 shipped: two_speaker_filter (the biggest K=20 win)
  - K=50 shipped: composition_v2_all (and ens_all_plus_crit as backup)

Outputs:
  results/oracle_ceiling_final.json  -- full per-question recall matrix
  results/oracle_ceiling_final.md    -- human-readable report
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

DATASETS = ["locomo_30q", "synthetic_19q", "puzzle_16q", "advanced_23q"]
KS = [20, 50]

# Shipped recipes (per-K, per-dataset where applicable).
# K=20 shipped = two_speaker_filter (where it has coverage) else v2f
# K=50 shipped = composition_v2_all (where it has coverage) else ens_all_plus_crit
SHIPPED_K20 = {
    "locomo_30q": "two_speaker_filter@20",
    "synthetic_19q": "two_speaker_filter@20",
    "puzzle_16q": "v2f@20",  # two_speaker is conversation-specific; fall back on meta_v2f / v2f
    "advanced_23q": "v2f@20",
}
SHIPPED_K50 = {
    "locomo_30q": "composition_v2_all@50",
    "synthetic_19q": "composition_v2_all@50",
    "puzzle_16q": "composition_v2_all@50",
    "advanced_23q": "composition_v2_all@50",
}


# ----- helpers ------------------------------------------------------------


def detect_dataset(fn: str) -> str | None:
    for ds in DATASETS:
        if ds in fn:
            return ds
    return None


def qkey(conv_id: Any, qidx: Any, question: str) -> str:
    norm = re.sub(r"\s+", " ", (question or "").strip().lower())
    return f"{conv_id}|{qidx}|{norm}"


def safe_load(path: Path) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ----- collectors ---------------------------------------------------------
# Each collector returns:
#   arch_results: list of (arch_label, dataset, per_question_dict)
# where per_question_dict maps qkey -> {"r@20": float|None, "r@50": float|None,
#                                        "question": str, "category": str,
#                                        "conversation_id": str, "question_index": int,
#                                        "num_source_turns": int}


def collect_standard_per_question_files() -> list[tuple[str, str, dict]]:
    """Collect architectures whose results are stored as per-result lists with
    a fair_backfill { arch_r@20, arch_r@50 } block."""
    out: list[tuple[str, str, dict]] = []
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if not fn.endswith(".json"):
            continue
        ds = detect_dataset(fn)
        if ds is None:
            continue
        data = safe_load(RESULTS_DIR / fn)
        if not isinstance(data, dict):
            continue
        results = data.get("results")
        if not isinstance(results, list) or not results:
            continue
        # Must have fair_backfill
        first = results[0]
        if not isinstance(first, dict):
            continue
        if not isinstance(first.get("fair_backfill"), dict):
            continue
        fb0 = first["fair_backfill"]
        if not ("arch_r@20" in fb0 or "arch_r@50" in fb0):
            continue

        arch_label = fn[:-5].replace(f"_{ds}", "")
        pq: dict[str, dict] = {}
        for r in results:
            if not isinstance(r, dict):
                continue
            q = r.get("question")
            if not q:
                continue
            fb = r.get("fair_backfill") or {}
            r20 = fb.get("arch_r@20")
            r50 = fb.get("arch_r@50")
            if r20 is None and r50 is None:
                continue
            k = qkey(r.get("conversation_id"), r.get("question_index"), q)
            pq[k] = {
                "r@20": float(r20) if r20 is not None else None,
                "r@50": float(r50) if r50 is not None else None,
                "question": q,
                "category": r.get("category"),
                "conversation_id": r.get("conversation_id"),
                "question_index": r.get("question_index"),
                "num_source_turns": r.get("num_source_turns"),
            }
        if pq:
            out.append((arch_label, ds, pq))
    return out


def collect_composition_v2() -> list[tuple[str, str, dict]]:
    """Expand final_composition_v2.json per_question variants as architectures."""
    out: list[tuple[str, str, dict]] = []
    data = safe_load(RESULTS_DIR / "final_composition_v2.json")
    if not isinstance(data, dict):
        return out
    results = data.get("results", {})
    for ds, ds_block in results.items():
        pqs = ds_block.get("per_question", [])
        if not pqs:
            continue
        # Collect all variant names
        variants: set[str] = set()
        for item in pqs:
            rec = item.get("recall", {})
            variants.update(rec.keys())
        # Build one arch per variant
        per_variant: dict[str, dict[str, dict]] = {v: {} for v in variants}
        for item in pqs:
            q = item.get("question") or ""  # may be absent; fall back to qidx key
            # final_composition_v2 doesn't store the question text directly; lookup via category + qidx
            cid = item.get("conversation_id")
            qidx = item.get("question_index")
            cat = item.get("category")
            # synthesize a qkey that uses (cid, qidx) (question text unknown here)
            k = f"{cid}|{qidx}|__unknown__"
            for variant, val in item.get("recall", {}).items():
                # variant like v2f@20, composition_v2_all@50
                per_variant[variant][k] = {
                    "recall": float(val),
                    "category": cat,
                    "conversation_id": cid,
                    "question_index": qidx,
                    "num_source_turns": item.get("num_source_turns"),
                }
        for variant, pq in per_variant.items():
            if not pq:
                continue
            # Convert to standard pq form
            # variant encodes K: "@20" or "@50"
            if "@20" in variant:
                kint = 20
            elif "@50" in variant:
                kint = 50
            else:
                continue
            base_label = variant.replace("@20", "").replace("@50", "")
            formatted: dict[str, dict] = {}
            for key, rec in pq.items():
                formatted[key] = {
                    "r@20": rec["recall"] if kint == 20 else None,
                    "r@50": rec["recall"] if kint == 50 else None,
                    "question": None,
                    "category": rec.get("category"),
                    "conversation_id": rec.get("conversation_id"),
                    "question_index": rec.get("question_index"),
                    "num_source_turns": rec.get("num_source_turns"),
                }
            arch_label = f"compv2_{base_label}_K{kint}"
            out.append((arch_label, ds, formatted))
    return out


def collect_two_speaker_composition() -> list[tuple[str, str, dict]]:
    """two_speaker_composition.json per_question variants (locomo+synth)."""
    out: list[tuple[str, str, dict]] = []
    data = safe_load(RESULTS_DIR / "two_speaker_composition.json")
    if not isinstance(data, dict):
        return out
    results = data.get("results", {})
    for ds, ds_block in results.items():
        pqs = ds_block.get("per_question", [])
        if not pqs:
            continue
        variants: set[str] = set()
        for item in pqs:
            rec = item.get("recall", {})
            variants.update(rec.keys())
        per_variant: dict[str, dict[str, dict]] = {v: {} for v in variants}
        for item in pqs:
            cid = item.get("conversation_id")
            qidx = item.get("question_index")
            cat = item.get("category")
            k = f"{cid}|{qidx}|__unknown__"
            for variant, val in item.get("recall", {}).items():
                per_variant[variant][k] = {
                    "recall": float(val),
                    "category": cat,
                    "conversation_id": cid,
                    "question_index": qidx,
                    "num_source_turns": item.get("num_source_turns"),
                }
        for variant, pq in per_variant.items():
            if not pq:
                continue
            if "@20" in variant:
                kint = 20
            elif "@50" in variant:
                kint = 50
            else:
                continue
            base_label = variant.replace("@20", "").replace("@50", "")
            formatted: dict[str, dict] = {}
            for key, rec in pq.items():
                formatted[key] = {
                    "r@20": rec["recall"] if kint == 20 else None,
                    "r@50": rec["recall"] if kint == 50 else None,
                    "question": None,
                    "category": rec.get("category"),
                    "conversation_id": rec.get("conversation_id"),
                    "question_index": rec.get("question_index"),
                    "num_source_turns": rec.get("num_source_turns"),
                }
            arch_label = f"tspcomp_{base_label}_K{kint}"
            out.append((arch_label, ds, formatted))
    return out


def collect_critical_info_store() -> list[tuple[str, str, dict]]:
    """critical_info_store.json per_question for baseline / crit variants."""
    out: list[tuple[str, str, dict]] = []
    data = safe_load(RESULTS_DIR / "critical_info_store.json")
    if not isinstance(data, dict):
        return out
    for ds, ds_block in data.get("results", {}).items():
        for variant in ("baseline", "crit_additive_bonus_0.1", "crit_always_top_M"):
            block = ds_block.get(variant)
            if not isinstance(block, dict):
                continue
            pqs = block.get("per_question")
            if not pqs:
                continue
            formatted: dict[str, dict] = {}
            for r in pqs:
                q = r.get("question") or ""
                k = qkey(r.get("conversation_id"), r.get("question_index"), q)
                formatted[k] = {
                    "r@20": float(r["r@20"]) if "r@20" in r else None,
                    "r@50": float(r["r@50"]) if "r@50" in r else None,
                    "question": q,
                    "category": r.get("category"),
                    "conversation_id": r.get("conversation_id"),
                    "question_index": r.get("question_index"),
                    "num_source_turns": r.get("num_source_turns"),
                }
            if formatted:
                out.append((f"critinfo_{variant}", ds, formatted))
    return out


# ----- unify qkeys across different files --------------------------------
# Some of our collectors (composition_v2, two_speaker_composition) don't have
# question text, so they use a qkey with __unknown__. We'll normalize all
# qkeys to the form `conv_id|qidx` (no text), and track question metadata
# from the text-containing sources.


def normalize_key(key: str) -> str:
    parts = key.split("|", 2)
    if len(parts) >= 2:
        return f"{parts[0]}|{parts[1]}"
    return key


# ----- oracle compute ----------------------------------------------------


def main() -> None:
    print("Collecting per-question recall from all architectures...")
    collections: list[tuple[str, str, dict]] = []
    collections.extend(collect_standard_per_question_files())
    print(f"  standard files: {len(collections)} arch/ds pairs")
    n_before = len(collections)
    collections.extend(collect_composition_v2())
    print(f"  +composition_v2: +{len(collections) - n_before}")
    n_before = len(collections)
    collections.extend(collect_two_speaker_composition())
    print(f"  +two_speaker_composition: +{len(collections) - n_before}")
    n_before = len(collections)
    collections.extend(collect_critical_info_store())
    print(f"  +critical_info_store: +{len(collections) - n_before}")
    print(f"  TOTAL: {len(collections)} arch/ds collections")

    # Build the full matrix.
    # matrix[ds][norm_key][arch_label] = {"r@20": float|None, "r@50": float|None}
    matrix: dict[str, dict[str, dict[str, dict]]] = {
        ds: defaultdict(dict) for ds in DATASETS
    }
    # metadata[ds][norm_key] = {"question": str, "category": str, ...}
    metadata: dict[str, dict[str, dict]] = {ds: {} for ds in DATASETS}
    # archs_per_ds: dataset -> set of arch labels
    archs_per_ds: dict[str, set[str]] = {ds: set() for ds in DATASETS}

    for arch_label, ds, pq in collections:
        for raw_k, rec in pq.items():
            nk = normalize_key(raw_k)
            matrix[ds][nk][arch_label] = {
                "r@20": rec.get("r@20"),
                "r@50": rec.get("r@50"),
            }
            # Keep metadata (prefer the first arch that has a non-empty question).
            existing = metadata[ds].get(nk)
            if existing is None or not existing.get("question"):
                metadata[ds][nk] = {
                    "question": rec.get("question"),
                    "category": rec.get("category"),
                    "conversation_id": rec.get("conversation_id"),
                    "question_index": rec.get("question_index"),
                    "num_source_turns": rec.get("num_source_turns"),
                }
            else:
                # Backfill missing fields
                for field in (
                    "category",
                    "conversation_id",
                    "question_index",
                    "num_source_turns",
                ):
                    if not existing.get(field):
                        v = rec.get(field)
                        if v is not None:
                            existing[field] = v
        archs_per_ds[ds].add(arch_label)

    for ds in DATASETS:
        print(f"  {ds}: {len(archs_per_ds[ds])} archs, {len(matrix[ds])} questions")

    # Quality check: we should have 30/19/16/23 = 88 questions. But many
    # architectures don't have puzzle/advanced per-question data — only the
    # "standard" collector gives those. So the union of questions across
    # architectures equals the dataset size for each dataset.
    expected = {
        "locomo_30q": 30,
        "synthetic_19q": 19,
        "puzzle_16q": 16,
        "advanced_23q": 23,
    }

    # ===== Compute ceilings =====
    # best_arch_recall per (ds, question, K)
    # shipped recall per (ds, question, K)
    # gap
    oracle = {}  # oracle[ds] = { "@20": list, "@50": list, "stubborn": list }
    stubborn_list: list[dict] = []  # entries where best_r50 < 0.5
    per_category_gap: dict[str, dict[str, dict]] = {}

    for ds in DATASETS:
        questions = matrix[ds]
        rows = []
        for nk, arch_map in questions.items():
            meta = metadata[ds].get(nk, {})
            # Collect all recalls at each K
            r20s = {
                a: v["r@20"] for a, v in arch_map.items() if v.get("r@20") is not None
            }
            r50s = {
                a: v["r@50"] for a, v in arch_map.items() if v.get("r@50") is not None
            }
            best_r20 = max(r20s.values()) if r20s else None
            best_r50 = max(r50s.values()) if r50s else None
            best_arch_r20 = max(r20s.items(), key=lambda t: t[1])[0] if r20s else None
            best_arch_r50 = max(r50s.items(), key=lambda t: t[1])[0] if r50s else None

            # Shipped recipes
            shipped20_name = SHIPPED_K20[ds]
            shipped50_name = SHIPPED_K50[ds]
            shipped_r20 = _lookup_shipped(r20s, arch_map, shipped20_name, 20)
            shipped_r50 = _lookup_shipped(r50s, arch_map, shipped50_name, 50)

            row = {
                "key": nk,
                "question": meta.get("question"),
                "category": meta.get("category"),
                "conversation_id": meta.get("conversation_id"),
                "question_index": meta.get("question_index"),
                "num_source_turns": meta.get("num_source_turns"),
                "n_archs_r20": len(r20s),
                "n_archs_r50": len(r50s),
                "best_r20": best_r20,
                "best_r50": best_r50,
                "best_arch_r20": best_arch_r20,
                "best_arch_r50": best_arch_r50,
                "shipped_r20": shipped_r20,
                "shipped_r50": shipped_r50,
            }
            rows.append(row)

            if best_r50 is not None and best_r50 < 0.5:
                stubborn_list.append({"dataset": ds, **row})

        oracle[ds] = rows

    # ===== Aggregate per-dataset and per-K =====
    agg = {}
    for ds in DATASETS:
        rows = oracle[ds]

        def _mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else None

        mean_best_r20 = _mean([r["best_r20"] for r in rows])
        mean_best_r50 = _mean([r["best_r50"] for r in rows])
        mean_ship_r20 = _mean([r["shipped_r20"] for r in rows])
        mean_ship_r50 = _mean([r["shipped_r50"] for r in rows])
        gap20 = (
            (mean_best_r20 - mean_ship_r20)
            if (mean_best_r20 is not None and mean_ship_r20 is not None)
            else None
        )
        gap50 = (
            (mean_best_r50 - mean_ship_r50)
            if (mean_best_r50 is not None and mean_ship_r50 is not None)
            else None
        )
        agg[ds] = {
            "n_questions": len(rows),
            "n_archs_avg_at_20": sum(r["n_archs_r20"] for r in rows)
            / max(len(rows), 1),
            "n_archs_avg_at_50": sum(r["n_archs_r50"] for r in rows)
            / max(len(rows), 1),
            "oracle_ceiling_r20": mean_best_r20,
            "oracle_ceiling_r50": mean_best_r50,
            "shipped_r20": mean_ship_r20,
            "shipped_r50": mean_ship_r50,
            "gap_r20": gap20,
            "gap_r50": gap50,
            "shipped_r20_name": SHIPPED_K20[ds],
            "shipped_r50_name": SHIPPED_K50[ds],
        }

    # ===== Per-category headroom =====
    cat_agg: dict[tuple[str, str], dict] = {}
    for ds in DATASETS:
        rows = oracle[ds]
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            cat = r.get("category") or "unknown"
            by_cat[cat].append(r)
        for cat, rs in by_cat.items():

            def _mean(xs):
                xs = [x for x in xs if x is not None]
                return sum(xs) / len(xs) if xs else None

            mbr50 = _mean([r["best_r50"] for r in rs])
            msr50 = _mean([r["shipped_r50"] for r in rs])
            mbr20 = _mean([r["best_r20"] for r in rs])
            msr20 = _mean([r["shipped_r20"] for r in rs])
            g50 = (mbr50 - msr50) if (mbr50 is not None and msr50 is not None) else None
            g20 = (mbr20 - msr20) if (mbr20 is not None and msr20 is not None) else None
            cat_agg[(ds, cat)] = {
                "n": len(rs),
                "oracle_r20": mbr20,
                "shipped_r20": msr20,
                "gap_r20": g20,
                "oracle_r50": mbr50,
                "shipped_r50": msr50,
                "gap_r50": g50,
            }

    # ===== Write outputs =====
    # JSON: full matrix (trimmed to per-question summary)
    json_out = {
        "datasets": DATASETS,
        "ks": KS,
        "shipped_k20": SHIPPED_K20,
        "shipped_k50": SHIPPED_K50,
        "n_archs_per_dataset": {ds: len(archs_per_ds[ds]) for ds in DATASETS},
        "archs_per_dataset": {ds: sorted(archs_per_ds[ds]) for ds in DATASETS},
        "per_dataset_aggregate": agg,
        "per_category_aggregate": {
            f"{ds}|{cat}": v for (ds, cat), v in cat_agg.items()
        },
        "oracle_rows": {ds: oracle[ds] for ds in DATASETS},
        "stubborn_failures_k50_lt_0.5": stubborn_list,
    }
    (RESULTS_DIR / "oracle_ceiling_final.json").write_text(
        json.dumps(json_out, indent=2, default=str)
    )

    # Markdown
    md = []
    md.append("# Final Oracle Ceiling Across All Architectures")
    md.append("")
    md.append(f"Datasets: {', '.join(DATASETS)}")
    md.append(
        f"Architectures collected per dataset: "
        f"{', '.join(f'{ds}={len(archs_per_ds[ds])}' for ds in DATASETS)}"
    )
    md.append("")
    md.append("## Oracle ceiling vs shipped recipe")
    md.append("")
    md.append(
        "| Dataset | N | Archs (≥ K=50) | Shipped @20 | Oracle @20 | Gap @20 | Shipped @50 | Oracle @50 | Gap @50 |"
    )
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ds in DATASETS:
        a = agg[ds]

        def _fmt(x):
            return f"{x:.4f}" if x is not None else "n/a"

        md.append(
            f"| {ds} | {a['n_questions']} | {a['n_archs_avg_at_50']:.1f} | "
            f"{_fmt(a['shipped_r20'])} ({a['shipped_r20_name']}) | "
            f"{_fmt(a['oracle_ceiling_r20'])} | {_fmt(a['gap_r20'])} | "
            f"{_fmt(a['shipped_r50'])} ({a['shipped_r50_name']}) | "
            f"{_fmt(a['oracle_ceiling_r50'])} | {_fmt(a['gap_r50'])} |"
        )
    md.append("")

    # Overall totals (question-weighted average across datasets)
    total_q = sum(agg[ds]["n_questions"] for ds in DATASETS)

    def _weighted(field):
        num = 0.0
        den = 0
        for ds in DATASETS:
            v = agg[ds].get(field)
            if v is None:
                continue
            num += v * agg[ds]["n_questions"]
            den += agg[ds]["n_questions"]
        return num / den if den else None

    md.append("**Overall (question-weighted):**")
    md.append("")
    md.append(f"- Oracle @20 = {_weighted('oracle_ceiling_r20'):.4f}")
    md.append(f"- Shipped @20 = {_weighted('shipped_r20'):.4f}")
    md.append(f"- Gap @20 = {_weighted('gap_r20'):.4f}")
    md.append(f"- Oracle @50 = {_weighted('oracle_ceiling_r50'):.4f}")
    md.append(f"- Shipped @50 = {_weighted('shipped_r50'):.4f}")
    md.append(f"- Gap @50 = {_weighted('gap_r50'):.4f}")
    md.append("")

    # Stubborn failures updated
    md.append("## Stubborn failures (best_r@50 < 0.5 across ALL archs)")
    md.append("")
    if not stubborn_list:
        md.append(
            "**None.** Every assessed question has at least one architecture reaching r@50 >= 0.5."
        )
        md.append("")
    else:
        md.append(f"Count: **{len(stubborn_list)}**")
        md.append("")
        md.append(
            "| Dataset | Conv | Category | # gold | Best @20 | Best @50 | Best arch @50 | Question |"
        )
        md.append("| --- | --- | --- | ---: | ---: | ---: | --- | --- |")
        for s in sorted(
            stubborn_list, key=lambda x: (x["dataset"], x.get("conversation_id") or "")
        ):
            q = (s.get("question") or "")[:160].replace("|", "\\|")
            md.append(
                f"| {s['dataset']} | {s.get('conversation_id', '')} | {s.get('category', '')} | "
                f"{s.get('num_source_turns', '')} | {s.get('best_r20', 'n/a')} | "
                f"{s.get('best_r50', 'n/a')} | {s.get('best_arch_r50', '')} | {q} |"
            )
        md.append("")

    # Per-category headroom
    md.append("## Per-category headroom @50 (gap = oracle - shipped)")
    md.append("")
    md.append("Sorted by largest gap first (bigger gap = more routing headroom).")
    md.append("")
    md.append(
        "| Dataset | Category | N | Oracle @50 | Shipped @50 | Gap @50 | Oracle @20 | Shipped @20 | Gap @20 |"
    )
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    sorted_cats = sorted(cat_agg.items(), key=lambda kv: -(kv[1].get("gap_r50") or 0))
    for (ds, cat), v in sorted_cats:

        def _fmt(x):
            return f"{x:.4f}" if x is not None else "n/a"

        md.append(
            f"| {ds} | {cat} | {v['n']} | {_fmt(v.get('oracle_r50'))} | "
            f"{_fmt(v.get('shipped_r50'))} | {_fmt(v.get('gap_r50'))} | "
            f"{_fmt(v.get('oracle_r20'))} | {_fmt(v.get('shipped_r20'))} | "
            f"{_fmt(v.get('gap_r20'))} |"
        )
    md.append("")

    # Verdict
    md.append("## Verdict")
    md.append("")
    ow50 = _weighted("oracle_ceiling_r50")
    sw50 = _weighted("shipped_r50")
    ow20 = _weighted("oracle_ceiling_r20")
    sw20 = _weighted("shipped_r20")
    g50 = ow50 - sw50 if (ow50 is not None and sw50 is not None) else None
    g20 = ow20 - sw20 if (ow20 is not None and sw20 is not None) else None

    def verdict_line(name, gap):
        if gap is None:
            return f"- {name}: could not compute."
        if gap < 0.01:
            return (
                f"- {name}: gap = {gap:.4f} pp; **saturated** (within 1pp of oracle)."
            )
        if gap < 0.03:
            return f"- {name}: gap = {gap:.4f}; near-saturated (1-3pp headroom)."
        if gap < 0.05:
            return f"- {name}: gap = {gap:.4f}; modest headroom (3-5pp) via routing/composition."
        return f"- {name}: gap = {gap:.4f}; meaningful headroom (>=5pp) — routing could still help."

    md.append(verdict_line("K=20 overall", g20))
    md.append(verdict_line("K=50 overall", g50))
    md.append("")
    md.append(f"- Stubborn failures at K=50 (< 0.5 best): **{len(stubborn_list)}**")
    md.append("")

    md.append(
        f"Captured fraction of oracle ceiling: "
        f"K=20 = {sw20 / ow20:.3f} of oracle ({sw20:.4f}/{ow20:.4f}), "
        f"K=50 = {sw50 / ow50:.3f} of oracle ({sw50:.4f}/{ow50:.4f})."
    )
    md.append("")

    md.append("## Pointers")
    md.append("")
    md.append("- Raw matrix: `results/oracle_ceiling_final.json`")
    md.append("- Source script: `oracle_ceiling_analysis.py`")
    md.append("")

    (RESULTS_DIR / "oracle_ceiling_final.md").write_text("\n".join(md))

    # Console summary
    print()
    print("=== ORACLE CEILING ===")
    for ds in DATASETS:
        a = agg[ds]
        print(
            f"{ds}: n={a['n_questions']}  oracle@20={a['oracle_ceiling_r20']:.4f}  "
            f"shipped@20={a['shipped_r20']:.4f}  gap@20={a['gap_r20']:.4f}  |  "
            f"oracle@50={a['oracle_ceiling_r50']:.4f}  shipped@50={a['shipped_r50']:.4f}  "
            f"gap@50={a['gap_r50']:.4f}"
        )
    print(f"OVERALL oracle@20={ow20:.4f} shipped@20={sw20:.4f} gap@20={g20:.4f}")
    print(f"OVERALL oracle@50={ow50:.4f} shipped@50={sw50:.4f} gap@50={g50:.4f}")
    print(f"Stubborn (best r@50 < 0.5): {len(stubborn_list)}")


def _lookup_shipped(
    r_at_k: dict, arch_map: dict, shipped_name: str, k: int
) -> float | None:
    """Resolve shipped recipe recall at K. shipped_name looks like
    'two_speaker_filter@20' or 'composition_v2_all@50'. We search the arch_map
    for matching labels, preferring:
      1) Exact arch with matching K (for ensembles from composition_v2 etc)
      2) An arch whose name contains the base string
    """
    base = shipped_name.split("@", 1)[0]
    # Exact match against K=suffix variants from composition_v2
    candidates_prefixed = [
        f"compv2_{base}_K{k}",
        f"tspcomp_{base}_K{k}",
    ]
    for c in candidates_prefixed:
        if c in arch_map:
            rec = arch_map[c]
            key = f"r@{k}"
            if rec.get(key) is not None:
                return rec[key]
    # Fallback: search arch_map for any arch whose label contains base text
    # Prefer names like "*{base}*" in the K=K r_at_k dict
    best = None
    for arch_label, v in r_at_k.items():
        ll = arch_label.lower()
        if base.lower() in ll:
            # Prefer match with same K tag
            if f"K{k}" in arch_label or f"@{k}" in arch_label:
                return v
            if best is None:
                best = v
    if best is not None:
        return best
    # Specific fallbacks: for two_speaker_filter, search for
    # "two_speaker_two_speaker_filter" style
    if base == "two_speaker_filter":
        for arch_label, v in r_at_k.items():
            if "two_speaker_filter" in arch_label:
                return v
    if base == "v2f":
        # use meta_v2f or antipara_v2f_anti_paraphrase / fairbackfill_meta_v2f
        for arch_label, v in r_at_k.items():
            if arch_label.endswith("meta_v2f") or "meta_v2f" in arch_label:
                return v
        for arch_label, v in r_at_k.items():
            if arch_label == "fairbackfill_meta_v2f":
                return v
    return None


if __name__ == "__main__":
    main()
