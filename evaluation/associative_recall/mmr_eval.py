"""Fair-backfill eval of MMR cue-selection variants vs v2f baseline.

Runs 4 MMR variants + v2f_3cues baseline + meta_v2f (dedicated) on LoCoMo-30
and synthetic-19 at K=20 and K=50 using the fair-backfill methodology.

Also computes diversity metric: mean pairwise cosine of selected cues (MMR
variants) vs v2f natural cues. If MMR variants have much lower pairwise
cosine but recall doesn't improve, v2f's natural cue diversity is already
adequate.

Usage:
    uv run python mmr_eval.py
    uv run python mmr_eval.py --archs mmr_lam0.5_k3,v2f_3cues
    uv run python mmr_eval.py --datasets locomo_30q
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from associative_recall import Segment
from fair_backfill_eval import (
    BUDGETS,
    DATASETS,
    RESULTS_DIR,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)
from mmr_cue_selection import (
    ARCH_CLASSES as MMR_ARCH_CLASSES,
    MMREmbeddingCache,
    mean_pairwise_cosine,
)
from antipara_cue_gen import MetaV2fDedicated

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")

ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    **MMR_ARCH_CLASSES,
}


def evaluate_question(arch, question: dict) -> dict:
    """Run arch on a single question, produce fair-backfill metrics + metadata."""
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
    cosine_result = arch.store.search(
        query_emb, top_k=max_K, conversation_id=conv_id
    )
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
        "candidate_cues": result.metadata.get("candidate_cues", []),
        "selected_cues": result.metadata.get(
            "selected_cues", result.metadata.get("cues", [])
        ),
        "diversity_selected_pairwise_cos": result.metadata.get(
            "diversity_selected_pairwise_cos"
        ),
        "diversity_all_candidates_pairwise_cos": result.metadata.get(
            "diversity_all_candidates_pairwise_cos"
        ),
        "n_candidates": result.metadata.get("n_candidates"),
        "n_selected": result.metadata.get("n_selected"),
        "lam": result.metadata.get("lam"),
        "k_cues": result.metadata.get("k_cues"),
        "hop0_empty": result.metadata.get("hop0_empty", False),
    }

    arch_ids_at_K: dict[int, set[int]] = {}
    for K in BUDGETS:
        b_rec, a_rec, a_segs_at_K = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)
        arch_ids_at_K[K] = {s.turn_id for s in a_segs_at_K}

    row["gold_found_at_K"] = {
        str(K): sorted(arch_ids_at_K[K] & source_ids) for K in BUDGETS
    }

    return row


def run_one(
    arch_name: str,
    arch,
    dataset: str,
    questions: list[dict],
) -> tuple[list[dict], dict, dict]:
    print(f"\n{'=' * 70}")
    print(f"{arch_name} | {dataset} | {len(questions)} questions")
    print(f"{'=' * 70}")

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

    print(f"\n--- {arch_name} on {dataset} ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: baseline={summary[f'baseline_r@{K}']:.3f} "
            f"arch={summary[f'arch_r@{K}']:.3f} "
            f"delta={summary[f'delta_r@{K}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    print(
        f"  avg total_retrieved={summary['avg_total_retrieved']:.0f} "
        f"llm={summary['avg_llm_calls']:.1f} "
        f"embed={summary['avg_embed_calls']:.1f}"
    )
    for cat, c in by_cat.items():
        print(
            f"    {cat:28s} (n={c['n']}): "
            f"r@20 d={c['delta_r@20']:+.3f} r@50 d={c['delta_r@50']:+.3f} "
            f"W/T/L@50={c['W/T/L_r@50']}"
        )

    return results, summary, by_cat


def compute_v2f_natural_diversity(
    v2f_rows: list[dict], cache: MMREmbeddingCache
) -> dict:
    """For each v2f question, embed its cues and compute pairwise cosine.
    Uses MMR embedding cache (which reads from many shared caches) so v2f cue
    embeddings are hit by cache without extra API calls.
    """
    per_q: list[dict] = []
    pairwise_vals: list[float] = []
    counts: list[int] = []
    for r in v2f_rows:
        cues = r.get("selected_cues", []) or r.get("cues", [])
        # antipara rows have "cues" key
        embs: list[np.ndarray] = []
        for c in cues:
            c_str = (c or "").strip()
            if not c_str:
                continue
            emb = cache.get(c_str)
            if emb is not None:
                embs.append(emb)
        pw = mean_pairwise_cosine(embs)
        per_q.append(
            {
                "conversation_id": r.get("conversation_id"),
                "question_index": r.get("question_index"),
                "n_cues": len(embs),
                "pairwise_cos": round(pw, 4),
            }
        )
        if len(embs) >= 2:
            pairwise_vals.append(pw)
            counts.append(len(embs))

    mean_pw = float(np.mean(pairwise_vals)) if pairwise_vals else 0.0
    return {
        "mean_pairwise_cos": round(mean_pw, 4),
        "num_questions_with_ge2_cues": len(pairwise_vals),
        "avg_n_cues_per_question": round(
            float(np.mean(counts)) if counts else 0.0, 2
        ),
        "per_question": per_q,
    }


def compute_mmr_diversity(results: list[dict]) -> dict:
    """Mean pairwise cosine among SELECTED MMR cues (across questions)."""
    sel_vals: list[float] = []
    all_vals: list[float] = []
    for r in results:
        sel = r.get("diversity_selected_pairwise_cos")
        if sel is not None and r.get("n_selected", 0) >= 2:
            sel_vals.append(sel)
        all_ = r.get("diversity_all_candidates_pairwise_cos")
        if all_ is not None and (r.get("n_candidates", 0) >= 2):
            all_vals.append(all_)
    return {
        "mean_pairwise_cos_selected": round(
            float(np.mean(sel_vals)) if sel_vals else 0.0, 4
        ),
        "mean_pairwise_cos_all_candidates": round(
            float(np.mean(all_vals)) if all_vals else 0.0, 4
        ),
        "n_questions_with_ge2_selected": len(sel_vals),
    }


def qualitative_sample(results: list[dict]) -> dict | None:
    """Return one question from `results` showing its 8 candidates and which
    ones MMR selected. Prefers a LoCoMo question with clearly different lam."""
    for r in results:
        cands = r.get("candidate_cues") or []
        sel = r.get("selected_cues") or []
        if len(cands) >= 6 and len(sel) >= 3:
            return {
                "conversation_id": r.get("conversation_id"),
                "question_index": r.get("question_index"),
                "category": r.get("category"),
                "question": r.get("question"),
                "candidate_cues": cands,
                "selected_cues": sel,
                "diversity_selected_pairwise_cos": r.get(
                    "diversity_selected_pairwise_cos"
                ),
                "diversity_all_candidates_pairwise_cos": r.get(
                    "diversity_all_candidates_pairwise_cos"
                ),
                "arch_r@20": r["fair_backfill"].get("arch_r@20"),
                "arch_r@50": r["fair_backfill"].get("arch_r@50"),
                "baseline_r@20": r["fair_backfill"].get("baseline_r@20"),
                "baseline_r@50": r["fair_backfill"].get("baseline_r@50"),
            }
    return None


def top_categories_delta(by_cat: dict, K: int = 50) -> tuple[list, list]:
    """Return top 2 gaining and top 2 losing categories by delta_r@K."""
    rows = []
    for cat, c in by_cat.items():
        rows.append((cat, c[f"delta_r@{K}"], c[f"W/T/L_r@{K}"], c["n"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    gaining = [
        {"category": cat, "delta": d, "W/T/L": wtl, "n": n}
        for cat, d, wtl, n in rows[:2]
        if d > 0
    ]
    losing = [
        {"category": cat, "delta": d, "W/T/L": wtl, "n": n}
        for cat, d, wtl, n in rows[-2:]
        if d < 0
    ]
    return gaining, losing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archs",
        default=",".join(ARCH_CLASSES.keys()),
        help="Comma-separated arch names",
    )
    p.add_argument(
        "--datasets",
        default=",".join(EVAL_DATASETS),
        help="Comma-separated dataset names",
    )
    args = p.parse_args()

    arch_names = [a.strip() for a in args.archs.split(",") if a.strip()]
    ds_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for a in arch_names:
        if a not in ARCH_CLASSES:
            raise SystemExit(f"Unknown arch: {a}")
    for d in ds_names:
        if d not in DATASETS:
            raise SystemExit(f"Unknown dataset: {d}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = defaultdict(dict)

    for ds_name in ds_names:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        for arch_name in arch_names:
            cls = ARCH_CLASSES[arch_name]
            arch = cls(store)
            results, summary, by_cat = run_one(
                arch_name, arch, ds_name, questions
            )
            all_results[arch_name][ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
            }

    # Diversity analysis
    # Use a fresh MMR embedding cache for v2f natural diversity so we can
    # read back v2f cue embeddings.
    mmr_emb_cache = MMREmbeddingCache()

    diversity: dict[str, dict] = {}
    for ds_name in ds_names:
        diversity[ds_name] = {}
        # v2f natural: take meta_v2f cues
        if ds_name in all_results.get("meta_v2f", {}):
            v2f_rows = all_results["meta_v2f"][ds_name]["results"]
            diversity[ds_name]["v2f_natural"] = compute_v2f_natural_diversity(
                v2f_rows, mmr_emb_cache
            )
        for arch_name in arch_names:
            if arch_name == "meta_v2f":
                continue
            if ds_name not in all_results.get(arch_name, {}):
                continue
            rows = all_results[arch_name][ds_name]["results"]
            diversity[ds_name][arch_name] = compute_mmr_diversity(rows)

    # Qualitative sample (from mmr_lam0.5_k3 on locomo)
    sample: dict | None = None
    if (
        "mmr_lam0.5_k3" in all_results
        and "locomo_30q" in all_results["mmr_lam0.5_k3"]
    ):
        sample = qualitative_sample(
            all_results["mmr_lam0.5_k3"]["locomo_30q"]["results"]
        )

    # Top gaining/losing categories for mmr_lam0.5_k3 on locomo_30q
    top_gaining: list = []
    top_losing: list = []
    if (
        "mmr_lam0.5_k3" in all_results
        and "locomo_30q" in all_results["mmr_lam0.5_k3"]
    ):
        top_gaining, top_losing = top_categories_delta(
            all_results["mmr_lam0.5_k3"]["locomo_30q"]["category_breakdown"],
            K=50,
        )

    # Raw JSON
    raw: dict = {
        "archs": arch_names,
        "datasets": ds_names,
        "summaries": {
            a: {
                d: {
                    "summary": all_results[a][d]["summary"],
                    "category_breakdown": all_results[a][d][
                        "category_breakdown"
                    ],
                }
                for d in all_results[a]
            }
            for a in all_results
        },
        "diversity": diversity,
        "qualitative_sample": sample,
        "top_gaining_categories": top_gaining,
        "top_losing_categories": top_losing,
    }

    raw_path = RESULTS_DIR / "mmr_cue_study.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset full results
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"mmr_{a}_{d}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "arch": a,
                        "dataset": d,
                        "summary": all_results[a][d]["summary"],
                        "category_breakdown": all_results[a][d][
                            "category_breakdown"
                        ],
                        "results": all_results[a][d]["results"],
                    },
                    f,
                    indent=2,
                    default=str,
                )

    # Markdown report
    md_lines: list[str] = []
    md_lines.append("# MMR Cue-Diversity Selection Study\n")
    md_lines.append(
        "Motivation: v2f generates 2-3 cues per call — are they redundant "
        "variants probing the same region of embedding space? MMR selection "
        "generates 8 candidates per call, selects 3 (or 4) maximizing "
        "mutual distance balanced against query relevance. Tests whether "
        "cue redundancy is leaving gold unretrieved.\n"
    )

    # Recall table
    md_lines.append("## Fair-backfill recall\n")
    md_lines.append(
        "| Arch | Dataset | base@20 | arch@20 | d@20 | base@50 | arch@50 | "
        "d@50 | llm/q |"
    )
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            md_lines.append(
                f"| {a} | {d} | "
                f"{s['baseline_r@20']:.3f} | {s['arch_r@20']:.3f} | "
                f"{s['delta_r@20']:+.3f} | "
                f"{s['baseline_r@50']:.3f} | {s['arch_r@50']:.3f} | "
                f"{s['delta_r@50']:+.3f} | "
                f"{s['avg_llm_calls']:.1f} |"
            )

    # Diversity
    md_lines.append("\n## Cue pairwise cosine (diversity metric)\n")
    md_lines.append(
        "Lower pairwise cosine = more diverse cues (cover distinct regions). "
        "Compare v2f natural (2 cues) vs MMR selected (3 or 4).\n"
    )
    md_lines.append(
        "| Dataset | Variant | mean pairwise cos | n_q w/ >=2 cues |"
    )
    md_lines.append("|---|---|---:|---:|")
    for ds_name in ds_names:
        ddiv = diversity.get(ds_name, {})
        v2f_info = ddiv.get("v2f_natural", {})
        if v2f_info:
            md_lines.append(
                f"| {ds_name} | v2f_natural | "
                f"{v2f_info['mean_pairwise_cos']:.3f} | "
                f"{v2f_info['num_questions_with_ge2_cues']} |"
            )
        for arch_name in arch_names:
            if arch_name == "meta_v2f":
                continue
            info = ddiv.get(arch_name)
            if info:
                md_lines.append(
                    f"| {ds_name} | {arch_name} (selected) | "
                    f"{info['mean_pairwise_cos_selected']:.3f} | "
                    f"{info['n_questions_with_ge2_selected']} |"
                )

    # Qualitative sample
    if sample:
        md_lines.append("\n## Sample: 8 candidates, which 3 MMR selected\n")
        md_lines.append(
            f"**Question** (conv={sample['conversation_id']}, "
            f"idx={sample['question_index']}, "
            f"category={sample['category']}): {sample['question']}\n"
        )
        md_lines.append(
            f"baseline_r@50={sample['baseline_r@50']}, "
            f"arch_r@50={sample['arch_r@50']} | "
            f"selected-pairwise-cos="
            f"{sample['diversity_selected_pairwise_cos']}, "
            f"all-candidates-pairwise-cos="
            f"{sample['diversity_all_candidates_pairwise_cos']}\n"
        )
        md_lines.append("\n**Candidates:**")
        for i, c in enumerate(sample["candidate_cues"]):
            marker = (
                "[SELECTED] " if c in sample["selected_cues"] else "           "
            )
            md_lines.append(f"\n{i+1}. {marker}{c}")

    # Top categories
    if top_gaining or top_losing:
        md_lines.append(
            "\n\n## Top categories by d_r@50 (mmr_lam0.5_k3 on LoCoMo-30)\n"
        )
        md_lines.append("Gaining:")
        for g in top_gaining:
            md_lines.append(
                f"  - {g['category']} (n={g['n']}): delta={g['delta']:+.3f} "
                f"W/T/L={g['W/T/L']}"
            )
        md_lines.append("Losing:")
        for l in top_losing:
            md_lines.append(
                f"  - {l['category']} (n={l['n']}): delta={l['delta']:+.3f} "
                f"W/T/L={l['W/T/L']}"
            )

    # Verdict
    md_lines.append("\n## Verdict\n")
    verdict = "(see numbers above)"
    if (
        "meta_v2f" in all_results
        and "locomo_30q" in all_results["meta_v2f"]
        and "mmr_lam0.5_k3" in all_results
        and "locomo_30q" in all_results["mmr_lam0.5_k3"]
    ):
        v2f_lc50 = all_results["meta_v2f"]["locomo_30q"]["summary"][
            "arch_r@50"
        ]
        mmr_lc50 = all_results["mmr_lam0.5_k3"]["locomo_30q"]["summary"][
            "arch_r@50"
        ]
        mmr3_lc50 = None
        if "v2f_3cues" in all_results and "locomo_30q" in all_results[
            "v2f_3cues"
        ]:
            mmr3_lc50 = all_results["v2f_3cues"]["locomo_30q"]["summary"][
                "arch_r@50"
            ]
        mmr07_lc50 = None
        if (
            "mmr_lam0.7_k3" in all_results
            and "locomo_30q" in all_results["mmr_lam0.7_k3"]
        ):
            mmr07_lc50 = all_results["mmr_lam0.7_k3"]["locomo_30q"][
                "summary"
            ]["arch_r@50"]

        if mmr_lc50 > v2f_lc50 + 0.005:
            # Differentiate: is the win MMR-specific, or just more-cues?
            more_cues_beats = (
                mmr3_lc50 is not None and mmr3_lc50 > v2f_lc50 + 0.005
            )
            rel_heavy_beats = (
                mmr07_lc50 is not None and mmr07_lc50 > v2f_lc50 + 0.005
            )
            if not more_cues_beats and not rel_heavy_beats:
                verdict = (
                    f"**SHIP**: mmr_lam0.5_k3 beats v2f on LoCoMo-30 @K=50 "
                    f"({mmr_lc50:.3f} vs {v2f_lc50:.3f}); the win requires "
                    f"both +cues and diversity (v2f_3cues and lam0.7 do not "
                    f"match it)."
                )
            else:
                verdict = (
                    f"**BORDERLINE**: mmr_lam0.5_k3 beats v2f on LoCoMo-30 "
                    f"@K=50 ({mmr_lc50:.3f} vs {v2f_lc50:.3f}), but "
                    f"v2f_3cues={mmr3_lc50} / lam0.7={mmr07_lc50} — win may "
                    f"be from 'more cues' not 'diversity'."
                )
        else:
            verdict = (
                f"**ABANDON**: mmr_lam0.5_k3 does NOT beat v2f on LoCoMo-30 "
                f"@K=50 (v2f={v2f_lc50:.3f}, mmr={mmr_lc50:.3f})."
            )
    md_lines.append(verdict + "\n")

    md_path = RESULTS_DIR / "mmr_cue_study.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved: {md_path}")

    # Final table
    print("\n" + "=" * 100)
    print("MMR CUE-SELECTION SUMMARY")
    print("=" * 100)
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:22s} {d:14s} "
                f"b@20={s['baseline_r@20']:.3f} a@20={s['arch_r@20']:.3f} "
                f"d@20={s['delta_r@20']:+.3f}  "
                f"b@50={s['baseline_r@50']:.3f} a@50={s['arch_r@50']:.3f} "
                f"d@50={s['delta_r@50']:+.3f}  "
                f"llm={s['avg_llm_calls']:.1f}"
            )

    print("\n--- Diversity metric ---")
    for ds_name in ds_names:
        ddiv = diversity.get(ds_name, {})
        v2f_info = ddiv.get("v2f_natural", {})
        if v2f_info:
            print(
                f"  {ds_name} v2f_natural: "
                f"mean_pairwise_cos={v2f_info['mean_pairwise_cos']:.3f} "
                f"(n_q>=2cues={v2f_info['num_questions_with_ge2_cues']}, "
                f"avg_n={v2f_info['avg_n_cues_per_question']:.2f})"
            )
        for arch_name in arch_names:
            if arch_name == "meta_v2f":
                continue
            info = ddiv.get(arch_name)
            if info:
                print(
                    f"  {ds_name} {arch_name}: "
                    f"selected={info['mean_pairwise_cos_selected']:.3f}  "
                    f"all_cands={info['mean_pairwise_cos_all_candidates']:.3f}"
                )


if __name__ == "__main__":
    main()
