"""Fair-backfill eval of iterative query refinement variants vs v2f baseline.

Runs iqr_beta_0.2_t1, iqr_beta_0.4_t1, iqr_beta_0.6_t1, iqr_beta_0.4_t2,
iqr_beta_0.4_filtered, iqr_plus_v2f and a dedicated cosine_baseline and
meta_v2f reference on LoCoMo-30 and synthetic-19 at K=20 and K=50.

Geometry check: for every question computes
    delta_cos_to_gold = mean_gold cos(q_refined, gold) - mean_gold cos(q_0, gold)
to quantify whether refinement pulls the query closer to gold embeddings even
when recall doesn't move.

Usage:
    uv run python iqr_eval.py
    uv run python iqr_eval.py --archs iqr_beta_0.4_t1
    uv run python iqr_eval.py --datasets locomo_30q
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from antipara_cue_gen import MetaV2fDedicated
from associative_recall import Segment
from best_shot import BestshotBase, BestshotResult
from dotenv import load_dotenv
from fair_backfill_eval import (
    BUDGETS,
    DATASETS,
    RESULTS_DIR,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)
from iterative_query_refinement import (
    IqrBeta02T1,
    IqrBeta04Filtered,
    IqrBeta04T1,
    IqrBeta04T2,
    IqrBeta06T1,
    IqrPlusV2f,
    _normalize,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")


class CosineBaseline(BestshotBase):
    """Raw cosine top-K retrieval with q_0 — used as a reference point."""

    arch_name = "cosine_baseline"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        q = self.embed_text(question)
        res = self.store.search(q, top_k=100, conversation_id=conversation_id)
        return BestshotResult(
            segments=list(res.segments),
            metadata={"name": self.arch_name, "num_probes": 1},
        )


ARCH_CLASSES: dict[str, type] = {
    "cosine_baseline": CosineBaseline,
    "meta_v2f": MetaV2fDedicated,
    "iqr_beta_0.2_t1": IqrBeta02T1,
    "iqr_beta_0.4_t1": IqrBeta04T1,
    "iqr_beta_0.6_t1": IqrBeta06T1,
    "iqr_beta_0.4_t2": IqrBeta04T2,
    "iqr_beta_0.4_filtered": IqrBeta04Filtered,
    "iqr_plus_v2f": IqrPlusV2f,
}


def _mean_cos_to_gold(
    q: np.ndarray,
    store,
    source_ids: set[int],
    conv_id: str,
) -> float | None:
    """Mean cosine(q, e_gold) over gold segments present in the store."""
    gold_vecs = []
    for i, cid in enumerate(store.conversation_ids):
        if str(cid) != conv_id:
            continue
        if int(store.turn_ids[i]) in source_ids:
            gold_vecs.append(store.normalized_embeddings[i])
    if not gold_vecs:
        return None
    qn = _normalize(q)
    sims = [float(np.dot(qn, v)) for v in gold_vecs]
    return sum(sims) / len(sims)


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

    # Geometry check — only meaningful for IQR variants exposing q_0/q_refined
    q0 = result.metadata.get("q_0")
    qref = result.metadata.get("q_refined")
    delta_cos_to_gold = None
    cos_q0_gold = None
    cos_qref_gold = None
    if q0 is not None and qref is not None:
        q0v = np.array(q0, dtype=np.float32)
        qrv = np.array(qref, dtype=np.float32)
        cos_q0_gold = _mean_cos_to_gold(q0v, arch.store, source_ids, conv_id)
        cos_qref_gold = _mean_cos_to_gold(qrv, arch.store, source_ids, conv_id)
        if cos_q0_gold is not None and cos_qref_gold is not None:
            delta_cos_to_gold = cos_qref_gold - cos_q0_gold

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
        "hop0_turn_ids": result.metadata.get("hop0_turn_ids", []),
        "num_probes": result.metadata.get("num_probes"),
        "hop0_empty": result.metadata.get("hop0_empty", False),
        "v2f_cues": result.metadata.get("v2f_cues", []),
        "beta": result.metadata.get("beta"),
        "num_iterations": result.metadata.get("num_iterations"),
        "filter_by_median": result.metadata.get("filter_by_median"),
        "geometry": {
            "cos_q0_gold": (round(cos_q0_gold, 4) if cos_q0_gold is not None else None),
            "cos_qref_gold": (
                round(cos_qref_gold, 4) if cos_qref_gold is not None else None
            ),
            "delta_cos_to_gold": (
                round(delta_cos_to_gold, 4) if delta_cos_to_gold is not None else None
            ),
        },
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
    arch_name: str, arch, dataset: str, questions: list[dict]
) -> tuple[list[dict], dict, dict]:
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


def summarize_geometry(results: list[dict]) -> dict:
    q_positive = 0
    q_total = 0
    deltas = []
    per_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        g = r.get("geometry", {})
        d = g.get("delta_cos_to_gold")
        if d is None:
            continue
        deltas.append(d)
        q_total += 1
        if d > 0:
            q_positive += 1
        per_cat[r.get("category", "unknown")].append(d)

    if not deltas:
        return {"n": 0}
    mean_d = sum(deltas) / len(deltas)
    cat_means = {cat: round(sum(vs) / len(vs), 4) for cat, vs in per_cat.items()}
    return {
        "n": q_total,
        "mean_delta_cos_to_gold": round(mean_d, 4),
        "frac_questions_closer_to_gold": round(q_positive / q_total, 4),
        "per_category_mean_delta": cat_means,
    }


def top_categories_delta(by_cat: dict, K: int = 50) -> tuple[list, list]:
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


def build_samples(
    iqr_rows: list[dict],
    cosine_rows: list[dict],
    K: int = 50,
    max_examples: int = 3,
) -> list[dict]:
    """For a handful of questions: show (q_0 top-3 turns, refined top-3 turns,
    gold turns). Useful for qualitative inspection.
    """
    cos_by_key: dict[tuple, list[int]] = {}
    for r in cosine_rows:
        key = (r["conversation_id"], r["question_index"])
        cos_by_key[key] = list(r.get("gold_found_at_K", {}).get(str(K), []))

    samples: list[dict] = []
    for r in iqr_rows:
        if len(samples) >= max_examples:
            break
        key = (r["conversation_id"], r["question_index"])
        iqr_gold = set(r.get("gold_found_at_K", {}).get(str(K), []))
        cos_gold = set(cos_by_key.get(key, []))
        novel = iqr_gold - cos_gold
        if not novel:
            continue
        samples.append(
            {
                "question": r["question"],
                "category": r["category"],
                "gold_turn_ids": r["source_chat_ids"],
                "iqr_gold_at_K": sorted(iqr_gold),
                "cosine_gold_at_K": sorted(cos_gold),
                "novel_vs_cosine": sorted(novel),
                "cos_q0_gold": r.get("geometry", {}).get("cos_q0_gold"),
                "cos_qref_gold": r.get("geometry", {}).get("cos_qref_gold"),
                "hop0_turn_ids": r.get("hop0_turn_ids", []),
            }
        )
    return samples


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
            results, summary, by_cat = run_one(arch_name, arch, ds_name, questions)
            all_results[arch_name][ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
                "geometry": summarize_geometry(results),
            }

    # Top categories for iqr_beta_0.4_t1 on LoCoMo
    top_gaining: list = []
    top_losing: list = []
    if (
        "iqr_beta_0.4_t1" in all_results
        and "locomo_30q" in all_results["iqr_beta_0.4_t1"]
    ):
        top_gaining, top_losing = top_categories_delta(
            all_results["iqr_beta_0.4_t1"]["locomo_30q"]["category_breakdown"],
            K=50,
        )

    # Samples: iqr_beta_0.4_t1 vs cosine_baseline on LoCoMo, K=50
    samples: list[dict] = []
    if (
        "iqr_beta_0.4_t1" in all_results
        and "cosine_baseline" in all_results
        and "locomo_30q" in all_results["iqr_beta_0.4_t1"]
        and "locomo_30q" in all_results["cosine_baseline"]
    ):
        samples = build_samples(
            all_results["iqr_beta_0.4_t1"]["locomo_30q"]["results"],
            all_results["cosine_baseline"]["locomo_30q"]["results"],
            K=50,
            max_examples=3,
        )

    raw: dict = {
        "archs": arch_names,
        "datasets": ds_names,
        "summaries": {
            a: {
                d: {
                    "summary": all_results[a][d]["summary"],
                    "category_breakdown": all_results[a][d]["category_breakdown"],
                    "geometry": all_results[a][d]["geometry"],
                }
                for d in all_results[a]
            }
            for a in all_results
        },
        "top_gaining_categories_iqr_beta_0.4_t1_locomo": top_gaining,
        "top_losing_categories_iqr_beta_0.4_t1_locomo": top_losing,
        "samples_iqr_beta_0.4_t1_vs_cosine_locomo": samples,
    }

    raw_path = RESULTS_DIR / "iterative_query_refinement.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset full results (heavy, but useful for postmortem)
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"iqr_{a}_{d}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "arch": a,
                        "dataset": d,
                        "summary": all_results[a][d]["summary"],
                        "category_breakdown": all_results[a][d]["category_breakdown"],
                        "geometry": all_results[a][d]["geometry"],
                        "results": all_results[a][d]["results"],
                    },
                    f,
                    indent=2,
                    default=str,
                )

    # Markdown report
    md: list[str] = []
    md.append("# Iterative Query Refinement (Hopfield-style attractor)\n")
    md.append(
        "Mechanism: after hop0 retrieval, centroid of retrieved embeddings is "
        "used to pull the query embedding toward the topic cluster. Pure IQR "
        "variants use zero LLM calls; iqr_plus_v2f adds one v2f LLM call "
        "anchored on the refined probe's retrieved context.\n"
    )

    md.append("## Fair-backfill recall\n")
    md.append(
        "| Arch | Dataset | base@20 | arch@20 | d@20 | base@50 | arch@50 | d@50 | llm | embed |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            md.append(
                f"| {a} | {d} | "
                f"{s['baseline_r@20']:.3f} | {s['arch_r@20']:.3f} | "
                f"{s['delta_r@20']:+.3f} | "
                f"{s['baseline_r@50']:.3f} | {s['arch_r@50']:.3f} | "
                f"{s['delta_r@50']:+.3f} | "
                f"{s['avg_llm_calls']:.1f} | {s['avg_embed_calls']:.1f} |"
            )

    # Geometry check
    md.append("\n## Geometry check: cos(q_refined, gold) - cos(q_0, gold)\n")
    md.append(
        "Mean cosine-to-gold delta across questions (positive = refinement "
        "moves the query closer to gold embeddings)."
    )
    md.append("\n| Arch | Dataset | n | mean Δcos | frac questions closer |")
    md.append("|---|---|---:|---:|---:|")
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            g = all_results[a][d].get("geometry", {})
            if g.get("n", 0) == 0:
                continue
            md.append(
                f"| {a} | {d} | {g['n']} | "
                f"{g['mean_delta_cos_to_gold']:+.4f} | "
                f"{g['frac_questions_closer_to_gold']:.3f} |"
            )

    if top_gaining or top_losing:
        md.append("\n## Top categories by Δr@50 (iqr_beta_0.4_t1, LoCoMo-30)\n")
        md.append("Gaining:")
        for g in top_gaining:
            md.append(
                f"  - {g['category']} (n={g['n']}): Δ={g['delta']:+.3f} "
                f"W/T/L={g['W/T/L']}"
            )
        md.append("Losing:")
        for l in top_losing:
            md.append(
                f"  - {l['category']} (n={l['n']}): Δ={l['delta']:+.3f} "
                f"W/T/L={l['W/T/L']}"
            )

    if samples:
        md.append(
            "\n## Samples where iqr_beta_0.4_t1 found gold that cosine missed "
            "(LoCoMo, K=50)\n"
        )
        for ex in samples:
            md.append(
                f"\n- **Q:** {ex['question']} ({ex['category']})\n"
                f"  - cos(q_0, gold)={ex['cos_q0_gold']}, "
                f"cos(q_refined, gold)={ex['cos_qref_gold']}\n"
                f"  - novel vs cosine at K={50}: turns "
                f"{ex['novel_vs_cosine']}"
            )

    # Verdict
    md.append("\n## Verdict\n")
    verdict = "(see numbers above)"
    if (
        "cosine_baseline" in all_results
        and "locomo_30q" in all_results["cosine_baseline"]
        and "meta_v2f" in all_results
        and "locomo_30q" in all_results["meta_v2f"]
    ):
        cos50 = all_results["cosine_baseline"]["locomo_30q"]["summary"]["arch_r@50"]
        v2f50 = all_results["meta_v2f"]["locomo_30q"]["summary"]["arch_r@50"]
        # Collect best IQR variant at K=50 on LoCoMo (among pure variants)
        pure_names = [
            n for n in arch_names if n.startswith("iqr_") and n != "iqr_plus_v2f"
        ]
        best_pure_name = None
        best_pure = -1.0
        for n in pure_names:
            if "locomo_30q" not in all_results.get(n, {}):
                continue
            v = all_results[n]["locomo_30q"]["summary"]["arch_r@50"]
            if v > best_pure:
                best_pure = v
                best_pure_name = n
        plus50 = None
        if (
            "iqr_plus_v2f" in all_results
            and "locomo_30q" in all_results["iqr_plus_v2f"]
        ):
            plus50 = all_results["iqr_plus_v2f"]["locomo_30q"]["summary"]["arch_r@50"]

        if plus50 is not None and plus50 > v2f50 + 0.005:
            verdict = (
                f"**SHIP**: iqr_plus_v2f beats v2f on LoCoMo-30 @K=50 "
                f"({plus50:.3f} vs {v2f50:.3f})."
            )
        elif best_pure_name is not None and best_pure > cos50 + 0.005:
            if best_pure < v2f50 - 0.005:
                verdict = (
                    f"**NARROW-USE**: best pure IQR ({best_pure_name}, "
                    f"{best_pure:.3f}) beats cosine ({cos50:.3f}) but loses "
                    f"to v2f ({v2f50:.3f}). Cheap alternative for "
                    f"budget-constrained (no-LLM) settings."
                )
            else:
                verdict = (
                    f"**NARROW-USE/TIE**: best pure IQR ({best_pure_name}, "
                    f"{best_pure:.3f}) within 0.005 of v2f ({v2f50:.3f}) with "
                    f"zero LLM calls."
                )
        else:
            verdict = (
                f"**ABANDON**: all IQR variants at or below cosine on "
                f"LoCoMo-30 @K=50 (cosine={cos50:.3f}, "
                f"best_pure={best_pure:.3f}"
                + (f", plus_v2f={plus50:.3f}" if plus50 is not None else "")
                + f", v2f={v2f50:.3f})."
            )
    md.append(verdict + "\n")

    md_path = RESULTS_DIR / "iterative_query_refinement.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final table
    print("\n" + "=" * 100)
    print("ITERATIVE QUERY REFINEMENT SUMMARY")
    print("=" * 100)
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            g = all_results[a][d].get("geometry", {})
            dcos = g.get("mean_delta_cos_to_gold")
            dcos_s = f"{dcos:+.4f}" if dcos is not None else "  n/a "
            print(
                f"{a:24s} {d:14s} "
                f"b@20={s['baseline_r@20']:.3f} a@20={s['arch_r@20']:.3f} "
                f"d@20={s['delta_r@20']:+.3f}  "
                f"b@50={s['baseline_r@50']:.3f} a@50={s['arch_r@50']:.3f} "
                f"d@50={s['delta_r@50']:+.3f}  "
                f"Δcos={dcos_s} llm={s['avg_llm_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
