"""Fair-backfill eval of anchor-turn expansion variants vs v2f baseline.

Runs anchor_exp_3anchors, anchor_exp_5anchors, anchor_exp_plus_v2f plus a
dedicated meta_v2f baseline on LoCoMo-30 and synthetic-19 at K=20 and K=50
using the fair-backfill methodology.

Also computes orthogonality: fraction of gold turns found by the variant that
were NOT found by v2f at K=50. Extracts 3 qualitative (anchor, imagined_cue,
gold_retrieved) trios demonstrating the mechanism.

Usage:
    uv run python anchor_exp_eval.py
    uv run python anchor_exp_eval.py --archs anchor_exp_3anchors
    uv run python anchor_exp_eval.py --datasets locomo_30q
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from anchor_expansion import (
    AnchorExp3Anchors,
    AnchorExp5Anchors,
    AnchorExpPlusV2f,
)
from antipara_cue_gen import MetaV2fDedicated
from associative_recall import Segment
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

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")

ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    "anchor_exp_3anchors": AnchorExp3Anchors,
    "anchor_exp_5anchors": AnchorExp5Anchors,
    "anchor_exp_plus_v2f": AnchorExpPlusV2f,
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
        "anchor_cues": result.metadata.get("anchor_cues", []),
        "anchor_turn_ids": result.metadata.get("anchor_turn_ids", []),
        "v2f_cues": result.metadata.get("v2f_cues", []),
        "probe_outcomes": result.metadata.get("probe_outcomes", []),
        "v2f_outcomes": result.metadata.get("v2f_outcomes", []),
        "num_probes": result.metadata.get("num_probes"),
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
        # Save caches every question to avoid data loss on long runs
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


def compute_orthogonality(
    anchor_rows: list[dict],
    v2f_rows: list[dict],
    K: int = 50,
) -> dict:
    """For each question: fraction of gold turns found by `anchor` variant at K
    that were NOT found by v2f at K.
    """
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    total_gold = 0
    novel_gold = 0
    per_q: list[dict] = []
    for r in anchor_rows:
        key = (r["conversation_id"], r["question_index"])
        gold = set(r["gold_found_at_K"].get(str(K), []))
        v2f_gold = v2f_by_key.get(key, set())
        novel = gold - v2f_gold
        total_gold += len(gold)
        novel_gold += len(novel)
        per_q.append(
            {
                "conversation_id": r["conversation_id"],
                "question_index": r["question_index"],
                "gold_count": len(gold),
                "novel_vs_v2f": len(novel),
                "novel_turn_ids": sorted(novel),
            }
        )

    frac_novel = novel_gold / total_gold if total_gold else 0.0
    return {
        "total_gold": total_gold,
        "novel_vs_v2f": novel_gold,
        "fraction_novel": round(frac_novel, 4),
        "per_question": per_q,
    }


def qualitative_trios(
    anchor_rows: list[dict],
    v2f_rows: list[dict],
    K: int = 50,
    max_examples: int = 5,
) -> list[dict]:
    """Find examples where anchor_exp found gold via an imagined cue. Attributes
    each novel gold back to the (anchor_turn, imagined_cue) pair via probe_outcomes.
    """
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    examples: list[dict] = []
    for r in anchor_rows:
        if len(examples) >= max_examples:
            break
        key = (r["conversation_id"], r["question_index"])
        gold = set(r["gold_found_at_K"].get(str(K), []))
        v2f_gold = v2f_by_key.get(key, set())
        novel = gold - v2f_gold
        if not novel and not gold:
            continue

        # Prefer novel-vs-v2f hits; fall back to any hit
        for target_set, is_novel in ((novel, True), (gold, False)):
            if not target_set:
                continue
            for probe in r.get("probe_outcomes", []):
                hit = target_set & set(probe.get("retrieved_turn_ids", []))
                if hit:
                    anchor_text = probe.get("anchor_text", "")
                    if len(anchor_text) > 300:
                        anchor_text = anchor_text[:300].rstrip() + "..."
                    examples.append(
                        {
                            "question": r["question"],
                            "category": r["category"],
                            "anchor_turn_id": probe["source_turn_id"],
                            "anchor_role": probe.get("source_role", ""),
                            "anchor_text": anchor_text,
                            "imagined_cue": probe["cue"],
                            "gold_turn_id_retrieved": sorted(hit)[0],
                            "novel_vs_v2f": is_novel,
                        }
                    )
                    break
            if examples and examples[-1]["question"] == r["question"]:
                break
    return examples


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


def drift_audit(anchor_rows: list[dict], max_cues: int = 30) -> dict:
    """Heuristic audit of whether imagined cues drift back toward the question
    rather than continuing the anchor. Measures token overlap.
    """
    import re

    def toks(s: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", s)}

    cue_anchor_overlaps: list[float] = []
    cue_question_overlaps: list[float] = []
    n_cues = 0
    for r in anchor_rows:
        q_toks = toks(r["question"])
        for probe in r.get("probe_outcomes", []):
            cue = probe.get("cue", "")
            anchor = probe.get("anchor_text", "")
            if not cue or not anchor:
                continue
            a_toks = toks(anchor)
            c_toks = toks(cue)
            if not c_toks:
                continue
            cue_anchor_overlaps.append(len(c_toks & a_toks) / len(c_toks))
            cue_question_overlaps.append(len(c_toks & q_toks) / len(c_toks))
            n_cues += 1
            if n_cues >= max_cues * 20:
                break
        if n_cues >= max_cues * 20:
            break

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "n_cues_scored": n_cues,
        "mean_cue_anchor_overlap": mean(cue_anchor_overlaps),
        "mean_cue_question_overlap": mean(cue_question_overlaps),
    }


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
            }

    # Orthogonality vs meta_v2f
    orthogonality: dict[str, dict] = {}
    if "meta_v2f" in all_results:
        for arch_name in arch_names:
            if arch_name == "meta_v2f":
                continue
            orthogonality[arch_name] = {}
            for ds_name in ds_names:
                if ds_name not in all_results[arch_name]:
                    continue
                if ds_name not in all_results["meta_v2f"]:
                    continue
                rows = all_results[arch_name][ds_name]["results"]
                v2f_rows = all_results["meta_v2f"][ds_name]["results"]
                orth = compute_orthogonality(rows, v2f_rows, K=50)
                orthogonality[arch_name][ds_name] = {
                    "total_gold": orth["total_gold"],
                    "novel_vs_v2f": orth["novel_vs_v2f"],
                    "fraction_novel": orth["fraction_novel"],
                }

    # Qualitative trios from anchor_exp_3anchors on LoCoMo
    trios: list[dict] = []
    if (
        "anchor_exp_3anchors" in all_results
        and "meta_v2f" in all_results
        and "locomo_30q" in all_results["anchor_exp_3anchors"]
        and "locomo_30q" in all_results["meta_v2f"]
    ):
        trios = qualitative_trios(
            all_results["anchor_exp_3anchors"]["locomo_30q"]["results"],
            all_results["meta_v2f"]["locomo_30q"]["results"],
            K=50,
            max_examples=5,
        )

    # Top categories for anchor_exp_3anchors on LoCoMo
    top_gaining: list = []
    top_losing: list = []
    if (
        "anchor_exp_3anchors" in all_results
        and "locomo_30q" in all_results["anchor_exp_3anchors"]
    ):
        top_gaining, top_losing = top_categories_delta(
            all_results["anchor_exp_3anchors"]["locomo_30q"]["category_breakdown"],
            K=50,
        )

    # Drift audit for main variant on LoCoMo (helps diagnose if LLM is
    # riffing on question rather than anchor)
    drift: dict = {}
    if (
        "anchor_exp_3anchors" in all_results
        and "locomo_30q" in all_results["anchor_exp_3anchors"]
    ):
        drift = drift_audit(all_results["anchor_exp_3anchors"]["locomo_30q"]["results"])

    raw: dict = {
        "archs": arch_names,
        "datasets": ds_names,
        "summaries": {
            a: {
                d: {
                    "summary": all_results[a][d]["summary"],
                    "category_breakdown": all_results[a][d]["category_breakdown"],
                }
                for d in all_results[a]
            }
            for a in all_results
        },
        "orthogonality_vs_v2f_at_K50": orthogonality,
        "qualitative_trios": trios,
        "top_gaining_categories": top_gaining,
        "top_losing_categories": top_losing,
        "drift_audit": drift,
    }

    raw_path = RESULTS_DIR / "anchor_expansion.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset full results
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"anchor_{a}_{d}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "arch": a,
                        "dataset": d,
                        "summary": all_results[a][d]["summary"],
                        "category_breakdown": all_results[a][d]["category_breakdown"],
                        "results": all_results[a][d]["results"],
                    },
                    f,
                    indent=2,
                    default=str,
                )

    # Markdown report
    md: list[str] = []
    md.append("# Anchor-Turn Expansion Retrieval\n")
    md.append(
        "Motivation: v2f imagines corpus content forward from the question, "
        "so cues land near the query in embedding space. Anchor expansion "
        "starts from ACTUAL retrieved turns and imagines their continuations "
        "— cues are anchored in real corpus vocabulary. Per-cue attribution "
        "showed winning cues sit ~0.575 cosine from gold and are entity-rich; "
        "anchoring in retrieved content should push cues closer to winners.\n"
    )

    md.append("## Fair-backfill recall\n")
    md.append(
        "| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
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
                f"{s['avg_llm_calls']:.1f} |"
            )

    if orthogonality:
        md.append("\n## Orthogonality vs v2f (K=50)\n")
        md.append(
            "Fraction of gold turns found by the variant that v2f did NOT find.\n"
        )
        md.append("| Arch | Dataset | gold_found | novel_vs_v2f | frac_novel |")
        md.append("|---|---|---:|---:|---:|")
        for a in orthogonality:
            for d in orthogonality[a]:
                o = orthogonality[a][d]
                md.append(
                    f"| {a} | {d} | {o['total_gold']} | "
                    f"{o['novel_vs_v2f']} | {o['fraction_novel']:.3f} |"
                )

    if trios:
        md.append("\n## Qualitative trios (anchor_exp_3anchors, LoCoMo, K=50)\n")
        md.append(
            "Each row: anchor turn the LLM imagined around → imagined cue → "
            "gold turn that cue retrieved."
        )
        for ex in trios:
            novel_tag = " **(novel vs v2f)**" if ex.get("novel_vs_v2f") else ""
            md.append(
                f"\n- **Q:** {ex['question']}{novel_tag}\n"
                f"  - Anchor ({ex['anchor_role']}, turn_id="
                f"{ex['anchor_turn_id']}): _{ex['anchor_text']}_\n"
                f"  - Imagined cue: _{ex['imagined_cue']}_\n"
                f"  - Gold turn retrieved: {ex['gold_turn_id_retrieved']}"
            )

    if top_gaining or top_losing:
        md.append("\n## Top categories by Δr@50 (anchor_exp_3anchors, LoCoMo-30)\n")
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

    if drift:
        md.append("\n## Drift audit (anchor_exp_3anchors, LoCoMo)\n")
        md.append(
            f"Over {drift['n_cues_scored']} imagined cues, mean token overlap "
            f"with anchor = {drift['mean_cue_anchor_overlap']:.3f}, mean "
            f"overlap with question = {drift['mean_cue_question_overlap']:.3f}. "
            f"Higher anchor overlap than question overlap → cues are following "
            f"anchor content; reverse → LLM is drifting back to the question."
        )

    # Verdict
    md.append("\n## Verdict\n")
    verdict = "(see numbers above)"
    if (
        "meta_v2f" in all_results
        and "locomo_30q" in all_results["meta_v2f"]
        and "anchor_exp_3anchors" in all_results
        and "locomo_30q" in all_results["anchor_exp_3anchors"]
    ):
        v2f50 = all_results["meta_v2f"]["locomo_30q"]["summary"]["arch_r@50"]
        anchor50 = all_results["anchor_exp_3anchors"]["locomo_30q"]["summary"][
            "arch_r@50"
        ]
        union50 = None
        if (
            "anchor_exp_plus_v2f" in all_results
            and "locomo_30q" in all_results["anchor_exp_plus_v2f"]
        ):
            union50 = all_results["anchor_exp_plus_v2f"]["locomo_30q"]["summary"][
                "arch_r@50"
            ]

        if anchor50 > v2f50 + 0.005:
            verdict = (
                f"**SHIP**: anchor_exp_3anchors beats v2f on LoCoMo-30 @K=50 "
                f"({anchor50:.3f} vs {v2f50:.3f})."
            )
        elif union50 is not None and union50 > v2f50 + 0.005:
            verdict = (
                f"**SUPPLEMENT**: anchor_exp alone ties/loses, but "
                f"anchor_exp_plus_v2f beats v2f @K=50 "
                f"({union50:.3f} vs {v2f50:.3f})."
            )
        else:
            verdict = (
                f"**ABANDON**: neither variant beats v2f on LoCoMo-30 @K=50 "
                f"(v2f={v2f50:.3f}, anchor_3={anchor50:.3f}"
                + (f", anchor+v2f={union50:.3f}" if union50 is not None else "")
                + ")."
            )
    md.append(verdict + "\n")

    md_path = RESULTS_DIR / "anchor_expansion.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final table
    print("\n" + "=" * 100)
    print("ANCHOR EXPANSION SUMMARY")
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


if __name__ == "__main__":
    main()
