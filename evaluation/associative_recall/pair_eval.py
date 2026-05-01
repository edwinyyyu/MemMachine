"""Fair-backfill eval of pair-level Q-A embedding variants.

Runs pair_roleprefix, pair_noprefix, pair_plus_v2f and meta_v2f baseline on
LoCoMo-30 and synthetic-19 at K=20 and K=50.

Also measures orthogonality: what fraction of the gold turns found by a
pair-variant is NOT found by the v2f baseline (at K=50)?

Usage:
    uv run python pair_eval.py
    uv run python pair_eval.py --archs pair_roleprefix,pair_plus_v2f
    uv run python pair_eval.py --datasets locomo_30q
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
from pair_embedding import (
    ARCH_CLASSES as PAIR_ARCH_CLASSES,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")

ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    **PAIR_ARCH_CLASSES,
}


def evaluate_question(arch, question: dict) -> dict:
    """Run arch on a question, produce fair-backfill metrics + metadata."""
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
        "num_pair_hits": result.metadata.get("num_pair_hits"),
        "pair_scores": result.metadata.get("pair_scores", []),
        "v2f_cues": result.metadata.get("v2f_cues", []),
        "use_role_prefix": result.metadata.get("use_role_prefix"),
        "attenuation": result.metadata.get("attenuation"),
        "top_m_pairs": result.metadata.get("top_m_pairs"),
        "pair_index_size": result.metadata.get("pair_index_size"),
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
            f"r@20 d={c['delta_r@20']:+.3f} "
            f"r@50 d={c['delta_r@50']:+.3f} "
            f"W/T/L@50={c['W/T/L_r@50']}"
        )

    return results, summary, by_cat


def orthogonality(
    arch_rows: list[dict],
    v2f_rows: list[dict],
    K: int = 50,
) -> dict:
    """For each question, compute fraction of arch's found gold that is NOT
    in v2f's found gold at budget K. Returns averages + counts."""
    # Index v2f rows by (conv_id, question_index)
    v2f_by_q = {(r["conversation_id"], r["question_index"]): r for r in v2f_rows}
    per_q: list[dict] = []
    fractions: list[float] = []
    arch_found_total = 0
    v2f_found_total = 0
    novel_total = 0
    intersect_total = 0
    for r in arch_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_r = v2f_by_q.get(key)
        if v2f_r is None:
            continue
        arch_gold = set(r.get("gold_found_at_K", {}).get(str(K), []))
        v2f_gold = set(v2f_r.get("gold_found_at_K", {}).get(str(K), []))
        novel = arch_gold - v2f_gold
        intersect = arch_gold & v2f_gold
        arch_found_total += len(arch_gold)
        v2f_found_total += len(v2f_gold)
        novel_total += len(novel)
        intersect_total += len(intersect)
        frac = len(novel) / len(arch_gold) if arch_gold else 0.0
        fractions.append(frac)
        per_q.append(
            {
                "conversation_id": r["conversation_id"],
                "question_index": r["question_index"],
                "category": r.get("category"),
                "arch_gold_count": len(arch_gold),
                "v2f_gold_count": len(v2f_gold),
                "novel_count": len(novel),
                "intersect_count": len(intersect),
                "frac_novel": round(frac, 3),
            }
        )

    mean_frac = float(np.mean(fractions)) if fractions else 0.0
    return {
        "K": K,
        "n_questions": len(per_q),
        "mean_frac_novel_per_q": round(mean_frac, 4),
        "pooled_arch_found": arch_found_total,
        "pooled_v2f_found": v2f_found_total,
        "pooled_novel": novel_total,
        "pooled_intersect": intersect_total,
        "pooled_frac_novel": round(novel_total / arch_found_total, 4)
        if arch_found_total
        else 0.0,
        "per_question": per_q,
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


def sample_pairs_for_report(
    arch,
    dataset_name: str,
    n: int = 4,
) -> list[dict]:
    """Dump a few pair_text examples with their constituent turn ids."""
    idx = arch._pair_index
    out: list[dict] = []
    for k in range(min(n, len(idx))):
        cid, sa, sb, text = idx.pairs[k]
        seg_a = arch.store.segments[sa]
        seg_b = arch.store.segments[sb]
        out.append(
            {
                "conversation_id": cid,
                "turn_a": seg_a.turn_id,
                "role_a": seg_a.role,
                "turn_b": seg_b.turn_id,
                "role_b": seg_b.role,
                "pair_text_preview": text[:300],
            }
        )
    return out


def dump_conversation_pairs(
    arch,
    path: Path,
) -> None:
    """Dump the pair index (reusable for analysis)."""
    idx = arch._pair_index
    payload = {
        "use_role_prefix": idx.use_role_prefix,
        "include_reverse": idx.include_reverse,
        "num_pairs": len(idx),
        "pairs": [
            {
                "conversation_id": cid,
                "seg_index_a": sa,
                "seg_index_b": sb,
                "turn_a": arch.store.segments[sa].turn_id,
                "role_a": arch.store.segments[sa].role,
                "turn_b": arch.store.segments[sb].turn_id,
                "role_b": arch.store.segments[sb].role,
            }
            for cid, sa, sb, _text in idx.pairs
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


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

    # Track first pair-variant arch per dataset so we can dump the pair
    # index exactly once per dataset.
    pair_dumped: set[str] = set()

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

            # Dump pair index (once per dataset, from any pair-variant)
            if arch_name != "meta_v2f" and ds_name not in pair_dumped:
                pair_path = RESULTS_DIR / f"conversation_pairs_{ds_name}.json"
                try:
                    dump_conversation_pairs(arch, pair_path)
                    print(f"  Saved pair index: {pair_path}")
                    pair_dumped.add(ds_name)
                except Exception as e:
                    print(f"  WARNING: could not dump pair index: {e}")

    # Sample pairs (for report)
    sample_pairs: dict[str, list[dict]] = {}
    for ds_name in ds_names:
        for arch_name in arch_names:
            if arch_name == "meta_v2f":
                continue
            if ds_name not in all_results.get(arch_name, {}):
                continue
            # Build one instance to pull sample pairs (cheap — embeddings
            # are cached)
            store, _ = load_dataset(ds_name)
            cls = ARCH_CLASSES[arch_name]
            arch = cls(store)
            sample_pairs[f"{ds_name}/{arch_name}"] = sample_pairs_for_report(
                arch, ds_name, n=3
            )
            break  # one sample per dataset is enough

    # Orthogonality: for each pair-arch vs meta_v2f, per dataset, at K=50
    orth: dict[str, dict] = {}
    for ds_name in ds_names:
        orth[ds_name] = {}
        v2f_rows = all_results.get("meta_v2f", {}).get(ds_name, {}).get("results")
        if not v2f_rows:
            continue
        for arch_name in arch_names:
            if arch_name == "meta_v2f":
                continue
            arch_rows = all_results.get(arch_name, {}).get(ds_name, {}).get("results")
            if not arch_rows:
                continue
            orth[ds_name][arch_name] = orthogonality(arch_rows, v2f_rows, K=50)

    # Top gaining/losing categories per pair variant on LoCoMo
    top_cats: dict[str, dict] = {}
    if "locomo_30q" in ds_names:
        for arch_name in arch_names:
            if arch_name == "meta_v2f":
                continue
            if "locomo_30q" not in all_results.get(arch_name, {}):
                continue
            g, l = top_categories_delta(
                all_results[arch_name]["locomo_30q"]["category_breakdown"],
                K=50,
            )
            top_cats[arch_name] = {"gaining": g, "losing": l}

    # Raw JSON
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
        "orthogonality": orth,
        "top_categories_locomo_30q": top_cats,
        "sample_pairs": sample_pairs,
    }
    raw_path = RESULTS_DIR / "pair_embedding_study.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset full results
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"pair_{a}_{d}.json"
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
    md.append("# Pair-Level Q-A Embedding Study\n")
    md.append(
        "Motivation: queries often ask questions; turns in the corpus are "
        "user asks and assistant replies separately. Embedding "
        "(user_turn, assistant_reply) as a single unit captures the Q-A "
        "structure. Pair-hits promote both constituent turns via an "
        "attenuated score.\n"
    )
    md.append(
        "Zero-LLM architecturally — the pair variants alone add no LLM "
        "calls beyond the main cosine retrieval. `pair_plus_v2f` layers "
        "v2f cues on top as the intended ship test.\n"
    )

    # Pair index stats
    md.append("## Pair index stats\n")
    md.append("| Dataset | num_pairs | use_role_prefix |")
    md.append("|---|---:|---|")
    for ds_name in ds_names:
        # Take a pair row to read out size
        size = None
        for arch_name in arch_names:
            if arch_name == "meta_v2f":
                continue
            rows = all_results.get(arch_name, {}).get(ds_name, {}).get("results", [])
            if rows:
                size = rows[0].get("pair_index_size")
                break
        md.append(f"| {ds_name} | {size} | True (default) |")

    # Recall table
    md.append("\n## Fair-backfill recall\n")
    md.append(
        "| Arch | Dataset | base@20 | arch@20 | d@20 | base@50 | arch@50 | "
        "d@50 | llm/q |"
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

    # Orthogonality
    md.append("\n## Orthogonality vs meta_v2f (K=50)\n")
    md.append(
        "% of arch's found gold turns NOT retrieved by meta_v2f at K=50. "
        "Higher = pair index covers different gold than v2f does.\n"
    )
    md.append(
        "| Dataset | Arch | pooled_arch_found | pooled_novel | "
        "pooled_frac_novel | mean_frac_novel/q |"
    )
    md.append("|---|---|---:|---:|---:|---:|")
    for ds_name in ds_names:
        for arch_name, info in orth.get(ds_name, {}).items():
            md.append(
                f"| {ds_name} | {arch_name} | "
                f"{info['pooled_arch_found']} | {info['pooled_novel']} | "
                f"{info['pooled_frac_novel']:.3f} | "
                f"{info['mean_frac_novel_per_q']:.3f} |"
            )

    # Top categories
    md.append("\n## Top categories by d_r@50 (on LoCoMo-30)\n")
    for arch_name, tc in top_cats.items():
        md.append(f"\n### {arch_name}\n")
        md.append("Gaining:")
        for g in tc["gaining"]:
            md.append(
                f"  - {g['category']} (n={g['n']}): "
                f"delta={g['delta']:+.3f} W/T/L={g['W/T/L']}"
            )
        md.append("Losing:")
        for l in tc["losing"]:
            md.append(
                f"  - {l['category']} (n={l['n']}): "
                f"delta={l['delta']:+.3f} W/T/L={l['W/T/L']}"
            )

    # Sample pairs
    md.append("\n## Sample pair texts\n")
    for k, samples in sample_pairs.items():
        md.append(f"\n### {k}")
        for i, sp in enumerate(samples):
            md.append(
                f"\n{i + 1}. conv={sp['conversation_id']} "
                f"turns=({sp['turn_a']}/{sp['role_a']}, "
                f"{sp['turn_b']}/{sp['role_b']})"
            )
            md.append(f"   `{sp['pair_text_preview']}`")

    # Verdict
    md.append("\n## Verdict\n")
    verdict_lines = []
    v2f_lc50 = (
        all_results.get("meta_v2f", {})
        .get("locomo_30q", {})
        .get("summary", {})
        .get("arch_r@50")
    )
    base_lc50 = (
        all_results.get("meta_v2f", {})
        .get("locomo_30q", {})
        .get("summary", {})
        .get("baseline_r@50")
    )
    for arch_name in ["pair_plus_v2f", "pair_roleprefix", "pair_noprefix"]:
        if arch_name not in all_results:
            continue
        if "locomo_30q" not in all_results[arch_name]:
            continue
        lc50 = all_results[arch_name]["locomo_30q"]["summary"]["arch_r@50"]
        if arch_name == "pair_plus_v2f":
            if v2f_lc50 is not None and lc50 > v2f_lc50 + 0.005:
                verdict_lines.append(
                    f"**SHIP candidate**: `pair_plus_v2f` LoCoMo@50="
                    f"{lc50:.3f} beats meta_v2f={v2f_lc50:.3f}."
                )
            elif v2f_lc50 is not None:
                verdict_lines.append(
                    f"`pair_plus_v2f` LoCoMo@50={lc50:.3f} vs "
                    f"meta_v2f={v2f_lc50:.3f} — no gain over v2f."
                )
        else:
            if base_lc50 is not None and lc50 > base_lc50 + 0.005:
                verdict_lines.append(
                    f"Zero-LLM `{arch_name}` LoCoMo@50={lc50:.3f} beats "
                    f"cosine baseline={base_lc50:.3f}."
                )
            elif base_lc50 is not None:
                verdict_lines.append(
                    f"`{arch_name}` LoCoMo@50={lc50:.3f} vs "
                    f"cosine baseline={base_lc50:.3f} — no zero-LLM win."
                )
    if verdict_lines:
        md.append("\n".join(verdict_lines))
    else:
        md.append("(see tables above)")

    md_path = RESULTS_DIR / "pair_embedding_study.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final table
    print("\n" + "=" * 100)
    print("PAIR EMBEDDING SUMMARY")
    print("=" * 100)
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:22s} {d:14s} "
                f"b@20={s['baseline_r@20']:.3f} "
                f"a@20={s['arch_r@20']:.3f} "
                f"d@20={s['delta_r@20']:+.3f}  "
                f"b@50={s['baseline_r@50']:.3f} "
                f"a@50={s['arch_r@50']:.3f} "
                f"d@50={s['delta_r@50']:+.3f}  "
                f"llm={s['avg_llm_calls']:.1f}"
            )

    print("\n--- Orthogonality vs meta_v2f @K=50 ---")
    for ds_name in ds_names:
        for arch_name, info in orth.get(ds_name, {}).items():
            print(
                f"  {ds_name} {arch_name}: "
                f"pooled_novel={info['pooled_novel']}/"
                f"{info['pooled_arch_found']} "
                f"({info['pooled_frac_novel']:.3f}), "
                f"mean/q={info['mean_frac_novel_per_q']:.3f}"
            )


if __name__ == "__main__":
    main()
