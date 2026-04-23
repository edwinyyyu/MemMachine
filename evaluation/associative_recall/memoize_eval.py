"""Fair-backfill eval of cue memoization variants vs v2f baseline.

Runs memoize_m2, memoize_m3, memoize_filtered, memoize_plus_v2f plus a
dedicated meta_v2f baseline on LoCoMo-30 and synthetic-19 at K=20 and K=50
using the fair-backfill methodology.

Also reports:
  - Orthogonality vs v2f: fraction of gold turns the variant finds at K=50
    that v2f does NOT find at K=50.
  - Mean query-to-nearest-exemplar cosine (weakness diagnostic).
  - Qualitative sample (new_q, nearest_exemplar, reused_cue, gold_found).
  - Top 2 gain / 2 loss categories for memoize_m2.

Usage:
    uv run python memoize_eval.py
    uv run python memoize_eval.py --archs memoize_m2
    uv run python memoize_eval.py --datasets locomo_30q
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import Segment, SegmentStore
from best_shot import (
    BestshotBase,
    BestshotResult,
    V2F_PROMPT,
    _format_segments,
    _parse_cues,
)
from fair_backfill_eval import (
    BUDGETS,
    DATASETS,
    RESULTS_DIR,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)
from cue_memoization import (
    ARCH_CLASSES as MEMOIZE_ARCH_CLASSES,
    MemoizeEmbeddingCache,
    MemoizeLLMCache,
    load_exemplar_bank,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")


class MetaV2fMemoizeCache(BestshotBase):
    """v2f baseline that writes only to memoize caches.

    Same retrieval logic as `best_shot.MetaV2f` but isolates from sibling
    agents' writes. Reads from shared caches for hits.
    """

    arch_name = "meta_v2f"

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        super().__init__(store, client)
        self.embedding_cache = MemoizeEmbeddingCache()
        self.llm_cache = MemoizeLLMCache()

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        hop0_segments = list(hop0.segments)
        hop0_scores = list(hop0.scores)

        score_map: dict[int, float] = {}
        seg_map: dict[int, Segment] = {}
        for seg, sc in zip(hop0_segments, hop0_scores):
            score_map[seg.index] = sc
            seg_map[seg.index] = seg

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(hop0_segments)
        )
        prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]

        v2f_outcomes: list[dict] = []
        for cue in cues:
            if not cue.strip():
                continue
            cue_emb = self.embed_text(cue)
            res = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id
            )
            retrieved_ids = []
            for seg, sc in zip(res.segments, res.scores):
                retrieved_ids.append(seg.index)
                if seg.index not in score_map or sc > score_map[seg.index]:
                    score_map[seg.index] = sc
                if seg.index not in seg_map:
                    seg_map[seg.index] = seg
            v2f_outcomes.append(
                {
                    "cue": cue,
                    "retrieved_turn_ids": [
                        seg_map[idx].turn_id for idx in retrieved_ids
                    ],
                }
            )

        ranked = sorted(
            score_map.keys(), key=lambda i: score_map[i], reverse=True
        )
        all_segments = [seg_map[i] for i in ranked]

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": "meta_v2f",
                "output": output,
                "v2f_cues": cues,
                "v2f_outcomes": v2f_outcomes,
                "cues": cues,  # compat with fewshot eval consumers
            },
        )


ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fMemoizeCache,
    **MEMOIZE_ARCH_CLASSES,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


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
        # Carry over memoize-specific metadata if present
        "nearest_exemplars": result.metadata.get("nearest_exemplars", []),
        "memoized_cues": result.metadata.get("memoized_cues", []),
        "memoized_probe_outcomes": result.metadata.get(
            "memoized_probe_outcomes", []
        ),
        "v2f_cues": result.metadata.get("v2f_cues", []),
        "v2f_outcomes": result.metadata.get("v2f_outcomes", []),
        "ran_v2f": result.metadata.get("ran_v2f", False),
        "nearest_exemplar_sim": result.metadata.get("nearest_exemplar_sim"),
        "num_probes": result.metadata.get("num_probes"),
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


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def compute_orthogonality(
    arch_rows: list[dict],
    v2f_rows: list[dict],
    K: int = 50,
) -> dict:
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    total_gold = 0
    novel_gold = 0
    for r in arch_rows:
        key = (r["conversation_id"], r["question_index"])
        gold = set(r["gold_found_at_K"].get(str(K), []))
        v2f_gold = v2f_by_key.get(key, set())
        novel = gold - v2f_gold
        total_gold += len(gold)
        novel_gold += len(novel)
    frac_novel = novel_gold / total_gold if total_gold else 0.0
    return {
        "total_gold": total_gold,
        "novel_vs_v2f": novel_gold,
        "fraction_novel": round(frac_novel, 4),
    }


def mean_nearest_exemplar_sim(rows: list[dict]) -> dict:
    sims = [
        r["nearest_exemplar_sim"]
        for r in rows
        if r.get("nearest_exemplar_sim") is not None
    ]
    if not sims:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(sims),
        "mean": round(sum(sims) / len(sims), 4),
        "min": round(min(sims), 4),
        "max": round(max(sims), 4),
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


def qualitative_samples(
    rows: list[dict],
    v2f_rows: list[dict],
    K: int = 50,
    max_examples: int = 5,
) -> list[dict]:
    """Find examples where a reused cue retrieved a gold turn that v2f missed.

    Each sample: (new_q, nearest_exemplar, reused_cue, gold_found_turn_id).
    """
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    examples: list[dict] = []
    for r in rows:
        if len(examples) >= max_examples:
            break
        key = (r["conversation_id"], r["question_index"])
        gold = set(r["gold_found_at_K"].get(str(K), []))
        v2f_gold = v2f_by_key.get(key, set())
        novel = gold - v2f_gold

        for target_set, is_novel in ((novel, True), (gold, False)):
            if not target_set:
                continue
            for probe in r.get("memoized_probe_outcomes", []):
                hit = target_set & set(probe.get("retrieved_turn_ids", []))
                if hit:
                    examples.append(
                        {
                            "new_question": r["question"],
                            "new_category": r["category"],
                            "nearest_exemplar_question": probe.get(
                                "source_question", ""
                            ),
                            "nearest_exemplar_dataset": probe.get(
                                "source_dataset", ""
                            ),
                            "nearest_exemplar_sim": probe.get(
                                "source_sim", 0.0
                            ),
                            "reused_cue": probe["cue"],
                            "gold_found_turn_id": sorted(hit)[0],
                            "novel_vs_v2f": is_novel,
                        }
                    )
                    break
            if examples and examples[-1]["new_question"] == r["question"]:
                break
    return examples


def probe_top1_gold_rate(rows: list[dict], cue_key: str) -> dict:
    """Fraction of cues whose top-1 retrieval (first in retrieved_turn_ids)
    equals a gold turn, across all rows for the given cue source.

    `cue_key`: either "memoized_probe_outcomes" or "v2f_outcomes".
    """
    total_cues = 0
    hit_top1 = 0
    hit_any_top_k = 0
    for r in rows:
        gold = set(r["source_chat_ids"])
        for probe in r.get(cue_key, []):
            retrieved = probe.get("retrieved_turn_ids", [])
            if not retrieved:
                continue
            total_cues += 1
            if retrieved[0] in gold:
                hit_top1 += 1
            if set(retrieved) & gold:
                hit_any_top_k += 1
    return {
        "total_cues": total_cues,
        "top1_hit_rate": (
            round(hit_top1 / total_cues, 4) if total_cues else 0.0
        ),
        "any_topk_hit_rate": (
            round(hit_any_top_k / total_cues, 4) if total_cues else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    # Load exemplar bank once (shared across arch instances)
    exemplars = load_exemplar_bank()
    print(
        f"Loaded {len(exemplars)} exemplars from bank "
        f"(across {len(set((e['dataset'], e['conversation_id']) for e in exemplars))} conv-dataset pairs)."
    )

    all_results: dict[str, dict] = defaultdict(dict)

    for ds_name in ds_names:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        for arch_name in arch_names:
            cls = ARCH_CLASSES[arch_name]
            if arch_name == "meta_v2f":
                arch = cls(store)
            else:
                arch = cls(store, exemplars=exemplars)
            results, summary, by_cat = run_one(
                arch_name, arch, ds_name, questions
            )
            all_results[arch_name][ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
            }

    # Orthogonality vs v2f
    orthogonality: dict[str, dict] = {}
    if "meta_v2f" in all_results:
        for arch_name in arch_names:
            if arch_name == "meta_v2f":
                continue
            orthogonality[arch_name] = {}
            for ds_name in ds_names:
                if ds_name not in all_results.get(arch_name, {}):
                    continue
                if ds_name not in all_results.get("meta_v2f", {}):
                    continue
                rows = all_results[arch_name][ds_name]["results"]
                v2f_rows = all_results["meta_v2f"][ds_name]["results"]
                orth = compute_orthogonality(rows, v2f_rows, K=50)
                orthogonality[arch_name][ds_name] = orth

    # Mean query-to-nearest-exemplar similarity (memoize variants only)
    nearest_sim_stats: dict[str, dict] = {}
    for arch_name in arch_names:
        if not arch_name.startswith("memoize"):
            continue
        nearest_sim_stats[arch_name] = {}
        for ds_name in ds_names:
            if ds_name not in all_results.get(arch_name, {}):
                continue
            rows = all_results[arch_name][ds_name]["results"]
            nearest_sim_stats[arch_name][ds_name] = mean_nearest_exemplar_sim(
                rows
            )

    # Top 2 gain / 2 loss categories for memoize_m2 on combined datasets
    top_gaining: list = []
    top_losing: list = []
    if (
        "memoize_m2" in all_results
        and all_results["memoize_m2"]
    ):
        # Merge category_breakdown across datasets
        merged: dict[str, dict] = {}
        for ds_name in all_results["memoize_m2"]:
            by_cat = all_results["memoize_m2"][ds_name]["category_breakdown"]
            for cat, c in by_cat.items():
                key = f"{cat}"
                if key not in merged:
                    merged[key] = {
                        "n": 0,
                        "delta_r@20_sum": 0.0,
                        "delta_r@50_sum": 0.0,
                        "wins": 0,
                        "ties": 0,
                        "losses": 0,
                    }
                merged[key]["n"] += c["n"]
                merged[key]["delta_r@20_sum"] += c["delta_r@20"] * c["n"]
                merged[key]["delta_r@50_sum"] += c["delta_r@50"] * c["n"]
                wtl = c["W/T/L_r@50"].split("/")
                merged[key]["wins"] += int(wtl[0])
                merged[key]["ties"] += int(wtl[1])
                merged[key]["losses"] += int(wtl[2])
        # Flatten
        final_by_cat: dict[str, dict] = {}
        for cat, m in merged.items():
            if m["n"] == 0:
                continue
            final_by_cat[cat] = {
                "n": m["n"],
                "delta_r@20": round(m["delta_r@20_sum"] / m["n"], 4),
                "delta_r@50": round(m["delta_r@50_sum"] / m["n"], 4),
                "W/T/L_r@50": f"{m['wins']}/{m['ties']}/{m['losses']}",
            }
        top_gaining, top_losing = top_categories_delta(final_by_cat, K=50)

    # Qualitative samples: memoize_m2 on LoCoMo first, fall back to synthetic
    samples: list[dict] = []
    if "memoize_m2" in all_results and "meta_v2f" in all_results:
        for ds in ("locomo_30q", "synthetic_19q"):
            if ds not in all_results["memoize_m2"]:
                continue
            if ds not in all_results["meta_v2f"]:
                continue
            new = qualitative_samples(
                all_results["memoize_m2"][ds]["results"],
                all_results["meta_v2f"][ds]["results"],
                K=50,
                max_examples=5 - len(samples),
            )
            for s in new:
                s["dataset"] = ds
            samples.extend(new)
            if len(samples) >= 5:
                break

    # Probe top-1 / any-top-K gold hit rates (comparing memoize cues vs v2f cues)
    probe_stats: dict[str, dict] = {}
    for ds_name in ds_names:
        probe_stats[ds_name] = {}
        for arch_name, cue_key in (
            ("memoize_m2", "memoized_probe_outcomes"),
            ("memoize_m3", "memoized_probe_outcomes"),
            ("meta_v2f", "v2f_outcomes"),
        ):
            if ds_name not in all_results.get(arch_name, {}):
                continue
            rows = all_results[arch_name][ds_name]["results"]
            probe_stats[ds_name][arch_name] = probe_top1_gold_rate(
                rows, cue_key
            )

    # ---------------------------------------------------------------------
    # Persist raw + per-arch JSON
    # ---------------------------------------------------------------------

    raw: dict = {
        "archs": arch_names,
        "datasets": ds_names,
        "exemplar_bank_size": len(exemplars),
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
        "orthogonality_vs_v2f_at_K50": orthogonality,
        "nearest_exemplar_sim_stats": nearest_sim_stats,
        "top_gaining_categories_memoize_m2": top_gaining,
        "top_losing_categories_memoize_m2": top_losing,
        "qualitative_samples": samples,
        "probe_top1_gold_stats": probe_stats,
    }

    raw_path = RESULTS_DIR / "cue_memoization.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"memoize_{a}_{d}.json"
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

    # ---------------------------------------------------------------------
    # Markdown report
    # ---------------------------------------------------------------------

    md: list[str] = []
    md.append("# Cue Memoization (retrieve-and-reuse past cues)\n")
    md.append(
        "Motivation: few-shot had the LLM imitate exemplars (and fabricate "
        "corpus-specific content); MMR/spreading activation/anchor stayed in "
        "the v2f basin. Memoization drops the LLM adaptation step and reuses "
        "exemplar cues verbatim as retrieval probes — probes come from an "
        "independent distribution (past successful runs on different "
        "questions), so their retrievals may be orthogonal to v2f.\n"
    )

    md.append("## Exemplar bank\n")
    dataset_counter: dict[str, int] = defaultdict(int)
    conv_pairs: set[tuple[str, str]] = set()
    for ex in exemplars:
        dataset_counter[ex["dataset"]] += 1
        conv_pairs.add((ex["dataset"], ex["conversation_id"]))
    md.append(f"- Total exemplars: **{len(exemplars)}**")
    md.append(
        f"- Unique (dataset, conversation_id) pairs: **{len(conv_pairs)}**"
    )
    md.append("- By dataset:")
    for ds, n in sorted(dataset_counter.items()):
        md.append(f"  - `{ds}`: {n}")
    md.append("")

    md.append("## Fair-backfill recall\n")
    md.append(
        "| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | W/T/L@50 | llm/q |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|")
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
                f"{s['W/T/L_r@50']} | {s['avg_llm_calls']:.1f} |"
            )

    if nearest_sim_stats:
        md.append("\n## Query-to-nearest-exemplar cosine\n")
        md.append(
            "Mean cosine between the new query embedding and its top-1 "
            "selected exemplar. Low values → weak match → memoization cannot "
            "cover query intent.\n"
        )
        md.append("| Arch | Dataset | n | mean | min | max |")
        md.append("|---|---|---:|---:|---:|---:|")
        for a in nearest_sim_stats:
            for d in nearest_sim_stats[a]:
                s = nearest_sim_stats[a][d]
                md.append(
                    f"| {a} | {d} | {s['n']} | "
                    f"{s['mean']:.4f} | {s['min']:.4f} | {s['max']:.4f} |"
                )

    if orthogonality:
        md.append("\n## Orthogonality vs v2f (K=50)\n")
        md.append(
            "Fraction of gold turns the variant found that v2f did NOT find.\n"
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

    if probe_stats:
        md.append("\n## Per-cue gold hit rates (top-1 and any-top-K)\n")
        md.append(
            "How often does a reused cue's top-1 retrieval equal a gold turn? "
            "Compared across memoize vs v2f cue sources.\n"
        )
        md.append("| Dataset | Cue source | #cues | top1_hit | any_topK_hit |")
        md.append("|---|---|---:|---:|---:|")
        for d in probe_stats:
            for a in probe_stats[d]:
                s = probe_stats[d][a]
                md.append(
                    f"| {d} | {a} | {s['total_cues']} | "
                    f"{s['top1_hit_rate']:.3f} | {s['any_topk_hit_rate']:.3f} |"
                )

    if top_gaining or top_losing:
        md.append(
            "\n## Top gain/loss categories for memoize_m2 (combined datasets)\n"
        )
        md.append("Gaining:")
        for g in top_gaining:
            md.append(
                f"- {g['category']} (n={g['n']}): Δ@50={g['delta']:+.3f} "
                f"W/T/L={g['W/T/L']}"
            )
        md.append("Losing:")
        for l in top_losing:
            md.append(
                f"- {l['category']} (n={l['n']}): Δ@50={l['delta']:+.3f} "
                f"W/T/L={l['W/T/L']}"
            )

    if samples:
        md.append("\n## Qualitative samples (memoize_m2)\n")
        md.append(
            "Each row: new_q → nearest_exemplar_q (sim) → reused_cue → gold_found_turn\n"
        )
        for s in samples:
            novel_tag = (
                " **(novel vs v2f)**" if s.get("novel_vs_v2f") else ""
            )
            md.append(
                f"- **{s['dataset']}** `{s['new_category']}`{novel_tag}\n"
                f"  - new_q: {s['new_question']}\n"
                f"  - nearest_exemplar: {s['nearest_exemplar_question']} "
                f"(sim={s['nearest_exemplar_sim']:.3f}, "
                f"from `{s['nearest_exemplar_dataset']}`)\n"
                f"  - reused_cue: {s['reused_cue'][:180]}\n"
                f"  - gold_found_turn_id: {s['gold_found_turn_id']}"
            )

    # -----------------------------------------------------------------
    # Verdict (heuristic)
    # -----------------------------------------------------------------
    md.append("\n## Verdict\n")

    def best_arch(arch_name: str) -> tuple[float, float]:
        """Return (mean delta@50 across datasets, mean arch@50)."""
        if arch_name not in all_results:
            return (0.0, 0.0)
        ds_entries = all_results[arch_name]
        if not ds_entries:
            return (0.0, 0.0)
        d50 = sum(
            e["summary"]["delta_r@50"] for e in ds_entries.values()
        ) / len(ds_entries)
        a50 = sum(
            e["summary"]["arch_r@50"] for e in ds_entries.values()
        ) / len(ds_entries)
        return (d50, a50)

    m2_delta, m2_arch = best_arch("memoize_m2")
    m3_delta, m3_arch = best_arch("memoize_m3")
    mf_delta, mf_arch = best_arch("memoize_filtered")
    mpv_delta, mpv_arch = best_arch("memoize_plus_v2f")
    v2f_delta, v2f_arch = best_arch("meta_v2f")

    md.append(
        f"- memoize_m2 mean Δ@50 (vs cosine baseline): {m2_delta:+.3f}, "
        f"arch@50={m2_arch:.3f}"
    )
    md.append(
        f"- memoize_m3 mean Δ@50 (vs cosine baseline): {m3_delta:+.3f}, "
        f"arch@50={m3_arch:.3f}"
    )
    md.append(
        f"- memoize_filtered mean Δ@50 (vs cosine baseline): {mf_delta:+.3f}, "
        f"arch@50={mf_arch:.3f}"
    )
    md.append(
        f"- memoize_plus_v2f mean Δ@50 (vs cosine baseline): {mpv_delta:+.3f}, "
        f"arch@50={mpv_arch:.3f}"
    )
    md.append(
        f"- meta_v2f mean Δ@50 (vs cosine baseline): {v2f_delta:+.3f}, "
        f"arch@50={v2f_arch:.3f}"
    )

    mpv_gain_over_v2f = mpv_arch - v2f_arch
    m2_gain_over_v2f = m2_arch - v2f_arch

    if mpv_gain_over_v2f > 0.02:
        verdict = "SHIP / supplement"
        reason = (
            f"memoize_plus_v2f beats v2f by +{mpv_gain_over_v2f:.3f} "
            f"at K=50 — memoized cues add orthogonal retrievals."
        )
    elif m2_gain_over_v2f >= -0.005:
        verdict = "SHIP as zero-LLM alternative"
        reason = (
            f"memoize_m2 matches or beats v2f ({m2_gain_over_v2f:+.3f}) "
            "with zero new LLM calls — v2f's LLM call appears redundant."
        )
    elif mpv_gain_over_v2f > 0.0:
        verdict = "marginal — supplement only"
        reason = (
            f"memoize_plus_v2f narrowly beats v2f (+{mpv_gain_over_v2f:.3f}); "
            "weak signal, probes may still overlap v2f basin."
        )
    else:
        verdict = "ABANDON"
        reason = (
            "No memoize variant beats v2f. Memoized cues land in the same "
            "basin v2f already covers — independent probe source does not "
            "open new retrievals."
        )
    md.append(f"\n**Verdict: {verdict}.** {reason}\n")

    md_path = RESULTS_DIR / "cue_memoization.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final console table
    print("\n" + "=" * 100)
    print("CUE MEMOIZATION SUMMARY")
    print("=" * 100)
    header = (
        f"{'Arch':<22s} {'Dataset':<14s} "
        f"{'base@20':>8s} {'arch@20':>8s} {'d@20':>7s} "
        f"{'base@50':>8s} {'arch@50':>8s} {'d@50':>7s} "
        f"{'W/T/L@50':>10s} {'llm/q':>6s}"
    )
    print(header)
    print("-" * len(header))
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:<22s} {d:<14s} "
                f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
                f"{s['delta_r@20']:>+7.3f} "
                f"{s['baseline_r@50']:>8.3f} {s['arch_r@50']:>8.3f} "
                f"{s['delta_r@50']:>+7.3f} "
                f"{s['W/T/L_r@50']:>10s} {s['avg_llm_calls']:>6.1f}"
            )


if __name__ == "__main__":
    main()
