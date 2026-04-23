"""Fair-backfill eval of entity-mention variants vs v2f baseline.

Runs entity_regex_b0.1, entity_regex_b0.2, entity_llm_b0.1,
entity_regex_plus_v2f plus the meta_v2f reference on LoCoMo-30 and
synthetic-19 at K=20 and K=50 using the fair-backfill methodology.

Also reports:
  - Entity extraction stats per extractor.
  - Per-category deltas vs v2f.
  - Orthogonality: gold turns found by entity variants that v2f missed.
  - Sample (query, extracted entities, boosted turns) trios.

Usage:
    uv run python entity_eval.py
    uv run python entity_eval.py --archs entity_regex_b0.1
    uv run python entity_eval.py --datasets locomo_30q
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

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
from entity_mention import (
    ARCH_CLASSES as ENTITY_ARCH_CLASSES,
    EntityRegexB005,
    EntityRegexB01,
    EntityRegexB02,
    EntityLLMB01,
    EntityRegexPlusV2f,
    TurnEntityExtractor,
    _TURN_ENTITIES_FILE,
)
from antipara_cue_gen import MetaV2fDedicated

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")

ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    "entity_regex_b0.05": EntityRegexB005,
    "entity_regex_b0.1": EntityRegexB01,
    "entity_regex_b0.2": EntityRegexB02,
    "entity_llm_b0.1": EntityLLMB01,
    "entity_regex_plus_v2f": EntityRegexPlusV2f,
}


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

    md = result.metadata or {}
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
        "query_entities": md.get("query_entities", []),
        "num_boosted_turns": md.get("num_boosted_turns", 0),
        "boosted_turn_ids": md.get("boosted_turn_ids", []),
        "v2f_cues": md.get("v2f_cues", []),
        "v2f_outcomes": md.get("v2f_outcomes", []),
        "index_num_entities": md.get("index_num_entities"),
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
    row["arch_ids_at_K"] = {
        str(K): sorted(arch_ids_at_K[K]) for K in BUDGETS
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
        f"  avg retrieved={summary['avg_total_retrieved']:.0f} "
        f"llm={summary['avg_llm_calls']:.1f} "
        f"embed={summary['avg_embed_calls']:.1f}"
    )
    for cat, c in by_cat.items():
        print(
            f"    {cat:30s} (n={c['n']}): "
            f"r@20 d={c['delta_r@20']:+.3f} r@50 d={c['delta_r@50']:+.3f} "
            f"W/T/L@50={c['W/T/L_r@50']}"
        )

    return results, summary, by_cat


def compute_orthogonality(
    arch_rows: list[dict],
    v2f_rows: list[dict],
    K: int = 50,
) -> dict:
    """Fraction of gold found by arch that v2f missed."""
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    total_gold = 0
    novel_gold = 0
    per_q: list[dict] = []
    for r in arch_rows:
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
    arch_rows: list[dict],
    v2f_rows: list[dict],
    K: int = 50,
    max_examples: int = 5,
) -> list[dict]:
    """Find examples where entity-mention boosted turns surfaced gold.

    Prefer trios where the entity variant retrieved gold v2f missed.
    """
    v2f_by_key: dict[tuple, set[int]] = {}
    for r in v2f_rows:
        key = (r["conversation_id"], r["question_index"])
        v2f_by_key[key] = set(r["gold_found_at_K"].get(str(K), []))

    examples: list[dict] = []
    for pass_mode in ("novel", "any_with_match", "any"):
        if len(examples) >= max_examples:
            break
        for r in arch_rows:
            if len(examples) >= max_examples:
                break
            qents = r.get("query_entities", [])
            if not qents and pass_mode != "any":
                continue
            key = (r["conversation_id"], r["question_index"])
            gold = set(r["gold_found_at_K"].get(str(K), []))
            v2f_gold = v2f_by_key.get(key, set())
            novel = gold - v2f_gold
            if pass_mode == "novel" and not novel:
                continue
            if pass_mode == "any_with_match" and not gold:
                continue
            if pass_mode == "any" and not gold:
                continue
            if any(
                ex["conversation_id"] == r["conversation_id"]
                and ex["question_index"] == r["question_index"]
                for ex in examples
            ):
                continue
            examples.append(
                {
                    "conversation_id": r["conversation_id"],
                    "question_index": r["question_index"],
                    "question": r["question"],
                    "category": r["category"],
                    "query_entities": qents,
                    "num_boosted_turns": r.get("num_boosted_turns", 0),
                    "boosted_turn_ids": r.get("boosted_turn_ids", []),
                    "gold_found": sorted(gold),
                    "novel_vs_v2f": sorted(novel),
                    "pass": pass_mode,
                }
            )

    return examples


def top_categories_delta(by_cat: dict, K: int = 50) -> tuple[list, list]:
    rows = []
    for cat, c in by_cat.items():
        rows.append((cat, c[f"delta_r@{K}"], c[f"W/T/L_r@{K}"], c["n"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    gaining = [
        {"category": cat, "delta": d, "W/T/L": wtl, "n": n}
        for cat, d, wtl, n in rows
        if d > 0.001
    ][:3]
    losing = [
        {"category": cat, "delta": d, "W/T/L": wtl, "n": n}
        for cat, d, wtl, n in rows[::-1]
        if d < -0.001
    ][:3]
    return gaining, losing


def extraction_stats(
    dataset: str, extractor_kind: str, store, questions: list[dict]
) -> dict:
    """Summarize per-turn entity extraction stats (entities/turn, unique)."""
    ext = TurnEntityExtractor(extractor=extractor_kind)
    # Walk ALL segments in the store (even beyond the question's
    # conversations, since a single store usually holds 1 benchmark's convs).
    relevant_cids = sorted({s.conversation_id for s in store.segments})
    data = ext._store.get(extractor_kind, {})
    per_turn_counts: list[int] = []
    all_entities_norm: set[str] = set()
    total_turns = 0
    turns_with_entities = 0
    for cid in relevant_cids:
        cd = data.get(cid, {})
        for _tid_str, ents in cd.items():
            total_turns += 1
            n = len(ents)
            per_turn_counts.append(n)
            if n > 0:
                turns_with_entities += 1
            for e in ents:
                from entity_mention import _normalize_entity
                norm = _normalize_entity(e)
                if norm:
                    all_entities_norm.add(norm)

    mean_per_turn = (
        sum(per_turn_counts) / len(per_turn_counts)
        if per_turn_counts else 0.0
    )
    return {
        "dataset": dataset,
        "extractor": extractor_kind,
        "total_turns": total_turns,
        "turns_with_entities": turns_with_entities,
        "frac_turns_with_entities": round(
            turns_with_entities / total_turns, 3
        ) if total_turns else 0.0,
        "mean_entities_per_turn": round(mean_per_turn, 2),
        "unique_entities": len(all_entities_norm),
    }


def query_entity_stats(rows: list[dict]) -> dict:
    n_with_ents = sum(1 for r in rows if r.get("query_entities"))
    counts = [len(r.get("query_entities", [])) for r in rows]
    mean = sum(counts) / len(counts) if counts else 0
    mean_boosted = (
        sum(r.get("num_boosted_turns", 0) for r in rows) / len(rows)
        if rows else 0
    )
    return {
        "n_queries": len(rows),
        "n_with_query_entities": n_with_ents,
        "mean_entities_per_query": round(mean, 2),
        "mean_boosted_turns_per_query": round(mean_boosted, 2),
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

    # Preload stores + questions per dataset (so extraction is done once per
    # extractor kind via the arch constructor).
    ds_cache: dict[str, tuple] = {}
    for ds_name in ds_names:
        store, questions = load_dataset(ds_name)
        ds_cache[ds_name] = (store, questions)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

    for ds_name in ds_names:
        store, questions = ds_cache[ds_name]

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

    # Extraction stats per dataset per extractor (regex, llm).
    extraction_stats_out: dict[str, dict[str, dict]] = {}
    for ds_name, (store, _qs) in ds_cache.items():
        extraction_stats_out[ds_name] = {}
        for ext_kind in ("regex", "llm"):
            try:
                extraction_stats_out[ds_name][ext_kind] = extraction_stats(
                    ds_name, ext_kind, store, ds_cache[ds_name][1]
                )
            except Exception as e:
                extraction_stats_out[ds_name][ext_kind] = {"error": str(e)}

    # Query-time entity stats per arch (from row data)
    query_stats_out: dict[str, dict[str, dict]] = {}
    for a in arch_names:
        if a == "meta_v2f":
            continue
        query_stats_out[a] = {}
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            query_stats_out[a][d] = query_entity_stats(
                all_results[a][d]["results"]
            )

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
                for K in (20, 50):
                    orth = compute_orthogonality(rows, v2f_rows, K=K)
                    orthogonality[arch_name][f"{ds_name}_K{K}"] = {
                        "total_gold": orth["total_gold"],
                        "novel_vs_v2f": orth["novel_vs_v2f"],
                        "fraction_novel": orth["fraction_novel"],
                    }

    # Qualitative trios (prefer regex_plus_v2f on LoCoMo, fall back to others)
    trios: list[dict] = []
    primary_arch = None
    for cand in ("entity_regex_plus_v2f", "entity_regex_b0.1",
                 "entity_llm_b0.1", "entity_regex_b0.2"):
        if cand in arch_names:
            primary_arch = cand
            break
    if primary_arch is None and arch_names:
        primary_arch = next(
            (a for a in arch_names if a.startswith("entity_")), arch_names[0]
        )
    if (
        primary_arch in all_results
        and "meta_v2f" in all_results
        and "locomo_30q" in all_results.get(primary_arch, {})
        and "locomo_30q" in all_results.get("meta_v2f", {})
    ):
        trios = qualitative_trios(
            all_results[primary_arch]["locomo_30q"]["results"],
            all_results["meta_v2f"]["locomo_30q"]["results"],
            K=50,
            max_examples=5,
        )

    # Top categories per variant per dataset
    top_cats: dict[str, dict[str, dict]] = {}
    for a in arch_names:
        if a == "meta_v2f":
            continue
        top_cats[a] = {}
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            top_gaining, top_losing = top_categories_delta(
                all_results[a][d]["category_breakdown"], K=50
            )
            top_cats[a][d] = {"gaining": top_gaining, "losing": top_losing}

    # Save raw
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
        "extraction_stats": extraction_stats_out,
        "query_entity_stats": query_stats_out,
        "orthogonality_vs_v2f": orthogonality,
        "qualitative_trios": trios,
        "top_categories": top_cats,
        "primary_arch": primary_arch,
    }

    raw_path = RESULTS_DIR / "entity_mention_study.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset full results
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"entity_{a}_{d}.json"
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
    md: list[str] = []
    md.append(
        "# Entity-mention exact-match index — non-cosine retrieval signal\n"
    )
    md.append(
        "Motivation: v2f's cosine retrieval is fuzzy. Gold turns that share a "
        "specific entity with the query (names, IDs, numbers) can be missed "
        "even when obvious. An inverted index {entity -> turns} built at "
        "ingest, queried for query entities, boosts exact-match turns via "
        "final_score = cosine + beta * I(turn mentions query entity). This "
        "is NOT BM25 — only named entities are indexed.\n"
    )

    md.append("## Entity extraction stats\n")
    md.append("| Dataset | Extractor | turns | turns w/ent | % | ent/turn | unique ents |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for d in ds_names:
        for kind in ("regex", "llm"):
            s = extraction_stats_out.get(d, {}).get(kind, {})
            if not s or "error" in s:
                continue
            md.append(
                f"| {d} | {kind} | {s.get('total_turns', 0)} | "
                f"{s.get('turns_with_entities', 0)} | "
                f"{s.get('frac_turns_with_entities', 0) * 100:.0f}% | "
                f"{s.get('mean_entities_per_turn', 0):.2f} | "
                f"{s.get('unique_entities', 0)} |"
            )

    md.append("\n## Query-entity & boost coverage\n")
    md.append("| Arch | Dataset | queries | w/ents | ent/q | boosted turns/q |")
    md.append("|---|---|---:|---:|---:|---:|")
    for a in arch_names:
        if a == "meta_v2f":
            continue
        for d in ds_names:
            s = query_stats_out.get(a, {}).get(d)
            if not s:
                continue
            md.append(
                f"| {a} | {d} | {s['n_queries']} | "
                f"{s['n_with_query_entities']} | "
                f"{s['mean_entities_per_query']:.2f} | "
                f"{s['mean_boosted_turns_per_query']:.1f} |"
            )

    md.append("\n## Fair-backfill recall\n")
    md.append(
        "| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q | embed/q |"
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
                f"{s['avg_llm_calls']:.1f} | "
                f"{s['avg_embed_calls']:.1f} |"
            )

    if orthogonality:
        md.append("\n## Orthogonality vs v2f\n")
        md.append(
            "Fraction of gold turns found by the variant that v2f did NOT "
            "find.\n"
        )
        md.append("| Arch | Dataset/K | gold_found | novel_vs_v2f | frac_novel |")
        md.append("|---|---|---:|---:|---:|")
        for a in orthogonality:
            for key, o in orthogonality[a].items():
                md.append(
                    f"| {a} | {key} | {o['total_gold']} | "
                    f"{o['novel_vs_v2f']} | {o['fraction_novel']:.3f} |"
                )

    if trios:
        md.append(
            f"\n## Qualitative trios ({primary_arch}, LoCoMo, K=50)\n"
        )
        md.append(
            "Each row: query, extracted query entities, boosted turn ids, "
            "gold found."
        )
        for ex in trios:
            novel_tag = (
                " **(novel vs v2f)**" if ex.get("novel_vs_v2f") else ""
            )
            md.append(f"\n- **Q:** {ex['question']}{novel_tag}")
            md.append(
                f"  - Query entities: {ex.get('query_entities', [])}"
            )
            md.append(
                f"  - Num boosted turns: {ex.get('num_boosted_turns', 0)}"
            )
            md.append(
                f"  - Boosted turn ids (sample): "
                f"{ex.get('boosted_turn_ids', [])[:20]}"
            )
            md.append(f"  - Gold found: {ex.get('gold_found', [])}")
            if ex.get("novel_vs_v2f"):
                md.append(
                    f"  - Novel vs v2f: {ex['novel_vs_v2f']}"
                )

    md.append("\n## Top categories by Δr@50\n")
    for a in top_cats:
        for d in top_cats[a]:
            tg = top_cats[a][d]["gaining"]
            tl = top_cats[a][d]["losing"]
            if not tg and not tl:
                continue
            md.append(f"\n### {a} on {d}\n")
            md.append("Gaining:")
            if not tg:
                md.append("  - (none with Δ > 0.001)")
            for g in tg:
                md.append(
                    f"  - {g['category']} (n={g['n']}): Δ={g['delta']:+.3f} "
                    f"W/T/L={g['W/T/L']}"
                )
            md.append("Losing:")
            if not tl:
                md.append("  - (none with Δ < -0.001)")
            for l in tl:
                md.append(
                    f"  - {l['category']} (n={l['n']}): Δ={l['delta']:+.3f} "
                    f"W/T/L={l['W/T/L']}"
                )

    # Verdict
    md.append("\n## Verdict\n")
    verdict = "(see numbers above)"
    if (
        "meta_v2f" in all_results
        and "locomo_30q" in all_results["meta_v2f"]
    ):
        v2f50_lc = all_results["meta_v2f"]["locomo_30q"]["summary"][
            "arch_r@50"
        ]
        v2f50_syn = None
        if "synthetic_19q" in all_results["meta_v2f"]:
            v2f50_syn = all_results["meta_v2f"]["synthetic_19q"]["summary"][
                "arch_r@50"
            ]

        variant_scores: list[tuple[str, float, float | None]] = []
        for a in arch_names:
            if a == "meta_v2f":
                continue
            if "locomo_30q" not in all_results.get(a, {}):
                continue
            lc = all_results[a]["locomo_30q"]["summary"]["arch_r@50"]
            syn = None
            if "synthetic_19q" in all_results.get(a, {}):
                syn = all_results[a]["synthetic_19q"]["summary"]["arch_r@50"]
            variant_scores.append((a, lc, syn))
        best = max(variant_scores, key=lambda t: t[1]) if variant_scores else None
        if best and best[1] > v2f50_lc + 0.005:
            verdict = (
                f"**SHIP**: {best[0]} beats v2f on LoCoMo-30 @K=50 "
                f"({best[1]:.3f} vs {v2f50_lc:.3f})."
            )
            if (
                best[2] is not None and v2f50_syn is not None
                and best[2] > v2f50_syn + 0.005
            ):
                verdict += (
                    f" Also beats v2f on synthetic @K=50 "
                    f"({best[2]:.3f} vs {v2f50_syn:.3f})."
                )
            elif best[2] is not None and v2f50_syn is not None:
                verdict += (
                    f" Synthetic @K=50: {best[2]:.3f} vs v2f {v2f50_syn:.3f}."
                )
        else:
            details = ", ".join(
                f"{n}=LC{lc:.3f}" for n, lc, _ in variant_scores
            )
            verdict = (
                f"**ABANDON**: no entity variant beats v2f on LoCoMo-30 "
                f"@K=50 (v2f={v2f50_lc:.3f}; {details})."
            )
    md.append(verdict + "\n")
    md.append(
        "\n## Outputs\n"
        "- `results/entity_mention_study.md` — this report\n"
        "- `results/entity_mention_study.json` — raw metrics + stats\n"
        f"- `results/turn_entities.json` — per-turn extracted entities\n"
        "- `results/entity_<arch>_<dataset>.json` — per-question detail\n"
    )

    md_path = RESULTS_DIR / "entity_mention_study.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final table
    print("\n" + "=" * 100)
    print("ENTITY MENTION SUMMARY")
    print("=" * 100)
    for a in arch_names:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:26s} {d:14s} "
                f"b@20={s['baseline_r@20']:.3f} a@20={s['arch_r@20']:.3f} "
                f"d@20={s['delta_r@20']:+.3f}  "
                f"b@50={s['baseline_r@50']:.3f} a@50={s['arch_r@50']:.3f} "
                f"d@50={s['delta_r@50']:+.3f}  "
                f"llm={s['avg_llm_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
