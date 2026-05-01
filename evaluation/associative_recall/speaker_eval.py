"""Fair-backfill eval for speaker-attributed retrieval.

Compares:
  - cosine_baseline (pure cosine, no v2f) -- implicit via fair_backfill baseline
  - meta_v2f (v2f reference, shared caches)
  - speaker_boost_0.02
  - speaker_boost_0.05
  - speaker_user_filter

on LoCoMo-30 and synthetic-19 at K=20 and K=50.

Usage:
    uv run python speaker_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from antipara_cue_gen import MetaV2fDedicated
from associative_recall import Segment
from dotenv import load_dotenv
from fair_backfill_eval import (
    BUDGETS,
    RESULTS_DIR,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)
from speaker_attributed import (
    _CONV_SPEAKERS_FILE,
)
from speaker_attributed import (
    ARCH_CLASSES as SPEAKER_ARCH_CLASSES,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


EVAL_DATASETS = ("locomo_30q", "synthetic_19q")
ARCH_CLASSES: dict[str, type] = {
    "meta_v2f": MetaV2fDedicated,
    **SPEAKER_ARCH_CLASSES,
}


def evaluate_question(arch, question: dict, is_speaker_arch: bool) -> dict:
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
    cosine_result = arch.store.search(query_emb, top_k=max_K, conversation_id=conv_id)
    cosine_segments = list(cosine_result.segments)

    md = result.metadata or {}
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
    }

    if is_speaker_arch:
        row.update(
            {
                "conv_user_name": md.get("conv_user_name"),
                "query_name_tokens": md.get("query_name_tokens", []),
                "query_mentions_conv_user": md.get("query_mentions_conv_user", False),
                "applied_speaker_transform": md.get("applied_speaker_transform", False),
                "n_user_in_v2f": md.get("n_user_in_v2f"),
                "appended_user_only_indices": md.get("appended_user_only_indices", []),
            }
        )

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
    # For speaker arches, compute gold-split by role @K=50 against the arch
    # output up to K=50.
    if is_speaker_arch:
        role_by_idx = {s.index: s.role for s in arch.store.segments}
        ids_in_arch_K50 = arch_ids_at_K[50]
        # Map turn_id back to role using conv_id filter.
        role_of_gold_found = {}
        for s in arch_segments[:50]:
            if s.turn_id in source_ids and s.turn_id in ids_in_arch_K50:
                role_of_gold_found[s.turn_id] = s.role
        row["gold_role_split_K50"] = role_of_gold_found
    return row


def run_one(
    arch_name: str,
    arch,
    dataset: str,
    questions: list[dict],
    is_speaker_arch: bool,
) -> tuple[list[dict], dict, dict]:
    print(f"\n{'=' * 70}")
    print(f"{arch_name} | {dataset} | {len(questions)} questions")
    print(f"{'=' * 70}")

    results: list[dict] = []
    for i, q in enumerate(questions):
        q_short = q["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(arch, q, is_speaker_arch)
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


def mention_coverage(speaker_results: list[dict]) -> dict:
    n = len(speaker_results)
    if n == 0:
        return {}
    n_mentions = sum(1 for r in speaker_results if r.get("query_mentions_conv_user"))
    mentioned_cats: dict[str, int] = defaultdict(int)
    for r in speaker_results:
        if r.get("query_mentions_conv_user"):
            mentioned_cats[r.get("category", "?")] += 1
    return {
        "n_queries": n,
        "n_query_mentions_conv_user": n_mentions,
        "frac_query_mentions": round(n_mentions / n, 4),
        "mentioned_by_category": dict(mentioned_cats),
    }


def mentioned_delta_table(
    arch_name: str,
    speaker_results: list[dict],
    v2f_results: list[dict],
) -> dict:
    """Per-K recall on the MENTIONED subset: speaker-arch vs meta_v2f."""
    v2f_by_key = {(r["conversation_id"], r["question_index"]): r for r in v2f_results}
    subset = [r for r in speaker_results if r.get("query_mentions_conv_user")]
    n = len(subset)
    out: dict[str, object] = {"n_mentioned": n}
    if n == 0:
        return out
    for K in BUDGETS:
        v2f_vals, arch_vals = [], []
        for r in subset:
            k = (r["conversation_id"], r["question_index"])
            v2f_r = v2f_by_key.get(k)
            if v2f_r is None:
                continue
            v2f_vals.append(v2f_r["fair_backfill"][f"arch_r@{K}"])
            arch_vals.append(r["fair_backfill"][f"arch_r@{K}"])
        if not v2f_vals:
            continue
        v_m = sum(v2f_vals) / len(v2f_vals)
        a_m = sum(arch_vals) / len(arch_vals)
        wins = sum(1 for v, a in zip(v2f_vals, arch_vals) if a > v + 0.001)
        losses = sum(1 for v, a in zip(v2f_vals, arch_vals) if v > a + 0.001)
        ties = len(v2f_vals) - wins - losses
        out[f"v2f_r@{K}"] = round(v_m, 4)
        out[f"arch_r@{K}"] = round(a_m, 4)
        out[f"delta_r@{K}"] = round(a_m - v_m, 4)
        out[f"W/T/L@{K}"] = f"{wins}/{ties}/{losses}"
    return out


def speaker_id_hit_rate(speakers: dict[str, str]) -> dict:
    n = len(speakers)
    known = sum(1 for v in speakers.values() if v and v != "UNKNOWN")
    return {
        "n_conversations": n,
        "n_identified": known,
        "hit_rate": round(known / n, 4) if n else 0.0,
        "speakers": speakers,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = defaultdict(dict)

    for ds_name in EVAL_DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        for arch_name, cls in ARCH_CLASSES.items():
            arch = cls(store)
            is_speaker = arch_name.startswith("speaker_")
            results, summary, by_cat = run_one(
                arch_name, arch, ds_name, questions, is_speaker
            )
            all_results[arch_name][ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
            }

    # --- Speaker IDs ---
    speakers_all: dict[str, str] = {}
    if _CONV_SPEAKERS_FILE.exists():
        try:
            with open(_CONV_SPEAKERS_FILE) as f:
                speakers_all = json.load(f).get("speakers", {}) or {}
        except (json.JSONDecodeError, OSError):
            speakers_all = {}
    sid_summary = speaker_id_hit_rate(speakers_all)

    # --- Coverage / mention diagnostics (from first speaker arch) ---
    primary_arch = "speaker_boost_0.02"
    coverage: dict[str, dict] = {}
    for ds_name in EVAL_DATASETS:
        if primary_arch in all_results and ds_name in all_results[primary_arch]:
            coverage[ds_name] = mention_coverage(
                all_results[primary_arch][ds_name]["results"]
            )

    # --- Mentioned-subset delta table vs meta_v2f ---
    mentioned_delta: dict[str, dict] = defaultdict(dict)
    for ds_name in EVAL_DATASETS:
        v2f_rs = all_results.get("meta_v2f", {}).get(ds_name, {}).get("results", [])
        for arch_name in SPEAKER_ARCH_CLASSES:
            sp_rs = all_results.get(arch_name, {}).get(ds_name, {}).get("results", [])
            if not sp_rs or not v2f_rs:
                continue
            mentioned_delta[arch_name][ds_name] = mentioned_delta_table(
                arch_name, sp_rs, v2f_rs
            )

    # --- Save raw JSON ---
    raw = {
        "archs": list(ARCH_CLASSES.keys()),
        "datasets": list(EVAL_DATASETS),
        "speaker_id_summary": sid_summary,
        "mention_coverage": coverage,
        "mentioned_subset_vs_meta_v2f": mentioned_delta,
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
    }
    raw_path = RESULTS_DIR / "speaker_attributed.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset details
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"speaker_{a}_{d}.json"
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

    # --- Markdown report ---
    md: list[str] = []
    md.append("# Speaker-attributed retrieval\n")
    md.append(
        "At ingest: identify which real name corresponds to the `user` role\n"
        "per conversation (1 LLM call per conv). At query time: regex-detect\n"
        "capitalized first-name tokens; if a token matches the conversation's\n"
        "user-name, apply a speaker-aware transform (boost or filter role=user\n"
        "turns). Rationale: dialog entities appear as vocatives ('Hey Caroline!')\n"
        "not subjects; turns ABOUT Caroline are often SPOKEN by Caroline, and\n"
        "her name isn't in the turn text.\n"
    )

    # Speaker IDs
    md.append("## Speaker identification\n")
    md.append(f"- Conversations scanned: {sid_summary['n_conversations']}")
    md.append(f"- Identified (non-UNKNOWN): {sid_summary['n_identified']}")
    md.append(f"- Hit rate: {sid_summary['hit_rate']:.2%}\n")
    md.append("| Conversation | user speaker |")
    md.append("|---|---|")
    for cid, name in sorted(sid_summary["speakers"].items()):
        md.append(f"| {cid} | {name} |")
    md.append("")

    # Query coverage
    md.append("## Query mention coverage\n")
    md.append(
        "Fraction of queries in each dataset that mention the conv-user's "
        "first name (case-insensitive, stop-word-filtered).\n"
    )
    md.append("| Dataset | n | mentions conv-user | frac |")
    md.append("|---|---:|---:|---:|")
    for ds_name, cov in coverage.items():
        md.append(
            f"| {ds_name} | {cov.get('n_queries')} | "
            f"{cov.get('n_query_mentions_conv_user')} | "
            f"{cov.get('frac_query_mentions'):.3f} |"
        )
    md.append("")
    md.append("### Mentions by category\n")
    for ds_name, cov in coverage.items():
        md.append(f"**{ds_name}**: {cov.get('mentioned_by_category', {})}\n")

    # Full-set recall
    md.append("## Fair-backfill recall (full question sets)\n")
    md.append(
        "| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in ARCH_CLASSES:
        for d in EVAL_DATASETS:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            md.append(
                f"| {a} | {d} | {s['baseline_r@20']:.3f} | "
                f"{s['arch_r@20']:.3f} | {s['delta_r@20']:+.3f} | "
                f"{s['baseline_r@50']:.3f} | {s['arch_r@50']:.3f} | "
                f"{s['delta_r@50']:+.3f} | {s['avg_llm_calls']:.1f} |"
            )
    md.append("")

    # Mentioned-subset recall
    md.append("## Mentioned-subset recall (speaker arch vs meta_v2f)\n")
    md.append(
        "On the SUBSET of queries that mention the conv-user. This is the\n"
        "subset where the mechanism can possibly activate.\n"
    )
    md.append(
        "| Arch | Dataset | n | v2f@20 | arch@20 | Δ@20 | v2f@50 | arch@50 | Δ@50 | W/T/L@50 |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arch_name, by_ds in mentioned_delta.items():
        for ds_name, row in by_ds.items():
            if not row or row.get("n_mentioned", 0) == 0:
                continue
            md.append(
                f"| {arch_name} | {ds_name} | {row['n_mentioned']} | "
                f"{row.get('v2f_r@20', 0):.3f} | {row.get('arch_r@20', 0):.3f} | "
                f"{row.get('delta_r@20', 0):+.3f} | "
                f"{row.get('v2f_r@50', 0):.3f} | {row.get('arch_r@50', 0):.3f} | "
                f"{row.get('delta_r@50', 0):+.3f} | "
                f"{row.get('W/T/L@50', 'NA')} |"
            )
    md.append("")

    # Per-category breakdown
    md.append("## Per-category (speaker_boost_0.02 vs v2f)\n")
    if "speaker_boost_0.02" in all_results:
        for d in EVAL_DATASETS:
            if d not in all_results["speaker_boost_0.02"]:
                continue
            by_cat = all_results["speaker_boost_0.02"][d]["category_breakdown"]
            md.append(f"### {d}\n")
            md.append("| category | n | Δ@20 | Δ@50 | W/T/L@50 |")
            md.append("|---|---:|---:|---:|---:|")
            for cat, c in by_cat.items():
                md.append(
                    f"| {cat} | {c['n']} | {c['delta_r@20']:+.3f} | "
                    f"{c['delta_r@50']:+.3f} | {c['W/T/L_r@50']} |"
                )
            md.append("")

    # Verdict
    md.append("## Verdict\n")
    # Decide:
    # - ship if speaker_boost variants beat v2f on LoCoMo (any K); also
    #   subset-wise ship-signal.
    # - narrow if only mentioned subset helps but full-set is flat.
    # - abandon if loses or mechanism barely fires.
    verdict_lines: list[str] = []
    coverage_locomo = coverage.get("locomo_30q", {}).get("frac_query_mentions", 0.0)
    coverage_synth = coverage.get("synthetic_19q", {}).get("frac_query_mentions", 0.0)
    v2f_lc = all_results.get("meta_v2f", {}).get("locomo_30q", {}).get("summary", {})
    verdict_lines.append(
        f"- Speaker-ID hit rate: {sid_summary['hit_rate']:.1%} "
        f"({sid_summary['n_identified']}/{sid_summary['n_conversations']})"
    )
    verdict_lines.append(
        f"- LoCoMo-30 conv-user mention coverage: {coverage_locomo:.1%}"
    )
    verdict_lines.append(
        f"- Synthetic-19 conv-user mention coverage: {coverage_synth:.1%}"
    )

    # Pick the best speaker variant on LoCoMo K=50 full-set.
    best_arch = None
    best_delta = -1e9
    for a in SPEAKER_ARCH_CLASSES:
        s = all_results.get(a, {}).get("locomo_30q", {}).get("summary", {})
        d = s.get("arch_r@50", 0) - v2f_lc.get("arch_r@50", 0)
        if d > best_delta:
            best_delta = d
            best_arch = a

    # Full-set verdict
    ship = False
    if best_arch is not None:
        verdict_lines.append(
            f"- Best variant on LoCoMo K=50 full-set: **{best_arch}** "
            f"(Δ vs v2f = {best_delta:+.3f})"
        )
        if best_delta > 0.005:
            verdict = (
                f"**SHIP {best_arch}**: beats v2f on LoCoMo K=50 full-set "
                f"(Δ={best_delta:+.3f}) at ~0 extra per-query LLM cost "
                f"(ingest-time speaker ID only)."
            )
            ship = True
        elif coverage_locomo < 0.15 and coverage_synth < 0.15:
            verdict = (
                f"**ABANDON (narrow applicability)**: only "
                f"{coverage_locomo:.0%}/{coverage_synth:.0%} of queries "
                f"mention the conv-user; the mechanism can activate on "
                f"<15% of the workload. Full-set lift {best_delta:+.3f} "
                f"is not enough to justify."
            )
        else:
            verdict = (
                f"**NARROW/ABANDON**: best variant "
                f"({best_arch}) is flat vs v2f (Δ={best_delta:+.3f}) on the "
                f"full set; even when the query mentions the conv-user, the "
                f"role-based filter does not add gold beyond what v2f+cosine "
                f"already surfaces. See the mentioned-subset table for "
                f"finer signal."
            )
    else:
        verdict = "no speaker arch completed"
    md.append("\n".join(verdict_lines))
    md.append("")
    md.append(verdict + "\n")

    md_path = RESULTS_DIR / "speaker_attributed.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final console table
    print("\n" + "=" * 100)
    print("SPEAKER-ATTRIBUTED SUMMARY")
    print("=" * 100)
    for a in ARCH_CLASSES:
        for d in EVAL_DATASETS:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:22s} {d:14s} "
                f"a@20={s['arch_r@20']:.3f} d@20={s['delta_r@20']:+.3f}  "
                f"a@50={s['arch_r@50']:.3f} d@50={s['delta_r@50']:+.3f}  "
                f"llm/q={s['avg_llm_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
