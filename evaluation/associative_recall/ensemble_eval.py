"""Fair-budget evaluation of specialist ensembles across 4 merging strategies.

For each dataset (locomo_30q, synthetic_19q, puzzle_16q, advanced_23q) and
each (ensemble composition, merging strategy, K) triple, compute mean recall
of retrieved turn_ids against source_chat_ids. Compare to v2f-alone (also at
fair-backfilled K) as the baseline.

Usage:
    uv run python ensemble_eval.py

Outputs (written to results/):
  - ensemble_study.json  — raw numbers
  - ensemble_study.md    — markdown summary
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from associative_recall import Segment, SegmentStore
from ensemble_retrieval import (
    ENSEMBLE_COMPOSITIONS,
    MERGING_STRATEGIES,
    SPECIALISTS,
    build_specialist,
    ensemble_at_k,
    fair_backfill,
    merge_max_cosine,
    run_specialists_cached,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS = (20, 50)

DATASETS: dict[str, dict] = {
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
    "puzzle_16q": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "filter": None,
        "max_questions": None,
    },
    "advanced_23q": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "filter": None,
        "max_questions": None,
    },
}

# Per-specialist LLM-call cost multipliers (relative to one v2f call),
# taken from specialist_complementarity.md.
SPECIALIST_COST: dict[str, float] = {
    "v2f": 1.0,
    "v2f_plus_types": 2.0,
    "type_enumerated": 1.0,
    "chain_with_scratchpad": 5.0,
    "v2f_style_explicit": 1.0,
}


def load_questions(ds_name: str) -> list[dict]:
    cfg = DATASETS[ds_name]
    with open(DATA_DIR / cfg["questions"]) as f:
        qs = json.load(f)
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    if cfg["max_questions"]:
        qs = qs[: cfg["max_questions"]]
    return qs


def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def evaluate_dataset(
    ds_name: str,
) -> tuple[list[dict], dict]:
    """Return (per_question_rows, meta).

    Each row records, per question:
      - per-specialist recall @ each K  (fair-backfilled)
      - per (ensemble, strategy, K) recall (fair-backfilled)
      - per-specialist LLM calls
      - category
    """
    cfg = DATASETS[ds_name]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    questions = load_questions(ds_name)
    specialists = {name: build_specialist(name, store) for name in SPECIALISTS}

    rows: list[dict] = []
    for qi, q in enumerate(questions):
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])
        cat = q.get("category", "unknown")

        # Compute shared resources
        query_emb = specialists["v2f"].embed_text(q_text)
        cosine_res = store.search(
            query_emb, top_k=max(BUDGETS), conversation_id=conv_id
        )
        cosine_segments = list(cosine_res.segments)

        # Run all 5 specialists, cache-only
        outputs = run_specialists_cached(
            specialists, store, q_text, conv_id, query_emb
        )

        row: dict = {
            "dataset": ds_name,
            "conversation_id": conv_id,
            "question_index": q.get("question_index", qi),
            "category": cat,
            "num_source_turns": len(source_ids),
            "per_specialist_recall": {},
            "per_specialist_llm_calls": {
                n: outputs[n].llm_calls for n in SPECIALISTS
            },
            "ensemble_recall": {},
        }
        if not source_ids:
            rows.append(row)
            continue

        # Per-specialist recall (fair-backfilled) at each K
        for name, so in outputs.items():
            # Fair-backfill by truncating spec's own segments to K then cosine
            # topup — mirrors fair_backfill_eval.py.
            seen: set[int] = set()
            arch_list: list[Segment] = []
            for s in so.segments:
                if s.index in seen:
                    continue
                seen.add(s.index)
                arch_list.append(s)
            row["per_specialist_recall"][name] = {}
            for K in BUDGETS:
                trunc = list(arch_list[:K])
                arch_indices = {s.index for s in trunc}
                if len(trunc) < K:
                    for s in cosine_segments:
                        if s.index in arch_indices:
                            continue
                        trunc.append(s)
                        arch_indices.add(s.index)
                        if len(trunc) >= K:
                            break
                trunc = trunc[:K]
                tids = {s.turn_id for s in trunc}
                row["per_specialist_recall"][name][f"r@{K}"] = round(
                    compute_recall(tids, source_ids), 4
                )

        # Per (ensemble, strategy, K) recall
        for comp_name, comp_specs in ENSEMBLE_COMPOSITIONS.items():
            for strat in MERGING_STRATEGIES:
                for K in BUDGETS:
                    segs = ensemble_at_k(
                        outputs,
                        comp_specs,
                        strat,
                        cosine_segments,
                        K,
                    )
                    tids = {s.turn_id for s in segs}
                    row["ensemble_recall"].setdefault(comp_name, {}).setdefault(
                        strat, {}
                    )[f"r@{K}"] = round(compute_recall(tids, source_ids), 4)

        rows.append(row)
        if (qi + 1) % 10 == 0:
            for arch in specialists.values():
                try:
                    arch.save_caches()
                except Exception:
                    pass
            print(f"  [{ds_name}] processed {qi + 1}/{len(questions)}",
                  flush=True)

    for arch in specialists.values():
        try:
            arch.save_caches()
        except Exception:
            pass

    meta = {
        "n_questions": len(rows),
        "n_with_gold": sum(1 for r in rows if r["num_source_turns"] > 0),
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def mean(vs):
    return (sum(vs) / len(vs)) if vs else 0.0


def aggregate(all_rows: dict[str, list[dict]]) -> dict:
    """Aggregate into:
      per_ds[ds][ens][strat][K] = mean recall (over questions with gold)
      per_ds[ds]['v2f_alone'][K] = mean recall
      all_ds[ens][strat][K]
      all_ds['v2f_alone'][K]
      per_cat[ds][cat][ens][strat][K] ... only for K=50 winners + v2f
      llm_cost[ens] = expected LLM calls per question
    """
    out: dict = {"per_ds": {}, "all_ds": {}, "per_category": {},
                 "llm_cost": {}, "ds_counts": {}}

    # LLM cost: use specialist_complementarity's per-call multipliers.
    for ens, specs in ENSEMBLE_COMPOSITIONS.items():
        out["llm_cost"][ens] = round(
            sum(SPECIALIST_COST[s] for s in specs), 2
        )
    out["llm_cost"]["v2f_alone"] = 1.0

    # Per-dataset aggregation
    all_q_by_ens: dict = defaultdict(lambda: defaultdict(
        lambda: defaultdict(list)))
    all_q_v2f: dict = defaultdict(list)
    # Per-category aggregation (across all datasets)
    all_cat_q: dict = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))))
    all_cat_v2f: dict = defaultdict(lambda: defaultdict(list))
    # Per-category counts
    all_cat_n: dict = defaultdict(int)

    for ds, rows in all_rows.items():
        n_gold = 0
        ds_ens = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        ds_v2f = defaultdict(list)
        for r in rows:
            if r["num_source_turns"] == 0:
                continue
            n_gold += 1
            cat = r["category"]
            all_cat_n[cat] += 1
            # v2f-alone recall (from per_specialist_recall)
            for K in BUDGETS:
                v2f_r = r["per_specialist_recall"]["v2f"][f"r@{K}"]
                ds_v2f[K].append(v2f_r)
                all_q_v2f[K].append(v2f_r)
                all_cat_v2f[cat][K].append(v2f_r)
            # Ensembles
            for ens, d in r["ensemble_recall"].items():
                for strat, dk in d.items():
                    for K in BUDGETS:
                        ds_ens[ens][strat][K].append(dk[f"r@{K}"])
                        all_q_by_ens[ens][strat][K].append(dk[f"r@{K}"])
                        all_cat_q[cat][ens][strat][K].append(dk[f"r@{K}"])

        out["per_ds"][ds] = {
            "n_with_gold": n_gold,
            "v2f_alone": {
                f"r@{K}": round(mean(ds_v2f[K]), 4) for K in BUDGETS
            },
            "ensembles": {
                ens: {
                    strat: {
                        f"r@{K}": round(mean(ds_ens[ens][strat][K]), 4)
                        for K in BUDGETS
                    }
                    for strat in MERGING_STRATEGIES
                }
                for ens in ENSEMBLE_COMPOSITIONS
            },
        }
        out["ds_counts"][ds] = n_gold

    out["all_ds"] = {
        "v2f_alone": {f"r@{K}": round(mean(all_q_v2f[K]), 4) for K in BUDGETS},
        "ensembles": {
            ens: {
                strat: {
                    f"r@{K}": round(mean(all_q_by_ens[ens][strat][K]), 4)
                    for K in BUDGETS
                }
                for strat in MERGING_STRATEGIES
            }
            for ens in ENSEMBLE_COMPOSITIONS
        },
    }

    out["per_category"] = {
        cat: {
            "n": all_cat_n[cat],
            "v2f_alone": {
                f"r@{K}": round(mean(all_cat_v2f[cat][K]), 4) for K in BUDGETS
            },
            "ensembles": {
                ens: {
                    strat: {
                        f"r@{K}": round(mean(
                            all_cat_q[cat][ens][strat][K]
                        ), 4)
                        for K in BUDGETS
                    }
                    for strat in MERGING_STRATEGIES
                }
                for ens in ENSEMBLE_COMPOSITIONS
            },
        }
        for cat in all_cat_n
    }

    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def render_markdown(aggr: dict, all_rows: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    lines.append("# Ensemble Retrieval Study\n")
    lines.append(
        "Does a specialist ensemble beat v2f-alone at **fair K budget** — "
        "where the ensemble's merged top-K is truncated to the same number of "
        "segments as v2f's top-K? If so, which merging strategy and ensemble "
        "composition ships?\n"
    )
    lines.append(
        f"**Compositions**: {list(ENSEMBLE_COMPOSITIONS.keys())}\n"
    )
    lines.append(f"**Merging strategies**: {list(MERGING_STRATEGIES)}\n")
    lines.append(f"**Budgets**: K={BUDGETS}\n")
    lines.append(
        "Each specialist is re-run cache-only; if a prompt is a cache miss "
        "the specialist emits `ACTION: DONE` (no new cues). Ensemble merging "
        "is followed by fair-backfill from cosine top-K of the raw query.\n"
    )

    # Cost table
    lines.append("\n## LLM-call cost per question (relative)\n")
    lines.append("| Setup | LLM calls / question |")
    lines.append("|---|---|")
    lines.append(f"| v2f-alone | {aggr['llm_cost']['v2f_alone']} |")
    for ens, cost in aggr["llm_cost"].items():
        if ens == "v2f_alone":
            continue
        lines.append(f"| {ens} ({'+'.join(ENSEMBLE_COMPOSITIONS[ens])}) "
                     f"| {cost} |")

    # Fair-budget verdict: LoCoMo @ K=50 headline table
    lines.append("\n## Headline: LoCoMo @ K=50 (primary fair-budget test)\n")
    loco = aggr["per_ds"]["locomo_30q"]
    v2f_loco_50 = loco["v2f_alone"]["r@50"]
    lines.append(f"- v2f-alone @K=50 on LoCoMo: **{v2f_loco_50:.4f}**\n")
    lines.append("| Ensemble | Strategy | r@50 | Δ vs v2f | LLM×v2f |")
    lines.append("|---|---|---|---|---|")
    all_rows_sorted = []
    for ens in ENSEMBLE_COMPOSITIONS:
        for strat in MERGING_STRATEGIES:
            r50 = loco["ensembles"][ens][strat]["r@50"]
            all_rows_sorted.append(
                (ens, strat, r50, r50 - v2f_loco_50, aggr["llm_cost"][ens])
            )
    all_rows_sorted.sort(key=lambda x: -x[2])
    for ens, strat, r50, delta, cost in all_rows_sorted:
        lines.append(
            f"| {ens} | {strat} | {r50:.4f} | {delta:+.4f} | {cost}× |"
        )

    # Per-dataset × K tables
    for ds in DATASETS:
        if ds not in aggr["per_ds"]:
            continue
        lines.append(f"\n## Dataset: {ds}\n")
        dsblk = aggr["per_ds"][ds]
        lines.append(f"n questions with gold = {dsblk['n_with_gold']}\n")
        for K in BUDGETS:
            lines.append(f"\n### K={K}\n")
            v2f_r = dsblk["v2f_alone"][f"r@{K}"]
            lines.append(f"v2f-alone recall = **{v2f_r:.4f}**\n")
            lines.append("| Ensemble | max_cosine | sum_cosine | rrf | "
                         "round_robin |")
            lines.append("|---|---|---|---|---|")
            for ens in ENSEMBLE_COMPOSITIONS:
                row = dsblk["ensembles"][ens]

                def cell(strat):
                    r = row[strat][f"r@{K}"]
                    d = r - v2f_r
                    return f"{r:.4f} ({d:+.4f})"
                lines.append(
                    f"| {ens} | {cell('max_cosine')} | {cell('sum_cosine')} "
                    f"| {cell('rrf')} | {cell('round_robin')} |"
                )

    # All-datasets aggregate
    lines.append("\n## All datasets aggregated\n")
    allblk = aggr["all_ds"]
    for K in BUDGETS:
        lines.append(f"\n### K={K}\n")
        v2f_r = allblk["v2f_alone"][f"r@{K}"]
        lines.append(f"v2f-alone recall = **{v2f_r:.4f}**\n")
        lines.append("| Ensemble | max_cosine | sum_cosine | rrf | "
                     "round_robin |")
        lines.append("|---|---|---|---|---|")
        for ens in ENSEMBLE_COMPOSITIONS:
            row = allblk["ensembles"][ens]

            def cell(strat):
                r = row[strat][f"r@{K}"]
                d = r - v2f_r
                return f"{r:.4f} ({d:+.4f})"
            lines.append(
                f"| {ens} | {cell('max_cosine')} | {cell('sum_cosine')} "
                f"| {cell('rrf')} | {cell('round_robin')} |"
            )

    # Per-category table — pick best ensemble×strategy at K=50 by all_ds gain
    lines.append("\n## Per-category gains at K=50 (best ensemble×strategy)\n")
    # Identify winner composition/strategy by all_ds K=50 delta vs v2f
    all_v2f50 = allblk["v2f_alone"]["r@50"]
    best = None
    best_delta = -1.0
    for ens in ENSEMBLE_COMPOSITIONS:
        for strat in MERGING_STRATEGIES:
            r = allblk["ensembles"][ens][strat]["r@50"]
            if r - all_v2f50 > best_delta:
                best_delta = r - all_v2f50
                best = (ens, strat, r)
    lines.append(
        f"Winner: `{best[0]} × {best[1]}` @ r@50={best[2]:.4f} "
        f"(Δ = {best_delta:+.4f}).\n"
    )
    lines.append("| Category | n | v2f r@50 | ens r@50 | Δ |")
    lines.append("|---|---|---|---|---|")
    cat_rows = []
    for cat, info in aggr["per_category"].items():
        v = info["v2f_alone"]["r@50"]
        e = info["ensembles"][best[0]][best[1]]["r@50"]
        cat_rows.append((cat, info["n"], v, e, e - v))
    cat_rows.sort(key=lambda r: -r[4])
    for cat, n, v, e, d in cat_rows:
        lines.append(f"| {cat} | {n} | {v:.4f} | {e:.4f} | {d:+.4f} |")

    # Decision
    lines.append("\n## Verdict\n")
    # LoCoMo K=50 leader
    best_loco = None
    best_loco_delta = -1.0
    for ens in ENSEMBLE_COMPOSITIONS:
        for strat in MERGING_STRATEGIES:
            r = aggr["per_ds"]["locomo_30q"]["ensembles"][ens][strat]["r@50"]
            if r - v2f_loco_50 > best_loco_delta:
                best_loco_delta = r - v2f_loco_50
                best_loco = (ens, strat, r)
    lines.append(
        f"- **LoCoMo K=50**: best ensemble×strategy = "
        f"`{best_loco[0]} × {best_loco[1]}` → r@50={best_loco[2]:.4f}, "
        f"Δ={best_loco_delta:+.4f} pp vs v2f-alone "
        f"(cost {aggr['llm_cost'][best_loco[0]]}×).\n"
    )
    # Union-ceiling vs realized
    lines.append(
        "- Union-5 set-theoretic ceiling on LoCoMo K=50 = 0.9492 "
        "(from specialist_complementarity). This is what we'd get if merge "
        "had no K-budget truncation penalty.\n"
    )

    # Also find the cheapest ensemble×strategy that beats v2f by ≥3pp on
    # LoCoMo K=50, since the spec explicitly flags this as ship-worthy.
    ship_candidates = []
    for ens in ENSEMBLE_COMPOSITIONS:
        for strat in MERGING_STRATEGIES:
            r = aggr["per_ds"]["locomo_30q"]["ensembles"][ens][strat]["r@50"]
            delta = r - v2f_loco_50
            if delta >= 0.03:
                ship_candidates.append(
                    (aggr["llm_cost"][ens], ens, strat, r, delta)
                )
    ship_candidates.sort()  # lowest cost first

    if ship_candidates:
        cost, ens, strat, r, delta = ship_candidates[0]
        if cost <= 2.0 and delta >= 0.03:
            ship = (
                f"SHIP — `{ens} × {strat}` gives Δ={delta:+.4f} pp at "
                f"{cost}× cost (meets the ≥3pp @ <3× ship-rule)"
            )
        elif cost <= 3.0:
            ship = (
                f"NARROW — `{ens} × {strat}` gives Δ={delta:+.4f} pp at "
                f"{cost}× cost (pp/call = {delta * 100 / cost:.2f})"
            )
        else:
            ship = (
                f"MARGINAL — best cost-meeting-bar is `{ens} × {strat}` at "
                f"{cost}× cost for Δ={delta:+.4f} pp; cost-per-pp heavy"
            )
    elif best_loco_delta >= 0.01:
        ship = "MARGINAL — ensemble beats v2f but <3pp"
    else:
        ship = "ABANDON — complementarity unrealizable in fair-budget"
    lines.append(f"- **Recommendation: {ship}**\n")

    if ship_candidates:
        lines.append(
            "\nAll LoCoMo K=50 ensemble×strategy settings meeting the "
            "≥3pp ship threshold (sorted by cost):\n"
        )
        lines.append("| cost | ensemble | strategy | r@50 | Δ | pp/call |")
        lines.append("|---|---|---|---|---|---|")
        for cost, ens, strat, r, delta in ship_candidates:
            lines.append(
                f"| {cost}× | {ens} | {strat} | {r:.4f} | {delta:+.4f} | "
                f"{delta * 100 / cost:.2f} |"
            )

    # Strategy comparison across compositions at K=50 (all_ds)
    lines.append("\n### Strategy comparison (all datasets, K=50)\n")
    lines.append("Mean r@50 across all compositions by strategy:\n")
    strat_means = {}
    for strat in MERGING_STRATEGIES:
        vals = []
        for ens in ENSEMBLE_COMPOSITIONS:
            vals.append(allblk["ensembles"][ens][strat]["r@50"])
        strat_means[strat] = mean(vals)
    ordered = sorted(strat_means.items(), key=lambda kv: -kv[1])
    lines.append("| Strategy | mean r@50 across ensembles |")
    lines.append("|---|---|")
    for s, v in ordered:
        lines.append(f"| {s} | {v:.4f} |")

    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    all_rows: dict[str, list[dict]] = {}
    dataset_meta: dict = {}

    for ds_name in DATASETS:
        print(f"\n[{ds_name}] running ...", flush=True)
        rows, meta = evaluate_dataset(ds_name)
        all_rows[ds_name] = rows
        dataset_meta[ds_name] = meta
        print(f"  n={meta['n_questions']} with_gold={meta['n_with_gold']}",
              flush=True)

    print("\nAggregating ...", flush=True)
    aggr = aggregate(all_rows)

    # Save JSON
    json_path = RESULTS_DIR / "ensemble_study.json"
    payload = {
        "dataset_meta": dataset_meta,
        "aggregate": aggr,
        "per_question_sample": {
            ds: rows[:3] for ds, rows in all_rows.items()
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved {json_path}", flush=True)

    md = render_markdown(aggr, all_rows)
    md_path = RESULTS_DIR / "ensemble_study.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved {md_path}", flush=True)

    # Headline print
    print("\n" + "=" * 70)
    print("ENSEMBLE STUDY SUMMARY")
    print("=" * 70)
    loco = aggr["per_ds"]["locomo_30q"]
    v2f_50 = loco["v2f_alone"]["r@50"]
    print(f"LoCoMo v2f-alone @K=50: {v2f_50:.4f}")
    for ens in ENSEMBLE_COMPOSITIONS:
        best_strat = None
        best_r = -1.0
        for strat in MERGING_STRATEGIES:
            r = loco["ensembles"][ens][strat]["r@50"]
            if r > best_r:
                best_r = r
                best_strat = strat
        print(f"  {ens:24s} best @K=50 via {best_strat:12s}: {best_r:.4f} "
              f"(Δ={best_r - v2f_50:+.4f}, cost={aggr['llm_cost'][ens]}×)")


if __name__ == "__main__":
    main()
