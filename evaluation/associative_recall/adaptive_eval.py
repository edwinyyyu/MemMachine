"""Evaluation harness for novelty-gated adaptive ensemble retrieval.

For each dataset (locomo_30q, synthetic_19q) and each (variant, K) pair,
compute mean recall of retrieved turn_ids against source_chat_ids.

Variants
--------
- `adaptive_tau_0.1`, `adaptive_tau_0.2`, `adaptive_tau_0.3` — gated ensembles
  that stop calling further specialists when incoming novelty <= tau.
- `v2f` — 1 specialist call.
- `ens_2_v2f_typeenum` — v2f + type_enumerated (sum_cosine merge).
- `ens_5` — all 5 specialists (sum_cosine merge).

All specialist outputs are retrieved cache-only (no new LLM calls) via the
existing `run_specialists_cached` helper from `ensemble_retrieval`.

Usage
-----
    uv run python adaptive_eval.py

Outputs
-------
- results/adaptive_ensemble.json  — raw per-question + aggregated numbers
- results/adaptive_ensemble.md    — recall table + cost/verdict summary
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from adaptive_ensemble import (
    ADAPTIVE_ORDER,
    SPECIALIST_COST,
    adaptive_ensemble,
)
from associative_recall import Segment, SegmentStore
from dotenv import load_dotenv
from ensemble_retrieval import (
    SpecialistOutput,
    build_specialist,
    fair_backfill,
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
}

TAU_VALUES = (0.1, 0.2, 0.3)


def _variant_names() -> list[str]:
    names = ["v2f", "ens_2_v2f_typeenum", "ens_5"]
    names += [f"adaptive_tau_{tau:.1f}" for tau in TAU_VALUES]
    return names


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


# ---------------------------------------------------------------------------
# Fixed-cost controls — reuse the merging logic from ensemble_retrieval.
# ---------------------------------------------------------------------------
def _v2f_alone_fairbackfill(
    outputs: dict[str, SpecialistOutput],
    cosine_segments: list[Segment],
    K: int,
) -> list[Segment]:
    # Truncate v2f to K, backfill from cosine.
    v2f = outputs["v2f"]
    seen: set[int] = set()
    trunc: list[Segment] = []
    for s in v2f.segments:
        if s.index in seen:
            continue
        seen.add(s.index)
        trunc.append(s)
        if len(trunc) >= K:
            break
    if len(trunc) < K:
        for s in cosine_segments:
            if s.index in seen:
                continue
            trunc.append(s)
            seen.add(s.index)
            if len(trunc) >= K:
                break
    return trunc[:K]


def _sum_cosine_merge(
    outputs: dict[str, SpecialistOutput],
    specialist_names: tuple[str, ...],
    cosine_segments: list[Segment],
    K: int,
) -> list[Segment]:
    """sum_cosine merge over a fixed specialist set + fair-backfill."""
    pool: dict[int, tuple[Segment, float]] = {}
    for name in specialist_names:
        so = outputs[name]
        for seg, cos in zip(so.segments, so.cosine_scores):
            if seg.index in pool:
                prev_seg, prev_score = pool[seg.index]
                pool[seg.index] = (prev_seg, prev_score + cos)
            else:
                pool[seg.index] = (seg, cos)
    ranked = sorted(pool.values(), key=lambda sc: -sc[1])
    return fair_backfill(ranked, cosine_segments, K)


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------
def evaluate_dataset(ds_name: str) -> tuple[list[dict], dict]:
    cfg = DATASETS[ds_name]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    questions = load_questions(ds_name)
    specialists = {name: build_specialist(name, store) for name in ADAPTIVE_ORDER}

    rows: list[dict] = []
    for qi, q in enumerate(questions):
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])
        cat = q.get("category", "unknown")

        query_emb = specialists["v2f"].embed_text(q_text)
        cosine_res = store.search(
            query_emb, top_k=max(BUDGETS), conversation_id=conv_id
        )
        cosine_segments = list(cosine_res.segments)

        outputs = run_specialists_cached(specialists, store, q_text, conv_id, query_emb)

        row: dict = {
            "dataset": ds_name,
            "conversation_id": conv_id,
            "question_index": q.get("question_index", qi),
            "category": cat,
            "num_source_turns": len(source_ids),
            "per_variant": {},
        }

        if not source_ids:
            # No-gold questions don't contribute to recall aggregates.
            rows.append(row)
            continue

        # Fixed-cost controls.
        for K in BUDGETS:
            # v2f
            segs_v2f = _v2f_alone_fairbackfill(outputs, cosine_segments, K)
            tids = {s.turn_id for s in segs_v2f}
            row["per_variant"].setdefault("v2f", {})[f"r@{K}"] = round(
                compute_recall(tids, source_ids), 4
            )
            row["per_variant"]["v2f"]["specialists_called"] = ["v2f"]
            row["per_variant"]["v2f"]["n_called"] = 1
            row["per_variant"]["v2f"]["llm_cost"] = SPECIALIST_COST["v2f"]

            # ens_2_v2f_typeenum (sum_cosine)
            segs_e2 = _sum_cosine_merge(
                outputs, ("v2f", "type_enumerated"), cosine_segments, K
            )
            tids = {s.turn_id for s in segs_e2}
            row["per_variant"].setdefault("ens_2_v2f_typeenum", {})[f"r@{K}"] = round(
                compute_recall(tids, source_ids), 4
            )
            row["per_variant"]["ens_2_v2f_typeenum"]["specialists_called"] = [
                "v2f",
                "type_enumerated",
            ]
            row["per_variant"]["ens_2_v2f_typeenum"]["n_called"] = 2
            row["per_variant"]["ens_2_v2f_typeenum"]["llm_cost"] = (
                SPECIALIST_COST["v2f"] + SPECIALIST_COST["type_enumerated"]
            )

            # ens_5 (sum_cosine, order from ADAPTIVE_ORDER)
            segs_e5 = _sum_cosine_merge(outputs, ADAPTIVE_ORDER, cosine_segments, K)
            tids = {s.turn_id for s in segs_e5}
            row["per_variant"].setdefault("ens_5", {})[f"r@{K}"] = round(
                compute_recall(tids, source_ids), 4
            )
            row["per_variant"]["ens_5"]["specialists_called"] = list(ADAPTIVE_ORDER)
            row["per_variant"]["ens_5"]["n_called"] = len(ADAPTIVE_ORDER)
            row["per_variant"]["ens_5"]["llm_cost"] = sum(
                SPECIALIST_COST[n] for n in ADAPTIVE_ORDER
            )

            # Adaptive variants
            for tau in TAU_VALUES:
                name = f"adaptive_tau_{tau:.1f}"
                ar = adaptive_ensemble(outputs, cosine_segments, K, tau, ADAPTIVE_ORDER)
                tids = {s.turn_id for s in ar.segments}
                row["per_variant"].setdefault(name, {})[f"r@{K}"] = round(
                    compute_recall(tids, source_ids), 4
                )
                # Record specialists_called only once per variant — but
                # adaptive gating can differ between K=20 and K=50 (since
                # novelty uses K). Store per-K.
                row["per_variant"][name][f"specialists_called@{K}"] = (
                    ar.specialists_called
                )
                row["per_variant"][name][f"n_called@{K}"] = len(ar.specialists_called)
                row["per_variant"][name][f"novelty_per_step@{K}"] = [
                    round(n, 4) for n in ar.novelty_per_step
                ]
                row["per_variant"][name][f"llm_cost@{K}"] = ar.llm_cost

        rows.append(row)
        if (qi + 1) % 10 == 0:
            print(f"  [{ds_name}] {qi + 1}/{len(questions)}", flush=True)

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


def aggregate(
    all_rows: dict[str, list[dict]],
) -> dict:
    out: dict = {"per_ds": {}, "per_category": {}, "variants": _variant_names()}

    for ds, rows in all_rows.items():
        per_v_recall: dict = defaultdict(lambda: defaultdict(list))
        per_v_n_called: dict = defaultdict(lambda: defaultdict(list))
        per_v_llm_cost: dict = defaultdict(lambda: defaultdict(list))
        per_v_called_dist: dict = defaultdict(lambda: defaultdict(Counter))
        n_gold = 0
        for r in rows:
            if r["num_source_turns"] == 0:
                continue
            n_gold += 1
            for v in _variant_names():
                vb = r["per_variant"].get(v, {})
                for K in BUDGETS:
                    if f"r@{K}" in vb:
                        per_v_recall[v][K].append(vb[f"r@{K}"])
                    if v.startswith("adaptive_tau_"):
                        n_key = f"n_called@{K}"
                        c_key = f"llm_cost@{K}"
                        if n_key in vb:
                            per_v_n_called[v][K].append(vb[n_key])
                            per_v_called_dist[v][K][vb[n_key]] += 1
                        if c_key in vb:
                            per_v_llm_cost[v][K].append(vb[c_key])
                    else:
                        # Fixed variants — static cost/count
                        per_v_n_called[v][K].append(vb.get("n_called", 1))
                        per_v_llm_cost[v][K].append(vb.get("llm_cost", 1.0))
                        per_v_called_dist[v][K][vb.get("n_called", 1)] += 1

        out["per_ds"][ds] = {
            "n_with_gold": n_gold,
            "variants": {
                v: {f"r@{K}": round(mean(per_v_recall[v][K]), 4) for K in BUDGETS}
                | {
                    f"mean_n_called@{K}": round(mean(per_v_n_called[v][K]), 3)
                    for K in BUDGETS
                }
                | {
                    f"mean_llm_cost@{K}": round(mean(per_v_llm_cost[v][K]), 3)
                    for K in BUDGETS
                }
                | {f"n_called_dist@{K}": dict(per_v_called_dist[v][K]) for K in BUDGETS}
                for v in _variant_names()
            },
        }

    # Combined (across both datasets) aggregate
    comb_r: dict = defaultdict(lambda: defaultdict(list))
    comb_n: dict = defaultdict(lambda: defaultdict(list))
    comb_c: dict = defaultdict(lambda: defaultdict(list))
    for ds, rows in all_rows.items():
        for r in rows:
            if r["num_source_turns"] == 0:
                continue
            for v in _variant_names():
                vb = r["per_variant"].get(v, {})
                for K in BUDGETS:
                    if f"r@{K}" in vb:
                        comb_r[v][K].append(vb[f"r@{K}"])
                    if v.startswith("adaptive_tau_"):
                        if f"n_called@{K}" in vb:
                            comb_n[v][K].append(vb[f"n_called@{K}"])
                        if f"llm_cost@{K}" in vb:
                            comb_c[v][K].append(vb[f"llm_cost@{K}"])
                    else:
                        comb_n[v][K].append(vb.get("n_called", 1))
                        comb_c[v][K].append(vb.get("llm_cost", 1.0))

    out["combined"] = {
        "variants": {
            v: {f"r@{K}": round(mean(comb_r[v][K]), 4) for K in BUDGETS}
            | {f"mean_n_called@{K}": round(mean(comb_n[v][K]), 3) for K in BUDGETS}
            | {f"mean_llm_cost@{K}": round(mean(comb_c[v][K]), 3) for K in BUDGETS}
            for v in _variant_names()
        },
    }

    # Per-category (combined across datasets) at K=50
    cat_r: dict = defaultdict(lambda: defaultdict(list))
    cat_n: dict = defaultdict(int)
    for ds, rows in all_rows.items():
        for r in rows:
            if r["num_source_turns"] == 0:
                continue
            cat = r["category"]
            cat_n[cat] += 1
            for v in _variant_names():
                vb = r["per_variant"].get(v, {})
                if "r@50" in vb:
                    cat_r[cat][v].append(vb["r@50"])
    out["per_category"] = {
        cat: {
            "n": cat_n[cat],
            "variants": {v: round(mean(cat_r[cat][v]), 4) for v in _variant_names()},
        }
        for cat in cat_n
    }

    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def render_markdown(aggr: dict) -> str:
    lines: list[str] = []
    lines.append("# Novelty-Gated Adaptive Ensemble\n")
    lines.append(
        "Sequentially run specialists in order "
        f"`{list(ADAPTIVE_ORDER)}`. After each specialist past the first, "
        "measure novelty = |R_s \\ R_accumulated| / K over its top-K "
        "turn_ids. If novelty > tau, merge (sum_cosine) and continue; "
        "else stop. Final top-K is fair-backfilled from raw-query cosine.\n"
    )
    lines.append(f"**Tau values**: {list(TAU_VALUES)}\n")
    lines.append(f"**Budgets**: K={list(BUDGETS)}\n")

    # LLM-call costs (from SPECIALIST_COST)
    lines.append("\n## Specialist costs (× v2f)\n")
    lines.append("| Specialist | cost |")
    lines.append("|---|---|")
    for n in ADAPTIVE_ORDER:
        lines.append(f"| {n} | {SPECIALIST_COST[n]} |")
    lines.append(
        f"\nTotal cost if all 5 called: "
        f"**{sum(SPECIALIST_COST[n] for n in ADAPTIVE_ORDER)}**×\n"
    )

    # Main recall table: variant × K × dataset
    for ds in DATASETS:
        if ds not in aggr["per_ds"]:
            continue
        dsblk = aggr["per_ds"][ds]
        lines.append(f"\n## Dataset: {ds}  (n_with_gold={dsblk['n_with_gold']})\n")
        lines.append(
            "| Variant | r@20 | r@50 | mean_n_called@20 | mean_n_called@50 | "
            "mean_cost@20 | mean_cost@50 |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for v in _variant_names():
            vd = dsblk["variants"][v]
            lines.append(
                f"| {v} | {vd['r@20']:.4f} | {vd['r@50']:.4f} | "
                f"{vd['mean_n_called@20']} | {vd['mean_n_called@50']} | "
                f"{vd['mean_llm_cost@20']}× | {vd['mean_llm_cost@50']}× |"
            )

    # Combined
    lines.append("\n## Combined (LoCoMo + synthetic)\n")
    lines.append(
        "| Variant | r@20 | r@50 | mean_n_called@20 | mean_n_called@50 | "
        "mean_cost@20 | mean_cost@50 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for v in _variant_names():
        vd = aggr["combined"]["variants"][v]
        lines.append(
            f"| {v} | {vd['r@20']:.4f} | {vd['r@50']:.4f} | "
            f"{vd['mean_n_called@20']} | {vd['mean_n_called@50']} | "
            f"{vd['mean_llm_cost@20']}× | {vd['mean_llm_cost@50']}× |"
        )

    # Distribution of n_called for adaptive variants at K=50 per dataset
    lines.append("\n## Distribution of specialists-called (adaptive) @K=50\n")
    for ds in DATASETS:
        if ds not in aggr["per_ds"]:
            continue
        lines.append(f"\n### {ds}\n")
        lines.append("| Variant | n=1 | n=2 | n=3 | n=4 | n=5 |")
        lines.append("|---|---|---|---|---|---|")
        for v in _variant_names():
            if not v.startswith("adaptive_tau_"):
                continue
            dist = aggr["per_ds"][ds]["variants"][v]["n_called_dist@50"]
            counts = [dist.get(i, 0) for i in (1, 2, 3, 4, 5)]
            lines.append(
                f"| {v} | {counts[0]} | {counts[1]} | {counts[2]} | "
                f"{counts[3]} | {counts[4]} |"
            )

    # Per-category @K=50 (combined)
    lines.append("\n## Per-category recall @K=50 (combined)\n")
    lines.append("| Category | n | v2f | ens_2 | ens_5 | a_0.1 | a_0.2 | a_0.3 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    cat_items = sorted(
        aggr["per_category"].items(),
        key=lambda kv: -kv[1]["variants"]["ens_5"],
    )
    for cat, info in cat_items:
        vs = info["variants"]
        lines.append(
            f"| {cat} | {info['n']} | {vs['v2f']:.4f} | "
            f"{vs['ens_2_v2f_typeenum']:.4f} | {vs['ens_5']:.4f} | "
            f"{vs['adaptive_tau_0.1']:.4f} | {vs['adaptive_tau_0.2']:.4f} | "
            f"{vs['adaptive_tau_0.3']:.4f} |"
        )

    # Cost-per-pp-gain versus ens_5 (combined, K=50)
    lines.append("\n## Cost-per-pp-gain vs v2f (combined, K=50)\n")
    cv = aggr["combined"]["variants"]
    v2f_r = cv["v2f"]["r@50"]
    lines.append("| Variant | r@50 | Δ vs v2f (pp) | mean_cost@50 | pp / cost |")
    lines.append("|---|---|---|---|---|")
    for v in _variant_names():
        r = cv[v]["r@50"]
        cost = cv[v]["mean_llm_cost@50"]
        delta_pp = (r - v2f_r) * 100.0
        pp_per_cost = delta_pp / cost if cost > 0 else 0.0
        lines.append(
            f"| {v} | {r:.4f} | {delta_pp:+.2f} | {cost}× | {pp_per_cost:+.2f} |"
        )

    # Verdict
    lines.append("\n## Verdict\n")
    ens5_r50 = cv["ens_5"]["r@50"]
    ens5_cost = cv["ens_5"]["mean_llm_cost@50"]

    best_adaptive = None
    for tau in TAU_VALUES:
        v = f"adaptive_tau_{tau:.1f}"
        r = cv[v]["r@50"]
        cost = cv[v]["mean_llm_cost@50"]
        if best_adaptive is None or (r, -cost) > (
            best_adaptive["r50"],
            -best_adaptive["cost"],
        ):
            best_adaptive = {"name": v, "r50": r, "cost": cost, "tau": tau}

    lines.append(
        f"- **ens_5 @K=50 combined**: r@50={ens5_r50:.4f}, cost={ens5_cost}×\n"
    )
    lines.append(
        f"- **best adaptive @K=50**: `{best_adaptive['name']}` r@50="
        f"{best_adaptive['r50']:.4f}, mean cost={best_adaptive['cost']}×\n"
    )
    # Ship logic:
    gap = ens5_r50 - best_adaptive["r50"]
    cost_ratio = best_adaptive["cost"] / ens5_cost if ens5_cost > 0 else 1.0
    if gap <= 0.005 and cost_ratio < 0.9:
        verdict = (
            f"SHIP — `{best_adaptive['name']}` matches ens_5 within "
            f"{gap * 100:.2f}pp at {cost_ratio * 100:.0f}% of its cost."
        )
    elif gap <= 0.02 and cost_ratio < 0.7:
        verdict = (
            f"NOTEWORTHY — `{best_adaptive['name']}` within "
            f"{gap * 100:.2f}pp of ens_5 at {cost_ratio * 100:.0f}% cost. "
            "Cost saver but slight recall loss."
        )
    elif cost_ratio > 0.95:
        verdict = (
            "ABANDON — gating rarely triggers; adaptive variant uses ~full "
            "ensemble cost."
        )
    else:
        verdict = (
            f"BORDERLINE — gap={gap * 100:.2f}pp at "
            f"{cost_ratio * 100:.0f}% cost; tune tau."
        )
    lines.append(f"- **Recommendation**: {verdict}\n")

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
        print(f"  n={meta['n_questions']} with_gold={meta['n_with_gold']}", flush=True)

    print("\nAggregating ...", flush=True)
    aggr = aggregate(all_rows)

    json_path = RESULTS_DIR / "adaptive_ensemble.json"
    payload = {
        "dataset_meta": dataset_meta,
        "aggregate": aggr,
        "per_question": all_rows,
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved {json_path}", flush=True)

    md = render_markdown(aggr)
    md_path = RESULTS_DIR / "adaptive_ensemble.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved {md_path}", flush=True)

    # Headline print
    print("\n" + "=" * 70)
    print("ADAPTIVE ENSEMBLE SUMMARY")
    print("=" * 70)
    cv = aggr["combined"]["variants"]
    for v in _variant_names():
        r = cv[v]["r@50"]
        c = cv[v]["mean_llm_cost@50"]
        print(f"  {v:26s} r@50={r:.4f}  mean_cost@50={c}×")


if __name__ == "__main__":
    main()
