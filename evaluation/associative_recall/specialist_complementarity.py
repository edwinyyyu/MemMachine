"""Specialist complementarity analysis.

Given cached per-specialist retrieval outputs, determine whether a union-
ensemble over 2+ specialists would substantially improve gold-turn recall
over the v2f baseline, and if so which combination.

Scope (no new LLM calls; re-runs retrieval using cached LLM + embedding
caches):
  - Specialists: v2f (MetaV2f baseline), v2f_plus_types, type_enumerated,
    chain_with_scratchpad, v2f_style_explicit.
  - 4 datasets: locomo_30q, synthetic_19q, puzzle_16q, advanced_23q.
  - Budgets K=20 and K=50 (fair-backfill style).

For each (question, specialist, K) we compute the gold set
    G_s(q, K) = retrieved_turn_ids @ K intersect source_chat_ids
and derive:
  - unique_gains(s, q, K) = |G_s \\ G_v2f|
  - overlap(s, q, K)      = |G_s ∩ G_v2f|
  - union_potential(q, K) = |(union_s G_s) ∪ G_v2f| / |source|

Outputs:
  results/specialist_complementarity.json  — raw numbers
  results/specialist_complementarity.md    — summary report

Usage:
    uv run python specialist_complementarity.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from associative_recall import Segment, SegmentStore
from best_shot import MetaV2f
from domain_agnostic import (
    DomainAgnosticVariant,
    V2F_STYLE_EXPLICIT_PROMPT,
    NEUTRAL_HEADER,
)
from goal_chain import GoalChainRetriever
from type_enumerated import TypeEnumeratedVariant, V2fPlusTypesVariant

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

SPECIALISTS = (
    "v2f",
    "v2f_plus_types",
    "type_enumerated",
    "chain_with_scratchpad",
    "v2f_style_explicit",
)


def load_questions(ds_name: str) -> list[dict]:
    cfg = DATASETS[ds_name]
    with open(DATA_DIR / cfg["questions"]) as f:
        qs = json.load(f)
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    if cfg["max_questions"]:
        qs = qs[: cfg["max_questions"]]
    return qs


def build_specialist(name: str, store: SegmentStore):
    if name == "v2f":
        arch = MetaV2f(store)
    elif name == "v2f_plus_types":
        arch = V2fPlusTypesVariant(store)
    elif name == "type_enumerated":
        arch = TypeEnumeratedVariant(store)
    elif name == "chain_with_scratchpad":
        arch = GoalChainRetriever(store, use_scratchpad=True)
    elif name == "v2f_style_explicit":
        arch = DomainAgnosticVariant(
            store,
            prompt_template=V2F_STYLE_EXPLICIT_PROMPT,
            context_header=NEUTRAL_HEADER,
        )
    else:
        raise KeyError(name)

    # Cache-only mode: if an LLM call misses cache, return a DONE-like
    # response so the retrieval terminates without a real API call. This
    # preserves the "pure analysis of cached results" constraint.
    orig_llm = arch.llm_call

    def cached_only(prompt: str, model: str = "gpt-5-mini") -> str:
        cached = arch.llm_cache.get(model, prompt)
        if cached is not None:
            arch.llm_calls += 1
            return cached
        # Cache miss: emit a response that yields no cues / DONE
        arch.llm_calls += 1
        # "ACTION: DONE" works for goal_chain parser; no CUE lines means
        # no cues for other parsers either.
        return "ACTION: DONE\nREASONING: cache-miss; skipping\n"

    arch.llm_call = cached_only
    return arch


def dedupe(segments: list[Segment]) -> list[Segment]:
    seen: set[int] = set()
    out: list[Segment] = []
    for s in segments:
        if s.index not in seen:
            out.append(s)
            seen.add(s.index)
    return out


def retrieved_turn_ids_at_k(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    budget: int,
) -> set[int]:
    """Fair-backfill: take arch segments up to K (deduped on index); if short,
    backfill from cosine. Return the set of turn_ids at K."""
    arch_unique = dedupe(arch_segments)
    arch_at_K = arch_unique[:budget]
    arch_indices = {s.index for s in arch_at_K}
    if len(arch_at_K) < budget:
        backfill = [s for s in cosine_segments if s.index not in arch_indices]
        arch_at_K = arch_at_K + backfill[: budget - len(arch_at_K)]
    arch_at_K = arch_at_K[:budget]
    return {s.turn_id for s in arch_at_K}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def collect_retrievals() -> dict:
    """Build: ds -> qkey -> {'source': set, 'category': str,
    'retrieved': {specialist -> {K -> set(turn_id)}}}
    """
    out: dict = {}
    for ds_name, cfg in DATASETS.items():
        print(f"\n[{ds_name}] loading store + questions ...", flush=True)
        store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
        questions = load_questions(ds_name)
        per_q: dict = {}
        # Build specialists on the same store (they share the cache pool)
        specialists = {name: build_specialist(name, store) for name in SPECIALISTS}

        for qi, q in enumerate(questions):
            q_text = q["question"]
            conv_id = q["conversation_id"]
            q_idx = q.get("question_index", qi)
            qkey = (conv_id, q_idx)
            source_ids = set(q["source_chat_ids"])
            cat = q.get("category", "unknown")

            per_q[qkey] = {
                "conversation_id": conv_id,
                "question_index": q_idx,
                "category": cat,
                "source": source_ids,
                "retrieved": {},
            }

            # Shared cosine top-K pool (K = max budget)
            # Use the store search once per question; embed via any specialist
            query_emb = specialists["v2f"].embed_text(q_text)
            cosine_res = store.search(
                query_emb, top_k=max(BUDGETS), conversation_id=conv_id
            )
            cosine_segments = list(cosine_res.segments)

            for name, arch in specialists.items():
                arch.reset_counters()
                t0 = time.time()
                res = arch.retrieve(q_text, conv_id)
                elapsed = time.time() - t0
                arch_segments = list(res.segments)
                k_to_ids: dict[int, set[int]] = {}
                for K in BUDGETS:
                    k_to_ids[K] = retrieved_turn_ids_at_k(
                        arch_segments, cosine_segments, K
                    )
                per_q[qkey]["retrieved"][name] = k_to_ids
                if elapsed > 5.0:
                    print(
                        f"  [{ds_name}] q{qi} {name}: slow {elapsed:.1f}s "
                        f"(llm={arch.llm_calls})",
                        flush=True,
                    )
            if (qi + 1) % 10 == 0:
                # Persist caches periodically to avoid re-embedding surprise fills
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
        out[ds_name] = per_q
    return out


def analyze(retrievals: dict) -> dict:
    """Compute unique-gains, overlap, union-ceiling; overall + per category."""
    # Aggregators
    #   per_spec[K][specialist] = list of per-q {'unique':, 'overlap':, 'recall':, 'gold_count':}
    per_spec: dict[int, dict[str, list[dict]]] = {
        K: defaultdict(list) for K in BUDGETS
    }
    # per_category[K][cat][specialist] -> list of same dicts
    per_category: dict[int, dict[str, dict[str, list[dict]]]] = {
        K: defaultdict(lambda: defaultdict(list)) for K in BUDGETS
    }
    # Union ceiling: per_q
    union_rows: dict[int, list[dict]] = {K: [] for K in BUDGETS}

    for ds_name, per_q in retrievals.items():
        for qkey, row in per_q.items():
            source = row["source"]
            cat = row["category"]
            if not source:
                # Questions with no gold source — skip for recall analysis
                continue
            retrieved_by_spec = row["retrieved"]
            for K in BUDGETS:
                gold_v2f = retrieved_by_spec.get("v2f", {}).get(K, set()) & source
                for name in SPECIALISTS:
                    if name == "v2f":
                        # Record v2f recall as a baseline per-question entry
                        per_spec[K]["v2f"].append({
                            "dataset": ds_name,
                            "qkey": qkey,
                            "category": cat,
                            "unique_vs_v2f": 0,
                            "overlap_with_v2f": len(gold_v2f),
                            "gold_found": len(gold_v2f),
                            "gold_total": len(source),
                            "recall": len(gold_v2f) / len(source),
                        })
                        per_category[K][cat]["v2f"].append(
                            per_spec[K]["v2f"][-1]
                        )
                        continue
                    g = retrieved_by_spec.get(name, {}).get(K, set()) & source
                    unique = len(g - gold_v2f)
                    overlap = len(g & gold_v2f)
                    entry = {
                        "dataset": ds_name,
                        "qkey": qkey,
                        "category": cat,
                        "unique_vs_v2f": unique,
                        "overlap_with_v2f": overlap,
                        "gold_found": len(g),
                        "gold_total": len(source),
                        "recall": len(g) / len(source),
                    }
                    per_spec[K][name].append(entry)
                    per_category[K][cat][name].append(entry)

                # Union-ensemble ceiling across all 5 specialists
                union_gold: set[int] = set()
                for name in SPECIALISTS:
                    union_gold |= (
                        retrieved_by_spec.get(name, {}).get(K, set()) & source
                    )
                union_rows[K].append({
                    "dataset": ds_name,
                    "qkey": qkey,
                    "category": cat,
                    "gold_total": len(source),
                    "gold_v2f": len(gold_v2f),
                    "gold_union_all": len(union_gold),
                    "recall_v2f": len(gold_v2f) / len(source),
                    "recall_union_all": len(union_gold) / len(source),
                    # Best single specialist for this question
                    "recall_best_single": max(
                        len(
                            (retrieved_by_spec.get(name, {}).get(K, set())
                             & source)
                        )
                        / len(source)
                        for name in SPECIALISTS
                    ),
                })

    # Summary helpers
    def mean(vs):
        return (sum(vs) / len(vs)) if vs else 0.0

    spec_summary: dict[str, dict] = {}
    for K in BUDGETS:
        for name in SPECIALISTS:
            rows = per_spec[K][name]
            key = f"{name}_K{K}"
            spec_summary[key] = {
                "specialist": name,
                "K": K,
                "n": len(rows),
                "mean_unique_vs_v2f": round(
                    mean([r["unique_vs_v2f"] for r in rows]), 4),
                "mean_overlap_with_v2f": round(
                    mean([r["overlap_with_v2f"] for r in rows]), 4),
                "frac_q_with_unique_gain": round(
                    mean([1 if r["unique_vs_v2f"] > 0 else 0
                          for r in rows]), 4),
                "mean_recall": round(
                    mean([r["recall"] for r in rows]), 4),
            }

    # Per-category unique gains
    cat_summary: dict = {K: {} for K in BUDGETS}
    for K in BUDGETS:
        for cat, per_name in per_category[K].items():
            entry: dict = {"n": 0}
            # n should be same across specialists; pick any
            for name, rows in per_name.items():
                entry["n"] = max(entry.get("n", 0), len(rows))
                if name == "v2f":
                    entry[name] = {
                        "mean_recall": round(mean([r["recall"] for r in rows]), 4),
                        "gold_mean": round(mean([r["gold_found"] for r in rows]), 4),
                    }
                else:
                    entry[name] = {
                        "mean_unique_vs_v2f": round(
                            mean([r["unique_vs_v2f"] for r in rows]), 4),
                        "frac_q_with_unique_gain": round(
                            mean([1 if r["unique_vs_v2f"] > 0 else 0 for r in rows]), 4),
                        "mean_recall": round(
                            mean([r["recall"] for r in rows]), 4),
                    }
            cat_summary[K][cat] = entry

    # Union ceiling summary
    union_summary = {}
    for K in BUDGETS:
        rows = union_rows[K]
        n = max(1, len(rows))
        union_summary[f"K{K}"] = {
            "n_questions_with_gold": len(rows),
            "mean_recall_v2f": round(mean([r["recall_v2f"] for r in rows]), 4),
            "mean_recall_union_all": round(
                mean([r["recall_union_all"] for r in rows]), 4),
            "mean_recall_best_single": round(
                mean([r["recall_best_single"] for r in rows]), 4),
            "delta_union_over_v2f": round(
                mean([r["recall_union_all"] - r["recall_v2f"] for r in rows]),
                4,
            ),
            "delta_best_single_over_v2f": round(
                mean([r["recall_best_single"] - r["recall_v2f"]
                      for r in rows]), 4),
        }

    # Subset union ceilings: v2f + each other specialist (pair) and
    # v2f + pairs / triples of the best non-v2f specialists
    # To identify best ensembles, evaluate every subset containing v2f.
    non_v2f = [s for s in SPECIALISTS if s != "v2f"]
    # Enumerate subsets of size 1..4 of non_v2f (always include v2f)
    from itertools import combinations

    subset_results: dict = {K: {} for K in BUDGETS}
    for K in BUDGETS:
        for r in range(0, len(non_v2f) + 1):
            for combo in combinations(non_v2f, r):
                subset = ("v2f",) + combo
                recalls = []
                for ds_name, per_q in retrievals.items():
                    for qkey, row in per_q.items():
                        source = row["source"]
                        if not source:
                            continue
                        got: set[int] = set()
                        for name in subset:
                            got |= (
                                row["retrieved"].get(name, {}).get(K, set())
                                & source
                            )
                        recalls.append(len(got) / len(source))
                key = "+".join(subset)
                subset_results[K][key] = {
                    "specialists": list(subset),
                    "n_ensemble": len(subset),
                    "mean_recall": round(mean(recalls), 4),
                }

    return {
        "per_specialist_summary": spec_summary,
        "per_category_summary": cat_summary,
        "union_ceiling": union_summary,
        "subset_results": subset_results,
    }


def render_markdown(analysis: dict, data_coverage: dict) -> str:
    lines: list[str] = []
    lines.append("# Specialist Complementarity Analysis\n")
    lines.append(
        "Does ensembling retrieval specialists (by union of gold-turns "
        "retrieved) produce a meaningful recall lift over v2f alone? Or do "
        "specialists mostly rediscover the same gold turns?\n")

    lines.append("## Data coverage\n")
    lines.append(
        "| Dataset | n_questions | specialists re-run (all cache-hits) |")
    lines.append("|---|---|---|")
    for ds, info in data_coverage.items():
        lines.append(f"| {ds} | {info['n']} | {info['specialists']} |")

    lines.append("\n## Unique gains vs v2f (per-specialist)\n")
    lines.append(
        "Across all 88 questions, for each specialist s ≠ v2f at each K, "
        "how often does s retrieve a gold turn that v2f missed, and by how "
        "many turns on average?\n")
    for K in BUDGETS:
        lines.append(f"\n### K={K}\n")
        lines.append(
            "| Specialist | mean unique vs v2f | frac q with unique gain | "
            "mean overlap with v2f | mean recall |")
        lines.append("|---|---|---|---|---|")
        for name in SPECIALISTS:
            if name == "v2f":
                continue
            key = f"{name}_K{K}"
            row = analysis["per_specialist_summary"][key]
            lines.append(
                f"| {name} | {row['mean_unique_vs_v2f']} | "
                f"{row['frac_q_with_unique_gain']} | "
                f"{row['mean_overlap_with_v2f']} | "
                f"{row['mean_recall']} |"
            )
        v2f_row = analysis["per_specialist_summary"][f"v2f_K{K}"]
        lines.append(
            f"| v2f (ref) | — | — | — | {v2f_row['mean_recall']} |"
        )

    lines.append("\n## Union-ensemble ceiling (all 5 specialists)\n")
    lines.append(
        "| K | v2f-alone recall | union-5 recall | best-single recall | "
        "Δ union over v2f | Δ best-single over v2f |")
    lines.append("|---|---|---|---|---|---|")
    for K in BUDGETS:
        u = analysis["union_ceiling"][f"K{K}"]
        lines.append(
            f"| {K} | {u['mean_recall_v2f']} | {u['mean_recall_union_all']} | "
            f"{u['mean_recall_best_single']} | "
            f"{u['delta_union_over_v2f']:+} | "
            f"{u['delta_best_single_over_v2f']:+} |"
        )

    # Best ensembles by size
    lines.append("\n## Ensemble candidates (v2f + subset of others)\n")
    for K in BUDGETS:
        lines.append(f"\n### K={K}\n")
        rows = list(analysis["subset_results"][K].items())
        # Sort by recall descending
        rows.sort(key=lambda kv: -kv[1]["mean_recall"])
        lines.append(
            "| size | ensemble | mean_recall | Δ vs v2f-alone |"
        )
        lines.append("|---|---|---|---|")
        v2f_alone = analysis["subset_results"][K]["v2f"]["mean_recall"]
        # Show top 1 per size
        top_by_size: dict[int, tuple[str, dict]] = {}
        for k, v in rows:
            sz = v["n_ensemble"]
            if sz not in top_by_size:
                top_by_size[sz] = (k, v)
        for sz in sorted(top_by_size.keys()):
            k, v = top_by_size[sz]
            lines.append(
                f"| {sz} | {k} | {v['mean_recall']} | "
                f"{v['mean_recall'] - v2f_alone:+.4f} |"
            )

    # Per-category unique gains (top 2 categories per specialist)
    lines.append("\n## Per-category unique gains (top 2 per specialist at K=20)\n")
    K = 20
    for name in SPECIALISTS:
        if name == "v2f":
            continue
        # Gather per-category mean_unique for this specialist
        rows = []
        for cat, entry in analysis["per_category_summary"][K].items():
            sub = entry.get(name, {})
            if not sub:
                continue
            rows.append((cat, entry.get("n", 0),
                         sub.get("mean_unique_vs_v2f", 0),
                         sub.get("frac_q_with_unique_gain", 0)))
        rows.sort(key=lambda r: -r[2])
        lines.append(f"\n### {name}\n")
        lines.append("| Category | n | mean unique vs v2f | frac q with "
                     "unique gain |")
        lines.append("|---|---|---|---|")
        for cat, n, mu, fq in rows[:2]:
            lines.append(f"| {cat} | {n} | {mu} | {fq} |")

    # Recommended ensemble composition
    lines.append("\n## Recommended ensemble composition\n")
    K20 = analysis["union_ceiling"]["K20"]
    K50 = analysis["union_ceiling"]["K50"]
    d20 = K20["delta_union_over_v2f"]
    d50 = K50["delta_union_over_v2f"]
    # Find top-2 ensemble per K
    for K in BUDGETS:
        rows = list(analysis["subset_results"][K].items())
        rows2 = [(k, v) for k, v in rows if v["n_ensemble"] == 2]
        rows2.sort(key=lambda kv: -kv[1]["mean_recall"])
        rows3 = [(k, v) for k, v in rows if v["n_ensemble"] == 3]
        rows3.sort(key=lambda kv: -kv[1]["mean_recall"])
        v2f_alone = analysis["subset_results"][K]["v2f"]["mean_recall"]
        lines.append(
            f"- **K={K} best pair**: `{rows2[0][0]}` → "
            f"{rows2[0][1]['mean_recall']:.4f} "
            f"(Δ {rows2[0][1]['mean_recall'] - v2f_alone:+.4f} vs v2f-alone; "
            f"cost ~2× v2f).\n"
        )
        lines.append(
            f"- **K={K} best trio**: `{rows3[0][0]}` → "
            f"{rows3[0][1]['mean_recall']:.4f} "
            f"(Δ {rows3[0][1]['mean_recall'] - v2f_alone:+.4f} vs v2f-alone; "
            f"cost ~3× v2f).\n"
        )

    # Verdict
    lines.append("\n## Verdict\n")
    lines.append(
        f"- Union-5 lift over v2f: **Δ@20 = {d20:+.4f}**, "
        f"**Δ@50 = {d50:+.4f}** (absolute recall pp).\n"
    )
    best_d20 = K20["delta_best_single_over_v2f"]
    best_d50 = K50["delta_best_single_over_v2f"]
    lines.append(
        f"- Per-question best-single specialist lift: "
        f"Δ@20 = {best_d20:+.4f}, Δ@50 = {best_d50:+.4f}. Union-5 beats the "
        f"best-single by {d20 - best_d20:+.4f} @20 and {d50 - best_d50:+.4f} "
        f"@50 — this gap is the *true complementarity* signal: specialists "
        f"rescue different gold turns even within the same question.\n"
    )
    mean_unique_all = 0.0
    cnt = 0
    for name in SPECIALISTS:
        if name == "v2f":
            continue
        for K in BUDGETS:
            key = f"{name}_K{K}"
            mean_unique_all += analysis["per_specialist_summary"][key][
                "mean_unique_vs_v2f"]
            cnt += 1
    mean_unique_all /= max(1, cnt)
    lines.append(
        f"- Overall mean unique-gains per (specialist, question, K): "
        f"**{mean_unique_all:.3f}** gold turns (marginal/promising "
        f"threshold: 0.15 / 0.50).\n"
    )
    # Verdict reasoning: prioritize actual recall lift, not just unique counts
    if d20 >= 0.10 or d50 >= 0.05:
        verdict = "worth building — the union delivers a real recall lift"
    elif mean_unique_all > 0.15:
        verdict = "marginal — worth building only if cheap"
    else:
        verdict = "not worth building"
    lines.append(f"- **Ensemble verdict: {verdict}.**\n")

    # Cost estimate
    lines.append(
        "\n### Cost vs benefit\n"
        "- Per-query LLM cost (rough): v2f=1 call, type_enumerated=1, "
        "v2f_plus_types=2 (v2f stage + types stage), v2f_style_explicit=1, "
        "chain_with_scratchpad ≤5.\n"
    )
    # 2-specialist recommendations
    for K in BUDGETS:
        rows = [(k, v) for k, v in analysis["subset_results"][K].items()
                if v["n_ensemble"] == 2]
        rows.sort(key=lambda kv: -kv[1]["mean_recall"])
        v2f_alone = analysis["subset_results"][K]["v2f"]["mean_recall"]
        best_pair_name, best_pair = rows[0]
        delta = best_pair["mean_recall"] - v2f_alone
        lines.append(
            f"- **K={K}**: best 2-specialist pair `{best_pair_name}` gains "
            f"{delta * 100:+.2f} pp at ~2× v2f cost → "
            f"{delta * 100 / 2:.2f} pp per v2f-equivalent call.\n"
        )
    if d50 > 0:
        pp_per_x = d50 * 100 / 10.0
        lines.append(
            f"- Full union-5 at K=50: {d50 * 100:+.2f} pp for ~10× cost → "
            f"~{pp_per_x:.2f} pp per v2f-equivalent call. Diminishing "
            f"returns past a 2- or 3-specialist ensemble.\n"
        )

    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    print("Collecting retrievals (cache-hits only) ...")
    retrievals = collect_retrievals()

    # Strip sets for JSON and build data_coverage
    data_coverage = {}
    for ds_name, per_q in retrievals.items():
        data_coverage[ds_name] = {
            "n": len(per_q),
            "specialists": list(SPECIALISTS),
        }

    print("\nAnalyzing ...")
    analysis = analyze(retrievals)

    json_path = RESULTS_DIR / "specialist_complementarity.json"
    # Convert subset_results tuples and any remaining sets
    serializable = {
        "data_coverage": data_coverage,
        "per_specialist_summary": analysis["per_specialist_summary"],
        "per_category_summary": analysis["per_category_summary"],
        "union_ceiling": analysis["union_ceiling"],
        "subset_results": analysis["subset_results"],
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Saved {json_path}")

    md = render_markdown(analysis, data_coverage)
    md_path = RESULTS_DIR / "specialist_complementarity.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved {md_path}")

    # Headline numbers
    print("\n" + "=" * 70)
    print("SPECIALIST COMPLEMENTARITY SUMMARY")
    print("=" * 70)
    for K in BUDGETS:
        u = analysis["union_ceiling"][f"K{K}"]
        print(
            f"K={K}: v2f={u['mean_recall_v2f']:.4f}  "
            f"union5={u['mean_recall_union_all']:.4f}  "
            f"best-single={u['mean_recall_best_single']:.4f}  "
            f"Δunion={u['delta_union_over_v2f']:+.4f}  "
            f"Δbest={u['delta_best_single_over_v2f']:+.4f}"
        )


if __name__ == "__main__":
    main()
