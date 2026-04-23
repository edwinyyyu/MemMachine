"""Composition study: do the three shipped architectural wins compose?

Three wins being composed:
  1. Keyword router at K=50 (router_study.KEYWORD_RULES) — picks specialist per q
  2. Ensemble ens_2_v2f_typeenum (v2f + type_enumerated, sum_cosine merge)
  3. Critical-info store (always_top_M, top_m=5, min_score=0.2)

Variants evaluated (on LoCoMo-30 + synthetic-19 + puzzle-16 + advanced-23,
K=20 and K=50, fair-backfilled):

  Baseline / controls:
    - v2f                         (reference)
    - router_v2fplus_default      (keyword router with v2f_plus_types default)
    - ens_2                       (v2f + type_enumerated sum_cosine)
    - crit_only                   (v2f + critical-info always_top_M)

  Compositions:
    - ens_2 + crit                (ens_2 then overlay crit always_top_M)
    - router_ens                  (router decides which ensemble to run)
    - router_ens + crit           (full stack)
    - ens_all + crit              (always run ens_5 sum_cosine + crit)

No framework / specialist / ensemble / critical-info files are modified — we
import and reuse.

Usage:
    uv run python composition_eval.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    Segment,
    SegmentStore,
)
from critical_info_store import (
    CriticalAltKey,
    CriticalInfoGenerator,
    CriticalInfoStore,
    classify_turns,
    decisions_to_altkeys,
    merge_always_top_m,
)
from ensemble_retrieval import (
    ENSEMBLE_COMPOSITIONS,
    SPECIALISTS,
    SpecialistOutput,
    build_specialist,
    ensemble_at_k,
    run_specialists_cached,
)
from ingest_regex_eval import (
    Embedder,
    embed_texts_cached,
    compute_recall,
    fair_backfill_turn_ids,
)
from router_study import KEYWORD_RULES

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ---------------------------------------------------------------------------
# Paths, datasets, budgets
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_MY_LLM = CACHE_DIR / "compose_llm_cache.json"
CACHE_MY_EMB = CACHE_DIR / "compose_embedding_cache.json"

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

# Specialist call costs (relative to 1 v2f call), per specialist_complementarity.
SPECIALIST_COST: dict[str, float] = {
    "v2f": 1.0,
    "v2f_plus_types": 2.0,
    "type_enumerated": 1.0,
    "chain_with_scratchpad": 5.0,
    "v2f_style_explicit": 1.0,
}

# Use prompt v3 — that's the currently-shipped classifier (matches
# critical_info_store.json on disk).
CRITICAL_PROMPT_VERSION = "v3"


# ---------------------------------------------------------------------------
# Keyword router mapping to ensemble-specialist names
# ---------------------------------------------------------------------------
# KEYWORD_RULES yields labels in router_study's vocabulary; for composition
# we map each decision to an ensemble composition (tuple of specialist names).
#
#   logic_constraint → type_enumerated          → run ens_2_v2f_typeenum
#   chain-structured categories → chain         → run v2f + chain_with_scratchpad
#   v2f_plus_types (default)                     → run v2f_plus_types alone
#   v2f_style_explicit                           → run v2f_style_explicit alone
#
# router_ens picks ONE ensemble composition per question (not a specialist).
COMPOSITION_FOR_LABEL: dict[str, tuple[str, ...]] = {
    "type_enumerated": ("v2f", "type_enumerated"),
    "chain": ("v2f", "chain_with_scratchpad"),
    "v2f_plus_types": ("v2f_plus_types",),
    "v2f_style_explicit": ("v2f_style_explicit",),
    "v2f": ("v2f",),
}


def route_keyword_label(question: str) -> str:
    """Return label from router_study.KEYWORD_RULES."""
    for pat, lab in KEYWORD_RULES:
        if pat.search(question):
            return lab
    return "v2f_plus_types"


# ---------------------------------------------------------------------------
# Segment loaders / question loaders
# ---------------------------------------------------------------------------
def load_questions(ds_name: str) -> list[dict]:
    cfg = DATASETS[ds_name]
    with open(DATA_DIR / cfg["questions"]) as f:
        qs = json.load(f)
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    if cfg["max_questions"]:
        qs = qs[: cfg["max_questions"]]
    return qs


# ---------------------------------------------------------------------------
# Fair-backfill wrapper that uses ensemble ranked output then cosine cushion
# ---------------------------------------------------------------------------
def _dedupe_by_index(segments: list[Segment]) -> list[Segment]:
    seen: set[int] = set()
    out: list[Segment] = []
    for s in segments:
        if s.index in seen:
            continue
        out.append(s)
        seen.add(s.index)
    return out


def v2f_main_ranked_with_scores(
    v2f_output: SpecialistOutput,
    cosine_segments: list[Segment],
    cosine_scores: list[float],
) -> list[tuple[Segment, float]]:
    """Build (segment, score) list in fair-backfill order: v2f's arch cues first
    (scored 10+ so they sort above any cosine), then cosine top-K by score."""
    seen: set[int] = set()
    ranked: list[tuple[Segment, float]] = []
    EPS = 0.001
    for rank, s in enumerate(v2f_output.segments):
        if s.index in seen:
            continue
        ranked.append((s, 10.0 - rank * EPS))
        seen.add(s.index)
    cos_by_idx = {s.index: sc for s, sc in zip(cosine_segments, cosine_scores)}
    for s in cosine_segments:
        if s.index in seen:
            continue
        ranked.append((s, cos_by_idx.get(s.index, 0.0)))
        seen.add(s.index)
    return ranked


def ensemble_main_ranked_with_scores(
    outputs: dict[str, SpecialistOutput],
    composition: tuple[str, ...],
    cosine_segments: list[Segment],
    cosine_scores: list[float],
) -> list[tuple[Segment, float]]:
    """For a composition, gather the sum-cosine merged ranking; backfill with
    cosine. Returns (segment, score) list.

    For a single-specialist composition (len=1) behaves like v2f_main_ranked
    style: specialist segments first, then cosine backfill.
    """
    if len(composition) == 1:
        name = composition[0]
        so = outputs[name]
        seen: set[int] = set()
        ranked: list[tuple[Segment, float]] = []
        EPS = 0.001
        for rank, s in enumerate(so.segments):
            if s.index in seen:
                continue
            ranked.append((s, 10.0 - rank * EPS))
            seen.add(s.index)
        cos_by_idx = {s.index: sc for s, sc in zip(cosine_segments,
                                                    cosine_scores)}
        for s in cosine_segments:
            if s.index in seen:
                continue
            ranked.append((s, cos_by_idx.get(s.index, 0.0)))
            seen.add(s.index)
        return ranked

    # Multi-specialist: use sum_cosine merge (preferred per ensemble_study)
    sub = {n: outputs[n] for n in composition if n in outputs}
    pool: dict[int, dict] = {}
    for name, so in sub.items():
        for rank, (seg, cos) in enumerate(zip(so.segments, so.cosine_scores)):
            entry = pool.setdefault(seg.index,
                                    {"segment": seg, "scores": []})
            entry["scores"].append(cos)
    # Score = sum of per-specialist cosine scores; unique across ensemble
    merged: list[tuple[Segment, float]] = []
    for idx, e in pool.items():
        cs = sum(e["scores"])
        # Bump so ensemble picks sort above cosine backfill
        merged.append((e["segment"], cs + 5.0))
    merged.sort(key=lambda rc: -rc[1])

    seen: set[int] = {s.index for s, _ in merged}
    cos_by_idx = {s.index: sc for s, sc in zip(cosine_segments, cosine_scores)}
    for s in cosine_segments:
        if s.index in seen:
            continue
        merged.append((s, cos_by_idx.get(s.index, 0.0)))
        seen.add(s.index)
    return merged


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
@dataclass
class QuestionContext:
    question: dict
    q_text: str
    conv_id: str
    source_ids: set[int]
    category: str
    query_emb: np.ndarray
    cosine_segments: list[Segment]
    cosine_scores: list[float]
    outputs: dict[str, SpecialistOutput]  # all 5 specialists, cached
    router_label: str
    router_composition: tuple[str, ...]
    crit_ranked: list[tuple[int, float, Segment]]  # ordered by score desc


def build_question_contexts(
    store: SegmentStore,
    specialists: dict,
    questions: list[dict],
    crit_store: CriticalInfoStore,
) -> list[QuestionContext]:
    ctxs: list[QuestionContext] = []
    for q in questions:
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])
        cat = q.get("category", "unknown")

        # Embed query via v2f's cache; shared across specialists
        q_emb = specialists["v2f"].embed_text(q_text)
        cos_res = store.search(q_emb, top_k=max(BUDGETS),
                               conversation_id=conv_id)
        cos_segs = list(cos_res.segments)
        cos_scores = list(cos_res.scores)

        outputs = run_specialists_cached(specialists, store, q_text,
                                          conv_id, q_emb)

        label = route_keyword_label(q_text)
        comp = COMPOSITION_FOR_LABEL.get(label, ("v2f",))

        # Pull crit top-M+ candidates (enough for max K budget)
        crit = crit_store.search_per_parent(
            q_emb,
            top_m=max(BUDGETS),
            conversation_id=conv_id,
            min_score=-1.0,  # filtering at merge time
        )

        ctxs.append(QuestionContext(
            question=q,
            q_text=q_text,
            conv_id=conv_id,
            source_ids=source_ids,
            category=cat,
            query_emb=q_emb,
            cosine_segments=cos_segs,
            cosine_scores=cos_scores,
            outputs=outputs,
            router_label=label,
            router_composition=comp,
            crit_ranked=crit,
        ))
    return ctxs


# ---------------------------------------------------------------------------
# Variant implementations — each returns retrieved turn_ids given context + K
# ---------------------------------------------------------------------------
def variant_v2f(ctx: QuestionContext, K: int) -> set[int]:
    segs = _dedupe_by_index(list(ctx.outputs["v2f"].segments))
    ids = fair_backfill_turn_ids(segs, ctx.cosine_segments, K)
    return ids


def variant_router_v2fplus_default(ctx: QuestionContext, K: int) -> set[int]:
    """Keyword router's picked single specialist, fair-backfilled.

    Mirrors router_study with v2f_plus_types default. Note: the composition
    here is the 1-specialist composition assigned by COMPOSITION_FOR_LABEL.
    For chain we route to 'chain_with_scratchpad' (from COMPOSITION_FOR_LABEL
    2nd element); but router_study routes to `chain` as single specialist.
    Use a simpler direct map here for clarity.
    """
    label = ctx.router_label
    # router_study defaults:
    #   logic_constraint → type_enumerated
    #   chain-structured categories → chain
    #   default → v2f_plus_types
    if label == "type_enumerated":
        spec_name = "type_enumerated"
    elif label == "chain":
        spec_name = "chain_with_scratchpad"
    elif label == "v2f_style_explicit":
        spec_name = "v2f_style_explicit"
    else:  # v2f_plus_types default
        spec_name = "v2f_plus_types"
    so = ctx.outputs[spec_name]
    segs = _dedupe_by_index(list(so.segments))
    return fair_backfill_turn_ids(segs, ctx.cosine_segments, K)


def variant_ens_2(ctx: QuestionContext, K: int) -> set[int]:
    segs = ensemble_at_k(
        ctx.outputs, ("v2f", "type_enumerated"), "sum_cosine",
        ctx.cosine_segments, K,
    )
    return {s.turn_id for s in segs}


def variant_crit_only(ctx: QuestionContext, K: int) -> set[int]:
    v2f_main = v2f_main_ranked_with_scores(
        ctx.outputs["v2f"], ctx.cosine_segments, ctx.cosine_scores,
    )
    merged_segs = merge_always_top_m(
        v2f_main, ctx.crit_ranked, K, top_m=5, min_score=0.2,
    )
    return {s.turn_id for s in merged_segs}


def variant_ens_2_plus_crit(ctx: QuestionContext, K: int) -> set[int]:
    main_ranked = ensemble_main_ranked_with_scores(
        ctx.outputs, ("v2f", "type_enumerated"),
        ctx.cosine_segments, ctx.cosine_scores,
    )
    merged_segs = merge_always_top_m(
        main_ranked, ctx.crit_ranked, K, top_m=5, min_score=0.2,
    )
    return {s.turn_id for s in merged_segs}


def variant_router_ens(ctx: QuestionContext, K: int) -> set[int]:
    """Router picks ensemble composition per question."""
    comp = ctx.router_composition
    if len(comp) == 1:
        so = ctx.outputs[comp[0]]
        segs = _dedupe_by_index(list(so.segments))
        return fair_backfill_turn_ids(segs, ctx.cosine_segments, K)
    segs = ensemble_at_k(
        ctx.outputs, comp, "sum_cosine", ctx.cosine_segments, K,
    )
    return {s.turn_id for s in segs}


def variant_router_ens_plus_crit(ctx: QuestionContext, K: int) -> set[int]:
    comp = ctx.router_composition
    main_ranked = ensemble_main_ranked_with_scores(
        ctx.outputs, comp, ctx.cosine_segments, ctx.cosine_scores,
    )
    merged_segs = merge_always_top_m(
        main_ranked, ctx.crit_ranked, K, top_m=5, min_score=0.2,
    )
    return {s.turn_id for s in merged_segs}


def variant_ens_all_plus_crit(ctx: QuestionContext, K: int) -> set[int]:
    comp = ENSEMBLE_COMPOSITIONS["ens_5"]
    main_ranked = ensemble_main_ranked_with_scores(
        ctx.outputs, comp, ctx.cosine_segments, ctx.cosine_scores,
    )
    merged_segs = merge_always_top_m(
        main_ranked, ctx.crit_ranked, K, top_m=5, min_score=0.2,
    )
    return {s.turn_id for s in merged_segs}


VARIANT_FUNCS = {
    "v2f": variant_v2f,
    "router_v2fplus_default": variant_router_v2fplus_default,
    "ens_2": variant_ens_2,
    "crit_only": variant_crit_only,
    "ens_2_plus_crit": variant_ens_2_plus_crit,
    "router_ens": variant_router_ens,
    "router_ens_plus_crit": variant_router_ens_plus_crit,
    "ens_all_plus_crit": variant_ens_all_plus_crit,
}

VARIANTS_ORDER = list(VARIANT_FUNCS.keys())


# ---------------------------------------------------------------------------
# LLM-call cost estimation per variant per question
# ---------------------------------------------------------------------------
def llm_cost_per_variant(variant: str, ctx: QuestionContext) -> float:
    """Expected LLM-call units per question (v2f-call units) for this
    variant on this question. Classifier cost is ingestion-time (amortized)
    so we only count retrieval-time LLM calls."""
    if variant == "v2f":
        return 1.0
    if variant == "router_v2fplus_default":
        label = ctx.router_label
        if label == "type_enumerated":
            return SPECIALIST_COST["type_enumerated"]
        if label == "chain":
            return SPECIALIST_COST["chain_with_scratchpad"]
        if label == "v2f_style_explicit":
            return SPECIALIST_COST["v2f_style_explicit"]
        return SPECIALIST_COST["v2f_plus_types"]
    if variant == "ens_2":
        return SPECIALIST_COST["v2f"] + SPECIALIST_COST["type_enumerated"]
    if variant == "crit_only":
        return SPECIALIST_COST["v2f"]
    if variant == "ens_2_plus_crit":
        return SPECIALIST_COST["v2f"] + SPECIALIST_COST["type_enumerated"]
    if variant == "router_ens":
        return sum(SPECIALIST_COST[s] for s in ctx.router_composition)
    if variant == "router_ens_plus_crit":
        return sum(SPECIALIST_COST[s] for s in ctx.router_composition)
    if variant == "ens_all_plus_crit":
        return sum(SPECIALIST_COST[s] for s in ENSEMBLE_COMPOSITIONS["ens_5"])
    return 1.0


# ---------------------------------------------------------------------------
# Per-dataset evaluator
# ---------------------------------------------------------------------------
def evaluate_dataset(
    ds_name: str,
    generator: CriticalInfoGenerator,
    client: OpenAI,
    embedder: Embedder,
) -> dict:
    cfg = DATASETS[ds_name]
    print(f"\n{'=' * 70}\n[{ds_name}]\n{'=' * 70}", flush=True)
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    questions = load_questions(ds_name)
    print(f"  questions={len(questions)} segments={len(store.segments)}",
          flush=True)

    # --- Build critical-info store for this dataset ---
    conv_ids = {q["conversation_id"] for q in questions}
    target = [s for s in store.segments if s.conversation_id in conv_ids]
    print(f"  target segments: {len(target)} — classifying (LLM cached)",
          flush=True)
    t_c = time.time()
    decisions = classify_turns(generator, target, log_every=200)
    print(f"  classify done in {time.time() - t_c:.1f}s — "
          f"crit={sum(1 for d in decisions if d.critical)}/{len(decisions)}",
          flush=True)
    alt_keys = decisions_to_altkeys(decisions)
    alt_texts = [k.text for k in alt_keys]
    if alt_texts:
        alt_embs = embed_texts_cached(
            client, embedder.embedding_cache, alt_texts,
        )
    else:
        alt_embs = np.zeros((0, 1536), dtype=np.float32)
    crit_store = CriticalInfoStore(store, alt_keys, alt_embs)
    try:
        embedder.save()
    except Exception as e:
        print(f"  (warn) embedder.save failed: {e}", flush=True)
    try:
        generator.save()
    except Exception as e:
        print(f"  (warn) generator.save failed: {e}", flush=True)

    # --- Build all specialists (cache-only) ---
    specialists = {name: build_specialist(name, store) for name in SPECIALISTS}

    # --- Collect per-question contexts ---
    print("  building per-question contexts (specialist outputs + crit)...",
          flush=True)
    t_ctx = time.time()
    ctxs = build_question_contexts(store, specialists, questions, crit_store)
    print(f"  contexts built in {time.time() - t_ctx:.1f}s", flush=True)

    # Save specialist caches (whatever minor updates happened)
    for arch in specialists.values():
        try:
            arch.save_caches()
        except Exception:
            pass

    # --- Evaluate each variant ---
    per_q_rows: list[dict] = []
    for ctx in ctxs:
        row: dict = {
            "dataset": ds_name,
            "conversation_id": ctx.conv_id,
            "question_index": ctx.question.get("question_index", -1),
            "category": ctx.category,
            "num_source_turns": len(ctx.source_ids),
            "router_label": ctx.router_label,
            "router_composition": list(ctx.router_composition),
            "recall": {},
            "llm_calls_per_variant": {},
        }
        if not ctx.source_ids:
            per_q_rows.append(row)
            continue
        for var_name, fn in VARIANT_FUNCS.items():
            row["recall"][var_name] = {}
            for K in BUDGETS:
                ids = fn(ctx, K)
                r = compute_recall(ids, ctx.source_ids)
                row["recall"][var_name][f"r@{K}"] = round(r, 4)
            row["llm_calls_per_variant"][var_name] = round(
                llm_cost_per_variant(var_name, ctx), 2,
            )
        per_q_rows.append(row)

    # --- Aggregate ---
    per_variant: dict = {}
    for var in VARIANTS_ORDER:
        per_variant[var] = {}
        for K in BUDGETS:
            vals = [r["recall"][var][f"r@{K}"]
                    for r in per_q_rows
                    if r["num_source_turns"] > 0]
            per_variant[var][f"r@{K}"] = (
                round(sum(vals) / len(vals), 4) if vals else 0.0
            )
    # Cost
    per_variant_cost: dict = {}
    for var in VARIANTS_ORDER:
        costs = [r["llm_calls_per_variant"][var]
                 for r in per_q_rows
                 if r["num_source_turns"] > 0]
        per_variant_cost[var] = round(
            sum(costs) / len(costs), 3) if costs else 0.0

    # Per-category
    by_cat: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cat_counts: dict = defaultdict(int)
    for r in per_q_rows:
        if r["num_source_turns"] == 0:
            continue
        cat = r["category"]
        cat_counts[cat] += 1
        for var in VARIANTS_ORDER:
            for K in BUDGETS:
                by_cat[cat][var][K].append(r["recall"][var][f"r@{K}"])
    per_category = {
        cat: {
            "n": cat_counts[cat],
            "variants": {
                var: {
                    f"r@{K}": round(
                        sum(by_cat[cat][var][K]) /
                        max(1, len(by_cat[cat][var][K])), 4,
                    )
                    for K in BUDGETS
                }
                for var in VARIANTS_ORDER
            },
        }
        for cat in cat_counts
    }

    # Router label distribution
    routing_dist: dict = defaultdict(int)
    for r in per_q_rows:
        routing_dist[r["router_label"]] += 1

    # Flagging stats
    n_crit = sum(1 for d in decisions if d.critical)

    return {
        "ds_name": ds_name,
        "n_questions": len(questions),
        "n_with_gold": sum(1 for r in per_q_rows if r["num_source_turns"] > 0),
        "n_target_segments": len(target),
        "n_critical_turns": n_crit,
        "flag_rate": round(n_crit / max(1, len(decisions)), 4),
        "n_altkeys_dedup": len(alt_keys),
        "routing_distribution": dict(routing_dist),
        "per_variant": per_variant,
        "per_variant_llm_cost": per_variant_cost,
        "per_category": per_category,
        "per_question": per_q_rows,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def render_markdown(all_results: dict, classifier_cost: dict,
                    total_elapsed: float) -> str:
    L: list[str] = []
    L.append("# Composition Study — do the three shipped wins compose?\n")
    L.append(
        "Three shipped wins being composed: **keyword router**, "
        "**ens_2_v2f_typeenum**, **critical-info always_top_M**.\n")
    L.append(
        "Variants evaluated on LoCoMo-30 + synthetic-19 + puzzle-16 + "
        "advanced-23 at K=20 and K=50, fair-backfilled.\n")
    L.append(f"\nElapsed: {total_elapsed:.0f}s.\n")

    # Headline table: every variant × K × each dataset
    for K in BUDGETS:
        L.append(f"\n## Recall table (r@{K})\n")
        L.append("| Variant | " + " | ".join(DATASETS) + " | overall |")
        L.append("|---|" + "---|" * (len(DATASETS) + 1))
        for var in VARIANTS_ORDER:
            row = [var]
            vals_all = []
            wt_all = 0
            for ds in DATASETS:
                res = all_results[ds]
                v = res["per_variant"][var][f"r@{K}"]
                n = res["n_with_gold"]
                row.append(f"{v:.4f}")
                vals_all.append(v * n)
                wt_all += n
            overall = (sum(vals_all) / wt_all) if wt_all else 0.0
            row.append(f"**{overall:.4f}**")
            L.append("| " + " | ".join(row) + " |")

    # Additivity check on LoCoMo and synthetic
    L.append("\n## Additivity check: ens_2 + crit vs sum of gains\n")
    L.append(
        "Compute Δ(ens_2 + crit) vs Δ(ens_2) + Δ(crit_only) per dataset × K.\n")
    L.append("| Dataset | K | v2f | ens_2 | crit_only | ens_2+crit | "
             "Δ_ens_2 | Δ_crit | Δ_both | sum_indiv | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        res = all_results[ds]
        for K in BUDGETS:
            v2f = res["per_variant"]["v2f"][f"r@{K}"]
            e2 = res["per_variant"]["ens_2"][f"r@{K}"]
            c = res["per_variant"]["crit_only"][f"r@{K}"]
            both = res["per_variant"]["ens_2_plus_crit"][f"r@{K}"]
            de2 = e2 - v2f
            dc = c - v2f
            db = both - v2f
            si = de2 + dc
            if si == 0 and db == 0:
                verdict = "no gain"
            elif db >= 0.9 * si and db > max(de2, dc):
                verdict = "additive"
            elif db >= max(de2, dc):
                verdict = "partial (dominant single sticks)"
            elif db < max(de2, dc):
                verdict = "cannibalize"
            else:
                verdict = "mixed"
            L.append(
                f"| {ds} | {K} | {v2f:.4f} | {e2:.4f} | {c:.4f} | {both:.4f}"
                f" | {de2:+.4f} | {dc:+.4f} | {db:+.4f} | {si:+.4f} | "
                f"{verdict} |"
            )

    # Router distribution
    L.append("\n## Keyword-router label distribution\n")
    L.append("| Dataset | " + " | ".join(
        ["v2f", "v2f_plus_types", "type_enumerated", "chain",
         "v2f_style_explicit"]) + " |")
    L.append("|---|" + "---|" * 5)
    for ds in DATASETS:
        rd = all_results[ds]["routing_distribution"]
        row = [ds]
        for lab in ["v2f", "v2f_plus_types", "type_enumerated", "chain",
                    "v2f_style_explicit"]:
            row.append(str(rd.get(lab, 0)))
        L.append("| " + " | ".join(row) + " |")

    # Per-category breakdown on LoCoMo and synthetic @ K=50
    for ds in ("locomo_30q", "synthetic_19q"):
        if ds not in all_results:
            continue
        res = all_results[ds]
        L.append(f"\n## Per-category r@50 on {ds}\n")
        L.append("| category | n | " + " | ".join(VARIANTS_ORDER) + " |")
        L.append("|---|---|" + "---|" * len(VARIANTS_ORDER))
        for cat in sorted(res["per_category"].keys()):
            e = res["per_category"][cat]
            row = [cat, str(e["n"])]
            for var in VARIANTS_ORDER:
                row.append(f"{e['variants'][var]['r@50']:.3f}")
            L.append("| " + " | ".join(row) + " |")

    # LLM-call cost per variant
    L.append("\n## LLM retrieval cost per question (relative to 1 v2f call)\n")
    L.append("| Variant | " + " | ".join(DATASETS) + " |")
    L.append("|---|" + "---|" * len(DATASETS))
    for var in VARIANTS_ORDER:
        row = [var]
        for ds in DATASETS:
            c = all_results[ds]["per_variant_llm_cost"][var]
            row.append(f"{c:.2f}×")
        L.append("| " + " | ".join(row) + " |")

    # Classifier cost
    L.append("\n## Critical-info classifier (ingest-time, one-off cost)\n")
    L.append(
        f"- Prompt version: {CRITICAL_PROMPT_VERSION}\n"
        f"- New calls this run: {classifier_cost['n_uncached']}, "
        f"cached: {classifier_cost['n_cached']}\n"
        f"- Input tokens: {classifier_cost['prompt_tokens']} "
        f"output tokens: {classifier_cost['completion_tokens']}\n"
        f"- Est USD (gpt-5-mini @ $0.25/M in, $2/M out): "
        f"${classifier_cost['est_usd']:.4f}\n")
    for ds in DATASETS:
        res = all_results[ds]
        L.append(
            f"  - {ds}: flag rate {res['flag_rate']*100:.2f}% "
            f"({res['n_critical_turns']}/{res['n_target_segments']} turns), "
            f"{res['n_altkeys_dedup']} alt-keys\n")

    # Verdict
    L.append("\n## Verdict\n")
    # Find best variant overall at K=50 by weighted mean
    best_var = None
    best_r = -1.0
    for var in VARIANTS_ORDER:
        total = 0.0
        wt = 0
        for ds in DATASETS:
            r = all_results[ds]["per_variant"][var]["r@50"]
            n = all_results[ds]["n_with_gold"]
            total += r * n
            wt += n
        mean_r = total / max(1, wt)
        if mean_r > best_r:
            best_r = mean_r
            best_var = var
    L.append(f"- Best variant overall @ K=50 (weighted): "
             f"**{best_var}** r@50={best_r:.4f}\n")

    # LoCoMo headline
    loc = all_results["locomo_30q"]["per_variant"]
    L.append(
        f"- LoCoMo-30 @ K=50: v2f={loc['v2f']['r@50']:.4f} "
        f"ens_2={loc['ens_2']['r@50']:.4f} "
        f"crit={loc['crit_only']['r@50']:.4f} "
        f"ens_2+crit={loc['ens_2_plus_crit']['r@50']:.4f} "
        f"router_ens={loc['router_ens']['r@50']:.4f} "
        f"router_ens+crit={loc['router_ens_plus_crit']['r@50']:.4f}\n")
    # Synthetic headline
    syn = all_results["synthetic_19q"]["per_variant"]
    L.append(
        f"- synthetic-19 @ K=20: v2f={syn['v2f']['r@20']:.4f} "
        f"ens_2={syn['ens_2']['r@20']:.4f} "
        f"crit={syn['crit_only']['r@20']:.4f} "
        f"ens_2+crit={syn['ens_2_plus_crit']['r@20']:.4f}\n")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    client = OpenAI(timeout=60.0)
    generator = CriticalInfoGenerator(
        client=client, prompt_version=CRITICAL_PROMPT_VERSION, max_workers=8,
    )
    embedder = Embedder(client)

    all_results: dict = {}
    for ds in DATASETS:
        res = evaluate_dataset(ds, generator, client, embedder)
        all_results[ds] = res
        # Interim save
        _flush_interim(all_results)

    # Classifier cost
    cost = {
        "n_uncached": generator.n_uncached,
        "n_cached": generator.n_cached,
        "prompt_tokens": generator.total_prompt_tokens,
        "completion_tokens": generator.total_completion_tokens,
    }
    cost["est_usd"] = round(
        cost["prompt_tokens"] * 0.25 / 1e6
        + cost["completion_tokens"] * 2.0 / 1e6,
        6,
    )

    try:
        generator.save()
    except Exception as e:
        print(f"  (warn) generator.save failed: {e}", flush=True)
    try:
        embedder.save()
    except Exception as e:
        print(f"  (warn) embedder.save failed: {e}", flush=True)

    total_elapsed = time.time() - t0

    # Save JSON with per_question pruned (only retain summary fields to keep
    # the file readable)
    def _strip(res: dict) -> dict:
        out = {k: v for k, v in res.items()}
        per_q = out.get("per_question", [])
        pruned = []
        for r in per_q:
            pruned.append({
                "dataset": r["dataset"],
                "conversation_id": r["conversation_id"],
                "question_index": r["question_index"],
                "category": r["category"],
                "num_source_turns": r["num_source_turns"],
                "router_label": r["router_label"],
                "recall": r["recall"],
            })
        out["per_question"] = pruned
        return out

    json_path = RESULTS_DIR / "composition_study.json"
    with open(json_path, "w") as f:
        json.dump({
            "prompt_version": CRITICAL_PROMPT_VERSION,
            "elapsed_s": round(total_elapsed, 2),
            "classifier_cost": cost,
            "results": {ds: _strip(res) for ds, res in all_results.items()},
        }, f, indent=2, default=str)
    print(f"\nWrote {json_path}", flush=True)

    md = render_markdown(all_results, cost, total_elapsed)
    md_path = RESULTS_DIR / "composition_study.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Wrote {md_path}", flush=True)

    # Console summary
    print("\n" + "=" * 70)
    print("COMPOSITION STUDY — r@50 overall")
    print("=" * 70)
    print(f"{'variant':28s} " + " ".join(f"{ds:>14s}" for ds in DATASETS))
    for var in VARIANTS_ORDER:
        row = f"{var:28s} "
        for ds in DATASETS:
            r = all_results[ds]["per_variant"][var]["r@50"]
            row += f"{r:>14.4f} "
        print(row)
    print(f"\nTotal elapsed: {total_elapsed:.0f}s  "
          f"classifier ${cost['est_usd']:.4f}")


def _flush_interim(all_results: dict) -> None:
    """Incremental checkpoint for safety."""
    tmp_path = RESULTS_DIR / "composition_study.interim.json"
    try:
        payload = {"partial_results": {
            ds: {
                "ds_name": res["ds_name"],
                "n_with_gold": res["n_with_gold"],
                "per_variant": res["per_variant"],
                "flag_rate": res["flag_rate"],
                "n_critical_turns": res["n_critical_turns"],
            }
            for ds, res in all_results.items()
        }}
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"  (warn) interim flush failed: {e}", flush=True)


if __name__ == "__main__":
    main()
