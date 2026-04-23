"""Empirical evaluation of the critical-info separate vector store.

For each turn, an LLM classifier (gpt-5-mini, strict "critical info" prompt)
decides SKIP vs CRITICAL; if CRITICAL, emits 3 short focused alt-keys. Those
alt-keys live in a SEPARATE vector store from the main one. At query time the
main cosine/v2f path is untouched; a secondary critical-pool retrieval is
merged in under one of two strategies:

  - crit_additive_bonus_0.1: union by parent_index, +0.1 bonus on critical
    scores, tie-break toward main.
  - crit_always_top_M: always include top-5 critical hits above min_score=0.2,
    then fill with main top-K.

Baseline: v2f on the main index only.

Runs on LoCoMo-30 (primary) and synthetic_19q (secondary) at K=20 and K=50.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    EMBED_MODEL,
    Segment,
    SegmentStore,
)

# Patch BestshotEmbeddingCache.__init__ to tolerate corrupt JSON cache files
# BEFORE importing anything that uses it (best_shot, ingest_regex_eval).
# This is necessary because concurrent agents may have corrupted a shared
# cache file on disk.
import best_shot as _best_shot_module
_ORIG_BEC_INIT = _best_shot_module.BestshotEmbeddingCache.__init__


def _safe_bec_init(self):
    from associative_recall import CACHE_DIR as _CACHE_DIR
    self.cache_dir = _CACHE_DIR
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    self._cache: dict = {}
    for name in (
        "embedding_cache.json",
        "arch_embedding_cache.json",
        "agent_embedding_cache.json",
        "frontier_embedding_cache.json",
        "meta_embedding_cache.json",
        "optim_embedding_cache.json",
        "synth_test_embedding_cache.json",
        "bestshot_embedding_cache.json",
    ):
        p = self.cache_dir / name
        if not p.exists():
            continue
        try:
            with open(p) as f:
                data = json.load(f)
            self._cache.update(data)
        except Exception:
            # Try raw-decode recovery for corrupted files
            try:
                with open(p, "rb") as f:
                    raw = f.read().decode("utf-8", errors="replace")
                obj, _end = json.JSONDecoder().raw_decode(raw)
                self._cache.update(obj)
                print(f"  (warn) cache file {name} corrupt; recovered "
                      f"{len(obj)} entries via raw_decode", flush=True)
            except Exception as e:
                print(f"  (warn) skipping corrupt cache file {name}: {e}",
                      flush=True)
    self.cache_file = self.cache_dir / "bestshot_embedding_cache.json"
    self._new_entries = {}


_best_shot_module.BestshotEmbeddingCache.__init__ = _safe_bec_init  # type: ignore[method-assign]

# Also wrap .save() to use a PID-unique tmp file to avoid concurrent
# replace-race with other agents.
_ORIG_BEC_SAVE = _best_shot_module.BestshotEmbeddingCache.save


def _safe_bec_save(self):
    if not self._new_entries:
        return
    try:
        existing = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except Exception:
                # try raw_decode
                try:
                    with open(self.cache_file, "rb") as f:
                        raw = f.read().decode("utf-8", errors="replace")
                    existing, _ = json.JSONDecoder().raw_decode(raw)
                except Exception:
                    existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(f".json.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
    except Exception as e:
        print(f"  (warn) BEC save race: {e}", flush=True)


_best_shot_module.BestshotEmbeddingCache.save = _safe_bec_save  # type: ignore[method-assign]

# os is needed for pid
import os

from best_shot import (
    BestshotEmbeddingCache,
    BestshotLLMCache,
    MetaV2f,
)

from critical_info_store import (
    CriticalAltKey,
    CriticalInfoGenerator,
    CriticalInfoStore,
    CriticalTurnDecision,
    classify_turns,
    decisions_to_altkeys,
    merge_additive_bonus,
    merge_always_top_m,
)
from ingest_regex_eval import (
    Embedder,
    embed_texts_cached,
    compute_recall,
    fair_backfill_turn_ids,
    BUDGETS,
)


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DATASETS = {
    "locomo_30q": {
        "npz": "segments_extended.npz",
        "questions": "questions_extended.json",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "max_questions": 30,
        "conv_prefix": "locomo_",
    },
    "synthetic_19q": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "filter": None,
        "max_questions": None,
        "conv_prefix": None,
    },
}


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------
def _safe_save_caches(arch) -> None:
    """Wrap arch.save_caches() to tolerate concurrent cache writers."""
    try:
        arch.save_caches()
    except Exception as e:
        print(f"  (warn) arch.save_caches failed: {e}", flush=True)


def run_baseline_v2f(
    store: SegmentStore,
    embedder: Embedder,
    questions: list[dict],
) -> list[dict]:
    """Baseline v2f fair-backfill recall on main index."""
    arch = MetaV2f(store)
    arch.embedding_cache = embedder.embedding_cache
    arch.llm_cache = embedder.llm_cache

    out: list[dict] = []
    for i, q in enumerate(questions):
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])

        arch.reset_counters()
        result = arch.retrieve(q_text, conv_id)
        arch_segs = list(result.segments)

        q_emb = arch.embed_text(q_text)
        max_K = max(BUDGETS)
        cos_res = store.search(q_emb, top_k=max_K, conversation_id=conv_id)
        cos_segs = list(cos_res.segments)

        row = {
            "conversation_id": conv_id,
            "category": q.get("category", "unknown"),
            "question_index": q.get("question_index", -1),
            "question": q_text,
            "source_chat_ids": sorted(source_ids),
            "llm_calls": arch.llm_calls,
            "embed_calls": arch.embed_calls,
        }
        for K in BUDGETS:
            ids = fair_backfill_turn_ids(arch_segs, cos_segs, K)
            row[f"r@{K}"] = compute_recall(ids, source_ids)
            row[f"retrieved_ids@{K}"] = sorted(ids)
        out.append(row)
        if (i + 1) % 5 == 0:
            _safe_save_caches(arch)
    _safe_save_caches(arch)
    return out


def v2f_fair_backfill_ranked(
    arch,
    store: SegmentStore,
    q: dict,
) -> tuple[
    list[tuple[Segment, float]],  # baseline main_ranked at max_K (segs + scores)
    list[Segment],  # arch segments (pre-backfill)
]:
    """Run v2f and return main-ranked segments with scores for the merge
    strategies. The returned main_ranked list mirrors the fair-backfill
    ordering: arch segments first (in their arch-retrieved order, scored
    arbitrarily high & descending), then cosine top-K backfill by score."""
    q_text = q["question"]
    conv_id = q["conversation_id"]

    arch.reset_counters()
    result = arch.retrieve(q_text, conv_id)
    arch_segs = list(result.segments)

    q_emb = arch.embed_text(q_text)
    max_K = max(BUDGETS)
    cos_res = store.search(q_emb, top_k=max_K, conversation_id=conv_id)
    cos_segs = list(cos_res.segments)
    cos_scores = list(cos_res.scores)

    # Build cosine index -> score map for lookup
    cos_score_by_idx = {
        s.index: sc for s, sc in zip(cos_segs, cos_scores)
    }

    # Dedupe arch preserving order; assign scores high-to-low to preserve ordering
    # on ties with cosine. Use 10.0 + epsilon decreasing to make sure arch wins
    # when comparing scores for ordering purposes.
    seen: set[int] = set()
    main_ranked: list[tuple[Segment, float]] = []
    EPS = 0.001
    for rank, s in enumerate(arch_segs):
        if s.index in seen:
            continue
        # Use arch score proxy that exceeds any cosine score (cosine <= 1.0)
        # so arch-found segments sort first.
        score = 10.0 - rank * EPS
        main_ranked.append((s, score))
        seen.add(s.index)

    # Append cosine segments by score descending (skipping already-seen)
    for s in cos_segs:
        if s.index in seen:
            continue
        sc = cos_score_by_idx.get(s.index, 0.0)
        main_ranked.append((s, sc))
        seen.add(s.index)

    return main_ranked, arch_segs


def run_variant(
    variant: str,
    store: SegmentStore,
    crit_store: CriticalInfoStore,
    embedder: Embedder,
    questions: list[dict],
    top_m_crit: int = 5,
    bonus: float = 0.1,
    min_score: float = 0.2,
) -> list[dict]:
    """Run a critical-info variant against the main index + separate crit store.

    variant in {"crit_additive_bonus_0.1", "crit_always_top_M"}.

    Returns per-question rows with r@K for K in BUDGETS and retrieved_ids.
    Also records critical-contribution counts.
    """
    arch = MetaV2f(store)
    arch.embedding_cache = embedder.embedding_cache
    arch.llm_cache = embedder.llm_cache

    out: list[dict] = []
    for i, q in enumerate(questions):
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])

        main_ranked, _arch_segs = v2f_fair_backfill_ranked(arch, store, q)

        # Critical-pool retrieval (conversation-scoped)
        q_emb = arch.embed_text(q["question"])
        crit_ranked = crit_store.search_per_parent(
            q_emb,
            top_m=max(top_m_crit, max(BUDGETS)),
            conversation_id=conv_id,
            min_score=-1.0,  # collect all; min_score filtering done at merge
        )

        row = {
            "conversation_id": conv_id,
            "category": q.get("category", "unknown"),
            "question_index": q.get("question_index", -1),
            "question": q["question"],
            "source_chat_ids": sorted(source_ids),
        }

        # For each K, compute the merged list
        for K in BUDGETS:
            if variant == "crit_additive_bonus_0.1":
                merged = merge_additive_bonus(
                    main_ranked, crit_ranked, K, bonus=bonus,
                )
            elif variant == "crit_always_top_M":
                merged = merge_always_top_m(
                    main_ranked, crit_ranked, K,
                    top_m=top_m_crit, min_score=min_score,
                )
            else:
                raise ValueError(f"unknown variant: {variant}")

            retrieved_ids = {s.turn_id for s in merged}
            row[f"r@{K}"] = compute_recall(retrieved_ids, source_ids)
            row[f"retrieved_ids@{K}"] = sorted(retrieved_ids)

            # Critical contribution: did any gold come from crit-only (not in
            # main top-K)?
            main_topK_ids = {s.turn_id for s, _ in main_ranked[:K]}
            crit_gold = retrieved_ids & source_ids - main_topK_ids
            row[f"crit_gold_contrib@{K}"] = sorted(crit_gold)

        out.append(row)
        if (i + 1) % 5 == 0:
            _safe_save_caches(arch)
    _safe_save_caches(arch)
    return out


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------
def summarize(per_q: list[dict]) -> dict:
    n = len(per_q)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for K in BUDGETS:
        vals = [r[f"r@{K}"] for r in per_q]
        out[f"mean_r@{K}"] = round(sum(vals) / n, 4) if vals else 0.0
    return out


def summarize_by_category(per_q: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in per_q:
        by_cat[r.get("category", "unknown")].append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry = {"n": n}
        for K in BUDGETS:
            vals = [r[f"r@{K}"] for r in rs]
            entry[f"mean_r@{K}"] = round(sum(vals) / n, 4) if n else 0.0
        out[cat] = entry
    return out


def critical_contribution(per_q: list[dict], K: int = 50) -> dict:
    """Fraction of questions where critical store surfaced at least one gold
    turn that wasn't in main top-K. Also compute share of gold surfaced via
    critical over all gold."""
    n = len(per_q)
    if n == 0:
        return {"n": 0}
    q_with_crit = 0
    total_gold = 0
    crit_gold = 0
    for r in per_q:
        gold = set(r["source_chat_ids"])
        contrib = set(r.get(f"crit_gold_contrib@{K}", []))
        total_gold += len(gold)
        crit_gold += len(contrib & gold)
        if contrib & gold:
            q_with_crit += 1
    return {
        "n": n,
        "K": K,
        "frac_questions_with_crit_gold": round(q_with_crit / n, 4),
        "frac_gold_via_crit": round(crit_gold / max(total_gold, 1), 4),
        "total_gold": total_gold,
        "crit_gold": crit_gold,
    }


def false_positive_rate(
    crit_store: CriticalInfoStore,
    embedder: Embedder,
    questions: list[dict],
    top_m: int = 5,
    min_score: float = 0.2,
) -> dict:
    """For each question, look at top-M critical hits (above min_score); how
    many are non-gold?"""
    n_hits = 0
    n_non_gold = 0
    for q in questions:
        q_emb = embedder.embed_text(q["question"])
        crit = crit_store.search_per_parent(
            q_emb,
            top_m=top_m,
            conversation_id=q["conversation_id"],
            min_score=min_score,
        )
        gold = set(q["source_chat_ids"])
        for parent_idx, _sc, seg in crit:
            n_hits += 1
            if seg.turn_id not in gold:
                n_non_gold += 1
    return {
        "n_hits": n_hits,
        "n_non_gold": n_non_gold,
        "fp_rate": round(n_non_gold / n_hits, 4) if n_hits else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_dataset(
    ds_name: str,
    generator: CriticalInfoGenerator,
    client: OpenAI,
    embedder: Embedder,
    shared_decisions: list[CriticalTurnDecision] | None = None,
    shared_crit_store: CriticalInfoStore | None = None,
    shared_alt_keys: list[CriticalAltKey] | None = None,
) -> dict:
    cfg = DATASETS[ds_name]
    print(f"\n{'=' * 70}\nDataset: {ds_name}\n{'=' * 70}", flush=True)
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        all_qs = json.load(f)
    qs = all_qs
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    if cfg["max_questions"]:
        qs = qs[: cfg["max_questions"]]
    print(f"  questions: {len(qs)} | segments: {len(store.segments)}", flush=True)

    # Subset segments to conversations that appear in the questions
    conv_ids_in_qs = {q["conversation_id"] for q in qs}
    target_segments = [s for s in store.segments if s.conversation_id in conv_ids_in_qs]
    print(f"  target segments (filtered to question convs): {len(target_segments)}",
          flush=True)

    # Classify
    if shared_decisions is not None and shared_crit_store is not None:
        decisions = shared_decisions
        crit_store = shared_crit_store
        alt_keys = shared_alt_keys or []
    else:
        print(f"  classifying {len(target_segments)} turns ...", flush=True)
        decisions = classify_turns(generator, target_segments)
        alt_keys = decisions_to_altkeys(decisions)
        alt_texts = [k.text for k in alt_keys]
        if alt_texts:
            print(f"  embedding {len(alt_texts)} alt-key texts ...", flush=True)
        # Wrap embed_texts_cached to tolerate concurrent cache save failures
        _orig_save = embedder.embedding_cache.save
        def _safe_save():
            try:
                return _orig_save()
            except Exception as e:
                print(f"  (warn) embedding_cache.save failed: {e}", flush=True)
        embedder.embedding_cache.save = _safe_save  # type: ignore[method-assign]
        try:
            alt_embs = embed_texts_cached(
                client, embedder.embedding_cache, alt_texts,
            )
        finally:
            embedder.embedding_cache.save = _orig_save  # type: ignore[method-assign]
        try:
            embedder.save()
        except Exception as e:
            print(f"  (warn) embedder.save failed: {e}", flush=True)
        crit_store = CriticalInfoStore(store, alt_keys, alt_embs)

    n_turns = len(decisions)
    n_crit = sum(1 for d in decisions if d.critical)
    n_alts_raw = sum(len(d.alt_keys) for d in decisions)
    flag_rate = n_crit / max(n_turns, 1)
    print(
        f"  classified: n={n_turns} critical={n_crit} ({flag_rate*100:.1f}%) "
        f"alts_raw={n_alts_raw} alts_dedup={len(alt_keys)}",
        flush=True,
    )

    # Baseline v2f
    print("  [1/3] baseline v2f ...", flush=True)
    baseline_rows = run_baseline_v2f(store, embedder, qs)
    baseline_summary = summarize(baseline_rows)
    baseline_by_cat = summarize_by_category(baseline_rows)

    # Variant 1
    print("  [2/3] crit_additive_bonus_0.1 ...", flush=True)
    variant_a_rows = run_variant(
        "crit_additive_bonus_0.1", store, crit_store, embedder, qs,
        bonus=0.1,
    )
    variant_a_summary = summarize(variant_a_rows)
    variant_a_by_cat = summarize_by_category(variant_a_rows)

    # Variant 2
    print("  [3/3] crit_always_top_M ...", flush=True)
    variant_b_rows = run_variant(
        "crit_always_top_M", store, crit_store, embedder, qs,
        top_m_crit=5, min_score=0.2,
    )
    variant_b_summary = summarize(variant_b_rows)
    variant_b_by_cat = summarize_by_category(variant_b_rows)

    crit_contrib_a_20 = critical_contribution(variant_a_rows, K=20)
    crit_contrib_a_50 = critical_contribution(variant_a_rows, K=50)
    crit_contrib_b_20 = critical_contribution(variant_b_rows, K=20)
    crit_contrib_b_50 = critical_contribution(variant_b_rows, K=50)

    fp = false_positive_rate(crit_store, embedder, qs, top_m=5, min_score=0.2)

    return {
        "ds_name": ds_name,
        "n_questions": len(qs),
        "n_target_segments": len(target_segments),
        "n_critical_turns": n_crit,
        "flag_rate": flag_rate,
        "n_altkeys_dedup": len(alt_keys),
        "baseline": {"summary": baseline_summary, "by_category": baseline_by_cat,
                     "per_question": baseline_rows},
        "crit_additive_bonus_0.1": {
            "summary": variant_a_summary, "by_category": variant_a_by_cat,
            "per_question": variant_a_rows,
            "crit_contribution_at_20": crit_contrib_a_20,
            "crit_contribution_at_50": crit_contrib_a_50,
        },
        "crit_always_top_M": {
            "summary": variant_b_summary, "by_category": variant_b_by_cat,
            "per_question": variant_b_rows,
            "crit_contribution_at_20": crit_contrib_b_20,
            "crit_contribution_at_50": crit_contrib_b_50,
        },
        "false_positive": fp,
    }


def render_markdown(results: dict, cost: dict, samples: list[dict]) -> str:
    L: list[str] = []
    L.append("# Critical-Info Separate Vector Store — Empirical Recall Test")
    L.append("")
    L.append(
        "A small subset of high-stakes turns (medications, allergies, key "
        "commitments, family facts, etc.) is flagged at ingestion time by an "
        "LLM classifier (gpt-5-mini, strict prompt). CRITICAL turns emit 3 "
        "short alt-keys, all pointing back to the original turn. Those alt-"
        "keys live in a SEPARATE vector store. At query time, the main v2f "
        "path is unchanged; a secondary critical-pool retrieval is merged "
        "under one of two strategies."
    )
    L.append("")
    L.append(
        "Distinct from prior alt-key tests (regex / LLM alt-keys): those "
        "merged alt-keys into the main pool and pushed v2f's clean retrievals "
        "out. Here the critical pool is disjoint; main retrieval is untouched."
    )
    L.append("")

    # Flagging stats
    L.append("## 1. Flagging statistics")
    L.append("")
    L.append("| dataset | turns | critical | flag rate | alt-keys (dedup) |")
    L.append("|---|---:|---:|---:|---:|")
    for ds_name, res in results.items():
        L.append(
            f"| {ds_name} | {res['n_target_segments']} "
            f"| {res['n_critical_turns']} | {res['flag_rate']*100:.1f}% "
            f"| {res['n_altkeys_dedup']} |"
        )
    L.append("")

    # Recall table
    L.append("## 2. Recall")
    L.append("")
    L.append(
        "Fair-backfill recall. Baseline = v2f on main index only. Variants "
        "merge a separate critical-pool retrieval with the main pool."
    )
    L.append("")
    L.append(
        "| dataset | K | baseline v2f | +crit_additive_0.1 | Δ | "
        "+crit_always_top_M | Δ |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for ds_name, res in results.items():
        for K in BUDGETS:
            b = res["baseline"]["summary"][f"mean_r@{K}"]
            va = res["crit_additive_bonus_0.1"]["summary"][f"mean_r@{K}"]
            vb = res["crit_always_top_M"]["summary"][f"mean_r@{K}"]
            L.append(
                f"| {ds_name} | {K} | {b:.4f} | {va:.4f} | {va-b:+.4f} "
                f"| {vb:.4f} | {vb-b:+.4f} |"
            )
    L.append("")

    # Per-category (LoCoMo only)
    if "locomo_30q" in results:
        L.append("## 3. Per-category (LoCoMo-30)")
        L.append("")
        res = results["locomo_30q"]
        cats = sorted(res["baseline"]["by_category"].keys())
        L.append("| category | n | base @20 | var_a @20 | var_b @20 | "
                 "base @50 | var_a @50 | var_b @50 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for cat in cats:
            bc = res["baseline"]["by_category"][cat]
            va = res["crit_additive_bonus_0.1"]["by_category"][cat]
            vb = res["crit_always_top_M"]["by_category"][cat]
            L.append(
                f"| {cat} | {bc['n']} "
                f"| {bc.get('mean_r@20', 0):.3f} | {va.get('mean_r@20', 0):.3f} "
                f"| {vb.get('mean_r@20', 0):.3f} "
                f"| {bc.get('mean_r@50', 0):.3f} | {va.get('mean_r@50', 0):.3f} "
                f"| {vb.get('mean_r@50', 0):.3f} |"
            )
        L.append("")

    # Critical contribution rate
    L.append("## 4. Critical-contribution rate")
    L.append("")
    L.append(
        "Fraction of gold surfaced via the critical store (i.e., gold that "
        "appears in the merged list but NOT in main top-K)."
    )
    L.append("")
    L.append(
        "| dataset | variant | K | frac questions with crit-gold | "
        "frac gold via crit |"
    )
    L.append("|---|---|---:|---:|---:|")
    for ds_name, res in results.items():
        for var in ["crit_additive_bonus_0.1", "crit_always_top_M"]:
            for K in BUDGETS:
                cc = res[var][f"crit_contribution_at_{K}"]
                L.append(
                    f"| {ds_name} | {var} | {K} "
                    f"| {cc['frac_questions_with_crit_gold']*100:.1f}% "
                    f"| {cc['frac_gold_via_crit']*100:.1f}% |"
                )
    L.append("")

    # FP rate
    L.append("## 5. False-positive rate (critical top-M hits that are not gold)")
    L.append("")
    L.append("| dataset | hits | non-gold | FP rate |")
    L.append("|---|---:|---:|---:|")
    for ds_name, res in results.items():
        fp = res["false_positive"]
        L.append(
            f"| {ds_name} | {fp['n_hits']} | {fp['n_non_gold']} | "
            f"{fp['fp_rate']*100:.1f}% |"
        )
    L.append("")

    # Cost
    L.append("## 6. Cost")
    L.append("")
    L.append(f"- LLM classification calls: uncached={cost['n_uncached']} "
             f"cached={cost['n_cached']}")
    L.append(f"- Input tokens: {cost['prompt_tokens']}")
    L.append(f"- Output tokens: {cost['completion_tokens']}")
    L.append(f"- Est. cost (gpt-5-mini): ${cost['est_usd']:.3f}")
    L.append("")

    # Samples
    L.append("## 7. Sample critical turns (first 10)")
    L.append("")
    for s in samples[:10]:
        flag = "CRITICAL" if s["critical"] else "SKIP"
        L.append(f"- [{flag}] ({s['role']}, turn {s['turn_id']}): "
                 f"{s['text'][:120]}")
        for alt in s["alt_keys"][:3]:
            L.append(f"    - ALT: {alt}")
    L.append("")

    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="v2", choices=["v1", "v2", "v3"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--datasets", default="locomo_30q,synthetic_19q",
        help="comma-separated list",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    client = OpenAI(timeout=60.0)
    generator = CriticalInfoGenerator(
        client=client, prompt_version=args.prompt, max_workers=args.workers,
    )
    embedder = Embedder(client)

    # We run each dataset independently (each has its own NPZ).
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results: dict = {}
    all_samples: list[dict] = []

    for ds_name in datasets:
        res = run_dataset(ds_name, generator, client, embedder)
        all_results[ds_name] = res

        # Pull a few samples (first dataset only — 30 random)
        if ds_name == "locomo_30q":
            # Reload decisions to build the samples list; do it lazily by
            # rerunning classify_turns under a cache-hot path (cheap).
            # Actually, the decisions aren't kept on res; rebuild by calling
            # classify_turns again with a limit — but it's cache hot.
            cfg = DATASETS[ds_name]
            store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
            with open(DATA_DIR / cfg["questions"]) as f:
                all_qs = json.load(f)
            qs = [q for q in all_qs if q.get("benchmark") == "locomo"][:30]
            conv_ids_in_qs = {q["conversation_id"] for q in qs}
            segs = [s for s in store.segments
                    if s.conversation_id in conv_ids_in_qs]
            decisions = classify_turns(generator, segs)
            rng = random.Random(7)
            crit_decisions = [d for d in decisions if d.critical]
            skip_decisions = [d for d in decisions if not d.critical]
            sample_ids = rng.sample(
                range(len(crit_decisions)),
                min(20, len(crit_decisions)),
            )
            for si in sample_ids:
                d = crit_decisions[si]
                all_samples.append({
                    "conversation_id": d.conversation_id,
                    "turn_id": d.turn_id,
                    "role": d.role,
                    "text": d.text,
                    "critical": d.critical,
                    "alt_keys": d.alt_keys,
                    "raw_response": d.raw_response,
                })
            # Also 10 SKIP samples for qualitative inspection
            skip_ids = rng.sample(
                range(len(skip_decisions)),
                min(10, len(skip_decisions)),
            )
            for si in skip_ids:
                d = skip_decisions[si]
                all_samples.append({
                    "conversation_id": d.conversation_id,
                    "turn_id": d.turn_id,
                    "role": d.role,
                    "text": d.text,
                    "critical": d.critical,
                    "alt_keys": d.alt_keys,
                    "raw_response": d.raw_response,
                })

    # Cost
    cost = {
        "n_uncached": generator.n_uncached,
        "n_cached": generator.n_cached,
        "prompt_tokens": generator.total_prompt_tokens,
        "completion_tokens": generator.total_completion_tokens,
    }
    cost["est_usd"] = round(
        cost["prompt_tokens"] * 0.25 / 1e6
        + cost["completion_tokens"] * 2.0 / 1e6,
        4,
    )

    try:
        generator.save()
    except Exception as e:
        print(f"  (warn) generator.save failed: {e}", flush=True)
    try:
        embedder.save()
    except Exception as e:
        print(f"  (warn) embedder.save failed: {e}", flush=True)

    # Write outputs
    md = render_markdown(all_results, cost, all_samples)
    md_path = RESULTS_DIR / "critical_info_store.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\nWrote {md_path}", flush=True)

    json_path = RESULTS_DIR / "critical_info_store.json"
    # Strip large per_question payloads to keep JSON reasonable but keep enough
    # for re-analysis.
    def strip_large(res: dict) -> dict:
        out = {k: v for k, v in res.items()}
        for key in [
            "baseline", "crit_additive_bonus_0.1", "crit_always_top_M",
        ]:
            if key in out and isinstance(out[key], dict):
                # strip retrieved_ids lists which are huge
                per_q = out[key].get("per_question", [])
                pruned = []
                for r in per_q:
                    pruned.append({
                        k: v for k, v in r.items()
                        if not k.startswith("retrieved_ids")
                    })
                out[key] = {**out[key], "per_question": pruned}
        return out

    with open(json_path, "w") as f:
        json.dump(
            {
                "prompt_version": args.prompt,
                "elapsed_s": round(time.time() - t0, 2),
                "cost": cost,
                "results": {
                    ds: strip_large(res) for ds, res in all_results.items()
                },
            },
            f, indent=2, default=str,
        )
    print(f"Wrote {json_path}", flush=True)

    samples_path = RESULTS_DIR / "critical_info_samples.json"
    with open(samples_path, "w") as f:
        json.dump({"prompt_version": args.prompt, "samples": all_samples},
                  f, indent=2)
    print(f"Wrote {samples_path}", flush=True)

    # Console summary
    print("\n" + "=" * 70)
    print("CRITICAL-INFO STORE RESULTS")
    print("=" * 70)
    for ds_name, res in all_results.items():
        print(f"\n{ds_name}:")
        print(f"  flag_rate={res['flag_rate']*100:.1f}%  "
              f"n_crit={res['n_critical_turns']}/{res['n_target_segments']}  "
              f"alt-keys={res['n_altkeys_dedup']}")
        for K in BUDGETS:
            b = res["baseline"]["summary"][f"mean_r@{K}"]
            va = res["crit_additive_bonus_0.1"]["summary"][f"mean_r@{K}"]
            vb = res["crit_always_top_M"]["summary"][f"mean_r@{K}"]
            print(f"  K={K}: baseline={b:.4f}  additive={va:.4f} "
                  f"(Δ{va-b:+.4f})  always_top_M={vb:.4f} (Δ{vb-b:+.4f})")
    print(f"\nLLM cost: ~${cost['est_usd']:.3f}")
    print(f"Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
