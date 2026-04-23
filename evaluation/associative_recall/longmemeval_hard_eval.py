"""LongMemEval HARD-category evaluation.

Run 6 shipped retrieval architectures on the 3 hard categories
(multi-session, single-session-preference, temporal-reasoning) using the
pre-built hard subsample (from longmemeval_hard_setup.py). Report fair-
backfill recall at K=20 and K=50, overall and per-category.

Architectures evaluated:
  - cosine_baseline         (no cues; pure cosine top-K)
  - meta_v2f                (v15 + completeness + anti-question)
  - ens_2_v2f_typeenum      (v2f + type_enumerated, sum_cosine merge)
  - critical_info_store     (v2f + crit-always-top-M overlay)
  - two_speaker_filter      (v2f + role filter when single speaker mentioned)
  - ens_all_plus_crit       (5-specialist ensemble + crit overlay; ref ceiling)

Dedicated caches are reused from prior runs; any new writes go to
bestshot_* caches and critical_info's shared cache namespace.

Outputs
-------
  results/longmemeval_hard_baseline.json
  results/longmemeval_hard_baseline.md

Usage
-----
    uv run python longmemeval_hard_eval.py
"""

from __future__ import annotations

import json
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from best_shot import MetaV2f, BestshotEmbeddingCache, BestshotLLMCache
from critical_info_store import (
    CriticalInfoGenerator,
    CriticalInfoStore,
    classify_turns,
    decisions_to_altkeys,
    merge_always_top_m,
)
from ensemble_retrieval import (
    SpecialistOutput,
    _attach_cosine_scores,
    _dedupe_preserve_order,
    build_specialist,
    merge_sum_cosine,
)
from ingest_regex_eval import Embedder, embed_texts_cached
from two_speaker_filter import TwoSpeakerFilter


# ---------------------------------------------------------------------------
# Dedicated lme-hard caches. Reads from lmehard + bestshot as warm start,
# writes only to lmehard namespace (avoids growing the 1.2GB bestshot cache).
# ---------------------------------------------------------------------------
class LmeHardEmbeddingCache(EmbeddingCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        # Read dedicated lmehard cache first (fast).
        self.cache_file = CACHE_DIR / "lmehard_embedding_cache.json"
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


class LmeHardLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        self.cache_file = CACHE_DIR / "lmehard_llm_cache.json"
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
            except (json.JSONDecodeError, OSError):
                pass
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


def _patch_arch_with_lmehard_caches(arch, emb_cache, llm_cache) -> None:
    """Redirect arch's caches to the shared lmehard caches so every
    specialist shares a single in-memory cache instead of reloading a
    1.2GB bestshot cache each."""
    arch.embedding_cache = emb_cache
    arch.llm_cache = llm_cache

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS_JSON = DATA_DIR / "questions_longmemeval_hard.json"
SEGMENTS_NPZ = "longmemeval_hard_segments.npz"

RESULTS_JSON = RESULTS_DIR / "longmemeval_hard_baseline.json"
RESULTS_MD = RESULTS_DIR / "longmemeval_hard_baseline.md"

BUDGETS = (20, 50)

ARCH_NAMES = (
    "cosine_baseline",
    "meta_v2f",
    "ens_2_v2f_typeenum",
    "critical_info_store",
    "two_speaker_filter",
    "ens_all_plus_crit",
)

CRITICAL_PROMPT_VERSION = "v3"

ENS_5_SPECIALISTS = (
    "v2f",
    "v2f_plus_types",
    "type_enumerated",
    "chain_with_scratchpad",
    "v2f_style_explicit",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill_segments(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    budget: int,
) -> list[Segment]:
    seen: set[int] = set()
    unique: list[Segment] = []
    for s in arch_segments:
        if s.index in seen:
            continue
        unique.append(s)
        seen.add(s.index)
    at_k = unique[:budget]
    idxs = {s.index for s in at_k}
    if len(at_k) < budget:
        for s in cosine_segments:
            if s.index in idxs:
                continue
            at_k.append(s)
            idxs.add(s.index)
            if len(at_k) >= budget:
                break
    return at_k[:budget]


def _main_ranked_with_scores_from_seglist(
    seg_list: list[Segment],
    cosine_segments: list[Segment],
    cosine_scores: list[float],
) -> list[tuple[Segment, float]]:
    seen: set[int] = set()
    ranked: list[tuple[Segment, float]] = []
    EPS = 0.001
    for rank, s in enumerate(seg_list):
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


# ---------------------------------------------------------------------------
# Critical-info store builder
# ---------------------------------------------------------------------------
def build_critical_store(
    store: SegmentStore,
    questions: list[dict],
    generator: CriticalInfoGenerator,
    client: OpenAI,
    embedder: Embedder,
) -> CriticalInfoStore | None:
    conv_ids = {q["conversation_id"] for q in questions}
    target = [s for s in store.segments if s.conversation_id in conv_ids]
    if not target:
        return None
    print(
        f"  classifying {len(target)} turns for critical-info...", flush=True
    )
    t0 = time.time()
    decisions = classify_turns(generator, target, log_every=500)
    n_crit = sum(1 for d in decisions if d.critical)
    print(
        f"  classify done in {time.time() - t0:.1f}s -- crit="
        f"{n_crit}/{len(decisions)}",
        flush=True,
    )
    alt_keys = decisions_to_altkeys(decisions)
    alt_texts = [k.text for k in alt_keys]
    if alt_texts:
        alt_embs = embed_texts_cached(
            client, embedder.embedding_cache, alt_texts
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
    return crit_store


# ---------------------------------------------------------------------------
# Per-question context
# ---------------------------------------------------------------------------
@dataclass
class QContext:
    question: dict
    q_text: str
    conv_id: str
    source_ids: set[int]
    category: str
    query_emb: np.ndarray
    cosine_segments: list[Segment]
    cosine_scores: list[float]
    specialist_outputs: dict[str, SpecialistOutput]
    two_speaker_segments: list[Segment]
    two_speaker_fired: bool
    crit_ranked: list[tuple[int, float, Segment]]


def run_specialists_on_q(
    specialists: dict,
    store: SegmentStore,
    q_text: str,
    conv_id: str,
    query_emb: np.ndarray,
) -> dict[str, SpecialistOutput]:
    out: dict[str, SpecialistOutput] = {}
    for name, arch in specialists.items():
        arch.reset_counters()
        try:
            res = arch.retrieve(q_text, conv_id)
            segs = _dedupe_preserve_order(res.segments)
        except Exception as e:
            print(f"    [warn] {name} retrieve failed: {e}", flush=True)
            segs = []
        scores = _attach_cosine_scores(store, segs, query_emb)
        out[name] = SpecialistOutput(
            name=name, segments=segs, cosine_scores=scores,
            llm_calls=arch.llm_calls,
        )
    return out


# ---------------------------------------------------------------------------
# Architecture evaluators
# ---------------------------------------------------------------------------
def arch_cosine_baseline(ctx: QContext, K: int) -> set[int]:
    return {s.turn_id for s in ctx.cosine_segments[:K]}


def arch_meta_v2f(ctx: QContext, K: int) -> set[int]:
    segs = fair_backfill_segments(
        list(ctx.specialist_outputs["v2f"].segments),
        ctx.cosine_segments,
        K,
    )
    return {s.turn_id for s in segs}


def arch_ens_2_v2f_typeenum(ctx: QContext, K: int) -> set[int]:
    subset = {
        n: ctx.specialist_outputs[n]
        for n in ("v2f", "type_enumerated")
        if n in ctx.specialist_outputs
    }
    ranked = merge_sum_cosine(subset)
    base = [s for s, _ in ranked]
    segs = fair_backfill_segments(base, ctx.cosine_segments, K)
    return {s.turn_id for s in segs}


def arch_critical_info(
    ctx: QContext, K: int, crit_store_ok: bool
) -> set[int]:
    base = list(ctx.specialist_outputs["v2f"].segments)
    if not crit_store_ok or not ctx.crit_ranked:
        segs = fair_backfill_segments(base, ctx.cosine_segments, K)
        return {s.turn_id for s in segs}
    main_ranked = _main_ranked_with_scores_from_seglist(
        base, ctx.cosine_segments, ctx.cosine_scores
    )
    merged_segs = merge_always_top_m(
        main_ranked, ctx.crit_ranked, K, top_m=5, min_score=0.2,
    )
    return {s.turn_id for s in merged_segs}


def arch_two_speaker_filter(ctx: QContext, K: int) -> set[int]:
    base = list(ctx.two_speaker_segments)
    segs = fair_backfill_segments(base, ctx.cosine_segments, K)
    return {s.turn_id for s in segs}


def arch_ens_all_plus_crit(
    ctx: QContext, K: int, crit_store_ok: bool
) -> set[int]:
    subset = {
        n: ctx.specialist_outputs[n]
        for n in ENS_5_SPECIALISTS
        if n in ctx.specialist_outputs
    }
    ranked = merge_sum_cosine(subset)
    base = [s for s, _ in ranked]
    if not crit_store_ok or not ctx.crit_ranked:
        segs = fair_backfill_segments(base, ctx.cosine_segments, K)
        return {s.turn_id for s in segs}
    main_ranked = _main_ranked_with_scores_from_seglist(
        base, ctx.cosine_segments, ctx.cosine_scores
    )
    merged_segs = merge_always_top_m(
        main_ranked, ctx.crit_ranked, K, top_m=5, min_score=0.2,
    )
    return {s.turn_id for s in merged_segs}


ARCH_FUNCS = {
    "cosine_baseline":      (arch_cosine_baseline, False),
    "meta_v2f":             (arch_meta_v2f, False),
    "ens_2_v2f_typeenum":   (arch_ens_2_v2f_typeenum, False),
    "critical_info_store":  (arch_critical_info, True),
    "two_speaker_filter":   (arch_two_speaker_filter, False),
    "ens_all_plus_crit":    (arch_ens_all_plus_crit, True),
}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main() -> None:
    t_all = time.time()
    client = OpenAI(timeout=90.0)

    print(f"Loading questions {QUESTIONS_JSON}", flush=True)
    with open(QUESTIONS_JSON) as f:
        questions = json.load(f)
    print(f"  n={len(questions)}", flush=True)

    print(f"Loading SegmentStore {SEGMENTS_NPZ}", flush=True)
    store = SegmentStore(data_dir=DATA_DIR, npz_name=SEGMENTS_NPZ)
    print(f"  segments={len(store.segments)}", flush=True)

    # Category summary
    cat_counts: dict[str, int] = defaultdict(int)
    turns_per_cat: dict[str, list[int]] = defaultdict(list)
    gold_per_cat: dict[str, list[int]] = defaultdict(list)
    for q in questions:
        cat_counts[q["category"]] += 1
        turns_per_cat[q["category"]].append(q["num_haystack_turns"])
        gold_per_cat[q["category"]].append(q["num_source_turns"])
    print("\nCategory composition:", flush=True)
    for cat in sorted(cat_counts):
        ts = turns_per_cat[cat]
        gs = gold_per_cat[cat]
        print(
            f"  {cat:30s} n={cat_counts[cat]:3d} "
            f"turns: mean={sum(ts)/len(ts):.0f} min={min(ts)} max={max(ts)}  "
            f"gold: mean={sum(gs)/len(gs):.0f} min={min(gs)} max={max(gs)}",
            flush=True,
        )

    # Sanity check: ensure gold source_ids exist in the store.
    seg_by_conv: dict[str, set[int]] = defaultdict(set)
    for s in store.segments:
        seg_by_conv[s.conversation_id].add(s.turn_id)
    missing_any = 0
    for q in questions:
        cid = q["conversation_id"]
        tids = set(q["source_ids"])
        if not tids.issubset(seg_by_conv.get(cid, set())):
            missing_any += 1
    if missing_any:
        print(
            f"  WARNING: {missing_any} questions have gold ids not in store",
            flush=True,
        )
    else:
        print("  sanity: all gold source_ids present in store.", flush=True)

    # Monkey-patch ALL specialist cache classes BEFORE instantiating any
    # arch or Embedder. Otherwise each constructor would re-load the 1.2GB
    # bestshot cache.
    import best_shot as _best_shot
    import ingest_regex_eval as _ingest_regex_eval
    import two_speaker_filter as _two_speaker_filter
    import type_enumerated as _type_enumerated
    import domain_agnostic as _domain_agnostic
    import goal_chain as _goal_chain

    _patched: list[tuple] = []
    def _patch(mod, attr):
        orig = getattr(mod, attr)
        _patched.append((mod, attr, orig))
        setattr(
            mod, attr,
            LmeHardEmbeddingCache if "Embedding" in attr
            else LmeHardLLMCache,
        )

    for mod, attr in (
        (_best_shot, "BestshotEmbeddingCache"),
        (_best_shot, "BestshotLLMCache"),
        (_ingest_regex_eval, "BestshotEmbeddingCache"),
        (_ingest_regex_eval, "BestshotLLMCache"),
        (_two_speaker_filter, "TwoSpeakerEmbeddingCache"),
        (_two_speaker_filter, "TwoSpeakerLLMCache"),
        (_type_enumerated, "TypeEnumEmbeddingCache"),
        (_type_enumerated, "TypeEnumLLMCache"),
        (_domain_agnostic, "DomainAgnosticEmbeddingCache"),
        (_domain_agnostic, "DomainAgnosticLLMCache"),
        (_goal_chain, "GoalChainEmbeddingCache"),
        (_goal_chain, "GoalChainLLMCache"),
    ):
        _patch(mod, attr)

    # Shared lmehard caches — every specialist reuses these in memory.
    print("\nInit lmehard caches...", flush=True)
    emb_cache = LmeHardEmbeddingCache()
    llm_cache = LmeHardLLMCache()
    print(
        f"  emb_cache entries={len(emb_cache._cache)} "
        f"llm_cache entries={len(llm_cache._cache)}",
        flush=True,
    )

    # Build critical-info store.
    generator = CriticalInfoGenerator(
        client=client,
        prompt_version=CRITICAL_PROMPT_VERSION,
        max_workers=24,
        cache=llm_cache,
    )
    embedder = Embedder(client=client)
    embedder.embedding_cache = emb_cache  # type: ignore[assignment]
    embedder.llm_cache = llm_cache  # type: ignore[assignment]

    print("\nBuilding critical-info store...", flush=True)
    t0 = time.time()
    crit_store = build_critical_store(
        store, questions, generator, client, embedder
    )
    crit_store_ok = crit_store is not None
    print(
        f"  crit_store_ok={crit_store_ok} "
        f"({time.time() - t0:.1f}s)", flush=True,
    )

    try:
        print("\nBuilding specialists...", flush=True)
        specialists: dict = {}
        for name in ENS_5_SPECIALISTS:
            t0 = time.time()
            arch = build_specialist(name, store)
            _patch_arch_with_lmehard_caches(arch, emb_cache, llm_cache)
            # Override: build_specialist returns a cache-only arch. That is
            # correct for "run on already-run benchmarks" workflows — but on
            # lme-hard every query is novel, so we need real LLM calls. We
            # rebuild a real llm_call that uses the lmehard cache.
            _model_default = "gpt-5-mini"

            def make_real_llm_call(arch_ref):
                cache = arch_ref.llm_cache
                client_ref = arch_ref.client

                def real_llm_call(prompt: str,
                                  model: str = _model_default) -> str:
                    cached = cache.get(model, prompt)
                    if cached is not None:
                        arch_ref.llm_calls += 1
                        return cached
                    resp = client_ref.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=2000,
                    )
                    text = resp.choices[0].message.content or ""
                    cache.put(model, prompt, text)
                    arch_ref.llm_calls += 1
                    return text
                return real_llm_call

            arch.llm_call = make_real_llm_call(arch)
            specialists[name] = arch
            print(f"    {name}: ready in {time.time() - t0:.1f}s",
                  flush=True)

        print("  Building two_speaker_filter...", flush=True)
        t0 = time.time()
        ts_filter = TwoSpeakerFilter(store, client=client)
        _patch_arch_with_lmehard_caches(ts_filter, emb_cache, llm_cache)
        print(f"    two_speaker_filter: ready in {time.time() - t0:.1f}s",
              flush=True)
    finally:
        for mod, attr, orig in _patched:
            setattr(mod, attr, orig)

    # Evaluate per question
    print("\nEvaluating...", flush=True)
    per_q_rows: list[dict] = []
    two_speaker_fires = 0
    for qi, q in enumerate(questions):
        t_q = time.time()
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_ids"])
        category = q["category"]

        query_emb = specialists["v2f"].embed_text(q_text)
        cos_res = store.search(
            query_emb, top_k=max(BUDGETS), conversation_id=conv_id
        )
        cos_segs = list(cos_res.segments)
        cos_scores = list(cos_res.scores)

        # Specialist outputs (ens_5)
        spec_out = run_specialists_on_q(
            specialists, store, q_text, conv_id, query_emb
        )

        # Two-speaker filter — run full retrieve(); check if transform applied
        ts_result = ts_filter.retrieve(q_text, conv_id)
        ts_segs = list(ts_result.segments)
        ts_fired = bool(
            ts_result.metadata.get("applied_speaker_transform", False)
        )
        if ts_fired:
            two_speaker_fires += 1

        # Critical ranked
        crit_ranked: list[tuple[int, float, Segment]] = []
        if crit_store is not None:
            crit_ranked = crit_store.search_per_parent(
                query_emb, top_m=max(BUDGETS),
                conversation_id=conv_id, min_score=-1.0,
            )

        ctx = QContext(
            question=q, q_text=q_text, conv_id=conv_id,
            source_ids=source_ids, category=category,
            query_emb=query_emb,
            cosine_segments=cos_segs, cosine_scores=cos_scores,
            specialist_outputs=spec_out,
            two_speaker_segments=ts_segs,
            two_speaker_fired=ts_fired,
            crit_ranked=crit_ranked,
        )

        recalls: dict[str, float] = {}
        for arch_name in ARCH_NAMES:
            fn, needs_crit = ARCH_FUNCS[arch_name]
            for K in BUDGETS:
                try:
                    if needs_crit:
                        ids = fn(ctx, K, crit_store_ok)
                    else:
                        ids = fn(ctx, K)
                    r = compute_recall(ids, source_ids) if source_ids else 1.0
                    recalls[f"{arch_name}@{K}"] = round(r, 4)
                except Exception as e:
                    print(
                        f"    [warn] {arch_name} K={K} on {conv_id}: {e}",
                        flush=True,
                    )
                    recalls[f"{arch_name}@{K}"] = 0.0

        per_q_rows.append({
            "question_index": qi,
            "conversation_id": conv_id,
            "category": category,
            "question": q_text[:200],
            "num_source_turns": len(source_ids),
            "two_speaker_fired": ts_fired,
            "recall": recalls,
            "time_s": round(time.time() - t_q, 2),
        })
        if (qi + 1) % 5 == 0:
            print(
                f"  [{qi+1}/{len(questions)}] cat={category} "
                f"v2f@50={recalls.get('meta_v2f@50', 0):.3f} "
                f"ens_all@50={recalls.get('ens_all_plus_crit@50', 0):.3f}",
                flush=True,
            )
        # Save caches periodically — all specialists share the same
        # emb/llm cache so one save() per cache is sufficient.
        if (qi + 1) % 15 == 0:
            try:
                emb_cache.save()
            except Exception:
                pass
            try:
                llm_cache.save()
            except Exception:
                pass

    # Final save
    try:
        emb_cache.save()
    except Exception:
        pass
    try:
        llm_cache.save()
    except Exception:
        pass

    # Aggregate
    def mean_recall(rows: list[dict], key: str) -> float:
        vs = [r["recall"].get(key, 0.0) for r in rows
              if r["num_source_turns"] > 0]
        return round(sum(vs) / len(vs), 4) if vs else 0.0

    overall_agg: dict[str, float] = {}
    for arch_name in ARCH_NAMES:
        for K in BUDGETS:
            overall_agg[f"{arch_name}@{K}"] = mean_recall(
                per_q_rows, f"{arch_name}@{K}"
            )

    per_cat_agg: dict[str, dict[str, float]] = {}
    for cat in sorted(cat_counts):
        cat_rows = [r for r in per_q_rows if r["category"] == cat]
        entry: dict[str, Any] = {"n": len(cat_rows)}
        for arch_name in ARCH_NAMES:
            for K in BUDGETS:
                entry[f"{arch_name}@{K}"] = mean_recall(
                    cat_rows, f"{arch_name}@{K}"
                )
        per_cat_agg[cat] = entry

    total_elapsed = time.time() - t_all

    out = {
        "n_questions": len(questions),
        "categories": list(sorted(cat_counts)),
        "cat_counts": dict(cat_counts),
        "two_speaker_fires": two_speaker_fires,
        "two_speaker_fire_rate": (
            round(two_speaker_fires / len(questions), 4) if questions else 0.0
        ),
        "total_elapsed_s": round(total_elapsed, 1),
        "overall": overall_agg,
        "per_category": per_cat_agg,
        "per_question": per_q_rows,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved raw: {RESULTS_JSON}", flush=True)

    # Markdown report
    md = render_markdown(out, cat_counts, turns_per_cat, gold_per_cat)
    with open(RESULTS_MD, "w") as f:
        f.write(md)
    print(f"Saved markdown: {RESULTS_MD}", flush=True)

    # Headline
    print("\n" + "=" * 80, flush=True)
    print("LONGMEMEVAL HARD — HEADLINE", flush=True)
    print("=" * 80, flush=True)
    for arch_name in ARCH_NAMES:
        print(
            f"  {arch_name:24s} "
            f"r@20={overall_agg[f'{arch_name}@20']:.3f}  "
            f"r@50={overall_agg[f'{arch_name}@50']:.3f}",
            flush=True,
        )
    print(f"\nTwo-speaker fire rate: {two_speaker_fires}/{len(questions)}",
          flush=True)


def render_markdown(
    out: dict,
    cat_counts: dict,
    turns_per_cat: dict,
    gold_per_cat: dict,
) -> str:
    L: list[str] = []
    L.append("# LongMemEval hard categories — baseline architecture comparison\n")
    L.append(
        "Evaluates 6 shipped architectures on the 3 HARD LongMemEval "
        "categories (multi-session, single-session-preference, "
        "temporal-reasoning). Fair-backfill recall at K=20 and K=50. "
        "Per-question conversation scoping.\n"
    )
    L.append(f"Elapsed: {out['total_elapsed_s']:.0f}s  |  "
             f"questions: {out['n_questions']}  |  "
             f"two_speaker_fires: {out['two_speaker_fires']} "
             f"({100 * out['two_speaker_fire_rate']:.1f}%)\n")

    # Sample composition
    L.append("\n## Sample composition\n")
    L.append("| Category | n | mean haystack turns | gold turns (mean/min/max) |")
    L.append("|---|---:|---:|---|")
    for cat in sorted(cat_counts):
        ts = turns_per_cat[cat]
        gs = gold_per_cat[cat]
        L.append(
            f"| {cat} | {cat_counts[cat]} | "
            f"{sum(ts)/len(ts):.0f} | "
            f"{sum(gs)/len(gs):.1f} / {min(gs)} / {max(gs)} |"
        )

    # Overall recall matrix
    L.append("\n## Overall recall matrix (3 hard categories combined)\n")
    L.append("| Architecture | r@20 | r@50 |")
    L.append("|---|---:|---:|")
    for arch_name in ARCH_NAMES:
        r20 = out["overall"][f"{arch_name}@20"]
        r50 = out["overall"][f"{arch_name}@50"]
        L.append(f"| {arch_name} | {r20:.4f} | {r50:.4f} |")

    # Per-category at K=20
    for K in BUDGETS:
        L.append(f"\n## Per-category recall @K={K}\n")
        header = "| Architecture | "
        sep = "|---|"
        for cat in sorted(cat_counts):
            header += f"{cat} | "
            sep += "---:|"
        L.append(header.rstrip())
        L.append(sep)
        for arch_name in ARCH_NAMES:
            row = f"| {arch_name} | "
            for cat in sorted(cat_counts):
                v = out["per_category"][cat][f"{arch_name}@{K}"]
                row += f"{v:.4f} | "
            L.append(row.rstrip())

    # Deltas vs cosine baseline and meta_v2f, per category
    L.append("\n## Δ vs meta_v2f per category @K=50\n")
    header = "| Architecture | "
    sep = "|---|"
    for cat in sorted(cat_counts):
        header += f"Δ {cat} | "
        sep += "---:|"
    L.append(header.rstrip())
    L.append(sep)
    for arch_name in ARCH_NAMES:
        if arch_name == "meta_v2f":
            continue
        row = f"| {arch_name} | "
        for cat in sorted(cat_counts):
            v2f = out["per_category"][cat]["meta_v2f@50"]
            a = out["per_category"][cat][f"{arch_name}@50"]
            row += f"{a - v2f:+.4f} | "
        L.append(row.rstrip())

    # Per-category verdict — generalization analysis
    L.append("\n## Per-category verdict\n")
    for cat in sorted(cat_counts):
        L.append(f"\n### {cat}")
        per = out["per_category"][cat]
        v2f20 = per["meta_v2f@20"]
        v2f50 = per["meta_v2f@50"]
        cos20 = per["cosine_baseline@20"]
        cos50 = per["cosine_baseline@50"]
        ens50 = per["ens_all_plus_crit@50"]
        L.append(
            f"- cosine_baseline: r@20={cos20:.3f}  r@50={cos50:.3f}"
        )
        L.append(
            f"- meta_v2f:        r@20={v2f20:.3f}  r@50={v2f50:.3f}  "
            f"Δ vs cosine: {v2f20 - cos20:+.3f} / {v2f50 - cos50:+.3f}"
        )
        L.append(
            f"- ens_all_plus_crit: r@50={ens50:.3f}  "
            f"Δ vs v2f@50: {ens50 - v2f50:+.3f}"
        )
        best_arch = max(
            ARCH_NAMES, key=lambda a: per[f"{a}@50"]
        )
        L.append(
            f"- best @K=50: **{best_arch}** "
            f"({per[f'{best_arch}@50']:.3f})"
        )

    # Note two_speaker fire rate
    L.append("\n## Two-speaker filter coverage\n")
    L.append(
        f"Two-speaker filter fired on "
        f"{out['two_speaker_fires']}/{out['n_questions']} questions "
        f"({100 * out['two_speaker_fire_rate']:.1f}%). "
        "LongMemEval questions are first-person (\"I\") and do not generally "
        "mention named participants, so this specialist is expected to not "
        "fire; when it does not fire, it falls through to v2f."
    )

    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
