"""Two-speaker base + narrow supplements composition test.

Question: does stacking the narrow supplements (ens_2 / critical / alias /
context / clause) on top of **two_speaker_filter** (instead of the single-sided
speaker_user_filter used in composition_v2_all) push LoCoMo K=50 past the
0.917 ceiling set by composition_v2_all?

Background (from this session):
  - two_speaker_filter alone (no comp): LoCoMo K=20=0.892 (+13.6pp),
    K=50=0.892 (+3.4pp). Zero per-query LLM beyond v2f.
  - composition_v2_all (speaker_user_filter + ens_2 + critical + alias +
    context + clause): LoCoMo K=50 = 0.917.
  - At K=20, two_speaker_filter alone BEATS full composition on LoCoMo
    (0.892 > 0.847). At K=50 it lags composition's 0.917 (0.892 < 0.917).
  - Question: drop the single-sided speaker filter out of composition_v2_all
    and swap in two_speaker_filter. Does the two-sided coverage + full
    supplements break 0.917?

Variants evaluated (each at K=20 and K=50):
  1. v2f                         — reference
  2. two_speaker_alone           — two_speaker_filter, no composition
  3. two_speaker_plus_ens2       — two_speaker base + ens_2 (sum_cosine merge)
  4. two_speaker_plus_critical   — two_speaker + critical always_top_M
  5. two_speaker_plus_alias      — two_speaker + alias_expand_v2f
  6. two_speaker_plus_context    — two_speaker + context-emb window_1
  7. two_speaker_all_supplements — ens_2 + alias + clause + context + critical

Apply order for two_speaker_all_supplements:
  1. If query mentions exactly one side, base = two_speaker_filter.retrieve()
     output (filter role-matched turns; fill with unfiltered cosine when
     needed). Otherwise base = v2f output.
  2. Merge ens_2 (v2f + type_enumerated sum_cosine) into base as stack-
     concatenation (base priority preserved via stack_merge order).
  3. Stack-append alias_expand_v2f hits (novel items only).
  4. Stack-append clause_plus_v2f hits (novel items only, if multi-clause).
  5. Stack-append context-emb window_1 hits (novel items only).
  6. Apply critical-info always_top_M overlay (bonus crits to head, capped
     at top_m=5 above min_score=0.2).

Datasets: LoCoMo-30 primary, synthetic-19 sanity check.

Caching:
  - Dedicated compv2_*_cache.json READ-reuse for cue caches (written by
    composition_v2 runs; we do NOT write to those).
  - Dedicated tspcomp_embedding_cache.json / tspcomp_llm_cache.json for
    any writes.
  - Shared reads from all known *_llm_cache.json / *_embedding_cache.json
    so warm caches are respected.

Usage:
    uv run python two_speaker_composition.py
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    Segment,
    SegmentStore,
)
from best_shot import MetaV2f
from critical_info_store import (
    CriticalInfoGenerator,
    CriticalInfoStore,
    classify_turns,
    decisions_to_altkeys,
    merge_always_top_m,
)
from domain_agnostic import (
    DomainAgnosticVariant,
    V2F_STYLE_EXPLICIT_PROMPT,
    NEUTRAL_HEADER,
)
from ensemble_retrieval import (
    SpecialistOutput,
    _attach_cosine_scores,
    _dedupe_preserve_order,
)
from goal_chain import GoalChainRetriever
from ingest_regex_eval import (
    Embedder,
    embed_texts_cached,
    compute_recall,
    fair_backfill_turn_ids,
)
from type_enumerated import TypeEnumeratedVariant, V2fPlusTypesVariant

# Narrow wins
from alias_expansion import (
    AliasExpandV2fFull,
    find_alias_matches,
    _ALIAS_GROUPS_FILE,
)
from clause_decomposition import ClausePlusV2f, split_query_into_clauses
from context_embedding import ContextEmbW1Stacked
from speaker_attributed import (
    SpeakerUserFilter,
    extract_name_mentions,
    _CONV_SPEAKERS_FILE,
)
from two_speaker_filter import (
    TwoSpeakerFilter,
    _CONV_TWO_SPEAKERS_FILE,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ---------------------------------------------------------------------------
# Paths / datasets / budgets
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dedicated write-only cache namespace for this evaluator.
CACHE_MY_LLM = CACHE_DIR / "tspcomp_llm_cache.json"
CACHE_MY_EMB = CACHE_DIR / "tspcomp_embedding_cache.json"

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

CRITICAL_PROMPT_VERSION = "v3"


# ---------------------------------------------------------------------------
# Specialist palette
# ---------------------------------------------------------------------------
# Core specialists whose outputs we need per-question.
CORE_SPECIALISTS = (
    "v2f",
    "type_enumerated",                # for ens_2 sum_cosine
    "two_speaker_filter",             # base layer
    "alias_expand_v2f",               # overlay
    "contextemb_w1_stacked",          # overlay
    "clause_plus_v2f",                # overlay
)


def build_specialist(name: str, store: SegmentStore):
    """Build a specialist arch with cache-only LLM proxy (matches
    final_composition_v2 pattern). Any LLM cache miss falls back to a
    DONE cue so the retrieval still runs with zero live LLM cost.
    """
    if name == "v2f":
        arch = MetaV2f(store)
    elif name == "type_enumerated":
        arch = TypeEnumeratedVariant(store)
    elif name == "two_speaker_filter":
        arch = TwoSpeakerFilter(store)
    elif name == "alias_expand_v2f":
        arch = AliasExpandV2fFull(store)
    elif name == "contextemb_w1_stacked":
        arch = ContextEmbW1Stacked(store)
    elif name == "clause_plus_v2f":
        arch = ClausePlusV2f(store)
    else:
        raise KeyError(name)

    original_llm = arch.llm_call

    def cache_only_llm(prompt: str, model: str = "gpt-5-mini") -> str:
        cached = arch.llm_cache.get(model, prompt)
        if cached is not None:
            arch.llm_calls += 1
            return cached
        arch.llm_calls += 1
        return "ACTION: DONE\nREASONING: cache-miss; skipping\n"

    arch.llm_call = cache_only_llm
    return arch


# ---------------------------------------------------------------------------
# Run specialists on one question (cache-only)
# ---------------------------------------------------------------------------
def run_specialists(
    specialists: dict[str, object],
    store: SegmentStore,
    question: str,
    conversation_id: str,
    query_emb: np.ndarray,
) -> dict[str, SpecialistOutput]:
    out: dict[str, SpecialistOutput] = {}
    for name, arch in specialists.items():
        arch.reset_counters()
        try:
            res = arch.retrieve(question, conversation_id)
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
# Two-speaker side detection (mirrors two_speaker_filter.classify_query)
# ---------------------------------------------------------------------------
def load_conv_two_speakers() -> dict[str, dict[str, str]]:
    if not _CONV_TWO_SPEAKERS_FILE.exists():
        return {}
    try:
        with open(_CONV_TWO_SPEAKERS_FILE) as f:
            data = json.load(f)
        pairs = data.get("speakers", {}) or {}
    except (json.JSONDecodeError, OSError):
        pairs = {}
    out: dict[str, dict[str, str]] = {}
    for cid, p in pairs.items():
        if isinstance(p, dict):
            out[cid] = {
                "user": p.get("user", "UNKNOWN") or "UNKNOWN",
                "assistant": p.get("assistant", "UNKNOWN") or "UNKNOWN",
            }
    return out


def query_mentions_one_side(
    question: str,
    conversation_id: str,
    conv_two_speakers: dict[str, dict[str, str]],
) -> str:
    """Return 'user', 'assistant', 'both', or 'none'."""
    pair = conv_two_speakers.get(conversation_id, {})
    u = pair.get("user", "UNKNOWN") or "UNKNOWN"
    a = pair.get("assistant", "UNKNOWN") or "UNKNOWN"
    toks = extract_name_mentions(question)
    tlow = {t.lower() for t in toks}
    hit_u = u != "UNKNOWN" and u.lower() in tlow
    hit_a = a != "UNKNOWN" and a.lower() in tlow
    if hit_u and hit_a:
        return "both"
    if hit_u:
        return "user"
    if hit_a:
        return "assistant"
    return "none"


def load_conv_aliases() -> dict[str, list[list[str]]]:
    if not _ALIAS_GROUPS_FILE.exists():
        return {}
    try:
        with open(_ALIAS_GROUPS_FILE) as f:
            return json.load(f).get("groups", {}) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def query_alias_fires(
    question: str,
    conversation_id: str,
    conv_aliases: dict[str, list[list[str]]],
) -> bool:
    groups = conv_aliases.get(conversation_id, [])
    if not groups:
        return False
    return len(find_alias_matches(question, groups)) > 0


def query_multi_clause(question: str) -> bool:
    clauses = split_query_into_clauses(question, max_clauses=2)
    return len(clauses) >= 2


# ---------------------------------------------------------------------------
# Ranked-list merge helpers
# ---------------------------------------------------------------------------
def fair_backfill_segments(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    budget: int,
) -> list[Segment]:
    seen: set[int] = set()
    unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
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


def stack_merge(specs: list[list[Segment]]) -> list[Segment]:
    seen: set[int] = set()
    out: list[Segment] = []
    for seg_list in specs:
        for s in seg_list:
            if s.index in seen:
                continue
            out.append(s)
            seen.add(s.index)
    return out


def merge_by_sum_cosine(
    ensemble_outputs: dict[str, SpecialistOutput],
    names: tuple[str, ...],
) -> list[tuple[Segment, float]]:
    """ens_2-style sum_cosine merge over named specialists."""
    pool: dict[int, dict] = {}
    for name in names:
        if name not in ensemble_outputs:
            continue
        so = ensemble_outputs[name]
        for rank, (seg, cos) in enumerate(zip(so.segments, so.cosine_scores)):
            entry = pool.setdefault(seg.index, {"segment": seg, "sum": 0.0})
            entry["sum"] += cos
    merged: list[tuple[Segment, float]] = []
    for idx, e in pool.items():
        merged.append((e["segment"], float(e["sum"])))
    merged.sort(key=lambda rc: -rc[1])
    return merged


def _main_ranked_with_scores_from_seglist(
    seg_list: list[Segment],
    cosine_segments: list[Segment],
    cosine_scores: list[float],
) -> list[tuple[Segment, float]]:
    """Build (seg, score) ranked: provided segments at top (decreasing
    placeholder scores 10..), then cosine hits by their score."""
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
    outputs: dict[str, SpecialistOutput]
    matched_side: str            # "user" | "assistant" | "both" | "none"
    speaker_fires: bool          # side in ("user", "assistant")
    alias_fires: bool
    multi_clause: bool
    crit_ranked: list[tuple[int, float, Segment]]


def build_contexts(
    store: SegmentStore,
    specialists: dict,
    questions: list[dict],
    conv_two_speakers: dict[str, dict[str, str]],
    conv_aliases: dict[str, list[list[str]]],
    crit_store: CriticalInfoStore | None,
) -> list[QContext]:
    ctxs: list[QContext] = []
    for q in questions:
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])
        cat = q.get("category", "unknown")

        q_emb = specialists["v2f"].embed_text(q_text)
        cos_res = store.search(
            q_emb, top_k=max(BUDGETS), conversation_id=conv_id
        )
        cos_segs = list(cos_res.segments)
        cos_scores = list(cos_res.scores)

        outputs = run_specialists(specialists, store, q_text, conv_id, q_emb)

        side = query_mentions_one_side(q_text, conv_id, conv_two_speakers)
        speaker_fires = side in ("user", "assistant")
        alias_fires = query_alias_fires(q_text, conv_id, conv_aliases)
        multi_clause = query_multi_clause(q_text)

        crit = []
        if crit_store is not None:
            crit = crit_store.search_per_parent(
                q_emb,
                top_m=max(BUDGETS),
                conversation_id=conv_id,
                min_score=-1.0,
            )

        ctxs.append(QContext(
            question=q,
            q_text=q_text,
            conv_id=conv_id,
            source_ids=source_ids,
            category=cat,
            query_emb=q_emb,
            cosine_segments=cos_segs,
            cosine_scores=cos_scores,
            outputs=outputs,
            matched_side=side,
            speaker_fires=speaker_fires,
            alias_fires=alias_fires,
            multi_clause=multi_clause,
            crit_ranked=crit,
        ))
    return ctxs


# ---------------------------------------------------------------------------
# Base-layer builder: two_speaker_filter when one side is mentioned, else v2f
# ---------------------------------------------------------------------------
def _two_speaker_base_seglist(ctx: QContext) -> list[Segment]:
    """Return the base-layer segment list. If the query mentions exactly one
    side and two_speaker_filter output is available, use that. Else fall back
    to v2f. two_speaker_filter's own logic already handles the 'both' and
    'none' cases by returning v2f output unchanged — but we defensively use
    v2f output directly when not speaker_fires to match expectations."""
    if ctx.speaker_fires:
        ts = ctx.outputs.get("two_speaker_filter")
        if ts and ts.segments:
            return list(ts.segments)
    return list(ctx.outputs["v2f"].segments)


# ---------------------------------------------------------------------------
# Variant builders (each returns set[int] retrieved turn_ids)
# ---------------------------------------------------------------------------
def var_v2f(ctx: QContext, K: int, crit_ok: bool) -> set[int]:
    segs = fair_backfill_segments(
        list(ctx.outputs["v2f"].segments), ctx.cosine_segments, K
    )
    return {s.turn_id for s in segs}


def var_two_speaker_alone(ctx: QContext, K: int, crit_ok: bool) -> set[int]:
    base = _two_speaker_base_seglist(ctx)
    segs = fair_backfill_segments(base, ctx.cosine_segments, K)
    return {s.turn_id for s in segs}


def var_two_speaker_plus_ens2(
    ctx: QContext, K: int, crit_ok: bool
) -> set[int]:
    base = _two_speaker_base_seglist(ctx)
    merged_2 = merge_by_sum_cosine(
        ctx.outputs, ("v2f", "type_enumerated")
    )
    ens_segs = [seg for seg, _ in merged_2]
    combined = stack_merge([base, ens_segs])
    segs = fair_backfill_segments(combined, ctx.cosine_segments, K)
    return {s.turn_id for s in segs}


def var_two_speaker_plus_critical(
    ctx: QContext, K: int, crit_ok: bool
) -> set[int]:
    base = _two_speaker_base_seglist(ctx)
    if not crit_ok or not ctx.crit_ranked:
        segs = fair_backfill_segments(base, ctx.cosine_segments, K)
        return {s.turn_id for s in segs}
    main_ranked = _main_ranked_with_scores_from_seglist(
        base, ctx.cosine_segments, ctx.cosine_scores,
    )
    merged = merge_always_top_m(
        main_ranked, ctx.crit_ranked, K, top_m=5, min_score=0.2,
    )
    return {s.turn_id for s in merged}


def var_two_speaker_plus_alias(
    ctx: QContext, K: int, crit_ok: bool
) -> set[int]:
    base = _two_speaker_base_seglist(ctx)
    alias_so = ctx.outputs.get("alias_expand_v2f")
    alias_segs = list(alias_so.segments) if alias_so else []
    combined = stack_merge([base, alias_segs])
    segs = fair_backfill_segments(combined, ctx.cosine_segments, K)
    return {s.turn_id for s in segs}


def var_two_speaker_plus_context(
    ctx: QContext, K: int, crit_ok: bool
) -> set[int]:
    base = _two_speaker_base_seglist(ctx)
    ctx_so = ctx.outputs.get("contextemb_w1_stacked")
    ctx_segs = list(ctx_so.segments) if ctx_so else []
    combined = stack_merge([base, ctx_segs])
    segs = fair_backfill_segments(combined, ctx.cosine_segments, K)
    return {s.turn_id for s in segs}


def var_two_speaker_all_supplements(
    ctx: QContext, K: int, crit_ok: bool
) -> set[int]:
    """Full composition v2 on top of two_speaker_filter base.

    Apply order:
      1. Base = two_speaker_filter output (when side in user/assistant);
         else v2f.
      2. Stack-append ens_2 (v2f + type_enumerated sum_cosine).
      3. Stack-append alias_expand_v2f hits.
      4. Stack-append clause_plus_v2f hits (if multi-clause).
      5. Stack-append contextemb_w1_stacked hits.
      6. Apply critical-info always_top_M overlay.
    """
    segs_order: list[list[Segment]] = []

    base = _two_speaker_base_seglist(ctx)
    segs_order.append(base)

    merged_2 = merge_by_sum_cosine(
        ctx.outputs, ("v2f", "type_enumerated")
    )
    segs_order.append([seg for seg, _ in merged_2])

    alias_so = ctx.outputs.get("alias_expand_v2f")
    if alias_so and alias_so.segments:
        segs_order.append(list(alias_so.segments))

    if ctx.multi_clause:
        clause_so = ctx.outputs.get("clause_plus_v2f")
        if clause_so and clause_so.segments:
            segs_order.append(list(clause_so.segments))

    ctx_so = ctx.outputs.get("contextemb_w1_stacked")
    if ctx_so and ctx_so.segments:
        segs_order.append(list(ctx_so.segments))

    combined = stack_merge(segs_order)

    if not crit_ok or not ctx.crit_ranked:
        segs = fair_backfill_segments(combined, ctx.cosine_segments, K)
        return {s.turn_id for s in segs}

    main_ranked = _main_ranked_with_scores_from_seglist(
        combined, ctx.cosine_segments, ctx.cosine_scores,
    )
    merged = merge_always_top_m(
        main_ranked, ctx.crit_ranked, K, top_m=5, min_score=0.2,
    )
    return {s.turn_id for s in merged}


# Variant registry
VARIANTS: list[tuple[str, callable]] = [
    ("v2f", var_v2f),
    ("two_speaker_alone", var_two_speaker_alone),
    ("two_speaker_plus_ens2", var_two_speaker_plus_ens2),
    ("two_speaker_plus_critical", var_two_speaker_plus_critical),
    ("two_speaker_plus_alias", var_two_speaker_plus_alias),
    ("two_speaker_plus_context", var_two_speaker_plus_context),
    ("two_speaker_all_supplements", var_two_speaker_all_supplements),
]


# ---------------------------------------------------------------------------
# Critical-info store
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
    print(f"  classifying {len(target)} turns for critical-info...",
          flush=True)
    t_c = time.time()
    decisions = classify_turns(generator, target, log_every=500)
    print(
        f"  classify done in {time.time() - t_c:.1f}s -- "
        f"crit={sum(1 for d in decisions if d.critical)}/{len(decisions)}",
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
# Per-dataset eval
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


def evaluate_dataset(
    ds_name: str,
    client: OpenAI,
    conv_two_speakers: dict[str, dict[str, str]],
    conv_aliases: dict[str, list[list[str]]],
    generator: CriticalInfoGenerator,
    embedder: Embedder,
) -> dict:
    cfg = DATASETS[ds_name]
    print(f"\n{'=' * 70}\n[{ds_name}]\n{'=' * 70}", flush=True)
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    qs = load_questions(ds_name)
    print(f"  questions={len(qs)} segments={len(store.segments)}", flush=True)

    crit_store = build_critical_store(store, qs, generator, client, embedder)
    crit_ok = crit_store is not None

    # Build specialists (ingest-time work on first call is cached to disk).
    t_spec = time.time()
    print("  building specialists (cache-only)...", flush=True)
    specialists: dict = {}
    for name in CORE_SPECIALISTS:
        t0 = time.time()
        specialists[name] = build_specialist(name, store)
        print(f"    {name}: ready in {time.time() - t0:.1f}s", flush=True)
    print(f"  specialists built in {time.time() - t_spec:.1f}s", flush=True)

    # Reload maps after init (two_speaker ID / alias extraction may have run).
    conv_two_speakers = load_conv_two_speakers()
    conv_aliases = load_conv_aliases()

    # Build per-question contexts.
    print("  building per-question contexts...", flush=True)
    t_ctx = time.time()
    ctxs = build_contexts(
        store, specialists, qs, conv_two_speakers, conv_aliases, crit_store,
    )
    print(f"  contexts built in {time.time() - t_ctx:.1f}s", flush=True)

    # Save arch caches (any new ingest-time writes).
    for arch in specialists.values():
        try:
            arch.save_caches()
        except Exception:
            pass

    # Evaluate each variant at each K per question.
    per_q_rows: list[dict] = []
    for ctx in ctxs:
        row: dict = {
            "dataset": ds_name,
            "conversation_id": ctx.conv_id,
            "question_index": ctx.question.get("question_index", -1),
            "category": ctx.category,
            "num_source_turns": len(ctx.source_ids),
            "matched_side": ctx.matched_side,
            "speaker_fires": ctx.speaker_fires,
            "alias_fires": ctx.alias_fires,
            "multi_clause": ctx.multi_clause,
            "recall": {},
        }
        if not ctx.source_ids:
            per_q_rows.append(row)
            continue
        for var_name, fn in VARIANTS:
            for K in BUDGETS:
                ids = fn(ctx, K, crit_ok)
                r = compute_recall(ids, ctx.source_ids)
                row["recall"][f"{var_name}@{K}"] = round(r, 4)
        per_q_rows.append(row)

    # Aggregate.
    def _agg(rows: list[dict], key: str) -> float:
        vals = [r["recall"].get(key, 0.0) for r in rows
                if r["num_source_turns"] > 0]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    agg: dict[str, float] = {}
    for var_name, _ in VARIANTS:
        for K in BUDGETS:
            agg[f"{var_name}@{K}"] = _agg(per_q_rows, f"{var_name}@{K}")

    # Per-category.
    by_cat: dict = defaultdict(lambda: defaultdict(list))
    for r in per_q_rows:
        if r["num_source_turns"] == 0:
            continue
        cat = r["category"]
        for var_name, _ in VARIANTS:
            for K in BUDGETS:
                by_cat[cat][f"{var_name}@{K}"].append(
                    r["recall"].get(f"{var_name}@{K}", 0.0)
                )
    cat_counts: dict[str, int] = defaultdict(int)
    for r in per_q_rows:
        if r["num_source_turns"] == 0:
            continue
        cat_counts[r["category"]] += 1

    per_cat: dict[str, dict] = {}
    for cat, buckets in by_cat.items():
        per_cat[cat] = {"n": cat_counts[cat]}
        for k, vs in buckets.items():
            per_cat[cat][k] = round(sum(vs) / len(vs), 4) if vs else 0.0

    # Per-subset.
    def _subset_agg(filter_fn) -> dict:
        subset = [r for r in per_q_rows
                  if r["num_source_turns"] > 0 and filter_fn(r)]
        if not subset:
            return {"n": 0}
        res: dict = {"n": len(subset)}
        for var_name, _ in VARIANTS:
            for K in BUDGETS:
                vs = [r["recall"].get(f"{var_name}@{K}", 0.0) for r in subset]
                res[f"{var_name}@{K}"] = round(sum(vs) / len(vs), 4)
        return res

    subsets = {
        "speaker_fires": _subset_agg(lambda r: r["speaker_fires"]),
        "matched_user": _subset_agg(lambda r: r["matched_side"] == "user"),
        "matched_assistant": _subset_agg(
            lambda r: r["matched_side"] == "assistant"
        ),
        "matched_both": _subset_agg(lambda r: r["matched_side"] == "both"),
        "matched_none": _subset_agg(lambda r: r["matched_side"] == "none"),
    }

    return {
        "ds_name": ds_name,
        "n_questions": len(qs),
        "n_with_gold": sum(1 for r in per_q_rows if r["num_source_turns"] > 0),
        "aggregated": agg,
        "per_category": per_cat,
        "subset_aggregates": subsets,
        "per_question": per_q_rows,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------
def render_markdown(all_results: dict, total_elapsed: float) -> str:
    L: list[str] = []
    L.append("# two_speaker_filter base + narrow supplements composition\n")
    L.append(
        "Swaps the single-sided `speaker_user_filter` base of "
        "`composition_v2_all` for the two-sided `two_speaker_filter` and "
        "measures whether the full composition breaks past the prior "
        "LoCoMo K=50 = 0.917 ceiling.\n"
    )
    L.append(f"\nElapsed: {total_elapsed:.0f}s.\n")

    datasets = list(DATASETS.keys())

    for K in BUDGETS:
        L.append(f"\n## Recall matrix (r@{K})\n")
        L.append("| Variant | " + " | ".join(datasets) + " | overall |")
        L.append("|---|" + "---|" * (len(datasets) + 1))
        for var_name, _ in VARIANTS:
            row = [var_name]
            vals_w = []
            wt = 0
            for ds in datasets:
                if ds not in all_results:
                    row.append("-")
                    continue
                v = all_results[ds]["aggregated"].get(f"{var_name}@{K}", 0.0)
                n = all_results[ds]["n_with_gold"]
                row.append(f"{v:.4f}")
                vals_w.append(v * n)
                wt += n
            overall = (sum(vals_w) / wt) if wt else 0.0
            row.append(f"**{overall:.4f}**")
            L.append("| " + " | ".join(row) + " |")

    # LoCoMo ablation vs two_speaker_all_supplements
    loc = all_results.get("locomo_30q", {}).get("aggregated", {})
    L.append("\n## LoCoMo K=50 ablation vs two_speaker_all_supplements\n")
    all_r = loc.get("two_speaker_all_supplements@50", 0.0)
    L.append("| Variant | r@50 | Δ vs all_supplements |")
    L.append("|---|---:|---:|")
    for var_name, _ in VARIANTS:
        r = loc.get(f"{var_name}@50", 0.0)
        L.append(f"| {var_name} | {r:.4f} | {r - all_r:+.4f} |")

    # LoCoMo ablation vs two_speaker_all_supplements at K=20
    L.append("\n## LoCoMo K=20 ablation vs two_speaker_all_supplements\n")
    all_r20 = loc.get("two_speaker_all_supplements@20", 0.0)
    L.append("| Variant | r@20 | Δ vs all_supplements |")
    L.append("|---|---:|---:|")
    for var_name, _ in VARIANTS:
        r = loc.get(f"{var_name}@20", 0.0)
        L.append(f"| {var_name} | {r:.4f} | {r - all_r20:+.4f} |")

    # Subset analysis on LoCoMo
    loc_sub = all_results.get("locomo_30q", {}).get("subset_aggregates", {})
    L.append("\n## LoCoMo subset analysis (two_speaker_all_supplements)\n")
    L.append("| Subset | n | v2f@50 | two_speaker_alone@50 | "
             "two_speaker_all_supplements@50 | Δ(all vs v2f) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for sub_name in ("matched_user", "matched_assistant", "matched_both",
                      "matched_none", "speaker_fires"):
        d = loc_sub.get(sub_name, {})
        n = d.get("n", 0)
        if n == 0:
            L.append(f"| {sub_name} | 0 | - | - | - | - |")
            continue
        v = d.get("v2f@50", 0.0)
        ts = d.get("two_speaker_alone@50", 0.0)
        a = d.get("two_speaker_all_supplements@50", 0.0)
        L.append(
            f"| {sub_name} | {n} | {v:.4f} | {ts:.4f} | {a:.4f} | "
            f"{a - v:+.4f} |"
        )

    # Decision + verdict
    L.append("\n## Decision\n")
    loc_v2f_20 = loc.get("v2f@20", 0.0)
    loc_v2f_50 = loc.get("v2f@50", 0.0)
    loc_ts_50 = loc.get("two_speaker_alone@50", 0.0)
    loc_ts_20 = loc.get("two_speaker_alone@20", 0.0)
    loc_all_50 = loc.get("two_speaker_all_supplements@50", 0.0)
    loc_all_20 = loc.get("two_speaker_all_supplements@20", 0.0)
    L.append(f"- LoCoMo K=20 v2f = {loc_v2f_20:.4f}")
    L.append(f"- LoCoMo K=20 two_speaker_alone = {loc_ts_20:.4f}")
    L.append(f"- LoCoMo K=20 two_speaker_all_supplements = {loc_all_20:.4f}")
    L.append(f"- LoCoMo K=50 v2f = {loc_v2f_50:.4f}")
    L.append(f"- LoCoMo K=50 two_speaker_alone = {loc_ts_50:.4f}")
    L.append(f"- LoCoMo K=50 two_speaker_all_supplements = {loc_all_50:.4f}")
    L.append(f"- Prior ceiling (composition_v2_all) = 0.9170")
    L.append(f"- Δ vs 0.9170: {loc_all_50 - 0.9170:+.4f}")
    if loc_all_50 > 0.9170 + 0.001:
        L.append(
            f"  => **NEW K=50 CEILING: two_speaker_all_supplements** "
            f"(+{(loc_all_50 - 0.9170) * 100:.1f}pp over 0.917)"
        )
    elif abs(loc_all_50 - 0.9170) <= 0.001:
        L.append(
            "  => TIE at 0.917 — supplements saturate equally on both "
            "speaker-filter bases; not worth swapping base."
        )
    else:
        L.append(
            f"  => two_speaker_filter base interacts negatively with "
            f"supplements (Δ={loc_all_50 - 0.9170:+.4f}). Interesting but "
            f"composition_v2_all's speaker_user_filter base is better."
        )

    L.append("")
    if loc_ts_20 > loc_all_20 + 0.001:
        L.append(
            f"- K=20 confirms: two_speaker_alone ({loc_ts_20:.4f}) > "
            f"two_speaker_all_supplements ({loc_all_20:.4f}) by "
            f"{(loc_ts_20 - loc_all_20)*100:.1f}pp. Supplements HURT at K=20."
        )
    elif abs(loc_ts_20 - loc_all_20) <= 0.001:
        L.append(
            f"- K=20: two_speaker_alone ≈ all_supplements "
            f"({loc_ts_20:.4f} vs {loc_all_20:.4f})."
        )
    else:
        L.append(
            f"- K=20: all_supplements ({loc_all_20:.4f}) > two_speaker_alone "
            f"({loc_ts_20:.4f}) by {(loc_all_20 - loc_ts_20)*100:.1f}pp. "
            f"Supplements HELP even at K=20 — surprising."
        )

    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    client = OpenAI(timeout=90.0)
    generator = CriticalInfoGenerator(
        client=client,
        prompt_version=CRITICAL_PROMPT_VERSION,
        max_workers=4,
    )
    embedder = Embedder(client=client)

    conv_two_speakers = load_conv_two_speakers()
    conv_aliases = load_conv_aliases()
    print(
        f"Loaded {len(conv_two_speakers)} conv_two_speaker entries, "
        f"{len(conv_aliases)} conv_alias entries.",
        flush=True,
    )

    t0 = time.time()
    all_results: dict[str, dict] = {}
    for ds in DATASETS:
        try:
            all_results[ds] = evaluate_dataset(
                ds, client, conv_two_speakers, conv_aliases,
                generator, embedder,
            )
        except Exception as e:
            print(f"  [fatal] {ds} evaluation failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

        # Interim flush
        tmp_path = RESULTS_DIR / "two_speaker_composition.interim.json"
        try:
            with open(tmp_path, "w") as f:
                json.dump(
                    {
                        ds2: {
                            "ds_name": r["ds_name"],
                            "n_with_gold": r["n_with_gold"],
                            "aggregated": r["aggregated"],
                        }
                        for ds2, r in all_results.items()
                    },
                    f, indent=2,
                )
        except Exception as e:
            print(f"  (warn) interim flush failed: {e}", flush=True)

    total_elapsed = time.time() - t0

    # Save JSON
    json_path = RESULTS_DIR / "two_speaker_composition.json"
    payload = {
        "elapsed_s": round(total_elapsed, 2),
        "prompt_version": CRITICAL_PROMPT_VERSION,
        "variants": [v for v, _ in VARIANTS],
        "budgets": list(BUDGETS),
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {json_path}", flush=True)

    md = render_markdown(all_results, total_elapsed)
    md_path = RESULTS_DIR / "two_speaker_composition.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Wrote {md_path}", flush=True)

    # Console summary
    print("\n" + "=" * 80)
    print("TWO-SPEAKER COMPOSITION — recall summary")
    print("=" * 80)
    print(f"{'variant':32s} " + " ".join(
        f"{ds:>14s}@20 {ds:>14s}@50" for ds in DATASETS
    ))
    for var_name, _ in VARIANTS:
        row = f"{var_name:32s} "
        for ds in DATASETS:
            if ds not in all_results:
                row += f"{'-':>17s} {'-':>17s} "
                continue
            a20 = all_results[ds]["aggregated"].get(f"{var_name}@20", 0.0)
            a50 = all_results[ds]["aggregated"].get(f"{var_name}@50", 0.0)
            row += f"{a20:17.4f} {a50:17.4f} "
        print(row)
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
