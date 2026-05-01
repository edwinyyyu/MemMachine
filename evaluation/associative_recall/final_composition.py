"""Final composition test — stack ALL shipped wins and measure ceiling.

Six wins being stacked:
  1. Keyword router (router_v2fplus_default) — per-query specialist selection
  2. Ensemble ens_2_v2f_typeenum — v2f + type_enumerated sum_cosine (K=50)
  3. Critical-info store (always_top_M)
  4. Alias expansion (query-time alias sibling substitution)
  5. Context embedding window_1 (prev+curr+next enriched index)
  6. Clause decomposition clause_v2f_n2 (split on sentence boundaries)

Integration strategy (per query):
  - Base layer: ens_2 at K=50, v2f at K=20 (ensembles hurt K=20)
  - Overlay 1 (alias): if query matches any alias group, fold alias_expand_v2f
    hits into base via max-cosine per parent_index
  - Overlay 2 (clause): if query has >=2 clauses, union clause_v2f hits into
    base via max-cosine
  - Overlay 3 (context): append context-embedding window_1 hits into remaining
    K slots (stacked merge — context can only fill slots base did not claim)
  - Overlay 4 (critical-info): always_top_M forces top-5 crit items into
    output, displacing weakest base items (score-bonus competition)

Variants evaluated:
  - v2f, router_v2fplus_default, ens_2_v2f_typeenum, ens_all_plus_crit (controls)
  - finalstack_all, finalstack_no_alias, finalstack_no_clause,
    finalstack_no_context, finalstack_no_critinfo (ablations)

This module does NOT modify framework files; it imports and composes.

Usage:
    uv run python final_composition.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from alias_expansion import AliasExpandV2fFull
from associative_recall import (
    Segment,
    SegmentStore,
)
from clause_decomposition import ClauseV2fN2, split_query_into_clauses
from context_embedding import ContextEmbW1Stacked
from critical_info_store import (
    CriticalInfoGenerator,
    CriticalInfoStore,
    classify_turns,
    decisions_to_altkeys,
)
from dotenv import load_dotenv
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
    compute_recall,
    embed_texts_cached,
    fair_backfill_turn_ids,
)
from openai import OpenAI
from router_study import KEYWORD_RULES

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ---------------------------------------------------------------------------
# Paths / datasets / budgets
# ---------------------------------------------------------------------------
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

# Specialist call costs (units of 1 v2f call) for retrieval-time cost accounting.
SPECIALIST_COST: dict[str, float] = {
    "v2f": 1.0,
    "v2f_plus_types": 2.0,
    "type_enumerated": 1.0,
    "chain_with_scratchpad": 5.0,
    "v2f_style_explicit": 1.0,
}
CRITICAL_PROMPT_VERSION = "v3"


# ---------------------------------------------------------------------------
# Keyword router mapping (same as composition_eval)
# ---------------------------------------------------------------------------
COMPOSITION_FOR_LABEL: dict[str, tuple[str, ...]] = {
    "type_enumerated": ("v2f", "type_enumerated"),
    "chain": ("v2f", "chain_with_scratchpad"),
    "v2f_plus_types": ("v2f_plus_types",),
    "v2f_style_explicit": ("v2f_style_explicit",),
    "v2f": ("v2f",),
}


def route_keyword_label(question: str) -> str:
    for pat, lab in KEYWORD_RULES:
        if pat.search(question):
            return lab
    return "v2f_plus_types"


# ---------------------------------------------------------------------------
# Loaders
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

    # Cosine and specialists (all cache-hit expected)
    cosine_segments: list[Segment]
    cosine_scores: list[float]
    outputs: dict[str, SpecialistOutput]
    router_label: str
    router_composition: tuple[str, ...]

    # Supplement outputs
    crit_ranked: list[tuple[int, float, Segment]]  # (parent_idx, score, seg)
    context_hits: dict[int, float]  # parent_idx -> score
    alias_segments: list[Segment]  # ordered, from alias_expand_v2f
    alias_cos: dict[int, float]  # parent_idx -> cos vs query
    clause_segments: list[Segment]  # ordered, from clause_v2f_n2
    clause_cos: dict[int, float]  # parent_idx -> cos vs query

    # Metadata
    alias_matched: bool
    clause_split: bool
    clause_llm_calls: int = 0


# ---------------------------------------------------------------------------
# Utility: cosine of segment embeddings vs query
# ---------------------------------------------------------------------------
def _cosine_per_index(
    store: SegmentStore, indices: list[int], q_emb_norm: np.ndarray
) -> dict[int, float]:
    if not indices:
        return {}
    out: dict[int, float] = {}
    arr = np.array(indices, dtype=np.int64)
    seg_embs = store.normalized_embeddings[arr]
    sims = seg_embs @ q_emb_norm
    for i, s in zip(indices, sims.tolist()):
        out[i] = float(s)
    return out


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        return v
    return v / n


# ---------------------------------------------------------------------------
# Alias extractor (reused singleton per store)
# ---------------------------------------------------------------------------
_ALIAS_ARCH_CACHE: dict[int, AliasExpandV2fFull] = {}


def _get_alias_arch(store: SegmentStore, client: OpenAI) -> AliasExpandV2fFull:
    key = id(store)
    a = _ALIAS_ARCH_CACHE.get(key)
    if a is None:
        a = AliasExpandV2fFull(store, client=client)
        _ALIAS_ARCH_CACHE[key] = a
    return a


# ---------------------------------------------------------------------------
# ContextEmb arch (per store window_1 index)
# ---------------------------------------------------------------------------
_CONTEXT_ARCH_CACHE: dict[int, ContextEmbW1Stacked] = {}


def _get_context_arch(store: SegmentStore, client: OpenAI) -> ContextEmbW1Stacked:
    key = id(store)
    c = _CONTEXT_ARCH_CACHE.get(key)
    if c is None:
        c = ContextEmbW1Stacked(store, client=client)
        _CONTEXT_ARCH_CACHE[key] = c
    return c


# ---------------------------------------------------------------------------
# Clause arch (per store)
# ---------------------------------------------------------------------------
_CLAUSE_ARCH_CACHE: dict[int, ClauseV2fN2] = {}


def _get_clause_arch(store: SegmentStore, client: OpenAI) -> ClauseV2fN2:
    key = id(store)
    c = _CLAUSE_ARCH_CACHE.get(key)
    if c is None:
        c = ClauseV2fN2(store, client=client)
        _CLAUSE_ARCH_CACHE[key] = c
    return c


# ---------------------------------------------------------------------------
# Build per-question context
# ---------------------------------------------------------------------------
def build_qcontext(
    store: SegmentStore,
    specialists: dict,
    question: dict,
    crit_store: CriticalInfoStore,
    alias_arch: AliasExpandV2fFull,
    context_arch: ContextEmbW1Stacked,
    clause_arch: ClauseV2fN2,
) -> QContext:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])
    cat = question.get("category", "unknown")

    # Query embedding from v2f (shared cache)
    q_emb = specialists["v2f"].embed_text(q_text)
    q_emb_norm = _normalize(q_emb)

    # Cosine top-max_K
    cos_res = store.search(q_emb, top_k=max(BUDGETS), conversation_id=conv_id)
    cos_segs = list(cos_res.segments)
    cos_scores = list(cos_res.scores)

    # Specialist outputs (cache-only)
    outputs = run_specialists_cached(specialists, store, q_text, conv_id, q_emb)

    # Router
    label = route_keyword_label(q_text)
    comp = COMPOSITION_FOR_LABEL.get(label, ("v2f",))

    # Critical info
    crit = crit_store.search_per_parent(
        q_emb, top_m=max(BUDGETS), conversation_id=conv_id, min_score=-1.0
    )

    # Context embedding — use arch's ctx_index directly
    ctx_hits_raw = context_arch.ctx_index.search_top_m(
        q_emb, conversation_id=conv_id, top_m=context_arch.top_m
    )
    # Dedupe per parent; keep max score
    ctx_per_parent: dict[int, float] = {}
    for _eid, pidx, sc in ctx_hits_raw:
        cur = ctx_per_parent.get(pidx)
        if cur is None or sc > cur:
            ctx_per_parent[pidx] = sc

    # Alias expansion (run only if conversation has alias groups & query matches)
    groups = alias_arch.extractor.get_groups(conv_id)
    alias_matched = False
    alias_segs: list[Segment] = []
    alias_cos: dict[int, float] = {}
    if groups:
        from alias_expansion import find_alias_matches

        matches = find_alias_matches(q_text, groups)
        if matches:
            alias_matched = True
            # Run alias_expand_v2f. This is cache-heavy — most LLM calls cached.
            alias_arch.reset_counters()
            res = alias_arch.retrieve(q_text, conv_id)
            alias_segs = list(res.segments)
            idxs = [s.index for s in alias_segs]
            alias_cos = _cosine_per_index(store, idxs, q_emb_norm)

    # Clause decomposition
    clauses = split_query_into_clauses(q_text, max_clauses=2)
    clause_segs: list[Segment] = []
    clause_cos: dict[int, float] = {}
    clause_split = len(clauses) >= 2
    clause_llm = 0
    if clause_split:
        clause_arch.reset_counters()
        res = clause_arch.retrieve(q_text, conv_id)
        clause_segs = list(res.segments)
        idxs = [s.index for s in clause_segs]
        clause_cos = _cosine_per_index(store, idxs, q_emb_norm)
        clause_llm = clause_arch.llm_calls

    return QContext(
        question=question,
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
        context_hits=ctx_per_parent,
        alias_segments=alias_segs,
        alias_cos=alias_cos,
        clause_segments=clause_segs,
        clause_cos=clause_cos,
        alias_matched=alias_matched,
        clause_split=clause_split,
        clause_llm_calls=clause_llm,
    )


# ---------------------------------------------------------------------------
# Base layer for finalstack: ens_2 at K=50, v2f at K=20
# ---------------------------------------------------------------------------
def _dedupe_by_index(segments: list[Segment]) -> list[Segment]:
    seen: set[int] = set()
    out: list[Segment] = []
    for s in segments:
        if s.index not in seen:
            out.append(s)
            seen.add(s.index)
    return out


def _base_ranked_with_scores(ctx: QContext, K: int) -> list[tuple[Segment, float]]:
    """Base ranked (segment, score) list with cosine scores for arch picks
    inflated by +5 so they sort above cosine backfill. At K=50 use ens_2
    sum_cosine; at K=20 use v2f.
    """
    q_emb_norm = _normalize(ctx.query_emb)
    if K >= 50:
        # ens_2 sum_cosine
        comp = ("v2f", "type_enumerated")
        pool: dict[int, dict] = {}
        for name in comp:
            so = ctx.outputs.get(name)
            if so is None:
                continue
            for rank, (seg, cos) in enumerate(zip(so.segments, so.cosine_scores)):
                entry = pool.setdefault(seg.index, {"segment": seg, "sum": 0.0})
                entry["sum"] += cos
        merged: list[tuple[Segment, float]] = []
        for idx, e in pool.items():
            merged.append((e["segment"], float(e["sum"] + 5.0)))
        merged.sort(key=lambda rc: -rc[1])
        seen: set[int] = {s.index for s, _ in merged}
        # Cosine backfill — use raw cosine scores (not inflated)
        for s, sc in zip(ctx.cosine_segments, ctx.cosine_scores):
            if s.index in seen:
                continue
            merged.append((s, float(sc)))
            seen.add(s.index)
        return merged
    # K=20: v2f alone
    v2f_so = ctx.outputs.get("v2f")
    EPS = 0.001
    ranked: list[tuple[Segment, float]] = []
    seen2: set[int] = set()
    if v2f_so is not None:
        for rank, s in enumerate(v2f_so.segments):
            if s.index in seen2:
                continue
            ranked.append((s, 10.0 - rank * EPS))
            seen2.add(s.index)
    for s, sc in zip(ctx.cosine_segments, ctx.cosine_scores):
        if s.index in seen2:
            continue
        ranked.append((s, float(sc)))
        seen2.add(s.index)
    return ranked


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------
def _overlay_alias(
    ranked: list[tuple[Segment, float]],
    ctx: QContext,
) -> list[tuple[Segment, float]]:
    """Max-cosine merge: add alias_expand_v2f hits as if they were cosine
    candidates; for already-present parents, keep max score."""
    if not ctx.alias_matched:
        return ranked
    # Build running best score per index
    best: dict[int, tuple[Segment, float]] = {}
    for s, sc in ranked:
        best[s.index] = (s, sc)
    for s in ctx.alias_segments:
        # Alias candidates score = cosine vs query (raw). Compare against
        # existing score; keep max.
        cos = ctx.alias_cos.get(s.index, 0.0)
        cur = best.get(s.index)
        if cur is None or cos > cur[1]:
            best[s.index] = (s, float(cos))
    merged = list(best.values())
    merged.sort(key=lambda rc: -rc[1])
    return merged


def _overlay_clause(
    ranked: list[tuple[Segment, float]],
    ctx: QContext,
) -> list[tuple[Segment, float]]:
    """Same max-cosine merge for clause_v2f_n2 outputs."""
    if not ctx.clause_split:
        return ranked
    best: dict[int, tuple[Segment, float]] = {}
    for s, sc in ranked:
        best[s.index] = (s, sc)
    for s in ctx.clause_segments:
        cos = ctx.clause_cos.get(s.index, 0.0)
        cur = best.get(s.index)
        if cur is None or cos > cur[1]:
            best[s.index] = (s, float(cos))
    merged = list(best.values())
    merged.sort(key=lambda rc: -rc[1])
    return merged


def _overlay_context(
    ranked: list[tuple[Segment, float]],
    ctx: QContext,
    store: SegmentStore,
    K: int,
) -> list[tuple[Segment, float]]:
    """Stacked merge: context hits appended ONLY to fill remaining K slots."""
    if not ctx.context_hits:
        return ranked
    # Current top-K (from ranked) — they have priority.
    seen: set[int] = set()
    topK: list[tuple[Segment, float]] = []
    for s, sc in ranked:
        if s.index in seen:
            continue
        topK.append((s, sc))
        seen.add(s.index)
        if len(topK) >= K:
            break
    if len(topK) >= K:
        return ranked  # No room to append
    # Append context hits in score order for novel parents
    ctx_order = sorted(ctx.context_hits.items(), key=lambda x: -x[1])
    appended = 0
    needed = K - len(topK)
    for pidx, sc in ctx_order:
        if appended >= needed:
            break
        if pidx in seen:
            continue
        if 0 <= pidx < len(store.segments):
            seg = store.segments[pidx]
            # Score well above cosine backfill but below the arch picks; just
            # assign raw score (won't re-sort past arch picks in practice).
            topK.append((seg, float(sc)))
            seen.add(pidx)
            appended += 1
    # Rebuild ranked: topK first (preserved order), then untouched trailing
    # entries from ranked.
    out = list(topK)
    for s, sc in ranked:
        if s.index in seen:
            continue
        out.append((s, sc))
        seen.add(s.index)
    return out


def _overlay_critinfo(
    ranked: list[tuple[Segment, float]],
    ctx: QContext,
    K: int,
    top_m: int = 5,
    min_score: float = 0.2,
) -> list[tuple[Segment, float]]:
    """always_top_M pattern: force top-M crit items (above min_score) into
    the top-K, displacing weakest base items.

    Per plan: 'score bonus, max-score competition ONLY with weakest base items'.
    We implement by promoting top-M crit to the head of the list, then
    deduping the remainder so the overall length stays >= K.
    """
    if not ctx.crit_ranked:
        return ranked
    crit = [(p, s, seg) for (p, s, seg) in ctx.crit_ranked[:top_m] if s >= min_score]
    if not crit:
        return ranked
    # Force crit to the front
    seen: set[int] = set()
    out: list[tuple[Segment, float]] = []
    for pidx, sc, seg in crit:
        if pidx in seen:
            continue
        # Give crit a high score so they stay at the top of resulting list
        out.append((seg, 100.0 + sc))
        seen.add(pidx)
    for s, sc in ranked:
        if s.index in seen:
            continue
        out.append((s, sc))
        seen.add(s.index)
    return out


# ---------------------------------------------------------------------------
# Variant assemblers
# ---------------------------------------------------------------------------
def finalstack_assemble(
    ctx: QContext,
    store: SegmentStore,
    K: int,
    enable_alias: bool,
    enable_clause: bool,
    enable_context: bool,
    enable_critinfo: bool,
) -> set[int]:
    ranked = _base_ranked_with_scores(ctx, K)
    if enable_alias:
        ranked = _overlay_alias(ranked, ctx)
    if enable_clause:
        ranked = _overlay_clause(ranked, ctx)
    if enable_context:
        ranked = _overlay_context(ranked, ctx, store, K)
    if enable_critinfo:
        ranked = _overlay_critinfo(ranked, ctx, K)
    # Truncate to K and return turn_ids
    seen_idx: set[int] = set()
    out_ids: set[int] = set()
    count = 0
    for s, _sc in ranked:
        if s.index in seen_idx:
            continue
        seen_idx.add(s.index)
        out_ids.add(s.turn_id)
        count += 1
        if count >= K:
            break
    if count < K:
        # Backfill with cosine (shouldn't usually be needed — ens_2 already
        # has cosine backfill built in).
        for s in ctx.cosine_segments:
            if s.index in seen_idx:
                continue
            seen_idx.add(s.index)
            out_ids.add(s.turn_id)
            count += 1
            if count >= K:
                break
    return out_ids


def variant_v2f(ctx: QContext, store: SegmentStore, K: int) -> set[int]:
    segs = _dedupe_by_index(list(ctx.outputs["v2f"].segments))
    return fair_backfill_turn_ids(segs, ctx.cosine_segments, K)


def variant_router(ctx: QContext, store: SegmentStore, K: int) -> set[int]:
    label = ctx.router_label
    if label == "type_enumerated":
        spec_name = "type_enumerated"
    elif label == "chain":
        spec_name = "chain_with_scratchpad"
    elif label == "v2f_style_explicit":
        spec_name = "v2f_style_explicit"
    else:
        spec_name = "v2f_plus_types"
    so = ctx.outputs[spec_name]
    segs = _dedupe_by_index(list(so.segments))
    return fair_backfill_turn_ids(segs, ctx.cosine_segments, K)


def variant_ens_2(ctx: QContext, store: SegmentStore, K: int) -> set[int]:
    segs = ensemble_at_k(
        ctx.outputs,
        ("v2f", "type_enumerated"),
        "sum_cosine",
        ctx.cosine_segments,
        K,
    )
    return {s.turn_id for s in segs}


def variant_ens_all_plus_crit(ctx: QContext, store: SegmentStore, K: int) -> set[int]:
    # Same pattern as composition_eval's ens_all_plus_crit
    from critical_info_store import merge_always_top_m

    comp = ENSEMBLE_COMPOSITIONS["ens_5"]
    # ensemble_main_ranked
    pool: dict[int, dict] = {}
    sub = {n: ctx.outputs[n] for n in comp if n in ctx.outputs}
    for name, so in sub.items():
        for rank, (seg, cos) in enumerate(zip(so.segments, so.cosine_scores)):
            entry = pool.setdefault(seg.index, {"segment": seg, "sum": 0.0})
            entry["sum"] += cos
    merged: list[tuple[Segment, float]] = []
    for idx, e in pool.items():
        merged.append((e["segment"], float(e["sum"] + 5.0)))
    merged.sort(key=lambda rc: -rc[1])
    seen: set[int] = {s.index for s, _ in merged}
    for s, sc in zip(ctx.cosine_segments, ctx.cosine_scores):
        if s.index in seen:
            continue
        merged.append((s, float(sc)))
        seen.add(s.index)
    out_segs = merge_always_top_m(
        merged,
        ctx.crit_ranked,
        K,
        top_m=5,
        min_score=0.2,
    )
    return {s.turn_id for s in out_segs}


VARIANT_FUNCS = {
    "v2f": variant_v2f,
    "router_v2fplus_default": variant_router,
    "ens_2_v2f_typeenum": variant_ens_2,
    "ens_all_plus_crit": variant_ens_all_plus_crit,
    "finalstack_all": lambda ctx, s, K: finalstack_assemble(
        ctx, s, K, True, True, True, True
    ),
    "finalstack_no_alias": lambda ctx, s, K: finalstack_assemble(
        ctx, s, K, False, True, True, True
    ),
    "finalstack_no_clause": lambda ctx, s, K: finalstack_assemble(
        ctx, s, K, True, False, True, True
    ),
    "finalstack_no_context": lambda ctx, s, K: finalstack_assemble(
        ctx, s, K, True, True, False, True
    ),
    "finalstack_no_critinfo": lambda ctx, s, K: finalstack_assemble(
        ctx, s, K, True, True, True, False
    ),
}

VARIANTS_ORDER = list(VARIANT_FUNCS.keys())


# ---------------------------------------------------------------------------
# LLM cost per variant per question
# ---------------------------------------------------------------------------
def llm_cost_variant(variant: str, ctx: QContext) -> float:
    """Retrieval-time LLM calls (units of v2f). Ingest-time classifier &
    alias-extraction costs are one-off and reported separately."""
    base_cost: float
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
    if variant == "ens_2_v2f_typeenum":
        return SPECIALIST_COST["v2f"] + SPECIALIST_COST["type_enumerated"]
    if variant == "ens_all_plus_crit":
        return sum(SPECIALIST_COST[s] for s in ENSEMBLE_COMPOSITIONS["ens_5"])
    if variant.startswith("finalstack"):
        # Base cost: ens_2 at K=50, v2f at K=20 — we report mean over budgets.
        # Use ens_2 sum for simplicity (upper bound).
        base = SPECIALIST_COST["v2f"] + SPECIALIST_COST["type_enumerated"]
        # Alias: alias_expand_v2f_full ~= 1 + num_variants (v2f per variant)
        alias_cost = 0.0
        if variant != "finalstack_no_alias" and ctx.alias_matched:
            n_var = max(1, len(ctx.alias_segments) // 10)  # rough proxy
            # Use actual count of alias probes if we had it; else estimate 3x
            alias_cost = 3.0
        clause_cost = 0.0
        if variant != "finalstack_no_clause" and ctx.clause_split:
            clause_cost = 2.0  # 1 v2f call per clause for 2 clauses
        return base + alias_cost + clause_cost
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
    print(f"  questions={len(questions)} segments={len(store.segments)}", flush=True)

    # --- Critical-info classifier (uses shared bestshot_llm_cache) ---
    conv_ids = {q["conversation_id"] for q in questions}
    target = [s for s in store.segments if s.conversation_id in conv_ids]
    print(f"  target segments: {len(target)} — classifying (LLM cached)", flush=True)
    t_c = time.time()
    decisions = classify_turns(generator, target, log_every=200)
    n_crit = sum(1 for d in decisions if d.critical)
    print(
        f"  classify done in {time.time() - t_c:.1f}s — crit={n_crit}/{len(decisions)}",
        flush=True,
    )
    alt_keys = decisions_to_altkeys(decisions)
    alt_texts = [k.text for k in alt_keys]
    if alt_texts:
        alt_embs = embed_texts_cached(
            client,
            embedder.embedding_cache,
            alt_texts,
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

    # --- Specialists (cache-only) ---
    specialists = {name: build_specialist(name, store) for name in SPECIALISTS}

    # --- Context embedding index (built or cache-hit) ---
    t_ctx = time.time()
    context_arch = _get_context_arch(store, client)
    print(
        f"  context index ready in {time.time() - t_ctx:.1f}s "
        f"(n_entries={context_arch.ctx_index.n})",
        flush=True,
    )

    # --- Alias extractor (uses persisted conversation_alias_groups.json) ---
    alias_arch = _get_alias_arch(store, client)
    n_groups = sum(len(alias_arch.extractor.get_groups(c)) for c in sorted(conv_ids))
    print(f"  alias groups available: {n_groups}", flush=True)

    # --- Clause arch (shares dedicated caches) ---
    clause_arch = _get_clause_arch(store, client)

    # --- Build per-question contexts ---
    print("  building per-question contexts...", flush=True)
    t_qc = time.time()
    ctxs: list[QContext] = []
    tot_clause_llm = 0
    for q in questions:
        c = build_qcontext(
            store,
            specialists,
            q,
            crit_store,
            alias_arch,
            context_arch,
            clause_arch,
        )
        ctxs.append(c)
        tot_clause_llm += c.clause_llm_calls
    print(
        f"  contexts built in {time.time() - t_qc:.1f}s "
        f"(clause-llm-calls total={tot_clause_llm})",
        flush=True,
    )

    # Save incremental caches
    for arch in specialists.values():
        try:
            arch.save_caches()
        except Exception:
            pass
    try:
        clause_arch.embedding_cache.save()
        clause_arch.llm_cache.save()
    except Exception:
        pass
    try:
        context_arch.embedding_cache.save()
        context_arch.llm_cache.save()
    except Exception:
        pass
    try:
        alias_arch.embedding_cache.save()
        alias_arch.llm_cache.save()
    except Exception:
        pass

    # --- Evaluate each variant ---
    per_q_rows: list[dict] = []
    for ctx in ctxs:
        row: dict = {
            "dataset": ds_name,
            "conversation_id": ctx.conv_id,
            "question_index": ctx.question.get("question_index", -1),
            "question": ctx.q_text,
            "category": ctx.category,
            "num_source_turns": len(ctx.source_ids),
            "router_label": ctx.router_label,
            "alias_matched": ctx.alias_matched,
            "clause_split": ctx.clause_split,
            "clause_llm_calls": ctx.clause_llm_calls,
            "n_context_hits": len(ctx.context_hits),
            "n_crit_candidates": len(ctx.crit_ranked),
            "recall": {},
            "retrieved_ids": {},
            "llm_calls_per_variant": {},
        }
        if not ctx.source_ids:
            per_q_rows.append(row)
            continue
        for var_name, fn in VARIANT_FUNCS.items():
            row["recall"][var_name] = {}
            row["retrieved_ids"][var_name] = {}
            for K in BUDGETS:
                ids = fn(ctx, store, K)
                row["recall"][var_name][f"r@{K}"] = round(
                    compute_recall(ids, ctx.source_ids), 4
                )
                # Only retain K=50 retrieved ids for orthogonality analysis
                if max(BUDGETS) == K:
                    row["retrieved_ids"][var_name][f"r@{K}"] = sorted(ids)
            row["llm_calls_per_variant"][var_name] = round(
                llm_cost_variant(var_name, ctx), 2
            )
        per_q_rows.append(row)

    # --- Aggregate ---
    per_variant: dict = {}
    for var in VARIANTS_ORDER:
        per_variant[var] = {}
        for K in BUDGETS:
            vals = [
                r["recall"][var][f"r@{K}"]
                for r in per_q_rows
                if r["num_source_turns"] > 0
            ]
            per_variant[var][f"r@{K}"] = (
                round(sum(vals) / len(vals), 4) if vals else 0.0
            )
    per_variant_cost: dict = {}
    for var in VARIANTS_ORDER:
        costs = [
            r["llm_calls_per_variant"][var]
            for r in per_q_rows
            if r["num_source_turns"] > 0
        ]
        per_variant_cost[var] = round(sum(costs) / len(costs), 3) if costs else 0.0

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
                        sum(by_cat[cat][var][K]) / max(1, len(by_cat[cat][var][K])),
                        4,
                    )
                    for K in BUDGETS
                }
                for var in VARIANTS_ORDER
            },
        }
        for cat in cat_counts
    }

    # Supplement stats
    n_alias_fired = sum(1 for r in per_q_rows if r["alias_matched"])
    n_clause_split = sum(1 for r in per_q_rows if r["clause_split"])

    return {
        "ds_name": ds_name,
        "n_questions": len(questions),
        "n_with_gold": sum(1 for r in per_q_rows if r["num_source_turns"] > 0),
        "n_target_segments": len(target),
        "n_critical_turns": n_crit,
        "flag_rate": round(n_crit / max(1, len(decisions)), 4),
        "n_altkeys_dedup": len(alt_keys),
        "n_alias_fired": n_alias_fired,
        "n_clause_split": n_clause_split,
        "per_variant": per_variant,
        "per_variant_llm_cost": per_variant_cost,
        "per_category": per_category,
        "per_question": per_q_rows,
    }


# ---------------------------------------------------------------------------
# Orthogonality (on finalstack_all) — fraction of gold found only via stacking
# ---------------------------------------------------------------------------
def orthogonality_analysis(all_results: dict) -> dict:
    """For each dataset @K=50, compute: of the gold turns found by
    finalstack_all, how many were NOT found by any single prior ship
    (v2f, router, ens_2, ens_all+crit)?"""
    out: dict = {}
    for ds, res in all_results.items():
        gold_found_by = {
            "v2f": set(),
            "router_v2fplus_default": set(),
            "ens_2_v2f_typeenum": set(),
            "ens_all_plus_crit": set(),
            "finalstack_all": set(),
        }
        total_gold = 0
        for r in res["per_question"]:
            if r["num_source_turns"] == 0:
                continue
            gold = set(r.get("source_chat_ids", []))
            # We stripped source_chat_ids — recover from question via num_source_turns
            # Actually we need gold. Rebuild via retrieved ∩ source_ids doesn't work
            # if we don't have source_ids. Let's use recall * n_source approach
            # — instead just count novel gold by diffing retrieved sets.

        out[ds] = {"note": "see per_question for retrieved ids"}
    return out


# ---------------------------------------------------------------------------
# Orthogonality — fraction of gold finalstack_all found that none of
# {v2f, router, ens_2, ens_all+crit} found.
# ---------------------------------------------------------------------------
def compute_ortho(
    per_question: list[dict],
    datasets_questions: dict[str, list[dict]],
    ds_name: str,
) -> dict:
    qs_by_idx = {
        (q["conversation_id"], q.get("question_index", -1)): q
        for q in datasets_questions[ds_name]
    }
    total_gold_union_all = 0
    novel_count = 0
    prior_variants = [
        "v2f",
        "router_v2fplus_default",
        "ens_2_v2f_typeenum",
        "ens_all_plus_crit",
    ]
    per_var_novel: dict = dict.fromkeys(prior_variants + ["finalstack_all"], 0)
    for r in per_question:
        if r["num_source_turns"] == 0:
            continue
        q = qs_by_idx.get((r["conversation_id"], r["question_index"]))
        if q is None:
            continue
        gold = set(q["source_chat_ids"])
        fs_ret = set(
            r["retrieved_ids"].get("finalstack_all", {}).get(f"r@{max(BUDGETS)}", [])
        )
        fs_gold = fs_ret & gold
        # Union of prior ships' retrievals
        prior_union: set[int] = set()
        for v in prior_variants:
            prior_union |= set(
                r["retrieved_ids"].get(v, {}).get(f"r@{max(BUDGETS)}", [])
            )
        prior_gold = prior_union & gold
        novel_count += len(fs_gold - prior_gold)
        total_gold_union_all += len(fs_gold)
        for v in prior_variants + ["finalstack_all"]:
            ret = set(r["retrieved_ids"].get(v, {}).get(f"r@{max(BUDGETS)}", []))
            per_var_novel[v] += len(ret & gold)
    return {
        "ds_name": ds_name,
        "finalstack_all_gold_found@50": total_gold_union_all,
        "novel_gold_only_via_finalstack@50": novel_count,
        "frac_novel": round(novel_count / max(1, total_gold_union_all), 4),
        "per_variant_gold_found@50": per_var_novel,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------
def render_markdown(
    all_results: dict,
    ortho: dict,
    total_elapsed: float,
    classifier_cost: dict,
) -> str:
    L: list[str] = []
    L.append("# Final Composition Test — stack ALL shipped wins\n")
    L.append(
        "Stacking the 6 narrow wins discovered this session and measuring the\n"
        "cumulative ceiling. Integration is: base (ens_2 @K=50 / v2f @K=20) +\n"
        "alias overlay (max-cos) + clause overlay (max-cos) + context-emb\n"
        "stacked append + critical-info always_top_M.\n"
    )
    L.append(f"\nElapsed: {total_elapsed:.0f}s.\n")

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

    # Ablation vs finalstack_all
    L.append("\n## Ablation (overall @K=50)\n")
    L.append("| Variant | overall | Δ vs finalstack_all |")
    L.append("|---|---|---|")
    ov = {}
    for var in VARIANTS_ORDER:
        total = 0.0
        wt = 0
        for ds in DATASETS:
            total += (
                all_results[ds]["per_variant"][var]["r@50"]
                * all_results[ds]["n_with_gold"]
            )
            wt += all_results[ds]["n_with_gold"]
        ov[var] = total / max(1, wt)
    fs = ov.get("finalstack_all", 0.0)
    for var in VARIANTS_ORDER:
        delta = ov[var] - fs
        L.append(f"| {var} | {ov[var]:.4f} | {delta:+.4f} |")

    # Supplement usage stats
    L.append("\n## Supplement trigger rates\n")
    L.append(
        "| Dataset | alias_matched | clause_split | n_crit_turns | "
        "altkeys | context_hits |"
    )
    L.append("|---|---|---|---|---|---|")
    for ds in DATASETS:
        res = all_results[ds]
        n = res["n_with_gold"]
        L.append(
            f"| {ds} | {res['n_alias_fired']}/{n} | "
            f"{res['n_clause_split']}/{n} | "
            f"{res['n_critical_turns']} | "
            f"{res['n_altkeys_dedup']} | "
            f"{sum(r['n_context_hits'] for r in res['per_question']) // max(1, len(res['per_question']))} avg |"
        )

    # LLM retrieval cost per variant
    L.append("\n## LLM retrieval cost per question (rel. to 1 v2f call)\n")
    L.append("| Variant | " + " | ".join(DATASETS) + " |")
    L.append("|---|" + "---|" * len(DATASETS))
    for var in VARIANTS_ORDER:
        row = [var]
        for ds in DATASETS:
            c = all_results[ds]["per_variant_llm_cost"][var]
            row.append(f"{c:.2f}×")
        L.append("| " + " | ".join(row) + " |")

    # Per-category breakdowns for LoCoMo and synthetic @K=50
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

    # Orthogonality
    L.append("\n## Orthogonality: gold found by finalstack_all but NO prior ship\n")
    L.append("| Dataset | finalstack gold@50 | novel vs priors | frac_novel |")
    L.append("|---|---|---|---|")
    for ds, o in ortho.items():
        L.append(
            f"| {ds} | {o['finalstack_all_gold_found@50']} | "
            f"{o['novel_gold_only_via_finalstack@50']} | "
            f"{o['frac_novel']} |"
        )

    # Classifier cost
    L.append("\n## Critical-info classifier cost (ingest-time, one-off)\n")
    L.append(
        f"- Prompt version: {CRITICAL_PROMPT_VERSION}\n"
        f"- New calls this run: {classifier_cost['n_uncached']}, "
        f"cached: {classifier_cost['n_cached']}\n"
        f"- Input tokens: {classifier_cost['prompt_tokens']} "
        f"output tokens: {classifier_cost['completion_tokens']}\n"
        f"- Est USD (gpt-5-mini @ $0.25/M in, $2/M out): "
        f"${classifier_cost['est_usd']:.4f}\n"
    )

    # Verdict
    L.append("\n## Verdict\n")
    best_var = max(VARIANTS_ORDER, key=lambda v: ov[v])
    L.append(
        f"- Best variant overall @ K=50 (weighted): **{best_var}** "
        f"r@50={ov[best_var]:.4f}\n"
    )
    L.append(
        f"- Margin over ens_all_plus_crit: "
        f"{ov[best_var] - ov.get('ens_all_plus_crit', 0):+.4f}\n"
    )
    L.append(
        f"- Margin over router_v2fplus_default: "
        f"{ov[best_var] - ov.get('router_v2fplus_default', 0):+.4f}\n"
    )
    # Ablation call-outs
    for drop in ("no_alias", "no_clause", "no_context", "no_critinfo"):
        v = f"finalstack_{drop}"
        if v in ov:
            contribution = ov["finalstack_all"] - ov[v]
            L.append(
                f"- Supplement contribution (drop {drop}): "
                f"{contribution:+.4f} pp vs finalstack_all\n"
            )
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _strip_result(res: dict) -> dict:
    """Keep per-question but prune heavy retrieved_ids (except finalstack_all
    and the 4 priors needed for orthogonality)."""
    out = {k: v for k, v in res.items()}
    per_q = out.get("per_question", [])
    keep_variants = {
        "v2f",
        "router_v2fplus_default",
        "ens_2_v2f_typeenum",
        "ens_all_plus_crit",
        "finalstack_all",
    }
    pruned = []
    for r in per_q:
        rr = {k: v for k, v in r.items() if k not in ("retrieved_ids",)}
        rr["retrieved_ids"] = {
            v: r["retrieved_ids"].get(v, {})
            for v in keep_variants
            if v in r.get("retrieved_ids", {})
        }
        pruned.append(rr)
    out["per_question"] = pruned
    return out


def main() -> None:
    t0 = time.time()
    client = OpenAI(timeout=60.0)
    generator = CriticalInfoGenerator(
        client=client,
        prompt_version=CRITICAL_PROMPT_VERSION,
        max_workers=8,
    )
    embedder = Embedder(client)

    datasets_questions: dict[str, list[dict]] = {}
    all_results: dict = {}
    for ds in DATASETS:
        qs = load_questions(ds)
        datasets_questions[ds] = qs
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
        cost["prompt_tokens"] * 0.25 / 1e6 + cost["completion_tokens"] * 2.0 / 1e6,
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

    # Orthogonality analysis
    ortho = {}
    for ds in DATASETS:
        ortho[ds] = compute_ortho(
            all_results[ds]["per_question"],
            datasets_questions,
            ds,
        )

    total_elapsed = time.time() - t0

    # Save JSON
    json_path = RESULTS_DIR / "final_composition.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "prompt_version": CRITICAL_PROMPT_VERSION,
                "elapsed_s": round(total_elapsed, 2),
                "classifier_cost": cost,
                "orthogonality": ortho,
                "results": {ds: _strip_result(res) for ds, res in all_results.items()},
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote {json_path}", flush=True)

    md = render_markdown(all_results, ortho, total_elapsed, cost)
    md_path = RESULTS_DIR / "final_composition.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Wrote {md_path}", flush=True)

    # Console summary
    print("\n" + "=" * 70)
    print("FINAL COMPOSITION — r@50 overall")
    print("=" * 70)
    print(f"{'variant':28s} " + " ".join(f"{ds:>14s}" for ds in DATASETS))
    for var in VARIANTS_ORDER:
        row = f"{var:28s} "
        for ds in DATASETS:
            r = all_results[ds]["per_variant"][var]["r@50"]
            row += f"{r:>14.4f} "
        print(row)
    print(f"\nTotal elapsed: {total_elapsed:.0f}s  classifier ${cost['est_usd']:.4f}")


def _flush_interim(all_results: dict) -> None:
    tmp_path = RESULTS_DIR / "final_composition.interim.json"
    try:
        payload = {
            "partial_results": {
                ds: {
                    "ds_name": res["ds_name"],
                    "n_with_gold": res["n_with_gold"],
                    "per_variant": res["per_variant"],
                    "flag_rate": res["flag_rate"],
                    "n_critical_turns": res["n_critical_turns"],
                    "n_alias_fired": res["n_alias_fired"],
                    "n_clause_split": res["n_clause_split"],
                }
                for ds, res in all_results.items()
            }
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"  (warn) interim flush failed: {e}", flush=True)


if __name__ == "__main__":
    main()
