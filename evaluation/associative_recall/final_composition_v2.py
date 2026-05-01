"""Final composition v2: stack narrow wins INCLUDING speaker_user_filter.

This is the v2 composition study. The headline is:
  speaker_user_filter (LoCoMo K=20 v2f 0.756 -> 0.839, +8.3pp) enters the stack.

Composition members (cache-only; no new per-query LLM calls beyond warm cache):
  - v2f (baseline specialist; MetaV2f)
  - v2f_plus_types
  - type_enumerated
  - chain_with_scratchpad
  - v2f_style_explicit
  - speaker_user_filter (NEW — biggest K=20 win on LoCoMo)
  - alias_expand_v2f
  - contextemb_w1_stacked
  - clause_plus_v2f
  - critical_info always_top_M (ingest-time classifier, separate pool)

The router (`router_study.KEYWORD_RULES`) picks per-question specialist from the
5-specialist palette. Compositions apply stack-merge of ranked segments then
fair-backfill to K. Speaker-filter is applied BEFORE base-layer stack when the
query mentions the conv-user; critical is overlaid at the end.

Runs on 4 datasets (locomo_30q, synthetic_19q, puzzle_16q, advanced_23q) at
K=20 and K=50, fair-backfilled.

Dedicated compv2_*_cache.json for any new writes; shared reads via each
specialist's own cache namespace.

Usage:
    uv run python final_composition_v2.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Narrow wins
from alias_expansion import _ALIAS_GROUPS_FILE, AliasExpandV2fFull, find_alias_matches
from associative_recall import (
    CACHE_DIR,
    Segment,
    SegmentStore,
)
from best_shot import MetaV2f
from clause_decomposition import ClausePlusV2f, split_query_into_clauses
from context_embedding import ContextEmbW1Stacked
from critical_info_store import (
    CriticalInfoGenerator,
    CriticalInfoStore,
    classify_turns,
    decisions_to_altkeys,
    merge_always_top_m,
)
from domain_agnostic import (
    NEUTRAL_HEADER,
    V2F_STYLE_EXPLICIT_PROMPT,
    DomainAgnosticVariant,
)
from dotenv import load_dotenv
from ensemble_retrieval import (
    SpecialistOutput,
    _attach_cosine_scores,
    _dedupe_preserve_order,
)
from goal_chain import GoalChainRetriever
from ingest_regex_eval import (
    Embedder,
    compute_recall,
    embed_texts_cached,
)
from openai import OpenAI
from router_study import KEYWORD_RULES
from speaker_attributed import (
    _CONV_SPEAKERS_FILE,
    SpeakerUserFilter,
    extract_name_mentions,
)
from type_enumerated import TypeEnumeratedVariant, V2fPlusTypesVariant

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ---------------------------------------------------------------------------
# Paths, datasets, budgets
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# compv2 dedicated cache namespace (we write only here)
CACHE_MY_LLM = CACHE_DIR / "compv2_llm_cache.json"
CACHE_MY_EMB = CACHE_DIR / "compv2_embedding_cache.json"

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

# Critical-info prompt version (matches shipped classifier)
CRITICAL_PROMPT_VERSION = "v3"


# ---------------------------------------------------------------------------
# Specialist palette + router mapping
# ---------------------------------------------------------------------------
CORE_SPECIALISTS = (
    "v2f",
    "v2f_plus_types",
    "type_enumerated",
    "chain_with_scratchpad",
    "v2f_style_explicit",
)

NARROW_WINS = (
    "speaker_user_filter",
    "alias_expand_v2f",
    "contextemb_w1_stacked",
    "clause_plus_v2f",
)

ALL_SPECIALISTS = CORE_SPECIALISTS + NARROW_WINS


# Router keyword rules map labels -> specialist composition.
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
# Build specialists (cache-only: missing LLM cache returns DONE)
# ---------------------------------------------------------------------------
def build_specialist(name: str, store: SegmentStore):
    """Build a specialist arch with cache-only LLM proxy.

    For narrow-win specialists (speaker, alias, context, clause), the
    initializer itself may do ingest-time LLM work (speaker ID, alias
    extraction). Those calls are cached to disk across runs.
    """
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
    elif name == "speaker_user_filter":
        arch = SpeakerUserFilter(store)
    elif name == "alias_expand_v2f":
        arch = AliasExpandV2fFull(store)
    elif name == "contextemb_w1_stacked":
        arch = ContextEmbW1Stacked(store)
    elif name == "clause_plus_v2f":
        arch = ClausePlusV2f(store)
    else:
        raise KeyError(name)

    # Replace the LLM call path with cache-only: any miss returns a DONE
    # cue, so the specialist can still run its retrieval logic without
    # making any per-query LLM calls. This is the same pattern as
    # ensemble_retrieval.build_specialist and composition_eval.build_specialist.
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
            name=name,
            segments=segs,
            cosine_scores=scores,
            llm_calls=arch.llm_calls,
        )
    return out


# ---------------------------------------------------------------------------
# Speaker-transform detection (reuses speaker_attributed logic)
# ---------------------------------------------------------------------------
def load_conv_speakers() -> dict[str, str]:
    if not _CONV_SPEAKERS_FILE.exists():
        return {}
    try:
        with open(_CONV_SPEAKERS_FILE) as f:
            return json.load(f).get("speakers", {}) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def query_mentions_conv_user(
    question: str,
    conversation_id: str,
    conv_speakers: dict[str, str],
) -> bool:
    name = conv_speakers.get(conversation_id, "UNKNOWN")
    if not name or name == "UNKNOWN":
        return False
    toks = extract_name_mentions(question)
    return any(t.lower() == name.lower() for t in toks)


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
    matches = find_alias_matches(question, groups)
    return len(matches) > 0


def query_multi_clause(question: str) -> bool:
    clauses = split_query_into_clauses(question, max_clauses=2)
    return len(clauses) >= 2


# ---------------------------------------------------------------------------
# Ranked-list merge helpers (fair-backfill to K)
# ---------------------------------------------------------------------------
def fair_backfill_segments(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    budget: int,
) -> list[Segment]:
    """Dedupe arch segments, take up to K, backfill with cosine.

    Returns list of segments of length <= budget.
    """
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


def stack_merge(specs: list[list[Segment]]) -> list[Segment]:
    """Concatenate specialist segment lists in order, preserving order and
    removing duplicates by index. specs[0] provides the first items."""
    seen: set[int] = set()
    out: list[Segment] = []
    for seg_list in specs:
        for s in seg_list:
            if s.index in seen:
                continue
            out.append(s)
            seen.add(s.index)
    return out


def merge_by_score(
    ensemble_outputs: dict[str, SpecialistOutput],
    names: tuple[str, ...],
    scaling: str = "sum",
) -> list[tuple[Segment, float]]:
    """Merge by summing per-specialist cosine scores (sum_cosine) or taking the
    max (max_cosine). Returns (segment, score) sorted desc."""
    pool: dict[int, dict] = {}
    for name in names:
        if name not in ensemble_outputs:
            continue
        so = ensemble_outputs[name]
        for rank, (seg, cos) in enumerate(zip(so.segments, so.cosine_scores)):
            entry = pool.setdefault(seg.index, {"segment": seg, "scores": []})
            entry["scores"].append(cos)
    merged: list[tuple[Segment, float]] = []
    for idx, e in pool.items():
        if scaling == "sum":
            s = sum(e["scores"])
        else:
            s = max(e["scores"])
        merged.append((e["segment"], s))
    merged.sort(key=lambda rc: -rc[1])
    return merged


# ---------------------------------------------------------------------------
# Critical-info overlay
# ---------------------------------------------------------------------------
def _main_ranked_with_scores_from_seglist(
    seg_list: list[Segment],
    cosine_segments: list[Segment],
    cosine_scores: list[float],
) -> list[tuple[Segment, float]]:
    """Build (seg, score) main_ranked list: provided segments at top
    (score=10..), then cosine top-K by score."""
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
    router_label: str
    router_composition: tuple[str, ...]
    speaker_fires: bool
    alias_fires: bool
    multi_clause: bool
    crit_ranked: list[tuple[int, float, Segment]]


def build_contexts(
    store: SegmentStore,
    specialists: dict,
    questions: list[dict],
    conv_speakers: dict[str, str],
    conv_aliases: dict[str, list[list[str]]],
    crit_store: CriticalInfoStore | None,
) -> list[QContext]:
    ctxs: list[QContext] = []
    for q in questions:
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])
        cat = q.get("category", "unknown")

        # Use v2f's embed cache
        q_emb = specialists["v2f"].embed_text(q_text)
        cos_res = store.search(q_emb, top_k=max(BUDGETS), conversation_id=conv_id)
        cos_segs = list(cos_res.segments)
        cos_scores = list(cos_res.scores)

        outputs = run_specialists(specialists, store, q_text, conv_id, q_emb)

        label = route_keyword_label(q_text)
        comp = COMPOSITION_FOR_LABEL.get(label, ("v2f",))

        speaker_fires = query_mentions_conv_user(q_text, conv_id, conv_speakers)
        alias_fires = query_alias_fires(q_text, conv_id, conv_aliases)
        multi_clause = query_multi_clause(q_text)

        # Critical-info retrieval (if crit_store available)
        crit = []
        if crit_store is not None:
            crit = crit_store.search_per_parent(
                q_emb,
                top_m=max(BUDGETS),
                conversation_id=conv_id,
                min_score=-1.0,
            )

        ctxs.append(
            QContext(
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
                speaker_fires=speaker_fires,
                alias_fires=alias_fires,
                multi_clause=multi_clause,
                crit_ranked=crit,
            )
        )
    return ctxs


# ---------------------------------------------------------------------------
# Variant builders — each returns set[int] (retrieved turn_ids) at budget K
# ---------------------------------------------------------------------------
def _speaker_transform_seglist(
    ctx: QContext, base_segments: list[Segment]
) -> list[Segment]:
    """If speaker fires, apply speaker_user_filter's effect: replace base
    segments with the speaker specialist's output (which filters assistants
    + appends user-only cosine). If not, return base unchanged."""
    if not ctx.speaker_fires:
        return base_segments
    sp = ctx.outputs.get("speaker_user_filter")
    if sp is None or not sp.segments:
        return base_segments
    # Stack-merge: speaker specialist output first, then base novel items.
    return stack_merge([list(sp.segments), base_segments])


def var_v2f(ctx: QContext, K: int) -> set[int]:
    segs = fair_backfill_segments(
        list(ctx.outputs["v2f"].segments), ctx.cosine_segments, K
    )
    return {s.turn_id for s in segs}


def var_speaker_filter_only(ctx: QContext, K: int) -> set[int]:
    """Apply speaker_user_filter when it fires; otherwise fall back to v2f."""
    if ctx.speaker_fires:
        base = list(ctx.outputs["speaker_user_filter"].segments)
    else:
        base = list(ctx.outputs["v2f"].segments)
    segs = fair_backfill_segments(base, ctx.cosine_segments, K)
    return {s.turn_id for s in segs}


def var_speaker_plus_router(ctx: QContext, K: int) -> set[int]:
    """Router's chosen specialist + speaker filter overlay."""
    comp = ctx.router_composition
    # Base: router's chosen composition, stack-merged
    if len(comp) == 1:
        base = list(ctx.outputs[comp[0]].segments)
    else:
        merged = merge_by_score(ctx.outputs, comp, scaling="sum")
        base = [seg for seg, _ in merged]
    base = _speaker_transform_seglist(ctx, base)
    segs = fair_backfill_segments(base, ctx.cosine_segments, K)
    return {s.turn_id for s in segs}


def var_speaker_plus_alias(ctx: QContext, K: int) -> set[int]:
    """alias_expand_v2f + speaker filter overlay."""
    base = list(ctx.outputs["alias_expand_v2f"].segments)
    if not base:
        base = list(ctx.outputs["v2f"].segments)
    base = _speaker_transform_seglist(ctx, base)
    segs = fair_backfill_segments(base, ctx.cosine_segments, K)
    return {s.turn_id for s in segs}


def var_speaker_plus_critical(ctx: QContext, K: int, crit_store_ok: bool) -> set[int]:
    """v2f + speaker + critical overlay."""
    base = list(ctx.outputs["v2f"].segments)
    base = _speaker_transform_seglist(ctx, base)
    if not crit_store_ok or not ctx.crit_ranked:
        segs = fair_backfill_segments(base, ctx.cosine_segments, K)
        return {s.turn_id for s in segs}
    main_ranked = _main_ranked_with_scores_from_seglist(
        base,
        ctx.cosine_segments,
        ctx.cosine_scores,
    )
    merged_segs = merge_always_top_m(
        main_ranked,
        ctx.crit_ranked,
        K,
        top_m=5,
        min_score=0.2,
    )
    return {s.turn_id for s in merged_segs}


def var_speaker_all_in(ctx: QContext, K: int, crit_store_ok: bool) -> set[int]:
    """speaker + ens_2 + critical + alias + context, stacked.

    Order:
      1. If speaker fires, speaker-filtered specialist first.
      2. ens_2 (v2f + type_enumerated) sum_cosine.
      3. alias_expand_v2f novel items.
      4. clause_plus_v2f novel items (if multi-clause).
      5. contextemb_w1_stacked novel items.
      6. critical-info overlay.
    """
    segs_order: list[list[Segment]] = []

    if ctx.speaker_fires:
        sp = ctx.outputs.get("speaker_user_filter")
        if sp and sp.segments:
            segs_order.append(list(sp.segments))

    # ens_2
    merged_2 = merge_by_score(ctx.outputs, ("v2f", "type_enumerated"), scaling="sum")
    segs_order.append([seg for seg, _ in merged_2])

    # alias
    alias_so = ctx.outputs.get("alias_expand_v2f")
    if alias_so and alias_so.segments:
        segs_order.append(list(alias_so.segments))

    # clause (if multi-clause)
    if ctx.multi_clause:
        clause_so = ctx.outputs.get("clause_plus_v2f")
        if clause_so and clause_so.segments:
            segs_order.append(list(clause_so.segments))

    # context-embedding
    ctx_so = ctx.outputs.get("contextemb_w1_stacked")
    if ctx_so and ctx_so.segments:
        segs_order.append(list(ctx_so.segments))

    base = stack_merge(segs_order)

    if not crit_store_ok or not ctx.crit_ranked:
        segs = fair_backfill_segments(base, ctx.cosine_segments, K)
        return {s.turn_id for s in segs}

    main_ranked = _main_ranked_with_scores_from_seglist(
        base,
        ctx.cosine_segments,
        ctx.cosine_scores,
    )
    merged_segs = merge_always_top_m(
        main_ranked,
        ctx.crit_ranked,
        K,
        top_m=5,
        min_score=0.2,
    )
    return {s.turn_id for s in merged_segs}


# --- K=50 reference / composition variants -------------------------------
def var_ens_all_plus_crit(ctx: QContext, K: int, crit_store_ok: bool) -> set[int]:
    """ens_5 sum_cosine + critical (prior max-effort reference)."""
    merged_5 = merge_by_score(
        ctx.outputs,
        (
            "v2f",
            "v2f_plus_types",
            "type_enumerated",
            "chain_with_scratchpad",
            "v2f_style_explicit",
        ),
        scaling="sum",
    )
    base = [seg for seg, _ in merged_5]
    if not crit_store_ok or not ctx.crit_ranked:
        segs = fair_backfill_segments(base, ctx.cosine_segments, K)
        return {s.turn_id for s in segs}
    main_ranked = _main_ranked_with_scores_from_seglist(
        base,
        ctx.cosine_segments,
        ctx.cosine_scores,
    )
    merged_segs = merge_always_top_m(
        main_ranked,
        ctx.crit_ranked,
        K,
        top_m=5,
        min_score=0.2,
    )
    return {s.turn_id for s in merged_segs}


def var_composition_v1(ctx: QContext, K: int, crit_store_ok: bool) -> set[int]:
    """6 wins WITHOUT speaker: router + ens_2 + alias + clause + context +
    critical."""
    segs_order: list[list[Segment]] = []
    # Router-picked composition first
    comp = ctx.router_composition
    if len(comp) == 1:
        segs_order.append(list(ctx.outputs[comp[0]].segments))
    else:
        m = merge_by_score(ctx.outputs, comp, scaling="sum")
        segs_order.append([seg for seg, _ in m])
    # ens_2
    merged_2 = merge_by_score(ctx.outputs, ("v2f", "type_enumerated"), scaling="sum")
    segs_order.append([seg for seg, _ in merged_2])
    # alias
    alias_so = ctx.outputs.get("alias_expand_v2f")
    if alias_so and alias_so.segments:
        segs_order.append(list(alias_so.segments))
    # clause
    if ctx.multi_clause:
        clause_so = ctx.outputs.get("clause_plus_v2f")
        if clause_so and clause_so.segments:
            segs_order.append(list(clause_so.segments))
    # context
    ctx_so = ctx.outputs.get("contextemb_w1_stacked")
    if ctx_so and ctx_so.segments:
        segs_order.append(list(ctx_so.segments))

    base = stack_merge(segs_order)
    if not crit_store_ok or not ctx.crit_ranked:
        segs = fair_backfill_segments(base, ctx.cosine_segments, K)
        return {s.turn_id for s in segs}
    main_ranked = _main_ranked_with_scores_from_seglist(
        base,
        ctx.cosine_segments,
        ctx.cosine_scores,
    )
    merged_segs = merge_always_top_m(
        main_ranked,
        ctx.crit_ranked,
        K,
        top_m=5,
        min_score=0.2,
    )
    return {s.turn_id for s in merged_segs}


def var_composition_v2_all(ctx: QContext, K: int, crit_store_ok: bool) -> set[int]:
    """7 wins: v1 + speaker (same as speaker_all_in but with router layer)."""
    segs_order: list[list[Segment]] = []
    # Speaker first (if fires)
    if ctx.speaker_fires:
        sp = ctx.outputs.get("speaker_user_filter")
        if sp and sp.segments:
            segs_order.append(list(sp.segments))
    # Router-picked composition
    comp = ctx.router_composition
    if len(comp) == 1:
        segs_order.append(list(ctx.outputs[comp[0]].segments))
    else:
        m = merge_by_score(ctx.outputs, comp, scaling="sum")
        segs_order.append([seg for seg, _ in m])
    # ens_2
    merged_2 = merge_by_score(ctx.outputs, ("v2f", "type_enumerated"), scaling="sum")
    segs_order.append([seg for seg, _ in merged_2])
    # alias
    alias_so = ctx.outputs.get("alias_expand_v2f")
    if alias_so and alias_so.segments:
        segs_order.append(list(alias_so.segments))
    # clause
    if ctx.multi_clause:
        clause_so = ctx.outputs.get("clause_plus_v2f")
        if clause_so and clause_so.segments:
            segs_order.append(list(clause_so.segments))
    # context
    ctx_so = ctx.outputs.get("contextemb_w1_stacked")
    if ctx_so and ctx_so.segments:
        segs_order.append(list(ctx_so.segments))

    base = stack_merge(segs_order)
    if not crit_store_ok or not ctx.crit_ranked:
        segs = fair_backfill_segments(base, ctx.cosine_segments, K)
        return {s.turn_id for s in segs}
    main_ranked = _main_ranked_with_scores_from_seglist(
        base,
        ctx.cosine_segments,
        ctx.cosine_scores,
    )
    merged_segs = merge_always_top_m(
        main_ranked,
        ctx.crit_ranked,
        K,
        top_m=5,
        min_score=0.2,
    )
    return {s.turn_id for s in merged_segs}


# --- Ablations (drop one from v2_all) ------------------------------------
def _var_drop(component: str):
    """Return a function that computes composition_v2_all minus `component`."""

    def f(ctx: QContext, K: int, crit_store_ok: bool) -> set[int]:
        segs_order: list[list[Segment]] = []
        if component != "speaker" and ctx.speaker_fires:
            sp = ctx.outputs.get("speaker_user_filter")
            if sp and sp.segments:
                segs_order.append(list(sp.segments))
        if component != "router":
            comp = ctx.router_composition
            if len(comp) == 1:
                segs_order.append(list(ctx.outputs[comp[0]].segments))
            else:
                m = merge_by_score(ctx.outputs, comp, scaling="sum")
                segs_order.append([seg for seg, _ in m])
        if component != "ens_2":
            merged_2 = merge_by_score(
                ctx.outputs, ("v2f", "type_enumerated"), scaling="sum"
            )
            segs_order.append([seg for seg, _ in merged_2])
        if component != "alias":
            alias_so = ctx.outputs.get("alias_expand_v2f")
            if alias_so and alias_so.segments:
                segs_order.append(list(alias_so.segments))
        if component != "clause" and ctx.multi_clause:
            clause_so = ctx.outputs.get("clause_plus_v2f")
            if clause_so and clause_so.segments:
                segs_order.append(list(clause_so.segments))
        if component != "context":
            ctx_so = ctx.outputs.get("contextemb_w1_stacked")
            if ctx_so and ctx_so.segments:
                segs_order.append(list(ctx_so.segments))

        base = stack_merge(segs_order)
        if component == "critical" or (not crit_store_ok) or (not ctx.crit_ranked):
            segs = fair_backfill_segments(base, ctx.cosine_segments, K)
            return {s.turn_id for s in segs}
        main_ranked = _main_ranked_with_scores_from_seglist(
            base,
            ctx.cosine_segments,
            ctx.cosine_scores,
        )
        merged_segs = merge_always_top_m(
            main_ranked,
            ctx.crit_ranked,
            K,
            top_m=5,
            min_score=0.2,
        )
        return {s.turn_id for s in merged_segs}

    return f


# --- Variant registry ----------------------------------------------------
# K=20 variants
K20_VARIANTS = [
    ("v2f", lambda c, k, s: var_v2f(c, k)),
    ("speaker_filter_only", lambda c, k, s: var_speaker_filter_only(c, k)),
    ("speaker_plus_router", lambda c, k, s: var_speaker_plus_router(c, k)),
    ("speaker_plus_alias", lambda c, k, s: var_speaker_plus_alias(c, k)),
    ("speaker_plus_critical", var_speaker_plus_critical),
    ("speaker_all_in", var_speaker_all_in),
]

# K=50 variants (plus ablations of composition_v2_all)
K50_VARIANTS = [
    ("v2f", lambda c, k, s: var_v2f(c, k)),
    ("ens_all_plus_crit", var_ens_all_plus_crit),
    ("composition_v1", var_composition_v1),
    ("composition_v2_all", var_composition_v2_all),
    ("drop_speaker", _var_drop("speaker")),
    ("drop_ens_2", _var_drop("ens_2")),
    ("drop_alias", _var_drop("alias")),
    ("drop_clause", _var_drop("clause")),
    ("drop_context", _var_drop("context")),
    ("drop_critical", _var_drop("critical")),
    ("drop_router", _var_drop("router")),
]


# ---------------------------------------------------------------------------
# Critical-info store builder (mirrors composition_eval)
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
    print(f"  classifying {len(target)} turns for critical-info...", flush=True)
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
        alt_embs = embed_texts_cached(client, embedder.embedding_cache, alt_texts)
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
# Per-dataset evaluator
# ---------------------------------------------------------------------------
def evaluate_dataset(
    ds_name: str,
    client: OpenAI,
    conv_speakers: dict[str, str],
    conv_aliases: dict[str, list[list[str]]],
    generator: CriticalInfoGenerator,
    embedder: Embedder,
) -> dict:
    cfg = DATASETS[ds_name]
    print(f"\n{'=' * 70}\n[{ds_name}]\n{'=' * 70}", flush=True)
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        qs = json.load(f)
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    if cfg["max_questions"]:
        qs = qs[: cfg["max_questions"]]
    print(f"  questions={len(qs)} segments={len(store.segments)}", flush=True)

    # Build critical-info store
    crit_store = build_critical_store(store, qs, generator, client, embedder)
    crit_store_ok = crit_store is not None

    # Build all specialists (ingest-time work may happen on first call:
    # speaker ID, alias extraction, context-embedding index)
    t_spec = time.time()
    print("  building specialists (cache-only)...", flush=True)
    specialists: dict = {}
    for name in ALL_SPECIALISTS:
        t0 = time.time()
        specialists[name] = build_specialist(name, store)
        print(f"    {name}: ready in {time.time() - t0:.1f}s", flush=True)
    print(f"  specialists built in {time.time() - t_spec:.1f}s", flush=True)

    # Reload conv_speakers + conv_aliases after specialist init (which may
    # have extracted new groups / identified new speakers for this dataset's
    # conversations).
    conv_speakers = load_conv_speakers()
    conv_aliases = load_conv_aliases()

    # Build per-question contexts
    print("  building per-question contexts...", flush=True)
    t_ctx = time.time()
    ctxs = build_contexts(
        store,
        specialists,
        qs,
        conv_speakers,
        conv_aliases,
        crit_store,
    )
    print(f"  contexts built in {time.time() - t_ctx:.1f}s", flush=True)

    # Save caches (any minor new writes)
    for arch in specialists.values():
        try:
            arch.save_caches()
        except Exception:
            pass

    # Evaluate all K=20 + K=50 variants per question
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
            "speaker_fires": ctx.speaker_fires,
            "alias_fires": ctx.alias_fires,
            "multi_clause": ctx.multi_clause,
            "recall": {},
        }
        if not ctx.source_ids:
            per_q_rows.append(row)
            continue
        # K=20 variants
        for var_name, fn in K20_VARIANTS:
            ids = fn(ctx, 20, crit_store_ok)
            r = compute_recall(ids, ctx.source_ids)
            row["recall"][f"{var_name}@20"] = round(r, 4)
        # K=50 variants
        for var_name, fn in K50_VARIANTS:
            ids = fn(ctx, 50, crit_store_ok)
            r = compute_recall(ids, ctx.source_ids)
            row["recall"][f"{var_name}@50"] = round(r, 4)
        per_q_rows.append(row)

    # Aggregate
    def _agg(rows: list[dict], key: str) -> float:
        vals = [r["recall"].get(key, 0.0) for r in rows if r["num_source_turns"] > 0]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    agg: dict[str, float] = {}
    for var_name, _ in K20_VARIANTS:
        agg[f"{var_name}@20"] = _agg(per_q_rows, f"{var_name}@20")
    for var_name, _ in K50_VARIANTS:
        agg[f"{var_name}@50"] = _agg(per_q_rows, f"{var_name}@50")

    # Per-category (K=50 composition_v2_all, K=20 speaker_all_in)
    by_cat: dict = defaultdict(lambda: defaultdict(list))
    for r in per_q_rows:
        if r["num_source_turns"] == 0:
            continue
        cat = r["category"]
        for var_name, _ in K20_VARIANTS:
            by_cat[cat][f"{var_name}@20"].append(r["recall"].get(f"{var_name}@20", 0.0))
        for var_name, _ in K50_VARIANTS:
            by_cat[cat][f"{var_name}@50"].append(r["recall"].get(f"{var_name}@50", 0.0))
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

    # Per-subset: speaker fires, alias fires, neither
    def _subset_agg(filter_fn) -> dict:
        subset = [r for r in per_q_rows if r["num_source_turns"] > 0 and filter_fn(r)]
        if not subset:
            return {"n": 0}
        res = {"n": len(subset)}
        for var_name, _ in K20_VARIANTS:
            vs = [r["recall"].get(f"{var_name}@20", 0.0) for r in subset]
            res[f"{var_name}@20"] = round(sum(vs) / len(vs), 4)
        for var_name, _ in K50_VARIANTS:
            vs = [r["recall"].get(f"{var_name}@50", 0.0) for r in subset]
            res[f"{var_name}@50"] = round(sum(vs) / len(vs), 4)
        return res

    subsets = {
        "speaker_fires": _subset_agg(lambda r: r["speaker_fires"]),
        "alias_fires": _subset_agg(lambda r: r["alias_fires"]),
        "speaker_and_alias": _subset_agg(
            lambda r: r["speaker_fires"] and r["alias_fires"]
        ),
        "speaker_not_alias": _subset_agg(
            lambda r: r["speaker_fires"] and not r["alias_fires"]
        ),
        "neither": _subset_agg(
            lambda r: not r["speaker_fires"] and not r["alias_fires"]
        ),
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
    L.append("# Final composition v2 — speaker-driven stacking\n")
    L.append(
        "Composition v2 enters **speaker_user_filter** (+8.3pp @ K=20 on "
        "LoCoMo when it fires) into the stack, alongside the other shipped "
        "narrow wins. Evaluated on 4 datasets at K=20 and K=50, fair-backfilled.\n"
    )
    L.append(f"\nElapsed: {total_elapsed:.0f}s.\n")

    # --- K=20 recall matrix ---
    L.append("\n## K=20 recall matrix\n")
    L.append("| Variant | " + " | ".join(DATASETS) + " | overall |")
    L.append("|---|" + "---|" * (len(DATASETS) + 1))
    for var_name, _ in K20_VARIANTS:
        row = [var_name]
        vals_w = []
        wt = 0
        for ds in DATASETS:
            if ds not in all_results:
                row.append("-")
                continue
            v = all_results[ds]["aggregated"].get(f"{var_name}@20", 0.0)
            n = all_results[ds]["n_with_gold"]
            row.append(f"{v:.4f}")
            vals_w.append(v * n)
            wt += n
        overall = (sum(vals_w) / wt) if wt else 0.0
        row.append(f"**{overall:.4f}**")
        L.append("| " + " | ".join(row) + " |")

    # --- K=50 recall matrix ---
    L.append("\n## K=50 recall matrix\n")
    L.append("| Variant | " + " | ".join(DATASETS) + " | overall |")
    L.append("|---|" + "---|" * (len(DATASETS) + 1))
    for var_name, _ in K50_VARIANTS:
        row = [var_name]
        vals_w = []
        wt = 0
        for ds in DATASETS:
            if ds not in all_results:
                row.append("-")
                continue
            v = all_results[ds]["aggregated"].get(f"{var_name}@50", 0.0)
            n = all_results[ds]["n_with_gold"]
            row.append(f"{v:.4f}")
            vals_w.append(v * n)
            wt += n
        overall = (sum(vals_w) / wt) if wt else 0.0
        row.append(f"**{overall:.4f}**")
        L.append("| " + " | ".join(row) + " |")

    # --- Ablations table (K=50 LoCoMo) ---
    L.append("\n## Ablation: drop-one from composition_v2_all (LoCoMo K=50)\n")
    loc = all_results.get("locomo_30q", {}).get("aggregated", {})
    v2_all = loc.get("composition_v2_all@50", 0.0)
    L.append("| Drop | r@50 | Δ vs v2_all |")
    L.append("|---|---:|---:|")
    L.append(f"| (none: v2_all) | {v2_all:.4f} | — |")
    for drop_name in (
        "drop_speaker",
        "drop_ens_2",
        "drop_alias",
        "drop_clause",
        "drop_context",
        "drop_critical",
        "drop_router",
    ):
        r = loc.get(f"{drop_name}@50", 0.0)
        L.append(f"| {drop_name} | {r:.4f} | {r - v2_all:+.4f} |")

    # --- Subset analysis (LoCoMo K=20) ---
    L.append("\n## LoCoMo per-subset delta analysis at K=20\n")
    L.append(
        "Comparing speaker_all_in vs v2f on subsets defined by which "
        "narrow win fires.\n"
    )
    loc_sub = all_results.get("locomo_30q", {}).get("subset_aggregates", {})
    L.append("| Subset | n | v2f@20 | speaker_all_in@20 | Δ |")
    L.append("|---|---:|---:|---:|---:|")
    for sub_name in (
        "speaker_fires",
        "alias_fires",
        "speaker_and_alias",
        "speaker_not_alias",
        "neither",
    ):
        d = loc_sub.get(sub_name, {})
        n = d.get("n", 0)
        if n == 0:
            L.append(f"| {sub_name} | 0 | - | - | - |")
            continue
        v = d.get("v2f@20", 0.0)
        s = d.get("speaker_all_in@20", 0.0)
        L.append(f"| {sub_name} | {n} | {v:.4f} | {s:.4f} | {s - v:+.4f} |")

    # --- Per-dataset K=20 ship table ---
    L.append("\n## Per-(dataset, K) production recipes\n")
    L.append(
        "Per-cell best variant. K=20 draws from K20_VARIANTS; K=50 draws "
        "from K50_VARIANTS.\n"
    )
    L.append("| Dataset | K | Best variant | Recall | Δ vs v2f |")
    L.append("|---|---:|---|---:|---:|")
    for ds in DATASETS:
        if ds not in all_results:
            continue
        agg = all_results[ds]["aggregated"]
        for K, VARIANTS in ((20, K20_VARIANTS), (50, K50_VARIANTS)):
            v2f_r = agg.get(f"v2f@{K}", 0.0)
            best_var = "v2f"
            best_r = v2f_r
            for var_name, _ in VARIANTS:
                r = agg.get(f"{var_name}@{K}", 0.0)
                if r > best_r + 1e-9:
                    best_r = r
                    best_var = var_name
            L.append(
                f"| {ds} | {K} | {best_var} | {best_r:.4f} | {best_r - v2f_r:+.4f} |"
            )

    # --- Decision rules ---
    L.append("\n## Decision rules\n")
    v2f_k20 = loc.get("v2f@20", 0.0)
    sp_all_k20 = loc.get("speaker_all_in@20", 0.0)
    delta_sp = sp_all_k20 - v2f_k20
    L.append(f"- LoCoMo K=20 v2f = {v2f_k20:.4f}")
    L.append(f"- LoCoMo K=20 speaker_all_in = {sp_all_k20:.4f}")
    L.append(f"- Δ speaker_all_in vs v2f = {delta_sp:+.4f}")
    if delta_sp >= 0.05:
        L.append(
            f"  => **NEW K=20 SHIP (LoCoMo): speaker_all_in** "
            f"(+{delta_sp * 100:.1f}pp >= 5pp threshold)"
        )
    else:
        L.append(
            "  => speaker_all_in does NOT beat v2f by 5pp on LoCoMo K=20; "
            "look at speaker_filter_only or speaker_plus_critical."
        )

    v2_all_50 = loc.get("composition_v2_all@50", 0.0)
    ens_crit_50 = loc.get("ens_all_plus_crit@50", 0.0)
    d2 = v2_all_50 - ens_crit_50
    L.append(f"\n- LoCoMo K=50 ens_all_plus_crit (prior reference) = {ens_crit_50:.4f}")
    L.append(f"- LoCoMo K=50 composition_v2_all = {v2_all_50:.4f}")
    L.append(f"- Δ composition_v2_all vs ens_all_plus_crit = {d2:+.4f}")
    if v2_all_50 > 0.922:
        L.append(
            "  => **NEW K=50 CEILING (LoCoMo): composition_v2_all** "
            "(beats prior 0.922)"
        )
    elif d2 > 0.01:
        L.append(
            "  => composition_v2_all beats prior max-effort by >1pp "
            "though short of 0.922 absolute."
        )
    else:
        L.append("  => composition_v2_all did NOT clearly beat prior ceiling.")

    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    client = OpenAI(timeout=90.0)
    # Critical-info generator (pinned prompt version + threading)
    generator = CriticalInfoGenerator(
        client=client,
        prompt_version=CRITICAL_PROMPT_VERSION,
        max_workers=4,
    )
    embedder = Embedder(client=client)

    conv_speakers = load_conv_speakers()
    conv_aliases = load_conv_aliases()
    print(
        f"Loaded {len(conv_speakers)} conv_speaker entries, "
        f"{len(conv_aliases)} conv_alias entries.",
        flush=True,
    )

    t0 = time.time()
    all_results: dict[str, dict] = {}
    for ds in DATASETS:
        try:
            all_results[ds] = evaluate_dataset(
                ds,
                client,
                conv_speakers,
                conv_aliases,
                generator,
                embedder,
            )
        except Exception as e:
            print(f"[{ds}] FAILED: {e}", flush=True)
            import traceback

            traceback.print_exc()
            all_results[ds] = {"error": str(e)}

    total_elapsed = time.time() - t0

    # Save JSON
    raw_path = RESULTS_DIR / "final_composition_v2.json"
    with open(raw_path, "w") as f:
        json.dump(
            {
                "datasets": list(DATASETS.keys()),
                "k20_variants": [v[0] for v in K20_VARIANTS],
                "k50_variants": [v[0] for v in K50_VARIANTS],
                "results": all_results,
                "total_elapsed_s": round(total_elapsed, 1),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved: {raw_path}", flush=True)

    # Markdown report
    md = render_markdown(all_results, total_elapsed)
    md_path = RESULTS_DIR / "final_composition_v2.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved: {md_path}", flush=True)

    # Print headline
    print("\n" + "=" * 100)
    print("FINAL COMPOSITION V2 — HEADLINE")
    print("=" * 100)
    for ds in DATASETS:
        agg = all_results.get(ds, {}).get("aggregated", {})
        if not agg:
            continue
        print(f"\n{ds}:")
        for var_name, _ in K20_VARIANTS:
            print(f"  K=20 {var_name:25s} = {agg.get(var_name + '@20', 0.0):.4f}")
        for var_name, _ in K50_VARIANTS:
            print(f"  K=50 {var_name:25s} = {agg.get(var_name + '@50', 0.0):.4f}")


if __name__ == "__main__":
    main()
