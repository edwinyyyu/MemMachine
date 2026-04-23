"""EventMemory intent_parser with REAL TIMESTAMPS.

Copy of intent_em.py, extended so that the `temporal_relation` constraint in
the parsed plan is translated into a real-datetime `property_filter` on
EM's base `timestamp` field (now populated from LoCoMo's
`session_{i}_date_time` by em_setup_rts.py).

Mapping notes
-------------
  Plan's `constraints.temporal_relation` has schema:
      {"marker": "after|before|during|when|between", "reference": "<free text>"}
  The reference is free-form ("4 years ago", "last weekend", "Monday
  meeting"). Resolving arbitrary references reliably requires an extra LLM
  call which we want to avoid for Step 4. Instead we implement a lightweight
  deterministic interpreter over LoCoMo's calendar:

    - If `reference` parses as an absolute date (via a few regex fallbacks
      or dateutil), build a [ref-window, ref+window] range. Default window
      = 7 days, or `relative_window_days` if provided in the plan.
    - If `reference` is a relative offset like "N <unit> ago", compute
      `now - N*unit` where `now` is the last session timestamp in the
      conversation (a reasonable proxy for "now" in the dataset).
    - `marker` ∈ {"before","after","during","when","between"} maps to the
      appropriate comparison operator:
        after  -> timestamp >= ref_start
        before -> timestamp <= ref_end
        during/when/between -> ref_start <= timestamp <= ref_end

  If we can't resolve the reference, we do NOT apply a temporal filter
  (fail-safe — don't zero recall).

  Speaker handling: unchanged from intent_em.py.

Variants defined here
---------------------
  intent_rts_full            speaker + temporal filter when parsed, v2f_speakerformat cues
  intent_rts_temporal_only   temporal filter only (no speaker), v2f cues
  intent_rts_speaker_only    speaker filter only, v2f_speakerformat cues (control)

Each variant reuses the pre-computed intent_parse_cache (NO new LLM plan
calls). Cue caches point at the speakerformat retuned cache for the speaker
variants and the em_v2f cache for the temporal_only variant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

from em_architectures import (
    EMHit,
    V2F_MODEL,
    V2F_PROMPT,
    _MergedLLMCache,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _query_em,
    format_primer_context,
    parse_v2f_cues,
)
from em_retuned_cue_gen import (
    build_v2f_speakerformat_prompt,
    parse_cues as parse_retuned_cues,
)
from intent_em import (
    resolve_speaker_to_name,
)


# ---------------------------------------------------------------------------
# Temporal reference resolution
# ---------------------------------------------------------------------------

_MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_UNIT_DAYS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
}


@dataclass(frozen=True)
class TemporalWindow:
    start: datetime
    end: datetime
    source: str  # how it was resolved (for logging)


def _maybe_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


def _parse_absolute_date(ref: str) -> datetime | None:
    """Try to parse `ref` as an absolute calendar date."""
    r = ref.strip().lower()
    # Year only: "2022", "2023"
    if re.fullmatch(r"20\d{2}", r):
        return datetime(int(r), 1, 1, tzinfo=UTC)
    # "4 May 2023" / "May 4, 2023" / "May 2023" / "4 May, 2023"
    m = re.fullmatch(
        r"(\d{1,2})\s+([a-zA-Z]+)[,\s]*(20\d{2})", r
    )
    if m:
        d, mo, y = int(m.group(1)), m.group(2), int(m.group(3))
        if mo in _MONTH_NAMES:
            return datetime(y, _MONTH_NAMES[mo], d, tzinfo=UTC)
    m = re.fullmatch(
        r"([a-zA-Z]+)\s+(\d{1,2})[,\s]*(20\d{2})", r
    )
    if m:
        mo, d, y = m.group(1), int(m.group(2)), int(m.group(3))
        if mo in _MONTH_NAMES:
            return datetime(y, _MONTH_NAMES[mo], d, tzinfo=UTC)
    m = re.fullmatch(r"([a-zA-Z]+)\s+(20\d{2})", r)
    if m:
        mo, y = m.group(1), int(m.group(2))
        if mo in _MONTH_NAMES:
            return datetime(y, _MONTH_NAMES[mo], 1, tzinfo=UTC)
    # Month only: "May", "august"
    if r in _MONTH_NAMES:
        # Attach to 1 Jan — not precise, we'll widen the window later.
        return None
    return None


def _parse_relative_offset(
    ref: str, anchor_now: datetime
) -> datetime | None:
    """Parse 'N <unit> ago' / 'last <unit>' / 'N <unit> later' etc."""
    r = ref.strip().lower()
    m = re.fullmatch(r"(\d+)\s+([a-zA-Z]+)\s+ago", r)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit in _UNIT_DAYS:
            return anchor_now - timedelta(days=n * _UNIT_DAYS[unit])
    m = re.fullmatch(r"last\s+([a-zA-Z]+)", r)
    if m:
        unit = m.group(1)
        if unit in _UNIT_DAYS:
            return anchor_now - timedelta(days=_UNIT_DAYS[unit])
    if r in ("now", "today", "currently"):
        return anchor_now
    if r in ("yesterday",):
        return anchor_now - timedelta(days=1)
    return None


def resolve_temporal_window(
    tr: dict,
    *,
    anchor_now: datetime,
    default_window_days: int = 7,
) -> TemporalWindow | None:
    """Translate a parsed `temporal_relation` into a concrete time window.

    tr = {"marker": "after|before|during|when|between", "reference": "..."}
    anchor_now = the "now" of the conversation (for relative offsets).
    """
    ref = (tr.get("reference") or "").strip()
    marker = (tr.get("marker") or "").strip().lower()
    if not ref:
        return None

    abs_dt = _parse_absolute_date(ref)
    if abs_dt is None:
        rel_dt = _parse_relative_offset(ref, anchor_now)
        if rel_dt is None:
            return None
        abs_dt = rel_dt
        source = f"relative({ref!r})"
    else:
        source = f"absolute({ref!r})"

    half = timedelta(days=default_window_days)
    if marker == "after":
        start = abs_dt
        end = anchor_now + timedelta(days=365)  # far future upper bound
    elif marker == "before":
        start = datetime(2000, 1, 1, tzinfo=UTC)
        end = abs_dt
    else:
        # during/when/between/unknown -> bracket window
        start = abs_dt - half
        end = abs_dt + half
    return TemporalWindow(start=start, end=end, source=source)


# ---------------------------------------------------------------------------
# Filter building
# ---------------------------------------------------------------------------
def build_rts_filter(
    plan: dict,
    question: str,
    conversation_id: str,
    speaker_map: dict[str, dict[str, str]],
    *,
    apply_speaker: bool,
    apply_temporal: bool,
    anchor_now: datetime,
) -> tuple[FilterExpr | None, dict]:
    """Build a (speaker ∧ temporal) filter per the flags.

    Returns (filter_expr, analysis).
    """
    constraints = plan.get("constraints", {}) or {}
    analysis: dict = {
        "detected": [],
        "applied": [],
        "dropped": [],
        "speaker_resolved": None,
        "matched_side": None,
        "temporal_window": None,
    }

    filters: list[FilterExpr] = []

    sp = constraints.get("speaker")
    if sp:
        analysis["detected"].append("speaker")
        if apply_speaker:
            matched_name, side = resolve_speaker_to_name(
                sp, question, conversation_id, speaker_map
            )
            if matched_name:
                filters.append(
                    Comparison(
                        field="context.source", op="=", value=matched_name
                    )
                )
                analysis["applied"].append("speaker")
                analysis["speaker_resolved"] = matched_name
                analysis["matched_side"] = side
            else:
                analysis["dropped"].append(
                    f"speaker:{sp!r} (no conv match / not in query tokens)"
                )
        else:
            analysis["dropped"].append("speaker (variant disables)")

    tr = constraints.get("temporal_relation")
    if tr:
        analysis["detected"].append("temporal_relation")
        if apply_temporal:
            window = resolve_temporal_window(tr, anchor_now=anchor_now)
            if window is not None:
                filters.append(
                    And(
                        left=Comparison(
                            field="timestamp", op=">=", value=window.start
                        ),
                        right=Comparison(
                            field="timestamp", op="<=", value=window.end
                        ),
                    )
                )
                analysis["applied"].append("temporal_relation")
                analysis["temporal_window"] = {
                    "start": window.start.isoformat(),
                    "end": window.end.isoformat(),
                    "source": window.source,
                    "marker": tr.get("marker"),
                    "reference": tr.get("reference"),
                }
            else:
                analysis["dropped"].append(
                    f"temporal_relation:{tr!r} (could not resolve reference)"
                )
        else:
            analysis["dropped"].append("temporal_relation (variant disables)")

    if not filters:
        return None, analysis
    expr: FilterExpr = filters[0]
    for f in filters[1:]:
        expr = And(left=expr, right=f)
    return expr, analysis


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
async def _query_em_filtered(
    memory: EventMemory,
    text: str,
    *,
    vector_search_limit: int,
    property_filter: FilterExpr | None,
) -> list[EMHit]:
    qr = await memory.query(
        query=text,
        vector_search_limit=vector_search_limit,
        expand_context=0,
        property_filter=property_filter,
    )
    hits: list[EMHit] = []
    for sc in qr.scored_segment_contexts:
        for seg in sc.segments:
            hits.append(
                EMHit(
                    turn_id=int(seg.properties.get("turn_id", -1)),
                    score=sc.score,
                    seed_segment_uuid=sc.seed_segment_uuid,
                    role=str(seg.properties.get("role", "")),
                    text=seg.block.text,
                )
            )
    return hits


async def _generate_v2f_cues(
    memory: EventMemory,
    question: str,
    *,
    llm_cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[str], dict]:
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text}
        for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)
    prompt = V2F_PROMPT.format(question=question, context_section=context_section)
    cached = llm_cache.get(V2F_MODEL, prompt)
    cache_hit = cached is not None
    if cached is None:
        if openai_client is None:
            return [], {"cues": [], "cache_hit": False}
        resp = await openai_client.chat.completions.create(
            model=V2F_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        cached = resp.choices[0].message.content or ""
        llm_cache.put(V2F_MODEL, prompt, cached)
    cues = parse_v2f_cues(cached, max_cues=2)
    return cues, {"cues": cues, "cache_hit": cache_hit}


async def _generate_speakerformat_cues(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    llm_cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[str], dict]:
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text}
        for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)
    p_user, p_asst = participants
    prompt = build_v2f_speakerformat_prompt(
        question, context_section, p_user, p_asst
    )
    cached = llm_cache.get(V2F_MODEL, prompt)
    cache_hit = cached is not None
    if cached is None:
        if openai_client is None:
            return [], {"cues": [], "cache_hit": False}
        resp = await openai_client.chat.completions.create(
            model=V2F_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        cached = resp.choices[0].message.content or ""
        llm_cache.put(V2F_MODEL, prompt, cached)
    cues = parse_retuned_cues(cached, max_cues=2)
    return cues, {"cues": cues, "cache_hit": cache_hit}


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
@dataclass
class IntentRTSResult:
    hits: list[EMHit]
    metadata: dict


async def _run_variant(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    plan: dict,
    speaker_map: dict[str, dict[str, str]],
    participants: tuple[str, str],
    anchor_now: datetime,
    apply_speaker: bool,
    apply_temporal: bool,
    cue_style: str,  # "speakerformat" | "v2f" | "none"
    variant_name: str,
    v2f_cache: _MergedLLMCache | None,
    speakerfmt_cache: _MergedLLMCache | None,
    openai_client,
) -> IntentRTSResult:
    filter_expr, analysis = build_rts_filter(
        plan, question, conversation_id, speaker_map,
        apply_speaker=apply_speaker,
        apply_temporal=apply_temporal,
        anchor_now=anchor_now,
    )
    primer_filtered = await _query_em_filtered(
        memory, question, vector_search_limit=K, property_filter=filter_expr
    )
    cues: list[str] = []
    cue_meta: dict = {"cues": [], "cache_hit": False}
    if cue_style == "v2f" and v2f_cache is not None:
        cues, cue_meta = await _generate_v2f_cues(
            memory, question, llm_cache=v2f_cache, openai_client=openai_client
        )
    elif cue_style == "speakerformat" and speakerfmt_cache is not None:
        cues, cue_meta = await _generate_speakerformat_cues(
            memory, question, participants,
            llm_cache=speakerfmt_cache, openai_client=openai_client,
        )
    cue_batches: list[list[EMHit]] = []
    for cue in cues[:2]:
        cue_batches.append(
            await _query_em_filtered(
                memory, cue, vector_search_limit=K, property_filter=filter_expr
            )
        )
    merged = _merge_by_max_score([primer_filtered, *cue_batches])

    # Safety backfill: if filter is too narrow, pull unfiltered.
    if filter_expr is not None and len(_dedupe_by_turn_id(merged)) < K:
        unfiltered = await _query_em_filtered(
            memory, question, vector_search_limit=K, property_filter=None
        )
        merged = _merge_by_max_score([merged, unfiltered])
        analysis.setdefault("applied", []).append("backfill_unfiltered")

    return IntentRTSResult(
        hits=merged[:K],
        metadata={
            "variant": variant_name,
            "plan": plan,
            "filter_analysis": analysis,
            "cues": cues,
            "cache_hit": cue_meta.get("cache_hit", False),
            "applied_filter": filter_expr is not None,
        },
    )


async def intent_rts_full(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    plan: dict,
    speaker_map: dict[str, dict[str, str]],
    participants: tuple[str, str],
    anchor_now: datetime,
    speakerfmt_cache: _MergedLLMCache,
    openai_client,
) -> IntentRTSResult:
    """Speaker filter + temporal filter + v2f_speakerformat cues."""
    return await _run_variant(
        memory, question, conversation_id,
        K=K, plan=plan, speaker_map=speaker_map, participants=participants,
        anchor_now=anchor_now,
        apply_speaker=True,
        apply_temporal=True,
        cue_style="speakerformat",
        variant_name="intent_rts_full",
        v2f_cache=None,
        speakerfmt_cache=speakerfmt_cache,
        openai_client=openai_client,
    )


async def intent_rts_temporal_only(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    plan: dict,
    speaker_map: dict[str, dict[str, str]],
    participants: tuple[str, str],
    anchor_now: datetime,
    v2f_cache: _MergedLLMCache,
    openai_client,
) -> IntentRTSResult:
    """Temporal filter only (NO speaker) + v2f cues."""
    return await _run_variant(
        memory, question, conversation_id,
        K=K, plan=plan, speaker_map=speaker_map, participants=participants,
        anchor_now=anchor_now,
        apply_speaker=False,
        apply_temporal=True,
        cue_style="v2f",
        variant_name="intent_rts_temporal_only",
        v2f_cache=v2f_cache,
        speakerfmt_cache=None,
        openai_client=openai_client,
    )


async def intent_rts_speaker_only(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    plan: dict,
    speaker_map: dict[str, dict[str, str]],
    participants: tuple[str, str],
    anchor_now: datetime,
    speakerfmt_cache: _MergedLLMCache,
    openai_client,
) -> IntentRTSResult:
    """Speaker filter only (NO temporal) + v2f_speakerformat cues.

    Control variant: on LoCoMo this should match em_two_speaker_filter up to
    LLM-parse noise. Useful as a regression check for the RTS collections
    themselves.
    """
    return await _run_variant(
        memory, question, conversation_id,
        K=K, plan=plan, speaker_map=speaker_map, participants=participants,
        anchor_now=anchor_now,
        apply_speaker=True,
        apply_temporal=False,
        cue_style="speakerformat",
        variant_name="intent_rts_speaker_only",
        v2f_cache=None,
        speakerfmt_cache=speakerfmt_cache,
        openai_client=openai_client,
    )
