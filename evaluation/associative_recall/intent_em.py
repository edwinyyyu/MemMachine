"""EventMemory-native intent_parser: parsed intent → EM property_filter.

Unlike the SS-era `intent_parser_full` which applied constraint signals as
stacked score bonuses, this variant translates parsed constraints into
FIRST-CLASS `property_filter` arguments to `EventMemory.query(...)`.

Schema available for filtering on the LoCoMo-30 EM ingestion
(`evaluation/associative_recall/em_setup.py`):
  - `context.source`           speaker name (always set via MessageContext)
  - `context.type`              always "message" (useless as a filter)
  - `timestamp`                 EM base field (synthesized: 2023-01-01 +
                               60s*turn_id). LoCoMo temporal references
                               ("4 years ago", "last weekend") do NOT map
                               to this synthesized wall-clock; applying a
                               timestamp filter would DROP all matches.
                               We therefore skip temporal filtering.
  - user metadata m.role / m.turn_id:  stored but NOT payload-indexed;
                               for two-speaker LoCoMo, role filtering
                               duplicates context.source. Skipped.

Concretely, only the speaker constraint translates to a cleanly-defined
EM filter on this corpus. The remaining constraints (negation, answer_form,
intent_type) either have no schema support OR would need ingest-time
annotation that the brief forbids us from regenerating.

For negation we DO keep a small post-retrieval score nudge (NOT a
filter) toward turns that contain explicit negation markers; this is
score-level, applied after property_filter has already executed.

Variants
--------
  intent_em_speaker_only          parsed speaker ONLY → EM property_filter
                                  on context.source. Otherwise v2f (raw
                                  question). Sanity control against
                                  em_two_speaker_filter (should match or
                                  mildly differ due to LLM vs regex
                                  speaker detection).
  intent_em_full_filter           parsed speaker → EM filter, PLUS
                                  post-retrieval negation score bonus when
                                  constraints.negation is true. v2f cues
                                  on top of filtered retrieval.
  intent_em_filter_no_cues        speaker filter applied to raw question
                                  only; NO v2f cue gen. Analog of
                                  em_two_speaker_query_only.
  intent_em_with_speakerformat_cues
                                  speaker filter + v2f_speakerformat cues
                                  (from em_retuned_cue_gen.py). Tests
                                  whether retuned cues compose with a
                                  hard filter.

Speaker resolution: the LLM's `constraints.speaker` is a NAME. We map
it to the conversation's user/assistant using
`conversation_two_speakers.json`. If the parsed name doesn't appear in
the query's raw tokens (defensive against LLM hallucination), we drop
the constraint.

Cache: reuses the existing `intent_parse_cache.json` produced by
`intent_parser.py` (30/30 LoCoMo-30 queries already parsed). No new
intent-parse LLM calls needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from em_architectures import (
    V2F_MODEL,
    V2F_PROMPT,
    EMHit,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _MergedLLMCache,
    _query_em,
    format_primer_context,
    parse_v2f_cues,
)
from em_retuned_cue_gen import (
    build_v2f_speakerformat_prompt,
)
from em_retuned_cue_gen import (
    parse_cues as parse_retuned_cues,
)
from memmachine_server.common.filter.filter_parser import Comparison, FilterExpr
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from speaker_attributed import extract_name_mentions

CACHE_DIR = Path(__file__).resolve().parent / "cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
INTENT_PARSE_CACHE_FILE = CACHE_DIR / "intent_parse_cache.json"
CONV_TWO_SPEAKERS_FILE = RESULTS_DIR / "conversation_two_speakers.json"


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------
def load_intent_parse_cache() -> dict[str, dict]:
    with open(INTENT_PARSE_CACHE_FILE) as f:
        return json.load(f)


def load_two_speaker_map() -> dict[str, dict[str, str]]:
    with open(CONV_TWO_SPEAKERS_FILE) as f:
        data = json.load(f)
    return data.get("speakers", {}) or {}


# ---------------------------------------------------------------------------
# Intent → EM filter mapping
# ---------------------------------------------------------------------------
def resolve_speaker_to_name(
    parsed_speaker: str | None,
    question: str,
    conversation_id: str,
    speaker_map: dict[str, dict[str, str]],
) -> tuple[str | None, str | None]:
    """Map parsed speaker name to a concrete conversation participant.

    Returns (matched_name, side) where side in {"user", "assistant"}.
    Returns (None, None) if no safe match. Defensive checks:
      - parsed name must appear in the raw query tokens.
      - parsed name first-word must match user/assistant first-word.
    """
    if not parsed_speaker:
        return None, None
    pair = speaker_map.get(conversation_id, {})
    user_name = (pair.get("user") or "").strip()
    asst_name = (pair.get("assistant") or "").strip()

    parsed_lower = parsed_speaker.lower().strip()

    # Defensive: parsed speaker must appear in the query.
    q_mentions = {t.lower() for t in extract_name_mentions(question)}
    q_tokens = {t.lower() for t in re.findall(r"[A-Za-z]+", question)}
    first = parsed_lower.split()[0] if parsed_lower.split() else parsed_lower
    if parsed_lower not in q_mentions and first not in q_tokens:
        return None, None

    def _first_match(a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a.lower() == b.lower():
            return True
        ap = a.lower().split()
        bp = b.lower().split()
        return bool(ap and bp and ap[0] == bp[0])

    if user_name and _first_match(parsed_speaker, user_name):
        return user_name, "user"
    if asst_name and _first_match(parsed_speaker, asst_name):
        return asst_name, "assistant"
    return None, None


def build_em_filter(
    plan: dict,
    question: str,
    conversation_id: str,
    speaker_map: dict[str, dict[str, str]],
) -> tuple[FilterExpr | None, dict]:
    """Translate a structured intent plan into an EM property_filter.

    Returns (filter_expr, analysis) where `analysis` records which
    constraints were DETECTED in the plan vs which were actually
    APPLIED as filters (for post-hoc firing analysis).
    """
    constraints = plan.get("constraints", {}) or {}
    analysis: dict = {
        "detected": [],
        "applied": [],
        "speaker_resolved": None,
        "matched_side": None,
        "dropped": [],  # constraints present in plan but dropped (why)
    }

    sp = constraints.get("speaker")
    if sp:
        analysis["detected"].append("speaker")
    tr = constraints.get("temporal_relation")
    if tr:
        analysis["detected"].append("temporal_relation")
    if constraints.get("negation"):
        analysis["detected"].append("negation")
    af = constraints.get("answer_form")
    if af:
        analysis["detected"].append(f"answer_form:{af}")
    if plan.get("needs_aggregation"):
        analysis["detected"].append("needs_aggregation")

    filter_expr: FilterExpr | None = None

    # Speaker → Comparison(context.source = name)
    if sp:
        matched_name, side = resolve_speaker_to_name(
            sp, question, conversation_id, speaker_map
        )
        if matched_name:
            filter_expr = Comparison(field="context.source", op="=", value=matched_name)
            analysis["applied"].append("speaker")
            analysis["speaker_resolved"] = matched_name
            analysis["matched_side"] = side
        else:
            analysis["dropped"].append(
                f"speaker:{sp!r} (no conv match or not in query tokens)"
            )

    # temporal_relation: intentionally skipped -- LoCoMo references are
    # real-world-relative, EM timestamps are synthesized per-turn spacing
    # rooted at 2023-01-01. A timestamp filter here would zero out recall.
    if tr:
        analysis["dropped"].append(
            "temporal_relation (EM timestamps are synthesized, not LoCoMo-aligned)"
        )

    # negation: not a filter; handled as a score nudge in the executor.
    # quantity_bound / answer_form / intent_type: no EM schema support.

    return filter_expr, analysis


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
_NEG_RE = re.compile(
    r"\b("
    r"not|never|didn'?t|don'?t|doesn'?t|won'?t|wouldn'?t|can'?t|cannot|"
    r"couldn'?t|refuse[ds]?|decline[ds]?|rejected?|avoid(?:ed|ing)?|"
    r"against|no longer|stop(?:ped)?|denied|deny|nope"
    r")\b",
    re.IGNORECASE,
)


def has_negation_marker(text: str) -> bool:
    return bool(_NEG_RE.search(text))


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
    """Raw v2f cue generation (matches em_architectures.em_v2f's prompt).

    Uses primer retrieval WITHOUT filter to build context section (same
    as em_v2f). Only filter is applied to the merge candidates, not to
    the primer context.
    """
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text} for h in primer_hits
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
    """V2fSpeakerFormat cue generation (each cue begins with '<name>: ')."""
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text} for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)
    p_user, p_asst = participants
    prompt = build_v2f_speakerformat_prompt(question, context_section, p_user, p_asst)
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


def _apply_negation_nudge(hits: list[EMHit], bonus: float = 0.02) -> list[EMHit]:
    """Post-retrieval score nudge for turns with negation markers."""
    nudged: list[EMHit] = []
    for h in hits:
        if has_negation_marker(h.text):
            nudged.append(
                EMHit(
                    turn_id=h.turn_id,
                    score=h.score + bonus,
                    seed_segment_uuid=h.seed_segment_uuid,
                    role=h.role,
                    text=h.text,
                )
            )
        else:
            nudged.append(h)
    nudged.sort(key=lambda x: -x.score)
    return nudged


# ---------------------------------------------------------------------------
# Public architectures
# ---------------------------------------------------------------------------
@dataclass
class IntentEMResult:
    hits: list[EMHit]
    metadata: dict


async def intent_em_speaker_only(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    plan: dict,
    speaker_map: dict[str, dict[str, str]],
    llm_cache: _MergedLLMCache,
    openai_client,
) -> IntentEMResult:
    """Parsed speaker → EM filter. v2f-with-filter on top.

    - Raw-question EM.query with filter → filtered primer
    - v2f cues generated from unfiltered primer context (same as em_v2f)
    - Cue EM.query calls ALSO apply the filter
    - Merge by max score
    """
    filter_expr, analysis = build_em_filter(
        plan, question, conversation_id, speaker_map
    )

    # Primer (filtered if filter is set)
    primer_filtered = await _query_em_filtered(
        memory, question, vector_search_limit=K, property_filter=filter_expr
    )

    cues, cue_meta = await _generate_v2f_cues(
        memory, question, llm_cache=llm_cache, openai_client=openai_client
    )

    cue_batches: list[list[EMHit]] = []
    for cue in cues[:2]:
        cue_batches.append(
            await _query_em_filtered(
                memory,
                cue,
                vector_search_limit=K,
                property_filter=filter_expr,
            )
        )
    merged = _merge_by_max_score([primer_filtered, *cue_batches])[:K]

    return IntentEMResult(
        hits=merged,
        metadata={
            "variant": "intent_em_speaker_only",
            "plan": plan,
            "filter_analysis": analysis,
            "cues": cues,
            "cache_hit": cue_meta["cache_hit"],
            "applied_filter": filter_expr is not None,
        },
    )


async def intent_em_full_filter(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    plan: dict,
    speaker_map: dict[str, dict[str, str]],
    llm_cache: _MergedLLMCache,
    openai_client,
) -> IntentEMResult:
    """Speaker filter + negation score-nudge + v2f cues.

    Matches intent_em_speaker_only, but if the plan has negation=True,
    turns containing negation markers receive a small score bonus
    post-retrieval.
    """
    filter_expr, analysis = build_em_filter(
        plan, question, conversation_id, speaker_map
    )

    primer_filtered = await _query_em_filtered(
        memory, question, vector_search_limit=K, property_filter=filter_expr
    )
    cues, cue_meta = await _generate_v2f_cues(
        memory, question, llm_cache=llm_cache, openai_client=openai_client
    )
    cue_batches: list[list[EMHit]] = []
    for cue in cues[:2]:
        cue_batches.append(
            await _query_em_filtered(
                memory,
                cue,
                vector_search_limit=K,
                property_filter=filter_expr,
            )
        )
    merged = _merge_by_max_score([primer_filtered, *cue_batches])

    constraints = plan.get("constraints", {}) or {}
    neg_applied = False
    if constraints.get("negation"):
        merged = _apply_negation_nudge(merged, bonus=0.02)
        neg_applied = True
        analysis["applied"].append("negation:score_nudge")

    return IntentEMResult(
        hits=merged[:K],
        metadata={
            "variant": "intent_em_full_filter",
            "plan": plan,
            "filter_analysis": analysis,
            "cues": cues,
            "cache_hit": cue_meta["cache_hit"],
            "applied_filter": filter_expr is not None,
            "applied_negation_nudge": neg_applied,
        },
    )


async def intent_em_filter_no_cues(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    plan: dict,
    speaker_map: dict[str, dict[str, str]],
) -> IntentEMResult:
    """Filter only on raw question; NO v2f cue generation.

    Analog to em_two_speaker_query_only: tests whether a structured-parse-
    driven property_filter alone, without cue gen, suffices.
    """
    filter_expr, analysis = build_em_filter(
        plan, question, conversation_id, speaker_map
    )
    filtered = await _query_em_filtered(
        memory, question, vector_search_limit=K, property_filter=filter_expr
    )
    # When no filter fires, we still want a useful baseline: fall back to
    # plain (unfiltered) EM.query on the raw question.
    if filter_expr is None:
        hits = _dedupe_by_turn_id(filtered)[:K]
    else:
        # Merge unfiltered raw-question hits under the filtered set only
        # when filter produced fewer than K -- a safety backfill.
        if len(_dedupe_by_turn_id(filtered)) < K:
            unfiltered = await _query_em_filtered(
                memory, question, vector_search_limit=K, property_filter=None
            )
            merged = _merge_by_max_score([filtered, unfiltered])
            hits = merged[:K]
            analysis["applied"].append("backfill_unfiltered")
        else:
            hits = _dedupe_by_turn_id(filtered)[:K]

    return IntentEMResult(
        hits=hits,
        metadata={
            "variant": "intent_em_filter_no_cues",
            "plan": plan,
            "filter_analysis": analysis,
            "applied_filter": filter_expr is not None,
        },
    )


async def intent_em_with_speakerformat_cues(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    plan: dict,
    speaker_map: dict[str, dict[str, str]],
    participants: tuple[str, str],
    llm_cache: _MergedLLMCache,
    openai_client,
) -> IntentEMResult:
    """Filter + speakerformat cues (from em_retuned_cue_gen).

    Tests whether retuned cues (each starting with '<speaker>: ') compose
    on top of a hard EM filter.
    """
    filter_expr, analysis = build_em_filter(
        plan, question, conversation_id, speaker_map
    )
    primer_filtered = await _query_em_filtered(
        memory, question, vector_search_limit=K, property_filter=filter_expr
    )
    cues, cue_meta = await _generate_speakerformat_cues(
        memory,
        question,
        participants,
        llm_cache=llm_cache,
        openai_client=openai_client,
    )
    cue_batches: list[list[EMHit]] = []
    for cue in cues[:2]:
        cue_batches.append(
            await _query_em_filtered(
                memory,
                cue,
                vector_search_limit=K,
                property_filter=filter_expr,
            )
        )
    merged = _merge_by_max_score([primer_filtered, *cue_batches])[:K]
    return IntentEMResult(
        hits=merged,
        metadata={
            "variant": "intent_em_with_speakerformat_cues",
            "plan": plan,
            "filter_analysis": analysis,
            "cues": cues,
            "cache_hit": cue_meta["cache_hit"],
            "applied_filter": filter_expr is not None,
        },
    )
