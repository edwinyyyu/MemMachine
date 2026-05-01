"""EventMemory-native two_speaker_filter.

Reuses `em_v2f` from `em_architectures` as the base retrieval pipeline, then
either hard-filters (filter_mode=True) by speaker using EventMemory's
`property_filter` on `context.source`, or appends role-filtered cosine hits.

Unlike the SS-era two_speaker_filter which ran role-filtered cosine manually,
here we rely on EM's native property_filter: we issue a SECOND EM.query()
with `property_filter = Comparison("context.source", "=", <speaker_name>)`
to fetch per-speaker candidates, then merge with the v2f output.

Variants exposed here:
  em_two_speaker_filter  : v2f -> if query mentions ONE side, drop the other
                           side from v2f output AND top up with property-
                           filtered EM.query hits. Mirrors SS filter_mode=True.
  em_two_speaker_query_only : property-filtered EM.query on raw question only
                           (diagnostic; tests whether the hard speaker filter
                           alone, without v2f cues, lifts recall over the
                           speaker-baked cosine baseline).

Speaker name source: reuses `results/conversation_two_speakers.json`
(Caroline/Melanie, Jon/Gina, John/Maria).

Query mention detection: reuses `speaker_attributed.extract_name_mentions`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from em_architectures import (
    EMHit,
    _dedupe_by_turn_id,
    _MergedLLMCache,
    _query_em,
    em_v2f,
)
from memmachine_server.common.filter.filter_parser import Comparison
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from speaker_attributed import extract_name_mentions

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONV_TWO_SPEAKERS_FILE = RESULTS_DIR / "conversation_two_speakers.json"


def load_two_speaker_map() -> dict[str, dict[str, str]]:
    with open(CONV_TWO_SPEAKERS_FILE) as f:
        data = json.load(f)
    return data.get("speakers", {}) or {}


def classify_speaker_side(
    question: str, conversation_id: str, speaker_map: dict[str, dict[str, str]]
) -> tuple[str, str, str, list[str]]:
    """Returns (side, user_name, assistant_name, name_tokens).

    side in {"user", "assistant", "both", "none"}.
    """
    pair = speaker_map.get(conversation_id, {})
    user_name = (pair.get("user") or "UNKNOWN").strip() or "UNKNOWN"
    asst_name = (pair.get("assistant") or "UNKNOWN").strip() or "UNKNOWN"
    tokens = extract_name_mentions(question)
    lowered = {t.lower() for t in tokens}
    hit_user = user_name != "UNKNOWN" and user_name.lower() in lowered
    hit_asst = asst_name != "UNKNOWN" and asst_name.lower() in lowered
    if hit_user and hit_asst:
        side = "both"
    elif hit_user:
        side = "user"
    elif hit_asst:
        side = "assistant"
    else:
        side = "none"
    return side, user_name, asst_name, tokens


@dataclass
class TwoSpeakerResult:
    hits: list[EMHit]
    metadata: dict


async def em_two_speaker_filter(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    speaker_map: dict[str, dict[str, str]],
    llm_cache: _MergedLLMCache,
    openai_client=None,
    role_only_top_m: int = 5,
) -> TwoSpeakerResult:
    """v2f base + speaker property_filter topup (filter_mode analogue).

    Pipeline:
      1. Run em_v2f on the question -> v2f ranked hits.
      2. If query mentions exactly ONE side, call EM.query with
         property_filter=context.source=<side_name>, K' = K + 5. Append the
         top (role_only_top_m + 5) hits that are NOT already in v2f.
      3. Drop v2f items whose speaker != matched side.
      4. Rank: kept v2f in v2f score order, then appended speaker-filtered
         hits in their score order. Dedup by turn_id.
    """
    # Base v2f pass.
    v2f_hits, v2f_meta = await em_v2f(
        memory,
        question,
        K=max(K, 50),
        llm_cache=llm_cache,
        openai_client=openai_client,
        expand_context=0,
    )

    side, user_name, asst_name, name_tokens = classify_speaker_side(
        question, conversation_id, speaker_map
    )

    metadata: dict = {
        "v2f_cues": v2f_meta.get("cues", []),
        "v2f_cache_hit": v2f_meta.get("cache_hit", False),
        "conv_user_name": user_name,
        "conv_assistant_name": asst_name,
        "query_name_tokens": name_tokens,
        "matched_side": side,
        "applied_speaker_filter": False,
        "appended_turn_ids": [],
        "dropped_v2f_turn_ids": [],
    }

    if side not in ("user", "assistant"):
        # Either no mention, both sides mentioned, or names unknown.
        return TwoSpeakerResult(hits=v2f_hits[:K], metadata=metadata)

    matched_name = user_name if side == "user" else asst_name
    metadata["applied_speaker_filter"] = True
    metadata["matched_name"] = matched_name

    # Property-filtered EM.query for the matched speaker.
    speaker_filter = Comparison(field="context.source", op="=", value=matched_name)
    speaker_hits_raw = await memory.query(
        query=question,
        vector_search_limit=K + 10,
        expand_context=0,
        property_filter=speaker_filter,
    )
    speaker_hits: list[EMHit] = []
    for sc in speaker_hits_raw.scored_segment_contexts:
        for seg in sc.segments:
            speaker_hits.append(
                EMHit(
                    turn_id=int(seg.properties.get("turn_id", -1)),
                    score=sc.score,
                    seed_segment_uuid=sc.seed_segment_uuid,
                    role=str(seg.properties.get("role", "")),
                    text=seg.block.text,
                )
            )
    speaker_hits = _dedupe_by_turn_id(speaker_hits)

    # Filter v2f to the matched role. role properties: role=user or role=assistant.
    matched_role = side  # "user" or "assistant"
    kept_v2f = [h for h in v2f_hits if h.role == matched_role]
    dropped = [h.turn_id for h in v2f_hits if h.role != matched_role]
    metadata["dropped_v2f_turn_ids"] = dropped

    # Append speaker_hits not already in kept_v2f.
    seen = {h.turn_id for h in kept_v2f}
    appended: list[EMHit] = []
    for h in speaker_hits:
        if h.turn_id in seen:
            continue
        appended.append(h)
        seen.add(h.turn_id)
        if len(appended) >= (role_only_top_m + 10):
            break
    metadata["appended_turn_ids"] = [h.turn_id for h in appended]

    merged = kept_v2f + appended
    return TwoSpeakerResult(hits=merged[:K], metadata=metadata)


async def em_two_speaker_query_only(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    speaker_map: dict[str, dict[str, str]],
) -> TwoSpeakerResult:
    """Diagnostic: property_filter applied to RAW question (no v2f).

    Tests whether the hard speaker filter alone lifts recall over
    em_cosine_baseline. When side in {"none", "both"}, falls back to raw
    cosine (no filter) so we still return K hits.
    """
    side, user_name, asst_name, name_tokens = classify_speaker_side(
        question, conversation_id, speaker_map
    )
    metadata: dict = {
        "conv_user_name": user_name,
        "conv_assistant_name": asst_name,
        "query_name_tokens": name_tokens,
        "matched_side": side,
        "applied_speaker_filter": False,
    }
    if side in ("user", "assistant"):
        matched_name = user_name if side == "user" else asst_name
        metadata["applied_speaker_filter"] = True
        metadata["matched_name"] = matched_name
        prop_filter = Comparison(field="context.source", op="=", value=matched_name)
        hits = await _query_em(
            memory,
            question,
            vector_search_limit=K,
            expand_context=0,
        )
        # Also retrieve with filter; merge by max.
        filtered_raw = await memory.query(
            query=question,
            vector_search_limit=K,
            expand_context=0,
            property_filter=prop_filter,
        )
        filtered: list[EMHit] = []
        for sc in filtered_raw.scored_segment_contexts:
            for seg in sc.segments:
                filtered.append(
                    EMHit(
                        turn_id=int(seg.properties.get("turn_id", -1)),
                        score=sc.score,
                        seed_segment_uuid=sc.seed_segment_uuid,
                        role=str(seg.properties.get("role", "")),
                        text=seg.block.text,
                    )
                )
        filtered = _dedupe_by_turn_id(filtered)
        # Filter unrestricted hits by matched_role first; then append filtered
        # hits that aren't already covered.
        matched_role = side
        kept = [h for h in hits if h.role == matched_role]
        kept = _dedupe_by_turn_id(kept)
        seen = {h.turn_id for h in kept}
        for h in filtered:
            if h.turn_id in seen:
                continue
            kept.append(h)
            seen.add(h.turn_id)
        return TwoSpeakerResult(hits=kept[:K], metadata=metadata)

    # side in ("none", "both") -> plain retrieval.
    hits = await _query_em(memory, question, vector_search_limit=K, expand_context=0)
    hits = _dedupe_by_turn_id(hits)
    return TwoSpeakerResult(hits=hits[:K], metadata=metadata)
