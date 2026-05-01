"""EventMemory-native confidence-gated overlay WITHOUT speaker channel.

The speaker channel is dropped here because `em_two_speaker_filter` (see
`em_two_speaker.py`) handles the speaker-side retrieval natively via
EventMemory's `property_filter`. Routing both would double-count the
mechanism.

Channels retained (subset of gated_overlay's SUPPLEMENT_NAMES):
  - alias_context   : alias-sibling query probes via EM.query
  - temporal_tokens : query retrieval with temporal token bias (we do NOT
                      re-derive turn-level temporal masks here; instead we
                      just rely on EM.query with the raw query but use a
                      modest K so it still contributes channel diversity)
  - entity_exact_match : exact-string-match on segment text over all
                      ingested segments (load via partition.get_segments)

Critical_info is DROPPED (LoCoMo has 0 flagged turns per
`critical_info_store.json`, so the channel is degenerate here).

Routing: reuses gated_overlay's ROUTING_PROMPT shape but only asks about the
three retained channels. Cached in a dedicated `emdef_gated_llm_cache.json`.

Base: em_v2f ranked list. Firing channels replace v2f's WEAKEST tail slots.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from alias_expansion import build_expanded_queries, find_alias_matches
from em_architectures import (
    EMHit,
    _dedupe_by_turn_id,
    _MergedLLMCache,
    _query_em,
    em_v2f,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from multichannel_weighted import extract_query_entities

CACHE_DIR = Path(__file__).resolve().parent / "cache"
EMDEF_GATED_LLM_CACHE = CACHE_DIR / "emdef_gated_llm_cache.json"
GATED_LLM_CACHE = CACHE_DIR / "gated_llm_cache.json"

ROUTER_MODEL = "gpt-5-mini"

SUPPLEMENT_NAMES_NOSPEAKER = (
    "alias_context",
    "temporal_tokens",
    "entity_exact_match",
)


# Dedicated routing prompt (no speaker channel). Kept short so cost is small.
ROUTING_PROMPT_NOSPEAKER = """\
You are deciding which retrieval supplement channels to engage for this \
query. The primary channel is v2f (LLM-imagined cue cosine; always active). \
Supplements can OPTIONALLY replace v2f's weakest candidates if they are \
high-confidence for this specific query.

For each supplement channel, output CONFIDENCE:
- 1.0 = "this channel will definitely find content v2f might miss"
- 0.5 = "might help"
- 0.0 = "irrelevant to this query"

Only channels with confidence >= threshold will be engaged. Be STRICT - \
running irrelevant channels harms retrieval by displacing v2f's strong picks.

Channels:
- alias_context: substitute entity aliases; confidence high if query \
mentions an entity with known aliases
- temporal_tokens: high if query has temporal constraint (when, after, \
during, by, specific date)
- entity_exact_match: high if query has distinctive proper noun (not common \
names)

Query: {query}

Output JSON: {{"alias_context": 0.x, "temporal_tokens": 0.x, \
"entity_exact_match": 0.x, "reasoning": "brief"}}

Output ONLY the JSON object, no prose before or after."""


_RE_DATE_WORDS = re.compile(
    r"\b(?:yesterday|today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|january|february|march|april|may|june|july|august|"
    r"september|october|november|december|last|next|morning|afternoon|"
    r"evening|night|week|weekend|month|year|weekday|weeknight|tonight|"
    r"recently|earlier|later|ago|after|before|during|by)\b",
    re.IGNORECASE,
)
_RE_TIME = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm|a\.m\.|p\.m\.)\b", re.IGNORECASE
)
_RE_DATE_DIGIT = re.compile(
    r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{4}-\d{1,2}-\d{1,2}\b"
)


def parse_confidences_nospeaker(raw: str) -> tuple[dict[str, float], str]:
    default = dict.fromkeys(SUPPLEMENT_NAMES_NOSPEAKER, 0.0)
    fallback_reason = "parse_failed_no_supplements"
    if not raw:
        return default, fallback_reason
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return default, fallback_reason
    if not isinstance(obj, dict):
        return default, fallback_reason
    confs: dict[str, float] = {}
    for ch in SUPPLEMENT_NAMES_NOSPEAKER:
        v = obj.get(ch, 0.0)
        try:
            c = float(v)
        except (TypeError, ValueError):
            c = 0.0
        confs[ch] = max(0.0, min(1.0, c))
    reasoning = str(obj.get("reasoning", "")).strip()[:300]
    return confs, reasoning


@dataclass
class GatedNoSpeakerResult:
    hits: list[EMHit]
    metadata: dict


async def _ch_alias_context(
    memory: EventMemory,
    question: str,
    alias_groups: list[list[str]],
    *,
    top_k: int,
) -> list[EMHit]:
    if not alias_groups:
        return []
    matches = find_alias_matches(question, alias_groups)
    if not matches:
        return []
    variants, _ = build_expanded_queries(question, alias_groups)
    probes: list[str] = [v for v in variants if v != question]
    for _matched, siblings in matches:
        for sib in siblings[:4]:
            if sib not in probes:
                probes.append(sib)
    agg: dict[int, EMHit] = {}
    for probe in probes[:8]:
        batch = await _query_em(
            memory, probe, vector_search_limit=top_k, expand_context=0
        )
        for h in batch:
            prev = agg.get(h.turn_id)
            if prev is None or h.score > prev.score:
                agg[h.turn_id] = h
    return sorted(agg.values(), key=lambda h: -h.score)


async def _ch_temporal(
    memory: EventMemory,
    question: str,
    *,
    top_k: int,
) -> list[EMHit]:
    # Check if the query has temporal tokens at all; if not, no contribution.
    has_temp = bool(
        _RE_DATE_WORDS.search(question)
        or _RE_TIME.search(question)
        or _RE_DATE_DIGIT.search(question)
    )
    if not has_temp:
        return []
    # EM-side we don't have a precomputed temporal mask; fall back to raw
    # EM.query which will at least pick up temporal-content turns near the
    # top of cosine if the query itself has temporal framing.
    hits = await _query_em(
        memory, question, vector_search_limit=top_k, expand_context=0
    )
    return _dedupe_by_turn_id(hits)


async def _ch_entity_exact_match(
    memory: EventMemory,
    question: str,
    all_segments_text: list[tuple[int, str]],
    *,
    top_k: int,
) -> list[EMHit]:
    """Exact-match proper-noun hits over all segments in the conversation.

    `all_segments_text` is a preloaded list of (turn_id, text) for the
    conversation; we do a regex scan per segment for each query entity.
    """
    entities = extract_query_entities(question)
    if not entities:
        return []
    ents_lower = [e.lower() for e in entities]
    scored: list[tuple[int, float, str]] = []
    for turn_id, text in all_segments_text:
        tl = text.lower()
        hits = 0
        for e in ents_lower:
            if re.search(r"\b" + re.escape(e) + r"\b", tl):
                hits += 1
        if hits > 0:
            scored.append((turn_id, float(hits) / float(len(ents_lower)), text))
    scored.sort(key=lambda t: -t[1])
    out: list[EMHit] = []
    # We don't have UUIDs for these synthetic hits; use a zero UUID.
    from uuid import UUID

    dummy_uuid = UUID("00000000-0000-0000-0000-000000000000")
    for turn_id, score, text in scored[:top_k]:
        out.append(
            EMHit(
                turn_id=turn_id,
                score=score,
                seed_segment_uuid=dummy_uuid,
                role="",
                text=text,
            )
        )
    return out


async def em_gated_no_speaker(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    alias_groups_by_conv: dict[str, list[list[str]]],
    all_segments_by_conv: dict[str, list[tuple[int, str]]],
    v2f_llm_cache: _MergedLLMCache,
    router_llm_cache: _MergedLLMCache,
    openai_client=None,
    threshold: float = 0.7,
    per_channel_top_m: int = 3,
    per_channel_retrieval_k: int = 20,
) -> GatedNoSpeakerResult:
    """Confidence-gated overlay WITHOUT speaker channel.

    - Base: em_v2f top-K.
    - Firing channels (confidence>=threshold): alias_context, temporal_tokens,
      entity_exact_match. Each provides up to
      `ceil(per_channel_top_m * confidence)` candidates that replace v2f's
      weakest tail slots.
    """
    # Routing.
    prompt = ROUTING_PROMPT_NOSPEAKER.format(query=question)
    cached = router_llm_cache.get(ROUTER_MODEL, prompt)
    if cached is None:
        if openai_client is None:
            raw = ""
        else:
            resp = await openai_client.chat.completions.create(
                model=ROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content or ""
            router_llm_cache.put(ROUTER_MODEL, prompt, raw)
    else:
        raw = cached
    confs, reasoning = parse_confidences_nospeaker(raw)

    # Base v2f retrieval.
    v2f_hits, v2f_meta = await em_v2f(
        memory,
        question,
        K=K,
        llm_cache=v2f_llm_cache,
        openai_client=openai_client,
        expand_context=0,
    )
    v2f_hits = v2f_hits[:K]

    # Determine firing channels.
    firing: list[str] = []
    m_effective: dict[str, int] = {}
    for ch in SUPPLEMENT_NAMES_NOSPEAKER:
        c = confs.get(ch, 0.0)
        if c < threshold:
            continue
        firing.append(ch)
        m_effective[ch] = max(1, int(math.ceil(per_channel_top_m * c)))

    # Run firing channels.
    channel_hits: dict[str, list[EMHit]] = {}
    for ch in firing:
        if ch == "alias_context":
            h = await _ch_alias_context(
                memory,
                question,
                alias_groups_by_conv.get(conversation_id, []),
                top_k=per_channel_retrieval_k,
            )
        elif ch == "temporal_tokens":
            h = await _ch_temporal(memory, question, top_k=per_channel_retrieval_k)
        elif ch == "entity_exact_match":
            h = await _ch_entity_exact_match(
                memory,
                question,
                all_segments_by_conv.get(conversation_id, []),
                top_k=per_channel_retrieval_k,
            )
        else:
            h = []
        if h:
            channel_hits[ch] = h

    # Overlay assembly: displace v2f's tail with channel picks.
    metadata: dict = {
        "confidences": confs,
        "reasoning": reasoning,
        "firing_channels": firing,
        "m_effective": m_effective,
        "v2f_cues": v2f_meta.get("cues", []),
        "overlay": {"displacements": {}, "channels_contributing": []},
    }
    if not channel_hits:
        return GatedNoSpeakerResult(hits=v2f_hits, metadata=metadata)

    total_displace = min(sum(m_effective.values()), max(K - 1, 0))
    if total_displace <= 0:
        return GatedNoSpeakerResult(hits=v2f_hits, metadata=metadata)

    v2f_turns = [h.turn_id for h in v2f_hits[:K]]
    v2f_turn_set = set(v2f_turns)

    # Build per-channel iterators (channel candidates minus v2f duplicates).
    channel_iters: dict[str, list[EMHit]] = {}
    channel_picked_count: dict[str, int] = {}
    for ch, cands in channel_hits.items():
        channel_iters[ch] = [c for c in cands if c.turn_id not in v2f_turn_set]
        channel_picked_count[ch] = 0

    picked: list[EMHit] = []
    used_turns: set[int] = set()
    order_active = list(channel_hits.keys())
    while len(picked) < total_displace and order_active:
        new_active: list[str] = []
        for ch in order_active:
            cap = m_effective.get(ch, 0)
            if channel_picked_count[ch] >= cap:
                continue
            cand = None
            for c in channel_iters[ch]:
                if c.turn_id in used_turns:
                    continue
                cand = c
                break
            if cand is None:
                continue
            channel_iters[ch] = [
                x for x in channel_iters[ch] if x.turn_id != cand.turn_id
            ]
            used_turns.add(cand.turn_id)
            picked.append(cand)
            channel_picked_count[ch] += 1
            if len(picked) >= total_displace:
                break
            if channel_picked_count[ch] < cap and channel_iters[ch]:
                new_active.append(ch)
        order_active = new_active or []

    metadata["overlay"]["displacements"] = dict(channel_picked_count)
    metadata["overlay"]["channels_contributing"] = [
        ch for ch, n in channel_picked_count.items() if n > 0
    ]

    if not picked:
        return GatedNoSpeakerResult(hits=v2f_hits, metadata=metadata)

    keep = max(K - len(picked), 1)
    picked = picked[: K - keep]
    final = v2f_hits[:keep] + picked
    return GatedNoSpeakerResult(hits=final, metadata=metadata)
