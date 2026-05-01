"""Iterative/agentic SS-era retrieval architectures ported to EventMemory.

Four architectures, each with speakerformat cue generation (V2F_SPEAKERFORMAT_PROMPT):
  em_hypothesis_driven_sf         : hypothesize -> search -> evaluate -> revise
  em_v15_conditional_hop2_sf      : v2f hop1 + conditional second hop
  em_v15_rerank_sf                : v2f retrieval + LLM rerank of top candidates
  em_working_memory_buffer_sf     : state-accumulating iterative, curated buffer

Each also has a _filter variant that composes with speaker property_filter
when the query mentions one participant (mirrors em_two_speaker_filter).

Dedicated caches written to cache/iter_*_cache.json.
Framework files, em_architectures.py, em_retuned_cue_gen.py, em_two_speaker.py
are IMPORTED not modified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from em_architectures import (
    V2F_MODEL,
    EMHit,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _MergedLLMCache,
    _query_em,
    format_primer_context,
)
from em_retuned_cue_gen import (
    build_v2f_speakerformat_prompt,
)
from em_retuned_cue_gen import (
    parse_cues as parse_sf_cues,
)
from em_two_speaker import classify_speaker_side
from memmachine_server.common.filter.filter_parser import Comparison
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

CACHE_DIR = Path(__file__).resolve().parent / "cache"

ITER_HD_CACHE = CACHE_DIR / "iter_hypothesis_driven_sf_cache.json"
ITER_CH2_CACHE = CACHE_DIR / "iter_v15_conditional_hop2_sf_cache.json"
ITER_RR_CACHE = CACHE_DIR / "iter_v15_rerank_sf_cache.json"
ITER_WMB_CACHE = CACHE_DIR / "iter_working_memory_buffer_sf_cache.json"


# --------------------------------------------------------------------------
# Prompts: speakerformat variants of the SS-era iterative architectures
# --------------------------------------------------------------------------

# Hypothesis-driven: initial hypothesis, then revision rounds.
# Cues in speaker-prefix format for EM embedded alignment.
HD_INITIAL_PROMPT = """\
You are searching a conversation history between {participant_1} and \
{participant_2} to answer a question. Turns are embedded in the format \
"<speaker_name>: <chat content>". Your approach: form a HYPOTHESIS about \
the answer, then search for evidence.

Question: {question}

{context_section}

STEP 1: Based on these excerpts, form a HYPOTHESIS about what the answer \
to this question is. Be specific -- include concrete details you'd expect.

STEP 2: Generate 2 search cues to find EVIDENCE for or against your \
hypothesis. Each cue MUST begin with "<speaker_name>: " where \
<speaker_name> is {participant_1} or {participant_2} (pick whichever would \
plausibly have said the content). Target content that would CONFIRM or \
CONTRADICT your hypothesis. Use specific vocabulary.

Format:
HYPOTHESIS: <your specific hypothesis about the answer>
EVIDENCE_NEEDED: <what would confirm or contradict this>
CUE: <speaker_name>: <search text>
CUE: <speaker_name>: <search text>
Nothing else."""

HD_REVISION_PROMPT = """\
You are testing a hypothesis about a conversation between {participant_1} \
and {participant_2} to answer a question. Turns are embedded in \
"<speaker_name>: <chat content>" format.

Question: {question}

CURRENT HYPOTHESIS: {hypothesis}

ALL EVIDENCE FOUND ({n_segments} segments):
{context}

PREVIOUS CUES TRIED:
{previous_cues}

EVALUATE: Does the evidence SUPPORT, CONTRADICT, or PARTIALLY SUPPORT your \
hypothesis? What specific details confirm or challenge it?

Then either:
- REVISE your hypothesis based on the evidence and generate 2 new cues
- CONFIRM the hypothesis is well-supported and STOP

Each cue MUST begin with "<speaker_name>: " where <speaker_name> is \
{participant_1} or {participant_2}.

Format:
EVALUATION: <SUPPORTS/CONTRADICTS/PARTIAL -- specific reasoning>
REVISED_HYPOTHESIS: <updated hypothesis, or CONFIRMED if stopping>
CUE: <speaker_name>: <search text> (omit if CONFIRMED)
CUE: <speaker_name>: <search text> (omit if CONFIRMED)
Nothing else."""


# v15 conditional hop2 -- hop1 uses the standard speakerformat prompt;
# hop2 is conditional on gaps.
CH2_HOP2_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history between {participant_1} and {participant_2}. Turns are embedded \
in "<speaker_name>: <chat content>" format. Each cue MUST begin with \
"<speaker_name>: " where <speaker_name> is {participant_1} or \
{participant_2}.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}

PREVIOUS CUES ALREADY TRIED (do NOT repeat or paraphrase):
{previous_cues}

First, briefly assess: Given what's been retrieved so far, how well is \
this search going? What kind of content is still missing?

Then EITHER:
- Generate 2 search cues if critical content is still missing
- Output STOP if the question is well-covered

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
OR:
ASSESSMENT: <1-2 sentence self-evaluation>
STOP
Nothing else."""


# Rerank prompt: ask LLM to rank candidates by relevance.
RERANK_PROMPT = """\
Given this question about a conversation, rank the following segments by \
relevance. List the MOST relevant segment numbers first.

Question: {question}

Segments:
{cand_text}

List the segment numbers in order of relevance (most relevant first). \
Include ALL segments. Just list numbers separated by commas.

RANKING:"""


# Working-memory buffer speakerformat prompt.
WMB_PROMPT = """\
You are managing a working memory buffer to answer a question from a \
conversation history between {participant_1} and {participant_2}. Turns \
are embedded in "<speaker_name>: <chat content>" format. Your buffer holds \
the {buffer_size} most important segments for answering the question.

Question: {question}

CURRENT BUFFER ({current_size} slots, max {buffer_size}):
{buffer_text}{new_text}{prev_cue_text}

INSTRUCTIONS:
1. ASSESS what the buffer covers and what's missing for the question.
2. If any buffer slots contain irrelevant content, EVICT them (list turn IDs).
3. Generate 2 search cues targeting MISSING information. Each cue MUST \
begin with "<speaker_name>: " where <speaker_name> is {participant_1} or \
{participant_2}. Use specific vocabulary.

Format:
ASSESSMENT: <what's in buffer, what's missing>
EVICT: <comma-separated turn IDs to remove, or NONE>
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
Nothing else."""


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class IterResult:
    hits: list[EMHit]
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


async def _llm_call(
    openai_client,
    prompt: str,
    cache: _MergedLLMCache,
) -> tuple[str, bool]:
    cached = cache.get(V2F_MODEL, prompt)
    if cached is not None:
        return cached, True
    resp = await openai_client.chat.completions.create(
        model=V2F_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    cache.put(V2F_MODEL, prompt, text)
    return text, False


def _hits_to_segments_dict(hits: list[EMHit], max_items: int = 12) -> list[dict]:
    sorted_hits = sorted(hits, key=lambda h: h.turn_id)[:max_items]
    return [{"turn_id": h.turn_id, "role": h.role, "text": h.text} for h in sorted_hits]


def _format_iter_context(hits: list[EMHit], *, max_items: int = 12) -> str:
    """RETRIEVED CONVERSATION EXCERPTS SO FAR: style (same as format_primer_context)."""
    segs = _hits_to_segments_dict(hits, max_items=max_items)
    return format_primer_context(segs, max_items=max_items)


# Parse REVISED_HYPOTHESIS, EVALUATION, HYPOTHESIS lines (case-insensitive).
_HYP_RE = re.compile(r"^\s*HYPOTHESIS\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_REVHYP_RE = re.compile(
    r"^\s*REVISED_HYPOTHESIS\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE
)
_EVAL_RE = re.compile(r"^\s*EVALUATION\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_EVICT_RE = re.compile(r"^\s*EVICT\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_ASSESS_RE = re.compile(r"^\s*ASSESSMENT\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _first(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _contains_stop(text: str) -> bool:
    for line in text.splitlines():
        if line.strip().upper() == "STOP":
            return True
    return False


# --------------------------------------------------------------------------
# Architecture 1: Hypothesis-driven (speakerformat)
# --------------------------------------------------------------------------


async def em_hypothesis_driven_sf(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    participants: tuple[str, str],
    cache: _MergedLLMCache,
    openai_client,
    max_revisions: int = 2,
    per_cue_k: int = 10,
) -> IterResult:
    p_user, p_asst = participants
    llm_calls = 0

    # Hop 0: raw-question primer (K=10, expand=0).
    primer = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    all_hits: list[EMHit] = list(primer)
    seen_turn_ids: set[int] = {h.turn_id for h in all_hits}
    previous_cues: list[str] = []
    hypothesis_log: list[dict] = []

    hypothesis = ""
    # Also accumulate per-cue hit batches for final merge.
    cue_batches: list[list[EMHit]] = []
    # Initial retrieval batch at max_K for merge.
    primer_for_merge = await _query_em(
        memory, question, vector_search_limit=K, expand_context=0
    )
    cue_batches.append(primer_for_merge)

    for revision in range(max_revisions + 1):
        context = _format_iter_context(all_hits)
        if revision == 0:
            prompt = HD_INITIAL_PROMPT.format(
                participant_1=p_user,
                participant_2=p_asst,
                question=question,
                context_section=context,
            )
        else:
            prev_cues_text = "\n".join(f"- {c}" for c in previous_cues)
            prompt = HD_REVISION_PROMPT.format(
                participant_1=p_user,
                participant_2=p_asst,
                question=question,
                hypothesis=hypothesis or "none",
                n_segments=len(all_hits),
                context=context,
                previous_cues=prev_cues_text,
            )
        raw, _hit = await _llm_call(openai_client, prompt, cache)
        llm_calls += 1

        if revision == 0:
            hyp = _first(_HYP_RE, raw)
            if hyp:
                hypothesis = hyp
            evaluation = ""
            confirmed = False
        else:
            evaluation = _first(_EVAL_RE, raw)
            rev_hyp = _first(_REVHYP_RE, raw)
            confirmed = "CONFIRMED" in rev_hyp.upper()
            if rev_hyp and not confirmed:
                hypothesis = rev_hyp

        cues = parse_sf_cues(raw, max_cues=2)
        hypothesis_log.append(
            {
                "revision": revision,
                "hypothesis": hypothesis,
                "evaluation": evaluation,
                "num_cues": len(cues),
                "confirmed": confirmed,
                "cues": cues,
            }
        )

        if confirmed or not cues:
            break

        # Run cues: one retrieval each. Update seen-set for display; merge later.
        for cue in cues[:2]:
            hits = await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
            cue_batches.append(hits)
            # Accumulate unseen hits for display context.
            for h in hits:
                if h.turn_id not in seen_turn_ids:
                    all_hits.append(h)
                    seen_turn_ids.add(h.turn_id)
            previous_cues.append(cue)

    merged = _merge_by_max_score(cue_batches)
    return IterResult(
        hits=merged[:K],
        metadata={
            "name": "em_hypothesis_driven_sf",
            "revisions": len(hypothesis_log),
            "hypothesis_log": hypothesis_log,
            "total_cues": len(previous_cues),
            "llm_calls": llm_calls,
        },
    )


# --------------------------------------------------------------------------
# Architecture 2: v15 conditional hop2 (speakerformat)
# --------------------------------------------------------------------------


async def em_v15_conditional_hop2_sf(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    participants: tuple[str, str],
    cache: _MergedLLMCache,
    openai_client,
    per_cue_k: int = 10,
) -> IterResult:
    p_user, p_asst = participants
    llm_calls = 0

    primer = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    all_hits_display: list[EMHit] = list(primer)
    seen: set[int] = {h.turn_id for h in all_hits_display}
    cue_batches: list[list[EMHit]] = [
        await _query_em(memory, question, vector_search_limit=K, expand_context=0)
    ]

    # Hop 1: standard speakerformat prompt.
    context = _format_iter_context(all_hits_display)
    hop1_prompt = build_v2f_speakerformat_prompt(question, context, p_user, p_asst)
    raw1, _hit1 = await _llm_call(openai_client, hop1_prompt, cache)
    llm_calls += 1
    hop1_cues = parse_sf_cues(raw1, max_cues=2)

    for cue in hop1_cues[:2]:
        hits = await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
        cue_batches.append(hits)
        for h in hits:
            if h.turn_id not in seen:
                all_hits_display.append(h)
                seen.add(h.turn_id)

    # Hop 2: conditional.
    context2 = _format_iter_context(all_hits_display)
    prev_cues_text = "\n".join(f"- {c}" for c in hop1_cues[:2])
    hop2_prompt = CH2_HOP2_PROMPT.format(
        participant_1=p_user,
        participant_2=p_asst,
        question=question,
        context=context2,
        previous_cues=prev_cues_text,
    )
    raw2, _hit2 = await _llm_call(openai_client, hop2_prompt, cache)
    llm_calls += 1

    hop2_stopped = _contains_stop(raw2)
    hop2_cues: list[str] = []
    if not hop2_stopped:
        hop2_cues = parse_sf_cues(raw2, max_cues=2)
        for cue in hop2_cues[:2]:
            hits = await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
            cue_batches.append(hits)

    merged = _merge_by_max_score(cue_batches)
    return IterResult(
        hits=merged[:K],
        metadata={
            "name": "em_v15_conditional_hop2_sf",
            "hop1_cues": hop1_cues[:2],
            "hop2_stopped": hop2_stopped,
            "hop2_cues": hop2_cues[:2],
            "total_cues": len(hop1_cues[:2]) + len(hop2_cues[:2]),
            "llm_calls": llm_calls,
        },
    )


# --------------------------------------------------------------------------
# Architecture 3: v15 rerank (speakerformat)
# --------------------------------------------------------------------------


async def em_v15_rerank_sf(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    participants: tuple[str, str],
    cache: _MergedLLMCache,
    openai_client,
    per_cue_k: int = 10,
) -> IterResult:
    p_user, p_asst = participants
    llm_calls = 0

    primer = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    all_hits_display: list[EMHit] = list(primer)
    seen: set[int] = {h.turn_id for h in all_hits_display}
    cue_batches: list[list[EMHit]] = [
        await _query_em(memory, question, vector_search_limit=K, expand_context=0)
    ]

    # v2f speakerformat hop.
    context = _format_iter_context(all_hits_display)
    prompt = build_v2f_speakerformat_prompt(question, context, p_user, p_asst)
    raw, _hit = await _llm_call(openai_client, prompt, cache)
    llm_calls += 1
    cues = parse_sf_cues(raw, max_cues=2)

    for cue in cues[:2]:
        hits = await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
        cue_batches.append(hits)

    # Merge.
    merged = _merge_by_max_score(cue_batches)

    # Rerank hop: ask LLM to rank top 25.
    # Dedup by turn_id, take top 25 by cosine.
    candidates = merged[:25]
    cand_text = "\n".join(
        f"[{i}] Turn {h.turn_id} ({h.role}): {h.text[:200]}"
        for i, h in enumerate(candidates)
    )
    rerank_prompt = RERANK_PROMPT.format(question=question, cand_text=cand_text)
    rr_raw, _rr_hit = await _llm_call(openai_client, rerank_prompt, cache)
    llm_calls += 1

    # Parse comma-separated indices.
    ranked_indices: list[int] = []
    seen_ranked: set[int] = set()
    for part in rr_raw.strip().split(","):
        part = part.strip().strip("[]").strip()
        # Handle newlines / prefixes.
        m = re.search(r"\d+", part)
        if not m:
            continue
        try:
            idx = int(m.group(0))
        except ValueError:
            continue
        if 0 <= idx < len(candidates) and idx not in seen_ranked:
            ranked_indices.append(idx)
            seen_ranked.add(idx)

    reranked: list[EMHit] = [candidates[i] for i in ranked_indices]
    # Append candidates not in ranking.
    for i, h in enumerate(candidates):
        if i not in seen_ranked:
            reranked.append(h)
    # Append remaining (beyond top-25) so K=50 still has depth.
    cand_ids = {h.turn_id for h in candidates}
    for h in merged:
        if h.turn_id not in cand_ids:
            reranked.append(h)

    return IterResult(
        hits=reranked[:K],
        metadata={
            "name": "em_v15_rerank_sf",
            "cues": cues[:2],
            "n_reranked": len(ranked_indices),
            "llm_calls": llm_calls,
        },
    )


# --------------------------------------------------------------------------
# Architecture 4: Working memory buffer (speakerformat)
# --------------------------------------------------------------------------


async def em_working_memory_buffer_sf(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    participants: tuple[str, str],
    cache: _MergedLLMCache,
    openai_client,
    buffer_size: int = 8,
    max_hops: int = 2,
    per_cue_k: int = 10,
) -> IterResult:
    p_user, p_asst = participants
    llm_calls = 0

    primer = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    all_hits: list[EMHit] = list(primer)
    seen: set[int] = {h.turn_id for h in all_hits}
    # Buffer begins with top results.
    buffer: list[EMHit] = list(primer[:buffer_size])
    cue_batches: list[list[EMHit]] = [
        await _query_em(memory, question, vector_search_limit=K, expand_context=0)
    ]
    previous_cues: list[str] = []
    hop_log: list[dict] = []

    for hop in range(max_hops):
        buffer_text = _format_iter_context(buffer, max_items=buffer_size)
        # New candidates.
        buf_ids = {h.turn_id for h in buffer}
        new_cands = [h for h in all_hits if h.turn_id not in buf_ids]
        new_text = ""
        if new_cands:
            recent = sorted(new_cands, key=lambda h: h.turn_id)[-6:]
            new_text = (
                "\n\nNEW SEGMENTS (not in buffer, available for swap):\n"
                + _format_iter_context(recent, max_items=6)
            )
        prev_cue_text = ""
        if previous_cues:
            prev_cue_text = "\n\nPREVIOUS CUES:\n" + "\n".join(
                f"- {c}" for c in previous_cues
            )

        prompt = WMB_PROMPT.format(
            participant_1=p_user,
            participant_2=p_asst,
            buffer_size=buffer_size,
            current_size=len(buffer),
            question=question,
            buffer_text=buffer_text,
            new_text=new_text,
            prev_cue_text=prev_cue_text,
        )
        raw, _hit = await _llm_call(openai_client, prompt, cache)
        llm_calls += 1

        assessment = _first(_ASSESS_RE, raw)
        evict_text = _first(_EVICT_RE, raw)
        evict_ids: set[int] = set()
        if evict_text and evict_text.upper() != "NONE":
            for part in evict_text.split(","):
                part = part.strip().strip("[]").strip()
                m = re.search(r"\d+", part)
                if m:
                    try:
                        evict_ids.add(int(m.group(0)))
                    except ValueError:
                        pass
        cues = parse_sf_cues(raw, max_cues=2)

        if evict_ids:
            buffer = [h for h in buffer if h.turn_id not in evict_ids]

        hop_log.append(
            {
                "hop": hop,
                "assessment": assessment,
                "evicted": sorted(evict_ids),
                "buffer_size_after_evict": len(buffer),
                "num_cues": len(cues),
                "cues": cues,
            }
        )

        if not cues:
            break

        # Retrieve for each cue.
        new_this_hop: list[EMHit] = []
        for cue in cues[:2]:
            hits = await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
            cue_batches.append(hits)
            for h in hits:
                if h.turn_id not in seen:
                    all_hits.append(h)
                    new_this_hop.append(h)
                    seen.add(h.turn_id)
            previous_cues.append(cue)

        for h in new_this_hop:
            if len(buffer) < buffer_size:
                buffer.append(h)

    merged = _merge_by_max_score(cue_batches)
    return IterResult(
        hits=merged[:K],
        metadata={
            "name": "em_working_memory_buffer_sf",
            "hops": len(hop_log),
            "hop_log": hop_log,
            "final_buffer_size": len(buffer),
            "total_cues": len(previous_cues),
            "llm_calls": llm_calls,
        },
    )


# --------------------------------------------------------------------------
# Speaker-filter composition: wraps any base arch.
# --------------------------------------------------------------------------


async def _speaker_filter_topup(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    base_hits: list[EMHit],
    speaker_map: dict[str, dict[str, str]],
    role_only_top_m: int = 5,
) -> tuple[list[EMHit], dict]:
    """Mirror em_two_speaker_filter: drop opposite-side hits and append
    speaker-filtered query results for the matched side."""
    side, user_name, asst_name, name_tokens = classify_speaker_side(
        question, conversation_id, speaker_map
    )
    metadata = {
        "conv_user_name": user_name,
        "conv_assistant_name": asst_name,
        "query_name_tokens": name_tokens,
        "matched_side": side,
        "applied_speaker_filter": False,
    }
    if side not in ("user", "assistant"):
        return base_hits[:K], metadata

    matched_name = user_name if side == "user" else asst_name
    metadata["applied_speaker_filter"] = True
    metadata["matched_name"] = matched_name

    prop_filter = Comparison(field="context.source", op="=", value=matched_name)
    raw = await memory.query(
        query=question,
        vector_search_limit=K + 10,
        expand_context=0,
        property_filter=prop_filter,
    )
    speaker_hits: list[EMHit] = []
    for sc in raw.scored_segment_contexts:
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

    matched_role = side
    kept = [h for h in base_hits if h.role == matched_role]
    dropped = [h.turn_id for h in base_hits if h.role != matched_role]
    metadata["dropped_base_turn_ids"] = dropped

    seen = {h.turn_id for h in kept}
    appended: list[EMHit] = []
    for h in speaker_hits:
        if h.turn_id in seen:
            continue
        appended.append(h)
        seen.add(h.turn_id)
        if len(appended) >= (role_only_top_m + 10):
            break
    metadata["appended_turn_ids"] = [h.turn_id for h in appended]
    merged = kept + appended
    return merged[:K], metadata


async def em_iterative_with_filter(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    *,
    K: int,
    participants: tuple[str, str],
    cache: _MergedLLMCache,
    openai_client,
    speaker_map: dict[str, dict[str, str]],
    arch_fn,
    **arch_kwargs,
) -> IterResult:
    base = await arch_fn(
        memory,
        question,
        K=max(K, 50),
        participants=participants,
        cache=cache,
        openai_client=openai_client,
        **arch_kwargs,
    )
    filtered_hits, filter_meta = await _speaker_filter_topup(
        memory,
        question,
        conversation_id,
        K=K,
        base_hits=base.hits,
        speaker_map=speaker_map,
    )
    meta = dict(base.metadata)
    meta.update({f"filter_{k}": v for k, v in filter_meta.items()})
    return IterResult(hits=filtered_hits, metadata=meta)
