"""Cross-model EventMemory cue-gen variants: gpt-5-nano with structural prompts.

Tests whether structural-constraint cue-gen prompts (speakerformat,
HyDE first_person) close the mini-nano model gap on EventMemory retrieval.

Reuses EXACT prompts from:
  - em_architectures.V2F_PROMPT              (vanilla control)
  - em_retuned_cue_gen.V2F_SPEAKERFORMAT_PROMPT (mini-retuned winner)
  - em_hyde_orient.HYDE_FIRST_PERSON_PROMPT  (current LoCoMo K=50 ceiling)

Variants exposed here (gpt-5-nano primary):
  nano_v2f                 - vanilla v2f prompt + nano (regression control)
  nano_v2f_speakerformat   - speakerformat prompt + nano
  nano_hyde_first_person   - HyDE first-person prompt + nano
  nano_hyde_first_person_filter - HyDE first-person + speaker_filter + nano

Each nano variant writes to a DEDICATED cache:
  cache/xmodel_<variant>_cache.json

Nano's reasoning tokens sometimes exhaust short budgets; we use
max_completion_tokens=6000 and re-call once if the response fails to
produce any parseable cue (post-filter retry).

No framework files are modified. No prior em_*.py files are modified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from em_architectures import (
    V2F_PROMPT,
    EMHit,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _MergedLLMCache,
    _query_em,
    format_primer_context,
    parse_v2f_cues,
)
from em_hyde_orient import (
    HYDE_FIRST_PERSON_PROMPT,
    parse_single_turn,
)
from em_retuned_cue_gen import (
    V2F_SPEAKERFORMAT_PROMPT,
)
from em_retuned_cue_gen import (
    parse_cues as parse_retuned_cues,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

CACHE_DIR = Path(__file__).resolve().parent / "cache"

NANO_MODEL = "gpt-5-nano"

# Dedicated caches per nano variant.
NANO_V2F_CACHE = CACHE_DIR / "xmodel_nano_v2f_cache.json"
NANO_SF_CACHE = CACHE_DIR / "xmodel_nano_v2f_speakerformat_cache.json"
NANO_HYDE_FP_CACHE = CACHE_DIR / "xmodel_nano_hyde_first_person_cache.json"


# --------------------------------------------------------------------------
# Nano LLM call helper (max_completion_tokens=6000, retry-on-empty-parse)
# --------------------------------------------------------------------------


async def _nano_llm_call_raw(
    openai_client,
    prompt: str,
    cache: _MergedLLMCache,
) -> tuple[str, bool]:
    """Call gpt-5-nano with reasoning-token budget; caches non-empty output."""
    cached = cache.get(NANO_MODEL, prompt)
    if cached is not None and cached.strip():
        return cached, True
    resp = await openai_client.chat.completions.create(
        model=NANO_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=6000,
    )
    text = resp.choices[0].message.content or ""
    if text.strip():
        cache.put(NANO_MODEL, prompt, text)
    return text, False


async def _nano_llm_call_with_retry(
    openai_client,
    prompt: str,
    cache: _MergedLLMCache,
    parse_fn,
    *,
    min_valid: int = 1,
) -> tuple[str, list, bool, bool]:
    """Call nano; parse output; if parse_fn returns < min_valid items and
    we did not hit cache, retry ONCE with the same prompt (no temperature
    lever on reasoning models).

    Returns (raw, parsed_items, cache_hit, retried).
    """
    raw, cache_hit = await _nano_llm_call_raw(openai_client, prompt, cache)
    parsed = parse_fn(raw) if raw else []
    retried = False
    if len(parsed) < min_valid and not cache_hit:
        # Bust the cache for this one call by re-invoking directly.
        resp = await openai_client.chat.completions.create(
            model=NANO_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=6000,
        )
        text2 = resp.choices[0].message.content or ""
        parsed2 = parse_fn(text2) if text2 else []
        retried = True
        if len(parsed2) >= len(parsed):
            # Overwrite cache with the better retry.
            if text2.strip():
                cache.put(NANO_MODEL, prompt, text2)
            raw = text2
            parsed = parsed2
    return raw, parsed, cache_hit, retried


# --------------------------------------------------------------------------
# Format-compliance helpers
# --------------------------------------------------------------------------


def speakerformat_compliant(cue: str, participants: tuple[str, str]) -> bool:
    """True iff cue starts with '<p_user>: ' or '<p_asst>: '."""
    p_user, p_asst = participants
    patterns = [
        re.compile(rf"^\s*{re.escape(p_user)}\s*:\s*\S"),
        re.compile(rf"^\s*{re.escape(p_asst)}\s*:\s*\S"),
    ]
    return any(p.match(cue) for p in patterns)


def hyde_first_person_compliant(
    turn: str | None, participants: tuple[str, str]
) -> bool:
    """True iff turn is a single '<speaker>: <content>' line."""
    if not turn:
        return False
    # parse_single_turn already enforces single-line format starting with
    # a participant name. If it extracted something, it's compliant.
    p_user, p_asst = participants
    patterns = [
        re.compile(rf"^\s*{re.escape(p_user)}\s*:\s*\S"),
        re.compile(rf"^\s*{re.escape(p_asst)}\s*:\s*\S"),
    ]
    return any(p.match(turn) for p in patterns)


# --------------------------------------------------------------------------
# Primer + merge helpers (shared pattern across all variants)
# --------------------------------------------------------------------------


async def _primer(
    memory: EventMemory, question: str, max_K: int
) -> tuple[list[EMHit], list[EMHit], str]:
    """Returns (primer_K10 for context_section, primer_full for merge, context_section)."""
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text} for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)
    primer_full = await _query_em(
        memory, question, vector_search_limit=max_K, expand_context=0
    )
    return primer_hits, primer_full, context_section


# --------------------------------------------------------------------------
# Variant results
# --------------------------------------------------------------------------


@dataclass
class XModelEMResult:
    hits: list[EMHit]
    metadata: dict


# --------------------------------------------------------------------------
# Variant: nano_v2f (vanilla V2F_PROMPT + nano)
# --------------------------------------------------------------------------


async def nano_v2f(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> XModelEMResult:
    _, primer_full, context_section = await _primer(memory, question, K)
    prompt = V2F_PROMPT.format(question=question, context_section=context_section)

    def _parse(raw: str) -> list[str]:
        return parse_v2f_cues(raw, max_cues=2)

    raw, cues, cache_hit, retried = await _nano_llm_call_with_retry(
        openai_client, prompt, cache, _parse, min_valid=1
    )

    cue_hits = []
    for cue in cues[:2]:
        cue_hits.append(
            await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
        )
    merged = _merge_by_max_score([primer_full, *cue_hits])
    return XModelEMResult(
        hits=merged[:K],
        metadata={
            "variant": "nano_v2f",
            "cues": cues,
            "raw_len": len(raw),
            "cache_hit": cache_hit,
            "retried": retried,
            "n_cues": len(cues),
            # v2f has no structural constraint; format_compliant is N/A.
            "format_compliant": None,
        },
    )


# --------------------------------------------------------------------------
# Variant: nano_v2f_speakerformat (V2F_SPEAKERFORMAT_PROMPT + nano)
# --------------------------------------------------------------------------


async def nano_v2f_speakerformat(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> XModelEMResult:
    p_user, p_asst = participants
    _, primer_full, context_section = await _primer(memory, question, K)
    prompt = V2F_SPEAKERFORMAT_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=p_user,
        participant_2=p_asst,
    )

    def _parse(raw: str) -> list[str]:
        return parse_retuned_cues(raw, max_cues=2)

    raw, cues, cache_hit, retried = await _nano_llm_call_with_retry(
        openai_client, prompt, cache, _parse, min_valid=1
    )

    compliant_flags = [speakerformat_compliant(c, participants) for c in cues]

    cue_hits = []
    for cue in cues[:2]:
        cue_hits.append(
            await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
        )
    merged = _merge_by_max_score([primer_full, *cue_hits])
    return XModelEMResult(
        hits=merged[:K],
        metadata={
            "variant": "nano_v2f_speakerformat",
            "cues": cues,
            "raw_len": len(raw),
            "cache_hit": cache_hit,
            "retried": retried,
            "n_cues": len(cues),
            "format_compliant": compliant_flags,
        },
    )


# --------------------------------------------------------------------------
# Variant: nano_hyde_first_person (HYDE_FIRST_PERSON_PROMPT + nano)
# --------------------------------------------------------------------------


async def nano_hyde_first_person(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> XModelEMResult:
    p_user, p_asst = participants
    _, primer_full, context_section = await _primer(memory, question, K)
    prompt = HYDE_FIRST_PERSON_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=p_user,
        participant_2=p_asst,
    )

    def _parse(raw: str) -> list[str]:
        t = parse_single_turn(raw, participants)
        return [t] if t else []

    raw, parsed, cache_hit, retried = await _nano_llm_call_with_retry(
        openai_client, prompt, cache, _parse, min_valid=1
    )
    turn = parsed[0] if parsed else None

    compliant = hyde_first_person_compliant(turn, participants)

    probe_hits = (
        await _query_em(memory, turn, vector_search_limit=K, expand_context=0)
        if turn
        else []
    )
    merged = _merge_by_max_score([primer_full, probe_hits])
    return XModelEMResult(
        hits=merged[:K],
        metadata={
            "variant": "nano_hyde_first_person",
            "turn": turn,
            "raw_len": len(raw),
            "cache_hit": cache_hit,
            "retried": retried,
            "format_compliant": compliant,
        },
    )
