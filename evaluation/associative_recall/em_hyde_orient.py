"""HyDE and orient-then-cue architectures for EventMemory on LoCoMo-30.

Mechanism-different retrievers (vs v2f which imagines adjacent chat):

HyDE (hypothetical answer generation):
  - em_hyde_narrative      : 1-2 paragraph narrative describing the answer
                              as if retelling what was discussed in the
                              conversation. Single embedded probe.
  - em_hyde_turn_format    : the hypothetical answer rendered as a sequence
                              of "{speaker_name}: ..." turns; each turn is
                              a separate probe, union by max score.
  - em_hyde_first_person   : the hypothetical answer as a single first-person
                              chat turn ("I remember when X said ...").
                              Single probe.

Orient-then-cue (two-stage):
  - em_orient_brief        : orientation = 1-sentence "what is this query
                              looking for?" Then cue-gen uses that
                              orientation as context. Cues in speakerformat.
  - em_orient_terminology  : orientation = expected vocabulary (the words
                              that would appear in target turns). Cues
                              explicitly include that vocabulary in
                              speakerformat.

All variants run a primer (raw-query K=10) first so cue generation sees
what has already been retrieved, matching the em_v2f pipeline shape.
The retrieval step itself always goes through `EventMemory.query()` via
`_query_em` from em_architectures (import-only).

Caches are dedicated so we never poison other specialists' caches:
    cache/hydeorient_<variant>_cache.json
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

CACHE_DIR = Path(__file__).resolve().parent / "cache"

HYDE_NARRATIVE_CACHE = CACHE_DIR / "hydeorient_hyde_narrative_cache.json"
HYDE_TURN_CACHE = CACHE_DIR / "hydeorient_hyde_turn_format_cache.json"
HYDE_FIRST_PERSON_CACHE = CACHE_DIR / "hydeorient_hyde_first_person_cache.json"
ORIENT_BRIEF_STAGE1_CACHE = CACHE_DIR / "hydeorient_orient_brief_stage1_cache.json"
ORIENT_BRIEF_STAGE2_CACHE = CACHE_DIR / "hydeorient_orient_brief_stage2_cache.json"
ORIENT_TERM_STAGE1_CACHE = CACHE_DIR / "hydeorient_orient_term_stage1_cache.json"
ORIENT_TERM_STAGE2_CACHE = CACHE_DIR / "hydeorient_orient_term_stage2_cache.json"


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


HYDE_NARRATIVE_PROMPT = """\
You will imagine the answer to a question about a conversation between \
{participant_1} and {participant_2}, as a NARRATIVE retelling of what \
was discussed. Your narrative will be embedded and compared via cosine \
similarity against the raw turns of the conversation.

Question: {question}

{context_section}

Write 1-2 short paragraphs (4-8 sentences total) narrating, in a \
natural descriptive voice, what the conversation must have contained \
that answers this question. Include concrete details, specific \
vocabulary, named objects/places, and the speakers' actions. Do NOT \
hedge ("maybe", "probably"); write as if you remember it happened.

Do NOT write questions. Do NOT prefix your output with labels. Output \
ONLY the narrative paragraph(s). Nothing else."""


HYDE_TURN_FORMAT_PROMPT = """\
You will imagine the portion of a chat between {participant_1} and \
{participant_2} that answers the question below. Each turn you write \
will be embedded independently and compared via cosine similarity \
against the real conversation turns, which are stored in the format:
"<speaker_name>: <chat content>"

Question: {question}

{context_section}

Write 3-5 turns that would plausibly appear in the conversation and \
collectively answer the question. Each turn MUST begin with \
"{participant_1}: " or "{participant_2}: ". Use natural chat \
register with concrete vocabulary the speakers would actually use. \
Do NOT write questions unless a chat turn would genuinely be a \
question. Do NOT put quotes around turn content.

Format (one turn per line, nothing else):
{participant_1}: <text>
{participant_2}: <text>
{participant_1}: <text>
..."""


HYDE_FIRST_PERSON_PROMPT = """\
You will imagine a single chat turn that directly answers the question \
below, written in first-person as if {participant_1} or {participant_2} \
were recalling the relevant conversation. The turn will be embedded and \
compared via cosine similarity against the real conversation turns, \
which are stored in the format "<speaker_name>: <chat content>".

Question: {question}

{context_section}

Write ONE chat turn that begins with "{participant_1}: " or \
"{participant_2}: " (whichever speaker is more likely to have said \
this). The content should be first-person and specific (e.g. "I \
remember when X mentioned Y, and then we ...") with concrete \
vocabulary. Do NOT write a question. Do NOT add quotes.

Output ONLY the single turn. Nothing else."""


ORIENT_BRIEF_STAGE1_PROMPT = """\
You will write a brief ORIENTATION that describes what this query is \
looking for in a conversation between {participant_1} and \
{participant_2}. The orientation will be used to guide a second step \
that generates search cues.

Question: {question}

Write ONE sentence (20-30 words) that captures:
- WHAT kind of content is being asked about (event, preference, \
decision, fact, feeling, schedule, etc.)
- WHICH speaker is most likely to have said it (or "either" if unclear)
- WHAT specific topic or entity is involved

Do NOT write a cue or a chat turn. Output ONLY the orientation \
sentence. Nothing else."""


ORIENT_BRIEF_STAGE2_PROMPT = """\
You are generating search cues for semantic retrieval over a \
conversation between {participant_1} and {participant_2}. Turns are \
embedded as "<speaker_name>: <chat content>", and your cues will be \
embedded the same way and compared via cosine similarity.

Question: {question}

Orientation (what this query is looking for): {orientation}

{context_section}

Generate 2 search cues. Each cue MUST begin with "{participant_1}: " \
or "{participant_2}: ". Use specific vocabulary that would actually \
appear in the target conversation turns. Align the cue content with \
the orientation above. Do NOT write questions.

Format:
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
Nothing else."""


ORIENT_TERMINOLOGY_STAGE1_PROMPT = """\
You will enumerate the expected VOCABULARY that would appear in the \
portion of a conversation between {participant_1} and {participant_2} \
that answers the question below. This vocabulary list will guide a \
second step that generates search cues.

Question: {question}

List 8-15 specific words or short phrases (nouns, verbs, named \
entities, adjectives) that a speaker answering this question would \
plausibly use. Prefer concrete and distinctive vocabulary over \
generic words. Do NOT include the speaker names themselves.

Format (comma-separated, nothing else):
word1, word2, phrase three, ..."""


ORIENT_TERMINOLOGY_STAGE2_PROMPT = """\
You are generating search cues for semantic retrieval over a \
conversation between {participant_1} and {participant_2}. Turns are \
embedded as "<speaker_name>: <chat content>", and your cues will be \
embedded the same way and compared via cosine similarity.

Question: {question}

Expected vocabulary (the content words that are likely to appear in \
the target turns): {vocabulary}

{context_section}

Generate 2 search cues that EXPLICITLY INCLUDE several words from the \
expected-vocabulary list. Each cue MUST begin with "{participant_1}: " \
or "{participant_2}: ". Do NOT write questions. Write text that would \
actually appear in a chat message.

Format:
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
Nothing else."""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

# Used for orient-then-cue variants (CUE: <speaker>: <text>).
CUE_RE = re.compile(
    r"^\s*(?:\[?[A-Z_]+\]?\s*[:\-]\s*)?CUE\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _strip_quotes(s: str) -> str:
    return s.strip().strip('"').strip("'").strip()


def parse_cues(response: str, max_cues: int = 2) -> list[str]:
    cues: list[str] = []
    for m in CUE_RE.finditer(response):
        cue = _strip_quotes(m.group(1))
        if cue:
            cues.append(cue)
        if len(cues) >= max_cues:
            break
    return cues


def parse_hyde_turns(
    response: str, participants: tuple[str, str], max_turns: int = 5
) -> list[str]:
    """Parse lines beginning with "<speaker_name>: " for either participant.

    Keeps the "<speaker>: <text>" prefix so the probe matches the embedded
    format. Strips stray quote chars and empty lines.
    """
    p_user, p_asst = participants
    patterns = [
        re.compile(rf"^\s*{re.escape(p_user)}\s*:\s*(.+?)\s*$"),
        re.compile(rf"^\s*{re.escape(p_asst)}\s*:\s*(.+?)\s*$"),
    ]
    turns: list[str] = []
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        for p, name in zip(patterns, (p_user, p_asst)):
            m = p.match(line)
            if m:
                content = _strip_quotes(m.group(1))
                if content:
                    turns.append(f"{name}: {content}")
                break
        if len(turns) >= max_turns:
            break
    return turns


def parse_single_turn(response: str, participants: tuple[str, str]) -> str | None:
    """Parse the first line that looks like "<speaker>: <content>"."""
    turns = parse_hyde_turns(response, participants, max_turns=1)
    return turns[0] if turns else None


# --------------------------------------------------------------------------
# LLM helper
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


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------


@dataclass
class HydeOrientResult:
    hits: list[EMHit]
    metadata: dict


async def _primer(
    memory: EventMemory, question: str, max_K: int
) -> tuple[list[EMHit], list[EMHit], str]:
    """Returns (primer_hits_K10, primer_full_maxK, context_section)."""
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


async def em_hyde_narrative(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> HydeOrientResult:
    p_user, p_asst = participants
    _, primer_full, context_section = await _primer(memory, question, K)
    prompt = HYDE_NARRATIVE_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=p_user,
        participant_2=p_asst,
    )
    raw, cache_hit = await _llm_call(openai_client, prompt, cache)
    probe = raw.strip()

    probe_hits = (
        await _query_em(memory, probe, vector_search_limit=K, expand_context=0)
        if probe
        else []
    )

    merged = _merge_by_max_score([primer_full, probe_hits])
    return HydeOrientResult(
        hits=merged[:K],
        metadata={
            "variant": "em_hyde_narrative",
            "probe": probe,
            "cache_hit": cache_hit,
        },
    )


async def em_hyde_turn_format(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
    max_turns: int = 5,
) -> HydeOrientResult:
    p_user, p_asst = participants
    _, primer_full, context_section = await _primer(memory, question, K)
    prompt = HYDE_TURN_FORMAT_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=p_user,
        participant_2=p_asst,
    )
    raw, cache_hit = await _llm_call(openai_client, prompt, cache)
    turns = parse_hyde_turns(raw, participants, max_turns=max_turns)

    batches = [primer_full]
    for t in turns:
        batches.append(
            await _query_em(memory, t, vector_search_limit=K, expand_context=0)
        )
    merged = _merge_by_max_score(batches)
    return HydeOrientResult(
        hits=merged[:K],
        metadata={
            "variant": "em_hyde_turn_format",
            "turns": turns,
            "n_turns": len(turns),
            "cache_hit": cache_hit,
        },
    )


async def em_hyde_first_person(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> HydeOrientResult:
    p_user, p_asst = participants
    _, primer_full, context_section = await _primer(memory, question, K)
    prompt = HYDE_FIRST_PERSON_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=p_user,
        participant_2=p_asst,
    )
    raw, cache_hit = await _llm_call(openai_client, prompt, cache)
    turn = parse_single_turn(raw, participants)

    probe_hits = (
        await _query_em(memory, turn, vector_search_limit=K, expand_context=0)
        if turn
        else []
    )
    merged = _merge_by_max_score([primer_full, probe_hits])
    return HydeOrientResult(
        hits=merged[:K],
        metadata={
            "variant": "em_hyde_first_person",
            "turn": turn,
            "cache_hit": cache_hit,
        },
    )


async def em_orient_brief(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    stage1_cache: _MergedLLMCache,
    stage2_cache: _MergedLLMCache,
    openai_client,
) -> HydeOrientResult:
    p_user, p_asst = participants
    _, primer_full, context_section = await _primer(memory, question, K)

    stage1_prompt = ORIENT_BRIEF_STAGE1_PROMPT.format(
        question=question, participant_1=p_user, participant_2=p_asst
    )
    orientation_raw, s1_hit = await _llm_call(
        openai_client, stage1_prompt, stage1_cache
    )
    orientation = orientation_raw.strip().splitlines()[0] if orientation_raw else ""

    stage2_prompt = ORIENT_BRIEF_STAGE2_PROMPT.format(
        question=question,
        orientation=orientation,
        context_section=context_section,
        participant_1=p_user,
        participant_2=p_asst,
    )
    cues_raw, s2_hit = await _llm_call(openai_client, stage2_prompt, stage2_cache)
    cues = parse_cues(cues_raw, max_cues=2)

    batches = [primer_full]
    for cue in cues:
        batches.append(
            await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
        )
    merged = _merge_by_max_score(batches)
    return HydeOrientResult(
        hits=merged[:K],
        metadata={
            "variant": "em_orient_brief",
            "orientation": orientation,
            "cues": cues,
            "stage1_cache_hit": s1_hit,
            "stage2_cache_hit": s2_hit,
        },
    )


async def em_orient_terminology(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    stage1_cache: _MergedLLMCache,
    stage2_cache: _MergedLLMCache,
    openai_client,
) -> HydeOrientResult:
    p_user, p_asst = participants
    _, primer_full, context_section = await _primer(memory, question, K)

    stage1_prompt = ORIENT_TERMINOLOGY_STAGE1_PROMPT.format(
        question=question, participant_1=p_user, participant_2=p_asst
    )
    vocab_raw, s1_hit = await _llm_call(openai_client, stage1_prompt, stage1_cache)
    vocabulary = vocab_raw.strip().splitlines()[0] if vocab_raw else ""

    stage2_prompt = ORIENT_TERMINOLOGY_STAGE2_PROMPT.format(
        question=question,
        vocabulary=vocabulary,
        context_section=context_section,
        participant_1=p_user,
        participant_2=p_asst,
    )
    cues_raw, s2_hit = await _llm_call(openai_client, stage2_prompt, stage2_cache)
    cues = parse_cues(cues_raw, max_cues=2)

    batches = [primer_full]
    for cue in cues:
        batches.append(
            await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
        )
    merged = _merge_by_max_score(batches)
    return HydeOrientResult(
        hits=merged[:K],
        metadata={
            "variant": "em_orient_terminology",
            "vocabulary": vocabulary,
            "cues": cues,
            "stage1_cache_hit": s1_hit,
            "stage2_cache_hit": s2_hit,
        },
    )
