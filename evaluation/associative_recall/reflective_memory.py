"""Reflective LLM writes-to-memory for EventMemory retrieval on LoCoMo-30.

Architecture: the LLM treats its own reasoning as memory entries. For each
query q, a per-query SCRATCH memory accumulates short reflection sentences
(learned / still_need) that the LLM produces after each round. These
reflections are embedded with the same embedder as the corpus and act as
AUXILIARY PROBES: high-scoring scratch entries are re-queried against the
corpus, expanding the retrieval pool in round N+1.

This differs from prior iterative agents (hypothesis_driven, working_memory
buffer, v15_conditional_hop2) in that reflections become FIRST-CLASS
INDEXABLE ENTRIES (embedded, cosine-searchable), not just LLM context.

Variants:
  reflmem_1round         -- single round: cue-gen -> retrieve -> reflect ->
                            write scratch -> scratch-entries re-probe ->
                            merge. Ablation: does WRITING help without
                            iteration?
  reflmem_2round         -- two full rounds. Round 2's cue-gen is informed
                            by the accumulated scratch state. Final retrieval
                            = corpus hits merged with scratch-entry re-probes.
  reflmem_2round_filter  -- reflmem_2round + speaker_filter topup
                            (composed in the eval driver, same scheme as
                            em_hyde_first_person + speaker_filter).

Prompts: speakerformat cue-gen (aligned with embedded `"<speaker>: <text>"`
distribution), plus a REFLECTION prompt that returns JSON-structured
{learned: [...], still_need: [...]}.

Scratch memory:
- Per-query, in-memory list of (text, embedding) tuples.
- Novel scratch entries embedded via the same embedder as the corpus.
- Round 2 treats each scratch entry text as an independent probe: queries
  EventMemory with that text and unions the results by max-score per
  turn_id.
- Stats: scratch entries written per query, rounds executed, round-2
  novelty rate.

Caches (dedicated so we never poison other specialists'):
  cache/reflmem_cuegen_r{N}_cache.json
  cache/reflmem_reflect_cache.json
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from memmachine_server.common.embedder.embedder import Embedder
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

from em_architectures import (
    V2F_MODEL,
    EMHit,
    _MergedLLMCache,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _query_em,
    format_primer_context,
)


CACHE_DIR = Path(__file__).resolve().parent / "cache"

REFLMEM_CUEGEN_R1_CACHE = CACHE_DIR / "reflmem_cuegen_r1_cache.json"
REFLMEM_REFLECT_CACHE = CACHE_DIR / "reflmem_reflect_cache.json"
REFLMEM_CUEGEN_R2_CACHE = CACHE_DIR / "reflmem_cuegen_r2_cache.json"


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


# Round-1 cue-gen is the same speakerformat prompt used elsewhere, kept
# local here so we have a dedicated cache and don't accidentally reuse
# other specialists' caches.
CUEGEN_R1_PROMPT = """\
You are generating search cues for semantic retrieval over a conversation \
between {participant_1} and {participant_2}. Turns are embedded in the \
format "<speaker_name>: <chat content>" and your cues will be embedded \
the same way.

Question: {question}

{context_section}

Generate 2 search cues. Each cue MUST begin with "{participant_1}: " or \
"{participant_2}: ". Use specific vocabulary that would appear in target \
turns. Do NOT write questions; write text that would actually appear in \
a chat message.

Format:
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
Nothing else."""


# Reflection prompt. Output must be JSON so parse is deterministic.
REFLECT_PROMPT = """\
You are reflecting on retrieval progress for a question about a \
conversation between {participant_1} and {participant_2}.

Question: {question}

Retrieved so far:
{retrieved_section}

Reflect on this retrieval. Produce two short lists:
- `learned`: 2-4 short declarative statements about what the retrieval \
  HAS surfaced that is relevant to the question. Each statement must be \
  CONCRETE and use vocabulary from the retrieved text. Each 10-25 words.
- `still_need`: 2-4 short statements describing what is STILL MISSING to \
  fully answer the question. Each statement should be declarative or \
  interrogative, concrete, and hint at vocabulary likely to appear in the \
  missing turns. Each 10-25 words.

These statements will be embedded and used as additional search probes \
against the conversation. Be specific; use the speakers' actual terminology.

Output ONLY a JSON object, no prose:
{{"learned": ["...", "..."], "still_need": ["...", "..."]}}"""


# Round-2 cue-gen, informed by the reflection state.
CUEGEN_R2_PROMPT = """\
You are generating NEW search cues for semantic retrieval over a \
conversation between {participant_1} and {participant_2}. Turns are \
embedded in the format "<speaker_name>: <chat content>".

Question: {question}

What has been retrieved and reflected on so far:
LEARNED:
{learned_section}
STILL NEED:
{still_need_section}

Generate 2 NEW search cues that target the STILL NEED gaps. Each cue \
MUST begin with "{participant_1}: " or "{participant_2}: ". Use \
vocabulary distinct from the LEARNED section; shift the probe toward \
UNRETRIEVED regions of the conversation. Do NOT write questions.

Format:
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
Nothing else."""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


CUE_RE = re.compile(
    r"^\s*CUE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE
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


def parse_reflection(response: str) -> tuple[list[str], list[str]]:
    """Parse reflection JSON. Returns (learned, still_need).

    Tolerant of common deviations: leading/trailing text, markdown fences.
    """
    if not response:
        return [], []
    # Strip markdown fences.
    text = response.strip()
    if text.startswith("```"):
        # remove first fence line
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # remove trailing fence
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Extract the first JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return [], []
    blob = text[start : end + 1]
    try:
        obj = json.loads(blob)
    except Exception:
        return [], []
    learned = obj.get("learned") or []
    still_need = obj.get("still_need") or []
    # Coerce to list[str].
    learned = [str(x).strip() for x in learned if str(x).strip()]
    still_need = [str(x).strip() for x in still_need if str(x).strip()]
    return learned, still_need


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
# Scratch memory
# --------------------------------------------------------------------------


@dataclass
class ScratchMemory:
    """In-memory scratch: list of (text, embedding) tuples.

    Not persisted. Embeddings stored as float32 numpy arrays.
    """

    texts: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)  # "learned" | "still_need"
    rounds: list[int] = field(default_factory=list)  # which round wrote it
    embeddings: np.ndarray | None = None  # (N, D)

    async def extend(
        self,
        new_texts: list[str],
        kind: str,
        round_idx: int,
        embedder: Embedder,
    ) -> None:
        if not new_texts:
            return
        embs = await embedder.search_embed(new_texts)
        arr = np.asarray(embs, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = arr
        else:
            self.embeddings = np.vstack([self.embeddings, arr])
        self.texts.extend(new_texts)
        self.kinds.extend([kind] * len(new_texts))
        self.rounds.extend([round_idx] * len(new_texts))

    def score_against_query(self, query_emb: np.ndarray) -> list[float]:
        """Cosine similarity of every scratch entry vs query_emb.

        Assumes both are normalized. OpenAI embeddings are L2-normalized,
        but we renormalize defensively here.
        """
        if self.embeddings is None or len(self.texts) == 0:
            return []
        q = query_emb.astype(np.float32)
        qn = np.linalg.norm(q) + 1e-12
        q = q / qn
        E = self.embeddings
        En = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        Enorm = E / En
        sims = Enorm @ q
        return [float(x) for x in sims.tolist()]


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class ReflMemResult:
    hits: list[EMHit]
    metadata: dict


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def format_retrieved_for_reflection(
    hits: list[EMHit], max_items: int = 10, max_len: int = 180
) -> str:
    if not hits:
        return "(no retrievals yet)"
    # Sort by score descending, then take first max_items.
    top = sorted(hits, key=lambda h: -h.score)[:max_items]
    # Re-sort by turn_id for readability.
    top = sorted(top, key=lambda h: h.turn_id)
    lines = []
    for h in top:
        txt = h.text.replace("\n", " ")
        if len(txt) > max_len:
            txt = txt[:max_len] + "..."
        lines.append(f"[Turn {h.turn_id}, {h.role}]: {txt}")
    return "\n".join(lines)


def format_list_section(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {s}" for s in items)


async def _run_round1_cuegen_and_retrieve(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[EMHit], list[str], list[list[EMHit]], bool]:
    """Returns (primer_full_K, cues, per_cue_hits_K, cache_hit)."""
    p_user, p_asst = participants
    primer_hits_10 = _dedupe_by_turn_id(
        await _query_em(
            memory, question, vector_search_limit=10, expand_context=0
        )
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text}
        for h in primer_hits_10
    ]
    context_section = format_primer_context(primer_segments)
    prompt = CUEGEN_R1_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=p_user,
        participant_2=p_asst,
    )
    raw, cache_hit = await _llm_call(openai_client, prompt, cache)
    cues = parse_cues(raw, max_cues=2)
    per_cue_hits: list[list[EMHit]] = []
    for cue in cues:
        per_cue_hits.append(
            await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
        )
    primer_for_merge = await _query_em(
        memory, question, vector_search_limit=K, expand_context=0
    )
    return primer_for_merge, cues, per_cue_hits, cache_hit


async def _reflect(
    question: str,
    participants: tuple[str, str],
    retrieved_hits: list[EMHit],
    *,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[str], list[str], bool]:
    p_user, p_asst = participants
    retrieved_section = format_retrieved_for_reflection(retrieved_hits)
    prompt = REFLECT_PROMPT.format(
        question=question,
        retrieved_section=retrieved_section,
        participant_1=p_user,
        participant_2=p_asst,
    )
    raw, cache_hit = await _llm_call(openai_client, prompt, cache)
    learned, still_need = parse_reflection(raw)
    return learned, still_need, cache_hit


async def _round2_cuegen_and_retrieve(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    scratch: ScratchMemory,
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[str], list[list[EMHit]], bool]:
    p_user, p_asst = participants
    learned = [t for t, k in zip(scratch.texts, scratch.kinds) if k == "learned"]
    still_need = [t for t, k in zip(scratch.texts, scratch.kinds) if k == "still_need"]
    prompt = CUEGEN_R2_PROMPT.format(
        question=question,
        learned_section=format_list_section(learned),
        still_need_section=format_list_section(still_need),
        participant_1=p_user,
        participant_2=p_asst,
    )
    raw, cache_hit = await _llm_call(openai_client, prompt, cache)
    cues = parse_cues(raw, max_cues=2)
    per_cue_hits: list[list[EMHit]] = []
    for cue in cues:
        per_cue_hits.append(
            await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
        )
    return cues, per_cue_hits, cache_hit


async def _scratch_reprobe(
    memory: EventMemory,
    scratch: ScratchMemory,
    question_emb: np.ndarray,
    *,
    K: int,
    top_scratch: int = 3,
) -> tuple[list[list[EMHit]], list[str], list[float]]:
    """Pick top-`top_scratch` scratch entries by cosine vs query, use each
    as an independent probe against EventMemory. Returns per-probe hit
    batches plus the chosen probe texts and their cosine scores.
    """
    if scratch.embeddings is None or len(scratch.texts) == 0:
        return [], [], []
    sims = scratch.score_against_query(question_emb)
    # Take top-k scratch entries (prefer higher sim).
    order = sorted(range(len(scratch.texts)), key=lambda i: -sims[i])
    chosen_texts: list[str] = []
    chosen_scores: list[float] = []
    per_probe_hits: list[list[EMHit]] = []
    for idx in order[:top_scratch]:
        text = scratch.texts[idx]
        chosen_texts.append(text)
        chosen_scores.append(sims[idx])
        per_probe_hits.append(
            await _query_em(memory, text, vector_search_limit=K, expand_context=0)
        )
    return per_probe_hits, chosen_texts, chosen_scores


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------


async def reflmem_1round(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    embedder: Embedder,
    cuegen_r1_cache: _MergedLLMCache,
    reflect_cache: _MergedLLMCache,
    openai_client,
    top_scratch: int = 3,
) -> ReflMemResult:
    """Single round with scratch-memory write and re-probe.

    Round 1:
      - Generate cues, retrieve (primer + 2 cue batches).
      - Reflect -> write learned/still_need to scratch.
      - Re-probe corpus with top-`top_scratch` scratch entries by cosine.
      - Merge all batches by max score per turn_id.
    """
    # Embed the query once (for scratch scoring).
    q_emb_list = await embedder.search_embed([question])
    q_emb = np.asarray(q_emb_list[0], dtype=np.float32)

    primer, cues_r1, cue_hits_r1, cache_hit_r1 = (
        await _run_round1_cuegen_and_retrieve(
            memory, question, participants,
            K=K, cache=cuegen_r1_cache, openai_client=openai_client,
        )
    )
    batches_r1 = [primer, *cue_hits_r1]
    merged_r1 = _merge_by_max_score(batches_r1)

    # Reflect on round-1 top hits.
    learned, still_need, reflect_cache_hit = await _reflect(
        question, participants, merged_r1[:K],
        cache=reflect_cache, openai_client=openai_client,
    )

    scratch = ScratchMemory()
    await scratch.extend(learned, "learned", 1, embedder)
    await scratch.extend(still_need, "still_need", 1, embedder)

    # Re-probe with top scratch entries.
    reprobe_batches, reprobe_texts, reprobe_scores = await _scratch_reprobe(
        memory, scratch, q_emb, K=K, top_scratch=top_scratch,
    )

    all_batches = [primer, *cue_hits_r1, *reprobe_batches]
    final = _merge_by_max_score(all_batches)

    # Track novelty: did scratch reprobes add any turn_ids not in merged_r1?
    r1_turns = {h.turn_id for h in merged_r1[:K]}
    reprobe_union: set[int] = set()
    for b in reprobe_batches:
        for h in b:
            reprobe_union.add(h.turn_id)
    novel_turns = reprobe_union - r1_turns

    return ReflMemResult(
        hits=final[:K],
        metadata={
            "variant": "reflmem_1round",
            "cues_r1": cues_r1,
            "cuegen_r1_cache_hit": cache_hit_r1,
            "reflect_cache_hit": reflect_cache_hit,
            "learned": learned,
            "still_need": still_need,
            "scratch_entries": len(scratch.texts),
            "reprobe_texts": reprobe_texts,
            "reprobe_scores": [round(s, 4) for s in reprobe_scores],
            "n_novel_turns_from_scratch": len(novel_turns),
            "rounds_executed": 1,
        },
    )


async def reflmem_2round(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    embedder: Embedder,
    cuegen_r1_cache: _MergedLLMCache,
    reflect_cache: _MergedLLMCache,
    cuegen_r2_cache: _MergedLLMCache,
    openai_client,
    top_scratch: int = 3,
) -> ReflMemResult:
    """Two full rounds.

    Round 1: cue-gen -> retrieve -> reflect -> write scratch.
    Round 2: cue-gen informed by scratch -> retrieve (new cues) + scratch
             reprobe. Merge EVERYTHING.
    """
    q_emb_list = await embedder.search_embed([question])
    q_emb = np.asarray(q_emb_list[0], dtype=np.float32)

    primer, cues_r1, cue_hits_r1, cache_hit_r1 = (
        await _run_round1_cuegen_and_retrieve(
            memory, question, participants,
            K=K, cache=cuegen_r1_cache, openai_client=openai_client,
        )
    )
    batches_r1 = [primer, *cue_hits_r1]
    merged_r1 = _merge_by_max_score(batches_r1)

    # Round 1 reflection.
    learned, still_need, reflect_cache_hit = await _reflect(
        question, participants, merged_r1[:K],
        cache=reflect_cache, openai_client=openai_client,
    )

    scratch = ScratchMemory()
    await scratch.extend(learned, "learned", 1, embedder)
    await scratch.extend(still_need, "still_need", 1, embedder)

    # Round 2 cue-gen.
    cues_r2, cue_hits_r2, cache_hit_r2 = await _round2_cuegen_and_retrieve(
        memory, question, participants, scratch,
        K=K, cache=cuegen_r2_cache, openai_client=openai_client,
    )

    # Scratch reprobe against the corpus (auxiliary probes).
    reprobe_batches, reprobe_texts, reprobe_scores = await _scratch_reprobe(
        memory, scratch, q_emb, K=K, top_scratch=top_scratch,
    )

    all_batches = [primer, *cue_hits_r1, *cue_hits_r2, *reprobe_batches]
    final = _merge_by_max_score(all_batches)

    # Novelty from round 2 (new cues + scratch reprobes) over round 1.
    r1_turns = {h.turn_id for h in merged_r1[:K]}
    r2_union: set[int] = set()
    for b in cue_hits_r2:
        for h in b:
            r2_union.add(h.turn_id)
    for b in reprobe_batches:
        for h in b:
            r2_union.add(h.turn_id)
    novel_turns = r2_union - r1_turns

    return ReflMemResult(
        hits=final[:K],
        metadata={
            "variant": "reflmem_2round",
            "cues_r1": cues_r1,
            "cues_r2": cues_r2,
            "cuegen_r1_cache_hit": cache_hit_r1,
            "cuegen_r2_cache_hit": cache_hit_r2,
            "reflect_cache_hit": reflect_cache_hit,
            "learned": learned,
            "still_need": still_need,
            "scratch_entries": len(scratch.texts),
            "reprobe_texts": reprobe_texts,
            "reprobe_scores": [round(s, 4) for s in reprobe_scores],
            "n_novel_turns_round2": len(novel_turns),
            "rounds_executed": 2,
        },
    )
