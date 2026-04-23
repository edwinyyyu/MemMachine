"""Best-shot comparison: every architecture gets optimized prompts.

Each architecture uses v15-quality prompt language for cue/query generation,
includes primer retrieval where applicable, and uses V2f's completeness
hint + anti-question instruction.

Architectures:
  1. decompose_then_retrieve (+ primer context)
  2. interleaved (fixed prompts)
  3. frontier_v2_iterative (v15-style reflect, 1 gap per round)
  4. meta_v2f (already optimized — the reference "best prompt")
  5. flat_multi_cue (+ primer context)
  6. retrieve_then_decompose (fixed prompts)
  7. v15_control (reference baseline)

Usage:
    uv run python best_shot.py [--arch <name>] [--all] [--list]
    uv run python best_shot.py --all --verbose
"""

import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
    RetrievalResult,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50, 100]


# ---------------------------------------------------------------------------
# Cache classes — bestshot-specific, reads from all existing caches
# ---------------------------------------------------------------------------
class BestshotEmbeddingCache(EmbeddingCache):
    """Reads all existing embedding caches, writes to bestshot-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in (
            "embedding_cache.json",
            "arch_embedding_cache.json",
            "agent_embedding_cache.json",
            "frontier_embedding_cache.json",
            "meta_embedding_cache.json",
            "optim_embedding_cache.json",
            "synth_test_embedding_cache.json",
            "bestshot_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "bestshot_embedding_cache.json"
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


class BestshotLLMCache(LLMCache):
    """Reads all existing LLM caches, writes to bestshot-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in (
            "llm_cache.json",
            "arch_llm_cache.json",
            "agent_llm_cache.json",
            "tree_llm_cache.json",
            "frontier_llm_cache.json",
            "meta_llm_cache.json",
            "optim_llm_cache.json",
            "synth_test_llm_cache.json",
            "bestshot_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "bestshot_llm_cache.json"
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


# ---------------------------------------------------------------------------
# Prompt templates — all fixed with v15-quality language
# ---------------------------------------------------------------------------

# -- V15 control (exact v15 prompt, reference) --
V15_CONTROL_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# -- V2f (v15 + completeness + anti-question) --
V2F_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# -- Decompose prompt (with primer context, v15-quality) --
BESTSHOT_DECOMPOSE_PROMPT = """\
You are decomposing a question about a past conversation into sub-questions \
for embedding-based retrieval. Each sub-question will be embedded and \
compared via cosine similarity against stored conversation turns.

QUESTION: {question}

Here is what an initial retrieval found — these conversation segments are \
already retrieved:

RETRIEVED:
{primer_context}

Based on what HAS been found and what is still MISSING, break the question \
into 2-3 focused sub-questions. Each sub-question should target a DIFFERENT \
aspect of the answer.

Each sub-question should use vocabulary and phrasing that matches the \
retrieved content above — the kind of text someone would actually say in \
a conversation.
Do NOT write questions ("Did you mention X?") or search commands \
("Search for...").
Write text that looks like it could be an excerpt from the content \
being searched.

If the question implies MULTIPLE items, keep searching for more even if \
some are already found.

Format — exactly 2-3 lines:
SUB: <text mimicking conversation content>
SUB: <text mimicking conversation content>
Nothing else."""

# -- Followup cue prompt (for decompose_then_retrieve branches) --
BESTSHOT_FOLLOWUP_CUE_PROMPT = """\
You are searching a conversation history to answer a question. Your cues \
will be embedded and compared via cosine similarity against stored content.

ORIGINAL QUESTION: {question}
SUB-QUESTION: {sub_question}

RETRIEVED FOR THIS BRANCH:
{retrieved_context}

Briefly assess: what aspect of the sub-question is NOT covered by the \
retrieved content?

Generate one search cue targeting the missing content. Use vocabulary and \
phrasing that matches conversation content — the kind of text someone \
would actually type in a chat.
Do NOT write questions ("Did you mention X?") or search commands.
Write text that looks like it could be an excerpt from the content \
being searched.

Format:
ASSESSMENT: <what's missing>
CUE: <text mimicking conversation content>"""

# -- Grounded decompose prompt (for interleaved and retrieve_then_decompose) --
BESTSHOT_GROUNDED_DECOMPOSE_PROMPT = """\
You are identifying gaps in retrieval results for a question about a past \
conversation. Your gap queries will be embedded and compared via cosine \
similarity against stored conversation turns.

QUESTION: {question}

Here is what an initial search found — these are conversation segments \
that are already retrieved:

RETRIEVED:
{retrieved_context}

Based on what HAS been found, identify what is still MISSING to answer \
the question. Generate 2-3 focused search cues targeting the GAPS — \
aspects of the question NOT covered by the retrieved content.

Each cue should use vocabulary and phrasing that matches the retrieved \
content above. Write text that would actually appear in a chat message.
Do NOT write questions ("Did you mention X?") or search commands \
("Search for...").
Do NOT use boolean operators (OR, AND).

If the question implies MULTIPLE items, keep searching for more even if \
some are already found.

Format — exactly 2-3 lines:
GAP: <text mimicking conversation content>
GAP: <text mimicking conversation content>
Nothing else."""

# -- Branch assess prompt (for interleaved per-branch assessment) --
BESTSHOT_BRANCH_ASSESS_PROMPT = """\
You searched a conversation history for a specific sub-question. Your \
cues will be embedded and compared via cosine similarity.

ORIGINAL QUESTION: {question}
SUB-QUESTION: {sub_question}

RETRIEVED:
{retrieved_context}

If the retrieved content adequately covers the sub-question, respond with \
just "DONE" on a line by itself.

If a specific piece is still missing, generate ONE short search cue \
using vocabulary that would appear in the missing conversation content.
Write text that looks like it could be an excerpt from the content \
being searched. Do NOT write questions or search commands.

Format (pick one):
DONE
or
CUE: <text mimicking conversation content>"""

# -- Deepen or pop prompt (for interleaved iterative deepening) --
BESTSHOT_DEEPEN_PROMPT = """\
You are navigating a reasoning tree to answer a question about a past \
conversation. Your sub-questions will be embedded and compared via cosine \
similarity against stored content.

QUESTION: {question}
PATH: {context_path}
DEPTH: {depth}/{max_depth}

RETRIEVED AT THIS LEVEL ({total_retrieved} total so far):
{current_retrieved}

ALL RETRIEVED SO FAR:
{all_retrieved}

If the retrieved content covers this branch well, respond with POP (done).
If important aspects are still missing and you can identify specific \
sub-topics to search, respond with PUSH and 1-2 sub-questions.

Each sub-question must use vocabulary that would appear in the target \
conversation turns. Write text that looks like conversation content, \
not search commands or meta-instructions.

Respond:
ACTION: PUSH or POP
SUB: <text mimicking conversation content>  (only if PUSH, 1-2 lines)"""

# -- V15-style single cue prompt (for retrieve_then_decompose phase 1b) --
BESTSHOT_V15_SINGLE_CUE_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cue will be embedded and compared via cosine similarity.

QUESTION: {question}

RETRIEVED SO FAR:
{retrieved_context}

Briefly assess: how is this search going? What content is still missing?

Then generate one search cue targeting the most important missing content. \
Use specific vocabulary that would appear in the target conversation turns.
Write text that looks like conversation content, not questions or search \
commands.

If the question implies MULTIPLE items, keep searching for more even if \
some are already found.

Format:
ASSESSMENT: <1-2 sentence evaluation>
CUE: <text mimicking conversation content>"""

# -- Frontier iterative reflect prompt (v15-style, 1 gap per round) --
BESTSHOT_FRONTIER_REFLECT_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}{explored_text}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate exactly 1 search cue targeting the most important missing \
content. Use specific vocabulary that would appear in the target \
conversation turns.

Do NOT write questions ("Did you mention X?") or search commands. \
Write text that would actually appear in a chat message.

If the retrieval looks complete, respond with DONE.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
(or)
ASSESSMENT: <evaluation>
DONE"""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BestshotBase:
    """Base class with embedding/LLM utilities and counters."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = BestshotEmbeddingCache()
        self.llm_cache = BestshotLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, embedding)
        self.embed_calls += 1
        return embedding

    def llm_call(self, prompt: str, model: str = MODEL) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(model, prompt, text)
        self.llm_calls += 1
        return text

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_segments(segments: list[Segment], max_items: int = 12,
                     max_chars: int = 250) -> str:
    """Format segments chronologically. Matches v15 control format."""
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def _parse_cues(response: str) -> list[str]:
    """Parse CUE: lines from LLM response."""
    cues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


def _parse_subs(response: str) -> list[str]:
    """Parse SUB: lines from LLM response."""
    subs = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("SUB:"):
            sub = line[4:].strip()
            if sub:
                subs.append(sub)
    return subs


def _parse_gaps(response: str) -> list[str]:
    """Parse GAP: lines from LLM response."""
    gaps = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("GAP:"):
            gap = line[4:].strip()
            if gap:
                gaps.append(gap)
    return gaps


def _build_context_section(
    all_segments: list[Segment],
    new_segments: list[Segment] | None = None,
    hop_number: int = 1,
    previous_cues: list[str] | None = None,
) -> str:
    """Build context section matching v15's accumulated format exactly."""
    if not all_segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    sorted_segs = sorted(all_segments, key=lambda s: s.turn_id)
    context_lines = []
    display_limit = 12 if hop_number <= 2 else 16
    for seg in sorted_segs[:display_limit]:
        context_lines.append(
            f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:250]}"
        )
    context_section = (
        "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + "\n".join(context_lines)
    )
    if new_segments and hop_number > 1:
        latest_lines = []
        for seg in sorted(new_segments, key=lambda s: s.turn_id)[:6]:
            latest_lines.append(
                f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:200]}"
            )
        context_section += (
            "\n\nMOST RECENTLY FOUND (last hop):\n" + "\n".join(latest_lines)
        )
    if previous_cues:
        context_section += (
            "\n\nPREVIOUS CUES ALREADY TRIED (do NOT repeat or paraphrase):\n"
            + "\n".join(f"- {c}" for c in previous_cues)
        )
    return context_section


@dataclass
class BestshotResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


# ===========================================================================
# 7. v15_control — reference baseline (no changes)
# ===========================================================================
class V15Control(BestshotBase):
    """Exact v15 prompt: question top-10, 1 LLM call, 2 cues top-10 each."""

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        # Hop 0: embed question, retrieve top-10
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        # Build context section
        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(all_segments)
        )

        # LLM call
        prompt = V15_CONTROL_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return BestshotResult(
            segments=all_segments,
            metadata={"name": "v15_control", "output": output, "cues": cues[:2]},
        )


# ===========================================================================
# 4. meta_v2f — already-optimized v15 + completeness + anti-question
# ===========================================================================
class MetaV2f(BestshotBase):
    """V2f prompt: v15 + completeness hint + anti-question instruction."""

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(all_segments)
        )

        prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return BestshotResult(
            segments=all_segments,
            metadata={"name": "meta_v2f", "output": output, "cues": cues[:2]},
        )


# ===========================================================================
# 1. decompose_then_retrieve — with primer retrieval before decomposition
# ===========================================================================
class DecomposeThenRetrieve(BestshotBase):
    """Primer retrieval -> grounded decomposition -> per-branch retrieve + followup."""

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        exclude: set[int] = set()
        all_segments: list[Segment] = []

        # Primer: retrieve top-10 with raw query to give decomposer context
        query_emb = self.embed_text(question)
        primer = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        primer_segs = list(primer.segments)
        all_segments.extend(primer_segs)
        for s in primer_segs:
            exclude.add(s.index)

        # Decompose with primer context
        primer_context = _format_segments(primer_segs, max_items=10)
        decompose_prompt = BESTSHOT_DECOMPOSE_PROMPT.format(
            question=question, primer_context=primer_context
        )
        decompose_output = self.llm_call(decompose_prompt)
        sub_questions = _parse_subs(decompose_output)
        if not sub_questions:
            sub_questions = [question]
        sub_questions = sub_questions[:4]

        # Per branch: retrieve + followup cue
        cues_used = []
        for sq in sub_questions:
            sq_emb = self.embed_text(sq)
            result = self.store.search(
                sq_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            branch_segs = []
            for seg in result.segments:
                if seg.index not in exclude:
                    branch_segs.append(seg)
                    all_segments.append(seg)
                    exclude.add(seg.index)

            # Followup cue
            context = _format_segments(branch_segs, max_items=8)
            followup_prompt = BESTSHOT_FOLLOWUP_CUE_PROMPT.format(
                question=question,
                sub_question=sq,
                retrieved_context=context,
            )
            followup_output = self.llm_call(followup_prompt)
            followup_cues = _parse_cues(followup_output)
            if followup_cues:
                cue = followup_cues[0]
                cues_used.append(cue)
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb, top_k=10, conversation_id=conversation_id,
                    exclude_indices=exclude,
                )
                for seg in result.segments:
                    if seg.index not in exclude:
                        all_segments.append(seg)
                        exclude.add(seg.index)

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": "decompose_then_retrieve",
                "sub_questions": sub_questions,
                "cues": cues_used,
            },
        )


# ===========================================================================
# 2. interleaved — retrieve first, grounded decompose, per-branch assess
# ===========================================================================
class Interleaved(BestshotBase):
    """Phase 1: question retrieval. Phase 2: grounded decompose on gaps.
    Phase 3: per-gap retrieval + per-branch assessment."""

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        exclude: set[int] = set()
        all_segments: list[Segment] = []

        # Phase 1: initial retrieval
        query_emb = self.embed_text(question)
        result = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        initial_segs = list(result.segments)
        all_segments.extend(initial_segs)
        for s in initial_segs:
            exclude.add(s.index)

        # Phase 2: grounded decomposition
        context = _format_segments(initial_segs, max_items=10)
        decompose_prompt = BESTSHOT_GROUNDED_DECOMPOSE_PROMPT.format(
            question=question, retrieved_context=context
        )
        decompose_output = self.llm_call(decompose_prompt)
        sub_questions = _parse_gaps(decompose_output)
        if not sub_questions:
            sub_questions = [question]
        sub_questions = sub_questions[:4]

        # Phase 3: per-gap retrieval + per-branch assessment
        cues_used = []
        for sq in sub_questions:
            sq_emb = self.embed_text(sq)
            result = self.store.search(
                sq_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            branch_segs = []
            for seg in result.segments:
                if seg.index not in exclude:
                    branch_segs.append(seg)
                    all_segments.append(seg)
                    exclude.add(seg.index)

            # Per-branch assessment
            branch_context = _format_segments(branch_segs, max_items=8)
            assess_prompt = BESTSHOT_BRANCH_ASSESS_PROMPT.format(
                question=question,
                sub_question=sq,
                retrieved_context=branch_context,
            )
            assess_output = self.llm_call(assess_prompt)

            if "DONE" not in assess_output.upper().split("\n")[0]:
                assess_cues = _parse_cues(assess_output)
                if assess_cues:
                    cue = assess_cues[0]
                    cues_used.append(cue)
                    cue_emb = self.embed_text(cue)
                    result = self.store.search(
                        cue_emb, top_k=10, conversation_id=conversation_id,
                        exclude_indices=exclude,
                    )
                    for seg in result.segments:
                        if seg.index not in exclude:
                            all_segments.append(seg)
                            exclude.add(seg.index)

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": "interleaved",
                "sub_questions": sub_questions,
                "cues": cues_used,
            },
        )


# ===========================================================================
# 3. frontier_v2_iterative — v15-style reflect, 1 gap per round, max 4
# ===========================================================================
class FrontierV2Iterative(BestshotBase):
    """Iterative frontier: initial probe, then reflect-explore loops.
    1 gap per reflect round, max 4 reflects. v15-quality prompts."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 max_reflects: int = 4, segment_budget: int = 80):
        super().__init__(store, client)
        self.max_reflects = max_reflects
        self.segment_budget = segment_budget

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        exclude: set[int] = set()
        all_segments: list[Segment] = []
        reflect_log: list[dict] = []

        # Phase 0: initial probe
        query_emb = self.embed_text(question)
        result = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments.extend(result.segments)
        for s in result.segments:
            exclude.add(s.index)

        for reflect_i in range(self.max_reflects):
            if len(all_segments) >= self.segment_budget:
                break

            context = _format_segments(all_segments)
            explored_text = ""
            if reflect_log:
                explored = []
                for entry in reflect_log:
                    for g in entry.get("gaps", []):
                        explored.append(f"- {g}")
                explored_text = (
                    "\n\nALREADY SEARCHED FOR (do NOT repeat these):\n"
                    + "\n".join(explored)
                )

            prompt = BESTSHOT_FRONTIER_REFLECT_PROMPT.format(
                question=question,
                context=context,
                explored_text=explored_text,
            )
            response = self.llm_call(prompt)

            # Parse
            assessment = ""
            gaps = []
            done = False
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("ASSESSMENT:"):
                    assessment = line[11:].strip()
                elif line.startswith("CUE:"):
                    cue = line[4:].strip()
                    if cue:
                        gaps.append(cue)
                elif line.strip().upper() == "DONE":
                    done = True

            reflect_log.append({
                "reflect": reflect_i,
                "assessment": assessment,
                "gaps": gaps,
                "done": done,
            })

            if done or not gaps:
                break

            # Explore exactly 1 gap per round (less dilution)
            for gap in gaps[:1]:
                if len(all_segments) >= self.segment_budget:
                    break
                gap_emb = self.embed_text(gap)
                result = self.store.search(
                    gap_emb, top_k=10, conversation_id=conversation_id,
                    exclude_indices=exclude,
                )
                for seg in result.segments:
                    if seg.index not in exclude:
                        all_segments.append(seg)
                        exclude.add(seg.index)

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": "frontier_v2_iterative",
                "reflect_log": reflect_log,
                "total_segments": len(all_segments),
                "num_reflects": len(reflect_log),
                "cues": [g for entry in reflect_log for g in entry.get("gaps", [])],
            },
        )


# ===========================================================================
# 5. flat_multi_cue — decompose into sub-questions, use as flat cues
# ===========================================================================
class FlatMultiCue(BestshotBase):
    """Primer retrieval -> decompose -> use sub-questions as flat cues."""

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        exclude: set[int] = set()
        all_segments: list[Segment] = []

        # Primer: retrieve top-10 with raw query
        query_emb = self.embed_text(question)
        primer = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        primer_segs = list(primer.segments)
        all_segments.extend(primer_segs)
        for s in primer_segs:
            exclude.add(s.index)

        # Decompose with primer context
        primer_context = _format_segments(primer_segs, max_items=10)
        decompose_prompt = BESTSHOT_DECOMPOSE_PROMPT.format(
            question=question, primer_context=primer_context
        )
        decompose_output = self.llm_call(decompose_prompt)
        sub_questions = _parse_subs(decompose_output)
        if not sub_questions:
            sub_questions = [question]
        sub_questions = sub_questions[:4]

        # Use each sub-question as a flat cue (no tree, no followup)
        for sq in sub_questions:
            sq_emb = self.embed_text(sq)
            result = self.store.search(
                sq_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": "flat_multi_cue",
                "sub_questions": sub_questions,
                "cues": sub_questions,
            },
        )


# ===========================================================================
# 6. retrieve_then_decompose — v15-style first, then decompose on gaps
# ===========================================================================
class RetrieveThenDecompose(BestshotBase):
    """Phase 1a: question retrieval. Phase 1b: v15-style single cue.
    Phase 2: grounded decompose. Phase 3: per-gap retrieval."""

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        exclude: set[int] = set()
        all_segments: list[Segment] = []

        # Phase 1a: initial retrieval
        query_emb = self.embed_text(question)
        result = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        initial_segs = list(result.segments)
        all_segments.extend(initial_segs)
        for s in initial_segs:
            exclude.add(s.index)

        # Phase 1b: v15-style single cue
        context = _format_segments(initial_segs, max_items=10)
        v15_prompt = BESTSHOT_V15_SINGLE_CUE_PROMPT.format(
            question=question, retrieved_context=context
        )
        v15_output = self.llm_call(v15_prompt)
        v15_cues = _parse_cues(v15_output)
        v15_segs: list[Segment] = []
        if v15_cues:
            cue = v15_cues[0]
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    v15_segs.append(seg)
                    all_segments.append(seg)
                    exclude.add(seg.index)

        # Phase 2: grounded decomposition
        found_so_far = initial_segs + v15_segs
        gap_context = _format_segments(found_so_far, max_items=12)
        decompose_prompt = BESTSHOT_GROUNDED_DECOMPOSE_PROMPT.format(
            question=question, retrieved_context=gap_context
        )
        decompose_output = self.llm_call(decompose_prompt)
        sub_questions = _parse_gaps(decompose_output)
        if not sub_questions:
            sub_questions = [question]
        sub_questions = sub_questions[:4]

        # Phase 3: per-gap retrieval
        for sq in sub_questions:
            sq_emb = self.embed_text(sq)
            result = self.store.search(
                sq_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": "retrieve_then_decompose",
                "v15_cue": v15_cues[0] if v15_cues else None,
                "sub_questions": sub_questions,
                "cues": (v15_cues[:1] + sub_questions),
            },
        )


# ===========================================================================
# Architecture registry
# ===========================================================================
def build_architectures(store: SegmentStore) -> dict[str, BestshotBase]:
    """Build all best-shot architecture instances."""
    return {
        "v15_control": V15Control(store),
        "meta_v2f": MetaV2f(store),
        "decompose_then_retrieve": DecomposeThenRetrieve(store),
        "interleaved": Interleaved(store),
        "frontier_v2_iterative": FrontierV2Iterative(store),
        "flat_multi_cue": FlatMultiCue(store),
        "retrieve_then_decompose": RetrieveThenDecompose(store),
    }


# ===========================================================================
# Evaluation
# ===========================================================================
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    arch: BestshotBase,
    question: dict,
    verbose: bool = False,
) -> dict:
    """Evaluate a single architecture on a single question."""
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Deduplicate preserving order
    seen: set[int] = set()
    deduped: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            deduped.append(seg)
            seen.add(seg.index)
    arch_segments = deduped
    total_retrieved = len(arch_segments)

    # Baseline: cosine top-N at same budget
    query_emb = arch.embed_text(q_text)
    max_budget = max(BUDGETS + [total_retrieved])
    baseline_result = arch.store.search(
        query_emb, top_k=max_budget, conversation_id=conv_id
    )

    baseline_recalls: dict[str, float] = {}
    arch_recalls: dict[str, float] = {}
    for budget in BUDGETS:
        baseline_ids = {s.turn_id for s in baseline_result.segments[:budget]}
        baseline_recalls[f"r@{budget}"] = compute_recall(baseline_ids, source_ids)

        arch_ids = {s.turn_id for s in arch_segments[:budget]}
        arch_recalls[f"r@{budget}"] = compute_recall(arch_ids, source_ids)

    # Also at actual retrieval size
    baseline_ids_actual = {
        s.turn_id for s in baseline_result.segments[:total_retrieved]
    }
    arch_ids_actual = {s.turn_id for s in arch_segments}
    baseline_recalls["r@actual"] = compute_recall(baseline_ids_actual, source_ids)
    arch_recalls["r@actual"] = compute_recall(arch_ids_actual, source_ids)

    row = {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question["question_index"],
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "baseline_recalls": baseline_recalls,
        "arch_recalls": arch_recalls,
        "total_retrieved": total_retrieved,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "metadata": result.metadata,
    }

    if verbose:
        print(f"  Source: {sorted(source_ids)} ({len(source_ids)} turns)")
        print(
            f"  Retrieved: {total_retrieved}, Embed: {arch.embed_calls}, "
            f"LLM: {arch.llm_calls}, Time: {elapsed:.1f}s"
        )
        for budget in BUDGETS:
            b = baseline_recalls[f"r@{budget}"]
            a = arch_recalls[f"r@{budget}"]
            delta = a - b
            marker = "W" if delta > 0.001 else ("L" if delta < -0.001 else "T")
            print(
                f"  @{budget:3d}: baseline={b:.3f} arch={a:.3f} "
                f"delta={delta:+.3f} [{marker}]"
            )
        cues = result.metadata.get("cues", [])
        for cue in cues[:4]:
            print(f"    Cue: {cue[:120]}")

    return row


def summarize(results: list[dict], variant_name: str, benchmark: str) -> dict:
    """Compute summary statistics."""
    n = len(results)
    if n == 0:
        return {}

    summary: dict = {"variant": variant_name, "benchmark": benchmark, "n": n}

    for label in [f"r@{b}" for b in BUDGETS] + ["r@actual"]:
        b_vals = [r["baseline_recalls"][label] for r in results]
        a_vals = [r["arch_recalls"][label] for r in results]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n

        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses

        summary[f"baseline_{label}"] = round(b_mean, 4)
        summary[f"arch_{label}"] = round(a_mean, 4)
        summary[f"delta_{label}"] = round(a_mean - b_mean, 4)
        summary[f"W/T/L_{label}"] = f"{wins}/{ties}/{losses}"

    summary["avg_total_retrieved"] = round(
        sum(r["total_retrieved"] for r in results) / n, 1
    )
    summary["avg_embed_calls"] = round(
        sum(r["embed_calls"] for r in results) / n, 1
    )
    summary["avg_llm_calls"] = round(
        sum(r["llm_calls"] for r in results) / n, 1
    )
    summary["avg_time_s"] = round(sum(r["time_s"] for r in results) / n, 2)

    return summary


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    """Per-category breakdown at r@20."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    cat_summaries = {}
    for cat, cat_results in sorted(by_cat.items()):
        n = len(cat_results)
        b_vals = [r["baseline_recalls"]["r@20"] for r in cat_results]
        a_vals = [r["arch_recalls"]["r@20"] for r in cat_results]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        cat_summaries[cat] = {
            "n": n,
            "baseline_r@20": round(b_mean, 4),
            "arch_r@20": round(a_mean, 4),
            "delta_r@20": round(a_mean - b_mean, 4),
            "W/T/L": f"{wins}/{n - wins - losses}/{losses}",
        }
    return cat_summaries


def run_architecture(
    arch_name: str,
    arch: BestshotBase,
    questions: list[dict],
    benchmark_label: str,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Run one architecture, return (results, summary)."""
    print(f"\n{'='*70}")
    print(
        f"ARCHITECTURE: {arch_name} | BENCHMARK: {benchmark_label} | "
        f"{len(questions)} questions"
    )
    print(f"{'='*70}")

    results = []
    for i, question in enumerate(questions):
        q_short = question["question"][:55]
        print(
            f"  [{i+1}/{len(questions)}] {question['category']}: "
            f"{q_short}...",
            flush=True,
        )
        try:
            result = evaluate_one(arch, question, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = summarize(results, arch_name, benchmark_label)

    # Print compact summary
    print(f"\n--- {arch_name} on {benchmark_label} ---")
    for budget in BUDGETS:
        lbl = f"r@{budget}"
        print(
            f"  {lbl}: baseline={summary.get(f'baseline_{lbl}', 0):.3f} "
            f"arch={summary.get(f'arch_{lbl}', 0):.3f} "
            f"delta={summary.get(f'delta_{lbl}', 0):+.3f} "
            f"W/T/L={summary.get(f'W/T/L_{lbl}', '?')}"
        )
    print(
        f"  Avg retrieved: {summary.get('avg_total_retrieved', 0):.0f}, "
        f"Embed: {summary.get('avg_embed_calls', 0):.1f}, "
        f"LLM: {summary.get('avg_llm_calls', 0):.1f}, "
        f"Time: {summary.get('avg_time_s', 0):.1f}s"
    )

    cat_summaries = summarize_by_category(results)
    print(f"\n  Per-category (r@20):")
    for cat, cs in cat_summaries.items():
        print(
            f"    {cat}: delta={cs['delta_r@20']:+.3f} "
            f"W/T/L={cs['W/T/L']} (n={cs['n']})"
        )

    return results, summary


def spot_check_outputs(results: list[dict], arch_name: str, num_checks: int = 5):
    """Spot-check LLM outputs for prompt quality issues."""
    print(f"\n--- Spot-check: {arch_name} ---")
    issues = []

    checked = 0
    for r in results[:num_checks]:
        meta = r.get("metadata", {})
        cues = meta.get("cues", [])
        output = meta.get("output", "")

        # Check reflect_log for frontier
        reflect_log = meta.get("reflect_log", [])
        if reflect_log:
            for entry in reflect_log:
                for gap in entry.get("gaps", []):
                    cues.append(gap)

        for cue in cues:
            cue_lower = cue.lower()
            # Check for meta-instructions
            if any(p in cue_lower for p in [
                "search for", "find the", "look for", "show me",
                "retrieve", "locate the",
            ]):
                issues.append(f"  META-INSTRUCTION: '{cue[:100]}'")
            # Check for boolean queries
            if " OR " in cue or " AND " in cue:
                issues.append(f"  BOOLEAN: '{cue[:100]}'")
            # Check for question-format
            if cue.rstrip().endswith("?"):
                issues.append(f"  QUESTION: '{cue[:100]}'")
            # Check for too-long cues
            if len(cue) > 200:
                issues.append(f"  TOO-LONG ({len(cue)} chars): '{cue[:80]}...'")

        checked += 1

    if issues:
        print(f"  ISSUES FOUND ({len(issues)}):")
        for issue in issues[:10]:
            print(f"    {issue}")
        return False
    else:
        print(f"  All {checked} outputs clean.")
        return True


def print_final_table(all_summaries: dict[str, dict[str, dict]]):
    """Print final comparison table."""
    print("\n" + "=" * 90)
    print("FINAL COMPARISON TABLE")
    print("=" * 90)

    for benchmark in ["locomo_30q", "synthetic_19q"]:
        print(f"\n--- {benchmark.upper()} ---")
        print(
            f"{'Architecture':<28s} {'delta r@20':>10s} {'W/T/L@20':>10s} "
            f"{'delta r@50':>10s} {'LLM calls':>10s} {'Embed calls':>12s} "
            f"{'Retrieved':>10s}"
        )
        print("-" * 90)

        # Collect and sort by delta r@20
        rows = []
        for arch_name, benchmarks in all_summaries.items():
            if benchmark in benchmarks:
                s = benchmarks[benchmark]
                rows.append((arch_name, s))

        rows.sort(key=lambda x: x[1].get("delta_r@20", 0), reverse=True)

        for arch_name, s in rows:
            delta_20 = s.get("delta_r@20", 0)
            wtl_20 = s.get("W/T/L_r@20", "?")
            delta_50 = s.get("delta_r@50", 0)
            llm = s.get("avg_llm_calls", 0)
            embed = s.get("avg_embed_calls", 0)
            retrieved = s.get("avg_total_retrieved", 0)
            print(
                f"  {arch_name:<26s} {delta_20:>+9.3f}  {wtl_20:>10s} "
                f"{delta_50:>+9.3f}  {llm:>9.1f}  {embed:>11.1f}  "
                f"{retrieved:>9.0f}"
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Best-shot comparison of all retrieval architectures"
    )
    parser.add_argument(
        "--arch", type=str, default=None,
        help="Run specific architecture (use --list to see available)",
    )
    parser.add_argument("--all", action="store_true", help="Run all architectures")
    parser.add_argument("--list", action="store_true", help="List all architectures")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--locomo-only", action="store_true",
        help="Skip synthetic benchmark",
    )
    parser.add_argument(
        "--synthetic-only", action="store_true",
        help="Skip LoCoMo benchmark",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing results",
    )
    args = parser.parse_args()

    # Load LoCoMo data
    with open(DATA_DIR / "questions_extended.json") as f:
        all_questions = json.load(f)
    locomo_store = SegmentStore(
        data_dir=DATA_DIR, npz_name="segments_extended.npz"
    )
    locomo_qs = [q for q in all_questions if q.get("benchmark") == "locomo"][:30]
    print(f"LoCoMo: {len(locomo_qs)} questions, {len(locomo_store.segments)} segments")

    # Load synthetic data
    synth_store = None
    synth_qs = []
    synth_path = DATA_DIR / "questions_synthetic.json"
    if synth_path.exists():
        with open(synth_path) as f:
            synth_qs = json.load(f)
        synth_store = SegmentStore(
            data_dir=DATA_DIR, npz_name="segments_synthetic.npz"
        )
        print(f"Synthetic: {len(synth_qs)} questions, {len(synth_store.segments)} segments")
    else:
        print("Synthetic data not found, skipping.")

    # Build architectures (need separate instances per store)
    locomo_archs = build_architectures(locomo_store)

    if args.list:
        print("\nAvailable architectures:")
        for name in locomo_archs:
            print(f"  {name}")
        return

    # Determine which architectures to run
    if args.arch:
        arch_names = [args.arch]
        if args.arch not in locomo_archs:
            print(f"Unknown architecture: {args.arch}")
            print(f"Available: {', '.join(locomo_archs.keys())}")
            return
    elif args.all:
        arch_names = list(locomo_archs.keys())
    else:
        # Default: run all
        arch_names = list(locomo_archs.keys())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all summaries: {arch_name: {benchmark: summary}}
    all_summaries: dict[str, dict[str, dict]] = {}

    # Run on LoCoMo
    if not args.synthetic_only:
        for arch_name in arch_names:
            result_file = RESULTS_DIR / f"bestshot_{arch_name}_locomo_30q.json"
            if result_file.exists() and not args.force:
                print(f"\nSkipping {arch_name} on LoCoMo (exists). Use --force to rerun.")
                with open(result_file) as f:
                    saved = json.load(f)
                results = saved["results"]
                summary = saved["summary"]
            else:
                arch = locomo_archs[arch_name]
                results, summary = run_architecture(
                    arch_name, arch, locomo_qs, "locomo_30q",
                    verbose=args.verbose,
                )

                # Spot-check
                clean = spot_check_outputs(results, arch_name)
                if not clean:
                    print(f"  WARNING: {arch_name} has prompt quality issues on LoCoMo!")

                # Save
                with open(result_file, "w") as f:
                    json.dump(
                        {"results": results, "summary": summary},
                        f, indent=2, default=str,
                    )
                print(f"  Saved: {result_file}")

            if arch_name not in all_summaries:
                all_summaries[arch_name] = {}
            all_summaries[arch_name]["locomo_30q"] = summary

    # Run on synthetic
    if not args.locomo_only and synth_store and synth_qs:
        synth_archs = build_architectures(synth_store)

        for arch_name in arch_names:
            result_file = RESULTS_DIR / f"bestshot_{arch_name}_synthetic_19q.json"
            if result_file.exists() and not args.force:
                print(f"\nSkipping {arch_name} on synthetic (exists). Use --force to rerun.")
                with open(result_file) as f:
                    saved = json.load(f)
                results = saved["results"]
                summary = saved["summary"]
            else:
                arch = synth_archs[arch_name]
                results, summary = run_architecture(
                    arch_name, arch, synth_qs, "synthetic_19q",
                    verbose=args.verbose,
                )

                # Spot-check
                clean = spot_check_outputs(results, arch_name)
                if not clean:
                    print(f"  WARNING: {arch_name} has prompt quality issues on synthetic!")

                # Save
                with open(result_file, "w") as f:
                    json.dump(
                        {"results": results, "summary": summary},
                        f, indent=2, default=str,
                    )
                print(f"  Saved: {result_file}")

            if arch_name not in all_summaries:
                all_summaries[arch_name] = {}
            all_summaries[arch_name]["synthetic_19q"] = summary

    # Final comparison table
    print_final_table(all_summaries)

    # Save all summaries
    summary_file = RESULTS_DIR / "bestshot_all_summaries.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nAll summaries saved: {summary_file}")


if __name__ == "__main__":
    main()
