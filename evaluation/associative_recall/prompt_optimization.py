"""Prompt optimization experiments.

Tests prompt variants on the meta-prompting V2 (strategist-only) and
frontier V2 (iterative re-reflection) architectures. Each variant uses
the same retrieval mechanism but different prompt text.

Usage:
    uv run python prompt_optimization.py --variant <name> [--force] [--verbose]
    uv run python prompt_optimization.py --all
    uv run python prompt_optimization.py --list
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
# Cache classes — optimization-specific, reads from all existing caches
# ---------------------------------------------------------------------------
class OptimEmbeddingCache(EmbeddingCache):
    """Reads all existing caches, writes to optim-specific file."""

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
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "optim_embedding_cache.json"
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


class OptimLLMCache(LLMCache):
    """Reads all existing caches, writes to optim-specific file."""

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
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "optim_llm_cache.json"
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
# Prompt variants — Meta-prompting V2 (strategist-only, 1 LLM call)
# ---------------------------------------------------------------------------

# V15_CONTROL: The exact v15 prompt — used as our control to validate
# the framework matches the reference +33.9pp result.
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

# V2c: EXACT v15 prompt + minimal completeness/conjunction instruction.
# The hypothesis: v15's exact wording is load-bearing. Adding ONE extra
# instruction about completeness awareness may help without disrupting
# the core prompt.
META_V2C_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Consider: Does the question imply MULTIPLE items (e.g., "all allergies", \
"every trip")? If so, keep searching for more. Does it require two topics \
to appear TOGETHER (e.g., "what did X say about Y")? Target that conjunction.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2a: Strategist prompt restructured to MATCH v15's ASSESSMENT + CUE format.
# Takes the strategist's analytical framing (what's covered/missing, completeness
# concerns, conjunction requirements) but outputs in v15 format.
META_V2A_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing?

Think about:
- Are there completeness concerns? (found one item but question implies multiples)
- Are there conjunction requirements? (need content about X AND Y together)
- What specific vocabulary from the retrieved text could anchor new searches?

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2b_v2: v15 preamble + strategist's vocab-matching emphasis + conjunction.
# Combines the proven v15 opening with stronger vocabulary guidance.
META_V2B_V2_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 2 search cues based on your assessment.

CRITICAL — what makes an effective cue:
- Use specific vocabulary that would APPEAR in the target conversation turns
- Short cues (under 100 characters) with dense vocabulary work best
- Write text that someone would actually type in a chat, NOT questions about \
the conversation
- If the question asks for "all" or a list, search for additional items
- If the question links two topics, target their conjunction

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2d: Stripped-down minimal version — what's the simplest prompt that captures
# the meta-prompting benefit?
META_V2D_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

What content is still missing? Generate 2 search cues to find it. \
Use specific vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2e: v15 exact + explicit length constraint (under 100 chars).
# Tests if forcing short cues helps the strategist.
META_V2E_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns. \
Each cue must be under 100 characters — short and vocabulary-dense.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2f: v15 + completeness + explicit "DO NOT" anti-patterns
# Kept for historical reference. The anti-question line was found to hurt
# proactive/task retrieval (pushes the model toward prose fluency over
# keyword density). Use META_V2F_V2_PROMPT for new work.
META_V2F_PROMPT = """\
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


# V2f_v2: V2f without the anti-question line. Proactive cue analysis found
# that the anti-question instruction pushed the model toward polished prose
# (grammatical sentences) and away from dense keyword bundles. Dense keyword
# bundles cover more semantic neighborhoods per cue, which matters for tasks
# with scattered evidence across sub-topics. This variant preserves V2f's
# LoCoMo gains (+5pp over v15) while eliminating the -10.8pp proactive
# regression.
META_V2F_V2_PROMPT = """\
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

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# V2g: V2f + instruction about not using boolean operators in cues.
META_V2G_PROMPT = """\
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
actually appear in a chat message. Do NOT use boolean operators (OR, AND) \
or quotation marks in your cues.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2h: V2f + pivoting emphasis (v15's "pivot to a different topic" is key)
META_V2H_PROMPT = """\
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
vocabulary that would appear in the target conversation turns. \
Make each cue target a DIFFERENT missing aspect — do not generate two \
cues about the same topic.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2i: Even more minimal addition to v15 — just the anti-question instruction.
# Tests whether the completeness hint matters or just the anti-question part.
META_V2I_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2j: Just the completeness hint, no anti-question. Tests completeness alone.
META_V2J_PROMPT = """\
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

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# ---------------------------------------------------------------------------
# Prompt variants — Frontier V2 (iterative re-reflection)
# ---------------------------------------------------------------------------

# Frontier_A: v15-style reflect prompt with explicit length constraints
FRONTIER_A_REFLECT_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({num_segments} segments):
{context}{explored_text}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 1-2 search cues targeting the missing content. Use specific \
vocabulary that would appear in the target conversation turns. \
Each cue must be under 100 characters.

If the retrieval looks complete, respond with DONE.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
(or)
ASSESSMENT: <evaluation>
DONE"""

# Frontier_B: Exact v15 framing, 1 gap only (less dilution)
FRONTIER_B_REFLECT_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}{explored_text}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate exactly 1 search cue targeting the most important missing \
content. Use specific vocabulary that would appear in the target \
conversation turns.

If the retrieval looks complete, respond with DONE.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
(or)
ASSESSMENT: <evaluation>
DONE"""

# Frontier_C: v15-style with vocabulary instruction from v2b
FRONTIER_C_REFLECT_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({num_segments} segments):
{context}{explored_text}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing?

Then generate 1-2 search cues targeting the missing content.

CRITICAL — what makes an effective cue:
- Use specific vocabulary that would APPEAR in the conversation
- Extract real words/phrases from the retrieved excerpts and combine them \
with hypothesized vocabulary for the missing content
- Short cues (under 100 characters) with dense vocabulary work best
- Do NOT write questions or meta-instructions

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
class OptimBase:
    """Base class with embedding/LLM utilities and counters."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = OptimEmbeddingCache()
        self.llm_cache = OptimLLMCache()
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
class OptimResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


# ===========================================================================
# Meta-prompting V2 variants (1 LLM call, same cost as v15)
# ===========================================================================
class MetaV2Variant(OptimBase):
    """Generic single-call strategist variant. Takes a prompt template.

    Matches v15_control retrieval logic exactly: no neighbor expansion.
    """

    def __init__(self, store: SegmentStore, prompt_template: str,
                 client: OpenAI | None = None):
        super().__init__(store, client)
        self.prompt_template = prompt_template

    def retrieve(self, question: str, conversation_id: str) -> OptimResult:
        # Hop 0: embed question, retrieve top-10
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        # Build context section matching v15 format exactly
        # v15_control uses _format_segments then puts it after
        # "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
        context = _format_segments(all_segments)
        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + context
        )

        # Single LLM call
        prompt = self.prompt_template.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        if not cues:
            return OptimResult(
                segments=all_segments,
                metadata={"output": output, "cues": []},
            )

        # Retrieve with cues — matching v15_control exactly (no neighbors)
        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    all_segments.append(seg)
                    exclude_indices.add(seg.index)

        return OptimResult(
            segments=all_segments,
            metadata={"output": output, "cues": cues[:2]},
        )


# ===========================================================================
# Frontier V2 variants (iterative re-reflection)
# ===========================================================================
class FrontierVariant(OptimBase):
    """Iterative frontier with configurable reflect prompt."""

    def __init__(self, store: SegmentStore, reflect_prompt: str,
                 client: OpenAI | None = None,
                 max_reflects: int = 4, gaps_per_reflect: int = 2,
                 segment_budget: int = 80):
        super().__init__(store, client)
        self.reflect_prompt = reflect_prompt
        self.max_reflects = max_reflects
        self.gaps_per_reflect = gaps_per_reflect
        self.segment_budget = segment_budget

    def retrieve(self, question: str, conversation_id: str) -> OptimResult:
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

            prompt = self.reflect_prompt.format(
                question=question,
                num_segments=len(all_segments),
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

            # Explore gaps
            for gap in gaps[:self.gaps_per_reflect]:
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

        return OptimResult(
            segments=all_segments,
            metadata={
                "reflect_log": reflect_log,
                "total_segments": len(all_segments),
                "num_reflects": len(reflect_log),
                "cues": [g for entry in reflect_log for g in entry.get("gaps", [])],
            },
        )


# ===========================================================================
# Variant registry
# ===========================================================================
def build_variants(store: SegmentStore) -> dict[str, OptimBase]:
    """Build all prompt variant instances."""
    return {
        # Control: exact v15 prompt with our framework (validates framework)
        "v15_control": MetaV2Variant(store, V15_CONTROL_PROMPT),
        # Meta-prompting V2 variants (1 LLM call, v15_control retrieval)
        "meta_v2c_completeness": MetaV2Variant(store, META_V2C_PROMPT),
        "meta_v2a_strategist_format": MetaV2Variant(store, META_V2A_PROMPT),
        "meta_v2b_v2_vocab": MetaV2Variant(store, META_V2B_V2_PROMPT),
        "meta_v2d_minimal": MetaV2Variant(store, META_V2D_PROMPT),
        "meta_v2e_length": MetaV2Variant(store, META_V2E_PROMPT),
        "meta_v2f_antipattern": MetaV2Variant(store, META_V2F_PROMPT),
        "meta_v2g_no_boolean": MetaV2Variant(store, META_V2G_PROMPT),
        "meta_v2h_diverse_cues": MetaV2Variant(store, META_V2H_PROMPT),
        "meta_v2i_anti_question_only": MetaV2Variant(store, META_V2I_PROMPT),
        "meta_v2j_completeness_only": MetaV2Variant(store, META_V2J_PROMPT),
        # Frontier variants (iterative, 2+ LLM calls)
        "frontier_a_v15_style_reflect": FrontierVariant(
            store, FRONTIER_A_REFLECT_PROMPT),
        "frontier_b_single_gap": FrontierVariant(
            store, FRONTIER_B_REFLECT_PROMPT, gaps_per_reflect=1),
        "frontier_c_vocab_emphasis": FrontierVariant(
            store, FRONTIER_C_REFLECT_PROMPT),
    }


# ===========================================================================
# Evaluation
# ===========================================================================
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    arch: OptimBase,
    question: dict,
    verbose: bool = False,
) -> dict:
    """Evaluate a single variant on a single question."""
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


def run_variant(
    variant_name: str,
    arch: OptimBase,
    questions: list[dict],
    benchmark_label: str,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Run one variant, return (results, summary)."""
    print(f"\n{'='*70}")
    print(
        f"VARIANT: {variant_name} | BENCHMARK: {benchmark_label} | "
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
    summary = summarize(results, variant_name, benchmark_label)

    # Print compact summary
    print(f"\n--- {variant_name} on {benchmark_label} ---")
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prompt optimization experiments"
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Run specific variant (use --list to see available)",
    )
    parser.add_argument("--all", action="store_true", help="Run all variants")
    parser.add_argument("--list", action="store_true", help="List all variants")
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing results"
    )
    args = parser.parse_args()

    # Load data
    with open(DATA_DIR / "questions_extended.json") as f:
        all_questions = json.load(f)

    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_extended.npz")
    print(f"Loaded {len(store.segments)} segments")

    locomo_qs = [q for q in all_questions if q.get("benchmark") == "locomo"][
        : args.num_questions
    ]
    print(f"LoCoMo: {len(locomo_qs)} questions")

    variants = build_variants(store)

    if args.list:
        print("\nAvailable variants:")
        for name in variants:
            print(f"  {name}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which variants to run
    if args.variant:
        variant_names = [args.variant]
    elif args.all:
        variant_names = list(variants.keys())
    else:
        # Default: run highest-priority variants first
        variant_names = [
            "meta_v2c_completeness",
            "meta_v2e_length",
            "meta_v2a_strategist_format",
        ]

    all_summaries = []

    for variant_name in variant_names:
        if variant_name not in variants:
            print(f"Unknown variant: {variant_name}")
            continue

        results_file = RESULTS_DIR / f"optim_{variant_name}_locomo_{args.num_questions}q.json"

        if results_file.exists() and not args.force:
            print(f"\nSkipping {variant_name} (exists, use --force to overwrite)")
            with open(results_file) as f:
                existing = json.load(f)
            summary = summarize(
                existing, variant_name, f"locomo_{args.num_questions}q"
            )
            all_summaries.append(summary)
            print(
                f"  r@20: delta={summary.get('delta_r@20', 0):+.3f} "
                f"W/T/L={summary.get('W/T/L_r@20', '?')}"
            )
            continue

        arch = variants[variant_name]
        results, summary = run_variant(
            variant_name,
            arch,
            locomo_qs,
            f"locomo_{args.num_questions}q",
            verbose=args.verbose,
        )
        all_summaries.append(summary)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved to {results_file}")

    # Save combined summaries
    summary_file = RESULTS_DIR / "optim_all_summaries.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Grand summary table
    print(f"\n{'='*120}")
    print("PROMPT OPTIMIZATION — Summary Table")
    print(f"{'='*120}")
    print(
        f"{'Variant':<40s} {'B-r@20':>8s} {'A-r@20':>8s} "
        f"{'Delta':>8s} {'W/T/L':>10s} {'#Ret':>6s} {'Emb':>5s} "
        f"{'LLM':>5s} {'Time':>6s}"
    )
    print("-" * 120)
    for s in all_summaries:
        if not s:
            continue
        print(
            f"{s['variant']:<40s} "
            f"{s.get('baseline_r@20', 0):>8.3f} "
            f"{s.get('arch_r@20', 0):>8.3f} "
            f"{s.get('delta_r@20', 0):>+8.3f} "
            f"{s.get('W/T/L_r@20', '?'):>10s} "
            f"{s.get('avg_total_retrieved', 0):>6.0f} "
            f"{s.get('avg_embed_calls', 0):>5.1f} "
            f"{s.get('avg_llm_calls', 0):>5.0f} "
            f"{s.get('avg_time_s', 0):>6.1f}"
        )
    print("-" * 120)
    print(
        "References:  v15 baseline = +0.339 (13W/17T/0L)  |  "
        "meta_v2 = +0.256 (10W/20T/0L)  |  "
        "meta_v2b = +0.261 (10W/19T/1L)"
    )
    print(
        "             frontier_v2 = +0.172 (8W/20T/2L)"
    )


if __name__ == "__main__":
    main()
