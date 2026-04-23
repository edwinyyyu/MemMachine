"""Meta-prompting approach to cue generation for conversation memory retrieval.

Uses a TWO-CALL pattern:
  Call 1 (strategist): Sees question + retrieved segments, produces dynamic
                       instructions for the searcher.
  Call 2 (searcher):   Receives strategist's instructions + segments, generates cues.

Variants:
  V1: Strategist + Searcher (basic two-call)
  V2: Strategist only (single call with meta-knowledge)
  V3: Strategist + Searcher with domain knowledge injection
  V4: Iterative strategist (2 rounds of strategic reflection)
  V5: v15 + strategist post-assessment (additive on top of proven v15)
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

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL = "gpt-5-mini"
BUDGETS = [20, 50, 100]


# ---------------------------------------------------------------------------
# Cache classes — isolated from other experiments
# ---------------------------------------------------------------------------
class MetaEmbeddingCache(EmbeddingCache):
    """Reads existing embedding caches, writes to meta-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in (
            "embedding_cache.json",
            "arch_embedding_cache.json",
            "agent_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        # Also try reading our own cache
        self.cache_file = self.cache_dir / "meta_embedding_cache.json"
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache.update(json.load(f))
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


class MetaLLMCache(LLMCache):
    """Reads existing LLM caches, writes to meta-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in (
            "llm_cache.json",
            "arch_llm_cache.json",
            "agent_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "meta_llm_cache.json"
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache.update(json.load(f))
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
# Prompt templates
# ---------------------------------------------------------------------------

STRATEGIST_PROMPT = """\
You are advising a memory search system. Given a question and what has been \
retrieved so far from a conversation, provide specific guidance for the next \
search.

QUESTION: {question}

RETRIEVED SO FAR:
{segments}

Your job:
1. What aspects of the question are well-covered by retrieved content?
2. What aspects are MISSING or incomplete?
3. Are there completeness concerns? (e.g., "found one allergy, might be more")
4. Are there conjunction requirements? (e.g., "need content about X AND Y together")
5. What specific vocabulary or topics should the next search target?
6. Any pitfalls to avoid? (e.g., "don't search too broadly")

Respond with ONLY your instructions for the searcher. Be specific and concise.

INSTRUCTIONS FOR SEARCHER:
"""

SEARCHER_PROMPT = """\
You are generating search text for semantic retrieval over a conversation. \
Your cues will be embedded and compared via cosine similarity.

QUESTION: {question}

RETRIEVED SO FAR:
{segments}

SEARCH INSTRUCTIONS (from strategist):
{strategist_instructions}

Generate exactly 2 search cues. Each cue should be 1-2 SHORT sentences \
(under 100 characters if possible) using specific vocabulary that would \
APPEAR in the conversation. Do NOT write meta-instructions. Write text \
that would actually appear in a chat message.

CUE: <text>
CUE: <text>
Nothing else."""

STRATEGIST_V2_PROMPT = """\
You are a memory search strategist generating search cues for semantic \
retrieval over a conversation history. Your cues will be embedded and \
compared via cosine similarity.

QUESTION: {question}

RETRIEVED SO FAR:
{segments}

STRATEGIC ANALYSIS:
1. What aspects of the question are well-covered by retrieved content?
2. What aspects are MISSING or incomplete?
3. Are there completeness concerns? (lists, multiples, updates)
4. Are there conjunction requirements? (X AND Y together)

Based on your analysis, generate exactly 2 search cues.

RULES for effective cues:
- Each cue should be 1-2 SHORT sentences (under 100 characters)
- Use specific vocabulary that would APPEAR in the conversation
- Do NOT write meta-instructions ("find the message about...")
- Write text that would actually appear in a chat message
- Target DIFFERENT missing aspects with each cue

Format:
ASSESSMENT: <1-2 sentence strategic analysis>
CUE: <text>
CUE: <text>
Nothing else."""

STRATEGIST_V3_PROMPT = """\
You are advising a memory search system. Given a question and what has been \
retrieved so far from a conversation, provide specific guidance for the next \
search.

QUESTION: {question}

RETRIEVED SO FAR:
{segments}

Apply your WORLD KNOWLEDGE to guide the search:
- If the question involves medical topics, remember that patients often have \
multiple conditions, medications, and allergies. Keep searching for more.
- If it involves preferences (food, music, etc.), check if preferences were \
updated or changed over time.
- If it involves a specific person + topic, you need content mentioning BOTH together.
- If it involves events, look for before/during/after mentions.
- If it asks about "all" or "every", assume there are more items than found so far.
- If it involves dates or timing, look for explicit temporal markers.

Your job:
1. What aspects are well-covered? What's MISSING?
2. Apply domain knowledge: what additional information likely exists?
3. What specific vocabulary or topics should the next search target?
4. What pitfalls should be avoided?

Respond with ONLY your instructions for the searcher. Be specific and concise.

INSTRUCTIONS FOR SEARCHER:
"""

STRATEGIST_V4_ROUND2_PROMPT = """\
You are advising a memory search system on its SECOND round of refinement. \
The first round already expanded the search. Now assess the full picture.

QUESTION: {question}

ALL RETRIEVED CONTENT (from initial search + first expansion):
{segments}

What is STILL missing after two rounds of search? Be very specific about:
1. Exact information gaps that remain
2. Whether we have found ENOUGH items (for list/enumeration questions)
3. Whether we have the right time periods covered
4. Whether conjunction requirements are met (X AND Y together)

Respond with ONLY your instructions for a final targeted search. Be very specific.

INSTRUCTIONS FOR FINAL SEARCH:
"""

V15_PROMPT = """\
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

STRATEGIST_V2B_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{segments}

First, strategically assess:
1. What aspects of the question are well-covered by retrieved content?
2. What aspects are MISSING or incomplete?
3. Are there completeness concerns? (found one item but question implies multiples)
4. Are there conjunction requirements? (need X AND Y together)
5. What specific VOCABULARY from the retrieved text could anchor new searches?

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

CRITICAL — what makes an effective cue:
- A cue that SHARES VOCABULARY with the target turn. If the conversation \
mentions "debounce delay 300ms", a cue mentioning "debounce" and "300ms" \
will score high.
- Write text that would actually APPEAR in the conversation — things a user \
or assistant would type. NOT questions about the conversation.
- Extract real words/phrases from the retrieved excerpts and combine them \
with hypothesized vocabulary for the missing content.
- Short cues (under 100 characters) with dense vocabulary work best.

WHAT NOT TO DO:
- Do NOT write questions like "Did you ever mention X?"
- Do NOT write meta-instructions ("find the message about...")
- Do NOT rephrase the original question

Format:
ASSESSMENT: <1-2 sentence strategic analysis of what's missing>
CUE: <text with specific vocabulary>
CUE: <text with specific vocabulary>
Nothing else."""

V5_GAP_FILL_PROMPT = """\
You are a search strategist reviewing retrieval results. A search system \
has already found some relevant conversation content. Your job: identify \
REMAINING GAPS and generate 1-2 targeted cues to fill them.

QUESTION: {question}

ALL RETRIEVED CONTENT SO FAR:
{segments}

STRATEGIC ASSESSMENT:
1. What aspects of the question are well-covered?
2. What specific information is STILL MISSING?
3. Are there completeness concerns? (e.g., found 2 items but question asks for "all")
4. Are conjunction requirements met? (e.g., need X AND Y together)

Based on your assessment, generate 1-2 gap-filling search cues. Each cue \
should be 1-2 SHORT sentences (under 100 characters) targeting SPECIFIC \
missing content. Use vocabulary that would appear in the conversation.

If everything seems well-covered, generate cues targeting less obvious \
aspects or confirming completeness.

Format:
ASSESSMENT: <what's covered, what's missing>
CUE: <text>
CUE: <text>
Nothing else."""


# ---------------------------------------------------------------------------
# Helper: format segments for display in prompts
# ---------------------------------------------------------------------------
def format_segments(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
    """Format segments chronologically for LLM context."""
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def format_context_section_v15(
    all_segments: list[Segment],
    new_segments: list[Segment] | None = None,
    hop_number: int = 1,
    previous_cues: list[str] | None = None,
) -> str:
    """Build context section matching v15's format from the engine."""
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


def parse_cues(text: str) -> list[str]:
    """Parse CUE: lines from LLM output."""
    cues = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class MetaResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class MetaBase:
    """Base class with embedding/LLM utilities and counters."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = MetaEmbeddingCache()
        self.llm_cache = MetaLLMCache()
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

    def search_with_cues(
        self,
        cues: list[str],
        conversation_id: str,
        exclude_indices: set[int],
        top_k: int = 10,
        neighbor_radius: int = 1,
    ) -> list[Segment]:
        """Embed cues, retrieve segments, expand neighbors, return new segments."""
        new_segments: list[Segment] = []
        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=top_k,
                conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    new_segments.append(seg)
                    exclude_indices.add(seg.index)

        # Neighbor expansion
        if neighbor_radius > 0:
            neighbor_segs: list[Segment] = []
            for seg in new_segments:
                neighbors = self.store.get_neighbors(
                    seg, radius=neighbor_radius, exclude_indices=exclude_indices
                )
                for n in neighbors:
                    if n.index not in exclude_indices:
                        neighbor_segs.append(n)
                        exclude_indices.add(n.index)
            new_segments.extend(neighbor_segs)

        return new_segments

    def retrieve(self, question: str, conversation_id: str) -> MetaResult:
        raise NotImplementedError


# ===========================================================================
# V1: Strategist + Searcher (basic two-call)
# ===========================================================================
class MetaV1StrategistSearcher(MetaBase):
    """Two-call pattern: strategist produces instructions, searcher generates cues.

    Flow: embed question -> top-10 -> strategist -> searcher -> embed cues -> retrieve
    LLM calls: 2 (1 strategist + 1 searcher)
    Embed calls: 3 (question + 2 cues)
    """

    def retrieve(self, question: str, conversation_id: str) -> MetaResult:
        # Hop 0: embed question, retrieve top-10
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)

        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        # Call 1: Strategist
        seg_text = format_segments(all_segments)
        strategist_prompt = STRATEGIST_PROMPT.format(
            question=question, segments=seg_text
        )
        strategist_output = self.llm_call(strategist_prompt)

        # Call 2: Searcher
        searcher_prompt = SEARCHER_PROMPT.format(
            question=question,
            segments=seg_text,
            strategist_instructions=strategist_output,
        )
        searcher_output = self.llm_call(searcher_prompt)
        cues = parse_cues(searcher_output)

        if not cues:
            return MetaResult(
                segments=all_segments,
                metadata={
                    "strategist_output": strategist_output,
                    "searcher_output": searcher_output,
                    "cues": [],
                },
            )

        # Retrieve with cues
        new_segments = self.search_with_cues(
            cues, conversation_id, exclude_indices
        )
        all_segments.extend(new_segments)

        return MetaResult(
            segments=all_segments,
            metadata={
                "strategist_output": strategist_output,
                "searcher_output": searcher_output,
                "cues": cues,
            },
        )


# ===========================================================================
# V2: Strategist only (single call with meta-knowledge framing)
# ===========================================================================
class MetaV2StrategistOnly(MetaBase):
    """Single call that combines strategic analysis with cue generation.

    Flow: embed question -> top-10 -> strategist (produces cues directly) -> retrieve
    LLM calls: 1 (identical cost to v15)
    Embed calls: 3 (question + 2 cues)
    """

    def retrieve(self, question: str, conversation_id: str) -> MetaResult:
        # Hop 0
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)

        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        # Single strategist call that also produces cues
        seg_text = format_segments(all_segments)
        prompt = STRATEGIST_V2_PROMPT.format(
            question=question, segments=seg_text
        )
        output = self.llm_call(prompt)
        cues = parse_cues(output)

        if not cues:
            return MetaResult(
                segments=all_segments,
                metadata={"output": output, "cues": []},
            )

        new_segments = self.search_with_cues(
            cues, conversation_id, exclude_indices
        )
        all_segments.extend(new_segments)

        return MetaResult(
            segments=all_segments,
            metadata={"output": output, "cues": cues},
        )


# ===========================================================================
# V2b: Improved strategist only (better vocabulary matching emphasis)
# ===========================================================================
class MetaV2bImprovedStrategist(MetaBase):
    """V2 with stronger emphasis on vocabulary matching (the v15 insight).

    Same structure as V2 but prompt explicitly instructs:
    - Extract vocabulary FROM retrieved text
    - Write cues as conversation content, NOT questions
    - Short dense cues

    LLM calls: 1
    Embed calls: 3
    """

    def retrieve(self, question: str, conversation_id: str) -> MetaResult:
        # Hop 0
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)

        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        seg_text = format_segments(all_segments)
        prompt = STRATEGIST_V2B_PROMPT.format(
            question=question, segments=seg_text
        )
        output = self.llm_call(prompt)
        cues = parse_cues(output)

        if not cues:
            return MetaResult(
                segments=all_segments,
                metadata={"output": output, "cues": []},
            )

        new_segments = self.search_with_cues(
            cues, conversation_id, exclude_indices
        )
        all_segments.extend(new_segments)

        return MetaResult(
            segments=all_segments,
            metadata={"output": output, "cues": cues},
        )


# ===========================================================================
# V3: Strategist + Searcher with domain knowledge injection
# ===========================================================================
class MetaV3DomainKnowledge(MetaBase):
    """Two-call with domain knowledge in strategist prompt.

    Strategist is explicitly prompted to apply world knowledge about
    medical topics, preferences, temporal patterns, etc.

    LLM calls: 2
    Embed calls: 3
    """

    def retrieve(self, question: str, conversation_id: str) -> MetaResult:
        # Hop 0
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)

        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        # Call 1: Strategist with domain knowledge
        seg_text = format_segments(all_segments)
        strategist_prompt = STRATEGIST_V3_PROMPT.format(
            question=question, segments=seg_text
        )
        strategist_output = self.llm_call(strategist_prompt)

        # Call 2: Searcher
        searcher_prompt = SEARCHER_PROMPT.format(
            question=question,
            segments=seg_text,
            strategist_instructions=strategist_output,
        )
        searcher_output = self.llm_call(searcher_prompt)
        cues = parse_cues(searcher_output)

        if not cues:
            return MetaResult(
                segments=all_segments,
                metadata={
                    "strategist_output": strategist_output,
                    "searcher_output": searcher_output,
                    "cues": [],
                },
            )

        new_segments = self.search_with_cues(
            cues, conversation_id, exclude_indices
        )
        all_segments.extend(new_segments)

        return MetaResult(
            segments=all_segments,
            metadata={
                "strategist_output": strategist_output,
                "searcher_output": searcher_output,
                "cues": cues,
            },
        )


# ===========================================================================
# V4: Iterative strategist (2 rounds)
# ===========================================================================
class MetaV4IterativeStrategist(MetaBase):
    """Two rounds of strategist+searcher for deeper reflection.

    Round 1: retrieve -> strategist -> searcher -> retrieve
    Round 2: strategist sees ALL -> searcher -> retrieve

    LLM calls: 4 (2 strategist + 2 searcher)
    Embed calls: 5 (question + 4 cues)
    """

    def retrieve(self, question: str, conversation_id: str) -> MetaResult:
        # Hop 0
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)

        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        # Round 1: Strategist + Searcher
        seg_text = format_segments(all_segments)
        strat1_prompt = STRATEGIST_PROMPT.format(
            question=question, segments=seg_text
        )
        strat1_output = self.llm_call(strat1_prompt)

        search1_prompt = SEARCHER_PROMPT.format(
            question=question,
            segments=seg_text,
            strategist_instructions=strat1_output,
        )
        search1_output = self.llm_call(search1_prompt)
        cues1 = parse_cues(search1_output)

        if cues1:
            new_segs = self.search_with_cues(
                cues1, conversation_id, exclude_indices
            )
            all_segments.extend(new_segs)

        # Round 2: Strategist sees ALL content, generates new instructions
        seg_text2 = format_segments(all_segments, max_items=16)
        strat2_prompt = STRATEGIST_V4_ROUND2_PROMPT.format(
            question=question, segments=seg_text2
        )
        strat2_output = self.llm_call(strat2_prompt)

        search2_prompt = SEARCHER_PROMPT.format(
            question=question,
            segments=seg_text2,
            strategist_instructions=strat2_output,
        )
        search2_output = self.llm_call(search2_prompt)
        cues2 = parse_cues(search2_output)

        if cues2:
            new_segs2 = self.search_with_cues(
                cues2, conversation_id, exclude_indices
            )
            all_segments.extend(new_segs2)

        return MetaResult(
            segments=all_segments,
            metadata={
                "round1_strategist": strat1_output,
                "round1_cues": cues1,
                "round2_strategist": strat2_output,
                "round2_cues": cues2,
            },
        )


# ===========================================================================
# V5: v15 + strategist post-assessment (additive)
# ===========================================================================
class MetaV5AdditiveTov15(MetaBase):
    """Run v15's proven approach first, then add strategist gap-filling.

    Phase 1: Exactly v15 (embed question -> top-10 -> v15 prompt -> 2 cues -> retrieve)
    Phase 2: Strategist sees ALL -> generates 1-2 gap-filling cues -> retrieve

    LLM calls: 2 (1 v15 + 1 strategist)
    Embed calls: 5 (question + 2 v15 cues + 2 gap cues)
    """

    def retrieve(self, question: str, conversation_id: str) -> MetaResult:
        # Hop 0
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)

        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        # Phase 1: v15 cue generation
        context_section = format_context_section_v15(all_segments)
        v15_prompt = V15_PROMPT.format(
            question=question, context_section=context_section
        )
        v15_output = self.llm_call(v15_prompt)
        v15_cues = parse_cues(v15_output)

        if v15_cues:
            v15_segments = self.search_with_cues(
                v15_cues, conversation_id, exclude_indices
            )
            all_segments.extend(v15_segments)

        # Phase 2: Strategist gap-filling
        all_seg_text = format_segments(all_segments, max_items=16)
        gap_prompt = V5_GAP_FILL_PROMPT.format(
            question=question, segments=all_seg_text
        )
        gap_output = self.llm_call(gap_prompt)
        gap_cues = parse_cues(gap_output)

        if gap_cues:
            gap_segments = self.search_with_cues(
                gap_cues, conversation_id, exclude_indices
            )
            all_segments.extend(gap_segments)

        return MetaResult(
            segments=all_segments,
            metadata={
                "v15_output": v15_output,
                "v15_cues": v15_cues,
                "gap_output": gap_output,
                "gap_cues": gap_cues,
            },
        )


# ===========================================================================
# Variant registry
# ===========================================================================
META_VARIANTS: dict[str, type[MetaBase]] = {
    "v1_strategist_searcher": MetaV1StrategistSearcher,
    "v2_strategist_only": MetaV2StrategistOnly,
    "v2b_improved_strategist": MetaV2bImprovedStrategist,
    "v3_domain_knowledge": MetaV3DomainKnowledge,
    "v4_iterative": MetaV4IterativeStrategist,
    "v5_additive_v15": MetaV5AdditiveTov15,
}


# ===========================================================================
# Evaluation
# ===========================================================================
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    arch: MetaBase,
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
        cues = result.metadata.get("cues") or result.metadata.get("v15_cues", [])
        for cue in cues[:4]:
            print(f"    Cue: {cue[:100]}")

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
    arch: MetaBase,
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
        description="Meta-prompting retrieval experiments"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=list(META_VARIANTS.keys()),
        help="Run specific variant (default: run according to process)",
    )
    parser.add_argument("--all", action="store_true", help="Run all variants")
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing results"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze cached LLM outputs for a variant",
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which variants to run
    if args.variant:
        variant_names = [args.variant]
    elif args.all:
        variant_names = list(META_VARIANTS.keys())
    else:
        # Default process: V1 and V2 first, then decide
        variant_names = ["v1_strategist_searcher", "v2_strategist_only"]

    all_summaries = []

    for variant_name in variant_names:
        if variant_name not in META_VARIANTS:
            print(f"Unknown variant: {variant_name}")
            continue

        results_file = RESULTS_DIR / f"meta_{variant_name}_locomo_{args.num_questions}q.json"

        if results_file.exists() and not args.force:
            print(f"\nSkipping {variant_name} (exists, use --force to overwrite)")
            with open(results_file) as f:
                existing = json.load(f)
            summary = summarize(existing, variant_name, f"locomo_{args.num_questions}q")
            all_summaries.append(summary)
            print(
                f"  r@20: delta={summary.get('delta_r@20', 0):+.3f} "
                f"W/T/L={summary.get('W/T/L_r@20', '?')}"
            )
            continue

        variant_cls = META_VARIANTS[variant_name]
        arch = variant_cls(store)
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
    summary_file = RESULTS_DIR / "meta_all_summaries.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Grand summary table
    print(f"\n{'='*110}")
    print("GRAND SUMMARY TABLE — Meta-Prompting Variants")
    print(f"{'='*110}")
    print(
        f"{'Variant':<30s} {'Bench':>12s} {'B-r@20':>8s} {'A-r@20':>8s} "
        f"{'Delta':>8s} {'W/T/L':>10s} {'#Ret':>6s} {'Emb':>5s} "
        f"{'LLM':>5s} {'Time':>6s}"
    )
    print("-" * 110)
    for s in all_summaries:
        if not s:
            continue
        print(
            f"{s['variant']:<30s} {s['benchmark']:>12s} "
            f"{s.get('baseline_r@20', 0):>8.3f} "
            f"{s.get('arch_r@20', 0):>8.3f} "
            f"{s.get('delta_r@20', 0):>+8.3f} "
            f"{s.get('W/T/L_r@20', '?'):>10s} "
            f"{s.get('avg_total_retrieved', 0):>6.0f} "
            f"{s.get('avg_embed_calls', 0):>5.1f} "
            f"{s.get('avg_llm_calls', 0):>5.0f} "
            f"{s.get('avg_time_s', 0):>6.1f}"
        )
    print("-" * 110)
    print(
        "Reference v15 (1 hop, 2 cues, nr=1): LoCoMo +0.339 (13W/17T/0L)"
    )


if __name__ == "__main__":
    main()
