"""Memory-index experiment for associative recall.

Concept
-------
Humans have a partial meta-model of their own memory: "I have memories about
my childhood, work projects, relationships, hobbies." This meta-model is
activated by a query and guides retrieval.

For a long conversation, we can pre-compute this meta-model as a "memory
index" — a structured summary describing topics, entities, temporal markers,
and key decisions that appear in the conversation. We pass that index as
conversational metadata to the cue generator BEFORE it generates cues.

Step 1: build (and cache) one memory index per conversation.
Step 2: run several cue-generation variants that use the index differently.
Step 3: fair-backfill evaluation at K=20, K=50 on 4 benchmarks.

Variants
--------
1. v15_with_index       - v15 prompt + memory index in context
2. v2f_v2_with_index    - v2f_v2 prompt + memory index in context
3. index_only           - generate cues from index alone (no segment context)
4. v2f_without_index    - control (equivalent to meta_v2f/V2F_V2)

Usage
-----
    uv run python memory_index.py                # run everything
    uv run python memory_index.py --indices-only # only (re)build indices
    uv run python memory_index.py --qualitative  # print a few indices/cues
    uv run python memory_index.py --variant v15_with_index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    Segment,
    SegmentStore,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50]

MEMINDEX_LLM_CACHE = CACHE_DIR / "memory_index_llm_cache.json"
MEMINDEX_EMBED_CACHE = CACHE_DIR / "memory_index_embedding_cache.json"
MEMINDEX_INDEX_FILE = CACHE_DIR / "memory_indices.json"

DATASETS = {
    "locomo_30q": {
        "npz": "segments_extended.npz",
        "questions": "questions_extended.json",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "max_questions": 30,
    },
    "synthetic_19q": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "filter": None,
        "max_questions": None,
    },
    "puzzle_16q": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "filter": None,
        "max_questions": None,
    },
    "advanced_23q": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "filter": None,
        "max_questions": None,
    },
}


# ---------------------------------------------------------------------------
# Cache classes -- namespaced to this experiment so we don't pollute others.
# ---------------------------------------------------------------------------
class _JSONCache:
    """Small JSON-backed dict with SHA256 keys and atomic write.

    Thread-safe: a single lock guards the dict and file writes.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, object] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with open(self.path) as f:
                self._cache = json.load(f)

    def get(self, key: str):
        with self._lock:
            return self._cache.get(key)

    def put(self, key: str, value) -> None:
        with self._lock:
            self._cache[key] = value

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(self._cache, f)
            tmp.replace(self.path)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Memory-index building
# ---------------------------------------------------------------------------
MEMINDEX_PROMPT = """\
You are building a META-MODEL of a long conversation's CONTENTS so a
retrieval system can later decide where to look. Think of it as the index at
the back of a book: a compact, faithful map of what is actually in the
conversation. DO NOT add information not present.

Output ONLY the four sections below, in this exact order, with each item on
its own line prefixed as shown. Keep items short (3-12 words), concrete, and
use the conversation's own vocabulary (names, places, jargon). Cover the
WHOLE conversation, not just the beginning.

TOPICS (5-15 items): high-level themes / sub-topics actually discussed.
ENTITIES (5-25 items): people, places, organizations, products, tools, and
   other named things mentioned. Include aliases and name variants on one
   line separated by "/" if both appear (e.g. "Bob / Robert").
TEMPORAL_MARKERS (3-15 items): dates, days/weeks/months/years, events
   anchored in time ("last Tuesday", "in October", "before the move").
DECISIONS (0-15 items): explicit decisions, commitments, plans, or chosen
   options that were actually agreed upon in the conversation.

Conversation ({n_turns} turns):
{conversation_text}

Format (exactly):
TOPICS:
- <topic>
- <topic>
ENTITIES:
- <entity>
- <entity>
TEMPORAL_MARKERS:
- <marker>
- <marker>
DECISIONS:
- <decision>
- <decision>
"""


@dataclass
class MemoryIndex:
    conversation_id: str
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    temporal: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    raw: str = ""

    def format_for_prompt(self) -> str:
        def _bul(xs: list[str]) -> str:
            return "\n".join(f"  - {x}" for x in xs) if xs else "  (none)"

        return (
            "CONVERSATION INDEX (what's actually in memory):\n"
            f"TOPICS:\n{_bul(self.topics)}\n"
            f"ENTITIES:\n{_bul(self.entities)}\n"
            f"TEMPORAL_MARKERS:\n{_bul(self.temporal)}\n"
            f"DECISIONS:\n{_bul(self.decisions)}"
        )

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "topics": self.topics,
            "entities": self.entities,
            "temporal": self.temporal,
            "decisions": self.decisions,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryIndex":
        return cls(
            conversation_id=d["conversation_id"],
            topics=d.get("topics", []),
            entities=d.get("entities", []),
            temporal=d.get("temporal", []),
            decisions=d.get("decisions", []),
            raw=d.get("raw", ""),
        )


def _parse_memindex(raw: str, conv_id: str) -> MemoryIndex:
    topics: list[str] = []
    entities: list[str] = []
    temporal: list[str] = []
    decisions: list[str] = []

    current: list[str] | None = None
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        head = s.upper().rstrip(":")
        if head == "TOPICS":
            current = topics
            continue
        if head == "ENTITIES":
            current = entities
            continue
        if head in ("TEMPORAL_MARKERS", "TEMPORAL MARKERS", "TEMPORAL"):
            current = temporal
            continue
        if head == "DECISIONS":
            current = decisions
            continue
        if current is None:
            continue
        # Strip bullet prefixes "- ", "* ", "• ", digits "1. ", etc.
        item = s.lstrip("-*• \t")
        # Strip "1. " or "1) " prefixes.
        if len(item) > 2 and item[0].isdigit():
            for pfx_len in (3, 4):
                if (
                    pfx_len < len(item)
                    and item[pfx_len - 1] in ".)"
                    and item[pfx_len - 2].isdigit()
                ):
                    item = item[pfx_len:].lstrip()
                    break
        item = item.strip()
        if item:
            current.append(item)

    return MemoryIndex(
        conversation_id=conv_id,
        topics=topics,
        entities=entities,
        temporal=temporal,
        decisions=decisions,
        raw=raw,
    )


def _format_conversation(
    segments: list[Segment], turn_char_cap: int = 800, total_char_cap: int = 120_000
) -> tuple[str, int]:
    """Format a full conversation chronologically for the index prompt.

    Truncates very long individual turns and caps total size to stay within
    gpt-5-mini's window. Returns (text, num_turns_used).
    """
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)
    lines: list[str] = []
    total = 0
    used = 0
    for seg in sorted_segs:
        text = seg.text
        if len(text) > turn_char_cap:
            text = text[:turn_char_cap] + "…"
        line = f"[Turn {seg.turn_id}, {seg.role}]: {text}"
        if total + len(line) + 1 > total_char_cap:
            break
        lines.append(line)
        total += len(line) + 1
        used += 1
    return "\n".join(lines), used


class MemIndexBuilder:
    """Builds and caches a MemoryIndex per conversation_id."""

    def __init__(self, client: OpenAI | None = None):
        self.client = client or OpenAI(timeout=120.0)
        self.llm_cache = _JSONCache(MEMINDEX_LLM_CACHE)
        self.index_cache = _JSONCache(MEMINDEX_INDEX_FILE)
        self.llm_calls = 0

    def _llm(self, prompt: str) -> str:
        key = _sha(f"{MODEL}:{prompt}")
        cached = self.llm_cache.get(key)
        if isinstance(cached, str) and cached:
            return cached
        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=8000,
        )
        text = resp.choices[0].message.content or ""
        self.llm_cache.put(key, text)
        self.llm_calls += 1
        return text

    def build(self, conv_id: str, segments: list[Segment]) -> MemoryIndex:
        cached = self.index_cache.get(conv_id)
        if cached:
            return MemoryIndex.from_dict(cached)
        conv_text, n_used = _format_conversation(segments)
        prompt = MEMINDEX_PROMPT.format(
            n_turns=n_used, conversation_text=conv_text
        )
        raw = self._llm(prompt)
        idx = _parse_memindex(raw, conv_id)
        self.index_cache.put(conv_id, idx.to_dict())
        return idx

    def save(self) -> None:
        self.llm_cache.save()
        self.index_cache.save()


def build_all_indices(
    builder: MemIndexBuilder,
    stores: dict[str, SegmentStore],
    verbose: bool = True,
) -> dict[str, MemoryIndex]:
    """Build a memory index for every conversation across all benchmarks."""
    indices: dict[str, MemoryIndex] = {}
    for ds_name, store in stores.items():
        by_conv: dict[str, list[Segment]] = defaultdict(list)
        for seg in store.segments:
            by_conv[seg.conversation_id].append(seg)
        for conv_id, segs in by_conv.items():
            if conv_id in indices:
                continue
            t0 = time.time()
            idx = builder.build(conv_id, segs)
            elapsed = time.time() - t0
            if verbose:
                print(
                    f"  [{ds_name}] {conv_id}: "
                    f"topics={len(idx.topics)} entities={len(idx.entities)} "
                    f"temporal={len(idx.temporal)} decisions={len(idx.decisions)} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
            indices[conv_id] = idx
        builder.save()
    return indices


# ---------------------------------------------------------------------------
# Cue-generation variants (use memory index)
# ---------------------------------------------------------------------------
V15_WITH_INDEX_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{memindex_section}

RETRIEVED SO FAR:
{context_section}

The CONVERSATION INDEX above tells you what's actually in this conversation's \
memory. Use it to:
1. Identify WHICH topic/entity/decision is relevant to the question.
2. Generate cues using vocabulary tied to those specific topics/entities.
3. If the question asks for something NOT in the index, still generate your \
best-guess cues but note it in the assessment.

First, briefly assess: which index items match the question? What's still \
missing from what's been retrieved? Should you search for similar content \
or pivot to a different topic?

Then generate 2 search cues grounded in the index's vocabulary. Use specific \
words that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentences; reference index items by name>
CUE: <text>
CUE: <text>
Nothing else."""


V2F_V2_WITH_INDEX_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{memindex_section}

RETRIEVED SO FAR:
{context_section}

The CONVERSATION INDEX above tells you what's actually in this conversation's \
memory. Use it to:
1. Identify WHICH topic/entity/decision is relevant to the question.
2. Generate cues using vocabulary tied to those specific topics/entities.
3. If the question asks for something NOT in the index, still generate your \
best-guess cues but note it in the assessment.

First, briefly assess: which index items match the question? How is the \
search going? What kind of content is still missing?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues grounded in the index's vocabulary. Use specific \
words that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentences; reference index items by name>
CUE: <text>
CUE: <text>
Nothing else."""


INDEX_ONLY_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{memindex_section}

You do NOT see any retrieved excerpts — ONLY the conversation index above. \
The index is a faithful map of the conversation's contents. Decide which \
index items are relevant, then generate cues using their vocabulary.

First, briefly assess: which index items match the question? Are all \
needed items present in the index? If not, note it.

If the question implies MULTIPLE items or asks "all/every", generate cues \
that target each relevant sub-topic.

Then generate 2 search cues grounded in the index's vocabulary. Use specific \
words that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentences; name the index items you matched>
CUE: <text>
CUE: <text>
Nothing else."""


V2F_WITHOUT_INDEX_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED SO FAR:
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


VARIANTS = {
    "v15_with_index": {
        "prompt": V15_WITH_INDEX_PROMPT,
        "use_index": True,
        "use_primer": True,
    },
    "v2f_v2_with_index": {
        "prompt": V2F_V2_WITH_INDEX_PROMPT,
        "use_index": True,
        "use_primer": True,
    },
    "index_only": {
        "prompt": INDEX_ONLY_PROMPT,
        "use_index": True,
        "use_primer": False,  # still retrieve primer for backfill; just not in prompt
    },
    "v2f_without_index": {
        "prompt": V2F_WITHOUT_INDEX_PROMPT,
        "use_index": False,
        "use_primer": True,
    },
}


# ---------------------------------------------------------------------------
# Cue retriever
# ---------------------------------------------------------------------------
def _format_segments(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}"
        for seg in sorted_segs
    )


def _parse_cues(response: str) -> list[str]:
    cues: list[str] = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


@dataclass
class CueResult:
    segments: list[Segment]
    metadata: dict


class MemIndexRetriever:
    """Cue-generation variant driver.

    For every variant we do the same two-pass retrieval as the existing
    best_shot variants:
      Pass 0: embed question, retrieve top-10.
      Pass 1: one LLM call to generate 2 cues; each cue embeds + top-10.
    The only thing that differs is the prompt seen by the LLM.
    """

    def __init__(
        self,
        store: SegmentStore,
        variant: str,
        memindex: dict[str, MemoryIndex],
        client: OpenAI | None = None,
    ):
        self.store = store
        self.variant = variant
        self.cfg = VARIANTS[variant]
        self.memindex = memindex
        self.client = client or OpenAI(timeout=60.0)
        self.llm_cache = _JSONCache(MEMINDEX_LLM_CACHE)
        self.embedding_cache = _JSONCache(MEMINDEX_EMBED_CACHE)
        # Per-thread counters; accessed via _get_counters().
        self._tls = threading.local()

    def _counters(self) -> dict:
        c = getattr(self._tls, "counters", None)
        if c is None:
            c = {"embed": 0, "llm": 0}
            self._tls.counters = c
        return c

    def reset_counters(self) -> None:
        self._tls.counters = {"embed": 0, "llm": 0}

    @property
    def embed_calls(self) -> int:
        return self._counters()["embed"]

    @property
    def llm_calls(self) -> int:
        return self._counters()["llm"]

    def save_caches(self) -> None:
        self.llm_cache.save()
        self.embedding_cache.save()

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        key = _sha(text)
        cached = self.embedding_cache.get(key)
        c = self._counters()
        if isinstance(cached, list) and cached:
            c["embed"] += 1
            return np.array(cached, dtype=np.float32)
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        emb = np.array(resp.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(key, emb.tolist())
        c["embed"] += 1
        return emb

    def _llm(self, prompt: str) -> str:
        key = _sha(f"{MODEL}:{prompt}")
        cached = self.llm_cache.get(key)
        c = self._counters()
        if isinstance(cached, str) and cached:
            c["llm"] += 1
            return cached
        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )
        text = resp.choices[0].message.content or ""
        self.llm_cache.put(key, text)
        c["llm"] += 1
        return text

    def retrieve(self, question: str, conversation_id: str) -> CueResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []

        # Pass 0: primer retrieval (always done, regardless of whether the
        # prompt sees it — so that backfill never collides with arch picks).
        q_emb = self.embed_text(question)
        primer = self.store.search(
            q_emb, top_k=10, conversation_id=conversation_id
        )
        primer_segs = list(primer.segments)
        all_segs.extend(primer_segs)
        for s in primer_segs:
            exclude.add(s.index)

        # Build prompt
        if self.cfg["use_primer"]:
            context_section = _format_segments(primer_segs)
        else:
            context_section = "(deliberately omitted in this variant)"

        if self.cfg["use_index"]:
            idx = self.memindex.get(conversation_id)
            if idx is None:
                memindex_section = (
                    "CONVERSATION INDEX (what's actually in memory):\n"
                    "(unavailable for this conversation)"
                )
            else:
                memindex_section = idx.format_for_prompt()
            prompt = self.cfg["prompt"].format(
                question=question,
                memindex_section=memindex_section,
                context_section=context_section,
            )
        else:
            prompt = self.cfg["prompt"].format(
                question=question, context_section=context_section
            )

        output = self._llm(prompt)
        cues = _parse_cues(output)[:2]

        # Pass 1: run each cue
        for cue in cues:
            cue_emb = self.embed_text(cue)
            res = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in res.segments:
                if seg.index not in exclude:
                    all_segs.append(seg)
                    exclude.add(seg.index)

        return CueResult(
            segments=all_segs,
            metadata={
                "variant": self.variant,
                "output": output,
                "cues": cues,
            },
        )


# ---------------------------------------------------------------------------
# Fair-backfill evaluation (K=20, K=50)
# ---------------------------------------------------------------------------
def _recall(got: set[int], src: set[int]) -> float:
    if not src:
        return 1.0
    return len(got & src) / len(src)


def _fair_backfill(
    arch_segs: list[Segment],
    cosine_segs: list[Segment],
    source_ids: set[int],
    budget: int,
) -> tuple[float, float]:
    seen: set[int] = set()
    arch_unique: list[Segment] = []
    for s in arch_segs:
        if s.index not in seen:
            arch_unique.append(s)
            seen.add(s.index)
    arch_at_K = arch_unique[:budget]
    arch_indices = {s.index for s in arch_at_K}
    if len(arch_at_K) < budget:
        backfill = [s for s in cosine_segs if s.index not in arch_indices]
        needed = budget - len(arch_at_K)
        arch_at_K = arch_at_K + backfill[:needed]
    arch_at_K = arch_at_K[:budget]
    baseline_at_K = cosine_segs[:budget]
    return (
        _recall({s.turn_id for s in baseline_at_K}, source_ids),
        _recall({s.turn_id for s in arch_at_K}, source_ids),
    )


def evaluate_question(retriever: MemIndexRetriever, question: dict) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    retriever.reset_counters()
    t0 = time.time()
    result = retriever.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedupe arch segs
    seen: set[int] = set()
    arch_segs: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            arch_segs.append(seg)
            seen.add(seg.index)

    # Cosine top-K (K=max budget)
    q_emb = retriever.embed_text(q_text)
    cosine_res = retriever.store.search(
        q_emb, top_k=max(BUDGETS), conversation_id=conv_id
    )
    cosine_segs = list(cosine_res.segments)

    row = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "total_arch_retrieved": len(arch_segs),
        "embed_calls": retriever.embed_calls,
        "llm_calls": retriever.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "metadata": result.metadata,
    }
    for K in BUDGETS:
        b, a = _fair_backfill(arch_segs, cosine_segs, source_ids, K)
        row["fair_backfill"][f"baseline_r@{K}"] = round(b, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a - b, 4)
    return row


def summarize(results: list[dict], variant: str, dataset: str) -> dict:
    n = len(results)
    if n == 0:
        return {"variant": variant, "dataset": dataset, "n": 0}
    s: dict = {"variant": variant, "dataset": dataset, "n": n}
    for K in BUDGETS:
        b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in results]
        a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in results]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses
        s[f"baseline_r@{K}"] = round(b_mean, 4)
        s[f"arch_r@{K}"] = round(a_mean, 4)
        s[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
        s[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
    s["avg_total_retrieved"] = round(
        sum(r["total_arch_retrieved"] for r in results) / n, 1
    )
    s["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
    s["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 1)
    return s


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out: dict[str, dict] = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry = {"n": n}
        for K in BUDGETS:
            b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in rs]
            a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in rs]
            b_mean = sum(b_vals) / n
            a_mean = sum(a_vals) / n
            wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
            losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
            ties = n - wins - losses
            entry[f"baseline_r@{K}"] = round(b_mean, 4)
            entry[f"arch_r@{K}"] = round(a_mean, 4)
            entry[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
            entry[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
        out[cat] = entry
    return out


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(ds_name: str) -> tuple[SegmentStore, list[dict]]:
    cfg = DATASETS[ds_name]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        questions = json.load(f)
    if cfg["filter"]:
        questions = [q for q in questions if cfg["filter"](q)]
    if cfg["max_questions"]:
        questions = questions[: cfg["max_questions"]]
    return store, questions


# ---------------------------------------------------------------------------
# Qualitative analysis helpers
# ---------------------------------------------------------------------------
def print_qualitative(
    indices: dict[str, MemoryIndex],
    results_by_variant: dict[str, list[dict]],
    num_indices: int = 4,
    num_cues: int = 6,
) -> None:
    print("\n" + "=" * 80)
    print("QUALITATIVE: sample memory indices")
    print("=" * 80)
    conv_ids = list(indices.keys())[:num_indices]
    for cid in conv_ids:
        idx = indices[cid]
        print(f"\n--- {cid} ---")
        print(
            f"topics({len(idx.topics)}): "
            + "; ".join(idx.topics[:10])
        )
        print(
            f"entities({len(idx.entities)}): "
            + "; ".join(idx.entities[:12])
        )
        print(
            f"temporal({len(idx.temporal)}): "
            + "; ".join(idx.temporal[:8])
        )
        print(
            f"decisions({len(idx.decisions)}): "
            + "; ".join(idx.decisions[:8])
        )

    print("\n" + "=" * 80)
    print("QUALITATIVE: sample generated cues per variant")
    print("=" * 80)
    for variant, rows in results_by_variant.items():
        print(f"\n--- {variant} ---")
        for row in rows[:num_cues]:
            cues = row["metadata"].get("cues", [])
            print(
                f"  Q: {row['question'][:90]} "
                f"(cat={row['category']})"
            )
            for c in cues:
                print(f"    CUE: {c[:150]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None,
                        help="Run only this variant (default: all)")
    parser.add_argument("--dataset", default=None,
                        help="Run only this dataset (default: all)")
    parser.add_argument("--indices-only", action="store_true",
                        help="Build (and cache) memory indices, then exit.")
    parser.add_argument("--qualitative", action="store_true",
                        help="After eval, print sample indices + cues.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results file exists.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Thread-pool workers for per-question calls.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all datasets (even when a subset requested, we may need other
    # conversations for the index cache to be complete — but we only BUILD
    # what's needed for the requested subset).
    ds_names = [args.dataset] if args.dataset else list(DATASETS.keys())
    for ds in ds_names:
        if ds not in DATASETS:
            raise SystemExit(f"unknown dataset: {ds}")

    stores: dict[str, SegmentStore] = {}
    questions_by_ds: dict[str, list[dict]] = {}
    for ds in ds_names:
        store, qs = load_dataset(ds)
        stores[ds] = store
        questions_by_ds[ds] = qs
        print(
            f"Loaded {ds}: {len(qs)} questions, "
            f"{len(store.segments)} segments, "
            f"{len(set(s.conversation_id for s in store.segments))} conversations"
        )

    # ---- Step 1: memory indices ----
    print("\n=== Step 1: building memory indices ===")
    builder = MemIndexBuilder()
    indices = build_all_indices(builder, stores)
    builder.save()
    print(f"Built/loaded {len(indices)} memory indices.")

    if args.indices_only:
        return

    # ---- Step 2: cue generation variants ----
    variants = [args.variant] if args.variant else list(VARIANTS.keys())
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant: {v}")

    all_summaries: dict = {}
    qualitative_samples: dict[str, list[dict]] = {}

    for variant in variants:
        print(f"\n=== Step 2: variant = {variant} ===")
        for ds_name in ds_names:
            out_path = RESULTS_DIR / f"memindex_{variant}_{ds_name}.json"
            if out_path.exists() and not args.force:
                print(f"  skip {variant} on {ds_name} (results exist)")
                with open(out_path) as f:
                    saved = json.load(f)
                all_summaries.setdefault(variant, {})[ds_name] = saved["summary"]
                if variant not in qualitative_samples and saved.get("results"):
                    qualitative_samples[variant] = saved["results"][:10]
                continue

            store = stores[ds_name]
            questions = questions_by_ds[ds_name]
            retriever = MemIndexRetriever(store, variant, indices)
            results: list[dict] = [None] * len(questions)  # type: ignore

            def _work(idx_q: tuple[int, dict]) -> tuple[int, dict | None]:
                i, q = idx_q
                try:
                    return i, evaluate_question(retriever, q)
                except Exception as e:
                    print(f"  [q{i}] ERROR: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    return i, None

            completed_count = [0]
            lock = threading.Lock()

            def _announce(i: int, q: dict) -> None:
                with lock:
                    completed_count[0] += 1
                    n = completed_count[0]
                    print(
                        f"  [{ds_name} {n}/{len(questions)}] "
                        f"{q.get('category', '?')}: {q['question'][:55]}...",
                        flush=True,
                    )

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {
                    ex.submit(_work, (i, q)): (i, q)
                    for i, q in enumerate(questions)
                }
                for f in as_completed(futures):
                    i, q = futures[f]
                    idx, row = f.result()
                    if row is not None:
                        results[idx] = row
                    _announce(idx, q)
                    if completed_count[0] % 5 == 0:
                        retriever.save_caches()

            # Drop any failed (None) rows, preserving order
            results = [r for r in results if r is not None]
            retriever.save_caches()

            summary = summarize(results, variant, ds_name)
            by_cat = summarize_by_category(results)

            print(f"\n  --- {variant} on {ds_name} ---")
            for K in BUDGETS:
                print(
                    f"    r@{K}: baseline={summary[f'baseline_r@{K}']:.3f} "
                    f"arch={summary[f'arch_r@{K}']:.3f} "
                    f"delta={summary[f'delta_r@{K}']:+.3f} "
                    f"W/T/L={summary[f'W/T/L_r@{K}']}"
                )
            print(
                f"    avg retrieved={summary['avg_total_retrieved']:.0f} "
                f"llm={summary['avg_llm_calls']:.1f} "
                f"embed={summary['avg_embed_calls']:.1f}"
            )
            for cat, c in by_cat.items():
                print(
                    f"    {cat:26s} (n={c['n']}): "
                    f"r@20 d={c['delta_r@20']:+.3f} "
                    f"r@50 d={c['delta_r@50']:+.3f} "
                    f"W/T/L@50={c['W/T/L_r@50']}"
                )

            with open(out_path, "w") as f:
                json.dump(
                    {
                        "variant": variant,
                        "dataset": ds_name,
                        "summary": summary,
                        "category_breakdown": by_cat,
                        "results": results,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            print(f"    saved {out_path}")
            all_summaries.setdefault(variant, {})[ds_name] = summary
            if variant not in qualitative_samples:
                qualitative_samples[variant] = results[:10]

    # ---- Aggregated summary + comparison table ----
    summary_path = RESULTS_DIR / "memindex_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")

    print("\n" + "=" * 110)
    print("MEMORY INDEX SUMMARY")
    print("=" * 110)
    header = (
        f"{'Variant':<22s} {'Dataset':<14s} "
        f"{'base@20':>8s} {'arch@20':>8s} {'d@20':>7s} {'W/T/L@20':>10s} "
        f"{'base@50':>8s} {'arch@50':>8s} {'d@50':>7s} {'W/T/L@50':>10s} "
        f"{'llm':>5s}"
    )
    print(header)
    print("-" * len(header))
    for variant in variants:
        for ds_name in ds_names:
            if ds_name not in all_summaries.get(variant, {}):
                continue
            s = all_summaries[variant][ds_name]
            print(
                f"{variant:<22s} {ds_name:<14s} "
                f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
                f"{s['delta_r@20']:>+7.3f} {s['W/T/L_r@20']:>10s} "
                f"{s['baseline_r@50']:>8.3f} {s['arch_r@50']:>8.3f} "
                f"{s['delta_r@50']:>+7.3f} {s['W/T/L_r@50']:>10s} "
                f"{s.get('avg_llm_calls', 0):>5.1f}"
            )

    if args.qualitative:
        print_qualitative(indices, qualitative_samples)


if __name__ == "__main__":
    main()
