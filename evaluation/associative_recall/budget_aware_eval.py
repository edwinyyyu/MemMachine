"""Budget-aware retrieval evaluation.

Every architecture must return EXACTLY K segments at budget K.
No reranking. No over-retrieve-then-prune. Budgets are planned up front
and the per-cue top_k is sized so the unioned (deduplicated) total lands
exactly at K.

Budgets tested:
  K=20 (tight), K=50 (standard), K=100 (generous)

Architectures:
  K=20:
    baseline_20:    cosine top-20
    v15_tight_20:   hop0=10 + 2 cues x 5 (V15 prompt)
    v2f_tight_20:   hop0=10 + 2 cues x 5 (V2f prompt)
    pure_cue_20:    no hop0, 4 cues x 5 (V2f-style, no context)
    single_cue_20:  hop0=15 + 1 cue x 5 (V15 prompt)

  K=50:
    baseline_50:    cosine top-50
    v15_tight_50:   hop0=20 + 2 cues x 15 (V15 prompt)
    v2f_tight_50:   hop0=20 + 2 cues x 15 (V2f prompt)
    wide_cue_50:    hop0=10 + 4 cues x 10 (V2f prompt, 4 cues)
    gencheck_50:    hop0=15 + 2 cues x 10 (V2f) + 3 gap cues x 5 = 50

  K=100:
    baseline_100:   cosine top-100
    v2f_100:        hop0=30 + 2 cues x 20 + 1 gap x 30 = 100

All runs use neighbor_radius=0 (no neighbor expansion).

Usage:
    uv run python budget_aware_eval.py [--arch <name>] [--budget 20|50|100]
    uv run python budget_aware_eval.py --all
    uv run python budget_aware_eval.py --list
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
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
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_FILE_EMBED = "budget_embedding_cache.json"
CACHE_FILE_LLM = "budget_llm_cache.json"
LOCK_DIR = Path(__file__).resolve().parent / "cache"


@contextmanager
def _file_lock(path: Path):
    """Acquire an exclusive flock on a lock file.

    Serializes cache writes across concurrent processes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.with_suffix(path.suffix + ".lock")
    with open(lock_file, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Cache — reads from all prior caches, writes to budget-specific file
# ---------------------------------------------------------------------------
class BudgetEmbeddingCache(EmbeddingCache):
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        # Read ALL existing embedding caches for maximum reuse
        for path in sorted(self.cache_dir.glob("*embedding_cache.json")):
            try:
                with open(path) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                # Try salvage partial JSON
                try:
                    with open(path) as f:
                        text = f.read()
                    obj, _ = json.JSONDecoder().raw_decode(text)
                    self._cache.update(obj)
                except Exception:
                    pass
        self.cache_file = self.cache_dir / CACHE_FILE_EMBED
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        with _file_lock(self.cache_file):
            existing = {}
            if self.cache_file.exists():
                try:
                    with open(self.cache_file) as f:
                        existing = json.load(f)
                except json.JSONDecodeError:
                    # Corrupt: attempt salvage
                    with open(self.cache_file) as f:
                        text = f.read()
                    try:
                        existing, _ = json.JSONDecoder().raw_decode(text)
                    except Exception:
                        existing = {}
            existing.update(self._new_entries)
            tmp = self.cache_file.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(existing, f)
            tmp.replace(self.cache_file)
        self._new_entries = {}


class BudgetLLMCache(LLMCache):
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for path in sorted(self.cache_dir.glob("*llm_cache.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                try:
                    with open(path) as f:
                        text = f.read()
                    data, _ = json.JSONDecoder().raw_decode(text)
                except Exception:
                    data = {}
            for k, v in data.items():
                if v:
                    self._cache[k] = v
        self.cache_file = self.cache_dir / CACHE_FILE_LLM
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        with _file_lock(self.cache_file):
            existing = {}
            if self.cache_file.exists():
                try:
                    with open(self.cache_file) as f:
                        existing = json.load(f)
                except json.JSONDecodeError:
                    with open(self.cache_file) as f:
                        text = f.read()
                    try:
                        existing, _ = json.JSONDecoder().raw_decode(text)
                    except Exception:
                        existing = {}
            existing.update(self._new_entries)
            tmp = self.cache_file.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(existing, f)
            tmp.replace(self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# V15 prompt, parameterized cue count, context section optional
V15_PROMPT_TEMPLATE = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate {num_cues} search {cue_word} based on your assessment. Use \
specific vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
{cue_format}
Nothing else."""

# V2f prompt with completeness + anti-question instructions
V2F_PROMPT_TEMPLATE = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate {num_cues} search {cue_word} based on your assessment. Use \
specific vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
{cue_format}
Nothing else."""

# Gap assessment prompt — parameterized number of gaps
GAP_ASSESSMENT_PROMPT = """\
You are reviewing retrieval results for a question about a past \
conversation. Given what has been retrieved so far, identify what is still \
MISSING to answer the question.

QUESTION: {question}

RETRIEVED SEGMENTS:
{formatted_segments}

Think critically:
1. What specific pieces of information for this question are NOT yet \
retrieved?
2. What adjacent topics would appear nearby in the conversation but have \
not been surfaced?

Generate exactly {num_gaps} search {gap_word} targeting the biggest gaps. \
Each should use vocabulary that would appear in the missing conversation \
content. Do NOT write questions or search commands.

Format:
ASSESSMENT: <1-2 sentence evaluation>
{gap_format}
Nothing else."""


def _render_cue_prompt(template: str, question: str,
                       context_section: str, num_cues: int) -> str:
    cue_word = "cue" if num_cues == 1 else "cues"
    cue_format = "\n".join(["CUE: <text>"] * num_cues)
    return template.format(
        question=question,
        context_section=context_section,
        num_cues=num_cues,
        cue_word=cue_word,
        cue_format=cue_format,
    )


def _render_gap_prompt(question: str, formatted_segments: str,
                       num_gaps: int) -> str:
    gap_word = "gap" if num_gaps == 1 else "gaps"
    gap_format = "\n".join(["GAP: <text>"] * num_gaps)
    return GAP_ASSESSMENT_PROMPT.format(
        question=question,
        formatted_segments=formatted_segments,
        num_gaps=num_gaps,
        gap_word=gap_word,
        gap_format=gap_format,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_segments(segments: list[Segment], max_items: int = 16,
                     max_chars: int = 250) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}"
        for seg in sorted_segs
    )


def _build_context_section(all_segments: list[Segment]) -> str:
    if not all_segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    return (
        "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
        + _format_segments(all_segments)
    )


def _parse_lines(response: str, prefix: str) -> list[str]:
    items: list[str] = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            if value:
                items.append(value)
    return items


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BudgetBase:
    """Base class with embed/LLM helpers + counters."""

    name: str = "base"
    budget: int = 0

    def __init__(self, store: SegmentStore, budget: int,
                 client: OpenAI | None = None):
        self.store = store
        self.budget = budget
        self.client = client or OpenAI(timeout=120.0)
        self.embedding_cache = BudgetEmbeddingCache()
        self.llm_cache = BudgetLLMCache()
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
        response = self.client.embeddings.create(
            model=EMBED_MODEL, input=[text]
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, embedding)
        self.embed_calls += 1
        return embedding

    def llm_call(self, prompt: str, model: str = MODEL,
                 max_tokens: int = 2000) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
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

    # Subclasses override
    def retrieve(self, question: str, conversation_id: str) -> "BudgetResult":
        raise NotImplementedError


@dataclass
class BudgetResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
def _retrieve_into(
    store: SegmentStore,
    query_emb: np.ndarray,
    top_k: int,
    conversation_id: str,
    exclude: set[int],
    into: list[Segment],
) -> int:
    """Retrieve top_k NEW segments for this query and append to `into`.

    Returns the number of new segments added (capped at top_k). Because
    store.search excludes `exclude`, asking for top_k returns up to top_k
    items that are all new (unless the conversation has fewer segments).
    """
    if top_k <= 0:
        return 0
    result = store.search(
        query_emb, top_k=top_k, conversation_id=conversation_id,
        exclude_indices=exclude,
    )
    added = 0
    for seg in result.segments:
        if seg.index not in exclude:
            into.append(seg)
            exclude.add(seg.index)
            added += 1
            if added >= top_k:
                break
    return added


def _top_up_with_baseline(
    store: SegmentStore,
    query_emb: np.ndarray,
    budget: int,
    conversation_id: str,
    exclude: set[int],
    segments: list[Segment],
) -> int:
    """If under budget (e.g., cue retrieval returned < requested because
    the conversation is too short), top up with cosine top-k on the
    question embedding. Returns segments added.
    """
    needed = budget - len(segments)
    if needed <= 0:
        return 0
    return _retrieve_into(
        store, query_emb, needed, conversation_id, exclude, segments
    )


# ===========================================================================
# Cosine baseline
# ===========================================================================
class CosineBaseline(BudgetBase):
    name_root = "baseline"

    def retrieve(self, question: str, conversation_id: str) -> BudgetResult:
        query_emb = self.embed_text(question)
        result = self.store.search(
            query_emb, top_k=self.budget, conversation_id=conversation_id
        )
        segments = list(result.segments[: self.budget])
        return BudgetResult(
            segments=segments,
            metadata={"name": f"baseline_{self.budget}", "cues": []},
        )


# ===========================================================================
# Generic hop-0 + cue architecture
# ===========================================================================
class HopCueArch(BudgetBase):
    """Generic architecture: hop0 + N cues, prompt parameterized.

    prompt_kind in {"v15", "v2f"}.
    hop0 can be 0 (pure cue). num_cues and per_cue_k must satisfy
    hop0 + num_cues * per_cue_k == budget.
    """

    def __init__(self, store: SegmentStore, budget: int, *,
                 hop0: int, num_cues: int, per_cue_k: int,
                 prompt_kind: str, name: str,
                 client: OpenAI | None = None):
        super().__init__(store, budget, client)
        self.hop0 = hop0
        self.num_cues = num_cues
        self.per_cue_k = per_cue_k
        self.prompt_kind = prompt_kind
        self.name = name
        assert hop0 + num_cues * per_cue_k == budget, (
            f"Budget mismatch: hop0={hop0} + {num_cues}*{per_cue_k} != {budget}"
        )

    def retrieve(self, question: str, conversation_id: str) -> BudgetResult:
        exclude: set[int] = set()
        all_segments: list[Segment] = []

        query_emb = self.embed_text(question)

        # Hop 0
        if self.hop0 > 0:
            _retrieve_into(
                self.store, query_emb, self.hop0, conversation_id,
                exclude, all_segments,
            )

        # Build context for prompt
        context_section = _build_context_section(all_segments)

        # Select prompt template
        if self.prompt_kind == "v15":
            template = V15_PROMPT_TEMPLATE
        elif self.prompt_kind == "v2f":
            template = V2F_PROMPT_TEMPLATE
        else:
            raise ValueError(f"Unknown prompt_kind: {self.prompt_kind}")

        prompt = _render_cue_prompt(
            template, question, context_section, self.num_cues
        )
        output = self.llm_call(prompt)
        cues = _parse_lines(output, "CUE:")

        # If parse fails, fall back to empty cues. We'll pad with question.
        used_cues: list[str] = []
        for i in range(self.num_cues):
            cue = cues[i] if i < len(cues) else question
            used_cues.append(cue)
            cue_emb = self.embed_text(cue)
            _retrieve_into(
                self.store, cue_emb, self.per_cue_k, conversation_id,
                exclude, all_segments,
            )

        # Top up if short
        _top_up_with_baseline(
            self.store, query_emb, self.budget, conversation_id,
            exclude, all_segments,
        )

        # Strict truncate (should be no-op if math is correct)
        all_segments = all_segments[: self.budget]

        return BudgetResult(
            segments=all_segments,
            metadata={
                "name": self.name,
                "output": output,
                "cues": used_cues,
                "hop0": self.hop0,
                "num_cues": self.num_cues,
                "per_cue_k": self.per_cue_k,
            },
        )


# ===========================================================================
# Hop-0 + Cues + Gencheck architecture (gencheck_50, v2f_100)
# ===========================================================================
class HopCueGapArch(BudgetBase):
    """hop0 + v2f cues + gap cues. All counts fixed up front.

    budget = hop0 + num_cues * per_cue_k + num_gaps * per_gap_k.
    """

    def __init__(self, store: SegmentStore, budget: int, *,
                 hop0: int, num_cues: int, per_cue_k: int,
                 num_gaps: int, per_gap_k: int,
                 name: str,
                 prompt_kind: str = "v2f",
                 client: OpenAI | None = None):
        super().__init__(store, budget, client)
        self.hop0 = hop0
        self.num_cues = num_cues
        self.per_cue_k = per_cue_k
        self.num_gaps = num_gaps
        self.per_gap_k = per_gap_k
        self.prompt_kind = prompt_kind
        self.name = name
        total = hop0 + num_cues * per_cue_k + num_gaps * per_gap_k
        assert total == budget, (
            f"Budget mismatch: {hop0} + {num_cues}*{per_cue_k} + "
            f"{num_gaps}*{per_gap_k} = {total} != {budget}"
        )

    def retrieve(self, question: str, conversation_id: str) -> BudgetResult:
        exclude: set[int] = set()
        all_segments: list[Segment] = []
        query_emb = self.embed_text(question)

        # Hop 0
        if self.hop0 > 0:
            _retrieve_into(
                self.store, query_emb, self.hop0, conversation_id,
                exclude, all_segments,
            )

        # Cue generation (v2f)
        context_section = _build_context_section(all_segments)
        template = (V2F_PROMPT_TEMPLATE if self.prompt_kind == "v2f"
                    else V15_PROMPT_TEMPLATE)
        cue_prompt = _render_cue_prompt(
            template, question, context_section, self.num_cues
        )
        cue_output = self.llm_call(cue_prompt)
        cues = _parse_lines(cue_output, "CUE:")

        used_cues: list[str] = []
        for i in range(self.num_cues):
            cue = cues[i] if i < len(cues) else question
            used_cues.append(cue)
            cue_emb = self.embed_text(cue)
            _retrieve_into(
                self.store, cue_emb, self.per_cue_k, conversation_id,
                exclude, all_segments,
            )

        # Gap generation
        gap_context = _format_segments(all_segments, max_items=20, max_chars=260)
        gap_prompt = _render_gap_prompt(question, gap_context, self.num_gaps)
        gap_output = self.llm_call(gap_prompt)
        gaps = _parse_lines(gap_output, "GAP:")

        used_gaps: list[str] = []
        for i in range(self.num_gaps):
            gap = gaps[i] if i < len(gaps) else question
            used_gaps.append(gap)
            gap_emb = self.embed_text(gap)
            _retrieve_into(
                self.store, gap_emb, self.per_gap_k, conversation_id,
                exclude, all_segments,
            )

        # Top up if needed
        _top_up_with_baseline(
            self.store, query_emb, self.budget, conversation_id,
            exclude, all_segments,
        )

        all_segments = all_segments[: self.budget]

        return BudgetResult(
            segments=all_segments,
            metadata={
                "name": self.name,
                "cue_output": cue_output,
                "gap_output": gap_output,
                "cues": used_cues,
                "gaps": used_gaps,
                "hop0": self.hop0,
                "num_cues": self.num_cues,
                "per_cue_k": self.per_cue_k,
                "num_gaps": self.num_gaps,
                "per_gap_k": self.per_gap_k,
            },
        )


# ===========================================================================
# Registry: factory per architecture
# ===========================================================================
def build_arch(name: str, store: SegmentStore) -> BudgetBase:
    """Build one architecture instance by name."""

    # Budget 20
    if name == "baseline_20":
        arch = CosineBaseline(store, budget=20)
        arch.name = name
        return arch
    if name == "v15_tight_20":
        return HopCueArch(
            store, budget=20, hop0=10, num_cues=2, per_cue_k=5,
            prompt_kind="v15", name=name,
        )
    if name == "v2f_tight_20":
        return HopCueArch(
            store, budget=20, hop0=10, num_cues=2, per_cue_k=5,
            prompt_kind="v2f", name=name,
        )
    if name == "pure_cue_20":
        return HopCueArch(
            store, budget=20, hop0=0, num_cues=4, per_cue_k=5,
            prompt_kind="v2f", name=name,
        )
    if name == "single_cue_20":
        return HopCueArch(
            store, budget=20, hop0=15, num_cues=1, per_cue_k=5,
            prompt_kind="v15", name=name,
        )

    # Budget 50
    if name == "baseline_50":
        arch = CosineBaseline(store, budget=50)
        arch.name = name
        return arch
    if name == "v15_tight_50":
        return HopCueArch(
            store, budget=50, hop0=20, num_cues=2, per_cue_k=15,
            prompt_kind="v15", name=name,
        )
    if name == "v2f_tight_50":
        return HopCueArch(
            store, budget=50, hop0=20, num_cues=2, per_cue_k=15,
            prompt_kind="v2f", name=name,
        )
    if name == "wide_cue_50":
        return HopCueArch(
            store, budget=50, hop0=10, num_cues=4, per_cue_k=10,
            prompt_kind="v2f", name=name,
        )
    if name == "gencheck_50":
        return HopCueGapArch(
            store, budget=50, hop0=15, num_cues=2, per_cue_k=10,
            num_gaps=3, per_gap_k=5, prompt_kind="v2f", name=name,
        )

    # Budget 100
    if name == "baseline_100":
        arch = CosineBaseline(store, budget=100)
        arch.name = name
        return arch
    if name == "v2f_100":
        return HopCueGapArch(
            store, budget=100, hop0=30, num_cues=2, per_cue_k=20,
            num_gaps=1, per_gap_k=30, prompt_kind="v2f", name=name,
        )

    raise ValueError(f"Unknown architecture: {name}")


BUDGET_K20 = ["baseline_20", "v15_tight_20", "v2f_tight_20",
              "pure_cue_20", "single_cue_20"]
BUDGET_K50 = ["baseline_50", "v15_tight_50", "v2f_tight_50",
              "wide_cue_50", "gencheck_50"]
BUDGET_K100 = ["baseline_100", "v2f_100"]
ALL_ARCH_NAMES = BUDGET_K20 + BUDGET_K50 + BUDGET_K100


ARCH_BUDGET = {a: 20 for a in BUDGET_K20}
ARCH_BUDGET.update({a: 50 for a in BUDGET_K50})
ARCH_BUDGET.update({a: 100 for a in BUDGET_K100})


# ===========================================================================
# Evaluation logic
# ===========================================================================
def compute_recall(retrieved_turn_ids: set[int],
                   source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(arch: BudgetBase, question: dict,
                 verbose: bool = False) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    segments = result.segments

    # Enforce exact budget
    budget = arch.budget
    actual_count = len(segments)
    turn_ids = {s.turn_id for s in segments}
    # Verify: count of segments equals budget (allow shortfall only when
    # the conversation is smaller than budget)
    recall = compute_recall(turn_ids, source_ids)

    row = {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question.get("question_index"),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "budget": budget,
        "actual_count": actual_count,
        "recall": recall,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "metadata": {
            k: v for k, v in result.metadata.items()
            if k in ("name", "cues", "gaps", "hop0", "num_cues",
                     "per_cue_k", "num_gaps", "per_gap_k")
        },
    }

    if verbose:
        print(
            f"    actual={actual_count}/{budget}  recall={recall:.3f}  "
            f"embed={arch.embed_calls} llm={arch.llm_calls} "
            f"time={elapsed:.1f}s"
        )

    return row


def summarize(results: list[dict], arch_name: str, benchmark: str,
              budget: int) -> dict:
    n = len(results)
    if n == 0:
        return {"arch": arch_name, "benchmark": benchmark,
                "budget": budget, "n": 0}

    recalls = [r["recall"] for r in results]
    actual_counts = [r["actual_count"] for r in results]
    under_budget = sum(1 for c in actual_counts if c < budget)

    per_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        per_cat[r["category"]].append(r["recall"])
    cat_summary = {
        cat: {
            "n": len(vals),
            "mean_recall": round(sum(vals) / len(vals), 4),
        }
        for cat, vals in sorted(per_cat.items())
    }

    return {
        "arch": arch_name,
        "benchmark": benchmark,
        "budget": budget,
        "n": n,
        "mean_recall": round(sum(recalls) / n, 4),
        "mean_actual_count": round(sum(actual_counts) / n, 2),
        "under_budget": under_budget,
        "avg_embed_calls": round(
            sum(r["embed_calls"] for r in results) / n, 2
        ),
        "avg_llm_calls": round(
            sum(r["llm_calls"] for r in results) / n, 2
        ),
        "avg_time_s": round(sum(r["time_s"] for r in results) / n, 2),
        "per_category": cat_summary,
    }


# ===========================================================================
# Data loading
# ===========================================================================
DATASETS = {
    "locomo_30q": {
        "questions_file": "questions_extended.json",
        "segments_npz": "segments_extended.npz",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "limit": 30,
    },
    "synthetic_19q": {
        "questions_file": "questions_synthetic.json",
        "segments_npz": "segments_synthetic.npz",
        "filter": lambda q: True,
        "limit": None,
    },
    "puzzle_16q": {
        "questions_file": "questions_puzzle.json",
        "segments_npz": "segments_puzzle.npz",
        "filter": lambda q: True,
        "limit": None,
    },
    "advanced_23q": {
        "questions_file": "questions_advanced.json",
        "segments_npz": "segments_advanced.npz",
        "filter": lambda q: True,
        "limit": None,
    },
}


def load_dataset(key: str) -> tuple[list[dict], SegmentStore]:
    meta = DATASETS[key]
    with open(DATA_DIR / meta["questions_file"]) as f:
        qs = json.load(f)
    qs = [q for q in qs if meta["filter"](q)]
    if meta["limit"] is not None:
        qs = qs[: meta["limit"]]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=meta["segments_npz"])
    return qs, store


# ===========================================================================
# Runner
# ===========================================================================
def run_architecture_on_dataset(
    arch_name: str,
    dataset_key: str,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    budget = ARCH_BUDGET[arch_name]
    result_file = RESULTS_DIR / f"budget_{arch_name}_{dataset_key}.json"

    if result_file.exists() and not force:
        with open(result_file) as f:
            saved = json.load(f)
        print(
            f"  [cache] {arch_name} on {dataset_key}: "
            f"recall={saved['summary']['mean_recall']:.3f} "
            f"(n={saved['summary']['n']}, "
            f"budget={saved['summary']['budget']})"
        )
        return saved["summary"]

    qs, store = load_dataset(dataset_key)
    arch = build_arch(arch_name, store)

    print(
        f"\n>>> {arch_name} (K={budget}) on {dataset_key} "
        f"({len(qs)} questions, {len(store.segments)} segments)"
    )
    t_arch = time.time()
    results: list[dict] = []
    for i, q in enumerate(qs):
        q_short = q["question"][:60].replace("\n", " ")
        print(
            f"  [{i+1}/{len(qs)}] {q['category']}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_one(arch, q, verbose=verbose)
            results.append(row)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    elapsed_total = time.time() - t_arch
    summary = summarize(results, arch_name, dataset_key, budget)
    summary["wall_time_s"] = round(elapsed_total, 1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(
            {"results": results, "summary": summary},
            f, indent=2, default=str,
        )

    # Sanity: verify budget enforcement
    over_budget = [r for r in results if r["actual_count"] > budget]
    if over_budget:
        print(
            f"  !! BUDGET VIOLATED on {len(over_budget)} questions "
            f"(actual > {budget})"
        )
    print(
        f"  -> mean recall={summary['mean_recall']:.3f}  "
        f"mean_count={summary['mean_actual_count']:.2f}/{budget}  "
        f"under_budget={summary['under_budget']}  "
        f"embed={summary['avg_embed_calls']:.1f} "
        f"llm={summary['avg_llm_calls']:.1f}  "
        f"time={elapsed_total:.0f}s"
    )
    return summary


def print_final_table(all_summaries: dict) -> None:
    """Summary layout:
       Architecture | K=20 LoCoMo | K=20 Synth | K=20 Puzzle | K=20 Advanced |
       K=50 LoCoMo | ...
    """
    print("\n" + "=" * 110)
    print("FINAL COMPARISON TABLE (r@K, absolute)")
    print("=" * 110)

    datasets = ["locomo_30q", "synthetic_19q", "puzzle_16q", "advanced_23q"]
    short_ds = {
        "locomo_30q": "LoCoMo",
        "synthetic_19q": "Synth",
        "puzzle_16q": "Puzzle",
        "advanced_23q": "Advanced",
    }

    for group_name, arch_list in [
        ("K=20", BUDGET_K20), ("K=50", BUDGET_K50), ("K=100", BUDGET_K100),
    ]:
        budget = int(group_name.split("=")[1])
        header = f"{'Architecture':<22s}" + "".join(
            f"{short_ds[ds]:>12s}" for ds in datasets
        )
        print(f"\n--- Budget {group_name} ---")
        print(header)
        print("-" * len(header))
        for arch_name in arch_list:
            row = f"{arch_name:<22s}"
            for ds in datasets:
                key = (arch_name, ds)
                if key in all_summaries:
                    rec = all_summaries[key].get("mean_recall", 0.0)
                    under = all_summaries[key].get("under_budget", 0)
                    suffix = "*" if under > 0 else " "
                    row += f"{rec:>11.3f}{suffix}"
                else:
                    row += f"{'—':>12s}"
            print(row)

    # Delta vs baseline at same budget
    print("\n" + "=" * 110)
    print("DELTA vs cosine baseline (same K, same dataset)")
    print("=" * 110)
    baselines = {
        20: "baseline_20", 50: "baseline_50", 100: "baseline_100",
    }
    for group_name, arch_list in [
        ("K=20", BUDGET_K20), ("K=50", BUDGET_K50), ("K=100", BUDGET_K100),
    ]:
        budget = int(group_name.split("=")[1])
        baseline_name = baselines[budget]
        header = f"{'Architecture':<22s}" + "".join(
            f"{short_ds[ds]:>12s}" for ds in datasets
        )
        print(f"\n--- Budget {group_name} (delta vs {baseline_name}) ---")
        print(header)
        print("-" * len(header))
        for arch_name in arch_list:
            row = f"{arch_name:<22s}"
            for ds in datasets:
                key = (arch_name, ds)
                base_key = (baseline_name, ds)
                if key in all_summaries and base_key in all_summaries:
                    r = all_summaries[key].get("mean_recall", 0.0)
                    b = all_summaries[base_key].get("mean_recall", 0.0)
                    row += f"{r - b:>+12.3f}"
                else:
                    row += f"{'—':>12s}"
            print(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Budget-aware retrieval evaluation"
    )
    parser.add_argument("--arch", type=str, default=None,
                        help="Run a single architecture")
    parser.add_argument("--budget", type=int, default=None,
                        choices=[20, 50, 100],
                        help="Run all architectures at this budget")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASETS.keys()),
                        help="Restrict to one dataset")
    parser.add_argument("--all", action="store_true",
                        help="Run all architectures on all datasets")
    parser.add_argument("--list", action="store_true",
                        help="List all architectures")
    parser.add_argument("--force", action="store_true",
                        help="Rerun even if result file exists")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Architectures:")
        for name in ALL_ARCH_NAMES:
            print(f"  {name} (budget={ARCH_BUDGET[name]})")
        print("\nDatasets:")
        for key in DATASETS:
            print(f"  {key}")
        return

    # Determine architectures to run
    if args.arch:
        if args.arch not in ALL_ARCH_NAMES:
            print(f"Unknown architecture: {args.arch}")
            print(f"Available: {', '.join(ALL_ARCH_NAMES)}")
            sys.exit(1)
        arch_names = [args.arch]
    elif args.budget is not None:
        arch_names = [
            a for a in ALL_ARCH_NAMES if ARCH_BUDGET[a] == args.budget
        ]
    else:
        arch_names = ALL_ARCH_NAMES

    # Determine datasets
    if args.dataset:
        dataset_keys = [args.dataset]
    else:
        dataset_keys = list(DATASETS.keys())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries: dict[tuple[str, str], dict] = {}
    for arch_name in arch_names:
        for dataset_key in dataset_keys:
            try:
                summary = run_architecture_on_dataset(
                    arch_name, dataset_key,
                    force=args.force, verbose=args.verbose,
                )
                all_summaries[(arch_name, dataset_key)] = summary
            except Exception as e:
                print(
                    f"  FATAL on {arch_name}/{dataset_key}: "
                    f"{type(e).__name__}: {e}"
                )
                import traceback
                traceback.print_exc()

    # Save overall summary
    summary_file = RESULTS_DIR / "budget_all_summaries.json"
    payload = {
        f"{a}@{ds}": summary
        for (a, ds), summary in all_summaries.items()
    }
    with open(summary_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nAll summaries saved: {summary_file}")

    print_final_table(all_summaries)


if __name__ == "__main__":
    main()
