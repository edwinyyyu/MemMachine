"""Entity extraction cue generation experiments.

Theory: v2f works by staying ON-TOPIC (re-finding relevant content). Extracting
concrete entities/keywords from the already-retrieved segments grounds cue
generation in terms that ACTUALLY exist in the conversation rather than
guessing aliases from general knowledge.

Four variants:
  A. entity_extract_simple   : initial retrieve -> extract terms -> 2 cues
  B. entity_extract_v2f      : A + v2f's assessment + completeness + anti-question
  C. entity_weighted_question: initial retrieve -> extract -> append entities
                               to question, retrieve (no cue generation)
  D. entity_per_segment_cue  : initial retrieve -> extract one phrase per
                               top-3 segment -> 3 standalone cues

FAIR K-budget evaluation at K=20 and K=50 on 4 datasets (locomo_30q,
synthetic_19q, puzzle_16q, advanced_23q). Architecture returns a segment
pool; pool + cosine baseline backfill truncated to exactly K (same protocol
as cot_universal.py).

Comparison (loaded from existing budget_*.json / cot results):
  - baseline  (cosine top-K)
  - v2f_tight (current reference)
  - CoT       (from cot_chain_of_thought_*.json, re-evaluated at fair K)

Usage:
    uv run python entity_extract.py [--variant VAR] [--dataset DS] [--force]
"""

from __future__ import annotations

import argparse
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
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_FILE_EMB = CACHE_DIR / "entity_extract_embedding_cache.json"
CACHE_FILE_LLM = CACHE_DIR / "entity_extract_llm_cache.json"
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Caches — read from all existing caches, write to entity_extract_* files
# ---------------------------------------------------------------------------
class EntityEmbeddingCache(EmbeddingCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for p in sorted(self.cache_dir.glob("*embedding_cache.json")):
            try:
                with open(p) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        self.cache_file = CACHE_FILE_EMB
        self._new: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        existing.update(self._new)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new = {}


class EntityLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for p in sorted(self.cache_dir.glob("*llm_cache.json")):
            try:
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
            except (json.JSONDecodeError, OSError):
                pass
        self.cache_file = CACHE_FILE_LLM
        self._new: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new[key] = response

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        existing.update(self._new)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_segments(
    segments: list[Segment], max_items: int = 12, max_chars: int = 260
) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}"
        for s in sorted_segs
    )


def _parse_prefixed(text: str, prefix: str) -> list[str]:
    out: list[str] = []
    pfx = prefix.upper()
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith(pfx):
            val = line[len(prefix):].strip()
            if val:
                out.append(val)
    return out


def _parse_terms(text: str) -> list[str]:
    """Parse the TERMS: line — comma-separated list of entities."""
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("TERMS:"):
            raw = line[6:].strip()
            terms = [t.strip() for t in raw.split(",")]
            return [t for t in terms if t]
    return []


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Variant A — simple extraction + 2 cues
ENTITY_SIMPLE_PROMPT = """\
You are performing semantic retrieval over a conversation history. Cues will \
be embedded and matched via cosine similarity.

Question: {question}

Retrieved segments:
{segments}

Step 1: List 5-10 specific terms from the retrieved segments (names, dates, \
tools, numbers, technical words). These are your anchors — they EXIST in the \
conversation.

TERMS: <comma-separated list>

Step 2: Generate 2 cues that combine your extracted terms with aspects of \
the question that aren't yet well-covered. Each cue should sound like actual \
conversation content.

CUE: <text using extracted terms>
CUE: <text using extracted terms>

Nothing else."""


# Variant B — extraction + v2f structure (assessment + completeness + anti-question)
ENTITY_V2F_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

Retrieved segments:
{segments}

Step 1: List 5-10 specific terms from the retrieved segments (names, dates, \
tools, numbers, technical words). These are your anchors — they EXIST in the \
conversation.

TERMS: <comma-separated list>

Step 2: Briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

ASSESSMENT: <1-2 sentence self-evaluation>

Step 3: Generate 2 cues that combine your extracted terms with missing \
aspects. Use specific vocabulary that would appear in the target conversation \
turns. Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

CUE: <text using extracted terms>
CUE: <text using extracted terms>

Nothing else."""


# Variant C — extract terms (used to enhance question)
ENTITY_TERMS_ONLY_PROMPT = """\
You are helping search a conversation history. We want to find more relevant \
turns by enriching the original question with specific terms.

Question: {question}

Retrieved segments:
{segments}

List 5-10 specific terms from the retrieved segments (names, dates, tools, \
numbers, technical words). These are your anchors — they EXIST in the \
conversation.

TERMS: <comma-separated list>

Nothing else."""


# Variant D — per-segment phrase extraction (3 phrases)
ENTITY_PER_SEGMENT_PROMPT = """\
You are performing semantic retrieval over a conversation history. Cues will \
be embedded and matched via cosine similarity.

Question: {question}

Here are the top 3 retrieved segments. For EACH segment, identify the single \
most important entity or short phrase (a name, concrete noun, or specific \
term — 1 to 5 words) that could serve as a search anchor for finding related \
content. Do not paraphrase the question. Use only words that appear in the \
segment.

{numbered_segments}

Format exactly:
CUE: <phrase from segment 1>
CUE: <phrase from segment 2>
CUE: <phrase from segment 3>

Nothing else."""


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
@dataclass
class EntityResult:
    segments: list[Segment]
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


class EntityExtractBase:
    """Base class: initial cosine retrieve, then variant-specific cue logic."""

    name: str = "entity_base"

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        initial_k: int = 10,
        per_cue_k: int = 10,
    ) -> None:
        self.store = store
        self.client = client or OpenAI(timeout=120.0)
        self.embedding_cache = EntityEmbeddingCache()
        self.llm_cache = EntityLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0
        self.initial_k = initial_k
        self.per_cue_k = per_cue_k

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

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

    def retrieve(
        self, question: str, conversation_id: str
    ) -> EntityResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Variant A — entity_extract_simple
# ---------------------------------------------------------------------------
class EntityExtractSimple(EntityExtractBase):
    name = "entity_simple"

    def retrieve(
        self, question: str, conversation_id: str
    ) -> EntityResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []

        # Initial retrieval
        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        # LLM: extract terms + 2 cues
        prompt = ENTITY_SIMPLE_PROMPT.format(
            question=question,
            segments=_format_segments(all_segs, max_items=12),
        )
        output = self.llm_call(prompt)
        terms = _parse_terms(output)
        cues = _parse_prefixed(output, "CUE:")[:2]

        # Retrieve per cue
        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=self.per_cue_k,
                conversation_id=conversation_id, exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return EntityResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": self.name,
                "terms": terms,
                "cues": cues,
                "output": output,
            },
        )


# ---------------------------------------------------------------------------
# Variant B — entity_extract_v2f
# ---------------------------------------------------------------------------
class EntityExtractV2f(EntityExtractBase):
    name = "entity_v2f"

    def retrieve(
        self, question: str, conversation_id: str
    ) -> EntityResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []

        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        prompt = ENTITY_V2F_PROMPT.format(
            question=question,
            segments=_format_segments(all_segs, max_items=12),
        )
        output = self.llm_call(prompt)
        terms = _parse_terms(output)
        assessment = ""
        for line in output.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("ASSESSMENT:"):
                assessment = line[11:].strip()
                break
        cues = _parse_prefixed(output, "CUE:")[:2]

        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=self.per_cue_k,
                conversation_id=conversation_id, exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return EntityResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": self.name,
                "terms": terms,
                "assessment": assessment,
                "cues": cues,
                "output": output,
            },
        )


# ---------------------------------------------------------------------------
# Variant C — entity_weighted_question (no cue gen; append entities)
# ---------------------------------------------------------------------------
class EntityWeightedQuestion(EntityExtractBase):
    name = "entity_weighted_question"

    def retrieve(
        self, question: str, conversation_id: str
    ) -> EntityResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []

        # Initial retrieval
        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        # Extract terms only
        prompt = ENTITY_TERMS_ONLY_PROMPT.format(
            question=question,
            segments=_format_segments(all_segs, max_items=12),
        )
        output = self.llm_call(prompt)
        terms = _parse_terms(output)

        # Append terms to original question for second retrieval
        enhanced_query = question
        if terms:
            enhanced_query = question + " " + ", ".join(terms)

        # One retrieve with enhanced query (ask for 2 * per_cue_k = 20 like A/B)
        total_cue_budget = 2 * self.per_cue_k
        eq_emb = self.embed_text(enhanced_query)
        result = self.store.search(
            eq_emb, top_k=total_cue_budget,
            conversation_id=conversation_id, exclude_indices=exclude,
        )
        for s in result.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        return EntityResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": self.name,
                "terms": terms,
                "enhanced_query": enhanced_query[:500],
                "output": output,
            },
        )


# ---------------------------------------------------------------------------
# Variant D — entity_per_segment_cue (3 phrases, one per top-3 segment)
# ---------------------------------------------------------------------------
class EntityPerSegmentCue(EntityExtractBase):
    name = "entity_per_segment"

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        initial_k: int = 10,
        # per_cue_k = 20/3 and 20 across 3 cues: keep per_cue_k at 10 to mirror
        # A/B's total retrieve breadth, though final pool is truncated at K anyway.
        per_cue_k: int = 7,
    ) -> None:
        super().__init__(store, client, initial_k, per_cue_k)

    def retrieve(
        self, question: str, conversation_id: str
    ) -> EntityResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []

        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        # Top 3 segments BY ORIGINAL SCORE (order in r0.segments is rank order)
        top3 = r0.segments[:3]
        numbered = "\n".join(
            f"Segment {i+1} [Turn {s.turn_id}, {s.role}]: {s.text[:400]}"
            for i, s in enumerate(top3)
        )

        prompt = ENTITY_PER_SEGMENT_PROMPT.format(
            question=question,
            numbered_segments=numbered,
        )
        output = self.llm_call(prompt)
        cues = _parse_prefixed(output, "CUE:")[:3]

        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=self.per_cue_k,
                conversation_id=conversation_id, exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return EntityResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": self.name,
                "cues": cues,
                "output": output,
            },
        )


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
VARIANTS: dict[str, type[EntityExtractBase]] = {
    "entity_simple": EntityExtractSimple,
    "entity_v2f": EntityExtractV2f,
    "entity_weighted_question": EntityWeightedQuestion,
    "entity_per_segment": EntityPerSegmentCue,
}


# ---------------------------------------------------------------------------
# FAIR K-budget evaluation (same protocol as cot_universal.py)
# ---------------------------------------------------------------------------
def compute_recall(
    retrieved_turn_ids: set[int], source_turn_ids: set[int]
) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    arch: EntityExtractBase, question: dict, verbose: bool = False
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for s in result.segments:
        if s.index not in seen:
            arch_segments.append(s)
            seen.add(s.index)

    # Cosine baseline at max(BUDGETS) for backfill
    q_emb = arch.embed_text(q_text)
    max_b = max(BUDGETS)
    baseline = arch.store.search(
        q_emb, top_k=max_b, conversation_id=conv_id
    )

    arch_idx = {s.index for s in arch_segments}
    backfilled = list(arch_segments) + [
        s for s in baseline.segments if s.index not in arch_idx
    ]

    recalls: dict[str, float] = {}
    baseline_recalls: dict[str, float] = {}
    for K in BUDGETS:
        a_ids = {s.turn_id for s in backfilled[:K]}
        b_ids = {s.turn_id for s in baseline.segments[:K]}
        recalls[f"r@{K}"] = compute_recall(a_ids, source_ids)
        baseline_recalls[f"r@{K}"] = compute_recall(b_ids, source_ids)

    row = {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question.get("question_index"),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "arch_pool_size": len(arch_segments),
        "baseline_recalls": baseline_recalls,
        "arch_recalls": recalls,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "metadata": {
            k: v for k, v in result.metadata.items()
            if k in ("name", "terms", "cues", "assessment", "enhanced_query")
        },
    }

    if verbose:
        print(
            f"    pool={len(arch_segments)} "
            f"r@20: base={baseline_recalls['r@20']:.3f} "
            f"arch={recalls['r@20']:.3f}  "
            f"r@50: base={baseline_recalls['r@50']:.3f} "
            f"arch={recalls['r@50']:.3f}  "
            f"emb={arch.embed_calls} llm={arch.llm_calls}"
        )
    return row


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Runner with incremental per-question flushing
# ---------------------------------------------------------------------------
def run_variant_on_dataset(
    variant_name: str,
    dataset_key: str,
    force: bool = False,
    verbose: bool = False,
) -> list[dict]:
    result_file = RESULTS_DIR / f"entity_{variant_name}_{dataset_key}.json"

    if result_file.exists() and not force:
        with open(result_file) as f:
            return json.load(f)

    qs, store = load_dataset(dataset_key)
    arch_cls = VARIANTS[variant_name]
    arch = arch_cls(store)

    print(
        f"\n>>> {variant_name} on {dataset_key}: {len(qs)} questions, "
        f"{len(store.segments)} segments",
        flush=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for i, q in enumerate(qs):
        q_short = q["question"][:60].replace("\n", " ")
        print(
            f"  [{i+1}/{len(qs)}] {q['category']}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_one(arch, q, verbose=verbose)
            rows.append(row)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        sys.stdout.flush()

        # Flush incrementally every question (crash safety)
        tmp = result_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        tmp.replace(result_file)

        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    print(f"  Saved -> {result_file}")
    return rows


# ---------------------------------------------------------------------------
# Cross-dataset aggregation + comparison
# ---------------------------------------------------------------------------
def load_budget_recall_by_qkey(
    arch_name: str, dataset_key: str
) -> dict[tuple, float]:
    path = RESULTS_DIR / f"budget_{arch_name}_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        payload = json.load(f)
    out: dict[tuple, float] = {}
    for r in payload.get("results", []):
        key = (r["conversation_id"], r.get("question_index"))
        out[key] = r["recall"]
    return out


def load_cot_recall_by_qkey(dataset_key: str) -> dict[int, dict[tuple, float]]:
    """Load CoT's fair-backfill recall per K from cot_chain_of_thought_*.json."""
    path = RESULTS_DIR / f"cot_chain_of_thought_{dataset_key}.json"
    if not path.exists():
        return {K: {} for K in BUDGETS}
    with open(path) as f:
        rows = json.load(f)
    out: dict[int, dict[tuple, float]] = {K: {} for K in BUDGETS}
    for r in rows:
        key = (r["conversation_id"], r.get("question_index"))
        for K in BUDGETS:
            lbl = f"r@{K}"
            if "cot_recalls" in r and lbl in r["cot_recalls"]:
                out[K][key] = r["cot_recalls"][lbl]
    return out


def per_category_rows(
    variant_name: str,
    dataset_key: str,
    arch_rows: list[dict],
) -> list[dict]:
    """Build per-category comparison rows (K=20, K=50)."""
    # Preload comparison arch recalls per K
    budget_by_arch: dict[str, dict[int, dict[tuple, float]]] = {
        "baseline": {
            K: load_budget_recall_by_qkey(f"baseline_{K}", dataset_key)
            for K in BUDGETS
        },
        "v2f_tight": {
            K: load_budget_recall_by_qkey(f"v2f_tight_{K}", dataset_key)
            for K in BUDGETS
        },
    }
    cot_by_k = load_cot_recall_by_qkey(dataset_key)

    rows_by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in arch_rows:
        rows_by_cat[r["category"]].append(r)

    out: list[dict] = []
    for K in BUDGETS:
        for cat, rows in sorted(rows_by_cat.items()):
            n = len(rows)
            if n == 0:
                continue
            arch_recs = [r["arch_recalls"][f"r@{K}"] for r in rows]
            arch_mean = sum(arch_recs) / n

            b_vals, v2f_vals, cot_vals = [], [], []
            for r in rows:
                key = (r["conversation_id"], r["question_index"])
                if key in budget_by_arch["baseline"][K]:
                    b_vals.append(budget_by_arch["baseline"][K][key])
                if key in budget_by_arch["v2f_tight"][K]:
                    v2f_vals.append(budget_by_arch["v2f_tight"][K][key])
                if key in cot_by_k[K]:
                    cot_vals.append(cot_by_k[K][key])

            def _mean(xs: list[float]) -> float | None:
                return (sum(xs) / len(xs)) if xs else None

            b_mean = _mean(b_vals)
            v2f_mean = _mean(v2f_vals)
            cot_mean = _mean(cot_vals)

            out.append({
                "variant": variant_name,
                "dataset": dataset_key,
                "category": cat,
                "K": K,
                "n": n,
                "baseline": b_mean,
                "v2f": v2f_mean,
                "cot": cot_mean,
                "arch": arch_mean,
                "vs_v2f": (arch_mean - v2f_mean) if v2f_mean is not None else None,
                "vs_cot": (arch_mean - cot_mean) if cot_mean is not None else None,
                "vs_base": (arch_mean - b_mean) if b_mean is not None else None,
            })
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def fmt(val: float | None, plus: bool = False) -> str:
    if val is None:
        return "    —"
    s = f"{val:+.3f}" if plus else f"{val:.3f}"
    return f"{s:>6s}"


def print_per_category_table(rows: list[dict], K: int) -> None:
    filtered = [r for r in rows if r["K"] == K]
    if not filtered:
        return
    print(f"\n{'='*120}")
    print(f"PER-CATEGORY at K={K} (recall, fair backfill)")
    print(f"{'='*120}")
    hdr = (
        f"{'Variant':<26s} {'Dataset':<14s} {'Category':<26s} {'n':>3s} "
        f"{'Base':>7s} {'v2f':>7s} {'CoT':>7s} {'Arch':>7s}  "
        f"{'vs v2f':>7s} {'vs CoT':>7s} {'vs base':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    last = (None, None)
    for r in filtered:
        if last != (None, None) and (r["variant"], r["dataset"]) != last:
            print("")
        last = (r["variant"], r["dataset"])
        print(
            f"{r['variant']:<26s} {r['dataset']:<14s} {r['category']:<26s} "
            f"{r['n']:>3d} "
            f"{fmt(r['baseline'])} {fmt(r['v2f'])} {fmt(r['cot'])} "
            f"{fmt(r['arch'])}  "
            f"{fmt(r['vs_v2f'], True)} {fmt(r['vs_cot'], True)} "
            f"{fmt(r['vs_base'], True)}"
        )


def print_overall_by_dataset(rows: list[dict], K: int) -> None:
    """Dataset-level mean (weighted by category n) per variant."""
    print(f"\n{'-'*96}")
    print(f"PER-DATASET at K={K} (weighted by category n within dataset)")
    print(f"{'-'*96}")
    hdr = (
        f"{'Variant':<26s} {'Dataset':<14s} {'n':>3s} "
        f"{'Base':>7s} {'v2f':>7s} {'CoT':>7s} {'Arch':>7s}  "
        f"{'vs v2f':>7s} {'vs CoT':>7s} {'vs base':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))

    by_vd: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["K"] == K:
            by_vd[(r["variant"], r["dataset"])].append(r)

    for (var, ds), rlist in sorted(by_vd.items()):
        total_n = sum(r["n"] for r in rlist)

        def _wavg(key: str) -> float | None:
            items = [(r[key], r["n"]) for r in rlist if r[key] is not None]
            if not items:
                return None
            tot = sum(n for _, n in items)
            return sum(v * n for v, n in items) / tot

        b = _wavg("baseline")
        v2f = _wavg("v2f")
        cot = _wavg("cot")
        arch = _wavg("arch")
        vs_v2f = (arch - v2f) if (arch is not None and v2f is not None) else None
        vs_cot = (arch - cot) if (arch is not None and cot is not None) else None
        vs_b = (arch - b) if (arch is not None and b is not None) else None
        print(
            f"{var:<26s} {ds:<14s} {total_n:>3d} "
            f"{fmt(b)} {fmt(v2f)} {fmt(cot)} {fmt(arch)}  "
            f"{fmt(vs_v2f, True)} {fmt(vs_cot, True)} {fmt(vs_b, True)}"
        )


def print_cross_dataset_mean(rows: list[dict], K: int) -> None:
    """Primary metric: cross-dataset mean r@K per variant.

    Computed as mean of per-dataset weighted means (dataset-equal weighting,
    avoids over-weighting larger datasets). Also reports a pooled "question-
    weighted" mean for reference.
    """
    print(f"\n{'='*96}")
    print(f"CROSS-DATASET MEAN r@{K} (primary headline)")
    print(f"{'='*96}")
    hdr = (
        f"{'Variant':<26s} {'nQ':>4s}  "
        f"{'Base':>7s} {'v2f':>7s} {'CoT':>7s} {'Arch':>7s}  "
        f"{'vs v2f':>7s} {'vs CoT':>7s} {'vs base':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))

    by_var_ds: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["K"] == K:
            by_var_ds[(r["variant"], r["dataset"])].append(r)

    # variant -> dict[metric][dataset] = weighted mean within that dataset
    variant_ds_means: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    variant_total_n: dict[str, int] = defaultdict(int)
    for (var, ds), rlist in by_var_ds.items():
        variant_total_n[var] += sum(r["n"] for r in rlist)
        for metric in ("baseline", "v2f", "cot", "arch"):
            items = [(r[metric], r["n"]) for r in rlist if r[metric] is not None]
            if not items:
                continue
            tot = sum(n for _, n in items)
            variant_ds_means[var][metric][ds] = sum(
                v * n for v, n in items
            ) / tot

    for var in sorted(variant_ds_means.keys()):
        def _ds_mean(m: str) -> float | None:
            ds_vals = list(variant_ds_means[var][m].values())
            if not ds_vals:
                return None
            return sum(ds_vals) / len(ds_vals)

        b = _ds_mean("baseline")
        v2f = _ds_mean("v2f")
        cot = _ds_mean("cot")
        arch = _ds_mean("arch")
        vs_v2f = (arch - v2f) if (arch is not None and v2f is not None) else None
        vs_cot = (arch - cot) if (arch is not None and cot is not None) else None
        vs_b = (arch - b) if (arch is not None and b is not None) else None
        print(
            f"{var:<26s} {variant_total_n[var]:>4d}  "
            f"{fmt(b)} {fmt(v2f)} {fmt(cot)} {fmt(arch)}  "
            f"{fmt(vs_v2f, True)} {fmt(vs_cot, True)} {fmt(vs_b, True)}"
        )


def print_category_deltas(rows: list[dict], K: int) -> None:
    """Best/worst categories per variant across datasets."""
    print(f"\n{'='*96}")
    print(f"BEST/WORST categories at K={K} (delta vs v2f)")
    print(f"{'='*96}")

    by_var_cat: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["K"] == K and r["vs_v2f"] is not None:
            by_var_cat[(r["variant"], r["category"])].append(r["vs_v2f"])

    # Aggregate per variant
    variant_cat_means: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (var, cat), deltas in by_var_cat.items():
        mean_delta = sum(deltas) / len(deltas)
        variant_cat_means[var].append((cat, mean_delta))

    for var in sorted(variant_cat_means.keys()):
        ordered = sorted(variant_cat_means[var], key=lambda x: x[1],
                         reverse=True)
        print(f"\n  {var}:")
        print("    Top helps (highest +delta vs v2f):")
        for cat, d in ordered[:5]:
            print(f"      {cat:<32s} {d:+.3f}")
        print("    Top regressions (lowest delta vs v2f):")
        for cat, d in ordered[-5:]:
            print(f"      {cat:<32s} {d:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", type=str, default=None,
        choices=list(VARIANTS.keys()),
        help="Run a single variant (default: all four)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list(DATASETS.keys()),
        help="Restrict to one dataset (default: all four)",
    )
    parser.add_argument("--force", action="store_true",
                        help="Rerun even if result file exists")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variants = [args.variant] if args.variant else list(VARIANTS.keys())
    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())

    all_cat_rows: list[dict] = []
    for var in variants:
        for ds in datasets:
            try:
                arch_rows = run_variant_on_dataset(
                    var, ds, force=args.force, verbose=args.verbose,
                )
                all_cat_rows.extend(per_category_rows(var, ds, arch_rows))
            except Exception as e:
                print(
                    f"  FATAL on {var}/{ds}: {type(e).__name__}: {e}",
                    flush=True,
                )
                import traceback
                traceback.print_exc()

    # Save roll-up summary
    out_path = RESULTS_DIR / "entity_extract_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_cat_rows, f, indent=2)
    print(f"\nSaved summary -> {out_path}")

    for K in BUDGETS:
        print_per_category_table(all_cat_rows, K)
        print_overall_by_dataset(all_cat_rows, K)

    for K in BUDGETS:
        print_cross_dataset_mean(all_cat_rows, K)
        print_category_deltas(all_cat_rows, K)


if __name__ == "__main__":
    main()
