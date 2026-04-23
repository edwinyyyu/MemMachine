"""Constraint-type-enumerated cue generator.

Tests three variants of a prompt that enumerates constraint TYPES and emits
one cue per type (or per selected subset) in casual chat register:

  A. type_enumerated         — 7 cues, one per type, retrieval only with these
  B. type_enumerated_selective — model picks 3-4 MOST RELEVANT types, one cue each
  C. v2f_plus_types          — v2f first (2 cues, 20 segments), then type
                                enumeration fills remaining budget slots at K=50

Evaluation: LoCoMo, Synthetic, Puzzle, Advanced — K=20 and K=50 with fair
backfill (matches fair_backfill_eval.py semantics).

Usage:
    uv run python type_enumerated.py [--variant <A|B|C|all>] [--force]
    uv run python type_enumerated.py --list
"""

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
from prompt_optimization import META_V2F_PROMPT, _format_segments


def _parse_cues(response: str) -> list[str]:
    """Parse CUE: lines from LLM response.

    Handles variations like:
        CUE: text
        [ARRIVAL] CUE: text
        ARRIVAL: CUE: text
        CUE (ARRIVAL): text
    """
    import re
    cues: list[str] = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Find "CUE" followed by optional bracketed/parenthesized text, then ":"
        m = re.search(r"\bCUE\b\s*(?:[\[(][^\])]*[\])]\s*)?:\s*(.+)", line)
        if m:
            cue = m.group(1).strip()
            if cue:
                cues.append(cue)
    return cues

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50]

CACHE_FILE_LLM = CACHE_DIR / "type_enum_llm_cache.json"
CACHE_FILE_EMB = CACHE_DIR / "type_enum_embedding_cache.json"


# ---------------------------------------------------------------------------
# Caches — read all existing caches, write to type_enum-specific files
# ---------------------------------------------------------------------------
class TypeEnumEmbeddingCache(EmbeddingCache):
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
            "task_exec_embedding_cache.json",
            "general_embedding_cache.json",
            "adaptive_embedding_cache.json",
            "fulleval_embedding_cache.json",
            "constraint_embedding_cache.json",
            "type_enum_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = CACHE_FILE_EMB
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


class TypeEnumLLMCache(LLMCache):
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
            "task_exec_llm_cache.json",
            "general_llm_cache.json",
            "adaptive_llm_cache.json",
            "fulleval_llm_cache.json",
            "constraint_llm_cache.json",
            "type_enum_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = CACHE_FILE_LLM
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

# Variant A: full enumeration — 7 cues, one per type
TYPE_ENUMERATED_PROMPT = """\
Generate cues to find scattered constraints/details in a conversation. Each cue \
should mimic how someone would ACTUALLY phrase that type of information in chat.

Question: {question}

RETRIEVED SO FAR:
{context_section}

Generate ONE cue per type below. Use casual first-person chat register. Use \
deictic pronouns (she, he, they) NOT named entities. No quotes around phrases.

[ARRIVAL]: when someone says they arrived/showed up somewhere
[PREFERENCE]: when someone expresses a like/dislike
[CONFLICT]: when a disagreement or issue is discussed
[UPDATE]: informal updates like "oh I forgot to mention" or "just got a message"
[RESOLUTION]: resolutions like "we cleared the air" or "actually it's fine now"
[AFTERTHOUGHT]: casual additions like "wait one more thing" or "btw"
[PHYSICAL]: spatial/physical details like seating, location, position

Format:
CUE: <casual chat text for this type>
(7 cues total, one per type)"""


# Variant B: selective — model picks 3-4 most relevant types
TYPE_ENUMERATED_SELECTIVE_PROMPT = """\
Generate cues to find scattered constraints/details in a conversation. Each cue \
should mimic how someone would ACTUALLY phrase that type of information in chat.

Question: {question}

RETRIEVED SO FAR:
{context_section}

Here are 7 constraint/detail TYPES that can appear in conversations:

[ARRIVAL]: when someone says they arrived/showed up somewhere
[PREFERENCE]: when someone expresses a like/dislike
[CONFLICT]: when a disagreement or issue is discussed
[UPDATE]: informal updates like "oh I forgot to mention" or "just got a message"
[RESOLUTION]: resolutions like "we cleared the air" or "actually it's fine now"
[AFTERTHOUGHT]: casual additions like "wait one more thing" or "btw"
[PHYSICAL]: spatial/physical details like seating, location, position

Pick the 3-4 types MOST RELEVANT to this question (given what's already been \
retrieved), and emit one cue per selected type. Use casual first-person chat \
register. Use deictic pronouns (she, he, they) NOT named entities. No quotes \
around phrases.

Format:
SELECTED: <comma-separated list of chosen type names, e.g. ARRIVAL, CONFLICT>
CUE: <casual chat text for first selected type>
CUE: <casual chat text for second selected type>
CUE: <casual chat text for third selected type>
(3-4 cues total, one per SELECTED type)"""


# Variant C reuses META_V2F_PROMPT for the first stage, then adds type enumeration
TYPE_ENUMERATED_ADDITIVE_PROMPT = TYPE_ENUMERATED_PROMPT  # same 7 types for stage 2


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
@dataclass
class TypeEnumResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


class TypeEnumBase:
    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = TypeEnumEmbeddingCache()
        self.llm_cache = TypeEnumLLMCache()
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
            max_completion_tokens=3000,
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


def _build_context_section(segments: list[Segment]) -> str:
    if not segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    return (
        "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
        + _format_segments(segments, max_items=12, max_chars=250)
    )


# ---------------------------------------------------------------------------
# Variant A: type_enumerated (7 cues, one per type, replacing v2f cue step)
# ---------------------------------------------------------------------------
class TypeEnumeratedVariant(TypeEnumBase):
    """Hop 0: cosine top-10. Then 1 LLM call emits 7 typed cues; each retrieves
    top-3 (to fit ~21 segments on top of 10 hop0 within a ~50 budget).

    Per-cue top_k is configurable. Default top_k=3 leads to ~10+21=31 segments.
    """

    def __init__(self, store: SegmentStore,
                 per_cue_top_k: int = 3,
                 client: OpenAI | None = None):
        super().__init__(store, client)
        self.per_cue_top_k = per_cue_top_k

    def retrieve(self, question: str, conversation_id: str) -> TypeEnumResult:
        # Hop 0: cosine top-10 (matches v2f hop-0 for apples-to-apples)
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        # Single LLM call emitting 7 typed cues
        context_section = _build_context_section(all_segments)
        prompt = TYPE_ENUMERATED_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        # Expand each cue
        for cue in cues[:7]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=self.per_cue_top_k,
                conversation_id=conversation_id, exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return TypeEnumResult(
            segments=all_segments,
            metadata={
                "variant": "type_enumerated",
                "output": output,
                "cues": cues[:7],
                "per_cue_top_k": self.per_cue_top_k,
            },
        )


# ---------------------------------------------------------------------------
# Variant B: type_enumerated_selective (3-4 cues)
# ---------------------------------------------------------------------------
class TypeEnumeratedSelectiveVariant(TypeEnumBase):
    """Hop 0: cosine top-10. Then 1 LLM call picks 3-4 relevant types and emits
    one cue per type. Each cue retrieves top-5 (~10 + 20 = 30 segments).
    """

    def __init__(self, store: SegmentStore,
                 per_cue_top_k: int = 5,
                 client: OpenAI | None = None):
        super().__init__(store, client)
        self.per_cue_top_k = per_cue_top_k

    def retrieve(self, question: str, conversation_id: str) -> TypeEnumResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        context_section = _build_context_section(all_segments)
        prompt = TYPE_ENUMERATED_SELECTIVE_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        # Parse SELECTED types for metadata/analysis (non-critical)
        selected_types: list[str] = []
        for line in output.splitlines():
            line = line.strip()
            if line.upper().startswith("SELECTED:"):
                rest = line[len("SELECTED:"):].strip()
                selected_types = [t.strip() for t in rest.split(",") if t.strip()]
                break

        for cue in cues[:4]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=self.per_cue_top_k,
                conversation_id=conversation_id, exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return TypeEnumResult(
            segments=all_segments,
            metadata={
                "variant": "type_enumerated_selective",
                "output": output,
                "cues": cues[:4],
                "selected_types": selected_types,
                "per_cue_top_k": self.per_cue_top_k,
            },
        )


# ---------------------------------------------------------------------------
# Variant C: v2f_plus_types (additive; types only fill beyond K=20)
# ---------------------------------------------------------------------------
class V2fPlusTypesVariant(TypeEnumBase):
    """Stage 1 (v2f): cosine top-10 + 2 v2f cues × top-10 each = ~30 segs that
    preserve v2f's r@20 behaviour.

    Stage 2 (types): 1 LLM call emits 7 typed cues; each retrieves top-3 using
    the same exclude set. Those segments append AFTER the v2f stage segments,
    so fair-backfill at K=20 uses only the v2f portion, and at K=50 additional
    typed segments can contribute.
    """

    def __init__(self, store: SegmentStore,
                 per_cue_top_k: int = 3,
                 client: OpenAI | None = None):
        super().__init__(store, client)
        self.per_cue_top_k = per_cue_top_k

    def retrieve(self, question: str, conversation_id: str) -> TypeEnumResult:
        # Stage 1: v2f (hop0 + 2 cues x top-10)
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        v2f_context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(all_segments)
        )
        v2f_prompt = META_V2F_PROMPT.format(
            question=question, context_section=v2f_context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        v2f_cues = _parse_cues(v2f_output)[:2]

        for cue in v2f_cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=10,
                conversation_id=conversation_id, exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        # Stage 2: type enumeration (7 cues × per_cue_top_k) — additive
        type_context_section = _build_context_section(all_segments)
        type_prompt = TYPE_ENUMERATED_PROMPT.format(
            question=question, context_section=type_context_section
        )
        type_output = self.llm_call(type_prompt)
        type_cues = _parse_cues(type_output)[:7]

        for cue in type_cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=self.per_cue_top_k,
                conversation_id=conversation_id, exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return TypeEnumResult(
            segments=all_segments,
            metadata={
                "variant": "v2f_plus_types",
                "v2f_output": v2f_output,
                "type_output": type_output,
                "v2f_cues": v2f_cues,
                "type_cues": type_cues,
                "per_cue_top_k": self.per_cue_top_k,
            },
        )


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
VARIANT_FACTORIES = {
    "type_enumerated": lambda store: TypeEnumeratedVariant(store, per_cue_top_k=3),
    "type_enumerated_selective": lambda store: TypeEnumeratedSelectiveVariant(
        store, per_cue_top_k=5
    ),
    "v2f_plus_types": lambda store: V2fPlusTypesVariant(store, per_cue_top_k=3),
}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
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


def load_dataset(ds_name: str) -> tuple[SegmentStore, list[dict]]:
    cfg = DATASETS[ds_name]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        questions = json.load(f)
    if cfg["filter"]:
        questions = [q for q in questions if cfg["filter"](q)]
    if cfg["max_questions"]:
        questions = questions[: cfg["max_questions"]]
    # Ensure each question has question_index
    for i, q in enumerate(questions):
        if "question_index" not in q:
            q["question_index"] = i
    return store, questions


# ---------------------------------------------------------------------------
# Fair-backfill evaluation (matches fair_backfill_eval.py semantics)
# ---------------------------------------------------------------------------
def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill_evaluate(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    source_ids: set[int],
    budget: int,
) -> tuple[float, float]:
    """Return (baseline_recall, arch_recall) with fair backfill at K=budget."""
    seen: set[int] = set()
    arch_unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            arch_unique.append(s)
            seen.add(s.index)

    arch_at_K = arch_unique[:budget]
    arch_indices = {s.index for s in arch_at_K}

    if len(arch_at_K) < budget:
        backfill = [s for s in cosine_segments if s.index not in arch_indices]
        needed = budget - len(arch_at_K)
        arch_at_K = arch_at_K + backfill[:needed]

    arch_at_K = arch_at_K[:budget]
    baseline_at_K = cosine_segments[:budget]

    arch_ids = {s.turn_id for s in arch_at_K}
    baseline_ids = {s.turn_id for s in baseline_at_K}

    return (
        compute_recall(baseline_ids, source_ids),
        compute_recall(arch_ids, source_ids),
    )


def evaluate_question(arch: TypeEnumBase, question: dict) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedupe arch segments preserving order
    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            arch_segments.append(seg)
            seen.add(seg.index)

    # Cosine top-max(BUDGETS)
    query_emb = arch.embed_text(q_text)
    max_K = max(BUDGETS)
    cosine_result = arch.store.search(
        query_emb, top_k=max_K, conversation_id=conv_id
    )
    cosine_segments = list(cosine_result.segments)

    row = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "total_arch_retrieved": len(arch_segments),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "metadata": result.metadata,
    }

    for K in BUDGETS:
        b_rec, a_rec = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)

    return row


def summarize(results: list[dict], arch_name: str, dataset: str) -> dict:
    n = len(results)
    if n == 0:
        return {"arch": arch_name, "dataset": dataset, "n": 0}

    summary: dict = {"arch": arch_name, "dataset": dataset, "n": n}
    for K in BUDGETS:
        b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in results]
        a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in results]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses
        summary[f"baseline_r@{K}"] = round(b_mean, 4)
        summary[f"arch_r@{K}"] = round(a_mean, 4)
        summary[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
        summary[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"

    summary["avg_total_retrieved"] = round(
        sum(r["total_arch_retrieved"] for r in results) / n, 1
    )
    summary["avg_llm_calls"] = round(
        sum(r["llm_calls"] for r in results) / n, 2
    )
    summary["avg_embed_calls"] = round(
        sum(r["embed_calls"] for r in results) / n, 2
    )
    summary["avg_time_s"] = round(
        sum(r["time_s"] for r in results) / n, 2
    )
    return summary


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    out: dict[str, dict] = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry: dict = {"n": n}
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
# Runner
# ---------------------------------------------------------------------------
def run_variant_on_dataset(
    variant_name: str,
    ds_name: str,
    force: bool = False,
) -> dict:
    out_path = RESULTS_DIR / f"type_enum_{variant_name}_{ds_name}.json"
    if out_path.exists() and not force:
        print(f"\n[SKIP] {variant_name} on {ds_name} — {out_path} exists")
        with open(out_path) as f:
            saved = json.load(f)
        return saved

    store, questions = load_dataset(ds_name)
    print(f"\n{'=' * 78}")
    print(f"VARIANT={variant_name} | DATASET={ds_name} | "
          f"n={len(questions)} questions, {len(store.segments)} segments")
    print(f"{'=' * 78}")

    arch = VARIANT_FACTORIES[variant_name](store)

    results: list[dict] = []
    for i, q in enumerate(questions):
        q_short = q["question"][:55]
        print(f"  [{i + 1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
              flush=True)
        try:
            row = evaluate_question(arch, q)
            results.append(row)

            # Save incrementally per-question
            with open(out_path, "w") as f:
                json.dump({
                    "variant": variant_name,
                    "dataset": ds_name,
                    "results": results,
                }, f, indent=2, default=str)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()

        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()

    summary = summarize(results, variant_name, ds_name)
    by_cat = summarize_by_category(results)

    saved = {
        "variant": variant_name,
        "dataset": ds_name,
        "summary": summary,
        "category_breakdown": by_cat,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(saved, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    print(f"\n--- {variant_name} on {ds_name} ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: baseline={summary[f'baseline_r@{K}']:.3f} "
            f"arch={summary[f'arch_r@{K}']:.3f} "
            f"delta={summary[f'delta_r@{K}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    print(f"  avg retrieved={summary['avg_total_retrieved']:.0f} "
          f"llm={summary['avg_llm_calls']:.1f} "
          f"embed={summary['avg_embed_calls']:.1f}")
    print(f"\n  Per-category:")
    for cat, c in by_cat.items():
        print(
            f"    {cat:28s} (n={c['n']}): "
            f"r@20 d={c['delta_r@20']:+.3f} W/T/L={c['W/T/L_r@20']} | "
            f"r@50 d={c['delta_r@50']:+.3f} W/T/L={c['W/T/L_r@50']}"
        )

    return saved


# ---------------------------------------------------------------------------
# Comparison vs v2f baselines (reads existing fair-backfill results)
# ---------------------------------------------------------------------------
def load_v2f_baseline(ds_name: str) -> dict | None:
    path = RESULTS_DIR / f"fairbackfill_meta_v2f_{ds_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compare_to_v2f(all_runs: dict) -> dict:
    """Build a comparison table: for each variant, each dataset, each category,
    compare to v2f's baselines.
    """
    comparisons: dict = {}
    for variant_name, ds_map in all_runs.items():
        comparisons[variant_name] = {}
        for ds_name, run in ds_map.items():
            v2f = load_v2f_baseline(ds_name)
            if not v2f:
                continue
            v2f_summary = v2f["summary"]
            v2f_cats = v2f["category_breakdown"]
            variant_summary = run["summary"]
            variant_cats = run["category_breakdown"]

            ds_entry: dict = {
                "overall": {},
                "per_category": {},
            }
            for K in BUDGETS:
                ds_entry["overall"][f"r@{K}"] = {
                    "v2f_arch": v2f_summary[f"arch_r@{K}"],
                    "variant_arch": variant_summary[f"arch_r@{K}"],
                    "delta_variant_vs_v2f": round(
                        variant_summary[f"arch_r@{K}"]
                        - v2f_summary[f"arch_r@{K}"], 4
                    ),
                    "baseline": variant_summary[f"baseline_r@{K}"],
                }
            # Per-category
            all_cats = sorted(set(v2f_cats.keys()) | set(variant_cats.keys()))
            for cat in all_cats:
                v2f_c = v2f_cats.get(cat, {})
                var_c = variant_cats.get(cat, {})
                cat_entry: dict = {"n": var_c.get("n", v2f_c.get("n"))}
                for K in BUDGETS:
                    cat_entry[f"r@{K}"] = {
                        "baseline": var_c.get(f"baseline_r@{K}"),
                        "v2f_arch": v2f_c.get(f"arch_r@{K}"),
                        "variant_arch": var_c.get(f"arch_r@{K}"),
                        "delta_variant_vs_v2f": (
                            round(var_c[f"arch_r@{K}"] - v2f_c[f"arch_r@{K}"], 4)
                            if f"arch_r@{K}" in var_c and f"arch_r@{K}" in v2f_c
                            else None
                        ),
                    }
                ds_entry["per_category"][cat] = cat_entry
            comparisons[variant_name][ds_name] = ds_entry
    return comparisons


def print_comparison_table(comparisons: dict) -> None:
    print("\n" + "=" * 100)
    print("TYPE-ENUMERATED VARIANTS vs v2f baseline (fair backfill)")
    print("=" * 100)

    for variant_name, ds_map in comparisons.items():
        for ds_name, entry in ds_map.items():
            print(f"\n--- {variant_name} on {ds_name} ---")
            ov = entry["overall"]
            for K in BUDGETS:
                o = ov[f"r@{K}"]
                print(
                    f"  r@{K}: baseline={o['baseline']:.3f} "
                    f"v2f={o['v2f_arch']:.3f} variant={o['variant_arch']:.3f} "
                    f"delta_v_vs_v2f={o['delta_variant_vs_v2f']:+.4f}"
                )
            print(f"  Per-category (r@20 | r@50 | delta vs v2f @20 | @50):")
            print(f"    {'Category':<28s} {'n':>3s} "
                  f"{'base20':>7s} {'v2f20':>7s} {'var20':>7s} {'d20':>7s}  "
                  f"{'base50':>7s} {'v2f50':>7s} {'var50':>7s} {'d50':>7s}")
            for cat, cd in entry["per_category"].items():
                r20 = cd["r@20"]
                r50 = cd["r@50"]
                def _f(x):
                    return f"{x:>7.3f}" if isinstance(x, (int, float)) else f"{'-':>7s}"

                d20 = r20["delta_variant_vs_v2f"]
                d50 = r50["delta_variant_vs_v2f"]
                d20_s = f"{d20:>+7.3f}" if isinstance(d20, (int, float)) else f"{'-':>7s}"
                d50_s = f"{d50:>+7.3f}" if isinstance(d50, (int, float)) else f"{'-':>7s}"
                print(
                    f"    {cat:<28s} {cd['n']:>3d} "
                    f"{_f(r20['baseline'])} {_f(r20['v2f_arch'])} {_f(r20['variant_arch'])} {d20_s}  "
                    f"{_f(r50['baseline'])} {_f(r50['v2f_arch'])} {_f(r50['variant_arch'])} {d50_s}"
                )

    # Key test categories summary
    KEY_CATS = {
        "puzzle_16q": ["logic_constraint"],
        "synthetic_19q": ["completeness", "procedural"],
        "advanced_23q": ["quantitative_aggregation"],
    }
    print("\n" + "=" * 100)
    print("KEY-CATEGORIES FOCUS (target: logic_constraint, completeness, "
          "procedural, quantitative_aggregation)")
    print("=" * 100)
    for variant_name, ds_map in comparisons.items():
        print(f"\n--- {variant_name} ---")
        for ds_name, cats in KEY_CATS.items():
            if ds_name not in ds_map:
                continue
            ds_entry = ds_map[ds_name]
            for cat in cats:
                cd = ds_entry["per_category"].get(cat)
                if not cd:
                    continue
                r20 = cd["r@20"]
                r50 = cd["r@50"]
                d20 = r20.get("delta_variant_vs_v2f")
                d50 = r50.get("delta_variant_vs_v2f")
                d20_s = f"{d20:+.4f}" if isinstance(d20, (int, float)) else "n/a"
                d50_s = f"{d50:+.4f}" if isinstance(d50, (int, float)) else "n/a"
                print(
                    f"  {ds_name}/{cat} (n={cd['n']}): "
                    f"base@20={r20['baseline']:.3f} v2f@20={r20['v2f_arch']:.3f} "
                    f"variant@20={r20['variant_arch']:.3f} (d vs v2f: {d20_s}) | "
                    f"base@50={r50['baseline']:.3f} v2f@50={r50['v2f_arch']:.3f} "
                    f"variant@50={r50['variant_arch']:.3f} (d vs v2f: {d50_s})"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Test constraint-type-enumerated cue generator variants"
    )
    parser.add_argument(
        "--variant", type=str, default="all",
        help="One of: type_enumerated, type_enumerated_selective, "
             "v2f_plus_types, all",
    )
    parser.add_argument("--list", action="store_true",
                        help="List available variants")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated list or 'all'")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("Available variants:")
        for name in VARIANT_FACTORIES:
            print(f"  {name}")
        print("\nAvailable datasets:")
        for name in DATASETS:
            print(f"  {name}")
        return

    if args.variant == "all":
        variant_names = list(VARIANT_FACTORIES.keys())
    else:
        variant_names = [v.strip() for v in args.variant.split(",")]

    if args.datasets == "all":
        ds_names = list(DATASETS.keys())
    else:
        ds_names = [d.strip() for d in args.datasets.split(",")]

    # all_runs[variant_name][ds_name] = saved_dict
    all_runs: dict[str, dict[str, dict]] = {}

    for variant_name in variant_names:
        if variant_name not in VARIANT_FACTORIES:
            print(f"Unknown variant: {variant_name}")
            continue
        all_runs[variant_name] = {}
        for ds_name in ds_names:
            if ds_name not in DATASETS:
                print(f"Unknown dataset: {ds_name}")
                continue
            saved = run_variant_on_dataset(
                variant_name, ds_name, force=args.force
            )
            all_runs[variant_name][ds_name] = saved

    # Build comparisons vs v2f and print
    comparisons = compare_to_v2f(all_runs)

    # Save combined summary
    summary_path = RESULTS_DIR / "type_enum_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "runs": {
                    v: {ds: s.get("summary", {}) for ds, s in m.items()}
                    for v, m in all_runs.items()
                },
                "category_breakdown": {
                    v: {ds: s.get("category_breakdown", {}) for ds, s in m.items()}
                    for v, m in all_runs.items()
                },
                "comparisons_vs_v2f": comparisons,
            },
            f, indent=2, default=str,
        )
    print(f"\nSaved combined summary: {summary_path}")

    print_comparison_table(comparisons)


if __name__ == "__main__":
    main()
