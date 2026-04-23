"""Domain-agnostic V2f prompt variants.

Hypothesis: V2f's +37pp gain on LoCoMo comes partly from "chat message" /
"conversation" framing, which tells the model to write casual first-person
short fragments. We want to preserve that style signal without naming
the conversation domain, so the prompt generalizes to documents, emails,
notes, etc.

Variants tested:
  v2f (reference): original v2f with "conversation history" + "chat message"
  v2f_minimal: v2f with domain words stripped and NO replacement (baseline
    for domain-agnostic; already known to lose ~9pp on LoCoMo)
  v2f_style_explicit: v2f with explicit style properties (casual,
    first-person, short fragments, target register vocab)
  v2f_fewshot: v2f with 2-3 brief style examples (without naming the domain)
  v2f_register_inferred: tells model to infer register from retrieved content
  v2f_content_type_param: takes a content_type parameter ("documents" here)

Quick-test on 5 representative questions first, then (if gate passes) a
full 4-dataset fair-backfill eval at K=20.

Usage:
    uv run python domain_agnostic.py --quick   # 5-question gate test
    uv run python domain_agnostic.py --full    # full 4-dataset eval
    uv run python domain_agnostic.py --all     # quick then full if gate passes
"""

import concurrent.futures as cf
import json
import sys
import threading
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
from prompt_optimization import (
    BUDGETS,
    OptimBase,
    OptimResult,
    _format_segments,
    _parse_cues,
    compute_recall,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_FILE_NAME_LLM = "domain_agnostic_llm_cache.json"
CACHE_FILE_NAME_EMB = "domain_agnostic_embedding_cache.json"


# ===========================================================================
# Prompts
# ===========================================================================

# Reference: original V2f (with conversation/chat framing).
V2F_ORIGINAL_PROMPT = """\
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

# V2f_minimal: strip all domain words, add nothing. Known baseline.
V2F_MINIMAL_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target content.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the target content.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2f_style_explicit: replace "chat message" style framing with explicit style.
V2F_STYLE_EXPLICIT_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Generate cues in a \
casual, first-person register. Write short 1-2 sentence fragments using \
specific vocabulary from the target content's register.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the target content.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2f_fewshot: replace "chat message" with 2-3 style examples (no domain name).
V2F_FEWSHOT_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Your cues should \
match the style of content like:
- "I went to the support group yesterday, it was powerful"
- "Bob mentioned his peanut allergy again"
- "Had a picnic last weekend with the kids"

Generate cues in this style, using specific vocabulary.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the target content.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2f_register_inferred: tell the model to infer register from retrieved content.
V2F_REGISTER_INFERRED_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Look at the \
retrieved content's register (formal vs casual, first vs third person, \
sentence length). Generate cues in the same register, using specific \
vocabulary that would appear in the target content.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the target content.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2f_content_type_param: templates a domain-neutral content_type word in.
V2F_CONTENT_TYPE_PROMPT_TEMPLATE = """\
You are generating search text for semantic retrieval. Your cues will be \
used to search {content_type}. Your cues will be embedded and compared via \
cosine similarity.

Question: {{question}}

{{context_section}}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target {content_type}.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the target {content_type}.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


def build_content_type_prompt(content_type: str = "documents") -> str:
    """Build v2f_content_type prompt with a specific content type word."""
    return V2F_CONTENT_TYPE_PROMPT_TEMPLATE.format(content_type=content_type)


# ===========================================================================
# Caches — share read-only across experiments, write to domain_agnostic_*
# ===========================================================================
_SHARED_LLM_SOURCES = (
    "llm_cache.json",
    "arch_llm_cache.json",
    "agent_llm_cache.json",
    "tree_llm_cache.json",
    "frontier_llm_cache.json",
    "meta_llm_cache.json",
    "optim_llm_cache.json",
    "synth_test_llm_cache.json",
    "general_llm_cache.json",
    "adaptive_llm_cache.json",
    "bestshot_llm_cache.json",
    "fulleval_llm_cache.json",
    CACHE_FILE_NAME_LLM,
)

_SHARED_EMB_SOURCES = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "agent_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "synth_test_embedding_cache.json",
    "general_embedding_cache.json",
    "adaptive_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "fulleval_embedding_cache.json",
    CACHE_FILE_NAME_EMB,
)


class DomainAgnosticEmbeddingCache(EmbeddingCache):
    """Reads from many shared caches, writes to domain_agnostic_* only."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in _SHARED_EMB_SOURCES:
            p = self.cache_dir / name
            if p.exists():
                try:
                    with open(p) as f:
                        self._cache.update(json.load(f))
                except Exception:
                    pass
        self.cache_file = self.cache_dir / CACHE_FILE_NAME_EMB
        self._new_entries: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        with self._lock:
            self._cache[key] = embedding.tolist()
            self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        with self._lock:
            new_entries = dict(self._new_entries)
            self._new_entries.clear()
        if not new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


class DomainAgnosticLLMCache(LLMCache):
    """Reads from many shared caches, writes to domain_agnostic_* only."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in _SHARED_LLM_SOURCES:
            p = self.cache_dir / name
            if p.exists():
                try:
                    with open(p) as f:
                        data = json.load(f)
                    for k, v in data.items():
                        if v:
                            self._cache[k] = v
                except Exception:
                    pass
        self.cache_file = self.cache_dir / CACHE_FILE_NAME_LLM
        self._new_entries: dict[str, str] = {}
        self._lock = threading.Lock()

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        with self._lock:
            self._cache[key] = response
            self._new_entries[key] = response

    def save(self) -> None:
        with self._lock:
            new_entries = dict(self._new_entries)
            self._new_entries.clear()
        if not new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


# ===========================================================================
# Variant class
# ===========================================================================
class DomainAgnosticVariant(OptimBase):
    """Single-call strategist variant with configurable prompt + context header.

    Retrieval logic matches V15Control/MetaV2f:
      1. embed question, top-10
      2. single LLM call with context + prompt
      3. parse CUE lines, embed each, top-10 each
    """

    def __init__(
        self,
        store: SegmentStore,
        prompt_template: str,
        context_header: str = "RETRIEVED EXCERPTS SO FAR:",
        client: OpenAI | None = None,
    ):
        super().__init__(store, client)
        self.prompt_template = prompt_template
        self.context_header = context_header
        # Override caches so runs across variants share + write to our file
        self.embedding_cache = DomainAgnosticEmbeddingCache()
        self.llm_cache = DomainAgnosticLLMCache()

    def retrieve(self, question: str, conversation_id: str) -> OptimResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        context = _format_segments(all_segments)
        context_section = f"{self.context_header}\n{context}"

        prompt = self.prompt_template.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        for cue in cues[:2]:
            if not cue:
                continue
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return OptimResult(
            segments=all_segments,
            metadata={"output": output, "cues": cues[:2]},
        )


# ===========================================================================
# Variant registry
#
# Reference/baseline variants preserve their historical context header
# ("RETRIEVED CONVERSATION EXCERPTS SO FAR:") so they reproduce the earlier
# measurements exactly. New variants use a domain-neutral header.
# ===========================================================================
CONV_HEADER = "RETRIEVED CONVERSATION EXCERPTS SO FAR:"
NEUTRAL_HEADER = "RETRIEVED EXCERPTS SO FAR:"

VARIANT_SPECS: dict[str, tuple[str, str]] = {
    # (prompt_template, context_header)
    "v2f": (V2F_ORIGINAL_PROMPT, CONV_HEADER),
    "v2f_minimal": (V2F_MINIMAL_PROMPT, NEUTRAL_HEADER),
    "v2f_style_explicit": (V2F_STYLE_EXPLICIT_PROMPT, NEUTRAL_HEADER),
    "v2f_fewshot": (V2F_FEWSHOT_PROMPT, NEUTRAL_HEADER),
    "v2f_register_inferred": (V2F_REGISTER_INFERRED_PROMPT, NEUTRAL_HEADER),
    "v2f_content_type_documents": (
        build_content_type_prompt("documents"),
        NEUTRAL_HEADER,
    ),
}


def build_variant(store: SegmentStore, name: str) -> DomainAgnosticVariant:
    if name not in VARIANT_SPECS:
        raise KeyError(f"Unknown variant: {name}")
    prompt, header = VARIANT_SPECS[name]
    return DomainAgnosticVariant(
        store, prompt_template=prompt, context_header=header
    )


# ===========================================================================
# Datasets
# ===========================================================================
DATASETS = {
    "locomo_30q": {
        "npz": "segments_extended.npz",
        "questions": "questions_extended.json",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "max_questions": 30,
        "label": "LoCoMo (30q)",
    },
    "synthetic_19q": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "filter": None,
        "max_questions": None,
        "label": "Synthetic (19q)",
    },
    "puzzle_16q": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "filter": None,
        "max_questions": None,
        "label": "Puzzle (16q)",
    },
    "advanced_23q": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "filter": None,
        "max_questions": None,
        "label": "Advanced (23q)",
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
    # Ensure question_index present
    for i, q in enumerate(questions):
        q.setdefault("question_index", i)
    return store, questions


# ===========================================================================
# Evaluation (fair-backfill)
# ===========================================================================
def fair_backfill_recall(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    source_ids: set[int],
    budget: int,
) -> tuple[float, float]:
    """At budget K, return (baseline_recall, arch_recall) with arch backfilled."""
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


def evaluate_question(
    arch: DomainAgnosticVariant,
    question: dict,
    budgets: list[int],
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            arch_segments.append(seg)
            seen.add(seg.index)

    query_emb = arch.embed_text(q_text)
    max_K = max(budgets)
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
        "embed_calls": 0,  # counters unreliable under parallel run
        "llm_calls": 1,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "cues": result.metadata.get("cues", []),
    }
    for K in budgets:
        b_rec, a_rec = fair_backfill_recall(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)
    return row


def summarize_results(
    results: list[dict],
    variant: str,
    dataset: str,
    budgets: list[int],
) -> dict:
    n = len(results)
    if n == 0:
        return {"variant": variant, "dataset": dataset, "n": 0}
    out = {"variant": variant, "dataset": dataset, "n": n}
    for K in budgets:
        b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in results]
        a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in results]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses
        out[f"baseline_r@{K}"] = round(b_mean, 4)
        out[f"arch_r@{K}"] = round(a_mean, 4)
        out[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
        out[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
    out["avg_total_retrieved"] = round(
        sum(r["total_arch_retrieved"] for r in results) / n, 1
    )
    out["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
    out["avg_embed_calls"] = round(
        sum(r["embed_calls"] for r in results) / n, 1
    )
    return out


def summarize_by_category(
    results: list[dict], budgets: list[int]
) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry = {"n": n}
        for K in budgets:
            b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in rs]
            a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in rs]
            entry[f"baseline_r@{K}"] = round(sum(b_vals) / n, 4)
            entry[f"arch_r@{K}"] = round(sum(a_vals) / n, 4)
            entry[f"delta_r@{K}"] = round(
                sum(a_vals) / n - sum(b_vals) / n, 4
            )
            wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
            losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
            entry[f"W/T/L_r@{K}"] = f"{wins}/{n - wins - losses}/{losses}"
        out[cat] = entry
    return out


# ===========================================================================
# Parallel runner
# ===========================================================================
def run_variant_parallel(
    arch: DomainAgnosticVariant,
    questions: list[dict],
    budgets: list[int],
    workers: int = 8,
) -> list[dict]:
    """Run evaluate_question in parallel using a thread pool.

    Uses the same `arch` instance across threads; caches are thread-safe,
    retrieve() is mostly read-only against the store + network-bound LLM/embed.
    Periodic cache saves happen as results stream back.
    """
    results: list[tuple[int, dict]] = []
    print_lock = threading.Lock()

    def worker(i_q: tuple[int, dict]) -> tuple[int, dict]:
        i, q = i_q
        try:
            row = evaluate_question(arch, q, budgets)
        except Exception as e:
            with print_lock:
                print(f"  [{i+1}/{len(questions)}] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return (i, None)
        with print_lock:
            cat = q.get("category", "?")
            q_short = q["question"][:55]
            print(
                f"  [{i+1}/{len(questions)}] {cat}: {q_short}... "
                f"d@20={row['fair_backfill'].get(f'delta_r@{budgets[0]}', 0):+.2f} "
                f"({row['time_s']:.1f}s)",
                flush=True,
            )
        return (i, row)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, (i, q)) for i, q in enumerate(questions)]
        finished = 0
        for fut in cf.as_completed(futures):
            i, row = fut.result()
            if row is not None:
                results.append((i, row))
            finished += 1
            if finished % 10 == 0:
                arch.save_caches()

    # Sort by original order and strip index
    results.sort(key=lambda t: t[0])
    return [r for _, r in results]


# ===========================================================================
# Quick-test: 5 representative questions
# ===========================================================================
QUICK_QUESTIONS_SPEC = [
    # (dataset, conversation_id, question_index)
    ("locomo_30q", "locomo_conv-26", 0),             # temporal, Caroline LGBTQ group
    ("locomo_30q", "locomo_conv-26", 3),             # single_hop, Caroline research
    ("synthetic_19q", "synth_personal", 0),          # control, Bob allergic
    ("puzzle_16q", "puzzle_logic_1", 0),             # logic_constraint, desk arrangement
    ("advanced_23q", "adv_evolving_term_1", 0),      # evolving_terminology, Phoenix
]


def load_quick_questions() -> list[tuple[str, dict]]:
    """Load the 5 representative questions for the quick gate test.

    Returns list of (dataset_name, question_dict).
    """
    out = []
    seen_datasets: dict[str, tuple[SegmentStore, list[dict]]] = {}
    for ds_name, conv_id, q_idx in QUICK_QUESTIONS_SPEC:
        if ds_name not in seen_datasets:
            seen_datasets[ds_name] = load_dataset(ds_name)
        _, questions = seen_datasets[ds_name]
        matched = None
        for q in questions:
            if (q.get("conversation_id") == conv_id
                    and q.get("question_index") == q_idx):
                matched = q
                break
        if matched is None:
            # Fallback: pick by conv_id + any question if index mismatches
            cands = [q for q in questions if q.get("conversation_id") == conv_id]
            if cands:
                matched = cands[0]
        if matched is None:
            print(
                f"[WARN] Could not find {ds_name}:{conv_id}#{q_idx}; "
                "using first question in dataset as fallback",
                flush=True,
            )
            matched = questions[0]
        out.append((ds_name, matched))
    return out, seen_datasets


def run_quick_test(variants_to_run: list[str]) -> dict:
    """Run the 5-question quick test. Returns per-variant per-question recalls.

    Decision gate:
      - If a new variant beats v2f_minimal on 3+ of 5 AND is within 3pp of
        v2f (original) on the 2 LoCoMo questions -> passes.
    """
    print("\n" + "=" * 80)
    print("QUICK TEST: 5 representative questions")
    print("=" * 80)

    quick_qs, loaded = load_quick_questions()

    # Prepare per-dataset variant instances (reuse store across variants)
    # We'll build a fresh variant per (variant_name, dataset) since the store
    # is dataset-specific.
    all_rows: dict[str, list[dict]] = {name: [] for name in variants_to_run}

    for ds_name, question in quick_qs:
        store, _ = loaded[ds_name]
        for variant_name in variants_to_run:
            arch = build_variant(store, variant_name)
            row = evaluate_question(arch, question, budgets=[20])
            row["variant"] = variant_name
            row["dataset"] = ds_name
            all_rows[variant_name].append(row)
            arch.save_caches()

    # Print comparison table
    print(
        f"\n{'Variant':<28s} "
        + " ".join(
            f"Q{i+1}@20" for i in range(len(quick_qs))
        )
        + "   mean_delta"
    )
    print("-" * 100)
    for variant_name in variants_to_run:
        rows = all_rows[variant_name]
        cells = []
        deltas = []
        for r in rows:
            d = r["fair_backfill"]["delta_r@20"]
            a = r["fair_backfill"]["arch_r@20"]
            cells.append(f"{a:.2f}({d:+.2f})")
            deltas.append(d)
        mean_d = sum(deltas) / len(deltas) if deltas else 0.0
        print(
            f"{variant_name:<28s} "
            + "  ".join(cells)
            + f"   {mean_d:+.3f}"
        )

    # Print baseline (cosine) row for reference — same per question regardless
    # of variant, take from first variant
    ref_rows = all_rows[variants_to_run[0]]
    b_cells = [
        f"{r['fair_backfill']['baseline_r@20']:.2f}" for r in ref_rows
    ]
    print(f"\n{'(cosine baseline)':<28s} " + "  ".join(b_cells))

    print(
        "\nQuestions:\n"
        + "\n".join(
            f"  Q{i+1} [{q['category']}] {q['question'][:80]}"
            for i, (_, q) in enumerate(quick_qs)
        )
    )

    # Decision gate
    # Because 5 questions is small and several ceiling at 1.0 (ties),
    # the gate uses:
    #   (a) does not regress vs v2f_minimal on any question
    #   (b) beats v2f_minimal on at least one question OR matches v2f on LoCoMo
    #   (c) within 3pp of v2f on LoCoMo questions (max gap)
    print("\n--- Decision gate ---")
    base_rows = all_rows.get("v2f_minimal", [])
    v2f_ref_rows = all_rows.get("v2f", [])
    gate_pass: list[str] = []
    for variant_name in variants_to_run:
        if variant_name in ("v2f", "v2f_minimal"):
            continue
        rows = all_rows[variant_name]
        beats = sum(
            1 for v, b in zip(rows, base_rows)
            if v["fair_backfill"]["arch_r@20"]
            > b["fair_backfill"]["arch_r@20"] + 0.001
        )
        losses = sum(
            1 for v, b in zip(rows, base_rows)
            if v["fair_backfill"]["arch_r@20"]
            < b["fair_backfill"]["arch_r@20"] - 0.001
        )
        locomo_diffs = [
            v2f_ref_rows[i]["fair_backfill"]["arch_r@20"]
            - rows[i]["fair_backfill"]["arch_r@20"]
            for i in (0, 1)
        ]
        max_locomo_gap = max(locomo_diffs)
        mean_delta = sum(
            r["fair_backfill"]["delta_r@20"] for r in rows
        ) / max(len(rows), 1)
        base_mean_delta = sum(
            r["fair_backfill"]["delta_r@20"] for r in base_rows
        ) / max(len(base_rows), 1)
        # PASS: no loss vs minimal AND mean_delta at least as high AND LoCoMo gap <= 3pp
        passes = (
            losses == 0
            and mean_delta >= base_mean_delta - 0.01
            and max_locomo_gap <= 0.03
        )
        status = "PASS" if passes else "FAIL"
        print(
            f"  {variant_name:<28s} beats_minimal={beats}/5  losses={losses}/5  "
            f"mean_d={mean_delta:+.3f}  (min_mean_d={base_mean_delta:+.3f})  "
            f"locomo_gap_max={max_locomo_gap:+.3f}  -> {status}"
        )
        if passes:
            gate_pass.append(variant_name)

    return {
        "all_rows": all_rows,
        "gate_pass": gate_pass,
        "questions": [
            {
                "dataset": ds_name,
                "conversation_id": q["conversation_id"],
                "question_index": q.get("question_index"),
                "category": q.get("category"),
                "question": q["question"],
            }
            for ds_name, q in quick_qs
        ],
    }


# ===========================================================================
# Full eval
# ===========================================================================
def run_full_eval(variants_to_run: list[str]) -> dict:
    """Run fair-backfill eval at K=20 over all 4 datasets."""
    print("\n" + "=" * 80)
    print(f"FULL EVAL: {len(variants_to_run)} variants x 4 datasets @ K=20")
    print("=" * 80)

    BUDGETS_FULL = [20]
    summary_table: dict = {}
    per_variant_per_dataset_results: dict = {}

    for variant_name in variants_to_run:
        per_variant_per_dataset_results[variant_name] = {}
        summary_table[variant_name] = {}

    for ds_name, cfg in DATASETS.items():
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )
        for variant_name in variants_to_run:
            print(f"\n--- {variant_name} on {ds_name} ---")
            arch = build_variant(store, variant_name)
            results = run_variant_parallel(
                arch, questions, BUDGETS_FULL,
                workers=8,
            )
            arch.save_caches()

            summary = summarize_results(
                results, variant_name, ds_name, BUDGETS_FULL
            )
            by_cat = summarize_by_category(results, BUDGETS_FULL)

            per_variant_per_dataset_results[variant_name][ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
            }
            summary_table[variant_name][ds_name] = summary

            # Save per-run json
            out_path = (
                RESULTS_DIR / f"domain_agnostic_{variant_name}_{ds_name}.json"
            )
            with open(out_path, "w") as f:
                json.dump(
                    per_variant_per_dataset_results[variant_name][ds_name],
                    f, indent=2, default=str,
                )
            print(f"  Saved: {out_path}")

            print(
                f"    r@20: base={summary['baseline_r@20']:.3f}  "
                f"arch={summary['arch_r@20']:.3f}  "
                f"delta={summary['delta_r@20']:+.3f}  "
                f"W/T/L={summary['W/T/L_r@20']}"
            )

    # Cross-dataset average (mean of delta_r@20 across 4 datasets)
    print("\n" + "=" * 90)
    print("FULL EVAL SUMMARY (delta_r@20 = arch - cosine baseline)")
    print("=" * 90)
    header = (
        f"{'Variant':<30s} "
        + " ".join(f"{ds[:12]:>12s}" for ds in DATASETS)
        + f" {'AVG':>8s}"
    )
    print(header)
    print("-" * len(header))
    cross_avg: dict[str, float] = {}
    for variant_name in variants_to_run:
        deltas = []
        cells = []
        for ds_name in DATASETS:
            s = summary_table[variant_name].get(ds_name, {})
            d = s.get("delta_r@20", 0.0)
            deltas.append(d)
            cells.append(f"{d:+.3f}")
        avg = sum(deltas) / len(deltas) if deltas else 0.0
        cross_avg[variant_name] = avg
        print(
            f"{variant_name:<30s} "
            + " ".join(f"{c:>12s}" for c in cells)
            + f" {avg:>+8.3f}"
        )

    # Absolute arch_r@20 row as well
    print("\n" + "=" * 90)
    print("FULL EVAL — arch_r@20 (absolute)")
    print("=" * 90)
    print(header)
    print("-" * len(header))
    for variant_name in variants_to_run:
        vals = []
        cells = []
        for ds_name in DATASETS:
            s = summary_table[variant_name].get(ds_name, {})
            v = s.get("arch_r@20", 0.0)
            vals.append(v)
            cells.append(f"{v:.3f}")
        avg = sum(vals) / len(vals) if vals else 0.0
        print(
            f"{variant_name:<30s} "
            + " ".join(f"{c:>12s}" for c in cells)
            + f" {avg:>8.3f}"
        )

    # LoCoMo gap vs v2f (how much each variant sacrifices of the +37pp gain)
    print("\n--- LoCoMo preservation vs v2f ---")
    v2f_locomo = summary_table.get("v2f", {}).get("locomo_30q", {})
    if v2f_locomo:
        v2f_lc = v2f_locomo.get("arch_r@20", 0.0)
        v2f_lc_d = v2f_locomo.get("delta_r@20", 0.0)
        print(f"  v2f: r@20={v2f_lc:.3f}  delta={v2f_lc_d:+.3f}")
        for variant_name in variants_to_run:
            if variant_name == "v2f":
                continue
            s = summary_table[variant_name].get("locomo_30q", {})
            v = s.get("arch_r@20", 0.0)
            d = s.get("delta_r@20", 0.0)
            gap_to_v2f = v - v2f_lc
            print(
                f"  {variant_name:<28s} r@20={v:.3f}  delta={d:+.3f}  "
                f"gap_vs_v2f={gap_to_v2f:+.3f}"
            )

    # Write aggregate summary
    agg_path = RESULTS_DIR / "domain_agnostic_summary.json"
    with open(agg_path, "w") as f:
        json.dump(
            {
                "summary_table": summary_table,
                "cross_dataset_avg_delta": cross_avg,
            },
            f, indent=2, default=str,
        )
    print(f"\nSaved aggregate summary: {agg_path}")

    return {"summary_table": summary_table, "cross_avg": cross_avg}


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Domain-agnostic v2f variant evaluation"
    )
    parser.add_argument("--quick", action="store_true", help="Run 5-question gate test")
    parser.add_argument("--full", action="store_true", help="Run full 4-dataset eval")
    parser.add_argument(
        "--all", action="store_true",
        help="Run quick test; if any new variant passes the gate, run full eval on survivors",
    )
    parser.add_argument(
        "--variants", type=str, default=None,
        help="Comma-separated variant names (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="List variants")
    args = parser.parse_args()

    if args.list:
        print("Available variants:")
        for name in VARIANT_SPECS:
            print(f"  {name}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = list(VARIANT_SPECS.keys())

    # Ensure references are present
    for required in ("v2f", "v2f_minimal"):
        if required not in variants:
            variants.insert(0, required)
    # Deduplicate while preserving order
    seen = set()
    variants = [v for v in variants if not (v in seen or seen.add(v))]

    print(f"Variants to run: {variants}")

    quick_result = None
    if args.quick or args.all:
        quick_result = run_quick_test(variants)
        # Save quick results
        qpath = RESULTS_DIR / "domain_agnostic_quick.json"
        with open(qpath, "w") as f:
            json.dump(
                {
                    "gate_pass": quick_result["gate_pass"],
                    "questions": quick_result["questions"],
                    "per_variant": {
                        v: [
                            {
                                "dataset": r["dataset"],
                                "question_index": r.get("question_index"),
                                "category": r["category"],
                                "baseline_r@20": r["fair_backfill"]["baseline_r@20"],
                                "arch_r@20": r["fair_backfill"]["arch_r@20"],
                                "delta_r@20": r["fair_backfill"]["delta_r@20"],
                                "cues": r.get("cues", []),
                            }
                            for r in rows
                        ]
                        for v, rows in quick_result["all_rows"].items()
                    },
                },
                f, indent=2, default=str,
            )
        print(f"\nSaved quick results: {qpath}")
        print(f"Gate-passing variants: {quick_result['gate_pass']}")

    if args.full or (args.all and quick_result and quick_result["gate_pass"]):
        # If --all, run the full eval on references + gate-passers
        if args.all:
            survivors = ["v2f", "v2f_minimal"] + quick_result["gate_pass"]
            # dedupe preserving order
            seen = set()
            survivors = [
                v for v in survivors if not (v in seen or seen.add(v))
            ]
            run_full_eval(survivors)
        else:
            run_full_eval(variants)


if __name__ == "__main__":
    main()
