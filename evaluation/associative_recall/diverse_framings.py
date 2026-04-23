"""Diverse framings for domain-agnostic cue generation.

Background: V2f uses "conversation history"/"chat message" — domain-coupled.
We've tested:
  - Strip it -> V2f_minimal loses 8.9pp on LoCoMo
  - "Match the style" -> over-mimics, -16pp
  - "Same register" (register_inferred) being tested in a parallel agent

These use similar framings. This file tests DIVERSE framings — different
angles at the same inference goal.

Variants tested here:
  v2f (reference): original v2f with "conversation history" + "chat message"
  v2f_minimal: v2f with domain words stripped (known baseline)
  v2f_voice: write cues "in that voice"
  v2f_continuation: write text that could naturally continue the excerpts
  v2f_author_simulation: pretend you authored the excerpts
  v2f_pattern_completion: sample from the same data distribution
  v2f_genre: write more text in the same genre
  v2f_imitation_bounded: indistinguishable in tone/format, DIFFERENT topics

Gate: variant passes if within 3pp of V2f original on LoCoMo questions on
the 5-question quick-test. Otherwise abandon. Survivors get full 4-dataset eval.

Usage:
    uv run python diverse_framings.py --quick
    uv run python diverse_framings.py --all
"""

import json
import sys
import time
from collections import defaultdict
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
CACHE_FILE_NAME_LLM = "diverse_framings_llm_cache.json"
CACHE_FILE_NAME_EMB = "diverse_framings_embedding_cache.json"


# ===========================================================================
# Prompts
#
# Keep the skeleton (assessment, multi-item guidance, negative question
# constraint, CUE: output format) identical to v2f. Only change the paragraph
# that conveys the content-type inference goal.
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


# V2f_voice: "voice of the excerpts".
V2F_VOICE_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Notice the voice of \
the excerpts above. Write cues in that voice — not quoting them, but using \
the same manner of expression. Use specific vocabulary that would appear \
in the target content.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the target content.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# V2f_continuation: "what else might naturally continue the excerpts".
V2F_CONTINUATION_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Write text that \
could naturally continue from the retrieved excerpts — think: what other \
things might be said in this kind of writing. Use specific vocabulary that \
would appear in the target content.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the target content.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# V2f_author_simulation: "pretend you authored the excerpts".
V2F_AUTHOR_SIMULATION_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Pretend you authored \
the retrieved excerpts. Write additional content you might have written on \
related topics — NOT copying, just in your established voice. Use specific \
vocabulary that would appear in the target content.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the target content.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# V2f_pattern_completion: "sample from the same data distribution".
V2F_PATTERN_COMPLETION_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Treat the retrieved \
excerpts as samples from a data distribution. Generate new samples from \
that same distribution, covering topics not yet in the samples. Use \
specific vocabulary that would appear in the target content.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the target content.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# V2f_genre: "belongs to a particular genre".
V2F_GENRE_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. These excerpts \
belong to a particular genre of text. Write more text in that genre, \
using specific vocabulary that would appear alongside the retrieved content.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the target content.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# V2f_imitation_bounded: indistinguishable tone/format, DIFFERENT topics.
V2F_IMITATION_BOUNDED_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Write cues that \
would be indistinguishable from the retrieved content in terms of tone \
and format — but about DIFFERENT topics from the retrieved content. Use \
specific vocabulary that would appear in the target content.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the target content.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# ===========================================================================
# Caches — share read-only across experiments, write to diverse_framings_*
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
    "domain_agnostic_llm_cache.json",
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
    "domain_agnostic_embedding_cache.json",
    CACHE_FILE_NAME_EMB,
)


class DiverseFramingsEmbeddingCache(EmbeddingCache):
    """Reads from many shared caches, writes to diverse_framings_* only."""

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


class DiverseFramingsLLMCache(LLMCache):
    """Reads from many shared caches, writes to diverse_framings_* only."""

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


# ===========================================================================
# Variant class
# ===========================================================================
class DiverseFramingVariant(OptimBase):
    """Single-call strategist variant — identical retrieval logic to V2f.

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
        self.embedding_cache = DiverseFramingsEmbeddingCache()
        self.llm_cache = DiverseFramingsLLMCache()

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
# ===========================================================================
CONV_HEADER = "RETRIEVED CONVERSATION EXCERPTS SO FAR:"
NEUTRAL_HEADER = "RETRIEVED EXCERPTS SO FAR:"

VARIANT_SPECS: dict[str, tuple[str, str]] = {
    # (prompt_template, context_header)
    "v2f": (V2F_ORIGINAL_PROMPT, CONV_HEADER),
    "v2f_minimal": (V2F_MINIMAL_PROMPT, NEUTRAL_HEADER),
    "v2f_voice": (V2F_VOICE_PROMPT, NEUTRAL_HEADER),
    "v2f_continuation": (V2F_CONTINUATION_PROMPT, NEUTRAL_HEADER),
    "v2f_author_simulation": (V2F_AUTHOR_SIMULATION_PROMPT, NEUTRAL_HEADER),
    "v2f_pattern_completion": (V2F_PATTERN_COMPLETION_PROMPT, NEUTRAL_HEADER),
    "v2f_genre": (V2F_GENRE_PROMPT, NEUTRAL_HEADER),
    "v2f_imitation_bounded": (V2F_IMITATION_BOUNDED_PROMPT, NEUTRAL_HEADER),
}

# New variants requested for diversity (excluding references)
NEW_VARIANTS = [
    "v2f_voice",
    "v2f_continuation",
    "v2f_author_simulation",
    "v2f_pattern_completion",
    "v2f_genre",
    "v2f_imitation_bounded",
]


def build_variant(store: SegmentStore, name: str) -> DiverseFramingVariant:
    if name not in VARIANT_SPECS:
        raise KeyError(f"Unknown variant: {name}")
    prompt, header = VARIANT_SPECS[name]
    return DiverseFramingVariant(
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
    arch: DiverseFramingVariant,
    question: dict,
    budgets: list[int],
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
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "cues": result.metadata.get("cues", []),
        "output": result.metadata.get("output", ""),
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
# Quick-test: 5 representative questions (same spec as domain_agnostic.py)
# ===========================================================================
QUICK_QUESTIONS_SPEC = [
    # (dataset, conversation_id, question_index)
    ("locomo_30q", "locomo_conv-26", 0),
    ("locomo_30q", "locomo_conv-26", 3),
    ("synthetic_19q", "synth_personal", 0),
    ("puzzle_16q", "puzzle_logic_1", 0),
    ("advanced_23q", "adv_evolving_term_1", 0),
]


def load_quick_questions() -> tuple[list[tuple[str, dict]], dict]:
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
    """Gate test on 5 representative questions. Gate: variant within 3pp
    of V2f original on the 2 LoCoMo questions."""
    print("\n" + "=" * 80)
    print("QUICK TEST: 5 representative questions")
    print("=" * 80)

    quick_qs, loaded = load_quick_questions()

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
        + " ".join(f"Q{i+1}@20" for i in range(len(quick_qs)))
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

    # Print the cues each variant produced on each question (for failure mode)
    print("\n--- Cues produced ---")
    for variant_name in variants_to_run:
        if variant_name in ("v2f", "v2f_minimal"):
            continue
        print(f"\n[{variant_name}]")
        for i, r in enumerate(all_rows[variant_name]):
            print(f"  Q{i+1}: ", end="")
            cues = r.get("cues", [])
            for c in cues[:2]:
                print(f"\n    - {c[:200]}", end="")
            print()

    # Decision gate — within 3pp of V2f on the 2 LoCoMo questions
    print("\n--- Decision gate (within 3pp of V2f on LoCoMo Q1 and Q2) ---")
    v2f_ref_rows = all_rows.get("v2f", [])
    gate_pass: list[str] = []
    for variant_name in variants_to_run:
        if variant_name in ("v2f", "v2f_minimal"):
            continue
        rows = all_rows[variant_name]
        locomo_diffs = [
            v2f_ref_rows[i]["fair_backfill"]["arch_r@20"]
            - rows[i]["fair_backfill"]["arch_r@20"]
            for i in (0, 1)
        ]
        max_locomo_gap = max(locomo_diffs)
        passes = max_locomo_gap <= 0.03
        status = "PASS" if passes else "FAIL"
        print(
            f"  {variant_name:<28s} "
            f"locomo_diffs={[f'{d:+.3f}' for d in locomo_diffs]}  "
            f"max_gap={max_locomo_gap:+.3f}  -> {status}"
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
            results = []
            for i, q in enumerate(questions):
                q_short = q["question"][:55]
                print(
                    f"  [{i+1}/{len(questions)}] {q.get('category', '?')}: "
                    f"{q_short}...",
                    flush=True,
                )
                try:
                    row = evaluate_question(arch, q, budgets=BUDGETS_FULL)
                    results.append(row)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                sys.stdout.flush()
                if (i + 1) % 5 == 0:
                    arch.save_caches()
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

            out_path = (
                RESULTS_DIR / f"diverse_framings_{variant_name}_{ds_name}.json"
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

    agg_path = RESULTS_DIR / "diverse_framings_summary.json"
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
        description="Diverse-framings v2f variant evaluation"
    )
    parser.add_argument("--quick", action="store_true", help="5-question gate test")
    parser.add_argument("--full", action="store_true", help="Full 4-dataset eval")
    parser.add_argument(
        "--all", action="store_true",
        help="Run quick; if any new variant passes, run full on survivors",
    )
    parser.add_argument(
        "--variants", type=str, default=None,
        help="Comma-separated variant names (default: all new + references)",
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
        variants = ["v2f", "v2f_minimal"] + NEW_VARIANTS

    for required in ("v2f", "v2f_minimal"):
        if required not in variants:
            variants.insert(0, required)
    seen = set()
    variants = [v for v in variants if not (v in seen or seen.add(v))]

    print(f"Variants to run: {variants}")

    quick_result = None
    if args.quick or args.all:
        quick_result = run_quick_test(variants)
        qpath = RESULTS_DIR / "diverse_framings_quick.json"
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
        if args.all:
            survivors = ["v2f", "v2f_minimal"] + quick_result["gate_pass"]
            seen = set()
            survivors = [
                v for v in survivors if not (v in seen or seen.add(v))
            ]
            run_full_eval(survivors)
        else:
            run_full_eval(variants)


if __name__ == "__main__":
    main()
