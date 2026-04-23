"""Adaptive prompt system for associative-recall retrieval.

Motivation
----------
V2f's anti-question instruction ("Do NOT write questions. Write text that
would actually appear in a chat message.") helps on LoCoMo (+5pp over v15)
but HURTS proactive / task-style retrieval by roughly -10.8pp. For
task/proactive queries the question-format cue ("What are Bob's allergies?")
is in fact effective because it embeds near conversation content ABOUT those
topics, rather than near the content that directly answers them.

This module implements a query-type dispatcher: a heuristic detects the
query type from the query text alone, and a prompt template specialized for
that type generates the cues. Everything else (one LLM call, hop-0 top-10,
two cues top-10 each, cosine retrieval, fair-backfill evaluation) is
identical to V15Control / MetaV2f so the comparison stays clean.

Query types
-----------
1. explicit_question  - starts with what/when/where/who/did/is/how/why
                        -> V2f-style declarative cues (anti-question)
2. task               - starts with help me / draft / prepare / cook / plan
                        / make a / write / create
                        -> task-aware prompt; question-style cues ALLOWED
3. completeness       - contains all / every / list / complete
                        -> emphasise item TYPES / CATEGORIES, not specific
                           items
4. temporal           - contains when / date / before / after / first / last
                        -> emphasise temporal vocabulary
5. comparison         - contains vs / compare / difference
                        -> one cue per side

Priority (first match wins, in this order): task > completeness >
comparison > temporal > explicit_question > default (falls through to V2f).

Usage
-----
    uv run python adaptive_prompts.py            # run full 4-dataset eval
    uv run python adaptive_prompts.py --dataset locomo_30q
    uv run python adaptive_prompts.py --dry-run  # classify without calling LLM
"""

from __future__ import annotations

import argparse
import json
import re
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
BUDGETS = [20, 50]
TOP_K_PER_HOP = 10


DATASETS: dict[str, dict] = {
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


# ===========================================================================
# Caches
# ===========================================================================
class AdaptiveEmbeddingCache(EmbeddingCache):
    """Reads every existing embedding cache (so we reuse prior work), writes
    to adaptive_embedding_cache.json."""

    def __init__(self) -> None:
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
            "bestshot_embedding_cache.json",
            "proactive_embedding_cache.json",
            "adaptive_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "adaptive_embedding_cache.json"
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


class AdaptiveLLMCache(LLMCache):
    """Reads every existing LLM cache (since the v15 and v2f baselines here
    share prompt strings with earlier runs and we can reuse those responses),
    writes to adaptive_llm_cache.json."""

    def __init__(self) -> None:
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
            "bestshot_llm_cache.json",
            "proactive_llm_cache.json",
            "adaptive_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "adaptive_llm_cache.json"
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


# ===========================================================================
# Prompt templates
# ===========================================================================
# The common preamble mirrors V15/V2f so side-by-side comparison is fair.

PROMPT_QUESTION = """\
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


PROMPT_TASK = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

The user is asking for help with a TASK (cooking, drafting, planning, \
preparing, setup, writing, creating). The information you need is \
spread across past messages covering DIFFERENT sub-topics (constraints, \
preferences, decisions, details). A single embedding of the task request \
will not surface all of it.

First, briefly assess: Given what's been retrieved so far, what sub-topics \
are still missing?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues. Each cue should be a DENSE BUNDLE of specific \
vocabulary (names, entities, constraints, actions, numbers, dates) from \
the relevant sub-topic. Probe-style cues are ALLOWED and encouraged when \
the surrounding conversation ABOUT a topic uses that vocabulary (e.g., \
"What are Bob's allergies?" is fine when Bob's allergies were discussed \
with that phrasing). Pick the cue form that best matches how the target \
content was ACTUALLY worded.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


PROMPT_COMPLETENESS = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

The question asks for a COMPLETE list — "all", "every", or similar. The \
answer is spread across many different messages, each mentioning DIFFERENT \
items of the same TYPE. A single embedding biased toward one item will \
miss the others.

First, briefly assess: Given what's been retrieved so far, which specific \
items are already covered and which CATEGORIES or TYPES of items are \
still missing?

Then generate 2 search cues that target DIFFERENT CATEGORIES or TYPES of \
items, not more instances of the same one. Use the generic vocabulary that \
would appear alongside ANY item of that category in the target turns \
(e.g., "allergic to", "can't eat", "avoids" rather than a specific food; \
"dosage", "take", "prescribed" rather than a specific medication).

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message. Do NOT just repeat vocabulary already \
covered by retrieved turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


PROMPT_TEMPORAL = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

The question is TEMPORAL: it asks about WHEN something happened, or \
references "before", "after", "first", "last", or a specific date. The \
target message probably contains explicit temporal vocabulary — a \
weekday, a month, a date, a relative time phrase (last week, yesterday, \
next Tuesday) — along with the event description.

First, briefly assess: what's retrieved so far, and what temporal anchor \
is still missing?

Then generate 2 search cues. Each cue should combine the event vocabulary \
with temporal vocabulary that would plausibly appear in the target turn: \
days of the week, months, numeric dates, or relative time phrases (e.g., \
"last Friday", "on Tuesday", "in March", "next week"). Include both the \
event and the temporal marker.

Do NOT write questions ("When did X?"). Write text that would actually \
appear in a chat message, like how someone announces or recounts an \
event with its time.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


PROMPT_COMPARISON = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

The question is a COMPARISON between two (or more) sides. The information \
for each side lives in DIFFERENT messages, so a single embedding of the \
comparison will bias toward one side.

First, briefly assess: which side(s) of the comparison are already \
covered, and which side(s) are still missing?

Then generate 2 search cues — ONE FOR EACH SIDE of the comparison. Each \
cue should use the specific vocabulary of that side (names, labels, \
adjectives, positions) as it would appear in conversation turns about \
that side in isolation. Do not blend the two sides into one cue.

Do NOT write questions ("What did X say?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text targeting side A>
CUE: <text targeting side B>
Nothing else."""


PROMPTS: dict[str, str] = {
    "explicit_question": PROMPT_QUESTION,
    "task": PROMPT_TASK,
    "completeness": PROMPT_COMPLETENESS,
    "temporal": PROMPT_TEMPORAL,
    "comparison": PROMPT_COMPARISON,
}


# ===========================================================================
# Query-type detection
# ===========================================================================
# Task verbs / phrases — generous since task queries are the ones V2f hurts.
TASK_PATTERNS = [
    r"^help me\b",
    r"^draft\b",
    r"^prepare\b",
    r"^cook\b",
    r"^plan\b",
    r"^make a\b",
    r"^write\b",
    r"^create\b",
    r"^compose\b",
    r"^build\b",
    r"^set up\b",
    r"^setup\b",
    r"^i (want|need|would like|'d like) to\b",
    r"^what needs to happen\b",
    r"^what (should|do) i\b",
    r"\b(draft|prepare|compose|create|write|build) (me |a |the |an )?\b",
    r"\blist of topics\b",
    r"\bchecklist\b",
    r"\bto-?do list\b",
    r"\bstatus update\b",
    r"\bremaining (tasks|phases|steps)\b",
]

COMPLETENESS_PATTERNS = [
    r"\ball\b",
    r"\bevery\b",
    r"\bevery single\b",
    r"\bcomplete\b",
    r"\bentire\b",
    r"\bfull\b",
    r"^list\b",
    r"\blist (all|every|of all|of every|out)\b",
    r"\beach of\b",
]

COMPARISON_PATTERNS = [
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bdifference(s)?\b",
    r"\bhow (does|do|did) .* compare\b",
    r"\bhow (does|do|did) .* differ\b",
    r"\bdiffer from\b",
]

TEMPORAL_PATTERNS = [
    r"^when\b",
    r"\bwhen did\b",
    r"\bwhen was\b",
    r"\bwhen is\b",
    r"\bwhen will\b",
    r"\bon what date\b",
    r"\bon which date\b",
    r"\bwhat date\b",
    r"\bwhat day\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bfirst time\b",
    r"\blast time\b",
    r"\bmost recent\b",
    r"\brecently\b",
    r"\byesterday\b",
    r"\blast (week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bnext (week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
]

EXPLICIT_QUESTION_STARTERS = (
    "what", "when", "where", "who", "whom", "whose",
    "did", "do", "does", "is", "are", "was", "were", "has", "have",
    "how", "why", "which", "can", "could", "should", "would",
)


def detect_query_type(query: str) -> str:
    """Classify `query` into one of the 5 types. First match in priority
    order (task > completeness > comparison > temporal > explicit_question)
    wins; otherwise we fall through to explicit_question, which is the V2f
    prompt (the current LoCoMo best)."""
    q = query.strip().lower()
    # Strip leading punctuation
    q_stripped = q.lstrip("\"'`([{ \t\n")

    # 1. Task — highest priority because V2f hurts task queries most.
    for pat in TASK_PATTERNS:
        if re.search(pat, q_stripped):
            return "task"

    # 2. Completeness — "all / every / list / complete" in the text.
    for pat in COMPLETENESS_PATTERNS:
        if re.search(pat, q_stripped):
            return "completeness"

    # 3. Comparison — explicit comparison vocabulary.
    for pat in COMPARISON_PATTERNS:
        if re.search(pat, q_stripped):
            return "comparison"

    # 4. Temporal — asking about when.
    for pat in TEMPORAL_PATTERNS:
        if re.search(pat, q_stripped):
            return "temporal"

    # 5. Explicit question starter
    first_word = q_stripped.split(" ", 1)[0].rstrip("?.,!:;")
    if first_word in EXPLICIT_QUESTION_STARTERS:
        return "explicit_question"

    # Default: fall back to V2f-style declarative cues.
    return "explicit_question"


# ===========================================================================
# Base runner — shares shape with V15Control / MetaV2f
# ===========================================================================
def _format_segments(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def _parse_cues(response: str) -> list[str]:
    cues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


@dataclass
class RetrievalOutput:
    segments: list[Segment]
    cues: list[str]
    query_type: str
    output: str
    metadata: dict = field(default_factory=dict)


class AdaptiveRunner:
    """Shared runner for all 4 prompt variants (baseline, v15, v2f, adaptive).

    Every variant uses the v15-shape retrieval loop:
      hop 0: embed question, top-10
      LLM call: generate exactly 2 cues
      per cue: embed, top-10, append
    Except `baseline` which only does hop 0 (but at the much higher top-K
    needed for fair evaluation).
    """

    PROMPTS = {
        "v15": (
            "You are generating search text for semantic retrieval over a "
            "conversation history. Your cues will be embedded and compared "
            "via cosine similarity.\n\nQuestion: {question}\n\n{context_section}\n"
            "\nFirst, briefly assess: Given what's been retrieved so far, how "
            "well is this search going? What kind of content is still missing? "
            "Should you search for similar content or pivot to a different "
            "topic?\n\nThen generate 2 search cues based on your assessment. "
            "Use specific vocabulary that would appear in the target "
            "conversation turns.\n\nFormat:\nASSESSMENT: <1-2 sentence "
            "self-evaluation>\nCUE: <text>\nCUE: <text>\nNothing else."
        ),
        "v2f": PROMPT_QUESTION,  # v2f = the "explicit_question" prompt text
    }

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.emb_cache = AdaptiveEmbeddingCache()
        self.llm_cache = AdaptiveLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.emb_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.emb_cache.put(text, embedding)
        self.embed_calls += 1
        return embedding

    def llm_call(self, prompt: str) -> str:
        cached = self.llm_cache.get(MODEL, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(MODEL, prompt, text)
        self.llm_calls += 1
        return text

    def save_caches(self) -> None:
        self.emb_cache.save()
        self.llm_cache.save()

    # -- Retrieval ---------------------------------------------------------
    def retrieve_cosine(
        self, question: str, conversation_id: str, top_k: int
    ) -> list[Segment]:
        """Plain cosine top-K over conversation."""
        q_emb = self.embed_text(question)
        res = self.store.search(
            q_emb, top_k=top_k, conversation_id=conversation_id
        )
        return list(res.segments)

    def retrieve_with_cues(
        self,
        question: str,
        conversation_id: str,
        prompt_template: str,
        query_type: str,
    ) -> RetrievalOutput:
        """V15-shape retrieval: hop 0 top-10, 2 cues top-10 each."""
        q_emb = self.embed_text(question)
        hop0 = self.store.search(
            q_emb, top_k=TOP_K_PER_HOP, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(hop0.segments)
        exclude: set[int] = {s.index for s in all_segments}

        ctx_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(all_segments)
        )
        prompt = prompt_template.format(
            question=question, context_section=ctx_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            res = self.store.search(
                cue_emb,
                top_k=TOP_K_PER_HOP,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in res.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return RetrievalOutput(
            segments=all_segments,
            cues=cues[:2],
            query_type=query_type,
            output=output,
        )

    def run_variant(
        self, variant: str, question: str, conversation_id: str
    ) -> RetrievalOutput:
        if variant == "baseline":
            # Cosine only; no cues. Retrieve max(BUDGETS) so we can truncate.
            segs = self.retrieve_cosine(
                question, conversation_id, top_k=max(BUDGETS)
            )
            return RetrievalOutput(
                segments=segs, cues=[], query_type="baseline", output=""
            )
        if variant == "v15":
            return self.retrieve_with_cues(
                question, conversation_id,
                prompt_template=self.PROMPTS["v15"],
                query_type="v15",
            )
        if variant == "v2f":
            return self.retrieve_with_cues(
                question, conversation_id,
                prompt_template=self.PROMPTS["v2f"],
                query_type="v2f",
            )
        if variant == "adaptive":
            qtype = detect_query_type(question)
            tpl = PROMPTS[qtype]
            return self.retrieve_with_cues(
                question, conversation_id,
                prompt_template=tpl,
                query_type=qtype,
            )
        raise ValueError(f"unknown variant: {variant}")


# ===========================================================================
# Fair-backfill evaluation
# ===========================================================================
def _compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill_evaluate(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    source_ids: set[int],
    budget: int,
) -> tuple[float, float, int]:
    """Compute (baseline_recall, arch_recall, n_backfilled) at budget K.
    Both sides use EXACTLY K segments. If arch returns fewer, backfill from
    cosine; if more, truncate."""
    seen: set[int] = set()
    arch_unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            arch_unique.append(s)
            seen.add(s.index)

    arch_at_K = arch_unique[:budget]
    arch_indices = {s.index for s in arch_at_K}
    n_backfilled = 0
    if len(arch_at_K) < budget:
        backfill = [s for s in cosine_segments if s.index not in arch_indices]
        need = budget - len(arch_at_K)
        arch_at_K = arch_at_K + backfill[:need]
        n_backfilled = min(need, len(backfill))
    arch_at_K = arch_at_K[:budget]
    baseline_at_K = cosine_segments[:budget]

    arch_ids = {s.turn_id for s in arch_at_K}
    baseline_ids = {s.turn_id for s in baseline_at_K}

    return (
        _compute_recall(baseline_ids, source_ids),
        _compute_recall(arch_ids, source_ids),
        n_backfilled,
    )


# ===========================================================================
# Dataset + evaluation harness
# ===========================================================================
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


def evaluate_question(
    runner: AdaptiveRunner,
    variant: str,
    question: dict,
    cosine_segments: list[Segment],
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    runner.reset_counters()
    t0 = time.time()
    out = runner.run_variant(variant, q_text, conv_id)
    elapsed = time.time() - t0

    # Dedupe preserving order
    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for seg in out.segments:
        if seg.index not in seen:
            arch_segments.append(seg)
            seen.add(seg.index)

    row: dict = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "variant": variant,
        "query_type": out.query_type,
        "cues": out.cues,
        "output": out.output,
        "total_arch_retrieved": len(arch_segments),
        "embed_calls": runner.embed_calls,
        "llm_calls": runner.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
    }

    for K in BUDGETS:
        b_rec, a_rec, n_bf = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)
        row["fair_backfill"][f"n_backfilled@{K}"] = n_bf

    return row


def summarize(
    results: list[dict], variant: str, dataset: str
) -> dict:
    n = len(results)
    if n == 0:
        return {"variant": variant, "dataset": dataset, "n": 0}
    out: dict = {"variant": variant, "dataset": dataset, "n": n}
    for K in BUDGETS:
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
    out["avg_llm_calls"] = round(
        sum(r["llm_calls"] for r in results) / n, 1
    )
    out["avg_embed_calls"] = round(
        sum(r["embed_calls"] for r in results) / n, 1
    )
    return out


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        e = {"n": n}
        for K in BUDGETS:
            b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in rs]
            a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in rs]
            b_mean = sum(b_vals) / n
            a_mean = sum(a_vals) / n
            wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
            losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
            ties = n - wins - losses
            e[f"baseline_r@{K}"] = round(b_mean, 4)
            e[f"arch_r@{K}"] = round(a_mean, 4)
            e[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
            e[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
        out[cat] = e
    return out


def summarize_by_query_type(results: list[dict]) -> dict[str, dict]:
    by_qt: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_qt[r.get("query_type", "unknown")].append(r)
    out = {}
    for qt, rs in sorted(by_qt.items()):
        n = len(rs)
        e = {"n": n}
        for K in BUDGETS:
            b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in rs]
            a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in rs]
            b_mean = sum(b_vals) / n
            a_mean = sum(a_vals) / n
            e[f"baseline_r@{K}"] = round(b_mean, 4)
            e[f"arch_r@{K}"] = round(a_mean, 4)
            e[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
        out[qt] = e
    return out


# ===========================================================================
# Main
# ===========================================================================
def run_one(
    runner: AdaptiveRunner,
    variant: str,
    dataset: str,
    questions: list[dict],
    cosine_by_qidx: dict[int, list[Segment]],
) -> tuple[list[dict], dict, dict, dict]:
    print(f"\n{'=' * 72}")
    print(f"VARIANT: {variant}  |  DATASET: {dataset}  |  {len(questions)} qs")
    print(f"{'=' * 72}")
    results = []
    for i, q in enumerate(questions):
        cosine_segs = cosine_by_qidx[i]
        q_short = q["question"][:55]
        print(
            f"  [{i+1}/{len(questions)}] {q.get('category','?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(runner, variant, q, cosine_segs)
            results.append(row)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            runner.save_caches()
    runner.save_caches()

    summary = summarize(results, variant, dataset)
    by_cat = summarize_by_category(results)
    by_qt = summarize_by_query_type(results)

    print(f"\n--- {variant} on {dataset} ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: baseline={summary[f'baseline_r@{K}']:.3f} "
            f"arch={summary[f'arch_r@{K}']:.3f} "
            f"delta={summary[f'delta_r@{K}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    print("  Per-category:")
    for cat, c in by_cat.items():
        print(
            f"    {cat:26s} (n={c['n']}): "
            f"r@20 d={c['delta_r@20']:+.3f} r@50 d={c['delta_r@50']:+.3f} "
            f"W/T/L@20={c['W/T/L_r@20']}"
        )
    if variant == "adaptive":
        print("  Per-query-type:")
        for qt, c in by_qt.items():
            print(
                f"    {qt:26s} (n={c['n']}): "
                f"r@20 d={c['delta_r@20']:+.3f} r@50 d={c['delta_r@50']:+.3f}"
            )
    return results, summary, by_cat, by_qt


def precompute_cosine(
    runner: AdaptiveRunner, questions: list[dict]
) -> dict[int, list[Segment]]:
    """Compute cosine top-max(BUDGETS) for every question, once. Used as
    both baseline and backfill source for all variants."""
    cos_by_idx: dict[int, list[Segment]] = {}
    for i, q in enumerate(questions):
        segs = runner.retrieve_cosine(
            q["question"], q["conversation_id"], top_k=max(BUDGETS)
        )
        cos_by_idx[i] = segs
    runner.save_caches()
    return cos_by_idx


def run_dataset(
    ds_name: str, variants: list[str]
) -> dict:
    store, questions = load_dataset(ds_name)
    print(
        f"\nLoaded {ds_name}: {len(questions)} questions, "
        f"{len(store.segments)} segments"
    )
    runner = AdaptiveRunner(store)

    # Precompute cosine top-max(BUDGETS) once per question (shared across
    # all variants, for baseline & backfill).
    print(f"  precomputing cosine top-{max(BUDGETS)} for {len(questions)} qs...")
    cos_by_idx = precompute_cosine(runner, questions)

    per_variant: dict[str, dict] = {}
    for v in variants:
        results, summary, by_cat, by_qt = run_one(
            runner, v, ds_name, questions, cos_by_idx
        )
        out_path = RESULTS_DIR / f"adaptive_{v}_{ds_name}.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "variant": v,
                    "dataset": ds_name,
                    "summary": summary,
                    "category_breakdown": by_cat,
                    "query_type_breakdown": by_qt,
                    "results": results,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"  Saved: {out_path}")
        per_variant[v] = {
            "summary": summary,
            "category_breakdown": by_cat,
            "query_type_breakdown": by_qt,
        }
    return per_variant


def print_master_table(all_summaries: dict) -> None:
    print("\n" + "=" * 110)
    print("ADAPTIVE PROMPTS — MASTER SUMMARY (fair-backfill, exactly K each)")
    print("=" * 110)
    header = (
        f"{'Variant':<12s} {'Dataset':<16s} "
        f"{'base@20':>8s} {'arch@20':>8s} {'d@20':>7s} {'W/T/L@20':>10s} "
        f"{'base@50':>8s} {'arch@50':>8s} {'d@50':>7s} {'W/T/L@50':>10s}"
    )
    print(header)
    print("-" * len(header))
    for variant, by_ds in all_summaries.items():
        for ds, entry in by_ds.items():
            s = entry["summary"]
            print(
                f"{variant:<12s} {ds:<16s} "
                f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
                f"{s['delta_r@20']:>+7.3f} {s['W/T/L_r@20']:>10s} "
                f"{s['baseline_r@50']:>8.3f} {s['arch_r@50']:>8.3f} "
                f"{s['delta_r@50']:>+7.3f} {s['W/T/L_r@50']:>10s}"
            )

    # Head-to-head per-dataset (adaptive vs best of v15/v2f)
    if "adaptive" in all_summaries:
        print("\nHEAD-TO-HEAD: adaptive vs best of {v15,v2f}")
        print("-" * 72)
        for ds in all_summaries["adaptive"]:
            for K in BUDGETS:
                vals = []
                for variant in ("v15", "v2f", "adaptive"):
                    if variant in all_summaries and ds in all_summaries[variant]:
                        vals.append(
                            (
                                variant,
                                all_summaries[variant][ds]["summary"][
                                    f"arch_r@{K}"
                                ],
                            )
                        )
                if not vals:
                    continue
                best_single = max(
                    (v for v in vals if v[0] != "adaptive"), key=lambda x: x[1]
                )
                adp = next(v for v in vals if v[0] == "adaptive")
                print(
                    f"  {ds:<16s} r@{K}: best_single={best_single[0]}={best_single[1]:.3f} "
                    f"adaptive={adp[1]:.3f} "
                    f"delta={adp[1] - best_single[1]:+.3f}"
                )

    # Category-level head-to-head: adaptive should match v15/v2f at their
    # strengths and exceed at their weaknesses.
    if "adaptive" in all_summaries:
        print("\nCATEGORY HEAD-TO-HEAD (arch_r@50)")
        print("-" * 110)
        print(
            f"{'Dataset':<16s} {'Category':<30s} {'v15':>7s} {'v2f':>7s} "
            f"{'adaptive':>10s} {'adp-v15':>8s} {'adp-v2f':>8s} {'adp-best':>9s}"
        )
        print("-" * 110)
        for ds in all_summaries["adaptive"]:
            if ds not in all_summaries.get("v15", {}):
                continue
            v15_cats = all_summaries["v15"][ds]["category_breakdown"]
            v2f_cats = all_summaries["v2f"][ds]["category_breakdown"]
            adp_cats = all_summaries["adaptive"][ds]["category_breakdown"]
            for cat in sorted(v15_cats):
                v15a = v15_cats[cat]["arch_r@50"]
                v2fa = v2f_cats.get(cat, {}).get("arch_r@50", float("nan"))
                adpa = adp_cats.get(cat, {}).get("arch_r@50", float("nan"))
                best_single = max(v15a, v2fa)
                print(
                    f"{ds:<16s} {cat:<30s} "
                    f"{v15a:>7.3f} {v2fa:>7.3f} {adpa:>10.3f} "
                    f"{adpa - v15a:>+8.3f} {adpa - v2fa:>+8.3f} "
                    f"{adpa - best_single:>+9.3f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "v15", "v2f", "adaptive"],
        choices=["baseline", "v15", "v2f", "adaptive"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print query-type classification on each dataset, no LLM calls.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        ds_names = (
            [args.dataset] if args.dataset != "all" else list(DATASETS.keys())
        )
        for ds in ds_names:
            _, qs = load_dataset(ds)
            print(f"\n=== {ds} ({len(qs)} questions) ===")
            counts: dict[str, int] = defaultdict(int)
            for q in qs:
                qt = detect_query_type(q["question"])
                counts[qt] += 1
                print(
                    f"  [{qt:18s}] {q.get('category','?'):22s} {q['question'][:80]}"
                )
            print(f"  query-type distribution: {dict(counts)}")
        return

    ds_names = (
        [args.dataset] if args.dataset != "all" else list(DATASETS.keys())
    )

    all_summaries: dict = {v: {} for v in args.variants}
    for ds in ds_names:
        per_variant = run_dataset(ds, args.variants)
        for v, entry in per_variant.items():
            all_summaries[v][ds] = entry

    # Save aggregated
    out_path = RESULTS_DIR / "adaptive_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved aggregated summary: {out_path}")
    print_master_table(all_summaries)


if __name__ == "__main__":
    main()
