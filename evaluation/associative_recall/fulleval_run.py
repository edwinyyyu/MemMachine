"""Full evaluation: 6 architectures x 3 hard benchmark datasets.

Architectures from best_shot.py:
  1. v15_control — reference baseline
  2. meta_v2f — V2f prompt (completeness + anti-question hints)
  3. frontier_v2_iterative — iterative reflection with 1 gap per round
  4. retrieve_then_decompose — v15 first hop + gap discovery

Architectures from task_execution.py:
  5. gen_check_v2 — Generate-and-Check v2, skeptical prompt
  6. decompose_retrieve — decompose-then-retrieve

Datasets:
  1. Synthetic 19q: segments_synthetic.npz + questions_synthetic.json
  2. Puzzle 16q: segments_puzzle.npz + questions_puzzle.json
  3. Advanced 23q: segments_advanced.npz + questions_advanced.json

Usage:
    uv run python fulleval_run.py
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BUDGETS = [20, 50]

CACHE_FILE_LLM = CACHE_DIR / "fulleval_llm_cache.json"
CACHE_FILE_EMB = CACHE_DIR / "fulleval_embedding_cache.json"


# ---------------------------------------------------------------------------
# Cache classes — fulleval-specific, reads from all existing caches
# ---------------------------------------------------------------------------
class FullevalEmbeddingCache(EmbeddingCache):
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


class FullevalLLMCache(LLMCache):
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
# Shared embedding/LLM infrastructure
# ---------------------------------------------------------------------------
embedding_cache = FullevalEmbeddingCache()
llm_cache = FullevalLLMCache()
client = OpenAI(timeout=60.0)


def embed_text(text: str) -> np.ndarray:
    text = text.strip()
    if not text:
        return np.zeros(1536, dtype=np.float32)
    cached = embedding_cache.get(text)
    if cached is not None:
        return cached
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    emb = np.array(response.data[0].embedding, dtype=np.float32)
    embedding_cache.put(text, emb)
    return emb


def llm_call(prompt: str, model: str = MODEL) -> str:
    cached = llm_cache.get(model, prompt)
    if cached is not None:
        return cached
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=3000,
    )
    text = response.choices[0].message.content or ""
    llm_cache.put(model, prompt, text)
    return text


def save_caches():
    embedding_cache.save()
    llm_cache.save()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_segments(segments: list[Segment], max_items: int = 12,
                    max_chars: int = 250) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def parse_cues(response: str) -> list[str]:
    cues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


def parse_gaps(response: str) -> list[str]:
    gaps = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("GAP:"):
            gap = line[4:].strip()
            if gap:
                gaps.append(gap)
    return gaps


def parse_queries(response: str) -> list[str]:
    queries = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("QUERY:"):
            q = line[6:].strip()
            if q:
                queries.append(q)
    return queries


def retrieve_top_k(
    store: SegmentStore,
    query: str,
    conversation_id: str,
    top_k: int = 10,
    exclude_indices: set[int] | None = None,
) -> list[Segment]:
    query_emb = embed_text(query)
    result = store.search(
        query_emb, top_k=top_k,
        conversation_id=conversation_id,
        exclude_indices=exclude_indices,
    )
    return list(result.segments)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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

FRONTIER_REFLECT_PROMPT = """\
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

V15_SINGLE_CUE_PROMPT = """\
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

GROUNDED_DECOMPOSE_PROMPT = """\
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

# -- Task execution prompts --

GEN_CHECK_V2_PROMPT = """\
You are completing a task using ONLY information from retrieved memories. \
You MUST NOT include any facts, names, dates, numbers, or details that are \
not explicitly present in the retrieved memories below.

TASK: {task}

RETRIEVED MEMORIES:
{formatted_segments}

INSTRUCTIONS:
Look at the task and the retrieved memories. For each piece of information \
the task requires, check whether it appears in the retrieved memories.

If you find information missing that the task needs, output EXACTLY:
NEED: <natural text describing what the conversation would have said>

The NEED query will be embedded and compared against stored conversation \
turns via cosine similarity. Write it as natural text that would APPEAR in \
the stored content. For example: "Bob mentioned he is allergic to peanuts \
and carries an EpiPen."

RULES:
- You MUST use NEED for any detail not explicitly in the retrieved memories.
- Scan the task requirements systematically. If the task asks about multiple \
people or items, check that you have info for EACH one.
- After your NEED line, STOP. Do not continue writing output.
- If all needed information is present, write your complete response.
- When writing your response, cite specific details from the memories.

{completeness_hint}"""

DECOMPOSE_PROMPT = """\
You are planning retrieval from a memory store to complete a task.

TASK: {task}

Break this task down into specific information needs. What distinct pieces \
of information would you need to retrieve from past conversations to \
complete this task thoroughly?

For each information need, write a natural-language search query that would \
match the relevant conversation content via cosine similarity. Write text \
that would APPEAR in the stored conversations, not search commands.

Good query: "Bob is allergic to peanuts, has been since he was a kid"
Bad query: "Bob dietary restrictions" or "search allergies"

List 4-8 distinct queries, each targeting a DIFFERENT aspect of the task.

Format:
QUERY: <natural text query>
QUERY: <natural text query>
...
Nothing else."""

DECOMPOSE_REFINE_PROMPT = """\
You are refining a retrieval plan for a task. You have already retrieved \
some information.

TASK: {task}

RETRIEVED SO FAR ({num_retrieved} segments):
{formatted_segments}

QUERIES ALREADY USED:
{queries_so_far}

Look at what you have. What information is still MISSING to complete the \
task? Generate 2-4 NEW queries targeting the missing information.

Each query should be natural text that would appear in conversation turns, \
matched via cosine similarity. Target DIFFERENT topics from what you've \
already found.

Format:
QUERY: <natural text query>
Nothing else. If you believe all needed information has been retrieved, \
write: COMPLETE"""


# ===========================================================================
# Architecture implementations
# ===========================================================================

def run_v15_control(store: SegmentStore, question: str, conv_id: str) -> list[Segment]:
    """v15_control: question top-10 + 1 LLM call producing 2 cues, each top-10."""
    query_emb = embed_text(question)
    hop0 = store.search(query_emb, top_k=10, conversation_id=conv_id)
    all_segments = list(hop0.segments)
    exclude = {s.index for s in all_segments}

    context_section = (
        "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
        + format_segments(all_segments)
    )
    prompt = V15_CONTROL_PROMPT.format(
        question=question, context_section=context_section
    )
    output = llm_call(prompt)
    cues = parse_cues(output)

    for cue in cues[:2]:
        cue_emb = embed_text(cue)
        result = store.search(
            cue_emb, top_k=10, conversation_id=conv_id,
            exclude_indices=exclude,
        )
        for seg in result.segments:
            if seg.index not in exclude:
                all_segments.append(seg)
                exclude.add(seg.index)

    return all_segments


def run_meta_v2f(store: SegmentStore, question: str, conv_id: str) -> list[Segment]:
    """meta_v2f: V2f prompt = v15 + completeness hint + anti-question."""
    query_emb = embed_text(question)
    hop0 = store.search(query_emb, top_k=10, conversation_id=conv_id)
    all_segments = list(hop0.segments)
    exclude = {s.index for s in all_segments}

    context_section = (
        "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
        + format_segments(all_segments)
    )
    prompt = V2F_PROMPT.format(
        question=question, context_section=context_section
    )
    output = llm_call(prompt)
    cues = parse_cues(output)

    for cue in cues[:2]:
        cue_emb = embed_text(cue)
        result = store.search(
            cue_emb, top_k=10, conversation_id=conv_id,
            exclude_indices=exclude,
        )
        for seg in result.segments:
            if seg.index not in exclude:
                all_segments.append(seg)
                exclude.add(seg.index)

    return all_segments


def run_frontier_v2_iterative(
    store: SegmentStore, question: str, conv_id: str,
    max_reflects: int = 4,
) -> list[Segment]:
    """frontier_v2_iterative: iterative reflect with 1 gap per round."""
    exclude: set[int] = set()
    all_segments: list[Segment] = []
    reflect_log: list[dict] = []

    query_emb = embed_text(question)
    result = store.search(query_emb, top_k=10, conversation_id=conv_id)
    all_segments.extend(result.segments)
    for s in result.segments:
        exclude.add(s.index)

    for reflect_i in range(max_reflects):
        if len(all_segments) >= 80:
            break

        context = format_segments(all_segments)
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

        prompt = FRONTIER_REFLECT_PROMPT.format(
            question=question, context=context, explored_text=explored_text
        )
        response = llm_call(prompt)

        gaps = []
        done = False
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    gaps.append(cue)
            elif line.strip().upper() == "DONE":
                done = True

        reflect_log.append({"reflect": reflect_i, "gaps": gaps, "done": done})

        if done or not gaps:
            break

        for gap in gaps[:1]:
            if len(all_segments) >= 80:
                break
            gap_emb = embed_text(gap)
            result = store.search(
                gap_emb, top_k=10, conversation_id=conv_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

    return all_segments


def run_retrieve_then_decompose(
    store: SegmentStore, question: str, conv_id: str,
) -> list[Segment]:
    """retrieve_then_decompose: v15 first hop + single cue + gap decompose."""
    exclude: set[int] = set()
    all_segments: list[Segment] = []

    # Phase 1a: initial retrieval
    query_emb = embed_text(question)
    result = store.search(query_emb, top_k=10, conversation_id=conv_id)
    initial_segs = list(result.segments)
    all_segments.extend(initial_segs)
    for s in initial_segs:
        exclude.add(s.index)

    # Phase 1b: v15-style single cue
    context = format_segments(initial_segs, max_items=10)
    v15_prompt = V15_SINGLE_CUE_PROMPT.format(
        question=question, retrieved_context=context
    )
    v15_output = llm_call(v15_prompt)
    v15_cues = parse_cues(v15_output)
    v15_segs: list[Segment] = []
    if v15_cues:
        cue = v15_cues[0]
        cue_emb = embed_text(cue)
        result = store.search(
            cue_emb, top_k=10, conversation_id=conv_id,
            exclude_indices=exclude,
        )
        for seg in result.segments:
            if seg.index not in exclude:
                v15_segs.append(seg)
                all_segments.append(seg)
                exclude.add(seg.index)

    # Phase 2: grounded decomposition
    found_so_far = initial_segs + v15_segs
    gap_context = format_segments(found_so_far, max_items=12)
    decompose_prompt = GROUNDED_DECOMPOSE_PROMPT.format(
        question=question, retrieved_context=gap_context
    )
    decompose_output = llm_call(decompose_prompt)
    sub_questions = parse_gaps(decompose_output)
    if not sub_questions:
        sub_questions = [question]
    sub_questions = sub_questions[:4]

    # Phase 3: per-gap retrieval
    for sq in sub_questions:
        sq_emb = embed_text(sq)
        result = store.search(
            sq_emb, top_k=10, conversation_id=conv_id,
            exclude_indices=exclude,
        )
        for seg in result.segments:
            if seg.index not in exclude:
                all_segments.append(seg)
                exclude.add(seg.index)

    return all_segments


def run_gen_check_v2(
    store: SegmentStore, question: str, conv_id: str,
    max_rounds: int = 6,
) -> list[Segment]:
    """gen_check_v2: retrieve-first, skeptical generation with NEED: triggers."""
    all_segments: list[Segment] = []
    exclude_indices: set[int] = set()

    # Initial retrieval
    initial_segments = retrieve_top_k(store, question, conv_id, top_k=10)
    all_segments.extend(initial_segments)
    exclude_indices.update(s.index for s in initial_segments)

    for round_num in range(max_rounds):
        formatted = format_segments(all_segments, max_items=16) if all_segments else "(none yet)"

        if round_num == 0:
            completeness_hint = (
                "This is the first retrieval. It is very likely that "
                "important information is still missing. Use NEED to "
                "search for what's missing before writing your response."
            )
        elif round_num < 3:
            completeness_hint = (
                "You have retrieved additional information. Check if "
                "anything is still missing before writing your response."
            )
        else:
            completeness_hint = (
                "You have done several retrieval rounds. If you believe "
                "you have enough information, write your response now."
            )

        prompt = GEN_CHECK_V2_PROMPT.format(
            task=question,
            formatted_segments=formatted,
            completeness_hint=completeness_hint,
        )

        response = llm_call(prompt)

        if "NEED:" in response:
            need_query = response.split("NEED:", 1)[1].strip()
            need_query = need_query.split("\n")[0].strip()
            new_segments = retrieve_top_k(
                store, need_query, conv_id, top_k=10,
                exclude_indices=exclude_indices,
            )
            all_segments.extend(new_segments)
            exclude_indices.update(s.index for s in new_segments)
        else:
            break

    return all_segments


def run_decompose_retrieve(
    store: SegmentStore, question: str, conv_id: str,
    max_refine_rounds: int = 2,
    top_k_per_query: int = 5,
) -> list[Segment]:
    """decompose_retrieve: plan queries, retrieve, refine."""
    all_segments: list[Segment] = []
    exclude_indices: set[int] = set()
    all_queries: list[str] = []

    # Step 1: Decompose
    decompose_prompt = DECOMPOSE_PROMPT.format(task=question)
    decompose_response = llm_call(decompose_prompt)
    initial_queries = parse_queries(decompose_response)

    all_query_list = [question] + initial_queries

    # Step 2: Retrieve for each
    for query in all_query_list:
        new_segments = retrieve_top_k(
            store, query, conv_id, top_k=top_k_per_query,
            exclude_indices=exclude_indices,
        )
        all_segments.extend(new_segments)
        exclude_indices.update(s.index for s in new_segments)
        all_queries.append(query)

    # Step 3: Refine
    for refine_round in range(max_refine_rounds):
        formatted = format_segments(all_segments, max_items=20)
        queries_so_far = "\n".join(f"- {q[:100]}" for q in all_queries)

        refine_prompt = DECOMPOSE_REFINE_PROMPT.format(
            task=question,
            num_retrieved=len(all_segments),
            formatted_segments=formatted,
            queries_so_far=queries_so_far,
        )

        refine_response = llm_call(refine_prompt)

        if "COMPLETE" in refine_response.upper() and "QUERY:" not in refine_response:
            break

        refine_queries = parse_queries(refine_response)
        if not refine_queries:
            break

        for query in refine_queries:
            new_segments = retrieve_top_k(
                store, query, conv_id, top_k=top_k_per_query,
                exclude_indices=exclude_indices,
            )
            all_segments.extend(new_segments)
            exclude_indices.update(s.index for s in new_segments)
            all_queries.append(query)

    return all_segments


# ===========================================================================
# Architecture registry
# ===========================================================================
ARCHITECTURES = {
    "v15_control": run_v15_control,
    "meta_v2f": run_meta_v2f,
    "frontier_v2_iterative": run_frontier_v2_iterative,
    "retrieve_then_decompose": run_retrieve_then_decompose,
    "gen_check_v2": run_gen_check_v2,
    "decompose_retrieve": run_decompose_retrieve,
}


# ===========================================================================
# Evaluation
# ===========================================================================
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def deduplicate(segments: list[Segment]) -> list[Segment]:
    seen: set[int] = set()
    deduped: list[Segment] = []
    for seg in segments:
        if seg.index not in seen:
            deduped.append(seg)
            seen.add(seg.index)
    return deduped


def evaluate_one(
    store: SegmentStore,
    arch_name: str,
    arch_fn,
    question: dict,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    t0 = time.time()
    raw_segments = arch_fn(store, q_text, conv_id)
    elapsed = time.time() - t0

    segments = deduplicate(raw_segments)
    total_retrieved = len(segments)

    # Baseline: cosine top-N
    query_emb = embed_text(q_text)
    max_budget = max(BUDGETS + [total_retrieved])
    baseline_result = store.search(query_emb, top_k=max_budget, conversation_id=conv_id)

    baseline_recalls: dict[str, float] = {}
    arch_recalls: dict[str, float] = {}
    for budget in BUDGETS:
        baseline_ids = {s.turn_id for s in baseline_result.segments[:budget]}
        baseline_recalls[f"r@{budget}"] = compute_recall(baseline_ids, source_ids)

        arch_ids = {s.turn_id for s in segments[:budget]}
        arch_recalls[f"r@{budget}"] = compute_recall(arch_ids, source_ids)

    return {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question.get("question_index", -1),
        "question": q_text[:120],
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "baseline_recalls": baseline_recalls,
        "arch_recalls": arch_recalls,
        "total_retrieved": total_retrieved,
        "time_s": round(elapsed, 2),
    }


# ===========================================================================
# Dataset definitions
# ===========================================================================
DATASETS = {
    "synthetic": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "label": "Synthetic 19q",
    },
    "puzzle": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "label": "Puzzle 16q",
    },
    "advanced": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "label": "Advanced 23q",
    },
}


# ===========================================================================
# Reporting
# ===========================================================================
def print_dataset_table(
    dataset_name: str,
    all_results: dict[str, list[dict]],
    budget: int = 20,
):
    """Print per-category table for one dataset across all architectures."""
    # Gather categories
    categories: dict[str, int] = {}
    for arch_name, results in all_results.items():
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = 0
            categories[cat] = max(categories[cat], 1)

    # Count per category
    cat_counts: dict[str, int] = defaultdict(int)
    first_results = next(iter(all_results.values()))
    for r in first_results:
        cat_counts[r["category"]] += 1

    arch_names = list(all_results.keys())

    # Compute per-category means
    cat_arch_means: dict[str, dict[str, float]] = {}
    for cat in cat_counts:
        cat_arch_means[cat] = {}
        for arch_name in arch_names:
            results = all_results[arch_name]
            cat_results = [r for r in results if r["category"] == cat]
            if not cat_results:
                cat_arch_means[cat][arch_name] = 0.0
                continue
            vals = [r["arch_recalls"][f"r@{budget}"] for r in cat_results]
            cat_arch_means[cat][arch_name] = sum(vals) / len(vals)

    # Also compute baseline means
    cat_baseline_means: dict[str, float] = {}
    for cat in cat_counts:
        first_arch = arch_names[0]
        results = all_results[first_arch]
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            vals = [r["baseline_recalls"][f"r@{budget}"] for r in cat_results]
            cat_baseline_means[cat] = sum(vals) / len(vals)
        else:
            cat_baseline_means[cat] = 0.0

    # Print table
    short_names = {
        "v15_control": "v15",
        "meta_v2f": "v2f",
        "frontier_v2_iterative": "frontier",
        "retrieve_then_decompose": "ret_dec",
        "gen_check_v2": "gen_chk",
        "decompose_retrieve": "dec_ret",
    }

    print(f"\n{'='*100}")
    print(f"DATASET: {dataset_name} | r@{budget}")
    print(f"{'='*100}")

    header = f"{'Category':<28} | {'Baseline':>8}"
    for an in arch_names:
        header += f" | {short_names.get(an, an[:7]):>8}"
    header += " | Best"
    print(header)
    print("-" * len(header))

    overall_arch_sums: dict[str, float] = defaultdict(float)
    overall_baseline_sum = 0.0
    total_q = 0

    for cat in sorted(cat_counts.keys()):
        n = cat_counts[cat]
        bl = cat_baseline_means[cat]
        overall_baseline_sum += bl * n
        total_q += n

        row = f"{cat} ({n}q)"
        row = f"{row:<28} | {bl:>8.2f}"

        best_val = -1.0
        best_arch = ""
        for an in arch_names:
            v = cat_arch_means[cat][an]
            overall_arch_sums[an] += v * n
            row += f" | {v:>8.2f}"
            if v > best_val:
                best_val = v
                best_arch = an

        row += f" | {short_names.get(best_arch, best_arch[:7])}"
        print(row)

    # Overall average
    print("-" * len(header))
    bl_avg = overall_baseline_sum / total_q if total_q else 0
    row = f"{'OVERALL':<28} | {bl_avg:>8.2f}"
    for an in arch_names:
        a_avg = overall_arch_sums[an] / total_q if total_q else 0
        row += f" | {a_avg:>8.2f}"
    print(row)
    print()


def print_cross_arch_summary(
    all_dataset_results: dict[str, dict[str, list[dict]]],
    budget: int = 20,
):
    """Print which architecture is best for each category across all datasets."""
    print(f"\n{'='*100}")
    print(f"CROSS-ARCHITECTURE SUMMARY: Best architecture per category (r@{budget})")
    print(f"{'='*100}")

    arch_names = list(next(iter(all_dataset_results.values())).keys())
    short_names = {
        "v15_control": "v15",
        "meta_v2f": "v2f",
        "frontier_v2_iterative": "frontier",
        "retrieve_then_decompose": "ret_dec",
        "gen_check_v2": "gen_chk",
        "decompose_retrieve": "dec_ret",
    }

    print(f"\n{'Dataset':<12} {'Category':<28} {'Best Arch':<12} {'Score':>6} {'2nd Best':<12} {'Score':>6} {'Baseline':>8}")
    print("-" * 100)

    arch_win_counts: dict[str, int] = defaultdict(int)

    for ds_name in ["synthetic", "puzzle", "advanced"]:
        ds_results = all_dataset_results.get(ds_name, {})
        if not ds_results:
            continue

        # Get categories
        cat_counts: dict[str, int] = defaultdict(int)
        first_arch = list(ds_results.keys())[0]
        for r in ds_results[first_arch]:
            cat_counts[r["category"]] += 1

        for cat in sorted(cat_counts.keys()):
            n = cat_counts[cat]

            # Baseline
            first_results = ds_results[first_arch]
            cat_results = [r for r in first_results if r["category"] == cat]
            bl_vals = [r["baseline_recalls"][f"r@{budget}"] for r in cat_results]
            bl_mean = sum(bl_vals) / len(bl_vals) if bl_vals else 0

            # Per-arch
            arch_scores: list[tuple[str, float]] = []
            for an in arch_names:
                results = ds_results.get(an, [])
                cr = [r for r in results if r["category"] == cat]
                if cr:
                    vals = [r["arch_recalls"][f"r@{budget}"] for r in cr]
                    arch_scores.append((an, sum(vals) / len(vals)))
                else:
                    arch_scores.append((an, 0.0))

            arch_scores.sort(key=lambda x: x[1], reverse=True)
            best_arch, best_score = arch_scores[0]
            second_arch, second_score = arch_scores[1] if len(arch_scores) > 1 else ("", 0)
            arch_win_counts[best_arch] += 1

            print(
                f"{ds_name:<12} {f'{cat} ({n}q)':<28} "
                f"{short_names.get(best_arch, best_arch[:10]):<12} {best_score:>6.2f} "
                f"{short_names.get(second_arch, second_arch[:10]):<12} {second_score:>6.2f} "
                f"{bl_mean:>8.2f}"
            )

    print(f"\n--- Win counts (number of categories where each architecture is best) ---")
    for an in sorted(arch_win_counts, key=arch_win_counts.get, reverse=True):
        print(f"  {short_names.get(an, an)}: {arch_win_counts[an]}")


def print_r50_table(
    dataset_name: str,
    all_results: dict[str, list[dict]],
):
    """Print per-category table at r@50."""
    print_dataset_table(dataset_name, all_results, budget=50)


# ===========================================================================
# Main
# ===========================================================================
def main():
    # Track all results for cross-architecture summary
    all_dataset_results: dict[str, dict[str, list[dict]]] = {}

    for ds_name, ds_info in DATASETS.items():
        print(f"\n{'#'*100}")
        print(f"# LOADING DATASET: {ds_info['label']}")
        print(f"{'#'*100}")

        store = SegmentStore(DATA_DIR, ds_info["npz"])
        with open(DATA_DIR / ds_info["questions"]) as f:
            questions = json.load(f)

        print(f"  Segments: {len(store.segments)}, Questions: {len(questions)}")

        ds_all_results: dict[str, list[dict]] = {}

        for arch_name, arch_fn in ARCHITECTURES.items():
            print(f"\n--- Running {arch_name} on {ds_info['label']} ---")

            results = []
            for i, question in enumerate(questions):
                q_short = question["question"][:55]
                cat = question["category"]
                print(
                    f"  [{i+1}/{len(questions)}] {cat}: {q_short}...",
                    end="", flush=True,
                )
                try:
                    result = evaluate_one(store, arch_name, arch_fn, question)
                    results.append(result)
                    r20 = result["arch_recalls"]["r@20"]
                    r50 = result["arch_recalls"]["r@50"]
                    bl20 = result["baseline_recalls"]["r@20"]
                    delta = r20 - bl20
                    print(
                        f" r@20={r20:.2f} (bl={bl20:.2f}, d={delta:+.2f})"
                        f" r@50={r50:.2f} [{result['time_s']:.1f}s]"
                    )
                except Exception as e:
                    print(f" ERROR: {e}")
                    import traceback
                    traceback.print_exc()

                if (i + 1) % 5 == 0:
                    save_caches()

            save_caches()

            # Save per-architecture results
            out_path = RESULTS_DIR / f"fulleval_{ds_name}_{arch_name}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved: {out_path}")

            ds_all_results[arch_name] = results

            # Print quick summary
            if results:
                for budget in BUDGETS:
                    lbl = f"r@{budget}"
                    bl_vals = [r["baseline_recalls"][lbl] for r in results]
                    a_vals = [r["arch_recalls"][lbl] for r in results]
                    bl_mean = sum(bl_vals) / len(bl_vals)
                    a_mean = sum(a_vals) / len(a_vals)
                    print(f"  {arch_name} {lbl}: baseline={bl_mean:.3f} arch={a_mean:.3f} delta={a_mean-bl_mean:+.3f}")

        all_dataset_results[ds_name] = ds_all_results

        # Print per-dataset table at r@20 and r@50
        print_dataset_table(ds_info["label"], ds_all_results, budget=20)
        print_dataset_table(ds_info["label"], ds_all_results, budget=50)

    # Cross-architecture summary
    print_cross_arch_summary(all_dataset_results, budget=20)
    print_cross_arch_summary(all_dataset_results, budget=50)

    # Final summary JSON
    summary = {}
    for ds_name, ds_results in all_dataset_results.items():
        summary[ds_name] = {}
        for arch_name, results in ds_results.items():
            if not results:
                continue
            for budget in BUDGETS:
                lbl = f"r@{budget}"
                bl_vals = [r["baseline_recalls"][lbl] for r in results]
                a_vals = [r["arch_recalls"][lbl] for r in results]
                summary[ds_name][f"{arch_name}_{lbl}"] = round(sum(a_vals)/len(a_vals), 4)
                summary[ds_name][f"baseline_{lbl}"] = round(sum(bl_vals)/len(bl_vals), 4)

    summary_path = RESULTS_DIR / "fulleval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
