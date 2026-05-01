"""Task-execution retrieval: retrieval emerges from task execution, not from questions.

Implements two architectures:
  1. Generate-and-Check: Model outputs text, emits NEED: when it lacks info.
  4. Autonomous with Tools: Model has THINK/RETRIEVE/WRITE/DONE actions.

Also runs V2f baseline for comparison.

Usage:
    uv run python task_execution.py --arch generate_and_check
    uv run python task_execution.py --arch autonomous
    uv run python task_execution.py --arch v2f_baseline
    uv run python task_execution.py --all
    uv run python task_execution.py --all --verbose
"""

import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-4.1-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50, 100]

# Target question indices (proactive, procedural, completeness)
TARGET_INDICES = {6, 7, 8, 9, 13, 14, 15, 16, 17, 18}


# ---------------------------------------------------------------------------
# Cache classes -- task_exec specific cache files
# ---------------------------------------------------------------------------
class TaskExecEmbeddingCache(EmbeddingCache):
    """Reads all existing caches, writes to task_exec_embedding_cache.json."""

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
            "general_embedding_cache.json",
            "synth_test_embedding_cache.json",
            "task_exec_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "task_exec_embedding_cache.json"
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


class TaskExecLLMCache(LLMCache):
    """Reads all existing caches, writes to task_exec_llm_cache.json."""

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
            "general_llm_cache.json",
            "synth_test_llm_cache.json",
            "task_exec_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "task_exec_llm_cache.json"
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
# Base class
# ---------------------------------------------------------------------------
class TaskExecBase:
    """Base class with embedding/LLM utilities and counters."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = TaskExecEmbeddingCache()
        self.llm_cache = TaskExecLLMCache()
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

    def retrieve_top_k(
        self,
        query: str,
        conversation_id: str,
        top_k: int = 10,
        exclude_indices: set[int] | None = None,
    ) -> list[Segment]:
        query_emb = self.embed_text(query)
        result = self.store.search(
            query_emb,
            top_k=top_k,
            conversation_id=conversation_id,
            exclude_indices=exclude_indices,
        )
        return list(result.segments)

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_segments(
    segments: list[Segment],
    max_items: int = 16,
    max_chars: int = 300,
) -> str:
    """Format segments chronologically for display in prompts."""
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


@dataclass
class TaskExecResult:
    """Result from task execution."""

    segments: list[Segment]
    output_text: str  # The generated task output
    metadata: dict = field(default_factory=dict)


# ===========================================================================
# Architecture 1: Generate-and-Check
# ===========================================================================

GENERATE_AND_CHECK_PROMPT = """\
You are completing a task using information from memory. Retrieved memories \
from past conversations are shown below.

TASK: {task}

RETRIEVED MEMORIES:
{formatted_segments}

OUTPUT SO FAR:
{output_so_far}

Continue the output. If you need specific information that isn't in the \
retrieved memories and you're not sure about it, write NEED: followed by a \
short natural-language description of what you need. This will trigger a \
memory search.

CRITICAL RULES:
- Do NOT guess or make up facts. If you're unsure about a specific detail \
(a name, date, number, preference, restriction), use NEED instead of guessing.
- The NEED query will be embedded and compared against stored conversation \
turns via cosine similarity. Write the query as natural text that would \
APPEAR in the stored content, NOT as a search command or boolean query.
- Good NEED example: "Bob mentioned he's allergic to peanuts and has been \
since childhood"
- Bad NEED example: "Search for Bob's allergies" or "allergy OR restriction \
OR dietary"
- You may write some output BEFORE a NEED line if you're confident about \
that part.
- Only use NEED when you genuinely need information you don't have. Do not \
use it speculatively.
- If the task involves multiple items or people, think about whether you \
have complete information for ALL of them before finishing.

Write naturally. When you have enough information, just write the complete \
output without any NEED lines."""


class GenerateAndCheck(TaskExecBase):
    """Architecture 1: Model generates output, emits NEED: when info is missing."""

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        max_rounds: int = 5,
        top_k_per_retrieve: int = 10,
    ):
        super().__init__(store, client)
        self.max_rounds = max_rounds
        self.top_k_per_retrieve = top_k_per_retrieve

    def execute(self, task: str, conversation_id: str) -> TaskExecResult:
        """Execute a task with generate-and-check pattern."""
        output_parts: list[str] = []
        all_segments: list[Segment] = []
        exclude_indices: set[int] = set()
        need_queries: list[str] = []
        round_details: list[dict] = []

        # Initial retrieval using task text
        initial_segments = self.retrieve_top_k(
            task,
            conversation_id,
            top_k=self.top_k_per_retrieve,
            exclude_indices=exclude_indices,
        )
        all_segments.extend(initial_segments)
        exclude_indices.update(s.index for s in initial_segments)

        for round_num in range(self.max_rounds):
            # Build prompt
            formatted = format_segments(all_segments) if all_segments else "(none yet)"
            output_so_far = "\n".join(output_parts) if output_parts else "(nothing yet)"

            prompt = GENERATE_AND_CHECK_PROMPT.format(
                task=task,
                formatted_segments=formatted,
                output_so_far=output_so_far,
            )

            response = self.llm_call(prompt)

            # Check for NEED: lines
            if "NEED:" in response:
                # Split at first NEED:
                parts = response.split("NEED:", 1)
                pre_need = parts[0].strip()
                need_query = parts[1].strip()

                # Take text before NEED as output (if any)
                if pre_need:
                    output_parts.append(pre_need)

                # Clean the need query (take first line only)
                need_query = need_query.split("\n")[0].strip()
                need_queries.append(need_query)

                # Retrieve for this need
                new_segments = self.retrieve_top_k(
                    need_query,
                    conversation_id,
                    top_k=self.top_k_per_retrieve,
                    exclude_indices=exclude_indices,
                )
                all_segments.extend(new_segments)
                exclude_indices.update(s.index for s in new_segments)

                round_details.append(
                    {
                        "round": round_num,
                        "action": "NEED",
                        "need_query": need_query,
                        "pre_need_text": pre_need[:200] if pre_need else "",
                        "new_segments_count": len(new_segments),
                    }
                )
            else:
                # Model completed without needing more info
                output_parts.append(response)
                round_details.append(
                    {
                        "round": round_num,
                        "action": "COMPLETE",
                        "output_length": len(response),
                    }
                )
                break

        final_output = "\n".join(output_parts)

        return TaskExecResult(
            segments=all_segments,
            output_text=final_output,
            metadata={
                "rounds": len(round_details),
                "need_queries": need_queries,
                "round_details": round_details,
                "output_preview": final_output[:500],
            },
        )


# ===========================================================================
# Architecture 4: Autonomous with Tools
# ===========================================================================

AUTONOMOUS_PROMPT = """\
You are executing a task. You have access to a memory store containing past \
conversations with a user.

TASK: {task}

SCRATCHPAD (your private notes):
{scratchpad}

RETRIEVED MEMORIES:
{formatted_segments}

OUTPUT SO FAR:
{output_so_far}

Available actions (choose exactly ONE per step):
- THINK: <private reasoning> -- plan your approach, assess what you know, \
identify what's missing. Not shown to user. Use this freely to strategize \
before retrieving.
- RETRIEVE: <search query> -- search memory for specific information. Your \
query will be embedded and compared against stored conversation turns via \
cosine similarity. Write the query as NATURAL TEXT that would appear in the \
stored content, NOT as a search command. Good: "Bob is allergic to peanuts \
and has been since childhood." Bad: "Bob allergies" or "dietary restrictions \
OR allergies OR preferences."
- WRITE: <output text> -- add to the task output. Only write when you have \
retrieved information to support it. Do not guess facts.
- DONE -- task is complete. Use this when your output fully addresses the task.

STRATEGY GUIDANCE:
- Start with THINK to plan what information you'll need for this task.
- Before writing anything, RETRIEVE information about the specific people, \
topics, or items involved.
- If the task involves a specific person, retrieve their relevant details \
(preferences, restrictions, history) before writing.
- If the task involves multiple items or a list, keep retrieving until you \
are confident you have ALL of them. Don't stop early.
- Each RETRIEVE query should target ONE specific topic. Use separate \
RETRIEVE calls for different topics.
- After several retrieves, THINK again to assess: do I have everything I need?

Step {step}/{max_steps}. Choose your next action:
ACTION: <THINK|RETRIEVE|WRITE|DONE>
ARGUMENT:
<text>"""


class Autonomous(TaskExecBase):
    """Architecture 4: Model has full control over THINK/RETRIEVE/WRITE/DONE."""

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        max_steps: int = 15,
        top_k_per_retrieve: int = 10,
    ):
        super().__init__(store, client)
        self.max_steps = max_steps
        self.top_k_per_retrieve = top_k_per_retrieve

    def _parse_action(self, response: str) -> tuple[str, str]:
        """Parse ACTION: and ARGUMENT: from response."""
        response = response.strip()

        # Try to find ACTION: line
        action_match = re.search(
            r"^ACTION:\s*(THINK|RETRIEVE|WRITE|DONE)",
            response,
            re.MULTILINE | re.IGNORECASE,
        )
        if not action_match:
            # Fallback: check if response starts with an action keyword
            for action in ("THINK:", "RETRIEVE:", "WRITE:", "DONE"):
                if response.upper().startswith(action):
                    action_name = action.rstrip(":")
                    argument = response[len(action) :].strip()
                    return action_name, argument
            # Default to THINK if unparseable
            return "THINK", response

        action_name = action_match.group(1).upper()

        # Extract argument - everything after ARGUMENT: or after ACTION line
        arg_match = re.search(
            r"ARGUMENT:\s*(.*)",
            response[action_match.end() :],
            re.DOTALL,
        )
        if arg_match:
            argument = arg_match.group(1).strip()
        else:
            # Take everything after the ACTION line
            rest = response[action_match.end() :].strip()
            argument = rest

        return action_name, argument

    def execute(self, task: str, conversation_id: str) -> TaskExecResult:
        """Execute a task with autonomous agent pattern."""
        scratchpad: list[str] = []
        output_parts: list[str] = []
        all_segments: list[Segment] = []
        exclude_indices: set[int] = set()
        step_details: list[dict] = []
        retrieve_queries: list[str] = []

        for step in range(1, self.max_steps + 1):
            # Build prompt
            scratchpad_text = (
                "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(scratchpad))
                if scratchpad
                else "(empty)"
            )
            formatted = format_segments(all_segments) if all_segments else "(none yet)"
            output_so_far = "\n".join(output_parts) if output_parts else "(nothing yet)"

            prompt = AUTONOMOUS_PROMPT.format(
                task=task,
                scratchpad=scratchpad_text,
                formatted_segments=formatted,
                output_so_far=output_so_far,
                step=step,
                max_steps=self.max_steps,
            )

            response = self.llm_call(prompt)
            action, argument = self._parse_action(response)

            if action == "THINK":
                scratchpad.append(argument[:500])
                step_details.append(
                    {
                        "step": step,
                        "action": "THINK",
                        "content": argument[:200],
                    }
                )

            elif action == "RETRIEVE":
                retrieve_queries.append(argument)
                new_segments = self.retrieve_top_k(
                    argument,
                    conversation_id,
                    top_k=self.top_k_per_retrieve,
                    exclude_indices=exclude_indices,
                )
                all_segments.extend(new_segments)
                exclude_indices.update(s.index for s in new_segments)
                step_details.append(
                    {
                        "step": step,
                        "action": "RETRIEVE",
                        "query": argument[:200],
                        "new_segments_count": len(new_segments),
                    }
                )

            elif action == "WRITE":
                output_parts.append(argument)
                step_details.append(
                    {
                        "step": step,
                        "action": "WRITE",
                        "content_length": len(argument),
                    }
                )

            elif action == "DONE":
                step_details.append({"step": step, "action": "DONE"})
                break

        final_output = "\n".join(output_parts)

        return TaskExecResult(
            segments=all_segments,
            output_text=final_output,
            metadata={
                "steps": len(step_details),
                "retrieve_queries": retrieve_queries,
                "think_count": sum(1 for d in step_details if d["action"] == "THINK"),
                "retrieve_count": sum(
                    1 for d in step_details if d["action"] == "RETRIEVE"
                ),
                "write_count": sum(1 for d in step_details if d["action"] == "WRITE"),
                "step_details": step_details,
                "output_preview": final_output[:500],
            },
        )


# ===========================================================================
# Architecture 1b: Generate-and-Check v2 (retrieve-first, skeptical)
# ===========================================================================

GENERATE_AND_CHECK_V2_PROMPT = """\
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


class GenerateAndCheckV2(TaskExecBase):
    """Architecture 1b: Retrieve-first, skeptical generation."""

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        max_rounds: int = 6,
        top_k_per_retrieve: int = 10,
    ):
        super().__init__(store, client)
        self.max_rounds = max_rounds
        self.top_k_per_retrieve = top_k_per_retrieve

    def execute(self, task: str, conversation_id: str) -> TaskExecResult:
        """Execute with retrieve-first, skeptical approach."""
        all_segments: list[Segment] = []
        exclude_indices: set[int] = set()
        need_queries: list[str] = []
        round_details: list[dict] = []

        # Initial retrieval using task text
        initial_segments = self.retrieve_top_k(
            task,
            conversation_id,
            top_k=self.top_k_per_retrieve,
            exclude_indices=exclude_indices,
        )
        all_segments.extend(initial_segments)
        exclude_indices.update(s.index for s in initial_segments)

        final_output = ""

        for round_num in range(self.max_rounds):
            formatted = format_segments(all_segments) if all_segments else "(none yet)"

            # Add completeness hint based on round number
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

            prompt = GENERATE_AND_CHECK_V2_PROMPT.format(
                task=task,
                formatted_segments=formatted,
                completeness_hint=completeness_hint,
            )

            response = self.llm_call(prompt)

            if "NEED:" in response:
                need_query = response.split("NEED:", 1)[1].strip()
                need_query = need_query.split("\n")[0].strip()
                need_queries.append(need_query)

                new_segments = self.retrieve_top_k(
                    need_query,
                    conversation_id,
                    top_k=self.top_k_per_retrieve,
                    exclude_indices=exclude_indices,
                )
                all_segments.extend(new_segments)
                exclude_indices.update(s.index for s in new_segments)

                round_details.append(
                    {
                        "round": round_num,
                        "action": "NEED",
                        "need_query": need_query,
                        "new_segments_count": len(new_segments),
                    }
                )
            else:
                final_output = response
                round_details.append(
                    {
                        "round": round_num,
                        "action": "COMPLETE",
                        "output_length": len(response),
                    }
                )
                break

        return TaskExecResult(
            segments=all_segments,
            output_text=final_output,
            metadata={
                "rounds": len(round_details),
                "need_queries": need_queries,
                "round_details": round_details,
                "output_preview": final_output[:500],
            },
        )


# ===========================================================================
# Architecture 4b: Autonomous v2 (tighter, fewer thinks, diverse queries)
# ===========================================================================

AUTONOMOUS_V2_PROMPT = """\
You are executing a task using a memory store of past conversations.

TASK: {task}

RETRIEVED MEMORIES ({num_retrieved} segments):
{formatted_segments}

QUERIES USED SO FAR: {queries_so_far}

OUTPUT WRITTEN SO FAR:
{output_so_far}

Choose ONE action:

RETRIEVE: <query text>
  Search memory. The query is embedded and matched via cosine similarity. \
Write NATURAL TEXT that would appear in the conversation, not search commands.
  Good: "I'm allergic to peanuts, have been since I was a kid"
  Bad: "peanut allergy Bob" or "search for allergies"
  Each query should target a DIFFERENT topic from previous queries.

WRITE: <text>
  Add to the output. Only write facts found in retrieved memories.

DONE
  Task is complete. Only use after you have written output.

GUIDELINES:
- Start by retrieving information relevant to the task.
- If the task mentions a person, retrieve their details before writing.
- If the task requires a list or covers multiple topics, retrieve for EACH \
topic separately. Use different, specific queries each time.
- After 2-3 retrieves, assess what you have and what's missing.
- Do not repeat queries. Each retrieve should target new information.
- You MUST write output before saying DONE.

Step {step}/{max_steps}."""


class AutonomousV2(TaskExecBase):
    """Architecture 4b: Tighter autonomous agent with less thinking."""

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        max_steps: int = 12,
        top_k_per_retrieve: int = 10,
    ):
        super().__init__(store, client)
        self.max_steps = max_steps
        self.top_k_per_retrieve = top_k_per_retrieve

    def _parse_action(self, response: str) -> tuple[str, str]:
        """Parse action from response."""
        response = response.strip()

        # Check for each action prefix
        for prefix in ("RETRIEVE:", "WRITE:", "DONE"):
            if response.upper().startswith(prefix):
                action = prefix.rstrip(":")
                argument = response[len(prefix) :].strip()
                return action, argument

        # Try to find action in response
        for prefix in ("RETRIEVE:", "WRITE:", "DONE"):
            match = re.search(
                rf"^{prefix}",
                response,
                re.MULTILINE | re.IGNORECASE,
            )
            if match:
                action = prefix.rstrip(":")
                argument = response[match.end() :].strip()
                # For multi-line WRITE, take everything after
                return action, argument

        # Default: treat as retrieve if it looks like a query
        if len(response) < 200:
            return "RETRIEVE", response
        return "WRITE", response

    def execute(self, task: str, conversation_id: str) -> TaskExecResult:
        """Execute task with tight autonomous pattern."""
        output_parts: list[str] = []
        all_segments: list[Segment] = []
        exclude_indices: set[int] = set()
        step_details: list[dict] = []
        retrieve_queries: list[str] = []

        for step in range(1, self.max_steps + 1):
            formatted = (
                format_segments(all_segments, max_items=20)
                if all_segments
                else "(none yet)"
            )
            output_so_far = "\n".join(output_parts) if output_parts else "(nothing yet)"
            queries_so_far = (
                "; ".join(f'"{q[:80]}"' for q in retrieve_queries)
                if retrieve_queries
                else "(none)"
            )

            prompt = AUTONOMOUS_V2_PROMPT.format(
                task=task,
                num_retrieved=len(all_segments),
                formatted_segments=formatted,
                queries_so_far=queries_so_far,
                output_so_far=output_so_far,
                step=step,
                max_steps=self.max_steps,
            )

            response = self.llm_call(prompt)
            action, argument = self._parse_action(response)

            if action == "RETRIEVE":
                retrieve_queries.append(argument)
                new_segments = self.retrieve_top_k(
                    argument,
                    conversation_id,
                    top_k=self.top_k_per_retrieve,
                    exclude_indices=exclude_indices,
                )
                all_segments.extend(new_segments)
                exclude_indices.update(s.index for s in new_segments)
                step_details.append(
                    {
                        "step": step,
                        "action": "RETRIEVE",
                        "query": argument[:200],
                        "new_segments_count": len(new_segments),
                    }
                )

            elif action == "WRITE":
                output_parts.append(argument)
                step_details.append(
                    {
                        "step": step,
                        "action": "WRITE",
                        "content_length": len(argument),
                    }
                )

            elif action == "DONE":
                step_details.append({"step": step, "action": "DONE"})
                break

        final_output = "\n".join(output_parts)

        return TaskExecResult(
            segments=all_segments,
            output_text=final_output,
            metadata={
                "steps": len(step_details),
                "retrieve_queries": retrieve_queries,
                "retrieve_count": sum(
                    1 for d in step_details if d["action"] == "RETRIEVE"
                ),
                "write_count": sum(1 for d in step_details if d["action"] == "WRITE"),
                "step_details": step_details,
                "output_preview": final_output[:500],
            },
        )


# ===========================================================================
# Architecture 5: Decompose-then-Retrieve
# ===========================================================================

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


class DecomposeThenRetrieve(TaskExecBase):
    """Architecture 5: Plan retrieval, execute, refine."""

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        max_refine_rounds: int = 2,
        top_k_per_query: int = 5,
    ):
        super().__init__(store, client)
        self.max_refine_rounds = max_refine_rounds
        self.top_k_per_query = top_k_per_query

    def _parse_queries(self, response: str) -> list[str]:
        """Parse QUERY: lines from response."""
        queries = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("QUERY:"):
                q = line[6:].strip()
                if q:
                    queries.append(q)
        return queries

    def execute(self, task: str, conversation_id: str) -> TaskExecResult:
        """Decompose task into queries, retrieve, refine."""
        all_segments: list[Segment] = []
        exclude_indices: set[int] = set()
        all_queries: list[str] = []
        round_details: list[dict] = []

        # Step 1: Decompose task into queries
        decompose_prompt = DECOMPOSE_PROMPT.format(task=task)
        decompose_response = self.llm_call(decompose_prompt)
        initial_queries = self._parse_queries(decompose_response)

        # Also include the raw task as a query
        all_query_list = [task] + initial_queries

        # Step 2: Retrieve for each query
        for query in all_query_list:
            new_segments = self.retrieve_top_k(
                query,
                conversation_id,
                top_k=self.top_k_per_query,
                exclude_indices=exclude_indices,
            )
            all_segments.extend(new_segments)
            exclude_indices.update(s.index for s in new_segments)
            all_queries.append(query)

        round_details.append(
            {
                "round": 0,
                "action": "DECOMPOSE",
                "num_queries": len(all_query_list),
                "queries": [q[:100] for q in all_query_list],
                "total_segments": len(all_segments),
            }
        )

        # Step 3: Refine -- look at what we have, identify gaps
        for refine_round in range(self.max_refine_rounds):
            formatted = format_segments(all_segments, max_items=20)
            queries_so_far = "\n".join(f"- {q[:100]}" for q in all_queries)

            refine_prompt = DECOMPOSE_REFINE_PROMPT.format(
                task=task,
                num_retrieved=len(all_segments),
                formatted_segments=formatted,
                queries_so_far=queries_so_far,
            )

            refine_response = self.llm_call(refine_prompt)

            if (
                "COMPLETE" in refine_response.upper()
                and "QUERY:" not in refine_response
            ):
                round_details.append(
                    {
                        "round": refine_round + 1,
                        "action": "COMPLETE",
                    }
                )
                break

            refine_queries = self._parse_queries(refine_response)
            if not refine_queries:
                break

            for query in refine_queries:
                new_segments = self.retrieve_top_k(
                    query,
                    conversation_id,
                    top_k=self.top_k_per_query,
                    exclude_indices=exclude_indices,
                )
                all_segments.extend(new_segments)
                exclude_indices.update(s.index for s in new_segments)
                all_queries.append(query)

            round_details.append(
                {
                    "round": refine_round + 1,
                    "action": "REFINE",
                    "num_queries": len(refine_queries),
                    "queries": [q[:100] for q in refine_queries],
                    "total_segments": len(all_segments),
                }
            )

        return TaskExecResult(
            segments=all_segments,
            output_text="",  # Focus on retrieval quality
            metadata={
                "rounds": len(round_details),
                "all_queries": all_queries,
                "num_queries": len(all_queries),
                "round_details": round_details,
                "output_preview": "",
            },
        )


# ===========================================================================
# V2f Baseline (using task text as the "question")
# ===========================================================================

V2F_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Task: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the task implies MULTIPLE items or asks for a comprehensive list, keep \
searching for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


class V2fBaseline(TaskExecBase):
    """V2f baseline: treat task text as a question, run standard V2f."""

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
    ):
        super().__init__(store, client)

    def execute(self, task: str, conversation_id: str) -> TaskExecResult:
        """Run V2f with task text as question."""
        # Hop 0: embed task, retrieve top-10
        initial_segments = self.retrieve_top_k(task, conversation_id, top_k=10)
        all_segments = list(initial_segments)
        exclude_indices = {s.index for s in all_segments}

        # Build context section
        context = format_segments(all_segments, max_items=12)
        context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + context

        # Single LLM call to generate cues
        prompt = V2F_PROMPT.format(question=task, context_section=context_section)
        output = self.llm_call(prompt)

        # Parse cues
        cues = []
        for line in output.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    cues.append(cue)

        # Retrieve with each cue
        for cue in cues[:2]:
            new_segments = self.retrieve_top_k(
                cue,
                conversation_id,
                top_k=10,
                exclude_indices=exclude_indices,
            )
            for seg in new_segments:
                if seg.index not in exclude_indices:
                    all_segments.append(seg)
                    exclude_indices.add(seg.index)

        return TaskExecResult(
            segments=all_segments,
            output_text="",  # V2f doesn't generate task output
            metadata={
                "llm_output": output,
                "cues": cues[:2],
            },
        )


# ===========================================================================
# Evaluation
# ===========================================================================
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    arch: TaskExecBase,
    question: dict,
    verbose: bool = False,
) -> dict:
    """Evaluate a single architecture on a single question."""
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.execute(q_text, conv_id)
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

    # Baseline: cosine top-N at same budgets
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
        retrieved_turn_ids = sorted(s.turn_id for s in arch_segments)
        hit_ids = sorted(source_ids & set(retrieved_turn_ids))
        miss_ids = sorted(source_ids - set(retrieved_turn_ids))
        print(f"  Hits: {hit_ids}")
        print(f"  Misses: {miss_ids}")
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

        # Show architecture-specific details
        meta = result.metadata
        if "need_queries" in meta:
            for nq in meta["need_queries"][:5]:
                print(f"    NEED: {nq[:120]}")
        elif "retrieve_queries" in meta:
            for rq in meta["retrieve_queries"][:5]:
                print(f"    RETRIEVE: {rq[:120]}")
        elif "cues" in meta:
            for cue in meta["cues"][:4]:
                print(f"    CUE: {cue[:120]}")

        # Show output preview
        if result.output_text:
            preview = result.output_text[:300].replace("\n", " | ")
            print(f"    Output: {preview}")

    return row


def summarize(results: list[dict], variant_name: str) -> dict:
    """Compute summary statistics."""
    n = len(results)
    if n == 0:
        return {}

    summary: dict = {"variant": variant_name, "n": n}

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
    summary["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 1)
    summary["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
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


def per_question_comparison(
    results_by_arch: dict[str, list[dict]],
) -> None:
    """Print per-question comparison across all architectures."""
    # Get all question indices
    q_indices = set()
    for results in results_by_arch.values():
        for r in results:
            q_indices.add(r["question_index"])

    print(f"\n{'=' * 90}")
    print("PER-QUESTION COMPARISON (r@20)")
    print(f"{'=' * 90}")
    print(
        f"{'Q#':>3s} {'Category':>12s} {'#src':>4s} {'baseline':>8s}",
        end="",
    )
    for arch_name in results_by_arch:
        print(f" {arch_name:>14s}", end="")
    print()
    print("-" * 90)

    for qi in sorted(q_indices):
        # Get baseline from first arch
        first_arch = list(results_by_arch.values())[0]
        q_row = next(r for r in first_arch if r["question_index"] == qi)

        bl = q_row["baseline_recalls"]["r@20"]
        cat = q_row["category"]
        n_src = q_row["num_source_turns"]

        print(f"{qi:3d} {cat:>12s} {n_src:4d} {bl:8.3f}", end="")

        for arch_name, results in results_by_arch.items():
            r = next(r for r in results if r["question_index"] == qi)
            ar = r["arch_recalls"]["r@20"]
            delta = ar - bl
            marker = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
            print(f" {ar:8.3f}{marker:>1s}({delta:+.2f})", end="")
        print()


def run_variant(
    variant_name: str,
    arch: TaskExecBase,
    questions: list[dict],
    verbose: bool = False,
) -> list[dict]:
    """Run one variant on all questions, return results."""
    print(f"\n{'=' * 70}")
    print(f"ARCHITECTURE: {variant_name} | {len(questions)} questions")
    print(f"{'=' * 70}")

    results = []
    for i, question in enumerate(questions):
        q_short = question["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] Q{question['question_index']} "
            f"[{question['category']}]: {q_short}...",
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
        if (i + 1) % 3 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = summarize(results, variant_name)

    # Print compact summary
    print(f"\n--- {variant_name} summary ---")
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
    print("\n  Per-category (r@20):")
    for cat, cs in cat_summaries.items():
        print(
            f"    {cat}: baseline={cs['baseline_r@20']:.3f} "
            f"arch={cs['arch_r@20']:.3f} "
            f"delta={cs['delta_r@20']:+.3f} "
            f"W/T/L={cs['W/T/L']} (n={cs['n']})"
        )

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Task-execution retrieval evaluation")
    parser.add_argument(
        "--arch",
        type=str,
        choices=[
            "generate_and_check",
            "generate_and_check_v2",
            "autonomous",
            "autonomous_v2",
            "v2f_baseline",
            "decompose",
        ],
        default=None,
        help="Which architecture to run",
    )
    parser.add_argument("--all", action="store_true", help="Run all architectures")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing")
    parser.add_argument(
        "--model", type=str, default=MODEL, help=f"LLM model (default: {MODEL})"
    )
    args = parser.parse_args()

    # Load data
    with open(DATA_DIR / "questions_synthetic.json") as f:
        all_questions = json.load(f)

    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_synthetic.npz")
    print(f"Loaded {len(store.segments)} segments")

    # Filter to target questions
    target_questions = [
        q for q in all_questions if q["question_index"] in TARGET_INDICES
    ]
    print(
        f"Target questions: {len(target_questions)} "
        f"(categories: proactive, procedural, completeness)"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    architectures_to_run = []
    if args.all:
        architectures_to_run = [
            "v2f_baseline",
            "generate_and_check",
            "generate_and_check_v2",
            "autonomous",
            "autonomous_v2",
            "decompose",
        ]
    elif args.arch:
        architectures_to_run = [args.arch]
    else:
        architectures_to_run = ["v2f_baseline", "generate_and_check", "autonomous"]

    all_results: dict[str, list[dict]] = {}

    for arch_name in architectures_to_run:
        results_file = RESULTS_DIR / f"task_exec_{arch_name}.json"

        if results_file.exists() and not args.force:
            print(f"\nLoading existing {arch_name} from {results_file}")
            with open(results_file) as f:
                results = json.load(f)
            all_results[arch_name] = results
            # Print summary
            summary = summarize(results, arch_name)
            print(
                f"  r@20: baseline={summary.get('baseline_r@20', 0):.3f} "
                f"arch={summary.get('arch_r@20', 0):.3f} "
                f"delta={summary.get('delta_r@20', 0):+.3f}"
            )
            continue

        # Create architecture
        if arch_name == "v2f_baseline":
            arch = V2fBaseline(store)
        elif arch_name == "generate_and_check":
            arch = GenerateAndCheck(store, max_rounds=5, top_k_per_retrieve=10)
        elif arch_name == "generate_and_check_v2":
            arch = GenerateAndCheckV2(store, max_rounds=6, top_k_per_retrieve=10)
        elif arch_name == "autonomous":
            arch = Autonomous(store, max_steps=15, top_k_per_retrieve=10)
        elif arch_name == "autonomous_v2":
            arch = AutonomousV2(store, max_steps=12, top_k_per_retrieve=10)
        elif arch_name == "decompose":
            arch = DecomposeThenRetrieve(store, max_refine_rounds=2, top_k_per_query=5)
        else:
            raise ValueError(f"Unknown architecture: {arch_name}")

        results = run_variant(arch_name, arch, target_questions, verbose=args.verbose)

        # Save results
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved to {results_file}")

        all_results[arch_name] = results

    # Cross-architecture comparison
    if len(all_results) > 1:
        per_question_comparison(all_results)

        # Overall comparison
        print(f"\n{'=' * 70}")
        print("OVERALL COMPARISON")
        print(f"{'=' * 70}")
        for arch_name, results in all_results.items():
            summary = summarize(results, arch_name)
            print(
                f"\n  {arch_name}:"
                f"\n    r@20:  baseline={summary.get('baseline_r@20', 0):.3f} "
                f"arch={summary.get('arch_r@20', 0):.3f} "
                f"delta={summary.get('delta_r@20', 0):+.3f} "
                f"W/T/L={summary.get('W/T/L_r@20', '?')}"
                f"\n    r@50:  baseline={summary.get('baseline_r@50', 0):.3f} "
                f"arch={summary.get('arch_r@50', 0):.3f} "
                f"delta={summary.get('delta_r@50', 0):+.3f}"
                f"\n    Avg retrieved: {summary.get('avg_total_retrieved', 0):.0f}, "
                f"Embed: {summary.get('avg_embed_calls', 0):.1f}, "
                f"LLM: {summary.get('avg_llm_calls', 0):.1f}"
            )


if __name__ == "__main__":
    main()
