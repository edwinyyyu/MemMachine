"""Goal-focused sequential chain retrieval with dead-end detection.

THEORY
------
Humans handling A -> B -> C -> D chain retrieval don't fan out. They:
  1. Commit to a single promising direction
  2. Follow that thread until it works or dead-ends
  3. Backtrack on felt dead-end
  4. Maintain goal focus across retrievals

Current architectures all FAN OUT: generate 2+ cues per step from the ORIGINAL
question. No committed depth-first threading.

ARCHITECTURE
------------
Sequential chain retrieval with goal tracking and dead-end detection.
  Round 0:  initial retrieve with question (initial_k segments)
  Round 1..N (max max_rounds):
    LLM reads: question + all retrieved + most recent retrieval
    LLM decides CONTINUE | DEAD_END | DONE
      - CONTINUE: generate ONE cue following the hottest thread toward the goal
      - DEAD_END: generate ONE cue from the QUESTION (different angle)
      - DONE: stop
    Retrieve with ONE cue (per_round_k segments)

Budget: 10 initial + 2 per round x up to 5 rounds = 20 segments max pool.

VARIANTS
--------
A. chain_goal_tracking      - base architecture
B. chain_with_scratchpad    - LLM carries a progress scratchpad across rounds

EVALUATION
----------
ALL 4 datasets (locomo_30q, synthetic_19q, puzzle_16q, advanced_23q) at K=20.
Fair backfill. Per-category breakdown.

Usage:
    uv run python goal_chain.py [--variant A|B|both] [--dataset NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
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

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_FILE_EMB = CACHE_DIR / "goal_chain_embedding_cache.json"
CACHE_FILE_LLM = CACHE_DIR / "goal_chain_llm_cache.json"
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Caches — read from existing caches, write to goal_chain-specific file
# ---------------------------------------------------------------------------
class GoalChainEmbeddingCache(EmbeddingCache):
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
            "bestshot_embedding_cache.json",
            "optim_embedding_cache.json",
            "chain_embedding_cache.json",
            "goal_chain_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                try:
                    with open(p) as f:
                        self._cache.update(json.load(f))
                except Exception:
                    continue
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
            except Exception:
                existing = {}
        existing.update(self._new)
        tmp = self.cache_file.with_suffix(
            f".json.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new = {}


class GoalChainLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in (
            "llm_cache.json",
            "arch_llm_cache.json",
            "agent_llm_cache.json",
            "frontier_llm_cache.json",
            "meta_llm_cache.json",
            "bestshot_llm_cache.json",
            "optim_llm_cache.json",
            "chain_llm_cache.json",
            "goal_chain_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                try:
                    with open(p) as f:
                        data = json.load(f)
                    for k, v in data.items():
                        if v:
                            self._cache[k] = v
                except Exception:
                    continue
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
            except Exception:
                existing = {}
        existing.update(self._new)
        tmp = self.cache_file.with_suffix(
            f".json.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new = {}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
@dataclass
class GoalChainResult:
    segments: list[Segment]
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


class GoalChainBase:
    def __init__(self, store: SegmentStore, client: OpenAI | None = None) -> None:
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = GoalChainEmbeddingCache()
        self.llm_cache = GoalChainLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

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

    def retrieve(self, question: str, conversation_id: str) -> GoalChainResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_segments(
    segments: list[Segment],
    max_items: int = 14,
    max_chars: int = 250,
) -> str:
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in sorted_segs
    )


def _format_segments_recent(segments: list[Segment], max_chars: int = 250) -> str:
    # Most-recently retrieved segments, sorted by retrieval order (not turn_id).
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in segments
    )


# ===========================================================================
# VARIANT A: chain_goal_tracking
# ===========================================================================
GOAL_CHAIN_PROMPT_A = """\
You are performing sequential thread-following retrieval over a conversation.
Each search cue you generate will be embedded and matched against conversation
turns via cosine similarity. You are NOT fanning out with multiple cues — you
are picking ONE thread and pulling on it.

GOAL (the question you must answer):
{question}

WHAT I'VE FOUND SO FAR (across all rounds, {num_total} segments):
{all_segs}

MOST RECENT RETRIEVAL (this is the current thread — round {round_num}):
{recent_segs}

ALREADY SEARCHED FOR (do NOT repeat):
{explored}

Thinking (internally):
1. Am I making progress toward the GOAL?
   - "Getting warmer": the current thread relates to the answer — follow it
   - "Dead-end": the thread isn't productive — try a different angle on the goal
   - "Done": I have enough information to answer the goal
2. If continuing: what's the most specific vocabulary/entity (noun, name, tool,
   symptom, decision, ticket, alias) in the RECENT RETRIEVAL that would lead
   one step CLOSER to the goal? Use ONE concrete phrase.
3. If dead-ending: what's a DIFFERENT angle on the goal I haven't tried?

Output exactly this format — no extra commentary:
ACTION: CONTINUE | DEAD_END | DONE
CUE: <if CONTINUE, ONE short (1-2 sentence) cue following the hottest thread in the RECENT retrieval; if DEAD_END, ONE cue from a DIFFERENT angle on the GOAL; if DONE, empty>
REASONING: <one sentence>
Nothing else."""


# ===========================================================================
# VARIANT B: chain_with_scratchpad
# ===========================================================================
GOAL_CHAIN_PROMPT_B = """\
You are performing sequential thread-following retrieval over a conversation.
Each search cue you generate will be embedded and matched against conversation
turns via cosine similarity. You are NOT fanning out with multiple cues — you
are picking ONE thread and pulling on it. You keep a SCRATCHPAD of your
progress toward the goal across rounds.

GOAL (the question you must answer):
{question}

PRIOR SCRATCHPAD (what you knew before this round):
{scratchpad}

WHAT I'VE FOUND SO FAR ({num_total} segments):
{all_segs}

MOST RECENT RETRIEVAL (this is the current thread — round {round_num}):
{recent_segs}

ALREADY SEARCHED FOR (do NOT repeat):
{explored}

Thinking:
1. UPDATE the scratchpad: what do I now know toward the goal? What's still
   missing? Is the current thread productive ("getting warmer"), a dead-end,
   or have I collected enough to answer ("done")?
2. If continuing: what's the most specific vocabulary/entity in the RECENT
   retrieval that would lead one step CLOSER to the answer?
3. If dead-ending: what's a DIFFERENT angle on the goal I haven't tried?

Output exactly this format — no extra commentary:
SCRATCHPAD: <2-3 sentences: what I know toward the goal, what's still missing>
ACTION: CONTINUE | DEAD_END | DONE
CUE: <if CONTINUE, ONE short (1-2 sentence) cue following the hottest thread; if DEAD_END, ONE cue from a DIFFERENT angle on the GOAL; if DONE, empty>
REASONING: <one sentence>
Nothing else."""


def _parse_decision(text: str) -> dict:
    """Parse ACTION / CUE / REASONING / SCRATCHPAD from LLM response."""
    out = {"action": "", "cue": "", "reasoning": "", "scratchpad": ""}
    lines = text.strip().split("\n")
    current_key = None
    # Single-line fields; collect multi-line cue/scratchpad values cleanly.
    buf: list[str] = []

    def _flush() -> None:
        nonlocal buf, current_key
        if current_key and buf:
            out[current_key] = (
                out.get(current_key, "") + " " + " ".join(buf).strip()
            ).strip()
        buf = []

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("ACTION:"):
            _flush()
            current_key = "action"
            out["action"] = stripped.split(":", 1)[1].strip().upper()
            current_key = None
        elif upper.startswith("CUE:"):
            _flush()
            current_key = "cue"
            val = stripped.split(":", 1)[1].strip()
            if val:
                buf.append(val)
        elif upper.startswith("REASONING:"):
            _flush()
            current_key = "reasoning"
            val = stripped.split(":", 1)[1].strip()
            if val:
                buf.append(val)
        elif upper.startswith("SCRATCHPAD:"):
            _flush()
            current_key = "scratchpad"
            val = stripped.split(":", 1)[1].strip()
            if val:
                buf.append(val)
        else:
            if current_key and stripped:
                buf.append(stripped)
    _flush()

    # Normalize action token
    action = out["action"].strip().upper()
    # Strip trailing punctuation / pipes
    for tok in ("CONTINUE", "DEAD_END", "DEAD-END", "DEADEND", "DONE"):
        if tok in action:
            if tok in ("DEAD-END", "DEADEND"):
                action = "DEAD_END"
            else:
                action = tok
            break
    else:
        action = ""
    out["action"] = action
    return out


class GoalChainRetriever(GoalChainBase):
    """Sequential chain retrieval with ONE cue per round, dead-end detection,
    and explicit DONE.

    Budget per question: initial_k + per_round_k * max_rounds.
    With defaults (10, 2, 5): up to 20 segment pool slots.
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        initial_k: int = 10,
        per_round_k: int = 2,
        max_rounds: int = 5,
        use_scratchpad: bool = False,
    ) -> None:
        super().__init__(store, client)
        self.initial_k = initial_k
        self.per_round_k = per_round_k
        self.max_rounds = max_rounds
        self.use_scratchpad = use_scratchpad

    def retrieve(self, question: str, conversation_id: str) -> GoalChainResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []
        explored: list[str] = []
        round_log: list[dict] = []
        recent_segs: list[Segment] = []
        scratchpad = "(this is the first round — no prior scratchpad)"

        # Round 0: initial retrieve with question embedding
        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        initial_new: list[Segment] = []
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)
                initial_new.append(s)
        recent_segs = initial_new

        # Rounds 1..max_rounds
        for rnd in range(1, self.max_rounds + 1):
            if self.use_scratchpad:
                prompt = GOAL_CHAIN_PROMPT_B.format(
                    question=question,
                    scratchpad=scratchpad,
                    num_total=len(all_segs),
                    all_segs=_format_segments(all_segs, max_items=14),
                    recent_segs=_format_segments_recent(recent_segs[:8]),
                    round_num=rnd,
                    explored=(
                        "\n".join(f"- {c}" for c in explored)
                        if explored
                        else "(none yet)"
                    ),
                )
            else:
                prompt = GOAL_CHAIN_PROMPT_A.format(
                    question=question,
                    num_total=len(all_segs),
                    all_segs=_format_segments(all_segs, max_items=14),
                    recent_segs=_format_segments_recent(recent_segs[:8]),
                    round_num=rnd,
                    explored=(
                        "\n".join(f"- {c}" for c in explored)
                        if explored
                        else "(none yet)"
                    ),
                )

            response = self.llm_call(prompt)
            parsed = _parse_decision(response)
            action = parsed["action"]
            cue = parsed["cue"].strip()
            reasoning = parsed["reasoning"]
            if self.use_scratchpad and parsed["scratchpad"]:
                scratchpad = parsed["scratchpad"]

            round_log.append(
                {
                    "round": rnd,
                    "action": action,
                    "cue": cue,
                    "reasoning": reasoning,
                    "scratchpad": parsed.get("scratchpad", ""),
                }
            )

            if action == "DONE":
                break
            if not cue:
                # Defensive: missing cue; treat as DONE to avoid infinite empty.
                break
            if cue in explored:
                # Don't re-run; treat as DONE.
                break

            explored.append(cue)
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_round_k,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            new_segs: list[Segment] = []
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)
                    new_segs.append(s)
            recent_segs = new_segs or recent_segs

        name = "chain_with_scratchpad" if self.use_scratchpad else "chain_goal_tracking"
        return GoalChainResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": name,
                "rounds": round_log,
                "total_segments": len(all_segs),
                "explored_cues": explored,
            },
        )


# ===========================================================================
# Evaluation — fair backfill across 4 datasets at K=20
# ===========================================================================
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

VARIANTS = {
    "chain_goal_tracking": {"use_scratchpad": False},
    "chain_with_scratchpad": {"use_scratchpad": True},
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
    return store, questions


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


def evaluate_question(arch: GoalChainBase, question: dict) -> dict:
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
    max_K = max(BUDGETS)
    cosine_result = arch.store.search(query_emb, top_k=max_K, conversation_id=conv_id)
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
    summary["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
    summary["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 1)
    summary["avg_time_s"] = round(sum(r["time_s"] for r in results) / n, 2)
    # Action distribution
    actions = defaultdict(int)
    round_counts: list[int] = []
    done_early = 0
    deadends = 0
    for r in results:
        rounds = r.get("metadata", {}).get("rounds", []) or []
        round_counts.append(len(rounds))
        for rd in rounds:
            a = rd.get("action") or "EMPTY"
            actions[a] += 1
        if any((rd.get("action") == "DONE") for rd in rounds):
            done_early += 1
        if any((rd.get("action") == "DEAD_END") for rd in rounds):
            deadends += 1
    summary["avg_rounds"] = round(sum(round_counts) / max(len(round_counts), 1), 2)
    summary["action_counts"] = dict(actions)
    summary["n_with_done"] = done_early
    summary["n_with_deadend"] = deadends
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


def run_variant_on_dataset(
    variant_name: str,
    ds_name: str,
    store: SegmentStore,
    questions: list[dict],
    save_every: int = 1,
    force: bool = False,
) -> tuple[list[dict], dict, dict]:
    out_path = RESULTS_DIR / f"goal_chain_{variant_name}_{ds_name}.json"
    existing_results: list[dict] = []
    if out_path.exists() and not force:
        with open(out_path) as f:
            doc = json.load(f)
        existing_results = doc.get("results", [])
        # If already complete, skip.
        if len(existing_results) >= len(questions):
            print(
                f"  [cached] {variant_name} on {ds_name}: "
                f"{len(existing_results)} questions"
            )
            summary = doc.get("summary") or summarize(
                existing_results, variant_name, ds_name
            )
            by_cat = doc.get("category_breakdown") or summarize_by_category(
                existing_results
            )
            return existing_results, summary, by_cat

    kwargs = VARIANTS[variant_name]
    arch = GoalChainRetriever(store, **kwargs)

    # Skip questions already in existing_results (keyed by (conv_id, q_idx))
    done_keys = {
        (r["conversation_id"], r.get("question_index", -1), r.get("question"))
        for r in existing_results
    }
    results: list[dict] = list(existing_results)

    print(f"\n{'=' * 70}")
    print(
        f"{variant_name} | {ds_name} | "
        f"{len(questions)} total, {len(existing_results)} done"
    )
    print(f"{'=' * 70}")

    for i, q in enumerate(questions):
        key = (q["conversation_id"], q.get("question_index", -1), q["question"])
        if key in done_keys:
            continue
        q_short = q["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(arch, q)
            results.append(row)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()
        sys.stdout.flush()

        # Save incrementally
        if (i + 1) % save_every == 0:
            arch.save_caches()
            interim_summary = summarize(results, variant_name, ds_name)
            interim_by_cat = summarize_by_category(results)
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "arch": variant_name,
                        "dataset": ds_name,
                        "summary": interim_summary,
                        "category_breakdown": interim_by_cat,
                        "results": results,
                    },
                    f,
                    indent=2,
                    default=str,
                )

    arch.save_caches()
    summary = summarize(results, variant_name, ds_name)
    by_cat = summarize_by_category(results)

    with open(out_path, "w") as f:
        json.dump(
            {
                "arch": variant_name,
                "dataset": ds_name,
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n--- {variant_name} on {ds_name} ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: base={summary[f'baseline_r@{K}']:.3f} "
            f"arch={summary[f'arch_r@{K}']:.3f} "
            f"delta={summary[f'delta_r@{K}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    print(
        f"  avg pool={summary['avg_total_retrieved']:.1f} "
        f"llm={summary['avg_llm_calls']:.1f} "
        f"emb={summary['avg_embed_calls']:.1f} "
        f"rounds={summary['avg_rounds']:.2f}"
    )
    print(f"  action_counts={summary['action_counts']}")
    print(f"  Saved: {out_path}")
    return results, summary, by_cat


def load_reference_baseline(
    ref_tag: str, ds_name: str
) -> tuple[dict | None, dict | None]:
    """Load existing results for v2f (meta_v2f) or cot (chain_of_thought)."""
    if ref_tag == "v2f":
        path = RESULTS_DIR / f"fairbackfill_meta_v2f_{ds_name}.json"
        if not path.exists():
            return None, None
        with open(path) as f:
            doc = json.load(f)
        return doc.get("summary"), doc.get("category_breakdown")
    if ref_tag == "cot":
        # CoT only exists for puzzle_seqchain / advanced_evolterm — use chain results
        mapping = {
            "puzzle_16q": "chain_chain_of_thought_puzzle_seqchain.json",
            "advanced_23q": "chain_chain_of_thought_advanced_evolterm.json",
        }
        fn = mapping.get(ds_name)
        if fn is None:
            return None, None
        path = RESULTS_DIR / fn
        if not path.exists():
            return None, None
        with open(path) as f:
            results = json.load(f)
        # These results use a different schema (arch_recalls/baseline_recalls).
        # Project to fair_backfill shape for comparison.
        projected = []
        for r in results:
            projected.append(
                {
                    "conversation_id": r["conversation_id"],
                    "category": r["category"],
                    "question_index": r["question_index"],
                    "question": r["question"],
                    "num_source_turns": r["num_source_turns"],
                    "total_arch_retrieved": r["total_retrieved"],
                    "embed_calls": r.get("embed_calls", 0),
                    "llm_calls": r.get("llm_calls", 0),
                    "time_s": r.get("time_s", 0),
                    "fair_backfill": {
                        "baseline_r@20": r["baseline_recalls"]["r@20"],
                        "arch_r@20": r["arch_recalls"]["r@20"],
                        "delta_r@20": (
                            r["arch_recalls"]["r@20"] - r["baseline_recalls"]["r@20"]
                        ),
                        "baseline_r@50": r["baseline_recalls"]["r@50"],
                        "arch_r@50": r["arch_recalls"]["r@50"],
                        "delta_r@50": (
                            r["arch_recalls"]["r@50"] - r["baseline_recalls"]["r@50"]
                        ),
                    },
                }
            )
        summary = summarize(projected, "chain_of_thought", ds_name)
        by_cat = summarize_by_category(projected)
        return summary, by_cat
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["A", "B", "both"],
        default="both",
        help="A=chain_goal_tracking, B=chain_with_scratchpad",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS) + ["all"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Variants:")
        for n in VARIANTS:
            print(f"  {n}")
        print("Datasets:")
        for n in DATASETS:
            print(f"  {n}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.variant == "both":
        variant_names = ["chain_goal_tracking", "chain_with_scratchpad"]
    elif args.variant == "A":
        variant_names = ["chain_goal_tracking"]
    else:
        variant_names = ["chain_with_scratchpad"]

    ds_names = list(DATASETS) if args.dataset == "all" else [args.dataset]

    all_summaries: dict = {}
    for ds_name in ds_names:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )
        for variant_name in variant_names:
            _, summary, by_cat = run_variant_on_dataset(
                variant_name, ds_name, store, questions, force=args.force
            )
            all_summaries.setdefault(variant_name, {})[ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
            }

    # Aggregated summary (our variants + reference comparisons)
    summary_path = RESULTS_DIR / "goal_chain_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")

    # ----- Grand table: variants vs v2f vs cot -----
    print("\n" + "=" * 110)
    print("GOAL CHAIN — FAIR BACKFILL SUMMARY (all 4 datasets, K=20 and K=50)")
    print("=" * 110)
    header = (
        f"{'Arch':<24s} {'Dataset':<14s} {'n':>3s}  "
        f"{'B@20':>6s} {'A@20':>6s} {'d@20':>7s}  "
        f"{'B@50':>6s} {'A@50':>6s} {'d@50':>7s}  "
        f"{'LLM':>4s} {'rounds':>6s}"
    )
    print(header)
    print("-" * len(header))
    for variant_name in variant_names:
        for ds_name in ds_names:
            if ds_name not in all_summaries.get(variant_name, {}):
                continue
            s = all_summaries[variant_name][ds_name]["summary"]
            print(
                f"{variant_name:<24s} {ds_name:<14s} {s['n']:>3d}  "
                f"{s['baseline_r@20']:>6.3f} {s['arch_r@20']:>6.3f} "
                f"{s['delta_r@20']:>+7.3f}  "
                f"{s['baseline_r@50']:>6.3f} {s['arch_r@50']:>6.3f} "
                f"{s['delta_r@50']:>+7.3f}  "
                f"{s['avg_llm_calls']:>4.1f} {s.get('avg_rounds', 0):>6.2f}"
            )
    # Reference: v2f
    print("-" * len(header))
    for ds_name in ds_names:
        ref_sum, _ = load_reference_baseline("v2f", ds_name)
        if not ref_sum:
            continue
        print(
            f"{'[ref] meta_v2f':<24s} {ds_name:<14s} {ref_sum['n']:>3d}  "
            f"{ref_sum['baseline_r@20']:>6.3f} {ref_sum['arch_r@20']:>6.3f} "
            f"{ref_sum['delta_r@20']:>+7.3f}  "
            f"{ref_sum['baseline_r@50']:>6.3f} {ref_sum['arch_r@50']:>6.3f} "
            f"{ref_sum['delta_r@50']:>+7.3f}  "
            f"{ref_sum['avg_llm_calls']:>4.1f} {'-':>6s}"
        )
    # Reference: cot
    for ds_name in ds_names:
        ref_sum, _ = load_reference_baseline("cot", ds_name)
        if not ref_sum:
            continue
        print(
            f"{'[ref] chain_of_thought':<24s} {ds_name:<14s} {ref_sum['n']:>3d}  "
            f"{ref_sum['baseline_r@20']:>6.3f} {ref_sum['arch_r@20']:>6.3f} "
            f"{ref_sum['delta_r@20']:>+7.3f}  "
            f"{ref_sum['baseline_r@50']:>6.3f} {ref_sum['arch_r@50']:>6.3f} "
            f"{ref_sum['delta_r@50']:>+7.3f}  "
            f"{ref_sum['avg_llm_calls']:>4.1f} {'-':>6s}"
        )

    # ----- Per-category table (highlights chain-structured cats) -----
    print("\n" + "=" * 110)
    print("PER-CATEGORY BREAKDOWN (our variants)")
    print("=" * 110)
    for variant_name in variant_names:
        print(f"\n--- {variant_name} ---")
        for ds_name in ds_names:
            entry = all_summaries.get(variant_name, {}).get(ds_name)
            if not entry:
                continue
            print(f"  [{ds_name}]")
            by_cat = entry["category_breakdown"]
            for cat, c in sorted(by_cat.items()):
                tag = ""
                if cat in (
                    "sequential_chain",
                    "evolving_terminology",
                    "proactive",
                ):
                    tag = "  <-- target"
                if cat in ("locomo_single_hop", "control"):
                    tag = "  <-- expect hurt"
                print(
                    f"    {cat:28s} n={c['n']:<3d}  "
                    f"r@20 d={c['delta_r@20']:>+.3f} "
                    f"W/T/L={c['W/T/L_r@20']:<10s}  "
                    f"r@50 d={c['delta_r@50']:>+.3f}{tag}"
                )

    # Target-category head-to-head vs v2f
    print("\n" + "=" * 110)
    print("TARGET-CATEGORY HEAD-TO-HEAD vs meta_v2f (r@20 delta)")
    print("=" * 110)
    target_cats = {
        "puzzle_16q": "sequential_chain",
        "advanced_23q": "evolving_terminology",
        "synthetic_19q": "proactive",
    }
    hurt_cats = {
        "locomo_30q": ["locomo_single_hop"],
        "synthetic_19q": ["control"],
    }
    for ds_name, target_cat in target_cats.items():
        _, v2f_bycat = load_reference_baseline("v2f", ds_name)
        v2f_d = (v2f_bycat or {}).get(target_cat, {}).get("delta_r@20")
        for variant_name in variant_names:
            my = (
                all_summaries.get(variant_name, {})
                .get(ds_name, {})
                .get("category_breakdown", {})
                .get(target_cat)
            )
            if not my:
                continue
            print(
                f"  {ds_name:<14s} {target_cat:<24s} "
                f"{variant_name:<24s} "
                f"delta@20={my['delta_r@20']:>+.3f}  "
                f"[v2f={'n/a' if v2f_d is None else f'{v2f_d:+.3f}'}]"
            )
    print("\n--- hurt categories (expect overthinking) ---")
    for ds_name, cats in hurt_cats.items():
        _, v2f_bycat = load_reference_baseline("v2f", ds_name)
        for cat in cats:
            v2f_d = (v2f_bycat or {}).get(cat, {}).get("delta_r@20")
            for variant_name in variant_names:
                my = (
                    all_summaries.get(variant_name, {})
                    .get(ds_name, {})
                    .get("category_breakdown", {})
                    .get(cat)
                )
                if not my:
                    continue
                print(
                    f"  {ds_name:<14s} {cat:<24s} "
                    f"{variant_name:<24s} "
                    f"delta@20={my['delta_r@20']:>+.3f}  "
                    f"[v2f={'n/a' if v2f_d is None else f'{v2f_d:+.3f}'}]"
                )


if __name__ == "__main__":
    main()
