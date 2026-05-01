"""Research-informed supervisor for cue-generation control flow.

Hypothesis
----------
Prior reflection/reflector experiments failed because (a) the reflector
modified cue content (errors propagate into retrieval), (b) the reflector
had generic instructions, (c) the supervisor had no more context than the
worker.

This experiment tests a supervisor that:
    1. Has a PURE CONTROL-FLOW role (STOP vs CONTINUE, never generates cues)
    2. Has a RESEARCH-INFORMED PROMPT with meta-lessons from ~20 prior
       experiments the worker model doesn't know about.

Architecture
------------
    Hop 0 (cosine top-k on the question)
    Round 1: v2f cue generation (two cues)  -> retrieve
    Supervisor call: CONTINUE or STOP?
      CONTINUE -> Round 2: v2f complementary cues -> retrieve -> supervisor
      STOP    -> backfill remaining budget with cosine

Variants
--------
    A. supervisor_stop_only   : supervisor can stop; max rounds = 2
    B. supervisor_adaptive    : supervisor can stop; max rounds = 3
    C. always_2_rounds        : no supervisor, always 2 rounds (control)
    D. always_1_round         : 1 round v2f (current behavior, control)

Run on all four datasets (LoCoMo 30q, Synthetic 19q, Puzzle 16q,
Advanced 23q) at K=20 and K=50.

Usage
-----
    uv run python supervisor_control.py --list
    uv run python supervisor_control.py --all
    uv run python supervisor_control.py --variant supervisor_stop_only \
        --dataset locomo_30q --budget 20
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

CACHE_FILE_EMBED = "supervisor_embedding_cache.json"
CACHE_FILE_LLM = "supervisor_llm_cache.json"


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.with_suffix(path.suffix + ".lock")
    with open(lock_file, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Caches — read every existing cache, write to supervisor-specific file
# ---------------------------------------------------------------------------
class SupervisorEmbeddingCache(EmbeddingCache):
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for path in sorted(self.cache_dir.glob("*embedding_cache.json")):
            try:
                with open(path) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
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
                    existing = {}
            existing.update(self._new_entries)
            tmp = self.cache_file.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(existing, f)
            tmp.replace(self.cache_file)
        self._new_entries = {}


class SupervisorLLMCache(LLMCache):
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
                    existing = {}
            existing.update(self._new_entries)
            tmp = self.cache_file.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(existing, f)
            tmp.replace(self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Round 1 cue generator — standard v2f (the established local optimum)
V2F_ROUND1_PROMPT = """\
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


# Round 2+ cue generator — same v2f but explicitly asks for complementary
# cues targeting content missed by the first round.
V2F_ROUND_N_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

You have already run one round of cue generation (cues shown above under \
"PREVIOUS CUES"). Now generate COMPLEMENTARY cues targeting content the \
first round likely missed — different vocabulary, different turns, a \
different angle on the question.

First, briefly assess: What did the first round cover well? What kind of \
content is still missing? What different vocabulary might surface it?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns and that \
DIFFERS from the previous cues.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# Supervisor prompt — pure control-flow, research-informed priors.
SUPERVISOR_PROMPT = """\
You are a RETRIEVAL SUPERVISOR. A worker model has generated cues and \
retrieved segments.

RESEARCH PRIORS (established from ~20 experiments):
- v2f cue generation is a local optimum; adding context to the worker \
prompt has hurt in all tested cases
- Diversification is NOT the goal of cue generation; re-finding relevant \
content is
- Low duplicate rate in retrieved segments correlates with LOWER recall \
(worker is drifting from the question)
- Short conversations (50-120 turns) have a ceiling ~+10pp above cosine \
baseline at r@20
- Long conversations (400+ turns) benefit from cue generation much more
- If cue duplicate rate exceeds 80%, further cue generation is wasted

QUESTION: {question}
CUES GENERATED SO FAR: {cues_block}
SEGMENTS FOUND: {n_segments} unique, {duplicate_rate}% duplication in \
latest round
BUDGET REMAINING: {budget_left} segments at K={k}

DECIDE:
- CONTINUE: run another round of cue generation (only if duplicate rate \
< 80% AND budget > 5 segments)
- STOP: backfill remaining budget with cosine similarity

Output exactly two lines:
DECISION: CONTINUE | STOP
REASON: <short justification>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_segments(
    segments: list[Segment], max_items: int = 16, max_chars: int = 250
) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}"
        for seg in sorted_segs
    )


def _build_context_section(
    all_segments: list[Segment],
    previous_cues: list[str] | None = None,
) -> str:
    if not all_segments:
        base = (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    else:
        base = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(
            all_segments
        )
    if previous_cues:
        base += (
            "\n\nPREVIOUS CUES ALREADY TRIED (generate DIFFERENT ones):\n"
            + "\n".join(f"- {c}" for c in previous_cues)
        )
    return base


def _parse_cues(response: str) -> list[str]:
    cues: list[str] = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


def _parse_supervisor(response: str) -> tuple[str, str]:
    """Parse supervisor response. Returns (decision, reason).

    Defaults to STOP if parsing fails — the safe (cheaper) default.
    """
    decision = "STOP"
    reason = ""
    for line in response.strip().split("\n"):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("DECISION:"):
            rest = stripped.split(":", 1)[1].strip().upper()
            # Strip any qualifiers
            token = rest.split()[0] if rest else ""
            if token.startswith("CONTINUE"):
                decision = "CONTINUE"
            elif token.startswith("STOP"):
                decision = "STOP"
        elif upper.startswith("REASON:"):
            reason = stripped.split(":", 1)[1].strip()
    return decision, reason


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
@dataclass
class SupervisorResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


class SupervisorBase:
    name: str = "base"
    budget: int = 0
    # Max rounds of cue generation (1 = just round-1 v2f; 2 = round-1 + 1
    # possible continue; 3 = up to 2 continues)
    max_rounds: int = 1
    # If True, always run max_rounds. If False, consult the supervisor.
    always_run: bool = False

    def __init__(
        self,
        store: SegmentStore,
        budget: int,
        *,
        max_rounds: int,
        always_run: bool,
        name: str,
        client: OpenAI | None = None,
    ):
        self.store = store
        self.budget = budget
        self.max_rounds = max_rounds
        self.always_run = always_run
        self.name = name
        self.client = client or OpenAI(timeout=120.0)
        self.embedding_cache = SupervisorEmbeddingCache()
        self.llm_cache = SupervisorLLMCache()
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

    def llm_call(self, prompt: str, model: str = MODEL, max_tokens: int = 2000) -> str:
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

    # --- per-round retrieval helpers ---
    def _retrieve_cue(
        self,
        cue_emb: np.ndarray,
        top_k: int,
        conversation_id: str,
        exclude: set[int],
        into: list[Segment],
    ) -> int:
        result = self.store.search(
            cue_emb,
            top_k=top_k,
            conversation_id=conversation_id,
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

    def retrieve(self, question: str, conversation_id: str) -> SupervisorResult:
        exclude: set[int] = set()
        all_segments: list[Segment] = []

        # --- Hop 0: cosine top-k on the question ---
        query_emb = self.embed_text(question)
        hop0_k = max(4, min(10, self.budget // 2))
        hop0_res = self.store.search(
            query_emb,
            top_k=hop0_k,
            conversation_id=conversation_id,
        )
        for seg in hop0_res.segments:
            if seg.index not in exclude:
                all_segments.append(seg)
                exclude.add(seg.index)

        # Plan per-round retrieval size. Each round: 2 cues.
        # per_cue_k chosen to split the remaining budget evenly across
        # planned rounds.
        planned_rounds = self.max_rounds
        # After hop 0 we have `hop0_k` segments; remaining budget for
        # cue-driven retrieval across all rounds:
        remaining_budget_total = max(0, self.budget - hop0_k)
        # We want per_cue_k such that 2 * per_cue_k * planned_rounds <=
        # remaining_budget_total (leave some slack for dup loss; we
        # backfill at the end).
        if planned_rounds > 0 and remaining_budget_total > 0:
            per_cue_k = max(2, remaining_budget_total // (2 * planned_rounds))
        else:
            per_cue_k = 0

        all_cues: list[str] = []
        all_round_cues: list[list[str]] = []
        rounds_run = 0
        supervisor_log: list[dict] = []
        duplicate_rates: list[float] = []

        for round_idx in range(self.max_rounds):
            if len(all_segments) >= self.budget:
                break
            budget_left = self.budget - len(all_segments)

            # ---- Generate cues for this round ----
            if round_idx == 0:
                template = V2F_ROUND1_PROMPT
                context_section = _build_context_section(all_segments)
            else:
                template = V2F_ROUND_N_PROMPT
                context_section = _build_context_section(
                    all_segments,
                    previous_cues=all_cues,
                )

            prompt = template.format(
                question=question,
                context_section=context_section,
            )
            output = self.llm_call(prompt)
            cues = _parse_cues(output)
            used_cues: list[str] = []
            dup_hits = 0
            total_attempted = 0
            for i in range(2):
                cue = cues[i] if i < len(cues) else question
                used_cues.append(cue)
                cue_emb = self.embed_text(cue)
                # For duplicate accounting: ask store for per_cue_k WITHOUT
                # exclude mask to measure raw overlap.
                raw_res = self.store.search(
                    cue_emb,
                    top_k=per_cue_k,
                    conversation_id=conversation_id,
                )
                for seg in raw_res.segments:
                    total_attempted += 1
                    if seg.index in exclude:
                        dup_hits += 1
                # Now actually add (excluding duplicates)
                room = self.budget - len(all_segments)
                if room <= 0:
                    break
                fetch_k = min(per_cue_k, room)
                self._retrieve_cue(
                    cue_emb,
                    fetch_k,
                    conversation_id,
                    exclude,
                    all_segments,
                )
            all_round_cues.append(used_cues)
            all_cues.extend(used_cues)
            rounds_run += 1

            dup_rate = (
                round((dup_hits / total_attempted) * 100, 1)
                if total_attempted > 0
                else 0.0
            )
            duplicate_rates.append(dup_rate)

            # ---- Decide whether to continue ----
            is_last_round = (round_idx + 1) >= self.max_rounds
            if is_last_round:
                break
            if self.always_run:
                continue  # always run all rounds, no supervisor

            # Supervisor decision
            budget_left = self.budget - len(all_segments)
            cues_block = "\n".join(
                f"  r{ri + 1}: {c}" for ri, cs in enumerate(all_round_cues) for c in cs
            )
            sup_prompt = SUPERVISOR_PROMPT.format(
                question=question,
                cues_block=cues_block,
                n_segments=len(all_segments),
                duplicate_rate=dup_rate,
                budget_left=budget_left,
                k=self.budget,
            )
            sup_out = self.llm_call(sup_prompt)
            decision, reason = _parse_supervisor(sup_out)
            supervisor_log.append(
                {
                    "round": round_idx + 1,
                    "decision": decision,
                    "reason": reason,
                    "duplicate_rate": dup_rate,
                    "n_segments": len(all_segments),
                    "budget_left": budget_left,
                    "raw_output": sup_out,
                }
            )
            if decision == "STOP":
                break

        # ---- Backfill with cosine to reach exact budget ----
        if len(all_segments) < self.budget:
            need = self.budget - len(all_segments)
            backfill = self.store.search(
                query_emb,
                top_k=need + len(exclude),
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in backfill.segments:
                if seg.index in exclude:
                    continue
                all_segments.append(seg)
                exclude.add(seg.index)
                if len(all_segments) >= self.budget:
                    break

        # Strict truncate
        all_segments = all_segments[: self.budget]

        return SupervisorResult(
            segments=all_segments,
            metadata={
                "name": self.name,
                "budget": self.budget,
                "hop0_k": hop0_k,
                "per_cue_k": per_cue_k,
                "rounds_run": rounds_run,
                "max_rounds": self.max_rounds,
                "always_run": self.always_run,
                "cues": all_cues,
                "round_cues": all_round_cues,
                "duplicate_rates": duplicate_rates,
                "supervisor_log": supervisor_log,
            },
        )


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
VARIANTS = {
    "supervisor_stop_only": {
        "max_rounds": 2,
        "always_run": False,
        "desc": "Up to 2 rounds, supervisor can stop",
    },
    "supervisor_adaptive": {
        "max_rounds": 3,
        "always_run": False,
        "desc": "Up to 3 rounds, supervisor can stop",
    },
    "always_2_rounds": {
        "max_rounds": 2,
        "always_run": True,
        "desc": "Always 2 rounds, no supervisor",
    },
    "always_1_round": {
        "max_rounds": 1,
        "always_run": True,
        "desc": "Always 1 round (v2f control)",
    },
}


def build_variant(name: str, store: SegmentStore, budget: int) -> SupervisorBase:
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant: {name}")
    cfg = VARIANTS[name]
    return SupervisorBase(
        store,
        budget=budget,
        max_rounds=cfg["max_rounds"],
        always_run=cfg["always_run"],
        name=name,
    )


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
# Evaluation
# ---------------------------------------------------------------------------
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(arch: SupervisorBase, question: dict, verbose: bool = False) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    segments = result.segments
    turn_ids = {s.turn_id for s in segments}
    recall = compute_recall(turn_ids, source_ids)

    # Extract supervisor decisions
    sup_log = result.metadata.get("supervisor_log", [])
    n_stop = sum(1 for e in sup_log if e.get("decision") == "STOP")
    n_continue = sum(1 for e in sup_log if e.get("decision") == "CONTINUE")

    row = {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question.get("question_index"),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "budget": arch.budget,
        "actual_count": len(segments),
        "recall": recall,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "rounds_run": result.metadata.get("rounds_run", 0),
        "supervisor_stops": n_stop,
        "supervisor_continues": n_continue,
        "duplicate_rates": result.metadata.get("duplicate_rates", []),
        "metadata": {
            k: v
            for k, v in result.metadata.items()
            if k
            in (
                "name",
                "cues",
                "round_cues",
                "hop0_k",
                "per_cue_k",
                "rounds_run",
                "supervisor_log",
                "duplicate_rates",
            )
        },
    }

    if verbose:
        print(
            f"    rounds={row['rounds_run']} recall={recall:.3f} "
            f"llm={arch.llm_calls} dup_rates={row['duplicate_rates']} "
            f"sup_stop={n_stop} sup_cont={n_continue} time={elapsed:.1f}s"
        )

    return row


def summarize(results: list[dict], variant: str, dataset: str, budget: int) -> dict:
    n = len(results)
    if n == 0:
        return {"variant": variant, "dataset": dataset, "budget": budget, "n": 0}

    recalls = [r["recall"] for r in results]
    llm_calls = [r["llm_calls"] for r in results]
    rounds_run = [r["rounds_run"] for r in results]
    n_stops = sum(r["supervisor_stops"] for r in results)
    n_conts = sum(r["supervisor_continues"] for r in results)

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
        "variant": variant,
        "dataset": dataset,
        "budget": budget,
        "n": n,
        "mean_recall": round(sum(recalls) / n, 4),
        "mean_llm_calls": round(sum(llm_calls) / n, 2),
        "mean_rounds_run": round(sum(rounds_run) / n, 2),
        "total_supervisor_stops": n_stops,
        "total_supervisor_continues": n_conts,
        "avg_embed_calls": round(sum(r["embed_calls"] for r in results) / n, 2),
        "avg_time_s": round(sum(r["time_s"] for r in results) / n, 2),
        "per_category": cat_summary,
    }


def run_one(
    variant: str, dataset: str, budget: int, force: bool = False, verbose: bool = False
) -> dict:
    result_file = RESULTS_DIR / (f"supervisor_{variant}_{dataset}_k{budget}.json")
    if result_file.exists() and not force:
        with open(result_file) as f:
            saved = json.load(f)
        s = saved.get("summary", {})
        print(
            f"  [cache] {variant} @ {dataset} K={budget}: "
            f"recall={s.get('mean_recall', 0):.3f} "
            f"llm={s.get('mean_llm_calls', 0):.2f} "
            f"stops={s.get('total_supervisor_stops', 0)} "
            f"conts={s.get('total_supervisor_continues', 0)}"
        )
        return s

    qs, store = load_dataset(dataset)
    arch = build_variant(variant, store, budget)

    print(
        f"\n>>> {variant} on {dataset} (K={budget}, "
        f"{len(qs)} questions, {len(store.segments)} segments)"
    )
    t_all = time.time()
    results: list[dict] = []
    for i, q in enumerate(qs):
        q_short = q["question"][:60].replace("\n", " ")
        print(
            f"  [{i + 1}/{len(qs)}] {q['category']}: {q_short}...",
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
        # Incremental per-question save
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_partial = summarize(results, variant, dataset, budget)
        with open(result_file, "w") as f:
            json.dump(
                {"results": results, "summary": summary_partial},
                f,
                indent=2,
                default=str,
            )
        if (i + 1) % 3 == 0:
            arch.save_caches()

    arch.save_caches()
    elapsed = time.time() - t_all

    summary = summarize(results, variant, dataset, budget)
    summary["wall_time_s"] = round(elapsed, 1)
    with open(result_file, "w") as f:
        json.dump(
            {"results": results, "summary": summary},
            f,
            indent=2,
            default=str,
        )

    print(
        f"  -> recall={summary['mean_recall']:.3f} "
        f"rounds={summary['mean_rounds_run']:.2f} "
        f"llm={summary['mean_llm_calls']:.2f} "
        f"stops={summary['total_supervisor_stops']} "
        f"conts={summary['total_supervisor_continues']} "
        f"time={elapsed:.0f}s"
    )
    return summary


def print_final_table(all_summaries: dict) -> None:
    datasets = ["locomo_30q", "synthetic_19q", "puzzle_16q", "advanced_23q"]
    short_ds = {
        "locomo_30q": "LoCoMo",
        "synthetic_19q": "Synth",
        "puzzle_16q": "Puzzle",
        "advanced_23q": "Advanced",
    }
    variants = list(VARIANTS.keys())

    for budget in (20, 50):
        print("\n" + "=" * 110)
        print(f"RECALL @ K={budget}")
        print("=" * 110)
        header = (
            f"{'Variant':<24s}"
            + "".join(f"{short_ds[ds]:>12s}" for ds in datasets)
            + f"{'Mean':>12s}"
        )
        print(header)
        print("-" * len(header))
        for v in variants:
            row = f"{v:<24s}"
            vals = []
            for ds in datasets:
                key = (v, ds, budget)
                if key in all_summaries:
                    r = all_summaries[key].get("mean_recall", 0.0)
                    row += f"{r:>12.3f}"
                    vals.append(r)
                else:
                    row += f"{'—':>12s}"
            if vals:
                row += f"{sum(vals) / len(vals):>12.3f}"
            else:
                row += f"{'—':>12s}"
            print(row)

        print(f"\n--- LLM calls per question (K={budget}) ---")
        print(header)
        print("-" * len(header))
        for v in variants:
            row = f"{v:<24s}"
            vals = []
            for ds in datasets:
                key = (v, ds, budget)
                if key in all_summaries:
                    llm = all_summaries[key].get("mean_llm_calls", 0.0)
                    row += f"{llm:>12.2f}"
                    vals.append(llm)
                else:
                    row += f"{'—':>12s}"
            if vals:
                row += f"{sum(vals) / len(vals):>12.2f}"
            else:
                row += f"{'—':>12s}"
            print(row)

        print(f"\n--- Supervisor decisions per dataset (K={budget}) ---")
        sup_header = f"{'Variant':<24s}" + "".join(
            f"{short_ds[ds]:>16s}" for ds in datasets
        )
        print(sup_header)
        print("-" * len(sup_header))
        for v in variants:
            row = f"{v:<24s}"
            for ds in datasets:
                key = (v, ds, budget)
                if key in all_summaries:
                    s = all_summaries[key]
                    stops = s.get("total_supervisor_stops", 0)
                    conts = s.get("total_supervisor_continues", 0)
                    row += f"{f'{stops}S/{conts}C':>16s}"
                else:
                    row += f"{'—':>16s}"
            print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Supervisor control-flow experiments")
    parser.add_argument(
        "--variant", type=str, default=None, choices=list(VARIANTS.keys())
    )
    parser.add_argument(
        "--dataset", type=str, default=None, choices=list(DATASETS.keys())
    )
    parser.add_argument("--budget", type=int, default=None, choices=[20, 50])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Variants:")
        for name, cfg in VARIANTS.items():
            print(
                f"  {name}: {cfg['desc']} "
                f"(max_rounds={cfg['max_rounds']}, "
                f"always_run={cfg['always_run']})"
            )
        print("\nDatasets:")
        for key in DATASETS:
            print(f"  {key}")
        print("\nBudgets: 20, 50")
        return

    variants = [args.variant] if args.variant else list(VARIANTS.keys())
    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    budgets = [args.budget] if args.budget else [20, 50]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[tuple[str, str, int], dict] = {}
    for v in variants:
        for ds in datasets:
            for b in budgets:
                try:
                    summary = run_one(
                        v,
                        ds,
                        b,
                        force=args.force,
                        verbose=args.verbose,
                    )
                    all_summaries[(v, ds, b)] = summary
                except Exception as e:
                    print(f"  FATAL on {v}/{ds}/K={b}: {type(e).__name__}: {e}")
                    import traceback

                    traceback.print_exc()

    # Persist overall summary
    summary_file = RESULTS_DIR / "supervisor_all_summaries.json"
    payload = {
        f"{v}@{ds}@K{b}": summary for (v, ds, b), summary in all_summaries.items()
    }
    with open(summary_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nAll summaries saved: {summary_file}")

    print_final_table(all_summaries)


if __name__ == "__main__":
    main()
