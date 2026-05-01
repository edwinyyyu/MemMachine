"""Streaming Accept/Commit retrieval architecture.

Theory:
  Standard pipelines are retrieve -> cue-generate -> retrieve -> rank ->
  truncate. Ranking is done statically (cosine / retrieval-order) or
  via a one-shot rerank.

  This architecture is iterative and goal-directed:
    1. Retrieve a small batch (5 segments).
    2. LLM decides per-segment: COMMIT (goes into answer set) or SKIP.
    3. LLM generates the NEXT cue based on what's been committed + what's
       still needed for the goal.
    4. Retrieve another batch; repeat until K commits or DONE.

  The accept/reject is grounded: the model judges a SPECIFIC segment as
  relevant/irrelevant, which is tractable vs. meta-questions like
  "do I have enough information?".

Variants
  A. stream_commit           - as described.
  B. stream_commit_no_cue    - skip cue generation; cosine with exclusion
                               only; keep per-segment commit judgment.
  C. stream_commit_goal_check- after each batch, LLM outputs a progress
                               note; if "close enough", stop early.

Evaluation
  4 datasets (locomo_30q, synthetic_19q, puzzle_16q, advanced_23q) at
  K=20 and K=50, fair backfill vs. cosine. Comparison against baseline,
  v15_tight, v2f_tight (loaded from budget_*.json), and CoT
  (cot_chain_of_thought_*.json).

Usage
  uv run python stream_commit.py [--force] [--variants A,B,C]
                                 [--dataset locomo_30q|...]
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
CACHE_FILE_EMB = CACHE_DIR / "stream_commit_embedding_cache.json"
CACHE_FILE_LLM = CACHE_DIR / "stream_commit_llm_cache.json"

BUDGETS = [20, 50]

# Streaming config
BATCH_SIZE = 5
MAX_ROUNDS = 5  # includes the initial round
COMMIT_TARGET = 20


# ---------------------------------------------------------------------------
# Caches (shared-read, isolated-write, like cot_universal.py)
# ---------------------------------------------------------------------------
class StreamCommitEmbeddingCache(EmbeddingCache):
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


class StreamCommitLLMCache(LLMCache):
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
# Helpers: formatting
# ---------------------------------------------------------------------------
def _format_committed(segments: list[Segment], max_chars: int = 260) -> str:
    if not segments:
        return "(none yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in sorted_segs
    )


def _format_candidates(segments: list[Segment], max_chars: int = 260) -> str:
    # Preserve the enumerated order the LLM will reference with SEG <idx>.
    return "\n".join(
        f"SEG {i} [Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}"
        for i, s in enumerate(segments)
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
COMMIT_PROMPT = """\
Goal: {question}

Already committed (segments we'll use to answer):
{committed}

New candidate segments to consider:
{candidates}

For each new candidate, decide COMMIT (relevant to goal, should be included in \
answer) or SKIP (not relevant).

Output one line per candidate, in the format:
SEG <index>: COMMIT
or
SEG <index>: SKIP

Nothing else."""


CUE_PROMPT = """\
Goal: {question}

Committed so far:
{committed}

What specific information is still missing to fully answer the goal? \
Generate ONE search cue to find it. Use vocabulary that would appear in the \
source content (names, terms, phrasing), not meta-questions.

Format:
CUE: <text>
Nothing else."""


GOAL_CHECK_PROMPT = """\
Goal: {question}

Committed so far:
{committed}

Do the committed segments already contain enough information to fully answer \
the goal? Reply with exactly one of:
STATUS: DONE
STATUS: CONTINUE

Nothing else."""


# ---------------------------------------------------------------------------
# Stream-Commit architecture
# ---------------------------------------------------------------------------
@dataclass
class StreamResult:
    committed: list[Segment]
    seen: list[Segment]  # everything the model saw, in order (for backfill)
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


class StreamCommitArch:
    """Streaming accept/commit retrieval."""

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        *,
        variant: str = "A",  # "A" | "B" | "C"
        batch_size: int = BATCH_SIZE,
        max_rounds: int = MAX_ROUNDS,
        commit_target: int = COMMIT_TARGET,
    ) -> None:
        assert variant in {"A", "B", "C"}
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = StreamCommitEmbeddingCache()
        self.llm_cache = StreamCommitLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0
        self.variant = variant
        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.commit_target = commit_target

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
            max_completion_tokens=800,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(model, prompt, text)
        self.llm_calls += 1
        return text

    # -----------------------------------------------------------------------
    # Parsing
    # -----------------------------------------------------------------------
    @staticmethod
    def _parse_commit_decisions(response: str, num_candidates: int) -> list[bool]:
        """Returns a bool list of length num_candidates; True=COMMIT."""
        decisions = [False] * num_candidates
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line.upper().startswith("SEG"):
                continue
            try:
                rest = line[3:].strip()
                idx_s, _, verdict = rest.partition(":")
                idx = int(idx_s.strip())
                if 0 <= idx < num_candidates:
                    decisions[idx] = "COMMIT" in verdict.upper()
            except (ValueError, IndexError):
                continue
        return decisions

    @staticmethod
    def _parse_cue(response: str) -> str:
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("CUE:"):
                return line[4:].strip()
        return ""

    @staticmethod
    def _parse_status_done(response: str) -> bool:
        for line in response.strip().split("\n"):
            line = line.strip().upper()
            if line.startswith("STATUS:"):
                return "DONE" in line
        return False

    # -----------------------------------------------------------------------
    # Core retrieval loop
    # -----------------------------------------------------------------------
    def retrieve(self, question: str, conversation_id: str) -> StreamResult:
        committed: list[Segment] = []
        seen: list[Segment] = []
        exclude: set[int] = set()
        round_log: list[dict] = []

        q_emb = self.embed_text(question)
        # ------ Round 0: cosine top-batch_size on question ------
        r0 = self.store.search(
            q_emb, top_k=self.batch_size, conversation_id=conversation_id
        )
        batch = r0.segments
        for s in batch:
            exclude.add(s.index)
            seen.append(s)

        if batch:
            prompt = COMMIT_PROMPT.format(
                question=question,
                committed=_format_committed(committed),
                candidates=_format_candidates(batch),
            )
            resp = self.llm_call(prompt)
            decisions = self._parse_commit_decisions(resp, len(batch))
            committed_this_round = [s for s, d in zip(batch, decisions) if d]
            committed.extend(committed_this_round)
            round_log.append(
                {
                    "round": 0,
                    "cue": None,
                    "batch_size": len(batch),
                    "committed_this_round": len(committed_this_round),
                }
            )

        stopped_early = False
        # ------ Rounds 1 .. max_rounds-1 ------
        for round_i in range(1, self.max_rounds):
            if len(committed) >= self.commit_target:
                break

            cue = ""
            if self.variant in {"A", "C"}:
                # Variant A and C: generate a cue.
                cue_prompt = CUE_PROMPT.format(
                    question=question,
                    committed=_format_committed(committed),
                )
                cue_resp = self.llm_call(cue_prompt)
                cue = self._parse_cue(cue_resp)

            # Embed the search vector. For variant A/C fall back to the
            # question when no cue was produced; for variant B use the
            # question directly.
            if self.variant == "B" or not cue:
                search_emb = q_emb
            else:
                search_emb = self.embed_text(cue)

            result = self.store.search(
                search_emb,
                top_k=self.batch_size,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            batch = result.segments
            if not batch:
                round_log.append(
                    {
                        "round": round_i,
                        "cue": cue,
                        "batch_size": 0,
                        "committed_this_round": 0,
                    }
                )
                break
            for s in batch:
                exclude.add(s.index)
                seen.append(s)

            commit_prompt = COMMIT_PROMPT.format(
                question=question,
                committed=_format_committed(committed),
                candidates=_format_candidates(batch),
            )
            resp = self.llm_call(commit_prompt)
            decisions = self._parse_commit_decisions(resp, len(batch))
            committed_this_round = [s for s, d in zip(batch, decisions) if d]
            committed.extend(committed_this_round)

            # Variant C: optional early-stop check.
            early_stop_checked = False
            if self.variant == "C" and committed:
                gc_prompt = GOAL_CHECK_PROMPT.format(
                    question=question,
                    committed=_format_committed(committed),
                )
                gc_resp = self.llm_call(gc_prompt)
                early_stop_checked = True
                if self._parse_status_done(gc_resp):
                    round_log.append(
                        {
                            "round": round_i,
                            "cue": cue,
                            "batch_size": len(batch),
                            "committed_this_round": len(committed_this_round),
                            "early_stop": True,
                        }
                    )
                    stopped_early = True
                    break

            round_log.append(
                {
                    "round": round_i,
                    "cue": cue,
                    "batch_size": len(batch),
                    "committed_this_round": len(committed_this_round),
                    "early_stop": False if early_stop_checked else None,
                }
            )

        return StreamResult(
            committed=committed,
            seen=seen,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "variant": self.variant,
                "rounds": round_log,
                "num_committed": len(committed),
                "num_seen": len(seen),
                "stopped_early": stopped_early,
            },
        )


# ---------------------------------------------------------------------------
# Evaluation: fair backfill against cosine
# ---------------------------------------------------------------------------
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(arch: StreamCommitArch, question: dict, verbose: bool = False) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedup committed preserving order.
    seen_ids: set[int] = set()
    committed_segments: list[Segment] = []
    for s in result.committed:
        if s.index not in seen_ids:
            committed_segments.append(s)
            seen_ids.add(s.index)

    # Baseline cosine top-max(BUDGETS) on the question.
    q_emb = arch.embed_text(q_text)
    max_b = max(BUDGETS)
    baseline = arch.store.search(q_emb, top_k=max_b, conversation_id=conv_id)

    # Fair backfill. Priority order:
    #   1. committed (LLM-approved)
    #   2. seen-but-skipped (already-retrieved but not committed)
    #   3. baseline cosine-top-K fillers (not in pool)
    committed_idx = {s.index for s in committed_segments}
    seen_but_skipped = [s for s in result.seen if s.index not in committed_idx]
    arch_pool = list(committed_segments) + seen_but_skipped
    arch_pool_idx = {s.index for s in arch_pool}
    backfilled = arch_pool + [
        s for s in baseline.segments if s.index not in arch_pool_idx
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
        "num_committed": len(committed_segments),
        "num_seen": len(result.seen),
        "baseline_recalls": baseline_recalls,
        "stream_recalls": recalls,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "metadata": result.metadata,
    }
    if verbose:
        print(
            f"    committed={len(committed_segments)} "
            f"seen={len(result.seen)} "
            f"r@20: base={baseline_recalls['r@20']:.3f} "
            f"stream={recalls['r@20']:.3f}  "
            f"r@50: base={baseline_recalls['r@50']:.3f} "
            f"stream={recalls['r@50']:.3f}  "
            f"emb={arch.embed_calls} llm={arch.llm_calls}"
            f" stopped_early={result.metadata.get('stopped_early')}"
        )
    return row


# ---------------------------------------------------------------------------
# Datasets (mirrors cot_universal.py)
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
# Comparison loaders
# ---------------------------------------------------------------------------
def load_budget_recall_by_qkey(arch_name: str, dataset_key: str) -> dict[tuple, float]:
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


def load_cot_recall_by_qkey(dataset_key: str, K: int) -> dict[tuple, float]:
    path = RESULTS_DIR / f"cot_chain_of_thought_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    out: dict[tuple, float] = {}
    for r in rows:
        key = (r["conversation_id"], r.get("question_index"))
        out[key] = r["cot_recalls"][f"r@{K}"]
    return out


# ---------------------------------------------------------------------------
# Main run / incremental save
# ---------------------------------------------------------------------------
def stream_run(
    dataset_key: str,
    variant: str,
    force: bool = False,
    verbose: bool = False,
) -> list[dict]:
    suffix = {"A": "commit", "B": "no_cue", "C": "goal_check"}[variant]
    result_file = RESULTS_DIR / (f"stream_commit_{suffix}_{dataset_key}.json")

    qs, store = load_dataset(dataset_key)
    rows: list[dict] = []
    # Load existing partial results if present (resume).
    if result_file.exists() and not force:
        try:
            with open(result_file) as f:
                rows = json.load(f)
        except json.JSONDecodeError:
            rows = []
        done_keys = {(r["conversation_id"], r.get("question_index")) for r in rows}
        # Already fully done? short-circuit.
        if all(
            (q["conversation_id"], q.get("question_index")) in done_keys for q in qs
        ):
            return rows

    arch = StreamCommitArch(store, variant=variant)
    print(
        f"\n>>> stream_commit(variant={variant}) on {dataset_key}: "
        f"{len(qs)} questions, {len(store.segments)} segments "
        f"(resuming from {len(rows)})"
    )
    done_keys = {(r["conversation_id"], r.get("question_index")) for r in rows}

    for i, q in enumerate(qs):
        key = (q["conversation_id"], q.get("question_index"))
        if not force and key in done_keys:
            continue
        q_short = q["question"][:60].replace("\n", " ")
        print(
            f"  [{i + 1}/{len(qs)}] {q['category']}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_one(arch, q, verbose=verbose)
            rows.append(row)
        except Exception as e:  # pragma: no cover
            print(f"    ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback

            traceback.print_exc()
        sys.stdout.flush()

        # Incremental save every question (as requested in the spec).
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    with open(result_file, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"  Saved -> {result_file}")
    return rows


# ---------------------------------------------------------------------------
# Per-category summary
# ---------------------------------------------------------------------------
def summarize(dataset_key: str, variant: str, rows: list[dict]) -> list[dict]:
    """Group by category; attach baseline/v15/v2f/cot per-question recall
    for head-to-head."""
    budget_by_arch: dict[str, dict[int, dict]] = {}
    for arch_name in ("baseline", "v15_tight", "v2f_tight"):
        budget_by_arch[arch_name] = {
            K: load_budget_recall_by_qkey(f"{arch_name}_{K}", dataset_key)
            for K in BUDGETS
        }
    cot_by_k: dict[int, dict] = {
        K: load_cot_recall_by_qkey(dataset_key, K) for K in BUDGETS
    }

    rows_by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rows_by_cat[r["category"]].append(r)

    out: list[dict] = []
    for K in BUDGETS:
        for cat, cat_rows in sorted(rows_by_cat.items()):
            n = len(cat_rows)
            if n == 0:
                continue
            stream_vals = [r["stream_recalls"][f"r@{K}"] for r in cat_rows]
            stream_mean = sum(stream_vals) / n

            def _collect(arch_name: str) -> list[float]:
                return [
                    budget_by_arch[arch_name][K][
                        (r["conversation_id"], r["question_index"])
                    ]
                    for r in cat_rows
                    if (r["conversation_id"], r["question_index"])
                    in budget_by_arch[arch_name][K]
                ]

            def _mean(xs: list[float]) -> float | None:
                return (sum(xs) / len(xs)) if xs else None

            b_mean = _mean(_collect("baseline"))
            v15_mean = _mean(_collect("v15_tight"))
            v2f_mean = _mean(_collect("v2f_tight"))
            cot_vals = [
                cot_by_k[K][(r["conversation_id"], r["question_index"])]
                for r in cat_rows
                if (r["conversation_id"], r["question_index"]) in cot_by_k[K]
            ]
            cot_mean = _mean(cot_vals)

            row = {
                "dataset": dataset_key,
                "variant": variant,
                "category": cat,
                "K": K,
                "n": n,
                "baseline": b_mean,
                "v15": v15_mean,
                "v2f": v2f_mean,
                "cot": cot_mean,
                "stream": stream_mean,
                "stream_vs_v2f": (
                    (stream_mean - v2f_mean) if v2f_mean is not None else None
                ),
                "stream_vs_cot": (
                    (stream_mean - cot_mean) if cot_mean is not None else None
                ),
                "stream_vs_baseline": (
                    (stream_mean - b_mean) if b_mean is not None else None
                ),
            }
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def fmt_cell(val: float | None, plus_sign: bool = False) -> str:
    if val is None:
        return "    —"
    s = f"{val:+.3f}" if plus_sign else f"{val:.3f}"
    return f"{s:>6s}"


def print_overall(all_rows: list[dict], K: int) -> None:
    rows = [r for r in all_rows if r["K"] == K]
    if not rows:
        return
    print(f"\n{'-' * 96}")
    print(
        f"OVERALL per DATASET at K={K}  (recall, fair backfill; weighted by category n)"
    )
    print(f"{'-' * 96}")
    hdr = (
        f"{'Dataset':<14s} {'Var':<4s} {'n':>3s} "
        f"{'Base':>7s} {'v15':>7s} {'v2f':>7s} {'CoT':>7s} {'Stream':>7s}  "
        f"{'vs v2f':>7s} {'vs CoT':>7s} {'vs base':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["variant"])].append(r)

    for (ds, variant), grp in sorted(groups.items()):
        total_n = sum(r["n"] for r in grp)

        def _weighted(key: str) -> float | None:
            vals = [(r[key], r["n"]) for r in grp if r[key] is not None]
            if not vals:
                return None
            tot = sum(n for _, n in vals)
            return sum(v * n for v, n in vals) / tot

        b = _weighted("baseline")
        v15 = _weighted("v15")
        v2f = _weighted("v2f")
        cot = _weighted("cot")
        stream = _weighted("stream")
        cv = (stream - v2f) if (stream is not None and v2f is not None) else None
        cc = (stream - cot) if (stream is not None and cot is not None) else None
        cb = (stream - b) if (stream is not None and b is not None) else None
        print(
            f"{ds:<14s} {variant:<4s} {total_n:>3d} "
            f"{fmt_cell(b)} {fmt_cell(v15)} {fmt_cell(v2f)} "
            f"{fmt_cell(cot)} {fmt_cell(stream)}  "
            f"{fmt_cell(cv, True)} {fmt_cell(cc, True)} "
            f"{fmt_cell(cb, True)}"
        )


def print_cross_dataset(all_rows: list[dict], K: int) -> None:
    rows = [r for r in all_rows if r["K"] == K]
    if not rows:
        return
    print(f"\n{'=' * 96}")
    print(
        f"CROSS-DATASET AVERAGE at K={K}  "
        f"(mean across all questions in all datasets, per variant)"
    )
    print(f"{'=' * 96}")
    hdr = (
        f"{'Var':<8s} {'n':>4s} "
        f"{'Base':>7s} {'v15':>7s} {'v2f':>7s} {'CoT':>7s} {'Stream':>7s}  "
        f"{'vs v2f':>7s} {'vs CoT':>7s} {'vs base':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))

    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    for variant, grp in sorted(by_variant.items()):
        total_n = sum(r["n"] for r in grp)

        def _weighted(key: str) -> float | None:
            vals = [(r[key], r["n"]) for r in grp if r[key] is not None]
            if not vals:
                return None
            tot = sum(n for _, n in vals)
            return sum(v * n for v, n in vals) / tot

        b = _weighted("baseline")
        v15 = _weighted("v15")
        v2f = _weighted("v2f")
        cot = _weighted("cot")
        stream = _weighted("stream")
        cv = (stream - v2f) if (stream is not None and v2f is not None) else None
        cc = (stream - cot) if (stream is not None and cot is not None) else None
        cb = (stream - b) if (stream is not None and b is not None) else None
        print(
            f"{variant:<8s} {total_n:>4d} "
            f"{fmt_cell(b)} {fmt_cell(v15)} {fmt_cell(v2f)} "
            f"{fmt_cell(cot)} {fmt_cell(stream)}  "
            f"{fmt_cell(cv, True)} {fmt_cell(cc, True)} "
            f"{fmt_cell(cb, True)}"
        )


# Hard categories called out in the design (spec: completeness,
# sequential_chain, proactive, logic_constraint).
HARD_CATEGORIES = {
    "completeness",
    "sequential_chain",
    "proactive",
    "logic_constraint",
}


def print_hard_categories(all_rows: list[dict], K: int) -> None:
    rows = [r for r in all_rows if r["K"] == K and r["category"] in HARD_CATEGORIES]
    if not rows:
        return
    print(f"\n{'=' * 96}")
    print(f"HARD-CATEGORY BREAKDOWN at K={K}")
    print(f"{'=' * 96}")
    hdr = (
        f"{'Dataset':<14s} {'Category':<22s} {'Var':<4s} {'n':>3s} "
        f"{'v2f':>7s} {'CoT':>7s} {'Stream':>7s}  "
        f"{'vs v2f':>7s} {'vs CoT':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(
        rows,
        key=lambda x: (x["dataset"], x["category"], x["variant"]),
    ):
        print(
            f"{r['dataset']:<14s} {r['category']:<22s} {r['variant']:<4s} "
            f"{r['n']:>3d} "
            f"{fmt_cell(r['v2f'])} {fmt_cell(r['cot'])} "
            f"{fmt_cell(r['stream'])}  "
            f"{fmt_cell(r['stream_vs_v2f'], True)} "
            f"{fmt_cell(r['stream_vs_cot'], True)}"
        )


def print_cost_summary(variant_rows: dict[str, list[dict]]) -> None:
    """Average embed/LLM call counts per question per variant and dataset."""
    print(f"\n{'=' * 96}")
    print("COST SUMMARY  (avg embed / LLM calls per question)")
    print(f"{'=' * 96}")
    hdr = (
        f"{'Dataset':<14s} {'Var':<4s} {'n':>3s} "
        f"{'avg_embed':>10s} {'avg_llm':>9s} {'avg_committed':>14s} "
        f"{'avg_seen':>9s} {'early_stop%':>12s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for variant, rows in sorted(variant_rows.items()):
        by_ds: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_ds[r["_dataset"]].append(r)
        for ds, grp in sorted(by_ds.items()):
            n = len(grp)
            if n == 0:
                continue
            avg_embed = sum(r["embed_calls"] for r in grp) / n
            avg_llm = sum(r["llm_calls"] for r in grp) / n
            avg_com = sum(r["num_committed"] for r in grp) / n
            avg_seen = sum(r["num_seen"] for r in grp) / n
            es = [1 if r.get("metadata", {}).get("stopped_early") else 0 for r in grp]
            es_pct = 100.0 * sum(es) / n if n else 0.0
            print(
                f"{ds:<14s} {variant:<4s} {n:>3d} "
                f"{avg_embed:>10.2f} {avg_llm:>9.2f} "
                f"{avg_com:>14.2f} {avg_seen:>9.2f} {es_pct:>11.1f}%"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Rerun even if result file exists"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASETS.keys()),
        help="Restrict to a single dataset (default: all)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="A,B,C",
        help="Comma-separated variants to run: A (commit), "
        "B (no_cue), C (goal_check). Default A,B,C.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only summarize existing result files; do not run anything.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_keys = [args.dataset] if args.dataset else list(DATASETS.keys())
    variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in {"A", "B", "C"}:
            raise SystemExit(f"Unknown variant: {v}")

    all_rows: list[dict] = []
    # per-variant raw rows for cost summary
    variant_rows: dict[str, list[dict]] = defaultdict(list)
    for variant in variants:
        suffix = {"A": "commit", "B": "no_cue", "C": "goal_check"}[variant]
        for ds in dataset_keys:
            result_file = RESULTS_DIR / (f"stream_commit_{suffix}_{ds}.json")
            if args.summary_only:
                if not result_file.exists():
                    continue
                with open(result_file) as f:
                    raw_rows = json.load(f)
            else:
                raw_rows = stream_run(
                    ds, variant, force=args.force, verbose=args.verbose
                )
            for r in raw_rows:
                rr = dict(r)
                rr["_dataset"] = ds
                variant_rows[variant].append(rr)
            all_rows.extend(summarize(ds, variant, raw_rows))

    # Rolled-up summary
    out_path = RESULTS_DIR / "stream_commit_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"\nSaved summary -> {out_path}")

    for K in BUDGETS:
        print_cross_dataset(all_rows, K)
        print_overall(all_rows, K)
        print_hard_categories(all_rows, K)

    print_cost_summary(variant_rows)


if __name__ == "__main__":
    main()
