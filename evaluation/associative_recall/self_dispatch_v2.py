"""Self-dispatching v2: single prompt that classifies the query then picks
reasoning mode, where the SIMPLE path uses the v2f prompt structure and
COMPLEX path uses the CoT 4-step reasoning.

Motivation
----------
Self-dispatch CoT v1 (self_dispatch_cot.py) proved classification works and
rescued CoT regressions on synthetic datasets, but collapsed on LoCoMo
(-33.6pp vs v2f at r@20). Root cause: the v1 SIMPLE path emitted bare
keyword bundles while v2f emits "ASSESSMENT: ... CUE: ... CUE: ..." using
question vocabulary. LoCoMo wins are driven by the v2f prompt STRUCTURE
(assessment + two cues in conversation-style vocabulary), not by reasoning.

This v2 fixes the collapse by dispatching the SIMPLE path to the exact v2f
prompt format, while still dispatching the COMPLEX path to CoT's 4-step
chain-of-thought.

Budget:
  SIMPLE  path: initial_k=10, 2 cues x 10 per_cue_k (matches v2f_tight at K=20)
  COMPLEX path: initial_k=10, 5 cues x 4  per_cue_k (matches CoT)

Both dispatch at the CUE level during the SAME round, so a single prompt per
round suffices. At K=20 with 2 rounds, cost is 2 LLM calls (comparable to
v2f's 2 rounds x 1 call and CoT's 2 rounds x 1 call).

Usage:
    uv run python self_dispatch_v2.py [--force] [--dataset NAME]
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
CACHE_FILE_EMB = CACHE_DIR / "self_v2_embedding_cache.json"
CACHE_FILE_LLM = CACHE_DIR / "self_v2_llm_cache.json"
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
class SelfV2EmbeddingCache(EmbeddingCache):
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


class SelfV2LLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        # Prefer self_v2 cache first (exact prompt match), then all others.
        for p in sorted(
            self.cache_dir.glob("*llm_cache.json"),
            key=lambda x: 0 if x.name.startswith("self_v2") else 1,
        ):
            try:
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v and k not in self._cache:
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
    segments: list[Segment], max_items: int = 14, max_chars: int = 260
) -> str:
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in sorted_segs
    )


def _parse_cues(text: str, key: str = "CUE:") -> list[str]:
    out: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith(key.upper()):
            val = line[len(key) :].strip()
            if val:
                out.append(val)
    return out


def _parse_classification(text: str) -> str:
    """Extract SIMPLE or COMPLEX from a CLASSIFICATION: line. Default COMPLEX
    if missing so we err on the side of reasoning when ambiguous."""
    for line in text.strip().split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("CLASSIFICATION:") or upper.startswith("CLASS:"):
            val = line.split(":", 1)[1].strip().upper()
            if "SIMPLE" in val:
                return "SIMPLE"
            if "COMPLEX" in val:
                return "COMPLEX"
    head = "\n".join(text.strip().split("\n")[:3]).upper()
    if "SIMPLE" in head and "COMPLEX" not in head:
        return "SIMPLE"
    if "COMPLEX" in head and "SIMPLE" not in head:
        return "COMPLEX"
    return "COMPLEX"


# ---------------------------------------------------------------------------
# Self-dispatching v2 prompt
# ---------------------------------------------------------------------------
# Design: ONE prompt, classifies inline, then emits EITHER the v2f format or
# the CoT format. No separate classification LLM call.
#
# SIMPLE branch mirrors V2F_PROMPT (best_shot.py:171-196):
#   - Brief assessment sentence(s)
#   - Completeness note ("If the question implies MULTIPLE items...")
#   - Anti-question instruction
#   - Exactly 2 cues, specific conversation-style vocabulary
#
# COMPLEX branch mirrors COT_PROMPT (cot_universal.py:163-198):
#   - 4-step reasoning (current vocabulary, related vocabulary, next link,
#     alternative names), then up to num_cues cues.
SELF_DISPATCH_V2_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity \
against stored conversation turns.

Question: {question}

RETRIEVED SO FAR ({num_segs} segments, chronological):
{all_segs}

ALREADY SEARCHED FOR (do NOT repeat):
{explored}

STEP 0 — Classify the question.
- SIMPLE: a direct fact lookup about a single entity (e.g. "When did \
Caroline join?", "What is Bob's favorite color?", "Where does Alice \
live?", "Who invited X?"). The answer is ONE value from ONE turn and no \
chain of reasoning is needed to find it.
- COMPLEX: multi-step reasoning, scattered evidence across several turns, \
chain of dependencies where each link uses different vocabulary, evolving \
terminology / aliases for the same thing, list-completeness / "all/every" \
questions, logical constraints ("X AND NOT Y"), or procedural / sequential \
reasoning.

Then dispatch on the classification and emit EXACTLY one of the two \
formats below. Nothing else.

=== FORMAT A — if SIMPLE ===
Briefly assess: how well is this search going given what's been retrieved? \
What kind of content is still missing? Should you search for similar \
content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate EXACTLY 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Emit EXACTLY:
CLASSIFICATION: SIMPLE
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>

=== FORMAT B — if COMPLEX ===
Think step by step:
STEP 1: What specific terminology appears in the retrieved segments (names, \
tools, symptoms, decisions, tickets, numbers)?
STEP 2: What RELATED terminology might be used elsewhere? (aliases, \
codenames, abbreviations, informal references like "the bird", "that \
thing", ...)
STEP 3: If this is a CHAIN (A -> B -> C where each link has different \
vocabulary), what is the NEXT link to search for?
STEP 4: If this topic has ALTERNATIVE NAMES, what are they? Include every \
alias you can justify from the retrieved text or reasonable guesses.

Then generate up to {num_cues} search cues that EXTEND the retrieval in \
the most promising directions. A cue may be:
  - a short alias/name phrase (1-5 words) that might appear inline
  - a 1-2 sentence plausible conversation snippet targeting the next link

Prefer DIVERSE cues (cover multiple aliases and/or multiple chain links). \
Do not rephrase the question.

Emit EXACTLY:
CLASSIFICATION: COMPLEX
STEP 1: <current vocabulary>
STEP 2: <related vocabulary>
STEP 3: <next link>
STEP 4: <alternative names>
CUE: <text>
CUE: <text>
(up to {num_cues} cues)

Nothing else."""


@dataclass
class SelfV2Result:
    segments: list[Segment]
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


class SelfDispatchV2:
    """Self-dispatching v2: single prompt that classifies and dispatches.

    SIMPLE branch uses v2f format:  2 cues x per_cue_k_simple = 20 pool additions
    COMPLEX branch uses CoT format: num_cues cues x per_cue_k_complex = 20 pool
    With initial_k=10 and 2 rounds, both branches target a pool of ~30
    unique segments so downstream fair K-budget at K=20 compares apples to
    apples with v2f_tight_20 and the CoT baseline.
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        initial_k: int = 10,
        num_cues_complex: int = 5,
        num_cues_simple: int = 2,
        per_cue_k_complex: int = 4,
        per_cue_k_simple: int = 10,
        rounds: int = 2,
    ) -> None:
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = SelfV2EmbeddingCache()
        self.llm_cache = SelfV2LLMCache()
        self.embed_calls = 0
        self.llm_calls = 0
        self.initial_k = initial_k
        self.num_cues_complex = num_cues_complex
        self.num_cues_simple = num_cues_simple
        self.per_cue_k_complex = per_cue_k_complex
        self.per_cue_k_simple = per_cue_k_simple
        self.rounds = rounds

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

    def retrieve(self, question: str, conversation_id: str) -> SelfV2Result:
        exclude: set[int] = set()
        all_segs: list[Segment] = []
        explored: list[str] = []
        round_log: list[dict] = []

        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        classifications: list[str] = []

        for round_i in range(self.rounds):
            prompt = SELF_DISPATCH_V2_PROMPT.format(
                question=question,
                all_segs=_format_segments(all_segs, max_items=14),
                num_segs=len(all_segs),
                explored=(
                    "\n".join(f"- {c}" for c in explored) if explored else "(none yet)"
                ),
                num_cues=self.num_cues_complex,
            )
            response = self.llm_call(prompt)

            classification = _parse_classification(response)
            classifications.append(classification)

            # Pull assessment (SIMPLE branch) or reason/step lines (COMPLEX).
            assessment = ""
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("ASSESSMENT:"):
                    assessment = line[len("ASSESSMENT:") :].strip()
                    break

            if classification == "SIMPLE":
                cap = self.num_cues_simple
                per_cue_k = self.per_cue_k_simple
            else:
                cap = self.num_cues_complex
                per_cue_k = self.per_cue_k_complex
            cues = _parse_cues(response, "CUE:")[:cap]

            round_log.append(
                {
                    "round": round_i,
                    "classification": classification,
                    "assessment": assessment,
                    "cues": cues,
                }
            )
            if not cues:
                break
            for cue in cues:
                if cue in explored:
                    continue
                explored.append(cue)
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb,
                    top_k=per_cue_k,
                    conversation_id=conversation_id,
                    exclude_indices=exclude,
                )
                for s in result.segments:
                    if s.index not in exclude:
                        all_segs.append(s)
                        exclude.add(s.index)

        final_class = (
            "SIMPLE"
            if classifications
            and classifications.count("SIMPLE") > classifications.count("COMPLEX")
            else "COMPLEX"
        )

        return SelfV2Result(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "self_dispatch_v2",
                "classifications": classifications,
                "final_classification": final_class,
                "rounds": round_log,
                "total_segments": len(all_segs),
            },
        )


# ---------------------------------------------------------------------------
# Fair K-budget evaluation
# ---------------------------------------------------------------------------
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(arch: SelfDispatchV2, question: dict, verbose: bool = False) -> dict:
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

    q_emb = arch.embed_text(q_text)
    max_b = max(BUDGETS)
    baseline = arch.store.search(q_emb, top_k=max_b, conversation_id=conv_id)

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
        "pool_size": len(arch_segments),
        "classification": result.metadata["final_classification"],
        "classifications_per_round": result.metadata["classifications"],
        "baseline_recalls": baseline_recalls,
        "self_v2_recalls": recalls,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "rounds_metadata": result.metadata["rounds"],
    }
    if verbose:
        print(
            f"    [{row['classification']}] pool={len(arch_segments)} "
            f"r@20: base={baseline_recalls['r@20']:.3f} "
            f"v2={recalls['r@20']:.3f}  "
            f"r@50: base={baseline_recalls['r@50']:.3f} "
            f"v2={recalls['r@50']:.3f}  "
            f"emb={arch.embed_calls} llm={arch.llm_calls}"
        )
    return row


# ---------------------------------------------------------------------------
# Datasets (identical to cot_universal.py)
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
# Load prior results for comparison
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


def load_cot_recall_by_qkey(dataset_key: str) -> dict[tuple, dict[int, float]]:
    path = RESULTS_DIR / f"cot_chain_of_thought_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    out: dict[tuple, dict[int, float]] = {}
    for r in rows:
        key = (r["conversation_id"], r.get("question_index"))
        out[key] = {
            20: r["cot_recalls"]["r@20"],
            50: r["cot_recalls"]["r@50"],
        }
    return out


def load_self_cot_v1_recall_by_qkey(
    dataset_key: str,
) -> dict[tuple, dict[int, float]]:
    path = RESULTS_DIR / f"self_cot_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    out: dict[tuple, dict[int, float]] = {}
    for r in rows:
        key = (r["conversation_id"], r.get("question_index"))
        out[key] = {
            20: r["self_cot_recalls"]["r@20"],
            50: r["self_cot_recalls"]["r@50"],
        }
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def self_v2_run(
    dataset_key: str, force: bool = False, verbose: bool = False
) -> list[dict]:
    result_file = RESULTS_DIR / f"self_v2_{dataset_key}.json"
    if result_file.exists() and not force:
        with open(result_file) as f:
            return json.load(f)

    qs, store = load_dataset(dataset_key)
    arch = SelfDispatchV2(store)
    print(
        f"\n>>> SelfV2 on {dataset_key}: {len(qs)} questions, "
        f"{len(store.segments)} segments"
    )
    rows: list[dict] = []
    for i, q in enumerate(qs):
        q_short = q["question"][:60].replace("\n", " ")
        print(
            f"  [{i + 1}/{len(qs)}] {q['category']}: {q_short}...",
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
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"  Saved -> {result_file}")
    return rows


# ---------------------------------------------------------------------------
# Per-category summary
# ---------------------------------------------------------------------------
def compare_per_category(dataset_key: str, self_rows: list[dict]) -> list[dict]:
    budget_by_arch: dict[str, dict[int, dict]] = {}
    for arch in ("baseline", "v2f_tight"):
        budget_by_arch[arch] = {
            K: load_budget_recall_by_qkey(f"{arch}_{K}", dataset_key) for K in BUDGETS
        }
    cot_by_q = load_cot_recall_by_qkey(dataset_key)
    scv1_by_q = load_self_cot_v1_recall_by_qkey(dataset_key)

    rows_by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in self_rows:
        rows_by_cat[r["category"]].append(r)

    out: list[dict] = []
    for K in BUDGETS:
        for cat, rows in sorted(rows_by_cat.items()):
            n = len(rows)
            if n == 0:
                continue
            self_recalls = [r["self_v2_recalls"][f"r@{K}"] for r in rows]
            self_mean = sum(self_recalls) / n

            b_vals, v2f_vals, cot_vals, scv1_vals = [], [], [], []
            for r in rows:
                key = (r["conversation_id"], r["question_index"])
                if key in budget_by_arch["baseline"][K]:
                    b_vals.append(budget_by_arch["baseline"][K][key])
                if key in budget_by_arch["v2f_tight"][K]:
                    v2f_vals.append(budget_by_arch["v2f_tight"][K][key])
                if key in cot_by_q:
                    cot_vals.append(cot_by_q[key][K])
                if key in scv1_by_q:
                    scv1_vals.append(scv1_by_q[key][K])

            def _mean(xs: list[float]) -> float | None:
                if not xs:
                    return None
                return sum(xs) / len(xs)

            b_mean = _mean(b_vals)
            v2f_mean = _mean(v2f_vals)
            cot_mean = _mean(cot_vals)
            scv1_mean = _mean(scv1_vals)

            simple_n = sum(1 for r in rows if r["classification"] == "SIMPLE")
            complex_n = n - simple_n

            row = {
                "dataset": dataset_key,
                "category": cat,
                "K": K,
                "n": n,
                "baseline": b_mean,
                "v2f": v2f_mean,
                "cot": cot_mean,
                "self_cot_v1": scv1_mean,
                "self_v2": self_mean,
                "vs_v2f": (self_mean - v2f_mean if v2f_mean is not None else None),
                "vs_cot": (self_mean - cot_mean if cot_mean is not None else None),
                "vs_scv1": (self_mean - scv1_mean if scv1_mean is not None else None),
                "vs_baseline": (self_mean - b_mean if b_mean is not None else None),
                "simple_n": simple_n,
                "complex_n": complex_n,
            }
            out.append(row)
    return out


def fmt_cell(val: float | None, plus_sign: bool = False) -> str:
    if val is None:
        return "    —"
    s = f"{val:+.3f}" if plus_sign else f"{val:.3f}"
    return f"{s:>6s}"


def print_overall_by_dataset(all_rows: list[dict], K: int) -> None:
    rows_k = [r for r in all_rows if r["K"] == K]
    if not rows_k:
        return
    print(f"\n{'-' * 124}")
    print(f"OVERALL per DATASET at K={K}")
    print(f"{'-' * 124}")
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows_k:
        by_ds[r["dataset"]].append(r)

    hdr = (
        f"{'Dataset':<14s} {'n':>3s} "
        f"{'Base':>6s} {'v2f':>6s} {'CoT':>6s} {'SCv1':>6s} {'SelfV2':>7s}  "
        f"{'vs_v2f':>7s} {'vs_CoT':>7s} {'vs_SCv1':>7s} {'vs_base':>7s}  "
        f"{'S/C':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for ds, rows in sorted(by_ds.items()):
        total_n = sum(r["n"] for r in rows)
        simple_tot = sum(r["simple_n"] for r in rows)
        complex_tot = sum(r["complex_n"] for r in rows)

        def _weighted(key: str) -> float | None:
            vals = [(r[key], r["n"]) for r in rows if r[key] is not None]
            if not vals:
                return None
            tot = sum(n for _, n in vals)
            return sum(v * n for v, n in vals) / tot

        b = _weighted("baseline")
        v2f = _weighted("v2f")
        cot = _weighted("cot")
        scv1 = _weighted("self_cot_v1")
        sv2 = _weighted("self_v2")
        vv = (sv2 - v2f) if (sv2 is not None and v2f is not None) else None
        vc = (sv2 - cot) if (sv2 is not None and cot is not None) else None
        vs = (sv2 - scv1) if (sv2 is not None and scv1 is not None) else None
        vb = (sv2 - b) if (sv2 is not None and b is not None) else None
        sc_str = f"{simple_tot}/{complex_tot}"
        print(
            f"{ds:<14s} {total_n:>3d} "
            f"{fmt_cell(b)} {fmt_cell(v2f)} {fmt_cell(cot)} "
            f"{fmt_cell(scv1)} {fmt_cell(sv2)}  "
            f"{fmt_cell(vv, True)} {fmt_cell(vc, True)} "
            f"{fmt_cell(vs, True)} {fmt_cell(vb, True)}  "
            f"{sc_str:>7s}"
        )


def print_per_category_table(
    all_rows: list[dict],
    K: int,
    focus_categories: list[str] | None = None,
) -> None:
    rows = [r for r in all_rows if r["K"] == K]
    if focus_categories is not None:
        rows = [r for r in rows if r["category"] in focus_categories]
    if not rows:
        return
    title = "PER-CATEGORY (hardest)" if focus_categories else "PER-CATEGORY"
    print(f"\n{'=' * 136}")
    print(f"{title} at K={K}")
    print(f"{'=' * 136}")
    hdr = (
        f"{'Dataset':<14s} {'Category':<26s} {'n':>3s} {'S/C':>7s} "
        f"{'Base':>6s} {'v2f':>6s} {'CoT':>6s} {'SCv1':>6s} {'SelfV2':>7s}  "
        f"{'vs_v2f':>7s} {'vs_CoT':>7s} {'vs_SCv1':>7s} {'vs_base':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    last_ds = None
    for r in sorted(rows, key=lambda x: (x["dataset"], x["category"])):
        if last_ds is not None and r["dataset"] != last_ds:
            print()
        last_ds = r["dataset"]
        sc_str = f"{r['simple_n']}/{r['complex_n']}"
        print(
            f"{r['dataset']:<14s} {r['category']:<26s} {r['n']:>3d} "
            f"{sc_str:>7s} "
            f"{fmt_cell(r['baseline'])} {fmt_cell(r['v2f'])} "
            f"{fmt_cell(r['cot'])} {fmt_cell(r['self_cot_v1'])} "
            f"{fmt_cell(r['self_v2'])}  "
            f"{fmt_cell(r['vs_v2f'], True)} "
            f"{fmt_cell(r['vs_cot'], True)} "
            f"{fmt_cell(r['vs_scv1'], True)} "
            f"{fmt_cell(r['vs_baseline'], True)}"
        )


def print_classification_spotcheck(all_self_rows: dict[str, list[dict]]) -> None:
    print(f"\n{'=' * 108}")
    print("CLASSIFICATION SPOT-CHECK")
    print(f"{'=' * 108}")
    for ds, rows in all_self_rows.items():
        print(f"\n-- {ds} --")
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r)
        for cat, rs in sorted(by_cat.items()):
            s_n = sum(1 for r in rs if r["classification"] == "SIMPLE")
            c_n = len(rs) - s_n
            pct_simple = 100 * s_n / len(rs)
            print(
                f"  {cat:<28s} n={len(rs):>2d}  "
                f"SIMPLE={s_n:>2d}  COMPLEX={c_n:>2d}  "
                f"({pct_simple:.0f}% simple)"
            )
            simples = [r for r in rs if r["classification"] == "SIMPLE"][:2]
            complexes = [r for r in rs if r["classification"] == "COMPLEX"][:2]
            for r in simples:
                q = r["question"][:70].replace("\n", " ")
                print(f"      SIMPLE  : {q}")
            for r in complexes:
                q = r["question"][:70].replace("\n", " ")
                print(f"      COMPLEX : {q}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
HARDEST_CATEGORIES = {
    "locomo_30q": ["locomo_single_hop"],
    "synthetic_19q": [
        "control",
        "single_hop",
        "inference",
        "completeness",
        "conjunction",
    ],
    "puzzle_16q": ["sequential_chain", "logic_constraint", "procedural"],
    "advanced_23q": [
        "evolving_terminology",
        "adversarial_distractor",
        "logic_constraint",
    ],
}


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
        help="Restrict to a single dataset",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_keys = [args.dataset] if args.dataset else list(DATASETS.keys())

    all_rows: list[dict] = []
    all_self_rows: dict[str, list[dict]] = {}
    for ds in dataset_keys:
        self_rows = self_v2_run(ds, force=args.force, verbose=args.verbose)
        all_self_rows[ds] = self_rows
        all_rows.extend(compare_per_category(ds, self_rows))

    out_path = RESULTS_DIR / "self_v2_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved summary -> {out_path}")

    for K in BUDGETS:
        print_overall_by_dataset(all_rows, K)

    all_focus: list[str] = []
    for v in HARDEST_CATEGORIES.values():
        all_focus.extend(v)
    for K in BUDGETS:
        print_per_category_table(all_rows, K, focus_categories=all_focus)

    for K in BUDGETS:
        print_per_category_table(all_rows, K)

    print_classification_spotcheck(all_self_rows)


if __name__ == "__main__":
    main()
