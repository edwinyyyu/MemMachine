"""Self-dispatching v3: SIMPLE branch is BYTE-IDENTICAL to META_V2F_PROMPT.

Motivation
----------
Self-dispatch v2 (self_dispatch_v2.py) beat v2f on LoCoMo at K=50 (+4.2pp)
but still lagged v2f by 8.1pp at K=20 on LoCoMo. Inspection shows the v2
SIMPLE branch prompt is structurally similar to META_V2F_PROMPT but NOT
byte-identical. v3 eliminates that variable by emitting EXACTLY the v2f
format for the SIMPLE branch.

Only changes vs a pure v2f prompt:
  1. A preceding classification line: "CLASSIFICATION: SIMPLE | COMPLEX"
  2. Branching on it
  3. For SIMPLE: the verbatim META_V2F_PROMPT body (same wording of
     "conversation history", the "First, briefly assess:" intro, the
     "If the question implies MULTIPLE items..." completeness line, the
     "Do NOT write questions..." line, and the ASSESSMENT/CUE/CUE format).
  4. For COMPLEX: the CoT 4-step reasoning (identical to self_v2).

Budget is identical to self_v2:
  SIMPLE  path: initial_k=10, 2 cues x 10 per_cue_k
  COMPLEX path: initial_k=10, 5 cues x 4  per_cue_k

Usage:
    uv run python self_dispatch_v3.py [--force] [--dataset NAME]
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
CACHE_FILE_EMB = CACHE_DIR / "self_v3_embedding_cache.json"
CACHE_FILE_LLM = CACHE_DIR / "self_v3_llm_cache.json"
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
class SelfV3EmbeddingCache(EmbeddingCache):
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


class SelfV3LLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        # Prefer self_v3 cache first (exact prompt match), then fallback.
        for p in sorted(
            self.cache_dir.glob("*llm_cache.json"),
            key=lambda x: 0 if x.name.startswith("self_v3") else 1,
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
def _format_segments_v2f(
    segments: list[Segment], max_items: int = 16, max_chars: int = 250
) -> str:
    """Match budget_aware_eval._format_segments defaults (16 items, 250 chars)
    so the v2f context is byte-identical."""
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}"
        for seg in sorted_segs
    )


def _build_v2f_context_section(all_segments: list[Segment]) -> str:
    """Exact copy of budget_aware_eval._build_context_section."""
    if not all_segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    return (
        "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
        + _format_segments_v2f(all_segments)
    )


def _format_segments_cot(
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
            val = line[len(key):].strip()
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
# Self-dispatching v3 prompt
# ---------------------------------------------------------------------------
# Design: ONE prompt, classifies inline, then emits EITHER the byte-identical
# v2f format or the CoT format.
#
# SIMPLE branch body: EXACTLY the META_V2F_PROMPT from prompt_optimization.py
# (and V2F_PROMPT_TEMPLATE from budget_aware_eval.py rendered with num_cues=2).
# Only additions are a preceding CLASSIFICATION line and a wrapper that picks
# this format.
#
# COMPLEX branch: CoT 4-step reasoning from cot_universal.COT_PROMPT.
SELF_DISPATCH_V3_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

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
CLASSIFICATION: SIMPLE
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else.

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
class SelfV3Result:
    segments: list[Segment]
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


class SelfDispatchV3:
    """Self-dispatching v3: single prompt that classifies and dispatches.

    SIMPLE branch uses byte-identical v2f format: 2 cues x per_cue_k_simple
    COMPLEX branch uses CoT 4-step format: num_cues cues x per_cue_k_complex
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
        self.embedding_cache = SelfV3EmbeddingCache()
        self.llm_cache = SelfV3LLMCache()
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
        response = self.client.embeddings.create(
            model=EMBED_MODEL, input=[text]
        )
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

    def retrieve(self, question: str, conversation_id: str) -> SelfV3Result:
        exclude: set[int] = set()
        all_segs: list[Segment] = []
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
            context_section = _build_v2f_context_section(all_segs)
            prompt = SELF_DISPATCH_V3_PROMPT.format(
                question=question,
                context_section=context_section,
                num_cues=self.num_cues_complex,
            )
            response = self.llm_call(prompt)

            classification = _parse_classification(response)
            classifications.append(classification)

            assessment = ""
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("ASSESSMENT:"):
                    assessment = line[len("ASSESSMENT:"):].strip()
                    break

            if classification == "SIMPLE":
                cap = self.num_cues_simple
                per_cue_k = self.per_cue_k_simple
            else:
                cap = self.num_cues_complex
                per_cue_k = self.per_cue_k_complex
            cues = _parse_cues(response, "CUE:")[:cap]

            round_log.append({
                "round": round_i,
                "classification": classification,
                "assessment": assessment,
                "cues": cues,
            })
            if not cues:
                break
            for cue in cues:
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb, top_k=per_cue_k,
                    conversation_id=conversation_id, exclude_indices=exclude,
                )
                for s in result.segments:
                    if s.index not in exclude:
                        all_segs.append(s)
                        exclude.add(s.index)

        final_class = (
            "SIMPLE"
            if classifications and classifications.count("SIMPLE")
            > classifications.count("COMPLEX")
            else "COMPLEX"
        )

        return SelfV3Result(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "self_dispatch_v3",
                "classifications": classifications,
                "final_classification": final_class,
                "rounds": round_log,
                "total_segments": len(all_segs),
            },
        )


# ---------------------------------------------------------------------------
# Fair K-budget evaluation
# ---------------------------------------------------------------------------
def compute_recall(retrieved_turn_ids: set[int],
                   source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(arch: SelfDispatchV3, question: dict,
                 verbose: bool = False) -> dict:
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
    baseline = arch.store.search(
        q_emb, top_k=max_b, conversation_id=conv_id
    )

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
        "self_v3_recalls": recalls,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "rounds_metadata": result.metadata["rounds"],
    }
    if verbose:
        print(
            f"    [{row['classification']}] pool={len(arch_segments)} "
            f"r@20: base={baseline_recalls['r@20']:.3f} "
            f"v3={recalls['r@20']:.3f}  "
            f"r@50: base={baseline_recalls['r@50']:.3f} "
            f"v3={recalls['r@50']:.3f}  "
            f"emb={arch.embed_calls} llm={arch.llm_calls}"
        )
    return row


# ---------------------------------------------------------------------------
# Datasets (identical to self_v2)
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
def load_budget_recall_by_qkey(
    arch_name: str, dataset_key: str
) -> dict[tuple, float]:
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


def load_self_v2_recall_by_qkey(
    dataset_key: str,
) -> dict[tuple, dict[int, float]]:
    path = RESULTS_DIR / f"self_v2_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    out: dict[tuple, dict[int, float]] = {}
    for r in rows:
        key = (r["conversation_id"], r.get("question_index"))
        out[key] = {
            20: r["self_v2_recalls"]["r@20"],
            50: r["self_v2_recalls"]["r@50"],
        }
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def self_v3_run(
    dataset_key: str, force: bool = False, verbose: bool = False
) -> list[dict]:
    result_file = RESULTS_DIR / f"self_v3_{dataset_key}.json"
    if result_file.exists() and not force:
        with open(result_file) as f:
            return json.load(f)

    qs, store = load_dataset(dataset_key)
    arch = SelfDispatchV3(store)
    print(
        f"\n>>> SelfV3 on {dataset_key}: {len(qs)} questions, "
        f"{len(store.segments)} segments"
    )
    rows: list[dict] = []
    for i, q in enumerate(qs):
        q_short = q["question"][:60].replace("\n", " ")
        print(
            f"  [{i+1}/{len(qs)}] {q['category']}: {q_short}...",
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
# Per-dataset + per-category summary
# ---------------------------------------------------------------------------
def compare_per_category(
    dataset_key: str, self_rows: list[dict]
) -> list[dict]:
    budget_by_arch: dict[str, dict[int, dict]] = {}
    for arch in ("baseline", "v2f_tight"):
        budget_by_arch[arch] = {
            K: load_budget_recall_by_qkey(f"{arch}_{K}", dataset_key)
            for K in BUDGETS
        }
    cot_by_q = load_cot_recall_by_qkey(dataset_key)
    sv2_by_q = load_self_v2_recall_by_qkey(dataset_key)

    rows_by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in self_rows:
        rows_by_cat[r["category"]].append(r)

    out: list[dict] = []
    for K in BUDGETS:
        for cat, rows in sorted(rows_by_cat.items()):
            n = len(rows)
            if n == 0:
                continue
            self_recalls = [r["self_v3_recalls"][f"r@{K}"] for r in rows]
            self_mean = sum(self_recalls) / n

            b_vals, v2f_vals, cot_vals, sv2_vals = [], [], [], []
            for r in rows:
                key = (r["conversation_id"], r["question_index"])
                if key in budget_by_arch["baseline"][K]:
                    b_vals.append(budget_by_arch["baseline"][K][key])
                if key in budget_by_arch["v2f_tight"][K]:
                    v2f_vals.append(budget_by_arch["v2f_tight"][K][key])
                if key in cot_by_q:
                    cot_vals.append(cot_by_q[key][K])
                if key in sv2_by_q:
                    sv2_vals.append(sv2_by_q[key][K])

            def _mean(xs: list[float]) -> float | None:
                if not xs:
                    return None
                return sum(xs) / len(xs)

            b_mean = _mean(b_vals)
            v2f_mean = _mean(v2f_vals)
            cot_mean = _mean(cot_vals)
            sv2_mean = _mean(sv2_vals)

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
                "self_v2": sv2_mean,
                "self_v3": self_mean,
                "vs_v2f": (
                    self_mean - v2f_mean if v2f_mean is not None else None
                ),
                "vs_cot": (
                    self_mean - cot_mean if cot_mean is not None else None
                ),
                "vs_sv2": (
                    self_mean - sv2_mean if sv2_mean is not None else None
                ),
                "vs_baseline": (
                    self_mean - b_mean if b_mean is not None else None
                ),
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
    print(f"\n{'-'*124}")
    print(f"OVERALL per DATASET at K={K}")
    print(f"{'-'*124}")
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows_k:
        by_ds[r["dataset"]].append(r)

    hdr = (
        f"{'Dataset':<14s} {'n':>3s} "
        f"{'Base':>6s} {'v2f':>6s} {'CoT':>6s} {'SV2':>6s} {'SelfV3':>7s}  "
        f"{'vs_v2f':>7s} {'vs_CoT':>7s} {'vs_SV2':>7s} {'vs_base':>7s}  "
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
        sv2 = _weighted("self_v2")
        sv3 = _weighted("self_v3")
        vv = (sv3 - v2f) if (sv3 is not None and v2f is not None) else None
        vc = (sv3 - cot) if (sv3 is not None and cot is not None) else None
        vs = (sv3 - sv2) if (sv3 is not None and sv2 is not None) else None
        vb = (sv3 - b) if (sv3 is not None and b is not None) else None
        sc_str = f"{simple_tot}/{complex_tot}"
        print(
            f"{ds:<14s} {total_n:>3d} "
            f"{fmt_cell(b)} {fmt_cell(v2f)} {fmt_cell(cot)} "
            f"{fmt_cell(sv2)} {fmt_cell(sv3)}  "
            f"{fmt_cell(vv, True)} {fmt_cell(vc, True)} "
            f"{fmt_cell(vs, True)} {fmt_cell(vb, True)}  "
            f"{sc_str:>7s}"
        )


def print_locomo_breakdown(all_rows: list[dict], K: int) -> None:
    """LoCoMo single_hop / temporal breakdown where the K=20 gap is focused."""
    cats = {"locomo_single_hop", "locomo_temporal"}
    rows = [r for r in all_rows
            if r["K"] == K and r["dataset"] == "locomo_30q"
            and r["category"] in cats]
    if not rows:
        return
    print(f"\n{'='*128}")
    print(f"LoCoMo BREAKDOWN at K={K}  (focus on residual K=20 gap)")
    print(f"{'='*128}")
    hdr = (
        f"{'Category':<22s} {'n':>3s} {'S/C':>7s} "
        f"{'Base':>6s} {'v2f':>6s} {'CoT':>6s} {'SV2':>6s} {'SelfV3':>7s}  "
        f"{'vs_v2f':>7s} {'vs_CoT':>7s} {'vs_SV2':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["category"]):
        sc_str = f"{r['simple_n']}/{r['complex_n']}"
        print(
            f"{r['category']:<22s} {r['n']:>3d} {sc_str:>7s} "
            f"{fmt_cell(r['baseline'])} {fmt_cell(r['v2f'])} "
            f"{fmt_cell(r['cot'])} {fmt_cell(r['self_v2'])} "
            f"{fmt_cell(r['self_v3'])}  "
            f"{fmt_cell(r['vs_v2f'], True)} "
            f"{fmt_cell(r['vs_cot'], True)} "
            f"{fmt_cell(r['vs_sv2'], True)}"
        )


def print_classification_spotcheck(
    all_self_rows: dict[str, list[dict]],
) -> None:
    print(f"\n{'='*108}")
    print("CLASSIFICATION SPOT-CHECK")
    print(f"{'='*108}")
    for ds, rows in all_self_rows.items():
        print(f"\n-- {ds} --")
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r)
        for cat, rs in sorted(by_cat.items()):
            s_n = sum(1 for r in rs if r["classification"] == "SIMPLE")
            c_n = len(rs) - s_n
            pct_simple = 100 * s_n / len(rs)
            print(f"  {cat:<28s} n={len(rs):>2d}  "
                  f"SIMPLE={s_n:>2d}  COMPLEX={c_n:>2d}  "
                  f"({pct_simple:.0f}% simple)")
            simples = [r for r in rs if r["classification"] == "SIMPLE"][:2]
            complexes = [r for r in rs if r["classification"] == "COMPLEX"][:2]
            for r in simples:
                q = r["question"][:70].replace("\n", " ")
                print(f"      SIMPLE  : {q}")
            for r in complexes:
                q = r["question"][:70].replace("\n", " ")
                print(f"      COMPLEX : {q}")


def print_universal_dominance_check(all_rows: list[dict]) -> None:
    """Key question: does v3 beat-or-tie v2f on EVERY dataset at K=20 AND K=50?"""
    print(f"\n{'='*100}")
    print("UNIVERSAL DOMINANCE vs v2f (key question)")
    print(f"{'='*100}")
    by_ds_k: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_ds_k[(r["dataset"], r["K"])].append(r)

    dominates = True
    for K in BUDGETS:
        for ds in sorted(set(r["dataset"] for r in all_rows)):
            rows = by_ds_k.get((ds, K), [])
            if not rows:
                continue

            def _weighted(key: str) -> float | None:
                vals = [(r[key], r["n"]) for r in rows if r[key] is not None]
                if not vals:
                    return None
                tot = sum(n for _, n in vals)
                return sum(v * n for v, n in vals) / tot

            sv3 = _weighted("self_v3")
            v2f = _weighted("v2f")
            if sv3 is None or v2f is None:
                continue
            delta = sv3 - v2f
            tick = "OK" if delta >= -0.001 else "FAIL"
            if delta < -0.001:
                dominates = False
            print(
                f"  K={K:<3d}  {ds:<14s}  "
                f"v3={sv3:.3f}  v2f={v2f:.3f}  "
                f"delta={delta:+.3f}   {tick}"
            )
    print(f"\n  STRICT UNIVERSAL DOMINANCE over v2f: "
          f"{'YES' if dominates else 'NO'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Rerun even if result file exists")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASETS.keys()),
                        help="Restrict to a single dataset")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_keys = (
        [args.dataset] if args.dataset else list(DATASETS.keys())
    )

    all_rows: list[dict] = []
    all_self_rows: dict[str, list[dict]] = {}
    for ds in dataset_keys:
        self_rows = self_v3_run(ds, force=args.force, verbose=args.verbose)
        all_self_rows[ds] = self_rows
        all_rows.extend(compare_per_category(ds, self_rows))

    out_path = RESULTS_DIR / "self_v3_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved summary -> {out_path}")

    for K in BUDGETS:
        print_overall_by_dataset(all_rows, K)

    for K in BUDGETS:
        print_locomo_breakdown(all_rows, K)

    print_classification_spotcheck(all_self_rows)

    print_universal_dominance_check(all_rows)


if __name__ == "__main__":
    main()
