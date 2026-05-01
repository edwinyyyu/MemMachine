"""Two-call dispatch architecture: classifier + specialist as SEPARATE LLM calls.

Motivation
----------
self_dispatch_v3 failed because a single LLM call that does BOTH classification
AND cue generation leaks the "meta-reasoning about classification" mode into
the cue output. Even byte-identical v2f SIMPLE-branch content degraded because
the classification preamble changed the model's mode mid-generation.

The fix tested here: TWO SEPARATE LLM CALLS.
  - Call 1 (classifier): only classifies the query. Output is exactly the word
    SIMPLE or COMPLEX. No cue generation, no context, no assessment.
  - Call 2 (specialist): clean context. Receives ONLY the classification
    decision and runs the appropriate specialist prompt:
      * SIMPLE  -> META_V2F_PROMPT (verbatim from prompt_optimization.py)
      * COMPLEX -> COT_PROMPT      (verbatim from chain_retrieval.py)
    The specialist prompt text is byte-identical to what v2f / CoT would use
    standalone. The specialist sees no hint that classification happened.

Budget matches self_v2 / self_v3 for fair K-budget comparison:
  SIMPLE  path: initial_k=10, 2 cues x 10 per_cue_k, 2 rounds
  COMPLEX path: initial_k=10, 5 cues x 4  per_cue_k, 2 rounds

The classifier runs ONCE per question (not per round); its answer is reused
across rounds. This costs +1 LLM call vs self_v2/self_v3 per question.

Usage:
    uv run python two_call_dispatch.py [--force] [--dataset NAME] [--verbose]
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
from chain_retrieval import COT_PROMPT
from dotenv import load_dotenv
from openai import OpenAI

# Import the SPECIALIST prompts verbatim so they are byte-identical.
from prompt_optimization import META_V2F_PROMPT

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_FILE_EMB = CACHE_DIR / "two_call_embedding_cache.json"
CACHE_FILE_LLM = CACHE_DIR / "two_call_llm_cache.json"
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Caches (read from all prior caches, write to two_call_* files)
# ---------------------------------------------------------------------------
class TwoCallEmbeddingCache(EmbeddingCache):
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


class TwoCallLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        # Prefer two_call cache first (exact prompt match), then fallback to
        # all others so we can reuse v2f/CoT specialist calls from existing
        # runs when the prompts match byte-for-byte.
        for p in sorted(
            self.cache_dir.glob("*llm_cache.json"),
            key=lambda x: 0 if x.name.startswith("two_call") else 1,
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
# Classifier prompt (Call 1)
# ---------------------------------------------------------------------------
# The classifier sees ONLY the question and the label taxonomy. It never sees
# retrieved context and never emits cues. Its sole output is one word.
CLASSIFIER_PROMPT = """\
Classify this question as SIMPLE or COMPLEX.

SIMPLE: single fact lookup, one entity, direct answer expected
COMPLEX: multi-step reasoning, scattered evidence, list/completeness, vocabulary chains

QUESTION: {question}

Respond with exactly one word: SIMPLE or COMPLEX"""


def _parse_classification(text: str) -> str:
    """Parse classifier output. Default COMPLEX if ambiguous so we err toward
    more reasoning (matches self_v3's default)."""
    if not text:
        return "COMPLEX"
    upper = text.strip().upper()
    # Fast path: response is exactly one word
    token = upper.split()[0].strip(".,:!?") if upper.split() else ""
    if token == "SIMPLE":
        return "SIMPLE"
    if token == "COMPLEX":
        return "COMPLEX"
    # Fallback: scan for whichever appears first
    i_s = upper.find("SIMPLE")
    i_c = upper.find("COMPLEX")
    if i_s >= 0 and (i_c < 0 or i_s < i_c):
        return "SIMPLE"
    if i_c >= 0:
        return "COMPLEX"
    return "COMPLEX"


# ---------------------------------------------------------------------------
# Specialist helpers — BYTE-IDENTICAL to the standalone implementations.
# SIMPLE: match MetaV2Variant retrieval logic from prompt_optimization.py
# COMPLEX: match ChainOfThoughtCue retrieval logic from chain_retrieval.py
# ---------------------------------------------------------------------------
def _format_segments_v2f(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
    """Mirror prompt_optimization._format_segments (max_items=12, max_chars=250).
    Used to build the v2f context section byte-identically."""
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in sorted_segs
    )


def _build_v2f_context_section(all_segments: list[Segment]) -> str:
    """Mirror MetaV2Variant.retrieve: always prefixes with the fixed header."""
    context = _format_segments_v2f(all_segments)
    return "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + context


def _format_segments_cot(
    segments: list[Segment], max_items: int = 14, max_chars: int = 260
) -> str:
    """Mirror chain_retrieval._format_segments (max_items=14, max_chars=260)."""
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


# ---------------------------------------------------------------------------
# Two-call dispatch
# ---------------------------------------------------------------------------
@dataclass
class TwoCallResult:
    segments: list[Segment]
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


class TwoCallDispatch:
    """Two-call dispatch: 1 classifier call + N specialist calls.

    Classifier runs ONCE per question. Its output (one word: SIMPLE or COMPLEX)
    selects which specialist prompt runs. The specialist prompt is byte-identical
    to META_V2F_PROMPT (SIMPLE) or COT_PROMPT (COMPLEX) as used standalone.

    The specialist has no awareness of the classification step — it sees only
    its normal inputs exactly as in the standalone runs.
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        # SIMPLE path: mirrors v2f_tight (initial=10, 2 cues x 10) but with
        # 2 rounds (self_v2/v3 parity) for fair budget.
        initial_k: int = 10,
        num_cues_simple: int = 2,
        per_cue_k_simple: int = 10,
        # COMPLEX path: mirrors CoT (initial=10, 5 cues x 4, 2 rounds).
        num_cues_complex: int = 5,
        per_cue_k_complex: int = 4,
        rounds: int = 2,
    ) -> None:
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = TwoCallEmbeddingCache()
        self.llm_cache = TwoCallLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0
        self.initial_k = initial_k
        self.num_cues_simple = num_cues_simple
        self.per_cue_k_simple = per_cue_k_simple
        self.num_cues_complex = num_cues_complex
        self.per_cue_k_complex = per_cue_k_complex
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

    # -----------------------------------------------------------------------
    # Call 1: classifier
    # -----------------------------------------------------------------------
    def classify(self, question: str) -> tuple[str, str]:
        """Return (classification, raw_response)."""
        prompt = CLASSIFIER_PROMPT.format(question=question)
        response = self.llm_call(prompt, max_tokens=16)
        return _parse_classification(response), response

    # -----------------------------------------------------------------------
    # Call 2: specialist (v2f or CoT, byte-identical prompts)
    # -----------------------------------------------------------------------
    def specialist_simple(
        self, question: str, all_segs: list[Segment]
    ) -> tuple[list[str], str]:
        """Run META_V2F_PROMPT verbatim. Mirror MetaV2Variant.retrieve.

        Returns (cues, raw_response).
        """
        context_section = _build_v2f_context_section(all_segs)
        prompt = META_V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        response = self.llm_call(prompt)
        cues = _parse_cues(response, "CUE:")[: self.num_cues_simple]
        return cues, response

    def specialist_complex(
        self,
        question: str,
        all_segs: list[Segment],
        explored: list[str],
    ) -> tuple[list[str], str]:
        """Run COT_PROMPT verbatim. Mirror ChainOfThoughtCue.retrieve.

        Returns (cues, raw_response).
        """
        prompt = COT_PROMPT.format(
            question=question,
            all_segs=_format_segments_cot(all_segs, max_items=14),
            num_segs=len(all_segs),
            explored=(
                "\n".join(f"- {c}" for c in explored) if explored else "(none yet)"
            ),
            num_cues=self.num_cues_complex,
        )
        response = self.llm_call(prompt)
        cues = _parse_cues(response, "CUE:")[: self.num_cues_complex]
        return cues, response

    # -----------------------------------------------------------------------
    # End-to-end retrieve
    # -----------------------------------------------------------------------
    def retrieve(self, question: str, conversation_id: str) -> TwoCallResult:
        # --- Classifier (Call 1, runs once per question) ---
        classification, classifier_raw = self.classify(question)

        # --- Specialist (Call 2, runs every round) ---
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

        if classification == "SIMPLE":
            cap = self.num_cues_simple
            per_cue_k = self.per_cue_k_simple
        else:
            cap = self.num_cues_complex
            per_cue_k = self.per_cue_k_complex

        for round_i in range(self.rounds):
            if classification == "SIMPLE":
                cues, spec_raw = self.specialist_simple(question, all_segs)
            else:
                cues, spec_raw = self.specialist_complex(question, all_segs, explored)

            round_log.append(
                {
                    "round": round_i,
                    "specialist": classification,
                    "cues": cues,
                }
            )
            if not cues:
                break

            for cue in cues[:cap]:
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

        return TwoCallResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "two_call_dispatch",
                "classification": classification,
                "classifier_raw": classifier_raw[:100],
                "rounds": round_log,
                "total_segments": len(all_segs),
            },
        )


# ---------------------------------------------------------------------------
# Fair K-budget evaluation (mirrors self_v3.evaluate_one)
# ---------------------------------------------------------------------------
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(arch: TwoCallDispatch, question: dict, verbose: bool = False) -> dict:
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
        "classification": result.metadata["classification"],
        "baseline_recalls": baseline_recalls,
        "two_call_recalls": recalls,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "rounds_metadata": result.metadata["rounds"],
    }
    if verbose:
        print(
            f"    [{row['classification']}] pool={len(arch_segments)} "
            f"r@20: base={baseline_recalls['r@20']:.3f} "
            f"2c={recalls['r@20']:.3f}  "
            f"r@50: base={baseline_recalls['r@50']:.3f} "
            f"2c={recalls['r@50']:.3f}  "
            f"emb={arch.embed_calls} llm={arch.llm_calls}"
        )
    return row


# ---------------------------------------------------------------------------
# Datasets (identical to self_v2/self_v3)
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
# Load prior results for comparison (baseline, v2f, CoT, self_v2, self_v3)
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


def load_self_vN_recall_by_qkey(
    version: str,
    dataset_key: str,
) -> dict[tuple, dict[int, float]]:
    path = RESULTS_DIR / f"self_{version}_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    out: dict[tuple, dict[int, float]] = {}
    recall_key = f"self_{version}_recalls"
    for r in rows:
        key = (r["conversation_id"], r.get("question_index"))
        if recall_key in r:
            out[key] = {
                20: r[recall_key]["r@20"],
                50: r[recall_key]["r@50"],
            }
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def two_call_run(
    dataset_key: str, force: bool = False, verbose: bool = False
) -> list[dict]:
    result_file = RESULTS_DIR / f"two_call_{dataset_key}.json"
    if result_file.exists() and not force:
        with open(result_file) as f:
            return json.load(f)

    qs, store = load_dataset(dataset_key)
    arch = TwoCallDispatch(store)
    print(
        f"\n>>> TwoCallDispatch on {dataset_key}: {len(qs)} questions, "
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
        # Save incrementally per question.
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    print(f"  Saved -> {result_file}")
    return rows


# ---------------------------------------------------------------------------
# Per-dataset + per-category summary (mirrors self_v3.compare_per_category)
# ---------------------------------------------------------------------------
def compare_per_category(dataset_key: str, two_call_rows: list[dict]) -> list[dict]:
    budget_by_arch: dict[str, dict[int, dict]] = {}
    for arch in ("baseline", "v2f_tight"):
        budget_by_arch[arch] = {
            K: load_budget_recall_by_qkey(f"{arch}_{K}", dataset_key) for K in BUDGETS
        }
    cot_by_q = load_cot_recall_by_qkey(dataset_key)
    sv2_by_q = load_self_vN_recall_by_qkey("v2", dataset_key)
    sv3_by_q = load_self_vN_recall_by_qkey("v3", dataset_key)

    rows_by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in two_call_rows:
        rows_by_cat[r["category"]].append(r)

    out: list[dict] = []
    for K in BUDGETS:
        for cat, rows in sorted(rows_by_cat.items()):
            n = len(rows)
            if n == 0:
                continue
            tc_recalls = [r["two_call_recalls"][f"r@{K}"] for r in rows]
            tc_mean = sum(tc_recalls) / n

            b_vals, v2f_vals, cot_vals, sv2_vals, sv3_vals = [], [], [], [], []
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
                if key in sv3_by_q:
                    sv3_vals.append(sv3_by_q[key][K])

            def _mean(xs: list[float]) -> float | None:
                if not xs:
                    return None
                return sum(xs) / len(xs)

            simple_n = sum(1 for r in rows if r["classification"] == "SIMPLE")
            complex_n = n - simple_n

            b_mean = _mean(b_vals)
            v2f_mean = _mean(v2f_vals)
            cot_mean = _mean(cot_vals)
            sv2_mean = _mean(sv2_vals)
            sv3_mean = _mean(sv3_vals)

            row = {
                "dataset": dataset_key,
                "category": cat,
                "K": K,
                "n": n,
                "baseline": b_mean,
                "v2f": v2f_mean,
                "cot": cot_mean,
                "self_v2": sv2_mean,
                "self_v3": sv3_mean,
                "two_call": tc_mean,
                "vs_v2f": (tc_mean - v2f_mean if v2f_mean is not None else None),
                "vs_cot": (tc_mean - cot_mean if cot_mean is not None else None),
                "vs_sv2": (tc_mean - sv2_mean if sv2_mean is not None else None),
                "vs_sv3": (tc_mean - sv3_mean if sv3_mean is not None else None),
                "vs_baseline": (tc_mean - b_mean if b_mean is not None else None),
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
    print(f"\n{'-' * 140}")
    print(f"OVERALL per DATASET at K={K}")
    print(f"{'-' * 140}")
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows_k:
        by_ds[r["dataset"]].append(r)

    hdr = (
        f"{'Dataset':<14s} {'n':>3s} "
        f"{'Base':>6s} {'v2f':>6s} {'CoT':>6s} {'SV2':>6s} {'SV3':>6s} "
        f"{'2Call':>6s}  "
        f"{'vs_v2f':>7s} {'vs_CoT':>7s} {'vs_SV2':>7s} {'vs_SV3':>7s}  "
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
        tc = _weighted("two_call")
        vv = (tc - v2f) if (tc is not None and v2f is not None) else None
        vc = (tc - cot) if (tc is not None and cot is not None) else None
        vs2 = (tc - sv2) if (tc is not None and sv2 is not None) else None
        vs3 = (tc - sv3) if (tc is not None and sv3 is not None) else None
        sc_str = f"{simple_tot}/{complex_tot}"
        print(
            f"{ds:<14s} {total_n:>3d} "
            f"{fmt_cell(b)} {fmt_cell(v2f)} {fmt_cell(cot)} "
            f"{fmt_cell(sv2)} {fmt_cell(sv3)} {fmt_cell(tc)}  "
            f"{fmt_cell(vv, True)} {fmt_cell(vc, True)} "
            f"{fmt_cell(vs2, True)} {fmt_cell(vs3, True)}  "
            f"{sc_str:>7s}"
        )


def print_per_category(all_rows: list[dict], K: int) -> None:
    rows_k = [r for r in all_rows if r["K"] == K]
    if not rows_k:
        return
    print(f"\n{'=' * 140}")
    print(f"PER-CATEGORY BREAKDOWN at K={K}")
    print(f"{'=' * 140}")
    hdr = (
        f"{'Dataset':<14s} {'Category':<24s} {'n':>3s} {'S/C':>7s} "
        f"{'Base':>6s} {'v2f':>6s} {'CoT':>6s} {'SV2':>6s} {'SV3':>6s} "
        f"{'2Call':>6s}  "
        f"{'vs_v2f':>7s} {'vs_CoT':>7s} {'vs_SV3':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows_k, key=lambda x: (x["dataset"], x["category"])):
        sc_str = f"{r['simple_n']}/{r['complex_n']}"
        print(
            f"{r['dataset']:<14s} {r['category']:<24s} {r['n']:>3d} "
            f"{sc_str:>7s} "
            f"{fmt_cell(r['baseline'])} {fmt_cell(r['v2f'])} "
            f"{fmt_cell(r['cot'])} {fmt_cell(r['self_v2'])} "
            f"{fmt_cell(r['self_v3'])} {fmt_cell(r['two_call'])}  "
            f"{fmt_cell(r['vs_v2f'], True)} "
            f"{fmt_cell(r['vs_cot'], True)} "
            f"{fmt_cell(r['vs_sv3'], True)}"
        )


def print_specialist_match_check(
    all_two_call_rows: dict[str, list[dict]], K: int = 20
) -> None:
    """Decompose recall by classification outcome and compare to the specialist.

    For SIMPLE-classified questions: does two_call >= v2f? (It should if the
    fix works — same prompt in a clean context.)
    For COMPLEX-classified questions: does two_call >= CoT? (Same logic.)
    """
    print(f"\n{'=' * 120}")
    print(f"SPECIALIST MATCH CHECK at K={K}")
    print(
        "  Hypothesis: separating classifier from specialist removes the "
        "self_v3 leakage."
    )
    print(
        "  Test: on SIMPLE-classified Qs, 2-call should match v2f; on "
        "COMPLEX, should match CoT."
    )
    print(f"{'=' * 120}")
    hdr = (
        f"{'Dataset':<14s} {'Branch':<9s} {'n':>3s} "
        f"{'2Call':>6s} {'Specialist':>11s} {'delta':>7s}  {'vs_SV3':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for ds, rows in sorted(all_two_call_rows.items()):
        cot_by_q = load_cot_recall_by_qkey(ds)
        v2f_by_q = {}
        path = RESULTS_DIR / f"budget_v2f_tight_{K}_{ds}.json"
        if path.exists():
            with open(path) as f:
                payload = json.load(f)
            for r in payload.get("results", []):
                key = (r["conversation_id"], r.get("question_index"))
                v2f_by_q[key] = r["recall"]
        sv3_by_q = load_self_vN_recall_by_qkey("v3", ds)

        for branch in ("SIMPLE", "COMPLEX"):
            branch_rows = [r for r in rows if r["classification"] == branch]
            if not branch_rows:
                continue
            tc_vals = [r["two_call_recalls"][f"r@{K}"] for r in branch_rows]
            tc_mean = sum(tc_vals) / len(tc_vals)

            if branch == "SIMPLE":
                spec_vals = [
                    v2f_by_q.get((r["conversation_id"], r["question_index"]))
                    for r in branch_rows
                ]
                spec_label = "v2f"
            else:
                spec_vals = [
                    cot_by_q.get(
                        (r["conversation_id"], r["question_index"]),
                        {},
                    ).get(K)
                    for r in branch_rows
                ]
                spec_label = "CoT"
            spec_vals = [v for v in spec_vals if v is not None]
            spec_mean = sum(spec_vals) / len(spec_vals) if spec_vals else None
            sv3_vals = [
                sv3_by_q.get(
                    (r["conversation_id"], r["question_index"]),
                    {},
                ).get(K)
                for r in branch_rows
            ]
            sv3_vals = [v for v in sv3_vals if v is not None]
            sv3_mean = sum(sv3_vals) / len(sv3_vals) if sv3_vals else None

            delta = (tc_mean - spec_mean) if spec_mean is not None else None
            delta_sv3 = tc_mean - sv3_mean if sv3_mean is not None else None
            print(
                f"{ds:<14s} {branch:<9s} {len(branch_rows):>3d} "
                f"{tc_mean:>6.3f} {spec_label + '=' + (f'{spec_mean:.3f}' if spec_mean is not None else '—'):>11s} "
                f"{fmt_cell(delta, True)}  "
                f"{fmt_cell(delta_sv3, True)}"
            )


def print_dominance_vs_v2f_cot(all_rows: list[dict]) -> None:
    """Does two_call dominate v2f AND CoT on every dataset x K?"""
    print(f"\n{'=' * 120}")
    print("UNIVERSAL DOMINANCE CHECK — two_call vs {v2f, CoT}")
    print("  (Key question: does splitting the call fix self_v3's failure?)")
    print(f"{'=' * 120}")
    by_ds_k: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_ds_k[(r["dataset"], r["K"])].append(r)

    dom_v2f = True
    dom_cot = True
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

            tc = _weighted("two_call")
            v2f = _weighted("v2f")
            cot = _weighted("cot")
            if tc is None:
                continue

            parts = [f"  K={K:<3d}  {ds:<14s}  2c={tc:.3f}"]
            if v2f is not None:
                d = tc - v2f
                tick = "OK" if d >= -0.001 else "FAIL"
                parts.append(f"v2f={v2f:.3f} (Δ={d:+.3f} {tick})")
                if d < -0.001:
                    dom_v2f = False
            if cot is not None:
                d = tc - cot
                tick = "OK" if d >= -0.001 else "FAIL"
                parts.append(f"CoT={cot:.3f} (Δ={d:+.3f} {tick})")
                if d < -0.001:
                    dom_cot = False
            print("  ".join(parts))
    print(f"\n  Universal dominance over v2f : {'YES' if dom_v2f else 'NO'}")
    print(f"  Universal dominance over CoT : {'YES' if dom_cot else 'NO'}")


def print_classification_spotcheck(
    all_rows: dict[str, list[dict]],
) -> None:
    print(f"\n{'=' * 108}")
    print("CLASSIFICATION SPOT-CHECK (per-dataset, per-category)")
    print(f"{'=' * 108}")
    for ds, rows in all_rows.items():
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
        help="Restrict to a single dataset",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_keys = [args.dataset] if args.dataset else list(DATASETS.keys())

    all_rows: list[dict] = []
    all_two_call_rows: dict[str, list[dict]] = {}
    for ds in dataset_keys:
        rows = two_call_run(ds, force=args.force, verbose=args.verbose)
        all_two_call_rows[ds] = rows
        all_rows.extend(compare_per_category(ds, rows))

    out_path = RESULTS_DIR / "two_call_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved summary -> {out_path}")

    for K in BUDGETS:
        print_overall_by_dataset(all_rows, K)

    for K in BUDGETS:
        print_per_category(all_rows, K)

    print_classification_spotcheck(all_two_call_rows)

    for K in BUDGETS:
        print_specialist_match_check(all_two_call_rows, K=K)

    print_dominance_vs_v2f_cot(all_rows)


if __name__ == "__main__":
    main()
