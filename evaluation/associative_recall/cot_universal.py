"""Test whether chain_retrieval.py's chain_of_thought (CoT) is a universal
improvement or category-specialist.

Runs the CoT architecture on ALL 4 datasets (LoCoMo 30q, Synthetic 19q,
Puzzle 16q, Advanced 23q) with FAIR K-budget evaluation at K=20 and K=50.

Evaluation protocol (mirrors budget_aware_eval.py v15_tight_20 style):
  - Run CoT; collect its segment pool.
  - Do cosine top-K baseline on the question.
  - Final pool = CoT segments + baseline-backfill (dedup, in order) truncated
    to EXACTLY K. So every comparison is at the same K-budget with no
    reranking.

Comparison baselines (loaded from existing `results/budget_*.json`):
  - baseline_{K}    : cosine top-K
  - v15_tight_{K}   : hop0=10 + 2 cues x 5 (v15 prompt)          [K=20 form]
                      hop0=20 + 2 cues x 15 (v15 prompt)         [K=50 form]
  - v2f_tight_{K}   : same structure, v2f prompt

Reports a per-category table showing CoT delta vs v2f to determine which
categories benefit from explicit vocabulary-transition reasoning.

Usage:
    uv run python cot_universal.py [--force]
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
CACHE_FILE_EMB = CACHE_DIR / "cot_universal_embedding_cache.json"
CACHE_FILE_LLM = CACHE_DIR / "cot_universal_llm_cache.json"
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Caches (read from all prior caches, write to cot_universal_* files)
# ---------------------------------------------------------------------------
class CoTEmbeddingCache(EmbeddingCache):
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


class CoTLLMCache(LLMCache):
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
# CoT architecture (verbatim from chain_retrieval.py)
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
            val = line[len(key):].strip()
            if val:
                out.append(val)
    return out


COT_PROMPT = """\
You are performing semantic retrieval over a conversation history. Cues will \
be embedded and matched via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({num_segs} segments, chronological):
{all_segs}

ALREADY SEARCHED FOR (do NOT repeat):
{explored}

Think step by step:
1. What specific terminology appears in the retrieved segments (names, tools, \
symptoms, decisions, tickets, numbers)?
2. What RELATED terminology might be used elsewhere? (aliases, codenames, \
abbreviations, informal references like "the bird", "that thing", ...)
3. If this is a CHAIN (A -> B -> C where each link has different vocabulary), \
what is the NEXT link to search for?
4. If this topic has ALTERNATIVE NAMES, what are they? Include every alias \
you can justify from the retrieved text or reasonable guesses.

Then generate {num_cues} search cues that EXTEND the retrieval in the most \
promising directions. A cue may be:
  - a short alias/name phrase (1-5 words) that might appear inline
  - a 1-2 sentence plausible conversation snippet targeting the next link

Prefer DIVERSE cues (cover multiple aliases and/or multiple chain links). \
Do not rephrase the question.

Format:
REASON: <1-2 sentences: what's current link or what aliases you identified>
CUE: <text>
CUE: <text>
(up to {num_cues} cues)
Nothing else."""


@dataclass
class CoTResult:
    segments: list[Segment]
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


class ChainOfThoughtArch:
    """COT (from chain_retrieval.py): initial retrieve + 2 rounds of
    step-by-step cue generation.

    Defaults match chain_retrieval.py ChainOfThoughtCue:
      initial_k=10, num_cues=5, per_cue_k=4, rounds=2
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        initial_k: int = 10,
        num_cues: int = 5,
        per_cue_k: int = 4,
        rounds: int = 2,
    ) -> None:
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = CoTEmbeddingCache()
        self.llm_cache = CoTLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0
        self.initial_k = initial_k
        self.num_cues = num_cues
        self.per_cue_k = per_cue_k
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

    def retrieve(self, question: str, conversation_id: str) -> CoTResult:
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

        for round_i in range(self.rounds):
            prompt = COT_PROMPT.format(
                question=question,
                all_segs=_format_segments(all_segs, max_items=14),
                num_segs=len(all_segs),
                explored=("\n".join(f"- {c}" for c in explored)
                          if explored else "(none yet)"),
                num_cues=self.num_cues,
            )
            response = self.llm_call(prompt)
            reason = ""
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("REASON:"):
                    reason = line[7:].strip()
                    break
            cues = _parse_cues(response, "CUE:")[: self.num_cues]
            round_log.append({"round": round_i, "reason": reason, "cues": cues})
            if not cues:
                break
            for cue in cues:
                if cue in explored:
                    continue
                explored.append(cue)
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb, top_k=self.per_cue_k,
                    conversation_id=conversation_id, exclude_indices=exclude,
                )
                for s in result.segments:
                    if s.index not in exclude:
                        all_segs.append(s)
                        exclude.add(s.index)

        return CoTResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "chain_of_thought",
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


def evaluate_one(arch: ChainOfThoughtArch, question: dict,
                 verbose: bool = False) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedup preserving order (CoT already dedups by exclude set, but double-check)
    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for s in result.segments:
        if s.index not in seen:
            arch_segments.append(s)
            seen.add(s.index)

    # Baseline cosine top-max(BUDGETS) on question embedding
    q_emb = arch.embed_text(q_text)
    max_b = max(BUDGETS)
    baseline = arch.store.search(
        q_emb, top_k=max_b, conversation_id=conv_id
    )

    # FAIR backfill: arch pool + baseline segments not already in pool, in order
    arch_idx = {s.index for s in arch_segments}
    backfilled = list(arch_segments) + [
        s for s in baseline.segments if s.index not in arch_idx
    ]

    # Per-budget recall at exactly K
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
        "cot_pool_size": len(arch_segments),
        "baseline_recalls": baseline_recalls,
        "cot_recalls": recalls,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
    }
    if verbose:
        print(
            f"    pool={len(arch_segments)} "
            f"r@20: base={baseline_recalls['r@20']:.3f} "
            f"cot={recalls['r@20']:.3f}  "
            f"r@50: base={baseline_recalls['r@50']:.3f} "
            f"cot={recalls['r@50']:.3f}  "
            f"emb={arch.embed_calls} llm={arch.llm_calls}"
        )
    return row


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
# Comparison: load existing budget results for baseline/v15/v2f
# ---------------------------------------------------------------------------
def load_budget_recall_by_qkey(
    arch_name: str, dataset_key: str
) -> dict[tuple, float]:
    """Load recall per question from a budget_*.json result file.

    Keyed by (conversation_id, question_index) -> recall.
    """
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


def cot_run(
    dataset_key: str, force: bool = False, verbose: bool = False
) -> list[dict]:
    """Run CoT on a dataset. Caches raw per-question results to
    results/cot_chain_of_thought_<dataset>.json for reuse.
    """
    result_file = RESULTS_DIR / f"cot_chain_of_thought_{dataset_key}.json"
    if result_file.exists() and not force:
        with open(result_file) as f:
            return json.load(f)

    qs, store = load_dataset(dataset_key)
    arch = ChainOfThoughtArch(store)
    print(
        f"\n>>> CoT on {dataset_key}: {len(qs)} questions, "
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
# Per-category summary table
# ---------------------------------------------------------------------------
def compare_per_category(
    dataset_key: str, cot_rows: list[dict]
) -> list[dict]:
    """Build per-category summary rows for K=20 and K=50.

    Returns list of dicts:
      {
        dataset, category, K, n,
        baseline, v15, v2f, cot,
        cot_vs_v2f, cot_vs_v15, cot_vs_baseline
      }
    """
    # Preload per-question recalls from budget files
    budget_by_arch: dict[str, dict[str, dict]] = {}
    for arch in ("baseline", "v15_tight", "v2f_tight"):
        budget_by_arch[arch] = {
            K: load_budget_recall_by_qkey(f"{arch}_{K}", dataset_key)
            for K in BUDGETS
        }

    # Group CoT rows by category
    rows_by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in cot_rows:
        rows_by_cat[r["category"]].append(r)

    out: list[dict] = []
    for K in BUDGETS:
        for cat, rows in sorted(rows_by_cat.items()):
            n = len(rows)
            if n == 0:
                continue
            # CoT mean recall at K
            cot_recalls = [r["cot_recalls"][f"r@{K}"] for r in rows]
            cot_mean = sum(cot_recalls) / n

            # baselines keyed by (conv_id, q_index)
            b_vals = []
            v15_vals = []
            v2f_vals = []
            for r in rows:
                key = (r["conversation_id"], r["question_index"])
                if key in budget_by_arch["baseline"][K]:
                    b_vals.append(budget_by_arch["baseline"][K][key])
                if key in budget_by_arch["v15_tight"][K]:
                    v15_vals.append(budget_by_arch["v15_tight"][K][key])
                if key in budget_by_arch["v2f_tight"][K]:
                    v2f_vals.append(budget_by_arch["v2f_tight"][K][key])

            def _mean(xs: list[float]) -> float | None:
                if not xs:
                    return None
                return sum(xs) / len(xs)

            b_mean = _mean(b_vals)
            v15_mean = _mean(v15_vals)
            v2f_mean = _mean(v2f_vals)

            row = {
                "dataset": dataset_key,
                "category": cat,
                "K": K,
                "n": n,
                "baseline": b_mean,
                "v15": v15_mean,
                "v2f": v2f_mean,
                "cot": cot_mean,
                "cot_vs_v2f": (cot_mean - v2f_mean) if v2f_mean is not None else None,
                "cot_vs_v15": (cot_mean - v15_mean) if v15_mean is not None else None,
                "cot_vs_baseline": (cot_mean - b_mean) if b_mean is not None else None,
            }
            out.append(row)
    return out


def fmt_cell(val: float | None, plus_sign: bool = False) -> str:
    if val is None:
        return "    —"
    s = f"{val:+.3f}" if plus_sign else f"{val:.3f}"
    return f"{s:>6s}"


def print_table(all_rows: list[dict], K: int) -> None:
    rows = [r for r in all_rows if r["K"] == K]
    if not rows:
        return
    print(f"\n{'='*108}")
    print(f"PER-CATEGORY RESULTS at K={K} (recall, fair backfill)")
    print(f"{'='*108}")
    hdr = (
        f"{'Dataset':<14s} {'Category':<26s} {'n':>3s} "
        f"{'Base':>7s} {'v15':>7s} {'v2f':>7s} {'CoT':>7s}  "
        f"{'vs v2f':>7s} {'vs v15':>7s} {'vs base':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    last_ds = None
    for r in rows:
        if last_ds is not None and r["dataset"] != last_ds:
            print("")
        last_ds = r["dataset"]
        print(
            f"{r['dataset']:<14s} {r['category']:<26s} {r['n']:>3d} "
            f"{fmt_cell(r['baseline'])} {fmt_cell(r['v15'])} "
            f"{fmt_cell(r['v2f'])} {fmt_cell(r['cot'])}  "
            f"{fmt_cell(r['cot_vs_v2f'], True)} "
            f"{fmt_cell(r['cot_vs_v15'], True)} "
            f"{fmt_cell(r['cot_vs_baseline'], True)}"
        )


def print_overall_by_dataset(all_rows: list[dict], K: int) -> None:
    """Overall mean recall per dataset at K, aggregated across all
    questions (weighted by per-category n)."""
    print(f"\n{'-'*108}")
    print(f"OVERALL per DATASET at K={K}")
    print(f"{'-'*108}")
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        if r["K"] == K:
            by_ds[r["dataset"]].append(r)

    hdr = (
        f"{'Dataset':<14s} {'n':>3s} "
        f"{'Base':>7s} {'v15':>7s} {'v2f':>7s} {'CoT':>7s}  "
        f"{'vs v2f':>7s} {'vs v15':>7s} {'vs base':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for ds, rows in by_ds.items():
        total_n = sum(r["n"] for r in rows)
        def _weighted(key: str) -> float | None:
            vals = [(r[key], r["n"]) for r in rows if r[key] is not None]
            if not vals:
                return None
            tot = sum(n for _, n in vals)
            return sum(v * n for v, n in vals) / tot
        b = _weighted("baseline")
        v15 = _weighted("v15")
        v2f = _weighted("v2f")
        cot = _weighted("cot")
        cv = (cot - v2f) if (cot is not None and v2f is not None) else None
        cv15 = (cot - v15) if (cot is not None and v15 is not None) else None
        cb = (cot - b) if (cot is not None and b is not None) else None
        print(
            f"{ds:<14s} {total_n:>3d} "
            f"{fmt_cell(b)} {fmt_cell(v15)} {fmt_cell(v2f)} {fmt_cell(cot)}  "
            f"{fmt_cell(cv, True)} {fmt_cell(cv15, True)} {fmt_cell(cb, True)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Rerun CoT even if result file exists")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASETS.keys()),
                        help="Restrict to a single dataset (default: all)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_keys = (
        [args.dataset] if args.dataset else list(DATASETS.keys())
    )

    all_rows: list[dict] = []
    for ds in dataset_keys:
        cot_rows = cot_run(ds, force=args.force, verbose=args.verbose)
        all_rows.extend(compare_per_category(ds, cot_rows))

    # Save rolled-up summaries
    out_path = RESULTS_DIR / "cot_universal_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved summary -> {out_path}")

    for K in BUDGETS:
        print_table(all_rows, K)
        print_overall_by_dataset(all_rows, K)

    # Headline table: concise CoT delta vs v2f summary
    print(f"\n{'='*108}")
    print(
        "HEADLINE TABLE (CoT delta vs v2f per category; +=CoT better, "
        "meaningful = >=+2pp or <=-2pp)"
    )
    print(f"{'='*108}")
    print(
        f"{'Dataset':<14s} {'Category':<26s} "
        f"{'K':>3s} {'v2f':>7s} {'CoT':>7s} {'delta':>7s} "
        f"{'n':>3s}  notes"
    )
    print("-" * 88)
    for r in sorted(
        all_rows,
        key=lambda x: (x["dataset"], x["K"], x["category"]),
    ):
        delta = r["cot_vs_v2f"]
        if delta is None:
            note = "no v2f baseline"
        elif delta >= 0.02:
            note = "CoT HELPS"
        elif delta <= -0.02:
            note = "CoT HURTS"
        else:
            note = "~neutral"
        print(
            f"{r['dataset']:<14s} {r['category']:<26s} "
            f"{r['K']:>3d} "
            f"{fmt_cell(r['v2f'])} {fmt_cell(r['cot'])} "
            f"{fmt_cell(delta, True)} "
            f"{r['n']:>3d}  {note}"
        )


if __name__ == "__main__":
    main()
