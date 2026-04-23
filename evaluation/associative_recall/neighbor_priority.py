"""Neighbor-priority v2f variants.

Tests whether explicit neighbor prioritization closes the gap identified in
error_analysis_summary.md, where ~50% of missed source turns are ±1 from a
retrieved turn.

Variants (all based on v2f cue generation):

A. v2f_nr0         — baseline: no neighbor expansion (same as meta_v2f).
B. v2f_nr1_priority — neighbor_radius=1. Neighbors are inserted IMMEDIATELY
                      AFTER their parent segment in retrieval order.
C. v2f_nr2_priority — neighbor_radius=2. Same priority ordering.
D. v2f_post_hoc_neighbors — Run v2f. After retrieval, for the top-15 arch
                            picks add ±1 neighbors into the final top-20 if
                            they are not already in it.

All run with fair backfill at K=20 against the existing meta_v2f LLM cache
(cues stay unchanged, so no new LLM calls).

Usage:
    uv run python neighbor_priority.py
"""

import json
import sys
import time
from collections import defaultdict
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
from best_shot import (
    BestshotEmbeddingCache,
    BestshotLLMCache,
    V2F_PROMPT,
    _format_segments,
    _parse_cues,
    BestshotResult,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
K_BUDGET = 20

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


# ---------------------------------------------------------------------------
# Cache (reuses the bestshot/meta_v2f caches + a neighbor_priority file)
# ---------------------------------------------------------------------------
class NeighborPriorityEmbeddingCache(BestshotEmbeddingCache):
    """Writes to a neighbor_priority-specific file but reads all existing."""

    def __init__(self):
        super().__init__()
        # Override write target; the parent constructor already loaded every
        # existing embedding cache.
        self.cache_file = (
            self.cache_dir / "neighbor_priority_embedding_cache.json"
        )
        extra = self.cache_dir / "neighbor_priority_embedding_cache.json"
        if extra.exists():
            with open(extra) as f:
                self._cache.update(json.load(f))


class NeighborPriorityLLMCache(BestshotLLMCache):
    """Writes to a neighbor_priority-specific LLM cache file, reads all."""

    def __init__(self):
        super().__init__()
        self.cache_file = self.cache_dir / "neighbor_priority_llm_cache.json"
        extra = self.cache_dir / "neighbor_priority_llm_cache.json"
        if extra.exists():
            with open(extra) as f:
                data = json.load(f)
            for k, v in data.items():
                if v:
                    self._cache[k] = v


# ---------------------------------------------------------------------------
# Retrieval engine with neighbor prioritization
# ---------------------------------------------------------------------------
class NeighborPriorityV2f:
    """V2f with configurable neighbor prioritization.

    Modes:
      - "nr0": neighbors disabled (baseline).
      - "nr_priority": every time a segment is appended from cosine top-K,
        its ±radius neighbors are inserted IMMEDIATELY after it (in order,
        skipping dupes) before the next cosine result is considered.
      - "post_hoc": run standard v2f (no neighbor expansion inline). After
        retrieval, take the top `post_hoc_source_k` arch picks. For each,
        attempt to insert its ±1 neighbors into the top-`k_budget` window
        whenever a slot is free. Neighbors bump cosine backfill fillers.

    The cue-generation is identical to meta_v2f.
    """

    def __init__(
        self,
        store: SegmentStore,
        mode: str = "nr0",
        radius: int = 0,
        post_hoc_source_k: int = 15,
        k_budget: int = K_BUDGET,
        client: OpenAI | None = None,
    ):
        self.store = store
        self.mode = mode
        self.radius = radius
        self.post_hoc_source_k = post_hoc_source_k
        self.k_budget = k_budget
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = NeighborPriorityEmbeddingCache()
        self.llm_cache = NeighborPriorityLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

    # ---- utilities ------------------------------------------------------
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

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    # ---- core retrieval -------------------------------------------------
    def _append_with_neighbors(
        self,
        seg: Segment,
        collected: list[Segment],
        seen: set[int],
    ) -> None:
        """Append `seg` and (if radius > 0) its neighbors immediately after."""
        if seg.index in seen:
            return
        collected.append(seg)
        seen.add(seg.index)

        if self.mode == "nr_priority" and self.radius > 0:
            neighbors = self.store.get_neighbors(
                seg, radius=self.radius, exclude_indices=seen
            )
            # Stable order: by offset (-radius..+radius, skipping 0) which
            # get_neighbors already honours.
            for nb in neighbors:
                if nb.index not in seen:
                    collected.append(nb)
                    seen.add(nb.index)

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )

        collected: list[Segment] = []
        seen: set[int] = set()
        for seg in hop0.segments:
            self._append_with_neighbors(seg, collected, seen)

        # Build context section using only the original cosine hits (the
        # cues were originally generated from just top-10 cosine results,
        # not from the neighbor-inflated list). This keeps the prompt
        # identical to meta_v2f so the LLM cache hits.
        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(list(hop0.segments))
        )
        prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            # Use the existing `exclude_indices` parameter to skip parents
            # already retrieved. Neighbors added from prior parents are in
            # `seen` and thus get excluded too — they never compete for
            # their own slot.
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=seen,
            )
            for seg in result.segments:
                self._append_with_neighbors(seg, collected, seen)

        return BestshotResult(
            segments=collected,
            metadata={
                "name": f"v2f_{self.mode}_r{self.radius}",
                "output": output,
                "cues": cues[:2],
            },
        )


# ---------------------------------------------------------------------------
# Post-hoc neighbor injection (variant D)
# ---------------------------------------------------------------------------
def apply_post_hoc_neighbors(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    store: SegmentStore,
    source_k: int,
    budget: int,
) -> list[Segment]:
    """Build a top-`budget` list with explicit post-hoc neighbor injection.

    Procedure, per the finding: reserve budget slots for neighbors of the
    top `source_k` arch picks rather than letting them fall to cosine.

      1. Start with the top `source_k` arch picks (kept, in order).
      2. For each of those, gather ±1 neighbors (in arch order, parent by
         parent, then offset -1, +1, skipping dupes). These go next.
      3. Fill any remaining budget with arch picks beyond `source_k`, then
         cosine backfill.
      4. Truncate to `budget`.
    """
    # Dedupe arch order
    seen: set[int] = set()
    arch_unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            arch_unique.append(s)
            seen.add(s.index)

    final: list[Segment] = []
    final_seen: set[int] = set()

    # 1. Source picks (top `source_k` arch segments).
    for seg in arch_unique[:source_k]:
        if seg.index not in final_seen:
            final.append(seg)
            final_seen.add(seg.index)

    # 2. Neighbors of those source picks, inserted in parent order.
    for parent in arch_unique[:source_k]:
        neighbors = store.get_neighbors(
            parent, radius=1, exclude_indices=final_seen
        )
        for nb in neighbors:
            if nb.index not in final_seen:
                final.append(nb)
                final_seen.add(nb.index)
                if len(final) >= budget:
                    break
        if len(final) >= budget:
            break

    # 3. Remaining arch picks past `source_k`, then cosine backfill.
    if len(final) < budget:
        for seg in arch_unique[source_k:]:
            if seg.index not in final_seen:
                final.append(seg)
                final_seen.add(seg.index)
                if len(final) >= budget:
                    break
    if len(final) < budget:
        for seg in cosine_segments:
            if seg.index not in final_seen:
                final.append(seg)
                final_seen.add(seg.index)
                if len(final) >= budget:
                    break

    return final[:budget]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


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


def fair_backfill(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    budget: int,
) -> list[Segment]:
    seen: set[int] = set()
    arch_unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            arch_unique.append(s)
            seen.add(s.index)
    at = arch_unique[:budget]
    arch_idx = {s.index for s in at}
    if len(at) < budget:
        backfill = [s for s in cosine_segments if s.index not in arch_idx]
        needed = budget - len(at)
        at = at + backfill[:needed]
    return at[:budget]


def evaluate_one(
    arch: NeighborPriorityV2f,
    question: dict,
    variant: str,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedupe arch
    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            arch_segments.append(seg)
            seen.add(seg.index)

    # Compute cosine top-K once
    query_emb = arch.embed_text(q_text)
    cosine_result = arch.store.search(
        query_emb, top_k=K_BUDGET, conversation_id=conv_id
    )
    cosine_segments = list(cosine_result.segments)

    if variant == "v2f_post_hoc_neighbors":
        final_segs = apply_post_hoc_neighbors(
            arch_segments, cosine_segments, arch.store,
            source_k=arch.post_hoc_source_k, budget=K_BUDGET,
        )
    else:
        final_segs = fair_backfill(arch_segments, cosine_segments, K_BUDGET)

    baseline_segs = cosine_segments[:K_BUDGET]

    retrieved_ids = {s.turn_id for s in final_segs}
    baseline_ids = {s.turn_id for s in baseline_segs}

    return {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "num_source_turns": len(source_ids),
        "num_arch_segments": len(arch_segments),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 3),
        "baseline_r@20": round(compute_recall(baseline_ids, source_ids), 4),
        "arch_r@20": round(compute_recall(retrieved_ids, source_ids), 4),
    }


def summarize(rows: list[dict], variant: str, dataset: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"variant": variant, "dataset": dataset, "n": 0}
    b_vals = [r["baseline_r@20"] for r in rows]
    a_vals = [r["arch_r@20"] for r in rows]
    wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
    losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
    ties = n - wins - losses
    return {
        "variant": variant,
        "dataset": dataset,
        "n": n,
        "baseline_r@20": round(sum(b_vals) / n, 4),
        "arch_r@20": round(sum(a_vals) / n, 4),
        "delta_r@20": round(sum(a_vals) / n - sum(b_vals) / n, 4),
        "W/T/L_r@20": f"{wins}/{ties}/{losses}",
        "avg_llm_calls": round(sum(r["llm_calls"] for r in rows) / n, 2),
        "avg_embed_calls": round(sum(r["embed_calls"] for r in rows) / n, 2),
    }


def summarize_by_category(rows: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    out: dict[str, dict] = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        b_vals = [r["baseline_r@20"] for r in rs]
        a_vals = [r["arch_r@20"] for r in rs]
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses
        out[cat] = {
            "n": n,
            "baseline_r@20": round(sum(b_vals) / n, 4),
            "arch_r@20": round(sum(a_vals) / n, 4),
            "delta_r@20": round(sum(a_vals) / n - sum(b_vals) / n, 4),
            "W/T/L_r@20": f"{wins}/{ties}/{losses}",
        }
    return out


VARIANTS = {
    "v2f_nr0": {"mode": "nr0", "radius": 0},
    "v2f_nr1_priority": {"mode": "nr_priority", "radius": 1},
    "v2f_nr2_priority": {"mode": "nr_priority", "radius": 2},
    "v2f_post_hoc_neighbors": {"mode": "nr0", "radius": 0},  # special-case
}


def run_variant(
    variant: str,
    dataset: str,
    store: SegmentStore,
    questions: list[dict],
) -> dict:
    cfg = VARIANTS[variant]
    arch = NeighborPriorityV2f(
        store, mode=cfg["mode"], radius=cfg["radius"]
    )

    print(f"\n[{variant} | {dataset} | n={len(questions)}]", flush=True)
    rows: list[dict] = []
    for i, q in enumerate(questions):
        try:
            row = evaluate_one(arch, q, variant)
            rows.append(row)
        except Exception as e:
            print(f"  ERROR on Q{i}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        if (i + 1) % 10 == 0:
            arch.save_caches()
            sys.stdout.flush()
    arch.save_caches()

    summary = summarize(rows, variant, dataset)
    by_cat = summarize_by_category(rows)
    print(
        f"  base@20={summary['baseline_r@20']:.3f} "
        f"arch@20={summary['arch_r@20']:.3f} "
        f"d@20={summary['delta_r@20']:+.3f} "
        f"W/T/L={summary['W/T/L_r@20']} "
        f"llm={summary['avg_llm_calls']:.2f}"
    )
    return {
        "variant": variant,
        "dataset": dataset,
        "summary": summary,
        "category_breakdown": by_cat,
        "results": rows,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries: dict = {}

    for ds_name in DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )
        for variant in VARIANTS:
            payload = run_variant(variant, ds_name, store, questions)
            out_path = (
                RESULTS_DIR / f"neighbor_{variant}_{ds_name}.json"
            )
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            all_summaries.setdefault(variant, {})[ds_name] = {
                "summary": payload["summary"],
                "category_breakdown": payload["category_breakdown"],
            }

    summary_path = RESULTS_DIR / "neighbor_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved: {summary_path}")

    # Final table
    print("\n" + "=" * 100)
    print("NEIGHBOR PRIORITY SUMMARY (K=20, fair backfill)")
    print("=" * 100)
    header = (
        f"{'Variant':<26s} {'Dataset':<14s} "
        f"{'base@20':>8s} {'arch@20':>8s} {'d@20':>7s} {'W/T/L':>10s}"
    )
    print(header)
    print("-" * len(header))
    for variant in VARIANTS:
        for ds_name in DATASETS:
            s = all_summaries[variant][ds_name]["summary"]
            print(
                f"{variant:<26s} {ds_name:<14s} "
                f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
                f"{s['delta_r@20']:>+7.3f} {s['W/T/L_r@20']:>10s}"
            )


if __name__ == "__main__":
    main()
