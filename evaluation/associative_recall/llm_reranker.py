"""LLM-based reranking for multi-retrieval architectures.

Post-processing step: takes a pool of retrieved segments + the original
question, has the LLM score/rank relevance, returns a reranked list.

Variants:
  - listwise: Show LLM all segments, ask it to select the top-K most relevant
  - batch_score: Show LLM all segments, ask it to score each 0-10

Usage:
    uv run python llm_reranker.py [--arch <name>] [--all] [--variant listwise|batch_score]
    uv run python llm_reranker.py --all --verbose
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
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50, 100]

# Target architectures from best_shot.py whose output we rerank
TARGET_ARCHS = [
    "decompose_then_retrieve",
    "retrieve_then_decompose",
    "frontier_v2_iterative",
    "v15_control",
    "interleaved",
    "meta_v2f",
    "flat_multi_cue",
]


# ---------------------------------------------------------------------------
# Cache — reads all existing caches, writes to rerank-specific file
# ---------------------------------------------------------------------------
class RerankLLMCache(LLMCache):
    """Reads all existing LLM caches, writes to rerank-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in (
            "llm_cache.json",
            "bestshot_llm_cache.json",
            "rerank_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "rerank_llm_cache.json"
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


class RerankEmbeddingCache(EmbeddingCache):
    """Reads existing embedding caches for loading segment stores."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in (
            "embedding_cache.json",
            "bestshot_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "rerank_embedding_cache.json"
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


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

LISTWISE_RERANK_PROMPT = """\
You are a relevance judge for a memory retrieval system. A user asked a \
question about a past conversation, and a retrieval system found the segments \
below. Your job: select the segments most relevant to ANSWERING the question.

QUESTION: {question}

SEGMENTS (numbered for reference):
{segments_text}

Select the {top_k} most relevant segments for answering this question. \
A segment is relevant if it contains information that directly helps answer \
the question — names, dates, events, opinions, decisions, or context \
mentioned in the question.

Rank them from MOST relevant to LEAST relevant. Output ONLY the segment \
numbers, one per line, most relevant first. Output exactly {top_k} numbers.

Format:
RANK: <number>
RANK: <number>
...
Nothing else."""

BATCH_SCORE_RERANK_PROMPT = """\
You are a relevance judge for a memory retrieval system. A user asked a \
question about a past conversation, and a retrieval system found the segments \
below. Your job: score each segment's relevance to ANSWERING the question.

QUESTION: {question}

SEGMENTS (numbered for reference):
{segments_text}

Score each segment from 0 to 10:
  10 = directly answers the question or contains key facts needed
  7-9 = strongly relevant, contains important context
  4-6 = somewhat relevant, mentions related topics
  1-3 = weakly relevant, tangentially related
  0 = not relevant at all

Output one score per segment, in order. Format:
SCORE 1: <score>
SCORE 2: <score>
...
Nothing else."""


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
class LLMReranker:
    """Reranks a pool of segments using LLM-based relevance scoring."""

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = MODEL,
        variant: str = "listwise",
        batch_size: int = 40,
    ):
        self.client = client or OpenAI(timeout=300.0)
        self.model = model
        self.variant = variant
        self.batch_size = batch_size
        self.llm_cache = RerankLLMCache()
        self.llm_calls = 0

    def llm_call(self, prompt: str) -> str:
        cached = self.llm_cache.get(self.model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        last_err = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4000,
                )
                text = response.choices[0].message.content or ""
                self.llm_cache.put(self.model, prompt, text)
                self.llm_calls += 1
                return text
            except Exception as e:
                last_err = e
                import time as _time
                _time.sleep(2 ** attempt)
        raise last_err  # type: ignore[misc]

    def save_caches(self) -> None:
        self.llm_cache.save()

    def reset_counters(self) -> None:
        self.llm_calls = 0

    def _format_segments_for_prompt(
        self, segments: list[Segment], max_chars: int = 300
    ) -> str:
        lines = []
        for i, seg in enumerate(segments, 1):
            text = seg.text[:max_chars]
            lines.append(f"[{i}] Turn {seg.turn_id}, {seg.role}: {text}")
        return "\n".join(lines)

    def rerank(
        self,
        question: str,
        segments: list[Segment],
        top_k: int = 20,
    ) -> list[Segment]:
        """Rerank segments by LLM-judged relevance.

        If there are more segments than batch_size, split into batches,
        rerank each batch, then merge-rerank the top results.
        """
        if not segments:
            return []

        if len(segments) <= self.batch_size:
            return self._rerank_single_batch(question, segments, top_k)

        # Multi-batch: split, rerank each, then merge
        batches = []
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i : i + self.batch_size]
            batches.append(batch)

        # Per-batch: get top_k from each batch
        per_batch_top = max(top_k, 20)
        candidates = []
        for batch in batches:
            batch_top = self._rerank_single_batch(
                question, batch, min(per_batch_top, len(batch))
            )
            candidates.extend(batch_top)

        # Final merge-rerank if we got more than top_k candidates
        if len(candidates) > top_k:
            return self._rerank_single_batch(question, candidates, top_k)
        return candidates

    def _rerank_single_batch(
        self,
        question: str,
        segments: list[Segment],
        top_k: int,
    ) -> list[Segment]:
        """Rerank a single batch of segments."""
        if self.variant == "listwise":
            return self._rerank_listwise(question, segments, top_k)
        elif self.variant == "batch_score":
            return self._rerank_batch_score(question, segments, top_k)
        else:
            raise ValueError(f"Unknown rerank variant: {self.variant}")

    def _rerank_listwise(
        self,
        question: str,
        segments: list[Segment],
        top_k: int,
    ) -> list[Segment]:
        """Variant 3: Listwise selection — LLM picks top-K in order."""
        effective_k = min(top_k, len(segments))
        segments_text = self._format_segments_for_prompt(segments)

        prompt = LISTWISE_RERANK_PROMPT.format(
            question=question,
            segments_text=segments_text,
            top_k=effective_k,
        )

        response = self.llm_call(prompt)
        ranked_indices = self._parse_rank_response(response, len(segments))

        # Build reranked list: ranked items first, then remaining in original order
        reranked = []
        seen = set()
        for idx in ranked_indices[:effective_k]:
            if idx < len(segments) and idx not in seen:
                reranked.append(segments[idx])
                seen.add(idx)

        # Append any segments not mentioned by the LLM (preserve original order)
        for i, seg in enumerate(segments):
            if i not in seen:
                reranked.append(seg)

        return reranked

    def _rerank_batch_score(
        self,
        question: str,
        segments: list[Segment],
        top_k: int,
    ) -> list[Segment]:
        """Variant 2: Batch scoring — LLM scores each segment 0-10."""
        segments_text = self._format_segments_for_prompt(segments)

        prompt = BATCH_SCORE_RERANK_PROMPT.format(
            question=question,
            segments_text=segments_text,
        )

        response = self.llm_call(prompt)
        scores = self._parse_score_response(response, len(segments))

        # Sort by score descending, break ties by original position
        indexed = list(enumerate(segments))
        indexed.sort(key=lambda x: (-scores.get(x[0], 0), x[0]))

        return [seg for _, seg in indexed]

    def _parse_rank_response(
        self, response: str, num_segments: int
    ) -> list[int]:
        """Parse RANK: <number> lines from response."""
        indices = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("RANK:"):
                try:
                    num = int(line.split(":", 1)[1].strip())
                    # Convert 1-based to 0-based
                    idx = num - 1
                    if 0 <= idx < num_segments:
                        indices.append(idx)
                except (ValueError, IndexError):
                    continue
            else:
                # Also try parsing bare numbers
                try:
                    num = int(line.strip().rstrip("."))
                    idx = num - 1
                    if 0 <= idx < num_segments:
                        indices.append(idx)
                except (ValueError, IndexError):
                    continue
        return indices

    def _parse_score_response(
        self, response: str, num_segments: int
    ) -> dict[int, float]:
        """Parse SCORE N: <score> lines from response."""
        scores: dict[int, float] = {}
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("SCORE"):
                try:
                    # "SCORE 1: 7" or "SCORE 1: 7/10"
                    parts = line.split(":", 1)
                    seg_num = int(parts[0].split()[-1])
                    score_text = parts[1].strip().split("/")[0].strip()
                    score = float(score_text)
                    idx = seg_num - 1
                    if 0 <= idx < num_segments:
                        scores[idx] = score
                except (ValueError, IndexError):
                    continue
        return scores


# ---------------------------------------------------------------------------
# Reconstruct segments from saved results
# ---------------------------------------------------------------------------
def reconstruct_segments(
    question_result: dict,
    store: SegmentStore,
    conversation_id: str,
) -> list[Segment]:
    """Reconstruct the ordered list of Segment objects that an architecture
    returned, using the SegmentStore to look up full segment data.

    We re-run the architecture's retrieval logic to get the exact segment
    ordering. But since all LLM/embedding calls are cached, this is fast.
    """
    # We cannot reconstruct from saved results alone (they don't save segment
    # indices). Instead, we'll re-run the architecture. This is done in the
    # main evaluation loop below.
    raise NotImplementedError("Use re-run approach instead")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def compute_recall(
    retrieved_turn_ids: set[int], source_turn_ids: set[int]
) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_reranked(
    question: dict,
    original_segments: list[Segment],
    reranked_segments: list[Segment],
    original_llm_calls: int,
    rerank_llm_calls: int,
) -> dict:
    """Compare original ordering vs reranked ordering."""
    source_ids = set(question["source_chat_ids"])

    original_recalls: dict[str, float] = {}
    reranked_recalls: dict[str, float] = {}

    for budget in BUDGETS:
        orig_ids = {s.turn_id for s in original_segments[:budget]}
        rerank_ids = {s.turn_id for s in reranked_segments[:budget]}
        original_recalls[f"r@{budget}"] = compute_recall(orig_ids, source_ids)
        reranked_recalls[f"r@{budget}"] = compute_recall(
            rerank_ids, source_ids
        )

    # Also compute baseline (cosine) recalls — load from saved results
    # r@actual for the whole pool
    orig_all_ids = {s.turn_id for s in original_segments}
    rerank_all_ids = {s.turn_id for s in reranked_segments}
    original_recalls["r@actual"] = compute_recall(orig_all_ids, source_ids)
    reranked_recalls["r@actual"] = compute_recall(rerank_all_ids, source_ids)

    return {
        "conversation_id": question["conversation_id"],
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": question["question"],
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "total_in_pool": len(original_segments),
        "original_recalls": original_recalls,
        "reranked_recalls": reranked_recalls,
        "original_llm_calls": original_llm_calls,
        "rerank_llm_calls": rerank_llm_calls,
        "total_llm_calls": original_llm_calls + rerank_llm_calls,
    }


def summarize_rerank(
    results: list[dict], arch_name: str, variant: str, benchmark: str
) -> dict:
    """Compute summary statistics for reranking results."""
    n = len(results)
    if n == 0:
        return {}

    summary: dict = {
        "arch": arch_name,
        "rerank_variant": variant,
        "benchmark": benchmark,
        "n": n,
    }

    for label in [f"r@{b}" for b in BUDGETS] + ["r@actual"]:
        orig_vals = [r["original_recalls"][label] for r in results]
        rerank_vals = [r["reranked_recalls"][label] for r in results]
        orig_mean = sum(orig_vals) / n
        rerank_mean = sum(rerank_vals) / n

        wins = sum(
            1 for o, r in zip(orig_vals, rerank_vals) if r > o + 0.001
        )
        losses = sum(
            1 for o, r in zip(orig_vals, rerank_vals) if o > r + 0.001
        )
        ties = n - wins - losses

        summary[f"original_{label}"] = round(orig_mean, 4)
        summary[f"reranked_{label}"] = round(rerank_mean, 4)
        summary[f"delta_{label}"] = round(rerank_mean - orig_mean, 4)
        summary[f"W/T/L_{label}"] = f"{wins}/{ties}/{losses}"

    summary["avg_pool_size"] = round(
        sum(r["total_in_pool"] for r in results) / n, 1
    )
    summary["avg_original_llm"] = round(
        sum(r["original_llm_calls"] for r in results) / n, 1
    )
    summary["avg_rerank_llm"] = round(
        sum(r["rerank_llm_calls"] for r in results) / n, 1
    )
    summary["avg_total_llm"] = round(
        sum(r["total_llm_calls"] for r in results) / n, 1
    )

    return summary


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    """Per-category breakdown at r@20."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    cat_summaries = {}
    for cat, cat_results in sorted(by_cat.items()):
        n = len(cat_results)
        orig_vals = [r["original_recalls"]["r@20"] for r in cat_results]
        rerank_vals = [r["reranked_recalls"]["r@20"] for r in cat_results]
        orig_mean = sum(orig_vals) / n
        rerank_mean = sum(rerank_vals) / n
        wins = sum(
            1 for o, r in zip(orig_vals, rerank_vals) if r > o + 0.001
        )
        losses = sum(
            1 for o, r in zip(orig_vals, rerank_vals) if o > r + 0.001
        )
        cat_summaries[cat] = {
            "n": n,
            "original_r@20": round(orig_mean, 4),
            "reranked_r@20": round(rerank_mean, 4),
            "delta_r@20": round(rerank_mean - orig_mean, 4),
            "W/T/L": f"{wins}/{n - wins - losses}/{losses}",
        }
    return cat_summaries


# ---------------------------------------------------------------------------
# Main evaluation: re-run architectures + rerank
# ---------------------------------------------------------------------------
def run_rerank_evaluation(
    arch_name: str,
    arch_instance,  # BestshotBase subclass
    reranker: LLMReranker,
    questions: list[dict],
    benchmark_label: str,
    baseline_results: list[dict] | None = None,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Run architecture + reranker on all questions."""
    print(f"\n{'='*70}")
    print(
        f"RERANK: {arch_name} + {reranker.variant} | "
        f"{benchmark_label} | {len(questions)} questions"
    )
    print(f"{'='*70}")

    results = []
    for i, question in enumerate(questions):
        q_text = question["question"]
        conv_id = question["conversation_id"]
        q_short = q_text[:55]
        print(
            f"  [{i+1}/{len(questions)}] {question.get('category', '?')}: "
            f"{q_short}...",
            flush=True,
        )

        try:
            # Step 1: Run the architecture to get the segment pool
            arch_instance.reset_counters()
            arch_result = arch_instance.retrieve(q_text, conv_id)
            arch_llm_calls = arch_instance.llm_calls

            # Deduplicate preserving order
            seen: set[int] = set()
            original_segments: list[Segment] = []
            for seg in arch_result.segments:
                if seg.index not in seen:
                    original_segments.append(seg)
                    seen.add(seg.index)

            # Step 2: Rerank
            reranker.reset_counters()
            reranked_segments = reranker.rerank(
                q_text, original_segments, top_k=20
            )
            rerank_llm_calls = reranker.llm_calls

            # Step 3: Evaluate
            result = evaluate_reranked(
                question,
                original_segments,
                reranked_segments,
                arch_llm_calls,
                rerank_llm_calls,
            )

            # Add baseline recalls from saved results if available
            if baseline_results:
                for br in baseline_results:
                    if br["question_index"] == question.get("question_index"):
                        result["baseline_recalls"] = br["baseline_recalls"]
                        break

            results.append(result)

            if verbose:
                orig_r20 = result["original_recalls"]["r@20"]
                rerank_r20 = result["reranked_recalls"]["r@20"]
                delta = rerank_r20 - orig_r20
                marker = (
                    "W" if delta > 0.001 else ("L" if delta < -0.001 else "T")
                )
                print(
                    f"    pool={len(original_segments)}, "
                    f"orig_r@20={orig_r20:.3f}, "
                    f"rerank_r@20={rerank_r20:.3f}, "
                    f"delta={delta:+.3f} [{marker}]"
                )

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()

        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            arch_instance.save_caches()
            reranker.save_caches()

    arch_instance.save_caches()
    reranker.save_caches()

    summary = summarize_rerank(
        results, arch_name, reranker.variant, benchmark_label
    )

    # Print compact summary
    print(f"\n--- {arch_name} + {reranker.variant} on {benchmark_label} ---")
    for budget in BUDGETS:
        lbl = f"r@{budget}"
        print(
            f"  {lbl}: original={summary.get(f'original_{lbl}', 0):.3f} "
            f"reranked={summary.get(f'reranked_{lbl}', 0):.3f} "
            f"delta={summary.get(f'delta_{lbl}', 0):+.3f} "
            f"W/T/L={summary.get(f'W/T/L_{lbl}', '?')}"
        )
    print(
        f"  Pool: {summary.get('avg_pool_size', 0):.0f}, "
        f"Retrieval LLM: {summary.get('avg_original_llm', 0):.1f}, "
        f"Rerank LLM: {summary.get('avg_rerank_llm', 0):.1f}, "
        f"Total LLM: {summary.get('avg_total_llm', 0):.1f}"
    )

    cat_summaries = summarize_by_category(results)
    print(f"\n  Per-category (r@20):")
    for cat, cs in cat_summaries.items():
        print(
            f"    {cat}: delta={cs['delta_r@20']:+.3f} "
            f"W/T/L={cs['W/T/L']} (n={cs['n']})"
        )

    return results, summary


def print_comparison_table(
    all_summaries: dict[str, dict[str, dict]],
    benchmark: str,
):
    """Print comparison table: original vs reranked vs baseline."""
    print(f"\n{'='*100}")
    print(f"COMPARISON TABLE — {benchmark.upper()}")
    print(f"{'='*100}")

    # Load baseline from bestshot results for reference
    baseline_r20 = None
    for arch_name in all_summaries:
        result_file = RESULTS_DIR / f"bestshot_{arch_name}_{benchmark}.json"
        if result_file.exists():
            with open(result_file) as f:
                saved = json.load(f)
            s = saved.get("summary", {})
            if baseline_r20 is None:
                baseline_r20 = s.get("baseline_r@20", 0)
            break

    header = (
        f"{'Architecture':<30s} {'Variant':<14s} "
        f"{'orig r@20':>10s} {'rerank r@20':>12s} {'delta':>8s} "
        f"{'W/T/L':>10s} {'orig r@50':>10s} {'rerank r@50':>12s} "
        f"{'Pool':>6s} {'LLM tot':>8s}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for arch_name in all_summaries:
        for variant_key, s in all_summaries[arch_name].items():
            if benchmark not in variant_key:
                continue
            rows.append((arch_name, s))

    rows.sort(key=lambda x: x[1].get("delta_r@20", 0), reverse=True)

    for arch_name, s in rows:
        variant = s.get("rerank_variant", "?")
        orig_20 = s.get("original_r@20", 0)
        rerank_20 = s.get("reranked_r@20", 0)
        delta_20 = s.get("delta_r@20", 0)
        wtl = s.get("W/T/L_r@20", "?")
        orig_50 = s.get("original_r@50", 0)
        rerank_50 = s.get("reranked_r@50", 0)
        pool = s.get("avg_pool_size", 0)
        llm_tot = s.get("avg_total_llm", 0)

        print(
            f"  {arch_name:<28s} {variant:<14s} "
            f"{orig_20:>9.3f}  {rerank_20:>11.3f}  {delta_20:>+7.3f} "
            f"{wtl:>10s} {orig_50:>9.3f}  {rerank_50:>11.3f}  "
            f"{pool:>5.0f}  {llm_tot:>7.1f}"
        )

    if baseline_r20 is not None:
        print(f"\n  Cosine baseline r@20 = {baseline_r20:.3f}")


def print_cost_benefit_table(
    all_summaries: dict[str, dict[str, dict]],
    benchmark: str,
):
    """Print cost-benefit analysis: is reranking worth the extra LLM cost?"""
    print(f"\n{'='*100}")
    print(f"COST-BENEFIT ANALYSIS — {benchmark.upper()}")
    print(f"{'='*100}")

    # Load bestshot summaries for comparison
    bestshot_file = RESULTS_DIR / "bestshot_all_summaries.json"
    bestshot_summaries = {}
    if bestshot_file.exists():
        with open(bestshot_file) as f:
            bestshot_summaries = json.load(f)

    print(
        f"\n{'System':<45s} {'r@20':>8s} {'r@50':>8s} "
        f"{'LLM calls':>10s} {'Pool':>6s}"
    )
    print("-" * 80)

    # First show bestshot baselines
    for arch_name in sorted(bestshot_summaries.keys()):
        if benchmark in bestshot_summaries[arch_name]:
            bs = bestshot_summaries[arch_name][benchmark]
            r20 = bs.get("arch_r@20", 0)
            r50 = bs.get("arch_r@50", 0)
            llm = bs.get("avg_llm_calls", 0)
            pool = bs.get("avg_total_retrieved", 0)
            print(
                f"  {arch_name + ' (no rerank)':<43s} "
                f"{r20:>7.3f}  {r50:>7.3f}  {llm:>9.1f}  {pool:>5.0f}"
            )

    print("-" * 80)

    # Then show reranked versions
    for arch_name in sorted(all_summaries.keys()):
        for variant_key, s in sorted(all_summaries[arch_name].items()):
            if benchmark not in variant_key:
                continue
            variant = s.get("rerank_variant", "?")
            r20 = s.get("reranked_r@20", 0)
            r50 = s.get("reranked_r@50", 0)
            llm = s.get("avg_total_llm", 0)
            pool = s.get("avg_pool_size", 0)
            label = f"{arch_name} + {variant}"
            print(
                f"  {label:<43s} "
                f"{r20:>7.3f}  {r50:>7.3f}  {llm:>9.1f}  {pool:>5.0f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-based reranking for multi-retrieval architectures"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Run specific architecture (e.g., decompose_then_retrieve)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all target architectures",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="listwise",
        choices=["listwise", "batch_score", "both"],
        help="Reranking variant",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--locomo-only",
        action="store_true",
        help="Skip synthetic benchmark",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip LoCoMo benchmark",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available architectures",
    )
    args = parser.parse_args()

    if args.list:
        print("Available architectures:")
        for name in TARGET_ARCHS:
            print(f"  {name}")
        return

    # Determine which architectures to run
    if args.arch:
        arch_names = [args.arch]
    elif args.all:
        arch_names = list(TARGET_ARCHS)
    else:
        # Default: the 3 main targets
        arch_names = [
            "decompose_then_retrieve",
            "retrieve_then_decompose",
            "frontier_v2_iterative",
        ]

    # Determine rerank variants
    if args.variant == "both":
        variants = ["listwise", "batch_score"]
    else:
        variants = [args.variant]

    # Import best_shot architecture classes
    from best_shot import (
        DecomposeThenRetrieve,
        FlatMultiCue,
        FrontierV2Iterative,
        Interleaved,
        MetaV2f,
        RetrieveThenDecompose,
        V15Control,
        build_architectures,
    )

    # Load LoCoMo data
    with open(DATA_DIR / "questions_extended.json") as f:
        all_questions = json.load(f)
    locomo_store = SegmentStore(
        data_dir=DATA_DIR, npz_name="segments_extended.npz"
    )
    locomo_qs = [
        q for q in all_questions if q.get("benchmark") == "locomo"
    ][:30]
    print(
        f"LoCoMo: {len(locomo_qs)} questions, "
        f"{len(locomo_store.segments)} segments"
    )

    # Load synthetic data
    synth_store = None
    synth_qs: list[dict] = []
    synth_path = DATA_DIR / "questions_synthetic.json"
    if synth_path.exists():
        with open(synth_path) as f:
            synth_qs = json.load(f)
        synth_store = SegmentStore(
            data_dir=DATA_DIR, npz_name="segments_synthetic.npz"
        )
        print(
            f"Synthetic: {len(synth_qs)} questions, "
            f"{len(synth_store.segments)} segments"
        )
    else:
        print("Synthetic data not found, skipping.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect summaries: {arch: {variant_benchmark: summary}}
    all_summaries: dict[str, dict[str, dict]] = {}

    # Run on LoCoMo
    if not args.synthetic_only:
        locomo_archs = build_architectures(locomo_store)

        for arch_name in arch_names:
            if arch_name not in locomo_archs:
                print(f"Unknown architecture: {arch_name}")
                continue

            # Load existing baseline results for this arch
            baseline_results = None
            baseline_file = (
                RESULTS_DIR / f"bestshot_{arch_name}_locomo_30q.json"
            )
            if baseline_file.exists():
                with open(baseline_file) as f:
                    baseline_data = json.load(f)
                baseline_results = baseline_data.get("results")

            for variant in variants:
                result_file = (
                    RESULTS_DIR
                    / f"rerank_{variant}_{arch_name}_locomo_30q.json"
                )
                if result_file.exists() and not args.force:
                    print(
                        f"\nSkipping {arch_name}+{variant} on LoCoMo "
                        f"(exists). Use --force to rerun."
                    )
                    with open(result_file) as f:
                        saved = json.load(f)
                    results = saved["results"]
                    summary = saved["summary"]
                else:
                    arch = locomo_archs[arch_name]
                    reranker = LLMReranker(
                        model=MODEL,
                        variant=variant,
                    )

                    results, summary = run_rerank_evaluation(
                        arch_name,
                        arch,
                        reranker,
                        locomo_qs,
                        "locomo_30q",
                        baseline_results=baseline_results,
                        verbose=args.verbose,
                    )

                    # Save
                    with open(result_file, "w") as f:
                        json.dump(
                            {"results": results, "summary": summary},
                            f,
                            indent=2,
                            default=str,
                        )
                    print(f"  Saved: {result_file}")

                if arch_name not in all_summaries:
                    all_summaries[arch_name] = {}
                key = f"{variant}_locomo_30q"
                all_summaries[arch_name][key] = summary

    # Run on synthetic
    if (
        not args.locomo_only
        and synth_store is not None
        and synth_qs
    ):
        from best_shot import build_architectures as build_archs

        synth_archs = build_archs(synth_store)

        for arch_name in arch_names:
            if arch_name not in synth_archs:
                print(f"Unknown architecture for synthetic: {arch_name}")
                continue

            # Load existing baseline results
            baseline_results = None
            baseline_file = (
                RESULTS_DIR / f"bestshot_{arch_name}_synthetic_19q.json"
            )
            if baseline_file.exists():
                with open(baseline_file) as f:
                    baseline_data = json.load(f)
                baseline_results = baseline_data.get("results")

            for variant in variants:
                result_file = (
                    RESULTS_DIR
                    / f"rerank_{variant}_{arch_name}_synthetic_19q.json"
                )
                if result_file.exists() and not args.force:
                    print(
                        f"\nSkipping {arch_name}+{variant} on synthetic "
                        f"(exists). Use --force to rerun."
                    )
                    with open(result_file) as f:
                        saved = json.load(f)
                    results = saved["results"]
                    summary = saved["summary"]
                else:
                    arch = synth_archs[arch_name]
                    reranker = LLMReranker(
                        model=MODEL,
                        variant=variant,
                    )

                    results, summary = run_rerank_evaluation(
                        arch_name,
                        arch,
                        reranker,
                        synth_qs,
                        "synthetic_19q",
                        baseline_results=baseline_results,
                        verbose=args.verbose,
                    )

                    # Save
                    with open(result_file, "w") as f:
                        json.dump(
                            {"results": results, "summary": summary},
                            f,
                            indent=2,
                            default=str,
                        )
                    print(f"  Saved: {result_file}")

                if arch_name not in all_summaries:
                    all_summaries[arch_name] = {}
                key = f"{variant}_synthetic_19q"
                all_summaries[arch_name][key] = summary

    # Print comparison tables
    for benchmark in ["locomo_30q", "synthetic_19q"]:
        has_data = any(
            any(benchmark in k for k in v)
            for v in all_summaries.values()
        )
        if has_data:
            print_comparison_table(all_summaries, benchmark)
            print_cost_benefit_table(all_summaries, benchmark)

    # Save all summaries
    summary_file = RESULTS_DIR / "rerank_all_summaries.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summaries to {summary_file}")


if __name__ == "__main__":
    main()
