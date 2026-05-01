"""Experiments: V2f generalized and double V2f.

Experiment 1 (V2f_general): V2f prompt with domain-agnostic language.
Experiment 2 (double_v2f): Two rounds of V2f retrieval.

Usage:
    uv run python eval_v2f_experiments.py --experiment v2f_general
    uv run python eval_v2f_experiments.py --experiment double_v2f
    uv run python eval_v2f_experiments.py --all
"""

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
BUDGETS = [20, 50, 100]


# ---------------------------------------------------------------------------
# Cache classes -- uses general_llm_cache.json / general_embedding_cache.json
# ---------------------------------------------------------------------------
class GeneralEmbeddingCache(EmbeddingCache):
    """Reads all existing caches, writes to general-specific file."""

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
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "general_embedding_cache.json"
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


class GeneralLLMCache(LLMCache):
    """Reads all existing caches, writes to general_llm_cache.json."""

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
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "general_llm_cache.json"
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
# Prompts
# ---------------------------------------------------------------------------

# Original V2f (conversation-specific language) -- control
META_V2F_PROMPT = """\
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

# V2f_general: domain-agnostic language
META_V2F_GENERAL_PROMPT = """\
You are generating search text for semantic retrieval over stored text. \
Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target stored content.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in the source content.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class ExperimentBase:
    """Base class with embedding/LLM utilities and counters."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = GeneralEmbeddingCache()
        self.llm_cache = GeneralLLMCache()
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_segments(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
    """Format segments chronologically. Matches v15 control format."""
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def _parse_cues(response: str) -> list[str]:
    """Parse CUE: lines from LLM response."""
    cues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


@dataclass
class ExperimentResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


# ===========================================================================
# V2f single-call variant (control + general)
# ===========================================================================
class MetaV2fVariant(ExperimentBase):
    """Single-call V2f variant. Takes a prompt template."""

    def __init__(
        self,
        store: SegmentStore,
        prompt_template: str,
        client: OpenAI | None = None,
    ):
        super().__init__(store, client)
        self.prompt_template = prompt_template

    def retrieve(self, question: str, conversation_id: str) -> ExperimentResult:
        # Hop 0: embed question, retrieve top-10
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        # Build context section
        context = _format_segments(all_segments)
        context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + context

        # Single LLM call
        prompt = self.prompt_template.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        if not cues:
            return ExperimentResult(
                segments=all_segments,
                metadata={"output": output, "cues": []},
            )

        # Retrieve with cues
        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    all_segments.append(seg)
                    exclude_indices.add(seg.index)

        return ExperimentResult(
            segments=all_segments,
            metadata={"output": output, "cues": cues[:2]},
        )


# ===========================================================================
# Double V2f: two consecutive V2f rounds
# ===========================================================================
class DoubleV2f(ExperimentBase):
    """Two rounds of V2f retrieval.

    Round 1: Standard V2f (question -> top-10, V2f 2 cues -> top-10 each)
    Round 2: Show ALL round 1 segments, V2f generates 2 MORE cues,
             retrieve per cue (top-10 each, excluding all found)
    """

    def __init__(
        self,
        store: SegmentStore,
        prompt_template: str,
        client: OpenAI | None = None,
    ):
        super().__init__(store, client)
        self.prompt_template = prompt_template

    def _v2f_round(
        self,
        question: str,
        conversation_id: str,
        existing_segments: list[Segment],
        exclude_indices: set[int],
        previous_cues: list[str],
    ) -> tuple[list[Segment], set[int], list[str], str]:
        """Execute one V2f round: LLM call + cue retrieval.

        Returns (new_segments_added, updated_exclude, cues, llm_output).
        """
        # Build context section showing all accumulated segments
        context = _format_segments(existing_segments, max_items=16)
        context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + context
        if previous_cues:
            context_section += (
                "\n\nPREVIOUS CUES ALREADY TRIED (do NOT repeat or paraphrase):\n"
                + "\n".join(f"- {c}" for c in previous_cues)
            )

        prompt = self.prompt_template.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        new_segments: list[Segment] = []
        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    new_segments.append(seg)
                    exclude_indices.add(seg.index)

        return new_segments, exclude_indices, cues[:2], output

    def retrieve(self, question: str, conversation_id: str) -> ExperimentResult:
        # Hop 0: embed question, retrieve top-10
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        # Round 1: V2f with initial context
        r1_new, exclude_indices, cues_r1, output_r1 = self._v2f_round(
            question, conversation_id, all_segments, exclude_indices, []
        )
        all_segments.extend(r1_new)

        # Round 2: V2f with ALL accumulated context + previous cues
        r2_new, exclude_indices, cues_r2, output_r2 = self._v2f_round(
            question, conversation_id, all_segments, exclude_indices, cues_r1
        )
        all_segments.extend(r2_new)

        return ExperimentResult(
            segments=all_segments,
            metadata={
                "cues_r1": cues_r1,
                "cues_r2": cues_r2,
                "output_r1": output_r1,
                "output_r2": output_r2,
                "total_segments": len(all_segments),
            },
        )


# ===========================================================================
# Evaluation functions (same as prompt_optimization.py)
# ===========================================================================
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    arch: ExperimentBase,
    question: dict,
    verbose: bool = False,
) -> dict:
    """Evaluate a single variant on a single question."""
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
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

    # Baseline: cosine top-N at same budget
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
        cues = result.metadata.get("cues", result.metadata.get("cues_r1", []))
        for cue in cues[:4]:
            print(f"    Cue: {cue[:120]}")

    return row


def summarize(results: list[dict], variant_name: str, benchmark: str) -> dict:
    """Compute summary statistics."""
    n = len(results)
    if n == 0:
        return {}

    summary: dict = {"variant": variant_name, "benchmark": benchmark, "n": n}

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


def run_variant(
    variant_name: str,
    arch: ExperimentBase,
    questions: list[dict],
    benchmark_label: str,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Run one variant, return (results, summary)."""
    print(f"\n{'=' * 70}")
    print(
        f"VARIANT: {variant_name} | BENCHMARK: {benchmark_label} | "
        f"{len(questions)} questions"
    )
    print(f"{'=' * 70}")

    results = []
    for i, question in enumerate(questions):
        q_short = question["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] {question['category']}: {q_short}...",
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
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = summarize(results, variant_name, benchmark_label)

    # Print compact summary
    print(f"\n--- {variant_name} on {benchmark_label} ---")
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
            f"    {cat}: delta={cs['delta_r@20']:+.3f} "
            f"W/T/L={cs['W/T/L']} (n={cs['n']})"
        )

    return results, summary


# ===========================================================================
# Head-to-head comparison (V2f vs variant)
# ===========================================================================
def compare_v2f_vs_variant(
    v2f_results: list[dict],
    variant_results: list[dict],
    variant_name: str,
) -> dict:
    """Compare variant against V2f (not baseline). Reports delta and W/T/L."""
    n = len(v2f_results)
    comparison = {"variant": variant_name, "n": n}

    for label in [f"r@{b}" for b in BUDGETS] + ["r@actual"]:
        v2f_vals = [r["arch_recalls"][label] for r in v2f_results]
        var_vals = [r["arch_recalls"][label] for r in variant_results]
        v2f_mean = sum(v2f_vals) / n
        var_mean = sum(var_vals) / n

        wins = sum(1 for v, a in zip(v2f_vals, var_vals) if a > v + 0.001)
        losses = sum(1 for v, a in zip(v2f_vals, var_vals) if v > a + 0.001)
        ties = n - wins - losses

        comparison[f"v2f_{label}"] = round(v2f_mean, 4)
        comparison[f"variant_{label}"] = round(var_mean, 4)
        comparison[f"delta_{label}"] = round(var_mean - v2f_mean, 4)
        comparison[f"W/T/L_{label}"] = f"{wins}/{ties}/{losses}"

    return comparison


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="V2f experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["v2f_general", "double_v2f"],
        default=None,
        help="Which experiment to run",
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing")
    args = parser.parse_args()

    # Load data
    with open(DATA_DIR / "questions_extended.json") as f:
        all_questions = json.load(f)

    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_extended.npz")
    print(f"Loaded {len(store.segments)} segments")

    locomo_qs = [q for q in all_questions if q.get("benchmark") == "locomo"][
        : args.num_questions
    ]
    print(f"LoCoMo: {len(locomo_qs)} questions")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments_to_run = []
    if args.all:
        experiments_to_run = ["v2f_general", "double_v2f"]
    elif args.experiment:
        experiments_to_run = [args.experiment]
    else:
        experiments_to_run = ["v2f_general", "double_v2f"]

    # Always run V2f control first (needed for head-to-head comparison)
    v2f_results_file = (
        RESULTS_DIR / f"optim_meta_v2f_antipattern_locomo_{args.num_questions}q.json"
    )
    v2f_results = None
    if v2f_results_file.exists():
        print(f"\nLoading V2f control from {v2f_results_file}")
        with open(v2f_results_file) as f:
            v2f_results = json.load(f)
        v2f_summary = summarize(
            v2f_results, "v2f_control", f"locomo_{args.num_questions}q"
        )
        print(
            f"  V2f control r@20: arch={v2f_summary.get('arch_r@20', 0):.3f} "
            f"delta={v2f_summary.get('delta_r@20', 0):+.3f}"
        )
    else:
        # Run V2f control
        print("\nRunning V2f control (needed for comparison)...")
        v2f_arch = MetaV2fVariant(store, META_V2F_PROMPT)
        v2f_results, v2f_summary = run_variant(
            "v2f_control",
            v2f_arch,
            locomo_qs,
            f"locomo_{args.num_questions}q",
            verbose=args.verbose,
        )
        # Save as the canonical V2f results
        with open(v2f_results_file, "w") as f:
            json.dump(v2f_results, f, indent=2, default=str)
        print(f"Saved V2f control to {v2f_results_file}")

    # ----- Experiment 1: V2f General -----
    if "v2f_general" in experiments_to_run:
        results_file = (
            RESULTS_DIR / f"optim_v2f_general_locomo_{args.num_questions}q.json"
        )

        if results_file.exists() and not args.force:
            print(f"\nLoading existing V2f_general from {results_file}")
            with open(results_file) as f:
                general_results = json.load(f)
        else:
            general_arch = MetaV2fVariant(store, META_V2F_GENERAL_PROMPT)
            general_results, general_summary = run_variant(
                "v2f_general",
                general_arch,
                locomo_qs,
                f"locomo_{args.num_questions}q",
                verbose=args.verbose,
            )
            with open(results_file, "w") as f:
                json.dump(general_results, f, indent=2, default=str)
            print(f"Saved to {results_file}")

        # Head-to-head comparison
        general_summary = summarize(
            general_results, "v2f_general", f"locomo_{args.num_questions}q"
        )
        if v2f_results:
            comparison = compare_v2f_vs_variant(
                v2f_results, general_results, "v2f_general"
            )
            print(f"\n{'=' * 70}")
            print("HEAD-TO-HEAD: V2f_general vs V2f (original)")
            print(f"{'=' * 70}")
            for budget in BUDGETS:
                lbl = f"r@{budget}"
                print(
                    f"  {lbl}: v2f={comparison[f'v2f_{lbl}']:.3f} "
                    f"general={comparison[f'variant_{lbl}']:.3f} "
                    f"delta={comparison[f'delta_{lbl}']:+.3f} "
                    f"W/T/L={comparison[f'W/T/L_{lbl}']}"
                )

    # ----- Experiment 2: Double V2f -----
    if "double_v2f" in experiments_to_run:
        results_file = (
            RESULTS_DIR / f"optim_double_v2f_locomo_{args.num_questions}q.json"
        )

        if results_file.exists() and not args.force:
            print(f"\nLoading existing double_v2f from {results_file}")
            with open(results_file) as f:
                double_results = json.load(f)
        else:
            double_arch = DoubleV2f(store, META_V2F_PROMPT)
            double_results, double_summary = run_variant(
                "double_v2f",
                double_arch,
                locomo_qs,
                f"locomo_{args.num_questions}q",
                verbose=args.verbose,
            )
            with open(results_file, "w") as f:
                json.dump(double_results, f, indent=2, default=str)
            print(f"Saved to {results_file}")

        # Head-to-head comparison
        double_summary = summarize(
            double_results, "double_v2f", f"locomo_{args.num_questions}q"
        )
        if v2f_results:
            comparison = compare_v2f_vs_variant(
                v2f_results, double_results, "double_v2f"
            )
            print(f"\n{'=' * 70}")
            print("HEAD-TO-HEAD: double_v2f vs V2f (original)")
            print(f"{'=' * 70}")
            for budget in BUDGETS:
                lbl = f"r@{budget}"
                print(
                    f"  {lbl}: v2f={comparison[f'v2f_{lbl}']:.3f} "
                    f"double={comparison[f'variant_{lbl}']:.3f} "
                    f"delta={comparison[f'delta_{lbl}']:+.3f} "
                    f"W/T/L={comparison[f'W/T/L_{lbl}']}"
                )


if __name__ == "__main__":
    main()
