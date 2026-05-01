"""Resume full evaluation: Tier 1 architectures on puzzle + advanced datasets.

Runs:
  - v15_control, meta_v2f, frontier_v2_iterative (from fulleval_run.py)
  - hybrid_v2f_gencheck (from hybrid_retrieval.py)

On:
  - Puzzle 16q: segments_puzzle.npz + questions_puzzle.json
  - Advanced 23q: segments_advanced.npz + questions_advanced.json

Saves results to results/fulleval_{puzzle|advanced}_{architecture}.json.

Usage:
    uv run python fulleval_resume.py
"""

import json
import time
from collections import defaultdict
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BUDGETS = [20, 50]

CACHE_FILE_LLM = CACHE_DIR / "resume_llm_cache.json"
CACHE_FILE_EMB = CACHE_DIR / "resume_embedding_cache.json"


# ---------------------------------------------------------------------------
# Cache classes — reads from all existing caches, writes to resume-specific
# ---------------------------------------------------------------------------
class ResumeEmbeddingCache(EmbeddingCache):
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
            "synth_test_embedding_cache.json",
            "bestshot_embedding_cache.json",
            "task_exec_embedding_cache.json",
            "general_embedding_cache.json",
            "adaptive_embedding_cache.json",
            "fulleval_embedding_cache.json",
            "hybrid_embedding_cache.json",
            "constraint_embedding_cache.json",
            "resume_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = CACHE_FILE_EMB
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


class ResumeLLMCache(LLMCache):
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
            "synth_test_llm_cache.json",
            "bestshot_llm_cache.json",
            "task_exec_llm_cache.json",
            "general_llm_cache.json",
            "adaptive_llm_cache.json",
            "fulleval_llm_cache.json",
            "hybrid_llm_cache.json",
            "constraint_llm_cache.json",
            "resume_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = CACHE_FILE_LLM
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
# Shared embedding/LLM infrastructure
# ---------------------------------------------------------------------------
embedding_cache = ResumeEmbeddingCache()
llm_cache = ResumeLLMCache()
client = OpenAI(timeout=120.0)


def embed_text(text: str) -> np.ndarray:
    text = text.strip()
    if not text:
        return np.zeros(1536, dtype=np.float32)
    cached = embedding_cache.get(text)
    if cached is not None:
        return cached
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    emb = np.array(response.data[0].embedding, dtype=np.float32)
    embedding_cache.put(text, emb)
    return emb


def llm_call(prompt: str, model: str = MODEL) -> str:
    cached = llm_cache.get(model, prompt)
    if cached is not None:
        return cached
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=3000,
    )
    text = response.choices[0].message.content or ""
    llm_cache.put(model, prompt, text)
    return text


def save_caches():
    embedding_cache.save()
    llm_cache.save()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_segments(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def parse_cues(response: str) -> list[str]:
    cues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


def parse_gaps(response: str) -> list[str]:
    gaps = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("GAP:"):
            gap = line[4:].strip()
            if gap:
                gaps.append(gap)
    return gaps


def retrieve_top_k(
    store: SegmentStore,
    query: str,
    conversation_id: str,
    top_k: int = 10,
    exclude_indices: set[int] | None = None,
) -> list[Segment]:
    query_emb = embed_text(query)
    result = store.search(
        query_emb,
        top_k=top_k,
        conversation_id=conversation_id,
        exclude_indices=exclude_indices,
    )
    return list(result.segments)


def build_context_section(
    all_segments: list[Segment],
    new_segments: list[Segment] | None = None,
    previous_cues: list[str] | None = None,
) -> str:
    if not all_segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    context = format_segments(all_segments)
    context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + context
    if new_segments:
        latest_lines = []
        for seg in sorted(new_segments, key=lambda s: s.turn_id)[:6]:
            latest_lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:200]}")
        context_section += "\n\nMOST RECENTLY FOUND (last hop):\n" + "\n".join(
            latest_lines
        )
    if previous_cues:
        context_section += (
            "\n\nPREVIOUS CUES ALREADY TRIED (do NOT repeat or paraphrase):\n"
            + "\n".join(f"- {c}" for c in previous_cues)
        )
    return context_section


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

V15_CONTROL_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

V2F_PROMPT = """\
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

FRONTIER_REFLECT_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}{explored_text}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate exactly 1 search cue targeting the most important missing \
content. Use specific vocabulary that would appear in the target \
conversation turns.

Do NOT write questions ("Did you mention X?") or search commands. \
Write text that would actually appear in a chat message.

If the retrieval looks complete, respond with DONE.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
(or)
ASSESSMENT: <evaluation>
DONE"""

GAP_ASSESSMENT_PROMPT = """\
You are reviewing retrieval results for a task. Given the question/task and \
the conversation segments retrieved so far, assess whether anything \
important is missing.

QUESTION/TASK: {question}

RETRIEVED SEGMENTS:
{formatted_segments}

Think critically:
1. Given what I've found, is there anything important for this task that \
I HAVEN'T retrieved?
2. What assumptions am I making that should be checked?
3. Are there implicit requirements (e.g., dietary restrictions, scheduling \
conflicts, prerequisites) that the question doesn't explicitly ask about \
but would be important?
4. If this is a proactive task (planning, drafting, preparing), what \
background information might I need that isn't directly mentioned in \
the question?

If there are genuine gaps, generate 1-2 targeted search cues. Each cue \
should sound like conversation content (not a search command).

If the retrieval looks comprehensive, respond with DONE.

Format:
ASSESSMENT: <what's missing or what assumptions need checking>
GAP: <text mimicking conversation content>
GAP: <text mimicking conversation content>
(or)
ASSESSMENT: <retrieval looks complete because...>
DONE"""


# ===========================================================================
# Architecture implementations
# ===========================================================================


def run_v15_control(store: SegmentStore, question: str, conv_id: str) -> list[Segment]:
    """v15_control: question top-10 + 1 LLM call producing 2 cues, each top-10."""
    query_emb = embed_text(question)
    hop0 = store.search(query_emb, top_k=10, conversation_id=conv_id)
    all_segments = list(hop0.segments)
    exclude = {s.index for s in all_segments}

    context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + format_segments(
        all_segments
    )
    prompt = V15_CONTROL_PROMPT.format(
        question=question, context_section=context_section
    )
    output = llm_call(prompt)
    cues = parse_cues(output)

    for cue in cues[:2]:
        cue_emb = embed_text(cue)
        result = store.search(
            cue_emb,
            top_k=10,
            conversation_id=conv_id,
            exclude_indices=exclude,
        )
        for seg in result.segments:
            if seg.index not in exclude:
                all_segments.append(seg)
                exclude.add(seg.index)

    return all_segments


def run_meta_v2f(store: SegmentStore, question: str, conv_id: str) -> list[Segment]:
    """meta_v2f: V2f prompt = v15 + completeness hint + anti-question."""
    query_emb = embed_text(question)
    hop0 = store.search(query_emb, top_k=10, conversation_id=conv_id)
    all_segments = list(hop0.segments)
    exclude = {s.index for s in all_segments}

    context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + format_segments(
        all_segments
    )
    prompt = V2F_PROMPT.format(question=question, context_section=context_section)
    output = llm_call(prompt)
    cues = parse_cues(output)

    for cue in cues[:2]:
        cue_emb = embed_text(cue)
        result = store.search(
            cue_emb,
            top_k=10,
            conversation_id=conv_id,
            exclude_indices=exclude,
        )
        for seg in result.segments:
            if seg.index not in exclude:
                all_segments.append(seg)
                exclude.add(seg.index)

    return all_segments


def run_frontier_v2_iterative(
    store: SegmentStore,
    question: str,
    conv_id: str,
    max_reflects: int = 4,
) -> list[Segment]:
    """frontier_v2_iterative: iterative reflect with 1 gap per round."""
    exclude: set[int] = set()
    all_segments: list[Segment] = []
    reflect_log: list[dict] = []

    query_emb = embed_text(question)
    result = store.search(query_emb, top_k=10, conversation_id=conv_id)
    all_segments.extend(result.segments)
    for s in result.segments:
        exclude.add(s.index)

    for reflect_i in range(max_reflects):
        if len(all_segments) >= 80:
            break

        context = format_segments(all_segments)
        explored_text = ""
        if reflect_log:
            explored = []
            for entry in reflect_log:
                for g in entry.get("gaps", []):
                    explored.append(f"- {g}")
            explored_text = (
                "\n\nALREADY SEARCHED FOR (do NOT repeat these):\n"
                + "\n".join(explored)
            )

        prompt = FRONTIER_REFLECT_PROMPT.format(
            question=question, context=context, explored_text=explored_text
        )
        response = llm_call(prompt)

        gaps = []
        done = False
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    gaps.append(cue)
            elif line.strip().upper() == "DONE":
                done = True

        reflect_log.append({"reflect": reflect_i, "gaps": gaps, "done": done})

        if done or not gaps:
            break

        for gap in gaps[:1]:
            if len(all_segments) >= 80:
                break
            gap_emb = embed_text(gap)
            result = store.search(
                gap_emb,
                top_k=10,
                conversation_id=conv_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

    return all_segments


def run_hybrid_v2f_gencheck(
    store: SegmentStore,
    question: str,
    conv_id: str,
) -> list[Segment]:
    """hybrid_v2f_gencheck: v2f cue generation + Gen-Check gap assessment.

    Flow:
      1. Initial retrieval with raw query (1 embed)
      2. V2f cue generation (1 LLM) -> retrieve 2 cues (2 embed)
      3. Gen-Check assessment of gaps (1 LLM)
      4. If gaps found, retrieve for each (1-2 embed)
    Total: 2 LLM calls, 5-6 embed calls.
    """
    # Step 1: Initial retrieval
    query_emb = embed_text(question)
    hop0 = store.search(query_emb, top_k=10, conversation_id=conv_id)
    all_segments = list(hop0.segments)
    exclude = {s.index for s in all_segments}

    # Step 2: V2f cue generation (1 LLM call)
    context_section = build_context_section(all_segments)
    v2f_prompt = V2F_PROMPT.format(question=question, context_section=context_section)
    v2f_output = llm_call(v2f_prompt)
    cues = parse_cues(v2f_output)

    for cue in cues[:2]:
        cue_emb = embed_text(cue)
        result = store.search(
            cue_emb,
            top_k=10,
            conversation_id=conv_id,
            exclude_indices=exclude,
        )
        for seg in result.segments:
            if seg.index not in exclude:
                all_segments.append(seg)
                exclude.add(seg.index)

    # Step 3: Gen-Check gap assessment (1 LLM call)
    formatted = format_segments(all_segments, max_items=16, max_chars=300)
    gap_prompt = GAP_ASSESSMENT_PROMPT.format(
        question=question, formatted_segments=formatted
    )
    gap_output = llm_call(gap_prompt)
    gaps = parse_gaps(gap_output)

    # Step 4: Retrieve for gaps (if any)
    done = "DONE" in gap_output.upper().split("\n")[-1] if gap_output else True
    if not done and gaps:
        for gap in gaps[:2]:
            gap_emb = embed_text(gap)
            result = store.search(
                gap_emb,
                top_k=10,
                conversation_id=conv_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

    return all_segments


# ===========================================================================
# Architecture registry
# ===========================================================================
ARCHITECTURES = {
    "v15_control": run_v15_control,
    "meta_v2f": run_meta_v2f,
    "frontier_v2_iterative": run_frontier_v2_iterative,
    "hybrid_v2f_gencheck": run_hybrid_v2f_gencheck,
}


# ===========================================================================
# Evaluation
# ===========================================================================
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def deduplicate(segments: list[Segment]) -> list[Segment]:
    seen: set[int] = set()
    deduped: list[Segment] = []
    for seg in segments:
        if seg.index not in seen:
            deduped.append(seg)
            seen.add(seg.index)
    return deduped


def evaluate_one(
    store: SegmentStore,
    arch_name: str,
    arch_fn,
    question: dict,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    t0 = time.time()
    raw_segments = arch_fn(store, q_text, conv_id)
    elapsed = time.time() - t0

    segments = deduplicate(raw_segments)
    total_retrieved = len(segments)

    # Baseline: cosine top-N
    query_emb = embed_text(q_text)
    max_budget = max(BUDGETS + [total_retrieved])
    baseline_result = store.search(query_emb, top_k=max_budget, conversation_id=conv_id)

    baseline_recalls: dict[str, float] = {}
    arch_recalls: dict[str, float] = {}
    for budget in BUDGETS:
        baseline_ids = {s.turn_id for s in baseline_result.segments[:budget]}
        baseline_recalls[f"r@{budget}"] = compute_recall(baseline_ids, source_ids)

        arch_ids = {s.turn_id for s in segments[:budget]}
        arch_recalls[f"r@{budget}"] = compute_recall(arch_ids, source_ids)

    return {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question.get("question_index", -1),
        "question": q_text[:120],
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "baseline_recalls": baseline_recalls,
        "arch_recalls": arch_recalls,
        "total_retrieved": total_retrieved,
        "time_s": round(elapsed, 2),
    }


# ===========================================================================
# Dataset definitions
# ===========================================================================
DATASETS = {
    "puzzle": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "label": "Puzzle 16q",
    },
    "advanced": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "label": "Advanced 23q",
    },
}


# ===========================================================================
# Reporting
# ===========================================================================
def print_dataset_table(
    dataset_name: str,
    all_results: dict[str, list[dict]],
    budget: int = 20,
):
    """Print per-category table for one dataset across all architectures."""
    cat_counts: dict[str, int] = defaultdict(int)
    first_results = next(iter(all_results.values()))
    for r in first_results:
        cat_counts[r["category"]] += 1

    arch_names = list(all_results.keys())
    short_names = {
        "v15_control": "v15",
        "meta_v2f": "v2f",
        "frontier_v2_iterative": "frontier",
        "hybrid_v2f_gencheck": "hybrid",
    }

    # Per-category means
    cat_arch_means: dict[str, dict[str, float]] = {}
    cat_baseline_means: dict[str, float] = {}
    for cat in cat_counts:
        cat_arch_means[cat] = {}
        for arch_name in arch_names:
            results = all_results[arch_name]
            cat_results = [r for r in results if r["category"] == cat]
            if not cat_results:
                cat_arch_means[cat][arch_name] = 0.0
                continue
            vals = [r["arch_recalls"][f"r@{budget}"] for r in cat_results]
            cat_arch_means[cat][arch_name] = sum(vals) / len(vals)

        first_arch = arch_names[0]
        results = all_results[first_arch]
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            vals = [r["baseline_recalls"][f"r@{budget}"] for r in cat_results]
            cat_baseline_means[cat] = sum(vals) / len(vals)
        else:
            cat_baseline_means[cat] = 0.0

    print(f"\n{'=' * 100}")
    print(f"DATASET: {dataset_name} | r@{budget}")
    print(f"{'=' * 100}")

    header = f"{'Category':<28} | {'Baseline':>8}"
    for an in arch_names:
        header += f" | {short_names.get(an, an[:7]):>8}"
    header += " | Best"
    print(header)
    print("-" * len(header))

    overall_arch_sums: dict[str, float] = defaultdict(float)
    overall_baseline_sum = 0.0
    total_q = 0

    for cat in sorted(cat_counts.keys()):
        n = cat_counts[cat]
        bl = cat_baseline_means[cat]
        overall_baseline_sum += bl * n
        total_q += n

        row = f"{cat} ({n}q)"
        row = f"{row:<28} | {bl:>8.2f}"

        best_val = -1.0
        best_arch = ""
        for an in arch_names:
            v = cat_arch_means[cat][an]
            overall_arch_sums[an] += v * n
            row += f" | {v:>8.2f}"
            if v > best_val:
                best_val = v
                best_arch = an

        row += f" | {short_names.get(best_arch, best_arch[:7])}"
        print(row)

    print("-" * len(header))
    bl_avg = overall_baseline_sum / total_q if total_q else 0
    row = f"{'OVERALL':<28} | {bl_avg:>8.2f}"
    for an in arch_names:
        a_avg = overall_arch_sums[an] / total_q if total_q else 0
        row += f" | {a_avg:>8.2f}"
    print(row)
    print()


# ===========================================================================
# Main
# ===========================================================================
def main():
    all_dataset_results: dict[str, dict[str, list[dict]]] = {}

    for ds_name, ds_info in DATASETS.items():
        print(f"\n{'#' * 100}")
        print(f"# LOADING DATASET: {ds_info['label']}")
        print(f"{'#' * 100}")

        store = SegmentStore(DATA_DIR, ds_info["npz"])
        with open(DATA_DIR / ds_info["questions"]) as f:
            questions = json.load(f)

        print(f"  Segments: {len(store.segments)}, Questions: {len(questions)}")

        ds_all_results: dict[str, list[dict]] = {}

        for arch_name, arch_fn in ARCHITECTURES.items():
            out_path = RESULTS_DIR / f"fulleval_{ds_name}_{arch_name}.json"

            # Skip if already done
            if out_path.exists():
                print(
                    f"\n--- Skipping {arch_name} on {ds_info['label']} (exists: {out_path.name}) ---"
                )
                with open(out_path) as f:
                    results = json.load(f)
                ds_all_results[arch_name] = results
                continue

            print(f"\n--- Running {arch_name} on {ds_info['label']} ---")

            results = []
            for i, question in enumerate(questions):
                q_short = question["question"][:55]
                cat = question["category"]
                print(
                    f"  [{i + 1}/{len(questions)}] {cat}: {q_short}...",
                    end="",
                    flush=True,
                )
                try:
                    result = evaluate_one(store, arch_name, arch_fn, question)
                    results.append(result)
                    r20 = result["arch_recalls"]["r@20"]
                    r50 = result["arch_recalls"]["r@50"]
                    bl20 = result["baseline_recalls"]["r@20"]
                    delta = r20 - bl20
                    print(
                        f" r@20={r20:.2f} (bl={bl20:.2f}, d={delta:+.2f})"
                        f" r@50={r50:.2f} [{result['time_s']:.1f}s]"
                    )
                except Exception as e:
                    print(f" ERROR: {e}")
                    import traceback

                    traceback.print_exc()

                if (i + 1) % 5 == 0:
                    save_caches()

            save_caches()

            # Save per-architecture results
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved: {out_path}")

            ds_all_results[arch_name] = results

            # Print quick summary
            if results:
                for budget in BUDGETS:
                    lbl = f"r@{budget}"
                    bl_vals = [r["baseline_recalls"][lbl] for r in results]
                    a_vals = [r["arch_recalls"][lbl] for r in results]
                    bl_mean = sum(bl_vals) / len(bl_vals)
                    a_mean = sum(a_vals) / len(a_vals)
                    print(
                        f"  {arch_name} {lbl}: baseline={bl_mean:.3f} arch={a_mean:.3f} delta={a_mean - bl_mean:+.3f}"
                    )

        all_dataset_results[ds_name] = ds_all_results

        # Print per-dataset table at r@20 and r@50
        if ds_all_results:
            print_dataset_table(ds_info["label"], ds_all_results, budget=20)
            print_dataset_table(ds_info["label"], ds_all_results, budget=50)

    # Final cross-dataset summary
    print(f"\n{'=' * 100}")
    print("FINAL SUMMARY")
    print(f"{'=' * 100}")
    for ds_name, ds_results in all_dataset_results.items():
        print(f"\n--- {ds_name.upper()} ---")
        for arch_name, results in ds_results.items():
            if not results:
                continue
            for budget in BUDGETS:
                lbl = f"r@{budget}"
                bl_vals = [r["baseline_recalls"][lbl] for r in results]
                a_vals = [r["arch_recalls"][lbl] for r in results]
                bl_mean = sum(bl_vals) / len(bl_vals)
                a_mean = sum(a_vals) / len(a_vals)
                print(
                    f"  {arch_name:30s} {lbl}: bl={bl_mean:.3f} arch={a_mean:.3f} delta={a_mean - bl_mean:+.3f}"
                )


if __name__ == "__main__":
    main()
