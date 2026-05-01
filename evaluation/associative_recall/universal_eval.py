"""Universal prompt evaluation: test cue prompts across ALL datasets.

Measures delta r@20 and delta r@50 using the UNION approach:
  arch_recall = recall(baseline_top_k UNION cue_segments, source_ids)
This ensures cues can only ADD to baseline, never displace.

Also measures STANDALONE performance to detect cue quality.

Usage:
    uv run python universal_eval.py --variant <name> [--force] [--verbose]
    uv run python universal_eval.py --all
    uv run python universal_eval.py --list
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

CACHE_FILE_LLM = CACHE_DIR / "universal_llm_cache.json"
CACHE_FILE_EMB = CACHE_DIR / "universal_embedding_cache.json"


# ---------------------------------------------------------------------------
# Cache classes
# ---------------------------------------------------------------------------
class UniversalEmbeddingCache(EmbeddingCache):
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
            "universal_embedding_cache.json",
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


class UniversalLLMCache(LLMCache):
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
            "universal_llm_cache.json",
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
# Shared infrastructure
# ---------------------------------------------------------------------------
embedding_cache = UniversalEmbeddingCache()
llm_cache = UniversalLLMCache()
client = OpenAI(timeout=60.0)


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
        max_completion_tokens=2000,
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


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

# V15_CONTROL: exact v15 prompt (reference)
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

# V2F: v15 + completeness + anti-question (current best on LoCoMo)
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

# V_UNIVERSAL: domain-agnostic, anti-paraphrase, vocabulary-diversity focused
V_UNIVERSAL_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity against stored text segments.

Question: {question}

{context_section}

First, assess: What content relevant to the question is still MISSING from \
what was retrieved? What DIFFERENT vocabulary would that missing content use?

CRITICAL: The initial search already found content matching the question's \
own vocabulary. Your cues must use DIFFERENT words to reach content the \
initial search missed. Do not paraphrase the question.

Generate 2 search cues. Each cue should:
- Use vocabulary that would appear in the TARGET content, not the question
- Target a different aspect of the question than the other cue
- Be a short phrase or sentence (not a question, not a meta-instruction)

Format:
ASSESSMENT: <1-2 sentences on what's missing and what vocabulary to target>
CUE: <text using target-content vocabulary>
CUE: <text using different target-content vocabulary>
Nothing else."""

# V_ANTI_PARAPHRASE: v15 structure + strong anti-paraphrase instruction
V_ANTI_PARAPHRASE_PROMPT = """\
You are generating search text for semantic retrieval over stored content. \
Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

CRITICAL: Do NOT generate cues that share vocabulary with the original \
question. The baseline search already found content matching the question's \
vocabulary. Your cues must use DIFFERENT words to find content the baseline \
missed.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target content — words and phrases \
from the ANSWER, not from the question.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V_VOCAB_EXTRACT: extract terms from retrieved text, use them in cues
V_VOCAB_EXTRACT_PROMPT = """\
You are generating search text for semantic retrieval over stored content. \
Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

Step 1: Extract 3-5 specific terms (names, numbers, technical words, \
unique phrases) from the retrieved text that are NOT in the original \
question. These are anchor terms.

Step 2: Generate 2 search cues using those anchor terms combined with \
unexplored aspects of the question. Each cue should reach content that \
is related to the retrieved text but covers DIFFERENT ground.

Do NOT paraphrase the question. Do NOT write meta-instructions. Write \
text that would appear in the source content.

Format:
TERMS: <comma-separated extracted terms>
ASSESSMENT: <what's missing>
CUE: <text using extracted terms + new direction>
CUE: <text using different extracted terms + different direction>
Nothing else."""

# V_DIVERSE_ANGLES: force maximum diversity between cues
V_DIVERSE_ANGLES_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity against stored text segments.

Question: {question}

{context_section}

Generate 2 cues that target completely DIFFERENT aspects of the question.

Rules:
- Cue 1 and Cue 2 must share NO significant vocabulary with each other
- Neither cue should share significant vocabulary with the original question
- Each cue should find content that the other cue would miss
- Write text that would appear in the source content, not questions or \
meta-instructions
- Use specific words: names, numbers, technical terms, action verbs

Format:
ASSESSMENT: <what 2 different aspects to target>
CUE: <text targeting aspect 1>
CUE: <text targeting aspect 2, using completely different words>
Nothing else."""

# V_MINIMAL_SAFE: ultra-minimal prompt that's unlikely to generate noise
V_MINIMAL_SAFE_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

What specific content is MISSING? Generate 2 short search phrases (under \
15 words each) using vocabulary from the retrieved text, not from the \
question. Target different missing topics.

Format:
ASSESSMENT: <what's missing>
CUE: <short phrase>
CUE: <short phrase>
Nothing else."""

# V_CONDITIONAL: only generate cues if there's actually a gap
V_CONDITIONAL_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, assess: Does the retrieved content already cover the question well? \
If so, generating more cues would just add noise.

If the retrieval looks COMPLETE or nearly complete, respond with:
ASSESSMENT: Retrieval looks sufficient.
DONE

If important content is clearly MISSING, generate cues to find it:
ASSESSMENT: <what's missing>
CUE: <text using target-content vocabulary, not question vocabulary>
CUE: <text targeting a different missing aspect>
Nothing else."""


# ---------------------------------------------------------------------------
# Architecture: single LLM call, matches v15_control logic
# ---------------------------------------------------------------------------
def run_variant(
    store: SegmentStore,
    question: str,
    conv_id: str,
    prompt_template: str,
) -> tuple[list[Segment], list[Segment], dict]:
    """Run v15-style architecture: top-10 + 1 LLM call producing 2 cues.

    Returns: (baseline_10, cue_segments, metadata)
    """
    # Hop 0: embed question, retrieve top-10
    query_emb = embed_text(question)
    hop0 = store.search(query_emb, top_k=10, conversation_id=conv_id)
    baseline_segments = list(hop0.segments)
    exclude = {s.index for s in baseline_segments}

    context_section = "RETRIEVED CONTENT SO FAR:\n" + format_segments(baseline_segments)
    prompt = prompt_template.format(question=question, context_section=context_section)
    output = llm_call(prompt)

    # Check for DONE
    done = False
    for line in output.strip().split("\n"):
        if line.strip().upper() == "DONE":
            done = True

    cue_segments = []
    cues = parse_cues(output)

    if not done:
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
                    cue_segments.append(seg)
                    exclude.add(seg.index)

    metadata = {"output": output, "cues": cues[:2], "done": done}
    return baseline_segments, cue_segments, metadata


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    store: SegmentStore,
    variant_name: str,
    prompt_template: str,
    question: dict,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    t0 = time.time()
    baseline_segs, cue_segs, metadata = run_variant(
        store, q_text, conv_id, prompt_template
    )
    elapsed = time.time() - t0

    # Baseline at various budgets: cosine top-N
    query_emb = embed_text(q_text)
    max_budget = max(BUDGETS)
    baseline_full = store.search(query_emb, top_k=max_budget, conversation_id=conv_id)

    baseline_recalls: dict[str, float] = {}
    union_recalls: dict[str, float] = {}
    standalone_recalls: dict[str, float] = {}

    cue_turn_ids = {s.turn_id for s in cue_segs}

    for budget in BUDGETS:
        # Baseline: cosine top-N
        baseline_ids = {s.turn_id for s in baseline_full.segments[:budget]}
        baseline_recalls[f"r@{budget}"] = compute_recall(baseline_ids, source_ids)

        # Union: baseline top-N + all cue segments (cues can only ADD)
        union_ids = baseline_ids | cue_turn_ids
        union_recalls[f"r@{budget}"] = compute_recall(union_ids, source_ids)

        # Standalone: first N from (baseline_segs + cue_segs)
        all_arch = baseline_segs + cue_segs
        standalone_ids = {s.turn_id for s in all_arch[:budget]}
        standalone_recalls[f"r@{budget}"] = compute_recall(standalone_ids, source_ids)

    return {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question.get("question_index", -1),
        "question": q_text[:120],
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "baseline_recalls": baseline_recalls,
        "union_recalls": union_recalls,
        "standalone_recalls": standalone_recalls,
        "num_cue_segments": len(cue_segs),
        "cue_hits": len(cue_turn_ids & source_ids),
        "done": metadata.get("done", False),
        "cues": metadata.get("cues", []),
        "time_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS = {
    "locomo": {
        "npz": "segments_extended.npz",
        "questions": "questions_extended.json",
        "label": "LoCoMo 30q",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "max_q": 30,
    },
    "synthetic": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "label": "Synthetic 19q",
        "filter": None,
        "max_q": None,
    },
    "puzzle": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "label": "Puzzle 16q",
        "filter": None,
        "max_q": None,
    },
    "advanced": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "label": "Advanced 23q",
        "filter": None,
        "max_q": None,
    },
}


VARIANTS = {
    "v15_control": V15_CONTROL_PROMPT,
    "v2f": V2F_PROMPT,
    "v_universal": V_UNIVERSAL_PROMPT,
    "v_anti_paraphrase": V_ANTI_PARAPHRASE_PROMPT,
    "v_vocab_extract": V_VOCAB_EXTRACT_PROMPT,
    "v_diverse_angles": V_DIVERSE_ANGLES_PROMPT,
    "v_minimal_safe": V_MINIMAL_SAFE_PROMPT,
    "v_conditional": V_CONDITIONAL_PROMPT,
}


# ---------------------------------------------------------------------------
# Summary and reporting
# ---------------------------------------------------------------------------
def summarize_results(
    results: list[dict],
    variant_name: str,
    dataset_name: str,
) -> dict:
    n = len(results)
    if n == 0:
        return {}

    summary: dict = {
        "variant": variant_name,
        "dataset": dataset_name,
        "n": n,
    }

    for budget in BUDGETS:
        lbl = f"r@{budget}"
        b_vals = [r["baseline_recalls"][lbl] for r in results]
        u_vals = [r["union_recalls"][lbl] for r in results]
        s_vals = [r["standalone_recalls"][lbl] for r in results]

        b_mean = sum(b_vals) / n
        u_mean = sum(u_vals) / n
        s_mean = sum(s_vals) / n

        u_wins = sum(1 for b, u in zip(b_vals, u_vals) if u > b + 0.001)
        u_losses = sum(1 for b, u in zip(b_vals, u_vals) if b > u + 0.001)
        s_wins = sum(1 for b, s in zip(b_vals, s_vals) if s > b + 0.001)
        s_losses = sum(1 for b, s in zip(b_vals, s_vals) if b > s + 0.001)

        summary[f"baseline_{lbl}"] = round(b_mean, 4)
        summary[f"union_{lbl}"] = round(u_mean, 4)
        summary[f"union_delta_{lbl}"] = round(u_mean - b_mean, 4)
        summary[f"union_W/L_{lbl}"] = f"{u_wins}/{u_losses}"
        summary[f"standalone_{lbl}"] = round(s_mean, 4)
        summary[f"standalone_delta_{lbl}"] = round(s_mean - b_mean, 4)
        summary[f"standalone_W/L_{lbl}"] = f"{s_wins}/{s_losses}"

    summary["avg_cue_hits"] = round(sum(r["cue_hits"] for r in results) / n, 2)
    summary["avg_cue_segments"] = round(
        sum(r["num_cue_segments"] for r in results) / n, 1
    )
    summary["pct_done"] = round(
        sum(1 for r in results if r.get("done", False)) / n * 100, 1
    )

    return summary


def print_grand_summary(all_summaries: list[dict]):
    """Print a compact summary table across all variants and datasets."""
    print(f"\n{'=' * 130}")
    print("GRAND SUMMARY: Union delta (cues can only ADD)")
    print(f"{'=' * 130}")

    # Group by variant
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for s in all_summaries:
        by_variant[s["variant"]].append(s)

    header = (
        f"{'Variant':<22} | "
        f"{'LoCoMo':>14} | "
        f"{'Synthetic':>14} | "
        f"{'Puzzle':>14} | "
        f"{'Advanced':>14} | "
        f"{'Avg':>8}"
    )
    print(header)
    print(
        f"{'':22} | {'r@20    r@50':>14} | {'r@20    r@50':>14} | {'r@20    r@50':>14} | {'r@20    r@50':>14} |"
    )
    print("-" * 130)

    for variant in VARIANTS:
        summaries = by_variant.get(variant, [])
        if not summaries:
            continue

        ds_deltas: dict[str, dict[str, float]] = {}
        for s in summaries:
            ds_deltas[s["dataset"]] = {
                "r@20": s.get("union_delta_r@20", 0),
                "r@50": s.get("union_delta_r@50", 0),
            }

        all_deltas_20 = []
        all_deltas_50 = []
        parts = []
        for ds in ["locomo", "synthetic", "puzzle", "advanced"]:
            d = ds_deltas.get(ds, {"r@20": 0, "r@50": 0})
            d20 = d["r@20"]
            d50 = d["r@50"]
            all_deltas_20.append(d20)
            all_deltas_50.append(d50)
            parts.append(f"{d20:+.3f} {d50:+.3f}")

        avg_d20 = sum(all_deltas_20) / len(all_deltas_20) if all_deltas_20 else 0
        avg_d50 = sum(all_deltas_50) / len(all_deltas_50) if all_deltas_50 else 0

        # Check for any negative
        any_neg_20 = any(d < -0.005 for d in all_deltas_20)
        any_neg_50 = any(d < -0.005 for d in all_deltas_50)
        flag = " <--" if not any_neg_20 and not any_neg_50 else ""

        row = f"{variant:<22} | "
        row += " | ".join(f"{p:>14}" for p in parts)
        row += f" | {avg_d20:+.3f}{flag}"
        print(row)

    print("-" * 130)
    print("Note: Union deltas should always be >= 0. Negative = bug in eval.")
    print("Flag '<--' marks variants with NO regressions on any dataset.")


def print_standalone_summary(all_summaries: list[dict]):
    """Print standalone (displacement) summary."""
    print(f"\n{'=' * 130}")
    print("STANDALONE SUMMARY: arch segments compete directly with baseline")
    print(f"{'=' * 130}")

    by_variant: dict[str, list[dict]] = defaultdict(list)
    for s in all_summaries:
        by_variant[s["variant"]].append(s)

    header = (
        f"{'Variant':<22} | "
        f"{'LoCoMo':>14} | "
        f"{'Synthetic':>14} | "
        f"{'Puzzle':>14} | "
        f"{'Advanced':>14}"
    )
    print(header)
    print(
        f"{'':22} | {'r@20    r@50':>14} | {'r@20    r@50':>14} | {'r@20    r@50':>14} | {'r@20    r@50':>14}"
    )
    print("-" * 130)

    for variant in VARIANTS:
        summaries = by_variant.get(variant, [])
        if not summaries:
            continue

        parts = []
        for ds in ["locomo", "synthetic", "puzzle", "advanced"]:
            match = [s for s in summaries if s["dataset"] == ds]
            if match:
                d20 = match[0].get("standalone_delta_r@20", 0)
                d50 = match[0].get("standalone_delta_r@50", 0)
                parts.append(f"{d20:+.3f} {d50:+.3f}")
            else:
                parts.append(f"{'N/A':>14}")

        row = f"{variant:<22} | "
        row += " | ".join(f"{p:>14}" for p in parts)
        print(row)


def spot_check_cues(results: list[dict], variant_name: str, n: int = 5):
    """Print cues from N randomly-chosen questions for quality check."""
    print(f"\n--- Spot-check cues for {variant_name} ---")
    import random

    sample = random.sample(results, min(n, len(results)))
    for r in sample:
        q = r["question"]
        cues = r.get("cues", [])
        done = r.get("done", False)
        delta20 = r["union_recalls"].get("r@20", 0) - r["baseline_recalls"].get(
            "r@20", 0
        )
        hits = r.get("cue_hits", 0)
        print(f"  Q: {q[:80]}...")
        if done:
            print("    -> DONE (no cues generated)")
        for cue in cues:
            print(f"    Cue: {cue[:120]}")
        print(f"    delta_r@20={delta20:+.3f}, cue_hits={hits}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal prompt evaluation across all datasets"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Run specific variant",
    )
    parser.add_argument("--all", action="store_true", help="Run all variants")
    parser.add_argument("--list", action="store_true", help="List variants")
    parser.add_argument("--force", action="store_true", help="Overwrite existing")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to run (default: all)",
    )
    parser.add_argument(
        "--spot-check",
        type=int,
        default=3,
        help="Number of questions to spot-check per variant-dataset",
    )
    args = parser.parse_args()

    if args.list:
        print("Available variants:")
        for name in VARIANTS:
            print(f"  {name}")
        print("\nAvailable datasets:")
        for name in DATASETS:
            print(f"  {name}")
        return

    # Determine variants and datasets
    if args.variant:
        variant_names = [args.variant]
    elif args.all:
        variant_names = list(VARIANTS.keys())
    else:
        # Default: run new variants only
        variant_names = [
            "v_universal",
            "v_anti_paraphrase",
            "v_vocab_extract",
            "v_diverse_angles",
            "v_minimal_safe",
            "v_conditional",
        ]

    dataset_names = args.datasets or list(DATASETS.keys())

    all_summaries: list[dict] = []

    for ds_name in dataset_names:
        ds_info = DATASETS[ds_name]
        print(f"\n{'#' * 100}")
        print(f"# LOADING DATASET: {ds_info['label']}")
        print(f"{'#' * 100}")

        store = SegmentStore(DATA_DIR, ds_info["npz"])
        with open(DATA_DIR / ds_info["questions"]) as f:
            questions = json.load(f)

        if ds_info.get("filter"):
            questions = [q for q in questions if ds_info["filter"](q)]
        if ds_info.get("max_q"):
            questions = questions[: ds_info["max_q"]]

        print(f"  Segments: {len(store.segments)}, Questions: {len(questions)}")

        for variant_name in variant_names:
            if variant_name not in VARIANTS:
                print(f"Unknown variant: {variant_name}")
                continue

            prompt_template = VARIANTS[variant_name]
            results_file = RESULTS_DIR / f"universal_{variant_name}_{ds_name}.json"

            if results_file.exists() and not args.force:
                print(f"\nSkipping {variant_name} on {ds_name} (exists)")
                with open(results_file) as f:
                    results = json.load(f)
                summary = summarize_results(results, variant_name, ds_name)
                all_summaries.append(summary)
                d20 = summary.get("union_delta_r@20", 0)
                d50 = summary.get("union_delta_r@50", 0)
                print(f"  union delta: r@20={d20:+.3f} r@50={d50:+.3f}")
                continue

            print(f"\n--- Running {variant_name} on {ds_info['label']} ---")

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
                    result = evaluate_one(
                        store, variant_name, prompt_template, question
                    )
                    results.append(result)
                    u20 = result["union_recalls"]["r@20"]
                    bl20 = result["baseline_recalls"]["r@20"]
                    d20 = u20 - bl20
                    hits = result["cue_hits"]
                    done = result.get("done", False)
                    status = "DONE" if done else f"hits={hits}"
                    print(f" delta_r@20={d20:+.2f} [{status}]")
                except Exception as e:
                    print(f" ERROR: {e}")
                    import traceback

                    traceback.print_exc()

                if (i + 1) % 5 == 0:
                    save_caches()

            save_caches()

            # Save results
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved: {results_file}")

            summary = summarize_results(results, variant_name, ds_name)
            all_summaries.append(summary)

            d20 = summary.get("union_delta_r@20", 0)
            d50 = summary.get("union_delta_r@50", 0)
            print(f"  union: r@20={d20:+.3f} r@50={d50:+.3f}")

            # Spot check
            if args.spot_check > 0 and results:
                spot_check_cues(results, f"{variant_name}/{ds_name}", n=args.spot_check)

    # Grand summary
    if len(all_summaries) > 1:
        print_grand_summary(all_summaries)
        print_standalone_summary(all_summaries)

    # Save all summaries
    summary_file = RESULTS_DIR / "universal_all_summaries.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved all summaries: {summary_file}")


if __name__ == "__main__":
    main()
