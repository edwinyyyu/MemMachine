"""Baseline-aware cue generation.

Show the LLM the concrete baseline top-20 segments that cosine already found,
and tell it that the next cue-based search will EXCLUDE those segments.
The model can then focus on generating cues that target content not yet
retrieved.

Hypothesis: Standard v15/v2f shows the top-10 retrieved segments but doesn't
distinguish "found" from "missing". By presenting the full baseline-20 split,
the model sees the concrete gap and can generate cues designed to be
supplementary rather than redundant.

Variants:
    baseline_explicit_v2f    — V2f-style prompt, 2 polished cues
    baseline_explicit_v15    — V15-style prompt, 2 polished cues
    baseline_explicit_dense  — dense keyword bundles (v15 style), 2 cues
    baseline_explicit_multi  — V2f-style, 3-4 cues targeting diverse gaps

Evaluation at K=20 and K=50, in two modes:
    Mode A: return baseline top-20 at K=20 (cues are supplementary)
    Mode B: best 20 from (cues + baseline)  — allow cues to displace baseline

Usage:
    uv run python baseline_aware_cues.py
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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
BUDGETS = [20, 50]
BASELINE_K = 20  # Size of baseline top-K shown to LLM and used as prefix


# ---------------------------------------------------------------------------
# Cache classes — reuse bestshot caches so prior embeddings are hit
# ---------------------------------------------------------------------------
class BaselineAwareEmbeddingCache(EmbeddingCache):
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
            "baseline_aware_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "baseline_aware_embedding_cache.json"
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


class BaselineAwareLLMCache(LLMCache):
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        # Only read the baseline_aware cache — do not mingle with other caches
        # since the prompts here are new/unique anyway.
        for name in ("baseline_aware_llm_cache.json",):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "baseline_aware_llm_cache.json"
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
# Prompts — baseline-explicit variants
# ---------------------------------------------------------------------------

# Variant 1: V2f-style cues (polished, conversational)
BASELINE_EXPLICIT_V2F_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

BASELINE TOP-{baseline_k} (already found by cosine similarity on the question):
{baseline_context}

Your cues will search for ADDITIONAL content beyond these {baseline_k} \
segments. The next search will EXCLUDE these already-found segments.

First, briefly assess: what kinds of content are likely MISSING from the \
baseline? Are there aspects of the question not covered? Are there temporal \
or topical gaps?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues targeting content NOT in the baseline above. \
Use specific vocabulary that would appear in conversation turns about the \
missing aspects.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentences about what baseline missed>
CUE: <text>
CUE: <text>
Nothing else."""


# Variant 2: V15-style cues (no anti-question instruction)
BASELINE_EXPLICIT_V15_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

BASELINE TOP-{baseline_k} (already found by cosine similarity on the question):
{baseline_context}

Your cues will search for ADDITIONAL content beyond these {baseline_k} \
segments. The next search will EXCLUDE these already-found segments.

First, briefly assess: what kinds of content are likely MISSING from the \
baseline? Are there aspects of the question not covered? Are there temporal \
or topical gaps?

Then generate 2 search cues targeting content NOT in the baseline above. \
Use specific vocabulary that would appear in conversation turns about the \
missing aspects.

Format:
ASSESSMENT: <1-2 sentences about what baseline missed>
CUE: <text>
CUE: <text>
Nothing else."""


# Variant 3: Dense keyword bundles (v15-style, emphasizing keyword density)
BASELINE_EXPLICIT_DENSE_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

BASELINE TOP-{baseline_k} (already found by cosine similarity on the question):
{baseline_context}

Your cues will search for ADDITIONAL content beyond these {baseline_k} \
segments. The next search will EXCLUDE these already-found segments.

First, briefly assess: what kinds of content are likely MISSING from the \
baseline? Are there aspects of the question not covered? Are there temporal \
or topical gaps?

Then generate 2 search cues as DENSE KEYWORD BUNDLES — short collections of \
specific nouns, named entities, numbers, and distinctive phrases that would \
appear in conversation turns about the missing aspects. Do NOT write polished \
sentences; pack vocabulary. Each cue should be under 100 characters.

Format:
ASSESSMENT: <1-2 sentences about what baseline missed>
CUE: <dense keyword bundle>
CUE: <dense keyword bundle>
Nothing else."""


# Variant 4: V2f-style with 3-4 cues targeting diverse missing aspects
BASELINE_EXPLICIT_MULTI_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

BASELINE TOP-{baseline_k} (already found by cosine similarity on the question):
{baseline_context}

Your cues will search for ADDITIONAL content beyond these {baseline_k} \
segments. The next search will EXCLUDE these already-found segments.

First, briefly assess: what kinds of content are likely MISSING from the \
baseline? Are there aspects of the question not covered? Are there temporal \
or topical gaps? List 3-4 DISTINCT missing aspects.

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 3-4 search cues, EACH targeting a DIFFERENT missing aspect \
you identified. Use specific vocabulary that would appear in conversation \
turns about those missing aspects.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <list distinct missing aspects>
CUE: <text for aspect 1>
CUE: <text for aspect 2>
CUE: <text for aspect 3>
CUE: <text for aspect 4 (optional)>
Nothing else."""


VARIANT_CONFIG = {
    "baseline_explicit_v2f": {
        "prompt": BASELINE_EXPLICIT_V2F_PROMPT,
        "max_cues": 2,
    },
    "baseline_explicit_v15": {
        "prompt": BASELINE_EXPLICIT_V15_PROMPT,
        "max_cues": 2,
    },
    "baseline_explicit_dense": {
        "prompt": BASELINE_EXPLICIT_DENSE_PROMPT,
        "max_cues": 2,
    },
    "baseline_explicit_multi": {
        "prompt": BASELINE_EXPLICIT_MULTI_PROMPT,
        "max_cues": 4,
    },
}


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------
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
# Helpers
# ---------------------------------------------------------------------------
def _format_segments(segments: list[Segment], max_chars: int = 220) -> str:
    """Format segments chronologically for LLM context."""
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def _parse_cues(response: str) -> list[str]:
    cues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
@dataclass
class BaselineAwareResult:
    baseline_segments: list[Segment]  # cosine top-BASELINE_K
    cue_segments: list[Segment]  # cue-found, in order (excluding baseline)
    assessment: str
    cues: list[str]
    llm_output: str


class BaselineAwareArch:
    """Run baseline retrieval of BASELINE_K segments, then LLM generates cues
    that explicitly avoid baseline content; we search with those cues excluding
    the baseline segments."""

    def __init__(self, store: SegmentStore, variant: str, client: OpenAI | None = None):
        assert variant in VARIANT_CONFIG, f"unknown variant: {variant}"
        self.store = store
        self.variant = variant
        self.prompt_template = VARIANT_CONFIG[variant]["prompt"]
        self.max_cues = VARIANT_CONFIG[variant]["max_cues"]
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = BaselineAwareEmbeddingCache()
        self.llm_cache = BaselineAwareLLMCache()
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
        emb = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, emb)
        self.embed_calls += 1
        return emb

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

    def retrieve(self, question: str, conversation_id: str) -> BaselineAwareResult:
        # Baseline retrieval: cosine top-BASELINE_K on the question
        query_emb = self.embed_text(question)
        baseline = self.store.search(
            query_emb, top_k=BASELINE_K, conversation_id=conversation_id
        )
        baseline_segs = list(baseline.segments)
        baseline_indices = {s.index for s in baseline_segs}

        # Format baseline context for LLM
        baseline_context = _format_segments(baseline_segs)

        # LLM call
        prompt = self.prompt_template.format(
            question=question,
            baseline_k=BASELINE_K,
            baseline_context=baseline_context,
        )
        output = self.llm_call(prompt)

        assessment = ""
        for line in output.strip().split("\n"):
            s = line.strip()
            if s.upper().startswith("ASSESSMENT:"):
                assessment = s[11:].strip()
                break
        cues = _parse_cues(output)[: self.max_cues]

        # Run each cue, excluding baseline segments (explicit split)
        exclude = set(baseline_indices)
        cue_segs: list[Segment] = []
        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    cue_segs.append(seg)
                    exclude.add(seg.index)

        return BaselineAwareResult(
            baseline_segments=baseline_segs,
            cue_segments=cue_segs,
            assessment=assessment,
            cues=cues,
            llm_output=output,
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def mode_a_at_k(
    baseline_segs: list[Segment],
    cue_segs: list[Segment],
    k: int,
) -> list[Segment]:
    """Mode A: baseline-first, then cue-found. Cues supplement baseline."""
    seen: set[int] = set()
    out: list[Segment] = []
    for s in baseline_segs:
        if s.index not in seen:
            out.append(s)
            seen.add(s.index)
        if len(out) >= k:
            return out[:k]
    for s in cue_segs:
        if s.index not in seen:
            out.append(s)
            seen.add(s.index)
        if len(out) >= k:
            break
    return out[:k]


def mode_b_at_k(
    baseline_segs: list[Segment],
    cue_segs: list[Segment],
    k: int,
) -> list[Segment]:
    """Mode B: cue-found first, then baseline. Cues displace baseline if k
    is too small to hold everything."""
    seen: set[int] = set()
    out: list[Segment] = []
    for s in cue_segs:
        if s.index not in seen:
            out.append(s)
            seen.add(s.index)
        if len(out) >= k:
            return out[:k]
    for s in baseline_segs:
        if s.index not in seen:
            out.append(s)
            seen.add(s.index)
        if len(out) >= k:
            break
    return out[:k]


def evaluate_question(
    arch: BaselineAwareArch,
    question: dict,
    max_k: int,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Pure cosine top-max_k for baseline_r@K comparison at each K
    query_emb = arch.embed_text(q_text)
    cosine_full = arch.store.search(query_emb, top_k=max_k, conversation_id=conv_id)
    cosine_segs = list(cosine_full.segments)

    row = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "num_baseline_segs": len(result.baseline_segments),
        "num_cue_segs": len(result.cue_segments),
        "cues": result.cues,
        "assessment": result.assessment,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "recalls": {},
    }

    for k in BUDGETS:
        baseline_at_k = cosine_segs[:k]
        mode_a = mode_a_at_k(result.baseline_segments, result.cue_segments, k)
        mode_b = mode_b_at_k(result.baseline_segments, result.cue_segments, k)

        b_turns = {s.turn_id for s in baseline_at_k}
        a_turns = {s.turn_id for s in mode_a}
        bb_turns = {s.turn_id for s in mode_b}

        row["recalls"][f"baseline_r@{k}"] = round(
            compute_recall(b_turns, source_ids), 4
        )
        row["recalls"][f"modeA_r@{k}"] = round(compute_recall(a_turns, source_ids), 4)
        row["recalls"][f"modeB_r@{k}"] = round(compute_recall(bb_turns, source_ids), 4)
        row["recalls"][f"modeA_delta@{k}"] = round(
            row["recalls"][f"modeA_r@{k}"] - row["recalls"][f"baseline_r@{k}"], 4
        )
        row["recalls"][f"modeB_delta@{k}"] = round(
            row["recalls"][f"modeB_r@{k}"] - row["recalls"][f"baseline_r@{k}"], 4
        )

    return row


def summarize(results: list[dict], variant: str, dataset: str) -> dict:
    n = len(results)
    summary: dict = {"variant": variant, "dataset": dataset, "n": n}
    if n == 0:
        return summary

    for k in BUDGETS:
        for key in (f"baseline_r@{k}", f"modeA_r@{k}", f"modeB_r@{k}"):
            vals = [r["recalls"][key] for r in results]
            summary[key] = round(sum(vals) / n, 4)
        for side in ("modeA", "modeB"):
            deltas = [r["recalls"][f"{side}_delta@{k}"] for r in results]
            wins = sum(1 for d in deltas if d > 0.001)
            losses = sum(1 for d in deltas if d < -0.001)
            ties = n - wins - losses
            summary[f"{side}_delta@{k}"] = round(sum(deltas) / n, 4)
            summary[f"{side}_W/T/L@{k}"] = f"{wins}/{ties}/{losses}"

    summary["avg_baseline_segs"] = round(
        sum(r["num_baseline_segs"] for r in results) / n, 1
    )
    summary["avg_cue_segs"] = round(sum(r["num_cue_segs"] for r in results) / n, 1)
    summary["avg_cues_per_q"] = round(sum(len(r["cues"]) for r in results) / n, 2)
    summary["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
    summary["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 1)
    summary["avg_time_s"] = round(sum(r["time_s"] for r in results) / n, 2)
    return summary


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    out = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry = {"n": n}
        for k in BUDGETS:
            for key in (f"baseline_r@{k}", f"modeA_r@{k}", f"modeB_r@{k}"):
                vals = [r["recalls"][key] for r in rs]
                entry[key] = round(sum(vals) / n, 4)
            for side in ("modeA", "modeB"):
                deltas = [r["recalls"][f"{side}_delta@{k}"] for r in rs]
                wins = sum(1 for d in deltas if d > 0.001)
                losses = sum(1 for d in deltas if d < -0.001)
                ties = n - wins - losses
                entry[f"{side}_delta@{k}"] = round(sum(deltas) / n, 4)
                entry[f"{side}_W/T/L@{k}"] = f"{wins}/{ties}/{losses}"
        out[cat] = entry
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
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


def run_one(
    variant: str,
    dataset: str,
    store: SegmentStore,
    questions: list[dict],
) -> tuple[list[dict], dict, dict]:
    print(f"\n{'=' * 74}")
    print(f"{variant} | {dataset} | {len(questions)} questions")
    print(f"{'=' * 74}", flush=True)

    arch = BaselineAwareArch(store, variant=variant)

    results = []
    for i, q in enumerate(questions):
        q_short = q["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(arch, q, max_k=max(BUDGETS))
            results.append(row)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = summarize(results, variant, dataset)
    by_cat = summarize_by_category(results)

    print(f"\n--- {variant} on {dataset} ---")
    for k in BUDGETS:
        print(
            f"  r@{k}: baseline={summary[f'baseline_r@{k}']:.3f}  "
            f"modeA={summary[f'modeA_r@{k}']:.3f} "
            f"(d={summary[f'modeA_delta@{k}']:+.3f} "
            f"W/T/L={summary[f'modeA_W/T/L@{k}']})  "
            f"modeB={summary[f'modeB_r@{k}']:.3f} "
            f"(d={summary[f'modeB_delta@{k}']:+.3f} "
            f"W/T/L={summary[f'modeB_W/T/L@{k}']})"
        )
    print(
        f"  avg baseline_segs={summary['avg_baseline_segs']}  "
        f"cue_segs={summary['avg_cue_segs']}  "
        f"cues/q={summary['avg_cues_per_q']}  "
        f"llm={summary['avg_llm_calls']}  "
        f"embed={summary['avg_embed_calls']}"
    )

    return results, summary, by_cat


def spot_check(results: list[dict], variant: str, n: int = 5) -> None:
    print(f"\n--- Spot-check cues: {variant} ---")
    rng = random.Random(42)
    sample = rng.sample(results, min(n, len(results)))
    for r in sample:
        print(f"  Q: {r['question'][:90]}")
        print(f"    assessment: {r['assessment'][:120]}")
        for cue in r["cues"]:
            print(f"    CUE: {cue[:140]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Run a single variant (default: all 4)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Subset of datasets to run (default: all 4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing result files",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variants = [args.variant] if args.variant else list(VARIANT_CONFIG.keys())
    datasets = args.datasets or list(DATASETS.keys())

    all_summaries: dict = {}

    for ds_name in datasets:
        if ds_name not in DATASETS:
            print(f"Unknown dataset {ds_name}, skipping")
            continue
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        for variant in variants:
            out_path = RESULTS_DIR / f"baseline_aware_{variant}_{ds_name}.json"
            if out_path.exists() and not args.force:
                print(
                    f"\nSkipping {variant} on {ds_name} (exists). Use --force to rerun."
                )
                with open(out_path) as f:
                    saved = json.load(f)
                results = saved["results"]
                summary = saved["summary"]
                by_cat = saved.get("category_breakdown", {})
            else:
                results, summary, by_cat = run_one(variant, ds_name, store, questions)
                with open(out_path, "w") as f:
                    json.dump(
                        {
                            "variant": variant,
                            "dataset": ds_name,
                            "summary": summary,
                            "category_breakdown": by_cat,
                            "results": results,
                        },
                        f,
                        indent=2,
                        default=str,
                    )
                print(f"  Saved: {out_path}")

                # Spot-check cues
                spot_check(results, variant, n=5)

            all_summaries.setdefault(variant, {})[ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
            }

    # Aggregated summary
    summary_path = RESULTS_DIR / "baseline_aware_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved aggregate summary: {summary_path}")

    # Final table
    print("\n" + "=" * 110)
    print("BASELINE-AWARE CUE GENERATION SUMMARY")
    print("=" * 110)
    header = (
        f"{'Variant':<26s} {'Dataset':<14s} "
        f"{'base@20':>8s} {'A@20':>7s} {'dA@20':>7s} "
        f"{'B@20':>7s} {'dB@20':>7s} "
        f"{'base@50':>8s} {'A@50':>7s} {'dA@50':>7s} "
        f"{'B@50':>7s} {'dB@50':>7s}"
    )
    print(header)
    print("-" * len(header))
    for variant in variants:
        for ds_name in datasets:
            if ds_name not in all_summaries.get(variant, {}):
                continue
            s = all_summaries[variant][ds_name]["summary"]
            print(
                f"{variant:<26s} {ds_name:<14s} "
                f"{s['baseline_r@20']:>8.3f} "
                f"{s['modeA_r@20']:>7.3f} {s['modeA_delta@20']:>+7.3f} "
                f"{s['modeB_r@20']:>7.3f} {s['modeB_delta@20']:>+7.3f} "
                f"{s['baseline_r@50']:>8.3f} "
                f"{s['modeA_r@50']:>7.3f} {s['modeA_delta@50']:>+7.3f} "
                f"{s['modeB_r@50']:>7.3f} {s['modeB_delta@50']:>+7.3f}"
            )


if __name__ == "__main__":
    main()
