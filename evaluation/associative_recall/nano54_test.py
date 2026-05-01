"""Test whether gpt-5.4-nano can replace gpt-5-mini for v2f / v2f_plus_types.

Mirrors nano_test.py (which targets gpt-5-nano), but:
  - Uses MODEL = gpt-5.4-nano for every LLM call we care about.
  - Reuses existing mini baselines (read from fairbackfill_meta_v2f_*.json
    and type_enum_v2f_plus_types_*.json) — no fresh mini calls.
  - Separate LLM cache: cache/nano54_llm_cache.json.
  - Results: results/nano54_*.json.

Discipline:
  1. Quick-test on 5 hand-picked questions (2 LoCoMo, 1 synthetic
     completeness, 1 puzzle logic_constraint, 1 advanced evolving_terminology).
     Gate: if nano54 is within 5pp on most, continue to full eval; abandon if
     meaningfully worse on most.
  2. If gate passes: full eval v2f and v2f_plus_types on all 4 datasets,
     fair-backfilled at K=20 and K=50, compared to the mini baseline files.

Usage:
    uv run python nano54_test.py                 # quick-test only
    uv run python nano54_test.py --full          # quick-test + full eval
    uv run python nano54_test.py --full --force  # overwrite cached outputs
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    LLMCache,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI
from prompt_optimization import META_V2F_PROMPT, _format_segments
from type_enumerated import (
    BUDGETS,
    DATASETS,
    TYPE_ENUMERATED_PROMPT,
    TypeEnumEmbeddingCache,
    _build_context_section,
    _parse_cues,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MINI_MODEL = "gpt-5-mini"  # baseline (not called — read from files)
NANO54_MODEL = "gpt-5.4-nano"

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_FILE_NANO54_LLM = CACHE_DIR / "nano54_llm_cache.json"


# ---------------------------------------------------------------------------
# nano5.4 LLM cache — reads all existing caches (harmless; keyed by
# model+prompt) and writes nano5.4 entries to a dedicated file.
# ---------------------------------------------------------------------------
class Nano54LLMCache(LLMCache):
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
            "constraint_llm_cache.json",
            "type_enum_llm_cache.json",
            "nano_llm_cache.json",
            "nano54_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = CACHE_FILE_NANO54_LLM
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
# Generic variant base with configurable model
# ---------------------------------------------------------------------------
@dataclass
class Nano54Result:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


class ModelAwareBase:
    """Configurable model name, same retrieval logic as the production v2f /
    v2f_plus_types architectures."""

    def __init__(
        self,
        store: SegmentStore,
        model: str,
        client: OpenAI | None = None,
    ):
        self.store = store
        self.model = model
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = TypeEnumEmbeddingCache()  # shared
        self.llm_cache = Nano54LLMCache()  # reads all, writes nano54-only
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

    def llm_call(self, prompt: str) -> str:
        cached = self.llm_cache.get(self.model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=3000,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(self.model, prompt, text)
        self.llm_calls += 1
        return text

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0


class V2fArch(ModelAwareBase):
    """Meta V2F, configurable model. Matches production v2f retrieval exactly."""

    def retrieve(self, question: str, conversation_id: str) -> Nano54Result:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(all_segments)
        )
        prompt = META_V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return Nano54Result(
            segments=all_segments,
            metadata={
                "variant": "v2f",
                "model": self.model,
                "output": output,
                "cues": cues[:2],
            },
        )


class V2fPlusTypesArch(ModelAwareBase):
    """v2f_plus_types, configurable model. Matches V2fPlusTypesVariant exactly."""

    def __init__(
        self,
        store: SegmentStore,
        model: str,
        per_cue_top_k: int = 3,
        client: OpenAI | None = None,
    ):
        super().__init__(store, model, client)
        self.per_cue_top_k = per_cue_top_k

    def retrieve(self, question: str, conversation_id: str) -> Nano54Result:
        # Stage 1: v2f
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        v2f_context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(all_segments)
        )
        v2f_prompt = META_V2F_PROMPT.format(
            question=question, context_section=v2f_context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        v2f_cues = _parse_cues(v2f_output)[:2]

        for cue in v2f_cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        # Stage 2: type enumeration (additive)
        type_context_section = _build_context_section(all_segments)
        type_prompt = TYPE_ENUMERATED_PROMPT.format(
            question=question, context_section=type_context_section
        )
        type_output = self.llm_call(type_prompt)
        type_cues = _parse_cues(type_output)[:7]

        for cue in type_cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_cue_top_k,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return Nano54Result(
            segments=all_segments,
            metadata={
                "variant": "v2f_plus_types",
                "model": self.model,
                "v2f_output": v2f_output,
                "type_output": type_output,
                "v2f_cues": v2f_cues,
                "type_cues": type_cues,
                "per_cue_top_k": self.per_cue_top_k,
            },
        )


# ---------------------------------------------------------------------------
# Quick-test spec (per task):
#   2 LoCoMo: 1 temporal + 1 single_hop
#   1 synthetic: completeness
#   1 puzzle: logic_constraint
#   1 advanced: evolving_terminology
# ---------------------------------------------------------------------------
QUICK_TEST_SPECS = [
    # (dataset, category, question_substring_match)
    (
        "locomo_30q",
        "locomo_temporal",
        "When did Caroline go to the LGBTQ support group?",
    ),
    ("locomo_30q", "locomo_single_hop", "What did Caroline research?"),
    (
        "synthetic_19q",
        "completeness",
        "List ALL dietary restrictions and food preferences",
    ),
    (
        "puzzle_16q",
        "logic_constraint",
        "Based on all constraints discussed, what is the final valid desk",
    ),
    (
        "advanced_23q",
        "evolving_terminology",
        "What is the current status of Project Phoenix",
    ),
]


def _load_datasets_cached() -> dict:
    out: dict[str, tuple[SegmentStore, list[dict]]] = {}
    for ds in DATASETS:
        out[ds] = load_dataset(ds)
    return out


def _find_question(questions: list[dict], category: str, substring: str):
    for q in questions:
        if q.get("category") == category and substring in q["question"]:
            return q
    return None


def evaluate_question(arch, question: dict) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            arch_segments.append(seg)
            seen.add(seg.index)

    query_emb = arch.embed_text(q_text)
    max_K = max(BUDGETS)
    cosine_result = arch.store.search(query_emb, top_k=max_K, conversation_id=conv_id)
    cosine_segments = list(cosine_result.segments)

    row = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "total_arch_retrieved": len(arch_segments),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "metadata": result.metadata,
    }
    for K in BUDGETS:
        b_rec, a_rec = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)
    return row


# ---------------------------------------------------------------------------
# Mini baseline lookup — reuse existing fair-backfill files
# ---------------------------------------------------------------------------
def _load_mini_v2f_result(ds_name: str) -> dict | None:
    """Read fairbackfill_meta_v2f_<ds>.json (gpt-5-mini v2f baseline)."""
    path = RESULTS_DIR / f"fairbackfill_meta_v2f_{ds_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_mini_v2fplus_result(ds_name: str) -> dict | None:
    """Read type_enum_v2f_plus_types_<ds>.json (gpt-5-mini v2f_plus_types)."""
    path = RESULTS_DIR / f"type_enum_v2f_plus_types_{ds_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _find_mini_row(mini: dict, category: str, question: str) -> dict | None:
    for r in mini.get("results", []):
        if r.get("category") == category and r.get("question") == question:
            return r
    # fallback: match on source_chat_ids + category
    for r in mini.get("results", []):
        if r.get("category") == category:
            return r
    return None


# ---------------------------------------------------------------------------
# Quick test: nano54 vs mini (from files) on 5 questions
# ---------------------------------------------------------------------------
def run_quick_test(datasets_cached: dict) -> dict:
    print("\n" + "=" * 78)
    print("QUICK TEST — v2f under gpt-5-mini (from files) vs gpt-5.4-nano")
    print("=" * 78)

    pairs: list[dict] = []

    for ds_name, cat, substr in QUICK_TEST_SPECS:
        store, questions = datasets_cached[ds_name]
        q = _find_question(questions, cat, substr)
        if q is None:
            print(
                f"  [WARN] could not find question: {ds_name} / {cat} / {substr[:40]!r}"
            )
            continue

        print(f"\n  [{ds_name}/{cat}] {q['question'][:70]}")

        # Load mini baseline row from existing file
        mini_full = _load_mini_v2f_result(ds_name)
        if mini_full is None:
            print(f"    [WARN] no mini baseline file for {ds_name}")
            continue
        mini_row = _find_mini_row(mini_full, cat, q["question"])
        if mini_row is None:
            print("    [WARN] no mini baseline row matching question")
            continue
        mini_r20 = mini_row["fair_backfill"]["arch_r@20"]
        mini_r50 = mini_row["fair_backfill"]["arch_r@50"]
        mini_base_r20 = mini_row["fair_backfill"]["baseline_r@20"]
        mini_cues = mini_row.get("metadata", {}).get("cues", [])
        mini_output = mini_row.get("metadata", {}).get("output", "")

        # Run nano5.4
        nano_arch = V2fArch(store, model=NANO54_MODEL)
        nano_row = evaluate_question(nano_arch, q)
        nano_arch.save_caches()

        pair = {
            "dataset": ds_name,
            "category": cat,
            "question": q["question"],
            "source_chat_ids": sorted(set(q["source_chat_ids"])),
            "mini": {
                "r@20": mini_r20,
                "r@50": mini_r50,
                "baseline_r@20": mini_base_r20,
                "cues": mini_cues,
                "output": mini_output,
            },
            "nano54": {
                "r@20": nano_row["fair_backfill"]["arch_r@20"],
                "r@50": nano_row["fair_backfill"]["arch_r@50"],
                "baseline_r@20": nano_row["fair_backfill"]["baseline_r@20"],
                "cues": nano_row["metadata"].get("cues", []),
                "output": nano_row["metadata"].get("output", ""),
                "llm_calls": nano_row["llm_calls"],
                "time_s": nano_row["time_s"],
            },
        }
        pair["delta_nano54_vs_mini_r@20"] = round(
            pair["nano54"]["r@20"] - pair["mini"]["r@20"], 4
        )
        pair["delta_nano54_vs_mini_r@50"] = round(
            pair["nano54"]["r@50"] - pair["mini"]["r@50"], 4
        )
        pairs.append(pair)

        print(
            f"    mini:    r@20={pair['mini']['r@20']:.3f} "
            f"r@50={pair['mini']['r@50']:.3f}"
        )
        print(
            f"    nano5.4: r@20={pair['nano54']['r@20']:.3f} "
            f"r@50={pair['nano54']['r@50']:.3f} | "
            f"d@20={pair['delta_nano54_vs_mini_r@20']:+.3f} "
            f"d@50={pair['delta_nano54_vs_mini_r@50']:+.3f}"
        )
        for i, c in enumerate(pair["mini"]["cues"][:2]):
            print(f"      mini    cue {i + 1}: {c[:110]}")
        for i, c in enumerate(pair["nano54"]["cues"][:2]):
            print(f"      nano5.4 cue {i + 1}: {c[:110]}")

    n = len(pairs)
    n_within_5pp = sum(1 for p in pairs if abs(p["delta_nano54_vs_mini_r@20"]) <= 0.05)
    n_worse_5pp = sum(1 for p in pairs if p["delta_nano54_vs_mini_r@20"] < -0.05)
    n_worse_10pp = sum(1 for p in pairs if p["delta_nano54_vs_mini_r@20"] < -0.10)
    avg_mini_r20 = sum(p["mini"]["r@20"] for p in pairs) / max(n, 1)
    avg_nano_r20 = sum(p["nano54"]["r@20"] for p in pairs) / max(n, 1)
    avg_mini_r50 = sum(p["mini"]["r@50"] for p in pairs) / max(n, 1)
    avg_nano_r50 = sum(p["nano54"]["r@50"] for p in pairs) / max(n, 1)

    gate = {
        "n": n,
        "n_within_5pp_r@20": n_within_5pp,
        "n_worse_than_5pp_r@20": n_worse_5pp,
        "n_worse_than_10pp_r@20": n_worse_10pp,
        "avg_mini_r@20": round(avg_mini_r20, 4),
        "avg_nano54_r@20": round(avg_nano_r20, 4),
        "avg_delta_r@20": round(avg_nano_r20 - avg_mini_r20, 4),
        "avg_mini_r@50": round(avg_mini_r50, 4),
        "avg_nano54_r@50": round(avg_nano_r50, 4),
        "avg_delta_r@50": round(avg_nano_r50 - avg_mini_r50, 4),
        # "within 5pp on most" per task spec
        "passes": n_within_5pp >= (n - n // 2),
        # "meaningfully worse on most" — abandon
        "abandon": n_worse_5pp > n // 2,
    }

    print("\n  Summary:")
    print(f"    avg mini r@20:    {gate['avg_mini_r@20']:.3f}")
    print(f"    avg nano5.4 r@20: {gate['avg_nano54_r@20']:.3f}")
    print(f"    avg delta r@20:   {gate['avg_delta_r@20']:+.3f}")
    print(f"    avg mini r@50:    {gate['avg_mini_r@50']:.3f}")
    print(f"    avg nano5.4 r@50: {gate['avg_nano54_r@50']:.3f}")
    print(f"    avg delta r@50:   {gate['avg_delta_r@50']:+.3f}")
    print(f"    within 5pp:       {n_within_5pp}/{n}")
    print(f"    >5pp worse:       {n_worse_5pp}/{n}")
    print(f"    >10pp worse:      {n_worse_10pp}/{n}")
    print(
        f"    gate:             "
        f"{'PASS' if gate['passes'] else 'FAIL'}"
        f"{'  (ABANDON)' if gate['abandon'] else ''}"
    )

    return {"pairs": pairs, "gate": gate}


# ---------------------------------------------------------------------------
# Cue quality inspection
# ---------------------------------------------------------------------------
def cue_quality_inspection(quick: dict) -> dict:
    pairs = quick["pairs"]
    import re

    stats = {
        "mini": {
            "avg_cue_len": 0,
            "n_cues": 0,
            "n_questions": 0,
            "n_boolean": 0,
            "n_meta": 0,
            "n_first_person": 0,
        },
        "nano54": {
            "avg_cue_len": 0,
            "n_cues": 0,
            "n_questions": 0,
            "n_boolean": 0,
            "n_meta": 0,
            "n_first_person": 0,
        },
    }
    meta_markers = [
        "think about",
        "what would",
        "consider",
        "search for",
        "based on",
        "as an ai",
    ]
    for side in ("mini", "nano54"):
        lens = []
        for p in pairs:
            if side not in p:
                continue
            for c in p[side]["cues"]:
                lens.append(len(c))
                stats[side]["n_cues"] += 1
                if c.rstrip().endswith("?"):
                    stats[side]["n_questions"] += 1
                if re.search(r"\b(OR|AND)\b", c) or '"' in c:
                    stats[side]["n_boolean"] += 1
                low = c.lower()
                if any(m in low for m in meta_markers):
                    stats[side]["n_meta"] += 1
                if re.search(r"\b(i|my|me|we|our)\b", low):
                    stats[side]["n_first_person"] += 1
        stats[side]["avg_cue_len"] = round(sum(lens) / len(lens), 1) if lens else 0

    print("\n" + "=" * 78)
    print("CUE QUALITY INSPECTION")
    print("=" * 78)
    for side in ("mini", "nano54"):
        s = stats[side]
        print(
            f"  {side:>7s}: n_cues={s['n_cues']} "
            f"avg_len={s['avg_cue_len']} chars  "
            f"questions={s['n_questions']}  boolean={s['n_boolean']}  "
            f"meta={s['n_meta']}  first_person={s['n_first_person']}"
        )
    return stats


# ---------------------------------------------------------------------------
# Full eval — v2f and v2f_plus_types on all 4 datasets with nano5.4
# ---------------------------------------------------------------------------
def run_nano54_on_dataset(
    variant_name: str,
    ds_name: str,
    store: SegmentStore,
    questions: list[dict],
    force: bool,
) -> dict:
    out_path = RESULTS_DIR / f"nano54_{variant_name}_{ds_name}.json"
    if out_path.exists() and not force:
        print(f"  [SKIP] {out_path.name} exists (use --force to overwrite)")
        with open(out_path) as f:
            return json.load(f)

    print(f"\n  --- {variant_name} on {ds_name} (n={len(questions)}) ---")
    if variant_name == "v2f":
        arch = V2fArch(store, model=NANO54_MODEL)
    elif variant_name == "v2f_plus_types":
        arch = V2fPlusTypesArch(store, model=NANO54_MODEL, per_cue_top_k=3)
    else:
        raise ValueError(f"unknown variant {variant_name}")

    results: list[dict] = []
    for i, q in enumerate(questions):
        q_short = q["question"][:60]
        print(
            f"    [{i + 1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(arch, q)
            results.append(row)
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "variant": variant_name,
                        "model": NANO54_MODEL,
                        "dataset": ds_name,
                        "results": results,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()

        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = summarize(results, f"nano54_{variant_name}", ds_name)
    by_cat = summarize_by_category(results)

    saved = {
        "variant": variant_name,
        "model": NANO54_MODEL,
        "dataset": ds_name,
        "summary": summary,
        "category_breakdown": by_cat,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(saved, f, indent=2, default=str)

    print(
        f"    r@20: baseline={summary['baseline_r@20']:.3f} "
        f"arch={summary['arch_r@20']:.3f} "
        f"delta={summary['delta_r@20']:+.3f} "
        f"W/T/L={summary['W/T/L_r@20']}"
    )
    print(
        f"    r@50: baseline={summary['baseline_r@50']:.3f} "
        f"arch={summary['arch_r@50']:.3f} "
        f"delta={summary['delta_r@50']:+.3f} "
        f"W/T/L={summary['W/T/L_r@50']}"
    )
    return saved


def _load_mini_baseline(variant_name: str, ds_name: str) -> dict | None:
    if variant_name == "v2f":
        return _load_mini_v2f_result(ds_name)
    if variant_name == "v2f_plus_types":
        return _load_mini_v2fplus_result(ds_name)
    return None


def run_full_eval(datasets_cached: dict, force: bool) -> dict:
    print("\n" + "=" * 78)
    print("FULL EVAL — v2f and v2f_plus_types on all 4 datasets with gpt-5.4-nano")
    print("=" * 78)

    nano54_runs: dict[str, dict[str, dict]] = {}
    for variant_name in ("v2f", "v2f_plus_types"):
        nano54_runs[variant_name] = {}
        for ds_name in DATASETS:
            store, questions = datasets_cached[ds_name]
            saved = run_nano54_on_dataset(
                variant_name, ds_name, store, questions, force=force
            )
            nano54_runs[variant_name][ds_name] = saved

    comparisons: dict = {}
    for variant_name, ds_map in nano54_runs.items():
        comparisons[variant_name] = {}
        for ds_name, saved in ds_map.items():
            nano_s = saved.get("summary", {})
            mini = _load_mini_baseline(variant_name, ds_name)
            mini_s = mini.get("summary", {}) if mini else {}
            entry: dict = {"overall": {}, "per_category": {}}
            for K in BUDGETS:
                entry["overall"][f"r@{K}"] = {
                    "baseline": nano_s.get(f"baseline_r@{K}"),
                    "mini_arch": mini_s.get(f"arch_r@{K}"),
                    "nano54_arch": nano_s.get(f"arch_r@{K}"),
                    "delta_nano54_vs_mini": (
                        round(nano_s[f"arch_r@{K}"] - mini_s[f"arch_r@{K}"], 4)
                        if f"arch_r@{K}" in nano_s and f"arch_r@{K}" in mini_s
                        else None
                    ),
                    "mini_W/T/L": mini_s.get(f"W/T/L_r@{K}"),
                    "nano54_W/T/L": nano_s.get(f"W/T/L_r@{K}"),
                }
            nano_cats = saved.get("category_breakdown", {})
            mini_cats = (mini or {}).get("category_breakdown", {})
            all_cats = sorted(set(nano_cats.keys()) | set(mini_cats.keys()))
            for cat in all_cats:
                n_c = nano_cats.get(cat, {})
                m_c = mini_cats.get(cat, {})
                cat_entry: dict = {"n": n_c.get("n", m_c.get("n"))}
                for K in BUDGETS:
                    cat_entry[f"r@{K}"] = {
                        "baseline": n_c.get(f"baseline_r@{K}"),
                        "mini_arch": m_c.get(f"arch_r@{K}"),
                        "nano54_arch": n_c.get(f"arch_r@{K}"),
                        "delta_nano54_vs_mini": (
                            round(n_c[f"arch_r@{K}"] - m_c[f"arch_r@{K}"], 4)
                            if f"arch_r@{K}" in n_c and f"arch_r@{K}" in m_c
                            else None
                        ),
                    }
                entry["per_category"][cat] = cat_entry
            comparisons[variant_name][ds_name] = entry

    return {"runs": nano54_runs, "comparisons": comparisons}


def print_full_eval_table(full: dict) -> None:
    print("\n" + "=" * 100)
    print("NANO5.4 vs MINI — FULL EVAL SUMMARY")
    print("=" * 100)

    for variant_name, ds_map in full["comparisons"].items():
        for ds_name, entry in ds_map.items():
            print(f"\n--- {variant_name} on {ds_name} ---")
            ov = entry["overall"]
            for K in BUDGETS:
                o = ov[f"r@{K}"]

                def _fmt(x):
                    return f"{x:.3f}" if isinstance(x, (int, float)) else "  n/a"

                delta_s = (
                    f"{o['delta_nano54_vs_mini']:+.4f}"
                    if o["delta_nano54_vs_mini"] is not None
                    else "   n/a"
                )
                print(
                    f"  r@{K}: baseline={_fmt(o['baseline'])} "
                    f"mini={_fmt(o['mini_arch'])} "
                    f"nano5.4={_fmt(o['nano54_arch'])} "
                    f"d_nano54_vs_mini={delta_s}"
                )

    print("\n" + "=" * 100)
    print("CROSS-DATASET AVERAGE (nano5.4 vs mini)")
    print("=" * 100)
    for variant_name, ds_map in full["comparisons"].items():
        sums = {f"mini_r@{K}": [] for K in BUDGETS}
        sums.update({f"nano54_r@{K}": [] for K in BUDGETS})
        sums.update({f"base_r@{K}": [] for K in BUDGETS})
        for ds_name, entry in ds_map.items():
            ov = entry["overall"]
            for K in BUDGETS:
                if ov[f"r@{K}"]["mini_arch"] is not None:
                    sums[f"mini_r@{K}"].append(ov[f"r@{K}"]["mini_arch"])
                if ov[f"r@{K}"]["nano54_arch"] is not None:
                    sums[f"nano54_r@{K}"].append(ov[f"r@{K}"]["nano54_arch"])
                if ov[f"r@{K}"]["baseline"] is not None:
                    sums[f"base_r@{K}"].append(ov[f"r@{K}"]["baseline"])
        print(f"\n  {variant_name}:")
        for K in BUDGETS:
            base_list = sums[f"base_r@{K}"]
            mini_list = sums[f"mini_r@{K}"]
            nano_list = sums[f"nano54_r@{K}"]
            base = sum(base_list) / len(base_list) if base_list else 0
            mini = sum(mini_list) / len(mini_list) if mini_list else 0
            nano = sum(nano_list) / len(nano_list) if nano_list else 0
            print(
                f"    r@{K}: base={base:.3f} mini={mini:.3f} "
                f"nano5.4={nano:.3f} "
                f"delta_nano54_vs_mini={nano - mini:+.4f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="gpt-5.4-nano vs gpt-5-mini replacement test"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="After quick-test, run full eval on all 4 datasets",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing nano5.4 result files"
    )
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Run full eval even if quick-test fails gate",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets_cached = _load_datasets_cached()
    for ds, (store, qs) in datasets_cached.items():
        print(f"  loaded {ds}: {len(qs)} questions, {len(store.segments)} segs")

    quick = run_quick_test(datasets_cached)
    cue_stats = cue_quality_inspection(quick)

    out = {"quick_test": quick, "cue_stats": cue_stats}

    quick_path = RESULTS_DIR / "nano54_quick_test.json"
    with open(quick_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  saved: {quick_path}")

    if not args.full:
        if not quick["gate"]["passes"]:
            print("\n  Quick-test FAILED gate. Not running full eval.")
        else:
            print("\n  Quick-test passed. Use --full to continue to full eval.")
        return

    if quick["gate"]["abandon"] and not args.skip_gate:
        print(
            "\n  Quick-test ABANDON (nano5.4 meaningfully worse on most). "
            "Use --skip-gate to run full eval anyway."
        )
        return

    full = run_full_eval(datasets_cached, force=args.force)
    print_full_eval_table(full)

    out["full_eval"] = full

    full_path = RESULTS_DIR / "nano54_full_eval_summary.json"
    with open(full_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  saved: {full_path}")


if __name__ == "__main__":
    main()
