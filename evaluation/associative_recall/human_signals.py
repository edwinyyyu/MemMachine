"""Human-memory-inspired retrieval signals on top of V2f.

Tests three signals:
  A. neighbors_v2f: show 2-3 nearest NON-retrieved turns for each retrieved segment
  B. temporal_position_v2f: normalize turn_id to 0-1 across conversation length
  C. distribution_v2f: show histogram of retrieved segments across timeline
  D. all_signals_v2f: all three signals combined

Evaluated with FAIR K-budget at K=20 on LoCoMo 30q, synthetic 19q, puzzle 16q,
advanced 23q, with cosine backfill when arch retrieves less than K.

Usage:
    uv run python human_signals.py
    uv run python human_signals.py --variant neighbors_v2f
    uv run python human_signals.py --datasets locomo_30q synthetic_19q
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGET = 20
MAX_K_FOR_BACKFILL = 20

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
# Caches (human_signals-specific; read from other caches for warmth)
# ---------------------------------------------------------------------------
class HumanSignalsEmbeddingCache:
    """Read from a union of embedding caches; write to human_signals cache."""

    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in (
            "embedding_cache.json",
            "bestshot_embedding_cache.json",
            "meta_embedding_cache.json",
            "arch_embedding_cache.json",
            "agent_embedding_cache.json",
            "frontier_embedding_cache.json",
            "optim_embedding_cache.json",
            "human_signals_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                try:
                    with open(p) as f:
                        self._cache.update(json.load(f))
                except Exception:
                    pass
        self.cache_file = self.cache_dir / "human_signals_embedding_cache.json"
        self._new_entries: dict[str, list[float]] = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        k = self._key(text)
        if k in self._cache:
            return np.array(self._cache[k], dtype=np.float32)
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        k = self._key(text)
        self._cache[k] = embedding.tolist()
        self._new_entries[k] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(f".json.tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        try:
            tmp.replace(self.cache_file)
        except FileNotFoundError:
            pass
        self._new_entries.clear()


class HumanSignalsLLMCache:
    """Read from a union of LLM caches; write to human_signals cache."""

    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in (
            "llm_cache.json",
            "bestshot_llm_cache.json",
            "meta_llm_cache.json",
            "arch_llm_cache.json",
            "agent_llm_cache.json",
            "frontier_llm_cache.json",
            "optim_llm_cache.json",
            "human_signals_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                try:
                    with open(p) as f:
                        data = json.load(f)
                    for k, v in data.items():
                        if v:
                            self._cache[k] = v
                except Exception:
                    pass
        self.cache_file = self.cache_dir / "human_signals_llm_cache.json"
        self._new_entries: dict[str, str] = {}

    @staticmethod
    def _key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str) -> str | None:
        return self._cache.get(self._key(model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        k = self._key(model, prompt)
        self._cache[k] = response
        self._new_entries[k] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(f".json.tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        try:
            tmp.replace(self.cache_file)
        except FileNotFoundError:
            pass
        self._new_entries.clear()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_cues(response: str) -> list[str]:
    cues: list[str] = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


def _conversation_length(store: SegmentStore, conversation_id: str) -> int:
    """Return max_turn_id + 1 within a conversation (used for normalization)."""
    conv_map = store._turn_index.get(conversation_id, {})
    if not conv_map:
        return 1
    max_tid = max(conv_map.keys())
    return max_tid + 1


def _pct_label(turn_id: int, conv_len: int) -> str:
    if conv_len <= 0:
        return "0%"
    pct = int(round(100 * turn_id / max(1, conv_len - 1)))
    pct = max(0, min(100, pct))
    return f"{pct}%"


def _format_segment_line(
    seg: Segment,
    *,
    with_temporal: bool,
    with_neighbors: bool,
    retrieved_indices: set[int],
    store: SegmentStore,
    conv_len: int,
    max_chars: int = 250,
) -> str:
    """Format one line describing a retrieved segment with optional signals."""
    head = f"[Turn {seg.turn_id}"
    if with_temporal:
        head += f" ({_pct_label(seg.turn_id, conv_len)} into conv)"
    head += f", {seg.role}]: {seg.text[:max_chars]}"

    if with_neighbors:
        conv_map = store._turn_index.get(seg.conversation_id, {})
        # Find nearest non-retrieved neighbors by absolute turn_id distance.
        nearby: list[tuple[int, int]] = []  # (distance, turn_id)
        for offset in (1, -1, 2, -2, 3, -3):
            tid = seg.turn_id + offset
            if tid in conv_map:
                idx = conv_map[tid]
                if idx not in retrieved_indices:
                    nearby.append((abs(offset), tid))
            if len(nearby) >= 3:
                break
        if nearby:
            nearby_parts = ", ".join(f"Turn {tid}" for _, tid in nearby[:3])
            head += f"\n  Nearby (not retrieved): {nearby_parts}"
    return head


def _format_context(
    segments: list[Segment],
    *,
    variant: str,
    store: SegmentStore,
    conv_len: int,
    max_items: int = 12,
) -> str:
    """Format retrieved segments into the context section used by V2f."""
    with_temporal = variant in {"temporal_position_v2f", "all_signals_v2f"}
    with_neighbors = variant in {"neighbors_v2f", "all_signals_v2f"}
    with_distribution = variant in {"distribution_v2f", "all_signals_v2f"}

    if not segments:
        body = (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
        return body

    retrieved_indices = {s.index for s in segments}
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines: list[str] = ["RETRIEVED CONVERSATION EXCERPTS SO FAR:"]
    for seg in sorted_segs:
        lines.append(
            _format_segment_line(
                seg,
                with_temporal=with_temporal,
                with_neighbors=with_neighbors,
                retrieved_indices=retrieved_indices,
                store=store,
                conv_len=conv_len,
            )
        )

    if with_distribution:
        lines.append("")
        lines.append(_format_distribution(segments, conv_len))

    return "\n".join(lines)


def _format_distribution(segments: list[Segment], conv_len: int) -> str:
    """Histogram of retrieved segments across timeline quartiles."""
    if conv_len <= 1:
        return ""
    buckets = [0, 0, 0, 0]
    for seg in segments:
        frac = seg.turn_id / max(1, conv_len - 1)
        b = min(3, int(frac * 4))
        buckets[b] += 1
    total = sum(buckets)
    labels = ["0-25%", "25-50%", "50-75%", "75-100%"]

    lines = [f"Retrieved so far: {total} segments", "Distribution:"]
    for lbl, cnt in zip(labels, buckets):
        lines.append(f"  {lbl}: {cnt} segments")

    # Highlight gaps: any bucket with <= 25% of mean coverage (and < 2 segs) is a gap
    if total > 0:
        mean_per_bucket = total / 4
        gaps = []
        for lbl, cnt in zip(labels, buckets):
            if cnt <= max(1, mean_per_bucket * 0.4):
                gaps.append(f"{lbl} (only {cnt})")
        if gaps:
            lines.append("Coverage gaps: " + ", ".join(gaps))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
@dataclass
class SignalsResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


class HumanSignalsV2f:
    """V2f prompt with one of {neighbors, temporal, distribution, all} signals."""

    def __init__(
        self,
        store: SegmentStore,
        variant: str,
        client: OpenAI | None = None,
    ) -> None:
        assert variant in {
            "v2f",
            "neighbors_v2f",
            "temporal_position_v2f",
            "distribution_v2f",
            "all_signals_v2f",
        }
        self.store = store
        self.variant = variant
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = HumanSignalsEmbeddingCache()
        self.llm_cache = HumanSignalsLLMCache()
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

    def retrieve(self, question: str, conversation_id: str) -> SignalsResult:
        conv_len = _conversation_length(self.store, conversation_id)

        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        context_section = _format_context(
            all_segments,
            variant=self.variant,
            store=self.store,
            conv_len=conv_len,
        )

        prompt = V2F_PROMPT.format(question=question, context_section=context_section)
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

        return SignalsResult(
            segments=all_segments,
            metadata={
                "variant": self.variant,
                "output": output,
                "cues": cues[:2],
                "conv_len": conv_len,
            },
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    source_ids: set[int],
    budget: int,
) -> tuple[float, float]:
    """Arch gets its own segments in order, backfilled from cosine top-K if
    fewer than K. Baseline = cosine top-K. Returns (baseline_recall, arch_recall).
    """
    seen: set[int] = set()
    arch_unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            arch_unique.append(s)
            seen.add(s.index)
    arch_at_K = arch_unique[:budget]
    arch_indices = {s.index for s in arch_at_K}
    if len(arch_at_K) < budget:
        backfill = [s for s in cosine_segments if s.index not in arch_indices]
        needed = budget - len(arch_at_K)
        arch_at_K = arch_at_K + backfill[:needed]
    arch_at_K = arch_at_K[:budget]
    baseline_at_K = cosine_segments[:budget]
    return (
        compute_recall({s.turn_id for s in baseline_at_K}, source_ids),
        compute_recall({s.turn_id for s in arch_at_K}, source_ids),
    )


def evaluate_question(arch: HumanSignalsV2f, question: dict) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedupe preserving order
    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            arch_segments.append(seg)
            seen.add(seg.index)

    query_emb = arch.embed_text(q_text)
    cosine_result = arch.store.search(
        query_emb, top_k=MAX_K_FOR_BACKFILL, conversation_id=conv_id
    )
    cosine_segments = list(cosine_result.segments)

    baseline_recall, arch_recall = fair_backfill(
        arch_segments, cosine_segments, source_ids, BUDGET
    )

    return {
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
        f"baseline_r@{BUDGET}": round(baseline_recall, 4),
        f"arch_r@{BUDGET}": round(arch_recall, 4),
        f"delta_r@{BUDGET}": round(arch_recall - baseline_recall, 4),
        "cues": result.metadata.get("cues", []),
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    b_vals = [r[f"baseline_r@{BUDGET}"] for r in results]
    a_vals = [r[f"arch_r@{BUDGET}"] for r in results]
    b_mean = sum(b_vals) / n
    a_mean = sum(a_vals) / n
    wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
    losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
    ties = n - wins - losses
    return {
        "n": n,
        f"baseline_r@{BUDGET}": round(b_mean, 4),
        f"arch_r@{BUDGET}": round(a_mean, 4),
        f"delta_r@{BUDGET}": round(a_mean - b_mean, 4),
        f"W/T/L_r@{BUDGET}": f"{wins}/{ties}/{losses}",
        "avg_total_retrieved": round(
            sum(r["total_arch_retrieved"] for r in results) / n, 1
        ),
        "avg_llm_calls": round(sum(r["llm_calls"] for r in results) / n, 1),
        "avg_embed_calls": round(sum(r["embed_calls"] for r in results) / n, 1),
    }


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out: dict[str, dict] = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        b_vals = [r[f"baseline_r@{BUDGET}"] for r in rs]
        a_vals = [r[f"arch_r@{BUDGET}"] for r in rs]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses
        out[cat] = {
            "n": n,
            f"baseline_r@{BUDGET}": round(b_mean, 4),
            f"arch_r@{BUDGET}": round(a_mean, 4),
            f"delta_r@{BUDGET}": round(a_mean - b_mean, 4),
            f"W/T/L_r@{BUDGET}": f"{wins}/{ties}/{losses}",
        }
    return out


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
    ds_name: str,
    store: SegmentStore,
    questions: list[dict],
    out_path: Path,
    force: bool = False,
) -> tuple[list[dict], dict, dict]:
    """Run one (variant, dataset) pair, save incrementally, return results."""
    print(f"\n{'=' * 70}")
    print(f"{variant} | {ds_name} | {len(questions)} questions")
    print(f"{'=' * 70}", flush=True)

    # Resume: load prior results keyed by (conversation_id, question_index)
    done_keys: set[tuple[str, int]] = set()
    results: list[dict] = []
    if out_path.exists() and not force:
        try:
            with open(out_path) as f:
                prior = json.load(f)
            results = list(prior.get("results", []))
            done_keys = {(r["conversation_id"], r["question_index"]) for r in results}
            print(f"  Resuming: {len(results)} prior rows")
        except Exception:
            results = []
            done_keys = set()

    arch = HumanSignalsV2f(store, variant=variant)

    for i, q in enumerate(questions):
        key = (q["conversation_id"], q.get("question_index", -1))
        if key in done_keys:
            continue
        q_short = q["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(arch, q)
            results.append(row)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()
            continue

        # Incremental save per question
        summary = summarize(results)
        by_cat = summarize_by_category(results)
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
        if (i + 1) % 3 == 0:
            arch.save_caches()
        sys.stdout.flush()

    arch.save_caches()

    summary = summarize(results)
    by_cat = summarize_by_category(results)

    # Final save
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

    print(f"\n--- {variant} on {ds_name} ---")
    if summary.get("n"):
        print(
            f"  r@{BUDGET}: baseline={summary[f'baseline_r@{BUDGET}']:.3f} "
            f"arch={summary[f'arch_r@{BUDGET}']:.3f} "
            f"delta={summary[f'delta_r@{BUDGET}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{BUDGET}']}"
        )
        print(
            f"  avg retrieved={summary['avg_total_retrieved']:.0f} "
            f"llm={summary['avg_llm_calls']:.1f} "
            f"embed={summary['avg_embed_calls']:.1f}"
        )
    return results, summary, by_cat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=[
            "v2f",
            "neighbors_v2f",
            "temporal_position_v2f",
            "distribution_v2f",
            "all_signals_v2f",
        ],
        help="Run a single variant (default: all four signal variants)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to run (default: all four)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results instead of resuming",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.variant:
        variants = [args.variant]
    else:
        variants = [
            "neighbors_v2f",
            "temporal_position_v2f",
            "distribution_v2f",
            "all_signals_v2f",
        ]

    datasets = args.datasets or list(DATASETS.keys())
    for ds in datasets:
        if ds not in DATASETS:
            raise SystemExit(f"Unknown dataset: {ds}")

    all_summaries: dict = {}

    for ds_name in datasets:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )
        for variant in variants:
            out_path = RESULTS_DIR / f"human_signals_{variant}_{ds_name}.json"
            _, summary, by_cat = run_one(
                variant, ds_name, store, questions, out_path, force=args.force
            )
            all_summaries.setdefault(variant, {})[ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
            }

    summary_path = RESULTS_DIR / "human_signals_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")

    print_final_table(all_summaries, datasets)


def print_final_table(all_summaries: dict, datasets: list[str]) -> None:
    print("\n" + "=" * 110)
    print(f"HUMAN SIGNALS SUMMARY  (fair K-budget at K={BUDGET})")
    print("=" * 110)
    hdr = f"{'Variant':<26s} " + " ".join(f"{ds:>14s}" for ds in datasets)
    print(hdr)
    print("-" * len(hdr))
    for variant in all_summaries:
        row = f"{variant:<26s} "
        for ds in datasets:
            s = all_summaries.get(variant, {}).get(ds, {}).get("summary", {})
            if s.get("n"):
                val = s.get(f"arch_r@{BUDGET}", 0)
                base = s.get(f"baseline_r@{BUDGET}", 0)
                delta = s.get(f"delta_r@{BUDGET}", 0)
                row += f"  {val:.3f}({delta:+.3f}) "
            else:
                row += f"  {'n/a':>12s} "
        print(row)


if __name__ == "__main__":
    main()
