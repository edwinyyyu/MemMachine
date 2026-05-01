"""Task-shape robustness test for gated_overlay.

Runs the primary `gated_threshold_0.7` variant of GatedOverlay on 90 LoCoMo
task-shape variants (30 originals x {CMD, DRAFT, META}). Reuses cached
ORIGINAL-shape results from results/gated_overlay.json — only the 90 new
variant inputs need fresh retrieval.

Measures:
  - fair-backfill r@20 and r@50 per (orig_idx, shape)
  - confidences and firing_channels per variant (to compare channel-fire
    patterns across shapes of the same query)

Outputs:
  results/gated_shape.json
  results/gated_shape.md

Dedicated caches (gatedTS_*) so concurrent agents cannot corrupt this run.

Usage:
    uv run python gated_shape_eval.py
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
VARIANTS_FILE = DATA_DIR / "questions_locomo_task_shape.json"
GATED_ORIGINAL_JSON = RESULTS_DIR / "gated_overlay.json"

BUDGETS = (20, 50)
SHAPES = ("CMD", "DRAFT", "META")
ORIGINAL_SHAPE = "ORIGINAL"

GATEDTS_EMB_FILE = CACHE_DIR / "gatedTS_embedding_cache.json"
GATEDTS_LLM_FILE = CACHE_DIR / "gatedTS_llm_cache.json"

# Shared warm-start caches (read-only). Writes go only to gatedTS_*.
SHARED_EMB_READ = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "antipara_embedding_cache.json",
    "alias_embedding_cache.json",
    "speaker_embedding_cache.json",
    "two_speaker_embedding_cache.json",
    "multich_embedding_cache.json",
    "gated_embedding_cache.json",
    "tasksh_embedding_cache.json",
    "gatedTS_embedding_cache.json",
)
SHARED_LLM_READ = (
    "llm_cache.json",
    "arch_llm_cache.json",
    "tree_llm_cache.json",
    "frontier_llm_cache.json",
    "meta_llm_cache.json",
    "optim_llm_cache.json",
    "bestshot_llm_cache.json",
    "alias_llm_cache.json",
    "speaker_llm_cache.json",
    "two_speaker_llm_cache.json",
    "antipara_llm_cache.json",
    "multich_llm_cache.json",
    "gated_llm_cache.json",
    "tasksh_llm_cache.json",
    "gatedTS_llm_cache.json",
)


# ---------------------------------------------------------------------------
# Dedicated cache classes (write to gatedTS_*, read from shared warm set)
# ---------------------------------------------------------------------------
class GatedTSEmbeddingCache(EmbeddingCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in SHARED_EMB_READ:
            p = self.cache_dir / name
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        self.cache_file = GATEDTS_EMB_FILE
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.parent / (self.cache_file.name + f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new_entries = {}


class GatedTSLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in SHARED_LLM_READ:
            p = self.cache_dir / name
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            for k, v in data.items():
                if v:
                    self._cache[k] = v
        self.cache_file = GATEDTS_LLM_FILE
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.parent / (self.cache_file.name + f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Monkey-patch GatedOverlay's cache classes BEFORE importing it, so its
# writes go only to gatedTS_* and cannot corrupt gated_* caches being read
# by concurrent agents.
# ---------------------------------------------------------------------------
import gated_overlay as _go


def _gated_emb_init(self):
    GatedTSEmbeddingCache.__init__(self)


def _gated_llm_init(self):
    GatedTSLLMCache.__init__(self)


_go.GatedEmbeddingCache.__init__ = _gated_emb_init  # type: ignore[method-assign]
_go.GatedLLMCache.__init__ = _gated_llm_init  # type: ignore[method-assign]


# Now safe to import architecture.
from gated_overlay import SUPPLEMENT_NAMES, GatedOverlay  # noqa: E402


# ---------------------------------------------------------------------------
# Fair-backfill helpers
# ---------------------------------------------------------------------------
def _recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    budget: int,
) -> list[Segment]:
    seen: set[int] = set()
    unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            unique.append(s)
            seen.add(s.index)
    at_K = unique[:budget]
    have = {s.index for s in at_K}
    if len(at_K) < budget:
        for s in cosine_segments:
            if s.index in have:
                continue
            at_K.append(s)
            have.add(s.index)
            if len(at_K) >= budget:
                break
    return at_K[:budget]


def evaluate_one(arch: GatedOverlay, q: dict) -> dict:
    """Run gated overlay at each K (overlay depends on K).

    Returns row with fair-backfill recall and firing metadata.
    """
    q_text = q["question"]
    conv_id = q["conversation_id"]
    source_ids = set(q["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    results_per_k: dict[int, object] = {}
    for K in BUDGETS:
        res = arch.retrieve(q_text, conv_id, K=K)
        results_per_k[K] = res
    elapsed = time.time() - t0

    query_emb = arch.embed_text(q_text)
    cosine_result = arch.store.search(
        query_emb, top_k=max(BUDGETS), conversation_id=conv_id
    )
    cosine_segments = list(cosine_result.segments)

    meta = results_per_k[max(BUDGETS)].metadata

    row: dict = {
        "orig_row_index": q.get("orig_row_index", -1),
        "shape": q.get("shape", ORIGINAL_SHAPE),
        "conversation_id": conv_id,
        "category": q.get("category", "unknown"),
        "question": q_text,
        "original_question": q.get("original_question", q_text),
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "metadata": {
            "confidences": meta.get("confidences", {}),
            "reasoning": meta.get("reasoning", ""),
            "firing_channels": meta.get("firing_channels", []),
            "m_effective": meta.get("m_effective", {}),
            "overlay": meta.get("overlay", {}),
        },
    }

    for K in BUDGETS:
        arch_segments = results_per_k[K].segments
        arch_at_K = fair_backfill(arch_segments, cosine_segments, K)
        baseline_at_K = cosine_segments[:K]
        arch_ids = {s.turn_id for s in arch_at_K}
        base_ids = {s.turn_id for s in baseline_at_K}
        b_rec = _recall(base_ids, source_ids)
        a_rec = _recall(arch_ids, source_ids)
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)
    return row


# ---------------------------------------------------------------------------
# Original-shape reuse: load cached rows from gated_overlay.json
# ---------------------------------------------------------------------------
def load_cached_originals() -> list[dict]:
    """Pull the 30 LoCoMo original rows from the shipped gated run and
    rebase them into our row schema."""
    with open(GATED_ORIGINAL_JSON) as f:
        d = json.load(f)
    src_rows = d["gated_threshold_0.7"]["locomo_30q"]["results"]
    # Load variants file to map question_index -> original_question
    with open(VARIANTS_FILE) as f:
        variants = json.load(f)
    orig_q_lookup: dict[int, str] = {}
    for v in variants:
        orig_q_lookup[v["orig_row_index"]] = v["original_question"]

    out: list[dict] = []
    for src in src_rows:
        qi = src.get("question_index", -1)
        meta = src.get("metadata", {}) or {}
        out.append(
            {
                "orig_row_index": qi,
                "shape": ORIGINAL_SHAPE,
                "conversation_id": src["conversation_id"],
                "category": src["category"],
                "question": src["question"],
                "original_question": orig_q_lookup.get(qi, src["question"]),
                "source_chat_ids": src["source_chat_ids"],
                "num_source_turns": src["num_source_turns"],
                "embed_calls": src.get("embed_calls", 0),
                "llm_calls": src.get("llm_calls", 0),
                "time_s": src.get("time_s", 0.0),
                "fair_backfill": src["fair_backfill"],
                "metadata": {
                    "confidences": meta.get("confidences", {}),
                    "reasoning": meta.get("reasoning", ""),
                    "firing_channels": meta.get("firing_channels", []),
                    "m_effective": meta.get("m_effective", {}),
                    "overlay": meta.get("overlay", {}),
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def per_shape_recall(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    if n == 0:
        return out
    for K in BUDGETS:
        a = sum(r["fair_backfill"][f"arch_r@{K}"] for r in rows) / n
        b = sum(r["fair_backfill"][f"baseline_r@{K}"] for r in rows) / n
        out[f"mean_arch_r@{K}"] = round(a, 4)
        out[f"mean_baseline_r@{K}"] = round(b, 4)
        out[f"mean_delta_r@{K}"] = round(a - b, 4)
    return out


def channel_fire_stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    fire: dict[str, int] = dict.fromkeys(SUPPLEMENT_NAMES, 0)
    contrib: dict[str, int] = dict.fromkeys(SUPPLEMENT_NAMES, 0)
    conf_sums: dict[str, float] = dict.fromkeys(SUPPLEMENT_NAMES, 0.0)
    firing_per_q: list[int] = []
    for r in rows:
        meta = r.get("metadata", {}) or {}
        confs = meta.get("confidences", {}) or {}
        firing = meta.get("firing_channels", []) or []
        overlay = meta.get("overlay", {}) or {}
        contribs = overlay.get("channels_contributing", []) or []
        firing_per_q.append(len(firing))
        for ch in firing:
            if ch in fire:
                fire[ch] += 1
        for ch in contribs:
            if ch in contrib:
                contrib[ch] += 1
        for ch in SUPPLEMENT_NAMES:
            conf_sums[ch] += float(confs.get(ch, 0.0))
    return {
        "n": n,
        "avg_firing_per_query": round(sum(firing_per_q) / n, 3),
        "fire_rate_per_channel": {
            ch: round(fire[ch] / n, 3) for ch in SUPPLEMENT_NAMES
        },
        "contribution_rate_per_channel": {
            ch: round(contrib[ch] / n, 3) for ch in SUPPLEMENT_NAMES
        },
        "avg_confidence_per_channel": {
            ch: round(conf_sums[ch] / n, 3) for ch in SUPPLEMENT_NAMES
        },
    }


def channel_agreement_vs_original(
    rows_by_shape: dict[str, list[dict]],
) -> dict:
    """% of queries where firing-channel set matches ORIGINAL, per shape."""
    orig = {
        r["orig_row_index"]: set(r["metadata"]["firing_channels"])
        for r in rows_by_shape.get(ORIGINAL_SHAPE, [])
    }
    out: dict = {}
    for sh in SHAPES:
        rows = rows_by_shape.get(sh, [])
        if not rows or not orig:
            continue
        agree = 0
        total = 0
        conf_cos_sum = 0.0
        for r in rows:
            key = r["orig_row_index"]
            if key not in orig:
                continue
            total += 1
            sh_set = set(r["metadata"]["firing_channels"])
            if sh_set == orig[key]:
                agree += 1
            # Cosine similarity of confidence vectors
            orig_row = next(
                (
                    x
                    for x in rows_by_shape[ORIGINAL_SHAPE]
                    if x["orig_row_index"] == key
                ),
                None,
            )
            if orig_row is not None:
                c_o = orig_row["metadata"].get("confidences", {}) or {}
                c_s = r["metadata"].get("confidences", {}) or {}
                v_o = np.array([float(c_o.get(ch, 0.0)) for ch in SUPPLEMENT_NAMES])
                v_s = np.array([float(c_s.get(ch, 0.0)) for ch in SUPPLEMENT_NAMES])
                nom = float(np.linalg.norm(v_o) * np.linalg.norm(v_s))
                if nom > 1e-10:
                    conf_cos_sum += float(v_o @ v_s) / nom
                else:
                    conf_cos_sum += (
                        1.0
                        if (np.linalg.norm(v_o) < 1e-10 and np.linalg.norm(v_s) < 1e-10)
                        else 0.0
                    )
        out[sh] = {
            "n_compared": total,
            "exact_firing_set_match_pct": (
                round(100.0 * agree / total, 2) if total else 0.0
            ),
            "mean_confidence_cosine_vs_original": (
                round(conf_cos_sum / total, 4) if total else 0.0
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_markdown(
    rows_by_shape: dict[str, list[dict]],
    shape_summary: dict,
    fire_stats: dict,
    agreement: dict,
) -> str:
    lines: list[str] = []
    lines.append("# Gated Overlay: Task-Shape Robustness\n")
    lines.append(
        "Tests whether the shipped `gated_overlay` architecture (primary "
        "variant: `gated_threshold_0.7`) retains its LoCoMo K=50 recall "
        "across task-shape rewrites of the same 30 LoCoMo questions "
        "({CMD, DRAFT, META}).\n"
    )
    lines.append(
        "Original shape results reused from "
        "`results/gated_overlay.json`. CMD/DRAFT/META were freshly "
        "retrieved with dedicated `gatedTS_*` caches.\n"
    )

    # --- Per-shape recall ---
    lines.append("\n## Recall by shape (fair-backfill)\n")
    lines.append(
        "| Shape | n | arch_r@20 | arch_r@50 | Δ_r@20 | Δ_r@50 | "
        "Drop vs ORIG @20 | Drop vs ORIG @50 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    order = (ORIGINAL_SHAPE, "CMD", "DRAFT", "META")
    orig_s = shape_summary.get(ORIGINAL_SHAPE, {})
    for sh in order:
        s = shape_summary.get(sh)
        if not s or s.get("n", 0) == 0:
            continue
        if sh == ORIGINAL_SHAPE or not orig_s:
            drop20 = 0.0
            drop50 = 0.0
        else:
            drop20 = orig_s.get("mean_arch_r@20", 0.0) - s.get("mean_arch_r@20", 0.0)
            drop50 = orig_s.get("mean_arch_r@50", 0.0) - s.get("mean_arch_r@50", 0.0)
        lines.append(
            f"| {sh} | {s['n']} | {s['mean_arch_r@20']:.4f} | "
            f"{s['mean_arch_r@50']:.4f} | "
            f"{s['mean_delta_r@20']:+.4f} | "
            f"{s['mean_delta_r@50']:+.4f} | "
            f"{drop20:+.4f} | {drop50:+.4f} |"
        )

    # --- Channel-fire stats ---
    lines.append("\n## Channel-fire patterns by shape\n")
    lines.append(
        "| Shape | avg fires/q | speaker_filter | alias_context | "
        "critical_info | temporal_tokens | entity_exact_match |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for sh in order:
        fs = fire_stats.get(sh)
        if not fs or fs.get("n", 0) == 0:
            continue
        fr = fs.get("fire_rate_per_channel", {})
        lines.append(
            f"| {sh} | {fs.get('avg_firing_per_query', 0.0):.2f} | "
            f"{fr.get('speaker_filter', 0):.2f} | "
            f"{fr.get('alias_context', 0):.2f} | "
            f"{fr.get('critical_info', 0):.2f} | "
            f"{fr.get('temporal_tokens', 0):.2f} | "
            f"{fr.get('entity_exact_match', 0):.2f} |"
        )

    lines.append("\n### Avg confidence per channel by shape\n")
    lines.append(
        "| Shape | speaker_filter | alias_context | critical_info | "
        "temporal_tokens | entity_exact_match |"
    )
    lines.append("|---|---|---|---|---|---|")
    for sh in order:
        fs = fire_stats.get(sh)
        if not fs or fs.get("n", 0) == 0:
            continue
        ac = fs.get("avg_confidence_per_channel", {})
        lines.append(
            f"| {sh} | {ac.get('speaker_filter', 0):.3f} | "
            f"{ac.get('alias_context', 0):.3f} | "
            f"{ac.get('critical_info', 0):.3f} | "
            f"{ac.get('temporal_tokens', 0):.3f} | "
            f"{ac.get('entity_exact_match', 0):.3f} |"
        )

    # --- Channel agreement vs original ---
    lines.append("\n## Routing agreement vs ORIGINAL\n")
    lines.append(
        "Exact match = the set of firing channels is IDENTICAL to the "
        "original-shape firing set. Confidence cosine = cosine similarity "
        "of the 5-dim channel confidence vector vs the original's.\n"
    )
    lines.append(
        "| Shape | n | Exact firing-set match % | Mean conf-vector cosine vs ORIGINAL |"
    )
    lines.append("|---|---|---|---|")
    for sh in SHAPES:
        a = agreement.get(sh, {})
        if not a:
            continue
        lines.append(
            f"| {sh} | {a.get('n_compared', 0)} | "
            f"{a.get('exact_firing_set_match_pct', 0.0):.1f}% | "
            f"{a.get('mean_confidence_cosine_vs_original', 0.0):.4f} |"
        )

    # --- Sample confidence comparison for 2 queries ---
    lines.append("\n## Sample: LLM confidences across shapes for 2 queries\n")
    sample_ids = [0, 1]  # first two originals
    by_idx_shape: dict[tuple[int, str], dict] = {}
    for sh_rows in rows_by_shape.values():
        for r in sh_rows:
            by_idx_shape[(r["orig_row_index"], r["shape"])] = r

    for oi in sample_ids:
        orig_row = by_idx_shape.get((oi, ORIGINAL_SHAPE))
        if not orig_row:
            continue
        lines.append(f"\n**Q{oi + 1} (original)**: {orig_row['original_question']}\n")
        lines.append(
            "| Shape | Question | " + " | ".join(SUPPLEMENT_NAMES) + " | fired |"
        )
        lines.append("|---|---|" + "---|" * (len(SUPPLEMENT_NAMES) + 1))
        for sh in (ORIGINAL_SHAPE, "CMD", "DRAFT", "META"):
            r = by_idx_shape.get((oi, sh))
            if not r:
                continue
            c = r["metadata"].get("confidences", {}) or {}
            fired = r["metadata"].get("firing_channels", []) or []
            row_parts = [
                sh,
                r["question"][:80],
            ]
            for ch in SUPPLEMENT_NAMES:
                row_parts.append(f"{float(c.get(ch, 0.0)):.2f}")
            row_parts.append(",".join(fired) or "—")
            lines.append("| " + " | ".join(row_parts) + " |")

    # --- Comparison vs other architectures ---
    lines.append("\n## Comparison vs other architectures (LoCoMo K=50)\n")
    lines.append(
        "Numbers for `meta_v2f` / `two_speaker_filter` / `keyword_router` "
        "are from `results/task_shape_adversarial.md`.\n"
    )
    lines.append("| Architecture | ORIG | CMD | DRAFT | META | Worst drop |")
    lines.append("|---|---|---|---|---|---|")
    # Other arch numbers (K=50 arch_r, worst drop). Lifted verbatim.
    prev_rows = [
        ("meta_v2f", 0.8583, 0.7333, 0.8167, 0.7417, 0.1250),
        ("two_speaker_filter", 0.8917, 0.8167, 0.8583, 0.8083, 0.0834),
        ("keyword_router", 0.8583, 0.7333, 0.8167, 0.7417, 0.1250),
    ]
    # gated row
    orig50 = shape_summary.get(ORIGINAL_SHAPE, {}).get("mean_arch_r@50")
    cmd50 = shape_summary.get("CMD", {}).get("mean_arch_r@50")
    drf50 = shape_summary.get("DRAFT", {}).get("mean_arch_r@50")
    mt50 = shape_summary.get("META", {}).get("mean_arch_r@50")
    g_worst = 0.0
    if orig50 is not None:
        for v in (cmd50, drf50, mt50):
            if v is None:
                continue
            g_worst = max(g_worst, orig50 - v)
    lines.append(
        f"| gated_threshold_0.7 | "
        f"{orig50:.4f} | "
        f"{cmd50:.4f} | "
        f"{drf50:.4f} | "
        f"{mt50:.4f} | {g_worst:+.4f} |"
    )
    for name, o, c, d, m, w in prev_rows:
        lines.append(f"| {name} | {o:.4f} | {c:.4f} | {d:.4f} | {m:.4f} | +{w:.4f} |")

    # --- Verdict ---
    lines.append("\n## Verdict\n")
    if g_worst <= 0.02:
        verdict = (
            "**Shape-ROBUST** — ships as production universal retrieval "
            "architecture. LLM confidence scoring reads semantic intent, "
            "not surface verb."
        )
    elif g_worst <= 0.06:
        verdict = (
            "**Partially robust** — within 3–6pp of ORIGINAL. Note which "
            "shape/channel destabilizes."
        )
    else:
        verdict = (
            "**Shape-SENSITIVE** — drops >6pp like meta_v2f. LLM "
            "confidence scoring is also shape-sensitive; a different "
            "routing mechanism is needed."
        )
    lines.append(f"- Worst drop vs ORIGINAL @K=50: **{g_worst:+.4f}**\n")
    lines.append(f"- {verdict}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load store
    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_extended.npz")
    print(f"Loaded {len(store.segments)} segments", flush=True)

    # Load task-shape variants
    with open(VARIANTS_FILE) as f:
        variants = json.load(f)
    rows_by_shape_in: dict[str, list[dict]] = defaultdict(list)
    for v in variants:
        rows_by_shape_in[v["shape"]].append(v)
    print(
        f"Loaded {len(variants)} variants: "
        + ", ".join(f"{sh}={len(rs)}" for sh, rs in rows_by_shape_in.items()),
        flush=True,
    )

    # Load cached originals
    originals = load_cached_originals()
    print(f"Loaded cached ORIGINAL rows: {len(originals)}", flush=True)

    # Build gated_threshold_0.7 (primary)
    client = OpenAI(timeout=60.0, max_retries=3)
    arch = GatedOverlay(store, client=client, threshold=0.7, name="gated_threshold_0.7")

    rows_by_shape: dict[str, list[dict]] = {
        ORIGINAL_SHAPE: originals,
        "CMD": [],
        "DRAFT": [],
        "META": [],
    }

    t_start = time.time()
    total_llm = 0
    for sh in SHAPES:
        qs = rows_by_shape_in[sh]
        t_sh = time.time()
        for i, q in enumerate(qs):
            try:
                row = evaluate_one(arch, q)
            except Exception as e:
                print(f"  ERROR [{sh} {i + 1}/{len(qs)}]: {e}", flush=True)
                import traceback

                traceback.print_exc()
                continue
            rows_by_shape[sh].append(row)
            total_llm += row["llm_calls"]
            if (i + 1) % 5 == 0:
                arch.save_caches()
        arch.save_caches()
        n = len(rows_by_shape[sh])
        if n:
            a20 = sum(r["fair_backfill"]["arch_r@20"] for r in rows_by_shape[sh]) / n
            a50 = sum(r["fair_backfill"]["arch_r@50"] for r in rows_by_shape[sh]) / n
            print(
                f"  {sh} (n={n}): a@20={a20:.4f} a@50={a50:.4f} "
                f"({time.time() - t_sh:.1f}s)",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(
        f"\nTotal elapsed: {elapsed:.1f}s   total LLM calls: {total_llm}",
        flush=True,
    )

    # Aggregate
    shape_summary = {sh: per_shape_recall(rs) for sh, rs in rows_by_shape.items()}
    fire_stats = {sh: channel_fire_stats(rs) for sh, rs in rows_by_shape.items()}
    agreement = channel_agreement_vs_original(rows_by_shape)

    # Save raw
    raw = {
        "arch": "gated_threshold_0.7",
        "n_originals": len(rows_by_shape[ORIGINAL_SHAPE]),
        "n_per_shape": {sh: len(rows_by_shape[sh]) for sh in rows_by_shape},
        "elapsed_s": round(elapsed, 2),
        "shape_summary": shape_summary,
        "channel_fire_stats": fire_stats,
        "routing_agreement_vs_original": agreement,
        "rows_by_shape": rows_by_shape,
    }
    raw_path = RESULTS_DIR / "gated_shape.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"Saved raw: {raw_path}", flush=True)

    md = render_markdown(rows_by_shape, shape_summary, fire_stats, agreement)
    md_path = RESULTS_DIR / "gated_shape.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved report: {md_path}", flush=True)


if __name__ == "__main__":
    main()
