"""Meta-router composition evaluation.

Architectures:
  - meta_router           : primary composition (speaker -> two_speaker, else gated)
  - meta_router_inverted  : control (flipped dispatch)
  - two_speaker_filter    : standalone (shape-robust, zero LLM)
  - gated_threshold_0.7   : standalone (semantic routing)
  - meta_v2f              : baseline

Datasets:
  - locomo_30q + synthetic_19q  at K=20, K=50 (primary)
  - questions_locomo_task_shape (90 variants: CMD/DRAFT/META x 30)   (shape-robustness)

Outputs:
  - results/meta_router.json
  - results/meta_router.md

Dedicated caches (metart_*_cache.json) for any new writes.

Usage:
    uv run python metart_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SHAPE_VARIANTS_FILE = DATA_DIR / "questions_locomo_task_shape.json"
GATED_ORIGINAL_JSON = RESULTS_DIR / "gated_overlay.json"
TWO_SPEAKER_ORIGINAL_JSON = RESULTS_DIR / "two_speaker_filter.json"

BUDGETS = (20, 50)

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
}


# ---------------------------------------------------------------------------
# Dedicated metart_* cache classes. Writes -> metart_*.json only.
# Reads -> shared warm-start caches (read-only).
# ---------------------------------------------------------------------------
METART_EMB_FILE = CACHE_DIR / "metart_embedding_cache.json"
METART_LLM_FILE = CACHE_DIR / "metart_llm_cache.json"

SHARED_EMB_READ = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "agent_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "synth_test_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "antipara_embedding_cache.json",
    "alias_embedding_cache.json",
    "speaker_embedding_cache.json",
    "two_speaker_embedding_cache.json",
    "multich_embedding_cache.json",
    "gated_embedding_cache.json",
    "gatedTS_embedding_cache.json",
    "tasksh_embedding_cache.json",
    "metart_embedding_cache.json",
)
SHARED_LLM_READ = (
    "llm_cache.json",
    "arch_llm_cache.json",
    "agent_llm_cache.json",
    "tree_llm_cache.json",
    "frontier_llm_cache.json",
    "meta_llm_cache.json",
    "optim_llm_cache.json",
    "synth_test_llm_cache.json",
    "bestshot_llm_cache.json",
    "alias_llm_cache.json",
    "speaker_llm_cache.json",
    "two_speaker_llm_cache.json",
    "antipara_llm_cache.json",
    "multich_llm_cache.json",
    "gated_llm_cache.json",
    "gatedTS_llm_cache.json",
    "tasksh_llm_cache.json",
    "metart_llm_cache.json",
)


class MetartEmbeddingCache(EmbeddingCache):
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
        self.cache_file = METART_EMB_FILE
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
        tmp = self.cache_file.parent / (
            self.cache_file.name + f".tmp.{os.getpid()}"
        )
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new_entries = {}


class MetartLLMCache(LLMCache):
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
        self.cache_file = METART_LLM_FILE
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
        tmp = self.cache_file.parent / (
            self.cache_file.name + f".tmp.{os.getpid()}"
        )
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Monkey-patch BOTH arches' cache classes BEFORE importing them, so all cache
# writes land in metart_* files and cannot corrupt peer agents' caches.
# ---------------------------------------------------------------------------
import gated_overlay as _go  # noqa: E402
import two_speaker_filter as _tsf  # noqa: E402


def _gated_emb_init(self):
    MetartEmbeddingCache.__init__(self)


def _gated_llm_init(self):
    MetartLLMCache.__init__(self)


_go.GatedEmbeddingCache.__init__ = _gated_emb_init  # type: ignore[method-assign]
_go.GatedLLMCache.__init__ = _gated_llm_init  # type: ignore[method-assign]
_tsf.TwoSpeakerEmbeddingCache.__init__ = _gated_emb_init  # type: ignore[method-assign]
_tsf.TwoSpeakerLLMCache.__init__ = _gated_llm_init  # type: ignore[method-assign]


# Now safe to import arches. (Our caches will be used via the classes above.)
from best_shot import MetaV2f  # noqa: E402
from gated_overlay import GatedOverlay  # noqa: E402
from meta_router import (  # noqa: E402
    MetaRouter,
    ROUTE_GATED,
    ROUTE_TWO_SPEAKER,
    build_meta_router,
    load_speaker_pairs,
    query_mentions_known_speaker,
)
from two_speaker_filter import TwoSpeakerFilter  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_dataset(ds_name: str) -> tuple[SegmentStore, list[dict]]:
    cfg = DATASETS[ds_name]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        qs = json.load(f)
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    if cfg["max_questions"]:
        qs = qs[: cfg["max_questions"]]
    return store, qs


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


def _cosine_topk(
    store: SegmentStore,
    query_emb: np.ndarray,
    conversation_id: str,
    top_k: int,
) -> list[Segment]:
    result = store.search(
        query_emb, top_k=top_k, conversation_id=conversation_id
    )
    return list(result.segments)


# ---------------------------------------------------------------------------
# Per-arch retrieve wrappers (each returns segments list, metadata dict,
# embed_calls, llm_calls, elapsed).
# ---------------------------------------------------------------------------
def _retrieve_with(arch, arch_name: str, question: str, conv_id: str, K: int):
    """Uniform interface. Returns (segments, metadata, embed_calls,
    llm_calls, elapsed).

    Gated and MetaRouter take K=. TwoSpeakerFilter and MetaV2f ignore K.
    """
    # Reset whichever counters the arch exposes.
    if hasattr(arch, "reset_counters"):
        arch.reset_counters()

    t0 = time.time()
    if arch_name in ("gated_threshold_0.7",):
        res = arch.retrieve(question, conv_id, K=K)
    elif arch_name in ("meta_router", "meta_router_inverted"):
        res = arch.retrieve(question, conv_id, K=K)
    else:
        res = arch.retrieve(question, conv_id)
    elapsed = time.time() - t0

    segments = list(res.segments)
    meta = getattr(res, "metadata", {}) or {}
    embed_calls = getattr(arch, "embed_calls", 0)
    llm_calls = getattr(arch, "llm_calls", 0)
    return segments, meta, embed_calls, llm_calls, elapsed


def evaluate_question(
    arch,
    arch_name: str,
    question: dict,
    store: SegmentStore,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    # For arches whose overlay depends on K (gated_*, meta_router), we run
    # once at max(K) and also at smaller K if they actually parameterize K.
    # Simpler: run once at max(K). Slice in fair_backfill.
    # BUT: gated_overlay and MetaRouter use K to size displacement.
    # Run at each K like gated_eval does.
    results_per_k: dict[int, tuple[list[Segment], dict, int, int, float]] = {}
    if arch_name in ("gated_threshold_0.7", "meta_router", "meta_router_inverted"):
        for K in BUDGETS:
            r = _retrieve_with(arch, arch_name, q_text, conv_id, K)
            results_per_k[K] = r
    else:
        r = _retrieve_with(arch, arch_name, q_text, conv_id, max(BUDGETS))
        for K in BUDGETS:
            results_per_k[K] = r

    # Cosine baseline
    # For cosine we need an embedding. Use arch.embed_text if available,
    # else instantiate a quick cache-aware embed via MetartEmbeddingCache.
    if hasattr(arch, "embed_text"):
        query_emb = arch.embed_text(q_text)
    else:
        # Fallback: use store-level cosine only
        query_emb = None
    if query_emb is not None:
        cosine_segments = _cosine_topk(
            store, query_emb, conv_id, max(BUDGETS)
        )
    else:
        cosine_segments = []

    # Build row. Use K=max metadata as the representative.
    max_K = max(BUDGETS)
    _, meta_max, emb_max, llm_max, elapsed_max = results_per_k[max_K]

    row = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "embed_calls": emb_max,
        "llm_calls": llm_max,
        "time_s": round(elapsed_max, 2),
        "fair_backfill": {},
        "metadata": meta_max,
    }

    for K in BUDGETS:
        segs, _, _, _, _ = results_per_k[K]
        arch_at_K = fair_backfill(segs, cosine_segments, K)
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
# Arch builders
# ---------------------------------------------------------------------------
def build_arch(arch_name: str, store: SegmentStore, client: OpenAI):
    if arch_name == "meta_v2f":
        return MetaV2f(store)
    if arch_name == "two_speaker_filter":
        return TwoSpeakerFilter(store, client=client)
    if arch_name == "gated_threshold_0.7":
        return GatedOverlay(
            store, client=client, threshold=0.7, name="gated_threshold_0.7"
        )
    if arch_name in ("meta_router", "meta_router_inverted"):
        return build_meta_router(arch_name, store, client=client)
    raise KeyError(arch_name)


PRIMARY_ARCHES = (
    "meta_v2f",
    "two_speaker_filter",
    "gated_threshold_0.7",
    "meta_router",
    "meta_router_inverted",
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def summarize(rows: list[dict], arch_name: str, dataset: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"arch": arch_name, "dataset": dataset, "n": 0}
    out: dict = {"arch": arch_name, "dataset": dataset, "n": n}
    for K in BUDGETS:
        b = [r["fair_backfill"][f"baseline_r@{K}"] for r in rows]
        a = [r["fair_backfill"][f"arch_r@{K}"] for r in rows]
        b_mean = sum(b) / n
        a_mean = sum(a) / n
        wins = sum(1 for bv, av in zip(b, a) if av > bv + 0.001)
        losses = sum(1 for bv, av in zip(b, a) if bv > av + 0.001)
        ties = n - wins - losses
        out[f"baseline_r@{K}"] = round(b_mean, 4)
        out[f"arch_r@{K}"] = round(a_mean, 4)
        out[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
        out[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
    out["avg_llm_calls"] = round(
        sum(r["llm_calls"] for r in rows) / n, 2
    )
    out["avg_embed_calls"] = round(
        sum(r["embed_calls"] for r in rows) / n, 2
    )
    out["avg_time_s"] = round(sum(r["time_s"] for r in rows) / n, 2)
    return out


def per_route_recall(rows: list[dict]) -> dict:
    """Split meta_router rows by route and report per-K recall for each."""
    by_route: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        route = (r.get("metadata") or {}).get("route", "unknown")
        by_route[route].append(r)
    out: dict[str, dict] = {}
    n_total = len(rows)
    for route, rs in by_route.items():
        n = len(rs)
        entry: dict = {
            "n": n,
            "pct": round(100.0 * n / max(n_total, 1), 2),
        }
        for K in BUDGETS:
            a = sum(r["fair_backfill"][f"arch_r@{K}"] for r in rs) / max(n, 1)
            b = sum(r["fair_backfill"][f"baseline_r@{K}"] for r in rs) / max(n, 1)
            entry[f"arch_r@{K}"] = round(a, 4)
            entry[f"baseline_r@{K}"] = round(b, 4)
            entry[f"delta_r@{K}"] = round(a - b, 4)
        out[route] = entry
    return out


def head_to_head(
    a_rows: list[dict], b_rows: list[dict], K: int
) -> dict:
    """Per-question W/T/L of a_rows vs b_rows at budget K."""
    b_map = {
        (r["conversation_id"], r["question_index"]): r for r in b_rows
    }
    wins = losses = ties = 0
    for ar in a_rows:
        key = (ar["conversation_id"], ar["question_index"])
        br = b_map.get(key)
        if not br:
            continue
        av = ar["fair_backfill"][f"arch_r@{K}"]
        bv = br["fair_backfill"][f"arch_r@{K}"]
        if av > bv + 0.001:
            wins += 1
        elif bv > av + 0.001:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "ties": ties, "losses": losses}


# ---------------------------------------------------------------------------
# Primary eval driver
# ---------------------------------------------------------------------------
def run_primary_eval(client: OpenAI) -> dict:
    all_results: dict[str, dict] = defaultdict(dict)

    for ds_name in DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\n[{ds_name}] {len(questions)} questions, "
            f"{len(store.segments)} segments",
            flush=True,
        )

        for arch_name in PRIMARY_ARCHES:
            print(f"\n=== {arch_name} on {ds_name} ===", flush=True)
            arch = build_arch(arch_name, store, client)
            rows: list[dict] = []
            t_arch = time.time()
            for i, q in enumerate(questions):
                q_short = q["question"][:55]
                print(
                    f"  [{i+1}/{len(questions)}] "
                    f"{q.get('category', '?')}: {q_short}",
                    flush=True,
                )
                try:
                    row = evaluate_question(arch, arch_name, q, store)
                    rows.append(row)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                sys.stdout.flush()
                if (i + 1) % 5 == 0:
                    if hasattr(arch, "save_caches"):
                        arch.save_caches()
            if hasattr(arch, "save_caches"):
                arch.save_caches()

            summary = summarize(rows, arch_name, ds_name)
            entry: dict = {
                "arch": arch_name,
                "dataset": ds_name,
                "summary": summary,
                "results": rows,
                "elapsed_s": round(time.time() - t_arch, 2),
            }
            if arch_name in ("meta_router", "meta_router_inverted"):
                entry["per_route"] = per_route_recall(rows)
            all_results[arch_name][ds_name] = entry

            print(
                f"  -> r@20={summary['arch_r@20']:.4f} "
                f"r@50={summary['arch_r@50']:.4f} "
                f"avg_llm={summary['avg_llm_calls']:.2f}",
                flush=True,
            )

    # Head-to-head tables for meta_router
    head_to_head_tables: dict[str, dict] = {}
    for ds_name in DATASETS:
        mr_rows = all_results.get("meta_router", {}).get(ds_name, {}).get(
            "results", []
        )
        if not mr_rows:
            continue
        h2h: dict = {}
        for ref_name in (
            "two_speaker_filter",
            "gated_threshold_0.7",
            "meta_router_inverted",
        ):
            ref_rows = all_results.get(ref_name, {}).get(ds_name, {}).get(
                "results", []
            )
            if not ref_rows:
                continue
            h2h[ref_name] = {
                f"K{K}": head_to_head(mr_rows, ref_rows, K)
                for K in BUDGETS
            }
        head_to_head_tables[ds_name] = h2h

    return {
        "primary_results": dict(all_results),
        "head_to_head_meta_router": head_to_head_tables,
    }


# ---------------------------------------------------------------------------
# Task-shape eval
# ---------------------------------------------------------------------------
ORIGINAL_SHAPE = "ORIGINAL"
SHAPES = ("CMD", "DRAFT", "META")


def run_shape_eval(client: OpenAI, primary_locomo_rows: list[dict]) -> dict:
    """Run meta_router on 90 task-shape variants. Reuse primary locomo_30q
    rows as ORIGINAL shape.
    """
    store = SegmentStore(
        data_dir=DATA_DIR, npz_name="segments_extended.npz"
    )
    print(
        f"\n[shape] {len(store.segments)} segments in store", flush=True
    )

    with open(SHAPE_VARIANTS_FILE) as f:
        variants = json.load(f)
    rows_by_shape_in: dict[str, list[dict]] = defaultdict(list)
    for v in variants:
        rows_by_shape_in[v["shape"]].append(v)
    print(
        f"[shape] {len(variants)} variants: "
        + ", ".join(f"{sh}={len(rs)}" for sh, rs in rows_by_shape_in.items()),
        flush=True,
    )

    # Build meta_router once.
    arch = build_meta_router("meta_router", store, client=client)

    # Normalize primary_locomo_rows as ORIGINAL-shape rows.
    originals: list[dict] = []
    for src in primary_locomo_rows:
        qi = src.get("question_index", -1)
        originals.append({
            "orig_row_index": qi,
            "shape": ORIGINAL_SHAPE,
            "conversation_id": src["conversation_id"],
            "category": src["category"],
            "question": src["question"],
            "original_question": src["question"],
            "source_chat_ids": src["source_chat_ids"],
            "num_source_turns": src["num_source_turns"],
            "embed_calls": src.get("embed_calls", 0),
            "llm_calls": src.get("llm_calls", 0),
            "time_s": src.get("time_s", 0.0),
            "fair_backfill": src["fair_backfill"],
            "metadata": src.get("metadata", {}),
        })

    rows_by_shape: dict[str, list[dict]] = {
        ORIGINAL_SHAPE: originals,
        "CMD": [],
        "DRAFT": [],
        "META": [],
    }

    t_start = time.time()
    for sh in SHAPES:
        qs = rows_by_shape_in[sh]
        t_sh = time.time()
        print(f"\n  === shape {sh} ({len(qs)} q) ===", flush=True)
        for i, q in enumerate(qs):
            print(
                f"  [{sh} {i+1}/{len(qs)}] "
                f"{q.get('category', '?')}: {q['question'][:60]}",
                flush=True,
            )
            try:
                row = evaluate_question(arch, "meta_router", q, store)
                # Enrich with shape metadata.
                row["orig_row_index"] = q.get("orig_row_index", -1)
                row["shape"] = sh
                row["original_question"] = q.get(
                    "original_question", q["question"]
                )
                rows_by_shape[sh].append(row)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
            sys.stdout.flush()
            if (i + 1) % 5 == 0:
                arch.save_caches()
        arch.save_caches()
        n = len(rows_by_shape[sh])
        if n:
            a20 = sum(
                r["fair_backfill"]["arch_r@20"] for r in rows_by_shape[sh]
            ) / n
            a50 = sum(
                r["fair_backfill"]["arch_r@50"] for r in rows_by_shape[sh]
            ) / n
            print(
                f"  {sh} (n={n}): a@20={a20:.4f} a@50={a50:.4f} "
                f"({time.time() - t_sh:.1f}s)",
                flush=True,
            )

    # Aggregate per shape
    shape_summary: dict[str, dict] = {}
    for sh, rs in rows_by_shape.items():
        n = len(rs)
        if n == 0:
            continue
        entry: dict = {"n": n}
        for K in BUDGETS:
            a = sum(r["fair_backfill"][f"arch_r@{K}"] for r in rs) / n
            b = sum(r["fair_backfill"][f"baseline_r@{K}"] for r in rs) / n
            entry[f"mean_arch_r@{K}"] = round(a, 4)
            entry[f"mean_baseline_r@{K}"] = round(b, 4)
            entry[f"mean_delta_r@{K}"] = round(a - b, 4)
        # Route distribution for this shape
        route_counts: dict[str, int] = defaultdict(int)
        for r in rs:
            route = (r.get("metadata") or {}).get("route", "unknown")
            route_counts[route] += 1
        entry["route_distribution"] = dict(route_counts)
        shape_summary[sh] = entry

    return {
        "shape_summary": shape_summary,
        "rows_by_shape": rows_by_shape,
        "elapsed_s": round(time.time() - t_start, 2),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def render_report(
    primary: dict,
    shape: dict,
    pairs: dict[str, dict[str, str]],
    baselines_from_disk: dict,
) -> str:
    L: list[str] = []
    L.append("# Meta-router composition\n")
    L.append(
        "Regex-based zero-LLM dispatch between `two_speaker_filter` "
        "(shape-robust, zero LLM) and `gated_overlay` v1 "
        "(confidence_threshold=0.7). If the query mentions a known "
        "conversation participant's first name -> two_speaker_filter; "
        "otherwise -> gated_overlay.\n"
    )

    # --- Route distribution ---
    L.append("\n## Route distribution\n")
    L.append(
        "| Dataset | n | two_speaker % | gated % |"
    )
    L.append("|---|---:|---:|---:|")
    for ds_name in DATASETS:
        mr = primary["primary_results"].get("meta_router", {}).get(
            ds_name, {}
        )
        pr = mr.get("per_route", {}) if mr else {}
        ts_pct = pr.get(ROUTE_TWO_SPEAKER, {}).get("pct", 0.0)
        g_pct = pr.get(ROUTE_GATED, {}).get("pct", 0.0)
        n = (mr.get("summary", {}) or {}).get("n", 0)
        L.append(
            f"| {ds_name} | {n} | {ts_pct:.1f}% | {g_pct:.1f}% |"
        )
    L.append("")

    # --- Recall matrix ---
    L.append("\n## Recall matrix (fair-backfill)\n")
    L.append(
        "| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | "
        "Δ@50 | avg LLM |"
    )
    L.append("|" + ("---|" * 9))
    for arch_name in PRIMARY_ARCHES:
        for ds_name in DATASETS:
            s = primary["primary_results"].get(arch_name, {}).get(
                ds_name, {}
            ).get("summary")
            if not s:
                continue
            L.append(
                f"| {arch_name} | {ds_name} | "
                f"{s['baseline_r@20']:.4f} | {s['arch_r@20']:.4f} | "
                f"{s['delta_r@20']:+.4f} | "
                f"{s['baseline_r@50']:.4f} | {s['arch_r@50']:.4f} | "
                f"{s['delta_r@50']:+.4f} | "
                f"{s['avg_llm_calls']:.2f} |"
            )
    L.append("")

    # --- Per-route recall ---
    L.append("\n## Per-route recall (meta_router slices)\n")
    L.append(
        "For each dataset, split meta_router results by which sub-arch ran.\n"
    )
    L.append(
        "| Dataset | Route | n | pct | arch@20 | arch@50 | Δ@20 | Δ@50 |"
    )
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for ds_name in DATASETS:
        mr = primary["primary_results"].get("meta_router", {}).get(
            ds_name, {}
        )
        pr = mr.get("per_route", {}) if mr else {}
        for route in (ROUTE_TWO_SPEAKER, ROUTE_GATED):
            e = pr.get(route)
            if not e:
                continue
            L.append(
                f"| {ds_name} | {route} | {e['n']} | {e['pct']:.1f}% | "
                f"{e['arch_r@20']:.4f} | {e['arch_r@50']:.4f} | "
                f"{e['delta_r@20']:+.4f} | {e['delta_r@50']:+.4f} |"
            )
    L.append("")

    # --- Head-to-head W/T/L ---
    L.append("\n## meta_router head-to-head (per-question W/T/L)\n")
    L.append("| Dataset | vs | K | W/T/L |")
    L.append("|---|---|---|---|")
    for ds_name in DATASETS:
        h2h = primary["head_to_head_meta_router"].get(ds_name, {})
        for ref_name, by_K in h2h.items():
            for K in BUDGETS:
                t = by_K.get(f"K{K}", {})
                if not t:
                    continue
                L.append(
                    f"| {ds_name} | {ref_name} | K={K} | "
                    f"{t['wins']}/{t['ties']}/{t['losses']} |"
                )
    L.append("")

    # --- Shape-robustness ---
    L.append("\n## Shape-robustness (LoCoMo task-shape variants)\n")
    ss = shape["shape_summary"]
    L.append(
        "meta_router recall on the 30 LoCoMo originals (ORIGINAL reused "
        "from this run's primary eval) vs the 30 CMD / 30 DRAFT / 30 META "
        "rewrites.\n"
    )
    L.append(
        "| Shape | n | arch@20 | arch@50 | Drop vs ORIG @20 | Drop vs ORIG @50 | routes |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---|")
    orig = ss.get(ORIGINAL_SHAPE, {})
    for sh in (ORIGINAL_SHAPE, "CMD", "DRAFT", "META"):
        e = ss.get(sh)
        if not e:
            continue
        if sh == ORIGINAL_SHAPE or not orig:
            d20 = d50 = 0.0
        else:
            d20 = orig.get("mean_arch_r@20", 0) - e.get("mean_arch_r@20", 0)
            d50 = orig.get("mean_arch_r@50", 0) - e.get("mean_arch_r@50", 0)
        rd = e.get("route_distribution", {})
        route_str = ", ".join(
            f"{r}={n}" for r, n in sorted(rd.items())
        )
        L.append(
            f"| {sh} | {e['n']} | {e['mean_arch_r@20']:.4f} | "
            f"{e['mean_arch_r@50']:.4f} | {d20:+.4f} | {d50:+.4f} | "
            f"{route_str} |"
        )
    L.append("")

    # Compare meta_router shape to known gated/two_speaker shape numbers.
    L.append(
        "\n### Comparison vs prior arches on shape-robustness (LoCoMo @K=50)\n"
    )
    L.append(
        "Numbers for `two_speaker_filter` and `gated_threshold_0.7` lifted "
        "from `results/gated_shape.md`.\n"
    )
    L.append(
        "| Architecture | ORIG | CMD | DRAFT | META | Worst drop |"
    )
    L.append("|---|---:|---:|---:|---:|---:|")
    # meta_router
    o50 = ss.get(ORIGINAL_SHAPE, {}).get("mean_arch_r@50")
    c50 = ss.get("CMD", {}).get("mean_arch_r@50")
    d50_ = ss.get("DRAFT", {}).get("mean_arch_r@50")
    m50 = ss.get("META", {}).get("mean_arch_r@50")
    worst = 0.0
    if o50 is not None:
        for v in (c50, d50_, m50):
            if v is not None:
                worst = max(worst, o50 - v)
    L.append(
        f"| meta_router | {o50:.4f} | {c50:.4f} | {d50_:.4f} | "
        f"{m50:.4f} | +{worst:.4f} |"
    )
    # Priors from gated_shape.md
    prior_rows = [
        ("two_speaker_filter", 0.8917, 0.8167, 0.8583, 0.8083, 0.0834),
        ("gated_threshold_0.7", 0.8917, 0.7333, 0.8167, 0.7417, 0.1584),
        ("meta_v2f", 0.8583, 0.7333, 0.8167, 0.7417, 0.1250),
    ]
    for name, o, c, d, m, w in prior_rows:
        L.append(
            f"| {name} | {o:.4f} | {c:.4f} | {d:.4f} | {m:.4f} | "
            f"+{w:.4f} |"
        )
    L.append("")

    # --- Speaker pairs used ---
    L.append("\n## Known speaker pairs (from conversation_two_speakers.json)\n")
    L.append("| Conversation | user | assistant |")
    L.append("|---|---|---|")
    for cid, p in sorted(pairs.items()):
        L.append(f"| {cid} | {p.get('user')} | {p.get('assistant')} |")
    L.append("")

    # --- Verdict ---
    L.append("\n## Verdict\n")
    pr = primary["primary_results"]
    lc_mr = pr.get("meta_router", {}).get("locomo_30q", {}).get(
        "summary", {}
    )
    lc_ts = pr.get("two_speaker_filter", {}).get("locomo_30q", {}).get(
        "summary", {}
    )
    lc_g = pr.get("gated_threshold_0.7", {}).get("locomo_30q", {}).get(
        "summary", {}
    )
    lc_inv = pr.get("meta_router_inverted", {}).get("locomo_30q", {}).get(
        "summary", {}
    )
    if lc_mr and lc_ts and lc_g:
        mr50 = lc_mr.get("arch_r@50", 0)
        ts50 = lc_ts.get("arch_r@50", 0)
        g50 = lc_g.get("arch_r@50", 0)
        inv50 = lc_inv.get("arch_r@50", 0) if lc_inv else 0.0
        best = max(ts50, g50)
        L.append(
            f"- LoCoMo K=50: meta_router={mr50:.4f}, "
            f"two_speaker_filter={ts50:.4f}, gated={g50:.4f}, "
            f"meta_router_inverted={inv50:.4f}"
        )
        if mr50 > best + 0.005:
            decision = (
                f"**SHIP meta_router as primary** — beats max(two_speaker, "
                f"gated)={best:.4f} by {mr50 - best:+.4f} at K=50 on LoCoMo."
            )
        elif abs(mr50 - best) <= 0.005:
            decision = (
                f"**SHIP meta_router for cost** — ties max(two_speaker, "
                f"gated)={best:.4f} at K=50, but saves the gated LLM call "
                f"on the {pr.get('meta_router', {}).get('locomo_30q', {}).get('per_route', {}).get(ROUTE_TWO_SPEAKER, {}).get('pct', 0):.0f}% "
                f"of queries routed to two_speaker."
            )
        else:
            decision = (
                f"**KEEP standalone** — meta_router ({mr50:.4f}) loses to "
                f"best standalone ({best:.4f}) by {mr50 - best:+.4f}."
            )
        if inv50 > mr50 + 0.005:
            decision += (
                " WARNING: meta_router_inverted beats meta_router — "
                "dispatch logic may be wrong. Investigate."
            )
        L.append(f"- {decision}")
    L.append("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(timeout=60.0, max_retries=3)

    t_start = time.time()
    print("=" * 70, flush=True)
    print("PRIMARY EVAL (LoCoMo-30 + synthetic-19, K=20, K=50)", flush=True)
    print("=" * 70, flush=True)
    primary = run_primary_eval(client)

    print("\n" + "=" * 70, flush=True)
    print("SHAPE EVAL (LoCoMo task-shape 90 variants)", flush=True)
    print("=" * 70, flush=True)
    primary_locomo_rows = primary["primary_results"].get(
        "meta_router", {}
    ).get("locomo_30q", {}).get("results", [])
    shape = run_shape_eval(client, primary_locomo_rows)

    pairs = load_speaker_pairs()

    # Save raw
    raw = {
        "primary": primary,
        "shape": shape,
        "speaker_pairs": pairs,
        "total_elapsed_s": round(time.time() - t_start, 2),
    }
    raw_path = RESULTS_DIR / "meta_router.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved raw: {raw_path}", flush=True)

    md = render_report(primary, shape, pairs, baselines_from_disk={})
    md_path = RESULTS_DIR / "meta_router.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved report: {md_path}", flush=True)

    # Final console summary
    print("\n" + "=" * 80)
    print("META-ROUTER SUMMARY")
    print("=" * 80)
    for arch_name in PRIMARY_ARCHES:
        for ds_name in DATASETS:
            s = primary["primary_results"].get(arch_name, {}).get(
                ds_name, {}
            ).get("summary")
            if not s:
                continue
            print(
                f"{arch_name:24s} {ds_name:14s} "
                f"a@20={s['arch_r@20']:.4f} a@50={s['arch_r@50']:.4f} "
                f"llm={s['avg_llm_calls']:.2f}"
            )

    ss = shape["shape_summary"]
    print("\nshape (meta_router):")
    for sh in (ORIGINAL_SHAPE, "CMD", "DRAFT", "META"):
        e = ss.get(sh)
        if not e:
            continue
        print(
            f"  {sh:10s} n={e['n']:<3d} "
            f"a@20={e['mean_arch_r@20']:.4f} a@50={e['mean_arch_r@50']:.4f}"
        )


if __name__ == "__main__":
    main()
