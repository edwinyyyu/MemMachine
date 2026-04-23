"""Chunk-level (sentence) retrieval evaluation.

Tests whether embedding conversation turns at sentence granularity changes
retrieval quality on our benchmark datasets. Compares three granularities:

    turn       - current behaviour, one embedding per turn
    sentence   - one embedding per sentence; search returns sentences which
                 are deduplicated to parent turns for recall
    combined   - both turn- and sentence-level embeddings pooled together;
                 still dedup to parent turns

Architectures evaluated: cosine baseline, V15Control, MetaV2f. All recalls
are computed at K=20 and K=50 unique turns.

Usage:
    uv run python chunk_level_eval.py [--datasets synthetic,puzzle,...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    RetrievalResult,
)
from best_shot import (
    BestshotBase,
    BestshotEmbeddingCache,
    BestshotLLMCache,
    V15Control,
    MetaV2f,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = (20, 50)

DATASETS: dict[str, dict] = {
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
    "locomo_30q": {
        "npz": "segments_extended.npz",
        "questions": "questions_extended.json",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "max_questions": 30,
    },
}


# ---------------------------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?\uff1f\uff01\u3002])\s+")


def extract_sentences_ordered(text: str) -> list[str]:
    """Split text into sentences preserving order. Minimal, no NLTK dependency.

    Splits on terminating punctuation (.!? and fullwidth equivalents) followed
    by whitespace, and also on line breaks. Drops tokens containing no
    alphanumeric characters. If no terminator fires, returns the whole text.
    """
    if not text or not text.strip():
        return []
    parts: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        for chunk in _SENT_SPLIT.split(line):
            chunk = chunk.strip()
            if chunk and any(c.isalnum() for c in chunk):
                parts.append(chunk)
    if not parts:
        stripped = text.strip()
        if stripped and any(c.isalnum() for c in stripped):
            return [stripped]
        return []
    return parts


# ---------------------------------------------------------------------------
# Chunk embedding cache
# ---------------------------------------------------------------------------
class ChunkEmbeddingCache(EmbeddingCache):
    """Reads from chunk cache + all existing bestshot caches, writes chunk."""

    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        # Read bestshot-style unified pool plus our chunk cache
        for name in (
            "embedding_cache.json",
            "arch_embedding_cache.json",
            "bestshot_embedding_cache.json",
            "agent_embedding_cache.json",
            "frontier_embedding_cache.json",
            "meta_embedding_cache.json",
            "optim_embedding_cache.json",
            "chunk_embed_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "chunk_embed_cache.json"
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
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


def _batch_embed(
    client: OpenAI,
    cache: ChunkEmbeddingCache,
    texts: Iterable[str],
    batch_size: int = 100,
) -> list[np.ndarray]:
    """Embed texts, filling from cache first. Returns one ndarray per input."""
    texts = list(texts)
    out: list[np.ndarray | None] = [None] * len(texts)
    missing: list[tuple[int, str]] = []
    for i, t in enumerate(texts):
        cached = cache.get(t)
        if cached is not None:
            out[i] = cached
        else:
            missing.append((i, t))
    if missing:
        print(
            f"    Embedding {len(missing)} new texts "
            f"(cache hits: {len(texts) - len(missing)})...",
            flush=True,
        )
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            batch_texts = [t[:8000] for _, t in batch]
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
            for (idx, t), item in zip(batch, resp.data):
                vec = np.array(item.embedding, dtype=np.float32)
                cache.put(t, vec)
                out[idx] = vec
        cache.save()
    return [v for v in out if v is not None]  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------
class ChunkedStore(SegmentStore):
    """SegmentStore variant where each row may be a sentence of a turn.

    Compatible with V15Control / MetaV2f / cosine baseline. Each "segment"
    has turn_id == parent turn's id so that dedup by turn_id works for recall.
    """

    def __init__(
        self,
        *,
        embeddings: np.ndarray,
        conversation_ids: np.ndarray,
        turn_ids: np.ndarray,
        roles: np.ndarray,
        texts: np.ndarray,
    ) -> None:
        self.embeddings = embeddings.astype(np.float32)
        self.conversation_ids = np.asarray(conversation_ids)
        self.turn_ids = np.asarray(turn_ids, dtype=np.int64)
        self.roles = np.asarray(roles)
        self.texts = np.asarray(texts)

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.normalized_embeddings = self.embeddings / norms

        self.segments = [
            Segment(
                conversation_id=str(self.conversation_ids[i]),
                turn_id=int(self.turn_ids[i]),
                role=str(self.roles[i]),
                text=str(self.texts[i]),
                index=i,
            )
            for i in range(len(self.texts))
        ]

        # For neighbor lookup: map (conv_id, turn_id) -> first row index with
        # that turn (only used if someone calls get_neighbors, which V15 and
        # MetaV2f don't).
        self._turn_index: dict[str, dict[int, int]] = {}
        for i, seg in enumerate(self.segments):
            self._turn_index.setdefault(seg.conversation_id, {}).setdefault(
                seg.turn_id, i
            )

    # search / get_neighbors inherited from SegmentStore work as-is.


def build_turn_store(ds_cfg: dict) -> SegmentStore:
    """Load the existing turn-level embedding npz as the baseline store."""
    return SegmentStore(data_dir=DATA_DIR, npz_name=ds_cfg["npz"])


def build_sentence_store(
    turn_store: SegmentStore,
    ds_filter,
    client: OpenAI,
    cache: ChunkEmbeddingCache,
) -> ChunkedStore:
    """Split each turn into sentences, embed them, return a ChunkedStore.

    If ds_filter is provided, only includes turns whose conversation_id
    passes the filter (used for locomo slice of extended npz).
    """
    sent_texts: list[str] = []
    sent_conv_ids: list[str] = []
    sent_turn_ids: list[int] = []
    sent_roles: list[str] = []
    parent_idx: list[int] = []

    for seg in turn_store.segments:
        if ds_filter and not ds_filter(seg.conversation_id):
            continue
        sentences = extract_sentences_ordered(seg.text)
        if not sentences:
            continue
        for s in sentences:
            sent_texts.append(s)
            sent_conv_ids.append(seg.conversation_id)
            sent_turn_ids.append(seg.turn_id)
            sent_roles.append(seg.role)
            parent_idx.append(seg.index)

    vecs = _batch_embed(client, cache, sent_texts)
    embeddings = np.stack(vecs) if vecs else np.zeros((0, 1536), dtype=np.float32)

    return ChunkedStore(
        embeddings=embeddings,
        conversation_ids=np.array(sent_conv_ids),
        turn_ids=np.array(sent_turn_ids, dtype=np.int64),
        roles=np.array(sent_roles),
        texts=np.array(sent_texts),
    )


def build_combined_store(
    turn_store: SegmentStore,
    sentence_store: ChunkedStore,
    ds_filter,
) -> ChunkedStore:
    """Stack turn-level and sentence-level rows into a single pool."""
    if ds_filter:
        keep = np.array(
            [ds_filter(str(c)) for c in turn_store.conversation_ids], dtype=bool
        )
    else:
        keep = np.ones(len(turn_store.texts), dtype=bool)

    turn_emb = turn_store.embeddings[keep].astype(np.float32)
    turn_cids = np.asarray(turn_store.conversation_ids)[keep]
    turn_tids = np.asarray(turn_store.turn_ids)[keep].astype(np.int64)
    turn_roles = np.asarray(turn_store.roles)[keep]
    turn_texts = np.asarray(turn_store.texts)[keep]

    embeddings = np.concatenate([turn_emb, sentence_store.embeddings], axis=0)
    conv_ids = np.concatenate([turn_cids, sentence_store.conversation_ids])
    turn_ids = np.concatenate([turn_tids, sentence_store.turn_ids])
    roles = np.concatenate([turn_roles, sentence_store.roles])
    texts = np.concatenate([turn_texts, sentence_store.texts])

    return ChunkedStore(
        embeddings=embeddings,
        conversation_ids=conv_ids,
        turn_ids=turn_ids,
        roles=roles,
        texts=texts,
    )


def build_filtered_turn_store(
    turn_store: SegmentStore,
    ds_filter,
) -> ChunkedStore:
    """Restrict a turn store to a subset of conversations (same embeddings)."""
    if ds_filter is None:
        # Re-wrap as ChunkedStore so both paths behave identically.
        return ChunkedStore(
            embeddings=turn_store.embeddings,
            conversation_ids=turn_store.conversation_ids,
            turn_ids=turn_store.turn_ids,
            roles=turn_store.roles,
            texts=turn_store.texts,
        )
    keep = np.array(
        [ds_filter(str(c)) for c in turn_store.conversation_ids], dtype=bool
    )
    return ChunkedStore(
        embeddings=turn_store.embeddings[keep],
        conversation_ids=np.asarray(turn_store.conversation_ids)[keep],
        turn_ids=np.asarray(turn_store.turn_ids)[keep],
        roles=np.asarray(turn_store.roles)[keep],
        texts=np.asarray(turn_store.texts)[keep],
    )


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
def dedup_by_turn(segments: list[Segment], limit: int) -> list[Segment]:
    """Keep segments in order, dropping duplicate (conv_id, turn_id). Stop at limit."""
    seen: set[tuple[str, int]] = set()
    out: list[Segment] = []
    for s in segments:
        key = (s.conversation_id, s.turn_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def compute_recall(retrieved_turn_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_turn_ids & source_ids) / len(source_ids)


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------
def eval_cosine(
    store: ChunkedStore,
    question: dict,
    client_cache: ChunkEmbeddingCache,
    client: OpenAI,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    # Query embedding via shared chunk cache
    cached = client_cache.get(q_text)
    if cached is None:
        resp = client.embeddings.create(model=EMBED_MODEL, input=[q_text[:8000]])
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        client_cache.put(q_text, vec)
    else:
        vec = cached

    # Retrieve more than max(BUDGETS) because multiple sentences can share turns.
    # 5x buffer is safe since avg sentences per turn stays below 4 in all our
    # datasets other than BEAM (not in scope here).
    raw = store.search(
        vec, top_k=max(BUDGETS) * 5, conversation_id=conv_id
    )
    out: dict = {
        "category": question.get("category"),
        "num_source": len(source_ids),
    }
    for K in BUDGETS:
        deduped = dedup_by_turn(raw.segments, K)
        out[f"r@{K}"] = compute_recall({s.turn_id for s in deduped}, source_ids)
    return out


def eval_arch(
    arch: BestshotBase,
    question: dict,
) -> dict:
    """Run an architecture on a question, compute recall at budgets using
    turn-dedup of the returned pool. Also falls back to cosine top-K for
    backfill if arch returned fewer than K unique turns."""
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Fall-back cosine top-K (in same store) to backfill if arch produced
    # fewer than K unique turns after dedup.
    query_emb = arch.embed_text(q_text)
    cosine_raw = arch.store.search(
        query_emb, top_k=max(BUDGETS) * 5, conversation_id=conv_id
    )
    cosine_segments = list(cosine_raw.segments)

    out: dict = {
        "category": question.get("category"),
        "num_source": len(source_ids),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
    }
    for K in BUDGETS:
        # Dedup arch's returned segments to unique turns (preserving order)
        deduped = dedup_by_turn(list(result.segments), K)
        if len(deduped) < K:
            # Backfill with cosine top-K to match budget
            existing = {(s.conversation_id, s.turn_id) for s in deduped}
            for cs in cosine_segments:
                key = (cs.conversation_id, cs.turn_id)
                if key in existing:
                    continue
                deduped.append(cs)
                existing.add(key)
                if len(deduped) >= K:
                    break
        out[f"r@{K}"] = compute_recall({s.turn_id for s in deduped}, source_ids)
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize_rows(rows: list[dict]) -> dict:
    n = len(rows)
    s: dict = {"n": n}
    for K in BUDGETS:
        vals = [r[f"r@{K}"] for r in rows]
        s[f"r@{K}"] = round(sum(vals) / n, 4) if n else 0.0
    return s


def summarize_by_category(rows: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r.get("category", "unknown")].append(r)
    return {cat: summarize_rows(rs) for cat, rs in sorted(by.items())}


def paired_wins(
    baseline_rows: list[dict], treatment_rows: list[dict], K: int
) -> tuple[int, int, int]:
    wins = losses = ties = 0
    for b, t in zip(baseline_rows, treatment_rows):
        bv = b[f"r@{K}"]
        tv = t[f"r@{K}"]
        if tv > bv + 0.001:
            wins += 1
        elif bv > tv + 0.001:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------
def load_questions(ds_cfg: dict) -> list[dict]:
    with open(DATA_DIR / ds_cfg["questions"]) as f:
        questions = json.load(f)
    if ds_cfg["filter"]:
        questions = [q for q in questions if ds_cfg["filter"](q)]
    if ds_cfg["max_questions"]:
        questions = questions[: ds_cfg["max_questions"]]
    return questions


def conversation_filter_for(ds_name: str):
    """Return a filter on conversation_id strings matching the dataset."""
    if ds_name == "locomo_30q":
        return lambda cid: str(cid).startswith("locomo_")
    return None


def length_stats(store: SegmentStore, conv_filter) -> dict:
    if conv_filter:
        keep = [conv_filter(str(c)) for c in store.conversation_ids]
        texts = [t for t, k in zip(store.texts, keep) if k]
    else:
        texts = list(store.texts)
    lens = [len(str(t)) for t in texts]
    n = len(lens)
    if not n:
        return {}
    multi_sent = 0
    sent_total = 0
    for t in texts:
        sents = extract_sentences_ordered(str(t))
        sent_total += len(sents)
        if len(sents) >= 2:
            multi_sent += 1
    return {
        "n_turns": n,
        "avg_codepoints": round(sum(lens) / n, 1),
        "max_codepoints": max(lens),
        "p95_codepoints": int(np.percentile(lens, 95)),
        "turns_over_300cp": sum(1 for x in lens if x > 300),
        "multi_sentence_turns": multi_sent,
        "avg_sents_per_turn": round(sent_total / n, 2),
        "total_sentences": sent_total,
    }


def run_dataset(
    ds_name: str,
    client: OpenAI,
    chunk_cache: ChunkEmbeddingCache,
) -> dict:
    ds_cfg = DATASETS[ds_name]
    print(f"\n{'=' * 70}\nDataset: {ds_name}\n{'=' * 70}", flush=True)

    questions = load_questions(ds_cfg)
    raw_turn_store = build_turn_store(ds_cfg)
    conv_filter = conversation_filter_for(ds_name)

    stats = length_stats(raw_turn_store, conv_filter)
    print(f"Length stats: {stats}", flush=True)

    # Build the three stores
    print("Building turn store...", flush=True)
    turn_store = build_filtered_turn_store(raw_turn_store, conv_filter)
    print(
        f"  turn rows: {len(turn_store.texts)}",
        flush=True,
    )

    print("Building sentence store (may embed new sentences)...", flush=True)
    sentence_store = build_sentence_store(
        raw_turn_store, conv_filter, client, chunk_cache
    )
    print(
        f"  sentence rows: {len(sentence_store.texts)}",
        flush=True,
    )

    print("Building combined store...", flush=True)
    combined_store = build_combined_store(
        raw_turn_store, sentence_store, conv_filter
    )
    print(
        f"  combined rows: {len(combined_store.texts)}",
        flush=True,
    )

    stores: dict[str, ChunkedStore] = {
        "turn": turn_store,
        "sentence": sentence_store,
        "combined": combined_store,
    }

    results: dict = {
        "dataset": ds_name,
        "num_questions": len(questions),
        "length_stats": stats,
        "per_granularity": {},
    }

    # Cosine + architectures on each granularity
    arch_classes = [("cosine", None), ("v15_control", V15Control), ("meta_v2f", MetaV2f)]

    # For each granularity, run all architectures (cosine + v15 + meta_v2f)
    rows_by_granularity: dict[str, dict[str, list[dict]]] = {}
    for gname, store in stores.items():
        print(f"\n-- Granularity: {gname} (rows={len(store.texts)}) --", flush=True)
        rows_by_granularity[gname] = {}
        for arch_name, arch_cls in arch_classes:
            print(f"  {arch_name}:", flush=True)
            rows: list[dict] = []
            if arch_cls is None:
                for i, q in enumerate(questions):
                    row = eval_cosine(store, q, chunk_cache, client)
                    rows.append(row)
                    if (i + 1) % 10 == 0:
                        chunk_cache.save()
            else:
                arch = arch_cls(store)
                for i, q in enumerate(questions):
                    try:
                        row = eval_arch(arch, q)
                    except Exception as e:
                        print(f"    ERROR on q[{i}]: {e}", flush=True)
                        row = {
                            "category": q.get("category"),
                            "num_source": len(q.get("source_chat_ids", [])),
                            "embed_calls": 0,
                            "llm_calls": 0,
                            "time_s": 0.0,
                            "error": str(e),
                        }
                        for K in BUDGETS:
                            row[f"r@{K}"] = 0.0
                    rows.append(row)
                    if (i + 1) % 10 == 0:
                        arch.save_caches()
                arch.save_caches()
            chunk_cache.save()
            rows_by_granularity[gname][arch_name] = rows
            summ = summarize_rows(rows)
            print(
                f"    n={summ['n']}  "
                + "  ".join(f"r@{K}={summ[f'r@{K}']:.3f}" for K in BUDGETS),
                flush=True,
            )

    # Build per-granularity summary with by-category breakdown
    for gname, arch_rows in rows_by_granularity.items():
        g_entry: dict = {"num_rows": len(stores[gname].texts), "architectures": {}}
        for arch_name, rows in arch_rows.items():
            g_entry["architectures"][arch_name] = {
                "summary": summarize_rows(rows),
                "category_breakdown": summarize_by_category(rows),
                "rows": rows,
            }
        results["per_granularity"][gname] = g_entry

    # Head-to-head paired comparisons: sentence/combined vs turn
    head2head: dict = {}
    for arch_name in ["cosine", "v15_control", "meta_v2f"]:
        head2head[arch_name] = {}
        turn_rows = rows_by_granularity["turn"][arch_name]
        for gname in ["sentence", "combined"]:
            other = rows_by_granularity[gname][arch_name]
            entry: dict = {}
            for K in BUDGETS:
                w, t, l = paired_wins(turn_rows, other, K)
                entry[f"W/T/L@{K}"] = f"{w}/{t}/{l}"
                base_mean = sum(r[f"r@{K}"] for r in turn_rows) / max(len(turn_rows), 1)
                new_mean = sum(r[f"r@{K}"] for r in other) / max(len(other), 1)
                entry[f"turn_r@{K}"] = round(base_mean, 4)
                entry[f"{gname}_r@{K}"] = round(new_mean, 4)
                entry[f"delta_r@{K}"] = round(new_mean - base_mean, 4)
            head2head[arch_name][gname] = entry
    results["head_to_head_vs_turn"] = head2head
    return results


def print_final_table(all_results: dict) -> None:
    print("\n" + "=" * 100)
    print("CHUNK-LEVEL vs TURN-LEVEL SUMMARY (recall@K, deduped by turn)")
    print("=" * 100)
    header = (
        f"{'Dataset':<14s} {'Arch':<14s} {'Gran':<10s} "
        f"{'r@20':>8s} {'r@50':>8s}"
    )
    print(header)
    print("-" * len(header))
    for ds_name, res in all_results.items():
        for arch_name in ["cosine", "v15_control", "meta_v2f"]:
            for gname in ["turn", "sentence", "combined"]:
                s = res["per_granularity"][gname]["architectures"][arch_name][
                    "summary"
                ]
                print(
                    f"{ds_name:<14s} {arch_name:<14s} {gname:<10s} "
                    f"{s['r@20']:>8.3f} {s['r@50']:>8.3f}"
                )
        print("-" * len(header))

    print("\nHead-to-head (sentence vs turn, combined vs turn)")
    print(
        f"{'Dataset':<14s} {'Arch':<14s} {'Variant':<10s} "
        f"{'d@20':>7s} {'W/T/L@20':>10s} {'d@50':>7s} {'W/T/L@50':>10s}"
    )
    for ds_name, res in all_results.items():
        for arch_name in ["cosine", "v15_control", "meta_v2f"]:
            for gname in ["sentence", "combined"]:
                e = res["head_to_head_vs_turn"][arch_name][gname]
                print(
                    f"{ds_name:<14s} {arch_name:<14s} {gname:<10s} "
                    f"{e['delta_r@20']:>+7.3f} {e['W/T/L@20']:>10s} "
                    f"{e['delta_r@50']:>+7.3f} {e['W/T/L@50']:>10s}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS.keys()),
        help="Comma-separated dataset names",
    )
    args = parser.parse_args()
    ds_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for n in ds_names:
        if n not in DATASETS:
            print(f"Unknown dataset: {n}. Available: {list(DATASETS)}")
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(timeout=60.0)
    chunk_cache = ChunkEmbeddingCache()

    all_results: dict = {}
    for ds_name in ds_names:
        res = run_dataset(ds_name, client, chunk_cache)
        all_results[ds_name] = res
        out_path = RESULTS_DIR / f"chunk_{ds_name}.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  Saved: {out_path}", flush=True)

    agg_path = RESULTS_DIR / "chunk_summary.json"
    with open(agg_path, "w") as f:
        # Drop per-row heavy data from aggregate summary
        compact = {}
        for ds_name, res in all_results.items():
            compact[ds_name] = {
                "dataset": ds_name,
                "num_questions": res["num_questions"],
                "length_stats": res["length_stats"],
                "per_granularity": {
                    gname: {
                        "num_rows": g["num_rows"],
                        "architectures": {
                            a: {
                                "summary": g["architectures"][a]["summary"],
                                "category_breakdown": g["architectures"][a][
                                    "category_breakdown"
                                ],
                            }
                            for a in g["architectures"]
                        },
                    }
                    for gname, g in res["per_granularity"].items()
                },
                "head_to_head_vs_turn": res["head_to_head_vs_turn"],
            }
        json.dump(compact, f, indent=2, default=str)
    print(f"Saved aggregate: {agg_path}")

    print_final_table(all_results)


if __name__ == "__main__":
    main()
