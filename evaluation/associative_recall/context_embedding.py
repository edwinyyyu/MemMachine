"""Universal context-enriched embeddings with stacked merge.

Motivation
----------
At ingest, embed each turn paired with its neighboring turns (prev/next
context) to capture anaphoric/implicit references (antecedent baked into
the turn representation). At query-time, retrieve from this context-index
and use it as a **stacked-merge supplement** to v2f (fills remaining
top-K slots after v2f claims its top-K in its natural stacked order).

Prior work
----------
- Regex / LLM alt-keys with MAX-score merge displaced v2f's clean
  retrievals (lost).
- Stacked merge (validated by critical-info-store: +3pp synth at 0 harm
  on LoCoMo) integrates supplements WITHOUT displacing v2f's primary
  picks. This test: universal context enrichment (no LLM gating) via
  stacked merge.

Variants
--------
- ``window_1``   : ``{prev} [SEP] {curr} [SEP] {next}`` (1 on each side)
- ``window_2``   : ``{prev2 prev1} [SEP] {curr} [SEP] {next1 next2}``
- ``prev_only``  : ``{prev} [SEP] {curr}``

For each variant, build a separate context-index and a bonus-variant
with +0.05 score (to test whether context hits need a nudge to enter
v2f's top-K).

Pipeline
--------
Ingest (once per ``SegmentStore``):
  For each turn ``t`` in the conversation, build the context-enriched
  text using the configured window. Embed. Map enriched-key -> parent
  turn index. Store in a per-variant ``ContextIndex``.

Query (per question):
  1. Run v2f on the original query (same as ``meta_v2f``). Keep its
     retrieved segment list in natural stacked order.
  2. Cosine-search the context-index with the raw query embedding.
     Get top-M hits (M=10).
  3. Dedupe hits to parent turn_ids (keep max score per parent).
  4. Stacked merge: start with v2f's list; append context-hit turns
     (in score order) that are NOT already present. Stop when output
     reaches the budget cap (handled by the fair-backfill framework at
     eval time; the retrieve() output is simply v2f + appended
     context-hits).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from best_shot import (
    MODEL,
    BestshotBase,
    BestshotResult,
    V2F_PROMPT,
    _format_segments,
    _parse_cues,
)


SEP = " [SEP] "

# ---------------------------------------------------------------------------
# Dedicated caches (do not pollute other agents' caches)
# ---------------------------------------------------------------------------
_CONTEXTEMB_EMB_FILE = CACHE_DIR / "contextemb_embedding_cache.json"
_CONTEXTEMB_LLM_FILE = CACHE_DIR / "contextemb_llm_cache.json"
_CONTEXT_INDEX_FILE = (
    Path(__file__).resolve().parent / "results" / "context_embedding_index.json"
)


# Read shared caches for warm-start; write only to dedicated files.
# Mirror antipara's read order to keep embedding precedence consistent.
_SHARED_EMB_READ = (
    "contextemb_embedding_cache.json",
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "agent_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "synth_test_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "fewshot_embedding_cache.json",
    "antipara_embedding_cache.json",
)
# Read order mirrors antipara_cue_gen._SHARED_LLM_READ so ctxemb's v2f run
# produces identical cached cues to MetaV2fDedicated. gpt-5-mini is
# non-deterministic -- if a key is in multiple shared caches with different
# values, whichever file is read LAST wins. Having the same read order
# guarantees the same cached cue → same top-K v2f retrieval → any recall
# delta is attributable ONLY to the context-index stacked append.
_SHARED_LLM_READ = (
    "contextemb_llm_cache.json",
    "llm_cache.json",
    "arch_llm_cache.json",
    "agent_llm_cache.json",
    "tree_llm_cache.json",
    "frontier_llm_cache.json",
    "meta_llm_cache.json",
    "optim_llm_cache.json",
    "synth_test_llm_cache.json",
    "bestshot_llm_cache.json",
    "fewshot_llm_cache.json",
    "antipara_llm_cache.json",
)


class ContextEmbEmbeddingCache(EmbeddingCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in _SHARED_EMB_READ:
            p = self.cache_dir / name
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        self.cache_file = _CONTEXTEMB_EMB_FILE
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
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


class ContextEmbLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in _SHARED_LLM_READ:
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
        self.cache_file = _CONTEXTEMB_LLM_FILE
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
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Enriched-text builders
# ---------------------------------------------------------------------------
def _format_turn(seg: Segment, max_chars: int = 800) -> str:
    text = seg.text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return f"{seg.role}: {text}"


def _enrich_turn(
    segs_by_turn: list[Segment | None],
    turn_id: int,
    variant: str,
    max_chars_per_turn: int = 800,
) -> str:
    """Build the enriched text for turn at position ``turn_id`` within
    the conversation-scoped list ``segs_by_turn`` (index = turn_id).

    ``variant`` one of: ``window_1``, ``window_2``, ``prev_only``.
    Missing neighbors (start/end of conversation) are silently dropped.
    """
    def fmt(i: int) -> str | None:
        if 0 <= i < len(segs_by_turn) and segs_by_turn[i] is not None:
            return _format_turn(segs_by_turn[i], max_chars_per_turn)
        return None

    curr = fmt(turn_id)
    if curr is None:
        return ""

    if variant == "window_1":
        prev = fmt(turn_id - 1)
        nxt = fmt(turn_id + 1)
        parts = [p for p in (prev, curr, nxt) if p is not None]
        return SEP.join(parts)

    if variant == "window_2":
        prev2 = fmt(turn_id - 2)
        prev1 = fmt(turn_id - 1)
        nxt1 = fmt(turn_id + 1)
        nxt2 = fmt(turn_id + 2)
        left = [p for p in (prev2, prev1) if p is not None]
        right = [p for p in (nxt1, nxt2) if p is not None]
        left_str = " ".join(left) if left else None
        right_str = " ".join(right) if right else None
        parts = [p for p in (left_str, curr, right_str) if p is not None]
        return SEP.join(parts)

    if variant == "prev_only":
        prev = fmt(turn_id - 1)
        parts = [p for p in (prev, curr) if p is not None]
        return SEP.join(parts)

    raise ValueError(f"unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Context-enriched index
# ---------------------------------------------------------------------------
class ContextIndex:
    """Per-variant enriched-embedding index.

    Each entry:
      - enriched_text (embedded)
      - parent_turn_index (Segment.index in base store)
      - conversation_id (filtering)
    """

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.enriched_texts: list[str] = []
        self.parent_indices: np.ndarray = np.zeros(0, dtype=np.int64)
        self.conversation_ids: np.ndarray = np.zeros(0, dtype=object)
        self.normalized_embeddings: np.ndarray = np.zeros(
            (0, 1536), dtype=np.float32
        )

    @property
    def n(self) -> int:
        return len(self.enriched_texts)

    def build(
        self,
        entries: list[tuple[int, str, str]],
        embeddings: np.ndarray,
    ) -> None:
        """``entries``: list of (parent_index, conversation_id, enriched_text)."""
        self.enriched_texts = [t[2] for t in entries]
        self.parent_indices = np.array([t[0] for t in entries], dtype=np.int64)
        self.conversation_ids = np.array(
            [t[1] for t in entries], dtype=object
        )
        if len(entries) == 0:
            self.normalized_embeddings = np.zeros((0, 1536), dtype=np.float32)
            return
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.normalized_embeddings = (embeddings / norms).astype(np.float32)

    def search_top_m(
        self,
        query_embedding: np.ndarray,
        conversation_id: str,
        top_m: int,
    ) -> list[tuple[int, int, float]]:
        """Return up to ``top_m`` hits as (entry_idx, parent_index, score).
        Scoped to ``conversation_id``."""
        if self.n == 0:
            return []
        q = query_embedding.astype(np.float32)
        qn = max(float(np.linalg.norm(q)), 1e-10)
        q = q / qn
        sims = self.normalized_embeddings @ q
        mask = self.conversation_ids == conversation_id
        sims = np.where(mask, sims, -1.0)
        order = np.argsort(sims)[::-1][: max(top_m, 1)]
        out: list[tuple[int, int, float]] = []
        for i in order:
            if sims[i] <= -0.5:
                break
            out.append((int(i), int(self.parent_indices[i]), float(sims[i])))
        return out


# ---------------------------------------------------------------------------
# Base class — variant is a class attribute
# ---------------------------------------------------------------------------
class _ContextEmbBase(BestshotBase):
    """Run v2f on the original query, retrieve from the context-enriched
    index, stacked-merge.
    """

    arch_name: str = "contextemb"
    variant: str = "window_1"   # override in subclasses
    top_m: int = 10
    max_appended: int = 40
    # Bonus added to context-hit scores when sorting (0.0 = pure stacked).
    # A positive bonus still runs AFTER v2f's top-K but re-sorts context
    # hits by (score + bonus) when deciding which to append first.
    score_bonus: float = 0.0

    # Cache the built index per (store, variant) so both K-variants share.
    _index_cache: dict[tuple[int, str], tuple[ContextIndex, dict]] = {}

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        if client is None:
            client = OpenAI(timeout=60.0, max_retries=3)
        super().__init__(store, client)
        self.embedding_cache = ContextEmbEmbeddingCache()
        self.llm_cache = ContextEmbLLMCache()

        key = (id(store), self.variant)
        cached = self._index_cache.get(key)
        if cached is None:
            idx, stats = self._build_index(store, self.variant)
            self._index_cache[key] = (idx, stats)
        self.ctx_index, self.index_stats = self._index_cache[key]

    def _build_index(
        self, store: SegmentStore, variant: str
    ) -> tuple[ContextIndex, dict]:
        # Group segments by conversation, index by turn_id.
        by_conv: dict[str, dict[int, Segment]] = {}
        for seg in store.segments:
            by_conv.setdefault(seg.conversation_id, {})[seg.turn_id] = seg

        entries: list[tuple[int, str, str]] = []
        per_conv_counts: dict[str, int] = {}

        for cid, turn_map in by_conv.items():
            if not turn_map:
                continue
            max_tid = max(turn_map.keys())
            segs_by_turn: list[Segment | None] = [
                turn_map.get(t) for t in range(max_tid + 1)
            ]
            conv_entries = 0
            for seg in turn_map.values():
                enriched = _enrich_turn(segs_by_turn, seg.turn_id, variant)
                if not enriched:
                    continue
                entries.append((seg.index, cid, enriched))
                conv_entries += 1
            per_conv_counts[cid] = conv_entries

        # Dedupe identical enriched texts pointing to same parent.
        seen: set[tuple[int, str]] = set()
        unique: list[tuple[int, str, str]] = []
        for e in entries:
            key = (e[0], e[2])
            if key in seen:
                continue
            seen.add(key)
            unique.append(e)

        print(
            f"  [contextemb/{variant}] built enriched corpus: "
            f"{len(entries)} raw, {len(unique)} deduped across "
            f"{len(per_conv_counts)} convs",
            flush=True,
        )

        # Embed enriched texts (uses batching + cache).
        texts = [t[2] for t in unique]
        embeddings = self._embed_batch(texts)

        idx = ContextIndex(variant)
        idx.build(unique, embeddings)

        stats = {
            "variant": variant,
            "n_entries_raw": len(entries),
            "n_entries_unique": len(unique),
            "n_convs": len(per_conv_counts),
            "per_conv_entries": per_conv_counts,
        }

        # Persist stats + samples per variant (one file per build; later
        # builds append a key).
        try:
            _CONTEXT_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
            existing: dict = {}
            if _CONTEXT_INDEX_FILE.exists():
                try:
                    with open(_CONTEXT_INDEX_FILE) as f:
                        existing = json.load(f)
                except (json.JSONDecodeError, OSError):
                    existing = {}
            existing[variant] = {
                "stats": stats,
                "samples": [
                    {
                        "parent_index": int(t[0]),
                        "conversation_id": t[1],
                        "enriched_text": t[2][:400],
                    }
                    for t in unique[:20]
                ],
            }
            with open(_CONTEXT_INDEX_FILE, "w") as f:
                json.dump(existing, f, indent=2, default=str)
        except OSError:
            pass

        return idx, stats

    def _embed_batch(self, texts: list[str], batch_size: int = 96) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)
        out: list[np.ndarray | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            tt = t.strip()
            if not tt:
                out[i] = np.zeros(1536, dtype=np.float32)
                continue
            cached = self.embedding_cache.get(tt)
            if cached is not None:
                out[i] = cached.astype(np.float32)
            else:
                pending.append((i, tt))

        if pending:
            print(
                f"  [contextemb] embedding {len(pending)} new enriched texts...",
                flush=True,
            )
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            batch_texts = [bt for _, bt in batch]
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    resp = self.client.embeddings.create(
                        model=EMBED_MODEL, input=batch_texts
                    )
                    break
                except Exception as e:
                    last_exc = e
                    time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"embed failed: {last_exc}")
            for (i, t), ed in zip(batch, resp.data):
                emb = np.array(ed.embedding, dtype=np.float32)
                self.embedding_cache.put(t, emb)
                out[i] = emb

        self.embedding_cache.save()
        return np.stack(out, axis=0)

    # ----- v2f run -----------------------------------------------------
    def _run_v2f(
        self, question: str, conversation_id: str
    ) -> tuple[list[Segment], dict]:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(hop0.segments)
        exclude: set[int] = {s.index for s in all_segments}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(all_segments)
        )
        prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]

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
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return all_segments, {"output": output, "cues": cues}

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        v2f_segments, v2f_meta = self._run_v2f(question, conversation_id)

        query_emb = self.embed_text(question)
        ctx_hits = self.ctx_index.search_top_m(
            query_emb, conversation_id, top_m=self.top_m
        )

        # Dedupe per parent_index, keeping max score.
        per_parent: dict[int, float] = {}
        for _eid, pidx, score in ctx_hits:
            cur = per_parent.get(pidx)
            if cur is None or score > cur:
                per_parent[pidx] = score

        v2f_indices = {s.index for s in v2f_segments}

        # Order context hits by (score + bonus), excluding those already in
        # v2f's top-K.
        ordered: list[tuple[int, float]] = sorted(
            (
                (pidx, score + self.score_bonus)
                for pidx, score in per_parent.items()
                if pidx not in v2f_indices
            ),
            key=lambda t: t[1],
            reverse=True,
        )[: self.max_appended]

        ctx_segments: list[Segment] = []
        ctx_records: list[dict] = []
        for pidx, adj_score in ordered:
            if 0 <= pidx < len(self.store.segments):
                seg = self.store.segments[pidx]
                ctx_segments.append(seg)
                ctx_records.append(
                    {
                        "parent_index": pidx,
                        "turn_id": seg.turn_id,
                        "raw_score": round(per_parent[pidx], 4),
                        "adj_score": round(adj_score, 4),
                    }
                )

        merged: list[Segment] = list(v2f_segments)
        for seg in ctx_segments:
            if seg.index not in v2f_indices:
                merged.append(seg)
                v2f_indices.add(seg.index)

        metadata = {
            "name": self.arch_name,
            "variant": self.variant,
            "score_bonus": self.score_bonus,
            "v2f_cues": v2f_meta.get("cues", []),
            "v2f_turn_ids": [s.turn_id for s in v2f_segments],
            "n_v2f_segments": len(v2f_segments),
            "n_ctx_hits_raw": len(ctx_hits),
            "n_ctx_turn_hits": len(per_parent),
            "n_ctx_turn_hits_novel": len(ordered),
            "ctx_records": ctx_records,
            "ctx_appended_turn_ids": [s.turn_id for s in ctx_segments],
        }

        return BestshotResult(segments=merged, metadata=metadata)


# ---------------------------------------------------------------------------
# Variant classes
# ---------------------------------------------------------------------------
class ContextEmbW1Stacked(_ContextEmbBase):
    """window_1 (prev,curr,next) with pure stacked merge."""

    arch_name = "contextemb_w1_stacked"
    variant = "window_1"
    score_bonus = 0.0


class ContextEmbW2Stacked(_ContextEmbBase):
    """window_2 (prev2..next2) with pure stacked merge."""

    arch_name = "contextemb_w2_stacked"
    variant = "window_2"
    score_bonus = 0.0


class ContextEmbPrevStacked(_ContextEmbBase):
    """prev_only (prev,curr) asymmetric with pure stacked merge."""

    arch_name = "contextemb_prev_stacked"
    variant = "prev_only"
    score_bonus = 0.0


class ContextEmbW1Bonus(_ContextEmbBase):
    """window_1 with +0.05 score bonus applied to context hits."""

    arch_name = "contextemb_w1_bonus"
    variant = "window_1"
    score_bonus = 0.05


ARCH_CLASSES: dict[str, type] = {
    "contextemb_w1_stacked": ContextEmbW1Stacked,
    "contextemb_w2_stacked": ContextEmbW2Stacked,
    "contextemb_prev_stacked": ContextEmbPrevStacked,
    "contextemb_w1_bonus": ContextEmbW1Bonus,
}
