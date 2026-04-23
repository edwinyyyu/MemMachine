"""Stacked-merge alias alt-keys: ingest-time alias substitution + stacked retrieval.

Motivation
----------
Prior result: `alias_expand_v2f` beats v2f by +2.3pp on LoCoMo K=50, but costs
~3x LLM calls per query (runs v2f on each alias variant). Alias groups are
already extracted at ingest (cached in `results/conversation_alias_groups.json`).
This test: build an alt-key index at ingest using alias substitutions, query
with stacked merge (v2f fills top-K first; alias alt-keys fill remaining slots
only). Zero per-query LLM overhead -- only ingest cost.

Pipeline
--------
Ingest-time (once per conversation):
  For each turn t in the conversation and each alias occurrence e in t's text
  (from this conversation's alias groups), generate an alt-key per sibling a
  in G \\ {e} by replacing e -> a in t's text. Embed each alt-key; map
  alt-key -> parent_turn_index. Store in a separate alias alt-key index.

Query-time (per question):
  1. Run v2f (MetaV2f). Get its retrieved segment list in its natural
     stacked order (hop0 top-10 first, then cue1 hits, then cue2 hits).
  2. Retrieve top-M from alias alt-key index using the raw query. Map hits
     back to parent turn indices, keeping max score per turn.
  3. Stacked merge: start with v2f's list, then append alias-hit turns (in
     score order) that are not already present, then cosine backfill for any
     remaining budget slots at eval time (handled by fair-backfill framework).
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
    RetrievalResult,
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
from alias_expansion import find_alias_matches, _replace_first_occurrence


# ---------------------------------------------------------------------------
# Dedicated caches (do not pollute other agents' caches)
# ---------------------------------------------------------------------------
_STACKED_ALIAS_EMB_FILE = CACHE_DIR / "stacked_alias_embedding_cache.json"
_STACKED_ALIAS_LLM_FILE = CACHE_DIR / "stacked_alias_llm_cache.json"
_ALIAS_GROUPS_FILE = (
    Path(__file__).resolve().parent / "results" / "conversation_alias_groups.json"
)
_ALT_KEY_INDEX_FILE = (
    Path(__file__).resolve().parent / "results" / "stacked_alias_altkey_index.json"
)
# Dedicated alt-key embeddings cache: compact .npz (hash -> embedding).
_ALT_KEY_EMB_NPZ = CACHE_DIR / "stacked_alias_altkey_embeddings.npz"


# Read shared caches for warm-start; write only to dedicated files.
# Order matches antipara_cue_gen's so stacked_alias and MetaV2fDedicated
# resolve v2f prompts identically (last-write-wins on key collisions).
_SHARED_EMB_READ = (
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
    "alias_embedding_cache.json",
    "stacked_alias_embedding_cache.json",
)
_SHARED_LLM_READ = (
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
    "alias_llm_cache.json",
    "stacked_alias_llm_cache.json",
    # antipara LAST: its entries (from our MetaV2fDedicated baseline) win
    # on V2F_PROMPT key collisions -> identical v2f cues.
    "antipara_llm_cache.json",
)


class StackedAliasEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _STACKED_ALIAS_EMB_FILE
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
        # Use a unique tmp file to avoid racing with any parallel writer and
        # tolerate the replace target disappearing underneath us.
        import os
        tmp = self.cache_file.parent / (
            self.cache_file.name + f".tmp.{os.getpid()}"
        )
        try:
            with open(tmp, "w") as f:
                json.dump(existing, f)
            os.replace(tmp, self.cache_file)
        except FileNotFoundError:
            # tmp vanished (concurrent cleanup) — skip this save, we'll try
            # again next flush
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return
        self._new_entries = {}


class StackedAliasLLMCache(LLMCache):
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
        self.cache_file = _STACKED_ALIAS_LLM_FILE
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
        import os
        tmp = self.cache_file.parent / (
            self.cache_file.name + f".tmp.{os.getpid()}"
        )
        try:
            with open(tmp, "w") as f:
                json.dump(existing, f)
            os.replace(tmp, self.cache_file)
        except FileNotFoundError:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Alt-key generation
# ---------------------------------------------------------------------------
def load_alias_groups() -> dict[str, list[list[str]]]:
    """Load the alias groups extracted by alias_expansion.py. Read-only."""
    if not _ALIAS_GROUPS_FILE.exists():
        raise FileNotFoundError(
            f"Missing alias groups cache: {_ALIAS_GROUPS_FILE}. "
            "Run alias_expansion first."
        )
    with open(_ALIAS_GROUPS_FILE) as f:
        data = json.load(f)
    return data.get("groups", {}) or {}


def generate_alt_keys_for_conversation(
    segments: list[Segment],
    alias_groups: list[list[str]],
    max_siblings_per_match: int = 4,
    max_chars: int = 4000,
) -> list[tuple[int, str, str, str]]:
    """For each turn, find alias matches; for each sibling, produce a
    substituted alt-key text (full turn with alias -> sibling).

    Returns list of (parent_index, alt_key_text, matched_alias, sibling).

    Alt-keys longer than `max_chars` are truncated (embedding models have an
    8192-token limit; ~4000 chars is a safe bound).
    """
    alt_keys: list[tuple[int, str, str, str]] = []
    for seg in segments:
        matches = find_alias_matches(seg.text, alias_groups)
        for matched_term, siblings in matches:
            for sib in siblings[:max_siblings_per_match]:
                variant = _replace_first_occurrence(seg.text, matched_term, sib)
                if not variant or variant == seg.text:
                    continue
                if len(variant) > max_chars:
                    variant = variant[:max_chars]
                alt_keys.append((seg.index, variant, matched_term, sib))
    return alt_keys


# ---------------------------------------------------------------------------
# Alt-key index (separate from main SegmentStore).
# ---------------------------------------------------------------------------
class AliasAltKeyIndex:
    """Per-conversation alt-key index. Each alt-key has:
      - text (embedded)
      - parent_turn_index (points to a Segment in the base store)
      - conversation_id (for filtering)
    """

    def __init__(self) -> None:
        self.alt_texts: list[str] = []
        self.parent_indices: np.ndarray = np.zeros(0, dtype=np.int64)
        self.conversation_ids: np.ndarray = np.zeros(0, dtype=object)
        self.normalized_embeddings: np.ndarray = np.zeros((0, 1536), dtype=np.float32)
        self.matched_alias: list[str] = []
        self.sibling: list[str] = []

    @property
    def n(self) -> int:
        return len(self.alt_texts)

    def build(
        self,
        alt_key_tuples: list[tuple[int, str, str, str]],
        conv_id_for_parent: dict[int, str],
        embeddings: np.ndarray,
    ) -> None:
        n = len(alt_key_tuples)
        self.alt_texts = [t[1] for t in alt_key_tuples]
        self.matched_alias = [t[2] for t in alt_key_tuples]
        self.sibling = [t[3] for t in alt_key_tuples]
        self.parent_indices = np.array([t[0] for t in alt_key_tuples], dtype=np.int64)
        self.conversation_ids = np.array(
            [conv_id_for_parent[t[0]] for t in alt_key_tuples], dtype=object
        )
        if n == 0:
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
    ) -> list[tuple[int, int, float, str, str]]:
        """Return up to top_m hits as (altkey_idx, parent_index, score,
        matched_alias, sibling). Scoped to a conversation_id."""
        if self.n == 0:
            return []
        q = query_embedding.astype(np.float32)
        qn = max(float(np.linalg.norm(q)), 1e-10)
        q = q / qn
        sims = self.normalized_embeddings @ q  # (N,)
        mask = self.conversation_ids == conversation_id
        sims = np.where(mask, sims, -1.0)
        order = np.argsort(sims)[::-1][: max(top_m, 1)]
        out: list[tuple[int, int, float, str, str]] = []
        for i in order:
            if sims[i] <= -0.5:
                break
            out.append(
                (
                    int(i),
                    int(self.parent_indices[i]),
                    float(sims[i]),
                    self.matched_alias[i],
                    self.sibling[i],
                )
            )
        return out


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class _StackedAliasBase(BestshotBase):
    """Run v2f on the original query, retrieve from the alias alt-key index
    separately, and stacked-merge.
    """

    arch_name: str = "stacked_alias"
    # How many alt-key hits to consider (before dedup to parent turns). Plan
    # says M=10. We widen a bit since multiple alt-keys can collapse to the
    # same parent.
    alt_key_top_m: int = 30
    # Max distinct alias-hit turns to append per query (prevents runaway
    # stacking when many alt-keys exist for the same conversation).
    max_alias_turns_appended: int = 40
    # Score bonus applied to alias hits before re-sorting with v2f items.
    # 0.0 => basic stacked mode (append after v2f).
    alias_bonus: float = 0.0

    _index_cache: dict[int, tuple[AliasAltKeyIndex, dict[str, int]]] = {}

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        if client is None:
            client = OpenAI(timeout=60.0, max_retries=3)
        super().__init__(store, client)
        self.embedding_cache = StackedAliasEmbeddingCache()
        self.llm_cache = StackedAliasLLMCache()

        key = id(store)
        cached = self._index_cache.get(key)
        if cached is None:
            idx, stats = self._build_index(store)
            self._index_cache[key] = (idx, stats)
        self.alt_index, self.index_stats = self._index_cache[key]

    def _build_index(
        self, store: SegmentStore
    ) -> tuple[AliasAltKeyIndex, dict[str, int]]:
        alias_groups = load_alias_groups()
        conv_ids = sorted({s.conversation_id for s in store.segments})

        alt_tuples: list[tuple[int, str, str, str]] = []
        conv_id_for_parent: dict[int, str] = {}

        per_conv_alt_count: dict[str, int] = {}
        per_conv_turn_match_count: dict[str, int] = {}
        turns_with_match = 0

        for cid in conv_ids:
            segs = [s for s in store.segments if s.conversation_id == cid]
            if not segs:
                continue
            groups = alias_groups.get(cid, [])
            if not groups:
                continue
            keys = generate_alt_keys_for_conversation(segs, groups)
            per_conv_alt_count[cid] = len(keys)
            matched_parents = {k[0] for k in keys}
            per_conv_turn_match_count[cid] = len(matched_parents)
            turns_with_match += len(matched_parents)
            for t in keys:
                alt_tuples.append(t)
            for s in segs:
                conv_id_for_parent[s.index] = s.conversation_id

        # Dedupe by (parent_index, alt_text) to avoid redundant embed calls
        seen: set[tuple[int, str]] = set()
        unique: list[tuple[int, str, str, str]] = []
        for t in alt_tuples:
            key = (t[0], t[1])
            if key in seen:
                continue
            seen.add(key)
            unique.append(t)

        print(
            f"  [stacked_alias] built alt-key corpus: "
            f"{len(alt_tuples)} raw, {len(unique)} deduped, "
            f"across {len([c for c in per_conv_alt_count if per_conv_alt_count[c] > 0])} convs",
            flush=True,
        )

        # Embed unique alt-keys
        texts = [t[1] for t in unique]
        embeddings = self._embed_batch(texts)

        idx = AliasAltKeyIndex()
        idx.build(unique, conv_id_for_parent, embeddings)

        stats = {
            "n_alt_keys_raw": len(alt_tuples),
            "n_alt_keys_unique": len(unique),
            "n_convs_with_altkeys": sum(
                1 for v in per_conv_alt_count.values() if v > 0
            ),
            "n_convs_total": len(conv_ids),
            "n_turns_with_match": turns_with_match,
            "per_conv_alt_count": per_conv_alt_count,
        }

        # Persist index stats
        try:
            _ALT_KEY_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_ALT_KEY_INDEX_FILE, "w") as f:
                json.dump(
                    {
                        "store_npz_hint": str(store.__class__.__name__),
                        "stats": stats,
                        # Small sample for inspection
                        "samples": [
                            {
                                "parent_index": int(t[0]),
                                "alt_text": t[1][:200],
                                "matched_alias": t[2],
                                "sibling": t[3],
                            }
                            for t in unique[:30]
                        ],
                    },
                    f,
                    indent=2,
                    default=str,
                )
        except OSError:
            pass

        return idx, stats

    def _embed_batch(self, texts: list[str], batch_size: int = 96) -> np.ndarray:
        """Embed alt-key texts using a compact NPZ-backed cache to avoid
        bloating the JSON embedding cache (6k+ vectors = 200MB json)."""
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)

        # Load NPZ cache if present
        npz_cache: dict[str, np.ndarray] = {}
        if _ALT_KEY_EMB_NPZ.exists():
            try:
                data = np.load(_ALT_KEY_EMB_NPZ, allow_pickle=True)
                keys = data["keys"]
                vecs = data["vecs"]
                for i in range(len(keys)):
                    npz_cache[str(keys[i])] = vecs[i].astype(np.float32)
            except (OSError, KeyError, ValueError):
                npz_cache = {}

        import hashlib

        def _key(text: str) -> str:
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        out: list[np.ndarray | None] = [None] * len(texts)
        pending: list[tuple[int, str, str]] = []
        for i, t in enumerate(texts):
            tt = t.strip()
            if not tt:
                out[i] = np.zeros(1536, dtype=np.float32)
                continue
            k = _key(tt)
            if k in npz_cache:
                out[i] = npz_cache[k]
            else:
                pending.append((i, tt, k))

        if pending:
            print(
                f"  [stacked_alias] embedding {len(pending)} new alt-keys"
                f" (npz-cache hits: {len(texts) - len(pending)})...",
                flush=True,
            )
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            batch_texts = [bt for _, bt, _ in batch]
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
            for (i, t, k), ed in zip(batch, resp.data):
                emb = np.array(ed.embedding, dtype=np.float32)
                out[i] = emb
                npz_cache[k] = emb

        # Save NPZ cache if we added anything
        if pending:
            try:
                import os
                keys_arr = np.array(list(npz_cache.keys()), dtype=object)
                vecs_arr = np.stack(
                    [npz_cache[k] for k in npz_cache.keys()], axis=0
                )
                # numpy.savez auto-adds '.npz' extension if absent, so write
                # to a name that already ends with .npz and rename.
                tmp_path = _ALT_KEY_EMB_NPZ.parent / (
                    _ALT_KEY_EMB_NPZ.stem + f".tmp.{os.getpid()}.npz"
                )
                np.savez_compressed(tmp_path, keys=keys_arr, vecs=vecs_arr)
                if tmp_path.exists():
                    os.replace(tmp_path, _ALT_KEY_EMB_NPZ)
            except OSError as e:
                print(f"  [stacked_alias] WARN: npz save failed: {e}", flush=True)

        return np.stack(out, axis=0)

    # --- v2f run (same code path as MetaV2f but using our caches) -----------
    def _run_v2f(
        self, question: str, conversation_id: str
    ) -> tuple[list[Segment], dict]:
        """Returns (ordered v2f segment list, metadata). Stacked: hop0
        top-10, then cue-1 extras, then cue-2 extras."""
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

        # Alias alt-key retrieval
        query_emb = self.embed_text(question)
        alt_hits = self.alt_index.search_top_m(
            query_emb, conversation_id, top_m=self.alt_key_top_m
        )

        # Dedupe alias hits by parent_index, keep max score
        per_parent: dict[int, tuple[float, str, str]] = {}
        for _alt_i, pidx, score, matched, sib in alt_hits:
            cur = per_parent.get(pidx)
            if cur is None or score > cur[0]:
                per_parent[pidx] = (score, matched, sib)

        v2f_indices = {s.index for s in v2f_segments}

        # Order alias hits by score desc, excluding those already in v2f
        ordered_alias = sorted(
            (
                (pidx, meta[0], meta[1], meta[2])
                for pidx, meta in per_parent.items()
                if pidx not in v2f_indices
            ),
            key=lambda t: t[1],
            reverse=True,
        )[: self.max_alias_turns_appended]

        # Map parent indices to segments via the store
        alias_segments: list[Segment] = []
        alias_records: list[dict] = []
        for pidx, score, matched, sib in ordered_alias:
            if 0 <= pidx < len(self.store.segments):
                seg = self.store.segments[pidx]
                alias_segments.append(seg)
                alias_records.append(
                    {
                        "parent_index": pidx,
                        "turn_id": seg.turn_id,
                        "score": round(score, 4),
                        "matched_alias": matched,
                        "sibling": sib,
                    }
                )

        # Stacked merge: v2f first, then alias hits (score order)
        merged: list[Segment] = list(v2f_segments)
        for seg in alias_segments:
            if seg.index not in v2f_indices:
                merged.append(seg)
                v2f_indices.add(seg.index)

        # Diagnostics: how many alias-hit turns actually entered the top-K?
        alias_appended_turn_ids = [s.turn_id for s in alias_segments]

        metadata = {
            "name": self.arch_name,
            "v2f_cues": v2f_meta.get("cues", []),
            "v2f_turn_ids": [s.turn_id for s in v2f_segments],
            "n_v2f_segments": len(v2f_segments),
            "n_alt_key_hits_raw": len(alt_hits),
            "n_alias_turn_hits": len(per_parent),
            "n_alias_turn_hits_novel": len(ordered_alias),
            "alias_appended_turn_ids": alias_appended_turn_ids,
            "alias_records": alias_records,
        }

        return BestshotResult(segments=merged, metadata=metadata)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
class StackedAliasK20(_StackedAliasBase):
    """Same retrieve() in both K variants; eval truncates at K. Kept as a
    distinct class so the evaluator can name them independently."""

    arch_name = "stacked_alias"


class StackedAlias(_StackedAliasBase):
    """Alias for backward-compat naming."""

    arch_name = "stacked_alias"


ARCH_CLASSES: dict[str, type] = {
    "stacked_alias": StackedAlias,
}
