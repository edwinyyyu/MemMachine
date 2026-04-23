"""Query-time alias expansion via dynamic alias tracker.

Motivation
----------
Anaphora-resolution at INGEST fails because the first alias introduction is
essentially a cue itself (and costly to re-embed each variant). Instead,
dynamic alias tracker is used at QUERY time: we inject alias-sibling context
into ONE v2f call, so the LLM picks whichever register fits its imagined
conversation turn. Cheap (~1x LLM cost vs 3x of alias_expand_v2f_full).

Plus: tracker records first_seen_turn / last_seen_turn so drift-aware variants
can restrict attention to recent or early aliases.

Variants
--------
  alias_trk_context — single v2f call with alias-context injected into prompt.
  alias_trk_drift   — adds recency-weighted alias filtering:
                       * default queries use aliases in top-60% of turn range
                       * queries with temporal markers ("originally",
                         "when X was new", "first", "initially") use early
                         aliases only (bottom-40% of turn range).

Reuses existing `conversation_alias_groups.json` (extracted by
alias_expansion.py) and supplements with first/last_seen_turn indices via a
lightweight whole-word grep — no extra LLM calls.

Caches
------
Dedicated aliastrk_*_cache.json. Reads shared caches for hits but writes only
to the dedicated files to avoid corrupting other agents' caches.
"""

from __future__ import annotations

import json
import re
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


# ---------------------------------------------------------------------------
# Dedicated caches
# ---------------------------------------------------------------------------

_TRK_EMB_FILE = CACHE_DIR / "aliastrk_embedding_cache.json"
_TRK_LLM_FILE = CACHE_DIR / "aliastrk_llm_cache.json"
_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_ALIAS_GROUPS_FILE = _RESULTS_DIR / "conversation_alias_groups.json"
_ENRICHED_FILE = _RESULTS_DIR / "alias_tracker_enriched.json"

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
    "inv_query_embedding_cache.json",
    "anchor_embedding_cache.json",
    "alias_embedding_cache.json",
    "aliastrk_embedding_cache.json",
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
    "antipara_llm_cache.json",
    "inv_query_llm_cache.json",
    "anchor_llm_cache.json",
    "alias_llm_cache.json",
    "aliastrk_llm_cache.json",
)


class TrkEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _TRK_EMB_FILE
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


class TrkLLMCache(LLMCache):
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
        self.cache_file = _TRK_LLM_FILE
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
# Tracker: enrich alias groups with first/last_seen turn indices
# ---------------------------------------------------------------------------


def _whole_word_positions(needle: str, haystack: str) -> bool:
    """Return True if needle occurs as a whole-word substring (non-alnum
    boundaries on both sides) in haystack. Case-insensitive."""
    hl = haystack.lower()
    nl = needle.lower()
    if not nl:
        return False
    start = 0
    while True:
        idx = hl.find(nl, start)
        if idx < 0:
            return False
        end = idx + len(nl)
        left_ok = idx == 0 or not hl[idx - 1].isalnum()
        right_ok = end == len(hl) or not hl[end].isalnum()
        if left_ok and right_ok:
            return True
        start = idx + 1


class AliasTracker:
    """Enriched alias tracker.

    For each conversation:
      enriched[cid] = [
        {
          "canonical": <most-used form>,
          "aliases": [form1, form2, ...],  # all forms (including canonical)
          "per_alias": {alias: {"count": n, "turns": [t1, t2, ...]}},
          "first_seen_turn": int,
          "last_seen_turn": int,
          "min_turn": int,
          "max_turn": int,  # conversation range
        },
        ...
      ]

    The per-conversation min/max turn (range) is stored alongside for drift
    computations.
    """

    def __init__(self) -> None:
        self._raw_groups: dict[str, list[list[str]]] = {}
        if _ALIAS_GROUPS_FILE.exists():
            try:
                with open(_ALIAS_GROUPS_FILE) as f:
                    data = json.load(f)
                self._raw_groups = data.get("groups", {}) or {}
            except (json.JSONDecodeError, OSError):
                self._raw_groups = {}

        self._enriched: dict[str, list[dict]] = {}
        self._conv_turn_range: dict[str, tuple[int, int]] = {}
        if _ENRICHED_FILE.exists():
            try:
                with open(_ENRICHED_FILE) as f:
                    data = json.load(f)
                self._enriched = data.get("enriched", {}) or {}
                self._conv_turn_range = {
                    k: tuple(v)
                    for k, v in (data.get("conv_turn_range", {}) or {}).items()
                }
            except (json.JSONDecodeError, OSError):
                self._enriched = {}
                self._conv_turn_range = {}

    def groups(self, conversation_id: str) -> list[dict]:
        return list(self._enriched.get(conversation_id, []))

    def conv_turn_range(self, conversation_id: str) -> tuple[int, int]:
        return self._conv_turn_range.get(conversation_id, (0, 0))

    def build_for_store(
        self, store: SegmentStore, conversation_ids: list[str] | None = None
    ) -> None:
        all_cids = sorted({s.conversation_id for s in store.segments})
        if conversation_ids is None:
            conversation_ids = all_cids

        pending = [cid for cid in conversation_ids if cid not in self._enriched]
        if not pending:
            return
        print(
            f"  [aliastrk] Enriching alias groups for {len(pending)} conv(s): "
            f"{pending}",
            flush=True,
        )
        for cid in pending:
            groups = self._raw_groups.get(cid, [])
            segs = sorted(
                [s for s in store.segments if s.conversation_id == cid],
                key=lambda s: s.turn_id,
            )
            if segs:
                min_t = segs[0].turn_id
                max_t = segs[-1].turn_id
            else:
                min_t = 0
                max_t = 0
            self._conv_turn_range[cid] = (min_t, max_t)

            enriched_list: list[dict] = []
            for group in groups:
                per_alias: dict[str, dict] = {}
                for alias in group:
                    turns: list[int] = []
                    for seg in segs:
                        if _whole_word_positions(alias, seg.text):
                            turns.append(seg.turn_id)
                    per_alias[alias] = {"count": len(turns), "turns": turns}

                # Canonical = highest count, tie-break by total string length
                # (longer form usually more canonical).
                canonical = max(
                    group,
                    key=lambda a: (per_alias[a]["count"], len(a)),
                )

                all_turns = [
                    t for info in per_alias.values() for t in info["turns"]
                ]
                if all_turns:
                    first_seen = min(all_turns)
                    last_seen = max(all_turns)
                else:
                    first_seen = -1
                    last_seen = -1

                enriched_list.append(
                    {
                        "canonical": canonical,
                        "aliases": list(group),
                        "per_alias": per_alias,
                        "first_seen_turn": first_seen,
                        "last_seen_turn": last_seen,
                        "min_turn": min_t,
                        "max_turn": max_t,
                    }
                )
            self._enriched[cid] = enriched_list

        # Persist
        _ENRICHED_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_ENRICHED_FILE, "w") as f:
            json.dump(
                {
                    "enriched": self._enriched,
                    "conv_turn_range": {
                        k: list(v) for k, v in self._conv_turn_range.items()
                    },
                },
                f,
                indent=2,
                default=str,
            )


# ---------------------------------------------------------------------------
# Query-time alias matching
# ---------------------------------------------------------------------------


def _whole_word_find(needle: str, haystack: str) -> tuple[int, int] | None:
    """Find first whole-word occurrence; return (start, end) indices in
    haystack (lowercased matching but indices index into haystack).
    """
    hl = haystack.lower()
    nl = needle.lower()
    if not nl:
        return None
    start = 0
    while True:
        idx = hl.find(nl, start)
        if idx < 0:
            return None
        end = idx + len(nl)
        left_ok = idx == 0 or not hl[idx - 1].isalnum()
        right_ok = end == len(hl) or not hl[end].isalnum()
        if left_ok and right_ok:
            return (idx, end)
        start = idx + 1


def find_alias_matches(
    query: str, groups: list[dict]
) -> list[dict]:
    """Find occurrences of any alias-group member in the query. One match per
    group (longest-first preference). Returns list of dicts:
      {
        "group_index": int,
        "matched_form": str,  # exact query substring
        "canonical": str,
        "siblings": [all OTHER forms in the group],
        "first_seen_turn": int,
        "last_seen_turn": int,
      }
    """
    # Flatten (alias, group_idx) pairs sorted by alias length desc.
    flat: list[tuple[str, int]] = []
    for gi, g in enumerate(groups):
        for alias in g["aliases"]:
            flat.append((alias, gi))
    flat.sort(key=lambda t: len(t[0]), reverse=True)

    matched_groups: set[int] = set()
    consumed_spans: list[tuple[int, int]] = []
    matches: list[dict] = []

    def overlaps(span: tuple[int, int]) -> bool:
        for s, e in consumed_spans:
            if not (span[1] <= s or span[0] >= e):
                return True
        return False

    for alias, gi in flat:
        if gi in matched_groups:
            continue
        pos = _whole_word_find(alias, query)
        if pos is None:
            continue
        if overlaps(pos):
            continue
        consumed_spans.append(pos)
        matched_groups.add(gi)
        group = groups[gi]
        matched_form = query[pos[0]: pos[1]]
        siblings = [a for a in group["aliases"] if a.lower() != alias.lower()]
        matches.append(
            {
                "group_index": gi,
                "matched_form": matched_form,
                "canonical": group["canonical"],
                "siblings": siblings,
                "first_seen_turn": group["first_seen_turn"],
                "last_seen_turn": group["last_seen_turn"],
                "min_turn": group["min_turn"],
                "max_turn": group["max_turn"],
            }
        )
    return matches


# ---------------------------------------------------------------------------
# Drift-aware sibling filtering
# ---------------------------------------------------------------------------

_EARLY_TEMPORAL_MARKERS = (
    "originally",
    "original",
    "initially",
    "initial",
    "first",
    "at first",
    "when .* was new",
    "when .* started",
    "when .* began",
    "back when",
    "beginning",
    "early on",
    "used to",
    "in the beginning",
    "at the start",
)


def _has_early_marker(query: str) -> bool:
    ql = query.lower()
    return any(bool(re.search(rf"\b{m}\b", ql)) for m in _EARLY_TEMPORAL_MARKERS)


def filter_siblings_by_drift(
    match: dict,
    group_aliases_per_info: dict[str, dict],
    early_bias: bool,
) -> list[str]:
    """Filter siblings by drift: for default queries keep top-60% recency; for
    early-biased queries keep bottom-40% (earliest-appearing forms).

    group_aliases_per_info: per_alias dict from the tracker (contains 'turns').
    """
    siblings = match["siblings"]
    min_t = match["min_turn"]
    max_t = match["max_turn"]
    if max_t <= min_t or len(siblings) <= 1:
        return siblings

    # Compute per-sibling representative turn (mean of appearances).
    sib_turns: list[tuple[str, float]] = []
    for sib in siblings:
        info = group_aliases_per_info.get(sib, {})
        turns = info.get("turns", [])
        if not turns:
            # Fall back to a neutral mid-point.
            sib_turns.append((sib, (min_t + max_t) / 2.0))
        else:
            sib_turns.append((sib, sum(turns) / len(turns)))

    span = max_t - min_t
    if early_bias:
        # Keep those whose rep turn is in the bottom 40% of range.
        cutoff = min_t + 0.4 * span
        kept = [s for s, t in sib_turns if t <= cutoff]
    else:
        # Keep those whose rep turn is in the top 60% of range (i.e. >= bottom
        # 40%).
        cutoff = min_t + 0.4 * span
        kept = [s for s, t in sib_turns if t >= cutoff]

    # If filter eliminates everyone, fall back to all siblings.
    if not kept:
        return siblings
    return kept


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_alias_context_note(matches: list[dict], max_siblings: int = 4) -> str:
    """Build the alias-context note block to prepend before v2f prompt.

    Format:
      Note: in the conversation, 'X' is also known as: 'A', 'B', 'C'. 'Y' is
      also known as: 'D', 'E'. Your cues may use any of these forms, whichever
      fits the imagined conversation register.
    """
    if not matches:
        return ""
    parts: list[str] = []
    for m in matches:
        sibs = m.get("_filtered_siblings", m["siblings"])[:max_siblings]
        if not sibs:
            continue
        sib_text = ", ".join(f"'{s}'" for s in sibs)
        parts.append(f"'{m['matched_form']}' is also known as: {sib_text}")
    if not parts:
        return ""
    body = ". ".join(parts)
    return (
        f"Note: in the conversation, {body}. Your cues may use any of these "
        "forms, whichever fits the imagined conversation register."
    )


# ---------------------------------------------------------------------------
# Base arch
# ---------------------------------------------------------------------------


class _AliasTrackerV2fBase(BestshotBase):
    """Single v2f call with alias-context injection. Cheaper than
    alias_expand_v2f (1x LLM vs ~3x)."""

    arch_name: str = "alias_trk_context"
    drift_aware: bool = False  # subclass flag

    _tracker_cache: dict[int, AliasTracker] = {}

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        if client is None:
            client = OpenAI(timeout=60.0, max_retries=3)
        super().__init__(store, client)
        self.embedding_cache = TrkEmbeddingCache()
        self.llm_cache = TrkLLMCache()

        key = id(store)
        trk = self._tracker_cache.get(key)
        if trk is None:
            trk = AliasTracker()
            conv_ids = sorted({s.conversation_id for s in store.segments})
            trk.build_for_store(store, conv_ids)
            self._tracker_cache[key] = trk
        self.tracker = trk

    def llm_call(self, prompt: str, model: str = MODEL) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2000,
                )
                text = response.choices[0].message.content or ""
                self.llm_cache.put(model, prompt, text)
                self.llm_calls += 1
                return text
            except Exception as e:
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
        print(f"    LLM call failed after 3 attempts: {last_exc}", flush=True)
        self.llm_cache.put(model, prompt, "")
        self.llm_calls += 1
        return ""

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.embeddings.create(
                    model=EMBED_MODEL, input=[text]
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                self.embedding_cache.put(text, embedding)
                self.embed_calls += 1
                return embedding
            except Exception as e:
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
        print(f"    Embed failed after 3 attempts: {last_exc}", flush=True)
        self.embed_calls += 1
        return np.zeros(1536, dtype=np.float32)

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        groups = self.tracker.groups(conversation_id)
        matches = find_alias_matches(question, groups)

        # Drift filtering (in place: set _filtered_siblings on each match).
        early = _has_early_marker(question) if self.drift_aware else False
        for m in matches:
            if self.drift_aware:
                gi = m["group_index"]
                per_alias = groups[gi]["per_alias"]
                filtered = filter_siblings_by_drift(m, per_alias, early)
                m["_filtered_siblings"] = filtered
            else:
                m["_filtered_siblings"] = m["siblings"]

        alias_note = build_alias_context_note(matches)

        # Hop 0: cosine search with original query embedding.
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(all_segments)
        )

        # Build prompt with alias-context note inserted at the top (between the
        # lead-in and the {question}).
        if alias_note:
            question_with_note = f"{question}\n\n{alias_note}"
        else:
            question_with_note = question
        prompt = V2F_PROMPT.format(
            question=question_with_note, context_section=context_section
        )

        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]

        for cue in cues:
            if not cue.strip():
                continue
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

        metadata = {
            "name": self.arch_name,
            "alias_note": alias_note,
            "matches": [
                {
                    "matched_form": m["matched_form"],
                    "canonical": m["canonical"],
                    "siblings": m["siblings"],
                    "filtered_siblings": m["_filtered_siblings"],
                    "first_seen_turn": m["first_seen_turn"],
                    "last_seen_turn": m["last_seen_turn"],
                }
                for m in matches
            ],
            "early_bias": early,
            "num_matches": len(matches),
            "cues": cues,
            "output": output,
        }

        return BestshotResult(segments=all_segments, metadata=metadata)


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class AliasTrkContext(_AliasTrackerV2fBase):
    """Single v2f call with alias-context injection (primary variant)."""

    arch_name = "alias_trk_context"
    drift_aware = False


class AliasTrkDrift(_AliasTrackerV2fBase):
    """Alias-context injection + recency-weighted sibling filtering."""

    arch_name = "alias_trk_drift"
    drift_aware = True


ARCH_CLASSES: dict[str, type] = {
    "alias_trk_context": AliasTrkContext,
    "alias_trk_drift": AliasTrkDrift,
}
