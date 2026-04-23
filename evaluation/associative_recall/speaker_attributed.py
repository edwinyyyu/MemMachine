"""Speaker-attributed retrieval.

Hypothesis: entity-mention retrieval often fails because dialog entities appear
as VOCATIVES ("Hey Caroline!") rather than as subjects. For a query about a
specific person, the turns spoken BY that person are frequently the gold, yet
that person's name doesn't appear in those turns. This architecture identifies
the real name attached to the `user` role per conversation at ingest, then
at query time — if the query mentions that person — boosts or filters turns
SPOKEN by the `user` role, bypassing the vocative problem.

Pipeline
--------
Ingest (once per conversation):
  LLM reads the first ~20 turns with role labels. Outputs the first name of
  the `user` speaker, or UNKNOWN. Cached to disk per conversation.

Query-time (per question):
  1. Run v2f normally.
  2. Regex-extract capitalized first-name tokens from the query.
  3. If any token matches the conversation's user-name, the query "mentions
     the conv-user" -> apply a speaker-aware transform.
  4. Three variants:
     - speaker_boost_0.02: mild rerank (boost role=user turns in v2f top-K
       by +0.02; also append top-5 user-only cosine hits).
     - speaker_boost_0.05: stronger boost.
     - speaker_user_filter: aggressive — drop role=assistant turns from the
       returned list; append user-only cosine top-K; cosine backfill happens
       outside via fair-backfill eval.

All three variants also append an additional top-5 user-only cosine segments
via stacked merge (dedup). The "boost" variants re-sort merged segments by
(cosine_score + bonus if role=user). The "filter" variant hard-drops
role=assistant turns from the arch output.
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
# Dedicated caches (do not pollute other agents' caches)
# ---------------------------------------------------------------------------
_SPEAKER_EMB_FILE = CACHE_DIR / "speaker_embedding_cache.json"
_SPEAKER_LLM_FILE = CACHE_DIR / "speaker_llm_cache.json"
_CONV_SPEAKERS_FILE = (
    Path(__file__).resolve().parent / "results" / "conversation_speakers.json"
)

# Read shared caches for warm-start; write only to dedicated files.
_SHARED_EMB_READ = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "agent_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "synth_test_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "antipara_embedding_cache.json",
    "speaker_embedding_cache.json",
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
    "speaker_llm_cache.json",
    # antipara LAST so V2F_PROMPT cues are identical to MetaV2fDedicated.
    "antipara_llm_cache.json",
)


class SpeakerEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _SPEAKER_EMB_FILE
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


class SpeakerLLMCache(LLMCache):
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
        self.cache_file = _SPEAKER_LLM_FILE
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
# Speaker identification
# ---------------------------------------------------------------------------
SPEAKER_ID_PROMPT = """\
Read these opening conversation turns. Identify the name of the person \
speaking as the 'user' (the human, not the AI assistant). Usually \
discoverable via self-introduction or name mentions.

{first_turns}

Output only the user's first name. If unclear, output UNKNOWN.\
"""


def _format_opening_turns(
    segments: list[Segment], num_turns: int = 20, max_chars: int = 220
) -> str:
    ordered = sorted(segments, key=lambda s: s.turn_id)[:num_turns]
    lines: list[str] = []
    for seg in ordered:
        txt = seg.text.strip().replace("\n", " ")
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "..."
        lines.append(f"[{seg.role}]: {txt}")
    return "\n".join(lines)


# Regex: capitalized first-name tokens (2+ chars), not all-caps acronyms.
_RE_NAME_TOKEN = re.compile(r"\b[A-Z][a-z]{1,}\b")
# Stop words / sentence-initial/false-positive tokens to ignore.
_NAME_STOPWORDS = {
    "The", "A", "An", "What", "When", "Where", "Who", "Why", "How",
    "Did", "Does", "Do", "Is", "Are", "Was", "Were", "Has", "Have",
    "Had", "Can", "Could", "Should", "Would", "Will", "Shall", "May",
    "Might", "Must", "This", "That", "These", "Those", "Please",
    "And", "Or", "But", "If", "Then", "So", "Also", "Given", "Based",
    "List", "Draft", "Help", "Create", "Include", "Tell", "Find",
    "Explain", "Describe", "Summarize", "Make", "Show", "Provide",
    "Identify", "Consider", "Note", "Indicate", "Output", "I", "I'm",
    "I've", "I'll", "Ive", "Im", "Hey", "Hi", "Hello", "Yes", "No",
    "Yeah", "Nope", "Ok", "Okay", "Sure", "Thanks", "Thank", "Let",
    "Lets", "Dr", "Mr", "Mrs", "Ms", "Prof", "Sir", "Madam",
    "LGBTQ", "CMS",
}


def extract_name_mentions(query: str) -> list[str]:
    """Return list of capitalized tokens that could be personal first names,
    filtered against common stop words. Preserves original case."""
    hits: list[str] = []
    for m in _RE_NAME_TOKEN.finditer(query):
        tok = m.group(0)
        if tok in _NAME_STOPWORDS:
            continue
        hits.append(tok)
    return hits


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class _SpeakerAttributedBase(BestshotBase):
    """Run v2f and append speaker-filtered retrieval with optional score boost.

    Subclasses set:
      - boost: float added to cosine score of role=user segments (0.0 = none)
      - filter_mode: if True, strip role=assistant from arch output when the
        query mentions the conv-user (aggressive hard filter).
    """

    arch_name: str = "speaker_base"
    boost: float = 0.0
    filter_mode: bool = False
    # How many user-only cosine hits to append (stacked) when query mentions
    # conv-user.
    user_only_top_m: int = 5

    # Per-store cache of user_mask (bool array aligned with store.segments).
    _user_mask_cache: dict[int, np.ndarray] = {}
    # Per-store cache of conversation speakers: conv_id -> user_name (str or
    # "UNKNOWN"). Shared across instances with the same store.
    _conv_speakers_cache: dict[int, dict[str, str]] = {}

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        if client is None:
            client = OpenAI(timeout=60.0, max_retries=3)
        super().__init__(store, client)
        self.embedding_cache = SpeakerEmbeddingCache()
        self.llm_cache = SpeakerLLMCache()

        key = id(store)
        if key not in self._user_mask_cache:
            self._user_mask_cache[key] = np.array(
                [s.role == "user" for s in store.segments], dtype=bool
            )
        self.user_mask: np.ndarray = self._user_mask_cache[key]

        # Identify user speaker per conversation (cached on disk + in-memory).
        if key not in self._conv_speakers_cache:
            self._conv_speakers_cache[key] = self._identify_all_speakers(store)
        self.conv_speakers: dict[str, str] = self._conv_speakers_cache[key]

    # --- Speaker ID over all conversations in the store ---
    def _identify_all_speakers(self, store: SegmentStore) -> dict[str, str]:
        # Load persisted results first (disk cache).
        persisted: dict[str, str] = {}
        if _CONV_SPEAKERS_FILE.exists():
            try:
                with open(_CONV_SPEAKERS_FILE) as f:
                    data = json.load(f)
                persisted = data.get("speakers", {}) or {}
            except (json.JSONDecodeError, OSError):
                persisted = {}

        conv_ids = sorted({s.conversation_id for s in store.segments})
        out: dict[str, str] = dict(persisted)

        any_new = False
        for cid in conv_ids:
            if cid in out and out[cid]:
                continue
            conv_segs = [s for s in store.segments if s.conversation_id == cid]
            if not conv_segs:
                out[cid] = "UNKNOWN"
                continue
            opening = _format_opening_turns(conv_segs, num_turns=20)
            prompt = SPEAKER_ID_PROMPT.format(first_turns=opening)
            raw = self.llm_call(prompt).strip()
            # Normalize: first word, capitalized.
            name = raw.split()[0] if raw else "UNKNOWN"
            name = name.strip(".,:;\"'!?")
            if not name or name.lower() == "unknown":
                name = "UNKNOWN"
            else:
                # Preserve original capitalization if already capitalized,
                # else title-case.
                if not name[0].isupper():
                    name = name.capitalize()
            out[cid] = name
            any_new = True
            print(f"  [speaker_id] {cid}: user = {name}", flush=True)

        if any_new:
            # Persist.
            try:
                _CONV_SPEAKERS_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(_CONV_SPEAKERS_FILE, "w") as f:
                    json.dump({"speakers": out}, f, indent=2, default=str)
            except OSError:
                pass

        return out

    # --- Query mention detection ---
    def query_mentions_conv_user(
        self, query: str, conversation_id: str
    ) -> tuple[bool, str, list[str]]:
        user_name = self.conv_speakers.get(conversation_id, "UNKNOWN")
        if not user_name or user_name == "UNKNOWN":
            return False, user_name, []
        tokens = extract_name_mentions(query)
        # Case-insensitive match.
        matched = [t for t in tokens if t.lower() == user_name.lower()]
        return (len(matched) > 0, user_name, tokens)

    # --- User-only cosine search ---
    def _user_only_search(
        self,
        query_embedding: np.ndarray,
        conversation_id: str,
        top_k: int,
        exclude_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        q = query_embedding.astype(np.float32)
        qn = max(float(np.linalg.norm(q)), 1e-10)
        q = q / qn
        sims = self.store.normalized_embeddings @ q  # (N,)
        conv_mask = self.store.conversation_ids == conversation_id
        combined_mask = conv_mask & self.user_mask
        sims = np.where(combined_mask, sims, -1.0)
        if exclude_indices:
            for idx in exclude_indices:
                if 0 <= idx < len(sims):
                    sims[idx] = -1.0
        order = np.argsort(sims)[::-1][: max(top_k, 1)]
        out: list[tuple[int, float]] = []
        for i in order:
            if sims[i] <= -0.5:
                break
            out.append((int(i), float(sims[i])))
        return out

    # --- v2f run (returns ordered segments + cues metadata) ---
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

    # --- Retrieve ---
    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        v2f_segments, v2f_meta = self._run_v2f(question, conversation_id)
        mentions, user_name, name_tokens = self.query_mentions_conv_user(
            question, conversation_id
        )

        metadata: dict = {
            "name": self.arch_name,
            "v2f_cues": v2f_meta.get("cues", []),
            "v2f_turn_ids": [s.turn_id for s in v2f_segments],
            "n_v2f_segments": len(v2f_segments),
            "conv_user_name": user_name,
            "query_name_tokens": name_tokens,
            "query_mentions_conv_user": mentions,
            "applied_speaker_transform": False,
            "appended_user_only_indices": [],
            "n_user_in_v2f": int(
                sum(1 for s in v2f_segments if s.role == "user")
            ),
        }

        if not mentions:
            # No speaker transform -> return plain v2f output.
            return BestshotResult(segments=list(v2f_segments), metadata=metadata)

        metadata["applied_speaker_transform"] = True

        # Append top-M user-only cosine hits not already in v2f.
        query_emb = self.embed_text(question)
        v2f_idx_set: set[int] = {s.index for s in v2f_segments}
        user_hits = self._user_only_search(
            query_emb,
            conversation_id,
            top_k=self.user_only_top_m + 5,  # widen a little for dedup
            exclude_indices=v2f_idx_set,
        )
        appended: list[tuple[int, float]] = user_hits[: self.user_only_top_m]
        appended_segments = [self.store.segments[i] for i, _ in appended]

        metadata["appended_user_only_indices"] = [i for i, _ in appended]

        if self.filter_mode:
            # Aggressive: drop role=assistant from v2f output, then append
            # user-only hits. Backfill (outside via fair-backfill eval) pulls
            # from cosine top-K; it may re-add assistant turns, which is fine
            # — the ARCH's retrieval itself is user-only.
            filtered_v2f = [s for s in v2f_segments if s.role == "user"]
            merged = list(filtered_v2f)
            seen = {s.index for s in merged}
            for s in appended_segments:
                if s.index not in seen:
                    merged.append(s)
                    seen.add(s.index)
            metadata["filter_dropped"] = len(v2f_segments) - len(filtered_v2f)
            return BestshotResult(segments=merged, metadata=metadata)

        # Boost variant: re-sort v2f output + appended user-only hits by
        # (cosine score + bonus if role=user). We don't have cosine scores for
        # v2f intermediate segments tracked per-hop, but we CAN recompute
        # cosine sim (query vs segment) to get an ordering signal. Simpler:
        # keep the v2f stacked order intact, but promote user-role items.
        #
        # Mechanism: take the v2f list as an ordered baseline. For each pair of
        # adjacent items where the LATER one is role=user and score-bumped by
        # `boost` exceeds the EARLIER one's score by enough, swap. This is
        # expensive. Cleaner: compute cosine for each v2f segment, apply boost
        # to user-role, resort stable.
        qn = float(np.linalg.norm(query_emb))
        qn = max(qn, 1e-10)
        qnorm = query_emb / qn

        def _cosine(idx: int) -> float:
            return float(
                self.store.normalized_embeddings[idx] @ qnorm.astype(np.float32)
            )

        # Build a (segment, score-after-boost, original_rank) list.
        v2f_with_scores: list[tuple[Segment, float, int]] = []
        for rank, seg in enumerate(v2f_segments):
            s = _cosine(seg.index)
            if seg.role == "user":
                s += self.boost
            v2f_with_scores.append((seg, s, rank))

        appended_with_scores: list[tuple[Segment, float, int]] = []
        base_rank = len(v2f_segments)
        for rank, (idx, sim) in enumerate(appended):
            s = sim + self.boost  # all appended are role=user
            appended_with_scores.append(
                (self.store.segments[idx], s, base_rank + rank)
            )

        combined: list[tuple[Segment, float, int]] = (
            v2f_with_scores + appended_with_scores
        )
        # Dedupe by seg.index (v2f may overlap appended via score — shouldn't
        # happen since we excluded, but be defensive).
        seen_idx: set[int] = set()
        unique: list[tuple[Segment, float, int]] = []
        for tup in combined:
            if tup[0].index in seen_idx:
                continue
            seen_idx.add(tup[0].index)
            unique.append(tup)

        # Sort by (boosted score desc, original rank asc) for stable tie-break.
        unique.sort(key=lambda t: (-t[1], t[2]))
        merged_segments = [tup[0] for tup in unique]
        metadata["boost_applied"] = self.boost
        return BestshotResult(segments=merged_segments, metadata=metadata)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
class SpeakerBoost002(_SpeakerAttributedBase):
    arch_name = "speaker_boost_0.02"
    boost = 0.02
    filter_mode = False


class SpeakerBoost005(_SpeakerAttributedBase):
    arch_name = "speaker_boost_0.05"
    boost = 0.05
    filter_mode = False


class SpeakerUserFilter(_SpeakerAttributedBase):
    arch_name = "speaker_user_filter"
    boost = 0.0
    filter_mode = True


ARCH_CLASSES: dict[str, type] = {
    "speaker_boost_0.02": SpeakerBoost002,
    "speaker_boost_0.05": SpeakerBoost005,
    "speaker_user_filter": SpeakerUserFilter,
}
