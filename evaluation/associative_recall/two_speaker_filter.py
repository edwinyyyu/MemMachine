"""Two-speaker attributed retrieval.

Extends the single-speaker `speaker_user_filter` to cover BOTH conversation
participants. LoCoMo conversations have TWO humans (e.g. Caroline <-> Melanie
in conv-26), with one arbitrarily mapped to the `user` role and one to the
`assistant` role. The single-speaker filter covers 60% of LoCoMo-30 queries
(those that mention the user-role name). The other 40% of queries mention
the OTHER participant's name; for those, filtering role=assistant should
yield the same mechanism.

Pipeline
--------
Ingest (once per conversation, 1 LLM call):
  LLM reads the first ~20 turns with role labels. Outputs BOTH participants'
  first names: `user: <name>` and `assistant: <name>` (or UNKNOWN). Cached to
  disk, extending `conversation_speakers.json` format with a two-speaker map.

Query-time (per question):
  - Extract capitalized first-name tokens from query (reuse
    speaker_attributed.extract_name_mentions).
  - Match against the conv's user_name and assistant_name.
  - Cases:
      * Mentions user only       -> hard filter role=user
      * Mentions assistant only  -> hard filter role=assistant
      * Mentions both            -> no filter (interaction query)
      * Mentions neither         -> no filter (normal v2f)

Variants:
  - two_speaker_filter            (hard role-filter when ONE side is mentioned)
  - two_speaker_boost_0.05        (score bonus on matched role instead of hard
                                   filter; mirrors speaker_boost_0.05)

All variants share the cache files `two_speaker_*_cache.json` and the
reusable speaker map file `conversation_two_speakers.json`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from best_shot import (
    V2F_PROMPT,
    BestshotBase,
    BestshotResult,
    _format_segments,
    _parse_cues,
)
from openai import OpenAI
from speaker_attributed import (
    extract_name_mentions,
)

# ---------------------------------------------------------------------------
# Dedicated caches
# ---------------------------------------------------------------------------
_TS_EMB_FILE = CACHE_DIR / "two_speaker_embedding_cache.json"
_TS_LLM_FILE = CACHE_DIR / "two_speaker_llm_cache.json"
_CONV_TWO_SPEAKERS_FILE = (
    Path(__file__).resolve().parent / "results" / "conversation_two_speakers.json"
)
_CONV_ONE_SPEAKER_FILE = (
    Path(__file__).resolve().parent / "results" / "conversation_speakers.json"
)

# Warm-start read paths (shared caches we can reuse but not pollute).
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
    "two_speaker_embedding_cache.json",
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
    "two_speaker_llm_cache.json",
    # antipara LAST so V2F_PROMPT cues are identical to MetaV2fDedicated.
    "antipara_llm_cache.json",
)


class TwoSpeakerEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _TS_EMB_FILE
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


class TwoSpeakerLLMCache(LLMCache):
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
        self.cache_file = _TS_LLM_FILE
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
# Two-speaker identification prompt
# ---------------------------------------------------------------------------
TWO_SPEAKER_ID_PROMPT = """\
Read these opening conversation turns. Identify the first names of BOTH \
participants. The "user" role is one participant; the "assistant" role is \
the other. Both may be human participants in a casual conversation.

{first_turns}

Output exactly two lines in this format (no extra commentary):
user: <first name or UNKNOWN>
assistant: <first name or UNKNOWN>
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


def _parse_two_speaker_output(raw: str) -> tuple[str, str]:
    """Parse 'user: X\\nassistant: Y' -> (X, Y). UNKNOWN on unparseable."""
    user_name = "UNKNOWN"
    asst_name = "UNKNOWN"
    if not raw:
        return user_name, asst_name
    for line in raw.strip().split("\n"):
        line = line.strip()
        low = line.lower()
        if low.startswith("user:") or low.startswith("user "):
            val = line.split(":", 1)[1].strip() if ":" in line else ""
            user_name = _normalize_name(val)
        elif low.startswith("assistant:") or low.startswith("assistant "):
            val = line.split(":", 1)[1].strip() if ":" in line else ""
            asst_name = _normalize_name(val)
    return user_name, asst_name


def _normalize_name(raw: str) -> str:
    if not raw:
        return "UNKNOWN"
    # Take the first token, strip punctuation.
    tok = raw.split(maxsplit=1)[0] if raw.split() else ""
    tok = tok.strip(".,:;\"'!?()[]<>")
    if not tok or tok.lower() == "unknown":
        return "UNKNOWN"
    if not tok[0].isupper():
        tok = tok.capitalize()
    return tok


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class _TwoSpeakerAttributedBase(BestshotBase):
    """Run v2f, then apply two-sided role filter / boost when the query
    mentions exactly one of the two conversation participants' names.

    Subclass contract:
      - boost: float added to cosine score of matched-role segments
               (0.0 = hard filter mode).
      - filter_mode: if True, drop role!=matched from arch output when
                     the query mentions exactly one side.
    """

    arch_name: str = "two_speaker_base"
    boost: float = 0.0
    filter_mode: bool = False
    # How many role-filtered cosine hits to append when query mentions a side.
    role_only_top_m: int = 5

    # Per-store caches.
    _role_masks_cache: dict[int, dict[str, np.ndarray]] = {}
    _conv_two_speakers_cache: dict[int, dict[str, dict[str, str]]] = {}

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        if client is None:
            client = OpenAI(timeout=60.0, max_retries=3)
        super().__init__(store, client)
        self.embedding_cache = TwoSpeakerEmbeddingCache()
        self.llm_cache = TwoSpeakerLLMCache()

        key = id(store)
        if key not in self._role_masks_cache:
            self._role_masks_cache[key] = {
                "user": np.array(
                    [s.role == "user" for s in store.segments], dtype=bool
                ),
                "assistant": np.array(
                    [s.role == "assistant" for s in store.segments], dtype=bool
                ),
            }
        self.role_masks: dict[str, np.ndarray] = self._role_masks_cache[key]

        if key not in self._conv_two_speakers_cache:
            self._conv_two_speakers_cache[key] = self._identify_all_speakers(store)
        # dict: conv_id -> {"user": name, "assistant": name}
        self.conv_two_speakers: dict[str, dict[str, str]] = (
            self._conv_two_speakers_cache[key]
        )

    # --- Speaker ID over all conversations in the store ---
    def _identify_all_speakers(self, store: SegmentStore) -> dict[str, dict[str, str]]:
        # Load persisted two-speaker results first (disk cache).
        persisted: dict[str, dict[str, str]] = {}
        if _CONV_TWO_SPEAKERS_FILE.exists():
            try:
                with open(_CONV_TWO_SPEAKERS_FILE) as f:
                    data = json.load(f)
                raw = data.get("speakers", {}) or {}
                for cid, pair in raw.items():
                    if isinstance(pair, dict):
                        persisted[cid] = {
                            "user": pair.get("user", "UNKNOWN") or "UNKNOWN",
                            "assistant": pair.get("assistant", "UNKNOWN") or "UNKNOWN",
                        }
            except (json.JSONDecodeError, OSError):
                persisted = {}

        # Fallback: warm-start user-side from the single-speaker file. We
        # still need an LLM call for the assistant side, but we can skip
        # re-confirming the user name where it's already known & non-UNKNOWN.
        one_side: dict[str, str] = {}
        if _CONV_ONE_SPEAKER_FILE.exists():
            try:
                with open(_CONV_ONE_SPEAKER_FILE) as f:
                    one_side = json.load(f).get("speakers", {}) or {}
            except (json.JSONDecodeError, OSError):
                one_side = {}

        conv_ids = sorted({s.conversation_id for s in store.segments})
        out: dict[str, dict[str, str]] = dict(persisted)
        any_new = False

        for cid in conv_ids:
            existing = out.get(cid)
            # Skip if we already have a fully-identified pair on disk.
            if existing and (
                existing.get("user", "UNKNOWN") != "UNKNOWN"
                or existing.get("assistant", "UNKNOWN") != "UNKNOWN"
            ):
                # Have at least partial info; don't re-call LLM.
                continue

            conv_segs = [s for s in store.segments if s.conversation_id == cid]
            if not conv_segs:
                out[cid] = {"user": "UNKNOWN", "assistant": "UNKNOWN"}
                continue

            opening = _format_opening_turns(conv_segs, num_turns=20)
            prompt = TWO_SPEAKER_ID_PROMPT.format(first_turns=opening)
            raw = self.llm_call(prompt)
            user_name, asst_name = _parse_two_speaker_output(raw)

            # If single-speaker file already identified the user, keep that
            # (it was derived under a slightly different prompt but should
            # agree; be conservative).
            prior_user = one_side.get(cid, "UNKNOWN")
            if prior_user and prior_user != "UNKNOWN" and user_name == "UNKNOWN":
                user_name = prior_user

            out[cid] = {"user": user_name, "assistant": asst_name}
            any_new = True
            print(
                f"  [two_speaker_id] {cid}: user={user_name} assistant={asst_name}",
                flush=True,
            )

        if any_new:
            try:
                _CONV_TWO_SPEAKERS_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(_CONV_TWO_SPEAKERS_FILE, "w") as f:
                    json.dump({"speakers": out}, f, indent=2, default=str)
            except OSError:
                pass

        return out

    # --- Query mention classification ---
    def classify_query(
        self, query: str, conversation_id: str
    ) -> tuple[str, str, str, list[str]]:
        """Returns (side, user_name, assistant_name, name_tokens).

        side is one of:
          "user"      -> mentions user only
          "assistant" -> mentions assistant only
          "both"      -> mentions both
          "none"      -> mentions neither (or names UNKNOWN)
        """
        pair = self.conv_two_speakers.get(
            conversation_id, {"user": "UNKNOWN", "assistant": "UNKNOWN"}
        )
        user_name = pair.get("user", "UNKNOWN") or "UNKNOWN"
        asst_name = pair.get("assistant", "UNKNOWN") or "UNKNOWN"
        tokens = extract_name_mentions(query)
        tlow = {t.lower() for t in tokens}

        hit_user = user_name != "UNKNOWN" and user_name.lower() in tlow
        hit_asst = asst_name != "UNKNOWN" and asst_name.lower() in tlow
        if hit_user and hit_asst:
            side = "both"
        elif hit_user:
            side = "user"
        elif hit_asst:
            side = "assistant"
        else:
            side = "none"
        return side, user_name, asst_name, tokens

    # --- Role-filtered cosine search ---
    def _role_only_search(
        self,
        query_embedding: np.ndarray,
        conversation_id: str,
        role: str,
        top_k: int,
        exclude_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        q = query_embedding.astype(np.float32)
        qn = max(float(np.linalg.norm(q)), 1e-10)
        q = q / qn
        sims = self.store.normalized_embeddings @ q
        conv_mask = self.store.conversation_ids == conversation_id
        role_mask = self.role_masks.get(role, np.zeros_like(conv_mask, dtype=bool))
        combined_mask = conv_mask & role_mask
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

    # --- v2f run ---
    def _run_v2f(
        self, question: str, conversation_id: str
    ) -> tuple[list[Segment], dict]:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments: list[Segment] = list(hop0.segments)
        exclude: set[int] = {s.index for s in all_segments}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(all_segments)
        )
        prompt = V2F_PROMPT.format(question=question, context_section=context_section)
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
        side, user_name, asst_name, name_tokens = self.classify_query(
            question, conversation_id
        )

        metadata: dict = {
            "name": self.arch_name,
            "v2f_cues": v2f_meta.get("cues", []),
            "v2f_turn_ids": [s.turn_id for s in v2f_segments],
            "n_v2f_segments": len(v2f_segments),
            "conv_user_name": user_name,
            "conv_assistant_name": asst_name,
            "query_name_tokens": name_tokens,
            "matched_side": side,
            "applied_speaker_transform": False,
            "appended_role_only_indices": [],
            "n_user_in_v2f": int(sum(1 for s in v2f_segments if s.role == "user")),
            "n_assistant_in_v2f": int(
                sum(1 for s in v2f_segments if s.role == "assistant")
            ),
        }

        # No transform if: nothing mentioned, both mentioned, or names unknown.
        if side in ("none", "both"):
            return BestshotResult(segments=list(v2f_segments), metadata=metadata)

        # side in ("user", "assistant") -> filter/boost that role.
        matched_role = side
        metadata["applied_speaker_transform"] = True

        query_emb = self.embed_text(question)
        v2f_idx_set: set[int] = {s.index for s in v2f_segments}
        role_hits = self._role_only_search(
            query_emb,
            conversation_id,
            matched_role,
            top_k=self.role_only_top_m + 5,
            exclude_indices=v2f_idx_set,
        )
        appended: list[tuple[int, float]] = role_hits[: self.role_only_top_m]
        appended_segments = [self.store.segments[i] for i, _ in appended]
        metadata["appended_role_only_indices"] = [i for i, _ in appended]

        if self.filter_mode:
            # Drop role != matched_role from the v2f output.
            filtered_v2f = [s for s in v2f_segments if s.role == matched_role]
            merged = list(filtered_v2f)
            seen = {s.index for s in merged}
            for s in appended_segments:
                if s.index not in seen:
                    merged.append(s)
                    seen.add(s.index)
            metadata["filter_dropped"] = len(v2f_segments) - len(filtered_v2f)
            return BestshotResult(segments=merged, metadata=metadata)

        # Boost variant: re-sort v2f + appended by cosine + bonus for matched.
        qn = max(float(np.linalg.norm(query_emb)), 1e-10)
        qnorm = query_emb / qn

        def _cosine(idx: int) -> float:
            return float(
                self.store.normalized_embeddings[idx] @ qnorm.astype(np.float32)
            )

        scored: list[tuple[Segment, float, int]] = []
        for rank, seg in enumerate(v2f_segments):
            s = _cosine(seg.index)
            if seg.role == matched_role:
                s += self.boost
            scored.append((seg, s, rank))

        base_rank = len(v2f_segments)
        for rank, (idx, sim) in enumerate(appended):
            s = sim + self.boost  # all appended are matched-role
            scored.append((self.store.segments[idx], s, base_rank + rank))

        # Dedup by idx, preserving the first occurrence.
        seen_idx: set[int] = set()
        unique: list[tuple[Segment, float, int]] = []
        for tup in scored:
            if tup[0].index in seen_idx:
                continue
            seen_idx.add(tup[0].index)
            unique.append(tup)

        unique.sort(key=lambda t: (-t[1], t[2]))
        merged_segments = [tup[0] for tup in unique]
        metadata["boost_applied"] = self.boost
        return BestshotResult(segments=merged_segments, metadata=metadata)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
class TwoSpeakerFilter(_TwoSpeakerAttributedBase):
    arch_name = "two_speaker_filter"
    boost = 0.0
    filter_mode = True


class TwoSpeakerBoost005(_TwoSpeakerAttributedBase):
    arch_name = "two_speaker_boost_0.05"
    boost = 0.05
    filter_mode = False


ARCH_CLASSES: dict[str, type] = {
    "two_speaker_filter": TwoSpeakerFilter,
    "two_speaker_boost_0.05": TwoSpeakerBoost005,
}
