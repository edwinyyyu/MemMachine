"""Speaker-conditional cue generation.

When a query mentions one of the two conversation participants' names, we
condition v2f's cue-generation to generate cues AS IF that participant were
speaking — first-person, casual chat register. The motivation is that gold
turns for "What did Caroline say..." are usually Caroline's OWN first-person
utterances; cues embedded in the same register should cosine-match more
tightly than generic third-person paraphrases.

Three variants in this file:
  - speaker_cond_cue_only        conditioning on; no role-filter
  - speaker_cond_plus_filter     conditioning on + role-filter (best-expected)
  - v2f_mention_tag              just adds "The question is about X" hint
                                 (ablation — tests whether NAME HINT alone
                                 suffices vs first-person conditioning)

All variants reuse the two-speaker ID map in
`results/conversation_two_speakers.json` (read-only).

Dedicated caches `speakerCue_*_cache.json` — these do NOT overwrite any
other agent's cache files.
"""

from __future__ import annotations

import json
import os
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
    BestshotBase,
    BestshotResult,
    V2F_PROMPT,
    _format_segments,
    _parse_cues,
)
from speaker_attributed import extract_name_mentions


# ---------------------------------------------------------------------------
# Dedicated caches
# ---------------------------------------------------------------------------
_SC_EMB_FILE = CACHE_DIR / "speakerCue_embedding_cache.json"
_SC_LLM_FILE = CACHE_DIR / "speakerCue_llm_cache.json"

_CONV_TWO_SPEAKERS_FILE = (
    Path(__file__).resolve().parent
    / "results"
    / "conversation_two_speakers.json"
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
    "speakerCue_embedding_cache.json",
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
    "antipara_llm_cache.json",
    "speakerCue_llm_cache.json",
)


class SpeakerCueEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _SC_EMB_FILE
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


class SpeakerCueLLMCache(LLMCache):
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
        self.cache_file = _SC_LLM_FILE
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
# Prompt templates
# ---------------------------------------------------------------------------
SPEAKER_COND_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

The question is about {speaker_name}. Generate cues AS IF {speaker_name} \
themselves is speaking — first-person, casual chat register, using specific \
vocabulary that {speaker_name} would use to describe their own experiences, \
plans, feelings, or observations. Do NOT write third-person paraphrases \
("{speaker_name} said that..."). Write utterances that {speaker_name} would \
actually send as a chat message.

Question: {question}

{context_section}

Generate exactly 2 cues. Each cue should be 1-2 sentences in {speaker_name}'s \
first-person voice.
Format:
CUE: <text>
CUE: <text>
Nothing else."""


MENTION_TAG_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

The question is about {speaker_name}.

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
# Base class
# ---------------------------------------------------------------------------
class _SpeakerCondBase(BestshotBase):
    """Run v2f with speaker-conditioned cue generation when the query mentions
    exactly one of the two known conversation participants. Falls back to
    standard v2f when no side or both sides are mentioned.

    Subclass contract:
      - prompt_template_cond: prompt used when query mentions one side.
      - filter_mode: if True, apply role-filter + backfill (like
                     two_speaker_filter) when the transform fires.
    """

    arch_name: str = "speaker_cond_base"
    prompt_template_cond: str = SPEAKER_COND_PROMPT
    filter_mode: bool = False
    role_only_top_m: int = 5

    _role_masks_cache: dict[int, dict[str, np.ndarray]] = {}
    _conv_two_speakers_cache: dict[int, dict[str, dict[str, str]]] = {}

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        if client is None:
            client = OpenAI(timeout=60.0, max_retries=3)
        super().__init__(store, client)
        self.embedding_cache = SpeakerCueEmbeddingCache()
        self.llm_cache = SpeakerCueLLMCache()

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
            self._conv_two_speakers_cache[key] = self._load_two_speakers()
        self.conv_two_speakers: dict[str, dict[str, str]] = (
            self._conv_two_speakers_cache[key]
        )

    # --- Load the persisted two-speaker ID map (read-only) ---
    def _load_two_speakers(self) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        if not _CONV_TWO_SPEAKERS_FILE.exists():
            return out
        try:
            with open(_CONV_TWO_SPEAKERS_FILE) as f:
                data = json.load(f)
            raw = data.get("speakers", {}) or {}
            for cid, pair in raw.items():
                if isinstance(pair, dict):
                    out[cid] = {
                        "user": pair.get("user", "UNKNOWN") or "UNKNOWN",
                        "assistant": pair.get("assistant", "UNKNOWN")
                        or "UNKNOWN",
                    }
        except (json.JSONDecodeError, OSError):
            pass
        return out

    # --- Query mention classification ---
    def classify_query(
        self, query: str, conversation_id: str
    ) -> tuple[str, str, str, list[str]]:
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
        role_mask = self.role_masks.get(
            role, np.zeros_like(conv_mask, dtype=bool)
        )
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

    # --- retrieve ---
    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(hop0.segments)
        exclude: set[int] = {s.index for s in all_segments}

        side, user_name, asst_name, name_tokens = self.classify_query(
            question, conversation_id
        )

        # Choose the speaker we're conditioning on; only ONE-side mentions.
        conditioned_on: str | None = None
        if side == "user" and user_name != "UNKNOWN":
            conditioned_on = user_name
        elif side == "assistant" and asst_name != "UNKNOWN":
            conditioned_on = asst_name

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(all_segments)
        )

        if conditioned_on is not None:
            prompt = self.prompt_template_cond.format(
                speaker_name=conditioned_on,
                question=question,
                context_section=context_section,
            )
        else:
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

        metadata: dict = {
            "name": self.arch_name,
            "v2f_cues": cues,
            "v2f_raw_output": output,
            "v2f_turn_ids": [s.turn_id for s in all_segments],
            "n_v2f_segments": len(all_segments),
            "conv_user_name": user_name,
            "conv_assistant_name": asst_name,
            "query_name_tokens": name_tokens,
            "matched_side": side,
            "conditioned_on": conditioned_on,
            "applied_cue_conditioning": conditioned_on is not None,
            "applied_filter": False,
            "n_user_in_v2f": int(
                sum(1 for s in all_segments if s.role == "user")
            ),
            "n_assistant_in_v2f": int(
                sum(1 for s in all_segments if s.role == "assistant")
            ),
        }

        # filter_mode: apply role-filter + role-only backfill if side fires.
        if (
            self.filter_mode
            and side in ("user", "assistant")
            and conditioned_on is not None
        ):
            matched_role = side
            v2f_idx_set = {s.index for s in all_segments}
            role_hits = self._role_only_search(
                query_emb,
                conversation_id,
                matched_role,
                top_k=self.role_only_top_m + 5,
                exclude_indices=v2f_idx_set,
            )
            appended = role_hits[: self.role_only_top_m]
            appended_segments = [
                self.store.segments[i] for i, _ in appended
            ]
            filtered_v2f = [
                s for s in all_segments if s.role == matched_role
            ]
            merged = list(filtered_v2f)
            seen = {s.index for s in merged}
            for s in appended_segments:
                if s.index not in seen:
                    merged.append(s)
                    seen.add(s.index)
            metadata["applied_filter"] = True
            metadata["filter_dropped"] = len(all_segments) - len(filtered_v2f)
            metadata["appended_role_only_indices"] = [i for i, _ in appended]
            return BestshotResult(segments=merged, metadata=metadata)

        return BestshotResult(segments=list(all_segments), metadata=metadata)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
class SpeakerCondCueOnly(_SpeakerCondBase):
    arch_name = "speaker_cond_cue_only"
    prompt_template_cond = SPEAKER_COND_PROMPT
    filter_mode = False


class SpeakerCondPlusFilter(_SpeakerCondBase):
    arch_name = "speaker_cond_plus_filter"
    prompt_template_cond = SPEAKER_COND_PROMPT
    filter_mode = True


class V2fMentionTag(_SpeakerCondBase):
    """Ablation: add 'The question is about X' hint without first-person
    conditioning. Tests whether the NAME HINT alone is what helps."""

    arch_name = "v2f_mention_tag"
    prompt_template_cond = MENTION_TAG_PROMPT
    filter_mode = False


ARCH_CLASSES: dict[str, type] = {
    "speaker_cond_cue_only": SpeakerCondCueOnly,
    "speaker_cond_plus_filter": SpeakerCondPlusFilter,
    "v2f_mention_tag": V2fMentionTag,
}
