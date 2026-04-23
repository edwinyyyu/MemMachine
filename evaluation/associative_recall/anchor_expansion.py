"""Anchor-turn expansion retrieval.

For each question q at budget K:
  1. Initial cosine retrieval: top-3 (or top-5) from raw query = ANCHOR turns.
  2. For each anchor turn, one LLM call imagines 2 neighboring turns that could
     appear immediately before or after it — turns that continue, elaborate,
     correct, or answer its content. Cues are anchored in real corpus vocabulary.
  3. Embed each imagined cue; retrieve top-5 per cue.
  4. Merge all retrieved segments (including anchors), dedupe by index, rank
     by max cosine across all probes (raw query + anchor-continuation cues).

Variants
  anchor_exp_3anchors   — 3 anchors x 2 cues = 6 imagined turns
  anchor_exp_5anchors   — 5 anchors x 2 cues = 10 imagined turns
  anchor_exp_plus_v2f   — anchor_exp (3) UNION v2f cues

Motivation
  v2f imagines chat content FORWARD from the question. Anchor expansion starts
  from ACTUAL retrieved turns and imagines continuation content — cues land in
  corpus vocabulary, not query vocabulary. Per-cue attribution shows winning
  cues sit ~0.575 cosine from gold and are entity-rich; anchoring in retrieved
  content should make cues more like winners.

Dedicated caches (anchor_*_cache.json) avoid concurrent-agent corruption.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

from associative_recall import CACHE_DIR, EMBED_MODEL, EmbeddingCache, LLMCache, Segment, SegmentStore
from best_shot import (
    MODEL,
    BestshotBase,
    BestshotResult,
    V2F_PROMPT,
    _format_segments,
    _parse_cues,
)


# ---------------------------------------------------------------------------
# Dedicated caches — read many shared caches for hits, write only to own files
# ---------------------------------------------------------------------------

_ANCHOR_EMB_FILE = CACHE_DIR / "anchor_embedding_cache.json"
_ANCHOR_LLM_FILE = CACHE_DIR / "anchor_llm_cache.json"

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
)


class AnchorEmbeddingCache(EmbeddingCache):
    """Reads shared embedding caches (best-effort), writes to dedicated file."""

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
        self.cache_file = _ANCHOR_EMB_FILE
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


class AnchorLLMCache(LLMCache):
    """Reads shared LLM caches (best-effort), writes to dedicated file."""

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
        self.cache_file = _ANCHOR_LLM_FILE
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
# Prompts
# ---------------------------------------------------------------------------

# One LLM call per anchor turn. Anchor content placed FIRST and labeled
# prominently so the LLM doesn't just riff on the question.
ANCHOR_EXPANSION_PROMPT = """\
You are reading a conversation. This specific turn was retrieved as potentially \
relevant to a search, but the full answer may be in NEIGHBORING turns (before \
or after). Imagine 2 plausible conversation turns that could appear immediately \
before or after this turn — turns that continue, elaborate, correct, or answer \
its content.

Anchor turn ({role}): {anchor_text}

Original search question (for context): {question}

Output exactly 2 imagined turns as:
CUE: <imagined turn 1, 1-2 sentences, casual chat register>
CUE: <imagined turn 2, 1-2 sentences>"""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _AnchorExpansionBase(BestshotBase):
    """Anchor-turn expansion: for each top-N anchor turn, imagine N_cues
    continuations (one LLM call per anchor) and retrieve top-K per cue.
    """

    arch_name: str = "anchor_expansion"
    n_anchors: int = 3  # number of top-cosine anchor turns to expand from
    n_cues_per_anchor: int = 2  # number of imagined cues per anchor
    per_probe_top_k: int = 5  # top-k per imagined cue
    include_v2f: bool = False  # union imagined cues with v2f cues

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        # Use a shorter per-request timeout so a single stalled call can't
        # block the whole run. We retry below.
        if client is None:
            client = OpenAI(timeout=45.0, max_retries=3)
        super().__init__(store, client)
        self.embedding_cache = AnchorEmbeddingCache()
        self.llm_cache = AnchorLLMCache()

    def llm_call(self, prompt: str, model: str = MODEL) -> str:
        """Retrying LLM call — avoids single-call stalls hanging the run."""
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
        # Final fallback — record empty so we don't spin indefinitely
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

    def _anchor_prompt(self, anchor: Segment, question: str) -> str:
        # Trim excessively long anchor text to keep prompts stable
        text = anchor.text.replace("\n", " ").strip()
        if len(text) > 600:
            text = text[:600].rstrip() + "..."
        return ANCHOR_EXPANSION_PROMPT.format(
            role=anchor.role,
            anchor_text=text,
            question=question,
        )

    def generate_anchor_cues(
        self, anchors: list[Segment], question: str
    ) -> list[dict]:
        """One LLM call per anchor, executed in parallel threads. Returns
        list of dicts with source_turn_id, source_index, cue.
        """
        prompts = [self._anchor_prompt(a, question) for a in anchors]
        # Use up to n_anchors threads; OpenAI calls are I/O bound so the GIL
        # is fine.
        with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as pool:
            outputs = list(pool.map(self.llm_call, prompts))

        outcomes: list[dict] = []
        for anchor, output in zip(anchors, outputs):
            cues = _parse_cues(output)[: self.n_cues_per_anchor]
            for cue in cues:
                outcomes.append(
                    {
                        "source_turn_id": anchor.turn_id,
                        "source_index": anchor.index,
                        "source_role": anchor.role,
                        "anchor_text": anchor.text,
                        "cue": cue,
                        "raw_output": output,
                    }
                )
        return outcomes

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        # Step 1: raw query top-10 (need at least n_anchors, keep 10 for
        # merging fallback)
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        hop0_segments = list(hop0.segments)
        hop0_scores = list(hop0.scores)

        if not hop0_segments:
            return BestshotResult(
                segments=[],
                metadata={
                    "name": self.arch_name,
                    "anchor_cues": [],
                    "v2f_cues": [],
                    "num_probes": 1,
                    "hop0_empty": True,
                },
            )

        score_map: dict[int, float] = {}
        seg_map: dict[int, Segment] = {}
        for seg, sc in zip(hop0_segments, hop0_scores):
            score_map[seg.index] = sc
            seg_map[seg.index] = seg

        # Step 2: anchors = top-N of hop0
        anchors = hop0_segments[: self.n_anchors]
        anchor_outcomes = self.generate_anchor_cues(anchors, question)

        # Step 3: embed cues in parallel, then retrieve per anchor cue
        cue_texts = [ac["cue"] for ac in anchor_outcomes]
        if cue_texts:
            with ThreadPoolExecutor(max_workers=max(1, len(cue_texts))) as pool:
                cue_embs = list(pool.map(self.embed_text, cue_texts))
        else:
            cue_embs = []

        probe_outcomes: list[dict] = []
        for ac, cue_emb in zip(anchor_outcomes, cue_embs):
            res = self.store.search(
                cue_emb,
                top_k=self.per_probe_top_k,
                conversation_id=conversation_id,
            )
            retrieved_ids = []
            for seg, sc in zip(res.segments, res.scores):
                retrieved_ids.append(seg.index)
                if seg.index not in score_map or sc > score_map[seg.index]:
                    score_map[seg.index] = sc
                if seg.index not in seg_map:
                    seg_map[seg.index] = seg
            probe_outcomes.append(
                {
                    "source_turn_id": ac["source_turn_id"],
                    "source_role": ac["source_role"],
                    "anchor_text": ac["anchor_text"],
                    "cue": ac["cue"],
                    "retrieved_turn_ids": [
                        seg_map[idx].turn_id for idx in retrieved_ids
                    ],
                }
            )

        # Optional step 4: v2f cues on same hop0 (for union variant)
        v2f_cues: list[str] = []
        v2f_outcomes: list[dict] = []
        if self.include_v2f:
            context_section = (
                "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
                + _format_segments(hop0_segments)
            )
            v2f_prompt = V2F_PROMPT.format(
                question=question, context_section=context_section
            )
            v2f_output = self.llm_call(v2f_prompt)
            v2f_cues = _parse_cues(v2f_output)[:2]
            for cue in v2f_cues:
                cue_emb = self.embed_text(cue)
                res = self.store.search(
                    cue_emb, top_k=10, conversation_id=conversation_id
                )
                retrieved_ids = []
                for seg, sc in zip(res.segments, res.scores):
                    retrieved_ids.append(seg.index)
                    if seg.index not in score_map or sc > score_map[seg.index]:
                        score_map[seg.index] = sc
                    if seg.index not in seg_map:
                        seg_map[seg.index] = seg
                v2f_outcomes.append(
                    {
                        "cue": cue,
                        "retrieved_turn_ids": [
                            seg_map[idx].turn_id for idx in retrieved_ids
                        ],
                    }
                )

        # Step 5: rank by max score
        ranked_indices = sorted(
            score_map.keys(), key=lambda idx: score_map[idx], reverse=True
        )
        all_segments = [seg_map[idx] for idx in ranked_indices]

        anchor_cues_flat = [ac["cue"] for ac in anchor_outcomes]

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": self.arch_name,
                "anchor_cues": anchor_cues_flat,
                "anchor_turn_ids": [a.turn_id for a in anchors],
                "probe_outcomes": probe_outcomes,
                "v2f_cues": v2f_cues,
                "v2f_outcomes": v2f_outcomes,
                "num_probes": 1 + len(anchor_outcomes) + len(v2f_cues),
                "hop0_empty": False,
                "n_anchors": self.n_anchors,
                "n_cues_per_anchor": self.n_cues_per_anchor,
            },
        )


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


class AnchorExp3Anchors(_AnchorExpansionBase):
    """Main variant: 3 anchors x 2 cues = 6 imagined turns."""

    arch_name = "anchor_exp_3anchors"
    n_anchors = 3
    n_cues_per_anchor = 2
    per_probe_top_k = 5
    include_v2f = False


class AnchorExp5Anchors(_AnchorExpansionBase):
    """Wider variant: 5 anchors x 2 cues = 10 imagined turns."""

    arch_name = "anchor_exp_5anchors"
    n_anchors = 5
    n_cues_per_anchor = 2
    per_probe_top_k = 5
    include_v2f = False


class AnchorExpPlusV2f(_AnchorExpansionBase):
    """Union variant: anchor_exp (3 anchors) + v2f cues."""

    arch_name = "anchor_exp_plus_v2f"
    n_anchors = 3
    n_cues_per_anchor = 2
    per_probe_top_k = 5
    include_v2f = True


ARCH_CLASSES: dict[str, type] = {
    "anchor_exp_3anchors": AnchorExp3Anchors,
    "anchor_exp_5anchors": AnchorExp5Anchors,
    "anchor_exp_plus_v2f": AnchorExpPlusV2f,
}
