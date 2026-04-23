"""Deep-narrow iterative retrieval ("20-questions" style).

Each hop generates ONE highly-conditioned cue asking "what's the most
informative thing I still don't know?", with many hops (default 18) and
multiple stop signals (saturation, self-assessed complete, budget cap).

Variants:
  DeepNarrowV1        - max_hops=18, top_k_per_hop=5, STOP allowed.
  DeepNarrowWideProbe - max_hops=18, top_k_per_hop=10, STOP allowed.
  DeepNarrowNoStop    - max_hops=18, top_k_per_hop=5, STOP disabled.

Usage:
    from deep_narrow import DeepNarrowV1, DeepNarrowWideProbe, DeepNarrowNoStop
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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

MODEL = "gpt-5-mini"

# ---------------------------------------------------------------------------
# Caches — read from many prior caches, write to deep_narrow-specific files.
# ---------------------------------------------------------------------------
CACHE_FILE_EMB = CACHE_DIR / "deep_narrow_embedding_cache.json"
CACHE_FILE_LLM = CACHE_DIR / "deep_narrow_llm_cache.json"


class DeepNarrowEmbeddingCache(EmbeddingCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in (
            "embedding_cache.json",
            "arch_embedding_cache.json",
            "agent_embedding_cache.json",
            "frontier_embedding_cache.json",
            "meta_embedding_cache.json",
            "bestshot_embedding_cache.json",
            "optim_embedding_cache.json",
            "chain_embedding_cache.json",
            "goal_chain_embedding_cache.json",
            "deep_narrow_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                try:
                    with open(p) as f:
                        self._cache.update(json.load(f))
                except Exception:
                    pass
        self.cache_file = CACHE_FILE_EMB
        self._new: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


class DeepNarrowLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in (
            "llm_cache.json",
            "arch_llm_cache.json",
            "agent_llm_cache.json",
            "frontier_llm_cache.json",
            "meta_llm_cache.json",
            "bestshot_llm_cache.json",
            "optim_llm_cache.json",
            "chain_llm_cache.json",
            "goal_chain_llm_cache.json",
            "deep_narrow_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                try:
                    with open(p) as f:
                        data = json.load(f)
                    for k, v in data.items():
                        if v:
                            self._cache[k] = v
                except Exception:
                    pass
        self.cache_file = CACHE_FILE_LLM
        self._new: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new[key] = response

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class HopRecord:
    hop: int
    gap: str
    cue: str
    stopped: bool
    stop_reason: str
    new_found: int
    total_after: int


@dataclass
class DeepNarrowResult:
    segments: list[Segment]
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
DEEP_NARROW_PROMPT = """\
You are doing a deep iterative search over a conversation history to build a \
COMPREHENSIVE evidence pool for a question. The goal is NOT to find ONE \
answer — it is to surface EVERY conversation turn that is related, \
supporting, contradictory, or contextually important for answering the \
question well. A downstream answerer will read all retrieved turns, so \
coverage matters more than any single hit. Each hop you ask ONE highly-\
focused cue that probes a specific gap. Cues are embedded and matched via \
cosine similarity to conversation turns.

Question: {question}

Retrieved so far ({n_retrieved} segments, chronological):
{context}

ALREADY TRIED (do NOT repeat any of these):
{explored}

Think: what related content might still be missing? Consider:
- other mentions of the same event/person/entity in DIFFERENT words
- surrounding context (before/after the event — setup, consequences, \
followups)
- alternative framings the participants might have used
- contradictory or qualifying turns (hedges, corrections, updates)
- related sessions or side-topics that would inform the answer

Then generate ONE cue targeting the most informative gap.

Rules:
- The cue MUST target content DIFFERENT from what is already in the pool \
AND DIFFERENT from every prior cue.
- Be concrete — use specific vocabulary that would LITERALLY appear in a \
chat message (names, tools, symptoms, numbers, decisions, casual phrases).
- 1-2 sentences, casual first-person register.
- Only STOP if you genuinely cannot think of any new related angle (this \
should be RARE; keep hunting).

Format:
GAP: <one sentence: what specific piece of info is still missing?>
CUE: <a single search cue>

OR (only if truly saturated)

STOP: <reason>
"""


DEEP_NARROW_PROMPT_NO_STOP = """\
You are doing a deep iterative search over a conversation history to build a \
COMPREHENSIVE evidence pool for a question. The goal is NOT to find ONE \
answer — it is to surface EVERY conversation turn that is related, \
supporting, contradictory, or contextually important. Each hop you ask ONE \
highly-focused cue that probes a specific gap. Cues are embedded and matched \
via cosine similarity to conversation turns.

Question: {question}

Retrieved so far ({n_retrieved} segments, chronological):
{context}

ALREADY TRIED (do NOT repeat any of these):
{explored}

Think: what related content might still be missing? Consider:
- other mentions of the same event/person/entity in DIFFERENT words
- surrounding context (before/after — setup, consequences, followups)
- alternative framings the participants might have used
- contradictory or qualifying turns (hedges, corrections, updates)
- related sessions or side-topics that would inform the answer

Then generate ONE cue targeting the most informative gap. Always generate a \
cue; do not stop early.

Rules:
- The cue MUST target content DIFFERENT from what is already in the pool \
AND DIFFERENT from every prior cue.
- Be concrete — use specific vocabulary that would LITERALLY appear in a \
chat message.
- 1-2 sentences, casual first-person register.

Format:
GAP: <one sentence: what specific piece of info is still missing?>
CUE: <a single search cue>
"""


def _format_segments(
    segments: list[Segment],
    max_items: int = 14,
    max_chars: int = 220,
) -> str:
    """Chronological context, truncated."""
    if not segments:
        return "(none yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)
    # Show up to max_items but prefer evenly spaced if there are many
    if len(sorted_segs) > max_items:
        # Keep first + last + evenly spaced middle
        idxs = sorted(set(
            list(range(min(5, len(sorted_segs))))
            + list(range(len(sorted_segs) - 5, len(sorted_segs)))
            + [int(i * (len(sorted_segs) - 1) / (max_items - 1))
               for i in range(max_items)]
        ))
        sorted_segs = [sorted_segs[i] for i in idxs][:max_items]
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in sorted_segs
    )


def _parse_response(text: str) -> tuple[str, str, str]:
    """Parse LLM response: returns (gap, cue, stop_reason).

    Either (gap, cue, '') for a cue, or ('', '', reason) for STOP, or
    ('', '', '') if parse failed.
    """
    gap = ""
    cue = ""
    stop_reason = ""
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("STOP:"):
            stop_reason = line[5:].strip()
            return "", "", stop_reason or "self-assessed-complete"
        if upper.startswith("GAP:"):
            gap = line[4:].strip()
        elif upper.startswith("CUE:"):
            cue = line[4:].strip()
    return gap, cue, stop_reason


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class DeepNarrowBase:
    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        max_hops: int = 18,
        top_k_per_hop: int = 5,
        initial_k: int = 10,
        segment_cap: int = 80,
        allow_stop: bool = True,
    ) -> None:
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = DeepNarrowEmbeddingCache()
        self.llm_cache = DeepNarrowLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0
        self.max_hops = max_hops
        self.top_k_per_hop = top_k_per_hop
        self.initial_k = initial_k
        self.segment_cap = segment_cap
        self.allow_stop = allow_stop

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, embedding)
        self.embed_calls += 1
        return embedding

    def llm_call(self, prompt: str, model: str = MODEL) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=3000,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(model, prompt, text)
        self.llm_calls += 1
        return text

    def retrieve(self, question: str, conversation_id: str) -> DeepNarrowResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []
        explored: list[str] = []
        hop_records: list[HopRecord] = []

        # Initial retrieval: cosine top-initial_k on the question itself.
        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        stop_reason = ""
        prompt_tmpl = DEEP_NARROW_PROMPT if self.allow_stop else DEEP_NARROW_PROMPT_NO_STOP

        for hop in range(1, self.max_hops + 1):
            if len(all_segs) >= self.segment_cap:
                stop_reason = f"segment cap {self.segment_cap} reached"
                hop_records.append(HopRecord(
                    hop=hop, gap="", cue="",
                    stopped=True, stop_reason=stop_reason,
                    new_found=0, total_after=len(all_segs),
                ))
                break

            context = _format_segments(all_segs, max_items=14)
            explored_str = (
                "\n".join(f"- {c}" for c in explored) if explored else "(none yet)"
            )
            prompt = prompt_tmpl.format(
                question=question,
                n_retrieved=len(all_segs),
                context=context,
                explored=explored_str,
            )
            response = self.llm_call(prompt)
            gap, cue, stop_msg = _parse_response(response)

            if self.allow_stop and stop_msg:
                stop_reason = f"self-stop: {stop_msg}"
                hop_records.append(HopRecord(
                    hop=hop, gap=gap, cue="",
                    stopped=True, stop_reason=stop_reason,
                    new_found=0, total_after=len(all_segs),
                ))
                break

            if not cue:
                stop_reason = "no cue parsed"
                hop_records.append(HopRecord(
                    hop=hop, gap=gap, cue="",
                    stopped=True, stop_reason=stop_reason,
                    new_found=0, total_after=len(all_segs),
                ))
                break

            # De-duplicate cues
            if cue in explored:
                stop_reason = "duplicate cue generated"
                hop_records.append(HopRecord(
                    hop=hop, gap=gap, cue=cue,
                    stopped=True, stop_reason=stop_reason,
                    new_found=0, total_after=len(all_segs),
                ))
                break

            explored.append(cue)
            cue_emb = self.embed_text(cue)
            res = self.store.search(
                cue_emb, top_k=self.top_k_per_hop,
                conversation_id=conversation_id, exclude_indices=exclude,
            )
            new_found = 0
            for s in res.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)
                    new_found += 1

            hop_records.append(HopRecord(
                hop=hop, gap=gap, cue=cue,
                stopped=False, stop_reason="",
                new_found=new_found, total_after=len(all_segs),
            ))

            if new_found == 0:
                stop_reason = "saturation: 0 new segments"
                # Mark the last record as triggering the stop
                hop_records[-1].stopped = True
                hop_records[-1].stop_reason = stop_reason
                break
        else:
            stop_reason = "max_hops reached"

        return DeepNarrowResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": self.__class__.__name__,
                "max_hops": self.max_hops,
                "top_k_per_hop": self.top_k_per_hop,
                "allow_stop": self.allow_stop,
                "hops_used": len(hop_records),
                "hit_max_hops": stop_reason == "max_hops reached",
                "stop_reason": stop_reason,
                "hop_records": [h.__dict__ for h in hop_records],
                "total_segments": len(all_segs),
                "explored_cues": explored,
            },
        )


class DeepNarrowV1(DeepNarrowBase):
    """max_hops=18, top_k_per_hop=5, STOP allowed."""
    def __init__(self, store: SegmentStore, client: OpenAI | None = None) -> None:
        super().__init__(
            store, client,
            max_hops=18, top_k_per_hop=5,
            initial_k=10, segment_cap=80, allow_stop=True,
        )


class DeepNarrowWideProbe(DeepNarrowBase):
    """max_hops=18, top_k_per_hop=10, STOP allowed."""
    def __init__(self, store: SegmentStore, client: OpenAI | None = None) -> None:
        super().__init__(
            store, client,
            max_hops=18, top_k_per_hop=10,
            initial_k=10, segment_cap=120, allow_stop=True,
        )


class DeepNarrowNoStop(DeepNarrowBase):
    """max_hops=18, top_k_per_hop=5, STOP disabled — forced 18 hops."""
    def __init__(self, store: SegmentStore, client: OpenAI | None = None) -> None:
        super().__init__(
            store, client,
            max_hops=18, top_k_per_hop=5,
            initial_k=10, segment_cap=120, allow_stop=False,
        )
