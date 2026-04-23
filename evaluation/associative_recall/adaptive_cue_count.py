"""Adaptive cue count per query difficulty.

Hypothesis: queries whose top-1 cosine match is weak (topic_drift) benefit
from more cues; queries with strong initial matches stay at v2f defaults.

Signal: cosine(raw query, top-1 segment) = c1
  - c1 >= 0.5: EASY (2 cues)
  - 0.3 <= c1 < 0.5: MEDIUM (4 cues)
  - c1 < 0.3: HARD (7 cues)

Variants:
  - adaptive_cue_3_tier (EASY=2, MEDIUM=4, HARD=7)
  - adaptive_cue_binary (c1>=0.4 -> 2 cues, else 6 cues)
  - always_6cues (control: always 6, no adaptation)
  - meta_v2f (reference, always 2)
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

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
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"


# ---------------------------------------------------------------------------
# Dedicated caches for this experiment
# ---------------------------------------------------------------------------
class AdaptiveCueEmbeddingCache(EmbeddingCache):
    """Reads all prior embedding caches, writes to adaptive_cue file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in (
            "embedding_cache.json",
            "arch_embedding_cache.json",
            "agent_embedding_cache.json",
            "frontier_embedding_cache.json",
            "meta_embedding_cache.json",
            "optim_embedding_cache.json",
            "synth_test_embedding_cache.json",
            "bestshot_embedding_cache.json",
            "adaptive_cue_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "adaptive_cue_embedding_cache.json"
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


class AdaptiveCueLLMCache(LLMCache):
    """Reads all prior LLM caches, writes to adaptive_cue file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in (
            "llm_cache.json",
            "arch_llm_cache.json",
            "agent_llm_cache.json",
            "tree_llm_cache.json",
            "frontier_llm_cache.json",
            "meta_llm_cache.json",
            "optim_llm_cache.json",
            "synth_test_llm_cache.json",
            "bestshot_llm_cache.json",
            "adaptive_cue_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "adaptive_cue_llm_cache.json"
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


# ---------------------------------------------------------------------------
# Parameterized v2f prompt with adaptive num_cues
# ---------------------------------------------------------------------------
ADAPTIVE_V2F_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate {num_cues} search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns. Each cue \
should target a DIFFERENT aspect or phrasing so together they cover the \
question more completely.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
{cue_format_lines}
Nothing else."""


def _format_segments(
    segments: list[Segment],
    max_items: int = 12,
    max_chars: int = 250,
) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in sorted_segs
    )


def _parse_cues(response: str) -> list[str]:
    cues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


@dataclass
class AdaptiveCueResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base architecture
# ---------------------------------------------------------------------------
class AdaptiveCueBase:
    """Parameterized v2f with adaptive num_cues based on top-1 cosine."""

    # Subclasses override difficulty -> num_cues mapping.
    name = "adaptive_base"

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
    ):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = AdaptiveCueEmbeddingCache()
        self.llm_cache = AdaptiveCueLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

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
            max_completion_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        try:
            usage = response.usage
            if usage is not None:
                self.total_completion_tokens += int(
                    usage.completion_tokens or 0
                )
                self.total_prompt_tokens += int(usage.prompt_tokens or 0)
        except Exception:
            pass
        self.llm_cache.put(model, prompt, text)
        self.llm_calls += 1
        return text

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    # ------------------------------------------------------------------
    # Difficulty-to-cue-count logic — overridden by subclasses
    # ------------------------------------------------------------------
    def pick_num_cues(self, c1: float) -> tuple[int, str]:
        """Return (num_cues, difficulty_label) based on top-1 cosine."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main retrieval loop
    # ------------------------------------------------------------------
    def retrieve(
        self, question: str, conversation_id: str
    ) -> AdaptiveCueResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        # Compute difficulty signal from top-1 cosine.
        top1_cosine = float(hop0.scores[0]) if hop0.scores else 0.0
        num_cues, difficulty = self.pick_num_cues(top1_cosine)

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(all_segments)
        )

        cue_format_lines = "\n".join(["CUE: <text>"] * num_cues)
        prompt = ADAPTIVE_V2F_PROMPT.format(
            question=question,
            context_section=context_section,
            num_cues=num_cues,
            cue_format_lines=cue_format_lines,
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:num_cues]

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

        return AdaptiveCueResult(
            segments=all_segments,
            metadata={
                "name": self.name,
                "output": output,
                "cues": cues,
                "top1_cosine": top1_cosine,
                "difficulty": difficulty,
                "num_cues_requested": num_cues,
                "num_cues_parsed": len(cues),
            },
        )


class AdaptiveCue3Tier(AdaptiveCueBase):
    """3-tier: EASY(c1>=0.5)=2, MEDIUM(0.3<=c1<0.5)=4, HARD(c1<0.3)=7."""

    name = "adaptive_cue_3_tier"

    def pick_num_cues(self, c1: float) -> tuple[int, str]:
        if c1 >= 0.5:
            return 2, "EASY"
        if c1 >= 0.3:
            return 4, "MEDIUM"
        return 7, "HARD"


class AdaptiveCueBinary(AdaptiveCueBase):
    """Binary: c1>=0.4 -> 2 cues, else 6 cues."""

    name = "adaptive_cue_binary"

    def pick_num_cues(self, c1: float) -> tuple[int, str]:
        if c1 >= 0.4:
            return 2, "EASY"
        return 6, "HARD"


class Always6Cues(AdaptiveCueBase):
    """Control: always generate 6 cues, no adaptation."""

    name = "always_6cues"

    def pick_num_cues(self, c1: float) -> tuple[int, str]:
        return 6, "FLAT"


class MetaV2fReference(AdaptiveCueBase):
    """Reference: always generate 2 cues (mirrors v2f default)."""

    name = "meta_v2f_ref"

    def pick_num_cues(self, c1: float) -> tuple[int, str]:
        return 2, "V2F"


ARCHITECTURES = {
    "meta_v2f_ref": MetaV2fReference,
    "adaptive_cue_3_tier": AdaptiveCue3Tier,
    "adaptive_cue_binary": AdaptiveCueBinary,
    "always_6cues": Always6Cues,
}
