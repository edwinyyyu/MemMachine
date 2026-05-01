"""Chain-aware retrieval architectures for sequential_chain and evolving_terminology.

Targets two hardest category groups:
  - sequential_chain: clue A uses different vocabulary than the question, A
    points to B which uses yet different vocabulary. Each retrieval result must
    inform the next.
  - evolving_terminology: same concept is called different names ("customer
    portal" -> "Project Phoenix" -> "PHX" -> "the bird"). Cues must discover
    the terminology map.

Approaches implemented (with reasonable ablations):
  ITER    - Iterative chained retrieval (question->find->next-cue->...)
  TERM    - Terminology discovery (extract alternative names, search each)
  COT     - Chain-of-thought cue generation (hybrid of both)
  EMB     - Embedding-space exploration (no LLM, residual from centroid)
  HYBRID  - v15 initial + terminology expansion

All are evaluated with a FAIR BUDGET:
  - K=20 and K=50: each architecture produces a pool, and is truncated/
    backfilled from raw cosine to exactly K segments before recall is scored.

Usage:
    uv run python chain_retrieval.py [--arch NAME] [--benchmark all|puzzle|advanced]
    uv run python chain_retrieval.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_FILE_EMB = CACHE_DIR / "chain_embedding_cache.json"
CACHE_FILE_LLM = CACHE_DIR / "chain_llm_cache.json"
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Caches (read from all prior caches, write to chain_* files)
# ---------------------------------------------------------------------------
class ChainEmbeddingCache(EmbeddingCache):
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
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
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


class ChainLLMCache(LLMCache):
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
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
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
# Base class
# ---------------------------------------------------------------------------
@dataclass
class ChainResult:
    segments: list[Segment]
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


class ChainBase:
    def __init__(self, store: SegmentStore, client: OpenAI | None = None) -> None:
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = ChainEmbeddingCache()
        self.llm_cache = ChainLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

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
            max_completion_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(model, prompt, text)
        self.llm_calls += 1
        return text

    def retrieve(self, question: str, conversation_id: str) -> ChainResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_segments(
    segments: list[Segment],
    max_items: int = 14,
    max_chars: int = 260,
) -> str:
    """Format segments chronologically for LLM context."""
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in sorted_segs
    )


def _format_segments_latest(
    segments: list[Segment], n: int = 6, max_chars: int = 240
) -> str:
    """Show the last-added segments (useful for chain-following)."""
    tail = segments[-n:]
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in tail
    )


def _parse_cues(text: str, key: str = "CUE:") -> list[str]:
    out: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith(key.upper()):
            val = line[len(key) :].strip()
            if val:
                out.append(val)
    return out


def _parse_names(text: str) -> list[str]:
    """Parse 'NAME:' lines. Strip quotes/brackets."""
    names: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("NAME:"):
            val = line[5:].strip().strip("\"'`").strip()
            # Strip enclosing brackets/parens
            while val and val[0] in "[(<":
                val = val[1:].strip()
            while val and val[-1] in "])>":
                val = val[:-1].strip()
            val = val.strip("\"'`").strip()
            if val:
                names.append(val)
    return names


# ===========================================================================
# 1. ITER — Iterative chained retrieval (sequential_chain)
# ===========================================================================
ITER_PROMPT = """\
You are performing iterative retrieval over a conversation history. Each search \
cue you generate will be embedded and matched against conversation turns via \
cosine similarity.

This question may involve a CHAIN OF CLUES where each clue uses different \
vocabulary than the question. To find the full chain, you must read what has \
been retrieved and propose a NEXT search that picks up the next link.

Question: {question}

MOST RECENTLY RETRIEVED EXCERPTS (chronological tail):
{latest}

ALL RETRIEVED SO FAR ({num_segs} segments, chronological):
{all_segs}

ALREADY SEARCHED FOR (do NOT repeat these):
{explored}

Think step by step (internally):
- What specific nouns, names, tools, actions, symptoms, or decisions appear in \
the retrieved excerpts that look like the CURRENT LINK in a chain?
- What NEXT LINK would be discussed in nearby or later turns? (e.g. "plant \
wilting" -> next search "tap water quality / mineral burn")
- Use concrete vocabulary that would literally appear in the target turns.

If there is nothing else worth searching for (the retrieval looks complete \
and coherent), respond with DONE.

Format (one cue only, no bullet lists, no questions):
REASON: <what current link is in the retrieved content, what next link to find>
CUE: <1-2 sentences of plausible conversation content targeting the next link>
(or)
REASON: <why retrieval looks complete>
DONE
Nothing else."""


class IterativeChain(ChainBase):
    """ITER: question -> retrieve -> LLM picks next link -> retrieve -> ...

    Each iteration uses fresh retrieval informed by the previous results.
    No branching; purely sequential.
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        max_iterations: int = 4,
        per_iter_k: int = 5,
        initial_k: int = 10,
    ) -> None:
        super().__init__(store, client)
        self.max_iterations = max_iterations
        self.per_iter_k = per_iter_k
        self.initial_k = initial_k

    def retrieve(self, question: str, conversation_id: str) -> ChainResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []
        explored: list[str] = []
        round_log: list[dict] = []

        # Round 0: question embedding
        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        for it in range(self.max_iterations):
            prompt = ITER_PROMPT.format(
                question=question,
                latest=_format_segments_latest(all_segs, n=6),
                all_segs=_format_segments(all_segs, max_items=14),
                num_segs=len(all_segs),
                explored=(
                    "\n".join(f"- {c}" for c in explored) if explored else "(none yet)"
                ),
            )
            response = self.llm_call(prompt)

            reason = ""
            cue = ""
            done = False
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("REASON:"):
                    reason = line[7:].strip()
                elif line.upper().startswith("CUE:"):
                    cue = line[4:].strip()
                elif line.upper() == "DONE":
                    done = True

            round_log.append(
                {
                    "iter": it,
                    "reason": reason,
                    "cue": cue,
                    "done": done,
                }
            )
            if done or not cue:
                break
            explored.append(cue)
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_iter_k,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return ChainResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "iterative_chain",
                "rounds": round_log,
                "total_segments": len(all_segs),
            },
        )


# ===========================================================================
# 2. TERM — Terminology-aware retrieval (evolving_terminology)
# ===========================================================================
TERM_PROMPT = """\
You are searching a conversation for all references to a concept/entity. The \
conversation may have used MANY DIFFERENT NAMES for the same thing (e.g. \
"customer portal" -> "Project Phoenix" -> "PHX" -> "the bird" -> "v2").

Question: {question}

RETRIEVED SO FAR ({num_segs} segments, chronological):
{all_segs}

Your task: list EVERY alternative name, abbreviation, nickname, ticket prefix, \
short reference, or pronoun-like phrase that the conversation uses for the \
thing being asked about. Pull names directly out of the retrieved excerpts \
when possible, and ALSO guess plausible additional names that might appear \
elsewhere in the conversation (e.g. typical codenames, "the <noun>" phrases, \
abbreviations of the official name, ticket prefixes).

If the question is about a single person, project, bug, event, or decision, \
list all its aliases. If the question is about how many names were used, \
list every name you can find.

Do NOT list full sentences. Each NAME should be a short noun phrase of 1-5 \
words that would literally appear inline in a message.

Format (one per line, 4-10 entries):
NAME: <alias 1>
NAME: <alias 2>
...
Nothing else."""


class TerminologyDiscovery(ChainBase):
    """TERM: initial retrieve -> LLM extracts alternative names -> retrieve each.

    Budget is split: initial gets roughly budget/3, rest is divided across
    discovered alternative names.
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        initial_k: int = 10,
        max_names: int = 6,
        per_name_k: int = 5,
    ) -> None:
        super().__init__(store, client)
        self.initial_k = initial_k
        self.max_names = max_names
        self.per_name_k = per_name_k

    def retrieve(self, question: str, conversation_id: str) -> ChainResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []

        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        prompt = TERM_PROMPT.format(
            question=question,
            all_segs=_format_segments(all_segs, max_items=14),
            num_segs=len(all_segs),
        )
        response = self.llm_call(prompt)
        names = _parse_names(response)
        names = names[: self.max_names]

        for name in names:
            cue_emb = self.embed_text(name)
            result = self.store.search(
                cue_emb,
                top_k=self.per_name_k,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return ChainResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "terminology_discovery",
                "alt_names": names,
                "total_segments": len(all_segs),
            },
        )


# ===========================================================================
# 3. COT — Chain-of-thought cue generation (unified: chain + terminology)
# ===========================================================================
COT_PROMPT = """\
You are performing semantic retrieval over a conversation history. Cues will \
be embedded and matched via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({num_segs} segments, chronological):
{all_segs}

ALREADY SEARCHED FOR (do NOT repeat):
{explored}

Think step by step:
1. What specific terminology appears in the retrieved segments (names, tools, \
symptoms, decisions, tickets, numbers)?
2. What RELATED terminology might be used elsewhere? (aliases, codenames, \
abbreviations, informal references like "the bird", "that thing", ...)
3. If this is a CHAIN (A -> B -> C where each link has different vocabulary), \
what is the NEXT link to search for?
4. If this topic has ALTERNATIVE NAMES, what are they? Include every alias \
you can justify from the retrieved text or reasonable guesses.

Then generate {num_cues} search cues that EXTEND the retrieval in the most \
promising directions. A cue may be:
  - a short alias/name phrase (1-5 words) that might appear inline
  - a 1-2 sentence plausible conversation snippet targeting the next link

Prefer DIVERSE cues (cover multiple aliases and/or multiple chain links). \
Do not rephrase the question.

Format:
REASON: <1-2 sentences: what's current link or what aliases you identified>
CUE: <text>
CUE: <text>
(up to {num_cues} cues)
Nothing else."""


class ChainOfThoughtCue(ChainBase):
    """COT: unified approach using step-by-step reasoning prompt.

    Runs a single initial retrieve, then one LLM call that can emit a mix of
    aliases (short names) and chain-link cues (longer snippets). Retrieves for
    each cue with a modest top-k.
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        initial_k: int = 10,
        num_cues: int = 5,
        per_cue_k: int = 4,
        rounds: int = 2,
    ) -> None:
        super().__init__(store, client)
        self.initial_k = initial_k
        self.num_cues = num_cues
        self.per_cue_k = per_cue_k
        self.rounds = rounds

    def retrieve(self, question: str, conversation_id: str) -> ChainResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []
        explored: list[str] = []
        round_log: list[dict] = []

        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        for round_i in range(self.rounds):
            prompt = COT_PROMPT.format(
                question=question,
                all_segs=_format_segments(all_segs, max_items=14),
                num_segs=len(all_segs),
                explored=(
                    "\n".join(f"- {c}" for c in explored) if explored else "(none yet)"
                ),
                num_cues=self.num_cues,
            )
            response = self.llm_call(prompt)
            reason = ""
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("REASON:"):
                    reason = line[7:].strip()
                    break
            cues = _parse_cues(response, "CUE:")[: self.num_cues]
            round_log.append({"round": round_i, "reason": reason, "cues": cues})

            if not cues:
                break
            for cue in cues:
                if cue in explored:
                    continue
                explored.append(cue)
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb,
                    top_k=self.per_cue_k,
                    conversation_id=conversation_id,
                    exclude_indices=exclude,
                )
                for s in result.segments:
                    if s.index not in exclude:
                        all_segs.append(s)
                        exclude.add(s.index)

        return ChainResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "chain_of_thought_cue",
                "rounds": round_log,
                "total_segments": len(all_segs),
            },
        )


# ===========================================================================
# 3b. IterativeChainNoStop — same as ITER but never allows DONE early exit
# ===========================================================================
class IterativeChainNoStop(IterativeChain):
    """ITER variant that forces the LLM to generate a cue every round.

    Original ITER stopped after 1 round on many questions because the LLM
    declared retrieval "complete". This variant ignores DONE and always
    demands a new cue for max_iterations rounds.
    """

    def retrieve(self, question: str, conversation_id: str) -> ChainResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []
        explored: list[str] = []
        round_log: list[dict] = []

        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        prompt_template = (
            "You are performing iterative retrieval over a conversation history. "
            "Each search cue you generate will be embedded and matched against "
            "conversation turns via cosine similarity.\n\n"
            "This question may involve a CHAIN OF CLUES where each clue uses "
            "different vocabulary than the question, OR the SAME CONCEPT MAY "
            "APPEAR UNDER DIFFERENT NAMES (aliases, nicknames, abbreviations). "
            "To find the full answer, you must read what has been retrieved and "
            "propose a NEW search that picks up a link or alias you haven't "
            "explored yet.\n\n"
            "Question: {question}\n\n"
            "ALL RETRIEVED SO FAR ({num_segs} segments):\n"
            "{all_segs}\n\n"
            "ALREADY SEARCHED FOR (do NOT repeat):\n"
            "{explored}\n\n"
            "Always generate ONE new cue. It may be:\n"
            "  - a short alias/name (1-5 words) the conversation might use\n"
            "  - a 1-2 sentence plausible conversation snippet for the next link\n"
            "Use specific vocabulary that would literally appear in the target "
            "turns. Do not rephrase the question. Do not output DONE.\n\n"
            "Format:\n"
            "REASON: <what link or alias you are targeting>\n"
            "CUE: <text>\n"
            "Nothing else."
        )

        for it in range(self.max_iterations):
            prompt = prompt_template.format(
                question=question,
                num_segs=len(all_segs),
                all_segs=_format_segments(all_segs, max_items=14),
                explored=(
                    "\n".join(f"- {c}" for c in explored) if explored else "(none yet)"
                ),
            )
            response = self.llm_call(prompt)
            reason = ""
            cue = ""
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("REASON:"):
                    reason = line[7:].strip()
                elif line.upper().startswith("CUE:"):
                    cue = line[4:].strip()

            round_log.append({"iter": it, "reason": reason, "cue": cue})
            if not cue:
                break
            explored.append(cue)
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_iter_k,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return ChainResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "iterative_chain_nostop",
                "rounds": round_log,
                "total_segments": len(all_segs),
            },
        )


# ===========================================================================
# 4. EMB — Embedding-space exploration (no LLM)
# ===========================================================================
class EmbeddingExplore(ChainBase):
    """EMB: take centroid of top-k, compute residuals (subtract centroid), and
    retrieve AWAY from centroid in the residual direction.

    Tests whether sequential chain can be solved purely by embedding-space
    diversity, no LLM reasoning required.

    Procedure:
      1. Initial: top_k_initial from question (normalized).
      2. Compute centroid c of retrieved embeddings (normalized mean).
      3. Residual directions = normalized embeddings minus projection onto c.
      4. Pick 3 "anchor" segments (far from centroid but still relevant) and
         retrieve in their direction, excluding already-seen.
      5. Also do one "anti-centroid" sweep: retrieve using (q - alpha * c)
         normalized, which biases away from dense question neighborhood.
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        initial_k: int = 10,
        num_anchors: int = 3,
        per_anchor_k: int = 4,
        alpha: float = 0.5,
    ) -> None:
        super().__init__(store, client)
        self.initial_k = initial_k
        self.num_anchors = num_anchors
        self.per_anchor_k = per_anchor_k
        self.alpha = alpha

    def retrieve(self, question: str, conversation_id: str) -> ChainResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []

        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=self.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        # Centroid over retrieved segment embeddings (using normalized store embs)
        retrieved_embs = np.stack(
            [self.store.normalized_embeddings[s.index] for s in all_segs]
        )
        centroid = retrieved_embs.mean(axis=0)
        c_norm = centroid / max(np.linalg.norm(centroid), 1e-10)

        # Anchor residuals: segments that are least-aligned with centroid
        alignments = retrieved_embs @ c_norm  # similarity with centroid
        # Anchors = bottom-N alignment (still retrieved, but most "peripheral")
        anchor_order = np.argsort(alignments)[: self.num_anchors]
        anchor_segs = [all_segs[i] for i in anchor_order]

        anchors_used = []
        for a in anchor_segs:
            a_emb = self.store.normalized_embeddings[a.index]
            # Residual direction: anchor minus centroid component
            resid = a_emb - (a_emb @ c_norm) * c_norm
            resid_norm = resid / max(np.linalg.norm(resid), 1e-10)
            # Blend anchor + residual to stay on-topic but push outward
            probe = 0.6 * a_emb + 0.4 * resid_norm
            probe = probe / max(np.linalg.norm(probe), 1e-10)
            result = self.store.search(
                probe,
                top_k=self.per_anchor_k,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)
            anchors_used.append(
                {
                    "turn_id": a.turn_id,
                    "align": float(alignments[anchor_order[anchors_used.__len__()]]),
                }
            )

        # Anti-centroid sweep: q - alpha * centroid
        q_norm = q_emb / max(np.linalg.norm(q_emb), 1e-10)
        anti = q_norm - self.alpha * c_norm
        anti = anti / max(np.linalg.norm(anti), 1e-10)
        result = self.store.search(
            anti,
            top_k=self.per_anchor_k,
            conversation_id=conversation_id,
            exclude_indices=exclude,
        )
        for s in result.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        return ChainResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "embedding_explore",
                "anchors": [a.turn_id for a in anchor_segs],
                "total_segments": len(all_segs),
            },
        )


# ===========================================================================
# 5. HYBRID — v15 initial + terminology expansion
# ===========================================================================
V15_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


class HybridV15Term(ChainBase):
    """HYBRID: v15 retrieval (top-10 + 2 cues top-10) then terminology expansion.

    Combines v15's strong baseline (good for generic questions) with
    terminology discovery (good for evolving_terminology).
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        max_names: int = 4,
        per_name_k: int = 3,
    ) -> None:
        super().__init__(store, client)
        self.max_names = max_names
        self.per_name_k = per_name_k

    def retrieve(self, question: str, conversation_id: str) -> ChainResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []

        # --- v15-style retrieval ---
        q_emb = self.embed_text(question)
        r0 = self.store.search(q_emb, top_k=10, conversation_id=conversation_id)
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        context = "\n".join(
            f"[Turn {s.turn_id}, {s.role}]: {s.text[:250]}"
            for s in sorted(all_segs, key=lambda s: s.turn_id)[:12]
        )
        v15_resp = self.llm_call(V15_PROMPT.format(question=question, context=context))
        cues = _parse_cues(v15_resp)[:2]
        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        # --- terminology expansion ---
        term_prompt = TERM_PROMPT.format(
            question=question,
            all_segs=_format_segments(all_segs, max_items=16),
            num_segs=len(all_segs),
        )
        term_resp = self.llm_call(term_prompt)
        names = _parse_names(term_resp)[: self.max_names]
        for name in names:
            name_emb = self.embed_text(name)
            result = self.store.search(
                name_emb,
                top_k=self.per_name_k,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return ChainResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "hybrid_v15_term",
                "v15_cues": cues,
                "alt_names": names,
                "total_segments": len(all_segs),
            },
        )


# ===========================================================================
# 6. HYBRID_FULL — v15 + terminology + 1 iterative chain step
# ===========================================================================
class HybridFull(ChainBase):
    """HYBRID_FULL: v15 (q + 2 cues) -> terminology names -> 1 chain step.

    Tests whether stacking all three strategies in a budget-aware way beats
    v15 or any single approach. Budget allocation (total ~35 pool slots):
      - v15:   10 (initial) + 10 + 10 = 30 (but capped via exclude)
      - names: 4 names x 2 = 8
      - chain: 1 cue x 5 = 5
    Duplicates are excluded so actual pool ~35.
    """

    def retrieve(self, question: str, conversation_id: str) -> ChainResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []

        # --- v15 step ---
        q_emb = self.embed_text(question)
        r0 = self.store.search(q_emb, top_k=10, conversation_id=conversation_id)
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        context = "\n".join(
            f"[Turn {s.turn_id}, {s.role}]: {s.text[:250]}"
            for s in sorted(all_segs, key=lambda s: s.turn_id)[:12]
        )
        v15_resp = self.llm_call(V15_PROMPT.format(question=question, context=context))
        v15_cues = _parse_cues(v15_resp)[:2]
        for cue in v15_cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        # --- terminology step ---
        term_prompt = TERM_PROMPT.format(
            question=question,
            all_segs=_format_segments(all_segs, max_items=16),
            num_segs=len(all_segs),
        )
        term_resp = self.llm_call(term_prompt)
        names = _parse_names(term_resp)[:4]
        for name in names:
            name_emb = self.embed_text(name)
            result = self.store.search(
                name_emb,
                top_k=2,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        # --- chain step: one more iterative reflection ---
        chain_prompt = (
            "You have retrieved segments for a question. Based on what's been "
            "found, identify ONE specific NEXT LINK in the causal/temporal "
            "chain that has NOT yet been retrieved. Generate one cue.\n\n"
            f"Question: {question}\n\n"
            f"RETRIEVED ({len(all_segs)} segs):\n"
            f"{_format_segments(all_segs, max_items=16)}\n\n"
            "Previous queries (do not repeat):\n"
            + "\n".join(f"- {c}" for c in v15_cues + names)
            + "\n\nFormat:\nREASON: <what next link>\nCUE: <1-2 sentence snippet>"
            "\nNothing else."
        )
        chain_resp = self.llm_call(chain_prompt)
        chain_cue = ""
        for line in chain_resp.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("CUE:"):
                chain_cue = line[4:].strip()
                break
        if chain_cue:
            cue_emb = self.embed_text(chain_cue)
            result = self.store.search(
                cue_emb,
                top_k=5,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return ChainResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "hybrid_full",
                "v15_cues": v15_cues,
                "alt_names": names,
                "chain_cue": chain_cue,
                "total_segments": len(all_segs),
            },
        )


# ===========================================================================
# V15 reference baseline (for direct comparison, same caches)
# ===========================================================================
class V15Reference(ChainBase):
    """Reference v15: question top-10, 1 LLM call, 2 cues top-10 each.

    Uses the SAME prompt/format as the v15 used in prior experiments so cache
    hits are possible.
    """

    def retrieve(self, question: str, conversation_id: str) -> ChainResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []

        q_emb = self.embed_text(question)
        r0 = self.store.search(q_emb, top_k=10, conversation_id=conversation_id)
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        context = "\n".join(
            f"[Turn {s.turn_id}, {s.role}]: {s.text[:250]}"
            for s in sorted(all_segs, key=lambda s: s.turn_id)[:12]
        )
        resp = self.llm_call(V15_PROMPT.format(question=question, context=context))
        cues = _parse_cues(resp)[:2]
        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return ChainResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "v15_reference",
                "cues": cues,
                "total_segments": len(all_segs),
            },
        )


# ===========================================================================
# Evaluation with FAIR BUDGET (K=20 and K=50)
# ===========================================================================
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    arch: ChainBase,
    question: dict,
    verbose: bool = False,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedup preserving order
    seen: set[int] = set()
    deduped: list[Segment] = []
    for s in result.segments:
        if s.index not in seen:
            deduped.append(s)
            seen.add(s.index)
    arch_segments = deduped

    # Baseline cosine top-max(BUDGETS)
    q_emb = arch.embed_text(q_text)
    max_b = max(BUDGETS)
    baseline = arch.store.search(q_emb, top_k=max_b, conversation_id=conv_id)

    # FAIR backfill: extend arch pool with baseline segments (not already in pool)
    arch_idx = {s.index for s in arch_segments}
    backfilled = list(arch_segments) + [
        s for s in baseline.segments if s.index not in arch_idx
    ]

    baseline_recalls = {}
    arch_recalls = {}
    for K in BUDGETS:
        b_ids = {s.turn_id for s in baseline.segments[:K]}
        a_ids = {s.turn_id for s in backfilled[:K]}
        baseline_recalls[f"r@{K}"] = compute_recall(b_ids, source_ids)
        arch_recalls[f"r@{K}"] = compute_recall(a_ids, source_ids)

    out = {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question["question_index"],
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "total_retrieved": len(arch_segments),
        "baseline_recalls": baseline_recalls,
        "arch_recalls": arch_recalls,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "metadata": result.metadata,
    }
    if verbose:
        print(
            f"    arch_pool={len(arch_segments)} "
            f"r@20: base={baseline_recalls['r@20']:.3f} "
            f"arch={arch_recalls['r@20']:.3f}  "
            f"r@50: base={baseline_recalls['r@50']:.3f} "
            f"arch={arch_recalls['r@50']:.3f}  "
            f"llm={arch.llm_calls} emb={arch.embed_calls}"
        )
    return out


def summarize(
    results: list[dict], arch_name: str, benchmark: str, category: str | None = None
) -> dict:
    if category is not None:
        results = [r for r in results if r["category"] == category]
    n = len(results)
    if n == 0:
        return {}
    summary = {
        "arch": arch_name,
        "benchmark": benchmark,
        "category": category or "ALL",
        "n": n,
    }
    for K in BUDGETS:
        b = sum(r["baseline_recalls"][f"r@{K}"] for r in results) / n
        a = sum(r["arch_recalls"][f"r@{K}"] for r in results) / n
        wins = sum(
            1
            for r in results
            if r["arch_recalls"][f"r@{K}"] > r["baseline_recalls"][f"r@{K}"] + 0.001
        )
        losses = sum(
            1
            for r in results
            if r["baseline_recalls"][f"r@{K}"] > r["arch_recalls"][f"r@{K}"] + 0.001
        )
        ties = n - wins - losses
        summary[f"baseline_r@{K}"] = round(b, 4)
        summary[f"arch_r@{K}"] = round(a, 4)
        summary[f"delta_r@{K}"] = round(a - b, 4)
        summary[f"WTL_r@{K}"] = f"{wins}/{ties}/{losses}"
    summary["avg_total_retrieved"] = round(
        sum(r["total_retrieved"] for r in results) / n, 1
    )
    summary["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
    summary["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 1)
    summary["avg_time_s"] = round(sum(r["time_s"] for r in results) / n, 2)
    return summary


def print_summary(summary: dict) -> None:
    if not summary:
        return
    print(
        f"  {summary['arch']:<22s} {summary['benchmark']:<12s} "
        f"{summary['category']:<22s} n={summary['n']}"
    )
    for K in BUDGETS:
        b = summary[f"baseline_r@{K}"]
        a = summary[f"arch_r@{K}"]
        d = summary[f"delta_r@{K}"]
        print(
            f"    r@{K:<2d}: base={b:.3f} arch={a:.3f} "
            f"delta={d:+.3f} WTL={summary[f'WTL_r@{K}']}"
        )
    print(
        f"    #pool={summary['avg_total_retrieved']:.1f} "
        f"llm={summary['avg_llm_calls']:.1f} "
        f"emb={summary['avg_embed_calls']:.1f} "
        f"time={summary['avg_time_s']:.1f}s"
    )


# ===========================================================================
# Architecture registry
# ===========================================================================
ARCHS = {
    "v15_reference": V15Reference,
    "iterative_chain": IterativeChain,
    "iterative_chain_nostop": IterativeChainNoStop,
    "terminology_discovery": TerminologyDiscovery,
    "chain_of_thought": ChainOfThoughtCue,
    "embedding_explore": EmbeddingExplore,
    "hybrid_v15_term": HybridV15Term,
    "hybrid_full": HybridFull,
}

BENCHMARKS = {
    "puzzle": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "category": "sequential_chain",
        "tag": "puzzle_seqchain",
    },
    "advanced": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "category": "evolving_terminology",
        "tag": "advanced_evolterm",
    },
}


# ===========================================================================
# Main runner
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Run a single architecture (default: all)",
    )
    parser.add_argument(
        "--benchmark", type=str, default="all", choices=["all", "puzzle", "advanced"]
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Architectures:")
        for n in ARCHS:
            print(f"  {n}")
        print("Benchmarks:")
        for n in BENCHMARKS:
            print(f"  {n}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bench_names = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]
    arch_names = list(ARCHS) if args.arch is None else [args.arch]

    all_summaries: list[dict] = []

    for bench in bench_names:
        cfg = BENCHMARKS[bench]
        store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
        with open(DATA_DIR / cfg["questions"]) as f:
            all_qs = json.load(f)
        questions = [q for q in all_qs if q["category"] == cfg["category"]]
        print(
            f"\n=== {bench.upper()} | {cfg['category']} | "
            f"{len(questions)} questions ==="
        )

        for arch_name in arch_names:
            if arch_name not in ARCHS:
                print(f"Unknown arch: {arch_name}")
                continue

            results_file = RESULTS_DIR / f"chain_{arch_name}_{cfg['tag']}.json"
            if results_file.exists() and not args.force:
                with open(results_file) as f:
                    results = json.load(f)
                summary = summarize(results, arch_name, bench, cfg["category"])
                all_summaries.append(summary)
                print(f"\n[cached] {arch_name}")
                print_summary(summary)
                continue

            arch = ARCHS[arch_name](store)
            print(f"\n-- {arch_name} --")
            results: list[dict] = []
            for i, q in enumerate(questions):
                q_short = q["question"][:55]
                print(f"  [{i + 1}/{len(questions)}] {q_short}...", flush=True)
                try:
                    out = evaluate_one(arch, q, verbose=args.verbose)
                    results.append(out)
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    import traceback

                    traceback.print_exc()
                sys.stdout.flush()

            arch.save_caches()
            summary = summarize(results, arch_name, bench, cfg["category"])
            all_summaries.append(summary)
            print_summary(summary)

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Saved -> {results_file}")

    # ---- GRAND SUMMARY TABLE ----
    print(f"\n{'=' * 100}")
    print("CHAIN RETRIEVAL — GRAND SUMMARY (FAIR BUDGET)")
    print(f"{'=' * 100}")
    print(
        f"{'Arch':<22s} {'Benchmark':<11s} {'Category':<22s} "
        f"{'n':>3s} "
        f"{'B@20':>6s} {'A@20':>6s} {'D@20':>7s} "
        f"{'B@50':>6s} {'A@50':>6s} {'D@50':>7s} "
        f"{'LLM':>5s} {'Emb':>5s} {'#Pool':>6s}"
    )
    print("-" * 100)
    for s in all_summaries:
        if not s:
            continue
        print(
            f"{s['arch']:<22s} {s['benchmark']:<11s} {s['category']:<22s} "
            f"{s['n']:>3d} "
            f"{s['baseline_r@20']:>6.3f} {s['arch_r@20']:>6.3f} "
            f"{s['delta_r@20']:>+7.3f} "
            f"{s['baseline_r@50']:>6.3f} {s['arch_r@50']:>6.3f} "
            f"{s['delta_r@50']:>+7.3f} "
            f"{s['avg_llm_calls']:>5.1f} "
            f"{s['avg_embed_calls']:>5.1f} "
            f"{s['avg_total_retrieved']:>6.1f}"
        )
    print("-" * 100)

    summary_file = RESULTS_DIR / "chain_all_summaries.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summaries -> {summary_file}")


if __name__ == "__main__":
    main()
