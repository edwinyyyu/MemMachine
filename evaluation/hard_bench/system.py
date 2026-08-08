"""Unified retrieval system: EventMemory substrate + multi-channel retrieval.

Channels:
- em_cosine: native EventMemory.query (semantic kNN)
- em_pattern_v15: V15 tier emission + cosine on tier probes + soft judge filter
- em_temporal: STUB — TemporalRetriever wrapper (not yet wired)
- em_entity: STUB — R25 prose-fact + DSU retrieval (not yet wired)

RRF ensemble combines top-K from each channel at fixed K=5 context budget.

Public API:
    system = await build_system(scenario_id, mem_turns)
    hits = await system.retrieve(query, channels=("em_cosine", "em_pattern_v15"), k=5)

LLM model: gpt-5-mini for V15 tier emission and judge.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.vector_store.data_types import VectorStoreCollectionConfig
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Content,
    Event,
    FormatOptions,
    MessageContext,
    Text,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

EVAL_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(EVAL_ROOT.parents[0] / ".env")

NAMESPACE = "hard_bench"
HARD_BENCH_DIR = Path(__file__).resolve().parent
CACHE_DIR = HARD_BENCH_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LLM_MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536


# -----------------------------------------------------------------------------
# LLM cache (synchronous, file-based, model-keyed)
# -----------------------------------------------------------------------------


class LLMCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._d: dict = {}
        if path.exists():
            try:
                self._d = json.loads(path.read_text())
            except Exception:
                self._d = {}
        self._dirty = False

    def get(self, key: str):
        return self._d.get(key)

    def put(self, key: str, value) -> None:
        self._d[key] = value
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        # Multi-process safe: hold an exclusive lock on a sidecar file across
        # the read-merge-write sequence so concurrent runners don't clobber
        # each other's updates and don't race on tmp->dest rename.
        import fcntl

        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock_path, "w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                # Re-read disk state and merge with in-memory updates.
                disk: dict = {}
                if self.path.exists():
                    try:
                        disk = json.loads(self.path.read_text())
                    except Exception:
                        disk = {}
                disk.update(self._d)
                self._d = disk
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text(json.dumps(self._d))
                tmp.replace(self.path)
                self._dirty = False
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _llm_cache_key(model: str, reasoning_effort: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}|{reasoning_effort}|{prompt}".encode()).hexdigest()


# -----------------------------------------------------------------------------
# Hit data type
# -----------------------------------------------------------------------------


@dataclass
class Hit:
    """A retrieved memory unit, channel-agnostic."""

    turn_id: int  # original memory turn id (for gold matching)
    text: str  # the formatted text shown to the agent
    score: float  # channel-local score (not cross-channel comparable)
    channel: str  # which retriever produced this hit
    properties: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# RRF fusion
# -----------------------------------------------------------------------------


def rrf_fuse(channel_hits: dict[str, list[Hit]], k: int = 5, c: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion across channels. Returns top-k unique hits.

    Hit identity = turn_id. Tie-break by max channel-score.
    """
    scores: dict[int, float] = defaultdict(float)
    best_hit: dict[int, Hit] = {}
    for channel, hits in channel_hits.items():
        for rank, hit in enumerate(hits, start=1):
            scores[hit.turn_id] += 1.0 / (c + rank)
            if hit.turn_id not in best_hit or hit.score > best_hit[hit.turn_id].score:
                best_hit[hit.turn_id] = hit
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [best_hit[tid] for tid, _ in ordered]


# -----------------------------------------------------------------------------
# V15 pattern matching channel (with soft judge weights)
# -----------------------------------------------------------------------------


V15_TIERS_PROMPT = """A user asked a debugging or task-related question; we want to surface earlier notes that touch on the same general pattern. Identify the pattern at THREE levels of abstraction so we can search at all of them.

Output a strict 3-tier hierarchy:
- META: the broadest category this question belongs to. One short phrase.
- MID:  the specific named pattern this question is an instance of. One short phrase.
- SPECIFIC: the very narrow sub-pattern. One short phrase.

For each tier, write 2 short probe sentences in casual prose, the way a colleague might describe an OLD instance at that abstraction level. Do NOT mention specific terms from the user's question's surface domain.

Return STRICT JSON ONLY:
{{
  "meta":     {{"name": "...", "probes": ["...", "..."]}},
  "mid":      {{"name": "...", "probes": ["...", "..."]}},
  "specific": {{"name": "...", "probes": ["...", "..."]}}
}}

Question:
{problem}
"""

V15_JUDGE_PROMPT = """A user asked a question. We compressed it to an abstract pattern, then retrieved candidate memories. Filter candidates by structural validity.

Pattern:
  META: {meta_name}
  MID:  {mid_name}
  SPECIFIC: {spec_name}

The user's question:
{problem}

Candidates:
{candidates}

For each candidate, judge: YES (clearly an instance of MID, even if surface domain differs) / PARTIAL / NO.

Return STRICT JSON: {{"judgments": [{{"id": <int>, "verdict": "YES|PARTIAL|NO"}}]}}
"""

V15_SOFT_WEIGHTS = {
    "YES": 1.0,
    "PARTIAL": 0.85,
    "NO": 0.7,
}  # per softjudge_winning memo


def _extract_json(text: str):
    import re

    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\[.*\]|\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# UnifiedSystem (per-scenario)
# -----------------------------------------------------------------------------


class UnifiedSystem:
    """Per-scenario retrieval system. One instance per scenario.

    Wires real EventMemory (Qdrant + SQLite) as the substrate.
    Provides retrieval channels with shared underlying corpus.
    """

    def __init__(
        self,
        memory: EventMemory,
        llm_cache: LLMCache,
        openai_client,
        scenario_id: str,
        # turn_id → (text, speaker, ts) for direct lookups (V15 needs raw texts)
        turn_table: dict[int, tuple[str, str, str]],
        current_time: dt.datetime | None = None,
    ) -> None:
        self.memory = memory
        self.llm_cache = llm_cache
        self.openai_client = openai_client
        self.scenario_id = scenario_id
        self.turn_table = turn_table  # for V15 corpus access
        # current_time = the FIXED "now" for this scenario. Required for
        # relative anchors ("last week"). Never datetime.now() — benchmarks
        # must have a deterministic current time. If not supplied, derive
        # from max memory turn timestamp + 1 day.
        if current_time is None:
            if not turn_table:
                raise ValueError(
                    "current_time must be supplied when turn_table is empty; "
                    "benchmark scenarios require a fixed deterministic current time."
                )
            latest = max(
                dt.datetime.fromisoformat(t[2].replace("Z", "+00:00"))
                for t in turn_table.values()
            )
            current_time = latest + dt.timedelta(days=1)
        self.current_time = current_time
        # Entity-resolution memory (R23+R24 prose-fact + DSU). Populated by
        # build_system when em_entity is requested. None means the channel
        # will return [].
        self._entity_store = None
        self._entity_cache = None
        self._entity_budget = None

    # ---- LLM helper -------------------------------------------------------

    async def llm(self, prompt: str, *, reasoning_effort: str = "medium") -> str:
        key = _llm_cache_key(LLM_MODEL, reasoning_effort, prompt)
        cached = self.llm_cache.get(key)
        if cached is not None:
            return cached
        resp = await self.openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=reasoning_effort,
        )
        text = resp.choices[0].message.content or ""
        self.llm_cache.put(key, text)
        self.llm_cache.save()
        return text

    # ---- Hit text formatting ---------------------------------------------

    def _format_turn_text(self, tid: int, fallback_text: str | None = None) -> str:
        """Render a memory turn for inclusion in agent prompts.

        Format: `[Weekday, Month D, YYYY, H:MM:SS AM/PM UTC] Speaker: "text"`
        Uses turn_table as the source of truth (text, speaker, ts_iso).
        Falls back to the supplied text if turn_table lookup fails.
        """
        entry = self.turn_table.get(tid)
        if entry is None:
            return fallback_text if fallback_text is not None else ""
        text, speaker, ts_iso = entry[0], entry[1], entry[2]
        try:
            ts = dt.datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
            stamp = ts.strftime("%A, %B %-d, %Y, %-I:%M:%S %p UTC")
            return f'[{stamp}] {speaker}: "{text}"'
        except Exception:
            return f'{speaker}: "{text}"'

    # ---- Retrieval channels ----------------------------------------------

    async def retrieve_em_cosine(self, query: str, k: int = 5) -> list[Hit]:
        """Native EventMemory query with formatted context."""
        result = await self.memory.query(
            query,
            vector_search_limit=k * 4,  # over-fetch for dedup
            expand_context=0,
            format_options=FormatOptions(date_style="medium", time_style="short"),
        )
        hits: list[Hit] = []
        seen: set[int] = set()
        for ssc in result.scored_segment_contexts:
            if not ssc.segments:
                continue
            seg = ssc.segments[0]
            tid = seg.properties.get("turn_id", -1)
            if tid in seen:
                continue
            seen.add(tid)
            fallback = EventMemory.string_from_segment_context(
                ssc.segments,
                format_options=FormatOptions(date_style="medium", time_style="short"),
            )
            text = self._format_turn_text(tid, fallback_text=fallback)
            hits.append(
                Hit(
                    turn_id=tid,
                    text=text,
                    score=ssc.score,
                    channel="em_cosine",
                    properties=dict(seg.properties),
                )
            )
            if len(hits) >= k:
                break
        return hits

    async def retrieve_em_pattern_v15(self, query: str, k: int = 5) -> list[Hit]:
        """V15: tier emission + cosine on probes + soft judge filter."""
        # Step 1: emit tiers
        tiers_raw = await self.llm(V15_TIERS_PROMPT.format(problem=query))
        tiers = _extract_json(tiers_raw)
        if tiers is None or "meta" not in tiers:
            return []  # tier emission failed; return empty (let other channels carry)
        # Step 2: probes via EM.query
        meta_probes = tiers["meta"]["probes"]
        mid_probes = tiers["mid"]["probes"]
        spec_probes = tiers["specific"]["probes"]
        # Run all probes in parallel; collect candidate set with score blending
        all_probes = (
            [(p, "meta") for p in meta_probes]
            + [(p, "mid") for p in mid_probes]
            + [(p, "spec") for p in spec_probes]
        )

        # Per-tier weights (from V15)
        tier_w = {"meta": 0.25, "mid": 0.5, "spec": 0.25}

        # Run probes (each is a small EM.query)
        probe_results = await asyncio.gather(
            *(self.memory.query(p, vector_search_limit=k * 2) for p, _ in all_probes)
        )

        # Aggregate candidate scores: per turn_id, max score per tier × weight
        per_tier_score: dict[int, dict[str, float]] = defaultdict(
            lambda: {"meta": 0.0, "mid": 0.0, "spec": 0.0}
        )
        candidate_meta: dict[int, dict] = {}
        for (probe_text, tier), result in zip(all_probes, probe_results):
            for ssc in result.scored_segment_contexts:
                if not ssc.segments:
                    continue
                seg = ssc.segments[0]
                tid = seg.properties.get("turn_id", -1)
                if tid < 0:
                    continue
                if ssc.score > per_tier_score[tid][tier]:
                    per_tier_score[tid][tier] = ssc.score
                if tid not in candidate_meta:
                    fallback = EventMemory.string_from_segment_context(
                        ssc.segments,
                        format_options=FormatOptions(
                            date_style="medium", time_style="short"
                        ),
                    )
                    text = self._format_turn_text(tid, fallback_text=fallback)
                    candidate_meta[tid] = {
                        "text": text,
                        "properties": dict(seg.properties),
                    }

        # Compute blended C_blend per candidate
        blended = []
        for tid, ts in per_tier_score.items():
            blend = (
                tier_w["meta"] * ts["meta"]
                + tier_w["mid"] * ts["mid"]
                + tier_w["spec"] * ts["spec"]
            )
            blended.append((tid, blend))
        blended.sort(key=lambda x: -x[1])
        top_25 = blended[: min(25, len(blended))]
        if not top_25:
            return []

        # Step 3: judge filter (soft weights)
        cand_lines = []
        for i, (tid, _b) in enumerate(top_25):
            cand_lines.append(f"  [{i}] {candidate_meta[tid]['text'][:240]}")
        judge_prompt = V15_JUDGE_PROMPT.format(
            meta_name=tiers["meta"]["name"],
            mid_name=tiers["mid"]["name"],
            spec_name=tiers["specific"]["name"],
            problem=query,
            candidates="\n".join(cand_lines),
        )
        judge_raw = await self.llm(judge_prompt)
        judgments_obj = _extract_json(judge_raw) or {}
        verdicts: dict[int, str] = {}
        for j in judgments_obj.get("judgments", []):
            try:
                verdicts[int(j["id"])] = str(j.get("verdict", "")).upper()
            except Exception:
                continue

        # Step 4: re-rank by C_blend × soft_weight
        weighted = []
        for i, (tid, blend) in enumerate(top_25):
            w = V15_SOFT_WEIGHTS.get(
                verdicts.get(i, "PARTIAL"), V15_SOFT_WEIGHTS["PARTIAL"]
            )
            weighted.append((tid, blend * w))
        weighted.sort(key=lambda x: -x[1])

        out: list[Hit] = []
        for tid, weighted_score in weighted[:k]:
            out.append(
                Hit(
                    turn_id=tid,
                    text=candidate_meta[tid]["text"],
                    score=weighted_score,
                    channel="em_pattern_v15",
                    properties=candidate_meta[tid]["properties"],
                )
            )
        return out

    # ---- TODO stubs for next phase ----------------------------------------

    async def retrieve_em_temporal(self, query: str, k: int = 5) -> list[Hit]:
        """Real temporal_retrieval channel. Requires _temporal_retriever set.

        Strategy: ignore the agent's spreading-activation probe text and use
        the original task_anchor_phrase (the task_prompt). Spreading-activation
        is concept-driven; it loses temporal anchors. The temporal channel's
        job is anchor filtering, not concept diversity. So we use the task
        prompt verbatim every time. RRF will dedup against em_cosine probes.
        """
        retriever = getattr(self, "_temporal_retriever", None)
        if retriever is None:
            return []
        ref_time_iso = self.current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        anchor = getattr(self, "task_anchor_phrase", None)
        effective_query = anchor or query
        results = await retriever.query(effective_query, ref_time=ref_time_iso, k=k * 2)
        hits: list[Hit] = []
        for res in results[:k]:
            try:
                tid = int(res.doc_id)
            except Exception:
                tid = -1
            if tid not in self.turn_table:
                continue
            entry = self.turn_table[tid]
            speaker = entry[1]
            full_props = (
                entry[3] if len(entry) > 3 else {"turn_id": tid, "speaker": speaker}
            )
            formatted = self._format_turn_text(tid)
            hits.append(
                Hit(
                    turn_id=tid,
                    text=formatted,
                    score=res.score,
                    channel="em_temporal",
                    properties=dict(full_props),
                )
            )
        return hits

    async def retrieve_entity(self, query: str, k: int = 14) -> dict | None:
        """Native R23+R24 entity-resolution retrieval.

        Returns the reference's native output bundle (facts + resolution_map
        + store) so the caller can render it via the reference's own
        format_facts_for_read + format_resolution_map. The agent renders
        this as a SEPARATE block in its context, NOT fused with event
        memory hits — entity memory is a parallel channel.

        Returns None when entity memory was not built for this scenario.
        """
        if self._entity_store is None or self._entity_cache is None:
            return None
        from hard_bench.entity import (
            v2,  # for format_*; loaded by entity.py's sys.path injection
            v3,
        )

        def _run():
            facts, resolution_map = v3.retrieve(
                query,
                self._entity_store,
                self._entity_cache,
                self._entity_budget,
                top_k=k,
                expand_hops=1,
                llm_dedup=True,
            )
            return facts, resolution_map

        facts, resolution_map = await asyncio.to_thread(_run)
        try:
            self._entity_cache.save()
        except Exception:
            pass
        return {
            "facts": facts,
            "resolution_map": resolution_map,
            "store": self._entity_store,
        }

    # ---- Filter-style temporal integration --------------------------------

    async def _compute_temporal_eligibility(self) -> set[int] | None:
        """One-shot per scenario: run planner+classifier+resolver on the
        task_anchor_phrase to determine which turn_ids fall within the
        anchor's window. Cached on self._eligible_turn_ids. Returns None
        if no temporal_retriever or no anchor.
        """
        cached = getattr(self, "_eligible_turn_ids", "UNSET")
        if cached != "UNSET":
            return cached
        retriever = getattr(self, "_temporal_retriever", None)
        anchor = getattr(self, "task_anchor_phrase", None)
        if retriever is None or anchor is None:
            self._eligible_turn_ids = None
            return None
        from temporal_retrieval.core import doc_passes_filter

        ref_time_iso = self.current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        plan = await retriever._planner.plan(anchor, ref_time_iso)
        leaves_flat = [
            (ci, li, leaf)
            for ci, clause in enumerate(plan.expr)
            for li, leaf in enumerate(clause)
        ]
        kinds = await retriever._classify_leaves(anchor, ref_time_iso, leaves_flat)
        anchors = await retriever._resolve_anchors(
            anchor, ref_time_iso, leaves_flat, kinds
        )
        valid_includes = []
        valid_excludes = []
        for ci, li, leaf in leaves_flat:
            ivs = anchors.get((ci, li), [])
            if not ivs:
                continue
            if leaf.direction == "not_in":
                valid_excludes.append(ivs)
            else:
                valid_includes.append((leaf.direction, ivs))
        # If no anchors resolved, no filtering
        if not valid_includes and not valid_excludes:
            self._eligible_turn_ids = None
            return None
        # Production-equivalent eligibility: docs pass the filter using their
        # ACTUAL extracted intervals only. No ref_time fallback. Authoring
        # date has no causal relationship to query-window relevance for
        # timeless policies / old precedents. Production retriever.py handles
        # empty-interval docs via empty_doc_mask=1.0 at the scoring layer; in
        # this binary-filter setup the equivalent is: empty-interval docs
        # fail the include filter and are NOT in eligible_filt, but they
        # still reach build_pool's pool via raw cosine top-(K/2) and topup
        # from raw ranks K..2K. Cosine carries them; the filter only boosts
        # docs that genuinely have temporal extracted intervals.
        eligible: set[int] = set()
        for did_str in retriever._doc_ref_us.keys():
            ivs = list(retriever._doc_ivs.get(did_str, []))
            if doc_passes_filter(ivs, valid_includes, valid_excludes):
                try:
                    eligible.add(int(did_str))
                except Exception:
                    continue
        self._eligible_turn_ids = eligible
        return eligible

    # ---- Multi-channel ensemble ------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        channels: tuple[str, ...] = ("em_cosine",),
        k: int = 5,
        temporal_filter: bool = False,
    ) -> list[Hit]:
        """Run requested channels in parallel and RRF-fuse to top-k.

        If temporal_filter=True, after fusion drop hits whose turn_id is
        outside the temporal anchor's eligible set. Filter is computed
        once per scenario via _compute_temporal_eligibility().
        """
        # Synthetic flag-channel: "temporal_filter" turns on post-filter rather
        # than retrieving its own hits. Strip from channels list and set flag.
        real_channels = [c for c in channels if c != "temporal_filter"]
        if "temporal_filter" in channels:
            temporal_filter = True

        registry = {
            "em_cosine": self.retrieve_em_cosine,
            "em_pattern_v15": self.retrieve_em_pattern_v15,
            "em_temporal": self.retrieve_em_temporal,
        }
        # em_entity is NOT a participant in the event-memory channel fusion;
        # it returns a native (facts, resolution_map, store) bundle via
        # retrieve_entity() and is rendered as a separate prompt block.
        real_channels = [c for c in real_channels if c != "em_entity"]
        # When filtering, over-fetch from each channel so post-filter still has K
        per_channel_k = k * 3 if temporal_filter else k
        tasks = {
            ch: registry[ch](query, k=per_channel_k)
            for ch in real_channels
            if ch in registry
        }
        results = await asyncio.gather(*tasks.values())
        channel_hits = dict(zip(tasks.keys(), results))

        if len(real_channels) == 1:
            fused = list(channel_hits[real_channels[0]])
        elif real_channels:
            fused = rrf_fuse(channel_hits, k=per_channel_k)
        else:
            fused = []

        if temporal_filter:
            eligible = await self._compute_temporal_eligibility()
            if eligible is not None:
                # Use reference temporal_retrieval.core.build_pool directly.
                # build_pool semantics: top-(K/2) raw-semantic ∪ top-(K/2)
                # filter-survivor-semantic, deduped, topped up from raw
                # ranks K+1..2K to reach pool_size. Always preserves at
                # least the raw top-(K/2) and pads from raw deeper ranks
                # rather than displacing them.
                from temporal_retrieval.core import build_pool

                hits_by_did = {str(h.turn_id): h for h in fused}
                sem_scores = {did: h.score for did, h in hits_by_did.items()}
                all_dids = list(hits_by_did.keys())
                eligible_filt = [d for d in all_dids if int(d) in eligible]
                pool_dids = build_pool(sem_scores, all_dids, eligible_filt, pool_size=k)
                pool_hits = [hits_by_did[d] for d in pool_dids if d in hits_by_did]
                # Post-limit chrono sort: turn_ids are sequential per scenario,
                # so sorting by turn_id presents results in chronological order
                # without exposing rank to the agent.
                pool_hits.sort(key=lambda h: h.turn_id)
                return pool_hits
        return fused[:k]


# -----------------------------------------------------------------------------
# Construction helpers
# -----------------------------------------------------------------------------


def _scenario_collection(scenario_id: str) -> str:
    """Return a Qdrant collection name (≤32 chars, [a-z0-9_]+)."""
    base = f"hb_{scenario_id.lower()}"
    if len(base) <= 32:
        return base
    return "hb_" + hashlib.sha1(scenario_id.encode()).hexdigest()[:28]


async def build_system(
    scenario_id: str,
    memory_turns: list[dict],
    *,
    qdrant_client: AsyncQdrantClient,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
    openai_client,
    llm_cache: LLMCache,
    overwrite: bool = True,
    current_time: dt.datetime | None = None,
    temporal_infra: tuple | None = None,
    entity_infra: tuple | None = None,
) -> UnifiedSystem:
    """Construct an EventMemory + UnifiedSystem for one scenario.

    memory_turns: list of {turn_id, speaker, timestamp (ISO), text, ...}
    temporal_infra: optional (embed_fn, rerank_fn) tuple — when provided,
      a per-scenario TemporalRetriever is built and indexed.
    entity_infra: optional (cache, budget) tuple — when provided, the
      R23+R24 prose-fact + DSU entity-resolution memory is ingested for
      the scenario and stored on the system.
    """
    collection_name = _scenario_collection(scenario_id)
    partition_key = collection_name

    vs_params = QdrantVectorStoreParams(client=qdrant_client)
    vector_store = QdrantVectorStore(vs_params)
    await vector_store.startup()

    if overwrite:
        await vector_store.delete_collection(namespace=NAMESPACE, name=collection_name)
        await segment_store.delete_partition(partition_key)

    collection = await vector_store.open_or_create_collection(
        namespace=NAMESPACE,
        name=collection_name,
        config=VectorStoreCollectionConfig(
            vector_dimensions=embedder.dimensions,
            similarity_metric=embedder.similarity_metric,
            properties_schema=EventMemory.expected_vector_store_collection_schema(),
        ),
    )
    partition = await segment_store.open_or_create_partition(partition_key)

    memory = EventMemory(
        EventMemoryParams(
            vector_store_collection=collection,
            segment_store_partition=partition,
            embedder=embedder,
            reranker=None,
            derive_sentences=False,
            max_text_chunk_length=1000,
        )
    )

    # Convert memory_turns to Events and ingest
    events: list[Event] = []
    turn_table: dict[int, tuple[str, str, str]] = {}
    for turn in memory_turns:
        ts = dt.datetime.fromisoformat(turn["timestamp"].replace("Z", "+00:00"))
        speaker = turn["speaker"]
        text = turn["text"]
        tid = int(turn["turn_id"])
        relevance = turn.get("relevance", "noise")
        plant_id = turn.get("plant_id")
        props: dict = {
            "scenario_id": scenario_id,
            "turn_id": tid,
            "speaker": speaker,
            "relevance": relevance,
        }
        if plant_id:
            props["plant_id"] = plant_id
        events.append(
            Event(
                uuid=uuid4(),
                timestamp=ts,
                body=Content(
                    context=MessageContext(source=speaker),
                    items=[Text(text=text.strip())],
                ),
                properties=props,
            )
        )
        # Store the full props dict (including plant_id) so em_temporal/em_entity
        # channels can preserve it in their Hit objects for plant_retrieved checks.
        turn_table[tid] = (text, speaker, turn["timestamp"], dict(props))

    await memory.encode_events(events)

    sys_obj = UnifiedSystem(
        memory=memory,
        current_time=current_time,
        llm_cache=llm_cache,
        openai_client=openai_client,
        scenario_id=scenario_id,
        turn_table=turn_table,
    )

    # Optional: build temporal_retriever for this scenario when temporal_infra is supplied
    if temporal_infra is not None:
        from temporal_retrieval import Doc as TRDoc
        from temporal_retrieval import TemporalRetriever

        embed_fn, rerank_fn = temporal_infra
        retriever = TemporalRetriever(
            embed_fn=embed_fn,
            rerank_fn=rerank_fn,
            cache_dir=str(CACHE_DIR / "temporal_retrieval"),
            pool_size=10,
        )
        docs = []
        for tid, entry in turn_table.items():
            text, speaker, ts = entry[0], entry[1], entry[2]
            ts_iso = ts if "T" in ts else ts + "T00:00:00Z"
            # Include explicit date in doc text so the temporal extractor can
            # pin a calendar interval to the doc. Without this, casual chat
            # text has no datey words and the filter passes everything.
            try:
                ts_dt = dt.datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
                date_prefix = f"On {ts_dt.strftime('%B %d, %Y')}, "
            except Exception:
                date_prefix = ""
            formatted = f"{date_prefix}{speaker} said: {text}"
            docs.append(TRDoc(id=str(tid), text=formatted, ref_time=ts_iso))
        await retriever.index(docs)
        sys_obj._temporal_retriever = retriever

    # Optional: ingest R23+R24 prose-fact + DSU entity-resolution memory.
    if entity_infra is not None:
        from hard_bench.entity import build_entity_store
        entity_cache, entity_budget = entity_infra
        ent_turns = []
        for tid in sorted(turn_table.keys()):
            entry = turn_table[tid]
            text, speaker = entry[0], entry[1]
            ent_turns.append((tid, f"{speaker}: {text}"))
        sys_obj._entity_store = await build_entity_store(
            ent_turns, entity_cache, entity_budget
        )
        sys_obj._entity_cache = entity_cache
        sys_obj._entity_budget = entity_budget

    return sys_obj


async def make_infrastructure():
    """One-shot infrastructure setup. Returns (qdrant_client, segment_store, embedder, openai_client, llm_cache)."""
    import openai

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )

    sqlite_path = CACHE_DIR / "hard_bench.sqlite3"
    sql_url = f"sqlite+aiosqlite:///{sqlite_path}"
    engine = create_async_engine(sql_url)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = openai.AsyncOpenAI()

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client, model=EMBED_MODEL, dimensions=EMBED_DIMS
        )
    )

    llm_cache = LLMCache(CACHE_DIR / "llm_cache.json")

    return qdrant_client, segment_store, embedder, openai_client, llm_cache


async def make_temporal_infra(openai_client):
    """Build (embed_fn, rerank_fn) for the temporal_retrieval channel.

    embed_fn uses the same OpenAI text-embedding-3-small as EM.
    rerank_fn uses sentence-transformers ms-marco-MiniLM cross-encoder
    (matches v5.1 production stack).
    """
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    async def embed_fn(texts):
        if not texts:
            return []
        resp = await openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )
        return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]

    async def rerank_fn(query, doc_texts):
        if not doc_texts:
            return []
        pairs = [[query, d] for d in doc_texts]
        scores = ce.predict(pairs)
        return [float(s) for s in scores]

    return embed_fn, rerank_fn


def make_entity_infra():
    """Build (cache, budget) for the em_entity channel.

    Returns a shared file-backed LLM/embed cache and a generous budget.
    The cache is shared across scenarios within one run, leveraging the
    R23+R24 reference modules' own caching layer.
    """
    from hard_bench.entity import make_cache, make_budget
    entity_cache = make_cache(CACHE_DIR / "entity_llm_cache.json")
    entity_budget = make_budget()
    return entity_cache, entity_budget
