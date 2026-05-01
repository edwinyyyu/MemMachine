"""Topic-segment hierarchical retrieval.

At ingest time, segment each conversation into topic-coherent chunks (either
fixed-size or LLM-driven) and produce a short summary for each chunk. At
retrieval, match segment summaries (coarse layer) plus raw turns (fine layer),
then expand summary hits to their constituent turns.

Two ingestion variants:
  - fixed: consecutive windows of N turns.
  - llm: LLM-driven boundary detection + 1-sentence summary per segment.

Retrieval: hierarchical — summary top-Ms plus turn top-Kt, merge via
score = max(cos(q, turn), alpha * cos(q, parent_summary)).

This file provides both the builder (run directly to create the segmentation)
and the retrieval class (TopicSegRetriever) used by topic_segment_eval.py.
"""

from __future__ import annotations

import argparse
import json
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

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEGMENT_MODEL = "gpt-5-mini"


# ---------------------------------------------------------------------------
# Caches — topic-segment-specific, read existing caches too
# ---------------------------------------------------------------------------
class TopicSegEmbeddingCache(EmbeddingCache):
    """Reads bestshot + existing caches, writes to topic_seg-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in (
            "embedding_cache.json",
            "bestshot_embedding_cache.json",
            "topic_seg_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "topic_seg_embedding_cache.json"
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


class TopicSegLLMCache(LLMCache):
    """Reads bestshot + existing caches, writes to topic_seg-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in (
            "llm_cache.json",
            "bestshot_llm_cache.json",
            "topic_seg_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "topic_seg_llm_cache.json"
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
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class TopicSegment:
    """A topic-coherent chunk of consecutive turns in a conversation."""

    conversation_id: str
    segment_idx: int  # per-conversation index
    turn_ids: list[int]  # constituent turn_ids (sorted)
    summary: str  # short summary (1-3 sentences)
    embedding: list[float] | None = None  # summary embedding


@dataclass
class SegmentationResult:
    variant: str
    conversations: dict[str, list[TopicSegment]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM segmentation prompt
# ---------------------------------------------------------------------------
LLM_SEGMENT_PROMPT = """\
You are segmenting a conversation into topic-coherent chunks. The conversation \
is provided as a numbered list of turns. Each turn is labeled with its turn_id.

Your task:
1. Identify topic boundaries — points where the conversation shifts to a new \
subject, task, or context.
2. For each segment (run of consecutive turns on the same topic), output:
   - The list of turn_ids in that segment
   - A short 1-2 sentence summary capturing the segment's topical content, \
using specific vocabulary that appears in the turns (names, tools, events, \
decisions). The summary will be embedded and matched against future queries.

Segments should typically span 5-25 turns. Very short segments (1-3 turns) \
are fine for quick subtopic switches. Avoid segments > 40 turns unless the \
topic is truly coherent throughout.

Conversation window (turn_ids {start_tid} to {end_tid}):
{conversation}

Output format (strict):
SEGMENT turn_ids=<comma-separated turn_ids>
SUMMARY: <1-2 sentence topical summary>
SEGMENT turn_ids=<comma-separated turn_ids>
SUMMARY: <1-2 sentence topical summary>
...

Cover ALL turn_ids in the window (every turn must appear in exactly one \
segment). Output nothing else."""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class TopicSegBuilder:
    """Builds topic segments for each conversation in a SegmentStore."""

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
    ):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = TopicSegEmbeddingCache()
        self.llm_cache = TopicSegLLMCache()

    def _conversation_turns(self, conv_id: str) -> list[Segment]:
        segs = [s for s in self.store.segments if s.conversation_id == conv_id]
        segs.sort(key=lambda s: s.turn_id)
        return segs

    def _embed(self, text: str) -> np.ndarray:
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        emb = np.array(resp.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, emb)
        return emb

    def _llm(self, prompt: str, model: str = SEGMENT_MODEL) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            return cached
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4000,
        )
        text = resp.choices[0].message.content or ""
        self.llm_cache.put(model, prompt, text)
        return text

    # -------- fixed-size segmentation --------
    def build_fixed(
        self,
        conv_ids: list[str],
        chunk_size: int = 10,
    ) -> SegmentationResult:
        result = SegmentationResult(variant=f"fixed_n{chunk_size}")
        for cid in conv_ids:
            turns = self._conversation_turns(cid)
            segs: list[TopicSegment] = []
            for seg_i, i in enumerate(range(0, len(turns), chunk_size)):
                chunk = turns[i : i + chunk_size]
                turn_ids = [t.turn_id for t in chunk]
                summary = self._summarize_fixed(chunk)
                emb = self._embed(summary)
                segs.append(
                    TopicSegment(
                        conversation_id=cid,
                        segment_idx=seg_i,
                        turn_ids=turn_ids,
                        summary=summary,
                        embedding=emb.tolist(),
                    )
                )
            result.conversations[cid] = segs
        return result

    def _summarize_fixed(self, chunk: list[Segment]) -> str:
        """Simple summary: concatenate first 40 chars of each turn with speaker.

        Fixed-size variant uses NO LLM — just a concatenated snippet. This
        means the summary embedding is essentially a pooled representation of
        the chunk's vocabulary, which is fine for testing the coarse-layer
        hypothesis.
        """
        parts = []
        for s in chunk:
            snippet = s.text.strip()[:120]
            parts.append(f"{s.role}: {snippet}")
        return " | ".join(parts)

    # -------- LLM-driven segmentation --------
    def build_llm(
        self,
        conv_ids: list[str],
        window_size: int = 40,
    ) -> SegmentationResult:
        result = SegmentationResult(variant=f"llm_w{window_size}")
        for cid in conv_ids:
            turns = self._conversation_turns(cid)
            segs = self._segment_one_conv_llm(cid, turns, window_size)
            result.conversations[cid] = segs
        return result

    def _segment_one_conv_llm(
        self,
        cid: str,
        turns: list[Segment],
        window_size: int,
    ) -> list[TopicSegment]:
        """Slide a window over turns; each window produces segments; stitch."""
        all_segs: list[TopicSegment] = []
        tid_to_seg = {t.turn_id: t for t in turns}
        seg_i = 0
        i = 0
        while i < len(turns):
            window_turns = turns[i : i + window_size]
            start_tid = window_turns[0].turn_id
            end_tid = window_turns[-1].turn_id
            conv_text = "\n".join(
                f"[{t.turn_id}] {t.role}: {t.text[:300]}" for t in window_turns
            )
            prompt = LLM_SEGMENT_PROMPT.format(
                start_tid=start_tid,
                end_tid=end_tid,
                conversation=conv_text,
            )
            output = self._llm(prompt)
            window_segs = self._parse_segmentation(output)
            # Post-process: validate + fill gaps
            covered_tids: set[int] = set()
            window_tids = {t.turn_id for t in window_turns}
            valid_segs: list[tuple[list[int], str]] = []
            for tids, summary in window_segs:
                tids_clean = [
                    t for t in tids if t in window_tids and t not in covered_tids
                ]
                if not tids_clean or not summary.strip():
                    continue
                covered_tids.update(tids_clean)
                valid_segs.append((tids_clean, summary.strip()))
            # Any uncovered turns: put into a fallback segment
            missing = [t.turn_id for t in window_turns if t.turn_id not in covered_tids]
            if missing:
                fallback_text = " | ".join(
                    f"{tid_to_seg[t].role}: {tid_to_seg[t].text[:80]}"
                    for t in missing[:8]
                )
                valid_segs.append((missing, fallback_text))
            # Build TopicSegments
            for tids, summary in valid_segs:
                tids_sorted = sorted(tids)
                emb = self._embed(summary)
                all_segs.append(
                    TopicSegment(
                        conversation_id=cid,
                        segment_idx=seg_i,
                        turn_ids=tids_sorted,
                        summary=summary,
                        embedding=emb.tolist(),
                    )
                )
                seg_i += 1
            i += window_size
        return all_segs

    def _parse_segmentation(self, text: str) -> list[tuple[list[int], str]]:
        segs: list[tuple[list[int], str]] = []
        current_tids: list[int] | None = None
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.upper().startswith("SEGMENT"):
                # e.g. "SEGMENT turn_ids=1,2,3"
                after = line.split("=", 1)
                if len(after) == 2:
                    tids_str = after[1].strip()
                    tids = []
                    for part in tids_str.split(","):
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            tids.append(int(part))
                        except ValueError:
                            pass
                    current_tids = tids
            elif line.upper().startswith("SUMMARY:"):
                summary = line[len("SUMMARY:") :].strip()
                if current_tids is not None:
                    segs.append((current_tids, summary))
                    current_tids = None
        return segs

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def save_segmentation(result: SegmentationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "variant": result.variant,
        "conversations": {
            cid: [
                {
                    "segment_idx": s.segment_idx,
                    "turn_ids": s.turn_ids,
                    "summary": s.summary,
                    "embedding": s.embedding,
                }
                for s in segs
            ]
            for cid, segs in result.conversations.items()
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)


def load_segmentation(path: Path) -> SegmentationResult:
    with open(path) as f:
        data = json.load(f)
    result = SegmentationResult(variant=data["variant"])
    for cid, segs in data["conversations"].items():
        result.conversations[cid] = [
            TopicSegment(
                conversation_id=cid,
                segment_idx=s["segment_idx"],
                turn_ids=s["turn_ids"],
                summary=s["summary"],
                embedding=s.get("embedding"),
            )
            for s in segs
        ]
    return result


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
@dataclass
class TopicSegRetrievalResult:
    segments: list[Segment]  # ordered, deduped by Segment.index
    metadata: dict = field(default_factory=dict)


class TopicSegStore:
    """In-memory per-conversation index of topic-segment summaries."""

    def __init__(self, seg_result: SegmentationResult, raw_store: SegmentStore):
        self.seg_result = seg_result
        self.raw_store = raw_store
        # Build normalized embeddings per conversation, plus tid->segment map.
        self.conv_embeddings: dict[str, np.ndarray] = {}
        self.conv_segments: dict[str, list[TopicSegment]] = {}
        for cid, segs in seg_result.conversations.items():
            if not segs:
                continue
            embs = np.array([s.embedding for s in segs], dtype=np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self.conv_embeddings[cid] = embs / norms
            self.conv_segments[cid] = segs

        # Map (conv_id, turn_id) -> raw Segment index
        self.turn_to_raw_idx: dict[tuple[str, int], int] = {}
        for seg in raw_store.segments:
            self.turn_to_raw_idx[(seg.conversation_id, seg.turn_id)] = seg.index

        # Map (conv_id, turn_id) -> parent TopicSegment
        self.turn_to_parent: dict[tuple[str, int], TopicSegment] = {}
        for cid, segs in self.conv_segments.items():
            for s in segs:
                for tid in s.turn_ids:
                    self.turn_to_parent[(cid, tid)] = s

    def search_summaries(
        self,
        query_emb: np.ndarray,
        conv_id: str,
        top_m: int,
    ) -> list[tuple[TopicSegment, float]]:
        if conv_id not in self.conv_embeddings:
            return []
        qn = query_emb / max(np.linalg.norm(query_emb), 1e-10)
        sims = self.conv_embeddings[conv_id] @ qn
        top_k = min(top_m, len(sims))
        idx = np.argsort(sims)[::-1][:top_k]
        return [(self.conv_segments[conv_id][i], float(sims[i])) for i in idx]


class TopicSegRetriever:
    """Hierarchical retrieval: summary-layer + turn-layer, merged via max-score.

    Exposes the interface expected by fair_backfill_eval's evaluator:
      .store (raw SegmentStore for baseline cosine search)
      .retrieve(question, conversation_id) -> object with .segments
      .embed_text(text) -> np.ndarray
      .reset_counters() / .save_caches()
      .embed_calls, .llm_calls
    """

    def __init__(
        self,
        raw_store: SegmentStore,
        topic_store: TopicSegStore,
        top_m_summaries: int = 3,
        top_kt_turns: int = 50,
        alpha: float = 1.0,
        base_source: str | None = None,
    ):
        """base_source: if set, denotes ensemble-style stacking name."""
        self.store = raw_store
        self.topic_store = topic_store
        self.top_m_summaries = top_m_summaries
        self.top_kt_turns = top_kt_turns
        self.alpha = alpha
        self.base_source = base_source

        self.client = OpenAI(timeout=60.0)
        self.embedding_cache = TopicSegEmbeddingCache()
        self.llm_cache = TopicSegLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        emb = np.array(resp.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, emb)
        self.embed_calls += 1
        return emb

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def retrieve(
        self,
        question: str,
        conversation_id: str,
    ) -> TopicSegRetrievalResult:
        q_emb = self.embed_text(question)

        # Turn-layer: cosine top-Kt
        turn_result = self.store.search(
            q_emb,
            top_k=self.top_kt_turns,
            conversation_id=conversation_id,
        )

        # Score turns by cosine
        turn_scores: dict[int, float] = {}
        turn_idx_to_seg: dict[int, Segment] = {}
        for seg, score in zip(turn_result.segments, turn_result.scores):
            turn_scores[seg.index] = score
            turn_idx_to_seg[seg.index] = seg

        # Summary-layer: top-Ms summaries
        summary_hits = self.topic_store.search_summaries(
            q_emb,
            conversation_id,
            self.top_m_summaries,
        )

        # Expand each hit to its constituent turns
        for tseg, summary_score in summary_hits:
            parent_boost = self.alpha * summary_score
            for tid in tseg.turn_ids:
                raw_idx = self.topic_store.turn_to_raw_idx.get((conversation_id, tid))
                if raw_idx is None:
                    continue
                if raw_idx not in turn_idx_to_seg:
                    turn_idx_to_seg[raw_idx] = self.store.segments[raw_idx]
                # Score turn as max(own_cos, alpha * parent_summary_score)
                prev = turn_scores.get(raw_idx, -1.0)
                if parent_boost > prev:
                    turn_scores[raw_idx] = parent_boost

        # Rank merged by score
        ranked_idx = sorted(
            turn_scores.keys(),
            key=lambda i: turn_scores[i],
            reverse=True,
        )
        segments = [turn_idx_to_seg[i] for i in ranked_idx]
        return TopicSegRetrievalResult(
            segments=segments,
            metadata={
                "num_summary_hits": len(summary_hits),
                "num_turns_from_summaries": sum(
                    len(ts.turn_ids) for ts, _ in summary_hits
                ),
            },
        )


# ---------------------------------------------------------------------------
# CLI: build segmentations for evaluation datasets
# ---------------------------------------------------------------------------
def _dataset_conv_ids(store: SegmentStore, filter_prefix: str | None) -> list[str]:
    cids = sorted({s.conversation_id for s in store.segments})
    if filter_prefix:
        cids = [c for c in cids if c.startswith(filter_prefix)]
    return cids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        choices=["fixed", "llm", "both"],
        default="both",
    )
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--window-size", type=int, default=40)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["locomo", "synthetic"],
    )
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_configs = {
        "locomo": ("segments_extended.npz", "locomo"),
        "synthetic": ("segments_synthetic.npz", None),
    }

    variants_to_run = []
    if args.variant in ("fixed", "both"):
        variants_to_run.append("fixed")
    if args.variant in ("llm", "both"):
        variants_to_run.append("llm")

    for ds_name in args.datasets:
        if ds_name not in dataset_configs:
            print(f"Unknown dataset: {ds_name}")
            continue
        npz_name, prefix = dataset_configs[ds_name]
        print(f"\n=== Dataset: {ds_name} (npz={npz_name}) ===")
        store = SegmentStore(data_dir=DATA_DIR, npz_name=npz_name)
        conv_ids = _dataset_conv_ids(store, prefix)
        print(f"  {len(conv_ids)} conversations")

        for variant in variants_to_run:
            print(f"\n  -- Variant: {variant} --")
            builder = TopicSegBuilder(store)
            t0 = time.time()
            if variant == "fixed":
                result = builder.build_fixed(conv_ids, chunk_size=args.chunk_size)
                suffix = f"fixed_n{args.chunk_size}"
            else:
                result = builder.build_llm(conv_ids, window_size=args.window_size)
                suffix = f"llm_w{args.window_size}"
            elapsed = time.time() - t0
            builder.save_caches()

            # Save to results
            out = RESULTS_DIR / f"topic_segments_{ds_name}_{suffix}.json"
            save_segmentation(result, out)
            total_segs = sum(len(v) for v in result.conversations.values())
            avg_turns = sum(
                len(s.turn_ids) for v in result.conversations.values() for s in v
            ) / max(total_segs, 1)
            avg_segs_per_conv = total_segs / max(len(conv_ids), 1)
            print(
                f"    built {total_segs} segments across {len(conv_ids)} convs "
                f"(avg {avg_segs_per_conv:.1f} segs/conv, "
                f"{avg_turns:.1f} turns/seg), "
                f"elapsed {elapsed:.1f}s"
            )
            print(f"    saved: {out}")
            # Sample
            first_cid = conv_ids[0]
            first_segs = result.conversations[first_cid][:3]
            for s in first_segs:
                print(
                    f"    [{first_cid} seg{s.segment_idx} tids={s.turn_ids[:3]}...]:"
                    f" {s.summary[:150]}"
                )


if __name__ == "__main__":
    main()
