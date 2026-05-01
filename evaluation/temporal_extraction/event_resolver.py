"""F5 — Event resolver for Allen-relation retrieval.

When an AllenExpression has an anchor with ``kind="event"`` (e.g.
"the meeting", "my wedding"), we need to map the text span to an
absolute time bracket. The strategy is:

1. Iterate over every indexed doc's TimeExpressions and, for each doc,
   extract a list of (event_span, absolute_time) pairs using the
   existing :mod:`event_binding` extractor (reused as a building block
   — this adds one cheap gpt-5-mini call per doc).
2. Embed every event_span with ``text-embedding-3-small``.
3. Build a corpus-wide event index: ``[(event_span, embedding, te)]``.
4. At query time, given an anchor span, embed it; find the nearest
   event_span across the whole corpus (cosine threshold 0.7).  If a
   match is found, use its TimeExpression as the anchor's resolved
   interval.

The resolver is also given access to the *query's own* event-tuple list
so it can prefer an anchor defined in the query when one exists. This
matches the task brief's "search both the query's own event-tuples
AND the corpus-wide event-tuples".

Cheap cache: results (event_span, time, embedding) cached under
``cache/allen/event_index.json``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from advanced_common import Embedder, LLMCaller
from event_binding import extract_pairs, resolve_time
from schema import (
    FuzzyInstant,
    TimeExpression,
    parse_iso,
    time_expression_from_dict,
    time_expression_to_dict,
)

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "allen"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
EVENT_INDEX_FILE = CACHE_DIR / "event_index.json"


@dataclass
class EventEntry:
    doc_id: str
    span: str
    time: TimeExpression
    embedding: np.ndarray  # shape (1536,)


def _te_from_resolve(raw: dict[str, Any], ref_time: datetime) -> TimeExpression | None:
    """Convert a resolve_time() dict into a TimeExpression (instant)."""
    try:
        instant = FuzzyInstant(
            earliest=parse_iso(raw["earliest"]),
            latest=parse_iso(raw["latest"]),
            best=parse_iso(raw.get("best")) if raw.get("best") else None,
            granularity=raw.get("granularity", "day"),
        )
    except Exception:
        return None
    return TimeExpression(
        kind="instant",
        surface=raw.get("surface", ""),
        reference_time=ref_time,
        instant=instant,
    )


class EventResolver:
    """Corpus-wide event -> time index with embedding-based lookup."""

    def __init__(self) -> None:
        self.entries: list[EventEntry] = []
        self.llm = LLMCaller()
        self.emb = Embedder()
        self._cache: dict[str, Any] = {}
        if EVENT_INDEX_FILE.exists():
            try:
                with EVENT_INDEX_FILE.open() as f:
                    self._cache = json.load(f)
            except json.JSONDecodeError:
                self._cache = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    async def index_docs(
        self,
        docs: list[dict[str, Any]],
        *,
        text_key: str = "text",
        id_key: str = "doc_id",
        ref_key: str = "ref_time",
    ) -> None:
        """Extract (event, time) pairs from every doc and build the index.

        docs: list of {id, text, ref_time}.
        """
        tasks = [self._index_one_doc(d, text_key, id_key, ref_key) for d in docs]
        await asyncio.gather(*tasks)
        # Also embed every span we didn't already embed
        self.llm.save()
        self.emb.save()
        self._save_cache()

    async def _index_one_doc(
        self,
        d: dict[str, Any],
        text_key: str,
        id_key: str,
        ref_key: str,
    ) -> None:
        doc_id = d[id_key]
        text = d[text_key]
        ref_time = parse_iso(d[ref_key])
        cache_key = f"doc:{doc_id}:{ref_time.isoformat()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            # Populate entries from cache.
            for item in cached:
                te = time_expression_from_dict(item["time"])
                emb = np.array(item["embedding"], dtype=np.float32)
                self.entries.append(
                    EventEntry(doc_id=doc_id, span=item["span"], time=te, embedding=emb)
                )
            return

        pairs = await extract_pairs(self.llm, text, ref_time)
        built: list[dict[str, Any]] = []
        for p in pairs:
            span = (p.get("event_span") or "").strip()
            time_surface = (p.get("time_surface") or "").strip()
            if not span or not time_surface:
                continue
            resolved = await resolve_time(self.llm, time_surface, ref_time, text)
            if resolved is None:
                continue
            resolved = {**resolved, "surface": time_surface}
            te = _te_from_resolve(resolved, ref_time)
            if te is None:
                continue
            emb = await self.emb.embed(span)
            entry = EventEntry(doc_id=doc_id, span=span, time=te, embedding=emb)
            self.entries.append(entry)
            built.append(
                {
                    "span": span,
                    "time": time_expression_to_dict(te),
                    "embedding": emb.tolist(),
                }
            )
        self._cache[cache_key] = built

    def _save_cache(self) -> None:
        tmp = EVENT_INDEX_FILE.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(self._cache, f)
        tmp.replace(EVENT_INDEX_FILE)

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------
    async def resolve(
        self,
        anchor_span: str,
        *,
        query_doc_id: str | None = None,
        query_local_entries: list[EventEntry] | None = None,
        threshold: float = 0.7,
    ) -> EventEntry | None:
        """Find the best match for an anchor span.

        Search order:
        1. Query-local entries (same doc as the query, if any).
        2. Corpus-wide entries.
        """
        if not anchor_span:
            return None
        anchor_emb = await self.emb.embed(anchor_span)

        def _cos(a: np.ndarray, b: np.ndarray) -> float:
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
            return float(np.dot(a, b) / denom)

        best: tuple[float, EventEntry | None] = (-1.0, None)
        candidates: list[EventEntry] = []
        if query_local_entries:
            candidates.extend(query_local_entries)
        candidates.extend(self.entries)
        for e in candidates:
            sim = _cos(anchor_emb, e.embedding)
            if sim > best[0]:
                best = (sim, e)
        self.emb.save()
        if best[0] >= threshold:
            return best[1]
        return None

    def cost_usd(self) -> float:
        return self.llm.cost_usd()


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
async def build_resolver(docs: list[dict[str, Any]]) -> EventResolver:
    res = EventResolver()
    await res.index_docs(docs)
    return res
