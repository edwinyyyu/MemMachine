"""CachedSegmenter: replays segments from a JSON cache, skipping the LLM.

Use this to iterate on the DERIVER without re-paying the segmenter LLM
cost. Build a cache once by dumping segments from a baseline ingest
DB, then point new ingest runs at the cache.

Cache format
------------

A JSON array of segment records. Each record has:

    {
      "partition_key": "group_4",
      "event_uuid": "<hex>",           # original ingest event UUID
      "index": int,                    # segment.index
      "offset": int,                   # segment.offset
      "timestamp": "ISO",              # segment.timestamp
      "timestamp_timezone_offset": int,
      "context_blob": <base64 of bytes>,   # JSON-encoded Context, blob
      "block_blob": <base64 of bytes>,     # JSON-encoded Block, blob
      "properties": <JSON string>,
    }

Lookup
------

Cached segments are indexed by (locomo_session_id, dia_id) from
``properties``. ``CachedSegmenter.segment(event)`` reads the event's
properties and returns the cached segments for that (session, dia_id),
reconstructing Segment objects with FRESH UUIDs (so re-ingestion
doesn't collide with the source DB's UUIDs).

Limitations
-----------

- Properties must include locomo_session_id + dia_id (the LoCoMo
  convention used by locomo_ingest.py).
- The Segment's context and block are reconstructed by deserializing
  the stored blob via the same SQLAlchemySegmentStore codec. This
  preserves SurroundingEventsContext / RewriteContext exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import override
from uuid import uuid4
import base64

from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    Segment,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    decode_block as _decode_block_fn,
    decode_context as _decode_context_fn,
)


class CachedSegmenter(Segmenter):
    def __init__(self, *, cache_path: str | Path) -> None:
        with open(cache_path) as f:
            records = json.load(f)
        # Index by (session_id, dia_id)
        self._by_key: dict[tuple[str, str], list[dict]] = {}
        for r in records:
            try:
                props = json.loads(r["properties"]) if isinstance(r["properties"], str) else r["properties"]
            except Exception:
                continue
            sid = props.get("locomo_session_id", {}).get("v") if isinstance(props.get("locomo_session_id"), dict) else props.get("locomo_session_id")
            dia = props.get("dia_id", {}).get("v") if isinstance(props.get("dia_id"), dict) else props.get("dia_id")
            if sid is None or dia is None:
                continue
            self._by_key.setdefault((sid, dia), []).append(r)
        # Sort each bucket by (index, offset)
        for k in self._by_key:
            self._by_key[k].sort(key=lambda r: (r["index"], r["offset"]))

    @override
    async def segment(self, event: Event) -> list[Segment]:
        props = event.properties or {}
        sid = props.get("locomo_session_id")
        dia = props.get("dia_id")
        if sid is None or dia is None:
            return []
        key = (sid, dia)
        cached = self._by_key.get(key, [])
        out: list[Segment] = []
        for r in cached:
            ctx_blob = base64.b64decode(r["context_blob"])
            blk_blob = base64.b64decode(r["block_blob"])
            ctx = _decode_context_fn(json.loads(ctx_blob.decode("utf-8")))
            blk = _decode_block_fn(json.loads(blk_blob.decode("utf-8")))
            out.append(
                Segment(
                    uuid=uuid4(),  # fresh UUID — no collision with the source DB
                    event_uuid=event.uuid,  # link to the fresh event
                    index=r["index"],
                    offset=r["offset"],
                    timestamp=event.timestamp,  # use fresh event's timestamp
                    block=blk,
                    context=ctx,
                    properties=event.properties,
                )
            )
        return out
