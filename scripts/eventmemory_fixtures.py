"""Generate tests/fixtures/eventmemory.json with MemMachine's own EventMemory functions.

Run with a Python that has memmachine_server installed (e.g. a MemMachine worktree's venv):
  /path/to/MemMachine/.venv/bin/python scripts/eventmemory_fixtures.py
The fixture holds random contexts and MemMachine's outputs for them; tests/test_eventmemory.py
checks rbench.eventmemory's copies against it.
"""

import datetime
import json
import random
import sys
from pathlib import Path
from uuid import UUID

from memmachine_server.episodic_memory.event_memory.data_types import (
    FormatOptions, NullContext, ProducerContext, QueryResult, ScoredSegmentContext, Segment, TextBlock)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import WholeTextDeriver
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import TextSegmenter

rng = random.Random(0)
WORDS = "the a memory search vector rerank chunk window user assistant date token query, said. ok? yes!".split()


def text(n):
    return " ".join(rng.choice(WORDS) for _ in range(n)) + rng.choice(["", "\n", "\n\n", ". Next line\nmore"])


def segments(n_events=12):
    out, i = [], 0
    base = datetime.datetime(2023, 3, 20, 17, 57, tzinfo=datetime.UTC)
    for e in range(n_events):
        producer = rng.choice(["User", "Assistant", None])
        for offset in range(rng.randint(1, 3)):
            out.append(dict(id=i, event=e, index=0, offset=offset, producer=producer, text=text(rng.randint(3, 30)),
                            timestamp=(base + datetime.timedelta(days=e // 4, seconds=e)).isoformat()))
            i += 1
    return out


def to_mm(s):
    return Segment(uuid=UUID(int=s["id"] + 1), event_uuid=UUID(int=10_000 + s["event"]), index=s["index"],
                   offset=s["offset"], timestamp=datetime.datetime.fromisoformat(s["timestamp"]),
                   context=ProducerContext(producer=s["producer"]) if s["producer"] else NullContext(),
                   block=TextBlock(text=s["text"]))


cases = []
for _ in range(40):
    segs = segments()
    contexts = []
    for _ in range(rng.randint(1, 6)):
        seed = rng.randrange(len(segs))
        lo, hi = max(0, seed - rng.randint(0, 2)), min(len(segs), seed + rng.randint(0, 3) + 1)
        contexts.append({"seed": seed, "ids": list(range(lo, hi))})
    mm = {s["id"]: to_mm(s) for s in segs}
    result = QueryResult(scored_segment_contexts=[
        ScoredSegmentContext(score=1.0 - k / 10, seed_segment_uuid=mm[c["seed"]].uuid, segments=[mm[i] for i in c["ids"]])
        for k, c in enumerate(contexts)])
    limit = rng.randint(1, 12)
    cases.append({
        "segments": segs, "contexts": contexts, "limit": limit,
        "rendered_one": [EventMemory.string_from_segment_context([mm[i] for i in c["ids"]],
                                                                 format_options=FormatOptions(time_style="short"))
                         for c in contexts],
        "rendered_all": EventMemory.string_from_query_result(result, format_options=FormatOptions(time_style="short")),
        "admitted": sorted(s.uuid.int - 1 for s in EventMemory.build_query_result_context(result, limit)),
        "derivative": [__import__("asyncio").run(WholeTextDeriver().derive(mm[s["id"]]))[0].block.text for s in segs[:5]],
    })
long_text = " ".join(text(40) for _ in range(30))
segmenter_cases = [{"text": long_text, "max": m} for m in (250, 500, 1000)]
import asyncio  # noqa: E402

from memmachine_server.episodic_memory.event_memory.data_types import Event  # noqa: E402

for c in segmenter_cases:
    ev = Event(uuid=UUID(int=1), timestamp=datetime.datetime(2023, 1, 1, tzinfo=datetime.UTC), blocks=[TextBlock(text=c["text"])])
    c["chunks"] = [s.block.text for s in asyncio.run(TextSegmenter(c["max"]).segment(ev))]
Path(__file__).resolve().parents[1].joinpath("tests/fixtures/eventmemory.json").write_text(
    json.dumps({"cases": cases, "segmenter": segmenter_cases}, indent=0))
print("wrote", len(cases), "cases", file=sys.stderr)
