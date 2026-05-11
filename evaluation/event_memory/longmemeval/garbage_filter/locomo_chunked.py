"""Run the cue-worthiness classifier on LoCoMo turns chunked by TextSegmenter.

The pipeline mirrors how the actual ingest works:
  1. Each LoCoMo turn becomes a TextBlock.
  2. TextSegmenter splits long turns into chunks (max_chunk_length=500).
  3. Each chunk is what would actually be embedded.
  4. Classify each chunk's text body and report rejection rates by length.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv

sys.path.insert(0, "/Users/eyu/edwinyyyu/mmcc/segment_store/packages/server/src")

from classifier import classify_many
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    ProducerContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")

LOCOMO_PATH = "/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/data/locomo10.json"
MAX_CHUNK_LEN = 500
N_PER_BUCKET = 80


async def collect_chunks() -> list[str]:
    data = json.load(open(LOCOMO_PATH))
    seg = TextSegmenter(max_chunk_length=MAX_CHUNK_LEN)
    chunks: list[str] = []
    for sample in data:
        conv = sample["conversation"]
        for k, sess in conv.items():
            if not (k.startswith("session_") and isinstance(sess, list)):
                continue
            for turn in sess:
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                ev = Event(
                    uuid=uuid4(),
                    timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    context=ProducerContext(producer=turn.get("speaker") or "User"),
                    blocks=[TextBlock(text=text)],
                    properties={},
                )
                for s in await seg.segment(ev):
                    chunks.append(s.block.text)
    return chunks


def stratify(chunks: list[str], seed: int = 0) -> dict[str, list[str]]:
    rng = random.Random(seed)
    buckets = {
        "tiny (≤15)": [c for c in chunks if 0 < len(c) <= 15],
        "short (16-40)": [c for c in chunks if 16 <= len(c) <= 40],
        "medium (41-120)": [c for c in chunks if 41 <= len(c) <= 120],
        "long (121-500)": [c for c in chunks if 121 <= len(c) <= 500],
    }
    return {k: rng.sample(v, min(N_PER_BUCKET, len(v))) for k, v in buckets.items()}


def main() -> None:
    chunks = asyncio.run(collect_chunks())
    print(f"LoCoMo chunks total: {len(chunks):,}")
    by_len: dict[str, int] = {}
    for c in chunks:
        L = len(c)
        b = (
            "≤15"
            if L <= 15
            else "16-40"
            if L <= 40
            else "41-120"
            if L <= 120
            else "121-500"
            if L <= 500
            else ">500"
        )
        by_len[b] = by_len.get(b, 0) + 1
    print(f"Chunk length histogram: {by_len}")

    sample = stratify(chunks)
    flat = [(b, t) for b, ts in sample.items() for t in ts]
    print(f"Classifying {len(flat)} stratified chunks…")

    import os

    prompt = os.environ.get("CLASSIFIER_PROMPT", "v1")
    print(f"prompt={prompt}")
    results = asyncio.run(
        classify_many(
            [t for _, t in flat],
            model="gpt-5-mini",
            prompt=prompt,
            reasoning_effort="low",
            concurrency=24,
        )
    )

    by_bucket: dict[str, list] = {b: [] for b in sample}
    for (b, _), r in zip(flat, results, strict=False):
        by_bucket[b].append(r)

    print()
    print(f"{'bucket':18s}  {'n':>4s}  {'reject%':>8s}")
    for b, rs in by_bucket.items():
        n = len(rs)
        n_rej = sum(1 for r in rs if r.label == "REJECT")
        print(f"{b:18s}  {n:>4d}  {n_rej / max(n, 1):>7.1%}")

    print()
    print(
        f"ALL LONG-text rejects ({len([r for b, rs in by_bucket.items() for r in rs if b == 'long (121-500)' and r.label == 'REJECT'])}):"
    )
    for r in [r for r in by_bucket["long (121-500)"] if r.label == "REJECT"]:
        print(f"  REJECT [{len(r.text)}]: {r.text!r}")
    print()
    print(
        f"ALL MEDIUM-text rejects ({len([r for r in by_bucket['medium (41-120)'] if r.label == 'REJECT'])}):"
    )
    for r in [r for r in by_bucket["medium (41-120)"] if r.label == "REJECT"][:30]:
        print(f"  REJECT [{len(r.text)}]: {r.text!r}")

    print()
    print("Sample TINY-text keeps (should be specific entities/facts):")
    tiny_keep = [r for r in by_bucket["tiny (≤15)"] if r.label == "KEEP"]
    for r in tiny_keep[:25]:
        print(f"  KEEP  [{len(r.text)}]: {r.text!r}")

    print()
    print("Sample TINY-text rejects (these are the wins):")
    tiny_rej = [r for r in by_bucket["tiny (≤15)"] if r.label == "REJECT"]
    for r in tiny_rej[:25]:
        print(f"  REJECT[{len(r.text)}]: {r.text!r}")


if __name__ == "__main__":
    main()
