"""Diagnose WHY the pointwise filter judged gold segments as NO.

For every question that lost >=1 gold to the filter (gold_dropped>0 in the
p100 low run), re-fetch its pool, isolate the GOLD segments, and run the
same pointwise_filter on each. Print question + gold text + verdict so we
can read whether the NO is a model error (gold clearly bears) or a genuine
out-of-context insufficiency (gold looks empty alone).

Usage:
  uv run python diag_false_no.py
"""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import create_async_engine

from embedder_factory import build_embedder
from locomo_models import load_locomo_dataset
from swiss_rerank_probe import (
    _FORMAT_OPTIONS, build_gold_timestamps, pointwise_filter,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory, EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore, SQLAlchemySegmentStoreParams,
)
from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
    SQLiteVecVectorStore, SQLiteVecVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)

SEG_DB = "swiss-textwhole-c2sub.sqlite"
VEC_DB = "swiss-textwhole-c2sub.vec.sqlite"
DATA = "../../data/locomo10_c2sub.json"
MODEL, EFFORT = "gpt-5-nano", "low"
POOL = 100


async def main() -> None:
    load_dotenv(
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/"
        "locomo/.env"
    )
    res = json.load(open("swiss-probe-filter-p100.json"))
    targets = {
        (r["category"], r["question"])
        for r in res["records"] if r.get("gold_dropped", 0) > 0
    }
    print(f"flagged questions (gold_dropped>0): {len(targets)}\n")

    locomo_data = load_locomo_dataset(DATA)
    seg_engine = create_async_engine(
        f"sqlite+aiosqlite:///{SEG_DB}", connect_args={"timeout": 30},
        pool_size=20, max_overflow=80,
    )
    segment_store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=seg_engine))
    await segment_store.startup()
    vec_engine = create_async_engine(
        f"sqlite+aiosqlite:///{VEC_DB}", connect_args={"timeout": 30},
        pool_size=20, max_overflow=80,
    )
    vector_store = SQLiteVecVectorStore(
        SQLiteVecVectorStoreParams(engine=vec_engine))
    await vector_store.startup()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedder = build_embedder("text-embedding-3-small", client)
    segmenter, deriver = TextSegmenter(), WholeTextDeriver()

    tally = {"HELPS": 0, "MAYBE": 0, "NO": 0}
    no_cases: list[tuple[str, str, str]] = []  # (cat, question, gold_text)

    for idx, item in enumerate(locomo_data):
        if "conversation" not in item:
            continue
        qs = [
            q for q in item["qa"]
            if (str(q["category"]), q["question"]) in targets
        ]
        if not qs:
            continue
        collection = await vector_store.open_collection(
            namespace="locomo", name=f"group_{idx}")
        partition = await segment_store.open_or_create_partition(
            f"group_{idx}", SegmentStorePartitionConfig())
        memory = EventMemory(EventMemoryParams(
            vector_store_collection=collection,
            segment_store_partition=partition,
            segmenter=segmenter, deriver=deriver,
            embedder=embedder, reranker=None,
        ))
        gold_ts = build_gold_timestamps(item["conversation"])

        for qa in qs:
            question = qa["question"]
            cat = str(qa["category"])
            gold_set = {gold_ts[e] for e in qa.get("evidence", [])
                        if e in gold_ts}
            qr = await memory.query(
                query=question, vector_search_limit=POOL, expand_context=0,
                format_options=_FORMAT_OPTIONS, bm25_fusion="none",
            )
            for ssc in qr.scored_segment_contexts:
                if not any(seg.timestamp in gold_set for seg in ssc.segments):
                    continue
                gold_text = EventMemory.string_from_segment_context(
                    ssc.segments, format_options=_FORMAT_OPTIONS)
                verdict, _, _ = await pointwise_filter(
                    client, MODEL, question, gold_text, EFFORT)
                tally[verdict] += 1
                if verdict == "NO":
                    no_cases.append((cat, question, gold_text))

    print("=== verdict tally on GOLD segments (flagged questions) ===")
    print(tally, "\n")
    print(f"=== {len(no_cases)} gold judged NO ===\n")
    for cat, q, txt in no_cases:
        print(f"[c{cat}] Q: {q}")
        print(f"   GOLD->NO: {txt!r}\n")


if __name__ == "__main__":
    asyncio.run(main())
