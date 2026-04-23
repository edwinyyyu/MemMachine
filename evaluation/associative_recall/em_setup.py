"""EventMemory ingestion for LoCoMo-30 architecture re-evaluation.

Ingests the three LoCoMo conversations (26, 30, 41) from segments_extended.npz
into a fresh EventMemory per conversation, using:
  - max_text_chunk_length = 500
  - derive_sentences = False
  - speaker baked in via MessageContext (Caroline/Melanie, Jon/Gina, John/Maria)

Partition / collection naming (unique prefix to avoid collision with existing
longmemeval/locomo data): arc_em_locomo30_v1_<conv_id>

Run once:
    uv run python evaluation/associative_recall/em_setup.py
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import openai
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Content,
    Event,
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

ROOT = Path(__file__).resolve().parents[2]
# Load associative_recall's local .env first (has the right SQL_URL / QDRANT host),
# then fall back to the repo root .env so nothing else we read is lost.
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
# Qdrant collection names must match [a-z0-9_]+ and be <=32 bytes, so the
# short prefix "arc_em_lc30_v1" is used in the collection layer while the
# conceptual prefix "arc_em_locomo30_v1" is recorded in the results file.
COLLECTION_PREFIX = "arc_em_lc30_v1"
LOGICAL_PREFIX = "arc_em_locomo30_v1"
NAMESPACE = "arc_em_locomo30"

LOCOMO_CONV_IDS = ["locomo_conv-26", "locomo_conv-30", "locomo_conv-41"]


def _conv_short(conv_id: str) -> str:
    # locomo_conv-26 -> 26
    return conv_id.rsplit("-", 1)[-1]


def load_speaker_map() -> dict[str, dict[str, str]]:
    with open(RESULTS_DIR / "conversation_two_speakers.json") as f:
        return json.load(f)["speakers"]


def load_conversation_segments(
    npz_path: Path,
) -> dict[str, list[tuple[int, str, str]]]:
    """Return dict conv_id -> sorted list of (turn_id, role, text)."""
    d = np.load(npz_path, allow_pickle=True)
    out: dict[str, list[tuple[int, str, str]]] = {}
    for i in range(len(d["texts"])):
        cid = str(d["conversation_ids"][i])
        if cid not in LOCOMO_CONV_IDS:
            continue
        out.setdefault(cid, []).append(
            (int(d["turn_ids"][i]), str(d["roles"][i]), str(d["texts"][i]))
        )
    for cid in out:
        out[cid].sort(key=lambda t: t[0])
    return out


def _event_timestamp(base: datetime, turn_id: int) -> datetime:
    # Synthesize monotonic timestamps so EventMemory can expand_context
    # by temporal neighborhood; 60-second spacing is arbitrary.
    return base + timedelta(seconds=60 * turn_id)


async def ingest_conversation(
    conv_id: str,
    segments: list[tuple[int, str, str]],
    speakers: dict[str, str],
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
) -> dict:
    """Ingest all turns of a conversation and close resources after."""
    collection_name = f"{COLLECTION_PREFIX}_{_conv_short(conv_id)}"
    partition_key = collection_name  # same identifier for simplicity

    # Delete any stale data from a previous partial run.
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
            reranker=None,  # score by cosine
            derive_sentences=False,
            max_text_chunk_length=500,
        )
    )

    user_name = speakers.get("user") or "User"
    asst_name = speakers.get("assistant") or "Assistant"

    base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    events: list[Event] = []
    for turn_id, role, text in segments:
        source_name = user_name if role == "user" else asst_name
        events.append(
            Event(
                uuid=uuid4(),
                timestamp=_event_timestamp(base_ts, turn_id),
                body=Content(
                    context=MessageContext(source=source_name),
                    items=[Text(text=text.strip())],
                ),
                properties={
                    "arc_conversation_id": conv_id,
                    "turn_id": turn_id,
                    "role": role,
                },
            )
        )

    t0 = time.monotonic()
    await memory.encode_events(events)
    ingest_time = time.monotonic() - t0

    await segment_store.close_partition(partition)
    await vector_store.close_collection(collection=collection)

    return {
        "conversation_id": conv_id,
        "collection_name": collection_name,
        "logical_collection_name": f"{LOGICAL_PREFIX}_{conv_id}",
        "partition_key": partition_key,
        "namespace": NAMESPACE,
        "n_events": len(events),
        "ingest_time_s": round(ingest_time, 2),
        "user_name": user_name,
        "assistant_name": asst_name,
    }


async def main() -> None:
    speakers_map = load_speaker_map()
    conv_segments = load_conversation_segments(DATA_DIR / "segments_extended.npz")

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    # Prefer Postgres from SQL_URL. If not reachable (auth / missing role),
    # fall back to a file-backed SQLite under the results dir — the framework
    # supports SQLite so long as we use a file path and a non-static pool.
    sql_url = os.getenv("SQL_URL") or f"sqlite+aiosqlite:///{RESULTS_DIR / 'eventmemory.sqlite3'}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        engine = create_async_engine(sql_url, pool_size=50, max_overflow=50)
        # Quick probe — validate auth before proceeding so the fallback
        # engages early rather than after the first partition op.
        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
    except Exception as exc:  # noqa: BLE001  (broad is intentional here)
        print(f"[em_setup] SQL_URL {sql_url!r} not reachable: {exc!r}")
        sqlite_path = RESULTS_DIR / "eventmemory.sqlite3"
        sql_url = f"sqlite+aiosqlite:///{sqlite_path}"
        print(f"[em_setup] falling back to SQLite at {sqlite_path}")
        engine = create_async_engine(sql_url)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=8192,
        )
    )

    records = []
    try:
        semaphore = asyncio.Semaphore(3)
        tasks = [
            async_with(
                semaphore,
                ingest_conversation(
                    conv_id,
                    conv_segments[conv_id],
                    speakers_map[conv_id],
                    vector_store,
                    segment_store,
                    embedder,
                ),
            )
            for conv_id in LOCOMO_CONV_IDS
        ]
        records = await asyncio.gather(*tasks)
    finally:
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eventmemory_collections.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "namespace": NAMESPACE,
                "prefix": COLLECTION_PREFIX,
                "logical_prefix": LOGICAL_PREFIX,
                "sql_url": sql_url,
                "max_text_chunk_length": 500,
                "derive_sentences": False,
                "conversations": records,
            },
            f,
            indent=2,
        )

    for r in records:
        print(
            f"[ingested] {r['conversation_id']}: "
            f"{r['n_events']} events in {r['ingest_time_s']}s "
            f"(user={r['user_name']}, asst={r['assistant_name']}) "
            f"-> {r['collection_name']}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
