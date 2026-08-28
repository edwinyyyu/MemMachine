"""Prove the fast paths actually serve: sabotage every canonical path and
assert results still come back correct."""
import asyncio, hashlib, random
from uuid import UUID
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine
import os

import memmachine_server
# Guard against benching the wrong build: when BENCH_EXPECT_PATH_SUBSTR is
# set, refuse to run unless the imported memmachine_server comes from a
# path containing it (e.g. the worktree name of the branch under test).
_expect = os.environ.get("BENCH_EXPECT_PATH_SUBSTR")
if _expect:
    assert _expect in memmachine_server.__file__, memmachine_server.__file__
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder.openai_embedder import OpenAIEmbedder, OpenAIEmbedderParams
from memmachine_server.common.vector_store.data_types import VectorStoreCollectionConfig
from memmachine_server.common.vector_store.qdrant_vector_store import QdrantVectorStore, QdrantVectorStoreParams
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import SegmentStorePartitionConfig
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import SQLAlchemySegmentStore, SQLAlchemySegmentStoreParams
from memmachine_server.episodic_memory.long_term_memory import EVENT_BACKEND_SYSTEM_FIELDS

PK = hashlib.sha256(b"benchorg/isolated1").hexdigest()[:32]

def boom(*a, **k):
    raise RuntimeError("canonical path reached!")

async def aboom(*a, **k):
    raise RuntimeError("canonical path reached!")

async def main():
    # embedder
    emb = OpenAIEmbedder(OpenAIEmbedderParams(
        client=AsyncOpenAI(api_key="sk-mock", base_url="http://127.0.0.1:8791/v1"),
        model="text-embedding-3-small", dimensions=1536, max_input_length=2048))
    v0 = (await emb.search_embed(["probe"]))[0]
    emb._client.embeddings.create = aboom  # sabotage SDK
    v1 = (await emb.search_embed(["probe"]))[0]
    assert v0 == v1 and len(v1) == 1536
    print("embedder: fast path serves under SDK sabotage: OK")

    # qdrant
    qc = AsyncQdrantClient(host="localhost", port=6343, prefer_grpc=False)
    vs = QdrantVectorStore(QdrantVectorStoreParams(client=qc))
    cfg = VectorStoreCollectionConfig(
        vector_dimensions=1536, similarity_metric=SimilarityMetric.COSINE,
        indexed_properties_schema={**EventMemory.expected_vector_store_collection_schema(),
                                   **EVENT_BACKEND_SYSTEM_FIELDS})
    coll = await vs.open_or_create_collection(namespace="long_term_memory", name=PK, config=cfg)
    rng = random.Random(5)
    vec = [rng.gauss(0, 1) for _ in range(1536)]
    [r0] = await coll.query(query_vectors=[vec], limit=50)
    qc.query_batch_points = aboom  # sabotage canonical
    [r1] = await coll.query(query_vectors=[vec], limit=50)
    # qdrant filtered ANN may under-fill the limit at low tenant selectivity;
    # the invariant is fast/canonical PARITY, not the absolute count.
    assert len(r1.matches) == len(r0.matches) > 0
    assert [m.record.uuid for m in r0.matches] == [m.record.uuid for m in r1.matches]
    print(f"qdrant: fast path serves under client sabotage: OK ({len(r1.matches)} matches)")

    # segment store
    engine = create_async_engine("postgresql+asyncpg://memmachine:memmachine@localhost:5442/memmachine", pool_size=4)
    store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await store.startup()
    part = await store.open_or_create_partition(PK, SegmentStorePartitionConfig())
    seeds = [UUID(str(m.record.properties[k])) for m in r0.matches
             for k in m.record.properties if "segment" in k and "uuid" in k][:50]
    if not seeds:
        import asyncpg
        c = await asyncpg.connect("postgresql://memmachine:memmachine@localhost:5442/memmachine")
        seeds = [UUID(str(x["uuid"])) for x in await c.fetch(
            "SELECT uuid FROM segment_store_sg WHERE partition_key=$1 LIMIT 50", PK)]
        await c.close()
    g0 = await part.get_segment_contexts(seeds)
    part._create_session = boom  # sabotage canonical session path
    g1 = await part.get_segment_contexts(seeds)
    assert set(g0) == set(g1) and all(g0[k] == g1[k] for k in g0) and len(g1) > 0
    print(f"segment store: fast path serves under session sabotage: OK ({len(g1)} seeds)")
    await qc.close(); await engine.dispose()

asyncio.run(main())
print("ALL SABOTAGE TESTS PASSED")
