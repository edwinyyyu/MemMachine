"""Differential test: hotpath fast paths vs canonical paths, on live bench2
data. Asserts exact semantic equality of results.
"""

import asyncio
import hashlib
import random
import sys
from uuid import UUID, uuid4

import asyncpg
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

import os

import memmachine_server

print("memmachine_server from:", memmachine_server.__file__)
# Guard against benching the wrong build: when BENCH_EXPECT_PATH_SUBSTR is
# set, refuse to run unless the imported memmachine_server comes from a
# path containing it (e.g. the worktree name of the branch under test).
_expect = os.environ.get("BENCH_EXPECT_PATH_SUBSTR")
if _expect:
    assert _expect in memmachine_server.__file__, memmachine_server.__file__

from memmachine_server.common.data_types import SimilarityMetric  # noqa: E402
from memmachine_server.common.filter.filter_parser import Comparison  # noqa: E402
from memmachine_server.common.vector_store.data_types import (  # noqa: E402
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (  # noqa: E402
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (  # noqa: E402
    EventMemory,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (  # noqa: E402
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (  # noqa: E402
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.episodic_memory.long_term_memory import (  # noqa: E402
    EVENT_BACKEND_SYSTEM_FIELDS,
)

PK = hashlib.sha256(b"benchorg/isolated1").hexdigest()[:32]
COLL_NS = "long_term_memory"


async def test_segment_store():
    engine = create_async_engine(
        "postgresql+asyncpg://memmachine:memmachine@localhost:5442/memmachine",
        pool_size=4)
    store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await store.startup()
    part = await store.open_or_create_partition(PK, SegmentStorePartitionConfig())

    conn = await asyncpg.connect(
        "postgresql://memmachine:memmachine@localhost:5442/memmachine")
    rows = await conn.fetch(
        "SELECT uuid FROM segment_store_sg WHERE partition_key = $1 LIMIT 60", PK)
    await conn.close()
    uuids = [UUID(str(r["uuid"])) for r in rows]

    cases = {
        "50 seeds": uuids[:50],
        "1 seed": uuids[:1],
        "mixed with missing": uuids[:10] + [uuid4() for _ in range(5)],
        "all missing": [uuid4() for _ in range(3)],
    }
    for name, seeds in cases.items():
        fast = await part.get_segment_contexts(seeds)
        # force canonical
        orig = part._get_seed_segments_fast
        part._get_seed_segments_fast = lambda s: _none()
        canonical = await part.get_segment_contexts(seeds)
        part._get_seed_segments_fast = orig
        assert set(fast.keys()) == set(canonical.keys()), name
        for k in fast:
            assert len(fast[k]) == len(canonical[k]) == 1, name
            a, b = fast[k][0], canonical[k][0]
            assert a == b, (name, k, a, b)
        print(f"  segment store: {name}: OK ({len(fast)} seeds returned)")

    # windows path (untouched code) still works
    w = await part.get_segment_contexts(
        uuids[:5], max_backward_segments=2, max_forward_segments=4)
    assert len(w) == 5 and all(1 <= len(v) <= 7 for v in w.values())
    print(f"  segment store: windowed path intact: OK")
    await engine.dispose()


async def _none():
    return None


async def test_qdrant():
    client = AsyncQdrantClient(host="localhost", port=6343, prefer_grpc=False)
    vs = QdrantVectorStore(QdrantVectorStoreParams(client=client))
    cfg = VectorStoreCollectionConfig(
        vector_dimensions=1536, similarity_metric=SimilarityMetric.COSINE,
        indexed_properties_schema={
            **EventMemory.expected_vector_store_collection_schema(),
            **EVENT_BACKEND_SYSTEM_FIELDS,
        })
    coll = await vs.open_or_create_collection(namespace=COLL_NS, name=PK, config=cfg)

    rng = random.Random(11)
    vecs = [[rng.gauss(0, 1) for _ in range(1536)] for _ in range(3)]

    async def run_pair(name, **kw):
        fast = await coll.query(**kw)
        coll._fast_http = False  # force canonical
        canonical = await coll.query(**kw)
        coll._fast_http = None
        assert len(fast) == len(canonical), name
        for fr, cr in zip(fast, canonical):
            assert len(fr.matches) == len(cr.matches), name
            for fm, cm in zip(fr.matches, cr.matches):
                assert fm.record.uuid == cm.record.uuid, name
                assert abs(fm.score - cm.score) < 1e-9, name
                assert fm.record.properties == cm.record.properties, (
                    name, fm.record.properties, cm.record.properties)
                assert fm.record.vector == cm.record.vector, name
        n = sum(len(r.matches) for r in fast)
        print(f"  qdrant: {name}: OK ({n} matches)")

    await run_pair("single vector", query_vectors=[vecs[0]], limit=50)
    await run_pair("batch of 3", query_vectors=vecs, limit=10)
    await run_pair("score threshold", query_vectors=[vecs[0]], limit=50,
                   score_threshold=-1.0)
    await run_pair("with property filter", query_vectors=[vecs[0]], limit=20,
                   property_filter=Comparison(field="_producer_id", op="=",
                                              value="alice"))
    await run_pair("with vectors returned", query_vectors=[vecs[0]], limit=5,
                   return_vector=True)
    await run_pair("no properties", query_vectors=[vecs[0]], limit=5,
                   return_properties=False)
    await client.close()


async def main():
    print("== segment store ==")
    await test_segment_store()
    print("== qdrant collection ==")
    await test_qdrant()
    print("ALL DIFFERENTIAL TESTS PASSED")


asyncio.run(main())


async def test_embedder():
    from openai import AsyncOpenAI
    from memmachine_server.common.embedder.openai_embedder import (
        OpenAIEmbedder, OpenAIEmbedderParams)
    emb = OpenAIEmbedder(OpenAIEmbedderParams(
        client=AsyncOpenAI(api_key="sk-mock", base_url="http://127.0.0.1:8791/v1"),
        model="text-embedding-3-small", dimensions=1536, max_input_length=2048))
    queries = ["what did alice say?", "shard rebalancing", "x"]
    fast = await emb.search_embed(queries)
    assert isinstance(emb._fast_http, object) and emb._fast_http not in (None, False), \
        "fast path did not engage"
    emb._fast_http = False
    canonical = await emb.search_embed(queries)
    assert len(fast) == len(canonical) == 3
    for f, c in zip(fast, canonical):
        assert f == c, (f[:3], c[:3])
    print(f"  embedder: fast == canonical for {len(queries)} queries: OK "
          f"(dim {len(fast[0])})")

asyncio.run(test_embedder())
print("EMBEDDER DIFF TEST PASSED")
