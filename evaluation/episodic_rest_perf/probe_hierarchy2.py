"""Hierarchical CPU breakdown of the ROUND-5 serving path.

The serving path is now: ASGI fast route -> spec validation -> embed ->
vector query -> payload walk (uid/score dedup) -> episode fetch ->
response dump. Each stage measured in isolation with process_time.
"""

import asyncio
import hashlib
import json
import random
import time

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

from memmachine_common.api.spec import SearchMemoriesSpec  # noqa: E402

from memmachine_server.common import fast_json  # noqa: E402
from memmachine_server.common.data_types import SimilarityMetric  # noqa: E402
from memmachine_server.common.embedder.openai_embedder import (  # noqa: E402
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.episode_store.episode_sqlalchemy_store import (  # noqa: E402
    SqlAlchemyEpisodeStore,
)
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
from memmachine_server.episodic_memory.long_term_memory import (  # noqa: E402
    EVENT_BACKEND_SYSTEM_FIELDS,
)

PK = hashlib.sha256(b"benchorg/isolated1").hexdigest()[:32]
N = 400
BODY = json.dumps({
    "org_id": "benchorg", "project_id": "isolated1", "top_k": 10,
    "query": "what did alice say about shard rebalancing?",
    "types": ["episodic"],
}).encode()


async def cpu(fn, n=N, warmup=30):
    for _ in range(warmup):
        await fn()
    t0 = time.process_time()
    for _ in range(n):
        await fn()
    return (time.process_time() - t0) * 1000 / n


async def main():
    emb = OpenAIEmbedder(OpenAIEmbedderParams(
        client=AsyncOpenAI(api_key="sk-mock", base_url="http://127.0.0.1:8791/v1"),
        model="text-embedding-3-small", dimensions=1536, max_input_length=2048))
    qc = AsyncQdrantClient(host="localhost", port=6343, prefer_grpc=False)
    vs = QdrantVectorStore(QdrantVectorStoreParams(client=qc))
    cfg = VectorStoreCollectionConfig(
        vector_dimensions=1536, similarity_metric=SimilarityMetric.COSINE,
        indexed_properties_schema={
            **EventMemory.expected_vector_store_collection_schema(),
            **EVENT_BACKEND_SYSTEM_FIELDS,
        })
    coll = await vs.open_or_create_collection(
        namespace="long_term_memory", name=PK, config=cfg)
    engine = create_async_engine(
        "postgresql+asyncpg://memmachine:memmachine@localhost:5442/memmachine",
        pool_size=4)
    episode_store = SqlAlchemyEpisodeStore(engine)

    vec = (await emb.search_embed(["what did alice say about shard rebalancing?"]))[0]
    [qr] = await coll.query(query_vectors=[vec], limit=50,
                            return_vector=False, return_properties=True)

    def payload_walk():
        ordered, scores = [], {}
        for m in qr.matches:
            uid = m.record.properties.get("_episode_uid")
            if uid is None or uid in scores:
                continue
            scores[uid] = m.score
            ordered.append(uid)
            if len(ordered) >= 10:
                break
        return ordered

    uids = payload_walk()
    episodes = await episode_store.get_episodes(uids)
    print(f"uids: {len(uids)}, episodes fetched: {len(episodes)}")

    async def spec_validate():
        SearchMemoriesSpec.model_validate_json(BODY)

    async def embed_one():
        await emb.search_embed(["what did alice say about shard rebalancing?"])

    async def vector_query():
        await coll.query(query_vectors=[vec], limit=50,
                         return_vector=False, return_properties=True)

    async def walk():
        payload_walk()

    async def episode_fetch():
        await episode_store.get_episodes(uids)

    # response dump: pydantic dump of episodes + orjson, approximating the
    # service path (episodes dominate the response payload)
    async def response_dump():
        content = {"episodes": [e.model_dump(mode="json", exclude_none=True)
                                for e in episodes]}
        fast_json.dumps({"status": 0, "content": content})

    sv = await cpu(spec_validate)
    e = await cpu(embed_one)
    v = await cpu(vector_query)
    w = await cpu(walk)
    ef = await cpu(episode_fetch)
    rd = await cpu(response_dump)

    print(f"spec validation (request side):        {sv:5.3f} core-ms")
    print(f"embed (raw pool, base64):              {e:5.3f} core-ms")
    print(f"vector query (raw pool, vsl=50):       {v:5.3f} core-ms")
    print(f"payload walk (uid/score dedup, top10): {w:5.3f} core-ms")
    print(f"episode fetch (raw asyncpg, 10 rows):  {ef:5.3f} core-ms")
    print(f"response dump (10 episodes -> orjson): {rd:5.3f} core-ms")
    print(f"sum of stages:                         {sv+e+v+w+ef+rd:5.3f} core-ms")

    await qc.close(); await engine.dispose()


asyncio.run(main())
