"""Decompose DeclarativeMemory query latency on the decl_50k session.

Times each stage of the search path in isolation against live Neo4j:
  A. ANN Cypher returning FULL nodes (what the code does: includes the
     768-float embedding property per node)
  B. same ANN Cypher returning uid only (isolates payload transfer)
  C. one derivative->episode traversal query
  D. 50 concurrent traversal queries (what one search fans out)
  E. full DeclarativeMemory.search_scored for reference
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import FakeEmbedder, make_queries
from neo4j import AsyncGraphDatabase

from memmachine_server.common.reranker.identity_reranker import IdentityReranker
from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine_server.episodic_memory.declarative_memory.declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryParams,
)

from common import NEO4J_AUTH as AUTH, NEO4J_URI as URI  # noqa: E402
REPS = 30


async def timeit(fn):
    xs = []
    for _ in range(REPS):
        s = time.perf_counter()
        await fn()
        xs.append((time.perf_counter() - s) * 1000)
    xs.sort()
    return statistics.fmean(xs), xs[len(xs) // 2]


async def main():
    driver = AsyncGraphDatabase.driver(URI, auth=AUTH)
    emb = FakeEmbedder()
    queries = make_queries(REPS + 5, seed=999)
    qvecs = [emb._embed_one(q) for q in queries]

    async with driver.session() as s:
        res = await s.run(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties "
            "WHERE type = 'VECTOR' RETURN name, labelsOrTypes, properties"
        )
        vecs = [r async for r in res]
        target = next(
            r for r in vecs if "decl_u5f_50k" in r["labelsOrTypes"][0]
        )
        index_name = target["name"]
        deriv_label = target["labelsOrTypes"][0]
        emb_prop = target["properties"][0]
        rel = deriv_label.replace("Derivative", "DERIVED_u5f_FROM")
        epi_label = deriv_label.replace("Derivative", "Episode")
        res = await s.run(
            f"MATCH (n:{deriv_label}) RETURN n.uid AS uid LIMIT 60"
        )
        uids = [r["uid"] async for r in res]
    print(f"index={index_name} label={deriv_label} emb_prop={emb_prop}")

    i = 0

    async def ann_full():
        nonlocal i
        async with driver.session() as s:
            res = await s.run(
                f"CALL db.index.vector.queryNodes($idx, 50, $v) "
                f"YIELD node AS n, score AS similarity WHERE TRUE "
                f"RETURN n ORDER BY similarity DESC LIMIT 50",
                idx=index_name, v=qvecs[i % REPS],
            )
            _ = [r async for r in res]
        i += 1

    async def ann_uid_only():
        nonlocal i
        async with driver.session() as s:
            res = await s.run(
                f"CALL db.index.vector.queryNodes($idx, 50, $v) "
                f"YIELD node AS n, score AS similarity WHERE TRUE "
                f"RETURN n.uid ORDER BY similarity DESC LIMIT 50",
                idx=index_name, v=qvecs[i % REPS],
            )
            _ = [r async for r in res]
        i += 1

    async def one_traversal(uid=None):
        async with driver.session() as s:
            res = await s.run(
                f"MATCH (m:{deriv_label} {{uid: $u}})-[r:{rel}]->"
                f"(n:{epi_label}) WHERE TRUE AND TRUE RETURN DISTINCT n",
                u=uid or uids[0],
            )
            _ = [r async for r in res]

    async def fifty_traversals():
        await asyncio.gather(*(one_traversal(u) for u in uids[:50]))

    store = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=driver,
            range_index_hierarchies=[["uid"], ["timestamp", "uid"]],
            range_index_creation_threshold=1,
            vector_index_creation_threshold=1,
        )
    )
    mem = DeclarativeMemory(
        DeclarativeMemoryParams(
            session_id="decl_50k", vector_graph_store=store,
            embedder=emb, reranker=IdentityReranker(),
        )
    )

    async def full_search():
        nonlocal i
        await mem.search_scored(queries[i % REPS], max_num_episodes=10)
        i += 1

    await full_search()  # warm index-state cache off the timed path

    for name, fn in [
        ("A ann RETURN full nodes (incl 768-float embedding)", ann_full),
        ("B ann RETURN uid only", ann_uid_only),
        ("C one traversal query", one_traversal),
        ("D 50 concurrent traversal queries (wall)", fifty_traversals),
        ("E full DeclarativeMemory.search_scored", full_search),
    ]:
        mean, p50 = await timeit(fn)
        print(f"{name:<52} mean {mean:7.2f} ms  p50 {p50:7.2f} ms")

    await driver.close()


asyncio.run(main())
