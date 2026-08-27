"""What if DeclarativeMemory's query architecture were fixed in place?

Measures a hypothetical repaired Neo4j query path on the decl_50k session:
  A. ANN returning derivative props WITHOUT the embedding
     (apoc.map.removeKeys; the ABC-preserving payload fix)
  B. FUSED single statement: ANN + derivative->episode traversal + episode
     projection, one round trip, no client-side join or unification
     (episodes carry no embedding, so the payload problem disappears)
  C. the fused statement under 16 concurrent clients, 300 queries
Compare against results-summary.md: decl search_scored 42.6 ms p50 c1 /
754.6 ms p50 c16; EventMemory.query 10.3 ms c1 / 70.0 ms c16.
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import FakeEmbedder, make_queries, run_workers, split_chunks
from neo4j import AsyncGraphDatabase

from common import NEO4J_AUTH as AUTH, NEO4J_URI as URI  # noqa: E402
REPS = 30


async def timeit(fn, reps=REPS):
    xs = []
    for _ in range(reps):
        s = time.perf_counter()
        await fn()
        xs.append((time.perf_counter() - s) * 1000)
    xs.sort()
    return statistics.fmean(xs), xs[len(xs) // 2]


async def main():
    driver = AsyncGraphDatabase.driver(URI, auth=AUTH, max_connection_pool_size=200)
    emb = FakeEmbedder()
    queries = make_queries(400, seed=555)
    qvecs = [emb._embed_one(q) for q in queries]

    async with driver.session() as s:
        res = await s.run(
            "SHOW INDEXES YIELD name, type, labelsOrTypes "
            "WHERE type = 'VECTOR' RETURN name, labelsOrTypes"
        )
        rows = [r async for r in res]
        target = next(r for r in rows if "decl_u5f_50k" in r["labelsOrTypes"][0])
        index_name = target["name"]
        deriv = target["labelsOrTypes"][0]
        rel = deriv.replace("Derivative", "DERIVED_u5f_FROM")
        epi = deriv.replace("Derivative", "Episode")
        res = await s.run(f"MATCH (n:{deriv}) RETURN keys(n) AS ks LIMIT 1")
        ks = (await res.single())["ks"]
        strip = [k for k in ks if "embedding" in k or "similarity" in k]
    print(f"label={deriv}\nstripped keys={strip}")

    i = 0

    async def ann_no_embedding():
        nonlocal i
        async with driver.session() as s:
            res = await s.run(
                f"CALL db.index.vector.queryNodes($idx, 50, $v) "
                f"YIELD node AS n, score AS s "
                f"RETURN s, apoc.map.removeKeys(properties(n), $strip) AS props "
                f"ORDER BY s DESC",
                idx=index_name, v=qvecs[i % 300], strip=strip,
            )
            _ = [r async for r in res]
        i += 1

    async def fused(qi=None):
        nonlocal i
        v = qvecs[(qi if qi is not None else i) % 300]
        async with driver.session() as s:
            res = await s.run(
                f"CALL db.index.vector.queryNodes($idx, 50, $v) "
                f"YIELD node AS d, score AS s "
                f"MATCH (d)-[:{rel}]->(e:{epi}) "
                f"RETURN s, e ORDER BY s DESC",
                idx=index_name, v=v,
            )
            _ = [r async for r in res]
        i += 1

    await fused()  # warm

    for name, fn in [
        ("A ann, derivative props minus embedding (apoc)", ann_no_embedding),
        ("B fused ann+traversal+episode, one statement  ", fused),
    ]:
        mean, p50 = await timeit(fn)
        print(f"{name} mean {mean:7.2f} ms  p50 {p50:7.2f} ms")

    thunks = [lambda q=q: fused(q) for q in range(300)]
    lat, _, wall = await run_workers(split_chunks(thunks, 16))
    ls = sorted(lat)
    print(
        f"C fused at c16: {300 / wall:6.1f} QPS  "
        f"p50 {ls[150] * 1000:7.2f} ms  p95 {ls[285] * 1000:7.2f} ms"
    )

    await driver.close()


asyncio.run(main())
