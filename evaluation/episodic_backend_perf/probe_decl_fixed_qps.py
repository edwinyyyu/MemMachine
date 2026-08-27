"""QPS of the hypothetical FIXED Neo4j query path, at concurrency.

Shapes:
  fused : one statement (ANN k=50 + traversal + episode projection)
  full  : fused, then batched CALL-subquery expansion (back 2 / fwd 4)
          over the top-10 episodes -- the complete repaired query
          (3 statements, 2 round-trip waves per query)

Usage: probe_decl_fixed_qps.py --shape fused|full --concurrency N
       [--queries N] [--seed-offset N]
Run two instances concurrently (different --seed-offset) to test the
server-side aggregate ceiling past one client process.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import FakeEmbedder, make_queries, run_workers, split_chunks
from neo4j import AsyncGraphDatabase

from common import NEO4J_AUTH as AUTH, NEO4J_URI as URI  # noqa: E402


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=["fused", "full"], required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--queries", type=int, default=300)
    p.add_argument("--seed-offset", type=int, default=0)
    args = p.parse_args()

    driver = AsyncGraphDatabase.driver(URI, auth=AUTH, max_connection_pool_size=200)
    emb = FakeEmbedder()
    queries = make_queries(args.queries + 10, seed=777 + args.seed_offset)
    qvecs = [emb._embed_one(q) for q in queries]

    async with driver.session() as s:
        res = await s.run(
            "SHOW INDEXES YIELD name, type, labelsOrTypes WHERE type='VECTOR' "
            "RETURN name, labelsOrTypes"
        )
        rows = [r async for r in res]
        target = next(r for r in rows if "decl_u5f_50k" in r["labelsOrTypes"][0])
        index_name = target["name"]
        deriv = target["labelsOrTypes"][0]
        rel = deriv.replace("Derivative", "DERIVED_u5f_FROM")
        epi = deriv.replace("Derivative", "Episode")
        res = await s.run(f"MATCH (n:{epi}) RETURN keys(n) AS ks LIMIT 1")
        ts_prop = next(k for k in (await res.single())["ks"] if "timestamp" in k)

    fused_q = (
        f"CALL db.index.vector.queryNodes($idx, 50, $v) "
        f"YIELD node AS d, score AS s "
        f"MATCH (d)-[:{rel}]->(e:{epi}) "
        f"RETURN s, e.uid AS uid, e.{ts_prop} AS ts, e ORDER BY s DESC"
    )

    def expand_q(back):
        cmp1, order, lim = ("<", "DESC", 2) if back else (">", "ASC", 4)
        return (
            "UNWIND $seeds AS seed "
            "CALL { WITH seed "
            f"MATCH (n:{epi}) "
            f"WHERE n.{ts_prop} {cmp1} seed.ts"
            f" OR (n.{ts_prop} = seed.ts AND n.uid {cmp1} seed.uid) "
            f"RETURN n ORDER BY n.{ts_prop} {order}, n.uid {order} LIMIT {lim}"
            " } RETURN seed.uid AS seed_uid, n"
        )

    qb, qf = expand_q(True), expand_q(False)

    async def one(i):
        async with driver.session() as s:
            res = await s.run(fused_q, idx=index_name, v=qvecs[i])
            hits = [r async for r in res]
        if args.shape == "full":
            seen, seeds = set(), []
            for r in hits:
                if r["uid"] not in seen:
                    seen.add(r["uid"])
                    seeds.append({"uid": r["uid"], "ts": r["ts"]})
                if len(seeds) == 10:
                    break

            async def run_dir(q):
                async with driver.session() as s:
                    res = await s.run(q, seeds=seeds)
                    _ = [r async for r in res]

            await asyncio.gather(run_dir(qb), run_dir(qf))

    await one(0)  # warm plans
    thunks = [lambda i=i: one(i) for i in range(args.queries)]
    lat, _, wall = await run_workers(split_chunks(thunks, args.concurrency))
    ls = sorted(lat)
    print(
        f"shape={args.shape} c={args.concurrency} n={args.queries}: "
        f"{args.queries / wall:7.1f} QPS  "
        f"p50 {ls[len(ls) // 2] * 1000:7.2f} ms  "
        f"p95 {ls[int(len(ls) * 0.95)] * 1000:7.2f} ms"
    )
    await driver.close()


asyncio.run(main())
