"""Can DeclarativeMemory's context expansion be fixed on Neo4j?

The shipped path issues 2 directional Cypher queries PER returned episode
(20 for K=10, expand=6), each with datetime comparisons expanded to
.epochSeconds/.nanosecond accessors (timezone-equality workaround).
Measured cost: +775 ms per query at c1 (results digest).

This probe measures, on the decl_50k session:
  E0. one shipped-shape directional query (epochSeconds expansion), timed
      + PROFILE operator/dbHits
  E1. same query with DIRECT temporal comparisons (valid when timestamps
      are written tz-normalized), timed + PROFILE
  E2. batched: ONE statement per direction for all 10 seeds via
      UNWIND + CALL subquery with per-seed LIMIT (Cypher's LATERAL)
  E3. E2 both directions = full expansion for a K=10 search, wall
Reference: event-side expansion delta is +7.5 ms (one LATERAL round trip).
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from neo4j import AsyncGraphDatabase

from common import NEO4J_AUTH as AUTH, NEO4J_URI as URI  # noqa: E402
REPS = 20


def profile_summary(profile):
    ops = []

    def walk(p, depth=0):
        ops.append((p["operatorType"], p["args"].get("DbHits", 0)))
        for c in p.get("children", []):
            walk(c, depth + 1)

    walk(profile)
    total = sum(h for _, h in ops)
    top = sorted(ops, key=lambda x: -x[1])[:3]
    return f"dbHits={total}  top={top}"


async def timeit(s_factory, reps=REPS):
    xs = []
    for _ in range(reps):
        t = time.perf_counter()
        await s_factory()
        xs.append((time.perf_counter() - t) * 1000)
    xs.sort()
    return statistics.fmean(xs), xs[len(xs) // 2]


async def main():
    driver = AsyncGraphDatabase.driver(URI, auth=AUTH)

    async with driver.session() as s:
        res = await s.run(
            "SHOW INDEXES YIELD name, type, labelsOrTypes WHERE type='VECTOR' "
            "RETURN labelsOrTypes"
        )
        rows = [r async for r in res]
        deriv = next(
            r["labelsOrTypes"][0] for r in rows
            if "decl_u5f_50k" in r["labelsOrTypes"][0]
        )
        epi = deriv.replace("Derivative", "Episode")
        res = await s.run(f"MATCH (n:{epi}) RETURN keys(n) AS ks LIMIT 1")
        ks = (await res.single())["ks"]
        ts_prop = next(k for k in ks if "timestamp" in k)
        res = await s.run(
            f"MATCH (n:{epi}) RETURN n.uid AS uid, n.{ts_prop} AS ts "
            f"ORDER BY n.{ts_prop} SKIP 25000 LIMIT 10"
        )
        seeds = [{"uid": r["uid"], "ts": r["ts"]} async for r in res]
    print(f"episode label={epi} ts_prop={ts_prop} seeds={len(seeds)}")

    # E0: shipped shape (backward, LIMIT 2), epochSeconds-expanded lexicographic
    shipped_where = (
        f"((n.{ts_prop} < $ts AND (n.{ts_prop}.epochSeconds < $ts.epochSeconds"
        f" OR (n.{ts_prop}.epochSeconds = $ts.epochSeconds"
        f" AND n.{ts_prop}.nanosecond < $ts.nanosecond)))"
        f" OR (n.uid < $uid AND (n.{ts_prop} = $ts"
        f" OR (n.{ts_prop}.epochSeconds = $ts.epochSeconds"
        f" AND n.{ts_prop}.nanosecond = $ts.nanosecond))))"
    )
    e0 = (
        f"MATCH (n:{epi}) WHERE {shipped_where} AND TRUE RETURN n "
        f"ORDER BY n.{ts_prop} DESC, n.uid DESC LIMIT 2"
    )
    # E1: direct comparisons (valid for tz-normalized writes)
    e1 = (
        f"MATCH (n:{epi}) WHERE (n.{ts_prop} < $ts"
        f" OR (n.{ts_prop} = $ts AND n.uid < $uid)) RETURN n "
        f"ORDER BY n.{ts_prop} DESC, n.uid DESC LIMIT 2"
    )

    async def run_one(q):
        async with driver.session() as s:
            res = await s.run(q, ts=seeds[5]["ts"], uid=seeds[5]["uid"])
            _ = [r async for r in res]

    for name, q in [("E0 shipped epochSeconds shape", e0),
                    ("E1 direct temporal comparison ", e1)]:
        async with driver.session() as s:
            res = await s.run(
                "PROFILE " + q, ts=seeds[5]["ts"], uid=seeds[5]["uid"]
            )
            _ = [r async for r in res]
            prof = (await res.consume()).profile
        mean, p50 = await timeit(lambda q=q: run_one(q))
        print(f"{name} mean {mean:7.2f} p50 {p50:7.2f} ms   {profile_summary(prof)}")
        print(f"  -> shipped path issues 20 of these per search "
              f"(~{p50 * 20:.0f} ms serialized-equivalent)" if "E0" in name else "", end="")
        print()

    # E2/E3: batched CALL-subquery (LATERAL equivalent), direct comparisons
    def batched(direction):
        back = direction == "back"
        cmp1 = "<" if back else ">"
        order = "DESC" if back else "ASC"
        lim = 2 if back else 4
        return (
            "UNWIND $seeds AS seed "
            "CALL { WITH seed "
            f"MATCH (n:{epi}) "
            f"WHERE n.{ts_prop} {cmp1} seed.ts"
            f" OR (n.{ts_prop} = seed.ts AND n.uid {cmp1} seed.uid) "
            f"RETURN n ORDER BY n.{ts_prop} {order}, n.uid {order} LIMIT {lim}"
            " } RETURN seed.uid AS seed_uid, n.uid AS ctx_uid, n"
        )

    async def run_batched(q):
        async with driver.session() as s:
            res = await s.run(q, seeds=seeds)
            _ = [r async for r in res]

    qb, qf = batched("back"), batched("fwd")
    mean, p50 = await timeit(lambda: run_batched(qb))
    print(f"E2 batched backward, 10 seeds, 1 stmt   mean {mean:7.2f} p50 {p50:7.2f} ms")

    async def both():
        await asyncio.gather(run_batched(qb), run_batched(qf))

    mean, p50 = await timeit(both)
    print(f"E3 full expansion (2 stmts, 10 seeds)   mean {mean:7.2f} p50 {p50:7.2f} ms"
          f"   [shipped: ~775 ms; event stack: ~7.5 ms]")

    await driver.close()


asyncio.run(main())
