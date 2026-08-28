"""Closed-loop REST bench against a local memmachine-server (v2 API).

Mirrors the artifact's REST harness shape: create project, ingest via
/memories (episodic only), search via /memories/search with top_k=10
(core derives vsl=50), closed-loop workers at given concurrency.
"""

import argparse
import asyncio
import random
import statistics
import sys
import time
from datetime import UTC, datetime, timedelta

import httpx

BASE = "http://127.0.0.1:8091/api/v2"  # overridden by --base
ORG = "benchorg"

_SPEAKERS = ["alice", "bob", "carol", "dave"]
_TOPICS = [
    "the quarterly latency report", "index compaction", "the staging deploy",
    "vacation planning", "the new espresso machine", "shard rebalancing",
    "the customer escalation", "backup retention", "the hiring loop",
    "graph migrations", "payload filtering", "the standup notes",
]
_VERBS = ["reviewed", "questioned", "summarized", "postponed", "escalated",
          "approved", "measured", "rewrote", "debugged", "documented"]


def texts(n, seed):
    rng = random.Random(seed)
    return [
        f"{rng.choice(_SPEAKERS)} {rng.choice(_VERBS)} {rng.choice(_TOPICS)}"
        f" and mentioned {rng.choice(_TOPICS)} in passing (msg {i})"
        for i in range(n)
    ]


def queries(n, seed):
    rng = random.Random(seed)
    return [
        f"what did {rng.choice(_SPEAKERS)} say about {rng.choice(_TOPICS)}? (q {i})"
        for i in range(n)
    ]


async def run_arm(name, thunks, c, json_out="", ramp=0.0):
    # (done_offset_s, latency_s) per request; ramp staggers worker starts so
    # an arm doesn't open c connections in one SYN burst (macOS drops SYNs
    # beyond kern.ipc.somaxconn=128, stalling some connections for seconds).
    recs = []

    async def worker(i, items):
        if ramp:
            await asyncio.sleep(ramp * i / c)
        for t in items:
            s = time.perf_counter()
            await t()
            d = time.perf_counter()
            recs.append((d - t0, d - s))

    chunks = [thunks[i::c] for i in range(c)]
    start_epoch = time.time()
    t0 = time.perf_counter()
    await asyncio.gather(*(worker(i, ch) for i, ch in enumerate(chunks)))
    wall = time.perf_counter() - t0
    ls = sorted(r[1] for r in recs)
    n = len(ls)
    print(f"{name:<28} n={n:<5} {n / wall:7.1f}/s  mean {statistics.fmean(ls) * 1000:7.1f}  "
          f"p50 {ls[n // 2] * 1000:7.1f}  p95 {ls[int(n * .95)] * 1000:7.1f}  "
          f"p99 {ls[int(n * .99)] * 1000:7.1f} ms", flush=True)
    if json_out:
        import json
        with open(json_out, "a") as f:
            f.write(json.dumps({"name": name, "c": c, "n": n, "wall": wall,
                                "start": start_epoch, "end": start_epoch + wall,
                                "ramp": ramp, "recs": recs}) + "\n")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="bench1")
    p.add_argument("--ingest", type=int, default=0)
    p.add_argument("--ingest-concurrency", type=int, default=4)
    p.add_argument("--search-arms", default="1,16,32")
    p.add_argument("--queries", type=int, default=280)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--types", default="")
    p.add_argument("--base", default="")
    p.add_argument("--auth", default="")
    p.add_argument("--max-conns", type=int, default=64)
    p.add_argument("--ramp", type=float, default=0.0)
    p.add_argument("--json-out", default="")
    p.add_argument("--skip-create", action="store_true")
    args = p.parse_args()
    global BASE
    if args.base:
        BASE = args.base

    headers = {"Authorization": f"Bearer {args.auth}"} if args.auth else {}
    async with httpx.AsyncClient(timeout=120, headers=headers, limits=httpx.Limits(
            max_connections=args.max_conns,
            max_keepalive_connections=args.max_conns)) as cl:
        if not args.skip_create:
            r = await cl.post(f"{BASE}/projects", json={
                "org_id": ORG, "project_id": args.project,
                "description": "bench", "config": {}})
            print("create project:", r.status_code, r.text[:200], flush=True)

        if args.ingest:
            base_ts = datetime(2026, 1, 1, tzinfo=UTC)
            ts_list = texts(args.ingest, args.seed)

            def add_thunk(i):
                async def go():
                    r = await cl.post(f"{BASE}/memories", json={
                        "org_id": ORG, "project_id": args.project,
                        "types": ["episodic"],
                        "messages": [{
                            "content": ts_list[i],
                            "producer": "alice" if i % 2 == 0 else "bob",
                            "produced_for": "assistant",
                            "timestamp": (base_ts + timedelta(seconds=i)).isoformat(),
                            "role": "user",
                        }]})
                    if r.status_code >= 300:
                        print("ingest err", r.status_code, r.text[:150], flush=True)
                return go

            await run_arm(f"ingest {args.ingest} c{args.ingest_concurrency}",
                          [add_thunk(i) for i in range(args.ingest)],
                          args.ingest_concurrency)

        qs = queries(args.queries + 10, args.seed + 1)

        transport_errors = [0]

        def q_thunk(i):
            async def go():
                body = {"org_id": ORG, "project_id": args.project,
                        "top_k": 10, "query": qs[i]}
                if args.types:
                    body["types"] = args.types.split(",")
                for attempt in (0, 1):
                    try:
                        r = await cl.post(f"{BASE}/memories/search", json=body)
                    except httpx.HTTPError as exc:
                        transport_errors[0] += 1
                        if attempt:
                            print("transport err (gave up):",
                                  type(exc).__name__, exc, flush=True)
                            return
                        await asyncio.sleep(0.05)
                        continue
                    if r.status_code >= 300:
                        print("search err", r.status_code, r.text[:150], flush=True)
                    return
            return go

        for i in range(5):
            await q_thunk(i)()  # warmup
        for c in map(int, args.search_arms.split(",")):
            await run_arm(f"search c{c}", [q_thunk(i) for i in range(args.queries)], c,
                          json_out=args.json_out, ramp=args.ramp)
            if transport_errors[0]:
                print(f"transport errors (retried): {transport_errors[0]}",
                      flush=True)
                transport_errors[0] = 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
