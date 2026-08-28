"""Run one search arm across several rest_bench client processes and merge.

A single asyncio httpx client process saturates around 300-400 req/s of
client-side CPU; to drive ~1000 req/s the load must come from several
processes. Each child runs the SAME closed-loop arm shape (seeded
differently), dumps raw latencies via --json-out, and this driver reports
the merged view: total n over the union window, plus merged percentiles.
"skew" is the sum of start spread and end spread across children — if it
is small relative to the window, the union-window throughput is honest.
"""

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import tempfile


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--c", type=int, required=True)
    ap.add_argument("--queries", type=int, required=True)
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--types", default="episodic")
    ap.add_argument("--ramp", type=float, default=0.0)
    args = ap.parse_args()

    procs = args.procs or max(1, math.ceil(args.c / 64))
    per_c = max(1, args.c // procs)
    per_q = max(per_c, args.queries // procs)
    outdir = tempfile.mkdtemp(prefix="mc_")
    bench = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rest_bench.py")

    cmds = []
    for i in range(procs):
        out = os.path.join(outdir, f"{i}.jsonl")
        cmds.append((out, [
            sys.executable, bench, "--project", args.project,
            "--search-arms", str(per_c), "--queries", str(per_q),
            "--seed", str(100 + i), "--types", args.types,
            "--max-conns", str(per_c), "--json-out", out, "--skip-create",
            "--ramp", str(args.ramp),
        ]))

    children = [
        await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        for _, cmd in cmds
    ]
    results = await asyncio.gather(*(ch.communicate() for ch in children))
    failed = 0
    errs = 0
    for ch, (out, err) in zip(children, results):
        for line in out.decode().splitlines():
            if "transport errors (retried):" in line:
                errs += int(line.rsplit(":", 1)[1])
            elif "err" in line and "create project" not in line:
                print("  child:", line[:200])
        if ch.returncode != 0:
            failed += 1
            print("CHILD FAILED (stderr tail):", err.decode()[-800:])

    runs = []
    for out, _ in cmds:
        try:
            with open(out) as f:
                runs.extend(json.loads(line) for line in f)
        except FileNotFoundError:
            pass
    if not runs:
        print(f"search c{args.c}: NO SURVIVING CHILDREN")
        sys.exit(1)
    tag = f" PARTIAL {procs - failed}/{procs}procs" if failed else ""
    if errs:
        tag += f" errs={errs}"

    n = sum(r["n"] for r in runs)
    starts = [r["start"] for r in runs]
    ends = [r["end"] for r in runs]
    window = max(ends) - min(starts)
    skew = (max(starts) - min(starts)) + (max(ends) - min(ends))
    lat = sorted(rec[1] for r in runs for rec in r["recs"])
    m = len(lat)
    # Steady-state rate: completions inside [latest start + ramp + 2s,
    # earliest end - 2s], excluding each child's ramp and tail drain.
    lo = max(starts) + args.ramp + 2.0
    hi = min(ends) - 2.0
    steady = ""
    if hi - lo >= 5.0:
        k = sum(1 for r in runs for rec in r["recs"]
                if lo <= r["start"] + rec[0] <= hi)
        steady = f"  steady {k / (hi - lo):7.1f}/s"
    print(f"search c{args.c} ({procs}proc x c{per_c}) n={n:<6} "
          f"{n / window:7.1f}/s  mean {statistics.fmean(lat) * 1000:7.1f}  "
          f"p50 {lat[m // 2] * 1000:7.1f}  p95 {lat[int(m * .95)] * 1000:7.1f}  "
          f"p99 {lat[int(m * .99)] * 1000:7.1f} ms  skew {skew:4.1f}s"
          f"{steady}{tag}", flush=True)


asyncio.run(main())
