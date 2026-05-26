"""Probe Mem0 cloud search latency via the official `mem0ai` SDK.

Uses MemoryClient (cloud), times its `search(...)` call against a sample
LoCoMo question. Reports per-call latency and a small summary.
"""

import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from mem0 import MemoryClient

ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.environ["MEM0_API_KEY"]
ORG_ID = os.getenv("MEM0_ORGANIZATION_ID") or None
PROJECT_ID = os.getenv("MEM0_PROJECT_ID") or None
HOST = os.getenv("MEM0_HOST", "https://api.mem0.ai").rstrip("/")

LOCOMO_DATA = Path(__file__).parent.parent / "data" / "locomo10.json"
USER_ID = os.getenv("MEM0_PROBE_USER_ID", "locomo_probe_user")
TOP_K = int(os.getenv("MEM0_PROBE_TOP_K", "200"))
N_RUNS = int(os.getenv("MEM0_PROBE_N_RUNS", "10"))


def main() -> None:
    with open(LOCOMO_DATA) as f:
        data = json.load(f)
    sample_q = data[0]["qa"][0]["question"]
    print(f"Sample query  : {sample_q!r}")
    print(f"user_id       : {USER_ID}")
    print(f"top_k         : {TOP_K}")
    print(f"host          : {HOST}")
    print(f"org_id set    : {bool(ORG_ID)}")
    print(f"project_id set: {bool(PROJECT_ID)}")
    print()

    client_kwargs = {"api_key": API_KEY, "host": HOST}
    if ORG_ID:
        client_kwargs["org_id"] = ORG_ID
    if PROJECT_ID:
        client_kwargs["project_id"] = PROJECT_ID
    client = MemoryClient(**client_kwargs)

    search_kwargs = {
        "query": sample_q,
        "filters": {"user_id": USER_ID},
        "top_k": TOP_K,
        "version": "v2",  # SDK uses /v2/memories/search/ when version='v2'
    }

    # Try a warmup call to see what the SDK does.
    print("warmup call...")
    try:
        results = client.search(**search_kwargs)
        n = (
            len(results)
            if isinstance(results, list)
            else len(results.get("results", []))
        )
        print(f"  ok, {n} results returned")
    except Exception as e:
        print(f"  warmup raised: {type(e).__name__}: {e}")
        return
    print()

    latencies = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        try:
            results = client.search(**search_kwargs)
            n = (
                len(results)
                if isinstance(results, list)
                else len(results.get("results", []))
            )
        except Exception as e:
            print(f"  run {i + 1}: ERROR {type(e).__name__}: {e}")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        print(f"  run {i + 1:>2}: {elapsed_ms:>6.1f} ms  ({n} results)")

    if not latencies:
        print("\nNo successful runs.")
        return

    latencies.sort()
    n = len(latencies)
    print()
    print(f"Latency over {n} calls (ms):")
    print(f"  min   {min(latencies):.1f}")
    print(f"  p50   {latencies[n // 2]:.1f}")
    print(f"  p90   {latencies[int(n * 0.9)]:.1f}")
    print(f"  max   {max(latencies):.1f}")
    print(f"  mean  {sum(latencies) / n:.1f}")
    if n >= 2:
        print(f"  stdev {statistics.stdev(latencies):.1f}")


if __name__ == "__main__":
    main()
