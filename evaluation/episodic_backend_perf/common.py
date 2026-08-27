"""Shared benchmark utilities: fake embedder, corpus, timing, results.

Run with the repo venv's python (memmachine_server importable from the uv
workspace). All randomness is seeded; the embedder is
deterministic (hash-seeded gaussian, L2-normalized) so both backends see
identical vectors and zero embedding latency -- benchmarks isolate the
storage layer.
"""

import asyncio
import hashlib
import json
import random
import statistics
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder.embedder import Embedder

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Endpoints/credentials: override via env; defaults match the repo
# docker-compose defaults (a local .env may change the passwords).
import os
NEO4J_URI = os.environ.get("BENCH_NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (
    os.environ.get("NEO4J_USER", "neo4j"),
    os.environ.get("NEO4J_PASSWORD", "neo4j_password"),
)
PG_DSN = os.environ.get(
    "BENCH_PG_DSN",
    "postgresql+asyncpg://{u}:{p}@localhost:5432/{db}".format(
        u=os.environ.get("POSTGRES_USER", "memmachine"),
        p=os.environ.get("POSTGRES_PASSWORD", "memmachine_password"),
        db=os.environ.get("POSTGRES_DB", "memmachine"),
    ),
)

DIMENSIONS = 768
NUM_CATEGORIES = 10
BASE_TS = datetime(2026, 1, 1, tzinfo=UTC)


class FakeEmbedder(Embedder):
    """Deterministic, zero-latency embedder isolating store performance."""

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        super().__init__(batch_size=None)
        self._dimensions = dimensions

    def _embed_one(self, text: str) -> list[float]:
        seed = int.from_bytes(
            hashlib.blake2b(str(text).encode(), digest_size=8).digest(), "big"
        )
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dimensions).astype(np.float32)
        v /= np.linalg.norm(v)
        return [float(x) for x in v]

    async def _ingest_embed(self, inputs, max_attempts=1):
        return [self._embed_one(t) for t in inputs]

    async def _search_embed(self, queries, max_attempts=1):
        return [self._embed_one(t) for t in queries]

    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


_SPEAKERS = ["alice", "bob", "carol", "dave"]
_TOPICS = [
    "the quarterly latency report", "index compaction", "the staging deploy",
    "vacation planning", "the new espresso machine", "shard rebalancing",
    "the customer escalation", "backup retention", "the hiring loop",
    "graph migrations", "payload filtering", "the standup notes",
    "cache eviction", "the incident retro", "embedding drift",
    "the pricing page", "connection pooling", "the release checklist",
    "timezone handling", "the demo script",
]
_VERBS = [
    "reviewed", "questioned", "summarized", "postponed", "escalated",
    "approved", "measured", "rewrote", "debugged", "documented",
]
_TAILS = [
    "and asked for a follow-up by Friday",
    "but the numbers did not add up",
    "so we agreed to revisit it next sprint",
    "which surprised everyone in the meeting",
    "and filed a ticket with the details",
    "after comparing three alternatives",
    "despite the flaky test results",
    "and pinged the on-call about it",
]


def make_texts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        s = rng.choice(_SPEAKERS)
        t = rng.choice(_TOPICS)
        v = rng.choice(_VERBS)
        tail = rng.choice(_TAILS)
        t2 = rng.choice(_TOPICS)
        out.append(
            f"{s} {v} {t} {tail}; they also mentioned {t2} "
            f"in passing (msg {i})"
        )
    return out


def make_queries(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = rng.choice(_TOPICS)
        s = rng.choice(_SPEAKERS)
        out.append(f"what did {s} say about {t}? (q {i})")
    return out


def category_of(i: int) -> str:
    return f"cat_{i % NUM_CATEGORIES}"


def ts_of(i: int) -> datetime:
    return BASE_TS + timedelta(seconds=i)


async def run_workers(
    thunk_lists: list[list],
) -> tuple[list[float], list[float], float]:
    """Each worker runs its thunk list sequentially; workers run concurrently.

    Returns (latencies, start_offsets, wall_seconds). latencies/start_offsets
    are in global submission order (worker-interleaved by completion index).
    """
    latencies: list[float] = []
    offsets: list[float] = []
    t0 = time.perf_counter()

    async def worker(thunks):
        for thunk in thunks:
            s = time.perf_counter()
            await thunk()
            e = time.perf_counter()
            latencies.append(e - s)
            offsets.append(s - t0)

    await asyncio.gather(*(worker(ts) for ts in thunk_lists))
    wall = time.perf_counter() - t0
    return latencies, offsets, wall


def split_chunks(items: list, k: int) -> list[list]:
    """Round-robin split preserving per-worker order."""
    chunks = [[] for _ in range(k)]
    for i, item in enumerate(items):
        chunks[i % k].append(item)
    return chunks


def summarize(latencies: list[float], wall: float, n_ops: int) -> dict:
    ls = sorted(latencies)

    def pct(p):
        return ls[min(len(ls) - 1, int(p / 100 * len(ls)))] * 1000

    return {
        "ops": n_ops,
        "wall_s": round(wall, 3),
        "throughput_ops_s": round(n_ops / wall, 2),
        "mean_ms": round(statistics.fmean(ls) * 1000, 3),
        "p50_ms": round(pct(50), 3),
        "p90_ms": round(pct(90), 3),
        "p95_ms": round(pct(95), 3),
        "p99_ms": round(pct(99), 3),
        "max_ms": round(ls[-1] * 1000, 3),
    }


def save_result(name: str, settings: dict, metrics: dict, raw: dict | None = None):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    payload = {
        "name": name,
        "recorded_at": datetime.now(UTC).isoformat(),
        "settings": settings,
        "metrics": metrics,
    }
    if raw:
        payload["raw"] = raw
    path.write_text(json.dumps(payload, indent=1))
    print(f"[saved] {path}")
    print(json.dumps(metrics, indent=1))
