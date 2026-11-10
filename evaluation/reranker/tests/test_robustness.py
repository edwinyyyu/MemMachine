"""Crash and failure safety (reranker-75's robustness review, the user's directions 2026-11-10):
index saves only when changed and atomic; one process per store; SQLite pragmas; a failed cache
write stops sending and keeps what was paid for; requests sized under MemMachine's per-call
cap; Ctrl-C sends nothing more; scores validated before caching; unreadable files measured
again."""

import asyncio
import json
import subprocess
import sys

import httpx
import numpy as np
import pytest

from rbench import dbs, depth, limits, memory, rerankers
from rbench.memory import ExactSearchEngine, Memory

from memmachine_server.common.data_types import SimilarityMetric  # noqa: I001

REAL_CLIENT = httpx.AsyncClient


def test_an_unchanged_index_is_not_rewritten_and_a_save_is_atomic(tmp_path):
    engine = ExactSearchEngine(4, SimilarityMetric.COSINE)
    path = str(tmp_path / "p.idx")

    async def go():
        await engine.add({1: [1, 0, 0, 0], 2: [0, 1, 0, 0]})
        await engine.save(path)
        first = (tmp_path / "p.idx").stat().st_mtime_ns
        again = ExactSearchEngine(4, SimilarityMetric.COSINE)
        await again.load(path)
        await again.save(path)  # loaded, unchanged: no write
        assert (tmp_path / "p.idx").stat().st_mtime_ns == first
        await again.remove([2])
        await again.save(path)  # changed: written, through a temporary file
        assert not list(tmp_path.glob("*.tmp"))
        third = ExactSearchEngine(4, SimilarityMetric.COSINE)
        await third.load(path)
        assert [m.key for m in (await third.search([[1, 0, 0, 0]], limit=5))[0].matches] == [1]

    asyncio.run(go())


def test_a_second_process_on_a_store_fails_fast(tmp_path):
    memory.lock_store(tmp_path)
    try:
        code = (f"import sys; sys.path.insert(0, {str(memory.__file__.rsplit('/rbench/', 1)[0])!r}); "
                f"from pathlib import Path; from rbench.memory import lock_store; lock_store(Path({str(tmp_path)!r}))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        assert out.returncode != 0 and "another rbench process is using" in (out.stderr + out.stdout)
        memory.lock_store(tmp_path)  # the same process may take it again (counted)
        memory.unlock_store(tmp_path)
    finally:
        memory.unlock_store(tmp_path)
    memory.lock_store(tmp_path)  # released: free again
    memory.unlock_store(tmp_path)


def test_every_database_gets_the_pragmas(tmp_path):
    db = dbs.connect(tmp_path / "x.sqlite")
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    assert db.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    assert db.execute("PRAGMA fullfsync").fetchone()[0] == 1

    async def go():
        from tests.test_memory import FakeEmbedder, _events  # noqa: PLC0415

        async with Memory("fake", "synthetic", "seg80", root=tmp_path / "s", backend=FakeEmbedder()) as m:
            await m.ingest("h", _events(5))
    asyncio.run(go())
    for name in ("segments", "records", "content"):  # the content store's: segments, records, and the vectors
        assert dbs.connect(tmp_path / "s" / f"{name}.sqlite").execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_requests_stay_under_memmachines_per_call_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "SECRETS", tmp_path)
    monkeypatch.setattr(memory, "EMBEDDINGS", tmp_path / "e")
    monkeypatch.setattr(memory, "questions_path", lambda e: tmp_path / "q.jsonl")
    (tmp_path / "deepinfra_api_key").write_text("test")
    embedder = memory.QueryCachingEmbedder("qwen3-8b-di")
    texts = ["x" * 2_000] * 64  # 128,000 characters: one call would be split by MemMachine
    ranges = embedder._requests(texts)
    assert all(sum(len(t) for t in texts[a:b]) <= memory.MAX_REQUEST_CHARS for a, b in ranges)
    assert all(b - a <= embedder._spec.request_inputs for a, b in ranges) and ranges[-1][1] == len(texts)
    asyncio.run(embedder.aclose())


def test_a_failed_cache_write_stops_sending(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "SECRETS", tmp_path)
    monkeypatch.setattr(memory, "EMBEDDINGS", tmp_path / "e")
    monkeypatch.setattr(memory, "questions_path", lambda e: tmp_path / "q.jsonl")
    (tmp_path / "deepinfra_api_key").write_text("test")
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(len(body["input"]))
        n = body.get("dimensions") or 1024
        data = [{"object": "embedding", "index": i, "embedding": [1.0] + [0.0] * (n - 1)} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"],
                                         "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    import openai

    real = openai.AsyncOpenAI
    monkeypatch.setattr(memory.openai, "AsyncOpenAI", lambda **kw: real(
        http_client=REAL_CLIENT(transport=httpx.MockTransport(handler)), **kw))
    monkeypatch.setattr(limits, "_LIMITS", {})
    embedder = memory.QueryCachingEmbedder("qwen3-8b-di")
    embedder._gate.state.limit = 1  # one request at a time: the failure is seen before the next is sent

    def full(rows):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(embedder._cache, "put", full)
    with pytest.raises(OSError):
        asyncio.run(embedder._ingest_embed([f"text {i}" for i in range(64 * 5)]))
    assert len(sent) == 1  # stopped after the first answer could not be cached
    asyncio.run(embedder.aclose())


def _rerank(handler, tmp_path, monkeypatch):
    monkeypatch.setattr(rerankers, "SCORES", tmp_path / "scores")
    monkeypatch.setattr(rerankers, "SECRETS", tmp_path)
    monkeypatch.setattr(limits, "_LIMITS", {})
    (tmp_path / "deepinfra_api_key").write_text("test")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))
    prompt = rerankers.prompt_key(rerankers.RERANKERS["q06"].template)
    work = [("q?", [(rerankers.ScoreCache.key(prompt, "em", "q?", f"doc {i}"), f"doc {i}") for i in range(10)])]
    rerankers.score("q06", work, "test")
    return work


def test_ctrl_c_sends_nothing_more(tmp_path, monkeypatch):
    sent = []

    def handler(request):
        sent.append(1)
        docs = json.loads(request.content)["documents"]
        return httpx.Response(200, json={"scores": [0.5] * len(docs), "input_tokens": 1})

    limits.CANCEL.set()
    try:
        with pytest.raises(limits.Interrupted):
            _rerank(handler, tmp_path, monkeypatch)
    finally:
        limits.CANCEL.clear()
    assert not sent


def test_scores_outside_a_probability_are_not_cached(tmp_path, monkeypatch):
    def handler(request):
        docs = json.loads(request.content)["documents"]
        return httpx.Response(200, json={"scores": [1.7] * len(docs), "input_tokens": 1})

    with pytest.raises(RuntimeError, match="outside"):
        _rerank(handler, tmp_path, monkeypatch)
    assert rerankers.ScoreCache("q06").db.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0


def test_an_unreadable_depth_file_counts_as_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(depth, "DEPTH", tmp_path)
    p = depth.path("e", "d", "c", "s")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"summary": {"fill"')  # cut off mid-write
    assert not depth._known("e", "d", "c", "s", 0)
    dbs.replace_atomically(p, lambda t: t.write_text(json.dumps({"summary": {"fill": {}}, "per_query": {}})))
    assert depth._read("e", "d", "c", "s") is not None and not list(tmp_path.rglob("*.partial*"))
    assert np.isfinite(1.0)


def test_vectors_survive_losing_the_stores(tmp_path, monkeypatch):
    """The embedding cache is kept: delete a finished partition's stores and index files and
    re-ingest the same events, and nothing is embedded again; the rebuilt index ranks the same."""
    import shutil
    import sqlite3

    from tests.test_memory import _events  # noqa: PLC0415

    monkeypatch.setattr(memory, "SECRETS", tmp_path)
    monkeypatch.setattr(memory, "EMBEDDINGS", tmp_path / "e")
    monkeypatch.setattr(memory, "questions_path", lambda e: tmp_path / "q.jsonl")
    monkeypatch.setattr(limits, "_LIMITS", {})
    (tmp_path / "deepinfra_api_key").write_text("test")
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(len(body["input"]))
        n = body.get("dimensions") or 1024
        data = [{"object": "embedding", "index": i,
                 "embedding": np.random.default_rng(len(t) * 7 + i).normal(size=n).tolist()}
                for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"],
                                         "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    import openai

    real = openai.AsyncOpenAI
    monkeypatch.setattr(memory.openai, "AsyncOpenAI", lambda **kw: real(
        http_client=REAL_CLIENT(transport=httpx.MockTransport(handler)), **kw))
    events = _events(30)
    root = tmp_path / "stores"

    async def ingest_and_query() -> list[str]:
        async with Memory("qwen3-8b-di", "synthetic", "seg80", root=root) as m:
            assert await m.ingest("h", events)
            partition = await (await m.event_memory()).open_or_create_partition("h")
            result = await partition.query("the price", vector_search_limit=10)
            return [s.block.text for c in result.scored_segment_contexts for s in c.segments]

    first = asyncio.run(ingest_and_query())
    documents = [n for n in sent]
    assert documents
    shutil.rmtree(root)  # both SQLite stores, every index file, the done markers: gone
    sent.clear()
    again = asyncio.run(ingest_and_query())
    assert sent == [1] or sent == []  # at most the search's question (it is cached too after the first run)
    assert again == first
    (cache,) = (tmp_path / "e").glob("*.sqlite")  # the store's vectors are the cache's: none stored twice
    db = sqlite3.connect(cache)
    assert db.execute("SELECT count(*) FROM vectors WHERE key LIKE 'v:%'").fetchone()[0] == 0
    db.close()


CHILD = r"""
import asyncio, dataclasses, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import httpx, numpy as np, openai
from rbench import embedding, limits, memory
from tests.test_memory import _events

tmp, log = Path(sys.argv[2]), Path(sys.argv[3])
memory.SECRETS, memory.EMBEDDINGS = tmp, tmp / "e"
memory.questions_path = lambda e: tmp / "q.jsonl"
embedding.EMBEDDERS["qwen3-8b-di"] = dataclasses.replace(embedding.EMBEDDERS["qwen3-8b-di"], request_inputs=8, in_flight=1)
REAL = httpx.AsyncClient

def handler(request):
    body = json.loads(request.content)
    time.sleep(0.05)
    with log.open("a") as f:
        f.write(json.dumps(body["input"]) + "\n")
    n = body.get("dimensions") or 1024
    data = [{"object": "embedding", "index": i, "embedding": np.random.default_rng(
        int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)).normal(size=n).tolist()} for i, t in enumerate(body["input"])]
    return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"],
                                     "usage": {"prompt_tokens": 1, "total_tokens": 1}})

real = openai.AsyncOpenAI
memory.openai.AsyncOpenAI = lambda **kw: real(http_client=REAL(transport=httpx.MockTransport(handler)), **kw)
memory.ENCODE_SEGMENTS = 40

async def go():
    async with memory.Memory("qwen3-8b-di", "synthetic", "seg80", root=tmp / "stores") as m:
        await m.ingest("h", _events(200))

asyncio.run(go())
"""


def test_kill_9_mid_ingest_keeps_the_cache_whole(tmp_path):
    """kill -9 while a hosted embedder's answers are being cached: the cache passes SQLite's
    integrity check, the resumed run sends none of the texts that were cached, and the finished
    store ranks exactly as an uninterrupted run."""
    import dataclasses
    import hashlib
    import os
    import signal
    import sqlite3
    import time

    import openai

    from rbench import embedding
    from tests.test_memory import _events  # noqa: PLC0415

    root = memory.__file__.rsplit("/rbench/", 1)[0]
    log = tmp_path / "sent.jsonl"
    (tmp_path / "deepinfra_api_key").write_text("test")
    child = subprocess.Popen([sys.executable, "-c", CHILD, root, str(tmp_path), str(log)], cwd=root)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and (not log.exists() or len(log.read_text().splitlines()) < 6):
        time.sleep(0.02)
    os.kill(child.pid, signal.SIGKILL)
    child.wait()
    assert child.returncode == -signal.SIGKILL

    caches = list((tmp_path / "e").glob("*.sqlite"))
    assert len(caches) == 1
    db = sqlite3.connect(caches[0])
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    cached = {k for (k,) in db.execute("SELECT key FROM vectors")}
    db.close()
    assert cached  # some answers were cached before the kill

    sent: list[str] = []

    def handler(request):
        body = json.loads(request.content)
        sent.extend(body["input"])
        n = body.get("dimensions") or 1024
        data = [{"object": "embedding", "index": i, "embedding": np.random.default_rng(
            int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)).normal(size=n).tolist()} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"],
                                         "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    real = openai.AsyncOpenAI
    saved = (memory.SECRETS, memory.EMBEDDINGS, memory.questions_path, memory.openai.AsyncOpenAI, memory.ENCODE_SEGMENTS,
             embedding.EMBEDDERS["qwen3-8b-di"], dict(limits._LIMITS))
    try:
        memory.SECRETS, memory.EMBEDDINGS = tmp_path, tmp_path / "e"
        memory.questions_path = lambda e: tmp_path / "q.jsonl"
        memory.openai.AsyncOpenAI = lambda **kw: real(http_client=REAL_CLIENT(transport=httpx.MockTransport(handler)), **kw)
        memory.ENCODE_SEGMENTS = 40
        embedding.EMBEDDERS["qwen3-8b-di"] = dataclasses.replace(saved[5], request_inputs=8, in_flight=1)

        async def run(stores) -> list[str]:
            async with Memory("qwen3-8b-di", "synthetic", "seg80", root=stores) as m:
                assert await m.ingest("h", _events(200))
                partition = await (await m.event_memory()).open_or_create_partition("h")
                result = await partition.query("the price", vector_search_limit=15)
                return [s.block.text for c in result.scored_segment_contexts for s in c.segments]

        resumed = asyncio.run(run(tmp_path / "stores"))
        repeated = {memory.EmbeddingCache.key(t) for t in sent} & cached
        assert not repeated  # nothing that had been cached was paid for again
        memory.EMBEDDINGS = tmp_path / "clean-cache"
        sent.clear()
        clean = asyncio.run(run(tmp_path / "clean"))  # an uninterrupted run, from nothing
        assert sent and resumed == clean
    finally:
        (memory.SECRETS, memory.EMBEDDINGS, memory.questions_path, memory.openai.AsyncOpenAI, memory.ENCODE_SEGMENTS,
         embedding.EMBEDDERS["qwen3-8b-di"]) = saved[:6]
        limits._LIMITS.clear()
        limits._LIMITS.update(saved[6])
