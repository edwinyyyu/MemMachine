"""Hosted calls under failure: out of credits mid-run stops cleanly with nothing finished lost
and a rerun completes (paying only for what was never embedded); 429s shrink the adaptive
limit and successes grow it; an hour of failures stops with GaveUp. The API is replaced by an
in-process transport (no network, no key)."""

import asyncio
import datetime
import json
import random
import uuid

import httpx
import numpy as np
import openai
import pytest

from rbench import credits, limits, memory, rerankers
from rbench.memory import Memory

from memmachine_server.episodic_memory.event_memory.data_types import Event, ProducerContext, TextBlock  # noqa: I001

BROKE = httpx.Response(402, json={"error": {"message": "Insufficient balance"}})
REAL_CLIENT = httpx.AsyncClient  # before any test patches it


def test_refusals_are_recognized():
    assert credits.refused(402, "")
    assert credits.refused(403, '{"detail": "Your account balance is too low"}')
    assert not credits.refused(429, "rate limited: add credits for higher limits")  # 429 is retried
    assert not credits.refused(400, "bad request") and not credits.refused(500, "credit")


def test_reranking_stops_cleanly_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(rerankers, "SCORES", tmp_path / "scores")
    monkeypatch.setattr(rerankers, "SECRETS", tmp_path)
    (tmp_path / "deepinfra_api_key").write_text("test")
    real = httpx.AsyncClient
    seen = {"ok": 0, "documents": 0, "limit": 3}

    def handler(request: httpx.Request) -> httpx.Response:
        if seen["ok"] >= seen["limit"]:
            return BROKE
        seen["ok"] += 1
        docs = json.loads(request.content)["documents"]
        seen["documents"] += len(docs)
        return httpx.Response(200, json={"scores": [0.5] * len(docs), "input_tokens": 10 * len(docs)})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    prompt = rerankers.prompt_key(rerankers.RERANKERS["q06"].template)
    work = [(f"question {q}?", [(rerankers.ScoreCache.key(prompt, "em", f"question {q}?", f"doc {q} {i}"),
                                 f"doc {q} {i}") for i in range(40)]) for q in range(3)]  # 6 requests of <= 32
    keys = [k for _, items in work for k, _ in items]
    with pytest.raises(SystemExit) as stop:
        rerankers.score("q06", work, "test")
    assert credits.MESSAGE in str(stop.value)
    cache = rerankers.ScoreCache("q06")
    kept = cache.get(keys)
    assert len(kept) == seen["documents"] > 0 and len(kept) < len(keys)  # the requests that returned are cached
    assert cache.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == seen["limit"]  # and logged

    seen.update(ok=0, documents=0, limit=10**9)  # topped up: the rerun sends only what is missing
    rest = [(q, [(k, d) for k, d in items if k not in kept]) for q, items in work]
    rerankers.score("q06", [(q, items) for q, items in rest if items], "test")
    assert seen["documents"] == len(keys) - len(kept) and len(cache.get(keys)) == len(keys)


def _events(n: int) -> list[Event]:
    rng = random.Random(0)
    base = datetime.datetime(2023, 5, 20, 9, 0, tzinfo=datetime.UTC)
    words = "we talked about the trip and the price of the tickets, then about dinner plans".split()
    return [Event(uuid=uuid.UUID(int=rng.getrandbits(128)), timestamp=base + datetime.timedelta(minutes=i),
                  context=ProducerContext(producer=rng.choice(["User", "Assistant"])),
                  blocks=[TextBlock(text=("SENTINEL " if i == n - 3 else "")
                                    + " ".join(rng.choice(words) for _ in range(rng.randint(5, 40))))])
            for i in range(n)]


def test_ingest_stops_cleanly_and_the_partition_is_encoded_again(tmp_path, monkeypatch):
    """Out of credits in a partition's second encode_events batch, after the first batch was
    written to both stores: no done marker; a fresh run deletes the half-written partition,
    reopens it and encodes it whole, the same as a clean encoding."""
    embedder = "qwen3-8b-di"
    monkeypatch.setattr(memory, "SECRETS", tmp_path)
    monkeypatch.setattr(memory, "questions_path", lambda e: tmp_path / "questions.jsonl")
    monkeypatch.setattr(memory, "ENCODE_SEGMENTS", 25)  # several encode_events calls per partition
    monkeypatch.setattr(memory, "EMBEDDINGS", tmp_path / "embeddings")
    (tmp_path / "deepinfra_api_key").write_text("test")
    state = {"broke": True, "sent": []}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if state["broke"] and any("SENTINEL" in t for t in body["input"]):
            return BROKE
        prompt = memory.emb.EMBEDDERS[embedder].query_prompt
        state["sent"] += [t for t in body["input"] if not t.startswith(prompt)]  # documents (not the search's question)
        n = body.get("dimensions") or 1024
        data = [{"object": "embedding", "index": i, "embedding": np.random.default_rng(len(t)).normal(size=n).tolist()}
                for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"],
                                         "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    real = openai.AsyncOpenAI
    monkeypatch.setattr(memory.openai, "AsyncOpenAI", lambda **kw: real(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw))
    events = _events(60)

    async def segments(m: Memory, key: str) -> int:
        segment_store, _, _ = await m.stores()
        part = await segment_store.open_partition(key)
        found = await part.get_segment_uuids_by_event_uuids([e.uuid for e in events])
        return sum(len(v) for v in found.values())

    async def broke() -> int:
        async with Memory(embedder, "synthetic", "seg80", root=tmp_path / "stores") as m:
            with pytest.raises(credits.OutOfCredits, match=credits.MESSAGE):
                await m.ingest("h", events)
            assert not m.done("h").exists()
            return await segments(m, "h")

    def cached() -> int:
        cache = memory.EmbeddingCache(memory.emb.EMBEDDERS[embedder])
        try:
            return cache.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        finally:
            cache.close()

    async def rerun(root) -> tuple[int, list]:
        async with Memory(embedder, "synthetic", "seg80", root=root) as m:
            assert await m.ingest("h", events)
            assert m.done("h").exists()
            partition = await (await m.event_memory()).open_or_create_partition("h")
            hits = (await partition.query("the price of the tickets", vector_search_limit=5)).scored_segment_contexts
            return await segments(m, "h"), hits

    partial = asyncio.run(broke())
    assert partial > 0  # the first batch was written before the failure: a half-written partition
    paid = len(state["sent"])
    paid_unique = len(set(state["sent"]))
    assert cached() == paid_unique > 0  # what was embedded before the failure is kept
    state.update(broke=False, sent=[])
    whole, hits = asyncio.run(rerun(tmp_path / "stores"))
    again = len(state["sent"])
    assert cached() == len(set(state["sent"])) + paid_unique  # kept: every text embedded so far
    state["sent"] = []
    clean, _ = asyncio.run(rerun(tmp_path / "clean"))
    assert whole == clean > partial and len(hits) == 5  # nothing doubled, nothing missing; it searches
    assert paid + again == cached()  # every text paid exactly once: the rerun paid only for what was never embedded
    assert len(state["sent"]) == 0  # and a clean encoding of the same texts elsewhere costs nothing (the cache is kept)


def _rerank_with(handler, tmp_path, monkeypatch, docs: int = 40, queries: int = 25):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rerankers, "SCORES", tmp_path / "scores")
    monkeypatch.setattr(rerankers, "SECRETS", tmp_path)
    monkeypatch.setattr(limits, "_LIMITS", {})
    monkeypatch.setattr(limits, "backoff", lambda attempt: 0.0)
    (tmp_path / "deepinfra_api_key").write_text("test")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))
    prompt = rerankers.prompt_key(rerankers.RERANKERS["q06"].template)
    work = [(f"question {q}?", [(rerankers.ScoreCache.key(prompt, "em", f"question {q}?", f"doc {q} {i}"),
                                 f"doc {q} {i}") for i in range(docs)]) for q in range(queries)]
    rerankers.score("q06", work, "test")
    return work


def ok(request: httpx.Request) -> httpx.Response:
    docs = json.loads(request.content)["documents"]
    return httpx.Response(200, json={"scores": [0.5] * len(docs), "input_tokens": 10 * len(docs)})


def test_the_limit_grows_on_success_and_shrinks_on_429(tmp_path, monkeypatch):
    seen = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(429, text="Rate limited") if seen["n"] % 7 == 3 else ok(request)

    work = _rerank_with(ok, tmp_path / "a", monkeypatch)
    grown = limits._LIMITS["deepinfra:Qwen/Qwen3-Reranker-0.6B"]
    assert grown.peak > limits.START and not grown.throttled_once  # 50 requests, no 429: it ramped up
    _rerank_with(flaky, tmp_path / "b", monkeypatch)
    hit = limits._LIMITS["deepinfra:Qwen/Qwen3-Reranker-0.6B"]
    assert hit.throttled_once and hit.cuts > 0  # cut on 429s, and every request still succeeded
    keys = [k for _, items in work for k, _ in items]
    assert len(rerankers.ScoreCache("q06").get(keys)) == len(keys)


def test_an_hour_of_failures_stops_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(limits, "PATIENCE", 0.05)
    with pytest.raises(SystemExit, match="has not answered reranking requests"):
        _rerank_with(lambda request: httpx.Response(503, text="overloaded"), tmp_path, monkeypatch, queries=2)
