"""Running out of DeepInfra credits mid-run: a clean stop with MESSAGE, nothing finished lost,
and a rerun completes. The API is replaced by an in-process transport (no network, no key)."""

import asyncio
import datetime
import json
import random
import uuid

import httpx
import numpy as np
import openai
import pytest

from rbench import credits, memory, rerankers
from rbench.memory import Memory

from memmachine_server.episodic_memory.event_memory.data_types import Event, ProducerContext, TextBlock  # noqa: I001

BROKE = httpx.Response(402, json={"error": {"message": "Insufficient balance"}})


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
    (tmp_path / "deepinfra_api_key").write_text("test")
    state = {"broke": True}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if state["broke"] and any("SENTINEL" in t for t in body["input"]):
            return BROKE
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

    async def rerun(root) -> tuple[int, list]:
        async with Memory(embedder, "synthetic", "seg80", root=root) as m:
            assert await m.ingest("h", events)
            assert m.done("h").exists()
            partition = await (await m.event_memory()).open_or_create_partition("h")
            hits = (await partition.query("the price of the tickets", vector_search_limit=5)).scored_segment_contexts
            return await segments(m, "h"), hits

    partial = asyncio.run(broke())
    assert partial > 0  # the first batch was written before the failure: a half-written partition
    state["broke"] = False
    whole, hits = asyncio.run(rerun(tmp_path / "stores"))
    clean, _ = asyncio.run(rerun(tmp_path / "clean"))
    assert whole == clean > partial and len(hits) == 5  # nothing doubled, nothing missing; it searches
