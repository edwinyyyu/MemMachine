"""Overlapped prepare and score (pipeline.prepare_and_score): with a hosted embedder and a
hosted reranker (both replaced by in-process transports), the reranker scores each history's
pairs while later histories embed, the measured depths are saved as `depth` would save them,
nothing needed is left unscored, and nothing is scored twice."""

import hashlib
import json

import httpx
import numpy as np
import openai
import pytest

from rbench import depth, limits, memory, pipeline, rerankers, simulation
from rbench.grid import Run
from rbench.paths import PREPARED

REAL_CLIENT = httpx.AsyncClient


@pytest.mark.skipif(not (PREPARED / "longmemeval_s").exists(), reason="needs LongMemEval-S prepared")
def test_reranking_overlaps_ingest_and_completes(tmp_path, monkeypatch):
    for module in (memory, rerankers):
        monkeypatch.setattr(module, "SECRETS", tmp_path)
    (tmp_path / "deepinfra_api_key").write_text("test")
    monkeypatch.setattr(memory, "MEMORY", tmp_path / "memory")
    monkeypatch.setattr(memory, "EMBEDDINGS", tmp_path / "embeddings")
    monkeypatch.setattr(memory, "questions_path", lambda e: tmp_path / "questions.jsonl")
    monkeypatch.setattr(rerankers, "SCORES", tmp_path / "scores")
    monkeypatch.setattr(depth, "DEPTH", tmp_path / "depth")
    monkeypatch.setattr(limits, "_LIMITS", {})
    events: list[str] = []  # the order things happened in: "embed" or "rerank"

    def embed(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        events.append("embed")
        n = body.get("dimensions") or 1024
        data = [{"object": "embedding", "index": i,
                 "embedding": np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)).normal(size=n).tolist()}
                for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"],
                                         "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    sent: list[str] = []

    def rerank(request: httpx.Request) -> httpx.Response:
        docs = json.loads(request.content)["documents"]
        events.append("rerank")
        sent.extend(docs)
        return httpx.Response(200, json={"scores": [0.5] * len(docs), "input_tokens": 10 * len(docs)})

    class Routed(REAL_CLIENT):  # one client type for both APIs (openai checks isinstance against httpx.AsyncClient)
        def __init__(self, **kw):
            route = lambda r: embed(r) if r.url.path.endswith("/embeddings") else rerank(r)  # noqa: E731
            super().__init__(transport=httpx.MockTransport(route), **kw)

    real_openai = openai.AsyncOpenAI
    monkeypatch.setattr(httpx, "AsyncClient", Routed)
    monkeypatch.setattr(memory.openai, "AsyncOpenAI", lambda **kw: real_openai(http_client=Routed(), **kw))

    embedder = "qwen3-8b-di"
    runs = [Run("longmemeval_s", "history", "question", ("seg500",), ("q06",), sample="tiny", expand=e, pool_seeds=12,
                embedder=embedder) for e in (0, 3)]
    lines = pipeline.prepare_and_score(runs, embedder)

    first_rerank = events.index("rerank")
    assert "embed" in events[first_rerank:]  # scoring started before the last history was embedded
    assert len(lines) == 2 and all("scored while embedding" in x and "0 past the measured pool" in x for x in lines)
    assert depth.path(embedder, "longmemeval_s", "seg500", "tiny").exists()  # the depths, measured in the loop
    for run in runs:  # nothing needed is left: a sequential rerank finds no work
        assert simulation.scoring_work(embedder, "longmemeval_s", "seg500", "tiny", "q06", run.expand, 12) == []
    calls = rerankers.ScoreCache("q06").db.execute("SELECT SUM(pairs) FROM calls").fetchone()[0]
    keys = rerankers.ScoreCache("q06").db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    assert calls == keys  # every pair scored once
