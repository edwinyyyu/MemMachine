"""Semantic-only baseline: text-embedding-3-small cosine rank."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

EMBED_MODEL = "text-embedding-3-small"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
EMBED_CACHE_FILE = CACHE_DIR / "embedding_cache.json"


class EmbeddingCache:
    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}
        if EMBED_CACHE_FILE.exists():
            with EMBED_CACHE_FILE.open() as f:
                self._cache = json.load(f)
        self._new: dict[str, list[float]] = {}

    @staticmethod
    def _key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}|{text}".encode()).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        return self._cache.get(self._key(model, text))

    def put(self, model: str, text: str, emb: list[float]) -> None:
        k = self._key(model, text)
        self._cache[k] = emb
        self._new[k] = emb

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, list[float]] = {}
        if EMBED_CACHE_FILE.exists():
            with EMBED_CACHE_FILE.open() as f:
                existing = json.load(f)
        existing.update(self._new)
        tmp = EMBED_CACHE_FILE.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(existing, f)
        tmp.replace(EMBED_CACHE_FILE)
        self._new.clear()


async def _embed_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    cache: EmbeddingCache,
    text: str,
) -> np.ndarray:
    cached = cache.get(EMBED_MODEL, text)
    if cached is not None:
        return np.array(cached)
    async with sem:
        resp = await client.embeddings.create(model=EMBED_MODEL, input=text)
    v = resp.data[0].embedding
    cache.put(EMBED_MODEL, text, v)
    return np.array(v)


async def embed_all(texts: list[str], concurrency: int = 10) -> list[np.ndarray]:
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    cache = EmbeddingCache()
    out = await asyncio.gather(*(_embed_one(client, sem, cache, t) for t in texts))
    cache.save()
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def semantic_rank(
    query_emb: np.ndarray,
    doc_embs: dict[str, np.ndarray],
) -> list[tuple[str, float]]:
    return sorted(
        ((d, cosine(query_emb, v)) for d, v in doc_embs.items()),
        key=lambda x: x[1],
        reverse=True,
    )
