"""Shared utilities for advanced experiments (E1-E4).

- separate LLM + embedding caches under cache/advanced/
- uses the same gpt-5-mini + text-embedding-3-small models
- provides minimal async helpers that do not collide with the running
  ablation study's files.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "advanced"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
LLM_CACHE_FILE = CACHE_DIR / "llm_cache.json"
EMBED_CACHE_FILE = CACHE_DIR / "embedding_cache.json"

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class JSONCache:
    """Simple persistent JSON dict cache, same pattern as baseline code."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: dict[str, Any] = {}
        if path.exists():
            try:
                with path.open() as f:
                    self._cache = json.load(f)
            except json.JSONDecodeError:
                self._cache = {}
        self._new: dict[str, Any] = {}

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def get(self, k: str) -> Any | None:
        return self._cache.get(k)

    def put(self, k: str, v: Any) -> None:
        self._cache[k] = v
        self._new[k] = v

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, Any] = {}
        if self.path.exists():
            try:
                with self.path.open() as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        existing.update(self._new)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(existing, f)
        tmp.replace(self.path)
        self._new.clear()


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
class LLMCaller:
    """Thin wrapper around AsyncOpenAI.chat.completions with caching."""

    def __init__(self, concurrency: int = 10) -> None:
        self.client = AsyncOpenAI()
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = JSONCache(LLM_CACHE_FILE)
        self.usage: dict[str, int] = {"input": 0, "output": 0}

    async def chat(
        self,
        system: str,
        user: str,
        *,
        model: str = MODEL,
        json_object: bool = False,
        json_schema: dict | None = None,
        max_completion_tokens: int = 1500,
        cache_tag: str = "",
    ) -> str:
        pkey = JSONCache.key(
            model,
            cache_tag,
            hashlib.sha256(system.encode()).hexdigest()[:16],
            user,
        )
        cached = self.cache.get(pkey)
        if cached is not None:
            return cached
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_completion_tokens,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        elif json_object:
            kwargs["response_format"] = {"type": "json_object"}
        # Try once; on token-limit or transient error, double budget and
        # retry up to twice. Cache only successful parseable content.
        current_tokens = max_completion_tokens
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                kwargs["max_completion_tokens"] = current_tokens
                async with self.sem:
                    resp = await self.client.chat.completions.create(**kwargs)
                if resp.usage:
                    self.usage["input"] += getattr(resp.usage, "prompt_tokens", 0) or 0
                    self.usage["output"] += (
                        getattr(resp.usage, "completion_tokens", 0) or 0
                    )
                content = resp.choices[0].message.content or ""
                if content:
                    self.cache.put(pkey, content)
                    return content
                # Empty output => bump and retry
                current_tokens = min(current_tokens * 2, 8000)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "max_tokens" in msg or "max tokens" in msg or "output limit" in msg:
                    current_tokens = min(current_tokens * 2, 8000)
                    continue
                await asyncio.sleep(0.5 * (attempt + 1))
        # Fallback: return empty to let caller JSON-decode fail gracefully
        if last_err is not None:
            print(f"  LLM call failed after retries: {last_err}")
        return ""

    def save(self) -> None:
        self.cache.save()

    def cost_usd(self) -> float:
        # gpt-5-mini: $0.25/M in, $2.00/M out
        return (
            self.usage["input"] * 0.25 / 1_000_000
            + self.usage["output"] * 2.0 / 1_000_000
        )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, concurrency: int = 10) -> None:
        self.client = AsyncOpenAI()
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = JSONCache(EMBED_CACHE_FILE)
        self.count = 0

    async def embed(self, text: str) -> np.ndarray:
        k = JSONCache.key(EMBED_MODEL, text)
        cached = self.cache.get(k)
        if cached is not None:
            return np.array(cached, dtype=np.float32)
        async with self.sem:
            resp = await self.client.embeddings.create(model=EMBED_MODEL, input=text)
        v = resp.data[0].embedding
        self.cache.put(k, v)
        self.count += 1
        return np.array(v, dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return list(await asyncio.gather(*(self.embed(t) for t in texts)))

    def save(self) -> None:
        self.cache.save()

    def cost_usd(self) -> float:
        # text-embedding-3-small: $0.02/M tokens. We don't track exact tokens
        # here; rough estimate 8 tokens per short phrase. Upstream caller can
        # override. Treat as negligible in our budget.
        return 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return float("nan")
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def mean(vs: list[float]) -> float:
    xs = [v for v in vs if not math.isnan(v)]
    return sum(xs) / len(xs) if xs else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) or 1e-9
    return float(np.dot(a, b) / denom)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out
