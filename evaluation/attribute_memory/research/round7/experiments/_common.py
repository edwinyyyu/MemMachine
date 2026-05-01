"""Shared infra for round7 experiments: LLM/embedding cache, budget, client."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path

import openai
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROUND7 = HERE.parent
EVAL_ROOT = HERE.parents[3]
CACHE_DIR = ROUND7 / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(EVAL_ROOT / ".env")


MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"
PRICE_LLM = 0.003
PRICE_EMBED = 0.00002


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


class Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._d: dict = {}
        if path.exists():
            try:
                self._d = json.loads(path.read_text())
            except Exception:
                self._d = {}
        self._dirty = False
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        v = self._d.get(key)
        if v is not None:
            self.hits += 1
        return v

    def put(self, key: str, value) -> None:
        self._d[key] = value
        self._dirty = True
        self.misses += 1

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._d))
        tmp.replace(self.path)
        self._dirty = False


class Budget:
    def __init__(
        self,
        max_llm: int = 350,
        max_embed: int = 100,
        stop_at_llm: int = 280,
        stop_at_embed: int = 80,
    ) -> None:
        self.max_llm = max_llm
        self.max_embed = max_embed
        self.stop_at_llm = stop_at_llm
        self.stop_at_embed = stop_at_embed
        self.llm_calls = 0
        self.embed_calls = 0

    def tick_llm(self) -> None:
        self.llm_calls += 1
        if self.llm_calls >= self.stop_at_llm:
            raise RuntimeError(
                f"Budget stop: {self.llm_calls}/{self.max_llm} LLM calls"
            )

    def tick_embed(self) -> None:
        self.embed_calls += 1
        if self.embed_calls >= self.stop_at_embed:
            raise RuntimeError(
                f"Budget stop: {self.embed_calls}/{self.max_embed} embed calls"
            )

    def cost(self) -> float:
        return self.llm_calls * PRICE_LLM + self.embed_calls * PRICE_EMBED


_client: openai.OpenAI | None = None


def client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def llm(
    prompt: str,
    cache: Cache,
    budget: Budget,
    reasoning_effort: str = "low",
    model: str = MODEL,
) -> str:
    key = _sha(model, reasoning_effort, prompt)
    cached = cache.get(key)
    if cached is not None:
        return cached
    resp = client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=reasoning_effort,
    )
    text = resp.choices[0].message.content or ""
    cache.put(key, text)
    budget.tick_llm()
    return text


def embed_batch(
    texts: Iterable[str], cache: Cache, budget: Budget
) -> list[list[float]]:
    texts = list(texts)
    out: list[list[float] | None] = [None] * len(texts)
    to_fetch_idx: list[int] = []
    to_fetch_text: list[str] = []
    for i, t in enumerate(texts):
        key = _sha(EMBED_MODEL, t)
        c = cache.get(key)
        if c is not None:
            out[i] = c
        else:
            to_fetch_idx.append(i)
            to_fetch_text.append(t)
    if to_fetch_text:
        resp = client().embeddings.create(model=EMBED_MODEL, input=to_fetch_text)
        budget.tick_embed()
        for idx, datum in zip(to_fetch_idx, resp.data, strict=True):
            vec = list(datum.embedding)
            out[idx] = vec
            cache.put(_sha(EMBED_MODEL, texts[idx]), vec)
    assert all(v is not None for v in out)
    return [v for v in out]  # type: ignore


def cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


def extract_json(text: str):
    """Extract JSON from an LLM reply (handles ``` fences)."""
    import re

    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\[.*\]|\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
