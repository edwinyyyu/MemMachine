"""Shared infrastructure for update-commands research rounds.

Keeps everything standalone (no attribute_memory framework imports).
Loads key from evaluation/.env, caches LLM calls keyed on (model, prompt),
tracks a running call count + cost estimate, and provides the sample-set loader.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parents[2]  # .../evaluation
load_dotenv(EVAL_ROOT / ".env")

MODEL = "gpt-5-mini"
# gpt-5-mini pricing (approx, $ per 1M tokens): ~$0.25 input / ~$2.00 output
# Rough per-call upper bound: 2k in + 1k out = ~$0.0025
# We'll track call count and save after each round.
PRICE_PER_CALL_APPROX = 0.0025
BUDGET_MAX_CALLS = 150
BUDGET_STOP_AT = int(BUDGET_MAX_CALLS * 0.80)  # stop at 80% = 120

CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"
SCENARIOS_PATH = HERE / "scenarios.json"


def _sha(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


class LLMCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._d: dict[str, str] = {}
        if path.exists():
            try:
                with open(path) as f:
                    self._d = json.load(f)
            except Exception:
                self._d = {}
        self._dirty = False
        self._hits = 0
        self._misses = 0

    def get(self, model: str, prompt: str) -> str | None:
        v = self._d.get(_sha(model, prompt))
        if v is not None:
            self._hits += 1
        return v

    def put(self, model: str, prompt: str, response: str) -> None:
        self._d[_sha(model, prompt)] = response
        self._dirty = True
        self._misses += 1

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._d)}

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._d, f)
        tmp.replace(self._path)
        self._dirty = False


@dataclass
class CallBudget:
    made: int = 0
    max_calls: int = BUDGET_MAX_CALLS
    stop_at: int = BUDGET_STOP_AT

    def check(self) -> None:
        if self.made >= self.stop_at:
            raise RuntimeError(
                f"Budget stop hit: {self.made}/{self.max_calls} LLM calls "
                f"(80% cap = {self.stop_at}). Halting."
            )

    def tick(self) -> None:
        self.made += 1
        self.check()

    def approx_cost(self) -> float:
        return self.made * PRICE_PER_CALL_APPROX


def load_scenarios() -> list[dict[str, Any]]:
    with open(SCENARIOS_PATH) as f:
        data = json.load(f)
    return data["scenarios"]


def make_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def llm_call(
    client: openai.OpenAI,
    cache: LLMCache,
    budget: CallBudget,
    prompt: str,
    model: str = MODEL,
    reasoning_effort: str = "low",
) -> str:
    """Synchronous cached LLM call. Budget is only ticked on a real miss."""
    cached = cache.get(model, prompt)
    if cached is not None:
        return cached
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=reasoning_effort,
    )
    text = resp.choices[0].message.content or ""
    cache.put(model, prompt, text)
    budget.tick()
    return text
