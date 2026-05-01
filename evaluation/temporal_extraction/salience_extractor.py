"""Per-doc cue salience extractor.

Given a document text (and optional v2'' structured times), produces a
4-channel salience vector summing to ~1.0 over:
  S = semantic     (intrinsic content relevance)
  T = temporal     (date-instant retrieval)
  L = lattice/axis (recurrence/cyclical pattern)
  E = era          (era-style coarse temporal block)

The vector is interpreted as P(doc retrieved via this channel | doc relevant).
At retrieval time, each channel's normalized score is multiplied by the
doc's salience for that channel before global weighting.

Uses gpt-5-mini with reasoning_effort=minimal. Hard 30s timeout per call.
On any failure, returns a default uniform-ish prior leaning S/T.

Cache: cache/salience/llm_cache.json (keyed by model + prompt hash).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parents[1] / ".env")

MODEL = "gpt-5-mini"
CACHE_DIR = ROOT / "cache" / "salience"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_cache.json"

PER_CALL_TIMEOUT_S = 30.0

DEFAULT_SALIENCE = {"S": 0.50, "T": 0.30, "L": 0.10, "E": 0.10}


SYSTEM_PROMPT = """You score how a document is most likely to be retrieved.

Given a passage, output a salience vector over 4 retrieval CHANNELS:

  S = SEMANTIC   — intrinsic topic/content relevance (entities, actions,
                   topics; would match queries about the same subject).
  T = TEMPORAL   — concrete date / month / day / year (would match queries
                   anchored to a specific date or near-date, e.g.
                   "what happened on March 15, 2024?").
  L = LATTICE    — recurrence / cyclical / axis-only pattern (every X,
                   weekly, Tuesdays, summers, mornings — would match
                   queries about regular patterns).
  E = ERA        — era / decade / era-of-life reference (the 90s, "back
                   in college", "in my twenties", "during the pandemic"
                   — coarse temporal block that does NOT pin to a date).

Output JSON exactly: {"S": float, "T": float, "L": float, "E": float}
Each in [0, 1]. They should sum approximately to 1.0 (we will normalize).

If a channel does NOT apply to the doc, set it to 0 (or near-0). A doc
with no temporal cue at all should give T near 0.

Hypothetical / fictional / quoted-embedded times do NOT count as
real temporal cues — those should give T ~ 0.

Examples:

Passage: "On March 15, 2024, I had dinner with Sarah at Gusto."
Output: {"S": 0.40, "T": 0.40, "L": 0.15, "E": 0.05}

Passage: "I love hiking in the mountains."
Output: {"S": 0.85, "T": 0.05, "L": 0.05, "E": 0.05}

Passage: "Every Thursday I do tennis lessons at 6pm."
Output: {"S": 0.20, "T": 0.10, "L": 0.65, "E": 0.05}

Passage: "Back in the 90s we used to spend summers in Maine."
Output: {"S": 0.20, "T": 0.05, "L": 0.05, "E": 0.70}

Passage: "Yesterday I picked up the package from the post office."
Output: {"S": 0.50, "T": 0.40, "L": 0.05, "E": 0.05}

Passage: "Vincent Ostrom works for Indiana University Bloomington from Jan, 1964 to Jan, 1990."
Output: {"S": 0.45, "T": 0.45, "L": 0.10, "E": 0.00}

Passage: "What if I had been born in 1980? How different things would be."
Output: {"S": 0.95, "T": 0.00, "L": 0.00, "E": 0.05}

Output JSON only, no commentary.
"""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class _Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        if path.exists():
            try:
                self._cache = json.loads(path.read_text())
            except json.JSONDecodeError:
                self._cache = {}
        self._dirty = False

    @staticmethod
    def _key(model: str, prompt_key: str) -> str:
        return hashlib.sha256(f"{model}|{prompt_key}".encode()).hexdigest()

    def get(self, model: str, prompt_key: str) -> str | None:
        return self._cache.get(self._key(model, prompt_key))

    def put(self, model: str, prompt_key: str, value: str) -> None:
        k = self._key(model, prompt_key)
        if self._cache.get(k) != value:
            self._cache[k] = value
            self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._cache))
        tmp.replace(self.path)
        self._dirty = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _normalize(s: dict[str, float]) -> dict[str, float]:
    """Clamp to [0,1] and renormalize to sum=1 across S/T/L/E."""
    out = {k: max(0.0, float(s.get(k, 0.0))) for k in ("S", "T", "L", "E")}
    total = sum(out.values())
    if total <= 1e-9:
        return dict(DEFAULT_SALIENCE)
    return {k: v / total for k, v in out.items()}


def _parse_response(raw: str) -> dict[str, float] | None:
    raw = raw.strip()
    # Strip code-fences if present
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Try last-resort regex
        m = re.search(r"\{[^{}]*\}", raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    out = {}
    for k in ("S", "T", "L", "E"):
        v = obj.get(k, obj.get(k.lower()))
        if v is None:
            v = 0.0
        try:
            out[k] = float(v)
        except (ValueError, TypeError):
            out[k] = 0.0
    return _normalize(out)


class SalienceExtractor:
    """Per-doc salience vector extractor using gpt-5-mini."""

    def __init__(
        self,
        concurrency: int = 8,
        model: str = MODEL,
        cache_path: Path = CACHE_FILE,
    ) -> None:
        self.client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S, max_retries=1)
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = _Cache(cache_path)
        self.model = model
        self.usage = {"input": 0, "output": 0}
        self.n_failed = 0
        self.n_timeout = 0

    async def _llm_call(self, system: str, user: str) -> str:
        async with self.sem:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_completion_tokens=200,
                    response_format={"type": "json_object"},
                    reasoning_effort="minimal",
                )
            except Exception:
                raise
        u = resp.usage
        if u:
            self.usage["input"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["output"] += getattr(u, "completion_tokens", 0) or 0
        return resp.choices[0].message.content or ""

    async def extract_one(
        self, text: str, doc_id: str | None = None
    ) -> dict[str, float]:
        user = f'Passage: "{text}"\n\nOutput the salience JSON.'
        prompt_key = user
        cached = self.cache.get(self.model, prompt_key)
        if cached is not None:
            parsed = _parse_response(cached)
            if parsed is not None:
                return parsed
        try:
            raw = await asyncio.wait_for(
                self._llm_call(SYSTEM_PROMPT, user),
                timeout=PER_CALL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            self.n_timeout += 1
            return dict(DEFAULT_SALIENCE)
        except Exception:
            self.n_failed += 1
            return dict(DEFAULT_SALIENCE)
        parsed = _parse_response(raw)
        if parsed is None:
            self.n_failed += 1
            return dict(DEFAULT_SALIENCE)
        self.cache.put(self.model, prompt_key, raw)
        return parsed

    async def extract_many(
        self,
        items: list[tuple[str, str]],  # list of (doc_id, text)
        progress_every: int = 100,
    ) -> dict[str, dict[str, float]]:
        """Batch extract. items=[(doc_id, text), ...]. Returns {doc_id: vec}."""
        results: dict[str, dict[str, float]] = {}
        completed = [0]
        total = len(items)

        async def one(did: str, txt: str):
            sal = await self.extract_one(txt, did)
            results[did] = sal
            completed[0] += 1
            if progress_every and completed[0] % progress_every == 0:
                print(
                    f"  salience: {completed[0]}/{total} "
                    f"(timeout={self.n_timeout}, fail={self.n_failed})",
                    flush=True,
                )

        await asyncio.gather(*(one(did, txt) for did, txt in items))
        self.cache.save()
        return results


# ---------------------------------------------------------------------------
# Convenience: run on a doc list
# ---------------------------------------------------------------------------
async def extract_salience(
    docs: list[dict],
    text_key: str = "text",
    id_key: str = "doc_id",
    concurrency: int = 8,
) -> dict[str, dict[str, float]]:
    ex = SalienceExtractor(concurrency=concurrency)
    items = [(d[id_key], d[text_key]) for d in docs]
    return await ex.extract_many(items)
