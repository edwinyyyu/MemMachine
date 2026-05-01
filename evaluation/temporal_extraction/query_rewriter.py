"""F11 — Temporal query rewriter.

Given a user query with temporal content, ask gpt-5-mini to produce K=3-5
paraphrase variants that preserve meaning but vary:

- phrasing register (formal vs casual)
- granularity (day / week / month / year)
- absolute vs relative anchoring
- cultural / era references where applicable

The prompt is deliberately domain-neutral and uses generic few-shot
examples so that it does not leak phrasings specific to any existing
synthetic corpus.

Results are cached at ``cache/rewrite/rewrite_cache.json`` keyed by
(model, prompt-hash, query-text, ref-time). Each call is bounded to 30s.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT.parents[1] / ".env")

MODEL = "gpt-5-mini"
CACHE_DIR = ROOT / "cache" / "rewrite"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "rewrite_cache.json"

CALL_TIMEOUT_SEC = 30.0
# gpt-5-mini token prices
PRICE_IN_PER_M = 0.25
PRICE_OUT_PER_M = 2.00


REWRITER_SYSTEM = """You are a careful query paraphraser that specialises
in temporal expressions. Given one user query that contains or implies
time references, produce between 3 and 5 alternative phrasings that
preserve meaning but vary HOW the time is expressed.

Vary along these orthogonal axes, picking different combinations across
the variants:

- phrasing register: formal vs casual vs terse.
- granularity: same slice can be described at day / week / month /
  quarter / year resolution ("March 15, 2026" <-> "mid-March 2026"
  <-> "Q1 2026").
- absolute vs relative anchoring: "in 2024" vs "two years ago" vs "the
  year before last". You may use the provided reference time to produce
  correct absolute equivalents of relative surfaces.
- cultural / era references where applicable ("the 90s" <-> "the
  nineties" <-> "late 20th century" <-> "roughly 1990-1999").

Hard rules:

1. Preserve the original intent exactly. Do not add or remove events,
   entities, or conditions. Only the time surface varies.
2. No duplicates (case-insensitive). No trivial tweaks ("the 90s" vs
   "the 90's" is a duplicate).
3. If the query has NO temporal content at all, return the original
   query as the single element of the list.
4. Keep each variant under 120 characters and as natural English.
5. Output STRICT JSON: {"variants": ["...", "...", ...]}. Nothing else.

Examples (domain-neutral):

Query: "what did I do 2 years ago?"  (ref: 2026-04-23)
{"variants": ["what did I do in 2024?", "what happened around the end of 2023 or start of 2024?", "any memories from a couple years back?", "what was I up to during the year before last?"]}

Query: "anything in March?"
{"variants": ["any events in March across any year?", "what's on for March?", "things that happened in March", "Q1-spring items falling in March"]}

Query: "the 90s"
{"variants": ["1990 to 1999", "the decade of the nineties", "late 20th century, roughly 1990-1999", "events from the nineties"]}

Query: "what happened last Thursday?"  (ref: 2026-04-23, Thursday)
{"variants": ["what happened on Thursday, April 16, 2026?", "what was that event a week ago yesterday?", "anything from the previous Thursday", "events seven days before today"]}

Query: "who is my dentist?"
{"variants": ["who is my dentist?"]}
"""

REWRITE_SCHEMA: dict[str, Any] = {
    "name": "temporal_rewrite",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["variants"],
    },
}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class RewriteCache:
    def __init__(self, path: Path = CACHE_FILE) -> None:
        self.path = path
        self._cache: dict[str, Any] = {}
        if path.exists():
            try:
                self._cache = json.loads(path.read_text())
            except json.JSONDecodeError:
                self._cache = {}
        self._new: dict[str, Any] = {}

    @staticmethod
    def _key(model: str, system_hash: str, query: str, ref_time: str) -> str:
        return hashlib.sha256(
            f"{model}|{system_hash}|{query}|{ref_time}".encode()
        ).hexdigest()

    def get(
        self, model: str, system: str, query: str, ref_time: str
    ) -> list[str] | None:
        system_hash = hashlib.sha256(system.encode()).hexdigest()[:16]
        return self._cache.get(self._key(model, system_hash, query, ref_time))

    def put(
        self, model: str, system: str, query: str, ref_time: str, variants: list[str]
    ) -> None:
        system_hash = hashlib.sha256(system.encode()).hexdigest()[:16]
        k = self._key(model, system_hash, query, ref_time)
        self._cache[k] = variants
        self._new[k] = variants

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, Any] = {}
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing.update(self._new)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=2))
        tmp.replace(self.path)
        self._new.clear()


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------
class QueryRewriter:
    def __init__(self, concurrency: int = 10) -> None:
        self.client = AsyncOpenAI()
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = RewriteCache()
        self.usage: dict[str, int] = {"input": 0, "output": 0}

    async def rewrite(self, query: str, ref_time_iso: str) -> list[str]:
        cached = self.cache.get(MODEL, REWRITER_SYSTEM, query, ref_time_iso)
        if cached is not None:
            return cached

        user = f'Reference time: {ref_time_iso}\nQuery: "{query}"\n\nReturn JSON.'
        messages = [
            {"role": "system", "content": REWRITER_SYSTEM},
            {"role": "user", "content": user},
        ]
        # NOTE on determinism: gpt-5-mini does not accept temperature != 1, so
        # we cannot pass temperature=0 directly. Determinism across re-runs is
        # instead guaranteed by our (model, system_hash, query, ref_time)
        # cache: identical inputs return the cached variants. For first-run
        # samples, response_format=json_schema + a rigid prompt already yields
        # near-deterministic output in practice.
        # reasoning_effort="low" trims reasoning tokens so the visible content
        # budget is actually spent on JSON, not reasoning.
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "max_completion_tokens": 2500,
            "response_format": {"type": "json_schema", "json_schema": REWRITE_SCHEMA},
            "reasoning_effort": "low",
        }
        try:
            async with self.sem:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(**kwargs),
                    timeout=CALL_TIMEOUT_SEC,
                )
        except asyncio.TimeoutError:
            print(f"  rewrite TIMEOUT on {query!r}")
            return [query]
        except Exception as e:
            # Older SDKs / deployments may not accept reasoning_effort. Retry
            # without it.
            if "reasoning_effort" in str(e).lower() or "unsupported" in str(e).lower():
                kwargs.pop("reasoning_effort", None)
                try:
                    async with self.sem:
                        resp = await asyncio.wait_for(
                            self.client.chat.completions.create(**kwargs),
                            timeout=CALL_TIMEOUT_SEC,
                        )
                except asyncio.TimeoutError:
                    print(f"  rewrite TIMEOUT on {query!r}")
                    return [query]
                except Exception as e2:
                    print(f"  rewrite failed on {query!r}: {e2}")
                    return [query]
            else:
                print(f"  rewrite failed on {query!r}: {e}")
                return [query]

        if resp.usage:
            self.usage["input"] += getattr(resp.usage, "prompt_tokens", 0) or 0
            self.usage["output"] += getattr(resp.usage, "completion_tokens", 0) or 0
        content = resp.choices[0].message.content or ""
        variants: list[str] = []
        try:
            data = json.loads(content)
            raw_variants = data.get("variants") or []
            seen_lower: set[str] = set()
            for v in raw_variants:
                if not isinstance(v, str):
                    continue
                v = v.strip()
                if not v:
                    continue
                key = v.lower()
                if key in seen_lower:
                    continue
                seen_lower.add(key)
                variants.append(v)
            # Clamp to 5 variants
            variants = variants[:5]
        except json.JSONDecodeError:
            variants = []

        if not variants:
            variants = [query]
        self.cache.put(MODEL, REWRITER_SYSTEM, query, ref_time_iso, variants)
        return variants

    async def rewrite_many(
        self, items: list[tuple[str, str, str]]
    ) -> dict[str, list[str]]:
        """items: (qid, query_text, ref_time_iso). Returns qid -> variants."""

        async def one(qid: str, q: str, ref: str):
            vs = await self.rewrite(q, ref)
            return qid, vs

        results = await asyncio.gather(*(one(*it) for it in items))
        self.cache.save()
        return {qid: vs for qid, vs in results}

    def cost_usd(self) -> float:
        return (
            self.usage["input"] * PRICE_IN_PER_M / 1_000_000
            + self.usage["output"] * PRICE_OUT_PER_M / 1_000_000
        )
