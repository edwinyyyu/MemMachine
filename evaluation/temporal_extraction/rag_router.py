"""LLM query-intent classifier for RAG routing.

Given a natural-language query, classify it into a subset of intents:
- temporal: concrete dates, months, years, relative times (last week, 2026)
- semantic: pure topic / entity (no time signal)
- relational: before/after/during/overlaps/contains another event
- era: named era ("during college", "the 90s", "post-COVID")
- mixed: multiple of the above

Uses gpt-5-mini JSON output. Cached. 30-second per-call timeout.
"""

from __future__ import annotations

import asyncio
import json

from advanced_common import LLMCaller

ROUTER_SYSTEM = """You classify a user query into temporal-retrieval intents.

Output JSON: {"intents": ["temporal" | "semantic" | "relational" | "era"]}.
You may output 1 to 3 intents (in priority order).

Definitions:
- temporal: query has a concrete/relative date or time (e.g., "March 15,
  2026", "last month", "yesterday", "in 2023", "on Thursdays").
  Also quarters/seasons ("Q2 2024", "summer 2025").
- relational: query asks about events before/after/during/overlapping another
  event, e.g., "before my wedding", "after graduation", "during the move".
- era: query uses a named era/personal-life-chapter WITHOUT a concrete
  date anchor, e.g., "during college", "the 90s", "the Obama years",
  "in my 20s", "post-COVID". If BOTH temporal and era fire, prefer era.
- semantic: no time signal at all — pure topic/entity question.

Guidelines:
- If the query names a decade or named historical era -> "era" (not temporal).
- If the query has both a concrete date AND a relational cue -> [relational, temporal].
- If the query has none of the above -> ["semantic"].
- Prefer the most specific intent FIRST.

Return STRICTLY JSON: {"intents": ["..."]}.
"""


class RagRouter:
    def __init__(self, llm: LLMCaller) -> None:
        self.llm = llm
        self.calls = 0

    async def classify(self, query: str) -> list[str]:
        user = f'Query: "{query}"\nReturn {{"intents": [...]}}'
        try:
            raw = await asyncio.wait_for(
                self.llm.chat(
                    ROUTER_SYSTEM,
                    user,
                    json_object=True,
                    max_completion_tokens=200,
                    cache_tag="rag_router_v1",
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            return ["semantic"]
        self.calls += 1
        if not raw:
            return ["semantic"]
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            return ["semantic"]
        intents = d.get("intents") or []
        out = [i for i in intents if i in {"temporal", "semantic", "relational", "era"}]
        return out or ["semantic"]

    async def classify_all(
        self, queries: list[tuple[str, str]]
    ) -> dict[str, list[str]]:
        """queries: list of (qid, text). Returns qid -> intents."""

        async def one(qid: str, text: str) -> tuple[str, list[str]]:
            return qid, await self.classify(text)

        results = await asyncio.gather(*(one(*q) for q in queries))
        return dict(results)
