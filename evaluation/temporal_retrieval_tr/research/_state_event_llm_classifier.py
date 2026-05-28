"""LLM-based state-shape classifier.

Production realistic: the doc-side extractor would emit an `is_state`
field per anchor as part of its existing schema. This module simulates
that with a one-shot LLM call per doc text, cached on disk so the A/B
test is cheap to re-run.

Prompt design follows the same conventions as the doc extractor v3.3:
- Concrete principle / decision test
- Explicit handling of edge cases (process vs state — both can have duration)
- Skip-don't-emit semantics: when in doubt, default to event
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from temporal_retrieval_min.extractor_common import _LLMCache

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

MODEL = "gpt-5-mini"
PROMPT_VERSION = "v1"
CACHE_ROOT = (
    Path(__file__).resolve().parent / "cache" / f"state_event_classifier_{PROMPT_VERSION}"
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = """You classify a passage's primary temporal claim as either
an EVENT-shape or a STATE-shape claim.

# The distinction

EVENT-shape: the passage describes something that HAPPENED — a
discrete occurrence, a process running, an action, an event with
duration. Examples:
- "We deployed v2 on March 15, 2024."  → EVENT (a deploy happened)
- "Conference from July 1 to July 5."  → EVENT (the conference was a
  happening that occupied those days)
- "I attended the offsite October 16-20."  → EVENT (the offsite is
  the happening; attendance is incidental)
- "Around April 8 I had a doctor's visit."  → EVENT (a visit happened
  somewhere in that window)
- "Tom completed his marathon training plan in May 2023." → EVENT

STATE-shape: the passage describes a CONDITION that held over a span
of time — a tenure, a residence, an ongoing arrangement, a role
held, a relationship. The passage's temporal claim is "this state
was in effect during this span," not "an event occurred." Examples:
- "I was married from 2015 to 2023." → STATE (marriage condition held)
- "Lived in NYC from 2018 to 2024." → STATE (residence)
- "Sarah was the chief legal officer from 2019 to 2024." → STATE (role)
- "Carla's research program has been running since 2019." → STATE
  (ongoing program-state)
- "I've been working with Aiden as his coach since 2020." → STATE
  (ongoing professional relationship)
- "Boston had been Priya's home for half a decade." → STATE

# The deciding test

Ask: "If I were asked 'what HAPPENED on a specific date covered by this
passage's temporal scope?' — does this passage answer that question?"

- If yes (the passage describes a happening that occupied or might
  have occupied that date) → EVENT
- If no (the passage only tells you a condition was active on that
  date, no specific happening claimed) → STATE

# Edge guidance

Some cases look state-syntactic but are event-shape:
- "I was at the conference from Jul 1 to Jul 5." → EVENT
  (attending the conference IS the happening — the conference was
  occurring throughout)
- "She was working on the project from March to May." → EVENT
  (the work was an ongoing process; activities happened throughout)

Some have explicit duration but are state-shape:
- "Her tenure as CFO spanned five years." → STATE
- "Marketing has been my function since 2020." → STATE

The distinguisher: "happenings" (events, processes) answer "what
happened?"; "holdings" (states, conditions, roles) answer "what was
true?". A doc can be either even with similar duration.

When ambiguous, default to EVENT (more permissive — captures
both event-locator and state-locator query matches).

# Output

A single JSON object: {"is_state": true|false}.

Set is_state=true ONLY when the passage's primary temporal claim is
a holding-condition / role / tenure / residence / relationship that
was in effect over the period — and does NOT describe a happening or
process.
"""


JSON_SCHEMA: dict[str, Any] = {
    "name": "state_event_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_state": {"type": "boolean"},
        },
        "required": ["is_state"],
        "additionalProperties": False,
    },
}


class LLMStateClassifier:
    def __init__(
        self,
        model: str = MODEL,
        client: AsyncOpenAI | None = None,
        cache_dir: Path | None = None,
        concurrency: int = 8,
    ) -> None:
        self.model = model
        self.client = client or AsyncOpenAI()
        cd = Path(cache_dir) if cache_dir else CACHE_ROOT
        self.cache = _LLMCache(cd / "is_state.json")
        self._sem = asyncio.Semaphore(concurrency)

    async def classify(self, text: str) -> bool:
        key = f"{PROMPT_VERSION}|{text}"
        cached = self.cache.get(self.model, key)
        if cached is None:
            async with self._sem:
                resp = await self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Passage:\n{text}"},
                    ],
                    text={"format": {"type": "json_schema", **JSON_SCHEMA}},
                )
                cached = resp.output_text
                self.cache.put(self.model, key, cached)
        try:
            data = json.loads(cached)
            return bool(data.get("is_state", False))
        except (json.JSONDecodeError, AttributeError, TypeError):
            return False  # default to event on parse failure

    async def classify_many(self, texts: list[str]) -> list[bool]:
        return await asyncio.gather(*(self.classify(t) for t in texts))

    def save(self) -> None:
        self.cache.save()


# Module-level singleton helper
_default_classifier: LLMStateClassifier | None = None


def get_classifier() -> LLMStateClassifier:
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = LLMStateClassifier()
    return _default_classifier


if __name__ == "__main__":
    # Stand-alone diagnostic: classify a few representative cases.
    cases = [
        "On March 15, 2024 we deployed v2 of the recommendation API.",
        "I was married from 2015 to 2023.",
        "I've been working on the recommendation system for the better part of three years now.",
        "Acme's been my employer since I graduated, around 2021.",
        "Marcus was on the marathon training team from 2022 to 2024.",
        "Eric attended the industry leadership conference June 5 to June 8, 2023.",
        "Mira's tenure as VP of Product spanned two CEO transitions.",
        "I was at the conference from July 1 to July 5.",
        "Held the CFO position 2020 through 2024.",
        "We launched the redesign on December 5, 2022.",
        "Hannah's PhD work consumed nearly five years of her life.",
        "Hannah turned in her thesis defense on September 12, 2022.",
    ]
    async def main():
        clf = LLMStateClassifier()
        results = await clf.classify_many(cases)
        clf.save()
        for c, r in zip(cases, results, strict=True):
            tag = "STATE" if r else "EVENT"
            print(f"  [{tag:5s}]  {c}")
    asyncio.run(main())
