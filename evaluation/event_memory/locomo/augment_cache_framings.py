"""Generate two extra index framings per cached terse-v2 segment.

The additive lattice showed M (3p statement), Q (questions) and C (1p raw
chunk) each add ~+0.65pp to text_to_embed -- multi-view embedding. The
open question: does a 4th framing on an axis M/Q/C do NOT cover add
more? M/Q/C span three registers; the uncovered axes are:

  atomic  -- GRANULARITY: decompose the memory into smallest standalone
             facts. Each sub-fact independently retrievable (multi-hop).
  topic   -- ABSTRACTION: concrete->abstract theme/domain/entity labels,
             no sentence. For thematic / open-domain queries.

First-person rewrite is deliberately skipped: C is already 1p, and the
"queries alone" diagnosis shows register-match to the (3p question)
query is what wins -- 1p is the wrong direction.

One LLM pass over the FIXED terse-v2 segmentation (no re-segmentation
noise), gpt-5-nano @ low (deriver-class model policy). Writes
cache-terse-v2-aug.json: every raw record + "atomic" and "topic" keys.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import openai
from dotenv import load_dotenv
from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from pydantic import BaseModel

RAW = "cache-terse-v2-raw.json"
OUT = "cache-terse-v2-aug.json"
CONCURRENCY = 150

PROMPT = """You are given ONE memory statement. Produce two alternative \
index representations of it. Do not add information not entailed by the \
statement.

(1) atomic -- break the statement into its smallest independent factual \
assertions. Each assertion is a complete standalone sentence stating \
exactly ONE fact, with every reference resolved so it stands alone. A \
statement holding a single fact yields a single-element list.

(2) topic -- a short comma-separated list of the themes, life domains \
and key entities the statement concerns, ordered concrete to abstract. \
Labels only, never a sentence.

Return JSON: {{"atomic": ["...", "..."], "topic": "..."}}

MEMORY STATEMENT:
{memory}"""


class _Framings(BaseModel):
    atomic: list[str]
    topic: str


async def _augment_one(
    lm: OpenAIResponsesLanguageModel,
    sem: asyncio.Semaphore,
    rec: dict,
) -> dict:
    memory = (rec.get("memory") or "").strip()
    if not memory:
        rec["atomic"] = []
        rec["topic"] = ""
        return rec
    async with sem:
        result = await lm.generate_parsed_response(
            output_format=_Framings,
            user_prompt=PROMPT.format(memory=memory),
            max_attempts=3,
        )
    if result is None:
        rec["atomic"] = [memory]
        rec["topic"] = ""
    else:
        rec["atomic"] = [a.strip() for a in result.atomic if a and a.strip()]
        rec["topic"] = result.topic.strip()
    return rec


async def main() -> None:
    load_dotenv()
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    lm = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=client, model="gpt-5-nano", reasoning_effort="low"
        )
    )
    with open(RAW) as f:
        records = json.load(f)
    print(f"augmenting {len(records)} records ...", flush=True)
    t0 = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)
    out = await asyncio.gather(
        *(_augment_one(lm, sem, rec) for rec in records)
    )
    with open(OUT, "w") as f:
        json.dump(out, f)
    empty = sum(1 for r in out if not r["atomic"])
    print(
        f"done in {time.time() - t0:.0f}s -- wrote {OUT} "
        f"({len(out)} records, {empty} with empty atomic)",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
