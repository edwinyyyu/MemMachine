"""Manual quality inspection for the sentence-level prompt.

Picks ~10 representative messages from c2sub spanning the categories
that matter for the principle:
  - Pure social (greeting, congrats, reaction)
  - Pure fact (allergy declaration, event report)
  - Mixed (greeting + fact)
  - Question with content vs question without
  - Multi-fact sentence
  - Anaphora-heavy

Prints LLM outputs for human eyeballing. NO bench accuracy measured.

Usage:
  uv run python eyeball_sentence_prompt.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent / "longmemeval/llm_pipeline_probe"
    ),
)

from openai import AsyncOpenAI
from dotenv import load_dotenv
from memmachine_server.common.language_model.openai_chat_completions_language_model import (
    OpenAIChatCompletionsLanguageModel,
    OpenAIChatCompletionsLanguageModelParams,
)
from probe_terse_decoupled_slim_v3_sentence import (
    PROMPT_SENTENCE,
    _RewriteResponse,
)


# Hand-picked representative messages.
# Each entry: (label, speaker, date, message, [prior_turns_for_context]).
PRIOR_GREETING_EXCHANGE = [
    ("Nate", "What's up, Joanna? Anything fun going on?"),
]

TEST_CASES = [
    (
        "pure_greeting",
        "Joanna",
        "2022-01-21",
        "Hey Nate! Long time no see! What about you - any fun projects or hobbies?",
        [("Nate", "Hi Joanna!")],
    ),
    (
        "pure_congrats",
        "Joanna",
        "2022-01-23",
        "Congrats, Nate!",
        [("Nate", "I won my first video game tournament last week.")],
    ),
    (
        "pure_reaction",
        "Joanna",
        "2022-01-23",
        "Wow, that's amazing!",
        [("Nate", "I attached a photo of my pet turtles!")],
    ),
    (
        "single_fact",
        "Nate",
        "2022-01-23",
        "I've had them for 3 years now and they bring me tons of joy!",
        [
            ("Nate", "I like having some pet turtles around."),
            ("Joanna", "Aww, how long have you had them?"),
        ],
    ),
    (
        "mixed_greeting_fact",
        "Nate",
        "2022-01-21",
        "Hey Joanna! That's cool! I won my first video game tournament last week - so exciting!",
        [("Joanna", "I've been working on a project lately - it's been pretty cool.")],
    ),
    (
        "multi_fact",
        "Joanna",
        "2022-01-23",
        "I'm allergic to most reptiles and animals with fur. I'm also allergic to cockroaches.",
        [("Nate", "What specifically are you allergic to?")],
    ),
    (
        "question_with_content",
        "Nate",
        "2022-04-17",
        "Have you been to the cocktail bar on 5th Street since you moved to Boston?",
        [("Joanna", "I moved to Boston a couple months ago.")],
    ),
    (
        "question_no_content",
        "Nate",
        "2022-01-23",
        "How are you?",
        [("Joanna", "Hey Nate, long time!")],
    ),
    (
        "anaphora_resolved",
        "Joanna",
        "2022-05-20",
        "Yeah, it really lifted my spirits. I cried when I saw it.",
        [
            ("Joanna", "I got a really kind letter from my dad yesterday."),
            ("Nate", "That's really sweet of him."),
        ],
    ),
    (
        "filler_then_fact",
        "Nate",
        "2022-08-22",
        "Yeah, for sure! Last Friday I went on a hike to Half Dome with my brother Sam.",
        [("Joanna", "What have you been up to?")],
    ),
]


def _format_neighbors(prior: list[tuple[str, str]]) -> str:
    if not prior:
        return ""
    lines = ["PRIOR TURNS (context only, do not emit):"]
    for who, what in prior:
        lines.append(f"- {who}: {what}")
    lines.append("")
    return "\n".join(lines) + "\n"


async def run_one(lm, label, speaker, date, message, prior):
    neighbors = _format_neighbors(prior)
    prompt = PROMPT_SENTENCE.format(
        speaker=speaker,
        date=date,
        passage=message,
        neighbors_block=neighbors,
    )
    response = await lm.generate_parsed_response(
        output_format=_RewriteResponse,
        user_prompt=prompt,
        max_attempts=3,
    )
    items = response.items if response else []
    print(f"\n{'=' * 78}")
    print(f"[{label}]")
    if prior:
        print("  PRIOR:")
        for who, what in prior:
            print(f"    {who}: {what}")
    print(f"  MESSAGE ({speaker} on {date}):")
    print(f"    {message}")
    print(f"  N_ITEMS: {len(items)}")
    for i, item in enumerate(items):
        print(f"  ---")
        print(f"  item {i}:")
        print(f"    sentence: {item.sentence!r}")
        print(f"    rewrite : {item.rewrite!r}")
        print(f"    queries : {item.queries}")


async def main():
    load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    lm = OpenAIChatCompletionsLanguageModel(
        OpenAIChatCompletionsLanguageModelParams(
            client=client,
            model="gpt-5.4-nano",
            reasoning_effort="low",
        )
    )
    for case in TEST_CASES:
        await run_one(lm, *case)


if __name__ == "__main__":
    asyncio.run(main())
