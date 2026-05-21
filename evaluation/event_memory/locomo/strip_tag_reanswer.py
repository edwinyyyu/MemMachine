"""Strip [category: ...] tags from a search file's conversation_memories and
re-call the answerer with the stripped context. Disambiguates how much of
v30's gain comes from the answerer seeing the tag vs retrieval-time signal."""

import argparse
import asyncio
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

ANSWER_PROMPT = """
You are a helpful assistant with access to extensive conversation history.
When answering questions, carefully review the conversation history to identify and use any relevant user preferences, interests, or specific details they have mentioned.

<history>
{memories}
</history>

Question: {question}
"""

TAG_RE = re.compile(r" ?\[category:[^\]]*\]")


async def reanswer(
    client: AsyncOpenAI, model: str, sem: asyncio.Semaphore, item: dict
) -> dict:
    stripped_memories = TAG_RE.sub("", item.get("conversation_memories", ""))
    async with sem:
        start = time.monotonic()
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(
                        memories=stripped_memories, question=item["question"]
                    ),
                },
            ],
        )
        latency = time.monotonic() - start
    new_item = dict(item)
    new_item["conversation_memories"] = stripped_memories
    new_item["model_answer"] = (resp.choices[0].message.content or "").strip()
    new_item["llm_latency"] = latency
    return new_item


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--concurrency", type=int, default=30)
    args = parser.parse_args()

    load_dotenv()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    with open(args.data_path) as f:
        d = json.load(f)

    tasks = []
    new_d: dict[str, list[dict]] = {}
    for cat, items in d.items():
        new_d[cat] = [None] * len(items)
        for i, item in enumerate(items):
            tasks.append(_wrap(client, args.model, sem, item, cat, i, new_d))

    await asyncio.gather(*tasks)

    with open(args.target_path, "w") as f:
        json.dump(new_d, f, indent=4)
    print(f"Wrote {args.target_path}")


async def _wrap(client, model, sem, item, cat, i, new_d):
    new_d[cat][i] = await reanswer(client, model, sem, item)


if __name__ == "__main__":
    asyncio.run(main())
