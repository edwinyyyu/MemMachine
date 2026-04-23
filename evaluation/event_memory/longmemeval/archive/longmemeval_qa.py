"""Run LLM QA evaluation using pre-computed search_raw results.

Takes the output of longmemeval_search_raw.py, reconstructs segments,
applies unification, formats the context string, and runs LLM QA.
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from uuid import UUID

from dotenv import load_dotenv
from longmemeval_models import get_datetime_from_timestamp
from memmachine_server.common.utils import async_with
from memmachine_server.episodic_memory.event_memory.data_types import (
    Context,
    QueryResult,
    ScoredSegmentContext,
    Segment,
    Text,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from openai import AsyncOpenAI
from pydantic import TypeAdapter

_ContextAdapter = TypeAdapter(Context | None)

# Parts of prompt borrowed from Mastra's OM.
# https://github.com/mastra-ai/mastra/blob/977b49e23d8b050a2c6a6a91c0aa38b28d6388ee/packages/memory/src/processors/observational-memory/observational-memory.ts#L312-L318
ANSWER_PROMPT = """
You are a helpful assistant with access to extensive conversation history.
When answering questions, carefully review the conversation history to identify and use any relevant user preferences, interests, or specific details they have mentioned.

<history>
{memories}
</history>

IMPORTANT: When responding, reference specific details from these observations. Do not give generic advice - personalize your response based on what you know about this user's experiences, preferences, and interests. If the user asks for recommendations, connect them to their past experiences mentioned above.

KNOWLEDGE UPDATES: When asked about current state (e.g., "where do I currently...", "what is my current..."), always prefer the MOST RECENT information. Observations include dates - if you see conflicting information, the newer observation supersedes the older one. Look for phrases like "will start", "is switching", "changed to", "moved to" as indicators that previous information has been updated.

PLANNED ACTIONS: If the user stated they planned to do something (e.g., "I'm going to...", "I'm looking forward to...", "I will...") and the date they planned to do it is now in the past (check the relative time like "3 weeks ago"), assume they completed the action unless there's evidence they didn't. For example, if someone said "I'll start my new diet on Monday" and that was 2 weeks ago, assume they started the diet.

Current date: {question_timestamp}
Question: {question}
"""


def _reconstruct_segment(seg_data: dict) -> Segment:
    return Segment(
        uuid=UUID(seg_data["uuid"]),
        event_uuid=UUID(seg_data["event_uuid"]),
        index=seg_data["index"],
        offset=seg_data["offset"],
        timestamp=datetime.fromisoformat(seg_data["timestamp"]),
        context=_ContextAdapter.validate_python(seg_data.get("context")),
        block=Text(text=seg_data.get("text") or ""),
        properties=seg_data.get("properties", {}),
    )


def build_context_string(item: dict, max_num_segments: int) -> str:
    """Reconstruct segments, unify, and format as context string."""
    scored_segment_contexts: list[ScoredSegmentContext] = []
    for rank, sc in enumerate(item["segment_contexts"]):
        segments = [_reconstruct_segment(s) for s in sc["segments"]]
        seed_uuid = UUID(sc["seed_segment_uuid"])
        scored_segment_contexts.append(
            ScoredSegmentContext(
                score=float(len(item["segment_contexts"]) - rank),
                seed_segment_uuid=seed_uuid,
                segments=segments,
            )
        )

    query_result = QueryResult(scored_segment_contexts=scored_segment_contexts)
    unified = EventMemory.build_query_result_context(
        query_result, max_num_segments=max_num_segments
    )
    return EventMemory.string_from_segment_context(unified)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-path", required=True, help="Path to longmemeval_search_raw output"
    )
    parser.add_argument("--target-path", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--max-num-segments",
        type=int,
        default=150,
        help="Max segments after unification (default: 150)",
    )
    args = parser.parse_args()

    with open(args.search_path) as f:
        search_results = json.load(f)

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def qa_eval(
        memories: str,
        question_timestamp: str,
        question: str,
        model: str = "gpt-5-mini",
    ):
        start_time = time.monotonic()
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(
                        memories=memories,
                        question_timestamp=question_timestamp,
                        question=question,
                    ),
                },
            ],
        )
        latency = time.monotonic() - start_time
        return {
            "response": response.choices[0].message.content.strip(),
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency": latency,
        }

    async def process_item(item: dict):
        formatted_context = build_context_string(item, args.max_num_segments)

        total_start = time.monotonic()
        response = await qa_eval(
            formatted_context,
            get_datetime_from_timestamp(item["question_date"]).strftime(
                "%A, %B %d, %Y at %I:%M %p"
            ),
            item["question"],
        )
        total_latency = time.monotonic() - total_start

        print(
            f"Question ID: {item['question_id']}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}\n"
            f"Response: {response['response']}\n"
            f"LLM response time: {response['latency']:.2f} seconds\n"
        )

        return {
            "question_id": item["question_id"],
            "question_date": item["question_date"],
            "question": item["question"],
            "answer": item["answer"],
            "response": response["response"],
            "question_type": item["question_type"],
            "abstention": item["abstention"],
            "total_latency": total_latency,
            "memory_latency": item.get("memory_latency"),
            "llm_latency": response["latency"],
            "episodes_text": formatted_context,
        }

    semaphore = asyncio.Semaphore(50)
    tasks = [async_with(semaphore, process_item(item)) for item in search_results]
    results = await asyncio.gather(*tasks)

    with open(args.target_path, "w") as f:
        json.dump(results, f, indent=4)

    await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
