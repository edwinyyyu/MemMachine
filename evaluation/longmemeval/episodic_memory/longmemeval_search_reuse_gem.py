import argparse
import asyncio
import json
import os
import time

from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    get_datetime_from_timestamp,
    load_longmemeval_dataset,
)
from google import genai

from memmachine.common.utils import async_with


ANSWER_PROMPT = """
You are asked to answer a question based on your memories of a conversation.

<instructions>
1. Prioritize memories that answer the question directly. Be meticulous about recalling details.
2. When there may be multiple answers to the question, think hard to remember and list all possible answers. Do not become satisfied with just the first few answers you remember.
3. When asked about time intervals or to count items, do not rush to answer immediately. Instead, carefully enumerate the items or subtract the times using numbers.
4. Your memories are episodic, meaning that they consist of only your raw observations of what was said. You may need to reason about or guess what the memories imply in order to answer the question.
5. The question may contain typos or be based on the asker's own unreliable memories. Do your best to answer the question using the most relevant information in your memories.
6. Your memories may include small or large jumps in time or context. You are not confused by this. You just did not bother to remember everything in between.
7. Your memories are ordered from earliest to latest.
</instructions>

<memories>
{memories}
</memories>

Question timestamp: {question_timestamp}
Question: {question}
Your short response to the question without fluff (no more than a couple of sentences):
"""

ANSWER_PROMPT = """
You are asked to answer a question from a user based on your memories of a conversation between the user and an assistant.

<instructions>
1. Prioritize memories that answer the question directly. Be meticulous about recalling details.
2. When there may be multiple answers to the question, think hard to remember and list all possible answers. Do not become satisfied with just the first few answers you remember.
3. When asked to count items, carefully enumerate the items using numbers.
4. When asked about time intervals, the duration between events is computed by subtracting the start date from the end date in the chosen unit.
5. When asked for advice or suggestions, synthesize your memories of the user's interests, preferences, possessions, and problems to provide tailored recommendations.
6. Your memories are episodic, meaning that they consist of only your raw observations of what was said. You may need to reason about or guess what the memories imply in order to answer the question.
7. Your memories may include small or large jumps in time or context. You are not confused by this. You just did not bother to remember everything in between.
8. Your memories are ordered from earliest to latest. Prioritize the latest memories if anything has changed over time. Consider the question datetime when determining whether an event has actually occurred.
</instructions>

<memories>
{memories}
</memories>

Question timestamp: {question_timestamp}
Question: {question}
Your short response to the question without fluff (no more than a couple of sentences):
"""


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--search-path", required=True, help="Path to the search data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )

    args = parser.parse_args()

    data_path = args.data_path
    search_path = args.search_path
    target_path = args.target_path

    all_questions = load_longmemeval_dataset(data_path)

    with open(search_path, "r") as f:
        search_data = json.load(f)

    question_items = {item["question_id"]: item for item in search_data}

    gemini_client = genai.Client()

    async def qa_eval(
        memories,
        question_timestamp,
        question: str,
        model: str = "gemini-3-pro-preview",
    ):
        start_time = time.monotonic()
        response = await gemini_client.aio.models.generate_content(
            model=model,
            contents=ANSWER_PROMPT.format(
                memories=memories,
                question_timestamp=question_timestamp,
                question=question,
            ),
        )
        end_time = time.monotonic()

        latency = end_time - start_time

        return {
            "response": response.text,
            "latency": latency,
        }

    async def process_question(
        question: LongMemEvalItem,
    ):
        total_start = time.monotonic()
        memory_start = time.monotonic()
        memory_end = time.monotonic()
        memory_latency = memory_end - memory_start

        formatted_context = question_items[question.question_id]["episodes_text"]
        # aformatted_context = memory.string_from_episode_context_additional(chunks)

        # await asyncio.sleep(0.5)
        response = await qa_eval(
            formatted_context,
            get_datetime_from_timestamp(question.question_date).strftime(
                "%A, %B %d, %Y at %I:%M %p"
            ),
            question.question,
        )
        total_end = time.monotonic()
        total_latency = total_end - total_start

        print(
            f"Question ID: {question.question_id}\n"
            f"Question: {question.question}\n"
            f"Question Date: {question.question_date}\n"
            f"Question Type: {question.question_type}\n"
            f"Answer: {question.answer}\n"
            f"Response: {response['response']}\n"
            f"Memory retrieval time: {memory_latency:.2f} seconds\n"
            f"LLM response time: {response['latency']:.2f} seconds\n"
            f"Total processing time: {total_latency:.2f} seconds\n"
            f"MEMORIES_START\n{formatted_context}MEMORIES_END\n"
        )

        return {
            "question_id": question.question_id,
            "question_date": question.question_date,
            "question": question.question,
            "answer": question.answer,
            "response": response["response"],
            "question_type": question.question_type.value,
            "abstention": question.abstention_question,
            "total_latency": total_latency,
            "memory_latency": memory_latency,
            "llm_latency": response["latency"],
            "episodes_text": formatted_context,
        }

    semaphore = asyncio.Semaphore(10)
    tasks = [
        async_with(
            semaphore,
            process_question(question),
        )
        for question in all_questions
    ]
    results = await asyncio.gather(*tasks)

    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
