import argparse
import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from memmachine.common.utils import async_with
from memmachine.episodic_memory.declarative_memory import (
    ContentType,
    DeclarativeMemory,
    Episode,
)


def datetime_from_locomo_time(locomo_time_str: str) -> datetime:
    return datetime.strptime(locomo_time_str, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=UTC
    )


async def process_question(
    locomo_item,
    question,
    answer,
    category,
    evidence,
    adversarial_answer,
):
    memory_start = time.time()
    conversation = locomo_item["conversation"]

    episodes = []
    session_idx = 0
    while True:
        session_idx += 1
        session_id = f"session_{session_idx}"

        if session_id not in conversation:
            break

        session = conversation[session_id]
        session_datetime = datetime_from_locomo_time(
            conversation[f"{session_id}_date_time"]
        )

        episodes += [
            Episode(
                uid=str(uuid4()),
                timestamp=session_datetime + message_index * timedelta(seconds=1),
                source=message["speaker"],
                content_type=ContentType.MESSAGE,
                content=message["text"]
                + (
                    f" [Attached {blip_caption}: {image_query}]"
                    if (
                        (
                            ((blip_caption := message.get("blip_caption")) or True)
                            and ((image_query := message.get("query")) or True)
                        )
                        and blip_caption
                        and image_query
                    )
                    else (
                        f" [Attached {blip_caption}]"
                        if blip_caption
                        else (
                            f" [Attached a photo: {image_query}]" if image_query else ""
                        )
                    )
                ),
                user_metadata={
                    "locomo_session_id": session_id,
                },
            )
            for message_index, message in enumerate(session)
            if message.get("dia_id") in evidence
        ]

    memory_end = time.time()

    formatted_context = DeclarativeMemory.string_from_episode_context(episodes)

    print(
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Memory retrieval time: {memory_end - memory_start:.2f} seconds\n"
        f"MEMORIES START\n{formatted_context}MEMORIES END\n"
    )
    return {
        "question": question,
        "locomo_answer": answer,
        "category": category,
        "evidence": evidence,
        "adversarial_answer": adversarial_answer,
        "conversation_memories": formatted_context,
    }


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )

    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    results: dict[str, Any] = {}
    for idx, item in enumerate(locomo_data):
        if "conversation" not in item:
            continue

        qa_list = item["qa"]

        print(f"Processing questions for group {idx}...")

        async def respond_question(qa):
            question = qa["question"]
            answer = qa.get("answer", "")
            category = qa["category"]
            evidence = qa["evidence"]

            adversarial_answer = qa.get("adversarial_answer", "")

            question_response = await process_question(
                item,
                question,
                answer,
                category,
                evidence,
                adversarial_answer,
            )
            return (
                category,
                question_response,
            )

        semaphore = asyncio.Semaphore(30)
        response_tasks = [
            async_with(
                semaphore,
                respond_question(qa),
            )
            for qa in qa_list
        ]

        responses = await asyncio.gather(*response_tasks)

        for category, response in responses:
            category_result = results.get(category, [])
            category_result.append(response)
            results[category] = category_result

    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
