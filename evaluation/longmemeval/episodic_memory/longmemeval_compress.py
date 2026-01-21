import argparse
import asyncio
import json
import os

import openai
from dotenv import load_dotenv

from memmachine.common.utils import async_with

from memmachine.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)

compression_system_prompt_template = "Compress the text within the <text></text> tags, preserving meaning exactly. Preserve causal relationships and antecedents for references from the original text. If it's unclear due to missing context, ensure that the sentence structure is flexible enough to be used in the same context as the original text. Do not add fluff. Respond under 200 words."

compression_user_prompt_template = "<text>\n{text}\n</text>"


async def process_conversation(language_model, question_item):
    for session in question_item["haystack_sessions"]:
        for turn in session:
            if len(turn["content"]) > 800:
                compressed_content, _ = await language_model.generate_response(
                    system_prompt=compression_system_prompt_template,
                    user_prompt=compression_user_prompt_template.format(
                        text=turn["content"],
                    ),
                )
                turn["content"] = compressed_content.strip()
                print(compressed_content.strip())

    return question_item


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )
    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path

    with open(data_path, "r") as f:
        lme_json = json.load(f)

    openai_client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    language_model = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=openai_client,
            model="gpt-4.1-nano",
        )
    )

    semaphore = asyncio.Semaphore(500)
    tasks = [
        async_with(semaphore, process_conversation(language_model, question_item))
        for question_item in lme_json
    ]
    result = await asyncio.gather(*tasks)

    with open(target_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
