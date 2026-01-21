import argparse
import asyncio
import json

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

questions = {}

def process_conversation(question_item):
    question = question_item["question"]
    question_type = question_item["question_type"]

    questions.setdefault(question_type, [])
    questions[question_type].append(question)


async def main():
    client = AsyncOpenAI()

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    data_path = args.data_path

    with open(data_path, "r") as f:
        lme_json = json.load(f)

    result = [process_conversation(question_item) for question_item in lme_json]
    for question_type, typed_questions in questions.items():
        response = await client.embeddings.create(
            input=typed_questions,
            model="text-embedding-3-small",
        )

        embeddings = [datum.embedding for datum in response.data]

        mean_embedding = np.mean(np.array(embeddings), axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        print(f"Question type: {question_type}")
        print(repr(mean_embedding.astype(float).tolist()))

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
