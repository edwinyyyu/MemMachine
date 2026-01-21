import argparse
import asyncio
import json
import os

import boto3
import neo4j
import openai
from dotenv import load_dotenv


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    data_path = args.data_path

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    lengths = []
    for idx, item in enumerate(locomo_data):
        if "conversation" not in item:
            return

        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        print(
            f"Processing conversation for group {idx} with speakers {speaker_a} and {speaker_b}..."
        )

        session_idx = 0

        while True:
            session_idx += 1
            session_id = f"session_{session_idx}"

            if session_id not in conversation:
                break

            session = conversation[session_id]
            for message in session:
                text = message["text"]
                lengths.append(len(text))

    average_length = sum(lengths) / len(lengths)
    median_length = sorted(lengths)[len(lengths) // 2]
    print(average_length)
    print(median_length)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
