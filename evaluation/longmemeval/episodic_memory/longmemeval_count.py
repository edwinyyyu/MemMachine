import argparse
import asyncio
import json

from dotenv import load_dotenv

from memmachine.common.utils import async_with


def process_conversation(question_item):
    print(question_item["question"])


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    data_path = args.data_path

    with open(data_path, "r") as f:
        lme_json = json.load(f)

    result = [process_conversation(question_item) for question_item in lme_json]


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
