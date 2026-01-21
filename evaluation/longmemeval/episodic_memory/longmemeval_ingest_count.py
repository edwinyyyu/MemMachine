import argparse
import asyncio

from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    load_longmemeval_dataset,
)

from memmachine.common.utils import async_with

longest_episode_text = ""


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    data_path = args.data_path

    all_questions = load_longmemeval_dataset(data_path)

    async def process_conversation(question: LongMemEvalItem):
        global longest_episode_text
        session_ids = list(question.session_id_map.keys())
        for session_id in session_ids:
            session = question.get_session(session_id)
            for turn in session:
                if len(turn.content.strip()) > len(longest_episode_text):
                    longest_episode_text = turn.content.strip()
                print(turn.timestamp)

    semaphore = asyncio.Semaphore(1)
    tasks = [
        async_with(semaphore, process_conversation(question))
        for question in all_questions
    ]
    await asyncio.gather(*tasks)
    # print(longest_episode_text)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
