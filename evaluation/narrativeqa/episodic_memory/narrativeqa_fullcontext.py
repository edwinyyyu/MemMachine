import argparse
import asyncio
import json
import os
import time

import openai
import pandas as pd
from dotenv import load_dotenv

from memmachine.common.utils import async_with

ANSWER_PROMPT = """
You are asked to answer a question based on your memories.

<instrcutions>
1. Use only the information in your memories to answer the question.
2. Be specific when referring to people, places, objects, events, dates, and concepts.
3. Think carefully about what your memories imply beyond their explicit content.
</instrcutions>

<memories>
{memories}
</memories>

Question: {question}
Your short response to the question without fluff (no more than a sentence):

"""

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--meta-path", required=True, help="Path to the meta file")
    parser.add_argument("--data-path", required=True, help="Path to the data file")
    parser.add_argument("--qa-path", required=True, help="Path to the qa file")
    parser.add_argument("--target-path", required=True, help="Path to the target file")

    args = parser.parse_args()

    meta_path = args.meta_path
    data_path = args.data_path
    qa_path = args.qa_path
    target_path = args.target_path

    narrativeqa_meta = pd.read_csv(meta_path)
    narrativeqa_qa = pd.read_csv(qa_path)

    openai_client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    async def process_document(row):
        document_id = row["document_id"]

        try:
            with open(f"{data_path}/{document_id}.content", "r") as f:
                document_content = f.read()
        except:
            print(f"Failed to read document {document_id}")
            return None, None

        if not document_content.strip():
            print(f"Empty content for document {document_id}")
            return None, None

        print(
            f"Processing document {document_id}"
        )

        question_results = []

        question_rows = narrativeqa_qa[
            narrativeqa_qa["document_id"] == document_id
        ]
        for _, question_row in question_rows.iterrows():
            question = question_row["question"]

            start_time = time.monotonic()
            formatted_memories = document_content
            end_time = time.monotonic()
            memory_latency = end_time - start_time

            start_time = time.monotonic()

            response = None
            wait_time_seconds = 1
            while response is None:
                try:
                    response = await openai_client.responses.create(
                        model="gpt-4.1-mini",
                        input=ANSWER_PROMPT.format(
                            memories=formatted_memories,
                            question=question,
                        )
                    )
                except Exception as e:
                    print(f"Error ({document_id}:{question}): {e}", flush=True)
                    await asyncio.sleep(wait_time_seconds)
                    wait_time_seconds = min(wait_time_seconds * 2, 60)

            end_time = time.monotonic()
            llm_latency = end_time - start_time

            response_text = response.output_text

            print(
                f"NQA-Document ID: {document_id}\n"
                f"NQA-Question: {question}\n"
                f"NQA-Answer1: {question_row["answer1"]}\n"
                f"NQA-Answer2: {question_row["answer2"]}\n"
                f"NQA-Response: {response_text}\n"
                f"NQA-Memory latency: {memory_latency:.2f} seconds\n"
                f"NQA-LLM latency: {llm_latency:.2f} seconds\n"
            )

            question_result = {
                "question": question,
                "answer1": question_row["answer1"],
                "answer2": question_row["answer2"],
                "response": response_text,
                "memory_latency": memory_latency,
                "llm_latency": llm_latency,
            }

            question_results.append(question_result)

        return document_id, question_results

    semaphore = asyncio.Semaphore(20)
    tasks = [
        async_with(semaphore, process_document(row))
        for _, row in narrativeqa_meta[:50].iterrows()]

    results = {
        document_id: question_results
        for document_id, question_results in await asyncio.gather(*tasks) if document_id
    }
    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
