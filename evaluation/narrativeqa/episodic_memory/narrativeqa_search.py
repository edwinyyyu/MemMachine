import argparse
import asyncio
import json
import os
import time

import neo4j
import openai
import pandas as pd
from dotenv import load_dotenv

from memmachine.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine.common.reranker.identity_reranker import IdentityReranker
from memmachine.common.utils import async_with
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine.episodic_memory.declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryParams,
)

ANSWER_PROMPT = """
You are asked to answer a question based on your memories.

<memories>
{memories}
</memories>

Question: {question}
Your short response to the question without fluff (no more than a couple of sentences):

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

    neo4j_driver = neo4j.AsyncGraphDatabase.driver(
        uri=os.getenv("NEO4J_URI"),
        auth=(
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
        ),
    )

    vector_graph_store = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            vector_index_creation_threshold=1,
        )
    )

    openai_client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
        )
    )

    language_model = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=openai_client,
            model="gpt-4.1-nano",
        )
    )

    reranker = IdentityReranker()

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

        memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=document_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
                language_model=language_model,
            )
        )

        question_results = []

        question_rows = narrativeqa_qa[
            narrativeqa_qa["document_id"] == document_id
        ]
        for _, question_row in question_rows.iterrows():
            question = question_row["question"]

            chunks = await memory.search(
                query=question,
                max_num_episodes=100,
            )

            start_time = time.monotonic()
            formatted_memories = memory.string_from_episode_context(
                episode_context=chunks,
            )
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
                f"MEMORIES_START\n{formatted_memories}MEMORIES_END\n"
            )

            question_result = {
                "question": question,
                "answer1": question_row["answer1"],
                "answer2": question_row["answer2"],
                "response": response_text,
                "memory_latency": memory_latency,
                "llm_latency": llm_latency,
                "memories": formatted_memories,
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
