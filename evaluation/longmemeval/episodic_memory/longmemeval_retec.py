import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timedelta

import boto3
import neo4j
from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    get_datetime_from_timestamp,
    load_longmemeval_dataset,
)
from openai import AsyncOpenAI

from memmachine.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine.common.reranker.identity_reranker import (
    IdentityReranker,
)
from memmachine.common.utils import async_with
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine.episodic_memory.declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryParams,
)


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

    all_questions = load_longmemeval_dataset(data_path)

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
            max_concurrent_transactions=1000,
        )
    )

    openai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=2048,
        )
    )

    region = "us-west-2"
    aws_client = boto3.client(
        "bedrock-agent-runtime",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    reranker = AmazonBedrockReranker(
        AmazonBedrockRerankerParams(
            client=aws_client,
            region=region,
            model_id="cohere.rerank-v3-5:0",
        )
    )

    # reranker = IdentityReranker()

    async def process_question(
        question: LongMemEvalItem,
    ):
        group_id = question.question_id

        memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=group_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
            )
        )

        search_query = question.question

        memory_start = time.monotonic()
        results = await memory.search(query=search_query, max_num_episodes=200)
        memory_end = time.monotonic()
        memory_latency = memory_end - memory_start

        print(
            f"Question ID: {question.question_id}\n"
            f"Question: {question.question}\n"
            f"Question Date: {question.question_date}\n"
            f"Question Type: {question.question_type}\n"
            f"Answer: {question.answer}\n"
            f"Memory retrieval time: {memory_latency:.2f} seconds\n"
        )

        return {
            "question_id": question.question_id,
            "question_date": question.question_date,
            "question": question.question,
            "answer": question.answer,
            "answer_turn_indices": question.answer_turn_indices,
            "question_type": question.question_type.value,
            "abstention": question.abstention_question,
            "memory_latency": memory_latency,
            "episode_contexts": [
                {
                    "score": score,
                    "nuclear_episode": nuclear_episode.uid,
                    "episodes": [
                        {
                            "uid": episode.uid,
                            "timestamp": episode.timestamp.isoformat(),
                            "source": episode.source,
                            "content_type": episode.content_type.value,
                            "content": episode.content,
                            # "additional": episode.additional,
                            "filterable_properties": episode.filterable_properties,
                            "user_metadata": episode.user_metadata,
                        }
                        for episode in episode_context
                    ],
                }
                for score, nuclear_episode, episode_context in results
            ],
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

    dump_data = {}
    for result in results:
        category_result = dump_data.get(result["question_type"], [])
        category_result.append(result)
        dump_data[result["question_type"]] = category_result

    with open(target_path, "w") as f:
        json.dump(dump_data, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
