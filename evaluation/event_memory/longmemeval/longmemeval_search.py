import argparse
import asyncio
import json
import logging
import os
import time

import boto3
from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    load_longmemeval_dataset,
)
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )
    parser.add_argument(
        "--vector-search-limit",
        type=int,
        default=100,
        help="Number of vectors to retrieve (default: 100)",
    )
    parser.add_argument(
        "--expand-context",
        type=int,
        default=0,
        help="Number of context segments to expand (default: 0)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent questions (default: 4)",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable the reranker and rank by embedding similarity only",
    )
    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path

    all_questions = load_longmemeval_dataset(data_path)

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    engine = create_async_engine(os.getenv("SQL_URL"))
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
        )
    )

    if args.no_reranker:
        reranker = None
    else:
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

    async def process_question(question: LongMemEvalItem):
        partition_key = question.question_id

        collection = await vector_store.open_collection(
            namespace="longmemeval", name=question.question_id
        )

        segment_store_partition = await segment_store.open_or_create_partition(
            partition_key
        )

        memory = EventMemory(
            EventMemoryParams(
                vector_store_collection=collection,
                segment_store_partition=segment_store_partition,
                embedder=embedder,
                reranker=reranker,
            )
        )

        search_query = f"User: {question.question}"

        memory_start = time.monotonic()
        query_result = await memory.query(
            query=search_query,
            vector_search_limit=args.vector_search_limit,
            expand_context=args.expand_context,
        )
        memory_latency = time.monotonic() - memory_start

        print(
            f"Question ID: {question.question_id}\n"
            f"Question: {question.question}\n"
            f"Question Type: {question.question_type}\n"
            f"Memory retrieval time: {memory_latency:.2f} seconds\n"
        )

        result = {
            "question_id": question.question_id,
            "question_date": question.question_date,
            "question": question.question,
            "answer": question.answer,
            "answer_turn_indices": question.answer_turn_indices,
            "question_type": question.question_type.value,
            "abstention": question.abstention_question,
            "memory_latency": memory_latency,
            "query_result": query_result.model_dump(mode="json"),
        }

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)

        return result

    semaphore = asyncio.Semaphore(args.concurrency)
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

    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()
    await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig()
    logging.getLogger(
        "memmachine_server.episodic_memory.event_memory.event_memory"
    ).setLevel(logging.DEBUG)
    asyncio.run(main())
