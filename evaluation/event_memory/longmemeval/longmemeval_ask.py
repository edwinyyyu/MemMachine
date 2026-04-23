"""Search and answer a single longmemeval question, printing everything to stdout.

Runs the full pipeline (search + QA) for one question and prints:
- question metadata (id, type, date, question, gold answer)
- raw QueryResult as JSON
- formatted context string
- LLM response and latencies
"""

import argparse
import asyncio
import json
import logging
import os
import time

import boto3
from dotenv import load_dotenv
from longmemeval_models import (
    get_datetime_from_timestamp,
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

# Parts of prompt borrowed from Mastra's OM.
# https://github.com/mastra-ai/mastra/blob/977b49e23d8b050a2c6a6a91c0aa38b28d6388ee/packages/memory/src/processors/observational-memory/observational-memory.ts#L312-L318
ANSWER_PROMPT = """
You are a helpful assistant with access to extensive conversation history.
When answering questions, carefully review the conversation history to identify and use any relevant user preferences, interests, or specific details they have mentioned.

<history>
{memories}
</history>

IMPORTANT: When responding, reference specific details from these observations. Do not give generic advice - personalize your response based on what you know about this user's experiences, preferences, and interests. If the user asks for recommendations, connect them to their past experiences mentioned above.

KNOWLEDGE UPDATES: When asked about current state (e.g., "where do I currently...", "what is my current..."), always prefer the MOST RECENT information. Observations include dates - if you see conflicting information, the newer observation supersedes the older one. Look for phrases like "will start", "is switching", "changed to", "moved to" as indicators that previous information has been updated.

PLANNED ACTIONS: If the user stated they planned to do something (e.g., "I'm going to...", "I'm looking forward to...", "I will...") and the date they planned to do it is now in the past (check the relative time like "3 weeks ago"), assume they completed the action unless there's evidence they didn't. For example, if someone said "I'll start my new diet on Monday" and that was 2 weeks ago, assume they started the diet.

Current date: {question_timestamp}
Question: {question}
"""


async def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument("--question-id", required=True, help="Question ID to ask")
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
        "--max-num-segments",
        type=int,
        default=20,
        help="Max segments after unification (default: 150)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="LLM model for answering (default: gpt-5-mini)",
    )
    args = parser.parse_args()

    all_questions = load_longmemeval_dataset(args.data_path)
    question = next(
        (q for q in all_questions if q.question_id == args.question_id), None
    )
    if question is None:
        print(f"No question found with ID: {args.question_id}")
        return

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

    collection = await vector_store.open_collection(
        namespace="longmemeval", name=question.question_id
    )
    segment_store_partition = await segment_store.open_or_create_partition(
        question.question_id
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

    unified = EventMemory.build_query_result_context(
        query_result, max_num_segments=args.max_num_segments
    )
    formatted_context = EventMemory.string_from_segment_context(unified)

    llm_start = time.monotonic()
    response = await openai_client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    memories=formatted_context,
                    question_timestamp=get_datetime_from_timestamp(
                        question.question_date
                    ).strftime("%A, %B %d, %Y at %I:%M %p"),
                    question=question.question,
                ),
            },
        ],
    )
    llm_latency = time.monotonic() - llm_start

    print(f"Question ID: {question.question_id}")
    print(f"Question Type: {question.question_type.value}")
    print(f"Question Date: {question.question_date}")
    print(f"Question: {question.question}")
    print(f"Answer: {question.answer}")
    print(f"Abstention: {question.abstention_question}")
    print(f"Answer Turn Indices: {question.answer_turn_indices}")
    print()
    print(f"Memory retrieval time: {memory_latency:.2f}s")
    print(f"LLM response time: {llm_latency:.2f}s")
    print(f"Input tokens: {response.usage.prompt_tokens}")
    print(f"Output tokens: {response.usage.completion_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print()
    print(f"Response: {response.choices[0].message.content.strip()}")
    print()
    print("QUERY_RESULT_JSON_START")
    print(json.dumps(query_result.model_dump(mode="json"), indent=2))
    print("QUERY_RESULT_JSON_END")
    print()
    print("FORMATTED_CONTEXT_START")
    print(formatted_context)
    print("FORMATTED_CONTEXT_END")

    await segment_store.close_partition(segment_store_partition)
    await vector_store.close_collection(collection=collection)
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
