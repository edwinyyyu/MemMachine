"""Official BEAM search + answer generation against EventMemory.

Based on the official BEAM generation prompt (`answer_generation_for_rag` from
https://github.com/mohammadtavakoli78/BEAM/blob/main/src/prompts.py). We add
one note that retrieved segments are ordered chronologically — this is a true
property of our `build_query_result_context` output and helps the answerer on
event_ordering / temporal_reasoning questions without leaking any category-
specific metadata.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import FormatOptions
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent))
from beam_models import BEAMConversation, BEAMQuestion, load_beam_dataset
from llm_provider import (
    DEFAULT_ANSWER_MODEL,
    PROVIDER_OPENAI,
    PROVIDERS,
    make_chat_client,
)

# BEAM's timeline is content-embedded, not wall-clock. Suppress timestamp
# prefixes on retrieved context (both for reranker scoring and for what the
# generator LLM sees).
_NO_TIMESTAMPS = FormatOptions(date_style=None, time_style=None)

OUR_ANSWER_PROMPT = """
You are a helpful assistant with access to extensive conversation history.
When answering questions, carefully review the conversation history to identify and use any relevant user preferences, interests, or specific details they have mentioned.

<history>
{context}
</history>

IMPORTANT RULES:
When responding, reference specific details from these observations. Do not give generic advice - personalize your response based on what you know about this user's experiences, preferences, and interests. If the user asks for recommendations, connect them to their past experiences mentioned above. Use the most recently stated preferences.

If the memories don't contain enough information to answer, say exactly: "I don't have enough information to answer this question."

For ordering questions: present events in chronological order.

Be specific and direct — include exact names, dates, numbers, and details from the memories. Do NOT invent or assume information that isn't in the memories.

Question: {question}
"""


def _partition_key(conversation_id: str) -> str:
    return conversation_id.lower().replace("-", "_")


async def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", required=True, help="Path to BEAM JSON file")
    parser.add_argument("--target-path", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--namespace",
        default="beam",
        help="Qdrant collection namespace",
    )
    parser.add_argument(
        "--vector-search-limit",
        type=int,
        default=100,
        help="Number of derivative vectors to retrieve pre-rerank",
    )
    parser.add_argument(
        "--expand-context",
        type=int,
        default=0,
        help="Number of context segments to expand",
    )
    parser.add_argument(
        "--max-num-segments",
        type=int,
        default=20,
        help="Max segments returned to the LLM post-unification",
    )
    parser.add_argument(
        "--provider",
        default=PROVIDER_OPENAI,
        choices=list(PROVIDERS),
        help="LLM provider for answer generation",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_ANSWER_MODEL,
        help=f"Answer-generation model (default: {DEFAULT_ANSWER_MODEL}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Max concurrent questions",
    )
    parser.add_argument(
        "--show-timestamps",
        action="store_true",
        help="Surface ingest-time Event.timestamps on retrieved context",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable the Bedrock reranker entirely",
    )
    parser.add_argument(
        "--embedder-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model",
    )
    args = parser.parse_args()

    format_options = FormatOptions() if args.show_timestamps else _NO_TIMESTAMPS

    conversations = load_beam_dataset(args.data_path)

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sql_url = os.getenv("SQL_URL")
    if not sql_url:
        raise ValueError("SQL_URL must be set for the SQLAlchemy segment store")
    engine = create_async_engine(sql_url)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model=args.embedder_model,
            dimensions=1536,
        )
    )
    chat_client = make_chat_client(args.provider)

    reranker: Reranker | None
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

    segmenter = TextSegmenter()
    deriver = WholeTextDeriver()

    async def answer(context: str, question: str) -> dict:
        start = time.monotonic()
        result = await chat_client.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": OUR_ANSWER_PROMPT.format(
                        context=context, question=question
                    ),
                }
            ],
        )
        latency = time.monotonic() - start
        return {
            "response": result["content"].strip(),
            "input_tokens": result["prompt_tokens"],
            "output_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "latency": latency,
        }

    async def process_question(conversation: BEAMConversation, question: BEAMQuestion):
        partition_key = _partition_key(conversation.conversation_id)

        collection = await vector_store.open_collection(
            namespace=args.namespace, name=partition_key
        )
        if collection is None:
            print(
                f"No collection for conversation {conversation.conversation_id}; "
                f"skipping question {question.index}."
            )
            return None
        segment_store_partition = await segment_store.open_or_create_partition(
            partition_key,
            SegmentStorePartitionConfig(),
        )

        memory = EventMemory(
            EventMemoryParams(
                vector_store_collection=collection,
                segment_store_partition=segment_store_partition,
                segmenter=segmenter,
                deriver=deriver,
                embedder=embedder,
                reranker=reranker,
            )
        )

        memory_start = time.monotonic()
        query_result = await memory.query(
            query=question.question,
            vector_search_limit=args.vector_search_limit,
            expand_context=args.expand_context,
            format_options=format_options,
        )
        memory_latency = time.monotonic() - memory_start

        unified = EventMemory.build_query_result_context(
            query_result, max_num_segments=args.max_num_segments
        )
        formatted_context = EventMemory.string_from_segment_context(
            unified, format_options=format_options
        )

        llm_result = await answer(formatted_context, question.question)

        print(
            f"[{question.category}] conv={partition_key} idx={question.index} "
            f"mem={memory_latency:.2f}s llm={llm_result['latency']:.2f}s"
        )

        result = {
            "conversation_id": conversation.conversation_id,
            "category": question.category,
            "question_index": question.index,
            "question": question.question,
            "gold_answer": question.answer,
            "rubric": question.rubric,
            "ordering_tested": question.ordering_tested,
            "preference_being_tested": question.preference_being_tested,
            "instruction_being_tested": question.instruction_being_tested,
            "compliance_indicators": question.compliance_indicators,
            "time_points": question.time_points,
            "calculation_required": question.calculation_required,
            "why_unanswerable": question.why_unanswerable,
            "difficulty": question.difficulty,
            "model_answer": llm_result["response"],
            "memory_latency": memory_latency,
            "llm_latency": llm_result["latency"],
            "input_tokens": llm_result["input_tokens"],
            "output_tokens": llm_result["output_tokens"],
            "formatted_context": formatted_context,
            "query_result": query_result.model_dump(mode="json"),
        }

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)
        return result

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        async_with(semaphore, process_question(conv, q))
        for conv in conversations
        for q in conv.questions
    ]
    raw_results = await asyncio.gather(*tasks)
    results = [r for r in raw_results if r is not None]

    by_category: dict[str, list[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    with open(args.target_path, "w") as f:
        json.dump(by_category, f, indent=4)

    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()
    await openai_client.close()
    await chat_client.close()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig()
    logging.getLogger(
        "memmachine_server.episodic_memory.event_memory.event_memory"
    ).setLevel(logging.DEBUG)
    asyncio.run(main())
