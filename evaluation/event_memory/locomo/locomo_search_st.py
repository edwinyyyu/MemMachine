"""Search EventMemory for LoCoMo questions and run an LLM QA pass.

Mirrors the structure of LongMemEval's search step but for LoCoMo. Each QA
result is grouped by category in the output JSON so the existing locomo
evaluation script can consume it directly.
"""

import argparse
import asyncio
import json
import logging
import os
import time

import boto3
from dotenv import load_dotenv
from locomo_models import load_locomo_dataset
from memmachine_server.common.embedder.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    SentenceTransformerEmbedderParams,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
    SQLiteVecVectorStore,
    SQLiteVecVectorStoreParams,
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
from sentence_transformers import SentenceTransformer
from sqlalchemy import event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry

# CLDR-styled timestamps on retrieved segments. "full" date keeps the
# weekday (LoCoMo questions sometimes hinge on "last Tuesday" style refs);
# "short" time trims `1:56:01 PM UTC` -> `1:56 PM` to cut format overhead.
_FORMAT_OPTIONS = FormatOptions(date_style="full", time_style="short")


def _configure_sqlite_for_perf(engine: AsyncEngine) -> None:
    """Set WAL + synchronous=NORMAL for ingest/search throughput.

    NORMAL is safe under WAL but can lose committed transactions on OS
    crash / power loss (no corruption). Acceptable tradeoff for benchmarks.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(
        dbapi_connection: DBAPIConnection,
        _connection_record: ConnectionPoolEntry,
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


ANSWER_PROMPT = """
You are asked to answer a question based on your memories of a conversation.

<instructions>
1. Prioritize memories that answer the question directly. Be meticulous about recalling details.
2. When there may be multiple answers to the question, think hard to remember and list all possible answers. Do not become satisfied with just the first few answers you remember.
3. When asked about time intervals or to count items, do not rush to answer immediately. Instead, carefully enumerate the items or subtract the times using numbers.
4. Your memories are episodic, meaning that they consist of only your raw observations of what was said. You may need to reason about or guess what the memories imply in order to answer the question.
5. The question may contain typos or be based on the asker's own unreliable memories. Do your best to answer the question using the most relevant information in your memories.
6. Your memories may include small or large jumps in time or context. You are not confused by this. You just did not bother to remember everything in between.
7. Your memories are ordered from earliest to latest.
</instructions>

<memories>
{memories}
</memories>

Question: {question}
Your short response to the question without fluff (no more than a couple of sentences):
"""

ANSWER_PROMPT = """
You are a helpful assistant with access to extensive conversation history.
When answering questions, carefully review the conversation history to identify and use any relevant user preferences, interests, or specific details they have mentioned.

<history>
{memories}
</history>

Question: {question}
"""


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )
    parser.add_argument(
        "--segment-db",
        default="locomo_segments.db",
        help="SQLite path for the segment store",
    )
    parser.add_argument(
        "--vector-db",
        default="locomo_vectors.db",
        help="SQLite path for the sqlite-vec vector store",
    )
    parser.add_argument(
        "--vector-search-limit",
        type=int,
        default=100,
        help="Number of vectors to retrieve",
    )
    parser.add_argument(
        "--expand-context",
        type=int,
        default=3,
        help="Number of context segments to expand",
    )
    parser.add_argument(
        "--max-num-segments",
        type=int,
        default=20,
        help="Max segments after unification",
    )
    parser.add_argument(
        "--separate-contexts",
        action="store_true",
        help="Use string_from_query_result, which keeps disconnected ranked contexts separated",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=15,
        help="Max concurrent questions per conversation",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI chat model for QA",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable the reranker and rank by embedding similarity only",
    )
    parser.add_argument(
        "--bm25-fusion",
        choices=["none", "additive", "rrf", "rsf"],
        default="none",
        help="BM25 fusion mode over the vector-retrieved candidate pool: "
        "'none' (default), 'additive' (calibrated additive with weighted "
        "semantic + sigmoid(bm25)), 'rrf' (Reciprocal Rank Fusion, k=60), "
        "'rsf' (Relative Score Fusion, max-normalized weighted average).",
    )
    parser.add_argument(
        "--bm25-fusion-weight",
        type=float,
        default=0.5,
        help="BM25 channel weight in [0.0, 1.0] for 'additive' and 'rsf' "
        "modes; semantic weight is 1 - weight (default: 0.5).",
    )
    args = parser.parse_args()

    locomo_data = load_locomo_dataset(args.data_path)

    segment_engine = create_async_engine(
        f"sqlite+aiosqlite:///{args.segment_db}",
        connect_args={"timeout": 30},
        pool_size=20,
        max_overflow=80,
    )
    _configure_sqlite_for_perf(segment_engine)
    segment_store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=segment_engine)
    )
    await segment_store.startup()

    vector_engine = create_async_engine(
        f"sqlite+aiosqlite:///{args.vector_db}",
        connect_args={"timeout": 30},
        pool_size=20,
        max_overflow=80,
    )
    _configure_sqlite_for_perf(vector_engine)
    vector_store = SQLiteVecVectorStore(
        SQLiteVecVectorStoreParams(engine=vector_engine)
    )
    await vector_store.startup()

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    st_model = SentenceTransformer(
        "google/embeddinggemma-300m",
        token=os.getenv("HF_TOKEN"),
    )
    embedder = SentenceTransformerEmbedder(
        SentenceTransformerEmbedderParams(
            model_name="google/embeddinggemma-300m",
            sentence_transformer=st_model,
            max_input_length=2048,
            batch_size=32,
        )
    )

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

    namespace = "locomo"

    async def qa_eval(memories: str, question: str) -> dict:
        start = time.monotonic()
        response = await openai_client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(
                        memories=memories, question=question
                    ),
                },
            ],
        )
        latency = time.monotonic() - start
        message = response.choices[0].message.content or ""
        usage = response.usage
        return {
            "response": message.strip(),
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "latency": latency,
        }

    results: dict[str, list[dict]] = {}

    for idx, item in enumerate(locomo_data):
        if "conversation" not in item:
            continue

        partition_key = f"group_{idx}"

        collection = await vector_store.open_collection(
            namespace=namespace, name=partition_key
        )
        if collection is None:
            print(f"No collection for group {idx}; skipping.")
            continue
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

        async def process_question(qa: dict) -> tuple[str, dict]:
            question = qa["question"]
            answer = qa.get("answer", "")
            category = str(qa["category"])
            evidence = qa.get("evidence", [])
            adversarial_answer = qa.get("adversarial_answer", "")

            memory_start = time.monotonic()
            query_result = await memory.query(
                query=question,
                vector_search_limit=args.vector_search_limit,
                expand_context=args.expand_context,
                format_options=_FORMAT_OPTIONS,
                bm25_fusion=args.bm25_fusion,
                bm25_fusion_weight=args.bm25_fusion_weight,
            )
            memory_latency = time.monotonic() - memory_start

            if args.separate_contexts:
                formatted_context = EventMemory.string_from_query_result(
                    query_result,
                    max_num_segments=args.max_num_segments,
                    format_options=_FORMAT_OPTIONS,
                )
            else:
                unified = EventMemory.build_query_result_context(
                    query_result, max_num_segments=args.max_num_segments
                )
                formatted_context = EventMemory.string_from_segment_context(
                    unified, format_options=_FORMAT_OPTIONS
                )

            response = await qa_eval(formatted_context, question)

            print(
                f"[group {idx}] cat={category}\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n"
                f"Response: {response['response']}\n"
                f"Memory retrieval time: {memory_latency:.2f} seconds\n"
                f"LLM response time: {response['latency']:.2f} seconds\n"
                f"MEMORIES_START\n{formatted_context}MEMORIES_END\n"
            )

            return category, {
                "question": question,
                "locomo_answer": str(answer),
                "model_answer": response["response"],
                "category": category,
                "evidence": evidence,
                "adversarial_answer": adversarial_answer,
                "conversation_memories": formatted_context,
                "memory_latency": memory_latency,
                "llm_latency": response["latency"],
            }

        qa_list = item["qa"]
        semaphore = asyncio.Semaphore(args.concurrency)
        responses = await asyncio.gather(
            *[async_with(semaphore, process_question(qa)) for qa in qa_list]
        )
        for category, response in responses:
            results.setdefault(category, []).append(response)

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)

    with open(args.target_path, "w") as f:
        json.dump(results, f, indent=4)

    await segment_store.shutdown()
    await vector_store.shutdown()
    await segment_engine.dispose()
    await vector_engine.dispose()
    await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig()
    logging.getLogger(
        "memmachine_server.episodic_memory.event_memory.event_memory"
    ).setLevel(logging.DEBUG)
    asyncio.run(main())
