import argparse
import asyncio
import json
import os
import time
from typing import cast
from uuid import UUID

from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    load_longmemeval_dataset,
)
from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store import VectorStoreCollection
from memmachine_server.common.vector_store.qdrant_vector_store_exact import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import Text
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartition,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

_SEGMENT_UUID_KEY = "_segment_uuid"


async def eval_query(
    embedder: Embedder,
    collection: VectorStoreCollection,
    segment_store_partition: SegmentStorePartition,
    query: str,
    *,
    max_num_segments: int = 20,
    expand_context: int = 0,
):
    """
    Query that returns detailed ranked segment contexts and per-derivative stats
    (no reranking or limiting), using the new segment store API.
    """
    query_embedding = (await embedder.search_embed([query]))[0]

    [query_result] = await collection.query(
        query_vectors=[query_embedding],
        limit=min(5 * max_num_segments, 200),
        return_vector=False,
        return_properties=True,
    )

    # Extract derivative → segment UUID mapping from vector metadata.
    matched_derivative_uuids: list[UUID] = []
    derivative_to_segment: dict[UUID, UUID] = {}
    for match in query_result.matches:
        derivative_uuid = match.record.uuid
        segment_uuid = UUID(
            str(
                cast(
                    dict[str, PropertyValue],
                    match.record.properties,
                )[_SEGMENT_UUID_KEY]
            )
        )
        matched_derivative_uuids.append(derivative_uuid)
        derivative_to_segment[derivative_uuid] = segment_uuid

    # Deduplicate seed segment UUIDs preserving similarity order.
    seed_segment_uuids = list(dict.fromkeys(derivative_to_segment.values()))

    expand_context = min(max(0, expand_context), max_num_segments - 1)
    max_backward_segments = expand_context // 3
    max_forward_segments = expand_context - max_backward_segments

    segment_contexts_by_seed = await segment_store_partition.get_segment_contexts(
        seed_segment_uuids=seed_segment_uuids,
        max_backward_segments=max_backward_segments,
        max_forward_segments=max_forward_segments,
    )

    # Build ranked results preserving similarity order.
    ranked_segment_contexts: list[dict] = []
    for seed_uuid in seed_segment_uuids:
        if seed_uuid not in segment_contexts_by_seed:
            continue
        context_segments = segment_contexts_by_seed[seed_uuid]
        # Find the seed segment in the context.
        seed_segment = next((s for s in context_segments if s.uuid == seed_uuid), None)
        ranked_segment_contexts.append(
            {
                "rank": len(ranked_segment_contexts),
                "seed_segment": seed_segment,
                "segment_context": context_segments,
            }
        )

    # Compute per-derivative stats: count how many segments each derivative maps to.
    # In the new model each derivative maps to exactly one segment, but we count
    # only derivatives whose segment was actually returned.
    returned_segments = {
        seed_uuid
        for seed_uuid in seed_segment_uuids
        if seed_uuid in segment_contexts_by_seed
    }
    segments_per_derivative: dict[UUID, int] = {
        derivative_uuid: 1
        for derivative_uuid, segment_uuid in derivative_to_segment.items()
        if segment_uuid in returned_segments
    }

    return ranked_segment_contexts, segments_per_derivative


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

    async def process_question(question: LongMemEvalItem):
        partition_key = question.question_id

        collection = await vector_store.open_collection(
            namespace="longmemeval", name=question.question_id
        )

        segment_store_partition = await segment_store.open_or_create_partition(
            partition_key
        )

        search_query = f"User: {question.question}"

        memory_start = time.monotonic()
        ranked_contexts, segments_per_derivative = await eval_query(
            embedder=embedder,
            collection=collection,
            segment_store_partition=segment_store_partition,
            query=search_query,
            max_num_segments=1000,
            expand_context=0,
        )
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

        total_segments = sum(len(sc["segment_context"]) for sc in ranked_contexts)
        derivative_segment_counts = list(segments_per_derivative.values())

        result = {
            "question_id": question.question_id,
            "question_date": question.question_date,
            "question": question.question,
            "answer": question.answer,
            "answer_turn_indices": question.answer_turn_indices,
            "question_type": question.question_type.value,
            "abstention": question.abstention_question,
            "memory_latency": memory_latency,
            "total_ranked_contexts": len(ranked_contexts),
            "total_segments": total_segments,
            "total_derivatives": len(derivative_segment_counts),
            "min_segments_per_derivative": min(derivative_segment_counts)
            if derivative_segment_counts
            else 0,
            "max_segments_per_derivative": max(derivative_segment_counts)
            if derivative_segment_counts
            else 0,
            "avg_segments_per_derivative": sum(derivative_segment_counts)
            / len(derivative_segment_counts)
            if derivative_segment_counts
            else 0,
            "segment_contexts": [
                {
                    "rank": sc["rank"],
                    "seed_segment_uuid": str(sc["seed_segment"].uuid)
                    if sc["seed_segment"]
                    else None,
                    "segments": [
                        {
                            "uuid": str(segment.uuid),
                            "event_uuid": str(segment.event_uuid),
                            "index": segment.index,
                            "offset": segment.offset,
                            "timestamp": segment.timestamp.isoformat(),
                            "context": segment.context.model_dump()
                            if segment.context
                            else None,
                            "properties": segment.properties,
                            "text": segment.block.text
                            if isinstance(segment.block, Text)
                            else None,
                        }
                        for segment in sc["segment_context"]
                    ],
                }
                for sc in ranked_contexts
            ],
        }

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)

        return result

    semaphore = asyncio.Semaphore(50)
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
    asyncio.run(main())
