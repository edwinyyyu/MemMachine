import argparse
import asyncio
import json
import os
import time

import neo4j
from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    load_longmemeval_dataset,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

from memmachine.common.embedder.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    SentenceTransformerEmbedderParams,
)
from memmachine.common.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerParams,
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

EMBEDDER_MODEL = "nomic-ai/nomic-embed-text-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"


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
        )
    )

    sentence_transformer = SentenceTransformer(
        EMBEDDER_MODEL,
        trust_remote_code=True,
    )
    # Pre-warm the rotary-embedding cache at full sequence length so concurrent
    # encode calls don't race on cache reallocation (nomic-bert isn't
    # thread-safe with respect to its cos/sin cache).
    sentence_transformer.encode(["x " * 2048], show_progress_bar=False)

    embedder = SentenceTransformerEmbedder(
        SentenceTransformerEmbedderParams(
            model_name=EMBEDDER_MODEL,
            sentence_transformer=sentence_transformer,
            max_input_length=2048,
        )
    )

    reranker = CrossEncoderReranker(
        CrossEncoderRerankerParams(
            cross_encoder=CrossEncoder(RERANKER_MODEL),
        )
    )

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

        search_query = f"User: {question.question}"

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
