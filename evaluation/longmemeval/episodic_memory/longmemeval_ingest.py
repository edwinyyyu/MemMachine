import argparse
import asyncio
import os
from datetime import datetime
from uuid import uuid4

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
    ContentType,
    DeclarativeMemory,
    DeclarativeMemoryParams,
    Episode,
)

EMBEDDER_MODEL = "nomic-ai/nomic-embed-text-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    data_path = args.data_path

    all_questions = load_longmemeval_dataset(data_path)
    num_questions = len(all_questions)
    print(f"{num_questions} total questions")

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
            range_index_creation_threshold=10000,
            vector_index_creation_threshold=10000,
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

    async def process_conversation(question: LongMemEvalItem):
        group_id = question.question_id
        session_ids = list(question.session_id_map.keys())

        memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=group_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
            )
        )

        # Sessions run sequentially: parallel add_episodes calls would fire
        # parallel embedder threads, racing nomic-bert's rotary cache.
        for session_id in session_ids:
            session = question.get_session(session_id)

            episodes = []
            for turn in session:
                timestamp = datetime.fromisoformat(turn.timestamp)
                episodes.append(
                    Episode(
                        uid=str(uuid4()),
                        timestamp=timestamp,
                        source="Assistant" if turn.role == "assistant" else "User",
                        content_type=ContentType.MESSAGE,
                        content=turn.content.strip(),
                        user_metadata={
                            "longmemeval_session_id": session_id,
                            "has_answer": turn.has_answer,
                            "turn_id": turn.index,
                        },
                    )
                )

            await memory.add_episodes(episodes=episodes)

    semaphore = asyncio.Semaphore(1)
    tasks = [
        async_with(semaphore, process_conversation(question))
        for question in all_questions
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
