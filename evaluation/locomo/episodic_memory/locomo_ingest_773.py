import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import neo4j
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel

from memmachine.common.embedder.flag_embedder import (
    FlagEmbedder,
    FlagEmbedderParams,
)
from memmachine.common.reranker.identity_reranker import IdentityReranker
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


def datetime_from_locomo_time(locomo_time_str: str) -> datetime:
    return datetime.strptime(locomo_time_str, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=UTC
    )


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    data_path = args.data_path

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

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

    embedder = FlagEmbedder(
        FlagEmbedderParams(flag_model=BGEM3FlagModel("BAAI/bge-m3"))
    )

    reranker = IdentityReranker()

    async def process_conversation(idx, item):
        if "conversation" not in item:
            return

        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        print(
            f"Processing conversation for group {idx} with speakers {speaker_a} and {speaker_b}..."
        )

        group_id = f"group_{idx}"

        memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=group_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
            )
        )

        session_idx = 0

        while True:
            session_idx += 1
            session_id = f"session_{session_idx}"

            if session_id not in conversation:
                break

            session = conversation[session_id]
            session_datetime = datetime_from_locomo_time(
                conversation[f"{session_id}_date_time"]
            )

            await memory.add_episodes(
                episodes=[
                    Episode(
                        uid=str(uuid4()),
                        timestamp=session_datetime
                        + message_index * timedelta(seconds=1),
                        source=message["speaker"],
                        content_type=ContentType.MESSAGE,
                        content=message["text"]
                        + (
                            f" [Attached {blip_caption}: {image_query}]"
                            if (
                                (
                                    (
                                        (blip_caption := message.get("blip_caption"))
                                        or True
                                    )
                                    and ((image_query := message.get("query")) or True)
                                )
                                and blip_caption
                                and image_query
                            )
                            else (
                                f" [Attached {blip_caption}]"
                                if blip_caption
                                else (
                                    f" [Attached a photo: {image_query}]"
                                    if image_query
                                    else ""
                                )
                            )
                        ),
                        user_metadata={
                            "locomo_session_id": session_id,
                            "dia_id": message.get("dia_id", ""),
                        },
                    )
                    for message_index, message in enumerate(session)
                ]
            )

    tasks = [process_conversation(idx, item) for idx, item in enumerate(locomo_data)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
