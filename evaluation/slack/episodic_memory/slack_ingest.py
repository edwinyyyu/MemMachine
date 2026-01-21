import asyncio
import os
from datetime import UTC, datetime
from uuid import uuid4

import boto3
import neo4j
import openai
import pandas as pd
from datasets import load_dataset, Dataset
from dotenv import load_dotenv

from memmachine.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
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


async def main():
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

    async def process_conversation(workspace):
        print(f"Processing conversation for workspace {workspace}...")

        group_id = f"workspace_{workspace}"

        memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=group_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
            )
        )

        num_shards = 100
        for shard_index in range(num_shards):
            shard = workspace.shard(num_shards, shard_index)

            await memory.add_episodes(
                episodes=[
                    Episode(
                        uid=str(uuid4()),
                        timestamp=datetime.fromisoformat(
                            message.get("ts", "1970-01-01T00:00:00")
                        ).replace(tzinfo=UTC),
                        source=message.get("user", "Unknown"),
                        content_type=ContentType.MESSAGE,
                        content=message.get("text", ""),
                        user_metadata={
                            "channel": message.get("channel", "unknown"),
                        },
                    )
                    for message in shard
                    if message.get("text").strip()
                ]
            )

    ds = load_dataset("spencer/software_slacks")
    df = pd.DataFrame(ds["train"])
    df_unique = df.drop_duplicates()
    ds = Dataset.from_pandas(df_unique)

    tasks = [
        process_conversation(ds.filter(lambda x: x["workspace"] == workspace))
        for workspace in ("pythondev",)
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
