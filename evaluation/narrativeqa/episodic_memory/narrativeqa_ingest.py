import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import boto3
import neo4j
import openai
import pandas as pd
from dotenv import load_dotenv

from memmachine.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
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

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--meta-path", required=True, help="Path to the meta file")
    parser.add_argument("--data-path", required=True, help="Path to the data file")

    args = parser.parse_args()

    meta_path = args.meta_path
    data_path = args.data_path

    narrativeqa_meta = pd.read_csv(meta_path)

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
            vector_index_creation_threshold=1,
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

    language_model = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=openai_client,
            model="gpt-4.1-nano",
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

    async def process_document(row):
        document_id = row["document_id"]

        try:
            with open(f"{data_path}/{document_id}.content", "r") as f:
                document_content = f.read()
        except:
            print(f"Failed to read document {document_id}")
            return

        if not document_content.strip():
            print(f"Empty content for document {document_id}")
            return

        print(
            f"Processing document {document_id}"
        )

        memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=document_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
                language_model=language_model,
            )
        )

        await memory.add_episodes(
            episodes=[
                Episode(
                    uid=str(uuid4()),
                    timestamp=datetime.min.replace(tzinfo=UTC),
                    source="",
                    content_type=ContentType.TEXT,
                    content=document_content,
                )
            ]
        )

    semaphore = asyncio.Semaphore(1)
    tasks = [
        async_with(semaphore, process_document(row))
        for _, row in narrativeqa_meta[:50].iterrows()]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
