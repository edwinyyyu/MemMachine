import argparse
import asyncio
import json
import os
import re
import time
from typing import Any

import boto3
import neo4j
import openai
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from memmachine.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
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
    DeclarativeMemory,
    DeclarativeMemoryParams,
)

language = "english"
stop_words = stopwords.words(language)


def default_tokenize(text: str) -> list[str]:
    """
    Preprocess the input text
    by removing non-alphanumeric characters,
    converting to lowercase,
    word-tokenizing,
    and removing stop words.

    Args:
        text (str): The input text to preprocess.

    Returns:
        list[str]: A list of tokens for use in BM25 scoring.
    """
    alphanumeric_text = re.sub(r"\W+", " ", text)
    lower_text = alphanumeric_text.lower()
    words = word_tokenize(lower_text, language)
    tokens = [word for word in words if word and word not in stop_words]
    return tokens


async def process_question(
    memory: DeclarativeMemory,
    question,
    answer,
    category,
    evidence,
    adversarial_answer,
):
    memory_start = time.time()
    results = await memory.search(
        query=question,
        max_num_episodes=200,
    )
    memory_end = time.time()

    gold_rr_ranks = []
    for idx, (_, _, episode_context) in enumerate(results):
        for episode in episode_context:
            if episode.user_metadata.get("dia_id") in evidence:
                gold_rr_ranks.append(idx)

    print(
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Memory retrieval time: {memory_end - memory_start:.2f} seconds\n"
    )
    return {
        "question": question,
        "locomo_answer": answer,
        "category": category,
        "evidence": evidence,
        "adversarial_answer": adversarial_answer,
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
        "gold_rr_ranks": gold_rr_ranks,
    }


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
            max_concurrent_transactions=1000,
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

    results: dict[str, Any] = {}
    for idx, item in enumerate(locomo_data):
        if "conversation" not in item:
            continue

        qa_list = item["qa"]

        print(f"Processing questions for group {idx}...")

        group_id = f"group_{idx}"

        memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=group_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
            )
        )

        async def respond_question(qa):
            question = qa["question"]
            answer = qa.get("answer", "")
            category = qa["category"]
            evidence = qa["evidence"]

            adversarial_answer = qa.get("adversarial_answer", "")

            question_response = await process_question(
                memory,
                question,
                answer,
                category,
                evidence,
                adversarial_answer,
            )
            return (
                category,
                question_response,
            )

        semaphore = asyncio.Semaphore(5)
        response_tasks = [
            async_with(
                semaphore,
                respond_question(qa),
            )
            for qa in qa_list
        ]

        responses = await asyncio.gather(*response_tasks)

        for category, response in responses:
            category_result = results.get(category, [])
            category_result.append(response)
            results[category] = category_result

    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
