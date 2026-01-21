import asyncio
import os
import time
from contextlib import suppress

import boto3
import neo4j
import openai
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
    DeclarativeMemory,
    DeclarativeMemoryParams,
)

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


async def process_question(
    memory: DeclarativeMemory,
    model: openai.AsyncOpenAI,
    question,
):
    memory_start = time.time()
    chunks = await memory.search(
        query=question,
        max_num_episodes=20,
    )
    memory_end = time.time()

    formatted_context = memory.string_from_episode_context(chunks)
    prompt = ANSWER_PROMPT.format(memories=formatted_context, question=question)

    llm_start = time.time()
    rsp = None
    while rsp is None:
        with suppress(openai.APIError):
            rsp = await model.responses.create(
                model="gpt-4.1-mini",
                max_output_tokens=4096,
                temperature=0.0,
                top_p=1,
                timeout=10,
                input=[{"role": "user", "content": prompt}],
            )
    llm_end = time.time()

    rsp_text = rsp.output_text

    print(
        f"Question: {question}\n"
        f"Response: {rsp_text}\n"
        f"Memory retrieval time: {memory_end - memory_start:.2f} seconds\n"
        f"LLM response time: {llm_end - llm_start:.2f} seconds\n"
        f"MEMORIES START\n{formatted_context}MEMORIES END\n"
    )
    return {
        "question": question,
        "model_answer": rsp_text,
        "conversation_memories": formatted_context,
    }


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

    model = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    memory = DeclarativeMemory(
        DeclarativeMemoryParams(
            session_id="workspace_Dataset({\n    features: ['workspace', 'channel', 'text', 'ts', 'user', '__index_level_0__'],\n    num_rows: 106262\n})",
            vector_graph_store=vector_graph_store,
            embedder=embedder,
            reranker=reranker,
        )
    )

    question_response = await process_question(
        memory,
        model,
        question="Why did Rubie need to remove the last digit?",
    )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
