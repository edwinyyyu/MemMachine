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
from memmachine.common.utils import merge_intersecting_sets

ANSWER_PROMPT = """
You are asked to answer a question based on your episodic memories at different points in time.

<instructions>
1. You have multiple turns to answer the question. Each of your responses represents a single turn.
2. The actions available to you in this turn are provided in between <actions></actions> tags. All actions are mututally exclusive within a single turn.
3. Your existing episodic memories are provided in between <memories></memories> tags. Each memory is labeled with an ID (e.g., id0, id1, etc.) for reference.
4. You may explain your reasoning process in each turn before the action-specific tags in at most 50 words.
</instructions>

<actions>
1. You may use your turn to recall the surrounding episodic context for existing memories if needed.
    a. To recall surrounding episodic context, output a JSON array of the string IDs of the memories you want to recall more context for in between <recall></recall> tags at the end of your response for this turn (e.g. <recall>["id0", "id2"]</recall>).
    b. The newly recalled context will be added to your memories for the next turn.
2. You may use your turn to retrieve new episodic memories using self-generated retrieval cues.
    a. To retrieve new memories, output a JSON array containing your string retrieval cues in between <retrieve></retrieve> tags at the end of your response for this turn (e.g. <retrieve>["first president of the United States", "When did the Curiosity rover land on Mars?"]</retrieve>).
    b. The new memories will be added to your memories for the next turn.
    c. You remember the retrieval cues that you have already used in between <cues></cues> tags, after your episodic memories. You should only use new retrieval cues that are sufficiently different from your previous cues.
3. You should use your turn to submit a final answer to the question using your memories when you have enough memories.
    a. To submit your final answer, output it in between <answer></answer> tags at the end of your response for this turn (e.g. <answer>The first president of the United States is George Washington.</answer>).
    b. Your final answer to the question should be without fluff (no more than a couple of sentences).
</actions>

<guidelines>
1. Prioritize memories that answer the question directly. Be meticulous about retrieving details.
2. When there may be multiple answers to the question, think hard to remember and list all possible answers. Do not become satisfied with just the first few answers you remember.
3. When asked about time intervals or to count items, do not rush to answer immediately. Instead, carefully enumerate the items or subtract the times using numbers.
4. Include year, month, and day when your answer involves dates.
5. Your memories are episodic, meaning that they consist of only your raw observations of what was said. You may need to reason about or guess what the memories imply in order to answer the question.
6. The question may contain typos or be based on the asker's own unreliable memories. Do your best to answer the question using the most relevant information in your memories.
7. Your memories may include small or large jumps in time or context. You may recall episodic context to bridge these gaps before answering the question.
8. If it is reasonable that the answer to the question is part of the surrounding context of a memory, you should recall the episodic context for that memory.
9. Use your memories to help you generate new retrieval cues that are sufficiently different from your previous cues.
10. If you do not find any relevant memories, you may provide your best guess based on common sense or general knowledge as your final answer.
</guidelines>

<memories>
{memories}
</memories>

<cues>
{cues}
</cues>

Question: {question}
Your response for this turn:
"""

FINAL_ANSWER_PROMPT = """
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


def parse_response(response_text: str) -> dict[str, str | list[str]]:
    response_dict: dict[str, list[str]] = {}

    tag_pattern = re.compile(r"<(.*?)>(.*?)</\1>", re.DOTALL)
    matches = list(tag_pattern.finditer(response_text))

    if matches:
        last_match = matches[-1]
        tag = last_match.group(1)
        content = last_match.group(2).strip()
        if tag in ("recall", "retrieve"):
            try:
                response_dict[tag] = json.loads(content)
            except json.JSONDecodeError:
                response_dict[tag] = []
        else:
            response_dict[tag] = content

        response_dict["reasoning"] = response_text[: last_match.start()].strip()
    else:
        response_dict["reasoning"] = response_text.strip()

    return response_dict


async def process_question(
    memory: DeclarativeMemory,
    model: openai.AsyncOpenAI,
    question,
    answer,
    category,
    evidence,
    adversarial_answer,
):
    start = time.monotonic()

    all_reasoning = []
    cues = [question]

    episode_contexts = await memory.search(
        query=question,
        limit=5,
    )

    answer_text = "No answer."
    for turn in range(1, 10 + 1):
        ided_contexts = {
            f"id{index}": episode_context
            for index, episode_context in enumerate(episode_contexts)
        }

        formatted_contexts = {
            memory_id: memory.string_from_episode_context(episode_context)
            for memory_id, episode_context in ided_contexts.items()
        }

        if turn < 10:
            prompt = ANSWER_PROMPT.format(
                memories=json.dumps(formatted_contexts),
                cues=json.dumps(cues),
                question=question,
            )
        else:
            prompt = FINAL_ANSWER_PROMPT.format(
                memories=json.dumps(formatted_contexts), question=question
            )

        response_text = ""
        for _ in range(3):
            try:
                response = await model.responses.create(
                    model="gpt-4.1-mini",
                    max_output_tokens=4096,
                    timeout=10,
                    input=[{"role": "user", "content": prompt}],
                )
                response_text = response.output_text.strip()
                break
            except Exception:
                pass

        if turn > 9:
            answer_text = response_text
            break

        response_dict = parse_response(response_text)

        reasoning = response_dict.get("reasoning", "")
        all_reasoning.append(reasoning)

        if "answer" in response_dict:
            answer_text = response_dict["answer"]
            break
        elif "recall" in response_dict:
            recall_ids = response_dict.get("recall", [])
            recall_ids = [recall_id.removeprefix("id") for recall_id in recall_ids]
            recall_ids = [
                int(recall_id) for recall_id in recall_ids if recall_id.isdigit()
            ]
            recall_ids = [
                recall_id
                for recall_id in recall_ids
                if 0 <= recall_id < len(episode_contexts)
            ]
            expand_context_awaitables = [
                memory.expand_episode_context(episode_contexts[recall_id])
                for recall_id in recall_ids
            ] + [
                memory.cued_recall(
                    formatted_contexts[f"id{recall_id}"],
                    exclude=ided_contexts[f"id{recall_id}"],
                )
                for recall_id in recall_ids
            ]
            expanded_contexts = await asyncio.gather(*expand_context_awaitables)
            episode_contexts += expanded_contexts
        elif "retrieve" in response_dict:
            retrieve = response_dict.get("retrieve", [])
            cued_recall_awaitables = [
                memory.cued_recall(cue, exclude=None) for cue in retrieve
            ]
            cues.extend(cue for cue in retrieve if cue not in cues)
            memories_cued_episodes = await asyncio.gather(*cued_recall_awaitables)
            episode_contexts += [
                [episode]
                for memory_cued_episodes in memories_cued_episodes
                for episode in memory_cued_episodes
            ]
        else:
            break

        episode_context_sets = [
            set(episode_context) for episode_context in episode_contexts
        ]
        episode_context_sets = merge_intersecting_sets(episode_context_sets)
        episode_contexts = [
            sorted(
                episode_context_set,
                key=lambda episode: (episode.timestamp, episode.uid),
            )
            for episode_context_set in episode_context_sets
        ]
        episode_contexts = sorted(
            episode_contexts, key=lambda episode_context: episode_context[0].timestamp
        )

    end = time.monotonic()

    print(
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Response: {answer_text}\n"
        f"Response time: {end - start:.2f} seconds\n"
        f"Number of turns: {turn}\n"
        f"Cues used: {cues}\n"
        f"MEMORIES START\n{
            '\n'.join(
                f'{memory_id}:\n{memory_content}'
                for memory_id, memory_content in formatted_contexts.items()
            )
        }\nMEMORIES END\n"
        f"Reasoning steps:\n{'\n'.join(all_reasoning)}\n",
        flush=True,
    )
    return {
        "question": question,
        "locomo_answer": answer,
        "model_answer": answer_text,
        "category": category,
        "evidence": evidence,
        "adversarial_answer": adversarial_answer,
        "num_turns": turn,
        "cues": cues,
        "conversation_memories": formatted_contexts,
        "reasoning_steps": all_reasoning,
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

    model = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
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
                model,
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
