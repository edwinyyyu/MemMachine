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
from datetime import timedelta

from locomo_models import (
    attachment_suffix,
    datetime_from_locomo_time,
    load_locomo_dataset,
)
from embedder_factory import EMBEDDING_CHOICES, build_embedder
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


ANSWER_PROMPT_DETAILED = """
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

ANSWER_PROMPT_SIMPLE = """
You are a helpful assistant with access to extensive conversation history.
When answering questions, carefully review the conversation history to identify and use any relevant user preferences, interests, or specific details they have mentioned.

<history>
{memories}
</history>

Question: {question}
"""

# Mem0's new benchmark answer prompt (https://github.com/mem0ai/memory-benchmarks
# /blob/main/benchmarks/locomo/prompts.py — ANSWER_GENERATION_PROMPT).
# Reasoning Steps 1-7 with temporal grounding (requires reference_date) and
# expects output ending with "ANSWER:".
ANSWER_PROMPT_MEM0V2 = """You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory — important details are often scattered across many memories, including ones far down the list. Give equal weight to ALL memories regardless of position — a memory near the end is just as likely to contain the answer as one near the beginning. In these memories, "User" refers to the main person whose memories these are.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity. If the question asks "What does Person A like?" and a memory says "Person B likes X", do NOT use that memory to answer about Person A. In two-person conversations, both speakers' actions are relevant — if the question asks about person A and a memory attributes an action to person B (the other speaker), that information is still valid evidence from their shared conversations, but always check the attribution is correct.

## Step 3: COMBINE AND CROSS-REFERENCE
- COMBINE facts from multiple memories about the same topic. If one memory says "won first place" and another says "performed a piece titled X," those describe the same event — connect them.
- For listing/counting questions, extract EVERY distinct item from ALL memories. A single memory may contain multiple items. Think about what CATEGORIES of answers the question could have, then re-scan specifically for each category.
- For counting questions ("how many times", "how many X"), enumerate each distinct instance explicitly with its date or context BEFORE giving a final count. Do not estimate — list them out, then count the list.
- DECOMPOSE complex sentences: "an immersive X with Y, enjoys Z" contains multiple distinct facts. Each could be the answer.
- Connect related facts across memories: if one says "nearby lake" and another says "Lake Tahoe is great for kayaking", the nearby lake IS Lake Tahoe. If one says "bought X in Paris", infer the country is France.

## Step 4: SELECT THE BEST ANSWER
- Do NOT assume the highest-ranked memory is correct. Multiple memories may describe different events for the same topic. Compare each candidate's relevance to the SPECIFIC question, not its retrieval score. A lower-ranked memory that directly answers the question beats a higher-ranked one that is only tangentially related.
- ALWAYS choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description. Rate each candidate as HIGH specificity (name, title, number, specific activity) or LOW (generic description), and prefer HIGH.
- Report what someone actually DID, not what was offered or available to them. "Has not tried X yet" means X was NOT done — disqualify it. "Joined X" or "has done X" means it WAS done — prefer it.
- When multiple memories repeat the same generic fact, that repetition does NOT make it more correct than a single memory with a more specific answer.
- Photos depict what was IN the photo, not facts about someone's daily life. Prefer direct statements over photo descriptions for inferences.
- Re-read the question carefully before answering. If it asks "what aspect/type/kind", answer with the specific aspect. If it asks "what did they discover they both enjoy", answer with the specific thing, not the setting.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024.
- Calculate time relative to this date, NOT today. Never output 2025 or 2026.
- Use dates explicitly stated in memory text. Do not invent or estimate dates.
- When a question asks what someone "shared" or "mentioned" on a date, that date is when they TALKED about it — look for events shortly BEFORE that date.
- For "how long" questions, find the start and end dates explicitly, then compute the duration. Do not guess.
- TEMPORAL DISAMBIGUATION: When you find MULTIPLE instances of similar events at different dates, enumerate them all with their dates before picking. If the question uses past tense + "the" → select the instance closest to (and before) the reference date. If future tense ("plans to", "going to") → select the earliest planned date. NEVER default to the first-mentioned or highest-scored instance — the DATE determines the answer.

## Step 6: INCLUSION CHECK (for lists and counts)
If you found items during reasoning that you're tempted to exclude from your answer — STOP. Include them unless you have STRONG evidence they are wrong. The most common mistake is finding relevant items but then dropping them due to overly strict filtering. More items is better than fewer when there is supporting evidence.
- For counting: after enumerating, re-verify each item. Check for duplicates (same event described differently) and ensure you haven't missed items from memories late in the list.
- The question assumes something happened. Find WHAT happened, don't say nothing happened.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified", "not mentioned", "no record", or "the memories don't say" — if ANY memory contains relevant information, give the best answer from available evidence. No hedging, no caveats. If the question asks for a list, include ALL items found. NEVER return an empty answer when relevant memories exist.
- NEVER generate specific names, titles, places, or dates that do not appear in any memory above. If no memory contains the specific detail the question asks for, answer with what the memories DO contain rather than guessing.
- For open-domain/opinion questions ("Would X do Y?", "Is X considered Z?"):
  * Follow the DIRECT causal reasoning in the memories. Do NOT construct elaborate counter-arguments.
  * "Would X still do Y without Z?" — If memories show X does Y BECAUSE of Z, then without Z, answer "likely no."
  * "Would X do Y again soon?" — If the most recent attempt involved a bad experience (accident, scare, trauma), answer "likely no." A recent negative experience outweighs historical positive patterns.
  * For trait questions ("Is X considered Z?"): weigh ALL evidence including symbolic/indirect references. If there is SOME but not strong evidence, answer with a qualified degree ("somewhat") rather than flat "no."

# Instructions

## Misc

1. Make reasonable deductions based on your memories. Memory shows store with a lot of working people -> store employs a lot of people
2. If a memory describes something recognizable (e.g., "romantic drama about memory and relationships"), you may name it (e.g., "Eternal Sunshine of the Spotless Mind").
3. Use domain knowledge to connect facts: a game exclusive to one platform implies ownership of that platform. An unnamed company deal can be linked to a previously expressed brand preference.

{memories}

Question: {question}

Work through Steps 1-7, then give your final answer after "ANSWER:".
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
        choices=["none", "additive", "rrf", "rsf", "noisy-or"],
        default="none",
        help="BM25 fusion mode over the vector-retrieved candidate pool: "
        "'none' (default), 'additive' (calibrated additive with weighted "
        "semantic + sigmoid(bm25)), 'rrf' (Reciprocal Rank Fusion, k=60), "
        "'rsf' (Relative Score Fusion, max-normalized weighted average), "
        "'noisy-or' (1 - (1-sem)*(1-bm25_sigmoid), probabilistic union).",
    )
    parser.add_argument(
        "--bm25-fusion-weight",
        type=float,
        default=0.5,
        help="BM25 channel weight in [0.0, 1.0] for 'additive' and 'rsf' "
        "modes; semantic weight is 1 - weight (default: 0.5).",
    )
    parser.add_argument(
        "--answer-prompt",
        choices=["simple", "detailed", "mem0v2"],
        default="simple",
        help="Answer prompt variant: 'simple' (default, original — helpful "
        "assistant), 'detailed' (mem0-style with 7 numbered instructions), "
        "'mem0v2' (mem0's new benchmark prompt with reasoning Steps 1-7 + "
        "temporal grounding). When using non-default, append matching tag to "
        "the target-path filename for unambiguous identification.",
    )
    parser.add_argument(
        "--answer-with-raw-events",
        action="store_true",
        help="After top-K retrieval, dedup retrieved segments by event_uuid "
        "and replace formatted context with the original raw message text "
        "(looked up from --data-path by group_idx + timestamp). Tests "
        "whether the segmenter's score advantage comes from retrieval "
        "signal vs answering material quality. Token budget will increase "
        "substantially compared to the rewritten segments.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        choices=EMBEDDING_CHOICES,
        help="Must match the embedding model used at ingest.",
    )
    parser.add_argument(
        "--timestamp-format",
        choices=["full", "short"],
        default="full",
        help="Rendered segment-header timestamp. 'full' (default) = Babel "
        "full date + short time ('Monday, August 14, 2023, 2:24 PM', "
        "~14 tok). 'short' = Babel short date, no time ('8/14/23', ~5 tok) "
        "-- cuts per-segment header overhead so more segments fit a fixed "
        "answer-token budget. Affects rendering only, not retrieval.",
    )
    args = parser.parse_args()

    global _FORMAT_OPTIONS
    if args.timestamp_format == "short":
        _FORMAT_OPTIONS = FormatOptions(date_style="short", time_style=None)

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
    embedder = build_embedder(args.embedding_model, openai_client)

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

    answer_prompt_by_name = {
        "simple": ANSWER_PROMPT_SIMPLE,
        "detailed": ANSWER_PROMPT_DETAILED,
        "mem0v2": ANSWER_PROMPT_MEM0V2,
    }
    selected_answer_prompt = answer_prompt_by_name[args.answer_prompt]

    namespace = "locomo"

    async def qa_eval(memories: str, question: str) -> dict:
        start = time.monotonic()
        format_kwargs = {"memories": memories, "question": question}
        if args.answer_prompt == "mem0v2":
            format_kwargs["reference_date"] = "January 1, 2024"
        response = await openai_client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": selected_answer_prompt.format(**format_kwargs),
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

        # Build (event_timestamp -> (speaker, raw_message_text)) lookup for
        # this conversation, matching the exact (session_datetime +
        # message_index * 1s) scheme used by locomo_ingest.
        event_text_lookup: dict = {}
        if args.answer_with_raw_events:
            conversation = item["conversation"]
            session_idx = 0
            while True:
                session_idx += 1
                session_id = f"session_{session_idx}"
                if session_id not in conversation:
                    break
                session = conversation[session_id]
                session_dt = datetime_from_locomo_time(
                    conversation[f"{session_id}_date_time"]
                )
                for msg_idx, msg in enumerate(session):
                    ts = session_dt + msg_idx * timedelta(seconds=1)
                    raw_text = msg["text"] + attachment_suffix(msg)
                    event_text_lookup[ts] = (msg["speaker"], raw_text)

        async def process_question(qa: dict) -> tuple[str, dict]:
            question = qa["question"]
            answer = qa.get("answer", "")
            category = str(qa["category"])
            evidence = qa.get("evidence", [])
            adversarial_answer = qa.get("adversarial_answer", "")

            memory_start = time.monotonic()
            query_result = None
            for attempt in range(5):
                try:
                    query_result = await memory.query(
                        query=question,
                        vector_search_limit=args.vector_search_limit,
                        expand_context=args.expand_context,
                        format_options=_FORMAT_OPTIONS,
                        bm25_fusion=args.bm25_fusion,
                        bm25_fusion_weight=args.bm25_fusion_weight,
                    )
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise
                    backoff = min(2**attempt, 30) + (attempt * 0.5)
                    print(
                        f"[group {idx}] memory.query failed (attempt {attempt + 1}/5): "
                        f"{type(exc).__name__}: {exc} — retrying in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
            memory_latency = time.monotonic() - memory_start

            if args.answer_with_raw_events:
                unified = EventMemory.build_query_result_context(
                    query_result, max_num_segments=args.max_num_segments
                )
                seen_event_uuids: set = set()
                raw_event_lines: list[str] = []
                for seg in sorted(unified, key=lambda s: s.timestamp):
                    if seg.event_uuid in seen_event_uuids:
                        continue
                    seen_event_uuids.add(seg.event_uuid)
                    entry = event_text_lookup.get(seg.timestamp)
                    if entry is None:
                        continue
                    speaker_name, raw_text = entry
                    # Date-only ISO timestamp for the raw-event header. The
                    # Babel "full" style ("Monday, May 8, 2023, 1:56 PM") costs
                    # ~14 tok/line; "YYYY-MM-DD" costs ~6. On LoCoMo qkmin3p K=7
                    # this cuts the answer context 378t -> 316t with no accuracy
                    # change (87.63 vs 87.60 c1234, gpt-5 judge, 2 runs each) --
                    # making K=7 fit the token budget. Time-of-day dropped:
                    # list order already preserves intra-day sequence.
                    raw_ts = seg.timestamp
                    if _FORMAT_OPTIONS.timezone is not None:
                        raw_ts = raw_ts.astimezone(_FORMAT_OPTIONS.timezone)
                    ts_str = raw_ts.strftime("%Y-%m-%d")
                    raw_event_lines.append(
                        f"[{ts_str}] {speaker_name}: {raw_text.strip()}"
                    )
                formatted_context = "\n".join(raw_event_lines)
            elif args.separate_contexts:
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
