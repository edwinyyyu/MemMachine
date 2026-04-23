"""Vectorize-style BEAM search + answer generation against EventMemory.

Faithful adaptation of
https://github.com/vectorize-io/agent-memory-benchmark/blob/main/src/memory_bench/dataset/beam.py
`build_rag_prompt`, including the category-specific generation prompts that
leak judge-side ground truth (`ordering_tested`, `preference_being_tested`,
`instruction_being_tested`, `compliance_indicators`, `time_points`,
`calculation_required`, `why_unanswerable`, rubric items for summarization)
into the model's context at answer-generation time.

**This methodology is NOT comparable to the official BEAM paper.** It is
included here so our own memory system can be benchmarked on Vectorize's
setup for head-to-head comparison against their published numbers, with full
awareness of the caveat.

EventMemory adapter notes:

- Retrieval context is built from `EventMemory.string_from_segment_context`,
  the canonical formatter. It emits `[timestamp] Source: "text"` per segment.
  Content blocks never hold timestamps or speaker identity — those live on
  `Event.timestamp` and `MessageContext.source` and are surfaced by the
  formatter at retrieval time.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import boto3
from dotenv import load_dotenv
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import FormatOptions
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent))
from beam_models import (
    BEAMConversation,
    BEAMQuestion,
    QuestionCategory,
    load_beam_dataset,
)
from llm_provider import PROVIDER_OPENAI, PROVIDERS, make_chat_client

_DEFAULT_VECTORIZE_ANSWER_MODEL = "gpt-5-mini"

NAMESPACE = "beam"

EMBEDDER_OPENAI = "openai"
EMBEDDER_ST = "sentence-transformer"
EMBEDDERS: tuple[str, ...] = (EMBEDDER_OPENAI, EMBEDDER_ST)
DEFAULT_ST_MODEL = "BAAI/bge-large-en-v1.5"

# BEAM's timeline is content-embedded, not wall-clock. Suppress timestamp
# prefixes on retrieved context (both for reranker scoring and for what the
# generator LLM sees) so the model reasons from in-content dates rather than
# our synthetic Event.timestamps.
_NO_TIMESTAMPS = FormatOptions(date_style=None, time_style=None)


def _guidance_abstention(q: BEAMQuestion) -> str:
    guidance = (
        "IMPORTANT — ABSTENTION TASK: This question may be about something NOT "
        "mentioned in the conversation. If the context does not explicitly contain "
        "the requested information, you MUST respond with: "
        '"Based on the provided chat, there is no information related to [topic]." '
        "Do NOT fabricate, infer, or guess. Only answer if the information is "
        "directly present in the retrieved context."
    )
    if q.why_unanswerable:
        guidance += f"\n\nHint about why this may be unanswerable: {q.why_unanswerable}"
    return guidance


def _guidance_event_ordering(q: BEAMQuestion) -> str:
    topics_str = ""
    if q.ordering_tested:
        topics_str = "\n" + "\n".join(f"  - {t}" for t in q.ordering_tested)
    return (
        "IMPORTANT — EVENT ORDERING TASK: The following specific topics need to be "
        "listed in the order they were FIRST mentioned in the conversation."
        + (f"\n\nTopics to order:{topics_str}\n" if topics_str else "")
        + "\nFor each topic, find the FIRST time it was mentioned (use Turn IDs and "
        "timestamps as ordering cues). List ONLY these topics in chronological order. "
        "Number each item. Do NOT add extra topics not listed above."
    )


def _guidance_contradiction_resolution(q: BEAMQuestion) -> str:
    return (
        "IMPORTANT — CONTRADICTION TASK: The conversation contains contradictory "
        "statements. Identify both contradictory statements and explicitly note the "
        "contradiction. Do not give a definitive answer — instead say what was "
        "claimed at different points."
    )


def _guidance_knowledge_update(q: BEAMQuestion) -> str:
    return (
        "IMPORTANT — KNOWLEDGE UPDATE TASK: Information was updated during the "
        "conversation. Report ONLY the most recent value. If you see older "
        "conflicting info, note the update."
    )


def _guidance_temporal_reasoning(q: BEAMQuestion) -> str:
    guidance = (
        "IMPORTANT — TEMPORAL REASONING TASK: Calculate the exact time duration. "
        "Show your arithmetic step by step."
    )
    if q.time_points:
        guidance += f"\n\nKey time points from context: {'; '.join(q.time_points)}"
    if q.calculation_required:
        guidance += f"\n\nCalculation hint: {q.calculation_required}"
    return guidance


def _guidance_preference_following(q: BEAMQuestion) -> str:
    guidance = (
        "IMPORTANT — PREFERENCE FOLLOWING TASK: The user has stated a specific "
        "preference earlier in the conversation. Your answer MUST respect and "
        "comply with that stated preference."
    )
    if q.preference_being_tested:
        guidance += (
            f"\n\nUser's preference (from conversation): {q.preference_being_tested}"
        )
    return guidance


def _guidance_instruction_following(q: BEAMQuestion) -> str:
    guidance = (
        "IMPORTANT — INSTRUCTION FOLLOWING TASK: The user gave a specific formatting "
        "or style instruction earlier in the conversation. Follow it exactly."
    )
    if q.instruction_being_tested:
        guidance += f"\n\nInstruction to follow: {q.instruction_being_tested}"
    if q.compliance_indicators:
        guidance += f"\n\nCompliance indicators: {', '.join(q.compliance_indicators)}"
    return guidance


def _guidance_summarization(q: BEAMQuestion) -> str:
    hints = [r.split(": ", 1)[-1] if ": " in r else r for r in q.rubric]
    hints_str = "\n".join(f"  - {h}" for h in hints) if hints else ""
    return (
        "IMPORTANT — SUMMARIZATION TASK: Provide a comprehensive chronological "
        "summary. Be specific about dates, versions, and key technical decisions."
        + (
            f"\n\nMake sure your summary covers these key aspects:\n{hints_str}"
            if hints_str
            else ""
        )
    )


def _guidance_multi_session_reasoning(q: BEAMQuestion) -> str:
    return (
        "IMPORTANT — MULTI-SESSION REASONING TASK: The answer requires combining "
        "facts from multiple parts of the conversation. Find ALL relevant facts "
        "across the entire history before answering. "
        "When counting items, be precise — count only the distinct types/categories "
        "explicitly mentioned, not every sub-feature. Give a concise, direct answer."
    )


_CATEGORY_GUIDANCE_BUILDERS: dict[str, Callable[[BEAMQuestion], str]] = {
    QuestionCategory.ABSTENTION.value: _guidance_abstention,
    QuestionCategory.EVENT_ORDERING.value: _guidance_event_ordering,
    QuestionCategory.CONTRADICTION_RESOLUTION.value: _guidance_contradiction_resolution,
    QuestionCategory.KNOWLEDGE_UPDATE.value: _guidance_knowledge_update,
    QuestionCategory.TEMPORAL_REASONING.value: _guidance_temporal_reasoning,
    QuestionCategory.PREFERENCE_FOLLOWING.value: _guidance_preference_following,
    QuestionCategory.INSTRUCTION_FOLLOWING.value: _guidance_instruction_following,
    QuestionCategory.SUMMARIZATION.value: _guidance_summarization,
    QuestionCategory.MULTI_SESSION_REASONING.value: _guidance_multi_session_reasoning,
}


def build_category_guidance(q: BEAMQuestion) -> str:
    """Category-specific generation guidance — mirrors Vectorize's `build_rag_prompt`.

    This is where ground-truth metadata leaks into the prompt. Kept faithful
    on purpose; the whole point of this variant is to reproduce Vectorize's
    numbers, not to judge whether those numbers are defensible.
    """
    builder = _CATEGORY_GUIDANCE_BUILDERS.get(q.category)
    return builder(q) if builder is not None else ""


def build_rag_prompt(q: BEAMQuestion, context: str) -> str:
    guidance = build_category_guidance(q)
    return (
        "You are a helpful assistant answering questions based on a long conversation history.\n"
        "Answer the question using ONLY information found in the retrieved context below.\n"
        "The retrieved context is ordered chronologically (earliest first).\n\n"
        f"{guidance}\n\n"
        f"Question: {q.question}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        "Answer:"
    )


async def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", required=True, help="Path to BEAM JSON file")
    parser.add_argument("--target-path", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--vector-search-limit",
        type=int,
        default=100,
        help="Number of derivative vectors to retrieve pre-rerank (default: 100)",
    )
    parser.add_argument(
        "--expand-context",
        type=int,
        default=0,
        help="Number of context segments to expand (default: 0)",
    )
    parser.add_argument(
        "--max-num-segments",
        type=int,
        default=20,
        help=(
            "Max segments returned to the LLM post-unification (default: 20). "
            "Capped low to match typical RAG budgets and Bedrock reranker client "
            "limits."
        ),
    )
    parser.add_argument(
        "--provider",
        default=PROVIDER_OPENAI,
        choices=list(PROVIDERS),
        help="LLM provider for answer generation (default: openai).",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_VECTORIZE_ANSWER_MODEL,
        help=(
            f"Answer-generation model (default: {_DEFAULT_VECTORIZE_ANSWER_MODEL}). "
            "For --provider google, use e.g. gemini-3.1-pro-preview to match "
            "Vectorize's leaderboard."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--show-timestamps",
        action="store_true",
        help=(
            "Surface ingest-time Event.timestamps on retrieved context (both "
            "reranker input and LLM context). BEAM's timeline is normally "
            "content-embedded, so we suppress our synthetic timestamps by "
            "default. Enable this to test whether even wrong-but-monotonic "
            "stamps help the answerer with event ordering."
        ),
    )
    parser.add_argument(
        "--embedder",
        default=EMBEDDER_OPENAI,
        choices=list(EMBEDDERS),
        help=(
            "Embedding backend (must match the ingest run). Default: openai. "
            "'sentence-transformer' loads a local SentenceTransformer model."
        ),
    )
    parser.add_argument(
        "--embedder-model",
        default=None,
        help=(
            "Embedder model name. Defaults: openai → text-embedding-3-small, "
            f"sentence-transformer → {DEFAULT_ST_MODEL}."
        ),
    )
    parser.add_argument(
        "--namespace",
        default=NAMESPACE,
        help=(
            f"Qdrant collection namespace (default: {NAMESPACE}). Must match "
            "the ingest run's namespace."
        ),
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help=(
            "Disable the Bedrock reranker entirely. Retrieval uses raw "
            "vector-search ranking only — matches LIGHT and Vectorize's RAG "
            "baseline (no cross-encoder rerank)."
        ),
    )
    args = parser.parse_args()
    namespace = args.namespace

    format_options = FormatOptions() if args.show_timestamps else _NO_TIMESTAMPS

    answer_model = args.model

    conversations = load_beam_dataset(args.data_path)

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
    if args.embedder == EMBEDDER_OPENAI:
        embedder = OpenAIEmbedder(
            OpenAIEmbedderParams(
                client=openai_client,
                model=args.embedder_model or "text-embedding-3-small",
                dimensions=1536,
            )
        )
    else:
        from memmachine_server.common.embedder.sentence_transformer_embedder import (
            SentenceTransformerEmbedder,
            SentenceTransformerEmbedderParams,
        )
        from sentence_transformers import SentenceTransformer

        st_model_name = args.embedder_model or DEFAULT_ST_MODEL
        st_model = SentenceTransformer(st_model_name)
        embedder = SentenceTransformerEmbedder(
            SentenceTransformerEmbedderParams(
                model_name=st_model_name,
                sentence_transformer=st_model,
            )
        )
    chat_client = make_chat_client(args.provider)

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

    async def answer(prompt: str) -> dict:
        start = time.monotonic()
        result = await chat_client.create(
            model=answer_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "response": result["content"].strip(),
            "input_tokens": result["prompt_tokens"],
            "output_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "latency": time.monotonic() - start,
        }

    async def process_question(conv: BEAMConversation, question: BEAMQuestion):
        partition_key = conv.conversation_id

        collection = await vector_store.open_collection(
            namespace=namespace, name=partition_key
        )
        segment_store_partition = await segment_store.open_or_create_partition(
            partition_key
        )

        memory = EventMemory(
            EventMemoryParams(
                vector_store_collection=collection,
                segment_store_partition=segment_store_partition,
                embedder=embedder,
                reranker=reranker,
            )
        )

        memory_start = time.monotonic()
        query_result = await memory.query(
            query=question.question,
            vector_search_limit=args.vector_search_limit,
            expand_context=args.expand_context,
            format_options=format_options,
        )
        memory_latency = time.monotonic() - memory_start

        unified = EventMemory.build_query_result_context(
            query_result, max_num_segments=args.max_num_segments
        )
        formatted_context = EventMemory.string_from_segment_context(
            unified, format_options=format_options
        )

        prompt = build_rag_prompt(question, formatted_context)
        llm_result = await answer(prompt)

        print(
            f"[{question.category}] conv={partition_key} idx={question.index} "
            f"mem={memory_latency:.2f}s llm={llm_result['latency']:.2f}s"
        )

        result = {
            "conversation_id": partition_key,
            "category": question.category,
            "question_index": question.index,
            "question": question.question,
            "gold_answer": question.answer,
            "rubric": question.rubric,
            "ordering_tested": question.ordering_tested,
            "preference_being_tested": question.preference_being_tested,
            "instruction_being_tested": question.instruction_being_tested,
            "compliance_indicators": question.compliance_indicators,
            "time_points": question.time_points,
            "calculation_required": question.calculation_required,
            "why_unanswerable": question.why_unanswerable,
            "difficulty": question.difficulty,
            "model_answer": llm_result["response"],
            "memory_latency": memory_latency,
            "llm_latency": llm_result["latency"],
            "input_tokens": llm_result["input_tokens"],
            "output_tokens": llm_result["output_tokens"],
            "formatted_context": formatted_context,
            "query_result": query_result.model_dump(mode="json"),
        }

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)
        return result

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        async_with(semaphore, process_question(conv, q))
        for conv in conversations
        for q in conv.questions
    ]
    results = await asyncio.gather(*tasks)

    by_category: dict[str, list[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    with open(args.target_path, "w") as f:
        json.dump(by_category, f, indent=4)

    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()
    await openai_client.close()
    await chat_client.close()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig()
    logging.getLogger(
        "memmachine_server.episodic_memory.event_memory.event_memory"
    ).setLevel(logging.DEBUG)
    asyncio.run(main())
