"""Run LLM QA evaluation using pre-computed search results.

Takes the output of longmemeval_search.py, deserializes the QueryResult,
applies unification, formats the context string, and runs LLM QA.

Two answerer prompt variants are supported via ``--answer-variant``:

- ``mastra`` (default) — short prompt borrowed from Mastra's observational
  memory processor.
- ``mem0-bench`` — Mem0's elaborate 7-step ANSWER_GENERATION_PROMPT from
  https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/prompts.py.
  Mem0's run.py defaults the answerer model to ``gpt-5``.
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections.abc import Iterable
from typing import Any
from uuid import UUID

from answer_prompts import (
    ANSWER_VARIANTS,
    MEM0_BENCH_DEFAULT_ANSWER_MODEL,
    AnswerVariant,
    build_prompt,
    postprocess_answer,
)
from dotenv import load_dotenv
from longmemeval_models import get_datetime_from_timestamp
from memmachine_server.common.utils import async_with
from memmachine_server.episodic_memory.event_memory.data_types import (
    FormatOptions,
    QueryResult,
    Segment,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from openai import AsyncOpenAI

# CLDR-styled timestamps when rendering segments into the answerer prompt.
# "long" date / "short" time gives e.g. "May 7, 2026 at 7:00 PM" in en_US.
_FORMAT_OPTIONS = FormatOptions(date_style="full", time_style="short")

_COMPACT_LINE_RE = re.compile(
    r"^\[([^\]]+)\]\s+(User|Assistant):\s*(.*)$",
    re.DOTALL,
)


def compact_memories_string(text: str) -> str:
    """Dedup the date in a formatted memories string, IRC-log style.

    Lines of the form ``[<date>, <time>] <Speaker>: <utterance>`` are
    re-emitted with one ``--- <date>`` separator per unique date, and the
    time stays inline as ``[<time>] <Speaker>: <utterance>`` on every
    message. The original ``User:`` / ``Assistant:`` speaker labels,
    utterances, and upstream CLDR-formatted date and time strings are
    preserved verbatim.

    The bracket prefix is split on its last comma so that date components
    that themselves contain commas (e.g. ``Saturday, May 20, 2023``) stay
    intact. Lines that don't match the expected shape pass through
    unchanged at the end of the output.
    """
    by_date: dict[str, list[str]] = {}
    pass_through: list[str] = []
    for line in text.split("\n"):
        match = _COMPACT_LINE_RE.match(line)
        if match is None:
            pass_through.append(line)
            continue
        bracket, speaker, utterance = (
            match.group(1),
            match.group(2),
            match.group(3),
        )
        last_comma = bracket.rfind(",")
        if last_comma == -1:
            pass_through.append(line)
            continue
        date = bracket[:last_comma].strip()
        time_str = bracket[last_comma + 1 :].strip()
        by_date.setdefault(date, []).append(f"[{time_str}] {speaker}: {utterance}")
    out: list[str] = []
    for date, msgs in by_date.items():
        out.append(f"--- {date}")
        out.extend(msgs)
    out.extend(pass_through)
    return "\n".join(out)


def _omits_temperature(model: str) -> bool:
    return model.lower().startswith(("gpt-5", "o1", "o3", "o4"))


def _original_string_from_segment_context(
    segment_context: Iterable[Segment],
    *,
    format_options: FormatOptions | None = None,
) -> str:
    """Pre-gap-separator implementation of `EventMemory.string_from_segment_context`.

    Kept here for A/B comparison against the in-tree formatter, which now
    inserts ` ... ` between blocks of the same segment when an offset is
    skipped.
    """
    if format_options is None:
        format_options = FormatOptions()

    context_string = ""
    last_segment: Segment | None = None
    accumulated_text = ""
    first = True

    for segment in segment_context:
        is_continuation = (
            last_segment is not None
            and segment.event_uuid == last_segment.event_uuid
            and segment.index == last_segment.index
        )

        if not is_continuation:
            if not first:
                context_string += (
                    json.dumps(accumulated_text, ensure_ascii=False) + "\n"
                )
            first = False
            accumulated_text = ""
            context_string += EventMemory._segment_header(segment, format_options)

        text = EventMemory._extract_text(segment.block)
        if text is not None:
            accumulated_text += text
        elif not is_continuation:
            context_string += f"[{segment.block.block_type}]\n"

        last_segment = segment

    if not first:
        context_string += json.dumps(accumulated_text, ensure_ascii=False) + "\n"

    return context_string.strip()


def _original_string_from_segment_contexts(
    segment_contexts: Iterable[Iterable[Segment]],
    *,
    format_options: FormatOptions | None = None,
) -> str:
    """Original-formatter equivalent of `EventMemory.string_from_segment_contexts`."""
    segment_contexts = [list(context) for context in segment_contexts]

    segments_by_uuid: dict[UUID, Segment] = {}
    component_parent: dict[UUID, UUID] = {}

    def find(uuid: UUID) -> UUID:
        component_parent.setdefault(uuid, uuid)
        root = uuid
        while component_parent[root] != root:
            root = component_parent[root]
        while component_parent[uuid] != root:
            parent = component_parent[uuid]
            component_parent[uuid] = root
            uuid = parent
        return root

    for context in segment_contexts:
        first_segment_root: UUID | None = None
        for segment in context:
            segments_by_uuid.setdefault(segment.uuid, segment)
            if first_segment_root is None:
                first_segment_root = find(segment.uuid)
            else:
                segment_root = find(segment.uuid)
                component_parent[segment_root] = first_segment_root

    segments_by_root: dict[UUID, list[Segment]] = {}
    for segment_uuid, segment in segments_by_uuid.items():
        segments_by_root.setdefault(find(segment_uuid), []).append(segment)

    def segment_key(segment: Segment) -> tuple:
        return (
            segment.timestamp,
            segment.event_uuid,
            segment.index,
            segment.offset,
        )

    components = list(segments_by_root.values())
    for component in components:
        component.sort(key=segment_key)
    components.sort(key=lambda segments: segment_key(segments[0]))

    return "\n\n".join(
        _original_string_from_segment_context(segments, format_options=format_options)
        for segments in components
    )


def _original_string_from_query_result(
    query_result: QueryResult,
    *,
    max_num_segments: int | None = None,
    format_options: FormatOptions | None = None,
) -> str:
    """Original-formatter equivalent of `EventMemory.string_from_query_result`."""
    contexts: list[list[Segment]] = [
        list(scored_context.segments)
        for scored_context in query_result.scored_segment_contexts
    ]

    if max_num_segments is not None:
        included = {
            segment.uuid
            for segment in EventMemory.build_query_result_context(
                query_result, max_num_segments
            )
        }
        contexts = [
            [segment for segment in context if segment.uuid in included]
            for context in contexts
        ]

    return _original_string_from_segment_contexts(
        contexts, format_options=format_options
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-path", required=True, help="Path to longmemeval_search output"
    )
    parser.add_argument("--target-path", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--max-num-segments",
        type=int,
        default=20,
        help="Max segments after unification",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Keep only the seed segment from each ranked context before unification",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=-1.0,
        help="Drop scored segment contexts with score below this threshold",
    )
    parser.add_argument(
        "--separate-contexts",
        action="store_true",
        help=(
            "Format using string_from_query_result, which keeps disconnected "
            "ranked contexts separated instead of unifying them into one list"
        ),
    )
    parser.add_argument(
        "--compact-memories",
        action="store_true",
        help=(
            "Compact the formatted context by emitting one `--- <date>` "
            "separator per unique date and keeping the time inline as "
            "`[<time>] <Speaker>: <utterance>` on every message."
        ),
    )
    parser.add_argument(
        "--original-format",
        action="store_true",
        help=(
            "Use the pre-gap-separator formatter (kept locally in this "
            "script for A/B comparison against the new in-tree behavior, "
            "which inserts ` ... ` between blocks of the same segment "
            "when an offset is skipped)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Max concurrent QA calls",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "OpenAI chat model for QA. Defaults to 'gpt-5-mini' for "
            "--answer-variant=mastra and "
            f"'{MEM0_BENCH_DEFAULT_ANSWER_MODEL}' for --answer-variant=mem0-bench "
            "(matching mem0ai/memory-benchmarks)."
        ),
    )
    parser.add_argument(
        "--answer-variant",
        default="mastra",
        choices=list(ANSWER_VARIANTS),
        help=(
            "Answerer prompt variant. 'mastra' is the in-house default; "
            "'mem0-bench' is Mem0's 7-step prompt from memory-benchmarks "
            "(uses <mem_thinking> chain-of-thought + ANSWER: post-processing)."
        ),
    )
    args = parser.parse_args()

    if args.model is None:
        args.model = (
            MEM0_BENCH_DEFAULT_ANSWER_MODEL
            if args.answer_variant == "mem0-bench"
            else "gpt-5-mini"
        )

    with open(args.search_path) as f:
        search_results = json.load(f)

    print(
        f"Answering {len(search_results)} questions with "
        f"variant={args.answer_variant!r} model={args.model!r}..."
    )

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    answer_variant: AnswerVariant = args.answer_variant

    async def qa_eval(prompt: str):
        start_time = time.monotonic()
        kwargs: dict[str, Any] = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not _omits_temperature(args.model):
            kwargs["temperature"] = 0.0
        response = await openai_client.chat.completions.create(**kwargs)
        latency = time.monotonic() - start_time
        message = response.choices[0].message.content or ""
        usage = response.usage
        return {
            "raw": message,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "latency": latency,
        }

    async def process_item(item: dict):
        query_result = QueryResult.model_validate(item["query_result"])
        query_result.scored_segment_contexts = [
            scored_context
            for scored_context in query_result.scored_segment_contexts
            if scored_context.score >= args.score_threshold
        ]
        if args.seed_only:
            for scored_context in query_result.scored_segment_contexts:
                scored_context.segments = [
                    segment
                    for segment in scored_context.segments
                    if segment.uuid == scored_context.seed_segment_uuid
                ]
        unified = EventMemory.build_query_result_context(
            query_result, max_num_segments=args.max_num_segments
        )
        if args.separate_contexts:
            if args.original_format:
                formatted_context = _original_string_from_query_result(
                    query_result,
                    max_num_segments=args.max_num_segments,
                    format_options=_FORMAT_OPTIONS,
                )
            else:
                formatted_context = EventMemory.string_from_query_result(
                    query_result,
                    max_num_segments=args.max_num_segments,
                    format_options=_FORMAT_OPTIONS,
                )
        else:
            if args.original_format:
                formatted_context = _original_string_from_segment_context(
                    unified, format_options=_FORMAT_OPTIONS
                )
            else:
                formatted_context = EventMemory.string_from_segment_context(
                    unified, format_options=_FORMAT_OPTIONS
                )
        if args.compact_memories:
            formatted_context = compact_memories_string(formatted_context)

        question_date_human = get_datetime_from_timestamp(
            item["question_date"]
        ).strftime("%A, %B %d, %Y at %I:%M %p")

        prompt = build_prompt(
            variant=answer_variant,
            question=item["question"],
            question_date=question_date_human,
            memories_string=formatted_context,
            segments=unified,
        )

        total_start = time.monotonic()
        response = await qa_eval(prompt)
        total_latency = time.monotonic() - total_start
        cleaned_response = postprocess_answer(response["raw"], variant=answer_variant)

        print(
            f"Question ID: {item['question_id']}\n"
            f"Question: {item['question']}\n"
            f"Question Date: {item['question_date']}\n"
            f"Question Type: {item['question_type']}\n"
            f"Answer: {item['answer']}\n"
            f"Response: {cleaned_response}\n"
            f"Memory retrieval time: {item.get('memory_latency', 0):.2f} seconds\n"
            f"LLM response time: {response['latency']:.2f} seconds\n"
            f"Total processing time: {total_latency:.2f} seconds\n"
            f"MEMORIES_START\n{formatted_context}MEMORIES_END\n"
        )

        return {
            "question_id": item["question_id"],
            "question_date": item["question_date"],
            "question": item["question"],
            "answer": item["answer"],
            "response": cleaned_response,
            "raw_response": response["raw"],
            "question_type": item["question_type"],
            "abstention": item["abstention"],
            "total_latency": total_latency,
            "memory_latency": item.get("memory_latency"),
            "llm_latency": response["latency"],
            "episodes_text": formatted_context,
        }

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [async_with(semaphore, process_item(item)) for item in search_results]
    results = await asyncio.gather(*tasks)

    with open(args.target_path, "w") as f:
        json.dump(results, f, indent=4)

    await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
