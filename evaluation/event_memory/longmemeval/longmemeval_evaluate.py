"""LongMemEval LLM judge — score each QA result on the official rubric.

Two judge prompt variants are supported via ``--judge-variant``:

- ``longmemeval-paper`` (default) — per-question-type templates from the
  original LongMemEval paper.
- ``mem0-bench`` — Mem0's unified judge prompt from
  https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/prompts.py.
  Default judge model for this variant matches Mem0's run.py: ``gpt-5``.
"""

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from time import time
from typing import Any

import numpy as np
from dotenv import load_dotenv
from judge_prompts import (
    JUDGE_VARIANTS,
    MEM0_BENCH_DEFAULT_JUDGE_MODEL,
    JudgeVariant,
    build_prompt,
    parse_yes_no,
)
from memmachine_server.common.utils import async_with
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


@dataclass
class _TaskMetrics:
    llm_scores: list[int] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    count: int = 0
    question_llm_map: dict[str, int] = field(default_factory=dict)


def _omits_temperature(model: str) -> bool:
    """gpt-5 / o-series only accept the default temperature; omit the param."""
    return model.lower().startswith(("gpt-5", "o1", "o3", "o4"))


async def get_llm_evaluation(client: AsyncOpenAI, prompt: str, model: str) -> str:
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not _omits_temperature(model):
            kwargs["temperature"] = 0.0
        response = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message.content or ""
        return message
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return ""


def compute_iqr(data):
    q75, q25 = np.percentile(data, [75, 25])
    return q75 - q25


async def evaluate_responses(
    client: AsyncOpenAI,
    responses: list[dict],
    llm_model: str,
    exclude_abstention: bool,
    judge_variant: JudgeVariant,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, _TaskMetrics] = {}
    binary_predictions: list[int] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def _judge_one(item: dict) -> tuple[dict, str, float, bool] | None:
        abstention = item["abstention"]
        if exclude_abstention and abstention:
            return None
        prompt = build_prompt(
            variant=judge_variant,
            task=item["question_type"],
            question=item["question"],
            answer=item["answer"],
            response=item.get("response", ""),
            abstention=abstention,
        )
        start_time = time()
        llm_response = await get_llm_evaluation(client, prompt, llm_model)
        latency = time() - start_time
        is_yes = parse_yes_no(llm_response, variant=judge_variant)
        return item, llm_response, latency, is_yes

    pending = [async_with(semaphore, _judge_one(item)) for item in responses]
    for coro in tqdm_asyncio.as_completed(
        pending, total=len(pending), desc="Evaluating", unit="q"
    ):
        outcome = await coro
        if outcome is None:
            continue
        item, llm_response, latency, is_yes = outcome
        print(llm_response)
        task = item["question_type"]
        qid = item["question_id"]
        llm_score = 1 if is_yes else 0
        metrics = results.setdefault(task, _TaskMetrics())
        metrics.llm_scores.append(llm_score)
        metrics.latencies.append(latency)
        metrics.count += 1
        metrics.question_llm_map[qid] = llm_score
        binary_predictions.append(llm_score)

    final_results: dict[str, dict[str, Any]] = {}
    for task, metrics in results.items():
        if metrics.count > 0:
            avg_llm = float(np.mean(metrics.llm_scores))
            avg_latency = float(np.mean(metrics.latencies))

            token_avg = float(np.mean(metrics.tokens)) if metrics.tokens else None
            token_iqr = float(compute_iqr(metrics.tokens)) if metrics.tokens else None

            final_results[task] = {
                "llm_score": avg_llm,
                "avg_latency": avg_latency,
                "token_avg": token_avg,
                "token_iqr": token_iqr,
                "count": metrics.count,
                "llm_scores_detail": list(metrics.llm_scores),
                "latencies_detail": list(metrics.latencies),
                "tokens_detail": list(metrics.tokens),
                "question_llm_map": dict(metrics.question_llm_map),
            }

    all_tokens: list[int] = [
        token for task_metrics in results.values() for token in task_metrics.tokens
    ]
    overall_llm = float(np.mean(binary_predictions)) if binary_predictions else 0.0
    all_latencies = [
        lat for task_metrics in results.values() for lat in task_metrics.latencies
    ]
    overall_latency = float(np.mean(all_latencies)) if all_latencies else 0.0
    overall_token_avg = float(np.mean(all_tokens)) if all_tokens else None
    overall_token_iqr = float(compute_iqr(all_tokens)) if all_tokens else None

    final_results["overall"] = {
        "llm_score": overall_llm,
        "avg_latency": overall_latency,
        "token_avg": overall_token_avg,
        "token_iqr": overall_token_iqr,
        "total_count": sum(metrics.count for metrics in results.values()),
    }

    return final_results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "OpenAI judge model. Defaults to 'gpt-4o' for "
            "--judge-variant=longmemeval-paper, and "
            f"'{MEM0_BENCH_DEFAULT_JUDGE_MODEL}' for --judge-variant=mem0-bench "
            "(matching mem0ai/memory-benchmarks)."
        ),
    )
    parser.add_argument(
        "--judge-variant",
        default="longmemeval-paper",
        choices=list(JUDGE_VARIANTS),
        help=(
            "Judge prompt variant. 'longmemeval-paper' uses the per-task "
            "templates from the original LongMemEval repo; 'mem0-bench' uses "
            "Mem0's unified semantic-equivalence prompt."
        ),
    )
    parser.add_argument(
        "--exclude-abstention",
        action="store_true",
        help="Exclude abstention questions",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=30,
        help="Max concurrent judge calls",
    )
    args = parser.parse_args()

    if args.llm_model is None:
        args.llm_model = (
            MEM0_BENCH_DEFAULT_JUDGE_MODEL
            if args.judge_variant == "mem0-bench"
            else "gpt-4o"
        )

    with open(args.data_path, encoding="utf-8") as f:
        responses = json.load(f)

    print(
        f"Evaluating {len(responses)} questions with "
        f"variant={args.judge_variant!r} model={args.llm_model!r}..."
    )

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results = await evaluate_responses(
        client,
        responses,
        llm_model=args.llm_model,
        exclude_abstention=args.exclude_abstention,
        judge_variant=args.judge_variant,
        concurrency=args.concurrency,
    )
    with open(args.target_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    await client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
