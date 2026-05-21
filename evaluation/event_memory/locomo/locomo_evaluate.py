"""LoCoMo LLM judge — score each QA result against the gold answer.

Mirrors the reference Mem0/MemMachine evaluator's UX: a per-category tqdm
progress bar, running per-category accuracy printed after each judgment, and
intermediate results written to disk after every category completes so the
run is resumable on interruption.
"""

import argparse
import asyncio
import json
import os
import threading
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
from llm_judge import JUDGE_VARIANTS, JudgeVariant, evaluate_llm_judge
from memmachine_server.common.utils import async_with
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


def _print_running_accuracy(results: dict[str, list[dict]]) -> None:
    print("All categories accuracy:")
    for category in sorted(results):
        scores = [s["llm_score"] for s in results[category]]
        if scores:
            print(
                f"  Category {category}: {np.mean(scores):.4f} "
                f"({sum(scores)}/{len(scores)})"
            )
    print("------------------------------------------")


async def process_category(
    client: AsyncOpenAI,
    judge_model: str,
    judge_variant: JudgeVariant,
    category: str,
    items: list[dict],
    semaphore: asyncio.Semaphore,
    results: dict[str, list[dict]],
    results_lock: threading.Lock,
    target_path: str,
) -> None:
    try:
        category_id: int | None = int(category)
    except (TypeError, ValueError):
        category_id = None

    async def score(item: dict) -> dict:
        question = str(item["question"])
        gold = str(item["locomo_answer"])
        response = str(item["model_answer"])
        llm_score = await evaluate_llm_judge(
            client,
            question,
            gold,
            response,
            model=judge_model,
            variant=judge_variant,
            category=category_id,
        )
        return {
            "question": question,
            "answer": gold,
            "response": response,
            "category": category,
            "llm_score": llm_score,
        }

    pending = [async_with(semaphore, score(it)) for it in items]
    progress = tqdm(
        asyncio.as_completed(pending),
        total=len(pending),
        desc=f"Processing {category} sample",
    )

    scored: list[dict] = []
    async for coro in progress:
        result = await coro
        scored.append(result)
        with results_lock:
            results[category].append(result)
            with open(target_path, "w") as f:
                json.dump(results, f, indent=4)
            _print_running_accuracy(results)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", required=True, help="Path to locomo_search.py output"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to evaluation output JSON"
    )
    parser.add_argument("--judge-model", default="gpt-5", help="OpenAI judge model")
    parser.add_argument(
        "--judge-variant",
        default="mem0-classic",
        choices=list(JUDGE_VARIANTS),
        help=(
            "Judge prompt variant. 'mem0-classic' is the original Mem0 LoCoMo "
            "evaluator prompt; 'mem0-bench' is the system+user prompt from the "
            "newer mem0ai/memory-benchmarks repo (with category-3 semicolon trim)."
        ),
    )
    parser.add_argument(
        "--concurrency", type=int, default=30, help="Max concurrent judge calls"
    )
    parser.add_argument(
        "--skip-category-5",
        action="store_true",
        help="Skip category 5 (matches the original Mem0/LoCoMo evaluator)",
    )
    args = parser.parse_args()

    with open(args.data_path) as f:
        data = json.load(f)

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(args.concurrency)

    results: dict[str, list[dict]] = defaultdict(list)
    results_lock = threading.Lock()

    tasks = []
    for category, items in data.items():
        if args.skip_category_5 and str(category) == "5":
            continue
        tasks.append(
            process_category(
                client,
                args.judge_model,
                args.judge_variant,
                category,
                items,
                semaphore,
                results,
                results_lock,
                args.target_path,
            )
        )
    await asyncio.gather(*tasks)

    with open(args.target_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {args.target_path}")
    print("\n=== LoCoMo LLM-judge accuracy ===")
    for category in sorted(results):
        scores = [s["llm_score"] for s in results[category]]
        if scores:
            print(
                f"  Category {category}: {np.mean(scores):.4f} "
                f"({sum(scores)}/{len(scores)})"
            )

    await client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
