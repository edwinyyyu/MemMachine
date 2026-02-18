import argparse
import asyncio
import json
import os
from collections import defaultdict
from time import time

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rouge_score import rouge_scorer
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def get_rouge_eval(reference, candidate):


async def evaluate_documents(documents):
    results = defaultdict(
        lambda: {
            "llm_scores": [],
            "latencies": [],
            "count": 0,
        }
    )

    binary_labels = []
    binary_predictions = []

    for document_id, question_items in documents.items():
        for question_item in tqdm_asyncio(question_items, desc=f"Evaluating Doc {document_id}"):
            quesiton = question_item["question"]
            answer1 = question_item["answer1"]
            answer2 = question_item["answer2"]
            response = question_item["response"]

            start_time = time()
            llm_response = await get_rouge_eval(prompt)
            print(llm_response)
            latency = time() - start_time

            llm_score = 1 if "yes" in llm_response.lower() else 0

            results[document_id]["llm_scores"].append(llm_score)
            results[document_id]["latencies"].append(latency)
            results[document_id]["count"] += 1

            binary_labels.append(1)  # adjust if you have true labels
            binary_predictions.append(llm_score)

    # Aggregate metrics
    final_results = {}
    for document_id, metrics in results.items():
        if metrics["count"] > 0:
            avg_llm = np.mean(metrics["llm_scores"])
            avg_latency = np.mean(metrics["latencies"])

            final_results[document_id] = {
                "llm_score": float(avg_llm),
                "avg_latency": float(avg_latency),
                "count": metrics["count"],
                "llm_scores_detail": metrics["llm_scores"],
                "latencies_detail": metrics["latencies"],
            }

    # Overall metrics
    overall_llm = np.mean(binary_predictions)
    overall_latency = np.mean(
        [lat for task_metrics in results.values() for lat in task_metrics["latencies"]]
    )

    final_results["overall"] = {
        "llm_score": float(overall_llm),
        "avg_latency": float(overall_latency),
        "total_count": sum(metrics["count"] for metrics in results.values()),
    }

    return final_results

async def main():
    """
    Main function to run the evaluation
    """
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

    # Load your dataset
    with open(data_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Evaluating {len(documents)} documents...")

    results = await evaluate_documents(
        documents
    )
    # Save results
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())
