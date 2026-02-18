import argparse
import asyncio
import json
import os
from collections import defaultdict
from time import time

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

JUDGE_PROMPT = """
You are an expert evaluator.
Given a question, two correct reference answers, and a model-generated response, determine if the response is correct.
Please answer "yes" if the response is a correct answer to the question based on the reference answers. Otherwise, answer "no".
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer "yes". If the response only contains a subset of the information required by the answer, answer "no".

Question: {question}
Correct Answer 1: {correct_answer1}
Correct Answer 2: {correct_answer2}
Model Response: {response}

Is the model response correct? Answer "yes" or "no" only.
"""

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def get_llm_evaluation(prompt, model="gpt-4.1-mini"):
    """
    Get LLM evaluation for a given prompt
    """
    try:
        # Replace with your preferred LLM API call
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return "error"

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

            prompt = JUDGE_PROMPT.format(
                question=quesiton,
                correct_answer1=answer1,
                correct_answer2=answer2,
                response=response,
            )

            # Measure latency
            start_time = time()
            llm_response = await get_llm_evaluation(prompt)
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
