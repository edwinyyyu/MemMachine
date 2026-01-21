import argparse
import asyncio
import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv

questions = {}

def process_lme(question_item):
    question = question_item["question"]
    question_type = question_item["question_type"]
    match question_type:
        case "single-session-user" | "single-session-assistant" | "single-session-preference":
            question_type = "quality"
        case "multi-session" | "knowledge-update" | "temporal-reasoning":
            question_type = "quantity"
        case _:
            raise ValueError(f"Unknown question type: {question_type}")

    questions.setdefault(question_type, [])
    questions[question_type].append(question)


def process_lcm(question_item):
    if "conversation" not in question_item:
        return

    qa_list = question_item["qa"]
    for qa in qa_list:
        match qa["category"]:
            case 1 | 3:
                question_type = "quantity"
            case 2 | 4:
                question_type = "quality"

        question = qa["question"]
        questions.setdefault(question_type, [])
        questions[question_type].append(question)


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lme-path", required=True, help="Path to the lme data file")
    parser.add_argument("--lcm-path", required=True, help="Path to the lcm data file")

    args = parser.parse_args()

    lme_path = args.lme_path
    lcm_path = args.lcm_path

    with open(lme_path, "r") as f:
        lme_json = json.load(f)
    with open(lcm_path, "r") as f:
        lcm_json = json.load(f)

    result = [process_lme(question_item) for question_item in lme_json]
    result = [process_lcm(question_item) for question_item in lcm_json]

    rows = []
    for question_type, typed_questions in questions.items():
        label = 0 if question_type == "quantity" else 1
        for question in typed_questions:
            rows.append({"text": question, "label": label})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
    df.to_csv("router_data.csv", index=False)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
