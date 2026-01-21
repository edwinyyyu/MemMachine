import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--data-path", required=True)
parser.add_argument("--search-path", required=True)
parser.add_argument("--eval-path", required=True)
parser.add_argument("--include-abs", action="store_true")

args = parser.parse_args()
data_path = args.data_path
search_path = args.search_path
eval_path = args.eval_path
include_abs = args.include_abs

with open(data_path, "r") as f:
    data = json.load(f)

with open(search_path, "r") as f:
    search = json.load(f)

with open(eval_path, "r") as f:
    evals = json.load(f)

search_results = {}
for item in search:
    search_result = {}
    search_result["response"] = item["response"]
    search_result["episodes"] = item["episodes_text"]
    search_results[item["question_id"]] = search_result

for item in data:
    question_id = item["question_id"]
    if not include_abs and question_id.endswith("_abs"):
        continue

    question_date = item["question_date"]
    question = item["question"]
    question_type = item["question_type"]
    answer = item["answer"]

    for question_type, details in evals.items():
        if "question_llm_map" not in details:
            continue

        question_llm_map = details["question_llm_map"]

        evidence = []
        for answer_session_id in item["answer_session_ids"]:
            answer_session_index = item["haystack_session_ids"].index(answer_session_id)
            answer_session_date = item["haystack_dates"][answer_session_index]
            answer_session = item["haystack_sessions"][answer_session_index]

            for turn in answer_session:
                if turn.get("has_answer", False):
                    evidence.append(
                        f"[{answer_session_date}] {turn['role']}: {json.dumps(turn['content'])}"
                    )

        if question_llm_map.get(question_id, 1) == 0:
            print(f"QUESTION ID: {question_id}")
            print(f"QUESTION: {question}")
            print(f"QUESTION DATE: {question_date}")
            print(f"QUESTION TYPE: {question_type}")
            print(f"ANSWER: {answer}")
            print(f"RESPONSE: {search_results.get(question_id, {}).get('response')}")
            print(f"EPISODES:\n{search_results.get(question_id, {}).get('episodes')}")
            print(f"EVIDENCE:\n{'\n'.join(evidence)}")
            print()
