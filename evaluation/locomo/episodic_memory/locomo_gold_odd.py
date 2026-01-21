import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True, help="Path to the source data file")

args = parser.parse_args()

data_path = args.data_path

with open(data_path, "r") as f:
    locomo_data = json.load(f)

odd_results = {}
for category, question_items in locomo_data.items():
    odd_results[category] = []
    for item in question_items:
        gold_em_ranks = item["gold_em_ranks"]
        gold_rr_ranks = item["gold_rr_ranks"]

        for rank_info in zip(gold_em_ranks, gold_rr_ranks):
            if abs(rank_info[0] - rank_info[1]) >= 20:
                odd_results[category].append(item)

with open(data_path.replace(".json", "_odd.json"), "w") as f:
    json.dump(odd_results, f, indent=4)
