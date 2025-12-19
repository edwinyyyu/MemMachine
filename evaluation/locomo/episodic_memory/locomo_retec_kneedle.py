import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True, help="Path to the source data file")

args = parser.parse_args()

data_path = args.data_path

MAX_NUM_EPISODES = 200

with open(data_path, "r") as f:
    locomo_data = json.load(f)

for category, question_items in locomo_data.items():
    if int(category) != 4:
        continue

    for item in question_items:
        scores = [ec_item["score"] for ec_item in item["episode_contexts"]]

        print(scores)
