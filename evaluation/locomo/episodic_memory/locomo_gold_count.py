import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-path", required=True, help="Path to the source data file"
)

args = parser.parse_args()

data_path = args.data_path

MAX_NUM_EPISODES = 200

with open(data_path, "r") as f:
    locomo_data = json.load(f)

total_good = {}
for category, question_items in locomo_data.items():
    if int(category) == 5:
        continue

    for item in question_items:
        good = {i: True for i in range(1, MAX_NUM_EPISODES + 1)}
        gold_rr_ranks = item["gold_rr_ranks"]
        gold_em_ranks = item["gold_em_ranks"]

        if not gold_rr_ranks:
            for i in range(1, MAX_NUM_EPISODES + 1):
                good[i] = False

        for rank in gold_rr_ranks:
            for i in range(1, MAX_NUM_EPISODES + 1):
                if rank > i:
                    good[i] = False

        for i in range(1, MAX_NUM_EPISODES + 1):
            if good[i]:
                total_good.setdefault(i, 0)
                total_good[i] += 1

print(list(total_good.values()))
