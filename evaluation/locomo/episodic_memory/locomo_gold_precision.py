import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True, help="Path to the source data file")

args = parser.parse_args()

data_path = args.data_path

MAX_NUM_EPISODES = 200

with open(data_path, "r") as f:
    locomo_data = json.load(f)

total_recall_at_num = {i: 0 for i in range(1, MAX_NUM_EPISODES + 1)}
for category, question_items in locomo_data.items():
    if int(category) == 5:
        continue

    for item in question_items:
        gold_rr_ranks = item["gold_rr_ranks"]
        evidence = item["evidence"]

        len(gold_rr_ranks)

        for rank in gold_rr_ranks:
            for i in range(1, MAX_NUM_EPISODES + 1):
                if rank < i:
                    total_recall_at_num[i] += 1

total_recalled = list(total_recall_at_num.values())
total_retrieved = [1540 * i for i in range(1, MAX_NUM_EPISODES + 1)]
precisions = [recalled / retrieved if retrieved > 0 else 0.0 for recalled, retrieved in zip(total_recalled, total_retrieved)]
print([0.0] + precisions)
