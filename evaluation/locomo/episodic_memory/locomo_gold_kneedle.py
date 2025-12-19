import argparse
import json

from memmachine.common.utils import (
    kneedle_cutoff,
    kneedle_cutoff_fit,
)

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
        rr_scores = item["rr_scores"]
        print()
        print(kneedle_cutoff(rr_scores))
        print(kneedle_cutoff_fit(rr_scores))
