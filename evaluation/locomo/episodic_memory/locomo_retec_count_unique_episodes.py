import argparse
import json
from datetime import datetime

from memmachine.episodic_memory.declarative_memory import (
    ContentType,
    Episode,
)

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True, help="Path to the source data file")

args = parser.parse_args()

data_path = args.data_path

with open(data_path, "r") as f:
    locomo_data = json.load(f)

unique_sum = 0
num_questions = 1540
for category, question_items in locomo_data.items():
    if int(category) == 5:
        continue

    for item in question_items:
        unique_episodes = set()
        for ec_item in item["episode_contexts"]:
            episode_items = ec_item["episodes"]
            episodes = [
                Episode(
                    uid=ep_item["uid"],
                    timestamp=datetime.fromisoformat(ep_item["timestamp"]),
                    source=ep_item["source"],
                    content_type=ContentType(ep_item["content_type"]),
                    content=ep_item["content"],
                    filterable_properties=ep_item["filterable_properties"],
                    user_metadata=ep_item["user_metadata"],
                )
                for ep_item in episode_items
            ]

            unique_episodes.update(ep.uid for ep in episodes)
        unique_sum += len(unique_episodes)

print(unique_sum / num_questions)
