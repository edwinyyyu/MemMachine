import argparse
import json
from datetime import datetime

from memmachine.episodic_memory.declarative_memory import (
    ContentType,
    DeclarativeMemory,
    Episode,
)

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True, help="Path to the source data file")

args = parser.parse_args()

data_path = args.data_path

MAX_NUM_EPISODES = 200

with open(data_path, "r") as f:
    locomo_data = json.load(f)

for category, question_items in locomo_data.items():
    total_recall_at_num = dict.fromkeys(range(1, MAX_NUM_EPISODES + 1), 0)
    if int(category) == 5:
        continue

    for item in question_items:
        episode_contexts = []
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

            nuclear_episode = None
            for episode in episodes:
                if episode.uid == ec_item["nuclear_episode"]:
                    nuclear_episode = episode
                    break

            assert nuclear_episode is not None

            episode_contexts.append(
                (
                    nuclear_episode,
                    episodes,
                )
            )

        for i in range(1, MAX_NUM_EPISODES + 1):
            unified_episode_context = (
                DeclarativeMemory._unify_anchored_episode_contexts(episode_contexts, i)
            )
            episode_context_dia_ids = [
                episode.user_metadata.get("dia_id")
                for episode in unified_episode_context
            ]

            for episode_context_dia_id in episode_context_dia_ids:
                if episode_context_dia_id is None:
                    continue
                elif episode_context_dia_id in item["evidence"]:
                    total_recall_at_num[i] += 1

    print([0] + list(total_recall_at_num.values()))
