import argparse
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memmachine.episodic_memory.declarative_memory import (
    ContentType,
    DeclarativeMemory,
    Episode,
)
from longmemeval_models import (
    get_datetime_from_timestamp,
    load_longmemeval_dataset,
)

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True, help="Path to the source data file")
parser.add_argument("--search-path", required=True, help="Path to the search data file")
parser.add_argument(
    "--output-path",
    default="retec_recall.png",
    help="Path to write the recall graph PNG",
)

args = parser.parse_args()

data_path = args.data_path
search_path = args.search_path
output_path = args.output_path

MAX_NUM_EPISODES = 200

all_questions = load_longmemeval_dataset(data_path)

with open(search_path, "r") as f:
    longmemeval_data = json.load(f)

question_map = {question.question_id: question for question in all_questions}

# Per-category recall curves, normalized by the category's total gold turns.
overall_recall_at_num = dict.fromkeys(range(1, MAX_NUM_EPISODES + 1), 0)
overall_total_golds = 0
category_curves = {}

for category, question_items in longmemeval_data.items():
    total_recall_at_num = dict.fromkeys(range(1, MAX_NUM_EPISODES + 1), 0)
    total_golds = 0
    for item in question_items:
        question_id = item["question_id"]
        lme_question_item = question_map[question_id]
        total_golds += len(item["answer_turn_indices"])
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

            for episode in unified_episode_context:
                session_id = episode.user_metadata["longmemeval_session_id"]
                timestamp = episode.timestamp
                session_date = lme_question_item.get_session_date(session_id)
                session_datetime = get_datetime_from_timestamp(session_date)
                difference = timestamp - session_datetime
                difference_seconds = int(difference.total_seconds())
                if f"{session_id}:{difference_seconds}" in item["answer_turn_indices"]:
                    total_recall_at_num[i] += 1

    for i in range(1, MAX_NUM_EPISODES + 1):
        overall_recall_at_num[i] += total_recall_at_num[i]
    overall_total_golds += total_golds

    if total_golds == 0:
        print(f"Skipping category {category!r}: 0 gold turns")
        continue

    category_curves[category] = (total_recall_at_num, total_golds)
    print(
        f"{category}: total golds = {total_golds}, "
        f"recall@{MAX_NUM_EPISODES} = "
        f"{total_recall_at_num[MAX_NUM_EPISODES] / total_golds:.4f}"
    )

# Plot: x = episodes per anchor (N), y = recall fraction, one line per category.
xs = list(range(0, MAX_NUM_EPISODES + 1))

plt.figure(figsize=(10, 6))

for category, (total_recall_at_num, total_golds) in category_curves.items():
    ys = [0.0] + [
        total_recall_at_num[i] / total_golds
        for i in range(1, MAX_NUM_EPISODES + 1)
    ]
    plt.plot(xs, ys, label=f"{category} (golds={total_golds})")

if overall_total_golds > 0:
    overall_ys = [0.0] + [
        overall_recall_at_num[i] / overall_total_golds
        for i in range(1, MAX_NUM_EPISODES + 1)
    ]
    plt.plot(
        xs,
        overall_ys,
        label=f"overall (golds={overall_total_golds})",
        color="black",
        linestyle="--",
        linewidth=2,
    )

plt.xlabel("Max episodes in unified context (N)")
plt.ylabel("Recall (fraction of gold turns)")
plt.title("Recall vs. unified context size cap, normalized by total gold turns")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"Saved graph to {output_path}")
