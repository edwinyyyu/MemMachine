import argparse
import json
from datetime import datetime

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

args = parser.parse_args()

data_path = args.data_path
search_path = args.search_path

MAX_NUM_EPISODES = 200

all_questions = load_longmemeval_dataset(data_path)

with open(search_path, "r") as f:
    longmemeval_data = json.load(f)

question_map = {question.question_id: question for question in all_questions}

for category, question_items in longmemeval_data.items():
    total_recall_at_num = dict.fromkeys(range(1, MAX_NUM_EPISODES + 1), 0)
    for item in question_items:
        question_id = item["question_id"]
        lme_question_item = question_map[question_id]
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

    print([0] + list(total_recall_at_num.values()))
