"""Prepare a small subset of BEAM 100K for associative recall experiments.

Loads 2-3 conversations, extracts turn-level segments, embeds them with
text-embedding-3-small, and saves everything to an .npz file for fast reload.
Also extracts probing questions with source_chat_ids for evaluation.
"""

import ast
import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "beam" / "100k.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
NUM_CONVERSATIONS = 3
BATCH_SIZE = 100

TARGET_CATEGORIES = {
    "event_ordering",
    "instruction_following",
    "summarization",
    "multi_session_reasoning",
    "information_extraction",
    "temporal_reasoning",
    "contradiction_resolution",
    "knowledge_update",
    "preference_following",
}


def flatten_source_chat_ids(raw: object) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        result = []
        for x in raw:
            if isinstance(x, list):
                result.extend(int(i) for i in x)
            elif isinstance(x, (int, float)):
                result.append(int(x))
        return result
    if isinstance(raw, dict):
        result = []
        for v in raw.values():
            result.extend(flatten_source_chat_ids(v))
        return result
    return []


def load_conversations(path: Path, num_convs: int) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data[:num_convs]


def extract_segments(conversation: dict) -> list[dict]:
    segments = []
    conv_id = str(conversation["conversation_id"])
    for session in conversation["chat"]:
        for turn in session:
            turn_id = turn["id"]
            role = turn["role"]
            content = turn["content"]
            if "->" in content:
                content = content.split("->")[0].strip()
            if not content.strip():
                continue
            segments.append(
                {
                    "conversation_id": conv_id,
                    "turn_id": turn_id,
                    "role": role,
                    "text": content,
                }
            )
    return segments


def extract_questions(conversation: dict) -> list[dict]:
    conv_id = str(conversation["conversation_id"])
    pq_raw = conversation.get("probing_questions", "{}")
    if isinstance(pq_raw, str):
        try:
            pq = json.loads(pq_raw)
        except Exception:
            pq = ast.literal_eval(pq_raw)
    else:
        pq = pq_raw

    questions = []
    for category, items in pq.items():
        if category == "abstention" or category not in TARGET_CATEGORIES:
            continue
        for i, q in enumerate(items):
            source_ids = flatten_source_chat_ids(q.get("source_chat_ids"))
            if not source_ids:
                continue
            questions.append(
                {
                    "conversation_id": conv_id,
                    "category": category,
                    "question_index": i,
                    "question": q["question"],
                    "source_chat_ids": source_ids,
                    "ideal_response": q.get("ideal_response", ""),
                }
            )
    return questions


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    all_embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        print(f"  Embedding batch {start // BATCH_SIZE + 1} ({len(batch)} texts)...")
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        time.sleep(0.1)
    return np.array(all_embeddings, dtype=np.float32)


def main() -> None:
    client = OpenAI()

    print(f"Loading {NUM_CONVERSATIONS} conversations from {DATA_PATH}...")
    conversations = load_conversations(DATA_PATH, NUM_CONVERSATIONS)

    all_segments: list[dict] = []
    all_questions: list[dict] = []

    for conv in conversations:
        segments = extract_segments(conv)
        questions = extract_questions(conv)
        all_segments.extend(segments)
        all_questions.extend(questions)
        print(
            f"  Conv {conv['conversation_id']}: "
            f"{len(segments)} segments, {len(questions)} questions"
        )

    print(f"\nTotal: {len(all_segments)} segments, {len(all_questions)} questions")

    texts = [s["text"] for s in all_segments]
    print(f"\nEmbedding {len(texts)} segments...")
    embeddings = embed_texts(client, texts)
    print(f"Embeddings shape: {embeddings.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    segment_data = {
        "conversation_ids": [s["conversation_id"] for s in all_segments],
        "turn_ids": [s["turn_id"] for s in all_segments],
        "roles": [s["role"] for s in all_segments],
        "texts": [s["text"] for s in all_segments],
    }

    output_path = OUTPUT_DIR / "segments.npz"
    np.savez(
        output_path,
        embeddings=embeddings,
        conversation_ids=np.array(segment_data["conversation_ids"]),
        turn_ids=np.array(segment_data["turn_ids"], dtype=np.int32),
        roles=np.array(segment_data["roles"]),
        texts=np.array(segment_data["texts"]),
    )
    print(f"Saved segments to {output_path}")

    questions_path = OUTPUT_DIR / "questions.json"
    with open(questions_path, "w") as f:
        json.dump(all_questions, f, indent=2)
    print(f"Saved {len(all_questions)} questions to {questions_path}")

    print("\nCategory breakdown:")
    from collections import Counter

    cats = Counter(q["category"] for q in all_questions)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
