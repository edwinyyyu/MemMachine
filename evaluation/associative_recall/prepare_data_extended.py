"""Prepare extended dataset for associative recall experiments.

Adds more BEAM conversations and LoCoMo conversations to the existing dataset.
Saves to a separate npz file to avoid overwriting the original.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

BEAM_PATH = Path(__file__).resolve().parents[1] / "data" / "beam" / "100k.json"
LOCOMO_PATH = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

# BEAM: use conversations 4-8 (we already have 1-3)
BEAM_CONV_RANGE = range(3, 8)  # indices 3-7 = conversations 4-8

# LoCoMo categories mapping:
# 1 = single-hop, 2 = temporal, 3 = multi-hop, 4 = open-domain, 5 = adversarial
LOCOMO_CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}

# Only use categories with evidence (skip adversarial/unanswerable)
LOCOMO_TARGET_CATEGORIES = {1, 2, 3}


def extract_beam_segments(conversation: dict) -> list[dict]:
    """Extract segments from a BEAM conversation."""
    segments = []
    conv_id = f"beam_{conversation['conversation_id']}"
    for session in conversation["chat"]:
        for turn in session:
            turn_id = turn["id"]
            role = turn["role"]
            content = turn["content"]
            if "->" in content:
                content = content.split("->")[0].strip()
            if not content.strip():
                continue
            segments.append({
                "conversation_id": conv_id,
                "turn_id": turn_id,
                "role": role,
                "text": content,
            })
    return segments


def extract_beam_questions(conversation: dict) -> list[dict]:
    """Extract questions from a BEAM conversation."""
    import ast
    conv_id = f"beam_{conversation['conversation_id']}"
    pq_raw = conversation.get("probing_questions", "{}")
    if isinstance(pq_raw, str):
        try:
            pq = json.loads(pq_raw)
        except Exception:
            pq = ast.literal_eval(pq_raw)
    else:
        pq = pq_raw

    target_categories = {
        "event_ordering", "instruction_following", "summarization",
        "multi_session_reasoning", "information_extraction",
        "temporal_reasoning", "contradiction_resolution",
        "knowledge_update", "preference_following",
    }

    questions = []
    for category, items in pq.items():
        if category == "abstention" or category not in target_categories:
            continue
        for i, q in enumerate(items):
            source_ids = _flatten_source_chat_ids(q.get("source_chat_ids"))
            if not source_ids:
                continue
            questions.append({
                "conversation_id": conv_id,
                "category": f"beam_{category}",
                "question_index": i,
                "question": q["question"],
                "source_chat_ids": source_ids,
                "ideal_response": q.get("ideal_response", ""),
                "benchmark": "beam",
            })
    return questions


def _flatten_source_chat_ids(raw: object) -> list[int]:
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
            result.extend(_flatten_source_chat_ids(v))
        return result
    return []


def extract_locomo_segments(entry: dict) -> list[dict]:
    """Extract segments from a LoCoMo conversation."""
    conv_id = f"locomo_{entry['sample_id']}"
    conv = entry["conversation"]
    segments = []
    global_turn_id = 0

    # Build dia_id -> global_turn_id mapping
    dia_id_map = {}

    session_idx = 1
    while True:
        session_key = f"session_{session_idx}"
        if session_key not in conv or not isinstance(conv[session_key], list):
            break
        for turn in conv[session_key]:
            dia_id = turn.get("dia_id", "")
            text = turn.get("text", "")
            speaker = turn.get("speaker", "unknown")
            if not text.strip():
                global_turn_id += 1
                continue
            role = "user" if speaker == conv.get("speaker_a", "") else "assistant"
            dia_id_map[dia_id] = global_turn_id
            segments.append({
                "conversation_id": conv_id,
                "turn_id": global_turn_id,
                "role": role,
                "text": text,
            })
            global_turn_id += 1
        session_idx += 1

    return segments, dia_id_map


def extract_locomo_questions(
    entry: dict, dia_id_map: dict[str, int]
) -> list[dict]:
    """Extract questions from a LoCoMo conversation."""
    conv_id = f"locomo_{entry['sample_id']}"
    questions = []

    for q in entry["qa"]:
        cat = q.get("category")
        if cat not in LOCOMO_TARGET_CATEGORIES:
            continue

        evidence = q.get("evidence", [])
        if not evidence:
            continue

        # Map evidence dia_ids to global turn_ids
        source_ids = []
        for e in evidence:
            if isinstance(e, str) and e in dia_id_map:
                source_ids.append(dia_id_map[e])

        if not source_ids:
            continue

        cat_name = LOCOMO_CATEGORY_NAMES.get(cat, f"cat{cat}")
        questions.append({
            "conversation_id": conv_id,
            "category": f"locomo_{cat_name}",
            "question_index": len(questions),
            "question": q["question"],
            "source_chat_ids": source_ids,
            "ideal_response": q.get("answer", ""),
            "benchmark": "locomo",
        })

    return questions


def truncate_text(text: str, max_chars: int = 8000) -> str:
    """Truncate text to stay under embedding model token limit."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    # Truncate long texts
    texts = [truncate_text(t) for t in texts]
    all_embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        print(f"  Embedding batch {start // BATCH_SIZE + 1} "
              f"({len(batch)} texts)...", flush=True)
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        time.sleep(0.1)
    return np.array(all_embeddings, dtype=np.float32)


def main() -> None:
    client = OpenAI()

    all_segments: list[dict] = []
    all_questions: list[dict] = []

    # --- BEAM conversations 4-8 ---
    print(f"Loading BEAM conversations from {BEAM_PATH}...")
    with open(BEAM_PATH) as f:
        beam_data = json.load(f)

    for idx in BEAM_CONV_RANGE:
        if idx >= len(beam_data):
            print(f"  BEAM conv index {idx} out of range (max {len(beam_data)-1})")
            break
        conv = beam_data[idx]
        segments = extract_beam_segments(conv)
        questions = extract_beam_questions(conv)
        all_segments.extend(segments)
        all_questions.extend(questions)
        print(f"  BEAM conv {conv['conversation_id']}: "
              f"{len(segments)} segments, {len(questions)} questions")

    # --- LoCoMo conversations ---
    print(f"\nLoading LoCoMo conversations from {LOCOMO_PATH}...")
    with open(LOCOMO_PATH) as f:
        locomo_data = json.load(f)

    # Use first 3 LoCoMo conversations
    for entry in locomo_data[:3]:
        segments, dia_id_map = extract_locomo_segments(entry)
        questions = extract_locomo_questions(entry, dia_id_map)
        all_segments.extend(segments)
        all_questions.extend(questions)
        print(f"  LoCoMo {entry['sample_id']}: "
              f"{len(segments)} segments, {len(questions)} questions")

    print(f"\nTotal NEW: {len(all_segments)} segments, "
          f"{len(all_questions)} questions")

    # Category breakdown
    cats = Counter(q["category"] for q in all_questions)
    print("\nCategory breakdown:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    # Benchmark breakdown
    benchmarks = Counter(q["benchmark"] for q in all_questions)
    print("\nBenchmark breakdown:")
    for b, count in sorted(benchmarks.items()):
        print(f"  {b}: {count}")

    # Embed
    texts = [s["text"] for s in all_segments]
    print(f"\nEmbedding {len(texts)} segments...")
    embeddings = embed_texts(client, texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "segments_extended.npz"
    np.savez(
        output_path,
        embeddings=embeddings,
        conversation_ids=np.array([s["conversation_id"] for s in all_segments]),
        turn_ids=np.array([s["turn_id"] for s in all_segments], dtype=np.int32),
        roles=np.array([s["role"] for s in all_segments]),
        texts=np.array([s["text"] for s in all_segments]),
    )
    print(f"\nSaved segments to {output_path}")

    questions_path = OUTPUT_DIR / "questions_extended.json"
    with open(questions_path, "w") as f:
        json.dump(all_questions, f, indent=2)
    print(f"Saved {len(all_questions)} questions to {questions_path}")


if __name__ == "__main__":
    main()
