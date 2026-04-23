"""LongMemEval setup: subsample, convert to SegmentStore format, embed.

Step 1: Load longmemeval_s_cleaned.json, stratified sample 50 questions
        (even-per-question_type, preferring moderate haystack sizes).
Step 2: Flatten haystack sessions into turn-level segments keyed by
        (conversation_id=question_id, turn_id=global_turn_idx).
Step 3: Embed all new segment texts using text-embedding-3-small with cache.
Step 4: Save data/longmemeval_s_50q.json (raw subsample),
             data/longmemeval_s_50q_segments.npz (store format),
             data/questions_longmemeval_s_50q.json (eval format).

Usage:
    uv run python longmemeval_setup.py

Cache: uses BestshotEmbeddingCache (reads all prior caches, writes bestshot).
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import EMBED_MODEL
from best_shot import BestshotEmbeddingCache

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

LONGMEM_SRC = (
    Path(__file__).resolve().parents[1] / "data" / "longmemeval_s_cleaned.json"
)
DATA_DIR = Path(__file__).resolve().parent / "data"
SUBSAMPLE_RAW = DATA_DIR / "longmemeval_s_50q.json"
SEGMENTS_NPZ = DATA_DIR / "longmemeval_s_50q_segments.npz"
QUESTIONS_JSON = DATA_DIR / "questions_longmemeval_s_50q.json"

SEED = 7
TARGET_N = 50


# ---------------------------------------------------------------------------
# Step 1: stratified subsample
# ---------------------------------------------------------------------------
def stratified_subsample(
    all_qs: list[dict], target_n: int, seed: int = SEED
) -> list[dict]:
    """Even-per-question_type; within type, prefer medium haystack sizes."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in all_qs:
        by_type[q["question_type"]].append(q)

    types = sorted(by_type.keys())
    n_types = len(types)
    base = target_n // n_types
    rem = target_n - base * n_types

    # Order types by count (smallest first so they don't over-consume)
    types_sorted = sorted(types, key=lambda t: len(by_type[t]))

    # Extra slots go to the types with the most questions (they have slack)
    extra_targets = {t: 0 for t in types}
    for t in sorted(types, key=lambda t: -len(by_type[t]))[:rem]:
        extra_targets[t] = 1

    chosen: list[dict] = []
    for t in types_sorted:
        pool = list(by_type[t])
        # sort by "moderate" haystack size: closest to median of the pool
        sizes = [sum(len(s) for s in q["haystack_sessions"]) for q in pool]
        med = sorted(sizes)[len(sizes) // 2] if sizes else 0
        pool_scored = [
            (abs(sizes[i] - med), i) for i in range(len(pool))
        ]
        # Shuffle ties, then sort so we pick close-to-median reliably
        rng.shuffle(pool_scored)
        pool_scored.sort(key=lambda x: x[0])
        want = base + extra_targets[t]
        want = min(want, len(pool))
        picked = [pool[i] for _, i in pool_scored[:want]]
        chosen.extend(picked)

    # Safety: if we under-picked (rare, only if base=0 and no extras),
    # top up with random remaining.
    remaining = [q for q in all_qs if q not in chosen]
    rng.shuffle(remaining)
    while len(chosen) < target_n and remaining:
        chosen.append(remaining.pop())

    return chosen[:target_n]


# ---------------------------------------------------------------------------
# Step 2: flatten haystack into segments
# ---------------------------------------------------------------------------
def flatten_to_segments(question: dict) -> tuple[list[dict], set[int]]:
    """Return (segments_for_this_question, gold_turn_ids).

    Each segment: {conversation_id, turn_id, role, text, session_id}.
    Gold turn_ids are the indices of ALL turns in any session whose
    session_id is in answer_session_ids.
    """
    qid = question["question_id"]
    gold_sessions = set(question["answer_session_ids"])
    segs: list[dict] = []
    gold_tids: set[int] = set()
    turn_idx = 0
    sess_ids = question["haystack_session_ids"]
    sessions = question["haystack_sessions"]
    for sess_id, turns in zip(sess_ids, sessions):
        for turn in turns:
            role = turn.get("role", "user")
            text = turn.get("content", "")
            if not isinstance(text, str):
                text = str(text)
            segs.append({
                "conversation_id": qid,
                "turn_id": turn_idx,
                "role": role,
                "text": text,
                "session_id": sess_id,
            })
            if sess_id in gold_sessions:
                gold_tids.add(turn_idx)
            turn_idx += 1
    return segs, gold_tids


# ---------------------------------------------------------------------------
# Step 3: embed with cache
# ---------------------------------------------------------------------------
def embed_all(
    client: OpenAI,
    cache: BestshotEmbeddingCache,
    texts: list[str],
    batch_size: int = 96,
) -> np.ndarray:
    """Embed every text; cached entries are reused. Returns (N, D) float32."""
    n = len(texts)
    out: list[np.ndarray | None] = [None] * n
    to_compute: list[tuple[int, str]] = []
    empty_vec = np.zeros(1536, dtype=np.float32)
    for i, t in enumerate(texts):
        t_str = (t or "").strip()
        if not t_str:
            out[i] = empty_vec
            continue
        cached = cache.get(t_str)
        if cached is not None:
            out[i] = cached.astype(np.float32)
        else:
            to_compute.append((i, t_str))

    total = len(to_compute)
    if total:
        print(f"  Embedding {total} new texts...", flush=True)
    t0 = time.time()
    done = 0
    save_every = max(1, total // 20)  # ~5% increments
    for start in range(0, total, batch_size):
        batch = to_compute[start:start + batch_size]
        batch_texts = [t for _, t in batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
        for (i, t), embed_data in zip(batch, resp.data):
            emb = np.array(embed_data.embedding, dtype=np.float32)
            cache.put(t, emb)
            out[i] = emb
        done += len(batch)
        if done % (save_every * batch_size // batch_size or 1) < batch_size:
            cache.save()
        if done % (batch_size * 10) == 0 or done >= total:
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            print(
                f"    [{done}/{total}] rate={rate:.1f}/s eta={eta:.0f}s",
                flush=True,
            )
    cache.save()
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_all = time.time()
    print(f"Loading {LONGMEM_SRC} ...", flush=True)
    with open(LONGMEM_SRC) as f:
        all_qs = json.load(f)
    print(f"  loaded {len(all_qs)} questions", flush=True)

    # Step 1: subsample
    chosen = stratified_subsample(all_qs, TARGET_N, seed=SEED)
    type_counts: dict[str, int] = defaultdict(int)
    for q in chosen:
        type_counts[q["question_type"]] += 1
    print(f"Subsampled {len(chosen)} questions:")
    for t in sorted(type_counts.keys()):
        print(f"  {t}: {type_counts[t]}")

    # Save raw subsample
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Strip to reduce size if possible (keep full — we need haystacks for
    # future experiments). Write it.
    with open(SUBSAMPLE_RAW, "w") as f:
        json.dump(chosen, f)
    print(f"  saved raw: {SUBSAMPLE_RAW}", flush=True)

    # Step 2: flatten each question's haystack into segments
    all_segments: list[dict] = []
    questions_out: list[dict] = []
    total_turns = 0
    for qi, q in enumerate(chosen):
        segs, gold_tids = flatten_to_segments(q)
        total_turns += len(segs)
        all_segments.extend(segs)
        questions_out.append({
            "question_id": q["question_id"],
            "conversation_id": q["question_id"],
            "question_index": qi,
            "question": q["question"],
            "answer": q["answer"],
            "category": q["question_type"],
            "question_type": q["question_type"],
            "answer_session_ids": q["answer_session_ids"],
            "source_chat_ids": sorted(gold_tids),
            "source_ids": sorted(gold_tids),
            "num_source_turns": len(gold_tids),
            "num_haystack_turns": len(segs),
        })
    turns_per_q = [qq["num_haystack_turns"] for qq in questions_out]
    gold_per_q = [qq["num_source_turns"] for qq in questions_out]
    print(
        f"Flattened: total_turns={total_turns} mean_turns_per_q="
        f"{total_turns/len(chosen):.1f} min={min(turns_per_q)} "
        f"max={max(turns_per_q)}",
        flush=True,
    )
    print(
        f"  gold turns per question: mean={sum(gold_per_q)/len(gold_per_q):.1f} "
        f"min={min(gold_per_q)} max={max(gold_per_q)}",
        flush=True,
    )

    # Step 3: embed
    client = OpenAI(timeout=60.0)
    cache = BestshotEmbeddingCache()
    texts = [s["text"] for s in all_segments]
    print(f"Embedding {len(texts)} turn texts ...", flush=True)
    embeddings = embed_all(client, cache, texts)
    print(
        f"  embeddings: shape={embeddings.shape} dtype={embeddings.dtype}",
        flush=True,
    )

    # Save NPZ
    conv_ids = np.array([s["conversation_id"] for s in all_segments], dtype=object)
    turn_ids = np.array([s["turn_id"] for s in all_segments], dtype=np.int64)
    roles = np.array([s["role"] for s in all_segments], dtype=object)
    texts_arr = np.array([s["text"] for s in all_segments], dtype=object)
    session_ids = np.array(
        [s["session_id"] for s in all_segments], dtype=object
    )
    np.savez(
        SEGMENTS_NPZ,
        embeddings=embeddings,
        conversation_ids=conv_ids,
        turn_ids=turn_ids,
        roles=roles,
        texts=texts_arr,
        session_ids=session_ids,
    )
    print(f"  saved npz: {SEGMENTS_NPZ}", flush=True)

    # Save questions
    with open(QUESTIONS_JSON, "w") as f:
        json.dump(questions_out, f, indent=2)
    print(f"  saved questions: {QUESTIONS_JSON}", flush=True)

    print(f"\nDone in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
