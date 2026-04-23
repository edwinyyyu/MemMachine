"""LongMemEval hard-category setup.

Build a subsample of LongMemEval questions in the three HARD categories:
  - multi-session
  - single-session-preference
  - temporal-reasoning

For each question, flatten its haystack sessions into per-turn Segment rows
keyed by conversation_id=question_id (so retrieval is scoped per-question).
Embed all turn texts with text-embedding-3-small using the shared bestshot
cache. Save:

  data/questions_longmemeval_hard.json  (eval-format questions list)
  data/longmemeval_hard_segments.npz     (SegmentStore-format segments)

Sample size: 30 per category (90 total if possible), preferring smaller
haystack sizes to cap ingestion cost.

Usage:
    uv run python longmemeval_hard_setup.py
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

from associative_recall import CACHE_DIR, EMBED_MODEL, EmbeddingCache

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

LONGMEM_SRC = (
    Path(__file__).resolve().parents[1] / "data" / "longmemeval_s_cleaned.json"
)
DATA_DIR = Path(__file__).resolve().parent / "data"
SEGMENTS_NPZ = DATA_DIR / "longmemeval_hard_segments.npz"
QUESTIONS_JSON = DATA_DIR / "questions_longmemeval_hard.json"

# Dedicated lme-hard caches (avoid polluting/bloating giant shared caches).
LMEHARD_EMB_CACHE = CACHE_DIR / "lmehard_embedding_cache.json"
LMEHARD_LLM_CACHE = CACHE_DIR / "lmehard_llm_cache.json"

HARD_TYPES = (
    "multi-session",
    "single-session-preference",
    "temporal-reasoning",
)
TARGET_PER_TYPE = 30
SEED = 7


class LmeHardEmbeddingCache(EmbeddingCache):
    """Dedicated lme-hard embedding cache. No inheritance of massive
    bestshot cache — we're embedding per-question turns not shared with
    other benchmarks.
    """

    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = LMEHARD_EMB_CACHE
        self._cache: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._cache = {}


def stratified_sample(
    all_qs: list[dict],
    target_per_type: int,
    seed: int = SEED,
) -> list[dict]:
    """Pick target_per_type per hard category. Prefer smaller haystacks."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in all_qs:
        if q["question_type"] in HARD_TYPES:
            by_type[q["question_type"]].append(q)

    chosen: list[dict] = []
    for t in HARD_TYPES:
        pool = list(by_type[t])
        # Tag each with haystack size, break ties randomly.
        sizes = [sum(len(s) for s in q["haystack_sessions"]) for q in pool]
        scored = [(sizes[i], rng.random(), i) for i in range(len(pool))]
        scored.sort(key=lambda x: (x[0], x[1]))
        want = min(target_per_type, len(pool))
        picked = [pool[i] for _, _, i in scored[:want]]
        chosen.extend(picked)
    return chosen


def flatten_to_segments(question: dict) -> tuple[list[dict], set[int]]:
    """Return (segments, gold_turn_ids) for one question.

    Each segment: {conversation_id, turn_id, role, text, session_id}.
    Gold turn_ids = all turns in any answer_session_ids session.
    """
    qid = question["question_id"]
    gold_sessions = set(question.get("answer_session_ids") or [])
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


def embed_all(
    client: OpenAI,
    cache: LmeHardEmbeddingCache,
    texts: list[str],
    batch_size: int = 128,
) -> np.ndarray:
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
    print(f"  Need to embed {total} new turn texts (cached: {n - total}).",
          flush=True)
    t0 = time.time()
    done = 0
    for start in range(0, total, batch_size):
        batch = to_compute[start:start + batch_size]
        batch_texts = [t for _, t in batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
        for (i, t), embed_data in zip(batch, resp.data):
            emb = np.array(embed_data.embedding, dtype=np.float32)
            cache.put(t, emb)
            out[i] = emb
        done += len(batch)
        if done % (batch_size * 10) == 0 or done >= total:
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            print(
                f"    [{done}/{total}] rate={rate:.1f}/s eta={eta:.0f}s",
                flush=True,
            )
            cache.save()
    cache.save()

    # Fill any holes with zero vector as safety.
    for i in range(n):
        if out[i] is None:
            out[i] = empty_vec
    return np.stack(out, axis=0)


def main() -> None:
    t_all = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {LONGMEM_SRC} ...", flush=True)
    with open(LONGMEM_SRC) as f:
        all_qs = json.load(f)
    print(f"  loaded {len(all_qs)} questions", flush=True)

    chosen = stratified_sample(all_qs, TARGET_PER_TYPE)
    cat_counts: dict[str, int] = defaultdict(int)
    for q in chosen:
        cat_counts[q["question_type"]] += 1
    print(f"Sampled {len(chosen)} questions:")
    for t in HARD_TYPES:
        print(f"  {t}: {cat_counts[t]}")

    all_segments: list[dict] = []
    questions_out: list[dict] = []
    for qi, q in enumerate(chosen):
        segs, gold_tids = flatten_to_segments(q)
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
        f"Flattened: total_turns={sum(turns_per_q)} "
        f"mean_turns/q={sum(turns_per_q)/len(chosen):.1f} "
        f"min={min(turns_per_q)} max={max(turns_per_q)}",
        flush=True,
    )
    print(
        f"  gold turns/q: mean={sum(gold_per_q)/len(gold_per_q):.1f} "
        f"min={min(gold_per_q)} max={max(gold_per_q)}",
        flush=True,
    )

    client = OpenAI(timeout=60.0)
    cache = LmeHardEmbeddingCache()
    texts = [s["text"] for s in all_segments]
    print(f"Embedding {len(texts)} turn texts...", flush=True)
    embeddings = embed_all(client, cache, texts)
    print(
        f"  embeddings shape={embeddings.shape} dtype={embeddings.dtype}",
        flush=True,
    )

    conv_ids = np.array(
        [s["conversation_id"] for s in all_segments], dtype=object
    )
    turn_ids = np.array(
        [s["turn_id"] for s in all_segments], dtype=np.int64
    )
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

    with open(QUESTIONS_JSON, "w") as f:
        json.dump(questions_out, f, indent=2)
    print(f"  saved questions: {QUESTIONS_JSON}", flush=True)

    print(f"\nDone in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
