"""Measure pairwise cosine-similarity distribution on LoCoMo turns.

Tells us whether text-embedding-3-small produces a separable distribution
between semantically-related and unrelated conversational turns, i.e.
whether any single similarity_threshold can cleanly separate "same topic"
from "different topic" on real dialogue.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from statistics import quantiles

import numpy as np
import openai
from dotenv import load_dotenv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_messages(conv_index: int, max_sessions: int) -> list[dict]:
    data_path = _repo_root() / "evaluation" / "data" / "locomo10.json"
    data = json.loads(data_path.read_text())
    item = data[conv_index]
    conv = item["conversation"]
    out = []
    for s in range(1, max_sessions + 1):
        key = f"session_{s}"
        if key not in conv:
            break
        for i, m in enumerate(conv[key]):
            out.append(
                {
                    "session": s,
                    "index": i,
                    "speaker": m["speaker"],
                    "text": m["text"],
                    "dia_id": m.get("dia_id"),
                }
            )
    return out


def build_cotopic_pairs(
    conv_index: int, max_sessions: int, msgs: list[dict]
) -> set[tuple[int, int]]:
    """Return a set of (i, j) index pairs that co-appear in some QA's evidence.

    LoCoMo evidence and dia_id both use the form "D<session>:<idx>"."""
    data_path = _repo_root() / "evaluation" / "data" / "locomo10.json"
    item = json.loads(data_path.read_text())[conv_index]
    dia_to_index: dict[str, int] = {
        str(m["dia_id"]): i for i, m in enumerate(msgs) if m.get("dia_id")
    }
    pairs: set[tuple[int, int]] = set()
    for qa in item["qa"]:
        evid = qa.get("evidence") or []
        ids: list[int] = []
        for e in evid:
            if not isinstance(e, str):
                continue
            idx = dia_to_index.get(e)
            if idx is not None:
                ids.append(idx)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = sorted((ids[a], ids[b]))
                pairs.add((i, j))
    return pairs


async def embed_all(texts: list[str]) -> np.ndarray:
    load_dotenv(_repo_root() / ".env", override=False)
    load_dotenv(_repo_root() / "evaluation" / ".env", override=True)
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions=1536,
    )
    vectors = np.array([d.embedding for d in resp.data])
    # L2-normalize so dot product = cosine
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T


def summarize(values: list[float], label: str) -> None:
    if not values:
        print(f"{label}: empty")
        return
    qs = quantiles(values, n=20)
    p5, p25, p50, p75, p95 = qs[0], qs[4], qs[9], qs[14], qs[18]
    print(
        f"{label:<40} "
        f"n={len(values):>5} "
        f"min={min(values):.3f} "
        f"p5={p5:.3f} p25={p25:.3f} p50={p50:.3f} p75={p75:.3f} p95={p95:.3f} "
        f"max={max(values):.3f}"
    )


async def main():
    max_sessions = int(os.getenv("PROBE_MAX_SESSIONS", "99"))
    msgs = load_messages(conv_index=0, max_sessions=max_sessions)
    texts = [f"{m['speaker']}: {m['text']}" for m in msgs]
    print(f"loaded {len(msgs)} messages from conv-0 (max_sessions={max_sessions})")

    cotopic = build_cotopic_pairs(conv_index=0, max_sessions=max_sessions, msgs=msgs)
    print(f"ground-truth same-topic pairs (QA evidence co-occurrence): {len(cotopic)}")

    emb = await embed_all(texts)
    sim = pairwise_cosine(emb)

    within_session = []
    across_session = []
    same_topic = []
    diff_topic = []
    for i in range(len(msgs)):
        for j in range(i + 1, len(msgs)):
            s = float(sim[i, j])
            if msgs[i]["session"] == msgs[j]["session"]:
                within_session.append(s)
            else:
                across_session.append(s)
            if (i, j) in cotopic:
                same_topic.append(s)
            else:
                diff_topic.append(s)

    print()
    print("=== SESSION-based buckets (epoch, not topic) ===")
    summarize(within_session, "within-session")
    summarize(across_session, "across-session")

    print("\n=== TOPIC-based buckets (QA-evidence co-occurrence) ===")
    summarize(same_topic, "same-topic (co-cited)")
    summarize(diff_topic, "diff-topic (not co-cited)")

    print()
    if same_topic and diff_topic:
        t_p25 = (
            quantiles(same_topic, n=4)[0] if len(same_topic) >= 4 else min(same_topic)
        )
        d_p75 = (
            quantiles(diff_topic, n=4)[2] if len(diff_topic) >= 4 else max(diff_topic)
        )
        t_p50 = quantiles(same_topic, n=2)[0] if len(same_topic) >= 2 else same_topic[0]
        d_p50 = quantiles(diff_topic, n=2)[0] if len(diff_topic) >= 2 else diff_topic[0]
        print(f"same-topic p25 = {t_p25:.3f}, diff-topic p75 = {d_p75:.3f}")
        print(f"same-topic p50 = {t_p50:.3f}, diff-topic p50 = {d_p50:.3f}")
        overlap = t_p25 <= d_p75
        print(
            f"distributions overlap at the quartile boundary: "
            f"{'YES (no clean threshold)' if overlap else 'NO (separable)'}"
        )

        # Best-threshold sweep
        print("\nThreshold sweep (separability):")
        print(f"{'threshold':>10}  {'TPR':>6}  {'FPR':>6}  {'precision':>10}")
        for t in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            tp = sum(1 for s in same_topic if s >= t)
            fp = sum(1 for s in diff_topic if s >= t)
            fn = len(same_topic) - tp
            tn = len(diff_topic) - fp
            tpr = tp / (tp + fn) if (tp + fn) else 0
            fpr = fp / (fp + tn) if (fp + tn) else 0
            prec = tp / (tp + fp) if (tp + fp) else 0
            print(f"{t:>10.2f}  {tpr:>6.3f}  {fpr:>6.3f}  {prec:>10.3f}")

    # Most-similar pairs overall
    print("\nTop 5 most-similar distinct pairs:")
    pairs = []
    for i in range(len(msgs)):
        for j in range(i + 1, len(msgs)):
            pairs.append((float(sim[i, j]), i, j))
    pairs.sort(reverse=True)
    for s, i, j in pairs[:5]:
        a = msgs[i]
        b = msgs[j]
        print(
            f"  sim={s:.3f}  s{a['session']}m{a['index']}[{a['speaker']}] "
            f"vs s{b['session']}m{b['index']}[{b['speaker']}]"
        )
        print(f"    A: {a['text'][:90]}")
        print(f"    B: {b['text'][:90]}")


if __name__ == "__main__":
    asyncio.run(main())
