"""Prototype LLM-based online clustering with minimal cluster context.

Each cluster carries a <=50-token description maintained incrementally by
gpt-5-mini. For each incoming event, the LLM sees only the event text and
the list of cluster descriptions (not full cluster contents) to decide
assignment. After assignment, the chosen cluster's description is updated
by feeding only the current description + new event.

Reports:
* resulting cluster count + sizes
* per-cluster descriptions
* agreement with session-boundary baseline
* comparison to cosine clustering at sim_threshold=0.55 (same data)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

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
                }
            )
    return out


@dataclass
class LLMCluster:
    id: str
    description: str
    members: list[int] = field(default_factory=list)


ASSIGN_PROMPT = """You are grouping conversation turns by semantic topic.

Existing topics:
{topics}

New turn:
{event}

Reply with JSON only, no prose:
{{"topic_id": "<id from above, or 'NEW'>", "new_desc": "<if NEW, a <=15-word description>"}}

Guidance:
- Merge into an existing topic only if the new turn continues or elaborates that topic.
- Don't merge based on conversational style (greetings, thanks, etc.).
- Prefer creating a new topic over forcing a weak fit."""

UPDATE_PROMPT = """Current topic description: "{desc}"
New turn added to the topic: "{event}"

Write an updated <=15-word description that captures the topic's substance. Reply with only the description text, no quotes, no prose."""


async def _call(client, prompt: str, model: str) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


async def assign(
    client, model: str, clusters: list[LLMCluster], event_text: str
) -> tuple[str, str | None]:
    if not clusters:
        return ("NEW", None)
    topics_block = "\n".join(f"  {c.id}: {c.description}" for c in clusters)
    prompt = ASSIGN_PROMPT.format(topics=topics_block, event=event_text)
    raw = await _call(client, prompt, model)
    # JSON may be wrapped in code fence
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to find topic_id via regex
        m = re.search(r'"topic_id"\s*:\s*"([^"]+)"', raw)
        return (m.group(1) if m else "NEW", None)
    return (str(data.get("topic_id", "NEW")), data.get("new_desc"))


async def update_description(
    client, model: str, cluster: LLMCluster, event_text: str
) -> str:
    prompt = UPDATE_PROMPT.format(desc=cluster.description, event=event_text)
    return await _call(client, prompt, model)


def cosine_cluster(embeddings: np.ndarray, threshold: float) -> list[int]:
    """Same greedy-nearest-centroid logic as ClusterManager.assign."""
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    assignment: list[int] = []
    for vec in embeddings:
        if not centroids:
            centroids.append(vec.copy())
            counts.append(1)
            assignment.append(0)
            continue
        sims = np.array([float(np.dot(vec, c)) for c in centroids])
        best = int(sims.argmax())
        if sims[best] >= threshold:
            n = counts[best]
            centroids[best] = (centroids[best] * n + vec) / (n + 1)
            counts[best] = n + 1
            assignment.append(best)
        else:
            centroids.append(vec.copy())
            counts.append(1)
            assignment.append(len(centroids) - 1)
    return assignment


async def embed_all(client, texts: list[str]) -> np.ndarray:
    resp = await client.embeddings.create(
        model="text-embedding-3-small", input=texts, dimensions=1536
    )
    vecs = np.array([d.embedding for d in resp.data])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


async def main():
    load_dotenv(_repo_root() / ".env", override=False)
    load_dotenv(_repo_root() / "evaluation" / ".env", override=True)
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    msgs = load_messages(conv_index=0, max_sessions=2)
    texts = [f"{m['speaker']}: {m['text']}" for m in msgs]
    print(f"loaded {len(msgs)} messages from conv-0 sessions 1-2")

    # Cosine baseline
    emb = await embed_all(client, texts)
    for sim in (0.50, 0.55, 0.60, 0.65):
        assignment = cosine_cluster(emb, threshold=sim)
        sizes = {}
        for c in assignment:
            sizes[c] = sizes.get(c, 0) + 1
        print(
            f"cosine sim={sim:.2f}  clusters={len(sizes)}  "
            f"sizes={sorted(sizes.values(), reverse=True)}"
        )

    # LLM clustering
    print(f"\n=== LLM online clustering ({model}) ===")
    clusters: list[LLMCluster] = []
    for i, (m, text) in enumerate(zip(msgs, texts, strict=True)):
        tid, new_desc = await assign(client, model, clusters, text)
        if tid == "NEW" or not any(c.id == tid for c in clusters):
            new_id = f"t{len(clusters)}"
            desc = (new_desc or text[:80]).strip().strip('"')
            cluster = LLMCluster(id=new_id, description=desc, members=[i])
            clusters.append(cluster)
            print(
                f"  [{i:>2}] s{m['session']} {m['speaker']:<10} -> NEW {new_id}: {desc}"
            )
        else:
            cluster = next(c for c in clusters if c.id == tid)
            cluster.members.append(i)
            new_desc_str = await update_description(client, model, cluster, text)
            cluster.description = new_desc_str.strip().strip('"')
            print(
                f"  [{i:>2}] s{m['session']} {m['speaker']:<10} -> {tid}: "
                f"{cluster.description[:70]}"
            )

    print(f"\nLLM clusters: {len(clusters)}")
    print(f"sizes: {sorted((len(c.members) for c in clusters), reverse=True)}")
    print("\nFinal cluster descriptions:")
    for c in clusters:
        members_str = ",".join(
            f"s{msgs[i]['session']}m{msgs[i]['index']}" for i in c.members
        )
        print(f"  {c.id} ({len(c.members)}): {c.description}")
        print(f"    members: {members_str}")


if __name__ == "__main__":
    asyncio.run(main())
