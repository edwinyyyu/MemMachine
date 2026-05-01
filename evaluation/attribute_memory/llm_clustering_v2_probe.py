"""Prototype variant of LLM clustering with stricter assignment + abstraction guard.

Two changes from llm_clustering_probe.py:

1. ASSIGN_PROMPT is stricter — explicitly rejects tangential merges and
   prefers NEW when in doubt.
2. After each description update, an ABSTRACTION_CHECK call asks whether
   the new description requires umbrella terms to cover its contents;
   if yes, the cluster is flagged and the current event is redirected
   to a NEW cluster instead.

Same data as llm_clustering_probe.py (conv-0 sess 1-2, 35 events).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

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
                {"session": s, "index": i, "speaker": m["speaker"], "text": m["text"]}
            )
    return out


@dataclass
class LLMCluster:
    id: str
    description: str
    members: list[int] = field(default_factory=list)


# Stricter: explicit about rejecting tangential merges.
ASSIGN_PROMPT = """You are grouping conversation turns by specific topic.

Existing topics:
{topics}

New turn:
{event}

Rules:
- Merge INTO an existing topic ONLY IF the new turn directly continues or elaborates that specific topic.
- DO NOT merge based on:
    * shared speaker or conversation partner
    * conversational style (greetings, thanks, praise)
    * the new turn tangentially referencing the topic while being about something different
    * a loose thematic umbrella ("self-care", "friendship", "hobbies")
- When in doubt, create NEW.

Reply with JSON only:
{{"topic_id": "<id from above, or 'NEW'>", "new_desc": "<if NEW, concrete <=15-word description>"}}"""

UPDATE_PROMPT = """Current topic description: "{desc}"
New turn added: "{event}"

Write an updated <=15-word description that captures the topic's SUBSTANCE using concrete nouns. Reply with only the description text."""

# Abstraction guard: asks whether the new description requires umbrella terms.
ABSTRACTION_CHECK = """Topic description: "{desc}"

Question: Does this description rely on umbrella/abstract terms (like "self-care", "activities", "hobbies", "interactions", "friendship", "lifestyle") to cover different specific topics, OR does it describe one concrete subject?

Reply with JSON only:
{{"verdict": "CONCRETE" | "UMBRELLA", "why": "<short reason>"}}"""


async def _call(client, prompt: str, model: str) -> str:
    resp = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'"topic_id"\s*:\s*"([^"]+)"', raw)
        if m:
            return {"topic_id": m.group(1)}
        m = re.search(r'"verdict"\s*:\s*"([^"]+)"', raw)
        if m:
            return {"verdict": m.group(1)}
        return {}


async def assign(client, model, clusters, event_text):
    if not clusters:
        return ("NEW", None)
    topics_block = "\n".join(f"  {c.id}: {c.description}" for c in clusters)
    prompt = ASSIGN_PROMPT.format(topics=topics_block, event=event_text)
    data = _parse_json(await _call(client, prompt, model))
    return (str(data.get("topic_id", "NEW")), data.get("new_desc"))


async def update_desc(client, model, cluster, event_text):
    prompt = UPDATE_PROMPT.format(desc=cluster.description, event=event_text)
    return (await _call(client, prompt, model)).strip().strip('"')


async def is_umbrella(client, model, description):
    prompt = ABSTRACTION_CHECK.format(desc=description)
    data = _parse_json(await _call(client, prompt, model))
    return str(data.get("verdict", "CONCRETE")).upper() == "UMBRELLA"


async def main():
    load_dotenv(_repo_root() / ".env", override=False)
    load_dotenv(_repo_root() / "evaluation" / ".env", override=True)
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    msgs = load_messages(conv_index=0, max_sessions=2)
    texts = [f"{m['speaker']}: {m['text']}" for m in msgs]
    print(f"loaded {len(msgs)} messages from conv-0 sessions 1-2")
    print(f"=== LLM online clustering v2 ({model}): stricter + abstraction guard ===")

    clusters: list[LLMCluster] = []
    for i, (m, text) in enumerate(zip(msgs, texts, strict=True)):
        tid, new_desc = await assign(client, model, clusters, text)
        picked = (
            None if tid == "NEW" else next((c for c in clusters if c.id == tid), None)
        )
        if picked is None:
            desc = (new_desc or text[:80]).strip().strip('"')
            new_id = f"t{len(clusters)}"
            clusters.append(LLMCluster(id=new_id, description=desc, members=[i]))
            print(
                f"  [{i:>2}] s{m['session']} {m['speaker']:<10} -> NEW {new_id}: {desc}"
            )
            continue

        # Provisional update
        proposed_desc = await update_desc(client, model, picked, text)
        if await is_umbrella(client, model, proposed_desc):
            # Reject the merge; start a fresh cluster with just this event.
            new_id = f"t{len(clusters)}"
            fresh = await update_desc(
                client,
                model,
                LLMCluster(id=new_id, description=text[:80], members=[]),
                text,
            )
            clusters.append(LLMCluster(id=new_id, description=fresh, members=[i]))
            print(
                f"  [{i:>2}] s{m['session']} {m['speaker']:<10} -> SPLIT from {picked.id}, NEW {new_id}: {fresh}"
            )
        else:
            picked.description = proposed_desc
            picked.members.append(i)
            print(
                f"  [{i:>2}] s{m['session']} {m['speaker']:<10} -> {picked.id}: {proposed_desc[:70]}"
            )

    print(f"\nClusters: {len(clusters)}")
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
