"""Partitioned architecture (round-7 style).

- Per-entity topic logs keyed <Entity>/<Category>
- Per-role slot logs keyed <Holder>/<Category>/<Role> (pointer to entity)
- Append-only; supersede/invalidate expressed via ref (uuid, relation)

Writer: fused LLM call that emits, per turn:
  {
    "facts": [
      {"topic": "User/Employment",
       "text": "User is a software engineer at Anthropic",
       "refs": [{"uuid": "<prior>", "relation": "invalidate"}]}
    ],
    "slot_updates": [
      {"slot_id": "User/Employment/boss", "filler": "@Alice",
       "refs": [{"uuid": "<prior slot entry>", "relation": "supersede"}]}
    ],
    "introduced_entities": [...]
  }

Reader: two-stage
  1. Determine target entities / slots from the question
  2. Pull relevant logs, render chronological entries with refs, LLM reader
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

HERE = Path(__file__).resolve().parent
ROUND9 = HERE.parent
ROUND7 = ROUND9.parent / "round7"
sys.path.insert(0, str(ROUND7 / "experiments"))

from _common import Budget, Cache, cosine, embed_batch, extract_json, llm  # noqa: E402

Relation = Literal["clarify", "refine", "supersede", "invalidate"]


@dataclass
class Ref:
    uuid: str
    relation: Relation


@dataclass
class TopicEntry:
    uuid: str
    topic: str  # "User/Employment"
    ts: int
    text: str
    refs: list[Ref] = field(default_factory=list)


@dataclass
class SlotEntry:
    uuid: str
    slot_id: str  # "User/Employment/boss"
    ts: int
    filler: str | None  # "@Marcus" or None
    refs: list[Ref] = field(default_factory=list)


@dataclass
class PartitionedStore:
    topics: dict[str, list[TopicEntry]] = field(
        default_factory=lambda: defaultdict(list)
    )
    slots: dict[str, list[SlotEntry]] = field(default_factory=lambda: defaultdict(list))
    known_entities: set[str] = field(default_factory=lambda: {"User"})
    # For embedding-based retrieval we also keep a flat index
    all_entries: list[TopicEntry] = field(default_factory=list)
    all_slot_entries: list[SlotEntry] = field(default_factory=list)


WRITE_PROMPT = """You are a semantic-memory writer using ENTITY-PARTITIONED
append-only logs. Each "topic" is a log for one entity's category, keyed
<Entity>/<Category> (e.g. User/Employment, Jamie/Profile, Luna/Profile).

Role assignments (boss, manager, partner, trainer, mentor, etc.) go on SLOT
LOGS keyed <Holder>/<Category>/<Role> with filler = "@<Entity>".

When the turn changes a prior value, write the new entry with refs pointing
at prior entries.

RELATIONS for refs:
- clarify: adds detail
- refine: narrows or qualifies
- supersede: replaces (old stays marked)
- invalidate: prior entry was wrong

KNOWN ENTITIES: {known_entities}
KNOWN SLOTS (with current filler):
{known_slots}

RECENT TOPIC LOGS (most relevant 10 entries from the full store):
{recent_topics}

TURN: "{turn_text}"

Emit JSON:
{{
  "facts": [
    {{"topic": "<Entity>/<Category>",
      "text": "<atomic fact>",
      "refs": [{{"uuid": "<prior entry uuid>", "relation": "..."}}]}}
  ],
  "slot_updates": [
    {{"slot_id": "<Holder>/<Category>/<Role>",
      "filler": "@<Entity>" | null,
      "refs": [{{"uuid": "<prior slot entry>", "relation": "supersede|invalidate"}}]}}
  ],
  "introduced_entities": ["<Entity>", ...]
}}

RULES
- Use exactly ONE topic per fact (the entity it's most directly about).
- Qualitative facts about a person go on their own Profile log (e.g.
  Marcus/Profile). Role assignments go on the slot log.
- If a fact introduces a NEW named entity, also emit a separate fact on that
  entity's own log AND list the entity in "introduced_entities".
- Filler turns (weather, chitchat) -> empty facts array.
- Output JSON only.
"""


def _slots_block(store: PartitionedStore) -> str:
    if not store.slots:
        return "(none)"
    lines = []
    for sid, history in store.slots.items():
        live = [
            e
            for e in history
            if not any(
                r.relation == "invalidate"
                for other in history
                if other.ts > e.ts
                for r in other.refs
                if r.uuid == e.uuid
            )
        ]
        cur = live[-1] if live else None
        lines.append(f"  {sid} -> {cur.filler if cur else 'vacant'}")
    return "\n".join(lines)


def _recent_topics_block(store: PartitionedStore, top_n: int = 10) -> str:
    # Show last 10 topic entries overall (time-ordered)
    recent = store.all_entries[-top_n:]
    lines = []
    for e in recent:
        ref_str = ""
        if e.refs:
            ref_str = " refs=" + ",".join(f"{r.uuid}:{r.relation}" for r in e.refs)
        lines.append(f"[{e.uuid}] t{e.ts} {e.topic} :: {e.text}{ref_str}")
    return "\n".join(lines) if lines else "(empty)"


def write_turn(
    turn_idx: int,
    turn_text: str,
    store: PartitionedStore,
    cache: Cache,
    budget: Budget,
) -> None:
    prompt = WRITE_PROMPT.format(
        known_entities=", ".join(sorted(store.known_entities)),
        known_slots=_slots_block(store),
        recent_topics=_recent_topics_block(store),
        turn_text=turn_text,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return
    # register introduced entities
    for e in obj.get("introduced_entities") or []:
        if isinstance(e, str) and e:
            store.known_entities.add(e)

    # facts
    facts = obj.get("facts") or []
    for i, f in enumerate(facts):
        if not isinstance(f, dict):
            continue
        topic = f.get("topic") or "User/Other"
        text = (f.get("text") or "").strip()
        if not text:
            continue
        refs_raw = f.get("refs") or []
        refs = []
        for r in refs_raw:
            if (
                isinstance(r, dict)
                and r.get("uuid")
                and r.get("relation")
                in ("clarify", "refine", "supersede", "invalidate")
            ):
                refs.append(Ref(uuid=r["uuid"], relation=r["relation"]))
        uuid = f"t{turn_idx:03d}_f{i}"
        entry = TopicEntry(uuid=uuid, topic=topic, ts=turn_idx, text=text, refs=refs)
        store.topics[topic].append(entry)
        store.all_entries.append(entry)
        # first segment is the entity
        ent = topic.split("/")[0]
        if ent:
            store.known_entities.add(ent)

    # slot_updates
    slot_updates = obj.get("slot_updates") or []
    for j, su in enumerate(slot_updates):
        if not isinstance(su, dict) or not su.get("slot_id"):
            continue
        sid = su["slot_id"]
        filler = su.get("filler")
        refs_raw = su.get("refs") or []
        refs = []
        for r in refs_raw:
            if (
                isinstance(r, dict)
                and r.get("uuid")
                and r.get("relation")
                in ("clarify", "refine", "supersede", "invalidate")
            ):
                refs.append(Ref(uuid=r["uuid"], relation=r["relation"]))
        uuid = f"t{turn_idx:03d}_s{j}"
        entry = SlotEntry(uuid=uuid, slot_id=sid, ts=turn_idx, filler=filler, refs=refs)
        store.slots[sid].append(entry)
        store.all_slot_entries.append(entry)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

READ_PROMPT = """Answer a question using entity-partitioned semantic-memory logs.

Each entry is tagged with its topic (e.g. User/Employment) or slot
(e.g. User/Employment/boss). Refs annotate how entries relate:
- clarify: adds detail
- refine: narrows/qualifies
- supersede: replaces (older entry is no longer current)
- invalidate: prior entry was wrong

The current state for a topic is the most recent non-invalidated claim.

RETRIEVED LOGS:
{entries_block}

ROLE SLOTS:
{slot_block}

QUESTION: {question}

Answer concisely.
"""


def extract_question_entities(question: str) -> list[str]:
    import re

    q = re.sub(r"[^a-zA-Z0-9\s']", " ", question)
    STOP = {
        "What",
        "Where",
        "When",
        "Who",
        "How",
        "Why",
        "Which",
        "Has",
        "Have",
        "Do",
        "Does",
        "Is",
        "Are",
        "Was",
        "Were",
        "Tell",
        "Before",
        "The",
        "A",
        "An",
        "And",
        "Or",
        "But",
        "I",
        "Know",
        "Me",
        "My",
        "Him",
        "Her",
        "Them",
        "They",
        "She",
        "He",
        "Currently",
        "Right",
        "Full",
        "Of",
    }
    ents = []
    for w in q.split():
        if len(w) > 1 and w[0].isupper() and w not in STOP:
            ents.append(w)
    return ents + ["User"]


def retrieve_topics(
    question: str,
    store: PartitionedStore,
    cache: Cache,
    budget: Budget,
    top_k: int = 10,
) -> tuple[list[TopicEntry], list[SlotEntry]]:
    # Step 1: figure out which entities are mentioned in question
    q_ents = extract_question_entities(question)

    # Step 2: gather topic entries from those entities
    ent_entries: list[TopicEntry] = []
    for ent in q_ents:
        for topic, entries in store.topics.items():
            if topic.startswith(ent + "/"):
                ent_entries.extend(entries)
    # Deduplicate
    seen = set()
    dedup = []
    for e in ent_entries:
        if e.uuid not in seen:
            seen.add(e.uuid)
            dedup.append(e)
    ent_entries = dedup

    # Step 3: embedding top-K across ALL topic entries (broader recall)
    all_texts = [e.text for e in store.all_entries]
    if all_texts:
        embs = embed_batch(all_texts + [question], cache, budget)
        q_emb = embs[-1]
        entry_embs = embs[:-1]
        scores = [cosine(q_emb, v) for v in entry_embs]
        ranked = sorted(range(len(all_texts)), key=lambda i: scores[i], reverse=True)
        top_set = {store.all_entries[i].uuid for i in ranked[:top_k]}
        ent_uuids = {e.uuid for e in ent_entries}
        combined_uuids = ent_uuids | top_set
        entries = [e for e in store.all_entries if e.uuid in combined_uuids]
    else:
        entries = []

    # Step 4: chase refs (bring in entries referenced by retrieved ones, and the other way)
    uuid_to_entry = {e.uuid: e for e in store.all_entries}
    added = set()
    selected_uuids = {e.uuid for e in entries}
    for e in entries:
        for r in e.refs:
            if r.uuid in uuid_to_entry and r.uuid not in selected_uuids:
                added.add(r.uuid)
    for e in store.all_entries:
        if e.uuid in selected_uuids:
            continue
        for r in e.refs:
            if r.uuid in selected_uuids:
                added.add(e.uuid)
    for u in added:
        entries.append(uuid_to_entry[u])

    # sort chronologically
    entries.sort(key=lambda e: e.ts)

    # Step 5: slot logs relevant to question
    slot_entries: list[SlotEntry] = []
    for ent in q_ents:
        for sid, history in store.slots.items():
            if sid.startswith(ent + "/"):
                slot_entries.extend(history)
    slot_entries.sort(key=lambda e: e.ts)

    return entries, slot_entries


def format_entries(entries: list[TopicEntry]) -> str:
    lines = []
    for e in entries:
        ref_str = ""
        if e.refs:
            ref_str = (
                " refs=[" + ",".join(f"{r.uuid}:{r.relation}" for r in e.refs) + "]"
            )
        lines.append(f"[{e.uuid}] t{e.ts} {e.topic} :: {e.text}{ref_str}")
    return "\n".join(lines) if lines else "(none)"


def format_slots(slots: list[SlotEntry]) -> str:
    lines = []
    for e in slots:
        ref_str = ""
        if e.refs:
            ref_str = (
                " refs=[" + ",".join(f"{r.uuid}:{r.relation}" for r in e.refs) + "]"
            )
        lines.append(f"[{e.uuid}] t{e.ts} {e.slot_id} filler={e.filler}{ref_str}")
    return "\n".join(lines) if lines else "(none)"


def answer_question(
    question: str,
    store: PartitionedStore,
    cache: Cache,
    budget: Budget,
    top_k: int = 10,
) -> str:
    entries, slots = retrieve_topics(question, store, cache, budget, top_k=top_k)
    prompt = READ_PROMPT.format(
        entries_block=format_entries(entries),
        slot_block=format_slots(slots),
        question=question,
    )
    return llm(prompt, cache, budget).strip()


def ingest_scenario(turns, cache: Cache, budget: Budget) -> PartitionedStore:
    store = PartitionedStore()
    for turn in turns:
        write_turn(turn.idx, turn.text, store, cache, budget)
    return store
