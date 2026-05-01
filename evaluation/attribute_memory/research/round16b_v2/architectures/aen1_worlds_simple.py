"""AEN-1 WORLDS SIMPLE — variant of aen1_worlds with simpler world category sets.

Round 16B v1 used a 4-category space: real / hypothetical / fiction:pyrrhus /
fiction:novel_project. This caused the classifier to invent inconsistent slug
names ("fiction:pyrrhus" vs the expected "fiction:fantasy_rp"), tanking exact
classifier accuracy and (downstream) QA-with-world.

This variant collapses fiction subtypes:
  - n_cats=3 → {real, hypothetical, fiction}
  - n_cats=2 → {real, non_real}

Active-state injection is still scoped by world. Retrieval still routes by
question world. The writer still emits world tags per entry.

Public API mirrors aen1_worlds:
  ingest_turns(turns, cache, budget, *, n_cats=3, ...)
  answer_question(question, idx, cache, budget, *, world=None, ...)
  answer_question_no_world(...)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND16B_V2 = HERE.parent
RESEARCH = ROUND16B_V2.parent
ROUND16B = RESEARCH / "round16b_world_scoping"
ROUND15 = RESEARCH / "round15_active_chains"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND16B / "architectures"))
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_active  # noqa: E402
import aen1_simple  # noqa: E402
from _common import Budget, Cache, embed_batch, extract_json, llm  # noqa: E402

# ---------------------------------------------------------------------------
# Data model — same shape as aen1_worlds
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
    uuid: str
    ts: int
    text: str
    mentions: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    predicate: str | None = None
    world: str = "real"


@dataclass
class IndexedLog:
    entries: list[LogEntry]
    by_uuid: dict[str, LogEntry]
    mention_index: dict[str, list[str]]
    superseded_by: dict[str, str]
    supersede_head: dict[tuple[str, str, str], str]
    embed_by_uuid: dict[str, list[float]]


def build_index(entries: list[LogEntry], cache: Cache, budget: Budget) -> IndexedLog:
    by_uuid = {e.uuid: e for e in entries}

    mention_index: dict[str, list[str]] = {}
    for e in entries:
        for m in e.mentions:
            mention_index.setdefault(m, []).append(e.uuid)

    superseded_by: dict[str, str] = {}
    for e in entries:
        for r_uuid in e.refs:
            if r_uuid in by_uuid and r_uuid not in superseded_by:
                superseded_by[r_uuid] = e.uuid

    supersede_head: dict[tuple[str, str, str], str] = {}
    for e in entries:
        if e.predicate is None:
            continue
        if e.uuid in superseded_by:
            continue
        m = re.match(r"(@?[A-Za-z0-9_]+)\.(.+)", e.predicate)
        if not m:
            continue
        ent, pred = m.group(1), m.group(2)
        if not ent.startswith("@"):
            ent = "@" + ent
        supersede_head[(ent, pred, e.world)] = e.uuid

    texts = [e.text for e in entries]
    embs = embed_batch(texts, cache, budget) if entries else []
    embed_by_uuid = {e.uuid: embs[i] for i, e in enumerate(entries)}

    return IndexedLog(
        entries=entries,
        by_uuid=by_uuid,
        mention_index=mention_index,
        superseded_by=superseded_by,
        supersede_head=supersede_head,
        embed_by_uuid=embed_by_uuid,
    )


# ---------------------------------------------------------------------------
# World classifier — variant-aware
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT_3 = """You decide what "world" a chunk of conversation belongs to,
so that fictional/hypothetical content is not stored as real facts about the
user.

WORLDS (pick exactly one):
  - "real"          — actual facts about the user's life and people they know
                      in real life.
  - "hypothetical"  — speculation: "what if I moved to Paris", "imagine I
                      lived...", "if I were a vegan". NOT actual plans, those
                      are real.
  - "fiction"       — role-play, novels, games, daydreams, stories, jokes about
                      obvious untruths. Treat ALL of these as the same world.

Strong cues to flip to non-real worlds:
  - Hypothetical: "what if", "imagine", "if I were", "hypothetically", "if I
    moved to..."
  - Fiction: "in my novel/story/game", "let's role-play", "[character: X]",
    "as a dragon", "you're the dungeon master", D&D session content, sarcastic
    obvious untruths the user wouldn't claim ("I'm a billionaire").

Cues to flip BACK to real:
  - "ok anyway, in real life", "back to reality", "for real though",
    "irl", "actually", a clear topic change to mundane life (work, partner,
    pets, errands).

PERSISTENCE: once a non-real world is established, stay in it until the user
signals a return. The CURRENT WORLD below is what the previous batch was
tagged. Lean toward keeping it unless cues say to switch.

CURRENT WORLD: {current_world}

BATCH:
{turn_block}

Output JSON only:
{{
  "world": "real" | "hypothetical" | "fiction",
  "reason": "<one short sentence>"
}}
"""


CLASSIFY_PROMPT_2 = """You decide whether a chunk of conversation is about the
user's REAL life or about a NON-REAL world (anything that isn't actual facts
about the user).

WORLDS (pick exactly one):
  - "real"      — actual facts about the user's life and people they know in
                  real life. Includes plans, work, partner, pets, errands.
  - "non_real"  — anything that isn't real: hypotheticals ("what if I moved to
                  Paris"), role-play, novels, games, daydreams, stories,
                  jokes about obvious untruths the user wouldn't claim.

Strong cues to flip to non_real:
  - "what if", "imagine", "if I were", "hypothetically"
  - "in my novel/story/game", "let's role-play", "[character: X]",
    "as a dragon", "you're the dungeon master", D&D session content
  - Sarcastic obvious untruths ("I'm a billionaire", "I will write the Linux
    kernel from scratch tomorrow")

Cues to flip BACK to real:
  - "ok anyway, in real life", "back to reality", "for real though", "irl",
    "actually", a clear topic change to mundane life (work, partner, pets,
    errands).

PERSISTENCE: once non_real is established, stay in it until the user signals a
return. The CURRENT WORLD below is what the previous batch was tagged. Lean
toward keeping it unless cues say to switch.

CURRENT WORLD: {current_world}

BATCH:
{turn_block}

Output JSON only:
{{
  "world": "real" | "non_real",
  "reason": "<one short sentence>"
}}
"""


VALID_WORLDS = {
    3: {"real", "hypothetical", "fiction"},
    2: {"real", "non_real"},
}


def classify_world(
    batch_turns: list[tuple[int, str]],
    current_world: str,
    cache: Cache,
    budget: Budget,
    n_cats: int,
) -> tuple[str, str]:
    turn_block = "\n".join(f"TURN {i}: {t}" for i, t in batch_turns)
    if n_cats == 3:
        prompt = CLASSIFY_PROMPT_3.format(
            current_world=current_world,
            turn_block=turn_block,
        )
    elif n_cats == 2:
        prompt = CLASSIFY_PROMPT_2.format(
            current_world=current_world,
            turn_block=turn_block,
        )
    else:
        raise ValueError(f"n_cats must be 2 or 3, got {n_cats}")
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return current_world, "classifier_parse_error"
    world = obj.get("world")
    if not isinstance(world, str) or not world.strip():
        return current_world, "classifier_empty_world"
    world = world.strip().lower()
    # Light normalization
    if world in ("real", "real_life", "reality"):
        world = "real"
    elif world in ("hypothetical", "hypothesis", "what_if"):
        world = "hypothetical"
    elif world in (
        "fiction",
        "fictional",
        "story",
        "novel",
        "roleplay",
        "role-play",
        "game",
    ):
        world = "fiction"
    elif world in ("non_real", "non-real", "nonreal", "not_real"):
        world = "non_real"
    if world not in VALID_WORLDS[n_cats]:
        # Best-effort fallback: anything unknown maps to non-real partition.
        if n_cats == 2:
            world = "non_real"
        else:
            world = "fiction"
    reason = obj.get("reason") or ""
    if not isinstance(reason, str):
        reason = ""
    return world, reason


# ---------------------------------------------------------------------------
# Active-state injection — filtered by world
# ---------------------------------------------------------------------------


def gather_active_state(
    idx: IndexedLog | None,
    entities: set[str],
    world: str,
    max_active_state_size: int,
) -> list[LogEntry]:
    if idx is None:
        return []
    ent_tags = {f"@{e}" for e in entities}
    heads: list[LogEntry] = []
    seen = set()
    for (tag, pred, w), uuid in idx.supersede_head.items():
        if w != world:
            continue
        if tag not in ent_tags:
            continue
        if uuid in seen:
            continue
        e = idx.by_uuid.get(uuid)
        if e is None:
            continue
        seen.add(uuid)
        heads.append(e)
    heads.sort(key=lambda e: e.ts, reverse=True)
    return heads[:max_active_state_size]


def render_active_state(heads: list[LogEntry]) -> str:
    if not heads:
        return "(no prior chain heads in this world for entities in this batch)"
    by_entity: dict[str, list[LogEntry]] = {}
    for e in heads:
        if not e.predicate:
            continue
        m = re.match(r"(@?[A-Za-z0-9_]+)\.(.+)", e.predicate)
        if not m:
            continue
        ent = m.group(1)
        if not ent.startswith("@"):
            ent = "@" + ent
        by_entity.setdefault(ent, []).append(e)
    lines = []
    for ent in sorted(by_entity):
        lines.append(f"{ent}:")
        for e in sorted(by_entity[ent], key=lambda x: x.predicate or ""):
            pred = (e.predicate or "").split(".", 1)[-1]
            snippet = e.text.strip()
            if len(snippet) > 110:
                snippet = snippet[:107] + "..."
            lines.append(f"  - {pred}: {snippet!r} ({e.uuid})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

WRITE_PROMPT_3 = """You are a semantic-memory writer using a SINGLE APPEND-ONLY LOG.
Each entry you write is an atomic natural-language fact. Use @Name tags for
entities. Use `refs` to point at prior entries when this entry updates,
corrects, or refines them.

WORLD CONTEXT
This batch belongs to the world: "{world}"
  - "real"         : actual facts about the user. Persist them.
  - "hypothetical" : speculation. Tag entries to this world; do NOT update
                     real-world chains.
  - "fiction"      : role-play, novel, game, daydream, sarcastic untruth. Treat
                     its entities as living ONLY in this world. Real-world
                     entities like the user's actual boss should NOT be
                     referenced here even if a name collides (e.g. fictional
                     Marcus the cartographer vs real boss Marcus are distinct
                     entities living in distinct worlds).

If a single batch contains BOTH real-world content (e.g. "my editor texted me")
and non-real content (e.g. "the protagonist meets a wizard"), emit SEPARATE
entries — one per world — and tag them accordingly.

REFS: list of prior entry UUIDs. Single ref type. Update/refine chains only
within the same world.

PREDICATE (optional, recommended for state-tracking facts in any world):
  Format: "@Entity.predicate_name". Example "@User.employer", "@Marcus.role".

KNOWN ENTITIES so far: {known_entities}

ACTIVE STATE in world "{world}" for entities in this batch (each line is the
current chain head; if your new entry updates one, include its uuid in `refs`):
{active_state}

PRIOR LOG SAMPLE (most recent entries; world tags shown):
{prior_log}

BATCH OF TURNS (process as a unit; emit 0+ entries):
{turn_block}

Emit JSON:
{{
  "entries": [
    {{
      "text": "<atomic fact>",
      "mentions": ["@Name", ...],
      "refs": ["<prior-uuid>", ...],
      "predicate": "@Entity.pred" or null,
      "world": "real" | "hypothetical" | "fiction"
    }}
  ]
}}

RULES
- Default the entry's "world" to "{world}". Only override if the specific turn
  is clearly in a different world (e.g. user says "btw irl my partner Jamie..."
  during a fiction batch — that single fact is "real").
- For state facts in non-real worlds, still emit predicates so chains work
  inside that world.
- Skip pure filler. Output {{"entries": []}} if nothing memory-worthy.
- Output JSON ONLY.
"""


WRITE_PROMPT_2 = """You are a semantic-memory writer using a SINGLE APPEND-ONLY LOG.
Each entry you write is an atomic natural-language fact. Use @Name tags for
entities. Use `refs` to point at prior entries when this entry updates,
corrects, or refines them.

WORLD CONTEXT
This batch belongs to the world: "{world}"
  - "real"     : actual facts about the user. Persist them.
  - "non_real" : speculation, role-play, novel, game, daydream, joke. Tag
                 entries to this world; do NOT update real-world chains.
                 Treat its entities as living ONLY in this world. Real-world
                 entities like the user's actual boss should NOT be referenced
                 here even if a name collides.

If a single batch contains BOTH real-world content (e.g. "my editor texted me")
and non-real content (e.g. "the protagonist meets a wizard"), emit SEPARATE
entries — one per world — and tag them accordingly.

REFS: list of prior entry UUIDs. Single ref type. Update/refine chains only
within the same world.

PREDICATE (optional, recommended for state-tracking facts in any world):
  Format: "@Entity.predicate_name". Example "@User.employer", "@Marcus.role".

KNOWN ENTITIES so far: {known_entities}

ACTIVE STATE in world "{world}" for entities in this batch (each line is the
current chain head; if your new entry updates one, include its uuid in `refs`):
{active_state}

PRIOR LOG SAMPLE (most recent entries; world tags shown):
{prior_log}

BATCH OF TURNS (process as a unit; emit 0+ entries):
{turn_block}

Emit JSON:
{{
  "entries": [
    {{
      "text": "<atomic fact>",
      "mentions": ["@Name", ...],
      "refs": ["<prior-uuid>", ...],
      "predicate": "@Entity.pred" or null,
      "world": "real" | "non_real"
    }}
  ]
}}

RULES
- Default the entry's "world" to "{world}". Only override if the specific turn
  is clearly in a different world (e.g. user says "btw irl my partner Jamie..."
  during a non_real batch — that single fact is "real").
- For state facts in non_real, still emit predicates so chains work inside
  that world.
- Skip pure filler. Output {{"entries": []}} if nothing memory-worthy.
- Output JSON ONLY.
"""


def _render_prior_log(prior_entries: list[LogEntry], max_recent: int = 12) -> str:
    recent = list(reversed(prior_entries[-max_recent:]))
    if not recent:
        return "(empty)"
    lines = []
    for e in recent:
        ref_str = f" refs=[{','.join(e.refs)}]" if e.refs else ""
        pred_str = f" pred={e.predicate}" if e.predicate else ""
        lines.append(
            f"[{e.uuid}] t{e.ts} world={e.world} mentions={','.join(e.mentions)} "
            f":: {e.text}{ref_str}{pred_str}"
        )
    return "\n".join(lines)


def write_batch(
    batch_turns: list[tuple[int, str]],
    prior_entries: list[LogEntry],
    idx: IndexedLog | None,
    known_entities: set[str],
    world: str,
    cache: Cache,
    budget: Budget,
    n_cats: int,
    max_active_state_size: int = 100,
) -> tuple[list[LogEntry], dict]:
    entities = aen1_active.extract_batch_entities(batch_turns)
    heads = gather_active_state(idx, entities, world, max_active_state_size)
    active_state_str = render_active_state(heads)

    prior_log = _render_prior_log(prior_entries)
    turn_block = "\n".join(f"TURN {i}: {t}" for i, t in batch_turns)
    template = WRITE_PROMPT_3 if n_cats == 3 else WRITE_PROMPT_2
    prompt = template.format(
        world=world,
        known_entities=", ".join(sorted(known_entities))
        if known_entities
        else "(none)",
        active_state=active_state_str,
        prior_log=prior_log,
        turn_block=turn_block,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    telemetry = {
        "world": world,
        "n_active_state_heads": len(heads),
        "active_state_chars": len(active_state_str),
        "prompt_chars": len(prompt),
        "batch_entities": sorted(entities),
    }
    if not isinstance(obj, dict):
        return [], telemetry
    entries_raw = obj.get("entries", []) or []
    entries: list[LogEntry] = []
    last_turn = batch_turns[-1][0] if batch_turns else 0
    for i, e in enumerate(entries_raw):
        if not isinstance(e, dict):
            continue
        text = (e.get("text") or "").strip()
        if not text:
            continue
        mentions = [m for m in (e.get("mentions") or []) if isinstance(m, str)]
        refs_raw = e.get("refs") or []
        refs = [r for r in refs_raw if isinstance(r, str)]
        predicate = e.get("predicate")
        if predicate is not None and not isinstance(predicate, str):
            predicate = None
        ent_world = e.get("world") or world
        if not isinstance(ent_world, str) or not ent_world.strip():
            ent_world = world
        ent_world = ent_world.strip().lower()
        if ent_world not in VALID_WORLDS[n_cats]:
            ent_world = world
        uuid = f"e{last_turn:04d}_{i}"
        entries.append(
            LogEntry(
                uuid=uuid,
                ts=last_turn,
                text=text,
                mentions=mentions,
                refs=refs,
                predicate=predicate,
                world=ent_world,
            )
        )
    return entries, telemetry


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def ingest_turns(
    turns: list[tuple[int, str]],
    cache: Cache,
    budget: Budget,
    n_cats: int,
    batch_size: int = 5,
    rebuild_index_every: int = 4,
    max_active_state_size: int = 100,
) -> tuple[list[LogEntry], IndexedLog, list[dict]]:
    log: list[LogEntry] = []
    known: set[str] = {"User"}
    current_world = "real"
    idx: IndexedLog | None = None
    telemetry: list[dict] = []

    for batch_no, i in enumerate(range(0, len(turns), batch_size)):
        batch = turns[i : i + batch_size]
        world, reason = classify_world(
            batch,
            current_world,
            cache,
            budget,
            n_cats=n_cats,
        )
        current_world = world

        new_entries, tele = write_batch(
            batch,
            log,
            idx,
            known,
            world,
            cache,
            budget,
            n_cats=n_cats,
            max_active_state_size=max_active_state_size,
        )
        for e in new_entries:
            for m in e.mentions:
                if m.startswith("@"):
                    known.add(m[1:])
        log.extend(new_entries)
        tele["batch_no"] = batch_no
        tele["last_turn"] = batch[-1][0] if batch else None
        tele["n_emitted"] = len(new_entries)
        tele["classified_world"] = world
        tele["classify_reason"] = reason
        telemetry.append(tele)
        if batch_no % rebuild_index_every == 0:
            idx = build_index(log, cache, budget)
    idx = build_index(log, cache, budget)
    return log, idx, telemetry


# ---------------------------------------------------------------------------
# Retrieval — world-filtered
# ---------------------------------------------------------------------------


def detect_question_world(question: str, n_cats: int) -> str:
    q = question.lower()
    if any(
        p in q
        for p in [
            "in real life",
            "real-life",
            "irl",
            "actually live",
            "actually like",
            "really live",
            "actually work",
            "actual",
            "really work",
        ]
    ):
        return "real"
    if any(
        p in q
        for p in [
            "what if",
            "imagine",
            "hypothetical",
            "if user",
            "if the user",
            "if they",
            "if i ",
        ]
    ):
        return "hypothetical" if n_cats == 3 else "non_real"
    if any(
        p in q
        for p in [
            "novel",
            "story",
            "the protagonist",
            "in the role-play",
            "roleplay",
            "role-play",
            "in the d&d",
            "in the dnd",
            "in the game",
            "in the dragon",
            "the dragon",
            "the dnd",
            "d&d game",
            "antagonist",
        ]
    ):
        return "fiction" if n_cats == 3 else "non_real"
    return "real"


def retrieve(
    question: str,
    idx: IndexedLog,
    cache: Cache,
    budget: Budget,
    n_cats: int,
    top_k: int = 12,
    world: str | None = None,
) -> list[LogEntry]:
    if not idx.entries:
        return []
    if world is None:
        world = detect_question_world(question, n_cats)

    sub_entries = [e for e in idx.entries if e.world == world]
    sub_uuid = {e.uuid for e in sub_entries}
    sub_by_uuid = {u: idx.by_uuid[u] for u in sub_uuid}
    sub_mention: dict[str, list[str]] = {}
    for tag, uuids in idx.mention_index.items():
        kept = [u for u in uuids if u in sub_uuid]
        if kept:
            sub_mention[tag] = kept
    sub_superseded_by = {
        u: v for u, v in idx.superseded_by.items() if u in sub_uuid and v in sub_uuid
    }
    sub_head: dict[tuple[str, str], str] = {}
    for (ent, pred, w), uuid in idx.supersede_head.items():
        if w == world and uuid in sub_uuid:
            sub_head[(ent, pred)] = uuid
    sub_emb = {u: idx.embed_by_uuid[u] for u in sub_uuid if u in idx.embed_by_uuid}

    sub_idx = aen1_simple.IndexedLog(
        entries=sub_entries,
        by_uuid=sub_by_uuid,
        mention_index=sub_mention,
        superseded_by=sub_superseded_by,
        supersede_head=sub_head,
        embed_by_uuid=sub_emb,
    )
    return aen1_simple.retrieve(question, sub_idx, cache, budget, top_k=top_k)


READ_PROMPT = """You are answering a question about User's life using a
semantic-memory log scoped to the world "{world}".

If the world is "real", answer as if these are actual facts about the user.
If the world is non-real (hypothetical/fiction/non_real), the answer is about
that world, not about the user's actual life.

Each entry is an atomic natural-language fact. `refs` points to prior entries
in the same world.

RETRIEVED ENTRIES (chronological, all in world "{world}"):
{entries_block}

QUESTION: {question}

Answer concisely. If no relevant entries exist in this world, say "I don't have
any facts in this world about that." For yes/no questions, start with Yes or No.
"""


def format_entries(entries: list[LogEntry]) -> str:
    lines = []
    for e in entries:
        ref_str = f" refs=[{','.join(e.refs)}]" if e.refs else ""
        mentions = ",".join(e.mentions)
        lines.append(
            f"[{e.uuid}] t{e.ts} world={e.world} {mentions} :: {e.text}{ref_str}"
        )
    return "\n".join(lines)


def answer_question(
    question: str,
    idx: IndexedLog,
    cache: Cache,
    budget: Budget,
    n_cats: int,
    top_k: int = 12,
    world: str | None = None,
) -> str:
    if world is None:
        world = detect_question_world(question, n_cats)
    retrieved = retrieve(
        question,
        idx,
        cache,
        budget,
        n_cats=n_cats,
        top_k=top_k,
        world=world,
    )
    block = format_entries(retrieved)
    prompt = READ_PROMPT.format(
        world=world,
        entries_block=block,
        question=question,
    )
    return llm(prompt, cache, budget).strip()


def answer_question_no_world(
    question: str,
    idx: IndexedLog,
    cache: Cache,
    budget: Budget,
    top_k: int = 12,
) -> str:
    flat_head: dict[tuple[str, str], str] = {}
    for (ent, pred, w), uuid in idx.supersede_head.items():
        cur = flat_head.get((ent, pred))
        if cur is None:
            flat_head[(ent, pred)] = uuid
        else:
            if idx.by_uuid[uuid].ts > idx.by_uuid[cur].ts:
                flat_head[(ent, pred)] = uuid
    sub_idx = aen1_simple.IndexedLog(
        entries=list(idx.entries),
        by_uuid=dict(idx.by_uuid),
        mention_index=dict(idx.mention_index),
        superseded_by=dict(idx.superseded_by),
        supersede_head=flat_head,
        embed_by_uuid=dict(idx.embed_by_uuid),
    )
    retrieved = aen1_simple.retrieve(question, sub_idx, cache, budget, top_k=top_k)
    block = "\n".join(
        f"[{e.uuid}] t{e.ts} {','.join(e.mentions)} :: {e.text}"
        + (f" refs=[{','.join(e.refs)}]" if e.refs else "")
        for e in retrieved
    )
    prompt = (
        "You are answering a question using a flat memory log (no world "
        "scoping). Treat all entries as the same world.\n\n"
        f"RETRIEVED ENTRIES:\n{block}\n\nQUESTION: {question}\n\n"
        "Answer concisely. For yes/no questions, start with Yes or No."
    )
    return llm(prompt, cache, budget).strip()
