"""AEN-1 WORLDS — world scoping for semantic memory.

Extends aen1_active with a `world` field on every entry. Entries default to
`world="real"`; a small per-batch classifier flips them to fictional/
hypothetical/joke contexts when the writer detects strong cues.

Why this matters: a real chat user might say "I'm a dragon" in role-play, or
"imagine I lived in Paris", or "in my novel the protagonist is named Marcus".
Without world scoping, those facts pollute the "real" view of the user.

Design

  - World is a string. Reserved values: "real", "hypothetical", "joke".
    Fictional worlds carry a slug: "fiction:vampire_novel", "game:dnd_campaign".
  - Each batch is classified ONCE by an LLM call. Worlds are STICKY across
    turns: once we enter "fiction:novel_a", subsequent batches stay in that
    world until a return-cue ("ok back to real life", "anyway, in reality")
    or a clear topic switch.
  - The writer prompt tells the LLM what world this batch is in and instructs
    it to tag every emitted entry with that world.
  - The active-state injection is filtered by world: when writing real-life
    facts we don't show fictional Marcus; when writing novel facts we don't
    show real boss-Marcus.
  - Retrieval is filtered by the question's world (default: "real"). Q/A
    questions can ask about specific worlds with phrases like "in the novel"
    / "in real life".

This file overrides write_batch and ingest_turns from aen1_active and adds
classify_world() + world-scoped retrieve()/answer_question() helpers.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND16B = HERE.parent
RESEARCH = ROUND16B.parent
ROUND15 = RESEARCH / "round15_active_chains"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_active  # noqa: E402
import aen1_simple  # noqa: E402
from _common import Budget, Cache, embed_batch, extract_json, llm  # noqa: E402

# ---------------------------------------------------------------------------
# Data model — extends aen1_simple.LogEntry with `world`
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
    # Now keyed by (entity, predicate, world) so chains in different worlds
    # don't collide (real Marcus vs novel Marcus).
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
# World classifier — one LLM call per batch
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """You decide what "world" a chunk of conversation belongs to,
so that fictional/hypothetical content is not stored as real facts about the
user.

WORLDS:
  - "real"           — actual facts about the user's life and people they know
  - "hypothetical"   — speculation, "what if I moved to Paris", "imagine I lived..."
                       (NOT actual plans — those are real)
  - "joke"           — sarcasm, jokes, obvious untruths the user wouldn't claim
  - "fiction:<slug>" — role-play, a novel, a game, a daydream, a story the user
                       is writing or playing in. Slug is a short stable label
                       like "fantasy_rp", "vampire_novel", "dnd_campaign".

Strong cues to flip to non-real worlds:
  - "in my novel/story/game", "imagine", "what if", "if I were", "let's pretend",
    "role-play", "[character: X]", "as a dragon", "you're the dungeon master"

Cues to flip BACK to real:
  - "ok anyway, in real life", "back to reality", "for real though",
    "irl", "actually", a clear topic change to mundane life (work, partner,
    pets, errands)

PERSISTENCE: once a fictional/hypothetical world is established, stay in it
until the user signals a return. The CURRENT WORLD below is what the previous
batch was tagged. Lean toward keeping it unless cues say to switch.

CURRENT WORLD: {current_world}
KNOWN WORLDS: {known_worlds}

BATCH:
{turn_block}

Output JSON only:
{{
  "world": "<world name>",
  "reason": "<one short sentence>"
}}

Pick from known worlds when continuing one; introduce a new fiction:<slug> only
when a clearly distinct fictional world starts. Use "real" by default.
"""


def classify_world(
    batch_turns: list[tuple[int, str]],
    current_world: str,
    known_worlds: set[str],
    cache: Cache,
    budget: Budget,
) -> tuple[str, str]:
    """Return (world, reason). One LLM call per batch."""
    turn_block = "\n".join(f"TURN {i}: {t}" for i, t in batch_turns)
    prompt = CLASSIFY_PROMPT.format(
        current_world=current_world,
        known_worlds=", ".join(sorted(known_worlds)) or "real",
        turn_block=turn_block,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return current_world, "classifier_parse_error"
    world = obj.get("world")
    if not isinstance(world, str) or not world.strip():
        return current_world, "classifier_empty_world"
    world = world.strip()
    # Light normalization
    if world.lower() in ("real", "real_life", "reality"):
        world = "real"
    elif world.lower() in ("hypothetical", "hypothesis", "what_if"):
        world = "hypothetical"
    elif world.lower() in ("joke", "sarcasm"):
        world = "joke"
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
# Writer — same as aen1_active but world-aware
# ---------------------------------------------------------------------------

WRITE_PROMPT = """You are a semantic-memory writer using a SINGLE APPEND-ONLY LOG.
Each entry you write is an atomic natural-language fact. Use @Name tags for
entities. Use `refs` to point at prior entries when this entry updates,
corrects, or refines them.

WORLD CONTEXT
This batch belongs to the world: "{world}"
  - "real"          : actual facts about the user. Persist them.
  - "hypothetical"  : speculation. Tag entries to this world; do NOT update
                      real-world chains.
  - "joke"          : sarcasm/jokes. Usually emit nothing; only emit if the
                      content is genuinely worth remembering as a joke.
  - "fiction:<slug>": role-play, novel, game. Treat its entities as living
                      ONLY in this world. Real-world entities like the user's
                      actual boss should NOT be referenced here even if a name
                      collides (e.g. fictional Marcus vs real Marcus are
                      distinct entities living in distinct worlds).

If a single batch contains BOTH real-world content (e.g. "my editor texted
me") and fiction-world content (e.g. "the protagonist meets a wizard"), emit
SEPARATE entries — one per world — and tag them accordingly.

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
      "world": "<world for this entry — usually the batch world; override only if
                the turn explicitly steps into another world>"
    }}
  ]
}}

RULES
- Default the entry's "world" to "{world}". Only override if the specific turn
  is clearly in a different world (e.g. user says "btw irl my partner Jamie..."
  during a fiction batch — that single fact is "real").
- For state facts in fiction/role-play, still emit predicates so chains work
  inside the fictional world.
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
    max_active_state_size: int = 100,
) -> tuple[list[LogEntry], dict]:
    entities = aen1_active.extract_batch_entities(batch_turns)
    heads = gather_active_state(idx, entities, world, max_active_state_size)
    active_state_str = render_active_state(heads)

    prior_log = _render_prior_log(prior_entries)
    turn_block = "\n".join(f"TURN {i}: {t}" for i, t in batch_turns)
    prompt = WRITE_PROMPT.format(
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
        ent_world = ent_world.strip()
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
# Ingest — orchestrates classify + write
# ---------------------------------------------------------------------------


def ingest_turns(
    turns: list[tuple[int, str]],
    cache: Cache,
    budget: Budget,
    batch_size: int = 5,
    rebuild_index_every: int = 4,
    max_active_state_size: int = 100,
) -> tuple[list[LogEntry], IndexedLog, list[dict]]:
    log: list[LogEntry] = []
    known: set[str] = {"User"}
    known_worlds: set[str] = {"real"}
    current_world = "real"
    idx: IndexedLog | None = None
    telemetry: list[dict] = []

    for batch_no, i in enumerate(range(0, len(turns), batch_size)):
        batch = turns[i : i + batch_size]
        # Classify the world first
        world, reason = classify_world(
            batch,
            current_world,
            known_worlds,
            cache,
            budget,
        )
        known_worlds.add(world)
        current_world = world

        new_entries, tele = write_batch(
            batch,
            log,
            idx,
            known,
            world,
            cache,
            budget,
            max_active_state_size=max_active_state_size,
        )
        for e in new_entries:
            for m in e.mentions:
                if m.startswith("@"):
                    known.add(m[1:])
            known_worlds.add(e.world)
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


def detect_question_world(question: str, known_worlds: set[str]) -> str:
    """Detect which world a question is asking about. Default: real."""
    q = question.lower()
    # Explicit "real life" / "actually" / "irl"
    if any(
        p in q
        for p in [
            "in real life",
            "real-life",
            "irl",
            "actually live",
            "actually like",
            "really live",
        ]
    ):
        return "real"
    # Hypothetical
    if any(
        p in q
        for p in [
            "what if",
            "imagine",
            "hypothetical",
            "if user",
            "if the user",
        ]
    ):
        return "hypothetical"
    # Fiction cues
    for w in known_worlds:
        if w.startswith("fiction:") or w.startswith("game:"):
            slug = w.split(":", 1)[1]
            tokens = slug.replace("_", " ").lower()
            if tokens in q:
                return w
    if "novel" in q or "story" in q or "the protagonist" in q:
        for w in known_worlds:
            if w.startswith("fiction:"):
                return w
    if "role-play" in q or "roleplay" in q or "the dragon" in q:
        for w in known_worlds:
            if w.startswith("fiction:"):
                return w
    if "game" in q:
        for w in known_worlds:
            if w.startswith("game:"):
                return w
    return "real"


def retrieve(
    question: str,
    idx: IndexedLog,
    cache: Cache,
    budget: Budget,
    top_k: int = 12,
    world: str | None = None,
) -> list[LogEntry]:
    if not idx.entries:
        return []
    known_worlds = {e.world for e in idx.entries}
    if world is None:
        world = detect_question_world(question, known_worlds)

    # Build a world-filtered sub-index that mirrors aen1_simple.IndexedLog so
    # we can reuse aen1_simple.retrieve verbatim. Convert our 3-tuple
    # supersede_head to aen1_simple's 2-tuple keyed by (entity, predicate).
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

    # aen1_simple.LogEntry doesn't have a world field but it ignores extra
    # attrs since it only reads .uuid/.ts/.text/.mentions/.refs/.predicate.
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
If the world is fictional/hypothetical/joke, the answer is about that world,
not about the user's actual life.

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
    top_k: int = 12,
    world: str | None = None,
) -> str:
    if world is None:
        known_worlds = {e.world for e in idx.entries}
        world = detect_question_world(question, known_worlds)
    retrieved = retrieve(question, idx, cache, budget, top_k=top_k, world=world)
    block = format_entries(retrieved)
    prompt = READ_PROMPT.format(
        world=world,
        entries_block=block,
        question=question,
    )
    return llm(prompt, cache, budget).strip()


# Convenience: a "no-world" answer that pools all entries together (mimics the
# baseline aen1_active behavior — used for the with-vs-without comparison).
def answer_question_no_world(
    question: str,
    idx: IndexedLog,
    cache: Cache,
    budget: Budget,
    top_k: int = 12,
) -> str:
    # Build a flat aen1_simple-style index from all entries (ignoring world).
    flat_head: dict[tuple[str, str], str] = {}
    for (ent, pred, w), uuid in idx.supersede_head.items():
        # If a key already exists, keep the most recent entry's head
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
