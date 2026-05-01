"""AEN-1 ACTIVE — at-write-time fix for ref-emission collapse.

Round 14 showed that aen1_simple's ref-emission rate collapses from 62% (in the
first 100 turns) to ~17-42% in the (200,800] tail. Mechanism: as the log grows,
chain heads from quiet predicates fall out of the writer's prompt window. The
writer can't ref what it can't see.

Fix (no retroactive passes): on every batch, identify the entities mentioned
in the incoming turns and inject their active chain heads — looked up directly
from the structural index `supersede_head` — into the writer's prompt as an
"ACTIVE STATE" block. The writer is instructed to emit `refs: [<uuid>]`
pointing at any of those active states it updates.

Reuses:
  - aen1_simple's data model (LogEntry, IndexedLog)
  - aen1_simple's build_index (structural index)
  - aen1_simple's retrieval + answer_question (reader)

Overrides:
  - write_batch  -> includes ACTIVE STATE block of chain heads
  - ingest_turns -> rebuilds the index more frequently so the active-state
    snapshot stays fresh (this is purely structural — no LLM cost)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND15 = HERE.parent
RESEARCH = ROUND15.parent
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_simple  # noqa: E402
from _common import Budget, Cache, extract_json, llm  # noqa: E402

# Re-export so callers can use `aen1_active.LogEntry`, `aen1_active.build_index`,
# etc. without reaching back into aen1_simple themselves.
LogEntry = aen1_simple.LogEntry
IndexedLog = aen1_simple.IndexedLog
build_index = aen1_simple.build_index
retrieve = aen1_simple.retrieve
answer_question = aen1_simple.answer_question


# ---------------------------------------------------------------------------
# Entity extraction from a batch
# ---------------------------------------------------------------------------

# Stop-word set for cheap proper-noun extraction (same flavor as aen1_simple's
# question entity extractor).
_BATCH_STOP = {
    "I",
    "Im",
    "Ill",
    "Ive",
    "Id",
    "OK",
    "Okay",
    "Yes",
    "No",
    "Hi",
    "Hey",
    "Hello",
    "Just",
    "Got",
    "Big",
    "Quit",
    "Started",
    "Switched",
    "Working",
    "Slack",
    "Email",
    "Coffee",
    "Weather",
    "Traffic",
    "Tired",
    "Should",
    "Random",
    "Listening",
    "Watching",
    "Cleaning",
    "Inbox",
    "Garbage",
    "Cold",
    "Long",
    "Pretty",
    "Dropped",
    "Picked",
    "Rainy",
    "Going",
    "Been",
    "Stomach",
    "Half",
    "Lots",
    "Most",
    "My",
    "Me",
    "We",
    "Us",
    "Our",
    "Mid",
    "Found",
}


def extract_batch_entities(batch_turns: list[tuple[int, str]]) -> set[str]:
    """Extract @-tags-likely-to-appear from raw batch text.

    Cheap rule: any TitleCased word not in the stop set, plus always
    "User" if first-person markers are present (I, me, my, we, our).
    """
    text = " ".join(t for _, t in batch_turns)
    words = re.findall(r"\b([A-Z][a-zA-Z]{1,20})\b", text)
    ents = {w for w in words if w not in _BATCH_STOP}
    # First-person -> @User
    if re.search(r"\b(I|I'm|I've|I'll|me|my|we|our|us)\b", text, re.IGNORECASE):
        ents.add("User")
    # Always include User as a default subject for state-tracking
    ents.add("User")
    return ents


# ---------------------------------------------------------------------------
# Active-state rendering
# ---------------------------------------------------------------------------


def gather_active_state(
    idx: IndexedLog | None,
    entities: set[str],
    max_active_state_size: int,
) -> list[LogEntry]:
    """Look up the chain heads for every (entity, predicate) where entity is
    in the batch's entity set. Returns the head LogEntry objects, capped at
    `max_active_state_size`, sorted by ts descending (most-recent first so
    truncation drops the oldest heads first)."""
    if idx is None:
        return []
    ent_tags = {f"@{e}" for e in entities}
    heads: list[LogEntry] = []
    seen = set()
    for (tag, pred), uuid in idx.supersede_head.items():
        if tag not in ent_tags:
            continue
        if uuid in seen:
            continue
        e = idx.by_uuid.get(uuid)
        if e is None:
            continue
        seen.add(uuid)
        heads.append(e)
    # Most recent first; cap.
    heads.sort(key=lambda e: e.ts, reverse=True)
    return heads[:max_active_state_size]


def render_active_state(heads: list[LogEntry]) -> str:
    """Render as a compact 'ACTIVE STATE OF ENTITIES IN THIS BATCH' block,
    grouped by entity."""
    if not heads:
        return "(no prior chain heads found for entities in this batch)"
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
            # Try to surface the value; fall back to the entry text.
            snippet = e.text.strip()
            if len(snippet) > 110:
                snippet = snippet[:107] + "..."
            lines.append(f"  - {pred}: {snippet!r} ({e.uuid})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Writer prompt (extends aen1_simple's prompt with ACTIVE STATE block)
# ---------------------------------------------------------------------------

WRITE_PROMPT = """You are a semantic-memory writer using a SINGLE APPEND-ONLY LOG.

Each entry you write is an atomic natural-language fact. Use @Name tags to
mention entities. Use `refs` to point at prior entries when this entry relates
to, updates, or corrects a prior fact.

REFS: a list of prior entry UUIDs - there is only ONE kind of ref. If this
entry updates/corrects/refines/clarifies a prior fact, include that prior entry
in `refs`. The prose text carries the nuance (replacement, retraction, detail,
etc.) - do NOT choose a relation label.

PREDICATE (optional but recommended for state-tracking facts):
  Format: "@Entity.predicate_name" - for example "@User.employer",
  "@Jamie.job", "@User.location". Omit for casual or multi-subject facts.

KNOWN ENTITIES so far: {known_entities}

ACTIVE STATE OF ENTITIES IN THIS BATCH (each line is the CURRENT chain head
for that (entity, predicate) - if your new entry updates/corrects/replaces one
of these states, you MUST include its uuid in `refs` so the chain stays linked.
Do NOT emit a ref if the new entry only mentions the entity casually or adds
unrelated detail - refs are for chain transitions only):
{active_state}

PRIOR LOG SAMPLE (most recent entries; cite older ones by uuid in `refs`):
{prior_log}

BATCH OF TURNS (process as a unit; emit 0+ entries covering the whole batch):
{turn_block}

Emit JSON:
{{
  "entries": [
    {{
      "text": "<atomic fact in one sentence, prose carries correction/supersede/nuance>",
      "mentions": ["@Name", ...],
      "refs": ["<prior-uuid>", ...],
      "predicate": "@Entity.pred" or null
    }}
  ]
}}

RULES
- Use @Name for every named entity (User, Jamie, Marcus, Luna, etc.). ALWAYS
  @User when the fact is about the speaker.
- If a turn is pure filler (weather, chitchat, jokes, noop), do NOT emit an
  entry for it. Skip silently.
- If a turn UPDATES or CORRECTS a prior fact (e.g. boss changed, moved cities,
  job changed), emit a new entry with `refs` pointing at the matching ACTIVE
  STATE entry's uuid above. Make the prose carry the nuance.
- For state-tracking facts (job, location, boss, employer, role, relationship,
  residence, partner, hobby, commute, car, gym, team, title, etc.), include
  the `predicate` in the form "@Entity.pred_name" using stable lowercase
  predicate names. REUSE the predicate names already shown in the ACTIVE STATE
  block when the new fact updates one of those chains (e.g. if the active
  state shows "@User.boss", a new manager update MUST also use "@User.boss",
  not "@User.manager").
- If the turn adds pure detail about a prior fact without contradicting it,
  include the prior uuid in `refs` but the prose can just state the new
  detail.
- Prefer ONE entry per turn; emit multiple only if the turn bundles unrelated
  facts. If the batch has no memory-worthy content, output {{"entries": []}}.
- Do NOT invent entities. Do NOT add @User to facts where User isn't mentioned
  or implied.
- Output JSON ONLY.
"""


def _render_prior_log(prior_entries: list[LogEntry], max_recent: int = 12) -> str:
    """Recent-tail prior log - same flavor as aen1_simple but without the
    relevant_heads block (active_state replaces that role)."""
    recent = list(reversed(prior_entries[-max_recent:]))
    if not recent:
        return "(empty)"
    lines = []
    for e in recent:
        ref_str = f" refs=[{','.join(e.refs)}]" if e.refs else ""
        pred_str = f" pred={e.predicate}" if e.predicate else ""
        lines.append(
            f"[{e.uuid}] t{e.ts} mentions={','.join(e.mentions)} "
            f":: {e.text}{ref_str}{pred_str}"
        )
    return "\n".join(lines)


def write_batch(
    batch_turns: list[tuple[int, str]],
    prior_entries: list[LogEntry],
    idx: IndexedLog | None,
    known_entities: set[str],
    cache: Cache,
    budget: Budget,
    max_active_state_size: int = 100,
) -> tuple[list[LogEntry], dict]:
    """Write a batch with ACTIVE STATE injection.

    Returns (entries, telemetry_dict) where telemetry contains
    `n_active_state_heads` and `active_state_chars` so we can compute
    overhead.
    """
    entities = extract_batch_entities(batch_turns)
    heads = gather_active_state(idx, entities, max_active_state_size)
    active_state_str = render_active_state(heads)

    prior_log = _render_prior_log(prior_entries)
    turn_block = "\n".join(f"TURN {i}: {t}" for i, t in batch_turns)
    prompt = WRITE_PROMPT.format(
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
        uuid = f"e{last_turn:04d}_{i}"
        entries.append(
            LogEntry(
                uuid=uuid,
                ts=last_turn,
                text=text,
                mentions=mentions,
                refs=refs,
                predicate=predicate,
            )
        )
    return entries, telemetry


def ingest_turns(
    turns: list[tuple[int, str]],
    cache: Cache,
    budget: Budget,
    batch_size: int = 5,
    rebuild_index_every: int = 4,
    max_active_state_size: int = 100,
) -> tuple[list[LogEntry], IndexedLog, list[dict]]:
    """Ingest with at-write-time active-state injection.

    rebuild_index_every: lower than aen1_simple (4 batches = ~20 turns) so
    the active-state snapshot stays current. Index rebuilds are purely
    structural - no LLM cost; only embeddings cost, and embed_batch caches
    by text so repeats are free.

    Returns (log, idx, telemetry_per_batch).
    """
    log: list[LogEntry] = []
    known: set[str] = {"User"}
    idx: IndexedLog | None = None
    telemetry: list[dict] = []

    for batch_no, i in enumerate(range(0, len(turns), batch_size)):
        batch = turns[i : i + batch_size]
        new_entries, tele = write_batch(
            batch,
            log,
            idx,
            known,
            cache,
            budget,
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
        telemetry.append(tele)
        if batch_no % rebuild_index_every == 0:
            idx = build_index(log, cache, budget)
    idx = build_index(log, cache, budget)
    return log, idx, telemetry
