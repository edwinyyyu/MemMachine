"""AEN-1 ACTIVE v3 — predicate-dictation writer prompt + deterministic relink.

v2 (post-hoc deterministic relink alone) lifted ref-correctness from 0.686 to
0.721. The remaining ~25% of errors are dominated by chain-pollution: the
writer assigns the same state-tracking predicate (e.g. `@User.team`) to
clarify/no-change entries, so the deterministic relink links to the wrong
prior. The clarify text-similarity heuristic doesn't catch them because
clarify entries can mention old values verbatim ("I'm on the platform side,
not applied") without low jaccard.

v3 fixes the source: a stronger writer prompt that explicitly forbids
state-tracking predicates on non-state-change entries. We then apply the
same deterministic relink as v2 on top.

Cost: requires re-running the writer (cache miss). ~149 writer calls + QA
+ judge ~ ~200 LLM calls for the variant. Still under budget at $3 hard cap.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND16C = HERE.parent
RESEARCH = ROUND16C.parent
ROUND15 = RESEARCH / "round15_active_chains"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_active  # noqa: E402
import aen1_active_v2  # noqa: E402  (relinker)
import aen1_simple  # noqa: E402
from _common import extract_json, llm  # noqa: E402

LogEntry = aen1_simple.LogEntry
IndexedLog = aen1_simple.IndexedLog
build_index = aen1_simple.build_index
retrieve = aen1_simple.retrieve
answer_question = aen1_simple.answer_question


# ---------------------------------------------------------------------------
# Writer prompt — explicit predicate-discipline + ref-discipline rules
# ---------------------------------------------------------------------------

WRITE_PROMPT = """You are a semantic-memory writer using a SINGLE APPEND-ONLY LOG.

Each entry you write is an atomic natural-language fact. Use @Name tags to
mention entities. Use `refs` to point at prior entries when this entry
updates/corrects/refines/clarifies a prior fact.

PREDICATES — strict discipline (read carefully):
  Format: "@Entity.predicate_name", e.g. "@User.title", "@Jamie.job",
  "@User.location".
  ONLY emit a predicate when the entry asserts a NEW VALUE for that
  predicate (a state CHANGE). Examples:
    - "I'm a senior engineer now" -> predicate "@User.title" (new value).
    - "Half of my role is mentoring" -> predicate=null (re-stating an
      existing role, not a new title).
    - "Marcus runs tight 1:1s" -> predicate=null (detail about Marcus,
      not a new value for any chain).
  When in doubt about whether the entry changes the chain's value:
  predicate=null.
  REUSE the canonical predicate name once a chain has one. If the ACTIVE
  STATE block shows "@User.title", new title-changes MUST use "@User.title"
  too (not "@User.role" or "@User.occupation").

REFS — strict discipline:
  If your new entry asserts a NEW VALUE for a predicate that already exists
  in the ACTIVE STATE block below, set `refs` to the active-state uuid for
  that EXACT (entity, predicate). One ref, one chain link.
  If your new entry is just clarifying/restating without a new value: you
  may either omit refs or include the relevant prior uuid; either way set
  predicate=null so the chain bookkeeping stays clean.

KNOWN ENTITIES so far: {known_entities}

ACTIVE STATE OF ENTITIES IN THIS BATCH (one line per (entity, predicate),
showing the current chain head's uuid and its value-text):
{active_state}

PRIOR LOG SAMPLE (most recent entries):
{prior_log}

BATCH OF TURNS (process as a unit; emit 0+ entries covering the whole batch):
{turn_block}

Emit JSON:
{{
  "entries": [
    {{
      "text": "<atomic fact in one sentence>",
      "mentions": ["@Name", ...],
      "refs": ["<prior-uuid>", ...],
      "predicate": "@Entity.pred" or null
    }}
  ]
}}

RULES (compact)
- @Name every named entity. ALWAYS @User when the fact is about the speaker.
- Filler/chitchat -> emit no entry.
- predicate=null unless this entry asserts a new value for a state-tracking
  chain. (See PREDICATES discipline above.)
- refs match active-state ONLY when the entry has a predicate AND the
  predicate matches an active-state entry's predicate exactly.
- Do NOT invent entities. Output JSON ONLY.
"""


def write_batch(
    batch_turns,
    prior_entries,
    idx,
    known_entities,
    cache,
    budget,
    max_active_state_size: int = 100,
):
    """Same shape as aen1_active.write_batch, but uses the v3 prompt."""
    entities = aen1_active.extract_batch_entities(batch_turns)
    heads = aen1_active.gather_active_state(idx, entities, max_active_state_size)
    active_state_str = aen1_active.render_active_state(heads)

    prior_log = aen1_active._render_prior_log(prior_entries)
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
    turns,
    cache,
    budget,
    batch_size: int = 5,
    rebuild_index_every: int = 4,
    max_active_state_size: int = 100,
    skip_clarify: bool = True,
    normalize: bool = True,
):
    """v3 ingest: uses v3 writer prompt, then deterministic relink at the end."""
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
    relinked = aen1_active_v2.deterministic_relink(
        log,
        skip_clarify=skip_clarify,
        normalize=normalize,
    )
    idx = build_index(relinked, cache, budget)
    return relinked, idx, telemetry
