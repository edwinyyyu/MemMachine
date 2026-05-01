"""R13b adapted with R22's general optimizations for fair architectural comparison.

Architectural delta vs R22:
  - R13b: persistent registry + per-turn coref pass + LRU active cache + lazy
    embedding pull. Writer sees @ent_<N> tags after coref rewriting.
  - R22: variable binding via cluster_id + canonical_label, no separate registry.

Generally-applicable optimizations ported from R22 (matched between both):
  1. Scheduler: K=3 centered window with w_past=w_future=7 (replacing R13b's
     5-turn non-overlapping batches).
  2. Writer prompt: R22's hard filler skip + durable predicate emphasis +
     window-CONTEXT-vs-TARGET separation.
  3. Active-state rendering: filtered to durable predicates only.

What stays R13b-specific:
  - PersistentRegistry, coref_turn (rewrites turn text with @ent_ tags),
    LRU active cache, lazy description-embedding pull.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND22 = HERE.parent
RESEARCH = ROUND22.parent
ROUND13 = RESEARCH / "round13_persistent_registry"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND13 / "aen3b"))
sys.path.insert(0, str(ROUND13 / "architectures"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_simple  # noqa: E402
import aen3b_persistent as r13b  # noqa: E402
from _common import extract_json, llm  # noqa: E402

LogEntry = aen1_simple.LogEntry
IndexedLog = aen1_simple.IndexedLog
build_index = aen1_simple.build_index


# Match R22's durable-predicate keyword set
DURABLE_PREDICATE_KEYWORDS = {
    "boss",
    "manager",
    "supervisor",
    "employer",
    "company",
    "workplace",
    "job",
    "occupation",
    "title",
    "role",
    "position",
    "team",
    "colleague",
    "coworker",
    "mentor",
    "advisor",
    "partner",
    "spouse",
    "fiance",
    "girlfriend",
    "boyfriend",
    "friend",
    "old_friend",
    "best_friend",
    "neighbor",
    "gym_buddy",
    "gym_friend",
    "senior",
    "junior",
    "location",
    "city",
    "country",
    "neighborhood",
    "residence",
    "home",
    "school",
    "university",
    "college",
    "hobby",
    "pet",
    "child",
    "parent",
    "sibling",
    "family",
    "name",
}


def is_durable_predicate(predicate):
    if not predicate:
        return False
    pl = predicate.lower()
    return any(kw in pl for kw in DURABLE_PREDICATE_KEYWORDS)


# R22's writer prompt, adapted: variable-binding-specific cluster_id field is
# REMOVED (R13b uses entity_id mentions instead). Filler skip + durable
# predicate emphasis kept.
WRITE_PROMPT = """You are a semantic-memory writer using a single append-only log.

Each entry is an atomic natural-language fact. Use @ent_<id> tags for entities
that have been resolved by an upstream coreference pass (you'll see them in
the conversation window). Use @User for the speaker. Add `predicate` for
state-tracking facts in the form "@ent_<id>.predicate_name" or
"@User.predicate_name".

ACTIVE STATE (current durable chain heads — only DURABLE predicates shown):
{active_state}

PRIOR LOG (most recent committed entries):
{prior_log}

CONVERSATION WINDOW
-------------------
{window_block}

Emit JSON for TARGET turns only:
{{
  "entries": [
    {{"turn": <int from TARGET turns>, "text": "...",
      "mentions": ["@ent_..."], "refs": ["<prior-uuid>"],
      "predicate": "@ent_X.role" or "@User.boss" or null}}
  ]
}}

ONLY EMIT FOR CHAIN-WORTHY EVENTS. These ARE chain-worthy:
  - New job, new boss, new team, new colleague mention
  - New location, new home, new school
  - New relationship, new friend, new mentor, new neighbor
  - New possession: car, bike, pet
  - Hobbies & recurring routines (started climbing, joined a gym, biking to work)
  - Confirmed plans/decisions
  - Naming an entity that was previously anonymous
  - Update/change to any of the above (job change, etc.)

DO NOT EMIT for these (skip silently):
  - Body sensations: "stomach hurts", "tired", "need a nap"
  - Weather: "weather is nice", "rainy day"
  - Transient feelings: "long day", "slow afternoon", "mellow morning"
  - Routine activities: "had coffee", "going for lunch", "on calls all morning"
  - Inbox/notification noise: "412 unread", "email avalanche"
  - Generic chitchat
  Output `{{"entries": []}}` if all TARGET turns are filler.

If a TARGET turn UPDATES or CORRECTS a prior fact, include the prior entry's
uuid in `refs` and let prose carry the supersede/correction nuance.

Output JSON ONLY.
"""


def render_active_state_durable(idx: IndexedLog | None, ent_tags: set[str]) -> str:
    """R22-style active-state render filtered to durable predicates."""
    if idx is None:
        return "(empty)"
    relevant = []
    for (tag, pred), uuid in idx.supersede_head.items():
        if not is_durable_predicate(pred):
            continue
        if tag in ent_tags or tag == "@User":
            relevant.append(((tag, pred), uuid))
    relevant.sort(key=lambda kv: idx.by_uuid[kv[1]].ts, reverse=True)
    lines = []
    for (tag, pred), uuid in relevant[:20]:
        e = idx.by_uuid[uuid]
        lines.append(f"  {tag}.{pred}: head=[{uuid} t={e.ts}] :: {e.text[:80]}")
    return "\n".join(lines) if lines else "(none)"


def render_window(window_turns, target_turn_lo):
    lines = []
    in_target = False
    for tidx, text in window_turns:
        if not in_target and tidx >= target_turn_lo:
            lines.append("--- TARGET TURNS (emit entries for these) ---")
            in_target = True
        prefix = "  TARGET" if in_target else "  CONTEXT"
        lines.append(f"{prefix} TURN {tidx}: {text}")
    if not in_target:
        lines.insert(0, "--- TARGET TURNS ---")
    return "\n".join(lines)


def render_prior_log(prior_entries, max_recent=8):
    if not prior_entries:
        return "(empty)"
    recent = list(reversed(prior_entries[-max_recent:]))
    lines = []
    for e in recent:
        ref_str = f" refs=[{','.join(e.refs)}]" if e.refs else ""
        pred_str = f" pred={e.predicate}" if e.predicate else ""
        lines.append(
            f"[{e.uuid}] t{e.ts} mentions={','.join(e.mentions)} :: {e.text}{ref_str}{pred_str}"
        )
    return "\n".join(lines)


def write_window_r13b(
    window_turns, target_turn_lo, target_turns, prior_entries, idx, cache, budget
):
    """Writer call: R22's centered-window prompt; entries use @ent_<id> mentions."""
    batch_text = " ".join(t for _, t in target_turns)
    ent_tags = set(re.findall(r"@ent_\d{5}", batch_text)) | {"@User"}
    active_state = render_active_state_durable(idx, ent_tags)
    prior_log = render_prior_log(prior_entries, max_recent=8)
    window_block = render_window(window_turns, target_turn_lo)

    prompt = WRITE_PROMPT.format(
        active_state=active_state,
        prior_log=prior_log,
        window_block=window_block,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    telemetry = {"prompt_chars": len(prompt), "window_size": len(window_turns)}
    if not isinstance(obj, dict):
        return [], telemetry
    target_turn_set = {t for t, _ in target_turns}
    items = obj.get("entries", []) or []
    entries = []
    counter = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        text = (it.get("text") or "").strip()
        if not text:
            continue
        ts_raw = it.get("turn")
        try:
            ts = int(ts_raw) if ts_raw is not None else target_turns[-1][0]
        except (TypeError, ValueError):
            ts = target_turns[-1][0]
        if ts not in target_turn_set:
            ts = target_turns[-1][0]
        mentions = [m for m in (it.get("mentions") or []) if isinstance(m, str)]
        refs = [r for r in (it.get("refs") or []) if isinstance(r, str)]
        predicate = (
            it.get("predicate") if isinstance(it.get("predicate"), str) else None
        )
        uuid = f"e{ts:04d}_{counter}"
        counter += 1
        entries.append(
            LogEntry(
                uuid=uuid,
                ts=ts,
                text=text,
                mentions=mentions,
                refs=refs,
                predicate=predicate,
            )
        )
    telemetry["n_emitted"] = len(entries)
    return entries, telemetry


def ingest_turns_fair(
    turns,
    cache,
    budget,
    *,
    w_past: int = 7,
    w_future: int = 7,
    k: int = 3,
    rebuild_index_every: int = 4,
    lru_size: int = 20,
    top_k: int = 5,
):
    """R13b's coref pass + R22's K-block centered window writer."""
    reg = r13b.PersistentRegistry(lru_size=lru_size)
    rewritten_turns = []
    coref_log = {}

    # Per-turn coref pass (R13b distinctive)
    for tidx, text in turns:
        new_text, decisions = r13b.coref_turn(
            tidx, text, reg, cache, budget, top_k=top_k
        )
        rewritten_turns.append((tidx, new_text))
        coref_log[tidx] = decisions
    cache.save()

    # Writer with R22's centered K=3 schedule
    log = []
    idx: IndexedLog | None = None
    n_turns = len(rewritten_turns)
    fire_no = 0
    target_lo = 0
    while target_lo < n_turns:
        target_hi = min(n_turns, target_lo + k)
        win_lo = max(0, target_lo - w_past)
        win_hi = min(n_turns, target_hi + w_future)
        window_turns = rewritten_turns[win_lo:win_hi]
        target_turns = rewritten_turns[target_lo:target_hi]
        if not target_turns:
            break
        target_turn_lo = target_turns[0][0]

        new_entries, _tele = write_window_r13b(
            window_turns,
            target_turn_lo,
            target_turns,
            log,
            idx,
            cache,
            budget,
        )
        log.extend(new_entries)
        if fire_no % rebuild_index_every == 0:
            idx = build_index(log, cache, budget)
        fire_no += 1
        target_lo = target_hi

    idx = build_index(log, cache, budget)
    return log, idx, reg, coref_log
