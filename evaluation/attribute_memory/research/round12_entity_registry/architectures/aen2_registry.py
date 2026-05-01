"""AEN-2 — entity-registry architecture.

Identity is decoupled from name. The registry stores entities by stable
internal id (`ent_00042`); names are aliases that can be added or contested.

Per-turn pipeline:
  1. LLM coref pass — identify pronouns, named mentions, descriptors in the
     turn text. The LRU cache (~20 entities) is passed in as candidate
     context. The LLM emits, for each mention, one of:
       - {"action": "resolve", "entity_id": "ent_00042"}    # cache hit
       - {"action": "resolve_by_alias", "alias": "Marcus"}   # name-match against
                                                              registry; we then
                                                              do alias-disambiguation
                                                              (LLM call only if
                                                              multiple candidates)
       - {"action": "create", "name": "Alice", "description": "..."}
       - {"action": "skip"}     # not an entity (e.g. distractor name)
  2. Alias disambiguation — for "resolve_by_alias" actions where the registry
     has multiple entities sharing that alias, fire a small LLM call with
     candidate descriptions + new turn context, asking which one it is or
     whether to create a new entity. Otherwise, take the single match (and add
     the new alias if novel) OR create a new entity.
  3. Rewrite turn text — replace each named/descriptor mention with
     `[@<entity_id> / <surface>]`. Pronouns get rewritten too. The result is
     the canonical form of the turn for downstream extraction.

Compatibility: produces LogEntry's compatible with aen1_simple's index +
retrieval. The mention list uses `@<entity_id>` (not `@User`/`@Marcus`),
which means at retrieval time we need to look up `entity_id` from a name in
the question. We pre-build a name->[entity_id] map from the registry.

The downstream writer (extraction LLM) is the same as aen1_simple's, except
the input turn text contains `@ent_00042` tags inline so the writer never has
to coref again.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND12 = HERE.parent
ROUND11 = ROUND12.parent / "round11_writer_stress"
ROUND7 = ROUND12.parent / "round7"
sys.path.insert(0, str(ROUND7 / "experiments"))
sys.path.insert(0, str(ROUND11 / "architectures"))

# Reuse aen1_simple's index/retrieval/writer — we only override the
# preprocessing (coref-pass) and the entity registry.
import aen1_simple  # noqa: E402
from _common import Budget, Cache, extract_json, llm  # noqa: E402

LogEntry = aen1_simple.LogEntry
IndexedLog = aen1_simple.IndexedLog
build_index = aen1_simple.build_index
write_batch = aen1_simple.write_batch


# ---------------------------------------------------------------------------
# Entity registry
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    id: str
    aliases: set[str] = field(default_factory=set)
    description: str = ""
    first_seen_turn: int = 0
    last_seen_turn: int = 0


@dataclass
class EntityRegistry:
    by_id: dict[str, Entity] = field(default_factory=dict)
    next_id: int = 1
    # LRU: list of entity_ids in order; most-recently-used at the END.
    lru_order: list[str] = field(default_factory=list)
    lru_size: int = 20

    def make_id(self) -> str:
        eid = f"ent_{self.next_id:05d}"
        self.next_id += 1
        return eid

    def create(self, aliases: list[str], description: str, turn_idx: int) -> Entity:
        eid = self.make_id()
        e = Entity(
            id=eid,
            aliases=set(a for a in aliases if a),
            description=description,
            first_seen_turn=turn_idx,
            last_seen_turn=turn_idx,
        )
        self.by_id[eid] = e
        self._touch(eid)
        return e

    def add_alias(self, entity_id: str, alias: str) -> None:
        if entity_id in self.by_id and alias:
            self.by_id[entity_id].aliases.add(alias)

    def update_description(self, entity_id: str, new_desc: str) -> None:
        if entity_id in self.by_id and new_desc:
            cur = self.by_id[entity_id].description
            if not cur:
                self.by_id[entity_id].description = new_desc
            elif new_desc.lower() not in cur.lower():
                self.by_id[entity_id].description = (cur + "; " + new_desc).strip("; ")

    def touch(self, entity_id: str, turn_idx: int) -> None:
        if entity_id in self.by_id:
            self.by_id[entity_id].last_seen_turn = turn_idx
        self._touch(entity_id)

    def _touch(self, entity_id: str) -> None:
        if entity_id in self.lru_order:
            self.lru_order.remove(entity_id)
        self.lru_order.append(entity_id)

    def lru_entities(self) -> list[Entity]:
        # most-recently-used last
        out: list[Entity] = []
        for eid in self.lru_order[-self.lru_size :]:
            if eid in self.by_id:
                out.append(self.by_id[eid])
        return out

    def candidates_by_alias(self, alias: str) -> list[Entity]:
        a_low = alias.lower().strip()
        out: list[Entity] = []
        for e in self.by_id.values():
            for al in e.aliases:
                if al.lower() == a_low:
                    out.append(e)
                    break
        return out


# ---------------------------------------------------------------------------
# Coref pass
# ---------------------------------------------------------------------------

COREF_PROMPT = """You are a coreference resolver for a chat assistant's memory.

For the user's NEW TURN below, identify every entity mention (named people,
pronouns, descriptive references like "the manager"). For each mention,
decide what to do.

The LRU cache below holds the most recently active entities. Each has a
stable internal id and a description. You should prefer to RESOLVE mentions
to a cached entity when context is consistent.

LRU CACHE (most-recently-active LAST):
{lru_block}

NEW TURN (turn {turn_idx}): "{turn_text}"

Output JSON of the form:
{{
  "mentions": [
    {{
      "surface": "<exact substring from the turn>",
      "kind": "named" | "pronoun" | "descriptor",
      "action": "resolve" | "resolve_by_alias" | "create" | "skip",
      "entity_id": "<ent_id from cache, ONLY if action=resolve>",
      "alias": "<canonical alias for resolve_by_alias / create>",
      "description": "<short identifying detail for create — role, relationship, etc.>"
    }}
  ]
}}

RULES
- If a pronoun (he/she/they/his/her/their/them) clearly refers to a single
  cached entity, use action="resolve" with that entity_id. If it is ambiguous
  among multiple cache entries, use action="skip" (we will leave it
  unresolved).
- For a NAMED mention, if exactly one cache entity has that name as an alias
  AND the turn context is consistent (or silent / no contradiction), use
  action="resolve" with that entity_id. SILENT context is the default —
  do NOT create a new entity just because the turn has no extra detail.
- For a NAMED mention not in the cache, use action="resolve_by_alias" with the
  alias text (we'll do a registry-wide alias check). The pipeline will create
  a new entity if there's no match in the broader registry.
- Only use action="create" when the turn EXPLICITLY introduces a new entity
  whose name DIFFERS from anyone in the cache, AND you can give a useful
  identifying description. The description should be concise (1 short clause).
- Only use action="skip" when a token isn't really an entity (e.g.
  "Tomorrow", "Monday", a noun like "cat" used generically) or when a pronoun
  is genuinely ambiguous.
- For DESCRIPTORS like "the recruiter", "my manager", "the dog walker": if
  the description matches a cached entity, RESOLVE. Otherwise use
  action="resolve_by_alias" with the descriptor as the alias (we will search
  the broader registry).
- The user's "I" / "me" / "my" should resolve to the entity whose alias is
  "User" (it should already be in the cache).
- Do NOT invent entity_ids. Only use ids you see in the cache.
- Do NOT split a single mention into multiple. Just one entry per textual
  occurrence.

Output ONLY the JSON.
"""


def _render_lru_block(reg: EntityRegistry) -> str:
    ents = reg.lru_entities()
    if not ents:
        return "(empty cache)"
    lines = []
    for e in ents:
        aliases = "/".join(sorted(e.aliases)) if e.aliases else "(no aliases)"
        desc = e.description or "(no description)"
        lines.append(
            f"  [{e.id}] aliases={aliases} :: {desc} (last_seen=t{e.last_seen_turn})"
        )
    return "\n".join(lines)


# Alias disambiguation prompt: when multiple registry entities share a name.
ALIAS_DISAMBIG_PROMPT = """A name in the user's turn matches multiple entities
in the registry. Decide which one is meant — or whether the turn introduces a
NEW distinct entity.

NEW TURN (turn {turn_idx}): "{turn_text}"

NAME OR DESCRIPTOR: "{alias}"

CANDIDATE ENTITIES:
{candidates_block}

Output JSON:
{{
  "decision": "match" | "create_new",
  "entity_id": "<ent_id if match>",
  "rationale": "<one short sentence>"
}}

RULES
- If the turn's context (role, relationship, location, attribute) is consistent
  with one candidate and not the others, pick that candidate.
- If the context is SILENT (no extra detail), pick the most-recently-active
  candidate (the one with the largest last_seen_turn). Do NOT create a new
  entity.
- Only create a new entity when the turn EXPLICITLY signals a distinct
  identity ("X from high school", "X — Alice Foster", "different X — my
  dentist", "another X").
- Output ONLY JSON.
"""


def _render_candidates(cands: list[Entity]) -> str:
    lines = []
    for e in cands:
        aliases = "/".join(sorted(e.aliases))
        desc = e.description or "(no description)"
        lines.append(
            f"  [{e.id}] aliases={aliases} :: {desc} (last_seen=t{e.last_seen_turn})"
        )
    return "\n".join(lines)


def disambiguate_alias(
    alias: str,
    candidates: list[Entity],
    turn_idx: int,
    turn_text: str,
    cache: Cache,
    budget: Budget,
) -> tuple[str | None, str]:
    """Returns (entity_id_or_None, rationale)."""
    if len(candidates) == 1:
        return candidates[0].id, "single-candidate"
    if not candidates:
        return None, "no-candidates"
    prompt = ALIAS_DISAMBIG_PROMPT.format(
        turn_idx=turn_idx,
        turn_text=turn_text,
        alias=alias,
        candidates_block=_render_candidates(candidates),
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        # default to most-recently-used
        cands_sorted = sorted(candidates, key=lambda e: -e.last_seen_turn)
        return cands_sorted[0].id, "fallback (parse error)"
    decision = obj.get("decision")
    if decision == "match":
        eid = obj.get("entity_id")
        if eid in {c.id for c in candidates}:
            return eid, obj.get("rationale", "")
    if decision == "create_new":
        return None, obj.get("rationale", "create_new")
    # fallback: most recent
    cands_sorted = sorted(candidates, key=lambda e: -e.last_seen_turn)
    return cands_sorted[0].id, "fallback"


# ---------------------------------------------------------------------------
# Per-turn coref + rewrite
# ---------------------------------------------------------------------------


@dataclass
class CorefDecision:
    surface: str
    kind: str
    entity_id: str | None  # None if skipped
    rationale: str = ""


def coref_turn(
    turn_idx: int,
    turn_text: str,
    reg: EntityRegistry,
    cache: Cache,
    budget: Budget,
) -> tuple[str, list[CorefDecision]]:
    """LLM coref pass + alias disambiguation; returns (rewritten_text, decisions).

    rewritten_text replaces each surface form with `[@<eid> / <surface>]`.
    """
    # Always pre-seed the registry with the User entity.
    if "ent_user" not in reg.by_id:
        reg.by_id["ent_user"] = Entity(
            id="ent_user",
            aliases={"User", "I", "me", "my"},
            description="The user / speaker (first-person).",
            first_seen_turn=0,
            last_seen_turn=turn_idx,
        )
        reg._touch("ent_user")

    prompt = COREF_PROMPT.format(
        lru_block=_render_lru_block(reg),
        turn_idx=turn_idx,
        turn_text=turn_text,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    decisions: list[CorefDecision] = []
    if not isinstance(obj, dict):
        return turn_text, decisions
    mentions = obj.get("mentions", []) or []

    # Process mentions in order. Build replacement list.
    replacements: list[tuple[str, str]] = []  # (surface, replacement_string)
    for m in mentions:
        if not isinstance(m, dict):
            continue
        surface = (m.get("surface") or "").strip()
        if not surface:
            continue
        kind = m.get("kind") or "named"
        action = m.get("action") or "skip"
        eid: str | None = None
        rationale = ""
        if action == "resolve":
            cand_eid = m.get("entity_id")
            if cand_eid in reg.by_id:
                eid = cand_eid
                # If named, ensure surface is in aliases
                if kind == "named":
                    reg.add_alias(eid, surface)
            rationale = "cache-resolve"
        elif action == "resolve_by_alias":
            alias = m.get("alias") or surface
            cands = reg.candidates_by_alias(alias)
            # Also try resolving descriptors against descriptions for kind=descriptor
            if kind == "descriptor" and not cands:
                cands = _candidates_by_description(alias, reg)
            if cands:
                eid, rationale = disambiguate_alias(
                    alias, cands, turn_idx, turn_text, cache, budget
                )
                if eid is None:
                    # disambiguation said "create_new"
                    desc = (m.get("description") or "").strip()
                    e = reg.create([alias], desc, turn_idx)
                    eid = e.id
                    rationale = "disambig-create-new"
            else:
                # No registry match — create
                desc = (m.get("description") or "").strip()
                e = reg.create([alias], desc, turn_idx)
                eid = e.id
                rationale = "registry-miss-create"
        elif action == "create":
            alias = m.get("alias") or surface
            desc = (m.get("description") or "").strip()
            # Defensive: still check registry first, this should be rare
            cands = reg.candidates_by_alias(alias)
            if cands:
                eid, rationale = disambiguate_alias(
                    alias, cands, turn_idx, turn_text, cache, budget
                )
                if eid is None:
                    e = reg.create([alias], desc, turn_idx)
                    eid = e.id
                    rationale = "create-after-disambig"
            else:
                e = reg.create([alias], desc, turn_idx)
                eid = e.id
                rationale = "create-fresh"
        else:  # skip
            decisions.append(CorefDecision(surface, kind, None, "skipped"))
            continue

        # Touch + register decision
        if eid is not None:
            reg.touch(eid, turn_idx)
            # Update description if create yielded one (or supplemental info from coref)
            extra_desc = (m.get("description") or "").strip()
            if extra_desc and kind == "named":
                reg.update_description(eid, extra_desc)
        decisions.append(CorefDecision(surface, kind, eid, rationale))
        if eid is not None:
            replacements.append((surface, f"[@{eid} / {surface}]"))

    # Apply replacements once, longest-first to avoid substring collisions
    rewritten = turn_text
    for surface, repl in sorted(replacements, key=lambda x: -len(x[0])):
        # Use word-boundary regex; descriptors might include spaces
        pattern = re.compile(re.escape(surface))
        # Replace only first occurrence to keep mention-count stable
        rewritten = pattern.sub(repl, rewritten, count=1)

    return rewritten, decisions


def _candidates_by_description(query: str, reg: EntityRegistry) -> list[Entity]:
    """Cheap fallback: substring match against entity descriptions/aliases."""
    q_low = query.lower()
    out: list[Entity] = []
    # Tokens to search for (skip articles)
    skip = {
        "the",
        "a",
        "an",
        "my",
        "our",
        "your",
        "this",
        "that",
        "his",
        "her",
        "their",
    }
    tokens = [
        t for t in re.findall(r"[a-zA-Z]+", q_low) if t not in skip and len(t) >= 4
    ]
    if not tokens:
        return out
    for e in reg.by_id.values():
        text = (e.description + " " + " ".join(e.aliases)).lower()
        if any(t in text for t in tokens):
            out.append(e)

    # Rank by number of token hits, then recency
    def score(e: Entity) -> tuple[int, int]:
        text = (e.description + " " + " ".join(e.aliases)).lower()
        hits = sum(1 for t in tokens if t in text)
        return (hits, e.last_seen_turn)

    out.sort(key=score, reverse=True)
    return out[:5]


# ---------------------------------------------------------------------------
# End-to-end ingestion
# ---------------------------------------------------------------------------


def ingest_turns_with_registry(
    turns: list[tuple[int, str]],  # (turn_idx, text)
    cache: Cache,
    budget: Budget,
    batch_size: int = 5,
    rebuild_index_every: int = 25,
    lru_size: int = 20,
) -> tuple[list[LogEntry], IndexedLog, EntityRegistry, dict[int, list[CorefDecision]]]:
    """Run coref-pass + writer on the supplied turns.

    Returns log, indexed log, registry, and per-turn coref decisions.
    """
    reg = EntityRegistry(lru_size=lru_size)
    rewritten_turns: list[tuple[int, str]] = []
    coref_log: dict[int, list[CorefDecision]] = {}

    for tidx, text in turns:
        new_text, decisions = coref_turn(tidx, text, reg, cache, budget)
        rewritten_turns.append((tidx, new_text))
        coref_log[tidx] = decisions
        cache.save()

    # Now use the standard writer (aen1_simple.write_batch / build_index) on
    # the rewritten text. Mention names are now `@ent_*` style — but
    # aen1_simple's writer regex looks for capitalized tokens. We pass the
    # rewritten text in as-is; the writer prompt explains.
    log: list[LogEntry] = []
    known: set[str] = {"User"}
    idx: IndexedLog | None = None

    for i in range(0, len(rewritten_turns), batch_size):
        batch = rewritten_turns[i : i + batch_size]
        new_entries = _write_batch_with_eids(batch, log, idx, known, reg, cache, budget)
        for e in new_entries:
            for m in e.mentions:
                if m.startswith("@"):
                    known.add(m[1:])
        log.extend(new_entries)
        if (i // batch_size) % rebuild_index_every == 0:
            idx = build_index(log, cache, budget)

    idx = build_index(log, cache, budget)
    return log, idx, reg, coref_log


# ---------------------------------------------------------------------------
# Writer (extension of aen1_simple's)
# ---------------------------------------------------------------------------

WRITE_PROMPT = """You are a semantic-memory writer using a SINGLE APPEND-ONLY LOG.

Each entry you write is an atomic natural-language fact. Use entity tags of
the form `@ent_NNNNN` (or @User for the speaker) to mention entities — these
are stable internal ids; they have already been resolved for you in the turn
text below, in the form [@ent_NNNNN / surface_form].

When you emit an entry, set `mentions` to the list of `@ent_*` tags for the
entities involved. Use `refs` to point at prior entry uuids when this entry
relates to, updates, or corrects a prior fact. There is only ONE kind of ref.
For state-tracking facts include `predicate` as `@ent_NNNNN.pred_name`.

KNOWN ENTITY TAGS so far: {known_entities}

PRIOR LOG SAMPLE (most recent + relevant chain heads):
{prior_log}

BATCH OF TURNS (entity tags already resolved inline):
{turn_block}

Emit JSON:
{{
  "entries": [
    {{
      "text": "<atomic fact in one sentence>",
      "mentions": ["@ent_NNNNN", ...],
      "refs": ["<prior-uuid>", ...],
      "predicate": "@ent_NNNNN.pred" or null
    }}
  ]
}}

RULES
- ALWAYS use the `@ent_*` tags exactly as they appear in the turn text. Do
  NOT use bare names like "Alice" or "Marcus" — only the resolved tags.
- If a turn is filler (weather, chitchat), skip silently.
- If a turn UPDATES or CORRECTS a prior fact, emit a new entry with `refs`
  pointing at the prior entry that stated the now-outdated value.
- For state-tracking facts (job, location, boss, employer, role,
  relationship), include `predicate` in the form `@ent_NNNNN.pred_name`.
- Output JSON ONLY.
"""


def _write_batch_with_eids(
    batch_turns: list[tuple[int, str]],
    prior_entries: list[LogEntry],
    idx: IndexedLog | None,
    known_entities: set[str],
    reg: EntityRegistry,
    cache: Cache,
    budget: Budget,
) -> list[LogEntry]:
    # Pull relevant chain heads by @ent_* match
    relevant: list[LogEntry] = []
    if idx is not None:
        batch_text = " ".join(t for _, t in batch_turns)
        ent_tags = set(re.findall(r"@ent_\d{5}", batch_text)) | {"@User"}
        seen: set[str] = set()
        for (tag, pred), uuid in idx.supersede_head.items():
            if tag in ent_tags and uuid not in seen:
                seen.add(uuid)
                relevant.append(idx.by_uuid[uuid])
        relevant.sort(key=lambda e: e.ts, reverse=True)
        relevant = relevant[:8]

    prior_log = aen1_simple._render_prior_log(prior_entries, relevant_heads=relevant)
    turn_block = "\n".join(f"TURN {i}: {t}" for i, t in batch_turns)
    prompt = WRITE_PROMPT.format(
        known_entities=", ".join(sorted(known_entities))
        if known_entities
        else "(none)",
        prior_log=prior_log,
        turn_block=turn_block,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return []
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
    return entries


# ---------------------------------------------------------------------------
# Snapshot helpers (for grading / inspection)
# ---------------------------------------------------------------------------


def registry_snapshot(reg: EntityRegistry) -> dict:
    return {
        "by_id": {
            eid: {
                "aliases": sorted(e.aliases),
                "description": e.description,
                "first_seen": e.first_seen_turn,
                "last_seen": e.last_seen_turn,
            }
            for eid, e in reg.by_id.items()
        },
        "lru_order": list(reg.lru_order),
    }
