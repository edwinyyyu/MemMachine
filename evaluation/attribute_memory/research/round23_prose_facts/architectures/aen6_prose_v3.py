"""AEN-6 PROSE — facts as prose with mentions; canonical entities as DSU.

Design principles:
- ALL facts are prose with mention markup. No specialized fact subtypes
  (no NameFact, no PredicateFact). The text carries the semantics; mentions
  carry the entity references.
- Mention IDs are opaque, per-occurrence: each surface occurrence gets a
  fresh mention_id. Two surfaces with the same name don't collide.
- Canonical entities are a disjoint-set (union-find) wrapper over mentions.
  When the writer judges two mentions refer to the same person, it merges
  them. find(mention_id) -> entity_id (the canonical class).
- Names/descriptions are not separate fields — they're prose in the facts
  that reference the entity. "User's new boss is named Marcus" is just a
  fact mentioning two mentions (boss-descriptor and Marcus-name) that
  resolve to the same entity.
- No mandatory subject/predicate/object on the fact. SPO chains are derived
  at index time when needed; raw facts are flexible prose.

Schema:
    Fact            : prose + mention_ids, in a collection
    Mention         : opaque id, surface, fact_uuid, ts
    EntityRegistry  : DSU over mentions (mention_id -> entity_id)
    BindingEvent    : audit log of merges/splits
    MemoryStore     : multi-collection (observations, cognition)

Disjoint-set ops (friendly wrapper):
    get_canonical(mention_id) -> entity_id
    get_class(entity_id)      -> set[mention_id]
    get_class_facts(entity_id)-> set[fact_uuid]
    get_surfaces(entity_id)   -> dict[surface, count]
    merge(m1, m2, evidence)
    split(m1, evidence)
    query_surface(surface)    -> list[entity_id with that surface]

Solved-problem coverage:
- Cross-batch coref: writer at name-turn merges name-mention with descriptor-mention's entity
- Same-name disambig: each "Alice" gets fresh mention_id; merge only when writer asserts
- Multi-name same-entity (alias): merge (Q-mention, Quentin-mention)
- Long-gap supersession: chain transitions retrievable via entity_facts; "current state"
  derived at query time from the latest fact about that slot
- Conditional/HBR: conditional fact retrievable via mention_index just like any other
- World scoping: hierarchical entity_ids ("dune.paul") via merge with namespace anchors,
  OR distinct entity_ids for fictional vs real (user picks via merge decisions)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND23 = HERE.parent
RESEARCH = ROUND23.parent
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND7 / "experiments"))

from _common import Budget, Cache, cosine, embed_batch, extract_json, llm  # noqa: E402

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Mention:
    mention_id: str  # opaque, per-occurrence
    surface: str  # displayed text token
    fact_uuid: str  # fact this mention appears in
    ts: int


@dataclass
class Fact:
    fact_uuid: str
    ts: int
    text: str  # prose; may contain [m_id|surface] markers
    mention_ids: list[str] = field(default_factory=list)
    collection: str = "observations"


@dataclass
class BindingEvent:
    """Audit log for merges and splits."""

    op: str  # "merge" | "split"
    ts: int
    mention_ids: list[str]  # mentions involved (2 for merge, 1 for split)
    evidence_fact_uuids: list[str] = field(default_factory=list)
    rationale: str | None = None


# ---------------------------------------------------------------------------
# EntityRegistry — disjoint-set wrapper with friendly ops
# ---------------------------------------------------------------------------


@dataclass
class EntityRegistry:
    """Disjoint-set over mentions. Each equivalence class IS a canonical entity."""

    # Each mention starts in its own class. mention_to_entity[m] = e (canonical id).
    # Default: e = "e_<m>" (mention's own id, prefixed). After merges, mentions in
    # the same class share an entity_id.
    mention_to_entity: dict[str, str] = field(default_factory=dict)
    entity_members: dict[str, set[str]] = field(default_factory=dict)
    binding_events: list[BindingEvent] = field(default_factory=list)

    def register(self, mention_id: str) -> str:
        """Add a fresh mention to the registry. Returns its entity_id."""
        if mention_id in self.mention_to_entity:
            return self.mention_to_entity[mention_id]
        entity_id = f"e_{mention_id}"
        self.mention_to_entity[mention_id] = entity_id
        self.entity_members.setdefault(entity_id, set()).add(mention_id)
        return entity_id

    def get_canonical(self, mention_id: str) -> str:
        """Return canonical entity_id for a mention. Self-binds if unknown."""
        if mention_id not in self.mention_to_entity:
            return self.register(mention_id)
        return self.mention_to_entity[mention_id]

    def get_class(self, entity_id: str) -> set[str]:
        """All mentions in the entity's equivalence class."""
        return self.entity_members.get(entity_id, set()).copy()

    def merge(
        self,
        m1: str,
        m2: str,
        ts: int,
        evidence_fact_uuids: list[str] | None = None,
        rationale: str | None = None,
    ) -> str:
        """Unify the equivalence classes of m1 and m2. Returns the surviving entity_id."""
        e1 = self.get_canonical(m1)
        e2 = self.get_canonical(m2)
        if e1 == e2:
            return e1
        # Pick survivor: earlier-created entity wins (alphabetical fallback for stability)
        survivor, loser = (e1, e2) if e1 <= e2 else (e2, e1)
        loser_members = self.entity_members.pop(loser, set())
        self.entity_members.setdefault(survivor, set()).update(loser_members)
        for m in loser_members:
            self.mention_to_entity[m] = survivor
        self.binding_events.append(
            BindingEvent(
                op="merge",
                ts=ts,
                mention_ids=[m1, m2],
                evidence_fact_uuids=list(evidence_fact_uuids or []),
                rationale=rationale,
            )
        )
        return survivor

    def split(
        self,
        m: str,
        ts: int,
        evidence_fact_uuids: list[str] | None = None,
        rationale: str | None = None,
    ) -> str:
        """Remove m from its current class; assign it a fresh entity_id."""
        old_entity = self.mention_to_entity.get(m)
        if old_entity and old_entity in self.entity_members:
            self.entity_members[old_entity].discard(m)
            if not self.entity_members[old_entity]:
                del self.entity_members[old_entity]
        new_entity_id = f"e_split_{m}_{ts}"
        self.mention_to_entity[m] = new_entity_id
        self.entity_members[new_entity_id] = {m}
        self.binding_events.append(
            BindingEvent(
                op="split",
                ts=ts,
                mention_ids=[m],
                evidence_fact_uuids=list(evidence_fact_uuids or []),
                rationale=rationale,
            )
        )
        return new_entity_id


# ---------------------------------------------------------------------------
# Indexes (per collection)
# ---------------------------------------------------------------------------


@dataclass
class IndexedCollection:
    name: str
    facts: list[Fact]
    by_uuid: dict[str, Fact]
    mentions_by_id: dict[str, Mention]
    mentions_by_fact: dict[str, list[str]]  # fact_uuid -> [mention_id]
    mentions_by_surface: dict[str, list[str]]  # normalized surface -> [mention_id]
    facts_by_entity: dict[str, list[str]]  # entity_id -> [fact_uuid]
    embed_by_uuid: dict[str, list[float]]


@dataclass
class MemoryStore:
    collections: dict[str, IndexedCollection] = field(default_factory=dict)
    registry: EntityRegistry = field(default_factory=EntityRegistry)


def _normalize_surface(s: str) -> str:
    s = s.strip().lower()
    # Strip leading articles
    for art in ("the ", "a ", "an ", "my ", "your ", "his ", "her ", "their ", "@"):
        s = s.removeprefix(art)
    return s


def build_collection(
    name: str,
    facts: list[Fact],
    mentions: list[Mention],
    registry: EntityRegistry,
    cache: Cache,
    budget: Budget,
) -> IndexedCollection:
    by_uuid = {f.fact_uuid: f for f in facts}
    mentions_by_id = {m.mention_id: m for m in mentions}

    mentions_by_fact: dict[str, list[str]] = {}
    for m in mentions:
        mentions_by_fact.setdefault(m.fact_uuid, []).append(m.mention_id)

    mentions_by_surface: dict[str, list[str]] = {}
    for m in mentions:
        mentions_by_surface.setdefault(_normalize_surface(m.surface), []).append(
            m.mention_id
        )

    facts_by_entity: dict[str, list[str]] = {}
    for f in facts:
        seen_entities = set()
        for mid in f.mention_ids:
            if mid not in mentions_by_id:
                continue
            eid = registry.get_canonical(mid)
            if eid in seen_entities:
                continue
            seen_entities.add(eid)
            facts_by_entity.setdefault(eid, []).append(f.fact_uuid)

    texts = [f.text for f in facts]
    embs = embed_batch(texts, cache, budget) if facts else []
    embed_by_uuid = {f.fact_uuid: embs[i] for i, f in enumerate(facts)}

    return IndexedCollection(
        name=name,
        facts=facts,
        by_uuid=by_uuid,
        mentions_by_id=mentions_by_id,
        mentions_by_fact=mentions_by_fact,
        mentions_by_surface=mentions_by_surface,
        facts_by_entity=facts_by_entity,
        embed_by_uuid=embed_by_uuid,
    )


# ---------------------------------------------------------------------------
# Active state rendering for the writer
# ---------------------------------------------------------------------------


def render_active_entities(
    obs_idx: IndexedCollection | None,
    registry: EntityRegistry,
    target_surfaces: set[str],
    max_entities: int = 20,
    excerpts_per_entity: int = 2,
) -> str:
    """Render entities relevant to the target window:
    - Entities that have any mention with a surface in target_surfaces.
    - Plus a few recently-mentioned entities as fallback context.
    """
    if obs_idx is None:
        return "(empty)"

    # Score entities: relevance (surface match) + recency
    candidate_entities: dict[str, dict] = {}
    norm_targets = {_normalize_surface(s) for s in target_surfaces}
    for surface_norm, mids in obs_idx.mentions_by_surface.items():
        match = any(t in surface_norm or surface_norm in t for t in norm_targets if t)
        for mid in mids:
            eid = registry.get_canonical(mid)
            ent = candidate_entities.setdefault(
                eid, {"latest_ts": 0, "surfaces": set(), "matched": False}
            )
            m = obs_idx.mentions_by_id[mid]
            ent["surfaces"].add(m.surface)
            ent["latest_ts"] = max(ent["latest_ts"], m.ts)
            if match:
                ent["matched"] = True

    if not candidate_entities:
        return "(none)"

    # Order: matched first, then recency
    ordered = sorted(
        candidate_entities.items(),
        key=lambda kv: (-int(kv[1]["matched"]), -kv[1]["latest_ts"]),
    )
    lines = []
    for eid, info in ordered[:max_entities]:
        surfaces = sorted(info["surfaces"])
        # Pull excerpts: most recent facts referencing this entity
        fact_uuids = obs_idx.facts_by_entity.get(eid, [])[-excerpts_per_entity:]
        excerpts = []
        for fu in fact_uuids:
            f = obs_idx.by_uuid.get(fu)
            if f:
                excerpts.append(f"      [{f.fact_uuid} t={f.ts}] {f.text[:120]}")
        excerpt_block = "\n".join(excerpts) if excerpts else "      (no recent facts)"
        lines.append(
            f"  - entity_id={eid}\n"
            f"      surfaces: {surfaces}\n"
            f"      recent_facts:\n{excerpt_block}"
        )
    return "\n".join(lines)


def render_recent_facts(prior_facts: list[Fact], cap: int = 8) -> str:
    if not prior_facts:
        return "(empty)"
    recent = list(reversed(prior_facts[-cap:]))
    lines = [f"  [{f.fact_uuid} t={f.ts}] {f.text[:120]}" for f in recent]
    return "\n".join(lines)


def extract_window_surfaces(turns: list[tuple[int, str]]) -> set[str]:
    text = " ".join(t for _, t in turns)
    tokens = re.findall(r"\b([A-Z][a-z]{1,30})\b", text)
    return set(tokens) | {"User"}


# ---------------------------------------------------------------------------
# Writer prompt
# ---------------------------------------------------------------------------


WRITE_PROMPT = """You are a semantic-memory writer that emits FACTS as prose.
Each fact is an atomic prose sentence with marked entity mentions.

DEFAULT IS TO EMIT NOTHING. Only emit facts for DURABLE LIFE EVENTS — facts
that would still be relevant a month later. Most turns produce no facts.

EMIT for these (chain-worthy):
  - New job, new boss, new team, new colleague mention (named introduction)
  - New location, new home, new school
  - New relationship, new friend, new mentor, new neighbor
  - New possession: car, bike, pet (e.g., "picked up a Bianchi")
  - Hobbies & recurring routines (e.g., "started climbing", "joined a gym")
  - Confirmed plans/decisions ("decided to move to Berlin", "starting at Notion")
  - Naming an entity that was previously anonymous (the binding event itself)
  - Update/change to any of the above
  - Conditionals about User's life ("If hired at X, my boss will be Y")

DO NOT EMIT (skip silently — output empty facts for filler-only TARGET turns):
  - Body sensations: "stomach hurts", "tired", "need a nap"
  - Weather: "rainy day", "weather is nice"
  - Transient feelings: "long day", "slow afternoon", "frustrated"
  - One-off mundane actions: "had coffee", "going to lunch", "on calls all morning"
  - Notification noise: "412 unread", "email avalanche", "Slack laggy"
  - Generic chitchat with no durable content

PREFER ONE FACT PER TURN. Multiple facts only if a turn bundles unrelated
durable events. Aim for HIGH SIGNAL DENSITY — short list of important facts.

ENTITY RESOLUTION RULES — CRITICAL:

Each mention has:
  - surface: the literal text token (e.g., "User", "Marcus", "the new boss")
  - resolves_to: an entity_id from ACTIVE ENTITIES if same entity, OR "new"

DEFAULT TO REUSING EXISTING ENTITIES, NOT CREATING NEW ONES.
  - When TARGET TURN says "his", "her", "they", "the boss", "the team", etc.
    and ACTIVE ENTITIES shows a matching entity, use its entity_id.
  - When TARGET TURN names an entity that ACTIVE ENTITIES already has under
    the same role/context, use the existing entity_id.
  - Only use "new" when there is NO plausible existing entity match.

MERGE NAME REVEALS WITH DESCRIPTOR ENTITIES — THE KEY MECHANIC:
  - When a TARGET TURN names an entity that ACTIVE ENTITIES has anonymously
    (e.g., active entity surfaces=["new boss"], TARGET says "his name is
    Marcus"), the new mention "Marcus" should resolve_to the SAME entity_id
    as the existing anonymous entity. This is the binding event.
  - Multiple mentions IN THE SAME FACT can resolve to the same entity_id —
    that's how you assert co-reference within a fact (e.g., "boss" and
    "Marcus" in "User's boss is Marcus" both resolve_to the same entity).

ACTIVE ENTITIES (PREFER THESE for resolves_to):
{active_entities}

RECENT FACTS (for context):
{recent_facts}

CONVERSATION WINDOW
-------------------
{window_block}

Schema:
{{
  "facts": [
    {{
      "turn": <int from TARGET turns>,
      "text": "<atomic prose sentence>",
      "mentions": [
        {{"surface": "<literal text token>", "resolves_to": "<entity_id or 'new'>"}}
      ]
    }}
  ]
}}

EXAMPLES:

(1) Filler turn ("Coffee was good this morning."): output `{{"facts": []}}`.

(2) Anonymous descriptor (turn 3 — boss not yet named):
  text: "User's new boss started this week."
  mentions: [
    {{"surface": "User", "resolves_to": "e_user"}},
    {{"surface": "new boss", "resolves_to": "new"}}
  ]

(3) Name reveal of an existing anon entity (turn 20):
  ACTIVE ENTITIES shows: entity_id=e_a01 with surfaces=["new boss"]
                          recent_facts: "User's new boss started this week"
  TARGET TURN: "Oh — his name is Marcus, by the way."
  text: "User's boss is named Marcus."
  mentions: [
    {{"surface": "User", "resolves_to": "e_user"}},
    {{"surface": "boss", "resolves_to": "e_a01"}},     # SAME entity (the anon boss)
    {{"surface": "Marcus", "resolves_to": "e_a01"}}    # SAME entity — naming event
  ]
  (Critical: BOTH mentions resolve to the EXISTING e_a01 — this is the bind.)

(4) Different person sharing a name (Alice neighbor vs Alice colleague):
  Active shows entity_id=e_b03 surfaces=["Alice", "the new neighbor"], at
  turn 200 TARGET says "Colleague Alice helped me debug today" — context
  does NOT match neighbor, so:
  mentions: [
    {{"surface": "User", "resolves_to": "e_user"}},
    {{"surface": "Alice", "resolves_to": "new"}}        # different Alice (colleague)
  ]

(5) Conditional / hypothetical (real world cognition):
  text: "If User is hired at Notion, User's boss will be Sam."
  mentions: [
    {{"surface": "User", "resolves_to": "e_user"}},
    {{"surface": "Notion", "resolves_to": "new"}},   (or existing if Notion already known)
    {{"surface": "Sam", "resolves_to": "new"}}
  ]

Output JSON ONLY. No commentary.
"""


def write_window(
    window_turns: list[tuple[int, str]],
    target_turn_lo: int,
    target_turns: list[tuple[int, str]],
    prior_facts: list[Fact],
    obs_idx: IndexedCollection | None,
    registry: EntityRegistry,
    cache: Cache,
    budget: Budget,
) -> tuple[list[Fact], list[Mention], list[BindingEvent], dict]:
    """Emit facts (with mentions) for TARGET turns. Apply binding decisions to registry."""
    target_surfaces = extract_window_surfaces(target_turns)
    active_entities = render_active_entities(obs_idx, registry, target_surfaces)
    recent_facts_str = render_recent_facts(prior_facts, cap=8)

    window_lines = []
    in_target = False
    for tidx, text in window_turns:
        if not in_target and tidx >= target_turn_lo:
            window_lines.append("--- TARGET TURNS (emit facts for these) ---")
            in_target = True
        prefix = "  TARGET" if in_target else "  CONTEXT"
        window_lines.append(f"{prefix} TURN {tidx}: {text}")
    if not in_target:
        window_lines.insert(0, "--- TARGET TURNS ---")
    window_block = "\n".join(window_lines)

    prompt = WRITE_PROMPT.format(
        active_entities=active_entities,
        recent_facts=recent_facts_str,
        window_block=window_block,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    telemetry = {"prompt_chars": len(prompt), "window_size": len(window_turns)}
    if not isinstance(obj, dict):
        return [], [], [], telemetry

    target_turn_set = {t for t, _ in target_turns}
    facts_raw = obj.get("facts", []) or []
    new_facts: list[Fact] = []
    new_mentions: list[Mention] = []
    new_bindings: list[BindingEvent] = []
    fact_counter = 0

    for fr in facts_raw:
        if not isinstance(fr, dict):
            continue
        text = (fr.get("text") or "").strip()
        if not text:
            continue
        ts_raw = fr.get("turn")
        try:
            ts = int(ts_raw) if ts_raw is not None else target_turns[-1][0]
        except (TypeError, ValueError):
            ts = target_turns[-1][0]
        if ts not in target_turn_set:
            ts = target_turns[-1][0]

        fact_uuid = f"f{ts:04d}_{fact_counter}"
        fact_counter += 1

        mention_specs = fr.get("mentions") or []
        # Phase 1: assign fresh mention_ids and register
        local_resolutions: list[tuple[str, str]] = []
        # Map from "intra-fact resolve target" to first mention_id with that target
        intra_resolve: dict[str, str] = {}

        for i, ms in enumerate(mention_specs):
            if not isinstance(ms, dict):
                continue
            surface = (ms.get("surface") or "").strip()
            if not surface:
                continue
            resolves_to = (ms.get("resolves_to") or "new").strip()
            mention_id = f"m{ts:04d}_{fact_counter}_{i}"
            new_mentions.append(
                Mention(
                    mention_id=mention_id,
                    surface=surface,
                    fact_uuid=fact_uuid,
                    ts=ts,
                )
            )
            registry.register(mention_id)
            local_resolutions.append((mention_id, resolves_to))

        # Phase 2: apply resolutions (merges)
        for mention_id, resolves_to in local_resolutions:
            if resolves_to == "new" or not resolves_to:
                continue
            # Check: is resolves_to a known entity_id?
            if resolves_to in registry.entity_members:
                # Merge mention's class with that entity's class
                existing_member = next(iter(registry.entity_members[resolves_to]))
                survivor = registry.merge(
                    mention_id,
                    existing_member,
                    ts=ts,
                    evidence_fact_uuids=[fact_uuid],
                    rationale=f"writer asserted resolves_to={resolves_to}",
                )
                new_bindings.append(registry.binding_events[-1])
            elif resolves_to in intra_resolve:
                # Co-ref within the same fact
                survivor = registry.merge(
                    mention_id,
                    intra_resolve[resolves_to],
                    ts=ts,
                    evidence_fact_uuids=[fact_uuid],
                    rationale="intra-fact coref",
                )
                new_bindings.append(registry.binding_events[-1])
            else:
                # Track as a new intra-fact resolution target
                intra_resolve[resolves_to] = mention_id

        new_facts.append(
            Fact(
                fact_uuid=fact_uuid,
                ts=ts,
                text=text,
                mention_ids=[m for m, _ in local_resolutions],
                collection="observations",
            )
        )

    telemetry["n_facts_emitted"] = len(new_facts)
    telemetry["n_mentions"] = len(new_mentions)
    telemetry["n_merges"] = sum(1 for b in new_bindings if b.op == "merge")
    return new_facts, new_mentions, new_bindings, telemetry


# ---------------------------------------------------------------------------
# Cognition pass (R23 v3 — adds cognition collection on top of v2 prose facts)
# ---------------------------------------------------------------------------


COGNIZE_PROMPT = """You are the COGNITION pass for a prose-fact memory.

The writer just emitted these OBSERVATION facts from the latest turns. Your
job: emit COGNITION facts that capture User's mental state — expectations,
plans, beliefs, fears, hopes, confirmations of prior plans, contradictions
of prior expectations.

Cognition facts use the SAME schema as observation facts (prose + mention
list with resolves_to). They live in a separate cognition collection so
they don't pollute observation retrieval.

DEFAULT IS EMPTY. Emit AT MOST one cognition fact per K-block, often zero.

EMIT cognition entries ONLY for SPECIFIC TRIGGERS:

  TRIGGER 1: CONDITIONAL — turn states "if X then Y" or describes a plan
  contingent on something.
    Cognition fact text: "User plans to do Y if X happens." OR "User
    expects Y conditional on X."

  TRIGGER 2: CONFIRMATION — new observation matches a prior conditional/plan
  visible in active state or recent log.
    Cognition fact text: "User's prior plan to do Y is confirmed by the
    Z observation."

  TRIGGER 3: CONTRADICTION — new observation contradicts a prior expectation.
    Cognition fact text: "User's prior expectation about X is contradicted
    by the Y observation."

  TRIGGER 4: NAMED HOPE/FEAR — User explicitly states a hope or fear.
    Cognition fact text: "User hopes for X." OR "User is worried about Y."

DO NOT emit:
  - Trivial restatements of the observation itself
  - Generic "User now knows X" entries
  - Speculation without an explicit trigger above
  - Cognition entries that duplicate factual claims (those belong in observations)

ENTITY RESOLUTION: same rules as observations. Use entity_ids from ACTIVE
ENTITIES when applicable, otherwise "new".

ACTIVE ENTITIES (observation chain heads — use these for resolves_to):
{active_entities}

RECENT LOG (most recent committed entries):
{recent_facts}

NEW OBSERVATIONS just emitted (the trigger for this cognition pass):
{new_observations}

CONVERSATION WINDOW (for context):
{window_block}

Schema (same as writer):
{{
  "facts": [
    {{
      "turn": <int from TARGET turns>,
      "text": "<atomic prose sentence about User's mental state>",
      "mentions": [
        {{"surface": "<literal text>", "resolves_to": "<entity_id or 'new'>"}}
      ]
    }}
  ]
}}

Default output is `{{"facts": []}}`. Emit non-empty ONLY when a trigger above
clearly fires.
"""


def cognize_window(
    target_turns,
    new_observations: list[Fact],
    obs_mentions_by_id: dict[str, Mention],
    prior_log_facts: list[Fact],
    obs_idx: IndexedCollection | None,
    registry: EntityRegistry,
    window_block: str,
    cache: Cache,
    budget: Budget,
) -> tuple[list[Fact], list[Mention], list[BindingEvent], dict]:
    """Emit cognition facts on top of fresh observations. Same DSU operations on registry."""
    if not new_observations:
        return [], [], [], {"n_cognitions_emitted": 0}

    target_surfaces = extract_window_surfaces(target_turns)
    for f in new_observations:
        for mid in f.mention_ids:
            m = obs_mentions_by_id.get(mid)
            if m:
                target_surfaces.add(m.surface)
    active_entities = render_active_entities(obs_idx, registry, target_surfaces)
    recent_facts_str = render_recent_facts(prior_log_facts, cap=8)

    obs_lines = []
    for f in new_observations:
        mention_summaries = []
        for mid in f.mention_ids:
            m = obs_mentions_by_id.get(mid)
            if not m:
                continue
            eid = registry.get_canonical(mid)
            mention_summaries.append(f"{m.surface}->{eid}")
        ms = ", ".join(mention_summaries) if mention_summaries else "(no mentions)"
        obs_lines.append(f"  [{f.fact_uuid} t={f.ts}] {f.text}  ({ms})")
    new_observations_str = "\n".join(obs_lines)

    prompt = COGNIZE_PROMPT.format(
        active_entities=active_entities,
        recent_facts=recent_facts_str,
        new_observations=new_observations_str,
        window_block=window_block,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    telemetry = {"cognize_prompt_chars": len(prompt)}
    if not isinstance(obj, dict):
        return [], [], [], telemetry

    target_turn_set = {t for t, _ in target_turns}
    facts_raw = obj.get("facts", []) or []
    new_cog_facts: list[Fact] = []
    new_cog_mentions: list[Mention] = []
    new_bindings: list[BindingEvent] = []
    fact_counter = 0

    for fr in facts_raw:
        if not isinstance(fr, dict):
            continue
        text = (fr.get("text") or "").strip()
        if not text:
            continue
        ts_raw = fr.get("turn")
        try:
            ts = int(ts_raw) if ts_raw is not None else target_turns[-1][0]
        except (TypeError, ValueError):
            ts = target_turns[-1][0]
        if ts not in target_turn_set:
            ts = target_turns[-1][0]

        fact_uuid = f"c{ts:04d}_{fact_counter}"  # 'c' prefix for cognition
        fact_counter += 1

        mention_specs = fr.get("mentions") or []
        local_resolutions: list[tuple[str, str]] = []
        intra_resolve: dict[str, str] = {}

        for i, ms in enumerate(mention_specs):
            if not isinstance(ms, dict):
                continue
            surface = (ms.get("surface") or "").strip()
            if not surface:
                continue
            resolves_to = (ms.get("resolves_to") or "new").strip()
            mention_id = f"cm{ts:04d}_{fact_counter}_{i}"  # 'cm' for cognition mention
            new_cog_mentions.append(
                Mention(
                    mention_id=mention_id,
                    surface=surface,
                    fact_uuid=fact_uuid,
                    ts=ts,
                )
            )
            registry.register(mention_id)
            local_resolutions.append((mention_id, resolves_to))

        for mention_id, resolves_to in local_resolutions:
            if resolves_to == "new" or not resolves_to:
                continue
            if resolves_to in registry.entity_members:
                existing_member = next(iter(registry.entity_members[resolves_to]))
                registry.merge(
                    mention_id,
                    existing_member,
                    ts=ts,
                    evidence_fact_uuids=[fact_uuid],
                    rationale=f"cognizer asserted resolves_to={resolves_to}",
                )
                new_bindings.append(registry.binding_events[-1])
            elif resolves_to in intra_resolve:
                registry.merge(
                    mention_id,
                    intra_resolve[resolves_to],
                    ts=ts,
                    evidence_fact_uuids=[fact_uuid],
                    rationale="intra-fact coref (cognition)",
                )
                new_bindings.append(registry.binding_events[-1])
            else:
                intra_resolve[resolves_to] = mention_id

        new_cog_facts.append(
            Fact(
                fact_uuid=fact_uuid,
                ts=ts,
                text=text,
                mention_ids=[m for m, _ in local_resolutions],
                collection="cognition",
            )
        )

    telemetry["n_cognitions_emitted"] = len(new_cog_facts)
    return new_cog_facts, new_cog_mentions, new_bindings, telemetry


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


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
    "User",
    "Users",
    "List",
    "Did",
    "Current",
    "It",
    "That",
    "This",
    "In",
    "On",
    "At",
    "About",
}


def extract_question_surfaces(question: str) -> list[str]:
    q = re.sub(r"[^a-zA-Z0-9\s']", " ", question)
    words = q.split()
    surfaces = []
    for w in words:
        if w.endswith("'s"):
            w = w[:-2]
        elif w.endswith("'"):
            w = w[:-1]
        if len(w) > 1 and w[0].isupper() and w not in STOP:
            surfaces.append(w)
    return surfaces + ["User"]


def _rank_by_embedding(q_emb, candidate_uuids, embed_by_uuid, top_k):
    scored = []
    for u in candidate_uuids:
        v = embed_by_uuid.get(u)
        if v is None:
            continue
        scored.append((cosine(q_emb, v), u))
    scored.sort(reverse=True)
    return [u for _, u in scored[:top_k]]


COGNITIVE_QUESTION_KEYWORDS = {
    "expect",
    "expected",
    "anticipate",
    "plan",
    "intend",
    "intended",
    "intention",
    "believe",
    "believed",
    "think",
    "thought",
    "fear",
    "worried",
    "concern",
    "hope",
    "wish",
    "wished",
    "predict",
    "predicted",
    "feel",
    "felt",
    "end up",
    "ended up",
    "actually buy",
    "actually buying",
    "actually do",
    "actually did",
    "did user end",
    "does user end",
    "would happen",
    "will happen",
    "going to do",
    "going to be",
}


def is_cognitive_question(question: str) -> bool:
    ql = question.lower()
    return any(kw in ql for kw in COGNITIVE_QUESTION_KEYWORDS)


def retrieve(
    question: str,
    store: MemoryStore,
    cache: Cache,
    budget: Budget,
    top_k: int = 14,
    collections: list[str] | None = None,
) -> tuple[list[Fact], dict[str, set[str]]]:
    """Returns (facts, entity_resolution_map) where entity_resolution_map is
    entity_id -> set of mention_ids in that class that appear in retrieved facts."""
    if collections is None:
        collections = ["observations"]
        if is_cognitive_question(question):
            collections = ["observations", "cognition"]
    q_emb = embed_batch([question], cache, budget)[0]
    surfaces = extract_question_surfaces(question)
    norm_surfaces = [_normalize_surface(s) for s in surfaces]

    selected_facts: dict[str, Fact] = {}

    for col_name in collections:
        idx = store.collections.get(col_name)
        if idx is None or not idx.facts:
            continue

        # 1. Surface-match: find mention_ids by surface, get their entities, get those entities' facts
        for nsurf in norm_surfaces:
            for surface_norm, mids in idx.mentions_by_surface.items():
                if not nsurf:
                    continue
                if nsurf in surface_norm or surface_norm in nsurf:
                    for mid in mids:
                        eid = store.registry.get_canonical(mid)
                        for fu in idx.facts_by_entity.get(eid, []):
                            f = idx.by_uuid.get(fu)
                            if f:
                                selected_facts[fu] = f

        # 2. Embedding kNN over all facts in collection
        all_uuids = [f.fact_uuid for f in idx.facts]
        for fu in _rank_by_embedding(q_emb, all_uuids, idx.embed_by_uuid, top_k=top_k):
            f = idx.by_uuid[fu]
            selected_facts[fu] = f

    facts = sorted(selected_facts.values(), key=lambda f: (f.ts, f.fact_uuid))
    if len(facts) > 60:
        facts = facts[-60:]

    # Build resolution map for retrieved facts
    resolution_map: dict[str, set[str]] = {}
    for f in facts:
        idx = store.collections.get(f.collection)
        if idx is None:
            continue
        for mid in f.mention_ids:
            eid = store.registry.get_canonical(mid)
            resolution_map.setdefault(eid, set()).add(mid)

    return facts, resolution_map


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


READ_PROMPT = """You are answering a question using a prose-fact memory.

Each fact is an atomic prose sentence with mentions of entities. Each mention
has a mention_id (per-occurrence) and a canonical entity_id. Multiple mentions
that resolve to the same entity_id refer to the SAME real entity (e.g.,
mentions of "the new boss" and "Marcus" might both resolve to entity e_a01,
indicating they're the same person).

ENTITY RESOLUTION (canonical entities and the mentions that resolve to them):
{resolution_map}

RETRIEVED FACTS (chronological):
{facts_block}

QUESTION: {question}

When mentions in different facts share the same entity_id, treat them as the
same entity. When a mention's entity has surfaces like ["the new boss",
"Marcus"], the entity is named Marcus and is User's boss.

Answer concisely. For yes/no questions, start with "Yes" or "No".
"""


def format_facts_for_read(facts: list[Fact], store: MemoryStore) -> str:
    lines = []
    for f in facts:
        # Annotate the fact text with mention -> entity_id resolutions
        mention_resolutions = []
        for mid in f.mention_ids:
            eid = store.registry.get_canonical(mid)
            idx = store.collections.get(f.collection)
            surface = "?"
            if idx and mid in idx.mentions_by_id:
                surface = idx.mentions_by_id[mid].surface
            mention_resolutions.append(f"  [{mid}|{surface!r}->{eid}]")
        meta = f"col={f.collection}"
        if mention_resolutions:
            meta += "\n  mentions:\n" + "\n".join(mention_resolutions)
        lines.append(f"[{f.fact_uuid}] t{f.ts} :: {f.text}  ({meta})")
    return "\n".join(lines)


def format_resolution_map(
    resolution_map: dict[str, set[str]],
    store: MemoryStore,
) -> str:
    if not resolution_map:
        return "(no entities)"
    lines = []
    for eid, mids in sorted(resolution_map.items(), key=lambda kv: kv[0]):
        # Collect all surfaces across mentions
        surfaces = set()
        for mid in mids:
            for col in store.collections.values():
                if mid in col.mentions_by_id:
                    surfaces.add(col.mentions_by_id[mid].surface)
        surfaces_list = sorted(surfaces)
        lines.append(f"  {eid}: surfaces={surfaces_list}  mentions={sorted(mids)}")
    return "\n".join(lines)


def answer_question(
    question: str, store: MemoryStore, cache: Cache, budget: Budget, top_k: int = 14
) -> str:
    facts, resolution_map = retrieve(question, store, cache, budget, top_k=top_k)
    facts_block = format_facts_for_read(facts, store)
    resolution_block = format_resolution_map(resolution_map, store)
    prompt = READ_PROMPT.format(
        resolution_map=resolution_block,
        facts_block=facts_block,
        question=question,
    )
    return llm(prompt, cache, budget).strip()


# ---------------------------------------------------------------------------
# Ingestion driver — K=3 centered window
# ---------------------------------------------------------------------------


def _render_window_block(window_turns, target_turn_lo):
    lines = []
    in_target = False
    for tidx, text in window_turns:
        if not in_target and tidx >= target_turn_lo:
            lines.append("--- TARGET TURNS ---")
            in_target = True
        prefix = "  TARGET" if in_target else "  CONTEXT"
        lines.append(f"{prefix} TURN {tidx}: {text}")
    if not in_target:
        lines.insert(0, "--- TARGET TURNS ---")
    return "\n".join(lines)


def ingest_turns(
    turns,
    cache,
    budget,
    *,
    w_past: int = 7,
    w_future: int = 7,
    k: int = 3,
    rebuild_index_every: int = 4,
    enable_cognition: bool = True,
):
    obs_facts: list[Fact] = []
    obs_mentions: list[Mention] = []
    cog_facts: list[Fact] = []
    cog_mentions: list[Mention] = []
    store = MemoryStore()
    # Pre-register the User entity so writer can resolve to "e_user"
    store.registry.register("m_user_root")
    store.registry.mention_to_entity["m_user_root"] = "e_user"
    store.registry.entity_members["e_user"] = {"m_user_root"}
    store.registry.entity_members.pop("e_m_user_root", None)

    telemetry = []
    n_turns = len(turns)
    fire_no = 0
    target_lo = 0
    while target_lo < n_turns:
        target_hi = min(n_turns, target_lo + k)
        win_lo = max(0, target_lo - w_past)
        win_hi = min(n_turns, target_hi + w_future)
        window_turns = turns[win_lo:win_hi]
        target_turns = turns[target_lo:target_hi]
        if not target_turns:
            break
        target_turn_lo = target_turns[0][0]

        obs_idx = store.collections.get("observations")
        new_facts, new_mentions, _bindings, tele = write_window(
            window_turns,
            target_turn_lo,
            target_turns,
            obs_facts,
            obs_idx,
            store.registry,
            cache,
            budget,
        )
        obs_facts.extend(new_facts)
        obs_mentions.extend(new_mentions)

        # Cognition pass
        new_cog: list[Fact] = []
        new_cog_m: list[Mention] = []
        c_tele = {"n_cognitions_emitted": 0}
        if enable_cognition and new_facts:
            obs_mentions_by_id = {m.mention_id: m for m in obs_mentions}
            window_block = _render_window_block(window_turns, target_turn_lo)
            new_cog, new_cog_m, _cb, c_tele = cognize_window(
                target_turns,
                new_facts,
                obs_mentions_by_id,
                obs_facts + cog_facts,
                obs_idx,
                store.registry,
                window_block,
                cache,
                budget,
            )
            cog_facts.extend(new_cog)
            cog_mentions.extend(new_cog_m)

        tele["fire_no"] = fire_no
        tele["last_turn"] = target_turns[-1][0]
        tele.update(c_tele)
        telemetry.append(tele)

        if fire_no % rebuild_index_every == 0:
            store.collections["observations"] = build_collection(
                "observations",
                obs_facts,
                obs_mentions,
                store.registry,
                cache,
                budget,
            )
            if enable_cognition:
                store.collections["cognition"] = build_collection(
                    "cognition",
                    cog_facts,
                    cog_mentions,
                    store.registry,
                    cache,
                    budget,
                )
        fire_no += 1
        target_lo = target_hi

    store.collections["observations"] = build_collection(
        "observations",
        obs_facts,
        obs_mentions,
        store.registry,
        cache,
        budget,
    )
    if enable_cognition:
        store.collections["cognition"] = build_collection(
            "cognition",
            cog_facts,
            cog_mentions,
            store.registry,
            cache,
            budget,
        )
    return obs_facts, obs_mentions, cog_facts, cog_mentions, store, telemetry
