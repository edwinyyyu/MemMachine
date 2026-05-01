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


def render_active_mentions(
    obs_idx: IndexedCollection | None,
    registry: EntityRegistry,
    target_surfaces: set[str],
    max_groups: int = 20,
    max_mentions_per_group: int = 4,
    excerpts_per_group: int = 2,
) -> str:
    """Render prior mentions visible to the writer.

    Mentions are grouped by their current DSU equivalence class for visual
    convenience, but cluster ids are render-local aliases (Group A, Group B)
    — never the global entity_id. The writer reasons about per-occurrence
    mention_ids; the cluster grouping is just a hint.
    """
    if obs_idx is None:
        return "(empty)"

    # Group mention_ids by current canonical entity_id (system-internal),
    # but never expose the entity_id label to the prompt.
    classes: dict[str, dict] = {}
    norm_targets = {_normalize_surface(s) for s in target_surfaces}
    for surface_norm, mids in obs_idx.mentions_by_surface.items():
        surface_match = any(
            t in surface_norm or surface_norm in t for t in norm_targets if t
        )
        for mid in mids:
            eid = registry.get_canonical(mid)
            if eid == "e_user":
                continue  # User self-reference, handled implicitly
            cls = classes.setdefault(
                eid, {"latest_ts": 0, "matched": False, "mids": []}
            )
            m = obs_idx.mentions_by_id[mid]
            cls["latest_ts"] = max(cls["latest_ts"], m.ts)
            if surface_match:
                cls["matched"] = True
            cls["mids"].append(mid)

    if not classes:
        return "(none)"

    ordered = sorted(
        classes.items(),
        key=lambda kv: (-int(kv[1]["matched"]), -kv[1]["latest_ts"]),
    )

    def _alias(idx: int) -> str:
        # A, B, ..., Z, AA, AB, ...
        s = ""
        n = idx
        while True:
            s = chr(ord("A") + (n % 26)) + s
            n = n // 26 - 1
            if n < 0:
                break
        return s

    # Detect surface collisions: which surfaces appear in more than one
    # group? These are the disambiguation hot-spots.
    eid_alias: dict[str, str] = {
        eid: _alias(i) for i, (eid, _) in enumerate(ordered[:max_groups])
    }
    surface_to_groups: dict[str, set[str]] = {}
    for eid, info in ordered[:max_groups]:
        for mid in set(info["mids"]):
            m = obs_idx.mentions_by_id.get(mid)
            if not m:
                continue
            surface_to_groups.setdefault(m.surface, set()).add(eid_alias[eid])
    collisions = {
        surface: sorted(group_set)
        for surface, group_set in surface_to_groups.items()
        if len(group_set) > 1
    }

    lines = []
    if collisions:
        lines.append(
            "  ⚠ COLLIDING SURFACES (same surface in multiple groups — DISAMBIGUATE via TARGET modifier; if no modifier matches a group's context, use 'new'):"
        )
        # For each colliding surface, render each group's distinguishing
        # excerpt inline so the writer can compare side-by-side.
        for surface, gs in sorted(collisions.items()):
            lines.append(f'      "{surface}" appears in:')
            for g_alias in gs:
                # Find the eid for this alias
                target_eid = next(
                    (eid for eid, a in eid_alias.items() if a == g_alias),
                    None,
                )
                if target_eid is None:
                    continue
                # Pick the most-recent fact text in this group as the
                # distinguishing excerpt.
                info = classes[target_eid]
                latest_mid = max(
                    set(info["mids"]),
                    key=lambda mid: obs_idx.mentions_by_id[mid].ts,
                )
                m = obs_idx.mentions_by_id[latest_mid]
                fact = obs_idx.by_uuid.get(m.fact_uuid)
                excerpt = fact.text[:90] if fact else ""
                lines.append(f'         Group {g_alias}: "{excerpt}"')
        lines.append("")

    for i, (eid, info) in enumerate(ordered[:max_groups]):
        alias = _alias(i)
        # Order mentions in the class by ts; cap to most recent
        mids_sorted = sorted(
            set(info["mids"]),
            key=lambda mid: obs_idx.mentions_by_id[mid].ts,
        )[-max_mentions_per_group:]
        mention_lines = []
        for mid in mids_sorted:
            m = obs_idx.mentions_by_id[mid]
            # Per-mention disambiguator: the fact text that contains this
            # specific occurrence. Crucial when multiple mentions share a
            # surface (e.g., several Alices) — gives the writer concrete
            # discriminating context per mention without bloating the tag.
            fact = obs_idx.by_uuid.get(m.fact_uuid)
            ctx = fact.text[:100] if fact else ""
            mention_lines.append(
                f'      {mid}  surface="{m.surface}"  (turn {m.ts}) — "{ctx}"'
            )
        block = "\n".join(mention_lines)
        lines.append(f"  Group {alias}:\n{block}")
    return "\n".join(lines)


# Back-compat alias — ingest path imports this name.
render_active_entities = render_active_mentions


def render_recent_facts(prior_facts: list[Fact], cap: int = 8) -> str:
    if not prior_facts:
        return "(empty)"
    recent = list(reversed(prior_facts[-cap:]))
    lines = [f"  [{f.fact_uuid} t={f.ts}] {f.text[:120]}" for f in recent]
    return "\n".join(lines)


def render_working_memory(
    prior_facts: list[Fact],
    obs_idx: IndexedCollection | None,
    registry: EntityRegistry,
    last_n: int = 7,
) -> str:
    """Recent context in chronological order (oldest → newest), with each
    referenced entity tagged with its most-recent KNOWN NAME (or '(unnamed)'
    if no proper-name surface has been bound to it yet).

    Working-memory analog: what's currently "in focus" — the recent thread
    of the conversation. Pronouns and bare descriptors typically refer to
    something recently in working memory, especially the most recent UNNAMED
    entity. Distinct from ACTIVE MENTIONS (which is the long-term store).
    """
    if not prior_facts:
        return "(empty)"
    recent = prior_facts[-last_n:]

    def _is_proper_name(s: str) -> bool:
        # Simple heuristic: starts with capital, no leading "the/a/my/...",
        # and not the User/I pronoun.
        if not s:
            return False
        first = s.split(maxsplit=1)[0]
        if first.lower() in (
            "the",
            "a",
            "an",
            "my",
            "your",
            "their",
            "his",
            "her",
            "our",
            "i",
            "user",
        ):
            return False
        return first[:1].isupper() and first != "User"

    def _entity_name(eid: str) -> str:
        if eid == "e_user":
            return "User"
        # Get the most recent proper-name surface in this entity's class
        if obs_idx is None:
            return "(unnamed)"
        members = registry.entity_members.get(eid, set())
        named_surfaces: list[tuple[int, str]] = []
        descriptor_surfaces: list[tuple[int, str]] = []
        for mid in members:
            m = obs_idx.mentions_by_id.get(mid)
            if not m:
                continue
            if _is_proper_name(m.surface):
                named_surfaces.append((m.ts, m.surface))
            else:
                descriptor_surfaces.append((m.ts, m.surface))
        if named_surfaces:
            named_surfaces.sort(reverse=True)
            return named_surfaces[0][1]
        if descriptor_surfaces:
            descriptor_surfaces.sort(reverse=True)
            return f"(unnamed: {descriptor_surfaces[0][1]})"
        return "(unnamed)"

    lines = []
    for f in recent:
        ent_labels: list[str] = []
        seen_eids: set[str] = set()
        for mid in f.mention_ids:
            eid = registry.get_canonical(mid)
            if eid in seen_eids:
                continue
            seen_eids.add(eid)
            ent_labels.append(_entity_name(eid))
        ent_str = ", ".join(ent_labels) if ent_labels else "—"
        lines.append(f"  [t={f.ts}] {f.text[:120]}   ↳ refs: {ent_str}")
    return "\n".join(lines)


def extract_window_surfaces(turns: list[tuple[int, str]]) -> set[str]:
    text = " ".join(t for _, t in turns)
    tokens = re.findall(r"\b([A-Z][a-z]{1,30})\b", text)
    return set(tokens) | {"User"}


def annotate_context_turn(
    turn_ts: int,
    text: str,
    mentions_by_ts: dict[int, list[Mention]],
    registry: EntityRegistry,
) -> str:
    """Inject `[mention_id]` tags after surface mentions in a prior turn.

    Tags expose the per-occurrence MENTION id (entity-in-context), not the
    global entity_id (concrete-entity). The writer can propose co-reference
    by referring to a specific prior mention; the DSU merge happens in the
    registry, never as a label baked into the prompt.

    First-occurrence per surface, longer surfaces first, no overlap, skip the
    User mention (too common to be useful, only adds noise).
    """
    mns = mentions_by_ts.get(turn_ts, [])
    if not mns:
        return text
    mns_sorted = sorted(mns, key=lambda m: -len(m.surface))
    used: list[tuple[int, int]] = []
    inserts: list[tuple[int, str]] = []
    low = text.lower()
    for m in mns_sorted:
        s = m.surface
        if not s:
            continue
        # Skip the canonical User mention (the system's pre-registered self
        # reference). Real user-mentions emitted by the writer with surface
        # like "I"/"my" are kept — they get their own mention_ids.
        if m.mention_id == "m_user_root":
            continue
        slo = s.lower()
        idx = low.find(slo)
        while idx >= 0:
            end = idx + len(s)
            overlap = any(rs < end and idx < re for rs, re in used)
            if not overlap:
                used.append((idx, end))
                inserts.append((end, f"[{m.mention_id}]"))
                break
            idx = low.find(slo, idx + 1)
    inserts.sort(key=lambda x: -x[0])
    out = text
    for pos, tag in inserts:
        out = out[:pos] + tag + out[pos:]
    return out


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
  - Consumption events ONLY IF they CONTRADICT a CLEAR PRIOR COMMITMENT
    visible in ACTIVE ENTITIES or RECENT FACTS (e.g., active state shows
    "User is vegetarian for 8 years" and target says "had a steak" → emit
    the steak. Active state shows "User doesn't drink coffee" and target
    says "downed espresso" → emit the espresso). Otherwise SKIP — generic
    "had coffee", "had pizza", "stomach hurts" are filler.
  - SPECIFIC IMPACTFUL EVENTS — when User describes a concrete event
    with concrete content, capture the EVENT AS USER DESCRIBED IT
    (preserve specifics, don't paraphrase the meaning away):
      "Deploy is broken AGAIN, third time this week." → keep specifics
      "WE GOT A DOG named Daisy, golden retriever puppy." → keep
      "Recruiter pushed the interview to next week. Internally screaming."
        → keep both the event and the tone-laden quote
    Don't categorize ("user was frustrated"). Don't summarize away
    quotation/tone. The READER will classify emotion / aggregate /
    attribute at retrieval time from raw events.
  - Reports of what OTHER people think, plan, or believe ("Marcus says the
    deploy is fine"). Capture as the literal report.
  - STRONG COMMITMENTS or never/always claims that establish a position:
    "I'd never move out of NYC", "I always do X by hand", "I'm never doing
    Y again", "I've been vegetarian for 8 years". These are direct
    statements; capture verbatim. Do NOT pre-classify as "contradiction
    candidate" — just store the claim.
  - PREFERENCE STATEMENTS — capture each one as the user stated it:
    "I prefer light roast — dark tastes burnt" → keep
    "Tried a dark roast at the new cafe — converted." → keep
    DO NOT synthesize "User now prefers dark roast (was light roast)" —
    that's a reader-side derivation. Two separate preference statements
    let the reader compare and infer transition.
  - NUMERICAL FACTS — when User states a specific number tied to a
    stateful attribute (weight 175 lb, salary $120k, 3,200 steps yesterday,
    18,000 steps today), capture the specific number AND its context AS
    STATED. Do not pre-classify as "extreme" or "PR" — the reader judges
    extremes when asked. Do skip generic recurring stats ("hit 10K steps,
    same as my goal" — only durable if outlier or claim-grounding).

DO NOT EMIT (skip silently — output empty facts for filler-only TARGET turns):
  - Body sensations alone: "stomach hurts", "tired", "need a nap"
  - Weather: "rainy day", "weather is nice"
  - Transient feelings WITHOUT cause: "long day", "slow afternoon"
  - Generic ambient activity: "going to lunch", "on calls all morning",
    "had coffee" (without specific kind or context)
  - Notification noise: "412 unread", "email avalanche", "Slack laggy"
  - Generic chitchat with no durable content

PREFER ONE FACT PER TURN. Multiple facts only if a turn bundles unrelated
durable events. Aim for HIGH SIGNAL DENSITY — short list of important facts.

CO-REFERENCE BY MENTION_ID — CRITICAL:

Each mention has:
  - surface: the literal text token (e.g., "I", "Marcus", "the new boss")
  - resolves_to: a PRIOR mention_id this mention is co-referent with,
    OR "user" for the User/speaker self-reference,
    OR "new" if it's a fresh mention with no co-referent prior.

CONTEXT TURNS show prior mentions tagged inline like this:
  `CONTEXT TURN 5: my new manager[m0005_2] started today`
The tag `[m0005_2]` is the MENTION_ID of "new manager" at turn 5. Treat
tags as inline metadata — NOT spoken dialogue.

HOW TO BIND CO-REFERENCE:
  - resolves_to encodes IDENTITY co-reference: "this mention refers to the
    SAME entity as that prior mention".
  - DO bind: pronouns ("he/she/they/it"), descriptors with definite
    article ("the boss", "the puppy"), or a name that IDENTIFIES a
    previously-anonymous or under-specified prior entity (the naming
    bind). Pronouns and naming events are the strongest identity signals.
  - When the prior descriptor is in ACTIVE MENTIONS but NOT in the
    CONTEXT window (too far back), still bind via the mention_id from
    ACTIVE MENTIONS. ACTIVE MENTIONS is your long-range binding source.
  - Two mentions in the SAME fact can share resolves_to when they refer
    to the same entity in that fact.

  - DO NOT bind when TARGET introduces a DIFFERENT entity that simply
    fills the same role/slot/position as a prior entity (e.g., a successor
    named explicitly as a different person REPLACING someone). Use "new"
    for the successor; the original entity stays bound to its prior
    mention_id.

NAMED SURFACES (CAPITALIZED FIRST NAMES) — DEFAULT TO "new" UNLESS YOU HAVE
POSITIVE EVIDENCE OF SAME-PERSON CONTINUITY:
  - First names like "Alice", "Bob", "Marcus", "Pat" do NOT identify a
    unique person across the conversation. Multiple distinct people can
    share a name. Recurrence of a name surface is NOT by itself evidence
    that the same person is meant.
  - Bind a named surface to a prior mention_id ONLY when ONE of these
    holds:
      (a) The TARGET fact ALSO contains a co-referring pronoun or
          descriptor that already binds to that prior mention's entity
          (e.g., "the new manager Quentin" where "the new manager"
          binds to a prior unnamed-manager mention).
      (b) The TARGET surface has a MODIFIER that matches the prior
          entity's discriminating context in ACTIVE MENTIONS (e.g.,
          "Alice from the platform team" matches a prior Alice with
          platform-team context).
      (c) The TARGET fact's content is the natural continuation of a
          prior fact about that named entity (e.g., the prior fact says
          "Marcus moved to a different team" and the TARGET says
          "Marcus mentioned he likes the new team" — same Marcus).
  - If none of (a), (b), (c) apply, use "new" — the named entity in
    TARGET may be a different person sharing the same name. Better to
    create a new entity than to wrongly merge distinct people.

  - For the User/speaker (any "I", "my", "me", "User"), use
    resolves_to="user".
  - If no prior mention plausibly matches the SAME-IDENTITY criterion,
    use "new".

DISAMBIGUATION when several prior mentions share a surface:
  - When the TARGET surface matches multiple ACTIVE MENTIONS groups,
    do NOT just pick the most recent or most prominent inline tag.
  - The ACTIVE MENTIONS section flags any "COLLIDING SURFACES" at the
    top — these are the names where multiple distinct entities exist.
    For TARGET mentions of a colliding surface, you MUST disambiguate
    via modifier before binding.
  - READ THE MODIFIER on the TARGET surface — any qualifier that narrows
    which entity is meant (a relation, a role, a place, an attribute).
  - Cross-reference each candidate group's per-mention context strings in
    ACTIVE MENTIONS. Match the TARGET modifier to the group whose context
    is consistent with that modifier.
  - Use a mention_id ONLY from the group whose context matches the
    modifier. If no group's context matches, use "new".

PRONOUN ANTECEDENTS:
  - When TARGET starts with or contains a bare pronoun ("he/she/they/it/
    his/her/their") with no clear in-window antecedent, the antecedent
    is most likely the most recent UNNAMED entity in WORKING MEMORY
    (an entity whose name hasn't been revealed yet — labeled
    "(unnamed: <descriptor>)" in WORKING MEMORY).
  - When TARGET names that entity ("his name is X", "she's X"), this is
    a naming bind: resolve_to the prior mention_id of the unnamed entity.

WORKING MEMORY (recent context — short-term, what's currently in focus):
{working_memory}
  - Each line shows a recent fact, with the named/unnamed entities it
    references. Unnamed entities ("(unnamed: <descriptor>)") are the
    most likely antecedents for TARGET pronouns and bare descriptors.
  - This is the SHORT-TERM thread of the conversation. ACTIVE MENTIONS
    below is the longer-term store.

ACTIVE MENTIONS (prior mentions, grouped by likely-same-entity):
{active_entities}
  - "Group A/B/..." headers are render-local hints — NOT ids you can use.
    Each line shows: mention_id, surface, turn, and the FACT TEXT that
    contains this occurrence — that fact text is the disambiguator.
    To bind, use the specific mention_id (e.g., m0005_2). If you
    disagree with a grouping, bind to the single mention_id you mean.

WITHIN-FIRE SHARED ENTITIES (NEW anonymous entities mentioned in MULTIPLE
facts of YOUR CURRENT OUTPUT):
  - When a fresh anonymous entity (no prior mention_id, would normally be
    "new") appears in multiple facts you're emitting in this batch, use a
    SHARED PLACEHOLDER for resolves_to in all those mentions. Pick any
    short label starting with "_" (e.g., "_anon_dog", "_anon_friend").
  - First mention with that placeholder anchors a fresh entity; later
    mentions with the SAME placeholder merge into it.
  - Use this whenever the same anonymous entity is referenced more than
    once across your emitted facts (descriptor in one fact, pronoun in
    another, name reveal in a third). Without a shared placeholder, the
    system splits them into separate entities.
  - Do NOT use placeholders if the entity already has a prior mention_id
    (use that mention_id instead).

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
        {{"surface": "<literal text token>", "resolves_to": "<mention_id | 'user' | '_anon_<label>' | 'new'>"}}
      ]
    }}
  ]
}}

EXAMPLES (illustrative — names and domains are placeholders):

(1) Filler turn (no durable content): output `{{"facts": []}}`.

(2) Anonymous descriptor with no relevant prior:
  text: "User picked up a new instrument this week."
  mentions: [
    {{"surface": "User", "resolves_to": "user"}},
    {{"surface": "new instrument", "resolves_to": "new"}}
  ]

(3) Earlier descriptor named in a later turn (identity bind across turns):
  CONTEXT TURN k: "User adopted a new puppy[m_p] from the shelter."
  TARGET TURN n: "We finally settled on a name — Mochi."
  text: "User's puppy is named Mochi."
  mentions: [
    {{"surface": "User",  "resolves_to": "user"}},
    {{"surface": "puppy", "resolves_to": "m_p"}},    # same entity
    {{"surface": "Mochi", "resolves_to": "m_p"}}     # naming bind — same entity
  ]
  (Replace m_p with the real mention_id from CONTEXT/ACTIVE MENTIONS.)

(3b) Same FRESH entity referenced in MULTIPLE facts of the same emission
     (within-fire placeholder):
  TARGET TURN n: "A new tenant moved in next door — works as a vet, named
                  Robin."
  Two facts emitted, both refer to the same fresh anonymous entity:
  facts: [
    {{
      "text": "A new tenant moved in next door and works as a vet.",
      "mentions": [
        {{"surface": "User",       "resolves_to": "user"}},
        {{"surface": "new tenant", "resolves_to": "_anon_tenant"}}
      ]
    }},
    {{
      "text": "User's new neighbor is named Robin.",
      "mentions": [
        {{"surface": "User",     "resolves_to": "user"}},
        {{"surface": "neighbor", "resolves_to": "_anon_tenant"}},  # same entity
        {{"surface": "Robin",    "resolves_to": "_anon_tenant"}}   # naming bind
      ]
    }}
  ]
  (Both "new tenant" and "neighbor" and "Robin" share resolves_to
   "_anon_tenant" → all bound to one fresh entity. Without the shared
   placeholder these would split into three separate entities.)

(4) Same surface, DIFFERENT entity (modifier disambiguation):
  ACTIVE MENTIONS Group A: m_a surface="Riley" — "Riley taught User the
                            mandolin chord progression"
  TARGET TURN n: "Riley from the trail-running club brought a casserole."
  Modifier ("from the trail-running club") does NOT match the music context
  in Group A → fresh mention (different Riley):
  mentions: [
    {{"surface": "User",  "resolves_to": "user"}},
    {{"surface": "Riley", "resolves_to": "new"}}
  ]

(5) Same role/slot, DIFFERENT entity (no identity bind):
  ACTIVE MENTIONS Group A: m_c surface="Casey" — "Casey hosted the
                            weekly book club"
  TARGET TURN n: "Casey is stepping down from book club; Jordan is taking
                  over as host next week."
  Casey's mention DOES bind to Group A (same person). Jordan is a NEW
  person filling Casey's prior role — does NOT bind to Casey:
  mentions: [
    {{"surface": "User",   "resolves_to": "user"}},
    {{"surface": "Casey",  "resolves_to": "m_c"}},
    {{"surface": "Jordan", "resolves_to": "new"}}    # different person, same role
  ]
  (Replace m_c with the real Casey mention_id from ACTIVE MENTIONS.)

(6) Conditional / hypothetical:
  text: "If User joins the climbing gym, User's belay partner will be Avery."
  mentions: [
    {{"surface": "User",  "resolves_to": "user"}},
    {{"surface": "climbing gym", "resolves_to": "<m_id of any prior gym mention, else 'new'>"}},
    {{"surface": "Avery", "resolves_to": "new"}}
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
    *,
    inline_anchors: bool = False,
    all_mentions: list[Mention] | None = None,
) -> tuple[list[Fact], list[Mention], list[BindingEvent], dict]:
    """Emit facts (with mentions) for TARGET turns. Apply binding decisions to registry."""
    target_surfaces = extract_window_surfaces(target_turns)
    active_entities = render_active_entities(obs_idx, registry, target_surfaces)
    working_memory = render_working_memory(prior_facts, obs_idx, registry, last_n=7)
    recent_facts_str = render_recent_facts(prior_facts, cap=8)

    mentions_by_ts: dict[int, list[Mention]] = {}
    if inline_anchors and all_mentions:
        for m in all_mentions:
            mentions_by_ts.setdefault(m.ts, []).append(m)

    window_lines = []
    in_target = False
    for tidx, text in window_turns:
        if not in_target and tidx >= target_turn_lo:
            window_lines.append(
                "--- TARGET TURNS (emit facts for these) ---\n"
                "Before deciding resolves_to for any mention here, scan ACTIVE\n"
                "MENTIONS above. If TARGET uses a pronoun, a descriptor, or names\n"
                "an entity that ACTIVE MENTIONS already lists (even from many\n"
                "turns ago), bind to that mention_id — do NOT default to 'new'.\n"
                "Recurring entities should reuse their earliest matching\n"
                "mention_id."
            )
            in_target = True
        prefix = "  TARGET" if in_target else "  CONTEXT"
        rendered = text
        if inline_anchors and not in_target:
            rendered = annotate_context_turn(tidx, text, mentions_by_ts, registry)
        window_lines.append(f"{prefix} TURN {tidx}: {rendered}")
    if not in_target:
        window_lines.insert(0, "--- TARGET TURNS ---")
    window_block = "\n".join(window_lines)

    prompt = WRITE_PROMPT.format(
        active_entities=active_entities,
        working_memory=working_memory,
        recent_facts=recent_facts_str,
        window_block=window_block,
    )
    raw = llm(prompt, cache, budget, reasoning_effort="medium")
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

    # PER-FIRE placeholder map: lets the writer use a shared anonymous
    # name (e.g., "_anon_neighbor") across MULTIPLE FACTS in the same fire.
    # First mention with that placeholder becomes the anchor; subsequent
    # mentions with the same placeholder bind to it via DSU. Resolves the
    # within-fire cross-fact split (e.g., "neighbor" in fact 1 and fact 2
    # of the same fire being split into two entities).
    fire_placeholders: dict[str, str] = {}

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
        # resolves_to is one of:
        #   "new" / ""  → no merge (fresh mention)
        #   "user"      → bind to canonical User mention (m_user_root)
        #   <mention_id>  → merge with that prior mention's DSU class
        #   (back-compat) <entity_id>  → still accepted, merges with that class
        for mention_id, resolves_to in local_resolutions:
            if resolves_to == "new" or not resolves_to:
                continue

            # User self-reference
            if resolves_to == "user":
                survivor = registry.merge(
                    mention_id,
                    "m_user_root",
                    ts=ts,
                    evidence_fact_uuids=[fact_uuid],
                    rationale="user self-reference",
                )
                new_bindings.append(registry.binding_events[-1])
                continue

            # mention_id reference (preferred path under inline_anchors)
            if resolves_to in registry.mention_to_entity:
                survivor = registry.merge(
                    mention_id,
                    resolves_to,
                    ts=ts,
                    evidence_fact_uuids=[fact_uuid],
                    rationale=f"writer bound to mention {resolves_to}",
                )
                new_bindings.append(registry.binding_events[-1])
                continue

            # back-compat: entity_id reference
            if resolves_to in registry.entity_members:
                existing_member = next(iter(registry.entity_members[resolves_to]))
                survivor = registry.merge(
                    mention_id,
                    existing_member,
                    ts=ts,
                    evidence_fact_uuids=[fact_uuid],
                    rationale=f"writer asserted resolves_to={resolves_to}",
                )
                new_bindings.append(registry.binding_events[-1])
                continue

            # Fire-scoped placeholder coref: writer used a placeholder name
            # (e.g., "_anon_neighbor") shared across mentions in this fire.
            # First occurrence becomes the anchor; subsequent ones merge.
            if resolves_to in fire_placeholders:
                survivor = registry.merge(
                    mention_id,
                    fire_placeholders[resolves_to],
                    ts=ts,
                    evidence_fact_uuids=[fact_uuid],
                    rationale=f"fire placeholder {resolves_to}",
                )
                new_bindings.append(registry.binding_events[-1])
            else:
                fire_placeholders[resolves_to] = mention_id

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

Each fact is an atomic prose sentence with mentions of entities. Mentions
that the system has unified into the same entity refer to the SAME real
entity. The ENTITIES IN RETRIEVED FACTS sidebar lists each distinct entity
under a render-local label (Entity A, Entity B, ...) with its surfaces and
a few discriminating fact excerpts.

The retrieved facts are RAW user observations and statements — not
pre-classified or pre-aggregated. You may need to:
  - INFER emotion from tone/word-choice in the fact text (caps, repeated
    punctuation, hyperbole like "ugh"/"argh"/"WHAT"/"FINALLY"/"broke me")
  - COUNT distinct events when asked "how many" or "how often"
  - SUM or compute quantities when asked totals/deltas
  - DISTINGUISH chosen from rejected alternatives
  - ATTRIBUTE quotes/beliefs to the right person
  - IDENTIFY transitions/changes by comparing earlier vs later statements
  - DETECT contradictions by comparing statements
  - JUDGE confidence/certainty from hedging language ("maybe", "I think")

Use chronological order: earlier facts (lower ts) are EARLIER in time;
later facts are MORE RECENT. For "current state" questions, prefer the
LATEST relevant fact. For "ever" / "history" / "all" questions, use ALL
relevant facts.

DISAMBIGUATION (when the question name matches multiple entities):
  - The sidebar may flag "⚠ COLLIDING SURFACES" — surfaces shared by
    multiple distinct entities (e.g., 3 different people named Alice).
  - Read the question carefully for a MODIFIER on the colliding surface
    (e.g., "Alice the neighbor", "Alice on the platform team",
    "Sara's sister Alice"). Match the modifier to the entity whose
    discriminating excerpts are consistent with it.
  - Answer using ONLY that entity's facts. Do NOT mix facts from other
    entities sharing the surface.

ENTITIES IN RETRIEVED FACTS (sidebar):
{resolution_map}

RETRIEVED FACTS (chronological):
{facts_block}

QUESTION: {question}

Answer concisely. For yes/no questions, start with "Yes" or "No".
"""


def format_facts_for_read(
    facts: list[Fact],
    store: MemoryStore,
    eid_alias: dict[str, str] | None = None,
) -> str:
    """Render facts with per-mention surface + render-local Entity alias
    annotations. Never expose raw entity_ids — use the same Entity A/B/C
    labels as the resolution_map sidebar so the reader can cross-reference.
    """
    lines = []
    for f in facts:
        mention_resolutions = []
        for mid in f.mention_ids:
            eid = store.registry.get_canonical(mid)
            idx = store.collections.get(f.collection)
            surface = "?"
            if idx and mid in idx.mentions_by_id:
                surface = idx.mentions_by_id[mid].surface
            alias = eid_alias.get(eid, "?") if eid_alias else "?"
            if eid == "e_user":
                alias = "User"
            mention_resolutions.append(f"      [{surface!r} → Entity {alias}]")
        meta = (
            "  mentions:\n" + "\n".join(mention_resolutions)
            if mention_resolutions
            else ""
        )
        if meta:
            lines.append(f"[t={f.ts}] {f.text}\n{meta}")
        else:
            lines.append(f"[t={f.ts}] {f.text}")
    return "\n".join(lines)


def format_resolution_map(
    resolution_map: dict[str, set[str]],
    store: MemoryStore,
    eid_alias: dict[str, str] | None = None,
) -> str:
    """Render a per-entity sidebar showing surfaces + a couple of
    discriminating fact excerpts per entity. When multiple entities share
    a surface (e.g., 3 Alices), the excerpts give the reader concrete
    distinguishing context to pick the right entity for the question.

    Uses render-local Group A/B/C labels rather than raw entity_ids,
    consistent with the writer-side principle (concrete-entity ids stay
    invisible).
    """
    if not resolution_map:
        return "(no entities)"

    def _alias(idx: int) -> str:
        s = ""
        n = idx
        while True:
            s = chr(ord("A") + (n % 26)) + s
            n = n // 26 - 1
            if n < 0:
                break
        return s

    # Detect surface collisions across entities (helps reader disambiguation)
    surface_to_entities: dict[str, set[str]] = {}
    entity_data: dict[str, dict] = {}
    for eid, mids in sorted(resolution_map.items(), key=lambda kv: kv[0]):
        surfaces = set()
        excerpts: list[tuple[int, str]] = []
        for mid in mids:
            for col in store.collections.values():
                if mid in col.mentions_by_id:
                    m = col.mentions_by_id[mid]
                    surfaces.add(m.surface)
                    f = col.by_uuid.get(m.fact_uuid)
                    if f:
                        excerpts.append((m.ts, f.text[:100]))
        for s in surfaces:
            surface_to_entities.setdefault(s, set()).add(eid)
        excerpts.sort()  # chronological
        entity_data[eid] = {
            "surfaces": sorted(surfaces),
            "excerpts": excerpts[:3],
            "mids": sorted(mids),
        }

    if eid_alias is None:
        eid_alias = {
            eid: _alias(i) for i, eid in enumerate(sorted(resolution_map.keys()))
        }
    collisions = {
        s: sorted(eid_alias[e] for e in eids)
        for s, eids in surface_to_entities.items()
        if len(eids) > 1
    }

    lines = []
    if collisions:
        lines.append(
            "⚠ COLLIDING SURFACES (same surface for multiple entities — read fact text to disambiguate):"
        )
        for s, gs in sorted(collisions.items()):
            lines.append(f'    "{s}" → Entities {", ".join(gs)}')
        lines.append("")

    lines.append("ENTITIES IN RETRIEVED FACTS:")
    for eid in sorted(resolution_map.keys()):
        info = entity_data[eid]
        alias = eid_alias[eid]
        surf = info["surfaces"]
        lines.append(f"  Entity {alias}: surfaces={surf}")
        for ts, txt in info["excerpts"]:
            lines.append(f'      [t={ts}] "{txt}"')
    return "\n".join(lines)


def _build_eid_alias(resolution_map: dict[str, set[str]]) -> dict[str, str]:
    def _alias(idx: int) -> str:
        s = ""
        n = idx
        while True:
            s = chr(ord("A") + (n % 26)) + s
            n = n // 26 - 1
            if n < 0:
                break
        return s

    return {eid: _alias(i) for i, eid in enumerate(sorted(resolution_map.keys()))}


def answer_question(
    question: str, store: MemoryStore, cache: Cache, budget: Budget, top_k: int = 14
) -> str:
    facts, resolution_map = retrieve(question, store, cache, budget, top_k=top_k)
    eid_alias = _build_eid_alias(resolution_map)
    facts_block = format_facts_for_read(facts, store, eid_alias=eid_alias)
    resolution_block = format_resolution_map(resolution_map, store, eid_alias=eid_alias)
    prompt = READ_PROMPT.format(
        resolution_map=resolution_block,
        facts_block=facts_block,
        question=question,
    )
    return llm(prompt, cache, budget, reasoning_effort="medium").strip()


# ---------------------------------------------------------------------------
# Ingestion driver — K=3 centered window
# ---------------------------------------------------------------------------


def ingest_turns(
    turns,
    cache,
    budget,
    *,
    w_past: int = 7,
    w_future: int = 7,
    k: int = 3,
    rebuild_index_every: int = 4,
    inline_anchors: bool = False,
):
    obs_facts: list[Fact] = []
    obs_mentions: list[Mention] = []
    store = MemoryStore()
    # Pre-register the User entity so writer can resolve to "e_user"
    store.registry.register("m_user_root")
    # Re-point: we want the canonical to literally be "e_user" not "e_m_user_root"
    # Easiest: synthesize the entity_id directly
    store.registry.mention_to_entity["m_user_root"] = "e_user"
    store.registry.entity_members["e_user"] = {"m_user_root"}
    # Drop the auto-registered e_m_user_root if any
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
        new_facts, new_mentions, new_bindings, tele = write_window(
            window_turns,
            target_turn_lo,
            target_turns,
            obs_facts,
            obs_idx,
            store.registry,
            cache,
            budget,
            inline_anchors=inline_anchors,
            all_mentions=obs_mentions if inline_anchors else None,
        )
        obs_facts.extend(new_facts)
        obs_mentions.extend(new_mentions)
        tele["fire_no"] = fire_no
        tele["last_turn"] = target_turns[-1][0]
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
    return obs_facts, obs_mentions, store, telemetry
