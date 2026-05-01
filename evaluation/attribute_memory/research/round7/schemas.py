"""Round 7 schema definitions — integrated solutions for 4 entity-layer problems.

P1: Multi-label routing with entity-introduction / relationship-event gate.
P2: Context buffer for anonymous -> named coreference resolution.
P3: Role slots as first-class memory objects (separated from entity profiles).
P4: Salience-gated entity extraction.

These schemas compose: a salient-entity signal helps P2 (we buffer only
interesting anonymous references); multi-label routing produces the dual
entities that P3 role-slot updates need; P2 resolutions trigger P3
role-slot fills.

All schemas are dataclass-driven, append-only where possible, and
deterministic in their evaluation-relevant parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# P1: Multi-label gate
# ---------------------------------------------------------------------------

MultiLabelReason = Literal[
    "new_entity",  # fact introduces an entity not yet in entities set
    "relationship_event",  # bi-entity relational state change
    "single",  # default: one subject
]


@dataclass
class MultiLabelDecision:
    """Decision for whether a fact is single-subject or multi-subject."""

    topics: list[str]  # 1 or 2+ (rarely 3)
    reason: MultiLabelReason
    introduced_entities: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# P2: Context buffer for coreference
# ---------------------------------------------------------------------------


@dataclass
class AnonymousMention:
    """A descriptive/anonymous reference to an unknown entity.

    E.g. "my boss", "this new guy at work", "my sister's kid"."""

    turn_idx: int
    descriptor: str  # "my boss"
    topic: str  # Where the fact was routed, e.g. "User/Employment"
    fact_text: str  # Original fact, for retroactive merge context
    entry_ids: list[int] = field(
        default_factory=list
    )  # log entries under the descriptor


@dataclass
class CoreferenceMerge:
    """Signal to the memory store: merge anon entity into a named one."""

    canonical_entity: str  # "Marcus"
    anonymous_topic: str  # "User/Employment" where "my boss" facts were filed
    anonymous_descriptor: str  # "my boss"
    matched_mention_turn_idx: int
    rationale: str


# Buffer size bounds — heuristic. We keep anonymous mentions that haven't
# been resolved for UP TO N turns. Bigger buffer = more recall for late names
# but more LLM-check cost per new named entity. Start at 30.
COREF_BUFFER_MAX_TURNS = 30
COREF_BUFFER_MAX_MENTIONS = 20


# ---------------------------------------------------------------------------
# P3: Role slots as first-class objects
# ---------------------------------------------------------------------------


@dataclass
class RoleSlot:
    """A role slot is a named relationship position whose filler is an entity
    pointer. E.g. User/Employment has slot `boss` currently pointing at
    `@Marcus`. When boss changes to Alice, we emit a supersede append on the
    slot's mini-log; entity logs for Marcus / Alice stay untouched."""

    slot_id: str  # e.g. "User/Employment/boss"
    # History of (valid_from_ts, entity_pointer | None) — None means 'vacant'.
    # Append-only: latest non-invalidated entry is current.
    history: list[RoleSlotEntry] = field(default_factory=list)

    def current(self) -> RoleSlotEntry | None:
        live = [e for e in self.history if not e.invalidated]
        return live[-1] if live else None


@dataclass
class RoleSlotEntry:
    slot_id: str
    ts: str
    filler: str | None  # "@Marcus" or None (vacant)
    invalidated: bool = False  # if the slot assignment itself was wrong
    source_turn: int | None = None
    source_fact: str | None = None


# The canonical slot-id grammar: "<Holder>/<Category>/<Role>"
# - Holder: the entity who "has" this role in their life. Usually User.
# - Category: topic subcategory for the role (Employment, Family, Gym).
# - Role: descriptive role name (boss, mentor, trainer, partner).


# ---------------------------------------------------------------------------
# P4: Salience state for entity extraction
# ---------------------------------------------------------------------------


@dataclass
class SalienceCandidate:
    """An extracted noun phrase that is *not yet* a first-class entity.
    Lives in the deferred-pool until the salience threshold is crossed."""

    descriptor: str  # "the bowl", "a cup", "grandmother's blue bowl"
    first_seen_turn: int
    mention_count: int = 1
    has_name: bool = False  # True if a proper-noun name has been attached
    has_state_change: bool = False  # True if fact implied a move/break/update
    has_identifying_detail: bool = False  # color/origin/owner/sentimental
    mentions: list[int] = field(default_factory=list)  # turn indexes


SALIENCE_ENTITY_THRESHOLD_SCORE = 2
# Scoring (lazy creation policy, Option A in the prompt spec):
#   +2  named_specificity (has proper noun, e.g. "Luna")
#   +1  repeated_mention (>=2 mentions)
#   +1  state_change (move, break, update)
#   +1  identifying_detail (color/origin/owner/sentimental)
# Threshold 2 => proper names pass immediately; unnamed items need to
# accumulate signals across turns.


# ---------------------------------------------------------------------------
# Integrated extraction output
# ---------------------------------------------------------------------------


@dataclass
class IntegratedExtraction:
    """What the integrated extractor emits for a batch of turns.

    This is what the pipeline applier consumes: multiple labeled subjects,
    coreference merges to execute, role-slot updates to apply, and
    salience-filtered entities (deferred vs admitted)."""

    facts: list[FactWithRouting]
    coref_merges: list[CoreferenceMerge] = field(default_factory=list)
    role_slot_updates: list[RoleSlotEntry] = field(default_factory=list)
    salience_admitted: list[str] = field(default_factory=list)
    salience_deferred: list[str] = field(default_factory=list)


@dataclass
class FactWithRouting:
    text: str
    topics: list[str]  # 1 or more; result of P1 multi-label gate
    multi_label_reason: MultiLabelReason = "single"
    anonymous_descriptors: list[str] = field(default_factory=list)
    named_entities: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def salience_score(c: SalienceCandidate) -> int:
    s = 0
    if c.has_name:
        s += 2
    if c.mention_count >= 2:
        s += 1
    if c.has_state_change:
        s += 1
    if c.has_identifying_detail:
        s += 1
    return s


def should_admit_entity(c: SalienceCandidate) -> bool:
    return salience_score(c) >= SALIENCE_ENTITY_THRESHOLD_SCORE
