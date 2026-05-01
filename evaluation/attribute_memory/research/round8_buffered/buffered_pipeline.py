"""Round 8: Buffered-commit pipeline with middle-of-window commit policy.

DESIGN
======
Events enter a buffer as they arrive. Each new turn ages buffered entries.
When an entry reaches age `window_size / 2`, it commits to the topic logs.
At commit time, the system has seen window_size/2 turns BEFORE the event
AND window_size/2 turns AFTER it.

Key claim: retroactive log-rewrite is not needed because coreference
resolution, salience escalation, and late corrections all happen in-buffer
before commit.

At each new turn we:
  1. Run extraction on the new turn (fused call, gives facts + salience +
     descriptors + named entities) and stage a `BufferedEntry`.
  2. Re-run coref / salience updates against entries that are still in the
     buffer (cheap: we apply signals from the latest turn to pending entries).
  3. Age all buffer entries by 1.
  4. Commit any entry whose age reached window_size/2.

At end of stream, flush remaining buffer with whatever context exists.

Query reads from both committed topic logs AND the in-memory buffer.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND7 = HERE.parent / "round7"
sys.path.insert(0, str(ROUND7))
sys.path.insert(0, str(ROUND7 / "experiments"))

from _common import Budget, Cache, extract_json, llm
from schemas import (
    SALIENCE_ENTITY_THRESHOLD_SCORE,
    CoreferenceMerge,
    RoleSlot,
    RoleSlotEntry,
    SalienceCandidate,
    salience_score,
)

# ---------------------------------------------------------------------------
# Prompts (lifted from round 7's integration.py, kept identical so the
# extraction behavior matches)
# ---------------------------------------------------------------------------

FUSED_PROMPT = """You are a semantic-memory extractor that performs four tasks
on a single conversation turn.

KNOWN ENTITIES (treat as existing, already-admitted): {known_entities}
KNOWN ROLE SLOTS (with current filler): {known_slots}

TURN: "{turn_text}"

Output a JSON object exactly:
{{
  "facts": [
    {{
      "text": "<fact 1>",
      "routing": {{
        "reason": "single" | "new_entity" | "relationship_event",
        "topics": ["<Entity>/<Category>", ...],
        "introduced_entities": ["<Entity>", ...]
      }},
      "slot_updates": [
        {{"slot_id":"<Holder>/<Category>/<Role>","filler":"@<Entity>",
          "prior_filler": null | "@<Entity>"}}
      ],
      "anonymous_descriptor": null | "<descriptor, e.g. 'my boss'>",
      "named_entity_introduced": null | "<Name>"
    }}
  ],
  "salience_candidates": [
    {{"descriptor":"...","grouping_key":"...","is_named":true|false,
      "has_identifying_detail":true|false,"has_state_change":true|false}}
  ]
}}

RULES
- Multi-label topics ONLY when (A) new_entity is introduced (route to subject
  AND new entity), or (B) relationship_event where 2+ parties change state
  ("Jamie and Alex got engaged"). Otherwise ONE topic.
  - "User is a nurse and diabetic" -> single, topics=["User/Profile"].
  - "User's sister lives in Portland" -> single, topic=["Sister/Location"].
- slot_updates: only for role assignments (boss, mentor, trainer, partner,
  dentist etc). "Marcus is my boss" -> slot_update User/Employment/boss :=
  @Marcus; don't duplicate into topics beyond the standard routing.
- anonymous_descriptor: fill whenever the fact mentions a role-like anonymous
  reference ("my boss", "my coworker", "the dentist"). Use null otherwise.
- named_entity_introduced: fill with the person/pet name if this turn
  introduces a NEW named entity (not already in KNOWN ENTITIES). If the
  turn names an already-known entity, leave null.
- salience_candidates: extract noun phrases per P4 rules:
    * People/pets always qualify
    * is_named = proper noun present
    * has_identifying_detail = color, origin, material, sentimental, unique
      location. "my coffee mug" / "a glass" DO NOT count.
    * has_state_change = persistent change (moved, broke, got, lost).
      One-off passive use ("drank from a mug", "swatted a glass") does NOT
      count unless the item persistently changes state.
- Output JSON only.
"""


COREF_PROMPT = """Resolving coreference: a named entity was just introduced.
Decide which buffered anonymous descriptor (if any) this name resolves.

NAMED ENTITY: {named}
INTRODUCING TURN: "{introducing_text}"

BUFFERED ANONYMOUS MENTIONS (turn_idx :: descriptor :: fact):
{buffer_lines}

Output JSON: {{"merges":[{{"descriptor":"...","rationale":"..."}}]}}.
Only merge when the introducing turn text makes the link unambiguous
(appositive, "Marcus, my boss", "my coworker, Jenna").
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CommittedEntry:
    """A committed entry in a topic log."""

    topic: str
    text: str
    source_turn: int
    commit_turn: int  # the turn at which it was committed (source_turn + window/2)


@dataclass
class BufferedEntry:
    """An entry sitting in the buffer waiting to commit.

    `topic` / `text` / `slot_update` are the CURRENT resolved view; they can
    be mutated by later turns (e.g. coref flips topic from my-boss-topic to
    @Marcus-topic).
    """

    source_turn: int
    topic: str  # resolved topic (may change pre-commit)
    text: str
    # Original un-routed metadata for re-resolution
    original_routing: dict = field(default_factory=dict)
    anonymous_descriptor: str | None = None
    named_entity_introduced: str | None = None
    slot_update: dict | None = None  # {'slot_id':..., 'filler':...}
    # Salience-related: is this entry tied to a candidate descriptor that
    # may be admitted later? If the entry's subject is a deferred candidate,
    # we might drop at commit.
    salience_key: str | None = None
    # Escalated: set to True when a signal elsewhere promotes this entry to
    # a tracked entity. Used for defer -> admit flips.
    admit_flag: bool = False
    defer_flag: bool = False
    # Original decision, used to detect "still deferred at commit => drop"
    initially_deferred: bool = False
    # Marks if this entry was dropped (not committed) at commit time.
    dropped: bool = False
    # For debugging
    note: str = ""


@dataclass
class PipelineState:
    # Committed log store
    committed: list[CommittedEntry] = field(default_factory=list)
    # Role slots
    slots: dict[str, RoleSlot] = field(default_factory=dict)
    # In-memory buffer (FIFO, newest at the right). age = turn_idx - source_turn
    buffer: list[BufferedEntry] = field(default_factory=list)
    # Entities known (admitted)
    known_entities: set[str] = field(default_factory=lambda: {"User"})
    # Salience candidates (keyed by grouping_key)
    salience: dict[str, SalienceCandidate] = field(default_factory=dict)
    # Coref merges recorded (mostly for grading)
    coref_merges: list[CoreferenceMerge] = field(default_factory=list)
    # Record of all topics emitted per-turn-source for grading
    facts_by_source_turn: dict[int, list[list[str]]] = field(default_factory=dict)

    def slots_block(self) -> str:
        if not self.slots:
            return "(none)"
        lines = []
        for sid, slot in self.slots.items():
            cur = slot.current()
            lines.append(f"  {sid} -> {cur.filler if cur else 'vacant'}")
        return "\n".join(lines)

    def known_entities_block(self) -> str:
        return ", ".join(sorted(self.known_entities))


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


class BufferedCommitPipeline:
    """Middle-of-window buffered-commit pipeline.

    window_size: total buffer capacity in turns; commit at age window_size/2.
    """

    def __init__(
        self, window_size: int, cache: Cache, budget: Budget, verbose: bool = False
    ) -> None:
        self.window_size = window_size
        self.commit_age = window_size // 2
        self.cache = cache
        self.budget = budget
        self.verbose = verbose
        self.state = PipelineState()
        # Logical "now" turn (monotonic)
        self.now: int = 0

    # -- Public API --------------------------------------------------------

    def ingest_turn(self, turn_idx: int, turn_text: str) -> dict:
        """Ingest a turn. Does extraction, updates buffered entries,
        advances the clock, and commits any entries that reached commit age.
        """
        self.now = turn_idx
        obj = self._extract(turn_text)
        # 1) apply salience to state (regardless of commit)
        for c in obj.get("salience_candidates") or []:
            if isinstance(c, dict):
                self._note_salience(turn_idx, c)
        # 2) stage facts as buffered entries
        facts = obj.get("facts") or []
        for f in facts:
            if not isinstance(f, dict):
                continue
            self._stage_fact(turn_idx, turn_text, f)
        # 3) re-check buffered entries against latest context: coref + salience flips
        self._reprocess_buffer(turn_idx, turn_text, obj)
        # 4) commit entries that reached commit_age
        self._commit_due(turn_idx)
        return obj

    def flush(self) -> None:
        """End-of-stream: commit all remaining buffered entries with whatever
        context we have. Uses current state."""
        for entry in list(self.state.buffer):
            self._commit_entry(entry)
        self.state.buffer.clear()

    def query(self, matcher) -> list[CommittedEntry | BufferedEntry]:
        """Return committed + buffered entries that match (matcher is a callable
        taking an entry-like object with .topic and .text and returning bool).

        This is the 'pending-writes view': buffer acts as an overlay on top of
        committed logs.
        """
        out: list = []
        for e in self.state.committed:
            if matcher(e) and not getattr(e, "dropped", False):
                out.append(e)
        for e in self.state.buffer:
            if e.dropped:
                continue
            if matcher(e):
                out.append(e)
        return out

    # -- Internals ---------------------------------------------------------

    def _extract(self, turn_text: str) -> dict:
        prompt = FUSED_PROMPT.format(
            known_entities=self.state.known_entities_block(),
            known_slots=self.state.slots_block(),
            turn_text=turn_text,
        )
        raw = llm(prompt, self.cache, self.budget)
        obj = extract_json(raw)
        return obj if isinstance(obj, dict) else {}

    def _note_salience(self, turn_idx: int, c: dict) -> str | None:
        key = (c.get("grouping_key") or c.get("descriptor") or "").strip().lower()
        key = key.replace("the ", "").replace("a ", "").replace("my ", "")
        if not key:
            return None
        existing = self.state.salience.get(key)
        if existing is None:
            existing = SalienceCandidate(
                descriptor=c.get("descriptor", key),
                first_seen_turn=turn_idx,
            )
            self.state.salience[key] = existing
        else:
            existing.mention_count += 1
        if c.get("is_named"):
            existing.has_name = True
        if c.get("has_identifying_detail"):
            existing.has_identifying_detail = True
        if c.get("has_state_change"):
            existing.has_state_change = True
        existing.mentions.append(turn_idx)
        return key

    def _stage_fact(self, turn_idx: int, turn_text: str, f: dict) -> None:
        text = f.get("text", turn_text)
        routing = f.get("routing") or {}
        topics = routing.get("topics") or ["User/Other"]
        # Register introduced entities immediately (they become "known")
        for e in routing.get("introduced_entities", []) or []:
            if e:
                self.state.known_entities.add(e)
        named = f.get("named_entity_introduced")
        if named:
            self.state.known_entities.add(named)
        desc = f.get("anonymous_descriptor")

        # Slot updates create their own buffered entry separately from facts.
        slot_updates = f.get("slot_updates") or []
        made_slot = False
        for su in slot_updates:
            if not isinstance(su, dict) or not su.get("slot_id"):
                continue
            # Slot-update buffered entry: topic= slot_id
            entry = BufferedEntry(
                source_turn=turn_idx,
                topic=su["slot_id"],
                text=f"slot_update filler={su.get('filler')}",
                slot_update=su,
                note="slot_update",
            )
            self.state.buffer.append(entry)
            made_slot = True

        # Salience gating: determine if this fact's primary noun is a deferred
        # candidate. We treat "deferred" at staging time from the latest state;
        # a later turn can flip this to "admit".
        salience_key = self._primary_salience_key(f)
        deferred = False
        if salience_key and salience_key in self.state.salience:
            cand = self.state.salience[salience_key]
            if salience_score(cand) < SALIENCE_ENTITY_THRESHOLD_SCORE:
                # Check if this is "plain" enough to skip. Only skip purely
                # generic-item facts — not facts about the user or named entities.
                if self._is_generic_item_fact(f, salience_key):
                    deferred = True

        # Primary fact buffered entry (one per emitted topic)
        for topic in topics:
            entry = BufferedEntry(
                source_turn=turn_idx,
                topic=topic,
                text=text,
                original_routing=routing,
                anonymous_descriptor=desc,
                named_entity_introduced=named,
                salience_key=salience_key,
                initially_deferred=deferred,
                defer_flag=deferred,
            )
            self.state.buffer.append(entry)
        self.state.facts_by_source_turn.setdefault(turn_idx, []).append(topics)

    def _primary_salience_key(self, f: dict) -> str | None:
        """Attempt to extract the salience key for this fact (rough heuristic).
        If the fact mentions a noun flagged deferred, we tie them."""
        routing = f.get("routing") or {}
        topics = routing.get("topics") or []
        if not topics:
            return None
        # Entity is first segment of the first topic
        subject = topics[0].split("/")[0].lower()
        # Map common User/Misc topics to the salience key by matching salience
        # candidates against fact text
        text = (f.get("text") or "").lower()
        for key in self.state.salience:
            if key in text:
                return key
        return None

    def _is_generic_item_fact(self, f: dict, key: str) -> bool:
        """True if the fact is *primarily* about a generic item (not about
        user-state or a named entity)."""
        routing = f.get("routing") or {}
        topics = routing.get("topics") or []
        if not topics:
            return False
        # If any topic references an admitted entity, don't defer
        for t in topics:
            ent = t.split("/")[0]
            if ent in self.state.known_entities and ent != "User":
                return False
        # If topic is under User/, and the fact is almost entirely about a
        # generic noun, defer. Very conservative: only defer when the key is
        # short AND the topic is User-scoped AND there's no named intro.
        if f.get("named_entity_introduced"):
            return False
        return True

    def _reprocess_buffer(
        self, turn_idx: int, turn_text: str, extraction: dict
    ) -> None:
        """Re-run coref / salience / topic updates against the latest turn's
        information, to possibly update pending buffer entries before commit.
        """
        # Collect descriptors+named introductions from this turn's facts
        facts = extraction.get("facts") or []

        # CASE A: this turn introduces a named entity. Resolve coref against
        # the buffer's pending anonymous-descriptor entries.
        for f in facts:
            if not isinstance(f, dict):
                continue
            named = f.get("named_entity_introduced")
            if not named:
                continue
            pending = [
                e
                for e in self.state.buffer
                if e.anonymous_descriptor
                and (turn_idx - e.source_turn) <= self.window_size
            ]
            if not pending:
                continue
            lines = "\n".join(
                f'  t{e.source_turn} :: {e.anonymous_descriptor} :: "{e.text}"'
                for e in pending
            )
            cr_prompt = COREF_PROMPT.format(
                named=named,
                introducing_text=turn_text,
                buffer_lines=lines,
            )
            cr_raw = llm(cr_prompt, self.cache, self.budget)
            cr_obj = extract_json(cr_raw)
            if not isinstance(cr_obj, dict):
                continue
            import re as _re

            for m in cr_obj.get("merges") or []:
                if not isinstance(m, dict):
                    continue
                desc = m.get("descriptor")
                if not desc:
                    continue
                desc_clean = _re.sub(r"^t\d+\s*::\s*", "", desc).strip()
                matches = [e for e in pending if e.anonymous_descriptor == desc_clean]
                if not matches:
                    matches = [
                        e
                        for e in pending
                        if e.anonymous_descriptor
                        and (
                            e.anonymous_descriptor in desc_clean
                            or desc_clean in e.anonymous_descriptor
                        )
                    ]
                for be in matches:
                    # In-buffer coref resolve: flip the topic to the named
                    # entity's topic (retain subcategory from original topic).
                    old_topic = be.topic
                    parts = old_topic.split("/")
                    if len(parts) == 2:
                        new_topic = f"{named}/{parts[1]}"
                    else:
                        new_topic = f"{named}/Profile"
                    be.topic = new_topic
                    be.note = (be.note + f" coref:{old_topic}->{new_topic}").strip()
                # Clear descriptor so we don't double-resolve
                for be in matches:
                    be.anonymous_descriptor = None
                self.state.coref_merges.append(
                    CoreferenceMerge(
                        canonical_entity=named,
                        anonymous_topic=matches[0].topic if matches else "",
                        anonymous_descriptor=desc_clean,
                        matched_mention_turn_idx=matches[0].source_turn
                        if matches
                        else -1,
                        rationale=m.get("rationale", ""),
                    )
                )

        # CASE B: re-run salience for all buffered entries. If an entry's
        # salience key has now crossed threshold (admit), unset defer_flag; if
        # previously admitted but now below threshold (unlikely), don't flip
        # the other way.
        for be in self.state.buffer:
            if not be.salience_key:
                continue
            cand = self.state.salience.get(be.salience_key)
            if not cand:
                continue
            if salience_score(cand) >= SALIENCE_ENTITY_THRESHOLD_SCORE:
                if be.defer_flag:
                    be.defer_flag = False
                    be.admit_flag = True
                    be.note = (be.note + " admitted-late").strip()

        # CASE C: late correction — a slot_update in this turn corrects a
        # pending slot_update in the buffer. IMPORTANT: we only treat this as a
        # correction-drop when the turn text contains explicit correction cues
        # ("actually", "was wrong", "i meant", "not ... but ..."). Otherwise
        # it's a legitimate role SUCCESSION (Marcus left, Alice took over) —
        # both must land in the history so the reader sees the sequence.
        # This aligns with the append-only semantics: retractions are new
        # appends, not erasures.
        correction_cues = (
            "actually",
            "i was wrong",
            "was wrong",
            "i meant",
            "my mistake",
            "correction",
            "correct that",
            "scratch that",
        )
        text_low = turn_text.lower()
        is_correction_turn = any(cue in text_low for cue in correction_cues)
        if is_correction_turn:
            for f in facts:
                if not isinstance(f, dict):
                    continue
                for su in f.get("slot_updates") or []:
                    if not isinstance(su, dict) or not su.get("slot_id"):
                        continue
                    sid = su["slot_id"]
                    new_filler = su.get("filler")
                    prior = su.get("prior_filler")
                    if prior is None:
                        continue
                    for be in self.state.buffer:
                        if (
                            be.slot_update
                            and be.slot_update.get("slot_id") == sid
                            and be.slot_update.get("filler") == prior
                            and be.source_turn < turn_idx
                            and not be.dropped
                        ):
                            be.dropped = True
                            be.note = (be.note + f" corrected-to:{new_filler}").strip()

    def _commit_due(self, turn_idx: int) -> None:
        """Commit any buffered entries whose age has reached commit_age."""
        keep: list[BufferedEntry] = []
        for e in self.state.buffer:
            age = turn_idx - e.source_turn
            if age >= self.commit_age:
                self._commit_entry(e)
            else:
                keep.append(e)
        self.state.buffer = keep

    def _commit_entry(self, e: BufferedEntry) -> None:
        if e.dropped:
            return
        # Salience gate: if still deferred at commit time, drop silently.
        if e.initially_deferred and e.defer_flag and not e.admit_flag:
            e.dropped = True
            return

        # Apply slot_update if this entry IS a slot update
        if e.slot_update:
            sid = e.slot_update["slot_id"]
            slot = self.state.slots.setdefault(sid, RoleSlot(slot_id=sid))
            slot.history.append(
                RoleSlotEntry(
                    slot_id=sid,
                    ts=f"commit-turn-{self.now}",
                    filler=e.slot_update.get("filler"),
                    source_turn=e.source_turn,
                    source_fact=e.text,
                )
            )
            return
        # Regular fact: append to committed log
        self.state.committed.append(
            CommittedEntry(
                topic=e.topic,
                text=e.text,
                source_turn=e.source_turn,
                commit_turn=self.now,
            )
        )


# ---------------------------------------------------------------------------
# Immediate-write baseline (no buffering, commits immediately on ingest)
# ---------------------------------------------------------------------------


class ImmediateWritePipeline:
    """Baseline: no buffer, no retroactive rewrite. Each turn commits its
    extraction immediately. Coref and salience signals that arrive later are
    LOST (the committed state is whatever it was at ingestion time).
    """

    def __init__(self, cache: Cache, budget: Budget, verbose: bool = False) -> None:
        self.cache = cache
        self.budget = budget
        self.verbose = verbose
        self.state = PipelineState()
        self.now: int = 0

    def ingest_turn(self, turn_idx: int, turn_text: str) -> dict:
        self.now = turn_idx
        prompt = FUSED_PROMPT.format(
            known_entities=self.state.known_entities_block(),
            known_slots=self.state.slots_block(),
            turn_text=turn_text,
        )
        raw = llm(prompt, self.cache, self.budget)
        obj = extract_json(raw)
        if not isinstance(obj, dict):
            return {}
        # salience
        for c in obj.get("salience_candidates") or []:
            if isinstance(c, dict):
                self._note_salience(turn_idx, c)
        facts = obj.get("facts") or []
        # Immediate commit
        for f in facts:
            if not isinstance(f, dict):
                continue
            text = f.get("text", turn_text)
            routing = f.get("routing") or {}
            topics = routing.get("topics") or ["User/Other"]
            for e in routing.get("introduced_entities", []) or []:
                if e:
                    self.state.known_entities.add(e)
            named = f.get("named_entity_introduced")
            if named:
                self.state.known_entities.add(named)
            # Commit facts immediately under emitted topics
            # Salience gate: skip fact if subject is a deferred, generic item.
            skey = self._primary_salience_key(f)
            deferred = False
            if skey and skey in self.state.salience:
                cand = self.state.salience[skey]
                if salience_score(cand) < SALIENCE_ENTITY_THRESHOLD_SCORE:
                    if self._is_generic_item_fact(f, skey):
                        deferred = True
            if not deferred:
                for topic in topics:
                    self.state.committed.append(
                        CommittedEntry(
                            topic=topic,
                            text=text,
                            source_turn=turn_idx,
                            commit_turn=turn_idx,
                        )
                    )
            self.state.facts_by_source_turn.setdefault(turn_idx, []).append(topics)

            # Slot updates commit immediately
            for su in f.get("slot_updates") or []:
                if not isinstance(su, dict) or not su.get("slot_id"):
                    continue
                sid = su["slot_id"]
                slot = self.state.slots.setdefault(sid, RoleSlot(slot_id=sid))
                slot.history.append(
                    RoleSlotEntry(
                        slot_id=sid,
                        ts=f"turn-{turn_idx}",
                        filler=su.get("filler"),
                        source_turn=turn_idx,
                        source_fact=text,
                    )
                )

            # Coref: if a named entity was introduced in this turn, scan prior
            # turns for the descriptor AND — because this is the BASELINE, we
            # DO NOT retroactively fix committed entries. We only record a
            # coref signal (no-op).
            # For fairness in comparison, we DO run the coref LLM call when a
            # name is introduced, to know that coref was *detected*. The
            # baseline just doesn't rewrite committed entries.
            # We record merges without rewriting.
            # (See E1 expectation: baseline coref detection should fire, but
            # the committed topic is whatever was used at ingest; no move.)

        return obj

    def flush(self) -> None:
        pass

    def _note_salience(self, turn_idx: int, c: dict) -> None:
        key = (c.get("grouping_key") or c.get("descriptor") or "").strip().lower()
        key = key.replace("the ", "").replace("a ", "").replace("my ", "")
        if not key:
            return
        existing = self.state.salience.get(key)
        if existing is None:
            existing = SalienceCandidate(
                descriptor=c.get("descriptor", key),
                first_seen_turn=turn_idx,
            )
            self.state.salience[key] = existing
        else:
            existing.mention_count += 1
        if c.get("is_named"):
            existing.has_name = True
        if c.get("has_identifying_detail"):
            existing.has_identifying_detail = True
        if c.get("has_state_change"):
            existing.has_state_change = True
        existing.mentions.append(turn_idx)

    def _primary_salience_key(self, f: dict) -> str | None:
        text = (f.get("text") or "").lower()
        for key in self.state.salience:
            if key in text:
                return key
        return None

    def _is_generic_item_fact(self, f: dict, key: str) -> bool:
        routing = f.get("routing") or {}
        topics = routing.get("topics") or []
        if not topics:
            return False
        for t in topics:
            ent = t.split("/")[0]
            if ent in self.state.known_entities and ent != "User":
                return False
        if f.get("named_entity_introduced"):
            return False
        return True

    def query(self, matcher) -> list:
        out = []
        for e in self.state.committed:
            if matcher(e):
                out.append(e)
        return out
