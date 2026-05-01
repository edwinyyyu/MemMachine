"""Integration test: run all four mechanisms on a single interleaved scenario.

Per turn:
  1. Salience extraction (P4): flag candidate entities, maintain salience scores.
  2. Multi-label routing (P1): the turn's fact becomes topic(s). Route with the
     gate.
  3. Coref detection (P2): if the turn introduces a name, resolve against the
     buffer. If the turn contains an anonymous descriptor, add to buffer.
  4. Role-slot extraction (P3): for facts that mention role assignments, emit
     a slot update.

We then grade against the integration expectations (coref merges, role slot
history, admitted entities).

To keep within budget we use a SINGLE fused LLM call per turn that does all
four tasks, plus a separate coref-resolve call when needed. Reuses the per-
problem prompts as sections, then unifies into one JSON schema. This is the
"shipping" version — the per-problem experiments isolated each component, the
integration tests them composed.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND7 = HERE.parent
sys.path.insert(0, str(ROUND7))
sys.path.insert(0, str(HERE))

from _common import CACHE_DIR, Budget, Cache, extract_json, llm
from schemas import (
    COREF_BUFFER_MAX_MENTIONS,
    COREF_BUFFER_MAX_TURNS,
    SALIENCE_ENTITY_THRESHOLD_SCORE,
    AnonymousMention,
    CoreferenceMerge,
    RoleSlot,
    RoleSlotEntry,
    SalienceCandidate,
    salience_score,
)

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
# State
# ---------------------------------------------------------------------------


@dataclass
class IntegratedStore:
    known_entities: set[str] = field(default_factory=lambda: {"User"})
    slots: dict[str, RoleSlot] = field(default_factory=dict)
    anon_buffer: deque[AnonymousMention] = field(default_factory=deque)
    salience: dict[str, SalienceCandidate] = field(default_factory=dict)
    coref_merges: list[CoreferenceMerge] = field(default_factory=list)
    entity_facts: list[dict] = field(default_factory=list)
    all_topics_emitted: list[list[str]] = field(default_factory=list)

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

    def add_anon(self, turn_idx: int, descriptor: str, topic: str, fact: str) -> None:
        self.anon_buffer.append(
            AnonymousMention(
                turn_idx=turn_idx,
                descriptor=descriptor,
                topic=topic,
                fact_text=fact,
            )
        )
        self._prune_buffer(turn_idx)

    def _prune_buffer(self, current_turn: int) -> None:
        while (
            self.anon_buffer
            and (current_turn - self.anon_buffer[0].turn_idx) > COREF_BUFFER_MAX_TURNS
        ):
            self.anon_buffer.popleft()
        while len(self.anon_buffer) > COREF_BUFFER_MAX_MENTIONS:
            self.anon_buffer.popleft()

    def active_buffer(self, turn_idx: int) -> list[AnonymousMention]:
        self._prune_buffer(turn_idx)
        return list(self.anon_buffer)

    def pop_descriptor(self, descriptor: str) -> None:
        for i, m in enumerate(self.anon_buffer):
            if m.descriptor == descriptor:
                del self.anon_buffer[i]
                return

    def add_salience(self, turn_idx: int, c: dict) -> None:
        key = (c.get("grouping_key") or c.get("descriptor") or "").strip().lower()
        key = key.replace("the ", "").replace("a ", "").replace("my ", "")
        if not key:
            return
        existing = self.salience.get(key)
        if existing is None:
            existing = SalienceCandidate(
                descriptor=c.get("descriptor", key),
                first_seen_turn=turn_idx,
            )
            self.salience[key] = existing
        else:
            existing.mention_count += 1
        if c.get("is_named"):
            existing.has_name = True
        if c.get("has_identifying_detail"):
            existing.has_identifying_detail = True
        if c.get("has_state_change"):
            existing.has_state_change = True
        existing.mentions.append(turn_idx)

    def admitted_and_deferred(self) -> tuple[list[str], list[str]]:
        admitted, deferred = [], []
        for c in self.salience.values():
            if salience_score(c) >= SALIENCE_ENTITY_THRESHOLD_SCORE:
                admitted.append(c.descriptor)
            else:
                deferred.append(c.descriptor)
        return admitted, deferred

    def apply_slot_update(self, su: dict, ts: str, turn_idx: int, fact: str) -> None:
        slot = self.slots.setdefault(su["slot_id"], RoleSlot(slot_id=su["slot_id"]))
        slot.history.append(
            RoleSlotEntry(
                slot_id=su["slot_id"],
                ts=ts,
                filler=su.get("filler"),
                source_turn=turn_idx,
                source_fact=fact,
            )
        )


def process_turn(
    store: IntegratedStore, turn_idx: int, turn_text: str, cache: Cache, budget: Budget
) -> dict:
    prompt = FUSED_PROMPT.format(
        known_entities=store.known_entities_block(),
        known_slots=store.slots_block(),
        turn_text=turn_text,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return {"facts": [], "salience": []}

    # Apply salience regardless of other outcomes
    for c in obj.get("salience_candidates", []) or []:
        if isinstance(c, dict):
            store.add_salience(turn_idx, c)

    facts = obj.get("facts", []) or []
    # First pass: add anonymous descriptors to buffer (so same-turn named-intro
    # sees them).
    for f in facts:
        if not isinstance(f, dict):
            continue
        fact_text = f.get("text", turn_text)
        if f.get("anonymous_descriptor"):
            # Use routed topic as home for the descriptor
            topics = (f.get("routing") or {}).get("topics") or ["User/Other"]
            store.add_anon(turn_idx, f["anonymous_descriptor"], topics[0], fact_text)

    # Second pass: process named intros (coref resolve), slot updates, entity
    # registration.
    for f in facts:
        if not isinstance(f, dict):
            continue
        fact_text = f.get("text", turn_text)
        routing = f.get("routing") or {}
        topics = routing.get("topics") or []
        store.all_topics_emitted.append(topics)
        for e in routing.get("introduced_entities", []) or []:
            if e:
                store.known_entities.add(e)

        for su in f.get("slot_updates", []) or []:
            if isinstance(su, dict) and su.get("slot_id") and "filler" in su:
                store.apply_slot_update(su, f"turn-{turn_idx}", turn_idx, fact_text)

        named = f.get("named_entity_introduced")
        if named:
            store.known_entities.add(named)
            # Resolve coref against buffer
            active = store.active_buffer(turn_idx)
            if active:
                lines = "\n".join(
                    f'  t{m.turn_idx} :: {m.descriptor} :: "{m.fact_text}"'
                    for m in active
                )
                cr_prompt = COREF_PROMPT.format(
                    named=named,
                    introducing_text=turn_text,
                    buffer_lines=lines,
                )
                cr_raw = llm(cr_prompt, cache, budget)
                cr_obj = extract_json(cr_raw)
                if isinstance(cr_obj, dict):
                    import re as _re

                    for m in cr_obj.get("merges", []) or []:
                        if not isinstance(m, dict):
                            continue
                        desc = m.get("descriptor")
                        if not desc:
                            continue
                        # Lenient parse: strip leading "t\d+ :: " that the LLM
                        # sometimes copies from the buffer rendering.
                        desc_clean = _re.sub(r"^t\d+\s*::\s*", "", desc).strip()
                        matches = [bm for bm in active if bm.descriptor == desc_clean]
                        if not matches:
                            # Fuzzy fallback: containment
                            matches = [
                                bm
                                for bm in active
                                if bm.descriptor in desc_clean
                                or desc_clean in bm.descriptor
                            ]
                        if not matches:
                            continue
                        bm = matches[0]
                        store.coref_merges.append(
                            CoreferenceMerge(
                                canonical_entity=named,
                                anonymous_topic=bm.topic,
                                anonymous_descriptor=bm.descriptor,
                                matched_mention_turn_idx=bm.turn_idx,
                                rationale=m.get("rationale", ""),
                            )
                        )
                        store.pop_descriptor(bm.descriptor)
    return obj


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def grade(scenario: dict, store: IntegratedStore) -> dict:
    exp = scenario["expected"]
    # --- coref ---
    exp_coref = {(m["canonical"], m["descriptor"]) for m in exp.get("coref_merges", [])}
    got_coref = {
        (m.canonical_entity, m.anonymous_descriptor) for m in store.coref_merges
    }
    coref_missing = exp_coref - got_coref
    coref_extra = got_coref - exp_coref
    coref_pass = not coref_missing and not coref_extra

    # --- role slots ---
    slot_pass = True
    slot_details = []
    for sid, seq in (exp.get("role_slot_history") or {}).items():
        exp_seq = [e["filler"] for e in seq]
        slot = store.slots.get(sid)
        got_seq = [e.filler for e in (slot.history if slot else [])]
        ok = got_seq == exp_seq
        if not ok:
            slot_pass = False
        slot_details.append({"slot": sid, "exp": exp_seq, "got": got_seq, "pass": ok})

    # --- entities admitted ---
    admitted, deferred = store.admitted_and_deferred()

    def norm(s):
        return (
            s.strip().lower().replace("the ", "").replace("a ", "").replace("my ", "")
        )

    admitted_keys = {norm(a) for a in admitted}
    exp_admit = {norm(a) for a in (exp.get("entities_admitted") or [])}
    exp_defer = {norm(d) for d in (exp.get("entities_deferred") or [])}

    def fuzzy_missing(missing_set, available_set):
        still = set()
        for m in missing_set:
            if not any(m in a or a in m for a in available_set):
                still.add(m)
        return still

    admit_missing = fuzzy_missing(exp_admit - admitted_keys, admitted_keys)
    admit_false = admitted_keys & exp_defer
    entities_pass = not admit_missing and not admit_false

    # --- multi-label introductions ---
    ml_pass = True
    ml_details = []
    for exp_ml in exp.get("multi_label_introductions", []) or []:
        turn = exp_ml["turn"]
        needed = exp_ml["topics_include"]
        # The turn-th emitted topic group should include these entities.
        if turn - 1 < 0 or turn - 1 >= len(store.all_topics_emitted):
            ml_pass = False
            ml_details.append({"turn": turn, "status": "no emission"})
            continue
        topics = store.all_topics_emitted[turn - 1]
        topic_entities = {t.split("/")[0] for t in topics}
        missing = [n for n in needed if n not in topic_entities]
        if missing:
            ml_pass = False
            ml_details.append(
                {"turn": turn, "missing": missing, "got": sorted(topic_entities)}
            )
        else:
            ml_details.append(
                {"turn": turn, "got": sorted(topic_entities), "pass": True}
            )

    return {
        "coref_pass": coref_pass,
        "coref_missing": sorted(coref_missing),
        "coref_extra": sorted(coref_extra),
        "slot_pass": slot_pass,
        "slot_details": slot_details,
        "entities_pass": entities_pass,
        "admitted": sorted(admitted),
        "admit_missing": sorted(admit_missing),
        "admit_false_positive": sorted(admit_false),
        "ml_pass": ml_pass,
        "ml_details": ml_details,
        "all_pass": coref_pass and slot_pass and entities_pass and ml_pass,
    }


def main() -> None:
    scenarios_path = ROUND7 / "scenarios" / "integration.json"
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]
    cache = Cache(CACHE_DIR / "integration_llm.json")
    budget = Budget()

    results = []
    t0 = time.time()
    try:
        for sc in scenarios:
            store = IntegratedStore()
            for turn in sc["turns"]:
                process_turn(store, turn["idx"], turn["text"], cache, budget)
            g = grade(sc, store)
            g["scenario_id"] = sc["id"]
            g["slots"] = {
                sid: [e.filler for e in s.history] for sid, s in store.slots.items()
            }
            g["coref_merges"] = [
                (m.canonical_entity, m.anonymous_descriptor) for m in store.coref_merges
            ]
            results.append(g)
            print(
                f"  {sc['id']}: coref={g['coref_pass']} slots={g['slot_pass']} "
                f"ents={g['entities_pass']} ml={g['ml_pass']} ALL={g['all_pass']}",
                flush=True,
            )
            if not g["coref_pass"]:
                print(
                    f"    coref missing={g['coref_missing']} extra={g['coref_extra']}"
                )
            if not g["slot_pass"]:
                for sd in g["slot_details"]:
                    if not sd["pass"]:
                        print(f"    slot {sd['slot']} exp={sd['exp']} got={sd['got']}")
            if not g["entities_pass"]:
                print(
                    f"    admit missing={g['admit_missing']} false={g['admit_false_positive']}"
                )
                print(f"    admitted={g['admitted']}")
            if not g["ml_pass"]:
                print(f"    ml={g['ml_details']}")
    finally:
        cache.save()

    passed = sum(1 for r in results if r["all_pass"])
    print(f"\nIntegration: {passed}/{len(results)}")
    print(
        f"LLM calls: {budget.llm_calls}, cost ~${budget.cost():.3f}, wall {time.time() - t0:.1f}s"
    )

    out = {
        "scenarios": len(results),
        "passed": passed,
        "results": results,
        "llm_calls": budget.llm_calls,
        "cost_usd": round(budget.cost(), 4),
    }
    (ROUND7 / "results" / "integration.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
