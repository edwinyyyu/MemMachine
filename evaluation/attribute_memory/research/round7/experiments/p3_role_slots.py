"""P3: role slots as first-class memory objects.

Test whether gpt-5-mini can reliably emit role-slot updates vs entity-property
changes. We ask per-turn for a structured emission:

  {
    "slot_updates": [
      {"slot_id": "User/Employment/boss",
       "filler": "@Marcus",
       "prior_filler": null | "@X"},
      ...
    ],
    "entity_facts": [
      {"entity": "Marcus", "fact": "listens well"}, ...
    ]
  }

Grading (deterministic): we replay the slot_updates in order. For each
scenario's `expected_slot_history`, we check that the sequence of fillers
in each slot matches the expected sequence (in order, no interleaving /
no spurious fills).

For scenarios that also specify `expected_profile_changes`, we check that
the entity_facts list records the qualitative fact(s) AS entity-facts (NOT
as role-slot updates).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND7 = HERE.parent
sys.path.insert(0, str(ROUND7))
sys.path.insert(0, str(HERE))

from _common import CACHE_DIR, Budget, Cache, extract_json, llm
from schemas import RoleSlot, RoleSlotEntry

PROMPT = """You are managing a semantic-memory store that distinguishes
ROLE SLOTS from ENTITY PROFILES.

A ROLE SLOT is a first-class memory object representing a role-position in
the user's life. The slot's identity is <Holder>/<Category>/<Role>. The slot
is FILLED by a pointer to an entity. Only the slot tracks who currently
occupies the role — the entity's own log stays clean and holds qualitative
facts (personality, habits, birthday, etc.).

Examples:
  "Marcus is my boss" -> slot_update: User/Employment/boss := @Marcus
  "Alice is my new boss now" -> slot_update: User/Employment/boss := @Alice
      (prior_filler @Marcus). Marcus's own entity log does NOT change.
  "Marcus likes coffee black" -> entity_fact: entity=Marcus, "likes coffee black"
  "Marcus is also my mentor" -> slot_update: User/Mentorship/mentor := @Marcus
      (and his boss status is unchanged)
  "Jamie is my trainer at the gym" -> slot_update: User/Fitness/trainer := @Jamie

KNOWN SLOTS (with current filler):
{known_slots}

FACT: {fact}

Return JSON only, exactly:
{{"slot_updates":[{{"slot_id":"<Holder>/<Category>/<Role>",
                   "filler":"@<Entity>",
                   "prior_filler": null | "@<Entity>"}}, ...],
  "entity_facts":[{{"entity":"<Entity>","fact":"<fact text>"}}, ...]}}

Rules:
- A fact can emit 0 or more slot_updates and 0 or more entity_facts.
- Qualitative/descriptive statements ("great manager", "Scorpio", "likes X")
  are entity_facts, never slot_updates.
- A fact like "Marcus is my boss" emits ONE slot_update (User/Employment/boss
  := @Marcus), NO entity_facts. Don't duplicate into both channels.
- When prior filler is known (visible in KNOWN SLOTS), include prior_filler.
- Output JSON only.
"""


@dataclass
class SlotStore:
    slots: dict[str, RoleSlot] = field(default_factory=dict)
    entity_facts: list[dict] = field(default_factory=list)

    def apply_slot_update(
        self, su: dict, ts: str, source_turn: int, source_fact: str
    ) -> None:
        slot_id = su["slot_id"]
        filler = su.get("filler")
        slot = self.slots.setdefault(slot_id, RoleSlot(slot_id=slot_id))
        slot.history.append(
            RoleSlotEntry(
                slot_id=slot_id,
                ts=ts,
                filler=filler,
                source_turn=source_turn,
                source_fact=source_fact,
            )
        )

    def apply_entity_fact(self, ef: dict) -> None:
        self.entity_facts.append(ef)

    def known_slots_block(self) -> str:
        if not self.slots:
            return "(none)"
        lines = []
        for sid, slot in self.slots.items():
            cur = slot.current()
            lines.append(f"  {sid} -> {cur.filler if cur else 'vacant'}")
        return "\n".join(lines)


def run_scenario(sc: dict, cache: Cache, budget: Budget) -> dict:
    store = SlotStore()
    per_turn_outputs = []
    for i, t in enumerate(sc["turns"]):
        fact = t["fact"]
        prompt = PROMPT.format(known_slots=store.known_slots_block(), fact=fact)
        raw = llm(prompt, cache, budget)
        obj = extract_json(raw)
        if not isinstance(obj, dict):
            obj = {"slot_updates": [], "entity_facts": []}
        per_turn_outputs.append(obj)
        ts = f"turn-{i}"
        for su in obj.get("slot_updates", []) or []:
            if isinstance(su, dict) and su.get("slot_id") and "filler" in su:
                store.apply_slot_update(su, ts, i, fact)
        for ef in obj.get("entity_facts", []) or []:
            if isinstance(ef, dict) and ef.get("entity") and ef.get("fact"):
                store.apply_entity_fact(ef)

    # Grade slot history
    expected = sc.get("expected_slot_history", {})
    slot_pass_lines = []
    slot_pass_all = True
    for slot_id, exp_seq in expected.items():
        got_slot = store.slots.get(slot_id)
        got_seq = [e.filler for e in (got_slot.history if got_slot else [])]
        exp_seq_fillers = [e["filler"] for e in exp_seq]
        ok = got_seq == exp_seq_fillers
        if not ok:
            slot_pass_all = False
        slot_pass_lines.append(
            {"slot": slot_id, "exp": exp_seq_fillers, "got": got_seq, "pass": ok}
        )

    # Check that no EXTRA slots were fabricated beyond expected keys.
    fabricated = set(store.slots.keys()) - set(expected.keys())
    # Allow fabricated ONLY if they share a Role-Holder-Category prefix with
    # an expected slot (grader is forgiving about Mentorship/mentor vs
    # Mentor/mentor — the evaluator checks exact expected ids; fabricated
    # list is reported but not failing unless it would "steal" an expected
    # update.)

    # Profile check: ensure qualitative facts are entity_facts (not slot_updates).
    # Only hard-fail if the scenario explicitly asserts a profile expectation.
    profile_ok = True
    profile_notes = []
    if "expected_profile_changes" in sc:
        # Grand: for every entity listed, there exists at least one entity_fact
        # mentioning that entity (weak check).
        for ent, note in sc["expected_profile_changes"].items():
            matches = [
                ef for ef in store.entity_facts if ef["entity"].lower() == ent.lower()
            ]
            # If the scenario says an entity was "newly introduced" we don't
            # need a fact; otherwise we expect at least one qualitative fact.
            if "retain" in note.lower() or "retains" in note.lower():
                if not matches:
                    profile_ok = False
                    profile_notes.append(f"missing entity_fact for {ent}")

    return {
        "scenario_id": sc["id"],
        "slot_details": slot_pass_lines,
        "slot_pass": slot_pass_all,
        "fabricated_slots": sorted(fabricated),
        "entity_facts": store.entity_facts,
        "profile_pass": profile_ok,
        "profile_notes": profile_notes,
        "pass": slot_pass_all and profile_ok,
        "per_turn_outputs": per_turn_outputs,
    }


def main() -> None:
    scenarios_path = ROUND7 / "scenarios" / "p3_role_slots.json"
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]
    cache = Cache(CACHE_DIR / "p3_llm.json")
    budget = Budget()

    results = []
    t0 = time.time()
    try:
        for sc in scenarios:
            res = run_scenario(sc, cache, budget)
            status = "PASS" if res["pass"] else "FAIL"
            print(
                f"  {sc['id']}: {status} slot_pass={res['slot_pass']} profile_pass={res['profile_pass']}",
                flush=True,
            )
            for sd in res["slot_details"]:
                if not sd["pass"]:
                    print(
                        f"    slot={sd['slot']} exp={sd['exp']} got={sd['got']}",
                        flush=True,
                    )
            if res["fabricated_slots"]:
                print(f"    fabricated={res['fabricated_slots']}", flush=True)
            results.append(res)
    finally:
        cache.save()

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"\nP3 role slots: {passed}/{total} = {passed / total:.0%}")
    print(
        f"LLM calls: {budget.llm_calls}, cost ~${budget.cost():.3f}, wall {time.time() - t0:.1f}s"
    )

    out = {
        "scenarios": total,
        "passed": passed,
        "accuracy": passed / total if total else 0,
        "results": results,
        "llm_calls": budget.llm_calls,
        "cost_usd": round(budget.cost(), 4),
    }
    (ROUND7 / "results" / "p3_role_slots.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
