"""P1: multi-label routing with a strict gate.

The gate explicitly asks the LLM a decision-tree question:

1. Does this fact INTRODUCE a new entity (not yet in known_entities)?
   -> yes: multi-label to (subject, new_entity).
2. Is this fact a RELATIONSHIP-MAKING event in which TWO OR MORE parties
   undergo a state change (engagement, adoption, mentoring, etc.)?
   -> yes: multi-label to each participating entity.
3. Otherwise -> single subject only.

Key anti-pattern to avoid: splitting "User is a nurse and diabetic" into
two topics. The gate handles this by treating 'multi-attribute single-subject'
as single-label.

We emit structured JSON:
  {
    "reason": "new_entity" | "relationship_event" | "single",
    "topics": ["<Subject>/<Category>", ...],
    "introduced_entities": ["<Entity>", ...]
  }
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND7 = HERE.parent
sys.path.insert(0, str(ROUND7))
sys.path.insert(0, str(HERE))

from _common import CACHE_DIR, Budget, Cache, extract_json, llm

PROMPT = """You are routing a single fact to one or more append-only topic logs.

HARD RULE: emit multiple topics ONLY in these two cases:
  (A) NEW_ENTITY — the fact introduces an entity not yet in the known-entities list.
      Then: route to (subject, introduced_entity). Example:
        Fact: "User got a new cat named Luna" (Luna not in known list)
        -> topics: ["User/Possessions", "Luna/Profile"], reason: "new_entity"
  (B) RELATIONSHIP_EVENT — the fact is primarily about a relationship or state
      being *made or changed* between 2+ named parties, each of whom undergoes
      a state change. Examples:
        "Jamie and Alex got engaged"     -> both (relationship state changed)
        "Marcus offered to mentor User"  -> both (new mentor-mentee relation)
        "Jamie convinced User to adopt Luna" -> three parties if all named
      Then: route to each party.

SINGLE SUBJECT (default): everything else. In particular:
  - "User is a nurse and diabetic" -> SINGLE subject User. Two attributes, one person.
  - "User's sister lives in Portland" -> SINGLE subject Sister. Fact is about Sister.
  - "User is 34 years old" -> SINGLE subject User.

TOPIC FORMAT: <Entity>/<Category> e.g. "User/Employment", "Luna/Profile",
"Jamie/Relationships", "Marcus/Profile".

KNOWN ENTITIES (an entity is "new" iff NOT in this list):
{known_entities}

FACT: {fact}

Output a JSON object, exactly:
{{"reason": "single" | "new_entity" | "relationship_event",
  "topics": ["<Entity>/<Category>", ...],
  "introduced_entities": ["<Entity>", ...]}}

Output JSON only, no markdown, no commentary.
"""


def route(fact: str, known_entities: list[str], cache: Cache, budget: Budget) -> dict:
    known = ", ".join(known_entities) if known_entities else "(none)"
    prompt = PROMPT.format(known_entities=known, fact=fact)
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return {"reason": "single", "topics": ["User/Other"], "introduced_entities": []}
    return obj


def grade_scenario(sc: dict, result: dict) -> dict:
    """Grade one scenario by comparing result to expected.

    Accepts the result as correct iff:
      - `topics` is a set-subset/superset relationship with the expected set
        under the scenario-specific policy.
      - `reason` matches expected (if specified).
    """
    out_topics = set(result.get("topics") or [])
    out_reason = result.get("reason", "")

    # Compare topic sets. If scenario provides expected_topics, check exact match
    # on normalized entity portion (allow category mismatch e.g. Profile vs
    # Possessions). We focus on "did multi-label fire correctly?" and "are the
    # right subjects present?".
    exp_topics_all = sc.get("expected_topics_any_of", [sc.get("expected_topics", [])])
    not_both = sc.get("expected_topics_not_both", False)

    def entity_part(tn: str) -> str:
        return tn.split("/", maxsplit=1)[0].lower()

    out_entities = {entity_part(t) for t in out_topics}

    passed_any = False
    for exp in exp_topics_all:
        exp_entities = {entity_part(t) for t in exp}
        if not_both:
            # Expect EXACTLY one entity-subject. If the output has more, fail.
            if len(out_entities) == 1 and out_entities.issubset(
                exp_entities | {list(exp_entities)[0]}
            ):
                passed_any = True
                break
        else:
            # Need to match the entity set (order-insensitive). For triangle
            # events we allow >= 2 of the expected 3.
            required = exp_entities
            if len(required) >= 3:
                if len(out_entities & required) >= 2:
                    passed_any = True
                    break
            else:
                if out_entities == required:
                    passed_any = True
                    break

    reason_ok = (
        (sc.get("expected_reason") == out_reason) if sc.get("expected_reason") else True
    )

    return {
        "scenario_id": sc["id"],
        "out_topics": sorted(out_topics),
        "out_reason": out_reason,
        "pass_topics": passed_any,
        "pass_reason": reason_ok,
        "pass": passed_any and reason_ok,
    }


def main() -> None:
    scenarios_path = ROUND7 / "scenarios" / "p1_multilabel.json"
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]

    cache = Cache(CACHE_DIR / "p1_llm.json")
    budget = Budget()

    results = []
    t0 = time.time()
    try:
        for sc in scenarios:
            res = route(sc["fact"], sc.get("known_entities", []), cache, budget)
            grade = grade_scenario(sc, res)
            grade["raw_result"] = res
            grade["fact"] = sc["fact"]
            grade["expected_reason"] = sc.get("expected_reason")
            results.append(grade)
            status = "PASS" if grade["pass"] else "FAIL"
            print(
                f"  {sc['id']}: {status} topics={grade['out_topics']} reason={grade['out_reason']}",
                flush=True,
            )
    finally:
        cache.save()

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"\nP1 multi-label gate: {passed}/{total} = {passed / total:.0%}")
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
    (ROUND7 / "results" / "p1_multilabel.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
