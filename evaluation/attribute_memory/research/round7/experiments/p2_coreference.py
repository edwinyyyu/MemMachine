"""P2: context buffer for anonymous->named coreference.

Design:
  - Per-turn, one LLM call (shared with extraction) labels each fact as
    containing an `anonymous_descriptor` ("my boss", "my coworker", "the
    dentist"), a `named_entity` introduction ("Marcus", "Dr. Patel"), or
    neither.
  - We maintain a ring buffer of unresolved anonymous mentions (last
    COREF_BUFFER_MAX_TURNS turns, or MAX_MENTIONS entries).
  - When a turn introduces a named entity, one LLM call asks:
      "Does `<named>` resolve any of these anonymous mentions? If yes, which?"
    with the full buffer supplied.
  - Match -> emit `CoreferenceMerge` signal AND drop the resolved mention
    from the buffer.

To keep costs bounded we: (a) only invoke the coref-check LLM call when the
incoming turn *actually* contains a named entity, AND (b) only when the
buffer is non-empty.

This experiment simulates the per-turn signals as if the extractor had
already flagged them (scenarios provide `descriptor` and `named_intro` hints).
That isolates the coref-merge decision from the extraction call, which is a
separate problem.
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
    AnonymousMention,
    CoreferenceMerge,
)

COREF_PROMPT = """You are resolving a cross-turn coreference.

A named entity was just introduced in a conversation. Decide which (if any) of
the previously-buffered anonymous descriptors this named entity resolves.

NAMED ENTITY (just introduced): {named}
INTRODUCING TURN TEXT: "{introducing_text}"

BUFFERED ANONYMOUS MENTIONS (turn_idx :: descriptor :: fact):
{buffer_lines}

Return JSON only:
{{"merges":[{{"descriptor": "<buffered descriptor verbatim>",
             "rationale": "<short reason>"}}]}}

Rules:
- Only merge when the introducing turn text makes the link unambiguous
  (co-mention in same sentence, appositive, or explicit resolution like
  "Marcus, my boss" or "his name is Marcus").
- If no clear link, return {{"merges": []}}.
- Do not merge distractor names (someone casually mentioned who isn't the
  referent of any descriptor).
"""


@dataclass
class CorefBuffer:
    mentions: deque[AnonymousMention] = field(default_factory=deque)

    def add(self, m: AnonymousMention) -> None:
        self.mentions.append(m)
        self._prune(m.turn_idx)

    def _prune(self, current_turn: int) -> None:
        # Drop stale entries by turn distance.
        while (
            self.mentions
            and (current_turn - self.mentions[0].turn_idx) > COREF_BUFFER_MAX_TURNS
        ):
            self.mentions.popleft()
        while len(self.mentions) > COREF_BUFFER_MAX_MENTIONS:
            self.mentions.popleft()

    def active(self, current_turn: int) -> list[AnonymousMention]:
        self._prune(current_turn)
        return list(self.mentions)

    def remove_by_descriptor(self, descriptor: str) -> int:
        # Remove the FIRST mention matching this descriptor (canonical case:
        # first unresolved mention wins).
        for i, m in enumerate(self.mentions):
            if m.descriptor == descriptor:
                del self.mentions[i]
                return 1
        return 0


def resolve_coref(
    named: str,
    introducing_text: str,
    buffer: CorefBuffer,
    turn_idx: int,
    cache: Cache,
    budget: Budget,
) -> list[CoreferenceMerge]:
    active = buffer.active(turn_idx)
    if not active:
        return []
    lines = "\n".join(
        f'  t{m.turn_idx} :: {m.descriptor} :: "{m.fact_text}"' for m in active
    )
    prompt = COREF_PROMPT.format(
        named=named,
        introducing_text=introducing_text,
        buffer_lines=lines,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    merges: list[CoreferenceMerge] = []
    if isinstance(obj, dict) and isinstance(obj.get("merges"), list):
        for m in obj["merges"]:
            if not isinstance(m, dict):
                continue
            desc = m.get("descriptor")
            if not desc:
                continue
            # Find the buffered mention with matching descriptor.
            candidates = [bm for bm in active if bm.descriptor == desc]
            if not candidates:
                continue
            bm = candidates[0]
            merges.append(
                CoreferenceMerge(
                    canonical_entity=named,
                    anonymous_topic=bm.topic,
                    anonymous_descriptor=desc,
                    matched_mention_turn_idx=bm.turn_idx,
                    rationale=m.get("rationale", ""),
                )
            )
            buffer.remove_by_descriptor(desc)
    return merges


def run_scenario(sc: dict, cache: Cache, budget: Budget) -> dict:
    buffer = CorefBuffer()
    merges_all: list[CoreferenceMerge] = []

    for turn in sc["turns"]:
        desc = turn.get("descriptor")
        named = turn.get("named_intro")
        idx = turn["idx"]
        text = turn["text"]

        # If this turn BOTH introduces a name and has an anon descriptor
        # (like "My boss is Marcus"), handle both in same step. The name-resolve
        # happens before the descriptor gets buffered (so self-resolves in
        # same turn work without needing buffer).
        if desc:
            # Buffer the mention (do BEFORE resolve so the named-intro in same
            # turn can see it).
            buffer.add(
                AnonymousMention(
                    turn_idx=idx,
                    descriptor=desc,
                    topic=turn.get("topic_hint", "User/Other"),
                    fact_text=text,
                )
            )
        if named:
            ms = resolve_coref(named, text, buffer, idx, cache, budget)
            merges_all.extend(ms)

    # Grade
    expected = sc.get("expected_merges", [])
    expected_pairs = {(m["canonical"], m["descriptor"]) for m in expected}
    got_pairs = {(m.canonical_entity, m.anonymous_descriptor) for m in merges_all}
    correct = expected_pairs & got_pairs
    missing = expected_pairs - got_pairs
    extra = got_pairs - expected_pairs

    return {
        "scenario_id": sc["id"],
        "expected": sorted(expected_pairs),
        "got": sorted(got_pairs),
        "correct": len(correct),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "pass": (len(missing) == 0 and len(extra) == 0),
    }


def main() -> None:
    scenarios_path = ROUND7 / "scenarios" / "p2_coreference.json"
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]
    cache = Cache(CACHE_DIR / "p2_llm.json")
    budget = Budget()

    results = []
    t0 = time.time()
    try:
        for sc in scenarios:
            res = run_scenario(sc, cache, budget)
            status = "PASS" if res["pass"] else "FAIL"
            print(
                f"  {sc['id']}: {status} got={res['got']} missing={res['missing']} extra={res['extra']}",
                flush=True,
            )
            results.append(res)
    finally:
        cache.save()

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"\nP2 coreference: {passed}/{total} = {passed / total:.0%}")
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
    (ROUND7 / "results" / "p2_coreference.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
