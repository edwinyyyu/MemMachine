"""P4: salience-gated entity extraction.

We test Option A (LAZY): the extractor emits candidate descriptors along
with signals, and a deterministic scorer decides whether to admit them as
first-class entities or defer. A threshold of score>=2 admits.

Signals extracted per candidate (LLM, one call per turn):
  - descriptor: the noun phrase as it appeared
  - is_named: True iff the phrase contains a proper noun (e.g. "Luna")
  - has_identifying_detail: color/origin/owner/sentimental/specific-location
  - has_state_change: moves, breaks, evolves
  - grouping_key: a normalization for deduping across turns ("the bowl" and
    "grandmother's blue ceramic bowl" -> same item)

Across the whole scenario we accumulate mentions (with grouping_key dedup).
After all turns, each candidate has a score; admit if >= threshold.

Then we compare to scenario expectations (set of admitted descriptors OR
grouping keys, set of deferred).
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
from schemas import SALIENCE_ENTITY_THRESHOLD_SCORE, SalienceCandidate, salience_score

PROMPT = """Extract SALIENCE CANDIDATES from a single sentence. A candidate is
a noun phrase that could POTENTIALLY become a tracked entity in the user's
memory. PEOPLE AND PETS ALWAYS qualify. Objects/possessions/places only when
they stand out from daily-life routine.

For each candidate, emit:
  - descriptor: the noun phrase as written (minimal, 1-6 words)
  - grouping_key: a short canonical label for dedup. IMPORTANT: make groupings
    coarse. "the bowl", "grandmother's blue ceramic bowl", "grandmother's bowl"
    must share the SAME grouping_key "bowl" unless context clearly indicates
    different bowls. The purpose is to merge references to the SAME item
    across turns. Use the HEAD NOUN of the phrase as the key when ambiguous.
  - is_named: true iff the phrase contains a PROPER NOUN (e.g. "Luna",
    "Sam", "Dr. Patel"). "a mug", "the bowl", "my coffee mug" are NOT named.
  - has_identifying_detail: true ONLY for distinctive/unique-identifier detail:
    color ("the blue bowl"), origin ("grandmother's bowl"), material ("the
    ceramic bowl"), sentimental value, a unique location ("the painting in
    the living room"). Possessives like "my cup" "my fork" DO NOT count —
    "my" is routine possession, not identifying detail.
  - has_state_change: true iff the sentence PREDICATES A PERSISTENT CHANGE on
    the item's state: moved, broke, got, lost, was painted, was gifted,
    evolves. One-off passive USE like "drank coffee from a mug", "ate with a
    fork", "grabbed a tissue", "cat knocked over a glass" DO NOT count — the
    item returns to baseline. Moving the painting between rooms DOES count
    (the item's location persists). Breaking the bowl DOES count (the item
    is destroyed). A cat knocking over a mug does NOT by itself count unless
    the mug broke or moved persistently.

Skip generic actions ("drank", "ate", "signed"). Skip "I"/"me"/"my" as
candidates unless they refer to a non-user entity. Skip ingredients that
are consumed (water, coffee, pasta, dinner).

SENTENCE: {sentence}

Return JSON only:
{{"candidates":[{{"descriptor":"...","grouping_key":"...","is_named":true|false,
                 "has_identifying_detail":true|false,"has_state_change":true|false}}]}}
"""


def normalize_key(k: str) -> str:
    return (
        (k or "")
        .strip()
        .lower()
        .replace("the ", "")
        .replace("a ", "")
        .replace("my ", "")
    )


def extract_candidates(sentence: str, cache: Cache, budget: Budget) -> list[dict]:
    prompt = PROMPT.format(sentence=sentence)
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return []
    cands = obj.get("candidates") or []
    out = []
    for c in cands:
        if isinstance(c, dict) and c.get("descriptor"):
            out.append(c)
    return out


@dataclass
class SalienceStore:
    by_key: dict[str, SalienceCandidate] = field(default_factory=dict)

    def add_mention(self, turn_idx: int, c: dict) -> None:
        key = normalize_key(c.get("grouping_key") or c.get("descriptor") or "")
        if not key:
            return
        existing = self.by_key.get(key)
        if existing is None:
            existing = SalienceCandidate(
                descriptor=c.get("descriptor", key),
                first_seen_turn=turn_idx,
            )
            self.by_key[key] = existing
        else:
            existing.mention_count += 1
        # Update signals monotonically (once a signal is observed, it stays).
        if c.get("is_named"):
            existing.has_name = True
        if c.get("has_identifying_detail"):
            existing.has_identifying_detail = True
        if c.get("has_state_change"):
            existing.has_state_change = True
        existing.mentions.append(turn_idx)

    def decide(self) -> tuple[list[str], list[str]]:
        admitted, deferred = [], []
        for key, c in self.by_key.items():
            if salience_score(c) >= SALIENCE_ENTITY_THRESHOLD_SCORE:
                admitted.append(c.descriptor)
            else:
                deferred.append(c.descriptor)
        return admitted, deferred


def run_scenario(sc: dict, cache: Cache, budget: Budget) -> dict:
    store = SalienceStore()
    for i, t in enumerate(sc["turns"]):
        cands = extract_candidates(t["text"], cache, budget)
        for c in cands:
            store.add_mention(i, c)

    admitted, deferred = store.decide()

    exp_admit = set(sc.get("expected_admitted") or [])
    exp_defer = set(sc.get("expected_deferred") or [])

    def norm(s):
        return normalize_key(s)

    admitted_keys = {norm(a) for a in admitted}
    deferred_keys = {norm(d) for d in deferred}
    exp_admit_keys = {norm(a) for a in exp_admit}
    exp_defer_keys = {norm(d) for d in exp_defer}

    # Admitted: at least the expected admits must be present (after normalize).
    # Allow overlap: we primarily require no expected-admit is missing AND no
    # expected-defer was admitted.
    admit_missing = exp_admit_keys - admitted_keys
    admit_false_positive = admitted_keys & exp_defer_keys

    # Accept fuzzy matches: e.g. expected "the bowl" matches admitted
    # "grandmother's blue ceramic bowl" if the underlying grouping_key was
    # shared. We grant credit if ANY stored candidate's key is a substring
    # of the expected or vice versa.
    def fuzzy_missing(missing_set, available_set):
        still_missing = set()
        for m in missing_set:
            hit = any(m in a or a in m for a in available_set)
            if not hit:
                still_missing.add(m)
        return still_missing

    admit_missing = fuzzy_missing(admit_missing, admitted_keys)
    # admit_false_positive stricter: exact match on key is required for
    # flagging (otherwise "mug"/"bowl" overlap creates noise). It already is
    # exact via set intersection.

    ok_admit = len(admit_missing) == 0 and len(admit_false_positive) == 0
    # Don't require defer list to match 1:1 — the extractor may produce extra
    # candidates we never expected (which is fine if they correctly defer).
    # The interesting failure is "should-have-admitted but didn't".

    return {
        "scenario_id": sc["id"],
        "admitted": admitted,
        "deferred": deferred,
        "expected_admit": sorted(exp_admit),
        "expected_defer": sorted(exp_defer),
        "admit_missing": sorted(admit_missing),
        "admit_false_positive": sorted(admit_false_positive),
        "pass": ok_admit,
    }


def main() -> None:
    scenarios_path = ROUND7 / "scenarios" / "p4_salience.json"
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]
    cache = Cache(CACHE_DIR / "p4_llm.json")
    budget = Budget()

    results = []
    t0 = time.time()
    try:
        for sc in scenarios:
            res = run_scenario(sc, cache, budget)
            status = "PASS" if res["pass"] else "FAIL"
            print(f"  {sc['id']}: {status}", flush=True)
            print(f"    admitted={res['admitted']}", flush=True)
            if res["admit_missing"]:
                print(f"    MISSING admit: {res['admit_missing']}", flush=True)
            if res["admit_false_positive"]:
                print(f"    FALSE admit: {res['admit_false_positive']}", flush=True)
            results.append(res)
    finally:
        cache.save()

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"\nP4 salience: {passed}/{total} = {passed / total:.0%}")
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
    (ROUND7 / "results" / "p4_salience.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
