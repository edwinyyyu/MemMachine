"""Extreme stress: very long supersede chain at scale 1000.

15-element chain with 2 paraphrase entries per transition (so 45 paraphrase
entries total). The reader must enumerate the full chain in order.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND10 = HERE.parent
ROUND7 = ROUND10.parent / "round7"
sys.path.insert(0, str(ROUND10 / "architectures"))
sys.path.insert(0, str(ROUND10 / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_indexed as indexed  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from generators import (
    EMPLOYER_POOL,
    ENTITY_POOL,
    Entry,
    Question,
    Ref,
    _mentions,
    _uuid,
)
from grader import grade_all  # noqa: E402

CACHE_DIR = ROUND10 / "cache"
RESULTS_DIR = ROUND10 / "results"


def gen_extreme(n_entries: int, chain_len: int, seed: int = 30):
    rng = random.Random(seed)
    entries = []
    ts = 0
    employers = EMPLOYER_POOL[:chain_len]
    distractors = ENTITY_POOL[:30]

    n_canonical = chain_len
    n_paraphrase = chain_len * 2
    n_filler = max(0, n_entries - n_canonical - n_paraphrase)
    schedule = ["C"] * n_canonical + ["P"] * n_paraphrase + ["F"] * n_filler
    rng.shuffle(schedule)

    canon_uuids = []
    canon_cursor = 0
    prev = None
    paraphrases = [
        "@User mentioned the {emp} office is great.",
        "@User said the {emp} commute is easy.",
        "@User likes their {emp} colleagues.",
    ]
    for slot in schedule:
        ts += 1
        if slot == "C" and canon_cursor < n_canonical:
            emp = employers[canon_cursor]
            canon_cursor += 1
            u = _uuid("e_", ts)
            refs = [Ref(prev, "supersede")] if prev else []
            text = f"@User started a new job at {emp}."
            e = Entry(
                u,
                ts,
                text,
                _mentions("User"),
                refs,
                predicate="@User.employer",
                value=emp,
                is_current=(canon_cursor == n_canonical),
            )
            entries.append(e)
            canon_uuids.append(u)
            prev = u
        elif slot == "P" and canon_cursor > 0:
            emp = employers[canon_cursor - 1]
            t = rng.choice(paraphrases).format(emp=emp)
            u = _uuid("e_", ts)
            refs = [Ref(canon_uuids[canon_cursor - 1], "clarify")]
            entries.append(Entry(u, ts, t, _mentions("User"), refs))
        else:
            n2 = rng.choice(distractors)
            t = rng.choice(
                [
                    f"@User and @{n2} chatted today.",
                    f"@{n2} sent @User a meme.",
                    f"@User had coffee with @{n2}.",
                    f"@{n2} mentioned a book to @User.",
                ]
            )
            entries.append(Entry(_uuid("e_", ts), ts, t, _mentions("User", n2), []))
        if len(entries) >= n_entries:
            break

    qs = [
        Question(
            "X1_Q01",
            "current",
            "What is User's current employer?",
            expected_contains=[employers[-1]],
        ),
        Question(
            "X1_Q02",
            "history",
            "List User's employers in chronological order, from earliest to most recent.",
            expected_contains=employers,
        ),
        Question(
            "X1_Q03",
            "supersede",
            f"Did User ever work at {employers[3]}?",
            expected_contains=["yes"],
        ),
        Question(
            "X1_Q04",
            "supersede",
            f"Is User still at {employers[0]}?",
            expected_contains=["no"],
        ),
        Question(
            "X1_Q05",
            "supersede",
            f"What employer did User work at right after {employers[chain_len // 2]}?",
            expected_contains=[employers[chain_len // 2 + 1]],
        ),
    ]
    return entries, qs


def main():
    budget = Budget(max_llm=300, max_embed=200, stop_at_llm=290, stop_at_embed=190)
    runs = []
    for scale, chain_len in [(1000, 15), (1000, 18), (2000, 18)]:
        print(f"\n=== extreme @ {scale} entries, chain={chain_len} ===")
        try:
            entries, qs = gen_extreme(scale, chain_len)
            cache = Cache(CACHE_DIR / f"extreme_{scale}_{chain_len}.json")
            idx = indexed.build_index(entries, cache, budget)
            per_arch = {}
            for arch_name in ["aen1_indexed", "aen1_plain"]:
                answers = {}
                for q in qs:
                    if arch_name == "aen1_indexed":
                        ans = indexed.answer_indexed(q.question, idx, cache, budget)
                    else:
                        ans = indexed.answer_plain(
                            q.question, entries, idx.embed_by_uuid, cache, budget
                        )
                    answers[q.qid] = ans
                cache.save()
                verdicts = grade_all(qs, answers)
                passed = sum(1 for v in verdicts if v.passed)
                per_arch[arch_name] = {
                    "answers": answers,
                    "verdicts": [asdict(v) for v in verdicts],
                    "passed": passed,
                    "total": len(verdicts),
                }
                print(f"  [{arch_name}] {passed}/{len(verdicts)}")
            runs.append({"scale": scale, "chain_len": chain_len, "archs": per_arch})
            cache.save()
        except RuntimeError as e:
            print(f"  BUDGET STOP: {e}")
            break
        print(
            f"  Cost: ${budget.cost():.3f}, LLM={budget.llm_calls}, embed={budget.embed_calls}"
        )

    out = RESULTS_DIR / "extreme_sweep.json"
    out.write_text(
        json.dumps(
            {
                "final_cost": budget.cost(),
                "llm_calls": budget.llm_calls,
                "embed_calls": budget.embed_calls,
                "runs": runs,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
