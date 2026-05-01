"""Smoke test: run the registry on S5 only (smallest scenario)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND12 = HERE.parent
ROUND11 = ROUND12.parent / "round11_writer_stress"
ROUND7 = ROUND12.parent / "round7"
sys.path.insert(0, str(ROUND12 / "architectures"))
sys.path.insert(0, str(ROUND12 / "scenarios"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen2_registry  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from coref_stress import scenario_s5  # noqa: E402

CACHE_DIR = ROUND12 / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

s = scenario_s5()
print(f"Scenario: {s.name} ({len(s.turns)} turns)")
budget = Budget(max_llm=80, max_embed=20, stop_at_llm=78, stop_at_embed=18)
cache = Cache(CACHE_DIR / f"smoke_reg_{s.name}.json")

pairs = [(t.idx, t.text) for t in s.turns]
log, idx, reg, coref_log = aen2_registry.ingest_turns_with_registry(
    pairs, cache, budget, batch_size=5, rebuild_index_every=100, lru_size=20
)
cache.save()

print(f"\nLog entries: {len(log)}")
print(f"Registry size: {len(reg.by_id)}")
print(
    f"LLM calls: {budget.llm_calls}, embed calls: {budget.embed_calls}, cost ${budget.cost():.3f}"
)
print("\nRegistry contents:")
for eid, e in reg.by_id.items():
    print(
        f"  [{eid}] aliases={sorted(e.aliases)} desc={e.description!r} last_seen=t{e.last_seen_turn}"
    )

print("\nFirst 5 coref decisions:")
for tidx in list(coref_log.keys())[:5]:
    print(f"  turn {tidx}:")
    for d in coref_log[tidx]:
        print(f"    {d.surface!r} ({d.kind}) -> {d.entity_id} [{d.rationale}]")
