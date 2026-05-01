"""Smoke test: run aen3_persistent on S5 (13 turns, smallest scenario)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND13 = HERE.parent
ROUND12 = ROUND13.parent / "round12_entity_registry"
ROUND11 = ROUND13.parent / "round11_writer_stress"
ROUND7 = ROUND13.parent / "round7"
sys.path.insert(0, str(ROUND13 / "architectures"))
sys.path.insert(0, str(ROUND13 / "scenarios"))
sys.path.insert(0, str(ROUND12 / "scenarios"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen3_persistent  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from coref_stress import scenario_s5  # noqa: E402

CACHE = ROUND13 / "cache" / "smoke_aen3_s5.json"
CACHE.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    s = scenario_s5()
    cache = Cache(CACHE)
    budget = Budget(max_llm=200, max_embed=50, stop_at_llm=180, stop_at_embed=45)
    pairs = [(t.idx, t.text) for t in s.turns]
    log, idx, reg, coref_log = aen3_persistent.ingest_turns_with_registry(
        pairs,
        cache,
        budget,
        batch_size=5,
        lru_size=20,
        top_k=5,
        run_writer=False,
    )
    cache.save()
    print(f"LLM calls: {budget.llm_calls}")
    print(f"Embed calls: {budget.embed_calls}")
    print(f"Cost: ${budget.cost():.4f}")
    print(f"Entities: {len(reg.by_id)}")
    print("Coref decisions:")
    for tidx, decs in coref_log.items():
        for d in decs:
            print(
                f"  t{tidx}: {d.surface!r} ({d.kind}) -> {d.entity_id} "
                f"[{d.rationale}, embed={d.used_embedding_search}]"
            )


if __name__ == "__main__":
    main()
