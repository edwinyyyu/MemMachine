"""Recompute R@1, R@3 on TempReason small from cached extractions.

All LLM calls hit cache, so this is math only.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import extractor_common
import tempreason_pipeline_eval as tre

# Monkey-patch: reasoning_effort=minimal for any gpt-5 call
_orig_call = extractor_common.BaseImprovedExtractor._call


async def _patched_call(self, *args, **kwargs):
    original_create = self.client.chat.completions.create

    async def patched_create(**call_kwargs):
        model = call_kwargs.get("model", "")
        if isinstance(model, str) and model.startswith("gpt-5"):
            call_kwargs["reasoning_effort"] = "minimal"
        return await original_create(**call_kwargs)

    self.client.chat.completions.create = patched_create
    try:
        return await _orig_call(self, *args, **kwargs)
    finally:
        self.client.chat.completions.create = original_create


extractor_common.BaseImprovedExtractor._call = _patched_call

# Patch eval_rankings to include R@1 and R@3
_orig_eval = tre.eval_rankings


def _eval_rankings_with_r1_r3(ranked_per_q, gold, qids):
    base = _orig_eval(ranked_per_q, gold, qids)
    r1 = []
    r3 = []
    for qid in qids:
        rel = set(gold.get(qid, []))
        if not rel:
            continue
        ranked = ranked_per_q.get(qid, [])
        r1.append(1.0 if any(d in rel for d in ranked[:1]) else 0.0)
        r3.append(1.0 if any(d in rel for d in ranked[:3]) else 0.0)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    base["recall@1"] = mean(r1)
    base["recall@3"] = mean(r3)
    return base


tre.eval_rankings = _eval_rankings_with_r1_r3

# Re-run
if __name__ == "__main__":
    asyncio.run(tre.main()) if hasattr(tre, "main") else None
    if not hasattr(tre, "main"):
        import runpy

        runpy.run_path(str(ROOT / "tempreason_pipeline_eval.py"), run_name="__main__")
