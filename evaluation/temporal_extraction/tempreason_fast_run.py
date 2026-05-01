"""Fast TempReason pipeline rerun.

Monkey-patches the v2 extractor's LLM calls to use reasoning_effort=low on
gpt-5-mini, which drastically reduces per-call wall time. Keeps timeout at
60s (was 20s before).

Reuses the full `tempreason_pipeline_eval` code path after the patch.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import extractor_common

# Monkey-patch: inject reasoning_effort="low" into all gpt-5-mini calls
_orig_call = extractor_common.BaseImprovedExtractor._call


async def _patched_call(self, *args, **kwargs):
    # Inject reasoning_effort via a wrapper around the client call
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

# Also patch the base Extractor (v1)
import extractor as extractor_v1_mod

if hasattr(extractor_v1_mod, "Extractor"):
    _orig_v1_call = getattr(extractor_v1_mod.Extractor, "_llm_call", None)
    if _orig_v1_call is not None:

        async def _patched_v1_call(self, *args, **kwargs):
            original_create = self.client.chat.completions.create

            async def patched_create(**call_kwargs):
                model = call_kwargs.get("model", "")
                if isinstance(model, str) and model.startswith("gpt-5"):
                    call_kwargs["reasoning_effort"] = "minimal"
                return await original_create(**call_kwargs)

            self.client.chat.completions.create = patched_create
            try:
                return await _orig_v1_call(self, *args, **kwargs)
            finally:
                self.client.chat.completions.create = original_create

        extractor_v1_mod.Extractor._llm_call = _patched_v1_call

# Now run the existing pipeline eval with the patches in place, but with
# larger per-call timeout to avoid spurious 20s cutoffs even at minimal
# reasoning.
import tempreason_pipeline_eval as tre

# Also bump any module-level timeouts
for attr in dir(tre):
    val = getattr(tre, attr, None)
    if attr.endswith("TIMEOUT") and isinstance(val, (int, float)):
        setattr(tre, attr, 60)

if __name__ == "__main__":
    if hasattr(tre, "main"):
        asyncio.run(tre.main())
    else:
        # Fall back: execute the module body
        import runpy

        runpy.run_path(str(ROOT / "tempreason_pipeline_eval.py"), run_name="__main__")
