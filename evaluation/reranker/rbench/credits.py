"""Running out of DeepInfra credits: one clear stop on every paid path, nothing finished lost.

There is no spend cap (the user's decision, 2026-11-06: "remove max usd argument if it makes
things complicated"; 2026-11-09: "Can the cap just be based on what credits I have on
DeepInfra?"). DeepInfra's prepaid credits are the cap, so its auto top-up should be off, and
`python -m rbench estimate` is the price check before a run. The call logs still record what
each run spent (rerankers.ScoreCache.spent_usd).

When the balance runs out mid-run:
  reranking  every request is cached and logged as it returns; the group in flight finishes
             its other requests (they are cached), then the run stops
  embedding  the partition being encoded gets no done marker, so the rerun deletes it and
             encodes it again (memory.Memory.ingest; about a cent of Qwen3-Embedding-8B);
             every partition encoded before it is kept
Either way the run ends with MESSAGE, and rerunning the same command after a top-up resumes.

DeepInfra does not document its out-of-balance answer (searched 2026-11-09), so both likely
forms are recognized: HTTP 402 (Payment Required), or any other 4xx whose body names the
balance, credits, funds or payment.
"""

from __future__ import annotations

import re

MESSAGE = "DeepInfra is out of credits: top up and rerun; finished work is kept"
_BODY = re.compile(r"balance|credit|funds|payment|billing", re.IGNORECASE)


class OutOfCredits(RuntimeError):
    """A paid request refused for lack of credits (an Exception, not SystemExit, so requests
    still in flight finish and are cached before the run stops)."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(MESSAGE + (f" ({detail})" if detail else ""))


def refused(status: int | None, body: str) -> bool:
    """Whether a response is DeepInfra refusing for lack of credits."""
    if status == 402:
        return True
    return status is not None and 400 <= status < 500 and status != 429 and bool(_BODY.search(body or ""))


def from_embedder(err: BaseException) -> OutOfCredits | None:
    """An embedding failure as OutOfCredits, if it is one: MemMachine's OpenAIEmbedder raises
    ExternalServiceAPIError from the openai APIStatusError it got."""
    import openai

    cause = err if isinstance(err, openai.APIStatusError) else err.__cause__
    if isinstance(cause, openai.APIStatusError) and refused(cause.status_code, _text(cause)):
        return OutOfCredits(f"embedding: HTTP {cause.status_code}")
    return None


def _text(err) -> str:
    try:
        return err.response.text
    except Exception:  # noqa: BLE001 - a body that cannot be read names nothing
        return str(err)
