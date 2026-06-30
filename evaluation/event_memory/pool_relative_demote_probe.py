"""Score-free, pool-relative demotion interface — WITH an absence/floor guard.

The model never supplies a number/score (the final score may be a fusion/rerank
output that's hard to invert). It signals strength ("mild"/"strong") on a memory
in the current query context; the SYSTEM auto-tunes the rotation in cosine space
from c0 = cos(q,d) and the pool's raw cosines to q.

PITFALL handled here: if there is nothing more relevant below the demoted items
(the query simply has no good memory), the model must be TOLD so — otherwise it
demotes forever and corrupts the store. We compute a relevance FLOOR (corpus
random-pair baseline ~0.42 p90, measured in sphere_density_probe) and report,
on every demote, whether a meaningfully-relevant alternative remains:
  - best remaining > floor   -> "better exists; demotion surfaces it"
  - best remaining <= floor   -> "ABSENCE: nothing more relevant; stop demoting"
  - whole pool <= floor       -> decline: query has no relevant memory at all

    PYTHONPATH=<repo> uv run python evaluation/event_memory/pool_relative_demote_probe.py
"""

import asyncio
import math

import numpy as np

FLOOR = 0.42  # corpus random-pair p90 (sphere_density_probe): above => non-random
MARGIN = 0.02

POOL = [
    (
        "GOOD-steps",
        "To deploy to production: run scripts/deploy.sh, confirm the release, then watch the rollout dashboard until healthy.",
    ),
    (
        "CI-pipeline",
        "Production deploys go through the CI pipeline; merging to main triggers the rollout.",
    ),
    (
        "AWS-profile d",
        "I switched the deploy script to use AWS_PROFILE=prod and it now runs the database migrations automatically.",
    ),
    (
        "staging",
        "The staging environment uses a separate AWS account and a nightly data refresh.",
    ),
    ("api-keys", "Remember to rotate the API keys quarterly for compliance."),
    ("raccoon", "I saw a raccoon near the coffee shop on my way to work."),
]
QUERIES = {
    "HAS-answer": "How do I deploy to production?",
    "NO-answer ": "What's a good recipe for sourdough bread?",
}


def _n(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def auto_target(c0: float, pool_cos: list[float], strength: str) -> float | None:
    """Target cosine for d, tuned ONLY against above-floor competitors and clamped
    to >= floor (so 'strong' demotes to the noise level, never past the cliff).
    Returns None if no relevant competitor exists (caller treats as absence)."""
    above = [c for c in pool_cos if c > FLOOR]
    if not above:
        return None
    base = float(np.median(above)) if strength == "mild" else FLOOR
    t = base - MARGIN
    return max(min(t, c0 - MARGIN), FLOOR - MARGIN)


def rotate_to(d: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    c0 = float(q @ d)
    d_perp = _n(d - c0 * q)
    tc = max(min(t, 1.0), -1.0)
    return tc * q + math.sqrt(max(1 - tc * tc, 0.0)) * d_perp


async def main() -> None:
    from claude_memory.engine import build_embedder

    emb = build_embedder("embeddinggemma")
    docs = {
        name: _n(np.asarray((await emb.ingest_embed([txt]))[0], dtype=float))
        for name, txt in POOL
    }

    for label, qtext in QUERIES.items():
        q = _n(np.asarray((await emb.search_embed([qtext]))[0], dtype=float))
        ranked = sorted(
            ((k, float(q @ v)) for k, v in docs.items()), key=lambda kv: -kv[1]
        )
        top_name, top_cos = ranked[0]
        print(f"\n=== {label.strip()}: {qtext!r}  (floor={FLOOR}) ===")
        for k, s in ranked:
            flag = "" if s > FLOOR else "  (<= floor: ~noise)"
            print(f"   {s:>6.3f}  {k}{flag}")

        # absence guard BEFORE acting on a demote of the current top item
        if top_cos <= FLOOR:
            print(
                "  VERDICT: whole pool <= floor -> NO relevant memory for this "
                "query. Demotion declined; tell the model 'nothing relevant exists'."
            )
            continue

        best_remaining = ranked[1][1]  # best after demoting the top item
        if best_remaining <= FLOOR:
            print(
                f"  VERDICT: top is the only above-floor hit (next-best "
                f"{best_remaining:.3f} <= floor). Demoting it surfaces only noise "
                "-> tell the model 'no BETTER memory exists; stop demoting'."
            )
            continue

        # there IS a relevant alternative -> demotion is meaningful
        d = docs[top_name]
        c0 = top_cos
        pool_cos = [float(q @ v) for k, v in docs.items() if k != top_name]
        print(
            f"  VERDICT: {ranked[1][0]} ({best_remaining:.3f}) is above floor -> "
            f"a relevant alternative exists; demotion is meaningful."
        )
        for strength in ("mild", "strong"):
            t = auto_target(c0, pool_cos, strength)
            d2 = rotate_to(d, q, t)
            new = sorted(
                ((k, float(q @ (d2 if k == top_name else v))) for k, v in docs.items()),
                key=lambda kv: -kv[1],
            )
            new_rank = [k for k, _ in new].index(top_name) + 1
            print(
                f"    {strength:<6} auto t={t:.3f} (q-drop {c0 - t:.3f}, no score): "
                f"rank 1 -> {new_rank}; new top = {new[0][0]} ({new[0][1]:.3f})"
            )


if __name__ == "__main__":
    asyncio.run(main())
