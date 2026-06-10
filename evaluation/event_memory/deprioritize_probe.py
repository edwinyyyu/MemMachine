"""Probe: can we DEMOTE a stored doc vector for a query (and its paraphrases)
WITHOUT damaging the doc's score for its other facets / unrelated queries?

Selective demotion is the whole point — otherwise just delete the vector. So for
each method, calibrate it to the SAME target reduction on the demote-query, then
measure the collateral on:
  q          - the demote query                  (want: DOWN to target)
  q_sim      - paraphrases of q                   (want: DOWN too — generalizes)
  q_preserve - OTHER facets of the same doc       (want: ~UNCHANGED — selective)
  q_unrel    - unrelated queries                  (want: ~UNCHANGED)

Methods:
  A rocchio/rotate   d' = normalize(d - alpha*q)          (== in-plane rotation)
  B orth-projection  d' = normalize(d - (q.d) q)          (A at target 0)
  C scale-pos-dims   shrink the dims with largest q_i*d_i, renormalize
  D scale-neg-dims   amplify the dims with most negative q_i*d_i, renormalize
  E promote-good     leave d; pull the GOOD doc g toward q (rank-based)

    PYTHONPATH=<repo> uv run python evaluation/event_memory/deprioritize_probe.py
"""

import asyncio
from collections.abc import Callable

import numpy as np

DOC = (
    "I switched the deploy script to use AWS_PROFILE=prod and it now runs the "
    "database migrations automatically."
)
GOOD = (
    "To deploy to production: run scripts/deploy.sh, confirm the release, then "
    "watch the rollout dashboard until healthy."
)
Q = "How do I deploy to production?"
Q_SIM = ["What's the process to deploy to prod?", "steps for a production deployment"]
Q_PRESERVE = [
    "What AWS profile does deployment use?",
    "Does the deploy run database migrations?",
]
Q_UNREL = [
    "How old am I?",
    "Who paid Alice?",
    "What animal did I see near the coffee shop?",
]
TARGET_DROP = 0.15  # reduce cos(q, d) by this much; calibrate every method to it


def _n(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _calibrate(
    make: Callable[[float], np.ndarray], hi: float, target: float, q: np.ndarray
) -> float:
    """Find param in [0, hi] s.t. cos(q, make(param)) ~= target (cos decreasing)."""
    lo, hi_b, best = 0.0, hi, hi
    for _ in range(48):
        mid = (lo + hi_b) / 2
        if float(q @ make(mid)) > target:
            lo = mid
        else:
            hi_b, best = mid, mid
    return best


def build_demotion_variants(
    d: np.ndarray, q: np.ndarray, target: float
) -> dict[str, np.ndarray]:
    """Each demotion method, calibrated to the same target cos(q, d')."""

    def rocchio(alpha: float) -> np.ndarray:
        return _n(d - alpha * q)

    def scale_pos(strength: float, frac: float = 0.1) -> np.ndarray:
        idx = np.argsort(q * d)[-max(1, int(len(d) * frac)) :]
        d2 = d.copy()
        d2[idx] *= 1.0 - strength
        return _n(d2)

    def scale_neg(strength: float, frac: float = 0.1) -> np.ndarray:
        idx = np.argsort(q * d)[: max(1, int(len(d) * frac))]
        d2 = d.copy()
        d2[idx] *= 1.0 + strength
        return _n(d2)

    return {
        "baseline (d)": d,
        "A rocchio/rotate": rocchio(_calibrate(rocchio, 4.0, target, q)),
        "B orth-projection": _n(d - (q @ d) * q),
        "C scale-pos-dims": scale_pos(_calibrate(scale_pos, 1.0, target, q)),
        "D scale-neg-dims": scale_neg(_calibrate(scale_neg, 200.0, target, q)),
    }


async def promote_section(emb, q: np.ndarray, d: np.ndarray, g: np.ndarray) -> None:
    """E: leave d untouched; pull the GOOD doc toward q (rank-based view)."""
    print("\n================ E: promote the good doc (don't touch d) ================")
    corpus = {"TARGET d": d, "GOOD g": g}
    for i, x in enumerate(Q_UNREL):
        corpus[f"distractor{i}"] = _n(np.asarray((await emb.ingest_embed([x]))[0], dtype=float))

    def rank(qv: np.ndarray, store: dict[str, np.ndarray]) -> list[tuple[str, float]]:
        return sorted(((k, round(float(qv @ v), 3)) for k, v in store.items()), key=lambda t: -t[1])

    g2 = _n(g + 0.6 * q)  # modest promote toward q (Rocchio on the good doc)
    print("rank for q BEFORE:", rank(q, corpus))
    print("cos(q, GOOD) before/after:", round(float(q @ g), 3), "->", round(float(q @ g2), 3))
    print("rank for q AFTER :", rank(q, {**corpus, "GOOD g": g2}))
    print("(TARGET d's absolute score is unchanged; only its RANK vs GOOD drops.)")


async def main() -> None:
    from claude_memory.engine import build_embedder

    emb = build_embedder("embeddinggemma")
    d = _n(np.asarray((await emb.ingest_embed([DOC]))[0], dtype=float))
    g = _n(np.asarray((await emb.ingest_embed([GOOD]))[0], dtype=float))
    qnames = [Q, *Q_SIM, *Q_PRESERVE, *Q_UNREL]
    qd = {
        name: _n(np.asarray(v, dtype=float))
        for name, v in zip(qnames, await emb.search_embed(qnames), strict=True)
    }
    q = qd[Q]
    c0 = float(q @ d)
    target = c0 - TARGET_DROP
    groups = {"q": [Q], "q_sim": Q_SIM, "q_preserve": Q_PRESERVE, "q_unrel": Q_UNREL}

    print(f"baseline cos(q,d) = {c0:.3f}   target = {target:.3f} (drop {TARGET_DROP})\n")
    print("how separable each group is from q (query-side cosine):")
    for gname, qs in groups.items():
        if gname != "q":
            sims = [round(float(q @ qd[x]), 2) for x in qs]
            print(f"  q . {gname:<11} = {np.mean(sims):.3f}  {sims}")

    variants = build_demotion_variants(d, q, target)
    base = {gn: float(np.mean([qd[x] @ d for x in qs])) for gn, qs in groups.items()}

    print("\n" + f"{'method':<20}" + "".join(f"{gn:>13}" for gn in groups))
    for vname, dv in variants.items():
        cells = [float(np.mean([qd[x] @ dv for x in qs])) for qs in groups.values()]
        print(f"{vname:<20}" + "".join(f"{c:>13.3f}" for c in cells))

    print("\nΔ from baseline (negative = demoted):")
    for vname, dv in variants.items():
        if not vname.startswith("baseline"):
            cells = [
                float(np.mean([qd[x] @ dv for x in qs])) - base[gn]
                for gn, qs in groups.items()
            ]
            print(f"{vname:<20}" + "".join(f"{c:>+13.3f}" for c in cells))

    await promote_section(emb, q, d, g)


if __name__ == "__main__":
    asyncio.run(main())
