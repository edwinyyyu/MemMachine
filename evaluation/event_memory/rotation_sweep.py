"""How much rotation (method A demotion) works best, and does it depend on the
initial similarity c0 = cos(q, d)?

For a target cosine t with q, the demoted vector is exactly
    d'(t) = t*q + sqrt(1 - t^2) * d_perp,   d_perp = normalize(d - (q.d) q)
(this is normalize(d - alpha*q) calibrated to t == an in-plane rotation by
 theta = arccos(t) - arccos(c0)). So we can grid t directly.

Two scenarios at different c0 (a medium and a high starting similarity), demoted
for the same query, each with paraphrase / other-facet / unrelated probes.

    PYTHONPATH=<repo> uv run python evaluation/event_memory/rotation_sweep.py
"""

import asyncio
import math

import numpy as np

Q = "How do I deploy to production?"
Q_SIM = ["What's the process to deploy to prod?", "steps for a production deployment"]
Q_UNREL = ["How old am I?", "Who paid Alice?", "what animal did I see near the coffee shop?"]

# scenario name -> (doc text, [other-facet preserve queries])
SCENARIOS = {
    "MED c0  (AWS-profile doc)": (
        "I switched the deploy script to use AWS_PROFILE=prod and it now runs the "
        "database migrations automatically.",
        ["What AWS profile does deployment use?", "Does the deploy run database migrations?"],
    ),
    "HIGH c0 (deploy-steps doc)": (
        "To deploy to production: run scripts/deploy.sh, confirm the release, then "
        "watch the rollout dashboard until healthy.",
        ["How do I check the rollout dashboard is healthy?", "Where do I confirm the release?"],
    ),
}


def _n(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


async def main() -> None:
    from claude_memory.engine import build_embedder

    emb = build_embedder("embeddinggemma")
    qnames = [Q, *Q_SIM, *Q_UNREL]
    qd = {
        name: _n(np.asarray(v, dtype=float))
        for name, v in zip(qnames, await emb.search_embed(qnames), strict=True)
    }
    q = qd[Q]

    for sname, (doc, preserve) in SCENARIOS.items():
        d = _n(np.asarray((await emb.ingest_embed([doc]))[0], dtype=float))
        pv = {
            p: _n(np.asarray(v, dtype=float))
            for p, v in zip(preserve, await emb.search_embed(preserve), strict=True)
        }
        c0 = float(q @ d)
        d_perp = _n(d - (q @ d) * q)

        def cos_avg(vecs, dprime: np.ndarray) -> float:
            return float(np.mean([v @ dprime for v in vecs]))

        sim0 = cos_avg([qd[s] for s in Q_SIM], d)
        pre0 = cos_avg(list(pv.values()), d)
        unr0 = cos_avg([qd[u] for u in Q_UNREL], d)
        print(f"\n=== {sname}:  c0 = {c0:.3f} ===")
        print(f"  baseline   q_sim={sim0:.3f}  q_preserve={pre0:.3f}  q_unrel={unr0:.3f}")
        print(f"  {'t(target)':>9} {'angle°':>7} {'q':>7} {'q_sim':>7} {'q_pres':>7} {'q_unrel':>7}  {'pres_keep%':>10}")
        # grid of target cosines from c0 (no rotation) down past 0 into anti-q
        for t in [c0, c0 - 0.05, c0 - 0.10, c0 - 0.15, c0 - 0.20, c0 - 0.30, 0.0, -0.15]:
            if t > 1.0:
                continue
            tc = max(min(t, 1.0), -1.0)
            dprime = tc * q + math.sqrt(max(1 - tc * tc, 0.0)) * d_perp
            angle = math.degrees(math.acos(tc) - math.acos(c0))
            qd_ = float(q @ dprime)
            sim = cos_avg([qd[s] for s in Q_SIM], dprime)
            pre = cos_avg(list(pv.values()), dprime)
            unr = cos_avg([qd[u] for u in Q_UNREL], dprime)
            # how much of the doc's other-facet score is retained vs how much q dropped
            keep = 100.0 * (pre / pre0) if pre0 else 0.0
            print(f"  {tc:>9.2f} {angle:>7.1f} {qd_:>7.3f} {sim:>7.3f} {pre:>7.3f} {unr:>7.3f}  {keep:>9.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
