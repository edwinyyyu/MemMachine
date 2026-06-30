"""How densely do MEANINGFUL embeddings fill the hypersphere, and does demoting a
vector (method A rotation) push it into an unrelated populated region?

1. Sample ~1500 real message texts from transcripts, embed (document mode) -> C.
2. Geometry of the meaningful manifold:
     - centroid norm  (anisotropy: 0 = isotropic on the sphere, 1 = a single ray)
     - pairwise-cosine distribution (how wide the "cone" is)
     - nearest-neighbor cosine + how many neighbors within cos thresholds (density)
3. Displacement test: rotate a target doc d away from a query q across t, and at
   each step report max cos(d', C) and #neighbors above thresholds + the nearest
   corpus texts -> does demotion gain SPURIOUS neighbors (land in a new cluster)?

    PYTHONPATH=<repo> uv run python evaluation/event_memory/sphere_density_probe.py
"""

import asyncio
import math
from pathlib import Path

import numpy as np

Q = "How do I deploy to production?"
DOC = (
    "I switched the deploy script to use AWS_PROFILE=prod and it now runs the "
    "database migrations automatically."
)
N_TARGET = 1500
PER_FILE = 25


def _n(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def _sample_texts() -> list[str]:
    from claude_memory.transcript import events_from_transcript
    from claude_memory.wire import Source

    projects = Path.home() / ".claude" / "projects"
    files = sorted(projects.glob("*/*.jsonl"), key=lambda p: p.stat().st_size)
    texts: list[str] = []
    seen: set[str] = set()
    for fp in files:
        events, _ = events_from_transcript(fp, session_id=fp.stem, start_line=0)
        msgs = [
            e.blocks[0].text.strip()
            for e in events
            if e.properties.get("source")
            in (Source.USER_MESSAGE, Source.ASSISTANT_MESSAGE)
            and e.blocks
            and e.blocks[0].text.strip()
        ]
        for t in msgs[:PER_FILE]:
            key = t[:200]
            if key not in seen:
                seen.add(key)
                texts.append(t[:2000])
        if len(texts) >= N_TARGET:
            break
    return texts[:N_TARGET]


async def _embed(emb, texts: list[str]) -> np.ndarray:
    out: list[list[float]] = []
    for i in range(0, len(texts), 256):
        out.extend(await emb.ingest_embed(texts[i : i + 256]))
    return _n(np.asarray(out, dtype=float))


async def main() -> None:
    from claude_memory.engine import build_embedder

    emb = build_embedder("embeddinggemma")
    texts = _sample_texts()
    c_mat = await _embed(emb, texts)
    n = len(c_mat)
    print(f"sampled {n} real message embeddings (document mode)\n")

    # --- geometry ---
    centroid = c_mat.mean(axis=0)
    print(
        f"centroid norm = {np.linalg.norm(centroid):.3f}  "
        "(0=isotropic on sphere, 1=single ray; >0 => a cone)"
    )
    rng = np.random.default_rng(0)
    ia = rng.integers(0, n, 60000)
    ib = rng.integers(0, n, 60000)
    mask = ia != ib
    pair = np.sum(c_mat[ia[mask]] * c_mat[ib[mask]], axis=1)
    qs = np.percentile(pair, [50, 90, 99])
    print(
        f"pairwise cosine (random pairs): mean {pair.mean():.3f}  "
        f"p50 {qs[0]:.3f}  p90 {qs[1]:.3f}  p99 {qs[2]:.3f}  max {pair.max():.3f}"
    )
    # nearest-neighbor + density at thresholds (subsample for the full NxN slice)
    sub = c_mat[rng.choice(n, min(400, n), replace=False)]
    sims = sub @ c_mat.T
    for i in range(len(sub)):  # null self-matches
        sims[i][np.argmax(sims[i])] = -1
    nn = sims.max(axis=1)
    print(
        f"nearest-neighbor cosine: p50 {np.percentile(nn, 50):.3f}  "
        f"p90 {np.percentile(nn, 90):.3f}  max {nn.max():.3f}"
    )
    for tau in (0.5, 0.6, 0.7, 0.8):
        avg = float(np.mean(np.sum(sims > tau, axis=1)))
        print(f"  avg #corpus neighbors with cos > {tau}: {avg:.1f}  (of {n})")

    # --- displacement test ---
    print("\n--- displacement: rotate DOC away from Q, watch corpus neighbors ---")
    q = _n(np.asarray((await emb.search_embed([Q]))[0], dtype=float))
    d = _n(np.asarray((await emb.ingest_embed([DOC]))[0], dtype=float))
    c0 = float(q @ d)
    d_perp = _n(d - c0 * q)
    print(f"c0 = cos(q,d) = {c0:.3f}\n")
    print(f"  {'t':>6} {'maxcos_C':>9} {'#>0.6':>6} {'#>0.7':>6}   nearest corpus text")
    for t in [c0, 0.7 * c0, 0.5 * c0, 0.3 * c0, 0.0, -0.15]:
        tc = max(min(t, 1.0), -1.0)
        dp = tc * q + math.sqrt(max(1 - tc * tc, 0.0)) * d_perp
        sims_c = c_mat @ dp
        order = np.argsort(-sims_c)
        top = order[0]
        print(
            f"  {tc:>6.2f} {sims_c[top]:>9.3f} {int(np.sum(sims_c > 0.6)):>6} "
            f"{int(np.sum(sims_c > 0.7)):>6}   {sims_c[top]:.2f} :: {texts[top][:60]!r}"
        )


if __name__ == "__main__":
    asyncio.run(main())
