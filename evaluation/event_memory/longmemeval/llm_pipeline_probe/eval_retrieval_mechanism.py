"""Surgical retrieval-mechanism evaluation for the v4 deriver.

For each (segment, query) test pair across 5 hypothesized "shapes that hurt
verbatim embedding" plus 2 controls:
  1. Embed segment and query with text-embedding-3-small.
  2. Run the v4 deriver to produce derivatives.
  3. Embed each derivative.
  4. Compute sim_verbatim = cos(query, segment) and sim_best_deriv =
     max_d cos(query, derivative_d). Delta = best_deriv - verbatim.
  5. Place the target segment in a pool with 50 random distractor segments
     from LongMemEval (lme-s-200.sqlite). Compute the rank of the target:
        rank_A: each segment represented by verbatim text only.
        rank_B: target carries verbatim + derivatives; the segment-level
                score is max sim across its embeddings. Distractors stay
                verbatim-only (we are not deriver-ing the whole pool —
                the question is whether deriving the TARGET helps).

Run:
    uv run python eval_retrieval_mechanism.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sqlite3
import textwrap
from typing import Any

import numpy as np
import openai
from dotenv import load_dotenv
from probe_deriver_v4_anti_fragment import DERIVATIVES_SCHEMA, PROMPT_DERIVER

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
DERIVER_MODEL = "gpt-5.4-nano"
DERIVER_REASONING = "medium"

DISTRACTOR_DB = "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval/lme-s-200.sqlite"
N_DISTRACTORS = 50
SEED = 42


# --------------------------------------------------------------------------
# TEST CASES
# --------------------------------------------------------------------------

# (case_id, shape, segment, query)
CASES: list[tuple[str, str, str, str]] = [
    # ----- TABLES -----
    (
        "T1",
        "TABLE",
        textwrap.dedent("""\
            | Model | Task A | Task B |
            | --- | --- | --- |
            | GPT-4 | 0.85 | 0.78 |
            | Claude | 0.91 | 0.82 |
            | Gemini | 0.79 | 0.81 |
        """).strip(),
        "what was Claude's score on Task B",
    ),
    (
        "T2",
        "TABLE",
        textwrap.dedent("""\
            | Bank | Loan amount | Rate | Monthly interest |
            | --- | --- | --- | --- |
            | Local Credit Union | $24,000 | 4.2% | $84.00 |
            | First National | $24,000 | 5.1% | $102.00 |
            | Online Lender | $24,000 | 6.8% | $136.00 |
        """).strip(),
        "what's the monthly interest charge for the local credit union",
    ),
    # ----- CODE -----
    (
        "C1",
        "CODE",
        textwrap.dedent("""\
            def find_max(node):
                if node is None:
                    return float('-inf')
                return max(node.value, find_max(node.left), find_max(node.right))
        """).strip(),
        "find the maximum value in a binary tree",
    ),
    (
        "C2",
        "CODE",
        "SELECT user_id, COUNT(*) AS n FROM orders WHERE created_at >= '2024-01-01' GROUP BY user_id HAVING COUNT(*) > 5;",
        "how many orders each user placed since the start of 2024, only users with more than five",
    ),
    # ----- ENCODED -----
    (
        "E1",
        "ENCODED",
        "Khoor, zruog! Wklv lv d phvvdjh.",
        "the message that says hello world",
    ),
    (
        "E2",
        "ENCODED",
        "VGhlIGxhdW5jaCBpcyBzY2hlZHVsZWQgZm9yIEp1bHkgMTQgYXQgQ2FwZSBDYW5hdmVyYWwu",
        "when and where is the rocket launch happening",
    ),
    # ----- ABBREVIATIONS -----
    (
        "A1",
        "ABBREV",
        "TIL JFK was POTUS during the CMC in '62; SAC went to DEFCON 2.",
        "John F. Kennedy was president during the Cuban Missile Crisis in 1962",
    ),
    (
        "A2",
        "ABBREV",
        "[2026-04-12T03:14:22Z] ERROR auth.service: token validation failed for user_id=42 reason=expired_jwt",
        "an authentication error caused by an expired JSON web token",
    ),
    # ----- OVERLOADED PROSE -----
    (
        "O1",
        "OVERLOADED",
        "Last March I went to Tokyo with my wife Anne, stayed at the Park Hyatt for 5 nights at $400/night, and had ramen at Ichiran in Shibuya.",
        "where did I stay in Tokyo",
    ),
    (
        "O2",
        "OVERLOADED",
        "This week was packed: my sister Priya gave birth to a baby girl named Aisha at Mount Sinai, the renovation crew finished tiling our master bathroom for $14,200, and I got the offer letter from Stripe — Senior Staff Engineer, base $310k, signing bonus $80k.",
        "what was the signing bonus on my Stripe offer",
    ),
    # ----- CONTROLS -----
    (
        "F1",
        "CONTROL",
        "I went to Tokyo last March with my wife Anne.",
        "who did I go to Tokyo with",
    ),
    (
        "F2",
        "CONTROL",
        "My favorite Spotify playlist is called Summer Vibes.",
        "what's my Spotify playlist called",
    ),
]


# --------------------------------------------------------------------------
# DERIVER
# --------------------------------------------------------------------------


async def derive_medium(client: openai.AsyncOpenAI, segment: str) -> list[str]:
    resp = await client.responses.create(
        model=DERIVER_MODEL,
        input=PROMPT_DERIVER.format(segment=segment),
        reasoning={"effort": DERIVER_REASONING},
        text={
            "format": {
                "type": "json_schema",
                "name": "derivatives",
                "schema": DERIVATIVES_SCHEMA,
                "strict": True,
            }
        },
    )
    payload = json.loads(resp.output_text)
    return list(payload.get("derivatives", []))


# --------------------------------------------------------------------------
# EMBEDDING
# --------------------------------------------------------------------------


async def embed_batch(client: openai.AsyncOpenAI, texts: list[str]) -> np.ndarray:
    """Return shape (len(texts), EMBED_DIM)."""
    out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
    # Avoid empty strings (API rejects).
    safe = [t or " " for t in texts]
    BATCH = 64
    for start in range(0, len(safe), BATCH):
        chunk = safe[start : start + BATCH]
        resp = await client.embeddings.create(
            model=EMBED_MODEL,
            input=chunk,
            dimensions=EMBED_DIM,
        )
        for i, item in enumerate(resp.data):
            out[start + i] = item.embedding
    arr = np.array(out, dtype=np.float32)
    # text-embedding-3-small returns L2-normalized vectors already, but
    # normalize defensively.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# --------------------------------------------------------------------------
# DISTRACTOR POOL
# --------------------------------------------------------------------------


def sample_distractors(n: int, seed: int) -> list[str]:
    conn = sqlite3.connect(DISTRACTOR_DB)
    cur = conn.cursor()
    cur.execute("SELECT block FROM segment_store_sg")
    texts: list[str] = []
    seen: set[str] = set()
    for (blob,) in cur.fetchall():
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        t = obj.get("text", "")
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not (100 <= len(t) <= 500):
            continue
        if t in seen:
            continue
        seen.add(t)
        texts.append(t)
    conn.close()
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[:n]


# --------------------------------------------------------------------------
# RANK COMPUTATION
# --------------------------------------------------------------------------


def rank_of_target(
    query_emb: np.ndarray,
    target_score: float,
    distractor_embs: np.ndarray,
) -> int:
    """1-based rank of target among (target + distractors).

    distractor_embs: shape (N, D). Each row is one distractor's verbatim emb.
    target_score: precomputed cos(query, target).
    """
    distractor_scores = distractor_embs @ query_emb
    higher = int(np.sum(distractor_scores > target_score))
    return higher + 1


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------


async def main() -> None:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("# eval_retrieval_mechanism — surgical deriver test")
    print(
        f"# embed={EMBED_MODEL} ({EMBED_DIM}d)  deriver={DERIVER_MODEL} reasoning={DERIVER_REASONING}"
    )
    print(f"# n_cases={len(CASES)}  n_distractors={N_DISTRACTORS}  seed={SEED}")

    # 1. Sample distractor pool (shared across all cases).
    distractors = sample_distractors(N_DISTRACTORS, SEED)
    print(f"# distractor pool size: {len(distractors)}")
    if len(distractors) < N_DISTRACTORS:
        print("# WARNING: not enough distractors; continuing with what we have")

    distractor_embs = await embed_batch(client, distractors)

    # 2. For each case: embed segment + query, derive, embed derivs.
    sem = asyncio.Semaphore(8)

    async def derive_for(case: tuple[str, str, str, str]) -> tuple[str, list[str]]:
        case_id, _, segment, _ = case
        async with sem:
            derivs = await derive_medium(client, segment)
        return case_id, derivs

    derive_results = await asyncio.gather(*(derive_for(c) for c in CASES))
    derivs_by_id: dict[str, list[str]] = dict(derive_results)

    # 3. Embed all queries + segments + derivatives in batched calls.
    queries = [q for _, _, _, q in CASES]
    segments = [s for _, _, s, _ in CASES]
    query_embs = await embed_batch(client, queries)
    segment_embs = await embed_batch(client, segments)

    # Embed derivs per case (variable count).
    deriv_embs_by_id: dict[str, np.ndarray] = {}
    flat_derivs: list[str] = []
    flat_owner: list[str] = []
    for case_id, _, _, _ in CASES:
        for d in derivs_by_id[case_id]:
            flat_derivs.append(d)
            flat_owner.append(case_id)
    if flat_derivs:
        flat_embs = await embed_batch(client, flat_derivs)
    else:
        flat_embs = np.zeros((0, EMBED_DIM), dtype=np.float32)

    for i, owner in enumerate(flat_owner):
        deriv_embs_by_id.setdefault(owner, []).append(flat_embs[i])  # type: ignore[arg-type]
    for k in list(deriv_embs_by_id.keys()):
        deriv_embs_by_id[k] = np.array(deriv_embs_by_id[k], dtype=np.float32)  # type: ignore[arg-type]
    for case_id, _, _, _ in CASES:
        if case_id not in deriv_embs_by_id:
            deriv_embs_by_id[case_id] = np.zeros((0, EMBED_DIM), dtype=np.float32)

    # 4. Per-case computation.
    rows: list[dict[str, Any]] = []
    for i, (case_id, shape, segment, query) in enumerate(CASES):
        q_emb = query_embs[i]
        s_emb = segment_embs[i]
        d_embs = deriv_embs_by_id[case_id]
        derivs = derivs_by_id[case_id]

        sim_verbatim = cos(q_emb, s_emb)
        if d_embs.shape[0] > 0:
            deriv_sims = (d_embs @ q_emb).tolist()
            best_idx = int(np.argmax(deriv_sims))
            sim_best_deriv = float(deriv_sims[best_idx])
            best_deriv = derivs[best_idx]
        else:
            deriv_sims = []
            sim_best_deriv = float("-inf")
            best_deriv = None

        delta = sim_best_deriv - sim_verbatim if d_embs.shape[0] > 0 else 0.0

        # Rank A: verbatim only.
        rank_a = rank_of_target(q_emb, sim_verbatim, distractor_embs)

        # Rank B: target = max(verbatim_sim, best_deriv_sim).
        target_score_b = (
            max(sim_verbatim, sim_best_deriv) if d_embs.shape[0] > 0 else sim_verbatim
        )
        rank_b = rank_of_target(q_emb, target_score_b, distractor_embs)

        if rank_b < rank_a or delta > 0.005:
            verdict = "WIN"
        elif rank_b > rank_a or delta < -0.005:
            verdict = "LOSS"
        else:
            verdict = "TIE"

        rows.append(
            {
                "case_id": case_id,
                "shape": shape,
                "segment": segment,
                "query": query,
                "n_derivatives": len(derivs),
                "derivatives": derivs,
                "sim_verbatim": round(sim_verbatim, 4),
                "sim_best_deriv": round(sim_best_deriv, 4)
                if sim_best_deriv != float("-inf")
                else None,
                "best_deriv": best_deriv,
                "deriv_sims": [round(x, 4) for x in deriv_sims],
                "delta_sim": round(delta, 4),
                "rank_A_verbatim": rank_a,
                "rank_B_with_derivs": rank_b,
                "verdict": verdict,
            }
        )

    # 5. Aggregate.
    n_win = sum(1 for r in rows if r["verdict"] == "WIN")
    n_tie = sum(1 for r in rows if r["verdict"] == "TIE")
    n_loss = sum(1 for r in rows if r["verdict"] == "LOSS")
    by_shape: dict[str, list[float]] = {}
    for r in rows:
        by_shape.setdefault(r["shape"], []).append(r["delta_sim"])
    mean_delta_by_shape = {k: round(sum(v) / len(v), 4) for k, v in by_shape.items()}

    summary = {
        "embed_model": EMBED_MODEL,
        "deriver_model": DERIVER_MODEL,
        "deriver_reasoning": DERIVER_REASONING,
        "n_distractors": len(distractors),
        "seed": SEED,
        "n_cases": len(CASES),
        "n_win": n_win,
        "n_tie": n_tie,
        "n_loss": n_loss,
        "mean_delta_by_shape": mean_delta_by_shape,
    }

    out = {
        "summary": summary,
        "cases": rows,
        "distractor_pool": distractors,
    }

    out_path = "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval/llm_pipeline_probe/eval_retrieval_mechanism.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # 6. Pretty-print.
    print()
    print("## SUMMARY")
    print(json.dumps(summary, indent=2))
    print()
    print("## PER-CASE")
    print(
        f"{'id':4} {'shape':10} {'sim_v':>7} {'sim_d':>7} {'delta':>7} {'rA':>3} {'rB':>3} {'n':>3} verdict  query"
    )
    for r in rows:
        sd = r["sim_best_deriv"]
        sd_s = f"{sd:.4f}" if sd is not None else "  N/A "
        print(
            f"{r['case_id']:4} {r['shape']:10} "
            f"{r['sim_verbatim']:7.4f} {sd_s:>7} "
            f"{r['delta_sim']:+7.4f} "
            f"{r['rank_A_verbatim']:3d} {r['rank_B_with_derivs']:3d} "
            f"{r['n_derivatives']:3d} "
            f"{r['verdict']:7} {r['query'][:60]}"
        )

    print()
    print("## DERIVATIVES (per case)")
    for r in rows:
        print(f"\n--- {r['case_id']} ({r['shape']}) {r['verdict']} ---")
        print(f"  segment: {r['segment'][:200]!r}")
        print(f"  query:   {r['query']!r}")
        print(f"  best_deriv: {r['best_deriv']!r}  (sim={r['sim_best_deriv']})")
        for j, d in enumerate(r["derivatives"]):
            sim = r["deriv_sims"][j] if j < len(r["deriv_sims"]) else None
            print(f"    [{j}] sim={sim} :: {d!r}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
