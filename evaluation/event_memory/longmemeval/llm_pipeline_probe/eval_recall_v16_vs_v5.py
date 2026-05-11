"""Recall@K: v16 deriver vs v5/v10 deriver on cached LongMemEval questions.

Reuses the existing eval_recall_cache (segments, segment embeddings, query
embedding, v5/v10 derivatives) and adds a v16 layer.

For each question, computes:
  A: verbatim only        (cached)
  B: + v5/v10 derivatives (cached)
  C: + v16 derivatives    (NEW)

Then reports R@1/5/10 averages and the C-A, C-B, B-A deltas.

Run:
    uv run python eval_recall_v16_vs_v5.py --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")

sys.path.insert(
    0,
    "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval",
)
from longmemeval_models import load_longmemeval_dataset  # noqa: E402

sys.path.insert(
    0,
    "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval/llm_pipeline_probe",
)
from probe_deriver_v16_binding import derive as derive_v16  # noqa: E402

DATA_PATH = (
    "/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/data/longmemeval_s_cleaned.json"
)
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
DERIVER_MODEL = "gpt-5.4-nano"
DERIVER_REASONING = "medium"

CACHE_DIR = Path(__file__).parent / "eval_recall_cache"
V16_CACHE_DIR = Path(__file__).parent / "eval_recall_cache_v16"
V16_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_paths(qid: str) -> dict[str, Path]:
    """Existing v5 cache paths (read-only here)."""
    return {
        "segs": CACHE_DIR / f"segments_{qid}.json",
        "seg_embs": CACHE_DIR / f"embeds_seg_{qid}.npy",
        "derivs": CACHE_DIR / f"derivs_{qid}.json",
        "deriv_embs": CACHE_DIR / f"embeds_deriv_{qid}.npy",
        "qemb": CACHE_DIR / f"qembed_{qid}.npy",
        "flag": CACHE_DIR / f"done_{qid}.flag",
    }


def v16_paths(qid: str) -> dict[str, Path]:
    return {
        "derivs": V16_CACHE_DIR / f"derivs_v16_{qid}.json",
        "deriv_embs": V16_CACHE_DIR / f"embeds_deriv_v16_{qid}.npy",
        "flag": V16_CACHE_DIR / f"done_v16_{qid}.flag",
    }


def normalize(mat: np.ndarray) -> np.ndarray:
    if mat.ndim == 1:
        n = np.linalg.norm(mat)
        return mat / max(n, 1e-12)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


async def embed_batch(client: openai.AsyncOpenAI, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIMS), dtype=np.float32)
    # API limit: max 2048 inputs per request.
    BATCH = 2000
    chunks = []
    for i in range(0, len(texts), BATCH):
        sub = texts[i : i + BATCH]
        resp = await client.embeddings.create(
            model=EMBED_MODEL, input=sub, dimensions=EMBED_DIMS
        )
        chunks.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
    return np.vstack(chunks) if chunks else np.zeros((0, EMBED_DIMS), dtype=np.float32)


async def derive_v16_one(client, text, sem):
    async with sem:
        for attempt in range(5):
            try:
                return await derive_v16(
                    client,
                    text,
                    model=DERIVER_MODEL,
                    reasoning=DERIVER_REASONING,
                )
            except Exception as e:
                if attempt == 4:
                    print(f"  [warn] v16 deriver fail: {e!s:.120}", flush=True)
                    return []
                await asyncio.sleep(2**attempt)
    return None


async def process_question(client, qid: str, deriver_sem):
    paths_v5 = cache_paths(qid)
    paths_v16 = v16_paths(qid)

    if not paths_v5["flag"].exists():
        return False, "no v5 cache"
    if paths_v16["flag"].exists():
        return True, "cached"

    # Load v5 segments — we need the same segments for v16 derivation.
    with paths_v5["segs"].open() as f:
        segments = json.load(f)
    seg_texts = [s["text"] for s in segments]

    if not seg_texts:
        return False, "no segments"

    # Generate v16 derivatives for each segment in parallel.
    derivs_per_seg = await asyncio.gather(
        *[derive_v16_one(client, t, deriver_sem) for t in seg_texts]
    )

    # Flatten + embed.
    flat_texts: list[str] = []
    flat_segids: list[str] = []
    for s, derivs in zip(segments, derivs_per_seg, strict=False):
        for d in derivs:
            d = (d or "").strip()
            if not d:
                continue
            flat_texts.append(d)
            flat_segids.append(s["segment_id"])

    if flat_texts:
        embs = await embed_batch(client, flat_texts)
        embs = normalize(embs)
    else:
        embs = np.zeros((0, EMBED_DIMS), dtype=np.float32)

    # Persist
    with paths_v16["derivs"].open("w") as f:
        json.dump(
            [
                {"segment_id": sid, "deriv_text": txt}
                for sid, txt in zip(flat_segids, flat_texts, strict=False)
            ],
            f,
        )
    np.save(paths_v16["deriv_embs"], embs)
    paths_v16["flag"].write_text(
        json.dumps(
            {
                "question_id": qid,
                "n_derivs": len(flat_texts),
                "ts": time.time(),
            }
        )
    )
    return True, "fresh"


def retrieve_topk(qemb, cand_embs, cand_segment_ids, k_max=10):
    if cand_embs.size == 0:
        return []
    sims = cand_embs @ qemb
    best: dict[str, tuple[float, int]] = {}
    for i, sid in enumerate(cand_segment_ids):
        s = float(sims[i])
        if sid not in best or s > best[sid][0]:
            best[sid] = (s, i)
    return sorted(
        [(sid, sc, idx) for sid, (sc, idx) in best.items()],
        key=lambda t: -t[1],
    )[:k_max]


def evaluate_qid(qid: str) -> dict | None:
    paths_v5 = cache_paths(qid)
    paths_v16 = v16_paths(qid)
    if not paths_v5["flag"].exists() or not paths_v16["flag"].exists():
        return None

    with paths_v5["segs"].open() as f:
        segments = json.load(f)
    seg_embs = np.load(paths_v5["seg_embs"])
    with paths_v5["derivs"].open() as f:
        derivs_v5 = json.load(f)
    deriv_embs_v5 = (
        np.load(paths_v5["deriv_embs"])
        if paths_v5["deriv_embs"].exists()
        else np.zeros((0, EMBED_DIMS), dtype=np.float32)
    )
    qemb = np.load(paths_v5["qemb"])

    with paths_v16["derivs"].open() as f:
        derivs_v16 = json.load(f)
    deriv_embs_v16 = (
        np.load(paths_v16["deriv_embs"])
        if paths_v16["deriv_embs"].exists()
        else np.zeros((0, EMBED_DIMS), dtype=np.float32)
    )

    seg_ids = [s["segment_id"] for s in segments]
    has_answer = {s["segment_id"] for s in segments if s["has_answer"]}

    # A: verbatim only
    topA = retrieve_topk(qemb, seg_embs, seg_ids, k_max=10)
    # B: + v5
    v5_segids = [d["segment_id"] for d in derivs_v5]
    cand_B = np.vstack([seg_embs, deriv_embs_v5]) if deriv_embs_v5.size else seg_embs
    cand_B_ids = seg_ids + v5_segids
    topB = retrieve_topk(qemb, cand_B, cand_B_ids, k_max=10)
    # C: + v16
    v16_segids = [d["segment_id"] for d in derivs_v16]
    cand_C = np.vstack([seg_embs, deriv_embs_v16]) if deriv_embs_v16.size else seg_embs
    cand_C_ids = seg_ids + v16_segids
    topC = retrieve_topk(qemb, cand_C, cand_C_ids, k_max=10)

    def hit(top, k):
        return int(any(sid in has_answer for sid, _, _ in top[:k]))

    return {
        "qid": qid,
        "n_segs": len(seg_embs),
        "n_derivs_v5": len(derivs_v5),
        "n_derivs_v16": len(derivs_v16),
        "rA": {k: hit(topA, k) for k in (1, 5, 10)},
        "rB": {k: hit(topB, k) for k in (1, 5, 10)},
        "rC": {k: hit(topC, k) for k in (1, 5, 10)},
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    # Find all already-cached qids from v5.
    flags = sorted(CACHE_DIR.glob("done_*.flag"))
    qids = []
    for fl in flags:
        name = fl.name
        if name.startswith("done_gpt4_"):
            continue  # skip alt-model cache
        if name.startswith("done_") and name.endswith(".flag"):
            qid = name[len("done_") : -len(".flag")]
            qids.append(qid)

    print(f"Found {len(qids)} cached v5 questions", flush=True)

    # Load the actual question records to attach question_type later.
    questions = load_longmemeval_dataset(DATA_PATH)
    q_by_id = {q.question_id: q for q in questions}

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=180)
    sem = asyncio.Semaphore(args.concurrency)

    # Generate v16 derivatives for any qid that doesn't have them yet.
    t_start = time.time()
    needs_gen = [q for q in qids if not v16_paths(q)["flag"].exists()]
    print(
        f"v16 fresh generation needed for {len(needs_gen)}/{len(qids)} questions",
        flush=True,
    )

    # Run multiple questions in parallel. Total deriver concurrency is sem
    # (shared); cross-question concurrency limited by question_sem so we don't
    # spike RAM holding huge derivative buffers in memory at once.
    question_sem = asyncio.Semaphore(6)

    async def run_one(i: int, qid: str):
        async with question_sem:
            t0 = time.time()
            try:
                ok, status = await process_question(client, qid, sem)
            except Exception as e:
                print(f"  [{i + 1}/{len(qids)} {qid[:8]}] EXC: {e!s:.140}", flush=True)
                return
            if not ok:
                print(f"  [{i + 1}/{len(qids)} {qid[:8]}] SKIP: {status}", flush=True)
                return
            if status == "fresh":
                print(
                    f"  [{i + 1}/{len(qids)} {qid[:8]}] v16 generated ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            else:
                print(f"  [{i + 1}/{len(qids)} {qid[:8]}] cached", flush=True)

    await asyncio.gather(*[run_one(i, qid) for i, qid in enumerate(qids)])

    await client.close()
    print(f"\nIngest done in {time.time() - t_start:.0f}s", flush=True)

    # Evaluate all
    results = []
    for qid in qids:
        r = evaluate_qid(qid)
        if r is None:
            continue
        q = q_by_id.get(qid)
        r["question_type"] = str(q.question_type) if q else "unknown"
        r["question"] = q.question if q else ""
        r["answer"] = q.answer if q else ""
        results.append(r)

    if not results:
        print("No results.")
        return

    # Aggregate
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print()
    print(f"# RESULTS (N={len(results)})")
    print()

    for tag, key in [("A (verbatim)", "rA"), ("B (v5/v10)", "rB"), ("C (v16)", "rC")]:
        r1 = mean([r[key][1] for r in results])
        r5 = mean([r[key][5] for r in results])
        r10 = mean([r[key][10] for r in results])
        print(f"  {tag:>16s}:  R@1={r1:.3f}  R@5={r5:.3f}  R@10={r10:.3f}")

    print()
    print("# DELTAS")
    for tag, k1, k2 in [
        ("B - A (v5 lift)", "rB", "rA"),
        ("C - A (v16 lift)", "rC", "rA"),
        ("C - B (v16 vs v5)", "rC", "rB"),
    ]:
        d1 = mean([r[k1][1] - r[k2][1] for r in results])
        d5 = mean([r[k1][5] - r[k2][5] for r in results])
        d10 = mean([r[k1][10] - r[k2][10] for r in results])
        print(f"  {tag:>20s}:  Δ@1={d1:+.3f}  Δ@5={d5:+.3f}  Δ@10={d10:+.3f}")

    print()
    print("# PER-TYPE Δ@5 (C-B = v16 vs v5)")
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r["rC"][5] - r["rB"][5])
    for t in sorted(by_type):
        deltas = by_type[t]
        print(f"  {t:>40s}: n={len(deltas):2d}  Δ@5={mean(deltas):+.3f}")

    # Cost: derivative ratio
    total_segs = sum(r["n_segs"] for r in results)
    total_v5 = sum(r["n_derivs_v5"] for r in results)
    total_v16 = sum(r["n_derivs_v16"] for r in results)
    print()
    print("# COST")
    print(f"  total segments: {total_segs}")
    print(
        f"  v5 derivatives: {total_v5}   (ratio {total_v5 / max(1, total_segs):.2f}/seg)"
    )
    print(
        f"  v16 derivatives: {total_v16}   (ratio {total_v16 / max(1, total_segs):.2f}/seg)"
    )

    # Save results
    out_path = Path(__file__).parent / "eval_v16_vs_v5_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
