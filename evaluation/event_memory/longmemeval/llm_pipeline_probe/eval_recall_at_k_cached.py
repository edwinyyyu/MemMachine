"""Cached recall@K eval of v5 deriver on LongMemEval.

Saves state per-question (segments / segment embs / derivatives / deriv embs /
query emb) and only writes a `done_<qid>.flag` when EVERYTHING is on disk.
Re-running picks up where it left off.

Run:
    uv run python eval_recall_at_k_cached.py --num-questions 12 --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import openai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")

sys.path.insert(
    0,
    "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval",
)
from longmemeval_models import LongMemEvalItem, load_longmemeval_dataset  # noqa: E402

sys.path.insert(
    0,
    "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval/llm_pipeline_probe",
)
from probe_deriver_v5_anchor_propagation import derive as derive_v5  # noqa: E402

DATA_PATH = (
    "/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/data/longmemeval_s_cleaned.json"
)
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
DERIVER_MODEL = "gpt-5.4-nano"
DERIVER_REASONING = "medium"
MAX_CHUNK_LENGTH = 500

CACHE_DIR = Path(__file__).parent / "eval_recall_cache"


# ---------------------------------------------------------------------------
# SEGMENTATION
# ---------------------------------------------------------------------------


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_LENGTH,
        chunk_overlap=0,
        separators=[
            "\n\n",
            "],\n",
            "},\n",
            "),\n",
            "]\n",
            "}\n",
            ")\n",
            ",\n",
            "？\n",
            "?\n",
            "！\n",
            "!\n",
            "。\n",
            ".\n",
            "？",
            "? ",
            "！",
            "! ",
            "。",
            ". ",
            "; ",
            ": ",
            "—",
            "--",
            "，",
            "、",
            ", ",
            "​",
            " ",
            "",
        ],
        keep_separator="end",
    )


def build_segments(q: LongMemEvalItem, splitter) -> list[dict]:
    out = []
    for sid in q.haystack_session_ids:
        for turn in q.get_session(sid):
            text = turn.content.strip()
            if not text:
                continue
            for offset, chunk in enumerate(splitter.split_text(text)):
                out.append(
                    {
                        "segment_id": f"{sid}:{turn.index}:{offset}",
                        "text": chunk,
                        "session_id": sid,
                        "turn_index": turn.index,
                        "role": turn.role,
                        "has_answer": turn.has_answer,
                    }
                )
    return out


# ---------------------------------------------------------------------------
# EMBEDDING
# ---------------------------------------------------------------------------


_SPECIAL_PAT = re.compile(
    r"<\|endoftext\|>"
    r"|<\|im_start\|>"
    r"|<\|im_end\|>"
    r"|<\|fim_prefix\|>"
    r"|<\|fim_middle\|>"
    r"|<\|fim_suffix\|>"
    r"|<\|endofprompt\|>"
)


async def embed_batch(
    client: openai.AsyncOpenAI,
    texts: list[str],
) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIMS), dtype=np.float32)
    texts = [_SPECIAL_PAT.sub("", t) or "." for t in texts]
    texts = [t[:30000] if len(t) > 30000 else t for t in texts]

    batches: list[list[int]] = []
    cur: list[int] = []
    cur_chars = 0
    for i, t in enumerate(texts):
        if cur and (len(cur) >= 2048 or cur_chars + len(t) > 70000):
            batches.append(cur)
            cur = []
            cur_chars = 0
        cur.append(i)
        cur_chars += len(t)
    if cur:
        batches.append(cur)

    async def _one_batch(idxs):
        for attempt in range(5):
            try:
                resp = await client.embeddings.create(
                    input=[texts[i] for i in idxs],
                    model=EMBED_MODEL,
                    dimensions=EMBED_DIMS,
                )
                return idxs, [d.embedding for d in resp.data]
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ):
                if attempt == 4:
                    raise
                await asyncio.sleep(2**attempt)
        return None

    results = await asyncio.gather(*[_one_batch(b) for b in batches])
    out = np.zeros((len(texts), EMBED_DIMS), dtype=np.float32)
    for idxs, embs in results:
        for i, e in zip(idxs, embs, strict=False):
            out[i] = np.asarray(e, dtype=np.float32)
    return out


def normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---------------------------------------------------------------------------
# DERIVE
# ---------------------------------------------------------------------------


async def derive_all(
    client: openai.AsyncOpenAI,
    texts: list[str],
    sem: asyncio.Semaphore,
) -> list[list[str]]:
    async def _one(t):
        async with sem:
            for attempt in range(5):
                try:
                    return await derive_v5(
                        client,
                        t,
                        model=DERIVER_MODEL,
                        reasoning=DERIVER_REASONING,
                    )
                except Exception as e:
                    if attempt == 4:
                        print(f"  [warn] deriver fail: {e!s:.120}", flush=True)
                        return []
                    await asyncio.sleep(2**attempt)
        return None

    return await asyncio.gather(*[_one(t) for t in texts])


# ---------------------------------------------------------------------------
# CACHE PATHS
# ---------------------------------------------------------------------------


def cache_paths(qid: str) -> dict[str, Path]:
    return {
        "segs": CACHE_DIR / f"segments_{qid}.json",
        "seg_embs": CACHE_DIR / f"embeds_seg_{qid}.npy",
        "derivs": CACHE_DIR / f"derivs_{qid}.json",
        "deriv_embs": CACHE_DIR / f"embeds_deriv_{qid}.npy",
        "qemb": CACHE_DIR / f"qembed_{qid}.npy",
        "flag": CACHE_DIR / f"done_{qid}.flag",
    }


# ---------------------------------------------------------------------------
# INGEST ONE QUESTION
# ---------------------------------------------------------------------------


async def ingest_question(
    client: openai.AsyncOpenAI,
    q: LongMemEvalItem,
    splitter,
    deriver_sem: asyncio.Semaphore,
) -> tuple[bool, str]:
    paths = cache_paths(q.question_id)
    if paths["flag"].exists():
        return True, "cached"

    segments = build_segments(q, splitter)
    if not segments:
        return False, "no_segments"
    if not any(s["has_answer"] for s in segments):
        return False, "no_gold"

    seg_texts = [s["text"] for s in segments]

    # Run all three steps concurrently:
    # 1) embed query
    # 2) embed segments verbatim
    # 3) derive on segments (then embed derivatives)
    qemb_task = embed_batch(client, [q.question])
    segemb_task = embed_batch(client, seg_texts)
    derivs_task = derive_all(client, seg_texts, deriver_sem)

    qemb, seg_embs, derivs_per_seg = await asyncio.gather(
        qemb_task, segemb_task, derivs_task
    )
    qemb = normalize(qemb)[0]
    seg_embs = normalize(seg_embs)

    # Flatten derivatives.
    flat_deriv_texts: list[str] = []
    flat_deriv_segids: list[str] = []
    for s, derivs in zip(segments, derivs_per_seg, strict=False):
        for d in derivs:
            d_clean = (d or "").strip()
            if not d_clean:
                continue
            flat_deriv_texts.append(d_clean)
            flat_deriv_segids.append(s["segment_id"])

    if flat_deriv_texts:
        deriv_embs = await embed_batch(client, flat_deriv_texts)
        deriv_embs = normalize(deriv_embs)
    else:
        deriv_embs = np.zeros((0, EMBED_DIMS), dtype=np.float32)

    # Persist (atomic-ish: write data files first, then flag).
    with paths["segs"].open("w") as f:
        json.dump(
            [
                {
                    "segment_id": s["segment_id"],
                    "text": s["text"],
                    "session_id": s["session_id"],
                    "turn_index": s["turn_index"],
                    "role": s["role"],
                    "has_answer": s["has_answer"],
                }
                for s in segments
            ],
            f,
        )
    np.save(paths["seg_embs"], seg_embs)

    with paths["derivs"].open("w") as f:
        json.dump(
            [
                {"segment_id": sid, "deriv_text": txt}
                for sid, txt in zip(flat_deriv_segids, flat_deriv_texts, strict=False)
            ],
            f,
        )
    np.save(paths["deriv_embs"], deriv_embs)
    np.save(paths["qemb"], qemb.astype(np.float32))

    # Flag last.
    paths["flag"].write_text(
        json.dumps(
            {
                "question_id": q.question_id,
                "n_segments": len(segments),
                "n_derivatives": len(flat_deriv_texts),
                "question_type": str(q.question_type),
                "ts": time.time(),
            }
        )
    )
    return True, "fresh"


# ---------------------------------------------------------------------------
# RETRIEVE
# ---------------------------------------------------------------------------


def retrieve_topk(
    qemb: np.ndarray,
    cand_embs: np.ndarray,
    cand_segment_ids: list[str],
    k_max: int = 10,
) -> list[tuple[str, float, int]]:
    """Return list of (segment_id, best_sim, best_idx) sorted desc, top k_max."""
    if cand_embs.size == 0:
        return []
    sims = cand_embs @ qemb
    best: dict[str, tuple[float, int]] = {}
    for i, sid in enumerate(cand_segment_ids):
        s = float(sims[i])
        if sid not in best or s > best[sid][0]:
            best[sid] = (s, i)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    return [(sid, sc, idx) for sid, (sc, idx) in ranked[:k_max]]


# ---------------------------------------------------------------------------
# AGGREGATE FROM CACHE
# ---------------------------------------------------------------------------


def evaluate_from_cache(qid: str, q: LongMemEvalItem) -> dict | None:
    paths = cache_paths(qid)
    if not paths["flag"].exists():
        return None

    with paths["segs"].open() as f:
        segments = json.load(f)
    seg_embs = np.load(paths["seg_embs"])
    with paths["derivs"].open() as f:
        derivs = json.load(f)
    deriv_embs = (
        np.load(paths["deriv_embs"])
        if paths["deriv_embs"].exists()
        else np.zeros((0, EMBED_DIMS), dtype=np.float32)
    )
    qemb = np.load(paths["qemb"])

    seg_ids = [s["segment_id"] for s in segments]
    has_answer_set = {s["segment_id"] for s in segments if s["has_answer"]}

    flat_deriv_texts = [d["deriv_text"] for d in derivs]
    flat_deriv_segids = [d["segment_id"] for d in derivs]

    # A: only segments
    topA = retrieve_topk(qemb, seg_embs, seg_ids, k_max=10)
    # B: segments + derivatives
    if deriv_embs.size:
        cand_embs_B = np.vstack([seg_embs, deriv_embs])
        cand_ids_B = seg_ids + flat_deriv_segids
    else:
        cand_embs_B = seg_embs
        cand_ids_B = seg_ids
    topB = retrieve_topk(qemb, cand_embs_B, cand_ids_B, k_max=10)

    def hit(top: list[tuple[str, float, int]], k: int) -> int:
        return int(any(sid in has_answer_set for sid, _, _ in top[:k]))

    rA = {k: hit(topA, k) for k in (1, 5, 10)}
    rB = {k: hit(topB, k) for k in (1, 5, 10)}

    # Diagnostic info
    {s["segment_id"]: s for s in segments}
    seg_text_by_idx = [s["text"] for s in segments]

    # B's top10 sources (was segment match or derivative match?)
    topB_sources = []
    n_seg = len(seg_embs)
    for sid, sc, idx in topB[:10]:
        is_deriv = idx >= n_seg
        src_text = flat_deriv_texts[idx - n_seg] if is_deriv else seg_text_by_idx[idx]
        topB_sources.append(
            {
                "segment_id": sid,
                "from_derivative": bool(is_deriv),
                "match_text": src_text,
                "is_gold": sid in has_answer_set,
                "sim": sc,
            }
        )

    gold_diag = []
    for s in segments:
        if not s["has_answer"]:
            continue
        sid = s["segment_id"]
        rank_A = next((i + 1 for i, (x, _, _) in enumerate(topA) if x == sid), None)
        rank_B = next((i + 1 for i, (x, _, _) in enumerate(topB) if x == sid), None)
        gold_diag.append(
            {
                "segment_id": sid,
                "role": s["role"],
                "text": s["text"],
                "rank_A": rank_A,
                "rank_B": rank_B,
                "derivatives": [
                    t
                    for t, ds in zip(flat_deriv_texts, flat_deriv_segids, strict=False)
                    if ds == sid
                ],
            }
        )

    return {
        "question_id": qid,
        "question_type": str(q.question_type),
        "question": q.question,
        "answer": q.answer,
        "n_segments": len(segments),
        "n_gold_segments": len(has_answer_set),
        "n_derivatives": len(flat_deriv_texts),
        "rA_at": rA,
        "rB_at": rB,
        "topB_sources": topB_sources,
        "gold_segments": gold_diag,
    }


# ---------------------------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------------------------


def stratified_sample(
    questions: list[LongMemEvalItem], n: int, seed: int = 42
) -> list[LongMemEvalItem]:
    rng = random.Random(seed)
    by_type: dict[str, list[LongMemEvalItem]] = defaultdict(list)
    for q in questions:
        by_type[str(q.question_type)].append(q)
    types = sorted(by_type.keys())
    pools = {t: rng.sample(by_type[t], len(by_type[t])) for t in types}
    out: list[LongMemEvalItem] = []
    while len(out) < n:
        progressed = False
        for t in types:
            if not pools[t]:
                continue
            out.append(pools[t].pop())
            progressed = True
            if len(out) >= n:
                break
        if not progressed:
            break
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--per-q-timeout", type=float, default=600.0)
    parser.add_argument("--output", default="eval_recall_at_k_cached_results.json")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_longmemeval_dataset(DATA_PATH)
    sample = stratified_sample(questions, args.num_questions)

    print(f"Sampled {len(sample)} questions:", flush=True)
    type_counts: dict[str, int] = defaultdict(int)
    for q in sample:
        type_counts[str(q.question_type)] += 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}", flush=True)

    # Pre-check cache
    pre_done = sum(1 for q in sample if cache_paths(q.question_id)["flag"].exists())
    print(f"Cache: {pre_done}/{len(sample)} already complete\n", flush=True)

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=180)
    splitter = make_splitter()
    deriver_sem = asyncio.Semaphore(args.concurrency)

    t_start = time.time()
    print(f"Running at {time.strftime('%H:%M:%S')}", flush=True)

    # Sequential per-question (so a SIGTERM only loses the in-flight one).
    for i, q in enumerate(sample):
        t0 = time.time()
        try:
            ok, status = await asyncio.wait_for(
                ingest_question(client, q, splitter, deriver_sem),
                timeout=args.per_q_timeout,
            )
        except asyncio.TimeoutError:
            print(
                f"  [{i + 1}/{len(sample)} {q.question_id}] TIMEOUT after {args.per_q_timeout:.0f}s",
                flush=True,
            )
            continue
        except Exception as e:
            print(
                f"  [{i + 1}/{len(sample)} {q.question_id}] ERROR: {e!s:.200}",
                flush=True,
            )
            continue
        elapsed = time.time() - t0
        if not ok:
            print(
                f"  [{i + 1}/{len(sample)} {q.question_id}] SKIP: {status}", flush=True
            )
            continue
        print(
            f"  [{i + 1}/{len(sample)} {q.question_id} {q.question_type!s:42s}] {status} ({elapsed:.0f}s)",
            flush=True,
        )

    await client.close()

    # Now scan cache and aggregate.
    print(f"\nIngest done in {time.time() - t_start:.0f}s; aggregating...", flush=True)
    results: list[dict] = []
    for q in sample:
        r = evaluate_from_cache(q.question_id, q)
        if r is not None:
            results.append(r)

    print(f"\n{len(results)}/{len(sample)} questions completed in cache.\n", flush=True)

    if not results:
        print("No completed questions; nothing to report.")
        return

    # Aggregate
    per_type: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[str, list[int]] = defaultdict(list)
    n_segs_total = 0
    n_derivs_total = 0
    for r in results:
        n_segs_total += r["n_segments"]
        n_derivs_total += r["n_derivatives"]
        for k in (1, 5, 10):
            overall[f"A@{k}"].append(r["rA_at"][k])
            overall[f"B@{k}"].append(r["rB_at"][k])
            per_type[r["question_type"]][f"A@{k}"].append(r["rA_at"][k])
            per_type[r["question_type"]][f"B@{k}"].append(r["rB_at"][k])

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary: dict[str, Any] = {
        "n_questions": len(results),
        "n_segments_total": n_segs_total,
        "n_derivatives_total": n_derivs_total,
        "ratio_deriv_to_seg": n_derivs_total / max(1, n_segs_total),
        "overall": {k: mean(v) for k, v in overall.items()},
        "delta": {
            f"@{k}": mean(overall[f"B@{k}"]) - mean(overall[f"A@{k}"])
            for k in (1, 5, 10)
        },
        "per_type": {
            t: {
                "n": len(v["A@1"]),
                "A@1": mean(v["A@1"]),
                "B@1": mean(v["B@1"]),
                "delta@1": mean(v["B@1"]) - mean(v["A@1"]),
                "A@5": mean(v["A@5"]),
                "B@5": mean(v["B@5"]),
                "delta@5": mean(v["B@5"]) - mean(v["A@5"]),
                "A@10": mean(v["A@10"]),
                "B@10": mean(v["B@10"]),
                "delta@10": mean(v["B@10"]) - mean(v["A@10"]),
            }
            for t, v in per_type.items()
        },
    }

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    out_path = Path(__file__).parent / args.output
    with out_path.open("w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, default=str)

    # Per-question table
    print("\n=== PER-QUESTION ===")
    print(f"{'qid':<40s} {'type':<35s} R@1   R@5   R@10")
    for r in results:
        a1, a5, a10 = r["rA_at"][1], r["rA_at"][5], r["rA_at"][10]
        b1, b5, b10 = r["rB_at"][1], r["rB_at"][5], r["rB_at"][10]
        print(
            f"{r['question_id']:<40s} {r['question_type']:<35s} "
            f"{a1}/{b1}  {a5}/{b5}  {a10}/{b10}"
        )

    # Wins/losses (B vs A on R@5)
    wins, losses = [], []
    for r in results:
        d5 = r["rB_at"][5] - r["rA_at"][5]
        if d5 > 0:
            wins.append(r)
        elif d5 < 0:
            losses.append(r)

    def fmt(r):
        gold_in_b_only = [
            g
            for g in r["gold_segments"]
            if g["rank_B"] is not None and g["rank_A"] is None
        ]
        gold_in_a_only = [
            g
            for g in r["gold_segments"]
            if g["rank_A"] is not None and g["rank_B"] is None
        ]
        focus = gold_in_b_only or gold_in_a_only or r["gold_segments"]
        out = [
            f"qid={r['question_id']} type={r['question_type']}",
            f"  Q: {r['question']}",
            f"  A: {r['answer']}",
            f"  R@5 A={r['rA_at'][5]} B={r['rB_at'][5]} (delta={r['rB_at'][5] - r['rA_at'][5]:+d})",
        ]
        for g in focus[:1]:
            txt = g["text"][:200].replace("\n", " ")
            out.append(
                f"  GOLD ({g['role']}, rank_A={g['rank_A']}, rank_B={g['rank_B']}): {txt}"
            )
            for d in g["derivatives"][:6]:
                out.append(f"    deriv: {d[:160]}")
        return "\n".join(out)

    print("\n=== UP-TO-3 WINS (B catches gold A missed at R@5) ===")
    for w in wins[:3]:
        print(fmt(w), "\n")
    print("\n=== UP-TO-3 LOSSES (B drops below A at R@5) ===")
    for loss in losses[:3]:
        print(fmt(loss), "\n")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
