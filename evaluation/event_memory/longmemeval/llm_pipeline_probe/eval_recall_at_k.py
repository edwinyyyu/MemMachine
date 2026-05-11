"""Quantitative validation of the v4 LLM deriver: recall@K with vs without derivatives.

Methodology
-----------
1. Sample N questions stratified by question_type (seed 42).
2. For each question, build two indices over its full session set:
     A: embed each turn-segment text directly (1 embedding per segment).
     B: same segments + run v4 deriver on each segment, embed each derivative.
        Each embedding maps back to its segment_id; dedup by segment at retrieve.
3. Embed the question, retrieve top-K via cosine. Mark recall@K hit if any
   returned segment came from a turn with `has_answer=True`.
4. Aggregate: per-type and overall mean R@1 / R@5 / R@10 with deltas.

Run:
    uv run python eval_recall_at_k.py --num-questions 30 --concurrency 64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
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
from probe_deriver_v4_anti_fragment import derive as derive_v4  # noqa: E402

DATA_PATH = (
    "/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/data/longmemeval_s_cleaned.json"
)
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
DERIVER_MODEL = "gpt-5.4-nano"
DERIVER_REASONING = "medium"
MAX_CHUNK_LENGTH = 500


# ---------------------------------------------------------------------------
# SEGMENTATION + DATA STRUCTURES
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
    """Return list of segment dicts with segment_id, text, has_answer, session_id, turn_index."""
    segments = []
    for sid in q.haystack_session_ids:
        for turn in q.get_session(sid):
            text = turn.content.strip()
            if not text:
                continue
            for offset, chunk in enumerate(splitter.split_text(text)):
                segments.append(
                    {
                        "segment_id": f"{sid}:{turn.index}:{offset}",
                        "text": chunk,
                        "session_id": sid,
                        "turn_index": turn.index,
                        "role": turn.role,
                        "has_answer": turn.has_answer,
                    }
                )
    return segments


# ---------------------------------------------------------------------------
# EMBEDDING (raw OpenAI client)
# ---------------------------------------------------------------------------


async def embed_batch(
    client: openai.AsyncOpenAI,
    texts: list[str],
    *,
    model: str = EMBED_MODEL,
    dimensions: int = EMBED_DIMS,
) -> np.ndarray:
    """Embed a list of texts. Batches into <=2048 inputs and <=75K codepoints per request."""
    if not texts:
        return np.zeros((0, dimensions), dtype=np.float32)

    # Replace special tokens that cause 500 errors (matches OpenAIEmbedder behavior)
    import re

    pat = re.compile(
        r"<\|endoftext\|>"
        r"|<\|im_start\|>"
        r"|<\|im_end\|>"
        r"|<\|fim_prefix\|>"
        r"|<\|fim_middle\|>"
        r"|<\|fim_suffix\|>"
        r"|<\|endofprompt\|>"
    )
    texts = [pat.sub("", t) or "." for t in texts]
    # Truncate any extreme-length single text to 30K chars for safety.
    texts = [t[:30000] if len(t) > 30000 else t for t in texts]

    # Batch by both input count (<=2048) and total chars (<=75K).
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

    # Fire batches concurrently.
    async def _one_batch(idxs):
        for attempt in range(5):
            try:
                resp = await client.embeddings.create(
                    input=[texts[i] for i in idxs],
                    model=model,
                    dimensions=dimensions,
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
    out = np.zeros((len(texts), dimensions), dtype=np.float32)
    for idxs, embs in results:
        for i, e in zip(idxs, embs, strict=False):
            out[i] = np.asarray(e, dtype=np.float32)
    return out


def normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---------------------------------------------------------------------------
# DERIVE BATCH
# ---------------------------------------------------------------------------


async def derive_all(
    client: openai.AsyncOpenAI,
    texts: list[str],
    sem: asyncio.Semaphore,
) -> list[list[str]]:
    """Run the v4 deriver on each text. Returns list-of-derivative-lists in input order."""

    async def _one(t):
        async with sem:
            for attempt in range(5):
                try:
                    return await derive_v4(
                        client,
                        t,
                        model=DERIVER_MODEL,
                        reasoning=DERIVER_REASONING,
                    )
                except Exception as e:
                    if attempt == 4:
                        # Fail soft: return empty derivative list for this segment.
                        # The verbatim segment is still embedded in B regardless.
                        print(f"  [warn] deriver failed: {e!s:.120}")
                        return []
                    await asyncio.sleep(2**attempt)
        return None

    return await asyncio.gather(*[_one(t) for t in texts])


# ---------------------------------------------------------------------------
# RETRIEVE
# ---------------------------------------------------------------------------


def retrieve_topk(
    query_emb: np.ndarray,
    cand_embs: np.ndarray,
    cand_segment_ids: list[str],
    k_max: int = 10,
) -> list[str]:
    """Cosine similarity, dedup by segment_id keeping max sim; return top-k_max segment_ids."""
    sims = cand_embs @ query_emb  # (N,)
    # Dedup by segment_id: for each segment keep max sim across all its embedding sources.
    best_per_seg: dict[str, float] = {}
    for sid, s in zip(cand_segment_ids, sims, strict=False):
        if sid not in best_per_seg or s > best_per_seg[sid]:
            best_per_seg[sid] = float(s)
    ranked = sorted(best_per_seg.items(), key=lambda kv: -kv[1])
    return [sid for sid, _ in ranked[:k_max]]


# ---------------------------------------------------------------------------
# PER-QUESTION EVAL
# ---------------------------------------------------------------------------


async def eval_question(
    client: openai.AsyncOpenAI,
    q: LongMemEvalItem,
    splitter,
    deriver_sem: asyncio.Semaphore,
    save_examples: bool = True,
) -> dict:
    segments = build_segments(q, splitter)
    if not segments:
        return {"question_id": q.question_id, "skipped": "no_segments"}

    # ---- Index A: embed verbatim segment text only.
    seg_texts = [s["text"] for s in segments]
    seg_ids = [s["segment_id"] for s in segments]
    has_answer_set = {s["segment_id"] for s in segments if s["has_answer"]}
    if not has_answer_set:
        return {"question_id": q.question_id, "skipped": "no_gold_segment"}

    # ---- Embed query.
    qemb = await embed_batch(client, [q.question])
    qemb = normalize(qemb)[0]

    # ---- Embed segments verbatim (used by both A and B).
    seg_embs = await embed_batch(client, seg_texts)
    seg_embs = normalize(seg_embs)

    # ---- Derive on each segment, embed derivatives.
    derivs_per_seg = await derive_all(client, seg_texts, deriver_sem)

    flat_deriv_texts: list[str] = []
    flat_deriv_segids: list[str] = []
    deriv_owner_per_text: list[str] = []  # for debugging
    for sid, derivs in zip(seg_ids, derivs_per_seg, strict=False):
        for d in derivs:
            d_clean = (d or "").strip()
            if not d_clean:
                continue
            flat_deriv_texts.append(d_clean)
            flat_deriv_segids.append(sid)
            deriv_owner_per_text.append(sid)
    deriv_embs = (
        await embed_batch(client, flat_deriv_texts)
        if flat_deriv_texts
        else np.zeros((0, EMBED_DIMS), dtype=np.float32)
    )
    deriv_embs = normalize(deriv_embs) if deriv_embs.size else deriv_embs

    # ---- Build A and B candidate matrices.
    # Index A: only seg_embs.
    cand_embs_A = seg_embs
    cand_ids_A = seg_ids

    # Index B: seg_embs + deriv_embs, all mapped back to segment_id.
    if deriv_embs.size:
        cand_embs_B = np.vstack([seg_embs, deriv_embs])
        cand_ids_B = seg_ids + flat_deriv_segids
    else:
        cand_embs_B = seg_embs
        cand_ids_B = seg_ids

    # ---- Retrieve top-10 from each.
    topk_A = retrieve_topk(qemb, cand_embs_A, cand_ids_A, k_max=10)
    topk_B = retrieve_topk(qemb, cand_embs_B, cand_ids_B, k_max=10)

    def hit_at_k(topk, k):
        return any(sid in has_answer_set for sid in topk[:k])

    rA = {k: int(hit_at_k(topk_A, k)) for k in (1, 5, 10)}
    rB = {k: int(hit_at_k(topk_B, k)) for k in (1, 5, 10)}

    # Capture diagnostic info for win/loss analysis.
    {s["segment_id"]: s for s in segments}
    diag: dict[str, Any] = {
        "topk_A": topk_A,
        "topk_B": topk_B,
        "rA": rA,
        "rB": rB,
    }

    if save_examples:
        # Identify, for B's top-10, which slots came from a derivative match (and
        # whether the verbatim seg was below the cutoff).
        sims_B = cand_embs_B @ qemb
        best_per_seg_B: dict[str, tuple[float, int]] = {}
        for i, sid in enumerate(cand_ids_B):
            s = float(sims_B[i])
            if sid not in best_per_seg_B or s > best_per_seg_B[sid][0]:
                best_per_seg_B[sid] = (s, i)
        # is_deriv_for_top: bool indicating whether top sim came from a derivative.
        topB_sources = []
        for sid in topk_B[:10]:
            _, idx = best_per_seg_B[sid]
            is_deriv = idx >= len(seg_embs)
            src_text = (
                seg_texts[idx]
                if not is_deriv
                else flat_deriv_texts[idx - len(seg_embs)]
            )
            topB_sources.append(
                {
                    "segment_id": sid,
                    "from_derivative": is_deriv,
                    "match_text": src_text,
                    "is_gold": sid in has_answer_set,
                }
            )
        diag["topB_sources"] = topB_sources

        # Capture gold segment(s) text for reporting.
        gold = [
            {
                "segment_id": s["segment_id"],
                "role": s["role"],
                "text": s["text"],
                "in_topA": s["segment_id"] in topk_A,
                "in_topB": s["segment_id"] in topk_B,
                "rank_A": (
                    topk_A.index(s["segment_id"]) + 1
                    if s["segment_id"] in topk_A
                    else None
                ),
                "rank_B": (
                    topk_B.index(s["segment_id"]) + 1
                    if s["segment_id"] in topk_B
                    else None
                ),
                "derivatives": [
                    flat_deriv_texts[i]
                    for i, dsid in enumerate(flat_deriv_segids)
                    if dsid == s["segment_id"]
                ],
            }
            for s in segments
            if s["has_answer"]
        ]
        diag["gold_segments"] = gold

    return {
        "question_id": q.question_id,
        "question_type": str(q.question_type),
        "question": q.question,
        "answer": q.answer,
        "n_segments": len(segments),
        "n_gold_segments": len(has_answer_set),
        "n_derivatives": len(flat_deriv_texts),
        "rA_at": rA,
        "rB_at": rB,
        "diag": diag,
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
    # Round-robin so each type gets at least one if possible.
    out: list[LongMemEvalItem] = []
    pools = {t: rng.sample(by_type[t], len(by_type[t])) for t in types}
    while len(out) < n:
        for t in types:
            if not pools[t]:
                continue
            out.append(pools[t].pop())
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Per-question deriver concurrency cap.",
    )
    parser.add_argument(
        "--question-concurrency",
        type=int,
        default=4,
        help="Number of questions running in parallel.",
    )
    parser.add_argument("--output", default="eval_recall_at_k.json")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test override: process at most N questions.",
    )
    args = parser.parse_args()

    questions = load_longmemeval_dataset(DATA_PATH)
    sample = stratified_sample(questions, args.num_questions)
    if args.limit:
        sample = sample[: args.limit]

    print(f"Sampled {len(sample)} questions:")
    type_counts: dict[str, int] = defaultdict(int)
    for q in sample:
        type_counts[str(q.question_type)] += 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=120)
    splitter = make_splitter()
    deriver_sem = asyncio.Semaphore(args.concurrency)
    q_sem = asyncio.Semaphore(args.question_concurrency)

    out_path = Path(__file__).parent / args.output
    # Keep partial progress: write after each question.
    results: list[dict] = []

    async def go(q: LongMemEvalItem):
        async with q_sem:
            t0 = time.time()
            try:
                r = await eval_question(client, q, splitter, deriver_sem)
            except Exception as e:
                print(f"  [error] {q.question_id}: {e}")
                return {"question_id": q.question_id, "error": str(e)}
            elapsed = time.time() - t0
            r["elapsed_s"] = elapsed
            print(
                f"  [{q.question_id} {q.question_type!s:40s}] "
                f"segs={r.get('n_segments', 0):4d} "
                f"derivs={r.get('n_derivatives', 0):4d} "
                f"R@1 A/B={r.get('rA_at', {}).get(1, '-')}/{r.get('rB_at', {}).get(1, '-')} "
                f"R@5 A/B={r.get('rA_at', {}).get(5, '-')}/{r.get('rB_at', {}).get(5, '-')} "
                f"R@10 A/B={r.get('rA_at', {}).get(10, '-')}/{r.get('rB_at', {}).get(10, '-')} "
                f"({elapsed:.0f}s)",
                flush=True,
            )
            results.append(r)
            # checkpoint
            with out_path.open("w") as f:
                json.dump({"results": results}, f, indent=2, default=str)
            return r

    t_start = time.time()
    print(f"\nRunning at {time.strftime('%H:%M:%S')}", flush=True)
    await asyncio.gather(*[go(q) for q in sample])
    elapsed = time.time() - t_start

    # ---- Aggregate.
    per_type: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[str, list[int]] = defaultdict(list)
    n_segs_total, n_derivs_total = 0, 0
    for r in results:
        if "rA_at" not in r:
            continue
        n_segs_total += r["n_segments"]
        n_derivs_total += r["n_derivatives"]
        for k in (1, 5, 10):
            overall[f"A@{k}"].append(r["rA_at"][k])
            overall[f"B@{k}"].append(r["rB_at"][k])
            per_type[r["question_type"]][f"A@{k}"].append(r["rA_at"][k])
            per_type[r["question_type"]][f"B@{k}"].append(r["rB_at"][k])

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "elapsed_s": elapsed,
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

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    with out_path.open("w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, default=str)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
