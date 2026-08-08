"""Swiss-tournament reranking probe (experimental branch).

Question: can a Swiss-system tournament of pairwise LM judgments produce a
top-k ranking that beats (a) the initial embedding ranking and (b) a
dedicated cross-encoder (Cohere rerank-v3-5), at sub-quadratic
super-linear cost?

Setup (per branch directive):
  - recursive TextSegmenter + WholeTextDeriver, embeddings only (no BM25)
  - one vector query per question yields the candidate POOL; all three
    arms reorder the SAME pool, so the comparison is apples-to-apples.

Arms (all reorder the identical pool):
  1. embedding   -- pool as returned (sorted by cosine)
  2. cohere      -- Cohere rerank-v3-5 .score() on the pool, inline
  3. swiss       -- Swiss tournament of gpt-5-nano pairwise judgments,
                    first-round pairing seeded by embedding score,
                    R = ceil(log2(N)) rounds, single randomized order.

Metric (judge-free, deterministic): recall@k and nDCG@k of GOLD evidence.
Gold message timestamp = session_datetime + (turn-1) seconds, matching
locomo_ingest. A pool item is gold if its seed segment timestamp matches
any gold-evidence timestamp for the question.

This is a standalone read-only probe: it opens the stores, uses a
reranker=None EventMemory only to fetch the embedding-ordered pool, and
does Cohere/Swiss inline. It does not touch the Reranker/EventMemory
wiring or any existing flow.

Usage:
  uv run python swiss_rerank_probe.py \
    --segment-db swiss-textwhole-c2sub.sqlite \
    --vector-db swiss-textwhole-c2sub.vec.sqlite \
    --data-path ../../data/locomo10_c2sub.json \
    --pool-size 30 --top-k 10 \
    --categories 1,2 --limit 60 \
    --out swiss-probe-c1c2-n30.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import time
from datetime import timedelta

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import create_async_engine

from embedder_factory import build_embedder
from locomo_models import (
    attachment_suffix,
    datetime_from_locomo_time,
    load_locomo_dataset,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    FormatOptions,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
    SQLiteVecVectorStore,
    SQLiteVecVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)


# Render candidates exactly as the production answerer sees them
# (locomo_search.py:62): full date + short time. The filter must judge a
# candidate in the same form it will be presented for answering -- and full
# dates disambiguate the temporal questions ("months between", "how long").
_FORMAT_OPTIONS = FormatOptions(date_style="full", time_style="short")


# ---------------------------------------------------------------- gold map


def build_gold_timestamps(conversation: dict) -> dict[str, object]:
    """dia_id ("D6:15") -> message timestamp, matching ingest scheme."""
    out: dict[str, object] = {}
    session_idx = 0
    while True:
        session_idx += 1
        session_id = f"session_{session_idx}"
        if session_id not in conversation:
            break
        session = conversation[session_id]
        session_dt = datetime_from_locomo_time(
            conversation[f"{session_id}_date_time"]
        )
        for msg_idx, msg in enumerate(session):
            ts = session_dt + msg_idx * timedelta(seconds=1)
            out[msg["dia_id"]] = ts
    return out


# ---------------------------------------------------------------- metrics


def recall_at_k(ranked_is_gold: list[bool], total_gold: int, k: int) -> float:
    if total_gold == 0:
        return float("nan")
    hits = sum(1 for g in ranked_is_gold[:k] if g)
    return hits / min(total_gold, k) if min(total_gold, k) else float("nan")


def ndcg_at_k(ranked_is_gold: list[bool], total_gold: int, k: int) -> float:
    if total_gold == 0:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, g in enumerate(ranked_is_gold[:k]) if g
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(total_gold, k)))
    return dcg / ideal if ideal else float("nan")


# ---------------------------------------------------------------- swiss


PAIRWISE_PROMPT = """\
You are given a QUERY and two candidate MEMORIES, A and B, each retrieved \
from a history. The QUERY may be a question, a search phrase, a topic, an \
instruction, or a task. Each memory may or may not be relevant.{context_block}

QUERY:
{query}

MEMORY A:
{doc_a}

MEMORY B:
{doc_b}

Which memory is more useful for the QUERY -- the one that more directly \
provides, contains, or satisfies what the QUERY calls for? A memory is \
strongly relevant only if it matches every specific constraint the QUERY \
expresses -- any particular entity, name, time or date, place, quantity, \
attribute, or condition -- not merely its general topic. Prefer the memory \
that exactly and specifically satisfies the QUERY over one that is only \
related or generally on-topic. If the two are genuinely equal, say TIE.

Reply with exactly one token: A, B, or TIE."""


# Prompt variants for A/B + noise-floor replication. Selected via
# --prompt-variant; reassigns PAIRWISE_PROMPT at startup.
PROMPTS = {
    "memhist": PAIRWISE_PROMPT,
    "genericv2": (
        "You are given a QUERY and two candidate items, A and B. The "
        "QUERY may be a question, a search phrase, a topic, an "
        "instruction, or a task. Each candidate is a piece of text that "
        "may or may not be relevant.{context_block}\n\nQUERY:\n{query}\n\n"
        "CANDIDATE A:\n{doc_a}\n\nCANDIDATE B:\n{doc_b}\n\n"
        "Which candidate is more useful for the QUERY -- the one that "
        "more directly provides, contains, or satisfies what the QUERY "
        "calls for? A candidate is strongly relevant only if it matches "
        "every specific constraint the QUERY expresses -- any particular "
        "entity, name, time or date, place, quantity, attribute, or "
        "condition -- not merely its general topic. Prefer the candidate "
        "that exactly and specifically satisfies the QUERY over one that "
        "is only related or generally on-topic. If the two are genuinely "
        "equal, say TIE.\n\nReply with exactly one token: A, B, or TIE."
    ),
}


async def pairwise_judge(
    client: AsyncOpenAI, model: str, question: str,
    doc_a: str, doc_b: str, effort: str = "low", domain_hint: str = "",
) -> tuple[str, int, int]:
    """Return (verdict, input_tokens, output_tokens). verdict in A/B/TIE."""
    context_block = (
        f"\n\nCONTEXT: {domain_hint}" if domain_hint else ""
    )
    prompt = PAIRWISE_PROMPT.format(
        query=question, doc_a=doc_a, doc_b=doc_b,
        context_block=context_block,
    )
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"reasoning_effort": effort},
        )
    except Exception:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    text = (resp.choices[0].message.content or "").strip().upper()
    usage = resp.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    if text.startswith("A"):
        return "A", in_tok, out_tok
    if text.startswith("B"):
        return "B", in_tok, out_tok
    return "TIE", in_tok, out_tok


async def swiss_tournament(
    client: AsyncOpenAI, model: str, question: str,
    docs: list[str], embed_order: list[int], rounds: int, rng: random.Random,
    effort: str = "low",
) -> tuple[list[int], int, int, int]:
    """Swiss tournament over candidate indices.

    docs[i] is candidate i's text. embed_order is candidate indices sorted
    best->worst by embedding score (the seeding). Returns (final_order,
    n_comparisons, input_tokens, output_tokens).
    """
    n = len(docs)
    if n <= 1:
        return list(range(n)), 0, 0, 0
    points: dict[int, float] = {i: 0.0 for i in range(n)}
    played: dict[int, set[int]] = {i: set() for i in range(n)}
    seed_rank = {cid: r for r, cid in enumerate(embed_order)}  # lower=better
    n_comparisons = 0
    tok_in = 0
    tok_out = 0

    async def one_game(a: int, b: int) -> None:
        nonlocal n_comparisons, tok_in, tok_out
        n_comparisons += 1
        # single randomized order to spread position bias
        if rng.random() < 0.5:
            verdict, ti, to = await pairwise_judge(client, model, question,
                                                   docs[a], docs[b], effort)
            winner = {"A": a, "B": b, "TIE": None}[verdict]
        else:
            verdict, ti, to = await pairwise_judge(client, model, question,
                                                   docs[b], docs[a], effort)
            winner = {"A": b, "B": a, "TIE": None}[verdict]
        tok_in += ti
        tok_out += to
        if winner is None:
            points[a] += 0.5
            points[b] += 0.5
        else:
            points[winner] += 1.0
        played[a].add(b)
        played[b].add(a)

    for _ in range(rounds):
        # standing: points desc, then seed (embedding) better-first
        standing = sorted(
            range(n), key=lambda c: (-points[c], seed_rank[c])
        )
        used: set[int] = set()
        games: list[tuple[int, int]] = []
        i = 0
        while i < len(standing):
            a = standing[i]
            if a in used:
                i += 1
                continue
            # find nearest next opponent not yet played, not used
            partner = None
            for j in range(i + 1, len(standing)):
                b = standing[j]
                if b in used:
                    continue
                if b in played[a]:
                    continue
                partner = b
                break
            if partner is None:
                # allow rematch with nearest available, else bye
                for j in range(i + 1, len(standing)):
                    b = standing[j]
                    if b not in used:
                        partner = b
                        break
            if partner is None:
                points[a] += 0.5  # bye
                used.add(a)
            else:
                used.add(a)
                used.add(partner)
                games.append((a, partner))
            i += 1
        await asyncio.gather(*(one_game(a, b) for a, b in games))

    # Buchholz tiebreak: sum of opponents' points
    buchholz = {
        c: sum(points[o] for o in played[c]) for c in range(n)
    }
    final = sorted(
        range(n),
        key=lambda c: (-points[c], -buchholz[c], seed_rank[c]),
    )
    return final, n_comparisons, tok_in, tok_out


# ------------------------------------------------ listwise (RankGPT/JointRank-family)


LISTWISE_PROMPT = """\
You are given a QUERY and a numbered list of candidate MEMORIES retrieved \
from a history. The QUERY may be a question, a search phrase, a topic, an \
instruction, or a task.

Rank the memories from MOST to LEAST useful for the QUERY. A memory is \
strongly relevant only if it matches every specific constraint the QUERY \
expresses -- any particular entity, name, time or date, place, quantity, \
attribute, or condition -- not merely its general topic.

QUERY:
{query}

MEMORIES:
{numbered_docs}

Output ONLY the memory numbers, most useful first, comma-separated \
(e.g. 7, 2, 15). Include every number exactly once."""


async def listwise_rank(
    client: AsyncOpenAI, model: str, question: str,
    docs: list[str], embed_order: list[int], effort: str = "low",
) -> tuple[list[int], int, int]:
    """One LLM call ranks the whole pool. Returns (order, ti, to).

    Missing/garbled labels are appended in embedding order (stable fallback).
    """
    numbered = "\n".join(f"[{i}] {docs[i]}" for i in range(len(docs)))
    prompt = LISTWISE_PROMPT.format(query=question, numbered_docs=numbered)
    try:
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            extra_body={"reasoning_effort": effort},
        )
    except Exception:
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
        )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    ti = usage.prompt_tokens if usage else 0
    to = usage.completion_tokens if usage else 0
    # parse integers in order, dedup, keep valid indices
    seen: set[int] = set()
    order: list[int] = []
    for tok in re.findall(r"\d+", text):
        idx = int(tok)
        if 0 <= idx < len(docs) and idx not in seen:
            seen.add(idx)
            order.append(idx)
    # append any missing in embedding order
    for c in embed_order:
        if c not in seen:
            order.append(c)
    return order, ti, to


# ------------------------------------------------ pointwise sufficiency filter


FILTER_PROMPT = """\
You are given a QUERY and one candidate ITEM. The ITEM begins with its date \
and speaker in brackets -- like [Friday, May 20, 2022, 7:49 PM] Speaker: \
"..." -- and that date and speaker ARE part of the ITEM's information: a date \
in the bracket is a date the ITEM supplies. Decide whether the ITEM supplies \
any information the QUERY's answer would use (KEEP) or none (DROP).

You are NOT answering the QUERY and NOT computing its answer. You do not need \
the ITEM to state a total, a count, an ordinal ("the second", "the third"), \
a duration, or the final answer. KEEP the ITEM if it supplies even one piece \
the answer is built from -- a single event, a date (including the bracketed \
one), a name, or one instance of the kind the QUERY counts or chooses among. \
The counting, ordering, and arithmetic happen later, with all items together.

- "how many X has P done" or "which X was the Nth": KEEP every ITEM reporting \
one X by P, even if it never says how many or which number -- it is one \
instance to be counted later.
- "time between A and B": KEEP an ITEM reporting A or B with its date -- it \
is one endpoint.
- If only PART of the ITEM supplies a piece, KEEP it; ignore surrounding \
greetings or small talk.

DROP the ITEM only if it supplies no such piece. The single thing you may not \
do is invent what the ITEM is ABOUT: if making it relevant requires supposing \
it concerns the QUERY's subject when nothing in the ITEM -- its text, date, \
or speaker -- says so, DROP it. ("7" for "how many pets": nothing in it is \
about pets -> DROP. A message where one person reports THEIR OWN action, for \
a QUERY about someone else's action -> DROP.) But combining a piece the ITEM \
does state with other items later is NOT invention.

QUERY:
{query}

ITEM:
{doc}

Reply with exactly one token: KEEP or DROP."""


async def pointwise_filter(
    client: AsyncOpenAI, model: str, question: str, doc: str,
    effort: str = "low",
) -> tuple[str, int, int]:
    """One pointwise helps-answer judgment. Returns (verdict, ti, to);
    verdict in KEEP/DROP. Anything but an explicit DROP -> KEEP (the filter
    drops only the confident non-contributor; uncertainty keeps)."""
    prompt = FILTER_PROMPT.format(query=question, doc=doc)
    try:
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            extra_body={"reasoning_effort": effort},
        )
    except Exception:
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
        )
    text = (resp.choices[0].message.content or "").strip().upper()
    usage = resp.usage
    ti = usage.prompt_tokens if usage else 0
    to = usage.completion_tokens if usage else 0
    if text.startswith("DROP"):
        return "DROP", ti, to
    return "KEEP", ti, to  # default KEEP (recall-safe), incl. unparseable


# ------------------------------------------------ JointRank (block-listwise)


def build_ebd_blocks(
    n: int, k: int, r: int, rng: random.Random
) -> list[list[int]]:
    """Regular Equi-Replicate Block Design (JointRank Sec 4.4): concatenate
    r independent shuffles of [0..n), partition into blocks of k. Each item
    appears in exactly r blocks; placement randomized; connectivity for
    aggregation."""
    seq: list[int] = []
    for _ in range(r):
        s = list(range(n))
        rng.shuffle(s)
        seq.extend(s)
    blocks = [seq[i:i + k] for i in range(0, len(seq), k)]
    return [b for b in blocks if len(b) >= 2]


async def jointrank_blocks(
    client: AsyncOpenAI, model: str, question: str,
    docs: list[str], embed_order: list[int], k: int, r: int,
    rng: random.Random, effort: str = "low", call_timeout: float = 0.0,
) -> tuple[np.ndarray, int, int, int, int, int]:
    """Rank EBD blocks listwise in ONE phase; accumulate implicit pairwise
    wins. Returns (W, n_blocks, n_calls_done, dropped, ti, to).
    W[i,j] = # blocks where i ranked above j."""
    n = len(docs)
    blocks = build_ebd_blocks(n, k, r, rng)

    async def do_block(block):
        bdocs = [docs[g] for g in block]
        try:
            coro = listwise_rank(client, model, question, bdocs,
                                 list(range(len(block))), effort)
            if call_timeout > 0:
                local, ti, to = await asyncio.wait_for(coro, call_timeout)
            else:
                local, ti, to = await coro
        except (asyncio.TimeoutError, Exception):
            return None, 0, 0
        return [block[p] for p in local], ti, to

    outs = await asyncio.gather(*(do_block(b) for b in blocks))
    W = np.zeros((n, n))
    ti = to = ncalls = dropped = 0
    for go, a, b in outs:
        ti += a
        to += b
        if go is None:
            dropped += 1
            continue
        ncalls += 1
        for p in range(len(go)):
            for q in range(p + 1, len(go)):
                W[go[p], go[q]] += 1  # go[p] ranked above go[q]
    return W, len(blocks), ncalls, dropped, ti, to


def agg_winrate(n: int, W: np.ndarray, embed_order: list[int]) -> list[int]:
    out = W.sum(1)
    inn = W.sum(0)
    tot = out + inn
    rate = np.where(tot > 0, out / np.maximum(tot, 1), 0.0)
    seed = {c: r for r, c in enumerate(embed_order)}
    return sorted(range(n), key=lambda c: (-rate[c], seed[c]))


def agg_pagerank(
    n: int, W: np.ndarray, embed_order: list[int],
    alpha: float = 0.15, to_seed: bool = False, iters: int = 100,
) -> list[int]:
    """PageRank on the win graph. Walk flows loser->winner: A[i,j]=W[j,i].
    alpha = teleport prob (to uniform, or to embedding seed if to_seed)."""
    A = W.T.astype(float).copy()
    dmax = max(1.0, A.sum(1).max())
    P = A / dmax
    np.fill_diagonal(P, 1.0 - P.sum(1))
    if to_seed:
        s0 = _seed_scores(n, embed_order)
        v = np.exp(s0)
        v = v / v.sum()
    else:
        v = np.ones(n) / n
    pi = v.copy()
    for _ in range(iters):
        pi = alpha * v + (1 - alpha) * (P.T @ pi)
        pi = pi / pi.sum()
    seed = {c: r for r, c in enumerate(embed_order)}
    return sorted(range(n), key=lambda c: (-pi[c], seed[c]))


# ------------------------------------------------ single-phase sparse


def build_sparse_pairs(
    n: int, degree: int, rng: random.Random
) -> list[tuple[int, int]]:
    """~degree-regular random comparison graph, m ~= n*degree/2 edges.

    Guarantees every node appears at least once (connected-ish coverage;
    seeding handles any residual disconnection at aggregation time).
    """
    if n <= 1:
        return []
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_pairs)
    m = min(len(all_pairs), max(n // 2, (n * degree) // 2))
    pairs = all_pairs[:m]
    seen = {x for p in pairs for x in p}
    # ensure coverage: pair any missing node with a random present node
    for node in range(n):
        if node not in seen:
            other = rng.randrange(n)
            while other == node:
                other = rng.randrange(n)
            pairs.append((node, other))
            seen.add(node)
    return pairs


async def sparse_single_phase(
    client: AsyncOpenAI, model: str, question: str,
    docs: list[str], pairs: list[tuple[int, int]], rng: random.Random,
    effort: str = "low", call_timeout: float = 0.0,
) -> tuple[dict, int, int, int, int]:
    """Fire ALL comparisons in one phase.

    Returns (results, n_done, dropped, ti, to). results maps (i,j) i<j ->
    +1 i wins / -1 j wins / 0 tie. Comparisons exceeding call_timeout are
    DROPPED (excluded) -- the aggregators (esp. seeded HodgeRank's lam-I)
    tolerate the resulting incomplete graph, so dropping stragglers caps
    tail latency at ~zero quality cost.
    """
    plan = []
    for (i, j) in pairs:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        plan.append((i, j, a, b))

    async def game(i, j, a, b):
        try:
            coro = pairwise_judge(client, model, question,
                                  docs[a], docs[b], effort)
            if call_timeout > 0:
                verdict, ti, to = await asyncio.wait_for(coro, call_timeout)
            else:
                verdict, ti, to = await coro
        except (asyncio.TimeoutError, Exception):
            return (i, j), None, 0, 0  # dropped straggler
        if verdict == "TIE":
            out = 0
        else:
            winner = a if verdict == "A" else b
            out = 1 if winner == i else -1
        return (i, j), out, ti, to

    outcomes = await asyncio.gather(
        *(game(i, j, a, b) for (i, j, a, b) in plan)
    )
    results: dict[tuple[int, int], int] = {}
    tok_in = tok_out = dropped = 0
    for key, out, ti, to in outcomes:
        if out is None:
            dropped += 1
            continue
        results[key] = out
        tok_in += ti
        tok_out += to
    return results, len(results), dropped, tok_in, tok_out


def _seed_scores(n: int, embed_order: list[int]) -> np.ndarray:
    """z-scored rank-based seed: best embedding rank -> highest score."""
    rank = np.empty(n)
    for r, cid in enumerate(embed_order):
        rank[cid] = r
    s = -(rank - rank.mean())
    sd = s.std()
    return s / sd if sd > 0 else s


def agg_borda(
    n: int, results: dict, embed_order: list[int]
) -> list[int]:
    wins = np.zeros(n)
    games = np.zeros(n)
    for (i, j), out in results.items():
        games[i] += 1
        games[j] += 1
        if out == 1:
            wins[i] += 1
        elif out == -1:
            wins[j] += 1
        else:
            wins[i] += 0.5
            wins[j] += 0.5
    rate = np.where(games > 0, wins / np.maximum(games, 1), 0.0)
    seed_rank = {cid: r for r, cid in enumerate(embed_order)}
    return sorted(range(n), key=lambda c: (-rate[c], seed_rank[c]))


def agg_hodgerank(
    n: int, results: dict, embed_order: list[int], lam: float
) -> tuple[list[int], float]:
    """Seeded regularized HodgeRank: (L + lam I) x = b + lam s0.

    Returns (order, intransitivity_index). lam=0 => unseeded (pinv).
    """
    L = np.zeros((n, n))
    b = np.zeros(n)
    energy_y = 0.0
    for (i, j), out in results.items():
        if out == 0:
            y = 0.0
        else:
            y = 1.0 if out == 1 else -1.0  # y_ij: i beats j -> x_i-x_j ~ +1
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1
        b[i] += y
        b[j] -= y
        energy_y += y * y
    s0 = _seed_scores(n, embed_order)
    if lam > 0:
        x = np.linalg.solve(L + lam * np.eye(n), b + lam * s0)
    else:
        x = np.linalg.lstsq(L, b, rcond=None)[0]
    # intransitivity: residual energy not explained by the gradient flow
    resid = 0.0
    for (i, j), out in results.items():
        if out == 0:
            y = 0.0
        else:
            y = 1.0 if out == 1 else -1.0
        resid += (x[i] - x[j] - y) ** 2
    intransitivity = resid / energy_y if energy_y > 0 else 0.0
    seed_rank = {cid: r for r, cid in enumerate(embed_order)}
    order = sorted(range(n), key=lambda c: (-x[c], seed_rank[c]))
    return order, float(intransitivity)


def agg_rank_centrality(
    n: int, results: dict, embed_order: list[int],
    alpha: float = 0.15, iters: int = 100,
) -> list[int]:
    """Seeded Rank Centrality via personalized PageRank teleportation."""
    # A[i,j] = number of times j beats i (walk moves toward winners)
    A = np.zeros((n, n))
    for (i, j), out in results.items():
        if out == 1:
            A[j, i] += 1  # i beats j -> mass flows j->i
        elif out == -1:
            A[i, j] += 1
        else:
            A[i, j] += 0.5
            A[j, i] += 0.5
    dmax = max(1.0, A.sum(axis=1).max())
    P = A / dmax
    np.fill_diagonal(P, 1.0 - P.sum(axis=1))
    # seed teleport vector from embedding rank (top-heavy)
    s0 = _seed_scores(n, embed_order)
    v = np.exp(s0)
    v = v / v.sum()
    pi = v.copy()
    for _ in range(iters):
        pi = alpha * v + (1 - alpha) * (P.T @ pi)
        pi = pi / pi.sum()
    seed_rank = {cid: r for r, cid in enumerate(embed_order)}
    return sorted(range(n), key=lambda c: (-pi[c], seed_rank[c]))


# ---------------------------------------------------------------- main


async def main() -> None:
    # locomo-local .env carries AWS creds (Cohere via Bedrock) + OpenAI
    load_dotenv(
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/"
        "locomo/.env"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-db", required=True)
    parser.add_argument("--vector-db", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", default="swiss",
                        choices=["swiss", "sparse", "listwise", "jointrank",
                                 "filter"],
                        help="swiss = adaptive rounds; sparse = single-phase "
                        "pairwise tournament; listwise = one call ranks whole "
                        "pool (RankGPT); jointrank = EBD block-listwise + agg; "
                        "filter = pointwise keep/drop sufficiency filter, no "
                        "ranking stage (passers keep embedding order)")
    parser.add_argument("--block-size", type=int, default=20,
                        help="jointrank: candidates per block (k)")
    parser.add_argument("--block-reps", type=int, default=3,
                        help="jointrank: replication factor (r), each item "
                        "in r blocks")
    parser.add_argument("--degree", type=int, default=0,
                        help="sparse: per-item comparison degree; "
                        "0 = ceil(log2(pool)) (~n log n budget)")
    parser.add_argument("--seed-lambda", type=float, default=0.5,
                        help="sparse: HodgeRank seed-trust (Tikhonov lambda)")
    parser.add_argument("--call-timeout", type=float, default=0.0,
                        help="sparse: per-comparison timeout (s); stragglers "
                        "dropped. 0 = off")
    parser.add_argument("--http-pool", type=int, default=0,
                        help="raise AsyncOpenAI connection pool (0 = SDK "
                        "default); set ~200 for full single-phase fan-out")
    parser.add_argument("--rounds", type=int, default=0,
                        help="0 = ceil(log2(pool))")
    parser.add_argument("--comparator-model", default="gpt-5-nano")
    parser.add_argument("--comparator-effort", default="low",
                        choices=["none", "minimal", "low", "medium", "high"])
    parser.add_argument("--categories", default="1,2,4")
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = all questions in selected categories")
    # gpt-5-nano list price (USD per 1M tokens); override if it changes.
    parser.add_argument("--price-in", type=float, default=0.05,
                        help="USD per 1M input tokens (comparator)")
    parser.add_argument("--price-out", type=float, default=0.40,
                        help="USD per 1M output tokens (comparator)")
    parser.add_argument("--cohere", action="store_true",
                        help="add Cohere rerank-v3-5 arm (AWS Bedrock)")
    parser.add_argument("--no-swiss", action="store_true",
                        help="skip the Swiss arm (e.g. fast Cohere-only bar)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-variant", default="memhist",
                        choices=["memhist", "genericv2"])
    parser.add_argument("--out", default="swiss-probe.json")
    args = parser.parse_args()
    global PAIRWISE_PROMPT
    PAIRWISE_PROMPT = PROMPTS[args.prompt_variant]

    rng = random.Random(args.seed)
    cats = {c.strip() for c in args.categories.split(",") if c.strip()}
    rounds = args.rounds or max(1, math.ceil(math.log2(args.pool_size)))
    degree = args.degree or max(2, math.ceil(math.log2(args.pool_size)))

    locomo_data = load_locomo_dataset(args.data_path)

    seg_engine = create_async_engine(
        f"sqlite+aiosqlite:///{args.segment_db}",
        connect_args={"timeout": 30}, pool_size=20, max_overflow=80,
    )
    segment_store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=seg_engine)
    )
    await segment_store.startup()
    vec_engine = create_async_engine(
        f"sqlite+aiosqlite:///{args.vector_db}",
        connect_args={"timeout": 30}, pool_size=20, max_overflow=80,
    )
    vector_store = SQLiteVecVectorStore(
        SQLiteVecVectorStoreParams(engine=vec_engine)
    )
    await vector_store.startup()

    if args.http_pool > 0:
        import httpx
        http_client = httpx.AsyncClient(limits=httpx.Limits(
            max_connections=args.http_pool,
            max_keepalive_connections=args.http_pool,
        ))
        # max_retries=0: SDK backoff-retries are the dominant tail; the
        # per-call timeout-drop handles failures instead.
        openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=http_client, max_retries=0,
        )
    else:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedder = build_embedder(args.embedding_model, openai_client)

    # Cohere rerank-v3-5 via AWS Bedrock (the frontier-reranker bar)
    cohere = None
    if args.cohere:
        import boto3
        from memmachine_server.common.reranker.amazon_bedrock_reranker import (
            AmazonBedrockReranker, AmazonBedrockRerankerParams,
        )
        region = "us-west-2"
        aws_client = boto3.client(
            "bedrock-agent-runtime", region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        cohere = AmazonBedrockReranker(AmazonBedrockRerankerParams(
            client=aws_client, region=region, model_id="cohere.rerank-v3-5:0",
        ))

    segmenter = TextSegmenter()  # unused for query; satisfies param validation
    deriver = WholeTextDeriver()

    records: list[dict] = []
    total_swiss_comparisons = 0
    total_tok_in = 0
    total_tok_out = 0
    t0 = time.monotonic()

    for idx, item in enumerate(locomo_data):
        if "conversation" not in item:
            continue
        partition_key = f"group_{idx}"
        collection = await vector_store.open_collection(
            namespace="locomo", name=partition_key
        )
        if collection is None:
            print(f"no collection group {idx}; skip")
            continue
        partition = await segment_store.open_or_create_partition(
            partition_key, SegmentStorePartitionConfig()
        )
        memory = EventMemory(EventMemoryParams(
            vector_store_collection=collection,
            segment_store_partition=partition,
            segmenter=segmenter, deriver=deriver,
            embedder=embedder, reranker=None,
        ))
        gold_ts = build_gold_timestamps(item["conversation"])

        questions = [q for q in item["qa"] if str(q["category"]) in cats]
        for qa in questions:
            question = qa["question"]
            category = str(qa["category"])
            evidence = qa.get("evidence", [])
            gold_set = {gold_ts[e] for e in evidence if e in gold_ts}
            if not gold_set:
                continue

            qr = await memory.query(
                query=question,
                vector_search_limit=args.pool_size,
                expand_context=0,
                format_options=_FORMAT_OPTIONS,
                bm25_fusion="none",
            )
            pool = qr.scored_segment_contexts
            if len(pool) < 2:
                continue

            docs = [
                EventMemory.string_from_segment_context(
                    ssc.segments, format_options=_FORMAT_OPTIONS
                )
                for ssc in pool
            ]
            is_gold = [
                any(seg.timestamp in gold_set for seg in ssc.segments)
                for ssc in pool
            ]
            total_gold = sum(is_gold)
            if total_gold == 0:
                # gold not even in the pool -> ranking can't fix; record
                records.append({
                    "category": category, "question": question,
                    "gold_in_pool": 0, "n_gold_evidence": len(gold_set),
                    "pool": len(pool),
                })
                continue

            npool = len(pool)
            emb_order = list(range(npool))  # arm: embedding (pool order)

            def metrics(order: list[int]) -> dict:
                rg = [is_gold[i] for i in order]
                return {
                    "recall@k": recall_at_k(rg, total_gold, args.top_k),
                    "ndcg@k": ndcg_at_k(rg, total_gold, args.top_k),
                    "first_gold_rank": next(
                        (r for r, g in enumerate(rg) if g), -1
                    ),
                    "gold_ranks": [r for r, g in enumerate(rg) if g],
                }

            rec = {
                "category": category, "question": question,
                "gold_in_pool": total_gold, "pool": npool,
                "embedding": metrics(emb_order),
            }

            # frontier-reranker arm (same pool). Cohere rerank is POINTWISE
            # (each query-doc scored independently), so >100 docs just chunk
            # into <=100-doc calls and merge -- scores are comparable across
            # chunks. n_calls = ceil(npool/100).
            if cohere is not None:
                cscores: list[float] = []
                for st in range(0, npool, 100):
                    cscores.extend(
                        await cohere.score(question, docs[st:st + 100])
                    )
                cohere_order = sorted(
                    range(npool), key=lambda i: cscores[i], reverse=True
                )
                rec["cohere"] = metrics(cohere_order)

            if args.mode == "listwise":
                t_q = time.monotonic()
                lw_order, ti, to = await listwise_rank(
                    openai_client, args.comparator_model, question,
                    docs, emb_order, args.comparator_effort,
                )
                rec["query_rerank_s"] = round(time.monotonic() - t_q, 3)
                rec["listwise_calls"] = 1
                ncmp = 1
                rec["listwise"] = metrics(lw_order)
            elif args.mode == "jointrank":
                t_q = time.monotonic()
                W, nblocks, ncalls, dropped, ti, to = await jointrank_blocks(
                    openai_client, args.comparator_model, question,
                    docs, emb_order, args.block_size, args.block_reps,
                    rng, args.comparator_effort, args.call_timeout,
                )
                rec["query_rerank_s"] = round(time.monotonic() - t_q, 3)
                rec["jr_blocks"] = nblocks
                rec["jr_calls"] = ncalls
                rec["dropped"] = dropped
                ncmp = ncalls
                rec["jr_winrate"] = metrics(
                    agg_winrate(npool, W, emb_order))
                rec["jr_pagerank"] = metrics(
                    agg_pagerank(npool, W, emb_order, to_seed=False))
                rec["jr_pagerank_seeded"] = metrics(
                    agg_pagerank(npool, W, emb_order, to_seed=True))
            elif args.mode == "filter":
                t_q = time.monotonic()

                async def fcall(i):
                    try:
                        coro = pointwise_filter(
                            openai_client, args.comparator_model, question,
                            docs[i], args.comparator_effort,
                        )
                        if args.call_timeout > 0:
                            return await asyncio.wait_for(
                                coro, args.call_timeout)
                        return await coro
                    except (asyncio.TimeoutError, Exception):
                        return "KEEP", 0, 0  # recall-safe default on timeout

                fouts = await asyncio.gather(
                    *(fcall(i) for i in range(npool))
                )
                verdicts = [v for v, _, _ in fouts]
                ti = sum(a for _, a, _ in fouts)
                to = sum(b for _, _, b in fouts)
                keepers = [i for i in emb_order if verdicts[i] == "KEEP"]
                droppers = [i for i in emb_order if verdicts[i] == "DROP"]
                rec["query_rerank_s"] = round(time.monotonic() - t_q, 3)
                rec["n_kept"] = len(keepers)
                # gold the filter dropped (recall harm; some may be FALSE
                # gold -- LoCoMo synthetic data has hallucinated evidence)
                rec["gold_dropped"] = sum(1 for i in droppers if is_gold[i])
                ncmp = npool
                # filter-only (binary keep/drop): keepers in embedding order,
                # droppers behind. Ranking among keepers is a SEPARATE stage.
                rec["filter_keep_embed"] = metrics(keepers + droppers)
                # retain enough to diagnose without re-running (verdicts are
                # stochastic; a re-run can't reproduce them): per-candidate
                # verdict, gold indices in pool order, and the text of any
                # gold the filter DROPped (for auditing false-DROP vs false-gold).
                rec["verdicts"] = verdicts
                rec["gold_pool_idx"] = [
                    i for i in range(npool) if is_gold[i]
                ]
                rec["gold_drop_docs"] = [
                    docs[i] for i in range(npool)
                    if is_gold[i] and verdicts[i] == "DROP"
                ]
            elif args.mode == "swiss":
                swiss_order, ncmp, ti, to = await swiss_tournament(
                    openai_client, args.comparator_model, question,
                    docs, emb_order, rounds, rng, args.comparator_effort,
                )
                rec["rounds"] = rounds
                rec["swiss_comparisons"] = ncmp
                rec["swiss"] = metrics(swiss_order)
            else:  # sparse single-phase
                pairs = build_sparse_pairs(npool, degree, rng)
                t_q = time.monotonic()
                results, ncmp, dropped, ti, to = await sparse_single_phase(
                    openai_client, args.comparator_model, question,
                    docs, pairs, rng, args.comparator_effort,
                    args.call_timeout,
                )
                rec["query_rerank_s"] = round(time.monotonic() - t_q, 3)
                rec["degree"] = degree
                rec["sparse_comparisons"] = ncmp
                rec["dropped"] = dropped
                rec["sparse_borda"] = metrics(
                    agg_borda(npool, results, emb_order)
                )
                ho, intrans0 = agg_hodgerank(npool, results, emb_order, 0.0)
                rec["sparse_hodge"] = metrics(ho)
                hos, _ = agg_hodgerank(
                    npool, results, emb_order, args.seed_lambda
                )
                rec["sparse_hodge_seeded"] = metrics(hos)
                rec["sparse_rc_seeded"] = metrics(
                    agg_rank_centrality(npool, results, emb_order)
                )
                rec["intransitivity"] = round(intrans0, 4)

            total_swiss_comparisons += ncmp
            total_tok_in += ti
            total_tok_out += to
            records.append(rec)

            main_arm = {"swiss": "swiss", "listwise": "listwise",
                        "jointrank": "jr_pagerank",
                        "filter": "filter_keep_embed"}.get(
                args.mode, "sparse_hodge_seeded")
            print(
                f"[g{idx} c{category}] gip={total_gold} "
                f"emb R@k={rec['embedding']['recall@k']:.2f} "
                + (f"coh={rec['cohere']['recall@k']:.2f} "
                   if 'cohere' in rec else "")
                + f"{args.mode}={rec[main_arm]['recall@k']:.2f} "
                + (f"kept={rec['n_kept']}/{npool} "
                   if 'n_kept' in rec else "")
                + (f"golddrop={rec['gold_dropped']} "
                   if rec.get('gold_dropped') else "")
                + (f"intrans={rec['intransitivity']:.2f} "
                   if 'intransitivity' in rec else "")
                + f"(cmp={ncmp})"
            )

            if args.limit and sum(
                1 for r in records if r.get("gold_in_pool", 0) > 0
            ) >= args.limit:
                break
        if args.limit and sum(
            1 for r in records if r.get("gold_in_pool", 0) > 0
        ) >= args.limit:
            break

    # ---- summary
    scored = [r for r in records if r.get("gold_in_pool", 0) > 0]
    rank_arms = {
        "swiss": ["swiss"],
        "listwise": ["listwise"],
        "jointrank": ["jr_winrate", "jr_pagerank", "jr_pagerank_seeded"],
        "filter": ["filter_keep_embed"],
    }.get(args.mode, ["sparse_borda", "sparse_hodge",
                      "sparse_hodge_seeded", "sparse_rc_seeded"])
    arms = ["embedding"] + (
        ["cohere"] if any("cohere" in r for r in scored) else []
    ) + rank_arms
    mean_intrans = (
        round(float(np.mean([r["intransitivity"] for r in scored
                             if "intransitivity" in r])), 4)
        if any("intransitivity" in r for r in scored) else None
    )

    def agg(arm: str, metric: str, cat: str | None) -> float:
        vals = [
            r[arm][metric] for r in scored
            if arm in r and (cat is None or r["category"] == cat)
            and not math.isnan(r[arm][metric])
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    n_scored = len(scored)
    cost_usd = (
        total_tok_in / 1e6 * args.price_in
        + total_tok_out / 1e6 * args.price_out
    )
    cost_per_q = cost_usd / n_scored if n_scored else float("nan")
    cmp_per_q = (
        total_swiss_comparisons / n_scored if n_scored else float("nan")
    )
    # Frontier reranker reference: Cohere rerank-v3.5 ~= $2.00 / 1000 queries
    # (1 search = 1 query over the pool). Express Swiss as a multiple.
    cohere_ref_per_q = 2.00 / 1000
    summary = {
        "n_scored": n_scored,
        "n_gold_not_in_pool": sum(
            1 for r in records if r.get("gold_in_pool", 0) == 0
            and "n_gold_evidence" in r
        ),
        "pool_size": args.pool_size, "top_k": args.top_k,
        "mode": args.mode, "rounds": rounds, "degree": degree,
        "comparator": args.comparator_model,
        "comparator_effort": args.comparator_effort,
        "mean_intransitivity": mean_intrans,
        "call_timeout": args.call_timeout, "http_pool": args.http_pool,
        "median_query_rerank_s": (
            round(float(np.median([r["query_rerank_s"] for r in scored
                                   if "query_rerank_s" in r])), 3)
            if any("query_rerank_s" in r for r in scored) else None
        ),
        "mean_dropped_per_q": (
            round(float(np.mean([r["dropped"] for r in scored
                                 if "dropped" in r])), 2)
            if any("dropped" in r for r in scored) else None
        ),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "filter_stats": ({
            "mean_kept_per_q": round(float(np.mean(
                [r["n_kept"] for r in scored if "n_kept" in r])), 2),
            # questions where the filter dropped >=1 gold (recall harm)
            "n_q_gold_dropped": sum(
                1 for r in scored if r.get("gold_dropped", 0) > 0),
            "total_gold_dropped": sum(
                r.get("gold_dropped", 0) for r in scored),
            # questions where >=1 gold survived AND n_kept <= top_k
            # (filter alone already fits the budget -> ranking unnecessary)
            "n_q_kept_le_k": sum(
                1 for r in scored
                if "n_kept" in r and r["n_kept"] <= args.top_k),
            "by_cat_mean_kept": {
                c: round(float(np.mean(
                    [r["n_kept"] for r in scored
                     if "n_kept" in r and r["category"] == c])), 2)
                for c in sorted({r["category"] for r in scored
                                 if "n_kept" in r})
            },
        } if any("n_kept" in r for r in scored) else None),
        "cost": {
            "total_swiss_comparisons": total_swiss_comparisons,
            "comparisons_per_question": round(cmp_per_q, 1),
            "tok_in": total_tok_in, "tok_out": total_tok_out,
            "price_in_per_1m": args.price_in,
            "price_out_per_1m": args.price_out,
            "total_usd": round(cost_usd, 4),
            "usd_per_question": round(cost_per_q, 6),
            "usd_per_1000_questions": round(cost_per_q * 1000, 3),
            "cohere_ref_usd_per_1000q": round(cohere_ref_per_q * 1000, 3),
            "swiss_vs_cohere_cost_multiple": round(
                cost_per_q / cohere_ref_per_q, 1
            ) if cohere_ref_per_q else None,
        },
        "by_arm": {},
    }
    cats_present = sorted({r["category"] for r in scored})
    for arm in arms:
        summary["by_arm"][arm] = {
            "overall": {
                "recall@k": round(agg(arm, "recall@k", None), 4),
                "ndcg@k": round(agg(arm, "ndcg@k", None), 4),
            },
            "by_cat": {
                c: {
                    "recall@k": round(agg(arm, "recall@k", c), 4),
                    "ndcg@k": round(agg(arm, "ndcg@k", c), 4),
                }
                for c in cats_present
            },
        }

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2,
                  default=str)

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
