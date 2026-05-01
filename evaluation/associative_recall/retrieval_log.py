"""Retrieval-log cue generation experiments.

Tests two novel framings of cue generation:

  Framing 1 -- Retrieval log reflection:
    Feed the cue generator a LOG of previous (cue -> new/duplicate segment
    counts) so it can self-correct when earlier cues were wasted.

  Framing 2 -- Challenge framing (the game):
    Reframe cue generation as an extraction challenge with budget/score:
    the model is told each cue costs budget and the objective is to maximize
    unique new relevant segments discovered.

Four variants are implemented:

  A. retrieval_log_v2f    -> log + v2f prompt (2 cues/round, per_cue_k=10,
                             rounds=2, matches v2f_tight_20)
  B. retrieval_log_cot    -> log + CoT 4-step prompt; Steps 1-4 explicitly
                             reason about WHAT PREVIOUS CUES MISSED
                             (5 cues/round, per_cue_k=4, rounds=2, matches CoT)
  C. challenge_framing    -> game/challenge framing + v2f-like output
                             (2-5 cues optimized for info gain)
  D. retrieval_log_challenge -> combines A + C (log + challenge + v2f-like out)

Evaluation: fair K-budget at K=20 and K=50 with cosine backfill, run across
LoCoMo 30q, Synthetic 19q, Puzzle 16q, and Advanced 23q.

Incremental saving: results are flushed to disk after every question so a
crash does not lose work.

Usage:
    uv run python retrieval_log.py [--dataset NAME] [--variant NAME] [--force]
    uv run python retrieval_log.py --summary  # recompute summary from cached
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_FILE_EMB = CACHE_DIR / "retlog_embedding_cache.json"
CACHE_FILE_LLM = CACHE_DIR / "retlog_llm_cache.json"
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Caches (read from all prior caches, write to retlog_* files)
# ---------------------------------------------------------------------------
class RetLogEmbeddingCache(EmbeddingCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for p in sorted(self.cache_dir.glob("*embedding_cache.json")):
            try:
                with open(p) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        self.cache_file = CACHE_FILE_EMB
        self._new: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        existing.update(self._new)
        import os

        tmp = self.cache_file.with_suffix(f".json.tmp.{os.getpid()}")
        try:
            with open(tmp, "w") as f:
                json.dump(existing, f)
            tmp.replace(self.cache_file)
        except FileNotFoundError:
            # Another process beat us to replace; that's OK - our data is in _new
            # and will be re-saved next time.
            if tmp.exists():
                tmp.unlink()
        self._new = {}


class RetLogLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        # Prefer retlog cache first (exact prompt match), then all others.
        for p in sorted(
            self.cache_dir.glob("*llm_cache.json"),
            key=lambda x: 0 if x.name.startswith("retlog") else 1,
        ):
            try:
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v and k not in self._cache:
                        self._cache[k] = v
            except (json.JSONDecodeError, OSError):
                pass
        self.cache_file = CACHE_FILE_LLM
        self._new: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new[key] = response

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        existing.update(self._new)
        import os

        tmp = self.cache_file.with_suffix(f".json.tmp.{os.getpid()}")
        try:
            with open(tmp, "w") as f:
                json.dump(existing, f)
            tmp.replace(self.cache_file)
        except FileNotFoundError:
            if tmp.exists():
                tmp.unlink()
        self._new = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_segments(
    segments: list[Segment], max_items: int = 14, max_chars: int = 260
) -> str:
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in sorted_segs
    )


def _parse_cues(text: str, key: str = "CUE:") -> list[str]:
    out: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith(key.upper()):
            val = line[len(key) :].strip()
            if val:
                out.append(val)
    return out


@dataclass
class CueRecord:
    """Tracks the effectiveness of a single cue in the retrieval log."""

    round: int
    cue: str
    all_retrieved_indices: set[int]
    new_indices: set[int]

    @property
    def n_all(self) -> int:
        return len(self.all_retrieved_indices)

    @property
    def n_new(self) -> int:
        return len(self.new_indices)

    @property
    def duplicate_rate(self) -> float:
        if self.n_all == 0:
            return 0.0
        return (self.n_all - self.n_new) / self.n_all


def _format_retrieval_log(
    initial_n: int,
    records: list[CueRecord],
) -> str:
    """Format the retrieval history as an inspectable log for the prompt."""
    if not records:
        return (
            f"[R0] Question -> {initial_n} segments (starting pool)\n"
            f"(no cues issued yet)"
        )
    lines = [f"[R0] Question -> {initial_n} segments (starting pool)"]
    for rec in records:
        pct = int(round(rec.duplicate_rate * 100))
        cue_short = rec.cue[:110].replace("\n", " ")
        lines.append(
            f'[R{rec.round + 1}] Cue: "{cue_short}" -> '
            f"{rec.n_new} new / {rec.n_all} total ({pct}% duplicates)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt templates -- four variants
# ---------------------------------------------------------------------------

# ======  A. retrieval_log_v2f  ======
# v2f assessment + 2-cue format, augmented with a retrieval log that shows
# previous cue effectiveness (new vs duplicate counts).
RETLOG_V2F_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({num_segs} segments, chronological):
{all_segs}

RETRIEVAL HISTORY:
{retrieval_log}

Reflect on the history above:
- Which cues were EFFECTIVE (low duplicate rate = found new content)?
- Which cues were WASTED (high duplicate rate = re-found the same turns)?
- What directions have NOT been tried yet?

Then briefly assess: how well is this search going given what's been \
retrieved? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Generate EXACTLY 2 search cues that AVOID duplicating previous retrievals \
and explore directions not yet covered. Use specific vocabulary that would \
appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation reflecting on previous cue \
effectiveness>
CUE: <text>
CUE: <text>
Nothing else."""


# ======  B. retrieval_log_cot  ======
# CoT 4-step reasoning, but steps are reframed around WHAT PREVIOUS CUES
# MISSED. Log of prior cues + effectiveness provided.
RETLOG_COT_PROMPT = """\
You are performing semantic retrieval over a conversation history. Cues \
will be embedded and matched via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({num_segs} segments, chronological):
{all_segs}

RETRIEVAL HISTORY:
{retrieval_log}

You have prior cue effectiveness telemetry above. Use it: reason \
explicitly about WHAT PREVIOUS CUES MISSED.

Think step by step:
STEP 1 -- What vocabulary did previous cues ALREADY COVER (and therefore \
hit the same turns)? List the dominant terms from the retrieved segments \
and the cues that were wasted / highly duplicative.
STEP 2 -- What RELATED terminology still has NOT been searched? Aliases, \
codenames, abbreviations, informal references ("the bird", "that thing", \
different role phrasings). Focus on vocabulary that previous cues did \
NOT use.
STEP 3 -- If this question is a CHAIN (A -> B -> C where each link uses \
different vocabulary), which link is still unexplored based on the log? \
Which next link should be targeted to open new territory?
STEP 4 -- What ALTERNATIVE NAMES or phrasings for the target have NOT \
been tried? Include every alias you can justify from the retrieved text \
or reasonable guesses, especially those orthogonal to previous cues.

Then generate up to {num_cues} search cues that EXTEND retrieval into \
genuinely NEW territory. Each cue should target vocabulary / chain links \
/ aliases that previous cues did NOT cover. A cue may be:
  - a short alias/name phrase (1-5 words) that might appear inline
  - a 1-2 sentence plausible conversation snippet targeting the next link

Prefer DIVERSE cues (cover multiple aliases and/or multiple chain links \
not yet tried). Do not rephrase the question. Avoid vocabulary that \
previous duplicative cues already used.

Format:
STEP 1: <what previous cues covered / what was wasted>
STEP 2: <related vocabulary not yet searched>
STEP 3: <next chain link to target>
STEP 4: <alternative names not yet tried>
CUE: <text>
CUE: <text>
(up to {num_cues} cues)
Nothing else."""


# ======  C. challenge_framing  ======
# Game / challenge framing. Budget + score is made explicit. NO retrieval
# log (isolates the effect of framing alone vs Framing 1).
CHALLENGE_PROMPT = """\
*** EXTRACTION CHALLENGE ***

You are extracting hidden facts from a memory system. Your objective is to \
retrieve every conversation turn that contains information relevant to a \
target question.

Rules of the game:
- Each cue you submit COSTS BUDGET.
- Each cue will be embedded and matched via cosine similarity against \
stored conversation turns. It will return a fixed number of turns.
- The system REWARDS cues that find turns NO PREVIOUS CUE FOUND.
- The system PENALIZES cues that re-find turns already returned.
- Your SCORE = count of UNIQUE relevant turns your cues discovered.

To maximize score: make every cue count. Each cue must target content \
that is plausibly in the conversation AND that previous cues have not \
already covered.

Question: {question}

ALREADY RETRIEVED ({num_segs} segments, chronological):
{all_segs}

ALREADY SEARCHED FOR (do NOT repeat these):
{explored}

Now submit 2-5 cues optimized for INFORMATION GAIN. Each cue must plausibly \
find turns that the already-retrieved segments do NOT cover. Use specific \
vocabulary that would appear inline in a chat message.

Hard rules:
- Do NOT rephrase the question.
- Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.
- Prefer DIVERSE cues (different angles, different vocabulary, different \
aliases).
- If the question implies MULTIPLE items ("all / every / list"), aim for \
cues that together cover the full set.

Format:
STRATEGY: <1-2 sentences on how your cues maximize unique new turns>
CUE: <text>
CUE: <text>
(up to 5 cues)
Nothing else."""


# ======  D. retrieval_log_challenge  ======
# Combines challenge framing with the retrieval log telemetry.
RETLOG_CHALLENGE_PROMPT = """\
*** EXTRACTION CHALLENGE ***

You are extracting hidden facts from a memory system. Your objective is to \
retrieve every conversation turn that contains information relevant to a \
target question.

Rules of the game:
- Each cue you submit COSTS BUDGET.
- Each cue will be embedded and matched via cosine similarity against \
stored conversation turns. It will return a fixed number of turns.
- The system REWARDS cues that find turns NO PREVIOUS CUE FOUND.
- The system PENALIZES cues that re-find turns already returned.
- Your SCORE = count of UNIQUE relevant turns your cues discovered.

You are being given a LIVE TELEMETRY LOG of previous cues and how many \
NEW vs DUPLICATE turns each found. Use it to self-correct.

Question: {question}

ALREADY RETRIEVED ({num_segs} segments, chronological):
{all_segs}

RETRIEVAL HISTORY (telemetry):
{retrieval_log}

Reflect on the telemetry:
- Which previous cues scored well (high NEW)?
- Which previous cues wasted budget (high DUPLICATES)?
- What search directions remain completely unexplored?

Now submit 2-5 NEW cues optimized for INFORMATION GAIN. Each cue must \
plausibly find turns that previous cues have NOT already returned. Do not \
re-use vocabulary from duplicative previous cues.

Hard rules:
- Do NOT rephrase the question.
- Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.
- Prefer DIVERSE cues (different angles, different vocabulary, different \
aliases).
- If the question implies MULTIPLE items ("all / every / list"), aim for \
cues that together cover the full set.

Format:
STRATEGY: <1-2 sentences on how your cues avoid duplicates and maximize \
unique new turns>
CUE: <text>
CUE: <text>
(up to 5 cues)
Nothing else."""


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------
@dataclass
class VariantConfig:
    name: str
    prompt: str
    uses_retrieval_log: bool
    initial_k: int
    max_cues: int  # cap per round
    per_cue_k: int
    rounds: int


VARIANTS: dict[str, VariantConfig] = {
    # A: log + v2f format. Budget: 10 initial + 2 cues x 10 x 2 rounds.
    "retrieval_log_v2f": VariantConfig(
        name="retrieval_log_v2f",
        prompt=RETLOG_V2F_PROMPT,
        uses_retrieval_log=True,
        initial_k=10,
        max_cues=2,
        per_cue_k=10,
        rounds=2,
    ),
    # B: log + CoT. Budget: 10 initial + 5 cues x 4 x 2 rounds.
    "retrieval_log_cot": VariantConfig(
        name="retrieval_log_cot",
        prompt=RETLOG_COT_PROMPT,
        uses_retrieval_log=True,
        initial_k=10,
        max_cues=5,
        per_cue_k=4,
        rounds=2,
    ),
    # C: challenge framing, NO log. Match v2f budget.
    "challenge_framing": VariantConfig(
        name="challenge_framing",
        prompt=CHALLENGE_PROMPT,
        uses_retrieval_log=False,
        initial_k=10,
        max_cues=5,
        per_cue_k=4,
        rounds=2,
    ),
    # D: log + challenge framing. Match v2f budget.
    "retrieval_log_challenge": VariantConfig(
        name="retrieval_log_challenge",
        prompt=RETLOG_CHALLENGE_PROMPT,
        uses_retrieval_log=True,
        initial_k=10,
        max_cues=5,
        per_cue_k=4,
        rounds=2,
    ),
}


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
@dataclass
class RetLogResult:
    segments: list[Segment]
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


class RetrievalLogArch:
    def __init__(
        self,
        store: SegmentStore,
        config: VariantConfig,
        client: OpenAI | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = RetLogEmbeddingCache()
        self.llm_cache = RetLogLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, embedding)
        self.embed_calls += 1
        return embedding

    def llm_call(self, prompt: str, model: str = MODEL) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(model, prompt, text)
        self.llm_calls += 1
        return text

    def retrieve(self, question: str, conversation_id: str) -> RetLogResult:
        cfg = self.config
        exclude: set[int] = set()
        all_segs: list[Segment] = []
        explored: list[str] = []
        records: list[CueRecord] = []
        round_log: list[dict] = []

        # Initial retrieval
        q_emb = self.embed_text(question)
        r0 = self.store.search(
            q_emb, top_k=cfg.initial_k, conversation_id=conversation_id
        )
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)
        initial_n = len(all_segs)

        for round_i in range(cfg.rounds):
            # Build prompt. Each variant's prompt string accepts the format
            # fields it actually uses; we pass all relevant fields.
            fmt_kwargs = {
                "question": question,
                "all_segs": _format_segments(all_segs, max_items=14),
                "num_segs": len(all_segs),
                "explored": (
                    "\n".join(f"- {c}" for c in explored) if explored else "(none yet)"
                ),
                "retrieval_log": _format_retrieval_log(initial_n, records),
                "num_cues": cfg.max_cues,
            }
            # Each prompt template only references a subset of these fields,
            # so we filter by .format() compatibility by feeding the full dict.
            try:
                prompt = cfg.prompt.format(**fmt_kwargs)
            except KeyError as e:
                raise RuntimeError(f"Prompt for variant {cfg.name} missing field: {e}")

            response = self.llm_call(prompt)
            cues = _parse_cues(response, "CUE:")[: cfg.max_cues]

            # Pull assessment/strategy line for logging
            meta_line = ""
            for line in response.strip().split("\n"):
                ls = line.strip()
                upper = ls.upper()
                if (
                    upper.startswith("ASSESSMENT:")
                    or upper.startswith("STRATEGY:")
                    or upper.startswith("REASON:")
                ):
                    meta_line = ls
                    break

            round_log.append(
                {
                    "round": round_i,
                    "meta": meta_line,
                    "cues": cues,
                }
            )
            if not cues:
                break

            for cue in cues:
                if cue in explored:
                    continue
                explored.append(cue)
                cue_emb = self.embed_text(cue)
                # Fetch WITHOUT excluding; we want to measure duplicates.
                result = self.store.search(
                    cue_emb,
                    top_k=cfg.per_cue_k,
                    conversation_id=conversation_id,
                    exclude_indices=None,
                )
                all_idx_set = {s.index for s in result.segments}
                new_idx_set = all_idx_set - exclude
                records.append(
                    CueRecord(
                        round=round_i,
                        cue=cue,
                        all_retrieved_indices=all_idx_set,
                        new_indices=new_idx_set,
                    )
                )
                # Update pool with only the NEW segments
                for s in result.segments:
                    if s.index not in exclude:
                        all_segs.append(s)
                        exclude.add(s.index)

        return RetLogResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "variant": cfg.name,
                "rounds": round_log,
                "total_segments": len(all_segs),
                "cue_records": [
                    {
                        "round": rec.round,
                        "cue": rec.cue,
                        "n_all": rec.n_all,
                        "n_new": rec.n_new,
                        "duplicate_rate": round(rec.duplicate_rate, 3),
                    }
                    for rec in records
                ],
            },
        )


# ---------------------------------------------------------------------------
# Fair K-budget evaluation (mirrors cot_universal.py / self_dispatch_v2.py)
# ---------------------------------------------------------------------------
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(arch: RetrievalLogArch, question: dict, verbose: bool = False) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for s in result.segments:
        if s.index not in seen:
            arch_segments.append(s)
            seen.add(s.index)

    q_emb = arch.embed_text(q_text)
    max_b = max(BUDGETS)
    baseline = arch.store.search(q_emb, top_k=max_b, conversation_id=conv_id)

    arch_idx = {s.index for s in arch_segments}
    backfilled = list(arch_segments) + [
        s for s in baseline.segments if s.index not in arch_idx
    ]

    recalls: dict[str, float] = {}
    baseline_recalls: dict[str, float] = {}
    for K in BUDGETS:
        a_ids = {s.turn_id for s in backfilled[:K]}
        b_ids = {s.turn_id for s in baseline.segments[:K]}
        recalls[f"r@{K}"] = compute_recall(a_ids, source_ids)
        baseline_recalls[f"r@{K}"] = compute_recall(b_ids, source_ids)

    cue_records = result.metadata.get("cue_records", [])
    avg_dup_rate = (
        sum(c["duplicate_rate"] for c in cue_records) / len(cue_records)
        if cue_records
        else 0.0
    )

    row = {
        "variant": arch.config.name,
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question.get("question_index"),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "pool_size": len(arch_segments),
        "baseline_recalls": baseline_recalls,
        "retlog_recalls": recalls,
        "avg_cue_duplicate_rate": round(avg_dup_rate, 3),
        "cue_records": cue_records,
        "rounds_metadata": result.metadata["rounds"],
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
    }
    if verbose:
        print(
            f"    [{arch.config.name}] pool={len(arch_segments)} "
            f"dup_rate={avg_dup_rate:.2f} "
            f"r@20: base={baseline_recalls['r@20']:.3f} "
            f"ret={recalls['r@20']:.3f}  "
            f"r@50: base={baseline_recalls['r@50']:.3f} "
            f"ret={recalls['r@50']:.3f}  "
            f"emb={arch.embed_calls} llm={arch.llm_calls}"
        )
    return row


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS = {
    "locomo_30q": {
        "questions_file": "questions_extended.json",
        "segments_npz": "segments_extended.npz",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "limit": 30,
    },
    "synthetic_19q": {
        "questions_file": "questions_synthetic.json",
        "segments_npz": "segments_synthetic.npz",
        "filter": lambda q: True,
        "limit": None,
    },
    "puzzle_16q": {
        "questions_file": "questions_puzzle.json",
        "segments_npz": "segments_puzzle.npz",
        "filter": lambda q: True,
        "limit": None,
    },
    "advanced_23q": {
        "questions_file": "questions_advanced.json",
        "segments_npz": "segments_advanced.npz",
        "filter": lambda q: True,
        "limit": None,
    },
}


def load_dataset(key: str) -> tuple[list[dict], SegmentStore]:
    meta = DATASETS[key]
    with open(DATA_DIR / meta["questions_file"]) as f:
        qs = json.load(f)
    qs = [q for q in qs if meta["filter"](q)]
    if meta["limit"] is not None:
        qs = qs[: meta["limit"]]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=meta["segments_npz"])
    return qs, store


# ---------------------------------------------------------------------------
# Loading prior baselines for comparison
# ---------------------------------------------------------------------------
def load_budget_recall_by_qkey(arch_name: str, dataset_key: str) -> dict[tuple, float]:
    path = RESULTS_DIR / f"budget_{arch_name}_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        payload = json.load(f)
    out: dict[tuple, float] = {}
    for r in payload.get("results", []):
        key = (r["conversation_id"], r.get("question_index"))
        out[key] = r["recall"]
    return out


def load_cot_recall_by_qkey(dataset_key: str) -> dict[tuple, dict[int, float]]:
    path = RESULTS_DIR / f"cot_chain_of_thought_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    out: dict[tuple, dict[int, float]] = {}
    for r in rows:
        key = (r["conversation_id"], r.get("question_index"))
        out[key] = {
            20: r["cot_recalls"]["r@20"],
            50: r["cot_recalls"]["r@50"],
        }
    return out


def load_self_v2_recall_by_qkey(
    dataset_key: str,
) -> dict[tuple, dict[int, float]]:
    path = RESULTS_DIR / f"self_v2_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    out: dict[tuple, dict[int, float]] = {}
    for r in rows:
        key = (r["conversation_id"], r.get("question_index"))
        out[key] = {
            20: r["self_v2_recalls"]["r@20"],
            50: r["self_v2_recalls"]["r@50"],
        }
    return out


# ---------------------------------------------------------------------------
# Runner with incremental save-per-question
# ---------------------------------------------------------------------------
def run_variant_on_dataset(
    variant_name: str,
    dataset_key: str,
    force: bool = False,
    verbose: bool = False,
) -> list[dict]:
    cfg = VARIANTS[variant_name]
    result_file = RESULTS_DIR / f"retlog_{variant_name}_{dataset_key}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if result_file.exists() and not force:
        with open(result_file) as f:
            existing = json.load(f)

    done_keys = {(r["conversation_id"], r.get("question_index")) for r in existing}

    qs, store = load_dataset(dataset_key)
    to_run = [
        q
        for q in qs
        if (q["conversation_id"], q.get("question_index")) not in done_keys
    ]
    if not to_run:
        print(f">>> [{variant_name}] {dataset_key}: all {len(qs)} done.")
        return existing

    arch = RetrievalLogArch(store, cfg)
    print(
        f"\n>>> [{variant_name}] on {dataset_key}: "
        f"{len(to_run)} new / {len(qs)} total (skipping {len(done_keys)}), "
        f"{len(store.segments)} segments"
    )
    rows: list[dict] = list(existing)
    for i, q in enumerate(to_run):
        q_short = q["question"][:60].replace("\n", " ")
        print(
            f"  [{i + 1}/{len(to_run)}] {q['category']}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_one(arch, q, verbose=verbose)
            rows.append(row)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback

            traceback.print_exc()
            # Save what we have so far before moving on
            with open(result_file, "w") as f:
                json.dump(rows, f, indent=2, default=str)
            arch.save_caches()
            continue
        # Incremental flush after EVERY question so a crash doesn't lose work
        with open(result_file, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        if (i + 1) % 3 == 0:
            arch.save_caches()
        sys.stdout.flush()

    arch.save_caches()
    with open(result_file, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"  Saved -> {result_file}")
    return rows


# ---------------------------------------------------------------------------
# Per-category summary
# ---------------------------------------------------------------------------
def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def compare_per_category(
    dataset_key: str,
    variant_rows_by_name: dict[str, list[dict]],
) -> list[dict]:
    """For each (variant, K, category), compute mean recall + deltas vs
    baseline / v2f / cot / self_v2.
    """
    budget_by_arch: dict[str, dict[int, dict]] = {}
    for arch in ("baseline", "v2f_tight"):
        budget_by_arch[arch] = {
            K: load_budget_recall_by_qkey(f"{arch}_{K}", dataset_key) for K in BUDGETS
        }
    cot_by_q = load_cot_recall_by_qkey(dataset_key)
    sv2_by_q = load_self_v2_recall_by_qkey(dataset_key)

    out: list[dict] = []
    for variant, rows in variant_rows_by_name.items():
        rows_by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            rows_by_cat[r["category"]].append(r)
        for K in BUDGETS:
            for cat, cat_rows in sorted(rows_by_cat.items()):
                n = len(cat_rows)
                if n == 0:
                    continue
                var_recalls = [r["retlog_recalls"][f"r@{K}"] for r in cat_rows]
                var_mean = sum(var_recalls) / n
                avg_dup = (
                    sum(r.get("avg_cue_duplicate_rate", 0.0) for r in cat_rows) / n
                )

                b_vals, v2f_vals, cot_vals, sv2_vals = [], [], [], []
                for r in cat_rows:
                    key = (r["conversation_id"], r["question_index"])
                    if key in budget_by_arch["baseline"][K]:
                        b_vals.append(budget_by_arch["baseline"][K][key])
                    if key in budget_by_arch["v2f_tight"][K]:
                        v2f_vals.append(budget_by_arch["v2f_tight"][K][key])
                    if key in cot_by_q:
                        cot_vals.append(cot_by_q[key][K])
                    if key in sv2_by_q:
                        sv2_vals.append(sv2_by_q[key][K])

                b_mean = _mean(b_vals)
                v2f_mean = _mean(v2f_vals)
                cot_mean = _mean(cot_vals)
                sv2_mean = _mean(sv2_vals)

                out.append(
                    {
                        "variant": variant,
                        "dataset": dataset_key,
                        "category": cat,
                        "K": K,
                        "n": n,
                        "baseline": b_mean,
                        "v2f": v2f_mean,
                        "cot": cot_mean,
                        "self_v2": sv2_mean,
                        variant: var_mean,
                        "mean": var_mean,
                        "avg_cue_duplicate_rate": round(avg_dup, 3),
                        "vs_v2f": (
                            var_mean - v2f_mean if v2f_mean is not None else None
                        ),
                        "vs_cot": (
                            var_mean - cot_mean if cot_mean is not None else None
                        ),
                        "vs_self_v2": (
                            var_mean - sv2_mean if sv2_mean is not None else None
                        ),
                        "vs_baseline": (
                            var_mean - b_mean if b_mean is not None else None
                        ),
                    }
                )
    return out


def fmt_cell(val: float | None, plus_sign: bool = False) -> str:
    if val is None:
        return "    -"
    s = f"{val:+.3f}" if plus_sign else f"{val:.3f}"
    return f"{s:>6s}"


def print_overall_by_dataset(all_rows: list[dict], K: int) -> None:
    rows_k = [r for r in all_rows if r["K"] == K]
    if not rows_k:
        return
    print(f"\n{'-' * 140}")
    print(f"OVERALL per (variant, dataset) at K={K}")
    print(f"{'-' * 140}")
    hdr = (
        f"{'Variant':<24s} {'Dataset':<14s} {'n':>3s} "
        f"{'Base':>6s} {'v2f':>6s} {'CoT':>6s} {'SV2':>6s} {'Mean':>6s}  "
        f"{'vs_v2f':>7s} {'vs_CoT':>7s} {'vs_SV2':>7s} {'vs_base':>7s}  "
        f"{'dup':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    by_vd: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows_k:
        by_vd[(r["variant"], r["dataset"])].append(r)
    last_variant = None
    for (variant, ds), rows in sorted(by_vd.items()):
        if last_variant is not None and variant != last_variant:
            print()
        last_variant = variant
        total_n = sum(r["n"] for r in rows)

        def _weighted(key: str) -> float | None:
            vals = [(r[key], r["n"]) for r in rows if r.get(key) is not None]
            if not vals:
                return None
            tot = sum(n for _, n in vals)
            return sum(v * n for v, n in vals) / tot

        b = _weighted("baseline")
        v2f = _weighted("v2f")
        cot = _weighted("cot")
        sv2 = _weighted("self_v2")
        mean = _weighted("mean")
        dup = _weighted("avg_cue_duplicate_rate")
        vv = (mean - v2f) if (mean is not None and v2f is not None) else None
        vc = (mean - cot) if (mean is not None and cot is not None) else None
        vs = (mean - sv2) if (mean is not None and sv2 is not None) else None
        vb = (mean - b) if (mean is not None and b is not None) else None
        print(
            f"{variant:<24s} {ds:<14s} {total_n:>3d} "
            f"{fmt_cell(b)} {fmt_cell(v2f)} {fmt_cell(cot)} "
            f"{fmt_cell(sv2)} {fmt_cell(mean)}  "
            f"{fmt_cell(vv, True)} {fmt_cell(vc, True)} "
            f"{fmt_cell(vs, True)} {fmt_cell(vb, True)}  "
            f"{dup:>5.2f}"
            if dup is not None
            else f"{variant:<24s} {ds:<14s} {total_n:>3d} "
            f"{fmt_cell(b)} {fmt_cell(v2f)} {fmt_cell(cot)} "
            f"{fmt_cell(sv2)} {fmt_cell(mean)}  "
            f"{fmt_cell(vv, True)} {fmt_cell(vc, True)} "
            f"{fmt_cell(vs, True)} {fmt_cell(vb, True)}  "
            f"    -"
        )


def print_variant_ranking(all_rows: list[dict], K: int) -> None:
    """Aggregate each variant across all datasets -- which variant wins?"""
    rows_k = [r for r in all_rows if r["K"] == K]
    if not rows_k:
        return
    print(f"\n{'=' * 100}")
    print(f"VARIANT RANKING at K={K} (weighted across all datasets)")
    print(f"{'=' * 100}")

    by_var: dict[str, list[dict]] = defaultdict(list)
    for r in rows_k:
        by_var[r["variant"]].append(r)

    hdr = (
        f"{'Variant':<24s} {'total_n':>7s} "
        f"{'Base':>6s} {'v2f':>6s} {'CoT':>6s} {'SV2':>6s} {'Mean':>6s}  "
        f"{'vs_v2f':>7s} {'vs_CoT':>7s} {'vs_SV2':>7s}  "
        f"{'dup':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    variant_scores: list[tuple[str, float]] = []
    for variant, rows in sorted(by_var.items()):
        total_n = sum(r["n"] for r in rows)

        def _weighted(key: str) -> float | None:
            vals = [(r[key], r["n"]) for r in rows if r.get(key) is not None]
            if not vals:
                return None
            tot = sum(n for _, n in vals)
            return sum(v * n for v, n in vals) / tot

        b = _weighted("baseline")
        v2f = _weighted("v2f")
        cot = _weighted("cot")
        sv2 = _weighted("self_v2")
        mean = _weighted("mean")
        dup = _weighted("avg_cue_duplicate_rate")
        vv = (mean - v2f) if (mean is not None and v2f is not None) else None
        vc = (mean - cot) if (mean is not None and cot is not None) else None
        vs = (mean - sv2) if (mean is not None and sv2 is not None) else None
        variant_scores.append((variant, mean if mean is not None else 0.0))
        dup_str = f"{dup:>5.2f}" if dup is not None else "    -"
        print(
            f"{variant:<24s} {total_n:>7d} "
            f"{fmt_cell(b)} {fmt_cell(v2f)} {fmt_cell(cot)} "
            f"{fmt_cell(sv2)} {fmt_cell(mean)}  "
            f"{fmt_cell(vv, True)} {fmt_cell(vc, True)} "
            f"{fmt_cell(vs, True)}  "
            f"{dup_str}"
        )

    variant_scores.sort(key=lambda x: -x[1])
    print(
        f"\n  Ranking (by weighted mean recall): "
        f"{', '.join(f'{v}={s:.3f}' for v, s in variant_scores)}"
    )


HARDEST_CATEGORIES = [
    "completeness",
    "logic_constraint",
    "procedural",
    "conjunction",
    "inference",
    "proactive",
    "evolving_terminology",
    "constraint_propagation",
    "sequential_chain",
    "absence_inference",
    "contradiction",
    "negation",
    "quantitative_aggregation",
    "unfinished_business",
    "locomo_single_hop",
]


def print_per_category_table(
    all_rows: list[dict],
    K: int,
    focus_categories: list[str] | None = None,
) -> None:
    rows = [r for r in all_rows if r["K"] == K]
    if focus_categories is not None:
        rows = [r for r in rows if r["category"] in focus_categories]
    if not rows:
        return
    title = "PER-CATEGORY (hardest)" if focus_categories else "PER-CATEGORY"
    print(f"\n{'=' * 148}")
    print(f"{title} at K={K}")
    print(f"{'=' * 148}")
    hdr = (
        f"{'Variant':<24s} {'Dataset':<14s} {'Category':<24s} {'n':>3s} "
        f"{'Base':>6s} {'v2f':>6s} {'CoT':>6s} {'SV2':>6s} {'Mean':>6s}  "
        f"{'vs_v2f':>7s} {'vs_CoT':>7s} {'vs_SV2':>7s} {'vs_base':>7s}  "
        f"{'dup':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    last_variant = None
    for r in sorted(rows, key=lambda x: (x["variant"], x["dataset"], x["category"])):
        if last_variant is not None and r["variant"] != last_variant:
            print()
        last_variant = r["variant"]
        dup = r.get("avg_cue_duplicate_rate")
        dup_str = f"{dup:>5.2f}" if dup is not None else "    -"
        print(
            f"{r['variant']:<24s} {r['dataset']:<14s} "
            f"{r['category']:<24s} {r['n']:>3d} "
            f"{fmt_cell(r['baseline'])} {fmt_cell(r['v2f'])} "
            f"{fmt_cell(r['cot'])} {fmt_cell(r['self_v2'])} "
            f"{fmt_cell(r['mean'])}  "
            f"{fmt_cell(r['vs_v2f'], True)} "
            f"{fmt_cell(r['vs_cot'], True)} "
            f"{fmt_cell(r['vs_self_v2'], True)} "
            f"{fmt_cell(r['vs_baseline'], True)}  "
            f"{dup_str}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Rerun even if result file exists"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASETS.keys()),
        help="Restrict to a single dataset",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=list(VARIANTS.keys()),
        help="Restrict to a single variant",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Recompute summary from existing result files, no new LLM calls",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_keys = [args.dataset] if args.dataset else list(DATASETS.keys())
    variant_keys = [args.variant] if args.variant else list(VARIANTS.keys())

    all_rows: list[dict] = []
    for ds in dataset_keys:
        variant_rows_by_name: dict[str, list[dict]] = {}
        for v in variant_keys:
            if args.summary:
                result_file = RESULTS_DIR / f"retlog_{v}_{ds}.json"
                if result_file.exists():
                    with open(result_file) as f:
                        variant_rows_by_name[v] = json.load(f)
                else:
                    variant_rows_by_name[v] = []
            else:
                variant_rows_by_name[v] = run_variant_on_dataset(
                    v, ds, force=args.force, verbose=args.verbose
                )
        all_rows.extend(compare_per_category(ds, variant_rows_by_name))

    # Save summary
    out_path = RESULTS_DIR / "retlog_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved summary -> {out_path}")

    for K in BUDGETS:
        print_variant_ranking(all_rows, K)
        print_overall_by_dataset(all_rows, K)

    for K in BUDGETS:
        print_per_category_table(all_rows, K, focus_categories=HARDEST_CATEGORIES)


if __name__ == "__main__":
    main()
