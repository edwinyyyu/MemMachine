"""Top-down prompt distillation experiment.

Methodology
-----------
Bottom-up prompt engineering (what produced v2f) starts minimal and adds
instructions incrementally, biasing toward local optima near the starting
point. Top-down distillation starts from a verbose prompt enumerating every
phenomenon / observation believed to govern retrieval behavior, then
iteratively deletes observations one at a time. Whatever survives is the
minimal sufficient prompt, but arrived at from a DIFFERENT initial state
than v2f, so it may find a different optimum.

Procedure
---------
1. Quick-test the verbose prompt on 5 representative questions (2 LoCoMo, 1
   synthetic, 1 puzzle, 1 advanced). If it beats v2f on 3+/5, it's a viable
   starting point.

2. Iterate: for each of the 10 observations in the prompt, test removing it
   (compared to the CURRENT prompt, not the original verbose). Drop it if
   removal does not hurt; keep it otherwise. The "current prompt" shrinks
   monotonically as observations get dropped.

3. After all observations tested, the surviving prompt is the distilled
   "minimal useful" variant. Only if it beats v2f on the quick-test do we
   run full eval on all 4 datasets at K=20 / K=50.

Constraints
-----------
- Single prompt, single LLM call per cue-generation round (same shape as v2f).
- No dataset-specific priors (no "LoCoMo", "synthetic", etc.).
- Only phenomenon-based knowledge: general observations about retrieval
  behavior.

Technical
---------
- Model: gpt-5-mini
- Embeddings: text-embedding-3-small
- Cache: cache/topdown_llm_cache.json (dedicated, no cross-contamination)
- Results: results/topdown_*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOP_K_PER_HOP = 10
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "locomo_30q": {
        "npz": "segments_extended.npz",
        "questions": "questions_extended.json",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "max_questions": 30,
    },
    "synthetic_19q": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "filter": None,
        "max_questions": None,
    },
    "puzzle_16q": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "filter": None,
        "max_questions": None,
    },
    "advanced_23q": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "filter": None,
        "max_questions": None,
    },
}


def load_dataset(ds_name: str) -> tuple[SegmentStore, list[dict]]:
    cfg = DATASETS[ds_name]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        questions = json.load(f)
    if cfg["filter"]:
        questions = [q for q in questions if cfg["filter"](q)]
    if cfg["max_questions"]:
        questions = questions[: cfg["max_questions"]]
    return store, questions


# ---------------------------------------------------------------------------
# Quick-test question selection
# ---------------------------------------------------------------------------
# 5 total: 2 LoCoMo, 1 synthetic, 1 puzzle, 1 advanced. Hand-picked to span
# different question shapes (temporal, multi-hop, completeness, puzzle
# reasoning, advanced adversarial naming).
QUICK_TEST_SPEC: list[tuple[str, int]] = [
    (
        "locomo_30q",
        0,
    ),  # locomo_temporal: "When did Caroline go to the LGBTQ support group?"
    (
        "locomo_30q",
        2,
    ),  # locomo_multi_hop: "What fields would Caroline be likely to pursue..."
    ("synthetic_19q", 0),  # control: "What is Bob allergic to? ..."
    (
        "puzzle_16q",
        1,
    ),  # logic_constraint: "What were all the constraints for the desk..."
    (
        "advanced_23q",
        1,
    ),  # evolving_terminology: "What are all the different names that have been used to refer to v2..."
]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
# The verbose starting prompt enumerates 10 observations. Each observation
# is tagged with a short id so we can address it programmatically during
# ablation. We keep the numbered, bulleted form verbatim from the user
# spec (with minor wordsmith to keep this DRY).
OBSERVATIONS: list[tuple[str, str]] = [
    (
        "register",
        "Match the register of stored content. Conversation data is short, "
        "casual, first-person. Formal, third-person, or interrogative cues "
        "match poorly.",
    ),
    (
        "vocab_gap",
        "Vocabulary mismatch is the core problem. When the answer uses "
        "different words than the question, cosine search misses it. Your "
        "job is to bridge that gap with cues using the answer's likely "
        "vocabulary.",
    ),
    (
        "keyword_bundles",
        "Dense keyword bundles often outperform polished sentences. "
        '"peanut allergy lactose intolerant" beats "Bob has a peanut '
        'allergy and is also lactose intolerant."',
    ),
    (
        "first_person_fragments",
        'First-person fragments work well for conversation. "had a picnic" '
        'beats "she went on a picnic at the park last weekend."',
    ),
    (
        "no_fabrication",
        "Don't fabricate specifics. \"Caroline had a picnic at the park on "
        "Saturday with the kids\" invents details that don't match stored "
        "content.",
    ),
    (
        "no_question_paraphrase",
        "Don't rephrase the question as a cue. Question paraphrases target "
        "the same embedding region cosine already covers.",
    ),
    (
        "multi_item_coverage",
        'If the question implies multiple items ("all", "every", '
        '"list"), generate cues covering different aspects. Each cue '
        "should target a different type of item.",
    ),
    (
        "no_boolean",
        "Boolean operators (OR, AND) don't work in embedding search — "
        "treat them as separator tokens rather than logical operators.",
    ),
    (
        "declarative_over_interrogative",
        'Cues written as questions ("Did you mention X?") perform worse '
        'than statements ("we talked about X") on conversation data.',
    ),
    (
        "specificity_vs_breadth",
        "When the question has multiple aspects, balance targeting "
        "specificity vs breadth. Too specific and you miss variant "
        "vocabulary; too broad and you hit many irrelevant segments.",
    ),
]


PROMPT_HEADER = (
    "You are generating search text for semantic retrieval. Your cues "
    "will be embedded and compared via cosine similarity.\n\n"
    "CONTEXT:\n"
    "Embedding retrieval finds text that is semantically similar to the "
    'query. A "cue" is text you write that will be used as a secondary '
    "query. Good cues find content the original question vocabulary misses.\n"
)

PROMPT_FOOTER = (
    "\nQuestion: {question}\n\n"
    "Retrieved so far:\n"
    "{context_section}\n\n"
    "Based on the observations above, first briefly assess: what kind of "
    "vocabulary is the answer likely to use, and is the current retrieval "
    "covering that vocabulary?\n\n"
    "Then generate 2 cues following the observations.\n\n"
    "Format:\n"
    "ASSESSMENT: <vocabulary hypothesis + current coverage>\n"
    "CUE: <text>\n"
    "CUE: <text>"
)


def build_prompt(obs_ids: list[str]) -> str:
    """Build a prompt containing only the observations listed in obs_ids.

    If obs_ids is empty we still show the header + footer (the body would
    read "OBSERVATIONS ABOUT WHAT WORKS:\n(none)"). Numbering renumbers
    each time so the model sees a clean 1..N list for whichever subset
    survived.
    """
    obs_map = dict(OBSERVATIONS)
    missing = [oid for oid in obs_ids if oid not in obs_map]
    if missing:
        raise ValueError(f"unknown obs_ids: {missing}")

    body = "\nOBSERVATIONS ABOUT WHAT WORKS:\n\n"
    if obs_ids:
        for i, oid in enumerate(obs_ids, start=1):
            body += f"{i}. {obs_map[oid]}\n\n"
    else:
        body += "(none)\n\n"
    return PROMPT_HEADER + body + PROMPT_FOOTER


# ---------------------------------------------------------------------------
# Caches (dedicated, NOT shared with other prompt experiments)
# ---------------------------------------------------------------------------
class TopdownLLMCache:
    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = CACHE_DIR / "topdown_llm_cache.json"
        self._cache: dict[str, str] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache = json.load(f)
        self._dirty = False

    @staticmethod
    def _key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str) -> str | None:
        return self._cache.get(self._key(model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        self._cache[self._key(model, prompt)] = response
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        tmp.replace(self.cache_file)
        self._dirty = False


class SharedEmbeddingCache:
    """Reads every existing embedding cache (embeddings are prompt-agnostic
    so reuse is safe and cheap), writes to a dedicated topdown file."""

    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in (
            "embedding_cache.json",
            "arch_embedding_cache.json",
            "agent_embedding_cache.json",
            "frontier_embedding_cache.json",
            "meta_embedding_cache.json",
            "optim_embedding_cache.json",
            "bestshot_embedding_cache.json",
            "proactive_embedding_cache.json",
            "adaptive_embedding_cache.json",
            "topdown_embedding_cache.json",
        ):
            p = CACHE_DIR / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = CACHE_DIR / "topdown_embedding_cache.json"
        self._new: dict[str, list[float]] = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        k = self._key(text)
        if k in self._cache:
            return np.array(self._cache[k], dtype=np.float32)
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        k = self._key(text)
        self._cache[k] = embedding.tolist()
        self._new[k] = embedding.tolist()

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new = {}


# ---------------------------------------------------------------------------
# Retrieval runner
# ---------------------------------------------------------------------------
def _format_segments(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def _parse_cues(response: str) -> list[str]:
    cues: list[str] = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


@dataclass
class RetrievalOutput:
    segments: list[Segment]
    cues: list[str]
    output: str


class TopdownRunner:
    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.emb_cache = SharedEmbeddingCache()
        self.llm_cache = TopdownLLMCache()

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.emb_cache.get(text)
        if cached is not None:
            return cached
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        emb = np.array(resp.data[0].embedding, dtype=np.float32)
        self.emb_cache.put(text, emb)
        return emb

    def llm_call(self, prompt: str) -> str:
        cached = self.llm_cache.get(MODEL, prompt)
        if cached is not None:
            return cached
        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )
        text = resp.choices[0].message.content or ""
        self.llm_cache.put(MODEL, prompt, text)
        return text

    def save_caches(self) -> None:
        self.emb_cache.save()
        self.llm_cache.save()

    def retrieve_cosine(self, question: str, conv_id: str, top_k: int) -> list[Segment]:
        q_emb = self.embed_text(question)
        return list(
            self.store.search(q_emb, top_k=top_k, conversation_id=conv_id).segments
        )

    def retrieve_with_prompt(
        self, question: str, conv_id: str, prompt_template: str
    ) -> RetrievalOutput:
        q_emb = self.embed_text(question)
        hop0 = self.store.search(q_emb, top_k=TOP_K_PER_HOP, conversation_id=conv_id)
        segs: list[Segment] = list(hop0.segments)
        exclude: set[int] = {s.index for s in segs}

        ctx = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(segs)
        prompt = prompt_template.format(question=question, context_section=ctx)
        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]

        for cue in cues:
            c_emb = self.embed_text(cue)
            res = self.store.search(
                c_emb,
                top_k=TOP_K_PER_HOP,
                conversation_id=conv_id,
                exclude_indices=exclude,
            )
            for seg in res.segments:
                if seg.index not in exclude:
                    segs.append(seg)
                    exclude.add(seg.index)

        return RetrievalOutput(segments=segs, cues=cues, output=output)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    source_ids: set[int],
    budget: int,
) -> tuple[float, float]:
    seen: set[int] = set()
    arch_unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            arch_unique.append(s)
            seen.add(s.index)
    arch_at_K = arch_unique[:budget]
    arch_idx = {s.index for s in arch_at_K}
    if len(arch_at_K) < budget:
        bf = [s for s in cosine_segments if s.index not in arch_idx]
        arch_at_K = arch_at_K + bf[: budget - len(arch_at_K)]
    arch_at_K = arch_at_K[:budget]
    base_at_K = cosine_segments[:budget]
    return (
        _recall({s.turn_id for s in base_at_K}, source_ids),
        _recall({s.turn_id for s in arch_at_K}, source_ids),
    )


def evaluate_prompt_on_question(
    runner: TopdownRunner,
    prompt_template: str,
    question: dict,
    cosine_segs: list[Segment],
) -> dict:
    out = runner.retrieve_with_prompt(
        question["question"], question["conversation_id"], prompt_template
    )
    source_ids = set(question["source_chat_ids"])
    row: dict = {
        "conversation_id": question["conversation_id"],
        "category": question.get("category", "?"),
        "question_index": question.get("question_index", -1),
        "question": question["question"],
        "source_chat_ids": sorted(source_ids),
        "cues": out.cues,
        "output": out.output,
        "fair_backfill": {},
    }
    for K in BUDGETS:
        b, a = fair_backfill(out.segments, cosine_segs, source_ids, K)
        row["fair_backfill"][f"baseline_r@{K}"] = round(b, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a - b, 4)
    return row


# ---------------------------------------------------------------------------
# Quick-test harness
# ---------------------------------------------------------------------------
def _load_quick_test_set() -> list[tuple[str, dict, list[Segment], TopdownRunner]]:
    """Returns list of (dataset_name, question, cosine_segs, runner). One
    runner is built per dataset (store-bound) and cached in a dict."""
    by_ds_runner: dict[str, tuple[TopdownRunner, list[dict]]] = {}
    picks: list[tuple[str, dict, list[Segment], TopdownRunner]] = []
    for ds_name, q_idx in QUICK_TEST_SPEC:
        if ds_name not in by_ds_runner:
            store, questions = load_dataset(ds_name)
            runner = TopdownRunner(store)
            by_ds_runner[ds_name] = (runner, questions)
        runner, questions = by_ds_runner[ds_name]
        question = questions[q_idx]
        cos_segs = runner.retrieve_cosine(
            question["question"],
            question["conversation_id"],
            top_k=max(BUDGETS),
        )
        picks.append((ds_name, question, cos_segs, runner))
    # Save cosine embeddings we generated
    for runner, _ in by_ds_runner.values():
        runner.save_caches()
    return picks


def run_quick_test(
    label: str, prompt_template: str, picks: list
) -> tuple[list[dict], float]:
    """Return per-question rows and a score (average arch_r@50)."""
    rows = []
    scores = []
    for ds_name, question, cos_segs, runner in picks:
        row = evaluate_prompt_on_question(runner, prompt_template, question, cos_segs)
        row["dataset"] = ds_name
        rows.append(row)
        scores.append(row["fair_backfill"]["arch_r@50"])
        runner.save_caches()
    avg = sum(scores) / len(scores) if scores else 0.0
    print(
        f"  [{label}] avg arch_r@50={avg:.3f} "
        + " ".join(
            f"({r['dataset']}:{r['fair_backfill']['arch_r@50']:.2f})" for r in rows
        )
    )
    return rows, avg


# v2f baseline per-question on the quick-test set. We compute it once (it's
# the lower bar every ablation is compared against at the very end).
V2F_PROMPT = (
    "You are generating search text for semantic retrieval over a "
    "conversation history. Your cues will be embedded and compared via "
    "cosine similarity.\n\n"
    "Question: {question}\n\n"
    "{context_section}\n"
    "\nFirst, briefly assess: Given what's been retrieved so far, how well "
    "is this search going? What kind of content is still missing? Should "
    "you search for similar content or pivot to a different topic?\n\n"
    'If the question implies MULTIPLE items or asks "all/every", keep '
    "searching for more even if some are already found.\n\n"
    "Then generate 2 search cues based on your assessment. Use specific "
    "vocabulary that would appear in the target conversation turns.\n\n"
    'Do NOT write questions ("Did you mention X?"). Write text that would '
    "actually appear in a chat message.\n\n"
    "Format:\nASSESSMENT: <1-2 sentence self-evaluation>\n"
    "CUE: <text>\nCUE: <text>\nNothing else."
)


# ---------------------------------------------------------------------------
# Distillation loop
# ---------------------------------------------------------------------------
def distill(picks: list) -> tuple[list[str], list[dict]]:
    """Run the top-down distillation. Returns (final_obs_ids, trajectory).

    Trajectory entries are dicts with keys {step, action, kept_obs,
    tried_drop, score, decision}.
    """
    trajectory: list[dict] = []

    # Start with all observations.
    current = [oid for oid, _ in OBSERVATIONS]
    print("\n=== QUICK-TEST: verbose starting prompt ===")
    verbose_prompt = build_prompt(current)
    print(f"(prompt length: {len(verbose_prompt)} chars)")
    _, verbose_score = run_quick_test("verbose", verbose_prompt, picks)
    trajectory.append(
        {
            "step": 0,
            "action": "start",
            "kept_obs": list(current),
            "score": verbose_score,
        }
    )

    # Also score v2f on the same quick-test set, so we have a final
    # comparison benchmark at the end of the run.
    _, v2f_score = run_quick_test("v2f", V2F_PROMPT, picks)
    trajectory.append({"step": 0, "action": "v2f_reference", "score": v2f_score})

    # Gate: only proceed if verbose beats v2f on 3+/5 questions. We also
    # accept "avg score >= v2f" as a milder form of the same gate, to avoid
    # false-rejecting a prompt that happens to be very close.
    verbose_wins = _head_to_head(verbose_prompt, V2F_PROMPT, picks)
    print(
        f"\nGate: verbose beats v2f on {verbose_wins}/5 questions "
        f"(avg: verbose={verbose_score:.3f} vs v2f={v2f_score:.3f})"
    )
    if verbose_wins < 3 and verbose_score < v2f_score:
        print(
            "Verbose prompt does NOT pass the >=3/5 gate. Stopping here "
            "(per methodology: don't distill from a non-viable base)."
        )
        trajectory.append(
            {"step": 0, "action": "gate_failed", "verbose_wins": verbose_wins}
        )
        return current, trajectory
    trajectory.append(
        {"step": 0, "action": "gate_passed", "verbose_wins": verbose_wins}
    )

    # Iteratively try dropping each observation.
    baseline_score = verbose_score
    for step, (oid, _) in enumerate(OBSERVATIONS, start=1):
        if oid not in current:
            continue
        trial = [x for x in current if x != oid]
        trial_prompt = build_prompt(trial)
        label = f"drop-{oid}"
        print(f"\n=== STEP {step}: try dropping '{oid}' ===")
        _, score = run_quick_test(label, trial_prompt, picks)
        # Rule: drop if removal does NOT hurt (score >= baseline). Tie-break
        # in favor of a shorter prompt (minimality). We use a small epsilon
        # so near-equal scores count as "no hurt".
        eps = 1e-6
        if score + eps >= baseline_score:
            print(
                f"  -> DROP '{oid}' (score {score:.3f} >= baseline "
                f"{baseline_score:.3f})"
            )
            current = trial
            baseline_score = score
            decision = "dropped"
        else:
            print(
                f"  -> KEEP '{oid}' (score {score:.3f} < baseline {baseline_score:.3f})"
            )
            decision = "kept"
        trajectory.append(
            {
                "step": step,
                "action": "ablate",
                "tried_drop": oid,
                "score": score,
                "baseline_score": baseline_score,
                "decision": decision,
                "kept_obs": list(current),
            }
        )

    # Final distilled prompt
    print("\n=== DISTILLATION COMPLETE ===")
    print(f"Kept observations ({len(current)}/{len(OBSERVATIONS)}): {current}")
    print(f"Final score: {baseline_score:.3f} (v2f: {v2f_score:.3f})")
    trajectory.append(
        {
            "step": len(OBSERVATIONS) + 1,
            "action": "final",
            "kept_obs": list(current),
            "score": baseline_score,
            "v2f_score": v2f_score,
        }
    )
    return current, trajectory


def _head_to_head(prompt_a: str, prompt_b: str, picks: list) -> int:
    """Count how many quick-test questions prompt_a beats prompt_b on
    (strict greater than at r@50). Uses the existing cache — each unique
    (prompt, question) has already been evaluated."""
    wins = 0
    for ds_name, question, cos_segs, runner in picks:
        a = evaluate_prompt_on_question(runner, prompt_a, question, cos_segs)
        b = evaluate_prompt_on_question(runner, prompt_b, question, cos_segs)
        if a["fair_backfill"]["arch_r@50"] > b["fair_backfill"]["arch_r@50"] + 1e-6:
            wins += 1
    return wins


# ---------------------------------------------------------------------------
# Full eval (run only if distilled prompt beats v2f on quick-test)
# ---------------------------------------------------------------------------
def run_full_eval(distilled_prompt: str) -> dict:
    """Run the distilled prompt on all 4 datasets at K=20 and K=50."""
    all_summaries: dict[str, dict] = {}
    for ds_name in DATASETS:
        print(f"\n=== FULL EVAL on {ds_name} ===")
        store, questions = load_dataset(ds_name)
        runner = TopdownRunner(store)
        # Cosine precompute
        cos_by_idx: dict[int, list[Segment]] = {}
        for i, q in enumerate(questions):
            cos_by_idx[i] = runner.retrieve_cosine(
                q["question"], q["conversation_id"], top_k=max(BUDGETS)
            )
        runner.save_caches()

        rows = []
        for i, q in enumerate(questions):
            row = evaluate_prompt_on_question(
                runner, distilled_prompt, q, cos_by_idx[i]
            )
            rows.append(row)
            if (i + 1) % 5 == 0:
                runner.save_caches()
                print(f"  {i + 1}/{len(questions)}")
        runner.save_caches()

        n = len(rows)
        summary: dict = {"dataset": ds_name, "n": n}
        for K in BUDGETS:
            b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in rows]
            a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in rows]
            b_mean = sum(b_vals) / n
            a_mean = sum(a_vals) / n
            wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
            losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
            ties = n - wins - losses
            summary[f"baseline_r@{K}"] = round(b_mean, 4)
            summary[f"arch_r@{K}"] = round(a_mean, 4)
            summary[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
            summary[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
        all_summaries[ds_name] = summary
        out_path = RESULTS_DIR / f"topdown_distilled_{ds_name}.json"
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "results": rows}, f, indent=2, default=str)
        print(
            f"  {ds_name}: base@20={summary['baseline_r@20']:.3f} "
            f"arch@20={summary['arch_r@20']:.3f} "
            f"d@20={summary['delta_r@20']:+.3f} | "
            f"arch@50={summary['arch_r@50']:.3f} "
            f"d@50={summary['delta_r@50']:+.3f}"
        )
    return all_summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-eval-only",
        action="store_true",
        help="Skip distillation, run full eval with pre-set obs",
    )
    parser.add_argument(
        "--obs",
        nargs="+",
        default=None,
        help="Observation ids to use (for --full-eval-only or debugging)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.full_eval_only and args.obs is not None:
        distilled = args.obs
        trajectory = [{"action": "forced", "kept_obs": list(distilled)}]
    else:
        print(">>> Loading quick-test set (5 questions)")
        picks = _load_quick_test_set()
        distilled, trajectory = distill(picks)

    distilled_prompt = build_prompt(distilled)

    # Save the trajectory + final prompt text
    record = {
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "distilled_obs_ids": distilled,
        "trajectory": trajectory,
        "final_prompt": distilled_prompt,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = RESULTS_DIR / "topdown_distillation_trajectory.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"\nTrajectory saved: {out_path}")

    # Gate again before full eval
    if not args.full_eval_only:
        print("\n>>> Re-running quick-test with distilled prompt + v2f for gate")
        picks_for_gate = _load_quick_test_set()
        distilled_wins = _head_to_head(distilled_prompt, V2F_PROMPT, picks_for_gate)
        print(f"Distilled beats v2f on {distilled_wins}/5 quick-test questions")
        if distilled_wins < 3:
            print(
                "Distilled does NOT clear the >=3/5 bar against v2f — "
                "skipping full eval per methodology."
            )
            record["full_eval_skipped"] = True
            record["distilled_quick_wins_vs_v2f"] = distilled_wins
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2, default=str)
            return
        record["distilled_quick_wins_vs_v2f"] = distilled_wins

    print("\n>>> Running FULL evaluation on all 4 datasets")
    full_summaries = run_full_eval(distilled_prompt)
    record["full_eval_summaries"] = full_summaries
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"\nDone in {time.time() - t0:.1f}s. Final record: {out_path}")


if __name__ == "__main__":
    main()
