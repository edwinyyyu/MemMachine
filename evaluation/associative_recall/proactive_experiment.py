"""Proactive retrieval prompt variants.

Tests whether the anti-question instruction in V2f hurts proactive tasks.

Variants:
  v15             — exact V15 prompt (baseline)
  v2f             — V15 + completeness + anti-question (LoCoMo winner)
  v2f_no_antiq    — V15 + completeness only (no anti-question)
  v15_completeness — V15 + completeness only (same as v2f_no_antiq but clearer name)
  adaptive        — detects task vs question; allows question-style cues only for tasks

Usage:
    uv run python proactive_experiment.py            # runs all variants on 4 proactive Qs
"""

import json
import re
import time
from dataclasses import dataclass
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
CACHE_FILE_LLM = CACHE_DIR / "proactive_llm_cache.json"
CACHE_FILE_EMB = CACHE_DIR / "proactive_embedding_cache.json"
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Caches — fresh proactive caches (but also read pre-existing shared caches
# to avoid re-embedding content like source segments that were already embedded).
# ---------------------------------------------------------------------------
class ProactiveEmbeddingCache(EmbeddingCache):
    """Reads all existing embedding caches, writes to proactive-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
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
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = CACHE_FILE_EMB
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


class ProactiveLLMCache(LLMCache):
    """Fresh proactive LLM cache (only reads its own file)."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        self.cache_file = CACHE_FILE_LLM
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache = json.load(f)
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
V15_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

V2F_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

V2F_NO_ANTIQ_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# Adaptive: adds a proactive-task-aware variant of the anti-question rule.
# For proactive/task-style queries, we ENCOURAGE generating "things to look
# into" cues (including question-like probes), because the question itself
# doesn't supply the vocabulary that will be present in the conversation.
# For direct info questions, we keep the V2f anti-question rule.
ADAPTIVE_TASK_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

The user is asking for help with a TASK (cooking, drafting, planning, \
setup, preparation). The relevant information is spread across multiple \
past messages covering different sub-topics (constraints, details, \
decisions, steps). A single embedding of the task request will not \
retrieve all of it.

First, briefly assess: Given what's been retrieved so far, what sub-topics \
are still missing?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Each cue should be a \
DENSE BUNDLE of specific vocabulary (names, entities, constraints, actions, \
numbers) from the relevant sub-topic — the kind of keywords that would \
appear in the target conversation turns. Probe-style cues are fine \
(you may mention what you are looking for), as long as they are packed \
with concrete vocabulary.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

ADAPTIVE_QUESTION_PROMPT = V2F_PROMPT  # direct questions -> standard V2f


TASK_VERB_PATTERNS = [
    r"\bcook\b",
    r"\bdraft\b",
    r"\bprepare\b",
    r"\bhelp me (with|prepare|plan|set|write|draft|get|make|build|create)\b",
    r"\bset up\b",
    r"\bsetup\b",
    r"\bwhat (should i|do i need|needs to)\b",
    r"\bwhat needs to happen\b",
    r"\bwrite (me|a|an|the)\b",
    r"\bcompose (me|a|an)\b",
    r"\bbuild me\b",
    r"\bi want to\b",
    r"\bi'd like to\b",
    r"\bcreate (me|a|an|the)\b",
    r"\bmake (me|a|an)\b.*\bfor\b",
    r"\bkeep in mind\b",
    r"\bplease consider\b",
    r"\blist of topics\b",
    r"\bchecklist\b",
    r"\bto-?do list\b",
    r"\bstatus update\b",
    r"\bremaining (tasks|phases|steps)\b",
]
QUESTION_WORD_PATTERNS = [
    r"^(what did|when did|who did|where did|how did|why did)\b",
    r"^(what was|when was|who was|where was)\b",
    r"^(what is|what's|when is|who is|where is)\b",
    r"^(did|does|do|was|were|has|have|is)\b",
]


def detect_task(question: str) -> bool:
    """Return True if the question looks like a proactive task request."""
    q = question.strip().lower()
    # Direct question signals
    for pat in QUESTION_WORD_PATTERNS:
        if re.search(pat, q):
            # Still check: "what needs to happen" is a task even though starts with "what"
            for tpat in TASK_VERB_PATTERNS:
                if re.search(tpat, q):
                    return True
            return False
    # Task verb signals
    for pat in TASK_VERB_PATTERNS:
        if re.search(pat, q):
            return True
    return False


# ---------------------------------------------------------------------------
# Retrieval driver (v15-shape: hop0 top-10 from question, then 2 cues top-10 each)
# ---------------------------------------------------------------------------
@dataclass
class VariantResult:
    question: str
    cues: list[str]
    output: str
    all_segments: list[Segment]
    retrieved_turn_ids: list[int]
    is_task: bool
    variant: str


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
    cues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


class ProactiveRunner:
    def __init__(self, store: SegmentStore):
        self.store = store
        self.client = OpenAI(timeout=60.0)
        self.emb_cache = ProactiveEmbeddingCache()
        self.llm_cache = ProactiveLLMCache()

    def embed(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.emb_cache.get(text)
        if cached is not None:
            return cached
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        e = np.array(resp.data[0].embedding, dtype=np.float32)
        self.emb_cache.put(text, e)
        return e

    def llm(self, prompt: str) -> str:
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

    def save(self) -> None:
        self.emb_cache.save()
        self.llm_cache.save()

    def run_variant(
        self, variant: str, question: str, conversation_id: str
    ) -> VariantResult:
        is_task = detect_task(question)
        if variant == "v15":
            prompt_tpl = V15_PROMPT
        elif variant == "v2f":
            prompt_tpl = V2F_PROMPT
        elif variant == "v2f_no_antiq" or variant == "v15_completeness":
            prompt_tpl = V2F_NO_ANTIQ_PROMPT
        elif variant == "adaptive":
            prompt_tpl = ADAPTIVE_TASK_PROMPT if is_task else ADAPTIVE_QUESTION_PROMPT
        else:
            raise ValueError(f"unknown variant {variant}")

        # Hop 0: embed question, retrieve top-10
        q_emb = self.embed(question)
        hop0 = self.store.search(q_emb, top_k=10, conversation_id=conversation_id)
        all_segments: list[Segment] = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        ctx = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(
            all_segments
        )
        prompt = prompt_tpl.format(question=question, context_section=ctx)
        output = self.llm(prompt)
        cues = _parse_cues(output)

        for cue in cues[:2]:
            cue_emb = self.embed(cue)
            res = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in res.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return VariantResult(
            question=question,
            cues=cues[:2],
            output=output,
            all_segments=all_segments,
            retrieved_turn_ids=[s.turn_id for s in all_segments],
            is_task=is_task,
            variant=variant,
        )


def compute_recall(
    retrieved_turn_ids: list[int], source_turn_ids: list[int], k: int
) -> float:
    retrieved_k = set(retrieved_turn_ids[:k])
    src = set(source_turn_ids)
    if not src:
        return 1.0
    return len(retrieved_k & src) / len(src)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "questions_synthetic.json") as f:
        qs = json.load(f)
    proactive = [q for q in qs if q.get("category") == "proactive"]
    print(f"Loaded {len(proactive)} proactive questions.")

    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_synthetic.npz")
    runner = ProactiveRunner(store)

    variants = ["v15", "v2f", "v2f_no_antiq", "adaptive"]
    all_results: dict[str, list[dict]] = {v: [] for v in variants}
    all_cues: dict[str, dict[str, list[str]]] = {v: {} for v in variants}

    for v in variants:
        print(f"\n{'=' * 70}\nVARIANT: {v}\n{'=' * 70}")
        for q in proactive:
            t0 = time.time()
            res = runner.run_variant(v, q["question"], q["conversation_id"])
            elapsed = time.time() - t0
            r20 = compute_recall(res.retrieved_turn_ids, q["source_chat_ids"], 20)
            r50 = compute_recall(res.retrieved_turn_ids, q["source_chat_ids"], 50)
            row = {
                "question": q["question"],
                "conversation_id": q["conversation_id"],
                "question_index": q["question_index"],
                "source_chat_ids": q["source_chat_ids"],
                "num_source_turns": len(q["source_chat_ids"]),
                "is_task": res.is_task,
                "variant": v,
                "cues": res.cues,
                "output": res.output,
                "retrieved_turn_ids": res.retrieved_turn_ids,
                "total_retrieved": len(res.retrieved_turn_ids),
                "r@20": r20,
                "r@50": r50,
                "time_s": round(elapsed, 2),
            }
            all_results[v].append(row)
            all_cues[v][q["question"][:50]] = res.cues
            print(f"  [{v}] Q: {q['question'][:60]}...")
            print(
                f"    is_task={res.is_task}, r@20={r20:.3f}, r@50={r50:.3f}, |src|={len(q['source_chat_ids'])}"
            )
            for c in res.cues:
                print(f"    CUE: {c[:120]}")
            runner.save()
        runner.save()

    # Summary
    print(f"\n{'=' * 70}\nSUMMARY — mean recall over 4 proactive questions\n{'=' * 70}")
    print(f"{'variant':<20s} {'mean_r@20':>11s} {'mean_r@50':>11s}")
    summary = {}
    for v in variants:
        r20s = [r["r@20"] for r in all_results[v]]
        r50s = [r["r@50"] for r in all_results[v]]
        m20 = sum(r20s) / len(r20s)
        m50 = sum(r50s) / len(r50s)
        print(f"{v:<20s} {m20:>11.4f} {m50:>11.4f}")
        summary[v] = {
            "mean_r@20": round(m20, 4),
            "mean_r@50": round(m50, 4),
            "per_question": [
                {"q": r["question"][:60], "r@20": r["r@20"], "r@50": r["r@50"]}
                for r in all_results[v]
            ],
        }

    # Save
    out_detailed = RESULTS_DIR / "proactive_detailed.json"
    with open(out_detailed, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved detailed: {out_detailed}")
    out_summary = RESULTS_DIR / "proactive_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {out_summary}")


if __name__ == "__main__":
    main()
