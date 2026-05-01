"""Test constraint-aware retrieval on completeness, procedural, and quantitative_aggregation.

The constraint investigation found that generating cues for CONSTRAINT TYPES
(rather than topic paraphrases) gets 100% recall on logic_constraint questions.
This script tests whether that approach generalizes to other "scattered items"
categories:

  - Synthetic completeness (4q): dietary restrictions, medications, budget items, smart home devices
  - Synthetic procedural (2q): party checklist, smart home phases
  - Advanced quantitative_aggregation (3q): total hours, budget compare, Owen's estimates

Uses the iterative constraint collection approach from constraint_retrieval.py,
adapted with category-specific type taxonomies.

Saves:
  - results/constraint_completeness.json
  - results/constraint_procedural.json

Usage:
    uv run python constraint_test.py [--verbose]
"""

import json
import sys
import time
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE_LLM = CACHE_DIR / "resume_llm_cache.json"
CACHE_FILE_EMB = CACHE_DIR / "resume_embedding_cache.json"


# ---------------------------------------------------------------------------
# Cache classes — reads from all existing caches, writes to resume-specific
# ---------------------------------------------------------------------------
class TestEmbeddingCache(EmbeddingCache):
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
            "synth_test_embedding_cache.json",
            "bestshot_embedding_cache.json",
            "task_exec_embedding_cache.json",
            "general_embedding_cache.json",
            "adaptive_embedding_cache.json",
            "fulleval_embedding_cache.json",
            "hybrid_embedding_cache.json",
            "constraint_embedding_cache.json",
            "resume_embedding_cache.json",
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


class TestLLMCache(LLMCache):
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in (
            "llm_cache.json",
            "arch_llm_cache.json",
            "agent_llm_cache.json",
            "tree_llm_cache.json",
            "frontier_llm_cache.json",
            "meta_llm_cache.json",
            "optim_llm_cache.json",
            "synth_test_llm_cache.json",
            "bestshot_llm_cache.json",
            "task_exec_llm_cache.json",
            "general_llm_cache.json",
            "adaptive_llm_cache.json",
            "fulleval_llm_cache.json",
            "hybrid_llm_cache.json",
            "constraint_llm_cache.json",
            "resume_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = CACHE_FILE_LLM
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
# Shared infra
# ---------------------------------------------------------------------------
embedding_cache = TestEmbeddingCache()
llm_cache = TestLLMCache()
client = OpenAI(timeout=120.0)


def embed_text(text: str) -> np.ndarray:
    text = text.strip()
    if not text:
        return np.zeros(1536, dtype=np.float32)
    cached = embedding_cache.get(text)
    if cached is not None:
        return cached
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    emb = np.array(response.data[0].embedding, dtype=np.float32)
    embedding_cache.put(text, emb)
    return emb


def llm_call(prompt: str, model: str = MODEL) -> str:
    cached = llm_cache.get(model, prompt)
    if cached is not None:
        return cached
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=3000,
    )
    text = response.choices[0].message.content or ""
    llm_cache.put(model, prompt, text)
    return text


def save_caches():
    embedding_cache.save()
    llm_cache.save()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_segments(
    segments: list[Segment], max_items: int = 16, max_chars: int = 300
) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def parse_cues(response: str) -> list[str]:
    cues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 0.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def load_questions(json_path: Path, categories: list[str]) -> list[dict]:
    with open(json_path) as f:
        questions = json.load(f)
    return [q for q in questions if q["category"] in categories]


# ---------------------------------------------------------------------------
# Prompts — category-specific constraint type taxonomies
# ---------------------------------------------------------------------------

COMPLETENESS_CONSTRAINT_PROMPT = """\
You are collecting ALL items from a conversation. The question asks for a \
COMPLETE list. Items are scattered across many turns — early mentions, \
updates, corrections, additions from different speakers.

Question: {question}

RETRIEVED SO FAR:
{context}

ITEMS IDENTIFIED SO FAR:
{items_summary}

In conversations where people list many items, there are different TYPES of \
mentions to search for:
- Initial mentions ("I need X", "let's get Y", "we should include Z")
- Detailed specifications ("the X should be 200W", "for the Y, I want the Pro model")
- Updates/changes ("actually, scratch the X, let's go with Y instead")
- Additions from other speakers ("don't forget about W", "we also need Q")
- Late additions ("oh, one more thing — we should also add R")
- Corrections ("I said X but I meant Y")
- Category-specific items (e.g., for dietary: allergies, preferences, religious restrictions, medical diets)
- Items mentioned in passing ("and of course there's the Z we talked about last week")

ROUND {round_num}: What TYPES of items haven't been found yet?

Generate 2 search cues targeting the most likely MISSING items. Write text \
that sounds like it would appear in the actual conversation. DO NOT write \
questions or search commands.

If you believe all relevant items have been found, respond with DONE.

Format:
FOUND_SO_FAR: <brief list of item types already covered>
STILL_MISSING: <what types might be missing>
CUE: <text>
CUE: <text>
(or DONE if complete)"""


PROCEDURAL_CONSTRAINT_PROMPT = """\
You are collecting ALL tasks/steps from a conversation. The question asks \
for a COMPLETE checklist or phase plan. Tasks are scattered across many \
turns — initial plans, revisions, delegations, status updates.

Question: {question}

RETRIEVED SO FAR:
{context}

TASKS IDENTIFIED SO FAR:
{items_summary}

In conversations about project planning, there are different TYPES of \
task-related information:
- Core tasks ("we need to book the venue", "order the decorations")
- Delegated tasks ("Sarah will handle the flowers", "I'll take care of the music")
- Deadline-linked tasks ("by next Friday we need to finalize the menu")
- Dependent tasks ("after the venue is confirmed, we can send invitations")
- Status updates ("the caterer is booked", "still waiting on the DJ")
- Phase descriptions ("Phase 1 is the basic setup", "Phase 2 adds automation")
- Cancelled/deferred tasks ("we decided to skip the ice sculpture")
- Budget-linked tasks ("if we're under budget, add the photo booth")
- Last-minute additions ("we should also arrange parking", "don't forget the gift bags")

ROUND {round_num}: What TYPES of tasks haven't been found yet?

Generate 2 search cues targeting the most likely MISSING tasks or steps. \
Write text that sounds like it would appear in the actual conversation.

If you believe all relevant tasks have been found, respond with DONE.

Format:
FOUND_SO_FAR: <brief list of task types already covered>
STILL_MISSING: <what types might be missing>
CUE: <text>
CUE: <text>
(or DONE if complete)"""


QUANTITATIVE_CONSTRAINT_PROMPT = """\
You are collecting ALL quantitative information from a conversation. The \
question asks about numbers, estimates, or aggregations that are scattered \
across many turns.

Question: {question}

RETRIEVED SO FAR:
{context}

NUMBERS IDENTIFIED SO FAR:
{items_summary}

In project estimation conversations, there are different TYPES of \
numerical information:
- Initial estimates ("I think the backend will take about 40 hours")
- Per-person breakdowns ("Owen estimated 35 hours for his part")
- Revised estimates ("actually, after the meeting I think it's more like 50")
- Budget figures ("the client's budget is $45,000")
- Rate/cost calculations ("at $150/hour, that's about $6,000")
- Scope changes ("if we add the API integration, add another 20 hours")
- Comparison figures ("that's 15% over the original estimate")
- Final/agreed numbers ("so we're going with 120 total hours")
- Category breakdowns ("frontend: 30h, backend: 50h, testing: 20h")

ROUND {round_num}: What TYPES of numbers haven't been found yet?

Generate 2 search cues targeting the most likely MISSING quantitative info. \
Write text that sounds like it would appear in the actual conversation.

If you believe all relevant numbers have been found, respond with DONE.

Format:
FOUND_SO_FAR: <brief list of number types already covered>
STILL_MISSING: <what types might be missing>
CUE: <text>
CUE: <text>
(or DONE if complete)"""


# Also include standard v2f as baseline comparison
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


# ===========================================================================
# Retrieval approaches
# ===========================================================================


def run_v2f_baseline(
    store: SegmentStore,
    question: str,
    conv_id: str,
) -> tuple[list[Segment], dict]:
    """Standard v2f: 1 LLM call, 2 cues."""
    query_emb = embed_text(question)
    hop0 = store.search(query_emb, top_k=10, conversation_id=conv_id)
    all_segments = list(hop0.segments)
    exclude = {s.index for s in all_segments}

    context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + format_segments(
        all_segments, max_items=12, max_chars=250
    )
    prompt = V2F_PROMPT.format(question=question, context_section=context_section)
    output = llm_call(prompt)
    cues = parse_cues(output)

    for cue in cues[:2]:
        cue_emb = embed_text(cue)
        result = store.search(
            cue_emb,
            top_k=10,
            conversation_id=conv_id,
            exclude_indices=exclude,
        )
        for seg in result.segments:
            if seg.index not in exclude:
                all_segments.append(seg)
                exclude.add(seg.index)

    return all_segments, {"method": "v2f", "cues": cues[:2], "llm_calls": 1}


def run_iterative_constraint(
    store: SegmentStore,
    question: str,
    conv_id: str,
    prompt_template: str,
    max_rounds: int = 4,
) -> tuple[list[Segment], dict]:
    """Iterative constraint collection with category-specific type taxonomy."""
    query_emb = embed_text(question)
    hop0 = store.search(query_emb, top_k=10, conversation_id=conv_id)
    all_segments = list(hop0.segments)
    exclude = {s.index for s in all_segments}
    total_llm_calls = 0
    all_cues: list[str] = []

    items_summary = "(none yet -- first round)"

    for round_num in range(max_rounds):
        context = format_segments(all_segments)
        prompt = prompt_template.format(
            question=question,
            context=context,
            items_summary=items_summary,
            round_num=round_num + 1,
        )
        output = llm_call(prompt)
        total_llm_calls += 1

        if "DONE" in output.upper() and "CUE:" not in output:
            break

        cues = parse_cues(output)
        if not cues:
            break

        all_cues.extend(cues[:2])

        # Extract items summary from output
        for line in output.split("\n"):
            if line.strip().startswith("FOUND_SO_FAR:"):
                items_summary = line.strip()[13:].strip()
                break

        # Retrieve for each cue
        for cue in cues[:2]:
            cue_emb = embed_text(cue)
            result = store.search(
                cue_emb,
                top_k=10,
                conversation_id=conv_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

    return all_segments, {
        "method": "iterative_constraint",
        "cues": all_cues,
        "llm_calls": total_llm_calls,
        "rounds": min(round_num + 1, max_rounds),
    }


# ===========================================================================
# Evaluation
# ===========================================================================


def evaluate_question(
    store: SegmentStore,
    question: dict,
    method_name: str,
    method_fn,
    method_kwargs: dict | None = None,
    verbose: bool = False,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    t0 = time.time()
    if method_kwargs:
        segments, metadata = method_fn(store, q_text, conv_id, **method_kwargs)
    else:
        segments, metadata = method_fn(store, q_text, conv_id)
    elapsed = time.time() - t0

    # Deduplicate
    seen: set[int] = set()
    deduped: list[Segment] = []
    for seg in segments:
        if seg.index not in seen:
            deduped.append(seg)
            seen.add(seg.index)

    retrieved_ids = {s.turn_id for s in deduped}
    total_retrieved = len(deduped)

    # Baseline at various budgets
    query_emb = embed_text(q_text)
    baseline_result = store.search(
        query_emb,
        top_k=max(50, total_retrieved),
        conversation_id=conv_id,
    )

    result = {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "method": method_name,
        "total_retrieved": total_retrieved,
        "time_s": round(elapsed, 2),
        "metadata": metadata,
    }

    for budget in [20, 50]:
        # Baseline
        bl_ids = {s.turn_id for s in baseline_result.segments[:budget]}
        bl_recall = compute_recall(bl_ids, source_ids)
        result[f"baseline_r@{budget}"] = round(bl_recall, 4)

        # Method (at budget or at total retrieved, whichever is smaller)
        method_ids = {s.turn_id for s in deduped[:budget]}
        method_recall = compute_recall(method_ids, source_ids)
        result[f"method_r@{budget}"] = round(method_recall, 4)

    # Full recall (all retrieved)
    result["method_r@all"] = round(compute_recall(retrieved_ids, source_ids), 4)

    # Detailed hit/miss info
    result["hits"] = sorted(retrieved_ids & source_ids)
    result["misses"] = sorted(source_ids - retrieved_ids)

    if verbose:
        print(f"    Source: {sorted(source_ids)} ({len(source_ids)} turns)")
        print(f"    Hits: {result['hits']}")
        print(f"    Misses: {result['misses']}")
        for budget in [20, 50]:
            print(
                f"    r@{budget}: method={result[f'method_r@{budget}']:.2f} "
                f"baseline={result[f'baseline_r@{budget}']:.2f}"
            )
        print(
            f"    r@all={result['method_r@all']:.2f} "
            f"(retrieved {total_retrieved} segs, {metadata.get('llm_calls', '?')} LLM calls)"
        )

    return result


# ===========================================================================
# Main
# ===========================================================================
def main():
    verbose = "--verbose" in sys.argv

    # -----------------------------------------------------------------------
    # Test 1: Completeness questions (synthetic, 4q)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 1: Constraint-aware retrieval on COMPLETENESS (4q)")
    print("=" * 80)

    synth_store = SegmentStore(DATA_DIR, "segments_synthetic.npz")
    completeness_qs = load_questions(
        DATA_DIR / "questions_synthetic.json", ["completeness"]
    )

    completeness_results = []
    for q in completeness_qs:
        print(f"\n  Q{q['question_index']}: {q['question'][:80]}...")

        # V2f baseline
        print("    --- v2f ---")
        v2f_result = evaluate_question(
            synth_store, q, "v2f", run_v2f_baseline, verbose=verbose
        )
        completeness_results.append(v2f_result)

        # Iterative constraint
        print("    --- iterative_constraint ---")
        ic_result = evaluate_question(
            synth_store,
            q,
            "iterative_constraint",
            run_iterative_constraint,
            method_kwargs={"prompt_template": COMPLETENESS_CONSTRAINT_PROMPT},
            verbose=verbose,
        )
        completeness_results.append(ic_result)

        # Print comparison
        for budget in [20, 50]:
            v_r = v2f_result[f"method_r@{budget}"]
            ic_r = ic_result[f"method_r@{budget}"]
            bl_r = v2f_result[f"baseline_r@{budget}"]
            delta = ic_r - v_r
            print(
                f"    r@{budget}: baseline={bl_r:.2f} v2f={v_r:.2f} "
                f"constraint={ic_r:.2f} delta={delta:+.2f}"
            )

    save_caches()

    # -----------------------------------------------------------------------
    # Test 2: Procedural questions (synthetic, 2q)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 2: Constraint-aware retrieval on PROCEDURAL (2q)")
    print("=" * 80)

    procedural_qs = load_questions(
        DATA_DIR / "questions_synthetic.json", ["procedural"]
    )

    procedural_results = []
    for q in procedural_qs:
        print(f"\n  Q{q['question_index']}: {q['question'][:80]}...")

        # V2f baseline
        print("    --- v2f ---")
        v2f_result = evaluate_question(
            synth_store, q, "v2f", run_v2f_baseline, verbose=verbose
        )
        procedural_results.append(v2f_result)

        # Iterative constraint
        print("    --- iterative_constraint ---")
        ic_result = evaluate_question(
            synth_store,
            q,
            "iterative_constraint",
            run_iterative_constraint,
            method_kwargs={"prompt_template": PROCEDURAL_CONSTRAINT_PROMPT},
            verbose=verbose,
        )
        procedural_results.append(ic_result)

        for budget in [20, 50]:
            v_r = v2f_result[f"method_r@{budget}"]
            ic_r = ic_result[f"method_r@{budget}"]
            bl_r = v2f_result[f"baseline_r@{budget}"]
            delta = ic_r - v_r
            print(
                f"    r@{budget}: baseline={bl_r:.2f} v2f={v_r:.2f} "
                f"constraint={ic_r:.2f} delta={delta:+.2f}"
            )

    save_caches()

    # -----------------------------------------------------------------------
    # Test 3: Quantitative aggregation (advanced, 3q)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 3: Constraint-aware retrieval on QUANTITATIVE_AGGREGATION (3q)")
    print("=" * 80)

    adv_store = SegmentStore(DATA_DIR, "segments_advanced.npz")
    quant_qs = load_questions(
        DATA_DIR / "questions_advanced.json", ["quantitative_aggregation"]
    )

    quant_results = []
    for q in quant_qs:
        print(f"\n  Q{q['question_index']}: {q['question'][:80]}...")

        # V2f baseline
        print("    --- v2f ---")
        v2f_result = evaluate_question(
            adv_store, q, "v2f", run_v2f_baseline, verbose=verbose
        )
        quant_results.append(v2f_result)

        # Iterative constraint
        print("    --- iterative_constraint ---")
        ic_result = evaluate_question(
            adv_store,
            q,
            "iterative_constraint",
            run_iterative_constraint,
            method_kwargs={"prompt_template": QUANTITATIVE_CONSTRAINT_PROMPT},
            verbose=verbose,
        )
        quant_results.append(ic_result)

        for budget in [20, 50]:
            v_r = v2f_result[f"method_r@{budget}"]
            ic_r = ic_result[f"method_r@{budget}"]
            bl_r = v2f_result[f"baseline_r@{budget}"]
            delta = ic_r - v_r
            print(
                f"    r@{budget}: baseline={bl_r:.2f} v2f={v_r:.2f} "
                f"constraint={ic_r:.2f} delta={delta:+.2f}"
            )

    save_caches()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    # Combine completeness + procedural into one file for completeness
    completeness_output = {
        "description": "Constraint-aware retrieval on completeness questions",
        "completeness": completeness_results,
        "summary": _summarize_paired(completeness_results),
    }
    out_path = RESULTS_DIR / "constraint_completeness.json"
    with open(out_path, "w") as f:
        json.dump(completeness_output, f, indent=2)
    print(f"\nSaved: {out_path}")

    procedural_output = {
        "description": "Constraint-aware retrieval on procedural + quantitative_aggregation questions",
        "procedural": procedural_results,
        "quantitative_aggregation": quant_results,
        "summary_procedural": _summarize_paired(procedural_results),
        "summary_quant": _summarize_paired(quant_results),
    }
    out_path = RESULTS_DIR / "constraint_procedural.json"
    with open(out_path, "w") as f:
        json.dump(procedural_output, f, indent=2)
    print(f"Saved: {out_path}")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY: Constraint-aware vs V2f baseline")
    print("=" * 80)

    all_paired = [
        ("completeness", completeness_results),
        ("procedural", procedural_results),
        ("quant_aggregation", quant_results),
    ]

    print(
        f"\n{'Category':<22} {'N':>3} {'Budget':>6} {'Baseline':>8} "
        f"{'V2f':>6} {'Constraint':>10} {'delta':>7}"
    )
    print("-" * 72)

    for cat_name, results in all_paired:
        summary = _summarize_paired(results)
        n = summary["n_questions"]
        for budget in [20, 50]:
            bl = summary[f"baseline_r@{budget}"]
            v = summary[f"v2f_r@{budget}"]
            c = summary[f"constraint_r@{budget}"]
            d = c - v
            print(
                f"{cat_name:<22} {n:>3} {'r@' + str(budget):>6} {bl:>8.2f} "
                f"{v:>6.2f} {c:>10.2f} {d:>+7.2f}"
            )


def _summarize_paired(results: list[dict]) -> dict:
    """Summarize paired (v2f, constraint) results."""
    v2f_results = [r for r in results if r["method"] == "v2f"]
    ic_results = [r for r in results if r["method"] == "iterative_constraint"]

    summary: dict = {"n_questions": len(v2f_results)}
    for budget in [20, 50]:
        lbl = f"r@{budget}"
        if v2f_results:
            bl_vals = [r[f"baseline_{lbl}"] for r in v2f_results]
            v2f_vals = [r[f"method_{lbl}"] for r in v2f_results]
            summary[f"baseline_{lbl}"] = round(sum(bl_vals) / len(bl_vals), 4)
            summary[f"v2f_{lbl}"] = round(sum(v2f_vals) / len(v2f_vals), 4)
        if ic_results:
            ic_vals = [r[f"method_{lbl}"] for r in ic_results]
            summary[f"constraint_{lbl}"] = round(sum(ic_vals) / len(ic_vals), 4)

    return summary


if __name__ == "__main__":
    main()
