"""Investigation: Why logic_constraint questions score BELOW baseline with cue-based retrieval.

Steps:
  1. Analyze baseline (cosine) retrieval — what it finds and misses
  2. Analyze why LLM-generated cues hurt retrieval on these questions
  3. Test alternative approaches:
     a. Expanded baseline (top-30, top-50 cosine only)
     b. Constraint-aware retrieval (typed constraint cues)
     c. Iterative constraint collection
     d. Broad then LLM-rerank (top-50 + LLM filter)
  4. Also test on completeness & procedural categories

Usage:
    uv run python constraint_retrieval.py [--step N] [--verbose]
"""

import json
from collections import defaultdict
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

CACHE_FILE_LLM = CACHE_DIR / "constraint_llm_cache.json"
CACHE_FILE_EMB = CACHE_DIR / "constraint_embedding_cache.json"


# ---------------------------------------------------------------------------
# Cache classes
# ---------------------------------------------------------------------------
class ConstraintEmbeddingCache(EmbeddingCache):
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
            "constraint_embedding_cache.json",
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


class ConstraintLLMCache(LLMCache):
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
            "constraint_llm_cache.json",
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
embedding_cache = ConstraintEmbeddingCache()
llm_cache = ConstraintLLMCache()
client = OpenAI(timeout=60.0)


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
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
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


def retrieve_top_k(
    store: SegmentStore,
    query: str,
    conversation_id: str,
    top_k: int = 10,
    exclude_indices: set[int] | None = None,
) -> list[Segment]:
    query_emb = embed_text(query)
    result = store.search(
        query_emb,
        top_k=top_k,
        conversation_id=conversation_id,
        exclude_indices=exclude_indices,
    )
    return list(result.segments)


def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 0.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def load_questions(json_path: Path, categories: list[str] | None = None) -> list[dict]:
    with open(json_path) as f:
        questions = json.load(f)
    if categories:
        questions = [q for q in questions if q["category"] in categories]
    return questions


# ===========================================================================
# Step 1: Analyze baseline cosine retrieval
# ===========================================================================
def step1_analyze_baseline(verbose: bool = False):
    """For each logic_constraint question, show what cosine top-k finds vs misses."""
    print("\n" + "=" * 70)
    print("STEP 1: Baseline cosine retrieval analysis for logic_constraint")
    print("=" * 70)

    store = SegmentStore(DATA_DIR, "segments_puzzle.npz")
    questions = load_questions(DATA_DIR / "questions_puzzle.json", ["logic_constraint"])

    for q in questions:
        conv_id = q["conversation_id"]
        question_text = q["question"]
        source_ids = set(q["source_chat_ids"])

        print(f"\n--- Q{q['question_index']}: {question_text[:80]}...")
        print(f"    Source turns ({len(source_ids)}): {sorted(source_ids)}")

        # Get ALL segments for this conversation to see what source turns look like
        conv_segments = [s for s in store.segments if s.conversation_id == conv_id]
        source_segments = [s for s in conv_segments if s.turn_id in source_ids]

        if verbose:
            print("\n    Source turn contents:")
            for seg in sorted(source_segments, key=lambda s: s.turn_id):
                print(f"      Turn {seg.turn_id:3d} [{seg.role}]: {seg.text[:120]}...")

        # Cosine retrieval at different k values
        query_emb = embed_text(question_text)
        for top_k in [20, 30, 50, 80]:
            result = store.search(query_emb, top_k=top_k, conversation_id=conv_id)
            retrieved_ids = {s.turn_id for s in result.segments}
            found = retrieved_ids & source_ids
            missed = source_ids - retrieved_ids
            recall = len(found) / len(source_ids)
            print(
                f"\n    Cosine top-{top_k}: recall={recall:.1%} ({len(found)}/{len(source_ids)})"
            )
            print(f"      Found: {sorted(found)}")
            print(f"      Missed: {sorted(missed)}")

            if verbose and top_k == 20:
                # Show scores and what was actually retrieved
                print("      Retrieved turn IDs with scores:")
                for seg, score in zip(result.segments[:20], result.scores[:20]):
                    marker = " <-- SOURCE" if seg.turn_id in source_ids else ""
                    print(
                        f"        Turn {seg.turn_id:3d} ({score:.4f}): "
                        f"{seg.text[:80]}...{marker}"
                    )

        # Analyze: for missed source turns, what is their cosine sim to the question?
        print("\n    Cosine similarity of each SOURCE turn to the question:")
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
        sims = []
        for seg in sorted(source_segments, key=lambda s: s.turn_id):
            seg_emb = store.normalized_embeddings[seg.index]
            sim = float(np.dot(seg_emb, query_norm))
            sims.append((seg.turn_id, sim, seg.text[:100]))

        sims.sort(key=lambda x: x[1], reverse=True)
        for turn_id, sim, text in sims:
            in_top20 = (
                "TOP20"
                if turn_id
                in {
                    s.turn_id
                    for s in store.search(
                        query_emb, top_k=20, conversation_id=conv_id
                    ).segments
                }
                else "MISSED"
            )
            print(f"      Turn {turn_id:3d} (sim={sim:.4f}) [{in_top20}]: {text}...")


# ===========================================================================
# Step 2: Analyze why cues hurt — read cached cues from existing results
# ===========================================================================
def step2_analyze_cue_damage(verbose: bool = False):
    """Examine what cues were generated and how they affected retrieval."""
    print("\n" + "=" * 70)
    print("STEP 2: Why do LLM cues HURT logic_constraint retrieval?")
    print("=" * 70)

    store = SegmentStore(DATA_DIR, "segments_puzzle.npz")
    questions = load_questions(DATA_DIR / "questions_puzzle.json", ["logic_constraint"])

    # Read results from v15 control
    v15_path = RESULTS_DIR / "puzzle_v15_control.json"
    v2f_path = RESULTS_DIR / "puzzle_v2f.json"

    for results_path, label in [(v15_path, "v15_control"), (v2f_path, "v2f")]:
        if not results_path.exists():
            print(f"\n  {label}: results file not found")
            continue

        with open(results_path) as f:
            all_results = json.load(f)

        lc_results = [r for r in all_results if r["category"] == "logic_constraint"]

        print(f"\n--- {label} results on logic_constraint ---")
        for r in lc_results:
            q_idx = r["question_index"]
            question_text = r["question"]
            source_ids = set(r["source_chat_ids"])
            baseline_r20 = r["baseline_recalls"]["r@20"]
            arch_r20 = r["arch_recalls"]["r@20"]
            delta = arch_r20 - baseline_r20
            meta = r.get("metadata", {})
            cues = meta.get("cues", [])

            print(
                f"\n  Q{q_idx}: baseline R@20={baseline_r20:.1%}, "
                f"{label} R@20={arch_r20:.1%}, delta={delta:+.1%}"
            )
            print(f"  Question: {question_text[:80]}...")
            print("  Cues generated:")
            for i, cue in enumerate(cues):
                print(f"    [{i}]: {cue[:150]}...")

            # Now analyze: what did each cue actually retrieve?
            conv_id = r["conversation_id"]
            query_emb = embed_text(question_text)
            hop0 = store.search(query_emb, top_k=10, conversation_id=conv_id)
            hop0_ids = {s.turn_id for s in hop0.segments}
            exclude = {s.index for s in hop0.segments}

            print("\n  Hop 0 (question cosine top-10):")
            print(f"    Found source turns: {sorted(hop0_ids & source_ids)}")
            print(f"    Missed source turns: {sorted(source_ids - hop0_ids)}")

            for i, cue in enumerate(cues[:2]):
                cue_emb = embed_text(cue)
                cue_result = store.search(
                    cue_emb,
                    top_k=10,
                    conversation_id=conv_id,
                    exclude_indices=exclude,
                )
                cue_ids = {s.turn_id for s in cue_result.segments}
                new_source = cue_ids & source_ids
                new_noise = cue_ids - source_ids

                print(f"\n  Cue {i} retrieval (top-10, excluding hop0):")
                print(f"    New source turns found: {sorted(new_source)}")
                print(f"    Non-source turns: {sorted(new_noise)}")

                if verbose:
                    print("    All retrieved:")
                    for seg, score in zip(
                        cue_result.segments[:10], cue_result.scores[:10]
                    ):
                        marker = " <-- SOURCE" if seg.turn_id in source_ids else ""
                        print(
                            f"      Turn {seg.turn_id:3d} ({score:.4f}): "
                            f"{seg.text[:80]}...{marker}"
                        )

                # Update exclude for next cue
                for s in cue_result.segments:
                    exclude.add(s.index)

    # Key diagnostic: compare hop0 at top-20 vs (hop0-top10 + cue-top10 + cue-top10)
    print(
        "\n\n--- CRITICAL COMPARISON: cosine-only top-20 vs hop0(10)+cue(10)+cue(10) ---"
    )
    for q in questions:
        conv_id = q["conversation_id"]
        question_text = q["question"]
        source_ids = set(q["source_chat_ids"])

        # Cosine top-20
        query_emb = embed_text(question_text)
        top20 = store.search(query_emb, top_k=20, conversation_id=conv_id)
        top20_source = {s.turn_id for s in top20.segments} & source_ids

        # Cosine top-30 (same budget as hop0+2cues at 10 each)
        top30 = store.search(query_emb, top_k=30, conversation_id=conv_id)
        top30_source = {s.turn_id for s in top30.segments} & source_ids

        print(f"\n  Q{q['question_index']}: Source turns = {len(source_ids)}")
        print(
            f"    Cosine top-20: {len(top20_source)} hits = "
            f"{compute_recall(top20_source, source_ids):.1%}"
        )
        print(
            f"    Cosine top-30: {len(top30_source)} hits = "
            f"{compute_recall(top30_source, source_ids):.1%}"
        )


# ===========================================================================
# Step 3a: Expanded baseline — just retrieve more with cosine
# ===========================================================================
def step3a_expanded_baseline(verbose: bool = False):
    """Test if simply retrieving more by cosine beats cue-based approaches."""
    print("\n" + "=" * 70)
    print("STEP 3a: Expanded cosine baseline (no LLM)")
    print("=" * 70)

    datasets = [
        (
            "puzzle",
            "segments_puzzle.npz",
            "questions_puzzle.json",
            ["logic_constraint"],
        ),
        (
            "synthetic",
            "segments_synthetic.npz",
            "questions_synthetic.json",
            ["completeness", "procedural"],
        ),
    ]

    results = []
    for ds_name, npz, qfile, cats in datasets:
        store = SegmentStore(DATA_DIR, npz)
        questions = load_questions(DATA_DIR / qfile, cats)

        print(f"\n--- {ds_name}: {cats} ---")
        for q in questions:
            conv_id = q["conversation_id"]
            question_text = q["question"]
            source_ids = set(q["source_chat_ids"])

            query_emb = embed_text(question_text)

            for top_k in [20, 30, 50, 80]:
                result = store.search(query_emb, top_k=top_k, conversation_id=conv_id)
                retrieved_ids = {s.turn_id for s in result.segments}
                recall = compute_recall(retrieved_ids, source_ids)
                results.append(
                    {
                        "dataset": ds_name,
                        "category": q["category"],
                        "question_index": q["question_index"],
                        "method": f"cosine_top{top_k}",
                        "recall": recall,
                        "hits": len(retrieved_ids & source_ids),
                        "total_source": len(source_ids),
                    }
                )

                if top_k in [20, 30, 50]:
                    print(
                        f"  Q{q['question_index']} [{q['category']}] "
                        f"top-{top_k}: {recall:.1%} "
                        f"({len(retrieved_ids & source_ids)}/{len(source_ids)})"
                    )

    return results


# ===========================================================================
# Step 3b: Constraint-aware retrieval
# ===========================================================================

CONSTRAINT_AWARE_PROMPT = """\
You are retrieving information from a conversation about {topic_hint}. \
The question asks about ALL constraints or requirements that were discussed.

Question: {question}

RETRIEVED SO FAR:
{context}

In conversations about scheduling, seating, or planning, constraints come \
in many TYPES:
- Location/proximity preferences ("I need to be near X")
- Interpersonal conflicts ("A can't be next to B")
- Physical requirements ("needs natural light", "needs AV equipment")
- Temporal constraints ("must be on Tuesday", "non-negotiable timeslot")
- Updates/overrides ("actually, that conflict was resolved", "they rescheduled")
- Capacity constraints ("20 people, needs the big room")
- Pair/group requirements ("C and D need to sit together")

For each constraint TYPE that might exist but ISN'T covered in the retrieved \
content, generate a search cue. The cue should sound like something someone \
would actually say in conversation when mentioning that type of constraint.

Focus on constraint types NOT YET found. Generate cues for missing types.

Format:
ASSESSMENT: <which constraint types are covered vs missing>
CUE: <text mimicking conversation content about a missing constraint type>
CUE: <text>
CUE: <text>
Nothing else."""


def step3b_constraint_aware(verbose: bool = False):
    """Typed constraint cues — generate cues for each missing constraint type."""
    print("\n" + "=" * 70)
    print("STEP 3b: Constraint-aware retrieval")
    print("=" * 70)

    datasets = [
        (
            "puzzle",
            "segments_puzzle.npz",
            "questions_puzzle.json",
            ["logic_constraint"],
        ),
        (
            "synthetic",
            "segments_synthetic.npz",
            "questions_synthetic.json",
            ["completeness", "procedural"],
        ),
    ]

    all_results = []
    for ds_name, npz, qfile, cats in datasets:
        store = SegmentStore(DATA_DIR, npz)
        questions = load_questions(DATA_DIR / qfile, cats)

        print(f"\n--- {ds_name}: {cats} ---")
        for q in questions:
            conv_id = q["conversation_id"]
            question_text = q["question"]
            source_ids = set(q["source_chat_ids"])

            # Hop 0: cosine top-10
            query_emb = embed_text(question_text)
            hop0 = store.search(query_emb, top_k=10, conversation_id=conv_id)
            all_segments = list(hop0.segments)
            exclude = {s.index for s in all_segments}

            # Determine topic hint from the question
            topic_hint = "planning or scheduling"
            if "desk" in question_text.lower() or "seat" in question_text.lower():
                topic_hint = "a desk/seating arrangement"
            elif "schedule" in question_text.lower() or "room" in question_text.lower():
                topic_hint = "conference room scheduling"
            elif "budget" in question_text.lower():
                topic_hint = "project budget changes"

            # Generate constraint-typed cues
            context = format_segments(all_segments, max_items=10)
            prompt = CONSTRAINT_AWARE_PROMPT.format(
                topic_hint=topic_hint,
                question=question_text,
                context=context,
            )
            output = llm_call(prompt)
            cues = parse_cues(output)

            if verbose:
                print(f"\n  Q{q['question_index']} LLM output:\n{output}")

            # Retrieve for each cue
            for cue in cues[:4]:
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

            retrieved_ids = {s.turn_id for s in all_segments}
            recall = compute_recall(retrieved_ids, source_ids)
            baseline_20 = compute_recall(
                {
                    s.turn_id
                    for s in store.search(
                        query_emb, top_k=20, conversation_id=conv_id
                    ).segments
                },
                source_ids,
            )

            num_cues = len(cues[:4])
            print(
                f"  Q{q['question_index']} [{q['category']}] "
                f"constraint_aware: {recall:.1%} "
                f"({len(retrieved_ids & source_ids)}/{len(source_ids)}) "
                f"[{num_cues} cues, {len(all_segments)} segs] "
                f"vs baseline_20={baseline_20:.1%}"
            )

            all_results.append(
                {
                    "dataset": ds_name,
                    "category": q["category"],
                    "question_index": q["question_index"],
                    "question": question_text,
                    "conversation_id": conv_id,
                    "method": "constraint_aware",
                    "recall": recall,
                    "hits": len(retrieved_ids & source_ids),
                    "total_source": len(source_ids),
                    "total_retrieved": len(all_segments),
                    "cues": cues[:4],
                    "baseline_r20": baseline_20,
                }
            )

    return all_results


# ===========================================================================
# Step 3c: Iterative constraint collection
# ===========================================================================

ITERATIVE_COLLECT_PROMPT = """\
You are helping collect ALL constraints/requirements from a conversation.

Question: {question}

WHAT WE HAVE SO FAR:
{context}

CONSTRAINTS IDENTIFIED SO FAR:
{constraints_summary}

ROUND {round_num}: Look at what we've found. Are there constraint types \
that probably exist in the conversation but haven't been found yet?

Think about:
- What KINDS of constraints would naturally come up in this discussion?
- Are there any constraint UPDATES (changes, resolutions, overrides)?
- Any constraints from OTHER people not yet represented?
- Any constraints mentioned LATER in the conversation as afterthoughts?

Generate 2 search cues targeting the most likely MISSING constraints. \
Write text that sounds like it would appear in the actual conversation.

If you believe we have found all relevant constraints, respond with DONE.

Format:
FOUND_SO_FAR: <brief list of constraint types already covered>
STILL_MISSING: <what types might be missing>
CUE: <text>
CUE: <text>
(or DONE if complete)"""


def step3c_iterative_collect(verbose: bool = False):
    """Iterative: retrieve, analyze, search for missing constraint types, repeat."""
    print("\n" + "=" * 70)
    print("STEP 3c: Iterative constraint collection")
    print("=" * 70)

    datasets = [
        (
            "puzzle",
            "segments_puzzle.npz",
            "questions_puzzle.json",
            ["logic_constraint"],
        ),
        (
            "synthetic",
            "segments_synthetic.npz",
            "questions_synthetic.json",
            ["completeness", "procedural"],
        ),
    ]

    all_results = []
    for ds_name, npz, qfile, cats in datasets:
        store = SegmentStore(DATA_DIR, npz)
        questions = load_questions(DATA_DIR / qfile, cats)

        print(f"\n--- {ds_name}: {cats} ---")
        for q in questions:
            conv_id = q["conversation_id"]
            question_text = q["question"]
            source_ids = set(q["source_chat_ids"])

            # Hop 0: cosine top-10
            query_emb = embed_text(question_text)
            hop0 = store.search(query_emb, top_k=10, conversation_id=conv_id)
            all_segments = list(hop0.segments)
            exclude = {s.index for s in all_segments}
            total_llm_calls = 0

            constraints_summary = "(none yet — first round)"

            for round_num in range(4):
                context = format_segments(all_segments, max_items=16, max_chars=300)
                prompt = ITERATIVE_COLLECT_PROMPT.format(
                    question=question_text,
                    context=context,
                    constraints_summary=constraints_summary,
                    round_num=round_num + 1,
                )
                output = llm_call(prompt)
                total_llm_calls += 1

                if "DONE" in output.upper() and "CUE:" not in output:
                    break

                cues = parse_cues(output)
                if not cues:
                    break

                # Extract constraint summary from output
                for line in output.split("\n"):
                    if line.strip().startswith("FOUND_SO_FAR:"):
                        constraints_summary = line.strip()[13:].strip()
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

                if verbose:
                    retrieved_ids = {s.turn_id for s in all_segments}
                    r = compute_recall(retrieved_ids, source_ids)
                    print(
                        f"    Round {round_num + 1}: {r:.1%} "
                        f"({len(retrieved_ids & source_ids)}/{len(source_ids)}) "
                        f"[{len(all_segments)} segs]"
                    )

            retrieved_ids = {s.turn_id for s in all_segments}
            recall = compute_recall(retrieved_ids, source_ids)
            baseline_20 = compute_recall(
                {
                    s.turn_id
                    for s in store.search(
                        query_emb, top_k=20, conversation_id=conv_id
                    ).segments
                },
                source_ids,
            )

            print(
                f"  Q{q['question_index']} [{q['category']}] "
                f"iterative: {recall:.1%} "
                f"({len(retrieved_ids & source_ids)}/{len(source_ids)}) "
                f"[{total_llm_calls} LLM calls, {len(all_segments)} segs] "
                f"vs baseline_20={baseline_20:.1%}"
            )

            all_results.append(
                {
                    "dataset": ds_name,
                    "category": q["category"],
                    "question_index": q["question_index"],
                    "method": "iterative_collect",
                    "recall": recall,
                    "hits": len(retrieved_ids & source_ids),
                    "total_source": len(source_ids),
                    "total_retrieved": len(all_segments),
                    "llm_calls": total_llm_calls,
                    "baseline_r20": baseline_20,
                }
            )

    return all_results


# ===========================================================================
# Step 3d: Broad then LLM rerank
# ===========================================================================

RERANK_PROMPT = """\
You are identifying which conversation segments contain constraints, \
requirements, or scheduling details relevant to answering this question.

Question: {question}

Below are {num_segments} conversation segments retrieved by broad search. \
For each segment, decide: does it contain a CONSTRAINT, REQUIREMENT, \
SCHEDULING DETAIL, UPDATE, or RESOLUTION that is relevant to answering \
the question?

SEGMENTS:
{segments_text}

List the turn IDs of segments that contain relevant constraints or \
requirements. Be INCLUSIVE — if a segment might contain useful information, \
include it.

Format:
RELEVANT: <comma-separated turn IDs>
Nothing else."""


def step3d_broad_then_rerank(verbose: bool = False):
    """Retrieve top-50 by cosine, then ask LLM to identify constraint-bearing turns."""
    print("\n" + "=" * 70)
    print("STEP 3d: Broad retrieval (top-50) + LLM constraint reranking")
    print("=" * 70)

    datasets = [
        (
            "puzzle",
            "segments_puzzle.npz",
            "questions_puzzle.json",
            ["logic_constraint"],
        ),
        (
            "synthetic",
            "segments_synthetic.npz",
            "questions_synthetic.json",
            ["completeness", "procedural"],
        ),
    ]

    all_results = []
    for ds_name, npz, qfile, cats in datasets:
        store = SegmentStore(DATA_DIR, npz)
        questions = load_questions(DATA_DIR / qfile, cats)

        print(f"\n--- {ds_name}: {cats} ---")
        for q in questions:
            conv_id = q["conversation_id"]
            question_text = q["question"]
            source_ids = set(q["source_chat_ids"])

            # Broad retrieval: top-50
            query_emb = embed_text(question_text)
            broad = store.search(query_emb, top_k=50, conversation_id=conv_id)
            broad_segments = list(broad.segments)

            # Format for LLM
            segments_text = ""
            for seg in sorted(broad_segments, key=lambda s: s.turn_id):
                segments_text += (
                    f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:300]}\n\n"
                )

            prompt = RERANK_PROMPT.format(
                question=question_text,
                num_segments=len(broad_segments),
                segments_text=segments_text,
            )
            output = llm_call(prompt)

            # Parse relevant turn IDs
            relevant_ids: set[int] = set()
            for line in output.strip().split("\n"):
                line = line.strip()
                if line.startswith("RELEVANT:"):
                    parts = line[9:].strip()
                    for p in parts.split(","):
                        p = p.strip()
                        try:
                            relevant_ids.add(int(p))
                        except ValueError:
                            pass

            # The "reranked" output: only relevant segments
            reranked_segments = [s for s in broad_segments if s.turn_id in relevant_ids]

            # But we also report recall metrics on the full broad set
            broad_ids = {s.turn_id for s in broad_segments}
            reranked_turn_ids = {s.turn_id for s in reranked_segments}

            broad_recall = compute_recall(broad_ids, source_ids)
            rerank_recall = compute_recall(reranked_turn_ids, source_ids)
            baseline_20 = compute_recall(
                {
                    s.turn_id
                    for s in store.search(
                        query_emb, top_k=20, conversation_id=conv_id
                    ).segments
                },
                source_ids,
            )

            print(
                f"  Q{q['question_index']} [{q['category']}] "
                f"broad_50: {broad_recall:.1%}, "
                f"reranked({len(reranked_segments)}): {rerank_recall:.1%}, "
                f"baseline_20: {baseline_20:.1%}"
            )

            if verbose:
                print(f"    Reranked hits: {sorted(reranked_turn_ids & source_ids)}")
                print(f"    Reranked misses: {sorted(source_ids - reranked_turn_ids)}")
                print(f"    Broad hits: {sorted(broad_ids & source_ids)}")

            all_results.append(
                {
                    "dataset": ds_name,
                    "category": q["category"],
                    "question_index": q["question_index"],
                    "method": "broad_then_rerank",
                    "broad_recall": broad_recall,
                    "rerank_recall": rerank_recall,
                    "num_reranked": len(reranked_segments),
                    "total_source": len(source_ids),
                    "baseline_r20": baseline_20,
                }
            )

    return all_results


# ===========================================================================
# Step 3e: Hybrid — broad cosine + constraint-aware cues + rerank
# ===========================================================================


def step3e_hybrid(verbose: bool = False):
    """Best of both: broad cosine pool + constraint-typed cues, then LLM rerank."""
    print("\n" + "=" * 70)
    print("STEP 3e: Hybrid — broad cosine + constraint cues + LLM rerank")
    print("=" * 70)

    datasets = [
        (
            "puzzle",
            "segments_puzzle.npz",
            "questions_puzzle.json",
            ["logic_constraint"],
        ),
        (
            "synthetic",
            "segments_synthetic.npz",
            "questions_synthetic.json",
            ["completeness", "procedural"],
        ),
    ]

    all_results = []
    for ds_name, npz, qfile, cats in datasets:
        store = SegmentStore(DATA_DIR, npz)
        questions = load_questions(DATA_DIR / qfile, cats)

        print(f"\n--- {ds_name}: {cats} ---")
        for q in questions:
            conv_id = q["conversation_id"]
            question_text = q["question"]
            source_ids = set(q["source_chat_ids"])

            # Phase 1: Broad cosine pool (top-30)
            query_emb = embed_text(question_text)
            broad = store.search(query_emb, top_k=30, conversation_id=conv_id)
            all_segments = list(broad.segments)
            exclude = {s.index for s in all_segments}

            # Phase 2: Constraint-typed cues to expand
            topic_hint = "planning or scheduling"
            if "desk" in question_text.lower() or "seat" in question_text.lower():
                topic_hint = "a desk/seating arrangement"
            elif "schedule" in question_text.lower() or "room" in question_text.lower():
                topic_hint = "conference room scheduling"

            context = format_segments(all_segments, max_items=12)
            prompt = CONSTRAINT_AWARE_PROMPT.format(
                topic_hint=topic_hint,
                question=question_text,
                context=context,
            )
            output = llm_call(prompt)
            cues = parse_cues(output)

            for cue in cues[:3]:
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

            # Phase 3: LLM rerank to identify constraint-bearing turns
            segments_text = ""
            for seg in sorted(all_segments, key=lambda s: s.turn_id):
                segments_text += (
                    f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:300]}\n\n"
                )

            rerank_prompt = RERANK_PROMPT.format(
                question=question_text,
                num_segments=len(all_segments),
                segments_text=segments_text,
            )
            rerank_output = llm_call(rerank_prompt)

            relevant_ids: set[int] = set()
            for line in rerank_output.strip().split("\n"):
                line = line.strip()
                if line.startswith("RELEVANT:"):
                    parts = line[9:].strip()
                    for p in parts.split(","):
                        p = p.strip()
                        try:
                            relevant_ids.add(int(p))
                        except ValueError:
                            pass

            # Results
            pool_ids = {s.turn_id for s in all_segments}
            pool_recall = compute_recall(pool_ids, source_ids)
            rerank_recall = compute_recall(relevant_ids, source_ids)
            baseline_20 = compute_recall(
                {
                    s.turn_id
                    for s in store.search(
                        query_emb, top_k=20, conversation_id=conv_id
                    ).segments
                },
                source_ids,
            )

            print(
                f"  Q{q['question_index']} [{q['category']}] "
                f"pool({len(all_segments)}): {pool_recall:.1%}, "
                f"reranked({len(relevant_ids)}): {rerank_recall:.1%}, "
                f"baseline_20: {baseline_20:.1%}"
            )

            all_results.append(
                {
                    "dataset": ds_name,
                    "category": q["category"],
                    "question_index": q["question_index"],
                    "method": "hybrid",
                    "pool_recall": pool_recall,
                    "rerank_recall": rerank_recall,
                    "num_pool": len(all_segments),
                    "num_reranked": len(relevant_ids),
                    "total_source": len(source_ids),
                    "baseline_r20": baseline_20,
                }
            )

    return all_results


# ===========================================================================
# Summary table
# ===========================================================================
def print_summary(
    baseline_results,
    constraint_results,
    iterative_results,
    rerank_results,
    hybrid_results,
):
    """Print a summary comparison table."""
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    # Group by (dataset, category, question_index)
    all_methods: dict[tuple, dict[str, float]] = defaultdict(dict)

    for r in baseline_results:
        key = (r["dataset"], r["category"], r["question_index"])
        all_methods[key][r["method"]] = r["recall"]

    for r in constraint_results:
        key = (r["dataset"], r["category"], r["question_index"])
        all_methods[key]["constraint_aware"] = r["recall"]
        all_methods[key]["baseline_r20"] = r["baseline_r20"]

    for r in iterative_results:
        key = (r["dataset"], r["category"], r["question_index"])
        all_methods[key]["iterative_collect"] = r["recall"]

    for r in rerank_results:
        key = (r["dataset"], r["category"], r["question_index"])
        all_methods[key]["broad_50"] = r["broad_recall"]
        all_methods[key]["broad50_reranked"] = r["rerank_recall"]

    for r in hybrid_results:
        key = (r["dataset"], r["category"], r["question_index"])
        all_methods[key]["hybrid_pool"] = r["pool_recall"]
        all_methods[key]["hybrid_reranked"] = r["rerank_recall"]

    # Print per-question
    methods_to_show = [
        "baseline_r20",
        "cosine_top30",
        "cosine_top50",
        "constraint_aware",
        "iterative_collect",
        "broad_50",
        "broad50_reranked",
        "hybrid_pool",
        "hybrid_reranked",
    ]

    print(
        f"\n{'Q':>4} {'Cat':>18} {'baseline':>8} {'cos30':>6} {'cos50':>6} "
        f"{'constr':>6} {'iter':>6} "
        f"{'brd50':>6} {'rrk50':>6} "
        f"{'hybP':>6} {'hybR':>6}"
    )
    print("-" * 100)

    cat_avgs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for key in sorted(all_methods.keys()):
        ds, cat, qi = key
        methods = all_methods[key]
        vals = []
        for m in methods_to_show:
            v = methods.get(m)
            vals.append(f"{v:.0%}" if v is not None else "  -")
            if v is not None:
                cat_avgs[cat][m].append(v)

        print(f"  {qi:>2} {cat:>18} " + " ".join(f"{v:>6}" for v in vals))

    # Category averages
    print(
        f"\n{'':>4} {'CATEGORY AVG':>18} {'baseline':>8} {'cos30':>6} {'cos50':>6} "
        f"{'constr':>6} {'iter':>6} "
        f"{'brd50':>6} {'rrk50':>6} "
        f"{'hybP':>6} {'hybR':>6}"
    )
    print("-" * 100)

    for cat in sorted(cat_avgs.keys()):
        vals = []
        for m in methods_to_show:
            scores = cat_avgs[cat].get(m, [])
            if scores:
                avg = sum(scores) / len(scores)
                vals.append(f"{avg:.0%}")
            else:
                vals.append("  -")
        print(f"  {'':>2} {cat:>18} " + " ".join(f"{v:>6}" for v in vals))


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step", type=int, default=0, help="Run specific step (1-5), 0=all"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    try:
        if args.step == 0 or args.step == 1:
            step1_analyze_baseline(verbose=args.verbose)

        if args.step == 0 or args.step == 2:
            step2_analyze_cue_damage(verbose=args.verbose)

        if args.step == 0 or args.step >= 3:
            baseline_results = step3a_expanded_baseline(verbose=args.verbose)
            constraint_results = step3b_constraint_aware(verbose=args.verbose)
            iterative_results = step3c_iterative_collect(verbose=args.verbose)
            rerank_results = step3d_broad_then_rerank(verbose=args.verbose)
            hybrid_results = step3e_hybrid(verbose=args.verbose)

            print_summary(
                baseline_results,
                constraint_results,
                iterative_results,
                rerank_results,
                hybrid_results,
            )

            # Save all results
            combined = {
                "baseline": baseline_results,
                "constraint_aware": constraint_results,
                "iterative_collect": iterative_results,
                "broad_then_rerank": rerank_results,
                "hybrid": hybrid_results,
            }
            out_path = RESULTS_DIR / "constraint_investigation.json"
            with open(out_path, "w") as f:
                json.dump(combined, f, indent=2)
            print(f"\nResults saved to {out_path}")

    finally:
        save_caches()
        print("\nCaches saved.")


if __name__ == "__main__":
    main()
