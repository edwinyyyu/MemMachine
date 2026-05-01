"""Speaker-prefix embedding study.

Tests whether prepending a speaker identifier to each segment's text before
embedding (e.g. "Caroline: I'm allergic to peanuts") changes retrieval
performance vs. the current practice of embedding just the raw text.

EventMemory already prepends sources to derivatives before embedding. Our
research SegmentStore does not. This script quantifies the delta.

Pipeline:
  Step 1: Re-embed each existing npz with speaker prefix, save to new npz.
  Step 2: Evaluate cosine baseline + v15_control + v2f_v2 on both original and
          prefixed variants at r@20 and r@50 (fair budget, no backfill).

Speakers used for the prefix:
  - synthetic / puzzle / advanced: role name ("user" / "assistant"). These
    conversations have no named speakers.
  - extended/LoCoMo: actual speaker names (Caroline, Melanie, ...) resolved
    via locomo10.json speaker_a / speaker_b. Only LoCoMo conversations are
    re-embedded to save budget.

Usage:
    uv run python speaker_prefix_eval.py --embed           # step 1 only
    uv run python speaker_prefix_eval.py --evaluate        # step 2 only
    uv run python speaker_prefix_eval.py --all             # both (default)
    uv run python speaker_prefix_eval.py --dataset synthetic
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    DATA_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI
from prompt_optimization import (
    META_V2F_V2_PROMPT,
    V15_CONTROL_PROMPT,
    _format_segments,
    _parse_cues,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
LOCOMO_RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"

# Budgets requested by the task
BUDGETS = [20, 50]

# Embedding batch size for re-embedding
EMBED_BATCH_SIZE = 100
EMBED_MAX_CHARS = 8000


# ---------------------------------------------------------------------------
# Dataset descriptors
# ---------------------------------------------------------------------------
# Each dataset has:
#   - source_npz: original embeddings
#   - output_npz: re-embedded with speaker prefix
#   - questions_file: the questions json
#   - question_filter: optional lambda to filter questions
#   - segment_filter: optional lambda (conversation_id) -> bool to restrict
#                     which segments get re-embedded (only used for extended)
#   - role_to_speaker: callable (conversation_id, role) -> speaker_name.
#                      This is what gets prepended before the text.
# ---------------------------------------------------------------------------


def _make_role_pass_through():
    """For synthetic/puzzle/advanced: the role itself is the speaker label."""

    def f(conversation_id: str, role: str) -> str:
        return role

    return f


def _make_locomo_role_to_speaker():
    """For extended/LoCoMo: map role -> actual speaker name.

    LoCoMo data was normalized to user=speaker_a, assistant=speaker_b during
    prep. Read locomo10.json and build the inverse mapping.
    """
    with open(LOCOMO_RAW_PATH) as f:
        raw = json.load(f)
    mapping: dict[str, dict[str, str]] = {}
    for entry in raw:
        sample_id = entry["sample_id"]
        conv_id = f"locomo_{sample_id}"
        conv = entry["conversation"]
        speaker_a = conv.get("speaker_a", "user")
        speaker_b = conv.get("speaker_b", "assistant")
        mapping[conv_id] = {"user": speaker_a, "assistant": speaker_b}

    def f(conversation_id: str, role: str) -> str:
        m = mapping.get(conversation_id)
        if m is None:
            return role  # fallback
        return m.get(role, role)

    return f


DATASETS = {
    "synthetic": {
        "source_npz": "segments_synthetic.npz",
        "output_npz": "segments_synthetic_prefixed.npz",
        "questions_file": "questions_synthetic.json",
        "question_filter": None,
        "segment_filter": None,
        "role_to_speaker": _make_role_pass_through(),
    },
    "puzzle": {
        "source_npz": "segments_puzzle.npz",
        "output_npz": "segments_puzzle_prefixed.npz",
        "questions_file": "questions_puzzle.json",
        "question_filter": None,
        "segment_filter": None,
        "role_to_speaker": _make_role_pass_through(),
    },
    "advanced": {
        "source_npz": "segments_advanced.npz",
        "output_npz": "segments_advanced_prefixed.npz",
        "questions_file": "questions_advanced.json",
        "question_filter": None,
        "segment_filter": None,
        "role_to_speaker": _make_role_pass_through(),
    },
    "locomo": {
        "source_npz": "segments_extended.npz",
        "output_npz": "segments_extended_locomo_prefixed.npz",
        "questions_file": "questions_extended.json",
        "question_filter": lambda q: q.get("benchmark") == "locomo",
        # Only re-embed LoCoMo conversations; drop BEAM from the output npz.
        "segment_filter": lambda cid: str(cid).startswith("locomo"),
        # Lazy init: constructed in main() so we don't read locomo10.json
        # unless needed.
        "role_to_speaker": None,
    },
}


# ---------------------------------------------------------------------------
# Caches — shared across this script
# ---------------------------------------------------------------------------
class PrefixEmbeddingCache(EmbeddingCache):
    """Dedicated cache for this experiment.

    Reads existing caches (to reuse raw-text embeddings from earlier work)
    PLUS the prefix cache. Writes only to prefix_embed_cache.json.
    """

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
            "prefix_embed_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "prefix_embed_cache.json"
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


class PrefixLLMCache(LLMCache):
    """Reads all existing LLM caches, writes to prefix-specific file."""

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
            "bestshot_llm_cache.json",
            "prefix_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "prefix_llm_cache.json"
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
# Step 1: re-embed
# ---------------------------------------------------------------------------
def _truncate(text: str) -> str:
    return text if len(text) <= EMBED_MAX_CHARS else text[:EMBED_MAX_CHARS]


def _embed_with_cache(
    client: OpenAI,
    cache: PrefixEmbeddingCache,
    texts: list[str],
) -> np.ndarray:
    """Embed a list of texts, reusing cache where possible."""
    embeddings: list[np.ndarray | None] = [None] * len(texts)
    pending_idx: list[int] = []
    pending_texts: list[str] = []
    for i, t in enumerate(texts):
        cached = cache.get(t)
        if cached is not None:
            embeddings[i] = cached
        else:
            pending_idx.append(i)
            pending_texts.append(_truncate(t))

    if pending_texts:
        for start in range(0, len(pending_texts), EMBED_BATCH_SIZE):
            batch = pending_texts[start : start + EMBED_BATCH_SIZE]
            batch_idx = pending_idx[start : start + EMBED_BATCH_SIZE]
            print(
                f"    Embedding batch {start // EMBED_BATCH_SIZE + 1} "
                f"({len(batch)} texts)...",
                flush=True,
            )
            response = client.embeddings.create(model=EMBED_MODEL, input=batch)
            for j, item in enumerate(response.data):
                emb = np.array(item.embedding, dtype=np.float32)
                cache.put(batch[j], emb)
                embeddings[batch_idx[j]] = emb
            cache.save()

    return np.array([e for e in embeddings], dtype=np.float32)


def embed_prefixed_dataset(
    dataset_name: str,
    client: OpenAI,
    cache: PrefixEmbeddingCache,
    force: bool = False,
) -> None:
    """Load source npz, build '{speaker}: {text}' strings, embed, save npz."""
    cfg = DATASETS[dataset_name]
    source_path = DATA_DIR / cfg["source_npz"]
    output_path = DATA_DIR / cfg["output_npz"]

    if output_path.exists() and not force:
        print(
            f"[{dataset_name}] Skipping — {output_path.name} exists "
            f"(use --force to rebuild)"
        )
        return

    print(f"[{dataset_name}] Loading {source_path.name}...")
    data = np.load(source_path, allow_pickle=True)
    conversation_ids = data["conversation_ids"]
    turn_ids = data["turn_ids"]
    roles = data["roles"]
    texts = data["texts"]

    # Apply segment filter if present (locomo: keep only LoCoMo conversations)
    if cfg["segment_filter"] is not None:
        mask = np.array([cfg["segment_filter"](str(cid)) for cid in conversation_ids])
        conversation_ids = conversation_ids[mask]
        turn_ids = turn_ids[mask]
        roles = roles[mask]
        texts = texts[mask]
        print(f"[{dataset_name}] Filtered to {mask.sum()} segments (of {len(mask)})")
    else:
        print(f"[{dataset_name}] {len(texts)} segments")

    # Build speaker label for each segment
    role_fn = cfg["role_to_speaker"]
    speakers = [
        role_fn(str(cid), str(role)) for cid, role in zip(conversation_ids, roles)
    ]

    # Build prefixed text
    prefixed_texts = [f"{speaker}: {text}" for speaker, text in zip(speakers, texts)]

    # Show a few examples
    print(f"[{dataset_name}] Examples of prefixed text:")
    for i in (0, 1, len(prefixed_texts) // 2, len(prefixed_texts) - 1):
        if 0 <= i < len(prefixed_texts):
            print(f"    [{i}] {prefixed_texts[i][:140]}")

    # Embed
    print(f"[{dataset_name}] Embedding {len(prefixed_texts)} segments...")
    embeddings = _embed_with_cache(client, cache, prefixed_texts)
    print(f"[{dataset_name}] Embeddings shape: {embeddings.shape}")

    # Save npz with the ORIGINAL (unprefixed) texts so downstream code
    # that displays text in LLM prompts doesn't see the prefix. Only the
    # embeddings change.
    np.savez(
        output_path,
        embeddings=embeddings,
        conversation_ids=np.array([str(c) for c in conversation_ids], dtype=object),
        turn_ids=np.array(turn_ids, dtype=np.int32),
        roles=np.array([str(r) for r in roles], dtype=object),
        texts=np.array([str(t) for t in texts], dtype=object),
    )
    print(f"[{dataset_name}] Saved to {output_path}")


# ---------------------------------------------------------------------------
# Step 2: evaluate
# ---------------------------------------------------------------------------
class ArchBase:
    """Shared scaffolding for the three 'architectures' we evaluate."""

    def __init__(
        self,
        store: SegmentStore,
        embedding_cache: PrefixEmbeddingCache,
        llm_cache: PrefixLLMCache,
        client: OpenAI,
    ):
        self.store = store
        self.embedding_cache = embedding_cache
        self.llm_cache = llm_cache
        self.client = client
        self.embed_calls = 0
        self.llm_calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        emb = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, emb)
        self.embed_calls += 1
        return emb

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

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()


class CosineArch(ArchBase):
    """Pure cosine top-K. No LLM calls."""

    name = "cosine"

    def retrieve(
        self, question: str, conversation_id: str, budget: int
    ) -> list[Segment]:
        query_emb = self.embed_text(question)
        result = self.store.search(
            query_emb, top_k=budget, conversation_id=conversation_id
        )
        return list(result.segments)


class PromptArch(ArchBase):
    """v15_control / v2f_v2 style: question top-10 + 2 cues top-10 each.

    Fair budget: we truncate to exactly `budget` segments (no cosine backfill).
    """

    def __init__(
        self,
        store: SegmentStore,
        embedding_cache: PrefixEmbeddingCache,
        llm_cache: PrefixLLMCache,
        client: OpenAI,
        prompt_template: str,
        name: str,
    ):
        super().__init__(store, embedding_cache, llm_cache, client)
        self.prompt_template = prompt_template
        self.name = name

    def retrieve(
        self, question: str, conversation_id: str, budget: int
    ) -> list[Segment]:
        # Hop 0: embed question, retrieve top-10
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments: list[Segment] = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        context = _format_segments(all_segments)
        context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + context
        prompt = self.prompt_template.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return all_segments[:budget]


def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


# Heuristic: does a question mention a proper-noun-like name?
_NAME_TOKEN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
# Words capitalized at start of sentence that are NOT names
_COMMON_WORDS = {
    "What",
    "When",
    "Where",
    "Why",
    "Who",
    "Which",
    "How",
    "Is",
    "Are",
    "Was",
    "Were",
    "Do",
    "Does",
    "Did",
    "Can",
    "Could",
    "Would",
    "Should",
    "Will",
    "Have",
    "Has",
    "Had",
    "The",
    "A",
    "An",
    "This",
    "That",
    "These",
    "Those",
    "My",
    "Your",
    "His",
    "Her",
    "Their",
    "Our",
    "If",
    "List",
    "Include",
    "Based",
    "Given",
    "Help",
    "Draft",
    "Create",
    "Tell",
    "Please",
    "Name",
    "Describe",
    "Explain",
    "Summarize",
    "Identify",
    "Find",
    "Show",
    "Dr",
    "Mr",
    "Mrs",
    "Ms",
    "Am",
    "Im",
}


def question_mentions_name(question: str) -> bool:
    """Rough heuristic: does the question contain a capitalized 3+-letter
    token that isn't a common sentence starter?"""
    for m in _NAME_TOKEN_RE.finditer(question):
        tok = m.group()
        if tok in _COMMON_WORDS:
            continue
        # Skip words at sentence start that are common verbs/pronouns
        start = m.start()
        if start == 0 and tok in _COMMON_WORDS:
            continue
        return True
    return False


def evaluate_one(arch, question: dict) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()

    # Run once at the largest budget, then truncate for smaller budgets.
    max_budget = max(BUDGETS)
    arch_segments = arch.retrieve(q_text, conv_id, max_budget)
    # Dedupe preserving order
    seen: set[int] = set()
    deduped: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            deduped.append(s)
            seen.add(s.index)
    arch_segments = deduped
    elapsed = time.time() - t0

    recalls: dict[str, float] = {}
    for b in BUDGETS:
        ids = {s.turn_id for s in arch_segments[:b]}
        recalls[f"r@{b}"] = compute_recall(ids, source_ids)

    return {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "mentions_name": question_mentions_name(q_text),
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "recalls": recalls,
        "total_retrieved": len(arch_segments),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    out: dict = {"n": n}
    for b in BUDGETS:
        vals = [r["recalls"][f"r@{b}"] for r in results]
        out[f"r@{b}"] = round(sum(vals) / n, 4)
    out["avg_retrieved"] = round(sum(r["total_retrieved"] for r in results) / n, 1)
    out["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 1)
    out["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
    return out


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out: dict[str, dict] = {}
    for cat, rs in sorted(by_cat.items()):
        out[cat] = summarize(rs)
    return out


def summarize_by_name_mention(results: list[dict]) -> dict[str, dict]:
    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_name["mentions_name" if r["mentions_name"] else "no_name"].append(r)
    out: dict[str, dict] = {}
    for k, rs in by_name.items():
        out[k] = summarize(rs)
    return out


def evaluate_dataset(
    dataset_name: str,
    client: OpenAI,
    embedding_cache: PrefixEmbeddingCache,
    llm_cache: PrefixLLMCache,
    verbose: bool = False,
    force: bool = False,
) -> dict:
    """Evaluate baseline + v15_control + v2f_v2 on both original and prefixed
    variants. Returns combined summary dict."""
    cfg = DATASETS[dataset_name]

    # Load questions
    with open(DATA_DIR / cfg["questions_file"]) as f:
        all_questions = json.load(f)
    if cfg["question_filter"] is not None:
        questions = [q for q in all_questions if cfg["question_filter"](q)]
    else:
        questions = list(all_questions)

    print(f"\n{'=' * 72}")
    print(f"DATASET: {dataset_name} | {len(questions)} questions")
    print(f"{'=' * 72}")

    # For locomo, the store should be loaded from the prefixed npz — which
    # only contains LoCoMo segments — so the original store for "locomo"
    # should also only contain LoCoMo conversations to keep cosine comparisons
    # apples-to-apples. Since the original segments_extended.npz contains BEAM
    # too, but we filter queries by conversation_id, that's actually OK: the
    # cosine search is already filtered to a specific conversation_id.
    # Still, to keep the candidate pool identical, we load the same segment
    # universe for both variants.

    variants = []

    # Variant A: original (raw text embedded)
    print("\n--- Variant A: RAW (no speaker prefix) ---")
    store_raw = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["source_npz"])
    variants.append(("raw", store_raw))

    # Variant B: prefixed
    print("\n--- Variant B: PREFIXED (speaker: text) ---")
    store_prefixed = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["output_npz"])
    variants.append(("prefixed", store_prefixed))

    archs_spec = [
        ("cosine", None),
        ("v15_control", V15_CONTROL_PROMPT),
        ("v2f_v2", META_V2F_V2_PROMPT),
    ]

    dataset_output: dict = {
        "dataset": dataset_name,
        "num_questions": len(questions),
        "variants": {},
    }

    for variant_label, store in variants:
        variant_out: dict = {}
        for arch_name, tmpl in archs_spec:
            print(f"\n  {variant_label} / {arch_name}")
            if arch_name == "cosine":
                arch = CosineArch(store, embedding_cache, llm_cache, client)
            else:
                arch = PromptArch(
                    store,
                    embedding_cache,
                    llm_cache,
                    client,
                    prompt_template=tmpl,
                    name=arch_name,
                )

            results = []
            for i, q in enumerate(questions):
                q_short = q["question"][:50]
                try:
                    row = evaluate_one(arch, q)
                    results.append(row)
                except Exception as e:
                    print(f"    ERROR on q[{i}]: {e}", flush=True)
                    import traceback

                    traceback.print_exc()
                if verbose:
                    r20 = row["recalls"].get("r@20", 0)
                    r50 = row["recalls"].get("r@50", 0)
                    print(
                        f"    [{i + 1}/{len(questions)}] "
                        f"r@20={r20:.3f} r@50={r50:.3f} | {q_short}"
                    )
                if (i + 1) % 10 == 0:
                    arch.save_caches()
                    sys.stdout.flush()
            arch.save_caches()

            summary = summarize(results)
            by_cat = summarize_by_category(results)
            by_name = summarize_by_name_mention(results)

            variant_out[arch_name] = {
                "summary": summary,
                "by_category": by_cat,
                "by_name_mention": by_name,
                "results": results,
            }

            print(
                f"    r@20={summary.get('r@20', 0):.4f} "
                f"r@50={summary.get('r@50', 0):.4f} "
                f"avg_llm={summary.get('avg_llm_calls', 0):.1f}"
            )

        dataset_output["variants"][variant_label] = variant_out

    # Compute deltas: prefixed - raw for each arch and budget
    deltas: dict = {}
    for arch_name, _ in archs_spec:
        raw_s = dataset_output["variants"]["raw"][arch_name]["summary"]
        pref_s = dataset_output["variants"]["prefixed"][arch_name]["summary"]
        deltas[arch_name] = {
            f"r@{b}": round(pref_s[f"r@{b}"] - raw_s[f"r@{b}"], 4) for b in BUDGETS
        }
        deltas[arch_name]["raw_r@20"] = raw_s["r@20"]
        deltas[arch_name]["prefixed_r@20"] = pref_s["r@20"]
        deltas[arch_name]["raw_r@50"] = raw_s["r@50"]
        deltas[arch_name]["prefixed_r@50"] = pref_s["r@50"]
    dataset_output["deltas"] = deltas

    # Per-category deltas (r@20)
    cat_deltas: dict = {}
    cats_raw = dataset_output["variants"]["raw"]["cosine"]["by_category"]
    for cat in cats_raw:
        cat_deltas[cat] = {}
        for arch_name, _ in archs_spec:
            r_by_cat = dataset_output["variants"]["raw"][arch_name]["by_category"].get(
                cat, {}
            )
            p_by_cat = dataset_output["variants"]["prefixed"][arch_name][
                "by_category"
            ].get(cat, {})
            if not r_by_cat or not p_by_cat:
                continue
            cat_deltas[cat][arch_name] = {
                "n": r_by_cat.get("n", 0),
                "raw_r@20": r_by_cat.get("r@20", 0),
                "prefixed_r@20": p_by_cat.get("r@20", 0),
                "delta_r@20": round(
                    p_by_cat.get("r@20", 0) - r_by_cat.get("r@20", 0), 4
                ),
                "raw_r@50": r_by_cat.get("r@50", 0),
                "prefixed_r@50": p_by_cat.get("r@50", 0),
                "delta_r@50": round(
                    p_by_cat.get("r@50", 0) - r_by_cat.get("r@50", 0), 4
                ),
            }
    dataset_output["category_deltas"] = cat_deltas

    # Name-mention deltas
    name_deltas: dict = {}
    for arch_name, _ in archs_spec:
        name_deltas[arch_name] = {}
        for slot in ("mentions_name", "no_name"):
            r = dataset_output["variants"]["raw"][arch_name]["by_name_mention"].get(
                slot, {}
            )
            p = dataset_output["variants"]["prefixed"][arch_name][
                "by_name_mention"
            ].get(slot, {})
            if not r or not p:
                continue
            name_deltas[arch_name][slot] = {
                "n": r.get("n", 0),
                "raw_r@20": r.get("r@20", 0),
                "prefixed_r@20": p.get("r@20", 0),
                "delta_r@20": round(p.get("r@20", 0) - r.get("r@20", 0), 4),
                "raw_r@50": r.get("r@50", 0),
                "prefixed_r@50": p.get("r@50", 0),
                "delta_r@50": round(p.get("r@50", 0) - r.get("r@50", 0), 4),
            }
    dataset_output["name_mention_deltas"] = name_deltas

    return dataset_output


def print_summary_table(per_dataset: dict[str, dict]) -> None:
    print("\n" + "=" * 100)
    print("SPEAKER-PREFIX SUMMARY — Overall (r@20, r@50)")
    print("=" * 100)
    header = (
        f"{'Dataset':<14} {'Arch':<14} "
        f"{'raw @20':>8} {'pre @20':>8} {'Δ @20':>8} "
        f"{'raw @50':>8} {'pre @50':>8} {'Δ @50':>8}"
    )
    print(header)
    print("-" * 100)
    for ds_name, ds in per_dataset.items():
        for arch_name in ("cosine", "v15_control", "v2f_v2"):
            d = ds["deltas"][arch_name]
            print(
                f"{ds_name:<14} {arch_name:<14} "
                f"{d['raw_r@20']:>8.4f} {d['prefixed_r@20']:>8.4f} "
                f"{d['r@20']:>+8.4f} "
                f"{d['raw_r@50']:>8.4f} {d['prefixed_r@50']:>8.4f} "
                f"{d['r@50']:>+8.4f}"
            )
        print("-" * 100)

    print("\n" + "=" * 100)
    print("SPEAKER-PREFIX SUMMARY — By name-mention in question (r@20)")
    print("=" * 100)
    print(
        f"{'Dataset':<14} {'Arch':<14} {'Slot':<16} "
        f"{'n':>4} {'raw @20':>8} {'pre @20':>8} {'Δ @20':>8} {'Δ @50':>8}"
    )
    print("-" * 100)
    for ds_name, ds in per_dataset.items():
        for arch_name in ("cosine", "v15_control", "v2f_v2"):
            nd = ds["name_mention_deltas"][arch_name]
            for slot in ("mentions_name", "no_name"):
                if slot not in nd:
                    continue
                d = nd[slot]
                print(
                    f"{ds_name:<14} {arch_name:<14} {slot:<16} "
                    f"{d['n']:>4} {d['raw_r@20']:>8.4f} "
                    f"{d['prefixed_r@20']:>8.4f} "
                    f"{d['delta_r@20']:>+8.4f} {d['delta_r@50']:>+8.4f}"
                )
        print("-" * 100)

    print("\n" + "=" * 100)
    print("SPEAKER-PREFIX SUMMARY — By category (r@20)")
    print("=" * 100)
    for ds_name, ds in per_dataset.items():
        print(f"\n  Dataset: {ds_name}")
        print(
            f"  {'Category':<40} {'Arch':<14} "
            f"{'n':>4} {'raw @20':>8} {'pre @20':>8} {'Δ @20':>8} "
            f"{'Δ @50':>8}"
        )
        print("  " + "-" * 96)
        for cat, arch_map in ds["category_deltas"].items():
            for arch_name in ("cosine", "v15_control", "v2f_v2"):
                if arch_name not in arch_map:
                    continue
                d = arch_map[arch_name]
                print(
                    f"  {cat:<40} {arch_name:<14} "
                    f"{d['n']:>4} {d['raw_r@20']:>8.4f} "
                    f"{d['prefixed_r@20']:>8.4f} "
                    f"{d['delta_r@20']:>+8.4f} {d['delta_r@50']:>+8.4f}"
                )
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker-prefix embedding study")
    parser.add_argument(
        "--embed", action="store_true", help="Step 1: build prefixed npz files"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Step 2: run retrieval evaluation"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both steps (default if neither flag set)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASETS.keys()) + ["all"],
        help="Which dataset(s) to process (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild npz / overwrite results"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.embed and not args.evaluate and not args.all:
        args.all = True
    if args.all:
        args.embed = True
        args.evaluate = True

    if args.dataset is None or args.dataset == "all":
        datasets = list(DATASETS.keys())
    else:
        datasets = [args.dataset]

    client = OpenAI(timeout=60.0)

    # Finalize locomo role_to_speaker if needed
    if DATASETS["locomo"]["role_to_speaker"] is None:
        DATASETS["locomo"]["role_to_speaker"] = _make_locomo_role_to_speaker()

    embedding_cache = PrefixEmbeddingCache()
    llm_cache = PrefixLLMCache()

    # ---------- Step 1: embed ----------
    if args.embed:
        for ds in datasets:
            embed_prefixed_dataset(ds, client, embedding_cache, force=args.force)
        embedding_cache.save()

    # ---------- Step 2: evaluate ----------
    if args.evaluate:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        per_dataset: dict[str, dict] = {}
        for ds in datasets:
            out_path = RESULTS_DIR / f"prefix_{ds}.json"
            if out_path.exists() and not args.force:
                print(f"[{ds}] Reading cached results from {out_path.name}")
                with open(out_path) as f:
                    per_dataset[ds] = json.load(f)
                continue

            ds_out = evaluate_dataset(
                ds,
                client,
                embedding_cache,
                llm_cache,
                verbose=args.verbose,
                force=args.force,
            )
            # Trim verbose per-question results before saving to keep the
            # file readable; still keep per-question recalls for later
            # re-analysis.
            with open(out_path, "w") as f:
                json.dump(ds_out, f, indent=2, default=str)
            print(f"[{ds}] Saved results to {out_path}")
            per_dataset[ds] = ds_out
            embedding_cache.save()
            llm_cache.save()

        # Print grand summary
        print_summary_table(per_dataset)

        combined_path = RESULTS_DIR / "prefix_all_summary.json"
        compact = {
            ds: {
                "deltas": d["deltas"],
                "name_mention_deltas": d["name_mention_deltas"],
                "category_deltas": d["category_deltas"],
                "num_questions": d["num_questions"],
            }
            for ds, d in per_dataset.items()
        }
        with open(combined_path, "w") as f:
            json.dump(compact, f, indent=2)
        print(f"\nSaved combined summary to {combined_path}")


if __name__ == "__main__":
    main()
