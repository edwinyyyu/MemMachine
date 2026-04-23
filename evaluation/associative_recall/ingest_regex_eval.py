"""Empirical evaluation of pure-regex ingestion-side alt-key tagging.

Runs LoCoMo-30 fair-backfill-style recall at K=20 and K=50 across four
conditions:
  1. cosine_baseline_no_altkeys   (query-only cosine, original index)
  2. cosine_baseline_with_altkeys (query-only cosine, augmented index)
  3. v2f_no_altkeys               (MetaV2f on original index)
  4. v2f_with_altkeys             (MetaV2f on augmented index)

Emits:
  results/ingestion_regex_empirical.md
  results/ingestion_regex_empirical.json

Usage: uv run python ingest_regex_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    EMBED_MODEL,
    Segment,
    SegmentStore,
    RetrievalResult,
)
from best_shot import (
    BestshotBase,
    BestshotEmbeddingCache,
    BestshotLLMCache,
    MetaV2f,
    _format_segments,
    _parse_cues,
    V2F_PROMPT,
)
from ingest_regex_altkeys import (
    AltKey,
    HEURISTIC_NAMES,
    generate_alt_keys_for_conversation,
    generate_all_alt_keys,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Augmented segment store: adds alt-key embeddings keyed to parent segments.
# ---------------------------------------------------------------------------
class AugmentedSegmentStore:
    """Thin wrapper around SegmentStore that exposes the same search() API but
    additionally considers alt-key embeddings via per-parent max-score.

    Exposes the same attributes the BestshotBase and MetaV2f code reads:
      - segments, conversation_ids, turn_ids, normalized_embeddings
      - get_neighbors, search
    """

    def __init__(self, base: SegmentStore, alt_keys: list[AltKey],
                 alt_embeddings: np.ndarray):
        self._base = base
        # Passthrough attrs
        self.segments = base.segments
        self.conversation_ids = base.conversation_ids
        self.turn_ids = base.turn_ids
        self.roles = base.roles
        self.texts = base.texts
        self.embeddings = base.embeddings
        self.normalized_embeddings = base.normalized_embeddings
        self._turn_index = base._turn_index
        self.get_neighbors = base.get_neighbors

        # Alt-keys
        self.alt_keys: list[AltKey] = alt_keys
        if len(alt_keys) == 0 or alt_embeddings.size == 0:
            self.alt_normalized = np.zeros((0, base.normalized_embeddings.shape[1]),
                                           dtype=np.float32)
            self.alt_parent_index = np.zeros(0, dtype=np.int64)
        else:
            norms = np.linalg.norm(alt_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self.alt_normalized = (alt_embeddings / norms).astype(np.float32)
            self.alt_parent_index = np.array(
                [k.parent_index for k in alt_keys], dtype=np.int64,
            )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        conversation_id: str | None = None,
        exclude_indices: set[int] | None = None,
    ) -> RetrievalResult:
        """Per-turn max of original-embedding and alt-key embeddings."""
        q = query_embedding.astype(np.float32)
        q_norm_val = max(float(np.linalg.norm(q)), 1e-10)
        q = q / q_norm_val

        # original similarities
        sims = self.normalized_embeddings @ q  # shape (N,)

        # alt-key similarities -> max per parent index
        if self.alt_normalized.shape[0] > 0:
            alt_sims = self.alt_normalized @ q  # shape (M,)
            # fast per-parent max using np.maximum.reduceat is tricky; use
            # direct scatter with np.maximum.at
            alt_per_parent = np.full(sims.shape, -np.inf, dtype=np.float32)
            np.maximum.at(alt_per_parent, self.alt_parent_index, alt_sims)
            # Take the elementwise max of original and alt-per-parent
            sims = np.maximum(sims, alt_per_parent)

        if conversation_id is not None:
            mask = self.conversation_ids == conversation_id
            sims = np.where(mask, sims, -1.0)

        if exclude_indices:
            for idx in exclude_indices:
                if 0 <= idx < sims.shape[0]:
                    sims[idx] = -1.0

        # Take top_k
        top_indices = np.argsort(sims)[::-1][:top_k]
        segments = [self.segments[i] for i in top_indices if sims[i] > -1.0]
        scores = [float(sims[i]) for i in top_indices if sims[i] > -1.0]
        return RetrievalResult(segments=segments, scores=scores)


# ---------------------------------------------------------------------------
# Embedding utility for alt-keys (uses BestshotEmbeddingCache for reuse).
# ---------------------------------------------------------------------------
def embed_texts_cached(
    client: OpenAI,
    cache: BestshotEmbeddingCache,
    texts: list[str],
    batch_size: int = 96,
) -> np.ndarray:
    """Embed a list of texts, using BestshotEmbeddingCache. Returns (N, D)."""
    if not texts:
        # infer dim from cache if possible
        return np.zeros((0, 1536), dtype=np.float32)

    results: list[np.ndarray | None] = [None] * len(texts)
    to_compute: list[tuple[int, str]] = []
    for i, t in enumerate(texts):
        t_stripped = t.strip()
        if not t_stripped:
            # zero vector
            results[i] = np.zeros(1536, dtype=np.float32)
            continue
        cached = cache.get(t_stripped)
        if cached is not None:
            results[i] = cached.astype(np.float32)
        else:
            to_compute.append((i, t_stripped))

    if to_compute:
        print(f"  Embedding {len(to_compute)} new alt-key texts (not in cache)...",
              flush=True)
    for start in range(0, len(to_compute), batch_size):
        batch = to_compute[start:start + batch_size]
        batch_texts = [t for _, t in batch]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
        for (i, t), embed_data in zip(batch, response.data):
            emb = np.array(embed_data.embedding, dtype=np.float32)
            cache.put(t, emb)
            results[i] = emb

    cache.save()
    return np.stack(results, axis=0)


# ---------------------------------------------------------------------------
# Fair-backfill recall
# ---------------------------------------------------------------------------
def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill_turn_ids(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    budget: int,
) -> set[int]:
    """Return the set of turn_ids making up the fair-backfill retrieval at K."""
    seen_idx: set[int] = set()
    unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen_idx:
            unique.append(s)
            seen_idx.add(s.index)

    at_k = unique[:budget]
    arch_indices = {s.index for s in at_k}

    if len(at_k) < budget:
        backfill = [s for s in cosine_segments if s.index not in arch_indices]
        needed = budget - len(at_k)
        at_k = at_k + backfill[:needed]
    at_k = at_k[:budget]

    return {s.turn_id for s in at_k}


# ---------------------------------------------------------------------------
# Eval a single condition on a question.
# ---------------------------------------------------------------------------
@dataclass
class ConditionResult:
    name: str
    per_question: list[dict]

    def by_K(self, K: int) -> list[float]:
        return [r[f"r@{K}"] for r in self.per_question]

    def mean_by_K(self, K: int) -> float:
        vals = self.by_K(K)
        return sum(vals) / len(vals) if vals else 0.0


def run_cosine_condition(
    store_for_search,
    embedder: "Embedder",
    questions: list[dict],
) -> ConditionResult:
    """Run pure cosine (no cues) on a given store."""
    out = []
    for q in questions:
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])

        q_emb = embedder.embed_text(q_text)
        max_K = max(BUDGETS)
        res = store_for_search.search(q_emb, top_k=max_K, conversation_id=conv_id)
        cos_segs = list(res.segments)

        row = {
            "conversation_id": conv_id,
            "category": q.get("category", "unknown"),
            "question_index": q.get("question_index", -1),
            "question": q_text,
            "source_chat_ids": sorted(source_ids),
        }
        for K in BUDGETS:
            # pure cosine: arch is cosine itself, backfill is identity
            at_k_ids = {s.turn_id for s in cos_segs[:K]}
            row[f"r@{K}"] = compute_recall(at_k_ids, source_ids)
        out.append(row)
    return ConditionResult(name="cosine", per_question=out)


def run_v2f_condition(
    store_for_search,
    embedder: "Embedder",
    questions: list[dict],
    arch_name: str,
) -> ConditionResult:
    """Run MetaV2f on the given store; compute fair-backfill recall."""
    # Build a MetaV2f that uses our store.
    arch = MetaV2f(store_for_search)  # type: ignore[arg-type]
    # Reuse the embedder's caches so we share cost with cosine condition.
    arch.embedding_cache = embedder.embedding_cache
    arch.llm_cache = embedder.llm_cache

    out = []
    for i, q in enumerate(questions):
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])

        arch.reset_counters()
        result = arch.retrieve(q_text, conv_id)
        arch_segs = list(result.segments)

        # Cosine top-max_K on same store for backfill
        q_emb = arch.embed_text(q_text)
        max_K = max(BUDGETS)
        cos_res = store_for_search.search(q_emb, top_k=max_K, conversation_id=conv_id)
        cos_segs = list(cos_res.segments)

        row = {
            "conversation_id": conv_id,
            "category": q.get("category", "unknown"),
            "question_index": q.get("question_index", -1),
            "question": q_text,
            "source_chat_ids": sorted(source_ids),
            "llm_calls": arch.llm_calls,
            "embed_calls": arch.embed_calls,
        }
        for K in BUDGETS:
            ids = fair_backfill_turn_ids(arch_segs, cos_segs, K)
            row[f"r@{K}"] = compute_recall(ids, source_ids)
        out.append(row)

        if (i + 1) % 5 == 0:
            arch.save_caches()
    arch.save_caches()
    return ConditionResult(name=arch_name, per_question=out)


# ---------------------------------------------------------------------------
# Tiny Embedder wrapper (for cosine conditions) reusing bestshot caches.
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, client: OpenAI):
        self.client = client
        self.embedding_cache = BestshotEmbeddingCache()
        self.llm_cache = BestshotLLMCache()
        self.embed_calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached.astype(np.float32)
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        emb = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, emb)
        self.embed_calls += 1
        return emb

    def save(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()


# ---------------------------------------------------------------------------
# Summary / table rendering
# ---------------------------------------------------------------------------
def summarize_conditions(conditions: dict[str, ConditionResult]) -> dict:
    """Overall + per-category summary."""
    # Overall
    overall = {}
    for name, cond in conditions.items():
        entry = {"n": len(cond.per_question)}
        for K in BUDGETS:
            entry[f"mean_r@{K}"] = round(cond.mean_by_K(K), 4)
        overall[name] = entry

    # Per-category (per condition)
    cats: set[str] = set()
    for cond in conditions.values():
        for r in cond.per_question:
            cats.add(r.get("category", "unknown"))

    per_cat: dict[str, dict[str, dict]] = {}
    for cat in sorted(cats):
        per_cat[cat] = {}
        for name, cond in conditions.items():
            rs = [r for r in cond.per_question if r.get("category") == cat]
            n = len(rs)
            entry = {"n": n}
            for K in BUDGETS:
                vals = [r[f"r@{K}"] for r in rs]
                entry[f"mean_r@{K}"] = round(sum(vals) / n, 4) if n else 0.0
            per_cat[cat][name] = entry

    return {"overall": overall, "per_category": per_cat}


def render_markdown(
    summary: dict,
    bloat: dict,
    fire_counts_corpus: dict,
    n_questions: int,
    n_source_segments: int,
    verdict: dict,
    fp_notes: dict,
    caveats: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Ingestion-Regex Alt-Key — Empirical Recall Test")
    lines.append("")
    lines.append(
        "This report tests whether applying the 7 cheap regex heuristics from "
        "`ingestion_predictability.md` §7 at INGESTION time — generating alt-keys "
        "that are embedded alongside the original segment — actually lifts "
        "retrieval recall, or whether the extra index mass washes out the signal."
    )
    lines.append("")
    lines.append(
        f"Benchmark: **LoCoMo-30** ({n_questions} questions, "
        f"{n_source_segments} segments in the LoCoMo-retrievable corpus)."
    )
    lines.append("")

    # --- Index bloat ---
    lines.append("## 1. Index bloat")
    lines.append("")
    lines.append(
        f"- Original segments in LoCoMo corpus: **{bloat['n_original']}**"
    )
    lines.append(
        f"- Alt-keys generated (deduped by text): **{bloat['n_altkeys']}**"
    )
    lines.append(
        f"- Bloat factor (alt / original): **{bloat['bloat_factor']:.2f}x**"
    )
    lines.append(
        f"- Fraction of segments that fire at least one heuristic: "
        f"**{bloat['fraction_any_fire']:.1%}**"
    )
    lines.append("")
    lines.append("### Per-heuristic fire counts on the full LoCoMo corpus")
    lines.append("")
    lines.append("| heuristic | fires | % of segments |")
    lines.append("|---|---:|---:|")
    n_seg = bloat["n_original"]
    for h in HEURISTIC_NAMES:
        fc = fire_counts_corpus.get(h, 0)
        pct = (fc / n_seg * 100.0) if n_seg else 0.0
        lines.append(f"| {h} | {fc} | {pct:.1f}% |")
    lines.append("")

    # --- Overall recall table ---
    lines.append("## 2. Overall recall")
    lines.append("")
    lines.append(
        "Fair-backfill recall on LoCoMo-30 at K=20 and K=50. For v2f conditions, "
        "any budget unused by v2f-found segments is backfilled by cosine on the same "
        "index, so all sides spend exactly K segments."
    )
    lines.append("")
    ov = summary["overall"]
    lines.append("| condition | mean r@20 | mean r@50 |")
    lines.append("|---|---:|---:|")
    for cond_name in [
        "cosine_no_altkeys", "cosine_with_altkeys",
        "v2f_no_altkeys", "v2f_with_altkeys",
    ]:
        s = ov.get(cond_name, {})
        lines.append(
            f"| {cond_name} | {s.get('mean_r@20', 0):.4f} | {s.get('mean_r@50', 0):.4f} |"
        )
    lines.append("")

    # --- Per-category ---
    lines.append("## 3. Per-category recall")
    lines.append("")
    lines.append(
        "LoCoMo-30's question-categories are just the three native LoCoMo "
        "categories (`locomo_single_hop`, `locomo_multi_hop`, "
        "`locomo_temporal`). The 22-category breakdown referenced in "
        "`ingestion_predictability.md` §6 is from the advanced_23q "
        "benchmark, not LoCoMo; this test uses the dataset the task "
        "specifies."
    )
    lines.append("")
    lines.append(
        "| category | n | cos_no @20 | cos_with @20 | v2f_no @20 | v2f_with @20 | "
        "cos_no @50 | cos_with @50 | v2f_no @50 | v2f_with @50 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cat, d in summary["per_category"].items():
        n_cat = d.get("cosine_no_altkeys", {}).get("n", 0)
        row = f"| {cat} | {n_cat} "
        for K in BUDGETS:
            for cond in [
                "cosine_no_altkeys", "cosine_with_altkeys",
                "v2f_no_altkeys", "v2f_with_altkeys",
            ]:
                v = d.get(cond, {}).get(f"mean_r@{K}", 0.0)
                row += f"| {v:.3f} "
        row += "|"
        lines.append(row)
    lines.append("")

    # --- Verdict ---
    lines.append("## 4. Verdict")
    lines.append("")
    lines.append(
        f"- v2f lift from alt-keys @ K=20: **{verdict['v2f_delta_at_20']:+.4f}**"
    )
    lines.append(
        f"- v2f lift from alt-keys @ K=50: **{verdict['v2f_delta_at_50']:+.4f}**"
    )
    lines.append(
        f"- cosine lift from alt-keys @ K=20: **{verdict['cos_delta_at_20']:+.4f}**"
    )
    lines.append(
        f"- cosine lift from alt-keys @ K=50: **{verdict['cos_delta_at_50']:+.4f}**"
    )
    lines.append("")
    gains = [x for x in verdict["category_deltas_v2f_at_20"] if x[1] > 0]
    losses = [x for x in verdict["category_deltas_v2f_at_20"] if x[1] < 0]
    ties = [x for x in verdict["category_deltas_v2f_at_20"] if x[1] == 0]
    lines.append("Per-category alt-key lift on v2f @ K=20 (sorted):")
    lines.append("")
    lines.append("| category | Δr@20 | sign |")
    lines.append("|---|---:|:---:|")
    for cat, delta in verdict["category_deltas_v2f_at_20"]:
        sign = "gain" if delta > 0 else ("loss" if delta < 0 else "tie")
        lines.append(f"| {cat} | {delta:+.4f} | {sign} |")
    lines.append("")
    lines.append(
        f"Summary: {len(gains)} category gained, {len(losses)} categories lost, "
        f"{len(ties)} tied."
    )
    lines.append("")
    lines.append(f"**One-line verdict: {verdict['one_liner']}**")
    lines.append("")

    # --- Caveats / FP notes ---
    lines.append("## 5. False-positive / precision notes")
    lines.append("")
    lines.append(
        "The §7 heuristics have high recall on known missed turns but unknown "
        "precision on the full corpus. Two heuristics are dominant bloat "
        "drivers; their fire-on-likely-irrelevant-turn rate is estimated below "
        "by sampling turns that fired them and are NOT in any gold source set."
    )
    lines.append("")
    lines.append("| heuristic | fires | turns NOT in any gold | est. FP share |")
    lines.append("|---|---:|---:|---:|")
    for h, stats in fp_notes.items():
        fires = stats["fires"]
        not_gold = stats["not_gold"]
        fp_share = (not_gold / fires) if fires else 0.0
        lines.append(f"| {h} | {fires} | {not_gold} | {fp_share:.1%} |")
    lines.append("")

    # --- Methodology caveats ---
    lines.append("## 6. Caveats")
    lines.append("")
    for c in caveats:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load data
    print("Loading SegmentStore and LoCoMo-30 questions...", flush=True)
    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_extended.npz")

    with open(DATA_DIR / "questions_extended.json") as f:
        all_qs = json.load(f)
    locomo_qs = [q for q in all_qs if q.get("benchmark") == "locomo"][:30]
    print(f"  LoCoMo-30: {len(locomo_qs)} questions, "
          f"{len(store.segments)} total segments (all benchmarks)", flush=True)

    # --- Generate alt-keys on the FULL LoCoMo sub-corpus (all locomo convs).
    locomo_conv_ids = {s.conversation_id for s in store.segments
                       if s.conversation_id.startswith("locomo_")}
    locomo_segments = [s for s in store.segments
                       if s.conversation_id in locomo_conv_ids]
    print(f"  LoCoMo sub-corpus: {len(locomo_segments)} segments across "
          f"{len(locomo_conv_ids)} conversations", flush=True)

    alt_keys_raw, fire_counts = generate_all_alt_keys(locomo_segments)
    # Dedup by text; keep first occurrence (retains parent_index + heuristic).
    seen_text: set[str] = set()
    alt_keys: list[AltKey] = []
    for k in alt_keys_raw:
        if k.text in seen_text:
            continue
        seen_text.add(k.text)
        alt_keys.append(k)
    print(f"  Alt-keys: {len(alt_keys_raw)} raw, {len(alt_keys)} deduped by text",
          flush=True)
    print(f"  Per-heuristic fire counts: {fire_counts}", flush=True)

    n_segments_with_any_fire = sum(
        1 for segs in [locomo_segments]
        for s in segs
        if any(
            k.parent_index == s.index for k in alt_keys_raw
        )
    )
    # faster: use a set
    parents_fired = {k.parent_index for k in alt_keys_raw}
    n_segments_with_any_fire = sum(1 for s in locomo_segments
                                   if s.index in parents_fired)
    fraction_any = n_segments_with_any_fire / len(locomo_segments)

    # --- Embed alt-keys ---
    client = OpenAI(timeout=60.0)
    embedder = Embedder(client)
    print(f"Embedding {len(alt_keys)} alt-keys...", flush=True)
    alt_embeddings = embed_texts_cached(
        client, embedder.embedding_cache, [k.text for k in alt_keys],
    )
    embedder.save()

    # --- Build augmented store ---
    aug_store = AugmentedSegmentStore(store, alt_keys, alt_embeddings)

    # --- Run four conditions ---
    conditions: dict[str, ConditionResult] = {}

    print("\n[1/4] cosine_no_altkeys ...", flush=True)
    conditions["cosine_no_altkeys"] = run_cosine_condition(store, embedder, locomo_qs)
    conditions["cosine_no_altkeys"].name = "cosine_no_altkeys"

    print("[2/4] cosine_with_altkeys ...", flush=True)
    conditions["cosine_with_altkeys"] = run_cosine_condition(
        aug_store, embedder, locomo_qs
    )
    conditions["cosine_with_altkeys"].name = "cosine_with_altkeys"

    print("[3/4] v2f_no_altkeys ...", flush=True)
    conditions["v2f_no_altkeys"] = run_v2f_condition(
        store, embedder, locomo_qs, "v2f_no_altkeys"
    )

    print("[4/4] v2f_with_altkeys ...", flush=True)
    conditions["v2f_with_altkeys"] = run_v2f_condition(
        aug_store, embedder, locomo_qs, "v2f_with_altkeys"
    )

    embedder.save()

    # --- Summarize ---
    summary = summarize_conditions(conditions)

    # --- Verdict ---
    cos_no_20 = summary["overall"]["cosine_no_altkeys"]["mean_r@20"]
    cos_with_20 = summary["overall"]["cosine_with_altkeys"]["mean_r@20"]
    cos_no_50 = summary["overall"]["cosine_no_altkeys"]["mean_r@50"]
    cos_with_50 = summary["overall"]["cosine_with_altkeys"]["mean_r@50"]
    v2f_no_20 = summary["overall"]["v2f_no_altkeys"]["mean_r@20"]
    v2f_with_20 = summary["overall"]["v2f_with_altkeys"]["mean_r@20"]
    v2f_no_50 = summary["overall"]["v2f_no_altkeys"]["mean_r@50"]
    v2f_with_50 = summary["overall"]["v2f_with_altkeys"]["mean_r@50"]

    v2f_delta_20 = v2f_with_20 - v2f_no_20
    v2f_delta_50 = v2f_with_50 - v2f_no_50
    cos_delta_20 = cos_with_20 - cos_no_20
    cos_delta_50 = cos_with_50 - cos_no_50

    # category deltas for v2f
    cat_deltas: list[tuple[str, float]] = []
    for cat, d in summary["per_category"].items():
        vn = d.get("v2f_no_altkeys", {}).get("mean_r@20", 0.0)
        vw = d.get("v2f_with_altkeys", {}).get("mean_r@20", 0.0)
        cat_deltas.append((cat, vw - vn))
    cat_deltas.sort(key=lambda x: x[1], reverse=True)

    if v2f_delta_20 > 0.01 and v2f_delta_50 > 0.01:
        verdict_line = "pure-regex ingestion is worth keeping"
    elif v2f_delta_20 < -0.01 or v2f_delta_50 < -0.01:
        verdict_line = "pure-regex ingestion is not worth keeping"
    else:
        verdict_line = "pure-regex ingestion is borderline"

    verdict = {
        "v2f_delta_at_20": v2f_delta_20,
        "v2f_delta_at_50": v2f_delta_50,
        "cos_delta_at_20": cos_delta_20,
        "cos_delta_at_50": cos_delta_50,
        "category_deltas_v2f_at_20": cat_deltas,
        "one_liner": verdict_line,
    }

    # --- False-positive estimate ---
    # For each heuristic: segments whose index fired that heuristic, minus those
    # whose turn_id is in any gold set in the question list.
    gold_by_conv: dict[str, set[int]] = defaultdict(set)
    for q in locomo_qs:
        gold_by_conv[q["conversation_id"]].update(q["source_chat_ids"])

    fires_by_heur: dict[str, list[int]] = {h: [] for h in HEURISTIC_NAMES}
    for k in alt_keys_raw:
        fires_by_heur[k.heuristic].append(k.parent_index)
    fp_notes: dict[str, dict] = {}
    seg_by_index = {s.index: s for s in locomo_segments}
    for h, indices in fires_by_heur.items():
        fires = len(indices)
        not_gold = 0
        for idx in indices:
            seg = seg_by_index.get(idx)
            if seg is None:
                continue
            golds = gold_by_conv.get(seg.conversation_id, set())
            if seg.turn_id not in golds:
                not_gold += 1
        fp_notes[h] = {"fires": fires, "not_gold": not_gold}

    bloat = {
        "n_original": len(locomo_segments),
        "n_altkeys_raw": len(alt_keys_raw),
        "n_altkeys": len(alt_keys),
        "bloat_factor": len(alt_keys) / max(len(locomo_segments), 1),
        "fraction_any_fire": fraction_any,
    }

    caveats = [
        "Only 30 questions, from one LoCoMo conversation. Deltas below "
        "~0.01 are within noise.",
        "Regex `by (day)` is interpreted as \"by <weekday>\" (case-insensitive, "
        "word-boundary); other interpretations are plausible.",
        "The `rare_entity` heuristic as specified emits every capitalized-not-"
        "sentence-initial token plus number/version tokens. True corpus-rare "
        "filtering is not possible at streaming ingest time and is NOT applied "
        "here; this means rare_entity is intentionally noisy, matching the "
        "analysis's §9 caveat.",
        "The anaphoric heuristic fires only on the pronoun set given in §7. "
        "A handful of `ingestion_predictability.md` examples (e.g. \"Yeah, 16 "
        "weeks...\") were labeled anaphoric in that report but do NOT match "
        "the literal first-token pronoun spec; our implementation follows "
        "the spec.",
        "Alt-key scoring is per-parent-max over original + alt-key "
        "embeddings. This is strictly non-decreasing in cosine for any single "
        "cosine query on any segment — so cosine_with_altkeys cannot DROP "
        "pure recall, only raise it or tie. Losses at fixed K come from "
        "non-gold segments being boosted past gold ones.",
        "v2f uses MetaV2f (gpt-5-mini) via the shared best-shot LLM cache. "
        "No new LLM calls are required for previously-cached questions.",
    ]

    # --- Render ---
    md = render_markdown(
        summary, bloat, fire_counts, len(locomo_qs), len(locomo_segments),
        verdict, fp_notes, caveats,
    )

    md_path = RESULTS_DIR / "ingestion_regex_empirical.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\nWrote {md_path}", flush=True)

    # --- Raw JSON ---
    raw = {
        "meta": {
            "benchmark": "locomo_30q",
            "n_questions": len(locomo_qs),
            "n_segments_full_corpus": len(store.segments),
            "n_segments_locomo_corpus": len(locomo_segments),
            "locomo_conversations": sorted(locomo_conv_ids),
            "elapsed_s": round(time.time() - t0, 2),
        },
        "index_bloat": bloat,
        "heuristic_fire_counts": fire_counts,
        "false_positive_notes": fp_notes,
        "summary": summary,
        "verdict": {
            "v2f_delta_at_20": v2f_delta_20,
            "v2f_delta_at_50": v2f_delta_50,
            "cos_delta_at_20": cos_delta_20,
            "cos_delta_at_50": cos_delta_50,
            "category_deltas_v2f_at_20": cat_deltas,
            "one_liner": verdict_line,
        },
        "per_question": {
            name: cond.per_question for name, cond in conditions.items()
        },
    }
    json_path = RESULTS_DIR / "ingestion_regex_empirical.json"
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"Wrote {json_path}", flush=True)

    # --- Console summary ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for cond in ["cosine_no_altkeys", "cosine_with_altkeys",
                 "v2f_no_altkeys", "v2f_with_altkeys"]:
        s = summary["overall"][cond]
        print(f"  {cond:28s} r@20={s['mean_r@20']:.4f}  r@50={s['mean_r@50']:.4f}")
    print(f"\n  bloat_factor={bloat['bloat_factor']:.2f}x  "
          f"frac_any_fire={bloat['fraction_any_fire']:.1%}")
    print(f"  v2f Δr@20={v2f_delta_20:+.4f}  v2f Δr@50={v2f_delta_50:+.4f}")
    print(f"  verdict: {verdict_line}")


if __name__ == "__main__":
    main()
