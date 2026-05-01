"""Router study: can a cheap router dispatch across specialists and beat v2f-only?

Five specialists (all existing, no modifications):
  v2f                  — MetaV2f (baseline)
  v2f_plus_types       — V2fPlusTypesVariant (strict Pareto winner at K=50)
  type_enumerated      — TypeEnumeratedVariant (logic_constraint specialist)
  chain                — GoalChainRetriever(use_scratchpad=True)
  v2f_style_explicit   — DomainAgnosticVariant (cross-dataset winner)

Six routers compared:
  v2f_only            — baseline control (always picks v2f)
  oracle              — ceiling; picks per-category best specialist
  llm_router_mini     — gpt-5-mini classifier
  llm_router_nano     — gpt-5-nano classifier
  keyword_router      — zero-LLM regex rules
  embedding_router    — zero-LLM nearest-exemplar

Runs on 4 datasets (locomo_30q, synthetic_19q, puzzle_16q, advanced_23q)
at K=20 and K=50.

Per-question recall tables are loaded from existing specialist result files.
v2f_style_explicit's K=50 is re-computed here (K=50 was not previously saved).

Usage:
    uv run python router_study.py
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
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
from domain_agnostic import (
    NEUTRAL_HEADER,
    V2F_STYLE_EXPLICIT_PROMPT,
    DomainAgnosticVariant,
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS = (20, 50)
CLASSIFIER_VERSION = "v2"  # bumps to invalidate router-LLM cache entries

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

# Specialists available to all routers
SPECIALISTS = (
    "v2f",
    "v2f_plus_types",
    "type_enumerated",
    "chain",
    "v2f_style_explicit",
)

# One-line descriptions used in LLM router prompts
SPECIALIST_DESCRIPTIONS = {
    "v2f": "fact lookup / simple direct questions (single entity, time, place)",
    "v2f_plus_types": "multi-constraint / aggregate questions asking for sets matching several properties",
    "type_enumerated": "logical-constraint satisfaction questions (which X meets criteria A, B, C)",
    "chain": "sequential / chain-following / evolving-terminology / proactive (draft/plan) questions",
    "v2f_style_explicit": "open-ended description or cross-domain questions that benefit from casual first-person retrieval style",
}

# Paths to cached per-question results
RESULT_PATTERNS = {
    "v2f": "fairbackfill_meta_v2f_{ds}.json",
    "v2f_plus_types": "type_enum_v2f_plus_types_{ds}.json",
    "type_enumerated": "type_enum_type_enumerated_{ds}.json",
    "chain": "goal_chain_chain_with_scratchpad_{ds}.json",
    "v2f_style_explicit": "domain_agnostic_v2f_style_explicit_{ds}.json",
}


# ---------------------------------------------------------------------------
# Utility: question key
# ---------------------------------------------------------------------------
def qkey(conv_id: str, q_index: int) -> tuple[str, int]:
    return (conv_id, q_index)


def load_questions(ds_name: str) -> list[dict]:
    cfg = DATASETS[ds_name]
    with open(DATA_DIR / cfg["questions"]) as f:
        qs = json.load(f)
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    if cfg["max_questions"]:
        qs = qs[: cfg["max_questions"]]
    return qs


# ---------------------------------------------------------------------------
# Specialist recall tables (per-question, per-K, per-dataset)
# ---------------------------------------------------------------------------
def load_specialist_table(ds: str, specialist: str) -> dict[tuple, dict]:
    """Returns map (conv_id, q_index) -> {'r@20': float, 'r@50': float, 'category': str}."""
    path = RESULTS_DIR / RESULT_PATTERNS[specialist].format(ds=ds)
    out: dict[tuple, dict] = {}
    with open(path) as f:
        data = json.load(f)
    for r in data["results"]:
        key = qkey(r["conversation_id"], r["question_index"])
        fb = r["fair_backfill"]
        entry = {
            "category": r["category"],
            "r@20": fb["arch_r@20"],
            "baseline_r@20": fb["baseline_r@20"],
        }
        if "arch_r@50" in fb:
            entry["r@50"] = fb["arch_r@50"]
            entry["baseline_r@50"] = fb["baseline_r@50"]
        out[key] = entry
    return out


# ---------------------------------------------------------------------------
# Fill missing K=50 for v2f_style_explicit by re-running the variant
# ---------------------------------------------------------------------------
def fair_backfill_recall(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    source_ids: set[int],
    budget: int,
) -> float:
    seen: set[int] = set()
    arch_unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            arch_unique.append(s)
            seen.add(s.index)
    arch_at_K = arch_unique[:budget]
    arch_indices = {s.index for s in arch_at_K}
    if len(arch_at_K) < budget:
        backfill = [s for s in cosine_segments if s.index not in arch_indices]
        arch_at_K = arch_at_K + backfill[: budget - len(arch_at_K)]
    arch_at_K = arch_at_K[:budget]
    arch_ids = {s.turn_id for s in arch_at_K}
    if not source_ids:
        return 1.0
    return len(arch_ids & source_ids) / len(source_ids)


def ensure_v2f_style_has_k50(verbose: bool = True) -> None:
    """Fill in K=50 recall for v2f_style_explicit if missing."""
    for ds in DATASETS:
        path = RESULTS_DIR / RESULT_PATTERNS["v2f_style_explicit"].format(ds=ds)
        with open(path) as f:
            data = json.load(f)
        if data["results"] and "arch_r@50" in data["results"][0]["fair_backfill"]:
            continue
        if verbose:
            print(
                f"[v2f_style_explicit] Re-running {ds} to compute K=50...", flush=True
            )
        cfg = DATASETS[ds]
        store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
        variant = DomainAgnosticVariant(
            store,
            prompt_template=V2F_STYLE_EXPLICIT_PROMPT,
            context_header=NEUTRAL_HEADER,
        )
        questions = load_questions(ds)
        new_results = []
        for i, q in enumerate(questions):
            q_text = q["question"]
            conv_id = q["conversation_id"]
            source_ids = set(q["source_chat_ids"])
            variant.reset_counters()
            res = variant.retrieve(q_text, conv_id)
            arch_segs = res.segments
            query_emb = variant.embed_text(q_text)
            cosine_res = store.search(
                query_emb, top_k=max(BUDGETS), conversation_id=conv_id
            )
            cosine_segs = list(cosine_res.segments)
            fb = {}
            for K in BUDGETS:
                b = fair_backfill_recall([], cosine_segs, source_ids, K)
                a = fair_backfill_recall(arch_segs, cosine_segs, source_ids, K)
                fb[f"baseline_r@{K}"] = round(b, 4)
                fb[f"arch_r@{K}"] = round(a, 4)
                fb[f"delta_r@{K}"] = round(a - b, 4)
            new_results.append(
                {
                    "conversation_id": conv_id,
                    "category": q.get("category", "unknown"),
                    "question_index": q.get("question_index", i),
                    "question": q_text,
                    "source_chat_ids": sorted(source_ids),
                    "num_source_turns": len(source_ids),
                    "total_arch_retrieved": len(arch_segs),
                    "embed_calls": variant.embed_calls,
                    "llm_calls": variant.llm_calls,
                    "fair_backfill": fb,
                    "metadata": {},
                }
            )
            if (i + 1) % 5 == 0:
                variant.save_caches()
        variant.save_caches()
        data["results"] = new_results
        # Overwrite
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        if verbose:
            print(f"  -> wrote K=50 results into {path.name}", flush=True)


# ---------------------------------------------------------------------------
# Router LLM / embedding cache
# ---------------------------------------------------------------------------
class RouterLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "router_study_llm_cache.json"
        self._cache: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
        self._dirty = False
        self._lock = threading.Lock()

    def _key(self, model: str, prompt: str) -> str:
        seed = f"{CLASSIFIER_VERSION}|{model}|{prompt}"
        return hashlib.sha256(seed.encode()).hexdigest()

    def get(self, model: str, prompt: str) -> str | None:
        return self._cache.get(self._key(model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        with self._lock:
            self._cache[self._key(model, prompt)] = response
            self._dirty = True

    def save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            tmp = self.cache_file.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(self._cache, f)
            tmp.replace(self.cache_file)
            self._dirty = False


class RouterEmbeddingCache(EmbeddingCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Reuse any existing cached embeddings from the broad pool
        sources = [
            "embedding_cache.json",
            "bestshot_embedding_cache.json",
            "domain_agnostic_embedding_cache.json",
            "type_enum_embedding_cache.json",
            "goal_chain_embedding_cache.json",
            "router_study_embedding_cache.json",
        ]
        self._cache: dict[str, list[float]] = {}
        for name in sources:
            p = self.cache_dir / name
            if p.exists():
                try:
                    with open(p) as f:
                        self._cache.update(json.load(f))
                except Exception:
                    pass
        self.cache_file = self.cache_dir / "router_study_embedding_cache.json"
        self._new: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        with self._lock:
            self._cache[key] = embedding.tolist()
            self._new[key] = embedding.tolist()

    def save(self) -> None:
        with self._lock:
            if not self._new:
                return
            existing: dict = {}
            if self.cache_file.exists():
                try:
                    with open(self.cache_file) as f:
                        existing = json.load(f)
                except Exception:
                    existing = {}
            existing.update(self._new)
            self._new = {}
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
CLIENT = OpenAI(timeout=60.0)
_llm_cache = RouterLLMCache()
_emb_cache = RouterEmbeddingCache()

ROUTER_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "calls": 0,
    "cache_hits": 0,
}


def _embed(text: str) -> np.ndarray:
    text = text.strip()
    if not text:
        return np.zeros(1536, dtype=np.float32)
    cached = _emb_cache.get(text)
    if cached is not None:
        return cached
    resp = CLIENT.embeddings.create(model=EMBED_MODEL, input=[text])
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    _emb_cache.put(text, emb)
    return emb


LLM_ROUTER_PROMPT = """\
You are a router that picks ONE retrieval specialist for a question about a
private conversation history. Consider only the question. Reply with a
single label from this list:

- v2f: simple fact lookup (one entity, time, or place)
- v2f_plus_types: multi-criteria question asking for items matching several properties
- type_enumerated: strict logical-constraint satisfaction across multiple hard constraints
- chain: sequential / step-by-step / evolving-terminology / draft-help / proactive task
- v2f_style_explicit: open-ended or descriptive question benefiting from casual retrieval style

Question: {question}

Reply with only the label, nothing else."""


def _llm_route(question: str, model: str) -> str:
    prompt = LLM_ROUTER_PROMPT.format(question=question.strip())
    cached = _llm_cache.get(model, prompt)
    if cached is not None:
        ROUTER_USAGE["cache_hits"] += 1
        return _parse_label(cached)
    try:
        resp = CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,  # gpt-5 reasoning consumes tokens
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        ROUTER_USAGE["calls"] += 1
        if usage is not None:
            ROUTER_USAGE["input_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            ROUTER_USAGE["output_tokens"] += int(
                getattr(usage, "completion_tokens", 0) or 0
            )
    except Exception as e:
        print(f"    LLM route error ({model}): {e}", flush=True)
        text = "v2f"
    _llm_cache.put(model, prompt, text)
    return _parse_label(text)


def _parse_label(text: str) -> str:
    text = text.strip().lower()
    # Strip markdown / code fences / quotes
    text = text.strip("`").strip("\"'")
    # Pick first non-empty line
    first = ""
    for line in text.splitlines():
        line = line.strip().strip("`").strip("\"'")
        if line:
            first = line
            break
    # Remove common prefixes
    first = re.sub(r"^(label:|answer:|specialist:|route:|choice:)\s*", "", first)
    first = first.strip().strip(".,;:`\"'").strip()
    # Longest-match first: order labels by length desc to avoid v2f
    # absorbing v2f_plus_types or v2f_style_explicit.
    labels_by_len = sorted(SPECIALISTS, key=len, reverse=True)
    # Exact match
    for label in labels_by_len:
        if first == label:
            return label
    # Starts-with match
    for label in labels_by_len:
        if first.startswith(label):
            return label
    # Substring anywhere in the full text (longest first)
    for label in labels_by_len:
        if label in text:
            return label
    return "v2f"


def route_v2f_only(question: str) -> str:
    return "v2f"


def route_llm_mini(question: str) -> str:
    return _llm_route(question, "gpt-5-mini")


def route_llm_nano(question: str) -> str:
    return _llm_route(question, "gpt-5-nano")


# Keyword rules, applied in order. First matching rule wins.
KEYWORD_RULES: list[tuple[re.Pattern, str]] = [
    # Chain / proactive / sequential / evolving-terminology cues
    (
        re.compile(
            r"\b(draft|prepare|plan for|help me (?:with|draft|prepare)|write me|compose)\b",
            re.IGNORECASE,
        ),
        "chain",
    ),
    (
        re.compile(
            r"\b(step[- ]by[- ]step|sequence of|order of|in order|in the order|chain of|progression|chronolog)\b",
            re.IGNORECASE,
        ),
        "chain",
    ),
    (
        re.compile(
            r"\b(current|latest|most recent) (?:status|state|version|plan|alias)\b",
            re.IGNORECASE,
        ),
        "chain",
    ),
    (
        re.compile(
            r"\b(history of|evolution of|evolv|renamed|now called|used to call|aka|alias)\b",
            re.IGNORECASE,
        ),
        "chain",
    ),
    # Logical-constraint satisfaction
    (
        re.compile(
            r"\b(all|every|which|who).*\b(satisf(?:y|ies)|meet(?:s)?|match(?:es)?|fit(?:s)?|agree|accommodat)\b",
            re.IGNORECASE,
        ),
        "type_enumerated",
    ),
    (
        re.compile(r"\b(under|subject to|given) (?:the )?constraint", re.IGNORECASE),
        "type_enumerated",
    ),
    # Multi-criterion set aggregations
    (
        re.compile(
            r"\b(list|enumerate|name all|how many.*\band\b|what are all|both.+and\b)\b",
            re.IGNORECASE,
        ),
        "v2f_plus_types",
    ),
    (
        re.compile(
            r"\b(every|all of the).+\b(with|having|that (?:are|were|have))\b", re.IGNORECASE
        ),
        "v2f_plus_types",
    ),
    # Descriptive / open-ended
    (
        re.compile(
            r"\b(describe|summarize|overview|what did.+?talk about|discuss(?:ed|ion))\b",
            re.IGNORECASE,
        ),
        "v2f_style_explicit",
    ),
]


def route_keyword(question: str) -> str:
    for pat, lab in KEYWORD_RULES:
        if pat.search(question):
            return lab
    return "v2f_plus_types"  # Pareto floor at K=50


# Exemplars for embedding router (hand-picked phrasings per specialist).
EMBEDDING_EXEMPLARS: dict[str, list[str]] = {
    "v2f": [
        "When did Alice go to the gym?",
        "Who attended the meeting on Friday?",
        "Where did Bob buy the car?",
        "What movie did Carol watch last week?",
    ],
    "v2f_plus_types": [
        "List all the restaurants Alice tried that were Italian and had outdoor seating.",
        "Which options are both under twenty dollars and gluten-free?",
        "Name every person who RSVPed yes and mentioned they can drive.",
        "Which candidates had experience in both Python and SQL?",
    ],
    "type_enumerated": [
        "Which meeting slots satisfy Alice being available, Bob being available, and the room being free?",
        "Which plan meets the budget constraint, the timeline constraint, and the staffing constraint?",
        "Which apartment fits under the rent cap, allows pets, and is near transit?",
        "Which schedule works given no conflicts and all stated preferences?",
    ],
    "chain": [
        "Walk me through the sequence of steps they took to debug the issue.",
        "Draft me a follow-up email based on what they discussed.",
        "Help me plan the next move given what they agreed on.",
        "What is the current alias of the project that was previously called Orion?",
    ],
    "v2f_style_explicit": [
        "Describe the vibe of their conversation about weekend plans.",
        "Summarize how they felt about the new policy.",
        "What did they generally talk about regarding travel?",
        "Give me an overview of the themes from the whole conversation.",
    ],
}


def _normalized(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0:
        return v
    return v / n


def _build_exemplar_embeddings() -> dict[str, np.ndarray]:
    ex: dict[str, np.ndarray] = {}
    for lab, phrases in EMBEDDING_EXEMPLARS.items():
        vs = [_normalized(_embed(p)) for p in phrases]
        ex[lab] = np.stack(vs, axis=0)
    return ex


_EXEMPLAR_EMB: dict[str, np.ndarray] | None = None


def route_embedding(question: str) -> str:
    global _EXEMPLAR_EMB
    if _EXEMPLAR_EMB is None:
        _EXEMPLAR_EMB = _build_exemplar_embeddings()
    q = _normalized(_embed(question))
    best_label = "v2f"
    best_sim = -1.0
    for lab, M in _EXEMPLAR_EMB.items():
        # Mean-of-top-2 cosine sim
        sims = M @ q
        topk = np.sort(sims)[-2:]
        avg = float(topk.mean())
        if avg > best_sim:
            best_sim = avg
            best_label = lab
    return best_label


# ---------------------------------------------------------------------------
# Oracle: per-category best specialist (determined from data)
# ---------------------------------------------------------------------------
def build_oracle_table(tables: dict, K: int) -> dict[str, str]:
    """For each (dataset, category), picks the specialist with highest
    average recall across that category's questions.

    Returns: cat_to_specialist (aggregated across all datasets).
    We use a global category table — this mimics "oracle knows the right
    specialist for each category" as the ceiling.
    """
    # Aggregate: (category) -> specialist -> list of recalls
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ds, per_spec in tables.items():
        for spec, table in per_spec.items():
            for k, entry in table.items():
                cat = entry["category"]
                key = f"r@{K}"
                if key in entry:
                    agg[cat][spec].append(entry[key])
    picks: dict[str, str] = {}
    for cat, per_spec in agg.items():
        best_spec, best_mean = None, -1.0
        for spec, vals in per_spec.items():
            m = sum(vals) / max(1, len(vals))
            if m > best_mean:
                best_mean = m
                best_spec = spec
        picks[cat] = best_spec or "v2f"
    return picks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_router(
    router_name: str,
    router_fn,  # callable(question_text, question_dict, K) -> specialist label
    tables: dict[str, dict[str, dict]],
    oracle_cat_to_spec: dict[int, dict[str, str]] | None = None,
) -> dict:
    """Runs router across all datasets at K=20 and K=50.

    tables: dataset -> specialist -> question_key -> {r@20, r@50, category}
    oracle_cat_to_spec: optional per-K dict category -> specialist (oracle path).
    router_fn takes (question_text, qdict, K) and returns a specialist label.

    Returns per-dataset per-K recall summary, routing distribution, and
    per-category recall.
    """
    summary: dict = {"router": router_name, "per_dataset": {}, "per_category": {}}
    all_per_q: list[dict] = []

    per_category_agg: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    routing_correct_counts = dict.fromkeys(BUDGETS, 0)
    routing_total = 0

    for ds_name in DATASETS:
        questions = load_questions(ds_name)
        per_ds: dict = {
            "n": len(questions),
            "routing": {K: defaultdict(int) for K in BUDGETS},
        }
        recalls: dict[int, list[float]] = {K: [] for K in BUDGETS}
        baselines: dict[int, list[float]] = {K: [] for K in BUDGETS}
        per_cat_ds: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for q in questions:
            routing_total += 1
            q_text = q["question"]
            conv_id = q["conversation_id"]
            q_idx = q.get("question_index", -1)
            key = qkey(conv_id, q_idx)
            cat = q.get("category", "unknown")

            per_q_routed: dict[int, str] = {}
            for K in BUDGETS:
                chosen = router_fn(q_text, q, K)
                if chosen not in SPECIALISTS:
                    chosen = "v2f"
                per_q_routed[K] = chosen
                per_ds["routing"][K][chosen] += 1

                # Look up specialist recall; fall back to v2f if missing entry
                spec_table = tables[ds_name].get(chosen, {})
                entry = spec_table.get(key)
                if entry is None or f"r@{K}" not in entry:
                    entry = tables[ds_name]["v2f"].get(key, {})
                r = entry.get(f"r@{K}", 0.0)
                base = entry.get(f"baseline_r@{K}", 0.0)
                recalls[K].append(r)
                baselines[K].append(base)
                per_cat_ds[cat][K].append(r)
                per_category_agg[cat][K].append(r)

                if oracle_cat_to_spec is not None:
                    target_spec = oracle_cat_to_spec[K].get(cat, "v2f")
                    if chosen == target_spec:
                        routing_correct_counts[K] += 1

            all_per_q.append(
                {
                    "dataset": ds_name,
                    "question_index": q_idx,
                    "conversation_id": conv_id,
                    "question": q_text[:120],
                    "category": cat,
                    "routed_to": per_q_routed,
                }
            )

        ds_entry: dict = {
            "n": len(questions),
            "routing": {str(K): dict(per_ds["routing"][K]) for K in BUDGETS},
        }
        for K in BUDGETS:
            vals = recalls[K]
            base_vals = baselines[K]
            n = max(1, len(vals))
            ds_entry[f"mean_r@{K}"] = round(sum(vals) / n, 4)
            ds_entry[f"baseline_r@{K}"] = round(sum(base_vals) / n, 4)
            ds_entry[f"delta_r@{K}"] = round((sum(vals) - sum(base_vals)) / n, 4)
        # Per-category for this dataset
        ds_entry["per_category"] = {}
        for cat, per_k in per_cat_ds.items():
            ds_entry["per_category"][cat] = {
                f"mean_r@{K}": round(sum(per_k[K]) / max(1, len(per_k[K])), 4)
                for K in BUDGETS
            } | {"n": len(per_k[BUDGETS[0]])}
        summary["per_dataset"][ds_name] = ds_entry

    # Aggregated per-category across datasets
    cats_out: dict = {}
    for cat, per_k in per_category_agg.items():
        cats_out[cat] = {
            f"mean_r@{K}": round(sum(per_k[K]) / max(1, len(per_k[K])), 4)
            for K in BUDGETS
        } | {"n": len(per_k[BUDGETS[0]])}
    summary["per_category"] = cats_out

    # Overall (across all questions in all datasets)
    overall_recalls: dict[int, list[float]] = {K: [] for K in BUDGETS}
    overall_base: dict[int, list[float]] = {K: [] for K in BUDGETS}
    for ds_name in summary["per_dataset"]:
        ds = summary["per_dataset"][ds_name]
        # we need per-question; reaggregate via per_category_agg — but that's across datasets.
        # Cleaner: sum weighted by n.
    # Compute weighted overall means
    for K in BUDGETS:
        total_sum = 0.0
        total_base = 0.0
        total_n = 0
        for ds_name in summary["per_dataset"]:
            ds = summary["per_dataset"][ds_name]
            total_sum += ds[f"mean_r@{K}"] * ds["n"]
            total_base += ds[f"baseline_r@{K}"] * ds["n"]
            total_n += ds["n"]
        if total_n == 0:
            continue
        summary[f"overall_r@{K}"] = round(total_sum / total_n, 4)
        summary[f"overall_baseline_r@{K}"] = round(total_base / total_n, 4)
        summary[f"overall_delta_r@{K}"] = round((total_sum - total_base) / total_n, 4)

    if oracle_cat_to_spec is not None and routing_total > 0:
        summary["routing_accuracy_vs_oracle"] = {
            f"r@{K}": round(routing_correct_counts[K] / routing_total, 4)
            for K in BUDGETS
        }
    summary["per_question"] = all_per_q
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()

    ensure_v2f_style_has_k50()

    # Load all specialist tables
    tables: dict[str, dict[str, dict]] = {}
    for ds in DATASETS:
        tables[ds] = {}
        for spec in SPECIALISTS:
            tables[ds][spec] = load_specialist_table(ds, spec)

    # Build oracle per-K
    oracle_tables = {K: build_oracle_table(tables, K) for K in BUDGETS}
    print("\n=== Oracle per-category picks ===")
    for K in BUDGETS:
        print(f"K={K}:")
        for cat, spec in sorted(oracle_tables[K].items()):
            print(f"  {cat:28s} -> {spec}")

    # Routers (signature: fn(question, qdict, K) -> specialist label)
    def route_oracle(question: str, qdict: dict, K: int) -> str:
        # Per-K oracle: picks best specialist for the target budget.
        cat = qdict.get("category", "unknown")
        return oracle_tables[K].get(cat, "v2f")

    routers: dict[str, object] = {
        "v2f_only": lambda q, qd, K: route_v2f_only(q),
        "oracle": route_oracle,
        "llm_router_mini": lambda q, qd, K: route_llm_mini(q),
        "llm_router_nano": lambda q, qd, K: route_llm_nano(q),
        "keyword_router": lambda q, qd, K: route_keyword(q),
        "embedding_router": lambda q, qd, K: route_embedding(q),
    }

    all_router_summaries: dict = {}
    for name, fn in routers.items():
        print(f"\n=== Running router: {name} ===", flush=True)
        t_start = time.time()
        summary = evaluate_router(
            name,
            fn,
            tables,
            oracle_cat_to_spec=oracle_tables,
        )
        all_router_summaries[name] = summary
        print(
            f"  overall r@20={summary.get('overall_r@20')}, "
            f"r@50={summary.get('overall_r@50')}, "
            f"elapsed={time.time() - t_start:.1f}s",
            flush=True,
        )
        # Save caches as we go
        _llm_cache.save()
        _emb_cache.save()

    _llm_cache.save()
    _emb_cache.save()

    # Router token usage
    router_tokens = dict(ROUTER_USAGE)
    # If the router ran entirely from cache, estimate typical token cost
    # from the prompt template length + observed average response length.
    # Approx rule: 4 chars ≈ 1 token.
    prompt_body_len = len(LLM_ROUTER_PROMPT)
    router_tokens["avg_prompt_chars"] = prompt_body_len
    router_tokens["est_input_tokens_per_call"] = int(prompt_body_len / 4)
    # Reasoning models consume hidden reasoning tokens (~100-500 typical).
    # For a 1-label output (< 30 chars), observed output usually ~20-300
    # tokens including hidden reasoning for gpt-5-mini.
    router_tokens["est_output_tokens_per_call_mini"] = 150  # typical
    router_tokens["est_output_tokens_per_call_nano"] = 80
    # Per-question router cost (each of mini and nano is called 2x per
    # question because router_fn receives K=20 and K=50; both identical
    # prompts so second call hits cache). So 1 external call per model.
    router_tokens["external_calls_per_q_mini"] = 1
    router_tokens["external_calls_per_q_nano"] = 1

    out_json = {
        "oracle_tables": oracle_tables,
        "router_summaries": all_router_summaries,
        "router_usage": router_tokens,
        "elapsed_s": round(time.time() - t0, 2),
    }
    json_path = RESULTS_DIR / "router_study.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"\nSaved raw numbers: {json_path}")

    md = render_markdown(all_router_summaries, oracle_tables, router_tokens, tables)
    md_path = RESULTS_DIR / "router_study.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved report: {md_path}")

    # Also print the headline table
    print("\n" + "=" * 72)
    print("ROUTER STUDY SUMMARY")
    print("=" * 72)
    hdr = f"{'Router':<22s} {'r@20':>8s} {'d@20':>7s} {'r@50':>8s} {'d@50':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for name, s in all_router_summaries.items():
        print(
            f"{name:<22s} "
            f"{s.get('overall_r@20', 0):>8.4f} "
            f"{s.get('overall_delta_r@20', 0):>+7.4f} "
            f"{s.get('overall_r@50', 0):>8.4f} "
            f"{s.get('overall_delta_r@50', 0):>+7.4f}"
        )

    # Per-dataset comparison at K=20
    print("\nPer-dataset r@20:")
    print(f"{'Router':<22s}  " + "  ".join(f"{ds:<14s}" for ds in DATASETS))
    for name, s in all_router_summaries.items():
        row = f"{name:<22s}  "
        for ds in DATASETS:
            v = s["per_dataset"].get(ds, {}).get("mean_r@20", 0)
            row += f"{v:<14.4f}  "
        print(row)


def render_markdown(
    summaries: dict, oracle_tables: dict, router_tokens: dict, tables: dict
) -> str:
    lines: list[str] = []
    lines.append("# Router Study\n")
    lines.append(
        "Question: can a cheap router dispatch across specialists and beat "
        "v2f-only across many question categories?\n"
    )

    lines.append("## Setup\n")
    lines.append(
        "- 5 specialists: v2f (baseline), v2f_plus_types (K=50 Pareto), "
        "type_enumerated (logic_constraint), chain (chain_with_scratchpad), "
        "v2f_style_explicit (cross-dataset winner).\n"
    )
    lines.append(
        "- 6 routers: v2f_only (control), oracle (ceiling via per-category "
        "best specialist), llm_router_mini (gpt-5-mini), llm_router_nano "
        "(gpt-5-nano), keyword_router (regex rules), embedding_router "
        "(nearest-exemplar cosine).\n"
    )
    lines.append(
        "- 4 datasets (88 questions total) × {K=20, K=50}. Per-question "
        "specialist recalls loaded from existing cached per-question result "
        "files; no specialist code was modified.\n"
    )

    # Headline table
    lines.append("## Overall recall\n")
    lines.append(
        "| Router | r@20 | Δ@20 | r@50 | Δ@50 | routing accuracy vs oracle @20 / @50 |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, s in summaries.items():
        acc = s.get("routing_accuracy_vs_oracle", {})
        a20 = acc.get("r@20", None)
        a50 = acc.get("r@50", None)
        acc_str = (
            f"{a20:.2f} / {a50:.2f}" if (a20 is not None and a50 is not None) else "—"
        )
        lines.append(
            f"| {name} | {s.get('overall_r@20', 0):.4f} | "
            f"{s.get('overall_delta_r@20', 0):+.4f} | "
            f"{s.get('overall_r@50', 0):.4f} | "
            f"{s.get('overall_delta_r@50', 0):+.4f} | "
            f"{acc_str} |"
        )

    # Per-dataset table at K=20 and K=50
    for K in BUDGETS:
        lines.append(f"\n## Per-dataset r@{K}\n")
        lines.append("| Router | " + " | ".join(DATASETS.keys()) + " |")
        lines.append("|---|" + "---|" * len(DATASETS))
        for name, s in summaries.items():
            row = [name]
            for ds in DATASETS:
                v = s["per_dataset"].get(ds, {}).get(f"mean_r@{K}", 0)
                row.append(f"{v:.4f}")
            lines.append("| " + " | ".join(row) + " |")

    # Per-category breakdown at K=20
    all_cats: set[str] = set()
    for s in summaries.values():
        all_cats.update(s["per_category"].keys())
    cats_sorted = sorted(all_cats)
    for K in BUDGETS:
        lines.append(f"\n## Per-category r@{K}\n")
        lines.append("| Category | n | " + " | ".join(summaries.keys()) + " |")
        lines.append("|---|---|" + "---|" * len(summaries))
        for cat in cats_sorted:
            ns = [
                summaries[name]["per_category"].get(cat, {}).get("n", 0)
                for name in summaries
            ]
            n = max(ns) if ns else 0
            row = [cat, str(n)]
            for name in summaries:
                v = (
                    summaries[name]["per_category"]
                    .get(cat, {})
                    .get(f"mean_r@{K}", None)
                )
                row.append("—" if v is None else f"{v:.4f}")
            lines.append("| " + " | ".join(row) + " |")

    # Oracle tables
    lines.append("\n## Oracle per-category picks\n")
    for K in BUDGETS:
        lines.append(f"\n### K={K}\n")
        lines.append("| Category | Specialist |")
        lines.append("|---|---|")
        for cat, spec in sorted(oracle_tables[K].items()):
            lines.append(f"| {cat} | {spec} |")

    # Top helps/hurts per cheap router vs v2f_only, per K
    v2f = summaries.get("v2f_only", {})
    for name in (
        "llm_router_mini",
        "llm_router_nano",
        "keyword_router",
        "embedding_router",
        "oracle",
    ):
        s = summaries.get(name)
        if not s:
            continue
        lines.append(
            f"\n## {name}: categories where routing helps / hurts vs v2f_only\n"
        )
        for K in BUDGETS:
            lines.append(f"\n### K={K}\n")
            diffs = []
            for cat in sorted(s["per_category"].keys()):
                r_here = s["per_category"][cat].get(f"mean_r@{K}", 0)
                r_v2f = v2f["per_category"].get(cat, {}).get(f"mean_r@{K}", 0)
                diffs.append(
                    (
                        cat,
                        s["per_category"][cat].get("n", 0),
                        r_here,
                        r_v2f,
                        r_here - r_v2f,
                    )
                )
            helps = [d for d in sorted(diffs, key=lambda x: -x[4]) if d[4] > 0][:3]
            hurts = [d for d in sorted(diffs, key=lambda x: x[4]) if d[4] < 0][:3]
            lines.append("- Top helps:")
            if not helps:
                lines.append("  - (none)")
            for cat, n, rh, rv, d in helps:
                lines.append(f"  - {cat} (n={n}): {rv:.3f} → {rh:.3f} ({d:+.3f})")
            lines.append("- Top hurts:")
            if not hurts:
                lines.append("  - (none)")
            for cat, n, rh, rv, d in hurts:
                lines.append(f"  - {cat} (n={n}): {rv:.3f} → {rh:.3f} ({d:+.3f})")

    # Router tokens
    lines.append("\n## Router cost\n")
    lines.append(
        f"- Router LLM calls this run (cache misses only): {router_tokens['calls']}\n"
    )
    lines.append(f"- Cache hits: {router_tokens['cache_hits']}\n")
    lines.append(f"- Fresh input tokens this run: {router_tokens['input_tokens']}\n")
    lines.append(f"- Fresh output tokens this run: {router_tokens['output_tokens']}\n")
    lines.append(
        f"- Estimated per-question cost (cold cache): "
        f"~{router_tokens.get('est_input_tokens_per_call', 0)} input tokens + "
        f"~{router_tokens.get('est_output_tokens_per_call_mini', 0)} output "
        f"tokens for gpt-5-mini (including reasoning tokens); "
        f"~{router_tokens.get('est_output_tokens_per_call_nano', 0)} for nano.\n"
    )
    lines.append(
        "- At gpt-5-mini list pricing (~$0.25/M input, $2/M output), "
        "one mini router call is ~$0.0004/question; one nano call ~$0.0002. "
        "88 questions ≈ $0.04 for mini, $0.02 for nano.\n"
    )
    lines.append(
        "- Keyword + embedding routers use $0 of new LLM budget per "
        "question (embedding router uses one cached cosine lookup).\n"
    )

    # Verdict
    lines.append("\n## Verdict\n")
    v2f = summaries.get("v2f_only", {})
    oracle = summaries.get("oracle", {})
    lines.append(
        f"- **v2f_only overall r@20 = {v2f.get('overall_r@20', 0):.4f}, "
        f"r@50 = {v2f.get('overall_r@50', 0):.4f}.**\n"
    )
    lines.append(
        f"- **Oracle ceiling r@20 = {oracle.get('overall_r@20', 0):.4f} "
        f"(Δ vs v2f = {oracle.get('overall_r@20', 0) - v2f.get('overall_r@20', 0):+.4f}), "
        f"r@50 = {oracle.get('overall_r@50', 0):.4f} "
        f"(Δ = {oracle.get('overall_r@50', 0) - v2f.get('overall_r@50', 0):+.4f}).**\n"
    )
    for K in BUDGETS:
        best_cheap_name = None
        best_cheap_r = v2f.get(f"overall_r@{K}", 0)
        for name in (
            "llm_router_mini",
            "llm_router_nano",
            "keyword_router",
            "embedding_router",
        ):
            r = summaries.get(name, {}).get(f"overall_r@{K}", 0)
            if r > best_cheap_r:
                best_cheap_r = r
                best_cheap_name = name
        if best_cheap_name is not None:
            delta = best_cheap_r - v2f.get(f"overall_r@{K}", 0)
            lines.append(
                f"- **Best cheap router at K={K}: {best_cheap_name} = "
                f"{best_cheap_r:.4f} (Δ vs v2f = {delta:+.4f}).**\n"
            )
        else:
            lines.append(f"- **No cheap router beat v2f_only at K={K}.**\n")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
