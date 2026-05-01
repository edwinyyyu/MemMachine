"""Earliest-anchor sweep for C4 query-gated retrieval.

Research question: at a fixed positional budget (beyond top-K relevance),
does splitting as (first=X, last=Y) with X>0 outperform pure-latest (first=0, last=N)?

Setup:
  - top_k = 10 (fixed, same as round 6B)
  - Positional total budgets: 30, optional sanity checks at 20 and 10
  - Splits at budget=30: (0,30), (5,25), (10,20), (15,15), (20,10), (30,0)
  - Deduplication with top-K via union semantics

Reuses embedding + LLM caches from round 6B where prompts are byte-identical.
Writes own caches to this directory.

Grading: deterministic keyword (same as round 6B) for consistency.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import openai
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROUND6B_DIR = HERE.parent  # .../round6/compaction/
sys.path.insert(0, str(ROUND6B_DIR))

from evaluate import (  # type: ignore
    READER_PROMPT,
    Budget,
    JSONCache,
    _sha,
    grade_answer,
)
from strategies import render_entries  # type: ignore

EVAL_ROOT = HERE.parents[4]  # evaluation/
load_dotenv(EVAL_ROOT / ".env")

MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"
PRICE_PER_LLM = 0.0025
PRICE_PER_EMBED = 0.0000002

# Reuse round6B caches (read-through); write updates to local cache dir
ROUND6B_LLM_CACHE = ROUND6B_DIR / "cache" / "llm_cache.json"
ROUND6B_EMBED_CACHE = ROUND6B_DIR / "cache" / "embed_cache.json"

LOCAL_CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_LLM_CACHE = LOCAL_CACHE_DIR / "llm_cache.json"
LOCAL_EMBED_CACHE = LOCAL_CACHE_DIR / "embed_cache.json"

SCENARIOS_FILE = ROUND6B_DIR / "scenarios.json"
SCENARIOS_EXT_FILE = HERE / "scenarios_extended.json"
RESULTS_FILE = RESULTS_DIR / "sweep_results.json"

MAX_LLM = 300
STOP_LLM = int(MAX_LLM * 0.80)  # 240
MAX_EMBED = 30


# ----------------------------- merged cache -----------------------------


class MergedJSONCache:
    """Read from round 6B cache first, then local; write to local."""

    def __init__(self, local_path: Path, round6b_path: Path) -> None:
        self._local = JSONCache(local_path)
        self._ro: dict[str, Any] = {}
        if round6b_path.exists():
            try:
                with open(round6b_path) as f:
                    self._ro = json.load(f)
            except Exception:
                self._ro = {}
        self.ro_hits = 0
        self.local_hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        v = self._local.get(key)
        if v is not None:
            self.local_hits += 1
            return v
        v = self._ro.get(key)
        if v is not None:
            self.ro_hits += 1
            return v
        return None

    def put(self, key: str, value: Any) -> None:
        self._local.put(key, value)
        self.misses += 1

    def save(self) -> None:
        self._local.save()


# ----------------------------- c4 variant -----------------------------


def c4_split_anchor(
    entries: list[dict[str, Any]],
    query: str,
    entry_embeds: np.ndarray,
    query_embed: np.ndarray,
    top_k: int,
    first_k: int,
    last_k: int,
) -> tuple[str, dict]:
    """C4 with a split positional anchor: top_k relevance + first_k earliest + last_k latest (union)."""
    a = entry_embeds / (np.linalg.norm(entry_embeds, axis=1, keepdims=True) + 1e-9)
    q = query_embed / (np.linalg.norm(query_embed) + 1e-9)
    sims = a @ q
    topk_idx = np.argsort(-sims)[:top_k]
    topk_set = set(int(i) for i in topk_idx)

    n = len(entries)
    first_idx = set(range(min(first_k, n)))
    last_idx = set(range(max(0, n - last_k), n))

    selected = sorted(topk_set | first_idx | last_idx)
    kept = [entries[i] for i in selected]
    rendered = (
        f"# Query-gated view for: {query!r}\n"
        f"# Showing top-{top_k} relevant + first {first_k} + last {last_k} entries "
        f"({len(kept)} unique of {n})\n" + render_entries(kept)
    )
    meta = {
        "kept": len(kept),
        "elided": n - len(kept),
        "top_k": top_k,
        "first_k": first_k,
        "last_k": last_k,
        "llm_calls": 0,
        "embed_calls": 0,
    }
    return rendered, meta


# ----------------------------- LLM + embed wrappers -----------------------------


def llm_call(
    client: openai.OpenAI,
    cache: MergedJSONCache,
    budget: Budget,
    prompt: str,
    model: str = MODEL,
    reasoning_effort: str = "low",
) -> str:
    key = _sha(model, reasoning_effort, prompt)
    cached = cache.get(key)
    if cached is not None:
        return cached
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=reasoning_effort,
    )
    text = resp.choices[0].message.content or ""
    cache.put(key, text)
    budget.llm_tick()
    return text


def embed_batch(
    client: openai.OpenAI,
    cache: MergedJSONCache,
    budget: Budget,
    texts: list[str],
    model: str = EMBED_MODEL,
) -> np.ndarray:
    missing_idx: list[int] = []
    missing_texts: list[str] = []
    vecs: list[np.ndarray | None] = [None] * len(texts)
    for i, t in enumerate(texts):
        key = _sha(model, t)
        v = cache.get(key)
        if v is not None:
            vecs[i] = np.array(v, dtype=np.float32)
        else:
            missing_idx.append(i)
            missing_texts.append(t)
    if missing_texts:
        resp = client.embeddings.create(model=model, input=missing_texts)
        budget.embed_tick()
        for k, d in enumerate(resp.data):
            v = np.array(d.embedding, dtype=np.float32)
            vecs[missing_idx[k]] = v
            cache.put(_sha(model, missing_texts[k]), v.tolist())
    return np.stack(vecs, axis=0)


# ----------------------------- main -----------------------------


def main() -> None:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm_cache = MergedJSONCache(LOCAL_LLM_CACHE, ROUND6B_LLM_CACHE)
    embed_cache = MergedJSONCache(LOCAL_EMBED_CACHE, ROUND6B_EMBED_CACHE)
    budget = Budget(max_llm=MAX_LLM, stop_llm=STOP_LLM, max_embed=MAX_EMBED)

    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)
    with open(SCENARIOS_EXT_FILE) as f:
        ext = json.load(f)
    origin_qs = ext["origin_questions"]

    profiles = scenarios["profiles"]

    # Merge origin Qs into each profile's question list
    for p in profiles:
        pid = p["id"]
        p["all_questions"] = list(p["questions"]) + list(origin_qs.get(pid, []))

    # Primary sweep: budget=30, splits spanning 0..30
    budget30_splits = [(0, 30), (5, 25), (10, 20), (15, 15), (20, 10), (30, 0)]

    # Sanity checks at smaller budget on a single profile (P2) to see if benefit
    # is budget-dependent. Budget=10 is dropped to fit call cap; first=5,last=5
    # adds little signal since first-5 and last-5 overlap heavily with top-10.
    budget20_splits = [(0, 20), (10, 10), (20, 0)]
    budget10_splits: list[tuple[int, int]] = []

    top_k = 10

    all_variants: list[tuple[str, int, int, int]] = []  # (label, top_k, first, last)
    for f_, l_ in budget30_splits:
        all_variants.append((f"b30_f{f_}_l{l_}", top_k, f_, l_))
    # sanity on P2 only
    for f_, l_ in budget20_splits:
        all_variants.append((f"b20_f{f_}_l{l_}", top_k, f_, l_))
    for f_, l_ in budget10_splits:
        all_variants.append((f"b10_f{f_}_l{l_}", top_k, f_, l_))

    # Compute embeddings (should all be cached from round 6B)
    print("Embedding profile entries + queries (expect all cache hits) ...")
    profile_embeds: dict[str, np.ndarray] = {}
    query_embeds: dict[str, list[np.ndarray]] = {}
    for p in profiles:
        pid = p["id"]
        texts = [f"{e['topic']}: {e['text']}" for e in p["entries"]]
        profile_embeds[pid] = embed_batch(client, embed_cache, budget, texts)
        qtexts = [q["q"] for q in p["all_questions"]]
        qe = embed_batch(client, embed_cache, budget, qtexts)
        query_embeds[pid] = [qe[i] for i in range(len(qtexts))]
        embed_cache.save()
    print(
        f"  Embed calls made this run: {budget.embed_made} "
        f"(round6B-hits={embed_cache.ro_hits}, local-hits={embed_cache.local_hits}, "
        f"misses={embed_cache.misses})"
    )

    results: dict[str, Any] = {
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "top_k": top_k,
        "variants": {},
        "profiles": {
            p["id"]: {
                "entry_count": len(p["entries"]),
                "question_count": len(p["all_questions"]),
            }
            for p in profiles
        },
        "budget": {"llm_made": 0, "embed_made": 0},
    }

    def run_variant(
        variant_label: str,
        first_k: int,
        last_k: int,
        profile_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        print(f"\n--- Variant {variant_label} (first={first_k}, last={last_k}) ---")
        per_profile: dict[str, Any] = {}
        for p in profiles:
            pid = p["id"]
            if profile_ids is not None and pid not in profile_ids:
                continue
            entries = p["entries"]
            questions = p["all_questions"]
            q_results: list[dict[str, Any]] = []
            correct_by_type: dict[str, list[int]] = {}
            for qi, question in enumerate(questions):
                qtype = question.get("type", "unknown")
                q_embed = query_embeds[pid][qi]
                compacted, meta = c4_split_anchor(
                    entries,
                    question["q"],
                    profile_embeds[pid],
                    q_embed,
                    top_k=top_k,
                    first_k=first_k,
                    last_k=last_k,
                )
                prompt = READER_PROMPT.format(log=compacted, question=question["q"])
                answer = llm_call(client, llm_cache, budget, prompt)
                llm_cache.save()
                grade = grade_answer(answer, question)
                q_results.append(
                    {
                        "qi": qi,
                        "q": question["q"],
                        "type": qtype,
                        "answer": answer,
                        "grade": grade,
                        "compacted_chars": len(compacted),
                        "kept": meta["kept"],
                    }
                )
                correct_by_type.setdefault(qtype, []).append(
                    1 if grade["correct"] else 0
                )
            acc_by_type = {t: sum(v) / len(v) for t, v in correct_by_type.items()}
            total_correct = sum(1 for r in q_results if r["grade"]["correct"])
            per_profile[pid] = {
                "questions": q_results,
                "correct": total_correct,
                "total": len(q_results),
                "accuracy": total_correct / len(q_results) if q_results else 0,
                "accuracy_by_type": acc_by_type,
                "count_by_type": {t: len(v) for t, v in correct_by_type.items()},
            }
            print(
                f"  {pid}: {total_correct}/{len(q_results)} acc="
                f"{per_profile[pid]['accuracy']:.0%} "
                f"by_type={ {t: f'{a:.0%}' for t, a in acc_by_type.items()} } "
                f"(budget: {budget.llm_made} LLM, ~${budget.cost():.2f})"
            )
        return per_profile

    try:
        # Budget=30 sweep on all profiles
        for f_, l_ in budget30_splits:
            label = f"b30_f{f_}_l{l_}"
            results["variants"][label] = run_variant(label, f_, l_)
            with open(RESULTS_FILE, "w") as w:
                json.dump(results, w, indent=2, default=str)

        # Sanity checks at budget=20 and budget=10 on P2 only (single profile)
        sanity_profile = ["P2_evolving_user"]
        for f_, l_ in budget20_splits:
            label = f"b20_f{f_}_l{l_}"
            results["variants"][label] = run_variant(label, f_, l_, sanity_profile)
            with open(RESULTS_FILE, "w") as w:
                json.dump(results, w, indent=2, default=str)
        for f_, l_ in budget10_splits:
            label = f"b10_f{f_}_l{l_}"
            results["variants"][label] = run_variant(label, f_, l_, sanity_profile)
            with open(RESULTS_FILE, "w") as w:
                json.dump(results, w, indent=2, default=str)
    except RuntimeError as e:
        print(f"[STOPPED] {e}")
    finally:
        llm_cache.save()
        embed_cache.save()
        results["budget"]["llm_made"] = budget.llm_made
        results["budget"]["embed_made"] = budget.embed_made
        results["budget"]["cost_estimate"] = budget.cost()
        with open(RESULTS_FILE, "w") as w:
            json.dump(results, w, indent=2, default=str)
        print(
            f"\nFinal: {budget.llm_made} LLM + {budget.embed_made} embed "
            f"(~${budget.cost():.2f})"
        )
        print(
            f"LLM cache stats: round6B-hits={llm_cache.ro_hits} "
            f"local-hits={llm_cache.local_hits} misses={llm_cache.misses}"
        )

    build_report(results)


# ----------------------------- reporting -----------------------------


def build_report(results: dict[str, Any]) -> None:
    """Write markdown tables: variant x question-type accuracy."""
    lines: list[str] = []
    lines.append("# Earliest-Anchor Sweep for C4 Query-Gated Retrieval\n")
    lines.append(
        f"Model: `{results['model']}`  Embeddings: `{results['embed_model']}`  top_k=`{results['top_k']}`\n"
    )
    lines.append(
        f"Budget used: {results['budget']['llm_made']} LLM + "
        f"{results['budget']['embed_made']} embed "
        f"(~${results['budget'].get('cost_estimate', 0):.2f})\n"
    )

    # Gather all question types
    all_types: list[str] = []
    for vlabel, vdata in results["variants"].items():
        for pid, pres in vdata.items():
            for t in pres.get("count_by_type", {}):
                if t not in all_types:
                    all_types.append(t)

    # --- Budget=30 table: variant x question-type (aggregated across all profiles) ---
    lines.append("\n## Budget=30 sweep (all 3 profiles, 11 questions each)\n")
    b30 = {k: v for k, v in results["variants"].items() if k.startswith("b30_")}
    header = "| Variant | Overall | " + " | ".join(all_types) + " |"
    lines.append(header)
    lines.append("|---------|---------|" + "-------|" * len(all_types))
    for vlabel in sorted(b30.keys(), key=_variant_sort_key):
        vdata = b30[vlabel]
        # Aggregate across profiles
        totals: dict[str, list[int]] = {}  # type -> correct list
        overall_correct = 0
        overall_total = 0
        for pid, pres in vdata.items():
            overall_correct += pres["correct"]
            overall_total += pres["total"]
            for q in pres["questions"]:
                t = q["type"]
                totals.setdefault(t, []).append(1 if q["grade"]["correct"] else 0)
        overall_acc = overall_correct / overall_total if overall_total else 0
        row = [f"`{vlabel}`", f"{overall_correct}/{overall_total} ({overall_acc:.0%})"]
        for t in all_types:
            if t in totals:
                row.append(
                    f"{sum(totals[t])}/{len(totals[t])} ({sum(totals[t]) / len(totals[t]):.0%})"
                )
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    # --- Per-profile breakdown for budget=30 ---
    for pid in results["profiles"]:
        lines.append(f"\n### Budget=30 per-profile: `{pid}`\n")
        lines.append(header)
        lines.append("|---------|---------|" + "-------|" * len(all_types))
        for vlabel in sorted(b30.keys(), key=_variant_sort_key):
            vdata = b30[vlabel]
            pres = vdata.get(pid)
            if not pres:
                continue
            by_type: dict[str, list[int]] = {}
            for q in pres["questions"]:
                t = q["type"]
                by_type.setdefault(t, []).append(1 if q["grade"]["correct"] else 0)
            row = [
                f"`{vlabel}`",
                f"{pres['correct']}/{pres['total']} ({pres['accuracy']:.0%})",
            ]
            for t in all_types:
                if t in by_type:
                    row.append(
                        f"{sum(by_type[t])}/{len(by_type[t])} ({sum(by_type[t]) / len(by_type[t]):.0%})"
                    )
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

    # --- Sanity-check budgets ---
    for bk in ("b20", "b10"):
        bres = {k: v for k, v in results["variants"].items() if k.startswith(bk + "_")}
        if not bres:
            continue
        budget_int = int(bk[1:])
        lines.append(f"\n## Sanity check: budget={budget_int} (P2 only)\n")
        lines.append(header)
        lines.append("|---------|---------|" + "-------|" * len(all_types))
        for vlabel in sorted(bres.keys(), key=_variant_sort_key):
            vdata = bres[vlabel]
            for pid, pres in vdata.items():
                by_type: dict[str, list[int]] = {}
                for q in pres["questions"]:
                    t = q["type"]
                    by_type.setdefault(t, []).append(1 if q["grade"]["correct"] else 0)
                row = [
                    f"`{vlabel}`",
                    f"{pres['correct']}/{pres['total']} ({pres['accuracy']:.0%})",
                ]
                for t in all_types:
                    if t in by_type:
                        row.append(
                            f"{sum(by_type[t])}/{len(by_type[t])} "
                            f"({sum(by_type[t]) / len(by_type[t]):.0%})"
                        )
                    else:
                        row.append("-")
                lines.append("| " + " | ".join(row) + " |")

    # --- Per-question answer appendix (compact) ---
    lines.append("\n## Appendix: per-question results (budget=30 sweep)\n")
    for pid in results["profiles"]:
        lines.append(f"\n### `{pid}` per-question\n")
        # Grab question text from the first variant
        first_variant = next(iter(b30.keys()))
        if pid not in b30[first_variant]:
            continue
        qs = b30[first_variant][pid]["questions"]
        header2 = (
            "| Q | type | "
            + " | ".join(_short(v) for v in sorted(b30.keys(), key=_variant_sort_key))
            + " |"
        )
        lines.append(header2)
        lines.append("|---|------|" + "---|" * len(b30))
        for qi, q in enumerate(qs):
            row = [f"Q{qi + 1}", q["type"]]
            for vlabel in sorted(b30.keys(), key=_variant_sort_key):
                vdata = b30[vlabel].get(pid, {})
                qrs = vdata.get("questions", [])
                mark = "Y" if qrs[qi]["grade"]["correct"] else "N"
                row.append(mark)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        for qi, q in enumerate(qs):
            qtxt = q["q"]
            lines.append(f"- Q{qi + 1} ({q['type']}): {qtxt}")

    out = HERE / "REPORT.md"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out}")


def _variant_sort_key(label: str) -> tuple:
    # e.g., b30_f5_l25
    parts = label.split("_")
    b = int(parts[0][1:])
    first = int(parts[1][1:])
    return (b, first)


def _short(label: str) -> str:
    # b30_f5_l25 -> f5l25
    parts = label.split("_")
    return f"{parts[1]}{parts[2]}"


if __name__ == "__main__":
    main()
