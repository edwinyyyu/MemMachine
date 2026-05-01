"""Round 6 -- Compaction strategies evaluation.

Flow:
  For each profile:
    For each strategy:
      compact the log (possibly 1 LLM call for C3/C5/C7; embeddings for C4)
      For each question:
        reader LLM call: answer question given compacted log
        deterministic grader checks gold_keywords / any_of / wrong_keywords

Budget: hard cap 250 LLM + 100 embedding calls. Stop at 80% = 200 LLM.
Per run estimate:
  * 3 profiles * 7 strategies * 10 questions = 210 reader calls
  * 3 * (C3 + 2*C5 + C7) = 12 compaction LLM calls
  * 3 embedding batches (profile entries) + 3 batches of 10 queries each = 6 embedding calls
  Total ~= 222 LLM + 6 embedding. Tight; cache everything.

We REDUCE reader cost by hitting C0_full (reference) once and sharing the answer
keyword grading (no LLM needed). The reference log answer ISN'T a reader call;
we use the gold keywords directly. The reader is evaluated on the COMPACTED log.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openai
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from strategies import (  # type: ignore
    STRATEGIES,
    c4_query_gated,
)

EVAL_ROOT = HERE.parents[3]  # evaluation/
load_dotenv(EVAL_ROOT / ".env")

MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"
PRICE_PER_LLM = 0.0025  # rough upper bound
PRICE_PER_EMBED = 0.0000002  # effectively free for our token counts

CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

LLM_CACHE_FILE = CACHE_DIR / "llm_cache.json"
EMBED_CACHE_FILE = CACHE_DIR / "embed_cache.json"
SCENARIOS_FILE = HERE / "scenarios.json"
RESULTS_FILE = RESULTS_DIR / "round6_compaction_results.json"
REPORT_FILE = HERE / "REPORT.md"

MAX_LLM = 250
STOP_LLM = int(MAX_LLM * 0.80)
MAX_EMBED = 100


# ----------------------------- cache/budget -----------------------------


def _sha(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


class JSONCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._d: dict[str, Any] = {}
        if path.exists():
            try:
                with open(path) as f:
                    self._d = json.load(f)
            except Exception:
                self._d = {}
        self._dirty = False
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        v = self._d.get(key)
        if v is not None:
            self.hits += 1
        return v

    def put(self, key: str, value: Any) -> None:
        self._d[key] = value
        self._dirty = True
        self.misses += 1

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._d, f)
        tmp.replace(self._path)
        self._dirty = False


@dataclass
class Budget:
    llm_made: int = 0
    embed_made: int = 0
    max_llm: int = MAX_LLM
    stop_llm: int = STOP_LLM
    max_embed: int = MAX_EMBED

    def llm_tick(self) -> None:
        self.llm_made += 1
        if self.llm_made >= self.stop_llm:
            raise RuntimeError(
                f"LLM budget stop hit at {self.llm_made}/{self.max_llm} (80% cap)"
            )

    def embed_tick(self) -> None:
        self.embed_made += 1
        if self.embed_made >= self.max_embed:
            raise RuntimeError(
                f"Embed budget stop hit at {self.embed_made}/{self.max_embed}"
            )

    def cost(self) -> float:
        return self.llm_made * PRICE_PER_LLM + self.embed_made * PRICE_PER_EMBED


# ----------------------------- LLM + embed wrappers -----------------------------


def llm_call(
    client: openai.OpenAI,
    cache: JSONCache,
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
    cache: JSONCache,
    budget: Budget,
    texts: list[str],
    model: str = EMBED_MODEL,
) -> np.ndarray:
    """Batch embedding with cache. Returns N x D array."""
    # Cache per (model, text)
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


# ----------------------------- reader prompt -----------------------------

READER_PROMPT = """You are answering a question about a user based on their memory log.

The log is APPEND-ONLY. Entries may be clarified, refined, superseded, or invalidated by later entries. Relations are marked like "[supersede of 6]" or "[INVALIDATED]". When reasoning about the CURRENT state, prefer the latest non-invalidated claim. For historical questions, earlier entries (including superseded/invalidated ones) are valid sources.

LOG:
{log}

QUESTION:
{question}

Answer concisely in 1-3 sentences. Be specific (include names/numbers if known). If the log does not contain enough information to answer, say "Not in the log." Do not invent facts.
"""


# ----------------------------- grading -----------------------------


def grade_answer(answer: str, question: dict[str, Any]) -> dict[str, Any]:
    """Deterministic grading.

    Pass if:
      - every gold_keyword appears in the lowercased answer, OR
      - if any_of is present, at least one sublist has all its tokens present
    Fail if:
      - any wrong_keyword appears AND no offsetting evidence (we still check
        contains, but mark a separate flag)
    """
    a = answer.lower()
    gold = [k.lower() for k in question.get("gold_keywords", [])]
    any_of = question.get("any_of")
    wrong = [k.lower() for k in question.get("wrong_keywords", [])]

    gold_ok = all(k in a for k in gold) if gold else True
    any_ok = True
    any_hit: list[str] | None = None
    if any_of:
        any_ok = False
        for opt in any_of:
            if all(k.lower() in a for k in opt):
                any_ok = True
                any_hit = opt
                break

    wrong_hits = [k for k in wrong if k in a]
    correct = gold_ok and any_ok and not wrong_hits

    return {
        "correct": correct,
        "gold_ok": gold_ok,
        "any_ok": any_ok,
        "any_hit": any_hit,
        "wrong_hits": wrong_hits,
        "answer_chars": len(answer),
    }


# ----------------------------- main eval -----------------------------


def main() -> None:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm_cache = JSONCache(LLM_CACHE_FILE)
    embed_cache = JSONCache(EMBED_CACHE_FILE)
    budget = Budget()

    with open(SCENARIOS_FILE) as f:
        data = json.load(f)
    profiles = data["profiles"]

    print(f"Loaded {len(profiles)} profiles.")
    for p in profiles:
        print(
            f"  {p['id']}: {len(p['entries'])} entries, {len(p['questions'])} questions"
        )

    # Pre-compute embeddings for all profile entries (for C4) and all queries.
    print("\nComputing embeddings for profile entries + queries ...")
    profile_embeds: dict[str, np.ndarray] = {}
    for p in profiles:
        texts = [f"{e['topic']}: {e['text']}" for e in p["entries"]]
        profile_embeds[p["id"]] = embed_batch(client, embed_cache, budget, texts)
        embed_cache.save()
    query_embeds: dict[str, np.ndarray] = {}
    for p in profiles:
        qtexts = [q["q"] for q in p["questions"]]
        query_embeds[p["id"]] = embed_batch(client, embed_cache, budget, qtexts)
        embed_cache.save()

    results: dict[str, Any] = {
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "profiles": {},
        "budget": {"llm_made": 0, "embed_made": 0},
    }

    # Helper to get an llm_call_fn bound to current client/cache/budget
    def make_llm_fn() -> Callable[[str], str]:
        def _fn(prompt: str) -> str:
            return llm_call(client, llm_cache, budget, prompt)

        return _fn

    try:
        for prof in profiles:
            pid = prof["id"]
            entries = prof["entries"]
            questions = prof["questions"]
            prof_result = {
                "entry_count": len(entries),
                "strategies": {},
            }
            print(f"\n=== Profile {pid} ===")
            for strat in STRATEGIES:
                print(f"  Strategy {strat.key} ...")
                strat_result: dict[str, Any] = {
                    "compacted_chars": 0,
                    "compaction_meta_by_question": [],
                    "questions": [],
                    "correct": 0,
                    "total": len(questions),
                }
                # Compute the compacted log ONCE per (profile, strategy),
                # unless C4 which depends on the query.
                fixed_compacted = None
                fixed_meta = None
                if strat.key != "C4_query_gated":
                    if strat.requires_llm:
                        fixed_compacted, fixed_meta = strat.fn(
                            entries, llm_call_fn=make_llm_fn()
                        )
                    else:
                        fixed_compacted, fixed_meta = strat.fn(entries)
                    strat_result["compacted_chars"] = len(fixed_compacted)
                    strat_result["compaction_meta"] = fixed_meta

                for qi, question in enumerate(questions):
                    if strat.key == "C4_query_gated":
                        q_embed = query_embeds[pid][qi]
                        compacted, meta = c4_query_gated(
                            entries,
                            question["q"],
                            profile_embeds[pid],
                            q_embed,
                            top_k=10,
                            recent_k=10,
                        )
                    else:
                        compacted = fixed_compacted
                        meta = fixed_meta

                    prompt = READER_PROMPT.format(log=compacted, question=question["q"])
                    answer = llm_call(client, llm_cache, budget, prompt)
                    llm_cache.save()
                    grade = grade_answer(answer, question)
                    strat_result["questions"].append(
                        {
                            "q": question["q"],
                            "type": question.get("type"),
                            "answer": answer,
                            "grade": grade,
                            "compacted_chars": len(compacted)
                            if strat.key == "C4_query_gated"
                            else strat_result["compacted_chars"],
                        }
                    )
                    strat_result["compaction_meta_by_question"].append(meta)
                    if grade["correct"]:
                        strat_result["correct"] += 1

                strat_result["accuracy"] = (
                    strat_result["correct"] / strat_result["total"]
                )
                # Average compacted chars across questions (only varies for C4)
                avg_chars = sum(
                    q["compacted_chars"] for q in strat_result["questions"]
                ) / len(strat_result["questions"])
                strat_result["avg_compacted_chars"] = avg_chars
                # Total LLM calls for compaction
                compaction_llm = sum(
                    m.get("llm_calls", 0)
                    for m in strat_result["compaction_meta_by_question"]
                )
                # For non-C4, compaction is computed once, so divide by N
                if strat.key != "C4_query_gated" and fixed_meta:
                    compaction_llm = fixed_meta.get("llm_calls", 0)
                strat_result["compaction_llm_calls"] = compaction_llm
                prof_result["strategies"][strat.key] = strat_result
                print(
                    f"    {strat.key}: {strat_result['correct']}/{strat_result['total']} "
                    f"acc={strat_result['accuracy']:.0%} "
                    f"avg_chars={int(avg_chars)} "
                    f"compaction_calls={compaction_llm} "
                    f"(budget: {budget.llm_made} LLM + {budget.embed_made} embed, "
                    f"~${budget.cost():.2f})"
                )

            results["profiles"][pid] = prof_result
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2, default=str)
    finally:
        llm_cache.save()
        embed_cache.save()
        results["budget"]["llm_made"] = budget.llm_made
        results["budget"]["embed_made"] = budget.embed_made
        results["budget"]["cost_estimate"] = budget.cost()
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(
            f"\nFinal budget: {budget.llm_made} LLM + {budget.embed_made} embed "
            f"(~${budget.cost():.2f})"
        )
        print(f"Saved {RESULTS_FILE}")

    write_report(results)


# ----------------------------- report -----------------------------


def write_report(results: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Round 6 -- Compaction Strategies for Append-Only Topic Logs\n")
    lines.append(
        f"Model: `{results['model']}`  Embeddings: `{results['embed_model']}`\n"
    )
    lines.append(
        f"Budget used: {results['budget']['llm_made']} LLM + "
        f"{results['budget']['embed_made']} embed "
        f"(~${results['budget'].get('cost_estimate', 0):.2f})\n"
    )

    # Per-profile tables
    for pid, pres in results["profiles"].items():
        lines.append(f"\n## Profile `{pid}` ({pres['entry_count']} entries)\n")
        lines.append(
            "| Strategy | Correct | Acc | Avg compacted chars | Compaction LLM calls |"
        )
        lines.append(
            "|----------|---------|-----|----------------------|----------------------|"
        )
        for sk, sr in pres["strategies"].items():
            lines.append(
                f"| `{sk}` | {sr['correct']}/{sr['total']} | "
                f"{sr['accuracy']:.0%} | {int(sr['avg_compacted_chars'])} | "
                f"{sr['compaction_llm_calls']} |"
            )

    # Aggregate leaderboard across profiles
    lines.append("\n## Aggregate (across profiles)\n")
    lines.append(
        "| Strategy | Total Correct / N | Acc | Avg chars | Total compaction LLM |"
    )
    lines.append(
        "|----------|-------------------|-----|-----------|----------------------|"
    )
    agg: dict[str, dict[str, float]] = {}
    for pid, pres in results["profiles"].items():
        for sk, sr in pres["strategies"].items():
            a = agg.setdefault(
                sk, {"correct": 0, "total": 0, "chars": 0.0, "llm": 0, "n_p": 0}
            )
            a["correct"] += sr["correct"]
            a["total"] += sr["total"]
            a["chars"] += sr["avg_compacted_chars"]
            a["llm"] += sr["compaction_llm_calls"]
            a["n_p"] += 1
    for sk in [s.key for s in STRATEGIES]:
        a = agg.get(sk)
        if not a:
            continue
        acc = a["correct"] / a["total"] if a["total"] else 0
        avg_chars = a["chars"] / a["n_p"] if a["n_p"] else 0
        lines.append(
            f"| `{sk}` | {int(a['correct'])}/{int(a['total'])} | "
            f"{acc:.0%} | {int(avg_chars)} | {int(a['llm'])} |"
        )

    # Per-question detail per profile
    for pid, pres in results["profiles"].items():
        lines.append(f"\n## Per-question detail -- `{pid}`\n")
        qs = next(iter(pres["strategies"].values()))["questions"]
        q_labels = [
            f"Q{i + 1}: {q['q'][:60]}..." if len(q["q"]) > 60 else f"Q{i + 1}: {q['q']}"
            for i, q in enumerate(qs)
        ]
        header = "| Q | type | " + " | ".join(sk for sk in pres["strategies"]) + " |"
        sep = "|---|------|" + "---|" * len(pres["strategies"])
        lines.append(header)
        lines.append(sep)
        for qi, ql in enumerate(q_labels):
            qtype = qs[qi].get("type", "")
            row = [f"Q{qi + 1}", qtype]
            for sk in pres["strategies"]:
                qr = pres["strategies"][sk]["questions"][qi]
                row.append("Y" if qr["grade"]["correct"] else "N")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        for qi, ql in enumerate(q_labels):
            lines.append(f"- {ql}")

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {REPORT_FILE}")


if __name__ == "__main__":
    main()
