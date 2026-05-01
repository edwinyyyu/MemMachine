"""E1 — compare cue-generation mechanisms on the mid-execution benchmark.

Five arms scored on the same per-scenario EM substrate, same gold, same K:

  A. task_prompt       (no per-step probe; baseline from E0)
  B. decision_text     (script-action embedded as cue; baseline from E0)
  C. agent_natural     (real executor agent; cue = content the agent wrote)
  D. agent_cue_aware   (real executor agent instructed to emit `CUE:` lines)
  E. decompose_upfront (one-shot DECOMPOSE+CUEGEN at task start; per-step
                        cue = decision_text MERGED with the upfront probe set)

Real-agent strategies (C, D) use gpt-5-mini at reasoning_effort=low to mirror
existing infrastructure. The agent runs a SINGLE multi-turn conversation per
scenario so it accumulates context between steps.

Usage:

    uv run python evaluation/associative_recall/mid_execution_eval_e1.py
    uv run python evaluation/associative_recall/mid_execution_eval_e1.py --scenario banquet-01
    uv run python evaluation/associative_recall/mid_execution_eval_e1.py --strategies A,B,C,D,E --K 1,3,5,10
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)

# Reuse E0 scaffolding.
from mid_execution_eval import (  # type: ignore
    RESULTS_DIR,
    Hit,
    false_positive_rate,
    ingest_scenario,
    load_locomo_segments,
    load_scenarios,
    load_speakers,
    probe,
    triggered_recall,
)

# Reuse existing CUEGEN/DECOMPOSE prompts from proactive_memory.
from proactive_memory import (  # type: ignore
    CUEGEN_PROMPT,
    DECOMPOSE_PROMPT,
    parse_cues,
    parse_needs,
)
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv(Path(__file__).resolve().parent / ".env")

CACHE_DIR = Path(__file__).resolve().parent / "cache"
AGENT_CACHE_FILE = CACHE_DIR / "mid_exec_e1_agent_cache.json"
DECOMPOSE_CACHE_FILE = CACHE_DIR / "mid_exec_e1_decompose_cache.json"

EXECUTOR_MODEL = "gpt-5-mini"
DECOMPOSE_MODEL = "gpt-5-mini"


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


class _SimpleCache:
    """File-backed cache keyed by sha256(model + prompt-or-history-json)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, str] = {}
        if path.exists():
            try:
                self._cache = json.loads(path.read_text())
            except Exception:
                self._cache = {}
        self._dirty = False

    @staticmethod
    def _key(tag: str, payload: str) -> str:
        return hashlib.sha256(f"{tag}:{payload}".encode()).hexdigest()

    def get(self, tag: str, payload: str) -> str | None:
        return self._cache.get(self._key(tag, payload))

    def put(self, tag: str, payload: str, value: str) -> None:
        self._cache[self._key(tag, payload)] = value
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._cache))
        tmp.replace(self._path)
        self._dirty = False


# --------------------------------------------------------------------------
# Multi-probe retrieval
# --------------------------------------------------------------------------


async def probe_multi(memory: EventMemory, queries: list[str], K: int) -> list[Hit]:
    """Run each query, merge hits by max score across probes, take top-K."""
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return []
    if len(queries) == 1:
        return await probe(memory, queries[0], K)
    all_lists = await asyncio.gather(*[probe(memory, q, K * 2) for q in queries])
    by_tid: dict[int, Hit] = {}
    for hits in all_lists:
        for h in hits:
            existing = by_tid.get(h.turn_id)
            if existing is None or h.score > existing.score:
                by_tid[h.turn_id] = h
    return sorted(by_tid.values(), key=lambda h: -h.score)[:K]


# --------------------------------------------------------------------------
# Real-agent executor (gpt-5-mini)
# --------------------------------------------------------------------------


NATURAL_SYSTEM = """\
You are an executor agent helping a coworker complete a multi-step task. The \
coworker will walk you through the task one step at a time, in order.

OVERALL TASK:
{task_prompt}

For each step they describe, write the actual content / deliverable for that \
step in 1-3 short sentences. Be concrete (use real names, real numbers, real \
phrasing — make it look like the actual artifact, not a meta-description). \
Do not write commentary, plans, or explanations. Just the deliverable text \
for the requested step.
"""


CUE_AWARE_SYSTEM = """\
You are an executor agent helping a coworker complete a multi-step task. The \
coworker will walk you through the task one step at a time. For each step you \
have access to a memory of past chat history that may contain useful context \
(brand guidelines, allergies, deadlines, preferences, etc.) — but you must \
ASK for it before you write the step.

OVERALL TASK:
{task_prompt}

PROTOCOL — for each step do EXACTLY two things, in order:

1. On a single line, emit a query of the form
   `CUE: <one short retrieval query (5-15 words), in plain English, describing \
what fact about the user/team/context bears on this specific step>`.
   The query will be embedded and used to fetch matching past chat turns. \
Make the query specific enough that it won't fetch generic chat about the \
overall task — focus it on the sub-decision at hand. If you genuinely \
believe nothing in past context is relevant, emit `CUE: none`.

2. Then write the deliverable for the step in 1-3 short sentences. Be \
concrete — real names, real numbers, real phrasing. No meta-commentary.

Format example for one step:

    CUE: brand color guidelines for the deck visual style
    Use the deep navy color across all title bars and accent ...
"""


CUE_AWARE_MULTI_SYSTEM = """\
You are an executor agent helping a coworker complete a multi-step task. The \
coworker will walk you through the task one step at a time. Past chat \
history may contain context (brand guidelines, allergies, deadlines, \
preferences, interpersonal dynamics, etc.) — you must ASK for it before \
acting on each step.

OVERALL TASK:
{task_prompt}

PROTOCOL — for each step do EXACTLY two things, in order:

1. Emit 3 retrieval queries, EACH ON ITS OWN LINE prefixed by `CUE: `. \
The 3 queries should attack the step from DIFFERENT ANGLES so that if your \
first guess about what's relevant is wrong, the others may still surface \
the right past chat turn. Useful angles include:

   - the obvious topical match for the action (e.g. "brand color for the deck");
   - a latent constraint angle (e.g. "accessibility / mobility / dietary / legal restrictions");
   - a past-failure or interpersonal angle (e.g. "previous incidents, conflicts, complaints related to this");
   - a personal preference angle (e.g. "person X's stated preferences about this");
   - a hard limit / quota / deadline angle.

   Pick 3 angles that genuinely apply to this step. Each query should be \
short (5-15 words), in plain English, and specific enough not to retrieve \
generic chat about the overall task.

   If you genuinely believe nothing in past context is relevant, emit \
exactly one line `CUE: none` and skip to step 2.

2. Then write the deliverable for the step in 1-3 short sentences. Be \
concrete — real names, real numbers, real phrasing. No meta-commentary.

Format example for one step:

    CUE: deck color palette and brand guidelines for this client
    CUE: past complaints from this client about visual elements
    CUE: accessibility constraints for executives reading the deck
    Use deep navy across title bars; bold sans-serif at 24pt minimum ...
"""


CUE_LINE_RE = re.compile(r"^\s*CUE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def parse_cue_aware(response: str) -> tuple[str, str]:
    """Return (first_cue_text, content_text). first_cue_text is "" if no CUE."""
    m = CUE_LINE_RE.search(response or "")
    if not m:
        return "", (response or "").strip()
    cue = m.group(1).strip()
    if cue.lower() in ("none", "(none)", "n/a"):
        cue = ""
    after = (response or "")[m.end() :].strip()
    return cue, after


def parse_cue_aware_multi(response: str) -> tuple[list[str], str]:
    """Return (list_of_cues, content_text). Empty list if all are 'none'."""
    cues: list[str] = []
    last_cue_end = 0
    for m in CUE_LINE_RE.finditer(response or ""):
        cue = m.group(1).strip()
        if cue.lower() in ("none", "(none)", "n/a"):
            continue
        cues.append(cue)
        last_cue_end = m.end()
    if not cues:
        return [], (response or "").strip()
    after = (response or "")[last_cue_end:].strip()
    return cues, after


async def run_executor_agent(
    scenario: dict,
    *,
    mode: str,  # "natural" or "cue_aware"
    openai_client,
    cache: _SimpleCache,
) -> list[dict]:
    """Walk an executor through scenario.subdecision_script in one conversation.

    Returns: [{step_id, decision_text, cue, content}, ...]
    """
    if mode == "natural":
        system = NATURAL_SYSTEM.format(task_prompt=scenario["task_prompt"])
    elif mode == "cue_aware":
        system = CUE_AWARE_SYSTEM.format(task_prompt=scenario["task_prompt"])
    elif mode == "cue_aware_multi":
        system = CUE_AWARE_MULTI_SYSTEM.format(task_prompt=scenario["task_prompt"])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    history: list[dict] = [{"role": "system", "content": system}]
    out: list[dict] = []

    for step in scenario["subdecision_script"]:
        user_msg = f"Step {step['step_id']}: {step['decision_text']}\n\nProceed."
        history.append({"role": "user", "content": user_msg})

        # Cache by (model, mode, history-json) so re-runs are free.
        cache_payload = json.dumps(
            {"model": EXECUTOR_MODEL, "mode": mode, "history": history},
            sort_keys=True,
        )
        cached = cache.get(EXECUTOR_MODEL, cache_payload)
        if cached is None:
            resp = await openai_client.chat.completions.create(
                model=EXECUTOR_MODEL,
                messages=history,
                reasoning_effort="low",
            )
            cached = resp.choices[0].message.content or ""
            cache.put(EXECUTOR_MODEL, cache_payload, cached)

        history.append({"role": "assistant", "content": cached})

        if mode == "cue_aware":
            cue_text, content_text = parse_cue_aware(cached)
            cues = [cue_text] if cue_text else []
        elif mode == "cue_aware_multi":
            cues, content_text = parse_cue_aware_multi(cached)
            cue_text = cues[0] if cues else ""
        else:
            cue_text = cached.strip()
            content_text = cached.strip()
            cues = [cue_text] if cue_text else []

        out.append(
            {
                "step_id": step["step_id"],
                "decision_text": step["decision_text"],
                "cue": cue_text,
                "cues": cues,
                "content": content_text,
                "raw_response": cached,
            }
        )

    return out


# --------------------------------------------------------------------------
# Pre-execution one-shot decompose + cuegen
# --------------------------------------------------------------------------


async def get_decompose_upfront_probes(
    scenario: dict,
    *,
    user_name: str,
    asst_name: str,
    openai_client,
    cache: _SimpleCache,
) -> list[str]:
    """Run DECOMPOSE once, then CUEGEN per need; return all probe texts.

    The probe set is scenario-level (not per-step). It will be MERGED with
    the per-step decision_text at scoring time.
    """
    decompose_prompt = DECOMPOSE_PROMPT.format(
        participant_1=user_name,
        participant_2=asst_name,
        task_prompt=scenario["task_prompt"],
    )
    cached = cache.get(DECOMPOSE_MODEL, decompose_prompt)
    if cached is None:
        resp = await openai_client.chat.completions.create(
            model=DECOMPOSE_MODEL,
            messages=[{"role": "user", "content": decompose_prompt}],
            reasoning_effort="low",
        )
        cached = resp.choices[0].message.content or ""
        cache.put(DECOMPOSE_MODEL, decompose_prompt, cached)
    needs = parse_needs(cached)

    probes: list[str] = []
    for need in needs:
        probes.append(need["need"])  # primer (no LLM)
        vocab_str = ", ".join(need.get("expected_vocab") or []) or "(none)"
        cue_prompt = CUEGEN_PROMPT.format(
            participant_1=user_name,
            participant_2=asst_name,
            task_prompt=scenario["task_prompt"],
            need=need["need"],
            expected_vocab=vocab_str,
            prior_section="",
        )
        cached_cue = cache.get(DECOMPOSE_MODEL, cue_prompt)
        if cached_cue is None:
            resp = await openai_client.chat.completions.create(
                model=DECOMPOSE_MODEL,
                messages=[{"role": "user", "content": cue_prompt}],
                reasoning_effort="low",
            )
            cached_cue = resp.choices[0].message.content or ""
            cache.put(DECOMPOSE_MODEL, cue_prompt, cached_cue)
        probes.extend(parse_cues(cached_cue, max_cues=2))

    return probes


# --------------------------------------------------------------------------
# Per-strategy cue resolution
# --------------------------------------------------------------------------


def cue_A_task_prompt(scenario, step, scenario_state):
    return [scenario["task_prompt"]]


def cue_B_decision_text(scenario, step, scenario_state):
    return [step["decision_text"]]


def cue_C_agent_natural(scenario, step, scenario_state):
    rec = scenario_state["agent_natural_by_step"].get(step["step_id"])
    return [rec["content"]] if rec and rec["content"] else []


def cue_D_agent_cue_aware(scenario, step, scenario_state):
    rec = scenario_state["agent_cue_aware_by_step"].get(step["step_id"])
    if not rec or not rec["cue"]:
        return []
    return [rec["cue"]]


def cue_E_decompose_upfront(scenario, step, scenario_state):
    return [step["decision_text"]] + scenario_state.get("decompose_probes", [])


def cue_F_agent_cue_aware_multi(scenario, step, scenario_state):
    rec = scenario_state["agent_cue_aware_multi_by_step"].get(step["step_id"])
    if not rec:
        return []
    return rec.get("cues") or []


def cue_G_combined(scenario, step, scenario_state):
    """decision_text PLUS the single agent_cue_aware CUE — multi-probe."""
    out = [step["decision_text"]]
    rec = scenario_state.get("agent_cue_aware_by_step", {}).get(step["step_id"])
    if rec and rec.get("cue"):
        out.append(rec["cue"])
    return out


STRATEGIES = {
    "A": ("task_prompt", cue_A_task_prompt),
    "B": ("decision_text", cue_B_decision_text),
    "C": ("agent_natural", cue_C_agent_natural),
    "D": ("agent_cue_aware", cue_D_agent_cue_aware),
    "E": ("decompose_upfront", cue_E_decompose_upfront),
    "F": ("agent_cue_aware_multi", cue_F_agent_cue_aware_multi),
    "G": ("combined_decision_plus_cue_aware", cue_G_combined),
}


# --------------------------------------------------------------------------
# Per-scenario orchestration
# --------------------------------------------------------------------------


async def run_scenario_e1(
    scenario: dict,
    locomo_segments: dict,
    speakers_map: dict,
    *,
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
    openai_client,
    agent_cache: _SimpleCache,
    decompose_cache: _SimpleCache,
    K_list: list[int],
    strategies: list[str],
    overwrite: bool = True,
) -> dict:
    sid = scenario["scenario_id"]
    base_conv = scenario["base_conversation"]
    locomo_turns = locomo_segments[base_conv]
    speakers = speakers_map.get(base_conv) or {}

    # ---- Ingest EM ----
    t0 = time.monotonic()
    memory, ingest_info = await ingest_scenario(
        scenario,
        locomo_turns,
        speakers,
        vector_store=vector_store,
        segment_store=segment_store,
        embedder=embedder,
        overwrite=overwrite,
    )
    ingest_time = time.monotonic() - t0

    # ---- Pre-compute per-strategy scenario-level state ----
    scenario_state: dict = {}

    if "C" in strategies:
        t = time.monotonic()
        natural_records = await run_executor_agent(
            scenario,
            mode="natural",
            openai_client=openai_client,
            cache=agent_cache,
        )
        scenario_state["agent_natural_by_step"] = {
            r["step_id"]: r for r in natural_records
        }
        scenario_state["agent_natural_time_s"] = round(time.monotonic() - t, 2)

    # D and G both need the cue_aware records.
    if "D" in strategies or "G" in strategies:
        t = time.monotonic()
        cue_aware_records = await run_executor_agent(
            scenario,
            mode="cue_aware",
            openai_client=openai_client,
            cache=agent_cache,
        )
        scenario_state["agent_cue_aware_by_step"] = {
            r["step_id"]: r for r in cue_aware_records
        }
        scenario_state["agent_cue_aware_time_s"] = round(time.monotonic() - t, 2)

    if "F" in strategies:
        t = time.monotonic()
        multi_records = await run_executor_agent(
            scenario,
            mode="cue_aware_multi",
            openai_client=openai_client,
            cache=agent_cache,
        )
        scenario_state["agent_cue_aware_multi_by_step"] = {
            r["step_id"]: r for r in multi_records
        }
        scenario_state["agent_cue_aware_multi_time_s"] = round(time.monotonic() - t, 2)

    if "E" in strategies:
        t = time.monotonic()
        scenario_state["decompose_probes"] = await get_decompose_upfront_probes(
            scenario,
            user_name=ingest_info["user_name"],
            asst_name=ingest_info["assistant_name"],
            openai_client=openai_client,
            cache=decompose_cache,
        )
        scenario_state["decompose_time_s"] = round(time.monotonic() - t, 2)

    # Save caches early so partial work isn't lost.
    agent_cache.save()
    decompose_cache.save()

    # ---- Per-step scoring ----
    K_max = max(K_list)
    per_step: list[dict] = []
    for step in scenario["subdecision_script"]:
        gold = step.get("gold_plant_ids") or []
        is_noop = len(gold) == 0
        per_strat: dict = {}
        for code in strategies:
            label, fn = STRATEGIES[code]
            queries = fn(scenario, step, scenario_state)
            queries = [q for q in queries if q and q.strip()]
            if not queries:
                per_strat[label] = {"queries": [], "skipped": True}
                continue
            hits = await probe_multi(memory, queries, K_max)
            entry = {
                "queries": [q[:160] for q in queries],
                "n_queries": len(queries),
                "top_hits": [
                    {
                        "rank": i + 1,
                        "turn_id": h.turn_id,
                        "plant_id": h.plant_id,
                        "score": round(h.score, 4),
                        "text_preview": h.text[:120],
                    }
                    for i, h in enumerate(hits[:K_max])
                ],
            }
            for K in K_list:
                if is_noop:
                    entry[f"fpr@{K}"] = false_positive_rate(hits, K)
                else:
                    entry[f"recall@{K}"] = triggered_recall(hits, gold, K)
            per_strat[label] = entry
        per_step.append(
            {
                "step_id": step["step_id"],
                "decision_text": step["decision_text"],
                "gold_plant_ids": gold,
                "is_noop": is_noop,
                "per_strategy": per_strat,
            }
        )

    # ---- Per-strategy aggregates (across non-no-op steps for recall;
    # across no-op steps for fpr).
    aggregates: dict = {}
    for code in strategies:
        label = STRATEGIES[code][0]
        for K in K_list:
            recalls = [
                s["per_strategy"][label].get(f"recall@{K}")
                for s in per_step
                if not s["is_noop"]
                and not s["per_strategy"][label].get("skipped")
                and s["per_strategy"][label].get(f"recall@{K}") is not None
            ]
            if recalls:
                aggregates[f"{label}.mean_recall@{K}"] = round(
                    sum(recalls) / len(recalls), 4
                )
        K = max(K_list)
        fps = [
            s["per_strategy"][label].get(f"fpr@{K}")
            for s in per_step
            if s["is_noop"]
            and not s["per_strategy"][label].get("skipped")
            and s["per_strategy"][label].get(f"fpr@{K}") is not None
        ]
        if fps:
            aggregates[f"{label}.mean_fpr@{K}_on_noop"] = round(sum(fps) / len(fps), 4)

    return {
        "scenario_id": sid,
        "category": scenario.get("category", ""),
        "base_conversation": base_conv,
        "ingest_time_s": round(ingest_time, 2),
        "ingest_info": ingest_info,
        "scenario_state_meta": {
            k: v
            for k, v in scenario_state.items()
            if not isinstance(v, dict) or k == "decompose_probes"
        },
        "K_list": K_list,
        "per_step": per_step,
        "aggregates": aggregates,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", default=None, help="Single scenario_id (default: all)"
    )
    parser.add_argument(
        "--K", default="1,3,5,10", help="Comma-separated K values for recall@K"
    )
    parser.add_argument(
        "--strategies",
        default="A,B,C,D,E",
        help=f"Comma-separated codes from {sorted(STRATEGIES)}",
    )
    parser.add_argument("--out", default=None, help="Where to write the JSON result")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Reuse existing collections if present",
    )
    args = parser.parse_args()

    K_list = sorted({int(x) for x in args.K.split(",") if x.strip()})
    strategies = [s.strip().upper() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in STRATEGIES:
            raise SystemExit(f"Unknown strategy {s}; valid: {sorted(STRATEGIES)}")

    scenarios = load_scenarios()
    if args.scenario:
        scenarios = [s for s in scenarios if s["scenario_id"] == args.scenario]
        if not scenarios:
            raise SystemExit(f"No scenario matched: {args.scenario}")

    locomo_segments = load_locomo_segments()
    speakers_map = load_speakers()

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sqlite_path = RESULTS_DIR / "eventmemory_mid_exec_e1.sqlite3"
    sql_url = f"sqlite+aiosqlite:///{sqlite_path}"
    engine = create_async_engine(sql_url)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=8192,
        )
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    agent_cache = _SimpleCache(AGENT_CACHE_FILE)
    decompose_cache = _SimpleCache(DECOMPOSE_CACHE_FILE)

    results: list[dict] = []
    try:
        for scenario in scenarios:
            print(
                f"[run] {scenario['scenario_id']} (base={scenario['base_conversation']}, "
                f"plants={sum(1 for p in scenario['preamble_turns'] if p.get('plant_id') is not None)}, "
                f"steps={len(scenario['subdecision_script'])}, "
                f"strategies={strategies})",
                flush=True,
            )
            r = await run_scenario_e1(
                scenario,
                locomo_segments,
                speakers_map,
                vector_store=vector_store,
                segment_store=segment_store,
                embedder=embedder,
                openai_client=openai_client,
                agent_cache=agent_cache,
                decompose_cache=decompose_cache,
                K_list=K_list,
                strategies=strategies,
                overwrite=not args.no_overwrite,
            )
            results.append(r)
            print(f"  ingested in {r['ingest_time_s']}s; aggregates:")
            for k, v in r["aggregates"].items():
                print(f"    {k} = {v}")
    finally:
        agent_cache.save()
        decompose_cache.save()
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    out_path = (
        Path(args.out)
        if args.out
        else (RESULTS_DIR / f"mid_execution_eval_e1_{int(time.time())}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "K_list": K_list,
                "strategies": strategies,
                "n_scenarios": len(results),
                "scenarios": results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")

    # Cross-scenario summary table.
    print("\n=== Cross-scenario means ===")
    label_for = {code: STRATEGIES[code][0] for code in strategies}
    header = ["strategy"] + [f"R@{K}" for K in K_list] + [f"FPR@{max(K_list)}_noop"]
    print("  " + " | ".join(f"{c:>22}" for c in header))
    for code in strategies:
        label = label_for[code]
        cells = [label]
        for K in K_list:
            vals = [
                r["aggregates"].get(f"{label}.mean_recall@{K}")
                for r in results
                if r["aggregates"].get(f"{label}.mean_recall@{K}") is not None
            ]
            cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "-")
        K = max(K_list)
        fps = [
            r["aggregates"].get(f"{label}.mean_fpr@{K}_on_noop")
            for r in results
            if r["aggregates"].get(f"{label}.mean_fpr@{K}_on_noop") is not None
        ]
        cells.append(f"{sum(fps) / len(fps):.3f}" if fps else "-")
        print("  " + " | ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    asyncio.run(main())
