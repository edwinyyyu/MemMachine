"""Evaluate System A (single-shot) vs System B (proactive) on task prompts.

Task-sufficiency (LLM judge, 0-10) is the main metric, not recall@K.

Outputs:
  results/proactive_memory.json
  results/proactive_memory.md

Usage:
  uv run python evaluation/associative_recall/proactive_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)

from em_architectures import V2F_MODEL, EMHit, _MergedLLMCache
from proactive_memory import (
    PROACTIVE_CUEGEN_CACHE,
    PROACTIVE_DECOMPOSE_CACHE,
    PROACTIVE_SUFFICIENCY_CACHE,
    ProactiveResult,
    _llm_call,
    _extract_json,
    run_proactive,
    run_single_shot,
)


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)


DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")

SINGLESHOT_CUEGEN_CACHE = CACHE_DIR / "proactive_singleshot_cuegen_cache.json"
JUDGE_CACHE = CACHE_DIR / "proactive_judge_cache.json"


JUDGE_PROMPT = """\
You are a retrieval sufficiency judge. You are given a user TASK and a \
set of chat-memory turns that were retrieved to help an AI complete the \
task. Rate the retrieval on four dimensions (all integers 0-10):

- COVERAGE: Does the retrieved content cover all the information types \
the task requires? (10 = every type present; 0 = nothing relevant.)
- DEPTH: Does it contain enough specific detail to actually complete the \
task? (10 = richly specific; 0 = only superficial snippets.)
- NOISE (reverse-scored): 10 means the retrieved set is tightly on-topic \
(low noise); 0 means almost everything is irrelevant.
- SUFFICIENCY: Overall, could an AI use this retrieval to complete the \
task well? (10 = yes, plenty; 0 = no, would mostly be guessing.)

TASK:
{task_prompt}

RETRIEVED TURNS ({n_turns} total, top-{n_shown} shown, sorted by turn_id):
{retrieved_section}

Output ONLY a JSON object, no prose:
{{"coverage": <0-10>, "depth": <0-10>, "noise": <0-10>, \
"sufficiency": <0-10>, "brief_reasoning": "one or two sentences"}}"""


def _format_hits_for_judge(hits: list[EMHit], max_items: int = 40, max_len: int = 220) -> str:
    if not hits:
        return "(no retrievals)"
    # Sort by turn_id for readability.
    hits_sorted = sorted(hits, key=lambda h: h.turn_id)[:max_items]
    lines = []
    for h in hits_sorted:
        txt = h.text.replace("\n", " ")
        if len(txt) > max_len:
            txt = txt[:max_len] + "..."
        lines.append(f"[Turn {h.turn_id}, {h.role}]: {txt}")
    return "\n".join(lines)


async def judge_retrieval(
    task_prompt: str,
    hits: list[EMHit],
    *,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[dict, bool]:
    shown = sorted(hits, key=lambda h: h.turn_id)[:40]
    retrieved_section = _format_hits_for_judge(shown, max_items=40)
    prompt = JUDGE_PROMPT.format(
        task_prompt=task_prompt,
        n_turns=len(hits),
        n_shown=len(shown),
        retrieved_section=retrieved_section,
    )
    raw, cache_hit = await _llm_call(openai_client, prompt, cache)
    obj = _extract_json(raw) or {}
    out = {
        "coverage": int(obj.get("coverage", 0) or 0),
        "depth": int(obj.get("depth", 0) or 0),
        "noise": int(obj.get("noise", 0) or 0),
        "sufficiency": int(obj.get("sufficiency", 0) or 0),
        "brief_reasoning": str(obj.get("brief_reasoning") or "").strip(),
    }
    return out, cache_hit


def load_tasks() -> list[dict]:
    with open(DATA_DIR / "proactive_tasks.json") as f:
        data = json.load(f)
    return data["tasks"]


def load_collections_meta() -> dict:
    with open(RESULTS_DIR / "eventmemory_collections.json") as f:
        return json.load(f)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run first N tasks (smoke test)",
    )
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--K_per_need", type=int, default=15)
    parser.add_argument("--max_rounds", type=int, default=2)
    args = parser.parse_args()

    collections_meta = load_collections_meta()
    tasks = load_tasks()
    if args.limit is not None:
        tasks = tasks[: args.limit]

    conv_to_meta = {r["conversation_id"]: r for r in collections_meta["conversations"]}

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sql_url = collections_meta.get("sql_url") or os.getenv("SQL_URL")
    if sql_url is None:
        raise RuntimeError("No SQL_URL in collections meta or env")
    if sql_url.startswith("sqlite"):
        engine = create_async_engine(sql_url)
    else:
        engine = create_async_engine(sql_url, pool_size=20, max_overflow=20)
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

    caches: dict[str, _MergedLLMCache] = {
        "singleshot_cuegen": _MergedLLMCache(
            reader_paths=[SINGLESHOT_CUEGEN_CACHE],
            writer_path=SINGLESHOT_CUEGEN_CACHE,
        ),
        "decompose": _MergedLLMCache(
            reader_paths=[PROACTIVE_DECOMPOSE_CACHE],
            writer_path=PROACTIVE_DECOMPOSE_CACHE,
        ),
        "cuegen": _MergedLLMCache(
            reader_paths=[PROACTIVE_CUEGEN_CACHE],
            writer_path=PROACTIVE_CUEGEN_CACHE,
        ),
        "sufficiency": _MergedLLMCache(
            reader_paths=[PROACTIVE_SUFFICIENCY_CACHE],
            writer_path=PROACTIVE_SUFFICIENCY_CACHE,
        ),
        "judge": _MergedLLMCache(
            reader_paths=[JUDGE_CACHE],
            writer_path=JUDGE_CACHE,
        ),
    }

    # Open EM per conversation.
    memories: dict[str, EventMemory] = {}
    participants_by_conv: dict[str, tuple[str, str]] = {}
    opened_resources: list = []
    for conv_id in LOCOMO_CONV_IDS:
        meta = conv_to_meta[conv_id]
        coll = await vector_store.open_collection(
            namespace=meta["namespace"], name=meta["collection_name"]
        )
        part = await segment_store.open_or_create_partition(meta["partition_key"])
        mem = EventMemory(
            EventMemoryParams(
                vector_store_collection=coll,
                segment_store_partition=part,
                embedder=embedder,
                reranker=None,
                derive_sentences=False,
                max_text_chunk_length=500,
            )
        )
        memories[conv_id] = mem
        participants_by_conv[conv_id] = (meta["user_name"], meta["assistant_name"])
        opened_resources.append((coll, part))

    rows: list[dict] = []
    try:
        for task in tasks:
            cid = task["conversation_id"]
            mem = memories[cid]
            participants = participants_by_conv[cid]
            task_prompt = task["task_prompt"]

            # System A
            tA = time.monotonic()
            resA: ProactiveResult = await run_single_shot(
                mem, task_prompt, participants,
                K=args.K,
                cuegen_cache=caches["singleshot_cuegen"],
                openai_client=openai_client,
            )
            tA = time.monotonic() - tA
            judgeA, judgeA_hit = await judge_retrieval(
                task_prompt, resA.hits,
                cache=caches["judge"], openai_client=openai_client,
            )

            # System B
            tB = time.monotonic()
            resB: ProactiveResult = await run_proactive(
                mem, task_prompt, participants,
                K_per_need=args.K_per_need,
                K_final=args.K,
                max_rounds=args.max_rounds,
                decompose_cache=caches["decompose"],
                cuegen_cache=caches["cuegen"],
                sufficiency_cache=caches["sufficiency"],
                openai_client=openai_client,
            )
            tB = time.monotonic() - tB
            judgeB, judgeB_hit = await judge_retrieval(
                task_prompt, resB.hits,
                cache=caches["judge"], openai_client=openai_client,
            )

            row = {
                "task_id": task["task_id"],
                "conversation_id": cid,
                "task_shape": task.get("task_shape", ""),
                "task_prompt": task_prompt,
                "required_info_categories": task.get("required_info_categories", []),
                "system_A": {
                    "metadata": resA.metadata,
                    "judge": judgeA,
                    "judge_cache_hit": judgeA_hit,
                    "time_s": round(tA, 2),
                    "n_llm_calls": resA.metadata["n_llm_calls"],
                    "n_turns_retrieved": resA.metadata["n_turns_retrieved"],
                    "hits_turn_ids": [h.turn_id for h in resA.hits],
                    "hits_preview": [
                        {"turn_id": h.turn_id, "role": h.role, "text": h.text[:200]}
                        for h in sorted(resA.hits, key=lambda h: h.turn_id)
                    ],
                },
                "system_B": {
                    "metadata": resB.metadata,
                    "judge": judgeB,
                    "judge_cache_hit": judgeB_hit,
                    "time_s": round(tB, 2),
                    "n_llm_calls": resB.metadata["n_llm_calls"],
                    "n_turns_retrieved": resB.metadata["n_turns_retrieved"],
                    "hits_turn_ids": [h.turn_id for h in resB.hits],
                    "hits_preview": [
                        {"turn_id": h.turn_id, "role": h.role, "text": h.text[:200]}
                        for h in sorted(resB.hits, key=lambda h: h.turn_id)
                    ],
                },
            }
            rows.append(row)

            # Incremental cache save.
            for c in caches.values():
                c.save()

            print(
                f"[{task['task_id']}] ({cid}) "
                f"A: suff={judgeA['sufficiency']} cov={judgeA['coverage']} "
                f"depth={judgeA['depth']} noise={judgeA['noise']} "
                f"turns={resA.metadata['n_turns_retrieved']} calls={resA.metadata['n_llm_calls']} "
                f"({tA:.1f}s) | "
                f"B: suff={judgeB['sufficiency']} cov={judgeB['coverage']} "
                f"depth={judgeB['depth']} noise={judgeB['noise']} "
                f"turns={resB.metadata['n_turns_retrieved']} calls={resB.metadata['n_llm_calls']} "
                f"rounds={resB.metadata['rounds_executed']} "
                f"({tB:.1f}s)"
            )

    finally:
        for c in caches.values():
            c.save()
        for coll, part in opened_resources:
            await segment_store.close_partition(part)
            await vector_store.close_collection(collection=coll)
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    # ----- Aggregate -----
    n = len(rows)
    def _mean(xs: list[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    agg = {
        "n_tasks": n,
        "system_A": {
            "mean_sufficiency": round(_mean([r["system_A"]["judge"]["sufficiency"] for r in rows]), 3),
            "mean_coverage": round(_mean([r["system_A"]["judge"]["coverage"] for r in rows]), 3),
            "mean_depth": round(_mean([r["system_A"]["judge"]["depth"] for r in rows]), 3),
            "mean_noise": round(_mean([r["system_A"]["judge"]["noise"] for r in rows]), 3),
            "mean_llm_calls": round(_mean([r["system_A"]["n_llm_calls"] for r in rows]), 2),
            "mean_turns_retrieved": round(_mean([r["system_A"]["n_turns_retrieved"] for r in rows]), 2),
            "mean_time_s": round(_mean([r["system_A"]["time_s"] for r in rows]), 2),
        },
        "system_B": {
            "mean_sufficiency": round(_mean([r["system_B"]["judge"]["sufficiency"] for r in rows]), 3),
            "mean_coverage": round(_mean([r["system_B"]["judge"]["coverage"] for r in rows]), 3),
            "mean_depth": round(_mean([r["system_B"]["judge"]["depth"] for r in rows]), 3),
            "mean_noise": round(_mean([r["system_B"]["judge"]["noise"] for r in rows]), 3),
            "mean_llm_calls": round(_mean([r["system_B"]["n_llm_calls"] for r in rows]), 2),
            "mean_turns_retrieved": round(_mean([r["system_B"]["n_turns_retrieved"] for r in rows]), 2),
            "mean_rounds": round(_mean([r["system_B"]["metadata"].get("rounds_executed", 1) for r in rows]), 2),
            "mean_needs": round(_mean([len(r["system_B"]["metadata"].get("needs", [])) for r in rows]), 2),
            "mean_time_s": round(_mean([r["system_B"]["time_s"] for r in rows]), 2),
        },
    }
    wins_A = sum(1 for r in rows if r["system_A"]["judge"]["sufficiency"] > r["system_B"]["judge"]["sufficiency"])
    wins_B = sum(1 for r in rows if r["system_B"]["judge"]["sufficiency"] > r["system_A"]["judge"]["sufficiency"])
    ties = n - wins_A - wins_B
    agg["per_task_winners"] = {"A_wins": wins_A, "B_wins": wins_B, "ties": ties}

    # Cost-per-sufficiency-point
    def _cps(calls: float, suff: float) -> float:
        if suff <= 0:
            return float("inf")
        return round(calls / suff, 4)

    agg["cost_per_sufficiency_point"] = {
        "A": _cps(agg["system_A"]["mean_llm_calls"], agg["system_A"]["mean_sufficiency"]),
        "B": _cps(agg["system_B"]["mean_llm_calls"], agg["system_B"]["mean_sufficiency"]),
    }

    # Breakdown by #required_info_categories
    by_ncat: dict[int, dict] = {}
    for r in rows:
        nc = len(r.get("required_info_categories", []))
        by_ncat.setdefault(nc, {"n": 0, "A_suff": [], "B_suff": []})
        by_ncat[nc]["n"] += 1
        by_ncat[nc]["A_suff"].append(r["system_A"]["judge"]["sufficiency"])
        by_ncat[nc]["B_suff"].append(r["system_B"]["judge"]["sufficiency"])
    ncat_summary = {}
    for nc, d in by_ncat.items():
        ncat_summary[str(nc)] = {
            "n": d["n"],
            "A_mean_suff": round(_mean(d["A_suff"]), 3),
            "B_mean_suff": round(_mean(d["B_suff"]), 3),
            "d_B_minus_A": round(_mean(d["B_suff"]) - _mean(d["A_suff"]), 3),
        }
    agg["by_n_required_categories"] = ncat_summary

    results = {
        "config": {
            "model": V2F_MODEL,
            "K": args.K,
            "K_per_need": args.K_per_need,
            "max_rounds": args.max_rounds,
        },
        "aggregate": agg,
        "per_task": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "proactive_memory.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    md = build_markdown_report(results)
    out_md = RESULTS_DIR / "proactive_memory.md"
    out_md.write_text(md)
    print(f"Saved: {out_md}")


def build_markdown_report(results: dict) -> str:
    agg = results["aggregate"]
    rows = results["per_task"]
    config = results["config"]
    A = agg["system_A"]
    B = agg["system_B"]
    w = agg["per_task_winners"]

    lines: list[str] = []
    lines.append("# Proactive memory (task-sufficiency evaluation)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- n_tasks = {agg['n_tasks']} task-shaped prompts across LoCoMo conv-26, 30, 41")
    lines.append(f"- Model: {config['model']} (fixed); text-embedding-3-small")
    lines.append(f"- K={config['K']} final turns, K_per_need={config['K_per_need']}, max_rounds={config['max_rounds']}")
    lines.append("- Backend: existing arc_em_lc30_v1_{26,30,41} EventMemory (reused)")
    lines.append("- Caches: `cache/proactive_{decompose,cuegen,sufficiency}_cache.json`, `proactive_singleshot_cuegen_cache.json`, `proactive_judge_cache.json`")
    lines.append("")

    lines.append("## Systems")
    lines.append("")
    lines.append("- **System A (single-shot)**: 1 LLM call -> 2 speaker-format cues -> retrieve top-K, merged with primer from the raw task prompt.")
    lines.append("- **System B (proactive)**: Decompose (1 call) -> per-need cue-gen (N calls) -> sufficiency audit (1 call) -> follow-up probes for under-covered needs. Max rounds = 2.")
    lines.append("")

    lines.append("## Task distribution")
    lines.append("")
    # shape counts
    shape_counts: dict[str, int] = {}
    for r in rows:
        s = r.get("task_shape", "unknown")
        shape_counts[s] = shape_counts.get(s, 0) + 1
    lines.append("Task shapes: " + ", ".join(f"{k}={v}" for k, v in sorted(shape_counts.items())))
    ncat = [len(r.get("required_info_categories", [])) for r in rows]
    lines.append(f"Required info categories per task: min={min(ncat)}, max={max(ncat)}, mean={sum(ncat)/max(len(ncat),1):.2f}")
    lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | System A | System B | d (B-A) |")
    lines.append("| --- | --- | --- | --- |")
    for label, key in [
        ("sufficiency (0-10)", "mean_sufficiency"),
        ("coverage (0-10)", "mean_coverage"),
        ("depth (0-10)", "mean_depth"),
        ("noise, higher=less-noise (0-10)", "mean_noise"),
        ("LLM calls / task", "mean_llm_calls"),
        ("turns retrieved / task", "mean_turns_retrieved"),
        ("time (s) / task", "mean_time_s"),
    ]:
        a = A.get(key, 0)
        b = B.get(key, 0)
        lines.append(f"| {label} | {a} | {b} | {b - a:+.3f} |")
    lines.append(f"| rounds executed (B) | - | {B.get('mean_rounds', 1)} | - |")
    lines.append(f"| info-needs decomposed (B) | - | {B.get('mean_needs', 0)} | - |")
    lines.append("")
    lines.append(f"**Per-task winners**: A={w['A_wins']}, B={w['B_wins']}, ties={w['ties']}")
    lines.append("")
    cps = agg["cost_per_sufficiency_point"]
    lines.append(f"**LLM calls per sufficiency point** (lower is better): A={cps['A']}, B={cps['B']}")
    lines.append("")

    lines.append("## Sufficiency by #required info categories")
    lines.append("")
    lines.append("| #categories | n | A mean suff | B mean suff | d (B-A) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for nc, d in sorted(agg["by_n_required_categories"].items(), key=lambda x: int(x[0])):
        lines.append(
            f"| {nc} | {d['n']} | {d['A_mean_suff']} | {d['B_mean_suff']} | {d['d_B_minus_A']:+.3f} |"
        )
    lines.append("")

    # Per-task table
    lines.append("## Per-task scores")
    lines.append("")
    lines.append("| task_id | conv | shape | #cats | A suff | B suff | d | A calls | B calls | B rounds |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        sA = r["system_A"]["judge"]["sufficiency"]
        sB = r["system_B"]["judge"]["sufficiency"]
        lines.append(
            f"| {r['task_id']} | {r['conversation_id'][-2:]} | {r.get('task_shape','?')} | "
            f"{len(r.get('required_info_categories', []))} | "
            f"{sA} | {sB} | {sB - sA:+d} | "
            f"{r['system_A']['n_llm_calls']} | {r['system_B']['n_llm_calls']} | "
            f"{r['system_B']['metadata'].get('rounds_executed', 1)} |"
        )
    lines.append("")

    # Qualitative examples: pick 2 tasks -- a big-B-wins and a big-A-wins (or tie).
    sorted_by_diff = sorted(rows, key=lambda r: r["system_B"]["judge"]["sufficiency"] - r["system_A"]["judge"]["sufficiency"])
    worst_for_B = sorted_by_diff[0] if sorted_by_diff else None
    best_for_B = sorted_by_diff[-1] if sorted_by_diff else None
    example_rows = []
    if best_for_B is not None:
        example_rows.append(("Largest B lead" if best_for_B is not worst_for_B else "Example", best_for_B))
    if worst_for_B is not None and worst_for_B is not best_for_B:
        example_rows.append(("Largest A lead (or tie closest to A winning)", worst_for_B))

    lines.append("## Qualitative examples")
    lines.append("")
    for label, r in example_rows:
        lines.append(f"### {label}: {r['task_id']} ({r['conversation_id']}, {r.get('task_shape','?')})")
        lines.append("")
        lines.append(f"**Task**: {r['task_prompt']}")
        lines.append("")
        lines.append(f"**Required info**: {', '.join(r.get('required_info_categories', []))}")
        lines.append("")
        lines.append(
            f"System A: sufficiency={r['system_A']['judge']['sufficiency']}, "
            f"coverage={r['system_A']['judge']['coverage']}, "
            f"depth={r['system_A']['judge']['depth']}, "
            f"noise={r['system_A']['judge']['noise']}, "
            f"turns={r['system_A']['n_turns_retrieved']}, "
            f"calls={r['system_A']['n_llm_calls']}"
        )
        lines.append("")
        lines.append(f"A judge reasoning: {r['system_A']['judge']['brief_reasoning']}")
        lines.append("")
        lines.append(f"A cues: {r['system_A']['metadata'].get('cues', [])}")
        lines.append("")
        lines.append(
            f"System B: sufficiency={r['system_B']['judge']['sufficiency']}, "
            f"coverage={r['system_B']['judge']['coverage']}, "
            f"depth={r['system_B']['judge']['depth']}, "
            f"noise={r['system_B']['judge']['noise']}, "
            f"turns={r['system_B']['n_turns_retrieved']}, "
            f"calls={r['system_B']['n_llm_calls']}, "
            f"rounds={r['system_B']['metadata'].get('rounds_executed', 1)}"
        )
        lines.append("")
        lines.append(f"B judge reasoning: {r['system_B']['judge']['brief_reasoning']}")
        lines.append("")
        lines.append("B decomposed needs:")
        for n in r["system_B"]["metadata"].get("needs", []):
            lines.append(
                f"  - ({n.get('priority','')}) {n.get('need','')}  "
                f"cues={n.get('cues', [])}  "
                f"followups={n.get('followup_probes', [])}"
            )
        lines.append("")
        # Final coverage from last audit
        fcc = r["system_B"]["metadata"].get("final_coverage_counts", {})
        if fcc:
            lines.append(f"B final coverage counts: {fcc}")
        lines.append("")

    # Verdict
    d_suff = B["mean_sufficiency"] - A["mean_sufficiency"]
    lines.append("## Verdict")
    lines.append("")
    if d_suff >= 1.0:
        lines.append(
            f"- **Proactive decomposition materially helps** on task-shaped inputs: "
            f"B beats A by d={d_suff:+.2f} sufficiency points (>=1.0 threshold)."
        )
    elif abs(d_suff) < 1.0 and d_suff >= -0.1:
        lines.append(
            f"- **B ties A** (d={d_suff:+.2f}, <1.0 threshold): decomposition does "
            "not add clear value on this corpus, OR the benchmark is too simple."
        )
    else:
        lines.append(
            f"- **B loses to A** (d={d_suff:+.2f}): extra LLM calls introduce noise "
            "or decomposition over-narrows the search."
        )
    cps_better = "A" if cps["A"] < cps["B"] else ("B" if cps["B"] < cps["A"] else "tie")
    lines.append(f"- LLM-calls-per-sufficiency-point better: **{cps_better}** (A={cps['A']}, B={cps['B']}).")
    # per-ncat trend
    ncat_items = sorted(agg["by_n_required_categories"].items(), key=lambda x: int(x[0]))
    if ncat_items:
        diffs = [d["d_B_minus_A"] for _, d in ncat_items]
        trend = "B's advantage grows with #required categories" if all(diffs[i] <= diffs[i+1] for i in range(len(diffs)-1)) and len(diffs) > 1 else "no monotonic trend with #required categories"
        lines.append(f"- Per-#categories breakdown: {trend}.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `results/proactive_memory.json`")
    lines.append("- `results/proactive_memory.md`")
    lines.append("- Source: `proactive_memory.py`, `proactive_eval.py`")
    lines.append("- Task set: `data/proactive_tasks.json`")
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
