"""Multi-strategy scoring of SA-full per-step retrievals.

Re-uses cached agent outputs (no fresh LLM calls) and compares different
ways of converting the agent's per-step state into a retrieval cue.

Strategies tested per gold step:
  - content        : full per-step deliverable (current SA-full default)
  - cue            : just the primary CUE: line emitted by the cue-gen call
  - all_cues_multi : multi-probe with every CUE: line emitted (1-3 per step)
  - cue_plus_cont  : cue + " " + content concatenated
  - content_head   : first 200 chars of content (focused)
  - cue_plus_head  : cue + first 200 chars of content

Usage:
    EXECUTOR_BACKEND=claude uv run python evaluation/associative_recall/mid_execution_eval_e2_score_variants.py \
        --scenarios vocab-bridge-trip-01,multi-hop-banquet-01,stacked-event-planning-01
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
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from mid_execution_eval import (  # type: ignore
    RESULTS_DIR,
    ingest_scenario,
    load_locomo_segments,
    load_scenarios,
    load_speakers,
)
from mid_execution_eval_e1 import probe_multi  # type: ignore
from mid_execution_eval_e2 import (  # type: ignore
    EXECUTOR_CACHE_FILE,
    JUDGE_CACHE_FILE,
    _SimpleCache,
    judge_coverage,
    run_freelance_executor,
)
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv(Path(__file__).resolve().parent / ".env")


def cue_content(step_rec: dict) -> list[str]:
    return [step_rec.get("content", "")]


def cue_primary(step_rec: dict) -> list[str]:
    c = step_rec.get("cue") or ""
    return [c] if c.strip() else []


def cue_all_multi(step_rec: dict) -> list[str]:
    cs = step_rec.get("all_cues") or []
    return [c for c in cs if c.strip()] or cue_primary(step_rec)


def cue_plus_content(step_rec: dict) -> list[str]:
    cue = step_rec.get("cue") or ""
    content = step_rec.get("content", "")
    return [(cue + " " + content).strip()]


def cue_content_head(step_rec: dict) -> list[str]:
    content = step_rec.get("content", "")
    return [content[:200]]


def cue_cue_plus_head(step_rec: dict) -> list[str]:
    cue = step_rec.get("cue") or ""
    content = step_rec.get("content", "")
    return [(cue + " " + content[:200]).strip()]


def cue_plan_label(step_rec: dict) -> list[str]:
    """The agent's plan-step LABEL — concise action-shaped, no markdown."""
    # Stored on the executor_out top-level "plan" array; we need to look it up
    # by step_id. Scoring code passes step_rec which includes step_id but not
    # the label. Workaround: signal via step_rec["plan_label"] if populated by
    # caller; otherwise fall back to content_head.
    label = step_rec.get("plan_label") or ""
    return [label] if label.strip() else [step_rec.get("content", "")[:200]]


def cue_label_plus_content(step_rec: dict) -> list[str]:
    label = step_rec.get("plan_label") or ""
    content = step_rec.get("content", "")
    return [(label + " " + content).strip()] if label else [content]


def cue_label_plus_cue(step_rec: dict) -> list[str]:
    label = step_rec.get("plan_label") or ""
    cue = step_rec.get("cue") or ""
    combined = (label + " " + cue).strip()
    return [combined] if combined else []


STRATEGIES = {
    "content": cue_content,
    "cue_only": cue_primary,
    "all_cues_multi": cue_all_multi,
    "cue_plus_content": cue_plus_content,
    "content_head": cue_content_head,
    "cue_plus_head": cue_cue_plus_head,
    "plan_label": cue_plan_label,
    "label_plus_content": cue_label_plus_content,
    "label_plus_cue": cue_label_plus_cue,
}


async def score_scenario(
    scenario: dict,
    locomo_segments,
    speakers_map,
    *,
    vector_store,
    segment_store,
    embedder,
    openai_client,
    executor_cache,
    judge_cache,
    K_list,
) -> dict:
    sid = scenario["scenario_id"]
    base_conv = scenario["base_conversation"]
    locomo_turns = locomo_segments[base_conv]
    speakers = speakers_map.get(base_conv) or {}
    extras = []
    for ec in scenario.get("extra_base_conversations") or []:
        extras.append((locomo_segments[ec], speakers_map.get(ec) or {}))

    memory, ingest_info = await ingest_scenario(
        scenario,
        locomo_turns,
        speakers,
        vector_store=vector_store,
        segment_store=segment_store,
        embedder=embedder,
        overwrite=True,
        extra_distractor_runs=extras or None,
    )

    # Run SA-full executor (cache hits expected)
    executor_out = await run_freelance_executor(
        scenario,
        mode="spreading_activation_full",
        openai_client=openai_client,
        cache=executor_cache,
        memory=memory,
    )
    executor_cache.save()

    # Pre-judge once per gold step (regardless of strategy — coverage is the
    # same across strategies, only the per-step retrieval cue changes).
    plants_by_id = {
        p["plant_id"]: p for p in scenario["preamble_turns"] if p.get("plant_id")
    }
    gold_steps = [s for s in scenario["subdecision_script"] if s.get("gold_plant_ids")]
    plan_label_by_id = {
        p["step_id"]: p["label"] for p in (executor_out.get("plan") or [])
    }
    # Inject plan_label into each step record so plan_label strategies can use it.
    steps_by_id = {}
    for s in executor_out["steps"]:
        s2 = dict(s)
        s2["plan_label"] = plan_label_by_id.get(s["step_id"], "")
        steps_by_id[s["step_id"]] = s2
    transcript = executor_out.get("raw", "")

    judgements = await asyncio.gather(
        *[
            judge_coverage(
                transcript=transcript,
                decision_text=g["decision_text"],
                plant_text=plants_by_id.get(g["gold_plant_ids"][0], {}).get("text", ""),
                openai_client=openai_client,
                cache=judge_cache,
            )
            for g in gold_steps
        ]
    )
    judge_cache.save()

    K_max = max(K_list)
    per_strategy_scores: dict[str, list] = {s: [] for s in STRATEGIES}

    for gold_step, judgement in zip(gold_steps, judgements):
        addressed = judgement["addressed"]
        step_label = judgement["step_label"]
        step_rec = steps_by_id.get(step_label) if isinstance(step_label, int) else None
        gold_ids = gold_step["gold_plant_ids"]

        if not addressed or not step_rec:
            for sname in STRATEGIES:
                per_strategy_scores[sname].append(
                    {
                        "step_id": gold_step["step_id"],
                        "addressed": False,
                        **{f"recall@{K}": 0.0 for K in K_list},
                    }
                )
            continue

        for sname, sfn in STRATEGIES.items():
            queries = sfn(step_rec)
            queries = [q for q in queries if q and q.strip()]
            if not queries:
                per_strategy_scores[sname].append(
                    {
                        "step_id": gold_step["step_id"],
                        "addressed": True,
                        "skipped": True,
                        **{f"recall@{K}": 0.0 for K in K_list},
                    }
                )
                continue
            hits = await probe_multi(memory, queries, K_max)
            entry = {
                "step_id": gold_step["step_id"],
                "addressed": True,
                "n_queries": len(queries),
                "queries": [q[:120] for q in queries],
            }
            for K in K_list:
                topK_pids = {h.plant_id for h in hits[:K] if h.plant_id}
                rec = sum(1 for g in gold_ids if g in topK_pids) / len(gold_ids)
                entry[f"recall@{K}"] = rec
            per_strategy_scores[sname].append(entry)

    n_gold = len(gold_steps)
    n_addressed = sum(1 for j in judgements if j["addressed"])
    coverage = n_addressed / n_gold if n_gold else 0.0

    aggregates = {"coverage_rate": round(coverage, 4)}
    for sname in STRATEGIES:
        for K in K_list:
            vals = [
                e[f"recall@{K}"]
                for e in per_strategy_scores[sname]
                if e.get("addressed")
            ]
            full_vals = [
                e[f"recall@{K}"] if e.get("addressed") else 0.0
                for e in per_strategy_scores[sname]
            ]
            if vals:
                aggregates[f"{sname}.cond_R@{K}"] = round(sum(vals) / len(vals), 4)
            if full_vals:
                aggregates[f"{sname}.full_R@{K}"] = round(
                    sum(full_vals) / len(full_vals), 4
                )

    return {
        "scenario_id": sid,
        "ingest_info": ingest_info,
        "n_gold": n_gold,
        "n_addressed": n_addressed,
        "per_strategy_scores": per_strategy_scores,
        "aggregates": aggregates,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios", default=None, help="Comma-separated scenario_ids (default: all)"
    )
    parser.add_argument("--K", default="1,3,5,10")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    K_list = sorted({int(x) for x in args.K.split(",") if x.strip()})

    scenarios = load_scenarios()
    if args.scenarios:
        wanted = {s.strip() for s in args.scenarios.split(",")}
        scenarios = [s for s in scenarios if s["scenario_id"] in wanted]

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

    sqlite_path = RESULTS_DIR / "eventmemory_mid_exec_e2_variants.sqlite3"
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

    executor_cache = _SimpleCache(EXECUTOR_CACHE_FILE)
    judge_cache = _SimpleCache(JUDGE_CACHE_FILE)

    results = []
    try:
        for scenario in scenarios:
            print(f"[score] {scenario['scenario_id']}", flush=True)
            r = await score_scenario(
                scenario,
                locomo_segments,
                speakers_map,
                vector_store=vector_store,
                segment_store=segment_store,
                embedder=embedder,
                openai_client=openai_client,
                executor_cache=executor_cache,
                judge_cache=judge_cache,
                K_list=K_list,
            )
            results.append(r)
            cov = r["aggregates"]["coverage_rate"]
            print(f"  coverage={cov}")
            for K in K_list:
                row = []
                for s in STRATEGIES:
                    v = r["aggregates"].get(f"{s}.full_R@{K}", "-")
                    row.append(f"{s}={v}")
                print(f"  K={K}: " + " | ".join(row))
    finally:
        executor_cache.save()
        judge_cache.save()
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    out_path = (
        Path(args.out)
        if args.out
        else (RESULTS_DIR / f"mid_execution_score_variants_{int(time.time())}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "K_list": K_list,
                "strategies": list(STRATEGIES),
                "scenarios": results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}\n")

    # Cross-scenario aggregate
    print("=== Cross-scenario means (full_R@K, coverage-gated) ===")
    print(f"  {'strategy':<24s} | " + " | ".join(f"R@{K}" for K in K_list))
    for sname in STRATEGIES:
        row = [sname]
        for K in K_list:
            vals = [
                r["aggregates"].get(f"{sname}.full_R@{K}")
                for r in results
                if r["aggregates"].get(f"{sname}.full_R@{K}") is not None
            ]
            row.append(f"{sum(vals) / len(vals):.3f}" if vals else "-")
        print(f"  {row[0]:<24s} | " + " | ".join(f"{c:>5}" for c in row[1:]))


if __name__ == "__main__":
    asyncio.run(main())
