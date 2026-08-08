"""Per-scenario runner. Ingest scenario into EM, run agent, score with judges.

Usage (from `extra_memory/evaluation`):
  uv run python -m hard_bench.runner --family guideline --scenario TG01 --channels em_cosine
  uv run python -m hard_bench.runner --family guideline --limit 3 --channels em_cosine,em_pattern_v15
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import time
from pathlib import Path


def _resolve_current_time(scenario: dict, bench_metadata: dict) -> dt.datetime | None:
    """Determine the fixed 'now' for a scenario.

    Priority:
      1. scenario['current_time'] (per-scenario override)
      2. bench_metadata['current_time'] (Family D sets this)
      3. None → build_system derives max(turn_ts) + 1 day
    """
    raw = scenario.get("current_time") or bench_metadata.get("current_time")
    if raw:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return None


# Ensure parent directory is importable (so we can use relative imports)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from hard_bench.agent import run_agent
from hard_bench.judge import judge_guideline, judge_subdecision, plant_retrieved
from hard_bench.system import build_system, make_infrastructure

DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def run_one_guideline(
    scenario: dict,
    *,
    bench_metadata: dict,
    qdrant_client,
    segment_store,
    embedder,
    openai_client,
    llm_cache,
    channels: tuple[str, ...],
    temporal_infra=None,
    entity_infra=None,
    k_per_probe: int = 3,
) -> dict:
    sid = scenario["scenario_id"]
    print(f"  [{sid}] building system...")
    t0 = time.monotonic()
    system = await build_system(
        scenario_id=sid,
        memory_turns=scenario["memory_turns"],
        qdrant_client=qdrant_client,
        segment_store=segment_store,
        embedder=embedder,
        openai_client=openai_client,
        llm_cache=llm_cache,
        overwrite=True,
        current_time=_resolve_current_time(scenario, bench_metadata),
        temporal_infra=temporal_infra,
        entity_infra=entity_infra,
    )
    t_build = time.monotonic() - t0

    print(f"  [{sid}] running agent on {len(channels)} channels...")
    t0 = time.monotonic()
    agent_result = await run_agent(
        scenario["task_prompt"],
        system,
        channels=channels,
        k_per_probe=k_per_probe,
        max_phase1_rounds=4,  # smaller for guideline (less memory)
        per_step_probe_rounds=1,
    )
    t_agent = time.monotonic() - t0

    print(f"  [{sid}] judging...")
    g = scenario["gold_guideline"]
    judgment = await judge_guideline(
        system,
        scenario["task_prompt"],
        g["guideline_text"],
        g["violation_in_task"],
        g["recommended_alternative"],
        agent_result.final_transcript,
    )

    return {
        "scenario_id": sid,
        "category": scenario["category"],
        "obscurity": scenario["obscurity"],
        "channels": list(channels),
        "agent_result": agent_result.to_dict(),
        "guideline_judgment": judgment,
        "timing": {"build_s": round(t_build, 2), "agent_s": round(t_agent, 2)},
    }


async def run_one_long(
    scenario: dict,
    *,
    bench_metadata: dict,
    qdrant_client,
    segment_store,
    embedder,
    openai_client,
    llm_cache,
    channels: tuple[str, ...],
    temporal_infra=None,
    entity_infra=None,
    k_per_probe: int = 3,
) -> dict:
    sid = scenario["scenario_id"]
    print(f"  [{sid}] building system...")
    t0 = time.monotonic()
    system = await build_system(
        scenario_id=sid,
        memory_turns=scenario["memory_turns"],
        qdrant_client=qdrant_client,
        segment_store=segment_store,
        embedder=embedder,
        openai_client=openai_client,
        llm_cache=llm_cache,
        overwrite=True,
        current_time=_resolve_current_time(scenario, bench_metadata),
        temporal_infra=temporal_infra,
        entity_infra=entity_infra,
    )
    t_build = time.monotonic() - t0

    print(f"  [{sid}] running agent on {len(channels)} channels...")
    t0 = time.monotonic()
    agent_result = await run_agent(
        scenario["task_prompt"],
        system,
        channels=channels,
        k_per_probe=k_per_probe,
        max_phase1_rounds=6,
        per_step_probe_rounds=2,
    )
    t_agent = time.monotonic() - t0

    print(f"  [{sid}] judging {len(scenario['gold_subdecisions'])} subdecisions...")
    # Build a quick turn_id → text map for gold_text lookup in judge prompt
    turn_text = {t["turn_id"]: t["text"] for t in scenario["memory_turns"]}
    plant_to_turn = {
        t.get("plant_id"): t["turn_id"]
        for t in scenario["memory_turns"]
        if t.get("plant_id")
    }

    sub_judgments = []
    # Collect all hits across phase1 + step exec for plant_retrieved check.
    # Includes both event-memory hits AND the source-turn properties of any
    # entity-memory facts the agent accumulated — both surface real turns.
    all_hit_props = []
    for so in agent_result.step_outputs:
        for h in so.get("hits", []):
            all_hit_props.append(h.properties if hasattr(h, "properties") else {})
        for ef in so.get("entity_facts", []):
            all_hit_props.append(ef.get("source_turn_properties", {}))
    for h in agent_result.phase1_hits:
        all_hit_props.append(h.properties)
    for ef in agent_result.phase1_entity_facts:
        all_hit_props.append(ef.get("source_turn_properties", {}))

    for sub in scenario["gold_subdecisions"]:
        gold_pid = sub["gold_plant_ids"][0] if sub["gold_plant_ids"] else None
        gold_text = ""
        if gold_pid and gold_pid in plant_to_turn:
            gold_text = turn_text.get(plant_to_turn[gold_pid], "")

        j = await judge_subdecision(
            system,
            scenario["task_prompt"],
            sub["description"],
            gold_text,
            agent_result.final_transcript,
        )
        retrieved = plant_retrieved(sub["gold_plant_ids"], all_hit_props)
        sub_judgments.append(
            {
                "subdecision_id": sub["subdecision_id"],
                "memory_capability_required": sub.get("memory_capability_required"),
                "gold_plant_ids": sub["gold_plant_ids"],
                "addressed": j["addressed"],
                "plant_retrieved": retrieved,
                "step_label": j.get("step_label"),
                "evidence_quote": j.get("evidence_quote", ""),
            }
        )

    return {
        "scenario_id": sid,
        "category": scenario["category"],
        "channels": list(channels),
        "agent_result": agent_result.to_dict(),
        "subdecision_judgments": sub_judgments,
        "timing": {"build_s": round(t_build, 2), "agent_s": round(t_agent, 2)},
    }


async def run_one_qa(
    scenario: dict,
    *,
    bench_metadata: dict,
    qdrant_client,
    segment_store,
    embedder,
    openai_client,
    llm_cache,
    channels: tuple[str, ...],
    temporal_infra=None,
    entity_infra=None,
    k_per_probe: int = 3,
) -> dict:
    sid = scenario["scenario_id"]
    print(f"  [{sid}] building system...")
    t0 = time.monotonic()
    system = await build_system(
        scenario_id=sid,
        memory_turns=scenario["memory_turns"],
        qdrant_client=qdrant_client,
        segment_store=segment_store,
        embedder=embedder,
        openai_client=openai_client,
        llm_cache=llm_cache,
        overwrite=True,
        current_time=_resolve_current_time(scenario, bench_metadata),
        temporal_infra=temporal_infra,
        entity_infra=entity_infra,
    )
    t_build = time.monotonic() - t0

    from hard_bench.judge import judge_qa

    print(f"  [{sid}] running agent...")
    t0 = time.monotonic()
    agent_result = await run_agent(
        scenario["task_prompt"],
        system,
        channels=channels,
        k_per_probe=k_per_probe,
        max_phase1_rounds=4,
        per_step_probe_rounds=1,
    )
    t_agent = time.monotonic() - t0

    print(f"  [{sid}] judging...")
    j = await judge_qa(
        system,
        scenario["task_prompt"],
        scenario["gold_answer"],
        agent_result.final_transcript,
    )

    # Plant retrieval check
    all_hit_props = [h.properties for h in agent_result.phase1_hits]
    for so in agent_result.step_outputs:
        for h in so.get("hits", []):
            all_hit_props.append(h.properties if hasattr(h, "properties") else {})
    retrieved = plant_retrieved(scenario["gold_evidence_plant_ids"], all_hit_props)

    return {
        "scenario_id": sid,
        "category": scenario.get("category"),
        "difficulty": scenario.get("difficulty"),
        "channels": list(channels),
        "agent_result": agent_result.to_dict(),
        "qa_judgment": j,
        "plant_retrieved": retrieved,
        "timing": {"build_s": round(t_build, 2), "agent_s": round(t_agent, 2)},
    }


async def run_one_temporal(
    scenario: dict,
    *,
    bench_metadata: dict,
    qdrant_client,
    segment_store,
    embedder,
    openai_client,
    llm_cache,
    channels: tuple[str, ...],
    temporal_infra=None,
    entity_infra=None,
    k_per_probe: int = 3,
) -> dict:
    sid = scenario["scenario_id"]
    print(f"  [{sid}] building system...")
    t0 = time.monotonic()
    system = await build_system(
        scenario_id=sid,
        memory_turns=scenario["memory_turns"],
        qdrant_client=qdrant_client,
        segment_store=segment_store,
        embedder=embedder,
        openai_client=openai_client,
        llm_cache=llm_cache,
        overwrite=True,
        current_time=_resolve_current_time(scenario, bench_metadata),
        temporal_infra=temporal_infra,
        entity_infra=entity_infra,
    )
    t_build = time.monotonic() - t0

    from hard_bench.judge import judge_temporal

    print(f"  [{sid}] running agent...")
    t0 = time.monotonic()
    agent_result = await run_agent(
        scenario["task_prompt"],
        system,
        channels=channels,
        k_per_probe=k_per_probe,
        max_phase1_rounds=4,
        per_step_probe_rounds=1,
    )
    t_agent = time.monotonic() - t0

    print(f"  [{sid}] judging...")
    j = await judge_temporal(
        system,
        scenario["task_prompt"],
        scenario.get("anchor_resolution", ""),
        scenario["gold_answer"],
        agent_result.final_transcript,
    )

    all_hit_props = [h.properties for h in agent_result.phase1_hits]
    for so in agent_result.step_outputs:
        for h in so.get("hits", []):
            all_hit_props.append(h.properties if hasattr(h, "properties") else {})
    retrieved_in_window = plant_retrieved(
        scenario["gold_evidence_plant_ids"], all_hit_props
    )
    retrieved_oow_decoy = plant_retrieved(
        scenario.get("out_of_window_decoy_plant_ids", []), all_hit_props
    )

    return {
        "scenario_id": sid,
        "anchor_type": scenario.get("anchor_type"),
        "channels": list(channels),
        "agent_result": agent_result.to_dict(),
        "temporal_judgment": j,
        "plant_retrieved": retrieved_in_window,
        "oow_decoy_retrieved": retrieved_oow_decoy,
        "timing": {"build_s": round(t_build, 2), "agent_s": round(t_agent, 2)},
    }


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--family", choices=["long", "guideline", "qa", "temporal"], required=True
    )
    ap.add_argument("--scenario", default=None, help="single scenario_id (e.g. TG01)")
    ap.add_argument("--limit", type=int, default=None, help="run first N scenarios")
    ap.add_argument(
        "--channels",
        default="em_cosine",
        help="comma-separated channels: em_cosine,em_pattern_v15,em_temporal,em_entity",
    )
    ap.add_argument("--out", default=None, help="output json path")
    ap.add_argument(
        "--k",
        type=int,
        default=3,
        help="k_per_probe for retrieval calls (and pool_size for hybrid)",
    )
    ap.add_argument(
        "--data",
        default=None,
        help="optional path to scenario data json; overrides default data/task_<family>.json",
    )
    args = ap.parse_args()

    family = args.family
    data_path = Path(args.data) if args.data else DATA_DIR / f"task_{family}.json"
    with open(data_path) as f:
        data = json.load(f)
    scenarios = data["scenarios"]
    if args.scenario:
        scenarios = [s for s in scenarios if s["scenario_id"] == args.scenario]
    elif args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        print(
            f"No scenarios matched (family={family}, scenario={args.scenario}, limit={args.limit})"
        )
        return

    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    print(f"Running {len(scenarios)} {family} scenarios on channels {channels}")

    (
        qdrant_client,
        segment_store,
        embedder,
        openai_client,
        llm_cache,
    ) = await make_infrastructure()

    # Build entity_infra (shared LLM/embed cache + budget) if em_entity requested
    entity_infra = None
    if "em_entity" in channels:
        from hard_bench.system import make_entity_infra
        print("Building em_entity infrastructure (R23+R24 prose-fact + DSU)...")
        entity_infra = make_entity_infra()

    # Build temporal_infra (cross-encoder + embed_fn) if em_temporal or temporal_filter requested
    temporal_infra = None
    if "em_temporal" in channels or "temporal_filter" in channels:
        from hard_bench.system import make_temporal_infra

        print("Building temporal_retrieval infrastructure (cross-encoder loading)...")
        temporal_infra = await make_temporal_infra(openai_client)

    runner = {
        "guideline": run_one_guideline,
        "long": run_one_long,
        "qa": run_one_qa,
        "temporal": run_one_temporal,
    }[family]

    ch_str = "_".join(channels)
    k_suffix = f"_k{args.k}" if args.k != 3 else ""
    out_path = args.out or (RESULTS_DIR / f"{family}_{ch_str}{k_suffix}_results.json")

    bench_metadata = data.get("metadata", {})

    # Resume: load existing results if file exists; skip completed scenario_ids
    results: list = []
    completed_ids: set = set()
    if Path(out_path).exists():
        try:
            with open(out_path) as f:
                existing = json.load(f)
            results = existing.get("results", [])
            completed_ids = {r["scenario_id"] for r in results}
            if completed_ids:
                print(
                    f"Resuming: {len(completed_ids)} scenarios already completed in {out_path}"
                )
        except Exception as e:
            print(f"Could not load existing results ({e}); starting fresh")

    for scenario in scenarios:
        if scenario["scenario_id"] in completed_ids:
            continue
        try:
            r = await runner(
                scenario,
                bench_metadata=bench_metadata,
                qdrant_client=qdrant_client,
                segment_store=segment_store,
                embedder=embedder,
                openai_client=openai_client,
                llm_cache=llm_cache,
                channels=channels,
                temporal_infra=temporal_infra,
                entity_infra=entity_infra,
                k_per_probe=args.k,
            )
            results.append(r)
            # Save incrementally
            ch_str = "_".join(channels)
            out_path = (
                args.out or RESULTS_DIR / f"{family}_{ch_str}{k_suffix}_results.json"
            )
            with open(out_path, "w") as f:
                json.dump(
                    {"family": family, "channels": list(channels), "results": results},
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"  [{scenario['scenario_id']}] ERROR: {e}")
            import traceback

            traceback.print_exc()

    # ---- Summary -------------------------------------------------------
    print(f"\n=== Summary ({family}, channels={channels}) ===")
    if family == "guideline":
        n = len(results)
        surfaced = sum(int(r["guideline_judgment"]["surfaced"]) for r in results)
        warned = sum(int(r["guideline_judgment"]["warned"]) for r in results)
        rec_alt = sum(
            int(r["guideline_judgment"]["recommended_alternative"]) for r in results
        )
        print(
            f"  n={n}  surfaced={surfaced}/{n}  warned={warned}/{n}  rec_alt={rec_alt}/{n}"
        )
        # By obscurity
        from collections import defaultdict

        bucks = defaultdict(lambda: {"n": 0, "surfaced": 0, "warned": 0, "rec_alt": 0})
        for r in results:
            o = r.get("obscurity", "?")
            bucks[o]["n"] += 1
            bucks[o]["surfaced"] += int(r["guideline_judgment"]["surfaced"])
            bucks[o]["warned"] += int(r["guideline_judgment"]["warned"])
            bucks[o]["rec_alt"] += int(
                r["guideline_judgment"]["recommended_alternative"]
            )
        for k in ["LOW", "MEDIUM", "HIGH"]:
            b = bucks[k]
            if b["n"]:
                print(
                    f"  obs={k} (n={b['n']}): surfaced={b['surfaced']}/{b['n']}  warned={b['warned']}/{b['n']}  rec_alt={b['rec_alt']}/{b['n']}"
                )
    elif family == "long":
        n_scenarios = len(results)
        sub_total = 0
        sub_addressed = 0
        sub_retrieved = 0
        from collections import defaultdict

        cap_stats = defaultdict(lambda: {"n": 0, "addr": 0, "retr": 0})
        for r in results:
            for sj in r["subdecision_judgments"]:
                sub_total += 1
                sub_addressed += int(sj["addressed"])
                sub_retrieved += int(sj["plant_retrieved"])
                cap = sj.get("memory_capability_required", "?")
                cap_stats[cap]["n"] += 1
                cap_stats[cap]["addr"] += int(sj["addressed"])
                cap_stats[cap]["retr"] += int(sj["plant_retrieved"])
        print(f"  n_scenarios={n_scenarios}  total_subdecisions={sub_total}")
        if sub_total:
            print(
                f"    addressed:        {sub_addressed}/{sub_total} = {sub_addressed / sub_total:.2%}"
            )
            print(
                f"    plant_retrieved:  {sub_retrieved}/{sub_total} = {sub_retrieved / sub_total:.2%}"
            )
            for cap, s in sorted(cap_stats.items()):
                print(
                    f"    cap={cap:25s} n={s['n']:3d}  addressed={s['addr']}/{s['n']}  retrieved={s['retr']}/{s['n']}"
                )
    elif family == "qa":
        n = len(results)
        correct = sum(int(r["qa_judgment"]["correct"]) for r in results)
        evid = sum(int(r["qa_judgment"]["evidence_cited"]) for r in results)
        retr = sum(int(r["plant_retrieved"]) for r in results)
        print(
            f"  n={n}  correct={correct}/{n}  evidence_cited={evid}/{n}  plant_retrieved={retr}/{n}"
        )
        from collections import defaultdict

        diff_stats = defaultdict(lambda: {"n": 0, "correct": 0, "retr": 0})
        for r in results:
            d = r.get("difficulty", "?")
            diff_stats[d]["n"] += 1
            diff_stats[d]["correct"] += int(r["qa_judgment"]["correct"])
            diff_stats[d]["retr"] += int(r["plant_retrieved"])
        for d in ["EASY", "MEDIUM", "HARD"]:
            if d in diff_stats:
                s = diff_stats[d]
                print(
                    f"  diff={d:6s} n={s['n']}: correct={s['correct']}/{s['n']}  retrieved={s['retr']}/{s['n']}"
                )
    elif family == "temporal":
        n = len(results)
        correct = sum(int(r["temporal_judgment"]["correct"]) for r in results)
        respected = sum(
            int(r["temporal_judgment"]["respected_anchor"]) for r in results
        )
        retr = sum(int(r["plant_retrieved"]) for r in results)
        oow = sum(int(r["oow_decoy_retrieved"]) for r in results)
        print(
            f"  n={n}  correct={correct}/{n}  respected_anchor={respected}/{n}  plant_retrieved={retr}/{n}  oow_decoy_retrieved={oow}/{n}"
        )
        from collections import defaultdict

        anchor_stats = defaultdict(
            lambda: {"n": 0, "correct": 0, "respected": 0, "retr": 0, "oow": 0}
        )
        for r in results:
            at = r.get("anchor_type", "?")
            anchor_stats[at]["n"] += 1
            anchor_stats[at]["correct"] += int(r["temporal_judgment"]["correct"])
            anchor_stats[at]["respected"] += int(
                r["temporal_judgment"]["respected_anchor"]
            )
            anchor_stats[at]["retr"] += int(r["plant_retrieved"])
            anchor_stats[at]["oow"] += int(r["oow_decoy_retrieved"])
        for at, s in sorted(anchor_stats.items()):
            print(
                f"  anchor={at:18s} n={s['n']}: correct={s['correct']}/{s['n']}  respect={s['respected']}/{s['n']}  retr={s['retr']}/{s['n']}  oow_decoy={s['oow']}/{s['n']}"
            )

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main_async())
