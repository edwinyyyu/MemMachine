"""Mid-execution retrieval-cue benchmark — E0 scaffolding.

Per-scenario flow:

  1. Build a fresh EventMemory collection.
  2. Ingest the scenario's preamble (planted facts) at low turn_ids, then the
     base LoCoMo conversation at higher turn_ids. Both feed the same EM.
  3. For each sub-decision step, run the requested cue strategies, query EM,
     and compute triggered_recall@K against gold plant_ids.

Three cue strategies built in (E0 has no LLM cue generation — that comes in E1):

  - "task_prompt"     : the global task prompt (bad-cue baseline; expect low)
  - "decision_text"   : the agent's stated next sub-action (action-as-cue)
  - "gold_text"       : the gold plant turn's own text (perfect-cue ceiling
                        check; verifies the planted fact is retrievable in
                        principle when the cue is the fact itself)

Usage:

    uv run python evaluation/associative_recall/mid_execution_eval.py
    uv run python evaluation/associative_recall/mid_execution_eval.py --scenario presentation-01
    uv run python evaluation/associative_recall/mid_execution_eval.py --K 5,10,20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import openai
from dotenv import load_dotenv
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Content,
    Event,
    MessageContext,
    Text,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SCENARIOS_FILE = DATA_DIR / "mid_execution_scenarios.json"
SEGMENTS_FILE = DATA_DIR / "segments_extended.npz"
SPEAKERS_FILE = RESULTS_DIR / "conversation_two_speakers.json"

NAMESPACE = "arc_em_mid_exec"
# Qdrant caps collection name at 32 bytes. Prefix = 7 chars + "_" + safe
# scenario_id (<=24 chars). NAMESPACE is uncapped — keep the long form there.
COLLECTION_PREFIX = "arc_mex"

EVENT_TYPE_PLANT = "plant"
EVENT_TYPE_MESSAGE = "message"


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def load_scenarios() -> list[dict]:
    return json.loads(SCENARIOS_FILE.read_text())


def load_locomo_segments() -> dict[str, list[tuple[int, str, str]]]:
    """Return {conv_id: [(turn_id, role, text), ...]} sorted by turn_id."""
    d = np.load(SEGMENTS_FILE, allow_pickle=True)
    cids = d["conversation_ids"]
    tids = d["turn_ids"]
    texts = d["texts"]
    roles = d["roles"]
    out: dict[str, list[tuple[int, str, str]]] = {}
    for i in range(len(texts)):
        cid = str(cids[i])
        if not cid.startswith("locomo_conv-"):
            continue
        out.setdefault(cid, []).append((int(tids[i]), str(roles[i]), str(texts[i])))
    for cid in out:
        out[cid].sort(key=lambda t: t[0])
    return out


def load_speakers() -> dict[str, dict[str, str]]:
    return json.loads(SPEAKERS_FILE.read_text())["speakers"]


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


def _scenario_collection(scenario_id: str) -> str:
    safe = scenario_id.replace("-", "_")
    name = f"{COLLECTION_PREFIX}_{safe}"
    if len(name) <= 32:
        return name
    # Qdrant cap is 32 bytes; long scenario_ids get hashed for uniqueness.
    import hashlib as _h

    digest = _h.sha256(scenario_id.encode()).hexdigest()[:8]
    return f"{COLLECTION_PREFIX}_{digest}"


def _turn_ts(base: datetime, turn_id: int) -> datetime:
    return base + timedelta(seconds=60 * turn_id)


async def ingest_scenario(
    scenario: dict,
    locomo_turns: list[tuple[int, str, str]],
    speakers: dict[str, str],
    *,
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
    overwrite: bool = True,
    extra_distractor_runs: list[tuple[list[tuple[int, str, str]], dict[str, str]]]
    | None = None,
) -> tuple[EventMemory, dict]:
    """Ingest preamble + LoCoMo conversation into a fresh EM collection.

    Plant turns are assigned turn_ids 0..N-1; LoCoMo turns are shifted to
    N..N+M-1. Each plant carries a `plant_id` property for gold-matching.

    `extra_distractor_runs`: optional list of additional (locomo_turns, speakers)
    pairs to append as more distractor density (multi-conversation scenarios).
    Each extra run is appended at the next available turn_id range.
    """
    sid = scenario["scenario_id"]
    collection_name = _scenario_collection(sid)
    partition_key = collection_name

    if overwrite:
        await vector_store.delete_collection(namespace=NAMESPACE, name=collection_name)
        await segment_store.delete_partition(partition_key)

    collection = await vector_store.open_or_create_collection(
        namespace=NAMESPACE,
        name=collection_name,
        config=VectorStoreCollectionConfig(
            vector_dimensions=embedder.dimensions,
            similarity_metric=embedder.similarity_metric,
            properties_schema=EventMemory.expected_vector_store_collection_schema(),
        ),
    )
    partition = await segment_store.open_or_create_partition(partition_key)

    memory = EventMemory(
        EventMemoryParams(
            vector_store_collection=collection,
            segment_store_partition=partition,
            embedder=embedder,
            reranker=None,
            derive_sentences=False,
            max_text_chunk_length=500,
        )
    )

    base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    # preamble_turns may mix real plants (plant_id: "p0") and on-topic decoys
    # (plant_id: null). Decoys share entities with plants so a generic cue
    # cannot trivially isolate gold — they increase discrimination difficulty
    # without being scoreable. Both ingest into EM identically.
    plants = scenario["preamble_turns"]
    n_plants = len(plants)

    # --- Preamble turns (plants + decoys) at turn_ids 0..n_plants-1 --------
    plant_events = []
    for plant_tid, plant in enumerate(plants):
        pid = plant.get("plant_id")
        is_decoy = pid is None
        props = {
            "scenario_id": sid,
            "turn_id": plant_tid,
            "speaker": plant["speaker"],
            "event_type": "decoy" if is_decoy else EVENT_TYPE_PLANT,
            "plant_tag": plant.get("tag", ""),
        }
        if pid is not None:
            props["plant_id"] = pid
        ev = Event(
            uuid=uuid4(),
            timestamp=_turn_ts(base_ts, plant_tid),
            body=Content(
                context=MessageContext(source=plant["speaker"]),
                items=[Text(text=plant["text"].strip())],
            ),
            properties=props,
        )
        plant_events.append(ev)
    await memory.encode_events(plant_events)

    # --- LoCoMo distractor turns at turn_ids n_plants..n_plants+M-1 --------
    user_name = speakers.get("user") or "User"
    asst_name = speakers.get("assistant") or "Assistant"

    runs = [(locomo_turns, user_name, asst_name)]
    if extra_distractor_runs:
        for extra_turns, extra_speakers in extra_distractor_runs:
            runs.append(
                (
                    extra_turns,
                    extra_speakers.get("user") or "User",
                    extra_speakers.get("assistant") or "Assistant",
                )
            )

    distractor_events = []
    next_tid = n_plants
    for run_idx, (run_turns, u_name, a_name) in enumerate(runs):
        for orig_tid, role, text in run_turns:
            speaker = u_name if role == "user" else a_name
            ev = Event(
                uuid=uuid4(),
                timestamp=_turn_ts(base_ts, next_tid),
                body=Content(
                    context=MessageContext(source=speaker),
                    items=[Text(text=text.strip())],
                ),
                properties={
                    "scenario_id": sid,
                    "turn_id": next_tid,
                    "locomo_orig_turn_id": orig_tid,
                    "locomo_run_idx": run_idx,
                    "speaker": speaker,
                    "role": role,
                    "event_type": EVENT_TYPE_MESSAGE,
                },
            )
            distractor_events.append(ev)
            next_tid += 1

    # Encode in batches to avoid one giant embedder call.
    BATCH = 64
    for i in range(0, len(distractor_events), BATCH):
        await memory.encode_events(distractor_events[i : i + BATCH])

    n_real_plants = sum(1 for p in plants if p.get("plant_id") is not None)
    n_decoys = n_plants - n_real_plants
    info = {
        "scenario_id": sid,
        "collection_name": collection_name,
        "n_plants": n_real_plants,
        "n_decoys": n_decoys,
        "n_distractor": len(distractor_events),
        "user_name": user_name,
        "assistant_name": asst_name,
    }
    return memory, info


# --------------------------------------------------------------------------
# Probing + scoring
# --------------------------------------------------------------------------


@dataclass
class Hit:
    turn_id: int
    plant_id: str | None
    event_type: str
    score: float
    text: str  # raw text body (back-compat)
    formatted_text: str = (
        ""  # "[timestamp] source: text" via EventMemory.string_from_segment_context
    )


# Use EM's canonical formatter so retrieved snippets include timestamp prefix
# the agent can reason about (active vs stale, recency).
from memmachine_server.episodic_memory.event_memory.data_types import (  # noqa: E402
    FormatOptions as _FormatOptions,
)

_EM_FORMAT = _FormatOptions(date_style="medium", time_style="short")


async def probe(memory: EventMemory, query_text: str, K: int) -> list[Hit]:
    """Query EM with a single cue; return top-K unique-segment hits.

    Each Hit carries both raw `text` (for back-compat) and `formatted_text`
    that includes the EM-canonical `[timestamp] source: ...` prefix, which
    the agent can use to reason about recency / supersession.
    """
    if not query_text.strip():
        return []
    qr = await memory.query(query=query_text, vector_search_limit=max(K * 2, K))
    hits: list[Hit] = []
    seen: set = set()
    for sc in qr.scored_segment_contexts:
        if sc.seed_segment_uuid in seen:
            continue
        for seg in sc.segments:
            props = seg.properties or {}
            tid = int(props.get("turn_id", -1))
            if tid in {h.turn_id for h in hits}:
                break
            text = seg.block.text if hasattr(seg.block, "text") else ""
            try:
                formatted = EventMemory.string_from_segment_context(
                    [seg],
                    format_options=_EM_FORMAT,
                )
            except Exception:
                formatted = text
            hits.append(
                Hit(
                    turn_id=tid,
                    plant_id=props.get("plant_id"),
                    event_type=str(props.get("event_type", "")),
                    score=float(sc.score),
                    text=text,
                    formatted_text=formatted,
                )
            )
            seen.add(sc.seed_segment_uuid)
            break
        if len(hits) >= K:
            break
    return hits[:K]


def triggered_recall(hits: list[Hit], gold_plant_ids: list[str], K: int) -> float:
    """Recall@K of gold plant_ids among the top-K hits.

    No-op steps (gold_plant_ids == []) collapse to a "false positive rate"
    measure: we report -1.0 for them in this fn and handle separately upstream.
    """
    if not gold_plant_ids:
        return -1.0
    found = {h.plant_id for h in hits[:K] if h.plant_id}
    return sum(1 for g in gold_plant_ids if g in found) / len(gold_plant_ids)


def false_positive_rate(hits: list[Hit], K: int) -> float:
    """Fraction of top-K hits that are plant turns (for no-op steps)."""
    return sum(1 for h in hits[:K] if h.plant_id) / max(1, K)


# --------------------------------------------------------------------------
# Cue strategies (E0 — no LLM)
# --------------------------------------------------------------------------


def cue_task_prompt(scenario: dict, step: dict) -> str:
    return scenario["task_prompt"]


def cue_decision_text(scenario: dict, step: dict) -> str:
    return step["decision_text"]


def cue_gold_text(scenario: dict, step: dict) -> str:
    """Perfect-cue ceiling: concatenate the gold plants' raw text. Returns
    empty for no-op steps (skipped upstream).
    """
    gold = step.get("gold_plant_ids") or []
    if not gold:
        return ""
    plants_by_id = {p["plant_id"]: p for p in scenario["preamble_turns"]}
    parts = [plants_by_id[g]["text"] for g in gold if g in plants_by_id]
    return " ".join(parts)


CUE_STRATEGIES = {
    "task_prompt": cue_task_prompt,
    "decision_text": cue_decision_text,
    "gold_text": cue_gold_text,
}


# --------------------------------------------------------------------------
# Per-scenario sanity run
# --------------------------------------------------------------------------


async def run_scenario(
    scenario: dict,
    locomo_segments: dict[str, list[tuple[int, str, str]]],
    speakers_map: dict[str, dict[str, str]],
    *,
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
    K_list: list[int],
    overwrite: bool = True,
) -> dict:
    sid = scenario["scenario_id"]
    base_conv = scenario["base_conversation"]
    if base_conv not in locomo_segments:
        raise ValueError(f"base_conversation {base_conv!r} not in segments file")
    locomo_turns = locomo_segments[base_conv]
    speakers = speakers_map.get(base_conv) or {}

    # Optional multi-conversation distractor density.
    extra_distractor_runs = []
    for extra_conv in scenario.get("extra_base_conversations") or []:
        if extra_conv not in locomo_segments:
            raise ValueError(
                f"extra_base_conversation {extra_conv!r} not in segments file"
            )
        extra_distractor_runs.append(
            (
                locomo_segments[extra_conv],
                speakers_map.get(extra_conv) or {},
            )
        )

    t0 = time.monotonic()
    memory, ingest_info = await ingest_scenario(
        scenario,
        locomo_turns,
        speakers,
        vector_store=vector_store,
        segment_store=segment_store,
        embedder=embedder,
        overwrite=overwrite,
        extra_distractor_runs=extra_distractor_runs or None,
    )
    ingest_time = time.monotonic() - t0

    K_max = max(K_list)
    per_step: list[dict] = []
    for step in scenario["subdecision_script"]:
        gold = step.get("gold_plant_ids") or []
        is_noop = len(gold) == 0
        per_strategy: dict = {}
        for strat_name, strat_fn in CUE_STRATEGIES.items():
            cue_text = strat_fn(scenario, step)
            if not cue_text.strip():
                # gold_text on no-op is empty; skip cleanly.
                per_strategy[strat_name] = {
                    "cue": "",
                    "skipped": True,
                }
                continue
            hits = await probe(memory, cue_text, K_max)
            recalls = {}
            fps = {}
            for K in K_list:
                if is_noop:
                    fps[f"fpr@{K}"] = false_positive_rate(hits, K)
                else:
                    recalls[f"recall@{K}"] = triggered_recall(hits, gold, K)
            per_strategy[strat_name] = {
                "cue": cue_text[:200],
                "top_hits": [
                    {
                        "rank": i + 1,
                        "turn_id": h.turn_id,
                        "plant_id": h.plant_id,
                        "score": round(h.score, 4),
                        "text_preview": h.text[:120],
                    }
                    for i, h in enumerate(hits[: max(K_list)])
                ],
                **recalls,
                **fps,
            }
        per_step.append(
            {
                "step_id": step["step_id"],
                "decision_text": step["decision_text"],
                "gold_plant_ids": gold,
                "is_noop": is_noop,
                "per_strategy": per_strategy,
            }
        )

    # Aggregate per-strategy means across non-no-op steps for each K.
    aggregates: dict = {}
    for strat_name in CUE_STRATEGIES:
        for K in K_list:
            vals = [
                s["per_strategy"][strat_name].get(f"recall@{K}")
                for s in per_step
                if not s["is_noop"]
                and not s["per_strategy"][strat_name].get("skipped")
                and s["per_strategy"][strat_name].get(f"recall@{K}") is not None
            ]
            if vals:
                aggregates[f"{strat_name}.mean_recall@{K}"] = round(
                    sum(vals) / len(vals), 4
                )
        noop_vals = [
            s["per_strategy"][strat_name].get(f"fpr@{max(K_list)}")
            for s in per_step
            if s["is_noop"]
            and not s["per_strategy"][strat_name].get("skipped")
            and s["per_strategy"][strat_name].get(f"fpr@{max(K_list)}") is not None
        ]
        if noop_vals:
            aggregates[f"{strat_name}.mean_fpr@{max(K_list)}_on_noop"] = round(
                sum(noop_vals) / len(noop_vals), 4
            )

    return {
        "scenario_id": sid,
        "category": scenario.get("category", ""),
        "base_conversation": base_conv,
        "ingest_time_s": round(ingest_time, 2),
        "ingest_info": ingest_info,
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
        "--K", default="5,10,20", help="Comma-separated K values for recall@K"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the JSON result (default: results/mid_execution_eval_<ts>.json)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Reuse existing collections if present (faster reruns)",
    )
    args = parser.parse_args()

    K_list = sorted({int(x) for x in args.K.split(",") if x.strip()})

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

    sqlite_path = RESULTS_DIR / "eventmemory_mid_exec.sqlite3"
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

    results: list[dict] = []
    try:
        for scenario in scenarios:
            print(
                f"[run] {scenario['scenario_id']} (base={scenario['base_conversation']}, "
                f"plants={len(scenario['preamble_turns'])}, "
                f"steps={len(scenario['subdecision_script'])})",
                flush=True,
            )
            r = await run_scenario(
                scenario,
                locomo_segments,
                speakers_map,
                vector_store=vector_store,
                segment_store=segment_store,
                embedder=embedder,
                K_list=K_list,
                overwrite=not args.no_overwrite,
            )
            results.append(r)
            print(f"  ingested in {r['ingest_time_s']}s; aggregates:")
            for k, v in r["aggregates"].items():
                print(f"    {k} = {v}")
    finally:
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    out_path = (
        Path(args.out)
        if args.out
        else (RESULTS_DIR / f"mid_execution_eval_{int(time.time())}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "K_list": K_list,
                "n_scenarios": len(results),
                "scenarios": results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")

    print("\n=== Cross-scenario summary ===")
    for K in K_list:
        for strat in CUE_STRATEGIES:
            vals = [
                r["aggregates"].get(f"{strat}.mean_recall@{K}")
                for r in results
                if r["aggregates"].get(f"{strat}.mean_recall@{K}") is not None
            ]
            if vals:
                print(
                    f"  {strat} mean_recall@{K} across {len(vals)} scenarios = {sum(vals) / len(vals):.3f}"
                )
    K = max(K_list)
    for strat in CUE_STRATEGIES:
        vals = [
            r["aggregates"].get(f"{strat}.mean_fpr@{K}_on_noop")
            for r in results
            if r["aggregates"].get(f"{strat}.mean_fpr@{K}_on_noop") is not None
        ]
        if vals:
            print(
                f"  {strat} mean_fpr@{K}_on_noop across {len(vals)} scenarios = {sum(vals) / len(vals):.3f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
