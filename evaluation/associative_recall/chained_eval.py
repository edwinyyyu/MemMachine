"""Evaluate chained_proactive vs flat_proactive vs single_shot on
task-shaped prompts that stress entity discovery + per-entity lookup.

LLM-judge sufficiency scoring (0-10) on four axes:
  COVERAGE / DEPTH / NOISE / TASK-COMPLETION

Outputs:
  results/chained_proactive.json    (raw per-task rows + summary)
  results/chained_proactive.md      (markdown report)

Usage:
    uv run python evaluation/associative_recall/chained_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import openai
from chained_proactive import (
    CHAINED_CUEGEN_CACHE,
    CHAINED_ENTITY_CACHE,
    CHAINED_FLAT_CACHE,
    CHAINED_PLAN_CACHE,
    CHAINED_SUFF_CACHE,
    RetrievalResult,
    _extract_json,
    run_chained_proactive,
    run_flat_proactive,
    run_single_shot,
)
from dotenv import load_dotenv
from em_architectures import (
    BESTSHOT_LLM_CACHE,
    EM_V2F_LLM_CACHE,
    EMHit,
    _MergedLLMCache,
)
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
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CHAINED_JUDGE_CACHE = CACHE_DIR / "chained_judge_cache.json"

K_FINAL = 50
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")

JUDGE_MODEL = "gpt-5-mini"


JUDGE_PROMPT = """\
You are scoring how well a RETRIEVAL SET covers what's needed to complete a \
task. The retrieval comes from a conversation memory between {participant_1} \
and {participant_2}.

TASK:
{task_prompt}

Expected entity types to be discovered: {expected_implicit_entity_types}
Expected per-entity facts to be gathered: {expected_per_entity_facts}

RETRIEVAL SET (top {n_shown} of {n_total} turns by retrieval score):
{hits_block}

Score 0-10 (integer) on each axis:
- COVERAGE: are the main info-needs (entities + per-entity facts) present \
  somewhere in the retrieval? (10 = everything a planner would need; 0 = \
  none of it is there)
- DEPTH: enough SPECIFIC detail to inform the task? (10 = concrete details; \
  0 = generic mentions only)
- NOISE: how much irrelevant content? (10 = ALL turns are relevant, LOW \
  noise; 0 = mostly off-topic)
- TASK_COMPLETION: could a competent AI use JUST this retrieval to produce \
  a good answer? (10 = yes, easily; 0 = definitely not)

Output ONLY JSON:
{{"coverage": <int>, "depth": <int>, "noise": <int>, "task_completion": <int>,
  "notes": "<one-sentence rationale>"}}"""


def _format_hits_for_judge(
    hits: list[EMHit], max_items: int = 20, max_len: int = 200
) -> str:
    if not hits:
        return "(empty)"
    top = sorted(hits, key=lambda h: -h.score)[:max_items]
    top = sorted(top, key=lambda h: h.turn_id)
    lines = []
    for h in top:
        txt = h.text.replace("\n", " ")
        if len(txt) > max_len:
            txt = txt[:max_len] + "..."
        lines.append(f"[Turn {h.turn_id}, {h.role}]: {txt}")
    return "\n".join(lines)


def _result_to_dict(res: RetrievalResult) -> dict:
    return {
        "hits": [
            {"turn_id": h.turn_id, "score": h.score, "role": h.role, "text": h.text}
            for h in sorted(res.hits, key=lambda h: -h.score)
        ],
        "metadata": res.metadata,
    }


async def judge_retrieval(
    task: dict,
    result: RetrievalResult,
    participants: tuple[str, str],
    judge_cache: _MergedLLMCache,
    openai_client,
) -> dict:
    p1, p2 = participants
    hits_block = _format_hits_for_judge(result.hits, max_items=20, max_len=200)
    prompt = JUDGE_PROMPT.format(
        participant_1=p1,
        participant_2=p2,
        task_prompt=task["task_prompt"],
        expected_implicit_entity_types=", ".join(
            task.get("expected_implicit_entity_types") or []
        )
        or "(none)",
        expected_per_entity_facts=", ".join(task.get("expected_per_entity_facts") or [])
        or "(none)",
        n_shown=min(20, len(result.hits)),
        n_total=len(result.hits),
        hits_block=hits_block,
    )
    cached = judge_cache.get(JUDGE_MODEL, prompt)
    if cached is not None:
        raw, hit = cached, True
    else:
        resp = await openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or ""
        judge_cache.put(JUDGE_MODEL, prompt, raw)
        hit = False

    obj = _extract_json(raw) or {}
    try:
        coverage = int(obj.get("coverage", 0))
    except Exception:
        coverage = 0
    try:
        depth = int(obj.get("depth", 0))
    except Exception:
        depth = 0
    try:
        noise = int(obj.get("noise", 0))
    except Exception:
        noise = 0
    try:
        task_completion = int(obj.get("task_completion", 0))
    except Exception:
        task_completion = 0
    return {
        "coverage": max(0, min(10, coverage)),
        "depth": max(0, min(10, depth)),
        "noise": max(0, min(10, noise)),
        "task_completion": max(0, min(10, task_completion)),
        "total": coverage + depth + noise + task_completion,
        "notes": str(obj.get("notes") or "").strip(),
        "cache_hit": hit,
    }


def load_tasks() -> list[dict]:
    path = DATA_DIR / "chained_proactive_tasks.json"
    with open(path) as f:
        tasks = json.load(f)
    return tasks


def load_collections_meta() -> dict:
    with open(RESULTS_DIR / "eventmemory_collections.json") as f:
        return json.load(f)


async def evaluate_task(
    task: dict,
    memory: EventMemory,
    participants: tuple[str, str],
    *,
    single_shot_cache: _MergedLLMCache,
    plan_cache: _MergedLLMCache,
    cuegen_cache: _MergedLLMCache,
    entity_cache: _MergedLLMCache,
    suff_cache: _MergedLLMCache,
    flat_cache: _MergedLLMCache,
    judge_cache: _MergedLLMCache,
    openai_client,
) -> dict:
    t_single = time.monotonic()
    single_res = await run_single_shot(
        memory,
        task["task_prompt"],
        participants,
        K=K_FINAL,
        cuegen_cache=single_shot_cache,
        openai_client=openai_client,
    )
    t_single = time.monotonic() - t_single

    t_flat = time.monotonic()
    flat_res = await run_flat_proactive(
        memory,
        task["task_prompt"],
        participants,
        K_per_need=15,
        K_final=K_FINAL,
        flat_cache=flat_cache,
        cuegen_cache=cuegen_cache,
        openai_client=openai_client,
    )
    t_flat = time.monotonic() - t_flat

    t_chain = time.monotonic()
    chain_res = await run_chained_proactive(
        memory,
        task["task_prompt"],
        participants,
        K_per_cue=10,
        K_final=K_FINAL,
        max_iterations=2,
        plan_cache=plan_cache,
        cuegen_cache=cuegen_cache,
        entity_cache=entity_cache,
        suff_cache=suff_cache,
        openai_client=openai_client,
    )
    t_chain = time.monotonic() - t_chain

    single_judge = await judge_retrieval(
        task, single_res, participants, judge_cache, openai_client
    )
    flat_judge = await judge_retrieval(
        task, flat_res, participants, judge_cache, openai_client
    )
    chain_judge = await judge_retrieval(
        task, chain_res, participants, judge_cache, openai_client
    )

    # Save caches after each task.
    single_shot_cache.save()
    plan_cache.save()
    cuegen_cache.save()
    entity_cache.save()
    suff_cache.save()
    flat_cache.save()
    judge_cache.save()

    return {
        "task_id": task["task_id"],
        "conversation_id": task["conversation_id"],
        "task_prompt": task["task_prompt"],
        "expected_implicit_entity_types": task.get(
            "expected_implicit_entity_types", []
        ),
        "expected_per_entity_facts": task.get("expected_per_entity_facts", []),
        "participants": list(participants),
        "single_shot": {
            "result": _result_to_dict(single_res),
            "judge": single_judge,
            "time_s": round(t_single, 2),
        },
        "flat_proactive": {
            "result": _result_to_dict(flat_res),
            "judge": flat_judge,
            "time_s": round(t_flat, 2),
        },
        "chained_proactive": {
            "result": _result_to_dict(chain_res),
            "judge": chain_judge,
            "time_s": round(t_chain, 2),
        },
    }


def _avg(vals: list[float]) -> float:
    return round(sum(vals) / max(len(vals), 1), 3)


def summarise(rows: list[dict]) -> dict:
    summary = {"n_tasks": len(rows), "by_variant": {}}
    for variant in ("single_shot", "flat_proactive", "chained_proactive"):
        vals = {
            "coverage": [r[variant]["judge"]["coverage"] for r in rows],
            "depth": [r[variant]["judge"]["depth"] for r in rows],
            "noise": [r[variant]["judge"]["noise"] for r in rows],
            "task_completion": [r[variant]["judge"]["task_completion"] for r in rows],
            "total": [r[variant]["judge"]["total"] for r in rows],
            "time_s": [r[variant]["time_s"] for r in rows],
            "n_llm_calls": [
                r[variant]["result"]["metadata"].get("n_llm_calls", 0) for r in rows
            ],
            "n_turns_retrieved": [
                r[variant]["result"]["metadata"].get("n_turns_retrieved", 0)
                for r in rows
            ],
        }
        summary["by_variant"][variant] = {k: _avg(v) for k, v in vals.items()}

    # Pair comparisons
    wins = {
        "chained_vs_flat": {"chain": 0, "flat": 0, "tie": 0},
        "chained_vs_single": {"chain": 0, "single": 0, "tie": 0},
        "flat_vs_single": {"flat": 0, "single": 0, "tie": 0},
    }
    deltas_cf = []
    for r in rows:
        tc_chain = r["chained_proactive"]["judge"]["task_completion"]
        tc_flat = r["flat_proactive"]["judge"]["task_completion"]
        tc_single = r["single_shot"]["judge"]["task_completion"]
        deltas_cf.append(tc_chain - tc_flat)
        if tc_chain > tc_flat:
            wins["chained_vs_flat"]["chain"] += 1
        elif tc_chain < tc_flat:
            wins["chained_vs_flat"]["flat"] += 1
        else:
            wins["chained_vs_flat"]["tie"] += 1
        if tc_chain > tc_single:
            wins["chained_vs_single"]["chain"] += 1
        elif tc_chain < tc_single:
            wins["chained_vs_single"]["single"] += 1
        else:
            wins["chained_vs_single"]["tie"] += 1
        if tc_flat > tc_single:
            wins["flat_vs_single"]["flat"] += 1
        elif tc_flat < tc_single:
            wins["flat_vs_single"]["single"] += 1
        else:
            wins["flat_vs_single"]["tie"] += 1

    summary["wins_task_completion"] = wins
    summary["mean_delta_task_completion_chain_minus_flat"] = _avg(deltas_cf)

    # Entity discovery stats (chained).
    n_entities_discovered = [
        r["chained_proactive"]["result"]["metadata"].get("n_entities_discovered", 0)
        for r in rows
    ]
    summary["chained_entity_discovery"] = {
        "mean_entities_discovered_per_task": _avg(n_entities_discovered),
        "tasks_with_any_discovery": sum(1 for n in n_entities_discovered if n > 0),
    }
    return summary


def render_markdown(summary: dict, rows: list[dict]) -> str:
    v = summary["by_variant"]
    lines = [
        "# Chained Proactive Retrieval — Entity-Discovery DAG",
        "",
        "## Setup",
        "",
        f"- n_tasks = {summary['n_tasks']} "
        "(authored to stress implicit entity discovery + per-entity facts)",
        "- Corpora: LoCoMo-30 (conversations 26, 30, 41), reusing "
        "`arc_em_lc30_v1_{26,30,41}` EventMemory + `results/eventmemory.sqlite3`.",
        "- Embedder: `text-embedding-3-small`, Model: `gpt-5-mini` (plan / cue-gen / entity extract / sufficiency / judge).",
        f"- K_final = {K_FINAL} turns. Judge: 4 axes, 0-10 each; scored over top-20 turns by retrieval score.",
        "",
        "## Variants",
        "",
        "- `single_shot`: em_v2f_speakerformat baseline — one cue-gen, retrieve top-K.",
        "- `flat_proactive`: LLM decomposes task into 3-6 info needs (no deps), retrieves each with only the task prompt as context.",
        "- `chained_proactive`: LLM emits a DAG of `entity_discovery` + `per_entity_fact` nodes. Downstream nodes receive entities extracted from upstream hits. Up to 2 iterations; sufficiency-audit can add nodes.",
        "",
        "## Aggregate sufficiency (LLM-judge, 0-10 per axis, mean)",
        "",
        "| Variant | Coverage | Depth | Noise (higher=cleaner) | Task-Completion | Total | n_llm_calls | time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("single_shot", "flat_proactive", "chained_proactive"):
        d = v[name]
        lines.append(
            f"| `{name}` | {d['coverage']:.2f} | {d['depth']:.2f} | "
            f"{d['noise']:.2f} | {d['task_completion']:.2f} | "
            f"{d['total']:.2f} | {d['n_llm_calls']:.1f} | {d['time_s']:.2f} |"
        )

    w = summary["wins_task_completion"]
    lines += [
        "",
        "## Pairwise task-completion winners",
        "",
        f"- chained vs flat: chained={w['chained_vs_flat']['chain']}, flat={w['chained_vs_flat']['flat']}, ties={w['chained_vs_flat']['tie']}. Mean delta (chain - flat) = {summary['mean_delta_task_completion_chain_minus_flat']:+.2f}.",
        f"- chained vs single-shot: chained={w['chained_vs_single']['chain']}, single={w['chained_vs_single']['single']}, ties={w['chained_vs_single']['tie']}.",
        f"- flat vs single-shot: flat={w['flat_vs_single']['flat']}, single={w['flat_vs_single']['single']}, ties={w['flat_vs_single']['tie']}.",
        "",
        "## Entity discovery (chained only)",
        "",
        f"- mean entities discovered per task: {summary['chained_entity_discovery']['mean_entities_discovered_per_task']:.2f}",
        f"- tasks with >=1 entity discovered: {summary['chained_entity_discovery']['tasks_with_any_discovery']}/{summary['n_tasks']}",
        "",
        "## Per-task task-completion scores",
        "",
        "| Task | single | flat | chain | chain-flat | Discovered entities |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        tc_s = r["single_shot"]["judge"]["task_completion"]
        tc_f = r["flat_proactive"]["judge"]["task_completion"]
        tc_c = r["chained_proactive"]["judge"]["task_completion"]
        ents = []
        for n in r["chained_proactive"]["result"]["metadata"].get("nodes", []):
            ents.extend(n.get("extracted_entities") or [])
        ents = list(dict.fromkeys(ents))[:6]
        ents_str = ", ".join(ents) if ents else "—"
        lines.append(
            f"| `{r['task_id']}` | {tc_s} | {tc_f} | {tc_c} | "
            f"{tc_c - tc_f:+d} | {ents_str} |"
        )

    # Qualitative examples: top 2 tasks where chain beats flat the most.
    ranked = sorted(
        rows,
        key=lambda r: (
            -(
                r["chained_proactive"]["judge"]["task_completion"]
                - r["flat_proactive"]["judge"]["task_completion"]
            )
        ),
    )
    lines += ["", "## Qualitative examples — chain vs flat", ""]
    for r in ranked[:2]:
        lines += [
            f"### `{r['task_id']}` (chain {r['chained_proactive']['judge']['task_completion']} vs flat {r['flat_proactive']['judge']['task_completion']})",
            "",
            f"> {r['task_prompt']}",
            "",
            "**Chained plan:**",
            "",
        ]
        for n in r["chained_proactive"]["result"]["metadata"].get("nodes", []):
            lines.append(
                f"- `{n['id']}` ({n['type']}): {n['target']}"
                + (
                    f" — discovered: {n.get('extracted_entities')}"
                    if n.get("extracted_entities")
                    else ""
                )
                + (f" — for_each={n.get('for_each')}" if n.get("for_each") else "")
            )
        lines += [
            "",
            "**Flat needs:**",
            "",
        ]
        for n in r["flat_proactive"]["result"]["metadata"].get("needs", []):
            lines.append(f"- {n['need']}")
        lines += [
            "",
            f"Chain judge notes: {r['chained_proactive']['judge'].get('notes', '')}",
            "",
            f"Flat judge notes: {r['flat_proactive']['judge'].get('notes', '')}",
            "",
        ]

    lines += [
        "",
        "## Verdict",
        "",
    ]
    delta = summary["mean_delta_task_completion_chain_minus_flat"]
    if delta >= 1.0:
        lines.append(
            f"Chained beats flat by {delta:+.2f} on mean task-completion — the DAG "
            "structure (entity discovery feeding per-entity lookup) is the lift."
        )
    elif delta <= -0.5:
        lines.append(
            f"Chained LOSES to flat by {delta:+.2f} — DAG overhead adds noise "
            "without benefit on these LoCoMo corpora, probably because flat "
            "decomposition already surfaces the needed content."
        )
    else:
        lines.append(
            f"Chained ~ flat (delta = {delta:+.2f}). Entity discovery happens "
            "naturally in flat decomposition, probably because the LLM listing "
            "3-6 info needs already includes entity-enumeration as one of them."
        )

    lines += [
        "",
        "## Outputs",
        "",
        "- Raw: `results/chained_proactive.json`",
        "- This report: `results/chained_proactive.md`",
        "- Tasks: `data/chained_proactive_tasks.json`",
        "- Source: `chained_proactive.py`, `chained_eval.py`",
        "",
    ]
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = load_tasks()
    if args.limit is not None:
        tasks = tasks[: args.limit]

    collections_meta = load_collections_meta()
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

    # Caches.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    single_shot_cache = _MergedLLMCache(
        reader_paths=[BESTSHOT_LLM_CACHE, EM_V2F_LLM_CACHE],
        writer_path=CACHE_DIR / "chained_single_cache.json",
    )
    plan_cache = _MergedLLMCache([CHAINED_PLAN_CACHE], CHAINED_PLAN_CACHE)
    cuegen_cache = _MergedLLMCache([CHAINED_CUEGEN_CACHE], CHAINED_CUEGEN_CACHE)
    entity_cache = _MergedLLMCache([CHAINED_ENTITY_CACHE], CHAINED_ENTITY_CACHE)
    suff_cache = _MergedLLMCache([CHAINED_SUFF_CACHE], CHAINED_SUFF_CACHE)
    flat_cache = _MergedLLMCache([CHAINED_FLAT_CACHE], CHAINED_FLAT_CACHE)
    judge_cache = _MergedLLMCache([CHAINED_JUDGE_CACHE], CHAINED_JUDGE_CACHE)

    # Open EM per conversation used in task set.
    used_conv_ids = sorted({t["conversation_id"] for t in tasks})
    memories: dict[str, EventMemory] = {}
    participants_by_conv: dict[str, tuple[str, str]] = {}
    opened_resources: list = []
    for conv_id in used_conv_ids:
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
        for i, task in enumerate(tasks):
            conv_id = task["conversation_id"]
            print(f"[{i + 1}/{len(tasks)}] {task['task_id']} ({conv_id}) ...")
            row = await evaluate_task(
                task,
                memories[conv_id],
                participants_by_conv[conv_id],
                single_shot_cache=single_shot_cache,
                plan_cache=plan_cache,
                cuegen_cache=cuegen_cache,
                entity_cache=entity_cache,
                suff_cache=suff_cache,
                flat_cache=flat_cache,
                judge_cache=judge_cache,
                openai_client=openai_client,
            )
            rows.append(row)
            j = row["chained_proactive"]["judge"]
            jf = row["flat_proactive"]["judge"]
            js = row["single_shot"]["judge"]
            print(
                f"    single tc={js['task_completion']} "
                f"flat tc={jf['task_completion']} "
                f"chain tc={j['task_completion']} "
                f"(discovered: {row['chained_proactive']['result']['metadata'].get('n_entities_discovered', 0)})"
            )
    finally:
        for coll, part in opened_resources:
            await segment_store.close_partition(part)
            await vector_store.close_collection(collection=coll)
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    summary = summarise(rows)
    out_json = RESULTS_DIR / "chained_proactive.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"Saved: {out_json}")

    md = render_markdown(summary, rows)
    out_md = RESULTS_DIR / "chained_proactive.md"
    out_md.write_text(md)
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    asyncio.run(main())
