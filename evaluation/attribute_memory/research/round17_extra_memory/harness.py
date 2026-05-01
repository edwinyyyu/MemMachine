"""Round 17 — wire agamemnon's `ExtraMemory` against round-14 dense_chains.

Setup notes (friction points discovered):

1. agamemnon's `ResolutionRow.metadata` collides with SQLAlchemy declarative's
   reserved `metadata` attribute, so the entry-store module fails to import as
   shipped. We work around it by loading the source, replacing the column
   attribute name with `extra_metadata` (column name kept as 'metadata'), and
   exec-ing the patched module under the canonical import path. This is a real
   bug in extra_memory and worth flagging.

2. agamemnon's `OpenAIChatCompletionsLanguageModel` is built around the
   chat-completions API and not aligned with the round-14 evaluation cache or
   gpt-5-mini's reasoning_effort. We instead implement a tiny `LanguageModel`
   adapter that wraps the round-7 `_common.llm` cache + json_repair fallback.
   The `Embedder` adapter wraps `_common.embed_batch`.

3. The vector_store ABC requires a `VectorStore`-managed collection. We use the
   round-7 InMemoryVectorStoreCollection-equivalent fake from agamemnon's
   server_tests for parity with how the production test harness exercises it.

The remainder is straightforward: round-14 turns -> MemoryIngestTurn ->
ExtraMemory.ingest_turns; cluster_id-based ref-emission/correctness grader;
ExtraMemory.query() -> reader LLM -> round-14 judge prompt.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import uuid as _uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

# ---------------------------------------------------------------------------
# Paths + agamemnon source loading
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROUND17 = HERE
RESEARCH = HERE.parent
ROUND14 = RESEARCH / "round14_chain_density"
ROUND7 = RESEARCH / "round7"

AGAMEMNON_SRV = Path("/Users/eyu/edwinyyyu/mmcc/agamemnon/packages/server/src")
AGAMEMNON_COMMON = Path("/Users/eyu/edwinyyyu/mmcc/agamemnon/packages/common/src")

if str(AGAMEMNON_SRV) not in sys.path:
    sys.path.insert(0, str(AGAMEMNON_SRV))
if str(AGAMEMNON_COMMON) not in sys.path:
    sys.path.insert(0, str(AGAMEMNON_COMMON))

# Patch sqlalchemy_entry_store before SQLAlchemy class registration runs.
_ENTRY_STORE_SRC = (
    AGAMEMNON_SRV
    / "memmachine_server"
    / "extra_memory"
    / "entry_store"
    / "sqlalchemy_entry_store.py"
)
_ENTRY_STORE_MODNAME = (
    "memmachine_server.extra_memory.entry_store.sqlalchemy_entry_store"
)
if _ENTRY_STORE_MODNAME not in sys.modules:
    _src = _ENTRY_STORE_SRC.read_text()
    _old = (
        "    metadata: MappedColumn[dict[str, JsonValue]] = mapped_column(\n"
        "        _JSON_AUTO,\n"
        "        nullable=False,\n"
        "        default=dict,\n"
        "    )"
    )
    _new = (
        "    extra_metadata: MappedColumn[dict[str, JsonValue]] = mapped_column(\n"
        '        "metadata",\n'
        "        _JSON_AUTO,\n"
        "        nullable=False,\n"
        "        default=dict,\n"
        "    )"
    )
    if _old not in _src:
        raise RuntimeError(
            "expected ResolutionRow.metadata pattern not found in agamemnon source"
        )
    _src = _src.replace(_old, _new)
    _src = _src.replace("metadata=row.metadata,", "metadata=row.extra_metadata,")
    spec = importlib.util.spec_from_loader(
        _ENTRY_STORE_MODNAME, loader=None, origin=str(_ENTRY_STORE_SRC)
    )
    _mod = importlib.util.module_from_spec(spec)
    _mod.__file__ = str(_ENTRY_STORE_SRC)
    sys.modules[_ENTRY_STORE_MODNAME] = _mod
    exec(compile(_src, str(_ENTRY_STORE_SRC), "exec"), _mod.__dict__)

sys.path.insert(0, str(ROUND14 / "scenarios"))
sys.path.insert(0, str(ROUND14 / "experiments"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import dense_chains  # noqa: E402

# Reuse round-14 grader (LLM judge prompt + budget/cache)
import run as r14_run  # noqa: E402
from _common import (  # noqa: E402
    EMBED_MODEL,
    Budget,
    Cache,
    embed_batch,
    extract_json,
    llm,
)
from memmachine_server.common.data_types import SimilarityMetric  # noqa: E402
from memmachine_server.common.embedder.embedder import Embedder  # noqa: E402
from memmachine_server.common.language_model.language_model import (  # noqa: E402
    LanguageModel,
)
from memmachine_server.common.payload_codec.payload_codec_config import (  # noqa: E402
    PlaintextPayloadCodecConfig,
)
from memmachine_server.common.vector_store.data_types import (  # noqa: E402
    VectorStoreCollectionConfig,
)
from memmachine_server.extra_memory.data_types import (  # noqa: E402
    MemoryEntry,
    MemoryIngestTurn,
)
from memmachine_server.extra_memory.entry_store.data_types import (  # noqa: E402
    EntryStorePartitionConfig,
)
from memmachine_server.extra_memory.entry_store.sqlalchemy_entry_store import (  # noqa: E402
    SQLAlchemyEntryStore,
    SQLAlchemyEntryStoreParams,
)
from memmachine_server.extra_memory.extra_memory import (  # noqa: E402
    ExtraMemory,
    ExtraMemoryParams,
)
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

# Reuse the agamemnon in-memory vector store fake from server_tests.
_INMEM_VEC_PATH = (
    Path("/Users/eyu/edwinyyyu/mmcc/agamemnon/packages/server/server_tests")
    / "memmachine_server"
    / "common"
    / "vector_store"
    / "in_memory_vector_store_collection.py"
)
spec = importlib.util.spec_from_file_location(
    "_in_memory_vector_store_collection", str(_INMEM_VEC_PATH)
)
_in_mem_vs_mod = importlib.util.module_from_spec(spec)
sys.modules["_in_memory_vector_store_collection"] = _in_mem_vs_mod
spec.loader.exec_module(_in_mem_vs_mod)  # type: ignore[union-attr]
InMemoryVectorStoreCollection = _in_mem_vs_mod.InMemoryVectorStoreCollection


# ---------------------------------------------------------------------------
# Embedder + LanguageModel adapters wired to round-7 cache/budget
# ---------------------------------------------------------------------------


class CachedEmbedder(Embedder):
    """Wraps round-7 embed_batch into the agamemnon Embedder ABC."""

    def __init__(self, cache: Cache, budget: Budget, dimensions: int = 1536):
        self._cache = cache
        self._budget = budget
        self._dimensions = dimensions

    async def ingest_embed(
        self, inputs: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        return embed_batch(list(inputs), self._cache, self._budget)

    async def search_embed(
        self, queries: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        return embed_batch(list(queries), self._cache, self._budget)

    @property
    def model_id(self) -> str:
        return EMBED_MODEL

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


T = TypeVar("T")


class CachedLanguageModel(LanguageModel):
    """Wraps round-7 llm() into the agamemnon LanguageModel ABC.

    For generate_parsed_response with MemoryWriteResponse, we ask the model to
    emit JSON (the existing extra_memory writer prompt already says so) and
    parse with json_repair.
    """

    def __init__(self, cache: Cache, budget: Budget, model: str = "gpt-5-mini"):
        self._cache = cache
        self._budget = budget
        self._model = model

    async def generate_parsed_response(
        self,
        output_format: type[T],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> T | None:
        prompt = (system_prompt or "") + "\n\n" + (user_prompt or "")
        raw = llm(prompt, self._cache, self._budget, model=self._model)
        obj = extract_json(raw)
        if obj is None:
            return None
        try:
            return output_format.model_validate(obj)  # type: ignore[attr-defined]
        except Exception:
            return None

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools=None,
        tool_choice=None,
        max_attempts: int = 1,
    ):
        prompt = (system_prompt or "") + "\n\n" + (user_prompt or "")
        text = llm(prompt, self._cache, self._budget, model=self._model)
        return text, []

    async def generate_response_with_token_usage(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools=None,
        tool_choice=None,
        max_attempts: int = 1,
    ):
        text, _ = await self.generate_response(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        return text, [], 0, 0


# ---------------------------------------------------------------------------
# Wiring: assemble ExtraMemory against in-memory vector store + sqlite entry store
# ---------------------------------------------------------------------------


async def build_extra_memory(
    cache: Cache,
    budget: Budget,
    *,
    sqlite_path: Path,
    partition_key: str = "round17",
    active_state_limit: int = 20,
    default_world: str = "real",
) -> tuple[ExtraMemory, SQLAlchemyEntryStore, Any]:
    embedder = CachedEmbedder(cache, budget)
    lm = CachedLanguageModel(cache, budget)

    # Vector store: InMemoryVectorStoreCollection from agamemnon server_tests.
    vs_config = VectorStoreCollectionConfig(
        vector_dimensions=embedder.dimensions,
        similarity_metric=embedder.similarity_metric,
        properties_schema={
            **ExtraMemory.expected_vector_store_collection_schema(),
        },
    )
    vector_collection = InMemoryVectorStoreCollection(vs_config)

    # Entry store: file-backed SQLite (StaticPool is rejected, so use NullPool
    # with aiosqlite over a file path).
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{sqlite_path}",
        future=True,
    )
    entry_store = SQLAlchemyEntryStore(
        SQLAlchemyEntryStoreParams(engine=engine),
    )
    await entry_store.startup()
    partition = await entry_store.open_or_create_partition(
        partition_key,
        EntryStorePartitionConfig(
            payload_codec_config=PlaintextPayloadCodecConfig(),
        ),
    )

    extra = ExtraMemory(
        ExtraMemoryParams(
            partition_key=partition_key,
            vector_store_collection=vector_collection,
            entry_store_partition=partition,
            embedder=embedder,
            language_model=lm,
            active_state_limit=active_state_limit,
            default_world=default_world,
        ),
    )
    return extra, entry_store, vector_collection


# ---------------------------------------------------------------------------
# Turn conversion
# ---------------------------------------------------------------------------

_TURN_BASE_TS = datetime(2026, 1, 1, tzinfo=UTC)


def turns_to_ingest(turns: list[dense_chains.Turn]) -> list[MemoryIngestTurn]:
    """Map round-14 Turn -> MemoryIngestTurn with deterministic uuid+timestamp."""
    out: list[MemoryIngestTurn] = []
    ns = _uuid.UUID("8ee7a7d2-4b8e-4f2a-9c7d-1e5f8a0b3d6c")
    for t in turns:
        u = _uuid.uuid5(ns, f"r17-turn-{t.idx}")
        ts = _TURN_BASE_TS + timedelta(seconds=t.idx)
        out.append(MemoryIngestTurn(uuid=u, timestamp=ts, text=t.text))
    return out


# ---------------------------------------------------------------------------
# Grader: cluster_id-based ref emission + correctness
# ---------------------------------------------------------------------------


@dataclass
class TurnIndex:
    by_turn: dict[int, list[MemoryEntry]]


def _turn_idx_from_ts(ts: datetime) -> int:
    return int((ts - _TURN_BASE_TS).total_seconds())


def index_entries_by_turn(entries: Iterable[MemoryEntry]) -> TurnIndex:
    by_turn: dict[int, list[MemoryEntry]] = {}
    for e in entries:
        by_turn.setdefault(_turn_idx_from_ts(e.timestamp), []).append(e)
    return TurnIndex(by_turn=by_turn)


def _find_covering_entry_em(
    idx: TurnIndex,
    turn_idx: int,
    entity_tag: str,
    pred: str,
    new_value: str,
) -> MemoryEntry | None:
    """Find the writer entry for this transition (search same-turn ±batch window)."""
    candidates: list[MemoryEntry] = []
    for ts_i in range(turn_idx, turn_idx + 6):
        for e in idx.by_turn.get(ts_i, []):
            if entity_tag in e.mentions or f"@{entity_tag.lstrip('@')}" in e.mentions:
                candidates.append(e)
    if not candidates:
        return None
    pred_full = f"{entity_tag.lstrip('@')}.{pred}".lower()
    pred_full_at = f"@{pred_full}"
    for c in candidates:
        if (
            c.predicate
            and c.predicate.lower().lstrip("@") == pred_full
            and new_value.lower() in c.text.lower()
        ):
            return c
        if (
            c.predicate
            and c.predicate.lower() == pred_full_at
            and new_value.lower() in c.text.lower()
        ):
            return c
    for c in candidates:
        if new_value.lower() in c.text.lower():
            return c
    for c in candidates:
        if c.predicate and c.predicate.lower().lstrip("@") == pred_full:
            return c
    return candidates[0]


def collect_metrics(
    turns: list[dense_chains.Turn],
    gt: dense_chains.GroundTruth,
    entries: list[MemoryEntry],
    *,
    bucket_size: int = 100,
) -> dict:
    """cluster_id-based grader.

    A transition is ref_correct if its covering entry shares cluster_id with the
    *previous* transition's covering entry on the same chain. This is the
    structurally cleaner version of round-14's refs-walk: we don't need to
    inspect refs at all — we just check cluster identity.
    """
    idx = index_entries_by_turn(entries)
    transitions = []
    for key, chain in gt.chains.items():
        prev_cluster_id: UUID | None = None
        prev_uuid: UUID | None = None
        for i, (t, v) in enumerate(chain):
            covering = _find_covering_entry_em(idx, t, key[0], key[1], v)
            is_first = i == 0
            cov_uuid = covering.uuid if covering else None
            cov_cluster = covering.cluster_id if covering else None
            cov_refs = [str(r) for r in covering.refs] if covering else []
            atag_ok = bool(covering and key[0] in covering.mentions)
            ref_emitted = bool(cov_refs)
            # cluster-based correctness: matches prior transition's cluster
            cluster_correct = (
                cov_cluster is not None
                and prev_cluster_id is not None
                and cov_cluster == prev_cluster_id
            )
            # legacy refs-walk correctness for parity with round-14
            ref_correct = (
                ref_emitted and prev_uuid is not None and str(prev_uuid) in cov_refs
            )
            transitions.append(
                {
                    "key": f"{key[0]}.{key[1]}",
                    "turn": t,
                    "value": v,
                    "is_first": is_first,
                    "covering_uuid": str(cov_uuid) if cov_uuid else None,
                    "covering_text": covering.text if covering else None,
                    "covering_cluster_id": str(cov_cluster) if cov_cluster else None,
                    "covering_predicate": covering.predicate if covering else None,
                    "covering_refs": cov_refs,
                    "expected_prev_uuid": str(prev_uuid) if prev_uuid else None,
                    "expected_prev_cluster_id": (
                        str(prev_cluster_id) if prev_cluster_id else None
                    ),
                    "emitted_entry": cov_uuid is not None,
                    "emitted_ref": ref_emitted,
                    "ref_correct": ref_correct,
                    "cluster_correct": cluster_correct,
                    "atag_ok": atag_ok,
                }
            )
            prev_uuid = cov_uuid
            prev_cluster_id = cov_cluster

    n_turns = len(turns)
    n_buckets = (n_turns + bucket_size - 1) // bucket_size
    buckets = [(i * bucket_size, (i + 1) * bucket_size) for i in range(n_buckets)]
    bucket_stats = []
    non_first = [r for r in transitions if not r["is_first"]]
    for lo, hi in buckets:
        in_b = [r for r in non_first if lo < r["turn"] <= hi]
        n_t = len(in_b)
        n_emit = sum(1 for r in in_b if r["emitted_ref"])
        n_correct_refs = sum(1 for r in in_b if r["ref_correct"])
        n_correct_cluster = sum(1 for r in in_b if r["cluster_correct"])
        n_entry = sum(1 for r in in_b if r["emitted_entry"])
        bucket_stats.append(
            {
                "range": f"({lo},{hi}]",
                "n_transitions": n_t,
                "n_entry": n_entry,
                "n_ref": n_emit,
                "n_correct": n_correct_refs,
                "n_cluster_correct": n_correct_cluster,
                "ref_emission_rate": (n_emit / n_t) if n_t else None,
                "ref_correctness_rate": (n_correct_refs / n_t) if n_t else None,
                "cluster_correctness_rate": (n_correct_cluster / n_t) if n_t else None,
            }
        )
    n_nf = len(non_first)
    summary = {
        "n_transitions_total": len(transitions),
        "n_transitions_non_first": n_nf,
        "entry_emission_rate": (
            sum(1 for r in non_first if r["emitted_entry"]) / n_nf if n_nf else None
        ),
        "ref_emission_rate": (
            sum(1 for r in non_first if r["emitted_ref"]) / n_nf if n_nf else None
        ),
        "ref_correctness_rate": (
            sum(1 for r in non_first if r["ref_correct"]) / n_nf if n_nf else None
        ),
        "cluster_correctness_rate": (
            sum(1 for r in non_first if r["cluster_correct"]) / n_nf if n_nf else None
        ),
        "atag_rate": (
            sum(1 for r in non_first if r["atag_ok"]) / n_nf if n_nf else None
        ),
        "bucket_stats": bucket_stats,
    }
    return {"summary": summary, "transitions": transitions}


# ---------------------------------------------------------------------------
# Q/A: query ExtraMemory, render contexts, ask reader LLM, judge
# ---------------------------------------------------------------------------

READER_PROMPT = """You are answering a question from a chat assistant's memory.
Use ONLY the provided memory context. Answer concisely. If the context implies
a list, list the items in chronological order.

QUESTION: {question}

MEMORY CONTEXT (most-relevant first):
{context}

ANSWER (one short paragraph or list):"""


def render_context(scored_contexts) -> str:
    lines = []
    for sc in scored_contexts:
        # scored context: list of MemoryEntry around a seed
        for e in sc.entries:
            ts_idx = _turn_idx_from_ts(e.timestamp)
            mentions = " ".join(e.mentions) if e.mentions else ""
            pred = f" pred={e.predicate}" if e.predicate else ""
            lines.append(f"[t={ts_idx} {mentions}{pred}] {e.text}")
    # de-dupe while preserving order
    seen = set()
    uniq = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            uniq.append(ln)
    return "\n".join(uniq[:80])


async def answer_question_em(
    extra: ExtraMemory,
    question: str,
    cache: Cache,
    budget: Budget,
    *,
    vector_search_limit: int = 20,
) -> str:
    result = await extra.query(
        question,
        vector_search_limit=vector_search_limit,
        expand_context=0,
        world="real",
    )
    ctx = render_context(result.scored_contexts)
    prompt = READER_PROMPT.format(question=question, context=ctx or "(empty)")
    return llm(prompt, cache, budget)


async def grade_qa(qs, extra, cache, budget):
    answers = {}
    for q in qs:
        try:
            a = (
                await answer_question_em(q.question, ans=None) if False else None
            )  # unused
        except Exception:
            pass
        ans = await answer_question_em(extra, q.question, cache, budget)
        answers[q.qid] = ans
    cache.save()
    verdicts = r14_run.grade_deterministic(qs, answers)
    det_pass = sum(1 for v in verdicts if v["passed"])
    judged = r14_run.judge_with_llm(verdicts, cache, budget)
    cache.save()
    judge_pass = sum(1 for v in judged if v["judge_correct"])
    return {
        "answers": answers,
        "verdicts": judged,
        "deterministic_pass": det_pass,
        "judge_pass": judge_pass,
        "total": len(verdicts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_variant(
    name: str,
    *,
    active_state_limit: int,
    do_qa: bool,
    cache_path: Path,
    sqlite_path: Path,
    turns: list[dense_chains.Turn],
    gt: dense_chains.GroundTruth,
    qs,
    budget: Budget,
):
    cache = Cache(cache_path)
    if sqlite_path.exists():
        sqlite_path.unlink()
    extra, _store, _vec = await build_extra_memory(
        cache,
        budget,
        sqlite_path=sqlite_path,
        partition_key=name.replace("-", "_"),
        active_state_limit=active_state_limit,
    )
    ingest_turns_in = turns_to_ingest(turns)
    print(
        f"\n=== variant: {name} (active_state_limit={active_state_limit}) "
        f"turns={len(ingest_turns_in)} ==="
    )
    print(f"[ingest] cost so far ${budget.cost():.3f} llm={budget.llm_calls}")
    entries = await extra.ingest_turns(ingest_turns_in)
    cache.save()
    print(
        f"[ingest] log size: {len(entries)}  "
        f"cost=${budget.cost():.3f} llm={budget.llm_calls} embed={budget.embed_calls}"
    )

    metrics = collect_metrics(turns, gt, entries, bucket_size=100)
    s = metrics["summary"]
    print(
        f"[metrics] ref_emission_rate={s['ref_emission_rate']:.3f}  "
        f"ref_correctness_rate(refs)={s['ref_correctness_rate']:.3f}  "
        f"cluster_correctness_rate={s['cluster_correctness_rate']:.3f}  "
        f"entry_emission_rate={s['entry_emission_rate']:.3f}"
    )
    print("[metrics] bucket curve (cluster correctness in parens):")
    for b in s["bucket_stats"]:
        rate_e = b["ref_emission_rate"]
        rate_c = b["ref_correctness_rate"]
        rate_cl = b["cluster_correctness_rate"]
        s_e = f"{rate_e:.2f}" if rate_e is not None else " -- "
        s_c = f"{rate_c:.2f}" if rate_c is not None else " -- "
        s_cl = f"{rate_cl:.2f}" if rate_cl is not None else " -- "
        print(
            f"  {b['range']:>14s}  trans={b['n_transitions']:>3d}  "
            f"emit={s_e}  refs={s_c}  cluster={s_cl}"
        )

    qa = None
    if do_qa:
        try:
            print(f"\n[QA] running {len(qs)} questions...")
            qa = await grade_qa(qs, extra, cache, budget)
            print(f"[QA] deterministic pass: {qa['deterministic_pass']}/{qa['total']}")
            print(f"[QA] judge-graded pass: {qa['judge_pass']}/{qa['total']}")
        except RuntimeError as e:
            print(f"!!! Budget stop during QA: {e}")
            cache.save()

    log_size_bytes = sqlite_path.stat().st_size if sqlite_path.exists() else 0

    return {
        "variant": name,
        "active_state_limit": active_state_limit,
        "log_size": len(entries),
        "log_size_bytes": log_size_bytes,
        "metrics_summary": s,
        "transitions": metrics["transitions"],
        "qa": qa,
    }


def main():
    # Hard cap 500 LLM, 200 embed, ~$1.50+. Stop at 80%: 400/160.
    budget = Budget(max_llm=500, max_embed=200, stop_at_llm=400, stop_at_embed=160)

    turns = dense_chains.generate()
    gt = dense_chains.ground_truth(turns)
    qs = dense_chains.build_questions(gt)
    print(f"[scenario] turns={len(turns)}  questions={len(qs)}")
    n_nf = sum(max(0, len(v) - 1) for v in gt.chains.values())
    print(f"[scenario] non-first transitions: {n_nf}")

    cache_dir = ROUND17 / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROUND17 / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Any] = {
        "scenario": {
            "name": "dense_chains",
            "n_turns": len(turns),
            "n_questions": len(qs),
            "n_non_first_transitions": n_nf,
        },
        "variants": {},
    }

    plan = [
        ("extra_memory_a20", 20, True),
    ]

    async def run_all():
        for name, asl, do_qa in plan:
            try:
                res = await run_variant(
                    name,
                    active_state_limit=asl,
                    do_qa=do_qa,
                    cache_path=cache_dir / f"{name}.json",
                    sqlite_path=cache_dir / f"{name}.sqlite",
                    turns=turns,
                    gt=gt,
                    qs=qs,
                    budget=budget,
                )
                out["variants"][name] = res
            except RuntimeError as e:
                print(f"\n!!! Budget stop on {name}: {e}")
                out["variants"][name] = {"skipped": True, "reason": str(e)}
                break
            out["budget"] = {
                "cost": budget.cost(),
                "llm_calls": budget.llm_calls,
                "embed_calls": budget.embed_calls,
            }
            (results_dir / "run.json").write_text(
                json.dumps(out, indent=2, default=str)
            )
            print(
                f"[checkpoint] cost=${budget.cost():.3f} "
                f"llm={budget.llm_calls} embed={budget.embed_calls}"
            )

    asyncio.run(run_all())
    out["budget"] = {
        "cost": budget.cost(),
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
    }
    (results_dir / "run.json").write_text(json.dumps(out, indent=2, default=str))
    print(
        f"\n[done] cost=${budget.cost():.3f} llm={budget.llm_calls} "
        f"embed={budget.embed_calls}"
    )


if __name__ == "__main__":
    main()
