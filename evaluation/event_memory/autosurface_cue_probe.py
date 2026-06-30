"""Probe: running-context-vector cue construction for autosurfacing.

Compares cue strategies for the UserPromptSubmit ambient recall, qualitatively,
on real transcripts ingested into a throwaway space. No agent is run — we observe
what each cue retrieves and how sharply.

Strategies (cue built per replay turn; matched against the document-embedded
corpus, which is how memories are stored):

  baseline : query_embed(current message) only                  [status quo]
  run_q    : EMA of query embeddings    (current + decayed prior, normalized)
  run_d    : EMA of document embeddings (current + decayed prior, normalized)
  hybrid   : decayed *document* context prior  +  query_embed(current)

Running vectors are L2-normalized to unit length each turn (TCM-style drift):
    t_i = normalize( rho * t_{i-1} + (1 - rho) * c_i )
rho=0 collapses run_q to baseline (sanity check).

Only PRIOR memories (timestamp < the current turn) are eligible — autosurface
recalls the past given the present, it shouldn't echo the live message back.

Run with sqlitevec (exact cosine) so quantization can't confound the comparison.

    uv run python evaluation/event_memory/autosurface_cue_probe.py
"""

import asyncio
import os
import tempfile
from pathlib import Path
from uuid import UUID

import numpy as np

_HOME = tempfile.mkdtemp(prefix="cue_probe_")
os.environ["CLAUDE_MEMORY_HOME"] = _HOME
os.environ["CLAUDE_MEMORY_EMBEDDING"] = "embeddinggemma"
os.environ["CLAUDE_MEMORY_VECTOR_BACKEND"] = "sqlitevec"  # exact cosine
os.environ["CLAUDE_MEMORY_PARTITION"] = "shared"
os.environ.pop("CLAUDE_MEMORY_EVICTION_THRESHOLD", None)

PROJECTS = Path.home() / ".claude" / "projects"
REPLAY = (
    PROJECTS
    / "-Users-eyu-edwinyyyu-mmcc-agentic-expansion"
    / "dac3e83e-583a-43df-80af-21474df6de0c.jsonl"
)
CORPUS = [
    REPLAY,
    PROJECTS
    / "-Users-eyu-edwinyyyu-mmcc-temporal-scoring"
    / "97a315c1-d46f-4ad3-a177-c752281b6d80.jsonl",
    PROJECTS
    / "-Users-eyu-edwinyyyu-mmcc-memmachine-core"
    / "95027f27-6c37-497d-8f9c-61d9128b80ba.jsonl",
]
REPLAY_CAP = 4000  # events ingested from the replay session (long, multi-topic)
CORPUS_CAP = 600  # events per distractor session
MAX_TURNS = 45  # replay user turns
RHOS = [0.5, 0.8, 0.9]  # prior-weight sweep; detailed dump uses RHO_SHOW
RHO_SHOW = 0.8
TOPK = 3  # displayed neighbors
POOL = 20  # candidates fetched per query (then prior-filtered)
TERSE_WORDS = 11  # <= this many words => "terse follow-up"


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _snippet(text: str, n: int = 90) -> str:
    one = " ".join(text.split())
    return one if len(one) <= n else one[:n].rstrip() + " …"


async def main() -> None:
    from claude_memory.engine import (
        DISPLAY_FORMAT,
        EventMemory,
        MemoryCore,
    )
    from claude_memory.transcript import events_from_transcript
    from claude_memory.wire import MemoryConfig, Source

    seg_field = EventMemory._SEGMENT_UUID_FIELD_NAME
    ts_field = EventMemory._TIMESTAMP_FIELD_NAME

    core = await MemoryCore.open(MemoryConfig.load())
    embedder = core.stores.embedder
    collection = core.opened.collection
    partition = core.opened.partition

    async def embed_query(text: str) -> np.ndarray:
        return _norm(np.asarray((await embedder.search_embed([text]))[0], dtype=float))

    async def embed_doc(text: str) -> np.ndarray:
        return _norm(np.asarray((await embedder.ingest_embed([text]))[0], dtype=float))

    # --- ingest corpus -----------------------------------------------------
    print(f"home={_HOME}")
    for path in CORPUS:
        cap = REPLAY_CAP if path == REPLAY else CORPUS_CAP
        events, _ = events_from_transcript(path, session_id=path.stem, start_line=0)
        events = events[:cap]
        n = await core.ingest(events)
        msgs = sum(
            1
            for e in events
            if e.properties.get("source")
            in (Source.USER_MESSAGE, Source.ASSISTANT_MESSAGE)
        )
        print(f"  ingested {n:>4} events ({msgs} messages) from {path.parent.name}")

    # --- replay turns + precompute both embeddings once --------------------
    replay_events, _ = events_from_transcript(
        REPLAY, session_id=REPLAY.stem, start_line=0
    )
    raw_turns = [
        (e.timestamp, e.blocks[0].text)
        for e in replay_events[:REPLAY_CAP]
        if e.properties.get("source") == Source.USER_MESSAGE
        and e.blocks
        and e.blocks[0].text.strip()
        and not e.blocks[0].text.lstrip().startswith("<")  # skip hook/system injections
    ][:MAX_TURNS]
    turns = []
    for ts, text in raw_turns:
        turns.append((ts, text, await embed_query(text), await embed_doc(text)))
    print(f"\nreplaying {len(turns)} user turns from {REPLAY.parent.name}\n")

    async def search(cue: np.ndarray, before_ts) -> list[tuple[float, UUID]]:
        """Prior-only candidate list (score, seed uuid), best-first, up to POOL."""
        [qr] = await collection.query(
            query_vectors=[cue.tolist()],
            limit=POOL,
            return_vector=False,
            return_properties=True,
        )
        out: list[tuple[float, UUID]] = []
        for m in qr.matches:
            props = m.record.properties or {}
            ts = props.get(ts_field)
            if ts is not None and before_ts is not None and ts >= before_ts:
                continue
            out.append((m.score, UUID(str(props[seg_field]))))
        return out

    async def text_of(seg_uuid: UUID) -> str:
        ctx = await partition.get_segment_contexts(
            seed_segment_uuids=[seg_uuid],
            max_backward_segments=0,
            max_forward_segments=0,
            property_filter=None,
        )
        segs = ctx.get(seg_uuid)
        return (
            EventMemory.string_from_segment_context(segs, format_options=DISPLAY_FORMAT)
            if segs
            else "(missing)"
        )

    strategies = ["baseline", "run_q", "run_d", "hybrid"]

    def build_cues(cq, cd, q_prev, d_prev, rho):
        return {
            "baseline": cq,
            "run_q": _norm(rho * q_prev + (1 - rho) * cq) if q_prev is not None else cq,
            "run_d": _norm(rho * d_prev + (1 - rho) * cd) if d_prev is not None else cd,
            "hybrid": _norm(rho * d_prev + (1 - rho) * cq)
            if d_prev is not None
            else cq,
        }

    # rho sweep: aggregate mean top1 + mean separation (top1 - mean of rest of pool)
    for rho in RHOS:
        agg: dict[str, list[tuple[float, float]]] = {s: [] for s in strategies}
        q_prev = d_prev = None
        show = rho == RHO_SHOW
        if show:
            print(f"───────── detailed dump (rho={rho}) ─────────")
        for i, (ts, text, cq, cd) in enumerate(turns):
            cues = build_cues(cq, cd, q_prev, d_prev, rho)
            q_prev, d_prev = cues["run_q"], cues["run_d"]
            results = {s: await search(cues[s], ts) for s in strategies}
            for s in strategies:
                r = results[s]
                if r:
                    bg = sum(x for x, _ in r[1:]) / max(len(r) - 1, 1)
                    agg[s].append((r[0][0], r[0][0] - bg))

            words = len(text.split())
            terse = words <= TERSE_WORDS
            if show and (terse or i % 11 == 0):
                print(
                    f"── turn {i:>2} [{'TERSE' if terse else '    '}]: {_snippet(text, 78)}"
                )
                for s in strategies:
                    r = results[s]
                    if not r:
                        print(f"     {s:<9} (no prior hits)")
                        continue
                    print(f"     {s:<9} top1={r[0][0]:.3f}")
                    for score, uid in r[:TOPK]:
                        print(f"        {score:.3f}  {_snippet(await text_of(uid))}")
                print()

        print(f"===== rho={rho} =====")
        print(f"{'strategy':<10} {'mean top1':>10} {'mean sep':>10} {'turns':>7}")
        for s in strategies:
            vals = agg[s]
            mt = sum(t for t, _ in vals) / len(vals)
            ms = sum(d for _, d in vals) / len(vals)
            print(f"{s:<10} {mt:>10.3f} {ms:>10.3f} {len(vals):>7}")
        print()

    await core.aclose()


if __name__ == "__main__":
    asyncio.run(main())
