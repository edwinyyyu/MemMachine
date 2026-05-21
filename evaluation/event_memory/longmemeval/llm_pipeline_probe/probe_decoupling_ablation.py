"""Decoupling ablation: re-assemble terse-v2 segments without the LLM.

The shipped terse-decoupled-v2 architecture gives each consumer its own
text:
  block.text         (answerer)  = terse
  text_to_embed      (embedder)  = memory + queries + raw chunk + dates
  text_to_score_bm25 (BM25)      = memory + dates

This probe asks: does each separation earn its keep? It replays the
fixed terse-v2 segmentation from a component cache (built by
dump_terse_v2_cache.py) and re-assembles the three texts per a named
VARIANT -- so every variant is measured on the IDENTICAL segmentation,
isolating the field-assignment effect with zero LLM cost.

Components per cached item: M=memory, T=terse, Q=queries, C=raw chunk
("{producer}: {chunk}"), D=date aliases. A recipe is a list of these
keys joined by newlines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import override
from uuid import uuid4

from memmachine_server.episodic_memory.event_memory.data_types import (
    DecoupledRetrievalContext,
    Event,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)

# Each variant: (block_recipe, embed_recipe, bm25_recipe).
# "cur" reproduces shipped terse-decoupled-v2.
VARIANTS: dict[str, tuple[list[str], list[str], list[str]]] = {
    # shipped 3-way decoupled architecture
    "cur": (["T"], ["M", "Q", "C", "D"], ["M", "D"]),
    # drop synthetic queries from the embed text
    "noq": (["T"], ["M", "C", "D"], ["M", "D"]),
    # drop raw chunk from the embed text (reconfirms v3)
    "noc": (["T"], ["M", "Q", "D"], ["M", "D"]),
    # BM25 scores the answer text instead of its own memory text
    "bm25terse": (["T"], ["M", "Q", "C", "D"], ["T", "D"]),
    # drop the terse field: answerer reads `memory`; retrieval unchanged
    "coupledM": (["M"], ["M", "Q", "C", "D"], ["M", "D"]),
    # drop terse + queries + chunk: keep only date-alias enrichment
    "coupledMsimple": (["M"], ["M", "D"], ["M", "D"]),
    # fully collapsed: one text (memory) for all three consumers
    "onetext": (["M"], ["M"], ["M"]),
    # embed-order ablation: permute memory / queries / raw chunk within
    # text_to_embed (dates stay last). "cur" is the M,Q,C baseline.
    "emb_mcq": (["T"], ["M", "C", "Q", "D"], ["M", "D"]),
    "emb_qmc": (["T"], ["Q", "M", "C", "D"], ["M", "D"]),
    "emb_qcm": (["T"], ["Q", "C", "M", "D"], ["M", "D"]),
    "emb_cmq": (["T"], ["C", "M", "Q", "D"], ["M", "D"]),
    "emb_cqm": (["T"], ["C", "Q", "M", "D"], ["M", "D"]),
    # date-alias ablation: drop the date aliases from text_to_embed only
    # (BM25 keeps them) -- isolates whether the EMBEDDING needs them.
    "emb_nodate": (["T"], ["M", "Q", "C"], ["M", "D"]),
    # drop date aliases everywhere -- isolates the whole v2 mechanism.
    "nodate_all": (["T"], ["M", "Q", "C"], ["M"]),
    # ADDITIVE design: the full {M,Q,C} subset lattice for text_to_embed,
    # holding block=[T] and bm25=[M,D] fixed. Removal ablations confound
    # a component's signal with the proportion-shift it causes in the
    # rest; the additive lattice isolates each. (e_mqc == emb_nodate;
    # e_mqcd == cur.)
    "e_m": (["T"], ["M"], ["M", "D"]),
    "e_q": (["T"], ["Q"], ["M", "D"]),
    "e_c": (["T"], ["C"], ["M", "D"]),
    "e_mq": (["T"], ["M", "Q"], ["M", "D"]),
    "e_mc": (["T"], ["M", "C"], ["M", "D"]),
    "e_qc": (["T"], ["Q", "C"], ["M", "D"]),
}


class CachedReassemblySegmenter(Segmenter):
    """Replays terse-v2 segments, re-assembling fields per a named variant."""

    def __init__(self, *, cache_path: str | Path, variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r}; choices: {sorted(VARIANTS)}"
            )
        self._block_recipe, self._embed_recipe, self._bm25_recipe = VARIANTS[
            variant
        ]
        with open(cache_path) as f:
            records = json.load(f)
        # Key by (group_idx, session_id, dia_id) -- dia_id alone collides
        # across conversations (871/5882 duplicated).
        self._by_key: dict[tuple, list[dict]] = {}
        for r in records:
            key = (r["partition_key"], r["session_id"], r["dia_id"])
            self._by_key.setdefault(key, []).append(r)
        for k in self._by_key:
            self._by_key[k].sort(key=lambda r: (r["index"], r["offset"]))

    @staticmethod
    def _component(key: str, rec: dict) -> str | None:
        """Render one recipe component for a cached record; None if empty."""
        if key == "M":
            return rec["memory"] or None
        if key == "T":
            return rec["terse"] or rec["memory"] or None
        if key == "Q":
            q = rec["queries"].strip()
            return f"Queries: {q}" if q else None
        if key == "C":
            c = rec["chunk"].strip()
            return f"{rec['producer']}: {c}" if c else None
        if key == "D":
            d = rec["dates"].strip()
            return f"Dates: {d}" if d else None
        raise ValueError(f"unknown component {key!r}")

    def _assemble(self, recipe: list[str], rec: dict) -> str:
        parts = [self._component(k, rec) for k in recipe]
        return "\n".join(p for p in parts if p)

    @override
    async def segment(self, event: Event) -> list[Segment]:
        props = event.properties or {}
        group_idx = props.get("group_idx")
        sid = props.get("locomo_session_id")
        dia = props.get("dia_id")
        if group_idx is None or sid is None or dia is None:
            return []
        key = (f"group_{group_idx}", sid, dia)
        out: list[Segment] = []
        for rec in self._by_key.get(key, []):
            block_text = self._assemble(self._block_recipe, rec)
            embed_text = self._assemble(self._embed_recipe, rec)
            bm25_text = self._assemble(self._bm25_recipe, rec)
            out.append(
                Segment(
                    uuid=uuid4(),
                    event_uuid=event.uuid,
                    index=rec["index"],
                    offset=rec["offset"],
                    timestamp=event.timestamp,
                    block=TextBlock(text=block_text),
                    context=DecoupledRetrievalContext(
                        producer=rec["producer"],
                        text_to_embed=embed_text,
                        text_to_score_bm25=bm25_text,
                    ),
                    properties=event.properties,
                )
            )
        return out
