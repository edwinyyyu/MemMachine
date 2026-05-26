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

# Each variant: (block_recipe, embed_recipe, bm25_recipe) or, optionally,
# a 4-tuple (..., embed_sep) where embed_sep overrides the string joining
# the embed-recipe components (default "\n"). block/bm25 always use "\n".
# "cur" reproduces shipped terse-decoupled-v2.
VARIANTS: dict[str, tuple] = {
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
    # SEPARATOR ablation: M,C,Q embed order (the reorder-ablation top),
    # dates dropped (emb_nodate showed they are free to remove). Vary
    # only the string joining the three embed components. bm25 keeps the
    # date alias since BM25 cannot bridge "August" <-> "08".
    "sep_mcq_nl": (["T"], ["M", "C", "Q"], ["M", "D"], "\n"),
    "sep_mcq_sp": (["T"], ["M", "C", "Q"], ["M", "D"], " "),
    "sep_mcq_nl2": (["T"], ["M", "C", "Q"], ["M", "D"], "\n\n"),
    # LABEL ablation: lowercase q/c/d strip the "Queries:" / "Dates:" /
    # producer header. Baseline = sep_mcq_nl (all labels on).
    #   lab_q_off  -- drop "Queries:" header from the embed text
    #   lab_d_off  -- drop "Dates:" header from the bm25 text
    #   lab_qd_off -- drop both
    "lab_q_off": (["T"], ["M", "C", "q"], ["M", "D"], "\n"),
    "lab_d_off": (["T"], ["M", "C", "Q"], ["M", "d"], "\n"),
    "lab_qd_off": (["T"], ["M", "C", "q"], ["M", "d"], "\n"),
    # 4th-FRAMING ablation: does a fresh LLM-generated view beyond M/Q/C
    # add signal? A = atomic-fact decomposition, P = topic labels.
    # Baseline = emb_nodate (M,Q,C, no dates) = 91.30. Requires the
    # augmented cache (augment_cache_framings.py -> cache-terse-v2-aug).
    "f_mqca": (["T"], ["M", "Q", "C", "A"], ["M", "D"]),
    "f_mqcp": (["T"], ["M", "Q", "C", "P"], ["M", "D"]),
    "f_mqcap": (["T"], ["M", "Q", "C", "A", "P"], ["M", "D"]),
    "f_a": (["T"], ["A"], ["M", "D"]),
    "f_p": (["T"], ["P"], ["M", "D"]),
    # BM25-text ablation: embed fixed at M,Q,C (the settled recipe);
    # vary only text_to_score_bm25. BM25 is purely lexical, so the raw
    # chunk C -- the actual conversation words a user may echo -- is the
    # one lexically-motivated option untested so far. Baseline =
    # emb_nodate (bm25 = M,D) = 91.30.
    "bm25_c": (["T"], ["M", "Q", "C"], ["C"]),
    "bm25_cd": (["T"], ["M", "Q", "C"], ["C", "D"]),
    "bm25_mc": (["T"], ["M", "Q", "C"], ["M", "C"]),
    "bm25_mcd": (["T"], ["M", "Q", "C"], ["M", "C", "D"]),
    # T-as-anchor ablation: T is the compressed form of M. Does it work
    # as the embedding/BM25 anchor too? If yes, the segmenter can drop
    # the `memory` field entirely. Baseline = emb_nodate (M,Q,C) 91.30.
    #   embed_t    -- T replaces M in the embed text
    #   embed_mqct -- T added as a 4th embed item alongside M
    #   all_t      -- T replaces M in BOTH embed and bm25 (M unused)
    "embed_t": (["T"], ["T", "Q", "C"], ["M", "D"]),
    "embed_mqct": (["T"], ["M", "Q", "C", "T"], ["M", "D"]),
    "all_t": (["T"], ["T", "Q", "C"], ["T", "D"]),
}


class CachedReassemblySegmenter(Segmenter):
    """Replays terse-v2 segments, re-assembling fields per a named variant."""

    def __init__(self, *, cache_path: str | Path, variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r}; choices: {sorted(VARIANTS)}"
            )
        spec = VARIANTS[variant]
        self._block_recipe, self._embed_recipe, self._bm25_recipe = spec[:3]
        self._embed_sep: str = spec[3] if len(spec) > 3 else "\n"
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
        # Lowercase keys = label-stripped: no "Queries:" / "Dates:" /
        # producer header. Isolates whether the header tokens help.
        if key == "q":
            q = rec["queries"].strip()
            return q or None
        if key == "c":
            c = rec["chunk"].strip()
            return c or None
        if key == "d":
            d = rec["dates"].strip()
            return d or None
        # 4th-framing keys (require the augmented cache). A = atomic-fact
        # decomposition (granularity axis); P = topic/theme labels
        # (abstraction axis). See augment_cache_framings.py.
        if key == "A":
            atomic = rec.get("atomic") or []
            joined = " ".join(a.strip() for a in atomic if a and a.strip())
            return joined or None
        if key == "P":
            topic = (rec.get("topic") or "").strip()
            return f"Topics: {topic}" if topic else None
        raise ValueError(f"unknown component {key!r}")

    def _assemble(
        self, recipe: list[str], rec: dict, sep: str = "\n"
    ) -> str:
        parts = [self._component(k, rec) for k in recipe]
        return sep.join(p for p in parts if p)

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
            embed_text = self._assemble(
                self._embed_recipe, rec, self._embed_sep
            )
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
