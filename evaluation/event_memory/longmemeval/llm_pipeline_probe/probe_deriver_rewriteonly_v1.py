"""Rewrite-only deriver v1.

Subtractive deriver: emits ONE derivative per segment whose embedding
text is the segment's formal third-person rewrite ONLY, with the raw
conversation chunk and speaker prefix DROPPED.

Hypothesis
----------

The v22 segmenter's RewriteContext stores
``text_to_embed = "{rewrite}\\n{speaker}: {original_chunk}"`` and the
default whole-text deriver embeds that dual-text directly. The raw
chunk carries conversational filler ("yeah", "I mean", greetings,
hedges, vocatives) that is not part of the fact being retrieved.
Concatenating it with the precise rewrite may DILUTE the embedding,
pulling the resulting vector toward generic-conversation regions of
embedding space and away from the rewrite's specifics. Embedding the
rewrite ALONE should produce a tighter, more specifics-anchored
vector that better matches specifics-anchored queries.

Risk
----

The dual-text embed also gives the segment a lexical foothold on the
raw query phrasing — questions that echo a user's own words ("what
did I say about ...", queries that quote a phrase verbatim) currently
benefit from the chunk's presence in the embedded text. Dropping the
chunk loses that lexical-match channel: the rewrite is a paraphrase
and may not surface for queries that match the original wording but
not the rewrite's vocabulary. This deriver tests whether the
specifics-tightening gain outweighs the lexical-match loss.

Generalizability
----------------

Subtractive: no new prompt, no LLM call, no new vocabulary. The
hypothesis is corpus-agnostic — it claims a property of the embedder
(concatenated heterogeneous text dilutes) that should hold across
any domain where segmenter rewrites are well-formed. If rewrite-only
wins on LongMemEval it should generalize to any pipeline whose
segmenter emits faithful third-person rewrites; if it loses, the
chunk's lexical channel is load-bearing and dual-text stays.

Output
------

One derivative per segment, ``block.text = segment.block.text`` (the
rewrite itself, as stored by RewriteSegmenter), with NullContext so
the format helper does NOT re-wrap into the dual-text form. Shares
segment_uuid with the source segment.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from memmachine_server.episodic_memory.event_memory.data_types import (
    Derivative,
    NullContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import (
    Deriver,
)


class GenericDeriver(Deriver):
    """Emits ONE rewrite-only derivative per segment.

    No prompt, no LLM call. ``block.text`` is the segment's rewrite
    (``segment.block.text``) with NullContext, bypassing the
    RewriteContext dual-text wrapping that the default whole-text
    deriver would apply.
    """

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        match segment.block:
            case TextBlock(text=text):
                pass
            case _:
                raise NotImplementedError(
                    f"Unsupported block type: {type(segment.block).__name__}"
                )

        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=NullContext(),
                block=TextBlock(text=text),
                properties=segment.properties,
            )
        ]
