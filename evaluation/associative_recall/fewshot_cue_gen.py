"""Few-shot cue generation: v2f prompt + in-context exemplars.

Variants:
  FewshotV2fK2         — 2 exemplars
  FewshotV2fK3         — 3 exemplars
  FewshotV2fCategoryK2 — 2 exemplars, filtered to same q-category as query

All variants leave-one-out: exclude exemplars from the same conversation_id.
Uses v2f's accumulated context section. Base prompt is V2f (gpt-5-mini).
"""

import json
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

from associative_recall import Segment, SegmentStore
from best_shot import (
    BestshotBase,
    BestshotResult,
    _format_segments,
    _parse_cues,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BANK_PATH = RESULTS_DIR / "fewshot_exemplar_bank.json"


# Few-shot prompt template
FEWSHOT_V2F_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Here are examples of cues that worked for similar questions:

{exemplar_block}

Now generate cues for this new question. Use specific vocabulary that a real \
conversation would contain — rephrase, don't copy exemplars verbatim, but \
match their style.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate exactly 2 search cues. Use specific vocabulary that would \
appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


def load_exemplar_bank() -> list[dict]:
    """Load exemplar bank with pre-computed embeddings."""
    if not BANK_PATH.exists():
        raise FileNotFoundError(
            f"Exemplar bank missing: {BANK_PATH}. "
            "Run build_exemplar_bank.py first."
        )
    with open(BANK_PATH) as f:
        data = json.load(f)
    exemplars = data["exemplars"]
    # Convert embeddings to numpy for fast cosine
    for ex in exemplars:
        ex["_embedding"] = np.array(
            ex["question_embedding"], dtype=np.float32
        )
        n = np.linalg.norm(ex["_embedding"])
        ex["_embedding_norm"] = ex["_embedding"] / max(n, 1e-10)
    return exemplars


def select_exemplars(
    query_emb: np.ndarray,
    exemplars: list[dict],
    k: int,
    exclude_conv_id: str,
    category_filter: str | None = None,
) -> list[dict]:
    """Select top-K nearest exemplars by cosine similarity.

    - Excludes any exemplar from `exclude_conv_id` (leave-one-out leakage guard).
    - If `category_filter` is set, only returns exemplars matching that
      category; falls back to unfiltered if fewer than k match.
    """
    candidates = [e for e in exemplars if e["conversation_id"] != exclude_conv_id]

    if category_filter is not None:
        cat_candidates = [e for e in candidates if e["category"] == category_filter]
        if len(cat_candidates) >= k:
            candidates = cat_candidates
        # else fall back to full list

    if not candidates:
        return []

    q_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
    sims = np.array([float(q_norm @ c["_embedding_norm"]) for c in candidates])
    top_idx = np.argsort(sims)[::-1][:k]
    selected = [candidates[i] for i in top_idx]
    # Attach similarity for logging
    for c, i in zip(selected, top_idx):
        c["_sim"] = float(sims[i])
    return selected


def build_exemplar_block(exemplars: list[dict]) -> str:
    """Build the exemplar section of the few-shot prompt."""
    if not exemplars:
        return "(No exemplars available — generate cues directly.)"
    blocks = []
    for i, ex in enumerate(exemplars, start=1):
        cue_lines = "\n".join(f"CUE: {c}" for c in ex["cues"])
        blocks.append(
            f"---\nExample {i}\n"
            f"Question: {ex['question']}\n"
            f"Cues that worked:\n{cue_lines}\n---"
        )
    return "\n".join(blocks)


class FewshotV2fBase(BestshotBase):
    """Base few-shot v2f: v2f prompt + in-context exemplars.

    Uses pre-built exemplar bank. For each query:
      1. Embed question
      2. Retrieve top-K nearest exemplar questions (leave-one-out on conv_id)
      3. Inject into prompt
      4. Parse CUE: lines and retrieve
    """

    exemplar_k: int = 2
    category_match: bool = False
    arch_name: str = "fewshot_v2f"

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        exemplars: list[dict] | None = None,
    ):
        super().__init__(store, client)
        self.exemplars = exemplars if exemplars is not None else load_exemplar_bank()

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(all_segments)
        )

        # Select exemplars via leave-one-out on conversation_id
        # For category match: use question's category if available.
        # Since we don't know the query's category at retrieve-time, category
        # matching is handled via subclass override (passes category in).
        selected = self._select_for_query(query_emb, conversation_id)
        exemplar_block = build_exemplar_block(selected)

        prompt = FEWSHOT_V2F_PROMPT.format(
            exemplar_block=exemplar_block,
            question=question,
            context_section=context_section,
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": self.arch_name,
                "output": output,
                "cues": cues[:2],
                "exemplars_used": [
                    {
                        "question": e["question"],
                        "category": e["category"],
                        "dataset": e["dataset"],
                        "conversation_id": e["conversation_id"],
                        "sim": e.get("_sim", 0.0),
                    }
                    for e in selected
                ],
            },
        )

    def _select_for_query(
        self, query_emb: np.ndarray, conversation_id: str
    ) -> list[dict]:
        return select_exemplars(
            query_emb,
            self.exemplars,
            k=self.exemplar_k,
            exclude_conv_id=conversation_id,
            category_filter=None,
        )


class FewshotV2fK2(FewshotV2fBase):
    exemplar_k = 2
    arch_name = "fewshot_v2f_k2"


class FewshotV2fK3(FewshotV2fBase):
    exemplar_k = 3
    arch_name = "fewshot_v2f_k3"


class FewshotV2fCategoryK2(FewshotV2fBase):
    """Exemplars filtered to same q-category as query (k=2).

    Category is passed via a thread-local-like instance attribute that the
    eval loop sets before calling retrieve(). For this reason this class uses
    `retrieve_with_category` when category is known, else falls back to
    regular cosine selection.
    """
    exemplar_k = 2
    category_match = True
    arch_name = "fewshot_v2f_category_k2"

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        exemplars: list[dict] | None = None,
    ):
        super().__init__(store, client, exemplars)
        self._current_category: str | None = None

    def set_category(self, category: str | None) -> None:
        self._current_category = category

    def _select_for_query(
        self, query_emb: np.ndarray, conversation_id: str
    ) -> list[dict]:
        return select_exemplars(
            query_emb,
            self.exemplars,
            k=self.exemplar_k,
            exclude_conv_id=conversation_id,
            category_filter=self._current_category,
        )
