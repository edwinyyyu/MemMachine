"""DFS stack-based context tree for reasoning-integrated retrieval (v2).

Key design:
  * The task is to PRODUCE TEXT OUTPUT (draft, plan, prepare). This grounds
    retrieval — "can I write the next paragraph?" is a question the model
    can answer, unlike "is this search complete?" (the pure-retrieval failure
    mode that scored -23.2pp).
  * Single execution thread. Children execute one at a time (DFS).
  * Three actions at any node:
      PUSH    -> create a child with a sub-task description
      RETRIEVE -> search memory with the current node's content as query
      POP     -> summarize and return to parent
  * Forced retrieve at leaf: if a leaf node has no retrieved segments yet,
    auto-retrieve before the next decision. Prevents "infinite push without
    retrieve" failure.
  * Leaf-level v15-style retrieval: multi-cue self-monitoring at leaves
    (the proven 2-cue technique).
  * Strict budget K: total retrieved <= K, shared across all leaves.
    If the tree explores L leaves, each gets floor(K/L). No reranking.

Usage:
    uv run python context_tree_v2.py --budget 20
    uv run python context_tree_v2.py --budget 50
    uv run python context_tree_v2.py --budget 20 --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    DATA_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
LLM_CACHE_FILE = CACHE_DIR / "tree_v2_llm_cache.json"

MAX_DEPTH = 3
MAX_ACTIONS = 10


# ---------------------------------------------------------------------------
# Combined store (synthetic + advanced conversations in one SegmentStore)
# ---------------------------------------------------------------------------
def build_combined_store() -> SegmentStore:
    """Load segments_synthetic.npz + segments_advanced.npz into one store.

    We do this by merging the npz files in memory and writing a single
    combined file to cache (so SegmentStore loads it cleanly).
    """
    combined_path = DATA_DIR / "segments_v2_combined.npz"
    if not combined_path.exists():
        syn = np.load(DATA_DIR / "segments_synthetic.npz", allow_pickle=True)
        adv = np.load(DATA_DIR / "segments_advanced.npz", allow_pickle=True)
        merged = {}
        for k in syn.files:
            merged[k] = np.concatenate([syn[k], adv[k]], axis=0)
        np.savez(combined_path, **merged)
    return SegmentStore(data_dir=DATA_DIR, npz_name="segments_v2_combined.npz")


# ---------------------------------------------------------------------------
# LLM / embedding cache (tree_v2 specific file, re-uses shared embeddings)
# ---------------------------------------------------------------------------
class TreeV2LLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = LLM_CACHE_FILE
        self._cache: dict[str, str] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache = json.load(f)
        self._dirty = False

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        tmp.replace(self.cache_file)
        self._dirty = False


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ACTION_DECISION_PROMPT = """\
You are executing a task by walking a reasoning tree. At each node you choose \
ONE action. The goal is to produce the final TEXT OUTPUT for the root task \
using information stored in a conversation-history memory.

ROOT TASK: {root_task}

CURRENT SUB-TASK PATH (root -> current):
{depth_path}

ALREADY-COMPLETED SIBLING SUB-TASKS:
{sibling_summaries}

{existing_children}

{grounding_block}

DEPTH: {depth} / MAX_DEPTH: {max_depth}
ACTIONS REMAINING: {actions_remaining}

CHOOSE ONE ACTION:

- PUSH: decompose this sub-task by adding a child that names ONE concrete \
sub-topic the conversation history likely discussed. Use PUSH when the \
current description still covers multiple distinct memory-topics.
- RETRIEVE: run one semantic search using this sub-task's description as the \
query. Use RETRIEVE when the description names a SINGLE concrete topic that \
one embedding search would find.
- POP: conclude this branch and return to the parent. Write a one-sentence \
summary of what this sub-task was.

HOW TO DECOMPOSE (critical):
Decompose by SUBJECT-MATTER TOPIC the memory likely contains, NOT by output \
structure. Each child description should be something a specific conversation \
message would plausibly BE ABOUT — naming real subjects (people, products, \
events, constraints, decisions, dates).

EXAMPLES of GOOD PUSH children:
  root task: "cook dinner for Bob — what to keep in mind"
    -> "Bob's food allergies and intolerances"
    -> "Bob's diet preferences and recent meals"
    -> "cooking equipment and ingredients currently available"

  root task: "draft status update for Acme Corp rebrand team"
    -> "design team's logo concepts and color palette decisions"
    -> "engineering migration from WordPress to Strapi/Next.js"
    -> "brand voice and copy guidelines from Hiroshi"
    -> "upcoming deadlines, meetings, and client commitments"

  root task: "topics to discuss with Dr. Patel at January 25th appointment"
    -> "current knee pain, injury history, and ibuprofen use"
    -> "sleep issues and potential melatonin/chamomile remedies"
    -> "metformin dosage change and ongoing medication interactions"
    -> "cardiac stress test, family heart-history, and PSA results"

EXAMPLES of BAD PUSH children (DO NOT produce these):
  "write the Completed Deliverables section"        (output structure)
  "compile a prioritized list of symptoms"          (output structure)
  "ask the user to list their allergies"            (meta-action, not memory)
  "determine the hardware required"                  (too vague)
  "for each area, draft a summary"                   (loop over output)
  "Bob's dietary restrictions" x 4 near-copies       (no new decomposition)

Children must be DIFFERENT topics from each other. Target 2-4 distinct \
subject-matter siblings when decomposing — not a long chain of reworded \
copies.

DEPTH GUIDANCE:
- At depth 0 with a broad multi-topic root task: strongly prefer PUSH, and \
target 3-4 DISTINCT children (one PUSH per decision). Do NOT POP the root \
until you have pushed enough distinct topics to cover the task.
- At depth 1-2: prefer RETRIEVE if your current description names one topic.
- At MAX_DEPTH: you cannot PUSH — choose RETRIEVE or POP.

BUDGET GUIDANCE:
- The total retrieval budget is shared across all leaves. If you create N \
leaves, each gets roughly K/N segments where K is the global budget (20-50).
- Aim for 3 top-level children for broad multi-topic tasks (not 1, not 5+). \
Only push children that target CLEARLY DISTINCT memory topics.
- When you have 3 solid distinct root-level children covering the main \
topics, POP the root.

Format exactly:
ACTION: <PUSH|RETRIEVE|POP>
ARGUMENT: <if PUSH: a SHORT (<=14 words) subject-matter-specific phrase naming \
ONE topic a conversation turn could plausibly be ABOUT; \
if POP: a one-sentence summary; \
if RETRIEVE: leave empty>
Nothing else."""


LEAF_CUE_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

ROOT TASK: {root_task}
CURRENT SUB-TASK: {sub_task}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{retrieved_context}

First, briefly assess: Given what's been retrieved for THIS sub-task so far, \
how well is the search going? What kind of content is still missing? Should \
you search for similar content or pivot to an adjacent topic?

Then generate {num_cues} search cues based on your assessment. Each cue \
should use specific vocabulary that would appear in the target conversation \
turns — concrete nouns, names, actions, numbers — not abstract summaries.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TreeNode:
    description: str
    depth: int
    parent: "TreeNode | None" = None
    children: list["TreeNode"] = field(default_factory=list)
    retrieved_segments: list[Segment] = field(default_factory=list)
    summary: str = ""
    popped: bool = False
    is_retrieve_leaf: bool = False
    grounding_segments: list[Segment] = field(default_factory=list)


def _subtree_has_retrieve(node: "TreeNode") -> bool:
    if node.is_retrieve_leaf:
        return True
    return any(_subtree_has_retrieve(c) for c in node.children)


@dataclass
class ActionLogEntry:
    step: int
    depth: int
    description: str
    action: str
    argument: str
    num_retrieved: int
    cues: list[str] = field(default_factory=list)


@dataclass
class TreeRunResult:
    root_task: str
    conversation_id: str
    all_segments: list[Segment]           # in insertion order
    leaf_segments_by_leaf: list[list[Segment]]
    leaf_descriptions: list[str]
    leaf_queries: list[str]               # what was actually embedded at leaf
    action_log: list[ActionLogEntry]
    num_leaves: int
    num_retrieve_actions: int
    num_push_actions: int
    num_pop_actions: int
    budget_per_leaf: int
    embed_calls: int
    llm_calls: int
    elapsed: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ContextTreeV2:
    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        model: str = MODEL,
        budget: int = 20,
        max_depth: int = MAX_DEPTH,
        max_actions: int = MAX_ACTIONS,
        leaf_num_cues: int = 2,
    ) -> None:
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.model = model
        self.budget = budget
        self.max_depth = max_depth
        self.max_actions = max_actions
        self.leaf_num_cues = leaf_num_cues
        self.embed_cache = EmbeddingCache()
        self.llm_cache = TreeV2LLMCache()

        self.embed_calls = 0
        self.llm_calls = 0

    # ------------------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        cached = self.embed_cache.get(text)
        if cached is not None:
            return cached
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        emb = np.array(resp.data[0].embedding, dtype=np.float32)
        self.embed_cache.put(text, emb)
        self.embed_calls += 1
        return emb

    def llm(self, prompt: str, max_tokens: int = 800) -> str:
        cached = self.llm_cache.get(self.model, prompt)
        if cached is not None:
            return cached
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        self.llm_cache.put(self.model, prompt, text)
        self.llm_calls += 1
        return text

    # ------------------------------------------------------------------
    def _format_segments(self, segments: list[Segment], limit: int = 10) -> str:
        if not segments:
            return "(no segments retrieved yet for this sub-task)"
        sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:limit]
        return "\n".join(
            f"[Turn {s.turn_id}, {s.role}]: {s.text[:220]}" for s in sorted_segs
        )

    def _depth_path_str(self, node: TreeNode) -> str:
        path: list[str] = []
        cur: TreeNode | None = node
        while cur is not None:
            path.append(cur.description)
            cur = cur.parent
        path.reverse()
        return " -> ".join(path)

    def _sibling_summary_str(self, node: TreeNode) -> str:
        if node.parent is None:
            return "(none — this is the root)"
        sibs = [c for c in node.parent.children if c is not node and c.popped]
        if not sibs:
            return "(no completed siblings yet)"
        return "\n".join(f"- {c.description}: {c.summary}" for c in sibs)

    # ------------------------------------------------------------------
    def _leaf_retrieve(
        self,
        root_task: str,
        node: TreeNode,
        per_leaf_budget: int,
        exclude: set[int],
        conversation_id: str,
    ) -> tuple[list[Segment], list[str], str]:
        """Leaf-level v15-style retrieval.

        Returns (segments, cues_used, query_text). The query_text is the
        node description — used as the HOP-0 query (embedding) with no
        explicit query parameter, as specified.
        """
        if per_leaf_budget <= 0:
            return [], [], node.description

        # Hop 0: embed the node description directly (as per spec:
        # "use the current frame's content as query, not an explicit query")
        query_text = node.description
        q_emb = self.embed(query_text)
        hop0 = self.store.search(
            q_emb,
            top_k=per_leaf_budget,
            conversation_id=conversation_id,
            exclude_indices=exclude,
        )
        segs: list[Segment] = []
        for s in hop0.segments:
            if s.index not in exclude:
                segs.append(s)
                exclude.add(s.index)
            if len(segs) >= per_leaf_budget:
                break

        if len(segs) >= per_leaf_budget:
            return segs, [query_text], query_text

        # Hop 1: v15-style 2-cue self-monitoring, adapted to the sub-task
        remaining = per_leaf_budget - len(segs)
        per_cue = max(1, (remaining + self.leaf_num_cues - 1) // self.leaf_num_cues)

        # Include already-retrieved segments from prior retrievals at this
        # leaf (for top-up passes) in the cue-generation context.
        ctx_segs = list(node.retrieved_segments) + segs
        prompt = LEAF_CUE_PROMPT.format(
            root_task=root_task,
            sub_task=node.description,
            retrieved_context=self._format_segments(ctx_segs, limit=12),
            num_cues=self.leaf_num_cues,
        )
        response = self.llm(prompt, max_tokens=700)
        cues = _parse_cues(response)[: self.leaf_num_cues]

        for cue in cues:
            if len(segs) >= per_leaf_budget:
                break
            cue_emb = self.embed(cue)
            result = self.store.search(
                cue_emb,
                top_k=per_cue,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    segs.append(s)
                    exclude.add(s.index)
                if len(segs) >= per_leaf_budget:
                    break

        return segs, [query_text] + cues, query_text

    # ------------------------------------------------------------------
    def _decide_action(
        self,
        root_task: str,
        node: TreeNode,
        actions_remaining: int,
    ) -> tuple[str, str]:
        if node.children:
            existing = "\n".join(
                f"- {c.description}" for c in node.children
            )
            existing_block = (
                "CHILDREN ALREADY PUSHED AT THIS NODE (do NOT repeat or "
                "paraphrase these; pick a DIFFERENT subject-matter topic, "
                "or RETRIEVE/POP):\n" + existing
            )
        else:
            existing_block = "CHILDREN ALREADY PUSHED AT THIS NODE: (none)"

        # Find grounding segments from the nearest ancestor (usually root)
        cur: TreeNode | None = node
        grounding: list[Segment] = []
        while cur is not None:
            if cur.grounding_segments:
                grounding = cur.grounding_segments
                break
            cur = cur.parent

        if grounding:
            grounding_block = (
                "GROUNDING — brief scan of the memory (use this to guide "
                "decomposition toward topics actually discussed):\n" +
                self._format_segments(grounding, limit=6)
            )
        else:
            grounding_block = "GROUNDING: (none)"

        prompt = ACTION_DECISION_PROMPT.format(
            root_task=root_task,
            depth_path=self._depth_path_str(node),
            sibling_summaries=self._sibling_summary_str(node),
            existing_children=existing_block,
            grounding_block=grounding_block,
            depth=node.depth,
            max_depth=self.max_depth,
            actions_remaining=actions_remaining,
        )
        response = self.llm(prompt, max_tokens=1200)
        return _parse_action(response)

    # ------------------------------------------------------------------
    def run(
        self,
        root_task: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeRunResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()

        root = TreeNode(description=root_task, depth=0, parent=None)
        stack: list[TreeNode] = [root]
        leaves: list[TreeNode] = []  # nodes marked as leaves (RETRIEVE chosen)
        # Note: retrieval is deferred to a second pass after tree exploration
        # completes. The first pass only does PUSH/POP and marks leaves.
        action_log: list[ActionLogEntry] = []

        step = 0

        # ---- Phase 0: grounding retrieval at root ----
        # Fetch a few segments by embedding the root task directly. These
        # ground the model's decomposition by showing what the memory
        # actually contains. The segments are added to `all_segments` (so
        # they count against budget) and stored on the root node (so the
        # decision prompt sees them as its "sibling summaries" equivalent).
        all_segments: list[Segment] = []
        exclude: set[int] = set()
        leaf_queries: list[str] = []

        # Grounding budget: ~10% of K, min 2, max 4. At K=20 -> 2 segs; at
        # K=50 -> 4 segs. This balances "know what memory discusses" with
        # "don't spend too much budget on scouting".
        GROUNDING_SIZE = max(2, min(4, self.budget // 10))
        if GROUNDING_SIZE > 0:
            ground_emb = self.embed(root_task)
            ground_result = self.store.search(
                ground_emb, top_k=GROUNDING_SIZE,
                conversation_id=conversation_id,
            )
            grounding_segs = list(ground_result.segments)[:GROUNDING_SIZE]
            all_segments.extend(grounding_segs)
            exclude.update(s.index for s in grounding_segs)
            root.grounding_segments = grounding_segs
            leaf_queries.append(root_task)  # tracks that we embedded the question
            action_log.append(ActionLogEntry(
                step=-1, depth=0,
                description=root_task[:120],
                action="GROUNDING",
                argument=root_task[:120],
                num_retrieved=len(grounding_segs),
                cues=[root_task],
            ))

        MAX_EXPECTED_LEAVES = 5  # used for minimum per-leaf allocation
        MIN_PER_LEAF = max(2, self.budget // MAX_EXPECTED_LEAVES)

        def remaining() -> int:
            return max(0, self.budget - len(all_segments))

        while stack and step < self.max_actions and remaining() > 0:
            node = stack[-1]
            if node.popped:
                stack.pop()
                continue

            actions_remaining = self.max_actions - step
            action, argument = self._decide_action(
                root_task, node, actions_remaining,
            )

            # At MAX_DEPTH, PUSH is not allowed.
            if action == "PUSH" and node.depth >= self.max_depth:
                action = "RETRIEVE"

            # Forced retrieve: if model POPs on a leaf with no children AND
            # no retrieve marked, override to RETRIEVE. Prevents the
            # "infinite push → POP without retrieving" failure.
            all_kids_done = bool(node.children) and all(c.popped for c in node.children)
            is_leaf_no_retrieve = (not node.children) and not node.is_retrieve_leaf
            if action == "POP" and is_leaf_no_retrieve:
                action = "RETRIEVE"
                argument = ""

            # If all children popped and none of them marked as retrieve-leaf,
            # and this node isn't a retrieve-leaf either -> force RETRIEVE
            # here before POP.
            if (action == "POP" and all_kids_done
                    and not node.is_retrieve_leaf
                    and not any(_subtree_has_retrieve(c) for c in node.children)):
                action = "RETRIEVE"
                argument = ""

            if verbose:
                indent = "  " * node.depth
                print(f"{indent}[d{node.depth}] action={action} "
                      f"arg='{argument[:60]}' node='{node.description[:40]}'")

            if action == "PUSH":
                if not argument:
                    action = "RETRIEVE"
                elif len(node.children) >= 4:
                    # Hard cap on siblings per parent (budget dilution).
                    action_log.append(ActionLogEntry(
                        step=step, depth=node.depth,
                        description=node.description[:120],
                        action="PUSH_REJECTED_CAP",
                        argument=argument.strip()[:120],
                        num_retrieved=0,
                    ))
                    step += 1
                    # Force POP on parent (enough siblings).
                    node.summary = "(sibling cap reached)"
                    node.popped = True
                    stack.pop()
                    continue
                elif _too_similar(argument.strip(), node.description):
                    action_log.append(ActionLogEntry(
                        step=step, depth=node.depth,
                        description=node.description[:120],
                        action="PUSH_REJECTED_DUPLICATE_PARENT",
                        argument=argument.strip()[:120],
                        num_retrieved=0,
                    ))
                    step += 1
                    action = "RETRIEVE"
                elif any(_too_similar(argument.strip(), c.description, threshold=0.7)
                         for c in node.children):
                    # Sibling duplicate — ignore this child silently.
                    # Let the parent try again or POP on its own.
                    action_log.append(ActionLogEntry(
                        step=step, depth=node.depth,
                        description=node.description[:120],
                        action="PUSH_REJECTED_DUPLICATE_SIBLING",
                        argument=argument.strip()[:120],
                        num_retrieved=0,
                    ))
                    step += 1
                    # If this is the Nth sibling dup in a row, force POP.
                    # Track by counting recent PUSH_REJECTED_DUPLICATE_SIBLING
                    # entries at the same depth.
                    recent_rejects = 0
                    for e in reversed(action_log):
                        if (e.action == "PUSH_REJECTED_DUPLICATE_SIBLING"
                                and e.depth == node.depth):
                            recent_rejects += 1
                        elif e.action in {"PUSH", "RETRIEVE_MARK", "POP"}:
                            break
                    if recent_rejects >= 2:
                        # Give up on this parent; pop it.
                        node.summary = "(too many sibling dups)"
                        node.popped = True
                        action_log.append(ActionLogEntry(
                            step=step, depth=node.depth,
                            description=node.description[:120],
                            action="POP_FORCED_SIBDUPS",
                            argument=node.summary,
                            num_retrieved=0,
                        ))
                        step += 1
                        stack.pop()
                    continue
                else:
                    child = TreeNode(
                        description=argument.strip(),
                        depth=node.depth + 1,
                        parent=node,
                    )
                    node.children.append(child)
                    stack.append(child)
                    action_log.append(ActionLogEntry(
                        step=step, depth=node.depth,
                        description=node.description[:120],
                        action="PUSH", argument=argument[:120],
                        num_retrieved=0,
                    ))
                    step += 1
                    continue

            if action == "RETRIEVE":
                if node.is_retrieve_leaf:
                    node.summary = "(already retrieved; auto-pop)"
                    node.popped = True
                    action_log.append(ActionLogEntry(
                        step=step, depth=node.depth,
                        description=node.description[:120],
                        action="POP_DUPMARK", argument=node.summary,
                        num_retrieved=len(node.retrieved_segments),
                    ))
                    step += 1
                    stack.pop()
                    continue

                # Interleaved retrieval: allocate MIN_PER_LEAF from remaining
                # budget. Keep some reserve for other leaves.
                room = remaining()
                per_leaf = min(MIN_PER_LEAF, room)
                if per_leaf <= 0:
                    node.summary = "(no budget left)"
                    node.popped = True
                    action_log.append(ActionLogEntry(
                        step=step, depth=node.depth,
                        description=node.description[:120],
                        action="POP_NOBUDGET", argument=node.summary,
                        num_retrieved=0,
                    ))
                    step += 1
                    stack.pop()
                    continue

                segs, cues, query_text = self._leaf_retrieve(
                    root_task, node, per_leaf, exclude, conversation_id,
                )
                segs = segs[:remaining()]
                node.retrieved_segments.extend(segs)
                all_segments.extend(segs)
                node.is_retrieve_leaf = True
                leaves.append(node)
                leaf_queries.append(query_text)

                # Build a summary from retrieved content (short keywords from
                # the top segments).
                summary_parts = []
                for s in segs[:3]:
                    text_trim = s.text[:120].replace("\n", " ")
                    summary_parts.append(f"turn {s.turn_id}: {text_trim}")
                node.summary = (
                    f"retrieved {len(segs)} segments. " +
                    " | ".join(summary_parts)
                )

                action_log.append(ActionLogEntry(
                    step=step, depth=node.depth,
                    description=node.description[:120],
                    action="RETRIEVE",
                    argument=query_text[:120],
                    num_retrieved=len(segs),
                    cues=cues,
                ))
                step += 1
                node.popped = True
                stack.pop()
                continue

            # POP
            node.summary = argument or "(no summary)"
            node.popped = True
            action_log.append(ActionLogEntry(
                step=step, depth=node.depth,
                description=node.description[:120],
                action="POP", argument=argument[:120],
                num_retrieved=0,
            ))
            step += 1
            stack.pop()

        # Fallback: if no leaves retrieved (pathological), do one retrieve
        # on the root.
        if not leaves and remaining() > 0:
            segs, cues, query_text = self._leaf_retrieve(
                root_task, root, remaining(), exclude, conversation_id,
            )
            root.retrieved_segments.extend(segs)
            all_segments.extend(segs)
            root.is_retrieve_leaf = True
            leaves.append(root)
            leaf_queries.append(query_text)
            action_log.append(ActionLogEntry(
                step=step, depth=0,
                description=root.description[:120],
                action="RETRIEVE_FALLBACK_ROOT",
                argument=query_text[:120],
                num_retrieved=len(segs),
                cues=cues,
            ))

        # ---- Phase 2: top-up unused budget across leaves ----
        # If we reserved MIN_PER_LEAF but ended up with only 2-3 leaves,
        # distribute remaining budget evenly among them (via more cue-driven
        # retrieval at each leaf).
        if remaining() > 0 and leaves:
            per_top = remaining() // len(leaves)
            extra_leftover = remaining() - per_top * len(leaves)
            for i, leaf in enumerate(leaves):
                if remaining() <= 0:
                    break
                top_share = per_top + (1 if i < extra_leftover else 0)
                if top_share <= 0:
                    continue
                room = remaining()
                top_share = min(top_share, room)
                extra, cues_extra, _ = self._leaf_retrieve(
                    root_task, leaf, top_share, exclude, conversation_id,
                )
                extra = extra[:remaining()]
                if extra:
                    leaf.retrieved_segments.extend(extra)
                    all_segments.extend(extra)

        # Safety truncate
        if len(all_segments) > self.budget:
            all_segments = all_segments[: self.budget]

        elapsed = time.time() - t0

        leaf_segments_by_leaf = [list(l.retrieved_segments) for l in leaves]
        leaf_descriptions = [l.description for l in leaves]

        return TreeRunResult(
            root_task=root_task,
            conversation_id=conversation_id,
            all_segments=all_segments,
            leaf_segments_by_leaf=leaf_segments_by_leaf,
            leaf_descriptions=leaf_descriptions,
            leaf_queries=leaf_queries,
            action_log=action_log,
            num_leaves=len(leaves),
            num_retrieve_actions=sum(1 for e in action_log if e.action.startswith("RETRIEVE")),
            num_push_actions=sum(1 for e in action_log if e.action == "PUSH"),
            num_pop_actions=sum(1 for e in action_log if e.action.startswith("POP")),
            budget_per_leaf=(self.budget // max(1, len(leaves))),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            elapsed=elapsed,
        )

    def save_caches(self) -> None:
        try:
            self.embed_cache.save()
        except OSError:
            pass
        self.llm_cache.save()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
_ACTION_RE = re.compile(r"^\s*ACTION\s*:\s*(PUSH|RETRIEVE|POP)\b", re.IGNORECASE)
_ARG_RE = re.compile(r"^\s*ARGUMENT\s*:\s*(.*)$", re.IGNORECASE)


def _parse_action(text: str) -> tuple[str, str]:
    action = ""
    argument = ""
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        m = _ACTION_RE.match(line)
        if m and not action:
            action = m.group(1).upper()
        m2 = _ARG_RE.match(line)
        if m2 and not argument:
            # Argument can span multiple lines until blank or next key.
            arg_lines = [m2.group(1).strip()]
            for j in range(i + 1, len(lines)):
                ln = lines[j].strip()
                if not ln or _ACTION_RE.match(ln) or _ARG_RE.match(ln):
                    break
                arg_lines.append(ln)
            argument = " ".join(a for a in arg_lines if a).strip()
            break

    if not action:
        # Fallback: if the model returned something like "PUSH: ..."
        upper = text.strip().upper()
        if upper.startswith("PUSH"):
            action = "PUSH"
        elif upper.startswith("RETRIEVE"):
            action = "RETRIEVE"
        elif upper.startswith("POP"):
            action = "POP"
        else:
            action = "POP"

    return action, argument


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2}


def _too_similar(a: str, b: str, threshold: float = 0.82) -> bool:
    """Jaccard-similarity check on tokens. True if the two descriptions are
    essentially the same thing."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return False
    jac = len(ta & tb) / len(ta | tb)
    return jac >= threshold


def _parse_cues(response: str) -> list[str]:
    cues = []
    for line in response.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


# ---------------------------------------------------------------------------
# Cosine baseline (no reranking, at same K)
# ---------------------------------------------------------------------------
def cosine_baseline(
    store: SegmentStore,
    embed_fn,
    question: str,
    conversation_id: str,
    k: int,
) -> list[Segment]:
    q_emb = embed_fn(question)
    result = store.search(q_emb, top_k=k, conversation_id=conversation_id)
    return result.segments[:k]


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------
def recall_at_k(retrieved: list[Segment], source_ids: set[int], k: int) -> float:
    if not source_ids:
        return 1.0
    top = {s.turn_id for s in retrieved[:k]}
    return len(top & source_ids) / len(source_ids)


def avg_leaf_similarity(
    embed_fn,
    question: str,
    leaf_queries: list[str],
) -> float | None:
    """Mean cosine similarity of leaf queries (used at retrieve) vs the
    original question embedding. Lower = decomposition pushed us away from
    the original phrasing (intended behavior).
    """
    if not leaf_queries:
        return None
    q_emb = embed_fn(question)
    q_norm = q_emb / max(np.linalg.norm(q_emb), 1e-10)
    sims = []
    for lq in leaf_queries:
        lq_emb = embed_fn(lq)
        lq_norm = lq_emb / max(np.linalg.norm(lq_emb), 1e-10)
        sims.append(float(np.dot(q_norm, lq_norm)))
    return sum(sims) / len(sims)


def evaluate(
    engine: ContextTreeV2,
    questions: list[dict],
    budgets: tuple[int, ...],
    verbose: bool = False,
) -> list[dict]:
    results = []
    for i, q in enumerate(questions):
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])

        # Tree retrieval at the PRIMARY budget (engine.budget)
        tree_result = engine.run(q_text, conv_id, verbose=verbose)
        tree_segs = tree_result.all_segments

        # Cosine baseline at the PRIMARY budget
        baseline_segs_primary = cosine_baseline(
            engine.store, engine.embed, q_text, conv_id, engine.budget,
        )

        # Recall at all budgets (but tree is capped at primary budget;
        # for larger budgets we pad by treating r@k for k>budget as r@budget)
        recalls_tree: dict[str, float] = {}
        recalls_base: dict[str, float] = {}
        for b in budgets:
            k_tree = min(b, engine.budget)
            recalls_tree[f"r@{b}"] = recall_at_k(tree_segs, source_ids, k_tree)
            recalls_base[f"r@{b}"] = recall_at_k(
                cosine_baseline(
                    engine.store, engine.embed, q_text, conv_id, b,
                ),
                source_ids,
                b,
            )

        avg_sim = avg_leaf_similarity(
            engine.embed, q_text, tree_result.leaf_queries,
        )

        row = {
            "conversation_id": conv_id,
            "category": q["category"],
            "question_index": q.get("question_index"),
            "question": q_text,
            "source_chat_ids": sorted(source_ids),
            "num_source_turns": len(source_ids),
            "budget": engine.budget,
            "tree_recalls": recalls_tree,
            "baseline_recalls": recalls_base,
            "tree_total_retrieved": len(tree_segs),
            "tree_num_leaves": tree_result.num_leaves,
            "tree_budget_per_leaf": tree_result.budget_per_leaf,
            "tree_num_retrieve": tree_result.num_retrieve_actions,
            "tree_num_push": tree_result.num_push_actions,
            "tree_num_pop": tree_result.num_pop_actions,
            "leaf_descriptions": tree_result.leaf_descriptions,
            "leaf_queries": tree_result.leaf_queries,
            "avg_leaf_query_sim_to_question": avg_sim,
            "action_log": [
                {
                    "step": e.step,
                    "depth": e.depth,
                    "action": e.action,
                    "argument": e.argument,
                    "num_retrieved": e.num_retrieved,
                    "cues": e.cues,
                }
                for e in tree_result.action_log
            ],
            "embed_calls": tree_result.embed_calls,
            "llm_calls": tree_result.llm_calls,
            "elapsed_s": round(tree_result.elapsed, 2),
        }
        results.append(row)

        primary = engine.budget
        d = recalls_tree[f"r@{primary}"] - recalls_base[f"r@{primary}"]
        marker = "+" if d > 0.001 else ("-" if d < -0.001 else "=")
        print(
            f"[{i+1}/{len(questions)}] {marker} "
            f"B={recalls_base[f'r@{primary}']:.3f} "
            f"T={recalls_tree[f'r@{primary}']:.3f} "
            f"d={d:+.3f} "
            f"leaves={tree_result.num_leaves} "
            f"segs={len(tree_segs)}/{engine.budget} "
            f"sim={avg_sim:.3f} | {q['category']}: {q_text[:55]}..."
        )

        engine.save_caches()

    return results


def summarize(results: list[dict], budgets: tuple[int, ...], label: str) -> dict:
    n = len(results)
    print(f"\n{'='*70}\n{label} ({n} questions)\n{'='*70}")
    summary = {"label": label, "n": n}
    for b in budgets:
        key = f"r@{b}"
        bvs = [r["baseline_recalls"][key] for r in results]
        tvs = [r["tree_recalls"][key] for r in results]
        deltas = [t - bv for t, bv in zip(tvs, bvs)]
        w = sum(1 for d in deltas if d > 0.001)
        ties = sum(1 for d in deltas if abs(d) <= 0.001)
        l = sum(1 for d in deltas if d < -0.001)
        avg_b = sum(bvs) / n
        avg_t = sum(tvs) / n
        avg_d = sum(deltas) / n
        print(f"  {key:>8s}: B={avg_b:.3f} T={avg_t:.3f} Δ={avg_d:+.3f} W/T/L={w}/{ties}/{l}")
        summary[f"baseline_{key}"] = round(avg_b, 4)
        summary[f"tree_{key}"] = round(avg_t, 4)
        summary[f"delta_{key}"] = round(avg_d, 4)
        summary[f"wtl_{key}"] = f"{w}/{ties}/{l}"

    avg_leaves = sum(r["tree_num_leaves"] for r in results) / n
    avg_push = sum(r["tree_num_push"] for r in results) / n
    avg_pop = sum(r["tree_num_pop"] for r in results) / n
    avg_ret = sum(r["tree_num_retrieve"] for r in results) / n
    sims = [r["avg_leaf_query_sim_to_question"] for r in results
            if r["avg_leaf_query_sim_to_question"] is not None]
    avg_sim = (sum(sims) / len(sims)) if sims else None
    avg_total = sum(r["tree_total_retrieved"] for r in results) / n
    avg_llm = sum(r["llm_calls"] for r in results) / n
    avg_emb = sum(r["embed_calls"] for r in results) / n

    print(f"\n  Avg leaves/q: {avg_leaves:.2f}   "
          f"PUSH/RETRIEVE/POP = {avg_push:.1f}/{avg_ret:.1f}/{avg_pop:.1f}")
    print(f"  Avg segs retrieved: {avg_total:.1f}   "
          f"Avg leaf-query cosine to Q: "
          f"{avg_sim:.3f}" if avg_sim is not None else "  (n/a)")
    print(f"  Avg LLM calls: {avg_llm:.1f}   Avg embed calls: {avg_emb:.1f}")

    summary["avg_num_leaves"] = round(avg_leaves, 2)
    summary["avg_push_actions"] = round(avg_push, 2)
    summary["avg_retrieve_actions"] = round(avg_ret, 2)
    summary["avg_pop_actions"] = round(avg_pop, 2)
    summary["avg_total_retrieved"] = round(avg_total, 2)
    summary["avg_leaf_query_sim_to_question"] = (
        round(avg_sim, 4) if avg_sim is not None else None
    )
    summary["avg_llm_calls"] = round(avg_llm, 2)
    summary["avg_embed_calls"] = round(avg_emb, 2)
    return summary


def per_category_summary(results: list[dict], budgets: tuple[int, ...]) -> dict:
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        print()
        out[cat] = summarize(rs, budgets, f"category={cat}")
    return out


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------
TARGET_CATEGORIES = {"proactive", "procedural", "constraint_propagation"}


def load_target_questions() -> list[dict]:
    questions: list[dict] = []
    with open(DATA_DIR / "questions_synthetic.json") as f:
        syn = json.load(f)
    with open(DATA_DIR / "questions_advanced.json") as f:
        adv = json.load(f)
    for q in syn + adv:
        if q.get("category") in TARGET_CATEGORIES:
            questions.append(q)
    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=20,
                        help="Total retrieved segments cap (K).")
    parser.add_argument("--all-budgets", action="store_true",
                        help="Run both K=20 and K=50.")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--category", default=None,
                        help="Filter by category (proactive|procedural|constraint_propagation).")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_target_questions()
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    print(f"Loaded {len(questions)} questions ({len(TARGET_CATEGORIES)} categories).")

    store = build_combined_store()
    print(f"Loaded combined store with {len(store.segments)} segments.")

    budgets_to_run = [20, 50] if args.all_budgets else [args.budget]

    for budget in budgets_to_run:
        eval_budgets = (20, 50) if budget >= 50 else (20,)
        print(f"\n{'#'*70}\n# RUN WITH BUDGET K={budget}\n{'#'*70}")
        engine = ContextTreeV2(
            store=store,
            budget=budget,
            max_depth=args.max_depth,
            max_actions=args.max_actions,
        )
        results = evaluate(engine, questions, eval_budgets, verbose=args.verbose)

        label = f"tree_v2_K{budget}"
        summary = summarize(results, eval_budgets, label)
        cat_summaries = per_category_summary(results, eval_budgets)

        out_file = RESULTS_DIR / f"tree_v2_K{budget}.json"
        with open(out_file, "w") as f:
            json.dump(
                {
                    "config": {
                        "budget": budget,
                        "max_depth": args.max_depth,
                        "max_actions": args.max_actions,
                        "model": MODEL,
                        "embed_model": EMBED_MODEL,
                    },
                    "summary": summary,
                    "per_category": cat_summaries,
                    "results": results,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nSaved: {out_file}")

        engine.save_caches()


if __name__ == "__main__":
    main()
