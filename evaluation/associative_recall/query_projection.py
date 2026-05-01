"""Query-direction embedding projection.

Zero-LLM architectural variant. Learn a "question-ness" direction once from
question/statement template pairs, then at query time project each query
embedding away from that direction before cosine retrieval.

Design:
  q_dir = normalize(mean(question_embs) - mean(statement_embs))
  q_emb' = q_emb - alpha * (q_emb . q_dir) * q_dir
  q_emb' = q_emb' / ||q_emb'||
  retrieve top-K from store using q_emb'

Variants implemented here:
  QProjOnly(alpha)  — projected query, no LLM cues
  QProjV2f(alpha)   — projected query for hop0, then v2f cues (cues NOT projected)

Dedicated caches avoid concurrent-agent corruption:
  qproj_embedding_cache.json
  qproj_llm_cache.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EmbeddingCache,
    LLMCache,
    SegmentStore,
)
from best_shot import (
    V2F_PROMPT,
    BestshotBase,
    BestshotResult,
    _format_segments,
    _parse_cues,
)
from openai import OpenAI

# ---------------------------------------------------------------------------
# Dedicated caches (read many shared caches for hits, write only to own files)
# ---------------------------------------------------------------------------

_QPROJ_EMB_FILE = CACHE_DIR / "qproj_embedding_cache.json"
_QPROJ_LLM_FILE = CACHE_DIR / "qproj_llm_cache.json"

_SHARED_EMB_READ = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "agent_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "synth_test_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "fewshot_embedding_cache.json",
    "antipara_embedding_cache.json",
    "inv_query_embedding_cache.json",
    "qproj_embedding_cache.json",
)
_SHARED_LLM_READ = (
    "llm_cache.json",
    "arch_llm_cache.json",
    "agent_llm_cache.json",
    "tree_llm_cache.json",
    "frontier_llm_cache.json",
    "meta_llm_cache.json",
    "optim_llm_cache.json",
    "synth_test_llm_cache.json",
    "bestshot_llm_cache.json",
    "fewshot_llm_cache.json",
    "antipara_llm_cache.json",
    "inv_query_llm_cache.json",
    "qproj_llm_cache.json",
)


class QProjEmbeddingCache(EmbeddingCache):
    """Reads shared embedding caches (best-effort), writes to dedicated file."""

    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in _SHARED_EMB_READ:
            p = self.cache_dir / name
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        self.cache_file = _QPROJ_EMB_FILE
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


class QProjLLMCache(LLMCache):
    """Reads shared LLM caches (best-effort), writes to dedicated file."""

    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in _SHARED_LLM_READ:
            p = self.cache_dir / name
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            for k, v in data.items():
                if v:
                    self._cache[k] = v
        self.cache_file = _QPROJ_LLM_FILE
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Question/Statement templates for learning the direction
# ---------------------------------------------------------------------------

QUESTION_TEMPLATES: list[str] = [
    # generic what/which
    "what is the status of that project",
    "what did we decide about the deadline",
    "what are the next steps we agreed on",
    "what happened with that issue last week",
    "what did the team conclude about the proposal",
    "what was the main problem with the approach",
    "what are the open questions still remaining",
    "what does the user want us to change",
    "what did i tell you about the budget",
    "what kind of setup are we using for deployment",
    "what tool did we pick for the job",
    "what were the trade-offs we discussed",
    "what is the plan for the weekend trip",
    "what did my friend say about the concert",
    "what book did i mention wanting to read",
    "what movie are we watching tonight",
    "what restaurant did we go to last time",
    "what time is the appointment",
    "what color did we choose for the room",
    "what ingredients do we need for the recipe",
    # how
    "how did we decide to do it",
    "how does this process actually work",
    "how many people are involved in the meeting",
    "how long will the trip take",
    "how should i handle this situation",
    "how far along is the draft",
    "how did you solve that bug",
    "how much did the repair cost",
    "how often do we meet with the client",
    "how is everyone feeling about the change",
    "how did the interview go yesterday",
    "how do we reset the password",
    "how big is the new apartment",
    "how hot does it get in summer",
    "how did you learn to play guitar",
    # when
    "when was the meeting scheduled",
    "when did we last talk about this",
    "when is the deadline for submission",
    "when should i send the follow-up email",
    "when did you move to the new place",
    "when will the package arrive",
    "when did we first meet",
    "when is the next family gathering",
    "when do you usually go to bed",
    "when was the last time we ate there",
    # why
    "why did we change direction on the design",
    "why is the server running slow",
    "why did you pick that framework",
    "why are we still debating this",
    "why did she cancel the trip",
    "why is the deadline so tight",
    "why does my code keep crashing",
    "why did we switch vendors",
    "why is this component failing",
    "why did the team reject the proposal",
    # who
    "who said they would handle it",
    "who is leading the project now",
    "who was at the dinner last night",
    "who should i reach out to about this",
    "who wrote the initial draft of the document",
    "who is responsible for the deployment",
    "who did you hire for the role",
    "who owns the repository",
    "who signed off on the change",
    "who taught you how to cook",
    # where
    "where did you put the keys",
    "where is the meeting being held",
    "where did we leave off in the conversation",
    "where can i find the documentation",
    "where is the bug actually coming from",
    "where did you go on vacation",
    "where did we park the car",
    "where is the data stored",
    "where should i send the invoice",
    "where do you want to go for dinner",
    # can/could/should/would/did/does
    "can you summarize what was discussed",
    "can you remind me of the decision",
    "could you walk me through the architecture",
    "should we move forward with option a",
    "would it be okay to reschedule",
    "did we ever resolve that issue",
    "did you hear back from the client",
    "does this approach make sense to you",
    "does anyone have strong feelings about it",
    "am i missing something important here",
    "is there anything i should know before the call",
    "are we still on track for friday",
    "was the feedback positive or negative",
    "were you able to finish the draft",
    "have we discussed this topic before",
    # follow-up style
    "what else did we talk about regarding this",
    "tell me more about the part we glossed over",
    "remind me what we planned for monday",
    "can you recall the specific number we used",
    "what was that one thing you mentioned earlier",
    "do you remember what i said about the goal",
    "what did i tell you yesterday about the client",
    "what conclusions did we reach",
    "what recommendations did the review surface",
    "what concerns were raised in the discussion",
]


STATEMENT_TEMPLATES: list[str] = [
    # generic statements about topics
    "the project status update is on the shared drive",
    "we decided to push the deadline by a week",
    "the next steps are to finalize the design and notify the team",
    "there was an issue with the pipeline that broke the build",
    "the team concluded the proposal needed more data",
    "the main problem with the approach was the latency cost",
    "a few open questions remain about the edge cases",
    "the user wants us to change the color scheme",
    "the budget for the quarter is already locked in",
    "the deployment setup uses kubernetes with helm charts",
    "we picked terraform for infrastructure provisioning",
    "the trade-offs included performance versus ease of maintenance",
    "the plan for the weekend trip is to drive up on friday evening",
    "my friend said the concert was amazing",
    "i mentioned a new book i wanted to read on history",
    "we are watching a thriller tonight",
    "we went to the italian restaurant last time",
    "the appointment is at three in the afternoon",
    "we chose a warm off-white for the living room",
    "the recipe needs flour eggs butter and sugar",
    # how-style (procedural)
    "we decided to do it by splitting into two subtasks",
    "the process works by first validating then persisting",
    "there are four people involved in the weekly meeting",
    "the trip will take about six hours with stops",
    "you should handle this by escalating to the lead",
    "the draft is about halfway finished",
    "i solved that bug by adding a null check in the parser",
    "the repair cost two hundred dollars in total",
    "we meet with the client every other wednesday",
    "everyone seems okay with the change but a bit cautious",
    "the interview went well and they seemed interested",
    "you reset the password from the account settings page",
    "the new apartment is about nine hundred square feet",
    "summers here get up into the nineties regularly",
    "i learned to play guitar through youtube lessons",
    # when-style
    "the meeting is scheduled for thursday at eleven",
    "we last talked about this during the sprint review",
    "the submission deadline is next friday at five",
    "send the follow-up email tomorrow morning",
    "i moved to the new place in september",
    "the package arrives on wednesday according to tracking",
    "we first met at the conference in boston",
    "the next family gathering is over thanksgiving",
    "i usually go to bed around eleven thirty",
    "we ate there a few weeks ago on tuesday",
    # why-style
    "we changed direction because the original approach scaled poorly",
    "the server is running slow because of a memory leak",
    "i picked the framework because of its type system support",
    "we are still debating because the data is inconclusive",
    "she canceled the trip because her flight was delayed",
    "the deadline is tight because the client moved it up",
    "the code keeps crashing because of a race condition in setup",
    "we switched vendors due to repeated quality issues",
    "the component is failing because of a misconfigured dependency",
    "the team rejected the proposal because the costs were too high",
    # who-style
    "alex said they would handle the rollout",
    "priya is leading the project now",
    "there were six of us at dinner last night",
    "reach out to dana for anything about this",
    "sam wrote the initial draft of the document",
    "the ops team is responsible for deployment",
    "we hired marcus for the senior role",
    "the platform team owns the repository",
    "the director signed off on the change last tuesday",
    "my grandmother taught me how to cook traditional dishes",
    # where-style
    "the keys are on the kitchen counter near the fruit bowl",
    "the meeting is being held in the second floor conference room",
    "we left off discussing the onboarding redesign",
    "the documentation is in the internal wiki under setup",
    "the bug is coming from the serialization layer",
    "we went to portugal on vacation last summer",
    "we parked the car in the garage across the street",
    "the data is stored in the primary postgres cluster",
    "send the invoice to the finance team at the main address",
    "i want to go to the new thai place for dinner",
    # declarative about conversations
    "we discussed the topic in depth during the last call",
    "the part we glossed over was the edge case handling",
    "on monday we planned to finish the migration",
    "the specific number we used was four hundred and twelve",
    "earlier you mentioned the feature flag strategy",
    "yesterday i said the goal was to simplify the api",
    "yesterday i told you about the client escalation",
    "our conclusions were to ship the first phase and iterate",
    "the review surfaced a few recommendations about testing",
    "the discussion raised concerns about security and access",
    "the documentation update was merged this morning",
    "i finished reading the draft yesterday evening",
    "we need more data before making a final call",
    "the vendor promised delivery by end of the month",
    "the bug only appears on the staging environment",
    "the team is aligned on the new architecture",
    "sales numbers dipped in march but recovered in april",
    "the client signed the contract on friday",
    "we added tracing to narrow down the performance issue",
    "the migration script ran for about four hours",
    "feedback from users has been mostly positive",
    "the new feature is behind a flag until we validate",
    "training starts next week for the ml pipeline",
    "the code review pointed out a few style nits",
    "memory usage dropped after we tuned the cache",
]


# ---------------------------------------------------------------------------
# Direction learning
# ---------------------------------------------------------------------------


def _mean_embedding(
    cache: QProjEmbeddingCache,
    client: OpenAI,
    texts: list[str],
    embed_model: str = "text-embedding-3-small",
) -> tuple[np.ndarray, int]:
    """Return (mean_embedding, num_newly_embedded). Uses cache."""

    embs = []
    new_count = 0
    for t in texts:
        cached = cache.get(t)
        if cached is not None:
            embs.append(cached)
        else:
            resp = client.embeddings.create(model=embed_model, input=[t])
            e = np.array(resp.data[0].embedding, dtype=np.float32)
            cache.put(t, e)
            embs.append(e)
            new_count += 1
    M = np.stack(embs, axis=0)
    return M.mean(axis=0), new_count


def learn_question_direction(
    question_templates: list[str] | None = None,
    statement_templates: list[str] | None = None,
    client: OpenAI | None = None,
    save_path: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Learn the unit-length question-ness direction.

    Returns (direction, stats).
    """
    from associative_recall import EMBED_MODEL

    qs = question_templates or QUESTION_TEMPLATES
    ss = statement_templates or STATEMENT_TEMPLATES
    cache = QProjEmbeddingCache()
    client = client or OpenAI(timeout=60.0)

    q_mean, new_q = _mean_embedding(cache, client, qs, EMBED_MODEL)
    s_mean, new_s = _mean_embedding(cache, client, ss, EMBED_MODEL)
    cache.save()

    diff = q_mean - s_mean
    norm = float(np.linalg.norm(diff))
    if norm < 1e-10:
        raise RuntimeError("Question and statement means are essentially equal.")
    direction = (diff / norm).astype(np.float32)

    stats = {
        "n_question_templates": len(qs),
        "n_statement_templates": len(ss),
        "newly_embedded": new_q + new_s,
        "diff_norm": norm,
        "q_mean_norm": float(np.linalg.norm(q_mean)),
        "s_mean_norm": float(np.linalg.norm(s_mean)),
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, direction)

    return direction, stats


def project_away(emb: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    """Project `emb` away from `direction` with strength `alpha`.

    direction should already be unit length. Output is normalized to unit
    length (matching how store does cosine via normalized_embeddings).
    """
    d = direction.astype(np.float32)
    e = emb.astype(np.float32)
    scalar = float(np.dot(e, d))
    projected = e - alpha * scalar * d
    n = float(np.linalg.norm(projected))
    if n < 1e-10:
        # Degenerate: query was entirely along direction. Fall back to raw.
        return e / max(np.linalg.norm(e), 1e-10)
    return (projected / n).astype(np.float32)


# ---------------------------------------------------------------------------
# Retrieval variants
# ---------------------------------------------------------------------------


class _QProjBase(BestshotBase):
    """Base that installs dedicated qproj caches and holds the direction."""

    arch_name = "qproj_base"

    def __init__(
        self,
        store: SegmentStore,
        direction: np.ndarray,
        alpha: float = 1.0,
        client: OpenAI | None = None,
    ):
        super().__init__(store, client)
        self.embedding_cache = QProjEmbeddingCache()
        self.llm_cache = QProjLLMCache()
        self.direction = direction.astype(np.float32)
        self.alpha = float(alpha)

    def projected_query_embedding(self, question: str) -> np.ndarray:
        raw = self.embed_text(question)
        return project_away(raw, self.direction, self.alpha)


class QProjOnly(_QProjBase):
    """Zero-LLM: just cosine retrieval using the projected query."""

    arch_name = "qproj_only"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        proj_emb = self.projected_query_embedding(question)
        # Retrieve top-K with max needed budget upstream (eval does its own
        # cosine@K for the baseline). Here, return top-50 to cover both K=20
        # and K=50 budgets.
        result = self.store.search(proj_emb, top_k=50, conversation_id=conversation_id)
        return BestshotResult(
            segments=list(result.segments),
            metadata={
                "name": self.arch_name,
                "alpha": self.alpha,
            },
        )


class QProjV2f(_QProjBase):
    """Hybrid: projected query for hop0, then v2f cue generation (cues NOT
    projected). Union scored by presence in retrieval (v2f-style: exclusion
    dedup, appended in order)."""

    arch_name = "qproj_v2f"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        proj_emb = self.projected_query_embedding(question)
        hop0 = self.store.search(proj_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(all_segments)
        )
        prompt = V2F_PROMPT.format(question=question, context_section=context_section)
        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]

        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
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
                "alpha": self.alpha,
                "output": output,
                "cues": cues,
            },
        )


# Factory helpers for eval driver


def build_qproj_only(alpha: float, direction: np.ndarray):
    def factory(store: SegmentStore):
        return QProjOnly(store, direction=direction, alpha=alpha)

    factory.arch_name = f"qproj_{alpha:.1f}".replace(".0", ".0")
    return factory


def build_qproj_v2f(alpha: float, direction: np.ndarray):
    def factory(store: SegmentStore):
        return QProjV2f(store, direction=direction, alpha=alpha)

    factory.arch_name = f"qproj_{alpha:.1f}_v2f"
    return factory
