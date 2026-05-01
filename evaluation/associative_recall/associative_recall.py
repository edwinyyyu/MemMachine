"""Iterative associative recall engine.

Implements both single-shot baseline and multi-hop associative retrieval
using LLM-generated cues to expand recall beyond the original query's
semantic neighborhood.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
EMBED_MODEL = "text-embedding-3-small"


CUE_GENERATION_PROMPT_V1 = """\
You are helping retrieve information from a long conversation history. \
The user has asked a question, and we have retrieved some potentially relevant \
conversation excerpts so far.

Your job: generate 2-3 NEW search cues that would help find additional relevant \
parts of the conversation that we haven't found yet.

CRITICAL RULES for generating cues:
- Each cue should be 1-3 natural sentences, the kind of text that would actually \
appear in a chat conversation
- Do NOT rephrase the original question. Instead, generate text that would appear \
NEAR the answer in the conversation
- Think about what the user or assistant would have ACTUALLY SAID in messages \
related to this topic
- Use concrete vocabulary: specific tools, frameworks, actions, decisions, \
not abstract summaries
- If the retrieved context mentions specific topics, generate cues that explore \
ADJACENT topics that would likely appear nearby in the same conversation
- Generate text in the register of casual chat messages, not formal documents

Original question: {question}

{context_section}

Generate exactly {num_cues} search cues, each on its own line, prefixed with \
"CUE: ". Nothing else."""

CUE_GENERATION_PROMPT_V2 = """\
You are an associative memory retrieval system. Given a question about a past \
conversation and some already-retrieved excerpts, generate new search queries \
to find missing information.

KEY INSIGHT: The conversation was between a user and an AI assistant. Think about \
what specific messages would contain the answer:
- What would the USER have typed when discussing this topic?
- What would the ASSISTANT have responded with?
- What related topics would appear in NEARBY messages?

DO NOT generate:
- Rewordings of the original question
- Abstract summaries or meta-descriptions
- Questions (generate statements that match conversation content)

DO generate:
- Plausible conversation snippets with specific vocabulary
- Messages mentioning concrete details (tools, names, dates, actions)
- Content from adjacent turns that would appear near the target information

Original question: {question}

{context_section}

Generate exactly {num_cues} search cues. Each should be 1-3 sentences of \
plausible conversation content. Prefix each with "CUE: ". Nothing else."""

CUE_GENERATION_PROMPT_V3 = """\
You are performing iterative associative recall on a conversation history. \
Your goal: generate search text that is semantically CLOSE to conversation \
content we haven't found yet.

STRATEGY:
1. From the question, infer what TOPICS the conversation must have covered
2. From retrieved excerpts, identify SPECIFIC details (names, tools, dates, \
decisions) that point to related content
3. Generate text that would share vocabulary with the TARGET content — not the \
question, but the ANSWER as it would appear in the conversation

Think step by step:
- What kind of message would contain this information?
- What specific words/phrases would that message use?
- What would the surrounding messages discuss?

Original question: {question}

{context_section}

Generate exactly {num_cues} cues. Each cue: 1-3 natural sentences mimicking \
conversation content. Use specific vocabulary, not abstractions.
Format: "CUE: <text>" per line. Nothing else."""

CUE_GENERATION_PROMPT_V4 = """\
You are expanding a memory search. Given a question and retrieved conversation \
excerpts, generate search text to find MORE relevant parts of the conversation.

RULES:
1. READ the retrieved excerpts carefully. Extract specific names, tools, \
dates, and technical terms that actually appear in them.
2. Generate cues that explore ADJACENT topics to what's already retrieved. \
If retrieved text mentions "Flask routes," generate a cue about what likely \
came BEFORE or AFTER that discussion (e.g., database setup, deployment).
3. Each cue should be 1-2 sentences written as if someone typed it in a chat.
4. Use vocabulary FROM the retrieved excerpts, extended to adjacent topics.
5. Do NOT repeat the question or previous cues.

Original question: {question}

{context_section}

Generate exactly {num_cues} search cues. Format: "CUE: <text>" per line. \
Nothing else."""

CUE_GENERATION_PROMPT_V5 = """\
You are performing iterative associative recall on a conversation history to \
answer a question. Your job: generate search cues to find MISSING relevant \
parts of the conversation.

WHAT YOU KNOW:
- Original question: {question}
- Total conversation length: ~{conv_length} turns (turn IDs 0 to ~{max_turn_id})

{context_section}

YOUR TASK:
1. Consider what sub-topics or aspects the ANSWER to this question would need \
to cover.
2. Look at what you've already found. What topics/aspects are COVERED? What's \
MISSING?
3. Generate cues that target the MISSING content. Each cue should sound like \
something a user or assistant would actually type in a chat message.

CRITICAL:
- Do NOT rephrase the question. Generate text that would APPEAR in the \
conversation near the missing information.
- Target DIFFERENT regions of the conversation (early/middle/late turns).
- Use concrete, specific vocabulary — tool names, error messages, function \
names — not abstract summaries.
- Each cue should explore a DIFFERENT topic or aspect from what's already found.

Generate exactly {num_cues} search cues. Format: "CUE: <text>" per line. \
Nothing else."""

CUE_GENERATION_PROMPT_V6 = """\
You are performing iterative associative recall on a conversation history.

Question: {question}
Conversation: ~{conv_length} turns (IDs 0 to ~{max_turn_id})

{context_section}

INSTRUCTIONS:
1. The answer to this question likely involves content scattered across \
MULTIPLE parts of the conversation.
2. From the retrieved excerpts, identify what TOPICS have been covered so far.
3. Generate cues targeting content you HAVEN'T found yet — different topics, \
different time periods in the conversation, different aspects of the question.
4. Each cue: 1-2 sentences mimicking actual chat messages. Use specific \
vocabulary (tool names, error messages, code snippets, decisions).
5. Make each cue about a DIFFERENT aspect or sub-topic. Spread your search \
across unexplored territory.

Generate exactly {num_cues} search cues. Format: "CUE: <text>" per line. \
Nothing else."""

CUE_GENERATION_PROMPT_V7 = """\
You are generating search text to find parts of a conversation that haven't \
been retrieved yet.

Question: {question}

{context_section}

RULES:
1. Generate text that would APPEAR in the conversation — things a user or \
assistant would actually type. NOT meta-instructions like "find the message \
about X" or "show me where we discussed Y."
2. Each cue should be 1-2 sentences using specific, concrete vocabulary: \
tool names, error messages, code patterns, technical terms.
3. Each cue MUST target a DIFFERENT sub-topic from what's already been found. \
Look at the retrieved excerpts: what topics do they cover? Generate cues for \
DIFFERENT topics that the answer would also need.
4. Think about what the USER would have typed or what the ASSISTANT would have \
replied with. Match that register — casual questions, code snippets, \
technical explanations.

Generate exactly {num_cues} search cues. Format: "CUE: <text>" per line. \
Nothing else."""

CUE_GENERATION_PROMPT_V8 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared against conversation turns \
via cosine similarity.

Question: {question}

{context_section}

IMPORTANT — what makes an effective cue:
- A cue that SHARES VOCABULARY with the target turn. If the conversation \
mentions "debounce delay 300ms", a cue mentioning "debounce" and "300ms" \
will score high.
- Use specific nouns, verbs, and technical terms that would ACTUALLY appear \
in chat messages about this topic.
- Short user messages in conversations often look like: "hmm, what about X?" \
or "how do I handle Y?" — generate cues in that register too.

WHAT NOT TO DO:
- Do NOT write meta-instructions ("find the message about...", "show the \
turn where..."). These won't match any conversation content.
- Do NOT rephrase the original question.
- Do NOT generate abstract summaries.

Each cue should target a DIFFERENT aspect/sub-topic of the question. \
Maximize topical diversity across your {num_cues} cues.

Generate exactly {num_cues} search cues. Format: "CUE: <text>" per line. \
Nothing else."""

CUE_GENERATION_PROMPT_V9 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared against conversation turns \
via cosine similarity.

Question: {question}

{context_section}

IMPORTANT — what makes an effective cue:
- A cue that SHARES VOCABULARY with the target turn. If the conversation \
mentions "debounce delay 300ms", a cue mentioning "debounce" and "300ms" \
will score high.
- Use specific nouns, verbs, and technical terms that would ACTUALLY appear \
in chat messages about this topic.
- Short user messages in conversations often look like: "hmm, what about X?" \
or "how do I handle Y?" — generate cues in that register too.

WHAT NOT TO DO:
- Do NOT write meta-instructions ("find the message about...", "show the \
turn where..."). These won't match any conversation content.
- Do NOT rephrase the original question.
- Do NOT generate abstract summaries.

Each cue should target a DIFFERENT aspect/sub-topic of the question. \
Maximize topical diversity across your {num_cues} cues.

EXPANSION: Some retrieved turns look like they're part of a longer relevant \
exchange. For those turns, we should also look at their immediate neighbors \
(adjacent turns). After your cues, list turn IDs that should be expanded. \
Only expand turns that seem to be MID-CONVERSATION in a relevant exchange — \
NOT turns that are clearly off-topic or self-contained.

Format:
CUE: <search text>
CUE: <search text>
EXPAND: <comma-separated turn IDs to expand, or NONE>

Generate exactly {num_cues} search cues and one EXPAND line. Nothing else."""

CUE_GENERATION_PROMPT_V10 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

Generate a hypothetical ANSWER to this question — a short paragraph that \
describes what the relevant part of the conversation would say. Write it as \
if you are quoting from the conversation itself. Use specific details, \
names, numbers, and technical terms that would actually appear in the chat.

Do NOT answer the question from your knowledge. Instead, IMAGINE what the \
conversation content would look like. Write the kind of text that would \
appear in the actual chat messages containing the answer.

Generate exactly {num_cues} different hypothetical answer passages, each \
targeting a different aspect of the question.

Format: "CUE: <hypothetical answer text>" per line. Nothing else."""

CUE_GENERATION_PROMPT_V11 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

Generate plausible CHAT MESSAGES that would appear in the conversation near \
the answer. For each cue, write EITHER:
- What the USER would have typed (casual, short, with typos or informal phrasing)
- What the ASSISTANT would have replied (helpful, specific, with concrete details)

Match the register of real chat messages. Users say things like "hey what \
about..." or "can you help me with..." and assistants respond with specific \
explanations, code, or suggestions.

Each cue should target a DIFFERENT part of the answer. Do NOT repeat the \
original question.

Generate exactly {num_cues} cues. Format: "CUE: <chat message>" per line. \
Nothing else."""

CUE_GENERATION_PROMPT_V12 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

Write a short narrative paragraph (2-3 sentences) describing what the \
relevant part of the conversation discussed. Use specific vocabulary \
that would appear in the actual messages — tool names, concepts, decisions, \
actions. Write as if summarizing a conversation excerpt.

Generate {num_cues} different narrative descriptions, each covering a \
different aspect or region of the conversation relevant to the question.

Format: "CUE: <narrative paragraph>" per line. Nothing else."""

CUE_GENERATION_PROMPT_V13 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

Generate {num_cues} search cues to find parts of the conversation relevant \
to answering this question. Each cue will be embedded and matched against \
conversation turns.

For each cue, use whatever format you think will best match the target \
content — a keyword list, a sentence, a hypothetical quote, a question, \
a code snippet, or anything else. Choose the format that maximizes vocabulary \
overlap with the conversation content you're trying to find.

Each cue should target DIFFERENT content. Do NOT repeat the original question.

Format: "CUE: <text>" per line. Nothing else."""

CUE_GENERATION_PROMPT_V14 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

Look at what has ALREADY been retrieved. Generate cues that target content \
you HAVEN'T found yet. Specifically:
- What aspects of the question are NOT covered by the retrieved excerpts?
- What related topics would appear ELSEWHERE in the conversation?
- What vocabulary would the MISSING content use?

Each cue must be about a DIFFERENT topic from what's already been found. \
Use specific words and phrases that would appear in the missing messages.

Generate exactly {num_cues} cues. Format: "CUE: <text>" per line. \
Nothing else."""

CUE_GENERATION_PROMPT_V15 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate {num_cues} search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

CUE_GENERATION_PROMPT_V16 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

SCRATCHPAD — build a brief hypothesis about the answer:
- Based on what you've seen so far, what do you think the answer involves?
- What specific information is still MISSING to fully answer the question?
- What vocabulary or topics would the missing content use?

Then generate {num_cues} search cues targeting the MISSING information. \
Each cue should use specific words that would appear in the target turns.

Format:
HYPOTHESIS: <1-2 sentence current understanding>
MISSING: <what you still need to find>
CUE: <text>
CUE: <text>
Nothing else."""

CUE_GENERATION_PROMPT_V17 = """\
Question: {question}

{context_section}

What conversation content is still missing? Generate {num_cues} search \
cues to find it. Each cue will be embedded and matched via cosine similarity.

Format:
CUE: <text>
Nothing else."""

CUE_GENERATION_PROMPT_V18 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First: What do you think the answer involves? What specific information \
is still MISSING from what's been retrieved? Is this search going well?

Then generate {num_cues} search cues targeting the missing content.

Format:
STATUS: <1-2 sentences: hypothesis + what's missing>
CUE: <text>
CUE: <text>
Nothing else."""

# v19: HyDE-style full hypothetical answer (not conversation fragments)
CUE_GENERATION_PROMPT_V19 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

Write a FULL HYPOTHETICAL ANSWER to this question as if you were directly \
answering it from memory. Include specific details, names, numbers, dates, \
tools, and technical terms that the conversation would have contained. \
Write naturally, as if recounting what was discussed.

Do NOT write conversation-style messages. Instead, write a direct answer \
paragraph that naturally shares vocabulary with the conversation content.

Generate {num_cues} different answer paragraphs, each covering a different \
aspect of what was discussed.

Format:
CUE: <full hypothetical answer paragraph>
Nothing else."""

# v20: Perspective-taking — recall from INSIDE the experience
CUE_GENERATION_PROMPT_V20 = """\
You are the person who had this conversation. Someone is asking you: \
{question}

{context_section}

Think back to your conversation. What do you remember discussing? What \
specific details come to mind? What were you thinking about when this \
topic came up?

Generate {num_cues} memory fragments — things you actually remember from \
the conversation, in your own words, with specific details. Include the \
exact words, phrases, tools, names, or numbers you recall.

First assess what you remember so far, then generate new memories.

Format:
ASSESSMENT: <what do I remember so far? what's fuzzy?>
CUE: <a specific memory fragment>
Nothing else."""

# v21: Question decomposition — break complex questions into sub-queries
CUE_GENERATION_PROMPT_V21 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, assess what has been found so far and what aspects of the question \
remain unanswered.

Then break the unanswered parts into distinct sub-questions. For each \
sub-question, generate a search cue using vocabulary that would appear \
in the conversation turns containing the answer.

Generate {num_cues} cues, each targeting a DIFFERENT sub-question.

Format:
ASSESSMENT: <what's covered, what's missing>
CUE: <search text for a specific sub-question>
Nothing else."""

# v22: Cue-as-continuation — generate what the NEXT message would have been
CUE_GENERATION_PROMPT_V22 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

Look at the most recently retrieved conversation excerpts. For each gap \
or transition you notice, generate what the NEXT message in the conversation \
would have been — the natural continuation of the discussion. This \
continuation will share vocabulary with adjacent content that hasn't been \
retrieved yet.

First assess what's been found, then generate continuations.

Generate {num_cues} continuation cues.

Format:
ASSESSMENT: <what transitions/gaps do I see?>
CUE: <natural continuation message>
Nothing else."""

# v23: Minimal — zero format guidance, let model choose freely
CUE_GENERATION_PROMPT_V23 = """\
Question: {question}

{context_section}

Generate {num_cues} search texts to find conversation content that would \
help answer this question. Each will be embedded and matched via cosine \
similarity against conversation turns.

Format:
CUE: <text>
Nothing else."""

# v24: Self-monitoring + explicit vocabulary extraction from retrieved content
CUE_GENERATION_PROMPT_V24 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, extract KEY TERMS from the retrieved excerpts — specific nouns, \
names, tools, numbers, or phrases that appear in the actual text. These \
are your anchors.

Then assess: what aspects of the question are NOT covered by the retrieved \
content? What vocabulary would the missing content use?

Generate {num_cues} cues that combine extracted key terms with hypothesized \
vocabulary for the missing content.

Format:
TERMS: <comma-separated key terms from retrieved text>
ASSESSMENT: <what's missing>
CUE: <text combining known terms + hypothesized missing vocabulary>
Nothing else."""

# v25: v15 + explicit keyword-density instruction
CUE_GENERATION_PROMPT_V25 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate {num_cues} search cues based on your assessment. Each cue \
should be a SHORT list of keywords and phrases (5-15 words) that would \
appear in the target conversation turns. Use "quoted phrases" and OR \
separators. Do NOT write full sentences.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <keyword list>
Nothing else."""

# v26: Scratchpad — carry forward a summary instead of raw segments
# The context section will be modified to show scratchpad instead of raw segments
CUE_GENERATION_PROMPT_V26 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

Based on the scratchpad above, what information is still MISSING to answer \
the question? Generate {num_cues} search cues targeting the missing content. \
Use specific vocabulary that would appear in the target conversation turns.

Format:
SCRATCHPAD_UPDATE: <1-2 sentences: what did we just learn? what's still missing?>
CUE: <text>
Nothing else."""

# v27: v15 self-monitoring + 2-hop (test multi-hop on v15)
# Same prompt as v15, just used with --max-hops 2

# v28: Adaptive — let the model choose strategy based on question type
CUE_GENERATION_PROMPT_V28 = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, assess:
1. What TYPE of question is this? (factual lookup, temporal, synthesis, \
comparison, evolution/change, opinion/preference)
2. How well is this search going? What's missing?
3. What search STRATEGY would work best for the missing content?

Then generate {num_cues} search cues using your chosen strategy. Use \
specific vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <question type + what's missing + strategy>
CUE: <text>
Nothing else."""

PROMPT_VERSIONS = {
    "v1": CUE_GENERATION_PROMPT_V1,
    "v2": CUE_GENERATION_PROMPT_V2,
    "v3": CUE_GENERATION_PROMPT_V3,
    "v4": CUE_GENERATION_PROMPT_V4,
    "v5": CUE_GENERATION_PROMPT_V5,
    "v6": CUE_GENERATION_PROMPT_V6,
    "v7": CUE_GENERATION_PROMPT_V7,
    "v8": CUE_GENERATION_PROMPT_V8,
    "v9": CUE_GENERATION_PROMPT_V9,
    "v10": CUE_GENERATION_PROMPT_V10,
    "v11": CUE_GENERATION_PROMPT_V11,
    "v12": CUE_GENERATION_PROMPT_V12,
    "v13": CUE_GENERATION_PROMPT_V13,
    "v14": CUE_GENERATION_PROMPT_V14,
    "v15": CUE_GENERATION_PROMPT_V15,
    "v16": CUE_GENERATION_PROMPT_V16,
    "v17": CUE_GENERATION_PROMPT_V17,
    "v18": CUE_GENERATION_PROMPT_V18,
    "v19": CUE_GENERATION_PROMPT_V19,
    "v20": CUE_GENERATION_PROMPT_V20,
    "v21": CUE_GENERATION_PROMPT_V21,
    "v22": CUE_GENERATION_PROMPT_V22,
    "v23": CUE_GENERATION_PROMPT_V23,
    "v24": CUE_GENERATION_PROMPT_V24,
    "v25": CUE_GENERATION_PROMPT_V25,
    "v26": CUE_GENERATION_PROMPT_V26,
    "v28": CUE_GENERATION_PROMPT_V28,
}


@dataclass
class Segment:
    conversation_id: str
    turn_id: int
    role: str
    text: str
    index: int


@dataclass
class RetrievalResult:
    segments: list[Segment]
    scores: list[float]


@dataclass
class HopResult:
    hop_number: int
    cues: list[str]
    retrieved: RetrievalResult
    new_turn_ids: set[int]
    expand_targets: set[int] | None = None
    num_expanded: int = 0


@dataclass
class AssociativeResult:
    question: str
    conversation_id: str
    hops: list[HopResult]
    all_retrieved_segments: list[Segment]
    all_retrieved_turn_ids: set[int]


class SegmentStore:
    def __init__(self, data_dir: Path = DATA_DIR, npz_name: str = "segments.npz"):
        npz_path = data_dir / npz_name
        data = np.load(npz_path, allow_pickle=True)
        self.embeddings: np.ndarray = data["embeddings"]
        self.conversation_ids: np.ndarray = data["conversation_ids"]
        self.turn_ids: np.ndarray = data["turn_ids"]
        self.roles: np.ndarray = data["roles"]
        self.texts: np.ndarray = data["texts"]

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.normalized_embeddings = self.embeddings / norms

        self.segments = [
            Segment(
                conversation_id=str(self.conversation_ids[i]),
                turn_id=int(self.turn_ids[i]),
                role=str(self.roles[i]),
                text=str(self.texts[i]),
                index=i,
            )
            for i in range(len(self.texts))
        ]

        # Build turn_id -> index mapping per conversation for neighbor lookup
        self._turn_index: dict[str, dict[int, int]] = {}
        for i, seg in enumerate(self.segments):
            cid = seg.conversation_id
            if cid not in self._turn_index:
                self._turn_index[cid] = {}
            self._turn_index[cid][seg.turn_id] = i

    def get_neighbors(
        self,
        segment: "Segment",
        radius: int = 1,
        exclude_indices: set[int] | None = None,
    ) -> list["Segment"]:
        """Get neighboring turns within `radius` of the given segment."""
        conv_map = self._turn_index.get(segment.conversation_id, {})
        neighbors = []
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            neighbor_tid = segment.turn_id + offset
            if neighbor_tid in conv_map:
                idx = conv_map[neighbor_tid]
                if exclude_indices and idx in exclude_indices:
                    continue
                neighbors.append(self.segments[idx])
        return neighbors

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        conversation_id: str | None = None,
        exclude_indices: set[int] | None = None,
    ) -> RetrievalResult:
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-10)
        similarities = self.normalized_embeddings @ query_norm

        if conversation_id is not None:
            mask = self.conversation_ids == conversation_id
            similarities = np.where(mask, similarities, -1.0)

        if exclude_indices:
            for idx in exclude_indices:
                similarities[idx] = -1.0

        top_indices = np.argsort(similarities)[::-1][:top_k]
        segments = [self.segments[i] for i in top_indices if similarities[i] > -1.0]
        scores = [float(similarities[i]) for i in top_indices if similarities[i] > -1.0]

        return RetrievalResult(segments=segments, scores=scores)


class EmbeddingCache:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / "embedding_cache.json"
        self._cache: dict[str, list[float]] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache = json.load(f)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        key = self._key(text)
        if key in self._cache:
            return np.array(self._cache[key], dtype=np.float32)
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()

    def save(self) -> None:
        tmp_file = self.cache_file.with_suffix(".json.tmp")
        with open(tmp_file, "w") as f:
            json.dump(self._cache, f)
        tmp_file.replace(self.cache_file)


class LLMCache:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / "llm_cache.json"
        self._cache: dict[str, str] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache = json.load(f)

    def _key(self, model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str) -> str | None:
        return self._cache.get(self._key(model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        self._cache[self._key(model, prompt)] = response

    def save(self) -> None:
        tmp_file = self.cache_file.with_suffix(".json.tmp")
        with open(tmp_file, "w") as f:
            json.dump(self._cache, f)
        tmp_file.replace(self.cache_file)


class AssociativeRecallEngine:
    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        cue_model: str = "gpt-5-mini",
        prompt_version: str = "v1",
        max_hops: int = 3,
        top_k_per_hop: int = 10,
        num_cues: int = 2,
        neighbor_radius: int = 0,
    ):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.cue_model = cue_model
        self.prompt_version = prompt_version
        self.max_hops = max_hops
        self.top_k_per_hop = top_k_per_hop
        self.num_cues = num_cues
        self.neighbor_radius = neighbor_radius
        self.prompt_template = PROMPT_VERSIONS[prompt_version]
        self.embedding_cache = EmbeddingCache()
        self.llm_cache = LLMCache()

    def embed_text(self, text: str) -> np.ndarray:
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, embedding)
        return embedding

    def _build_context_section_legacy(
        self,
        all_segments: list[Segment],
        new_segments: list[Segment],
        hop_number: int,
        previous_cues: list[str] | None = None,
    ) -> str:
        """Build context section for v1-v4 prompts (legacy behavior)."""
        if new_segments or all_segments:
            context_lines = []
            display_segments = new_segments or all_segments[:8]
            for seg in display_segments[:8]:
                context_lines.append(
                    f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:300]}"
                )
            context_section = (
                f"Hop {hop_number}: most recently retrieved conversation excerpts:\n"
                + "\n".join(context_lines)
            )
            if previous_cues:
                context_section += (
                    "\n\nPrevious cues that were already tried (do NOT repeat these):\n"
                    + "\n".join(f"- {c}" for c in previous_cues)
                )
        else:
            context_section = (
                "No conversation excerpts retrieved yet. Generate cues based on "
                "what you'd expect to find in a conversation about this topic."
            )
        return context_section

    def _build_context_section_accumulated(
        self,
        all_segments: list[Segment],
        hop_number: int,
        previous_cues: list[str] | None = None,
        new_segments: list[Segment] | None = None,
    ) -> str:
        """Build context section for v5+ prompts with accumulated context."""
        if not all_segments:
            return (
                "No conversation excerpts retrieved yet. Generate cues based on "
                "what you'd expect to find in a conversation about this topic."
            )

        # Sort segments by turn_id for chronological view
        sorted_segs = sorted(all_segments, key=lambda s: s.turn_id)

        # Build territory coverage summary
        covered_ids = sorted(set(s.turn_id for s in sorted_segs))
        # Identify clusters of covered turns
        clusters: list[tuple[int, int]] = []
        cluster_start = covered_ids[0]
        cluster_end = covered_ids[0]
        for tid in covered_ids[1:]:
            if tid - cluster_end <= 2:  # Allow small gaps
                cluster_end = tid
            else:
                clusters.append((cluster_start, cluster_end))
                cluster_start = tid
                cluster_end = tid
        clusters.append((cluster_start, cluster_end))

        territory_desc = (
            f"Retrieved {len(covered_ids)} turns so far, covering these regions: "
            + ", ".join(f"turns {s}-{e}" for s, e in clusters)
        )

        # Show all accumulated segments (up to limit), sorted chronologically
        context_lines = []
        display_limit = 12 if hop_number <= 2 else 16
        for seg in sorted_segs[:display_limit]:
            context_lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:250]}")

        # v26: scratchpad mode — show only brief summaries, not raw text
        if self.prompt_version == "v26":
            # Show only turn IDs and first 60 chars, forcing synthesis
            brief_lines = []
            for seg in sorted_segs[:16]:
                brief_lines.append(f"[Turn {seg.turn_id}]: {seg.text[:60]}...")
            context_section = (
                f"SCRATCHPAD — {len(covered_ids)} turns found so far:\n"
                + "\n".join(brief_lines)
            )
            if previous_cues:
                context_section += "\n\nPREVIOUS CUES:\n" + "\n".join(
                    f"- {c}" for c in previous_cues
                )
            return context_section

        # For v7+ prompts: simpler context without territory tracking emphasis
        if self.prompt_version in (
            "v7",
            "v8",
            "v9",
            "v10",
            "v11",
            "v12",
            "v13",
            "v14",
            "v15",
            "v16",
            "v17",
            "v18",
            "v19",
            "v20",
            "v21",
            "v22",
            "v23",
            "v24",
            "v25",
            "v28",
        ):
            context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + "\n".join(
                context_lines
            )
            # Also show what the latest hop found separately
            if new_segments and hop_number > 1:
                latest_lines = []
                for seg in sorted(new_segments, key=lambda s: s.turn_id)[:6]:
                    latest_lines.append(
                        f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:200]}"
                    )
                context_section += "\n\nMOST RECENTLY FOUND (last hop):\n" + "\n".join(
                    latest_lines
                )
        else:
            context_section = (
                f"TERRITORY COVERED (hop {hop_number}):\n"
                f"{territory_desc}\n\n"
                f"ALL RETRIEVED EXCERPTS (chronological order):\n"
                + "\n".join(context_lines)
            )

        if previous_cues:
            context_section += (
                "\n\nPREVIOUS CUES ALREADY TRIED (do NOT repeat or paraphrase):\n"
                + "\n".join(f"- {c}" for c in previous_cues)
            )

        return context_section

    def generate_cues(
        self,
        question: str,
        all_segments: list[Segment],
        new_segments: list[Segment],
        hop_number: int,
        previous_cues: list[str] | None = None,
        conv_length: int = 0,
        max_turn_id: int = 0,
    ) -> tuple[list[str], set[int]]:
        """Generate cues and optionally expansion targets.

        Returns (cues, expand_turn_ids). expand_turn_ids is non-empty only for
        v9 prompts that output EXPAND lines.
        """
        accumulated_versions = {
            "v5",
            "v6",
            "v7",
            "v8",
            "v9",
            "v10",
            "v11",
            "v12",
            "v13",
            "v14",
            "v15",
            "v16",
            "v17",
            "v18",
            "v19",
            "v20",
            "v21",
            "v22",
            "v23",
            "v24",
            "v25",
            "v26",
            "v28",
        }
        use_accumulated = self.prompt_version in accumulated_versions

        if use_accumulated:
            context_section = self._build_context_section_accumulated(
                all_segments,
                hop_number,
                previous_cues,
                new_segments,
            )
        else:
            context_section = self._build_context_section_legacy(
                all_segments,
                new_segments,
                hop_number,
                previous_cues,
            )

        format_kwargs = {
            "question": question,
            "context_section": context_section,
            "num_cues": self.num_cues,
        }
        if self.prompt_version in ("v5", "v6"):
            format_kwargs["conv_length"] = conv_length
            format_kwargs["max_turn_id"] = max_turn_id
        # Additional format args for specific prompt versions
        if "hop_number" in self.prompt_template:
            format_kwargs["hop_number"] = hop_number

        prompt = self.prompt_template.format(**format_kwargs)

        cached_response = self.llm_cache.get(self.cue_model, prompt)
        if cached_response is not None:
            return self._parse_cues_and_expansions(cached_response)

        response = self.client.chat.completions.create(
            model=self.cue_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(self.cue_model, prompt, text)
        return self._parse_cues_and_expansions(text)

    def _parse_cues_and_expansions(self, text: str) -> tuple[list[str], set[int]]:
        """Parse cues and optional EXPAND lines from LLM output.

        Ignores ASSESSMENT:, HYPOTHESIS:, MISSING: lines (used by v15/v16
        for self-monitoring and scratchpad but not needed for retrieval).
        """
        cues = []
        expand_ids: set[int] = set()
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    cues.append(cue)
            elif line.upper().startswith("EXPAND:"):
                expand_text = line[7:].strip()
                if expand_text.upper() != "NONE":
                    for part in expand_text.split(","):
                        part = part.strip()
                        try:
                            expand_ids.add(int(part))
                        except ValueError:
                            pass
            # Silently skip ASSESSMENT:, HYPOTHESIS:, MISSING: lines
        return cues, expand_ids

    def single_shot_retrieve(
        self,
        question: str,
        conversation_id: str,
        top_k: int = 20,
    ) -> RetrievalResult:
        query_embedding = self.embed_text(question)
        return self.store.search(
            query_embedding,
            top_k=top_k,
            conversation_id=conversation_id,
        )

    def _get_conversation_metadata(self, conversation_id: str) -> tuple[int, int]:
        """Return (num_segments, max_turn_id) for a conversation."""
        mask = self.store.conversation_ids == conversation_id
        conv_turn_ids = self.store.turn_ids[mask]
        return int(mask.sum()), int(conv_turn_ids.max()) if len(
            conv_turn_ids
        ) > 0 else 0

    def associative_retrieve(
        self,
        question: str,
        conversation_id: str,
        top_k_initial: int = 10,
    ) -> AssociativeResult:
        hops: list[HopResult] = []
        all_segments: list[Segment] = []
        all_turn_ids: set[int] = set()
        excluded_indices: set[int] = set()

        conv_length, max_turn_id = self._get_conversation_metadata(conversation_id)

        query_embedding = self.embed_text(question)
        hop0_result = self.store.search(
            query_embedding,
            top_k=top_k_initial,
            conversation_id=conversation_id,
        )
        new_turn_ids = {s.turn_id for s in hop0_result.segments}
        hops.append(
            HopResult(
                hop_number=0,
                cues=[question],
                retrieved=hop0_result,
                new_turn_ids=new_turn_ids,
            )
        )
        all_segments.extend(hop0_result.segments)
        all_turn_ids.update(new_turn_ids)
        excluded_indices.update(s.index for s in hop0_result.segments)

        all_previous_cues: list[str] = []
        last_hop_segments: list[Segment] = list(hop0_result.segments)

        for hop_num in range(1, self.max_hops + 1):
            cues, expand_targets = self.generate_cues(
                question,
                all_segments,
                last_hop_segments,
                hop_num,
                previous_cues=all_previous_cues or None,
                conv_length=conv_length,
                max_turn_id=max_turn_id,
            )
            all_previous_cues.extend(cues)
            if not cues:
                break

            hop_segments: list[Segment] = []
            hop_scores: list[float] = []

            for cue in cues:
                cue_embedding = self.embed_text(cue)
                result = self.store.search(
                    cue_embedding,
                    top_k=self.top_k_per_hop,
                    conversation_id=conversation_id,
                    exclude_indices=excluded_indices,
                )
                for seg, score in zip(result.segments, result.scores):
                    if seg.index not in excluded_indices:
                        hop_segments.append(seg)
                        hop_scores.append(score)
                        excluded_indices.add(seg.index)

            # Neighbor expansion
            num_expanded = 0
            if self.neighbor_radius > 0:
                neighbor_segments = []

                if self.prompt_version == "v9" and expand_targets:
                    # Selective expansion: only expand turns the model chose
                    segments_to_expand = [
                        s
                        for s in (all_segments + hop_segments)
                        if s.turn_id in expand_targets
                    ]
                else:
                    # Default: expand all segments from this hop
                    segments_to_expand = hop_segments

                num_expanded = len(segments_to_expand)
                for seg in segments_to_expand:
                    neighbors = self.store.get_neighbors(
                        seg,
                        radius=self.neighbor_radius,
                        exclude_indices=excluded_indices,
                    )
                    for n in neighbors:
                        if n.index not in excluded_indices:
                            neighbor_segments.append(n)
                            excluded_indices.add(n.index)
                hop_segments.extend(neighbor_segments)
                hop_scores.extend([0.0] * len(neighbor_segments))

            hop_new_ids = {s.turn_id for s in hop_segments} - all_turn_ids
            hops.append(
                HopResult(
                    hop_number=hop_num,
                    cues=cues,
                    retrieved=RetrievalResult(segments=hop_segments, scores=hop_scores),
                    new_turn_ids=hop_new_ids,
                    expand_targets=expand_targets or None,
                    num_expanded=num_expanded,
                )
            )
            all_segments.extend(hop_segments)
            all_turn_ids.update(hop_new_ids)
            last_hop_segments = hop_segments

            if not hop_new_ids:
                break

        return AssociativeResult(
            question=question,
            conversation_id=conversation_id,
            hops=hops,
            all_retrieved_segments=all_segments,
            all_retrieved_turn_ids=all_turn_ids,
        )

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()
