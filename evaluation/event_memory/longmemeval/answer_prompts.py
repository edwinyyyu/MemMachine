"""LongMemEval answer-generation prompt variants.

Two variants:

- ``mastra`` (default) — short prompt borrowed from Mastra's observational
  memory processor (the existing in-house default).
- ``mem0-bench`` — Mem0's elaborate 7-step `ANSWER_GENERATION_PROMPT` from
  https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/prompts.py
  with the same chain-of-thought scaffolding (``<mem_thinking>`` tags) and
  post-processing (strip thinking block, extract text after ``ANSWER:``).
  Default answerer model for this variant matches Mem0's run.py: ``gpt-5``.
"""

import re
from typing import Literal

from memmachine_server.episodic_memory.event_memory.data_types import Segment

AnswerVariant = Literal["mastra", "mem0-bench"]
ANSWER_VARIANTS: tuple[AnswerVariant, ...] = ("mastra", "mem0-bench")

# Default answerer model from Mem0's memory-benchmarks LongMemEval runner
# (`benchmarks/longmemeval/run.py`: `--answerer-model default="gpt-5"`).
MEM0_BENCH_DEFAULT_ANSWER_MODEL = "gpt-5"


# ===============================================================================
# mastra — existing in-house default
# ===============================================================================

# Parts of prompt borrowed from Mastra's OM.
# https://github.com/mastra-ai/mastra/blob/977b49e23d8b050a2c6a6a91c0aa38b28d6388ee/packages/memory/src/processors/observational-memory/observational-memory.ts#L312-L318
MASTRA_PROMPT = """
You are a helpful assistant with access to extensive conversation history.
When answering questions, carefully review the conversation history to identify and use any relevant user preferences, interests, or specific details they have mentioned.

<history>
{memories}
</history>

IMPORTANT: When responding, reference specific details from these observations. Do not give generic advice - personalize your response based on what you know about this user's experiences, preferences, and interests. If the user asks for recommendations, connect them to their past experiences mentioned above.

KNOWLEDGE UPDATES: When asked about current state (e.g., "where do I currently...", "what is my current..."), always prefer the MOST RECENT information. Observations include dates - if you see conflicting information, the newer observation supersedes the older one. Look for phrases like "will start", "is switching", "changed to", "moved to" as indicators that previous information has been updated.

PLANNED ACTIONS: If the user stated they planned to do something (e.g., "I'm going to...", "I'm looking forward to...", "I will...") and the date they planned to do it is now in the past (check the relative time like "3 weeks ago"), assume they completed the action unless there's evidence they didn't. For example, if someone said "I'll start my new diet on Monday" and that was 2 weeks ago, assume they started the diet.

Current date: {question_date}
Question: {question}
"""


# ===============================================================================
# mem0-bench — Mem0 memory-benchmarks unified answerer prompt
# ===============================================================================

# Verbatim from
# https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/prompts.py
# `ANSWER_GENERATION_PROMPT`.
MEM0_BENCH_PROMPT = """You are a personal assistant with access to memories from past conversations with a user. Answer the question using information from the memories below. Be direct and concise.

IMPORTANT: Today's date is {question_date}. All relative time expressions MUST be computed relative to this date.

IMPORTANT: If memories indicate the user wants to avoid something, your answer must NOT contain it — not as primary, secondary, or context.

IMPORTANT: If memories contain the numbers needed to compute the answer (ages to subtract, prices, dates to diff), DO the computation. NEVER abstain when the raw data exists — even scattered across different conversations.

IMPORTANT: Keep your responses short. No need to go into too much detail, no need to describe things at the lowest level. You can generally describe events and ideas abstractly.

IMPORTANT: Pay close attention to the EXACT entity in the question. If the question asks about a specific variant and memories only mention a DIFFERENT variant (e.g., "electric guitar" vs "acoustic guitar"), abstain — these are talking about different things!

IMPORTANT: For comparison/savings questions, BOTH costs must come from USER-stated facts (or user-relayed, e.g., "my friend said"). Do NOT use assistant-provided general info. If only one side has a user-stated cost, abstain.

IMPORTANT: If the query uses a specific but WRONG role/title/entity (e.g., asks about experience as a "Sales Manager" but memories say "Senior Sales Engineer"), do NOT answer as if they match — instead say you don't have the information! Always lean towards abstention in these cases! Do not mix up different role titles, they are not the same roles and you should say you don't have information.

Before answering, reason step-by-step inside <mem_thinking> tags:
- List every relevant memory; try to list all memories relevant to what the user wants to do! Eg. List memory of Payment management apps if query is about paying someone; list memory of travel management apps if query is about going somewhere.

- For counting: enumerate each item with date. Apply the question's EXACT verb/qualifier strictly (e.g., "LED" = leader only, "BAKED" = completed baking only, "RAISED" = total from events user participated in (include team/event totals), "COMPLETED writing" = each distinct finished piece). Count multiple items in a single memory separately. Do a SECOND full scan of all memories after initial count — items at positions 30-200 are commonly missed. Verify each item is a completed action (past tense), not a plan ("plans to", "intends to").
- For cross-topic computation: scan ALL memories for each needed fact independently — they're often in unrelated conversations. List: (a) what you need, (b) where each appears, (c) the computation.
- For temporal questions: identify dates, compute intervals from {question_date}
- CONTEXT CHECK: Before using a memory's value, verify it applies to the SAME context as the question. A wake-up time "while traveling" is NOT the same as a regular weekday wake-up time. A "general daily" schedule may conflict with a "specific weekday" schedule — always prefer the more specific memory that matches the question's context. List the context of each memory (weekday routine vs. travel vs. weekend vs. specific day) and only use values from the matching context.
- For time-bounded counting: compute the INCLUSIVE date window first, then check EVERY item's date. Err on inclusion for ambiguous dates.
- For "where is X": trace location chronologically through memories
- For suggestions: list (a) what user has/does, (b) what they avoid/dislike, (c) what they want to explore. Check every suggestion against (b) before including.
- State your conclusion

The user will only see text outside the <mem_thinking> tags.

Rules:

1. **Always try to answer**: If the topic appears in any memory — even indirectly — answer using what you have. Don't refuse for one missing detail.

2. **Most recent wins**: For conflicting values of the same fact, use the most recent memory. But: (a) memories about different people/contexts aren't conflicting; (b) for historical event dates, use the memory recorded closest to the event; (c) for current counts/scores/status, the latest value REPLACES all earlier ones — don't sum or average.

Similarly, when memories give two numbers for the same metric (e.g., "has 1,250 followers" and "close to 1,300 followers") on the same date, treat the HIGHER/UPDATED value as current — "close to 1,300" means the count has grown from 1,250 to approximately 1,300.

3. **Time-bounded questions**: Compute the date window from {question_date}. Show date arithmetic in <mem_thinking>. Scan EVERY memory for events in range. "Last weekend" is imprecise — could mean up to 10 days ago as people sometimes mean weekend before the latest one. "Last 3 months" can include boundary days of the 4th month back.

"Last month" includes the current month so far as well as the previous month. Eg. "last month" in Late May includes all of April. If the literal window yields nothing, check the immediately preceding period.

4. **Temporal reference points**: "How many days ago did X when Y happened" — compute interval between X and Y, NOT between X and today.

5. **Counting and ordering**: Scan ALL memories first to last. Build a numbered list in <mem_thinking> with date and position. Deduplicate by matching dates/descriptions. Count items in a single memory separately.
Any addition to a list on the same day as a stated count is already included in the count

When asked to count all instances of an event *before* a specific one, obviously don't include the specific one in the count. Eg. "how many restaurants did i visit before eating at Pizza Hut?". Obviously don't include Pizza Hut in the count

6. **Use only the memories**: Don't invent numbers, prices, or addresses.

7. **When to abstain**: Say "The information provided is not enough" when:
   - The topic is genuinely unmentioned

- The question asks about a specific event that doesn't exist, even if a related topic does

- IMPORTANT: If the query uses a specific but WRONG role/title/entity (e.g., asks about experience as a "Sales Manager" but memories say "Senior Sales Engineer"), do NOT answer as if they match — instead say you don't have the information! Always lean towards abstention in these cases! Do not mix up different role titles, they are not the same roles and you should say you don't have information.

   - For comparison/ordering, BOTH items must be present as completed events
   If query asks to compare timings of two tasks and one of them did not even happen, abstain.
   Before abstaining, do a keyword scan of ALL memories (they're chronological, not relevance-sorted — check positions 1-200). Only abstain if NO keywords match.
   EXCEPTIONS: For suggestion questions, don't abstain for lack of real-time info — recommend based on known preferences. If you lack exact brand but have the store, output the store.

8. **Yes/no and comparison**: "Did I ever do X?" with no matching memory = "No." For comparisons, find both values across all memories and compare directly.

9. **Actions vs intentions**: Use the date of actual execution, not the plan date. "Decided to" or "took X for servicing" = action initiated. Only treat as plan if explicit future-tense ("plans to", "will"). A plan with a specified date and no update = assume completed on that date. If a later memory confirms execution, use the execution date — it supersedes the earlier plan.

When a query asks: "when I decided to do X", it means they are asking when X was actually done.

10. **User facts vs assistant advice**: "User..." = actual experience. "Assistant..." = advice. Prefer user-stated facts for personal questions. Don't convert currencies unless user stated the conversion.

11. **Connect memories across topics**: Facts needed for computation are often in unrelated conversations (age in travel advice + relative's age in birthday discussion; cashback rate in membership talk + purchase amount in expense tracking). Search ALL memories for each fact independently.

12. **Personalization**: For suggestions/recommendations:
   - Prioritize personal preferences over informational content
   - Apply known preferences to new contexts — don't abstain for unfamiliar destinations
   - Acknowledge prior work before suggesting next steps
   - Respect anti-preferences — check every suggestion against known dislikes
   - Reference existing tools owned, not to acquire
   - Lead with personalization, don't pad with generic alternatives
   - Suggest similar things to the user as their habits. Eg. Logging basketball scores in a app they do usually. Eg. Adding travel logs to a travel logging app they use usually.
   - IMPORTANT: Scan ALL top memories for user-owned tools, apps, and resources relevant to the question. If the user has a travel card (Suica), a trip organizer app (TripIt), a budgeting tool, etc., mention ALL of them — not just the most obvious one. Do a SECOND pass of the top 30 memories specifically looking for apps, tools, and resources the user has mentioned owning or using.

13. **Reasonable deduction**:
- Infer from patterns
IMPORTANT: Assume that similar items referenced in the same sentence have the same type.
Eg. "User ate lunch, which was the third meal with this chicken fajitas". This means the other meals with these chicken fajitas were lunch meals too, should be treated as explicit lunches.

14. IMPORTANT: If two pieces of memory directly contradict each other (not just an update, a direct contradiction), then assume that the memory that was created later is true. Doesn't matter if a different one "appears" more reliable. If on the same day, trust the one at a later time.

- Chronological actions:
If the user is watching the 11th episode of a series is watching it normally, assume they have completed the earlier 10 too.

- If you lack a name but have a description, answer with the description.

**Memory grouping rules**: Memories under the same date heading are from the same conversation.
- A count + "added X items" on the SAME date = count already includes them
- "Aims to beat X" = X is the current value
- "Previous" = the value superseded by a more recent one
- Events described as just completed ("attended", "went to", "just got back from", "completed") = happened on/near that date. Undated actions = assume the event happened on the memory's date.

# Misc Rules
- Count class projects too when asked about users' projects. Class projects = projects.
- Most old (Eg. ancestral, vintage, heritage) items count as antiques too!
- If you don't have chords for a song (but have notes), output the notes. Song notes count as chord progressions.
- Starting a *diorama project* (eg. diorama work, working on terrain) EXPLICITLY COUNTS AS working on that model kit; these are equivalent! Always count such items.
- Running into someone at a coffee shop and exchanging numbers DOES NOT count as meeting them; lunch meetings do count.
- Potlucks/feasts/birthday parties count as dinner parties (BBQ doesn't).
- chandelier counts as jewelry
- Always assume birthdays cleanly follow years. Ie. User was 22 in 2022; they will be 23 in 2023.
- "scratch grains" count as "new layer feed", always include them when interpreting "new layer feed"

Memories (sorted newest-first, grouped by date):
{memories}

Today's Date: {question_date}
Question: {question}

IMPORTANT: You MUST provide your full thinking in <mem_thinking> tags BEFORE giving your answer.; Reasoning and answer:"""


# ===============================================================================
# Memory formatting
# ===============================================================================


def _segment_text(segment: Segment) -> str:
    """Return the segment's text content with the producer prefix, if any."""
    text_attr = getattr(segment.block, "text", None)
    text = text_attr if isinstance(text_attr, str) else ""
    producer = getattr(segment.context, "producer", None)
    if producer:
        return f"{producer}: {text}"
    return text


def _format_memories_mem0_bench(segments: list[Segment]) -> str:
    """Group segments by date heading, newest first, matching Mem0's format."""
    if not segments:
        return "(No relevant memories found)"

    # Mem0 says "sorted newest-first" — sort by timestamp descending.
    sorted_segments = sorted(segments, key=lambda s: s.timestamp, reverse=True)
    lines: list[str] = []
    current_date: str | None = None
    for segment in sorted_segments:
        date_str = segment.timestamp.strftime("%A, %B %d, %Y")
        text = _segment_text(segment).strip()
        if not text:
            continue
        if date_str != current_date:
            current_date = date_str
            lines.append(f"\n--- {date_str} ---")
        lines.append(f"- {text}")
    return "\n".join(lines).strip()


# ===============================================================================
# Prompt build & post-processing
# ===============================================================================


def build_prompt(
    *,
    variant: AnswerVariant,
    question: str,
    question_date: str,
    memories_string: str,
    segments: list[Segment],
) -> str:
    """Render the answerer prompt for one item.

    `memories_string` is the EventMemory-formatted context (used for the
    `mastra` variant). `segments` is the underlying segment list (used by
    `mem0-bench` to re-format with date headings).
    """
    if variant == "mastra":
        return MASTRA_PROMPT.format(
            memories=memories_string,
            question_date=question_date,
            question=question,
        )
    return MEM0_BENCH_PROMPT.format(
        memories=_format_memories_mem0_bench(segments),
        question_date=question_date,
        question=question,
    )


_MEM_THINKING_RE = re.compile(
    r"[<\[]mem_thinking[>\]].*?[<\[]/mem_thinking[>\]]", re.DOTALL
)


def postprocess_answer(raw: str, *, variant: AnswerVariant) -> str:
    """Strip Mem0's chain-of-thought and extract the final answer."""
    if variant == "mastra":
        return raw.strip()
    cleaned = _MEM_THINKING_RE.sub("", raw).strip()
    if "ANSWER:" in cleaned:
        cleaned = cleaned.rsplit("ANSWER:", 1)[-1].strip()
    return cleaned
