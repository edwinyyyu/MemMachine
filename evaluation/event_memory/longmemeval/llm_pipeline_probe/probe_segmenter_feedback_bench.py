"""Targeted test bench for segmenter feedback iterations.

Three feedback areas:
  F1 OPINION-VS-FILLER. "I love/hate X" can be genuine opinion (with
     elaboration) OR pure solidarity scaffolding (no follow-up). Current
     rule 2 drops "polite restatement" but doesn't address love/hate filler;
     could miss genuine opinions OR over-keep flatter solidarity tokens.
  F2 SCAFFOLDING DIVERSITY. Current rule 2 examples are all user-assistant
     ("Hi!", "What a great question!"); insufficient for peer dialog, group
     chat, agent logs, email-style scaffolding.
  F3 RULE 3 ENUMERATION. Current "keep" list (entity, place, person, brand,
     work, date, price, preference, plan, decision, factual claim,
     eccentric phrasing). May miss: measurements (bpm, mph), relationships
     ("my therapist"), identifiers (PR #1289), emotion states ("anxious"),
     constraints/policies ("can't ship without review").

Methodology: address ONE feedback at a time. For each iteration:
  1. Draft a candidate prompt edit targeting that feedback only.
  2. Run baseline AND candidate on the full bench.
  3. Auto-check must_keep/must_drop substrings.
  4. Commit only if target cases improve AND no regression elsewhere.

Run:
    uv run python probe_segmenter_feedback_bench.py
    uv run python probe_segmenter_feedback_bench.py --prompt v2
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass, field

import openai
from dotenv import load_dotenv
from probe_segmenter_F_natural import (
    call,
)

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


# --------------------------------------------------------------------------
# Test cases — keyed by feedback area
# --------------------------------------------------------------------------


@dataclass
class TestCase:
    id: str
    area: str  # "F1" / "F2" / "F3"
    description: str
    passage: str
    must_keep: list[str] = field(default_factory=list)  # substrings ⊆ some seg
    must_drop: list[str] = field(default_factory=list)  # substrings ⊈ all segs


# F1 OPINION VS FILLER ------------------------------------------------------

F1_CASES: list[TestCase] = [
    TestCase(
        id="F1-genuine-with-elaboration",
        area="F1",
        description="'I love jazz' followed by behavioral evidence — should KEEP.",
        passage=(
            "I love jazz. Last week I went to Smalls on West 10th and "
            "heard Bill Frisell. Then last night I put Coltrane's A Love "
            "Supreme on for the hundredth time."
        ),
        must_keep=[
            "I love jazz",
            "Smalls",
            "Bill Frisell",
            "Coltrane",
            "A Love Supreme",
        ],
        must_drop=[],
    ),
    TestCase(
        id="F1-filler-reply",
        area="F1",
        description="'I love X!' as agreement reply — affirmative reaction, KEEP for reconstruction.",
        passage=(
            "Hey what's a good restaurant near Union Square? Try Brindle "
            "Room — it's great for steak. I love brindle room!"
        ),
        # Under segment-as-reconstruction: "I love brindle room!" is an
        # opinion/affirmation, not pure receipt acknowledgment. Keep.
        must_keep=["Brindle Room", "Union Square"],
        must_drop=[],
    ),
    TestCase(
        id="F1-short-negative-opinion",
        area="F1",
        description="Short but specific negative opinion — should KEEP.",
        passage="Honestly, I think Sleep is overrated.",
        must_keep=["Sleep", "overrated"],
        must_drop=[],
    ),
    TestCase(
        id="F1-pure-agent-support",
        area="F1",
        description="'I love your reasoning' — appreciation/agreement, KEEP for reconstruction.",
        passage=(
            "Should I buy the M2 MacBook? It depends on your workload — "
            "the M2 is solid for development. I love your reasoning."
        ),
        # Under segment-as-reconstruction: appreciation conveys user's
        # reaction to the answer. Not pure receipt acknowledgment. Keep.
        must_keep=["M2 MacBook"],
        must_drop=[],
    ),
    TestCase(
        id="F1-opinion-plus-facts",
        area="F1",
        description="Negative opinion bracketed by specific facts — should KEEP.",
        passage=(
            "I hated the dentist visit. They charged $400 and didn't even fix the chip."
        ),
        must_keep=["hated", "dentist", "$400", "chip"],
        must_drop=[],
    ),
]

# F2 SCAFFOLDING DIVERSITY ---------------------------------------------------

F2_CASES: list[TestCase] = [
    TestCase(
        id="F2-peer-dialog",
        area="F2",
        description="Casual peer dialog with greeting and pivot fillers.",
        passage=(
            "Hey, that movie was insane! Did you see how they did the "
            "time loop? Anyway, want to grab dinner Saturday?"
        ),
        must_keep=["movie was insane", "time loop", "dinner Saturday"],
        # "Hey, " and "Anyway, " are conversational fillers — but they're
        # tightly fused with the substantive content. Acceptable if model
        # widens to include them; what we really care about is no separate
        # filler-only segment.
        must_drop=[],
    ),
    TestCase(
        id="F2-agent-log",
        area="F2",
        description="Machine log style — no scaffolding to drop, keep everything.",
        passage=(
            "Triggered alert: deploy to staging failed at 14:32. Retrying "
            "in 5 minutes. Notifying #infra-on-call. Pipeline run ID: 8127."
        ),
        must_keep=[
            "deploy to staging failed",
            "14:32",
            "5 minutes",
            "#infra-on-call",
            "8127",
        ],
        must_drop=[],
    ),
    TestCase(
        id="F2-group-chat",
        area="F2",
        description="Affirmative reaction + substantive update — both KEEP.",
        passage=(
            "omg yes!! Mark just shared the slides from his Q3 review. "
            "Link: q3-2025.com/slides. tl;dr he's leaving for OpenAI next "
            "month."
        ),
        # Under segment-as-reconstruction: "omg yes!!" contains "yes" —
        # an affirmative reaction conveying excitement. Keep.
        must_keep=["Mark", "Q3 review", "q3-2025.com/slides", "OpenAI", "next month"],
        must_drop=[],
    ),
    TestCase(
        id="F2-email-scaffolding",
        area="F2",
        description="Email envelope phrases (greetings / generic closers) "
        "with no specifics should be dropped. Note: 'please find attached "
        "the proposal' has the referent 'the proposal' -- content, not "
        "pure envelope. 'Thanks, Sarah' is borderline (useless when sender "
        "metadata is on the event, but useful if copy-pasted elsewhere or "
        "the only place a nickname appears), so it is not in must_drop.",
        passage=(
            "Hi team, As discussed, please find attached the proposal. "
            "Project budget is $45k. Deadline is March 30. Let me know if "
            "any questions. Thanks, Sarah"
        ),
        must_keep=["budget", "$45k", "March 30"],
        must_drop=["Hi team", "Let me know if any questions"],
    ),
]

# F3 RULE 3 ENUMERATION COMPLETENESS ----------------------------------------

F3_CASES: list[TestCase] = [
    TestCase(
        id="F3-measurement",
        area="F3",
        description="Physiological measurement (not 'price' or 'date').",
        passage=(
            "My resting heart rate dropped from 72 to 58 bpm after I started running."
        ),
        must_keep=["72", "58 bpm", "running"],
        must_drop=[],
    ),
    TestCase(
        id="F3-relationship",
        area="F3",
        description="Personal-relationship role (not strictly 'person' alone).",
        passage="Sarah is my therapist.",
        must_keep=["Sarah", "therapist"],
        must_drop=[],
    ),
    TestCase(
        id="F3-identifier",
        area="F3",
        description="Issue/PR identifier — not in 'brand' or 'work'.",
        passage="PR #1289 fixed the deadlock in the connection pool.",
        must_keep=["PR #1289", "deadlock", "connection pool"],
        must_drop=[],
    ),
    TestCase(
        id="F3-emotion-state",
        area="F3",
        description="Persistent emotional state tied to an event.",
        passage=("I've been anxious about the move ever since the lease started."),
        must_keep=["anxious", "move", "lease"],
        must_drop=[],
    ),
    TestCase(
        id="F3-policy-constraint",
        area="F3",
        description="Workflow constraint / policy rule.",
        passage="We can't ship before legal review per company policy.",
        must_keep=["legal review", "company policy"],
        must_drop=[],
    ),
]

ALL_CASES: list[TestCase] = F1_CASES + F2_CASES + F3_CASES


# --------------------------------------------------------------------------
# Prompts under test
# --------------------------------------------------------------------------

# v1 = original shipped segmenter prompt (frozen here for the bench; the
# production file probe_segmenter_F_natural.py may track a later version).
PROMPT_V1_BASELINE = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract. The only edits \
allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP generic scaffolding by simply not including it in any segment: \
"Hi!", "What a great question!", "I hope this helps!", "Let me know if \
you have any other questions", a polite restatement of what the other \
party just asked.
  3. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim, plus eccentric \
or distinctive phrasing.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  6. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), widen the quote to start where the referent \
is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


# v9 — generalize filler vs real-content distinction on top of v8.
# v8 had two issues: (a) a special clause for love/hate exclamations
# was over-specialized — the rule should generalize; (b) a heterogeneous
# enumeration "no examples, behavior, prices, dates, or other specifics"
# mixed general and specific terms. v9 reframes rule 2 around a single
# principle: "does the utterance contribute information someone would
# want to recall?" — applied uniformly, no love/hate special case.
PROMPT_V9_GENERALIZED_FILLER = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract. The only edits \
allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP content whose role is conversational framing rather than \
substance: standalone greetings ("Hi!", "Hi team"), sign-offs ("Thanks, \
Sarah", "Best regards"), polite restatements of what the other party \
just asked, envelope phrases that gesture at content without conveying \
it ("As discussed,", "please find attached", "Let me know if any \
questions", "What a great question!", "I hope this helps!"), chat \
reactions ("omg yes!!", "lol", "fwiw"), reactive filler ("I love \
that!", "great point", "I love your reasoning", "sounds great!"). The \
test for any utterance: does it contribute information someone would \
want to recall — a fact, name, claim, decision, plan, observation, or \
opinion with backing? If yes, KEEP. If it only echoes prior content or \
expresses pure approval/disapproval without new substance, DROP. An \
utterance that begins with a greeting or softener but carries \
substantive content is KEPT ("Hey, what's the deal with X?" contains \
the question — drop only the leading greeting, not the content).
  3. KEEP concrete particulars — anything specific to this passage \
that a future reader would want to recall. Names, places, dates, \
numbers, identifiers, decisions, plans, preferences, relationships, \
emotional states tied to events, constraints, and distinctive phrasing \
all qualify. Drop generic abstractions or stock phrases that would fit \
many situations.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  6. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), widen the quote to start where the referent \
is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


# v8 — F3 reframe on top of v7: replace rule 3's enumerated list with
# a principle (concrete particulars vs generic abstractions). The
# enumeration in v1/v7 currently passes all 5 F3 cases — this version
# tests whether the model retains that recall when given a principle
# with suggestive but non-exhaustive examples. Also more robust to
# edge cases not in the test bench.
PROMPT_V8_F3_PRINCIPLE = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract. The only edits \
allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP content whose role is conversational framing rather than \
substance. Common forms: standalone greetings ("Hi!", "Hi team"), \
sign-offs ("Thanks, Sarah", "Best regards"), polite restatements of \
what the other party just asked, envelope phrases that announce or \
gesture-toward content without conveying it ("As discussed,", "please \
find attached", "Let me know if any questions", "What a great \
question!", "I hope this helps!"), chat reactions ("omg yes!!", "lol", \
"fwiw"), AND reactive filler ("I love that!", "great point", "I love \
your reasoning", "sounds great!") that echoes the prior turn without \
adding new specifics. A substantive utterance is KEPT even when it \
begins with a greeting or softener word ("Hey, what's the deal with \
X?" carries the question — drop only the leading greeting, not the \
content). Brief love/hate exclamations are reactive filler when the \
passage offers no elaboration on what they react to — no examples, \
behavior, prices, dates, or other specifics tied to it. KEEP love/hate \
statements that come with elaboration ("I love jazz" followed by \
examples; "I hated the dentist visit. They charged $400.") or that the \
speaker frames as a position ("I think Sleep is overrated").
  3. KEEP concrete particulars — anything specific to this passage \
that a future reader would want to recall. Names, places, dates, \
numbers, identifiers, decisions, plans, preferences, relationships, \
emotional states tied to events, constraints, and distinctive phrasing \
all qualify. Drop generic abstractions or stock phrases that would fit \
many situations.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  6. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), widen the quote to start where the referent \
is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


# v7 — F2 fix v3: remove "Hey," from greetings list (model was treating
# any utterance starting with "Hey" as filler, dropping substantive
# questions). Frame envelope-phrase category by purpose rather than
# email-specific idioms.
PROMPT_V7_F2_FIX_V3 = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract. The only edits \
allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP content whose role is conversational framing rather than \
substance. Common forms: standalone greetings ("Hi!", "Hi team"), \
sign-offs ("Thanks, Sarah", "Best regards"), polite restatements of \
what the other party just asked, envelope phrases that announce or \
gesture-toward content without conveying it ("As discussed,", "please \
find attached", "Let me know if any questions", "What a great \
question!", "I hope this helps!"), chat reactions ("omg yes!!", "lol", \
"fwiw"), AND reactive filler ("I love that!", "great point", "I love \
your reasoning", "sounds great!") that echoes the prior turn without \
adding new specifics. A substantive utterance is KEPT even when it \
begins with a greeting or softener word ("Hey, what's the deal with \
X?" carries the question — drop only the leading greeting, not the \
content). Brief love/hate exclamations are reactive filler when the \
passage offers no elaboration on what they react to — no examples, \
behavior, prices, dates, or other specifics tied to it. KEEP love/hate \
statements that come with elaboration ("I love jazz" followed by \
examples; "I hated the dentist visit. They charged $400.") or that the \
speaker frames as a position ("I think Sleep is overrated").
  3. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim, plus eccentric \
or distinctive phrasing.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  6. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), widen the quote to start where the referent \
is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


# v6 — F2 fix v2: principle-first with representative examples instead of
# domain-specific (email) idioms. v5 added 'as discussed' / 'please find
# attached' which inadvertently caused the model to treat structurally
# similar recommendations ('Try X — it's great for Y') as scaffolding,
# regressing F1-filler-reply keep 3/3→1/3. v6 keeps the diverse-context
# coverage via category names rather than long example lists.
PROMPT_V6_F2_FIX_V2 = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract. The only edits \
allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP content whose role is conversational framing rather than \
substance. Common forms: greetings ("Hi!", "Hi team", "Hey,"), \
sign-offs ("Thanks, Sarah", "Best regards"), polite restatements of \
what the other party just asked, envelope phrases ("Let me know if any \
questions", "What a great question!", "I hope this helps!"), chat \
reactions ("omg yes!!", "lol", "fwiw"), AND reactive filler ("I love \
that!", "great point", "I love your reasoning", "sounds great!") that \
echoes the prior turn without adding new specifics. Brief love/hate \
exclamations are reactive filler when the passage offers no elaboration \
on what they react to — no examples, behavior, prices, dates, or other \
specifics tied to it. KEEP love/hate statements that come with \
elaboration ("I love jazz" followed by examples; "I hated the dentist \
visit. They charged $400.") or that the speaker frames as a position \
("I think Sleep is overrated").
  3. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim, plus eccentric \
or distinctive phrasing.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  6. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), widen the quote to start where the referent \
is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


# v5 — F2 fix on top of v4: broaden scaffolding examples beyond
# user-assistant chat to cover email envelopes (Hi team / Thanks, [name] /
# As discussed / please find attached / Let me know if any questions),
# chat reactions (omg yes!! / lol / fwiw), and peer-dialog openers.
# v4's F1 fix preserved verbatim.
PROMPT_V5_F2_FIX = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract. The only edits \
allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP generic scaffolding by simply not including it in any segment. \
This includes greetings ("Hi!", "Hi team", "Hey,"), sign-offs ("Thanks, \
Sarah", "Best regards"), polite restatements of what the other party \
just asked, envelope phrases ("As discussed,", "please find attached", \
"What a great question!", "I hope this helps!", "Let me know if any \
questions"), chat reactions ("omg yes!!", "lol", "fwiw"), AND reactive \
filler ("I love that!", "great point", "I love your reasoning", \
"sounds great!") that echoes the prior turn without adding new \
specifics. Brief love/hate exclamations are reactive filler when the \
passage offers no elaboration on what they react to — no examples, \
behavior, prices, dates, or other specifics tied to it. KEEP love/hate \
statements that come with elaboration ("I love jazz" followed by \
examples; "I hated the dentist visit. They charged $400.") or that the \
speaker frames as a position ("I think Sleep is overrated").
  3. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim, plus eccentric \
or distinctive phrasing.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  6. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), widen the quote to start where the referent \
is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


# v4 — F1 fix v3: elaboration-as-discriminator. v3 over-specified the
# referent-borrow pattern and regressed both pure-agent-support and
# filler-reply keep. v4 reframes around the user's actual cue: elaboration.
# Drop love/hate when there's no follow-on specifics; keep when paired
# with elaboration or speaker-introduced.
PROMPT_V4_F1_FIX_V3 = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract. The only edits \
allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP generic scaffolding by simply not including it in any segment: \
"Hi!", "What a great question!", "I hope this helps!", "Let me know if \
you have any other questions", a polite restatement of what the other \
party just asked, AND reactive filler ("I love that!", "great point", \
"I love your reasoning", "sounds great!") that echoes the prior turn \
without adding new specifics. Brief love/hate exclamations are reactive \
filler when the passage offers no elaboration on what they react to — \
no examples, behavior, prices, dates, or other specifics tied to it. \
KEEP love/hate statements that come with elaboration ("I love jazz" \
followed by examples; "I hated the dentist visit. They charged $400.") \
or that the speaker frames as a position ("I think Sleep is \
overrated").
  3. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim, plus eccentric \
or distinctive phrasing.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  6. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), widen the quote to start where the referent \
is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


# v3 — F1 fix v2: also catch referent-borrowed filler ("I love [X]" where X
# was just named by another party). v2 fixed pure-agent-support 0/3→2/3 but
# not filler-reply 0/3; model parses "I love brindle room" as a preference
# because of the named entity. v3 makes the borrow pattern explicit.
PROMPT_V3_F1_FIX_V2 = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract. The only edits \
allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP generic scaffolding by simply not including it in any segment: \
"Hi!", "What a great question!", "I hope this helps!", "Let me know if \
you have any other questions", a polite restatement of what the other \
party just asked, AND reactive filler ("I love that!", "great point", \
"I love your reasoning") that echoes the prior turn without adding new \
specifics. Bare "I love/hate [X]!" or "[X] is great/awful!" is reactive \
filler when [X] was just named by another party and the speaker offers \
no new fact, evidence, or commitment beyond the reaction. KEEP opinion \
statements that bring their own subject or elaboration ("I love jazz" \
followed by examples; "I think Sleep is overrated"; "I hated the \
dentist visit. They charged $400.").
  3. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim, plus eccentric \
or distinctive phrasing.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  6. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), widen the quote to start where the referent \
is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


# v2 — F1 fix: distinguish reactive filler from genuine opinion.
# Rule 2 only; rules 1, 3, 4, 5, 6 unchanged. Other phrasing identical to v1.
PROMPT_V2_F1_FIX = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract. The only edits \
allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP generic scaffolding by simply not including it in any segment: \
"Hi!", "What a great question!", "I hope this helps!", "Let me know if \
you have any other questions", a polite restatement of what the other \
party just asked, AND reactive filler ("I love that!", "great point", \
"I love your reasoning") that echoes the prior turn without adding new \
specifics. KEEP opinion statements that bring their own subject or \
elaboration ("I love jazz" followed by examples; "I think Sleep is \
overrated").
  3. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim, plus eccentric \
or distinctive phrasing.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  6. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), widen the quote to start where the referent \
is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


def _ci_in(needle: str, haystack: str) -> bool:
    """Case-insensitive substring check (the segmenter preserves case
    verbatim, but case-insensitive is more forgiving for sanity checks)."""
    return needle.lower() in haystack.lower()


def evaluate(case: TestCase, segs: list[str]) -> dict:
    joined = " || ".join(segs)
    keep_hits = [t for t in case.must_keep if _ci_in(t, joined)]
    keep_miss = [t for t in case.must_keep if not _ci_in(t, joined)]
    drop_violations = [t for t in case.must_drop if _ci_in(t, joined)]
    drop_clean = [t for t in case.must_drop if not _ci_in(t, joined)]
    return {
        "n_segs": len(segs),
        "keep_hits": keep_hits,
        "keep_miss": keep_miss,
        "drop_violations": drop_violations,
        "drop_clean": drop_clean,
        "ok_keep": len(keep_miss) == 0,
        "ok_drop": len(drop_violations) == 0,
    }


def fmt_case(case: TestCase, segs: list[str], ev: dict) -> str:
    verdict_keep = "✓" if ev["ok_keep"] else "✗"
    verdict_drop = "✓" if ev["ok_drop"] else "✗"
    lines = [
        f"--- [{case.area}] {case.id} ---",
        f"  desc: {case.description}",
        f"  n_segs: {ev['n_segs']}  keep:{verdict_keep}  drop:{verdict_drop}",
    ]
    for i, s in enumerate(segs):
        lines.append(f"    seg[{i}]: {s[:200]}{'…' if len(s) > 200 else ''}")
    if ev["keep_miss"]:
        lines.append(f"  MISSED (should have been in some seg): {ev['keep_miss']}")
    if ev["drop_violations"]:
        lines.append(f"  KEPT BUT SHOULD HAVE DROPPED: {ev['drop_violations']}")
    return "\n".join(lines)


async def run_bench(
    client, model: str, reasoning: str, prompt: str, cases: list[TestCase]
):
    sem = asyncio.Semaphore(8)

    async def go(case: TestCase):
        async with sem:
            p = prompt.format(passage=case.passage)
            segs = await call(client, model, p, reasoning)
            ev = evaluate(case, segs)
            return case, segs, ev

    results = await asyncio.gather(*(go(c) for c in cases))
    return results


def summary(results, label: str) -> None:
    print(f"\n## {label} summary")
    for area in ["F1", "F2", "F3"]:
        area_results = [(c, s, e) for c, s, e in results if c.area == area]
        ok_keep = sum(1 for _, _, e in area_results if e["ok_keep"])
        ok_drop = sum(1 for _, _, e in area_results if e["ok_drop"])
        total = len(area_results)
        print(f"  {area}: keep {ok_keep}/{total}  drop {ok_drop}/{total}")
    total_keep = sum(1 for _, _, e in results if e["ok_keep"])
    total_drop = sum(1 for _, _, e in results if e["ok_drop"])
    n = len(results)
    print(f"  TOTAL: keep {total_keep}/{n}  drop {total_drop}/{n}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument(
        "--prompt",
        default="v1",
        choices=[
            "v1",
            "v2",
            "v3",
            "v4",
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
            "v27",
            "v28",
            "v29",
            "v30",
            "v31",
            "v32",
            "v33",
        ],
    )
    parser.add_argument("--reps", type=int, default=1)
    args = parser.parse_args()

    from probe_segmenter_F_natural_v10 import PROMPT_F_NATURAL_V10
    from probe_segmenter_F_natural_v11 import PROMPT_F_NATURAL_V11
    from probe_segmenter_F_natural_v12 import PROMPT_F_NATURAL_V12
    from probe_segmenter_F_natural_v13 import PROMPT_F_NATURAL_V13
    from probe_segmenter_F_natural_v14 import PROMPT_F_NATURAL_V14
    from probe_segmenter_F_natural_v15 import PROMPT_F_NATURAL_V15
    from probe_segmenter_F_natural_v16 import PROMPT_F_NATURAL_V16
    from probe_segmenter_F_natural_v17 import PROMPT_F_NATURAL_V17
    from probe_segmenter_F_natural_v18 import PROMPT_F_NATURAL_V18
    from probe_segmenter_F_natural_v19 import PROMPT_F_NATURAL_V19
    from probe_segmenter_F_natural_v20 import PROMPT_F_NATURAL_V20
    from probe_segmenter_F_natural_v21 import PROMPT_F_NATURAL_V21
    from probe_segmenter_F_natural_v22 import PROMPT_F_NATURAL_V22
    from probe_segmenter_F_natural_v23 import PROMPT_F_NATURAL_V23
    from probe_segmenter_F_natural_v24 import PROMPT_F_NATURAL_V24
    from probe_segmenter_F_natural_v25 import PROMPT_F_NATURAL_V25
    from probe_segmenter_F_natural_v26 import PROMPT_F_NATURAL_V26
    from probe_segmenter_F_natural_v27 import PROMPT_F_NATURAL_V27
    from probe_segmenter_F_natural_v28 import PROMPT_F_NATURAL_V28
    from probe_segmenter_F_natural_v29 import PROMPT_F_NATURAL_V29
    from probe_segmenter_F_natural_v30 import PROMPT_F_NATURAL_V30
    from probe_segmenter_F_natural_v31 import PROMPT_F_NATURAL_V31
    from probe_segmenter_F_natural_v32 import PROMPT_F_NATURAL_V32
    from probe_segmenter_F_natural_v33 import PROMPT_F_NATURAL_V33

    prompts = {
        "v1": PROMPT_V1_BASELINE,
        "v2": PROMPT_V2_F1_FIX,
        "v3": PROMPT_V3_F1_FIX_V2,
        "v4": PROMPT_V4_F1_FIX_V3,
        "v5": PROMPT_V5_F2_FIX,
        "v6": PROMPT_V6_F2_FIX_V2,
        "v7": PROMPT_V7_F2_FIX_V3,
        "v8": PROMPT_V8_F3_PRINCIPLE,
        "v9": PROMPT_V9_GENERALIZED_FILLER,
        "v10": PROMPT_F_NATURAL_V10,
        "v11": PROMPT_F_NATURAL_V11,
        "v12": PROMPT_F_NATURAL_V12,
        "v13": PROMPT_F_NATURAL_V13,
        "v14": PROMPT_F_NATURAL_V14,
        "v15": PROMPT_F_NATURAL_V15,
        "v16": PROMPT_F_NATURAL_V16,
        "v17": PROMPT_F_NATURAL_V17,
        "v18": PROMPT_F_NATURAL_V18,
        "v19": PROMPT_F_NATURAL_V19,
        "v20": PROMPT_F_NATURAL_V20,
        "v21": PROMPT_F_NATURAL_V21,
        "v22": PROMPT_F_NATURAL_V22,
        "v23": PROMPT_F_NATURAL_V23,
        "v24": PROMPT_F_NATURAL_V24,
        "v25": PROMPT_F_NATURAL_V25,
        "v26": PROMPT_F_NATURAL_V26,
        "v27": PROMPT_F_NATURAL_V27,
        "v28": PROMPT_F_NATURAL_V28,
        "v29": PROMPT_F_NATURAL_V29,
        "v30": PROMPT_F_NATURAL_V30,
        "v31": PROMPT_F_NATURAL_V31,
        "v32": PROMPT_F_NATURAL_V32,
        "v33": PROMPT_F_NATURAL_V33,
    }
    prompt = prompts[args.prompt]
    print(
        f"# Segmenter test bench — model={args.model} reasoning={args.reasoning} prompt={args.prompt}"
    )
    print(f"# prompt length: {len(prompt)} chars / {len(prompt.split())} words")

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if args.reps == 1:
        results = await run_bench(client, args.model, args.reasoning, prompt, ALL_CASES)
        for case, segs, ev in results:
            print()
            print(fmt_case(case, segs, ev))
        summary(results, args.prompt)
    else:
        all_runs = []
        for rep in range(args.reps):
            print(f"\n## REP {rep + 1}/{args.reps}")
            results = await run_bench(
                client, args.model, args.reasoning, prompt, ALL_CASES
            )
            all_runs.append(results)
            summary(results, f"{args.prompt} rep{rep + 1}")
        # aggregate
        print(f"\n## Aggregate across {args.reps} reps")
        for i, case in enumerate(ALL_CASES):
            keep_ok = sum(1 for r in all_runs if r[i][2]["ok_keep"])
            drop_ok = sum(1 for r in all_runs if r[i][2]["ok_drop"])
            print(
                f"  [{case.area}] {case.id}: keep {keep_ok}/{args.reps}  drop {drop_ok}/{args.reps}"
            )

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
