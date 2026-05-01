"""Scaling evaluation: test how retrieval architecture performance scales
with conversation length.

Hypothesis: cue-based retrieval has an activation threshold around 250-400
turns. Below it, cues are noise; above it, they are transformative.

For each length in [100, 250, 500, 1000, 2000]:
  - Generate 3 conversations (personal-assistant, work project, health/medical)
    of that length, with planted anchors
  - Build 4-5 questions per topic requiring scattered retrieval:
    completeness, temporal, aggregation, cross-reference
  - Embed with text-embedding-3-small -> data/segments_scaling_<N>turns.npz
  - Run 5 architectures (cosine, v15_control, v2f, v2f_v2, hybrid_v2f_gencheck)
    at K=20 and K=50 with fair-backfill

Data files written:
  data/segments_scaling_100turns.npz, ..._2000turns.npz
  data/questions_scaling.json  (with conversation_length field)

Cache:  cache/scaling_llm_cache.json, scaling_embedding_cache.json
Results: results/scaling_<arch>_<N>turns.json and scaling_summary.json

Usage:
    uv run python scaling_eval.py --generate       # just generate data
    uv run python scaling_eval.py --embed          # embed the generated data
    uv run python scaling_eval.py --run            # run retrieval eval
    uv run python scaling_eval.py --lengths 100 250 500
    uv run python scaling_eval.py --all            # generate + embed + run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from associative_recall import (
    EMBED_MODEL,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"

MODEL = "gpt-5-mini"
BUDGETS = [20, 50]
EMBED_BATCH = 100


# ---------------------------------------------------------------------------
# Conversation generators
# ---------------------------------------------------------------------------
# Each generator returns (turns, anchors) where:
#   turns: list of {turn_id, role, text}
#   anchors: dict mapping anchor_name -> list[turn_id]
#
# We build 3 conversation templates, each parameterized by a target length.
# We plant N anchor "events" across the conversation, with filler between.
# For long convs, filler is many many repetitive (but realistic) turns of
# weekly meals, daily check-ins, small-talk, etc. The anchor positions are
# spread so that completeness/aggregation questions require finding scattered
# turns.


def _turn(turn_id: int, role: str, text: str) -> dict:
    return {"turn_id": turn_id, "role": role, "text": text}


def _emit_pair(turn_id: int, user: str, asst: str) -> tuple[list[dict], int]:
    """Emit a user/assistant pair. Returns (turns, next_turn_id)."""
    return [
        _turn(turn_id, "user", user),
        _turn(turn_id + 1, "assistant", asst),
    ], turn_id + 2


# ---- FILLER POOLS (small-talk, realistic daily check-ins) ----

PERSONAL_FILLER = [
    (
        "Morning! What's on the calendar today?",
        "Good morning! Just a usual day — no major appointments.",
    ),
    ("Any reminders for today?", "Nothing urgent. Want me to suggest a priority list?"),
    (
        "I had oatmeal for breakfast again.",
        "Consistent! Oatmeal is a solid choice. Any toppings today?",
    ),
    ("Just coffee so far.", "Coffee first, food later — classic."),
    ("What time is sunset today?", "Around 7:42pm. Nice long evening."),
    ("I'm going for a walk.", "Have a good one! Weather looks fine."),
    ("Back from the walk, felt good.", "Glad to hear! Did you go your usual route?"),
    ("Yeah, the park loop.", "Nice. That one's about 2 miles, right?"),
    (
        "Any good podcasts to recommend?",
        "How about 'Huberman Lab' or 'The Daily'? Depends on mood.",
    ),
    ("I'll try The Daily. Thanks.", "Good choice, it's a short daily briefing."),
    (
        "What's the weather this weekend?",
        "Looks mild — 65 and sunny Saturday, 60 with clouds Sunday.",
    ),
    ("Perfect hiking weather.", "Indeed. Any trail in mind?"),
    ("Maybe Eagle Ridge.", "Classic. That's the 4-mile loop, right?"),
    (
        "Yeah. Should I pack snacks?",
        "Definitely. A granola bar and water go a long way.",
    ),
    ("I finished that book you recommended.", "Great! How did you like it?"),
    (
        "Really enjoyed it. Any more like it?",
        "If you liked that one, try 'Project Hail Mary' by Andy Weir.",
    ),
    ("I'll add it to my list.", "Enjoy! It's a fun read."),
    (
        "What should I have for lunch today?",
        "Something light — maybe a salad with grilled chicken.",
    ),
    ("Good idea. I have leftovers actually.", "Perfect — no waste."),
    (
        "How do I keep avocados from browning?",
        "A squeeze of lemon and an airtight container works well.",
    ),
    ("Thanks for the tip.", "Anytime."),
    ("I'm so tired today.", "Did you sleep well last night?"),
    (
        "Not really, woke up at 3am.",
        "Sorry to hear. Short nap might help this afternoon.",
    ),
    ("What's a good dinner idea?", "How about salmon with roasted veggies?"),
    ("Already had fish this week.", "Then maybe chicken stir-fry — quick and easy."),
    ("I watched that new Netflix show.", "Oh nice, which one?"),
    ("The mystery one everyone's talking about.", "Ah yes, I've heard good things."),
    ("Laundry day today.", "Always satisfying to have it done."),
    ("Yep. Folding is the worst part though.", "Agreed, that's the tedious half."),
    (
        "Any advice for getting motivated in the morning?",
        "Consistent sleep schedule and a quick walk outside help most.",
    ),
]


WORK_FILLER = [
    ("Standup in 10 minutes.", "Got it. Want me to prep a quick status bullet?"),
    (
        "Quick sync — anything pending on my side?",
        "Nothing critical. Mostly awaiting client feedback.",
    ),
    ("Reviewing PRs this morning.", "Good idea — merges tend to pile up mid-week."),
    ("Client email inbox is a mess.", "Want to triage together?"),
    ("Later. Just venting.", "Fair enough."),
    ("Team lunch at noon.", "Nice break. Where are you all going?"),
    ("The sandwich place downstairs.", "Classic choice."),
    ("Vanessa sent over the revised mockups.", "Great — quick turnaround from her."),
    ("Tom is blocked on the API spec.", "Want me to draft a quick outline?"),
    ("I'll handle it, thanks.", "Sounds good."),
    (
        "Slack notifications are overwhelming today.",
        "Could snooze them for a focused block.",
    ),
    ("Good idea. Doing that now.", "Enjoy the quiet."),
    (
        "Did we ever decide on the staging URL pattern?",
        "I believe it was the preview-branch subdomain approach.",
    ),
    ("Right, that's what Maria set up.", "Yes, she configured it last week."),
    ("Running low on coffee. BRB.", "Priorities."),
    ("Back. Now what was I doing?", "You were reviewing the latest mockup revisions."),
    ("Right, thanks.", "No problem."),
    ("Any blockers I should escalate?", "Not that I'm aware of."),
    ("Ok quiet morning then.", "Make the most of it."),
    ("Team retro tomorrow.", "Want to prep any talking points?"),
    ("Later today.", "Sure."),
    ("I need to update the Notion board.", "Want me to help structure it?"),
    ("No, I got it.", "Alright."),
    ("Just pushed the hotfix.", "Good — anything to test manually?"),
    ("I already did. Green.", "Nice."),
    ("Vendor meeting moved to Friday.", "Noted. Friday at the usual time?"),
    ("Yeah, 2pm.", "Got it."),
    (
        "How do I export from Figma again?",
        "Select frames, use Export panel, add formats, hit Export Selected.",
    ),
    ("Right, thanks.", "Anytime."),
    ("Working from home tomorrow.", "Sounds good. Any morning meetings to call out?"),
]


MEDICAL_FILLER = [
    ("Took my morning pills.", "Good. Metformin and lisinopril as usual?"),
    ("Yep.", "On track."),
    ("Walked 30 minutes again today.", "Nice — consistency really pays off for A1C."),
    ("Feeling pretty good today.", "Glad to hear."),
    ("Blood sugar reading this morning was 118.", "Solid morning number."),
    ("Had oatmeal for breakfast.", "Good choice — slow carbs, minimal spike."),
    (
        "Drank plenty of water today.",
        "Hydration matters a lot, especially on metformin.",
    ),
    ("No knee pain today, surprisingly.", "Some days are better. Good news."),
    ("Slept through the night last night.", "Wonderful! That's rare lately."),
    ("Quick stretch routine done.", "Helps the knee too."),
    ("Bought more vitamin D today.", "Good — running out would mess up the trend."),
    ("Bought groceries — low sugar week.", "Smart. Keeps the A1C trending down."),
    ("Had a small dessert though.", "A little is fine. Everything in moderation."),
    ("Doctor's office confirmed my appointment.", "Good. Dr. Patel, right?"),
    ("Yes. I'll write down the questions I want to ask.", "Great idea."),
    ("Wife went to her PT session today.", "How's her shoulder doing?"),
    ("Slowly improving.", "PT takes time but it works."),
    (
        "Skipped ibuprofen today. Knee was fine.",
        "Good — less NSAID use is safer with your lisinopril.",
    ),
    ("No sharp pains today.", "Encouraging."),
    ("Walking routine is becoming automatic.", "Habit formation at work."),
    (
        "Tried a yoga video this morning.",
        "Nice — easier on the joints than pounding pavement.",
    ),
    ("Liked it a lot.", "Could become a regular thing."),
    ("Making a salad for lunch today.", "Keep it simple and filling."),
    ("Added chickpeas for protein.", "Great move — fiber + protein."),
    ("Avoided dessert today.", "Nice."),
    ("Read up on glycemic index.", "Understanding the science helps a lot."),
    (
        "Finally figured out which carbs spike me most.",
        "That's useful data to carry forward.",
    ),
    ("White bread is definitely a no-go.", "Yeah, that's a common finding."),
    ("Switching to whole grain.", "Much better option."),
    ("Had my tea and did evening stretches.", "Nice wind-down routine."),
]


def _pick_filler(
    pool: list[tuple[str, str]], rng: random.Random, k: int
) -> list[tuple[str, str]]:
    """Pick k filler pairs. If k > len(pool), cycles through."""
    out = []
    for i in range(k):
        out.append(pool[i % len(pool)])
    # Shuffle the cycles in place so it's not a perfect repeat
    rng.shuffle(out)
    return out


# ---- ANCHOR TEMPLATES ----
# Each conversation has 8-14 "anchor" events that ground the retrieval questions.
# These anchors are planted at roughly even intervals across the conversation.


@dataclass
class Anchor:
    """A conversation anchor: specific content that question source_ids point to."""

    name: str  # e.g., "bob_peanut_allergy"
    turns: list[tuple[str, str]]  # list of (user, assistant) pairs
    # question categories this anchor is referenced by
    tags: list[str] = field(default_factory=list)


def _personal_anchors() -> list[Anchor]:
    """Anchors for personal assistant conversation (1-year scenario).

    Covers: dinner party dietary restrictions (9 anchors), birthday gifts,
    appointments, gifts, health events. Designed so:
    - completeness: list all dietary restrictions across all anchors
    - temporal: when did Bob's allergy list first change?
    - aggregation: how did the dinner plan evolve over time?
    - cross-reference: after Bob was diagnosed lactose-intolerant, what
      menu changes did we make?
    """
    return [
        Anchor(
            name="p_dinner_plan",
            turns=[
                (
                    "I'm planning a dinner party for 8 people on Saturday.",
                    "How fun! Want help with menu planning? Any dietary notes?",
                ),
                (
                    "Yeah, guests are Bob, Linda, Sarah, James, Priya, Dev, and me plus Rachel.",
                    "Noted: 8 total. Let's figure out dietary restrictions.",
                ),
            ],
            tags=["completeness", "cross_reference"],
        ),
        Anchor(
            name="p_bob_peanut",
            turns=[
                (
                    "Bob is allergic to peanuts. Severe, since childhood.",
                    "Noted: severe peanut allergy for Bob.",
                ),
            ],
            tags=["completeness", "temporal"],
        ),
        Anchor(
            name="p_sarah_vegan",
            turns=[
                (
                    "Sarah is vegetarian, and she also avoids eggs and dairy now, so effectively vegan.",
                    "Got it — Sarah is effectively vegan. No eggs, no dairy.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="p_patels_no_beef",
            turns=[
                (
                    "Priya and Dev Patel don't eat beef — they're Hindu.",
                    "Noted: no beef for the Patels.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="p_linda_no_mushrooms",
            turns=[
                (
                    "Just heard from Bob — Linda hates mushrooms. Preference, not allergy.",
                    "Noted: no mushrooms for Linda.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="p_rachel_celiac",
            turns=[
                (
                    "My sister Rachel has celiac disease — strict gluten-free, even trace matters.",
                    "Understood — celiac means strict gluten-free for Rachel.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="p_bob_lactose",
            turns=[
                (
                    "Bob just told me his doctor said he's lactose intolerant as of last week.",
                    "Adding: Bob is now also lactose intolerant. Aged cheese/butter OK?",
                ),
                (
                    "Yes, aged cheese and butter he handles fine.",
                    "Got it — no milk/cream/soft cheese for Bob.",
                ),
            ],
            tags=["completeness", "temporal", "cross_reference"],
        ),
        Anchor(
            name="p_emma_tree_nut",
            turns=[
                (
                    "Emma, Bob's daughter, will likely come. She has a tree nut allergy — serious, EpiPen.",
                    "Very important — tree nuts and peanuts both avoided. No-nut policy for safety.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="p_sarah_soy",
            turns=[
                (
                    "Sarah also has a soy allergy — I didn't know that before.",
                    "Important — Sarah is vegan AND soy-allergic. Skip tamari, use coconut aminos.",
                ),
            ],
            tags=["completeness", "cross_reference"],
        ),
        Anchor(
            name="p_bob_shellfish_retracted",
            turns=[
                (
                    "Bob said his allergist retested him — he outgrew the shellfish allergy. Scratched.",
                    "Good news — shellfish allergy is RETRACTED per allergist. Peanut and lactose still there.",
                ),
            ],
            tags=["temporal", "aggregation"],
        ),
        Anchor(
            name="p_menu_finalized",
            turns=[
                (
                    "Final menu: rice crackers with bruschetta topping, grilled salmon with lemon dill, "
                    "vegan black bean sweet potato stew, roasted veggies no mushrooms, jasmine rice, "
                    "green salad, mango sorbet. All gluten-free and nut-free.",
                    "Locked in: GF, nut-free, no beef, no mushrooms, dairy-free main, vegan alternative. "
                    "Shopping list to follow.",
                ),
            ],
            tags=["aggregation", "cross_reference"],
        ),
        Anchor(
            name="p_bob_birthday",
            turns=[
                (
                    "Bob's birthday is next Thursday. He's into craft beer and woodworking.",
                    "Ideas: carving chisels or craft beer subscription. I'll remind you.",
                ),
            ],
            tags=["temporal"],
        ),
        Anchor(
            name="p_rachel_xmas",
            turns=[
                (
                    "I want to plan Christmas gifts for Rachel. She's into pottery, hiking, and birdwatching.",
                    "Pottery tools, binoculars, or hiking gear would all work well.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="p_mom_anniversary",
            turns=[
                (
                    "Mom's anniversary is on the 28th — we're going to Giovanni's.",
                    "Giovanni's on the 28th. I'll remind you closer to the date.",
                ),
            ],
            tags=["temporal"],
        ),
    ]


def _work_anchors() -> list[Anchor]:
    """Work project (Acme rebrand) anchors."""
    return [
        Anchor(
            name="w_kickoff",
            turns=[
                (
                    "We just signed Acme Corp for a complete rebrand. Kickoff January 5th.",
                    "Big one! Logo, website, marketing materials. Timeline?",
                ),
                (
                    "Deadline March 15th — non-negotiable. Launch March 20th.",
                    "10 weeks. Tight but doable.",
                ),
            ],
            tags=["temporal"],
        ),
        Anchor(
            name="w_team",
            turns=[
                (
                    "Team: me as PM, Vanessa design, Tom frontend, Maria backend, Hiroshi copy.",
                    "Solid five-person team.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="w_brand_primary",
            turns=[
                (
                    "Acme's primary color stays as Pantone 2945 C — deep blue.",
                    "Noted: primary is Pantone 2945 C.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="w_brand_secondary",
            turns=[
                (
                    "Vanessa is proposing new secondaries: cool gray and teal accent, replacing the burnt orange.",
                    "Modern combo. Client sees it Wednesday.",
                ),
            ],
            tags=["completeness", "cross_reference"],
        ),
        Anchor(
            name="w_cms_migration",
            turns=[
                (
                    "Tom says their current stack is WordPress with custom theme. "
                    "We're proposing Strapi + Next.js.",
                    "Headless CMS + Next.js is modern. Marketing training needed.",
                ),
            ],
            tags=["completeness", "cross_reference"],
        ),
        Anchor(
            name="w_wcag",
            turns=[
                (
                    "Acme's legal requires WCAG 2.1 AA compliance on the site.",
                    "Tom should factor in contrast, keyboard nav, alt text from the start.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="w_animation_scope",
            turns=[
                (
                    "Patricia wants Apple-style scroll animations — major scope creep.",
                    "Motion design + GSAP/Framer Motion work. Timeline or budget impact.",
                ),
                (
                    "Compromise reached: subtle parallax instead of Apple-style.",
                    "Much more manageable. Tom says very doable.",
                ),
            ],
            tags=["aggregation", "temporal", "cross_reference"],
        ),
        Anchor(
            name="w_typography",
            turns=[
                (
                    "Acme uses a custom typeface — Acme Sans. Fallback is Inter.",
                    "Noted: Acme Sans with Inter fallback. font-display: swap matters.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="w_brand_voice",
            turns=[
                (
                    "Hiroshi settled on 'confident but conversational' — Slack/Stripe style.",
                    "Fits new CEO Daniel Park's vision.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="w_social_media",
            turns=[
                (
                    "Client asked for social media templates — Instagram, LinkedIn, Twitter. Phase 2.",
                    "Noted: social templates in Phase 2, mid-Feb start.",
                ),
            ],
            tags=["completeness", "temporal"],
        ),
        Anchor(
            name="w_motion_guidelines",
            turns=[
                (
                    "Vanessa is adding a motion guidelines section to the brand book.",
                    "Smart — easing curves, duration standards, consistency.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="w_ceo_name",
            turns=[
                (
                    "The new Acme CEO is Daniel Park, from a tech startup background.",
                    "Daniel Park — he's pushing the startup-y brand voice.",
                ),
            ],
            tags=["temporal"],
        ),
        Anchor(
            name="w_staging_vercel",
            turns=[
                (
                    "Maria is setting up staging on Vercel. Preview deployments per PR.",
                    "Great — client reviews get their own preview URLs.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="w_launch",
            turns=[
                (
                    "Launch is on track for March 20th. Everyone knows the deadline.",
                    "Good. Phase 4 final revisions March 9-15.",
                ),
            ],
            tags=["aggregation", "temporal"],
        ),
    ]


def _medical_anchors() -> list[Anchor]:
    """Medical/health anchors."""
    return [
        Anchor(
            name="m_metformin",
            turns=[
                (
                    "I take metformin 500mg twice daily for type 2 diabetes.",
                    "Noted: metformin 500mg BID for T2DM.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="m_lisinopril",
            turns=[
                (
                    "I take lisinopril 10mg once daily for blood pressure. Dr. Patel started me on it 6 months ago.",
                    "Added: lisinopril 10mg QD, started 6 months ago.",
                ),
            ],
            tags=["completeness", "temporal"],
        ),
        Anchor(
            name="m_atorvastatin",
            turns=[
                (
                    "Atorvastatin 20mg at bedtime for cholesterol — been on it for years.",
                    "Noted: atorvastatin 20mg HS, long-term.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="m_vitamin_d",
            turns=[
                (
                    "I take vitamin D 2000 IU daily — my levels were low.",
                    "Added to the list.",
                ),
            ],
            tags=["completeness"],
        ),
        Anchor(
            name="m_a1c_improving",
            turns=[
                (
                    "Last A1C was 6.8, down from 7.2 six months ago. Dr. Patel was pleased.",
                    "Great trend. Metformin plus lifestyle changes showing results.",
                ),
            ],
            tags=["aggregation", "temporal"],
        ),
        Anchor(
            name="m_knee_pain",
            turns=[
                (
                    "My right knee has been aching for 3 weeks. Dull ache, sharp on stairs.",
                    "Sorry — gradual-onset knee pain at 54 could be OA. Mention it at your next appt.",
                ),
            ],
            tags=["temporal", "cross_reference"],
        ),
        Anchor(
            name="m_ibuprofen_warning",
            turns=[
                (
                    "I've been taking 400mg ibuprofen 2-3 times a week when the knee flares up.",
                    "Heads up: NSAIDs can interact with lisinopril — reduces BP effectiveness "
                    "and affects kidneys, more concerning with diabetes. Discuss with Dr. Patel.",
                ),
            ],
            tags=["cross_reference", "aggregation"],
        ),
        Anchor(
            name="m_blood_panel",
            turns=[
                (
                    "Blood panel is February 10th — A1C, lipids, vitamin D.",
                    "Noted. That's your quarterly check.",
                ),
            ],
            tags=["temporal"],
        ),
        Anchor(
            name="m_appt_jan25",
            turns=[
                (
                    "I see Dr. Patel on January 25th.",
                    "I'll remind you — things to bring up: knee pain and ibuprofen use.",
                ),
            ],
            tags=["temporal", "cross_reference"],
        ),
        Anchor(
            name="m_sleep_issue",
            turns=[
                (
                    "I've been waking up at 3am and can't get back to sleep. Going on 2 weeks.",
                    "Middle-of-the-night waking is often stress-related. Worth mentioning to Dr. Patel.",
                ),
            ],
            tags=["temporal"],
        ),
        Anchor(
            name="m_vitamin_d_levels",
            turns=[
                (
                    "My vitamin D went from 22 to 38. Target is 40.",
                    "Good progress. Almost at goal.",
                ),
            ],
            tags=["aggregation", "completeness"],
        ),
        Anchor(
            name="m_walking",
            turns=[
                (
                    "I've been walking 30 minutes daily and cut back on sugar.",
                    "Big reason your A1C dropped.",
                ),
            ],
            tags=["aggregation", "cross_reference"],
        ),
        Anchor(
            name="m_pt_referral",
            turns=[
                (
                    "Dr. Patel referred me to ProMotion Physical Therapy for the knee.",
                    "Good — PT has no drug interactions. Same place your wife goes.",
                ),
            ],
            tags=["cross_reference", "temporal"],
        ),
        Anchor(
            name="m_new_med_tylenol",
            turns=[
                (
                    "Dr. Patel said to switch from ibuprofen to Tylenol for the knee — safer with lisinopril.",
                    "Good change — acetaminophen avoids the NSAID-lisinopril interaction.",
                ),
            ],
            tags=["cross_reference", "aggregation"],
        ),
    ]


# ---- CONVERSATION BUILDER ----
def build_conversation(
    anchors: list[Anchor],
    filler_pool: list[tuple[str, str]],
    target_turns: int,
    rng: random.Random,
) -> tuple[list[dict], dict[str, list[int]]]:
    """Build a conversation of ~target_turns turns, planting anchors evenly.

    Anchors are placed at fractional positions so they stay roughly evenly
    distributed even at long lengths. Between anchors, filler turns are
    inserted.

    Returns (turns, anchor_turn_ids) where anchor_turn_ids maps
    anchor name -> list[int] of the turn_ids that make up that anchor.
    """
    # How many turns each anchor takes up (each anchor is a list of pairs)
    anchor_sizes = [len(a.turns) * 2 for a in anchors]  # each pair = 2 turns
    total_anchor_turns = sum(anchor_sizes)
    filler_turns_needed = max(0, target_turns - total_anchor_turns)
    # Filler turns are emitted as pairs, so ensure even count
    filler_pairs_needed = filler_turns_needed // 2

    # Distribute filler pairs across (N+1) slots: before, between, after
    n_slots = len(anchors) + 1
    # Even split with remainders distributed
    base = filler_pairs_needed // n_slots
    remainder = filler_pairs_needed - (base * n_slots)
    slot_sizes = [base] * n_slots
    # Distribute remainder evenly
    for i in range(remainder):
        slot_sizes[i * n_slots // max(1, remainder) % n_slots] += 1

    turns: list[dict] = []
    anchor_turn_ids: dict[str, list[int]] = {}
    tid = 0

    def emit_filler(n_pairs: int) -> None:
        nonlocal tid
        fillers = _pick_filler(filler_pool, rng, n_pairs)
        for u, a in fillers:
            t, tid = _emit_pair(tid, u, a)
            turns.extend(t)

    # Leading filler
    emit_filler(slot_sizes[0])

    for i, anchor in enumerate(anchors):
        anchor_ids: list[int] = []
        for u, a in anchor.turns:
            t, tid = _emit_pair(tid, u, a)
            anchor_ids.append(t[0]["turn_id"])
            anchor_ids.append(t[1]["turn_id"])
            turns.extend(t)
        anchor_turn_ids[anchor.name] = anchor_ids
        # Following filler
        emit_filler(slot_sizes[i + 1])

    return turns, anchor_turn_ids


# ---------------------------------------------------------------------------
# Questions (defined per topic, using anchor names, filled in at gen time)
# ---------------------------------------------------------------------------

# Each question template references anchors by name. We resolve source_chat_ids
# at conversation-build time by looking up the anchor's turn_ids.
QUESTIONS_TEMPLATES = {
    "synth_personal": [
        {
            "question": "List ALL dietary restrictions and food-related allergies "
            "for every guest attending the Saturday dinner party, "
            "including any updates or retractions discussed later.",
            "category": "completeness",
            "anchors": [
                "p_dinner_plan",
                "p_bob_peanut",
                "p_sarah_vegan",
                "p_patels_no_beef",
                "p_linda_no_mushrooms",
                "p_rachel_celiac",
                "p_bob_lactose",
                "p_emma_tree_nut",
                "p_sarah_soy",
                "p_bob_shellfish_retracted",
            ],
        },
        {
            "question": "When did the user first learn about Bob's lactose "
            "intolerance, and what menu changes resulted?",
            "category": "temporal",
            "anchors": ["p_bob_lactose", "p_menu_finalized"],
        },
        {
            "question": "How did the list of Bob's medical conditions or "
            "allergies change over the course of the conversation?",
            "category": "aggregation",
            "anchors": [
                "p_bob_peanut",
                "p_bob_lactose",
                "p_bob_shellfish_retracted",
            ],
        },
        {
            "question": "After learning about Sarah's soy allergy, what "
            "specific ingredient substitutions were made to the menu?",
            "category": "cross_reference",
            "anchors": ["p_sarah_soy", "p_menu_finalized"],
        },
        {
            "question": "List all events, appointments, or gift plans that were "
            "scheduled or mentioned during these conversations.",
            "category": "completeness",
            "anchors": [
                "p_bob_birthday",
                "p_rachel_xmas",
                "p_mom_anniversary",
            ],
        },
    ],
    "synth_work": [
        {
            "question": "What are all of the agenda items that should be covered "
            "in the upcoming Wednesday client review meeting?",
            "category": "completeness",
            "anchors": [
                "w_brand_primary",
                "w_brand_secondary",
                "w_typography",
                "w_brand_voice",
                "w_cms_migration",
                "w_animation_scope",
                "w_motion_guidelines",
            ],
        },
        {
            "question": "When was the Acme Corp project kicked off, and when "
            "is the hard launch deadline?",
            "category": "temporal",
            "anchors": ["w_kickoff", "w_launch"],
        },
        {
            "question": "How did the scope of the animated hero section on the "
            "website change throughout the conversation?",
            "category": "aggregation",
            "anchors": ["w_animation_scope"],
        },
        {
            "question": "After the WCAG 2.1 AA compliance requirement was "
            "introduced, what implementation choices did Tom need "
            "to factor in?",
            "category": "cross_reference",
            "anchors": ["w_wcag", "w_cms_migration"],
        },
        {
            "question": "List all team members on the Acme Corp project and "
            "their responsibilities.",
            "category": "completeness",
            "anchors": ["w_team", "w_staging_vercel", "w_brand_voice"],
        },
    ],
    "synth_medical": [
        {
            "question": "List all medications the user is currently taking, "
            "with dose, frequency, and reason.",
            "category": "completeness",
            "anchors": [
                "m_metformin",
                "m_lisinopril",
                "m_atorvastatin",
                "m_vitamin_d",
            ],
        },
        {
            "question": "When did the knee pain first start, and when is the "
            "next scheduled appointment with Dr. Patel?",
            "category": "temporal",
            "anchors": ["m_knee_pain", "m_appt_jan25"],
        },
        {
            "question": "How has the user's A1C and vitamin D level changed over time?",
            "category": "aggregation",
            "anchors": [
                "m_a1c_improving",
                "m_vitamin_d_levels",
                "m_walking",
            ],
        },
        {
            "question": "After the ibuprofen-lisinopril interaction was "
            "flagged, what change did Dr. Patel recommend, and "
            "what alternative was substituted?",
            "category": "cross_reference",
            "anchors": [
                "m_ibuprofen_warning",
                "m_new_med_tylenol",
            ],
        },
        {
            "question": "List every health-related appointment or referral "
            "that has been scheduled.",
            "category": "completeness",
            "anchors": [
                "m_blood_panel",
                "m_appt_jan25",
                "m_pt_referral",
            ],
        },
    ],
}


TOPICS = {
    "synth_personal": {
        "anchors": _personal_anchors,
        "filler": PERSONAL_FILLER,
    },
    "synth_work": {
        "anchors": _work_anchors,
        "filler": WORK_FILLER,
    },
    "synth_medical": {
        "anchors": _medical_anchors,
        "filler": MEDICAL_FILLER,
    },
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_all_data(lengths: list[int], seed: int = 42) -> None:
    """Generate conversations and questions for each length.

    Writes segments_scaling_<N>turns.npz raw (no embeddings yet — use embed
    phase for that). To keep generation cheap, we just write JSON sidecars
    of the raw turns, then the embed phase produces the npz files.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_questions: list[dict] = []

    for length in lengths:
        rng = random.Random(seed + length)
        print(f"\n=== Generating conversations at target length {length} ===")

        all_turns_rows: list[dict] = []
        for conv_name, cfg in TOPICS.items():
            anchors = cfg["anchors"]()
            filler = cfg["filler"]
            turns, anchor_ids = build_conversation(
                anchors,
                filler,
                length,
                rng,
            )
            actual = len(turns)
            print(
                f"  {conv_name}: anchor_count={len(anchors)}, "
                f"target={length}, actual={actual}"
            )
            for t in turns:
                all_turns_rows.append(
                    {
                        "conversation_id": conv_name,
                        "turn_id": t["turn_id"],
                        "role": t["role"],
                        "text": t["text"],
                    }
                )
            # Resolve questions for this conversation
            for q_tmpl in QUESTIONS_TEMPLATES[conv_name]:
                source_ids: list[int] = []
                for a_name in q_tmpl["anchors"]:
                    source_ids.extend(anchor_ids.get(a_name, []))
                source_ids = sorted(set(source_ids))
                all_questions.append(
                    {
                        "conversation_id": conv_name,
                        "conversation_length": length,
                        "category": q_tmpl["category"],
                        "question_index": len(all_questions),
                        "question": q_tmpl["question"],
                        "source_chat_ids": source_ids,
                        "benchmark": f"scaling_{length}turns",
                    }
                )

        # Save turns json (unembedded). Embed phase will read this.
        turns_json = DATA_DIR / f"segments_scaling_{length}turns.json"
        with open(turns_json, "w") as f:
            json.dump(all_turns_rows, f)
        print(f"  Saved raw turns: {turns_json} ({len(all_turns_rows)} rows)")

    q_path = DATA_DIR / "questions_scaling.json"
    with open(q_path, "w") as f:
        json.dump(all_questions, f, indent=2)
    print(f"\nSaved {len(all_questions)} questions to {q_path}")


# ---------------------------------------------------------------------------
# Embedding phase
# ---------------------------------------------------------------------------
class ScalingEmbeddingCache:
    """Embedding cache specific to scaling experiment."""

    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = CACHE_DIR / "scaling_embedding_cache.json"
        self._cache: dict[str, list[float]] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache = json.load(f)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        k = self._key(text)
        if k in self._cache:
            return np.array(self._cache[k], dtype=np.float32)
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        self._cache[self._key(text)] = embedding.tolist()

    def save(self) -> None:
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        tmp.replace(self.cache_file)


def embed_all_data(lengths: list[int]) -> None:
    """For each length, read the turns JSON, embed, and save the npz.

    Uses the scaling embedding cache so re-runs are free. Batches API calls
    to keep embedding cost manageable.
    """
    client = OpenAI(timeout=60.0)
    cache = ScalingEmbeddingCache()

    for length in lengths:
        turns_json = DATA_DIR / f"segments_scaling_{length}turns.json"
        if not turns_json.exists():
            print(f"  Missing {turns_json}, skipping (run --generate first)")
            continue
        with open(turns_json) as f:
            rows = json.load(f)
        print(f"\nEmbedding {length}turns: {len(rows)} rows")

        texts = [r["text"] for r in rows]
        embeddings = np.zeros((len(texts), 1536), dtype=np.float32)
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, t in enumerate(texts):
            cached = cache.get(t)
            if cached is not None:
                embeddings[i] = cached
            else:
                missing_idx.append(i)
                missing_texts.append(t)

        print(f"  cached={len(texts) - len(missing_idx)}, new={len(missing_idx)}")

        for start in range(0, len(missing_texts), EMBED_BATCH):
            batch = missing_texts[start : start + EMBED_BATCH]
            batch_idx = missing_idx[start : start + EMBED_BATCH]
            print(
                f"    Embedding batch {start // EMBED_BATCH + 1} "
                f"({len(batch)} texts)...",
                flush=True,
            )
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            for j, item in zip(batch_idx, resp.data):
                emb = np.array(item.embedding, dtype=np.float32)
                embeddings[j] = emb
                cache.put(texts[j], emb)
            cache.save()
            time.sleep(0.05)

        out = DATA_DIR / f"segments_scaling_{length}turns.npz"
        np.savez(
            out,
            embeddings=embeddings,
            conversation_ids=np.array([r["conversation_id"] for r in rows]),
            turn_ids=np.array([r["turn_id"] for r in rows], dtype=np.int32),
            roles=np.array([r["role"] for r in rows]),
            texts=np.array([r["text"] for r in rows]),
        )
        print(f"  Saved {out} shape={embeddings.shape}")

    cache.save()


# ---------------------------------------------------------------------------
# Architectures: cosine_baseline + v15_control + v2f + v2f_v2 + hybrid_v2f_gencheck
# (Self-contained here so we only depend on common SegmentStore/cache plumbing.)
# ---------------------------------------------------------------------------
class ScalingLLMCache:
    """LLM cache for scaling experiment."""

    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = CACHE_DIR / "scaling_llm_cache.json"
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
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        tmp.replace(self.cache_file)


V15_CONTROL_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

V2F_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# v2f_v2: v15 + completeness hint, but NO anti-question rule
V2F_V2_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

GAP_ASSESSMENT_PROMPT = """\
You are reviewing retrieval results for a task. Given the question/task and \
the conversation segments retrieved so far, assess whether anything \
important is missing.

QUESTION/TASK: {question}

RETRIEVED SEGMENTS:
{formatted_segments}

Think critically:
1. Given what I've found, is there anything important for this task that \
I HAVEN'T retrieved?
2. What assumptions am I making that should be checked?
3. Are there implicit requirements (e.g., dietary restrictions, scheduling \
conflicts, prerequisites) that the question doesn't explicitly ask about \
but would be important?
4. If this is a proactive task (planning, drafting, preparing), what \
background information might I need that isn't directly mentioned in \
the question?

If there are genuine gaps, generate 1-2 targeted search cues. Each cue \
should sound like conversation content (not a search command).

If the retrieval looks comprehensive, respond with DONE.

Format:
ASSESSMENT: <what's missing or what assumptions need checking>
GAP: <text mimicking conversation content>
GAP: <text mimicking conversation content>
(or)
ASSESSMENT: <retrieval looks complete because...>
DONE"""


def _format_segments(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def _build_context_section(all_segments: list[Segment]) -> str:
    if not all_segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    return "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(all_segments)


def _parse_cues(response: str, tag: str = "CUE") -> list[str]:
    cues = []
    needle = f"{tag}:"
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith(needle):
            c = line[len(needle) :].strip()
            if c:
                cues.append(c)
    return cues


class ScalingArch:
    """Base for scaling-experiment architectures."""

    name = "base"

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI,
        emb_cache: ScalingEmbeddingCache,
        llm_cache: ScalingLLMCache,
    ) -> None:
        self.store = store
        self.client = client
        self.emb_cache = emb_cache
        self.llm_cache = llm_cache
        self.embed_calls = 0
        self.llm_calls = 0

    def reset(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.emb_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        emb = np.array(resp.data[0].embedding, dtype=np.float32)
        self.emb_cache.put(text, emb)
        self.embed_calls += 1
        return emb

    def llm(self, prompt: str, max_tokens: int = 2000) -> str:
        cached = self.llm_cache.get(MODEL, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        self.llm_cache.put(MODEL, prompt, text)
        self.llm_calls += 1
        return text

    def retrieve(self, question: str, conv_id: str) -> list[Segment]:
        raise NotImplementedError


class CosineBaseline(ScalingArch):
    """Pure cosine: retrieve top 50 with the raw question. No LLM calls."""

    name = "cosine_baseline"

    def retrieve(self, question: str, conv_id: str) -> list[Segment]:
        q_emb = self.embed_text(question)
        result = self.store.search(q_emb, top_k=50, conversation_id=conv_id)
        return list(result.segments)


class V15ControlArch(ScalingArch):
    name = "v15_control"

    def retrieve(self, question: str, conv_id: str) -> list[Segment]:
        q_emb = self.embed_text(question)
        hop0 = self.store.search(q_emb, top_k=10, conversation_id=conv_id)
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        ctx = _build_context_section(all_segments)
        prompt = V15_CONTROL_PROMPT.format(question=question, context_section=ctx)
        out = self.llm(prompt)
        cues = _parse_cues(out)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            r = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conv_id,
                exclude_indices=exclude,
            )
            for s in r.segments:
                if s.index not in exclude:
                    all_segments.append(s)
                    exclude.add(s.index)
        return all_segments


class V2fArch(ScalingArch):
    name = "v2f"

    def retrieve(self, question: str, conv_id: str) -> list[Segment]:
        q_emb = self.embed_text(question)
        hop0 = self.store.search(q_emb, top_k=10, conversation_id=conv_id)
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        ctx = _build_context_section(all_segments)
        prompt = V2F_PROMPT.format(question=question, context_section=ctx)
        out = self.llm(prompt)
        cues = _parse_cues(out)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            r = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conv_id,
                exclude_indices=exclude,
            )
            for s in r.segments:
                if s.index not in exclude:
                    all_segments.append(s)
                    exclude.add(s.index)
        return all_segments


class V2fV2Arch(ScalingArch):
    name = "v2f_v2"

    def retrieve(self, question: str, conv_id: str) -> list[Segment]:
        q_emb = self.embed_text(question)
        hop0 = self.store.search(q_emb, top_k=10, conversation_id=conv_id)
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        ctx = _build_context_section(all_segments)
        prompt = V2F_V2_PROMPT.format(question=question, context_section=ctx)
        out = self.llm(prompt)
        cues = _parse_cues(out)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            r = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conv_id,
                exclude_indices=exclude,
            )
            for s in r.segments:
                if s.index not in exclude:
                    all_segments.append(s)
                    exclude.add(s.index)
        return all_segments


class HybridV2fGenCheckArch(ScalingArch):
    name = "hybrid_v2f_gencheck"

    def retrieve(self, question: str, conv_id: str) -> list[Segment]:
        q_emb = self.embed_text(question)
        hop0 = self.store.search(q_emb, top_k=10, conversation_id=conv_id)
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        # v2f cue generation
        ctx = _build_context_section(all_segments)
        prompt = V2F_PROMPT.format(question=question, context_section=ctx)
        out = self.llm(prompt)
        cues = _parse_cues(out)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            r = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conv_id,
                exclude_indices=exclude,
            )
            for s in r.segments:
                if s.index not in exclude:
                    all_segments.append(s)
                    exclude.add(s.index)

        # Gen-check gap assessment
        formatted = _format_segments(all_segments, max_items=16, max_chars=300)
        gap_prompt = GAP_ASSESSMENT_PROMPT.format(
            question=question, formatted_segments=formatted
        )
        gap_out = self.llm(gap_prompt)
        gaps = _parse_cues(gap_out, tag="GAP")
        done = bool(gap_out) and "DONE" in gap_out.upper().split("\n")[-1]

        if not done and gaps:
            for gap in gaps[:2]:
                gap_emb = self.embed_text(gap)
                r = self.store.search(
                    gap_emb,
                    top_k=10,
                    conversation_id=conv_id,
                    exclude_indices=exclude,
                )
                for s in r.segments:
                    if s.index not in exclude:
                        all_segments.append(s)
                        exclude.add(s.index)

        return all_segments


ARCHITECTURES = {
    "cosine_baseline": CosineBaseline,
    "v15_control": V15ControlArch,
    "v2f": V2fArch,
    "v2f_v2": V2fV2Arch,
    "hybrid_v2f_gencheck": HybridV2fGenCheckArch,
}


# ---------------------------------------------------------------------------
# Evaluation (fair backfill at K=20, 50)
# ---------------------------------------------------------------------------
def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    source_ids: set[int],
    budget: int,
) -> tuple[float, float]:
    """At budget K: baseline = cosine top-K recall; arch = arch segments (in
    order) backfilled with cosine top-K, truncated to K."""
    seen: set[int] = set()
    arch_unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            arch_unique.append(s)
            seen.add(s.index)

    arch_at_K = arch_unique[:budget]
    arch_indices = {s.index for s in arch_at_K}
    if len(arch_at_K) < budget:
        backfill = [s for s in cosine_segments if s.index not in arch_indices]
        arch_at_K = arch_at_K + backfill[: budget - len(arch_at_K)]
    arch_at_K = arch_at_K[:budget]

    baseline_at_K = cosine_segments[:budget]
    arch_ids = {s.turn_id for s in arch_at_K}
    baseline_ids = {s.turn_id for s in baseline_at_K}

    return (
        compute_recall(baseline_ids, source_ids),
        compute_recall(arch_ids, source_ids),
    )


def evaluate_question(arch: ScalingArch, q: dict) -> dict:
    q_text = q["question"]
    conv_id = q["conversation_id"]
    source_ids = set(q["source_chat_ids"])

    arch.reset()
    t0 = time.time()
    arch_segments = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedupe preserving order
    seen: set[int] = set()
    deduped: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            deduped.append(s)
            seen.add(s.index)
    arch_segments = deduped

    # Cosine top-50 for fair backfill
    q_emb = arch.embed_text(q_text)
    cosine_result = arch.store.search(q_emb, top_k=50, conversation_id=conv_id)
    cosine_segments = list(cosine_result.segments)

    row = {
        "conversation_id": conv_id,
        "conversation_length": q.get("conversation_length"),
        "category": q["category"],
        "question_index": q["question_index"],
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "total_arch_retrieved": len(arch_segments),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
    }
    for K in BUDGETS:
        b, a = fair_backfill(arch_segments, cosine_segments, source_ids, K)
        row["fair_backfill"][f"baseline_r@{K}"] = round(b, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a - b, 4)
    return row


def summarize(rows: list[dict], arch_name: str, length: int) -> dict:
    n = len(rows)
    summary: dict = {
        "arch": arch_name,
        "conversation_length": length,
        "n": n,
    }
    if n == 0:
        return summary
    for K in BUDGETS:
        b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in rows]
        a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in rows]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses
        summary[f"baseline_r@{K}"] = round(b_mean, 4)
        summary[f"arch_r@{K}"] = round(a_mean, 4)
        summary[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
        summary[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
    summary["avg_total_retrieved"] = round(
        sum(r["total_arch_retrieved"] for r in rows) / n, 1
    )
    summary["avg_llm_calls"] = round(sum(r["llm_calls"] for r in rows) / n, 1)
    summary["avg_embed_calls"] = round(sum(r["embed_calls"] for r in rows) / n, 1)
    summary["avg_time_s"] = round(sum(r["time_s"] for r in rows) / n, 2)
    return summary


def summarize_by_category(rows: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry: dict = {"n": n}
        for K in BUDGETS:
            b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in rs]
            a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in rs]
            b_mean = sum(b_vals) / n
            a_mean = sum(a_vals) / n
            wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
            losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
            entry[f"baseline_r@{K}"] = round(b_mean, 4)
            entry[f"arch_r@{K}"] = round(a_mean, 4)
            entry[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
            entry[f"W/T/L_r@{K}"] = f"{wins}/{n - wins - losses}/{losses}"
        out[cat] = entry
    return out


def run_eval(lengths: list[int], arch_names: list[str]) -> None:
    """Run all architectures across all lengths. Writes per-(arch, length)
    results plus aggregated summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(timeout=120.0)
    emb_cache = ScalingEmbeddingCache()
    llm_cache = ScalingLLMCache()

    # Load questions once
    q_path = DATA_DIR / "questions_scaling.json"
    with open(q_path) as f:
        all_questions = json.load(f)

    # Group by length
    by_length: dict[int, list[dict]] = defaultdict(list)
    for q in all_questions:
        by_length[q["conversation_length"]].append(q)

    all_summaries: dict = {}  # {arch_name: {length: summary}}

    for length in lengths:
        npz = DATA_DIR / f"segments_scaling_{length}turns.npz"
        if not npz.exists():
            print(f"Missing {npz}, skip length {length}")
            continue
        store = SegmentStore(data_dir=DATA_DIR, npz_name=npz.name)
        qs = by_length.get(length, [])
        print(f"\n{'=' * 70}")
        print(f"LENGTH={length}  segments={len(store.segments)}  questions={len(qs)}")
        print(f"{'=' * 70}")

        for arch_name in arch_names:
            cls = ARCHITECTURES[arch_name]
            arch = cls(store, client, emb_cache, llm_cache)

            print(f"\n--- {arch_name} @ {length}turns ---", flush=True)
            rows: list[dict] = []
            for i, q in enumerate(qs):
                q_short = q["question"][:55]
                print(
                    f"  [{i + 1}/{len(qs)}] {q['category']}: {q_short}...", flush=True
                )
                try:
                    rows.append(evaluate_question(arch, q))
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    import traceback

                    traceback.print_exc()
                sys.stdout.flush()
                if (i + 1) % 5 == 0:
                    emb_cache.save()
                    llm_cache.save()
            emb_cache.save()
            llm_cache.save()

            summary = summarize(rows, arch_name, length)
            by_cat = summarize_by_category(rows)

            # Print compact
            for K in BUDGETS:
                print(
                    f"  r@{K}: baseline={summary.get(f'baseline_r@{K}', 0):.3f} "
                    f"arch={summary.get(f'arch_r@{K}', 0):.3f} "
                    f"delta={summary.get(f'delta_r@{K}', 0):+.3f} "
                    f"W/T/L={summary.get(f'W/T/L_r@{K}', '?')}"
                )
            print(
                f"  avg_retrieved={summary.get('avg_total_retrieved', 0):.0f} "
                f"llm={summary.get('avg_llm_calls', 0):.1f} "
                f"embed={summary.get('avg_embed_calls', 0):.1f} "
                f"time={summary.get('avg_time_s', 0):.1f}s"
            )

            out = {
                "arch": arch_name,
                "conversation_length": length,
                "summary": summary,
                "category_breakdown": by_cat,
                "results": rows,
            }
            out_path = RESULTS_DIR / f"scaling_{arch_name}_{length}turns.json"
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"  saved {out_path}")

            all_summaries.setdefault(arch_name, {})[str(length)] = {
                "summary": summary,
                "category_breakdown": by_cat,
            }

    # Aggregated summary
    agg = RESULTS_DIR / "scaling_summary.json"
    with open(agg, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved aggregated summary: {agg}")

    # Final scaling table (delta r@20, delta r@50 by length per arch)
    print("\n" + "=" * 92)
    print("SCALING TABLE (delta r@20 | delta r@50 vs. cosine baseline)")
    print("=" * 92)
    header = (
        f"{'Arch':<22s}  "
        + "  ".join(f"{L:>6d}T" for L in lengths)
        + "      "
        + "  ".join(f"{L:>6d}T" for L in lengths)
    )
    print(f"{'Architecture':<22s}  " + "  ".join(f"{L:>7d}" for L in lengths))
    print("delta r@20:")
    for arch_name in arch_names:
        row = [arch_name]
        for L in lengths:
            s = all_summaries.get(arch_name, {}).get(str(L), {}).get("summary", {})
            d = s.get("delta_r@20")
            row.append(f"{d:+.3f}" if d is not None else "  --  ")
        print(f"  {row[0]:<20s}  " + "  ".join(f"{v:>7s}" for v in row[1:]))
    print("delta r@50:")
    for arch_name in arch_names:
        row = [arch_name]
        for L in lengths:
            s = all_summaries.get(arch_name, {}).get(str(L), {}).get("summary", {})
            d = s.get("delta_r@50")
            row.append(f"{d:+.3f}" if d is not None else "  --  ")
        print(f"  {row[0]:<20s}  " + "  ".join(f"{v:>7s}" for v in row[1:]))

    print("\nabsolute r@20:")
    for arch_name in arch_names:
        row = [arch_name]
        for L in lengths:
            s = all_summaries.get(arch_name, {}).get(str(L), {}).get("summary", {})
            v = s.get("arch_r@20")
            row.append(f"{v:.3f}" if v is not None else "  -- ")
        print(f"  {row[0]:<20s}  " + "  ".join(f"{v:>7s}" for v in row[1:]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--generate",
        action="store_true",
        help="Generate conversations and questions (no embedding).",
    )
    p.add_argument(
        "--embed", action="store_true", help="Embed generated data to npz files."
    )
    p.add_argument(
        "--run", action="store_true", help="Run retrieval eval across archs."
    )
    p.add_argument("--all", action="store_true", help="Generate + embed + run.")
    p.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[100, 250, 500, 1000, 2000],
        help="Conversation lengths to evaluate.",
    )
    p.add_argument(
        "--archs",
        type=str,
        nargs="+",
        default=list(ARCHITECTURES.keys()),
        help="Architectures to run.",
    )
    args = p.parse_args()

    if args.all:
        args.generate = True
        args.embed = True
        args.run = True

    if not any([args.generate, args.embed, args.run]):
        p.error("Specify at least one of --generate, --embed, --run, --all.")

    if args.generate:
        generate_all_data(args.lengths)
    if args.embed:
        embed_all_data(args.lengths)
    if args.run:
        run_eval(args.lengths, args.archs)


if __name__ == "__main__":
    main()
