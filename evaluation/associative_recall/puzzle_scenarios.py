"""Puzzle conversation scenarios for testing deep, chained retrieval.

Inspired by knowledge-gated exploration games (like Outer Wilds) where finding
clue A tells you what to search for to find clue B. Each conversation embeds
puzzle clues naturally in 80-120 turns of realistic chat.

Categories:
- logic_constraint: scattered constraints that must all be found and reconciled
- sequential_chain: information where clue A reveals what to search for in clue B
- state_change: values/situations that change multiple times across the conversation
- contradiction: conflicting information from different sources requiring resolution
- open_exploration: open-ended pattern recognition across scattered mentions
- absence_inference: answers inferred from patterns, never explicitly stated
"""

import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _t(turn_id: int, role: str, text: str) -> dict:
    return {"turn_id": turn_id, "role": role, "text": text}


# ---------------------------------------------------------------------------
# 1. Logic constraint conversations
# ---------------------------------------------------------------------------


def _build_logic_seating() -> list[dict]:
    """Logic puzzle 1: Office seating arrangement with scattered constraints."""
    return [
        _t(0, "user", "Morning! How's it going?"),
        _t(1, "assistant", "Good morning! Doing well. What can I help you with today?"),
        _t(
            2,
            "user",
            "Not much yet, just getting my coffee. Did you see the game last night?",
        ),
        _t(3, "assistant", "I didn't catch it! Who was playing?"),
        _t(
            4,
            "user",
            "Celtics vs Knicks. Went to overtime, it was wild. Anyway, I need to figure out this new desk assignment thing at work.",
        ),
        _t(5, "assistant", "Sounds like an exciting game! What's the desk situation?"),
        _t(
            6,
            "user",
            "So our office is moving to a new floor and we need to figure out where everyone sits. There are 6 desks in a row by the windows. The team is me, Alice, Bob, Carol, Dave, and Eve.",
        ),
        _t(
            7,
            "assistant",
            "Got it - 6 desks in a row, 6 people. Any preferences or constraints to start with?",
        ),
        _t(
            8,
            "user",
            "Let me think... I know I want to be near the whiteboard, which is at one end. But let me check with everyone first.",
        ),
        _t(9, "assistant", "Makes sense. Just let me know as preferences come in."),
        _t(
            10,
            "user",
            "Actually quick question first - do you know any good recipes for a slow cooker? I want to meal prep this weekend.",
        ),
        _t(
            11,
            "assistant",
            "Absolutely! A classic is pulled pork - you can do it with just a pork shoulder, BBQ sauce, and some seasonings. About 8 hours on low. Want details?",
        ),
        _t(
            12,
            "user",
            "Maybe later. Oh, I just got a message from Alice. She says she absolutely cannot sit next to Dave. They had that whole argument about the Henderson project and things are still tense.",
        ),
        _t(
            13,
            "assistant",
            "Understood. So Alice and Dave need at least one desk between them. That's constraint number one.",
        ),
        _t(
            14,
            "user",
            "Yeah. Hey, remind me to call the vet too. My dog has his annual checkup coming up.",
        ),
        _t(15, "assistant", "Will do! When is the checkup?"),
        _t(16, "user", "Next Tuesday at 3pm. Dr. Morrison at the Lakeside clinic."),
        _t(17, "assistant", "Got it noted down. Anything else on the desk situation?"),
        _t(
            18,
            "user",
            "Not yet. Let me tell you about this documentary I watched about octopuses though. Absolutely fascinating creatures.",
        ),
        _t(
            19,
            "assistant",
            "Oh I love a good nature documentary! What was it about specifically?",
        ),
        _t(
            20,
            "user",
            "It was about how they can change colors and textures, and apparently they're incredibly intelligent. They can solve puzzles and open jars.",
        ),
        _t(
            21,
            "assistant",
            "Their intelligence is remarkable. Some researchers think they might have a form of consciousness different from mammals.",
        ),
        _t(
            22,
            "user",
            "Wild stuff. Ok back to work things - I forgot to mention, the whiteboard is on the LEFT end of the row if you're facing the windows.",
        ),
        _t(
            23,
            "assistant",
            "So desk 1 is the leftmost, next to the whiteboard, and desk 6 is the rightmost. Got it.",
        ),
        _t(
            24,
            "user",
            "Right. And I want desk 1 since I use the whiteboard constantly for sketching out architectures.",
        ),
        _t(
            25,
            "assistant",
            "Makes sense. So you're in desk 1. That leaves desks 2-6 for Alice, Bob, Carol, Dave, and Eve.",
        ),
        _t(
            26,
            "user",
            "Correct. Let me check Slack... Bob just said he really wants a window seat with good natural light. He gets migraines under fluorescent lights.",
        ),
        _t(
            27,
            "assistant",
            "All the desks are by the windows though, right? Does he have a preference for a specific position?",
        ),
        _t(
            28,
            "user",
            "Good point, they're all window desks. But the ones at the ends get the most light because of how the building is angled. So desk 1 or desk 6 would be best for him.",
        ),
        _t(
            29,
            "assistant",
            "Since you're taking desk 1, that means Bob would prefer desk 6 for the best natural light. Want to tentatively put him there?",
        ),
        _t(
            30,
            "user",
            "Yeah, let's pencil that in. Oh, have you tried that new Thai place on Elm Street? My wife went with her book club.",
        ),
        _t(31, "assistant", "I haven't, but I've seen good reviews. Was it good?"),
        _t(
            32,
            "user",
            "She said the green curry was amazing but the pad thai was disappointing. Typical - it's always one or the other with Thai restaurants.",
        ),
        _t(
            33,
            "assistant",
            "Haha, that does seem to be the pattern. Should I add it to your restaurant list?",
        ),
        _t(
            34,
            "user",
            "Sure, under 'to try.' By the way, I need to figure out the rest of these desks. Carol mentioned something to me in the hallway earlier.",
        ),
        _t(35, "assistant", "What did Carol say?"),
        _t(
            36,
            "user",
            "She said she needs to be close to the server room door, which is on the right side of the room. She's the sysadmin so she goes in and out all day.",
        ),
        _t(
            37,
            "assistant",
            "That suggests Carol should be in one of the higher-numbered desks, like 5 or 6. But Bob is penciled in for desk 6. So desk 5 for Carol?",
        ),
        _t(38, "user", "That works. Let me text Dave and Eve and see if they care."),
        _t(
            39,
            "assistant",
            "Sounds good. We have you at 1, Bob at 6, and Carol at 5 so far. Desks 2, 3, and 4 are open for Alice, Dave, and Eve.",
        ),
        _t(
            40,
            "user",
            "I forgot to mention - we also need to figure out the lunch schedule for the office birthday celebrations this month. Three birthdays coming up.",
        ),
        _t(41, "assistant", "Oh nice! Whose birthdays?"),
        _t(
            42,
            "user",
            "Eve on the 15th, my boss Greg on the 22nd, and the new intern Zara on the 28th. But that's separate from the desk thing.",
        ),
        _t(
            43,
            "assistant",
            "Got it, we can tackle those after the desk arrangement. Any word from Dave or Eve?",
        ),
        _t(
            44,
            "user",
            "Eve just responded. She says she doesn't really care where she sits as long as she's not at either end. She finds the foot traffic at the ends distracting.",
        ),
        _t(
            45,
            "assistant",
            "So Eve prefers desks 2, 3, or 4 - middle positions. That works out since those are what's left for Alice, Dave, and Eve.",
        ),
        _t(
            46,
            "user",
            "Dave hasn't answered yet. He's probably in a meeting. Let me check the kitchen - I think someone brought donuts.",
        ),
        _t(47, "assistant", "Enjoy! Let me know when you hear from Dave."),
        _t(
            48,
            "user",
            "False alarm, no donuts. But I ran into Carol at the coffee machine and she mentioned something I forgot to tell you.",
        ),
        _t(49, "assistant", "What's that?"),
        _t(
            50,
            "user",
            "She said she and Eve work on the same sub-team and they pair-program a lot, so it'd be nice if they sat next to each other.",
        ),
        _t(
            51,
            "assistant",
            "That's useful. Carol is at desk 5, so if Eve sits at desk 4, they'd be adjacent. And desk 4 is a middle seat, which Eve wants. That could work perfectly.",
        ),
        _t(52, "user", "Oh that's neat. Yeah let's do that. Eve at 4, Carol at 5."),
        _t(
            53,
            "assistant",
            "Great. So now: You at 1, Eve at 4, Carol at 5, Bob at 6. That leaves desks 2 and 3 for Alice and Dave.",
        ),
        _t(54, "user", "Hold on though. Remember Alice can't be next to Dave?"),
        _t(
            55,
            "assistant",
            "Right! If Alice and Dave are in desks 2 and 3 respectively (or vice versa), they'd be adjacent. That violates the constraint from the Henderson project argument.",
        ),
        _t(56, "user", "Exactly. So we have a problem."),
        _t(
            57,
            "assistant",
            "We do. Let me think... the only way to separate Alice and Dave with only desks 2 and 3 left would require putting someone between them, but all other desks are assigned. We might need to rearrange.",
        ),
        _t(
            58,
            "user",
            "Ugh, this is getting complicated. Let me step back - maybe Bob doesn't HAVE to be at desk 6.",
        ),
        _t(
            59,
            "assistant",
            "True. If Bob moves to a different desk, it opens up more possibilities. The key constraint is Alice and Dave can't be adjacent.",
        ),
        _t(
            60,
            "user",
            "I just realized something. My monitor has a really bad glare problem at that end too. Let me swap with someone. Actually no, I need the whiteboard.",
        ),
        _t(
            61,
            "assistant",
            "Yeah, desk 1 makes the most sense for you with the whiteboard. Let's think about other arrangements.",
        ),
        _t(
            62,
            "user",
            "Ok let me regroup. Can you hold on, I need to handle something for my wife. She's asking about the school pickup schedule this week.",
        ),
        _t(63, "assistant", "Of course, take your time."),
        _t(
            64,
            "user",
            "Ok I'm back. So here's what I'm thinking about the pickup schedule - Monday I'll get the kids, Tuesday she will, we'll alternate the rest of the week. Actually that's not what I wanted to talk about, back to the desks.",
        ),
        _t(
            65,
            "assistant",
            "No worries! Where were we - we need to place Alice and Dave non-adjacently.",
        ),
        _t(
            66,
            "user",
            "Right. What if we put Alice at desk 2 and Dave at desk 4, and move Eve somewhere else?",
        ),
        _t(
            67,
            "assistant",
            "If Dave is at 4 and Carol at 5, that works for Alice-Dave separation. But then Eve needs desk 3 - which is still a middle desk, so she'd be happy.",
        ),
        _t(
            68,
            "user",
            "Wait, but if Dave is at desk 4 and Eve was supposed to be at desk 4... hmm.",
        ),
        _t(
            69,
            "assistant",
            "Right, we'd need to move Eve to desk 3. The question is whether Carol and Eve being one desk apart (3 and 5) still works for their pair programming.",
        ),
        _t(
            70,
            "user",
            "That's not ideal. Carol specifically said they need to be NEXT to each other, not just close.",
        ),
        _t(
            71,
            "assistant",
            "Got it. So Eve must be at 4 (next to Carol at 5), which means Dave and Alice share desks 2 and 3. But they can't be adjacent. We're stuck in the same loop.",
        ),
        _t(
            72,
            "user",
            "Wait, I just got a text from Dave. He says 'Hey, just a heads up - I talked to Alice at lunch and we actually cleared the air about the Henderson thing. We're good now. She apologized and I apologized. Put us wherever.'",
        ),
        _t(
            73,
            "assistant",
            "Well that changes everything! If Alice and Dave are fine sitting next to each other now, the constraint is resolved.",
        ),
        _t(
            74,
            "user",
            "Thank goodness for that. So we're back to the arrangement: me at 1, then Alice and Dave in 2 and 3 in either order, Eve at 4, Carol at 5, Bob at 6.",
        ),
        _t(
            75,
            "assistant",
            "Exactly. Does it matter which of Alice or Dave takes desk 2 vs 3?",
        ),
        _t(
            76,
            "user",
            "Hmm, I work with Alice more directly on the frontend, so having her at desk 2 right next to me would be great.",
        ),
        _t(
            77,
            "assistant",
            "Perfect. So the final arrangement is: 1-You, 2-Alice, 3-Dave, 4-Eve, 5-Carol, 6-Bob.",
        ),
        _t(
            78,
            "user",
            "Oh wait, one more thing. Eve just sent another message - she says she also wants to be near the office plants. Where are the plants?",
        ),
        _t(79, "assistant", "You haven't mentioned where the plants are. Do you know?"),
        _t(
            80,
            "user",
            "The big fiddle leaf fig is between desks 3 and 4. So desk 4 is literally right next to it.",
        ),
        _t(
            81,
            "assistant",
            "Then Eve at desk 4 is perfect for the plant proximity too. Everything aligns.",
        ),
        _t(
            82,
            "user",
            "Great. And actually you know what, Bob said something else I forgot. He said he's bringing a standing desk converter and the IT closet with the extra monitors is near desk 6 so that works out.",
        ),
        _t(
            83,
            "assistant",
            "Sounds like desk 6 is ideal for Bob for multiple reasons - best light AND close to IT resources.",
        ),
        _t(
            84,
            "user",
            "Alright, I think we're done with the desk thing. Let me write this up and send it to the team.",
        ),
        _t(
            85,
            "assistant",
            "Sounds good! The final arrangement: 1-You, 2-Alice, 3-Dave, 4-Eve, 5-Carol, 6-Bob.",
        ),
        _t(
            86,
            "user",
            "Perfect. Now about those birthday celebrations - can you help me plan Eve's? It's first since it's the 15th.",
        ),
        _t(
            87,
            "assistant",
            "Sure! What kind of celebration are you thinking? Lunch out, cake in the office, or something else?",
        ),
        _t(
            88,
            "user",
            "Just a cake and card. She mentioned she loves carrot cake, so let's get one from that bakery on Main.",
        ),
        _t(
            89,
            "assistant",
            "Carrot cake from the Main Street bakery for Eve's birthday on the 15th. Should I draft a card message?",
        ),
        _t(90, "user", "Nah, I'll handle that. Thanks for the help today."),
        _t(
            91,
            "assistant",
            "You're welcome! Good luck with the desk move and the birthday planning.",
        ),
    ]


def _build_logic_scheduling() -> list[dict]:
    """Logic puzzle 2: Conference room scheduling with overlapping needs."""
    return [
        _t(
            0,
            "user",
            "Hey, we need to sort out the conference room bookings for next week. It's a mess.",
        ),
        _t(1, "assistant", "Oh no, what happened?"),
        _t(
            2,
            "user",
            "Three teams all need the big conference room (Room A) on the same days, and there's only Room A and the smaller Room B available. Room C is being renovated.",
        ),
        _t(3, "assistant", "Got it. What are the three teams and their needs?"),
        _t(
            4,
            "user",
            "Marketing, Engineering, and Sales. Let me figure out the details. But first, did you hear about the new parking policy? Everyone's upset.",
        ),
        _t(5, "assistant", "No, what's the new policy?"),
        _t(
            6,
            "user",
            "They're charging $5 a day for parking now. Used to be free. People are furious. Anyway, back to the rooms.",
        ),
        _t(7, "assistant", "That does sound frustrating. So what does each team need?"),
        _t(
            8,
            "user",
            "Marketing has their weekly strategy session. It's usually Monday mornings, 9-11. They need the big room because they have 15 people.",
        ),
        _t(
            9,
            "assistant",
            "Room A for Marketing, Monday 9-11am. Room B wouldn't fit 15 people?",
        ),
        _t(
            10,
            "user",
            "Room B only fits 8 comfortably. So Marketing needs Room A. Oh speaking of which, let me get lunch. I'll be back in 20.",
        ),
        _t(11, "assistant", "Sure, enjoy your lunch!"),
        _t(
            12,
            "user",
            "Back. Had a burrito. Anyway, Engineering wants Room A for their sprint planning on Monday afternoon, 1-3pm, and their stand-up reviews on Wednesday and Friday mornings 9-10am.",
        ),
        _t(
            13,
            "assistant",
            "Engineering: Room A Monday 1-3pm, Wednesday 9-10am, Friday 9-10am. Any flexibility there?",
        ),
        _t(
            14,
            "user",
            "The sprint planning is set in stone - the whole team of 20 needs to be there. But the stand-ups could potentially go in Room B since it's just the 6 leads.",
        ),
        _t(
            15,
            "assistant",
            "Good, so the Wednesday and Friday stand-ups could move to Room B. That helps. What about Sales?",
        ),
        _t(
            16,
            "user",
            "Sales needs Room A for their big client presentation on Wednesday from 10am to 2pm. Like 12 people including the clients. It absolutely has to be Room A because they need the AV setup.",
        ),
        _t(17, "assistant", "Room A only has the AV setup?"),
        _t(
            18,
            "user",
            "Right, Room B just has a whiteboard and a screen. No proper projector or video conferencing rig. The clients are partly remote.",
        ),
        _t(
            19,
            "assistant",
            "So Sales has a hard requirement for Room A on Wednesday 10am-2pm. That doesn't overlap with Engineering's Wednesday stand-up at 9-10am, so both could use Room A on Wednesday at different times.",
        ),
        _t(
            20,
            "user",
            "True. But wait, I forgot - Lisa from Marketing told me they ALSO want to do a brainstorm session on Wednesday from 2-4pm. She just added it.",
        ),
        _t(
            21,
            "assistant",
            "Wednesday is getting busy for Room A: Engineering 9-10, Sales 10-2, Marketing 2-4. That actually works since they don't overlap!",
        ),
        _t(
            22,
            "user",
            "Oh nice, I didn't realize they lined up like that. But actually, Sales said their meeting might run over. Kevin said to block until 3pm just in case.",
        ),
        _t(
            23,
            "assistant",
            "If Sales needs until 3pm, then Marketing's 2-4pm brainstorm overlaps from 2-3pm. We'd need to adjust.",
        ),
        _t(24, "user", "Hmm. Can Marketing start at 3 instead?"),
        _t(
            25,
            "assistant",
            "They could do 3-5pm if they're flexible. Let me check - is 5pm too late?",
        ),
        _t(
            26,
            "user",
            "People leave at 5, so 3-4:30 would be better. Lisa works part time and leaves at 4:30.",
        ),
        _t(
            27,
            "assistant",
            "OK so Marketing brainstorm Wednesday 3-4:30pm in Room A. That works after Sales finishes by 3pm at latest.",
        ),
        _t(
            28,
            "user",
            "Actually you know what, let me call Lisa real quick about something else too. Be right back.",
        ),
        _t(29, "assistant", "Take your time!"),
        _t(
            30,
            "user",
            "Ok so I talked to Lisa. She says the Wednesday brainstorm is actually flexible on the day - they could do it Thursday instead. She just needs it sometime mid-week.",
        ),
        _t(
            31,
            "assistant",
            "Thursday would simplify Wednesday a lot. Is Room A free on Thursday?",
        ),
        _t(32, "user", "Thursday is wide open as of now. Nobody has anything booked."),
        _t(
            33,
            "assistant",
            "Great. So Marketing brainstorm moves to Thursday in Room A. What time works?",
        ),
        _t(
            34,
            "user",
            "Same as before, 2-4pm works for the team. Actually, Lisa just mentioned she'd prefer morning. Let's say 10-12.",
        ),
        _t(
            35,
            "assistant",
            "Marketing brainstorm Thursday 10am-12pm in Room A. That clears up Wednesday nicely.",
        ),
        _t(
            36,
            "user",
            "Good. Oh by the way, I need to remember to water my desk plants. The succulent is looking rough.",
        ),
        _t(
            37,
            "assistant",
            "Succulents can be tricky - they actually need less water than most people think. Once every week or two is usually enough.",
        ),
        _t(
            38,
            "user",
            "Yeah I've been over-watering it probably. Ok one more room conflict I forgot about.",
        ),
        _t(39, "assistant", "What's that?"),
        _t(
            40,
            "user",
            "HR has their monthly all-hands on Tuesday. They always use Room A, 10am-12pm. That's been on the books forever, it's non-negotiable.",
        ),
        _t(
            41,
            "assistant",
            "Good to know. So Tuesday morning Room A is HR's. Are any of our three teams affected?",
        ),
        _t(
            42,
            "user",
            "Not directly, but Engineering also wanted to do a code review session Tuesday afternoon if possible. Like 2-4pm.",
        ),
        _t(
            43,
            "assistant",
            "Tuesday 2-4pm in Room A should work since HR finishes at noon. How many people for the code review?",
        ),
        _t(44, "user", "Just 5 people, so Room B would actually be fine for that."),
        _t(
            45,
            "assistant",
            "Even better - that keeps Room A free Tuesday afternoon in case something else comes up.",
        ),
        _t(
            46,
            "user",
            "Smart. Let's do Room B for the code review. Is there anything left that conflicts?",
        ),
        _t(
            47,
            "assistant",
            "Let me lay it out. Room A: Monday 9-11 Marketing, Monday 1-3 Engineering sprint, Tuesday 10-12 HR, Wednesday 9-10 Engineering stand-up, Wednesday 10-3 Sales presentation, Thursday 10-12 Marketing brainstorm, Friday 9-10 Engineering stand-up. Room B: Tuesday 2-4 Engineering code review. Does that look right?",
        ),
        _t(
            48,
            "user",
            "Wait, we said the Engineering stand-ups could go in Room B since it's just 6 people.",
        ),
        _t(
            49,
            "assistant",
            "Right, good catch. So move Wednesday 9-10 and Friday 9-10 Engineering stand-ups to Room B.",
        ),
        _t(
            50,
            "user",
            "Yes. That frees Room A on those mornings before the other bookings.",
        ),
        _t(
            51,
            "assistant",
            "Updated. Room A is now a bit less packed. Room B has: Tuesday 2-4 code review, Wednesday 9-10 stand-up, Friday 9-10 stand-up.",
        ),
        _t(
            52,
            "user",
            "Oh crud, I just remembered. The Sales presentation - Kevin told me this morning that the clients rescheduled. They want to come Thursday now, not Wednesday.",
        ),
        _t(
            53,
            "assistant",
            "That's a big change. Thursday now has both the Sales presentation and the Marketing brainstorm. Let me check the times: Sales would presumably still be 10-3pm, and Marketing brainstorm was 10-12pm. That's a direct conflict.",
        ),
        _t(54, "user", "Yep. That's a problem."),
        _t(
            55,
            "assistant",
            "Options: Marketing brainstorm moves to a different time Thursday, or a different day entirely. Or Sales goes to a different day.",
        ),
        _t(
            56,
            "user",
            "Sales can't move - the clients picked Thursday and that's final. Let me check with Lisa about the brainstorm again.",
        ),
        _t(
            57,
            "assistant",
            "Sure. We could also consider Wednesday again now that Sales has vacated it.",
        ),
        _t(
            58,
            "user",
            "Oh right! If Sales isn't on Wednesday anymore, then Marketing could go back to Wednesday afternoon.",
        ),
        _t(
            59,
            "assistant",
            "Wednesday afternoon is wide open now. Marketing brainstorm Wednesday 2-4pm in Room A would work.",
        ),
        _t(
            60,
            "user",
            "Actually Lisa said she leaves at 4:30, not that the meeting ends at 4. Let me ask her about Wednesday afternoon again specifically.",
        ),
        _t(
            61,
            "assistant",
            "If it's 2-4pm, she'd still be there since she doesn't leave until 4:30. Does that work?",
        ),
        _t(
            62,
            "user",
            "She says yes, Wednesday 2-4 works for her and the team. Lock it in.",
        ),
        _t(
            63,
            "assistant",
            "Done. Marketing brainstorm back to Wednesday 2-4pm, Room A. And Sales presentation now Thursday 10-3pm, Room A.",
        ),
        _t(
            64,
            "user",
            "Perfect. I think that sorts everything out. Let me also mention - next month we might need to account for the new product team. They're hiring 8 people who start in three weeks.",
        ),
        _t(
            65,
            "assistant",
            "Good heads up. That'll add more demand for both rooms. Something to plan for.",
        ),
        _t(
            66,
            "user",
            "Yeah. For now though, can you give me the final schedule? I need to send it to facilities.",
        ),
        _t(
            67,
            "assistant",
            "Here's the final Room A schedule for next week: Monday 9-11am Marketing strategy, Monday 1-3pm Engineering sprint planning, Tuesday 10am-12pm HR all-hands, Wednesday 2-4pm Marketing brainstorm, Thursday 10am-3pm Sales client presentation, Friday is open. Room B: Tuesday 2-4pm Engineering code review, Wednesday 9-10am Engineering stand-up, Friday 9-10am Engineering stand-up. Want me to format that more formally?",
        ),
        _t(
            68,
            "user",
            "That's great. Actually, quick question - is Room A free Friday afternoon? I might want to do a team-building thing.",
        ),
        _t(
            69,
            "assistant",
            "Based on the schedule, Room A is completely free on Friday. The only Friday booking is the Engineering stand-up in Room B from 9-10am.",
        ),
        _t(
            70,
            "user",
            "Let me pencil in a team-building session Friday 2-4pm in Room A then. Nothing firm yet.",
        ),
        _t(
            71,
            "assistant",
            "Tentatively noted. Want me to add that to the schedule I'm sending?",
        ),
        _t(
            72,
            "user",
            "Mark it as tentative. Oh I also need to talk about the budget for the team event but that can wait.",
        ),
        _t(
            73,
            "assistant",
            "Sure thing. Here's the updated final schedule with the tentative Friday addition. Ready to send?",
        ),
        _t(74, "user", "Yep, send it. Thanks for untangling all of that."),
        _t(
            75,
            "assistant",
            "You're welcome! It was quite the puzzle with all the changes, but we got there.",
        ),
        _t(76, "user", "Story of my life with this office. See you later."),
        _t(77, "assistant", "Have a good rest of your day!"),
    ]


# ---------------------------------------------------------------------------
# 2. Sequential dependency chain conversations
# ---------------------------------------------------------------------------


def _build_chain_plant() -> list[dict]:
    """Chain 1: Plant health -> water quality -> mineral content -> filtration -> aquarium."""
    return [
        _t(
            0,
            "user",
            "Hey, can you help me figure out what's going on with my tomato plants? They look terrible.",
        ),
        _t(
            1,
            "assistant",
            "I'd be happy to help! What do they look like? Describe the symptoms.",
        ),
        _t(
            2,
            "user",
            "The leaves are getting these brownish-yellow edges, kind of crispy. And the growth has really slowed down. I've been watering them regularly.",
        ),
        _t(
            3,
            "assistant",
            "That sounds like it could be a nutrient deficiency or a watering issue. Are you using tap water or filtered water?",
        ),
        _t(
            4,
            "user",
            "Just straight from the tap. Should I be doing something different?",
        ),
        _t(
            5,
            "assistant",
            "Tap water is usually fine for most plants. The brown crispy edges could suggest salt burn from the soil, over-fertilizing, or inconsistent watering. How often are you watering?",
        ),
        _t(
            6,
            "user",
            "Every other day, about a cup of water per plant. Anyway, let me change the subject for a sec - have you seen the new season of that cooking show?",
        ),
        _t(
            7,
            "assistant",
            "I haven't, but I've heard it's really popular! Are you enjoying it?",
        ),
        _t(
            8,
            "user",
            "Yeah, they had this incredible episode about fermented foods. Made me want to try making kimchi.",
        ),
        _t(
            9,
            "assistant",
            "Kimchi is a great fermentation project for beginners! You need napa cabbage, Korean chili flakes, fish sauce, and some patience. Want a basic recipe?",
        ),
        _t(
            10,
            "user",
            "Maybe another time. My wife said the shower head has been getting these white crusty deposits lately. I had to scrub it off with vinegar.",
        ),
        _t(
            11,
            "assistant",
            "Those white deposits are mineral buildup - usually calcium and magnesium. That's a sign of hard water. It's common and not harmful, but it can be annoying on fixtures.",
        ),
        _t(
            12,
            "user",
            "Huh. We didn't used to have that problem. We've only been in this house for 6 months.",
        ),
        _t(
            13,
            "assistant",
            "The previous house might have had a water softener, or the water source might be different in your new area. You could check with your local water utility about recent changes in mineral content.",
        ),
        _t(
            14,
            "user",
            "Good idea. My neighbor Karen mentioned the city switched to a different reservoir last year or something. I didn't think much of it at the time.",
        ),
        _t(
            15,
            "assistant",
            "That could explain it! Different reservoirs can have very different mineral profiles. A reservoir switch could increase calcium, magnesium, and other dissolved minerals in the tap water.",
        ),
        _t(
            16,
            "user",
            "I wonder if that's connected to the plant problem. Like, could the hard water be hurting them?",
        ),
        _t(
            17,
            "assistant",
            "Actually, yes - that's a great connection to make. High mineral content in water can cause salt buildup in soil, which is exactly what causes those brown, crispy leaf edges you described. The minerals accumulate in the soil over time and essentially burn the roots.",
        ),
        _t(
            18,
            "user",
            "Wow, really? I just assumed it was a fertilizer issue or something. So what should I do?",
        ),
        _t(
            19,
            "assistant",
            "You have a few options: you could flush the soil with distilled water periodically to wash out mineral buildup, use a water filter, or collect rainwater for your plants. A basic carbon filter won't remove minerals though - you'd need a reverse osmosis system or a water softener.",
        ),
        _t(
            20,
            "user",
            "That seems like overkill for some tomato plants. What about the cheaper options?",
        ),
        _t(
            21,
            "assistant",
            "Rainwater collection is basically free if you set up a barrel. Or you could buy distilled water at the store and use that just for the plants. The flushing method - pour 3x the pot volume of clean water through to leach out salts - can help too.",
        ),
        _t(
            22,
            "user",
            "Ok let me think about that. Hey, random question - do you know if there's a good auto mechanic near downtown? My check engine light came on.",
        ),
        _t(
            23,
            "assistant",
            "I don't have specific local recommendations, but you could check reviews on Google Maps or ask on your local neighborhood app. What kind of car do you have?",
        ),
        _t(
            24,
            "user",
            "2019 Honda Civic. It's probably nothing serious but I should get it checked. The light came on right after I got gas, might just be the gas cap.",
        ),
        _t(
            25,
            "assistant",
            "That's actually a really common cause! Try tightening the gas cap and driving for a day or two - the light often resets on its own if that was the issue.",
        ),
        _t(
            26,
            "user",
            "I'll try that, thanks. So I asked my neighbor Karen about her garden. Her tomatoes look amazing and she waters from the same tap.",
        ),
        _t(
            27,
            "assistant",
            "Interesting! Does she do anything differently with her water or soil?",
        ),
        _t(
            28,
            "user",
            "She said she runs her water through a Brita-style pitcher first. But you said those don't remove minerals?",
        ),
        _t(
            29,
            "assistant",
            "Standard Brita filters use activated carbon which mainly removes chlorine, some organic compounds, and improves taste. They don't significantly reduce hardness minerals. However, some newer pitcher filters do include ion exchange resins that can soften water.",
        ),
        _t(
            30,
            "user",
            "I asked her and she said it's a ZeroWater pitcher actually, not Brita. She swears by it.",
        ),
        _t(
            31,
            "assistant",
            "ZeroWater uses a 5-stage filtration that includes ion exchange, which DOES remove dissolved minerals. That's the key difference. Her plants are essentially getting demineralized water while yours are getting the full mineral load from the tap.",
        ),
        _t(
            32,
            "user",
            "Mystery solved! I need to get one of those. Are they expensive?",
        ),
        _t(
            33,
            "assistant",
            "Around $30-40 for the pitcher. The replacement filters are about $15 each and last 2-4 months depending on use. Definitely cheaper than a whole-house RO system.",
        ),
        _t(
            34,
            "user",
            "I'll order one today. Let me ask you something else - my daughter's school project is due next week. She's doing something about the water cycle.",
        ),
        _t(
            35,
            "assistant",
            "Fun topic! What grade is she in? That'll help me gauge the right level of detail.",
        ),
        _t(
            36,
            "user",
            "4th grade. She needs a poster with a diagram and a short explanation. Nothing too fancy.",
        ),
        _t(
            37,
            "assistant",
            "For 4th grade, a nice colorful diagram showing evaporation, condensation, precipitation, and collection should be perfect. You could even tie in your own experience with different water sources!",
        ),
        _t(
            38,
            "user",
            "Ha, that's actually a cute idea. She could talk about how water quality changes depending on the source.",
        ),
        _t(
            39,
            "assistant",
            "That would make it a standout project. She could mention how water from different reservoirs has different mineral content, which you've learned firsthand.",
        ),
        _t(
            40,
            "user",
            "Speaking of which, I got the ZeroWater filter. Been using it for about a week now on the plants. They're already looking better.",
        ),
        _t(
            41,
            "assistant",
            "That's great news! The leaf tips might not recover since the damage is done on those, but new growth should come in healthy without the mineral burn.",
        ),
        _t(
            42,
            "user",
            "Yeah, I can see new bright green growth on a couple of them. Really happy about that. Anyway, what should I make for dinner tonight?",
        ),
        _t(
            43,
            "assistant",
            "How about something with your tomatoes? If they're producing, a fresh caprese salad or bruschetta would be perfect.",
        ),
        _t(
            44,
            "user",
            "They won't be ripe for a while yet! But good idea for later. I was thinking more like a stir fry.",
        ),
        _t(
            45,
            "assistant",
            "A chicken stir fry is always solid. Soy sauce, garlic, ginger, and whatever veggies you have on hand. Serve over rice.",
        ),
        _t(
            46,
            "user",
            "Perfect, I think we have everything for that. Oh by the way, I'm setting up a little tropical fish tank for my daughter. We bought a 20-gallon setup last weekend.",
        ),
        _t(
            47,
            "assistant",
            "How exciting! Tropical fish are great for kids. Have you started cycling the tank yet?",
        ),
        _t(
            48,
            "user",
            "Yeah, we filled it up a few days ago and the filter is running. I added some dechlorinator. We're going to wait a couple weeks before adding fish.",
        ),
        _t(
            49,
            "assistant",
            "Smart approach. Cycling is important for establishing the beneficial bacteria. What kind of fish are you planning?",
        ),
        _t(
            50,
            "user",
            "My daughter wants neon tetras and some guppies. Maybe a snail or two.",
        ),
        _t(
            51,
            "assistant",
            "Great choices for beginners! Neon tetras and guppies are both hardy and colorful. They get along well together too.",
        ),
        _t(
            52,
            "user",
            "Awesome. So the tank has been running for about a week now but the water is looking kind of cloudy. Is that normal?",
        ),
        _t(
            53,
            "assistant",
            "Slight cloudiness in the first week can be a bacterial bloom, which is actually normal during cycling. It usually clears up on its own in a few days. What kind of cloudiness - white/milky or green?",
        ),
        _t(
            54,
            "user",
            "More like a whitish haze. Not green. But it's not clearing up, it's been like this for 4 days now.",
        ),
        _t(
            55,
            "assistant",
            "Persistent white cloudiness could be a bacterial bloom that's taking longer to resolve, or it could be something in the water itself. What water are you using to fill the tank?",
        ),
        _t(
            56,
            "user",
            "Just tap water with dechlorinator. Should I be doing something else?",
        ),
        _t(
            57,
            "assistant",
            "Given what you've discovered about your tap water recently, you might want to consider whether the high mineral content could be playing a role here too. Dissolved minerals can contribute to persistent cloudiness in aquariums and can also affect the pH and hardness levels that tropical fish are sensitive to.",
        ),
        _t(
            58,
            "user",
            "Oh man, I didn't even think of that. The same water problem that was killing my tomatoes could mess up the fish tank?",
        ),
        _t(
            59,
            "assistant",
            "Exactly. Neon tetras in particular prefer soft, slightly acidic water. If your tap water is mineral-heavy, the high hardness could stress them. I'd recommend testing the water hardness with a kit and potentially using filtered water for the tank too.",
        ),
        _t(
            60,
            "user",
            "I should test the tank water with those strips. Can I use the ZeroWater filter for the fish tank too?",
        ),
        _t(
            61,
            "assistant",
            "You can, but be careful - fully demineralized water isn't ideal for fish either. They need some minerals. The best approach for aquariums is to mix filtered and tap water to reach the target hardness for your fish species. For neon tetras, you'd want moderate softness, around 2-10 dGH.",
        ),
        _t(
            62,
            "user",
            "This is more complicated than I thought. Ok I'll get a test kit and figure it out before we add any fish.",
        ),
        _t(
            63,
            "assistant",
            "That's the right approach. Test first, then adjust. The cycling period gives you time to dial in the water parameters before any fish are at risk.",
        ),
        _t(
            64,
            "user",
            "Thanks. My daughter is going to be impatient but I'd rather do it right.",
        ),
        _t(
            65,
            "assistant",
            "Better to wait and do it right than to rush and lose fish. She'll appreciate healthy, happy fish more than fast fish.",
        ),
        _t(
            66,
            "user",
            "True. Hey, I also want to ask about something for work but that can wait till tomorrow. I'm logging off for the night.",
        ),
        _t(
            67,
            "assistant",
            "Sounds good! Good luck with the fish tank water testing. Talk to you tomorrow.",
        ),
        _t(
            68,
            "user",
            "Morning! So yesterday I was thinking more about the water issue. I checked our city's water quality report online.",
        ),
        _t(69, "assistant", "Oh interesting! What did it say?"),
        _t(
            70,
            "user",
            "Total dissolved solids are at 450 ppm, which seems high. The report says the acceptable range is up to 500 ppm but the old reservoir was around 200 ppm.",
        ),
        _t(
            71,
            "assistant",
            "450 ppm is more than double the old level. That's a significant increase in dissolved minerals. It explains everything - the shower deposits, the plant damage, and likely the aquarium cloudiness too.",
        ),
        _t(
            72,
            "user",
            "Yeah. I'm going to write a letter to the city council about this. Other neighbors are complaining too. Karen started a petition.",
        ),
        _t(
            73,
            "assistant",
            "That's proactive. A high TDS level isn't necessarily a health concern for people, but it affects appliances, plants, and aquatic life. Worth raising with the city.",
        ),
        _t(
            74,
            "user",
            "Agreed. Alright, I had that work question. We're doing a database migration and I need to understand partitioning strategies. Can we talk about that?",
        ),
        _t(
            75,
            "assistant",
            "Sure! Database partitioning - are you looking at horizontal partitioning (sharding) or vertical partitioning?",
        ),
        _t(
            76,
            "user",
            "Horizontal. We have a table with 200 million rows and queries are getting slow. But I'll save the details for another time - I just realized I have a meeting in 5 minutes.",
        ),
        _t(
            77,
            "assistant",
            "No worries, we can pick that up anytime. Good luck with the meeting!",
        ),
        _t(
            78,
            "user",
            "Thanks! Oh, quick update - I tested the aquarium water. Hardness is 18 dGH. Way too high for neon tetras right?",
        ),
        _t(
            79,
            "assistant",
            "Yes, 18 dGH is quite hard. Neon tetras prefer 2-10 dGH. You'll definitely want to mix in some filtered water to bring that down. I'd aim for about 6-8 dGH for a community tank with tetras and guppies.",
        ),
        _t(
            80,
            "user",
            "I'll do a partial water change with ZeroWater filtered water and retest. Thanks for connecting all these dots - I never would have realized the tomato problem and the fish tank issue were related.",
        ),
        _t(
            81,
            "assistant",
            "It all traces back to the same root cause! The reservoir switch changed your water chemistry and it cascaded to everything that uses your tap water. Good detective work on your part for noticing all the symptoms.",
        ),
        _t(
            82,
            "user",
            "Ha, I think you deserve the credit for that. Alright, off to my meeting. Talk later!",
        ),
        _t(83, "assistant", "Good luck! Let me know how the water change goes."),
    ]


def _build_chain_recipe() -> list[dict]:
    """Chain 2: Forgotten family recipe -> grandmother's journal -> spice market ->
    specific ingredient -> cooking technique. Uses different vocabulary at each link."""
    return [
        _t(
            0,
            "user",
            "Hey, I've been trying to recreate my grandmother's lamb stew but it never tastes right. It's driving me crazy.",
        ),
        _t(
            1,
            "assistant",
            "That's frustrating! What do you remember about her version? Sometimes the smallest detail makes all the difference.",
        ),
        _t(
            2,
            "user",
            "It had this really complex, slightly smoky warmth to it. Not like regular stew. She made it every Sunday when I was a kid. She called it her 'Sunday pot.'",
        ),
        _t(
            3,
            "assistant",
            "That 'smoky warmth' is a good clue. Could be a specific spice or a cooking technique. Did she use a particular kind of pot or cooking method?",
        ),
        _t(
            4,
            "user",
            "She always used this big cast iron Dutch oven. But I have one too and it doesn't taste the same. I think it's a missing ingredient.",
        ),
        _t(
            5,
            "assistant",
            "Cast iron can add some flavor over decades of seasoning, but you're probably right that it's a specific ingredient. Did she ever share the recipe?",
        ),
        _t(
            6,
            "user",
            "Never. She kept it in her head. She passed away 5 years ago and the recipe went with her. My mom doesn't know it either.",
        ),
        _t(
            7,
            "assistant",
            "I'm sorry for your loss. Have you tried asking other family members? Sometimes aunts, cousins, or family friends watched the cooking process.",
        ),
        _t(
            8,
            "user",
            "Good idea. Let me switch topics though - I need to plan my vacation for August. Thinking about hiking in Colorado.",
        ),
        _t(
            9,
            "assistant",
            "Colorado in August is beautiful! Rocky Mountain National Park is a classic choice. What kind of hiking are you into - day hikes or multi-day backpacking?",
        ),
        _t(
            10,
            "user",
            "Day hikes mostly. My knees aren't what they used to be. Maybe something moderate with great views.",
        ),
        _t(
            11,
            "assistant",
            "Bear Lake to Emerald Lake is a classic moderate hike with stunning views. About 3.6 miles round trip. Or if you want something slightly longer, Sky Pond is incredible.",
        ),
        _t(
            12,
            "user",
            "I'll look into those. Actually, I was cleaning out my parents' attic last weekend and I found something interesting.",
        ),
        _t(13, "assistant", "What did you find?"),
        _t(
            14,
            "user",
            "An old leather journal that belonged to my grandmother. It's mostly in Farsi - she was Iranian. She immigrated here in the 70s.",
        ),
        _t(15, "assistant", "What a treasure! Can you read Farsi?"),
        _t(
            16,
            "user",
            "Not well, but my cousin Dariush can. He's going to help me translate some of it. There are a few pages that look like they have recipes.",
        ),
        _t(
            17,
            "assistant",
            "That could be the breakthrough you need! If her Sunday stew recipe is in there, you might finally crack the mystery ingredient.",
        ),
        _t(
            18,
            "user",
            "That's what I'm hoping. Dariush is coming over this weekend. Can we talk about something else while I wait? I need to buy a new laptop.",
        ),
        _t(
            19,
            "assistant",
            "Sure! What are you looking to use it for? Work, creative stuff, gaming?",
        ),
        _t(
            20,
            "user",
            "Mostly work - spreadsheets, email, video calls. Nothing intense. Budget is about $800.",
        ),
        _t(
            21,
            "assistant",
            "A MacBook Air M3 would be perfect for that if you're in the Apple ecosystem. Otherwise, a ThinkPad T14s or Dell Latitude are solid Windows options in that range.",
        ),
        _t(
            22,
            "user",
            "I'll check those out. Hey, so Dariush translated some of the journal entries! He sent me photos of his notes.",
        ),
        _t(23, "assistant", "Exciting! Did you find the stew recipe?"),
        _t(
            24,
            "user",
            "Sort of! She listed some of the ingredients but it's not a complete recipe with measurements. Just notes to herself. But one thing jumped out - she mentions something called 'golpar.'",
        ),
        _t(
            25,
            "assistant",
            "Golpar! That's Persian hogweed seed, also known as angelica seed. It's a very distinctive Iranian spice with a warm, musky, slightly bitter flavor. It's quite uncommon outside of Iranian cooking.",
        ),
        _t(26, "user", "I've never even heard of it. Could that be the missing piece?"),
        _t(
            27,
            "assistant",
            "It could absolutely be a key component of that 'smoky warmth' you remember. Golpar is often used in Iranian stews and bean dishes. It has a unique earthy depth that you can't replicate with common Western spices.",
        ),
        _t(
            28,
            "user",
            "Where would I even find that? I don't think my regular grocery store carries it.",
        ),
        _t(
            29,
            "assistant",
            "You'd need to look at a Middle Eastern or Persian grocery store. If you don't have one nearby, you can order it online from specialty spice retailers. It's sometimes sold as 'angelica powder' or 'Persian hogweed.'",
        ),
        _t(
            30,
            "user",
            "There's actually a Persian market across town that my aunt goes to. Sahel Market I think. I'll try there.",
        ),
        _t(
            31,
            "assistant",
            "Perfect. While you're there, you might also look for saffron and dried limes (limoo amani) - those are also common in Iranian stews and could be other elements you're missing.",
        ),
        _t(
            32,
            "user",
            "Wait, dried limes? That rings a bell. I think I remember her putting these dark round things in the pot. I asked what they were once and she said something in Farsi that I didn't understand.",
        ),
        _t(
            33,
            "assistant",
            "That could absolutely be dried limes! They're small, dark, hard spheres. When cooked in stew, they add a tangy, slightly fermented depth that's very characteristic of Persian cuisine. You either pierce them or crack them before adding to the pot.",
        ),
        _t(
            34,
            "user",
            "Yes! That's them! I remember her cracking them against the counter. I just thought they were some kind of nut. Oh my god, this is all coming together.",
        ),
        _t(
            35,
            "assistant",
            "This is great progress! The combination of golpar and dried limes would create exactly that complex, layered flavor you've been missing. Regular Western stew recipes wouldn't include either.",
        ),
        _t(
            36,
            "user",
            "I'm getting emotional about this honestly. I feel like I'm getting a piece of her back. Ok let me calm down. What else did the journal say?",
        ),
        _t(37, "assistant", "Take your time. What other notes did Dariush translate?"),
        _t(
            38,
            "user",
            "She mentions toasting the spices in oil before adding the meat. Dariush said the exact phrase was something like 'bloom the aromatics in hot fat until the kitchen sings.'",
        ),
        _t(
            39,
            "assistant",
            "That's a beautiful description of what's called 'blooming' spices - heating them in oil to release their essential oils and deepen their flavor. It's a technique used in many cuisines but it's especially important in Persian cooking where you're using whole spices.",
        ),
        _t(
            40,
            "user",
            "She always made the whole house smell amazing. I think that's the technique I've been skipping - I just throw everything in the pot together.",
        ),
        _t(
            41,
            "assistant",
            "That blooming step could make a massive difference. The heat activates compounds in the spices that wouldn't be released from just simmering in liquid. For golpar especially, dry-toasting or oil-blooming would transform its flavor.",
        ),
        _t(
            42,
            "user",
            "Ok so the chain is: get golpar and dried limes from Sahel Market, bloom the spices in oil first, then build the stew in the Dutch oven. I'm going to try this Sunday.",
        ),
        _t(
            43,
            "assistant",
            "That sounds like a perfect plan. You might want to do a test batch first since you don't have exact measurements. Start with less golpar - it's potent.",
        ),
        _t(
            44,
            "user",
            "Good advice. Let me ask you about something completely different. I need to renew my car registration.",
        ),
        _t(
            45,
            "assistant",
            "Usually you can do that online through your state's DMV website. Do you have the renewal notice?",
        ),
        _t(
            46,
            "user",
            "Yeah it came in the mail. I'll do it online tonight. Anyway, I went to Sahel Market today!",
        ),
        _t(47, "assistant", "Oh great! How was it? Did you find what you needed?"),
        _t(
            48,
            "user",
            "It was incredible. The owner, this guy named Hamid, was super helpful when I told him what I was looking for. He found me the golpar right away, and the dried limes.",
        ),
        _t(49, "assistant", "That's wonderful! Did Hamid have any cooking tips?"),
        _t(
            50,
            "user",
            "Actually yes. When I described the stew to him, he said it sounds like khoresht-e ghormeh sabzi or maybe a variation. He said the key most people miss is that you need to cook the herbs LOW AND SLOW for a really long time. Like, the stew should simmer for 3-4 hours minimum.",
        ),
        _t(
            51,
            "assistant",
            "That's a critical detail. Persian stews are typically slow-cooked to develop deep, melded flavors. The long cooking time also allows the dried limes to fully break down and infuse the liquid.",
        ),
        _t(
            52,
            "user",
            "I've been doing like 90 minutes. No wonder it tasted flat compared to hers.",
        ),
        _t(
            53,
            "assistant",
            "Going from 90 minutes to 3-4 hours will transform the dish. Between the golpar, the dried limes, the spice blooming technique, and the extended cooking time, you're rebuilding the full picture.",
        ),
        _t(
            54,
            "user",
            "I'm so excited for Sunday. Oh but first I need to deal with my work presentation on Friday. Can you help me organize my slides?",
        ),
        _t(55, "assistant", "Of course! What's the presentation about?"),
        _t(
            56,
            "user",
            "Quarterly revenue projections. Nothing exciting but my boss wants clean visuals. There are 20 slides.",
        ),
        _t(
            57,
            "assistant",
            "For clean revenue projections, I'd suggest: start with an executive summary slide, then revenue by segment, then by region, then trends over time, and end with key takeaways. Keep charts simple - no more than 5 data series per chart.",
        ),
        _t(
            58,
            "user",
            "That's helpful. I'll work on those this week. Thanks for all the help with the stew mystery. I feel like a detective who just cracked the case.",
        ),
        _t(
            59,
            "assistant",
            "It really was like detective work! From the journal discovery to identifying golpar, to the dried limes memory, to the cooking technique. Each clue led to the next. I hope Sunday's attempt is a success.",
        ),
        _t(60, "user", "I'll report back! Have a good evening."),
        _t(
            61,
            "assistant",
            "Looking forward to hearing how it goes. Enjoy the rest of your evening!",
        ),
        _t(
            62,
            "user",
            "THE STEW WAS PERFECT. I literally cried. It tasted exactly like hers.",
        ),
        _t(
            63,
            "assistant",
            "That's absolutely wonderful! What a special moment. Was it the combination of everything - the golpar, the dried limes, the blooming, and the long cook time?",
        ),
        _t(
            64,
            "user",
            "All of it together. The golpar was the biggest single revelation but the dried limes brought back so many childhood memories when I smelled them. And the long cooking time made everything meld perfectly. My mom tasted it and started crying too.",
        ),
        _t(
            65,
            "assistant",
            "That is incredibly touching. You've essentially recovered a family treasure through a chain of discoveries. The journal led to the golpar, which led to the spice market, where the cooking technique was revealed. It all came together beautifully.",
        ),
        _t(
            66,
            "user",
            "I wrote down everything with measurements this time. The recipe lives on.",
        ),
        _t(
            67,
            "assistant",
            "Your grandmother would be so proud. That's a legacy worth preserving.",
        ),
    ]


# ---------------------------------------------------------------------------
# 3. State change tracking conversations
# ---------------------------------------------------------------------------


def _build_state_budget() -> list[dict]:
    """State change 1: Project budget that shifts multiple times."""
    return [
        _t(
            0, "user", "Good morning! We need to talk about the new mobile app project."
        ),
        _t(1, "assistant", "Morning! Sure, what's the latest on it?"),
        _t(
            2,
            "user",
            "We just got the green light from leadership. They've allocated $50,000 for the whole thing.",
        ),
        _t(
            3,
            "assistant",
            "Congrats! $50K for a mobile app - is that for development only or does it include marketing too?",
        ),
        _t(
            4,
            "user",
            "Everything. Dev, design, testing, and a small marketing push at launch. It's tight but doable.",
        ),
        _t(5, "assistant", "That is tight. How are you planning to split it up?"),
        _t(
            6,
            "user",
            "I'm thinking $30K for development, $10K for design, $5K for testing/QA, and $5K for marketing. But those are rough numbers.",
        ),
        _t(
            7,
            "assistant",
            "Seems reasonable as a starting point. When's the target launch?",
        ),
        _t(
            8,
            "user",
            "End of Q3. So about 5 months from now. My boss Sandra wants a demo by month 3.",
        ),
        _t(
            9,
            "assistant",
            "Ambitious but feasible. Are you building in-house or contracting out?",
        ),
        _t(
            10,
            "user",
            "Mix of both. Our in-house team will handle the backend, and we're hiring a contractor for the iOS frontend. The Android version comes later.",
        ),
        _t(
            11,
            "assistant",
            "That makes sense for phasing. iOS first, then Android. Contractor costs will be a big chunk of that $30K dev budget.",
        ),
        _t(
            12,
            "user",
            "Actually, speaking of that, I need your advice on something personal. My sister is thinking about going back to school for nursing.",
        ),
        _t(
            13,
            "assistant",
            "That's a big decision! Nursing is a great field. What does she want to know?",
        ),
        _t(
            14,
            "user",
            "She's wondering about the time commitment. She has two kids and works part-time.",
        ),
        _t(
            15,
            "assistant",
            "Most nursing programs are 2-4 years depending on whether she's going for an ADN or BSN. Some programs offer evening or weekend tracks specifically for working adults.",
        ),
        _t(
            16,
            "user",
            "I'll pass that along. Ok back to the budget - good news! Sandra just told me that the VP was impressed with our proposal and approved an additional $20,000. We're now at $70K total.",
        ),
        _t(
            17,
            "assistant",
            "Excellent! That extra $20K gives you a lot more breathing room. Where will you allocate it?",
        ),
        _t(
            18,
            "user",
            "I'm thinking $10K more to development to get a better contractor, $5K more to marketing, and $5K as a contingency fund.",
        ),
        _t(
            19,
            "assistant",
            "Smart to add a contingency. So updated budget: $40K dev, $10K design, $5K QA, $10K marketing, $5K contingency. Total $70K.",
        ),
        _t(
            20,
            "user",
            "That's right. Feeling much better about this. Let me grab some coffee and we'll continue.",
        ),
        _t(21, "assistant", "Sure thing!"),
        _t(
            22,
            "user",
            "I'm back. So I was interviewing contractors yesterday. Found a great iOS developer named Miguel. His rate is $150/hour and he estimates 200 hours of work.",
        ),
        _t(
            23,
            "assistant",
            "That's $30,000 just for Miguel. That eats up 75% of the $40K dev budget and all of the original $30K.",
        ),
        _t(
            24,
            "user",
            "Yeah, he's not cheap but he's really good. Former Apple engineer. I think it's worth it. The in-house backend work is covered by existing salaries so it doesn't hit this budget.",
        ),
        _t(
            25,
            "assistant",
            "Good point - if the backend is salaried employees, then the $40K dev budget is mainly for Miguel and any other external costs. $30K for Miguel leaves $10K for other dev expenses.",
        ),
        _t(
            26,
            "user",
            "Exactly. I need to buy some development tools and API subscriptions too. Probably $3-4K there.",
        ),
        _t(
            27,
            "assistant",
            "That leaves $6-7K of the dev budget as buffer. Manageable.",
        ),
        _t(
            28,
            "user",
            "Hey, random thing - my kid's school is having a science fair. He wants to build a volcano. Classic.",
        ),
        _t(
            29,
            "assistant",
            "The baking soda and vinegar volcano! A rite of passage. You can make it more impressive with dish soap for extra foam and food coloring for the lava effect.",
        ),
        _t(
            30,
            "user",
            "He wants to do something more advanced. Apparently his friend did a Tesla coil last year and now he wants to top that.",
        ),
        _t(
            31,
            "assistant",
            "Ambitious kid! A simple electromagnetic motor or a solar-powered car would be impressive but safer than a Tesla coil for a school setting.",
        ),
        _t(
            32,
            "user",
            "Good ideas. I'll suggest those. OK so bad news on the project front. The vendor we were going to use for the push notification service just doubled their pricing. It was supposed to be $2K per year, now it's $4K.",
        ),
        _t(
            33,
            "assistant",
            "Ouch. That's an extra $2K you weren't planning for. Does it come out of the dev budget or somewhere else?",
        ),
        _t(
            34,
            "user",
            "It has to come out of the contingency fund. So contingency drops from $5K to $3K. I'm annoyed but it is what it is.",
        ),
        _t(
            35,
            "assistant",
            "At least you had the contingency for exactly this kind of thing. Still have $3K left for other surprises.",
        ),
        _t(
            36,
            "user",
            "Right. And then there's another issue. Sandra said we need to cut $15K from the total budget because the company's overall Q2 numbers came in lower than expected. Everyone's getting trimmed.",
        ),
        _t(
            37,
            "assistant",
            "That's a significant cut. $70K minus $15K brings you down to $55K. Where will you make the cuts?",
        ),
        _t(
            38,
            "user",
            "I'm going to have to reduce the marketing budget and maybe renegotiate with Miguel. Let me figure this out.",
        ),
        _t(39, "assistant", "Want to walk through the options?"),
        _t(
            40,
            "user",
            "Sure. So I think we cut marketing from $10K down to $4K, that saves $6K. We cut the contingency entirely - use what's left of it to cover the notification vendor and eliminate the line item. That saves the remaining $3K. And I'll ask Miguel to reduce scope to 175 hours instead of 200, saving $3,750.",
        ),
        _t(
            41,
            "assistant",
            "So that's $6K + $3K + $3,750 = $12,750 in cuts. You need $15K. Still short by $2,250.",
        ),
        _t(
            42,
            "user",
            "Hmm. Ok, I'll cut the design budget by $2,250. Take it from $10K to $7,750. We'll have the junior designer do more of the work instead of the senior freelancer.",
        ),
        _t(
            43,
            "assistant",
            "That gets you to $15K in cuts. New budget: Dev $36,250 ($30K Miguel at 175hrs + rest), Design $7,750, QA $5K, Marketing $4K, Contingency $0, Vendor $4K. Wait, the vendor cost was already in contingency?",
        ),
        _t(
            44,
            "user",
            "The vendor cost I'll just absorb into the dev tools line. It's a dev expense. So dev is $36,250 for Miguel, $3-4K for tools including the push notification vendor. The $40K dev line gets cut to about $40K minus... actually let me just redo the whole thing properly.",
        ),
        _t(45, "assistant", "Good idea. Let's recalculate from the new total of $55K."),
        _t(
            46,
            "user",
            "OK: Development $35,250 (Miguel at 175 hrs = $26,250, plus tools/APIs/vendor = $9,000). Design $7,750. QA $5,000. Marketing $4,000. No contingency. Adds up to $52,000. Wait that's not right...",
        ),
        _t(
            47,
            "assistant",
            "$35,250 + $7,750 + $5,000 + $4,000 = $52,000. You have $55K, so there's $3K unallocated.",
        ),
        _t(
            48,
            "user",
            "Oh good, I miscounted. Let me put that $3K back into marketing. So marketing is $7K now. Total: $35,250 + $7,750 + $5,000 + $7,000 = $55,000. Perfect.",
        ),
        _t(
            49,
            "assistant",
            "That's a much cleaner breakdown. Development $35,250, Design $7,750, QA $5,000, Marketing $7,000. Total $55,000.",
        ),
        _t(
            50,
            "user",
            "Great. I need to present this to Sandra tomorrow. Hopefully no more cuts.",
        ),
        _t(
            51,
            "assistant",
            "Fingers crossed. The budget has been through quite a journey already.",
        ),
        _t(
            52,
            "user",
            "Tell me about it. Started at $50K, went up to $70K, then back down to $55K. At least we're still above where we started.",
        ),
        _t(
            53,
            "assistant",
            "True! And you've got a clearer picture of exactly where every dollar is going now.",
        ),
        _t(
            54,
            "user",
            "Hey, can you help me with something unrelated? I need to find a birthday present for my wife. Her birthday is in two weeks.",
        ),
        _t(55, "assistant", "Of course! What does she enjoy? Hobbies, interests?"),
        _t(56, "user", "She's really into pottery. She just started taking classes."),
        _t(
            57,
            "assistant",
            "A nice pottery toolkit, a gift card for more classes, or even a beautiful handmade ceramic piece from a local artist could be great options.",
        ),
        _t(
            58,
            "user",
            "The class gift card is a great idea. I'll do that plus maybe a nice book about ceramics. OK so update on the budget situation.",
        ),
        _t(59, "assistant", "What happened?"),
        _t(
            60,
            "user",
            "So Sandra liked the revised budget but she said we found $10K in savings because the company just signed an enterprise deal with AWS and our cloud hosting costs dropped. She's giving us that $10K back.",
        ),
        _t(
            61,
            "assistant",
            "Great news! That brings you back up to $65K. Where will you put the extra funds?",
        ),
        _t(
            62,
            "user",
            "I want to bring Miguel back to the full 200 hours. That's an extra $3,750. And I want a proper contingency fund again - let's say $5K. The remaining $1,250 I'll add back to design.",
        ),
        _t(
            63,
            "assistant",
            "Updated: Dev ~$39K (Miguel 200hrs at $30K + $9K tools), Design $9K, QA $5K, Marketing $7K, Contingency $5K. Total $65K.",
        ),
        _t(64, "user", "That's the one. Final answer, I hope."),
        _t(65, "assistant", "Let's hope so too! That's a solid allocation."),
        _t(
            66,
            "user",
            "One more small thing - the QA lead said she can save $1K by using an open-source testing framework instead of the paid tool she originally planned. So QA drops to $4K.",
        ),
        _t(
            67,
            "assistant",
            "Nice. That frees up $1K. Want to add it to contingency? That'd make it $6K.",
        ),
        _t(
            68,
            "user",
            "Actually put it into marketing. We could use it for some App Store optimization. So marketing becomes $8K, QA is $4K. Everything else stays the same.",
        ),
        _t(
            69,
            "assistant",
            "Final budget: Dev $39K, Design $9K, QA $4K, Marketing $8K, Contingency $5K. Total $65K.",
        ),
        _t(
            70,
            "user",
            "That's it. Lock it in. Now let me tell you about my weekend plans.",
        ),
        _t(71, "assistant", "What are you up to?"),
        _t(
            72,
            "user",
            "Taking the family to the aquarium. My daughter has been obsessed with jellyfish lately.",
        ),
        _t(
            73,
            "assistant",
            "Jellyfish are mesmerizing! Some aquariums have incredible jellyfish exhibits with colored lighting. I'm sure she'll love it.",
        ),
        _t(
            74,
            "user",
            "Yeah, she'll go nuts. Alright, I think we're done for today. Budget's sorted, work is on track.",
        ),
        _t(
            75,
            "assistant",
            "Sounds good! Enjoy the aquarium. Let me know if you need anything else with the project.",
        ),
        _t(76, "user", "Will do. Thanks!"),
        _t(77, "assistant", "Have a great rest of your day!"),
    ]


def _build_state_diagnosis() -> list[dict]:
    """State change 2: Car problem diagnosis that evolves as new information surfaces."""
    return [
        _t(
            0,
            "user",
            "My car is making a weird noise and I need help figuring out what's wrong.",
        ),
        _t(
            1,
            "assistant",
            "I'll do my best to help diagnose it. Can you describe the noise? When does it happen?",
        ),
        _t(
            2,
            "user",
            "It's a grinding sound when I brake. Started a few days ago. I think it's probably the brake pads - they're probably worn down.",
        ),
        _t(
            3,
            "assistant",
            "A grinding noise when braking does commonly point to worn brake pads. If the pad material has worn away, the metal backing plate contacts the rotor directly, creating that grinding sound. How many miles are on the current pads?",
        ),
        _t(
            4,
            "user",
            "I'm not sure, I bought the car used about a year ago. 2018 Subaru Outback. I haven't done any brake work on it.",
        ),
        _t(
            5,
            "assistant",
            "With an unknown brake history, worn pads are a strong possibility. I'd recommend getting them inspected soon - driving on metal-to-metal can damage the rotors, which gets expensive.",
        ),
        _t(
            6,
            "user",
            "Yeah I'll take it in this week. Hey, do you have any recommendations for podcast apps? I've been using Apple Podcasts and it's fine but I want something better.",
        ),
        _t(
            7,
            "assistant",
            "Overcast is great for iOS - it has smart speed that shortens silences and excellent voice boost. If you want cross-platform, Pocket Casts is solid.",
        ),
        _t(
            8,
            "user",
            "I'll try Overcast. Anyway, I took the car to my mechanic this morning.",
        ),
        _t(9, "assistant", "What did they say?"),
        _t(
            10,
            "user",
            "He looked at the brakes and said the pads actually have plenty of life left. Like 60% remaining. So it's NOT the brake pads.",
        ),
        _t(
            11,
            "assistant",
            "Interesting! That rules out the most common cause. Did the mechanic have any other theories?",
        ),
        _t(
            12,
            "user",
            "He said it could be a stuck caliper or debris caught between the pad and rotor. He couldn't find anything obvious though. He charged me $40 for the inspection.",
        ),
        _t(
            13,
            "assistant",
            "A stuck caliper would cause uneven wear and heat, which could produce grinding. Debris is harder to spot if it's small. Did the mechanic hear the noise firsthand?",
        ),
        _t(
            14,
            "user",
            "No, it wasn't doing it while he was test driving. Of course. It only seems to happen after the car has been parked overnight and I brake within the first few minutes of driving.",
        ),
        _t(
            15,
            "assistant",
            "That's a very important detail! Grinding only when cold, especially first thing in the morning, is often caused by a thin layer of rust that forms on the rotors overnight. When you first brake, the pads scrape off the rust. It's actually completely normal and not a problem.",
        ),
        _t(16, "user", "Really? But it sounds terrible."),
        _t(
            17,
            "assistant",
            "The sound can be alarming, but surface rust on rotors is very common, especially in humid climates or when the car sits overnight. It should go away after a few brake applications. Does the noise stop after the first few stops?",
        ),
        _t(
            18,
            "user",
            "Actually... yeah, now that you mention it. It's always just the first time or two I brake. Then it's fine for the rest of the drive.",
        ),
        _t(
            19,
            "assistant",
            "That confirms it. Normal rotor surface rust. Not a safety concern. You can mention that to your mechanic next time.",
        ),
        _t(
            20,
            "user",
            "What a relief. I was worried about a huge repair bill. Speaking of bills, my electric bill was insane last month. But that's a separate conversation.",
        ),
        _t(
            21,
            "assistant",
            "Glad the car issue seems minor! We can look at the electric bill anytime.",
        ),
        _t(
            22,
            "user",
            "So I spoke too soon. I know we just said the noise was surface rust, but something new happened this morning.",
        ),
        _t(23, "assistant", "What happened?"),
        _t(
            24,
            "user",
            "The grinding happened in the REAR, not the front like before. And it persisted even after driving for 10 minutes. It didn't go away.",
        ),
        _t(
            25,
            "assistant",
            "That changes things significantly. Persistent grinding from the rear rules out the surface rust explanation. Rear brake issues on a Subaru Outback could involve the drum brakes (some models have rear drums), a failing wheel bearing, or a parking brake that's not fully releasing.",
        ),
        _t(
            26,
            "user",
            "It does have rear drum brakes actually. I remember the mechanic mentioning that.",
        ),
        _t(
            27,
            "assistant",
            "Drum brakes have different failure modes than disc brakes. The grinding could be a broken shoe retaining spring, a cracked drum, or even a rear wheel bearing issue that manifests under braking load.",
        ),
        _t(28, "user", "Should I take it back to the same mechanic?"),
        _t(
            29,
            "assistant",
            "Yes, but this time mention specifically that it's the REAR, that it's persistent, and that it started recently. He probably focused on the front disc brakes during the first inspection.",
        ),
        _t(
            30,
            "user",
            "Good point. Let me call and schedule. Actually first, do you know a good recipe for banana bread? I have like 5 overripe bananas.",
        ),
        _t(
            31,
            "assistant",
            "Classic banana bread: mash bananas, mix with melted butter, sugar, egg, vanilla, then fold in flour, baking soda, and salt. Bake at 350F for about 60 minutes. Adding walnuts or chocolate chips is optional but recommended.",
        ),
        _t(
            32,
            "user",
            "Yum. Chocolate chips for sure. Ok so I took the car back to the mechanic.",
        ),
        _t(33, "assistant", "What's the verdict this time?"),
        _t(
            34,
            "user",
            "He found it. The rear left drum brake had a broken return spring. The shoe was dragging against the drum constantly. That's what was causing the persistent grinding.",
        ),
        _t(
            35,
            "assistant",
            "Mystery solved! A broken return spring means the brake shoe doesn't retract when you release the pedal, causing continuous contact and grinding. It's a straightforward repair.",
        ),
        _t(
            36,
            "user",
            "Yeah he's fixing it now. Cost is about $180 for parts and labor. He also said the drum surface is slightly scored so he's going to resurface it.",
        ),
        _t(
            37,
            "assistant",
            "That's a very reasonable price. Resurfacing the drum is smart - it'll smooth out any grooves the dragging shoe made. Should be good as new after that.",
        ),
        _t(
            38,
            "user",
            "Great. And he confirmed the front brakes are totally fine. So the morning grinding up front was indeed just surface rust like you said.",
        ),
        _t(
            39,
            "assistant",
            "So the full picture: front grinding was harmless surface rust (still happening but normal), rear grinding was a broken return spring in the left drum brake (being repaired now). Two separate issues that happened to appear around the same time.",
        ),
        _t(
            40,
            "user",
            "Exactly. Having two different issues at once is what made this so confusing. I thought it was one big problem.",
        ),
        _t(
            41,
            "assistant",
            "Understandable! Cars can be tricky when multiple things overlap. At least you caught the real problem before it caused more damage.",
        ),
        _t(
            42,
            "user",
            "True. Hey, I want to talk about vacation plans too but let me pick up the car first. It should be done in an hour.",
        ),
        _t(43, "assistant", "Sounds good! Let me know when you're back."),
        _t(
            44,
            "user",
            "Car is fixed and driving great. No more grinding from either end. Well, the front might still do the surface rust thing tomorrow morning but I won't worry about it now.",
        ),
        _t(
            45,
            "assistant",
            "Perfect! The front surface rust is purely cosmetic noise. Your real issue - the broken spring - is resolved. How does the braking feel overall?",
        ),
        _t(
            46,
            "user",
            "Really smooth actually. Better than before the whole thing started. I think that shoe was probably dragging slightly for a while before it got bad enough to make noise.",
        ),
        _t(
            47,
            "assistant",
            "Very likely. A partially dragging shoe would have been reducing your fuel efficiency and causing uneven wear before it started grinding audibly.",
        ),
        _t(
            48,
            "user",
            "Well, at least it's done now. $220 total including the first inspection. Not bad.",
        ),
        _t(
            49,
            "assistant",
            "$220 for a complete diagnosis and repair is very reasonable. And you have peace of mind.",
        ),
        _t(
            50,
            "user",
            "Yep. Now about that vacation - I'm thinking Hawaii in September.",
        ),
        _t(
            51,
            "assistant",
            "September is actually a great time for Hawaii - fewer tourists, still warm, and prices are lower than peak summer. Which island are you considering?",
        ),
        _t(
            52,
            "user",
            "Maui probably. My wife has always wanted to go. But let me think about it more. I'll circle back on that.",
        ),
        _t(53, "assistant", "Sounds good! Let me know when you're ready to plan."),
        _t(54, "user", "Will do. Thanks for all the help with the car detective work!"),
        _t(55, "assistant", "Happy to help! Enjoy the smooth, quiet rides."),
    ]


# ---------------------------------------------------------------------------
# 4. Multi-perspective contradiction conversation
# ---------------------------------------------------------------------------


def _build_contradiction_meeting() -> list[dict]:
    """Multiple conflicting accounts of event details requiring careful resolution."""
    return [
        _t(
            0,
            "user",
            "I'm so confused about the logistics for the company retreat next month. Everyone is telling me different things.",
        ),
        _t(1, "assistant", "That sounds stressful! What's the confusion about?"),
        _t(
            2,
            "user",
            "Let's start with the basics. My manager Tom told me during standup that the retreat is at the Lakewood Resort, April 18-20.",
        ),
        _t(
            3,
            "assistant",
            "Lakewood Resort, April 18-20 - got it. Is that a Friday through Sunday?",
        ),
        _t(
            4,
            "user",
            "Yeah. Three days, two nights. Tom sounded pretty confident about it.",
        ),
        _t(5, "assistant", "Good. What's the conflicting information?"),
        _t(
            6,
            "user",
            "Well, Sarah from HR sent an email yesterday saying it's at the Ridgeview Lodge, not Lakewood. Same dates though.",
        ),
        _t(
            7,
            "assistant",
            "So the venue is in question but the dates match. Did Sarah's email look official? Usually HR handles the logistics.",
        ),
        _t(
            8,
            "user",
            "It was a pretty official-looking email with the company letterhead. But Tom is on the planning committee, so he should know too. Ugh.",
        ),
        _t(9, "assistant", "Both sources seem credible. Did anyone else weigh in?"),
        _t(
            10,
            "user",
            "Not yet on the venue. But it gets worse. Let me tell you about the activities first. Oh wait, hold on - my dog is going crazy at the mailman. One sec.",
        ),
        _t(11, "assistant", "Take your time!"),
        _t(
            12,
            "user",
            "OK back. Sorry, she always loses it when the mail comes. So about the retreat activities.",
        ),
        _t(13, "assistant", "What's the activity confusion?"),
        _t(
            14,
            "user",
            "My coworker Jake, who went last year, said the retreat always includes a ropes course. He said it's mandatory. I really don't want to do a ropes course - I'm afraid of heights.",
        ),
        _t(
            15,
            "assistant",
            "That's understandable. Did you confirm with anyone official that there's a ropes course?",
        ),
        _t(
            16,
            "user",
            "I asked Tom and he said 'I think there might be some outdoor activities but nothing's confirmed.' Very vague.",
        ),
        _t(
            17,
            "assistant",
            "Tom's response doesn't confirm a ropes course specifically. Jake might be assuming this year matches last year's format. I wouldn't worry too much until the official agenda comes out.",
        ),
        _t(
            18,
            "user",
            "Fair point. Oh, I also need to figure out food. I have a shellfish allergy and I need to make sure they accommodate that.",
        ),
        _t(
            19,
            "assistant",
            "Definitely important to flag early. You should email the organizer directly about dietary needs.",
        ),
        _t(
            20,
            "user",
            "Yeah I will. Now here's another conflicting piece. My teammate Priya told me over lunch that she heard the dates changed to April 25-27.",
        ),
        _t(
            21,
            "assistant",
            "A week later than what Tom and Sarah both said. Where did Priya hear this?",
        ),
        _t(
            22,
            "user",
            "She said she overheard the VP Marcus talking about it in the break room. But she admitted she might have misheard - he could have been talking about something else.",
        ),
        _t(
            23,
            "assistant",
            "Given that both Tom (planning committee) and Sarah (HR, official email) agreed on April 18-20, and Priya's info is secondhand and uncertain, I'd trust the 18-20 dates for now.",
        ),
        _t(
            24,
            "user",
            "That's what I'm leaning towards too. Now let me tell you about something completely different. I'm trying to learn to play guitar.",
        ),
        _t(25, "assistant", "How fun! Are you a complete beginner?"),
        _t(
            26,
            "user",
            "Pretty much. I can play a few chords - G, C, D, E minor. Just started a month ago.",
        ),
        _t(
            27,
            "assistant",
            "Those are great starter chords! With just those four you can play hundreds of songs. Have you tried any songs yet?",
        ),
        _t(
            28, "user", "Working on 'Knockin on Heaven's Door.' Slow but getting there."
        ),
        _t(
            29,
            "assistant",
            "Perfect song for beginners - simple chord progression and a great feeling when you can play along with the recording.",
        ),
        _t(
            30,
            "user",
            "Yeah! Ok back to the retreat drama. I just got a new email from HR. It's from Sarah's boss, Diana.",
        ),
        _t(31, "assistant", "What does Diana say?"),
        _t(
            32,
            "user",
            "She says the retreat venue has been finalized: it's at Lakewood Resort. She says there was confusion because they originally booked Ridgeview Lodge but had to switch to Lakewood due to a double-booking at Ridgeview.",
        ),
        _t(
            33,
            "assistant",
            "That explains the discrepancy! Sarah's earlier email was based on the original booking, and Tom had the updated info. Lakewood Resort is confirmed.",
        ),
        _t(
            34,
            "user",
            "Right. And Diana confirmed April 18-20, so Priya's overheard dates were wrong.",
        ),
        _t(
            35,
            "assistant",
            "Good. So we now have definitive answers from Diana (Sarah's boss): Lakewood Resort, April 18-20. The venue confusion is resolved and the dates are confirmed.",
        ),
        _t(
            36,
            "user",
            "Exactly. But now there's ANOTHER thing. Diana's email says the retreat starts at 10am on Friday.",
        ),
        _t(37, "assistant", "OK, 10am Friday start. What's the issue?"),
        _t(
            38,
            "user",
            "Tom told me separately that our team needs to be there by 8am for some pre-retreat planning session that's just for the engineering department.",
        ),
        _t(
            39,
            "assistant",
            "So there's an engineering-specific session at 8am before the main retreat starts at 10am. That's not necessarily a contradiction - it could be an add-on for your department.",
        ),
        _t(
            40,
            "user",
            "That's what I thought. But then I asked our engineering lead, Rachel, about it and she said she has no idea what Tom is talking about. There's no 8am engineering session that she's aware of.",
        ),
        _t(
            41,
            "assistant",
            "Hmm. Tom is on the planning committee but Rachel is the engineering lead and doesn't know about an 8am session. Could Tom have confused it with something else, or could it be something that's being planned but hasn't been communicated to Rachel yet?",
        ),
        _t(
            42,
            "user",
            "I texted Tom about it and he said 'oh wait, I was thinking of the OTHER offsite we're planning for May. Disregard the 8am thing for the retreat. My bad.'",
        ),
        _t(
            43,
            "assistant",
            "There it is. Tom mixed up two different events. So for the retreat, the standard 10am start from Diana's email is correct. No need to arrive at 8am.",
        ),
        _t(
            44,
            "user",
            "Thank god. I was not looking forward to waking up that early. The resort is like 90 minutes away.",
        ),
        _t(
            45,
            "assistant",
            "90 minutes means leaving by 8:30am for a 10am start, even without the false alarm 8am session. That's still early but more reasonable.",
        ),
        _t(
            46,
            "user",
            "True. Let me also sort out the roommate situation. Jake said we'll all have single rooms.",
        ),
        _t(47, "assistant", "That would be nice! Is that confirmed?"),
        _t(
            48, "user", "Diana's email says it's double rooms, shared with a colleague."
        ),
        _t(
            49,
            "assistant",
            "So Jake was wrong about single rooms. Diana's official email says double occupancy.",
        ),
        _t(
            50,
            "user",
            "Yep. I asked Jake where he heard single rooms and he said 'that's how it was last year.' He keeps assuming everything is the same as last year.",
        ),
        _t(
            51,
            "assistant",
            "Jake seems to be your least reliable source - he's extrapolating from last year rather than having current information. I'd weight anything he says lower than Tom (planning committee) or Diana/Sarah (HR).",
        ),
        _t(
            52,
            "user",
            "Ha, yeah. I love Jake but he's not great with details. OK let me bring up something different - my fantasy football league draft is coming up.",
        ),
        _t(53, "assistant", "When is the draft?"),
        _t(
            54,
            "user",
            "End of August, but I like to research early. I'm thinking about my first-round pick strategy.",
        ),
        _t(
            55,
            "assistant",
            "Early research pays off! Are you in a PPR or standard scoring league?",
        ),
        _t(56, "user", "PPR. 12-team. I have the 4th pick."),
        _t(
            57,
            "assistant",
            "4th pick in a 12-team PPR - you'll likely see the top RBs go 1-3. You could grab the best remaining RB or pivot to a top WR like...",
        ),
        _t(
            58,
            "user",
            "Let me think about that later. One more retreat thing. I just asked Priya about the activities since she works with the events team on the side.",
        ),
        _t(59, "assistant", "What did she find out?"),
        _t(
            60,
            "user",
            "She actually pulled up the draft agenda. No ropes course! The outdoor activities are a nature hike and kayaking, both optional.",
        ),
        _t(
            61,
            "assistant",
            "Great news for you! No mandatory ropes course. The hike and kayaking are optional so you can participate based on your comfort level.",
        ),
        _t(
            62,
            "user",
            "Kayaking sounds fun actually. I'll do that. OK so let me summarize what we know for sure vs what was wrong.",
        ),
        _t(63, "assistant", "Go for it, let's lay it all out."),
        _t(
            64,
            "user",
            "CONFIRMED: Lakewood Resort (not Ridgeview), April 18-20 (not 25-27), starts at 10am (not 8am), double rooms (not single), no mandatory ropes course, nature hike and kayaking are optional.",
        ),
        _t(
            65,
            "assistant",
            "Perfect summary. The wrong info came from: Ridgeview Lodge (Sarah's outdated email), April 25-27 (Priya's misheard conversation), 8am start (Tom confused with May offsite), single rooms (Jake assuming same as last year), mandatory ropes course (Jake again, assuming last year).",
        ),
        _t(
            66,
            "user",
            "Exactly. Jake was wrong twice, Tom was wrong once, Sarah had outdated info, and Priya had secondhand info. Diana from HR had the truth.",
        ),
        _t(
            67,
            "assistant",
            "Lesson learned: go to the source! Diana's official communications were right every time. The informal grapevine introduced a lot of noise.",
        ),
        _t(
            68,
            "user",
            "So true. I'll just wait for official emails from now on instead of relying on hallway chatter.",
        ),
        _t(
            69,
            "assistant",
            "That's a good policy. Don't forget to email about your shellfish allergy before the retreat.",
        ),
        _t(
            70,
            "user",
            "Right! I'll do that today. Thanks for helping me untangle all of that.",
        ),
        _t(71, "assistant", "Happy to help! Enjoy the retreat at Lakewood."),
        _t(72, "user", "Will do. Chat later!"),
        _t(73, "assistant", "Take care!"),
    ]


# ---------------------------------------------------------------------------
# 5. Open-ended exploration conversation
# ---------------------------------------------------------------------------


def _build_exploration_health() -> list[dict]:
    """Scattered health/lifestyle mentions with hidden patterns to detect."""
    return [
        _t(0, "user", "Morning! Rough start today. Woke up with a splitting headache."),
        _t(
            1,
            "assistant",
            "Sorry to hear that! Did you sleep ok? Headaches can come from poor sleep, dehydration, or stress.",
        ),
        _t(
            2,
            "user",
            "Sleep was fine, about 7 hours. I had plenty of water yesterday. Probably just stress from the new project that started last week.",
        ),
        _t(
            3,
            "assistant",
            "New projects can definitely bring stress headaches. What's the project about?",
        ),
        _t(
            4,
            "user",
            "We're migrating our entire infrastructure to Kubernetes. It's a huge undertaking. The deadline is tight.",
        ),
        _t(
            5,
            "assistant",
            "K8s migrations are no joke. How many services are you moving?",
        ),
        _t(
            6,
            "user",
            "About 40 microservices. It's going to be a long few months. Anyway, the headache is fading. Let me ask about something else - I want to repaint my living room.",
        ),
        _t(7, "assistant", "What color are you thinking?"),
        _t(
            8,
            "user",
            "Something warm. Maybe a sage green? My wife wants white but I think that's boring.",
        ),
        _t(
            9,
            "assistant",
            "Sage green is really popular right now and creates a calming atmosphere. You could compromise with a very light sage that reads as a soft neutral from a distance.",
        ),
        _t(
            10,
            "user",
            "Good idea, I'll show her some swatches. Hey, we changed floors at work last month. Did I mention that?",
        ),
        _t(11, "assistant", "No, you didn't! What prompted the move?"),
        _t(
            12,
            "user",
            "Our team got consolidated. We used to be on floor 3, now we're on floor 7. The views are better but the lighting is different - these super bright overhead fluorescents. Kind of harsh.",
        ),
        _t(
            13,
            "assistant",
            "Harsh fluorescent lighting can be unpleasant. Some people find it causes eye strain or headaches. Have you noticed any difference since the move?",
        ),
        _t(
            14,
            "user",
            "Hmm, I hadn't really thought about it. Maybe? I don't know. Anyway, it's Monday and I have back-to-back meetings starting at 10. Let me prep for those.",
        ),
        _t(
            15,
            "assistant",
            "Good luck with the meetings! Let me know if you need anything.",
        ),
        _t(
            16,
            "user",
            "Thanks. Oh, I also started drinking more coffee lately. Like 4 cups a day instead of my usual 2. The K8s project has me working longer hours.",
        ),
        _t(
            17,
            "assistant",
            "Be careful with doubling your caffeine - it can affect sleep quality even if you feel like you're sleeping enough hours. It can also cause headaches if your body isn't used to that much.",
        ),
        _t(
            18,
            "user",
            "Yeah, you're probably right. I should cut back. But I need the energy honestly.",
        ),
        _t(
            19,
            "assistant",
            "Maybe try keeping the last coffee before 2pm so it doesn't affect your sleep cycle. And staying hydrated extra helps offset caffeine's dehydrating effect.",
        ),
        _t(20, "user", "Good tips. I'll try that. Talk to you later, meeting time."),
        _t(21, "assistant", "Have a productive day!"),
        _t(
            22,
            "user",
            "Hey, it's Wednesday. Quick question - what's the best way to store leftover pizza?",
        ),
        _t(
            23,
            "assistant",
            "Don't stack the slices! Lay them flat in a container with parchment paper between layers. Reheat in a skillet on medium-low for the crispiest results.",
        ),
        _t(24, "user", "Skillet, interesting. I always use the microwave."),
        _t(
            25,
            "assistant",
            "The skillet method is a game-changer - crispy bottom, melty top. Cover with a lid for the last minute and add a few drops of water to steam the cheese.",
        ),
        _t(
            26,
            "user",
            "I'll try that tonight. We ordered way too much pizza last night.",
        ),
        _t(
            27,
            "assistant",
            "Better too much than too little! How's the K8s migration going?",
        ),
        _t(
            28,
            "user",
            "Stressful. We hit a networking issue with the service mesh. Spent all day debugging. My neck is killing me too, probably from hunching over my laptop.",
        ),
        _t(
            29,
            "assistant",
            "Service mesh issues are always tricky. For the neck, try some gentle stretches every hour - tilt your head side to side, roll your shoulders. Is your workstation ergonomic?",
        ),
        _t(
            30,
            "user",
            "Not really. The new floor doesn't have the adjustable desks we had downstairs. I've been meaning to ask facilities for a monitor riser.",
        ),
        _t(
            31,
            "assistant",
            "That would help a lot. Having your monitor at eye level prevents the downward neck tilt that causes strain.",
        ),
        _t(
            32,
            "user",
            "I'll put in a request. Hey, random - do you think electric cars are worth it? My lease is up in a few months.",
        ),
        _t(
            33,
            "assistant",
            "Depends on your driving patterns. How long is your commute and do you have home charging access?",
        ),
        _t(
            34,
            "user",
            "20 miles each way. We have a garage with a regular outlet but no dedicated charger.",
        ),
        _t(
            35,
            "assistant",
            "A 40-mile round trip is well within EV range. A standard outlet can charge overnight for that distance. You'd save a lot on gas. I'd recommend test driving a few options.",
        ),
        _t(36, "user", "I'll look into it this weekend. Have a good night!"),
        _t(37, "assistant", "You too!"),
        _t(38, "user", "Ugh, another Monday. Started the day with a headache again."),
        _t(
            39,
            "assistant",
            "That's the second Monday headache you've mentioned. Is it the same kind of headache as before?",
        ),
        _t(
            40,
            "user",
            "Yeah, same thing. Kind of a pressure behind my eyes and temples. Took some Advil and it helped. Probably just the Sunday-to-Monday stress transition.",
        ),
        _t(
            41,
            "assistant",
            "Could be. Do you do anything different on weekends versus weekdays that might be contributing? Different sleep schedule, different caffeine intake?",
        ),
        _t(
            42,
            "user",
            "I do sleep in on weekends - like 9 hours instead of 7. And I probably drink less coffee on the weekend since I'm more relaxed.",
        ),
        _t(
            43,
            "assistant",
            "Both of those could play a role. The sleep schedule shift can trigger headaches, and caffeine withdrawal on weekends followed by resuming on Monday could cause rebound headaches.",
        ),
        _t(
            44,
            "user",
            "Huh, interesting. I'll keep that in mind. But honestly I just think it's stress. This project is killing me.",
        ),
        _t(
            45,
            "assistant",
            "Stress is definitely a factor. But the Monday-specific pattern is worth paying attention to.",
        ),
        _t(
            46,
            "user",
            "Yeah maybe. Let me ask you about cooking - I want to try making homemade pasta this weekend.",
        ),
        _t(
            47,
            "assistant",
            "Homemade pasta is incredibly satisfying! Basic recipe: 2 cups flour, 3 eggs, pinch of salt. Mix, knead for 10 minutes, rest for 30, then roll and cut.",
        ),
        _t(48, "user", "Do I need a pasta machine?"),
        _t(
            49,
            "assistant",
            "Not necessary but helpful. You can roll by hand with a rolling pin - just takes more effort. Get the dough as thin as possible, especially for fettuccine.",
        ),
        _t(
            50,
            "user",
            "I'll try by hand first. If I like it, I'll invest in a machine.",
        ),
        _t(
            51,
            "assistant",
            "Smart approach. The dough itself is the most important part - get that right and it's delicious either way.",
        ),
        _t(
            52,
            "user",
            "Cool. Hey, is it normal to feel more tired in the afternoon since starting at a new workspace? I've been hitting a wall around 2pm every day.",
        ),
        _t(
            53,
            "assistant",
            "A 2pm energy dip is common and often related to circadian rhythms and post-lunch blood sugar changes. But if it's new since moving floors, the lighting environment could play a role. Harsh fluorescents without natural light cues can disrupt your body's alertness signals.",
        ),
        _t(
            54,
            "user",
            "We are further from the windows on floor 7 than we were on floor 3. My old desk was right by a window, now I'm in the interior.",
        ),
        _t(
            55,
            "assistant",
            "That's significant. Natural light exposure during the day helps regulate your circadian rhythm. Being further from windows, especially under fluorescents, can genuinely affect energy levels, mood, and even headache frequency.",
        ),
        _t(
            56,
            "user",
            "Hmm, I never connected those dots. Maybe I should take walks outside during lunch.",
        ),
        _t(
            57,
            "assistant",
            "That would be an excellent countermeasure. Even 15-20 minutes of natural light exposure, especially in the morning, can make a big difference for your alertness and mood.",
        ),
        _t(
            58,
            "user",
            "I'll start doing that. OK gotta run to another meeting. This project has so many meetings.",
        ),
        _t(59, "assistant", "Good luck! And seriously, try the outdoor lunch break."),
        _t(
            60,
            "user",
            "Wednesday again. Feeling great today actually. No headache, good energy.",
        ),
        _t(61, "assistant", "That's good to hear! Anything different about today?"),
        _t(
            62,
            "user",
            "Not really. Just a regular Wednesday. Working from home actually, since I have a plumber coming for the kitchen sink.",
        ),
        _t(
            63,
            "assistant",
            "Working from home might be part of why you feel better - your home lighting is probably easier on the eyes than the office fluorescents.",
        ),
        _t(
            64,
            "user",
            "Yeah, I have my desk by the window at home. Lots of natural light. Hadn't thought about it that way.",
        ),
        _t(
            65,
            "assistant",
            "It's worth noting. On WFH days you feel good, in the office under fluorescents you get more headaches and afternoon slumps. That's a data point.",
        ),
        _t(
            66,
            "user",
            "True. Anyway, the plumber is here. Kitchen faucet has been dripping for weeks. Drives my wife nuts.",
        ),
        _t(
            67,
            "assistant",
            "Dripping faucets usually need a new cartridge or washer. Relatively quick fix for a plumber.",
        ),
        _t(
            68,
            "user",
            "Yeah he said 20 minutes. Easy. OK back to work stuff - can you help me understand Kubernetes network policies?",
        ),
        _t(
            69,
            "assistant",
            "Sure! Kubernetes NetworkPolicy resources control traffic flow at the IP/port level for pods. By default, pods accept traffic from any source. A NetworkPolicy lets you restrict ingress and egress...",
        ),
        _t(
            70,
            "user",
            "Actually hold on, let me save that for when I can focus. I'm distracted by the plumber noise.",
        ),
        _t(71, "assistant", "No worries, we can dig into that anytime."),
        _t(72, "user", "Hey it's Monday again. Guess what."),
        _t(73, "assistant", "Another headache?"),
        _t(
            74,
            "user",
            "Yep. Third Monday in a row. This is getting annoying. I'm starting to think you're right that it's a pattern.",
        ),
        _t(
            75,
            "assistant",
            "Three consecutive Mondays is hard to dismiss as coincidence. Let's think about what's different about Mondays specifically: you're back in the office after the weekend, back under the fluorescent lights, you've had a different sleep and caffeine schedule over the weekend, and the K8s project stress ramps up for the work week.",
        ),
        _t(76, "user", "Yeah. Any one of those could be it. Or all of them together."),
        _t(
            77,
            "assistant",
            "Multiple factors compounding is actually the most likely scenario. The sleep schedule shift, caffeine withdrawal and then reloading, the lighting change, and the stress kicking in simultaneously on Monday morning.",
        ),
        _t(78, "user", "What should I do about it?"),
        _t(
            79,
            "assistant",
            "A few things to try: keep your weekend sleep schedule closer to weekday (don't sleep in more than 1 extra hour), maintain consistent caffeine intake (don't skip coffee on weekends if you drink 4 cups on weekdays), take that outdoor lunch walk especially on Mondays, and see if you can get a desk lamp with warmer light for your office space to counteract the fluorescents.",
        ),
        _t(
            80,
            "user",
            "Those are actionable. I'll try all of them starting this weekend. If it doesn't help I'll see my doctor.",
        ),
        _t(
            81,
            "assistant",
            "Good plan. Give it 2-3 weeks to see if the pattern breaks. If not, definitely see a doctor to rule out anything else.",
        ),
        _t(
            82,
            "user",
            "Will do. Hey, one more thing - my back has been bothering me too. Not badly, just a dull ache.",
        ),
        _t(83, "assistant", "Where in the back?"),
        _t(
            84,
            "user",
            "Lower back, left side. It comes and goes. I think the office chairs on floor 7 are worse than what we had before. They're these cheap ones without lumbar support.",
        ),
        _t(
            85,
            "assistant",
            "Between the non-adjustable desks, harsh fluorescents, cheap chairs, and distance from windows, your new office setup sounds like it's affecting you in multiple ways.",
        ),
        _t(
            86,
            "user",
            "Yeah, floor 3 was way better for my physical setup. I should probably push back with facilities more aggressively.",
        ),
        _t(
            87,
            "assistant",
            "Absolutely. An ergonomic assessment from facilities could address the desk height, chair quality, and possibly even the lighting. Many companies offer this, especially if you mention health impacts.",
        ),
        _t(88, "user", "I'll email them today. OK, meetings start soon. Talk later."),
        _t(89, "assistant", "Good luck! And remember the outdoor walk at lunch."),
        _t(
            90,
            "user",
            "Ha, I will. Thanks for keeping tabs on my health better than I do.",
        ),
        _t(91, "assistant", "That's what I'm here for! Take care."),
    ]


def _build_exploration_spending() -> list[dict]:
    """Scattered spending mentions with hidden patterns about impulse buying."""
    return [
        _t(
            0,
            "user",
            "Hey, I impulse-bought a new keyboard yesterday. Mechanical, cherry MX browns. $180.",
        ),
        _t(
            1,
            "assistant",
            "Cherry MX Browns are great - nice tactile bump without being too loud. Was it a brand you had your eye on?",
        ),
        _t(
            2,
            "user",
            "The Keychron Q1. I've been researching keyboards for months so I guess it wasn't really impulse, but it felt like it in the moment.",
        ),
        _t(
            3,
            "assistant",
            "If you'd been researching for months, that's a considered purchase! The Keychron Q1 is well-regarded. Enjoy it!",
        ),
        _t(
            4,
            "user",
            "Thanks. I need to be more careful with spending though. I'm trying to save for a house down payment.",
        ),
        _t(
            5,
            "assistant",
            "That's a big goal! How much are you targeting for the down payment?",
        ),
        _t(
            6,
            "user",
            "About $60K for a 20% down on a $300K home. We've got about $35K saved so far.",
        ),
        _t(7, "assistant", "More than halfway there! At what rate are you saving?"),
        _t(
            8,
            "user",
            "About $2K a month into the house fund. So roughly another year if we stick to it.",
        ),
        _t(9, "assistant", "That's a solid plan. Consistency is key."),
        _t(
            10,
            "user",
            "Yeah. Hey, can you recommend a good book? I just finished the one I was reading and need something new.",
        ),
        _t(11, "assistant", "What genres do you enjoy?"),
        _t(
            12, "user", "Sci-fi mostly. I just finished Project Hail Mary and loved it."
        ),
        _t(
            13,
            "assistant",
            "If you loved Project Hail Mary, try 'Children of Time' by Adrian Tchaikovsky or 'The Long Way to a Small Angry Planet' by Becky Chambers. Both are excellent sci-fi with great world-building.",
        ),
        _t(
            14,
            "user",
            "I'll check those out! So I was looking at my credit card statement last night. Lots of small purchases adding up.",
        ),
        _t(
            15,
            "assistant",
            "That's how it usually works - the small ones fly under the radar. What kind of purchases?",
        ),
        _t(
            16,
            "user",
            "Let me see... $45 for a Udemy course I haven't started, $60 on Amazon for random stuff, $35 for a Spotify and Audible subscription I forgot to cancel, and then eating out like 3 times this week at $25-30 each.",
        ),
        _t(
            17,
            "assistant",
            "That's roughly $230 in discretionary spending in one week. The forgotten subscriptions are the easiest win - cancel what you're not using. And Audible especially adds up at $15/month if you're not listening.",
        ),
        _t(
            18,
            "user",
            "You're right. I'll cancel Audible today. Spotify I actually use. The Udemy course I'll commit to this weekend.",
        ),
        _t(
            19,
            "assistant",
            "Good plan. The eating out is where most people overspend consistently. Even cutting one of those three meals out per week saves $100-130 a month.",
        ),
        _t(
            20,
            "user",
            "True. I cook most nights, it's just when I'm stressed or tired that I default to eating out. And this month has been rough at work.",
        ),
        _t(
            21,
            "assistant",
            "Meal prepping on weekends could help - when you're tired, having food ready to go removes the temptation.",
        ),
        _t(
            22,
            "user",
            "I should do that. Anyway, different topic - I'm thinking about adopting a cat. My girlfriend has been wanting one.",
        ),
        _t(23, "assistant", "Cats are wonderful! Are you looking at shelters?"),
        _t(
            24,
            "user",
            "Yeah, we'll go to the local shelter this weekend. I grew up with cats so I know the basics.",
        ),
        _t(
            25,
            "assistant",
            "Shelter cats are great. Budget around $50-100 for adoption fees, plus initial vet visits, food, litter, and supplies. First month might be $300-400.",
        ),
        _t(26, "user", "That's fine. Monthly upkeep is like what, $50-100?"),
        _t(
            27,
            "assistant",
            "About that, yes. Food and litter are the ongoing costs. Add in annual vet checkups around $200-300.",
        ),
        _t(
            28,
            "user",
            "Manageable. OK so we adopted a cat! His name is Mochi. He's a 3-year-old orange tabby.",
        ),
        _t(
            29,
            "assistant",
            "Congratulations! Orange tabbies are known for being super friendly and goofy. How's Mochi settling in?",
        ),
        _t(
            30,
            "user",
            "He's hiding under the bed right now but the shelter said that's normal for the first few days. We got him set up with a litter box, food, some toys.",
        ),
        _t(
            31,
            "assistant",
            "Give him time. Most cats come out of their shell within a week. Keep things quiet and let him explore at his own pace.",
        ),
        _t(
            32,
            "user",
            "Will do. Oh, I bought some stuff for him at PetSmart. Spent $250. That's more than I planned.",
        ),
        _t(33, "assistant", "First-time pet parent syndrome! What did you get?"),
        _t(
            34,
            "user",
            "Cat tree ($80), premium food ($40), fancy water fountain ($55), multiple toys ($45), and a grooming kit ($30). I went a little overboard.",
        ),
        _t(
            35,
            "assistant",
            "A cat tree and water fountain are good investments. The toys - cats are famously more interested in the box than the toy, so you might not need to resupply those for a while.",
        ),
        _t(
            36,
            "user",
            "Ha, so true. He's already ignoring the toys and playing with a crinkled receipt. Money well spent.",
        ),
        _t(37, "assistant", "Classic cat behavior!"),
        _t(
            38,
            "user",
            "Hey, so I bought a new gadget. A Raspberry Pi kit for $120. I want to set up a home server.",
        ),
        _t(39, "assistant", "Fun project! What are you planning to run on it?"),
        _t(
            40,
            "user",
            "Pi-hole for ad blocking and maybe a NAS for local storage. I watched a YouTube video about it and got inspired.",
        ),
        _t(
            41,
            "assistant",
            "Pi-hole is one of the best Pi projects. The NAS idea is cool too, though a Pi might be a bit slow for heavy NAS usage.",
        ),
        _t(
            42,
            "user",
            "Yeah, we'll see. I also signed up for a Hacker News Pro account or something... wait no, that's free. I signed up for a VPN service. $100 for a year.",
        ),
        _t(
            43,
            "assistant",
            "A VPN is useful, especially if you're running a home server. Which service?",
        ),
        _t(
            44,
            "user",
            "Mullvad. Privacy-focused, no logging. I probably didn't NEED it right now but figured I might as well while setting up the Pi.",
        ),
        _t(
            45,
            "assistant",
            "Mullvad is a great choice. The timing makes sense if you're already in 'home networking mode.'",
        ),
        _t(
            46,
            "user",
            "I realize I've been spending a lot this month though. Keyboard $180, cat stuff $250, adoption fee $75, Raspberry Pi kit $120, VPN $100. That's $725 in non-essential purchases.",
        ),
        _t(
            47,
            "assistant",
            "That is a significant chunk, especially when you're saving $2K a month for a house. That's more than a third of your monthly savings going to discretionary purchases.",
        ),
        _t(
            48,
            "user",
            "When you put it that way... yikes. And I haven't even counted eating out.",
        ),
        _t(
            49,
            "assistant",
            "On the positive side, several of those are one-time costs (keyboard, Pi, adoption fee) and the VPN is annual. It's not a recurring monthly drain. But it's worth being aware of the pattern.",
        ),
        _t(50, "user", "What pattern?"),
        _t(
            51,
            "assistant",
            "Each purchase had a reasonable justification, but they clustered together. The keyboard after months of research, the cat was planned, the Pi was inspiration-driven, the VPN tagged along with the Pi. When you're in a 'spending mode,' purchases tend to enable each other.",
        ),
        _t(
            52,
            "user",
            "Hmm. You might be right. Once I made one purchase, the others felt easier to justify. Like a dam breaking.",
        ),
        _t(
            53,
            "assistant",
            "It's a well-documented behavioral pattern. After making one significant purchase, the psychological barrier to the next one lowers. Retailers actually use this with 'frequently bought together' recommendations.",
        ),
        _t(
            54,
            "user",
            "That's insightful. I need a cooling-off rule or something. Like, wait 48 hours before any purchase over $50.",
        ),
        _t(
            55,
            "assistant",
            "The 48-hour rule is effective! Another approach: keep a 'want list' and revisit it monthly. Many things you wanted lose their appeal after a few weeks.",
        ),
        _t(
            56,
            "user",
            "I'll start doing that. Ok, totally different topic - any good hiking trails near me? I'm in the Pacific Northwest.",
        ),
        _t(
            57,
            "assistant",
            "PNW has incredible hiking! What area specifically? Seattle, Portland, somewhere else?",
        ),
        _t(58, "user", "Portland area. I want something moderate, 5-8 miles."),
        _t(
            59,
            "assistant",
            "Eagle Creek Trail is a classic - gorgeous waterfalls. Or try Angel's Rest for amazing Columbia River Gorge views. Both are moderate and in that range.",
        ),
        _t(
            60,
            "user",
            "I'll check those out this weekend. Gotta get outside more instead of buying things online.",
        ),
        _t(61, "assistant", "That's a great healthy swap! Nature over retail therapy."),
        _t(
            62,
            "user",
            "Ha, retail therapy. That's exactly what it's been. I tend to shop more when I'm stressed from work, which is a lot lately.",
        ),
        _t(
            63,
            "assistant",
            "Recognizing the stress-shopping connection is the first step. Channeling that energy into something like hiking, cooking, or even the Raspberry Pi project itself (which is already paid for) is a better outlet.",
        ),
        _t(
            64,
            "user",
            "You're right. The Pi project keeps me entertained without spending more money. OK I should get back to work. Thanks for the real talk.",
        ),
        _t(
            65,
            "assistant",
            "Anytime! Enjoy the hike this weekend, and give Mochi a pat from me.",
        ),
        _t(
            66,
            "user",
            "Will do! He's actually sitting on my lap right now purring. Best $75 I ever spent.",
        ),
        _t(67, "assistant", "Now THAT was a great purchase. Have a good one!"),
    ]


# ---------------------------------------------------------------------------
# 6. Absence inference conversations
# ---------------------------------------------------------------------------


def _build_absence_seasonal() -> list[dict]:
    """Seasonal outdoor preference where the December indoor choice is never explained."""
    return [
        _t(
            0,
            "user",
            "Hey, I want to start planning more outings with friends. I feel like we've been cooped up.",
        ),
        _t(
            1,
            "assistant",
            "That's a great idea! What kinds of activities do you and your friends enjoy?",
        ),
        _t(
            2,
            "user",
            "All sorts of stuff. Last June we did a barbecue at the park and it was perfect. Great weather, everyone loved it.",
        ),
        _t(3, "assistant", "A park barbecue sounds lovely! Do you do that often?"),
        _t(
            4,
            "user",
            "We try to do something every month or two. In July we went to an outdoor concert series at the amphitheater downtown. That was awesome.",
        ),
        _t(
            5,
            "assistant",
            "Outdoor concerts in summer are the best. What kind of music?",
        ),
        _t(
            6,
            "user",
            "It was a jazz series. Really chill vibe, we brought blankets and wine. Perfect summer evening.",
        ),
        _t(7, "assistant", "Sounds wonderful! What other outings have you done?"),
        _t(
            8,
            "user",
            "Let me think... In August we did a hiking day trip to the gorge. About 8 of us went. Then September we had a big soccer game in the park.",
        ),
        _t(
            9,
            "assistant",
            "So you're an active, outdoorsy group! What about fall and winter activities?",
        ),
        _t(
            10,
            "user",
            "October we carved pumpkins at my friend Lisa's house. That was indoors because, well, it's messy and her dining table is perfect for it.",
        ),
        _t(
            11,
            "assistant",
            "Pumpkin carving is a great October activity! What about November?",
        ),
        _t(
            12,
            "user",
            "November we did a Friendsgiving potluck at my place. Everyone brought a dish. It was really cozy.",
        ),
        _t(13, "assistant", "Love Friendsgiving! And December?"),
        _t(
            14,
            "user",
            "December is kind of dead for us socially. The holidays take over and everyone is traveling or with family. We usually don't do a group thing.",
        ),
        _t(15, "assistant", "That makes sense. January?"),
        _t(
            16,
            "user",
            "We did a game night at my place in January. Board games, snacks, the works. February was a cooking class.",
        ),
        _t(17, "assistant", "Both indoor activities. Do you notice a pattern there?"),
        _t(
            18,
            "user",
            "Ha, I guess we go indoors when it's cold. I don't really do much outside in the winter. I kind of hate being cold, to be honest.",
        ),
        _t(
            19,
            "assistant",
            "That's very common! Some people are warm-weather people and some are cold-weather people.",
        ),
        _t(
            20,
            "user",
            "I am definitely a warm-weather person. My friends joke that I hibernate from November to March. I only go outside to get to my car and back.",
        ),
        _t(
            21,
            "assistant",
            "You and the bears have the right idea! So you probably enjoy March when things start warming up?",
        ),
        _t(
            22,
            "user",
            "Yes! March is when I start to come alive again. Last March we did a picnic as soon as it was warm enough. It was only like 55 degrees but I was so happy to be outside.",
        ),
        _t(
            23,
            "assistant",
            "A 55-degree picnic takes dedication! What about April and May?",
        ),
        _t(
            24,
            "user",
            "April is tough because of the rain here in Portland. But we did do a farmers market visit. May we had a beach bonfire and it was one of my favorite outings ever.",
        ),
        _t(
            25,
            "assistant",
            "Beach bonfires are magical. Sounds like your group has a really nice rhythm going.",
        ),
        _t(
            26,
            "user",
            "We do. I just want to be more intentional about it this year. Anyway, can you help me think of ideas for the next few months?",
        ),
        _t(27, "assistant", "Sure! What month are we planning for?"),
        _t(
            28,
            "user",
            "Let me think about the coming months first. Actually, I have a separate question. I need a gift for my mother-in-law's birthday. She's into gardening.",
        ),
        _t(
            29,
            "assistant",
            "A nice set of Japanese gardening tools, a pretty planter, or a gift card to a local nursery could all work well.",
        ),
        _t(
            30,
            "user",
            "I'll go with the tools. She'd love that. OK so back to planning. Let me tell you about some options for this summer first.",
        ),
        _t(31, "assistant", "Go for it!"),
        _t(
            32,
            "user",
            "I'm thinking a kayaking trip in June, a rooftop bar crawl in July, and maybe a camping trip in August. All outdoor stuff.",
        ),
        _t(
            33,
            "assistant",
            "Those all sound amazing! Any camping experience in the group?",
        ),
        _t(
            34,
            "user",
            "Mixed. Some have gear, some don't. We'd probably do a car camping spot with facilities. Nothing too rugged.",
        ),
        _t(
            35,
            "assistant",
            "Smart for a group. State parks with reserved sites are perfect - they have restrooms and fire pits.",
        ),
        _t(
            36,
            "user",
            "Exactly what I was thinking. OK so now for the harder question. My friend group wants to do something big for the holidays. Like a December thing.",
        ),
        _t(37, "assistant", "That's exciting! What are you thinking?"),
        _t(
            38,
            "user",
            "We want to do a holiday party. Big thing, maybe 20 people. The question is where.",
        ),
        _t(
            39,
            "assistant",
            "For 20 people you'd need a good-sized space. Options could include a restaurant private dining room, a rented event space, or someone's house if it's big enough.",
        ),
        _t(
            40,
            "user",
            "My friend Marcus has a huge house with a great backyard. We thought about doing it there. But I'm leaning towards renting a private room at a restaurant instead.",
        ),
        _t(
            41,
            "assistant",
            "Both are good options. A restaurant takes the hosting burden off Marcus and handles food and cleanup. But a house party is more personal and usually cheaper. What's making you lean restaurant?",
        ),
        _t(
            42,
            "user",
            "Honestly... I think it would just be easier for everyone. Less setup, less cleanup. Plus the restaurant has a great atmosphere. The place I'm looking at has this gorgeous private room with a fireplace.",
        ),
        _t(
            43,
            "assistant",
            "A private room with a fireplace sounds perfect for a holiday party. Very cozy. What restaurant?",
        ),
        _t(
            44,
            "user",
            "It's called The Fireside. They do a really nice prix fixe holiday menu. About $65 per person.",
        ),
        _t(
            45,
            "assistant",
            "$65 per person for 20 people is $1,300 total. That's reasonable for a nice holiday event split across the group. Want me to help you draft an invitation?",
        ),
        _t(
            46,
            "user",
            "Not yet, I need to confirm the date first. We're looking at December 14th or December 21st.",
        ),
        _t(
            47,
            "assistant",
            "The 14th gives more buffer before Christmas travel starts. The 21st is closer to the holiday but some people might already be leaving town.",
        ),
        _t(
            48,
            "user",
            "Good point. Let me poll the group. I'll go with the 14th if it works for most people.",
        ),
        _t(49, "assistant", "Sounds like a plan!"),
        _t(
            50,
            "user",
            "Hey, so my coworker suggested we do the December party outdoors instead. She said Marcus's backyard is gorgeous and we could do fire pits and string lights.",
        ),
        _t(
            51,
            "assistant",
            "An outdoor December party in Portland could be beautiful but also risky weather-wise. December in Portland averages around 40-45 degrees with frequent rain.",
        ),
        _t(
            52,
            "user",
            "Yeah... I don't think so. I'd rather not risk it. If it rains, the whole thing is ruined. And 40 degrees is miserable.",
        ),
        _t(
            53,
            "assistant",
            "The restaurant with the private room and fireplace is a much safer bet for December. No weather worries, built-in atmosphere, and comfortable for everyone.",
        ),
        _t(54, "user", "Exactly. Let me lock in The Fireside. December 14th it is."),
        _t(
            55,
            "assistant",
            "Great choice! A warm, cozy indoor holiday party. Want to start planning the details?",
        ),
        _t(
            56,
            "user",
            "Maybe next week. I need to call the restaurant first and confirm availability.",
        ),
        _t(57, "assistant", "Sounds good. Let me know how it goes!"),
        _t(
            58,
            "user",
            "Hey, quick update. The Fireside confirmed December 14th! Private room with fireplace, prix fixe menu, 20 guests. All set.",
        ),
        _t(59, "assistant", "Wonderful! That's going to be a fantastic party."),
        _t(
            60,
            "user",
            "I'm excited. Marcus was a little disappointed we're not doing his backyard but he understands. He said 'yeah you hate the cold, makes sense.' I can't help it!",
        ),
        _t(
            61,
            "assistant",
            "Everyone has their preferences! And honestly, your group will probably enjoy the restaurant setting more than shivering in a backyard.",
        ),
        _t(
            62,
            "user",
            "I think so too. Now I need to figure out the details - should we do a gift exchange?",
        ),
        _t(
            63,
            "assistant",
            "A white elephant exchange is always fun and keeps costs down. Set a spending limit like $25 so it stays lighthearted.",
        ),
        _t(
            64,
            "user",
            "Perfect idea. $25 limit, white elephant style. I'll add that to the invite.",
        ),
        _t(65, "assistant", "Everyone will love it. Anything else for the party?"),
        _t(
            66,
            "user",
            "I think that covers the big stuff. I'm already looking forward to summer activities too - I told the group about the kayaking and camping ideas and everyone's pumped.",
        ),
        _t(
            67,
            "assistant",
            "You'll have a great year of activities! A good mix of seasonal options.",
        ),
        _t(68, "user", "Yeah. Alright, I'm going to head out. Talk tomorrow!"),
        _t(69, "assistant", "Have a good evening!"),
    ]


def _build_absence_dietary() -> list[dict]:
    """User never says they're vegetarian, but all evidence points to it."""
    return [
        _t(
            0,
            "user",
            "I need help meal planning for the week. I always run out of ideas by Wednesday.",
        ),
        _t(
            1,
            "assistant",
            "I can definitely help! What kinds of meals do you typically enjoy?",
        ),
        _t(
            2,
            "user",
            "Lots of variety. I love Indian food, Mediterranean, Mexican, Asian. Pretty eclectic.",
        ),
        _t(
            3, "assistant", "Great range! Any dietary restrictions I should know about?"
        ),
        _t(
            4,
            "user",
            "Nothing major. I try to eat healthy but I'm not super strict about it.",
        ),
        _t(
            5,
            "assistant",
            "Got it. Let me suggest some meals then. How about starting Monday with a chicken tikka masala?",
        ),
        _t(
            6,
            "user",
            "Hmm, what about chana masala instead? I love the chickpea version.",
        ),
        _t(7, "assistant", "Chana masala is delicious! Great choice. Tuesday?"),
        _t(8, "user", "Something Mediterranean maybe. Falafel wraps?"),
        _t(9, "assistant", "Love it. Wednesday?"),
        _t(
            10,
            "user",
            "I'm thinking a stir-fry. Tofu with vegetables and peanut sauce over rice.",
        ),
        _t(
            11,
            "assistant",
            "Sounds great. Thursday - how about some tacos? You could do ground beef or chicken.",
        ),
        _t(
            12,
            "user",
            "Black bean tacos with avocado crema! That's one of my go-to recipes.",
        ),
        _t(13, "assistant", "Yum! And Friday?"),
        _t(
            14,
            "user",
            "Friday is usually pizza night. I make my own dough. This week I'm thinking a roasted vegetable pizza with caramelized onions.",
        ),
        _t(15, "assistant", "What a great week of meals! Saturday?"),
        _t(
            16,
            "user",
            "Saturday we're going out to that new Thai place. I've been eyeing their pad see ew.",
        ),
        _t(
            17,
            "assistant",
            "Pad see ew is wonderful. Are you getting the shrimp or chicken version?",
        ),
        _t(
            18,
            "user",
            "They have a version with crispy tofu that sounds amazing. The reviews rave about it.",
        ),
        _t(19, "assistant", "Crispy tofu pad see ew - nice! And Sunday?"),
        _t(
            20,
            "user",
            "Sunday I want to try making ramen from scratch. The whole thing - broth, noodles, toppings.",
        ),
        _t(
            21,
            "assistant",
            "Ambitious! For the broth, are you thinking tonkotsu (pork bone), shoyu (soy sauce), or miso-based?",
        ),
        _t(
            22,
            "user",
            "Miso ramen. I love that rich, savory flavor. I'll do a mushroom dashi base and load it up with corn, nori, soft-boiled egg, and scallions.",
        ),
        _t(
            23,
            "assistant",
            "Mushroom dashi with miso is incredible. That's going to be a wonderful bowl. The corn adds a nice sweetness.",
        ),
        _t(
            24,
            "user",
            "Can't wait. OK separate topic - I need a gift for my sister. She's turning 30.",
        ),
        _t(25, "assistant", "Big birthday! What's she into?"),
        _t(
            26,
            "user",
            "She's a runner. Like, marathons and everything. Very serious about it.",
        ),
        _t(
            27,
            "assistant",
            "A nice pair of running socks (Balega or Feetures), a massage gun, or a registration for a race she's been wanting to do could all be great.",
        ),
        _t(
            28,
            "user",
            "The race registration is a great idea! She's been talking about the Big Sur Marathon. I'll look into that.",
        ),
        _t(
            29,
            "assistant",
            "That would be an incredible gift. Big Sur is one of the most scenic marathons in the world.",
        ),
        _t(
            30,
            "user",
            "Yeah she'll love it. OK back to food stuff - I want to talk about my lunch game. I always pack lunch for work.",
        ),
        _t(31, "assistant", "Nice! What do you usually bring?"),
        _t(
            32,
            "user",
            "Grain bowls are my staple. Like, quinoa or rice with roasted vegetables, beans, and some kind of dressing. I rotate between tahini, chimichurri, and peanut sauce.",
        ),
        _t(
            33,
            "assistant",
            "That's a solid formula. Hearty, nutritious, and the dressing rotation keeps it interesting. Do you ever add protein like grilled chicken or salmon?",
        ),
        _t(
            34,
            "user",
            "The beans are my protein source. I do lentils, chickpeas, or black beans usually. Sometimes I'll throw in some edamame for variety.",
        ),
        _t(
            35,
            "assistant",
            "Legumes are a great protein source. You're getting good variety there.",
        ),
        _t(
            36,
            "user",
            "Yeah, I try to hit around 25-30g of protein per meal. Between the grains, legumes, and sometimes nuts, I get there.",
        ),
        _t(
            37,
            "assistant",
            "That's a solid target. Plant-based protein combining - grains plus legumes gives you a complete amino acid profile.",
        ),
        _t(
            38,
            "user",
            "Exactly. It works well for me. Hey, do you have any good snack recommendations? I crash around 3pm.",
        ),
        _t(
            39,
            "assistant",
            "For a 3pm energy boost, try hummus with carrots and celery, a handful of mixed nuts, or apple slices with almond butter. All provide sustained energy without the sugar crash.",
        ),
        _t(
            40,
            "user",
            "I do the hummus thing already. The apple and almond butter sounds great though. I'll add that to my rotation.",
        ),
        _t(
            41,
            "assistant",
            "It's one of those perfect snack combinations - fiber, healthy fat, protein, and natural sweetness.",
        ),
        _t(
            42,
            "user",
            "Speaking of food, my coworker invited me to a barbecue this weekend. He's grilling burgers and ribs. Not really my scene but I'll go to be social.",
        ),
        _t(
            43,
            "assistant",
            "That's nice of him! You could always bring a dish to share that you know you'll enjoy, in case the grill options aren't your thing.",
        ),
        _t(
            44,
            "user",
            "That's a good idea. I'll bring a big Mediterranean pasta salad. Everyone usually loves that. Plus I know I'll eat well regardless of what else is there.",
        ),
        _t(
            45,
            "assistant",
            "Perfect strategy. A good pasta salad always disappears at barbecues.",
        ),
        _t(
            46,
            "user",
            "True. I'll do orzo, sun-dried tomatoes, kalamata olives, red onion, fresh basil, and a lemon vinaigrette.",
        ),
        _t(
            47,
            "assistant",
            "That sounds restaurant-quality. Your coworker will be impressed.",
        ),
        _t(
            48,
            "user",
            "Ha, thanks. Cooking is my biggest hobby. I spend like 2 hours a day in the kitchen.",
        ),
        _t(
            49,
            "assistant",
            "That's dedication! Have you ever thought about taking professional cooking classes?",
        ),
        _t(
            50,
            "user",
            "I actually took a plant-based cooking course at the community college last year. Learned some amazing techniques for making vegetables the star of the plate.",
        ),
        _t(
            51,
            "assistant",
            "That sounds right up your alley! What was the most useful thing you learned?",
        ),
        _t(
            52,
            "user",
            "Honestly, roasting techniques. How to get that perfect caramelization on cauliflower, Brussels sprouts, sweet potatoes. Game-changer.",
        ),
        _t(
            53,
            "assistant",
            "High-heat roasting is one of the best ways to bring out natural sweetness and create amazing texture in vegetables. It's a technique that works for almost everything.",
        ),
        _t(
            54,
            "user",
            "Yep. I also learned a lot about umami sources. Like how miso, nutritional yeast, mushrooms, and soy sauce can create that deep savory flavor without needing any meat stock.",
        ),
        _t(
            55,
            "assistant",
            "Those are all excellent umami bombs. Nutritional yeast especially is underrated as a flavor enhancer.",
        ),
        _t(
            56,
            "user",
            "I put it on everything. Popcorn, pasta, scrambled eggs. It's amazing.",
        ),
        _t(
            57,
            "assistant",
            "Nutritional yeast popcorn is a revelation! Have you tried making cheese sauces with it?",
        ),
        _t(
            58,
            "user",
            "Yes! Cashew-based cheese sauce with nooch. It's my guilty pleasure. I could eat it by the spoonful.",
        ),
        _t(
            59,
            "assistant",
            "Not even guilty - cashew cheese sauce is delicious and wholesome. Great on pasta, nachos, or as a dip.",
        ),
        _t(
            60,
            "user",
            "OK let me ask about something else. I'm getting into composting and I need tips.",
        ),
        _t(
            61,
            "assistant",
            "Great for reducing food waste! What's your setup - do you have outdoor space?",
        ),
        _t(
            62,
            "user",
            "Small backyard. I'm thinking of getting one of those tumbler composters.",
        ),
        _t(
            63,
            "assistant",
            "Tumblers are great for smaller spaces. They speed up decomposition since you can turn them easily. Key ratio is 3 parts brown (leaves, cardboard) to 1 part green (food scraps).",
        ),
        _t(
            64,
            "user",
            "I generate a LOT of vegetable scraps from cooking. Like, onion peels, pepper cores, carrot tops, herb stems. Would that be enough greens?",
        ),
        _t(
            65,
            "assistant",
            "More than enough! With 2 hours of cooking daily, you're probably producing plenty of green material. Just avoid adding citrus peels in large quantities and no garlic or onion if you're vermicomposting.",
        ),
        _t(
            66,
            "user",
            "Good tips. I'll get the tumbler this weekend. Time to close the loop - grow veggies, cook them, compost the scraps, feed the garden.",
        ),
        _t(
            67,
            "assistant",
            "That's the perfect sustainable cycle. Your homegrown vegetables will taste even better knowing they were fed by your compost.",
        ),
        _t(
            68,
            "user",
            "Exactly. Alright, I need to prep tomorrow's lunch. Grain bowl time!",
        ),
        _t(69, "assistant", "Enjoy the prep! What's in tomorrow's bowl?"),
        _t(
            70,
            "user",
            "Farro, roasted beets and sweet potato, white beans, arugula, and a balsamic tahini dressing.",
        ),
        _t(
            71,
            "assistant",
            "That sounds gorgeous - the beet color against the arugula with that dressing... chef's kiss.",
        ),
        _t(72, "user", "Thanks! Talk tomorrow!"),
        _t(73, "assistant", "Have a great evening!"),
    ]


# ---------------------------------------------------------------------------
# Build all conversations
# ---------------------------------------------------------------------------


def build_conversations() -> dict[str, list[dict]]:
    """Return conversation_id -> list of turn dicts."""
    return {
        "puzzle_logic_1": _build_logic_seating(),
        "puzzle_logic_2": _build_logic_scheduling(),
        "puzzle_chain_1": _build_chain_plant(),
        "puzzle_chain_2": _build_chain_recipe(),
        "puzzle_state_1": _build_state_budget(),
        "puzzle_state_2": _build_state_diagnosis(),
        "puzzle_contradict_1": _build_contradiction_meeting(),
        "puzzle_explore_1": _build_exploration_health(),
        "puzzle_explore_2": _build_exploration_spending(),
        "puzzle_absence_1": _build_absence_seasonal(),
        "puzzle_absence_2": _build_absence_dietary(),
    }


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

QUESTIONS: list[dict] = [
    # ---------- LOGIC CONSTRAINT ----------
    {
        "conversation_id": "puzzle_logic_1",
        "category": "logic_constraint",
        "question": "Based on all constraints discussed, what is the final valid desk arrangement for the 6 desks?",
        "source_chat_ids": [6, 12, 24, 28, 29, 36, 44, 50, 52, 72, 74, 76],
        "ideal_response": "Final arrangement: Desk 1 - User (needs whiteboard, which is at the left end), Desk 2 - Alice (user works with her on frontend, and the Alice-Dave conflict was resolved when they cleared the air), Desk 3 - Dave (conflict with Alice resolved per turn 72), Desk 4 - Eve (middle desk per her preference, next to Carol for pair programming, near the plants between desks 3-4), Desk 5 - Carol (near server room on the right side, next to Eve for pair programming), Desk 6 - Bob (best natural light at the end, near IT closet). Key constraint updates: Alice and Dave's conflict was RESOLVED in turn 72 when Dave texted that they cleared the air, so the original constraint from turn 12 no longer applies.",
        "depth_needed": 3,
        "benchmark": "puzzle",
    },
    {
        "conversation_id": "puzzle_logic_1",
        "category": "logic_constraint",
        "question": "What were all the constraints for the desk arrangement, including which ones were eventually resolved or became irrelevant?",
        "source_chat_ids": [6, 12, 24, 28, 36, 44, 50, 72, 78, 80, 82],
        "ideal_response": "Constraints: (1) 6 desks in a row by windows, desk 1 is leftmost near whiteboard. (2) User needs desk 1 for whiteboard access. (3) Alice cannot sit next to Dave (Henderson argument) - RESOLVED in turn 72, they cleared the air. (4) Bob prefers desk 1 or 6 for best natural light (migraines under fluorescents) - desk 6 chosen since user has desk 1. (5) Carol needs to be near server room on right side - desk 5. (6) Eve cannot be at either end (foot traffic distracting) - needs middle desk. (7) Carol and Eve must be adjacent for pair programming. (8) Eve wants to be near the plants (between desks 3 and 4). (9) Bob wants desk 6 also for IT closet proximity. (10) User prefers Alice at desk 2 to collaborate on frontend.",
        "depth_needed": 3,
        "benchmark": "puzzle",
    },
    {
        "conversation_id": "puzzle_logic_2",
        "category": "logic_constraint",
        "question": "What is the final conference room schedule for next week, including all the changes that were made?",
        "source_chat_ids": [
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            30,
            34,
            40,
            42,
            44,
            48,
            52,
            58,
            62,
            67,
            70,
        ],
        "ideal_response": "Final Room A schedule: Monday 9-11am Marketing strategy (15 people, needs Room A), Monday 1-3pm Engineering sprint planning (20 people, needs Room A), Tuesday 10am-12pm HR all-hands (non-negotiable), Wednesday 2-4pm Marketing brainstorm (moved back to Wednesday after Sales rescheduled), Thursday 10am-3pm Sales client presentation (originally Wednesday, clients rescheduled to Thursday; needs AV setup in Room A), Friday 2-4pm tentative team-building. Room B schedule: Tuesday 2-4pm Engineering code review (5 people, fits Room B), Wednesday 9-10am Engineering stand-up (6 leads, moved from Room A), Friday 9-10am Engineering stand-up (6 leads, moved from Room A). Key changes: (1) Engineering stand-ups moved to Room B since only 6 people. (2) Marketing brainstorm was originally Wednesday 2-4, moved to Thursday 10-12 to avoid Sales conflict, then moved BACK to Wednesday 2-4 when Sales shifted to Thursday. (3) Sales presentation moved from Wednesday to Thursday per client request.",
        "depth_needed": 3,
        "benchmark": "puzzle",
    },
    # ---------- SEQUENTIAL CHAIN ----------
    {
        "conversation_id": "puzzle_chain_1",
        "category": "sequential_chain",
        "question": "What is likely causing the cloudiness in the fish tank, and how does it connect to other issues discussed?",
        "source_chat_ids": [2, 11, 14, 15, 17, 31, 40, 52, 54, 57, 58, 70, 78],
        "ideal_response": "The fish tank cloudiness is most likely caused by high mineral content in the tap water. The chain of evidence: (1) The user's tomato plants had brown, crispy leaf edges - a sign of salt/mineral burn in the soil. (2) The user's wife noticed white crusty deposits on the shower head - mineral buildup from hard water. (3) Neighbor Karen mentioned the city switched to a different reservoir, which doubled the total dissolved solids (450 ppm vs old 200 ppm). (4) The hard water was causing mineral accumulation in the plant soil. (5) Neighbor Karen's plants were fine because she used a ZeroWater filter that removes dissolved minerals. (6) User got a ZeroWater filter and plants recovered. (7) The fish tank was filled with the same unfiltered tap water and has persistent white cloudiness. (8) Testing confirmed 18 dGH water hardness, far too high for neon tetras (prefer 2-10 dGH). All these symptoms trace back to the reservoir switch increasing mineral content in the city water supply.",
        "depth_needed": 4,
        "benchmark": "puzzle",
    },
    {
        "conversation_id": "puzzle_chain_1",
        "category": "sequential_chain",
        "question": "Why did the user's neighbor's plants thrive while the user's plants were dying, even though they use the same tap water?",
        "source_chat_ids": [2, 14, 15, 17, 26, 28, 30, 31, 40, 41],
        "ideal_response": "The neighbor Karen's plants thrived because she runs her tap water through a ZeroWater pitcher filter before using it on plants. ZeroWater uses a 5-stage filtration system including ion exchange that removes dissolved minerals. The city's tap water has high mineral content (450 ppm TDS) due to a reservoir switch. The unfiltered water causes mineral/salt buildup in the soil, which burns plant roots and causes the brown crispy leaf edges the user observed. Karen's filtered water removes these minerals, protecting her plants. When the user also got a ZeroWater filter, their plants recovered with new healthy growth.",
        "depth_needed": 3,
        "benchmark": "puzzle",
    },
    {
        "conversation_id": "puzzle_chain_2",
        "category": "sequential_chain",
        "question": "What chain of discoveries led to successfully recreating the grandmother's lamb stew?",
        "source_chat_ids": [
            2,
            4,
            14,
            16,
            24,
            25,
            27,
            32,
            33,
            38,
            39,
            41,
            50,
            52,
            62,
            64,
        ],
        "ideal_response": "The chain of discoveries: (1) User remembered the stew had a 'complex, smoky warmth' that they couldn't replicate. Cast iron Dutch oven was not the key difference. (2) Found grandmother's leather journal written in Farsi in parents' attic. (3) Cousin Dariush translated recipe pages, revealing the ingredient 'golpar' (Persian hogweed seed / angelica seed) - a distinctive Iranian spice with warm, musky, earthy depth. (4) This triggered a memory of 'dark round things' grandmother cracked against the counter - identified as dried limes (limoo amani), which add tangy, fermented depth to Persian stews. (5) The journal described 'bloom the aromatics in hot fat until the kitchen sings' - the technique of blooming spices in oil to release essential oils, which the user had been skipping. (6) Hamid at Sahel Persian Market confirmed the stew sounded like a khoresht variation and revealed the critical detail: it must simmer LOW AND SLOW for 3-4 hours minimum, not the 90 minutes the user had been doing. (7) The combination of golpar, dried limes, spice blooming, and extended cooking time recreated the authentic flavor perfectly.",
        "depth_needed": 4,
        "benchmark": "puzzle",
    },
    # ---------- STATE CHANGE ----------
    {
        "conversation_id": "puzzle_state_1",
        "category": "state_change",
        "question": "Walk me through every budget change for the mobile app project, from initial allocation to final numbers.",
        "source_chat_ids": [
            2,
            6,
            16,
            18,
            32,
            34,
            36,
            40,
            42,
            46,
            48,
            60,
            62,
            66,
            68,
            69,
        ],
        "ideal_response": "Budget changes in order: (1) Initial allocation: $50K total. Split: $30K dev, $10K design, $5K QA, $5K marketing. (2) VP approved additional $20K, bringing total to $70K. Extra allocated: +$10K dev, +$5K marketing, +$5K contingency. New split: $40K dev, $10K design, $5K QA, $10K marketing, $5K contingency. (3) Push notification vendor doubled pricing from $2K to $4K, extra $2K taken from contingency. Contingency drops to $3K. (4) Company-wide $15K cut due to weak Q2 numbers. Total drops to $55K. Cuts: marketing $10K->$4K (-$6K), contingency eliminated (-$3K), Miguel hours reduced 200->175 (-$3,750), design $10K->$7,750 (-$2,250). (5) Recalculation to $55K: Dev $35,250, Design $7,750, QA $5K, Marketing $4K. Found $3K unallocated, added to marketing. Final: Dev $35,250, Design $7,750, QA $5K, Marketing $7K. (6) AWS enterprise deal saves $10K, added back. Total now $65K. Miguel back to 200hrs (+$3,750), contingency restored to $5K, $1,250 added to design. (7) QA saves $1K with open-source tool, QA drops to $4K, extra $1K to marketing. FINAL: Dev $39K, Design $9K, QA $4K, Marketing $8K, Contingency $5K. Total $65K.",
        "depth_needed": 3,
        "benchmark": "puzzle",
    },
    {
        "conversation_id": "puzzle_state_1",
        "category": "state_change",
        "question": "What is the current project budget total and how does it compare to the initial allocation?",
        "source_chat_ids": [2, 16, 36, 60, 66, 68, 69],
        "ideal_response": "The current total budget is $65,000, which is $15K more than the initial $50K allocation. It went through several changes: started at $50K, went up to $70K after VP approval, was cut to $55K due to weak Q2 numbers, then restored to $65K thanks to $10K in AWS savings. The final breakdown is: Development $39K, Design $9K, QA $4K, Marketing $8K, Contingency $5K.",
        "depth_needed": 2,
        "benchmark": "puzzle",
    },
    {
        "conversation_id": "puzzle_state_2",
        "category": "state_change",
        "question": "Summarize the evolving diagnosis of the car's grinding noise, including what was ruled out and what the final answer was.",
        "source_chat_ids": [2, 10, 14, 15, 18, 24, 25, 26, 34, 35, 38, 39],
        "ideal_response": "The diagnosis evolved through several stages: (1) Initial suspicion: worn brake pads, based on grinding when braking. (2) Mechanic inspection ruled out brake pads - they had 60% life remaining. Mechanic couldn't reproduce the noise. (3) Key detail emerged: noise only occurred in the first few minutes after the car sat overnight, then stopped. This was identified as surface rust on the front brake rotors - a normal, harmless condition in humid environments. (4) NEW symptom appeared: grinding from the REAR of the car that was persistent and didn't go away after initial braking. This was a separate issue from the front. (5) Final diagnosis for rear grinding: a broken return spring in the rear left drum brake, causing the brake shoe to drag constantly against the drum. (6) Resolution: two separate issues. Front = harmless surface rust on disc rotors (normal). Rear = broken return spring in drum brake (repaired for $180, drum resurfaced). Total cost including initial inspection: $220.",
        "depth_needed": 3,
        "benchmark": "puzzle",
    },
    # ---------- CONTRADICTION ----------
    {
        "conversation_id": "puzzle_contradict_1",
        "category": "contradiction",
        "question": "Where is the company retreat being held and when does it start?",
        "source_chat_ids": [2, 6, 20, 22, 32, 34, 36, 38, 40, 42],
        "ideal_response": "The retreat is at Lakewood Resort, April 18-20, starting at 10am on Friday. Resolving the contradictions: (1) Venue: Tom said Lakewood, Sarah said Ridgeview Lodge. Diana (Sarah's boss) confirmed Lakewood - Ridgeview was the original booking but had to switch due to a double-booking. (2) Dates: Tom and Sarah both said April 18-20. Priya overheard VP Marcus say April 25-27 but that was confirmed wrong (she may have misheard or Marcus was discussing something else). (3) Start time: Diana's official email says 10am. Tom said 8am for an engineering pre-session, but then admitted he confused it with a separate May offsite event. There is no 8am engineering session for this retreat.",
        "depth_needed": 3,
        "benchmark": "puzzle",
    },
    {
        "conversation_id": "puzzle_contradict_1",
        "category": "contradiction",
        "question": "Which sources of information about the retreat turned out to be wrong, and why?",
        "source_chat_ids": [6, 14, 20, 22, 32, 42, 46, 48, 50, 60, 65, 66],
        "ideal_response": "Wrong information and reasons: (1) Sarah (HR) said Ridgeview Lodge - this was based on the original booking before it was changed to Lakewood due to a double-booking. Outdated info, not intentionally wrong. (2) Priya said April 25-27 - she overheard VP Marcus in the break room but admitted she might have misheard or he could have been discussing something else. Secondhand, uncertain info. (3) Tom said 8am start for engineering - he confused it with a separate May offsite. Mixed up two events. (4) Jake said single rooms - he was assuming it was the same as last year. Extrapolating from outdated experience. (5) Jake said mandatory ropes course - same thing, assuming last year's format. The draft agenda showed nature hike and kayaking (both optional). Diana from HR was the most reliable source throughout, being confirmed correct every time. Jake was wrong twice, both times because he assumed this year = last year.",
        "depth_needed": 3,
        "benchmark": "puzzle",
    },
    # ---------- OPEN EXPLORATION ----------
    {
        "conversation_id": "puzzle_explore_1",
        "category": "open_exploration",
        "question": "Based on all the patterns in our conversations, are there any health concerns about the user I should flag?",
        "source_chat_ids": [
            0,
            2,
            10,
            12,
            14,
            16,
            28,
            30,
            38,
            40,
            42,
            43,
            52,
            54,
            55,
            72,
            74,
            75,
            77,
            82,
            84,
        ],
        "ideal_response": "Several health patterns worth flagging: (1) RECURRING MONDAY HEADACHES: The user reported headaches on three separate Mondays (turns 0, 38, 72). The pattern is specific to Mondays, characterized by pressure behind eyes and temples. (2) LIKELY CONTRIBUTING FACTORS: (a) Changed office environment - moved from floor 3 (near window, natural light) to floor 7 (interior desk, harsh fluorescent lighting, poor ergonomic setup). (b) Weekend-to-weekday transition - sleeps 9 hours on weekends vs 7 on weekdays, drinks less coffee on weekends then resumes 4 cups Monday, creating a caffeine rebound cycle. (c) Increased stress from K8s migration project with tight deadlines. (3) AFTERNOON ENERGY CRASHES: New since moving floors, likely related to reduced natural light exposure and fluorescent lighting disrupting circadian cues. The user feels better on WFH days when sitting by a window. (4) PHYSICAL COMPLAINTS: Neck pain from poor ergonomic setup (no adjustable desk, hunching over laptop), lower back pain from cheap chairs without lumbar support on floor 7. (5) The office environment change is a major unifying factor - headaches, energy crashes, neck pain, and back pain all worsened after the floor move.",
        "depth_needed": 4,
        "benchmark": "puzzle",
    },
    {
        "conversation_id": "puzzle_explore_2",
        "category": "open_exploration",
        "question": "What patterns do you notice in the user's spending behavior, and what advice would you give?",
        "source_chat_ids": [
            0,
            4,
            6,
            8,
            14,
            16,
            20,
            32,
            38,
            42,
            44,
            46,
            47,
            48,
            50,
            51,
            52,
            53,
            54,
            62,
            63,
        ],
        "ideal_response": "Key spending patterns: (1) PURCHASE CLUSTERING: The user made $725 in discretionary purchases in a short period (keyboard $180, cat supplies $250, adoption $75, Raspberry Pi $120, VPN $100). Each purchase lowered the psychological barrier to the next one - the 'dam breaking' effect. (2) STRESS-DRIVEN SPENDING: The user explicitly connected work stress to eating out more and to retail therapy. When stressed from work, the user defaults to spending instead of other outlets. (3) FORGOTTEN SUBSCRIPTIONS: $35/month in subscriptions they forgot about (Spotify + Audible). Small leaks in the budget. (4) IMPULSE VS PLANNED: The user frames purchases as 'impulse' but actually researches them (keyboard researched for months, Pi kit inspired by YouTube video). The spending feels impulsive because it clusters, not because the individual purchases are unconsidered. (5) IMPACT ON SAVINGS GOAL: Saving $2K/month toward $60K house down payment ($35K saved so far). The $725 discretionary month represents more than a third of monthly savings capacity. Advice: (a) Implement the 48-hour rule for purchases over $50. (b) Keep a 'want list' and revisit monthly. (c) Cancel unused subscriptions immediately. (d) Recognize stress as a spending trigger and channel energy into free/already-paid-for activities (hiking, Pi projects, cooking). (e) The individual purchases aren't problematic - the clustering pattern is.",
        "depth_needed": 3,
        "benchmark": "puzzle",
    },
    # ---------- ABSENCE INFERENCE ----------
    {
        "conversation_id": "puzzle_absence_1",
        "category": "absence_inference",
        "question": "Why did the user choose an indoor restaurant venue for the December holiday party instead of their friend Marcus's backyard?",
        "source_chat_ids": [2, 4, 8, 10, 12, 14, 18, 20, 22, 40, 42, 50, 52, 54, 60],
        "ideal_response": "The user chose an indoor venue because they strongly dislike being cold and avoid outdoor activities in winter. This is never stated as the explicit reason for choosing the restaurant, but it's clear from the pattern of behavior: (1) All summer activities (June BBQ, July outdoor concert, August hiking, September soccer) were outdoors. (2) All fall/winter activities (October pumpkin carving, November Friendsgiving, January game night, February cooking class) were indoors. (3) The user explicitly said 'I don't really do much outside in the winter. I kind of hate being cold' and that friends joke they 'hibernate from November to March.' (4) When a coworker suggested the outdoor backyard party with fire pits, the user rejected it, citing weather risk (rain, 40 degrees). (5) Marcus himself acknowledged it by saying 'yeah you hate the cold, makes sense.' The user gave practical reasons (weather risk, easier logistics) but the underlying driver is their consistent pattern of avoiding cold-weather outdoor activities.",
        "depth_needed": 3,
        "benchmark": "puzzle",
    },
    {
        "conversation_id": "puzzle_absence_2",
        "category": "absence_inference",
        "question": "Based on the conversation, does the user follow any specific dietary pattern? What evidence supports your conclusion?",
        "source_chat_ids": [
            6,
            10,
            12,
            14,
            18,
            22,
            32,
            34,
            36,
            42,
            44,
            46,
            50,
            54,
            56,
            58,
            64,
            70,
        ],
        "ideal_response": "The user is almost certainly vegetarian (possibly fully plant-based), though they never explicitly say so. Evidence: (1) When offered chicken tikka masala, they chose chana masala (chickpeas) instead. (2) Stir-fry choice was tofu, not meat. (3) Chose black bean tacos instead of beef or chicken tacos. (4) Pizza with roasted vegetables, no meat toppings. (5) Chose crispy tofu pad see ew instead of shrimp or chicken. (6) Ramen was mushroom dashi miso base, not tonkotsu (pork). (7) Lunch grain bowls use beans as protein, explicitly stated 'the beans are my protein source' when asked about adding chicken or salmon. (8) Described their protein strategy as plant-based: 'grains plus legumes gives you a complete amino acid profile.' (9) Took a 'plant-based cooking course.' (10) Knows about using miso, nutritional yeast, mushrooms for umami 'without needing any meat stock.' (11) Makes cashew-based cheese sauces. (12) For a barbecue with burgers and ribs, said 'not really my scene' and brought a pasta salad to ensure they'd eat well. (13) Composting scraps described as all vegetable matter. (14) When asked about dietary restrictions, said 'nothing major' - suggesting they don't frame vegetarianism as a restriction but as their normal way of eating.",
        "depth_needed": 3,
        "benchmark": "puzzle",
    },
    {
        "conversation_id": "puzzle_absence_2",
        "category": "absence_inference",
        "question": "If I were cooking dinner for this user, what should I keep in mind?",
        "source_chat_ids": [6, 10, 12, 18, 22, 34, 36, 42, 44, 50, 54, 58],
        "ideal_response": "You should prepare a vegetarian/plant-based meal. Although the user never explicitly says they're vegetarian, every single meal choice across the entire conversation avoids meat and fish: chickpea curry, tofu stir-fry, black bean tacos, vegetable pizza, tofu pad see ew, mushroom-based miso ramen, bean-and-grain lunch bowls. They get protein from legumes, tofu, and nuts. They said a coworker's burger-and-ribs barbecue was 'not really my scene.' They took a plant-based cooking course and make cashew cheese sauce. They're a skilled and enthusiastic cook who values flavor and variety, so a well-prepared plant-based meal would be ideal. They enjoy Indian, Mediterranean, Mexican, and Asian cuisines. Umami-rich preparations with mushrooms, miso, and nutritional yeast would be appreciated.",
        "depth_needed": 2,
        "benchmark": "puzzle",
    },
]


def build_questions() -> list[dict]:
    """Return the questions list with question_index added."""
    questions = []
    for i, q in enumerate(QUESTIONS):
        entry = dict(q)
        entry["question_index"] = i
        questions.append(entry)
    return questions


# ---------------------------------------------------------------------------
# Embedding and output
# ---------------------------------------------------------------------------


def truncate_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    texts = [truncate_text(t) for t in texts]
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        print(
            f"  Embedding batch {start // BATCH_SIZE + 1} ({len(batch)} texts)...",
            flush=True,
        )
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        time.sleep(0.1)
    return np.array(all_embeddings, dtype=np.float32)


def main() -> None:
    client = OpenAI()

    # Build conversations
    print("Building puzzle conversations...")
    conversations = build_conversations()
    total_turns = sum(len(turns) for turns in conversations.values())
    for conv_id, turns in conversations.items():
        print(f"  {conv_id}: {len(turns)} turns")
    print(f"  Total: {total_turns} turns across {len(conversations)} conversations")

    # Build questions
    print("\nBuilding questions...")
    questions = build_questions()
    print(f"  {len(questions)} questions generated")
    from collections import Counter

    category_counts = Counter(q["category"] for q in questions)
    for cat, count in sorted(category_counts.items()):
        print(f"    {cat}: {count}")

    # Flatten all turns into segments
    all_segments: list[dict] = []
    for conv_id, turns in conversations.items():
        for turn in turns:
            all_segments.append(
                {
                    "conversation_id": conv_id,
                    "turn_id": turn["turn_id"],
                    "role": turn["role"],
                    "text": turn["text"],
                }
            )
    print(f"\n{len(all_segments)} total segments to embed")

    # Embed
    texts = [s["text"] for s in all_segments]
    print(f"Embedding {len(texts)} segments...")
    embeddings = embed_texts(client, texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save NPZ
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / "segments_puzzle.npz"
    np.savez(
        npz_path,
        embeddings=embeddings,
        conversation_ids=np.array([s["conversation_id"] for s in all_segments]),
        turn_ids=np.array([s["turn_id"] for s in all_segments], dtype=np.int32),
        roles=np.array([s["role"] for s in all_segments]),
        texts=np.array([s["text"] for s in all_segments]),
    )
    print(f"\nSaved segments to {npz_path}")

    # Save questions JSON
    json_path = OUTPUT_DIR / "questions_puzzle.json"
    with open(json_path, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Saved {len(questions)} questions to {json_path}")

    # Verification
    print("\n--- Verification ---")
    d = np.load(npz_path, allow_pickle=True)
    print(f"NPZ keys: {list(d.keys())}")
    print(f"Shapes: {dict({k: d[k].shape for k in d.keys()})}")
    print(f"Conversation IDs: {sorted(set(d['conversation_ids']))}")
    print(f"Embedding dtype: {d['embeddings'].dtype}")
    print(f"Embedding norm (first): {np.linalg.norm(d['embeddings'][0]):.4f}")

    with open(json_path) as f:
        loaded_q = json.load(f)
    print(f"Questions loaded: {len(loaded_q)}")
    cats = Counter(q["category"] for q in loaded_q)
    print(f"Category breakdown: {dict(sorted(cats.items()))}")


if __name__ == "__main__":
    main()
