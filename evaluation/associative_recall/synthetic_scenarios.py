"""Synthetic conversation data and task scenarios for testing deep retrieval.

Tests cases where simple cosine similarity fails and the system needs to reason
about what to retrieve: conjunction cues, completeness, inference-driven retrieval,
proactive recall, and procedural checklists.
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
# 1. Synthetic conversations
# ---------------------------------------------------------------------------


def build_conversations() -> dict[str, list[dict]]:
    """Return conversation_id -> list of turn dicts."""
    conversations = {}

    # ===== CONVERSATION 1: Personal assistant (food, allergies, schedules) =====
    conversations["synth_personal"] = _build_personal_assistant_conversation()

    # ===== CONVERSATION 2: Work / project management =====
    conversations["synth_work"] = _build_work_conversation()

    # ===== CONVERSATION 3: Medical / health =====
    conversations["synth_medical"] = _build_medical_conversation()

    # ===== CONVERSATION 4: Event planning =====
    conversations["synth_planning"] = _build_planning_conversation()

    # ===== CONVERSATION 5: Technical / home setup =====
    conversations["synth_technical"] = _build_technical_conversation()

    return conversations


def _turn(turn_id: int, role: str, text: str) -> dict:
    return {"turn_id": turn_id, "role": role, "text": text}


def _build_personal_assistant_conversation() -> list[dict]:
    """Personal assistant: food preferences, allergies, schedules, relationships."""
    turns = [
        _turn(
            0,
            "user",
            "Hey, good morning! Can you remind me what I have going on today?",
        ),
        _turn(
            1,
            "assistant",
            "Good morning! Let me check. You have a dentist appointment at 10am and then lunch with Marcus at 12:30.",
        ),
        _turn(
            2,
            "user",
            "Right, Marcus. He's my college roommate - haven't seen him in years. We're going to that new Thai place on 5th street.",
        ),
        _turn(
            3,
            "assistant",
            "Sounds fun! Thai Orchid on 5th? I've heard great things about it.",
        ),
        _turn(
            4,
            "user",
            "Yeah that's the one. Oh by the way, remind me to pick up Bob's birthday present. His birthday is next Thursday.",
        ),
        _turn(
            5,
            "assistant",
            "Got it! Bob's birthday is next Thursday. Any gift ideas in mind?",
        ),
        _turn(
            6,
            "user",
            "Not yet. He's really into craft beer and woodworking. Maybe something related to one of those.",
        ),
        _turn(
            7,
            "assistant",
            "A nice set of carving chisels or a craft beer subscription could work well. Want me to look into options?",
        ),
        _turn(8, "user", "Sure, but not now. Let me get through today first."),
        _turn(
            9, "assistant", "Of course! Anything else before your dentist appointment?"
        ),
        _turn(
            10,
            "user",
            "Actually yes - I need to plan a dinner party for Saturday. It's going to be a small gathering.",
        ),
        _turn(11, "assistant", "How many people are you thinking?"),
        _turn(
            12,
            "user",
            "About 8 people. Bob and his wife Linda are coming, plus my coworkers Sarah and James, and my neighbors the Patels - that's Priya and Dev.",
        ),
        _turn(
            13,
            "assistant",
            "Nice group! That's 8 including you. Should I help with the menu?",
        ),
        _turn(
            14,
            "user",
            "Yeah but first let me tell you the dietary stuff. Bob is allergic to peanuts. He's been allergic since childhood.",
        ),
        _turn(15, "assistant", "Noted - peanut allergy for Bob. Anyone else?"),
        _turn(
            16,
            "user",
            "Sarah is vegetarian. She's been vegetarian for about 5 years now. Oh and she doesn't eat eggs either actually, so more like vegan for cooking purposes.",
        ),
        _turn(
            17,
            "assistant",
            "Got it - Sarah is effectively vegan. I'll keep that in mind for the menu.",
        ),
        _turn(
            18,
            "user",
            "The Patels don't eat beef - they're Hindu. But everything else is fine for them.",
        ),
        _turn(
            19,
            "assistant",
            "Understood. No beef for Priya and Dev Patel. What about James and Linda?",
        ),
        _turn(
            20,
            "user",
            "James eats everything, he's the easiest. Linda I'm not sure about, I should ask Bob.",
        ),
        _turn(
            21,
            "assistant",
            "Good idea. So far: peanut allergy (Bob), vegan-ish (Sarah), no beef (Patels), no known restrictions (James), TBD (Linda). Want me to draft some menu ideas based on what we know?",
        ),
        _turn(
            22,
            "user",
            "Actually hold on, I need to deal with something for work first. My boss just emailed about the Q3 report.",
        ),
        _turn(
            23,
            "assistant",
            "No problem! Take care of that and we can come back to the dinner planning whenever you're ready.",
        ),
        _turn(
            24,
            "user",
            "Ok so the Q3 report - I need the revenue figures broken down by region. Can you pull up what we discussed last week?",
        ),
        _turn(
            25,
            "assistant",
            "I don't have access to your work documents, but I remember you mentioned the APAC numbers were surprisingly strong. Would you like me to help organize the report structure?",
        ),
        _turn(
            26,
            "user",
            "No that's fine, I'll handle it. Just needed to remind myself. Let me get back to the dinner party.",
        ),
        _turn(
            27,
            "assistant",
            "Sure! Where were we - menu planning with the dietary restrictions.",
        ),
        _turn(
            28,
            "user",
            "Right. Oh I just texted Bob about Linda and he says she's fine with everything except she really doesn't like mushrooms. Not allergic, just hates them.",
        ),
        _turn(
            29,
            "assistant",
            "Got it - no mushrooms for Linda, preference not allergy. Updated the list.",
        ),
        _turn(
            30,
            "user",
            "Actually you know what, I just realized I should invite my sister Rachel too. She lives nearby and would love to come. So that's 9 people total.",
        ),
        _turn(31, "assistant", "Great! Any dietary notes for Rachel?"),
        _turn(
            32,
            "user",
            "Rachel is gluten-free. She was diagnosed with celiac disease last year. It's the real deal, not a preference thing - even trace amounts bother her.",
        ),
        _turn(
            33,
            "assistant",
            "Important to note - celiac disease for Rachel means we need to be very careful about cross-contamination with gluten, not just avoiding obvious wheat products.",
        ),
        _turn(
            34,
            "user",
            "Exactly. She brings her own soy sauce to restaurants because regular soy sauce has wheat in it.",
        ),
        _turn(
            35,
            "assistant",
            "That's a good detail. I'll make sure any sauces in the menu are gluten-free. Tamari instead of regular soy sauce, for example.",
        ),
        _turn(
            36,
            "user",
            "Perfect. Ok so can you also remind me - when is Mom's anniversary dinner? I know it's coming up.",
        ),
        _turn(
            37,
            "assistant",
            "You mentioned your parents' anniversary is on the 28th. Were you planning something special?",
        ),
        _turn(
            38,
            "user",
            "Yeah we're all going to Giovanni's, that Italian place Mom loves. That's a separate thing from Saturday's dinner.",
        ),
        _turn(
            39,
            "assistant",
            "Two events to plan around then! Let's focus on Saturday's dinner party. With the restrictions we have, I'm thinking a build-your-own taco bar could work well - naturally gluten-free with corn tortillas, can have plant-based proteins for Sarah, chicken or fish options (no beef), and easy to keep peanut-free.",
        ),
        _turn(
            40,
            "user",
            "Hmm maybe. Let me think about it. Oh I just got a text from Bob - apparently he found out he's lactose intolerant now. His doctor just told him last week.",
        ),
        _turn(
            41,
            "assistant",
            "Oh, that's a new one to add. So Bob now has: peanut allergy AND lactose intolerance. That does narrow things a bit more for cheese-heavy dishes.",
        ),
        _turn(
            42,
            "user",
            "Yeah. He said mild stuff is ok, like aged cheeses or butter, but no straight milk or cream or soft cheese.",
        ),
        _turn(
            43,
            "assistant",
            "That's helpful context - aged cheeses and butter are OK, just no milk, cream, or soft cheeses. So a cream-based sauce would be out for Bob, but a dish with parmesan sprinkled on top might be fine.",
        ),
        _turn(
            44,
            "user",
            "Right. Ok I need to run to the dentist now. Let's pick this up later.",
        ),
        _turn(
            45,
            "assistant",
            "Sounds good! Good luck at the dentist. I'll have some menu ideas ready when you're back.",
        ),
        _turn(
            46,
            "user",
            "I'm back. Dentist was fine, no cavities. Now I'm heading to lunch with Marcus.",
        ),
        _turn(
            47,
            "assistant",
            "Great news about the dentist! Enjoy your lunch with Marcus at Thai Orchid.",
        ),
        _turn(
            48,
            "user",
            "Thanks! Oh by the way, I forgot to mention that Bob's daughter Emma might tag along to the dinner on Saturday. She's 8.",
        ),
        _turn(
            49,
            "assistant",
            "Fun! So potentially 10 people now. Any food things to know about Emma?",
        ),
        _turn(
            50,
            "user",
            "She has a tree nut allergy. Pretty serious one too - Bob carries an EpiPen for her. It's different from Bob's peanut allergy since peanuts are legumes, not tree nuts, but you still have to be careful.",
        ),
        _turn(
            51,
            "assistant",
            "Very important distinction. So we need to avoid tree nuts (almonds, walnuts, cashews, pecans, etc.) for Emma AND peanuts for Bob. Even though they're different allergens, we should probably just avoid all nuts to be safe.",
        ),
        _turn(52, "user", "Good thinking. Yeah let's just do no nuts of any kind."),
        _turn(
            53,
            "assistant",
            "Agreed - a blanket no-nut policy for the dinner. I'll make sure nothing has peanuts, tree nuts, or nut-derived oils.",
        ),
        _turn(
            54,
            "user",
            "So I had lunch with Marcus and we caught up. He's working at a startup now doing AI stuff. He mentioned something about how his company uses some retrieval system. Random but interesting.",
        ),
        _turn(55, "assistant", "That is interesting! How was the food at Thai Orchid?"),
        _turn(
            56,
            "user",
            "Really good actually. I had the pad see ew and Marcus got the green curry. We should go back sometime.",
        ),
        _turn(
            57,
            "assistant",
            "Noted - Thai Orchid is a good one to remember for future meals out. Now, should we get back to Saturday's dinner planning?",
        ),
        _turn(
            58,
            "user",
            "Yes please. But actually, I want to switch gears for a sec. I need to figure out what to get Rachel for Christmas. I know it's months away but I like to plan ahead.",
        ),
        _turn(
            59, "assistant", "Smart to plan ahead! What sorts of things is Rachel into?"
        ),
        _turn(
            60,
            "user",
            "She's really into pottery. She takes a class on Tuesday nights. She also likes hiking and she's been getting into birdwatching recently.",
        ),
        _turn(
            61,
            "assistant",
            "Nice range of interests! A good pottery tool set, some quality binoculars for birdwatching, or gear for hiking could all work. Want me to bookmark some ideas?",
        ),
        _turn(
            62,
            "user",
            "Sure, put a pin in it. Back to Saturday - oh wait, I also wanted to tell you that Bob mentioned he's been doing keto for the last month. So he's trying to keep carbs low.",
        ),
        _turn(
            63,
            "assistant",
            "Keto on top of peanut allergy and lactose intolerance - Bob's got quite the restriction list! Low carb, no peanuts, and limited dairy. That does shape what we can serve him.",
        ),
        _turn(
            64,
            "user",
            "Haha yeah, Bob's always been the complicated one with food. But he said don't worry too much about the keto thing, if the dinner doesn't accommodate it perfectly he'll manage. The allergies are the non-negotiable part.",
        ),
        _turn(
            65,
            "assistant",
            "Makes sense - allergies are safety issues, keto is a preference he can flex on. So hard requirements for Bob: no peanuts, limited dairy (aged cheese/butter OK). Soft preference: low carb.",
        ),
        _turn(
            66,
            "user",
            "Exactly. Ok so let's actually pick a menu now. What do you think?",
        ),
        _turn(
            67,
            "assistant",
            "Let me compile all the restrictions: peanut allergy (Bob), tree nut allergy (Emma), lactose intolerance (Bob, aged cheese/butter OK), vegan (Sarah), no beef (Patels), no mushrooms (Linda), celiac/gluten-free (Rachel), keto preference (Bob, flexible). How about: Appetizer - bruschetta on gluten-free bread with tomatoes, basil, olive oil. Main - grilled salmon with lemon herb sauce, plus a hearty vegan black bean and sweet potato stew. Sides - roasted vegetables, rice, green salad. All naturally gluten-free, nut-free, no beef, no mushrooms.",
        ),
        _turn(
            68,
            "user",
            "That sounds pretty good. But wait, is bruschetta going to work for Rachel? Even on gluten-free bread, isn't there a cross-contamination risk with the toasting?",
        ),
        _turn(
            69,
            "assistant",
            "Good catch. If you use a dedicated gluten-free bread and toast it separately from any regular bread, it should be fine. Or we could do crostini-style using rice crackers instead.",
        ),
        _turn(70, "user", "Rice crackers work. Let's do that. What about dessert?"),
        _turn(
            71,
            "assistant",
            "For dessert, how about a fruit sorbet? It's naturally dairy-free (for Bob and Sarah), gluten-free (for Rachel), and we can pick a nut-free variety. Mango sorbet or mixed berry would be crowd-pleasers.",
        ),
        _turn(
            72,
            "user",
            "Love it. Mango sorbet. Ok I think we have a plan. Can you write up the full menu so I can make a shopping list?",
        ),
        _turn(
            73,
            "assistant",
            "Here's the full menu for Saturday's dinner party (10 guests including you): Appetizer - Rice crackers with fresh bruschetta topping (tomatoes, basil, garlic, olive oil). Main courses - Herb-crusted grilled salmon with lemon dill sauce / Vegan black bean and sweet potato stew (for Sarah and as a side for everyone). Sides - Roasted seasonal vegetables (no mushrooms) / Jasmine rice / Mixed green salad with balsamic vinaigrette. Dessert - Mango sorbet. Drinks - TBD. All dishes are gluten-free, nut-free, beef-free, and mushroom-free. The salmon course avoids dairy. Want me to generate the shopping list?",
        ),
        _turn(
            74,
            "user",
            "Yes please! Oh also, I should mention - Dev Patel told me he's bringing a bottle of wine, which is nice. And I want to have some non-alcoholic options too since Emma is coming.",
        ),
        _turn(
            75,
            "assistant",
            "That's thoughtful of Dev! For non-alcoholic options, you could do sparkling water with fruit, a homemade lemonade, or some juice boxes for Emma. I'll add those to the shopping list.",
        ),
        _turn(
            76, "user", "Sparkling water and lemonade sound great. Ok make the list."
        ),
        _turn(
            77,
            "assistant",
            "Shopping list for Saturday dinner: Proteins - salmon fillets (enough for 8-9 adults). Produce - tomatoes, basil, garlic, sweet potatoes, seasonal vegetables (zucchini, bell peppers, carrots - no mushrooms), mixed greens, lemons, limes, mangoes (or pre-made mango sorbet). Pantry - black beans (canned or dried), rice crackers, jasmine rice, olive oil, balsamic vinegar, dried dill, salt, pepper. Drinks - sparkling water, lemons for lemonade, sugar. Check labels for hidden nuts, gluten, and dairy on any packaged items. Anything I'm missing?",
        ),
        _turn(
            78,
            "user",
            "Looks good! I'll go shopping tomorrow. Thanks for all the help today.",
        ),
        _turn(
            79,
            "assistant",
            "You're welcome! Have a great evening. Don't forget about Bob's birthday present - his birthday is next Thursday!",
        ),
        _turn(80, "user", "Oh right! I'll think about that tomorrow too. Good night!"),
        _turn(
            81, "assistant", "Good night! Let me know if you need anything tomorrow."
        ),
        _turn(
            82,
            "user",
            "Morning! Quick question - do you remember the name of that restaurant we talked about for Mom's anniversary?",
        ),
        _turn(
            83,
            "assistant",
            "Good morning! Yes, it's Giovanni's - the Italian place your mom loves. The anniversary dinner is on the 28th.",
        ),
        _turn(
            84,
            "user",
            "Perfect thanks. Oh also, I was just texting with Sarah and she mentioned she also has a soy allergy. I didn't know that! Should we adjust the menu?",
        ),
        _turn(
            85,
            "assistant",
            "Oh that's important! Soy allergy on top of being vegan does make things trickier. The black bean stew should be fine since it doesn't use soy. But let's make sure the rice crackers don't contain soy, and check any other packaged items. The tamari sauce I suggested earlier is soy-based, so we'd need to skip that.",
        ),
        _turn(
            86,
            "user",
            "Good point about the tamari. What can we use instead for the Asian-ish flavors?",
        ),
        _turn(
            87,
            "assistant",
            "Coconut aminos are a great soy-free alternative to soy sauce and tamari. They have a similar umami flavor. I'll update the shopping list.",
        ),
        _turn(
            88,
            "user",
            "Great, let's do that. By the way, I just remembered something else about Bob - he told me ages ago that he's also allergic to shellfish. So no shrimp or crab.",
        ),
        _turn(
            89,
            "assistant",
            "Added to Bob's list. So Bob has: peanut allergy, shellfish allergy, lactose intolerance, and keto preference. Since we're doing salmon, we're fine - salmon is a fish, not a shellfish.",
        ),
        _turn(
            90,
            "user",
            "Oh wait actually, I just got a message from Bob. He says he saw an allergist recently and they retested him. Turns out he OUTGREW the shellfish allergy! The doctor confirmed he's clear. So scratch that one.",
        ),
        _turn(
            91,
            "assistant",
            "Oh that's great news for Bob! So updated: Bob has peanut allergy, lactose intolerance (aged cheese/butter OK), and keto preference. The shellfish allergy is officially gone per his allergist.",
        ),
        _turn(
            92,
            "user",
            "Yep. But the peanut one is still very much there. Ok I think we're finally set on the dietary stuff. Let me go shopping.",
        ),
        _turn(
            93,
            "assistant",
            "Sounds good! Here's the final restriction summary: Bob - peanut allergy, lactose intolerant (aged cheese/butter OK), keto pref. Emma (Bob's daughter) - tree nut allergy. Sarah - vegan + soy allergy. Priya & Dev Patel - no beef. Linda - no mushrooms. Rachel - celiac (strict gluten-free). James - no restrictions. Have a good shopping trip!",
        ),
        _turn(
            94,
            "user",
            "Oh one more thing I keep forgetting - what's a good side dish I could make ahead of time? I don't want to be cooking everything last minute.",
        ),
        _turn(
            95,
            "assistant",
            "The roasted vegetables are great make-ahead - you can roast them Friday night and reheat. The black bean stew actually gets better overnight too. The salad dressing can be made in advance, just toss the greens right before serving. Even the bruschetta topping can be prepped the morning of.",
        ),
        _turn(
            96,
            "user",
            "Perfect plan. Ok I'm really heading to the store now. Talk later!",
        ),
        _turn(
            97,
            "assistant",
            "Happy shopping! Let me know if you have any questions at the store about ingredient labels.",
        ),
    ]
    return turns


def _build_work_conversation() -> list[dict]:
    """Work / project management: team, deadlines, brand guidelines, tools."""
    turns = [
        _turn(
            0,
            "user",
            "Hey, I need help organizing the Acme Corp project. We just got the contract signed last week.",
        ),
        _turn(
            1,
            "assistant",
            "Congratulations on the Acme Corp contract! What does the project involve?",
        ),
        _turn(
            2,
            "user",
            "It's a complete rebrand. New logo, website, marketing materials, the whole thing. Our team is handling design and development.",
        ),
        _turn(3, "assistant", "That's a big project! Who's on the team?"),
        _turn(
            4,
            "user",
            "So the core team is me (project lead), Vanessa on design, Tom on frontend dev, Maria on backend, and Hiroshi on content/copy.",
        ),
        _turn(5, "assistant", "Good team. What's the timeline looking like?"),
        _turn(
            6,
            "user",
            "Deadline is March 15th. Non-negotiable, they have a product launch on March 20th.",
        ),
        _turn(
            7, "assistant", "That's a tight turnaround. When did the contract start?"
        ),
        _turn(8, "user", "We officially kicked off January 5th. So about 10 weeks."),
        _turn(
            9, "assistant", "Let me help you plan milestones. What's been done so far?"
        ),
        _turn(
            10,
            "user",
            "Vanessa has done the initial mood boards and we had a first review with the client. They liked the modern minimalist direction.",
        ),
        _turn(
            11,
            "assistant",
            "That's good progress on the design direction. What about brand guidelines?",
        ),
        _turn(
            12,
            "user",
            "Oh yeah, so Acme sent over their existing brand book. Their primary color is Pantone 2945 C - it's a deep blue. They want to keep that as their primary but are open to updating the secondary palette.",
        ),
        _turn(
            13,
            "assistant",
            "Keeping the deep blue is smart for brand continuity. What are their current secondary colors?",
        ),
        _turn(
            14,
            "user",
            "Current secondaries are a warm gray and a burnt orange. But honestly they look dated. Vanessa is proposing we swap them for a cool gray and a teal accent.",
        ),
        _turn(
            15,
            "assistant",
            "That sounds like a more modern combination. Has the client seen the proposed palette?",
        ),
        _turn(
            16,
            "user",
            "Not yet, that's in the next review meeting. Which is... let me check... next Wednesday at 2pm. Can you remind me?",
        ),
        _turn(
            17,
            "assistant",
            "I'll make note of it - client review meeting next Wednesday at 2pm for the color palette presentation.",
        ),
        _turn(
            18,
            "user",
            "Thanks. Oh, one thing I need to figure out - Tom said something about the website. What CMS are they currently using?",
        ),
        _turn(
            19,
            "assistant",
            "You haven't mentioned their current CMS to me. Did they share that in the brand book or in a separate technical document?",
        ),
        _turn(
            20,
            "user",
            "Right, let me check. Actually Tom talked to their IT team and they're on WordPress with a custom theme. We're proposing to migrate to a headless CMS - probably Strapi with a Next.js frontend.",
        ),
        _turn(
            21,
            "assistant",
            "Strapi + Next.js is a solid stack for a modern website. Is the client on board with the migration?",
        ),
        _turn(
            22,
            "user",
            "They're nervous about it honestly. Their marketing team is used to WordPress. We need to make the admin panel really intuitive. That's one of Maria's priorities.",
        ),
        _turn(
            23,
            "assistant",
            "Makes sense. Strapi's admin panel is pretty user-friendly out of the box, and Maria can customize it. Maybe a training session for their marketing team would help ease the transition.",
        ),
        _turn(
            24,
            "user",
            "Good idea, I'll add that to the deliverables. Can you help me with something unrelated? I need to expense my lunch from yesterday.",
        ),
        _turn(
            25,
            "assistant",
            "Sure, but I don't have access to your expense system. I can help you draft the expense report though. Where did you eat and how much was it?",
        ),
        _turn(
            26,
            "user",
            "It was Cafe Milano, $34.50 with a client. Actually it was the Acme folks, we were reviewing the initial concepts over lunch.",
        ),
        _turn(
            27,
            "assistant",
            "That should be billable to the Acme project then. $34.50 at Cafe Milano, client lunch. What date was it?",
        ),
        _turn(
            28,
            "user",
            "Yesterday, January 15th. Ok back to the project. I need to brief Hiroshi on the content strategy. Acme wants a complete tone refresh.",
        ),
        _turn(29, "assistant", "What's their current brand voice like?"),
        _turn(
            30,
            "user",
            "Very corporate and stiff. Like 'Acme Corporation delivers enterprise-grade solutions.' They want to sound more approachable and human. Their new CEO is really pushing for a more startup-y feel.",
        ),
        _turn(
            31,
            "assistant",
            "That's a significant shift. Does Hiroshi have experience with that kind of tone transition?",
        ),
        _turn(
            32,
            "user",
            "Yeah, he did the same thing for the Nexus rebrand last year. That went really well. His style guide for them was excellent.",
        ),
        _turn(
            33,
            "assistant",
            "Great, so he has a track record. Maybe he can use the Nexus project as a template. What about the logo - is that being redesigned too?",
        ),
        _turn(
            34,
            "user",
            "Yes. Vanessa has three concepts. One is an evolution of the current logo, one is completely new, and one is somewhere in between. The client hasn't seen them yet.",
        ),
        _turn(
            35, "assistant", "Smart to give them options. When will those be presented?"
        ),
        _turn(
            36,
            "user",
            "Same meeting next Wednesday. We're presenting colors and logo together.",
        ),
        _turn(
            37,
            "assistant",
            "Big meeting then. Color palette and logo concepts both. Should Vanessa present, or will you?",
        ),
        _turn(
            38,
            "user",
            "Vanessa will present the designs and I'll handle the Q&A and timeline discussion.",
        ),
        _turn(
            39,
            "assistant",
            "Good division of labor. Anything else for the Wednesday meeting?",
        ),
        _turn(
            40,
            "user",
            "Actually yes - the client asked about social media templates. They want a cohesive look across Instagram, LinkedIn, and Twitter. I told them we'd include that in the deliverables.",
        ),
        _turn(
            41,
            "assistant",
            "Social media templates are a smart addition. That'll help them maintain consistency after the rebrand launches. Who'll create those?",
        ),
        _turn(
            42,
            "user",
            "Vanessa, but not until after the core brand identity is locked in. So probably mid-February.",
        ),
        _turn(
            43,
            "assistant",
            "Makes sense - you need the foundation before the templates. So the social templates are a Phase 2 deliverable, starting mid-Feb.",
        ),
        _turn(
            44,
            "user",
            "Exactly. Hey, random question - do you know if there's a way to batch export from Figma to multiple formats at once? Vanessa was asking.",
        ),
        _turn(
            45,
            "assistant",
            "Yes! In Figma, you can select multiple frames, then in the Export section of the right panel, you can add multiple export formats (PNG, SVG, PDF) with different scales. When you hit 'Export Selected', it exports all combinations. There are also plugins like 'Batch Export' that give more control.",
        ),
        _turn(
            46,
            "user",
            "Perfect, I'll let Vanessa know. She's been doing them one by one, poor thing.",
        ),
        _turn(
            47,
            "assistant",
            "That'll save her a lot of time! Anything else for the project?",
        ),
        _turn(
            48,
            "user",
            "Yeah, one more thing. Acme's legal team sent over some requirements for the website. They need WCAG 2.1 AA compliance. Tom needs to know about this.",
        ),
        _turn(
            49,
            "assistant",
            "WCAG 2.1 AA is pretty standard for corporate sites. That means proper color contrast ratios (at least 4.5:1 for normal text), keyboard navigation, alt text for images, semantic HTML, and ARIA labels where needed. Tom should factor that in from the start rather than retrofitting.",
        ),
        _turn(
            50,
            "user",
            "Agreed. I'll brief him today. Oh, and Maria just pinged me - she wants to know if we should set up a staging environment now or wait until the design is locked.",
        ),
        _turn(
            51,
            "assistant",
            "I'd say set it up now. Even if the design isn't final, having the staging environment ready means Tom and Maria can start on the infrastructure, deployment pipeline, and basic CMS configuration. That way when designs are approved, they can move fast.",
        ),
        _turn(
            52,
            "user",
            "Smart. I'll tell Maria to go ahead with staging. She usually uses Vercel for Next.js projects.",
        ),
        _turn(
            53,
            "assistant",
            "Vercel is a natural fit for Next.js. She can set up preview deployments too, which will be great for client reviews - each PR gets its own preview URL.",
        ),
        _turn(
            54,
            "user",
            "Oh that's a great feature. The client will love being able to see changes in real time.",
        ),
        _turn(55, "assistant", "Definitely. Anything else on your plate today?"),
        _turn(
            56,
            "user",
            "I need to update the project timeline in Notion. We use Notion for all project management.",
        ),
        _turn(
            57,
            "assistant",
            "Want me to help you structure the timeline? I can lay out the phases and milestones based on what we've discussed.",
        ),
        _turn(
            58,
            "user",
            "Yeah let's do that. But wait, first - I got an email from the Acme CMO, Patricia. She wants to add a video component to the website. Like an animated hero section. This is scope creep but I'm not sure I can push back.",
        ),
        _turn(
            59,
            "assistant",
            "Classic scope creep. An animated hero section can range from simple CSS animations to full motion graphics. How ambitious is Patricia's vision?",
        ),
        _turn(
            60,
            "user",
            "She sent a reference - it's the Apple product page style where the product animates as you scroll. That's... not trivial.",
        ),
        _turn(
            61,
            "assistant",
            "That kind of scroll-triggered animation is definitely a significant add. It involves motion design (not just static design), and complex frontend work with libraries like GSAP or Framer Motion. You might need to either push the timeline or add budget.",
        ),
        _turn(
            62,
            "user",
            "I'll talk to her about it. Maybe we can do a simpler version. Ok, let me actually get that timeline done now.",
        ),
        _turn(
            63,
            "assistant",
            "Here's a rough timeline based on everything: Phase 1 (Jan 5 - Jan 31): Brand identity - logo, colors, typography, brand voice guide. Phase 2 (Feb 1 - Feb 21): Website design and development, social media templates. Phase 3 (Feb 22 - Mar 8): Content migration, testing, WCAG compliance audit. Phase 4 (Mar 9 - Mar 15): Client review, final revisions, launch prep. That's tight but doable if no major scope additions.",
        ),
        _turn(
            64,
            "user",
            "Yeah, that animation thing could blow Phase 2. I need to resolve that quickly. Ok I'm going to update Notion now. Thanks!",
        ),
        _turn(
            65,
            "assistant",
            "Good luck! Let me know how the conversation with Patricia goes about the animation scope.",
        ),
        _turn(
            66,
            "user",
            "Hey, I'm back. Had a good chat with Patricia. She agreed to a simplified animation - a subtle parallax effect rather than the Apple-style scroll animation. Tom says that's very doable.",
        ),
        _turn(
            67,
            "assistant",
            "That's a great compromise! Parallax is much more manageable and still looks polished. Tom should be able to handle that without major timeline impact.",
        ),
        _turn(
            68,
            "user",
            "Yeah exactly. Oh, one more brand guideline thing I forgot to mention - Acme's font. They use a custom typeface called Acme Sans that was designed for them. We have the license to use it on the website. The fallback is Inter.",
        ),
        _turn(
            69,
            "assistant",
            "Good to know - Acme Sans as the primary typeface with Inter as fallback. Tom should set up the @font-face declarations early and make sure the custom font loads properly with appropriate font-display settings to avoid FOIT.",
        ),
        _turn(70, "user", "What's FOIT?"),
        _turn(
            71,
            "assistant",
            "Flash of Invisible Text - it's when the browser hides text until the custom font loads. Using font-display: swap makes the browser show the fallback font immediately, then swaps to the custom font once loaded. Much better user experience.",
        ),
        _turn(
            72,
            "user",
            "Good to know. Ok, I think I have a handle on the project now. Let me go brief the team.",
        ),
        _turn(
            73,
            "assistant",
            "Sounds like a plan! Don't forget the Wednesday meeting with the client. Want me to help you prep a presentation deck?",
        ),
        _turn(
            74,
            "user",
            "Actually yes, but I'll tackle that Monday. Thanks for all the help!",
        ),
        _turn(75, "assistant", "Happy to help! Have a great rest of the day."),
        _turn(
            76,
            "user",
            "Quick question before I go - what was the name of the Acme CEO again? The new one who wants the startup-y feel?",
        ),
        _turn(
            77,
            "assistant",
            "You didn't actually mention the CEO's name - just that the new CEO is pushing for a more approachable, startup-y brand voice.",
        ),
        _turn(
            78,
            "user",
            "Oh right, it's Daniel Park. He came from a tech startup background. Good to note.",
        ),
        _turn(
            79,
            "assistant",
            "Daniel Park, new CEO from tech startup background, driving the brand refresh toward approachable/human tone. Got it!",
        ),
        _turn(80, "user", "Thanks! Talk to you Monday."),
        _turn(81, "assistant", "See you Monday! Good luck with the team briefing."),
        _turn(
            82,
            "user",
            "Hey it's Monday. I need to prep for the Wednesday meeting. Can you help me put together a presentation outline?",
        ),
        _turn(
            83,
            "assistant",
            "Of course! Based on what we've discussed, here's what the meeting should cover: 1) Brand color palette (primary Pantone 2945 C blue, proposed new secondaries: cool gray + teal). 2) Logo concepts (three options from Vanessa). 3) Typography (Acme Sans with Inter fallback). 4) Website direction (Strapi + Next.js, simplified parallax hero). 5) Timeline review. 6) Social media template preview/timeline. Sound right?",
        ),
        _turn(
            84,
            "user",
            "That's perfect. Oh, but we should also include the brand voice section. Hiroshi finished the initial tone guide over the weekend.",
        ),
        _turn(
            85,
            "assistant",
            "Great, let's add that after typography then: Brand Voice Guide overview. What direction did Hiroshi settle on?",
        ),
        _turn(
            86,
            "user",
            "He went with 'confident but conversational.' Think Slack or Stripe's marketing copy. Professional but not stuffy.",
        ),
        _turn(
            87,
            "assistant",
            "'Confident but conversational' - that fits perfectly with Daniel Park's vision. Slack and Stripe are excellent reference points for that tone. Hiroshi clearly understood the brief.",
        ),
        _turn(
            88,
            "user",
            "Yeah he nailed it. Ok I'll work on the deck using these sections. One more thing - Vanessa just told me she wants to add a motion guidelines section to the brand book. Like how things should animate on the website.",
        ),
        _turn(
            89,
            "assistant",
            "Motion guidelines are a smart addition to the brand book - they ensure consistency in how elements move across the website and future applications. Things like easing curves, duration standards, entrance/exit patterns. It'll help Tom implement animations consistently too.",
        ),
        _turn(
            90,
            "user",
            "Exactly what Vanessa said. Ok I think we're good. I'll work on this and ping you if I need anything.",
        ),
        _turn(
            91,
            "assistant",
            "Sounds great! Good luck with the deck. Let me know if you want me to review anything before Wednesday.",
        ),
    ]
    return turns


def _build_medical_conversation() -> list[dict]:
    """Medical / health: medications, conditions, appointments, interactions."""
    turns = [
        _turn(
            0,
            "user",
            "Hi, I need to organize my health information. It's getting hard to keep track of everything.",
        ),
        _turn(
            1,
            "assistant",
            "I'd be happy to help you organize your health info. Where would you like to start?",
        ),
        _turn(
            2,
            "user",
            "Let me start with my medications. I take metformin 500mg twice daily for type 2 diabetes.",
        ),
        _turn(
            3,
            "assistant",
            "Got it - metformin 500mg, twice daily, for type 2 diabetes. What else?",
        ),
        _turn(
            4,
            "user",
            "I also take lisinopril 10mg once daily for blood pressure. My doctor Dr. Patel started me on that about 6 months ago.",
        ),
        _turn(
            5,
            "assistant",
            "Lisinopril 10mg daily for blood pressure, prescribed by Dr. Patel about 6 months ago. Anything else?",
        ),
        _turn(
            6,
            "user",
            "And atorvastatin 20mg at night for cholesterol. That one I've been on for years.",
        ),
        _turn(
            7,
            "assistant",
            "So your current medications are: 1) Metformin 500mg twice daily (diabetes), 2) Lisinopril 10mg daily (blood pressure), 3) Atorvastatin 20mg nightly (cholesterol). Anything else?",
        ),
        _turn(
            8,
            "user",
            "Oh yeah, I take a daily vitamin D supplement because my levels were low. 2000 IU.",
        ),
        _turn(
            9,
            "assistant",
            "Added - Vitamin D 2000 IU daily. Good that you're supplementing. How are your levels doing now?",
        ),
        _turn(
            10,
            "user",
            "Last check they were at 38, up from 22. Dr. Patel wants me above 40 so we're getting close.",
        ),
        _turn(
            11,
            "assistant",
            "Good progress! From 22 to 38. Almost at the target of 40. When's your next blood test?",
        ),
        _turn(
            12,
            "user",
            "February 10th. It's my quarterly check - they do A1C, lipids, vitamin D, the whole panel.",
        ),
        _turn(
            13,
            "assistant",
            "I'll note that - comprehensive blood panel on February 10th. How has your A1C been trending?",
        ),
        _turn(
            14,
            "user",
            "Last one was 6.8. Down from 7.2 six months ago. Dr. Patel was really pleased.",
        ),
        _turn(
            15,
            "assistant",
            "That's excellent progress! 7.2 down to 6.8 is significant. The metformin and any lifestyle changes are clearly working.",
        ),
        _turn(
            16,
            "user",
            "Yeah, I've been walking 30 minutes every day and cut back on sugar. Makes a big difference.",
        ),
        _turn(
            17,
            "assistant",
            "Those lifestyle changes are huge. Walking daily and reducing sugar intake complement the metformin really well.",
        ),
        _turn(
            18,
            "user",
            "Thanks. Ok, I also wanted to talk about my knee. I've been having pain in my right knee for about 3 weeks.",
        ),
        _turn(
            19,
            "assistant",
            "Sorry to hear that. What kind of pain - sharp, dull ache, swelling? And did something trigger it?",
        ),
        _turn(
            20,
            "user",
            "Dull ache, sometimes sharp when I go up stairs. No specific injury, it just started gradually. I'm 54, so it's probably age-related.",
        ),
        _turn(
            21,
            "assistant",
            "Gradual onset knee pain at 54 could be a few things - mild osteoarthritis is common. Have you mentioned it to your doctor?",
        ),
        _turn(
            22,
            "user",
            "Not yet, I have an appointment with Dr. Patel on January 25th. I'll bring it up then. For now I've been taking ibuprofen when it's bad.",
        ),
        _turn(
            23,
            "assistant",
            "That's a good plan to discuss it with Dr. Patel. How much ibuprofen are you taking, and how often?",
        ),
        _turn(
            24,
            "user",
            "Usually 400mg, maybe 2-3 times a week. Only when it really bothers me.",
        ),
        _turn(
            25,
            "assistant",
            "That's a moderate amount. One thing to be aware of - NSAIDs like ibuprofen can interact with your lisinopril (blood pressure medication). They can reduce its effectiveness and also affect kidney function, which is especially important to monitor with diabetes. Definitely mention the ibuprofen use to Dr. Patel.",
        ),
        _turn(
            26,
            "user",
            "Oh I didn't know that! That's concerning. Should I stop taking it?",
        ),
        _turn(
            27,
            "assistant",
            "I'm not a doctor, but occasional use at that dose is generally considered low risk. However, given your specific combination of diabetes + blood pressure medication, it's worth discussing alternatives with Dr. Patel. Acetaminophen (Tylenol) doesn't have the same interaction with lisinopril, though it's less effective for inflammation.",
        ),
        _turn(
            28,
            "user",
            "Good to know. I'll ask Dr. Patel about it. Maybe he can recommend something better for the knee.",
        ),
        _turn(
            29,
            "assistant",
            "Physical therapy is often very effective for knee pain too, and has no drug interactions!",
        ),
        _turn(
            30,
            "user",
            "Ha, good point. My wife's been telling me to try physical therapy. She goes for her shoulder.",
        ),
        _turn(
            31,
            "assistant",
            "Your wife might be onto something. Who does she see for her shoulder?",
        ),
        _turn(
            32,
            "user",
            "Some place called ProMotion Physical Therapy on Oak Street. She really likes them.",
        ),
        _turn(
            33,
            "assistant",
            "Could be worth trying the same place. Personal recommendations are usually the best way to find a good PT.",
        ),
        _turn(
            34,
            "user",
            "Yeah I might. Ok, switching topics - I also wanted to mention that I've been having trouble sleeping lately. Waking up around 3am and can't get back to sleep.",
        ),
        _turn(
            35,
            "assistant",
            "Middle-of-the-night awakening is frustrating. How long has this been going on?",
        ),
        _turn(
            36,
            "user",
            "About 2 weeks. I think it might be stress-related - work has been intense lately.",
        ),
        _turn(
            37,
            "assistant",
            "Stress is one of the most common causes of sleep disruption. Have you tried anything to help?",
        ),
        _turn(
            38,
            "user",
            "I've been drinking chamomile tea before bed. Not sure if it helps. My friend recommended melatonin but I haven't tried it.",
        ),
        _turn(
            39,
            "assistant",
            "Chamomile is mild and generally safe. Melatonin can help with sleep onset but isn't always great for middle-of-the-night waking. Good sleep hygiene basics like consistent bed/wake times, no screens before bed, and a cool room temperature are often more effective. Worth mentioning to Dr. Patel at your January 25th appointment too.",
        ),
        _turn(
            40,
            "user",
            "Yeah I'll add it to the list for that appointment. It's turning into a big appointment - knee, sleep, and the regular stuff.",
        ),
        _turn(
            41,
            "assistant",
            "It might be worth writing down your topics so you don't forget any. Doctors appreciate when patients come prepared with a list.",
        ),
        _turn(
            42,
            "user",
            "Good idea. Ok let me also tell you about my family health history since we're on this topic.",
        ),
        _turn(43, "assistant", "Family history is very useful context. Go ahead."),
        _turn(
            44,
            "user",
            "My father had a heart attack at 62. He survived but had bypass surgery. My mother has type 2 diabetes like me. My brother was diagnosed with high cholesterol at 45.",
        ),
        _turn(
            45,
            "assistant",
            "There's a clear pattern of cardiovascular risk factors in your family - heart disease, diabetes, and high cholesterol. This makes your current medications (metformin, lisinopril, atorvastatin) all the more important for prevention. Has Dr. Patel discussed any additional cardiac screening given the family history?",
        ),
        _turn(
            46,
            "user",
            "He mentioned I should get a cardiac stress test. I've been putting it off honestly. I'm a bit scared.",
        ),
        _turn(
            47,
            "assistant",
            "That's completely understandable. But a stress test is preventive - it's much better to catch any issues early. Given your dad's history at 62, getting a baseline at 54 is actually very smart timing.",
        ),
        _turn(
            48,
            "user",
            "You're right. I'll schedule it. My wife keeps telling me the same thing.",
        ),
        _turn(
            49,
            "assistant",
            "Sounds like your wife gives good health advice! Between the PT recommendation and the stress test nudge, she's on top of it.",
        ),
        _turn(
            50,
            "user",
            "Haha she really is. She's a nurse actually, so she knows what she's talking about.",
        ),
        _turn(
            51,
            "assistant",
            "That explains a lot! Having a nurse in the family is a real advantage for health management.",
        ),
        _turn(
            52,
            "user",
            "It is, though she also worries more than most people. Anyway, let me also tell you about my eye appointment. I have diabetic retinopathy screening once a year.",
        ),
        _turn(
            53,
            "assistant",
            "Annual diabetic eye exams are important. When's the next one?",
        ),
        _turn(
            54,
            "user",
            "March 3rd with Dr. Kim, the ophthalmologist. Last year's screening was clear, no signs of retinopathy.",
        ),
        _turn(
            55,
            "assistant",
            "That's great - clear screening is very encouraging. With your A1C trending down, you're reducing the risk factors too.",
        ),
        _turn(
            56,
            "user",
            "Yeah exactly. Ok, different topic - I want to ask about supplements. My buddy at work takes fish oil and says it helps with inflammation. Would that help my knee?",
        ),
        _turn(
            57,
            "assistant",
            "Fish oil (omega-3 fatty acids) does have some anti-inflammatory properties and there's decent evidence for its benefits with joint pain. It's generally safe, but one thing to consider - fish oil can have a mild blood-thinning effect. Given that you're on atorvastatin and lisinopril, you'd want to mention it to Dr. Patel before starting. The good news is fish oil and metformin don't typically interact.",
        ),
        _turn(
            58,
            "user",
            "Ok I'll ask him. Another question - can I take Tylenol and ibuprofen at the same time? Like alternating?",
        ),
        _turn(
            59,
            "assistant",
            "They can actually be used together or alternated since they work through different mechanisms. Many doctors recommend alternating them for better pain control. But again, given your specific situation with lisinopril, the ibuprofen piece needs to be cleared by your doctor first. Tylenol on its own is fine with all your current medications.",
        ),
        _turn(
            60,
            "user",
            "Got it. Ok I think I've covered all my health stuff. Let me just run through the upcoming appointments again.",
        ),
        _turn(
            61,
            "assistant",
            "Sure! Here's what we have: 1) Dr. Patel appointment - January 25th (knee, sleep, general check-in), 2) Blood panel - February 10th (A1C, lipids, vitamin D, full panel), 3) Dr. Kim ophthalmology - March 3rd (diabetic retinopathy screening). Plus you want to schedule a cardiac stress test and potentially start physical therapy.",
        ),
        _turn(
            62,
            "user",
            "Perfect list. Oh wait, I also have a dentist appointment February 3rd. Nothing health-critical, just a cleaning.",
        ),
        _turn(
            63,
            "assistant",
            "Added - dentist cleaning on February 3rd. Actually, one thing to be aware of: with diabetes, dental health is particularly important since high blood sugar can increase risk of gum disease. So those regular cleanings are more important than they might seem.",
        ),
        _turn(
            64,
            "user",
            "Didn't know that either! You're full of health connections today.",
        ),
        _turn(
            65,
            "assistant",
            "Everything in the body is connected! Diabetes especially has wide-reaching effects. That's why your comprehensive approach to management - medications, exercise, diet, regular screenings - is so important.",
        ),
        _turn(
            66,
            "user",
            "Thanks for helping me organize all this. One last thing - my pharmacy changed. I now use the Walgreens on Main Street instead of the CVS. I switched because Walgreens is closer to my office.",
        ),
        _turn(
            67,
            "assistant",
            "Noted - pharmacy changed from CVS to Walgreens on Main Street. Make sure all your prescriptions have been transferred, including any refills that might still be on file at CVS.",
        ),
        _turn(
            68,
            "user",
            "Already done. The pharmacist at Walgreens was really helpful with the transfer.",
        ),
        _turn(
            69,
            "assistant",
            "Great, sounds like a smooth transition. Anything else you want to go over?",
        ),
        _turn(
            70, "user", "Nope, I think we covered everything. This was really helpful."
        ),
        _turn(
            71,
            "assistant",
            "Glad I could help! Remember to write up that list for Dr. Patel before the 25th. And don't put off scheduling that stress test!",
        ),
        _turn(72, "user", "I won't! Talk to you later."),
        _turn(73, "assistant", "Take care, and keep up those daily walks!"),
        _turn(
            74,
            "user",
            "Hey, quick update. I just got off the phone with Dr. Patel's office. They want to increase my metformin to 1000mg twice daily. My last fasting glucose was a bit high.",
        ),
        _turn(
            75,
            "assistant",
            "That's a significant increase - from 500mg to 1000mg twice daily. When does the new dosage start?",
        ),
        _turn(
            76,
            "user",
            "Starting tomorrow. He said to take it with food to minimize stomach upset. And I'll get retested in 6 weeks.",
        ),
        _turn(
            77,
            "assistant",
            "Good advice about taking it with food - metformin can cause GI side effects, especially at higher doses. If you experience persistent nausea or diarrhea, let Dr. Patel know. Some patients do better with the extended-release version.",
        ),
        _turn(
            78,
            "user",
            "Ok noted. Also, I forgot to mention earlier - I also take a baby aspirin daily. 81mg. Dr. Patel recommended it for heart health given my family history.",
        ),
        _turn(
            79,
            "assistant",
            "Important addition! Daily low-dose aspirin (81mg) for cardiac prevention. This is consistent with the cardiovascular risk in your family. Note that the baby aspirin is another reason to be careful with ibuprofen - both affect platelet function, and using them together increases the risk of GI bleeding.",
        ),
        _turn(
            80,
            "user",
            "Wow, that's good to know. So between the blood pressure thing and this, I really should minimize the ibuprofen.",
        ),
        _turn(
            81,
            "assistant",
            "Exactly. Between the lisinopril interaction and the aspirin overlap, there are two separate reasons to limit ibuprofen use. Definitely discuss pain alternatives with Dr. Patel at your upcoming appointment.",
        ),
        _turn(82, "user", "Will do. Thanks again."),
        _turn(
            83,
            "assistant",
            "You're welcome! Stay healthy and let me know how the appointment goes.",
        ),
        _turn(
            84,
            "user",
            "One more thing I just remembered - I was also told I have slightly elevated PSA levels. Dr. Patel said it's not alarming but wants to monitor it. I need to get retested in 6 months.",
        ),
        _turn(
            85,
            "assistant",
            "Slightly elevated PSA at 54 should indeed be monitored but isn't necessarily cause for alarm - it can be elevated for many reasons beyond prostate cancer, including benign prostate enlargement or even recent exercise. When was the last PSA test?",
        ),
        _turn(
            86,
            "user",
            "About 3 months ago. Level was 4.5, normal range is under 4. So just slightly above.",
        ),
        _turn(
            87,
            "assistant",
            "4.5 is borderline - just slightly above the 4.0 threshold. Monitoring is the right approach. You'll want that retest in about 3 more months (so roughly April). I'll add it to the timeline.",
        ),
        _turn(
            88,
            "user",
            "Yeah, April retest. Ok NOW I'm really done with health stuff. Thanks!",
        ),
        _turn(
            89,
            "assistant",
            "Take care! Updated medication list: Metformin 1000mg 2x daily, Lisinopril 10mg daily, Atorvastatin 20mg nightly, Baby aspirin 81mg daily, Vitamin D 2000 IU daily. Appointments: Jan 25 (Dr. Patel), Feb 3 (dentist), Feb 10 (blood panel), Mar 3 (Dr. Kim eye exam), April (PSA retest), plus stress test and PT to schedule.",
        ),
    ]
    return turns


def _build_planning_conversation() -> list[dict]:
    """Event planning with multiple constraints, checklists, vendors."""
    turns = [
        _turn(
            0,
            "user",
            "Hi! I need help planning my parents' 40th wedding anniversary party. It's going to be a big deal.",
        ),
        _turn(
            1,
            "assistant",
            "How exciting! A 40th anniversary is a wonderful milestone. When is the party?",
        ),
        _turn(
            2,
            "user",
            "June 15th. We want it to be a surprise, so everything has to be coordinated without them knowing.",
        ),
        _turn(
            3,
            "assistant",
            "A surprise party adds a layer of complexity! How many guests are you expecting?",
        ),
        _turn(
            4,
            "user",
            "About 60 people. Family and close friends. My sister Karen and I are organizing it together.",
        ),
        _turn(
            5,
            "assistant",
            "Great that you have a co-organizer! What's the budget looking like?",
        ),
        _turn(
            6,
            "user",
            "We're thinking around $5,000 total. Karen and I are splitting it, and a few aunts and uncles offered to chip in.",
        ),
        _turn(
            7,
            "assistant",
            "$5,000 for 60 people is about $83 per person, which is doable for a nice event. Have you thought about a venue?",
        ),
        _turn(
            8,
            "user",
            "We're looking at the Riverside Pavilion. It's an outdoor covered space by the river. Beautiful in June. They quoted us $800 for a Saturday evening.",
        ),
        _turn(
            9,
            "assistant",
            "Riverside Pavilion sounds lovely for a June evening event. $800 for the venue is reasonable. What does that include?",
        ),
        _turn(
            10,
            "user",
            "The space, tables and chairs for 60, basic lighting, and access to an outdoor lawn area. We'd need to bring our own catering, decorations, and any extra rentals.",
        ),
        _turn(
            11,
            "assistant",
            "Good starting point. So you have $4,200 remaining for catering, decorations, entertainment, and everything else. Should we break that down?",
        ),
        _turn(
            12,
            "user",
            "Yes please. But first - the catering. My aunt Rosa offered to cook. She's amazing - she used to own a restaurant in Mexico City. She wants to do a Mexican-themed buffet.",
        ),
        _turn(
            13,
            "assistant",
            "That's an incredible offer! Homemade Mexican food from a former restaurateur will be both delicious and budget-friendly. What does she need from you?",
        ),
        _turn(
            14,
            "user",
            "She needs us to cover ingredients, which she estimates at about $600 for 60 people. She'll handle all the cooking with help from my cousins.",
        ),
        _turn(
            15,
            "assistant",
            "$600 for catering 60 people is fantastic. That leaves $3,600 for everything else. What did you have in mind for the menu?",
        ),
        _turn(
            16,
            "user",
            "Aunt Rosa is planning: carnitas, chicken mole, rice, beans, a big salad, guacamole, and her famous tres leches cake for dessert.",
        ),
        _turn(
            17,
            "assistant",
            "That sounds like a wonderful spread! The tres leches cake is a great alternative to a traditional anniversary cake. Now, are there any dietary restrictions among the 60 guests?",
        ),
        _turn(
            18,
            "user",
            "Good question. My cousin David is vegetarian. Uncle Frank is diabetic. And Karen's husband Mike has a severe shellfish allergy, though I don't think there's shellfish in any of those dishes.",
        ),
        _turn(
            19,
            "assistant",
            "You're right - the planned menu is naturally shellfish-free. For David (vegetarian), the rice, beans, salad, and guacamole are solid options, but Aunt Rosa might want to add a vegetarian main like cheese enchiladas or stuffed peppers. For Uncle Frank (diabetic), the tres leches might be a concern - maybe a small sugar-free option on the side?",
        ),
        _turn(
            20,
            "user",
            "Great ideas. I'll talk to Aunt Rosa. She'll probably do cheese enchiladas for David, she makes great ones. And yeah, maybe a small fruit plate option for Uncle Frank's dessert.",
        ),
        _turn(
            21,
            "assistant",
            "Perfect. Now let's talk about decorations. What's the vibe?",
        ),
        _turn(
            22,
            "user",
            "My parents got married in 1985 in a small church ceremony. We want to recreate some of that feeling - romantic, warm, maybe some vintage touches. Their wedding colors were burgundy and gold.",
        ),
        _turn(
            23,
            "assistant",
            "Burgundy and gold are beautiful together, and very fitting for a 40th (ruby) anniversary! I'd suggest: burgundy table runners, gold charger plates or accents, tea light candles, maybe some old photos of them from through the years as table centerpieces.",
        ),
        _turn(
            24,
            "user",
            "Love the photo idea! Karen actually has a whole box of old photos from Mom and Dad's wedding and through the years. We could make a photo timeline display.",
        ),
        _turn(
            25,
            "assistant",
            "A photo timeline would be a wonderful centerpiece! You could string photos along a clothesline with mini clothespins, or create a display board by decade. How much are you budgeting for decorations?",
        ),
        _turn(
            26,
            "user",
            "Maybe $500 for decorations? We can DIY a lot of it. My friend Jenny is really crafty, she offered to help.",
        ),
        _turn(
            27,
            "assistant",
            "$500 for DIY decorations is generous. Between you, Karen, and Jenny, you can do a lot. What about music and entertainment?",
        ),
        _turn(
            28,
            "user",
            "So here's a fun thing - my parents' first dance at their wedding was to 'Can't Help Falling in Love' by Elvis. We want to play that at the party and have them dance to it again.",
        ),
        _turn(
            29,
            "assistant",
            "That's so sweet! Are you thinking a DJ, a live musician, or just a curated playlist?",
        ),
        _turn(
            30,
            "user",
            "Karen found a DJ who specializes in events like this. He charges $600 for 4 hours. His name is DJ Marco and he has great reviews.",
        ),
        _turn(
            31,
            "assistant",
            "DJ Marco at $600 for 4 hours sounds reasonable. He can handle the music flow and do the announcement for the first dance recreation. Running budget: venue $800, food $600, decorations $500, DJ $600 = $2,500 so far. You have $2,500 remaining.",
        ),
        _turn(
            32,
            "user",
            "Oh wait, I need to factor in drinks too. We'll need beer, wine, and non-alcoholic options. My dad loves Mexican beer and my mom is a red wine person.",
        ),
        _turn(
            33,
            "assistant",
            "For 60 people, a basic bar with Mexican beer (Modelo, Dos Equis), a couple of red wines, and non-alcoholic drinks (agua fresca would fit the Mexican theme!) might run $400-600 depending on consumption.",
        ),
        _turn(
            34,
            "user",
            "Let's budget $500 for drinks. Agua fresca is a brilliant idea, Aunt Rosa can make that too. She does an amazing horchata.",
        ),
        _turn(
            35,
            "assistant",
            "Horchata from Aunt Rosa - yes! So drinks budget $500. That puts us at $3,000, leaving $2,000 for remaining items. What else is on the list?",
        ),
        _turn(
            36,
            "user",
            "Hmm, let me think. We need: invitations, a photographer, maybe some kind of party favors, and a videographer would be nice.",
        ),
        _turn(
            37,
            "assistant",
            "For invitations, since it's a surprise party, you could do digital invitations (Paperless Post is free or cheap) to keep costs down and avoid paper mail that parents might see.",
        ),
        _turn(
            38,
            "user",
            "Digital invitations make total sense. We definitely don't want Mom finding a stray envelope. Let's keep that free with Paperless Post.",
        ),
        _turn(
            39,
            "assistant",
            "Smart. For a photographer, you're looking at $500-1500 for event coverage. Any budget preference?",
        ),
        _turn(
            40,
            "user",
            "My nephew Jason is in photography school. He offered to do it for $200 and said he'd treat it as a portfolio piece. His stuff is actually pretty good.",
        ),
        _turn(
            41,
            "assistant",
            "That's a great deal! $200 for a motivated photography student who has a personal connection to the family. He'll probably capture more genuine moments because he knows everyone.",
        ),
        _turn(
            42,
            "user",
            "Exactly. And Karen knows a videographer who charges $400 for a 2-hour highlight reel.",
        ),
        _turn(
            43,
            "assistant",
            "Nice. So $200 photographer + $400 videographer = $600 for capturing the event. Running total: $3,600. That leaves $1,400 for party favors and any extras.",
        ),
        _turn(
            44,
            "user",
            "For party favors, I was thinking small bottles of hot sauce with a custom label. Aunt Rosa makes her own hot sauce and it's incredible. We could bottle like 70 small bottles.",
        ),
        _turn(
            45,
            "assistant",
            "Homemade hot sauce with custom labels is such a personal and unique favor! The bottles and labels might cost around $100-150 for 70. That's a great use of budget.",
        ),
        _turn(
            46,
            "user",
            "Yeah I think $150 covers it. Ok so let me think about what else... Oh! We need to figure out how to actually GET my parents there without spoiling the surprise.",
        ),
        _turn(
            47,
            "assistant",
            "The logistics of the surprise! This is crucial. What are you thinking?",
        ),
        _turn(
            48,
            "user",
            "Karen's plan is to tell Mom and Dad that she's having a small family dinner at the pavilion for Karen's birthday (which is actually June 12th, close enough to be believable). So they'll dress up but think it's a small gathering.",
        ),
        _turn(
            49,
            "assistant",
            "Using Karen's actual birthday as the cover story is clever - it's believable and close to the real date. They'll already be dressed nicely for a birthday dinner. Who will the point person be to coordinate the 'they're arriving' signal?",
        ),
        _turn(
            50,
            "user",
            "I'll be at the venue. Karen will drive them. She'll text me when they're 10 minutes away so everyone can get in position.",
        ),
        _turn(
            51,
            "assistant",
            "Good plan. Make sure DJ Marco knows the signal too - he'll need to cue the right music or announcement. Now, is there a Plan B if something goes wrong? Like if your parents want to change the date?",
        ),
        _turn(
            52,
            "user",
            "Hmm, good point. I guess Karen could say the pavilion is already reserved for her birthday and can't change. That should work.",
        ),
        _turn(
            53,
            "assistant",
            "That's a reasonable fallback. OK so where are we on the checklist?",
        ),
        _turn(
            54,
            "user",
            "Let me think... Actually, can we talk about the seating arrangement? With 60 people there are some dynamics to manage.",
        ),
        _turn(55, "assistant", "Absolutely. Any family dynamics or considerations?"),
        _turn(
            56,
            "user",
            "Well, my Uncle Steve and Aunt Patricia aren't speaking to each other since last Thanksgiving. Long story. They need to be at opposite ends of the venue.",
        ),
        _turn(
            57,
            "assistant",
            "Noted - keep Uncle Steve and Aunt Patricia well separated. Family events with feuds require strategic seating. Anyone else?",
        ),
        _turn(
            58,
            "user",
            "Grandma Helen is in a wheelchair so she needs to be somewhere accessible, near the entrance and not on grass. And she can't hear well, so put her near the speakers but not too close.",
        ),
        _turn(
            59,
            "assistant",
            "Grandma Helen: wheelchair accessible spot near the entrance, on the paved area, moderate distance from speakers. Got it. Any other considerations?",
        ),
        _turn(
            60,
            "user",
            "My cousin Maria's kids are 3 and 5. We might need a little kids' area or at least a table with coloring books or something to keep them occupied.",
        ),
        _turn(
            61,
            "assistant",
            "Great idea. A small kids' corner with coloring books, crayons, and maybe some simple games would be easy and cheap to set up. Maybe $30 worth of supplies.",
        ),
        _turn(
            62,
            "user",
            "Perfect. Oh and I just remembered - we should do a speech or toast. Karen and I will both say something, and we want to invite anyone else who wants to speak.",
        ),
        _turn(
            63,
            "assistant",
            "An open mic for toasts is lovely. You and Karen can go first to set the tone, then open it up. DJ Marco can facilitate that. Do you want to coordinate with anyone in advance so there's no awkward silence?",
        ),
        _turn(
            64,
            "user",
            "Good idea. I'll ask Uncle Roberto - he's the family storyteller. He always has great anecdotes. And maybe our family friend Gloria, she's been Mom's best friend for 30 years.",
        ),
        _turn(
            65,
            "assistant",
            "Uncle Roberto and Gloria as planned speakers, with open mic for others. That should give you a nice mix of stories and sentiments.",
        ),
        _turn(
            66,
            "user",
            "Hey, quick detour - I need help with something at work. Can you look at an email I'm drafting?",
        ),
        _turn(67, "assistant", "Sure, go ahead and share the draft."),
        _turn(
            68,
            "user",
            "Actually never mind, I'll handle it myself. Where were we with the party?",
        ),
        _turn(
            69,
            "assistant",
            "We were on speeches/toasts. Let me give you the current checklist: Venue (Riverside Pavilion, $800, booked). Catering (Aunt Rosa, $600, Mexican buffet + cheese enchiladas). Drinks ($500, Mexican beer, red wine, horchata, agua fresca). DJ Marco ($600, 4 hours). Decorations ($500, DIY with Karen and Jenny, burgundy and gold theme). Photographer (Jason, $200). Videographer ($400, Karen's contact). Party favors ($150, hot sauce bottles). Kids corner ($30). Digital invitations (free, Paperless Post). Total committed: $3,780. Remaining: $1,220. Still to address: seating chart, surprise logistics, speeches, and any extras.",
        ),
        _turn(
            70,
            "user",
            "Wow, that's a great overview. I think we're in good shape. Oh, one thing I forgot - my parents' favorite flowers are gardenias. Mom always had gardenias at the house when we were growing up. Can we incorporate those into the decorations?",
        ),
        _turn(
            71,
            "assistant",
            "Gardenias would add a beautiful touch and a personal connection! They have a wonderful fragrance too. You could use them as table centerpieces alongside the photos. Fresh gardenias for 8-10 tables might run $150-200. Or you could do a mix of real gardenias on key tables and gardenia-scented candles for the others to stretch the budget.",
        ),
        _turn(
            72,
            "user",
            "Gardenias on the head table and a few key tables, candles for the rest. Smart. Let's budget $200 for flowers.",
        ),
        _turn(
            73,
            "assistant",
            "Updated budget: $3,980 of $5,000 committed. $1,020 remaining for emergencies and last-minute needs.",
        ),
        _turn(
            74,
            "user",
            "Good buffer. Karen just texted me - she found the original church program from Mom and Dad's wedding! She wants to recreate it as the party program.",
        ),
        _turn(
            75,
            "assistant",
            "What a find! Recreating the original church program as the party program would be an incredibly sentimental touch. You could print it in the same style with updated content - the order of events for the anniversary party instead of the wedding ceremony. Maybe $50-75 for printing 70 copies.",
        ),
        _turn(
            76,
            "user",
            "Love it. Add $75 for that. Ok I think we have a solid plan now.",
        ),
        _turn(
            77,
            "assistant",
            "Running total: $4,055. Buffer: $945. Here's the master checklist for the June 15th surprise 40th anniversary party...",
        ),
        _turn(
            78,
            "user",
            "Wait, one more thing. Aunt Rosa just called. She can't find her big serving platters. She lent them to someone and never got them back. We might need to rent platters.",
        ),
        _turn(
            79,
            "assistant",
            "Rental platters and serving ware shouldn't be too expensive. A party rental company might charge $50-100 for large platters, serving utensils, and chafing dishes to keep the food warm. Want me to research options?",
        ),
        _turn(
            80,
            "user",
            "Yeah, look into it. Oh, and we also need napkins, plates, and utensils for 60 people. I assumed the venue provided that but re-reading the contract, they don't.",
        ),
        _turn(
            81,
            "assistant",
            "Common oversight! For 60 people, quality disposable plates, napkins, and utensils in burgundy and gold to match the theme might run $100-150. Or real rentals for about $200-250. Which direction?",
        ),
        _turn(
            82,
            "user",
            "Let's do nice disposable ones in the theme colors. $150 should cover it.",
        ),
        _turn(
            83,
            "assistant",
            "Disposable tableware in burgundy and gold: $150. Plus serving platter rentals: ~$75. Updated total: $4,280. Buffer: $720.",
        ),
        _turn(
            84,
            "user",
            "Perfect. I think we're ACTUALLY done now. Let me save all this and send Karen the summary.",
        ),
        _turn(
            85,
            "assistant",
            "Great plan! Here's the final summary to send Karen: [Budget breakdown, vendor contacts, timeline, logistics, and checklist as discussed]. Want me to format it into a nice sendable format?",
        ),
        _turn(
            86,
            "user",
            "Yes please! And can you make a timeline of what needs to happen when? Working backwards from June 15th.",
        ),
        _turn(
            87,
            "assistant",
            "Here's the reverse timeline: By May 1 - send digital invitations (6 weeks out, giving people time to RSVP). By May 15 - finalize headcount and menu with Aunt Rosa, confirm all vendors (DJ Marco, videographer, Jason). By June 1 - buy decorations, order hot sauce bottles and labels, start DIY projects with Jenny. June 8-12 - Karen gathers photos for timeline display, print programs, buy non-perishable supplies. June 14 - Aunt Rosa starts cooking, buy fresh ingredients. June 15 morning - set up venue (you, Karen, Jenny). June 15 afternoon - final setup, flowers, sound check with DJ Marco. June 15 evening - Karen drives parents, you coordinate the surprise. Does that work?",
        ),
        _turn(88, "user", "That's exactly what we need. Perfect. Thank you so much!"),
        _turn(
            89,
            "assistant",
            "You're welcome! This is going to be a beautiful celebration. Your parents are lucky to have kids who put this much thought into it. Let me know if anything changes or if you need help with any of the steps!",
        ),
        _turn(
            90,
            "user",
            "Oh actually, one last LAST thing - I want to make a slideshow of photos to project during dinner. Can you suggest some software?",
        ),
        _turn(
            91,
            "assistant",
            "For a photo slideshow with music, Google Slides is free and easy, Canva has beautiful templates, or if you want something more cinematic, iMovie (free on Mac) or Windows Movie Maker work well. Since you have Jason doing photography, he might have access to Adobe Lightroom which can create nice slideshows too.",
        ),
        _turn(
            92,
            "user",
            "I'll use Canva, I already have an account. Set to 'Can't Help Falling in Love' as the background music for the first part, then Aunt Rosa suggested 'Cielito Lindo' for the Mexican family portion.",
        ),
        _turn(
            93,
            "assistant",
            "Beautiful music choices! Elvis for the romance, Cielito Lindo for the family heritage. That slideshow is going to make everyone cry happy tears.",
        ),
        _turn(
            94, "user", "Ha, that's the goal! Ok NOW I'm done. Thanks for everything."
        ),
        _turn(
            95,
            "assistant",
            "Happy planning! Feel free to come back anytime as things progress. Good luck keeping the surprise!",
        ),
    ]
    return turns


def _build_technical_conversation() -> list[dict]:
    """Technical / home setup: smart home, networking, automation rules."""
    turns = [
        _turn(
            0,
            "user",
            "Hey, I'm setting up a smart home system and I'm overwhelmed by the options. Can you help?",
        ),
        _turn(
            1,
            "assistant",
            "Of course! Smart home can be complex. What are your main goals? Security, convenience, energy savings?",
        ),
        _turn(
            2,
            "user",
            "Mainly convenience and energy savings. I just moved into a 3-bedroom house. I want to automate lighting, thermostat, and maybe some appliances.",
        ),
        _turn(
            3,
            "assistant",
            "Great goals. What's your current setup? Do you have any smart devices already?",
        ),
        _turn(
            4,
            "user",
            "I have a Google Nest thermostat I brought from my old apartment. And a couple of basic Philips Hue bulbs. That's it.",
        ),
        _turn(
            5,
            "assistant",
            "Good start! Since you have Google Nest, you're somewhat in the Google ecosystem. Are you leaning toward staying Google-centric, or are you open to mixing ecosystems?",
        ),
        _turn(
            6,
            "user",
            "I'd prefer to keep things somewhat unified. I have an Android phone and a Google Home speaker in the kitchen.",
        ),
        _turn(
            7,
            "assistant",
            "Google ecosystem makes sense then. Google Home as the hub, Nest for thermostat, and we can add compatible devices. What rooms do you want to prioritize?",
        ),
        _turn(
            8,
            "user",
            "Living room first, then bedroom, then kitchen. The third bedroom is my home office, that can wait.",
        ),
        _turn(
            9,
            "assistant",
            "Let's start with the living room. What do you want to automate there?",
        ),
        _turn(
            10,
            "user",
            "Lights for sure. I have a floor lamp, two table lamps, and overhead recessed lighting (4 cans). I want them all smart.",
        ),
        _turn(
            11,
            "assistant",
            "For the recessed cans, smart bulbs (like more Hue bulbs) or a smart switch that controls all 4 at once would work. The switch is usually more practical for recessed lights - one switch instead of 4 bulbs. For the floor lamp and table lamps, smart plugs or smart bulbs work. What's your preference?",
        ),
        _turn(
            12,
            "user",
            "Smart switch for the recessed lights, smart bulbs for the lamps since I want color changing.",
        ),
        _turn(
            13,
            "assistant",
            "Good plan. Lutron Caseta smart switches are excellent and work great with Google Home. For the color bulbs, your existing Hue ecosystem is the way to go - just add more Hue Color bulbs. You'll need a Hue Bridge if you don't have one already.",
        ),
        _turn(
            14,
            "user",
            "I do have a Hue Bridge, brought it from the apartment. It's set up in the living room closet near the router.",
        ),
        _turn(
            15,
            "assistant",
            "Perfect. So for the living room: 1 Lutron Caseta switch + bridge (~$100), 2-3 Hue Color bulbs (~$45 each). Does the floor lamp need color too?",
        ),
        _turn(
            16,
            "user",
            "Yes, color for all three lamps. So 3 Hue Color bulbs. The recessed lights can be just white - I want to dim them for movie watching but don't need colors.",
        ),
        _turn(
            17,
            "assistant",
            "Makes sense. Lutron Caseta is dimmable which is perfect for movie mode. Now, do you want to set up any automations? Like 'movie mode' that dims the recessed lights and sets the lamps to a warm color?",
        ),
        _turn(
            18,
            "user",
            "Yes! Movie mode is exactly what I want. Also a 'good morning' routine and a 'good night' one.",
        ),
        _turn(
            19,
            "assistant",
            "Great! Let's define those: 'Movie mode' - recessed lights dim to 10%, lamps set to warm amber at 30%. 'Good morning' - all lights on at 70%, cool white. 'Good night' - all lights off. Sound right?",
        ),
        _turn(
            20,
            "user",
            "Perfect. But for good morning, I want it to happen automatically at 6:45am on weekdays, and 8:30am on weekends.",
        ),
        _turn(
            21,
            "assistant",
            "Google Home routines can handle time-based triggers with different schedules for weekdays vs weekends. We can set that up once everything is installed. What about the thermostat - any specific schedule?",
        ),
        _turn(
            22,
            "user",
            "The Nest thermostat is already set pretty well from my old place. But in this house, the upstairs bedrooms get really hot in the afternoon. There are no smart vents yet - should I get some?",
        ),
        _turn(
            23,
            "assistant",
            "Smart vents like Flair or Keen Home can help with room-by-room temperature control. They're especially useful for two-story homes where heat rises. Flair Pucks integrate with Nest and can redirect airflow based on room temperature sensors. Might be worth considering for the bedrooms.",
        ),
        _turn(
            24,
            "user",
            "Interesting. Let me think about that. It's not urgent since it's winter right now. I'll revisit in spring.",
        ),
        _turn(
            25,
            "assistant",
            "Smart timing - no rush when it's cold! Let's continue with the current setup. What about the bedroom?",
        ),
        _turn(
            26,
            "user",
            "For the bedroom I want smart lighting too. Just the bedside lamps - two of them. And I want motion sensors.",
        ),
        _turn(
            27,
            "assistant",
            "Motion sensors for what purpose - security, or convenience like turning on a light when you get up at night?",
        ),
        _turn(
            28,
            "user",
            "Convenience. I want a dim nightlight to come on if I get up to use the bathroom at night. My wife trips in the dark all the time.",
        ),
        _turn(
            29,
            "assistant",
            "A Hue motion sensor would work great - it detects movement and can trigger a dim light. Set it to only activate between like 10pm and 6am, with the lights at maybe 5-10% brightness so it doesn't wake anyone up fully.",
        ),
        _turn(
            30,
            "user",
            "Exactly what I was thinking. And can it turn off automatically after a few minutes?",
        ),
        _turn(
            31,
            "assistant",
            "Yes, you can set an auto-off timer in the Hue app. Usually 2-5 minutes works well for a bathroom trip. The motion sensor stops detecting after the set time and turns the light off.",
        ),
        _turn(
            32,
            "user",
            "Great. Now for the kitchen - I already have the Google Home speaker there. I want to add some smart plugs for the coffee maker and maybe the slow cooker.",
        ),
        _turn(
            33,
            "assistant",
            "Smart plugs for the coffee maker are a classic automation - start brewing before you even get out of bed. For the slow cooker, smart plugs are great but make sure you use a plug with energy monitoring so you can verify it's actually off when it should be. TP-Link Kasa plugs with energy monitoring work well with Google Home.",
        ),
        _turn(
            34,
            "user",
            "Good call on the energy monitoring. Oh, one thing I should mention - my wife is NOT a techie at all. Everything needs to be simple enough that she can use it without getting frustrated.",
        ),
        _turn(
            35,
            "assistant",
            "That's a crucial design constraint! Voice control through Google Home will be her best friend - she can just say 'Hey Google, movie mode' or 'turn off the kitchen lights.' We should also keep physical switches functional as backup. The Lutron Caseta switch still works as a normal switch. And for Hue bulbs, make sure the wall switches stay ON so the bulbs stay powered.",
        ),
        _turn(
            36,
            "user",
            "Yeah, she already complains about the Hue bulbs when someone turns off the wall switch and the bulbs become unresponsive. Is there a solution?",
        ),
        _turn(
            37,
            "assistant",
            "A few options: 1) Hue dimmer switch covers - they go over the existing switch to prevent it being turned off, with a Hue remote attached. 2) Switch guards - cheap plastic covers. 3) Lutron Aurora dimmer - a knob that goes over the toggle switch, dims Hue lights, and prevents the switch from being turned off. The Aurora is the most elegant solution.",
        ),
        _turn(
            38,
            "user",
            "The Lutron Aurora sounds perfect. I'll get a couple of those for the rooms with Hue bulbs.",
        ),
        _turn(
            39,
            "assistant",
            "Good choice. They're about $40 each. So we're looking at 2 Auroras minimum - one for the living room lamps, one for the bedroom. Maybe a third if you add Hue bulbs to the kitchen.",
        ),
        _turn(
            40,
            "user",
            "Let me think about the kitchen bulbs. I might not need color there. Actually, I have a question about security. Should I get a doorbell camera?",
        ),
        _turn(
            41,
            "assistant",
            "A doorbell camera is one of the most practical smart home additions. The Nest Doorbell integrates perfectly with your Google ecosystem - you can see who's at the door on your phone or on a Nest Hub display, and it works with Google Home routines.",
        ),
        _turn(
            42,
            "user",
            "Makes sense. I've had a couple of package thefts since moving here, so a camera would be nice.",
        ),
        _turn(
            43,
            "assistant",
            "The Nest Doorbell (battery or wired) also has package detection alerts - it can specifically notify you when a package is dropped off. That's a great deterrent and evidence tool. You might also want a smart lock for the front door so you can buzz in delivery people remotely.",
        ),
        _turn(
            44,
            "user",
            "Smart lock is tempting but my wife would freak out. She doesn't trust digital locks. Let's hold off on that.",
        ),
        _turn(
            45,
            "assistant",
            "Totally understandable - physical keys feel more secure to many people. The doorbell camera alone is a great start for security.",
        ),
        _turn(
            46,
            "user",
            "Ok, let me also ask about my WiFi setup. I have the basic router from my ISP and with all these smart devices, I'm worried about bandwidth.",
        ),
        _turn(
            47,
            "assistant",
            "That's a smart concern. How many devices are on your network right now?",
        ),
        _turn(
            48,
            "user",
            "Let me count... 2 phones, 2 laptops, a tablet, the TV (it's a Chromecast), the Nest thermostat, the Google Home, the Hue Bridge, and I think that's it. About 10 devices.",
        ),
        _turn(
            49,
            "assistant",
            "With the smart home additions, you could easily get to 20-25 devices. ISP-provided routers often struggle past 15-20 devices. A mesh WiFi system like Google Nest WiFi (fits your ecosystem) or an Eero would handle more devices and give better coverage in a 3-bedroom house.",
        ),
        _turn(
            50,
            "user",
            "The Google Nest WiFi Pro - is that the newer one? How many units for a 3-bedroom, about 2000 sq ft?",
        ),
        _turn(
            51,
            "assistant",
            "The Nest WiFi Pro is the newer model with WiFi 6E support. For 2000 sq ft, a 2-pack should cover you well, maybe a 3-pack if the house layout has thick walls. It creates a mesh network so all the smart devices stay connected as you move around.",
        ),
        _turn(
            52,
            "user",
            "Let me think about that. It's about $300 for a 2-pack. That's a chunk on top of everything else.",
        ),
        _turn(
            53,
            "assistant",
            "True, but reliable WiFi is the foundation for everything else. Without it, all the smart devices become frustrating. You could also look at the regular Nest WiFi (non-Pro) for around $200 for a 2-pack - still a big upgrade from an ISP router.",
        ),
        _turn(
            54,
            "user",
            "Good point. Maybe the regular Nest WiFi. Let me budget that in.",
        ),
        _turn(
            55,
            "assistant",
            "Let me help with the overall budget. Shall I add up everything we've discussed?",
        ),
        _turn(56, "user", "Please do!"),
        _turn(
            57,
            "assistant",
            "Living room: Lutron Caseta switch + bridge ($100), 3 Hue Color bulbs ($135), 2 Lutron Auroras ($80). Bedroom: 2 Hue Color bulbs ($90), 1 Hue motion sensor ($40), 1 Lutron Aurora ($40). Kitchen: 2 TP-Link Kasa smart plugs ($30). Security: Nest Doorbell ($180). Network: Nest WiFi 2-pack ($200). Total estimate: $895. Plus you already have the thermostat, Hue Bridge, and Google Home.",
        ),
        _turn(
            58,
            "user",
            "Under $900 for the whole setup is not bad. But I'm going to phase it. Living room first, then bedroom and kitchen, then the doorbell and WiFi upgrade.",
        ),
        _turn(
            59,
            "assistant",
            "Phasing is smart. Phase 1 (living room) is about $315, Phase 2 (bedroom + kitchen) is about $200, Phase 3 (security + network) is about $380. That's much more manageable spread out.",
        ),
        _turn(
            60,
            "user",
            "Phase 1 this weekend. I'll order the stuff tonight. One question - will the Lutron Caseta and Hue work together in the same Google Home routines?",
        ),
        _turn(
            61,
            "assistant",
            "Yes, absolutely. Google Home acts as the unifying layer. You can create a routine that says 'Movie mode' and it will send commands to both the Lutron switch (dim recessed lights) and the Hue bulbs (set color/brightness) simultaneously.",
        ),
        _turn(
            62,
            "user",
            "Perfect. Ok I think I have a solid plan. Oh wait - one more thing. My home office. I said it could wait, but I do want one thing in there: a smart plug for my monitor so I can turn it off when I forget. I leave it on overnight all the time.",
        ),
        _turn(
            63,
            "assistant",
            "Easy add - one more Kasa smart plug ($15). You could even set an automation: if no motion detected in the office after 30 minutes, turn off the monitor. Just add a Hue motion sensor in there too ($40) for that automation.",
        ),
        _turn(
            64,
            "user",
            "Ooh, motion-based auto-off for the monitor. That's clever. But that's Phase... 4? Eventually. For now just the manual smart plug.",
        ),
        _turn(
            65,
            "assistant",
            "Smart plug now, motion automation later. Got it. Anything else?",
        ),
        _turn(
            66,
            "user",
            "I think that covers it! Oh, actually - one thing about the thermostat. I forgot to mention the Nest thermostat uses a C-wire, right? I had it professionally installed at the old place. In this house I'm not sure the wiring supports it.",
        ),
        _turn(
            67,
            "assistant",
            "Good catch. The Nest thermostat can work without a C-wire by charging its battery from other wires, but it's more reliable with one. If your current thermostat wiring doesn't have a C-wire, you can use a Nest Power Connector (comes with the thermostat or is about $25) as an adapter. Have you looked at the wires behind the current thermostat?",
        ),
        _turn(
            68,
            "user",
            "I haven't. I'll check tonight. The current thermostat is a basic Honeywell from like 2005.",
        ),
        _turn(
            69,
            "assistant",
            "Older Honeywell units sometimes have a C-wire that was just tucked away unused. Pull off the faceplate and look for a blue wire that might not be connected. If it's there, you're golden.",
        ),
        _turn(
            70,
            "user",
            "Cool, I'll check. Thanks for all the help! I'll let you know how the installation goes this weekend.",
        ),
        _turn(
            71,
            "assistant",
            "Sounds great! Enjoy the setup. Let me know if you run into any issues - smart home installation can sometimes have surprises.",
        ),
        _turn(
            72,
            "user",
            "Hey, it's the weekend. I installed the Lutron Caseta switch and it was easier than expected. But I have a problem with the Hue bulbs - two of the three I bought are showing warm white only, no color options in the app.",
        ),
        _turn(
            73,
            "assistant",
            "Hmm, that sounds like you may have accidentally bought Hue White Ambiance bulbs instead of Hue Color bulbs. The White Ambiance can do warm to cool white but no colors. Can you check the model number on the box?",
        ),
        _turn(
            74,
            "user",
            "Let me check... oh you're right. I grabbed the wrong ones at the store. The box says 'White Ambiance' not 'White and Color Ambiance.' Ugh.",
        ),
        _turn(
            75,
            "assistant",
            "Easy mistake - the packaging is similar. You can exchange them at the store. Make sure the box says 'White and Color Ambiance' and the model starts with 9290... The color bulbs are also slightly heavier and have a different diffuser shape.",
        ),
        _turn(
            76,
            "user",
            "I'll swap them tomorrow. At least the one color bulb I already had is working great. And the Caseta switch is awesome - dimming the recessed lights is so smooth.",
        ),
        _turn(
            77,
            "assistant",
            "Caseta switches really are premium quality - the fade is beautiful. Glad it went smoothly! Anything else from the installation?",
        ),
        _turn(
            78,
            "user",
            "Nope, just the bulb mixup. I'll finish up once I exchange them. Thanks!",
        ),
        _turn(
            79,
            "assistant",
            "Good luck with the exchange! Once you have the right bulbs, setting up movie mode is going to be satisfying.",
        ),
        _turn(
            80,
            "user",
            "Hey, quick question. I was telling my neighbor about the smart home setup and he mentioned something called Matter. Should I care about it?",
        ),
        _turn(
            81,
            "assistant",
            "Matter is a smart home protocol that Google, Apple, Amazon, and many device makers developed together. It's designed to make devices work across all ecosystems without bridges or proprietary protocols. Your newer Nest devices likely support it. Going forward, buying Matter-compatible devices is a good idea because it future-proofs you - if you ever switch ecosystems, your devices still work.",
        ),
        _turn(82, "user", "Good to know. Does my Hue Bridge support Matter?"),
        _turn(
            83,
            "assistant",
            "Yes! Philips Hue updated the Bridge to support Matter via a firmware update. So your existing setup is already compatible. The Lutron Caseta system also has a path to Matter through their newer hub (Smart Bridge Pro), though your current bridge may not support it directly.",
        ),
        _turn(
            84,
            "user",
            "Interesting. I'll keep Matter in mind for future purchases. Ok that's all, just curiosity. Thanks!",
        ),
        _turn(
            85,
            "assistant",
            "Anytime! Smart home is an evolving space, and staying informed about standards like Matter is a good move. Enjoy the weekend!",
        ),
    ]
    return turns


# ---------------------------------------------------------------------------
# 2. Task scenarios
# ---------------------------------------------------------------------------


def build_scenarios() -> list[dict]:
    """Build task scenarios with ground truth retrieval needs."""
    return [
        # ----- CONTROLS (direct retrieval) -----
        {
            "task": "What is Bob allergic to?",
            "conversation_id": "synth_personal",
            "source_turn_ids": [14, 91],
            "difficulty": "control",
            "depth_needed": 1,
            "notes": "Direct keyword match. The answer is peanuts. The shellfish allergy was retracted in turn 90-91. A correct system should find turn 14 (peanut allergy stated) and turn 90-91 (shellfish retracted) and report only peanuts.",
        },
        {
            "task": "What CMS is Acme Corp currently using?",
            "conversation_id": "synth_work",
            "source_turn_ids": [20],
            "difficulty": "control",
            "depth_needed": 1,
            "notes": "Direct factual lookup. The answer is WordPress with a custom theme.",
        },
        {
            "task": "What is the user's current A1C level?",
            "conversation_id": "synth_medical",
            "source_turn_ids": [14],
            "difficulty": "control",
            "depth_needed": 1,
            "notes": "Direct factual lookup. Answer is 6.8.",
        },
        # ----- CONJUNCTION CUES -----
        {
            "task": "Prepare a birthday gift for Bob.",
            "conversation_id": "synth_personal",
            "source_turn_ids": [4, 6],
            "difficulty": "conjunction",
            "depth_needed": 2,
            "notes": "Requires finding BOTH Bob's birthday timing (turn 4: next Thursday) AND his interests (turn 6: craft beer and woodworking). Neither 'birthday' alone nor 'Bob' alone is sufficient - the system needs the conjunction of gift-relevant info about Bob specifically.",
        },
        {
            "task": "Make a presentation for the Acme Corp client meeting on Wednesday.",
            "conversation_id": "synth_work",
            "source_turn_ids": [12, 14, 34, 36, 68, 86, 88],
            "difficulty": "conjunction",
            "depth_needed": 3,
            "notes": "Requires finding: brand colors (turns 12, 14), logo concepts (turn 34), meeting time (turn 36), typography (turn 68), brand voice direction (turn 86), and motion guidelines (turn 88). The task mentions 'presentation' and 'Acme Corp' but the relevant information is scattered across color, logo, font, voice, and motion topics.",
        },
        {
            "task": "Set up the living room smart home automation for movie mode.",
            "conversation_id": "synth_technical",
            "source_turn_ids": [12, 13, 16, 17, 18, 19, 35],
            "difficulty": "conjunction",
            "depth_needed": 2,
            "notes": "Requires knowing the specific devices (Lutron Caseta for recessed, Hue Color for lamps from turns 12-13, 16), the movie mode spec (turn 19: recessed 10%, lamps warm amber 30%), and the wife-friendliness constraint (turn 35: voice control, physical switches). The query is about 'movie mode' but implementation needs device-specific details scattered across the conversation.",
        },
        # ----- COMPLETENESS -----
        {
            "task": "What are ALL the dietary restrictions for the Saturday dinner party?",
            "conversation_id": "synth_personal",
            "source_turn_ids": [14, 16, 18, 28, 32, 40, 42, 50, 62, 64, 84, 91, 93],
            "difficulty": "completeness",
            "depth_needed": 3,
            "notes": "Must find ALL restrictions: Bob - peanut allergy (14), lactose intolerant (40, 42), keto preference (62, 64), shellfish allergy RETRACTED (91). Emma - tree nut allergy (50). Sarah - vegan (16), soy allergy (84). Patels - no beef (18). Linda - no mushrooms (28). Rachel - celiac (32). James - none. A system that finds only 2-3 of these gets partial credit. The shellfish retraction is a trap - must be caught. Turn 93 has the final summary but a system shouldn't rely on summaries alone.",
        },
        {
            "task": "List all of the user's current medications and dosages.",
            "conversation_id": "synth_medical",
            "source_turn_ids": [2, 4, 6, 8, 74, 78],
            "difficulty": "completeness",
            "depth_needed": 2,
            "notes": "Must find ALL medications: Metformin (originally 500mg turn 2, UPDATED to 1000mg turn 74), Lisinopril 10mg (turn 4), Atorvastatin 20mg (turn 6), Vitamin D 2000 IU (turn 8), Baby aspirin 81mg (turn 78, mentioned late). The metformin dosage update is critical - a system that reports 500mg is using stale info. The baby aspirin is mentioned casually much later.",
        },
        {
            "task": "What are all the budget line items for the anniversary party?",
            "conversation_id": "synth_planning",
            "source_turn_ids": [8, 14, 26, 30, 34, 40, 42, 44, 61, 72, 75, 80, 82],
            "difficulty": "completeness",
            "depth_needed": 2,
            "notes": "Must find ALL budget items: venue $800 (8), food $600 (14), decorations $500 (26), DJ $600 (30), drinks $500 (34), photographer $200 (40), videographer $400 (42), party favors $150 (44), kids corner $30 (61), flowers $200 (72), programs $75 (75), serving platters ~$75 (80 referenced, mentioned in context), tableware $150 (82). Total $4,280 of $5,000.",
        },
        {
            "task": "What smart home devices does the user plan to buy?",
            "conversation_id": "synth_technical",
            "source_turn_ids": [12, 13, 15, 16, 26, 29, 32, 33, 38, 41, 54, 57, 62],
            "difficulty": "completeness",
            "depth_needed": 2,
            "notes": "Must find: Lutron Caseta switch + bridge (12-13), 3 Hue Color bulbs (15-16), 2 Lutron Auroras living room + bedroom (38), 2 Hue Color bedside bulbs (26), 1 Hue motion sensor bedroom (29), 2 TP-Link Kasa smart plugs kitchen (32-33), Nest Doorbell (41), Nest WiFi 2-pack (54), 1 Kasa smart plug office (62). Turn 57 has the budget summary. Phasing info in turn 58.",
        },
        # ----- INFERENCE-DRIVEN -----
        {
            "task": "Is it safe for the user to take ibuprofen for their knee pain?",
            "conversation_id": "synth_medical",
            "source_turn_ids": [4, 22, 24, 25, 27, 78, 79],
            "difficulty": "inference",
            "depth_needed": 3,
            "notes": "Requires multi-step reasoning: (1) User takes ibuprofen for knee (turns 22, 24). (2) Ibuprofen interacts with lisinopril (turn 4 establishes lisinopril, turn 25 flags the interaction). (3) User also takes baby aspirin (turn 78), which compounds the risk (turn 79). A system needs world knowledge that NSAIDs + ACE inhibitors + aspirin is a problematic combination, then retrieves the specific medications to check.",
        },
        {
            "task": "Can Sarah eat the tres leches cake at the anniversary party?",
            "conversation_id": "synth_personal",
            "source_turn_ids": [16, 84],
            "difficulty": "inference",
            "depth_needed": 2,
            "notes": "Wait - Sarah (from the personal conversation) is not attending the anniversary party (that's a different conversation). But if someone conflates the two, the answer still requires knowing: Sarah is vegan (turn 16 in personal conv) and has a soy allergy (turn 84). Tres leches contains dairy (cream, milk, condensed milk), so she cannot eat it due to being vegan. This tests whether the system correctly scopes to the right conversation. However, tres leches is mentioned in the PLANNING conversation (turn 16 there), not the personal one. This is a cross-conversation inference scenario.",
        },
        {
            "task": "What medication interactions should the user discuss with Dr. Patel?",
            "conversation_id": "synth_medical",
            "source_turn_ids": [2, 4, 6, 8, 22, 24, 25, 57, 74, 78, 79],
            "difficulty": "inference",
            "depth_needed": 3,
            "notes": "Requires: (1) Enumerate all current meds (metformin 1000mg, lisinopril, atorvastatin, baby aspirin, vitamin D). (2) Know that ibuprofen (turn 22-24) interacts with lisinopril (turn 25) and aspirin (turn 79). (3) Know that fish oil (turn 57, user is considering it) has mild blood-thinning effect and should be discussed. A thorough system would also flag the metformin dosage increase (turn 74) as relevant context.",
        },
        # ----- PROACTIVE / PREEMPTIVE -----
        {
            "task": "Help me cook dinner for Bob tonight.",
            "conversation_id": "synth_personal",
            "source_turn_ids": [14, 40, 42, 62, 64, 91],
            "difficulty": "proactive",
            "depth_needed": 2,
            "notes": "The task doesn't mention dietary restrictions, but a proactive system should recall that Bob has a peanut allergy (turn 14), is lactose intolerant (turns 40, 42), prefers keto (turns 62, 64), and the shellfish allergy was retracted (turn 91). The retrieval need EMERGES from understanding that cooking for someone requires knowing their dietary constraints.",
        },
        {
            "task": "Send the team an update about where we stand on the Acme project.",
            "conversation_id": "synth_work",
            "source_turn_ids": [4, 6, 10, 20, 48, 52, 58, 63, 66, 68, 84, 86, 88],
            "difficulty": "proactive",
            "depth_needed": 3,
            "notes": "To write a meaningful team update, the system needs: team roster (turn 4), deadline (turn 6), design status (turn 10), tech stack decision (turn 20), accessibility requirement (turn 48), staging environment (turn 52), animation scope resolution (turns 58, 66), timeline phases (turn 63), typography (turn 68), presentation plan (turn 84), brand voice (turn 86), motion guidelines (turn 88). The task says 'update' but doesn't specify what about - the system must proactively determine all project-relevant info.",
        },
        {
            "task": "Prepare for the Dr. Patel appointment on January 25th.",
            "conversation_id": "synth_medical",
            "source_turn_ids": [18, 20, 22, 24, 34, 36, 46, 74, 78, 84],
            "difficulty": "proactive",
            "depth_needed": 3,
            "notes": "The system should proactively identify all topics for the appointment: knee pain (turns 18, 20, 22, 24), sleep issues (turns 34, 36), stress test scheduling (turn 46), metformin dosage change (turn 74), baby aspirin (turn 78), PSA monitoring (turn 84), and the ibuprofen interaction concern. The task says 'prepare' - the system must reason about what preparation means for a doctor visit.",
        },
        {
            "task": "Set up the bedroom smart home features.",
            "conversation_id": "synth_technical",
            "source_turn_ids": [26, 28, 29, 30, 31, 34, 38, 39],
            "difficulty": "proactive",
            "depth_needed": 2,
            "notes": "Direct request but needs proactive recall of: bedside lamp automation (turn 26), motion sensor for bathroom trips (turns 28-31), wife's tech comfort level as a constraint (turn 34), and Lutron Aurora switch needed (turns 38-39). The wife constraint is especially important - it's mentioned in a different part of the conversation but critically affects bedroom setup.",
        },
        # ----- PROCEDURAL CHECKLISTS -----
        {
            "task": "What still needs to be done before the anniversary party on June 15th?",
            "conversation_id": "synth_planning",
            "source_turn_ids": [
                2,
                46,
                48,
                50,
                54,
                56,
                58,
                62,
                64,
                69,
                78,
                80,
                82,
                86,
                87,
                90,
                92,
            ],
            "difficulty": "procedural",
            "depth_needed": 3,
            "notes": "Requires assembling a full checklist from scattered mentions: surprise logistics (turns 46-50), seating chart with constraints (turns 54-58), speeches coordination (turns 62-64), vendor confirmations needed (turn 69 summary), serving platter rental (turn 78-80), tableware purchase (turn 82), slideshow creation (turns 90, 92), and the reverse timeline (turn 87). The system must also distinguish between items done and items pending.",
        },
        {
            "task": "Walk me through the remaining phases of my smart home setup.",
            "conversation_id": "synth_technical",
            "source_turn_ids": [
                22,
                24,
                40,
                41,
                44,
                46,
                49,
                52,
                54,
                57,
                58,
                62,
                63,
                64,
                66,
            ],
            "difficulty": "procedural",
            "depth_needed": 2,
            "notes": "Phase 1 (living room) is done or in progress. Remaining: Phase 2 bedroom + kitchen (turns 26-33), Phase 3 security + network (doorbell turn 41, WiFi turns 49-54), Phase 4 office monitor auto-off (turns 62-64), and deferred items: smart vents for upstairs (turns 22-24), smart lock (turn 44, rejected by wife), kitchen Hue decision (turn 40). The system must reconstruct the phased plan and identify what's done vs pending.",
        },
    ]


# ---------------------------------------------------------------------------
# 3. Questions JSON (matching existing format)
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    "control": "control",
    "conjunction": "conjunction",
    "completeness": "completeness",
    "inference": "inference",
    "proactive": "proactive",
    "procedural": "procedural",
}

# Map each scenario to a user-facing question for evaluation
SCENARIO_QUESTIONS = {
    # Controls
    "What is Bob allergic to?": {
        "question": "What is Bob allergic to? Please include any updates or corrections mentioned later in the conversation.",
        "ideal_response": "Bob is allergic to peanuts. He was previously reported as allergic to shellfish, but his allergist recently confirmed he outgrew that allergy.",
    },
    "What CMS is Acme Corp currently using?": {
        "question": "What content management system (CMS) is Acme Corp currently using for their website?",
        "ideal_response": "Acme Corp is currently using WordPress with a custom theme. The plan is to migrate to Strapi (headless CMS) with a Next.js frontend.",
    },
    "What is the user's current A1C level?": {
        "question": "What is the user's most recent A1C level and how has it been trending?",
        "ideal_response": "The user's most recent A1C is 6.8, down from 7.2 six months ago. Dr. Patel was pleased with the improvement.",
    },
    # Conjunction
    "Prepare a birthday gift for Bob.": {
        "question": "I need to buy a birthday gift for Bob. When is his birthday and what are his interests?",
        "ideal_response": "Bob's birthday is next Thursday. He's really into craft beer and woodworking. A nice set of carving chisels or a craft beer subscription were suggested as options.",
    },
    "Make a presentation for the Acme Corp client meeting on Wednesday.": {
        "question": "What content should be included in the presentation for the Acme Corp client meeting on Wednesday?",
        "ideal_response": "The presentation should cover: (1) Brand color palette - primary Pantone 2945 C deep blue, proposed new secondaries of cool gray and teal. (2) Three logo concepts from Vanessa. (3) Typography - Acme Sans custom typeface with Inter fallback. (4) Brand voice guide - 'confident but conversational' per Hiroshi. (5) Website direction - Strapi + Next.js with simplified parallax hero. (6) Motion guidelines from Vanessa. (7) Timeline review. Vanessa presents designs, project lead handles Q&A.",
    },
    "Set up the living room smart home automation for movie mode.": {
        "question": "What are the exact specifications for the living room movie mode automation?",
        "ideal_response": "Movie mode: Lutron Caseta switch dims the 4 recessed lights to 10%. Three Hue Color bulbs in the floor lamp and two table lamps set to warm amber at 30% brightness. Voice control via Google Home ('Hey Google, movie mode'). Physical switches still work as backup (Lutron Auroras prevent wall switch from cutting power to Hue bulbs).",
    },
    # Completeness
    "What are ALL the dietary restrictions for the Saturday dinner party?": {
        "question": "List ALL dietary restrictions and food preferences for every guest at the Saturday dinner party, including any updates or corrections.",
        "ideal_response": "Bob: peanut allergy (severe, since childhood), lactose intolerant (aged cheese/butter OK, no milk/cream/soft cheese), keto preference (flexible). Shellfish allergy was RETRACTED - allergist confirmed he outgrew it. Emma (Bob's daughter, age 8): tree nut allergy (severe, carries EpiPen). Sarah: vegan (no eggs or dairy), soy allergy. Priya & Dev Patel: no beef (Hindu). Linda (Bob's wife): no mushrooms (preference, not allergy). Rachel (user's sister): celiac disease (strict gluten-free, even trace amounts). James: no restrictions.",
    },
    "List all of the user's current medications and dosages.": {
        "question": "What are all of the user's current medications, including dosages and what they're for? Include any recent changes.",
        "ideal_response": "1) Metformin 1000mg twice daily (type 2 diabetes) - UPDATED from 500mg to 1000mg. 2) Lisinopril 10mg once daily (blood pressure), prescribed by Dr. Patel. 3) Atorvastatin 20mg at night (cholesterol), been on it for years. 4) Baby aspirin 81mg daily (cardiac prevention, due to family history). 5) Vitamin D 2000 IU daily (levels were low at 22, now at 38, target 40).",
    },
    "What are all the budget line items for the anniversary party?": {
        "question": "List every budget line item for the 40th anniversary party with costs.",
        "ideal_response": "Venue (Riverside Pavilion): $800. Catering (Aunt Rosa, Mexican buffet): $600. Decorations (DIY, burgundy & gold): $500. DJ Marco (4 hours): $600. Drinks (beer, wine, horchata, agua fresca): $500. Photographer (nephew Jason): $200. Videographer: $400. Party favors (hot sauce bottles): $150. Kids corner supplies: $30. Flowers (gardenias): $200. Printed programs: $75. Serving platter rentals: ~$75. Disposable tableware: $150. Total: $4,280 of $5,000 budget. Buffer: $720.",
    },
    "What smart home devices does the user plan to buy?": {
        "question": "List all smart home devices the user plans to purchase, organized by room and phase.",
        "ideal_response": "Phase 1 (living room): 1 Lutron Caseta switch + bridge ($100), 3 Hue Color bulbs ($135), 2 Lutron Auroras ($80). Phase 2 (bedroom + kitchen): 2 Hue Color bulbs ($90), 1 Hue motion sensor ($40), 1 Lutron Aurora ($40), 2 TP-Link Kasa smart plugs ($30). Phase 3 (security + network): Nest Doorbell ($180), Nest WiFi 2-pack ($200). Phase 4 (office, eventually): 1 Kasa smart plug ($15). Total: ~$910. Already owned: Google Nest thermostat, Hue Bridge, Google Home speaker, some Hue bulbs.",
    },
    # Inference
    "Is it safe for the user to take ibuprofen for their knee pain?": {
        "question": "Given the user's complete medication list, is it safe for them to continue taking ibuprofen for their knee pain?",
        "ideal_response": "There are two concerns with ibuprofen use: (1) Ibuprofen (NSAID) can reduce the effectiveness of lisinopril (their blood pressure medication) and affect kidney function, which is especially important given their diabetes. (2) Ibuprofen combined with daily baby aspirin (81mg) increases the risk of GI bleeding since both affect platelet function. Recommendation: discuss alternatives with Dr. Patel. Acetaminophen (Tylenol) doesn't have these interactions. Physical therapy is also an option.",
    },
    "What medication interactions should the user discuss with Dr. Patel?": {
        "question": "Based on everything in the conversation, what medication interactions and health concerns should the user bring up with Dr. Patel at their January 25th appointment?",
        "ideal_response": "Key interactions: (1) Ibuprofen + lisinopril: reduced blood pressure control and kidney risk. (2) Ibuprofen + baby aspirin: increased GI bleeding risk. (3) Fish oil (under consideration) has mild blood-thinning effect, should be cleared given existing meds. Other topics: knee pain evaluation (3 weeks, dull ache, worse on stairs), sleep disruption (waking at 3am for 2 weeks, likely stress-related), metformin increase to 1000mg (recent change, monitoring needed), cardiac stress test scheduling (family history: father's heart attack at 62), PSA retest planning (4.5, slightly elevated).",
    },
    # Proactive
    "Help me cook dinner for Bob tonight.": {
        "question": "I want to cook dinner for Bob tonight. What should I keep in mind?",
        "ideal_response": "Bob's dietary constraints: (1) Peanut allergy - severe, since childhood. Avoid all peanuts and peanut-derived ingredients. (2) Lactose intolerant - no milk, cream, or soft cheese. Aged cheeses and butter are OK. (3) Keto preference - prefers low carb, but flexible on this. (4) Previously had shellfish allergy but allergist confirmed he outgrew it, so shellfish is now OK. Suggested approach: protein-focused dish (fish, chicken, steak) with vegetable sides, using butter or aged cheese if needed, avoiding any nut ingredients.",
    },
    "Send the team an update about where we stand on the Acme project.": {
        "question": "Draft a project status update for the Acme Corp rebrand team covering current progress and next steps.",
        "ideal_response": "Team: user (project lead), Vanessa (design), Tom (frontend), Maria (backend), Hiroshi (content). Deadline: March 15 (non-negotiable, client product launch March 20). Status: Design direction approved (modern minimalist). Color palette proposed (keep Pantone 2945 C blue, new secondaries: cool gray + teal). Three logo concepts ready. Acme Sans typography with Inter fallback. Brand voice: 'confident but conversational' (Hiroshi completed guide). Website: migrating from WordPress to Strapi + Next.js. Simplified parallax hero (compromise on animation scope with CMO Patricia). WCAG 2.1 AA compliance required. Maria setting up Vercel staging. Motion guidelines being added to brand book. Wednesday meeting: Vanessa presents colors + logos, project lead handles Q&A. Daniel Park (new CEO) driving approachable brand direction.",
    },
    "Prepare for the Dr. Patel appointment on January 25th.": {
        "question": "Help me prepare a list of topics to discuss with Dr. Patel at my January 25th appointment.",
        "ideal_response": "Topics: (1) Knee pain - right knee, 3 weeks, dull ache/sharp on stairs, no specific injury, currently using ibuprofen 400mg 2-3x/week. (2) Ibuprofen safety - interactions with lisinopril (blood pressure) and baby aspirin (bleeding risk). Ask about alternatives. (3) Sleep issues - waking at 3am for 2 weeks, likely stress-related, chamomile tea tried, considering melatonin. (4) Metformin increase follow-up - recently increased to 1000mg 2x daily. (5) Cardiac stress test - schedule this (family history: father heart attack at 62). (6) Fish oil supplement - considering for knee inflammation, check interactions. (7) Physical therapy referral - wife recommends ProMotion PT on Oak Street. (8) PSA monitoring - last was 4.5, slightly elevated, April retest planned.",
    },
    "Set up the bedroom smart home features.": {
        "question": "What needs to happen to set up the bedroom smart home features?",
        "ideal_response": "Bedroom setup: (1) Two Hue Color bulbs for bedside lamps. (2) One Hue motion sensor - set to activate between 10pm-6am, trigger dim nightlight at 5-10% for bathroom trips (wife trips in the dark). Auto-off after 2-5 minutes via Hue app. (3) One Lutron Aurora knob over the wall switch to prevent killing power to Hue bulbs. Important constraint: wife is not tech-savvy, so everything must be voice-controllable via Google Home and intuitive. Physical switches must remain functional as backup.",
    },
    # Procedural
    "What still needs to be done before the anniversary party on June 15th?": {
        "question": "Create a complete checklist of remaining tasks for the June 15th anniversary party, organized by timeline.",
        "ideal_response": "By May 1: Send digital invitations via Paperless Post. By May 15: Finalize headcount, confirm menu with Aunt Rosa (add cheese enchiladas for David, fruit plate for Uncle Frank), confirm DJ Marco, videographer, Jason (photographer). By June 1: Buy decorations (burgundy & gold), order hot sauce bottles + custom labels, start DIY projects with Jenny, rent serving platters. Seating plan: separate Uncle Steve and Aunt Patricia, wheelchair-accessible spot for Grandma Helen near entrance on paved area. June 8-12: Karen gathers photos for timeline display, print recreated church programs ($75), buy tableware. June 14: Aunt Rosa starts cooking, buy fresh ingredients. Create Canva slideshow (Elvis + Cielito Lindo music). June 15 morning: Set up venue with Karen and Jenny. June 15 afternoon: Flowers (gardenias for head table, candles for others), kids corner (coloring books), sound check with DJ Marco. Coordinate speeches: Uncle Roberto + Gloria + open mic. Surprise logistics: Karen tells parents it's her birthday dinner, drives them, texts 10 min warning.",
    },
    "Walk me through the remaining phases of my smart home setup.": {
        "question": "What are the remaining phases of the smart home setup and what's in each phase?",
        "ideal_response": "Phase 1 (this weekend, ~$315): Living room - Lutron Caseta switch for recessed lights, 3 Hue Color bulbs for lamps (NOTE: need to exchange 2 White Ambiance for Color), 2 Lutron Auroras. Set up 'movie mode,' 'good morning' (6:45am weekdays, 8:30am weekends), 'good night' routines. Phase 2 (~$200): Bedroom - 2 Hue Color bedside bulbs, 1 Hue motion sensor (nightlight 10pm-6am), 1 Lutron Aurora. Kitchen - 2 TP-Link Kasa smart plugs (coffee maker, slow cooker with energy monitoring). Phase 3 (~$380): Security - Nest Doorbell (package theft deterrent). Network - Nest WiFi 2-pack ($200, needed for 20+ devices). Phase 4 (eventually): Office - Kasa smart plug for monitor auto-off. Deferred: Smart vents for upstairs bedrooms (revisit in spring), smart lock (wife vetoed), kitchen Hue bulbs (undecided), office motion sensor automation. C-wire check needed for Nest thermostat installation.",
    },
}


def build_questions(scenarios: list[dict]) -> list[dict]:
    """Convert scenarios to the questions JSON format."""
    questions = []
    for i, scenario in enumerate(scenarios):
        task = scenario["task"]
        q_info = SCENARIO_QUESTIONS.get(task, {})
        questions.append(
            {
                "conversation_id": scenario["conversation_id"],
                "category": scenario["difficulty"],
                "question_index": i,
                "question": q_info.get("question", task),
                "source_chat_ids": scenario["source_turn_ids"],
                "ideal_response": q_info.get("ideal_response", ""),
                "benchmark": "synthetic",
            }
        )
    return questions


# ---------------------------------------------------------------------------
# 4. Embedding and output
# ---------------------------------------------------------------------------


def truncate_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    texts = [truncate_text(t) for t in texts]
    all_embeddings = []
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
    print("Building synthetic conversations...")
    conversations = build_conversations()
    total_turns = sum(len(turns) for turns in conversations.values())
    for conv_id, turns in conversations.items():
        print(f"  {conv_id}: {len(turns)} turns")
    print(f"  Total: {total_turns} turns across {len(conversations)} conversations")

    # Build scenarios and questions
    print("\nBuilding task scenarios...")
    scenarios = build_scenarios()
    print(f"  {len(scenarios)} scenarios")
    from collections import Counter

    difficulty_counts = Counter(s["difficulty"] for s in scenarios)
    for d, c in sorted(difficulty_counts.items()):
        print(f"    {d}: {c}")

    questions = build_questions(scenarios)
    print(f"  {len(questions)} questions generated")

    # Flatten all turns into segments
    all_segments = []
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
    npz_path = OUTPUT_DIR / "segments_synthetic.npz"
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
    json_path = OUTPUT_DIR / "questions_synthetic.json"
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
