"""Metacognitive completeness gate experiment.

Architecture (REQUIRED):
  (A) BOUNDED WORKING MEMORY <= 10k tokens. Enforced by tiktoken counting + LLM compaction.
  (B) EXTERNAL MEMORY 30-100k tokens. Lives in a Python dict and a numpy embedding index. NOT in
      the prompt. Retrieval surfaces top-K snippets per probe.
  (C) RETRIEVAL ON DEMAND. Each round emits a probe -> embedding kNN against external store ->
      top-K snippets returned to WM.
  (D) COMPACTION between rounds. When WM > threshold (8k), an LLM-summarize pass compacts the
      "older" portion of WM into a structured GATHERED_NOTES section, dropping raw snippet text.
  (E) SUBSTANTIVE TASK. Total external memory per case ~50k tokens; gold facts are sprinkled
      among >95% noise.

Two variants:
  - BASELINE: terminate when a probe round surfaces no new fact-IDs (saturation) OR rounds
    cap = 8.
  - OPERATOR: a separate completeness-gate LLM call decomposes the task into sub-questions, checks
    coverage in compacted WM, emits OPEN gaps + NEXT_QUERY, or COMPLETE. Iter cap = 12.

Metric: gold facts surfaced / total gold facts; token-trace per round.

Run:  uv run python evaluation/associative_recall/metacog/completeness_gate/main.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI

THIS_DIR = Path(__file__).resolve().parent
ENV_PATH = THIS_DIR.parent.parent / ".env"
load_dotenv(ENV_PATH)

CHAT_MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

ENC = tiktoken.encoding_for_model("gpt-4o")  # close enough proxy for token counting

# Architecture knobs
WM_TOKEN_CAP = 10_000  # hard ceiling for working memory
WM_COMPACT_TRIGGER = 8_000  # compact when above this
TOP_K = 4  # retrieval depth per probe
MAX_BASELINE_ROUNDS = 8
MAX_OPERATOR_ROUNDS = 12


# ============================================================================
# External memory entries (synthetic)
# ============================================================================


@dataclass
class MemEntry:
    fact_id: str
    text: str  # 100-500 token blob


@dataclass
class Case:
    case_id: str
    domain: str
    task: str
    entries: list[MemEntry]
    gold_fact_ids: list[str]
    gold_subquestions: list[str]  # used only for analysis, not given to model


def _toks(text: str) -> int:
    return len(ENC.encode(text))


# ----------------------------------------------------------------------------
# Synthetic case construction
# ----------------------------------------------------------------------------
#
# Each case has:
#   - 8-12 GOLD entries: load-bearing facts that any sound plan must consider
#   - ~150-250 NOISE entries: domain-relevant flavor text + cross-domain noise
# Each entry is ~150-300 tokens. Target total: 40-55k tokens external.

NOISE_PARAGRAPHS = [
    "The quarterly fiscal report from the regional office indicated a 4% increase in operational efficiency, driven primarily by upgraded HVAC systems in the northern facility. Maintenance logs corroborate the trend.",
    "Local newspaper clippings from the 1987 town festival describe a community dunking booth, a bake-off won by Mrs. Petersen's lemon meringue pie, and an unusually warm October evening with light winds from the south.",
    "An academic paper on the migration patterns of European starlings concluded that urban populations exhibit reduced sensitivity to artificial light cycles compared to rural counterparts, with implications for circadian research.",
    "Notes from a craft beer tasting in 2019 listed eight stouts, four IPAs, and two saisons, with detailed flavor profiles emphasizing oak-aged characteristics and faint citrus undertones.",
    "An old maintenance manual for steam locomotives, page 43, describes the procedure for inspecting the smokebox door for warping and recommends a quarterly silicone gasket reseal during the off-season.",
    "Survey results from a 2014 transportation study covering metropolitan commuting times found average peak-hour duration of 38 minutes, with a 12% standard deviation across the sample.",
    "An archived blog post about home brewing outlines step-by-step instructions for a five-gallon batch of saison, including a warm-fermentation schedule and dry-hopping at day 14.",
    "Specifications for the model 1985 lawn tractor list a 17 horsepower Kawasaki engine, 42-inch deck, and a 1.6-gallon fuel tank, with maintenance intervals of 50 hours.",
    "A travelogue from a backpacking trip through Patagonia records a five-day trek, a glacier crossing, and a chance meeting with a German ornithologist studying Andean condor nesting cliffs.",
    "Press release from a regional hospital announcing a new MRI suite, complete with a 3-Tesla machine and fully shielded copper-lined imaging room, expected to come online in Q4.",
    "Recipe for a traditional Hungarian goulash with paprika, marrow bones, beef chuck, and slow-simmered onions; passed down from a grandmother in the village of Eger.",
    "Field journal entry from a botanist describing the spring wildflower bloom along the eastern ridge, including indian paintbrush, lupine, and patches of glacier lilies near melting snowfields.",
    "An old shipping manifest from 1952 lists crates of dried apricots, machine parts, sealed barrels of olive oil, and one stowaway cat that was discovered three days into the voyage.",
    "Computer science lecture notes on B-tree indexing explain that fan-out reduces the height of the tree, and that block-aligned reads dominate practical performance for disk-backed structures.",
    "An interview transcript with a retired jazz drummer recalls touring small venues across the Midwest in the 1970s, sleeping in vans, and a fondness for hotel coffee that rarely lived up to expectation.",
    "Inventory list from a small bakery: 12 sourdough loaves, 30 baguettes, 24 croissants, 18 pain au chocolat, and a remaining supply of imported French butter sufficient for three more days of production.",
    "An academic abstract on glacial retreat in the Karakoram noted that some glaciers in the region exhibit anomalous stability or even slight advance, attributed to localized precipitation patterns.",
    "Memo from a 2008 conference call discussing inventory shortages, supplier delays, and a contingent plan involving two backup vendors with overnight shipping arrangements.",
    "A history of the local railroad describes the closing of three small stations between 1962 and 1971, citing declining passenger volume and the rise of automobile-based commuting in the region.",
    "Detailed weather observations from a remote alpine weather station: wind 18 km/h NW, temperature -3C at dawn rising to 4C by noon, scattered cumulus, visibility unrestricted.",
    "Catalog description of an heirloom tomato variety: deep red with green shoulders, prone to cracking under inconsistent watering, best pruned to single-stem culture for high-tunnel production.",
    "Notes from a homeowner's association meeting: a discussion of pool hours, a proposal to repaint the clubhouse, and a heated debate over whether to allow chickens in residential yards.",
    "A 1974 advertisement for a shortwave radio receiver claims dual-conversion superheterodyne architecture, BFO for SSB, and coverage from 100 kHz to 30 MHz in eleven bands.",
    "Travel advisory from a national park: bear activity reported near the eastern ridge campground, mandatory food storage in approved canisters, and a temporary closure of the Cedar Creek trail.",
    "Book review of a 2003 historical novel set in 17th-century Amsterdam: praised for vivid period detail and a complex protagonist, faulted for its subplot involving a tulip merchant.",
    "Building specs for a turn-of-the-century mill: 38 by 120 feet, post-and-beam framing, original waterwheel preserved in the lower level, and slate roofing replaced in 1998.",
    "A vintage car-club newsletter describes a summer rally for pre-1965 British sports cars, with technical inspection requirements, route notes through wine country, and a final-day concours.",
    "Excerpts from a diary kept during a months-long writing residency in rural Iceland, recording the rhythm of wind, sheep, and silent weeks of work on a novel manuscript.",
    "Investment letter discussing the cyclical nature of small-cap value stocks, with reference to multi-decade rolling returns and the role of patient capital in capturing the value premium.",
    "A child's hand-drawn map of a backyard, complete with treasure hidden under the swing, a path past the rosebush, and a warning sign reading 'WATCH OUT FOR THE DOG'.",
    "Scientific paper on superconducting magnets for fusion devices, discussing tape-stack architectures, quench-detection circuits, and the engineering trade-offs of magnetic field uniformity.",
    "Restaurant review of a small Italian trattoria praising the handmade pici pasta, a slow-braised wild boar ragu, and the chef's grandmother's pistachio gelato recipe.",
    "Notes from a high school chess club tournament: pairings, time controls of 60+30, and the dramatic final-round draw between the defending champion and a transfer student.",
    "An obituary of a 20th-century painter known for large abstract expressionist canvases inspired by the Maine coastline, with a retrospective in 1992 at a major regional museum.",
    "A short biography of a 19th-century clockmaker whose marine chronometers were prized by both merchant captains and the Royal Navy for their accuracy in tropical climates.",
]


def _make_noise(seed: int, min_count: int, max_count: int) -> list[str]:
    """Build long-form noise entries 200-450 tokens each by concatenating 3-6 paragraphs."""
    rng = random.Random(seed)
    n = rng.randint(min_count, max_count)
    out = []
    for i in range(n):
        k = rng.randint(3, 6)
        chosen = [rng.choice(NOISE_PARAGRAPHS) for _ in range(k)]
        out.append(" ".join(chosen))
    return out


def _build_case(
    case_id: str,
    domain: str,
    task: str,
    gold: list[tuple[str, str]],  # (fact_id, text)
    domain_filler: list[str],
    gold_subquestions: list[str],
    seed: int,
) -> Case:
    rng = random.Random(seed)
    entries: list[MemEntry] = []

    # Add domain filler that LOOKS relevant but isn't load-bearing for any sub-question.
    for i, txt in enumerate(domain_filler):
        entries.append(MemEntry(fact_id=f"{case_id}_filler_{i:03d}", text=txt))

    # Add cross-domain noise.
    for i, txt in enumerate(_make_noise(seed=seed, min_count=160, max_count=200)):
        entries.append(MemEntry(fact_id=f"{case_id}_noise_{i:03d}", text=txt))

    # Add gold last, then shuffle.
    gold_ids = []
    for fid, txt in gold:
        entries.append(MemEntry(fact_id=fid, text=txt))
        gold_ids.append(fid)

    rng.shuffle(entries)
    return Case(
        case_id=case_id,
        domain=domain,
        task=task,
        entries=entries,
        gold_fact_ids=gold_ids,
        gold_subquestions=gold_subquestions,
    )


# Domain filler: sentences that share vocabulary with the task / look plausibly relevant
# but do NOT carry information that changes the plan. Designed to lure baseline cue gen.

BANQUET_FILLER = [
    "Last spring's track meet had record-breaking attendance, with parents reporting that the snack stand sold out of pretzels by halftime.",
    "The high school yearbook committee usually orders catering for their end-of-year picnic from a local pizza chain on Pine Street.",
    "Coach Hernandez once remarked that he ran a 4:12 mile in college but his knees no longer permit jogging more than a few times a week.",
    "The track team's away meet schedule includes three regional invitationals and one state qualifier, depending on team rankings going into May.",
    "Fundraising for new pole-vault standards involved a car wash and a bake sale that netted around $1,800, well above the original target.",
    "The school's marching band rehearses on the football field on Tuesdays and Thursdays, occasionally overlapping with track practice in the spring.",
    "Last year's track team t-shirts featured a bold stylized silhouette of a runner in mid-stride, screen-printed in navy on heather grey heavyweight cotton.",
    "Several team members carpool from the eastern part of town, often stopping at a bagel shop on the way to early-morning weekend practices.",
    "The school's PTA newsletter covered the spring sports preview but only briefly mentioned track and field, focusing instead on baseball and lacrosse.",
    "The PE department keeps a stocked first-aid kit in the locker room, which is checked monthly by Coach Diaz and the assistant athletic trainer.",
    "Last year's senior prank involved a herd of inflatable flamingos placed in front of the principal's office; no track team members were implicated.",
    "Local radio station 92.3 FM frequently plays the school fight song during pep rallies, which the cheer squad choreographs new routines to each year.",
    "The varsity track captain wrote a winning college essay about resilience after recovering from a stress fracture during sophomore year.",
    "An adjacent middle school sometimes uses the high school track for their gym class, scheduled around the high school team's afternoon training blocks.",
    "Records from the 2011 banquet at a different high school show that the catering came from a deli, with sandwiches, fruit trays, and lemonade.",
    "The school's mascot, a lion, makes appearances at home meets, performed by a senior in a hot full-body costume with limited visibility.",
    "Custodial staff prefer that any school events end by 9 PM to allow for cleanup, although exceptions are routinely granted for official functions.",
    "The school's official motto, embossed on the gym wall, reads 'Strength through perseverance' in stylized gold lettering above the basketball hoops.",
    "A former track captain became a sports nutritionist and occasionally posts protein-bar recipes on the school's alumni Facebook page.",
    "Practice on hot days often moves to early morning to avoid heat exhaustion, with hydration breaks every 20 minutes per coach's standing rule.",
    "The local newspaper covered the team's regional finish last year with a photo of the sprint relay team holding up a third-place trophy.",
    "Some parents on the booster club prefer to volunteer for setup rather than donate cash, which has occasionally caused friction with the treasurer.",
    "The track team's pre-meet ritual includes a five-minute team huddle and a chant adopted from the 2018 squad, repeated three times before warmups.",
    "An old equipment shed behind the long-jump pit holds extra hurdles and a few cracked starting blocks slated for replacement next budget cycle.",
    "Maya's older brother also ran on the team three years ago and held a school record for the 800-meter that was broken last season by a freshman.",
    "The cafeteria's lunchroom has 30 round tables that seat 8 each, and a small stage that the drama club uses during after-school rehearsals.",
    "A booster club volunteer once made a homemade banner for senior night that had a typo of 'Senoirs' visible from the bleachers.",
    "Three years ago, the team did a pasta dinner the night before the league championship at a local Italian restaurant on Oak Avenue.",
    "Devon's mom is a part-time photographer who has volunteered to take team portraits during the past two seasons at no charge.",
    "The athletic department's annual budget allocates around $2,200 for end-of-season banquets across all spring sports, split by participation count.",
    "School-wide announcements typically air at 7:55 AM and include weather, lunch menu, and sports results from the previous day.",
    "The team's captain is responsible for organizing the warmup at home meets, customarily a 10-minute jog and dynamic stretching sequence.",
    "Many seniors apply to the local state university, where the track program is Division II and offers small scholarships for walk-on athletes.",
    "An assistant coach drives the bus to most away meets, having earned the appropriate commercial license through a district training program.",
    "The school's gym was renovated in 2009 with new bleachers and a refinished floor, but the locker rooms have not been upgraded since.",
    "Track athletes are required to maintain at least a 2.5 GPA to remain on the varsity roster, per the district's academic eligibility policy.",
    "Last year's awards banquet featured a slideshow of season photos set to music chosen by the senior class president and the team's sprint coach.",
]


BANQUET_GOLD = [
    (
        "banq_g_allergy_peanut",
        "Two members of the track team, Maya Watson and Devon Cole, have a confirmed peanut allergy that has triggered a hospital visit in the past, and any food served at team events must be peanut-free including shared utensils.",
    ),
    (
        "banq_g_allergy_dairy",
        "Captain Lina Park is severely lactose intolerant, requiring strict avoidance of cheese, milk-based sauces, and butter, and previous events that overlooked this caused her noticeable discomfort and embarrassment in front of teammates.",
    ),
    (
        "banq_g_halal",
        "Two team members, Yusuf Khan and Aaliyah Said, observe halal dietary requirements as practicing Muslims, and previous team events have accommodated this with a separate halal-certified main course supplied by a local catering partner.",
    ),
    (
        "banq_g_lowsodium",
        "Coach Hernandez has a documented heart condition and his cardiologist requires him to follow a low-sodium diet, which is something past banquets have respected by including unsalted bread and a low-sodium entree option.",
    ),
    (
        "banq_g_dry_county",
        "The school district lies in a legally dry county where alcoholic beverages are banned at all school-sponsored events without exception, and any beverage list must be alcohol-free including non-alcoholic substitutes for any traditional toasts.",
    ),
    (
        "banq_g_seniors",
        "The graduating senior class on the track team this year includes Lina Park (4 years), Patricia 'Pat' Reyes (4 years), and Ouma Diallo (3 years), all of whom have qualified for state at least once during their high school careers.",
    ),
    (
        "banq_g_venue",
        "The school cafeteria is the only on-campus venue with sufficient capacity (seats up to 240) and is available free of charge to recognized clubs on weekday evenings between 6pm and 9pm.",
    ),
    (
        "banq_g_caterer_policy",
        "School district policy 12.4 requires that any food served at school events come from either an in-house cafeteria preparation or a licensed external caterer registered with the district; outside potluck contributions are not permitted.",
    ),
    (
        "banq_g_budget",
        "The track program's banquet budget for this year was set at $1,400 based on prior years' attendance and donations, including any catering, decoration, and recognition gifts for graduating seniors.",
    ),
    (
        "banq_g_award_overrun",
        "Last year the awards portion of the banquet ran 35 minutes longer than scheduled because each senior was given an unscripted speech, leading to families leaving early and several recognition moments missed.",
    ),
]

BANQUET_SUBQ = [
    "menu (must respect peanut allergy, halal, lactose, low-sodium)",
    "venue/setup (cafeteria available; licensed-caterer rule; budget)",
    "drinks (no alcohol — dry county)",
    "senior recognition (3 named seniors; manage award duration)",
]

# --------------------- Case 2: client pitch ---------------------

PITCH_FILLER = [
    "Acme Foods's website prominently features their new line of organic cereals and a podcast hosted by their head of marketing on supply-chain sustainability.",
    "Our company recently moved offices to a building with rooftop access, frequently used for after-hours team gatherings during summer.",
    "The most recent industry conference featured a panel on B2B SaaS pricing that included two of our competitors and one of our existing customers.",
    "Many sales calls in our team's pipeline use the same standard slide template that includes our company history and a customer logo wall.",
    "Our company's quarterly all-hands meeting is held the first Thursday of each quarter at 10am Pacific via Zoom, with breakouts by department.",
    "The marketing team recently produced a one-page brand-positioning document that outlines our value proposition for mid-market buyers.",
    "Acme's headquarters are in suburban Chicago, in a renovated warehouse with an open floor plan and a small coffee bar in the lobby.",
    "Our intern wrote up notes from a previous sales call with a different food-industry client; the notes mention an interest in API integrations.",
    "The conference room near the kitchen has a slightly squeaky chair that the office manager has been meaning to replace for months.",
    "An older version of our pitch deck included a slide with industry trends from 2020 that has since been removed for being outdated.",
    "Acme has a careers page advertising openings for a senior data engineer and a procurement manager based in their Chicago headquarters.",
    "Some of our customers run weekly office hours where their internal teams can ask questions about the platform; engagement varies by team.",
    "The kitchen on our floor stocks three types of tea: English Breakfast, peppermint, and a green tea that no one ever drinks.",
    "The customer success team uses a separate Slack workspace from sales, which occasionally causes context to drop between handoffs.",
    "Acme's annual report mentions a recent acquisition of a regional dairy distributor and their plans to consolidate logistics over the next 18 months.",
    "Our internal style guide allows two heading typefaces and discourages the use of italics in body copy on customer-facing slides.",
    "Mid-market food companies have been a growing segment for our sales team, with closure rates near 24% over the last four quarters.",
    "The receptionist keeps a list of frequent visitors and their preferred drink (coffee, water, sparkling water) as a small hospitality touch.",
    "Our CEO sometimes joins the back half of important pitch meetings if availability permits, especially for accounts above $500k ARR.",
    "The pitch team's shared Google Drive folder has 14 different versions of last quarter's deck, only one of which is current.",
    "Last year a similar pitch to a beverage company included a competitive comparison slide that took longer than expected to walk through.",
    "Our ROI calculator is a simple spreadsheet maintained by the solutions engineering team; it has a clean output page suitable for live demos.",
    "Acme's social media presence is strongest on LinkedIn, with regular posts about supply-chain technology and a quarterly company newsletter.",
    "An old internal wiki page describes the etiquette for client lunches, including suggested restaurants in the Chicago downtown area.",
    "Our sales operations team tracks pipeline aging in Salesforce, with custom dashboards reviewed each Monday morning by the VP of sales.",
    "The conference room's projector model is a Sony VPL-VW295ES, mounted to the ceiling, with a remote that frequently goes missing.",
    "Some clients prefer to share screens via the in-room Polycom system; setup typically takes 5-10 minutes depending on hardware.",
    "Acme's investor day is scheduled for next month and is expected to highlight their digital transformation initiative across their facilities.",
    "Our sales kickoff each January features a guest speaker, and last year's was a former Chief Revenue Officer of a major SaaS company.",
    "The intern took notes from a vendor's webinar on consumer-packaged-goods buying signals; the notes are filed in the shared drive but not indexed.",
    "Acme's CFO publicly praised a competitor's product on a podcast last year, though the deal ultimately did not move forward.",
    "Our office printer on the third floor jams about once a week, despite multiple service calls and a recent toner replacement.",
    "There is a recurring weekly sales standup at 9 AM Tuesday for any rep working active enterprise opportunities.",
    "Acme's mission statement emphasizes 'wholesome ingredients, family-owned roots, and modern operational excellence' across all their product lines.",
    "Our customer reference library includes three case studies in the food and beverage space, two of which are publicly published on our website.",
    "The IT department supplied loaner laptops for sales reps who travel internationally; checkout requires two weeks notice and a justification.",
    "Recent product release notes include improvements to our reporting module and a beta feature for forecasting that some customers have requested.",
]

PITCH_GOLD = [
    (
        "pitch_g_brand_palette",
        "Acme Foods's official brand colors are deep red (#9E1B32) and warm cream (#F5EFE0); their public style guide explicitly forbids vendor decks from using clashing palettes such as orange or navy when presenting to their executives.",
    ),
    (
        "pitch_g_our_palette_clash",
        "Our standard pitch deck template is built on a navy and orange color system, which directly clashes with Acme's deep red / cream palette and would be visually jarring if used unchanged for the Acme pitch.",
    ),
    (
        "pitch_g_cto_pref_live",
        "Acme's CTO Pia Rao previously stated in a pre-meeting email that she dislikes long pre-recorded video demos and strongly prefers short live walk-throughs with the ability to ask interactive questions.",
    ),
    (
        "pitch_g_ceo_captions",
        "Acme's CEO Dawit Mehari has a hearing impairment and has explicitly requested that any pre-recorded video material in vendor presentations include accurate burned-in or live captions.",
    ),
    (
        "pitch_g_hdmi",
        "The conference room reserved for the meeting at our office has a 4K projector that ONLY accepts HDMI input; presenters whose laptops have only USB-C must bring an HDMI adapter or the demo cannot be projected.",
    ),
    (
        "pitch_g_handout",
        "Acme's procurement team has standardized on receiving a single-page printed executive summary handout at every vendor meeting; arriving without one signals lack of preparation per their published vendor guidelines.",
    ),
    (
        "pitch_g_roi_winners",
        "Our last three closed-won deals in the food-and-beverage segment all included a live walk-through of the ROI calculator demo, and post-deal interviews confirmed it was the most influential moment of the pitch for buyers.",
    ),
    (
        "pitch_g_meeting_length",
        "Acme's procurement department enforces a strict 20-minute hard cap on vendor pitch meetings; presenters who run over have been formally warned that further violations would disqualify them from the procurement pool.",
    ),
]

PITCH_SUBQ = [
    "content focus (ROI walkthrough; respect 20 min cap)",
    "visual design (Acme palette; replace clashing template; accessibility/captions)",
    "demo selection (live not pre-recorded; captioned if any)",
    "logistics (HDMI adapter; one-page handout)",
]

# --------------------- Case 3: camping trip ---------------------

CAMPING_FILLER = [
    "Greenlake park is most popular in late spring and early autumn, when wildflowers are abundant and the mosquito population is significantly lower than midsummer.",
    "Many campers at Greenlake bring inflatable kayaks for the upper pond, which is calm and well suited to recreational paddling on most weekends.",
    "The local outdoor store sells a popular brand of camp chair that folds into a small carrying case and is consistently rated four stars on customer reviews.",
    "A nearby diner near the park entrance is known for serving pancakes the size of dinner plates and a bottomless cup of coffee for $3.75.",
    "The kids enjoy roasting marshmallows over a fire when permitted, and the older child has gotten quite good at not letting the marshmallow catch fire.",
    "Last summer the family drove to a different park, two hours west, and the kids said the trip was the highlight of their summer break.",
    "The neighbor's dog Buster sometimes joins family hikes, although he tires out faster than the kids and needs to be carried back the last mile.",
    "Greenlake's main lake has a swimming area marked off with buoys, but the water is cold even in late summer and most visitors only wade in briefly.",
    "Many trails at Greenlake are well marked with colored blazes, although a few connector trails have signs that have been weathered by years of harsh winters.",
    "The campground has a bulletin board near the ranger entrance that lists daily wildlife sightings, recent trail closures, and a weather forecast updated each morning.",
    "Mom enjoys photographing wildflowers and has a small field guide she takes on every camping trip, identifying plants by leaf shape and habitat.",
    "The 7-year-old has been working on a sticker collection at school and earns a sticker every time she helps set up the tent without complaining.",
    "Dad's favorite camping recipe is a foil-pack meal of potatoes, sausage, and onions, which has been a family tradition for many summers.",
    "The minivan has a roof rack that can hold a cargo box, which the family uses for longer trips when the back of the car is full of gear.",
    "Greenlake's eastern trailhead has a small parking lot that fills up by 9 AM on summer weekends, especially when the weather is good.",
    "A wooden sign at the campground entrance reads 'Welcome to Greenlake — please pack out what you pack in', a request that most campers respect.",
    "Both parents enjoy reading at night by lantern, with mom currently working through a 600-page novel and dad re-reading a thriller from years ago.",
    "The 5-year-old recently started learning the names of constellations, with mom teaching him to find Orion's belt and the Big Dipper after dinner.",
    "Camping merit badges are popular among local Scout troops, several of which use Greenlake as their primary outing destination each spring.",
    "The campground's bathrooms include cold-water showers; the family has learned to bring a solar shower bag for warm rinses on multi-day trips.",
    "A nearby town hosts a small farmers' market on Saturday mornings, with fresh berries and homemade jams that the family enjoys picking up before heading to the park.",
    "The kids' bicycles are too small for the 12-mile loop trail at Greenlake, so the family typically does shorter hikes on bike-friendly access roads.",
    "Mom keeps a packing list in a shared note that she revises after every trip based on what was missing or unused last time.",
    "Dad has a rugged old hatchet his father gave him, which lives in a leather sheath in the camping bin and gets used for splitting kindling.",
    "The family has a rule that screens stay home when camping, although the parents' phones come along for emergency use only.",
    "Greenlake's pet policy allows leashed dogs in some sections of the campground but prohibits them on backcountry trails to protect wildlife.",
    "A local company organizes guided naturalist hikes at Greenlake on summer weekends, with topics ranging from wildflower identification to bird calls.",
    "The minivan gets about 26 mpg highway, which makes the round trip to Greenlake roughly $30 in gas at current prices.",
    "Both kids have small headlamps that they use for after-dark trips to the bathroom, although the 5-year-old still prefers to be accompanied.",
    "The family's tent is a six-person dome model, lightly used, which they bought on sale three years ago at a regional outfitter.",
    "Mom has a favorite thermos that she insists on bringing despite its weight, because she says nothing else keeps coffee hot for as many hours.",
    "The grandparents recently sent a photo album of past camping trips dating back 15 years, which the kids enjoy flipping through on car rides.",
    "A regional outdoor magazine recently featured Greenlake in a list of family-friendly campgrounds, citing its variety of trails and campground amenities.",
    "Dad's old hiking boots have been re-soled twice and might be due for a third resoling before the next major trip is planned.",
    "The kids have been earning a small allowance for chores, which the older one has been saving toward a small portable bluetooth speaker for camping.",
    "Both kids enjoy birdwatching from the picnic table, especially trying to identify woodpeckers and various species of jays in the surrounding pines.",
    "The minivan has a CD player that still works, and mom keeps a few audiobook CDs in the glove compartment for long drives.",
]

CAMPING_GOLD = [
    (
        "camp_g_epipen_kid",
        "The 5-year-old has a confirmed severe bee-sting allergy and must travel at all times with a current EpiPen kit (currently expires next month so a replacement should be picked up before departure).",
    ),
    (
        "camp_g_forecast_cold_rain",
        "The 10-day forecast for Greenlake next weekend shows nighttime lows of 35F (about 2C) and a 60% chance of rain on Saturday, much colder than the family is used to for spring camping.",
    ),
    (
        "camp_g_spare_tire",
        "The family minivan's spare tire was used on a separate trip last month and has not been replaced; the vehicle currently has no functional spare, which is a significant risk on a remote-access trip.",
    ),
    (
        "camp_g_no_fire_rule",
        "Greenlake park has issued a hard ban on open campfires for the rest of the season due to extreme drought conditions; rangers are actively ticketing violators with a $300 fine.",
    ),
    (
        "camp_g_pregnant_mom",
        "Mom is six months pregnant and her obstetrician has explicitly advised against heavy lifting (anything above ~25 lbs) and against long hikes with significant downhill, which limits the gear-setup work she can do.",
    ),
    (
        "camp_g_water_filter",
        "The family's portable water filter is rated for 40 liters of total lifetime use and has already been used for approximately 30 liters; remaining capacity is therefore only about 10 liters, well short of a 4-person 2-night need.",
    ),
    (
        "camp_g_no_cell_8mi_ranger",
        "Greenlake's main campground has no cell phone coverage on any major US carrier and the nearest ranger station with a phone is 8 miles away by gravel road, which materially affects emergency planning.",
    ),
    (
        "camp_g_dad_knee",
        "Dad has a chronic 'trick knee' that flares badly during long downhill descents; on the last camping trip he had to skip the second-day hike because of swelling, which is relevant when picking trails.",
    ),
    (
        "camp_g_kid_dog_fear",
        "The 7-year-old has a strong fear of large unleashed dogs after being knocked down at a park last summer; campgrounds with off-leash sections cause her significant anxiety and disrupted sleep.",
    ),
]

CAMPING_SUBQ = [
    "location (cold/rain forecast; no-fire rule; remoteness/no-cell)",
    "food/water (filter remaining capacity; cooking without fire)",
    "gear (warm sleeping; spare tire; pregnant mom limits)",
    "safety/emergency (EpiPen; no cell service; trail choice for dad's knee)",
]


def build_cases() -> list[Case]:
    return [
        _build_case(
            case_id="banquet",
            domain="event_planning",
            task=(
                "Plan and prepare a season-end banquet for the high school track team. You must "
                "decide the menu, beverages, venue setup, recognition for graduating seniors, and "
                "any logistical compliance with school policy. Produce a plan that respects every "
                "real constraint that exists in the team's situation."
            ),
            gold=BANQUET_GOLD,
            domain_filler=BANQUET_FILLER,
            gold_subquestions=BANQUET_SUBQ,
            seed=11,
        ),
        _build_case(
            case_id="acme_pitch",
            domain="b2b_sales",
            task=(
                "Prepare a 20-minute pitch presentation for the prospective enterprise client Acme "
                "Foods. You must decide content focus, visual design, demo selection, accessibility, "
                "and meeting logistics. Produce a plan that respects every real constraint that "
                "exists in the situation."
            ),
            gold=PITCH_GOLD,
            domain_filler=PITCH_FILLER,
            gold_subquestions=PITCH_SUBQ,
            seed=23,
        ),
        _build_case(
            case_id="camping",
            domain="family_planning",
            task=(
                "Plan a 2-night weekend camping trip for the family of four (two adults, two kids). "
                "You must decide location, food and water, gear, and safety/emergency setup. "
                "Produce a plan that respects every real constraint that exists in the family's "
                "situation."
            ),
            gold=CAMPING_GOLD,
            domain_filler=CAMPING_FILLER,
            gold_subquestions=CAMPING_SUBQ,
            seed=37,
        ),
    ]


# ============================================================================
# Embedding-backed external memory
# ============================================================================


@dataclass
class ExternalMemory:
    case: Case
    embeddings: np.ndarray  # shape (N, D)
    fact_ids: list[str]
    fact_lookup: dict[str, MemEntry]

    def total_tokens(self) -> int:
        return sum(_toks(e.text) for e in self.case.entries)

    async def retrieve(self, query: str, k: int = TOP_K) -> list[MemEntry]:
        emb = await _embed_one(query)
        # cosine similarity
        sims = (
            self.embeddings
            @ emb
            / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(emb) + 1e-9)
        )
        top_idx = np.argsort(-sims)[:k]
        return [self.fact_lookup[self.fact_ids[i]] for i in top_idx]


async def _embed_one(text: str) -> np.ndarray:
    resp = await client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype=np.float32)


async def _embed_many(texts: list[str]) -> np.ndarray:
    out: list[list[float]] = []
    BATCH = 128
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp = await client.embeddings.create(model=EMBED_MODEL, input=batch)
        # API returns in input order
        out.extend(d.embedding for d in resp.data)
    return np.array(out, dtype=np.float32)


async def build_external_memory(case: Case) -> ExternalMemory:
    fact_ids = [e.fact_id for e in case.entries]
    texts = [e.text for e in case.entries]
    embs = await _embed_many(texts)
    lookup = {e.fact_id: e for e in case.entries}
    return ExternalMemory(
        case=case, embeddings=embs, fact_ids=fact_ids, fact_lookup=lookup
    )


# ============================================================================
# Working memory (bounded, with compaction)
# ============================================================================


@dataclass
class WMEntry:
    fact_id: str  # may be "" for compacted-summary entries
    text: str
    is_summary: bool = False  # True after compaction


@dataclass
class WorkingMemory:
    entries: list[WMEntry] = field(default_factory=list)
    summary: str = ""  # accumulated GATHERED_NOTES from compaction

    def token_count(self) -> int:
        rendered = self.render_for_prompt()
        return _toks(rendered)

    def render_for_prompt(self) -> str:
        parts = []
        if self.summary:
            parts.append("GATHERED_NOTES (compacted from earlier rounds):")
            parts.append(self.summary)
            parts.append("")
        if self.entries:
            parts.append("RECENT_RETRIEVALS:")
            for e in self.entries:
                parts.append(f"  [{e.fact_id}] {e.text}")
        if not parts:
            return "(working memory empty)"
        return "\n".join(parts)

    def fact_ids(self) -> set[str]:
        return {e.fact_id for e in self.entries if e.fact_id}

    def add(self, entries: list[WMEntry]) -> None:
        # Dedup by fact_id; keep last seen
        seen = {e.fact_id for e in self.entries if e.fact_id}
        for e in entries:
            if e.fact_id and e.fact_id in seen:
                continue
            self.entries.append(e)
            seen.add(e.fact_id)

    def all_fact_ids_ever(self) -> set[str]:
        # For metric purposes — track ids ever seen including compacted
        return self.fact_ids() | self._compacted_ids

    _compacted_ids: set[str] = field(default_factory=set)


COMPACT_SYSTEM = """You are the working-memory compactor for a multi-step task agent.

You will be given a TASK and a list of recent retrieval snippets. Your job is to compress them
into a tight, structured GATHERED_NOTES block that preserves load-bearing details (numbers, names,
constraints, hard rules, dates) but drops verbose framing. Aim for under 300 tokens. Use bullet
points keyed by fact_id so downstream reasoning can still cite the source. If a snippet is
clearly off-task, omit it.

Output exactly the compressed notes, no preamble. Format:
- [fact_id] one tight line preserving the load-bearing claim
"""


async def compact_wm(task: str, wm: WorkingMemory) -> None:
    """Compact older WM entries into the running summary, keeping recent retrievals raw."""
    if not wm.entries:
        return
    # Keep the most recent ~K snippets raw; compact the rest.
    if len(wm.entries) <= 4:
        return
    to_compact = wm.entries[:-4]
    keep = wm.entries[-4:]
    snippets = "\n".join(f"[{e.fact_id}] {e.text}" for e in to_compact)
    user = f"TASK: {task}\n\nRECENT RETRIEVAL SNIPPETS:\n{snippets}\n\nCompress."
    resp = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": COMPACT_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    notes = (resp.choices[0].message.content or "").strip()
    if wm.summary:
        wm.summary = wm.summary + "\n" + notes
    else:
        wm.summary = notes
    # Update compacted-id tracking
    wm._compacted_ids |= {e.fact_id for e in to_compact if e.fact_id}
    wm.entries = keep


async def maybe_compact(task: str, wm: WorkingMemory, log: list[dict]) -> None:
    before = wm.token_count()
    if before <= WM_COMPACT_TRIGGER:
        return
    await compact_wm(task, wm)
    after = wm.token_count()
    log.append({"event": "compact", "before_tokens": before, "after_tokens": after})


# ============================================================================
# Probe + cue gen prompts
# ============================================================================


CUE_GEN_SYSTEM = """You are the planning module of a multi-step task agent with a small bounded
working memory and a much larger external memory you can probe by emitting a natural-language
QUERY string. The probe will return the top-K most similar memory entries.

Given the TASK and your current bounded WORKING MEMORY (which may be compacted), emit ONE next
QUERY that probes a region of external memory likely to surface NOT-YET-GATHERED information
relevant to the task. Keep it focused (under 25 words). Do NOT restate the whole task; do NOT
enumerate everything; pick a single concrete angle.

Output (strict):
QUERY: <one short natural-language query>
"""

CUE_GEN_USER = """TASK:
{task}

WORKING MEMORY:
{wm}

Emit your next QUERY (focus on something the working memory does not already cover)."""

_QUERY_LINE_RE = re.compile(r"QUERY:\s*(.+)", re.IGNORECASE)


def _parse_query(text: str) -> str:
    m = _QUERY_LINE_RE.search(text)
    if m:
        return m.group(1).strip().splitlines()[0]
    return text.strip().splitlines()[0][:200]


# ============================================================================
# Completeness gate
# ============================================================================


GATE_SYSTEM = """You are a metacognitive completeness gate sitting on top of a memory-retrieval
loop in a multi-step task agent.

WHY YOU EXIST
A retrieval loop has two natural termination signals:
  - SATURATION: "no new content surfaced this round." This is a CONCEPT-DENSITY signal — the
    region of memory we've been probing has gone dry.
  - GOAL-COMPLETENESS: "given the task, are there concrete sub-questions whose answers would
    change the plan, and do we have answers to all of them?" This is a GOAL-AWARE, ANTICIPATORY
    signal — the human felt-sense-of-incompleteness BEFORE you've consciously noticed what's
    missing. ("I haven't checked the venue's policy on outside food yet.")

Saturation alone is brittle: the agent can stop because the local neighborhood is exhausted while
critical sub-questions remain unaddressed. Your job is to enforce GOAL-completeness instead.

HOW TO REASON
1. DECOMPOSE the task into a small set of sub-questions any sound plan MUST resolve. Be honest
   and exhaustive — but task-relevant only, not exhaustive in the abstract. Each sub-question
   should be a thing that, if unanswered, would change the plan.
2. For each sub-question, check the WORKING MEMORY. Distinguish between "we have a relevant fact
   that meaningfully addresses this" vs "we have an empty gap, just guesses, or only tangential
   info."
3. If gaps exist, emit them as OPEN: lines (one per gap, phrased as a probe-able question), then
   emit ONE NEXT_QUERY: the single best natural-language probe to issue next. Do not pick
   an open gap that the prior round just probed unsuccessfully — vary your angle.
4. If no gaps remain — the task is genuinely covered enough to plan against — emit COMPLETE
   with a one-sentence reason.

KEEP OUTPUT UNDER 300 TOKENS.

OUTPUT FORMAT (strict)
Either:
  OPEN: <sub-question 1>
  OPEN: <sub-question 2>
  ...
  NEXT_QUERY: <one short natural-language query for the next probe>
Or:
  COMPLETE
  REASON: <one short sentence>
"""

GATE_USER = """TASK:
{task}

WORKING MEMORY:
{wm}

ROUNDS USED: {rounds}; ROUNDS REMAINING (cap): {remaining}

Decompose the task into goal-relevant sub-questions. Identify which are unanswered. Emit OPEN +
NEXT_QUERY, or COMPLETE."""

_OPEN_RE = re.compile(r"^OPEN:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_NEXT_RE = re.compile(r"NEXT_QUERY:\s*(.+)", re.IGNORECASE)
_COMPLETE_RE = re.compile(r"^\s*COMPLETE\b", re.IGNORECASE | re.MULTILINE)


@dataclass
class GateOutput:
    complete: bool
    open_gaps: list[str]
    next_query: str | None
    raw: str


def _parse_gate(text: str) -> GateOutput:
    if _COMPLETE_RE.search(text):
        return GateOutput(complete=True, open_gaps=[], next_query=None, raw=text)
    gaps = [m.group(1).strip() for m in _OPEN_RE.finditer(text)]
    nq_match = _NEXT_RE.search(text)
    nq = nq_match.group(1).strip() if nq_match else None
    return GateOutput(complete=False, open_gaps=gaps, next_query=nq, raw=text)


# ============================================================================
# LLM helpers
# ============================================================================


async def _llm(messages: list[dict]) -> str:
    resp = await client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return resp.choices[0].message.content or ""


# ============================================================================
# Run loops
# ============================================================================


@dataclass
class RunResult:
    case_id: str
    variant: str
    rounds: int
    surfaced: list[str]
    gold: list[str]
    queries: list[str]
    terminated_by: str
    token_trace: list[dict]
    max_wm_tokens: int

    @property
    def recall(self) -> float:
        if not self.gold:
            return 0.0
        return len(set(self.surfaced) & set(self.gold)) / len(self.gold)


async def run_baseline(case: Case, ext: ExternalMemory) -> RunResult:
    wm = WorkingMemory()
    queries: list[str] = []
    token_trace: list[dict] = []
    surfaced_total: set[str] = set()
    terminated_by = "max_rounds"
    max_wm = 0

    for r in range(MAX_BASELINE_ROUNDS):
        # Cue generation
        cue_resp = await _llm(
            [
                {"role": "system", "content": CUE_GEN_SYSTEM},
                {
                    "role": "user",
                    "content": CUE_GEN_USER.format(
                        task=case.task, wm=wm.render_for_prompt()
                    ),
                },
            ]
        )
        query = _parse_query(cue_resp)
        queries.append(query)

        # External retrieval
        hits = await ext.retrieve(query, k=TOP_K)
        before_ids = surfaced_total | wm._compacted_ids
        new_entries = [WMEntry(fact_id=h.fact_id, text=h.text) for h in hits]
        wm.add(new_entries)
        round_new = {h.fact_id for h in hits} - before_ids
        surfaced_total |= {h.fact_id for h in hits}

        wm_tok = wm.token_count()
        max_wm = max(max_wm, wm_tok)
        token_trace.append(
            {
                "round": r + 1,
                "variant": "baseline",
                "wm_tokens_pre_compact": wm_tok,
                "new_in_round": sorted(round_new),
                "query": query,
            }
        )

        # Compact if over threshold
        await maybe_compact(case.task, wm, token_trace)
        token_trace[-1]["wm_tokens_post_compact"] = wm.token_count()

        # Saturation check (after first round)
        if r > 0 and not round_new:
            terminated_by = "saturation"
            break

    return RunResult(
        case_id=case.case_id,
        variant="baseline",
        rounds=len(queries),
        surfaced=sorted(surfaced_total | wm._compacted_ids),
        gold=case.gold_fact_ids,
        queries=queries,
        terminated_by=terminated_by,
        token_trace=token_trace,
        max_wm_tokens=max_wm,
    )


async def run_operator(case: Case, ext: ExternalMemory) -> RunResult:
    wm = WorkingMemory()
    queries: list[str] = []
    token_trace: list[dict] = []
    surfaced_total: set[str] = set()
    terminated_by = "max_rounds"
    max_wm = 0
    next_query_override: str | None = None

    for r in range(MAX_OPERATOR_ROUNDS):
        if next_query_override:
            query = next_query_override
            next_query_override = None
        else:
            cue_resp = await _llm(
                [
                    {"role": "system", "content": CUE_GEN_SYSTEM},
                    {
                        "role": "user",
                        "content": CUE_GEN_USER.format(
                            task=case.task, wm=wm.render_for_prompt()
                        ),
                    },
                ]
            )
            query = _parse_query(cue_resp)
        queries.append(query)

        hits = await ext.retrieve(query, k=TOP_K)
        new_entries = [WMEntry(fact_id=h.fact_id, text=h.text) for h in hits]
        wm.add(new_entries)
        surfaced_total |= {h.fact_id for h in hits}

        wm_tok = wm.token_count()
        max_wm = max(max_wm, wm_tok)
        token_trace.append(
            {
                "round": r + 1,
                "variant": "operator",
                "wm_tokens_pre_compact": wm_tok,
                "query": query,
            }
        )

        await maybe_compact(case.task, wm, token_trace)
        token_trace[-1]["wm_tokens_post_compact"] = wm.token_count()

        # Gate
        gate_resp = await _llm(
            [
                {"role": "system", "content": GATE_SYSTEM},
                {
                    "role": "user",
                    "content": GATE_USER.format(
                        task=case.task,
                        wm=wm.render_for_prompt(),
                        rounds=r + 1,
                        remaining=MAX_OPERATOR_ROUNDS - (r + 1),
                    ),
                },
            ]
        )
        gate = _parse_gate(gate_resp)
        token_trace[-1]["gate_complete"] = gate.complete
        token_trace[-1]["gate_open_count"] = len(gate.open_gaps)
        if gate.complete:
            terminated_by = "complete"
            break
        next_query_override = gate.next_query

    return RunResult(
        case_id=case.case_id,
        variant="operator",
        rounds=len(queries),
        surfaced=sorted(surfaced_total | wm._compacted_ids),
        gold=case.gold_fact_ids,
        queries=queries,
        terminated_by=terminated_by,
        token_trace=token_trace,
        max_wm_tokens=max_wm,
    )


# ============================================================================
# Main
# ============================================================================


async def main() -> None:
    cases = build_cases()
    print(f"Building external memory for {len(cases)} cases...")
    cases_with_mem = []
    for c in cases:
        ext = await build_external_memory(c)
        toks = ext.total_tokens()
        print(
            f"  [{c.case_id}] {len(c.entries)} entries, {toks} total tokens, "
            f"{len(c.gold_fact_ids)} gold"
        )
        cases_with_mem.append((c, ext))

    print(f"\nRunning baseline + operator on {CHAT_MODEL}...\n")
    all_results: list[RunResult] = []
    for c, ext in cases_with_mem:
        b = await run_baseline(c, ext)
        o = await run_operator(c, ext)
        all_results.extend([b, o])
        print(
            f"[{c.case_id}] BASELINE rounds={b.rounds} term={b.terminated_by} "
            f"recall={b.recall:.2f} ({len(set(b.surfaced) & set(c.gold_fact_ids))}/{len(c.gold_fact_ids)}) "
            f"max_wm={b.max_wm_tokens} | "
            f"OPERATOR rounds={o.rounds} term={o.terminated_by} "
            f"recall={o.recall:.2f} ({len(set(o.surfaced) & set(c.gold_fact_ids))}/{len(c.gold_fact_ids)}) "
            f"max_wm={o.max_wm_tokens}"
        )

    # Aggregate
    print("\n=== Per-case table ===")
    print(
        f"{'case':<12} {'b_rnds':>7} {'b_recall':>9} {'b_term':>11} {'b_maxWM':>8}  "
        f"{'o_rnds':>7} {'o_recall':>9} {'o_term':>11} {'o_maxWM':>8}  {'delta':>6}"
    )
    by_case: dict[str, dict[str, RunResult]] = {}
    for r in all_results:
        by_case.setdefault(r.case_id, {})[r.variant] = r

    bsum = osum = 0.0
    for cid, d in by_case.items():
        b, o = d["baseline"], d["operator"]
        bsum += b.recall
        osum += o.recall
        print(
            f"{cid:<12} {b.rounds:>7} {b.recall:>9.2f} {b.terminated_by:>11} {b.max_wm_tokens:>8}  "
            f"{o.rounds:>7} {o.recall:>9.2f} {o.terminated_by:>11} {o.max_wm_tokens:>8}  "
            f"{(o.recall - b.recall):>+6.2f}"
        )
    n = len(by_case)
    print(
        f"{'MEAN':<12} {'':>7} {bsum / n:>9.2f} {'':>11} {'':>8}  "
        f"{'':>7} {osum / n:>9.2f} {'':>11} {'':>8}  {(osum - bsum) / n:>+6.2f}"
    )

    # Write detailed results + token traces
    out_path = THIS_DIR / "results.json"
    payload = {
        "model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "wm_token_cap": WM_TOKEN_CAP,
        "wm_compact_trigger": WM_COMPACT_TRIGGER,
        "max_baseline_rounds": MAX_BASELINE_ROUNDS,
        "max_operator_rounds": MAX_OPERATOR_ROUNDS,
        "top_k": TOP_K,
        "cases": [
            {
                "case_id": c.case_id,
                "domain": c.domain,
                "n_entries": len(c.entries),
                "total_external_tokens": ext.total_tokens(),
                "gold_fact_ids": c.gold_fact_ids,
                "gold_subquestions": c.gold_subquestions,
            }
            for c, ext in cases_with_mem
        ],
        "results": [
            {
                "case_id": r.case_id,
                "variant": r.variant,
                "rounds": r.rounds,
                "terminated_by": r.terminated_by,
                "recall": r.recall,
                "surfaced": r.surfaced,
                "gold": r.gold,
                "queries": r.queries,
                "max_wm_tokens": r.max_wm_tokens,
                "token_trace": r.token_trace,
            }
            for r in all_results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")

    # Per-case token traces
    for r in all_results:
        trace_path = THIS_DIR / f"trace_{r.case_id}_{r.variant}.json"
        trace_path.write_text(json.dumps(r.token_trace, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
