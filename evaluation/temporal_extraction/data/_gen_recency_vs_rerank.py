"""Generate recency_vs_rerank synthetic bench.

Design:
- For each scenario, gold doc text is paraphrased/elliptical (low surface overlap with query)
- Decoy doc texts repeat the query's key terms verbatim (high surface overlap)
- Gold is most recent (70%) or earliest (30%) among same-topic docs
- All same-topic docs describe the same activity but with different wording
- No month names or day-of-month numbers in doc text
"""

import json
import os
import random
from datetime import datetime, timedelta, timezone

random.seed(42)

OUT_DIR = "/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/temporal_extraction/data"

REF_TIME = "2026-04-01T00:00:00Z"

# Disadvantage classes per scenario:
#   max     -> gold text shares almost no keyword tokens with query
#   partial -> gold uses partial paraphrase (some terms replaced)
#   minor   -> gold uses minor wording shifts (adj order, light synonym)

# Each scenario:
#   topic (slug), query text, gold text, list of decoy texts (7-9),
#   most_recent (True) or first (False), disadvantage class
SCENARIOS = [
    # ---------- MAX disadvantage (5 scenarios): gold has no overlap with query keywords ----------
    {
        "topic": "morningrun",
        "query": "What was my most recent morning run along the river path?",
        "gold_text": "Ran by the water.",
        "decoys": [
            "Did a morning run along the river path.",
            "Did a morning run along the river path before work.",
            "Morning run along the river path with light pace.",
            "Went on a morning run along the river path early.",
            "Took a morning run along the river path with friends.",
            "Did a slow morning run along the river path.",
            "Logged a morning run along the river path after stretching.",
            "Morning run along the river path solo.",
        ],
        "most_recent": True,
        "disadvantage": "max",
    },
    {
        "topic": "eveningwalk",
        "query": "When was my most recent evening walk through the suburb?",
        "gold_text": "Strolled around the block after dark.",
        "decoys": [
            "Took an evening walk through the suburb.",
            "Did an evening walk through the suburb after dinner.",
            "Evening walk through the suburb with the dog.",
            "Went on an evening walk through the suburb.",
            "Long evening walk through the suburb at dusk.",
            "Brisk evening walk through the suburb under streetlights.",
            "Evening walk through the suburb listening to a podcast.",
            "Quiet evening walk through the suburb.",
        ],
        "most_recent": True,
        "disadvantage": "max",
    },
    {
        "topic": "swimlaps",
        "query": "What was my most recent set of swimming laps at the rec pool?",
        "gold_text": "Knocked out a workout at the aquatics center.",
        "decoys": [
            "Swam laps at the rec pool for thirty minutes.",
            "Did swimming laps at the rec pool in the morning.",
            "Swimming laps at the rec pool with the masters group.",
            "Logged swimming laps at the rec pool after work.",
            "Hit the rec pool for swimming laps.",
            "Easy swimming laps at the rec pool to loosen up.",
            "Swimming laps at the rec pool with a kickboard.",
            "Quick swimming laps at the rec pool before closing.",
        ],
        "most_recent": True,
        "disadvantage": "max",
    },
    {
        "topic": "garagesale",
        "query": "When was my most recent neighborhood garage sale browsing trip?",
        "gold_text": "Wandered around looking for old stuff people were getting rid of.",
        "decoys": [
            "Went neighborhood garage sale browsing on the weekend.",
            "Did some neighborhood garage sale browsing with my sister.",
            "Neighborhood garage sale browsing in the next block over.",
            "Light neighborhood garage sale browsing without buying anything.",
            "Long morning of neighborhood garage sale browsing.",
            "Neighborhood garage sale browsing in the rain briefly.",
            "Slow neighborhood garage sale browsing while sipping coffee.",
            "Productive neighborhood garage sale browsing — found a lamp.",
        ],
        "most_recent": True,
        "disadvantage": "max",
    },
    {
        "topic": "calisthenics",
        "query": "When was the first calisthenics workout I did at the park?",
        "gold_text": "Tried some bodyweight exercises outdoors near the playground.",
        "decoys": [
            "Did a calisthenics workout at the park near the playground.",
            "Full calisthenics workout at the park with pull-up bars.",
            "Light calisthenics workout at the park to test the equipment.",
            "Group calisthenics workout at the park with a meetup.",
            "Solo calisthenics workout at the park before sunset.",
            "Calisthenics workout at the park focused on push variations.",
            "Calisthenics workout at the park with a friend coaching.",
            "Long calisthenics workout at the park ending with stretching.",
        ],
        "most_recent": False,
        "disadvantage": "max",
    },

    # ---------- PARTIAL disadvantage (10 scenarios): gold uses partial paraphrase ----------
    {
        "topic": "treadmill",
        "query": "What was my most recent 30-minute treadmill session?",
        "gold_text": "Half-hour on the running machine.",
        "decoys": [
            "Did a 30-minute treadmill session at the gym.",
            "30-minute treadmill session at moderate pace.",
            "Quick 30-minute treadmill session before lifting.",
            "Easy 30-minute treadmill session as a warm-up.",
            "Hot 30-minute treadmill session on incline.",
            "Solid 30-minute treadmill session in the morning.",
            "30-minute treadmill session listening to music.",
            "Recovery 30-minute treadmill session after a long run.",
        ],
        "most_recent": True,
        "disadvantage": "partial",
    },
    {
        "topic": "grocerypickup",
        "query": "What was my most recent weekly grocery pickup at the local market?",
        "gold_text": "Picked up the weekly food order at the corner shop.",
        "decoys": [
            "Did the weekly grocery pickup at the local market.",
            "Quick weekly grocery pickup at the local market on the way home.",
            "Weekly grocery pickup at the local market with my partner.",
            "Big weekly grocery pickup at the local market.",
            "Light weekly grocery pickup at the local market for the weekend.",
            "Weekly grocery pickup at the local market — they were busy.",
            "Weekly grocery pickup at the local market with a short list.",
            "Weekly grocery pickup at the local market after work.",
        ],
        "most_recent": True,
        "disadvantage": "partial",
    },
    {
        "topic": "inlawsdinner",
        "query": "When was my most recent dinner with the in-laws?",
        "gold_text": "Ate supper at my spouse's parents' place.",
        "decoys": [
            "Had dinner with the in-laws at their house.",
            "Long dinner with the in-laws and their friends.",
            "Quiet dinner with the in-laws on a weeknight.",
            "Big dinner with the in-laws for a birthday.",
            "Casual dinner with the in-laws and the kids.",
            "Dinner with the in-laws — they made roast chicken.",
            "Dinner with the in-laws at a restaurant downtown.",
            "Dinner with the in-laws after a long workweek.",
        ],
        "most_recent": True,
        "disadvantage": "partial",
    },
    {
        "topic": "childhoodfriends",
        "query": "When was my most recent video call with childhood friends?",
        "gold_text": "Got on a Zoom with my old school crew.",
        "decoys": [
            "Had a video call with childhood friends in the evening.",
            "Long video call with childhood friends from back home.",
            "Quick video call with childhood friends to catch up.",
            "Group video call with childhood friends and their partners.",
            "Video call with childhood friends about an upcoming trip.",
            "Late-night video call with childhood friends.",
            "Video call with childhood friends from college days.",
            "Video call with childhood friends to plan a reunion.",
        ],
        "most_recent": True,
        "disadvantage": "partial",
    },
    {
        "topic": "boardgames",
        "query": "When was my most recent board game session with the housemates?",
        "gold_text": "Played tabletop with my roommates after dinner.",
        "decoys": [
            "Had a board game session with the housemates on a Friday.",
            "Long board game session with the housemates that ran late.",
            "Casual board game session with the housemates over pizza.",
            "Heavy strategy board game session with the housemates.",
            "Light board game session with the housemates before bed.",
            "Loud board game session with the housemates and guests.",
            "Quick board game session with the housemates between meals.",
            "Board game session with the housemates focused on a new game.",
        ],
        "most_recent": True,
        "disadvantage": "partial",
    },
    {
        "topic": "kayaktrip",
        "query": "What was my most recent kayak trip on the bay?",
        "gold_text": "Paddled around the inlet for the afternoon.",
        "decoys": [
            "Took a kayak trip on the bay during the morning.",
            "Long kayak trip on the bay with two friends.",
            "Short kayak trip on the bay to test new gear.",
            "Solo kayak trip on the bay at sunrise.",
            "Choppy kayak trip on the bay due to weather.",
            "Calm kayak trip on the bay at low tide.",
            "Guided kayak trip on the bay with a club.",
            "Kayak trip on the bay with a picnic stop.",
        ],
        "most_recent": True,
        "disadvantage": "partial",
    },
    {
        "topic": "sushidinner",
        "query": "When was my most recent sushi dinner at the new spot?",
        "gold_text": "Had raw fish at the place that just opened.",
        "decoys": [
            "Went to the new spot for sushi dinner with a friend.",
            "Quick sushi dinner at the new spot before a movie.",
            "Long sushi dinner at the new spot with the omakase menu.",
            "Casual sushi dinner at the new spot on a weeknight.",
            "Sushi dinner at the new spot — sat at the bar.",
            "Big sushi dinner at the new spot for a birthday.",
            "Sushi dinner at the new spot with sake pairing.",
            "Sushi dinner at the new spot — they had a waitlist.",
        ],
        "most_recent": True,
        "disadvantage": "partial",
    },
    {
        "topic": "coffeeshopwork",
        "query": "When was my first coffee shop work session?",
        "gold_text": "Took my laptop to the cafe to grind on a deadline.",
        "decoys": [
            "Did a coffee shop work session for two hours.",
            "Long coffee shop work session before a meeting.",
            "Quick coffee shop work session between errands.",
            "Coffee shop work session at the place near the park.",
            "Coffee shop work session writing the proposal.",
            "Coffee shop work session with my noise-cancelling headphones.",
            "Coffee shop work session after the gym.",
            "Coffee shop work session — got a lot done.",
        ],
        "most_recent": False,
        "disadvantage": "partial",
    },
    {
        "topic": "photowalk",
        "query": "What was my most recent photo walk through downtown?",
        "gold_text": "Wandered the city center with the camera.",
        "decoys": [
            "Took a photo walk through downtown on the weekend.",
            "Long photo walk through downtown with the prime lens.",
            "Quick photo walk through downtown after lunch.",
            "Solo photo walk through downtown at golden hour.",
            "Photo walk through downtown with a friend who shoots film.",
            "Photo walk through downtown focused on architecture.",
            "Photo walk through downtown ending at the waterfront.",
            "Photo walk through downtown in light rain.",
        ],
        "most_recent": True,
        "disadvantage": "partial",
    },
    {
        "topic": "modeltrain",
        "query": "When was my first model train workshop?",
        "gold_text": "Tinkered with miniature railroad gear at a hobby class.",
        "decoys": [
            "Attended a model train workshop on a Saturday.",
            "Long model train workshop focused on weathering.",
            "Hands-on model train workshop with the club.",
            "Beginner model train workshop with the instructor.",
            "Advanced model train workshop on electronics.",
            "Model train workshop at the hobby store across town.",
            "Model train workshop on track planning.",
            "Model train workshop with a layout demo.",
        ],
        "most_recent": False,
        "disadvantage": "partial",
    },

    # ---------- MINOR disadvantage (5 scenarios): gold uses minor wording shifts ----------
    {
        "topic": "languageexchange",
        "query": "When was my most recent language exchange meetup?",
        "gold_text": "Went to the meetup for language exchange last night.",
        "decoys": [
            "Attended a language exchange meetup downtown.",
            "Long language exchange meetup with a big turnout.",
            "Quick language exchange meetup at the cafe.",
            "Language exchange meetup with the Spanish group.",
            "Language exchange meetup focused on conversation pairs.",
            "Language exchange meetup organized by the library.",
            "Language exchange meetup at the new venue.",
            "Language exchange meetup with the regulars.",
        ],
        "most_recent": True,
        "disadvantage": "minor",
    },
    {
        "topic": "bakesale",
        "query": "When was my most recent charity bake sale shift?",
        "gold_text": "Took a shift at the bake sale for charity.",
        "decoys": [
            "Worked a charity bake sale shift at the school.",
            "Long charity bake sale shift on a busy Saturday.",
            "Quick charity bake sale shift in the afternoon.",
            "Charity bake sale shift behind the cash table.",
            "Charity bake sale shift restocking trays.",
            "Charity bake sale shift with two friends.",
            "Charity bake sale shift for the animal shelter.",
            "Charity bake sale shift at the church fundraiser.",
        ],
        "most_recent": True,
        "disadvantage": "minor",
    },
    {
        "topic": "usedbookstore",
        "query": "What was my first used bookstore browse?",
        "gold_text": "Browsed the used bookstore for an hour.",
        "decoys": [
            "Did a used bookstore browse with no shopping list.",
            "Long used bookstore browse on a rainy day.",
            "Quick used bookstore browse before catching a train.",
            "Used bookstore browse focused on the sci-fi section.",
            "Used bookstore browse at the shop on the corner.",
            "Used bookstore browse with my partner.",
            "Used bookstore browse looking for travel guides.",
            "Used bookstore browse ending with three paperbacks.",
        ],
        "most_recent": False,
        "disadvantage": "minor",
    },
    {
        "topic": "pickupvolleyball",
        "query": "When was my most recent pickup volleyball game at the gym?",
        "gold_text": "Played pickup volleyball at the gym last week.",
        "decoys": [
            "Had a pickup volleyball game at the gym after work.",
            "Long pickup volleyball game at the gym with three sets.",
            "Quick pickup volleyball game at the gym at lunch.",
            "Casual pickup volleyball game at the gym with newcomers.",
            "Competitive pickup volleyball game at the gym on league night.",
            "Pickup volleyball game at the gym with a strong team.",
            "Pickup volleyball game at the gym with a short roster.",
            "Pickup volleyball game at the gym ending in a tiebreaker.",
        ],
        "most_recent": True,
        "disadvantage": "minor",
    },
    {
        "topic": "jamsession",
        "query": "What was my first jam session in the garage?",
        "gold_text": "Held a jam session out in the garage.",
        "decoys": [
            "Had a jam session in the garage with the band.",
            "Long jam session in the garage focused on new songs.",
            "Quick jam session in the garage to warm up.",
            "Loud jam session in the garage on a Friday.",
            "Jam session in the garage with the drummer's setup.",
            "Jam session in the garage with two amps.",
            "Jam session in the garage trying out a cover.",
            "Jam session in the garage that ran late.",
        ],
        "most_recent": False,
        "disadvantage": "minor",
    },
]

assert len(SCENARIOS) == 20, f"need 20 scenarios, got {len(SCENARIOS)}"

# Verify topics distinct
topics = [s["topic"] for s in SCENARIOS]
assert len(set(topics)) == len(topics), "topics must be unique"

# Verify disadvantage distribution
from collections import Counter
dist = Counter(s["disadvantage"] for s in SCENARIOS)
print(f"Disadvantage distribution: {dict(dist)}")
assert dist["max"] == 5 and dist["partial"] == 10 and dist["minor"] == 5

# Generate dates per scenario.
# We want 7-9 decoys at older (or for first-queries: newer) dates than gold.
# To match the prior bench style we'll use full ISO-8601 with date-only granularity.
# We'll spread dates across ~2-3 years.

def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    secs = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=secs)

def fmt_dt(d: datetime) -> str:
    # date-only, with T00:00:00Z to match style
    return d.strftime("%Y-%m-%dT00:00:00Z")

# Year-range plan:
# - For most_recent queries (gold latest):
#     decoys spread across 2021-01-01..2024-12-31
#     gold in 2025-06-01..2025-12-31 (clearly latest, before ref_time 2026-04-01)
# - For first queries (gold earliest):
#     gold in 2021-01-01..2021-03-31 (clearly earliest)
#     decoys spread across 2022-01-01..2025-06-30

# Tweak: gold dates must be unique vs decoys; decoys must all differ from each other.

docs_out = []
queries_out = []
gold_out = []

for i, scen in enumerate(SCENARIOS, start=1):
    topic = scen["topic"]
    nn = f"{i:03d}"
    num_decoys = len(scen["decoys"])
    assert 7 <= num_decoys <= 9, f"{topic}: decoys must be 7-9, got {num_decoys}"

    if scen["most_recent"]:
        # decoys: 2021-01..2024-12 spread
        decoy_dates = []
        used = set()
        while len(decoy_dates) < num_decoys:
            d = random_date(datetime(2021,1,1,tzinfo=timezone.utc),
                            datetime(2024,12,31,tzinfo=timezone.utc))
            d = d.replace(hour=0, minute=0, second=0, microsecond=0)
            if d in used:
                continue
            used.add(d)
            decoy_dates.append(d)
        decoy_dates.sort()  # older first
        gold_date = random_date(datetime(2025,6,1,tzinfo=timezone.utc),
                                datetime(2025,12,31,tzinfo=timezone.utc))
        gold_date = gold_date.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # first/earliest query: gold earliest; decoys after
        gold_date = random_date(datetime(2021,1,1,tzinfo=timezone.utc),
                                datetime(2021,3,31,tzinfo=timezone.utc))
        gold_date = gold_date.replace(hour=0, minute=0, second=0, microsecond=0)
        decoy_dates = []
        used = {gold_date}
        while len(decoy_dates) < num_decoys:
            d = random_date(datetime(2022,1,1,tzinfo=timezone.utc),
                            datetime(2025,6,30,tzinfo=timezone.utc))
            d = d.replace(hour=0, minute=0, second=0, microsecond=0)
            if d in used:
                continue
            used.add(d)
            decoy_dates.append(d)
        decoy_dates.sort()

    # Verify gold is most-recent or earliest
    if scen["most_recent"]:
        assert all(d < gold_date for d in decoy_dates), f"{topic}: gold not latest"
    else:
        assert all(d > gold_date for d in decoy_dates), f"{topic}: gold not earliest"

    # Build docs
    gold_id = f"rvr_{topic}_{nn}_g0"
    docs_out.append({"doc_id": gold_id, "text": scen["gold_text"], "ref_time": fmt_dt(gold_date)})
    for j, dtext in enumerate(scen["decoys"]):
        did = f"rvr_{topic}_{nn}_d{j}"
        docs_out.append({"doc_id": did, "text": dtext, "ref_time": fmt_dt(decoy_dates[j])})

    # Build query
    qid = f"rvr_q_{topic}_{nn}"
    queries_out.append({"query_id": qid, "text": scen["query"], "ref_time": REF_TIME})

    # Build gold
    gold_out.append({"query_id": qid, "relevant_doc_ids": [gold_id]})

# Verification pass

# 1. Every gold id in docs
doc_ids = {d["doc_id"] for d in docs_out}
for g in gold_out:
    for rid in g["relevant_doc_ids"]:
        assert rid in doc_ids, f"gold {rid} not in docs"

# 2 and 3 done in loop above.

# 4. No month names or day-of-month numbers in doc text.
MONTHS = ["January","February","March","April","May","June","July","August",
          "September","October","November","December",
          "Jan","Feb","Mar","Apr","Jun","Jul","Aug","Sep","Sept","Oct","Nov","Dec"]
# Note: don't include "May" as a substring check would false-positive trivially; we'll check word-boundary.
import re
MONTH_RE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b")
DAY_RE = re.compile(r"\b([1-9]|[12][0-9]|3[01])(st|nd|rd|th)?\b")
# Day check: any 1-2 digit day-of-month number standalone. We allow "30-minute" as compound; require word boundary plus not preceded/followed by - or digit
DAY_STRICT_RE = re.compile(r"(?<![\w-])(?:[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?(?![\w-])")

for d in docs_out:
    text = d["text"]
    m = MONTH_RE.search(text)
    if m:
        raise AssertionError(f"month name in doc {d['doc_id']}: {text!r} match={m.group()}")
    # day check
    m2 = DAY_STRICT_RE.search(text)
    if m2:
        raise AssertionError(f"day number in doc {d['doc_id']}: {text!r} match={m2.group()}")

# 5. Within each scenario verify gold paraphrase vs decoys keyword overlap (informational).
def tokens(s):
    return set(re.findall(r"[a-z]+", s.lower()))

for i, scen in enumerate(SCENARIOS, start=1):
    nn = f"{i:03d}"
    gtoks = tokens(scen["gold_text"])
    qtoks = tokens(scen["query"])
    g_overlap = len(gtoks & qtoks)
    dec_overlaps = [len(tokens(d) & qtoks) for d in scen["decoys"]]
    avg_dec = sum(dec_overlaps)/len(dec_overlaps)
    if not (g_overlap < avg_dec):
        print(f"WARN: {scen['topic']}: gold_overlap={g_overlap} avg_decoy_overlap={avg_dec:.2f}")

# Counts
print(f"docs: {len(docs_out)}  queries: {len(queries_out)}  gold: {len(gold_out)}")

# Write files
def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

write_jsonl(os.path.join(OUT_DIR, "recency_vs_rerank_docs.jsonl"), docs_out)
write_jsonl(os.path.join(OUT_DIR, "recency_vs_rerank_queries.jsonl"), queries_out)
write_jsonl(os.path.join(OUT_DIR, "recency_vs_rerank_gold.jsonl"), gold_out)

print("done")
