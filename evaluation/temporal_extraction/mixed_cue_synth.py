"""Mixed-cue stress test synthesizer.

Generates 200 docs (50 each of 4 cue types) and 40 queries (10 per type)
that genuinely test retrieval via the dominant cue:

  - DATE-EXPLICIT  (T-cue):  "On April 5, 2023, I bought a kayak."
  - PURE-CONTENT   (S-cue):  "I prefer dark roast coffee."
  - RECURRENCE     (L-cue):  "Every Tuesday is book club at 7pm."
  - ERA-REFERENCE  (E-cue):  "Back in college I worked at the bookstore."

Domain-neutral. Names rotate through generic anglosphere first names so
the semantic channel can distinguish.

Gold: 1 doc per query.

Output:
  data/mixed_cue_docs.jsonl
  data/mixed_cue_queries.jsonl
  data/mixed_cue_gold.jsonl

Run once; deterministic via seeded random.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2026-01-15T00:00:00Z"

NAMES = [
    "Alice",
    "Ben",
    "Carla",
    "David",
    "Elena",
    "Felix",
    "Gabriela",
    "Henry",
    "Ines",
    "Jonah",
    "Kira",
    "Liam",
    "Maya",
    "Noah",
    "Olive",
    "Pavel",
    "Quinn",
    "Rosa",
    "Sam",
    "Tara",
    "Umar",
    "Vera",
    "Walter",
    "Xenia",
    "Yara",
    "Zane",
]

# DATE-EXPLICIT events: (verb-phrase, object) — paired with explicit dates.
DATE_EVENTS = [
    ("bought a kayak", "kayak"),
    ("adopted a beagle", "beagle"),
    ("ran the marathon downtown", "marathon"),
    ("opened a savings account", "savings account"),
    ("hosted a housewarming", "housewarming"),
    ("upgraded the kitchen sink", "kitchen sink"),
    ("met the new neighbors", "new neighbors"),
    ("finished the basement renovation", "basement renovation"),
    ("submitted the visa paperwork", "visa paperwork"),
    ("sold the old sedan", "old sedan"),
    ("renewed the apartment lease", "apartment lease"),
    ("got a flu shot", "flu shot"),
    ("visited the natural history museum", "natural history museum"),
    ("started piano lessons", "piano lessons"),
    ("joined a community choir", "community choir"),
    ("photographed the lunar eclipse", "lunar eclipse"),
    ("attended a pottery class", "pottery class"),
    ("bought hiking boots", "hiking boots"),
    ("planted a fig tree", "fig tree"),
    ("painted the garage door", "garage door"),
    ("repaired the leaky faucet", "leaky faucet"),
    ("baked a sourdough loaf", "sourdough loaf"),
    ("returned the library books", "library books"),
    ("filed the tax return", "tax return"),
    ("changed the car tires", "car tires"),
    ("organized a yard sale", "yard sale"),
    ("watched the comet pass overhead", "comet"),
    ("flew a kite at the park", "kite"),
    ("welcomed a new puppy home", "new puppy"),
    ("installed solar panels", "solar panels"),
    ("read the entire trilogy", "trilogy"),
    ("planted tulip bulbs", "tulip bulbs"),
    ("hiked Mount Storm", "Mount Storm"),
    ("attended the science fair", "science fair"),
    ("rebuilt the bicycle gears", "bicycle gears"),
    ("hosted a board game night", "board game night"),
    ("wrote a short story", "short story"),
    ("painted the bedroom blue", "bedroom blue"),
    ("registered for citizenship", "citizenship"),
    ("renewed the gym membership", "gym membership"),
    ("toured the botanical gardens", "botanical gardens"),
    ("rented a beach cottage", "beach cottage"),
    ("repaired the mailbox post", "mailbox post"),
    ("photographed the heron rookery", "heron rookery"),
    ("delivered a wedding toast", "wedding toast"),
    ("submitted the mortgage application", "mortgage application"),
    ("planted a row of sunflowers", "sunflowers"),
    ("attended a chess tournament", "chess tournament"),
    ("started a meditation practice", "meditation practice"),
    ("hosted the school bake sale", "school bake sale"),
]

DATE_DATES = [
    ("April 5, 2023", "April 5, 2023"),
    ("June 12, 2022", "June 12, 2022"),
    ("September 30, 2024", "September 30, 2024"),
    ("January 8, 2023", "January 8, 2023"),
    ("November 14, 2021", "November 14, 2021"),
    ("February 20, 2024", "February 20, 2024"),
    ("July 4, 2023", "July 4, 2023"),
    ("October 22, 2022", "October 22, 2022"),
    ("December 1, 2024", "December 1, 2024"),
    ("March 17, 2023", "March 17, 2023"),
    ("August 9, 2022", "August 9, 2022"),
    ("May 23, 2024", "May 23, 2024"),
    ("April 14, 2025", "April 14, 2025"),
    ("June 28, 2023", "June 28, 2023"),
    ("September 5, 2022", "September 5, 2022"),
    ("November 30, 2024", "November 30, 2024"),
    ("January 19, 2023", "January 19, 2023"),
    ("February 11, 2025", "February 11, 2025"),
    ("July 17, 2023", "July 17, 2023"),
    ("October 3, 2024", "October 3, 2024"),
    ("December 22, 2022", "December 22, 2022"),
    ("March 8, 2025", "March 8, 2025"),
    ("August 16, 2023", "August 16, 2023"),
    ("May 4, 2022", "May 4, 2022"),
    ("April 27, 2025", "April 27, 2025"),
    ("June 6, 2024", "June 6, 2024"),
    ("September 13, 2023", "September 13, 2023"),
    ("November 7, 2022", "November 7, 2022"),
    ("January 25, 2025", "January 25, 2025"),
    ("February 2, 2023", "February 2, 2023"),
    ("July 21, 2024", "July 21, 2024"),
    ("October 11, 2025", "October 11, 2025"),
    ("December 9, 2023", "December 9, 2023"),
    ("March 24, 2024", "March 24, 2024"),
    ("August 30, 2025", "August 30, 2025"),
    ("May 15, 2023", "May 15, 2023"),
    ("April 18, 2022", "April 18, 2022"),
    ("June 1, 2025", "June 1, 2025"),
    ("September 19, 2024", "September 19, 2024"),
    ("November 28, 2023", "November 28, 2023"),
    ("January 6, 2022", "January 6, 2022"),
    ("February 17, 2024", "February 17, 2024"),
    ("July 30, 2022", "July 30, 2022"),
    ("October 25, 2023", "October 25, 2023"),
    ("December 16, 2024", "December 16, 2024"),
    ("March 1, 2022", "March 1, 2022"),
    ("August 23, 2023", "August 23, 2023"),
    ("May 9, 2025", "May 9, 2025"),
    ("April 11, 2024", "April 11, 2024"),
    ("June 19, 2022", "June 19, 2022"),
]

# PURE-CONTENT: hobbies/preferences with no time signal at all.
PURE_CONTENT_TEMPLATES = [
    ("prefers dark roast coffee", "dark roast coffee"),
    ("loves indoor rock climbing", "indoor rock climbing"),
    ("collects vintage postcards", "vintage postcards"),
    ("hates cilantro on tacos", "cilantro on tacos"),
    ("plays the ukulele on weekends", "ukulele"),
    ("paints abstract watercolors", "abstract watercolors"),
    ("reads detective fiction", "detective fiction"),
    ("bakes oatmeal raisin cookies", "oatmeal raisin cookies"),
    ("grows heirloom tomatoes", "heirloom tomatoes"),
    ("collects sea glass at the shore", "sea glass"),
    ("listens to ambient electronic music", "ambient electronic music"),
    ("brews chai from scratch", "chai"),
    ("watches old samurai films", "samurai films"),
    ("favors merino wool socks", "merino wool socks"),
    ("grows bonsai trees", "bonsai trees"),
    ("collects fountain pens", "fountain pens"),
    ("makes pickled red onions", "pickled red onions"),
    ("knits chunky scarves", "chunky scarves"),
    ("avoids caffeine entirely", "caffeine"),
    ("prefers window seats on flights", "window seats"),
    ("plays competitive scrabble", "scrabble"),
    ("collects mid-century lamps", "mid-century lamps"),
    ("enjoys long walks along canals", "canals"),
    ("writes poetry in journals", "poetry"),
    ("follows formula one racing", "formula one racing"),
    ("breeds ornamental koi", "ornamental koi"),
    ("photographs urban graffiti", "urban graffiti"),
    ("sews quilted bags", "quilted bags"),
    ("brews kombucha at home", "kombucha"),
    ("collects rare cookbooks", "rare cookbooks"),
    ("loves spicy szechuan dishes", "spicy szechuan dishes"),
    ("plays disc golf at the park", "disc golf"),
    ("makes cold-process soaps", "cold-process soaps"),
    ("enjoys birdwatching at marshes", "birdwatching"),
    ("collects antique compasses", "antique compasses"),
    ("trains for trail running", "trail running"),
    ("plays vintage synthesizers", "vintage synthesizers"),
    ("forages for wild mushrooms", "wild mushrooms"),
    ("builds custom mechanical keyboards", "mechanical keyboards"),
    ("paints miniature figurines", "miniature figurines"),
    ("brews espresso with a manual lever", "espresso"),
    ("collects vinyl jazz records", "vinyl jazz records"),
    ("loves spicy laksa noodles", "laksa noodles"),
    ("sketches architectural buildings", "architectural buildings"),
    ("plays competitive bridge", "bridge"),
    ("makes handmade pasta from scratch", "handmade pasta"),
    ("collects pressed flowers in books", "pressed flowers"),
    ("rebuilds carburetors as a hobby", "carburetors"),
    ("hosts a podcast on gardening", "podcast on gardening"),
    ("reads philosophy after dinner", "philosophy"),
]

# RECURRENCE: weekly / monthly recurring events.
RECURRENCE_TEMPLATES = [
    ("Every Tuesday is book club at 7pm", "book club", "Tuesday"),
    ("Every Saturday morning is the farmers market", "farmers market", "Saturday"),
    ("Every Wednesday evening is yoga class", "yoga class", "Wednesday"),
    ("Every Friday night is family dinner", "family dinner", "Friday"),
    ("Every Sunday afternoon is hiking with the dog", "hiking with the dog", "Sunday"),
    ("Every Monday morning is the team standup", "team standup", "Monday"),
    ("Every Thursday is pickleball league", "pickleball league", "Thursday"),
    ("Every Tuesday night is pottery class", "pottery class", "Tuesday"),
    (
        "Every Saturday is volunteer shift at the shelter",
        "volunteer shift at the shelter",
        "Saturday",
    ),
    ("Every Wednesday is kids' swim practice", "swim practice", "Wednesday"),
    ("Every Friday is cocktail night with neighbors", "cocktail night", "Friday"),
    ("Every Sunday morning is church choir rehearsal", "choir rehearsal", "Sunday"),
    ("Every Monday is grocery shopping at dawn", "grocery shopping", "Monday"),
    ("Every Thursday is poker night with old friends", "poker night", "Thursday"),
    ("Every Tuesday is meal prep for the week", "meal prep", "Tuesday"),
    ("Every Saturday is a long bike ride", "long bike ride", "Saturday"),
    ("Every Wednesday is karaoke at the local pub", "karaoke", "Wednesday"),
    ("Every Friday is take-out sushi night", "take-out sushi night", "Friday"),
    ("Every Sunday evening is letter-writing time", "letter-writing time", "Sunday"),
    ("Every Monday is tennis lessons after work", "tennis lessons", "Monday"),
    ("Every Thursday is open mic at the cafe", "open mic", "Thursday"),
    (
        "Every Tuesday is German conversation group",
        "German conversation group",
        "Tuesday",
    ),
    ("Every Saturday morning is parkrun at the trail", "parkrun", "Saturday"),
    ("Every Wednesday is improv class", "improv class", "Wednesday"),
    ("Every Friday is gardening with the kids", "gardening with the kids", "Friday"),
    ("Every Sunday is the antique market visit", "antique market", "Sunday"),
    ("Every Monday is figure-drawing studio", "figure-drawing studio", "Monday"),
    ("Every Thursday is salsa dancing downtown", "salsa dancing", "Thursday"),
    ("Every Tuesday is pottery wheel time", "pottery wheel", "Tuesday"),
    (
        "Every Saturday is brunch with the cousins",
        "brunch with the cousins",
        "Saturday",
    ),
    (
        "Every Wednesday is jazz quartet rehearsal",
        "jazz quartet rehearsal",
        "Wednesday",
    ),
    ("Every Friday is movie night at home", "movie night", "Friday"),
    ("Every Sunday is the long swim at the lake", "long swim at the lake", "Sunday"),
    ("Every Monday is a calligraphy session", "calligraphy session", "Monday"),
    ("Every Thursday is lap swim before work", "lap swim", "Thursday"),
    ("Every Tuesday is chess club after school", "chess club", "Tuesday"),
    (
        "Every Saturday is woodworking in the garage",
        "woodworking in the garage",
        "Saturday",
    ),
    ("Every Wednesday is meditation circle", "meditation circle", "Wednesday"),
    ("Every Friday is bouldering at the gym", "bouldering at the gym", "Friday"),
    ("Every Sunday is grandma's pierogi lunch", "pierogi lunch", "Sunday"),
    ("Every Monday is morning piano practice", "piano practice", "Monday"),
    ("Every Thursday is cycling club ride", "cycling club ride", "Thursday"),
    (
        "Every Tuesday is short story writing group",
        "short story writing group",
        "Tuesday",
    ),
    ("Every Saturday is the coastal walk", "coastal walk", "Saturday"),
    (
        "Every Wednesday is community garden weeding",
        "community garden weeding",
        "Wednesday",
    ),
    ("Every Friday is fish-fry at the diner", "fish-fry", "Friday"),
    ("Every Sunday is birding at the marsh", "birding at the marsh", "Sunday"),
    ("Every Monday is choir warm-ups", "choir warm-ups", "Monday"),
    ("Every Thursday is bridge club afternoon", "bridge club", "Thursday"),
    ("Every Tuesday is judo dojo practice", "judo dojo practice", "Tuesday"),
]

# ERA-REFERENCE: era anchors with no concrete date.
ERA_TEMPLATES = [
    ("Back in college I worked at the bookstore", "bookstore job", "college"),
    ("In the 90s we used to spend summers in Maine", "summers in Maine", "90s"),
    (
        "Back when the kids were toddlers we lived in Phoenix",
        "lived in Phoenix",
        "kids were toddlers",
    ),
    (
        "During the pandemic I learned to bake bread",
        "learned to bake bread",
        "pandemic",
    ),
    ("In my twenties I traveled across Asia", "traveled across Asia", "twenties"),
    (
        "Back in high school I played varsity tennis",
        "played varsity tennis",
        "high school",
    ),
    (
        "In the early 2000s I drove a beat-up Volvo",
        "drove a beat-up Volvo",
        "early 2000s",
    ),
    ("During grad school I tutored undergrads", "tutored undergrads", "grad school"),
    (
        "Back when we lived in Brooklyn we had a rooftop garden",
        "rooftop garden",
        "lived in Brooklyn",
    ),
    ("In the 80s I had a paper route", "paper route", "80s"),
    ("During the pandemic everyone took up sourdough", "took up sourdough", "pandemic"),
    (
        "Back in the day we had a chocolate Lab named Buster",
        "Lab named Buster",
        "back in the day",
    ),
    ("In college I joined the rowing team", "rowing team", "college"),
    (
        "Back when I commuted by train I read constantly",
        "read constantly",
        "commuted by train",
    ),
    ("During my thirties I learned to surf", "learned to surf", "thirties"),
    (
        "In the early 90s we drove cross-country in a station wagon",
        "drove cross-country",
        "early 90s",
    ),
    (
        "Back in elementary school I won the spelling bee",
        "won the spelling bee",
        "elementary school",
    ),
    (
        "During my first marriage we lived in Toronto",
        "lived in Toronto",
        "first marriage",
    ),
    ("In the 70s my parents ran a record store", "ran a record store", "70s"),
    (
        "Back when I was an intern I shared an apartment with three roommates",
        "shared an apartment with three roommates",
        "intern",
    ),
    ("In my forties I started running ultras", "started running ultras", "forties"),
    (
        "During the move to Seattle I lost a box of photos",
        "lost a box of photos",
        "move to Seattle",
    ),
    ("Back in middle school we wrote in cursive", "wrote in cursive", "middle school"),
    ("In the 60s my grandfather worked the docks", "worked the docks", "60s"),
    (
        "During my time in the Peace Corps I taught English",
        "taught English",
        "Peace Corps",
    ),
    (
        "Back when I had long hair I went to many concerts",
        "went to many concerts",
        "long hair",
    ),
    (
        "In college my roommate brewed terrible coffee",
        "brewed terrible coffee",
        "college",
    ),
    (
        "Back when we lived in Austin we had a porch swing",
        "porch swing",
        "lived in Austin",
    ),
    ("During my fifties I took up woodturning", "took up woodturning", "fifties"),
    ("In the early 80s we got our first VCR", "first VCR", "early 80s"),
    (
        "Back in seminary I memorized hymns in Latin",
        "memorized hymns in Latin",
        "seminary",
    ),
    (
        "During the recession I drove a delivery van",
        "drove a delivery van",
        "recession",
    ),
    (
        "In law school I clerked for a federal judge",
        "clerked for a federal judge",
        "law school",
    ),
    (
        "Back when I had a roommate named Tony we ate ramen nightly",
        "ate ramen nightly",
        "roommate named Tony",
    ),
    ("In the 90s my dad sold restored jukeboxes", "sold restored jukeboxes", "90s"),
    ("During my year abroad I lived in Lisbon", "lived in Lisbon", "year abroad"),
    ("Back in art school I welded sculptures", "welded sculptures", "art school"),
    (
        "In the early 2010s I worked for a startup",
        "worked for a startup",
        "early 2010s",
    ),
    (
        "Back when we owned the cabin we hosted huge gatherings",
        "hosted huge gatherings",
        "owned the cabin",
    ),
    ("During my postdoc I studied marine algae", "studied marine algae", "postdoc"),
    (
        "In the late 2000s I rode a fixed-gear bike everywhere",
        "rode a fixed-gear bike",
        "late 2000s",
    ),
    (
        "Back when the dog was a puppy he chewed every shoe",
        "chewed every shoe",
        "dog was a puppy",
    ),
    (
        "In medical residency I worked thirty-six hour shifts",
        "worked thirty-six hour shifts",
        "medical residency",
    ),
    (
        "During my years as a paralegal I learned shorthand",
        "learned shorthand",
        "paralegal",
    ),
    (
        "Back in boarding school we wrote letters home",
        "wrote letters home",
        "boarding school",
    ),
    ("In the 50s my grandparents ran a diner", "ran a diner", "50s"),
    (
        "Back when we lived in Denver we skied every weekend",
        "skied every weekend",
        "lived in Denver",
    ),
    (
        "During my first job I commuted two hours each way",
        "commuted two hours",
        "first job",
    ),
    (
        "In the early 70s my mother sewed all our clothes",
        "sewed all our clothes",
        "early 70s",
    ),
    (
        "Back in journalism school we used film cameras",
        "used film cameras",
        "journalism school",
    ),
]


def main() -> None:
    random.seed(7)
    docs: list[dict] = []
    queries: list[dict] = []
    gold: list[dict] = []
    doc_idx = 0

    # ---- 50 DATE-EXPLICIT docs ----
    date_indices = list(range(50))
    random.shuffle(date_indices)
    for i in range(50):
        verb_phrase, noun = DATE_EVENTS[i]
        date_str, date_query_str = DATE_DATES[i]
        name = NAMES[i % len(NAMES)]
        text = f"On {date_str}, {name} {verb_phrase}."
        did = f"mc_date_{i:03d}"
        docs.append(
            {"doc_id": did, "text": text, "ref_time": REF_TIME, "cue_type": "date"}
        )
        doc_idx += 1

    # 10 date-explicit queries: pick docs whose date is searchable by month+year
    date_q_picks = random.sample(range(50), 10)
    for j, idx in enumerate(date_q_picks):
        verb_phrase, noun = DATE_EVENTS[idx]
        date_str, _ = DATE_DATES[idx]
        # Query queries by date — strip noun
        qid = f"qmc_date_{j:03d}"
        # Use the date as the salient cue
        q_text = f"What happened on {date_str}?"
        queries.append(
            {"query_id": qid, "text": q_text, "ref_time": REF_TIME, "cue_type": "date"}
        )
        gold.append({"query_id": qid, "relevant_doc_ids": [f"mc_date_{idx:03d}"]})

    # ---- 50 PURE-CONTENT docs ----
    for i in range(50):
        phrase, topic = PURE_CONTENT_TEMPLATES[i]
        name = NAMES[(i + 5) % len(NAMES)]
        text = f"{name} {phrase}."
        did = f"mc_content_{i:03d}"
        docs.append(
            {"doc_id": did, "text": text, "ref_time": REF_TIME, "cue_type": "content"}
        )

    content_q_picks = random.sample(range(50), 10)
    for j, idx in enumerate(content_q_picks):
        phrase, topic = PURE_CONTENT_TEMPLATES[idx]
        qid = f"qmc_content_{j:03d}"
        q_text = f"Who is into {topic}?"
        queries.append(
            {
                "query_id": qid,
                "text": q_text,
                "ref_time": REF_TIME,
                "cue_type": "content",
            }
        )
        gold.append({"query_id": qid, "relevant_doc_ids": [f"mc_content_{idx:03d}"]})

    # ---- 50 RECURRENCE docs ----
    for i in range(50):
        phrase, topic, weekday = RECURRENCE_TEMPLATES[i]
        name = NAMES[(i + 11) % len(NAMES)]
        text = f"{phrase} for {name}."
        did = f"mc_recur_{i:03d}"
        docs.append(
            {
                "doc_id": did,
                "text": text,
                "ref_time": REF_TIME,
                "cue_type": "recurrence",
            }
        )

    recur_q_picks = random.sample(range(50), 10)
    for j, idx in enumerate(recur_q_picks):
        phrase, topic, weekday = RECURRENCE_TEMPLATES[idx]
        qid = f"qmc_recur_{j:03d}"
        # Query about the recurring pattern
        q_text = f"What does {NAMES[(idx + 11) % len(NAMES)]} do every {weekday}?"
        queries.append(
            {
                "query_id": qid,
                "text": q_text,
                "ref_time": REF_TIME,
                "cue_type": "recurrence",
            }
        )
        gold.append({"query_id": qid, "relevant_doc_ids": [f"mc_recur_{idx:03d}"]})

    # ---- 50 ERA-REFERENCE docs ----
    for i in range(50):
        phrase, topic, era = ERA_TEMPLATES[i]
        name = NAMES[(i + 17) % len(NAMES)]
        text = f"{phrase} (according to {name})."
        did = f"mc_era_{i:03d}"
        docs.append(
            {"doc_id": did, "text": text, "ref_time": REF_TIME, "cue_type": "era"}
        )

    era_q_picks = random.sample(range(50), 10)
    for j, idx in enumerate(era_q_picks):
        phrase, topic, era = ERA_TEMPLATES[idx]
        qid = f"qmc_era_{j:03d}"
        # Query about the era — use topic+era language
        q_text = f"What did people do {era}?"
        queries.append(
            {"query_id": qid, "text": q_text, "ref_time": REF_TIME, "cue_type": "era"}
        )
        gold.append({"query_id": qid, "relevant_doc_ids": [f"mc_era_{idx:03d}"]})

    docs_path = DATA_DIR / "mixed_cue_docs.jsonl"
    queries_path = DATA_DIR / "mixed_cue_queries.jsonl"
    gold_path = DATA_DIR / "mixed_cue_gold.jsonl"
    with docs_path.open("w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with queries_path.open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    with gold_path.open("w") as f:
        for g in gold:
            f.write(json.dumps(g) + "\n")

    print(f"Wrote {len(docs)} docs to {docs_path}")
    print(f"Wrote {len(queries)} queries to {queries_path}")
    print(f"Wrote {len(gold)} gold to {gold_path}")


if __name__ == "__main__":
    main()
