"""Synthetic test cases for the project-schema operator eval.

Each TEST CASE is a *long* external memory: 30-100k tokens of mixed content
(prior chats with the user, project docs, supporting notes, distractor noise).
A handful of LOAD-BEARING constraints / preferences / goals are scattered
across many chunks. The agent runs a multi-step task that requires invoking
those scattered constraints at the FINAL answer / sub-decision.

Diversity: travel, software architecture, dietary planning. Different vocabulary
forces the schema to be GENERAL — slot-filling, not domain-pattern matching.

Each MemoryChunk is independently retrievable. Cumulative tokens per scenario
target 30-100k so the full memory cannot fit into a 10k WM budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------


@dataclass
class MemoryChunk:
    chunk_id: str
    title: str
    text: str
    tags: list[str] = field(default_factory=list)  # for retrieval scoring


@dataclass
class TestCase:
    name: str
    domain: str
    task_brief: str  # what the user asks the agent to do (visible up-front)
    plan_steps: list[str]  # multi-step plan executed by the agent
    final_question: str  # final sub-decision; rubric judges answer to THIS
    rubric: list[dict]  # each: {id, description, needles_any/needles_all/forbidden}
    memory: list[MemoryChunk]  # external store


# ----------------------------------------------------------------------------
# Helper to bulk distractors of about the right size
# ----------------------------------------------------------------------------


def _filler(topic: str, paragraphs: int, seed: int = 0) -> str:
    """Generate plausible but constraint-free filler text on `topic`."""
    sentences = [
        f"This note discusses {topic} in some detail.",
        f"The general background on {topic} includes historical context, terminology, and common practitioner opinions.",
        f"Discussions of {topic} often reference adjacent fields and overlapping concerns.",
        f"Practitioners working on {topic} tend to emphasize practical heuristics over theoretical purity.",
        f"There is broad agreement that {topic} is non-trivial and benefits from careful planning.",
        f"Several published guides on {topic} converge on a small set of best-practice steps.",
        f"Edge cases around {topic} usually arise from poorly understood inputs rather than algorithmic flaws.",
        f"When teaching {topic}, instructors often start with motivating examples before formalizing.",
        f"A typical project encounters {topic} indirectly through dependencies and integration concerns.",
        f"Performance considerations for {topic} usually trade off latency against throughput.",
        f"Cost considerations around {topic} tend to dominate in larger deployments.",
        f"Documentation quality for {topic} varies widely across vendors and open-source projects.",
        f"Beginner-level errors in {topic} usually stem from skipping the discovery step.",
        f"Mature teams treat {topic} as a first-class concern with explicit owners.",
        f"Postmortem analyses involving {topic} often surface previously implicit assumptions.",
    ]
    out: list[str] = []
    for p in range(paragraphs):
        offset = (seed + p) % len(sentences)
        rotated = sentences[offset:] + sentences[:offset]
        out.append(" ".join(rotated[: 6 + (p % 4)]))
    return "\n\n".join(out)


# ----------------------------------------------------------------------------
# Scenario 1 — Travel planning for Kyoto
# ----------------------------------------------------------------------------


def _travel_case() -> TestCase:
    M: list[MemoryChunk] = []

    # Load-bearing constraint chunks (user's prior conversations, scattered)
    M.append(
        MemoryChunk(
            "travel_chat_001",
            "Earlier chat — Marcus accessibility",
            "User (Feb 12): My partner Marcus will travel with me. He uses a "
            "wheelchair full-time, so accessibility is a hard requirement at "
            "every stop — hotel, restaurants, transit, sights. He can transfer "
            "to a regular chair for short stretches, but he needs step-free "
            "paths and accessible restrooms throughout the trip.",
            tags=["marcus", "accessibility", "wheelchair", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "travel_chat_017",
            "Earlier chat — Marcus shellfish allergy",
            "User (Feb 14): I should mention — Marcus has an ER-level shellfish "
            "allergy. We've done the epi-pen route once. So at restaurants we "
            "have to ask about cross-contamination, and absolutely no shellfish "
            "anywhere — shrimp, crab, lobster, anything in the dashi sometimes "
            "uses bonito flakes which is fine but watch for shrimp-stock too.",
            tags=["marcus", "shellfish", "allergy", "constraint", "food"],
        )
    )
    M.append(
        MemoryChunk(
            "travel_chat_034",
            "Earlier chat — ryokan preference and bathing",
            "User (Feb 21): I've always wanted to stay at a ryokan, even one "
            "night. But Marcus and I agreed: NO shared bathing facilities. So "
            "if you find a ryokan it must have private en-suite baths, not the "
            "communal onsen setup.",
            tags=["ryokan", "bath", "preference", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "travel_chat_052",
            "Earlier chat — mom's stamina",
            "User (Feb 28): Quick update — my mom (72) is joining us for the "
            "Kyoto leg. She gets tired fast, recovering from a hip thing. Cap "
            "her active time at about 4 hours a day. After that she wants to "
            "rest at the hotel. Build in a midday return-to-hotel block.",
            tags=["mom", "stamina", "constraint", "kyoto"],
        )
    )
    M.append(
        MemoryChunk(
            "travel_chat_061",
            "Earlier chat — vegetarian schedule",
            "User (Mar 03): One more food note: we're vegetarian on weekdays, "
            "Mon through Fri. Weekends we eat more flexibly (still no shellfish "
            "for Marcus obviously). Plan weekday meals vegetarian; weekend "
            "meals can be regular Japanese.",
            tags=["vegetarian", "weekday", "diet", "constraint", "food"],
        )
    )
    M.append(
        MemoryChunk(
            "travel_chat_073",
            "Earlier chat — budget envelope",
            "User (Mar 09): Budget for the Kyoto leg is roughly $2200 USD for "
            "the three of us (excluding flights, which are already booked).",
            tags=["budget", "kyoto", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "travel_chat_088",
            "Earlier chat — actual Kyoto dates",
            "User (Mar 14): Kyoto leg is Tuesday June 25 through Thursday June "
            "27 — three days. So it's all weekday dining for us.",
            tags=["kyoto", "dates", "june", "weekday"],
        )
    )

    # Distractor chunks — plausible content from a long memory but not load-bearing
    distractor_topics = [
        "general Japan train pass options",
        "Tokyo neighborhoods overview",
        "luggage forwarding (takkyubin) basics",
        "Japanese vending machine etiquette",
        "convenience store food survey",
        "konbini ATMs and cash logistics",
        "Japanese phone SIM options",
        "hot spring (sento) etiquette generally",
        "tea ceremony historical background",
        "Japanese garden design principles",
        "bullet train seat reservation tactics",
        "Japan-rail transit map highlights",
        "anime / manga district shopping",
        "Tokyo nightlife landmarks",
        "Hokkaido alternative itineraries",
        "Osaka day-trip ideas",
        "Hiroshima memorial trip notes",
        "Mount Fuji viewing windows",
        "calligraphy class options",
        "kimono rental general info",
        "general yen exchange tips",
        "convenience-store coffee comparisons",
        "popular Japanese souvenir categories",
        "kabuki theatre etiquette",
        "Studio Ghibli museum logistics",
        "Tsukiji vs Toyosu market history",
        "shrines vs temples primer",
        "chopstick etiquette myths",
        "Japanese bathing etiquette generally",
        "earthquake preparedness for tourists",
    ]
    for i, topic in enumerate(distractor_topics):
        M.append(
            MemoryChunk(
                f"travel_doc_{100 + i:03d}",
                f"Travel notes — {topic}",
                _filler(topic, paragraphs=8, seed=i),
                tags=topic.split()[:3],
            )
        )

    return TestCase(
        name="kyoto_trip",
        domain="travel",
        task_brief=(
            "Help the user finalize a 3-day Kyoto itinerary. They want a "
            "morning / afternoon / evening block per day, hotel "
            "recommendation type, and one concrete dinner suggestion per day."
        ),
        plan_steps=[
            "Recall who is traveling and any health/accessibility constraints.",
            "Recall any food restrictions (allergies, diet, schedule).",
            "Recall budget envelope and trip dates.",
            "Recall lodging preferences and any hard 'no's.",
            "Sketch the daily structure (morning/afternoon/evening) given pacing constraints.",
        ],
        final_question=(
            "Now produce the final Kyoto itinerary draft: 3 days, each with "
            "morning / afternoon / evening blocks, plus a hotel-style "
            "recommendation and one concrete dinner suggestion per day. "
            "Make sure your plan honors every prior constraint the user has "
            "shared."
        ),
        rubric=[
            {
                "id": "wheelchair_accessible",
                "description": "Plan must call out wheelchair / step-free / accessible accommodations and routing.",
                "needles_any": [
                    "wheelchair",
                    "accessib",
                    "step-free",
                    "barrier-free",
                    "step free",
                ],
            },
            {
                "id": "no_shellfish",
                "description": "Plan must avoid shellfish (no shrimp/crab/lobster) and ideally call out cross-contamination.",
                "needles_any": [
                    "no shellfish",
                    "without shellfish",
                    "shellfish-free",
                    "no shrimp",
                    "avoid shellfish",
                    "shellfish allerg",
                    "no crab",
                    "no lobster",
                ],
            },
            {
                "id": "ryokan_private_bath_or_skip",
                "description": "If a ryokan is included, it must have private (en-suite) baths — NOT communal onsen.",
                "needles_any": [
                    "private bath",
                    "en-suite bath",
                    "ensuite bath",
                    "private onsen",
                    "no shared bath",
                    "without shared bath",
                    "no communal",
                    "without onsen",
                ],
                "ok_if_absent_when": [
                    "no ryokan",
                    "skip ryokan",
                    "no traditional ryokan",
                ],
            },
            {
                "id": "mom_pacing",
                "description": "Daily activity for mom capped near 4 hours; midday rest block / return to hotel built in.",
                "needles_any": [
                    "4 hour",
                    "four hour",
                    "midday rest",
                    "rest at hotel",
                    "return to hotel",
                    "afternoon rest",
                    "pacing",
                    "shorter day",
                    "rest break",
                    "low-energy",
                ],
            },
            {
                "id": "vegetarian_weekdays",
                "description": "Kyoto leg falls Tue-Thu (weekdays) so meals must be vegetarian.",
                "needles_any": [
                    "vegetarian",
                    "no meat",
                    "shojin",
                    "plant-based",
                    "veggie",
                ],
            },
        ],
        memory=M,
    )


# ----------------------------------------------------------------------------
# Scenario 2 — Software architecture decision
# ----------------------------------------------------------------------------


def _software_case() -> TestCase:
    M: list[MemoryChunk] = []

    M.append(
        MemoryChunk(
            "arch_thread_004",
            "Earlier thread — language constraint",
            "Eng channel (Apr 01): Reminder for the new notification router — "
            "we are a Python shop. Leadership and the platform team have been "
            "explicit: no new languages introduced this fiscal year. The "
            "service must be Python-compatible end-to-end.",
            tags=["python", "language", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "arch_thread_011",
            "Earlier thread — queue/streaming policy",
            "Infra meeting notes (Apr 03): Infra has a hard rule: no managed "
            "Kafka, no MSK. The team has had two outages tied to MSK and the "
            "leadership has signed off on AWS-native alternatives only — SQS, "
            "SNS, Kinesis, EventBridge. Anything streaming has to be one of "
            "those.",
            tags=["queue", "kafka", "aws", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "arch_thread_022",
            "Earlier thread — HIPAA and PHI",
            "Compliance review (Apr 05): This service is HIPAA in-scope. PHI "
            "may be present in webhook payloads. Encryption at rest AND in "
            "transit is mandatory; audit logging of access events is mandatory; "
            "all data stores must support customer-managed keys (CMK) via KMS.",
            tags=["hipaa", "phi", "compliance", "encryption", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "arch_thread_034",
            "Earlier thread — availability mandate",
            "Leadership memo (Apr 08): After last quarter's single-AZ outage, "
            "all customer-facing services must be deployed multi-AZ. Target "
            "availability is 99.9%. No exceptions for new services going "
            "forward.",
            tags=["multi-az", "availability", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "arch_thread_046",
            "Earlier thread — secrets management",
            "Security ticket SEC-1284 (Apr 10): No service is permitted to "
            "store secrets in environment variables or in source-mounted files. "
            "All secrets must be retrieved at runtime via AWS Secrets Manager. "
            "This applies to NEW services starting this quarter.",
            tags=["secrets", "secrets-manager", "security", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "arch_thread_058",
            "Earlier thread — observability tooling",
            "Platform sync (Apr 12): We are NOT spinning up new observability "
            "tooling. The org pays for Datadog. Use it. Logs go to Datadog Log "
            "Management; metrics to Datadog metrics; traces via dd-trace-py.",
            tags=["observability", "datadog", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "arch_thread_063",
            "Earlier thread — throughput and latency",
            "Product brief (Apr 14): Steady state is ~20 events/sec, peaks to "
            "200 events/sec during incident storms. End-to-end latency budget "
            "from webhook receipt to downstream notification dispatch is under "
            "2 seconds.",
            tags=["throughput", "latency", "constraint"],
        )
    )

    distractor_topics = [
        "general AWS networking VPC concepts",
        "IAM role-naming conventions in our org",
        "RDS Postgres performance tips generally",
        "CodePipeline overview",
        "Lambda vs Fargate trade-offs in general",
        "Step Functions state machine patterns",
        "API Gateway throttling and quotas",
        "Datadog APM general onboarding",
        "Datadog dashboard patterns",
        "Slack workflow notification ergonomics",
        "PagerDuty escalation policy basics",
        "Terraform module conventions in our org",
        "CDK vs Terraform debate",
        "GitHub Actions CI patterns",
        "OpenTelemetry standards overview",
        "structured logging best practices",
        "service mesh (Istio) overview",
        "blue-green vs canary release strategies",
        "feature flag rollout patterns",
        "S3 lifecycle policies for cost",
        "EC2 Spot Instance economics",
        "container image scanning options",
        "SBOM and dependency hygiene",
        "OAuth2 vs OIDC primer",
        "SAML SSO integration steps",
        "CSP and CORS gotchas",
        "GDPR vs HIPAA differences general",
        "data classification taxonomies",
        "incident response runbook templates",
        "postmortem culture norms",
        "RACI charts for service ownership",
        "engineering ladder career framework",
        "Terraform state-file management",
        "DynamoDB single-table design",
    ]
    for i, topic in enumerate(distractor_topics):
        M.append(
            MemoryChunk(
                f"arch_doc_{100 + i:03d}",
                f"Engineering note — {topic}",
                _filler(topic, paragraphs=9, seed=i + 50),
                tags=topic.split()[:3],
            )
        )

    return TestCase(
        name="notification_router_arch",
        domain="software_architecture",
        task_brief=(
            "Help engineering finalize the architecture for a new "
            "internal notification router service. Deliver: chosen "
            "queue/streaming layer, compute layer, secrets story, and "
            "observability story — each with a one-sentence justification."
        ),
        plan_steps=[
            "Recall the language / runtime constraint for the team.",
            "Recall the queue/streaming policy from infra.",
            "Recall compliance posture (HIPAA / PHI / encryption / audit).",
            "Recall availability mandate and any deployment shape rules.",
            "Recall secrets management and observability tooling rules.",
        ],
        final_question=(
            "Now write the architecture proposal: pick one queue/streaming "
            "layer, one compute layer, the secrets approach, and the "
            "observability approach — each with a one-sentence "
            "justification. Make sure every choice honors the constraints "
            "the team has already agreed to."
        ),
        rubric=[
            {
                "id": "python_compatible_compute",
                "description": "Compute layer must be Python-compatible (Lambda Python / ECS Fargate Python / EC2 Python). No Go/Rust/Java/.NET as the *primary* language.",
                "needles_any": ["python", "lambda", "fargate", "ecs"],
            },
            {
                "id": "aws_native_no_kafka",
                "description": "Queue/streaming layer must be SQS / SNS / Kinesis / EventBridge — NOT MSK / Kafka / Confluent.",
                "needles_any": ["sqs", "sns", "kinesis", "eventbridge"],
                "forbidden_any": ["msk", "managed kafka", "amazon kafka", "confluent"],
            },
            {
                "id": "hipaa_encryption_audit",
                "description": "Mentions encryption (at rest / in transit / KMS / CMK) AND audit logging.",
                "needles_any": ["encrypt", "kms", "cmk", "tls"],
                "needles_secondary_any": [
                    "audit log",
                    "audit trail",
                    "cloudtrail",
                    "audit",
                ],
            },
            {
                "id": "multi_az",
                "description": "Mentions multi-AZ deployment / multi availability zone.",
                "needles_any": [
                    "multi-az",
                    "multi az",
                    "multiple az",
                    "availability zones",
                    "multi-availability-zone",
                ],
            },
            {
                "id": "aws_secrets_manager",
                "description": "Secrets via AWS Secrets Manager — NOT env vars / NOT param store / NOT plaintext.",
                "needles_any": ["secrets manager", "aws secrets manager"],
                "forbidden_any": [
                    "environment variable for secret",
                    "env var for secret",
                    "store secrets in env",
                    "secrets in environment variable",
                ],
            },
            {
                "id": "datadog_observability",
                "description": "Observability via Datadog — NOT a new tool like New Relic / Honeycomb / Grafana stack.",
                "needles_any": ["datadog"],
                "forbidden_any": [
                    "new relic",
                    "honeycomb",
                    "grafana stack",
                    "grafana cloud",
                    "splunk",
                ],
            },
        ],
        memory=M,
    )


# ----------------------------------------------------------------------------
# Scenario 3 — Family meal plan (Tuesday)
# ----------------------------------------------------------------------------


def _meal_case() -> TestCase:
    M: list[MemoryChunk] = []

    M.append(
        MemoryChunk(
            "meal_chat_002",
            "Earlier chat — family overview",
            "User (Jan 04): There are four of us at the table — me, my husband "
            "Raj, our 8-year-old Priya, and my dad who lives with us. Dad is "
            "78. Whatever you plan needs to work for all four of us at once.",
            tags=["family", "people"],
        )
    )
    M.append(
        MemoryChunk(
            "meal_chat_011",
            "Earlier chat — Dad's diabetes carb cap",
            "User (Jan 06): Dad has type-2 diabetes. His doctor put him on a "
            "STRICT carb limit: 45g of carbs per meal, max. We've been tracking "
            "this. Any meal you plan needs carb counts shown so I can verify "
            "for him.",
            tags=["dad", "diabetes", "carbs", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "meal_chat_018",
            "Earlier chat — Raj bariatric history",
            "User (Jan 09): Raj had bariatric surgery two years ago. He eats "
            "very small portions, prioritizes protein, keeps fat low, and can't "
            "tolerate sugary food (dumping syndrome). Plan small high-protein "
            "low-fat portions for him specifically.",
            tags=["raj", "bariatric", "constraint", "protein"],
        )
    )
    M.append(
        MemoryChunk(
            "meal_chat_027",
            "Earlier chat — vegetarian + Tue/Sat onion-garlic rule",
            "User (Jan 12): Cultural: we're lacto-vegetarian — no meat, no "
            "eggs, dairy is fine. Also we observe NO onion / NO garlic on "
            "Tuesdays and Saturdays. Religious reason. Other days are fine for "
            "onion and garlic.",
            tags=["vegetarian", "onion", "garlic", "tuesday", "saturday", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "meal_chat_039",
            "Earlier chat — Priya allergies and pickiness",
            "User (Jan 15): Priya has a tree-nut allergy: almonds, cashews, "
            "walnuts, pistachios, pecans, all of it. Peanuts are fine since "
            "those are legumes. She's also a stubborn picky eater — she will "
            "absolutely refuse anything obviously green (spinach, broccoli, "
            "raw greens).",
            tags=["priya", "tree-nut", "allergy", "picky", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "meal_chat_048",
            "Earlier chat — weeknight time + budget",
            "User (Jan 18): Practical: weeknight dinner I can spend 30 minutes "
            "tops. Weekend I can do longer. Grocery budget about $200/wk for "
            "the family.",
            tags=["weeknight", "30min", "budget", "constraint"],
        )
    )
    M.append(
        MemoryChunk(
            "meal_chat_054",
            "Earlier chat — kitchen equipment + Raj fast",
            "User (Jan 22): Equipment-wise: we have an Instant Pot and an "
            "oven. NO microwave (we got rid of it). One more thing: Raj is "
            "doing 16:8 intermittent fasting on weekdays — he SKIPS BREAKFAST "
            "Monday through Friday. Plan accordingly.",
            tags=["raj", "fasting", "breakfast", "weekday", "constraint", "equipment"],
        )
    )

    distractor_topics = [
        "Indian dal varieties primer",
        "Instant Pot pressure-cooking timing tables",
        "general meal-prep batch strategies",
        "freezer-to-table soup ideas (general)",
        "general weeknight curry templates",
        "yogurt-marinade techniques",
        "seasonal vegetable shopping notes",
        "common low-glycemic-index swaps",
        "kid-friendly Indian sweets generally",
        "ideas for tiffin-box lunches",
        "general history of South Indian breakfast",
        "Gujarati thali composition basics",
        "pickled vegetable preparation primer",
        "popular paneer dish overview",
        "rotis vs parathas culinary basics",
        "ghee vs butter vs olive oil notes",
        "general spice-grinding tips",
        "rice variety overview (basmati/jasmine/etc.)",
        "lentil cooking-time table generally",
        "yogurt-making fundamentals",
        "general fermentation basics",
        "Indian festival meal traditions overview",
        "fasting traditions (general religious overview)",
        "ayurvedic eating principles generally",
        "seasonal eating across India overview",
        "monsoon-season meal traditions",
        "general school-lunch packing tips",
        "knife sharpening 101",
        "kitchen storage organization general",
        "pantry rotation best practices",
        "induction vs gas vs electric stove",
        "cookware safety: nonstick caveats",
        "general food-safety temperature primer",
        "leftover-storage shelf-life chart",
        "general holiday-meal planning notes",
        "potluck-coordination etiquette",
    ]
    for i, topic in enumerate(distractor_topics):
        M.append(
            MemoryChunk(
                f"meal_doc_{100 + i:03d}",
                f"Cooking note — {topic}",
                _filler(topic, paragraphs=9, seed=i + 25),
                tags=topic.split()[:3],
            )
        )

    return TestCase(
        name="tuesday_meal_plan",
        domain="nutrition",
        task_brief=(
            "Build the family's TUESDAY meal plan. Output: breakfast (for "
            "those who eat it), lunch, and dinner — with carb counts and a "
            "note on per-person portions where relevant."
        ),
        plan_steps=[
            "Recall the family roster and any per-person medical/health constraints.",
            "Recall dietary identity (vegetarian, religious day-of-week rules).",
            "Recall allergies and pickiness for the kids.",
            "Recall Raj's bariatric protocol and any fasting schedule.",
            "Recall weeknight time budget and equipment constraints.",
        ],
        final_question=(
            "Now produce TUESDAY's full menu: breakfast (skip for anyone "
            "fasting), lunch, dinner — with carb counts per dish for Dad, "
            "and a portion note for Raj. Make sure every dish honors all "
            "the family's constraints."
        ),
        rubric=[
            {
                "id": "no_onion_garlic_tuesday",
                "description": "Tuesday: no onion, no garlic in any dish.",
                "needles_any": [
                    "no onion",
                    "without onion",
                    "no garlic",
                    "without garlic",
                    "onion-free",
                    "garlic-free",
                    "jain-style",
                    "no-onion-no-garlic",
                ],
            },
            {
                "id": "lacto_vegetarian",
                "description": "All dishes are vegetarian (no meat, no egg). Dairy is fine.",
                "needles_any": [
                    "vegetarian",
                    "no meat",
                    "no egg",
                    "lacto-vegetarian",
                    "veg ",
                ],
            },
            {
                "id": "carb_counts_present",
                "description": "Carb counts shown for the meals (so Dad can stay under 45g/meal).",
                "needles_any": [
                    "g carb",
                    "g of carbs",
                    "grams carb",
                    "carbs:",
                    "carb count",
                    "carbs ~",
                    "carb ~",
                    "≈",
                ],
            },
            {
                "id": "no_tree_nuts",
                "description": "No almond/cashew/walnut/pistachio/pecan in any dish.",
                "needles_any": [
                    "no tree nut",
                    "tree-nut-free",
                    "without tree nut",
                    "no almond",
                    "no cashew",
                    "no walnut",
                    "no pistachio",
                    "no pecan",
                ],
                "forbidden_any": ["almond", "cashew", "walnut", "pistachio", "pecan"],
                "forbidden_exceptions": [
                    "no almond",
                    "no cashew",
                    "no walnut",
                    "no pistachio",
                    "no pecan",
                    "tree-nut-free",
                    "without tree nut",
                    "no tree nut",
                ],
            },
            {
                "id": "raj_skips_breakfast",
                "description": "Tuesday is a weekday — Raj on 16:8 fast, breakfast skipped or noted as omitted.",
                "needles_any": [
                    "raj skip",
                    "raj omits",
                    "skip for raj",
                    "raj not eating",
                    "raj fasting",
                    "raj will skip",
                    "no breakfast for raj",
                    "skip breakfast",
                ],
            },
            {
                "id": "raj_high_protein_low_fat_small",
                "description": "Raj's portion noted as small / high-protein / low-fat.",
                "needles_any": [
                    "small portion",
                    "high-protein",
                    "high protein",
                    "low-fat",
                    "low fat",
                    "bariatric",
                ],
            },
        ],
        memory=M,
    )


SCENARIOS: list[TestCase] = [
    _travel_case(),
    _software_case(),
    _meal_case(),
]
