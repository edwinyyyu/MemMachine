"""Advanced synthetic conversations testing the hardest retrieval categories.

Categories:
- evolving_terminology: same concept referred to by different names over time
- negation: decisions where options were explicitly rejected
- unfinished_business: assigned tasks that were never followed up on
- perspective_separation: multiple people with different views on the same topic
- quantitative_aggregation: scattered numeric values needing collection
- frequency_detection: recurring patterns across the conversation
- constraint_propagation: changes in one domain affecting plans in another
- consistency_checking: contradictory information scattered across turns
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
# Conversation builders
# ---------------------------------------------------------------------------


def _build_evolving_terminology_1() -> list[dict]:
    """A software project that accumulates many names over 100 turns."""
    return [
        _t(
            0,
            "user",
            "Morning! Ready for a long week. We have that kickoff meeting for the customer portal redesign at 10.",
        ),
        _t(
            1,
            "assistant",
            "Good morning! Yes, the customer portal redesign is a big initiative. Are you leading it?",
        ),
        _t(
            2,
            "user",
            "Co-leading with Priya. She's handling the backend services and I've got the frontend and UX. We should have the full team in the room today.",
        ),
        _t(3, "assistant", "Sounds like a good split. Who else is on the team?"),
        _t(
            4,
            "user",
            "Let's see - there's Marcus on infrastructure, Jenny doing QA, and Ravi on the mobile side. Oh and we pulled in Diane from the analytics team part-time.",
        ),
        _t(
            5,
            "assistant",
            "Solid team. Six people should be enough to move quickly. What's the current portal's biggest problem?",
        ),
        _t(
            6,
            "user",
            "Honestly it's embarrassing. The login flow takes 4 clicks, the dashboard loads in 8 seconds on a good day, and nobody can find anything. Customer support tickets about the portal are through the roof.",
        ),
        _t(
            7,
            "assistant",
            "Those are clear pain points. The login flow and performance issues alone would justify the project.",
        ),
        _t(
            8,
            "user",
            "Exactly what we told the VP. Anyway the meeting went well. Priya suggested we call it Project Phoenix since we're basically rebuilding from the ashes of the old portal.",
        ),
        _t(
            9,
            "assistant",
            "I like it - Project Phoenix. Rising from the ashes of the legacy system. Did the team agree on a timeline?",
        ),
        _t(
            10,
            "user",
            "Yeah, 16 weeks. Aggressive but doable if we don't get pulled into other stuff.",
        ),
        _t(
            11, "assistant", "That's tight for a full portal rebuild. Are there phases?"
        ),
        _t(
            12,
            "user",
            "Three phases. Phase 1 is the new auth flow and dashboard, phase 2 is the reporting module, phase 3 is the self-service features.",
        ),
        _t(13, "assistant", "Good breakdown. When does phase 1 need to be done?"),
        _t(
            14,
            "user",
            "End of week 6. That's the part the VP cares about most because it's the most visible improvement.",
        ),
        _t(
            15,
            "assistant",
            "Makes sense - the login and dashboard improvements will be immediately noticeable to customers.",
        ),
        _t(
            16,
            "user",
            "Hey, completely different topic. Do you know anything about sourdough starters? My wife wants to try making bread.",
        ),
        _t(
            17,
            "assistant",
            "I do! A sourdough starter is a mix of flour and water that captures wild yeast. You feed it daily for about a week until it's bubbly and active.",
        ),
        _t(
            18,
            "user",
            "Cool. She's really into baking lately. Made this incredible cinnamon roll thing last weekend.",
        ),
        _t(
            19,
            "assistant",
            "That sounds delicious! Sourdough is a fun next step from regular baking.",
        ),
        _t(
            20,
            "user",
            "Yeah. Ok back to work stuff. Priya just Slacked me - she got the architecture doc drafted for Phoenix. She's proposing microservices instead of the monolith we have now.",
        ),
        _t(
            21,
            "assistant",
            "Microservices would make sense for a portal, especially if different teams will own different services. What's the current stack?",
        ),
        _t(
            22,
            "user",
            "Legacy Java monolith, JSP pages, Oracle database. It's ancient. For Phoenix she wants to go with Go microservices, React frontend, and Postgres.",
        ),
        _t(
            23,
            "assistant",
            "That's a significant modernization. The Go + React + Postgres stack is popular and well-supported.",
        ),
        _t(
            24,
            "user",
            "One concern I have is the migration. We have 50,000 active users and their data needs to come over cleanly.",
        ),
        _t(
            25,
            "assistant",
            "Data migration is always the riskiest part of a replatform. You'll want a parallel-run period where both systems are active.",
        ),
        _t(
            26,
            "user",
            "That's what Marcus suggested too. He's already looking into setting up the new infrastructure on AWS.",
        ),
        _t(
            27,
            "assistant",
            "Good to have the infrastructure planning start early. Is the current system on-prem or cloud?",
        ),
        _t(
            28,
            "user",
            "On-prem, which is part of the problem. Moving to AWS gives us autoscaling and better reliability.",
        ),
        _t(
            29,
            "assistant",
            "Definitely. The performance improvements from cloud hosting alone might help with that 8-second dashboard load time.",
        ),
        _t(
            30,
            "user",
            "Quick question - do you remember the name of that book about distributed systems someone recommended to me?",
        ),
        _t(
            31,
            "assistant",
            "I don't have that in our conversation history. Was it 'Designing Data-Intensive Applications' by Martin Kleppmann? That's a very popular one.",
        ),
        _t(
            32,
            "user",
            "Yes that's the one! I should buy that. Anyway, we had our first sprint planning for Phoenix today. Priya ran it well.",
        ),
        _t(33, "assistant", "Nice! What are the first sprint goals?"),
        _t(
            34,
            "user",
            "Sprint 1 is all about the auth service. Single sign-on, OAuth2, the whole nine yards. Jenny is going to set up the test framework in parallel.",
        ),
        _t(
            35,
            "assistant",
            "Starting with auth makes sense - everything else depends on it.",
        ),
        _t(
            36,
            "user",
            "So people on the team are already shortening Phoenix to just 'PHX' in Slack. Like 'PHX standup in 5' or 'PHX-123 is blocked.'",
        ),
        _t(
            37,
            "assistant",
            "That's typical for project names - they always get abbreviated. PHX is concise for Jira tickets and Slack.",
        ),
        _t(
            38,
            "user",
            "Yeah Ravi even made a custom Slack emoji for it. A little phoenix bird. The team is really into this.",
        ),
        _t(
            39,
            "assistant",
            "Team morale and identity around a project makes a big difference. Sounds like people are genuinely excited.",
        ),
        _t(
            40,
            "user",
            "They are. Oh, I need to tell you about my weekend plans. We're taking the kids to the science museum on Saturday.",
        ),
        _t(41, "assistant", "That sounds fun! Which museum?"),
        _t(
            42,
            "user",
            "The natural history one downtown. They have a new dinosaur exhibit my daughter is obsessed with.",
        ),
        _t(43, "assistant", "Kids and dinosaurs - a classic combo. How old is she?"),
        _t(
            44,
            "user",
            "She's 7. My son is 4 and he's more into the space stuff. We'll probably split up at some point.",
        ),
        _t(
            45, "assistant", "Divide and conquer! Each kid gets their favorite exhibit."
        ),
        _t(
            46,
            "user",
            "Exactly. OK so quick Phoenix update - we hit our first real blocker. The OAuth2 integration with the corporate identity provider is way more complicated than we thought.",
        ),
        _t(
            47,
            "assistant",
            "What's the complication? Is it the identity provider itself or the configuration?",
        ),
        _t(
            48,
            "user",
            "It's the identity provider. They're using an old version of SAML and the bridge to OAuth2 is flaky. Marcus is working with their team but it's slow going.",
        ),
        _t(
            49,
            "assistant",
            "Identity provider issues can really slow down an auth project. Is there a fallback plan?",
        ),
        _t(
            50,
            "user",
            "We might have to build a custom adapter. Priya is not thrilled about that but it might be the only option.",
        ),
        _t(
            51,
            "assistant",
            "A custom adapter adds technical debt but sometimes it's necessary to keep moving. How much time would it add?",
        ),
        _t(
            52,
            "user",
            "Marcus estimates about a week of work. Which would put sprint 1 behind.",
        ),
        _t(
            53,
            "assistant",
            "A week delay on sprint 1 could cascade to the phase 1 deadline. Worth raising early with stakeholders.",
        ),
        _t(
            54,
            "user",
            "Yeah. So the VP came to our standup today asking about 'the new portal' - she never uses the Phoenix name, just calls it 'the new portal' or 'the portal project.'",
        ),
        _t(
            55,
            "assistant",
            "Executives tend to use straightforward names. As long as everyone knows what she's referring to.",
        ),
        _t(
            56,
            "user",
            "It gets confusing sometimes because we also have a vendor portal that's being updated separately. When she says 'the portal' people have to figure out which one from context.",
        ),
        _t(
            57,
            "assistant",
            "That could lead to miscommunication. Maybe clarifying in meetings would help - 'customer portal' vs 'vendor portal.'",
        ),
        _t(
            58,
            "user",
            "Good point. So remember that OAuth problem? We found a workaround. There's an open source library that handles the SAML-to-OAuth bridge. Marcus tested it and it works.",
        ),
        _t(59, "assistant", "That's a relief. Which library?"),
        _t(
            60,
            "user",
            "Something called saml2-bridge. It's maintained by a company called AuthForge. Marcus is happy with the quality.",
        ),
        _t(
            61,
            "assistant",
            "Great find. Using a maintained library is much better than building a custom adapter.",
        ),
        _t(62, "user", "Agreed. Crisis averted. Sprint 1 is back on track."),
        _t(63, "assistant", "Excellent. So the phase 1 deadline should still hold?"),
        _t(
            64,
            "user",
            "Looking that way. Oh hey, unrelated but I need to book flights for Thanksgiving. We're going to my parents' place in Phoenix, Arizona. Funny coincidence with the project name.",
        ),
        _t(
            65,
            "assistant",
            "Ha! Phoenix the city and Phoenix the project. When are you flying out?",
        ),
        _t(
            66,
            "user",
            "Wednesday before Thanksgiving, back Sunday. My parents have a nice place in Scottsdale, which is basically Phoenix.",
        ),
        _t(
            67,
            "assistant",
            "Nice! Should be good weather that time of year too. Want me to help compare flight options?",
        ),
        _t(
            68,
            "user",
            "Nah I'll handle it. Back to work - so the team has started calling the project 'v2' informally. Like 'is that a v1 bug or a v2 thing?' when discussing issues.",
        ),
        _t(
            69,
            "assistant",
            "That makes sense - v1 being the existing legacy portal and v2 being Phoenix. Simple shorthand.",
        ),
        _t(
            70,
            "user",
            "Exactly. The product manager wrote a blog post for the company newsletter about 'Portal 2.0' which is yet ANOTHER name for it. I swear this project has more names than a con artist.",
        ),
        _t(
            71,
            "assistant",
            "Ha! Let me count: customer portal redesign, Project Phoenix, PHX, the new portal, v2, Portal 2.0. That's six names for one project.",
        ),
        _t(
            72,
            "user",
            "And Ravi just started calling it 'the bird' because of the phoenix emoji. Seven names.",
        ),
        _t(
            73,
            "assistant",
            "At some point you might need a glossary just for the project name.",
        ),
        _t(
            74,
            "user",
            "Seriously. OK so we had the phase 1 demo today. Dashboard loads in under 2 seconds now - down from 8. The VP was impressed.",
        ),
        _t(
            75,
            "assistant",
            "That's a massive improvement! A 4x speed boost is the kind of thing customers will immediately notice.",
        ),
        _t(
            76,
            "user",
            "She said, and I quote, 'this is exactly what the portal project needed.' Still won't call it Phoenix.",
        ),
        _t(
            77,
            "assistant",
            "At least she's happy with the results, whatever she calls it!",
        ),
        _t(
            78,
            "user",
            "So now we're moving into phase 2. The reporting module. This is Diane's time to shine since she's from analytics.",
        ),
        _t(
            79,
            "assistant",
            "Perfect timing to have an analytics person on the team. What kind of reports do customers need?",
        ),
        _t(
            80,
            "user",
            "Usage dashboards, billing summaries, and exportable data tables. The current portal has reporting but it's clunky and the exports break in Excel half the time.",
        ),
        _t(
            81,
            "assistant",
            "Broken Excel exports are a classic complaint. Getting that right will reduce a lot of support tickets.",
        ),
        _t(
            82,
            "user",
            "My grocery list for tonight - can you help me remember? Milk, eggs, bread, that fancy cheese my wife likes - the gouda from the Dutch store.",
        ),
        _t(
            83,
            "assistant",
            "Got it: milk, eggs, bread, and gouda from the Dutch store. Anything else?",
        ),
        _t(
            84, "user", "Oh and coffee filters. We're almost out. The size 4 cone ones."
        ),
        _t(
            85,
            "assistant",
            "Added: size 4 cone coffee filters. So your list is milk, eggs, bread, gouda, and coffee filters.",
        ),
        _t(
            86,
            "user",
            "Perfect thanks. Oh by the way, in the Phoenix standup today Priya mentioned we need to update the project charter since the scope expanded slightly.",
        ),
        _t(87, "assistant", "Scope expansion already? What changed?"),
        _t(
            88,
            "user",
            "The VP wants us to add a notification center to phase 3. Not huge but it's new work. Priya wants to make sure it's documented.",
        ),
        _t(
            89,
            "assistant",
            "Smart to document scope changes formally. A notification center is a meaningful addition.",
        ),
        _t(
            90,
            "user",
            "Yeah. So the latest from the bird is that we're on track for phase 2 completion by week 11.",
        ),
        _t(
            91,
            "assistant",
            "Good progress. That would leave 5 weeks for phase 3 plus the new notification center feature.",
        ),
        _t(
            92,
            "user",
            "Which is tight with the added scope. We might need to negotiate on the self-service features in phase 3.",
        ),
        _t(
            93,
            "assistant",
            "Makes sense to prioritize. What's the minimum viable set of self-service features?",
        ),
        _t(
            94,
            "user",
            "Password reset and profile management. The fancier stuff like subscription management can wait for a fast-follow after launch.",
        ),
        _t(
            95,
            "assistant",
            "Password reset and profile management are the essentials. Subscription management as a fast-follow is reasonable.",
        ),
        _t(
            96,
            "user",
            "That's the plan. Ravi already has a prototype of the mobile responsive dashboard for v2 and it looks slick.",
        ),
        _t(
            97,
            "assistant",
            "Mobile responsiveness is critical for a modern portal. Good that Ravi is ahead of it.",
        ),
        _t(
            98,
            "user",
            "Yeah. Quick question - is there anything we discussed about Phoenix that I should bring up in tomorrow's steering committee?",
        ),
        _t(
            99,
            "assistant",
            "Key points: phase 1 is complete with the dashboard loading 4x faster, phase 2 reporting module is on track for week 11, scope expanded to include a notification center in phase 3, and you may need to defer subscription management to a post-launch release.",
        ),
        _t(
            100,
            "user",
            "Perfect summary. I'll present that. Oh and I should mention the OAuth issue we resolved - it shows the team can problem-solve.",
        ),
        _t(
            101,
            "assistant",
            "Good idea. The SAML-to-OAuth bridge solution shows technical resourcefulness. Also worth mentioning the team's morale is high.",
        ),
        _t(102, "user", "Great. Talk to you later!"),
        _t(103, "assistant", "Good luck with the steering committee!"),
    ]


def _build_evolving_terminology_2() -> list[dict]:
    """A bug that accumulates many names over 90 turns."""
    return [
        _t(
            0,
            "user",
            "Hey, got a weird issue. Users are reporting they get kicked out of the app randomly. Just started this morning.",
        ),
        _t(1, "assistant", "That sounds urgent. What kind of app is this?"),
        _t(
            2,
            "user",
            "Our internal HR tool. About 300 people use it daily. The login issue started around 9am according to the first reports.",
        ),
        _t(
            3,
            "assistant",
            "Any recent deployments or infrastructure changes that could have caused it?",
        ),
        _t(
            4,
            "user",
            "We deployed a patch last night for the timesheet module. But the login system wasn't touched.",
        ),
        _t(
            5,
            "assistant",
            "Sometimes patches have unexpected side effects. Have you checked the logs?",
        ),
        _t(
            6,
            "user",
            "Kai is looking at the logs now. He's our most senior backend person. In the meantime we've gotten 15 more tickets about it.",
        ),
        _t(
            7,
            "assistant",
            "15 tickets in a short time suggests it's widespread, not an edge case. Is it affecting all users or a subset?",
        ),
        _t(
            8,
            "user",
            "Good question. Let me check with support... Ok so it seems like it's mainly people in the Chicago and London offices. New York seems fine.",
        ),
        _t(
            9,
            "assistant",
            "Geographic pattern is interesting. Could be related to load balancing or a specific server instance serving those regions.",
        ),
        _t(
            10,
            "user",
            "That's what Kai thinks too. He just pinged me - he found something. The session tokens are expiring way too fast. Like, after 2-3 minutes instead of the normal 30 minutes.",
        ),
        _t(
            11,
            "assistant",
            "Premature session expiry would definitely cause random logouts. Was the session timeout configuration changed?",
        ),
        _t(
            12,
            "user",
            "Not intentionally. But here's the thing - it's not all sessions. Some people are fine. It's like a coin flip.",
        ),
        _t(
            13,
            "assistant",
            "Inconsistent behavior like that often points to a load balancer issue - some requests hitting one server with correct config, others hitting one with bad config.",
        ),
        _t(
            14,
            "user",
            "Hold on, completely different thing - my daughter has a piano recital tonight. Do you remember what time I said it was?",
        ),
        _t(
            15,
            "assistant",
            "I don't have that in our conversation history. Did you mention it recently?",
        ),
        _t(
            16,
            "user",
            "No I guess not. It's at 6pm at the school auditorium. I need to leave work by 5:15 to make it.",
        ),
        _t(
            17,
            "assistant",
            "Got it - piano recital at 6pm, leave by 5:15. I'll remind you later if you'd like.",
        ),
        _t(
            18,
            "user",
            "Please. Ok so the timeout thing - Kai found it. The last deployment accidentally included a config file that was meant for the staging environment. Staging has a 3-minute session timeout for testing purposes.",
        ),
        _t(
            19,
            "assistant",
            "Ah, a classic config leak from staging to production. That explains the short timeouts. But why is it only affecting some users?",
        ),
        _t(
            20,
            "user",
            "Because the deployment went to 2 of our 4 app server pods. The load balancer round-robins between them. So roughly half the requests hit the bad config.",
        ),
        _t(
            21,
            "assistant",
            "That perfectly explains the coin-flip behavior. Fix would be to redeploy the correct config to those two pods.",
        ),
        _t(
            22,
            "user",
            "Already on it. Kai's pushing the fix now. Should be live in 10 minutes.",
        ),
        _t(
            23,
            "assistant",
            "Quick turnaround. Are you going to do a post-mortem on this one?",
        ),
        _t(
            24,
            "user",
            "Yeah we have to. This is the kind of thing that shouldn't happen. We need better config management between environments.",
        ),
        _t(
            25,
            "assistant",
            "Agreed. Environment-specific configs leaking to production is a process issue worth addressing.",
        ),
        _t(
            26,
            "user",
            "So the fix is live. Kai confirmed sessions are back to 30-minute timeout. But now people are complaining about something new.",
        ),
        _t(27, "assistant", "Oh no. What's the new issue?"),
        _t(
            28,
            "user",
            "The timesheet submission button is timing out. When you click submit, it spins for about 60 seconds and then shows an error. This one IS related to last night's patch.",
        ),
        _t(
            29,
            "assistant",
            "That lines up with the timesheet module patch from last night. What was the patch supposed to fix?",
        ),
        _t(
            30,
            "user",
            "It was supposed to fix a calculation error in overtime hours. But it looks like it introduced a performance regression.",
        ),
        _t(31, "assistant", "Did someone already file a ticket for this?"),
        _t(
            32,
            "user",
            "Yeah, Nadia from support created JIRA-4521 for it. She's triaging all the timesheet reports.",
        ),
        _t(
            33,
            "assistant",
            "JIRA-4521 for the timesheet timeout. Is Kai investigating this one too?",
        ),
        _t(
            34,
            "user",
            "He handed it off to Sara since she wrote the original patch. She's looking at it now.",
        ),
        _t(
            35,
            "assistant",
            "Makes sense to have the patch author investigate. Any initial findings?",
        ),
        _t(
            36,
            "user",
            "Sara says the query she added for the overtime calculation is doing a full table scan on the timesheet_entries table. That table has 2 million rows.",
        ),
        _t(
            37,
            "assistant",
            "A full table scan on 2 million rows would absolutely cause a 60-second timeout. Missing index?",
        ),
        _t(
            38,
            "user",
            "Exactly. She forgot to add an index for the new query pattern. She's writing the migration now.",
        ),
        _t(
            39,
            "assistant",
            "A simple index should fix the performance. How long until it can be deployed?",
        ),
        _t(
            40,
            "user",
            "She says the migration will take about 20 minutes to run on the production database. But she wants to test it in staging first, which adds another hour.",
        ),
        _t(
            41,
            "assistant",
            "Testing in staging is the right call, especially given that a staging config just leaked to production. Don't want to rush and cause another issue.",
        ),
        _t(
            42,
            "user",
            "Ironic, right? Ok so while we wait, let me ask you about something else. I'm thinking about getting a new grill for the backyard. Any recommendations?",
        ),
        _t(
            43,
            "assistant",
            "Depends on your preference - gas, charcoal, or pellet? Each has its pros and cons.",
        ),
        _t(
            44,
            "user",
            "I've always used gas but I'm curious about pellet grills. I hear they're good for smoking.",
        ),
        _t(
            45,
            "assistant",
            "Pellet grills are excellent for smoking and they offer consistent temperature control. Traeger and Weber both make popular models.",
        ),
        _t(
            46,
            "user",
            "I'll look into those. Hey Sara just Slacked the team. She says the index migration worked great in staging. Query time went from 58 seconds to 0.2 seconds.",
        ),
        _t(
            47,
            "assistant",
            "Dramatic improvement! That should completely resolve the timesheet timeout. When is she deploying to production?",
        ),
        _t(
            48,
            "user",
            "Starting now. She needs a 20-minute maintenance window for the migration.",
        ),
        _t(
            49,
            "assistant",
            "Are you going to notify users about the maintenance window?",
        ),
        _t(
            50,
            "user",
            "Already did. Nadia sent out an email. People are grumbling but they understand.",
        ),
        _t(
            51,
            "assistant",
            "Better a planned 20-minute window than ongoing timeouts. Is this being tracked as part of JIRA-4521 still?",
        ),
        _t(
            52,
            "user",
            "Yeah, Sara's updating the ticket with all her findings. By the way, the team has started calling this 'the monster bug' because it took down two things in one morning - the login issue AND the timesheet thing.",
        ),
        _t(
            53,
            "assistant",
            "That's a memorable name for a bad day. Though technically it was two separate issues that happened to coincide.",
        ),
        _t(
            54,
            "user",
            "True but it FEELS like one big monster. Same deployment triggered both problems, just in different ways. The staging config caused the session issue and the missing index caused the timeout.",
        ),
        _t(
            55,
            "assistant",
            "Good point - same root cause in the deployment process, just different manifestations. The post-mortem should cover both.",
        ),
        _t(
            56,
            "user",
            "So Sara's migration is done. Timesheet submissions are working again. But now she noticed something weird in the logs - there are a bunch of failed API calls to the payroll service.",
        ),
        _t(57, "assistant", "Related to the timesheet changes or a separate issue?"),
        _t(
            58,
            "user",
            "She thinks related. When the timesheets were timing out, some partial data got written to the queue that feeds the payroll system. Now the payroll service is choking on malformed records.",
        ),
        _t(
            59,
            "assistant",
            "So the monster bug has a third head. The timesheet timeout caused data corruption in the payroll queue.",
        ),
        _t(
            60,
            "user",
            "Perfect metaphor. Three heads: login sessions, timesheet timeout, payroll queue corruption. All from one deployment. This is going to be a legendary post-mortem.",
        ),
        _t(
            61,
            "assistant",
            "It really is a hydra of a bug. Is the payroll queue issue critical? Could it affect people's paychecks?",
        ),
        _t(
            62,
            "user",
            "Potentially, yes. Payroll runs on Friday. Sara and Kai are both on the payroll queue issue now. They need to clean up the bad records.",
        ),
        _t(
            63,
            "assistant",
            "High priority then. Do they know how many records are corrupted?",
        ),
        _t(
            64,
            "user",
            "About 200 records out of the 1,500 submitted today. They can identify them by the timestamp range during the outage.",
        ),
        _t(
            65,
            "assistant",
            "That's manageable. Filter by the outage window and reprocess those submissions.",
        ),
        _t(
            66,
            "user",
            "That's the plan. Sara's writing a cleanup script. Kai is monitoring to make sure no new bad records are being created.",
        ),
        _t(67, "assistant", "Good division of labor. Any ETA on the cleanup?"),
        _t(
            68,
            "user",
            "Sara says about 2 hours. She wants to be very careful since it's payroll data.",
        ),
        _t(
            69,
            "assistant",
            "Absolutely - payroll data errors can have real consequences for employees. Better to be thorough.",
        ),
        _t(
            70,
            "user",
            "Hey I need to grab lunch. Can you remind me about my daughter's recital when I get back?",
        ),
        _t(
            71,
            "assistant",
            "Of course! Enjoy your lunch. I'll remind you about the 6pm piano recital - leave by 5:15.",
        ),
        _t(72, "user", "Back from lunch. What's the status on that timeout thing?"),
        _t(
            73,
            "assistant",
            "The timesheet timeout (JIRA-4521) is fixed - Sara added the missing index. But they discovered the outage caused about 200 corrupted records in the payroll queue. Sara is writing a cleanup script, estimated 2 hours. Kai is monitoring for new issues.",
        ),
        _t(
            74,
            "user",
            "Right. Sara just told me the cleanup is about halfway done. She found a few more records than expected - 230 total.",
        ),
        _t(
            75,
            "assistant",
            "Slightly more than the initial 200 estimate, but the same order of magnitude. Still on track for the 2-hour estimate?",
        ),
        _t(
            76,
            "user",
            "She says yes. Oh, Nadia wants to know if she should close JIRA-4521 or keep it open.",
        ),
        _t(
            77,
            "assistant",
            "I'd keep it open until the payroll cleanup is verified. The timeout fix is in, but the downstream effects are still being resolved.",
        ),
        _t(
            78,
            "user",
            "Good call. I told her to keep it open and add the payroll cleanup as a sub-task.",
        ),
        _t(
            79,
            "assistant",
            "Smart tracking. Will the post-mortem document be linked to the JIRA ticket too?",
        ),
        _t(
            80,
            "user",
            "Yeah I'll write it up tomorrow. Today is too chaotic. Let me tell you about the meeting I had over lunch with marketing about the Q4 campaign.",
        ),
        _t(81, "assistant", "Sure, go ahead. Different topic entirely?"),
        _t(
            82,
            "user",
            "Totally different. They want a landing page for the holiday sale. Standard stuff, needs to be done by November 15th.",
        ),
        _t(
            83,
            "assistant",
            "Straightforward enough. Is your team building it or is it outsourced?",
        ),
        _t(
            84,
            "user",
            "My team. It'll be Dev since he does most of our marketing pages. Anyway, back to the monster - Sara just finished the payroll cleanup.",
        ),
        _t(85, "assistant", "All 230 records cleaned up?"),
        _t(
            86,
            "user",
            "Yep. She reprocessed them and verified against the source timesheet data. Everything matches now. Kai confirmed the payroll service is processing normally.",
        ),
        _t(
            87,
            "assistant",
            "Great work by both of them. So all three heads of the monster bug are now resolved: session timeouts, timesheet performance, and payroll data cleanup.",
        ),
        _t(
            88,
            "user",
            "Finally. What a day. Oh - don't forget to remind me about the recital!",
        ),
        _t(
            89,
            "assistant",
            "It's currently 4:45pm - you need to leave by 5:15 for the 6pm piano recital at the school auditorium. You have about 30 minutes.",
        ),
        _t(
            90,
            "user",
            "Perfect, wrapping up now. One more thing - should we rollback the overtime calculation patch entirely, or is the indexed version OK?",
        ),
        _t(
            91,
            "assistant",
            "The indexed version should be fine. The overtime calculation itself works correctly - it was just the missing index causing the performance problem. Rolling back would reintroduce the overtime calculation error that the patch was meant to fix.",
        ),
        _t(
            92,
            "user",
            "Makes sense. Sara confirmed the overtime numbers are correct now. OK I'm heading out. Monster bug is slain.",
        ),
        _t(
            93,
            "assistant",
            "Enjoy the recital! And yes, the monster is down. Don't forget to write that post-mortem tomorrow.",
        ),
    ]


def _build_negation_1() -> list[dict]:
    """Team evaluates and rejects framework options for a new project."""
    return [
        _t(
            0,
            "user",
            "Team meeting notes from this morning. We need to pick a frontend framework for the dashboard project.",
        ),
        _t(1, "assistant", "Important decision. What are the candidates?"),
        _t(
            2,
            "user",
            "So we started with a long list and narrowed it down. The main contenders are React, Vue, Svelte, and Angular.",
        ),
        _t(3, "assistant", "Four strong options. What are the evaluation criteria?"),
        _t(
            4,
            "user",
            "Performance, team familiarity, ecosystem maturity, and hiring ease. We want to be able to bring on contractors if needed.",
        ),
        _t(
            5,
            "assistant",
            "Good criteria. How does the team's current experience break down?",
        ),
        _t(
            6,
            "user",
            "Most of the team knows React well. Two people have used Vue. Nobody has production Svelte experience. Everyone's done at least some Angular.",
        ),
        _t(
            7,
            "assistant",
            "React has the clear familiarity advantage then. What about the performance requirements?",
        ),
        _t(
            8,
            "user",
            "The dashboard needs to handle real-time data updates - think stock ticker style, with hundreds of data points refreshing every second. Performance is critical.",
        ),
        _t(
            9,
            "assistant",
            "That's a demanding use case. Svelte's compiled approach could be an advantage there. React with proper optimization can handle it too.",
        ),
        _t(
            10,
            "user",
            "That's exactly what came up. So let me tell you about each option and what the team thought.",
        ),
        _t(11, "assistant", "Go ahead."),
        _t(
            12,
            "user",
            "Angular first. Derek made the case for it. He said the opinionated structure would help with code consistency since we have 8 developers. He also pointed out that Angular has built-in RxJS support which is great for reactive data streams.",
        ),
        _t(
            13,
            "assistant",
            "Valid points. RxJS is excellent for real-time data handling. What was the pushback?",
        ),
        _t(
            14,
            "user",
            "Lisa pointed out that Angular's bundle size is significantly larger than the alternatives, which matters for our use case since the dashboard needs to load fast on low-bandwidth connections - some of our users are on field devices with spotty cellular.",
        ),
        _t(
            15,
            "assistant",
            "That's a real constraint. Angular's initial bundle can be heavy.",
        ),
        _t(
            16,
            "user",
            "And then Tomoko mentioned that Angular's learning curve is steep for anyone who doesn't already know it well. We have two junior devs joining next month and onboarding them on Angular would take longer.",
        ),
        _t(
            17,
            "assistant",
            "So Angular has strengths in structure and reactive data, but weaknesses in bundle size and learning curve. Where did it land?",
        ),
        _t(
            18,
            "user",
            "We decided against Angular. The bundle size issue was the killer. With field devices on cellular connections, we can't afford that overhead. Derek was disappointed but understood.",
        ),
        _t(
            19,
            "assistant",
            "Fair decision given the bandwidth constraint. What about Vue?",
        ),
        _t(
            20,
            "user",
            "Oh wait, before that - can you remind me what the deadline is for the Q3 budget submission? I keep forgetting.",
        ),
        _t(
            21,
            "assistant",
            "I don't have that date in our conversation history. Do you want me to note it when you find out?",
        ),
        _t(
            22,
            "user",
            "It's next Friday. OK so Vue. Priya presented the case for Vue. She argued that its reactivity system is elegant and would map well to our data flow needs. She also noted the Composition API in Vue 3 is very similar to React hooks, so the learning curve for the React-experienced people would be minimal.",
        ),
        _t(
            23,
            "assistant",
            "Good argument. Vue 3's Composition API does bridge the gap nicely. What were the concerns?",
        ),
        _t(
            24,
            "user",
            "The main concern was ecosystem maturity for our specific needs. We need a high-performance data grid component. React has AG Grid, Tanstack Table, and others that are battle-tested with millions of rows. Vue's equivalents exist but they're less mature.",
        ),
        _t(
            25,
            "assistant",
            "Data grid ecosystem is a real differentiator for dashboards. What else?",
        ),
        _t(
            26,
            "user",
            "Marcus raised the hiring point. He said if we need to bring on contractors quickly, finding experienced Vue developers is harder than React. Not impossible, just a smaller pool.",
        ),
        _t(
            27,
            "assistant",
            "That's a practical concern, especially if the project needs to scale up quickly.",
        ),
        _t(
            28,
            "user",
            "Yeah. And then Lisa looked up the npm download stats. React gets about 20 million weekly downloads versus Vue's 4 million. That's a rough proxy for ecosystem health.",
        ),
        _t(
            29,
            "assistant",
            "A 5x difference in download volume does suggest a significantly larger ecosystem. So where did Vue land?",
        ),
        _t(
            30,
            "user",
            "We ruled out Vue as well. Priya argued hard for it but the data grid ecosystem gap and the contractor pool size were too significant. She proposed we revisit Vue for the next greenfield project that doesn't need heavy data visualization.",
        ),
        _t(
            31,
            "assistant",
            "A fair compromise. Save Vue for a project where its advantages matter more and its gaps matter less.",
        ),
        _t(
            32,
            "user",
            "Exactly. So now it was down to React and Svelte. This is where it got heated.",
        ),
        _t(
            33,
            "assistant",
            "The two with the strongest performance stories. What were the arguments?",
        ),
        _t(
            34,
            "user",
            "Tomoko championed Svelte. Her main pitch was performance. Svelte compiles to vanilla JS with no virtual DOM overhead. For our real-time data scenario, she showed benchmarks where Svelte was 3x faster than React on rapid DOM updates.",
        ),
        _t(
            35,
            "assistant",
            "Svelte's compilation approach is genuinely faster for fine-grained reactivity. Those benchmarks are compelling.",
        ),
        _t(
            36,
            "user",
            "She also showed a prototype she built over the weekend. A mini dashboard with 500 updating data points. It was buttery smooth even on her old phone.",
        ),
        _t(
            37,
            "assistant",
            "Real prototype evidence is powerful. That's going above and beyond for the evaluation.",
        ),
        _t(
            38,
            "user",
            "It was impressive. But then Marcus raised some serious concerns. First, nobody on the team has production Svelte experience. We'd all be learning on the job.",
        ),
        _t(39, "assistant", "That's a risk on a timeline-critical project. What else?"),
        _t(
            40,
            "user",
            "Second, the Svelte ecosystem for enterprise-grade components is thin. There's no equivalent to AG Grid for Svelte that handles millions of rows with virtual scrolling. We'd have to build our own data grid or use a framework-agnostic one.",
        ),
        _t(
            41,
            "assistant",
            "Building a custom high-performance data grid is a project in itself. That could blow the timeline.",
        ),
        _t(
            42,
            "user",
            "That was Lisa's point exactly. She estimated building a custom grid would add 6-8 weeks to the project, and we only have 12 weeks total.",
        ),
        _t(
            43,
            "assistant",
            "That's 50-65% of the project timeline on a single component. Hard to justify.",
        ),
        _t(
            44,
            "user",
            "And third - and this was the one that really swung it - Derek asked about Svelte's story for server-side rendering and incremental static regeneration. SvelteKit handles it, but he pointed out that SvelteKit was only recently stable and has less battle-testing in production at scale than Next.js.",
        ),
        _t(
            45,
            "assistant",
            "The maturity gap between SvelteKit and Next.js is real, especially for production at scale.",
        ),
        _t(
            46,
            "user",
            "Tomoko pushed back. She said the performance benefits were so significant that they were worth the risk. She proposed a 2-week spike to validate the Svelte approach, including building a data grid prototype.",
        ),
        _t(
            47,
            "assistant",
            "A spike is a reasonable risk-mitigation approach. How did the team react?",
        ),
        _t(
            48,
            "user",
            "Marcus said we couldn't afford 2 weeks of exploration on a 12-week timeline. And if the spike fails, we've wasted that time and still have to use React.",
        ),
        _t(
            49,
            "assistant",
            "He has a point about the time budget. A failed spike would be costly.",
        ),
        _t(
            50,
            "user",
            "So in the end, we chose React. The ecosystem maturity, team familiarity, hiring pool, and the availability of AG Grid were decisive. We'll use React with Next.js for the framework and AG Grid Enterprise for the data tables.",
        ),
        _t(
            51,
            "assistant",
            "A pragmatic choice. React may not win on raw performance but it wins on risk mitigation and ecosystem.",
        ),
        _t(
            52,
            "user",
            "Tomoko was visibly frustrated. She thinks we're making a safe but suboptimal choice. She says in two years we'll wish we'd picked Svelte when we're fighting React re-render issues.",
        ),
        _t(
            53,
            "assistant",
            "Her concern isn't unfounded. React's re-render behavior can be challenging with high-frequency updates. Did you agree on a mitigation strategy?",
        ),
        _t(
            54,
            "user",
            "Yeah - we agreed to invest in proper React optimization patterns from day one. React.memo, useMemo, useCallback, and potentially using a state management library like Zustand that can minimize re-renders.",
        ),
        _t(
            55,
            "assistant",
            "Good plan. Proactive optimization is better than retrofitting it later.",
        ),
        _t(
            56,
            "user",
            "Derek also suggested we consider React Server Components for the less dynamic parts of the dashboard. That would give us some of the SSR benefits without the overhead.",
        ),
        _t(
            57,
            "assistant",
            "Smart - use RSC for the static/slow-changing content and client components for the real-time data widgets.",
        ),
        _t(
            58,
            "user",
            "Exactly. OK so the decision is made. React with Next.js and AG Grid Enterprise. Can you summarize the decision and the rationale? I need to write it up for the architecture decision record.",
        ),
        _t(
            59,
            "assistant",
            "Decision: React with Next.js (framework) and AG Grid Enterprise (data grid). Rationale: Team has strong React familiarity, largest ecosystem for enterprise components, best contractor hiring pool, AG Grid provides battle-tested high-performance data visualization. Trade-offs accepted: React's virtual DOM overhead for rapid updates, mitigated by optimization patterns (React.memo, useMemo, Zustand) and React Server Components for non-dynamic content.",
        ),
        _t(
            60,
            "user",
            "Perfect. Now I need to order lunch. What's good around here that delivers?",
        ),
        _t(
            61,
            "assistant",
            "I'd need to know your location to suggest delivery options. Any cuisine preferences?",
        ),
        _t(62, "user", "I'm in downtown Portland. And I want something healthy-ish."),
        _t(
            63,
            "assistant",
            "Portland has great options - maybe a poke bowl or a grain bowl place?",
        ),
        _t(
            64,
            "user",
            "Poke sounds good. Let me look on DoorDash. Anyway, one more thing from the meeting - we also discussed CSS approaches.",
        ),
        _t(
            65,
            "assistant",
            "CSS is another decision with many options these days. What came up?",
        ),
        _t(
            66,
            "user",
            "Tailwind CSS vs CSS Modules vs styled-components. We picked Tailwind. Lisa argued for styled-components because of better component encapsulation, but we decided Tailwind's utility-first approach is faster for building dashboard layouts.",
        ),
        _t(
            67,
            "assistant",
            "Tailwind is popular for dashboards because of its responsive utilities and design token system. Lisa's point about encapsulation is valid but Tailwind's @apply directive can help there.",
        ),
        _t(
            68,
            "user",
            "Yeah. Marcus actually wanted to use vanilla CSS with BEM naming, which is kind of old school, but nobody else was on board.",
        ),
        _t(
            69,
            "assistant",
            "BEM is reliable but does require more manual discipline. Tailwind automates a lot of what BEM provides structurally.",
        ),
        _t(
            70,
            "user",
            "For the record, we also briefly considered Emotion but dropped it since it has similar trade-offs to styled-components and the team preferred Tailwind anyway.",
        ),
        _t(
            71,
            "assistant",
            "So the CSS decision was clearer than the framework one - Tailwind won with less debate.",
        ),
        _t(
            72,
            "user",
            "Much less. Oh I also need to book travel for the conference next month. Have you heard of ReactConf?",
        ),
        _t(
            73,
            "assistant",
            "Yes! React Conf is a major conference organized by the React team. It's a great opportunity to stay current on the ecosystem, especially now that you've committed to React.",
        ),
        _t(
            74,
            "user",
            "That's what I was thinking. Going with Derek and Tomoko. Maybe it'll make Tomoko feel better about the React decision if she sees what's coming next.",
        ),
        _t(
            75,
            "assistant",
            "That's thoughtful. Seeing the React roadmap firsthand might address some of her performance concerns.",
        ),
        _t(
            76,
            "user",
            "Alright, I think that covers the major decisions. Let me go write up the ADR.",
        ),
        _t(
            77,
            "assistant",
            "Good luck with the write-up! Do you want me to help draft any section of the ADR?",
        ),
        _t(78, "user", "Nah I'll handle it. Thanks for helping me think through it."),
        _t(
            79,
            "assistant",
            "Anytime! Let me know if anything comes up during implementation.",
        ),
    ]


def _build_negation_2() -> list[dict]:
    """Choosing a cloud provider after evaluating multiple options."""
    return [
        _t(
            0,
            "user",
            "We need to migrate our on-prem infrastructure to the cloud. I've been asked to evaluate the big three: AWS, GCP, and Azure.",
        ),
        _t(
            1,
            "assistant",
            "That's a significant decision. What's your current on-prem setup like?",
        ),
        _t(
            2,
            "user",
            "About 40 VMs running a mix of workloads - web servers, databases, batch processing, and a machine learning pipeline. We also have about 50TB of data in a SAN.",
        ),
        _t(
            3,
            "assistant",
            "Diverse workload. Each cloud has strengths for different parts of that. What's driving the migration?",
        ),
        _t(
            4,
            "user",
            "Costs and scalability. Our hardware is aging and we're spending too much on maintenance. The CEO wants us off on-prem within 18 months.",
        ),
        _t(
            5,
            "assistant",
            "18 months for a full migration is aggressive but doable. Let's go through each option.",
        ),
        _t(
            6,
            "user",
            "Start with AWS. Our DevOps lead, Frank, is a big AWS advocate. He's certified and has run production workloads there at his previous job.",
        ),
        _t(
            7,
            "assistant",
            "Having someone with hands-on AWS experience is valuable. What are the specific AWS services you're considering?",
        ),
        _t(
            8,
            "user",
            "EC2 for compute, RDS for databases, S3 for storage, and SageMaker for the ML pipeline. Frank already has a rough architecture diagram.",
        ),
        _t(
            9,
            "assistant",
            "Solid service choices. AWS's ecosystem is the most mature. What does the pricing look like?",
        ),
        _t(
            10,
            "user",
            "Frank estimated about $28K per month for our workload. That includes reserved instances for the steady-state VMs and spot instances for batch processing.",
        ),
        _t(11, "assistant", "How does that compare to your current on-prem costs?"),
        _t(
            12,
            "user",
            "We're spending about $35K per month on-prem including hardware amortization, power, cooling, and the two sysadmins. So AWS would save us about $7K monthly.",
        ),
        _t(
            13,
            "assistant",
            "A 20% reduction. Not dramatic but meaningful over 18 months. What about GCP?",
        ),
        _t(14, "user", "Hold on, need to deal with a quick HR thing. Be right back."),
        _t(15, "assistant", "Take your time!"),
        _t(
            16,
            "user",
            "OK back. So GCP. Our data scientist, Min, is pushing hard for GCP because of BigQuery and Vertex AI. She says GCP's ML tools are significantly better than AWS's.",
        ),
        _t(
            17,
            "assistant",
            "Min has a point - GCP's ML and data analytics offerings are considered best-in-class by many. How did the pricing compare?",
        ),
        _t(
            18,
            "user",
            "Min got a quote from Google Cloud's sales team. They came in at $24K per month, which includes sustained use discounts and some migration credits.",
        ),
        _t(
            19,
            "assistant",
            "$4K less than AWS per month. The sustained use discounts are automatic, which is nice. Any concerns?",
        ),
        _t(
            20,
            "user",
            "Frank's main concern is that nobody on the team has GCP production experience except Min, and her experience is limited to the ML side. The infrastructure pieces like networking and security are unfamiliar to the team.",
        ),
        _t(
            21,
            "assistant",
            "Team expertise gap is a real risk factor. What about GCP's networking - are there specific concerns?",
        ),
        _t(
            22,
            "user",
            "Yeah, Frank said GCP's VPC model is different from AWS's and some of our compliance requirements around network isolation are easier to implement in AWS because he knows how.",
        ),
        _t(23, "assistant", "Familiarity versus cost trade-off. What about Azure?"),
        _t(
            24,
            "user",
            "Azure is interesting because we're a Microsoft shop for a lot of things. We use Active Directory, Office 365, and SQL Server. Leah from IT pointed out the integration benefits.",
        ),
        _t(
            25,
            "assistant",
            "The Microsoft ecosystem integration is Azure's strongest selling point for organizations already invested in MS tools. What's the pricing?",
        ),
        _t(
            26,
            "user",
            "Azure came in at $30K per month. More expensive than both AWS and GCP. Leah says we can get it down to $26K if we commit to a 3-year reserved instance plan.",
        ),
        _t(
            27,
            "assistant",
            "$26K with a 3-year commitment puts it between AWS and GCP. But a 3-year commitment is risky during a migration.",
        ),
        _t(
            28,
            "user",
            "That's exactly what Frank said. And Min pointed out that Azure's ML tools are weaker than both AWS and GCP for our specific use case - we're doing a lot of custom model training and Azure ML Studio is more oriented toward automated ML.",
        ),
        _t(
            29,
            "assistant",
            "Custom model training is better served by SageMaker or Vertex AI. Was there anything else in Azure's favor?",
        ),
        _t(
            30,
            "user",
            "Leah mentioned the Active Directory integration would simplify our identity management. But Frank pointed out that AWS has very good AD integration now through AWS Directory Service.",
        ),
        _t(
            31,
            "assistant",
            "So Azure's main advantage - MS integration - is partially available on AWS too. How did the team come down on Azure?",
        ),
        _t(
            32,
            "user",
            "We eliminated Azure. The higher cost, weaker ML tools, and the fact that we could get AD integration on AWS anyway made it hard to justify. Leah was fine with the decision - she admitted the pricing was tough to defend.",
        ),
        _t(
            33,
            "assistant",
            "Reasonable elimination. So it's between AWS and GCP. How did that final decision go?",
        ),
        _t(
            34,
            "user",
            "This was a long discussion. Min made the case that GCP would save us $4K per month - that's $48K per year - and the ML tools would accelerate our model development. She estimated we'd save an additional $20K per year in ML engineer productivity because Vertex AI is more streamlined than SageMaker.",
        ),
        _t(
            35,
            "assistant",
            "So roughly $68K annual savings with GCP. That's significant.",
        ),
        _t(
            36,
            "user",
            "But Frank countered with the risk argument. He said the learning curve for the infrastructure team would add 3-4 months to the migration timeline. At our current on-prem cost of $35K/month, those extra months cost $105K to $140K.",
        ),
        _t(
            37,
            "assistant",
            "That's a compelling risk-cost calculation. The GCP savings could be eaten up by extended migration time.",
        ),
        _t(
            38,
            "user",
            "And he added that during the learning curve period, we'd be more likely to make configuration mistakes that could cause outages or security incidents. The compliance team would not be happy.",
        ),
        _t(
            39,
            "assistant",
            "Compliance and security risks during migration are hard to quantify but very real. What was the final call?",
        ),
        _t(
            40,
            "user",
            "We went with AWS. Frank's expertise and the risk mitigation won out. The $7K monthly savings over on-prem is sufficient, and we believe we can hit the 18-month deadline.",
        ),
        _t(
            41,
            "assistant",
            "A pragmatic choice. AWS gets you off on-prem on time with someone who knows the platform.",
        ),
        _t(
            42,
            "user",
            "Min was disappointed but we agreed to keep evaluating GCP for new ML workloads that don't need to be on the same infrastructure as the main migration. She'll run a pilot on Vertex AI for the next model training cycle.",
        ),
        _t(
            43,
            "assistant",
            "A hybrid approach where ML experiments happen on GCP while production infrastructure is on AWS. That's a reasonable compromise.",
        ),
        _t(
            44,
            "user",
            "Right. We're not religious about single-cloud. If GCP proves itself for ML, we might move more workloads there over time.",
        ),
        _t(
            45,
            "assistant",
            "Multi-cloud is increasingly common for exactly this reason - use each platform's strengths.",
        ),
        _t(
            46,
            "user",
            "Cool. Can you summarize the full evaluation for our architecture review board?",
        ),
        _t(
            47,
            "assistant",
            "Cloud Migration Decision Summary: Evaluated AWS, GCP, and Azure for migrating 40 VMs and 50TB storage. Azure eliminated due to highest cost ($30K/mo, $26K with 3yr lock-in), weaker ML tools for custom training, and AD integration available on AWS. GCP eliminated despite lowest cost ($24K/mo) and best ML platform, due to team expertise gap risking 3-4 month timeline extension ($105-140K additional on-prem cost) and security/compliance concerns during learning curve. AWS selected at $28K/mo: team lead certified with production experience, mature ecosystem, AG Grid equivalent for infrastructure tooling, and meets 18-month migration deadline. Compromise: GCP pilot for ML workloads via Vertex AI.",
        ),
        _t(
            48,
            "user",
            "Looks good. I also need to tell you about the vendor evaluation for our monitoring solution but that can wait.",
        ),
        _t(
            49,
            "assistant",
            "Sure, whenever you're ready. Shall I draft the formal ARB document based on this summary?",
        ),
        _t(50, "user", "Yeah, I'll do that later. Thanks."),
        _t(
            51,
            "assistant",
            "You're welcome! Let me know when you want to discuss the monitoring vendor evaluation.",
        ),
    ]


def _build_unfinished_business() -> list[dict]:
    """Multiple tasks assigned but never completed, buried in natural conversation."""
    return [
        _t(0, "user", "Monday morning. Let's plan out the week."),
        _t(1, "assistant", "Happy Monday! What's on the agenda?"),
        _t(
            2,
            "user",
            "Lots of stuff. First, I need someone to research pricing for cloud storage options. I'm going to ask Jamie to put together a comparison of AWS S3, Google Cloud Storage, and Azure Blob by Wednesday.",
        ),
        _t(
            3,
            "assistant",
            "Good to get competitive pricing. Wednesday gives Jamie two days for a solid comparison.",
        ),
        _t(
            4,
            "user",
            "Yeah. Also, Preet said he'd have the security audit report done by Thursday. He's been working on it for two weeks already.",
        ),
        _t(
            5,
            "assistant",
            "Two weeks on a security audit is reasonable. Thursday delivery works with the Friday leadership meeting.",
        ),
        _t(
            6,
            "user",
            "Exactly, I want to present the findings at the Friday meeting. Let me also tell you about the team offsite we're planning. It's going to be somewhere in the mountains, a retreat kind of thing.",
        ),
        _t(7, "assistant", "That sounds nice! When is it planned for?"),
        _t(
            8,
            "user",
            "Late October. Hannah volunteered to research venues and send out options by this Friday. She's really into that kind of thing, always organizing team events.",
        ),
        _t(
            9,
            "assistant",
            "Great to have someone enthusiastic about it. Mountain retreats are perfect for team bonding.",
        ),
        _t(
            10,
            "user",
            "Oh and one more thing - I promised our VP Carla that I'd send her the updated roadmap by end of day Tuesday. She's been asking about it.",
        ),
        _t(
            11,
            "assistant",
            "That's tomorrow. Do you have the roadmap ready or do you still need to work on it?",
        ),
        _t(
            12,
            "user",
            "Mostly done, just need to finalize the Q4 section. I'll knock it out this afternoon.",
        ),
        _t(
            13,
            "assistant",
            "Sounds manageable. So to recap your week: Tuesday - roadmap to Carla, Wednesday - Jamie's pricing comparison, Thursday - Preet's security audit, Friday - leadership meeting + Hannah's venue options.",
        ),
        _t(
            14,
            "user",
            "Yep, that's the plan. Let me tell you about my weekend though. We went apple picking.",
        ),
        _t(15, "assistant", "That sounds lovely! Where did you go?"),
        _t(
            16,
            "user",
            "This orchard about 45 minutes outside the city. The kids loved it. My 5-year-old ate more apples than she picked.",
        ),
        _t(
            17,
            "assistant",
            "Ha! That's the best part of apple picking. Did you get a good haul?",
        ),
        _t(
            18,
            "user",
            "Two huge bags. My wife is going to make pies. She found some recipe from her grandmother's cookbook that she's excited about.",
        ),
        _t(
            19,
            "assistant",
            "Grandmother's recipe apple pies sound amazing. Fresh-picked apples make all the difference.",
        ),
        _t(
            20,
            "user",
            "Yeah she's pumped. OK so it's Tuesday now. I finished the roadmap and sent it to Carla.",
        ),
        _t(21, "assistant", "Great, that's one down! Did she have any feedback?"),
        _t(
            22,
            "user",
            "She replied saying she'd review it later this week. Now I need to deal with this production incident. Our main API is returning 503 errors intermittently.",
        ),
        _t(23, "assistant", "That's urgent. What percentage of requests are failing?"),
        _t(
            24,
            "user",
            "About 15%. It started an hour ago. The on-call engineer, Tomas, is investigating.",
        ),
        _t(
            25,
            "assistant",
            "15% error rate is significant. Is it affecting all endpoints or specific ones?",
        ),
        _t(
            26,
            "user",
            "Looks like it's the payment processing endpoints specifically. Other endpoints are fine.",
        ),
        _t(
            27,
            "assistant",
            "Payment processing issues are especially urgent. Any recent changes to those services?",
        ),
        _t(
            28,
            "user",
            "Tomas found it - one of the payment service replicas has a bad connection pool. He's draining it and spinning up a replacement.",
        ),
        _t(
            29,
            "assistant",
            "Good catch. Connection pool issues are a common culprit for intermittent 503s.",
        ),
        _t(
            30,
            "user",
            "Fix is in. Error rate dropped to zero. Tomas is going to write up the incident report. Oh, and I want him to also look into why our monitoring didn't catch this sooner. Can you note that I asked him to investigate the monitoring gap?",
        ),
        _t(
            31,
            "assistant",
            "Noted - you've asked Tomas to investigate why monitoring didn't alert on the connection pool issue sooner. Good follow-up.",
        ),
        _t(
            32,
            "user",
            "Right. This is actually related to something else - we've been saying for months we need to improve our alerting. I should put that on the Q4 roadmap. Speaking of which, let me tell you about the new hire we're onboarding.",
        ),
        _t(33, "assistant", "Who's the new person?"),
        _t(
            34,
            "user",
            "Riley. She starts next Monday. She's coming from a fintech startup so she has great payment systems experience. I asked Marcus to prepare an onboarding doc for her by Friday - you know, the usual stuff about our systems, access requests, team norms.",
        ),
        _t(
            35,
            "assistant",
            "Timely hire given the payment system issues. Having Marcus prep the onboarding doc gives him the rest of the week.",
        ),
        _t(
            36,
            "user",
            "Exactly. Oh, one more thing I keep forgetting - I need to send Derek that link to the architecture patterns talk I mentioned in our 1:1 last week. I told him I'd send it the next day and it's been a week.",
        ),
        _t(37, "assistant", "Better late than never! Do you have the link handy?"),
        _t(
            38,
            "user",
            "I know it was a Strange Loop talk. Let me look it up later. Remind me.",
        ),
        _t(
            39,
            "assistant",
            "Will do. You mentioned you'd send Derek the Strange Loop architecture patterns talk link.",
        ),
        _t(
            40,
            "user",
            "Wednesday now. Jamie hasn't sent the cloud storage pricing comparison yet. Let me ping him.",
        ),
        _t(41, "assistant", "It was due today. Has Jamie mentioned any delays?"),
        _t(
            42,
            "user",
            "He says he got pulled into the production incident fallout yesterday and couldn't finish. He'll have it by Friday instead.",
        ),
        _t(
            43,
            "assistant",
            "The incident took priority, understandable. Friday still works before the leadership meeting.",
        ),
        _t(
            44,
            "user",
            "Yeah but it means I won't have it for my prep. Whatever, I'll work with what I have. Let me tell you about this restaurant my wife and I tried last night.",
        ),
        _t(45, "assistant", "Ooh, where did you go?"),
        _t(
            46,
            "user",
            "New Ethiopian place downtown. The injera was incredible. We shared a veggie platter and some lamb tibs.",
        ),
        _t(
            47,
            "assistant",
            "Ethiopian food is so good for sharing. How was the atmosphere?",
        ),
        _t(
            48,
            "user",
            "Really cozy. Low lighting, good music. We'll definitely go back. Anyway, it's Thursday. Expecting Preet's security audit today.",
        ),
        _t(49, "assistant", "Right on schedule. Has he sent it over?"),
        _t(
            50,
            "user",
            "Not yet, it's still morning. Let me work on other things. I need to prep my slides for tomorrow's leadership meeting.",
        ),
        _t(51, "assistant", "What topics are you covering in the leadership meeting?"),
        _t(
            52,
            "user",
            "Three things: the production incident and our response, the Q4 roadmap highlights, and if I have the security audit findings I'll include those.",
        ),
        _t(
            53,
            "assistant",
            "Solid agenda. The incident response story shows operational maturity. Any updates from Tomas on the monitoring investigation?",
        ),
        _t(
            54,
            "user",
            "Hmm, I haven't heard from him actually. Let me follow up... He says he's been swamped and hasn't started looking into it yet. He promises he'll get to it next week.",
        ),
        _t(
            55,
            "assistant",
            "At least he's being honest about the delay. The monitoring gap investigation isn't blocking anything immediate.",
        ),
        _t(
            56,
            "user",
            "True. OK it's now Friday. Leadership meeting is in 2 hours. Quick status check - did I get everything I needed?",
        ),
        _t(
            57,
            "assistant",
            "Let me check: Roadmap to Carla - done Tuesday. Jamie's pricing comparison - delayed to today. Preet's security audit - due yesterday. Hannah's venue options - due today. Tomas's monitoring investigation - pushed to next week. Derek's architecture talk link - not sent yet. Marcus's onboarding doc - due today. What have you actually received?",
        ),
        _t(
            58,
            "user",
            "Hmm. Jamie sent the pricing comparison this morning, that's good. But actually it's just a spreadsheet with prices, no analysis or recommendation. Not what I asked for.",
        ),
        _t(
            59,
            "assistant",
            "So Jamie delivered the raw data but not the comparative analysis you wanted. You could still use the numbers for the meeting.",
        ),
        _t(
            60,
            "user",
            "I guess. Preet still hasn't sent the security audit. Let me ping him... He says he found some critical vulnerabilities and wants to review them with me before the report goes to leadership. He can meet Monday.",
        ),
        _t(
            61,
            "assistant",
            "Critical findings do warrant a pre-review. Better to discuss privately first. So no security findings in today's meeting.",
        ),
        _t(
            62,
            "user",
            "Right, I'll defer that to next week's meeting. What about the other stuff?",
        ),
        _t(
            63,
            "assistant",
            "Hannah's venue options and Marcus's onboarding doc for Riley were both due today. Have those come in?",
        ),
        _t(
            64,
            "user",
            "Let me check Slack... Marcus sent a draft of the onboarding doc. Looks pretty thorough. Hannah though - nothing from her. That's odd, she was so enthusiastic.",
        ),
        _t(65, "assistant", "Maybe she's still working on it. Worth a quick check-in."),
        _t(
            66,
            "user",
            "I'll message her after the meeting. OK I'm going to go prep my slides with what I have. Talk after the meeting.",
        ),
        _t(
            67,
            "assistant",
            "Good luck! You have plenty of material even without the security audit.",
        ),
        _t(
            68,
            "user",
            "Meeting went well! Carla liked the roadmap. The incident response section got a lot of questions but I handled it.",
        ),
        _t(
            69,
            "assistant",
            "Great to hear! Did the pricing numbers from Jamie come in handy?",
        ),
        _t(
            70,
            "user",
            "Not really, they were too raw to present. I mentioned we're evaluating cloud options and will have a recommendation next week. Which means I need to actually do that analysis myself now.",
        ),
        _t(
            71,
            "assistant",
            "So Jamie's output wasn't quite presentation-ready. At least you have the data to work from.",
        ),
        _t(
            72,
            "user",
            "Yeah. Oh, did you check in with Hannah about the venue research?",
        ),
        _t(
            73,
            "assistant",
            "That's for you to do - I don't have access to your Slack. Did you message her?",
        ),
        _t(
            74,
            "user",
            "Right, sorry. Let me do that now... She says she got completely sidetracked by a client emergency. She hasn't even started the venue research. She's apologetic and says she'll have it early next week.",
        ),
        _t(
            75,
            "assistant",
            "Client emergencies happen. At least the offsite isn't until late October so there's time.",
        ),
        _t(
            76,
            "user",
            "True. OK so now it's the following Monday. New week. Riley starts today! I need to make sure everything is set up for her.",
        ),
        _t(77, "assistant", "Exciting! Did Marcus finish the onboarding doc?"),
        _t(
            78,
            "user",
            "Yeah he sent the final version on Friday. It's really good. Riley is going to shadow the team this week.",
        ),
        _t(79, "assistant", "Great start. What else is happening this week?"),
        _t(
            80,
            "user",
            "Preet and I are meeting today about the security audit findings. That's the big one.",
        ),
        _t(
            81,
            "assistant",
            "Right, the critical vulnerabilities he found. Is Preet coming to your office or is it a video call?",
        ),
        _t(
            82,
            "user",
            "In person. He's very serious about these findings - doesn't want them in email or Slack. That tells me it's bad.",
        ),
        _t(
            83,
            "assistant",
            "Understandable caution with security vulnerabilities. The sensitivity suggests it could be a significant finding.",
        ),
        _t(
            84,
            "user",
            "So I met with Preet. It's bad. He found that our user authentication tokens aren't being invalidated on password change. So if someone's account is compromised and they change their password, the attacker's existing session stays active.",
        ),
        _t(
            85,
            "assistant",
            "That's a serious vulnerability. Session invalidation on credential change is a basic security requirement.",
        ),
        _t(
            86,
            "user",
            "Yeah. It needs to be fixed ASAP. I'm pulling Tomas and Riley onto this since it's related to our auth system.",
        ),
        _t(
            87,
            "assistant",
            "Good use of Riley's fintech security experience right off the bat. Speaking of Tomas, did he ever complete the monitoring gap investigation you asked about?",
        ),
        _t(
            88,
            "user",
            "You know what, I don't think he did. Let me check... No, he hasn't. And now I'm pulling him onto the auth fix so it'll be further delayed.",
        ),
        _t(
            89,
            "assistant",
            "The auth vulnerability is more urgent. The monitoring investigation can wait.",
        ),
        _t(
            90,
            "user",
            "Agreed. Oh, and I still haven't sent Derek that architecture talk link. It's been almost two weeks now. I'm the worst.",
        ),
        _t(91, "assistant", "Want to look it up now while you're thinking of it?"),
        _t(
            92,
            "user",
            "No, I'm about to go into another meeting. Remind me again later. Anyway, the auth fix is priority one this week. Everything else is secondary.",
        ),
        _t(
            93,
            "assistant",
            "Understood. Auth vulnerability fix is the top priority. I'll keep track of the other open items.",
        ),
        _t(
            94,
            "user",
            "Good. I also need to follow up with Hannah about the venue research. She said early this week. Let me check... Still nothing from her. I'll give her until Wednesday.",
        ),
        _t(
            95,
            "assistant",
            "Two missed deadlines on the venue research now. Might be worth a direct conversation rather than waiting.",
        ),
        _t(
            96,
            "user",
            "Fair point. But honestly the offsite isn't urgent and I have bigger fires. Auth fix first.",
        ),
        _t(
            97,
            "assistant",
            "Agreed on priorities. The auth vulnerability affects all users immediately.",
        ),
        _t(
            98,
            "user",
            "OK so it's Wednesday. Tomas and Riley have a fix for the token invalidation issue. They're deploying it tonight during maintenance window.",
        ),
        _t(99, "assistant", "Fast turnaround! Riley is already making an impact."),
        _t(
            100,
            "user",
            "She's sharp. Tomas says she found the root cause within the first hour. OK so I need to start thinking about next steps. What was I supposed to do the cloud storage analysis from Jamie's data?",
        ),
        _t(
            101,
            "assistant",
            "Yes - you mentioned you'd do the analysis yourself since Jamie's deliverable was raw data without recommendations.",
        ),
        _t(
            102,
            "user",
            "Right. I should get to that. But first, how's the rest of the week looking? What's still outstanding?",
        ),
        _t(
            103,
            "assistant",
            "Outstanding items: Tomas's monitoring gap investigation (assigned two weeks ago, not started), Derek's architecture talk link (you've been meaning to send for almost two weeks), Hannah's venue research (two missed deadlines), and your own cloud storage analysis.",
        ),
        _t(
            104,
            "user",
            "Ugh, that's a lot of loose ends. The monitoring thing and Derek's link are on me to follow up. The venue thing... let me message Hannah again.",
        ),
        _t(
            105,
            "assistant",
            "Do you want to set specific deadlines for the remaining items?",
        ),
        _t(
            106,
            "user",
            "Not right now. Let's see how the auth fix deployment goes tonight and I'll regroup tomorrow.",
        ),
        _t(107, "assistant", "Sounds good. Good luck with the deployment."),
        _t(
            108,
            "user",
            "Auth fix deployed successfully. All tokens now invalidate on password change. Preet verified it.",
        ),
        _t(
            109,
            "assistant",
            "Excellent! A critical vulnerability closed. That's the most important thing that happened this week.",
        ),
        _t(
            110,
            "user",
            "Absolutely. OK I'm heading into the weekend. We can pick up loose ends next week.",
        ),
        _t(
            111,
            "assistant",
            "Have a good weekend! Plenty of open items to tackle Monday.",
        ),
    ]


def _build_perspective_separation() -> list[dict]:
    """Team meeting with multiple people having distinct views on deadline and scope."""
    return [
        _t(
            0,
            "user",
            "Let me tell you about the most intense team meeting we've ever had. It went for 2 hours.",
        ),
        _t(1, "assistant", "Two hours! What was it about?"),
        _t(
            2,
            "user",
            "The launch deadline for our flagship product, Meridian. The release is set for June 15th and things are getting real.",
        ),
        _t(
            3,
            "assistant",
            "Launch deadlines always create tension. What was the core debate?",
        ),
        _t(
            4,
            "user",
            "Whether to push the deadline or ship on time with reduced scope. The team is split.",
        ),
        _t(5, "assistant", "Classic trade-off. Who was in the meeting?"),
        _t(
            6,
            "user",
            "Me, Alice the engineering lead, Bob the product manager, Carol the QA lead, Dave the designer, and Elena our VP of Engineering who joined for the last 30 minutes.",
        ),
        _t(7, "assistant", "Full cross-functional crew. Where did each person land?"),
        _t(
            8,
            "user",
            "Let me walk you through it. Alice spoke first. She's been tracking velocity and she says the team is behind by about 3 weeks. She thinks the June 15th deadline is unrealistic.",
        ),
        _t(
            9,
            "assistant",
            "Alice is basing this on data, which is hard to argue with. What's her proposed solution?",
        ),
        _t(
            10,
            "user",
            "She wants to push the deadline to July 7th. She says cramming to hit June 15th will result in technical debt that'll slow us down for months. She specifically said 'I'd rather ship a week late than spend three months cleaning up rushed code.'",
        ),
        _t(
            11,
            "assistant",
            "She's prioritizing code quality and long-term velocity. Three weeks of delay versus three months of tech debt cleanup is a compelling argument.",
        ),
        _t(
            12,
            "user",
            "Bob completely disagreed. He said the June 15th date was committed to the board and to early-access customers. He showed us that 200 enterprise customers are scheduled to start onboarding on June 20th. Moving the date means rescheduling all of them and potentially losing some to competitors.",
        ),
        _t(
            13,
            "assistant",
            "Bob's bringing the business reality. 200 enterprise customers in the pipeline is serious. What's his proposed solution?",
        ),
        _t(
            14,
            "user",
            "He wants to cut features to hit the date. Specifically, he'd drop the advanced analytics module and the custom workflow builder from the initial release. Those could come in a July update. He said, 'Core functionality on time beats complete functionality late.'",
        ),
        _t(
            15,
            "assistant",
            "Interesting - Bob is prioritizing the date over completeness. That's a product manager's typical lean. How did others react?",
        ),
        _t(
            16,
            "user",
            "Dave the designer was upset. He said the custom workflow builder is the main differentiator from competitors. Without it, Meridian looks like every other product in the space. His exact words: 'Shipping without the workflow builder is like selling a car without the engine.'",
        ),
        _t(
            17,
            "assistant",
            "Strong analogy from Dave. He sees the workflow builder as core, not optional. So Dave's position is...",
        ),
        _t(
            18,
            "user",
            "Dave agrees with Alice on the delay but for a completely different reason. Alice wants the delay for code quality, Dave wants it so the full feature set ships. He doesn't care as much about tech debt - he just wants the product to be complete and polished.",
        ),
        _t(
            19,
            "assistant",
            "So Alice and Dave both want a delay but are motivated by different things - engineering quality vs product completeness. Where does Carol stand?",
        ),
        _t(
            20,
            "user",
            "Carol was fascinating. She didn't pick a side. She said, 'I can work with either plan, but here's what keeps me up at night.' Then she showed a QA metrics dashboard.",
        ),
        _t(21, "assistant", "She brought data to the meeting too. What did it show?"),
        _t(
            22,
            "user",
            "The bug count has been rising steadily. There are currently 47 open bugs, 12 of which she classifies as release-blockers. She said we can't ship until those 12 are fixed regardless of when the deadline is. At current fix rates, those 12 bugs alone take 2 weeks.",
        ),
        _t(
            23,
            "assistant",
            "So Carol's position is that the timeline isn't just about features - even the minimum scope has a QA bottleneck. Where does that leave June 15th?",
        ),
        _t(
            24,
            "user",
            "She said June 15th is theoretically possible if we stop adding new features immediately and put the whole team on bug fixes. But that means nothing new gets done for 2 weeks.",
        ),
        _t(
            25,
            "assistant",
            "A bug fix sprint as the path to June 15th. That would address Carol's concerns but neither Alice's nor Dave's.",
        ),
        _t(
            26,
            "user",
            "Exactly. And Bob didn't like it either because it would mean the features already in development but not finished - including the workflow builder - wouldn't make the cut.",
        ),
        _t(
            27,
            "assistant",
            "So Carol's approach effectively gives Bob his date but forces Dave's feared outcome of shipping without key features. Complex dynamics.",
        ),
        _t(
            28,
            "user",
            "Oh before I continue - random thing - can you remind me to pick up my dry cleaning before 6pm? The suit for my cousin's wedding.",
        ),
        _t(
            29,
            "assistant",
            "Noted - pick up dry cleaning before 6pm, wedding suit. I'll remind you. Now, you mentioned Elena joined for the last 30 minutes?",
        ),
        _t(
            30,
            "user",
            "Yeah. So Elena walked in and listened for about 10 minutes without saying anything. You could feel the room tense up because she's the ultimate decision-maker.",
        ),
        _t(31, "assistant", "How did she finally weigh in?"),
        _t(
            32,
            "user",
            "She asked each person to summarize their position in one sentence. Then she said something that surprised everyone.",
        ),
        _t(33, "assistant", "What was it?"),
        _t(
            34,
            "user",
            "She said, 'June 15th is the date. No negotiation on that. But I'm adding 4 engineers to the team effective immediately.' She'd already arranged the resources.",
        ),
        _t(
            35,
            "assistant",
            "Bold move. More people on the problem instead of more time. How did that land?",
        ),
        _t(
            36,
            "user",
            "Alice pointed out that adding engineers to a late project makes it later. She literally quoted Brooks's Law - 'adding manpower to a late software project makes it later.' She said the new people would need 2 weeks just to get up to speed.",
        ),
        _t(
            37,
            "assistant",
            "Alice is citing a foundational software engineering principle. She has a strong point about onboarding overhead.",
        ),
        _t(
            38,
            "user",
            "Elena acknowledged it but said these aren't random engineers. She picked 4 people who worked on a similar project at her previous company and they're familiar with the tech stack. She estimated 3-4 days of onboarding, not 2 weeks.",
        ),
        _t(
            39,
            "assistant",
            "That changes the math significantly. Pre-vetted engineers who know the stack can ramp up much faster.",
        ),
        _t(
            40,
            "user",
            "Bob loved this. He said with 4 more engineers, the team could split into a feature track and a bug fix track. Feature track finishes the workflow builder while bug fix track clears Carol's 12 blockers.",
        ),
        _t(
            41,
            "assistant",
            "Two parallel tracks would address both the bug backlog and the feature completeness. How did Dave react?",
        ),
        _t(
            42,
            "user",
            "Dave was cautiously optimistic. He said if the workflow builder gets the attention it deserves, he's OK with June 15th. But he made Elena promise it wouldn't be a half-baked version.",
        ),
        _t(
            43,
            "assistant",
            "Dave wants full quality, not a rushed MVP of the workflow builder. Where did Carol land?",
        ),
        _t(
            44,
            "user",
            "Carol said she could live with it as long as the new engineers don't introduce MORE bugs. She said, 'I'm already drowning in 47 bugs. If the new people add 20 more, we're worse off.'",
        ),
        _t(
            45,
            "assistant",
            "Valid concern - more developers can mean more bugs. What was the resolution?",
        ),
        _t(
            46,
            "user",
            "Elena proposed that the new engineers would pair with existing team members for the first week. No solo PRs until Carol's team reviews their work patterns.",
        ),
        _t(
            47,
            "assistant",
            "Pairing mitigates the quality risk. Did Alice come around?",
        ),
        _t(
            48,
            "user",
            "Not entirely. She said, and this was telling: 'I'll execute the plan, but I'm going on record that June 15th is a mistake. We'll ship something on June 15th, but we'll spend July and August fixing what we rushed.' She was the lone dissenter.",
        ),
        _t(
            49,
            "assistant",
            "Alice is complying but registering her disagreement formally. That's actually a healthy team dynamic - she'll execute while maintaining her professional opinion.",
        ),
        _t(
            50,
            "user",
            "Yeah, Elena respected that. She said she'd check in every week and if Alice's predictions start coming true, they'd adjust.",
        ),
        _t(
            51,
            "assistant",
            "So the final decision is: June 15th date holds, 4 engineers added, split into feature and bug fix tracks, pairing for the first week, weekly check-ins with Elena.",
        ),
        _t(
            52,
            "user",
            "That's it. But here's the thing I'm worried about - the team is not unified. Bob and Dave are cautiously optimistic. Carol is nervous but willing. Alice thinks it's going to fail. The meeting ended with tension.",
        ),
        _t(
            53,
            "assistant",
            "Execution with a divided team is harder. Alice's lack of buy-in as the engineering lead could affect morale and pace.",
        ),
        _t(
            54,
            "user",
            "That's exactly what keeps me up at night. I'm going to have a 1:1 with Alice this afternoon to make sure she's OK.",
        ),
        _t(
            55,
            "assistant",
            "Smart move. She needs to feel heard even if the decision didn't go her way.",
        ),
        _t(
            56,
            "user",
            "Yeah. By the way, we had the new engineers start and it's been a week. Want to know how it's going?",
        ),
        _t(57, "assistant", "Absolutely! Are Alice's concerns playing out?"),
        _t(
            58,
            "user",
            "Mixed. Two of the four new engineers are killing it. They ramped up fast just like Elena predicted. But the other two are struggling with our codebase - it's more complex than their previous project.",
        ),
        _t(
            59,
            "assistant",
            "So half of Elena's reinforcements are delivering, half aren't - at least not yet. What's the velocity looking like?",
        ),
        _t(
            60,
            "user",
            "Overall velocity is up about 30%. Not the doubling Elena hoped for. Bug fix track has closed 7 of the 12 release blockers. Feature track has the workflow builder at about 60% complete.",
        ),
        _t(
            61,
            "assistant",
            "30% velocity increase is meaningful. 7 of 12 blockers closed is good progress. Workflow builder at 60% with about 3 weeks to go is tight.",
        ),
        _t(
            62,
            "user",
            "Very tight. Bob is starting to look nervous. Dave reviewed the workflow builder progress and says the UX needs more polish. He asked for one more week than planned.",
        ),
        _t(
            63,
            "assistant",
            "Dave is prioritizing quality again. Does one more week push past June 15th?",
        ),
        _t(
            64,
            "user",
            "Just barely. The workflow builder would be done June 18th by Dave's estimate. Bob says absolutely not - June 15th is the line.",
        ),
        _t(
            65,
            "assistant",
            "Three days difference. That's close enough to negotiate. What does Alice say?",
        ),
        _t(
            66,
            "user",
            "She says, 'I told you so.' She thinks even Dave's June 18th estimate is optimistic and it'll actually be June 22nd.",
        ),
        _t(
            67,
            "assistant",
            "Alice is doubling down on her original skepticism. A week late instead of three days would validate her initial position.",
        ),
        _t(
            68,
            "user",
            "Carol chimed in saying the remaining 5 release blockers each need about 2 days of work. That's 10 days if sequential, maybe 5 days in parallel. So that's tight too.",
        ),
        _t(
            69,
            "assistant",
            "Both the feature and bug fix tracks are tight. What's Elena going to say?",
        ),
        _t(
            70,
            "user",
            "We have a call with her tomorrow. I need to present the status honestly. This is going to be a hard conversation.",
        ),
        _t(
            71,
            "assistant",
            "Honesty about the timeline risk is the right approach. Elena can't make good decisions without accurate information.",
        ),
        _t(
            72,
            "user",
            "Agreed. I'll let you know how it goes. Oh, can you remind me about my dry cleaning?",
        ),
        _t(
            73,
            "assistant",
            "Yes - pick up your dry cleaning before 6pm! The wedding suit.",
        ),
        _t(74, "user", "Thanks! I better go now. We'll continue this saga later."),
        _t(75, "assistant", "Good luck with the Elena call. Keep me posted."),
        _t(
            76,
            "user",
            "So the Elena call happened. She was pragmatic about it. She approved a 'soft launch' on June 15th with the workflow builder in beta, and a full launch on June 22nd.",
        ),
        _t(
            77,
            "assistant",
            "A compromise - the date is technically met for the board commitment but the full product comes a week later. How did everyone react?",
        ),
        _t(
            78,
            "user",
            "Bob was relieved because he can tell the enterprise customers they're on track for June 15th onboarding, even if the workflow builder says 'beta' for a week. Dave was OK with it because the beta buys time for polish. Carol was indifferent as long as the 12 blockers are fixed.",
        ),
        _t(79, "assistant", "And Alice?"),
        _t(
            80,
            "user",
            "She said it's the least bad option. Still thinks we'll be cleaning up in July but at least the scope is more realistic now. She's not happy but she's accepted it.",
        ),
        _t(
            81,
            "assistant",
            "Least bad option from Alice is probably the best you'll get. The team has a path forward with defined milestones.",
        ),
        _t(
            82,
            "user",
            "Exactly. OK that's the Meridian saga for now. I'm tired just thinking about it.",
        ),
        _t(
            83,
            "assistant",
            "It's a complex situation with strong personalities and legitimate concerns on all sides. The soft launch compromise seems like a workable solution.",
        ),
        _t(84, "user", "It is. Talk to you later."),
        _t(
            85,
            "assistant",
            "Take care! Let me know if you need help with the launch prep.",
        ),
    ]


def _build_quantitative_aggregation() -> list[dict]:
    """Time estimates scattered across 90+ turns of natural conversation."""
    return [
        _t(
            0,
            "user",
            "We need to estimate the total effort for the website migration project. I'm going to collect estimates from the team over the next few days.",
        ),
        _t(1, "assistant", "Good approach. Who's providing estimates?"),
        _t(
            2,
            "user",
            "The core team: me for project management, Nina for frontend, Owen for backend, Patricia for database, and Quinn for DevOps. Each person estimates their own work.",
        ),
        _t(
            3,
            "assistant",
            "Five estimators. Are you using any specific methodology - t-shirt sizes, story points, hours?",
        ),
        _t(
            4,
            "user",
            "Hours. The client wants a fixed-price quote so I need to convert to actual hours for the proposal.",
        ),
        _t(5, "assistant", "Makes sense. Let me track the estimates as they come in."),
        _t(
            6,
            "user",
            "OK so Nina just gave me her frontend estimate. She says the homepage redesign is 40 hours, the product pages are 60 hours, and the checkout flow is 80 hours.",
        ),
        _t(
            7,
            "assistant",
            "Nina's frontend: homepage 40h, product pages 60h, checkout flow 80h. That's 180 hours total for frontend so far.",
        ),
        _t(
            8,
            "user",
            "Right. But she said she might need additional time for browser testing, she'll let me know.",
        ),
        _t(9, "assistant", "Noted - browser testing estimate pending from Nina."),
        _t(
            10,
            "user",
            "Hey, totally unrelated, but my neighbor is having a party this weekend and asked me to bring a dessert. Any ideas?",
        ),
        _t(
            11,
            "assistant",
            "A crowd-pleaser like brownies or a fruit tart would be easy. What's the occasion?",
        ),
        _t(
            12,
            "user",
            "Their kid's birthday. I think I'll do brownies. Easy and universally loved.",
        ),
        _t(
            13, "assistant", "Can't go wrong with brownies! Do you have a go-to recipe?"
        ),
        _t(
            14,
            "user",
            "Yeah, the one from Smitten Kitchen. It's never failed me. Anyway, Owen sent me his backend estimates.",
        ),
        _t(15, "assistant", "What did Owen come up with?"),
        _t(
            16,
            "user",
            "API development is 120 hours. He broke it down: user authentication 30 hours, product catalog API 45 hours, order processing 35 hours, and search functionality 10 hours.",
        ),
        _t(
            17,
            "assistant",
            "Owen's backend API: auth 30h, product catalog 45h, order processing 35h, search 10h = 120 hours. Does that include integration testing?",
        ),
        _t(
            18,
            "user",
            "Good question. He says no, add another 20 hours for integration testing.",
        ),
        _t(
            19,
            "assistant",
            "So Owen's total is 140 hours with integration testing. Got it.",
        ),
        _t(
            20,
            "user",
            "He also mentioned he'd need about 15 hours for third-party integrations - payment gateway and shipping calculator. Those are fiddly APIs to work with.",
        ),
        _t(
            21,
            "assistant",
            "Owen's revised total: 155 hours (120 API + 20 integration testing + 15 third-party integrations).",
        ),
        _t(
            22,
            "user",
            "Correct. Patricia is still working on her database estimate. She said something about the data migration being more complex than she initially thought.",
        ),
        _t(
            23,
            "assistant",
            "Data migrations often reveal hidden complexity. What's the legacy database like?",
        ),
        _t(
            24,
            "user",
            "It's a mess. Old MySQL with some tables that have hundreds of columns. Decades of accumulated schema changes without cleanup.",
        ),
        _t(
            25,
            "assistant",
            "A schema archaeology project. Patricia's estimate might be significant.",
        ),
        _t(
            26,
            "user",
            "Yeah. Let me tell you about this hike I did last weekend instead. I need a mental break from work estimates.",
        ),
        _t(27, "assistant", "Sure! Where did you hike?"),
        _t(
            28,
            "user",
            "Mount Tamalpais. The Dipsea trail. About 7 miles round trip. The views from the top were incredible - you can see the whole bay.",
        ),
        _t(
            29,
            "assistant",
            "Dipsea is a classic! The staircase section is brutal though. How long did it take?",
        ),
        _t(30, "user", "About 3.5 hours with breaks. My legs were jelly the next day."),
        _t(31, "assistant", "Worth it for those views though!"),
        _t(
            32,
            "user",
            "Totally. OK Patricia finally sent her estimate. She's breaking it into two parts: schema redesign and data migration.",
        ),
        _t(33, "assistant", "Makes sense to separate them. What are the numbers?"),
        _t(
            34,
            "user",
            "Schema redesign is 50 hours. She needs to normalize the tables, add proper indexes, and handle all the legacy column cleanup.",
        ),
        _t(
            35,
            "assistant",
            "50 hours for schema redesign on a messy legacy database sounds reasonable. What about the data migration itself?",
        ),
        _t(
            36,
            "user",
            "Here's where it gets interesting. She says 70 hours for the migration scripts, but that includes a lot of data cleaning. Some of the old records have character encoding issues, duplicates, and orphaned foreign keys.",
        ),
        _t(
            37,
            "assistant",
            "Data cleaning is always more work than people expect. So Patricia's total is 120 hours (50 schema + 70 migration).",
        ),
        _t(
            38,
            "user",
            "She also flagged that she'll need about 25 hours for performance testing and optimization of the new schema. She wants to make sure queries perform well under load.",
        ),
        _t(
            39,
            "assistant",
            "Patricia's revised total: 145 hours (50 schema + 70 migration + 25 performance testing).",
        ),
        _t(
            40,
            "user",
            "That's right. Quinn is next but he's in meetings all day. I'll get his DevOps estimate tomorrow.",
        ),
        _t(
            41,
            "assistant",
            "No rush. The running total so far: Nina 180h (plus pending browser testing), Owen 155h, Patricia 145h. That's 480 hours without Quinn and your PM hours.",
        ),
        _t(
            42,
            "user",
            "That's already a big number and we're not done. The client is going to be surprised.",
        ),
        _t(
            43,
            "assistant",
            "Setting expectations early is important. Do you have a sense of the client's budget?",
        ),
        _t(
            44,
            "user",
            "They said they were thinking around 600-700 hours total. We're already at 480 with two people not yet counted.",
        ),
        _t(
            45,
            "assistant",
            "That's going to be a tough conversation. You're likely looking at 700-900 hours total.",
        ),
        _t(
            46,
            "user",
            "Yeah. Hey, do you remember what brownie recipe I mentioned? My wife is asking.",
        ),
        _t(
            47,
            "assistant",
            "The Smitten Kitchen brownie recipe. You said it's never failed you.",
        ),
        _t(
            48,
            "user",
            "Right, thanks. OK so it's the next day. Quinn sent his DevOps estimate.",
        ),
        _t(49, "assistant", "What's Quinn's breakdown?"),
        _t(
            50,
            "user",
            "CI/CD pipeline setup: 25 hours. Infrastructure as code (Terraform): 35 hours. Monitoring and alerting setup: 20 hours. That's 80 hours from Quinn.",
        ),
        _t(
            51,
            "assistant",
            "Quinn at 80 hours. Running total: Nina 180h, Owen 155h, Patricia 145h, Quinn 80h = 560 hours.",
        ),
        _t(
            52,
            "user",
            "But wait, Quinn also said he'd need about 15 hours for the staging environment setup and another 10 hours for security hardening. He forgot to include those initially.",
        ),
        _t(
            53,
            "assistant",
            "Quinn revised to 105 hours (80 + 15 staging + 10 security). Running total now: 585 hours.",
        ),
        _t(
            54,
            "user",
            "And then there's me. Project management, client communication, sprint planning, and oversight. I estimate about 80 hours across the project.",
        ),
        _t(
            55,
            "assistant",
            "You at 80 hours. Grand total: Nina 180h, Owen 155h, Patricia 145h, Quinn 105h, You 80h = 665 hours.",
        ),
        _t(
            56,
            "user",
            "Hmm, that's right at the top of the client's budget. But we still have Nina's browser testing.",
        ),
        _t(
            57,
            "assistant",
            "Right, that's still pending. Have you heard back from her?",
        ),
        _t(
            58,
            "user",
            "Just now actually. She says browser testing across Chrome, Firefox, Safari, and Edge will be 30 hours. She also realized she forgot to account for responsive design adjustments - another 25 hours.",
        ),
        _t(
            59,
            "assistant",
            "Nina's revised total: 235 hours (original 180 + 30 browser testing + 25 responsive design). Grand total is now 720 hours.",
        ),
        _t(
            60,
            "user",
            "Which exceeds the client's budget. This is going to be a hard sell.",
        ),
        _t(
            61,
            "assistant",
            "720 hours at the top end. You could look for ways to reduce scope or find efficiencies. Are there any phases that could be deferred?",
        ),
        _t(
            62,
            "user",
            "The client absolutely needs the homepage and product pages for launch. The checkout flow could potentially use their existing one temporarily. Let me think about that.",
        ),
        _t(
            63,
            "assistant",
            "If you defer the checkout flow redesign, that's Nina's 80 hours back, bringing it to 640 hours.",
        ),
        _t(
            64,
            "user",
            "Hmm, true but then we'd also save some of Owen's order processing time since the current checkout talks to the old API. Maybe 20 hours from Owen?",
        ),
        _t(
            65,
            "assistant",
            "640 minus 20 = 620 hours. That's within the client's range.",
        ),
        _t(
            66,
            "user",
            "Let me talk to the team about it. But first I need to check something. Owen just messaged saying he underestimated the search functionality. He originally said 10 hours but after looking at the requirements, it's more like 30 hours because they want faceted search with autocomplete.",
        ),
        _t(
            67,
            "assistant",
            "Owen's search estimate jumps from 10h to 30h, adding 20 hours. That brings Owen to 175 hours and the full project back to 740 hours.",
        ),
        _t(
            68,
            "user",
            "Great, now we're even more over budget. Let me go think about this. I need to go pick up the kids from soccer practice.",
        ),
        _t(
            69,
            "assistant",
            "Take a break from the numbers! We can figure out the scoping conversation when you're back.",
        ),
        _t(
            70,
            "user",
            "Back. So I've been thinking about it. Let me just confirm where we are on the estimates.",
        ),
        _t(
            71,
            "assistant",
            "Current totals: Nina 235h (homepage 40, product pages 60, checkout 80, browser testing 30, responsive 25), Owen 175h (API 120 with revised search at 30, integration testing 20, third-party 15, plus the 20h search increase), Patricia 145h (schema 50, migration 70, perf testing 25), Quinn 105h (CI/CD 25, Terraform 35, monitoring 20, staging 15, security 10), You 80h. Grand total: 740 hours.",
        ),
        _t(
            72,
            "user",
            "Actually wait. I think Owen's math is off. Let me recalculate. His API breakdown was auth 30, product catalog 45, order processing 35, and search was originally 10 but now 30. So that's 30+45+35+30 = 140 for the APIs. Plus 20 for integration testing and 15 for third-party. So 175 total. OK that matches.",
        ),
        _t(73, "assistant", "Yes, 175 checks out for Owen."),
        _t(
            74,
            "user",
            "So the plan is to present 740 hours to the client and offer a phased approach where the checkout redesign moves to phase 2. That brings phase 1 to roughly 640 hours.",
        ),
        _t(
            75,
            "assistant",
            "And with Owen's reduced order processing, closer to 620 for phase 1.",
        ),
        _t(
            76,
            "user",
            "Right. Let me draft the proposal. Oh, one more thing - Patricia just told me she found some additional complexity in the data migration. Some of the legacy data has different date formats across tables. She needs another 10 hours for the date normalization.",
        ),
        _t(
            77,
            "assistant",
            "Patricia goes from 145h to 155h. Full project total: 750 hours. Phase 1 estimate adjusts to 630 hours.",
        ),
        _t(78, "user", "Yeah. OK I'll send this to the client. Wish me luck."),
        _t(
            79,
            "assistant",
            "Good luck! The phased approach is a reasonable way to stay within budget while delivering the full scope eventually.",
        ),
        _t(
            80,
            "user",
            "Client got back to me. They approved phase 1 at 630 hours and want a separate quote for phase 2 later. We're a go!",
        ),
        _t(
            81,
            "assistant",
            "Congratulations! That's a solid outcome. You preserved the relationship while being transparent about the effort.",
        ),
        _t(
            82,
            "user",
            "Thanks. Now the real work begins. At least I have solid estimates to manage against.",
        ),
        _t(
            83,
            "assistant",
            "And a clear scope boundary between phase 1 and phase 2. Good planning work.",
        ),
    ]


def _build_frequency_detection() -> list[dict]:
    """User mentions coffee in many contexts across 90 turns - ordering, preferences, complaints, rituals."""
    return [
        _t(
            0,
            "user",
            "Morning! Just grabbed my usual oat milk latte from the cafe downstairs. Ready to start the day.",
        ),
        _t(
            1,
            "assistant",
            "Good morning! An oat milk latte is a nice way to start. What's on today's agenda?",
        ),
        _t(
            2,
            "user",
            "Need to finalize the presentation for Thursday. The quarterly review for the Seattle office.",
        ),
        _t(3, "assistant", "How far along is the presentation?"),
        _t(
            4,
            "user",
            "About 60%. I need to add the financial slides and the team updates.",
        ),
        _t(
            5,
            "assistant",
            "You have two full days, that should be enough. Do you need help structuring any section?",
        ),
        _t(
            6,
            "user",
            "Maybe later. Let me focus on the numbers first. I've got a spreadsheet from accounting I need to parse through.",
        ),
        _t(7, "assistant", "Sounds like a plan. Let me know when you need help."),
        _t(
            8,
            "user",
            "Ugh, the numbers are a mess. I need more coffee for this. Going to get my second cup.",
        ),
        _t(
            9,
            "assistant",
            "Sometimes accounting spreadsheets require caffeination. Take your time!",
        ),
        _t(
            10,
            "user",
            "OK back. Got a black drip this time. No milk, no sugar - that's my 'serious work' coffee.",
        ),
        _t(
            11,
            "assistant",
            "Ha! Different coffee modes for different tasks. So what's the situation with the numbers?",
        ),
        _t(
            12,
            "user",
            "Revenue is up 12% which is great. But expenses grew 18% which is concerning. I need to break down where the spending increase is coming from.",
        ),
        _t(
            13,
            "assistant",
            "A 6-point gap between revenue and expense growth rates is notable. Any obvious culprits?",
        ),
        _t(
            14,
            "user",
            "Looks like headcount is the big one. We added 15 people in Q3 across engineering and sales.",
        ),
        _t(
            15,
            "assistant",
            "That's a significant hiring push. Were those planned hires?",
        ),
        _t(
            16,
            "user",
            "Yes, they were budgeted. The issue is that the hiring was frontloaded - most started in July, so we got 3 full months of salary cost but they're not fully productive yet.",
        ),
        _t(
            17,
            "assistant",
            "So the expense hit is immediate but the revenue contribution lags. That should normalize in Q4.",
        ),
        _t(
            18,
            "user",
            "Exactly what I plan to say in the presentation. Oh, speaking of the Seattle office, they sent us a care package. Local coffee beans from some artisanal roaster. Apparently it's called Lighthouse Roasters.",
        ),
        _t(
            19,
            "assistant",
            "That's a thoughtful gift! Seattle is famous for its coffee culture. Have you tried the beans yet?",
        ),
        _t(
            20,
            "user",
            "Not yet, I'll bring them home for the weekend. My pour-over setup at home would do them justice. The office drip machine would be a waste of good beans.",
        ),
        _t(
            21,
            "assistant",
            "A pour-over is definitely the right choice for specialty beans. What pour-over do you use?",
        ),
        _t(
            22,
            "user",
            "Hario V60. I've been using it for about two years. It was a game-changer compared to the old French press I had.",
        ),
        _t(
            23,
            "assistant",
            "The V60 is beloved by coffee enthusiasts. The control over water flow really makes a difference.",
        ),
        _t(
            24,
            "user",
            "Totally. Anyway, back to work. I need to put together the team update slides. Each team lead sent me their bullet points.",
        ),
        _t(25, "assistant", "How many teams are you covering?"),
        _t(
            26,
            "user",
            "Four: engineering, sales, product, and customer success. I just need to synthesize their updates into a coherent story.",
        ),
        _t(27, "assistant", "Any themes emerging across the teams?"),
        _t(
            28,
            "user",
            "The main theme is that we're investing now for growth later. Engineering is building the platform, sales is expanding the pipeline, product is doing research, customer success is reducing churn.",
        ),
        _t(
            29,
            "assistant",
            "That's a cohesive narrative - short-term investment for long-term gain. Connects well with the expense growth story.",
        ),
        _t(
            30,
            "user",
            "Exactly. OK it's lunch time. I'm going to grab a sandwich and a cold brew from that new place across the street.",
        ),
        _t(
            31,
            "assistant",
            "A cold brew on a warm day sounds refreshing. What's the new place?",
        ),
        _t(
            32,
            "user",
            "Called 'Pressed.' They do sandwiches and specialty coffee. Someone on my team said their nitro cold brew is amazing.",
        ),
        _t(33, "assistant", "Nitro cold brew is so smooth. Enjoy your lunch!"),
        _t(
            34,
            "user",
            "Back from lunch. The nitro cold brew lived up to the hype. Incredibly smooth, with this chocolate undertone. I might become a regular.",
        ),
        _t(
            35,
            "assistant",
            "A new coffee spot is always exciting. How's the presentation coming along?",
        ),
        _t(
            36,
            "user",
            "Good progress. Financial slides are done. Working on the team updates now. Oh, I should mention - I have a 1:1 with my manager at 3pm.",
        ),
        _t(37, "assistant", "What's on the agenda for the 1:1?"),
        _t(
            38,
            "user",
            "My career development and the annual review cycle. She's been hinting that a promotion might be on the table.",
        ),
        _t(
            39,
            "assistant",
            "That's exciting! Sounds like your work is being recognized.",
        ),
        _t(
            40,
            "user",
            "Maybe. We'll see. I need to not get ahead of myself. Can you help me think through how to position the quarterly review presentation?",
        ),
        _t(41, "assistant", "Sure! What angle are you thinking?"),
        _t(
            42,
            "user",
            "I want to lead with the revenue growth, acknowledge the expense increase, explain it as deliberate investment, and close with the growth trajectory for Q4.",
        ),
        _t(
            43,
            "assistant",
            "That's a solid structure: good news, honest acknowledgment, context, and forward-looking optimism. Classic investor deck flow.",
        ),
        _t(
            44,
            "user",
            "Perfect. Oh man, the office coffee machine is broken again. Third time this month. The thing is ancient.",
        ),
        _t(
            45,
            "assistant",
            "Three breakdowns in a month suggests it's time for a replacement. Is facilities going to fix it?",
        ),
        _t(
            46,
            "user",
            "They put a sign on it saying 'out of order, parts on order.' Last time the parts took a week. Half the office is going to Pressed today.",
        ),
        _t(
            47,
            "assistant",
            "Pressed is going to love the surge in business. Maybe the broken machine is a blessing in disguise for your cold brew discovery.",
        ),
        _t(
            48,
            "user",
            "Ha! Silver lining. We should really lobby for a better machine. The one we have is a basic drip. I'd love an espresso machine in the office.",
        ),
        _t(
            49,
            "assistant",
            "An office espresso machine would be a nice perk. Some offices have fully automatic ones that are relatively low maintenance.",
        ),
        _t(
            50,
            "user",
            "I'll mention it to facilities. Cost of employee happiness, right? Anyway, my 1:1 went well. My manager is supportive of the promotion. It would be to Senior Director.",
        ),
        _t(
            51,
            "assistant",
            "Senior Director - congratulations on being considered! What are the next steps?",
        ),
        _t(
            52,
            "user",
            "She's going to put the recommendation in front of the leadership team next month. The quarterly review presentation is actually part of my 'evidence' - showing I can represent the business to the Seattle office.",
        ),
        _t(
            53,
            "assistant",
            "So the quarterly review has double stakes - both informing the Seattle team and demonstrating your leadership capability.",
        ),
        _t(
            54,
            "user",
            "No pressure, right? OK it's the next morning. I'm doing my final run-through of the presentation. Had my oat milk latte and I'm ready.",
        ),
        _t(
            55,
            "assistant",
            "Back to the morning ritual! How does the presentation look?",
        ),
        _t(
            56,
            "user",
            "Pretty solid. The story flows well. Revenue up 12%, expenses up 18% due to intentional hiring, Q4 projections look good.",
        ),
        _t(57, "assistant", "Sounds polished. Any areas you're nervous about?"),
        _t(
            58,
            "user",
            "The expense question. I know someone is going to push back on 18% expense growth. I need to be ready with the headcount breakdown and the productivity lag explanation.",
        ),
        _t(
            59,
            "assistant",
            "Prepare a backup slide with the detailed headcount numbers. If someone pushes, you can pull it up rather than trying to explain from memory.",
        ),
        _t(
            60,
            "user",
            "Smart. Adding that now. By the way, the office coffee machine is still broken. Day 3. People are getting grumpy.",
        ),
        _t(
            61,
            "assistant",
            "A coffee-deprived office is not a productive office. Has facilities given an update?",
        ),
        _t(
            62,
            "user",
            "They said the parts ship Friday. So maybe next Monday? In the meantime I'm making do with Pressed and the cafe downstairs.",
        ),
        _t(
            63,
            "assistant",
            "At least you have options. Some people only rely on the office machine.",
        ),
        _t(
            64,
            "user",
            "I know, I feel bad for them. Several people have started bringing their own coffee from home. One guy brought a whole pour-over setup and set it up in the kitchen.",
        ),
        _t(
            65,
            "assistant",
            "Now that's dedication! Desperate times call for pour-over measures.",
        ),
        _t(
            66,
            "user",
            "Ha! I admire the commitment. OK so it's now Thursday. Presentation day.",
        ),
        _t(67, "assistant", "The big day! How are you feeling?"),
        _t(
            68,
            "user",
            "Good. Nervous but prepared. Had a double-shot espresso from Pressed this morning instead of my usual latte. Needed the extra kick.",
        ),
        _t(
            69,
            "assistant",
            "A double-shot for presentation day! You're fueled up. When does it start?",
        ),
        _t(
            70,
            "user",
            "2pm PST since it's a video call with the Seattle team. That's 5pm my time, which is late but works for the West Coast.",
        ),
        _t(
            71,
            "assistant",
            "Might want a late afternoon coffee pick-me-up before the call.",
        ),
        _t(
            72,
            "user",
            "Already planning on it. I'll grab something around 4:30. Anyway, I've been rehearsing the presentation all morning.",
        ),
        _t(
            73,
            "assistant",
            "Good preparation. Is anyone else presenting or is it all you?",
        ),
        _t(
            74,
            "user",
            "Mostly me. The four team leads will each do a 2-minute update for their section. I'm doing the intro, financials, and wrap-up.",
        ),
        _t(
            75,
            "assistant",
            "Nice structure. You set the context, they provide detail, you bring it home.",
        ),
        _t(76, "user", "That's the plan. Wish me luck."),
        _t(77, "assistant", "You've got this! Let me know how it goes."),
        _t(
            78,
            "user",
            "It went great! The Seattle team was impressed with the numbers. The expense question came up exactly as I predicted and the backup slide saved me.",
        ),
        _t(79, "assistant", "The preparation paid off! How did the team leads do?"),
        _t(
            80,
            "user",
            "Really well. Everyone was concise and on-message. My manager was watching and gave me a thumbs up after.",
        ),
        _t(
            81,
            "assistant",
            "That's a great signal for the promotion. She saw you lead a polished, prepared presentation.",
        ),
        _t(
            82,
            "user",
            "I hope so. I'm celebrating with a nice pour-over of those Seattle beans tonight. Lighthouse Roasters, remember?",
        ),
        _t(
            83,
            "assistant",
            "The care package beans! A perfect way to celebrate a successful presentation about the Seattle office. Full circle.",
        ),
        _t(
            84,
            "user",
            "Ha, didn't think of it that way but you're right! OK I'm heading out. Oh and the office coffee machine was fixed today, finally.",
        ),
        _t(
            85,
            "assistant",
            "Just in time for Friday! The office can return to normal caffeine levels.",
        ),
        _t(
            86,
            "user",
            "After 4 days of outage. I'm still lobbying for that espresso machine though. Drip just doesn't cut it anymore after the Pressed cold brews.",
        ),
        _t(
            87,
            "assistant",
            "Your coffee standards have been permanently elevated. That's the danger of discovering good coffee shops.",
        ),
        _t(
            88,
            "user",
            "True. My wallet would prefer I lower my standards back. Alright, have a good evening!",
        ),
        _t(89, "assistant", "Enjoy those Seattle beans! Have a great weekend."),
    ]


def _build_constraint_propagation() -> list[dict]:
    """Budget cut cascading through vendor selection and timeline."""
    return [
        _t(
            0,
            "user",
            "So we've got the budget approved for the new CRM implementation. $250K for the full project including software licenses, consulting, and customization.",
        ),
        _t(
            1,
            "assistant",
            "That's a healthy budget for a CRM implementation. What's the timeline?",
        ),
        _t(
            2,
            "user",
            "We want to go live by October 1st. That gives us about 5 months. Plenty of time if we pick the right vendor.",
        ),
        _t(
            3,
            "assistant",
            "Five months is reasonable for a CRM rollout. Which vendors are you evaluating?",
        ),
        _t(
            4,
            "user",
            "Salesforce, HubSpot, and Zoho. Each has different strengths. We've been through demos with all three.",
        ),
        _t(
            5,
            "assistant",
            "Good that you've already done demos. What are the key differentiators?",
        ),
        _t(
            6,
            "user",
            "Salesforce is the most powerful but most expensive. Their quote was $180K for licenses and implementation. HubSpot came in at $110K. Zoho is cheapest at $65K.",
        ),
        _t(
            7,
            "assistant",
            "Big range. Salesforce at $180K would eat 72% of the $250K budget, leaving only $70K for customization and training.",
        ),
        _t(
            8,
            "user",
            "Right. We estimated we need about $50K for custom development and $20K for training. So Salesforce at $180K plus $70K in other costs is exactly $250K. Tight but doable.",
        ),
        _t(
            9,
            "assistant",
            "No wiggle room with Salesforce. HubSpot at $110K would leave $140K - much more comfortable. What about Zoho?",
        ),
        _t(
            10,
            "user",
            "Zoho at $65K leaves $185K which is way more than we need. The problem with Zoho is it doesn't have the enterprise-grade workflow automation we need. We'd have to build a lot of custom integrations.",
        ),
        _t(
            11,
            "assistant",
            "Custom integrations could eat into those savings quickly. What did you decide?",
        ),
        _t(
            12,
            "user",
            "We went with Salesforce. The VP of Sales pushed hard for it because he used it at his last company and loves it. The implementation partner, Cloudforce Consulting, is starting next week.",
        ),
        _t(
            13,
            "assistant",
            "Salesforce it is. Cloudforce Consulting is the implementer. What's their track record?",
        ),
        _t(
            14,
            "user",
            "They're a certified Salesforce platinum partner. They've done over 200 implementations. They quoted 10 weeks for the setup and another 2 weeks for testing.",
        ),
        _t(
            15,
            "assistant",
            "12 weeks implementation plus testing, then presumably some time for training and go-live prep. That fits within 5 months.",
        ),
        _t(
            16,
            "user",
            "Exactly. The timeline works out to: Weeks 1-10 implementation, weeks 11-12 testing, weeks 13-14 training, weeks 15-16 parallel run, week 17 go-live. That's October 1st.",
        ),
        _t(17, "assistant", "Clean timeline with buffer. All systems go?"),
        _t(
            18,
            "user",
            "All systems go. Let me tell you about my weekend though. Went to a wine tasting in Napa.",
        ),
        _t(
            19, "assistant", "Napa! That sounds wonderful. What wineries did you visit?"
        ),
        _t(
            20,
            "user",
            "Three of them. Opus One, Robert Mondavi, and a small family-owned place called Tres Sabores. The small one was actually my favorite.",
        ),
        _t(
            21,
            "assistant",
            "The hidden gems are often the best. What did you like about Tres Sabores?",
        ),
        _t(
            22,
            "user",
            "Really personal experience. The owner poured the wine herself and told stories about each vintage. We bought a case of their zinfandel.",
        ),
        _t(
            23,
            "assistant",
            "That's the kind of experience you can't get at the big names. Sounds like a great weekend.",
        ),
        _t(
            24,
            "user",
            "It was. OK so it's been two weeks. Cloudforce Consulting started and they're making good progress on the Salesforce setup. But I just got some bad news.",
        ),
        _t(25, "assistant", "What happened?"),
        _t(
            26,
            "user",
            "The CEO announced company-wide budget cuts. Every department has to cut 20% from discretionary spending. That includes our CRM project.",
        ),
        _t(
            27,
            "assistant",
            "20% cut on $250K means the budget drops to $200K. But you've already committed to Salesforce at $180K plus the implementation costs. Where does that leave you?",
        ),
        _t(
            28,
            "user",
            "In a very bad spot. We've already signed the Salesforce contract - that $180K is committed. So our remaining budget goes from $70K to $20K.",
        ),
        _t(
            29,
            "assistant",
            "$20K for customization and training when you estimated needing $70K. That's a $50K shortfall. Something has to give.",
        ),
        _t(
            30,
            "user",
            "Exactly. I had a panic meeting with the project team this morning. Here's what's affected.",
        ),
        _t(31, "assistant", "Walk me through it."),
        _t(
            32,
            "user",
            "First, the $50K custom development we planned? We can only afford about $10K now. That means we're cutting most of the custom workflows and integrations.",
        ),
        _t(33, "assistant", "Which specific customizations are being cut?"),
        _t(
            34,
            "user",
            "The custom quoting tool, the integration with our ERP system, and the automated lead scoring model. We'll have to use Salesforce's out-of-the-box features instead.",
        ),
        _t(
            35,
            "assistant",
            "Those are significant cuts. The ERP integration especially could create manual work for the team.",
        ),
        _t(
            36,
            "user",
            "Yeah, the sales ops team will have to manually enter data from the ERP until we can fund the integration later. That's probably 5-10 hours per week of manual work.",
        ),
        _t(
            37,
            "assistant",
            "Ongoing manual work is a hidden cost of the budget cut. What about training?",
        ),
        _t(
            38,
            "user",
            "The $20K training budget is now $10K. We were going to do in-person training with Cloudforce for the whole sales team of 40 people. Now we can only afford training for 10 people - the managers. They'll have to train their teams themselves.",
        ),
        _t(
            39,
            "assistant",
            "Train-the-trainer approach. It works but it's less effective and puts extra burden on managers. What about the timeline?",
        ),
        _t(
            40,
            "user",
            "Here's where it gets really bad. Remember the vendor selection for Salesforce was based on the full $250K budget. Now that we're cutting customizations, Cloudforce Consulting says the implementation is actually faster since there's less to build. They can finish in 8 weeks instead of 10.",
        ),
        _t(
            41,
            "assistant",
            "Faster implementation is a silver lining, but you still need the testing, training, and parallel run periods.",
        ),
        _t(
            42,
            "user",
            "Right. But the VP of Sales just told me we CANNOT change the October 1st go-live date. He's already committed to it with the CEO and the board. And he added a new requirement - he wants the quoting tool included even though we said it was cut.",
        ),
        _t(
            43,
            "assistant",
            "He wants to maintain the date AND add back a cut feature, while having less money? That's a classic impossible triangle: scope, timeline, budget - pick two.",
        ),
        _t(
            44,
            "user",
            "Tell me about it. I told him something had to give. He said 'figure it out.' So now I'm trying to figure it out.",
        ),
        _t(
            45,
            "assistant",
            "Your options are limited. Can you build a simpler version of the quoting tool in-house?",
        ),
        _t(
            46,
            "user",
            "That's what I'm thinking. Our internal dev team could build a basic quoting tool, but it would pull them off the mobile app project for about 3 weeks.",
        ),
        _t(
            47,
            "assistant",
            "So the budget cut cascades: CRM budget cut -> custom development cut -> quoting tool dropped -> VP demands it back -> internal team pulled off mobile app -> mobile app delayed.",
        ),
        _t(
            48,
            "user",
            "Exactly. The mobile app was supposed to launch November 15th. If the dev team loses 3 weeks, it pushes to mid-December. Right before the holidays, which is the worst time to launch.",
        ),
        _t(
            49,
            "assistant",
            "So the realistic mobile app launch date becomes January? The holiday freeze would prevent a December launch.",
        ),
        _t(
            50,
            "user",
            "Probably, yes. The mobile team lead is not going to be happy. Let me go talk to her... Yeah she's upset. She said January launch means we miss the holiday shopping season for the mobile app. That's potentially $200K in lost revenue.",
        ),
        _t(
            51,
            "assistant",
            "So a $50K budget cut to the CRM could indirectly cause $200K in lost mobile revenue. That's the cascade effect.",
        ),
        _t(
            52,
            "user",
            "When I put it that way to the VP, he finally listened. We had a long discussion about trade-offs.",
        ),
        _t(53, "assistant", "What was the resolution?"),
        _t(
            54,
            "user",
            "He agreed to drop the quoting tool requirement for now. We'll stick with the reduced CRM scope and not pull the dev team off the mobile app. The quoting tool goes on the wish list for Q1 next year when hopefully there's budget.",
        ),
        _t(
            55,
            "assistant",
            "That preserves the mobile app timeline and the associated revenue. The CRM still launches on time but with less functionality.",
        ),
        _t(
            56,
            "user",
            "Right. It's not ideal but it's the least damaging option. The train-the-trainer approach for training is still happening though.",
        ),
        _t(
            57,
            "assistant",
            "So the final plan is: Salesforce implementation with out-of-the-box features (no custom quoting, ERP integration, or lead scoring), 8-week implementation by Cloudforce, training for 10 managers only, October 1st go-live maintained, mobile app stays on track for November 15th.",
        ),
        _t(
            58,
            "user",
            "That's the plan. I need to update Cloudforce on the reduced scope and make sure the timeline still works.",
        ),
        _t(
            59,
            "assistant",
            "With reduced scope, 8 weeks of implementation, 2 weeks testing, and the remaining time for training and parallel run should still hit October 1st.",
        ),
        _t(
            60,
            "user",
            "Yeah. It's frustrating though. We picked Salesforce because of its capabilities, and now we can't afford to use half of them.",
        ),
        _t(
            61,
            "assistant",
            "That's the irony - a cheaper CRM might have been fine for the reduced scope. But the contract is signed.",
        ),
        _t(
            62,
            "user",
            "Hindsight, right? If we'd picked HubSpot at $110K, the budget cut would have left us $90K for customization - more than the $70K we originally planned. We'd actually be ahead.",
        ),
        _t(
            63,
            "assistant",
            "A painful observation. The Salesforce choice was right with the original budget but became the worst option after the cut.",
        ),
        _t(
            64,
            "user",
            "Lesson learned for next time. Anyway, I need to go. It's been a stressful week.",
        ),
        _t(
            65,
            "assistant",
            "Hang in there. The plan is workable even if it's not what you originally envisioned. Let me know how the implementation progresses.",
        ),
    ]


def _build_consistency_checking() -> list[dict]:
    """Contradictions and scheduling conflicts buried in natural conversation."""
    return [
        _t(
            0,
            "user",
            "Let me tell you about this week's schedule. It's going to be crazy.",
        ),
        _t(1, "assistant", "Lay it on me! What's happening?"),
        _t(
            2,
            "user",
            "Monday I have a client meeting at 10am at the downtown office. Then lunch with my brother at 12:30. Afternoon is clear for deep work.",
        ),
        _t(
            3,
            "assistant",
            "That's a manageable Monday. What about the rest of the week?",
        ),
        _t(
            4,
            "user",
            "Tuesday I'm working from home all day. I have a dentist appointment at 2pm. And I need to prep for the board presentation which is on Thursday.",
        ),
        _t(
            5,
            "assistant",
            "Home office day with a dentist break. Plenty of time for board prep. How about Wednesday?",
        ),
        _t(
            6,
            "user",
            "Wednesday is meeting-heavy. Team standup at 9, then a design review at 10:30, lunch with the vendor at 12, and the marketing sync at 3pm. Oh and I need to be at the midtown office for the design review.",
        ),
        _t(
            7,
            "assistant",
            "That's a packed Wednesday. The design review requires you to be at the midtown office. Are the other meetings virtual?",
        ),
        _t(
            8,
            "user",
            "The standup is virtual, the lunch is at a restaurant, and the marketing sync is virtual. So I really only need to go to midtown for the design review.",
        ),
        _t(9, "assistant", "Makes sense. Thursday?"),
        _t(
            10,
            "user",
            "Thursday is the big day - board presentation at 2pm. It's at the downtown office in the main conference room. I'll probably go in early to set up.",
        ),
        _t(
            11,
            "assistant",
            "The board presentation at 2pm downtown. You mentioned you're prepping for it on Tuesday. How far along is it?",
        ),
        _t(
            12,
            "user",
            "About halfway done. I need to finalize the financials and the product roadmap section. Should be fine with Tuesday's prep time.",
        ),
        _t(13, "assistant", "Sounds like a plan. And Friday?"),
        _t(
            14,
            "user",
            "Friday is light. Just a 1:1 with my manager at 11am and then I'm off to the airport. Flying to Denver for the weekend to visit my college roommate.",
        ),
        _t(
            15,
            "assistant",
            "Nice! A light Friday before a fun weekend. What time is your flight?",
        ),
        _t(
            16,
            "user",
            "Flight's at 3:45pm. I'll leave the office around 1:30 to get to JFK with plenty of time.",
        ),
        _t(
            17,
            "assistant",
            "With a 1:30 departure, you have time for the 11am 1:1. Sounds all planned out.",
        ),
        _t(
            18,
            "user",
            "Yeah. Oh by the way, totally random - have you read any good books lately? I need something for the flight.",
        ),
        _t(19, "assistant", "What genres do you enjoy? Fiction, nonfiction, mystery?"),
        _t(
            20,
            "user",
            "I've been on a nonfiction kick. Business and science mostly. Last book I read was 'Thinking Fast and Slow' which blew my mind.",
        ),
        _t(
            21,
            "assistant",
            "If you liked Kahneman, you might enjoy 'Predictably Irrational' by Dan Ariely or 'The Black Swan' by Nassim Taleb. Both are behavioral economics adjacent.",
        ),
        _t(
            22,
            "user",
            "I've read The Black Swan, loved it. I'll try Predictably Irrational. Thanks for the rec.",
        ),
        _t(
            23,
            "assistant",
            "Great choice! Now, back to the week - anything else to plan for?",
        ),
        _t(
            24,
            "user",
            "Actually yes. I just remembered - my boss scheduled a strategy session for Thursday morning at 9am. It's at the uptown office.",
        ),
        _t(
            25,
            "assistant",
            "Thursday is getting busy now - strategy session at 9am uptown, then board presentation at 2pm downtown. Are those offices far apart?",
        ),
        _t(
            26,
            "user",
            "About 40 minutes by subway. So I'll need to head downtown after the strategy session.",
        ),
        _t(
            27,
            "assistant",
            "That works. If the strategy session runs to noon, you'd have 2 hours to get downtown and set up.",
        ),
        _t(
            28,
            "user",
            "Should be fine. The strategy session is only supposed to be 2 hours so I should be done by 11.",
        ),
        _t(
            29,
            "assistant",
            "Plenty of buffer time then. Let me recap the week for you.",
        ),
        _t(30, "user", "Sure go ahead."),
        _t(
            31,
            "assistant",
            "Monday: client meeting 10am downtown, lunch with brother 12:30. Tuesday: WFH, dentist 2pm, board prep. Wednesday: standup 9am virtual, design review 10:30 midtown, vendor lunch 12pm, marketing sync 3pm virtual. Thursday: strategy session 9am uptown, board presentation 2pm downtown. Friday: 1:1 with manager 11am, leave 1:30 for JFK flight at 3:45.",
        ),
        _t(
            32,
            "user",
            "Perfect, that's the plan. I'll put it all in my calendar. Oh also, I need to schedule a check-in with the engineering team about the API migration. I was thinking Wednesday at 4pm.",
        ),
        _t(
            33,
            "assistant",
            "Wednesday at 4pm works since the marketing sync at 3pm is virtual and usually runs about 45 minutes. Would the API migration check-in be virtual too?",
        ),
        _t(
            34,
            "user",
            "Yes, all virtual. Good. Let me see about other things... Oh, I told the Denver office I'd join their all-hands on Thursday at 11am Mountain Time. That's 1pm Eastern.",
        ),
        _t(
            35,
            "assistant",
            "Wait - Thursday at 1pm Eastern you'd be joining the Denver all-hands, but the board presentation is at 2pm downtown. Will the Denver all-hands be a quick thing? That's only a 1-hour gap, and you need to be physically set up in the conference room.",
        ),
        _t(
            36,
            "user",
            "Hmm, you're right. The Denver all-hands usually runs an hour. So it would go until 2pm which is exactly when the board presentation starts. That's a conflict.",
        ),
        _t(
            37,
            "assistant",
            "You'd need to either skip the Denver all-hands, join for just the first portion, or see if you can present at the board meeting slightly later. Which is more important?",
        ),
        _t(
            38,
            "user",
            "Board presentation can't move. I'll see if I can present my Denver update earlier in their all-hands. Let me email them.",
        ),
        _t(
            39,
            "assistant",
            "Good plan. Present your section first, then drop off to set up for the board meeting.",
        ),
        _t(
            40,
            "user",
            "Yeah. Oh, I also told Sarah I'd review her design doc by Wednesday. Let me make sure I have time... I'll do it Tuesday afternoon after the dentist.",
        ),
        _t(
            41,
            "assistant",
            "Tuesday afternoon should work - you're working from home with just the dentist appointment. How long is the design doc?",
        ),
        _t(
            42,
            "user",
            "About 20 pages. An hour should be enough. I'll do it before the dentist actually - wait, didn't I say the dentist is at 2pm? I'll review from 12 to 1 and then head to the dentist.",
        ),
        _t(
            43,
            "assistant",
            "That works. Gives you a solid block for the design doc review.",
        ),
        _t(
            44,
            "user",
            "Great. So I just realized something else - I told my team the weekly retro is this Thursday at 3pm. But if I'm doing the board presentation at 2pm and it usually runs 90 minutes, that goes until 3:30. I might miss the retro.",
        ),
        _t(
            45,
            "assistant",
            "Another Thursday conflict. The board presentation (2pm-3:30pm estimated) overlaps with the team retro at 3pm. Can someone else facilitate the retro?",
        ),
        _t(
            46,
            "user",
            "I'll ask Tony to run it. He's been wanting more leadership visibility. It would be good practice for him.",
        ),
        _t(
            47,
            "assistant",
            "Good delegation. Tony gets a growth opportunity and you don't have to rush out of the board meeting.",
        ),
        _t(
            48,
            "user",
            "Perfect. Let me also mention - I'll be at the uptown office all day Thursday since the strategy session is there in the morning.",
        ),
        _t(
            49,
            "assistant",
            "Hold on - you mentioned the board presentation is at the downtown office at 2pm. If you're at uptown all day Thursday, you'd need to go downtown for the board meeting. Did you mean you'll be at uptown in the morning and downtown in the afternoon?",
        ),
        _t(
            50,
            "user",
            "Oh right, I forgot. Yeah I'll be at uptown for the strategy session and then head downtown. I misspoke about being uptown all day.",
        ),
        _t(
            51,
            "assistant",
            "Got it - uptown morning, downtown afternoon. That's what we discussed earlier with the 40-minute subway ride.",
        ),
        _t(
            52,
            "user",
            "Right, right. OK so that's the week planned out. Let me switch gears - I need to figure out this gift for my wife's birthday.",
        ),
        _t(53, "assistant", "When is her birthday?"),
        _t(
            54,
            "user",
            "Next Saturday. She's been hinting about wanting a new kitchen gadget - specifically a KitchenAid mixer. But I also want to do something experiential.",
        ),
        _t(
            55,
            "assistant",
            "The KitchenAid plus a cooking class together could be a nice combination of practical and experiential.",
        ),
        _t(
            56,
            "user",
            "That's a great idea! She'd love a pasta-making class. I'll look into that.",
        ),
        _t(
            57,
            "assistant",
            "Perfect combo - she gets the mixer and then learns to make fresh pasta with it!",
        ),
        _t(
            58,
            "user",
            "Love it. OK one more work thing - I realized the Wednesday vendor lunch is at a restaurant downtown, but I need to be at the midtown office for the 10:30 design review. The restaurant is near the downtown office. How am I getting from midtown to downtown in time for a noon lunch?",
        ),
        _t(
            59,
            "assistant",
            "That could be tight. Midtown to downtown is about 20-25 minutes by subway. If the design review runs until 11:30 or later, you'd need to rush.",
        ),
        _t(
            60,
            "user",
            "The design review is supposed to be an hour, so 10:30 to 11:30. That gives me 30 minutes to get downtown. Doable but tight.",
        ),
        _t(
            61,
            "assistant",
            "30 minutes should work if the subway cooperates. You could also take a cab to be safe.",
        ),
        _t(
            62,
            "user",
            "Good fallback. I'll plan for the subway but keep Uber open just in case.",
        ),
        _t(63, "assistant", "Smart. Any other schedule items to worry about?"),
        _t(
            64,
            "user",
            "Actually, I just got a Slack from my boss. She wants to reschedule the Friday 1:1 to Thursday at 10am. But I have the strategy session at 9am uptown.",
        ),
        _t(
            65,
            "assistant",
            "If the strategy session runs 9am-11am as planned, a 10am 1:1 would overlap. Can she do 11:15am instead?",
        ),
        _t(
            66,
            "user",
            "Let me ask... She says 11:15 works. She'll come to the uptown office since she'll be there for another meeting anyway.",
        ),
        _t(
            67,
            "assistant",
            "Good - that avoids any transit issues. Thursday is now: strategy session 9-11am uptown, 1:1 with boss 11:15am uptown, Denver all-hands 1pm virtual (your section first), board presentation 2pm downtown, Tony runs team retro 3pm.",
        ),
        _t(
            68,
            "user",
            "That's a marathon Thursday. I better get good sleep Wednesday night.",
        ),
        _t(
            69,
            "assistant",
            "Definitely! Wednesday's meeting load is heavy too. Take it easy Wednesday evening.",
        ),
        _t(
            70,
            "user",
            "Plan to. Alright, I think we've covered everything. Thanks for helping me sort through the scheduling chaos.",
        ),
        _t(
            71,
            "assistant",
            "Happy to help! The main things to watch: the midtown-to-downtown transit for Wednesday's vendor lunch, the Denver all-hands timing on Thursday, and making sure the board presentation has proper setup time.",
        ),
        _t(72, "user", "Perfect. Talk to you tomorrow."),
        _t(73, "assistant", "Have a good evening! Good luck with the week ahead."),
    ]


def _build_perspective_separation_2() -> list[dict]:
    """Product strategy meeting where people have different views on pricing model."""
    return [
        _t(
            0,
            "user",
            "Our product team just had a big debate about pricing for the new tier. Let me debrief you.",
        ),
        _t(
            1,
            "assistant",
            "I'm all ears. What product and what pricing model are we talking about?",
        ),
        _t(
            2,
            "user",
            "Our SaaS platform, CloudDeck. We're adding an Enterprise tier. The debate is over whether to do flat-rate annual pricing or usage-based pricing.",
        ),
        _t(
            3,
            "assistant",
            "That's a fundamental product-market fit question. Who was in the discussion?",
        ),
        _t(
            4,
            "user",
            "Five people: Gina the CPO, Harris the CFO, Ines from Sales, Jake from Customer Success, and Kara the pricing analyst.",
        ),
        _t(5, "assistant", "Good cross-functional group. Where did each person land?"),
        _t(
            6,
            "user",
            "Gina kicked it off. She strongly favors usage-based pricing. She said it aligns incentives with customer value - the more they use, the more they pay, and presumably the more value they're getting.",
        ),
        _t(
            7,
            "assistant",
            "Gina's perspective makes sense from a product standpoint. Usage-based creates a natural correlation between value delivered and revenue captured.",
        ),
        _t(
            8,
            "user",
            "She showed data from Snowflake and Datadog as examples of successful usage-based SaaS companies. She thinks this is where the market is heading and we'd be ahead of competitors.",
        ),
        _t(
            9,
            "assistant",
            "Those are strong reference companies. Did she address the downsides of usage-based pricing?",
        ),
        _t(
            10,
            "user",
            "She acknowledged that revenue becomes less predictable but said the upside is higher total contract values from heavy users. Her exact words: 'We're leaving money on the table with flat-rate. Our biggest customers use 10x more than average but pay the same.'",
        ),
        _t(
            11,
            "assistant",
            "That's a quantifiable argument. The 10x usage disparity is a real revenue opportunity. How did Harris respond?",
        ),
        _t(
            12,
            "user",
            "Harris is the CFO so predictability is his world. He hates usage-based pricing. He said, 'I need to give the board quarterly revenue projections. With usage-based, I'm basically guessing.'",
        ),
        _t(
            13,
            "assistant",
            "Classic CFO concern - revenue predictability for planning and investor communication. What does he prefer?",
        ),
        _t(
            14,
            "user",
            "Flat-rate annual contracts, paid upfront. He wants to lock in Annual Recurring Revenue that he can forecast confidently. He showed a model where flat-rate gives us 15% less total revenue but 40% less variance in projections.",
        ),
        _t(
            15,
            "assistant",
            "The variance reduction is significant for a CFO. Investors like predictability. How did he address Gina's point about heavy users?",
        ),
        _t(
            16,
            "user",
            "He said we could create multiple flat-rate tiers based on team size - Small ($50K/year), Medium ($100K/year), Large ($200K/year), Enterprise ($400K/year). Heavy users would naturally fall into higher tiers.",
        ),
        _t(
            17,
            "assistant",
            "Tiered flat-rate is a middle ground. Captures some of the usage variation without the forecasting challenges. What did Ines from Sales think?",
        ),
        _t(
            18,
            "user",
            "Ines had a completely different angle. She's on the ground selling every day. She said flat-rate is easier to sell because customers know exactly what they'll pay. But she also said usage-based can help get customers in the door with low initial costs.",
        ),
        _t(19, "assistant", "So Ines sees benefits to both. Does she have a lean?"),
        _t(
            20,
            "user",
            "She proposed a hybrid: a base flat fee plus usage overage charges. Her reasoning was, 'I can sell the base fee as the predictable floor for the customer. And if they blow past the usage limits, the overage feels fair because they're clearly getting value.'",
        ),
        _t(
            21,
            "assistant",
            "The hybrid model is increasingly common. It gives customers predictability up to a point and captures upside. How did others respond to Ines's proposal?",
        ),
        _t(
            22,
            "user",
            "Gina liked it conceptually but worried it would be too complex to explain. Harris said the base fee gave him the predictability he needs. So there was some alignment on the hybrid.",
        ),
        _t(
            23,
            "assistant",
            "Sounds like the hybrid could be a consensus builder. What about Jake from Customer Success?",
        ),
        _t(
            24,
            "user",
            "Jake's perspective was all about churn. He said, and this was powerful: 'I've seen what happens with usage-based pricing at other companies. Customers who have a slow month see a small bill and think they don't need us. Then they churn. With flat-rate, they've already committed and they find reasons to use it more.'",
        ),
        _t(
            25,
            "assistant",
            "That's a nuanced behavioral insight. The sunk cost psychology of flat-rate can actually increase engagement and retention.",
        ),
        _t(
            26,
            "user",
            "Exactly. Jake is firmly in the flat-rate camp. He showed churn data from companies that switched from flat to usage-based and saw 15-20% churn increases in the first year.",
        ),
        _t(
            27,
            "assistant",
            "That's compelling data. Churn increases negate the per-customer revenue gains from usage-based. Where did Kara the pricing analyst fall?",
        ),
        _t(
            28,
            "user",
            "Kara had done the most analysis. She built a Monte Carlo simulation modeling 1000 scenarios. Her conclusion was that usage-based generates 22% more revenue over 3 years IF churn stays constant, but a 10% churn increase wipes out the revenue gain entirely.",
        ),
        _t(
            29,
            "assistant",
            "So the entire question hinges on churn impact. Kara's analysis perfectly quantifies Jake's concern.",
        ),
        _t(
            30,
            "user",
            "Right. She recommended a phased approach: launch Enterprise tier as flat-rate with Harris's tier structure, then add an optional usage-based addon for specific high-consumption features after 6 months.",
        ),
        _t(
            31,
            "assistant",
            "Start conservative, add complexity once you have data on actual usage patterns. That seems wise. How did the group react?",
        ),
        _t(
            32,
            "user",
            "There was general agreement to go with Kara's phased approach. Gina wasn't thrilled about delaying usage-based but acknowledged the churn risk was real. Harris was happy. Ines said she could sell the tiered flat-rate. Jake was relieved.",
        ),
        _t(
            33,
            "assistant",
            "So the decision is tiered flat-rate at launch (Small $50K, Medium $100K, Large $200K, Enterprise $400K), with usage-based addon to be evaluated in 6 months based on actual usage data. Gina wants usage-based but accepted the compromise.",
        ),
        _t(
            34,
            "user",
            "That's it. One thing that stuck with me was how different each person's framing was. Gina thinks about value alignment, Harris thinks about predictability, Ines thinks about sellability, Jake thinks about retention, and Kara thinks about the math.",
        ),
        _t(
            35,
            "assistant",
            "Each person is optimizing for their function. That's why cross-functional decisions are messy but produce better outcomes than any single perspective.",
        ),
        _t(
            36,
            "user",
            "Totally agree. OK I need to get back to work. Thanks for letting me debrief.",
        ),
        _t(
            37,
            "assistant",
            "Anytime! It sounds like the team made a well-reasoned decision.",
        ),
    ]


# ---------------------------------------------------------------------------
# Build all conversations and questions
# ---------------------------------------------------------------------------


def build_conversations() -> dict[str, list[dict]]:
    """Return conversation_id -> list of turn dicts."""
    return {
        "adv_evolving_term_1": _build_evolving_terminology_1(),
        "adv_evolving_term_2": _build_evolving_terminology_2(),
        "adv_negation_1": _build_negation_1(),
        "adv_negation_2": _build_negation_2(),
        "adv_unfinished": _build_unfinished_business(),
        "adv_perspective_1": _build_perspective_separation(),
        "adv_perspective_2": _build_perspective_separation_2(),
        "adv_quantitative": _build_quantitative_aggregation(),
        "adv_frequency": _build_frequency_detection(),
        "adv_constraint": _build_constraint_propagation(),
        "adv_consistency": _build_consistency_checking(),
    }


QUESTIONS: list[dict] = [
    # === Evolving terminology ===
    {
        "conversation_id": "adv_evolving_term_1",
        "category": "evolving_terminology",
        "question_index": 0,
        "question": "What is the current status of Project Phoenix? Include any milestones reached and upcoming work.",
        "source_chat_ids": [8, 10, 12, 14, 36, 54, 68, 70, 72, 74, 78, 90, 92, 96, 99],
        "ideal_response": "Project Phoenix (also known as the customer portal redesign, PHX, v2, Portal 2.0, 'the new portal,' and 'the bird') is a 16-week project in three phases. Phase 1 (auth + dashboard) is complete - dashboard load time dropped from 8 seconds to under 2 seconds. Phase 2 (reporting module) is on track for week 11. Phase 3 includes self-service features plus a newly added notification center. Subscription management may be deferred to post-launch. The scope expanded slightly with the notification center addition.",
        "benchmark": "advanced",
    },
    {
        "conversation_id": "adv_evolving_term_1",
        "category": "evolving_terminology",
        "question_index": 1,
        "question": "What are all the different names that have been used to refer to v2 throughout the conversation?",
        "source_chat_ids": [0, 8, 36, 54, 68, 70, 72],
        "ideal_response": "The project has been called: (1) 'the customer portal redesign' (original formal name), (2) 'Project Phoenix' (team codename), (3) 'PHX' (Slack abbreviation used for tickets like PHX-123), (4) 'the new portal' or 'the portal project' (VP's name for it), (5) 'v2' (informal team shorthand, contrasted with v1 for the legacy portal), (6) 'Portal 2.0' (product manager's newsletter name), (7) 'the bird' (Ravi's nickname based on the phoenix emoji).",
        "benchmark": "advanced",
    },
    {
        "conversation_id": "adv_evolving_term_2",
        "category": "evolving_terminology",
        "question_index": 2,
        "question": "What is the full story of the JIRA-4521 bug? What caused it and what were all its effects?",
        "source_chat_ids": [
            2,
            10,
            18,
            20,
            28,
            32,
            36,
            46,
            52,
            54,
            56,
            58,
            60,
            64,
            84,
            86,
            87,
        ],
        "ideal_response": "JIRA-4521 (also called 'the login issue,' 'that timeout thing,' and 'the monster bug') started as a deployment issue. The root cause was a single deployment that caused multiple problems: (1) A staging config file leaked to production, setting session timeouts to 3 minutes instead of 30 on 2 of 4 app server pods, causing random user logouts (the 'login issue'). (2) The same deployment's timesheet patch had a missing database index causing 60-second timeouts on submission (the 'timeout thing'). (3) The timeout caused 230 corrupted records in the payroll queue. All three issues were resolved: config fix for sessions, index migration for timesheet performance, and Sara's cleanup script for payroll data.",
        "benchmark": "advanced",
    },
    {
        "conversation_id": "adv_evolving_term_2",
        "category": "evolving_terminology",
        "question_index": 3,
        "question": "What names were used for the JIRA-4521 bug throughout the investigation?",
        "source_chat_ids": [2, 10, 28, 32, 52, 54, 60, 72],
        "ideal_response": "The bug was referred to as: 'the login issue' (initial user reports of being kicked out), 'that timeout thing' (once session expiry was identified), 'JIRA-4521' (formal ticket created by Nadia), and 'the monster bug' (team nickname because it manifested in three ways from one deployment). It was also metaphorically called a 'hydra' because of its three heads: login sessions, timesheet timeout, and payroll queue corruption.",
        "benchmark": "advanced",
    },
    # === Negation / rejected alternatives ===
    {
        "conversation_id": "adv_negation_1",
        "category": "negation",
        "question_index": 4,
        "question": "What frontend frameworks were considered and rejected for the dashboard project, and why was each one eliminated?",
        "source_chat_ids": [
            2,
            12,
            14,
            16,
            18,
            22,
            24,
            26,
            28,
            30,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
        ],
        "ideal_response": "Three frameworks were rejected: (1) Angular - rejected because of large bundle size (critical since users on field devices with spotty cellular need fast loads), steep learning curve for two incoming junior devs, despite strengths in opinionated structure and RxJS support. (2) Vue - rejected due to less mature data grid ecosystem (no battle-tested equivalent to AG Grid for millions of rows) and smaller contractor hiring pool (4M weekly npm downloads vs React's 20M), despite elegant reactivity and low learning curve for React devs. (3) Svelte - rejected despite 3x better performance benchmarks on rapid DOM updates and impressive prototype, because no team member had production experience, thin enterprise component ecosystem (would need custom data grid adding 6-8 weeks to 12-week timeline), and SvelteKit less battle-tested than Next.js at scale. A proposed 2-week spike was denied due to timeline constraints.",
        "benchmark": "advanced",
    },
    {
        "conversation_id": "adv_negation_1",
        "category": "negation",
        "question_index": 5,
        "question": "What was the final technology decision and what mitigation strategies were agreed upon to address known weaknesses?",
        "source_chat_ids": [50, 52, 54, 56, 66, 68, 70],
        "ideal_response": "React with Next.js and AG Grid Enterprise was selected. Mitigation strategies for React's known re-render performance issues: (1) proactive optimization patterns from day one (React.memo, useMemo, useCallback), (2) Zustand for state management to minimize re-renders, (3) React Server Components for static/slow-changing dashboard sections to reduce client overhead. For CSS, Tailwind CSS was chosen over styled-components (which Lisa argued for due to encapsulation), CSS Modules, Emotion, and vanilla CSS with BEM (Marcus's suggestion). Tomoko, who championed Svelte, remained frustrated and predicted they'd spend months fighting React re-render issues.",
        "benchmark": "advanced",
    },
    {
        "conversation_id": "adv_negation_2",
        "category": "negation",
        "question_index": 6,
        "question": "Which cloud providers were rejected in the migration evaluation and what were the specific reasons for each rejection?",
        "source_chat_ids": [
            6,
            8,
            10,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
        ],
        "ideal_response": "Two providers were rejected: (1) Azure - eliminated due to highest cost ($30K/mo, or $26K with 3-year lock-in that was deemed risky during migration), weaker ML tools for custom model training (Azure ML Studio more oriented toward automated ML, not their use case), and its main advantage of Active Directory integration being available on AWS through AWS Directory Service. Leah from IT acknowledged the pricing was hard to defend. (2) GCP - eliminated despite lowest cost ($24K/mo with sustained use discounts) and best ML platform (BigQuery, Vertex AI). Rejected because of team expertise gap (only Min had GCP experience, limited to ML), estimated 3-4 month timeline extension costing $105-140K in additional on-prem hosting, and security/compliance risks during the learning curve period. As a compromise, Min will pilot Vertex AI for ML workloads while production infrastructure goes to AWS ($28K/mo).",
        "benchmark": "advanced",
    },
    # === Unfinished business ===
    {
        "conversation_id": "adv_unfinished",
        "category": "unfinished_business",
        "question_index": 7,
        "question": "What tasks or promises were assigned during the conversation that were never completed or followed up on?",
        "source_chat_ids": [2, 8, 10, 30, 34, 36, 54, 74, 88, 90, 94, 103, 104],
        "ideal_response": "Several items remain unresolved by the end of the conversation: (1) Tomas's monitoring gap investigation - assigned after the production incident to determine why monitoring didn't catch the connection pool issue sooner. He never started it, citing being swamped, then got pulled onto the auth vulnerability fix. Still not done. (2) The architecture patterns talk link for Derek - the user promised to send this Strange Loop talk link after a 1:1 'the next day' and it's been almost two weeks with multiple reminders but never sent. (3) Hannah's venue research for the team offsite - she volunteered to research mountain retreat venues by Friday, missed that deadline, promised early next week, then missed again. Client emergency was her reason. Last mention shows she still hasn't delivered. (4) Jamie's cloud storage pricing analysis - he delivered raw price data but not the comparative analysis and recommendation that was originally requested. The user said they'd do the analysis themselves but there's no indication it was completed. Note: some items WERE completed: roadmap sent to Carla (Tuesday), Marcus's onboarding doc (Friday), auth vulnerability fix (deployed Wednesday), and Preet's security audit (reviewed Monday).",
        "benchmark": "advanced",
    },
    {
        "conversation_id": "adv_unfinished",
        "category": "unfinished_business",
        "question_index": 8,
        "question": "What was promised to Derek and what happened with it?",
        "source_chat_ids": [36, 38, 90, 92],
        "ideal_response": "The user promised to send Derek a link to an architecture patterns talk from the Strange Loop conference. The promise was made during a 1:1 meeting, with the user saying they'd send it 'the next day.' A week later it still hadn't been sent, and after another week the user acknowledged again they hadn't sent it ('I'm the worst'). They asked to be reminded again but then went into a meeting and deprioritized it. By the end of the conversation (over two weeks later), the link was never sent.",
        "benchmark": "advanced",
    },
    # === Perspective separation ===
    {
        "conversation_id": "adv_perspective_1",
        "category": "perspective_separation",
        "question_index": 9,
        "question": "What is Alice's position on the June 15th Meridian launch deadline, and how has it evolved?",
        "source_chat_ids": [8, 10, 36, 48, 66, 80],
        "ideal_response": "Alice, the engineering lead, has consistently opposed the June 15th deadline. Initially she argued the team is 3 weeks behind and wants to push to July 7th, citing her tracked velocity data. She specifically said 'I'd rather ship a week late than spend three months cleaning up rushed code,' prioritizing code quality and long-term team velocity. When Elena added 4 engineers, Alice pushed back citing Brooks's Law that adding people to a late project makes it later. When progress showed only 30% velocity increase (not the hoped-for doubling), she said 'I told you so.' Her final position was accepting the soft launch compromise as 'the least bad option' while going on record that 'we'll ship something on June 15th, but we'll spend July and August fixing what we rushed.' She complied but never agreed the timeline was achievable without quality costs.",
        "benchmark": "advanced",
    },
    {
        "conversation_id": "adv_perspective_1",
        "category": "perspective_separation",
        "question_index": 10,
        "question": "What was Bob's position on the launch deadline and how did it differ from Carol's?",
        "source_chat_ids": [12, 14, 20, 22, 24, 26, 40, 62, 64, 68, 78],
        "ideal_response": "Bob (product manager) was the strongest advocate for keeping June 15th. His argument was business-driven: 200 enterprise customers scheduled for onboarding June 20th, board commitment, and competitive risk. He proposed cutting features (analytics module and workflow builder) to hit the date, saying 'Core functionality on time beats complete functionality late.' He supported Elena's resource addition enthusiastically, seeing it as a way to have both features AND the date. He got nervous when the workflow builder was only 60% complete with 3 weeks to go and Dave estimated June 18th. He was relieved by the soft-launch compromise. Carol (QA lead) was neutral on the date itself but raised an orthogonal concern: 47 open bugs including 12 release-blockers that require 2 weeks of fix time regardless of scope decisions. She said June 15th was possible only with a total feature freeze and full bug-fix sprint. She was worried about quality, specifically that new engineers might add more bugs ('I'm already drowning in 47 bugs'). She accepted the compromise as long as the 12 blockers were fixed.",
        "benchmark": "advanced",
    },
    {
        "conversation_id": "adv_perspective_2",
        "category": "perspective_separation",
        "question_index": 11,
        "question": "What is Harris the CFO's position on pricing for the CloudDeck Enterprise tier, and how does it contrast with Gina's?",
        "source_chat_ids": [6, 8, 10, 12, 14, 16],
        "ideal_response": "Harris strongly favors flat-rate annual pricing with tiered plans (Small $50K, Medium $100K, Large $200K, Enterprise $400K). His primary concern is revenue predictability for board projections and investor communication. He showed a model where flat-rate yields 15% less total revenue but 40% less variance in projections. He said 'I need to give the board quarterly revenue projections. With usage-based, I'm basically guessing.' Gina (CPO) strongly favors usage-based pricing, arguing it aligns incentives with customer value. She cited Snowflake and Datadog as successful models and noted their biggest customers use 10x more than average but pay the same under flat-rate, saying 'We're leaving money on the table.' The fundamental tension: Gina optimizes for total revenue capture and value alignment, Harris optimizes for predictability and investor confidence.",
        "benchmark": "advanced",
    },
    # === Quantitative aggregation ===
    {
        "conversation_id": "adv_quantitative",
        "category": "quantitative_aggregation",
        "question_index": 12,
        "question": "What is the total estimated hours for the website migration project, broken down by person? Include all revisions to the estimates.",
        "source_chat_ids": [6, 16, 18, 20, 34, 36, 38, 50, 52, 54, 58, 66, 76],
        "ideal_response": "Final estimates by person: Nina (frontend) - 235 hours: homepage 40h + product pages 60h + checkout flow 80h + browser testing 30h + responsive design 25h (original 180h revised upward twice). Owen (backend) - 175 hours: auth 30h + product catalog 45h + order processing 35h + search 30h (revised up from 10h) + integration testing 20h + third-party integrations 15h. Patricia (database) - 155 hours: schema redesign 50h + data migration 70h + performance testing 25h + date normalization 10h (added late due to discovery of mixed date formats). Quinn (DevOps) - 105 hours: CI/CD pipeline 25h + Terraform IaC 35h + monitoring/alerting 20h + staging environment 15h + security hardening 10h (latter two added after initial estimate). User (project management) - 80 hours. Grand total: 750 hours. Phase 1 (deferring checkout flow): approximately 630 hours, which the client approved.",
        "benchmark": "advanced",
    },
    {
        "conversation_id": "adv_quantitative",
        "category": "quantitative_aggregation",
        "question_index": 13,
        "question": "How did the project estimate compare to the client's budget, and what was the resolution?",
        "source_chat_ids": [44, 55, 59, 62, 63, 64, 66, 67, 74, 75, 77, 80],
        "ideal_response": "The client's budget was 600-700 hours. The final estimate was 750 hours, exceeding the budget by 50-150 hours. The resolution was a phased approach: Phase 1 defers the checkout flow redesign (saving Nina's 80h and about 20h from Owen's order processing), bringing the estimate to approximately 630 hours - within the client's range. The client approved Phase 1 at 630 hours and requested a separate quote for Phase 2 later. Key progression: initial partial estimates already reached 480h before Quinn and PM were counted, signaling early that the budget would be tight.",
        "benchmark": "advanced",
    },
    # === Frequency detection ===
    {
        "conversation_id": "adv_frequency",
        "category": "frequency_detection",
        "question_index": 14,
        "question": "How does the user relate to coffee throughout the conversation? Track all mentions and the contexts they appear in.",
        "source_chat_ids": [
            0,
            8,
            10,
            18,
            20,
            22,
            30,
            34,
            44,
            46,
            48,
            54,
            60,
            62,
            64,
            68,
            82,
            84,
            86,
        ],
        "ideal_response": "Coffee appears in at least 12 distinct contexts: (1) Morning ritual - oat milk latte from the cafe downstairs (turns 0, 54). (2) Work fuel - black drip coffee as 'serious work' mode (turn 10). (3) Gift/culture - Seattle office sent Lighthouse Roasters beans as care package (turn 18). (4) Home brewing enthusiasm - Hario V60 pour-over setup, upgraded from French press 2 years ago (turns 20, 22). (5) Lunch beverage - nitro cold brew from Pressed, described as incredibly smooth with chocolate undertones (turns 30, 34). (6) Office machine frustration - machine broke 3 times in a month, was down for 4 days, people brought their own setups (turns 44, 46, 60, 62, 64, 84). (7) Office upgrade desire - lobbying for an espresso machine instead of basic drip (turns 48, 86). (8) Presentation fuel - double-shot espresso on presentation day for extra kick (turn 68). (9) Afternoon pick-me-up - planned late afternoon coffee before 5pm presentation (turn 72). (10) Celebration - planning to use Seattle beans for pour-over to celebrate successful presentation (turn 82). The user has a sophisticated coffee identity with specific preferences for different occasions.",
        "benchmark": "advanced",
    },
    # === Constraint propagation ===
    {
        "conversation_id": "adv_constraint",
        "category": "constraint_propagation",
        "question_index": 15,
        "question": "How did the budget cut cascade through the CRM project and what other projects were affected?",
        "source_chat_ids": [
            0,
            6,
            8,
            12,
            26,
            28,
            32,
            34,
            36,
            38,
            40,
            42,
            46,
            48,
            50,
            54,
            57,
            60,
            62,
        ],
        "ideal_response": "The cascade: (1) 20% company-wide budget cut reduced the CRM budget from $250K to $200K. (2) Since the $180K Salesforce contract was already signed, remaining budget dropped from $70K to $20K. (3) Custom development budget cut from $50K to $10K, eliminating the custom quoting tool, ERP integration, and automated lead scoring. (4) Training budget cut from $20K to $10K, switching from full-team in-person training (40 people) to train-the-trainer for 10 managers only. (5) The VP demanded the quoting tool be added back despite the cut. (6) Building the quoting tool internally would pull developers off the mobile app project for 3 weeks. (7) The mobile app launch would slip from November 15th to mid-December, which effectively means January due to holiday code freeze. (8) Missing the holiday shopping season could mean $200K in lost mobile revenue. (9) When presented with the $50K cut potentially causing $200K in lost revenue, the VP relented and dropped the quoting tool requirement. Final irony: if HubSpot ($110K) had been chosen instead of Salesforce ($180K), the budget cut would have left $90K for customization - more than the original $70K plan.",
        "benchmark": "advanced",
    },
    {
        "conversation_id": "adv_constraint",
        "category": "constraint_propagation",
        "question_index": 16,
        "question": "Given the budget cut, what plans need to be revised from the original CRM implementation plan?",
        "source_chat_ids": [8, 14, 16, 28, 32, 34, 38, 40, 54, 57],
        "ideal_response": "Revised from original: (1) Custom development scope: cut from $50K to $10K. Lost features: custom quoting tool, ERP integration (creating 5-10 hours/week of manual data entry for sales ops), automated lead scoring. (2) Training approach: changed from professional in-person training for all 40 sales team members to train-the-trainer with only 10 managers trained by Cloudforce. (3) Implementation timeline: actually shortened from 10 weeks to 8 weeks due to reduced scope, but this is offset by the reduced functionality. (4) Cloudforce Consulting scope: reduced to match the cut custom development. (5) Go-live date: maintained at October 1st. (6) Post-launch plan: quoting tool, ERP integration, and lead scoring all deferred to Q1 'wish list' contingent on future budget. The CRM will launch on time but with significantly less functionality than planned, essentially using Salesforce out-of-the-box features.",
        "benchmark": "advanced",
    },
    # === Consistency checking ===
    {
        "conversation_id": "adv_consistency",
        "category": "consistency_checking",
        "question_index": 17,
        "question": "Are there any scheduling conflicts or inconsistencies in the user's week that were identified or remain unresolved?",
        "source_chat_ids": [10, 24, 34, 35, 36, 44, 45, 48, 49, 50, 58, 59, 64, 65],
        "ideal_response": "Several conflicts were identified: (1) RESOLVED - Denver all-hands Thursday at 1pm ET overlaps with board presentation at 2pm ET (the all-hands usually runs an hour). Solution: user will present their section first, then drop off. (2) RESOLVED - Board presentation (2pm, runs 90 min to ~3:30pm) overlaps with team retro at 3pm. Solution: Tony will facilitate the retro. (3) RESOLVED - User said they'd be 'at the uptown office all day Thursday' but the board presentation is at the downtown office at 2pm. User corrected themselves: uptown in the morning for strategy session, then downtown for the afternoon. (4) RESOLVED - Friday 1:1 with manager originally scheduled, then boss wanted to reschedule to Thursday 10am which overlapped with the 9-11am strategy session. Solution: moved to 11:15am at uptown office. (5) POTENTIAL ISSUE - Wednesday design review at midtown (10:30-11:30) followed by vendor lunch downtown at noon with only 20-25 minutes subway time. Identified as tight; cab as backup plan. (6) NOTE - The board presentation timing (2pm) with setup needs was discussed but the Denver all-hands from 1-2pm means the user would only arrive at the downtown office at or just before 2pm with minimal setup time.",
        "benchmark": "advanced",
    },
    {
        "conversation_id": "adv_consistency",
        "category": "consistency_checking",
        "question_index": 18,
        "question": "The user said they'd be at the uptown office all day Thursday. Is that consistent with the rest of the schedule?",
        "source_chat_ids": [10, 24, 48, 49, 50, 51],
        "ideal_response": "No, it's inconsistent. The user said in turn 48 they'd be 'at the uptown office all day Thursday' because of the strategy session there. However, the board presentation is at 2pm at the downtown office (established in turn 10, confirmed in turn 24-26). The user corrected themselves in turn 50 when this was pointed out, acknowledging they misspoke and would actually go from uptown (morning strategy session) to downtown (afternoon board presentation) via a 40-minute subway ride.",
        "benchmark": "advanced",
    },
    # === Additional evolving_terminology question (cross-reference) ===
    {
        "conversation_id": "adv_evolving_term_2",
        "category": "evolving_terminology",
        "question_index": 19,
        "question": "What was the root cause of the monster bug and how was each aspect resolved?",
        "source_chat_ids": [
            4,
            18,
            20,
            36,
            38,
            46,
            52,
            54,
            58,
            60,
            62,
            64,
            66,
            68,
            84,
            86,
            91,
        ],
        "ideal_response": "The root cause was a single deployment that contained two problems: (1) a staging config file with 3-minute session timeouts leaked to production (only 2 of 4 pods received it, causing coin-flip behavior), and (2) a timesheet patch missing a database index. Resolution of each 'head': (1) Session timeouts (the 'login issue') - fixed by redeploying the correct config with 30-minute timeout to the affected pods. (2) Timesheet timeout (JIRA-4521 / 'that timeout thing') - Sara added the missing index on the timesheet_entries table (2M rows), reducing query time from 58 seconds to 0.2 seconds. Required a 20-minute maintenance window. (3) Payroll queue corruption - 230 malformed records were created during the timeout window. Sara wrote a cleanup script and reprocessed the records, verified against source timesheet data. Kai confirmed normal payroll service operation afterward. The overtime calculation itself (the original purpose of the patch) was correct - only the missing index caused the performance problem.",
        "benchmark": "advanced",
    },
    # === Additional unfinished_business question ===
    {
        "conversation_id": "adv_unfinished",
        "category": "unfinished_business",
        "question_index": 20,
        "question": "What happened with the monitoring gap investigation that Tomas was asked to do?",
        "source_chat_ids": [30, 31, 54, 55, 87, 88, 89],
        "ideal_response": "After the production incident (payment processing 503 errors caused by a bad connection pool), the user asked Tomas to investigate why monitoring didn't catch the issue sooner. Tomas initially promised to look into it but never started - he said he was swamped when checked on Thursday. The following week, the user still hadn't heard back, and then the auth vulnerability fix became the top priority, pulling Tomas onto that instead. By the end of the conversation, the monitoring investigation was never completed. It was repeatedly deprioritized in favor of more urgent work.",
        "benchmark": "advanced",
    },
    # === Additional perspective_separation ===
    {
        "conversation_id": "adv_perspective_2",
        "category": "perspective_separation",
        "question_index": 21,
        "question": "What was Jake's argument against usage-based pricing, and what data did he use to support it?",
        "source_chat_ids": [24, 26],
        "ideal_response": "Jake, from Customer Success, argued against usage-based pricing from a churn perspective. His key insight was behavioral: customers who have a slow usage month see a small bill and conclude they don't need the product, then churn. With flat-rate pricing, the sunk cost psychology of having already committed drives customers to find reasons to increase usage. He supported his argument with churn data from companies that switched from flat-rate to usage-based pricing, showing 15-20% churn increases in the first year after the switch. This data was pivotal because Kara's Monte Carlo analysis later showed that a mere 10% churn increase would wipe out all the revenue gains from usage-based pricing.",
        "benchmark": "advanced",
    },
    # === Additional quantitative ===
    {
        "conversation_id": "adv_quantitative",
        "category": "quantitative_aggregation",
        "question_index": 22,
        "question": "What revisions were made to Owen's backend estimates during the conversation, and what was his final total?",
        "source_chat_ids": [16, 18, 20, 66, 72],
        "ideal_response": "Owen's estimates evolved through several revisions: Initial API breakdown - auth 30h, product catalog 45h, order processing 35h, search 10h = 120h. First revision - added 20h for integration testing (not included initially) = 140h. Second revision - added 15h for third-party integrations (payment gateway + shipping calculator) = 155h. Third revision - search functionality increased from 10h to 30h after reviewing requirements for faceted search with autocomplete, adding 20h = 175h. Final total: 175 hours. The user double-checked the math: 30 (auth) + 45 (catalog) + 35 (orders) + 30 (search) = 140 API + 20 integration testing + 15 third-party = 175h confirmed.",
        "benchmark": "advanced",
    },
]


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
    print("Building advanced scenario conversations...")
    conversations = build_conversations()
    total_turns = sum(len(turns) for turns in conversations.values())
    for conv_id, turns in conversations.items():
        print(f"  {conv_id}: {len(turns)} turns")
    print(f"  Total: {total_turns} turns across {len(conversations)} conversations")

    # Build questions
    print(f"\n{len(QUESTIONS)} questions across categories:")
    from collections import Counter

    cats = Counter(q["category"] for q in QUESTIONS)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

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
    npz_path = OUTPUT_DIR / "segments_advanced.npz"
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
    json_path = OUTPUT_DIR / "questions_advanced.json"
    with open(json_path, "w") as f:
        json.dump(QUESTIONS, f, indent=2)
    print(f"Saved {len(QUESTIONS)} questions to {json_path}")

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

    # Validate source_chat_ids exist in conversations
    print("\n--- Source chat ID validation ---")
    turn_ids_by_conv: dict[str, set[int]] = {}
    for conv_id, turns in conversations.items():
        turn_ids_by_conv[conv_id] = {t["turn_id"] for t in turns}

    issues = 0
    for q in loaded_q:
        conv = q["conversation_id"]
        if conv not in turn_ids_by_conv:
            print(
                f"  WARNING: question {q['question_index']} references unknown conversation {conv}"
            )
            issues += 1
            continue
        valid_ids = turn_ids_by_conv[conv]
        for tid in q["source_chat_ids"]:
            if tid not in valid_ids:
                print(
                    f"  WARNING: question {q['question_index']} ({q['category']}) references "
                    f"turn {tid} not in {conv} (max: {max(valid_ids)})"
                )
                issues += 1
    if issues == 0:
        print("  All source_chat_ids validated successfully!")
    else:
        print(f"  {issues} issues found!")


if __name__ == "__main__":
    main()
