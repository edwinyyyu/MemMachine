"""Extra-adversarial synthetic conversations testing hard retrieval categories
that prior benchmarks don't cover.

Categories:
- causal_antecedents: why X happened — causes discussed BEFORE X without
  mentioning the eventual effect.
- counterfactual: if we hadn't done X — requires the decision + alternatives
  + reasons + implications of the rejected options.
- perspective_swap: track ONE person's changing stance across many turns,
  ignoring other people's comments.
- ellipsis_resolution: referent is named once, then referred to implicitly.
- cross_topic_bridging: two topics discussed separately; implicit connection.
- retraction_detection: flip-flops where only the final decision counts; need
  to recognize the retraction chain.
- commitment_tracking: explicit self-commitments (not tasks assigned TO the
  user), filtering out commitments that were later cancelled.
- hypothetical_to_actual: "IF we go with A..." (hypothetical) later followed
  by actually doing A; compare plan vs. actual.

Usage:
    uv run python extra_adversarial.py

This script writes:
    data/segments_extra_adversarial.npz
    data/questions_extra_adversarial.json
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def _t(turn_id: int, role: str, text: str) -> dict:
    return {"turn_id": turn_id, "role": role, "text": text}


# ===========================================================================
# Conversation 1: Causal antecedents + retraction detection + commitments
# A product launch that slips its date. The causes are scattered WEEKS earlier
# and never labeled as "this will cause a delay." There are also several
# commitments the user makes and later retracts.
# ===========================================================================

def _build_launch_causal() -> list[dict]:
    return [
        _t(0, "user", "Morning. Planning meeting with the Nimbus launch team at 10. Supposedly we're still on track for the March 3 date."),
        _t(1, "assistant", "Good morning. March 3 is the public launch of Nimbus, right? What's on the agenda?"),
        _t(2, "user", "Yeah. Going through the burndown, checking dependencies, usual stuff. Should be routine."),
        _t(3, "assistant", "Anything you want to think through before the meeting?"),
        _t(4, "user", "Not really. I'll give you an update after."),
        _t(5, "assistant", "Sounds good."),
        _t(6, "user", "Hey quick question totally unrelated — my brother's getting married in May and I need to pick a suit. Navy or charcoal?"),
        _t(7, "assistant", "Navy photographs better in outdoor light if the venue is outside. Charcoal is more formal."),
        _t(8, "user", "Outdoor reception, vineyard thing. Navy it is. Ok meeting's over. Went fine. Anton raised a concern about the onboarding flow being janky but nothing we can't smooth out in the next sprint."),
        _t(9, "assistant", "Good that Anton flagged it early. Who's handling the onboarding polish?"),
        _t(10, "user", "Reyna. She said she'd take it. She's reliable."),
        _t(11, "assistant", "Okay. Anything else from the meeting?"),
        _t(12, "user", "Oh, one thing. Jules mentioned in passing she's been approached by a recruiter at Airtable. She laughed it off but I don't know. Worth keeping an eye on."),
        _t(13, "assistant", "Jules is your senior backend engineer on Nimbus, right?"),
        _t(14, "user", "Basically the lynchpin of the auth service. Yeah."),
        _t(15, "assistant", "That's worth monitoring. Is there anything being done to retain her?"),
        _t(16, "user", "Her review is in April. I'll push for a raise then."),
        _t(17, "assistant", "Understood. Keeping an ear out is probably all you can do for now."),
        _t(18, "user", "Hey by the way, I'm going to commit to running a 10k by the end of summer. I've been putting it off for years. This is the year."),
        _t(19, "assistant", "That's a great goal. What's your current running baseline?"),
        _t(20, "user", "I can do 3 miles at a jog. So probably need to build up. I'll start a training plan this weekend."),
        _t(21, "assistant", "A 10k is 6.2 miles, so about doubling your current distance. Very achievable over a few months."),
        _t(22, "user", "Yep. I'll report back. Ok another week, another set of fires. Apparently our analytics pipeline is throwing weird errors on the ingestion side. The data team is looking into it."),
        _t(23, "assistant", "Is it affecting Nimbus specifically or everything?"),
        _t(24, "user", "Everything that uses the shared ingestion layer. Which is... a lot. Including the telemetry Nimbus depends on."),
        _t(25, "assistant", "That could block your instrumentation work. Who owns that layer?"),
        _t(26, "user", "Data platform team, which means Vikram. He's been stretched thin since his two contractors ended last quarter."),
        _t(27, "assistant", "So the ingestion issue might take a while to resolve."),
        _t(28, "user", "Probably. He said he'd get to it this week. We'll see."),
        _t(29, "assistant", "Okay. Is there a workaround in the meantime?"),
        _t(30, "user", "We can fake the telemetry for dev, but we can't launch without real numbers from day one."),
        _t(31, "assistant", "Got it. That's a launch-blocker if it doesn't get fixed in time."),
        _t(32, "user", "Yeah. Alright different topic. I promised my wife I'd plan our anniversary weekend. I'm going to book that Catskills cabin she liked last time. Doing it tonight."),
        _t(33, "assistant", "Nice. The one with the wood stove?"),
        _t(34, "user", "That's the one. Dates are May 10-12. Locking it in."),
        _t(35, "assistant", "Have fun planning. Let me know if you want help thinking through restaurants."),
        _t(36, "user", "Will do. Nimbus update: Jules resigned today. Two weeks notice. I'm gutted."),
        _t(37, "assistant", "I'm sorry. That's a big loss right before the launch."),
        _t(38, "user", "She took the Airtable offer. I should have pushed for the raise sooner instead of waiting for April."),
        _t(39, "assistant", "Hindsight. Is there an obvious successor for the auth service?"),
        _t(40, "user", "Nobody who knows it as well. Pradeep has touched it but he's junior. This is going to hurt."),
        _t(41, "assistant", "What's the plan for coverage during the notice period?"),
        _t(42, "user", "Jules is doing a knowledge transfer, and Pradeep is shadowing her. But realistically there's a 3-4 week gap where the auth service has nobody really senior on it."),
        _t(43, "assistant", "That overlaps with your launch window. Risky."),
        _t(44, "user", "Very. Anyway. Scope update — product is now asking us to add SSO support. Apparently two of the beta customers require it."),
        _t(45, "assistant", "SSO is a significant addition. Who proposed adding it mid-stream?"),
        _t(46, "user", "Dana, the PM. She negotiated with the beta customers and they flat out said they can't evaluate without SSO."),
        _t(47, "assistant", "Estimated effort?"),
        _t(48, "user", "Anton thinks 3 weeks if we use an off-the-shelf library, longer if we roll our own. I said use the library."),
        _t(49, "assistant", "Is the 3-week estimate accounting for the Jules transition?"),
        _t(50, "user", "No, he estimated it cold. So realistically more like 4-5 weeks with ramp-up."),
        _t(51, "assistant", "That's significant scope creep. Is the launch date moving?"),
        _t(52, "user", "Not yet officially. I told the VP we'd make it work."),
        _t(53, "assistant", "Okay. Hope the SSO work doesn't compound with the auth service transition."),
        _t(54, "user", "Actually — I need to cancel the Catskills trip. With Jules leaving and SSO added, I can't take a weekend off that close to launch. I told my wife. She was not thrilled."),
        _t(55, "assistant", "I'm sorry. Is there a way to make it up — maybe after launch?"),
        _t(56, "user", "Yeah I'm going to book us somewhere in April, post-launch, assuming we actually hit the date."),
        _t(57, "assistant", "That's a reasonable pivot."),
        _t(58, "user", "Random: saw a really cool bookstore in Brooklyn today. Oldy-timey place. Going to take the kid there Sunday."),
        _t(59, "assistant", "That sounds fun. Which one?"),
        _t(60, "user", "It's called Lenox Hill Books. Tiny place, stacks to the ceiling."),
        _t(61, "assistant", "Enjoy!"),
        _t(62, "user", "Ok work. Apparently the API vendor we use for the risk scoring service — CipherNode — is raising their rates by 40% starting in April. Finance just forwarded me the notice."),
        _t(63, "assistant", "That's a huge jump. Is risk scoring on the critical path for Nimbus?"),
        _t(64, "user", "Yeah. Every submission gets scored. At the projected usage for launch it's an extra $18k per month."),
        _t(65, "assistant", "Ouch. Can you negotiate, or switch providers?"),
        _t(66, "user", "Finance is pushing us to evaluate alternatives. Which would be yet another engineering effort."),
        _t(67, "assistant", "You don't have the bandwidth for that right now."),
        _t(68, "user", "No we don't. But if we eat the 40% there's a budget fight with finance."),
        _t(69, "assistant", "Either option costs you time or money. Worth flagging to the VP."),
        _t(70, "user", "Already did. She said figure it out after launch. Reasonable."),
        _t(71, "assistant", "Good. One less thing before March 3."),
        _t(72, "user", "Hey reminder — I told the team I'd write up the onboarding spec by end of this week. Need to block time."),
        _t(73, "assistant", "You said Reyna was handling the onboarding polish. Is the spec separate?"),
        _t(74, "user", "Oh you're right, Reyna has the polish. The spec I meant is the post-signup tutorial flow, different thing. I still have that on me."),
        _t(75, "assistant", "Got it. Block time Thursday morning maybe?"),
        _t(76, "user", "Thursday afternoon. Thursday morning I'm at my kid's school play."),
        _t(77, "assistant", "Right."),
        _t(78, "user", "Ok status meeting on Nimbus. Reyna raised the scope creep concern — the onboarding work is bleeding into all the connected surfaces. She estimates another 2 weeks beyond her current work to do it right. We had budgeted 3 days."),
        _t(79, "assistant", "What's driving the expansion?"),
        _t(80, "user", "The tutorial flow I'm writing the spec for touches onboarding. And the SSO work changes the signup path, which she now has to rework too."),
        _t(81, "assistant", "Everything is coupling together."),
        _t(82, "user", "Exactly. Scope feature creep on the auth feature specifically — SSO plus onboarding plus tutorial, all entangled."),
        _t(83, "assistant", "How does that affect the launch timeline?"),
        _t(84, "user", "I don't want to say it out loud yet."),
        _t(85, "assistant", "Understood."),
        _t(86, "user", "Anyway. I'm going to commit to doing 25 minutes of meditation every morning for the next month. I've been wound too tight."),
        _t(87, "assistant", "Good call. Do you have an app or just sit?"),
        _t(88, "user", "Just sit. I had a Headspace subscription but I let it lapse. I'll just use a timer."),
        _t(89, "assistant", "Nice. Start tomorrow?"),
        _t(90, "user", "Yeah. Ok update — turns out the ingestion layer fix isn't coming. Vikram says the fundamental schema change they wanted to make requires 4-6 weeks of work because the downstream consumers are coupled badly."),
        _t(91, "assistant", "So the telemetry for Nimbus launch..."),
        _t(92, "user", "We either launch without real-time telemetry, or we wait."),
        _t(93, "assistant", "That's a launch-blocker given what you said earlier about needing real numbers from day one."),
        _t(94, "user", "Yeah."),
        _t(95, "assistant", "Does this change the launch date conversation?"),
        _t(96, "user", "I have to raise it with the VP tomorrow. I've been burying this."),
        _t(97, "assistant", "Okay. Nothing wrong with raising it when you have the full picture."),
        _t(98, "user", "Hey — the meditation thing, I'm calling that off. I tried it for 4 days and my mornings are too chaotic. Maybe after launch."),
        _t(99, "assistant", "No worries. It's okay to shelve it."),
        _t(100, "user", "Talked to the VP. Nimbus launch is officially slipping. New date is April 7."),
        _t(101, "assistant", "How did she take it?"),
        _t(102, "user", "Not happy but not surprised. She said 'we were kidding ourselves.' Her words."),
        _t(103, "assistant", "Honest is better than political."),
        _t(104, "user", "Agreed. OK new date. Doable if the stars align. Moving on."),
        _t(105, "assistant", "Good luck. Let me know how the next sprint goes."),
        _t(106, "user", "Will do. By the way, I'm still on for that 10k training. Started running Tuesdays and Thursdays at lunch. 4 miles yesterday, felt good."),
        _t(107, "assistant", "Nice, that's progress."),
        _t(108, "user", "Yeah. And I booked a little weekend thing for my wife and me, not the Catskills, but a place in the Berkshires. April 12-14, after the new launch date."),
        _t(109, "assistant", "Good. Hopefully you'll actually get there this time."),
        _t(110, "user", "Knock on wood."),
    ]


# ===========================================================================
# Conversation 2: Counterfactual reasoning + cross-topic bridging
# Cloud provider debate + simultaneously an engineering reorg. Both
# independently discussed; the implicit connection is that the AWS decision
# was partly driven by the reorg (the team that would have learned GCP was
# being dissolved).
# ===========================================================================

def _build_cloud_reorg() -> list[dict]:
    return [
        _t(0, "user", "Alright, kicking off the infrastructure RFP work. We need to pick a cloud provider by end of Q2."),
        _t(1, "assistant", "For the whole company or a specific product?"),
        _t(2, "user", "Whole company. We're finally consolidating off our own data center. It's been a zombie for 3 years."),
        _t(3, "assistant", "Big decision. Who's on the evaluation committee?"),
        _t(4, "user", "Me, Priya from platform, Demetri from security, Kala from finance, and Ben from the ML team."),
        _t(5, "assistant", "Diverse group. How are you framing the options?"),
        _t(6, "user", "AWS, GCP, Azure, and a hybrid 'stay partially on-prem' option. Last one is there mostly so we can say we considered it."),
        _t(7, "assistant", "Do you have a leaning yet?"),
        _t(8, "user", "Not formally. I think the real decision is AWS or GCP. Azure has a Windows-heavy pitch that doesn't fit us."),
        _t(9, "assistant", "Understood. AWS vs. GCP — what's the strongest argument for each?"),
        _t(10, "user", "AWS is the incumbent for everyone's resumes. GCP has better ML tooling and Kala likes their pricing model."),
        _t(11, "assistant", "Who's strongest advocate for each?"),
        _t(12, "user", "Ben is strongly GCP for the ML tools. Priya is lukewarm on GCP — she's used it at a previous job and hated the docs."),
        _t(13, "assistant", "Docs often matter more than people admit."),
        _t(14, "user", "Yeah. Demetri is AWS because their compliance certifications are broader, especially for the FedRAMP stuff we need for the government contracts."),
        _t(15, "assistant", "FedRAMP is a meaningful gate. Does GCP have equivalent coverage?"),
        _t(16, "user", "They have some but not as complete. It would require extra paperwork and a slower audit cycle."),
        _t(17, "assistant", "So Demetri's concern is real, not just preference."),
        _t(18, "user", "Right. Ok unrelated — I'm binging a new show. 'Slow Horses.' Heard of it?"),
        _t(19, "assistant", "Yes, Gary Oldman as Jackson Lamb. Supposed to be excellent."),
        _t(20, "user", "It's so good. Watched the first three episodes last night. Slept 5 hours. Worth it."),
        _t(21, "assistant", "Sleep deprivation for good TV is a classic trade."),
        _t(22, "user", "Ok. So here's something else happening in parallel. Word is there's a big reorg coming in engineering. CTO has been hinting for months."),
        _t(23, "assistant", "Do you know the shape of it?"),
        _t(24, "user", "Rumor is they're dissolving the platform team and redistributing its people into the product-aligned teams."),
        _t(25, "assistant", "Priya's team, you mean."),
        _t(26, "user", "Yes. And Priya is pretty unhappy about it. She's been pushing back but it seems decided."),
        _t(27, "assistant", "That could affect your cloud decision indirectly — platform team is usually the cloud expertise pool."),
        _t(28, "user", "Yeah. But I'm trying to keep them separate in my head. Cloud decision is cloud decision."),
        _t(29, "assistant", "Fair, but worth noting the coupling."),
        _t(30, "user", "Noted. Next evaluation meeting is tomorrow — Ben is presenting the ML platform comparison."),
        _t(31, "assistant", "What's he going to show?"),
        _t(32, "user", "Vertex AI vs. SageMaker. He's done a proof of concept on both. Vertex is faster to get going, SageMaker is more flexible."),
        _t(33, "assistant", "Interesting tradeoff."),
        _t(34, "user", "He has real benchmarks. Like specific training time and cost numbers on our actual dataset."),
        _t(35, "assistant", "Concrete data should cut through a lot of arguing."),
        _t(36, "user", "Fingers crossed. Meeting happened. Ben's numbers were decisive for ML: Vertex was 30% cheaper for our workload and 2x faster to set up. Really strong case."),
        _t(37, "assistant", "So the ML side of the house wants GCP."),
        _t(38, "user", "Yeah. But ML is one team out of seven."),
        _t(39, "assistant", "Right."),
        _t(40, "user", "Random — my mother-in-law is visiting next week. Pray for me."),
        _t(41, "assistant", "How long?"),
        _t(42, "user", "Four days. We get along okay but by day three the commentary gets pointed."),
        _t(43, "assistant", "Is there a preferred activity that keeps things smooth?"),
        _t(44, "user", "Gardening. Keep her busy in the yard."),
        _t(45, "assistant", "Solid strategy. Ok, back to cloud — any updates?"),
        _t(46, "user", "Demetri presented the compliance comparison. AWS has 47 relevant certs, GCP has 31. Specifically three we need for the government contracts are AWS-only currently."),
        _t(47, "assistant", "Three gaps is not trivial."),
        _t(48, "user", "No. GCP says they're 'in progress' on the missing three but no firm date."),
        _t(49, "assistant", "That's a risk."),
        _t(50, "user", "A real one. The contracts we're chasing aren't hypothetical — we have two LOIs on the desk contingent on the compliance work."),
        _t(51, "assistant", "So the decision has a revenue tie."),
        _t(52, "user", "Exactly. Kala's been modeling it. If we lose those two contracts because of slow compliance, that's $12M ARR."),
        _t(53, "assistant", "That dwarfs any cloud cost difference."),
        _t(54, "user", "Yeah. It's making GCP harder to justify."),
        _t(55, "assistant", "Even with Ben's ML case?"),
        _t(56, "user", "Even with that. A 30% savings on ML compute is maybe $200k/year. Compare to $12M."),
        _t(57, "assistant", "Clear trade."),
        _t(58, "user", "There's still the option of AWS primary + GCP for ML specifically. Multi-cloud."),
        _t(59, "assistant", "Complexity penalty on that, but the 'best of both' pitch is appealing."),
        _t(60, "user", "That's what Ben is now advocating. Priya hates it because she'd have to staff expertise on two clouds."),
        _t(61, "assistant", "Staffing! How's the reorg affecting that conversation?"),
        _t(62, "user", "It's final. Platform team is being dissolved at end of Q2. Priya got formal notice today."),
        _t(63, "assistant", "So her staffing math just changed dramatically."),
        _t(64, "user", "Right. Three of her five engineers are being scattered across product teams. Two are being let go."),
        _t(65, "assistant", "That's going to reduce cloud expertise significantly."),
        _t(66, "user", "Yeah. And it gets worse — the two being let go are the two with GCP experience."),
        _t(67, "assistant", "So if GCP wins, you have no one with hands-on expertise."),
        _t(68, "user", "Exactly the problem. Priya made this point in the committee meeting today. It was quiet for a long time."),
        _t(69, "assistant", "That's a real consideration. Multi-cloud with no GCP expertise is asking for trouble."),
        _t(70, "user", "Right. Multi-cloud option is basically off the table now. It's AWS vs. GCP single provider."),
        _t(71, "assistant", "Quiet but important shift."),
        _t(72, "user", "Hey, watched episode 4 of Slow Horses. Characters are really growing on me."),
        _t(73, "assistant", "The ensemble work is strong."),
        _t(74, "user", "Yeah. Okay so the decision came down last Thursday. AWS."),
        _t(75, "assistant", "Unanimous?"),
        _t(76, "user", "Not quite. Ben dissented. He thinks we're being short-sighted about ML capabilities. But the FedRAMP issue plus the expertise loss from the reorg made it a clear call for the other four of us."),
        _t(77, "assistant", "Understood."),
        _t(78, "user", "Demetri had a good framing: 'We're picking the cloud that matches the team we'll have in six months, not the team we have now.'"),
        _t(79, "assistant", "Smart framing. The reorg really did shape the decision."),
        _t(80, "user", "It did. It's funny because formally the reorg wasn't in the RFP criteria."),
        _t(81, "assistant", "But it set the expertise context."),
        _t(82, "user", "Right. Moving on. Migration planning starts next month."),
        _t(83, "assistant", "Who's leading that?"),
        _t(84, "user", "Priya, ironically. She's staying but being renamed to 'infrastructure lead' in the new org structure, reporting to me directly."),
        _t(85, "assistant", "That's a positive pivot for her."),
        _t(86, "user", "Yeah. She negotiated well. She gets to keep two of her three engineers instead of losing them to product teams."),
        _t(87, "assistant", "Good outcome."),
        _t(88, "user", "Alright. And on the ML side, Ben is setting up a pilot on AWS Bedrock to see what's possible without Vertex. If Bedrock gets close on his benchmarks, he'll be okay with the decision."),
        _t(89, "assistant", "Healthy compromise."),
        _t(90, "user", "Last Slow Horses thought — I finished season 1. It was outstanding."),
        _t(91, "assistant", "Worth the sleep deprivation."),
        _t(92, "user", "Fully. Ok see you tomorrow."),
        _t(93, "assistant", "Talk soon."),
        _t(94, "user", "Oh one thing. I realized the reorg announcement actually predates the first cloud evaluation meeting. So in retrospect, the reorg was always going to shape the decision. We just didn't frame it that way."),
        _t(95, "assistant", "The capability loss was baked in from day one."),
        _t(96, "user", "Yeah. In some ways, the moment the CTO committed to dissolving platform, AWS became the right answer. We just didn't know it yet."),
        _t(97, "assistant", "Interesting way to think about it."),
        _t(98, "user", "Anyway. Migration begins."),
        _t(99, "assistant", "Onward."),
    ]


# ===========================================================================
# Conversation 3: Perspective swap (Sarah's arc) + retraction detection
# A product design debate with 5 participants. Track Sarah's evolving stance.
# Her stance changes 4 times, interleaved with many turns from other people.
# ===========================================================================

def _build_perspective_swap() -> list[dict]:
    return [
        _t(0, "user", "Kickoff for the search redesign project today. Wild first meeting."),
        _t(1, "assistant", "Five-person team, right?"),
        _t(2, "user", "Yeah. Me, Sarah, Devon, Marcus, and Lillian. Sarah opened by saying we should go with a full AI-first approach — conversational search, chat interface, everything."),
        _t(3, "assistant", "Bold opener."),
        _t(4, "user", "She's been at a conference all week and came back energized. She said, quote, 'the future is not keyword matching, it's dialog.'"),
        _t(5, "assistant", "Where did others land?"),
        _t(6, "user", "Devon pushed back hard. Said our users don't want chat — they want results. Quick, clean, keyboard-friendly."),
        _t(7, "assistant", "Two opposed views. Marcus and Lillian?"),
        _t(8, "user", "Marcus was neutral, wanted to see user research. Lillian agreed with Devon more than Sarah."),
        _t(9, "assistant", "So Sarah was outnumbered."),
        _t(10, "user", "A bit. But Sarah was persistent. Kept bringing the conversation back to chat UX."),
        _t(11, "assistant", "Understandable — she's a senior designer, she has strong instincts."),
        _t(12, "user", "Yeah. We ended with a plan to do user interviews before deciding."),
        _t(13, "assistant", "Sensible."),
        _t(14, "user", "Quick personal thing — I think my cat has fleas. Taking her to the vet Saturday."),
        _t(15, "assistant", "Poor kitty. Hopefully an easy fix."),
        _t(16, "user", "Yeah. Ok so the user interviews are done. We talked to 15 power users and 10 casual ones."),
        _t(17, "assistant", "What did you learn?"),
        _t(18, "user", "Power users overwhelmingly want faster results and keyboard shortcuts. Casuals were mixed — some liked the idea of chat, others found it overwhelming."),
        _t(19, "assistant", "How did the team react?"),
        _t(20, "user", "Devon did a victory lap. Marcus said the data supports a traditional search refresh. Sarah was quieter than usual."),
        _t(21, "assistant", "And Lillian?"),
        _t(22, "user", "Lillian said she thinks we should go with a traditional interface but with a chat option as a secondary mode."),
        _t(23, "assistant", "Reasonable compromise."),
        _t(24, "user", "Yeah. Sarah later said — 'maybe chat-first was too aggressive given the data, but a hybrid has potential.'"),
        _t(25, "assistant", "That's a softening of her position."),
        _t(26, "user", "Yeah. She's still in the chat camp but not as strongly."),
        _t(27, "assistant", "Fair."),
        _t(28, "user", "Had lunch with a friend who works at Shopify. Their search team has 40 people. Ours has 5."),
        _t(29, "assistant", "Scale difference matters. You can't replicate their ambition."),
        _t(30, "user", "Agreed. Ok next meeting. Devon presented his mockups for a keyboard-centric search. Really clean stuff."),
        _t(31, "assistant", "What was the reaction?"),
        _t(32, "user", "Marcus loved it. Lillian said she'd add her chat-as-secondary idea to it. Sarah was critical — said the mockups felt 'like every other search box on the internet.'"),
        _t(33, "assistant", "So Sarah isn't a fan."),
        _t(34, "user", "She's not. She says without something distinctive, we won't stand out."),
        _t(35, "assistant", "Counterargument?"),
        _t(36, "user", "Devon said 'search should be invisible. Distinctiveness is for landing pages.'"),
        _t(37, "assistant", "Good line."),
        _t(38, "user", "Yeah. Sarah went quiet. We ended the meeting inconclusively."),
        _t(39, "assistant", "Tough room."),
        _t(40, "user", "Cat update — she did have fleas. On flea medication now. Also vacuumed the entire house."),
        _t(41, "assistant", "Thorough. That should clear it up."),
        _t(42, "user", "Fingers crossed. Ok work — I met with Sarah 1:1. She's really thinking about this."),
        _t(43, "assistant", "What did she say?"),
        _t(44, "user", "She said — 'I've been sitting with Devon's mockups and I think he's actually right. Users don't care about distinctiveness, they care about getting what they need. I was overreaching.'"),
        _t(45, "assistant", "A meaningful shift. She went from chat-first to chat-hybrid to now endorsing traditional search?"),
        _t(46, "user", "Yeah. She said the conference had her on a different wavelength and being grounded by the user data was humbling."),
        _t(47, "assistant", "Gracious reset."),
        _t(48, "user", "She's going to help Devon polish the mockups. Said she wants to bring her visual design skills to the clean-search direction instead of fighting it."),
        _t(49, "assistant", "That's strong teamwork."),
        _t(50, "user", "Yeah I was impressed. Good team."),
        _t(51, "assistant", "Progress."),
        _t(52, "user", "Quick totally off-topic — do you know anything about sourdough? I want to try making it but the instructions online are all different."),
        _t(53, "assistant", "Start with a simple recipe from King Arthur Flour. They're reliable."),
        _t(54, "user", "Ok will do. Work — sprint 1 review today. Devon showed the updated mockups, now with Sarah's design polish. Looks really nice."),
        _t(55, "assistant", "How's the mood?"),
        _t(56, "user", "Good. Marcus is happy, Lillian added her secondary-mode chat idea which is now a small 'ask anything' pop-out, optional."),
        _t(57, "assistant", "Did Sarah engage with that?"),
        _t(58, "user", "She said it's fine but doesn't need to be prominent. She's fully on the traditional search train now."),
        _t(59, "assistant", "Complete arc from where she started."),
        _t(60, "user", "Really is. Anyway we're going into build."),
        _t(61, "assistant", "Good luck."),
        _t(62, "user", "Hey by the way, I tried the King Arthur recipe. Starter is bubbling after 4 days. This is more satisfying than I expected."),
        _t(63, "assistant", "It's a small wonder every time."),
        _t(64, "user", "Yeah. Build phase. Two sprints in. Devon is ripping through the keyboard features. Marcus built the indexing pipeline. Lillian is doing the visual QA."),
        _t(65, "assistant", "Sarah?"),
        _t(66, "user", "Doing the chat pop-out design work, the secondary mode. She's been more quiet in meetings. Showing up, contributing, but not driving."),
        _t(67, "assistant", "Sometimes after a big reversal, people take a back seat for a while."),
        _t(68, "user", "Yeah that's what I figured. Gave her space."),
        _t(69, "assistant", "Reasonable."),
        _t(70, "user", "Sourdough loaf #1 happened. Dense brick. But with discernible sourdough character."),
        _t(71, "assistant", "Progress."),
        _t(72, "user", "Ok so two days ago, Sarah came back with a proposal that's going to be interesting. She wants to rebuild the chat pop-out as a more prominent feature — not just a secondary mode but something that's surfaced after 3 failed keyword searches."),
        _t(73, "assistant", "Tell me more."),
        _t(74, "user", "She says she's been watching user sessions in our test environment and whenever someone strikes out 3 times in a row, chat is exactly what they need. Like a fallback for frustrated users."),
        _t(75, "assistant", "That's actually a grounded, data-backed version of her original chat enthusiasm."),
        _t(76, "user", "Yeah. It's the first time she's advocated strongly for something since her big reversal. And importantly, she's framing it from user evidence, not personal conviction."),
        _t(77, "assistant", "Does it have a chance?"),
        _t(78, "user", "The team's interested. Devon said 'this feels right.' Marcus wants to see the session data. Lillian supports it."),
        _t(79, "assistant", "Sarah's arc is remarkable — chat-first, then chat-hybrid, then traditional-only, now back to data-driven chat-as-fallback."),
        _t(80, "user", "Yeah. She's grown up in this project. Different person than at the kickoff."),
        _t(81, "assistant", "Sometimes the best design work happens after someone is humbled."),
        _t(82, "user", "Agreed. Ok sourdough #2 was much better. Not bakery-level but recognizably bread."),
        _t(83, "assistant", "You're on track."),
        _t(84, "user", "Yeah. Last project note — we implemented Sarah's fallback idea. Early metrics show 8% of users trigger it after 3 failed searches, and 80% of those find what they want via the chat."),
        _t(85, "assistant", "That's a huge save rate for a 'fallback.'"),
        _t(86, "user", "Yeah. Sarah said 'feels nice to have been right on something' and everyone laughed."),
        _t(87, "assistant", "Good outcome all around."),
        _t(88, "user", "Yeah. Design crit next week on v2."),
        _t(89, "assistant", "You'll do fine."),
        _t(90, "user", "Btw I'm at loaf #5. Getting pretty good oven spring now."),
        _t(91, "assistant", "Impressive."),
        _t(92, "user", "Thanks."),
    ]


# ===========================================================================
# Conversation 4: Ellipsis resolution + retraction detection
# A recurring "it" / "that thing" / "the one we talked about" referring to
# a specific product feature that is named ONCE and then only referred to
# implicitly across 80+ turns. Also flip-flops on launch decision.
# ===========================================================================

def _build_ellipsis_feature() -> list[dict]:
    return [
        _t(0, "user", "Ok so I was thinking about what we talked about yesterday. I think it could actually work if we scope it down."),
        _t(1, "assistant", "You're going to need to remind me what we talked about."),
        _t(2, "user", "The referral rewards system. The one with the tiered bonuses for successful referrals."),
        _t(3, "assistant", "Right. You'd mentioned tiered rewards — first referral, fifth, tenth milestones."),
        _t(4, "user", "Yeah. So I think we can ship something simple by end of month."),
        _t(5, "assistant", "What's the minimum to make it useful?"),
        _t(6, "user", "Referral link generation, tracking a successful signup, and a small reward email when the new person activates. That's it for v1."),
        _t(7, "assistant", "Reasonable scope."),
        _t(8, "user", "Yeah. I'm going to pitch it to leadership Wednesday."),
        _t(9, "assistant", "Good luck."),
        _t(10, "user", "Thanks. Hey, the new gym opens today. Going to check it out at 6."),
        _t(11, "assistant", "What kind of gym?"),
        _t(12, "user", "Climbing. Bouldering-focused but they have rope routes too."),
        _t(13, "assistant", "Nice. Have fun."),
        _t(14, "user", "Ok pitched it to leadership. They're... interested but want more analysis. What's the expected lift on user acquisition? What's the gross cost?"),
        _t(15, "assistant", "Reasonable questions."),
        _t(16, "user", "Yeah. I need to get numbers. I'll work with growth team to estimate."),
        _t(17, "assistant", "Okay."),
        _t(18, "user", "Growth team pulled data from 3 similar programs at other companies. Estimated 8-12% lift on new signups. Cost is 4% of LTV for the reward amount."),
        _t(19, "assistant", "Healthy ROI if those numbers hold."),
        _t(20, "user", "That's what I think. I'm going to re-pitch with these numbers."),
        _t(21, "assistant", "Good."),
        _t(22, "user", "Gym was fun. Also I'm terrible at climbing. Like, embarrassingly so."),
        _t(23, "assistant", "Everyone starts somewhere."),
        _t(24, "user", "Yeah. Ok re-pitched it. They green-lit the scope for v1."),
        _t(25, "assistant", "Great."),
        _t(26, "user", "We kicked off today. Team of three engineers plus me. 4-week timeline."),
        _t(27, "assistant", "Who's on the team?"),
        _t(28, "user", "Karim on backend, Riley on frontend, Peng on infrastructure. Riley is pretty new so I'll pair with her more."),
        _t(29, "assistant", "Sounds good. What's the first sprint goal?"),
        _t(30, "user", "Link generation and tracking. The data model piece."),
        _t(31, "assistant", "Straightforward. Good starting point."),
        _t(32, "user", "Yeah. Made a quick start. Karim has the database schema drafted. Simple stuff."),
        _t(33, "assistant", "Nice."),
        _t(34, "user", "Ok climbing take 2. Day 3. I fell off a route and pulled something in my shoulder."),
        _t(35, "assistant", "Oh no. Going to the doctor?"),
        _t(36, "user", "If it doesn't improve by Monday. For now just icing it."),
        _t(37, "assistant", "Take it easy."),
        _t(38, "user", "Sprint 1 review — we're ahead of schedule. Karim has the schema and API stubs. Riley is cranking on the referral link UI."),
        _t(39, "assistant", "Nice pace."),
        _t(40, "user", "Peng set up the infrastructure piece. All looking good."),
        _t(41, "assistant", "Sprint 2?"),
        _t(42, "user", "The signup tracking piece. The hardest part — linking a new signup back to the referrer's link."),
        _t(43, "assistant", "Attribution is always tricky."),
        _t(44, "user", "Yeah. We're using UTM params and first-party cookies. Should work for the majority."),
        _t(45, "assistant", "What about cross-device?"),
        _t(46, "user", "For v1 we're accepting that we'll miss some cross-device attributions. If v2 happens we'll do a proper identity resolution."),
        _t(47, "assistant", "Pragmatic."),
        _t(48, "user", "Shoulder is fine by the way. Week of icing did it."),
        _t(49, "assistant", "Glad."),
        _t(50, "user", "Going back to the gym tonight. Slower this time."),
        _t(51, "assistant", "Good."),
        _t(52, "user", "Ok big development. The CEO showed up at our standup today. Asked about our project."),
        _t(53, "assistant", "Unexpected cameo."),
        _t(54, "user", "Yeah. He wanted to know if we could launch it by end of month to coincide with the annual customer conference."),
        _t(55, "assistant", "What did you tell him?"),
        _t(56, "user", "I said we could if we skip the reward email and just show the reward in the dashboard. Not ideal but possible."),
        _t(57, "assistant", "Compromise to hit the date."),
        _t(58, "user", "Yeah. He agreed."),
        _t(59, "assistant", "Okay."),
        _t(60, "user", "Ok so the reward email thing. I was thinking more about it and I don't actually want to cut it. It's the part that drives people to share more."),
        _t(61, "assistant", "So you're reversing."),
        _t(62, "user", "Yeah. Going to tell the CEO we'll launch a week after the conference but with the full v1 scope."),
        _t(63, "assistant", "How will he take it?"),
        _t(64, "user", "Not sure. But it's the right call."),
        _t(65, "assistant", "Bold."),
        _t(66, "user", "Told him. He was annoyed but ok. So we're launching the week after the conference with the full feature."),
        _t(67, "assistant", "Good outcome."),
        _t(68, "user", "Yeah. Moving on. Sprint 3 is the email piece plus reward fulfillment."),
        _t(69, "assistant", "What's the reward exactly?"),
        _t(70, "user", "$20 credit for the referrer on successful activation. Simple."),
        _t(71, "assistant", "Clear mechanics."),
        _t(72, "user", "Gym is going well. Sent my first V3 problem yesterday."),
        _t(73, "assistant", "What's V3?"),
        _t(74, "user", "A difficulty grade. V0 is beginner, V3 is 'you're making progress.' V5+ is 'strong.'"),
        _t(75, "assistant", "Sweet. Congrats."),
        _t(76, "user", "Thanks. Alright. Hit a snag on the email work. The way our transactional email system is set up, we can't easily send on a specific trigger like 'referral activated.' We'd have to build a new email template flow."),
        _t(77, "assistant", "Who owns the email system?"),
        _t(78, "user", "Marketing ops. I have a ticket with them but it's going to take 2 weeks to get scheduled."),
        _t(79, "assistant", "That pushes your timeline."),
        _t(80, "user", "Yeah. I have to decide — wait 2 weeks and slip, or cut the email for v1 again."),
        _t(81, "assistant", "You already went through this loop."),
        _t(82, "user", "I know. Ok I'm going to bite the bullet. We'll ship the project as a two-phase: v1 without the email, then v1.5 adds the email two weeks later when marketing ops unblocks us."),
        _t(83, "assistant", "Pragmatic reversal. Back to the CEO's original timing?"),
        _t(84, "user", "Close to it. Not end of month — two days after the conference but that's close enough."),
        _t(85, "assistant", "Okay. Tell the CEO?"),
        _t(86, "user", "Yeah. He was ok with it because the delay is small. I think he appreciated that we didn't flatly miss the date."),
        _t(87, "assistant", "Good."),
        _t(88, "user", "Friday demo went great. Everyone on the team looked good."),
        _t(89, "assistant", "Launch plan?"),
        _t(90, "user", "Launch Tuesday, announcement in the keynote Wednesday."),
        _t(91, "assistant", "Nice synergy."),
        _t(92, "user", "Yeah. Ok gym news — V4 yesterday. It broke a mental wall."),
        _t(93, "assistant", "You've been climbing for how long now?"),
        _t(94, "user", "Eight weeks? Nine? Lost track."),
        _t(95, "assistant", "Steady progress."),
        _t(96, "user", "Yeah. Back to the project. It launched. 1200 referral links generated in the first week."),
        _t(97, "assistant", "Strong start."),
        _t(98, "user", "Yeah. 320 successful activations. That's a 26% conversion which is above what we modeled."),
        _t(99, "assistant", "Great."),
        _t(100, "user", "Marketing ops just confirmed the email flow is ready for next week."),
        _t(101, "assistant", "So the phase 2 addition is on track."),
        _t(102, "user", "Yeah. Should be live by next Thursday."),
        _t(103, "assistant", "Nice."),
        _t(104, "user", "Team celebration dinner Friday. Well earned."),
        _t(105, "assistant", "Agreed."),
        _t(106, "user", "One thought — the thing we were originally going to build, the full tiered version with 5th and 10th referral bonuses? I actually think we should revisit that as a v2."),
        _t(107, "assistant", "So the tiered rewards — the whole original concept — come back around."),
        _t(108, "user", "Yeah. Now that the base is live and performing, we have real data to justify the tiers."),
        _t(109, "assistant", "Good plan."),
    ]


# ===========================================================================
# Conversation 5: Commitment tracking + cross-topic bridging
# The user makes many personal commitments across the conversation. Some they
# complete, some they silently drop, some they explicitly cancel and some
# they explicitly reaffirm. Also cross-topic bridges: a fitness goal and a
# house-renovation goal are more connected than they first appear.
# ===========================================================================

def _build_commitments() -> list[dict]:
    return [
        _t(0, "user", "New year, new habits. I'm committing to three things: no alcohol for 90 days, journaling every night, and finishing that kitchen renovation that's been stalled."),
        _t(1, "assistant", "Ambitious. Which one do you think will be the hardest?"),
        _t(2, "user", "The kitchen. Alcohol and journaling are behavior. Kitchen is actual construction work on weekends."),
        _t(3, "assistant", "Fair. Do you have a plan for the kitchen?"),
        _t(4, "user", "Demolishing the old countertops first. Then cabinets, then counters. I'll do two Saturdays a month."),
        _t(5, "assistant", "Paced. Good approach."),
        _t(6, "user", "Yeah. Ok meeting prep — I have that board meeting Friday. Need to finalize the revenue projections tonight."),
        _t(7, "assistant", "I'll remind you if you'd like."),
        _t(8, "user", "Please. Day 5 — no alcohol still going. Journaling 4/5 nights, missed Tuesday. Kitchen: demo'd the west counter this weekend."),
        _t(9, "assistant", "Good progress all around."),
        _t(10, "user", "Yeah. Felt good smashing the counter."),
        _t(11, "assistant", "Cathartic."),
        _t(12, "user", "Hey I'm also going to commit to reaching out to an old mentor I haven't talked to in 3 years. It's been on my mind."),
        _t(13, "assistant", "Who?"),
        _t(14, "user", "Elena. She was my first manager. Moved to Austin. We lost touch."),
        _t(15, "assistant", "Good idea. Email?"),
        _t(16, "user", "I'll send something Friday after the board meeting."),
        _t(17, "assistant", "Got it."),
        _t(18, "user", "Speaking of — boss wants me to take on the platform migration project on top of my current stuff. She asked today."),
        _t(19, "assistant", "How did you respond?"),
        _t(20, "user", "Said I'd think about it. It's a lot."),
        _t(21, "assistant", "What's your gut?"),
        _t(22, "user", "Career-wise, yes. Work-life-wise, no. I'm already at capacity."),
        _t(23, "assistant", "Worth saying that directly if so."),
        _t(24, "user", "Yeah I'll tell her no tomorrow."),
        _t(25, "assistant", "Direct is good."),
        _t(26, "user", "Told her. She was fine about it. Moving on."),
        _t(27, "assistant", "Okay."),
        _t(28, "user", "Board meeting went well. Revenue numbers came out clean. Nobody asked anything too scary."),
        _t(29, "assistant", "Nice."),
        _t(30, "user", "Yeah. Friday night celebration. With one beer. Sorry to my 90-day goal."),
        _t(31, "assistant", "You're breaking the alcohol commitment?"),
        _t(32, "user", "Not breaking. Just... not being weird about it. One beer on a good night isn't derailing."),
        _t(33, "assistant", "Okay."),
        _t(34, "user", "Actually wait. I'm being honest with myself — if I start making exceptions, the 90 days becomes meaningless. I'm going to not have it. Dry sparkling water tonight."),
        _t(35, "assistant", "That's a strong choice."),
        _t(36, "user", "Yeah. Day 12 now alcohol-free. Journaling streak broke twice but I'm back on it. Kitchen: bought the new cabinets from the warehouse place."),
        _t(37, "assistant", "Expensive?"),
        _t(38, "user", "Less than I expected. $3200 for the full set. Custom would have been triple."),
        _t(39, "assistant", "Good win."),
        _t(40, "user", "Ok did not email Elena yet. I've been avoiding it honestly. Don't know what to say."),
        _t(41, "assistant", "Keep it simple — 'I've been thinking of you, wanted to catch up.' She'll like that."),
        _t(42, "user", "Yeah. I'll do it this weekend."),
        _t(43, "assistant", "Okay."),
        _t(44, "user", "Also committing to running a half marathon in November. Training plan starts March."),
        _t(45, "assistant", "That's a real goal. Have you run before?"),
        _t(46, "user", "5ks. This is a big jump."),
        _t(47, "assistant", "Training will be the thing that makes it feasible."),
        _t(48, "user", "Yeah. I have a plan off Runner's World. 22 weeks."),
        _t(49, "assistant", "Solid."),
        _t(50, "user", "OK sent the email to Elena. Short and honest. Just said I've missed her perspective and wanted to catch up."),
        _t(51, "assistant", "Nice."),
        _t(52, "user", "Let's see if she responds."),
        _t(53, "assistant", "Fingers crossed."),
        _t(54, "user", "Quick work thing — I volunteered to give a talk at the team offsite next month. On how our team handles incidents."),
        _t(55, "assistant", "That's a lot of prep."),
        _t(56, "user", "30 minutes. I'll prep the last weekend before. 3-4 hours of work probably."),
        _t(57, "assistant", "Okay."),
        _t(58, "user", "Day 21 alcohol-free. Got invited to a wedding in Mexico. Gonna be hard."),
        _t(59, "assistant", "When?"),
        _t(60, "user", "End of March. Still within the 90 days."),
        _t(61, "assistant", "Will you hold?"),
        _t(62, "user", "I'm going to. I'll order non-alcoholic drinks and hope nobody cares."),
        _t(63, "assistant", "Plenty won't notice."),
        _t(64, "user", "Yeah. Ok kitchen — cabinets installed this weekend. Took forever. Had to re-level the floor in two spots."),
        _t(65, "assistant", "DIY?"),
        _t(66, "user", "All me. Except for the electrical which I'm punting to a licensed electrician next Saturday."),
        _t(67, "assistant", "Smart."),
        _t(68, "user", "Yeah. I don't mess with electrical."),
        _t(69, "assistant", "Wise policy."),
        _t(70, "user", "Ok something came up — my dad fell. He's in the hospital. Not life-threatening but he broke a hip. Flying to Chicago Thursday to help my mom."),
        _t(71, "assistant", "I'm sorry. How's he doing?"),
        _t(72, "user", "Stable. Surgery tomorrow. I'll be there two weeks probably."),
        _t(73, "assistant", "Take care of yourself."),
        _t(74, "user", "Yeah. Journaling goes on pause. I know it does."),
        _t(75, "assistant", "Totally understandable."),
        _t(76, "user", "Also the half marathon plan — I was supposed to start training in March. I might shift the race to spring next year. Can't start a 22-week plan while traveling."),
        _t(77, "assistant", "Flexible is right given circumstances."),
        _t(78, "user", "Yeah. Officially moving the half marathon to April next year."),
        _t(79, "assistant", "Noted."),
        _t(80, "user", "Chicago update. Dad's recovering. Mom is stressed. Me too but keeping it together."),
        _t(81, "assistant", "How many days are you in?"),
        _t(82, "user", "Eight. Extending for another week."),
        _t(83, "assistant", "Okay. What's the kitchen situation?"),
        _t(84, "user", "Frozen mid-project. Countertops ordered but not installed. Electrical not done. It's ugly but functional enough."),
        _t(85, "assistant", "After you're home?"),
        _t(86, "user", "I'll pick it up. Goal of finishing by end of April."),
        _t(87, "assistant", "Realistic."),
        _t(88, "user", "Elena emailed back. Really warmly. Wants to FaceTime next week. She's going through her own stuff apparently."),
        _t(89, "assistant", "Glad you reached out."),
        _t(90, "user", "Yeah. Ok back home. Dad doing well. Took 17 days total."),
        _t(91, "assistant", "Welcome back."),
        _t(92, "user", "Alcohol day 48. Journaling restart from scratch, new streak is at 6 days. Kitchen back in motion — electrician came Saturday and finished the wiring."),
        _t(93, "assistant", "Good."),
        _t(94, "user", "Countertops arriving Friday. Install is Saturday. Going to be a long day."),
        _t(95, "assistant", "Okay. Anything else?"),
        _t(96, "user", "The talk at the offsite — that happened last week. Went well. Got nice feedback."),
        _t(97, "assistant", "Good."),
        _t(98, "user", "Had the first FaceTime with Elena. Three hours. Great catch-up."),
        _t(99, "assistant", "Nice."),
        _t(100, "user", "Kitchen — countertops are IN. It looks amazing. Last weekend of work to do the backsplash tile and it's done."),
        _t(101, "assistant", "Wrapping up."),
        _t(102, "user", "Yeah. Alcohol day 82. I've done the full 90 basically."),
        _t(103, "assistant", "Almost there."),
        _t(104, "user", "Yeah. Day 90 tomorrow. I don't know if I'll continue or reintroduce."),
        _t(105, "assistant", "Either works."),
        _t(106, "user", "Hit day 90. Had a glass of wine at dinner to celebrate. Was totally fine."),
        _t(107, "assistant", "Good balance."),
        _t(108, "user", "Kitchen — finished last Sunday. Tile and everything. Officially done after 14 weekends of sporadic work."),
        _t(109, "assistant", "Huge."),
        _t(110, "user", "Yeah. Funny thing — the kitchen reno actually helped the half marathon thing. Physical labor gave me fitness endurance I didn't have before. When I do start running I think I'll be further along than if I'd just been desk-sitting."),
        _t(111, "assistant", "Interesting connection."),
        _t(112, "user", "Yeah. Body's in better shape thanks to 14 weekends of manual work."),
        _t(113, "assistant", "Nice."),
    ]


# ===========================================================================
# Conversation 6: Hypothetical-to-actual matching + causal antecedents
# User discusses a hypothetical approach early. Much later they implement
# something similar. Question: how did it compare? Also causal antecedents:
# why did a customer escalation happen later?
# ===========================================================================

def _build_hypothetical_actual() -> list[dict]:
    return [
        _t(0, "user", "Brainstorming our queueing architecture. If we went with a pub/sub model with per-tenant partitioning, what would the implications be?"),
        _t(1, "assistant", "Per-tenant partitioning on pub/sub — you'd get isolation between tenants so one noisy tenant can't flood others, right?"),
        _t(2, "user", "Exactly. And if we used Kafka with a partition-per-tenant scheme, we'd get ordering guarantees per tenant."),
        _t(3, "assistant", "Kafka partition-per-tenant could scale to thousands of tenants but partition count has operational limits."),
        _t(4, "user", "Yeah, we'd probably top out around 4000 active tenants per cluster."),
        _t(5, "assistant", "And you'd need consumer groups sized to match."),
        _t(6, "user", "Right. Let's say we did this — we'd use 8-byte tenant IDs as the partition key, route everything through our ingress service. Dead-letter queue per tenant for failed messages. Retention would be 7 days."),
        _t(7, "assistant", "That's a solid hypothetical design. What are you actually evaluating against?"),
        _t(8, "user", "We're comparing this approach to a simpler shared-queue approach with tenant tags on messages. The shared approach is way simpler but doesn't give isolation."),
        _t(9, "assistant", "Classic simplicity vs. isolation tradeoff."),
        _t(10, "user", "Yeah. I'm just sketching out both and thinking about which we'd pick."),
        _t(11, "assistant", "Okay. Any other constraints to flag?"),
        _t(12, "user", "We need at-least-once semantics. And ideally idempotency at the consumer level."),
        _t(13, "assistant", "Those are universal in both approaches."),
        _t(14, "user", "Right. OK enough of that, lunch. Taco place opens at 11:30."),
        _t(15, "assistant", "Enjoy."),
        _t(16, "user", "Back. I want to mention — totally unrelated — we had a customer site visit planned next week for Zenrich, our biggest logo. Getting ready."),
        _t(17, "assistant", "Who's visiting?"),
        _t(18, "user", "Their whole platform team. 8 people. It's a quarterly business review plus technical deep dive."),
        _t(19, "assistant", "Any topics they're especially focused on?"),
        _t(20, "user", "They've been asking about our roadmap for multi-region support. Their data sovereignty requirements are tightening."),
        _t(21, "assistant", "Multi-region is a big feature request."),
        _t(22, "user", "Yeah. We don't have a firm plan yet. I'll tell them 'next year' honestly."),
        _t(23, "assistant", "Honesty's usually better than false promises."),
        _t(24, "user", "Yeah. Zenrich visit went fine. They seemed satisfied. Mentioned a couple of smaller concerns about our API rate limiting."),
        _t(25, "assistant", "What specifically?"),
        _t(26, "user", "They said they occasionally hit 429 errors during their batch processing windows. Not frequent but annoying."),
        _t(27, "assistant", "Is their plan tier-appropriate for their traffic?"),
        _t(28, "user", "They're on Enterprise but their rate limits haven't been raised even though their usage grew 3x in 6 months."),
        _t(29, "assistant", "Sounds like a simple account fix."),
        _t(30, "user", "Yeah I'll file an internal ticket to raise their limits."),
        _t(31, "assistant", "Good."),
        _t(32, "user", "Something else off-topic — I had a dentist appointment today. Cavity."),
        _t(33, "assistant", "Ouch. Filling?"),
        _t(34, "user", "Next Thursday."),
        _t(35, "assistant", "Annoying."),
        _t(36, "user", "Yeah. Back to work — my team has been debating the queueing thing for a few days more. We're going to make the call next week."),
        _t(37, "assistant", "Which way is the team leaning?"),
        _t(38, "user", "Split. Infrastructure folks like the simple shared-queue approach because it's less to operate. App team likes the per-tenant approach because of isolation."),
        _t(39, "assistant", "Classic infra vs. app tension."),
        _t(40, "user", "Yeah. OK call was made. We're going with per-tenant Kafka partitions."),
        _t(41, "assistant", "So the hypothetical became actual."),
        _t(42, "user", "Yeah. Actually let me tell you about how it's going."),
        _t(43, "assistant", "Please."),
        _t(44, "user", "We set it up with per-tenant partitions, tenant ID as partition key, dedicated DLQs. Retention is 14 days actually, not 7 — infrastructure team wanted more buffer for debugging."),
        _t(45, "assistant", "Sensible revision."),
        _t(46, "user", "Yeah. And we're using 12-byte tenant IDs instead of 8 because we ran into collisions in testing."),
        _t(47, "assistant", "Interesting, collisions at 8 bytes on... how many tenants?"),
        _t(48, "user", "Fewer than we expected. Something like 200 tenants showed a birthday-paradox-adjacent collision in hash bucketing. 12 bytes gave us room."),
        _t(49, "assistant", "Implementation detail that matters."),
        _t(50, "user", "Yeah. Cluster-wise we're running 3 clusters instead of 1 because of a regional setup. Each one can hold about 4000 tenants which matches what I'd sketched."),
        _t(51, "assistant", "Total capacity around 12,000 tenants then."),
        _t(52, "user", "Right. Way more than we need."),
        _t(53, "assistant", "Good."),
        _t(54, "user", "Dentist was fine. Filling was quick."),
        _t(55, "assistant", "Glad."),
        _t(56, "user", "Alright new topic — Zenrich escalated. They had a 4-hour outage on their side because our system started throttling them hard Monday morning."),
        _t(57, "assistant", "Whoa. Rate limits again?"),
        _t(58, "user", "Yeah. Apparently my ticket to raise their limits never got resolved. It sat in the queue for 3 weeks."),
        _t(59, "assistant", "How did they escalate?"),
        _t(60, "user", "Their CTO emailed our CEO directly. It was unpleasant."),
        _t(61, "assistant", "Understandable."),
        _t(62, "user", "Yeah. Fixed the limits immediately obviously. And we're doing a postmortem on how the ticket got lost."),
        _t(63, "assistant", "Key to catch the ticket-routing process issue, not just the immediate problem."),
        _t(64, "user", "Agreed."),
        _t(65, "assistant", "Anything else?"),
        _t(66, "user", "The queueing system has been live for 3 weeks now. Mostly good. One notable thing — the per-tenant isolation saved us during a noisy-neighbor incident last week. One tenant had a bug that produced 10x their normal message volume for 45 minutes."),
        _t(67, "assistant", "And it didn't affect other tenants?"),
        _t(68, "user", "Right. In the old shared-queue world that would have backed up everyone."),
        _t(69, "assistant", "Validates the isolation tradeoff."),
        _t(70, "user", "Yeah. Exactly what we designed for."),
        _t(71, "assistant", "Nice."),
        _t(72, "user", "One thing that's been tougher than expected: the operations load. Each cluster has its own monitoring dashboard and we're on-call rotation for 3 clusters instead of 1 queue. Infrastructure team is feeling it."),
        _t(73, "assistant", "Simple-vs-isolation tradeoff materialized."),
        _t(74, "user", "Yeah. We're evaluating if cluster consolidation is possible — maybe 2 clusters instead of 3."),
        _t(75, "assistant", "Could halve the ops toil."),
        _t(76, "user", "Right. It's a TBD. I'll think about it."),
        _t(77, "assistant", "Okay."),
        _t(78, "user", "Oh, Zenrich postmortem is complete. The ticket got lost because our account manager resigned the day after I filed it and their tickets were reassigned to a person who was already overwhelmed."),
        _t(79, "assistant", "So the chain: rate limit ticket → AM resignation → reassignment to overworked person → ticket starvation → hit limits → outage."),
        _t(80, "user", "Exactly. And apparently this has happened before at lower severity. We're adding SLA-based alerting on account management tickets."),
        _t(81, "assistant", "Good outcome from a bad event."),
        _t(82, "user", "Yeah. Zenrich is okay. They got credits and a call from our CEO."),
        _t(83, "assistant", "Good."),
        _t(84, "user", "Last note on the queueing project — we added one thing to the plan that wasn't in the original hypothetical: cross-region replication for 5 specific tenants who need disaster recovery guarantees."),
        _t(85, "assistant", "Wasn't part of the original design."),
        _t(86, "user", "Right. Customer-driven addition. Actually it's a subset of the multi-region work Zenrich was asking about."),
        _t(87, "assistant", "So Zenrich indirectly shaped the queueing system's evolution."),
        _t(88, "user", "Yeah. Everything connects."),
        _t(89, "assistant", "It does."),
        _t(90, "user", "Ok week off now, going hiking."),
        _t(91, "assistant", "Enjoy. Where?"),
        _t(92, "user", "White Mountains. Few days of huts."),
        _t(93, "assistant", "Have fun."),
        _t(94, "user", "Thanks."),
    ]


# ===========================================================================
# Conversation 7: Retraction-heavy flip-flop on database choice + perspective swap
# Multiple decision reversals on database choice: starts Postgres, switches
# to Mongo, switches to Postgres+Mongo split, switches back to Mongo-only,
# finally ends on CockroachDB. Also track what one team member (Hiroki) thinks.
# ===========================================================================

def _build_database_flipflop() -> list[dict]:
    return [
        _t(0, "user", "Starting the data layer design for our new service. We need to pick a database."),
        _t(1, "assistant", "What's the service?"),
        _t(2, "user", "A workflow automation engine. Stores workflow definitions (structured), execution state (structured), and per-execution event logs (semi-structured)."),
        _t(3, "assistant", "Mix of structured and semi-structured."),
        _t(4, "user", "Yeah. Gut feel is to go Postgres. We know it, we have operational expertise, JSONB handles the semi-structured stuff."),
        _t(5, "assistant", "Postgres is a reasonable default."),
        _t(6, "user", "Discussed it with the team. Nolan agreed. Hiroki wants to push for Mongo — says the event logs will really be more document-y than we think."),
        _t(7, "assistant", "Any data to support Hiroki's view?"),
        _t(8, "user", "He pulled examples from a similar system he built before. The events had deeply nested objects with varying schemas."),
        _t(9, "assistant", "Mongo handles that pattern well."),
        _t(10, "user", "Yeah. Decision for now — Postgres. We'll see how the event schemas actually look."),
        _t(11, "assistant", "Okay. Prototype first?"),
        _t(12, "user", "Yes. Quick prototype."),
        _t(13, "assistant", "Good."),
        _t(14, "user", "Two weeks in. The event data is getting hairy. Every new workflow type has different event payload shapes and we're bending the schema."),
        _t(15, "assistant", "Hiroki's concern materializing?"),
        _t(16, "user", "Somewhat. Nolan still wants to stick with Postgres using JSONB. Hiroki is saying the JSONB indices don't work well for our query patterns."),
        _t(17, "assistant", "That's a real concern. Partial indexes on JSONB can be tricky."),
        _t(18, "user", "Yeah. I'm leaning toward switching to Mongo for at least the event logs."),
        _t(19, "assistant", "A split?"),
        _t(20, "user", "Postgres for definitions and execution state, Mongo for events. Hiroki proposed it and I like it."),
        _t(21, "assistant", "Polyglot approach."),
        _t(22, "user", "Yeah. OK, I'm not going to wait. Decision revised: Postgres for structured stuff, Mongo for event logs."),
        _t(23, "assistant", "Got it."),
        _t(24, "user", "Random thing — going to try a new cuisine this weekend. My wife wants Ethiopian food. Never had it."),
        _t(25, "assistant", "Injera plus stews. Shared plate. Worth trying."),
        _t(26, "user", "Cool. Ok update — we've started the Mongo prototype for events. Nolan is grumpy. He hates operating two databases."),
        _t(27, "assistant", "Operational load is a real cost."),
        _t(28, "user", "Yeah. He keeps bringing it up."),
        _t(29, "assistant", "Noted."),
        _t(30, "user", "Big development today. Nolan did a deep dive on Postgres JSONB with GIN indexes and showed a benchmark where the event log queries actually perform well — within 20% of Mongo. He's really pushing to go back to Postgres-only."),
        _t(31, "assistant", "Interesting."),
        _t(32, "user", "Hiroki is not happy. He thinks the benchmark doesn't reflect real usage patterns."),
        _t(33, "assistant", "Do you agree?"),
        _t(34, "user", "I think Nolan's benchmark is more like 80% representative."),
        _t(35, "assistant", "20% gap in representativeness could hide a lot."),
        _t(36, "user", "Yeah. But for operational simplicity, 80% representative + one database is appealing."),
        _t(37, "assistant", "Trade."),
        _t(38, "user", "Ok I'm going to reverse again. Postgres-only with JSONB. I know this is the third flip. Nolan makes a strong case."),
        _t(39, "assistant", "Logged."),
        _t(40, "user", "Ethiopian was amazing by the way. The doro wat and the awaze beef. We're going back next week."),
        _t(41, "assistant", "Good find."),
        _t(42, "user", "Yeah. Ok work. Four weeks into Postgres-only. And we just hit a real wall. The query patterns for cross-workflow event analytics are doing huge in-memory joins that can't be indexed. We're talking query times of 45 seconds for common dashboards."),
        _t(43, "assistant", "That's not going to work at scale."),
        _t(44, "user", "No. Hiroki is doing the 'I told you so' dance internally."),
        _t(45, "assistant", "And?"),
        _t(46, "user", "Nolan is stuck on how to index our way out. He has a plan involving materialized views and partial indexes but it's complex."),
        _t(47, "assistant", "Complexity of indexes vs. operational complexity of a second DB."),
        _t(48, "user", "Yeah. Actually there's a third option we haven't really considered. Hiroki floated CockroachDB today."),
        _t(49, "assistant", "Why CockroachDB?"),
        _t(50, "user", "Postgres-compatible but distributed from the start. Better for the scale we're expecting. Plus has native JSON."),
        _t(51, "assistant", "Interesting pivot."),
        _t(52, "user", "I need to think."),
        _t(53, "assistant", "Take the time."),
        _t(54, "user", "Went Ethiopian again. Tibs this time. Good."),
        _t(55, "assistant", "Becoming a regular."),
        _t(56, "user", "OK database decision. We're going back to the Mongo split. Postgres for structured, Mongo for events."),
        _t(57, "assistant", "The 4th reversal. Back to the 2nd position."),
        _t(58, "user", "Yeah. Rationale is we need to move quickly and the Mongo path is proven in Hiroki's prior experience. CockroachDB is interesting but we'd be learning a new DB."),
        _t(59, "assistant", "Pragmatic."),
        _t(60, "user", "Nolan is grumpy again but accepted it."),
        _t(61, "assistant", "Good for morale that the call was made."),
        _t(62, "user", "Yeah. Ok started building on the Mongo + Postgres split. 3 weeks in. Mostly working."),
        _t(63, "assistant", "Good."),
        _t(64, "user", "Hit another snag. Cross-DB transactions. We have cases where we need to update execution state (Postgres) and append events (Mongo) atomically. Without distributed transactions, we're doing workaround patterns — event-sourcing-like outbox."),
        _t(65, "assistant", "The outbox pattern works but adds complexity."),
        _t(66, "user", "Yeah. Hiroki actually came back today and said — and this was unusual for him — that he thinks we should reconsider CockroachDB. Because with CRDB, we'd have the JSON support AND distributed transactions AND one database."),
        _t(67, "assistant", "He's shifted his position."),
        _t(68, "user", "Yeah. He was the Mongo advocate and now he's pushing away from it."),
        _t(69, "assistant", "Because of the outbox complexity?"),
        _t(70, "user", "That's his framing. He said, 'I was right that we needed document support, but I was wrong that Mongo was the way.'"),
        _t(71, "assistant", "Honest of him."),
        _t(72, "user", "Yeah. Impressed."),
        _t(73, "assistant", "So is CRDB a real contender now?"),
        _t(74, "user", "I think so. I talked to Nolan. He's more open than I expected — CRDB being Postgres-compatible is a big win for him."),
        _t(75, "assistant", "So both main stakeholders are converging."),
        _t(76, "user", "Yeah. I'm going to propose migrating to CRDB. Full reset."),
        _t(77, "assistant", "That would be the 5th decision."),
        _t(78, "user", "Yeah. 5th and hopefully final. The previous 4 decisions were right given what we knew at each point but the constraints kept evolving."),
        _t(79, "assistant", "Sounds like honest iteration, not indecision."),
        _t(80, "user", "Agreed. I hope leadership sees it that way."),
        _t(81, "assistant", "You can frame it that way."),
        _t(82, "user", "Yeah. Proposed to leadership. Some eye rolls but they agreed. Migration starts next sprint."),
        _t(83, "assistant", "Good."),
        _t(84, "user", "Reposting: the final database is CockroachDB, not Postgres, not Mongo, not the split. CockroachDB."),
        _t(85, "assistant", "Final."),
        _t(86, "user", "Yeah. Moving on. Any recs for Ethiopian coffee ceremony stuff to try?"),
        _t(87, "assistant", "Look up 'bunna' ceremony — fresh-roasted beans ground, brewed three rounds. Full experience."),
        _t(88, "user", "Cool. Going to find a place."),
        _t(89, "assistant", "Enjoy."),
        _t(90, "user", "Alright. CRDB migration is done. 6 weeks. Working well. Event queries are fast, distributed transactions handle the atomicity cases, and Nolan is happy operating one database again."),
        _t(91, "assistant", "Happy ending."),
        _t(92, "user", "Yeah. Hiroki admitted that his initial Mongo push was more about the document shape than the database itself. He'd have been happy with any JSON-capable DB. So CRDB was the right answer all along, we just took a scenic route."),
        _t(93, "assistant", "Five reversals later."),
        _t(94, "user", "Yeah. Expensive learning but the team aligned."),
        _t(95, "assistant", "Onward."),
    ]


# ===========================================================================
# Builder registry
# ===========================================================================

def build_conversations() -> dict[str, list[dict]]:
    return {
        "xadv_launch_causal": _build_launch_causal(),
        "xadv_cloud_reorg": _build_cloud_reorg(),
        "xadv_perspective_swap": _build_perspective_swap(),
        "xadv_ellipsis_feature": _build_ellipsis_feature(),
        "xadv_commitments": _build_commitments(),
        "xadv_hypothetical_actual": _build_hypothetical_actual(),
        "xadv_db_flipflop": _build_database_flipflop(),
    }


# ===========================================================================
# Questions
# ===========================================================================

QUESTIONS: list[dict] = [
    # === causal_antecedents (xadv_launch_causal) ===
    {
        "conversation_id": "xadv_launch_causal",
        "category": "causal_antecedents",
        "question_index": 0,
        "question": "Why did the Nimbus launch slip from March 3 to April 7?",
        # Key causes were discussed BEFORE the slip was acknowledged:
        # - Jules's resignation risk (turn 12, 14, 16) and actual resignation (36, 42)
        # - Analytics ingestion issue (22, 24, 30, 90, 92)
        # - SSO scope creep (44, 46, 48, 50)
        # - Onboarding scope creep entangling with SSO (78, 80, 82)
        "source_chat_ids": [12, 14, 22, 24, 30, 36, 42, 44, 46, 48, 50, 78, 80, 82, 90, 92, 100],
        "ideal_response": "Multiple causes, each discussed before the slip was officially acknowledged. (1) Loss of Jules, the senior backend lead on the auth service — she was approached by Airtable weeks earlier, the user deferred her retention raise to her April review, and she resigned creating a 3-4 week senior coverage gap overlapping the launch window. (2) Analytics ingestion layer failures: Vikram's data platform team discovered the issue required 4-6 weeks of schema work due to coupled consumers, blocking the real-time telemetry Nimbus needed from day one. (3) SSO scope addition from Dana the PM (driven by two beta customers who couldn't evaluate without it), estimated at 3 weeks cold but realistically 4-5 with Jules-transition ramp. (4) Onboarding and tutorial work entangled with SSO and grew by an additional ~2 weeks (originally budgeted 3 days). The slip was acknowledged in turn 100; every one of these causes was discussed earlier without being labeled as a delay risk.",
        "benchmark": "extra_adversarial",
    },
    {
        "conversation_id": "xadv_hypothetical_actual",
        "category": "causal_antecedents",
        "question_index": 1,
        "question": "Why did Zenrich have a 4-hour outage?",
        "source_chat_ids": [24, 26, 28, 30, 56, 58, 62, 78, 80],
        "ideal_response": "Cause chain: (1) During Zenrich's quarterly business review they flagged occasional 429 rate-limit errors; their Enterprise-tier rate limits hadn't been raised despite 3x usage growth in 6 months. (2) The user filed an internal ticket to raise their limits. (3) The user's Zenrich account manager resigned the day after the ticket was filed, and Zenrich's tickets were reassigned to an already-overloaded person. (4) The ticket sat unresolved for 3 weeks. (5) On Monday morning Zenrich's batch processing hit the unchanged limits and the system throttled them hard, causing a 4-hour outage on their side, leading to a CTO-to-CEO escalation. The user's initial conversation about their rate-limiting complaint and about filing the ticket contains the causal antecedents — none of those turns said 'this will cause an outage.'",
        "benchmark": "extra_adversarial",
    },
    # === counterfactual (xadv_cloud_reorg) ===
    {
        "conversation_id": "xadv_cloud_reorg",
        "category": "counterfactual",
        "question_index": 2,
        "question": "If the company had chosen GCP instead of AWS, what would be different?",
        "source_chat_ids": [4, 10, 12, 14, 16, 30, 32, 36, 46, 48, 50, 52, 54, 56, 60, 66, 68, 74, 76, 78],
        "ideal_response": "Differences vs. the AWS decision: (1) ML workloads would run on Vertex AI — 30% cheaper and 2x faster setup per Ben's benchmarks, saving about $200k/year. (2) The two government-contract LOIs worth ~$12M ARR would be at risk because three needed compliance certifications are AWS-only (GCP in-progress with no firm date). (3) The company would face a capability crisis after the platform team reorg, because the two engineers with GCP experience were being let go — GCP adoption would have required hiring or retraining from zero. (4) Priya (cloud lead) had previous negative experience with GCP docs. (5) The committee vote (4-1) would have flipped — only Ben was for GCP. Demetri's compliance framing plus Priya's expertise concern were decisive. Would lose: compliance speed, expertise continuity. Would gain: marginal ML savings, better ML tooling. Net: the reorg's capability impact alone made AWS the pragmatic pick — per Demetri, 'the cloud that matches the team we'll have in six months.'",
        "benchmark": "extra_adversarial",
    },
    {
        "conversation_id": "xadv_db_flipflop",
        "category": "counterfactual",
        "question_index": 3,
        "question": "If the team had stayed with Postgres-only instead of eventually switching to CockroachDB, what problems would have persisted?",
        "source_chat_ids": [4, 6, 8, 14, 16, 30, 32, 34, 42, 44, 46, 48, 50, 66, 70, 74, 90, 92],
        "ideal_response": "Problems that would have persisted under Postgres-only: (1) JSONB indexing issues with varying event payload shapes — schema was being bent for every new workflow type (Hiroki's original concern, turn 14-16). (2) Cross-workflow analytics queries hitting 45-second query times due to large un-indexable in-memory joins. (3) Nolan's proposed rescue via materialized views and partial indexes adds significant operational complexity. (4) Postgres-alone would lack CRDB's distributed transaction support, which was a separate motivator when the Mongo split emerged. (5) Wouldn't have the distributed-from-the-start property CRDB provides for future scale. The team would have been stuck in indexing complexity to emulate what CRDB supports natively. Alternative missed: Mongo or Postgres+Mongo split. Mongo-only would have lost the structured workflow-definition/execution-state ergonomics; the split introduced cross-DB transaction complexity (the outbox pattern) that ultimately pushed Hiroki to reverse his own Mongo advocacy toward CRDB.",
        "benchmark": "extra_adversarial",
    },
    # === perspective_swap (xadv_perspective_swap) — Sarah's arc ===
    {
        "conversation_id": "xadv_perspective_swap",
        "category": "perspective_swap",
        "question_index": 4,
        "question": "How did Sarah's position on the search redesign evolve throughout the project?",
        "source_chat_ids": [2, 4, 10, 20, 24, 26, 32, 34, 38, 44, 46, 48, 58, 66, 72, 74, 76, 84, 86],
        "ideal_response": "Sarah's stance evolved through four distinct positions: (1) Chat-first (kickoff): fresh from a conference, pitched a full conversational search redesign — 'the future is not keyword matching, it's dialog.' (2) Chat-hybrid softening: after user-interview data showed power users wanted speed and keyboard shortcuts, she said 'maybe chat-first was too aggressive given the data, but a hybrid has potential.' (3) Full traditional-search endorsement: after sitting with Devon's mockups, she came to a 1:1 and said 'I think he's actually right. Users don't care about distinctiveness, they care about getting what they need. I was overreaching' — she then helped polish Devon's mockups and said the chat pop-out didn't need to be prominent. (4) Data-driven chat-as-fallback: after observing test-environment user sessions, she proposed surfacing chat after 3 failed keyword searches as a fallback for frustrated users. This final proposal landed, shipped, and saw 8% of users triggering it with 80% success rate. The arc is chat-first -> chat-hybrid -> traditional-only -> data-driven chat-fallback — four positions, the last one grounded in evidence rather than conviction.",
        "benchmark": "extra_adversarial",
    },
    {
        "conversation_id": "xadv_db_flipflop",
        "category": "perspective_swap",
        "question_index": 5,
        "question": "How did Hiroki's position on the database choice evolve?",
        "source_chat_ids": [6, 8, 16, 18, 20, 32, 44, 48, 50, 66, 68, 70, 92],
        "ideal_response": "Hiroki's positions: (1) Early: push for Mongo for event logs, backed by prior-project examples with nested schemas. (2) When the initial Postgres prototype's schema-bending showed, reaffirmed the Mongo need, proposed a Postgres-for-structured + Mongo-for-events split. (3) When Nolan's JSONB benchmark moved the decision back to Postgres-only, disagreed — said benchmarks didn't reflect real usage. (4) After the Postgres-only 45-second analytics query wall hit, did an internal 'I told you so' and first floated CockroachDB as a third option. (5) After the Mongo-split outbox complexity hit, explicitly reversed his own advocacy: 'I was right that we needed document support, but I was wrong that Mongo was the way' — pushing actively toward CRDB. (6) Final reflection post-migration: said his original Mongo push was really about document shape, not the database, so CRDB was the right answer from the start. Arc: pro-Mongo -> pro-split -> anti-Postgres -> neutral-introducing-CRDB -> anti-Mongo-pro-CRDB.",
        "benchmark": "extra_adversarial",
    },
    # === ellipsis_resolution (xadv_ellipsis_feature) ===
    {
        "conversation_id": "xadv_ellipsis_feature",
        "category": "ellipsis_resolution",
        "question_index": 6,
        "question": "What was the project the user was working on, and what were its main milestones and setbacks?",
        # This question relies on resolving "what we talked about", "it",
        # "the project", "the thing", "the feature" — the system is named
        # ONCE (turn 2) as "the referral rewards system."
        "source_chat_ids": [0, 2, 4, 6, 14, 18, 24, 26, 38, 42, 54, 56, 60, 62, 66, 76, 80, 82, 90, 96, 98, 106, 108],
        "ideal_response": "The project is the referral rewards system (named explicitly only in turn 2). Milestones and setbacks in order: (1) Initial scope narrowed from a tiered-rewards concept to a v1 of link generation, signup tracking, and a $20 reward email. (2) Leadership asked for ROI analysis; growth team modeled 8-12% signup lift at 4% LTV cost, unlocking green-light. (3) Kickoff with 3 engineers (Karim, Riley, Peng) on a 4-week timeline. (4) CEO pushed to launch at the customer conference; user initially agreed to cut the reward email, then reversed and rescheduled to post-conference. (5) Hit an email-system snag (marketing ops couldn't trigger transactional emails easily, 2-week delay), decided to ship v1 without email and add it in v1.5. (6) Launched successfully — 1200 referral links, 320 activations (26% conversion, above model). (7) Email flow (v1.5) confirmed for next Thursday post-launch. (8) Revisit of original tiered-rewards scope as v2 now that base data supports justification.",
        "benchmark": "extra_adversarial",
    },
    {
        "conversation_id": "xadv_ellipsis_feature",
        "category": "ellipsis_resolution",
        "question_index": 7,
        "question": "What were all the timeline reversals on the project that was being built?",
        "source_chat_ids": [4, 8, 24, 26, 54, 56, 60, 62, 66, 76, 80, 82, 86, 90, 96, 100],
        "ideal_response": "The project (referral rewards system) had three explicit timeline reversals. (1) Initial: user pitches end-of-month target; leadership asks for more analysis delaying the pitch. Then re-pitched and green-lit with a 4-week timeline. (2) CEO intervention: asked for launch-at-conference. User agreed to drop the reward email to hit the date, then reversed the next day to protect v1 scope, pushing launch a week past the conference. (3) Email-system snag: marketing ops backlog meant 2 weeks to unblock; user reversed again to ship v1 without email (reinstating the earlier cut), with a phased v1.5 for the email two weeks later — landing 2 days after the conference instead of end of month. Net: the product shipped nearly on the CEO's original timing but with the email arriving in a phase-2 increment, which was arguably the best of the paths considered despite multiple reversals.",
        "benchmark": "extra_adversarial",
    },
    # === cross_topic_bridging ===
    {
        "conversation_id": "xadv_cloud_reorg",
        "category": "cross_topic_bridging",
        "question_index": 8,
        "question": "How are the engineering reorg and the cloud provider decision related?",
        "source_chat_ids": [22, 24, 26, 28, 60, 62, 64, 66, 68, 70, 76, 78, 80, 94, 96],
        "ideal_response": "They were discussed as separate tracks and the user initially tried to keep them apart ('cloud decision is cloud decision'), but they were deeply linked: (1) The reorg dissolved the platform team which is 'the cloud expertise pool.' (2) Three of Priya's five engineers were scattered to product teams; two were let go. (3) Critically, the two being let go were the only engineers with GCP experience. (4) This meant if GCP won, the company would have no hands-on GCP expertise post-reorg. (5) Priya made this point in the evaluation committee and it effectively killed the multi-cloud AWS+GCP option too, because multi-cloud with no GCP expertise was untenable. (6) Demetri captured the framing: 'We're picking the cloud that matches the team we'll have in six months, not the team we have now.' (7) In retrospect the user noted the reorg announcement predated the first cloud evaluation meeting, so the capability loss was baked in from day one — 'the moment the CTO committed to dissolving platform, AWS became the right answer.' The reorg was formally outside the RFP criteria but determined the outcome.",
        "benchmark": "extra_adversarial",
    },
    {
        "conversation_id": "xadv_commitments",
        "category": "cross_topic_bridging",
        "question_index": 9,
        "question": "How did the kitchen renovation relate to the user's half-marathon plan?",
        "source_chat_ids": [0, 2, 4, 36, 64, 76, 78, 108, 110, 112],
        "ideal_response": "The two were discussed as completely separate commitments — kitchen renovation as a weekend-construction goal, half-marathon as an endurance-running goal deferred after a family emergency. Their connection surfaced only at the end: the user reflected that 14 weekends of manual kitchen labor (demolition, cabinet installation, levelling floors, tile work) built physical endurance such that when the half-marathon training actually starts (now shifted to April next year), they expect to be in better shape than if they'd been desk-sitting. The kitchen reno inadvertently preserved fitness progress during the period when the training plan was postponed. The connection is causal but implicit — neither goal was framed as supporting the other until the closing reflection.",
        "benchmark": "extra_adversarial",
    },
    # === retraction_detection ===
    {
        "conversation_id": "xadv_db_flipflop",
        "category": "retraction_detection",
        "question_index": 10,
        "question": "What database did the team ultimately choose?",
        "source_chat_ids": [22, 38, 56, 76, 82, 84, 90, 94],
        "ideal_response": "CockroachDB. The full decision chain (5 reversals): (1) Initial: Postgres. (2) First reversal: Postgres + Mongo split after schema-bending problems on event data. (3) Second reversal: back to Postgres-only after Nolan's JSONB benchmark. (4) Third reversal: back to Postgres + Mongo split after 45-second analytics queries hit a wall. (5) Fourth reversal (final): CockroachDB, after outbox-pattern complexity motivated Hiroki to reverse his own Mongo advocacy. CockroachDB is the final answer; the previous four were superseded. Note that 'Postgres+Mongo split' appears twice in the history but is NOT the final state.",
        "benchmark": "extra_adversarial",
    },
    {
        "conversation_id": "xadv_ellipsis_feature",
        "category": "retraction_detection",
        "question_index": 11,
        "question": "What was the final launch plan for the project, including scope and timing?",
        "source_chat_ids": [4, 6, 54, 56, 60, 62, 66, 80, 82, 86, 90, 96, 100, 104],
        "ideal_response": "Final launch: ship v1 WITHOUT the reward email two days after the customer conference (not end of month, not the full v1 originally scoped). v1 scope: referral link generation, signup tracking via UTM + first-party cookies (with accepted cross-device gaps), and a $20 reward shown in-dashboard. Launch happened on that Tuesday with keynote announcement Wednesday. Email flow (v1.5) is planned to follow two weeks later once marketing ops unblocks. Results after one week: 1200 links, 320 activations, 26% conversion above model. Note the retraction chain: the user initially targeted end-of-month, then accepted cutting the email to meet conference timing, then reversed to reject cutting the email (pushing past the conference), then reversed again to reinstate cutting the email in exchange for a two-phase launch that delivered the email later. Only the final state (no-email-in-v1, 2 days post-conference, v1.5 for email) is correct.",
        "benchmark": "extra_adversarial",
    },
    {
        "conversation_id": "xadv_perspective_swap",
        "category": "retraction_detection",
        "question_index": 12,
        "question": "What did the team ultimately build for search?",
        "source_chat_ids": [22, 36, 44, 48, 54, 56, 58, 72, 74, 76, 84, 86],
        "ideal_response": "A traditional keyword-centric search (Devon's mockups, polished by Sarah) as the primary interface, with Sarah's data-driven chat fallback triggered after 3 failed keyword searches. The conversation went through several contending designs: Sarah's initial chat-first (rejected after user interviews showed power users wanted speed/keyboard), Lillian's chat-as-secondary-mode (implemented as an 'ask anything' pop-out but de-emphasized), and Sarah's final chat-fallback-after-3-failed-searches (implemented and 8% of users triggered it with 80% success). The final product is: traditional search primary + pop-out optional + chat fallback on failed-search patterns. Initial 'chat-first' and 'no chat at all' positions were both retracted — only the final hybrid with specific triggers is what shipped.",
        "benchmark": "extra_adversarial",
    },
    # === commitment_tracking (xadv_commitments) ===
    {
        "conversation_id": "xadv_commitments",
        "category": "commitment_tracking",
        "question_index": 13,
        "question": "Of all the personal commitments the user made in the conversation, which ones did they ultimately follow through on, and which did they explicitly cancel or drop?",
        # Key: distinguish among
        # - 90-day no alcohol (followed through, day 90 hit)
        # - journaling every night (broken multiple times, restarted, eventually maintained partial)
        # - finish kitchen (completed)
        # - reach out to Elena (done, FaceTime happened)
        # - half marathon in November (explicitly pushed to April next year - counts as cancelled for original timing, replacement committed)
        # - give talk at offsite (done)
        # - take on platform migration (DECLINED — this was offered, the user committed to declining)
        # Meditation 25min/morning commit was mentioned in OTHER convo (launch), so NOT in this one. Good, keep focused.
        "source_chat_ids": [0, 2, 4, 12, 16, 26, 30, 32, 34, 36, 44, 46, 50, 54, 56, 74, 76, 78, 88, 96, 98, 100, 102, 104, 106, 108],
        "ideal_response": "Commitments followed through: (1) 90-day no-alcohol — reached day 90 (there was a brief moment of considering a beer on a good Friday night that was self-reversed). (2) Finish the stalled kitchen renovation — done after 14 sporadic weekends including demolition, cabinet install, electrical via licensed electrician, countertops, and backsplash tile. (3) Reach out to mentor Elena — email sent (initially avoided for a week), she replied warmly, 3-hour FaceTime happened. (4) Declining the platform migration — committed to saying no to boss, actually said no, she accepted. (5) Giving the team offsite talk on incident handling — done, good feedback. Commitments dropped or pushed: (1) Journaling every night — broke streak multiple times (missed Tuesday on day 5, paused entirely during 17-day Chicago trip for father's hip injury, restarted from scratch afterwards); the nightly-every-night promise was not maintained. (2) Half-marathon in November — explicitly pushed to April of the following year due to travel during the training window. The user stated 'Officially moving the half marathon to April next year.' The original November commitment is effectively cancelled and replaced with a new commitment.",
        "benchmark": "extra_adversarial",
    },
    {
        "conversation_id": "xadv_launch_causal",
        "category": "commitment_tracking",
        "question_index": 14,
        "question": "What personal commitments did the user make during the Nimbus launch conversation, and which did they keep?",
        # Commitments in xadv_launch_causal:
        # - run a 10k by end of summer (maintained; reported running Tuesdays/Thursdays 4 miles)
        # - book Catskills anniversary weekend (explicitly cancelled, then rebooked Berkshires)
        # - write onboarding spec Thursday afternoon (implicit, not clearly followed up)
        # - 25 min meditation every morning for a month (explicitly cancelled after 4 days)
        "source_chat_ids": [18, 20, 32, 34, 54, 56, 72, 74, 76, 86, 88, 98, 106, 108],
        "ideal_response": "Commitments and their outcomes: (1) Running a 10k by end of summer — KEPT and on track; the user started Tuesday/Thursday lunch runs and reported 4 miles feeling good. (2) Anniversary weekend at the Catskills cabin (May 10-12) — EXPLICITLY CANCELLED due to Jules's resignation and SSO scope stress; replaced with a Berkshires weekend April 12-14 after the new launch date. (3) 25 minutes of daily morning meditation for a month — EXPLICITLY CANCELLED after 4 days ('my mornings are too chaotic. Maybe after launch'). (4) Writing the post-signup tutorial spec by end of week (with Thursday afternoon blocked) — implicitly committed, follow-through unclear from the conversation. Net: one major professional/personal commitment was cancelled (Catskills) and compensated with a replacement trip; one health commitment (meditation) was cancelled without replacement; one aspirational goal (10k) was kept and is progressing.",
        "benchmark": "extra_adversarial",
    },
    # === hypothetical_to_actual (xadv_hypothetical_actual) ===
    {
        "conversation_id": "xadv_hypothetical_actual",
        "category": "hypothetical_to_actual",
        "question_index": 15,
        "question": "How did the actual implementation of the per-tenant Kafka queueing system compare to the initial hypothetical sketch?",
        # Hypothetical turns: 0-6 (8-byte tenant ID, 1 cluster, 4000 tenants cap, 7-day retention, Kafka partition per tenant, DLQ per tenant)
        # Actual turns: 40-50 (12-byte tenant ID, 3 clusters, 14-day retention, collisions found in testing, DLQ confirmed)
        # Plus 66-70 noisy-neighbor validation, 72-74 ops cost surprise, 84-88 added cross-region replication
        "source_chat_ids": [0, 2, 4, 6, 40, 42, 44, 46, 48, 50, 52, 66, 68, 72, 74, 84, 86],
        "ideal_response": "Initial hypothetical sketch (turns 0-6): Kafka pub/sub with per-tenant partitions, 8-byte tenant IDs as partition keys, ingress-service routed, per-tenant dead-letter queues, 7-day retention, ~4000 active tenants per cluster, single-cluster implied. Actual implementation: (1) Partitioning as planned. (2) Tenant ID size changed from 8 to 12 bytes because testing revealed birthday-paradox hash collisions around 200 tenants. (3) Retention extended from 7 to 14 days for operational debugging buffer. (4) Ran 3 clusters instead of 1 (regional setup), ~12,000 total tenant capacity. (5) DLQ per tenant implemented as designed. (6) Validation: noisy-neighbor incident confirmed the isolation property held when one tenant produced 10x normal volume for 45 minutes without impacting others. (7) Surprise downside: 3x ops load from 3 on-call rotations; consolidation to 2 clusters under consideration. (8) Added feature not in original sketch: cross-region replication for 5 tenants with DR needs, indirectly motivated by Zenrich's earlier multi-region asks. Core design validated; implementation corrected 3 specifics and added 1 customer-driven scope increment.",
        "benchmark": "extra_adversarial",
    },
    # === more ellipsis + hypothetical-to-actual mixed ===
    {
        "conversation_id": "xadv_hypothetical_actual",
        "category": "hypothetical_to_actual",
        "question_index": 16,
        "question": "What was the original retention policy proposed for the queueing system, and what did it become in practice?",
        "source_chat_ids": [0, 6, 44],
        "ideal_response": "Originally proposed: 7-day retention. In practice: 14-day retention. The change was driven by the infrastructure team wanting a larger debugging buffer. This is a minor but specific example of a hypothetical-to-actual drift — the design was changed during implementation without being flagged as a deviation.",
        "benchmark": "extra_adversarial",
    },
    # === Additional perspective_swap on different person  ===
    {
        "conversation_id": "xadv_launch_causal",
        "category": "perspective_swap",
        "question_index": 17,
        "question": "What did Reyna do on the Nimbus project, and how did her role evolve?",
        "source_chat_ids": [8, 10, 72, 74, 78, 80, 82],
        "ideal_response": "Reyna's role evolved from scoped polish owner to scope-expansion signaler. Initially (turn 8-10) Anton raised an onboarding flow concern in planning and Reyna volunteered to take the polish (described by the user as 'reliable'). Later (turn 74) the user confirmed Reyna owns the onboarding polish as distinct from the tutorial-spec work the user was writing. In the status meeting (turn 78-82), Reyna herself raised scope creep as the onboarding work bled into all connected surfaces: she estimated 2 additional weeks beyond current scope (vs. the originally budgeted 3 days) because the SSO work changes the signup path, which couples into the tutorial flow, which couples back into onboarding. Reyna's arc: quiet task accept -> execution -> whistle-blower on scope entanglement. Her signal was a key antecedent to the launch slip that the user ultimately had to acknowledge.",
        "benchmark": "extra_adversarial",
    },
    # === cross-topic bridging in hypothetical-actual conversation ===
    {
        "conversation_id": "xadv_hypothetical_actual",
        "category": "cross_topic_bridging",
        "question_index": 18,
        "question": "How was the Zenrich relationship connected to the design evolution of the queueing system?",
        "source_chat_ids": [16, 20, 22, 56, 84, 86, 88],
        "ideal_response": "Discussed as two separate topics (customer QBR vs. internal infrastructure design), but connected in three ways: (1) Zenrich's flagged 429 rate-limit issues came from the QBR, but the resulting ticket routing failure (tied to an unrelated account-manager resignation) caused a 4-hour outage and CEO-level escalation. (2) Zenrich's multi-region data sovereignty requirements (raised at QBR) had no roadmap support at the time. (3) After the queueing system launched, it was extended — not part of the original hypothetical design — with cross-region replication for 5 tenants with DR guarantees, which the user explicitly noted is a subset of the multi-region work Zenrich had asked about. So Zenrich's asks indirectly shaped the queueing system's post-launch evolution, and the ticket-routing process improvements (SLA alerting on account management tickets) touched operational practice more broadly. Two separate narratives, linked by customer-driven scope pressure.",
        "benchmark": "extra_adversarial",
    },
]


# ===========================================================================
# Embedding and output
# ===========================================================================

def truncate_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    texts = [truncate_text(t) for t in texts]
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start: start + BATCH_SIZE]
        print(
            f"  Embedding batch {start // BATCH_SIZE + 1} "
            f"({len(batch)} texts)...",
            flush=True,
        )
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        time.sleep(0.1)
    return np.array(all_embeddings, dtype=np.float32)


def main() -> None:
    client = OpenAI()

    print("Building extra-adversarial conversations...")
    conversations = build_conversations()
    total_turns = sum(len(turns) for turns in conversations.values())
    for conv_id, turns in conversations.items():
        print(f"  {conv_id}: {len(turns)} turns")
    print(
        f"  Total: {total_turns} turns across {len(conversations)} conversations"
    )

    print(f"\n{len(QUESTIONS)} questions across categories:")
    cats = Counter(q["category"] for q in QUESTIONS)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    # Flatten all turns
    all_segments: list[dict] = []
    for conv_id, turns in conversations.items():
        for turn in turns:
            all_segments.append({
                "conversation_id": conv_id,
                "turn_id": turn["turn_id"],
                "role": turn["role"],
                "text": turn["text"],
            })
    print(f"\n{len(all_segments)} total segments to embed")

    texts = [s["text"] for s in all_segments]
    print(f"Embedding {len(texts)} segments with {EMBED_MODEL}...")
    embeddings = embed_texts(client, texts)
    print(f"Embeddings shape: {embeddings.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / "segments_extra_adversarial.npz"
    np.savez(
        npz_path,
        embeddings=embeddings,
        conversation_ids=np.array([s["conversation_id"] for s in all_segments]),
        turn_ids=np.array(
            [s["turn_id"] for s in all_segments], dtype=np.int32
        ),
        roles=np.array([s["role"] for s in all_segments]),
        texts=np.array([s["text"] for s in all_segments]),
    )
    print(f"\nSaved segments to {npz_path}")

    json_path = OUTPUT_DIR / "questions_extra_adversarial.json"
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
    cats2 = Counter(q["category"] for q in loaded_q)
    print(f"Category breakdown: {dict(sorted(cats2.items()))}")

    # Validate source_chat_ids exist
    print("\n--- Source chat ID validation ---")
    turn_ids_by_conv: dict[str, set[int]] = {}
    for conv_id, turns in conversations.items():
        turn_ids_by_conv[conv_id] = {t["turn_id"] for t in turns}

    issues = 0
    for q in loaded_q:
        conv = q["conversation_id"]
        if conv not in turn_ids_by_conv:
            print(
                f"  WARNING: question {q['question_index']} references "
                f"unknown conversation {conv}"
            )
            issues += 1
            continue
        valid_ids = turn_ids_by_conv[conv]
        for tid in q["source_chat_ids"]:
            if tid not in valid_ids:
                print(
                    f"  WARNING: question {q['question_index']} ({q['category']}) "
                    f"references turn {tid} not in {conv} "
                    f"(max: {max(valid_ids)})"
                )
                issues += 1
    if issues == 0:
        print("  All source_chat_ids validated successfully!")
    else:
        print(f"  {issues} issues found!")


if __name__ == "__main__":
    main()
