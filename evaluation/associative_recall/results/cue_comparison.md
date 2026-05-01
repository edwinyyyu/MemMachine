# Cue Comparison Across Architectures
Qualitative comparison of cues produced by different retrieval architectures on the same questions.

Pulled from cached evaluation results (no new LLM calls).

## Data coverage

- **locomo**: `v15` (30), `v2f` (30), `self_v3` (30), `human_signals_v2f` (30), `human_signals_temporal` (30), `retlog_v2f` (28), `retlog_cot` (26), `v15_hybrid_cot_v15` (30), `v15_hybrid_cot_cot` (30), `meta_v2f` (30)
- **synthetic**: `v15` (0), `v2f` (19), `self_v3` (13), `human_signals_v2f` (19), `human_signals_temporal` (19), `retlog_v2f` (18), `retlog_cot` (6), `v15_hybrid_cot_v15` (19), `v15_hybrid_cot_cot` (19)
- **puzzle**: `v15` (16), `v2f` (16), `self_v3` (13), `human_signals_v2f` (15), `human_signals_temporal` (15), `retlog_v2f` (15), `retlog_cot` (5), `v15_hybrid_cot_v15` (16), `v15_hybrid_cot_cot` (16)
- **advanced**: `v15` (23), `v2f` (23), `self_v3` (19), `human_signals_v2f` (23), `human_signals_temporal` (22), `retlog_v2f` (22), `retlog_cot` (12), `v15_hybrid_cot_v15` (23), `v15_hybrid_cot_cot` (23), `chain_hybrid_v15_term` (5), `chain_cot` (5)

## Per-question cue comparisons

### [locomo] v2f_wins_big delta_v2f-v15=+1.00 cat=locomo_multi_hop

**Question:** Would Caroline pursue writing as a career option?

_conversation_id=locomo_conv-26, qidx=27_

**v15** [recall=0.0 (baseline=0.0) | n_cues=2, avg_words=16.5]

- `Caroline writing "want to be a writer" author publish novel poetry blog "creative writing" "writing career"`
- `Caroline career option consider pursue "writing as a career" freelance writer journalism editor "writing internship" English major`

**v2f** [recall=1.0 | n_cues=2, avg_words=18.5]

- `I'm thinking about pursuing writing as a career — I love writing stories, journaling, and wonder if I could be a writer`
- `I've considered becoming an author or freelance writer, maybe blogging or publishing a novel someday`

**v15_hybrid_cot_v15** [recall=0.0 (baseline=0.0) | n_cues=1, avg_words=24.0]

- `"writing as a career" OR "becoming a writer" OR "author" OR "publish a book" OR "freelance writer" OR "consider writing" OR "career options: writing"`

**v15_hybrid_cot_cot** [recall=0.0 (baseline=0.0) | n_cues=1, avg_words=8.0]

- `journaling / poetry / creative writing / storytelling`

**retlog_cot** [recall=0.0 (baseline=0.0) | n_cues=10, avg_words=5.4]

- `writing`
- `author`
- `novelist`
- `poet / poetry`
- `Are you interested in writing professionally — e.g., freelance/content/copywriting, journalism, or trying to publish as an author?`
- `memoir`
- `personal essays / creative nonfiction`
- `"I've thought about starting a blog or writing memoir/personal essays about my trans and mental-health journey."`
- `self-publishing / indie author`
- `advocacy writing / mental-health writer`

**retlog_v2f** [recall=0.0 (baseline=0.0) | n_cues=4, avg_words=24.8]

- `I've been writing short stories and wondering if I could turn that into a career—freelance work, content writing, or publishing a novel.`
- `I'm thinking about combining counseling with writing, like blogging about mental health or writing a book to help others.`
- `I'm thinking about turning my experience into a memoir or a series of personal essays about transition and recovery, maybe self-publishing or querying an agent to get it out there.`
- `I write fiction and have wondered about professional routes like freelancing for mental-health sites, ghostwriting, applying to literary magazines or grants, or doing copy/content work on the side.`

**self_v3** [recall=0.0 (baseline=0.0) | n_cues=4, avg_words=16.5]

- `I'm thinking about pursuing writing as a career — maybe becoming a professional writer or author.`
- `I've been working on a book and seriously considering making writing my full-time job.`
- `I'm thinking about using my story to help others — maybe writing a book or becoming an author as a career option.`
- `I've considered blogging, writing memoirs, or pursuing writing professionally alongside or instead of counseling.`

**human_signals_v2f** [recall=0.0 (baseline=0.0) | n_cues=2, avg_words=29.0]

- `I've been thinking about pursuing writing as a career—considering freelance writing, journalism, or becoming a fiction author and wondering how to get started with a portfolio and publishing.`
- `I want to be a writer and am weighing options like self-publishing a novel, applying to creative writing programs, or combining writing with my counseling work as a career path.`

**human_signals_temporal** [recall=0.0 (baseline=0.0) | n_cues=2, avg_words=24.0]

- `Caroline, you mentioned working on writing projects like a book, stories, or a blog and weighing whether to pursue writing professionally as an author or freelance writer.`
- `You discussed career choices—comparing counseling to creative paths—specifically considering becoming a writer, publishing a book, doing freelance/content writing, or blogging full-time.`

**meta_v2f** [recall=0.0 (baseline=0.0) | n_cues=2, avg_words=29.0]

- `I've been thinking about pursuing writing as a career—considering freelance writing, journalism, or becoming a fiction author and wondering how to get started with a portfolio and publishing.`
- `I want to be a writer and am weighing options like self-publishing a novel, applying to creative writing programs, or combining writing with my counseling work as a career path.`

---

### [locomo] v2f_wins_big delta_v2f-v15=+1.00 cat=locomo_temporal

**Question:** When did Caroline have a picnic?

_conversation_id=locomo_conv-26, qidx=21_

**v15** [recall=0.0 (baseline=0.0) | n_cues=2, avg_words=17.0]

- `"Caroline, we had a picnic last weekend in the park — blanket, sandwiches, the kids loved it."`
- `"Hey Caroline! We went for a picnic yesterday in the park with the kids on Sunday afternoon."`

**v2f** [recall=1.0 | n_cues=2, avg_words=16.5]

- `I had a picnic last Sunday at the park with the kids — took sandwiches and a blanket`
- `Went on a picnic Saturday afternoon by the lake last weekend, it was gorgeous out`

**v15_hybrid_cot_v15** [recall=0.0 (baseline=0.0) | n_cues=1, avg_words=20.0]

- `Caroline picnic "had a picnic" "yesterday" "last week" "last weekend" "on Sunday" "on Saturday" "at the park" "with the kids"`

**v15_hybrid_cot_cot** [recall=0.0 (baseline=0.0) | n_cues=1, avg_words=1.0]

- `picnicked`

**retlog_cot** [recall=1.0 (baseline=0.0) | n_cues=10, avg_words=6.6]

- `had a picnic`
- `picnicked`
- `we had a picnic last weekend`
- `picnic basket`
- `Caroline's picnic`
- `"We went for an outdoor lunch in the park last week"`
- `"It was the other day — we ate outside on Sunday"`
- `"A week ago we had an outing to the park"`
- `"We ate al fresco over the weekend"`
- `"We had a little get-together in the park recently (about a week ago)"`

**retlog_v2f** [recall=0.0 (baseline=0.0) | n_cues=4, avg_words=17.5]

- `Caroline had a picnic last weekend at the park on Saturday afternoon with the kids.`
- `I remember Caroline saying she picnicked on Sunday morning two weeks ago down by the lake.`
- `We had a cozy backyard picnic yesterday evening with a few friends and the kids — it felt so relaxing.`
- `Caroline mentioned organizing a school picnic last month on Friday, May 12, with snacks and games in the schoolyard.`

**self_v3** [recall=0.0 (baseline=0.0) | n_cues=4, avg_words=8.5]

- `Caroline had a picnic last weekend`
- `We had a picnic yesterday at the park with Caroline`
- `Caroline had a picnic last Sunday at the park.`
- `I went on a picnic with Caroline yesterday afternoon.`

**human_signals_v2f** [recall=1.0 (baseline=0.0) | n_cues=2, avg_words=8.0]

- `we had a picnic yesterday at the park`
- `had a picnic last weekend on Sunday afternoon`

**human_signals_temporal** [recall=0.0 (baseline=0.0) | n_cues=2, avg_words=20.0]

- `Hey Caroline, we had a picnic at the park yesterday — spread a blanket, packed sandwiches and stayed all afternoon.`
- `Caroline, we went for a picnic last weekend (Saturday afternoon) with the kids; it was sunny and everyone loved it.`

**meta_v2f** [recall=1.0 (baseline=0.0) | n_cues=2, avg_words=8.0]

- `we had a picnic yesterday at the park`
- `had a picnic last weekend on Sunday afternoon`

---

### [synthetic] proactive recall=0.5

**Question:** I want to cook dinner for Bob tonight. What should I keep in mind?

_conversation_id=synth_personal, qidx=13_

**v2f** [recall=0.5 (baseline=0.5) | n_cues=2, avg_words=34.0]

- `Bob has peanut allergy and shellfish allergy, plus lactose intolerance (aged cheeses and butter are OK); avoid cream-based sauces for Bob but parmesan sprinkled on top might be fine, and salmon is okay since it’s not shellfish.`
- `Looking for make-ahead side dish ideas and practical cooking notes that accommodate Bob’s allergies (no peanuts, no shellfish), gluten-free and vegan options, no mushrooms, no beef, and guidance on avoiding cross-contamination.`

**v15_hybrid_cot_v15** [recall=0.3333333333333333 (baseline=0.5) | n_cues=1, avg_words=28.0]

- `Bob peanut allergy shellfish allergy no shrimp/crab lactose intolerance aged cheeses and butter OK no milk/cream/soft cheeses keto preference flexible salmon ok make-ahead side dish suggestions dinner tonight`

**v15_hybrid_cot_cot** [recall=0.3333333333333333 (baseline=0.5) | n_cues=1, avg_words=31.0]

- `When cooking for someone with a severe peanut allergy tonight, what cross-contamination precautions, cleaning/utensil steps, and label-checking should I follow — plus quick low-dairy, low-carb dinner swaps to keep in mind?`

**retlog_v2f** [recall=0.5 (baseline=0.5) | n_cues=4, avg_words=38.2]

- `Steps to prevent peanut and shellfish cross-contamination in the kitchen and at the serving table — cleaning, separate utensils/boards, labeling, and plating strategies for Bob.`
- `Make-ahead, keto-friendly, lactose-free side dish ideas and reheating/storage instructions (cauliflower mash, roasted Brussels sprouts, garlicky sautéed greens) that work tonight and respect Bob's aged-cheese/butter allowance.`
- `Practical kitchen workflow and serving plan to protect Bob (peanut allergy; lactose intolerance — aged cheese and butter OK) when cooking for mixed-diet guests (vegan Sarah, tree-nut allergy Emma, gluten-free Rachel): separate prep stations, order of cooking, dedicated pans/utensils, dish-labeling at buffet, plating sequence, seating/serving order, handling leftovers, and what to tell guests about severity/anaphylaxis precautions.`
- `Nut-free, dairy-free creamy and low-carb recipe substitutions and safe vegan options (avoid cashew/almond creams): coconut cream and silken tofu sauces, avocado or mayonnaise-based dressings, egg/olive-oil emulsions, where to use butter/parm only on specific plates, and nut-free dessert alternatives to sorbet that suit vegan, gluten-free, and lactose-intolerant guests.`

**self_v3** [recall=0.6666666666666666 (baseline=0.5) | n_cues=10, avg_words=16.8]

- `Bob — peanut allergy, shellfish allergy (no shrimp/crab), lactose intolerant but aged cheeses and butter are OK, and keto is just a preference.`
- `Avoid peanuts and peanut oil entirely; prevent cross‑contamination (separate utensils, no shared frying oil).`
- `No milk, cream, or soft cheeses for Bob — cream‑based sauces are out, but parmesan or butter can be used.`
- `Treat keto as flexible — prioritize avoiding sugars/carbs if convenient, but don’t stress perfection.`
- `Check labels and processed ingredients (broths, sauces, dressings) for hidden peanuts, shellfish, or milk.`
- `Bob: peanut allergy and lactose intolerance — he said aged cheeses and butter are fine, but no milk, cream, or soft cheeses.`
- `Bob's been doing keto for the last month and is trying to keep carbs low.`
- `Blanket no-nut policy for the dinner — avoid peanuts, tree nuts, and nut-derived oils (no almond, cashew, walnut, peanut).`
- `Be careful about cross-contamination with peanuts/nuts — shared utensils, oils, and traces could be dangerous.`
- `Is this dinner just for Bob tonight or the Saturday group dinner?`

**human_signals_v2f** [recall=0.6667 (baseline=0.5) | n_cues=2, avg_words=25.0]

- `Bob peanut allergy shellfish allergy lactose intolerant — no milk/cream/soft cheeses; aged cheeses and butter OK; keto preference flexible; avoid peanut/shellfish cross-contamination; salmon fillets previously planned.`
- `Safe dinner ideas and make-ahead sides for peanut- and shellfish-allergic, lactose-intolerant (but OK with butter/parmesan) guests; keto-friendly or adaptable recipes; cross-contact prevention and labeling.`

**human_signals_temporal** [recall=0.5 (baseline=0.5) | n_cues=2, avg_words=29.0]

- `Bob: peanut allergy; shellfish allergy (no shrimp or crab); lactose intolerance — no milk, cream, or soft cheeses; aged cheeses and butter OK; keto preference but flexible`
- `Cooking for Bob tonight — avoid peanuts and shellfish, avoid dairy ingredients like milk/cream/soft cheeses, consider keto-friendly swaps and make-ahead side dishes that are safe for his allergies and lactose intolerance`

---

### [synthetic] proactive recall=0.23076923076923078

**Question:** Draft a project status update for the Acme Corp rebrand team covering current progress and next steps.

_conversation_id=synth_work, qidx=14_

**v2f** [recall=0.23076923076923078 (baseline=0.3076923076923077) | n_cues=2, avg_words=27.5]

- `Status of logo concepts — three options from Vanessa — latest feedback, approvals, and chosen direction`
- `Hiroshi brief on content strategy and brand voice guide (tone refresh); Phase 1 (Jan 5–Jan 31) brand identity status and Phase 2 (Feb 1–Feb 21) website design/dev progress; name of the new Acme CEO who wants the startup-y feel`

**v15_hybrid_cot_v15** [recall=0.38461538461538464 (baseline=0.3076923076923077) | n_cues=1, avg_words=43.0]

- `"Acme rebrand status update progress logo concepts Vanessa brand voice guide Pantone 2945 C secondary palette Hiroshi content strategy brief Jan 15 Phase 1 Jan 5–Jan 31 Phase 2 Feb 1–Feb 21 Phase 3 Feb 22–Mar 8 deliverables owners approvals next steps timeline"`

**v15_hybrid_cot_cot** [recall=0.38461538461538464 (baseline=0.3076923076923077) | n_cues=1, avg_words=31.0]

- `Can you confirm the status and file locations of Vanessa’s three logo concepts (finalized?), Hiroshi’s final tone guide, who the Acme decision‑maker/CEO is, and whether the mid‑February presentation date is locked?`

**retlog_v2f** [recall=0.23076923076923078 (baseline=0.3076923076923077) | n_cues=4, avg_words=35.0]

- `Provide current status for each deliverable with owner and next action: logo concepts (Vanessa), brand guidelines, website design, social templates, and content migration — include approval status, due dates, file locations (Drive/Dropbox), and any blockers.`
- `Provide the Acme CEO's name and any executive/stakeholder feedback about the desired "startup-y" direction, plus the detailed content strategy brief for Hiroshi: tone changes, content owners, content calendar start and completion targets, and the content migration plan.`
- `Add a concise project timeline with key milestones and dates (kickoff, Jan 15 moodboard review, logo concept presentation, brand guidelines draft, website launch target), plus owners and next actions for each milestone.`
- `Summarize the scope-change request from CMO Patricia to add an animated hero hero section: proposed options (accept with budget/time adjustment, deliver as phase 2, or decline), estimated impact on timeline/resources, and recommended communication for the client.`

**self_v3** [recall=0.15384615384615385 (baseline=0.3076923076923077) | n_cues=10, avg_words=22.0]

- `Status update on Phase 1 deliverables — logo concepts from Vanessa, typography selection (Acme Sans), brand voice guide completion and approvals`
- `Decision on secondary palette — confirmation/approval of cool gray + teal to pair with Pantone 2945 C`
- `Has Hiroshi been briefed on the content strategy and tone refresh? current status and next actions for content deliverables`
- `Website development readiness and handoff notes for Phase 2 (Feb 1–Feb 21) — design acceptance, dev tickets, outstanding items`
- `Any CEO feedback or directives about the “startup-y” tone and final sign-offs or blockers impacting timeline`
- `Project status: logo concepts (3) are ready from Vanessa (evolution, new, hybrid); client hasn’t seen them; they’re planned to be presented with the palette next Wednesday at 2pm.`
- `Content brief for Hiroshi: transition tone from “Acme Corporation delivers enterprise‑grade solutions” to an approachable, startup‑y voice; need sample homepage hero and about page copy.`
- `Palette update: keep Pantone 2945 C deep blue as primary; propose swapping warm gray + burnt orange for cool gray + teal accent; client color review scheduled.`
- `Development status check: Tom (frontend) and Maria (backend) wireframe/prototype progress, integration dependencies, and estimated delivery dates for the website.`
- `Team coordination / next steps: confirm who will present at the client meeting, finalize deliverables to show (logo comps, palette, sample copy), and list any blockers needing client decisions.`

**human_signals_v2f** [recall=0.2308 (baseline=0.3077) | n_cues=2, avg_words=36.5]

- `Acme rebrand status: contract signed last week; Phase 1 (Jan 5–Jan 31) brand identity work underway — logo, colors (primary Pantone 2945 C), typography, brand voice guide; three logo concepts from Vanessa.`
- `Next steps and actions: brief Hiroshi on the content strategy and tone refresh; finalize secondary palette (cool gray + teal); Phase 2 website design Feb 1–Feb 21 and Phase 3 content Feb 22–Mar 8; confirm CEO name and deliverable completion statuses.`

**human_signals_temporal** [recall=0.3846 (baseline=0.3077) | n_cues=2, avg_words=39.0]

- `Current progress summary for Acme rebrand including completed deliverables to date, logo concepts from Vanessa, brand book review (primary Pantone 2945 C), any client feedback, and who on our team is assigned (Vanessa, Hiroshi, etc.).`
- `Next steps and timeline details for Acme rebrand: Phase 1 (Jan 5–Jan 31) deliverables and outstanding items, Phase 2 (Feb 1–Feb 21) website design/development status, Phase 3 (Feb 22–Mar 8) content milestones, upcoming deadlines, and notes about briefing Hiroshi on the tone refresh.`

---

### [puzzle] logic_constraint v2f-v15=+0.00

**Question:** Based on all constraints discussed, what is the final valid desk arrangement for the 6 desks?

_conversation_id=puzzle_logic_1, qidx=0_

**v15** [recall=0.25 (baseline=0.3333333333333333) | n_cues=2, avg_words=26.5]

- `"final valid desk arrangement desks 1-6 you in desk 1 Bob in desk 6 Alice and Dave must have at least one desk between them Henderson project constraint"`
- `"final seating order 'desk 1' 'desk 2' 'desk 3' 'desk 4' 'desk 5' 'desk 6' Alice Dave separated by one desk Carol Eve Bob assignment"`

**v2f** [recall=0.25 (baseline=0.3333333333333333) | n_cues=2, avg_words=31.0]

- `Final valid desk arrangement for the six desks listing who sits at desks 1–6, respecting "you in desk 1", "Bob desk 6", and "Alice and Dave need at least one desk between them"`
- `Henderson project argument that "Alice and Dave cannot be adjacent" plus any lines assigning Alice, Carol, Dave, Eve to specific desk numbers (desks numbered 1 leftmost to 6 rightmost)`

**v15_hybrid_cot_v15** [recall=0.3333333333333333 (baseline=0.3333333333333333) | n_cues=1, avg_words=29.0]

- `"final arrangement" OR "valid desk arrangement" OR "Alice and Dave" "at least one desk between" OR "Bob desk 6" OR "you're in desk 1" OR "Carol" OR "Eve" "desk"`

**v15_hybrid_cot_cot** [recall=0.3333333333333333 (baseline=0.3333333333333333) | n_cues=1, avg_words=8.0]

- `"Final arrangement: 1-You, 2-Alice, 3-Eve, 4-Dave, 5-Carol, 6-Bob"`

**retlog_v2f** [recall=0.25 (baseline=0.3333333333333333) | n_cues=2, avg_words=26.5]

- `Final desk arrangement: Desk 1 - You; Desk 2 - Alice; Desk 3 - Carol; Desk 4 - Dave; Desk 5 - Eve; Desk 6 - Bob`
- `Placing Carol and Eve into the remaining desks (2–5) so Alice and Dave have at least one person between them and Bob stays at desk 6`

**self_v3** [recall=0.25 (baseline=0.3333333333333333) | n_cues=5, avg_words=21.0]

- `Final valid desk arrangement with you in desk 1, Bob in desk 6, and Alice and Dave separated by at least one desk.`
- `"Alice and Dave must have at least one desk between them" — seating that satisfies this.`
- `"You take desk 1 by the whiteboard; Bob at desk 6 for best light and IT" — remaining assignment for Carol, Eve, Alice, Dave.`
- `"Final seating chart — list desks 1 through 6 with names (resolve Carol and Eve placements to satisfy non-adjacency of Alice/Dave)."`
- `Henderson project constraint caused Alice and Dave to be non-adjacent; find the resolved arrangement that fixes the earlier 2-and-3 adjacency problem.`

**human_signals_v2f** [recall=0.25 (baseline=0.3333) | n_cues=2, avg_words=31.5]

- `Final valid desk arrangement for the 6 desks given you in desk 1, Bob in desk 6, and Alice and Dave must have at least one desk between them — assign Alice, Bob, Carol, Dave, and Eve to desks 2–6.`
- `Confirmed assignments or proposals mentioning Alice and Dave not adjacent (Henderson project constraint), desks 1 leftmost/6 rightmost, and placements for Carol and Eve.`

**human_signals_temporal** [recall=0.25 (baseline=0.3333) | n_cues=2, avg_words=33.5]

- `Final valid desk arrangement for desks 1–6: you in desk 1, Bob in desk 6, with explicit assignments for Alice, Dave, Carol, and Eve ensuring Alice and Dave have at least one desk between them.`
- `Alice and Dave need at least one desk between them — adjacent placement violates the Henderson project argument — provide a valid assignment of desks 2–5 for Carol, Eve, Alice, and Dave.`

---

### [puzzle] logic_constraint v2f-v15=+0.00

**Question:** What were all the constraints for the desk arrangement, including which ones were eventually resolved or became irrelevant?

_conversation_id=puzzle_logic_1, qidx=1_

**v15** [recall=0.09090909090909091 (baseline=0.18181818181818182) | n_cues=2, avg_words=28.0]

- `"6 desks in a row" "6 people" Alice Dave "need at least one desk between them" Henderson project adjacency`
- `"desk 1 makes the most sense for you with the whiteboard" "All the desks are by the windows" "Bob moves to a different desk" Eve Carol "which of Alice or Dave takes desk 2 vs 3" resolved`

**v2f** [recall=0.09090909090909091 (baseline=0.18181818181818182) | n_cues=2, avg_words=31.5]

- `Summarize all desk-arrangement constraints mentioned (Alice and Dave not adjacent; desk 1 preferred for the whiteboard; all desks by the windows; Bob may move) and explicitly state which of those were later resolved, changed, or declared irrelevant.`
- `Any messages that list final or confirmed desk assignments, note "resolved" or "irrelevant", or mention Eve/Carol/Dave preferences (including references to the Henderson project or Bob moving)`

**v15_hybrid_cot_v15** [recall=0.09090909090909091 (baseline=0.18181818181818182) | n_cues=1, avg_words=26.0]

- `"constraints list" "Alice and Dave" "at least one desk between" "can't be adjacent" Bob move desk 1 whiteboard Eve preference Dave preference Henderson project resolved irrelevant`

**v15_hybrid_cot_cot** [recall=0.09090909090909091 (baseline=0.18181818181818182) | n_cues=1, avg_words=48.0]

- `Can you list every constraint for the desk arrangement (e.g., Henderson rule that Alice and Dave can't be adjacent, user wants desk 1 by the whiteboard, Eve prefers a middle desk, Carol next to Eve/desk 5, Bob's flexibility) and indicate which constraints were later resolved or became irrelevant?`

**retlog_v2f** [recall=0.09090909090909091 (baseline=0.18181818181818182) | n_cues=4, avg_words=34.5]

- `List any other seating constraints mentioned for Bob, Carol, Eve, Frank, or the manager — including window or whiteboard preferences, need for a power outlet or proximity to the printer, requests to sit together or apart, and any exact desk numbers referenced.`
- `Indicate which constraints were later resolved, abandoned, or called irrelevant (look for phrases like "no longer applies", "dropped", "resolved", "decided to ignore") and note any turn that describes a final decision or reassignment because of those changes.`
- `Include any seating constraints or preferences specifically for Dave or Alice (explicit desk numbers, adjacency or separation rules, any "changed his/her mind" or "no longer" language), and mark which of those constraints were later resolved.`
- `Include any mentions of Frank or the manager and any window-specific constraints, plus any constraints described as "abandoned," "no longer an issue," or "irrelevant."`

**self_v3** [recall=0.09090909090909091 (baseline=0.18181818181818182) | n_cues=5, avg_words=21.6]

- `Alice and Dave need at least one desk between them — they cannot be adjacent (Henderson project constraint).`
- `If Alice and Dave are in desks 2 and 3 respectively (or vice versa), they'd be adjacent and that violates the constraint.`
- `All the desks are by the windows though — mention any window-related preference or if that was dismissed.`
- `Desk 1 makes the most sense for you with the whiteboard — note who required the whiteboard and whether that was fixed.`
- `We might need to rearrange; if Bob moves to a different desk it opens up more possibilities — search for whether Bob actually moved or constraints were changed.`

**human_signals_v2f** [recall=0.0909 (baseline=0.1818) | n_cues=2, avg_words=11.0]

- `Alice and Dave need at least one desk between them`
- `final desk arrangement assigned desks rearrange Bob moved opens up more possibilities`

**human_signals_temporal** [recall=0.0909 (baseline=0.1818) | n_cues=2, avg_words=34.0]

- `6 desks in a row, 6 people; Alice and Dave need at least one desk between them (can't be adjacent); desk 1 makes the most sense for you because of the whiteboard.`
- `All the desks are by the windows; Henderson project adjacency violation if Alice and Dave are in desks 2 and 3; Bob moving opens up possibilities — look for mentions of rearrange, resolved, or became irrelevant.`

---

### [advanced] evolving_terminology recall=0.4666666666666667

**Question:** What is the current status of Project Phoenix? Include any milestones reached and upcoming work.

_conversation_id=adv_evolving_term_1, qidx=0_

**v15** [recall=0.26666666666666666 (baseline=0.3333333333333333) | n_cues=2, avg_words=21.5]

- `Project Phoenix OAuth2 integration corporate identity provider blocker resolution PHX-123 blocked Priya standup update`
- `Phase 1 dashboard 4x faster complete; phase 2 reporting module on track week 11; scope expanded notification center phase 3; subscription management deferred post-launch; project charter update steering committee`

**v2f** [recall=0.4666666666666667 (baseline=0.3333333333333333) | n_cues=2, avg_words=34.5]

- `Phase 1 complete — dashboard loads 4x faster; Phase 2 reporting module is on track for week 11; scope expanded to include a notification center in Phase 3; subscription management deferred to post‑launch.`
- `We hit our first real blocker: the OAuth2 integration with the corporate identity provider is more complicated than we thought; Priya said we need to update the project charter and raise this at the steering committee.`

**v15_hybrid_cot_v15** [recall=0.4666666666666667 (baseline=0.3333333333333333) | n_cues=1, avg_words=35.0]

- `"Project Phoenix status update milestones reached phase 1 dashboard 4x faster phase 2 reporting week 11 OAuth2 integration blocker resolved? notification center scope expanded subscription management deferred post-launch update project charter steering committee PHX standup"`

**v15_hybrid_cot_cot** [recall=0.4666666666666667 (baseline=0.3333333333333333) | n_cues=1, avg_words=51.0]

- `Any updates on Phoenix/PHX — is the OAuth2 integration blocker resolved, was the phase‑1 demo/dashboard milestone officially signed off, is phase‑2 still on track for week 11, has the project charter been updated for the scope expansion, and what are the next planned milestones (notification center, phase‑3 self‑service, steering committee items)?`

**chain_cot** [recall=0.26666666666666666 (baseline=0.3333333333333333) | n_cues=10, avg_words=15.4]

- `Project Phoenix status update`
- `PHX standup update — blockers, milestones, PHX-123`
- `OAuth2 integration with the corporate identity provider is blocking PHX — impact and next steps`
- `Phase 1 complete (dashboard 4x faster); phase 2 reporting module week 11; phase 3 notification center scope expansion`
- `Project charter update / steering committee discussion — scope expanded and subscription management deferral`
- `PHX sprint 1 status: which stories were completed, which remain in progress, and are there any outstanding blockers?`
- `Project Phoenix milestones reached — any deployments to staging, demos completed, or performance targets met for the new dashboard/auth?`
- `Update from Marcus on the SAML-to-OAuth bridge: has the identity-provider issue been resolved or is there a temporary workaround in place?`
- `Priya architecture doc status — has the microservices (Go), React, Postgres architecture been approved and has implementation started?`
- `Phoenix roadmap / upcoming work: confirmed kickoff date and scope for Phase 2 (reporting) and priorities for the next sprint.`

**chain_hybrid_v15_term** [recall=0.3333333333333333 (baseline=0.3333333333333333) | n_cues=2, avg_words=23.5]

- `"first real blocker" "OAuth2 integration" "corporate identity provider" "PHX-123 is blocked" "any update"`
- `"phase 1 is complete" "dashboard loading 4x faster" "phase 2 reporting module on track for week 11" "scope expanded" "notification center phase 3" "defer subscription management to post-launch" "update the project charter" "steering committee"`

**retlog_v2f** [recall=0.26666666666666666 (baseline=0.3333333333333333) | n_cues=4, avg_words=28.2]

- `Sam to pair with the corporate IdP team this week to unblock the OAuth2 integration; temporary token-proxy workaround and a staging deploy are scheduled as mitigation.`
- `Updated schedule: internal beta targeted week 14, release candidate week 18, go/no-go at the next steering committee; migration runbook, canary deploy, and full QA regression are listed as upcoming milestones.`
- `Sprint 1 is complete — OAuth2 auth service merged and deployed to staging; Jenny finished the test framework and QA started smoke and end-to-end tests.`
- `Sprint 2 kickoff Monday: API gateway, user profile pages, and analytics integration are prioritized; demo for stakeholders planned end of week 10 and we still need a production runbook and deployment automation.`

**self_v3** [recall=0.4666666666666667 (baseline=0.3333333333333333) | n_cues=10, avg_words=19.6]

- `OAuth2 integration with the corporate identity provider is blocking PHX; need status, owner, and mitigation plan.`
- `Phase 1 complete (dashboard loads 4x faster); phase 2 reporting module — confirm on-track for week 11 and list open tickets (e.g., PHX-123) and owners.`
- `Scope expanded to include a notification center in phase 3 — has the project charter been updated and is there a timeline/spec for the notification center?`
- `Subscription management decision: deferred to post-launch — confirm and provide backlog ticket or planned schedule.`
- `Steering committee update for Project Phoenix — any approvals, risks, or upcoming milestones we should present?`
- `Architecture doc drafted for Phoenix — Priya proposing microservices instead of the monolith.`
- `We had our first sprint planning for Phoenix today; Priya ran it and the team started work.`
- `Hit our first real blocker: OAuth2 integration with corporate identity provider using old SAML; Marcus is working with their team and saml2-bridge (AuthForge).`
- `Phase 1 is new auth flow and dashboard; Phase 2 reporting module; Phase 3 self-service features — upcoming work and phase 1 timeline.`
- `PHX standup / PHX-123 is blocked — sprint 1 delay could cascade to the phase 1 deadline; need to raise with stakeholders.`

**human_signals_v2f** [recall=0.5333 (baseline=0.3333) | n_cues=2, avg_words=24.5]

- `phase 1 complete — dashboard loading 4x faster; phase 2 reporting module on track for week 11; timeline, milestones, release dates, status update`
- `OAuth2 integration with corporate identity provider blocker; update project charter scope expanded to include notification center (phase 3); defer subscription management to post-launch; steering committee notes`

**human_signals_temporal** [recall=0.4667 (baseline=0.3333) | n_cues=2, avg_words=30.0]

- `Phase 1 complete — dashboard loads 4x faster; phase 2 reporting module on track for week 11; scope expanded to include a notification center; subscription management deferred to post-launch`
- `PHX OAuth2 integration with the corporate identity provider is a blocker; Priya said we need to update the project charter after the scope expansion; bring Phoenix status to the steering committee`

---

### [advanced] evolving_terminology recall=0.8571428571428571

**Question:** What are all the different names that have been used to refer to v2 throughout the conversation?

_conversation_id=adv_evolving_term_1, qidx=1_

**v15** [recall=1.0 (baseline=0.8571428571428571) | n_cues=2, avg_words=15.5]

- `"Project Phoenix", "Phoenix", "PHX", "Portal 2.0", "customer portal redesign", "customer portal"`
- `"v2", "is that a v1 bug or a v2 thing?", "the new portal", "the portal project", "the bird", "the portal"`

**v2f** [recall=0.8571428571428571 (baseline=0.8571428571428571) | n_cues=2, avg_words=25.5]

- `We should call it Project Phoenix — people also refer to Phoenix, PHX, and use the phoenix emoji when naming it`
- `The team calls it v2 or Portal 2.0; execs say the new portal or the portal project; Ravi calls it the bird; also mention customer portal redesign and v1/legacy portal`

**v15_hybrid_cot_v15** [recall=0.8571428571428571 (baseline=0.8571428571428571) | n_cues=1, avg_words=21.0]

- `Project Phoenix PHX Phoenix "Portal 2.0" "Portal v2" v2 "the new portal" "portal project" "the portal" "customer portal redesign" "the bird"`

**v15_hybrid_cot_cot** [recall=0.8571428571428571 (baseline=0.8571428571428571) | n_cues=1, avg_words=1.0]

- `PHX-`

**chain_cot** [recall=0.8571428571428571 (baseline=0.8571428571428571) | n_cues=5, avg_words=4.6]

- `Project Phoenix`
- `PHX`
- `Portal 2.0`
- `"the bird" because of the phoenix emoji`
- `"the VP called it 'the new portal' or 'the portal project'"`

**chain_hybrid_v15_term** [recall=0.8571428571428571 (baseline=0.8571428571428571) | n_cues=2, avg_words=12.0]

- `"call it Project Phoenix" PHX "phoenix emoji" "the bird"`
- `"v2" "Portal 2.0" "the new portal" "portal project" "customer portal redesign" "vendor portal" "Portal V2"`

**retlog_cot** [recall=0.7142857142857143 (baseline=0.8571428571428571) | n_cues=5, avg_words=1.6]

- `Portal v2`
- `P2`
- `Portal2`
- `NextGen portal`
- `portal replatform`

**retlog_v2f** [recall=0.7142857142857143 (baseline=0.8571428571428571) | n_cues=4, avg_words=2.0]

- `Phoenix Project`
- `portal relaunch`
- `the new portal`
- `PHX`

**self_v3** [recall=0.8571428571428571 (baseline=0.8571428571428571) | n_cues=10, avg_words=2.4]

- `Project Phoenix`
- `PHX`
- `Portal 2.0`
- `the bird`
- `the new portal`
- `Project Phoenix`
- `PHX`
- `PHX-123`
- `the new portal`
- `v2 OR "version 2" OR "portal v2"`

**human_signals_v2f** [recall=0.8571 (baseline=0.8571) | n_cues=2, avg_words=10.5]

- `Project Phoenix PHX Phoenix "the bird"`
- `v2 "Portal 2.0" "Portal 2" "the new portal" "the portal project" customer portal vendor portal`

**human_signals_temporal** [recall=0.8571 (baseline=0.8571) | n_cues=2, avg_words=18.5]

- `We're calling it Project Phoenix — people also say Phoenix, PHX, use the phoenix emoji, or just "the bird"`
- `Team shorthand includes v2, Portal 2.0, portal v2, "the new portal", "the portal project", and customer portal redesign`

---

### [synthetic] completeness recall=0.3076923076923077

**Question:** List ALL dietary restrictions and food preferences for every guest at the Saturday dinner party, including any updates or corrections.

_conversation_id=synth_personal, qidx=6_

**v2f** [recall=0.3076923076923077 (baseline=0.23076923076923078) | n_cues=2, avg_words=38.5]

- `Final restriction summary: Bob — peanut allergy, lactose intolerant (aged cheese/butter OK), keto preference; Emma — tree nut allergy; Sarah — vegan and soy allergy; Priya & Dev Patel — no beef; Linda — no mushrooms; Rachel — gluten‑free; James — no known restrictions.`
- `Reminder/decision: blanket no‑nut policy for the dinner (no peanuts, tree nuts, or nut‑derived oils) and any updates or corrections to individual restrictions should be listed (e.g., Sarah soy allergy added by user text).`

**v15_hybrid_cot_v15** [recall=0.23076923076923078 (baseline=0.23076923076923078) | n_cues=1, avg_words=39.0]

- `"final restriction summary Bob peanut allergy lactose intolerant aged cheese/butter OK keto preference; Emma tree nut allergy; Sarah vegan soy allergy; Priya & Dev Patel no beef; Linda no mushrooms; Rachel celiac/gluten-free; James no known restrictions; blanket no-nut policy"`

**v15_hybrid_cot_cot** [recall=0.23076923076923078 (baseline=0.23076923076923078) | n_cues=1, avg_words=34.0]

- `Please confirm the final, corrected dietary restrictions for Saturday dinner guests — Bob, Emma, Sarah, Priya, Dev, Linda, Rachel, and James — explicitly noting Bob’s shellfish allergy removal and Sarah’s newly reported soy allergy.`

**retlog_v2f** [recall=0.15384615384615385 (baseline=0.23076923076923078) | n_cues=4, avg_words=37.8]

- `Please provide the complete, final dietary restrictions and food preferences for every guest attending Saturday's dinner party, including any updates or corrections and noting which items are confirmed versus tentative`
- `Confirm Rachel's gluten/celiac status and list any last-minute changes or corrections for Emma, James, Linda, Sarah, Bob, and the Patels before finalizing the menu`
- `Provide a final, per-guest breakdown of dietary restrictions and food preferences for the Saturday dinner: list each guest by name, mark each item as "allergy" vs "preference," include severity and cross-contamination precautions (EpiPen, strict gluten-free handling for celiac), and note any allowed exceptions (e.g., aged cheese/butter OK).`
- `Confirm the complete guest list and any last-minute updates or corrections for Linda, James, Emma, Priya, Dev, Rachel, Sarah, Bob, and any other attendees; explicitly include Sarah's soy allergy, the blanket no-nut policy details (peanuts, tree nuts, nut-derived oils), and whether Linda's "no mushrooms" or James's "no restrictions" status changed.`

**human_signals_v2f** [recall=0.3077 (baseline=0.2308) | n_cues=2, avg_words=38.0]

- `Saturday's dinner party final restriction summary: Bob - peanut allergy, lactose intolerant (aged cheese/butter OK), keto preference; Emma - tree nut allergy; Sarah - vegan and soy allergy; Priya & Dev Patel - no beef; Linda - no mushrooms; Rachel - celiac/gluten-free; James - no known restrictions.`
- `Update for Saturday dinner: blanket no-nut policy nothing with peanuts tree nuts or nut-derived oils, and Sarah texted she also has a soy allergy so adjust vegan options accordingly.`

**human_signals_temporal** [recall=0.3077 (baseline=0.2308) | n_cues=2, avg_words=36.0]

- `Final restriction summary: Bob - peanut allergy, lactose intolerant (aged cheese/butter OK), keto pref; Emma (Bob's daughter) - tree nut allergy; Sarah - vegan + soy allergy; Priya & Dev Patel - no beef; Linda - no mushrooms; Rachel - celiac/gluten‑free; James - no known restrictions`
- `Sarah mentioned she also has a soy allergy; agreed to a blanket no‑nut policy for the dinner — nothing with peanuts, tree nuts, or nut‑derived oils.`

---

### [synthetic] completeness recall=0.6666666666666666

**Question:** What are all of the user's current medications, including dosages and what they're for? Include any recent changes.

_conversation_id=synth_medical, qidx=7_

**v2f** [recall=0.6666666666666666 (baseline=0.8333333333333334) | n_cues=2, avg_words=21.0]

- `metformin 500mg twice daily; significant increase from 500mg to 1000mg twice daily; new dosage start date`
- `Updated medication list: Metformin 1000mg 2x daily, Lisinopril 10mg daily, Atorvastatin 20mg nightly, Baby aspirin 81mg daily, Vitamin D 2000 IU daily; ibuprofen amount and frequency`

**v15_hybrid_cot_v15** [recall=0.6666666666666666 (baseline=0.8333333333333334) | n_cues=1, avg_words=36.0]

- `"Metformin 1000mg 2x daily" "increase from 500mg to 1000mg twice daily" "When does the new dosage start?" "Lisinopril 10mg daily" "Atorvastatin 20mg nightly" "Baby aspirin 81mg daily" "Vitamin D 2000 IU daily" "prescribed by Dr. Patel"`

**v15_hybrid_cot_cot** [recall=0.6666666666666666 (baseline=0.8333333333333334) | n_cues=1, avg_words=48.0]

- `Please list every medication and supplement you currently take (prescription and OTC), giving the name, dose, what it's for, when you started each or any recent changes — specifically include when the metformin dose increase takes effect and the dose/frequency for ibuprofen, aspirin, atorvastatin, lisinopril, and any vitamins.`

**retlog_v2f** [recall=0.5 (baseline=0.8333333333333334) | n_cues=4, avg_words=9.0]

- `I take ibuprofen as needed for pain`
- `My metformin increase to 1000mg twice daily starts`
- `I take atorvastatin 20 mg every night for high cholesterol.`
- `I recently had my metformin increased to 1000 mg twice daily.`

**self_v3** [recall=0.8333333333333334 (baseline=0.8333333333333334) | n_cues=5, avg_words=17.6]

- `My current medications are: Metformin 1000mg twice daily, Lisinopril 10mg daily, Atorvastatin 20mg nightly, Baby aspirin 81mg daily, Vitamin D 2000 IU daily.`
- `I was on metformin 500mg twice daily but my doctor increased it to 1000mg twice daily; the new metformin dose starts [start date].`
- `Ibuprofen (Advil/Motrin) — I take ibuprofen [dosage] as needed, [frequency] (user's ibuprofen dose/frequency reply).`
- `Lisinopril 10mg daily, prescribed by Dr. Patel about six months ago (prescriber/history line).`
- `Alternate names: Glucophage/metformin XR, Prinivil/Zestril (lisinopril), Lipitor (atorvastatin), ASA 81mg (baby aspirin), cholecalciferol (vitamin D).`

**human_signals_v2f** [recall=1.0 (baseline=0.8333) | n_cues=2, avg_words=20.0]

- `I take metformin 500mg twice daily for type 2 diabetes.`
- `Updated medication list: Metformin 1000mg 2x daily, Lisinopril 10mg daily (prescribed by Dr. Patel about 6 months ago), Atorvastatin 20mg nightly, Baby aspirin 81mg daily, Vitamin D 2000 IU daily.`

**human_signals_temporal** [recall=0.8333 (baseline=0.8333) | n_cues=2, avg_words=30.0]

- `Metformin 1000mg twice daily; metformin increased from 500mg to 1000mg 2x daily; Lisinopril 10mg daily (Dr. Patel ~6 months ago); Atorvastatin 20mg nightly; Baby aspirin 81mg daily; Vitamin D 2000 IU daily — updated medication list`
- `Ibuprofen amount and frequency mentioned; any other OTC meds; start dates/prescribers and reasons for baby aspirin and vitamin D; recent medication changes or additions`

---

## Aggregate cue statistics (across all loaded questions per arch)

| Architecture | # cues | avg words | avg chars | %question | %boolean | %meta |
|---|---|---|---|---|---|---|
| v15 | 138 | 16.6 | 108.3 | 5.8% | 6.5% | 0.0% |
| v2f | 176 | 22.2 | 135.3 | 0.0% | 0.6% | 0.0% |
| v15_hybrid_cot_v15 | 88 | 27.2 | 182.7 | 5.7% | 12.5% | 1.1% |
| v15_hybrid_cot_cot | 88 | 22.6 | 150.4 | 34.1% | 0.0% | 0.0% |
| chain_cot | 40 | 11.8 | 79.8 | 22.5% | 0.0% | 0.0% |
| chain_hybrid_v15_term | 10 | 17.3 | 128.9 | 0.0% | 10.0% | 0.0% |
| retlog_cot | 355 | 11.1 | 74.1 | 23.1% | 0.3% | 0.0% |
| retlog_v2f | 294 | 25.3 | 164.0 | 2.7% | 0.3% | 0.3% |
| self_v3 | 444 | 16.2 | 99.4 | 3.4% | 0.2% | 0.5% |
| human_signals_v2f | 174 | 22.5 | 139.0 | 0.6% | 0.6% | 0.6% |
| human_signals_temporal | 172 | 22.0 | 135.8 | 0.0% | 0.6% | 1.2% |
| meta_v2f | 60 | 14.1 | 78.8 | 1.7% | 0.0% | 0.0% |

## Per-dataset aggregate statistics

### locomo

| Architecture | # cues | avg words | %question | %boolean | %meta |
|---|---|---|---|---|---|
| v15 | 60 | 11.4 | 6.7% | 8.3% | 0.0% |
| v2f | 60 | 14.8 | 0.0% | 0.0% | 0.0% |
| v15_hybrid_cot_v15 | 30 | 19.1 | 0.0% | 30.0% | 0.0% |
| v15_hybrid_cot_cot | 30 | 8.4 | 13.3% | 0.0% | 0.0% |
| retlog_cot | 210 | 6.8 | 6.7% | 0.5% | 0.0% |
| retlog_v2f | 102 | 19.5 | 2.0% | 0.0% | 0.0% |
| self_v3 | 129 | 13.0 | 0.0% | 0.0% | 0.8% |
| human_signals_v2f | 60 | 14.1 | 1.7% | 0.0% | 0.0% |
| human_signals_temporal | 60 | 12.8 | 0.0% | 0.0% | 0.0% |
| meta_v2f | 60 | 14.1 | 1.7% | 0.0% | 0.0% |

### synthetic

| Architecture | # cues | avg words | %question | %boolean | %meta |
|---|---|---|---|---|---|
| v2f | 38 | 28.4 | 0.0% | 0.0% | 0.0% |
| v15_hybrid_cot_v15 | 19 | 35.5 | 10.5% | 0.0% | 0.0% |
| v15_hybrid_cot_cot | 19 | 35.5 | 42.1% | 0.0% | 0.0% |
| retlog_cot | 40 | 16.1 | 10.0% | 0.0% | 0.0% |
| retlog_v2f | 66 | 34.2 | 0.0% | 0.0% | 0.0% |
| self_v3 | 84 | 17.6 | 6.0% | 0.0% | 0.0% |
| human_signals_v2f | 38 | 32.2 | 0.0% | 0.0% | 2.6% |
| human_signals_temporal | 38 | 28.0 | 0.0% | 0.0% | 0.0% |

### puzzle

| Architecture | # cues | avg words | %question | %boolean | %meta |
|---|---|---|---|---|---|
| v15 | 32 | 21.3 | 9.4% | 0.0% | 0.0% |
| v2f | 32 | 27.8 | 0.0% | 0.0% | 0.0% |
| v15_hybrid_cot_v15 | 16 | 28.9 | 0.0% | 12.5% | 0.0% |
| v15_hybrid_cot_cot | 16 | 26.8 | 50.0% | 0.0% | 0.0% |
| retlog_cot | 30 | 18.6 | 63.3% | 0.0% | 0.0% |
| retlog_v2f | 48 | 27.6 | 8.3% | 0.0% | 2.1% |
| self_v3 | 96 | 15.6 | 3.1% | 0.0% | 1.0% |
| human_signals_v2f | 30 | 25.7 | 0.0% | 0.0% | 0.0% |
| human_signals_temporal | 30 | 28.5 | 0.0% | 0.0% | 3.3% |

### advanced

| Architecture | # cues | avg words | %question | %boolean | %meta |
|---|---|---|---|---|---|
| v15 | 46 | 20.0 | 2.2% | 8.7% | 0.0% |
| v2f | 46 | 22.9 | 0.0% | 2.2% | 0.0% |
| v15_hybrid_cot_v15 | 23 | 29.5 | 13.0% | 0.0% | 4.3% |
| v15_hybrid_cot_cot | 23 | 27.5 | 43.5% | 0.0% | 0.0% |
| chain_cot | 40 | 11.8 | 22.5% | 0.0% | 0.0% |
| chain_hybrid_v15_term | 10 | 17.3 | 0.0% | 10.0% | 0.0% |
| retlog_cot | 75 | 17.3 | 60.0% | 0.0% | 0.0% |
| retlog_v2f | 78 | 23.9 | 2.6% | 1.3% | 0.0% |
| self_v3 | 135 | 18.8 | 5.2% | 0.7% | 0.0% |
| human_signals_v2f | 46 | 23.3 | 0.0% | 2.2% | 0.0% |
| human_signals_temporal | 44 | 24.7 | 0.0% | 2.3% | 2.3% |

## Key insights

### 1. v2f vs v15 on LoCoMo: paraphrase vs keyword-bundle

v15 produces 60 total cues averaging 11.4 words each; v2f produces 60 cues averaging 14.8 words. Both generate the same number of cues (2/q on average). The key qualitative differences observed verbatim in the samples above:

- **v15 cues look like keyword bundles** with quoted phrases and boolean-like vocabulary (e.g., `Caroline writing "want to be a writer" author publish novel poetry blog`). 8.3% of v15 locomo cues contain OR/AND operators vs 0.0% for v2f.
- **v2f cues are paraphrased first-person statements** mimicking how a memory turn would actually be written (e.g., `I'm thinking about pursuing writing as a career — I love writing stories, journaling`). This better matches the underlying embedding targets (user utterances), which explains the v2f_wins_big deltas.
- v15's boolean phrasing underperforms on multi-hop/temporal locomo because embeddings don't model operator semantics; v2f's dense first-person paraphrase lands closer in embedding space to actual recorded user lines.

### 2. self_v3 (SIMPLE-classified) vs v2f: quantity over quality

Out of 30 self_v3 questions on locomo: 0 had cues identical to v2f, 30 had different cues. self_v3 generated **4.3 cues per question on average**, vs v2f's 2.0. Classification routing changes both **how many** cues and **what they look like** — even for the 'simpler' branch. The SIMPLE template appears to encourage more short near-paraphrases of the same utterance (e.g., 4 variants of 'I went to the LGBTQ support group yesterday') rather than v2f's 2 diverse full paraphrases. This is cue-diversity bleeding: classification changes prompt framing even when the classification label is SIMPLE.

**Q:** When did Caroline go to the LGBTQ support group?

- self_v3 (4 cues): ['I went to a LGBTQ support group yesterday and it was so powerful.', 'went to a support group yesterday', 'I went to a LGBTQ support group yesterday and it was so powerful.', 'Last Friday, I went to an LGBTQ+ counseling workshop and it was really enlightening.']
- v2f (2 cues): ['I went to a LGBTQ support group yesterday and it was so powerful.', 'I attended an LGBTQ support group yesterday']

**Q:** When did Melanie paint a sunrise?

- self_v3 (4 cues): ['I painted that lake sunrise last year!', 'Melanie painted the lake sunrise last year', 'I painted that lake sunrise last year!', 'I painted that sunrise last year.']
- v2f (2 cues): ["I painted that lake sunrise last year! It's special to me.", 'Melanie painted that lake sunrise last year']

**Q:** What fields would Caroline be likely to pursue in her educaton?

- self_v3 (7 cues): ["I'm studying counseling and want to become a therapist", "I'm majoring in psychology", "I'm thinking about social work or getting an MSW", "I'm interested in art therapy or combining pottery with counseling", 'I take pottery/ceramics classes or workshops', "I'm keen on counseling or working in mental health - I'd love to support those with similar issues.", "I'm thinking of working with trans people, helping them accept themselves and supporting their mental health; I went to an LGBTQ+ counseling workshop."]
- v2f (2 cues): ["I'm gonna continue my edu and check out career options — thinking about counseling or social work to help kids in need and support adoptive families", "Motivated by my own experience, I'm looking into counseling, child welfare, or family services so I can work with kids and adoptive parents"]


### 3. CoT cue style by question type

CoT cues (cot_cues column in v15_hybrid_hybrid_v15_cot_*) vary dramatically by question category. Question categories where CoT produces **short, answer-paraphrase** cues (~5–8 words, few question marks) tend to be where CoT helps; categories where CoT produces **long, literal question-shaped** cues (~30–50 words, ≥50% question marks) are where it underperforms.

**puzzle** CoT cue averages by category:

- state_change: n=3 avg_words=15.0 pct_Q=0.0% pct_bool=0.0%
- contradiction: n=2 avg_words=22.0 pct_Q=0.0% pct_bool=0.0%
- absence_inference: n=3 avg_words=28.3 pct_Q=33.3% pct_bool=0.0%
- open_exploration: n=2 avg_words=30.5 pct_Q=100.0% pct_bool=0.0%
- sequential_chain: n=3 avg_words=32.0 pct_Q=100.0% pct_bool=0.0%
- logic_constraint: n=3 avg_words=32.7 pct_Q=66.7% pct_bool=0.0%

**advanced** CoT cue averages by category:

- unfinished_business: n=3 avg_words=17.7 pct_Q=66.7% pct_bool=0.0%
- consistency_checking: n=2 avg_words=23.0 pct_Q=50.0% pct_bool=0.0%
- evolving_terminology: n=5 avg_words=23.4 pct_Q=20.0% pct_bool=0.0%
- constraint_propagation: n=2 avg_words=24.0 pct_Q=50.0% pct_bool=0.0%
- perspective_separation: n=4 avg_words=28.5 pct_Q=50.0% pct_bool=0.0%
- quantitative_aggregation: n=3 avg_words=30.7 pct_Q=66.7% pct_bool=0.0%
- negation: n=3 avg_words=40.7 pct_Q=33.3% pct_bool=0.0%
- frequency_detection: n=1 avg_words=41.0 pct_Q=0.0% pct_bool=0.0%

**locomo** CoT cue averages by category:

- locomo_temporal: n=16 avg_words=7.1 pct_Q=6.2% pct_bool=0.0%
- locomo_single_hop: n=10 avg_words=8.2 pct_Q=10.0% pct_bool=0.0%
- locomo_multi_hop: n=4 avg_words=14.2 pct_Q=50.0% pct_bool=0.0%

**synthetic** CoT cue averages by category:

- inference: n=3 avg_words=13.3 pct_Q=0.0% pct_bool=0.0%
- control: n=3 avg_words=20.0 pct_Q=66.7% pct_bool=0.0%
- completeness: n=4 avg_words=35.0 pct_Q=50.0% pct_bool=0.0%
- proactive: n=4 avg_words=46.8 pct_Q=50.0% pct_bool=0.0%
- conjunction: n=3 avg_words=47.7 pct_Q=66.7% pct_bool=0.0%
- procedural: n=2 avg_words=52.0 pct_Q=0.0% pct_bool=0.0%

On LoCoMo, CoT single_hop cues (8.2 words) and temporal cues (7.1 words) are short keyword-like strings. On puzzle/advanced, CoT often writes literal meta-questions (e.g., 'Can you list every constraint...?'), which don't match any stored user utterance semantically — producing worse recall than plain v2f paraphrases.

### 4. Human-signals temporal metadata: surface length effect

The human_signals_temporal variant injects temporal/position metadata into the prompt. Explicit 'turn 12', 'chunk', 'segment' meta-content leaks rarely (<4% in any dataset), but the variant does subtly change cue style — per-dataset comparison vs plain v2f:

- **locomo**: temporal_signals pct_meta=0.0%, avg_words=12.8 vs v2f pct_meta=0.0%, avg_words=14.8
- **synthetic**: temporal_signals pct_meta=0.0%, avg_words=28.0 vs v2f pct_meta=0.0%, avg_words=28.4
- **puzzle**: temporal_signals pct_meta=3.3%, avg_words=28.5 vs v2f pct_meta=0.0%, avg_words=27.8
- **advanced**: temporal_signals pct_meta=2.3%, avg_words=24.7 vs v2f pct_meta=0.0%, avg_words=22.9

Observed in the question samples above: temporal_signals sometimes adds second-person framing ("Caroline, you mentioned...") which shifts the embedding away from recorded first-person utterances. The effect is small but consistent with the measured lower recall on locomo.

### 5. What distinguishes effective cues from ineffective ones

From the 10 case studies above, cross-referencing cue text against per-question recall:

- **Effective** (high recall): first-person utterance-shaped cues that echo how the target was originally written. e.g. on picnic question, retlog_cot hit recall=1.0 with short naturalistic phrases (`had a picnic`, `picnicked`) while retlog_v2f scored 0.0 with fully elaborated fabrications (`Caroline had a picnic last weekend at the park on Saturday...`) — the hallucinated specifics pushed the embedding away from real stored lines.
- **Ineffective** (zero recall): 
   - Boolean-joined quoted phrases (`"writing as a career" OR "becoming a writer" OR ...`) — embedding models don't respect OR;
   - Literal interrogative queries (`Can you list every constraint for the desk arrangement...?`) — no user ever wrote a question like that;
   - Over-specific fabrications (`Saturday afternoon by the lake`, `school picnic last month on Friday, May 12`) — the invented details out-vote the one true anchor phrase;
   - Second-person addressing (`Caroline, you mentioned writing projects...`) on data stored as first-person.

- **v15's signature failure**: boolean-style keyword bundles work for BM25 but not for dense retrieval — v15 loses locomo_multi_hop/temporal exactly where the gold lives in a single semantically-rich first-person utterance.
- **v2f's signature failure**: when the answer is a *structural artifact* (list of constraints, final desk map, all medications) rather than a single utterance, v2f's 2 fluent paraphrases can't cover the structural space. retlog_v2f (4 cues) and self_v3 (5–10 cues) help marginally by covering more angles.
- **CoT's signature failure**: on categories where the correct move is to *widen* the surface area (proactive, completeness, conjunction), CoT instead writes one long interrogative meta-question, which matches nothing.

