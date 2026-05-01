# Round 6 Semantic Memory Pipeline -- Results

End-to-end pipeline with salience pre-filter, 5-turn/silence batching, 
LLM extraction into append-only logs with fuzzy topic routing, 
and background consolidation at 8 live entries.


Budget used: 83 LLM + 2 embed (~$0.25)


## Scenario accuracy

| Scenario | Topics | Entries | Extraction calls | Rubric |
|----------|--------|---------|------------------|--------|
| S1_simple_user | 7 | 22 | 9 | 8/8 (100%) |
| S2_evolving_life | 9 | 54 | 21 | 4/5 (80%) |
| S3_multi_entity | 7 | 31 | 17 | 5/6 (83%) |
| S4_novel_chunk | 12 | 24 | 1 | 9/9 (100%) |


### S1_simple_user

- ~30 turns. User drips facts, makes a correction, adds a set member.
- Ingest stats: `{'num_events': 30, 'num_batches': 9, 'num_phatic_filtered': 0, 'num_extraction_calls': 9, 'num_consolidation_calls': 0, 'num_noop_batches': 0, 'num_append_commands': 18, 'num_append_ref_commands': 4}`
- Topics: ['User/Name', 'User/Work', 'User/Cats/Luna', 'User/Allergies', 'User/Cats/Miso', 'User/Residence', 'User/Preferences/Film']

**Query**: Tell me about the user.

**Answer**: Here’s what I know about the user from memory:

- Name: Alex [User/Name].  
- Lives in Brooklyn, in Park Slope [User/Residence].  
- Works as a software engineer at Stripe, has been there 4 years, works on payments infrastructure and builds backend systems primarily in Go [User/Work].  
- Has a severe peanut allergy and is also allergic to tree nuts (reaction confirmed by an allergist) [User/Allergies].  
- Owns two cats: Luna (female tabby, 4 years old) and a new black kitten named Miso; Luna is currently hissing at the new kitten but the user expects them to adjust [User/Cats/Luna], [User/Cats/Miso].  
- Big fan of Akira Kurosawa; favorite film is Seven Samurai [User/Preferences/Film].

If you want more details or a different summary focus, tell me which area to expand.

**Rubric**: 8/8

| Point | Pass | Reason |
|-------|------|--------|
| mentions User is Alex | Y | The answer explicitly states the user's name as Alex in the first bullet point. |
| mentions Stripe / software engineer / 4 years | Y | The answer explicitly states the user works as a software engineer at Stripe and has been there 4 years. |
| mentions payments / backend / Go | Y | The answer explicitly states the user works on payments infrastructure and builds backend systems primarily in Go. |
| mentions cats Luna AND Miso (both) | Y | The answer explicitly names both cats, Luna and a new black kitten named Miso, satisfying the requirement. |
| Luna is 4 years old (corrected, not 3) | Y | The answer explicitly states Luna is 4 years old, matching the corrected age in the rubric point. |
| mentions peanut AND tree nut allergy (both) | Y | The answer explicitly states the user has a severe peanut allergy and is also allergic to tree nuts, satisfying the requirement to mention both. |
| mentions Brooklyn / Park Slope | Y | The answer explicitly states the user 'Lives in Brooklyn, in Park Slope.' |
| mentions Kurosawa / Seven Samurai interest | Y | The answer explicitly states the user is a big fan of Akira Kurosawa and that their favorite film is Seven Samurai. |


### S2_evolving_life

- ~60 turns. Life changes over a year. Query asks about evolution.
- Ingest stats: `{'num_events': 55, 'num_batches': 22, 'num_phatic_filtered': 0, 'num_extraction_calls': 21, 'num_consolidation_calls': 3, 'num_noop_batches': 4, 'num_append_commands': 39, 'num_append_ref_commands': 12}`
- Topics: ['User/Name', 'User/Partner/Riley', 'User/Plans/Wedding', 'User/Work', 'User/Location/Residence', 'User/Cars', 'User/Cars/Rivian R1S', 'User/Plans/ParentalLeave', 'User/Music']

**Query**: How has the user's life changed over the last year?

**Answer**: Here's a concise summary of major life changes for you over the last year, based only on the memories I have:

- Got married to Riley in September 2025; the wedding was small, intimate, and went well [Plans/Wedding].  
- Moved out of San Francisco and into a 3‑bedroom Craftsman in Rockridge, Oakland (moved in 2025-12-01); now commuting to SF and finding the commute brutal [Location/Residence].  
- Career change: left Google and accepted a role at Anthropic starting 2025-08-01; since Jan 2026 your role shifted from PM to product lead with increased scope [Work].  
- Family growing: Riley is pregnant, the baby is due in June 2026 and is a girl named Nova; you and Riley have started prenatal classes, and Riley has begun maternity leave (finished last class 2026-04-20) [Partner/Riley].  
- Preparing for parenthood: you’re planning three months of parental leave starting in June 2026 [Plans/ParentalLeave].  
- Transportation/home changes: sold your old car and bought a Rivian R1S to make space for the baby [Cars; Cars/Rivian R1S].  
- Small personal note: you’ve been listening to a lot of Phoebe Bridgers in April 2026 [User/Music].

If you want this reframed as a timeline or need me to highlight what to prioritize next (moving projects, parental‑leave planning, commute solutions, etc.), I can do that.

**Rubric**: 4/5

| Point | Pass | Reason |
|-------|------|--------|
| got engaged (May 2025) then married (Sept 2025) to Riley | N | The answer mentions the September 2025 marriage to Riley but does not state that they got engaged in May 2025. |
| changed jobs: Google PM -> Anthropic senior PM (Aug 2025), then promoted to prod | Y | The answer explicitly states you left Google for a role at Anthropic starting 2025-08-01 and that your role shifted from PM to product lead in Jan 2026, matching the rubric's job change and promotion  |
| moved from San Francisco Mission to Oakland Rockridge (Dec 2025) | Y | The answer explicitly states the user moved out of San Francisco into a 3‑bedroom in Rockridge, Oakland with a move date of 2025-12-01, matching the rubric point. |
| expecting a child, Nova, due June 2026 | Y | The answer explicitly states Riley is pregnant and that the baby, a girl named Nova, is due in June 2026. |
| bought a Rivian R1S | Y | The answer explicitly states the user sold their old car and bought a Rivian R1S to make space for the baby, directly matching the rubric point. |


### S3_multi_entity

- ~40 turns. User + partner + pets + employer; queries target different entities.
- Ingest stats: `{'num_events': 39, 'num_batches': 17, 'num_phatic_filtered': 0, 'num_extraction_calls': 17, 'num_consolidation_calls': 0, 'num_noop_batches': 2, 'num_append_commands': 16, 'num_append_ref_commands': 15}`
- Topics: ['User/Name', 'User/Work/Company', 'Partner/Taylor/Occupation', 'User/Pets/Dogs', 'User/Pets/Dogs/Barkley', 'User/Pets/Cats/Mochi', 'User/Background']

**Query**: Summarize the user's pets.

**Answer**: You have three pets:

- Pixel — 2-year-old Border Collie, very high-energy, gets frequent zoomies, does agility training on Saturdays and is skilled at it. [User/Pets/Dogs]
- Barkley — 5-year-old Labrador with hip dysplasia; started a joint supplement that seems to be helping. [User/Pets/Dogs/Barkley]
- Mochi — 12-year-old senior female cat who recently had a mild kidney issue and is now on a new diet. [User/Pets/Cats/Mochi]

**Rubric**: 3/3

| Point | Pass | Reason |
|-------|------|--------|
| Barkley: Labrador, 5yo, hip dysplasia | Y | The answer explicitly lists Barkley as a 5-year-old Labrador with hip dysplasia, matching the rubric point exactly. |
| Pixel: Border Collie, 2yo, high-energy / agility | Y | The answer explicitly states Pixel is a 2-year-old Border Collie, very high-energy, and participates in agility training, matching the rubric point. |
| Mochi: cat, 12yo / senior, kidney issue / special diet | Y | The answer explicitly states Mochi is a 12-year-old senior female cat with a recent mild kidney issue and now on a new diet, matching all rubric elements. |

**Query**: What does the user do for work?

**Answer**: They work at Netflix as a Staff Data Scientist on the recommendations team (recently moved from the movies-focused subteam to the series-focused subteam) [User/Work/Company; User/Background].

**Rubric**: 2/3

| Point | Pass | Reason |
|-------|------|--------|
| staff data scientist at Netflix | Y | The answer explicitly states they work at Netflix as a Staff Data Scientist, matching the rubric point exactly. |
| on recommendation team (series or movies) | Y | The answer explicitly states they work on Netflix's recommendations team and notes movement between the movies-focused and series-focused subteams, matching the rubric point. |
| 6 years tenure OR previously at Meta OR recent promotion | N | The answer states current role and a recent subteam move but does not mention 6 years tenure, prior employment at Meta, or a recent promotion. |


### S4_novel_chunk

- One long paragraph about a character. Non-dialog ingest.
- Ingest stats: `{'num_events': 1, 'num_batches': 1, 'num_phatic_filtered': 0, 'num_extraction_calls': 1, 'num_consolidation_calls': 0, 'num_noop_batches': 0, 'num_append_commands': 24, 'num_append_ref_commands': 0}`
- Topics: ['Elena/Identity', 'Elena/Location', 'Elena/Education', 'Elena/Research/Specialty', 'Elena/Career', 'Elena/Family/Spouse', 'Elena/Family/Diego', 'Elena/Health', 'Elena/Interests', 'Elena/Projects', 'Elena/Languages', 'Elena/Research/Publications']

**Query**: Tell me everything you know about Elena Vasquez.

**Answer**: Here’s what I know from your memory about Dr. Elena Vasquez:

- Identity/age/location: Dr. Elena Vasquez, age 47, based in Monterey, California. [Elena/Identity] [Elena/Location]
- Education/career: PhD from Scripps Institution of Oceanography (2008). Principal investigator at the Monterey Bay Aquarium Research Institute (MBARI) for the past twelve years, leading a team of six researchers. [Elena/Education] [Elena/Career]
- Research/specialty: Specializes in cephalopod neural systems, with a focus on cognition of the giant Pacific octopus. Her team recently published a landmark paper in Nature Neuroscience documenting novel problem-solving behaviors in octopuses; that work received significant press including an NPR feature. [Elena/Research/Specialty] [Elena/Research/Publications]
- Projects/writing: Authoring a popular-science book titled Minds in the Deep about invertebrate cognition. [Elena/Projects]
- Languages/interests: Speaks fluent Spanish and is learning Portuguese for an upcoming research collaboration in the Azores. Avid sea kayaker. [Elena/Languages] [Elena/Interests]
- Health/allergies: Has a severe shellfish allergy and carries an EpiPen at all times. [Elena/Health]
- Family: Married to Dr. Marcus Chen (a software architect at Salesforce); they met in graduate school. Children: daughter Sofia, 15, who is passionate about competitive swimming; son Diego, 11, who is a budding guitarist. Her mother is from Guadalajara. [Elena/Family/Spouse] [Elena/Family/Diego]

If you want more detail on any item (publications, MBARI projects, book timeline, or contact info), I don’t have that in memory and can try to look it up if you want.

**Rubric**: 9/9

| Point | Pass | Reason |
|-------|------|--------|
| marine biologist, Monterey, age 47 | Y | The answer explicitly states she is Dr. Elena Vasquez, age 47, based in Monterey, CA, and a principal investigator at MBARI specializing in cephalopod research, satisfying 'marine biologist, Monterey, |
| PhD from Scripps 2008, cephalopod / octopus research | Y | The answer explicitly states she has a PhD from Scripps Institution of Oceanography (2008) and that she specializes in cephalopod neural systems with a focus on octopus cognition. |
| PI at MBARI for ~12 years | Y | The answer explicitly states she has been a principal investigator at MBARI for the past twelve years, matching the rubric point. |
| married to Marcus Chen, software architect at Salesforce | Y | The answer explicitly states she is married to Dr. Marcus Chen, who is a software architect at Salesforce. |
| children Sofia (15, swimming) and Diego (11, guitar) | Y | The answer explicitly lists children Sofia (15) who is into competitive swimming and Diego (11) who is a budding guitarist, matching the rubric point. |
| shellfish allergy / EpiPen | Y | The answer explicitly states she has a severe shellfish allergy and carries an EpiPen, directly matching the rubric point. |
| sea kayaking, book 'Minds in the Deep' | Y | The answer explicitly states she is authoring a book titled 'Minds in the Deep' and that she is an avid sea kayaker, satisfying both items. |
| Spanish (Guadalajara heritage), learning Portuguese / Azores | Y | The answer explicitly states she speaks fluent Spanish, her mother is from Guadalajara, and that she is learning Portuguese for a research collaboration in the Azores, matching the rubric point. |
| recent Nature Neuroscience paper / NPR feature | Y | The answer explicitly states her team recently published a landmark paper in Nature Neuroscience and that the work received significant press including an NPR feature. |


## Rollback primitive

- success: True
- entries before: 1
- entries removed: 1
- entries after: 0

## Ablation: batching policy on S1

| Variant | LLM calls | Extraction calls | Rubric | Topics |
|---------|-----------|------------------|--------|--------|
| per_turn | 9 | 16 | 8/8 (100%) | 7 |
| per_5_turns | 9 | 9 | 8/8 (100%) | 7 |
| silence_only | 9 | 8 | 8/8 (100%) | 7 |

## Ablation: no salience pre-filter on S1

- LLM calls: 9
- Extraction calls: 9
- Rubric: 8/8
- Topics: 7