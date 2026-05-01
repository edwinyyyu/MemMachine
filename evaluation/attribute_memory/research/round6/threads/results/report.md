# Round 6 -- Topic Routing Results

## Leaderboard (aggregate across 6 scenarios)

| Strategy | Consistency | Entity-match | Topics/Fact | Balance CV | LLM calls | Embed calls | Cost |
|----------|-------------|--------------|-------------|------------|-----------|-------------|------|
| R5_hybrid | 100.00% (4/4) | 78.10% (25/32) | 0.53 | 0.27 | 0 | 0 | $0.0000 |
| R7_entity_plus_embed | 100.00% (4/4) | 96.90% (31/32) | 0.47 | 0.29 | 32 | 0 | $0.0800 |
| R1_fixed_taxonomy | 75.00% (3/4) | 53.10% (17/32) | 0.53 | 0.23 | 0 | 0 | $0.0000 |
| R3_entity_first | 75.00% (3/4) | 93.80% (30/32) | 0.62 | 0.12 | 0 | 0 | $0.0000 |
| R2_llm_proposed | 50.00% (2/4) | 90.60% (29/32) | 0.59 | 0.25 | 0 | 0 | $0.0000 |
| R4_embedding_only | 50.00% (2/4) | 68.80% (22/32) | 0.56 | 0.39 | 0 | 0 | $0.0000 |
| R6_cheap_cascade | 50.00% (2/4) | 50.00% (16/32) | 0.59 | 0.33 | 0 | 0 | $0.0000 |

## Per-scenario consistency (multi-fact equivalence groups)

| Scenario | R1_fixed_taxonomy | R2_llm_proposed | R3_entity_first | R4_embedding_only | R5_hybrid | R6_cheap_cascade | R7_entity_plus_embed |
|---|---|---|---|---|---|---|---|
| S1_multi_entity_household | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| S2_paraphrase_consistency | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 |
| S3_medical_evolution | 1/1 | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 |
| S4_ambiguous_referent | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| S5_boundary_cases | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| S6_novel_domain | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 | 1/1 |

## Per-scenario entity-match

| Scenario | R1_fixed_taxonomy | R2_llm_proposed | R3_entity_first | R4_embedding_only | R5_hybrid | R6_cheap_cascade | R7_entity_plus_embed |
|---|---|---|---|---|---|---|---|
| S1_multi_entity_household | 2/9 | 9/9 | 9/9 | 8/9 | 7/9 | 8/9 | 9/9 |
| S2_paraphrase_consistency | 5/5 | 5/5 | 5/5 | 3/5 | 5/5 | 1/5 | 5/5 |
| S3_medical_evolution | 4/4 | 4/4 | 4/4 | 3/4 | 4/4 | 0/4 | 4/4 |
| S4_ambiguous_referent | 1/5 | 4/5 | 5/5 | 4/5 | 5/5 | 2/5 | 5/5 |
| S5_boundary_cases | 1/5 | 4/5 | 5/5 | 4/5 | 3/5 | 5/5 | 4/5 |
| S6_novel_domain | 4/4 | 3/4 | 2/4 | 0/4 | 1/4 | 0/4 | 4/4 |

## Per-scenario topic count

| Scenario | R1_fixed_taxonomy | R2_llm_proposed | R3_entity_first | R4_embedding_only | R5_hybrid | R6_cheap_cascade | R7_entity_plus_embed |
|---|---|---|---|---|---|---|---|
| S1_multi_entity_household | 3 | 4 | 4 | 5 | 5 | 6 | 4 |
| S2_paraphrase_consistency | 2 | 2 | 2 | 2 | 1 | 2 | 2 |
| S3_medical_evolution | 1 | 1 | 1 | 2 | 1 | 1 | 1 |
| S4_ambiguous_referent | 4 | 4 | 5 | 3 | 4 | 3 | 3 |
| S5_boundary_cases | 3 | 5 | 4 | 3 | 4 | 3 | 3 |
| S6_novel_domain | 4 | 3 | 4 | 3 | 2 | 4 | 2 |

## Routing traces (per strategy)

### R1_fixed_taxonomy

**S1_multi_entity_household**
  - t1.f1: `User/Possessions` (NEW) <- 'User adopted calico cat Luna'   [r1_fixed]
  - t1.f2: `User/Possessions` <- 'Luna is about 2 years old'   [r1_fixed]
  - t2.f3: `User/Relationships` (NEW) <- 'User has a partner named Jamie'   [r1_fixed]
  - t2.f4: `User/Employment` (NEW) <- 'Jamie is a graphic designer'   [r1_fixed]
  - t3.f5: `User/Employment` <- 'User works at Anthropic for ~3 years'   [r1_fixed]
  - t3.f6: `User/Employment` <- "User's role is safety research"   [r1_fixed]
  - t4.f7: `User/Possessions` <- 'User has a dog named Rex, Golden Retriever'   [r1_fixed]
  - t4.f8: `User/Possessions` <- 'Rex is about 11 years old, senior'   [r1_fixed]
  - t6.f9: `User/Possessions` <- 'Luna has a troublemaker/playful personality'   [r1_fixed]
  topics: User/Possessions(5), User/Relationships(1), User/Employment(3)

**S2_paraphrase_consistency**
  - t1.f1: `User/Medical` (NEW) <- 'User has type 2 diabetes'   [r1_fixed]
  - t2.f2: `User/Employment` (NEW) <- "User is a nurse at St. Mary's"   [r1_fixed]
  - t3.f3: `User/Medical` <- 'User manages blood sugar daily due to diabetes'   [r1_fixed]
  - t4.f4: `User/Medical` <- 'User is both a nurse and diabetic (occupation+condition restated)'   [r1_fixed]
  - t5.f5: `User/Employment` <- 'User reassigned to pediatric ward (job detail)'   [r1_fixed]
  topics: User/Medical(3), User/Employment(2)

**S3_medical_evolution**
  - t1.f1: `User/Medical` (NEW) <- 'User has lower back pain for ~1 month'   [r1_fixed]
  - t2.f2: `User/Medical` <- "User's back pain is from herniated disc at L4-L5"   [r1_fixed]
  - t3.f3: `User/Medical` <- 'User prescribed PT 2x/week and naproxen'   [r1_fixed]
  - t4.f4: `User/Medical` <- "User's back pain improved ~60% after 2 weeks of PT"   [r1_fixed]
  topics: User/Medical(4)

**S4_ambiguous_referent**
  - t1.f1: `User/Medical` (NEW) <- "Marco (user's friend) has severe shellfish allergy"   [r1_fixed]
  - t2.f2: `User/Medical` <- 'User has no food allergies'   [r1_fixed]
  - t3.f3: `User/Events` (NEW) <- "User's daughter Ellie started violin lessons"   [r1_fixed]
  - t3.f4: `User/Biography` (NEW) <- 'Ellie is 8 years old'   [r1_fixed]
  - t4.f5: `User/Employment` (NEW) <- 'Marco is switching jobs, leaving Google for a startup'   [r1_fixed]
  topics: User/Medical(2), User/Events(1), User/Biography(1), User/Employment(1)

**S5_boundary_cases**
  - t1.f1: `User/Possessions` (NEW) <- "User's son Theo got a hamster named Peanut for birthday"   [r1_fixed]
  - t1.f2: `User/Relationships` (NEW) <- 'User has son named Theo'   [r1_fixed]
  - t2.f3: `User/Skills` (NEW) <- 'Theo plays soccer on U10 team'   [r1_fixed]
  - t4.f4: `User/Skills` <- 'User took up woodworking hobby, built bookshelf'   [r1_fixed]
  - t5.f5: `User/Possessions` <- 'Peanut (hamster) escapes her cage, playful/adventurous'   [r1_fixed]
  topics: User/Possessions(2), User/Relationships(1), User/Skills(2)

**S6_novel_domain**
  - t1.f1: `User/Employment` (NEW) <- 'User owns/runs coffee roastery called Magpie Roasters in Portland'   [r1_fixed]
  - t2.f2: `User/Relationships` (NEW) <- "Magpie's biggest wholesale client is Neon Cafes (14 PNW stores)"   [r1_fixed]
  - t3.f3: `User/Events` (NEW) <- "User's business launching Rio Azul single-origin from Colombia"   [r1_fixed]
  - t4.f4: `User/Skills` (NEW) <- 'User training for half-marathon in October'   [r1_fixed]
  topics: User/Employment(1), User/Relationships(1), User/Events(1), User/Skills(1)

### R2_llm_proposed

**S1_multi_entity_household**
  - t1.f1: `Luna/Profile` (NEW) <- 'User adopted calico cat Luna'   [r2_new_topic]
  - t1.f2: `Luna/Profile` <- 'Luna is about 2 years old'   [r2_reuse_llm_pick (emb_best=Luna/Profile@0.59)]
  - t2.f3: `Jamie/Profile` (NEW) <- 'User has a partner named Jamie'   [r2_new_topic]
  - t2.f4: `Jamie/Profile` <- 'Jamie is a graphic designer'   [r2_reuse_llm_pick (emb_best=Jamie/Profile@0.55)]
  - t3.f5: `User/Employment` (NEW) <- 'User works at Anthropic for ~3 years'   [r2_new_topic]
  - t3.f6: `User/Employment` <- "User's role is safety research"   [r2_reuse_llm_pick (emb_best=User/Employment@0.33)]
  - t4.f7: `Rex/Profile` (NEW) <- 'User has a dog named Rex, Golden Retriever'   [r2_new_topic]
  - t4.f8: `Rex/Profile` <- 'Rex is about 11 years old, senior'   [r2_reuse_llm_pick (emb_best=Rex/Profile@0.63)]
  - t6.f9: `Luna/Profile` <- 'Luna has a troublemaker/playful personality'   [r2_reuse_llm_pick (emb_best=Luna/Profile@0.67)]
  topics: Luna/Profile(3), Jamie/Profile(2), User/Employment(2), Rex/Profile(2)

**S2_paraphrase_consistency**
  - t1.f1: `User/Medical` (NEW) <- 'User has type 2 diabetes'   [r2_new_topic]
  - t2.f2: `User/Medical` <- "User is a nurse at St. Mary's"   [r2_reuse_llm_pick (emb_best=User/Medical@0.39)]
  - t3.f3: `User/Medical` <- 'User manages blood sugar daily due to diabetes'   [r2_reuse_llm_pick (emb_best=User/Medical@0.59)]
  - t4.f4: `User/Medical` <- 'User is both a nurse and diabetic (occupation+condition restated)'   [r2_reuse_llm_pick (emb_best=User/Medical@0.72)]
  - t5.f5: `User/Employment` (NEW) <- 'User reassigned to pediatric ward (job detail)'   [r2_new_topic]
  topics: User/Medical(4), User/Employment(1)

**S3_medical_evolution**
  - t1.f1: `User/Medical` (NEW) <- 'User has lower back pain for ~1 month'   [r2_new_topic]
  - t2.f2: `User/Medical` <- "User's back pain is from herniated disc at L4-L5"   [r2_reuse_llm_pick (emb_best=User/Medical@0.62)]
  - t3.f3: `User/Medical` <- 'User prescribed PT 2x/week and naproxen'   [r2_reuse_llm_pick (emb_best=User/Medical@0.39)]
  - t4.f4: `User/Medical` <- "User's back pain improved ~60% after 2 weeks of PT"   [r2_reuse_llm_pick (emb_best=User/Medical@0.72)]
  topics: User/Medical(4)

**S4_ambiguous_referent**
  - t1.f1: `Marco/Allergies` (NEW) <- "Marco (user's friend) has severe shellfish allergy"   [r2_new_topic]
  - t2.f2: `Marco/Allergies` <- 'User has no food allergies'   [r2_reuse_llm_pick (emb_best=Marco/Allergies@0.52)]
  - t3.f3: `Ellie/Music` (NEW) <- "User's daughter Ellie started violin lessons"   [r2_new_topic]
  - t3.f4: `Ellie/Age` (NEW) <- 'Ellie is 8 years old'   [r2_new_topic]
  - t4.f5: `Marco/Employment` (NEW) <- 'Marco is switching jobs, leaving Google for a startup'   [r2_new_topic]
  topics: Marco/Allergies(2), Ellie/Music(1), Ellie/Age(1), Marco/Employment(1)

**S5_boundary_cases**
  - t1.f1: `Theo/Pets` (NEW) <- "User's son Theo got a hamster named Peanut for birthday"   [r2_new_topic]
  - t1.f2: `Theo/Family` (NEW) <- 'User has son named Theo'   [r2_new_topic]
  - t2.f3: `Theo/Sports` (NEW) <- 'Theo plays soccer on U10 team'   [r2_new_topic]
  - t4.f4: `User/Woodworking` (NEW) <- 'User took up woodworking hobby, built bookshelf'   [r2_new_topic]
  - t5.f5: `Peanut/Pets` (NEW) <- 'Peanut (hamster) escapes her cage, playful/adventurous'   [r2_new_topic]
  topics: Theo/Pets(1), Theo/Family(1), Theo/Sports(1), User/Woodworking(1), Peanut/Pets(1)

**S6_novel_domain**
  - t1.f1: `User/Business` (NEW) <- 'User owns/runs coffee roastery called Magpie Roasters in Portland'   [r2_new_topic]
  - t2.f2: `Magpie/Wholesale` (NEW) <- "Magpie's biggest wholesale client is Neon Cafes (14 PNW stores)"   [r2_new_topic]
  - t3.f3: `User/Business` <- "User's business launching Rio Azul single-origin from Colombia"   [r2_reuse_llm_pick (emb_best=User/Business@0.47)]
  - t4.f4: `User/Running` (NEW) <- 'User training for half-marathon in October'   [r2_new_topic]
  topics: User/Business(2), Magpie/Wholesale(1), User/Running(1)

### R3_entity_first

**S1_multi_entity_household**
  - t1.f1: `Luna/Profile` (NEW) <- 'User adopted calico cat Luna'   [r3_entity_first]
  - t1.f2: `Luna/Profile` <- 'Luna is about 2 years old'   [r3_entity_first]
  - t2.f3: `Jamie/Profile` (NEW) <- 'User has a partner named Jamie'   [r3_entity_first]
  - t2.f4: `Jamie/Profile` <- 'Jamie is a graphic designer'   [r3_entity_first]
  - t3.f5: `User/Employment` (NEW) <- 'User works at Anthropic for ~3 years'   [r3_entity_first]
  - t3.f6: `User/Employment` <- "User's role is safety research"   [r3_entity_first]
  - t4.f7: `Rex/Profile` (NEW) <- 'User has a dog named Rex, Golden Retriever'   [r3_entity_first]
  - t4.f8: `Rex/Profile` <- 'Rex is about 11 years old, senior'   [r3_entity_first]
  - t6.f9: `Luna/Profile` <- 'Luna has a troublemaker/playful personality'   [r3_entity_first]
  topics: Luna/Profile(3), Jamie/Profile(2), User/Employment(2), Rex/Profile(2)

**S2_paraphrase_consistency**
  - t1.f1: `User/Medical` (NEW) <- 'User has type 2 diabetes'   [r3_entity_first]
  - t2.f2: `User/Employment` (NEW) <- "User is a nurse at St. Mary's"   [r3_entity_first]
  - t3.f3: `User/Medical` <- 'User manages blood sugar daily due to diabetes'   [r3_entity_first]
  - t4.f4: `User/Medical` <- 'User is both a nurse and diabetic (occupation+condition restated)'   [r3_entity_first]
  - t5.f5: `User/Employment` <- 'User reassigned to pediatric ward (job detail)'   [r3_entity_first]
  topics: User/Medical(3), User/Employment(2)

**S3_medical_evolution**
  - t1.f1: `User/Medical` (NEW) <- 'User has lower back pain for ~1 month'   [r3_entity_first]
  - t2.f2: `User/Medical` <- "User's back pain is from herniated disc at L4-L5"   [r3_entity_first]
  - t3.f3: `User/Medical` <- 'User prescribed PT 2x/week and naproxen'   [r3_entity_first]
  - t4.f4: `User/Medical` <- "User's back pain improved ~60% after 2 weeks of PT"   [r3_entity_first]
  topics: User/Medical(4)

**S4_ambiguous_referent**
  - t1.f1: `Marco/Medical` (NEW) <- "Marco (user's friend) has severe shellfish allergy"   [r3_entity_first]
  - t2.f2: `User/Medical` (NEW) <- 'User has no food allergies'   [r3_entity_first]
  - t3.f3: `Ellie/Education` (NEW) <- "User's daughter Ellie started violin lessons"   [r3_entity_first]
  - t3.f4: `Ellie/Profile` (NEW) <- 'Ellie is 8 years old'   [r3_entity_first]
  - t4.f5: `Marco/Employment` (NEW) <- 'Marco is switching jobs, leaving Google for a startup'   [r3_entity_first]
  topics: Marco/Medical(1), User/Medical(1), Ellie/Education(1), Ellie/Profile(1), Marco/Employment(1)

**S5_boundary_cases**
  - t1.f1: `Peanut/Profile` (NEW) <- "User's son Theo got a hamster named Peanut for birthday"   [r3_entity_first]
  - t1.f2: `Theo/Profile` (NEW) <- 'User has son named Theo'   [r3_entity_first]
  - t2.f3: `Theo/Sports` (NEW) <- 'Theo plays soccer on U10 team'   [r3_entity_first]
  - t4.f4: `User/Hobby` (NEW) <- 'User took up woodworking hobby, built bookshelf'   [r3_entity_first]
  - t5.f5: `Peanut/Profile` <- 'Peanut (hamster) escapes her cage, playful/adventurous'   [r3_entity_first]
  topics: Peanut/Profile(2), Theo/Profile(1), Theo/Sports(1), User/Hobby(1)

**S6_novel_domain**
  - t1.f1: `Magpie Roasters/Business` (NEW) <- 'User owns/runs coffee roastery called Magpie Roasters in Portland'   [r3_entity_first]
  - t2.f2: `Neon Cafes/Business` (NEW) <- "Magpie's biggest wholesale client is Neon Cafes (14 PNW stores)"   [r3_entity_first]
  - t3.f3: `User's business/Business` (NEW) <- "User's business launching Rio Azul single-origin from Colombia"   [r3_entity_first]
  - t4.f4: `User/Training` (NEW) <- 'User training for half-marathon in October'   [r3_entity_first]
  topics: Magpie Roasters/Business(1), Neon Cafes/Business(1), User's business/Business(1), User/Training(1)

### R4_embedding_only

**S1_multi_entity_household**
  - t1.f1: `Luna/Profile` (NEW) <- 'User adopted calico cat Luna'   [r4_new_heuristic]
  - t1.f2: `Luna/Profile` <- 'Luna is about 2 years old'   [r4_embed_reuse@0.59]
  - t2.f3: `Jamie/Other` (NEW) <- 'User has a partner named Jamie'   [r4_new_heuristic]
  - t2.f4: `Jamie/Other` <- 'Jamie is a graphic designer'   [r4_embed_reuse@0.55]
  - t3.f5: `Anthropic/Employment` (NEW) <- 'User works at Anthropic for ~3 years'   [r4_new_heuristic]
  - t3.f6: `User/Employment` (NEW) <- "User's role is safety research"   [r4_new_heuristic]
  - t4.f7: `Rex/Profile` (NEW) <- 'User has a dog named Rex, Golden Retriever'   [r4_new_heuristic]
  - t4.f8: `Rex/Profile` <- 'Rex is about 11 years old, senior'   [r4_embed_reuse@0.63]
  - t6.f9: `Luna/Profile` <- 'Luna has a troublemaker/playful personality'   [r4_embed_reuse@0.67]
  topics: Luna/Profile(3), Jamie/Other(2), Anthropic/Employment(1), User/Employment(1), Rex/Profile(2)

**S2_paraphrase_consistency**
  - t1.f1: `User/Medical` (NEW) <- 'User has type 2 diabetes'   [r4_new_heuristic]
  - t2.f2: `St/Employment` (NEW) <- "User is a nurse at St. Mary's"   [r4_new_heuristic]
  - t3.f3: `User/Medical` <- 'User manages blood sugar daily due to diabetes'   [r4_embed_reuse@0.65]
  - t4.f4: `User/Medical` <- 'User is both a nurse and diabetic (occupation+condition restated)'   [r4_embed_reuse@0.65]
  - t5.f5: `St/Employment` <- 'User reassigned to pediatric ward (job detail)'   [r4_embed_reuse@0.51]
  topics: User/Medical(3), St/Employment(2)

**S3_medical_evolution**
  - t1.f1: `User/Medical` (NEW) <- 'User has lower back pain for ~1 month'   [r4_new_heuristic]
  - t2.f2: `User/Medical` <- "User's back pain is from herniated disc at L4-L5"   [r4_embed_reuse@0.62]
  - t3.f3: `PT/Medical` (NEW) <- 'User prescribed PT 2x/week and naproxen'   [r4_new_heuristic]
  - t4.f4: `User/Medical` <- "User's back pain improved ~60% after 2 weeks of PT"   [r4_embed_reuse@0.64]
  topics: User/Medical(3), PT/Medical(1)

**S4_ambiguous_referent**
  - t1.f1: `Marco/Medical` (NEW) <- "Marco (user's friend) has severe shellfish allergy"   [r4_new_heuristic]
  - t2.f2: `Marco/Medical` <- 'User has no food allergies'   [r4_embed_reuse@0.52]
  - t3.f3: `Ellie/Activity` (NEW) <- "User's daughter Ellie started violin lessons"   [r4_new_heuristic]
  - t3.f4: `Ellie/Activity` <- 'Ellie is 8 years old'   [r4_embed_reuse@0.52]
  - t4.f5: `Marco/Employment` (NEW) <- 'Marco is switching jobs, leaving Google for a startup'   [r4_new_heuristic]
  topics: Marco/Medical(2), Ellie/Activity(2), Marco/Employment(1)

**S5_boundary_cases**
  - t1.f1: `Theo/Profile` (NEW) <- "User's son Theo got a hamster named Peanut for birthday"   [r4_new_heuristic]
  - t1.f2: `Theo/Profile` <- 'User has son named Theo'   [r4_embed_reuse@0.59]
  - t2.f3: `Theo/Profile` <- 'Theo plays soccer on U10 team'   [r4_embed_reuse@0.52]
  - t4.f4: `User/Employment` (NEW) <- 'User took up woodworking hobby, built bookshelf'   [r4_new_heuristic]
  - t5.f5: `Peanut/Profile` (NEW) <- 'Peanut (hamster) escapes her cage, playful/adventurous'   [r4_new_heuristic]
  topics: Theo/Profile(3), User/Employment(1), Peanut/Profile(1)

**S6_novel_domain**
  - t1.f1: `Magpie/Business` (NEW) <- 'User owns/runs coffee roastery called Magpie Roasters in Portland'   [r4_new_heuristic]
  - t2.f2: `Magpie/Business` <- "Magpie's biggest wholesale client is Neon Cafes (14 PNW stores)"   [r4_embed_reuse@0.55]
  - t3.f3: `Rio/Business` (NEW) <- "User's business launching Rio Azul single-origin from Colombia"   [r4_new_heuristic]
  - t4.f4: `October/Activity` (NEW) <- 'User training for half-marathon in October'   [r4_new_heuristic]
  topics: Magpie/Business(2), Rio/Business(1), October/Activity(1)

### R5_hybrid

**S1_multi_entity_household**
  - t1.f1: `Luna/Profile` (NEW) <- 'User adopted calico cat Luna'   [r5_accept]
  - t1.f2: `Luna/Profile` <- 'Luna is about 2 years old'   [r5_accept]
  - t2.f3: `User/Relationships` (NEW) <- 'User has a partner named Jamie'   [r5_accept]
  - t2.f4: `Jamie/Profile` (NEW) <- 'Jamie is a graphic designer'   [r5_accept]
  - t3.f5: `User/Profile` (NEW) <- 'User works at Anthropic for ~3 years'   [r5_accept]
  - t3.f6: `User/Profile` <- "User's role is safety research"   [r5_accept]
  - t4.f7: `User/Relationships` <- 'User has a dog named Rex, Golden Retriever'   [r5_accept]
  - t4.f8: `Rex/Profile` (NEW) <- 'Rex is about 11 years old, senior'   [r5_accept]
  - t6.f9: `Luna/Profile` <- 'Luna has a troublemaker/playful personality'   [r5_accept]
  topics: Luna/Profile(3), User/Relationships(2), Jamie/Profile(1), User/Profile(2), Rex/Profile(1)

**S2_paraphrase_consistency**
  - t1.f1: `User/Medical` (NEW) <- 'User has type 2 diabetes'   [r5_accept]
  - t2.f2: `User/Medical` <- "User is a nurse at St. Mary's"   [r5_accept]
  - t3.f3: `User/Medical` <- 'User manages blood sugar daily due to diabetes'   [r5_accept]
  - t4.f4: `User/Medical` <- 'User is both a nurse and diabetic (occupation+condition restated)'   [r5_accept]
  - t5.f5: `User/Medical` <- 'User reassigned to pediatric ward (job detail)'   [r5_accept]
  topics: User/Medical(5)

**S3_medical_evolution**
  - t1.f1: `User/Medical` (NEW) <- 'User has lower back pain for ~1 month'   [r5_accept]
  - t2.f2: `User/Medical` <- "User's back pain is from herniated disc at L4-L5"   [r5_accept]
  - t3.f3: `User/Medical` <- 'User prescribed PT 2x/week and naproxen'   [r5_accept]
  - t4.f4: `User/Medical` <- "User's back pain improved ~60% after 2 weeks of PT"   [r5_accept]
  topics: User/Medical(4)

**S4_ambiguous_referent**
  - t1.f1: `Marco/Medical` (NEW) <- "Marco (user's friend) has severe shellfish allergy"   [r5_accept]
  - t2.f2: `User/Medical` (NEW) <- 'User has no food allergies'   [r5_accept]
  - t3.f3: `Ellie/Profile` (NEW) <- "User's daughter Ellie started violin lessons"   [r5_accept]
  - t3.f4: `Ellie/Profile` <- 'Ellie is 8 years old'   [r5_accept]
  - t4.f5: `Marco/Profile` (NEW) <- 'Marco is switching jobs, leaving Google for a startup'   [r5_accept]
  topics: Marco/Medical(1), User/Medical(1), Ellie/Profile(2), Marco/Profile(1)

**S5_boundary_cases**
  - t1.f1: `Theo/Pets` (NEW) <- "User's son Theo got a hamster named Peanut for birthday"   [r5_accept]
  - t1.f2: `Theo/Family` (NEW) <- 'User has son named Theo'   [r5_accept]
  - t2.f3: `Theo/Sports` (NEW) <- 'Theo plays soccer on U10 team'   [r5_accept]
  - t4.f4: `User/Hobbies` (NEW) <- 'User took up woodworking hobby, built bookshelf'   [r5_accept]
  - t5.f5: `Theo/Pets` <- 'Peanut (hamster) escapes her cage, playful/adventurous'   [r5_accept]
  topics: Theo/Pets(2), Theo/Family(1), Theo/Sports(1), User/Hobbies(1)

**S6_novel_domain**
  - t1.f1: `Magpie Roasters/Business` (NEW) <- 'User owns/runs coffee roastery called Magpie Roasters in Portland'   [r5_accept]
  - t2.f2: `Magpie Roasters/Business` <- "Magpie's biggest wholesale client is Neon Cafes (14 PNW stores)"   [r5_accept]
  - t3.f3: `Magpie Roasters/Business` <- "User's business launching Rio Azul single-origin from Colombia"   [r5_accept]
  - t4.f4: `User/Training` (NEW) <- 'User training for half-marathon in October'   [r5_accept]
  topics: Magpie Roasters/Business(3), User/Training(1)

### R6_cheap_cascade

**S1_multi_entity_household**
  - t1.f1: `Luna/Cat` (NEW) <- 'User adopted calico cat Luna'   [r6_llm_fallback]
  - t1.f2: `Luna/Cat` <- 'Luna is about 2 years old'   [r6_llm_fallback]
  - t2.f3: `Jamie/Partner` (NEW) <- 'User has a partner named Jamie'   [r6_llm_fallback]
  - t2.f4: `Jamie/Occupation` (NEW) <- 'Jamie is a graphic designer'   [r6_llm_fallback]
  - t3.f5: `User/Occupation` (NEW) <- 'User works at Anthropic for ~3 years'   [r6_llm_fallback]
  - t3.f6: `User/Occupation` <- "User's role is safety research"   [r6_llm_fallback]
  - t4.f7: `User/Router` (NEW) <- 'User has a dog named Rex, Golden Retriever'   [r6_llm_fallback]
  - t4.f8: `Rex/Dog` (NEW) <- 'Rex is about 11 years old, senior'   [r6_llm_fallback]
  - t6.f9: `Luna/Cat` <- 'Luna has a troublemaker/playful personality'   [r6_strong_embed@0.67]
  topics: Luna/Cat(3), Jamie/Partner(1), Jamie/Occupation(1), User/Occupation(2), User/Router(1), Rex/Dog(1)

**S2_paraphrase_consistency**
  - t1.f1: `Router/Networking` (NEW) <- 'User has type 2 diabetes'   [r6_llm_fallback]
  - t2.f2: `Router/Networking` <- "User is a nurse at St. Mary's"   [r6_llm_fallback]
  - t3.f3: `Router/Networking` <- 'User manages blood sugar daily due to diabetes'   [r6_llm_fallback]
  - t4.f4: `Router/Networking` <- 'User is both a nurse and diabetic (occupation+condition restated)'   [r6_strong_embed@0.72]
  - t5.f5: `User/Job` (NEW) <- 'User reassigned to pediatric ward (job detail)'   [r6_llm_fallback]
  topics: Router/Networking(4), User/Job(1)

**S3_medical_evolution**
  - t1.f1: `Lower back pain/Health` (NEW) <- 'User has lower back pain for ~1 month'   [r6_llm_fallback]
  - t2.f2: `Lower back pain/Health` <- "User's back pain is from herniated disc at L4-L5"   [r6_llm_fallback]
  - t3.f3: `Lower back pain/Health` <- 'User prescribed PT 2x/week and naproxen'   [r6_llm_fallback]
  - t4.f4: `Lower back pain/Health` <- "User's back pain improved ~60% after 2 weeks of PT"   [r6_strong_embed@0.72]
  topics: Lower back pain/Health(4)

**S4_ambiguous_referent**
  - t1.f1: `router/technology` (NEW) <- "Marco (user's friend) has severe shellfish allergy"   [r6_llm_fallback]
  - t2.f2: `router/technology` <- 'User has no food allergies'   [r6_llm_fallback]
  - t3.f3: `Ellie/music` (NEW) <- "User's daughter Ellie started violin lessons"   [r6_llm_fallback]
  - t3.f4: `router/technology` <- 'Ellie is 8 years old'   [r6_llm_fallback]
  - t4.f5: `Marco/career` (NEW) <- 'Marco is switching jobs, leaving Google for a startup'   [r6_llm_fallback]
  topics: router/technology(3), Ellie/music(1), Marco/career(1)

**S5_boundary_cases**
  - t1.f1: `<Peanut/Hamster>` (NEW) <- "User's son Theo got a hamster named Peanut for birthday"   [r6_llm_fallback]
  - t1.f2: `<Theo/Son>` (NEW) <- 'User has son named Theo'   [r6_llm_fallback]
  - t2.f3: `<Theo/Son>` <- 'Theo plays soccer on U10 team'   [r6_llm_fallback]
  - t4.f4: `<User/Woodworking>` (NEW) <- 'User took up woodworking hobby, built bookshelf'   [r6_llm_fallback]
  - t5.f5: `<Peanut/Hamster>` <- 'Peanut (hamster) escapes her cage, playful/adventurous'   [r6_llm_fallback]
  topics: <Peanut/Hamster>(2), <Theo/Son>(2), <User/Woodworking>(1)

**S6_novel_domain**
  - t1.f1: `<Magpie Roasters/Coffee Roastery>` (NEW) <- 'User owns/runs coffee roastery called Magpie Roasters in Portland'   [r6_llm_fallback]
  - t2.f2: `Neon Cafes/Cafe Chain` (NEW) <- "Magpie's biggest wholesale client is Neon Cafes (14 PNW stores)"   [r6_llm_fallback]
  - t3.f3: `Rio Azul/Single-Origin Coffee` (NEW) <- "User's business launching Rio Azul single-origin from Colombia"   [r6_llm_fallback]
  - t4.f4: `Half-Marathon/Running Training` (NEW) <- 'User training for half-marathon in October'   [r6_llm_fallback]
  topics: <Magpie Roasters/Coffee Roastery>(1), Neon Cafes/Cafe Chain(1), Rio Azul/Single-Origin Coffee(1), Half-Marathon/Running Training(1)

### R7_entity_plus_embed

**S1_multi_entity_household**
  - t1.f1: `Luna/Profile` (NEW) <- 'User adopted calico cat Luna'   [r7_accept]
  - t1.f2: `Luna/Profile` <- 'Luna is about 2 years old'   [r7_accept]
  - t2.f3: `Jamie/Profile` (NEW) <- 'User has a partner named Jamie'   [r7_accept]
  - t2.f4: `Jamie/Profile` <- 'Jamie is a graphic designer'   [r7_accept]
  - t3.f5: `User/Employment` (NEW) <- 'User works at Anthropic for ~3 years'   [r7_accept]
  - t3.f6: `User/Employment` <- "User's role is safety research"   [r7_accept]
  - t4.f7: `Rex/Profile` (NEW) <- 'User has a dog named Rex, Golden Retriever'   [r7_accept]
  - t4.f8: `Rex/Profile` <- 'Rex is about 11 years old, senior'   [r7_accept]
  - t6.f9: `Luna/Profile` <- 'Luna has a troublemaker/playful personality'   [r7_accept]
  topics: Luna/Profile(3), Jamie/Profile(2), User/Employment(2), Rex/Profile(2)

**S2_paraphrase_consistency**
  - t1.f1: `User/Health` (NEW) <- 'User has type 2 diabetes'   [r7_accept]
  - t2.f2: `User/Employment` (NEW) <- "User is a nurse at St. Mary's"   [r7_accept]
  - t3.f3: `User/Health` <- 'User manages blood sugar daily due to diabetes'   [r7_accept]
  - t4.f4: `User/Health` <- 'User is both a nurse and diabetic (occupation+condition restated)'   [r7_accept]
  - t5.f5: `User/Employment` <- 'User reassigned to pediatric ward (job detail)'   [r7_accept]
  topics: User/Health(3), User/Employment(2)

**S3_medical_evolution**
  - t1.f1: `User/Health` (NEW) <- 'User has lower back pain for ~1 month'   [r7_accept]
  - t2.f2: `User/Health` <- "User's back pain is from herniated disc at L4-L5"   [r7_accept]
  - t3.f3: `User/Health` <- 'User prescribed PT 2x/week and naproxen'   [r7_accept]
  - t4.f4: `User/Health` <- "User's back pain improved ~60% after 2 weeks of PT"   [r7_accept]
  topics: User/Health(4)

**S4_ambiguous_referent**
  - t1.f1: `Marco/Profile` (NEW) <- "Marco (user's friend) has severe shellfish allergy"   [r7_accept]
  - t2.f2: `User/Profile` (NEW) <- 'User has no food allergies'   [r7_accept]
  - t3.f3: `Ellie/Profile` (NEW) <- "User's daughter Ellie started violin lessons"   [r7_accept]
  - t3.f4: `Ellie/Profile` <- 'Ellie is 8 years old'   [r7_accept]
  - t4.f5: `Marco/Profile` <- 'Marco is switching jobs, leaving Google for a startup'   [r7_accept]
  topics: Marco/Profile(2), User/Profile(1), Ellie/Profile(2)

**S5_boundary_cases**
  - t1.f1: `Theo/Profile` (NEW) <- "User's son Theo got a hamster named Peanut for birthday"   [r7_accept]
  - t1.f2: `Theo/Profile` <- 'User has son named Theo'   [r7_accept]
  - t2.f3: `Theo/Profile` <- 'Theo plays soccer on U10 team'   [r7_accept]
  - t4.f4: `User/Profile` (NEW) <- 'User took up woodworking hobby, built bookshelf'   [r7_accept]
  - t5.f5: `Peanut/Profile` (NEW) <- 'Peanut (hamster) escapes her cage, playful/adventurous'   [r7_accept]
  topics: Theo/Profile(3), User/Profile(1), Peanut/Profile(1)

**S6_novel_domain**
  - t1.f1: `User/Business` (NEW) <- 'User owns/runs coffee roastery called Magpie Roasters in Portland'   [r7_accept]
  - t2.f2: `User/Business` <- "Magpie's biggest wholesale client is Neon Cafes (14 PNW stores)"   [r7_accept]
  - t3.f3: `User/Business` <- "User's business launching Rio Azul single-origin from Colombia"   [r7_accept]
  - t4.f4: `User/Profile` (NEW) <- 'User training for half-marathon in October'   [r7_accept]
  topics: User/Business(3), User/Profile(1)
