# Hard temporal-stress benchmark — pipeline results

**Benchmark**: hard_bench (synthetic stress)
**Docs**: 600, **queries**: 75 (30 easy / 30 medium / 15 hard)
**Cost**: $0.1910, wall: 157.6s

**Extraction**: 1.06 mean te/doc, 1.05 mean te/query, timeouts=0+0, errors=0+0
**Lattice**: {'n_docs_tagged': 600, 'n_rows': 3600, 'n_unique_tags': 529, 'avg_tags_per_doc': 6.0, 'max_tags_per_doc': 6, 'min_tags_per_doc': 6}

## ALL subset

| Variant | n | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@10 |
|---------|---|-----|-----|-----|------|-----|---------|
| SEMANTIC-ONLY | 75 | 0.600 | 0.733 | 0.800 | 0.853 | 0.689 | 0.722 |
| T-only | 75 | 0.013 | 0.027 | 0.067 | 0.093 | 0.059 | 0.047 |
| V7 (T+S, cv=0.20) | 75 | 0.613 | 0.653 | 0.667 | 0.747 | 0.657 | 0.668 |
| V7 (T+S, cv=0.10) | 75 | 0.693 | 0.760 | 0.800 | 0.867 | 0.749 | 0.772 |
| V7 (T+S, cv=0.30) | 75 | 0.613 | 0.653 | 0.667 | 0.747 | 0.656 | 0.668 |
| V7 (T+S, cv=0.50) | 75 | 0.613 | 0.653 | 0.667 | 0.747 | 0.656 | 0.668 |
| V7L (T+S+L, cv=0.20) | 75 | 0.613 | 0.653 | 0.667 | 0.747 | 0.657 | 0.668 |
| V7L (T+S+L, cv=0.10) | 75 | 0.693 | 0.760 | 0.800 | 0.867 | 0.749 | 0.772 |
| V7L (T+S+L, cv=0.30) | 75 | 0.613 | 0.653 | 0.667 | 0.747 | 0.656 | 0.668 |
| V7L (T+S+L, cv=0.50) | 75 | 0.613 | 0.653 | 0.667 | 0.747 | 0.656 | 0.668 |

## EASY subset

| Variant | n | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@10 |
|---------|---|-----|-----|-----|------|-----|---------|
| SEMANTIC-ONLY | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| T-only | 30 | 0.000 | 0.000 | 0.000 | 0.000 | 0.016 | 0.000 |
| V7 (T+S, cv=0.20) | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7 (T+S, cv=0.10) | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7 (T+S, cv=0.30) | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7 (T+S, cv=0.50) | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7L (T+S+L, cv=0.20) | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7L (T+S+L, cv=0.10) | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7L (T+S+L, cv=0.30) | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7L (T+S+L, cv=0.50) | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## MEDIUM subset

| Variant | n | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@10 |
|---------|---|-----|-----|-----|------|-----|---------|
| SEMANTIC-ONLY | 30 | 0.467 | 0.767 | 0.900 | 0.967 | 0.641 | 0.719 |
| T-only | 30 | 0.033 | 0.067 | 0.100 | 0.100 | 0.089 | 0.067 |
| V7 (T+S, cv=0.20) | 30 | 0.433 | 0.433 | 0.467 | 0.567 | 0.473 | 0.480 |
| V7 (T+S, cv=0.10) | 30 | 0.533 | 0.567 | 0.633 | 0.800 | 0.593 | 0.633 |
| V7 (T+S, cv=0.30) | 30 | 0.433 | 0.433 | 0.467 | 0.567 | 0.473 | 0.480 |
| V7 (T+S, cv=0.50) | 30 | 0.433 | 0.433 | 0.467 | 0.567 | 0.473 | 0.480 |
| V7L (T+S+L, cv=0.20) | 30 | 0.433 | 0.433 | 0.467 | 0.567 | 0.473 | 0.480 |
| V7L (T+S+L, cv=0.10) | 30 | 0.533 | 0.567 | 0.633 | 0.800 | 0.593 | 0.633 |
| V7L (T+S+L, cv=0.30) | 30 | 0.433 | 0.433 | 0.467 | 0.567 | 0.473 | 0.480 |
| V7L (T+S+L, cv=0.50) | 30 | 0.433 | 0.433 | 0.467 | 0.567 | 0.473 | 0.480 |

## HARD subset

| Variant | n | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@10 |
|---------|---|-----|-----|-----|------|-----|---------|
| SEMANTIC-ONLY | 15 | 0.067 | 0.133 | 0.200 | 0.333 | 0.160 | 0.170 |
| T-only | 15 | 0.000 | 0.000 | 0.133 | 0.267 | 0.084 | 0.100 |
| V7 (T+S, cv=0.20) | 15 | 0.200 | 0.400 | 0.400 | 0.600 | 0.338 | 0.383 |
| V7 (T+S, cv=0.10) | 15 | 0.400 | 0.667 | 0.733 | 0.733 | 0.557 | 0.594 |
| V7 (T+S, cv=0.30) | 15 | 0.200 | 0.400 | 0.400 | 0.600 | 0.336 | 0.383 |
| V7 (T+S, cv=0.50) | 15 | 0.200 | 0.400 | 0.400 | 0.600 | 0.336 | 0.383 |
| V7L (T+S+L, cv=0.20) | 15 | 0.200 | 0.400 | 0.400 | 0.600 | 0.338 | 0.383 |
| V7L (T+S+L, cv=0.10) | 15 | 0.400 | 0.667 | 0.733 | 0.733 | 0.557 | 0.594 |
| V7L (T+S+L, cv=0.30) | 15 | 0.200 | 0.400 | 0.400 | 0.600 | 0.336 | 0.383 |
| V7L (T+S+L, cv=0.50) | 15 | 0.200 | 0.400 | 0.400 | 0.600 | 0.336 | 0.383 |

## Failure analysis

- Total queries: 75
- V7 wins (V7 rank=1, sem rank>1): 10
- V7 losses (V7 rank>1, sem rank=1): 9
- Persistent misses (both >5): 9

### V7 wins

- **q_medium_005** [medium] `When was Davis promoted in Q1 2023?`
  - gold: `Marcus Davis was promoted to engineering manager on Jan 30, 2023.` (sem rank=3, V7 rank=1, V7L rank=1, T-only rank=13)
  - sem top3: `['Sarah Davis was promoted to senior PM on Nov 6, 2023.', 'Marcus Davis was promoted to senior PM on Sep 8, 2023.', 'Marcus Davis was promoted to engineering manager on Jan 30, 2023.']`
- **q_medium_006** [medium] `When did Kim deliver the quarterly review in Q3 2023?`
  - gold: `Kim Davis delivered the quarterly review to leadership on Aug 14, 2023.` (sem rank=13, V7 rank=1, V7L rank=1, T-only rank=1)
  - sem top3: `['Kim Park delivered the quarterly review to leadership on Mar 30, 2022.', 'Kim Johnson delivered the quarterly review to leadership on Nov 23, 2024.', 'Daniel Chen delivered the quarterly review to leadership on Dec 3, 2023.']`
- **q_medium_010** [medium] `When did Kim attend the company offsite in Q1 2022?`
  - gold: `Kim Johnson attended the company offsite in Toronto on Jan 25, 2022.` (sem rank=3, V7 rank=1, V7L rank=1, T-only rank=11)
  - sem top3: `['Kim Johnson attended the company offsite in Tokyo on Mar 13, 2023.', 'Kim Chen attended the company offsite in Dublin on Jun 17, 2022.', 'Kim Johnson attended the company offsite in Toronto on Jan 25, 2022.']`
- **q_medium_013** [medium] `When did Kim attend the company offsite in Q1 2024?`
  - gold: `Kim Patel attended the company offsite in Seattle on Jan 28, 2024.` (sem rank=5, V7 rank=1, V7L rank=1, T-only rank=24)
  - sem top3: `['Kim Johnson attended the company offsite in Tokyo on Mar 13, 2023.', 'Kim Johnson attended the company offsite in Toronto on Jan 25, 2022.', 'Kim Chen attended the company offsite in Dublin on Jun 17, 2022.']`
- **q_medium_018** [medium] `When was Sarah awarded employee of the month in Q4 2023?`
  - gold: `Sarah Park was awarded employee of the month on Nov 9, 2023.` (sem rank=2, V7 rank=1, V7L rank=1, T-only rank=27)
  - sem top3: `['Sarah Johnson was awarded employee of the month on Mar 1, 2023.', 'Sarah Park was awarded employee of the month on Nov 9, 2023.', 'Sarah Lee was awarded employee of the month on Aug 13, 2024.']`
- **q_medium_020** [medium] `When was Kim awarded employee of the month in Q1 2024?`
  - gold: `Kim Patel was awarded employee of the month on Jan 3, 2024.` (sem rank=7, V7 rank=1, V7L rank=1, T-only rank=2)
  - sem top3: `['Kim Nguyen was awarded employee of the month on Apr 6, 2022.', 'Kim Park was awarded employee of the month on Nov 3, 2023.', 'Quinn King was awarded employee of the month on Jun 25, 2024.']`
- **q_medium_027** [medium] `When was Priya awarded employee of the month in Q2 2023?`
  - gold: `Priya Nguyen was awarded employee of the month on May 3, 2023.` (sem rank=2, V7 rank=1, V7L rank=1, T-only rank=16)
  - sem top3: `['Priya Park was awarded employee of the month on Mar 27, 2022.', 'Priya Nguyen was awarded employee of the month on May 3, 2023.', 'Priya Davis was awarded employee of the month on Dec 14, 2024.']`
- **q_hard_001** [hard] `When did someone on the team move to a new office in Q1 2024?`
  - gold: `Casey Smith moved to the Singapore office on Jan 5, 2024.` (sem rank=15, V7 rank=1, V7L rank=1, T-only rank=4)
  - sem top3: `['Quinn Young moved to the Dublin office on Nov 27, 2023.', 'Marcus Park moved to the London office on Dec 14, 2023.', 'Daniel Chen moved to the Singapore office on May 10, 2024.']`
- **q_hard_006** [hard] `When did someone on the team lead the project kickoff in Q4 2022?`
  - gold: `Ava Harris led the Polaris project kickoff on Oct 15, 2022.` (sem rank=3, V7 rank=1, V7L rank=1, T-only rank=13)
  - sem top3: `['Ethan King led the Phoenix project kickoff on Dec 25, 2024.', 'Priya Davis led the Aurora project kickoff on Jun 30, 2023.', 'Ava Harris led the Polaris project kickoff on Oct 15, 2022.']`
- **q_hard_011** [hard] `When did someone on the team attend the company offsite in Q4 2023?`
  - gold: `Taylor Harris attended the company offsite in London on Oct 17, 2023.` (sem rank=7, V7 rank=1, V7L rank=1, T-only rank=7)
  - sem top3: `['Kim Patel attended the company offsite in Seattle on Jan 28, 2024.', 'Marcus Patel attended the company offsite in Seattle on Jul 24, 2023.', 'Taylor Hall attended the company offsite in Seattle on Mar 3, 2024.']`

### V7 losses

- **q_medium_000** [medium] `When did Kim deliver the quarterly review in Q4 2024?`
  - gold: `Kim Johnson delivered the quarterly review to leadership on Nov 23, 2024.` (sem rank=1, V7 rank=7, V7L rank=7, T-only rank=27)
  - V7 top3: `['Casey Martinez hit a five-year work anniversary on Oct 7, 2024.', 'Kim Johnson completed onboarding on Oct 9, 2024.', 'Ava Young left the company on Oct 10, 2024.']`
- **q_medium_001** [medium] `When did Priya lead the project kickoff in Q2 2023?`
  - gold: `Priya Davis led the Aurora project kickoff on Jun 30, 2023.` (sem rank=1, V7 rank=6, V7L rank=6, T-only rank=59)
  - V7 top3: `['Lucas Garcia completed onboarding on Apr 4, 2023.', 'Alex Young completed the PMP certification on Apr 5, 2023.', 'Robin Anderson was awarded employee of the month on Apr 7, 2023.']`
- **q_medium_002** [medium] `When did Patel complete onboarding in Q4 2024?`
  - gold: `Priya Patel completed onboarding on Dec 24, 2024.` (sem rank=1, V7 rank=14, V7L rank=14, T-only rank=45)
  - V7 top3: `['Priya Johnson completed onboarding on Oct 15, 2024.', 'Casey Martinez hit a five-year work anniversary on Oct 7, 2024.', 'Kim Johnson completed onboarding on Oct 9, 2024.']`
- **q_medium_003** [medium] `When did Marcus deliver the quarterly review in Q4 2022?`
  - gold: `Marcus Davis delivered the quarterly review to leadership on Dec 10, 2022.` (sem rank=1, V7 rank=30, V7L rank=30, T-only rank=33)
  - V7 top3: `['Alex Walker delivered the quarterly review to leadership on Oct 25, 2022.', 'Ava Harris delivered the quarterly review to leadership on Oct 3, 2022.', 'Liam Anderson hosted a workshop on design systems on Oct 5, 2022.']`
- **q_medium_012** [medium] `When was Sarah awarded employee of the month in Q1 2023?`
  - gold: `Sarah Johnson was awarded employee of the month on Mar 1, 2023.` (sem rank=1, V7 rank=26, V7L rank=26, T-only rank=30)
  - V7 top3: `['Morgan Harris delivered the quarterly review to leadership on Jan 4, 2023.', 'Kim Chen hit a five-year work anniversary on Jan 1, 2023.', 'Ava Martinez moved to the Austin office on Jan 9, 2023.']`
- **q_medium_014** [medium] `When did Marcus complete onboarding in Q1 2022?`
  - gold: `Marcus Patel completed onboarding on Mar 25, 2022.` (sem rank=1, V7 rank=33, V7L rank=33, T-only rank=47)
  - V7 top3: `['Sarah Park was promoted to director on Jan 2, 2022.', 'Jordan Thomas attended the company offsite in Mexico City on Jan 2, 2022.', 'Ava Martinez presented at the KubeCon conference on Jan 2, 2022.']`
- **q_medium_015** [medium] `When did Sarah deliver the quarterly review in Q4 2024?`
  - gold: `Sarah Lee delivered the quarterly review to leadership on Dec 25, 2024.` (sem rank=1, V7 rank=27, V7L rank=27, T-only rank=46)
  - V7 top3: `['Morgan Anderson delivered the quarterly review to leadership on Nov 14, 2024.', 'Liam Anderson delivered the quarterly review to leadership on Nov 7, 2024.', 'Kim Johnson completed onboarding on Oct 9, 2024.']`
- **q_medium_024** [medium] `When did Marcus leave the company in Q3 2024?`
  - gold: `Marcus Park left the company on Sep 3, 2024.` (sem rank=1, V7 rank=23, V7L rank=23, T-only rank=30)
  - V7 top3: `['Maya Anderson delivered the quarterly review to leadership on Jul 1, 2024.', 'Liam Walker mentored a new hire starting on Jul 2, 2024.', 'Mia Garcia completed the AWS Solutions Architect certification on Jul 3, 2024.']`
- **q_hard_008** [hard] `When did someone on the team present at a conference in Q4 2023?`
  - gold: `Ava White presented at the ICML conference on Oct 27, 2023.` (sem rank=1, V7 rank=2, V7L rank=2, T-only rank=15)
  - V7 top3: `['Taylor Harris delivered the quarterly review to leadership on Oct 4, 2023.', 'Ava White presented at the ICML conference on Oct 27, 2023.', 'Riley Young left the company on Oct 3, 2023.']`

### Persistent misses (both lost)

- **q_medium_009** [medium] `When did Marcus complete onboarding in Q3 2022?`
  - gold: `Marcus Park completed onboarding on Aug 19, 2022.` (sem rank=6, V7 rank=17)
  - sem top3: `['Marcus Patel completed onboarding on Mar 25, 2022.', 'Marcus Johnson completed onboarding on Mar 5, 2024.', 'Morgan Garcia completed onboarding on Jun 13, 2023.']`
- **q_hard_002** [hard] `When did someone on the team attend the company offsite in Q2 2024?`
  - gold: `Taylor Walker attended the company offsite in Seattle on May 29, 2024.` (sem rank=16, V7 rank=11)
  - sem top3: `['Kim Patel attended the company offsite in Seattle on Jan 28, 2024.', 'Marcus Patel attended the company offsite in Seattle on Jul 24, 2023.', 'Quinn Young attended the company offsite in Berlin on Oct 21, 2022.']`
- **q_hard_003** [hard] `When did someone on the team host a workshop in Q3 2023?`
  - gold: `Sarah Johnson hosted a workshop on feature flags on Aug 26, 2023.` (sem rank=14, V7 rank=40)
  - sem top3: `['Quinn King hosted a workshop on feature flags on Nov 3, 2023.', 'Liam White hosted a workshop on leadership on Dec 18, 2023.', 'Quinn Harris hosted a workshop on leadership on Apr 21, 2022.']`
- **q_hard_004** [hard] `When did someone on the team join the team in Q2 2022?`
  - gold: `Liam Harris joined the search team on Jun 24, 2022.` (sem rank=25, V7 rank=61)
  - sem top3: `['Quinn Garcia joined the infrastructure team on Dec 26, 2024.', 'Taylor Wilson joined the ML team on Jul 15, 2022.', 'Riley Brown joined the frontend team on Aug 8, 2022.']`
- **q_hard_005** [hard] `When did someone on the team hit a work anniversary in Q3 2022?`
  - gold: `Noah Martinez hit a five-year work anniversary on Aug 20, 2022.` (sem rank=34, V7 rank=28)
  - sem top3: `['Quinn Anderson hit a five-year work anniversary on Aug 26, 2023.', 'Quinn Young was awarded employee of the month on Oct 28, 2023.', 'Sarah Nguyen hit a five-year work anniversary on Dec 18, 2022.']`
