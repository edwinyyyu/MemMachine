# Chained Proactive Retrieval — Entity-Discovery DAG

## Setup

- n_tasks = 15 (authored to stress implicit entity discovery + per-entity facts)
- Corpora: LoCoMo-30 (conversations 26, 30, 41), reusing `arc_em_lc30_v1_{26,30,41}` EventMemory + `results/eventmemory.sqlite3`.
- Embedder: `text-embedding-3-small`, Model: `gpt-5-mini` (plan / cue-gen / entity extract / sufficiency / judge).
- K_final = 50 turns. Judge: 4 axes, 0-10 each; scored over top-20 turns by retrieval score.

## Variants

- `single_shot`: em_v2f_speakerformat baseline — one cue-gen, retrieve top-K.
- `flat_proactive`: LLM decomposes task into 3-6 info needs (no deps), retrieves each with only the task prompt as context.
- `chained_proactive`: LLM emits a DAG of `entity_discovery` + `per_entity_fact` nodes. Downstream nodes receive entities extracted from upstream hits. Up to 2 iterations; sufficiency-audit can add nodes.

## Aggregate sufficiency (LLM-judge, 0-10 per axis, mean)

| Variant | Coverage | Depth | Noise (higher=cleaner) | Task-Completion | Total | n_llm_calls | time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `single_shot` | 2.07 | 1.87 | 2.80 | 1.73 | 8.47 | 1.0 | 13.86 |
| `flat_proactive` | 2.20 | 1.60 | 2.67 | 1.73 | 8.20 | 6.1 | 60.50 |
| `chained_proactive` | 2.47 | 1.93 | 3.33 | 2.00 | 9.73 | 7.5 | 82.83 |

## Pairwise task-completion winners

- chained vs flat: chained=4, flat=4, ties=7. Mean delta (chain - flat) = +0.27.
- chained vs single-shot: chained=4, single=2, ties=9.
- flat vs single-shot: flat=4, single=4, ties=7.

## Entity discovery (chained only)

- mean entities discovered per task: 2.07
- tasks with >=1 entity discovered: 9/15

## Per-task task-completion scores

| Task | single | flat | chain | chain-flat | Discovered entities |
| --- | --- | --- | --- | --- | --- |
| `t01_surprise_party_caroline` | 2 | 1 | 4 | +3 | Mel, Caroline |
| `t02_caroline_stressors_coping` | 1 | 4 | 1 | -3 | challenges of pursuing counseling/mental health work |
| `t03_melanie_creative_projects` | 5 | 4 | 8 | +4 | painting, landscape paintings, still life paintings, nature-inspired paintings, autumn-inspired paintings, playing violin |
| `t04_gift_for_jons_mother` | 0 | 0 | 0 | +0 | Jon, Gina |
| `t05_jon_travel_plans` | 2 | 3 | 2 | -1 | Rome |
| `t06_gina_food_preferences` | 0 | 0 | 0 | +0 | — |
| `t07_maria_concerns_with_john` | 1 | 1 | 1 | +0 | Ideas not being heard, Overwhelmed by growing demand for help, Work-life balance, Bringing work home, Managing workload |
| `t08_john_work_projects` | 4 | 1 | 2 | +1 | — |
| `t09_couple_shared_hobbies` | 9 | 8 | 7 | -1 | Maria, John |
| `t10_melanie_upcoming_commitments` | 0 | 1 | 0 | -1 | — |
| `t11_jon_family_members` | 0 | 0 | 0 | +0 | — |
| `t12_caroline_health_issues` | 1 | 1 | 3 | +2 | mental health, mental health issues |
| `t13_book_club_recommendations` | 0 | 0 | 0 | +0 | Maria, John, poetry, creative writing, fairy tale, castles |
| `t14_gina_pets` | 0 | 0 | 0 | +0 | — |
| `t15_caroline_relationships` | 1 | 2 | 2 | +0 | — |

## Qualitative examples — chain vs flat

### `t03_melanie_creative_projects` (chain 8 vs flat 4)

> List the creative or artistic projects Melanie has described working on or planning in her chat with Caroline. For each project, note its current progress state and any obstacles she mentioned.

**Chained plan:**

- `n1` (entity_discovery): Enumerate all creative or artistic projects Melanie mentions — discovered: ['painting', 'landscape paintings', 'still life paintings', 'nature-inspired paintings', 'autumn-inspired paintings', 'playing violin']
- `n2` (per_entity_fact): For each project, retrieve current progress state (e.g., started, draft, finished, stalled) — for_each=n1
- `n3` (per_entity_fact): For each project, retrieve any obstacles or blockers Melanie mentions (e.g., time, materials, inspiration) — for_each=n1

**Flat needs:**

- Melanie's writing projects
- Melanie's visual-art projects
- Melanie's music or performance projects
- Melanie's craft or design projects
- Progress status and obstacles for projects

Chain judge notes: Retrieval clearly documents Melanie's main artistic work (painting—multiple pieces/types and planned autumn paintings—and pottery) with recent completions and planning notes, but gives little explicit information about obstacles or detailed progress stages.

Flat judge notes: The retrieval captures Melanie's main creative activities (painting—landscapes/still life/nature-inspired—and music/violin) with some progress notes (recent pieces, planning autumn works), but lacks explicit obstacle details and mentions of other projects, so it's only partially sufficient for the task.

### `t01_surprise_party_caroline` (chain 4 vs flat 1)

> Plan a small surprise party for Caroline using what's in her chat with Melanie. First, figure out who her close friends are from the conversation. Then, for each of those friends, note any interests or hobbies mentioned so the party activities will appeal to them.

**Chained plan:**

- `n1` (entity_discovery): Enumerate close friends / people Caroline mentions inviting — discovered: ['Mel', 'Caroline']
- `n2` (per_entity_fact): Interests, hobbies, and activity preferences mentioned for each friend — for_each=n1
- `n3` (per_entity_fact): Dietary restrictions or food preferences mentioned for each friend — for_each=n1

**Flat needs:**

- Identify names of Caroline's close friends mentioned
- Locate statements that indicate closeness or relationship type
- Extract each friend's interests, hobbies, or favorite activities
- Find any dietary restrictions, allergies, or strong dislikes
- Find scheduling notes or availability windows for planning the surprise

Chain judge notes: The retrieval contains clear hobby details for Melanie and Caroline (pottery, painting, piano, hiking, biking, camping) but no other named close friends—only vague mentions like 'the gang'—so friend identification is incomplete for a full party plan.

Flat judge notes: The snippets only hint that Caroline has close friends (known 4 years) and include a stray music/guitar question, but provide no friend names or per-friend interests—mostly generic small talk—so there's insufficient specific information to plan the party.


## Verdict

Chained ~ flat (delta = +0.27). Entity discovery happens naturally in flat decomposition, probably because the LLM listing 3-6 info needs already includes entity-enumeration as one of them.

## Outputs

- Raw: `results/chained_proactive.json`
- This report: `results/chained_proactive.md`
- Tasks: `data/chained_proactive_tasks.json`
- Source: `chained_proactive.py`, `chained_eval.py`
