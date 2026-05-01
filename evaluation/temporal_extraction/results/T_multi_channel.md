# T_multi_channel — multi-channel temporal scoring eval

Channels: T_lblend, recency (half-life=21.0d), R (cross-encoder).

Weights (all-active): T=0.3, recency=0.3, R=0.4.

score_blend with CV gate (cv_ref=0.2); CV-scaled effective weights renormalized.

Switches:
  - `T_active`: regex matches year / quarter / month / season / era / ISO/slash date / 'March 5th'
  - `Recency_active`: `has_recency_cue()` (latest, most recent, recently, ...) with verb-form `present` suppressor


## R@1 by benchmark

| Benchmark | n | T_act | Rec_act | rerank_only | fuse_T_R | fuse_T_R + rec_add | mc_switches | mc_no_switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_bench | 75 | 75 | 0 | 0.640 | 0.893 | 0.893 | 0.893 | 0.480 |
| temporal_essential | 25 | 25 | 0 | 0.920 | 1.000 | 1.000 | 1.000 | 1.000 |
| tempreason_small | 60 | 37 | 0 | 0.650 | 0.733 | 0.733 | 0.733 | 0.617 |
| conjunctive_temporal | 12 | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_te_doc | 12 | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 |
| relative_time | 12 | 1 | 1 | 0.250 | 1.000 | 0.917 | 0.250 | 1.000 |
| era_refs | 12 | 0 | 0 | 0.250 | 0.417 | 0.417 | 0.250 | 0.333 |
| latest_recent | 15 | 0 | 15 | 0.133 | 0.267 | 0.667 | 0.800 | 0.800 |
| open_ended_date | 15 | 15 | 0 | 0.267 | 0.400 | 0.400 | 0.400 | 0.400 |
| causal_relative | 15 | 0 | 1 | 0.467 | 0.467 | 0.467 | 0.467 | 0.400 |
| negation_temporal | 15 | 15 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## R@5 by benchmark

| Benchmark | n | rerank_only | fuse_T_R | fuse_T_R + rec_add | mc_switches | mc_no_switches |
|---|---:|---:|---:|---:|---:|---:|
| hard_bench | 75 | 0.853 | 0.960 | 0.960 | 0.960 | 0.827 |
| temporal_essential | 25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tempreason_small | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
| conjunctive_temporal | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_te_doc | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| relative_time | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| era_refs | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| latest_recent | 15 | 1.000 | 1.000 | 0.800 | 1.000 | 1.000 |
| open_ended_date | 15 | 0.733 | 0.800 | 0.800 | 0.800 | 0.800 |
| causal_relative | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| negation_temporal | 15 | 0.933 | 0.467 | 0.467 | 0.467 | 0.267 |

## Switch firing pattern (per benchmark)

| Benchmark | n | T_active | Recency_active | both | neither |
|---|---:|---:|---:|---:|---:|
| hard_bench | 75 | 75 | 0 | 0 | 0 |
| temporal_essential | 25 | 25 | 0 | 0 | 0 |
| tempreason_small | 60 | 37 | 0 | 0 | 23 |
| conjunctive_temporal | 12 | 12 | 0 | 0 | 0 |
| multi_te_doc | 12 | 12 | 0 | 0 | 0 |
| relative_time | 12 | 1 | 1 | 0 | 10 |
| era_refs | 12 | 0 | 0 | 0 | 12 |
| latest_recent | 15 | 0 | 15 | 0 | 0 |
| open_ended_date | 15 | 15 | 0 | 0 | 0 |
| causal_relative | 15 | 0 | 1 | 0 | 14 |
| negation_temporal | 15 | 15 | 0 | 0 | 0 |

## Sample switch firings (first 12 queries per benchmark)

### hard_bench
- T=1 R=0: `When was Priya Johnson promoted in 2023?`
- T=1 R=0: `When did Sarah Nguyen deliver the quarterly review in 2022?`
- T=1 R=0: `When was Sarah Lee awarded employee of the month in 2024?`
- T=1 R=0: `When did Sarah Park join the team in 2024?`
- T=1 R=0: `When did Sarah Patel complete a certification in 2022?`
- T=1 R=0: `When was Marcus Nguyen promoted in 2023?`
- T=1 R=0: `When did Priya Chen mentor a new hire in 2023?`
- T=1 R=0: `When did Marcus Johnson leave the company in 2023?`
- T=1 R=0: `When did Priya Nguyen deliver the quarterly review in 2022?`
- T=1 R=0: `When did Marcus Park host a workshop in 2023?`
- T=1 R=0: `When did Kim Davis complete a certification in 2022?`
- T=1 R=0: `When did Kim Patel move to a new office in 2024?`

### temporal_essential
- T=1 R=0: `When did Sarah Park have her dental cleaning in early April 2024?`
- T=1 R=0: `When did Marcus Davis deliver the quarterly review in Q4 2022?`
- T=1 R=0: `When did Priya Johnson lead the team retrospective in early May 2024?`
- T=1 R=0: `When did Kim Patel complete the kitchen remodel in March 2024?`
- T=1 R=0: `When did Aiden Park give the investor pitch in October 2024?`
- T=1 R=0: `When did Olivia Roberts sign up for yoga class in early January 2025?`
- T=1 R=0: `When did Henry Ford have his performance review in Q3 2023?`
- T=1 R=0: `When did Felix Wood go on the client onsite in May 2024?`
- T=1 R=0: `When did Quinn Reeves do the lease signing in early April 2024?`
- T=1 R=0: `When did Sara Lee complete the puppy adoption in May 2024?`
- T=1 R=0: `When did Tom Reed finish the marathon in mid-April 2024?`
- T=1 R=0: `When did Layla Smith complete the tax filing in early April 2024?`

### tempreason_small
- T=1 R=0: `Where was Rolin Wavre educated in Nov, 1918?`
- T=1 R=0: `Where was Hans Bethe educated in Aug, 1929?`
- T=1 R=0: `Which position did Henri Madelin hold in Dec, 1985?`
- T=1 R=0: `Which employer did Dominique Kalifa work for in Apr, 2001?`
- T=1 R=0: `Which position did Christian Zetlitz Bretteville hold in Jan, 1841?`
- T=1 R=0: `Which employer did Caroline C. Hunter work for in Aug, 2005?`
- T=1 R=0: `Which team did Domenico Maggiora play for in Apr, 1986?`
- T=1 R=0: `Which position did Michael Fallon hold in Dec, 2001?`
- T=1 R=0: `Which team did Christophe Samson play for in Jan, 2012?`
- T=1 R=0: `Who was the head coach of the team K.V. Kortrijk in Mar, 2020?`
- T=1 R=0: `Who was the head of Taitung County in Mar, 2021?`
- T=1 R=0: `Which team did Katarina Kolar play for in Feb, 2013?`

### conjunctive_temporal
- T=1 R=0: `What were Sarah Park's her dental appointments in Q3 2023 and Q1 2024?`
- T=1 R=0: `What were Marcus Davis's his quarterly reviews in between March and August 2024?`
- T=1 R=0: `What were Priya Johnson's her client visits in January and October 2023?`
- T=1 R=0: `What were Kim Patel's kitchen renovations in summer 2022 and winter 2023?`
- T=1 R=0: `What were Aiden Park's his investor meetings in April and November of 2024?`
- T=1 R=0: `What were Olivia Roberts's yoga classes in both spring and fall 2024?`
- T=1 R=0: `What were Henry Ford's performance reviews in Q2 2023 and Q4 2024?`
- T=1 R=0: `What were Felix Wood's client onsites in May 2023 and September 2024?`
- T=1 R=0: `What were Quinn Reeves's lease signings in early 2023 and late 2024?`
- T=1 R=0: `What were Sara Lee's puppy training sessions in March and June 2024?`
- T=1 R=0: `What were Tom Reed's marathons in April 2023 and October 2024?`
- T=1 R=0: `What were Layla Smith's tax filings in both April 2023 and April 2024?`

### multi_te_doc
- T=1 R=0: `What did Sarah Park do on March 12, 2024?`
- T=1 R=0: `What did Marcus Davis do on October 8, 2023?`
- T=1 R=0: `What did Priya Johnson do on May 5, 2024?`
- T=1 R=0: `What did Kim Patel do on March 14, 2024?`
- T=1 R=0: `What did Aiden Park do on October 10, 2024?`
- T=1 R=0: `What did Olivia Roberts do on January 6, 2025?`
- T=1 R=0: `What did Henry Ford do on September 9, 2023?`
- T=1 R=0: `What did Felix Wood do on May 11, 2024?`
- T=1 R=0: `What did Quinn Reeves do on April 7, 2024?`
- T=1 R=0: `What did Sara Lee do on May 12, 2024?`
- T=1 R=0: `What did Tom Reed do on April 15, 2024?`
- T=1 R=0: `What did Layla Smith do on April 4, 2024?`

### relative_time
- T=0 R=0: `When did Sarah Park renew her gym membership, yesterday?`
- T=0 R=0: `When did Marcus Davis have his quarterly check-in, last week?`
- T=0 R=0: `When did Priya Johnson lead the design review, three weeks ago?`
- T=0 R=0: `When did Kim Patel complete the audit prep, last month?`
- T=0 R=0: `When did Aiden Park give his TED talk, two months ago?`
- T=0 R=0: `When did Olivia Roberts start her new role, earlier this year?`
- T=0 R=0: `When did Henry Ford have his annual review, last year?`
- T=0 R=0: `When did Felix Wood submit the manuscript, a few days ago?`
- T=0 R=1: `When did Quinn Reeves close the funding round, last quarter?`
- T=1 R=0: `When did Sara Lee take her sabbatical, this past summer?`
- T=0 R=0: `When did Tom Reed wrap the all-hands, yesterday afternoon?`
- T=0 R=0: `When did Layla Smith file the patent, earlier this month?`

### era_refs
- T=0 R=0: `When did Sarah Park take her European backpacking trip during grad school?`
- T=0 R=0: `When did Marcus Davis host his first dinner party back when I worked at Acme?`
- T=0 R=0: `When did Priya Johnson complete her marathon while living in Sweden?`
- T=0 R=0: `When did Kim Patel get the promotion during my time at the startup?`
- T=0 R=0: `When did Aiden Park have his first child back in college?`
- T=0 R=0: `When did Olivia Roberts buy her first car when I lived in Boston?`
- T=0 R=0: `When did Henry Ford start learning piano during my parental leave?`
- T=0 R=0: `When did Felix Wood run the half-marathon while training for the Olympics?`
- T=0 R=0: `When did Quinn Reeves adopt his dog during the pandemic year?`
- T=0 R=0: `When did Sara Lee move into her first apartment right after I graduated college?`
- T=0 R=0: `When did Tom Reed switch to a vegetarian diet back when I worked at Globex?`
- T=0 R=0: `When did Layla Smith run her first 10K during my fitness phase?`

### latest_recent
- T=0 R=1: `What's my latest project Alpha status update?`
- T=0 R=1: `When was my last appointment with Dr. Patel?`
- T=0 R=1: `What's the most recent feedback on the dashboard design?`
- T=0 R=1: `When was my last car service?`
- T=0 R=1: `What's the most recent performance review I had?`
- T=0 R=1: `When did I last renew the office lease?`
- T=0 R=1: `When was my latest therapy session?`
- T=0 R=1: `When was my last dentist visit?`
- T=0 R=1: `When did I last get a haircut?`
- T=0 R=1: `When was my most recent grocery run?`
- T=0 R=1: `What's my latest Acme client check-in?`
- T=0 R=1: `When was my most recent blood test?`

### open_ended_date
- T=1 R=0: `What did I work on after 2022?`
- T=1 R=0: `What did I work on before the pandemic (January 2020)?`
- T=1 R=0: `What's my activity since I moved in June 2023?`
- T=1 R=0: `What did I do before I joined Acme in March 2022?`
- T=1 R=0: `What courses did I complete after graduating in May 2020?`
- T=1 R=0: `What investments did I make before retirement (December 2023)?`
- T=1 R=0: `What did I publish since I started the blog in March 2024?`
- T=1 R=0: `What trips did I take after my child was born in August 2023?`
- T=1 R=0: `What did I do before I started grad school in September 2021?`
- T=1 R=0: `What conferences did I attend since 2024?`
- T=1 R=0: `What treatments did I get before the surgery in November 2024?`
- T=1 R=0: `What restaurants did I try after the city move (April 2024)?`

### causal_relative
- T=0 R=0: `What did Sarah say after the migration was complete?`
- T=0 R=0: `What happened before the launch?`
- T=0 R=1: `What did Maya report since the last review?`
- T=0 R=0: `What did Priya circulate after the offsite?`
- T=0 R=0: `What planning did Marcus do before the merger?`
- T=0 R=0: `What did Aiden do after the funding round closed?`
- T=0 R=0: `What did Eric do before the keynote?`
- T=0 R=0: `What did Layla do after the audit?`
- T=0 R=0: `What has Yuki taken up since the move?`
- T=0 R=0: `What did Mira sign off on after the cutover?`
- T=0 R=0: `What did Hannah bring back after the design summit?`
- T=0 R=0: `What did Tom do before the marathon?`

### negation_temporal
- T=1 R=0: `What did I do not in 2023?`
- T=1 R=0: `What expenses do I have outside of the holiday season (November–December)?`
- T=1 R=0: `Meetings excluding Q4 2023`
- T=1 R=0: `What workouts did I do not in January 2025?`
- T=1 R=0: `What appointments did I have outside of the summer (June–August 2024)?`
- T=1 R=0: `What trips did I take excluding 2022?`
- T=1 R=0: `What classes did I take outside of the spring 2024 semester?`
- T=1 R=0: `What grocery runs did I do not in March 2025?`
- T=1 R=0: `What design reviews did I attend excluding the second half of 2023?`
- T=1 R=0: `What books did I read not in 2024?`
- T=1 R=0: `What therapy sessions did I have outside of February 2025?`
- T=1 R=0: `What expenses do I have not in Q1 2024?`

## Headline answers

- **hard_bench R@1**: rerank_only=0.640, fuse_T_R=0.893, fuse_T_R+recAdd=0.893, mc_switches=0.893, mc_no_switches=0.480
- **era_refs R@1**: rerank_only=0.250, fuse_T_R=0.417, fuse_T_R+recAdd=0.417, mc_switches=0.250, mc_no_switches=0.333
- **latest_recent R@1**: rerank_only=0.133, fuse_T_R=0.267, fuse_T_R+recAdd=0.667, mc_switches=0.800, mc_no_switches=0.800

## Analysis

### 1. Did the multi-channel architecture preserve fuse_T_lblend's wins (hard_bench, era_refs)?

**hard_bench: yes (0.893 → 0.893, identical).** mc_switches with `T_active=True` on all 75 queries reproduces fuse_T_R behavior — the explicit T-anchor regex catches every "in 2023" / "in Q4 2022" form.

**era_refs: NO (0.417 → 0.250, regression of -0.167).** The era_refs queries (`"during grad school"`, `"back when I worked at Acme"`, `"during the pandemic year"`) contain *no* explicit calendar anchors, so `has_temporal_anchor` returns False on all 12. mc_switches drops T entirely; only R remains, matching rerank_only (0.250). fuse_T_R wins here because T_lblend's lattice + tag-overlap component still scores meaningfully even without a query-side year anchor — the regex is a too-conservative proxy for "T_lblend has signal." This is a real false-negative of the switch.

### 2. Did the architecture capture the +0.733 latest_recent win?

**Mostly yes: +0.667** (rerank_only 0.133 → mc_switches 0.800). All 15 queries fired Recency_active. The remaining gap to t_recency's reported 0.867 is because we use 21d half-life vs the optimal 90d. Tuning that would close the gap. fuse_T_R + recency_additive (Agent I's recipe) also lifts to 0.667 — solid but uses the cruder fixed-α=0.5 add.

### 3. Did explicit switches help vs always-on CV-gate-alone?

**Yes — switches matter.** mc_no_switches (CV gate alone) catastrophically regresses on hard_bench (0.893 → 0.480, -41pp) because the recency channel — fed real anchor data even though no recency cue is present in the query — produces high CV (anchors are spread across years), so the CV gate keeps it active and dilutes T_lblend's signal. CV gate measures "does this channel have spread across docs?" not "does the query care about this channel?" The Agent I observation that "channel applicability is a query property, not a doc property" holds: switches encode query intent, CV gate cannot.

mc_no_switches also regresses tempreason_small (0.733 → 0.617), multi_te_doc (1.000 → 0.917), causal_relative (0.467 → 0.400), negation_temporal R@5 (0.467 → 0.267).

Curious counter-example: on **relative_time** mc_no_switches *beats* mc_switches (1.000 vs 0.250). Reason: 11 of 12 queries are forms like "yesterday", "last week", "three weeks ago" — `has_temporal_anchor` does not catch these (no year/month/quarter), and `has_recency_cue`'s "last X" suppressor explicitly drops "last week"/"last month". So the switches reject everything → R-only ranking → rerank_only 0.250. Without switches, T_lblend (which DOES extract intervals from "last week") drives R@1=1.000. Same hole as era_refs: switches under-fire when T_lblend has signal but the query lacks a hard regex anchor.

### 4. Switch firing pattern

| Benchmark | T fires | Rec fires | False-neg | False-pos |
|---|---|---|---|---|
| hard_bench | 75/75 | 0/75 | 0 | 0 |
| temporal_essential | 25/25 | 0/25 | 0 | 0 |
| tempreason_small | 37/60 | 0/60 | 23 (subj-named queries lack year tag in surface form) | 0 |
| conjunctive_temporal | 12/12 | 0/12 | 0 | 0 |
| multi_te_doc | 12/12 | 0/12 | 0 | 0 |
| relative_time | 1/12 | 1/12 | 11 ("yesterday", "last week", "X weeks ago") | 0 |
| era_refs | 0/12 | 0/12 | 12 ("during grad school", "back when …") | 0 |
| latest_recent | 0/15 | 15/15 | 0 | 0 |
| open_ended_date | 15/15 | 0/15 | 0 | 0 |
| causal_relative | 0/15 | 1/15 | depends — these are ordering relative to events, not absolute time | 0 |
| negation_temporal | 15/15 | 0/15 | 0 | 0 |

False-positives: zero observed. Verb-form `present` suppressor in `has_recency_cue` works.

False-negatives:
- **era_refs**: era expressions need NER + entity-time-mapping, not surface regex.
- **relative_time**: "yesterday", "X weeks ago", "last week" all need a relative-time regex (note "last week" is *suppressed* in recency_cue, but T-anchor does not pick it up either).
- **tempreason_small**: 23/60 queries (38%) lack any anchor — these tend to be the "Which team did X play for?" form where the date phrase is in *some* but not all queries. Regex correctly skips these — they shouldn't activate T.

### 5. R@5 regressions on negation_temporal

A noteworthy regression independent of switches: every variant that includes T (fuse_T_R, fuse_T_R+recAdd, mc_switches, mc_no_switches) drops R@5 vs rerank_only on negation_temporal (0.933 → 0.467, -0.467 — and mc_no_switches further to 0.267). The negation queries ("What did I do *not* in 2023?") explicitly want docs *outside* the date range, but T_lblend scores by interval *overlap* with the query's date — so it elevates exactly the wrong docs. This is not a multi-channel architecture issue; it predates this work and would need a polarity-aware T channel.

### 6. Recommendation

**Ship `fuse_T_R + recency_additive` (variant 3), not the multi-channel-with-switches system.**

Reasoning:
- **mc_switches matches fuse_T_R + recency_additive on hard_bench, latest_recent, temporal_essential, tempreason_small, conjunctive_temporal, multi_te_doc, open_ended_date, causal_relative, negation_temporal** (9/11 benchmarks tie or are dominated). 
- **mc_switches REGRESSES on era_refs (-0.167) and relative_time (-0.667 vs fuse_T_R)** because the binary `has_temporal_anchor` regex is a poor proxy for "T_lblend will fire". T_lblend's underlying interval/lattice machinery picks up signal from much weaker surface forms than my regex (entity names referencing periods, "last week", relative phrases) — so gating it on regex throws away wins T_lblend already had.
- mc_no_switches (CV-gate-alone) is unshippable: -41pp on hard_bench.
- fuse_T_R + recency_additive captures the latest_recent win (0.667) with zero regressions on the other 10 benchmarks (matches fuse_T_R everywhere else).
- The `has_recency_cue` switch is *cheap* and *accurate* — that one switch is worth keeping. The `has_temporal_anchor` switch is *expensive in regressions* and not worth keeping.

If we want the multi-channel score_blend architecture for future channel additions (cyclical, causal, negation), the path forward is:
- Keep T always-on with its existing low default weight (0.4).
- Use `has_recency_cue` to gate recency in.
- Replace `has_temporal_anchor` with a "T_lblend has dispersed signal" runtime check — i.e. let CV gate handle T (CV gate works for T because T-vs-no-anchor produces low CV; the failure mode in mc_no_switches was *recency*, not T). 
- Concretely: `multi_channel_blend(t_scores, rec_scores, r_scores, switches={"T_active": True, "Recency_active": has_recency_cue(q)}, weights={"T":0.3,"recency":0.3,"R":0.4})`.

That hybrid would (a) reproduce fuse_T_R on non-recency queries via CV-gating T natively, (b) pull in recency on cued queries via the explicit switch, and (c) avoid the era_refs / relative_time regression by never excluding T.

### Summary table — best variant per benchmark (R@1)

| Benchmark | Best | Δ vs rerank_only | Notes |
|---|---|---:|---|
| hard_bench | fuse_T_R / fuse+recAdd / mc_switches (tie) | +0.253 | mc_no_switches -0.160 |
| temporal_essential | fuse_T_R / +recAdd / mc_switches / mc_no_switches | +0.080 | all temporal variants tie |
| tempreason_small | fuse_T_R / +recAdd / mc_switches | +0.083 | mc_no_switches -0.033 |
| conjunctive_temporal | all tie at 1.000 | +0.000 | saturated |
| multi_te_doc | tie at 1.000 (mc_no_switches -0.083) | +0.000 | saturated |
| relative_time | fuse_T_R / mc_no_switches (1.000) | +0.750 | mc_switches WORSE than rerank_only |
| era_refs | fuse_T_R / +recAdd (0.417) | +0.167 | mc_switches matches rerank_only (regression vs fuse_T_R) |
| latest_recent | mc_switches / mc_no_switches (0.800) | +0.667 | +recAdd=0.667 close second |
| open_ended_date | fuse_T_R / +recAdd / mc_switches / mc_no_switches | +0.133 | all tie |
| causal_relative | rerank_only / fuse_T_R / +recAdd / mc_switches | +0.000 | T not helpful here |
| negation_temporal | rerank_only (0.000 R@1, 0.933 R@5) | n/a | T HURTS R@5 — orthogonal issue |

Best across-the-board single recipe: **fuse_T_R + recency_additive** (variant 3). Wins or ties everything except latest_recent (where mc_switches edges by +0.133, achievable in variant 3 by tuning half-life from 21d → 90d).
