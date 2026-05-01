❯ outline the architecture as it is now?

⏺ Current architecture

  Single LLM call → single scoring kernel → ranking.

  1. Query planner (query_planner_v2.py)

  One gpt-5-mini call per query (cached) emits a structured plan:

  @dataclass
  class Constraint:
      phrase: str                      # date text from query, verbatim
      direction: "in" | "after" | "before" | "not_in"

  @dataclass
  class QueryPlan:
      constraints: list[Constraint]
      extremum: "latest" | "earliest" | None

  JSON schema enforced via Responses API text.format = {type: json_schema, strict: true}. Zero parse failures across 293 queries.

  Prompt skips event/entity names (the launch, George Washington), keeps possessive-ordinals like "his first dinner party" out of
  extremum.

  2. Per-constraint anchor extraction

  Each constraint's phrase runs through the v2 temporal extractor → list of FuzzyIntervals. Phrases that don't resolve to dates yield
   empty intervals → that constraint becomes a no-op.

  3. Doc-side preprocessing (per benchmark)

  - v2 temporal extractor on every doc → per-doc TE list and flat intervals
  - T_lblend lattice index built over doc TEs
  - Cross-encoder reranks the union of (top-50 semantic, top-50 T_lblend, top-50 T_v5)

  4. Scoring kernel (composition_eval_v3.py)

  Per query, single multiplicative regime:

  T_router = T_v5 if any open constraint else T_lblend
  base     = score_blend({T_router, R}, weights, CV-gate)

  # Constraint masks
  include_factor = max(constraint_factor(doc, c) for c in includes) or 1.0
  exclude_factor = ∏ (1 − excluded_containment(doc, c)) for c in excludes
  mask           = include_factor × exclude_factor

  # Candidate set for within-set rec_lin normalization
  in_set = include-mask survivors          if any window
         = top-15 by base                  otherwise

  rec_lin = linear_recency_scores(in_set)  # [0..1] across in_set

  # Final
  score = base × mask × (1 + α·rec_lin)    # α=3 if extremum=latest
                                            # α=3 over (1−rec_lin) if extremum=earliest
                                            # α=0 if no extremum

  Constraint factor:
  - in: doc TE overlaps anchor
  - after: doc TE has any time strictly past anchor.latest
  - before: doc TE has any time strictly before anchor.earliest
  - not_in: contributes via excluded_containment to exclude_factor

  5. Ranking

  rank_from_scores(scores) — out-of-mask docs collapse to 0; in-mask docs sort by score.

  What's not here (architecturally dropped from the old stack)

  - Causal stack (event-name resolution → cosine top-1 doc → directional filter) — multi-hop, not temporal retrieval
  - normalize_plan post-processor for the absolute_anchor↔open_ended bug (impossible by construction now)
  - Era extractor fallback (era_refs uses the same path as everything else)
  - Cyclical scoring path (no benchmark exercises it; unscored)
  - recency_intent / earliest_intent as separate fields → folded into extremum

  Files

  - query_planner_v2.py — LLM planner + JSON schema + cache
  - composition_eval_v3.py — eval driver with the scoring kernel above

  Per-query cost

  - Plan: 1 LLM call, cached → ~$0.0004 cold, $0 warm
  - Extraction: N v2 calls per query (one per constraint phrase), cached
  - Rerank: cross-encoder over union of top-50 from 3 channels

  Macro on the 12-bench suite

  - 0.757 overall
  - 0.783 excluding causal_relative (architecturally dropped)
  - Beats old llm_planner_stack by +0.010 on the apples-to-apples set

✻ Sautéed for 37s

❯ How well does it perform against semantic/reranker only?

⏺ planner_v2 vs cross-encoder-rerank-only baseline

  The rerank_only column from the same eval = semantic top-50 ∪ T_lblend top-50 ∪ T_v5 top-50, fed into the cross-encoder reranker,
  then tail-padded by semantic. So it's "best non-planner baseline" — the temporal channels still inform retrieval, but no temporal
  scoring is applied.

  ┌──────────────────────┬─────┬─────────────┬────────────┬──────────────┐
  │        Bench         │  n  │ rerank_only │ planner_v2 │      Δ       │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ composition          │  25 │       0.040 │      0.280 │       +0.240 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ hard_bench           │  75 │       0.640 │      1.000 │       +0.360 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ temporal_essential   │  25 │       0.920 │      1.000 │       +0.080 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ tempreason_small     │  60 │       0.650 │      0.733 │       +0.083 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ conjunctive_temporal │  12 │       1.000 │      1.000 │            0 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ multi_te_doc         │  12 │       1.000 │      1.000 │            0 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ relative_time        │  12 │       0.250 │      0.917 │       +0.667 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ era_refs             │  12 │       0.250 │      0.417 │       +0.167 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ open_ended_date      │  15 │       0.267 │      0.733 │       +0.466 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ causal_relative      │  15 │       0.467 │      0.467 │            0 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ latest_recent        │  15 │       0.133 │      0.800 │       +0.667 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ negation_temporal    │  15 │       0.000 │      0.733 │       +0.733 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ Macro                │     │       0.476 │      0.757 │       +0.281 │
  ├──────────────────────┼─────┼─────────────┼────────────┼──────────────┤
  │ Total R@1            │ 293 │         132 │        232 │ +100 queries │
  └──────────────────────┴─────┴─────────────┴────────────┴──────────────┘

  Where the planner adds the most

  - negation_temporal +0.733 — the cross-encoder has no way to handle "not in 2024", just sees the words and matches docs that
  mention them. The constraint mask does the right thing.
  - latest_recent +0.667 and relative_time +0.667 — without recency-aware scoring, the cross-encoder optimizes topic match only.
  - open_ended_date +0.466 — the cross-encoder can't filter "after 2020" precisely; it embeds "after 2020" as semantic context but
  doesn't enforce the bound.
  - hard_bench +0.360 — date-anchored queries on a 600-doc corpus where the cross-encoder gets distracted by topic-matching
  distractors at wrong dates.
  - composition +0.240 — multi-cue queries where every cue helps narrow the set; cross-encoder sees them all as soft hints.

  Where it doesn't help

  - conjunctive_temporal, multi_te_doc — cross-encoder already saturates at 1.000.
  - causal_relative — by design (causal layer architecturally dropped). The constraint extractor skips event names, so these queries
  fall through to base.

  TL;DR

  Adding the planner over a cross-encoder-rerank-only stack:
  - 12-bench macro: 0.476 → 0.757
  - 100 more queries hit R@1 (out of 293 total)
  - One extra LLM call per unique query (cached after first call), one v2-extractor call per constraint phrase
