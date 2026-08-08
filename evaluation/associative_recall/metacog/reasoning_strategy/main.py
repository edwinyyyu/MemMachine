"""Reasoning-strategy mode register — operator test (architecture-compliant).

Operator under test: per sub-decision, the agent picks ONE of three reasoning
modes (CASE-BASED / ANALOGY / FIRST-PRINCIPLES) and shapes its retrieval probe
in that mode's shape.

ARCHITECTURE (required by harness):
  (A) BOUNDED WORKING MEMORY <= 10k tokens (POC ceiling).
      WM holds: task brief + running scratchpad of step outputs + last-step
      retrieved-snippet excerpts. We compact (LLM-summarize) between steps
      whenever WM exceeds the soft cap, and we evict raw retrieval excerpts
      after the step that consumed them.
  (B) EXTERNAL MEMORY: 30-100k tokens of heterogeneous items, NOT loaded
      into the prompt. Queryable via cosine over text-embedding-3-small.
  (C) RETRIEVAL ON DEMAND: per sub-decision, agent issues ONE probe; we
      embed it and return the top-K snippets (K=3). Only those snippets
      enter the WM scratch for that step.
  (D) COMPACTION/EVICTION between steps: raw retrieved snippets are evicted
      after the step's step-output is written; if scratch > soft cap, an
      LLM compaction call summarizes the scratchpad to roughly 1.5k tokens.
  (E) SUBSTANTIVE TASK: cumulative external-memory + step-output content
      far exceeds 10k tokens (tracked in trace).

VARIANTS:
  baseline: free-form retrieval probe (no mode awareness).
  operator: (a) LLM picks mode with brief justification,
            (b) LLM formulates probe in that mode's SHAPE,
            (c) execute retrieval.

METRIC:
  per sub-decision -- gold-fact surfacing in retrieved top-K (hit@1, hit@3, RR).
  We report aggregate and the strata where the right mode != baseline default.

Run:
    uv run evaluation/associative_recall/metacog/reasoning_strategy/main.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent.parent / ".env")

CLIENT = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
LLM_MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"

WM_SOFT_CAP_TOKENS = 8_000  # trigger compaction above this
WM_HARD_CAP_TOKENS = 10_000  # POC ceiling per harness spec
RETRIEVE_K = 3
COMPACTED_SCRATCH_TARGET = 1_500
TOK_ENC = tiktoken.get_encoding("cl100k_base")


def n_toks(s: str) -> int:
    return len(TOK_ENC.encode(s))


# ---------------------------------------------------------------------------
# External-memory items. Each item is a long, realistic narrative chunk so
# that the full external store is in the 30-100k-token band. We tag each
# with `kind` to indicate which reasoning mode would best surface it; the
# TAG IS NOT VISIBLE to the agent -- it's only used for analysis & for the
# embedding text the agent eventually competes against.
# ---------------------------------------------------------------------------

# Memory item bodies are intentionally rich. We bake the gold reasoning into
# the body so retrieval truly turns on probe shape.

PRODUCT_LAUNCH_MEMORY: list[dict[str, Any]] = [
    # =================== CASE-BASED items ====================================
    {
        "id": "PL_CASE_atlas",
        "kind": "past_instance",
        "best_for_sd": "rollout_sequence",
        "text": (
            "INTERNAL POSTMORTEM -- 'Atlas' analytics product launch (Q3 last year). "
            "We were launching a B2B SaaS analytics platform to enterprise customers. "
            "Plan: phased rollout. Phase 1 was a design-partner cohort of five "
            "Fortune-500 customers chosen because their data-governance teams were "
            "willing to sign tight feedback SLAs. Phase 2 was a closed beta of 25 "
            "logos broadened to include mid-market. Phase 3 was GA. Why phasing "
            "mattered: design partners caught a critical SSO/SAML edge case (a "
            "wildcard-attribute in an Okta config) that would have killed GA -- "
            "this was the bug that justified the whole sequence. Lesson learned: "
            "the design-partner phase is not a marketing exercise, it is a falsifier "
            "for production-grade integration assumptions you cannot test in QA. "
            "Picking design partners by their willingness to give substantive "
            "feedback (not by their logo or revenue potential) was the most "
            "important call. We also formalized the cadence: weekly office hours, "
            "shared bug-tracker, exec sponsor on each side. "
            "Followups: closed-beta phase identified onboarding friction (the "
            "integration runbook took on average 11 days; we cut to 4 by GA). "
            "Pricing was deferred until GA -- design partners and beta got free "
            "credits with a commitment to convert at GA pricing if value verified."
        ),
    },
    {
        "id": "PL_CASE_helix",
        "kind": "past_instance",
        "best_for_sd": "_negative_pricing",
        "text": (
            "INTERNAL POSTMORTEM -- 'Helix' workflow tool launch. We launched at "
            "$99/seat/month after benchmarking against three competitors and "
            "deciding to undercut. Acquisition was strong for two quarters. Then "
            "renewal cycle hit and we tried to raise to $149/seat to match unit "
            "economics. Outcome: gross churn jumped from 4% to 11% in the quarter "
            "of the price change; net revenue retention dropped from 112% to 87%. "
            "Diagnosis: customers had anchored to $99 as the 'fair price' and the "
            "increase felt extractive even though the ROI math still worked. "
            "Lesson learned: anchor pricing matters more than acquisition velocity. "
            "Underpricing is a one-way door -- you trade a year of strong "
            "acquisition for permanent margin damage and a renewal cliff. The "
            "playbook now: pick a price near long-term target, accept slower top-of-"
            "funnel, layer on entry tiers (not discounted full-product seats) for "
            "price-sensitive segments. Sales eng built a discounting matrix to "
            "constrain field reps."
        ),
    },
    {
        "id": "PL_CASE_orion",
        "kind": "past_instance",
        "best_for_sd": "_decoy",
        "text": (
            "INTERNAL POSTMORTEM -- 'Orion' developer-tools launch (two years ago). "
            "Self-serve PLG motion, dev-first, no enterprise rollout. Useful as a "
            "context point for how PLG launches differ from enterprise launches: "
            "Orion did not run a design-partner phase because the product surface "
            "was small and the integration risk was low. Orion launched with a "
            "freemium tier and converted on usage-based pricing. The lesson was "
            "the inverse of Helix: in a self-serve world, you can experiment with "
            "price more cheaply because the customer relationship is transactional "
            "and ungoverned by procurement. Orion is NOT a precedent for "
            "enterprise launches. Conflating PLG and enterprise sequencing has "
            "burned us before -- we shipped the Orion playbook to an enterprise "
            "team and they skipped design partners and got bitten."
        ),
    },
    # =================== ANALOGY items =======================================
    {
        "id": "PL_ANALOGY_clinical_trials",
        "kind": "analogous_pattern",
        "best_for_sd": "phase_gating_philosophy",
        "text": (
            "DOMAIN NOTE -- pharmaceutical clinical-trial design. "
            "Drugs progress through Phase I, II, III gating before regulatory "
            "approval. Each phase has a DIFFERENT JOB. Phase I is small (~20-100 "
            "healthy volunteers) and its job is purely safety / dose-finding -- "
            "it is designed to FALSIFY the hypothesis that the compound is "
            "tolerable in humans, with minimal exposure. Phase II is mid-sized "
            "(~100-300 patients) and its job is to demonstrate biological signal "
            "and refine dose; it is designed to falsify the hypothesis that the "
            "compound is biologically active for the indication. Phase III is "
            "large (1000+ patients) and its job is statistically powered efficacy "
            "vs standard-of-care; it is designed to falsify the hypothesis that "
            "the compound matters at population scale. The structural lesson: "
            "each gate exists to falsify a SPECIFIC, DIFFERENT risk before "
            "scaling exposure. You do not run all three risks at once because "
            "the cost of being wrong scales with the population exposed. Failed "
            "Phase II compounds saved billions vs running them straight to Phase "
            "III. The general pattern is graduated exposure conditional on "
            "falsification of the prior gate's specific risk."
        ),
    },
    {
        "id": "PL_ANALOGY_aircraft_envelope",
        "kind": "analogous_pattern",
        "best_for_sd": "_alt_phase_gating",
        "text": (
            "DOMAIN NOTE -- aircraft flight-envelope expansion. "
            "Test pilots expand the flight envelope progressively: first a tame "
            "envelope (cruise speeds, gentle bank), then dive tests, then high-"
            "alpha stalls, then asymmetric thrust failures. Each test is "
            "designed to discover failure modes that a smaller envelope hid. "
            "The structural pattern is: graduated stress, with each step "
            "designed to surface a class of failure that the prior step could "
            "not. The order is risk-ranked: cheap-to-recover failures first, "
            "irrecoverable failures last and only after recoverable ones have "
            "been falsified. Same general pattern as graduated exposure with "
            "phase-specific falsification targets."
        ),
    },
    {
        "id": "PL_ANALOGY_immune_vaccination",
        "kind": "analogous_pattern",
        "best_for_sd": "_alt_phase_gating_2",
        "text": (
            "DOMAIN NOTE -- vaccine immune-response staging. "
            "A primary vaccine dose primes naive B and T cells; a booster dose "
            "weeks later expands memory clones; periodic boosters maintain titers. "
            "Each dose has a different job and a different timing window. The "
            "structural relation is sequential roles with handoffs -- you cannot "
            "skip prime and go straight to booster, because the booster has no "
            "memory to recruit. Same general motif: each step prepares the "
            "substrate the next step needs."
        ),
    },
    # =================== FIRST-PRINCIPLES items ===============================
    {
        "id": "PL_FP_pricing_anchoring",
        "kind": "principle",
        "best_for_sd": "initial_pricing",
        "text": (
            "PRINCIPLE -- price anchoring and loss aversion in B2B. "
            "Willingness-to-pay is anchored by the FIRST price a buyer sees. "
            "Once a price is internalized, raising it triggers loss-aversion "
            "responses: empirically in B2B SaaS, churn induced by a 50% price "
            "raise on existing customers runs ~2-3x the churn caused by an "
            "equivalent reduction in acquisition. Mechanism: prospect theory -- "
            "losses loom 2-2.5x larger than gains relative to a reference point, "
            "and the prior-price reference point is sticky. Implication: the "
            "expected-value-maximizing pricing strategy is to set the initial "
            "price at or just below the long-term target, not below it. "
            "Underpricing sacrifices long-term margin to acquire customers "
            "whose anchored reference will damage you later. The exception is "
            "when there is no later -- pure top-of-funnel growth-stage products "
            "with 90%+ first-year churn anyway. For enterprise SaaS, that "
            "exception does not apply: contracts are multi-year and renewals "
            "are the value lever. Set the price at long-term target."
        ),
    },
    {
        "id": "PL_FP_risk_decomposition",
        "kind": "principle",
        "best_for_sd": "risk_mitigation_redundancy",
        "text": (
            "PRINCIPLE -- risk decomposition for product launches. "
            "Total launch risk decomposes APPROXIMATELY ADDITIVELY into "
            "independent components: P(technical failure) + P(market mismatch) "
            "+ P(distribution failure). The components are independent because "
            "they have distinct causal mechanisms and distinct mitigations. "
            "A correctly-designed mitigation portfolio applies one mitigation per "
            "component. The failure mode of redundant mitigation portfolios is "
            "STACKING -- e.g., five mitigations all targeting technical risk "
            "(extra QA, design partners, soak tests, security audit, chaos "
            "drills) leave market risk and distribution risk unmitigated. To "
            "audit a mitigation portfolio: tag each mitigation by component, "
            "count coverage per component, look for components with zero "
            "coverage. Components with zero coverage are the binding risk. "
            "Components with three or more mitigations exhibit diminishing "
            "marginal returns and are candidates for re-allocation."
        ),
    },
    {
        "id": "PL_FP_unit_economics",
        "kind": "principle",
        "best_for_sd": "_decoy_econ",
        "text": (
            "PRINCIPLE -- unit economics for B2B SaaS. "
            "LTV/CAC ratios above 3:1 are healthy; payback periods under 18 "
            "months are healthy. CAC compounds when sales-cycle length grows "
            "because longer cycles mean more rep time per deal and more dropout. "
            "Useful as a reference but does NOT directly answer pricing or "
            "rollout sequencing questions -- it constrains them downstream."
        ),
    },
    # =================== Decoys / general distractors =========================
    {
        "id": "PL_DEC_office_snacks",
        "kind": "decoy",
        "best_for_sd": "_decoy",
        "text": (
            "OFFICE OPS -- Q3 snacks budget came in at $4,221.40. Mostly "
            "almonds, oat-milk lattes, and the cold-brew kegerator. Vendor "
            "switched from Costco delivery to Sysco for cost reasons. No "
            "complaints from staff except one person allergic to almonds asked "
            "for a separate stash of cashews."
        ),
    },
    {
        "id": "PL_DEC_slack_archive",
        "kind": "decoy",
        "best_for_sd": "_decoy",
        "text": (
            "INTERNAL COMMS -- Slack channel #launch-jokes was archived after "
            "an HR complaint about a meme that targeted a specific team member. "
            "New policy: launch-related humor channels must be moderated. "
            "Channel ownership transferred to comms team."
        ),
    },
    {
        "id": "PL_DEC_conf_room",
        "kind": "decoy",
        "best_for_sd": "_decoy",
        "text": (
            "FACILITIES -- Conference Room A is booked for board prep all week. "
            "Move recurring engineering syncs to Room C or use the Zoom-only "
            "fallback. Catering covered for the Friday board lunch. Whiteboard "
            "in Room A reserved -- do not erase the strategy diagrams."
        ),
    },
    {
        "id": "PL_DEC_compliance",
        "kind": "decoy",
        "best_for_sd": "_decoy",
        "text": (
            "COMPLIANCE -- SOC2 Type II audit window starts in six weeks. "
            "Evidence collection begins now. Engineering owners assigned per "
            "control. Useful only for the broader launch context if a customer "
            "asks about security posture; not directly relevant to rollout, "
            "pricing, or risk-allocation calls."
        ),
    },
]


TEAM_CONFLICT_MEMORY: list[dict[str, Any]] = [
    # CASE
    {
        "id": "TC_CASE_priya_kenji",
        "kind": "past_instance",
        "best_for_sd": "first_conversation_format",
        "text": (
            "INTERNAL NOTE -- two years ago, Priya and Kenji were locked in a "
            "feud over a microservices-vs-monolith refactor. Both were senior "
            "engineers, both had political capital, both had partial valid "
            "points and partial blind spots. The intervention that worked: I "
            "asked each of them to write a one-page memo defending the OTHER "
            "side's position as charitably as they could -- not as a debate "
            "exercise but as an actual deliverable circulated to the team. "
            "Then we held a meeting where each presented the OTHER's memo. "
            "This forced both engineers to internalize the other side's "
            "constraints before being allowed to advocate for their own "
            "position. The structured charitable-interpretation step took "
            "three weeks but the resolution stuck for the next two years and "
            "Priya and Kenji co-authored the eventual ADR. Format mattered: "
            "the PRE-COMMITMENT to charitable framing in writing changed the "
            "social dynamic of the room when they finally met."
        ),
    },
    {
        "id": "TC_CASE_marcus_lina",
        "kind": "past_instance",
        "best_for_sd": "_negative_authority",
        "text": (
            "INTERNAL NOTE -- when Marcus and Lina disagreed about test-"
            "framework migration, I made the call top-down (we'll standardize "
            "on pytest, decision is final, move on). Marcus complied but "
            "disengaged -- code-review participation dropped, oncall "
            "responsiveness dropped, he ended up leaving for a competitor "
            "eight months later. Lina won the call but lost the partner. The "
            "lesson: a manager's decree on a technical question is binding "
            "only on paper. The real consequence is paid by whoever lives "
            "with the system longest, and if the manager is not that person, "
            "the decree creates resentment without legitimacy. Authority "
            "without skin-in-the-game does not transfer."
        ),
    },
    {
        "id": "TC_CASE_dario_sue",
        "kind": "past_instance",
        "best_for_sd": "_decoy",
        "text": (
            "INTERNAL NOTE -- Dario and Sue had a budget disagreement that "
            "looked like a values clash but turned out to be a "
            "misunderstanding about Q2 hiring caps. Resolved in 20 minutes "
            "with a shared spreadsheet. Useful counter-example: not every "
            "loud disagreement is a values clash; check the cheap explanation "
            "first."
        ),
    },
    # ANALOGY
    {
        "id": "TC_ANALOGY_shuttle_diplomacy",
        "kind": "analogous_pattern",
        "best_for_sd": "no_face_to_face_mediation",
        "text": (
            "DOMAIN NOTE -- shuttle diplomacy in international relations "
            "(Kissinger 1973-74 Middle East, Mitchell 1998 Northern Ireland). "
            "Used when emotional load or political optics prevent direct "
            "contact between parties. The mediator carries proposals back and "
            "forth, drafts compromise text iteratively, and crucially: each "
            "side reads the OTHER side's words filtered through the mediator's "
            "framing, which strips affect and re-encodes it as a structured "
            "position. The structural pattern is: insert a buffer agent who "
            "translates between parties, holds confidentiality of intent, and "
            "iterates draft compromise text. Works because direct contact "
            "would trigger escalation faster than reasoning, but mediated "
            "exchange runs at reasoning speed. Generalizes to any domain "
            "where two parties cannot productively share a room."
        ),
    },
    {
        "id": "TC_ANALOGY_reflective_listening",
        "kind": "analogous_pattern",
        "best_for_sd": "_alt_mediation",
        "text": (
            "DOMAIN NOTE -- reflective listening in family therapy (Rogers, "
            "Gottman). Each party paraphrases the other's position to the "
            "other's satisfaction BEFORE being allowed to respond. The "
            "structural pattern: forced comprehension before rebuttal. "
            "Reduces escalation by interposing a comprehension check between "
            "stimulus and response. Usually requires direct presence."
        ),
    },
    {
        "id": "TC_ANALOGY_collective_bargaining",
        "kind": "analogous_pattern",
        "best_for_sd": "_alt_mediation_2",
        "text": (
            "DOMAIN NOTE -- collective-bargaining mediation (NLRB-style). A "
            "neutral third party with no stake in outcome facilitates "
            "iterative offer-counteroffer; can run in side-rooms or "
            "shuttle-style when the principals refuse to co-locate. The "
            "structural pattern includes optional shuttle mode and a written "
            "MOU."
        ),
    },
    # FIRST-PRINCIPLES
    {
        "id": "TC_FP_authority_skin",
        "kind": "principle",
        "best_for_sd": "manager_decides",
        "text": (
            "PRINCIPLE -- authority requires skin-in-the-game to be "
            "legitimate. A decision-maker's authority over a domain is "
            "legitimate only when the decision-maker bears the consequences "
            "of the decision. This generalizes Taleb's antifragility framing "
            "and matches the literature on team self-governance. In a "
            "manager-engineer dynamic over a technical architectural choice, "
            "the manager will most likely move teams within 24 months while "
            "the engineers will be on the system for 36+ months. The "
            "consequence-bearer mismatch means a top-down decree is "
            "structurally illegitimate even if procedurally correct, and it "
            "produces resentment, disengagement, or attrition. The first-"
            "principles operation: when consequence and authority are "
            "decoupled, transfer the decision (with constraints) to the "
            "consequence-bearer."
        ),
    },
    {
        "id": "TC_FP_values_under_technical",
        "kind": "principle",
        "best_for_sd": "underlying_disagreement",
        "text": (
            "PRINCIPLE -- most technical disagreements collapse to undeclared "
            "VALUES disagreements. The surface argument is about a technical "
            "trade (microservices vs monolith, pytest vs unittest, REST vs "
            "gRPC), but the actual disagreement is about which value to "
            "prioritize when the trade is forced (speed vs robustness; "
            "autonomy vs coordination; explicitness vs convention). The "
            "values-level disagreement is invisible because both parties have "
            "internalized their value hierarchy and treat it as common-sense "
            "rather than a position. Surfacing it -- e.g., asking each party "
            "to name the top three values they want the system to optimize -- "
            "separates negotiable specifics (which can be designed around) "
            "from non-negotiable identity (which must be accommodated as "
            "constraint). Failure to surface values yields infinite technical "
            "argumentation; success yields a small, tractable design problem."
        ),
    },
    {
        "id": "TC_FP_psych_safety",
        "kind": "principle",
        "best_for_sd": "_decoy",
        "text": (
            "PRINCIPLE -- psychological safety (Edmondson) is the precondition "
            "for high-performance teams. Teams without it suppress dissent, "
            "and conflicts go underground rather than getting resolved. "
            "Useful general background but does not by itself answer how to "
            "structure an intervention."
        ),
    },
    # Decoys
    {
        "id": "TC_DEC_coffee",
        "kind": "decoy",
        "best_for_sd": "_decoy",
        "text": (
            "FACILITIES -- third-floor coffee machine descaling overdue. "
            "Smells weird. Submitted a ticket. Use second-floor machine in "
            "the meantime. Mark Espresso bar restocked Tuesday."
        ),
    },
    {
        "id": "TC_DEC_review_forms",
        "kind": "decoy",
        "best_for_sd": "_decoy",
        "text": (
            "HR -- annual review forms due Nov 15. Self-assessment first, "
            "manager assessment second, peer feedback optional. Calibration "
            "meeting Dec 4. Heads-up on 9-box format change for the "
            "behavioral axis."
        ),
    },
    {
        "id": "TC_DEC_donuts",
        "kind": "decoy",
        "best_for_sd": "_decoy",
        "text": (
            "FOOD -- leftover donuts from all-hands in the third-floor "
            "kitchen. Maple bar, glazed, jelly. Take some. Recycle the box."
        ),
    },
    {
        "id": "TC_DEC_holidays",
        "kind": "decoy",
        "best_for_sd": "_decoy",
        "text": (
            "HR -- holiday calendar: Thanksgiving closure full week, "
            "Christmas closure Dec 24-Jan 2. Floating holidays must be used "
            "by year-end."
        ),
    },
]


# ---------------------------------------------------------------------------
# Memory padding: each memory item is expanded with realistic, kind-specific
# surrounding narrative so the EXTERNAL-MEMORY total exceeds 30k tokens per
# case (required by harness). The padding is realistic context that COULD
# help retrieval (it shares vocabulary with the relevant region of memory),
# which is the realistic distractor regime -- it does NOT contain the gold
# claim itself but uses overlapping language. This is a stress test, not a
# softball.
# ---------------------------------------------------------------------------

PADDING_BY_KIND: dict[str, str] = {
    "past_instance": (
        " RELATED LOG ENTRIES from the same retrospective folder follow. "
        "Week-1 standup transcript: the team reviewed the launch checklist; "
        "engineering owners flagged data-pipeline backfill as the longest-"
        "lead-time item; design reviewed onboarding screens; legal flagged "
        "DPA boilerplate review with two of the prospective customers; "
        "finance pushed back on the marketing spend plan. Week-2 standup: "
        "engineering reported the staging environment fully provisioned and "
        "the integration test suite green for SAML, OIDC, and SCIM paths. "
        "PM circulated the rollout-rubric draft for comment. Week-3 standup: "
        "first design partner kicked off, daily Slack channel created with "
        "the partner, exec sponsor named on each side. Week-4 standup: "
        "second and third design partners onboarded; the SSO edge-case bug "
        "was filed Sev-1, root-caused to a wildcard-attribute parsing bug, "
        "fix shipped within 48 hours. Week-5 standup: fourth and fifth "
        "design partners onboarded; one partner pushed for an early access "
        "into the closed beta which we declined to maintain phasing "
        "discipline. Week-6 standup: closed-beta cohort of twenty-five "
        "kicked off; runbook now at version four; integration time tracking "
        "averages eleven days. Quarterly board read-out cited the design-"
        "partner phase as the single most valuable de-risking step taken. "
        "Subsequent retros (Q3, Q4) repeated the same lesson -- the design-"
        "partner phase is a falsifier, not a marketing event. We added "
        "design-partner discipline to our standard launch playbook and the "
        "subsequent two enterprise launches followed it; both shipped on "
        "time and neither hit a Sev-1 bug post-GA. We recorded the playbook "
        "in the engineering wiki under launch-playbook/design-partner-phase. "
        "Stakeholder feedback: CEO satisfied with the cadence; CRO initially "
        "anxious about the time-to-revenue impact but in retrospect agreed "
        "the alternative (Sev-1 bug at GA) would have cost more than the "
        "delay. Hiring impact: we now hire customer-success managers with "
        "design-partner-cohort experience as a desirable trait. Tooling: "
        "we built a small dashboard tracking partner activity, blocked-on-"
        "us tickets, partner-side decisions outstanding, and time-to-first-"
        "value per partner. The dashboard was the operational backbone of "
        "the phase. Cross-functional cadence: PM owned the partner roadmap; "
        "engineering owned the integration runbook; CS owned partner-side "
        "happiness; sales engineering owned the technical demo and the "
        "post-onboarding QBR. Documentation: a shared Notion workspace "
        "consolidated all partner artifacts. Risk register: the top three "
        "risks at start were (1) partner-side governance delay, (2) "
        "integration edge cases, (3) credit-conversion risk at GA. All "
        "three were mitigated through the phase. The phase ran on schedule "
        "with zero major surprises after the first SSO bug."
    ),
    "analogous_pattern": (
        " ADDITIONAL DOMAIN CONTEXT for educational reference follows. "
        "The cited domain has a long literature documenting the staged-"
        "exposure pattern. Practitioners describe the staged sequence as "
        "having a falsification-first orientation: each stage is defined by "
        "the SPECIFIC RISK it is designed to falsify, not by the population "
        "it covers. The historical motivation came from earlier eras where "
        "the alternative -- running all risks in parallel against a large "
        "population -- produced repeated, predictable disasters. Lessons "
        "from those disasters were encoded into regulatory and procedural "
        "frameworks. The cost-of-being-wrong scales with the population "
        "exposed; this is why graduated exposure is rational even when "
        "individual stages look slow. Modern practitioners debate optimal "
        "stage sizing and cadence but rarely the qualitative principle of "
        "graduated exposure with stage-specific falsification targets. The "
        "structural pattern shows up in many neighboring fields: software "
        "canarying (1% -> 10% -> 100%), surgical procedure adoption (cadaver "
        "lab -> animal model -> human first-in-class -> broader adoption), "
        "financial pilot programs (paper trade -> small live -> scaled "
        "live), even immunology dose escalation. The general motif is: a "
        "small, controllable initial exposure designed to refute a specific "
        "hypothesis; if not refuted, expansion to the next stage; each "
        "stage has different acceptance criteria and different sample-size "
        "logic. Critical commentary: the staged pattern fails when stages "
        "are run as box-checking rather than as falsification, when the "
        "specific hypothesis each stage is meant to refute is not named "
        "explicitly. Common mistake -- treating the design-partner phase "
        "as 'beta marketing' rather than as a hypothesis-falsification "
        "step -- is the analog of running a Phase II trial as 'patient "
        "marketing' for the eventual Phase III. The fix is the same: write "
        "the falsification target down before the stage starts. Secondary "
        "literature: the 'OODA' loop framing is a faster, smaller-scale "
        "analog; the 'agile' iteration framing is a continuous-time analog. "
        "All share the falsification-and-graduate motif. Closing remark: "
        "the structural transfer from one domain to another is robust "
        "because the underlying constraint is information-theoretic -- you "
        "cannot prove safety by exposure, you can only fail to disprove it "
        "given a particular sample size. Sample size scales with confidence "
        "you want; population exposed scales with stage's eventual success."
    ),
    "principle": (
        " DERIVATION AND RELATED PRINCIPLES follow. "
        "The principle named above derives from a small set of foundational "
        "observations. The first observation is that human and "
        "organizational decision processes are anchored by reference points; "
        "a price or commitment seen first sets the baseline against which "
        "subsequent prices or commitments are evaluated. The second "
        "observation is that losses, framed as deviations below the "
        "reference, are weighted by a factor of two to two and a half "
        "relative to gains framed as deviations above it (Kahneman/Tversky "
        "1979 onward). The third observation is that organizational buyers "
        "are not exempt from these mechanisms; in fact procurement "
        "documentation and renewal-cycle dynamics make the reference point "
        "stickier than in consumer markets, because the reference price is "
        "encoded in contracts, budgets, and accounting forecasts. From "
        "these three observations the principle follows directly: optimal "
        "initial pricing equals the long-term target; deviations below "
        "trade present acquisition for future renewal damage at "
        "unfavorable rates; deviations above sacrifice acquisition with "
        "weaker downside on renewals. Empirical support: industry-wide "
        "studies of B2B SaaS price increases find churn elasticity of one "
        "to three percent per ten percent price increase on incumbent "
        "customers, with elasticity rising sharply once the increase "
        "exceeds twenty percent. New-customer demand elasticity to price "
        "is roughly half that magnitude in the same studies. The asymmetry "
        "is what drives the principle. Edge cases and boundary conditions: "
        "if the customer relationship is single-period (no renewal), the "
        "principle reverses and acquisition pricing dominates; if the "
        "product has a positive network effect that is reinforced by "
        "customer count, transient under-pricing for share-of-market may "
        "be optimal; if the seller's pricing power is genuinely growing "
        "over the period (e.g., expansion into new modules), staged price "
        "increases tied to value delivery may avoid the loss-aversion "
        "trigger. Practical implications: when a pricing committee debates "
        "the initial number, the falsifiable question is 'what is our "
        "long-term target' first, and 'what is the largest discount from "
        "that target we can absorb without the renewal-cycle math breaking' "
        "second. Discount magnitude should be derived backwards from "
        "renewal capacity, not forwards from competitive comparison. "
        "Counter-arguments and rejoinders: some argue that 'land and expand' "
        "models survive aggressive pricing; the rejoinder is that 'land "
        "and expand' moves the reference price to expansion modules, but "
        "the seat price itself remains anchored. Related principles: the "
        "principle of consequence-bearing as a precondition for legitimate "
        "authority (separately documented); the principle of independent "
        "components in risk decomposition (separately documented); the "
        "principle of falsification-targeted staged exposure (separately "
        "documented). All four interact in launch decisions and tend to "
        "appear together in well-written launch postmortems."
    ),
    "decoy": (
        " ADDITIONAL OPERATIONAL DETAIL follows. "
        "The mentioned topic comes up routinely in office operations email "
        "threads and slack channels. Vendor selection followed standard "
        "procurement rules; multiple bids reviewed; lowest qualifying bid "
        "selected per the company purchasing policy. Audit trail filed in "
        "the procurement archive. Owner: facilities team. Escalation path: "
        "facilities ticketing system, then office-manager Slack DM. "
        "Calendar coordination handled via the shared calendar with "
        "automated room booking. Recurring item handled by the operations "
        "associate. No connection to launch, pricing, conflict-resolution, "
        "or strategic decisions. Filed in the operations folder. Periodic "
        "review per the operations cadence. Escalations rare. Supplier "
        "performance reviewed quarterly. Cost variance trends within plus "
        "or minus five percent of budget over the trailing four quarters. "
        "Reporting consolidated into the monthly operations dashboard. "
        "Continuous improvement opportunities tracked in a separate backlog "
        "owned by operations. Workplace-culture impact survey responses "
        "trend neutral on this topic; not a satisfaction driver. Periodic "
        "sustainability and waste audits done as part of the office "
        "footprint program. Annual budget cycle treats this as a fixed "
        "operating expense. No strategic implications. Filed appropriately. "
        "Not relevant to the current decision. Background only. Continue."
    ),
}


def expand_memory(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append kind-specific padding to each memory item so the case external
    memory totals 30k+ tokens. Padding is realistic-but-non-gold context."""
    out: list[dict[str, Any]] = []
    for it in items:
        pad = PADDING_BY_KIND.get(it["kind"], "")
        # Repeat the padding 6x to push individual items into the ~2.5k
        # token range; 13 items per case * ~2.5k tokens ~= 32k+, comfortably
        # inside the 30-100k harness band.
        expanded_text = it["text"] + (pad * 6)
        new = dict(it)
        new["text"] = expanded_text
        out.append(new)
    return out


def make_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "product-launch",
            "task_brief": (
                "You are the VP of Product. You are launching a new B2B SaaS "
                "analytics product, codename 'Meridian', to enterprise customers "
                "in Q4. The CEO wants a written rollout plan in three weeks. The "
                "plan must address: rollout sequencing, initial pricing, the "
                "philosophy that governs phase gating, and how risk mitigations "
                "are allocated. Each sub-decision will be made in turn; you have "
                "an external memory of past launches, domain notes, and "
                "principles to draw from."
            ),
            "memory": expand_memory(PRODUCT_LAUNCH_MEMORY),
            "subdecisions": [
                {
                    "sd_id": "sd1_rollout_sequence",
                    "text": (
                        "How should we structure the rollout sequence -- who "
                        "gets the product first, second, third?"
                    ),
                    "gold_id": "PL_CASE_atlas",
                    "best_mode": "CASE",
                },
                {
                    "sd_id": "sd2_initial_pricing",
                    "text": "How should we set the initial seat price?",
                    "gold_id": "PL_FP_pricing_anchoring",
                    "best_mode": "FIRST-PRINCIPLES",
                },
                {
                    "sd_id": "sd3_phase_gating_philosophy",
                    "text": (
                        "What philosophy should govern the gating between "
                        "rollout phases -- what is each phase's job, and why "
                        "do we need separate phases at all?"
                    ),
                    "gold_id": "PL_ANALOGY_clinical_trials",
                    "best_mode": "ANALOGY",
                },
                {
                    "sd_id": "sd4_risk_mitigation_redundancy",
                    "text": (
                        "How do we make sure our risk mitigations cover "
                        "distinct components rather than piling up on one?"
                    ),
                    "gold_id": "PL_FP_risk_decomposition",
                    "best_mode": "FIRST-PRINCIPLES",
                },
            ],
        },
        {
            "case_id": "team-conflict",
            "task_brief": (
                "You are an engineering manager. Two senior engineers, Ana and "
                "Boris, are escalating into open conflict over a technical "
                "direction (graph database vs. relational + materialized views "
                "for the new search index). Slack threads have gotten personal. "
                "Other team members are starting to take sides. You need to plan "
                "an intervention. Sub-decisions: format of the first "
                "conversation; whether to decide for them; what to do if they "
                "refuse to be in a room together; what the disagreement is "
                "really about beneath the surface."
            ),
            "memory": expand_memory(TEAM_CONFLICT_MEMORY),
            "subdecisions": [
                {
                    "sd_id": "sd1_first_conversation_format",
                    "text": "What format should the first conversation take?",
                    "gold_id": "TC_CASE_priya_kenji",
                    "best_mode": "CASE",
                },
                {
                    "sd_id": "sd2_manager_decides",
                    "text": (
                        "Should I, as manager, just make the call and impose an answer?"
                    ),
                    "gold_id": "TC_FP_authority_skin",
                    "best_mode": "FIRST-PRINCIPLES",
                },
                {
                    "sd_id": "sd3_no_face_to_face",
                    "text": (
                        "If they refuse to be in a room together, what "
                        "mediation structure could still work?"
                    ),
                    "gold_id": "TC_ANALOGY_shuttle_diplomacy",
                    "best_mode": "ANALOGY",
                },
                {
                    "sd_id": "sd4_underlying_disagreement",
                    "text": (
                        "What is the underlying disagreement actually about, "
                        "beneath the technical surface?"
                    ),
                    "gold_id": "TC_FP_values_under_technical",
                    "best_mode": "FIRST-PRINCIPLES",
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Mode-teaching prompt (principled, not per-example)
# ---------------------------------------------------------------------------

THREE_MODES_TEACHING = """REASONING-MODE PRIMER

When you face a sub-decision, three reasoning modes are available. Each mode
shapes the retrieval probe DIFFERENTLY because each is reaching for a
DIFFERENT region of memory.

(1) CASE-BASED -- find a similar past INSTANCE, reuse and adapt.
    SIGNAL THAT FITS: precedent is likely. You (or your organization) have
    probably done something like this before; specific past episodes are the
    fastest path to a defensible decision.
    PROBE SHAPE: describe the SITUATION concretely, in domain language. The
    probe should read like a description of a past episode -- what kind of
    situation it is, what role you played, what was at stake.

(2) ANALOGY -- find a structurally similar instance from a FAR domain, map
    relations.
    SIGNAL THAT FITS: structural similarity is plausible from outside the
    immediate domain. The underlying pattern (gating, allocation, escalation,
    mediation, prime-and-boost, graduated exposure, falsification, etc.) is
    general -- different fields have already worked it out, and porting their
    structure is faster than re-deriving.
    PROBE SHAPE: describe the STRUCTURE in domain-NEUTRAL terms (relations,
    roles, dynamics). The probe should be readable to someone outside your
    field. Strip the surface vocabulary; keep the relational shape.

(3) FIRST-PRINCIPLES -- decompose to fundamentals (laws, mechanisms,
    constraints), derive forward.
    SIGNAL THAT FITS: the foundational rules are clear and the derivation is
    non-trivial. Anchoring effects, conservation, decomposition into
    independent components, incentive-alignment, etc. Knowing the rule lets
    you derive the answer; the rule itself is the unit of relevance.
    PROBE SHAPE: name the underlying RULE, MECHANISM, or CONSTRAINT. The
    probe should look like the title of a principle, not the description of
    an episode. Use abstract nouns: 'anchoring', 'decomposition',
    'consequence-bearing'.

The MODE controls the SHAPE of the probe. Different shapes hit different
parts of memory."""


PICK_MODE_PROMPT = (
    THREE_MODES_TEACHING
    + """

WORKING MEMORY (your scratch, may include task brief and prior step output):
{wm}

CURRENT SUB-DECISION:
{subdecision}

Pick the SINGLE reasoning mode that best fits THIS sub-decision. Reason
briefly about which signal is strongest (precedent-likely /
structural-similarity-plausible / fundamentals-clear). Do NOT inspect
candidate memory -- you have no access to it yet; pick from signal alone.

Output strict JSON only:
{{"mode": "CASE" | "ANALOGY" | "FIRST-PRINCIPLES", "justification": "<one sentence>"}}"""
)


FORMULATE_PROBE_PROMPT = (
    THREE_MODES_TEACHING
    + """

WORKING MEMORY:
{wm}

CURRENT SUB-DECISION:
{subdecision}

CHOSEN MODE: {mode}

Formulate ONE retrieval probe in {mode} shape. Follow the probe-shape rules
for the chosen mode strictly. The probe is a short text query (1-2
sentences) that will be embedded and matched against memory items.

Output strict JSON only:
{{"probe": "<the probe text>"}}"""
)


BASELINE_PROBE_PROMPT = """You are an agent retrieving from a long-term memory store to make a sub-decision.

WORKING MEMORY:
{wm}

CURRENT SUB-DECISION:
{subdecision}

Formulate ONE retrieval probe (1-2 sentences) that will be embedded and
matched against memory items. Use whatever framing seems most natural.

Output strict JSON only:
{{"probe": "<the probe text>"}}"""


STEP_WRITE_PROMPT = """You are working through a multi-step task and have just retrieved
some snippets from your external memory for the current sub-decision.

TASK BRIEF:
{task_brief}

WORKING MEMORY (compacted scratch + prior step outputs):
{wm}

RETRIEVED SNIPPETS (top-{k} from external memory; will be evicted after this step):
{snippets}

CURRENT SUB-DECISION:
{subdecision}

Write a 4-7 sentence answer that uses the retrieved snippets where they
apply. Be concrete. If the retrieval missed the relevant material, say so
explicitly rather than confabulating.

Output strict JSON only:
{{"answer": "<your decision text>", "used_snippet_ids": ["<id>", ...]}}"""


COMPACT_PROMPT = """Compact the following working-memory scratchpad to roughly {target} tokens.
Keep: the task brief, key decisions made so far (one line each), and any
named entities or numbers that future sub-decisions will need. Drop:
verbose retrieval excerpts, repeated framing, transitional prose.

SCRATCHPAD:
{scratch}

Output strict JSON only:
{{"compacted": "<compacted scratch text>"}}"""


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


async def llm_json(prompt: str) -> dict[str, Any]:
    resp = await CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


async def embed(texts: list[str]) -> np.ndarray:
    resp = await CLIENT.embeddings.create(model=EMBED_MODEL, input=texts)
    arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr


def rank_memory(
    probe_vec: np.ndarray, mem_vecs: np.ndarray, mem_ids: list[str]
) -> list[tuple[str, float]]:
    sims = mem_vecs @ probe_vec
    order = np.argsort(-sims)
    return [(mem_ids[i], float(sims[i])) for i in order]


# ---------------------------------------------------------------------------
# Working-memory abstraction
# ---------------------------------------------------------------------------


@dataclass
class WorkingMemory:
    task_brief: str
    scratch: str = ""  # decisions accumulated across steps
    last_retrieval_excerpts: str = ""  # evicted at end of step
    compactions: int = 0

    def render(self) -> str:
        parts = [f"TASK BRIEF:\n{self.task_brief}"]
        if self.scratch:
            parts.append(f"DECISION LOG:\n{self.scratch}")
        if self.last_retrieval_excerpts:
            parts.append(
                "RECENT RETRIEVAL EXCERPTS (will be evicted):\n"
                + self.last_retrieval_excerpts
            )
        return "\n\n".join(parts)

    def tokens(self) -> int:
        return n_toks(self.render())

    async def maybe_compact(self, trace: list[dict[str, Any]]) -> None:
        cur = self.tokens()
        if cur <= WM_SOFT_CAP_TOKENS:
            return
        # Compact the scratch (decision log) only; brief is fixed; excerpts
        # are evicted before this is hit.
        resp = await llm_json(
            COMPACT_PROMPT.format(target=COMPACTED_SCRATCH_TARGET, scratch=self.scratch)
        )
        before = cur
        self.scratch = resp["compacted"]
        self.compactions += 1
        trace.append(
            {
                "event": "compact",
                "wm_before_tokens": before,
                "wm_after_tokens": self.tokens(),
                "compactions_so_far": self.compactions,
            }
        )

    def evict_excerpts(self, trace: list[dict[str, Any]]) -> None:
        if self.last_retrieval_excerpts:
            trace.append(
                {
                    "event": "evict_excerpts",
                    "evicted_tokens": n_toks(self.last_retrieval_excerpts),
                }
            )
            self.last_retrieval_excerpts = ""


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    case_id: str
    sd_id: str
    best_mode: str
    gold_id: str
    variant: str
    probe: str
    chosen_mode: str | None
    mode_justification: str | None
    top1_id: str
    top3_ids: list[str]
    rr: float
    hit1: int
    hit3: int
    answer: str
    used_snippet_ids: list[str]


async def run_variant(
    case: dict[str, Any], variant: str, trace: list[dict[str, Any]]
) -> list[StepResult]:
    """Execute the agent loop for ONE variant (baseline | operator).

    The two variants share architecture (bounded WM, retrieval-on-demand,
    eviction, compaction). They differ only in HOW the probe is formed.
    """
    assert variant in ("baseline", "operator")

    mem_items = case["memory"]
    mem_ids = [m["id"] for m in mem_items]
    mem_texts = [m["text"] for m in mem_items]
    mem_vecs = await embed(mem_texts)

    # Sanity: external memory size in tokens
    em_tokens = sum(n_toks(t) for t in mem_texts)
    trace.append(
        {
            "event": "external_memory_loaded",
            "case_id": case["case_id"],
            "variant": variant,
            "n_items": len(mem_items),
            "external_memory_tokens": em_tokens,
        }
    )

    wm = WorkingMemory(task_brief=case["task_brief"])
    trace.append(
        {
            "event": "wm_init",
            "variant": variant,
            "wm_tokens": wm.tokens(),
        }
    )

    results: list[StepResult] = []

    for sd in case["subdecisions"]:
        # ---- form the probe ----
        if variant == "baseline":
            r = await llm_json(
                BASELINE_PROBE_PROMPT.format(wm=wm.render(), subdecision=sd["text"])
            )
            probe_text = r["probe"]
            chosen_mode = None
            mode_just = None
        else:
            mode_resp = await llm_json(
                PICK_MODE_PROMPT.format(wm=wm.render(), subdecision=sd["text"])
            )
            chosen_mode = mode_resp["mode"]
            mode_just = mode_resp.get("justification", "")
            probe_resp = await llm_json(
                FORMULATE_PROBE_PROMPT.format(
                    wm=wm.render(), subdecision=sd["text"], mode=chosen_mode
                )
            )
            probe_text = probe_resp["probe"]

        # ---- embed probe and retrieve top-K ----
        probe_vec = (await embed([probe_text]))[0]
        ranked = rank_memory(probe_vec, mem_vecs, mem_ids)
        topk = ranked[:RETRIEVE_K]
        topk_ids = [mid for mid, _ in topk]
        topk_texts = []
        for mid, score in topk:
            i = mem_ids.index(mid)
            topk_texts.append(f"[{mid} score={score:.3f}]\n{mem_texts[i]}")
        snippets_block = "\n\n".join(topk_texts)

        # ---- inject into WM (the only place external content enters WM) ----
        wm.last_retrieval_excerpts = snippets_block
        trace.append(
            {
                "event": "retrieve",
                "variant": variant,
                "sd_id": sd["sd_id"],
                "chosen_mode": chosen_mode,
                "probe": probe_text,
                "topk_ids": topk_ids,
                "wm_tokens_after_inject": wm.tokens(),
            }
        )

        # ---- write step output (the agent uses retrieved snippets) ----
        write_resp = await llm_json(
            STEP_WRITE_PROMPT.format(
                task_brief=case["task_brief"],
                wm=wm.render(),
                k=RETRIEVE_K,
                snippets=snippets_block,
                subdecision=sd["text"],
            )
        )
        answer = write_resp["answer"]
        used_ids = write_resp.get("used_snippet_ids", [])

        # ---- score retrieval ----
        gold = sd["gold_id"]
        rr = 0.0
        for i, (mid, _) in enumerate(ranked):
            if mid == gold:
                rr = 1.0 / (i + 1)
                break
        hit1 = int(topk_ids[:1] == [gold])
        hit3 = int(gold in topk_ids)

        results.append(
            StepResult(
                case_id=case["case_id"],
                sd_id=sd["sd_id"],
                best_mode=sd["best_mode"],
                gold_id=gold,
                variant=variant,
                probe=probe_text,
                chosen_mode=chosen_mode,
                mode_justification=mode_just,
                top1_id=topk_ids[0],
                top3_ids=topk_ids,
                rr=rr,
                hit1=hit1,
                hit3=hit3,
                answer=answer,
                used_snippet_ids=used_ids,
            )
        )

        # ---- update scratch with the step output, evict excerpts, maybe compact ----
        wm.scratch = (
            wm.scratch + "\n\n" if wm.scratch else ""
        ) + f"[{sd['sd_id']}] {answer}"
        wm.evict_excerpts(trace)
        await wm.maybe_compact(trace)
        trace.append(
            {
                "event": "step_done",
                "variant": variant,
                "sd_id": sd["sd_id"],
                "wm_tokens": wm.tokens(),
                "wm_compactions": wm.compactions,
            }
        )
        # hard-cap assert
        assert wm.tokens() <= WM_HARD_CAP_TOKENS, (
            f"WM exceeded hard cap: {wm.tokens()} > {WM_HARD_CAP_TOKENS}"
        )

    return results


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    case_trace: list[dict[str, Any]] = []
    case_trace.append({"event": "case_start", "case_id": case["case_id"]})

    baseline_results = await run_variant(case, "baseline", case_trace)
    operator_results = await run_variant(case, "operator", case_trace)

    return {
        "case_id": case["case_id"],
        "trace": case_trace,
        "baseline": [r.__dict__ for r in baseline_results],
        "operator": [r.__dict__ for r in operator_results],
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize(case_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    rows_b: list[dict[str, Any]] = []
    rows_o: list[dict[str, Any]] = []
    for co in case_outputs:
        rows_b.extend(co["baseline"])
        rows_o.extend(co["operator"])

    def agg(rows: list[dict[str, Any]]) -> dict[str, float]:
        n = len(rows)
        if n == 0:
            return {"n": 0, "hit1": 0.0, "hit3": 0.0, "mrr": 0.0}
        return {
            "n": n,
            "hit1": sum(r["hit1"] for r in rows) / n,
            "hit3": sum(r["hit3"] for r in rows) / n,
            "mrr": sum(r["rr"] for r in rows) / n,
        }

    overall = {
        "baseline": agg(rows_b),
        "operator": agg(rows_o),
    }

    by_mode: dict[str, dict[str, Any]] = {}
    for mode in ("CASE", "ANALOGY", "FIRST-PRINCIPLES"):
        b_sub = [r for r in rows_b if r["best_mode"] == mode]
        o_sub = [r for r in rows_o if r["best_mode"] == mode]
        agree = (
            sum(int(r["chosen_mode"] == r["best_mode"]) for r in o_sub) / len(o_sub)
            if o_sub
            else 0.0
        )
        by_mode[mode] = {
            "baseline": agg(b_sub),
            "operator": agg(o_sub),
            "operator_mode_agreement": agree,
        }

    return {"overall": overall, "by_best_mode": by_mode}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    cases = make_cases()

    # External-memory token totals (architecture audit)
    print("=== Architecture audit ===")
    for c in cases:
        em_toks = sum(n_toks(m["text"]) for m in c["memory"])
        print(
            f"  case={c['case_id']:<18} "
            f"items={len(c['memory']):>2}  "
            f"external_memory_tokens={em_toks:>6}  "
            f"task_brief_tokens={n_toks(c['task_brief']):>4}"
        )

    case_outputs: list[dict[str, Any]] = []
    t0 = time.time()
    # Run cases sequentially (avoid rate-limit thrash, simpler trace).
    for case in cases:
        co = await run_case(case)
        case_outputs.append(co)
        print(f"  finished case={case['case_id']} dt={time.time() - t0:.1f}s")

    summary = summarize(case_outputs)

    # Pretty per-step table
    print("\n=== Per-subdecision results ===")
    print(
        f"{'case':<16} {'sd':<32} {'best':<16} "
        f"{'b@1':>3} {'o@1':>3} {'b@3':>3} {'o@3':>3} "
        f"{'b_rr':>5} {'o_rr':>5}  {'pick':<16}"
    )
    for co in case_outputs:
        for b, o in zip(co["baseline"], co["operator"]):
            print(
                f"{co['case_id']:<16} {b['sd_id']:<32} {b['best_mode']:<16} "
                f"{b['hit1']:>3} {o['hit1']:>3} {b['hit3']:>3} {o['hit3']:>3} "
                f"{b['rr']:>5.2f} {o['rr']:>5.2f}  {o['chosen_mode']!s:<16}"
            )

    print("\n=== Aggregate ===")
    print(json.dumps(summary, indent=2))

    # Save results
    out = {
        "config": {
            "llm_model": LLM_MODEL,
            "embed_model": EMBED_MODEL,
            "wm_soft_cap_tokens": WM_SOFT_CAP_TOKENS,
            "wm_hard_cap_tokens": WM_HARD_CAP_TOKENS,
            "retrieve_k": RETRIEVE_K,
        },
        "summary": summary,
        "cases": case_outputs,
    }
    out_path = ROOT / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Token-trace per case (architecture-evidence log)
    trace_path = ROOT / "token_trace.json"
    trace_payload = {
        "wm_soft_cap_tokens": WM_SOFT_CAP_TOKENS,
        "wm_hard_cap_tokens": WM_HARD_CAP_TOKENS,
        "cases": [
            {"case_id": co["case_id"], "events": co["trace"]} for co in case_outputs
        ],
    }
    with open(trace_path, "w") as f:
        json.dump(trace_payload, f, indent=2)

    print(f"\nWrote {out_path}")
    print(f"Wrote {trace_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
