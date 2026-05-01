# Retrieval Architecture Research — Current State

## Evaluation Methodology Issues (Critical)

Three evaluation methods were used across experiments; they give different answers:

1. **Standalone** (`arch_segments[:K]` vs `baseline_segments[:K]`): unfair when arch has <K segments. Used in `best_shot.py`, `fulleval_run.py`. This is what produced the "r@50 regression" findings that turned out to be artifacts.

2. **Union** (`baseline_top_K ∪ all_arch_segments`): unfair in the other direction — gives arch bigger effective budget. Used in `universal_eval.py`. This is what produced the "everything is positive" findings that were overly optimistic.

3. **Backfill** (arch segments, then cosine backfill to exactly K): truly fair, both sides have exactly K segments. Used in `precision_retrieval.py` backfill mode. This is the correct method.

## Trustworthy Findings

### r@20 results are mostly trustworthy
Since most architectures retrieve ≥20 segments, the r@20 comparisons are fair in standalone mode.

**Winning architectures at r@20 on LoCoMo 30q:**
| Architecture | r@20 | delta | W/T/L | LLM/q |
|---|---|---|---|---|
| full_pipeline (with LLM reranking) | 86.7% | +48.3pp | 18/12/0 | 4.5 |
| meta_v2f | 75.6% | +37.2pp | 13/17/0 | 1.0 |
| hybrid_v2f_gencheck | 75.6% | +37.2pp | 13/17/0 | 2.0 |
| v15_control | 70.6% | +32.2pp | 12/18/0 | 1.0 |

### Cue generation finds genuinely new content
71% of cue-found segments are NOT in baseline top-20. Cues reach parts of embedding space cosine can't.

### Cosine reranking reverts to baseline
Can't use cosine to rerank cue-found pool — those segments aren't cosine-similar to the question by design.

### LLM reranking works
9 wins, 0 losses through 20/30 on decompose_then_retrieve. Promotes relevant segments from positions 21+ into top-20.

### Domain language is load-bearing for conversations
"conversation history" and "chat message" in v2f prompt help on LoCoMo. V2f_minimal (domain-agnostic) loses 8.9pp on LoCoMo but gains 2pp on synthetic/advanced.

### Completeness + anti-question prompts synergize on LoCoMo only
V2f's two additions help on LoCoMo (+5pp over v15) but the anti-question instruction actively HURTS proactive tasks (where question-style cues are effective for discovering implicit needs).

### Gen-Check gap assessment discovers proactive needs
Skeptical prompt ("what assumptions am I making?") gets +19.6pp on synthetic proactive category. Works where cue-based retrieval can't.

### Model behavior findings
- Models will NOT admit uncertainty unless forced (Gen-Check v1 never emitted NEED)
- Models under-generate cues when given freedom
- More rounds = dilution at r@20
- THINK/daydream steps lead to better retrieval decisions

## Findings That Need Re-Verification

### r@50 results across all evaluations
Most architectures retrieve ~30 segments, making standalone r@50 unfair and union r@50 over-optimistic. The fair backfill evaluation is running now.

### "Regression on short conversations"
Was an artifact of standalone evaluation. In backfill mode, r@50 can only equal or exceed baseline.

### Gen-Check r@50 advantage
The +33pp swing on proactive r@50 used standalone evaluation. Needs re-verification.

## Known Prompt Bugs (Fixed)
- Boolean queries ("X OR Y") in cue outputs — ~1% of v15/v2f cache entries
- Meta-instructions ("Search for messages...") — up to 10% in tree cache, fixed in best_shot
- These prompt bugs explained most of the "architecture gap" — frontier_v2 went from +17.2pp to +35.6pp just by fixing prompts

## Open Research Directions

1. **Fair backfill evaluation across all datasets** (running now)
2. **LLM reranker as a universal post-processor** — does it help all architectures equally?
3. **Adaptive activation** — skip cue generation for short conversations or simple questions
4. **Content-type parameterization** — pass domain hint ("conversations", "documents") to prompt
5. **Constraint-type cues** for scattered-item queries (100% recall on logic_constraint)
6. **Task execution integration** — Gen-Check v2 unlocked proactive retrieval but hasn't been combined with other components in a unified system

## Recommended Production Architecture

Based on consistent findings (not affected by evaluation bugs):

1. **Initial cosine retrieval** (top-10, 1 embed call)
2. **V2f cue generation** (1 LLM call, 2 embed calls) — proven best single-call architecture
3. **Backfill mode**: baseline top-20 always included, cue-found fills positions 21+
4. **[Optional] Gen-Check for task-framed queries**: skeptical gap assessment (1 LLM call, 1-2 embed calls)
5. **[Optional] LLM reranking when pool > 30**: listwise selection of top-20

Total cost: 1-3 LLM calls depending on complexity. Zero r@20 regression risk due to backfill.
