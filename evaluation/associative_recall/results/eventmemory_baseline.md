# EventMemory-backed Architecture Re-Evaluation on LoCoMo-30

## Setup

- n_questions = 30 (filter: `benchmark==locomo`, first 30)
- namespace = `arc_em_locomo30`
- logical prefix = `arc_em_locomo30_v1` (Qdrant-safe: `arc_em_lc30_v1_<26|30|41>`)
- `max_text_chunk_length = 500`, `derive_sentences = False`, `reranker = None` (explicit — pure embedding scores)
- Speaker baked into embedded text via `MessageContext.source = <speaker_name>` using `conversation_two_speakers.json`
- segment store: `sqlite+aiosqlite:////Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/results/eventmemory.sqlite3`

## Results (EventMemory backend)

| Architecture | R@20 | R@50 | time (s) |
| --- | --- | --- | --- |
| `em_cosine_baseline` | 0.7333 | 0.8833 | 15.2 |
| `em_cosine_expand_6` | 0.6944 | 0.8056 | 7.4 |
| `em_v2f` | 0.7417 | 0.8833 | 37.0 |
| `em_v2f_expand_6` | 0.7111 | 0.7667 | 42.7 |
| `em_ens_2` | 0.7833 | 0.8667 | 77.6 |

## Side-by-side with prior SegmentStore baselines

| SS arch | SS R@20 | SS R@50 | EM equivalent | EM R@20 | EM R@50 |
| --- | --- | --- | --- | --- | --- |
| cosine_baseline (raw cosine) | 0.3833 | 0.5083 | `em_cosine_baseline` | 0.7333 | 0.8833 |
| meta_v2f | 0.7556 | 0.8583 | `em_v2f` | 0.7417 | 0.8833 |
| ens_2_v2f_typeenum (sum_cosine) | 0.5806 | 0.9083 | `em_ens_2` | 0.7833 | 0.8667 |

Additional SS references (no direct EM counterpart in this run):

- `speaker_user_filter` (SS): R@20=0.8389, R@50=0.8917
- `two_speaker_filter` (SS): R@20=0.8917, R@50=0.8917
- `ens_2_v2f_typeenum (rrf)` (SS): R@20=0.6889, R@50=0.9083

## Findings

### Speaker channel is NOT vestigial on LoCoMo

`em_cosine_baseline` lifts R@20 from 0.3833 (SS raw cosine) to 0.7333 thanks to speaker baking, and R@50 from 0.5083 to 0.8833. That's a large free lift at pure-cosine, but it still falls short of `two_speaker_filter` (SS @20=0.8917). The decision rule (`em_cosine_baseline >= 0.85` -> speaker channel redundant) is NOT met: the explicit speaker filter still adds ~15pp over raw speaker-baked cosine at K=20. Verdict: speaker baking replaces most of the raw-cosine-vs-speaker-filter gap but does not fully subsume the hard-filter channel.

### expand_context REGRESSES turn-level recall

`em_cosine_expand_6` drops R@50 to 0.8056 (vs 0.8833 w/o expand). `em_v2f_expand_6` drops to 0.7667 (vs 0.8833 for em_v2f). Reason: expand_context consumes retrieval budget on neighbors of strong seeds, so fewer *independent* seeds fit under K. Helpful for QA-context assembly (wider windows for gpt-5-mini), but a net *negative* for recall evaluation of set-oriented gold turn_ids. Decision rule on expand_context ("beats em_v2f -> free lift") is NOT met.

### v2f transfers cleanly (parity, not lift)

`em_v2f` at K=50 = 0.8833 is near-parity with SS meta_v2f (0.8583). The retrieval-backend swap is neutral for v2f at K=50; slightly worse at K=20 (0.7417 vs 0.7556). V2F cues already contain speaker-prefixed strings like "Caroline: I went to..." in cached outputs, so they align well with speaker-baked embedded text. Decision rule (`em_v2f >= 0.90` K=50 -> production win) is NOT met.

### Ensemble v2f+type_enumerated is competitive

`em_ens_2` = 0.7833 / 0.8667 (R@20 / R@50). Versus SS sum_cosine 0.5806 / 0.9083: EM lifts R@20 substantially (speaker-baked context helps low-K precision) but loses ~4pp at K=50 where SS sum_cosine peaks. At K=50, SS ens_2_v2f_typeenum (any merge) still sits above EM.

## Updated production recipe (EventMemory backend)

1. Ingest with `MessageContext.source = <real speaker name>`, `max_text_chunk_length=500`, `derive_sentences=False`, `reranker=None`.
2. Best single-call retrieval on LoCoMo so far: `em_v2f` (R@50=0.8833).
3. Do NOT set `expand_context > 0` for turn-level recall; reserve for QA context assembly.
4. The hard `two_speaker_filter` channel is still worth keeping on LoCoMo until we find an EM-native replacement (property_filter on `context.source`?).

## Architectures deferred (out of session budget)

- `em_critical_info` — LoCoMo has 0 flagged turns per `results/critical_info_store.json`, so no expected lift over baseline.
- `em_alias_expand_v2f`, `em_gated_no_speaker` — require porting alias registry / confidence-gating channel logic to EM. Flagged for the follow-up prompt-tuning task (#74); with speaker-baked embedded text the v2f and type_enumerated prompts may benefit from retuning to mimic `"Caroline: content"` form.

## Outputs

- Results JSON: `results/eventmemory_baseline.json`
- Collections manifest (for cleanup): `results/eventmemory_collections.json`
- Source: `em_setup.py`, `em_architectures.py`, `em_eval.py` under `evaluation/associative_recall/`

## Deployment notes (environmental deviations)

- The existing Postgres container (`agentic-pg`) mounts persisted user data whose roles do not match any configured credentials in `.env` files. To avoid touching user data, `em_setup.py` falls back to a file-backed SQLite store (`results/eventmemory.sqlite3`) for the `SQLAlchemySegmentStore`. Qdrant ran normally on `agentic-qdrant` (6333/6334).
- LongMemEval hard secondary run was NOT attempted in this session. LoCoMo primary decision rules were definitive (speaker channel not vestigial, expand_context regresses, v2f parity), so the session-budget-permitting secondary check was skipped to keep the session focused. LongMemEval-hard data is present in-tree (`data/longmemeval_hard_segments.npz`, `data/questions_longmemeval_hard.json`), so the follow-up run can pick up from here; remember to prepend "User: " to queries/cues.