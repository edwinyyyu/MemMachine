# LoCoMo / segment-store benchmark work

**Before any benchmark run, read `./HANDOFF.md` in full.** Methodology errors in
this work have cost hundreds of wasted runs. Do not start runs from memory alone.

Non-negotiables (full detail and rationale in `HANDOFF.md`):

- **Segmenter → segments; deriver → derivatives.** Segments are returned, scored,
  and answered from. Derivatives are embedded. They are not the same thing.
- **`--answer-with-raw-events` is a diagnostic, not the architecture.** The real
  answer path is `string_from_segment_context` at `expand_context=0`. Never use
  `--separate-contexts` at `expand_context=0`.
- **The headline metric is accuracy at a fixed token budget — ≤340 tok/q target,
  350 tok/q HARD MAX — vs Mem0's claimed 87.3%.** fixed-K compares only same-system runs at unchanged segment
  granularity (e.g. deriver-only changes) — compare to Mem0 and across granularity
  changes at the token budget, reporting `avg(tokens/q)`; ALSO check no regression at
  the same `max_num_segments` so a low-token win doesn't sacrifice high-token
  retrieval. A real win is Pareto across the whole accuracy-vs-tokens curve.
- **Eval stack:** gpt-5-mini ans+judge for iteration. Mem0-comparable = gpt-5
  judge + `mem0-bench` (answerer may stay gpt-5-mini; if the answerer is gpt-5,
  the Mem0 answering prompt must be used too).
- **Record every setting of every run** (segmenter, deriver, embedder, answerer,
  judge model, judge variant, `max_num_segments`, BM25 mode+weight, pool,
  `expand_context`) in the filename. Unrecorded = wasted (see `eval_unknown_judge/`).
- **Attribute gains to the actual change.** A BM25-fusion-weight tweak or a
  model/embedder swap is NOT a deriver or architecture win — it is off-task.
- **The focus is justified conclusions** — an unjustified conclusion (e.g. a
  win claimed from a within-noise ~±0.25pp gain) is a serious error in any
  situation. Run more items only as needed to justify a conclusion — not
  reflexively, not never. Don't grind ~0.1% gains; prioritize speed and big
  gains. Validate with evidence, not guessing.

Invalid `--answer-with-raw-events` runs are archived in `wrong_methodology_rawev/`.
