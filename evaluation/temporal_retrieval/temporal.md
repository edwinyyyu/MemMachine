Final state of temporal_retrieval/:

  classifier.py            (unchanged)
  core.py                  Interval(earliest_us, latest_us) — simplified
  extractor_common.py      shared _LLMCache + full_ref_context
  extractor_v2.py          two-pass, emits TimeEnvelope
  extractor_v3.py          single-pass (production default), emits TimeEnvelope
  planner.py               (unchanged)
  retriever.py             default extractor=TemporalExtractorV3
  schema.py                ONE struct: TimeEnvelope(surface, earliest, latest,
                                                    granularity, confidence, ...)
