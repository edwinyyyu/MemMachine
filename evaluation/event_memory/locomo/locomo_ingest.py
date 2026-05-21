"""Ingest LoCoMo10 conversations into EventMemory.

Uses SQLite for everything: SQLAlchemySegmentStore for the segment store,
and SQLiteVecVectorStore (sqlite-vec) for the vector store. One segment-store
partition + one vector-store collection per conversation group.
"""

import argparse
import asyncio
import os

# Keep v2-fp and v3 imports for ablation reproducibility. The production
# rewriting segmenter is imported above from the package proper.
import sys as _sys
import time
from datetime import timedelta
from uuid import uuid4

import openai
from dotenv import load_dotenv
from length_routed_segmenter import LengthRoutedSegmenter
from locomo_models import (
    attachment_suffix,
    datetime_from_locomo_time,
    load_locomo_dataset,
)
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store import (
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
    SQLiteVecVectorStore,
    SQLiteVecVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    SurroundingEvent,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.llm_text_deriver import (
    LLMTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.llm_text_segmenter import (
    LLMTextSegmenter,
)
from memmachine_server.episodic_memory.event_memory.segmenter.rewrite_segmenter import (
    RewriteSegmenter as _RewriteSegmenter,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)

_sys.path.insert(
    0,
    "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval/llm_pipeline_probe",
)
from probe_segmenter_rewrite_v2_fp import RewriteSegmenterFP as _RewriteSegmenterV2FP
from probe_segmenter_rewrite_v3 import RewriteSegmenter as _RewriteSegmenterV3
from probe_segmenter_rewrite_v4 import RewriteSegmenter as _RewriteSegmenterV4
from probe_segmenter_rewrite_v5 import RewriteSegmenter as _RewriteSegmenterV5
from probe_segmenter_rewrite_v6 import RewriteSegmenter as _RewriteSegmenterV6
from probe_segmenter_rewrite_v7 import RewriteSegmenter as _RewriteSegmenterV7
from probe_segmenter_rewrite_v8 import RewriteSegmenter as _RewriteSegmenterV8
from probe_segmenter_rewrite_v9 import RewriteSegmenter as _RewriteSegmenterV9
from probe_segmenter_rewrite_v10 import RewriteSegmenter as _RewriteSegmenterV10
from probe_segmenter_rewrite_v11 import RewriteSegmenter as _RewriteSegmenterV11
from probe_segmenter_rewrite_v12 import RewriteSegmenter as _RewriteSegmenterV12
from probe_segmenter_rewrite_v13 import RewriteSegmenter as _RewriteSegmenterV13
from probe_segmenter_rewrite_v14 import RewriteSegmenter as _RewriteSegmenterV14
from probe_segmenter_rewrite_v15 import RewriteSegmenter as _RewriteSegmenterV15
from probe_segmenter_rewrite_v16 import RewriteSegmenter as _RewriteSegmenterV16
from probe_segmenter_rewrite_v17 import RewriteSegmenter as _RewriteSegmenterV17
from probe_segmenter_rewrite_v18 import RewriteSegmenter as _RewriteSegmenterV18
from probe_segmenter_rewrite_v19 import RewriteSegmenter as _RewriteSegmenterV19
from probe_segmenter_rewrite_v20 import RewriteSegmenter as _RewriteSegmenterV20
from probe_segmenter_rewrite_v21 import RewriteSegmenter as _RewriteSegmenterV21
from probe_segmenter_rewrite_v22 import RewriteSegmenter as _RewriteSegmenterV22
from sqlalchemy import event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry


def _configure_sqlite_for_perf(engine: AsyncEngine) -> None:
    """Set WAL + synchronous=NORMAL for ingest/search throughput.

    NORMAL is safe under WAL but can lose committed transactions on OS
    crash / power loss (no corruption). Acceptable tradeoff for benchmarks.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(
        dbapi_connection: DBAPIConnection,
        _connection_record: ConnectionPoolEntry,
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


def _build_segmenter(args, openai_client):
    """Build the segmenter selected by --segmenter."""
    match args.segmenter:
        case "llm":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return LLMTextSegmenter(language_model=lm)
        case "llm-routed":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return LengthRoutedSegmenter(
                language_model=lm,
                threshold_chars=args.routed_threshold,
            )
        case "rewrite" | "rewrite-v2":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenter(language_model=lm)
        case "rewrite-v3":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV3(language_model=lm)
        case "rewrite-v2-fp":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV2FP(language_model=lm)
        case "rewrite-v4":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV4(language_model=lm)
        case "rewrite-v5":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV5(language_model=lm)
        case "rewrite-v6":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV6(language_model=lm)
        case "rewrite-v7":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV7(language_model=lm)
        case "rewrite-v8":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV8(language_model=lm)
        case "rewrite-v9":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV9(language_model=lm)
        case "rewrite-v10":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV10(language_model=lm)
        case "rewrite-v11":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV11(language_model=lm)
        case "rewrite-v12":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV12(language_model=lm)
        case "rewrite-v13":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV13(language_model=lm)
        case "rewrite-v14":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV14(language_model=lm)
        case "rewrite-v15":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV15(language_model=lm)
        case "rewrite-v16":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV16(language_model=lm)
        case "rewrite-v17":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV17(language_model=lm)
        case "rewrite-v18":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV18(language_model=lm)
        case "rewrite-v19":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV19(language_model=lm)
        case "rewrite-v20":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV20(language_model=lm)
        case "rewrite-v21":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV21(language_model=lm)
        case "rewrite-v22":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22(language_model=lm)
        case "rewrite-v25":
            from probe_segmenter_rewrite_v25 import (
                RewriteSegmenter as _RewriteSegmenterV25,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV25(language_model=lm)
        case "rewrite-v23":
            from probe_segmenter_rewrite_v23 import (
                RewriteSegmenter as _RewriteSegmenterV23,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV23(language_model=lm)
        case "rewrite-v24":
            from probe_segmenter_rewrite_v24 import (
                RewriteSegmenter as _RewriteSegmenterV24,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV24(language_model=lm)
        case "rewrite-v22-nbn":
            from probe_segmenter_rewrite_v22_nbn import (
                RewriteSegmenter as _RewriteSegmenterV22NBN,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22NBN(language_model=lm)
        case "rewrite-v22-origemb":
            from probe_segmenter_rewrite_v22_origemb import (
                RewriteSegmenter as _RewriteSegmenterV22OE,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22OE(language_model=lm)
        case "rewrite-v22-rwemb":
            from probe_segmenter_rewrite_v22_rwemb import (
                RewriteSegmenter as _RewriteSegmenterV22RW,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22RW(language_model=lm)
        case "rewrite-v22-dates":
            from probe_segmenter_rewrite_v22_dates import (
                RewriteSegmenter as _RewriteSegmenterV22Dates,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22Dates(language_model=lm)
        case "rewrite-v22-fp":
            from probe_segmenter_rewrite_v22_fp import (
                RewriteSegmenter as _RewriteSegmenterV22FP,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22FP(language_model=lm)
        case "rewrite-v22-fp-cot":
            from probe_segmenter_rewrite_v22_fp_cot import (
                RewriteSegmenter as _RewriteSegmenterV22FPCot,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22FPCot(language_model=lm)
        case "rewrite-v22-fp-min":
            from probe_segmenter_rewrite_v22_fp_min import (
                RewriteSegmenter as _RewriteSegmenterV22FPMin,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22FPMin(language_model=lm)
        case "rewrite-v22-fp-minchg":
            from probe_segmenter_rewrite_v22_fp_minchg import (
                RewriteSegmenter as _RewriteSegmenterV22FPMinChg,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22FPMinChg(language_model=lm)
        case "rewrite-v22-says":
            from probe_segmenter_rewrite_v22_says import (
                RewriteSegmenter as _RewriteSegmenterV22Says,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22Says(language_model=lm)
        case "rewrite-v22-min3p":
            from probe_segmenter_rewrite_v22_min3p import (
                RewriteSegmenter as _RewriteSegmenterV22Min3p,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22Min3p(language_model=lm)
        case "rewrite-v22-saysmin":
            from probe_segmenter_rewrite_v22_saysmin import (
                RewriteSegmenter as _RewriteSegmenterV22SaysMin,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22SaysMin(language_model=lm)
        case "rewrite-v22-quoted":
            from probe_segmenter_rewrite_v22_quoted import (
                RewriteSegmenter as _RewriteSegmenterV22Quoted,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22Quoted(language_model=lm)
        case "rewrite-v22-headline":
            from probe_segmenter_rewrite_v22_headline import (
                RewriteSegmenter as _RewriteSegmenterV22Headline,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22Headline(language_model=lm)
        case "rewrite-v22-temporal":
            from probe_segmenter_rewrite_v22_temporal import (
                RewriteSegmenter as _RewriteSegmenterV22Temporal,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22Temporal(language_model=lm)
        case "rewrite-v22-fp-min-v2":
            from probe_segmenter_rewrite_v22_fp_min_v2 import (
                RewriteSegmenter as _RewriteSegmenterV22FPMinV2,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22FPMinV2(language_model=lm)
        case "rewrite-v22-fp-dual-v1":
            from probe_segmenter_rewrite_v22_fp_dual_v1 import (
                RewriteSegmenter as _RewriteSegmenterV22FPDualV1,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22FPDualV1(language_model=lm)
        case "rewrite-v22-fp-dual-v2":
            from probe_segmenter_rewrite_v22_fp_dual_v2 import (
                RewriteSegmenter as _RewriteSegmenterV22FPDualV2,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22FPDualV2(language_model=lm)
        case "rewrite-v22-min3p-dual":
            from probe_segmenter_rewrite_v22_min3p_dual import (
                RewriteSegmenter as _RewriteSegmenterV22Min3pDual,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22Min3pDual(language_model=lm)
        case "rewrite-v22-saysdecoupled":
            from probe_segmenter_rewrite_v22_saysdecoupled import (
                RewriteSegmenter as _RewriteSegmenterV22SaysDecoupled,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22SaysDecoupled(language_model=lm)
        case "rewrite-v22-naturaltime":
            from probe_segmenter_rewrite_v22_naturaltime import (
                RewriteSegmenter as _RewriteSegmenterV22NaturalTime,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22NaturalTime(language_model=lm)
        case "rewrite-v22-nodaterule":
            from probe_segmenter_rewrite_v22_nodaterule import (
                RewriteSegmenter as _RewriteSegmenterV22NoDateRule,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22NoDateRule(language_model=lm)
        case "rewrite-v22-qkey":
            from probe_segmenter_rewrite_v22_qkey import (
                RewriteSegmenter as _RewriteSegmenterV22QKey,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22QKey(language_model=lm)
        case "rewrite-v22-qkey-tight":
            from probe_segmenter_rewrite_v22_qkey_tight import (
                RewriteSegmenter as _RewriteSegmenterV22QKeyTight,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22QKeyTight(language_model=lm)
        case "rewrite-v22-qkey2x":
            from probe_segmenter_rewrite_v22_qkey2x import (
                RewriteSegmenter as _RewriteSegmenterV22QKey2X,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22QKey2X(language_model=lm)
        case "rewrite-v22-5w1h":
            from probe_segmenter_rewrite_v22_5w1h import (
                RewriteSegmenter as _RewriteSegmenterV225W1H,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV225W1H(language_model=lm)
        case "rewrite-v22-qkey-min3p":
            from probe_segmenter_rewrite_v22_qkey_min3p import (
                RewriteSegmenter as _RewriteSegmenterV22QKeyMin3p,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22QKeyMin3p(language_model=lm)
        case "rewrite-v22-qkey-min3p-nq":
            from probe_segmenter_rewrite_v22_qkey_min3p_nq import (
                RewriteSegmenter as _RewriteSegmenterV22QKeyMin3pNQ,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22QKeyMin3pNQ(language_model=lm)
        case "rewrite-v22-qa":
            from probe_segmenter_rewrite_v22_qa import (
                RewriteSegmenter as _RewriteSegmenterV22QA,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22QA(language_model=lm)
        case "rewrite-v22-paraphrase":
            from probe_segmenter_rewrite_v22_paraphrase import (
                RewriteSegmenter as _RewriteSegmenterV22Paraphrase,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22Paraphrase(language_model=lm)
        case "rewrite-v22-anchors":
            from probe_segmenter_rewrite_v22_anchors import (
                RewriteSegmenter as _RewriteSegmenterV22Anchors,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22Anchors(language_model=lm)
        case "rewrite-v22-richblock":
            from probe_segmenter_rewrite_v22_richblock import (
                RewriteSegmenter as _RewriteSegmenterV22RichBlock,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22RichBlock(language_model=lm)
        case "rewrite-v22-memonly":
            from probe_segmenter_rewrite_v22_memonly import (
                RewriteSegmenter as _RewriteSegmenterV22MemOnly,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22MemOnly(language_model=lm)
        case "rewrite-v22-naturaltime-v2":
            from probe_segmenter_rewrite_v22_naturaltime_v2 import (
                RewriteSegmenter as _RewriteSegmenterV22NaturalTimeV2,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22NaturalTimeV2(language_model=lm)
        case "rawchunk":
            from probe_raw_seg_llm_deriver_v1 import RawChunkSegmenter
            return RawChunkSegmenter()
        case "rewrite-v22-natural-says-v1":
            from probe_segmenter_rewrite_v22_natural_says_v1 import (
                RewriteSegmenter as _RewriteSegmenterV22NaturalSaysV1,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22NaturalSaysV1(language_model=lm)
        case "rewrite-v22-first-person-clean-v1":
            from probe_segmenter_rewrite_v22_first_person_clean_v1 import (
                RewriteSegmenter as _RewriteSegmenterV22FirstPersonCleanV1,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22FirstPersonCleanV1(language_model=lm)
        case "rewrite-v22-min3p-keepdate":
            from probe_segmenter_rewrite_v22_min3p_keepdate import (
                RewriteSegmenter as _RewriteSegmenterV22Min3pKeepdate,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22Min3pKeepdate(language_model=lm)
        case "rewrite-v22-nomsgdate":
            from probe_segmenter_rewrite_v22_nomsgdate import (
                RewriteSegmenter as _RewriteSegmenterV22NoMsgDate,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22NoMsgDate(language_model=lm)
        case "rewrite-v22-says-v2":
            from probe_segmenter_rewrite_v22_says_v2 import (
                RewriteSegmenter as _RewriteSegmenterV22SaysV2,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22SaysV2(language_model=lm)
        case "rewrite-v22-headline-v2":
            from probe_segmenter_rewrite_v22_headline_v2 import (
                RewriteSegmenter as _RewriteSegmenterV22HeadlineV2,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22HeadlineV2(language_model=lm)
        case "rewrite-v22-fp-min-v3":
            from probe_segmenter_rewrite_v22_fp_min_v3 import (
                RewriteSegmenter as _RewriteSegmenterV22FPMinV3,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22FPMinV3(language_model=lm)
        case "rewrite-v22-says-v3":
            from probe_segmenter_rewrite_v22_says_v3 import (
                RewriteSegmenter as _RewriteSegmenterV22SaysV3,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22SaysV3(language_model=lm)
        case "rewrite-v22-quoted-v2":
            from probe_segmenter_rewrite_v22_quoted_v2 import (
                RewriteSegmenter as _RewriteSegmenterV22QuotedV2,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22QuotedV2(language_model=lm)
        case "rewrite-v22-temporal-v2":
            from probe_segmenter_rewrite_v22_temporal_v2 import (
                RewriteSegmenter as _RewriteSegmenterV22TemporalV2,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22TemporalV2(language_model=lm)
        case "rewrite-v22-min3p-v2":
            from probe_segmenter_rewrite_v22_min3p_v2 import (
                RewriteSegmenter as _RewriteSegmenterV22Min3pV2,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22Min3pV2(language_model=lm)
        case "rewrite-v22-listfaith":
            from probe_segmenter_rewrite_v22_listfaith import (
                RewriteSegmenter as _RewriteSegmenterV22ListFaith,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22ListFaith(language_model=lm)
        case "rewrite-v22-qc":
            from probe_segmenter_rewrite_v22_qc import (
                RewriteSegmenter as _RewriteSegmenterV22QC,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22QC(language_model=lm)
        case "rewrite-v26":
            from probe_segmenter_rewrite_v26 import (
                RewriteSegmenter as _RewriteSegmenterV26,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV26(language_model=lm)
        case "rewrite-v27":
            from probe_segmenter_rewrite_v27 import (
                RewriteSegmenter as _RewriteSegmenterV27,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV27(language_model=lm)
        case "rewrite-v28":
            from probe_segmenter_rewrite_v28 import (
                RewriteSegmenter as _RewriteSegmenterV28,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV28(language_model=lm)
        case "rewrite-v29":
            from probe_segmenter_rewrite_v29 import (
                RewriteSegmenter as _RewriteSegmenterV29,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV29(language_model=lm)
        case "rewrite-v30":
            from probe_segmenter_rewrite_v30 import (
                RewriteSegmenter as _RewriteSegmenterV30,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV30(language_model=lm)
        case "rewrite-v31":
            from probe_segmenter_rewrite_v31 import (
                RewriteSegmenter as _RewriteSegmenterV31,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV31(language_model=lm)
        case "rewrite-v32":
            from probe_segmenter_rewrite_v32 import (
                RewriteSegmenter as _RewriteSegmenterV32,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV32(language_model=lm)
        case "rewrite-v33":
            from probe_segmenter_rewrite_v33 import (
                RewriteSegmenter as _RewriteSegmenterV33,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV33(language_model=lm)
        case "cached":
            from cached_segmenter import CachedSegmenter as _CachedSegmenter
            if not getattr(args, 'segments_cache', None):
                raise ValueError("--segmenter cached requires --segments-cache PATH")
            return _CachedSegmenter(cache_path=args.segments_cache)
        case "decoupling-ablation":
            from probe_decoupling_ablation import CachedReassemblySegmenter
            if not getattr(args, "segments_cache", None):
                raise ValueError(
                    "--segmenter decoupling-ablation requires --segments-cache PATH"
                )
            return CachedReassemblySegmenter(
                cache_path=args.segments_cache,
                variant=args.reassembly_variant,
            )
        case "rewrite-v27q":
            from probe_segmenter_rewrite_v27q import (
                RewriteSegmenter as _RewriteSegmenterV27Q,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV27Q(language_model=lm)
        case "text":
            return TextSegmenter(max_chunk_length=args.max_text_chunk_length)
        case "windowchunk":
            from probe_chunk_deriver_v1 import WindowChunkSegmenter

            return WindowChunkSegmenter(chunk_size=args.max_text_chunk_length)
        case "decoupled-chunk-v1":
            from probe_decoupled_chunk_v1 import RawChunkRewriteSegmenter

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return RawChunkRewriteSegmenter(
                language_model=lm,
                chunk_size=args.max_text_chunk_length,
            )
        case "terse-decoupled-v1":
            from probe_terse_decoupled_v1 import TerseDecoupledSegmenter

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return TerseDecoupledSegmenter(language_model=lm)
        case "terse-decoupled-v2":
            from probe_terse_decoupled_v2 import (
                TerseDecoupledSegmenter as TerseDecoupledSegmenterV2,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return TerseDecoupledSegmenterV2(language_model=lm)
        case "terse-decoupled-slim-v1":
            from probe_terse_decoupled_slim_v1 import (
                TerseDecoupledSegmenter as TerseDecoupledSegmenterSlimV1,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return TerseDecoupledSegmenterSlimV1(language_model=lm)
        case "terse-decoupled-slim-v2":
            from probe_terse_decoupled_slim_v2 import (
                TerseDecoupledSegmenter as TerseDecoupledSegmenterSlimV2,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return TerseDecoupledSegmenterSlimV2(language_model=lm)
        case "terse-decoupled-slim-v3":
            from probe_terse_decoupled_slim_v3 import (
                TerseDecoupledSegmenter as TerseDecoupledSegmenterSlimV3,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return TerseDecoupledSegmenterSlimV3(language_model=lm)
        case "terse-decoupled-v3":
            from probe_terse_decoupled_v2 import (
                TerseDecoupledSegmenter as TerseDecoupledSegmenterV3,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return TerseDecoupledSegmenterV3(
                language_model=lm, include_raw_chunk_in_embed=False
            )


def _build_deriver(args, openai_client):
    """Build the deriver selected by --deriver."""
    match args.deriver:
        case "llm":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return LLMTextDeriver(language_model=lm)
        case "atomic-v1":
            from probe_deriver_atomic_v1 import AtomicDeriver as _AtomicDeriverV1

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return _AtomicDeriverV1(language_model=lm)
        case "generic-v1":
            from probe_deriver_generic_v1 import GenericDeriver as _GenericDeriverV1

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return _GenericDeriverV1(language_model=lm)
        case "qshape-v1":
            from probe_deriver_qshape_v1 import GenericDeriver as _QShapeDeriverV1
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return _QShapeDeriverV1(language_model=lm)
        case "multiaxis-v1":
            from probe_deriver_multiaxis_v1 import GenericDeriver as _MultiAxisDeriverV1
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return _MultiAxisDeriverV1(language_model=lm)
        case "topiccue-v1":
            from probe_deriver_topiccue_v1 import GenericDeriver as _TopicCueDeriverV1
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return _TopicCueDeriverV1(language_model=lm)
        case "stable-v1":
            from probe_deriver_stable_v1 import GenericDeriver as _StableDeriverV1
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return _StableDeriverV1(language_model=lm)
        case "rewriteonly-v1":
            from probe_deriver_rewriteonly_v1 import GenericDeriver as _RewriteOnlyV1
            return _RewriteOnlyV1()
        case "tagsuffix-v1":
            from probe_deriver_tagsuffix_v1 import GenericDeriver as _TagSuffixV1
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return _TagSuffixV1(language_model=lm)
        case "subjectline-v1":
            from probe_deriver_subjectline_v1 import GenericDeriver as _SubjectLineV1
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return _SubjectLineV1(language_model=lm)
        case "routed-v1":
            from probe_deriver_routed_v1 import GenericDeriver as _RoutedV1
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return _RoutedV1(language_model=lm)
        case "whole":
            return WholeTextDeriver()
        case "sentence":
            return SentenceTextDeriver()
        case "rawseg-llm-v1":
            from probe_raw_seg_llm_deriver_v1 import LLMRewriteDeriver

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return LLMRewriteDeriver(language_model=lm)
        case "rawseg-llm-v2-v65":
            from probe_raw_seg_llm_deriver_v2_v65 import V65DeriverProbe

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return V65DeriverProbe(language_model=lm)
        case "rawseg-llm-v3-dual":
            from probe_raw_seg_llm_deriver_v3_dual import DualTextLLMRewriteDeriver

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return DualTextLLMRewriteDeriver(language_model=lm)
        case "rawseg-llm-v4-nodate":
            from probe_raw_seg_llm_deriver_v4_nodate import NoDateLLMRewriteDeriver

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return NoDateLLMRewriteDeriver(language_model=lm)
        case "rawseg-llm-v5-parallel":
            from probe_raw_seg_llm_deriver_v5_parallel import (
                ParallelLLMRewriteDeriver,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return ParallelLLMRewriteDeriver(language_model=lm)
        case "rawseg-llm-v6-richprompt":
            from probe_raw_seg_llm_deriver_v6_richprompt import (
                RichPromptDualTextDeriver,
            )

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return RichPromptDualTextDeriver(language_model=lm)
        case "chunk-deriver-v1":
            from probe_chunk_deriver_v1 import ChunkContextDeriver

            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return ChunkContextDeriver(language_model=lm)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to the data file")
    parser.add_argument(
        "--segment-db",
        default="locomo_segments.db",
        help="SQLite path for the segment store",
    )
    parser.add_argument(
        "--vector-db",
        default="locomo_vectors.db",
        help="SQLite path for the sqlite-vec vector store",
    )
    parser.add_argument(
        "--group-index",
        type=int,
        default=None,
        help="Ingest only this group index",
    )
    parser.add_argument(
        "--max-text-chunk-length",
        type=int,
        default=500,
        help="Max code-point length for text chunks (TextSegmenter only)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent conversations",
    )
    parser.add_argument(
        "--segmenter",
        choices=[
            "text",
            "llm",
            "llm-routed",
            "rewrite",
            "rewrite-v2",
            "rewrite-v3",
            "rewrite-v2-fp",
            "rewrite-v4",
            "rewrite-v5",
            "rewrite-v6",
            "rewrite-v7",
            "rewrite-v8",
            "rewrite-v9",
            "rewrite-v10",
            "rewrite-v11",
            "rewrite-v12",
            "rewrite-v13",
            "rewrite-v14",
            "rewrite-v15",
            "rewrite-v16",
            "rewrite-v17",
            "rewrite-v18",
            "rewrite-v19",
            "rewrite-v20",
            "rewrite-v21",
            "rewrite-v22",
            "rewrite-v23",
            "rewrite-v24",
            "rewrite-v25",
            "rewrite-v26",
            "rewrite-v27",
            "rewrite-v27q",
            "rewrite-v28",
            "rewrite-v29",
            "rewrite-v30",
            "rewrite-v31",
            "rewrite-v32",
            "rewrite-v33",
            "cached",
            "decoupling-ablation",
            "rewrite-v22-qc",
            "rewrite-v22-nbn",
            "rewrite-v22-origemb",
            "rewrite-v22-rwemb",
            "rewrite-v22-dates",
            "rewrite-v22-fp",
            "rewrite-v22-fp-cot",
            "rewrite-v22-fp-min",
            "rewrite-v22-fp-minchg",
            "rewrite-v22-says",
            "rewrite-v22-min3p",
            "rewrite-v22-saysmin",
            "rewrite-v22-quoted",
            "rewrite-v22-headline",
            "rewrite-v22-temporal",
            "rewrite-v22-fp-min-v2",
            "rewrite-v22-fp-dual-v1",
            "rewrite-v22-fp-dual-v2",
            "rewrite-v22-min3p-dual",
            "rewrite-v22-nomsgdate",
            "rewrite-v22-min3p-keepdate",
            "rewrite-v22-saysdecoupled",
            "rewrite-v22-naturaltime",
            "rewrite-v22-naturaltime-v2",
            "rewrite-v22-nodaterule",
            "rewrite-v22-qkey",
            "rewrite-v22-qkey-tight",
            "rewrite-v22-qkey2x",
            "rewrite-v22-5w1h",
            "rewrite-v22-qkey-min3p",
            "rewrite-v22-qkey-min3p-nq",
            "rewrite-v22-qa",
            "rewrite-v22-paraphrase",
            "rewrite-v22-anchors",
            "rewrite-v22-richblock",
            "rewrite-v22-memonly",
            "rewrite-v22-natural-says-v1",
            "rewrite-v22-first-person-clean-v1",
            "rawchunk",
            "windowchunk",
            "decoupled-chunk-v1",
            "terse-decoupled-v1",
            "terse-decoupled-v2",
            "terse-decoupled-slim-v1",
            "terse-decoupled-slim-v2",
            "terse-decoupled-slim-v3",
            "terse-decoupled-v3",
            "rewrite-v22-says-v2",
            "rewrite-v22-headline-v2",
            "rewrite-v22-fp-min-v3",
            "rewrite-v22-says-v3",
            "rewrite-v22-quoted-v2",
            "rewrite-v22-temporal-v2",
            "rewrite-v22-min3p-v2",
            "rewrite-v22-listfaith",
        ],
        default="text",
        help="Segmenter type: 'text' (TextSegmenter, recursive splitter), "
        "'llm' (LLMTextSegmenter, v33 prompt), or 'llm-routed' (v47s for "
        "short inputs, v33 for long, threshold via --routed-threshold). "
        "Default: text.",
    )
    parser.add_argument(
        "--routed-threshold",
        type=int,
        default=200,
        help="Char-length threshold for --segmenter llm-routed (default: 200).",
    )
    parser.add_argument(
        "--segmenter-model",
        default="gpt-5.4-nano",
        help="OpenAI model for --segmenter llm (default: gpt-5.4-nano).",
    )
    parser.add_argument(
        "--segmenter-reasoning",
        default="low",
        help="reasoning_effort for --segmenter llm (default: low).",
    )
    parser.add_argument(
        "--deriver",
        choices=["sentence", "whole", "llm", "atomic-v1", "generic-v1", "qshape-v1", "multiaxis-v1", "topiccue-v1", "stable-v1", "rewriteonly-v1", "tagsuffix-v1", "subjectline-v1", "routed-v1", "rawseg-llm-v1", "rawseg-llm-v2-v65", "rawseg-llm-v3-dual", "rawseg-llm-v4-nodate", "rawseg-llm-v5-parallel", "rawseg-llm-v6-richprompt", "chunk-deriver-v1"],
        default="sentence",
        help="Deriver type: 'sentence' (one derivative per sentence), "
        "'whole' (one derivative per whole segment), "
        "or 'llm' (LLMTextDeriver, v65 prompt). Default: sentence.",
    )
    parser.add_argument(
        "--deriver-model",
        default="gpt-5-nano",
        help="OpenAI model for --deriver llm (default: gpt-5-nano).",
    )
    parser.add_argument(
        "--deriver-reasoning",
        default="low",
        help="reasoning_effort for --deriver llm (default: low).",
    )
    parser.add_argument(
        "--segments-cache",
        type=str,
        default=None,
        help="If set with --segmenter cached, read pre-computed segments "
        "from this JSON cache (built by dumping segment_store_sg).",
    )
    parser.add_argument(
        "--reassembly-variant",
        type=str,
        default="cur",
        help="Variant name for --segmenter decoupling-ablation (see "
        "probe_decoupling_ablation.VARIANTS).",
    )
    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=2,
        help="Number of neighboring messages to include in SurroundingEventsContext (default: 2).",
    )
    parser.add_argument(
        "--neighbor-direction",
        choices=["both", "before", "after", "none"],
        default="both",
        help="Direction of neighbor inclusion. 'none' = empty context (default: both).",
    )
    parser.add_argument(
        "--neighbor-before-window",
        type=int,
        default=None,
        help="If set, overrides --neighbor-window for the BEFORE side.",
    )
    parser.add_argument(
        "--neighbor-after-window",
        type=int,
        default=None,
        help="If set, overrides --neighbor-window for the AFTER side.",
    )
    parser.add_argument(
        "--filter-cues",
        action="store_true",
        help="Wrap the deriver in CueWorthinessFilteringDeriver to drop "
        "low-utility segments (conversational plumbing).",
    )
    parser.add_argument(
        "--cue-filter-model",
        default="gpt-5.4-nano",
        help="OpenAI model for cue-worthiness classification.",
    )
    parser.add_argument(
        "--cue-filter-reasoning",
        default="low",
        help="reasoning_effort for cue-worthiness model.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        choices=["text-embedding-3-small", "text-embedding-3-large"],
        help="OpenAI embedding model. 'small' (default, 1536d) or 'large' (3072d).",
    )

    args = parser.parse_args()

    locomo_data = load_locomo_dataset(args.data_path)

    segment_engine = create_async_engine(
        f"sqlite+aiosqlite:///{args.segment_db}",
        connect_args={"timeout": 30},
        pool_size=20,
        max_overflow=80,
    )
    _configure_sqlite_for_perf(segment_engine)
    segment_store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=segment_engine)
    )
    await segment_store.startup()

    vector_engine = create_async_engine(
        f"sqlite+aiosqlite:///{args.vector_db}",
        connect_args={"timeout": 30},
        pool_size=20,
        max_overflow=80,
    )
    _configure_sqlite_for_perf(vector_engine)
    vector_store = SQLiteVecVectorStore(
        SQLiteVecVectorStoreParams(engine=vector_engine)
    )
    await vector_store.startup()

    openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if args.embedding_model == "text-embedding-3-small":
        emb_dims = 1536
    elif args.embedding_model == "text-embedding-3-large":
        emb_dims = 3072
    else:
        raise ValueError(f"Unknown embedding model: {args.embedding_model}")
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model=args.embedding_model,
            dimensions=emb_dims,
            max_input_length=8192,
        )
    )

    segmenter = _build_segmenter(args, openai_client)
    deriver = _build_deriver(args, openai_client)
    if args.filter_cues:
        from memmachine_server.episodic_memory.event_memory.deriver.cue_worthiness_filtering_deriver import (
            CueWorthinessFilteringDeriver,
        )

        cue_lm = OpenAIResponsesLanguageModel(
            OpenAIResponsesLanguageModelParams(
                client=openai_client,
                model=args.cue_filter_model,
                reasoning_effort=args.cue_filter_reasoning,
            )
        )
        deriver = CueWorthinessFilteringDeriver(
            inner=deriver,
            language_model=cue_lm,
        )

    namespace = "locomo"
    schema = EventMemory.expected_vector_store_collection_schema()

    async def process_conversation(idx: int, item: dict) -> None:
        if "conversation" not in item:
            return

        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        print(
            f"Processing conversation for group {idx} with speakers "
            f"{speaker_a} and {speaker_b}..."
        )

        partition_key = f"group_{idx}"

        await vector_store.delete_collection(namespace=namespace, name=partition_key)
        await segment_store.delete_partition(partition_key)

        collection = await vector_store.open_or_create_collection(
            namespace=namespace,
            name=partition_key,
            config=VectorStoreCollectionConfig(
                vector_dimensions=embedder.dimensions,
                similarity_metric=embedder.similarity_metric,
                indexed_properties_schema=schema,
            ),
        )
        segment_store_partition = await segment_store.open_or_create_partition(
            partition_key,
            SegmentStorePartitionConfig(),
        )

        memory = EventMemory(
            EventMemoryParams(
                vector_store_collection=collection,
                segment_store_partition=segment_store_partition,
                segmenter=segmenter,
                deriver=deriver,
                embedder=embedder,
            )
        )

        session_idx = 0
        while True:
            session_idx += 1
            session_id = f"session_{session_idx}"
            if session_id not in conversation:
                break

            session = conversation[session_id]
            session_datetime = datetime_from_locomo_time(
                conversation[f"{session_id}_date_time"]
            )

            message_texts = [m["text"] + attachment_suffix(m) for m in session]
            neighbor_window = getattr(args, "neighbor_window", 2)
            neighbor_direction = getattr(args, "neighbor_direction", "both")
            neighbor_before_override = getattr(args, "neighbor_before_window", None)
            neighbor_after_override = getattr(args, "neighbor_after_window", None)
            events: list[Event] = []
            for message_index, message in enumerate(session):
                content = message_texts[message_index]
                base_before = (
                    neighbor_window if neighbor_direction in ("both", "before") else 0
                )
                base_after = (
                    neighbor_window if neighbor_direction in ("both", "after") else 0
                )
                before_window = (
                    neighbor_before_override
                    if neighbor_before_override is not None
                    else base_before
                )
                after_window = (
                    neighbor_after_override
                    if neighbor_after_override is not None
                    else base_after
                )
                lo = max(0, message_index - before_window)
                before = [
                    SurroundingEvent(
                        producer=session[j]["speaker"],
                        text=message_texts[j].strip(),
                    )
                    for j in range(lo, message_index)
                ]
                hi = min(len(session), message_index + 1 + after_window)
                after = [
                    SurroundingEvent(
                        producer=session[j]["speaker"],
                        text=message_texts[j].strip(),
                    )
                    for j in range(message_index + 1, hi)
                ]
                events.append(
                    Event(
                        uuid=uuid4(),
                        timestamp=session_datetime
                        + message_index * timedelta(seconds=1),
                        context=SurroundingEventsContext(
                            producer=message["speaker"],
                            before=before,
                            after=after,
                        ),
                        blocks=[TextBlock(text=content.strip())],
                        properties={
                            "locomo_session_id": session_id,
                            "dia_id": message.get("dia_id", ""),
                            "group_idx": idx,
                        },
                    )
                )

            try:
                await memory.encode_events(events)
            except Exception as e:
                print(
                    f"Error ingesting group={idx} session={session_id} "
                    f"({len(events)} events): {e}"
                )
                raise

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)

    indices = (
        [args.group_index] if args.group_index is not None else range(len(locomo_data))
    )

    start_time = time.monotonic()
    print(f"Ingestion started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            async_with(semaphore, process_conversation(idx, locomo_data[idx]))
            for idx in indices
        ]
        await asyncio.gather(*tasks)
    finally:
        elapsed = time.monotonic() - start_time
        print(
            f"Ingestion finished at {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(elapsed: {elapsed:.1f}s)"
        )

    await segment_store.shutdown()
    await vector_store.shutdown()
    await segment_engine.dispose()
    await vector_engine.dispose()
    await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
