"""The datasets, why each is in the benchmark, and how to prepare it.

Inclusion rules (revised 2026-11-06), applied to every candidate dataset:
  1. Questions are not generated combinatorially and are not contrived. Model-written or
     model-drafted questions are acceptable when people curated them (LongMemEval's were
     drafted by an LLM and rewritten by experts).
  2. Gold is granular enough to tell which retrieved text holds the answer: character
     spans, or units such as turns that map to them. Document-level gold is not enough.
  3. Each dataset adds a query kind, corpus kind or gold shape the others lack; that is
     the reason to add a dataset on top of one already used.
  4. Preparation is reproducible from public sources: each preparer downloads its source
     and writes deterministic output.

Active (DATASETS): LongMemEval-M, the chat-memory case; LongMemEval-S only to test the
benchmark itself. INACTIVE holds the datasets prepared earlier, to be reconsidered later.
Flat datasets expose download(raw_dir) and prepare(raw_dir) -> (docs, queries, stats);
partitioned ones (one corpus per query) are prepared by their own module (longmemeval.py).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from rbench import schema
from rbench.paths import RAW


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    documents: str
    queries: str
    gold: str
    scope: str  # "corpus", "doc" or "scope_docs"
    scope_reason: str
    default_variant: str
    variants: dict[str, str]
    adds: str
    caveats: str
    source: str
    license: str
    variant_scopes: dict[str, str] = field(default_factory=dict)  # variants valid only in another scope
    # What bootstrap intervals resample: queries that share it are not independent.
    # "doc_id" (queries about one document share its chunks) or a feature name.
    cluster: str = "doc_id"
    partitioned: bool = False  # each query has its own corpus (a chat history), processed alone (histories.py)
    evaluated: bool = True  # False: kept to test the benchmark itself, not reported as a result


_LONGMEMEVAL = dict(
    documents="One user's chat history with an assistant per question: ~500 sessions (M) of user and assistant "
              "turns, sessions dated; filler sessions from ShareGPT and UltraChat, evidence sessions LLM-simulated "
              "and human-screened (about 70% human-edited)",
    queries="Questions about the user's past conversations (single-session user/assistant/preference, multi-session, "
            "knowledge update, temporal reasoning), LLM-drafted and rewritten by human experts (~5% of drafts kept)",
    gold="Every turn marked has_answer (1.9 per question; 94% are user turns, median 67 tokens; assistant evidence "
         "turns median 248), all needed; the evidence sessions as a second, session-level view",
    scope="history",
    scope_reason="a chatbot searches the memory of the user it is talking to: each question's own history",
    default_variant="question",
    variants={"question": "the question as asked"},
    adds="The chat-memory case EventMemory serves: long per-user histories, dated sessions, questions needing one "
         "or several sessions",
    caveats="Questions and evidence sessions involve LLM generation (human-rewritten and screened). The 30 "
            "abstention questions have no answer in the history and are excluded. Gold is whole turns: long "
            "assistant turns make 'complete' strict, which is why 'touched' and 'fraction' are reported too.",
    source="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
    license="MIT",
    cluster="qid",
    partitioned=True,
)

DATASETS: dict[str, DatasetInfo] = {
    "longmemeval_m": DatasetInfo(name="longmemeval_m", **_LONGMEMEVAL),
    "longmemeval_s": DatasetInfo(name="longmemeval_s", **{**_LONGMEMEVAL, "evaluated": False,
                                 "caveats": "The same questions over ~50-session histories. Used only to test the "
                                            "benchmark itself (fast), not reported as a result."}),
}

# Datasets prepared and profiled before the focus moved to LongMemEval-M; kept so they can be
# re-added by moving them back into DATASETS.
INACTIVE: dict[str, DatasetInfo] = {
    "nq": DatasetInfo(
        name="nq",
        documents="Wikipedia articles (one per query; DAPR's plain-text rendering), median 2.5k tokens",
        queries="Real Google search queries: about 10 tokens, lowercase, mostly wh-questions without a question mark",
        gold="The annotated long answer: one paragraph, list or table (median 118 tokens); any one suffices",
        scope="corpus",
        scope_reason="the queries stand alone, so the whole corpus is the realistic candidate set",
        default_variant="question",
        variants={"question": "the query as typed"},
        adds="Genuine user search queries (not written by annotators who had seen the answer), and gold "
             "passages longer than the smallest chunks, so small chunks must be read together",
        caveats="DAPR selected NQ-hard queries (479) that need document context; 1,000 more are a fixed random "
                "sample. Each article belongs to one query, so distractor articles are unrelated topics. The gold "
                "paragraph sits early in its article (median 7% in).",
        source="https://huggingface.co/datasets/UKPLab/dapr (NaturalQuestions)",
        license="Natural Questions: CC BY-SA 3.0 (unverified for DAPR's packaging)",
    ),
    "conditionalqa": DatasetInfo(
        name="conditionalqa",
        documents="UK government guidance pages (gov.uk), median 1.3k tokens, headed sections and conditions",
        queries="A person's scenario followed by their question (median 55 tokens), written by crowdworkers",
        gold="Every passage the answer depends on, including its conditions (4.3 spans of about 16 tokens)",
        scope="corpus",
        scope_reason="the scenario names the situation, so the question stands alone",
        default_variant="scenario_question",
        variants={"scenario_question": "scenario, then question (as DAPR gives it)",
                  "question": "the question alone, from the original ConditionalQA release",
                  "scenario": "the scenario alone"},
        adds="Contextualized, multi-condition prompts like a chatbot user's; many near-duplicate pages "
             "(benefits, visas) make the corpus hard to separate by topic",
        caveats="Small (271 test queries). Annotators wrote questions after reading the page.",
        source="https://huggingface.co/datasets/UKPLab/dapr (ConditionalQA); "
               "https://github.com/haitian-sun/ConditionalQA (scenario and question separately)",
        license="unverified; check the ConditionalQA repository",
    ),
    "legalbenchrag": DatasetInfo(
        name="legalbenchrag",
        documents="Contracts and privacy policies (NDAs, CUAD commercial contracts, 30 sampled MAUD merger "
                  "agreements), median 6.3k tokens, up to 156k",
        queries="Templated legal questions that name the contract (median 31 tokens)",
        gold="Expert-annotated clauses, all needed (1.6 spans, median 59 tokens)",
        scope="corpus",
        scope_reason="each query names its contract, so the corpus is searchable as one",
        default_variant="question",
        variants={"question": "'Consider <contract>; <ask>'", "ask": "the ask without naming the contract"},
        variant_scopes={"ask": "doc"},
        adds="Long, heavily structured documents (clauses, numbered sections) with near-identical boilerplate "
             "across documents, and expert gold",
        caveats="Queries come from dataset templates rather than real users. MAUD is subsampled to 30 of 150 "
                "agreements to bound the corpus.",
        source="https://github.com/zeroentropy-ai/legalbenchrag",
        license="code MIT; each source corpus (ContractNLI, CUAD, MAUD, PrivacyQA) has its own terms",
    ),
    "qasper": DatasetInfo(
        name="qasper",
        documents="Full NLP research papers (sectioned), median 4.7k tokens",
        queries="Short questions (median 9 tokens) by NLP practitioners who had read only the title and abstract",
        gold="Evidence sentences highlighted by other experts (1.2 spans, median 45 tokens)",
        scope="doc",
        scope_reason="questions assume the paper ('what datasets do they use?'), so each is searched in its paper",
        default_variant="question",
        variants={"question": "the question as written", "scoped": "prefixed with the paper's title"},
        adds="Information-seeking questions written without seeing the answer text, so lexical overlap with "
             "the gold is moderate (0.34), unlike extractive datasets",
        caveats="Unanswerable and figure-only questions are dropped.",
        source="https://allenai.org/data/qasper (v0.3)",
        license="CC BY 4.0",
    ),
    "covid_qa": DatasetInfo(
        name="covid_qa",
        documents="Biomedical research articles (CORD-19), median 5.4k tokens",
        queries="Questions written by biomedical experts while reading the article (median 12 tokens)",
        gold="A short extractive answer span (median 12 tokens)",
        scope="doc",
        scope_reason="questions assume the article",
        default_variant="question",
        variants={"question": "the question as written", "scoped": "prefixed with the article's title"},
        adds="Very short gold inside long technical prose: the smallest gold of any dataset",
        caveats="Closest in profile to QASPER (scientific articles, short questions, doc scope); the first to "
                "drop if cost must be cut. No official split: dev is a hash of the article id.",
        source="https://huggingface.co/datasets/deepset/covid_qa_deepset",
        license="Apache 2.0",
    ),
    "multidoc2dial": DatasetInfo(
        name="multidoc2dial",
        documents="US government service pages (SSA, VA, DMV, student aid), median 0.8k tokens, many headings",
        queries="A user's turn in a support dialogue, with the dialogue so far (median 126 tokens with history); "
                "requests, statements and follow-ups as well as questions",
        gold="The passages the agent's next reply is grounded in (1.6 spans of about 14 tokens)",
        scope="corpus",
        scope_reason="the dialogue is grounded in one of the domain's documents, which the system must find",
        default_variant="history",
        variants={"history": "the whole dialogue so far (the raw chatbot prompt)",
                  "recent": "the previous agent turn and the user turn",
                  "turn": "the user turn alone"},
        adds="Chatbot prompts: multi-turn context, non-question forms, and short context-dependent follow-ups "
             "(18% of turns are under 6 words)",
        caveats="Crowdworkers role-played the dialogues from document outlines. Pages repeat across versions; "
                "verbatim copies of the gold in other documents are accepted as alternatives.",
        source="https://doc2dial.github.io/multidoc2dial/",
        license="Apache 2.0 (per the Hugging Face card of IBM/multidoc2dial)",
        cluster="dialogue_id",  # turns of one dialogue share its history and documents
    ),
    "swebench_lite": DatasetInfo(
        name="swebench_lite",
        documents="Python source files of 12 open-source repositories at each issue's base commit, median 5k tokens",
        queries="Real GitHub issues (median 334 tokens): bug reports with code, tracebacks and expected behavior",
        gold="The original lines the fix edits (1.8 spans of about 15 tokens; an insertion anchors on the "
             "nearest non-blank line)",
        scope="scope_docs",
        scope_reason="each issue has its own repository snapshot; candidates are the edited files plus the files "
                     "SWE-bench's BM25 retrieved for it (4.3 files, about 27k tokens)",
        default_variant="issue",
        variants={"issue": "the issue as filed", "title": "the issue's title only"},
        adds="The coding-agent case: long raw prompts, code documents, and the lowest query-gold lexical "
             "overlap of any dataset (0.05)",
        caveats="Candidates are the gold files plus SWE-bench's BM25 files for the issue, so the gold file is "
                "always present (52% of gold files were not BM25 hits): first-stage recall is higher than a search "
                "of the whole repository would give, and the distractors are lexically close to the issue. Edited "
                "lines are where the fix goes, not everything an agent must read.",
        source="https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite (+ _oracle, _bm25_27K)",
        license="SWE-bench MIT; each repository's own license applies to its code",
        cluster="instance_id",  # each issue has its own snapshot of candidate files
    ),
}


def info(name: str) -> DatasetInfo:
    return DATASETS.get(name) or INACTIVE[name]


def module(name: str):
    return importlib.import_module(f"rbench.datasets.{name}")


def prepare(name: str) -> None:
    """Download (if needed) and prepare one dataset, then validate and write it."""
    if info(name).partitioned:
        from rbench.datasets import longmemeval

        print(longmemeval.prepare(name, RAW))
        return
    raw = RAW / name
    mod = module(name)
    mod.download(raw)
    docs, rows, stats = mod.prepare(raw)
    info_ = info(name)
    stats = {"name": name, "default_variant": info_.default_variant, "scope": info_.scope, **stats}
    print(schema.write(name, docs, rows, stats))
