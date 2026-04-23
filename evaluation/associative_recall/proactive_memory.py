"""Proactive memory system: task -> decompose -> iterate -> gather -> stop.

Unlike recall@K-oriented retrievers, this architecture accepts task-shaped
prompts (drafts, plans, analyses) and proactively gathers evidence sufficient
to complete the task.

Two systems are defined here; the eval driver (`proactive_eval.py`) compares
them on LLM-judge task-sufficiency (0-10), not recall@K.

  System A  -- "single-shot":  em_v2f_speakerformat style. 1 cue-gen LLM
              call over the raw task prompt, retrieve K=50 unique turns.
              Reflects current best single-call recipe.

  System B  -- "proactive":
              LLM #1 DECOMPOSE: break the task into 3-6 info NEEDS with
                priority and expected-content hints.
              For each need: LLM cue-gen (speakerformat), retrieve top-K.
              LLM last SUFFICIENCY: inspect per-need retrieval, decide
                which needs are still under-covered. For each under-covered
                need, run a follow-up cue-gen + retrieval.
              Stop when all needs marked sufficient, or after max_rounds.
              Return UNION of retrieved turns, deduped by turn_id (ranked
              by max cosine score across probes).

Caches (dedicated so nothing collides with other agents):
  cache/proactive_decompose_cache.json
  cache/proactive_cuegen_cache.json
  cache/proactive_sufficiency_cache.json

The per-need cue-gen prompt is structurally aligned with em_retuned_cue_gen's
V2F_SPEAKERFORMAT_PROMPT so the distribution matches the speaker-baked
EventMemory embeddings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

from em_architectures import (
    V2F_MODEL,
    EMHit,
    _MergedLLMCache,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _query_em,
)


CACHE_DIR = Path(__file__).resolve().parent / "cache"

PROACTIVE_DECOMPOSE_CACHE = CACHE_DIR / "proactive_decompose_cache.json"
PROACTIVE_CUEGEN_CACHE = CACHE_DIR / "proactive_cuegen_cache.json"
PROACTIVE_SUFFICIENCY_CACHE = CACHE_DIR / "proactive_sufficiency_cache.json"


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


DECOMPOSE_PROMPT = """\
You are planning an information-gathering run to complete a task using a \
conversation memory. The memory holds chat turns between {participant_1} \
and {participant_2}.

TASK:
{task_prompt}

Decompose the task into 3-6 INFO NEEDS. Each need describes a distinct \
kind of information you must retrieve before you can do the task well. \
Think about what categories of content the task requires (e.g. preferences, \
unfinished threads, past events, emotional state, prior decisions, stated \
constraints).

For each need produce:
- `need`: short declarative phrase describing the info category
- `why`: one sentence explaining why the task requires this
- `priority`: "high" | "medium" | "low"
- `expected_vocab`: 3-6 words or short phrases you'd expect to appear in \
  matching chat turns (this will inform cue generation)

Output ONLY a JSON object, no prose:
{{"needs": [
  {{"need": "...", "why": "...", "priority": "...", "expected_vocab": ["...", "..."]}},
  ...
]}}"""


CUEGEN_PROMPT = """\
You are generating search cues for semantic retrieval over a conversation \
between {participant_1} and {participant_2}. Turns are embedded in the \
format "<speaker_name>: <chat content>" and your cues will be embedded \
the same way.

Overall TASK (for context):
{task_prompt}

Specific INFO NEED (what this cue-set targets):
{need}
Expected vocabulary: {expected_vocab}
{prior_section}

Generate 2 search cues that would retrieve chat turns matching THIS need. \
Each cue MUST begin with "{participant_1}: " or "{participant_2}: ". Use \
specific vocabulary that would appear in target turns. Do NOT write \
questions; write text that would actually appear in a chat message.

Format:
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
Nothing else."""


SUFFICIENCY_PROMPT = """\
You are auditing whether the retrieved content sufficiently covers each \
info need for the task.

TASK:
{task_prompt}

Per-need retrieval (top excerpts):
{per_need_section}

For each need, classify coverage as:
- "sufficient": enough specific content to inform the task
- "partial": some content, but key angles still missing
- "empty": little or no relevant content retrieved

For needs you mark "partial" or "empty", propose a SHORT follow-up probe \
(declarative text, ideally in chat register, no question) that would \
retrieve the missing angle. The probe will be embedded and used as a \
new search.

Output ONLY a JSON object, no prose:
{{"by_need": [
  {{"need": "...", "coverage": "sufficient|partial|empty", \
"followup_probe": "text or null"}},
  ...
]}}"""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


CUE_RE = re.compile(r"^\s*CUE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _strip_quotes(s: str) -> str:
    return s.strip().strip('"').strip("'").strip()


def parse_cues(response: str, max_cues: int = 2) -> list[str]:
    cues: list[str] = []
    for m in CUE_RE.finditer(response):
        cue = _strip_quotes(m.group(1))
        if cue:
            cues.append(cue)
        if len(cues) >= max_cues:
            break
    return cues


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(t[start:end + 1])
    except Exception:
        return None


def parse_needs(response: str) -> list[dict]:
    obj = _extract_json(response) or {}
    needs = obj.get("needs") or []
    out: list[dict] = []
    for n in needs:
        if not isinstance(n, dict):
            continue
        need = str(n.get("need") or "").strip()
        if not need:
            continue
        out.append({
            "need": need,
            "why": str(n.get("why") or "").strip(),
            "priority": str(n.get("priority") or "medium").strip().lower(),
            "expected_vocab": [str(x).strip() for x in (n.get("expected_vocab") or []) if str(x).strip()],
        })
    return out


def parse_sufficiency(response: str) -> list[dict]:
    obj = _extract_json(response) or {}
    bn = obj.get("by_need") or []
    out: list[dict] = []
    for item in bn:
        if not isinstance(item, dict):
            continue
        need = str(item.get("need") or "").strip()
        coverage = str(item.get("coverage") or "").strip().lower()
        fp = item.get("followup_probe")
        followup = str(fp).strip() if fp else ""
        out.append({
            "need": need,
            "coverage": coverage,
            "followup_probe": followup or None,
        })
    return out


# --------------------------------------------------------------------------
# LLM helper
# --------------------------------------------------------------------------


async def _llm_call(
    openai_client,
    prompt: str,
    cache: _MergedLLMCache,
) -> tuple[str, bool]:
    cached = cache.get(V2F_MODEL, prompt)
    if cached is not None:
        return cached, True
    resp = await openai_client.chat.completions.create(
        model=V2F_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    cache.put(V2F_MODEL, prompt, text)
    return text, False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _format_hits_for_judge(hits: list[EMHit], max_items: int = 8, max_len: int = 160) -> str:
    if not hits:
        return "(none)"
    top = sorted(hits, key=lambda h: -h.score)[:max_items]
    top = sorted(top, key=lambda h: h.turn_id)
    lines = []
    for h in top:
        txt = h.text.replace("\n", " ")
        if len(txt) > max_len:
            txt = txt[:max_len] + "..."
        lines.append(f"[Turn {h.turn_id}, {h.role}]: {txt}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------


@dataclass
class ProactiveResult:
    hits: list[EMHit]
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# System A: single-shot baseline (em_v2f_speakerformat shape)
# --------------------------------------------------------------------------


SINGLE_SHOT_PROMPT = """\
You are generating search cues for semantic retrieval over a conversation \
between {participant_1} and {participant_2}. Turns are embedded in the \
format "<speaker_name>: <chat content>" and your cues will be embedded \
the same way.

Task:
{task_prompt}

Generate 2 search cues. Each cue MUST begin with "{participant_1}: " or \
"{participant_2}: ". Use specific vocabulary that would appear in target \
turns. Do NOT write questions; write text that would actually appear in a \
chat message.

Format:
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
Nothing else."""


async def run_single_shot(
    memory: EventMemory,
    task_prompt: str,
    participants: tuple[str, str],
    *,
    K: int,
    cuegen_cache: _MergedLLMCache,
    openai_client,
) -> ProactiveResult:
    p1, p2 = participants
    prompt = SINGLE_SHOT_PROMPT.format(
        task_prompt=task_prompt, participant_1=p1, participant_2=p2,
    )
    raw, cache_hit = await _llm_call(openai_client, prompt, cuegen_cache)
    cues = parse_cues(raw, max_cues=2)

    primer = await _query_em(
        memory, task_prompt, vector_search_limit=K, expand_context=0
    )
    per_cue_hits: list[list[EMHit]] = []
    for cue in cues:
        per_cue_hits.append(
            await _query_em(memory, cue, vector_search_limit=K, expand_context=0)
        )
    merged = _merge_by_max_score([primer, *per_cue_hits])
    merged = _dedupe_by_turn_id(merged)[:K]
    return ProactiveResult(
        hits=merged,
        metadata={
            "system": "single_shot",
            "cues": cues,
            "n_llm_calls": 1,
            "cuegen_cache_hit": cache_hit,
            "n_turns_retrieved": len(merged),
        },
    )


# --------------------------------------------------------------------------
# System B: proactive decompose + iterate
# --------------------------------------------------------------------------


async def run_proactive(
    memory: EventMemory,
    task_prompt: str,
    participants: tuple[str, str],
    *,
    K_per_need: int = 15,
    K_final: int = 50,
    max_rounds: int = 2,
    decompose_cache: _MergedLLMCache,
    cuegen_cache: _MergedLLMCache,
    sufficiency_cache: _MergedLLMCache,
    openai_client,
) -> ProactiveResult:
    """Decompose -> per-need cue-gen+retrieve -> sufficiency audit -> followups.

    max_rounds=2 means at most: initial retrieval for all needs + 1 audit
    round with follow-up probes. max_rounds=3 allows a second audit pass.
    """
    p1, p2 = participants

    # ----- Call 1: DECOMPOSE -----
    decompose_prompt = DECOMPOSE_PROMPT.format(
        participant_1=p1, participant_2=p2, task_prompt=task_prompt,
    )
    decompose_raw, decompose_cache_hit = await _llm_call(
        openai_client, decompose_prompt, decompose_cache
    )
    needs = parse_needs(decompose_raw)
    if not needs:
        # Degenerate fallback: treat entire task as one need.
        needs = [{
            "need": task_prompt.strip().split("\n")[0][:120],
            "why": "fallback: parser failed",
            "priority": "high",
            "expected_vocab": [],
        }]

    n_llm_calls = 1  # decompose

    # ----- Per-need cue-gen + retrieval -----
    # all_probes: list of (probe_text, batch_of_hits) per probe
    per_need_state: list[dict] = []
    all_batches: list[list[EMHit]] = []

    for need in needs:
        vocab_str = ", ".join(need.get("expected_vocab") or []) or "(none)"
        cue_prompt = CUEGEN_PROMPT.format(
            participant_1=p1, participant_2=p2, task_prompt=task_prompt,
            need=need["need"], expected_vocab=vocab_str,
            prior_section="",
        )
        cue_raw, cue_hit = await _llm_call(openai_client, cue_prompt, cuegen_cache)
        n_llm_calls += 1
        cues = parse_cues(cue_raw, max_cues=2)

        batches_this_need: list[list[EMHit]] = []
        # Also do a primer on the need text itself (no LLM call).
        primer = await _query_em(
            memory, need["need"], vector_search_limit=K_per_need, expand_context=0
        )
        batches_this_need.append(primer)
        for cue in cues:
            hits = await _query_em(
                memory, cue, vector_search_limit=K_per_need, expand_context=0
            )
            batches_this_need.append(hits)
        merged_this_need = _merge_by_max_score(batches_this_need)
        merged_this_need = _dedupe_by_turn_id(merged_this_need)

        per_need_state.append({
            "need": need,
            "cues": cues,
            "cue_cache_hit": cue_hit,
            "hits": merged_this_need,
            "followup_probes": [],
        })
        all_batches.extend(batches_this_need)

    # ----- Rounds: sufficiency audit + follow-ups -----
    rounds_executed = 1  # initial per-need round = round 1
    sufficiency_reports: list[list[dict]] = []

    for round_idx in range(max_rounds - 1):
        # Build per-need section for the audit prompt.
        per_need_section_lines: list[str] = []
        for st in per_need_state:
            per_need_section_lines.append(f"### NEED: {st['need']['need']}")
            per_need_section_lines.append(
                _format_hits_for_judge(st["hits"], max_items=6, max_len=160)
            )
            per_need_section_lines.append("")
        per_need_section = "\n".join(per_need_section_lines)

        sufficiency_prompt = SUFFICIENCY_PROMPT.format(
            task_prompt=task_prompt,
            per_need_section=per_need_section,
        )
        suff_raw, suff_hit = await _llm_call(
            openai_client, sufficiency_prompt, sufficiency_cache
        )
        n_llm_calls += 1
        report = parse_sufficiency(suff_raw)
        sufficiency_reports.append(report)

        # Match report entries to needs by textual proximity (fall back to
        # positional if mismatch).
        need_texts = [st["need"]["need"] for st in per_need_state]
        # Build a simple lookup by case-insensitive substring.
        def _match_idx(entry_need: str) -> int | None:
            en = entry_need.strip().lower()
            for i, nt in enumerate(need_texts):
                if en == nt.strip().lower() or en in nt.lower() or nt.lower() in en:
                    return i
            return None

        any_probe_issued = False
        for j, entry in enumerate(report):
            idx = _match_idx(entry.get("need", ""))
            if idx is None:
                # Fallback: positional when lengths match.
                if len(report) == len(per_need_state):
                    idx = j
                else:
                    continue
            coverage = entry.get("coverage", "")
            probe = entry.get("followup_probe")
            if coverage == "sufficient" or not probe:
                continue
            # Issue a probe retrieval.
            hits = await _query_em(
                memory, probe, vector_search_limit=K_per_need, expand_context=0
            )
            per_need_state[idx]["followup_probes"].append(probe)
            # Merge into this need's hits.
            merged_this_need = _merge_by_max_score(
                [per_need_state[idx]["hits"], hits]
            )
            per_need_state[idx]["hits"] = _dedupe_by_turn_id(merged_this_need)
            all_batches.append(hits)
            any_probe_issued = True

        rounds_executed += 1
        if not any_probe_issued:
            # Converged: every need sufficient.
            break

    # ----- Global merge across all needs & probes -----
    global_merged = _merge_by_max_score(all_batches)
    global_merged = _dedupe_by_turn_id(global_merged)[:K_final]

    # Coverage stats from the LAST sufficiency report (if any).
    final_cov_counts = {"sufficient": 0, "partial": 0, "empty": 0, "unknown": 0}
    if sufficiency_reports:
        for e in sufficiency_reports[-1]:
            c = e.get("coverage", "")
            if c not in final_cov_counts:
                c = "unknown"
            final_cov_counts[c] += 1

    return ProactiveResult(
        hits=global_merged,
        metadata={
            "system": "proactive",
            "n_llm_calls": n_llm_calls,
            "n_turns_retrieved": len(global_merged),
            "rounds_executed": rounds_executed,
            "decompose_cache_hit": decompose_cache_hit,
            "needs": [
                {
                    "need": st["need"]["need"],
                    "priority": st["need"].get("priority", ""),
                    "expected_vocab": st["need"].get("expected_vocab", []),
                    "cues": st["cues"],
                    "cue_cache_hit": st["cue_cache_hit"],
                    "followup_probes": st["followup_probes"],
                    "n_hits_for_need": len(st["hits"]),
                    "top_turn_ids": [h.turn_id for h in sorted(st["hits"], key=lambda h: -h.score)[:10]],
                }
                for st in per_need_state
            ],
            "sufficiency_reports": sufficiency_reports,
            "final_coverage_counts": final_cov_counts,
        },
    )
