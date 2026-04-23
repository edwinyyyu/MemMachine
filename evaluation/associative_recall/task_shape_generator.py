"""Generate task-shape adversarial input variants of LoCoMo questions.

For each of the first 30 LoCoMo questions in questions_extended.json,
produce 3 rewrites that preserve the information need but change surface
form:

  CMD   — imperative command ("Find ...", "Locate ...", "List ...")
  DRAFT — synthesis / draft request ("Summarize ...", "Draft ...")
  META  — open-ended meta-query ("What do we know about ...",
          "Walk me through ...", "Tell me about ...")

The generator uses gpt-5-mini with a dedicated tasksh_* cache so it
cannot conflict with concurrent agents. Gold source_ids are carried
from the original question unchanged (retrieval target is identical).

Output:
    data/questions_locomo_task_shape.json — 90 rows, each with:
      - conversation_id, category, question_index (original)
      - original_question
      - shape: one of {"CMD", "DRAFT", "META"}
      - question (the variant text)
      - source_chat_ids (same as original)
      - ideal_response, benchmark, orig_row_index
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import CACHE_DIR, LLMCache

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = DATA_DIR / "questions_locomo_task_shape.json"
N_QUESTIONS = 30
MODEL = "gpt-5-mini"

TASKSH_LLM_CACHE_FILE = CACHE_DIR / "tasksh_llm_cache.json"


PROMPT_TEMPLATE = """\
Rewrite this question as 3 alternative user inputs that have the SAME \
information need but different surface form. The answer should be \
exactly the same. Do NOT change entity names, dates, or specific details.

Original question: {question}

Output EXACTLY 3 alternatives, one per line:
CMD: <imperative command form, e.g., starting with "Find" or "List" or \
"Locate">
DRAFT: <synthesis/draft request, e.g., starting with "Summarize" or \
"Draft" or "Prepare">
META: <open-ended meta-query, e.g., starting with "What do we know about" \
or "Walk me through" or "Tell me about">
"""


class TaskShapeLLMCache(LLMCache):
    """Dedicated cache file so other agents' caches aren't corrupted."""

    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        if TASKSH_LLM_CACHE_FILE.exists():
            try:
                with open(TASKSH_LLM_CACHE_FILE) as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        self.cache_file = TASKSH_LLM_CACHE_FILE
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


def parse_rewrites(text: str) -> dict[str, str]:
    """Parse CMD/DRAFT/META lines. Returns mapping shape->text; missing
    shapes absent from output."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(CMD|DRAFT|META)\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            shape = m.group(1).upper()
            body = m.group(2).strip()
            # strip a leading markdown bullet/asterisk if present
            body = re.sub(r"^[-*]\s*", "", body)
            if body:
                out[shape] = body
    return out


def generate_one(
    client: OpenAI,
    cache: TaskShapeLLMCache,
    question: str,
) -> dict[str, str]:
    prompt = PROMPT_TEMPLATE.format(question=question)
    cached = cache.get(MODEL, prompt)
    if cached is not None:
        return parse_rewrites(cached)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2000,
    )
    text = resp.choices[0].message.content or ""
    cache.put(MODEL, prompt, text)
    return parse_rewrites(text)


def main() -> None:
    with open(DATA_DIR / "questions_extended.json") as f:
        all_qs = json.load(f)
    locomo_qs = [q for q in all_qs if q.get("benchmark") == "locomo"][
        :N_QUESTIONS
    ]
    print(f"Loaded {len(locomo_qs)} LoCoMo questions", flush=True)

    client = OpenAI(timeout=60.0)
    cache = TaskShapeLLMCache()

    out_rows: list[dict] = []
    for i, q in enumerate(locomo_qs):
        orig = q["question"]
        t0 = time.time()
        rewrites = generate_one(client, cache, orig)
        elapsed = time.time() - t0
        missing = [s for s in ("CMD", "DRAFT", "META") if s not in rewrites]
        if missing:
            print(
                f"  [{i + 1}/{len(locomo_qs)}] WARN missing {missing} "
                f"for: {orig[:60]}...",
                flush=True,
            )
        for shape in ("CMD", "DRAFT", "META"):
            variant_text = rewrites.get(shape, orig)  # fallback: keep orig
            out_rows.append({
                "orig_row_index": i,
                "conversation_id": q["conversation_id"],
                "category": q["category"],
                "question_index": q["question_index"],
                "shape": shape,
                "original_question": orig,
                "question": variant_text,
                "source_chat_ids": q["source_chat_ids"],
                "ideal_response": q.get("ideal_response", ""),
                "benchmark": q["benchmark"],
            })
        print(
            f"  [{i + 1}/{len(locomo_qs)}] {q['category']}: {orig[:55]}... "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        if (i + 1) % 5 == 0:
            cache.save()

    cache.save()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out_rows, f, indent=2, default=str)
    print(f"\nSaved {len(out_rows)} task-shape rows -> {OUT_PATH}", flush=True)

    # Quick sanity sample
    print("\nSample rewrites:")
    for i in (0, 1, 2):
        orig = locomo_qs[i]["question"]
        print(f"\nORIG ({locomo_qs[i]['category']}): {orig}")
        for shape in ("CMD", "DRAFT", "META"):
            row = next(
                (r for r in out_rows
                 if r["orig_row_index"] == i and r["shape"] == shape),
                None,
            )
            if row:
                print(f"  {shape}: {row['question']}")


if __name__ == "__main__":
    main()
