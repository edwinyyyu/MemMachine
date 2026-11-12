"""Per-query features for slicing results by query type. Rule-based, no model calls.

  qtokens    Qwen3 tokens of the query text
  length     short | medium | long: the dataset's own tertiles of qtokens (each label states
             its range), so each holds about a third of the queries
  form       question | command | keyword | statement, from the query's last sentence:
               question   ends with "?" or opens with an interrogative or auxiliary word
               command    opens with an imperative verb or a request ("I need", "please")
               keyword    up to 8 words with no sentence punctuation and no verb cue
               statement  anything else (reports, descriptions, bug reports)
  sentences  one | several: several sentences, lines or dialogue turns (on LongMemEval mostly
             a sentence pointing back to an earlier chat before the ask)
  parts      single | multi: multi when the best gold alternative has two or more spans
             (the answer needs several separate pieces of the source)

The labels are deliberately simple and auditable (every rule is below). They slice
results; they are not ground truth about intent. A feature with one value for every query
(form on LongMemEval: all questions) is left out.
"""

from __future__ import annotations

import re

import numpy as np

from rbench.schema import queries
from rbench.text import QWEN, count

WH = {"what", "who", "whom", "whose", "when", "where", "which", "why", "how"}
AUX = {"is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would", "will", "shall",
       "may", "might", "must", "has", "have", "had", "am", "isn't", "aren't", "don't", "doesn't", "didn't", "can't"}
IMPERATIVE = {"find", "show", "give", "list", "tell", "explain", "describe", "fix", "add", "make", "help", "please",
              "remove", "change", "update", "implement", "support", "allow", "use", "get", "check", "summarize",
              "compare", "write", "create", "let", "provide", "consider", "identify"}
REQUEST = re.compile(r"^(i (need|want|would like|'d like|wish|am looking|'m looking)|i'd like|could you|can you|"
                     r"would you|please)\b", re.I)
VERB_CUE = re.compile(r"\b(is|are|was|were|be|been|do|does|did|have|has|had|can|will|would|should|i|you|we|my)\b", re.I)


def _last_sentence(text: str) -> str:
    text = text.strip().splitlines()[-1] if text.strip() else ""
    if text.startswith(("User:", "Agent:")):
        text = text.split(":", 1)[1]
    parts = [p for p in re.split(r"(?<=[.?!])\s+", text.strip()) if p]
    return parts[-1] if parts else text


def form(text: str) -> str:
    last = _last_sentence(text)
    words = re.findall(r"[A-Za-z']+", last.lower())
    if not words:
        return "statement"
    if last.rstrip().endswith("?") or words[0] in WH or words[0] in AUX:
        return "question"
    if words[0] in IMPERATIVE or REQUEST.search(last.strip()):
        return "command"
    if len(text.split()) <= 8 and not re.search(r"[.!?;:]", text) and not VERB_CUE.search(text):
        return "keyword"
    return "statement"


def sentences(text: str) -> str:
    parts = [p for p in re.split(r"(?<=[.?!])\s+|\n+", text.strip()) if p.strip()]
    return "several" if len(parts) > 1 else "one"


def query_features(dataset: str, variant: str, indices: list[int]) -> dict[str, list]:
    """Labels for the queries at `indices`, one list per feature."""
    rows = queries(dataset)
    texts = [rows[i]["variants"][variant] for i in indices]
    qtokens = [int(n) for n in count(texts, QWEN)]
    parts = ["multi" if min(len(a) for a in rows[i].get("gold") or rows[i]["gold_turns"]) >= 2 else "single"
             for i in indices]
    every = count([q["variants"][variant] for q in rows], QWEN)  # the dataset's own tertiles, over all its queries
    t1, t2 = (int(x) for x in np.percentile(every, [100 / 3, 200 / 3]))
    names = {"short": f"short (≤{t1} tokens)", "medium": f"medium ({t1 + 1}–{t2})", "long": f"long (>{t2})"}
    out = {
        "qtokens": qtokens,
        "length": [names["short"] if n <= t1 else names["medium"] if n <= t2 else names["long"] for n in qtokens],
        "form": [form(t) for t in texts],
        "sentences": [sentences(t) for t in texts],
        "parts": parts,
    }
    if len(set(out["form"])) == 1:  # one value for every query slices nothing
        del out["form"]
    return out
