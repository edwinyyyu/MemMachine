"""Round 5 candidate schemas and framings.

Candidates:
  S1 baseline (round-4 winner): revise, add_member, remove_member, add, remove, noop
  S2 upsert_replace: upsert, add_member, remove_member, remove, noop
  S3 upsert_only:   upsert, noop
  S4 append_ref:    append, append_ref (with relation), noop
  S5 append_plain:  append, noop

Framings (for ablation on S1 and the winner):
  F_editor (round-4 default): "copy editor marking up a fact sheet"
  F_journal: "journal keeper, writing chronological entries"
  F_archivist: "archivist, preserving prior text verbatim where possible"
  F_diff: "patch author, edit as little as possible; preserve tokens"

Each framing supplies a top block; each schema supplies the verb-list
block. We paste framing + schema + input to build the final prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# FRAMING BLOCKS
# ---------------------------------------------------------------------------
F_EDITOR = """\
You are a copy editor marking up a numbered fact sheet. Each numbered line is one
fact, in the form:
  [n] topic.category | attribute: value (confidence_tag)

Confidence tag -- if present, it is EXACTLY one of: (confirmed), (hedged), (intended), (negated).

Before emitting any edit, decide: does this statement contain something that
BELONGS on a permanent fact sheet about the person? If not, emit noop.

DO NOT write to the sheet (emit noop instead):
- Weather comments, seasonal gripes, chitchat.
- Transient moods, fleeting reactions, jokes, sarcasm.
- Generic filler / acknowledgements.
- Repetitions of facts already on the sheet that add no new detail.

DO write to the sheet:
- Durable attributes, preferences, traits, plans/events.
- Any correction, addition, or removal to an existing fact.

Match facts by MEANING, not exact text. An ambiguous referent -> prefer noop.
When multiple distinct changes land in one turn, emit multiple edits.

IMPORTANT: When editing an existing fact, COPY THE EXISTING VALUE BODY VERBATIM
and only change the parts the new statement actually touches. Do not rephrase,
reorder, or summarize the unchanged portion.
"""


F_DIFF = """\
You are a minimal-diff patch author. You see a numbered fact sheet and a new
statement. Your job is to emit the SMALLEST set of commands that makes the
state correct.

Rules:
- Edit as little as possible. Prefer fine-grained ops over whole-value rewrites.
- Preserve tokens: if a value's body is unchanged and only confidence shifts,
  NEVER rewrite the value body. If you can change one member of a set rather
  than rewriting the whole set, do that.
- Copy existing text verbatim when you do need to rewrite -- do not paraphrase,
  reorder, or summarize.
- Noop on chitchat / mood / filler / repetition.
- Match facts by meaning; if ambiguous, noop.
"""


F_JOURNAL = """\
You are a journal keeper. Every turn you decide what, if anything, to write
down about the user. Write complete, self-contained observations. Prefer
writing down what was SAID rather than trying to infer what the current state
SHOULD be -- someone else will read the journal later to reconstruct state.

Do NOT write:
- Weather, mood, filler, chitchat, jokes, or acknowledgements.
- Repetitions of what was already written that add no new information.

Do write:
- Durable attributes, preferences, traits, plans/events.
- Corrections, clarifications, or retractions (use the REFERENCE mechanism).
- Ambiguous statements if they carry any information (better to capture than lose).
"""


F_ARCHIVIST = """\
You are a cautious archivist. You maintain a numbered fact sheet about a user.
You may edit facts but you preserve prior wording where possible. When you
revise a fact, you COPY THE UNCHANGED PORTION OF THE VALUE VERBATIM and only
change the parts the new turn actually touches.

Do NOT write:
- Weather, mood, chitchat, jokes, repetitions.

Do write:
- Durable attributes, preferences, traits, plans/events.
- Corrections, additions, retractions.

Match facts by meaning. Ambiguous -> noop.
"""


# ---------------------------------------------------------------------------
# SCHEMA VERB-LIST BLOCKS
# ---------------------------------------------------------------------------

S1_VERBS = """\
Edit verbs:

  {"op": "revise", "index": n, "new_text": "topic.category | attribute: new_value"}
      // replace the content of fact [n] entirely.
  {"op": "remove", "index": n}
      // strike fact [n].
  {"op": "add", "new_text": "topic.category | attribute: value", "cardinality": "single"|"set"}
      // append a brand-new numbered line.
  {"op": "add_member", "index": n, "member": "..."}
      // add one member to a set-valued row [n] (cardinality=set only).
  {"op": "remove_member", "index": n, "member": "..."}
      // remove one member from a set-valued row [n].
  {"op": "noop"}

Use add_member / remove_member ONLY when cardinality=set for row [n].
For confidence changes where value body is unchanged, revise with the body
copied verbatim and end the value with exactly one of (confirmed), (hedged),
(intended), (negated).
"""


S2_VERBS = """\
Edit verbs:

  {"op": "upsert", "topic_category": "...", "attribute": "...",
                   "value": "...", "cardinality": "single"|"set",
                   "confidence": "confirmed"|"hedged"|"intended"|"negated"}
      // create-or-replace: if a row with the same (topic_category, attribute)
      // exists, replace its value/cardinality/confidence. Otherwise insert
      // a new row. For set rows, `value` must be the full NEW set (comma-
      // separated string or JSON list).
  {"op": "remove", "index": n}
      // strike fact [n] (use for full retractions).
  {"op": "add_member", "index": n, "member": "..."}
      // add one member to a set-valued row [n].
  {"op": "remove_member", "index": n, "member": "..."}
      // remove one member from a set-valued row [n].
  {"op": "noop"}

Use `upsert` whether or not the row already exists -- you do not need to
decide between 'add' and 'revise'. The backend figures it out.
Use `add_member` / `remove_member` for single-member changes on set rows;
upsert for wholesale set replacement.
For retractions: prefer `remove` over upsert with confidence=negated.
"""


S3_VERBS = """\
Edit verbs (minimal):

  {"op": "upsert", "topic_category": "...", "attribute": "...",
                   "value": "...", "cardinality": "single"|"set",
                   "confidence": "confirmed"|"hedged"|"intended"|"negated"}
      // create-or-replace a row keyed by (topic_category, attribute).
      // For set-valued attributes, `value` is the full set (comma-separated
      // string or JSON list).
  {"op": "noop"}

Retractions: use upsert with confidence="negated" and the same value body
(or "[retracted]" as a placeholder for unknown values).
Add/remove single members of a set: upsert with the new full set.
"""


S4_VERBS = """\
This is an append-only log. You cannot edit or delete past entries. You can
only append a new entry.

Edit verbs:

  {"op": "append", "topic": "...", "text": "..."}
      // append a fresh observation. `text` is one sentence in natural
      // English; `topic` is a broad category (e.g., user.location,
      // user.health). Write complete, self-contained observations.

  {"op": "append_ref", "topic": "...", "refs": [id1, id2, ...],
                       "relation": "clarify"|"refine"|"supersede"|"invalidate",
                       "text": "..."}
      // append a new entry that is linked to one or more prior entry ids.
      // Relations:
      //   clarify    -- the new text adds a detail to the referenced entry
      //   refine     -- the new text narrows / strengthens / weakens the prior
      //   supersede  -- the new text replaces the prior (prior stays in log
      //                 but is marked invalidated for state derivation)
      //   invalidate -- the prior was flat-out wrong; marks it invalidated

  {"op": "noop"}

Current state is derived from the log. You do NOT need to compute the final
state -- just write accurate entries.

Write durable observations. Noop for weather/mood/filler/repetition/jokes.
"""


S5_VERBS = """\
This is an append-only log. You cannot edit or delete past entries.

Edit verbs:

  {"op": "append", "topic": "...", "text": "..."}
      // append a fresh observation. `text` is one sentence in natural
      // English.
  {"op": "noop"}

If the user corrects or retracts something, write the correction / retraction
as a new entry -- do not try to go back and edit. State is derived from the
full log later.

Noop for weather/mood/filler/repetition/jokes.
"""


# ---------------------------------------------------------------------------
# FOOTER (shared)
# ---------------------------------------------------------------------------

ROW_FOOTER = """\

CURRENT FACT SHEET:
{prior_state}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""


LOG_FOOTER = """\

CURRENT LOG (chronological, oldest first):
{prior_state}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""


# ---------------------------------------------------------------------------
# CANDIDATE registry
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    key: str
    name: str
    schema_family: str  # "row" or "log"
    framing: str
    verbs: str
    footer: str
    show_cardinality: bool = False

    def build_prompt(self, prior_state: str, turn: str) -> str:
        p = self.framing + "\n" + self.verbs + self.footer
        return p.replace("{prior_state}", prior_state).replace("{turn}", turn)


CANDIDATES: list[Candidate] = [
    # === Schema comparison (fixed framing = editor/archivist) ===
    Candidate(
        "S1_baseline_editor",
        "S1 baseline / editor framing",
        "row",
        F_EDITOR,
        S1_VERBS,
        ROW_FOOTER,
        show_cardinality=True,
    ),
    Candidate(
        "S2_upsert_replace",
        "S2 upsert + member ops / editor framing",
        "row",
        F_EDITOR,
        S2_VERBS,
        ROW_FOOTER,
        show_cardinality=True,
    ),
    Candidate(
        "S3_upsert_only",
        "S3 upsert only / editor framing",
        "row",
        F_EDITOR,
        S3_VERBS,
        ROW_FOOTER,
        show_cardinality=True,
    ),
    Candidate(
        "S4_append_ref",
        "S4 append + append_ref / journal framing",
        "log",
        F_JOURNAL,
        S4_VERBS,
        LOG_FOOTER,
    ),
    Candidate(
        "S5_append_plain",
        "S5 append only / journal framing",
        "log",
        F_JOURNAL,
        S5_VERBS,
        LOG_FOOTER,
    ),
    # === Framing ablations on S1 ===
    Candidate(
        "S1_diff_framing",
        "S1 baseline / minimal-diff framing (Q1)",
        "row",
        F_DIFF,
        S1_VERBS,
        ROW_FOOTER,
        show_cardinality=True,
    ),
    Candidate(
        "S1_archivist_framing",
        "S1 baseline / archivist framing (Q1)",
        "row",
        F_ARCHIVIST,
        S1_VERBS,
        ROW_FOOTER,
        show_cardinality=True,
    ),
]
