"""LLM segmenter, short-input specialization v48.

v47s correctly merged short multi-sentence messages into one entry,
but had a failure mode: messages that LEAD with a reaction
("Wow!", "Thanks!", "Glad ...") were classified as pure framing and
dropped, even when later sentences contained substantive content --
an attached photo description, a specific referent ("Those posters"),
a named activity ("painting", "hiking"), an opinion with backing, or
a referent-bearing question ("what about your library?").

Diagnostic on LoCoMo group_0: ~20-25 of 52 dropped events under v47s
actually contained substantive content. Pattern: the model
overweighted the leading rhetorical mode.

v48 fixes this by:
  1. Replacing the holistic "is this pure framing?" test with an
     existential "does ANY part contain specifics?" test.
  2. Listing concretely what counts as a specific (added: attached
     media descriptions; referent-bearing questions; specific
     activities and preferences).
  3. Defining PURE FRAMING as the negation: every part of the message
     is interchangeable filler, NO specifics anywhere.

No outcome bias (the test is symmetric); no inline input->output
example.
"""

from __future__ import annotations

PROMPT_SHORT_V48 = """\
Decide whether this short conversational message contains anything \
worth retaining for memory retrieval.

Rules:
1. VERBATIM. If you emit the message, emit it as a contiguous verbatim \
quote. Never paraphrase, swap synonyms, or change wording -- \
"fabulous" stays "fabulous"; preserve whitespace, newlines, and \
special characters exactly. The only allowed trim is dropping a \
leading greeting that prefixes substantive content -- the substantive \
content stays.
2. SPECIFICS TEST. Scan the ENTIRE message. The message contains \
specifics if any part of it has even one of: a named entity (person, \
place, brand, work), a date, a time, a number or identifier, a \
specific activity ("painting", "hiking"), a specific object \
("posters", "library", "guitar"), a preference, an opinion with \
backing, an emotional state tied to a specific event, a decision, a \
plan, a constraint, a description of attached media (a photo, \
video, file with its content described), a referent-bearing question \
that points to something concrete ("what kind of X", "what about \
your X"). A leading reaction or sign-off does not negate specifics \
that appear later in the same message -- scan to the end.
3. PURE FRAMING is the negation: every sentence is interchangeable \
filler with no specifics anywhere. Bare greetings ("Hey, how are \
you?"), pure reactions with no follow-up specifics ("Wow, that's \
amazing!"), sign-offs ("Bye!", "Have a great day!"), generic \
affirmations ("Glad you're happy!", "Life is short!"), and abstract \
wisdom unmoored from specifics qualify.
4. OUTPUT. If the message contains any specifics, emit it as ONE \
verbatim entry. If the message is pure framing, emit an empty list. \
The whole message is one unit -- it is not split at sentence \
boundaries.

Output: a JSON object {{ "segments": [...] }} and nothing else.

MESSAGE:
{passage}"""
