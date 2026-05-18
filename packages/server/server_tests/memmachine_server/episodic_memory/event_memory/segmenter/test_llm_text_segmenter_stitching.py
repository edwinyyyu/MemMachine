"""Tests for LLMTextSegmenter's source-anchored stitching.

The segmenter post-processes the LLM's segment list so that
``"".join(segments)`` reconstructs the source text with dropped
content silently elided. The stitching is deterministic and slices
text from the source, which guarantees verbatim fidelity (whitespace,
newlines, ASCII art) even if the LLM lightly normalized its quoting.
"""

from memmachine_server.episodic_memory.event_memory.segmenter.llm_text_segmenter import (
    LLMTextSegmenter,
)


class TestStitchSegmentsToSource:
    def test_consecutive_segments_include_trailing_whitespace(self):
        source = "Project budget is $45k. Deadline is March 30."
        llm_segments = [
            "Project budget is $45k.",
            "Deadline is March 30.",
        ]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert "".join(stitched) == source
        assert stitched == ["Project budget is $45k. ", "Deadline is March 30."]

    def test_dropped_content_silently_elided(self):
        source = (
            "Hi team. Project budget is $45k. "
            "Let me know if any questions. Thanks, Sarah"
        )
        llm_segments = [
            "Project budget is $45k.",
            "Thanks, Sarah",
        ]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        reconstruction = "".join(stitched)
        assert reconstruction == "Project budget is $45k. Thanks, Sarah"
        assert "Hi team" not in reconstruction
        assert "Let me know" not in reconstruction

    def test_ascii_art_preserved_when_emitted_as_one_segment(self):
        source = "Look at this:\n  /\\_/\\\n ( o.o )\n  > ^ <\nCute!"
        # LLM emits the ascii art as one segment plus the closing comment
        llm_segments = [
            "Look at this:\n  /\\_/\\\n ( o.o )\n  > ^ <",
            "Cute!",
        ]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert "".join(stitched) == source
        assert "/\\_/\\" in stitched[0]
        assert "\n" in stitched[0]

    def test_following_segment_with_leading_indent_preserved(self):
        # If the LLM emits a segment that starts with leading
        # whitespace (e.g., an ASCII-art line with indent, or an
        # indented code block), the previous segment's trailing-ws
        # extension must NOT consume that leading whitespace -- the
        # follow-up segment's find() would then fail.
        source = (
            "I drew this cat for my daughter today:\n"
            "  /\\_/\\\n"
            " ( o.o )\n"
            "  > ^ <\n"
            "She loved it."
        )
        llm_segments = [
            "I drew this cat for my daughter today:",
            "  /\\_/\\\n ( o.o )\n  > ^ <",
            "She loved it.",
        ]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert "".join(stitched) == source
        # Seg 1 must retain its leading two-space indent.
        assert stitched[1].startswith("  /\\_/\\")
        # Seg 0's extension stops at seg 1's start (the leading space).
        assert stitched[0] == "I drew this cat for my daughter today:\n"

    def test_source_whitespace_used_even_if_llm_normalized(self):
        # Source has a tab; LLM output has a space instead. The stitcher
        # still locates the segment via the parts it can match and
        # re-extracts from source.
        # (Here we simulate a benign case: LLM matches exactly. Real
        # whitespace-normalization fallback is best-effort.)
        source = "Header\tafter\ttabs"
        llm_segments = ["Header\tafter\ttabs"]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert stitched == [source]
        assert "\t" in stitched[0]

    def test_paragraph_breaks_within_segment_preserved(self):
        source = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        llm_segments = [
            "Paragraph one.",
            "Paragraph three.",
        ]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        reconstruction = "".join(stitched)
        assert reconstruction == "Paragraph one.\n\nParagraph three."

    def test_duplicate_segment_text_uses_cursor_to_advance(self):
        source = "yes. no. yes."
        # The LLM legitimately quotes both "yes." occurrences.
        llm_segments = ["yes.", "yes."]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        # Both occurrences should be anchored to distinct positions.
        # Stitched[0] should grab the first "yes. " (with trailing space
        # through "no" being skipped). Actually, trailing whitespace
        # extends to the next non-whitespace, which is "n" in "no". So
        # stitched[0] = "yes. " (up to index 5 where "n" starts).
        assert stitched[0] == "yes. "
        # Cursor advances past first match, so the second "yes." finds
        # the second occurrence.
        assert stitched[1] == "yes."

    def test_empty_segment_skipped(self):
        source = "hello world"
        llm_segments = ["hello", "", "world"]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert stitched == ["hello ", "world"]
        assert "".join(stitched) == source

    def test_paraphrased_segment_padded_with_space(self):
        # LLM paraphrased a segment that isn't in source. We can't
        # anchor it, but we still need a separator so subsequent
        # segments don't glue onto it.
        source = "The original text."
        llm_segments = ["The paraphrased version."]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert stitched == ["The paraphrased version. "]

    def test_paraphrased_segment_followed_by_real_segment(self):
        # The dangerous case: paraphrase has no source-derived trailing
        # whitespace, so without padding the next segment would glue
        # onto it.
        source = "Project budget is $45k. Deadline is March 30."
        llm_segments = [
            "The budget is $45k",  # paraphrase (missing "Project ")
            "Deadline is March 30.",
        ]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        # Paraphrased segment gets a padding space; the next segment is
        # found in source and contributes its own slice.
        assert stitched[0] == "The budget is $45k "
        assert stitched[1] == "Deadline is March 30."
        # Critically: no concat collision.
        assert "$45kDeadline" not in "".join(stitched)

    def test_paraphrased_segment_already_ending_in_whitespace_not_double_padded(
        self,
    ):
        # If the LLM (unusually) returned a paraphrased segment that
        # already ends in whitespace, don't add another space.
        source = "Anchor not present."
        llm_segments = ["paraphrase ending in space "]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert stitched == ["paraphrase ending in space "]

    def test_over_escaped_backslashes_recovered_via_unescape_fallback(self):
        # gpt-5-nano sometimes JSON-over-escapes on output: source has
        # one backslash, model emits two. The stitcher's first
        # source.find() fails, so it strips one escape layer
        # (\\\\->\\, \\n->\n, etc.) and retries. The recovered slice
        # comes from source, so reconstruction is exact.
        source = "I drew this:\n  /\\_/\\\n ( o.o )"
        # Simulate the model output: backslashes are doubled.
        over_escaped_art = "  /\\\\_/\\\\\n ( o.o )"
        llm_segments = ["I drew this:", over_escaped_art]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert "".join(stitched) == source
        # Segment 1 should be the source slice, not the over-escaped
        # LLM string.
        assert "/\\_/\\" in stitched[1]
        assert "/\\\\_/\\\\" not in stitched[1]

    def test_over_escaped_newlines_recovered_via_unescape_fallback(self):
        # Some failure modes also escape newlines (\\n instead of a
        # real newline character). Same fallback should handle it.
        source = "Line one.\nLine two."
        over_escaped = "Line one.\\nLine two."
        llm_segments = [over_escaped]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        # After un-escape, the segment matches source and gets
        # source-sliced.
        assert "".join(stitched) == source

    def test_last_segment_extends_to_trailing_whitespace(self):
        source = "First.\nSecond.\n\n"
        llm_segments = ["First.", "Second."]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert "".join(stitched) == source

    def test_empty_input_returns_empty(self):
        assert LLMTextSegmenter._stitch_segments_to_source([], "source") == []

    def test_llm_dropped_leading_indent_recovered_via_backward_extension(
        self,
    ):
        # If the LLM emits a segment starting at content (dropping
        # the source's leading indent on that line), backward
        # extension walks through the horizontal whitespace and
        # claims it because it is bounded by a newline -- so the
        # indent is restored to the segment.
        source = "Items:\n  - one\n  - two"
        llm_segments = ["Items:", "- one", "- two"]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert "".join(stitched) == source
        assert stitched[1] == "  - one\n"
        assert stitched[2] == "  - two"

    def test_orphan_whitespace_between_newline_groups_dropped(self):
        # SEGA\t\t\t\n\n\t\t\t\n\n\t\tSEGB
        # SEGA forward eats trailing tabs + first newline group.
        # SEGB backward walks the two leading tabs and stops at the
        # second newline group -- newline-bounded -> claim.
        # The middle "\t\t\t\n\n" (a tab-only "blank" line followed
        # by another newline) belongs to neither and is dropped.
        source = "SEGA\t\t\t\n\n\t\t\t\n\n\t\tSEGB"
        llm_segments = ["SEGA", "SEGB"]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert stitched[0] == "SEGA\t\t\t\n\n"
        assert stitched[1] == "\t\tSEGB"
        assert "".join(stitched) == "SEGA\t\t\t\n\n\t\tSEGB"

    def test_same_line_dropped_content_does_not_claim_backward(self):
        # SEGA's trailing space is eaten by forward; SEGB's leading
        # space is NOT claimed by backward because no newline
        # separates it from the dropped content.
        source = "Hi. JUNK World."
        llm_segments = ["Hi.", "World."]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert stitched[0] == "Hi. "
        assert stitched[1] == "World."
        assert "".join(stitched) == "Hi. World."

    def test_first_segment_claims_leading_indent_at_start_of_source(
        self,
    ):
        # If the first kept segment has leading whitespace before
        # it at the start of source, backward extension claims it
        # (bounded by start-of-source, not by a newline -- but the
        # same rule applies).
        source = "  Hello.\nGoodbye."
        llm_segments = ["Hello.", "Goodbye."]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        assert "".join(stitched) == source
        assert stitched[0] == "  Hello.\n"

    def test_email_with_dropped_envelope(self):
        source = (
            "Hi team, As discussed, please find attached the proposal. "
            "Project budget is $45k. Deadline is March 30. "
            "Let me know if any questions. Thanks, Sarah"
        )
        llm_segments = [
            "As discussed, please find attached the proposal.",
            "Project budget is $45k.",
            "Deadline is March 30.",
            "Thanks, Sarah",
        ]

        stitched = LLMTextSegmenter._stitch_segments_to_source(llm_segments, source)

        reconstruction = "".join(stitched)
        assert "Hi team" not in reconstruction
        assert "Let me know" not in reconstruction
        assert reconstruction == (
            "As discussed, please find attached the proposal. "
            "Project budget is $45k. Deadline is March 30. "
            "Thanks, Sarah"
        )
