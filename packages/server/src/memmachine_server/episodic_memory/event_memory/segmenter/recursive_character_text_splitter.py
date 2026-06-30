"""Dependency-free recursive character text splitter.

A faithful reimplementation of the only piece of ``langchain_text_splitters``
EventMemory used — ``RecursiveCharacterTextSplitter`` — so importing the
segmenter no longer drags in ``langchain_text_splitters`` -> ``transformers`` ->
``torch`` (~490 MB of import footprint in every process that imports the engine).

Ported verbatim from ``langchain_text_splitters`` (base ``TextSplitter`` +
``RecursiveCharacterTextSplitter``); behavioral parity is verified by a
differential test against the original. Uses only the standard library (``re``).
"""

import logging
import re
from collections.abc import Callable, Iterable
from typing import Literal

logger = logging.getLogger(__name__)


def _split_text_with_regex(
    text: str,
    separator: str,
    *,
    keep_separator: bool | Literal["start", "end"],
) -> list[str]:
    """Split ``text`` on ``separator`` (a regex), optionally keeping it."""
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            splits_ = re.split(f"({separator})", text)
            splits = (
                [splits_[i] + splits_[i + 1] for i in range(0, len(splits_) - 1, 2)]
                if keep_separator == "end"
                else [splits_[i] + splits_[i + 1] for i in range(1, len(splits_), 2)]
            )
            if len(splits_) % 2 == 0:
                splits += splits_[-1:]
            splits = (
                [*splits, splits_[-1]]
                if keep_separator == "end"
                else [splits_[0], *splits]
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s]


class RecursiveCharacterTextSplitter:
    """Recursively split text by a prioritized list of separators.

    Drop-in for ``langchain_text_splitters.RecursiveCharacterTextSplitter`` for
    the configuration EventMemory uses (and faithful to it in general).
    """

    def __init__(
        self,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,
        is_separator_regex: bool = False,
        *,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        strip_whitespace: bool = True,
    ) -> None:
        """Create a new splitter (defaults mirror langchain's)."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._keep_separator = keep_separator
        self._is_separator_regex = is_separator_regex
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._strip_whitespace = strip_whitespace

    def split_text(self, text: str) -> list[str]:
        """Split ``text`` into chunks no larger than ``chunk_size`` where possible."""
        return self._split_text(text, self._separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        # Get appropriate separator to use.
        separator = separators[-1]
        new_separators: list[str] = []
        for i, s_ in enumerate(separators):
            separator_ = s_ if self._is_separator_regex else re.escape(s_)
            if not s_:
                separator = s_
                break
            if re.search(separator_, text):
                separator = s_
                new_separators = separators[i + 1 :]
                break

        separator_ = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(
            text, separator_, keep_separator=self._keep_separator
        )

        # Now go merging things, recursively splitting longer texts.
        good_splits: list[str] = []
        merge_separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self._merge_splits(good_splits, merge_separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if good_splits:
            merged_text = self._merge_splits(good_splits, merge_separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def _join_docs(self, docs: list[str], separator: str) -> str | None:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        return text or None

    def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
        # Combine smaller pieces into chunks up to chunk_size.
        separator_len = self._length_function(separator)

        docs: list[str] = []
        current_doc: list[str] = []
        total = 0
        for d in splits:
            len_ = self._length_function(d)
            if (
                total + len_ + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        "Created a chunk of size %d, which is longer than the "
                        "specified %d",
                        total,
                        self._chunk_size,
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep popping while the chunk exceeds the overlap (or is
                    # still too large with the next piece added).
                    while total > self._chunk_overlap or (
                        total + len_ + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += len_ + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs
