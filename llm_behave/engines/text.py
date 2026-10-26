"""Text utilities shared by the engines and assertions. Pure Python, no ML."""

from __future__ import annotations

import re

# A sentence ends at ., ! or ? only when followed by whitespace -- so "$4.99",
# "v2.1" and "example.com" stay intact. Blank lines and list items also start
# a new sentence, since LLMs often emit bullet points without trailing periods.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+"
    r"|\n\s*\n"
    r"|\n(?=\s*(?:[-*•]|\d+[.)])\s)"
)

# Abbreviations whose trailing period must not end a sentence ("e.g. section 2").
_ABBREVIATIONS = frozenset({
    "e.g.", "i.e.", "etc.", "vs.", "approx.", "dr.", "mr.", "mrs.", "ms.",
    "prof.", "st.", "no.", "inc.", "ltd.", "co.", "corp.", "jr.", "sr.",
})

_TRAILING_PUNCTUATION = ".!?"

# A chunk that is only a list number ("1." / "2)") belongs to the next chunk.
_LIST_NUMBER = re.compile(r"\d+[.)]")


def _is_continuation(chunk: str) -> bool:
    """True if a split after `chunk` was a false sentence boundary."""
    last_word = chunk.split()[-1]
    return last_word.lower() in _ABBREVIATIONS or bool(_LIST_NUMBER.fullmatch(chunk))


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty, trimmed sentences.

    Trailing sentence punctuation is stripped from each sentence. Returns
    [text] for non-empty text with no boundaries, and [] for blank text.
    """
    chunks = [c.strip() for c in _SENTENCE_BOUNDARY.split(text) if c and c.strip()]

    merged: list[str] = []
    for chunk in chunks:
        if merged and _is_continuation(merged[-1]):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)

    sentences = [s.rstrip(_TRAILING_PUNCTUATION).strip() for s in merged]
    sentences = [s for s in sentences if s]
    if not sentences and text.strip():
        return [text.strip()]
    return sentences
