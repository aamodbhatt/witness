"""Rule-based sentence segmentation that returns character spans.

Why not a statistical splitter: the extension underlines text in place, so what matters
is exact, reproducible offsets into the paragraph string the browser sent. A transparent
rule set with an abbreviation list is deterministic, dependency-free and property-tested;
its known blind spot (an abbreviation that really does end a sentence) only merges two
sentences into one claim.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open character interval ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")

    def __len__(self) -> int:
        return self.end - self.start


_ABBREVIATIONS: frozenset[str] = frozenset(
    """
    mr mrs ms dr prof sr jr st mt ft vs etc e.g i.e cf al approx dept est fig inc ltd co
    corp no nos vol vols ed eds rev gen col lt sgt capt cmdr adm gov sen rep pres jan feb
    mar apr jun jul aug sep sept oct nov dec u.s u.k u.n a.m p.m ph.d b.a m.a d.c op pp ca
    """.split()  # noqa: SIM905 - a word table reads better as prose
)

# Candidate boundary: terminal punctuation, optional closing quotes/brackets, whitespace.
_BOUNDARY_RE = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)")
_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n")
_NEXT_START_RE = re.compile(r"\s+[\"'“‘(\[]*[A-Z0-9À-Ö]")
_TOKEN_BEFORE_RE = re.compile(r"([\w.]+)$")


def _is_abbreviation(text: str, punct_start: int) -> bool:
    """True if the period at ``punct_start`` ends an abbreviation or an initial."""
    if text[punct_start] != ".":
        return False
    match = _TOKEN_BEFORE_RE.search(text, 0, punct_start)
    if match is None:
        return False
    token = match.group(1).lower().rstrip(".")
    if token in _ABBREVIATIONS:
        return True
    # Single-letter initials such as "J. K. Rowling" or "Franklin D. Roosevelt".
    return len(token) == 1 and token.isalpha()


def _strip(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start, end) if end > start else None


def split_sentences(text: str) -> list[Span]:
    """Split ``text`` into sentence spans.

    Invariants (checked by property tests): spans are sorted, disjoint, non-empty,
    have no surrounding whitespace, and together cover every non-whitespace character.
    """
    cuts: set[int] = {0, len(text)}
    for m in _PARAGRAPH_RE.finditer(text):
        cuts.add(m.start())
    for m in _BOUNDARY_RE.finditer(text):
        end = m.end()
        if _is_abbreviation(text, m.start()):
            continue
        if _NEXT_START_RE.match(text, end) is None:
            continue
        cuts.add(end)
    ordered = sorted(cuts)
    spans: list[Span] = []
    for a, b in itertools.pairwise(ordered):
        span = _strip(text, a, b)
        if span is not None:
            spans.append(span)
    return spans
