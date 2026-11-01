from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from witness.claims.segment import split_sentences


def _texts(text: str) -> list[str]:
    return [text[s.start : s.end] for s in split_sentences(text)]


def test_splits_on_terminal_punctuation() -> None:
    assert _texts("Paris is in France. Berlin is in Germany! Is Rome in Italy?") == [
        "Paris is in France.",
        "Berlin is in Germany!",
        "Is Rome in Italy?",
    ]


def test_abbreviations_and_initials_do_not_split() -> None:
    text = "Dr. Smith met J. K. Rowling in the U.S. last year. They talked."
    assert _texts(text) == ["Dr. Smith met J. K. Rowling in the U.S. last year.", "They talked."]


@given(st.text(alphabet=st.characters(codec="utf-8"), max_size=300))
def test_spans_are_sorted_disjoint_trimmed_and_cover_all_text(text: str) -> None:
    spans = split_sentences(text)
    covered = [False] * len(text)
    previous_end = 0
    for span in spans:
        assert previous_end <= span.start < span.end <= len(text)
        piece = text[span.start : span.end]
        assert piece == piece.strip()
        for i in range(span.start, span.end):
            covered[i] = True
        previous_end = span.end
    for i, ch in enumerate(text):
        if not ch.isspace():
            assert covered[i], f"character {ch!r} at {i} not in any sentence"
