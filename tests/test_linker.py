from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from witness.config import LinkerConfig
from witness.retrieval.linker import EntityLinker, find_mentions
from witness.text import alias_key
from witness.wiki.store import ALIAS_ANCHOR, ALIAS_BASE, ALIAS_TITLE, AliasHit

TITLES = {
    0: "Soul Food",
    1: "Soul Food (film)",
    2: "Soul Food (soundtrack)",
    3: "Fox 2000 Pictures",
    4: "Fox",
    5: "Soul Food (disambiguation)",
    6: "Johann Wolfgang von Goethe",
    7: "Johann",
}
ANCHORS = [("fox 2000", 3, 9), ("fox", 4, 5), ("goethe", 6, 40)]


class FakeLookup:
    def __init__(self) -> None:
        self.rows: list[AliasHit] = []
        for page, title in TITLES.items():
            self.rows.append(AliasHit(alias_key(title), page, ALIAS_TITLE, 1))
            if "(" in title:
                base = title.split(" (")[0]
                self.rows.append(AliasHit(alias_key(base), page, ALIAS_BASE, 1))
        for key, page, count in ANCHORS:
            self.rows.append(AliasHit(key, page, ALIAS_ANCHOR, count))

    def aliases(self, keys: Iterable[str]) -> list[AliasHit]:
        wanted = set(keys)
        return [r for r in self.rows if r.key in wanted]

    def titles(self, pages: Iterable[int]) -> dict[int, str]:
        return {p: TITLES[p] for p in pages}


def _linker() -> EntityLinker:
    inlinks = np.asarray([50, 30, 5, 10, 100, 1, 80, 2], dtype=np.int32)
    return EntityLinker(FakeLookup(), inlinks, LinkerConfig(top_k=10))


def test_title_and_disambiguated_pages_are_found() -> None:
    pages = [p.page for p in _linker().link("Fox 2000 Pictures released the film Soul Food.")]
    assert pages[0] == 3  # the longest exact title wins
    assert {0, 1}.issubset(pages)
    assert 5 not in pages  # disambiguation lists are never evidence


def test_suffix_words_in_claim_promote_the_matching_page() -> None:
    ranked = [p.page for p in _linker().link("Soul Food is a film.")]
    assert ranked.index(1) < ranked.index(2)  # "(film)" beats "(soundtrack)"


def test_nested_mentions_are_down_weighted() -> None:
    linked = {p.page: p.score for p in _linker().link("Johann Wolfgang von Goethe wrote Faust.")}
    assert linked[6] > linked[7]


def test_anchor_commonness_links_aliases() -> None:
    pages = [p.page for p in _linker().link("Goethe wrote poems.")]
    assert pages == [6]


def test_lowercase_single_words_never_link() -> None:
    assert _linker().link("the film was released") == []


def test_mentions_require_capital_or_digit_for_single_words() -> None:
    keys = {m.key for m in find_mentions("Shut Up is a song from 2016", 4)}
    assert "shut up" in keys
    assert "2016" in keys
    assert "song" not in keys


@given(st.text(max_size=80))
def test_mention_spans_are_valid(text: str) -> None:
    for m in find_mentions(text, 5):
        assert 0 <= m.start < m.end <= len(text)
        assert m.key == alias_key(text[m.start : m.end])
