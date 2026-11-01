"""Entity linking from claim n-grams to Wikipedia pages.

FEVER claims are entity-centric ("Soul Food was released by Fox 2000 Pictures"), so the
strongest document retriever is a lookup, not a search: every n-gram of the claim is
normalised and matched against three alias sources built from the dump itself:

* the page title (``soul food``),
* the title without its disambiguation suffix (``soul food`` -> ``Soul Food (film)``),
* hyperlink anchor texts, with link counts giving a commonness prior p(page | text).

A disambiguated page gets a bonus when its suffix words occur in the claim ("film"
selects ``Soul Food (film)`` over the soundtrack). Longer, capitalised mentions are more
specific and weigh more. Everything is integer lookups in SQLite, so linking costs
milliseconds and needs no model.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from witness.config import LinkerConfig
from witness.text import STOPWORDS, alias_key, content_words
from witness.wiki.store import ALIAS_ANCHOR, ALIAS_BASE, ALIAS_TITLE, AliasHit

_TOKEN = re.compile(r"[\w]+(?:[-'’.&][\w]+)*", re.UNICODE)
_SUFFIX = re.compile(r"\(([^()]*)\)\s*$")


@dataclass(frozen=True, slots=True)
class Mention:
    start: int
    end: int
    key: str
    words: int
    capitalised: bool

    @property
    def weight(self) -> float:
        return self.words * (1.0 if self.capitalised else 0.5)


class AliasLookup(Protocol):
    """The two store queries the linker needs (a small seam for tests)."""

    def aliases(self, keys: Iterable[str]) -> list[AliasHit]: ...

    def titles(self, pages: Iterable[int]) -> dict[int, str]: ...


@dataclass(frozen=True, slots=True)
class LinkedPage:
    page: int
    score: float
    mention: str  # the alias key that produced the best score


def find_mentions(text: str, max_ngram: int) -> list[Mention]:
    """Candidate mentions: all n-grams that contain a capitalised word or a digit.

    Lower-case n-grams are kept only when they span two or more words, so "film" alone
    never links but "climate change" can.
    """
    tokens = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
    out: list[Mention] = []
    for i in range(len(tokens)):
        for n in range(1, min(max_ngram, len(tokens) - i) + 1):
            start, end = tokens[i][0], tokens[i + n - 1][1]
            span = text[start:end]
            key = alias_key(span)
            if len(key) < 2 or key in STOPWORDS:
                continue
            words = [text[a:b] for a, b in tokens[i : i + n]]
            capitalised = any(w[:1].isupper() or w[:1].isdigit() for w in words)
            if not capitalised and n == 1:
                continue
            last = words[-1]
            if last.lower() in STOPWORDS and not last[:1].isupper():
                continue  # "Obama was" is never a better name than "Obama"; "Shut Up" is
            out.append(Mention(start, end, key, n, capitalised))
    return out


class EntityLinker:
    """Ranks pages for a claim using the alias table of a :class:`WikiStore`."""

    def __init__(
        self,
        lookup: AliasLookup,
        inlinks: NDArray[np.int32],
        config: LinkerConfig,
    ) -> None:
        self._lookup = lookup
        self._inlinks = inlinks
        self.config = config

    def link(self, claim: str, k: int | None = None) -> list[LinkedPage]:
        k = k or self.config.top_k
        mentions = find_mentions(claim, self.config.max_ngram)
        if not mentions:
            return []
        best_mention: dict[str, Mention] = {}
        for m in mentions:
            if m.key not in best_mention or m.weight > best_mention[m.key].weight:
                best_mention[m.key] = m
        hits = self._lookup.aliases(best_mention)
        claim_words = set(content_words(claim))
        titles = self._lookup.titles({h.page for h in hits if h.kind == ALIAS_BASE})

        # A mention nested inside a longer mention that is itself a page title ("Johann"
        # inside "Johann Wolfgang von Goethe") is probably not a separate entity.
        titled = [best_mention[h.key] for h in hits if h.kind == ALIAS_TITLE]

        def weight_of(key: str) -> float:
            m = best_mention[key]
            nested = any(
                t.words > m.words and t.start <= m.start and m.end <= t.end for t in titled
            )
            return m.weight * (0.5 if nested else 1.0)

        by_key: dict[tuple[str, int], list[AliasHit]] = defaultdict(list)
        for hit in hits:
            by_key[(hit.key, hit.kind)].append(hit)
        scores: dict[int, tuple[float, str]] = {}

        def offer(page: int, score: float, key: str) -> None:
            if page not in scores or score > scores[page][0]:
                scores[page] = (score, key)

        limit = self.config.pages_per_mention
        for (key, kind), group in by_key.items():
            weight = weight_of(key)
            if kind == ALIAS_TITLE:
                for hit in group:
                    offer(hit.page, weight * 1.0, key)
            elif kind == ALIAS_BASE:
                ranked = sorted(group, key=lambda h: -self._popularity(h.page))
                kept = 0
                for hit in ranked:
                    suffix = _SUFFIX.search(titles.get(hit.page, ""))
                    if suffix is None or suffix.group(1) == "disambiguation":
                        continue  # list pages are never evidence
                    suffix_words = set(content_words(suffix.group(1)))
                    overlap = len(suffix_words & claim_words) / max(len(suffix_words), 1)
                    offer(hit.page, weight * (0.75 + 0.4 * overlap), key)
                    kept += 1
                    if kept >= limit * 2:
                        break
            elif kind == ALIAS_ANCHOR:
                total = sum(h.count for h in group)
                for hit in sorted(group, key=lambda h: -h.count)[:limit]:
                    offer(hit.page, weight * 0.7 * hit.count / total, key)
        ranked_pages = sorted(
            scores.items(), key=lambda kv: (-kv[1][0], -self._popularity(kv[0]), kv[0])
        )
        return [LinkedPage(p, s, key) for p, (s, key) in ranked_pages[:k]]

    def _popularity(self, page: int) -> int:
        return int(self._inlinks[page]) if page < len(self._inlinks) else 0
