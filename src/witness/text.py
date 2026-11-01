"""Small, dependency-free text utilities shared across modules."""

from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"[0-9a-z]+(?:'[a-z]+)?")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

# A compact English stopword list (function words only). Kept local so that sparse
# retrieval, the entity linker and the test fakes tokenize identically.
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is
    it its itself just me more most my myself no nor not now of off on once only or other
    our ours ourselves out over own same she should so some such than that the their theirs
    them themselves then there these they this those through to too under until up very was
    we were what when where which while who whom why will with you your yours yourself
    yourselves also would could may might shall must s t
    """.split()  # noqa: SIM905 - a word table reads better as prose
)


def fold(text: str) -> str:
    """Lowercase and strip accents, so that 'Beyoncé' and 'beyonce' compare equal."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().replace("’", "'")


def words(text: str) -> list[str]:
    """Case-folded word tokens (letters, digits, internal apostrophes)."""
    return _WORD_RE.findall(fold(text))


def content_words(text: str) -> list[str]:
    """Word tokens without stopwords."""
    return [w for w in words(text) if w not in STOPWORDS]


def alias_key(text: str) -> str:
    """Canonical lookup key for entity names: folded, punctuation collapsed to spaces.

    ``"Savages (2012 film)"`` and ``"savages 2012 film"`` share a key, so titles, anchor
    texts and claim n-grams can be matched with a plain equality lookup.
    """
    return _NON_ALNUM.sub(" ", fold(text)).strip()


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())
