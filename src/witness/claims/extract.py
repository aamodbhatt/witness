"""From page paragraphs to check-worthy, self-contained claims with exact offsets.

Three steps per sentence:

1. **Segmentation** with character spans into the paragraph string, so the browser can
   underline exactly the text it sent.
2. **Check-worthiness filtering**: questions, first/second-person opinions, hedged or
   evaluative statements, boilerplate and fragments are skipped, each with a reason.
3. **Decontextualisation**: a sentence-initial pronoun ("He", "Its") or generic noun
   phrase ("The film") is replaced by the entity it refers to: the subject of the
   paragraph so far, else the page subject from the title. Only the *query* sent to the
   verifier changes; the underline stays on the original words.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from witness.claims.segment import split_sentences
from witness.config import ClaimsConfig

_BOILERPLATE = re.compile(
    r"\b(cookies?|subscribe|newsletter|sign (up|in)|log ?in|all rights reserved|copyright"
    r"|click here|read more|share this|follow us|terms of (service|use)|privacy policy"
    r"|advertisement|sponsored|related articles?|comments?)\b",
    re.IGNORECASE,
)
_PERSONAL = re.compile(r"\b(I|I'm|I've|me|my|mine|we|we're|us|our|ours|you|your|yours)\b")
_SUBJECTIVE = re.compile(
    r"\b(think|believe|feel|opinion|should|ought|best|worst|amazing|awesome|terrible"
    r"|wonderful|beautiful|ugly|favou?rite|love|hate|perhaps|maybe|arguably|probably"
    r"|seems?|apparently|hopefully|unfortunately|fortunately|incredible|stunning)\b",
    re.IGNORECASE,
)
_HAS_ANCHOR = re.compile(r"(?<!^)\b[A-Z][a-z]|\d")
_WORD = re.compile(r"\w+(?:['’]\w+)?")

_SUBJECT_PRONOUN = re.compile(r"^(He|She|It|They)\b")
_POSSESSIVE_PRONOUN = re.compile(r"^(His|Her|Its|Their)\b")
_GENERIC_NP = re.compile(
    r"^The (film|movie|album|song|single|band|group|company|firm|city|town|village|river"
    r"|book|novel|series|show|game|team|club|building|university|school|species|country"
    r"|party|station|airport|organi[sz]ation|ship|mountain|lake|island|museum|church)\b"
)
_LEADING_NAME = re.compile(
    r"^((?:[A-Z][\w'’.-]+)(?:\s+(?:of|de|von|van|the|and|&)?\s*[A-Z][\w'’.-]+){0,5})\s+"
    r"(is|was|are|were|has|had|became|served|won|played|wrote|founded|released)\b"
)
# "Article - Site", "Article | Site", "Article — Site": the article name comes first.
_TITLE_SEPARATOR = re.compile(r"\s+[|\-–—]\s+")
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")


@dataclass(frozen=True, slots=True)
class Claim:
    id: str
    paragraph: int
    start: int
    end: int
    text: str  # the sentence exactly as it appears on the page
    query: str  # the decontextualised sentence sent to the verifier


@dataclass(frozen=True, slots=True)
class Skipped:
    paragraph: int
    start: int
    end: int
    reason: str


def page_subject(title: str | None) -> str | None:
    """``"Marie Curie - Wikipedia"`` -> ``"Marie Curie"``; ``None`` if nothing is left."""
    if not title:
        return None
    first = _TITLE_SEPARATOR.split(title.strip(), maxsplit=1)[0]
    subject = _PARENTHETICAL.sub("", first).strip()
    return subject or None


def skip_reason(sentence: str, config: ClaimsConfig) -> str | None:
    """Why ``sentence`` is not worth checking, or ``None`` if it is."""
    words = _WORD.findall(sentence)
    if sentence.rstrip().endswith("?"):
        return "question"
    if len(words) < config.min_words:
        return "too short"
    if len(words) > config.max_words:
        return "too long"
    if _BOILERPLATE.search(sentence):
        return "boilerplate"
    if _PERSONAL.search(sentence) or _SUBJECTIVE.search(sentence):
        return "opinion"
    return None


def decontextualise(sentence: str, subject: str | None) -> str:
    """Replace a sentence-initial pronoun or generic noun phrase with ``subject``."""
    if not subject:
        return sentence
    if _POSSESSIVE_PRONOUN.match(sentence):
        return _POSSESSIVE_PRONOUN.sub(f"{subject}'s", sentence, count=1)
    if _SUBJECT_PRONOUN.match(sentence):
        return _SUBJECT_PRONOUN.sub(subject, sentence, count=1)
    if _GENERIC_NP.match(sentence):
        return _GENERIC_NP.sub(subject, sentence, count=1)
    return sentence


def extract_claims(
    paragraphs: Sequence[str], title: str | None, config: ClaimsConfig
) -> tuple[list[Claim], list[Skipped]]:
    """Check-worthy claims (with offsets into their paragraph) and the skipped sentences."""
    claims: list[Claim] = []
    skipped: list[Skipped] = []
    default_subject = page_subject(title)
    for p_index, paragraph in enumerate(paragraphs):
        subject = default_subject
        for span in split_sentences(paragraph):
            sentence = paragraph[span.start : span.end]
            reason = skip_reason(sentence, config)
            query = decontextualise(sentence, subject) if reason is None else sentence
            anchored = query != sentence or _HAS_ANCHOR.search(query) is not None
            if reason is None and not anchored:
                reason = "nothing checkable (no name or number)"
            leading = _LEADING_NAME.match(sentence)
            if leading:
                subject = leading.group(1)
            if reason is not None:
                skipped.append(Skipped(p_index, span.start, span.end, reason))
                continue
            if len(claims) >= config.max_claims_per_page:
                skipped.append(Skipped(p_index, span.start, span.end, "page claim limit"))
                continue
            claims.append(Claim(f"c{len(claims)}", p_index, span.start, span.end, sentence, query))
    return claims, skipped
