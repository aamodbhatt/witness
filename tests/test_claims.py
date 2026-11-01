from __future__ import annotations

import pytest

from witness.claims.extract import decontextualise, extract_claims, page_subject, skip_reason
from witness.claims.readable import readable_text
from witness.config import ClaimsConfig

CFG = ClaimsConfig()


@pytest.mark.parametrize(
    ("sentence", "reason"),
    [
        ("Is Marie Curie the only person to win two Nobel Prizes?", "question"),
        ("She won it.", "too short"),
        ("I think Marie Curie was the greatest scientist of her era.", "opinion"),
        ("Subscribe to our newsletter for more stories like this one.", "boilerplate"),
        ("Arguably, the Eiffel Tower is the best landmark in Paris.", "opinion"),
        ("Marie Curie won the Nobel Prize in Physics in 1903.", None),
    ],
)
def test_check_worthiness(sentence: str, reason: str | None) -> None:
    assert skip_reason(sentence, CFG) == reason


def test_page_subject_strips_site_names() -> None:
    assert page_subject("Marie Curie - Wikipedia") == "Marie Curie"
    assert page_subject("Inception (2010 film)") == "Inception"
    assert page_subject("Marie Curie | Field Notes") == "Marie Curie"
    assert page_subject(None) is None


def test_decontextualise_pronouns_and_generic_noun_phrases() -> None:
    assert decontextualise("She was born in Warsaw.", "Marie Curie") == (
        "Marie Curie was born in Warsaw."
    )
    assert decontextualise("Her husband was Pierre Curie.", "Marie Curie") == (
        "Marie Curie's husband was Pierre Curie."
    )
    assert decontextualise("The film was directed by Christopher Nolan.", "Inception") == (
        "Inception was directed by Christopher Nolan."
    )
    assert decontextualise("Then she left.", "Marie Curie") == "Then she left."


def test_extract_claims_offsets_and_subject_tracking() -> None:
    paragraphs = [
        "Marie Curie was a physicist. She was born in Warsaw in 1867. Do you agree?",
        "Pierre Curie was a French physicist. He shared the 1903 Nobel Prize.",
    ]
    claims, skipped = extract_claims(paragraphs, "Marie Curie - Wikipedia", CFG)
    for c in claims:
        assert paragraphs[c.paragraph][c.start : c.end] == c.text
    queries = [c.query for c in claims]
    assert "Marie Curie was born in Warsaw in 1867." in queries
    assert "Pierre Curie shared the 1903 Nobel Prize." in queries
    assert any(s.reason == "question" for s in skipped)


def test_claim_limit_is_enforced() -> None:
    paragraphs = [f"Paris had {i} bridges in the year 19{i:02d}." for i in range(10)]
    claims, skipped = extract_claims(paragraphs, None, ClaimsConfig(max_claims_per_page=3))
    assert len(claims) == 3
    assert sum(s.reason == "page claim limit" for s in skipped) == 7


def test_readable_text_keeps_article_paragraphs_only() -> None:
    html = """<html><head><title>Marie Curie - Example</title></head><body>
      <nav><p>Home | About | A very long navigation paragraph that should vanish.</p></nav>
      <article><h1>Marie Curie</h1>
        <p>Marie Curie was a Polish and naturalised-French physicist and chemist.</p>
        <p>Short.</p>
        <ul><li>She was the first woman to win a Nobel Prize in any field.</li></ul>
        <script>var x = "not text at all, but long enough to be kept by mistake";</script>
      </article>
      <footer><p>Copyright 2026, all rights reserved, by an example publisher.</p></footer>
    </body></html>"""
    page = readable_text(html)
    assert page.title == "Marie Curie - Example"
    assert page.paragraphs == [
        "Marie Curie was a Polish and naturalised-French physicist and chemist.",
        "She was the first woman to win a Nobel Prize in any field.",
    ]
