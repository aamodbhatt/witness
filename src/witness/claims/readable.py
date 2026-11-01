"""Readable-text extraction for ``witness check-url``: the article's paragraphs, no chrome.

The browser extension does this in the DOM itself (it needs the nodes to underline);
this is the server-side equivalent for pages fetched by URL.
"""

from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

_DROP = ("script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg",
         "template", "figure", "table", "sup")  # fmt: skip
_BLOCKS = ("p", "li", "blockquote")
_MIN_CHARS = 40


@dataclass(frozen=True, slots=True)
class ReadablePage:
    title: str | None
    paragraphs: list[str]


def _main_root(soup: BeautifulSoup) -> Tag:
    for selector in ("article", "main", "[role=main]", "#mw-content-text", "#content"):
        found = soup.select_one(selector)
        if isinstance(found, Tag):
            return found
    return soup.body if isinstance(soup.body, Tag) else soup


def readable_text(html: str) -> ReadablePage:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("meta", property="og:title")
    title: str | None = None
    if isinstance(title_tag, Tag) and title_tag.get("content"):
        title = str(title_tag["content"]).strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()
    for tag in soup.find_all(_DROP):
        tag.decompose()
    paragraphs: list[str] = []
    for block in _main_root(soup).find_all(_BLOCKS):
        if block.find_parent(_BLOCKS) is not None:
            continue  # nested blocks are covered by their parent
        text = " ".join(block.get_text(" ", strip=True).split())
        if len(text) >= _MIN_CHARS:
            paragraphs.append(text)
    return ReadablePage(title, paragraphs)
