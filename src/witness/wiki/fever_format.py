"""Parsing the FEVER 2017 Wikipedia dump (``wiki-pages/wiki-NNN.jsonl``).

Each JSON line holds one page: ``id`` is the page title with underscores and PTB escapes
(``Savages_-LRB-2012_film-RRB-``), and ``lines`` is a newline-separated list of
``"<n>\\t<sentence>\\t<anchor>\\t<target>\\t<anchor>\\t<target>..."`` records. Sentence
numbers are what FEVER gold evidence points at, so they are preserved exactly, including
the empty slots the dump leaves between paragraphs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_PTB = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "-COLON-": ":",
    "``": '"',
    "''": '"',
    "--": "–",
}
_PTB_RE = re.compile("|".join(re.escape(k) for k in sorted(_PTB, key=len, reverse=True)))
_SPACE_BEFORE = re.compile(r" ([,.;:!?%)\]}]|'s\b|'re\b|'ve\b|'ll\b|'d\b|'m\b|n't\b)")
_SPACE_AFTER = re.compile(r"([(\[{$]) ")
_QUOTED = re.compile(r'" ([^"]*?) "')
_EMPTY_PARENS = re.compile(r"\(\s*[;,]?\s*\)")
_PLURAL_POSSESSIVE = re.compile(r"(?<=s) '(?= )")


@dataclass(frozen=True, slots=True)
class WikiLine:
    """One numbered sentence slot of a page. ``text`` is empty for the dump's blank slots."""

    text: str
    links: tuple[str, ...]  # hyperlink target titles mentioned in this sentence
    anchors: tuple[tuple[str, str], ...]  # (surface text, target title) pairs


@dataclass(frozen=True, slots=True)
class WikiPage:
    raw_title: str  # FEVER page id, e.g. "Savages_-LRB-2012_film-RRB-"
    lines: tuple[WikiLine, ...]

    @property
    def title(self) -> str:
        return decode_title(self.raw_title)


def decode_title(raw_title: str) -> str:
    """``"Savages_-LRB-2012_film-RRB-"`` -> ``"Savages (2012 film)"``."""
    return _PTB_RE.sub(lambda m: _PTB[m.group(0)], raw_title.replace("_", " "))


def encode_title(title: str) -> str:
    """Inverse of :func:`decode_title` for the escapes that occur in FEVER page ids."""
    out = title.replace(" ", "_")
    for raw, char in (("-LRB-", "("), ("-RRB-", ")"), ("-COLON-", ":")):
        out = out.replace(char, raw)
    return out


def detokenize(sentence: str) -> str:
    """Turn the dump's PTB-tokenized text into readable text.

    ``"Coster-Waldau -LRB- born 27 July 1970 -RRB- is a Danish actor ."`` becomes
    ``"Coster-Waldau (born 27 July 1970) is a Danish actor."``. Models see this form, which
    is closer to the natural text they were pre-trained on.
    """
    text = _PTB_RE.sub(lambda m: _PTB[m.group(0)], sentence)
    text = _QUOTED.sub(r'"\1"', text)
    text = _SPACE_BEFORE.sub(r"\1", text)
    text = _SPACE_AFTER.sub(r"\1", text)
    text = _EMPTY_PARENS.sub("", text)
    text = _PLURAL_POSSESSIVE.sub("'", text)
    return " ".join(text.split())


def parse_lines(lines_field: str) -> tuple[WikiLine, ...]:
    """Parse the ``lines`` field; the i-th element is sentence number i."""
    if not lines_field:
        return ()
    out: list[WikiLine] = []
    for record in lines_field.split("\n"):
        parts = record.split("\t")
        if not parts[0].isdigit():
            # A stray newline inside a sentence: glue it onto the previous slot.
            if out:
                prev = out[-1]
                out[-1] = WikiLine(f"{prev.text} {detokenize(record)}".strip(), prev.links, ())
            continue
        index = int(parts[0])
        while len(out) < index:  # the dump occasionally skips numbers
            out.append(WikiLine("", (), ()))
        text = detokenize(parts[1]) if len(parts) > 1 else ""
        pairs = parts[2:]
        anchors = tuple(
            (pairs[i], pairs[i + 1]) for i in range(0, len(pairs) - 1, 2) if pairs[i + 1]
        )
        out.append(WikiLine(text, tuple(dict.fromkeys(t for _, t in anchors)), anchors))
    return tuple(out)


def parse_page(json_line: str) -> WikiPage | None:
    """Parse one dump line; returns ``None`` for pages without any sentence text."""
    record = json.loads(json_line)
    raw_title = str(record.get("id") or "")
    if not raw_title or not record.get("text"):
        return None
    lines = parse_lines(str(record.get("lines") or ""))
    if not any(line.text for line in lines):
        return None
    return WikiPage(raw_title, lines)
