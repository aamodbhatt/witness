"""Read-only access to the processed Wikipedia snapshot (one SQLite file).

Pages are addressed by a dense integer id ``0..N-1`` shared with the BM25 matrix rows, so
a retrieval hit maps to its sentences with one primary-key lookup. SQLite is in the
standard library on every platform and memory-maps cleanly, which keeps a 5M-page store
usable on a laptop.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from witness.errors import IndexStoreError
from witness.wiki.fever_format import decode_title

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY,
    raw_title TEXT NOT NULL,
    sentences TEXT NOT NULL,
    links TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS aliases (
    key TEXT NOT NULL,
    page INTEGER NOT NULL,
    kind INTEGER NOT NULL,
    count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS pages_title ON pages(raw_title);
CREATE INDEX IF NOT EXISTS aliases_key ON aliases(key, kind, page, count);
"""

# Alias kinds, strongest first.
ALIAS_TITLE = 0  # the normalised page title itself
ALIAS_BASE = 1  # the title without its disambiguation suffix: "Savages" for "Savages (2012 film)"
ALIAS_ANCHOR = 2  # hyperlink anchor text pointing at the page


@dataclass(frozen=True, slots=True)
class Page:
    id: int
    raw_title: str
    sentences: tuple[str, ...]  # index i is FEVER sentence number i ("" for blank slots)
    links: tuple[tuple[str, ...], ...]  # hyperlink targets per sentence

    @property
    def title(self) -> str:
        return decode_title(self.raw_title)

    @property
    def url(self) -> str:
        return "https://en.wikipedia.org/wiki/" + self.title.replace(" ", "_")


@dataclass(frozen=True, slots=True)
class AliasHit:
    key: str
    page: int
    kind: int
    count: int


class WikiStore:
    """Thread-safe read-only view of ``wiki.sqlite`` (one connection per thread)."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise IndexStoreError(
                f"Wikipedia store not found at {path}; build it with `witness build-index`"
            )
        self.path = path
        self._local = threading.local()
        self.size = int(self._scalar("SELECT value FROM meta WHERE key = 'pages'"))

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            uri = f"{self.path.resolve().as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            conn.execute("PRAGMA mmap_size = 1073741824")
            conn.execute("PRAGMA query_only = 1")
            self._local.conn = conn
        return conn

    def _scalar(self, sql: str) -> str:
        row = self._conn().execute(sql).fetchone()
        if row is None:
            raise IndexStoreError(f"{self.path} is missing metadata; rebuild the index")
        return str(row[0])

    def page_id(self, raw_title: str) -> int | None:
        row = self._conn().execute("SELECT id FROM pages WHERE raw_title = ?", (raw_title,))
        hit = row.fetchone()
        return None if hit is None else int(hit[0])

    def pages(self, ids: Sequence[int]) -> list[Page]:
        """Pages for ``ids`` in the same order (unknown ids are an error)."""
        if not ids:
            return []
        unique = list(dict.fromkeys(int(i) for i in ids))
        marks = ",".join("?" * len(unique))
        rows = self._conn().execute(
            f"SELECT id, raw_title, sentences, links FROM pages WHERE id IN ({marks})", unique
        )
        found = {int(r[0]): _row_to_page(r) for r in rows}
        missing = [i for i in unique if i not in found]
        if missing:
            raise IndexStoreError(f"page ids {missing[:5]} not in {self.path}; index is stale")
        return [found[int(i)] for i in ids]

    def page(self, page_id: int) -> Page:
        return self.pages([page_id])[0]

    def titles(self, pages: Iterable[int]) -> dict[int, str]:
        """Decoded titles for page ids (unknown ids are skipped)."""
        unique = list(dict.fromkeys(int(p) for p in pages))
        if not unique:
            return {}
        marks = ",".join("?" * len(unique))
        rows = self._conn().execute(
            f"SELECT id, raw_title FROM pages WHERE id IN ({marks})", unique
        )
        return {int(i): decode_title(str(t)) for i, t in rows}

    def aliases(self, keys: Iterable[str]) -> list[AliasHit]:
        unique = list(dict.fromkeys(keys))
        if not unique:
            return []
        marks = ",".join("?" * len(unique))
        rows = self._conn().execute(
            f"SELECT key, page, kind, count FROM aliases WHERE key IN ({marks})", unique
        )
        return [AliasHit(str(k), int(p), int(kind), int(c)) for k, p, kind, c in rows]


def _row_to_page(row: tuple[object, ...]) -> Page:
    page_id, raw_title, sentences, links = row
    sentence_list = str(sentences).split("\n")
    link_list = str(links).split("\n")
    return Page(
        id=int(str(page_id)),
        raw_title=str(raw_title),
        sentences=tuple(sentence_list),
        links=tuple(tuple(x for x in line.split("\t") if x) for line in link_list),
    )
