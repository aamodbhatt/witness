"""Build the Wikipedia store, the alias table and the BM25 index from ``wiki-pages.zip``.

One streaming pass over the 109 dump shards, parsed in a process pool. The main process
only appends: page rows and raw anchor rows go to SQLite, per-document term counts go to
compact numpy arrays. Anchor aggregation (``GROUP BY``) is pushed into SQLite so peak
memory stays bounded by the BM25 arrays rather than by Python dicts of anchors.
"""

from __future__ import annotations

import heapq
import io
import json
import logging
import multiprocessing as mp
import re
import sqlite3
import time
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from witness.errors import IndexStoreError
from witness.retrieval.bm25 import BM25Index, BM25Params, Tokenizer, bm25_matrix, count_terms
from witness.text import STOPWORDS, alias_key
from witness.wiki.fever_format import decode_title, encode_title, parse_page
from witness.wiki.store import (
    ALIAS_ANCHOR,
    ALIAS_BASE,
    ALIAS_TITLE,
    INDEXES,
    SCHEMA,
)

log = logging.getLogger(__name__)

_DISAMBIGUATION = re.compile(r"\s*\([^()]*\)$")
_MAX_ANCHOR_WORDS = 8


@dataclass
class ShardResult:
    """Everything one dump shard contributes; plain data so it pickles cheaply."""

    titles: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    vocab: list[str] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)  # unique terms per document
    term_ids: list[int] = field(default_factory=list)
    term_freqs: list[int] = field(default_factory=list)
    anchors: list[tuple[str, str, int]] = field(default_factory=list)  # key, raw target, n


def anchor_target(target: str) -> str:
    """Raw FEVER id for a hyperlink target (links omit the capitalised first letter)."""
    target = target.strip()
    return encode_title(target[:1].upper() + target[1:]) if target else ""


def base_title(title: str) -> str:
    """``"Savages (2012 film)"`` -> ``"Savages"``."""
    return _DISAMBIGUATION.sub("", title)


_KEEP: frozenset[str] | None = None


def _init_worker(keep: frozenset[str] | None) -> None:
    global _KEEP  # noqa: PLW0603 - per-process read-only filter set by the pool initializer
    _KEEP = keep


def _process_shard(args: tuple[str, str]) -> ShardResult:
    zip_path, member = args
    tokenize = Tokenizer()
    local_vocab: dict[str, int] = {}
    anchors: Counter[tuple[str, str]] = Counter()
    out = ShardResult()
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            page = parse_page(line)
            if page is None or (_KEEP is not None and page.raw_title not in _KEEP):
                continue
            texts = [ln.text for ln in page.lines]
            out.titles.append(page.raw_title)
            out.sentences.append("\n".join(texts))
            out.links.append("\n".join("\t".join(ln.links) for ln in page.lines))
            ids, freqs = count_terms(tokenize(page.title + " " + " ".join(texts)), local_vocab)
            out.doc_lengths.append(len(ids))
            out.term_ids.extend(ids)
            out.term_freqs.extend(freqs)
            for ln in page.lines:
                for surface, target in ln.anchors:
                    key = alias_key(surface)
                    short = 2 <= len(key) <= 80 and key.count(" ") < _MAX_ANCHOR_WORDS
                    if short and key not in STOPWORDS:
                        anchors[(key, anchor_target(target))] += 1
    out.vocab = sorted(local_vocab, key=local_vocab.__getitem__)
    out.anchors = [(k, t, n) for (k, t), n in anchors.items()]
    return out


def _shards(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            n for n in zf.namelist() if n.startswith("wiki-pages/") and n.endswith(".jsonl")
        )
    if not names:
        raise IndexStoreError(
            f"{zip_path} has no wiki-pages/*.jsonl members; is it the FEVER dump?"
        )
    return names


def _results(
    zip_path: Path, workers: int, keep: frozenset[str] | None, limit_shards: int | None
) -> Iterator[ShardResult]:
    shards = _shards(zip_path)[:limit_shards]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(keep,)) as pool:
        yield from pool.imap(_process_shard, [(str(zip_path), s) for s in shards])


def _count_links(args: tuple[str, str]) -> Counter[str]:
    zip_path, member = args
    counts: Counter[str] = Counter()
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            page = parse_page(line)
            if page is not None:
                for ln in page.lines:
                    counts.update(anchor_target(t) for t in ln.links)
    return counts


def most_linked_titles(zip_path: Path, n: int, workers: int) -> frozenset[str]:
    """Raw titles of the ``n`` pages with the most incoming hyperlinks (the lite profile)."""
    total: Counter[str] = Counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for counts in pool.imap_unordered(
            _count_links, [(str(zip_path), s) for s in _shards(zip_path)]
        ):
            total.update(counts)
    return top_titles(total, n)


def top_titles(counts: Mapping[str, int], n: int) -> frozenset[str]:
    """The ``n`` titles with the highest counts.

    Ties at the cutoff go to the alphabetically first title. ``Counter.most_common`` would
    break them by insertion order, which follows whichever worker finished first, so the
    lite page set would change from one build to the next.
    """
    ranked = heapq.nsmallest(n, counts.items(), key=lambda item: (-item[1], item[0]))
    return frozenset(title for title, _ in ranked)


def build_index(
    zip_path: Path,
    out_dir: Path,
    *,
    workers: int = 7,
    keep: frozenset[str] | None = None,
    limit_shards: int | None = None,
    params: BM25Params | None = None,
) -> dict[str, float]:
    """Build ``wiki.sqlite``, ``bm25/`` and ``inlinks.npy`` under ``out_dir``; returns stats."""
    if not zip_path.is_file():
        raise IndexStoreError(
            f"FEVER dump not found at {zip_path}; download it from "
            "https://fever.ai/download/fever/wiki-pages.zip (see the README quickstart)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "wiki.sqlite"
    tmp_path = out_dir / "wiki.sqlite.partial"
    tmp_path.unlink(missing_ok=True)
    started = time.perf_counter()
    conn = sqlite3.connect(tmp_path)
    conn.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;" + SCHEMA)
    conn.execute("CREATE TABLE anchors_raw (key TEXT, target TEXT, n INTEGER)")

    vocab: dict[str, int] = {}
    lengths: list[NDArray[np.int64]] = []
    term_chunks: list[NDArray[np.int32]] = []
    freq_chunks: list[NDArray[np.float32]] = []
    next_id = 0
    for shard_no, result in enumerate(_results(zip_path, workers, keep, limit_shards)):
        ids = range(next_id, next_id + len(result.titles))
        next_id += len(result.titles)
        conn.executemany(
            "INSERT INTO pages VALUES (?, ?, ?, ?)",
            zip(ids, result.titles, result.sentences, result.links, strict=True),
        )
        alias_rows: list[tuple[str, int, int, int]] = []
        for page_id, raw in zip(ids, result.titles, strict=True):
            title = decode_title(raw)
            alias_rows.append((alias_key(title), page_id, ALIAS_TITLE, 1))
            base = base_title(title)
            if base != title:
                alias_rows.append((alias_key(base), page_id, ALIAS_BASE, 1))
        conn.executemany("INSERT INTO aliases VALUES (?, ?, ?, ?)", alias_rows)
        conn.executemany("INSERT INTO anchors_raw VALUES (?, ?, ?)", result.anchors)
        remap = np.fromiter(
            (vocab.setdefault(t, len(vocab)) for t in result.vocab), dtype=np.int32,
            count=len(result.vocab),
        )  # fmt: skip
        lengths.append(np.asarray(result.doc_lengths, dtype=np.int64))
        term_chunks.append(remap[np.asarray(result.term_ids, dtype=np.int32)])
        freq_chunks.append(np.asarray(result.term_freqs, dtype=np.float32))
        conn.commit()
        log.info(
            "shard %d: %d pages total, vocab %d, %.0fs",
            shard_no + 1, next_id, len(vocab), time.perf_counter() - started,
        )  # fmt: skip

    log.info("aggregating anchors and building SQL indexes")
    conn.executescript(
        f"""
        CREATE UNIQUE INDEX pages_title ON pages(raw_title);
        INSERT INTO aliases
            SELECT a.key, p.id, {ALIAS_ANCHOR}, SUM(a.n)
            FROM anchors_raw a JOIN pages p ON p.raw_title = a.target
            GROUP BY a.key, p.id;
        DROP TABLE anchors_raw;
        """
    )
    conn.executescript(INDEXES)
    inlinks = np.zeros(next_id, dtype=np.int32)
    for page_id, n in conn.execute(
        f"SELECT page, SUM(count) FROM aliases WHERE kind = {ALIAS_ANCHOR} GROUP BY page"
    ):
        inlinks[int(page_id)] = int(n)
    n_aliases = int(conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0])
    conn.execute("INSERT INTO meta VALUES ('pages', ?)", (str(next_id),))
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    tmp_path.replace(db_path)
    np.save(out_dir / "inlinks.npy", inlinks)
    sql_seconds = time.perf_counter() - started

    log.info("building BM25 matrix")
    doc_lengths = np.concatenate(lengths)
    indptr = np.zeros(len(doc_lengths) + 1, dtype=np.int64)
    np.cumsum(doc_lengths, out=indptr[1:])
    term_ids = np.concatenate(term_chunks)
    del term_chunks
    term_freqs = np.concatenate(freq_chunks)
    del freq_chunks
    matrix = bm25_matrix(indptr, term_ids, term_freqs, len(vocab), params or BM25Params())
    nnz = int(matrix.nnz)
    del term_ids, term_freqs
    BM25Index.from_matrix(matrix, vocab).save(out_dir / "bm25")
    stats = {
        "pages": float(next_id),
        "aliases": float(n_aliases),
        "bm25_vocab": float(len(vocab)),
        "bm25_postings": float(nnz),
        "store_and_alias_seconds": round(sql_seconds, 1),
        "total_seconds": round(time.perf_counter() - started, 1),
        "store_bytes": float(db_path.stat().st_size),
        "bm25_bytes": float(sum(p.stat().st_size for p in (out_dir / "bm25").iterdir())),
    }
    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
