"""Page-level BM25 with a streaming builder and a memory-mapped term-major index.

``bm25s`` builds its score matrix from the whole tokenized corpus held as Python lists,
which for 5.4M Wikipedia pages costs tens of GB. The builder here consumes per-document
term counts shard by shard and produces the same eager-scored CSC matrix as
``bm25s.BM25(method="lucene")`` (``tests/test_bm25.py`` checks the two agree), so the
score of a query is a sum of precomputed posting-list entries.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.sparse as sp
import snowballstemmer
from numpy.typing import NDArray

from witness.errors import IndexStoreError
from witness.text import content_words


@functools.cache
def _stemmer() -> snowballstemmer.stemmer:
    return snowballstemmer.stemmer("english")


class Tokenizer:
    """Folded, stopword-free, Snowball-stemmed tokens with a per-instance stem cache."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def __call__(self, text: str) -> list[str]:
        cache = self._cache
        out: list[str] = []
        for word in content_words(text):
            stem = cache.get(word)
            if stem is None:
                stem = str(_stemmer().stemWord(word))
                if len(cache) < 2_000_000:
                    cache[word] = stem
            out.append(stem)
        return out


@dataclass(frozen=True, slots=True)
class BM25Params:
    k1: float = 1.2
    b: float = 0.75


def bm25_matrix(
    doc_indptr: NDArray[np.int64],
    term_ids: NDArray[np.int32],
    term_freqs: NDArray[np.float32],
    vocab_size: int,
    params: BM25Params,
) -> sp.csc_matrix:
    """Eager BM25 weights as a (docs x terms) CSC matrix (Lucene variant, as in bm25s).

    ``doc_indptr``/``term_ids``/``term_freqs`` describe a CSR matrix of raw term counts
    with unique terms per document. Weight = idf(t) * tf / (tf + k1 * (1 - b + b * dl/avgdl)),
    idf(t) = log(1 + (N - df + 0.5) / (df + 0.5)).
    """
    n_docs = len(doc_indptr) - 1
    lengths = np.add.reduceat(term_freqs, doc_indptr[:-1]) if len(term_freqs) else np.zeros(0)
    lengths = np.where(np.diff(doc_indptr) > 0, lengths, 0.0).astype(np.float32)
    avgdl = float(lengths.mean()) if n_docs else 1.0
    df = np.bincount(term_ids, minlength=vocab_size).astype(np.float64)
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    doc_of_entry = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_indptr))
    norm = (params.k1 * (1.0 - params.b + params.b * lengths / max(avgdl, 1e-9))).astype(np.float32)
    weights = term_freqs / (term_freqs + norm[doc_of_entry])
    del doc_of_entry
    weights *= idf[term_ids]
    csr = sp.csr_matrix(
        (weights.astype(np.float32), term_ids, doc_indptr), shape=(n_docs, vocab_size)
    )
    return csr.tocsc()


class BM25Index:
    """Term-at-a-time BM25 search over a memory-mapped CSC matrix."""

    def __init__(
        self,
        vocab: dict[str, int],
        data: NDArray[np.float32],
        indices: NDArray[np.int32],
        indptr: NDArray[np.int64],
        num_docs: int,
    ) -> None:
        self.vocab = vocab
        self._data = data
        self._indices = indices
        self._indptr = indptr
        self.num_docs = num_docs
        self._tokenize = Tokenizer()

    @classmethod
    def from_matrix(cls, matrix: sp.csc_matrix, vocab: dict[str, int]) -> BM25Index:
        return cls(
            vocab,
            np.asarray(matrix.data, dtype=np.float32),
            np.asarray(matrix.indices, dtype=np.int32),
            np.asarray(matrix.indptr, dtype=np.int64),
            int(matrix.shape[0]),
        )

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-``k`` (doc id, score) pairs, best first; ties broken by lower doc id.

        Postings are accumulated into a dense score vector (each term's posting list has
        unique doc ids, so fancy-index ``+=`` is exact). Only touched documents are
        ranked: a document appears at most once per query term, so the best
        ``k * terms`` touched entries always contain the ``k`` best distinct documents.
        """
        term_ids = [self.vocab[t] for t in self._tokenize(query) if t in self.vocab]
        if not term_ids or k <= 0:
            return []
        scores = np.zeros(self.num_docs, dtype=np.float32)
        touched_parts: list[NDArray[np.int32]] = []
        for t in term_ids:
            start, end = int(self._indptr[t]), int(self._indptr[t + 1])
            docs = np.asarray(self._indices[start:end])
            scores[docs] += self._data[start:end]
            touched_parts.append(docs)
        touched = np.concatenate(touched_parts)
        pool = min(len(touched), k * len(term_ids))
        values = scores[touched]
        if pool < len(touched):
            touched = touched[np.argpartition(-values, pool - 1)[:pool]]
        candidates = np.unique(touched)
        order = np.lexsort((candidates, -scores[candidates]))[:k]
        return [(int(candidates[i]), float(scores[candidates[i]])) for i in order]

    # ------------------------------------------------------------ persistence

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "data.npy", self._data)
        np.save(directory / "indices.npy", self._indices)
        np.save(directory / "indptr.npy", self._indptr)
        meta = {"num_docs": self.num_docs, "vocab_size": len(self.vocab)}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        terms = sorted(self.vocab, key=self.vocab.__getitem__)
        (directory / "vocab.txt").write_text("\n".join(terms), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, *, mmap: bool = True) -> BM25Index:
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            terms = (directory / "vocab.txt").read_text(encoding="utf-8").split("\n")
            mode: Literal["r"] | None = "r" if mmap else None
            data = np.load(directory / "data.npy", mmap_mode=mode)
            indices = np.load(directory / "indices.npy", mmap_mode=mode)
            indptr = np.load(directory / "indptr.npy")
        except FileNotFoundError as exc:
            raise IndexStoreError(
                f"BM25 index incomplete in {directory} ({exc.filename} missing); "
                "rebuild with `witness build-index`"
            ) from exc
        if len(terms) != int(meta["vocab_size"]):
            raise IndexStoreError(f"BM25 vocabulary in {directory} does not match its metadata")
        vocab = {t: i for i, t in enumerate(terms)}
        return cls(vocab, data, indices, indptr, int(meta["num_docs"]))


def count_terms(tokens: Sequence[str], vocab: dict[str, int]) -> tuple[list[int], list[int]]:
    """Unique term ids (growing ``vocab``) and their counts for one document."""
    counts: dict[int, int] = {}
    for token in tokens:
        tid = vocab.get(token)
        if tid is None:
            tid = len(vocab)
            vocab[token] = tid
        counts[tid] = counts.get(tid, 0) + 1
    return list(counts), list(counts.values())
