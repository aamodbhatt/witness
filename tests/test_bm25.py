"""The streaming BM25 builder must reproduce bm25s (Lucene variant) exactly."""

from __future__ import annotations

from pathlib import Path

import bm25s
import numpy as np
import pytest

from witness.retrieval.bm25 import BM25Index, BM25Params, Tokenizer, bm25_matrix, count_terms

CORPUS = [
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    "Berlin is the capital of Germany and its largest city.",
    "Marie Curie was a Polish and naturalised-French physicist and chemist.",
    "The Seine is a river that flows through Paris and northern France.",
    "Germany borders France, and the Rhine forms part of the border.",
]
QUERIES = ["capital of France", "Paris river", "Curie chemist Poland", "Germany border Rhine"]


def _build(corpus: list[str]) -> tuple[BM25Index, dict[str, int]]:
    tokenize = Tokenizer()
    vocab: dict[str, int] = {}
    lengths, term_ids, freqs = [], [], []
    for text in corpus:
        ids, counts = count_terms(tokenize(text), vocab)
        lengths.append(len(ids))
        term_ids.extend(ids)
        freqs.extend(counts)
    indptr = np.zeros(len(corpus) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    matrix = bm25_matrix(
        indptr,
        np.asarray(term_ids, dtype=np.int32),
        np.asarray(freqs, dtype=np.float32),
        len(vocab),
        BM25Params(),
    )
    return BM25Index.from_matrix(matrix, vocab), vocab


def test_scores_match_bm25s_lucene() -> None:
    index, _ = _build(CORPUS)
    tokenize = Tokenizer()
    reference = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    reference.index([tokenize(t) for t in CORPUS], show_progress=False)
    for query in QUERIES:
        expected = reference.get_scores(tokenize(query))
        ours = np.zeros(len(CORPUS))
        for doc, score in index.search(query, k=len(CORPUS)):
            ours[doc] = score
        np.testing.assert_allclose(ours, expected, rtol=1e-5, atol=1e-6)


def test_search_orders_by_score_and_truncates() -> None:
    index, _ = _build(CORPUS)
    hits = index.search("Paris France capital", k=3)
    assert len(hits) == 3
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0][0] == 0


def test_unknown_terms_return_nothing() -> None:
    index, _ = _build(CORPUS)
    assert index.search("zzzz qqqq", k=5) == []
    assert index.search("the of and", k=5) == []  # stopwords only


def test_save_and_mmap_load_roundtrip(tmp_path: Path) -> None:
    index, _ = _build(CORPUS)
    index.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25", mmap=True)
    for query in QUERIES:
        got, want = loaded.search(query, 4), index.search(query, 4)
        assert [d for d, _ in got] == [d for d, _ in want]
        assert [x for _, x in got] == pytest.approx([x for _, x in want])
