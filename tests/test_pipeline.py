from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tests.fakes import FakeLinker, FakeSearch, FakeStore, KeywordVerifier, OverlapScorer
from witness.checker import FactChecker
from witness.config import WitnessConfig
from witness.retrieval.pipeline import EvidenceRetriever, reciprocal_rank_fusion


def _retriever() -> EvidenceRetriever:
    return EvidenceRetriever(FakeStore(), FakeLinker(), FakeSearch(), OverlapScorer(),
                             WitnessConfig())  # fmt: skip


def test_rrf_prefers_documents_ranked_well_by_both() -> None:
    fused = reciprocal_rank_fusion([([1, 2, 3], 1.0), ([3, 1, 4], 1.0)], k=10, top=4)
    assert fused[0] == 1
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_zero_weight_ignores_a_ranking() -> None:
    assert reciprocal_rank_fusion([([1, 2], 1.0), ([9], 0.0)], k=10, top=5) == [1, 2]


@given(st.lists(st.lists(st.integers(0, 20), unique=True, max_size=8), max_size=3),
       st.integers(1, 10))  # fmt: skip
def test_rrf_output_is_unique_and_bounded(rankings: list[list[int]], top: int) -> None:
    fused = reciprocal_rank_fusion([(r, 1.0) for r in rankings], k=5, top=top)
    assert len(fused) == len(set(fused)) <= top
    assert set(fused) <= {d for r in rankings for d in r}


def test_evidence_comes_from_the_right_sentence() -> None:
    retrieval = _retriever().retrieve("Paris is the capital of France.")
    best = retrieval.evidence[0]
    assert best.raw_title == "Paris"
    assert best.line == 0
    assert best.text.startswith("Paris is the capital")
    assert retrieval.candidates.fused[0] == 0


def test_blank_sentence_slots_keep_fever_line_numbers() -> None:
    retrieval = _retriever().retrieve("Marie_Curie won two Nobel Prizes.")
    lines = {(e.raw_title, e.line) for e in retrieval.evidence}
    assert ("Marie_Curie", 2) in lines  # line 1 is blank in the dump and must be skipped


def test_fact_checker_end_to_end_with_fakes() -> None:
    checker = FactChecker(_retriever(), KeywordVerifier(), temperature=1.0, abstain_below=0.6)
    results = checker.check_many(["Paris is the capital of France.", "Paris is not in France."])
    assert [r.decision.verdict for r in results] == ["supported", "refuted"]
    assert results[0].evidence
    assert "verify_ms" in results[0].timings_ms
