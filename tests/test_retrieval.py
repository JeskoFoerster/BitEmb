"""Tests for bitemb.retrieval - Phase 4 exact retrieval evaluation."""

import numpy as np
import pytest

from bitemb.quantization import PCAReducer
from bitemb.retrieval import (
    REPRESENTATIONS,
    RETRIEVAL_METRICS,
    RetrievalResult,
    _ndcg_at_k,
    _recall_at_k,
    _reciprocal_rank_asc,
    _reciprocal_rank_desc,
    _topk_asc,
    _topk_desc,
    compute_retrieval_evaluation,
    compute_significance_tests,
)


@pytest.fixture
def retrieval_fixture():
    """Small deterministic corpus/query setup with known relevant documents."""
    corpus = np.eye(24, dtype=np.float32)
    queries = corpus[[0, 1, 2, 3]].copy()
    qrels = {
        0: {0: 1},
        1: {1: 1},
        2: {2: 2},
        3: {3: 1},
    }
    return corpus, queries, qrels


class TestRetrievalMetrics:
    def test_recall_at_k(self):
        ranked = np.array([9, 2, 5, 7], dtype=np.int64)
        rels = {2: 1, 5: 1, 8: 1}
        assert _recall_at_k(ranked, rels, 2) == pytest.approx(1 / 3)
        assert _recall_at_k(ranked, rels, 4) == pytest.approx(2 / 3)

    def test_ndcg_perfect_is_one(self):
        ranked = np.array([2, 5, 9], dtype=np.int64)
        rels = {2: 2, 5: 1, 9: 1}
        assert _ndcg_at_k(ranked, rels, 3) == pytest.approx(1.0)

    def test_ndcg_zero_without_relevant_hits(self):
        ranked = np.array([1, 3, 4], dtype=np.int64)
        rels = {2: 1, 5: 1}
        assert _ndcg_at_k(ranked, rels, 3) == 0.0


class TestTieBreaking:
    def test_topk_desc_ties_by_doc_id(self):
        scores = np.array([0.7, 0.9, 0.9, 0.9, 0.1], dtype=np.float64)
        np.testing.assert_array_equal(_topk_desc(scores, k=2), np.array([1, 2]))

    def test_topk_asc_ties_by_doc_id(self):
        distances = np.array([3.0, 1.0, 1.0, 1.0, 4.0], dtype=np.float64)
        np.testing.assert_array_equal(_topk_asc(distances, k=2), np.array([1, 2]))

    def test_reciprocal_rank_desc_respects_ties(self):
        scores = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float64)
        assert _reciprocal_rank_desc(scores, {2: 1}) == pytest.approx(1 / 3)

    def test_reciprocal_rank_asc_respects_ties(self):
        distances = np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float64)
        assert _reciprocal_rank_asc(distances, {2: 1}) == pytest.approx(1 / 3)


class TestComputeRetrievalEvaluation:
    def test_returns_all_representations(self, retrieval_fixture):
        corpus, queries, qrels = retrieval_fixture
        results, per_query = compute_retrieval_evaluation(
            corpus, queries, qrels, dim=24, batch_size=2,
        )

        assert len(results) == 10
        assert [r.representation for r in results] == list(REPRESENTATIONS)
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert set(per_query) == set(REPRESENTATIONS)

    def test_metric_ranges_and_shapes(self, retrieval_fixture):
        corpus, queries, qrels = retrieval_fixture
        results, per_query = compute_retrieval_evaluation(
            corpus, queries, qrels, dim=24, batch_size=2,
        )

        for result in results:
            assert result.n_queries == 4
            assert 0.0 <= result.ndcg_at_10 <= 1.0
            assert 0.0 <= result.recall_at_10 <= 1.0
            assert 0.0 <= result.recall_at_100 <= 1.0
            assert 0.0 <= result.mrr <= 1.0

        for metrics in per_query.values():
            for metric in RETRIEVAL_METRICS:
                assert metrics[metric].shape == (4,)
                assert ((0.0 <= metrics[metric]) & (metrics[metric] <= 1.0)).all()

    def test_float_space_finds_identical_single_relevant_docs(self, retrieval_fixture):
        corpus, queries, qrels = retrieval_fixture
        results, _ = compute_retrieval_evaluation(corpus, queries, qrels, dim=24)
        float_result = next(r for r in results if r.representation == "float32")

        assert float_result.ndcg_at_10 >= 0.95
        assert float_result.recall_at_100 == 1.0
        assert float_result.mrr == 1.0

    def test_with_pca(self, retrieval_fixture):
        corpus, queries, qrels = retrieval_fixture
        pca = PCAReducer(n_components=8).fit(corpus)
        results, per_query = compute_retrieval_evaluation(
            corpus, queries, qrels, dim=8, pca_reducer=pca, batch_size=2,
        )

        assert len(results) == 10
        assert all(r.dim == 8 for r in results)
        assert per_query["float32"]["ndcg_at_10"].shape == (4,)

    def test_rejects_missing_qrels(self, retrieval_fixture):
        corpus, queries, qrels = retrieval_fixture
        qrels = dict(qrels)
        qrels.pop(3)
        with pytest.raises(ValueError, match="Expected qrels"):
            compute_retrieval_evaluation(corpus, queries, qrels, dim=24)


class TestSignificance:
    def test_pairwise_count(self):
        per_query = {
            rep: {metric: np.linspace(0.1, 0.9, 8) for metric in RETRIEVAL_METRICS}
            for rep in REPRESENTATIONS
        }
        results = compute_significance_tests(per_query, dim=24)
        assert len(results) == 180  # 4 metrics * 45 pairwise comparisons
        assert all(r.p_value == 1.0 for r in results)
        assert all(not r.significant for r in results)

    def test_detects_difference(self):
        per_query = {
            rep: {metric: np.linspace(0.1, 0.9, 8) for metric in RETRIEVAL_METRICS}
            for rep in REPRESENTATIONS
        }
        per_query["naive_1bit"] = {
            metric: np.zeros(8, dtype=np.float64) for metric in RETRIEVAL_METRICS
        }

        results = compute_significance_tests(per_query, dim=24)
        mrr_float_vs_binary = next(
            r for r in results
            if r.metric == "mrr"
            and r.representation_a == "float32"
            and r.representation_b == "naive_1bit"
        )
        assert mrr_float_vs_binary.p_value < 0.05

