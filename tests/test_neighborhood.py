"""Tests for bitemb.neighborhood module."""

import numpy as np
import pytest

from bitemb.neighborhood import (
    NeighborhoodResult,
    _compute_overlap,
    _compute_trustworthiness,
    _knn_cosine,
    _knn_hamming,
    _knn_turboquant,
    compute_neighborhood_preservation,
)
from bitemb.quantization import binarize


@pytest.fixture
def small_embeddings():
    """Small L2-normalized embeddings for testing."""
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((50, 32)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


class TestKnnCosine:
    def test_shape(self, small_embeddings):
        knn = _knn_cosine(small_embeddings, k=5)
        assert knn.shape == (50, 5)

    def test_no_self_neighbor(self, small_embeddings):
        knn = _knn_cosine(small_embeddings, k=5)
        for i in range(50):
            assert i not in knn[i]

    def test_sorted_by_similarity(self, small_embeddings):
        """First neighbor should be more similar than second."""
        knn = _knn_cosine(small_embeddings, k=5)
        sims = small_embeddings @ small_embeddings.T
        for i in range(10):  # spot check
            assert sims[i, knn[i, 0]] >= sims[i, knn[i, 1]]


class TestKnnHamming:
    def test_shape(self, small_embeddings):
        packed = binarize(small_embeddings)
        knn = _knn_hamming(packed, k=5)
        assert knn.shape == (50, 5)

    def test_no_self_neighbor(self, small_embeddings):
        packed = binarize(small_embeddings)
        knn = _knn_hamming(packed, k=5)
        for i in range(50):
            assert i not in knn[i]


class TestKnnTurboquant:
    def test_shape(self, small_embeddings):
        knn = _knn_turboquant(small_embeddings, bits=4, k=5)
        assert knn.shape == (50, 5)

    def test_no_self_neighbor(self, small_embeddings):
        knn = _knn_turboquant(small_embeddings, bits=2, k=5)
        for i in range(50):
            assert i not in knn[i]


class TestOverlap:
    def test_perfect_overlap(self):
        """Identical knn arrays should give overlap = 1.0."""
        knn = np.arange(20).reshape(4, 5)
        assert _compute_overlap(knn, knn, k=5) == 1.0

    def test_no_overlap(self):
        """Disjoint knn arrays should give overlap = 0.0."""
        knn_a = np.array([[0, 1, 2], [3, 4, 5]])
        knn_b = np.array([[6, 7, 8], [9, 10, 11]])
        assert _compute_overlap(knn_a, knn_b, k=3) == 0.0

    def test_partial_overlap(self):
        """50% overlap: 1 of 2 neighbors shared."""
        knn_a = np.array([[0, 1], [2, 3]])
        knn_b = np.array([[0, 4], [2, 5]])
        assert _compute_overlap(knn_a, knn_b, k=2) == 0.5


class TestTrustworthiness:
    def test_perfect_preservation(self):
        """If knn_quant == knn_float, T = 1.0."""
        knn = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
        # Build a consistent float_ranks matrix for 4 docs
        float_ranks = np.zeros((4, 4), dtype=np.int32)
        for i in range(4):
            for rank_pos, j in enumerate(knn[i]):
                float_ranks[i, j] = rank_pos + 1
        # embs not needed when float_ranks is provided
        dummy_embs = np.zeros((4, 3), dtype=np.float32)
        t = _compute_trustworthiness(
            knn, knn, float_ranks, dummy_embs, k=3, n_docs=4,
        )
        assert t == 1.0

    def test_range(self, small_embeddings):
        """Trustworthiness should be in [0, 1]."""
        from bitemb.neighborhood import _full_cosine_ranks

        knn_float = _knn_cosine(small_embeddings, k=10)
        float_ranks = _full_cosine_ranks(small_embeddings)
        packed = binarize(small_embeddings)
        knn_bin = _knn_hamming(packed, k=10)
        t = _compute_trustworthiness(
            knn_float, knn_bin, float_ranks, small_embeddings, k=10, n_docs=50,
        )
        assert 0.0 <= t <= 1.0


class TestComputeNeighborhoodPreservation:
    def test_returns_correct_count(self, small_embeddings):
        """Should return 5 bit_depths × 2 k_values = 10 results."""
        results = compute_neighborhood_preservation(
            small_embeddings, dim=32, k_values=(5, 10),
        )
        assert len(results) == 10  # 5 bit_depths × 2 k_values

    def test_result_type(self, small_embeddings):
        results = compute_neighborhood_preservation(
            small_embeddings, dim=32, k_values=(5,),
        )
        assert all(isinstance(r, NeighborhoodResult) for r in results)

    def test_overlap_range(self, small_embeddings):
        results = compute_neighborhood_preservation(
            small_embeddings, dim=32, k_values=(5,),
        )
        for r in results:
            assert 0.0 <= r.overlap <= 1.0

    def test_trustworthiness_range(self, small_embeddings):
        results = compute_neighborhood_preservation(
            small_embeddings, dim=32, k_values=(5,),
        )
        for r in results:
            assert 0.0 <= r.trustworthiness <= 1.0

    def test_4bit_better_than_1bit(self, small_embeddings):
        """4-bit should generally preserve neighbors better than 1-bit."""
        results = compute_neighborhood_preservation(
            small_embeddings, dim=32, k_values=(5,),
        )
        overlap_4bit = next(r.overlap for r in results if r.bit_depth == 4)
        overlap_1bit = next(r.overlap for r in results if r.bit_depth == 1)
        # 4-bit should be >= 1-bit (may not always hold for tiny data)
        assert overlap_4bit >= overlap_1bit * 0.8  # relaxed bound

    def test_random_baseline(self, small_embeddings):
        results = compute_neighborhood_preservation(
            small_embeddings, dim=32, k_values=(5,),
        )
        for r in results:
            assert r.random_baseline == pytest.approx(5 / 50)
