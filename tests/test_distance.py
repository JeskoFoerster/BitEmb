"""Tests for bitemb.distance – Phase 2 pairwise distance analysis."""

import numpy as np
import pytest

from bitemb.distance import (
    REPRESENTATIONS,
    DistortionResult,
    _compute_metrics,
    _cosine_distance_pairs,
    _hamming_distance_pairs,
    _normalize_to_unit,
    _quantized_distance_pairs,
    _sample_pairs,
    _turboquant_distance_pairs,
    compute_distance_distortion,
)
from bitemb.quantization import PCAReducer, binarize, quantize_encode, turboquant_encode


@pytest.fixture
def corpus_embs():
    """200 normalized random 1024-d vectors (sufficient for 10k pair sampling)."""
    rng = np.random.default_rng(42)
    embs = rng.normal(size=(200, 1024)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


class TestSamplePairs:
    def test_shape_and_count(self):
        pairs = _sample_pairs(500, n_pairs=100, seed=42)
        assert pairs.shape == (100, 2)

    def test_no_self_pairs(self):
        pairs = _sample_pairs(500, n_pairs=1000, seed=42)
        assert (pairs[:, 0] != pairs[:, 1]).all()

    def test_ordered(self):
        pairs = _sample_pairs(500, n_pairs=1000, seed=42)
        assert (pairs[:, 0] < pairs[:, 1]).all()

    def test_unique(self):
        pairs = _sample_pairs(500, n_pairs=1000, seed=42)
        pair_set = set(map(tuple, pairs.tolist()))
        assert len(pair_set) == 1000

    def test_deterministic(self):
        p1 = _sample_pairs(500, n_pairs=100, seed=42)
        p2 = _sample_pairs(500, n_pairs=100, seed=42)
        np.testing.assert_array_equal(p1, p2)


class TestNormalize:
    def test_range_01(self):
        d = np.array([2.0, 5.0, 8.0, 10.0])
        n = _normalize_to_unit(d)
        assert abs(n.min()) < 1e-10
        assert abs(n.max() - 1.0) < 1e-10

    def test_constant_returns_zeros(self):
        d = np.array([3.0, 3.0, 3.0])
        n = _normalize_to_unit(d)
        assert (n == 0).all()


class TestCosineDistancePairs:
    def test_self_pair_zero(self, corpus_embs):
        # Distance to self should be ~0
        pairs = np.array([[0, 0]], dtype=np.int64)
        d = _cosine_distance_pairs(corpus_embs, pairs)
        assert abs(d[0]) < 1e-5

    def test_range(self, corpus_embs):
        pairs = _sample_pairs(200, n_pairs=100, seed=42)
        d = _cosine_distance_pairs(corpus_embs, pairs)
        # Cosine distance for normalized vecs: [0, 2]
        assert d.min() >= -1e-6
        assert d.max() <= 2.0 + 1e-6


class TestHammingDistancePairs:
    def test_self_zero(self, corpus_embs):
        packed = binarize(corpus_embs)
        pairs = np.array([[0, 0]], dtype=np.int64)
        d = _hamming_distance_pairs(packed, pairs)
        assert d[0] == 0

    def test_range(self, corpus_embs):
        packed = binarize(corpus_embs)
        pairs = _sample_pairs(200, n_pairs=100, seed=42)
        d = _hamming_distance_pairs(packed, pairs)
        # Hamming is integer in [0, dim]
        assert d.min() >= 0
        assert d.max() <= 1024


class TestQuantizedDistancePairs:
    """Tests for _quantized_distance_pairs (renamed from _turboquant_distance_pairs)."""

    def test_self_near_zero(self, corpus_embs):
        idx = quantize_encode(corpus_embs, bits=4, is_rotated=True)
        pairs = np.array([[0, 0]], dtype=np.int64)
        d = _quantized_distance_pairs(idx, pairs)
        assert d[0] < 1e-6

    def test_nonnegative(self, corpus_embs):
        idx = quantize_encode(corpus_embs, bits=2, is_rotated=True)
        pairs = _sample_pairs(200, n_pairs=100, seed=42)
        d = _quantized_distance_pairs(idx, pairs)
        assert (d >= -1e-10).all()

    def test_naive_self_near_zero(self, corpus_embs):
        idx = quantize_encode(corpus_embs, bits=4, is_rotated=False)
        pairs = np.array([[0, 0]], dtype=np.int64)
        d = _quantized_distance_pairs(idx, pairs)
        assert d[0] < 1e-6

    def test_naive_nonnegative(self, corpus_embs):
        idx = quantize_encode(corpus_embs, bits=2, is_rotated=False)
        pairs = _sample_pairs(200, n_pairs=100, seed=42)
        d = _quantized_distance_pairs(idx, pairs)
        assert (d >= -1e-10).all()


class TestTurboQuantDistancePairs:
    """Tests for backward-compatible _turboquant_distance_pairs alias."""

    def test_self_near_zero(self, corpus_embs):
        idx = turboquant_encode(corpus_embs, bits=4)
        pairs = np.array([[0, 0]], dtype=np.int64)
        d = _turboquant_distance_pairs(idx, pairs)
        assert d[0] < 1e-6

    def test_nonnegative(self, corpus_embs):
        idx = turboquant_encode(corpus_embs, bits=2)
        pairs = _sample_pairs(200, n_pairs=100, seed=42)
        d = _turboquant_distance_pairs(idx, pairs)
        assert (d >= -1e-10).all()


class TestComputeMetrics:
    def test_perfect_correlation(self):
        d_float = np.linspace(0.1, 0.9, 100)
        d_quant = d_float * 2.0 + 0.5  # linear transform
        pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_quant)
        assert pr > 0.999
        assert sr > 0.999
        # After normalization, linear transform → identical → MAE ≈ 0
        assert mae < 0.01

    def test_anticorrelation(self):
        d_float = np.linspace(0.1, 0.9, 100)
        d_quant = -d_float
        pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_quant)
        assert pr < -0.99
        assert sr < -0.99


class TestComputeDistanceDistortion:
    def test_returns_nine_results(self, corpus_embs):
        """Now returns 9 results (one per non-float32 representation)."""
        results = compute_distance_distortion(corpus_embs, dim=1024, n_pairs=500)
        assert len(results) == 9
        representations = [r.representation for r in results]
        expected = [
            "16bit", "naive_8bit", "tq_8bit", "naive_4bit", "tq_4bit",
            "naive_2bit", "tq_2bit", "naive_1bit", "tq_1bit",
        ]
        assert representations == expected

    def test_result_types(self, corpus_embs):
        results = compute_distance_distortion(corpus_embs, dim=1024, n_pairs=500)
        for r in results:
            assert isinstance(r, DistortionResult)
            assert 0 <= abs(r.pearson_r) <= 1.0 + 1e-6
            assert 0 <= abs(r.spearman_rho) <= 1.0 + 1e-6
            assert r.mae >= 0
            assert r.rmse >= 0
            assert r.rmse >= r.mae  # RMSE ≥ MAE always

    def test_backward_compat_bit_depth(self, corpus_embs):
        """Backward compatibility: bit_depth property still works."""
        results = compute_distance_distortion(corpus_embs, dim=1024, n_pairs=500)
        by_rep = {r.representation: r for r in results}
        assert by_rep["16bit"].bit_depth == 16
        assert by_rep["tq_8bit"].bit_depth == 8
        assert by_rep["tq_4bit"].bit_depth == 4
        assert by_rep["tq_2bit"].bit_depth == 2
        assert by_rep["naive_1bit"].bit_depth == 1

    def test_tq_4bit_better_than_tq_2bit(self, corpus_embs):
        results = compute_distance_distortion(corpus_embs, dim=1024, n_pairs=1000)
        by_rep = {r.representation: r for r in results}
        # 4-bit should correlate better than 2-bit
        assert by_rep["tq_4bit"].pearson_r >= by_rep["tq_2bit"].pearson_r - 0.05

    def test_with_pca(self, corpus_embs):
        pca = PCAReducer(n_components=128).fit(corpus_embs)
        results = compute_distance_distortion(
            corpus_embs, dim=128, pca_reducer=pca, n_pairs=500,
        )
        assert len(results) == 9
        for r in results:
            assert r.dim == 128

    def test_deterministic(self, corpus_embs):
        r1 = compute_distance_distortion(corpus_embs, dim=1024, n_pairs=500, seed=42)
        r2 = compute_distance_distortion(corpus_embs, dim=1024, n_pairs=500, seed=42)
        for a, b in zip(r1, r2):
            assert a.pearson_r == b.pearson_r
            assert a.mae == b.mae

    def test_tq_better_than_naive_at_same_bits(self, corpus_embs):
        """TurboQuant (rotated) should generally correlate as well or better than naive."""
        results = compute_distance_distortion(corpus_embs, dim=1024, n_pairs=1000)
        by_rep = {r.representation: r for r in results}
        # TQ 4-bit should be at least close to naive 4-bit (usually better)
        # Allow small tolerance since random data may vary
        assert by_rep["tq_4bit"].pearson_r >= by_rep["naive_4bit"].pearson_r - 0.1
