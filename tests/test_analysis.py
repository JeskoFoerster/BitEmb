"""Tests for bitemb.analysis – Phase 1 float space characterization."""

import numpy as np
import pytest

from bitemb.analysis import (
    compute_dimension_stats,
    compute_intrinsic_dimensionality,
    compute_norm_distribution,
)


@pytest.fixture
def normalized_embs():
    """100 normalized random 1024-d vectors (enough for TwoNN)."""
    rng = np.random.default_rng(42)
    embs = rng.normal(size=(100, 1024)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


class TestNormDistribution:
    def test_normalized_vectors_near_unit_sphere(self, normalized_embs):
        stats = compute_norm_distribution(normalized_embs)
        assert stats.is_near_unit_sphere()
        assert abs(stats.mean - 1.0) < 1e-5
        assert stats.cv < 1e-5

    def test_unnormalized_vectors_not_unit_sphere(self):
        rng = np.random.default_rng(99)
        embs = rng.normal(size=(50, 1024)).astype(np.float32)
        stats = compute_norm_distribution(embs)
        assert not stats.is_near_unit_sphere()

    def test_stats_fields(self, normalized_embs):
        stats = compute_norm_distribution(normalized_embs)
        assert stats.min <= stats.mean <= stats.max
        assert stats.std >= 0


class TestDimensionStats:
    def test_output_shapes(self, normalized_embs):
        stats = compute_dimension_stats(normalized_embs)
        assert stats.mean.shape == (1024,)
        assert stats.std.shape == (1024,)
        assert stats.skewness.shape == (1024,)
        assert stats.kurtosis.shape == (1024,)

    def test_std_positive(self, normalized_embs):
        stats = compute_dimension_stats(normalized_embs)
        assert (stats.std >= 0).all()

    def test_mean_near_zero_for_normalized(self, normalized_embs):
        stats = compute_dimension_stats(normalized_embs)
        # Random normalized vectors should have near-zero mean per dim
        assert np.abs(stats.mean).mean() < 0.05


class TestIntrinsicDimensionality:
    def test_twonn_positive(self, normalized_embs):
        result = compute_intrinsic_dimensionality(normalized_embs)
        assert result.twonn > 0

    def test_pca_95_within_bounds(self, normalized_embs):
        result = compute_intrinsic_dimensionality(normalized_embs)
        # Must be between 1 and 1024
        assert 1 <= result.pca_95 <= 1024

    def test_cumulative_variance_monotonic(self, normalized_embs):
        result = compute_intrinsic_dimensionality(normalized_embs)
        diffs = np.diff(result.pca_cumulative_variance)
        assert (diffs >= -1e-10).all()  # monotonically non-decreasing

    def test_cumulative_variance_ends_near_one(self, normalized_embs):
        result = compute_intrinsic_dimensionality(normalized_embs)
        assert abs(result.pca_cumulative_variance[-1] - 1.0) < 1e-6
