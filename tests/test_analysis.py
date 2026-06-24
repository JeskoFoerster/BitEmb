"""Tests for analysis module."""

import numpy as np
import pytest

from bitemb.analysis import analyze_vector_space, analyze_information_loss


class TestAnalyzeVectorSpace:
    def test_returns_expected_keys(self, random_float_embs):
        result = analyze_vector_space(random_float_embs)
        expected = {
            "num_vectors", "dimension", "mean_pairwise_sim",
            "std_pairwise_sim", "component_variance",
            "effective_dim_95pct", "isotropy_score",
        }
        assert set(result.keys()) == expected

    def test_num_vectors(self, random_float_embs):
        result = analyze_vector_space(random_float_embs)
        assert result["num_vectors"] == 10

    def test_similarity_range(self, random_float_embs):
        result = analyze_vector_space(random_float_embs)
        assert -1.0 <= result["mean_pairwise_sim"] <= 1.0

    def test_isotropy_range(self, random_float_embs):
        result = analyze_vector_space(random_float_embs)
        assert 0.0 <= result["isotropy_score"] <= 1.0


class TestAnalyzeInformationLoss:
    def test_returns_expected_keys(self, random_float_embs, random_bit_embs):
        result = analyze_information_loss(random_float_embs, random_bit_embs, 768)
        expected = {
            "cosine_preservation_mean", "cosine_preservation_std",
            "cosine_preservation_min", "rank_correlation",
            "compression_ratio",
        }
        assert set(result.keys()) == expected

    def test_compression_ratio(self, random_float_embs, random_bit_embs):
        result = analyze_information_loss(random_float_embs, random_bit_embs, 768)
        assert result["compression_ratio"] == pytest.approx(32.0, abs=0.1)

    def test_cosine_preservation_range(self, random_float_embs, random_bit_embs):
        result = analyze_information_loss(random_float_embs, random_bit_embs, 768)
        assert 0.0 <= result["cosine_preservation_mean"] <= 1.0

    def test_rank_correlation_range(self, random_float_embs, random_bit_embs):
        result = analyze_information_loss(random_float_embs, random_bit_embs, 768)
        assert -1.0 <= result["rank_correlation"] <= 1.0
