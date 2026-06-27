"""Tests for bitemb.engine – EmbeddingEngine."""

import numpy as np
import pytest

from bitemb.config import MODEL_DIM
from bitemb.engine import EmbeddingEngine


@pytest.fixture(scope="module")
def engine():
    return EmbeddingEngine()


@pytest.mark.slow
class TestEmbeddingEngine:
    def test_passages_shape(self, engine, sample_texts):
        emb = engine.encode_passages(sample_texts)
        assert emb.shape == (5, MODEL_DIM)

    def test_queries_shape(self, engine, sample_texts):
        emb = engine.encode_queries(sample_texts)
        assert emb.shape == (5, MODEL_DIM)

    def test_normalized_passages(self, engine, sample_texts):
        emb = engine.encode_passages(sample_texts)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_normalized_queries(self, engine, sample_texts):
        emb = engine.encode_queries(sample_texts)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_deterministic(self, engine, sample_texts):
        a = engine.encode_passages(sample_texts)
        b = engine.encode_passages(sample_texts)
        np.testing.assert_array_equal(a, b)

    def test_dtype(self, engine, sample_texts):
        emb = engine.encode_passages(sample_texts)
        assert emb.dtype == np.float32

    def test_queries_differ_from_passages(self, engine, sample_texts):
        """Query prefix should produce different embeddings than raw passages."""
        p = engine.encode_passages(sample_texts)
        q = engine.encode_queries(sample_texts)
        # They share same text but different prefix → should not be identical
        assert not np.allclose(p, q, atol=1e-3)
