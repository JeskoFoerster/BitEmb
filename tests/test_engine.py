"""Tests for bitemb.engine – EmbeddingEngine."""

import numpy as np
import pytest

from bitemb.engine import EmbeddingEngine


@pytest.fixture(scope="module")
def engine():
    return EmbeddingEngine()


@pytest.fixture(scope="module")
def sample_texts():
    return [
        "Machine learning models learn patterns from data",
        "Neural networks use layers of interconnected nodes",
        "Binary quantization reduces embeddings to single bits",
        "The cat sat on the mat",
        "Quantum computing uses qubits for computation",
    ]


@pytest.mark.slow
class TestEmbedFloat:
    def test_shape_default(self, engine, sample_texts):
        emb = engine.embed_float(sample_texts)
        assert emb.shape == (5, 768)

    def test_shape_truncated(self, engine, sample_texts):
        emb = engine.embed_float(sample_texts, dim=256)
        assert emb.shape == (5, 256)

    def test_normalized(self, engine, sample_texts):
        emb = engine.embed_float(sample_texts)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_normalized_after_truncation(self, engine, sample_texts):
        emb = engine.embed_float(sample_texts, dim=256)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_deterministic(self, engine, sample_texts):
        a = engine.embed_float(sample_texts)
        b = engine.embed_float(sample_texts)
        np.testing.assert_array_equal(a, b)


@pytest.mark.slow
class TestEmbedBit:
    def test_sign_shape(self, engine, sample_texts):
        packed = engine.embed_bit_sign(sample_texts)
        assert packed.shape == (5, 96)

    def test_sign_truncated(self, engine, sample_texts):
        packed = engine.embed_bit_sign(sample_texts, dim=256)
        assert packed.shape == (5, 32)

    def test_matryoshka(self, engine, sample_texts):
        packed = engine.embed_bit_matryoshka(sample_texts, dim=64)
        assert packed.shape == (5, 8)


class TestPackUnpack:
    def test_roundtrip(self):
        bits = np.array([[1, 0, 1, 1, 0, 0, 1, 0]], dtype=np.uint8)
        packed = EmbeddingEngine._pack_bits(bits)
        unpacked = EmbeddingEngine.unpack_bits(packed, dim=8)
        np.testing.assert_array_equal(unpacked, bits)

    def test_unpack_dim_crop(self):
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=(3, 768), dtype=np.uint8)
        packed = EmbeddingEngine._pack_bits(bits)
        unpacked = EmbeddingEngine.unpack_bits(packed, dim=768)
        np.testing.assert_array_equal(unpacked, bits)
