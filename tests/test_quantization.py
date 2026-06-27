"""Tests for bitemb.quantization – binary and TurboQuant."""

import numpy as np
import pytest

from bitemb.quantization import (
    binarize,
    hamming_distance,
    turboquant_distance,
    turboquant_encode,
)


class TestBinarize:
    def test_output_shape(self, random_float_embs):
        packed = binarize(random_float_embs)
        # 768 bits → 96 bytes
        assert packed.shape == (10, 96)

    def test_dtype(self, random_float_embs):
        packed = binarize(random_float_embs)
        assert packed.dtype == np.uint8

    def test_compression_ratio(self, random_float_embs):
        packed = binarize(random_float_embs)
        assert random_float_embs.nbytes / packed.nbytes == 32.0


class TestHammingDistance:
    def test_self_distance_zero(self, random_float_embs):
        packed = binarize(random_float_embs)
        dist = hamming_distance(packed, packed[0:1])
        assert dist[0] == 0

    def test_symmetry(self, random_float_embs):
        packed = binarize(random_float_embs)
        d_ab = hamming_distance(packed[0:1], packed[1:2])
        d_ba = hamming_distance(packed[1:2], packed[0:1])
        assert d_ab[0] == d_ba[0]


class TestTurboQuant:
    @pytest.mark.parametrize("bits", [2, 4])
    def test_encode_shape(self, random_float_embs, bits):
        index = turboquant_encode(random_float_embs, bits=bits)
        assert index.codes.shape == (10, 768)
        assert index.bits == bits

    @pytest.mark.parametrize("bits", [2, 4])
    def test_code_range(self, random_float_embs, bits):
        index = turboquant_encode(random_float_embs, bits=bits)
        max_val = (1 << bits) - 1
        assert index.codes.max() <= max_val
        assert index.codes.min() >= 0

    def test_invalid_bits(self, random_float_embs):
        with pytest.raises(ValueError):
            turboquant_encode(random_float_embs, bits=3)

    @pytest.mark.parametrize("bits", [2, 4])
    def test_dequantize_shape(self, random_float_embs, bits):
        index = turboquant_encode(random_float_embs, bits=bits)
        recon = index.dequantize()
        assert recon.shape == (10, 768)

    @pytest.mark.parametrize("bits", [2, 4])
    def test_distance_shape(self, random_float_embs, bits):
        index = turboquant_encode(random_float_embs, bits=bits)
        queries = random_float_embs[:3]
        dists = turboquant_distance(index, queries)
        assert dists.shape == (3, 10)

    @pytest.mark.parametrize("bits", [2, 4])
    def test_self_distance_near_zero(self, random_float_embs, bits):
        index = turboquant_encode(random_float_embs, bits=bits)
        dists = turboquant_distance(index, random_float_embs)
        # Diagonal should be near zero (self-distance after quantization)
        diag = np.diag(dists)
        # 4-bit should have lower error than 2-bit
        assert diag.max() < 1.0 if bits == 4 else diag.max() < 5.0

    def test_4bit_more_precise_than_2bit(self, random_float_embs):
        idx2 = turboquant_encode(random_float_embs, bits=2)
        idx4 = turboquant_encode(random_float_embs, bits=4)
        d2 = np.diag(turboquant_distance(idx2, random_float_embs))
        d4 = np.diag(turboquant_distance(idx4, random_float_embs))
        # 4-bit reconstruction should be closer to original
        assert d4.mean() < d2.mean()

    def test_memory_bytes(self, random_float_embs):
        idx2 = turboquant_encode(random_float_embs, bits=2)
        idx4 = turboquant_encode(random_float_embs, bits=4)
        # 2-bit should use half the storage of 4-bit
        assert idx2.memory_bytes() < idx4.memory_bytes()
