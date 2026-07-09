"""Tests for Phase 5 efficiency helpers."""

import numpy as np
import pytest

from bitemb.efficiency import (
    REPRESENTATIONS,
    build_numpy_layouts,
    pack_codes,
    practical_memory_for_layouts,
    theoretical_memory,
    theoretical_work,
    unpack_codes,
)
from bitemb.quantization import turboquant_encode


@pytest.fixture
def normalized_embs():
    rng = np.random.default_rng(42)
    embs = rng.normal(size=(32, 64)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def test_theoretical_memory_float32_768():
    mem = theoretical_memory(10, 768, "float32")
    assert mem.bits_per_vector == 768 * 32
    assert mem.total_bytes == 10 * 768 * 4
    assert mem.compression_ratio_vs_float768 == pytest.approx(1.0)


@pytest.mark.parametrize(("rep", "expected_ratio"), [("4bit", 8.0), ("2bit", 16.0), ("1bit", 32.0)])
def test_theoretical_memory_compression_full_dim(rep, expected_ratio):
    mem = theoretical_memory(10, 768, rep)
    assert mem.compression_ratio_vs_float768 == pytest.approx(expected_ratio)


@pytest.mark.parametrize("bits", [2, 4])
def test_pack_unpack_roundtrip(bits):
    rng = np.random.default_rng(7)
    dim = 65
    max_code = (1 << bits) - 1
    codes = rng.integers(0, max_code + 1, size=(5, dim), dtype=np.uint8)
    packed = pack_codes(codes, bits=bits)
    unpacked = unpack_codes(packed, bits=bits, dim=dim)
    np.testing.assert_array_equal(unpacked, codes)


@pytest.mark.parametrize("bits", [2, 4])
def test_packed_turboquant_index_is_smaller_than_uint8(normalized_embs, bits):
    idx = turboquant_encode(normalized_embs, bits=bits)
    packed = pack_codes(idx.codes, bits=bits)
    assert packed.nbytes <= idx.codes.nbytes
    if bits == 2:
        assert packed.nbytes == normalized_embs.shape[0] * ((normalized_embs.shape[1] * 2 + 7) // 8)
    else:
        assert packed.nbytes == normalized_embs.shape[0] * ((normalized_embs.shape[1] * 4 + 7) // 8)


def test_build_numpy_layouts_memory_records(normalized_embs):
    layouts = build_numpy_layouts(normalized_embs)
    rows = practical_memory_for_layouts(layouts)
    by_rep = {row.representation: row for row in rows}
    assert set(by_rep) == set(REPRESENTATIONS)
    assert by_rep["float32"].total_bytes == normalized_embs.nbytes
    assert by_rep["1bit"].total_bytes < by_rep["float32"].total_bytes
    assert by_rep["2bit"].total_bytes < by_rep["float32"].total_bytes
    assert by_rep["4bit"].total_bytes < by_rep["float32"].total_bytes


@pytest.mark.parametrize("rep", REPRESENTATIONS)
def test_theoretical_work_has_nonnegative_counts(rep):
    work = theoretical_work(64, rep)
    assert work.packed_bytes_per_vector > 0
    assert work.float_ops_per_distance >= 0
    assert work.integer_ops_per_distance >= 0
    assert work.unpack_ops_per_distance >= 0
    assert work.popcount_ops_per_distance >= 0

