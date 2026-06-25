"""Tests for bitemb.benchmark."""

import numpy as np
import pytest

from bitemb.benchmark import BenchmarkResult, benchmark_search


@pytest.fixture()
def bench_result():
    rng = np.random.default_rng(0)
    dim = 768
    n = 50
    floats = rng.normal(size=(n, dim)).astype(np.float32)
    floats /= np.linalg.norm(floats, axis=1, keepdims=True)
    bits = np.packbits(floats > 0, axis=1)
    return benchmark_search(floats, bits, dim, n_queries=5)


class TestBenchmarkSearch:
    def test_returns_expected_keys(self, bench_result):
        assert set(bench_result.keys()) == {"float", "bit", "speedup", "memory_ratio"}

    def test_result_types(self, bench_result):
        assert isinstance(bench_result["float"], BenchmarkResult)
        assert isinstance(bench_result["bit"], BenchmarkResult)

    def test_memory_ratio(self, bench_result):
        assert bench_result["memory_ratio"] == pytest.approx(32.0, rel=1e-2)

    def test_positive_latency(self, bench_result):
        assert bench_result["float"].search_time_ms > 0
        assert bench_result["bit"].search_time_ms > 0
