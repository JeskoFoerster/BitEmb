"""Performance benchmarks for float and binary embedding search."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


@dataclass
class BenchmarkResult:
    """Search benchmark metrics for one embedding mode."""

    n_docs: int
    dim: int
    mode: str
    index_time_ms: float
    search_time_ms: float
    memory_bytes: int
    throughput_qps: float


_POPCOUNT_TABLE = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def _hamming_distances(
    corpus_bits: NDArray[np.uint8], query_bits: NDArray[np.uint8]
) -> NDArray[np.uint64]:
    """Compute Hamming distances from packed corpus bits to one packed query."""
    query = np.asarray(query_bits, dtype=np.uint8)
    if query.ndim == 2:
        if query.shape[0] != 1:
            raise ValueError("query_bits must contain exactly one query")
        query = query[0]
    if query.ndim != 1:
        raise ValueError("query_bits must be a 1D vector or a single-row 2D array")
    if corpus_bits.ndim != 2:
        raise ValueError("corpus_bits must be a 2D packed-bit array")
    if corpus_bits.shape[1] != query.shape[0]:
        raise ValueError("corpus_bits and query_bits must have the same packed width")

    xor = np.bitwise_xor(corpus_bits, query)
    distances = _POPCOUNT_TABLE[xor].sum(axis=1, dtype=np.uint64)
    return cast(NDArray[np.uint64], distances)


def benchmark_search(
    float_embs: NDArray[np.float32],
    bit_embs: NDArray[np.uint8],
    dim: int,
    n_queries: int = 100,
) -> dict[str, Any]:
    """Benchmark float cosine search against packed-bit Hamming search."""
    if float_embs.ndim != 2:
        raise ValueError("float_embs must be a 2D array")
    if bit_embs.ndim != 2:
        raise ValueError("bit_embs must be a 2D array")
    if float_embs.shape[0] != bit_embs.shape[0]:
        raise ValueError("float_embs and bit_embs must contain the same number of documents")
    if dim <= 0:
        raise ValueError("dim must be greater than 0")
    if n_queries <= 0:
        raise ValueError("n_queries must be greater than 0")

    n_docs = float_embs.shape[0]
    query_count = min(n_queries, n_docs)
    float_queries = float_embs[:query_count]
    bit_queries = bit_embs[:query_count]

    float_start = time.perf_counter()
    for query in float_queries:
        scores = float_embs @ query
        np.argmax(scores)
    float_search_time_ms = (time.perf_counter() - float_start) * 1000

    bit_start = time.perf_counter()
    for query in bit_queries:
        distances = _hamming_distances(bit_embs, query)
        np.argmin(distances)
    bit_search_time_ms = (time.perf_counter() - bit_start) * 1000

    float_seconds = float_search_time_ms / 1000
    bit_seconds = bit_search_time_ms / 1000
    float_qps = query_count / float_seconds if float_seconds > 0 else float("inf")
    bit_qps = query_count / bit_seconds if bit_seconds > 0 else float("inf")

    float_result = BenchmarkResult(
        n_docs=n_docs,
        dim=dim,
        mode="float",
        index_time_ms=0.0,
        search_time_ms=float_search_time_ms,
        memory_bytes=int(float_embs.nbytes),
        throughput_qps=float_qps,
    )
    bit_result = BenchmarkResult(
        n_docs=n_docs,
        dim=dim,
        mode="bit",
        index_time_ms=0.0,
        search_time_ms=bit_search_time_ms,
        memory_bytes=int(bit_embs.nbytes),
        throughput_qps=bit_qps,
    )

    return {
        "float": float_result,
        "bit": bit_result,
        "speedup": float_search_time_ms / bit_search_time_ms
        if bit_search_time_ms > 0
        else float("inf"),
        "memory_ratio": float_embs.nbytes / bit_embs.nbytes
        if bit_embs.nbytes > 0
        else float("inf"),
    }


def benchmark_scaling(
    engine: object, sizes: list[int] | None = None, dim: int = 768
) -> list[dict[str, Any]]:
    """Benchmark synthetic normalized embeddings for multiple corpus sizes."""
    del engine

    if sizes is None:
        sizes = [100, 1_000, 10_000]
    if dim <= 0:
        raise ValueError("dim must be greater than 0")

    rng = np.random.default_rng(42)
    results: list[dict[str, Any]] = []

    for n_docs in sizes:
        if n_docs <= 0:
            raise ValueError("sizes must contain only positive integers")

        float_start = time.perf_counter()
        float_embs = rng.normal(size=(n_docs, dim)).astype(np.float32)
        norms = np.linalg.norm(float_embs, axis=1, keepdims=True)
        float_embs = float_embs / norms
        float_index_time_ms = (time.perf_counter() - float_start) * 1000

        bit_start = time.perf_counter()
        bit_embs = np.packbits(float_embs > 0, axis=1)
        bit_index_time_ms = (time.perf_counter() - bit_start) * 1000

        result = benchmark_search(float_embs, bit_embs, dim)
        result["float"].index_time_ms = float_index_time_ms
        result["bit"].index_time_ms = bit_index_time_ms
        results.append(result)

    return results


def format_benchmark_results(results: dict[str, Any] | list[dict[str, Any]]) -> str:
    """Format benchmark results as a plain-text table."""
    result_list = results if isinstance(results, list) else [results]
    rows: list[BenchmarkResult] = []
    summaries: list[str] = []

    for result in result_list:
        rows.extend([result["float"], result["bit"]])
        n_docs = result["float"].n_docs
        summaries.append(
            f"{n_docs} docs: speedup={result['speedup']:.2f}x, "
            f"memory_ratio={result['memory_ratio']:.2f}x"
        )

    header = (
        f"{'Docs':>8}  {'Dim':>5}  {'Mode':<6}  {'Index ms':>10}  "
        f"{'Search ms':>10}  {'Memory':>12}  {'QPS':>12}"
    )
    separator = "-" * len(header)
    lines = [header, separator]

    for row in rows:
        lines.append(
            f"{row.n_docs:>8}  {row.dim:>5}  {row.mode:<6}  "
            f"{row.index_time_ms:>10.2f}  {row.search_time_ms:>10.2f}  "
            f"{row.memory_bytes:>12}  {row.throughput_qps:>12.2f}"
        )

    lines.extend(["", *summaries])
    return "\n".join(lines)


def main() -> None:
    """Run the default synthetic scaling benchmark."""
    print(format_benchmark_results(benchmark_scaling(engine=None)))


if __name__ == "__main__":
    main()
