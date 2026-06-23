"""Native FAISS benchmarks comparing float IP, binary Hamming, and Python baseline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    import faiss
except ImportError as e:
    raise SystemExit("faiss-cpu is required: pip install faiss-cpu") from e

import matplotlib.pyplot as plt

SIZES = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
DIM = 768
N_QUERIES = 100
K = 10

_POPCOUNT_TABLE = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def generate_data(
    n: int, dim: int, seed: int = 42
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Generate normalized float32 vectors and their packed-bit representation."""
    rng = np.random.default_rng(seed)
    floats = rng.normal(size=(n, dim)).astype(np.float32)
    floats /= np.linalg.norm(floats, axis=1, keepdims=True)
    bits = np.packbits(floats > 0, axis=1)
    return floats, bits


def bench_faiss_float(
    floats: NDArray[np.float32], queries: NDArray[np.float32], k: int
) -> float:
    """Benchmark FAISS IndexFlatIP. Returns ms/query."""
    index = faiss.IndexFlatIP(floats.shape[1])
    index.add(floats)
    start = time.perf_counter()
    index.search(queries, k)
    elapsed = time.perf_counter() - start
    return (elapsed * 1000) / len(queries)


def bench_faiss_hamming(
    bits: NDArray[np.uint8], query_bits: NDArray[np.uint8], k: int
) -> float:
    """Benchmark FAISS IndexBinaryFlat. Returns ms/query."""
    nbits = bits.shape[1] * 8
    index = faiss.IndexBinaryFlat(nbits)
    index.add(bits)
    start = time.perf_counter()
    index.search(query_bits, k)
    elapsed = time.perf_counter() - start
    return (elapsed * 1000) / len(query_bits)


def bench_python_hamming(
    bits: NDArray[np.uint8], query_bits: NDArray[np.uint8]
) -> float:
    """Python LUT-based Hamming baseline. Returns ms/query."""
    start = time.perf_counter()
    for q in query_bits:
        dists = _POPCOUNT_TABLE[np.bitwise_xor(bits, q)].sum(axis=1)
        np.argpartition(dists, K)[:K]
    elapsed = time.perf_counter() - start
    return (elapsed * 1000) / len(query_bits)


def run_benchmark() -> list[dict]:
    """Run all benchmarks across corpus sizes."""
    results = []
    for n in SIZES:
        print(f"  n={n:>7,} ...", end=" ", flush=True)
        floats, bits = generate_data(n, DIM)
        q_floats = floats[:N_QUERIES]
        q_bits = bits[:N_QUERIES]

        lat_float = bench_faiss_float(floats, q_floats, K)
        lat_hamming = bench_faiss_hamming(bits, q_bits, K)
        lat_python = bench_python_hamming(bits, q_bits)

        row = {
            "n": n,
            "faiss_float_ms": round(lat_float, 4),
            "faiss_hamming_ms": round(lat_hamming, 4),
            "python_hamming_ms": round(lat_python, 4),
            "speedup_faiss_ham_vs_python": round(lat_python / lat_hamming, 2),
            "speedup_faiss_ham_vs_float": round(lat_float / lat_hamming, 2),
        }
        results.append(row)
        print(f"done (FAISS ham {lat_hamming:.3f} ms/q)")
    return results


def plot_results(results: list[dict]) -> Path:
    """Create 2-panel log-log plot and save PNG."""
    ns = [r["n"] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Latency
    ax1.loglog(ns, [r["faiss_float_ms"] for r in results], "o-", label="FAISS Float IP")
    ax1.loglog(ns, [r["faiss_hamming_ms"] for r in results], "s-", label="FAISS Hamming")
    ax1.loglog(ns, [r["python_hamming_ms"] for r in results], "^-", label="Python Hamming")
    ax1.set_xlabel("Corpus size")
    ax1.set_ylabel("Latency (ms/query)")
    ax1.set_title("Search Latency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Speedup
    ax2.semilogx(ns, [r["speedup_faiss_ham_vs_python"] for r in results], "s-", label="FAISS Ham / Python")
    ax2.semilogx(ns, [r["speedup_faiss_ham_vs_float"] for r in results], "o-", label="FAISS Ham / FAISS Float")
    ax2.set_xlabel("Corpus size")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Speedup Factors")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path("output/faiss_benchmark.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    """Run benchmarks, save JSON, plot, print table."""
    print("Running FAISS benchmarks...")
    results = run_benchmark()

    # Save JSON
    out_json = Path("output/faiss_benchmark.json")
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    # Plot
    png_path = plot_results(results)

    # Print table
    print(f"\n{'n':>10} {'FAISS Float':>12} {'FAISS Ham':>12} {'Python Ham':>12} {'Spd(H/Py)':>10} {'Spd(H/F)':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['n']:>10,} {r['faiss_float_ms']:>12.4f} {r['faiss_hamming_ms']:>12.4f} "
            f"{r['python_hamming_ms']:>12.4f} {r['speedup_faiss_ham_vs_python']:>10.1f}x "
            f"{r['speedup_faiss_ham_vs_float']:>10.1f}x"
        )
    print(f"\nResults saved: {out_json}")
    print(f"Plot saved:    {png_path}")


if __name__ == "__main__":
    main()
