"""Phase 5 efficiency analysis helpers.

This module separates theoretical efficiency estimates from practical NumPy
layouts. Runtime measurements use vectorized NumPy operations so the project
does not require a custom C extension or compiler setup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Literal, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from bitemb.quantization import TurboQuantIndex, binarize, turboquant_encode

Representation = Literal["float32", "4bit", "2bit", "1bit"]
Operation = Literal["memory", "pairwise_distance", "top_k", "index_build"]
Implementation = Literal[
    "theoretical",
    "numpy_vectorized",
    "dequantized_baseline",
]

REPRESENTATIONS: tuple[Representation, ...] = ("float32", "4bit", "2bit", "1bit")


@dataclass(frozen=True)
class TheoreticalMemory:
    """Theoretical packed storage for one representation."""

    representation: Representation
    n_vectors: int
    dim: int
    bit_depth: int
    bits_per_vector: int
    bytes_per_vector: float
    total_bytes: int
    compression_ratio_vs_float768: float


@dataclass(frozen=True)
class TheoreticalWork:
    """Coarse operation model for one vector distance comparison."""

    representation: Representation
    dim: int
    packed_bytes_per_vector: int
    float_ops_per_distance: int
    integer_ops_per_distance: int
    unpack_ops_per_distance: int
    popcount_ops_per_distance: int
    notes: str


@dataclass(frozen=True)
class PracticalMemory:
    """Measured payload size for a concrete index layout."""

    representation: Representation
    n_vectors: int
    dim: int
    implementation: Implementation
    index_bytes: int
    metadata_bytes: int
    total_bytes: int
    bytes_per_vector: float
    compression_ratio_vs_float768: float
    notes: str


@dataclass(frozen=True)
class RuntimeMeasurement:
    """Runtime result for one operation and representation."""

    representation: Representation
    operation: Operation
    implementation: Implementation
    n_vectors: int
    dim: int
    n_pairs: int | None
    k: int | None
    warmup_runs: int
    measurement_runs: int
    times_ms: list[float]
    median_ms: float | None
    min_ms: float | None
    mean_ms: float | None
    std_ms: float | None
    throughput_per_second: float | None
    status: str
    notes: str


@dataclass
class PackedTurboQuantIndex:
    """TurboQuant codes stored in an actual packed byte layout."""

    packed_codes: NDArray[np.uint8]
    bits: Literal[2, 4]
    dim: int
    n: int
    col_min: NDArray[np.float64]
    col_max: NDArray[np.float64]

    @property
    def metadata_bytes(self) -> int:
        return int(self.col_min.nbytes + self.col_max.nbytes)

    @property
    def index_bytes(self) -> int:
        return int(self.packed_codes.nbytes + self.metadata_bytes)


@dataclass
class NumpyLayouts:
    """Concrete NumPy layouts used by Phase 5 memory/runtime analysis."""

    float32: NDArray[np.float32]
    binary1: NDArray[np.uint8]
    tq2: PackedTurboQuantIndex
    tq4: PackedTurboQuantIndex


def bit_depth_for_representation(representation: Representation) -> int:
    if representation == "float32":
        return 32
    if representation == "4bit":
        return 4
    if representation == "2bit":
        return 2
    if representation == "1bit":
        return 1
    raise ValueError(f"Unknown representation: {representation}")


def theoretical_memory(
    n_vectors: int,
    dim: int,
    representation: Representation,
    *,
    baseline_dim: int = 768,
) -> TheoreticalMemory:
    """Compute ideal packed storage for a representation."""
    bit_depth = bit_depth_for_representation(representation)
    bits_per_vector = dim * bit_depth
    total_bits = n_vectors * bits_per_vector
    total_bytes = int((total_bits + 7) // 8)
    bytes_per_vector = total_bytes / n_vectors if n_vectors else 0.0
    baseline_bytes = n_vectors * baseline_dim * 4
    compression = baseline_bytes / total_bytes if total_bytes else 0.0
    return TheoreticalMemory(
        representation=representation,
        n_vectors=n_vectors,
        dim=dim,
        bit_depth=bit_depth,
        bits_per_vector=bits_per_vector,
        bytes_per_vector=bytes_per_vector,
        total_bytes=total_bytes,
        compression_ratio_vs_float768=float(compression),
    )


def theoretical_work(dim: int, representation: Representation) -> TheoreticalWork:
    """Return a simple per-distance work model for explanation and comparison."""
    if representation == "float32":
        return TheoreticalWork(
            representation=representation,
            dim=dim,
            packed_bytes_per_vector=dim * 4,
            float_ops_per_distance=2 * dim,
            integer_ops_per_distance=0,
            unpack_ops_per_distance=0,
            popcount_ops_per_distance=0,
            notes="dot product: dim multiplies plus dim adds",
        )
    if representation == "1bit":
        words = (dim + 63) // 64
        return TheoreticalWork(
            representation=representation,
            dim=dim,
            packed_bytes_per_vector=(dim + 7) // 8,
            float_ops_per_distance=0,
            integer_ops_per_distance=words,
            unpack_ops_per_distance=0,
            popcount_ops_per_distance=words,
            notes="packed XOR plus popcount per 64-bit word",
        )
    if representation == "2bit":
        return TheoreticalWork(
            representation=representation,
            dim=dim,
            packed_bytes_per_vector=(dim * 2 + 7) // 8,
            float_ops_per_distance=dim,
            integer_ops_per_distance=dim,
            unpack_ops_per_distance=dim,
            popcount_ops_per_distance=0,
            notes="packed code unpack plus per-dimension quantized distance",
        )
    if representation == "4bit":
        return TheoreticalWork(
            representation=representation,
            dim=dim,
            packed_bytes_per_vector=(dim * 4 + 7) // 8,
            float_ops_per_distance=dim,
            integer_ops_per_distance=dim,
            unpack_ops_per_distance=dim,
            popcount_ops_per_distance=0,
            notes="packed nibble unpack plus per-dimension quantized distance",
        )
    raise ValueError(f"Unknown representation: {representation}")


def pack_codes(codes: NDArray[np.uint8], bits: Literal[2, 4]) -> NDArray[np.uint8]:
    """Pack uint8 quantization codes into 2-bit or 4-bit byte layout."""
    if bits not in (2, 4):
        raise ValueError("Only 2-bit and 4-bit packing are supported")
    if codes.ndim != 2:
        raise ValueError("codes must have shape (n, dim)")
    max_code = (1 << bits) - 1
    if codes.size and int(codes.max()) > max_code:
        raise ValueError(f"codes contain values above {max_code}")

    n, dim = codes.shape
    per_byte = 8 // bits
    out_dim = (dim + per_byte - 1) // per_byte
    packed = np.zeros((n, out_dim), dtype=np.uint8)
    clean = codes.astype(np.uint8, copy=False)
    for offset in range(per_byte):
        src = clean[:, offset::per_byte]
        shifted = np.left_shift(src, offset * bits).astype(np.uint8, copy=False)
        target = packed[:, : src.shape[1]]
        packed[:, : src.shape[1]] = np.bitwise_or(target, shifted)
    return packed


def unpack_codes(packed: NDArray[np.uint8], bits: Literal[2, 4], dim: int) -> NDArray[np.uint8]:
    """Unpack 2-bit or 4-bit codes. Used for tests and validation only."""
    if bits not in (2, 4):
        raise ValueError("Only 2-bit and 4-bit unpacking are supported")
    if packed.ndim != 2:
        raise ValueError("packed must have shape (n, packed_dim)")
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    out = np.empty((packed.shape[0], dim), dtype=np.uint8)
    for offset in range(per_byte):
        values = (packed >> (offset * bits)) & mask
        out[:, offset::per_byte] = values[:, : out[:, offset::per_byte].shape[1]]
    return out


def pack_turboquant_index(index: TurboQuantIndex) -> PackedTurboQuantIndex:
    """Convert the current uint8 TurboQuant codes into a packed byte layout."""
    if index.bits not in (2, 4):
        raise ValueError("Packed TurboQuant supports only 2-bit and 4-bit indexes")
    bits: Literal[2, 4] = 2 if index.bits == 2 else 4
    packed = pack_codes(index.codes, bits=bits)
    return PackedTurboQuantIndex(
        packed_codes=packed,
        bits=bits,
        dim=index.dim,
        n=index.n,
        col_min=np.ascontiguousarray(index.col_min, dtype=np.float64),
        col_max=np.ascontiguousarray(index.col_max, dtype=np.float64),
    )


def build_numpy_layouts(embeddings: NDArray[np.float32]) -> NumpyLayouts:
    """Build all practical layouts used by Phase 5 for one embedding matrix."""
    embs = np.ascontiguousarray(embeddings, dtype=np.float32)
    tq2 = pack_turboquant_index(turboquant_encode(embs, bits=2))
    tq4 = pack_turboquant_index(turboquant_encode(embs, bits=4))
    return NumpyLayouts(
        float32=embs,
        binary1=np.ascontiguousarray(binarize(embs), dtype=np.uint8),
        tq2=tq2,
        tq4=tq4,
    )


def practical_memory_for_layouts(layouts: NumpyLayouts) -> list[PracticalMemory]:
    """Measure practical payload/index sizes for all NumPy layouts."""
    n, dim = layouts.float32.shape
    baseline_bytes = n * 768 * 4

    def row(
        rep: Representation,
        index_bytes: int,
        metadata_bytes: int,
        notes: str,
    ) -> PracticalMemory:
        total = int(index_bytes + metadata_bytes)
        return PracticalMemory(
            representation=rep,
            n_vectors=n,
            dim=dim,
            implementation="numpy_vectorized",
            index_bytes=int(index_bytes),
            metadata_bytes=int(metadata_bytes),
            total_bytes=total,
            bytes_per_vector=total / n if n else 0.0,
            compression_ratio_vs_float768=baseline_bytes / total if total else 0.0,
            notes=notes,
        )

    return [
        row("float32", int(layouts.float32.nbytes), 0, "contiguous NumPy float32 matrix"),
        row("1bit", int(layouts.binary1.nbytes), 0, "NumPy packbits-compatible binary codes"),
        row(
            "2bit",
            int(layouts.tq2.packed_codes.nbytes),
            layouts.tq2.metadata_bytes,
            "packed 2-bit codes plus min/max metadata",
        ),
        row(
            "4bit",
            int(layouts.tq4.packed_codes.nbytes),
            layouts.tq4.metadata_bytes,
            "packed 4-bit codes plus min/max metadata",
        ),
    ]



def cosine_distance_pairs_numpy(
    embeddings: NDArray[np.float32], pairs: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Vectorized cosine distance for sampled pairs."""
    embs = np.ascontiguousarray(embeddings, dtype=np.float32)
    p = np.ascontiguousarray(pairs, dtype=np.int64)
    dots = np.einsum("ij,ij->i", embs[p[:, 0]], embs[p[:, 1]], optimize=True)
    return (1.0 - dots).astype(np.float64, copy=False)


_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_distance_pairs_numpy(
    packed: NDArray[np.uint8], pairs: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Vectorized Hamming distance for sampled packed binary pairs."""
    codes = np.ascontiguousarray(packed, dtype=np.uint8)
    p = np.ascontiguousarray(pairs, dtype=np.int64)
    xor = np.bitwise_xor(codes[p[:, 0]], codes[p[:, 1]])
    return _POPCOUNT_TABLE[xor].sum(axis=1, dtype=np.int64).astype(np.float64)


def turboquant_distance_pairs_numpy(
    index: PackedTurboQuantIndex,
    pairs: NDArray[np.int64],
    *,
    chunk_size: int = 8192,
) -> NDArray[np.float64]:
    """Vectorized TurboQuant squared distance for sampled pairs."""
    codes = unpack_codes(index.packed_codes, bits=index.bits, dim=index.dim)
    scale = (index.col_max - index.col_min) / float((1 << index.bits) - 1)
    p = np.ascontiguousarray(pairs, dtype=np.int64)
    out = np.empty(p.shape[0], dtype=np.float64)
    for start in range(0, p.shape[0], chunk_size):
        end = min(start + chunk_size, p.shape[0])
        diff_codes = codes[p[start:end, 0]].astype(np.float64) - codes[p[start:end, 1]]
        diff = diff_codes * scale
        out[start:end] = np.einsum("ij,ij->i", diff, diff, optimize=True)
    return out


def knn_cosine_numpy(embeddings: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    """Brute-force top-k cosine search using NumPy matrix multiplication."""
    embs = np.ascontiguousarray(embeddings, dtype=np.float32)
    distances = 1.0 - (embs @ embs.T)
    np.fill_diagonal(distances, np.inf)
    idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    row = np.arange(embs.shape[0])[:, None]
    order = np.argsort(distances[row, idx], axis=1)
    return idx[row, order].astype(np.int64, copy=False)


def knn_hamming_numpy(
    packed: NDArray[np.uint8], k: int, *, batch_size: int = 256
) -> NDArray[np.int64]:
    """Brute-force top-k Hamming search using batched NumPy broadcasting."""
    codes = np.ascontiguousarray(packed, dtype=np.uint8)
    n = codes.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xor = np.bitwise_xor(codes[start:end, None, :], codes[None, :, :])
        distances = _POPCOUNT_TABLE[xor].sum(axis=2, dtype=np.int64)
        distances[np.arange(end - start), np.arange(start, end)] = np.iinfo(np.int64).max
        idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        row = np.arange(end - start)[:, None]
        order = np.argsort(distances[row, idx], axis=1)
        out[start:end] = idx[row, order]
    return out


def knn_turboquant_numpy(
    index: PackedTurboQuantIndex, k: int, *, batch_size: int = 256
) -> NDArray[np.int64]:
    """Brute-force top-k TurboQuant search using dequantized NumPy blocks."""
    codes = unpack_codes(index.packed_codes, bits=index.bits, dim=index.dim).astype(np.float64)
    scale = (index.col_max - index.col_min) / float((1 << index.bits) - 1)
    values = codes * scale + index.col_min
    norms = np.einsum("ij,ij->i", values, values, optimize=True)
    n = values.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dots = values[start:end] @ values.T
        distances = norms[start:end, None] + norms[None, :] - 2.0 * dots
        distances[np.arange(end - start), np.arange(start, end)] = np.inf
        idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        row = np.arange(end - start)[:, None]
        order = np.argsort(distances[row, idx], axis=1)
        out[start:end] = idx[row, order]
    return out.astype(np.int64, copy=False)

def measure_runtime(
    fn: Callable[[], object],
    *,
    representation: Representation,
    operation: Operation,
    implementation: Implementation,
    n_vectors: int,
    dim: int,
    n_pairs: int | None = None,
    k: int | None = None,
    warmup_runs: int = 3,
    measurement_runs: int = 10,
    throughput_units: int | None = None,
    notes: str = "",
) -> RuntimeMeasurement:
    """Measure a callable with warmup and repeated timings."""
    for _ in range(warmup_runs):
        fn()

    times_ms: list[float] = []
    for _ in range(measurement_runs):
        start = perf_counter()
        fn()
        times_ms.append((perf_counter() - start) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    median_ms = float(np.median(arr)) if arr.size else None
    throughput = None
    if throughput_units is not None and median_ms and median_ms > 0:
        throughput = float(throughput_units / (median_ms / 1000.0))

    return RuntimeMeasurement(
        representation=representation,
        operation=operation,
        implementation=implementation,
        n_vectors=n_vectors,
        dim=dim,
        n_pairs=n_pairs,
        k=k,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        times_ms=times_ms,
        median_ms=median_ms,
        min_ms=float(arr.min()) if arr.size else None,
        mean_ms=float(arr.mean()) if arr.size else None,
        std_ms=float(arr.std(ddof=0)) if arr.size else None,
        throughput_per_second=throughput,
        status="ok",
        notes=notes,
    )


def unavailable_runtime(
    *,
    representation: Representation,
    operation: Operation,
    implementation: Implementation,
    n_vectors: int,
    dim: int,
    n_pairs: int | None = None,
    k: int | None = None,
    notes: str,
) -> RuntimeMeasurement:
    """Create a structured skipped runtime record."""
    return RuntimeMeasurement(
        representation=representation,
        operation=operation,
        implementation=implementation,
        n_vectors=n_vectors,
        dim=dim,
        n_pairs=n_pairs,
        k=k,
        warmup_runs=0,
        measurement_runs=0,
        times_ms=[],
        median_ms=None,
        min_ms=None,
        mean_ms=None,
        std_ms=None,
        throughput_per_second=None,
        status="unavailable",
        notes=notes,
    )


def dataclasses_to_dicts(items: Sequence[object]) -> list[dict[str, Any]]:
    """Convert dataclass result objects into JSON-serializable dictionaries."""
    return [asdict(cast(Any, item)) for item in items]

