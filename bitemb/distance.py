"""Phase 2: Pairwise distance correlation and distortion analysis.

Measures how well quantized representations preserve pairwise distances
from the float reference space. For each pair, distances are computed under
all 10 quantization representations and compared via Pearson r, Spearman ρ,
MAE, and RMSE.

The analysis spans a 2D experimental matrix:
  - Representation: float32, 16bit, naive_8bit, tq_8bit, naive_4bit, tq_4bit,
                    naive_2bit, tq_2bit, naive_1bit, tq_1bit
  - Dimensionality: 64, 128, 256, 384, 512, 768, 1024

Reference: chapMethodik.tex, Section 5 (Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr, spearmanr

from bitemb.config import SEED
from bitemb.quantization import (
    PCAReducer,
    QuantizedIndex,
    binarize,
    binarize_rotated,
    quantize_encode,
)

N_PAIRS = 10_000

# Canonical representation order (matches Phase 4/5)
REPRESENTATIONS = [
    "float32",
    "16bit",
    "naive_8bit",
    "tq_8bit",
    "naive_4bit",
    "tq_4bit",
    "naive_2bit",
    "tq_2bit",
    "naive_1bit",
    "tq_1bit",
]


@dataclass
class DistortionResult:
    """Distance preservation metrics for one (representation, dim) combination."""

    representation: str
    dim: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    mae: float
    rmse: float
    n_pairs: int

    # Backward compatibility property
    @property
    def bit_depth(self) -> int:
        """Backward compatibility: derive bit_depth from representation."""
        _map = {
            "16bit": 16,
            "naive_8bit": 8, "tq_8bit": 8,
            "naive_4bit": 4, "tq_4bit": 4,
            "naive_2bit": 2, "tq_2bit": 2,
            "naive_1bit": 1, "tq_1bit": 1,
        }
        return _map.get(self.representation, 32)


def _normalize_to_unit(d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Min-max normalize distances to [0, 1]."""
    d_min, d_max = d.min(), d.max()
    if d_max - d_min < 1e-12:
        return np.zeros_like(d)
    return (d - d_min) / (d_max - d_min)


def _sample_pairs(n_corpus: int, n_pairs: int = N_PAIRS, seed: int = SEED) -> NDArray[np.int64]:
    """Draw n_pairs uniformly random index pairs (i, j) with i < j.

    Returns array of shape (n_pairs, 2).

    Raises:
        ValueError: If n_corpus is too small to yield n_pairs unique pairs.
    """
    max_pairs = n_corpus * (n_corpus - 1) // 2
    if n_pairs > max_pairs:
        raise ValueError(
            f"Cannot draw {n_pairs} unique pairs from {n_corpus} documents "
            f"(max possible: {max_pairs})"
        )
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n_pairs:
        batch_size = (n_pairs - len(pairs)) * 2
        i = rng.integers(0, n_corpus, size=batch_size)
        j = rng.integers(0, n_corpus, size=batch_size)
        for a, b in zip(i, j):
            if a != b:
                pairs.add((min(a, b), max(a, b)))
            if len(pairs) >= n_pairs:
                break
    return np.array(sorted(pairs)[:n_pairs], dtype=np.int64)


def _cosine_distance_pairs(
    embs: NDArray[np.float32], pairs: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Compute cosine distance (1 - cos_sim) for given pairs.

    Assumes L2-normalized embeddings → cos_sim = dot product.
    """
    a = embs[pairs[:, 0]]
    b = embs[pairs[:, 1]]
    cos_sim = (a * b).sum(axis=1)
    return (1.0 - cos_sim).astype(np.float64)


def _hamming_distance_pairs(
    packed: NDArray[np.uint8], pairs: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Compute Hamming distance for given pairs of packed binary vectors."""
    a = packed[pairs[:, 0]]
    b = packed[pairs[:, 1]]
    xor = np.bitwise_xor(a, b)
    # Popcount via lookup table
    popcount_table = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
    return popcount_table[xor].sum(axis=1).astype(np.float64)


def _quantized_distance_pairs(
    index: QuantizedIndex, pairs: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Compute L2² distance between dequantized vectors for given pairs.

    Uses symmetric comparison (both vectors from the quantized index).
    Works for both naive and TurboQuant (rotated) quantization.
    """
    recon = index.dequantize()
    a = recon[pairs[:, 0]]
    b = recon[pairs[:, 1]]
    return ((a - b) ** 2).sum(axis=1)


# Backward compatibility alias
_turboquant_distance_pairs = _quantized_distance_pairs


def _compute_metrics(
    d_float: NDArray[np.float64], d_quant: NDArray[np.float64]
) -> tuple[float, float, float, float, float, float]:
    """Compute Pearson, Spearman, MAE, RMSE on [0,1]-normalized distances.

    Returns: (pearson_r, pearson_p, spearman_rho, spearman_p, mae, rmse)
    """
    d_float_norm = _normalize_to_unit(d_float)
    d_quant_norm = _normalize_to_unit(d_quant)

    pr, pp = pearsonr(d_float_norm, d_quant_norm)
    sr, sp = spearmanr(d_float_norm, d_quant_norm)

    errors = np.abs(d_float_norm - d_quant_norm)
    mae = float(errors.mean())
    rmse = float(np.sqrt((errors**2).mean()))

    return float(pr), float(pp), float(sr), float(sp), mae, rmse


@dataclass
class RawDistances:
    """Raw normalized distance arrays for visualization (scatter/histogram)."""

    dim: int
    d_float32: NDArray[np.float64]  # (n_pairs,) normalized [0,1]
    d_16bit: NDArray[np.float64]
    d_naive_8bit: NDArray[np.float64]
    d_tq_8bit: NDArray[np.float64]
    d_naive_4bit: NDArray[np.float64]
    d_tq_4bit: NDArray[np.float64]
    d_naive_2bit: NDArray[np.float64]
    d_tq_2bit: NDArray[np.float64]
    d_naive_1bit: NDArray[np.float64]
    d_tq_1bit: NDArray[np.float64]

    # Backward compatibility properties
    @property
    def d_float(self) -> NDArray[np.float64]:
        return self.d_float32

    @property
    def d_8bit(self) -> NDArray[np.float64]:
        return self.d_tq_8bit

    @property
    def d_4bit(self) -> NDArray[np.float64]:
        return self.d_tq_4bit

    @property
    def d_2bit(self) -> NDArray[np.float64]:
        return self.d_tq_2bit

    @property
    def d_1bit(self) -> NDArray[np.float64]:
        return self.d_naive_1bit


def compute_raw_distances(
    embeddings: NDArray[np.float32],
    dim: int = 1024,
    pca_reducer: PCAReducer | None = None,
    n_pairs: int = N_PAIRS,
    seed: int = SEED,
) -> RawDistances:
    """Compute normalized pairwise distances for scatter/histogram plots.

    Returns the [0,1]-normalized distance vectors for float and all quantized
    representations, allowing direct visual comparison.
    """
    if pca_reducer is not None and dim < 1024:
        embs = pca_reducer.transform(embeddings)
    else:
        embs = embeddings

    pairs = _sample_pairs(embs.shape[0], n_pairs, seed)

    # Float32 reference
    d_float32 = _cosine_distance_pairs(embs, pairs)

    # Float16
    d_16bit = _cosine_distance_pairs(embs.astype(np.float16), pairs)

    # Naive 8-bit (no rotation)
    d_naive_8bit = _quantized_distance_pairs(
        quantize_encode(embs, bits=8, is_rotated=False), pairs
    )

    # TurboQuant 8-bit (rotated)
    d_tq_8bit = _quantized_distance_pairs(
        quantize_encode(embs, bits=8, is_rotated=True), pairs
    )

    # Naive 4-bit (no rotation)
    d_naive_4bit = _quantized_distance_pairs(
        quantize_encode(embs, bits=4, is_rotated=False), pairs
    )

    # TurboQuant 4-bit (rotated)
    d_tq_4bit = _quantized_distance_pairs(
        quantize_encode(embs, bits=4, is_rotated=True), pairs
    )

    # Naive 2-bit (no rotation)
    d_naive_2bit = _quantized_distance_pairs(
        quantize_encode(embs, bits=2, is_rotated=False), pairs
    )

    # TurboQuant 2-bit (rotated)
    d_tq_2bit = _quantized_distance_pairs(
        quantize_encode(embs, bits=2, is_rotated=True), pairs
    )

    # Naive 1-bit (sign binarization)
    d_naive_1bit = _hamming_distance_pairs(binarize(embs), pairs)

    # TurboQuant 1-bit (rotated binarization)
    d_tq_1bit = _hamming_distance_pairs(binarize_rotated(embs), pairs)

    return RawDistances(
        dim=dim,
        d_float32=_normalize_to_unit(d_float32),
        d_16bit=_normalize_to_unit(d_16bit),
        d_naive_8bit=_normalize_to_unit(d_naive_8bit),
        d_tq_8bit=_normalize_to_unit(d_tq_8bit),
        d_naive_4bit=_normalize_to_unit(d_naive_4bit),
        d_tq_4bit=_normalize_to_unit(d_tq_4bit),
        d_naive_2bit=_normalize_to_unit(d_naive_2bit),
        d_tq_2bit=_normalize_to_unit(d_tq_2bit),
        d_naive_1bit=_normalize_to_unit(d_naive_1bit),
        d_tq_1bit=_normalize_to_unit(d_tq_1bit),
    )


def compute_distance_distortion(
    embeddings: NDArray[np.float32],
    dim: int = 1024,
    pca_reducer: PCAReducer | None = None,
    n_pairs: int = N_PAIRS,
    seed: int = SEED,
) -> list[DistortionResult]:
    """Run the full Phase 2 distortion analysis for one PCA dimension.

    Args:
        embeddings: L2-normalized float32 corpus embeddings (n, 1024).
        dim: Target dimensionality (1024 = no reduction).
        pca_reducer: Fitted PCAReducer if dim < 1024, else None.
        n_pairs: Number of random pairs to sample.
        seed: Random seed for pair sampling.

    Returns:
        List of 9 DistortionResult (one per non-float32 representation).
    """
    # Apply PCA reduction if needed
    if pca_reducer is not None and dim < 1024:
        embs = pca_reducer.transform(embeddings)
    else:
        embs = embeddings

    # Sample pairs
    pairs = _sample_pairs(embs.shape[0], n_pairs, seed)

    # Ground truth: float32 cosine distance
    d_float = _cosine_distance_pairs(embs, pairs)

    results = []

    # Float16 (16-bit)
    d_16 = _cosine_distance_pairs(embs.astype(np.float16), pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_16)
    results.append(DistortionResult(
        representation="16bit", dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    # Naive 8-bit (no rotation)
    naive8 = quantize_encode(embs, bits=8, is_rotated=False)
    d_naive8 = _quantized_distance_pairs(naive8, pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_naive8)
    results.append(DistortionResult(
        representation="naive_8bit", dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    # TurboQuant 8-bit (rotated)
    tq8 = quantize_encode(embs, bits=8, is_rotated=True)
    d_tq8 = _quantized_distance_pairs(tq8, pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_tq8)
    results.append(DistortionResult(
        representation="tq_8bit", dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    # Naive 4-bit (no rotation)
    naive4 = quantize_encode(embs, bits=4, is_rotated=False)
    d_naive4 = _quantized_distance_pairs(naive4, pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_naive4)
    results.append(DistortionResult(
        representation="naive_4bit", dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    # TurboQuant 4-bit (rotated)
    tq4 = quantize_encode(embs, bits=4, is_rotated=True)
    d_tq4 = _quantized_distance_pairs(tq4, pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_tq4)
    results.append(DistortionResult(
        representation="tq_4bit", dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    # Naive 2-bit (no rotation)
    naive2 = quantize_encode(embs, bits=2, is_rotated=False)
    d_naive2 = _quantized_distance_pairs(naive2, pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_naive2)
    results.append(DistortionResult(
        representation="naive_2bit", dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    # TurboQuant 2-bit (rotated)
    tq2 = quantize_encode(embs, bits=2, is_rotated=True)
    d_tq2 = _quantized_distance_pairs(tq2, pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_tq2)
    results.append(DistortionResult(
        representation="tq_2bit", dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    # Naive 1-bit (sign binarization)
    packed_naive = binarize(embs)
    d_naive1 = _hamming_distance_pairs(packed_naive, pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_naive1)
    results.append(DistortionResult(
        representation="naive_1bit", dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    # TurboQuant 1-bit (rotated binarization)
    packed_tq = binarize_rotated(embs)
    d_tq1 = _hamming_distance_pairs(packed_tq, pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_tq1)
    results.append(DistortionResult(
        representation="tq_1bit", dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    return results
