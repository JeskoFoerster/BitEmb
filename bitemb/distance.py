"""Phase 2: Pairwise distance correlation and distortion analysis.

Measures how well quantized representations preserve pairwise distances
from the float reference space. For each pair, four distance measures are
computed (float cosine, TurboQuant 4-bit, TurboQuant 2-bit, binary Hamming)
and compared via Pearson r, Spearman ρ, MAE, and RMSE.

The analysis spans a 2D experimental matrix:
  - Bit depth: float32, 4-bit, 2-bit, 1-bit (binary)
  - Dimensionality: 64, 128, 256, 384, 768

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
    TurboQuantIndex,
    binarize,
    turboquant_encode,
)

N_PAIRS = 10_000


@dataclass
class DistortionResult:
    """Distance preservation metrics for one (bit_depth, dim) combination."""

    bit_depth: int  # 1, 2, or 4
    dim: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    mae: float
    rmse: float
    n_pairs: int


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
    pairs = set()
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


def _turboquant_distance_pairs(
    index: TurboQuantIndex, pairs: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Compute L2² distance between dequantized vectors for given pairs.

    Uses symmetric comparison (both vectors from the quantized index).
    """
    recon = index.dequantize()
    a = recon[pairs[:, 0]]
    b = recon[pairs[:, 1]]
    return ((a - b) ** 2).sum(axis=1)


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


def compute_distance_distortion(
    embeddings: NDArray[np.float32],
    dim: int = 768,
    pca_reducer: PCAReducer | None = None,
    n_pairs: int = N_PAIRS,
    seed: int = SEED,
) -> list[DistortionResult]:
    """Run the full Phase 2 distortion analysis for one PCA dimension.

    Args:
        embeddings: L2-normalized float32 corpus embeddings (n, 768).
        dim: Target dimensionality (768 = no reduction).
        pca_reducer: Fitted PCAReducer if dim < 768, else None.
        n_pairs: Number of random pairs to sample.
        seed: Random seed for pair sampling.

    Returns:
        List of 3 DistortionResult (one per bit depth: 4, 2, 1).
    """
    # Apply PCA reduction if needed
    if pca_reducer is not None and dim < 768:
        embs = pca_reducer.transform(embeddings)
    else:
        embs = embeddings

    # Sample pairs
    pairs = _sample_pairs(embs.shape[0], n_pairs, seed)

    # Ground truth: float cosine distance
    d_float = _cosine_distance_pairs(embs, pairs)

    results = []

    # TurboQuant 4-bit
    tq4 = turboquant_encode(embs, bits=4)
    d_tq4 = _turboquant_distance_pairs(tq4, pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_tq4)
    results.append(DistortionResult(
        bit_depth=4, dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    # TurboQuant 2-bit
    tq2 = turboquant_encode(embs, bits=2)
    d_tq2 = _turboquant_distance_pairs(tq2, pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_tq2)
    results.append(DistortionResult(
        bit_depth=2, dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    # Binary (1-bit)
    packed = binarize(embs)
    d_bin = _hamming_distance_pairs(packed, pairs)
    pr, pp, sr, sp, mae, rmse = _compute_metrics(d_float, d_bin)
    results.append(DistortionResult(
        bit_depth=1, dim=dim, pearson_r=pr, pearson_p=pp,
        spearman_rho=sr, spearman_p=sp, mae=mae, rmse=rmse, n_pairs=n_pairs,
    ))

    return results
