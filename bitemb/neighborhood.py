"""Phase 3: Neighborhood preservation analysis.

Measures whether local structure (nearest neighbors) is preserved under
quantization. Two complementary metrics:

1. Neighborhood Overlap: fraction of k-NN shared between float and
   quantized space (symmetric measure).
2. Trustworthiness (Venna & Kaski, 2006): penalizes false neighbors
   introduced in the quantized space, weighted by their rank displacement.

Both metrics are computed over the full 2D experimental matrix
(bit_depth × PCA_dim) for k ∈ {10, 50, 100}.

Reference: chapMethodik.tex, Section "Phase 3: Nachbarschaftserhaltung".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bitemb.config import SEED
from bitemb.quantization import (
    PCAReducer,
    binarize,
    hamming_distance,
    turboquant_encode,
)

K_VALUES = (10, 50, 100)


@dataclass
class NeighborhoodResult:
    """Neighborhood preservation metrics for one (bit_depth, dim, k) combo."""

    bit_depth: int
    dim: int
    k: int
    overlap: float  # mean neighborhood overlap ∈ [0, 1]
    trustworthiness: float  # T(k) ∈ [0, 1]
    n_docs: int
    random_baseline: float  # E[overlap] = k / N


def _knn_cosine(embs: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    """Compute exact k-NN indices via cosine similarity (brute-force).

    Args:
        embs: L2-normalized embeddings (n, d). Cosine sim = dot product.
        k: Number of neighbors (excluding self).

    Returns:
        Indices array of shape (n, k) with the k nearest neighbor indices.
    """
    n = embs.shape[0]
    # Compute full similarity matrix in batches to manage memory
    batch_size = min(512, n)
    knn = np.empty((n, k), dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Similarity of batch against all docs
        sims = embs[start:end] @ embs.T  # (batch, n)
        # Exclude self-similarity by setting to -inf
        for i in range(end - start):
            sims[i, start + i] = -np.inf
        # Argpartition for top-k (faster than full sort)
        top_k_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
        # Sort the top-k by similarity (descending)
        for i in range(end - start):
            order = np.argsort(sims[i, top_k_idx[i]])[::-1]
            knn[start + i] = top_k_idx[i][order]

    return knn


def _knn_hamming(packed: NDArray[np.uint8], k: int) -> NDArray[np.int64]:
    """Compute exact k-NN indices via Hamming distance (brute-force).

    Args:
        packed: Packed binary vectors (n, ceil(d/8)).
        k: Number of neighbors (excluding self).

    Returns:
        Indices array of shape (n, k).
    """
    n = packed.shape[0]
    batch_size = min(512, n)
    knn = np.empty((n, k), dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        for i in range(start, end):
            dists = hamming_distance(packed, packed[i:i + 1])
            dists[i] = np.iinfo(np.int64).max  # exclude self
            top_k_idx = np.argpartition(dists, k)[:k]
            order = np.argsort(dists[top_k_idx])
            knn[i] = top_k_idx[order]

    return knn


def _knn_turboquant(
    embs: NDArray[np.float32], bits: int, k: int
) -> NDArray[np.int64]:
    """Compute exact k-NN indices in TurboQuant space (L2 on dequantized).

    Args:
        embs: L2-normalized embeddings (n, d) to quantize.
        bits: Quantization depth (2 or 4).
        k: Number of neighbors (excluding self).

    Returns:
        Indices array of shape (n, k).
    """
    index = turboquant_encode(embs, bits=bits)
    recon = index.dequantize().astype(np.float32)
    # L2 distance on reconstructed vectors; use dot-product trick
    # ||a-b||² = ||a||² + ||b||² - 2a·b
    norms_sq = (recon**2).sum(axis=1)

    n = recon.shape[0]
    batch_size = min(512, n)
    knn = np.empty((n, k), dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dots = recon[start:end] @ recon.T  # (batch, n)
        dists = norms_sq[start:end, None] + norms_sq[None, :] - 2 * dots
        # Exclude self
        for i in range(end - start):
            dists[i, start + i] = np.inf
        top_k_idx = np.argpartition(dists, k, axis=1)[:, :k]
        for i in range(end - start):
            order = np.argsort(dists[i, top_k_idx[i]])
            knn[start + i] = top_k_idx[i][order]

    return knn


def _compute_overlap(
    knn_float: NDArray[np.int64], knn_quant: NDArray[np.int64], k: int
) -> float:
    """Mean neighborhood overlap across all documents.

    Overlap(i) = |N_k^float(i) ∩ N_k^quant(i)| / k
    """
    n = knn_float.shape[0]
    overlaps = np.empty(n)
    for i in range(n):
        float_set = set(knn_float[i, :k])
        quant_set = set(knn_quant[i, :k])
        overlaps[i] = len(float_set & quant_set) / k
    return float(overlaps.mean())


def _cosine_ranks_row(
    embs: NDArray[np.float32], i: int
) -> NDArray[np.int32]:
    """Compute 1-based cosine similarity ranks for a single query point.

    Returns:
        ranks: (n,) array where ranks[j] = rank of j w.r.t. point i.
               rank[i] = 0 (self).
    """
    n = embs.shape[0]
    sims = embs[i] @ embs.T  # (n,)
    sims[i] = -np.inf
    order = np.argsort(-sims)
    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = np.arange(1, n + 1, dtype=np.int32)
    ranks[i] = 0
    return ranks


def _full_cosine_ranks(embs: NDArray[np.float32]) -> NDArray[np.int32]:
    """Compute full ranking matrix. Only for small corpora (n ≤ ~10k).

    Returns:
        ranks: (n, n) where ranks[i, j] = 1-based rank of j w.r.t. i.
    """
    n = embs.shape[0]
    batch_size = min(512, n)
    ranks = np.empty((n, n), dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = embs[start:end] @ embs.T
        for i in range(end - start):
            sims[i, start + i] = -np.inf
        order = np.argsort(-sims, axis=1)
        for i in range(end - start):
            ranks[start + i][order[i]] = np.arange(1, n + 1, dtype=np.int32)
            ranks[start + i, start + i] = 0

    return ranks


# Threshold for full rank matrix (n² × 4 bytes must fit in RAM)
_FULL_RANK_THRESHOLD = 8000  # 8000² × 4 = ~256 MB


def _compute_trustworthiness(
    knn_float: NDArray[np.int64],
    knn_quant: NDArray[np.int64],
    float_ranks: NDArray[np.int32] | None,
    embs: NDArray[np.float32],
    k: int,
    n_docs: int,
) -> float:
    """Trustworthiness T(k) per Venna & Kaski (2006).

    T(k) = 1 - (2 / (Nk(2N - 3k - 1))) * Σ_i Σ_{j ∈ U_k(i)} (r(i,j) - k)

    where U_k(i) = points in quant k-NN of i but NOT in float k-NN of i,
    and r(i,j) = rank of j w.r.t. i in the float space (1-based).

    Args:
        knn_float: (n, max_k) float space k-NN indices.
        knn_quant: (n, max_k) quantized space k-NN indices.
        float_ranks: (n, n) full ranking matrix, or None for on-demand.
        embs: Embeddings (needed if float_ranks is None).
        k: Number of neighbors for evaluation.
        n_docs: Total number of documents.
    """
    n = n_docs
    normalization = n * k * (2 * n - 3 * k - 1)
    if normalization == 0:
        return 1.0

    penalty = 0.0
    for i in range(n):
        float_neighbors_k = set(knn_float[i, :k])
        quant_neighbors_k = set(knn_quant[i, :k])
        false_neighbors = quant_neighbors_k - float_neighbors_k
        if not false_neighbors:
            continue
        # Get ranks for point i
        if float_ranks is not None:
            ranks_i = float_ranks[i]
        else:
            ranks_i = _cosine_ranks_row(embs, i)
        for j in false_neighbors:
            penalty += ranks_i[j] - k

    t = 1.0 - (2.0 / normalization) * penalty
    return float(t)


def compute_neighborhood_preservation(
    embeddings: NDArray[np.float32],
    dim: int = 768,
    pca_reducer: PCAReducer | None = None,
    k_values: tuple[int, ...] = K_VALUES,
    seed: int = SEED,
) -> list[NeighborhoodResult]:
    """Run Phase 3 neighborhood analysis for one PCA dimension.

    Computes full pairwise k-NN in float space and each quantized space,
    then measures overlap and trustworthiness.

    Args:
        embeddings: L2-normalized float32 corpus embeddings (n, 768).
        dim: Target dimensionality (768 = no reduction).
        pca_reducer: Fitted PCAReducer if dim < 768, else None.
        k_values: Tuple of k values to evaluate.
        seed: Random seed (unused here, kept for API consistency).

    Returns:
        List of NeighborhoodResult (3 bit_depths × len(k_values)).
    """
    # Apply PCA reduction if needed
    if pca_reducer is not None and dim < 768:
        embs = pca_reducer.transform(embeddings)
    else:
        embs = embeddings

    n_docs = embs.shape[0]
    max_k = max(k_values)

    # Compute k-NN in float space (ground truth) — need max_k for overlap
    knn_float = _knn_cosine(embs, max_k)

    # Precompute full rank matrix if corpus is small enough
    float_ranks: NDArray[np.int32] | None = None
    if n_docs <= _FULL_RANK_THRESHOLD:
        float_ranks = _full_cosine_ranks(embs)

    # Compute k-NN in each quantized space
    knn_by_bits: dict[int, NDArray[np.int64]] = {
        4: _knn_turboquant(embs, bits=4, k=max_k),
        2: _knn_turboquant(embs, bits=2, k=max_k),
        1: _knn_hamming(binarize(embs), k=max_k),
    }

    results = []
    for bit_depth in (4, 2, 1):
        knn_quant = knn_by_bits[bit_depth]
        for k in k_values:
            overlap = _compute_overlap(knn_float, knn_quant, k)
            trust = _compute_trustworthiness(
                knn_float, knn_quant, float_ranks, embs, k, n_docs,
            )
            results.append(NeighborhoodResult(
                bit_depth=bit_depth,
                dim=dim,
                k=k,
                overlap=overlap,
                trustworthiness=trust,
                n_docs=n_docs,
                random_baseline=k / n_docs,
            ))

    return results
