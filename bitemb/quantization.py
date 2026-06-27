"""Quantization methods: naive binary, TurboQuant 2-bit and 4-bit.

TurboQuant (Zandieh et al., 2025):
  1. Multiply embeddings by a random orthogonal matrix (decorrelates dims).
  2. Apply uniform scalar quantization per dimension.

The rotation is data-independent and preserves all distances/angles exactly.
It redistributes variance uniformly so that a fixed-grid quantizer works
efficiently across all dimensions.

Additionally provides PCA-based dimension reduction (Section 3.3.3 of Methodik):
  - Fit PCA on corpus embeddings (unsupervised, task-agnostic).
  - Project both corpus and queries to d ∈ {64, 128, 256, 384, 768}.
  - Quantization is then applied on the reduced vectors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import special_ortho_group
from sklearn.decomposition import PCA

from bitemb.config import MODEL_DIM, SEED


def _random_rotation_matrix(dim: int = MODEL_DIM, seed: int = SEED) -> NDArray[np.float64]:
    """Generate a fixed random orthogonal matrix (SO(dim))."""
    return special_ortho_group.rvs(dim, random_state=seed).astype(np.float64)


# Module-level cached rotation matrices (one per dimension, computed once).
_ROTATIONS: dict[int, NDArray[np.float64]] = {}


def _get_rotation(dim: int = MODEL_DIM) -> NDArray[np.float64]:
    if dim not in _ROTATIONS:
        _ROTATIONS[dim] = _random_rotation_matrix(dim)
    return _ROTATIONS[dim]


# ---------- Binary quantization ----------


def binarize(embeddings: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Naive sign-threshold binarization → packed uint8.

    Each dimension becomes 1 if positive, 0 otherwise.
    Compression: 32x vs float32.
    """
    return np.packbits((embeddings > 0).astype(np.uint8), axis=1)


_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def hamming_distance(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> NDArray[np.int64]:
    """Hamming distance between packed bit vectors (a: n×d', b: 1×d' or m×d').

    Returns integer distance array.
    """
    xor = np.bitwise_xor(a, b)
    return _POPCOUNT_TABLE[xor].sum(axis=-1)


# ---------- TurboQuant ----------


def _rotate(embeddings: NDArray[np.float32]) -> NDArray[np.float64]:
    """Apply the cached random orthogonal rotation."""
    rot = _get_rotation(embeddings.shape[1])
    return embeddings.astype(np.float64) @ rot.T


def _uniform_scalar_quantize(
    values: NDArray[np.float64], bits: int
) -> NDArray[np.uint8]:
    """Uniform scalar quantization per-dimension to `bits` bit levels.

    Maps each dimension's range [min, max] to {0, ..., 2^bits - 1}.
    Returns uint8 codes (sufficient for up to 8 bits).
    """
    levels = (1 << bits) - 1  # e.g. 3 for 2-bit, 15 for 4-bit
    col_min = values.min(axis=0, keepdims=True)
    col_max = values.max(axis=0, keepdims=True)
    span = col_max - col_min
    span[span == 0] = 1.0  # avoid division by zero for constant dims
    normalized = (values - col_min) / span  # [0, 1]
    codes = np.round(normalized * levels).astype(np.uint8)
    return codes


class TurboQuantIndex:
    """Stores TurboQuant-compressed embeddings with metadata for distance computation.

    Attributes:
        codes: uint8 quantization codes (n, dim)
        bits: bit depth (2 or 4)
        col_min: per-dimension minimum of rotated embeddings
        col_max: per-dimension maximum of rotated embeddings
    """

    def __init__(
        self,
        codes: NDArray[np.uint8],
        bits: int,
        col_min: NDArray[np.float64],
        col_max: NDArray[np.float64],
    ) -> None:
        self.codes = codes
        self.bits = bits
        self.col_min = col_min
        self.col_max = col_max
        self._levels = (1 << bits) - 1

    def dequantize(self) -> NDArray[np.float64]:
        """Reconstruct approximate rotated vectors from codes."""
        span = self.col_max - self.col_min
        return self.codes.astype(np.float64) / self._levels * span + self.col_min

    @property
    def n(self) -> int:
        return self.codes.shape[0]

    @property
    def dim(self) -> int:
        return self.codes.shape[1]

    def memory_bytes(self) -> int:
        """Effective storage: codes packed at `bits` per value."""
        return int(np.ceil(self.n * self.dim * self.bits / 8))


def turboquant_encode(
    embeddings: NDArray[np.float32], bits: int = 2
) -> TurboQuantIndex:
    """Encode float embeddings with TurboQuant.

    Args:
        embeddings: L2-normalized float32 embeddings (n, dim).
        bits: Quantization depth (2 or 4).

    Returns:
        TurboQuantIndex with quantized codes and reconstruction params.
    """
    if bits not in (2, 4):
        raise ValueError("TurboQuant supports 2-bit or 4-bit quantization")
    rotated = _rotate(embeddings)
    col_min = rotated.min(axis=0)
    col_max = rotated.max(axis=0)
    codes = _uniform_scalar_quantize(rotated, bits)
    return TurboQuantIndex(codes, bits, col_min, col_max)


def turboquant_distance(
    index: TurboQuantIndex, query_embeddings: NDArray[np.float32]
) -> NDArray[np.float64]:
    """Compute asymmetric L2² distance: exact rotated query vs quantized corpus.

    Returns shape (n_queries, n_corpus).
    """
    query_rotated = _rotate(query_embeddings)
    corpus_approx = index.dequantize()
    # Efficient batch distance: ||q - c||² for all pairs
    # = ||q||² + ||c||² - 2·q·cᵀ
    q_sq = (query_rotated**2).sum(axis=1, keepdims=True)
    c_sq = (corpus_approx**2).sum(axis=1)
    dots = query_rotated @ corpus_approx.T
    return q_sq + c_sq - 2 * dots


# ---------- PCA Dimension Reduction ----------


class PCAReducer:
    """PCA-based dimension reduction fitted on corpus embeddings.

    Fits an unsupervised, task-agnostic PCA on the corpus float space.
    The same projection is then applied to queries, ensuring no information
    leakage from relevance labels into the reduction.

    Attributes:
        n_components: Target dimensionality after reduction.
        explained_variance_ratio_: Per-component explained variance (after fit).
    """

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self._pca = PCA(n_components=n_components, random_state=SEED)

    def fit(self, corpus_embs: NDArray[np.float32]) -> "PCAReducer":
        """Fit PCA on corpus embeddings (unsupervised).

        Args:
            corpus_embs: L2-normalized float32 corpus embeddings (n, 768).
        """
        self._pca.fit(corpus_embs)
        return self

    def transform(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Project embeddings to reduced space and re-normalize.

        Args:
            embeddings: Float32 embeddings (n, 768).

        Returns:
            L2-normalized float32 embeddings (n, n_components).
        """
        reduced = self._pca.transform(embeddings).astype(np.float32)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return reduced / norms

    @property
    def explained_variance_ratio_(self) -> NDArray[np.float64]:
        """Explained variance ratio per component (available after fit)."""
        return self._pca.explained_variance_ratio_

    @property
    def cumulative_variance(self) -> NDArray[np.float64]:
        """Cumulative explained variance (available after fit)."""
        return np.cumsum(self._pca.explained_variance_ratio_)
