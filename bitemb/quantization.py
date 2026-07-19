"""Quantization methods: naive and TurboQuant (rotated) versions for 1, 2, 4, and 8 bits.

Provides:
  - Naive and rotated binarization (1-bit)
  - Uniform scalar quantization with/without rotation (2-bit, 4-bit, 8-bit)
  - PCA-based dimension reduction
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


def _rotate(embeddings: NDArray[np.float32]) -> NDArray[np.float64]:
    """Apply the cached random orthogonal rotation."""
    rot = _get_rotation(embeddings.shape[1])
    return embeddings.astype(np.float64) @ rot.T


# ---------- 1-Bit Binarization (Naive & Rotated) ----------


def binarize(embeddings: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Naive sign-threshold binarization → packed uint8."""
    return np.packbits((embeddings > 0).astype(np.uint8), axis=1)


def binarize_rotated(embeddings: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Rotated sign-threshold binarization → packed uint8."""
    rotated = _rotate(embeddings).astype(np.float32)
    return binarize(rotated)


_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def hamming_distance(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> NDArray[np.int64]:
    """Hamming distance between packed bit vectors."""
    xor = np.bitwise_xor(a, b)
    return _POPCOUNT_TABLE[xor].sum(axis=-1)


# ---------- Multi-Bit Uniform Quantization (Naive & TurboQuant) ----------


def _uniform_scalar_quantize(
    values: NDArray[np.float64], bits: int
) -> NDArray[np.uint8]:
    """Uniform scalar quantization per-dimension to `bits` bit levels."""
    levels = (1 << bits) - 1
    col_min = values.min(axis=0, keepdims=True)
    col_max = values.max(axis=0, keepdims=True)
    span = col_max - col_min
    span[span == 0] = 1.0  # avoid division by zero for constant dims
    normalized = (values - col_min) / span  # [0, 1]
    codes = np.round(normalized * levels).astype(np.uint8)
    return codes


class QuantizedIndex:
    """Stores quantized embeddings (naive or TurboQuant) with reconstruction metadata."""

    def __init__(
        self,
        codes: NDArray[np.uint8],
        bits: int,
        col_min: NDArray[np.float64],
        col_max: NDArray[np.float64],
        is_rotated: bool = True,
    ) -> None:
        self.codes = codes
        self.bits = bits
        self.col_min = col_min
        self.col_max = col_max
        self._levels = (1 << bits) - 1
        self.is_rotated = is_rotated

    def dequantize(self) -> NDArray[np.float64]:
        """Reconstruct approximate rotated/unrotated vectors from codes."""
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


# Backwards compatibility alias
TurboQuantIndex = QuantizedIndex


def quantize_encode(
    embeddings: NDArray[np.float32], bits: int = 2, is_rotated: bool = True
) -> QuantizedIndex:
    """Encode float embeddings using naive or rotated uniform scalar quantization."""
    if bits not in (2, 4, 8):
        raise ValueError("Quantization supports 2-bit, 4-bit, or 8-bit")
        
    if is_rotated:
        processed = _rotate(embeddings)
    else:
        processed = embeddings.astype(np.float64)
        
    col_min = processed.min(axis=0)
    col_max = processed.max(axis=0)
    codes = _uniform_scalar_quantize(processed, bits)
    return QuantizedIndex(codes, bits, col_min, col_max, is_rotated=is_rotated)


# Backwards compatibility alias
def turboquant_encode(embeddings: NDArray[np.float32], bits: int = 2) -> TurboQuantIndex:
    return quantize_encode(embeddings, bits=bits, is_rotated=True)


def quantize_distance(
    index: QuantizedIndex, query_embeddings: NDArray[np.float32]
) -> NDArray[np.float64]:
    """Compute asymmetric L2² distance between query and quantized index."""
    if index.is_rotated:
        query_processed = _rotate(query_embeddings)
    else:
        query_processed = query_embeddings.astype(np.float64)
        
    corpus_approx = index.dequantize()
    q_sq = (query_processed**2).sum(axis=1, keepdims=True)
    c_sq = (corpus_approx**2).sum(axis=1)
    dots = query_processed @ corpus_approx.T
    return q_sq + c_sq - 2 * dots


# Backwards compatibility alias
def turboquant_distance(
    index: TurboQuantIndex, query_embeddings: NDArray[np.float32]
) -> NDArray[np.float64]:
    return quantize_distance(index, query_embeddings)


# ---------- PCA Dimension Reduction ----------


class PCAReducer:
    """PCA-based dimension reduction fitted on corpus embeddings."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self._pca = PCA(n_components=n_components, random_state=SEED)

    def fit(self, corpus_embs: NDArray[np.float32]) -> "PCAReducer":
        self._pca.fit(corpus_embs)
        return self

    def transform(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        reduced = self._pca.transform(embeddings).astype(np.float32)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return reduced / norms

    @property
    def explained_variance_ratio_(self) -> NDArray[np.float64]:
        return self._pca.explained_variance_ratio_

    @property
    def cumulative_variance(self) -> NDArray[np.float64]:
        return np.cumsum(self._pca.explained_variance_ratio_)
