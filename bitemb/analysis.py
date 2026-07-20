"""Phase 1: Characterization of the float embedding space.

Analyzes the geometric and statistical properties of the unquantized float
space to formulate expectations about quantization behavior:

1. Norm distribution — Are vectors on a hypersphere? (relevant for binarization)
2. Per-dimension statistics — Skewness/kurtosis predict quantization error.
3. Intrinsic dimensionality — How much redundancy exists? (TwoNN + PCA)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA

from bitemb.config import SEED

# ---------- 1. Norm Distribution ----------


@dataclass
class NormStats:
    """L2-norm distribution statistics of the embedding matrix."""

    mean: float
    std: float
    min: float
    max: float
    cv: float  # coefficient of variation (std/mean)

    def is_near_unit_sphere(self, threshold: float = 0.01) -> bool:
        """True if vectors lie approximately on a unit hypersphere."""
        return self.cv < threshold


def compute_norm_distribution(embeddings: NDArray[np.float32]) -> NormStats:
    """Compute L2-norm statistics across all vectors.

    If the engine produces normalized embeddings, norms should be ≈ 1.0
    with near-zero variance. This confirms that binarization discards
    only sign information, not magnitude information.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    mean = float(norms.mean())
    return NormStats(
        mean=mean,
        std=float(norms.std()),
        min=float(norms.min()),
        max=float(norms.max()),
        cv=float(norms.std() / mean) if mean > 0 else 0.0,
    )


# ---------- 2. Per-Dimension Statistics ----------


@dataclass
class DimensionStats:
    """Per-dimension statistics across the corpus (shape: (1024,) each)."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    skewness: NDArray[np.float64]
    kurtosis: NDArray[np.float64]


def compute_dimension_stats(embeddings: NDArray[np.float32]) -> DimensionStats:
    """Compute mean, std, skewness, and kurtosis for each dimension.

    - Skewness: High |skew| → sign threshold at 0 is suboptimal for binarization.
    - Kurtosis: High kurtosis → outliers cause large quantization error at few levels.
    """
    return DimensionStats(
        mean=embeddings.mean(axis=0).astype(np.float64),
        std=embeddings.std(axis=0).astype(np.float64),
        skewness=skew(embeddings, axis=0).astype(np.float64),
        kurtosis=kurtosis(embeddings, axis=0, fisher=True).astype(np.float64),
    )


# ---------- 3. Intrinsic Dimensionality ----------


@dataclass
class IntrinsicDimensionality:
    """Intrinsic dimensionality estimates from two complementary methods."""

    twonn: float  # TwoNN estimate (local manifold dimension)
    pca_95: int  # Number of PCA components for 95% explained variance
    pca_cumulative_variance: NDArray[np.float64]  # Full cumulative variance curve


def _estimate_twonn(embeddings: NDArray[np.float32], seed: int = SEED) -> float:
    """TwoNN intrinsic dimensionality estimator (Facco et al., 2017).

    Uses the ratio of distances to the 2nd and 1st nearest neighbor.
    The maximum likelihood estimator is: d = n / Σ log(μ_i)
    where μ_i = r2_i / r1_i (ratio of 2nd to 1st neighbor distance).
    """
    n = embeddings.shape[0]
    # Subsample for efficiency if corpus is large (>10k)
    rng = np.random.default_rng(seed)
    if n > 10_000:
        idx = rng.choice(n, size=10_000, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings

    tree = KDTree(sample)
    # Query 3 nearest neighbors (self + 2 neighbors)
    dists, _ = tree.query(sample, k=3)
    r1 = dists[:, 1]  # distance to 1st neighbor (skip self at index 0)
    r2 = dists[:, 2]  # distance to 2nd neighbor

    # Remove zero-distance pairs (duplicates)
    valid = r1 > 0
    r1 = r1[valid]
    r2 = r2[valid]

    mu = r2 / r1
    # MLE: d = n / Σ log(μ_i)
    log_mu_sum = np.log(mu).sum()
    if log_mu_sum <= 0:
        return float(embeddings.shape[1])  # fallback to nominal dim
    return float(len(mu) / log_mu_sum)


def _estimate_pca_dimension(
    embeddings: NDArray[np.float32], threshold: float = 0.95
) -> tuple[int, NDArray[np.float64]]:
    """Number of PCA components explaining `threshold` of total variance."""
    pca = PCA(random_state=SEED)
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, threshold)) + 1
    return n_components, cumvar.astype(np.float64)


def compute_intrinsic_dimensionality(
    embeddings: NDArray[np.float32],
) -> IntrinsicDimensionality:
    """Estimate intrinsic dimensionality via TwoNN and PCA.

    TwoNN captures local manifold structure.
    PCA captures global linear redundancy.
    Agreement → robust estimate. Divergence → nonlinear structure.
    """
    twonn = _estimate_twonn(embeddings)
    pca_95, cumvar = _estimate_pca_dimension(embeddings, threshold=0.95)
    return IntrinsicDimensionality(
        twonn=twonn,
        pca_95=pca_95,
        pca_cumulative_variance=cumvar,
    )
