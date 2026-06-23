"""Visualisierung & Analyse für Float- und Bit-Embeddings."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from bitemb.engine import MATRYOSHKA_DIMS, EmbeddingEngine

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_pca_2d(
    float_embs: NDArray[np.float32], title: str, labels: list[str] | None = None
) -> Path:
    """PCA projection to 2D with scatter plot.

    Args:
        float_embs: Float embeddings (n, dim).
        title: Plot title.
        labels: Optional text labels for annotation.

    Returns:
        Path to saved PNG file.
    """
    coords = PCA(n_components=2).fit_transform(float_embs)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7)
    if labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=7)
    ax.set_title(title)
    path = OUTPUT_DIR / f"{title.replace(' ', '_').lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tsne_2d(
    float_embs: NDArray[np.float32], bit_embs_unpacked: NDArray, title: str
) -> Path:
    """t-SNE visualization comparing float and bit embeddings.

    Args:
        float_embs: Float embeddings (n, dim).
        bit_embs_unpacked: Unpacked binary embeddings (n, dim).
        title: Plot title.

    Returns:
        Path to saved PNG file.
    """
    combined = np.vstack([float_embs, bit_embs_unpacked.astype(np.float32)])
    n = len(float_embs)
    coords = TSNE(n_components=2, random_state=42).fit_transform(combined)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:n, 0], coords[:n, 1], c="blue", label="float", alpha=0.6)
    ax.scatter(coords[n:, 0], coords[n:, 1], c="red", label="bit", alpha=0.6)
    ax.legend()
    ax.set_title(title)
    path = OUTPUT_DIR / f"{title.replace(' ', '_').lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def analyze_vector_space(float_embs: NDArray[np.float32]) -> dict:
    """Analyze properties of the float embedding space.

    Args:
        float_embs: Normalized float embeddings (n, dim).

    Returns:
        Dict with num_vectors, dimension, pairwise sim stats,
        component variance, effective_dim_95pct, isotropy_score.
    """
    n, dim = float_embs.shape
    sim = float_embs @ float_embs.T
    triu_idx = np.triu_indices(n, k=1)
    pairwise = sim[triu_idx]

    pca = PCA().fit(float_embs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    effective_dim = int(np.searchsorted(cumvar, 0.95)) + 1

    variances = pca.explained_variance_
    isotropy = float(variances.min() / variances.max()) if variances.max() > 0 else 0.0

    return {
        "num_vectors": n,
        "dimension": dim,
        "mean_pairwise_sim": float(pairwise.mean()),
        "std_pairwise_sim": float(pairwise.std()),
        "component_variance": pca.explained_variance_ratio_.tolist(),
        "effective_dim_95pct": effective_dim,
        "isotropy_score": isotropy,
    }


def analyze_information_loss(
    float_embs: NDArray[np.float32], bit_embs_packed: NDArray[np.uint8], dim: int
) -> dict:
    """Analyze information loss from float to binary quantization.

    Args:
        float_embs: Normalized float embeddings (n, dim).
        bit_embs_packed: Packed binary embeddings.
        dim: Original embedding dimensionality.

    Returns:
        Dict with cosine preservation stats, rank correlation, compression ratio.
    """
    n = len(float_embs)
    unpacked = np.unpackbits(bit_embs_packed, axis=1)[:, :dim].astype(np.float32)
    signed = unpacked * 2 - 1
    norms = np.linalg.norm(signed, axis=1, keepdims=True)
    signed = signed / norms

    # Cosine preservation
    float_sim = float_embs @ float_embs.T
    bit_sim = signed @ signed.T
    triu_idx = np.triu_indices(n, k=1)
    float_pairs = float_sim[triu_idx]
    bit_pairs = bit_sim[triu_idx]
    cosine_diff = np.abs(float_pairs - bit_pairs)

    # Rank correlation (cap at n=200)
    cap = min(n, 200)
    float_sub = float_embs[:cap]
    sim_sub = float_sub @ float_sub.T
    signed_sub = signed[:cap]
    bit_sim_sub = signed_sub @ signed_sub.T
    triu_sub = np.triu_indices(cap, k=1)
    float_ranks = np.argsort(sim_sub[triu_sub])
    bit_ranks = np.argsort(bit_sim_sub[triu_sub])
    rank_corr = float(np.corrcoef(float_ranks, bit_ranks)[0, 1])

    # Compression ratio: float32 * dim vs uint8 * ceil(dim/8)
    compression = (4 * dim) / bit_embs_packed.shape[1]

    return {
        "cosine_preservation_mean": float(1 - cosine_diff.mean()),
        "cosine_preservation_std": float(cosine_diff.std()),
        "cosine_preservation_min": float(1 - cosine_diff.max()),
        "rank_correlation": rank_corr,
        "compression_ratio": float(compression),
    }


def plot_information_loss_by_dim(engine: EmbeddingEngine, texts: list[str]) -> Path:
    """Plot information loss across Matryoshka dimensions.

    Args:
        engine: Initialized EmbeddingEngine.
        texts: Texts to embed.

    Returns:
        Path to saved PNG file.
    """
    means = []
    for dim in MATRYOSHKA_DIMS:
        float_embs = engine.embed_float(texts, dim=dim)
        bit_packed = engine.embed_bit_sign(texts, dim=dim)
        loss = analyze_information_loss(float_embs, bit_packed, dim)
        means.append(loss["cosine_preservation_mean"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(MATRYOSHKA_DIMS, means, marker="o")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Cosine Preservation (mean)")
    ax.set_title("Information Loss by Dimension")
    ax.invert_xaxis()
    path = OUTPUT_DIR / "information_loss_by_dim.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
